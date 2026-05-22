"""Compatibility stubs for the engine transplant.

The reframer-engine swap deletes ~65 legacy service modules (face detection,
transcription, clip scoring, content classification, camera solvers, ...).
A number of *surviving* files still import individual symbols from those
modules — mostly lazily, inside functions or try/except blocks.

Rather than chase every call site, this module re-exports a minimal, benign
stand-in for each symbol the surviving code references. Imports that used to
read ``from backend.services.<deleted> import X`` are redirected here:

    from backend.services.compat_stubs import X

Every stub is intentionally inert: it returns empty/neutral data so the
degraded code path does not crash. The real work now flows through the
reframer engine (reframer_*.py) and its bridge (reframer_bridge.py).
"""

import logging
import threading

logger = logging.getLogger("clipai.compat_stubs")


# ═══════════════════════════════════════════════════════════════════════════
#  content_classifier / content_type_config / content_type_strings
# ═══════════════════════════════════════════════════════════════════════════

from enum import Enum


class ClipContentType(str, Enum):
    GENERIC = "generic"
    PODCAST = "podcast"
    VLOG = "vlog"
    EDUCATIONAL = "educational"
    DOCUMENTARY = "documentary"
    NARRATIVE = "narrative"
    MUSIC_VIDEO = "music_video"
    SPORTS = "sports"
    GAMEPLAY = "gameplay"
    ANIMATION = "animation"
    ANIMATION_DIALOGUE = "animation_dialogue"
    CINEMATIC_DIALOGUE = "cinematic_dialogue"


# content_type_config historically exposed a parallel ``ContentType`` enum.
ContentType = ClipContentType


class ContentProfile:
    """Minimal stand-in for code that inspects ``content_profile`` attributes."""

    def __init__(self, content_type="generic", **kwargs):
        self.content_type = content_type
        for k, v in kwargs.items():
            setattr(self, k, v)


def classify_clip(*args, **kwargs) -> ClipContentType:
    """Legacy classifier — the reframer Planner now routes content internally."""
    return ClipContentType.GENERIC


def classify_content(*args, **kwargs) -> ClipContentType:
    return ClipContentType.GENERIC


def get_vote_weights(*args, **kwargs) -> dict:
    return {}


def normalize_content_type(value="", *args, **kwargs) -> str:
    """content_type_strings.normalize_content_type — pass the value through."""
    return str(value or "generic")


def normalize_ui_content_type(value="", *args, **kwargs) -> str:
    return str(value or "generic")


def content_type_label(value="", *args, **kwargs) -> str:
    return str(value or "Generic").replace("_", " ").title()


def normalize_game_type(value="", *args, **kwargs) -> str:
    return str(value or "")


# ═══════════════════════════════════════════════════════════════════════════
#  clip_scoring
# ═══════════════════════════════════════════════════════════════════════════

_AXES = ("hook_score", "flow_score", "value_score", "trend_score")


def _as_dict(clip):
    """Return a mutable mapping view of a clip (dict or pydantic/obj)."""
    if isinstance(clip, dict):
        return clip
    return getattr(clip, "__dict__", {})


def fill_axes_from_legacy(clip_dict: dict) -> dict:
    """If the 4 decomposed axes are all zero, seed them from ``viral_score``."""
    d = _as_dict(clip_dict)
    vs = d.get("viral_score", 50) or 50
    if all(not d.get(k, 0) for k in _AXES):
        for k in _AXES:
            d[k] = vs
    return clip_dict


def composite_score(hook=0, flow=0, value=0, trend=0, content_type=None, **kwargs) -> int:
    """Genre-weighted composite — the stub uses an unweighted mean."""
    return int(round((hook + flow + value + trend) / 4.0))


def finalize_clip_scores(clips: list, content_type=None, **kwargs) -> list:
    """Backfill axes + composite ``viral_score`` for a list of clips."""
    for c in (clips or []):
        fill_axes_from_legacy(c)
        d = _as_dict(c)
        if not d.get("viral_score"):
            d["viral_score"] = composite_score(
                d.get("hook_score", 0), d.get("flow_score", 0),
                d.get("value_score", 0), d.get("trend_score", 0),
            )
    return clips


def deduplicate_overlapping_clips(clips, overlap_threshold: float = 0.5):
    """Drop lower-scored clips that overlap a kept clip beyond the threshold."""
    if not clips:
        return clips
    ordered = sorted(
        clips, key=lambda c: getattr(c, "viral_score", 0)
        if not isinstance(c, dict) else c.get("viral_score", 0),
        reverse=True,
    )

    def _span(c):
        if isinstance(c, dict):
            return c.get("start_time", 0), c.get("end_time", 0)
        return getattr(c, "start_time", 0), getattr(c, "end_time", 0)

    kept = []
    for clip in ordered:
        cs, ce = _span(clip)
        clash = False
        for k in kept:
            ks, ke = _span(k)
            overlap = max(0, min(ce, ke) - max(cs, ks))
            shorter = min(ce - cs, ke - ks) or 1
            if overlap / shorter > overlap_threshold:
                clash = True
                break
        if not clash:
            kept.append(clip)
    return kept


# ═══════════════════════════════════════════════════════════════════════════
#  transcription / whisper_worker / retranscribe
# ═══════════════════════════════════════════════════════════════════════════

_model = None
_model_lock = threading.Lock()
_last_detected_language = ""
_last_diarization_method = ""


def _detect_cuda_available():
    """Return (cuda_available, device_count, gpu_name, driver_version)."""
    try:
        import torch
        avail = torch.cuda.is_available()
        count = torch.cuda.device_count() if avail else 0
        name = torch.cuda.get_device_name(0) if count > 0 else ""
        return (avail, count, name, "")
    except Exception:
        return (False, 0, "", "")


def _get_gpu_vram_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def whisper_device_info() -> dict:
    avail, count, name, _ = _detect_cuda_available()
    return {
        "cuda_available": avail,
        "gpu_count": count,
        "gpu_name": name,
        "model_loaded": _model is not None,
    }


def is_whisper_model_cached(model_name: str = "small") -> bool:
    return True  # the engine loads Whisper on demand inside AudioIntelligence


def ensure_whisper_model_downloaded(model_name: str = "small"):
    return None


def preload_model(*args, **kwargs):
    return None


def reload_model(*args, **kwargs):
    return None


def reload_diarization(*args, **kwargs):
    return None


def _cleanup_old_model(*args, **kwargs):
    return None


def extract_word_timestamps(segments) -> list:
    """Flatten per-segment word lists into a single list."""
    words = []
    for seg in (segments or []):
        if isinstance(seg, dict):
            seg_words = seg.get("words") or []
        else:
            seg_words = getattr(seg, "words", None) or []
        words.extend(seg_words)
    return words


def assign_speakers_heuristic(segments, *args, **kwargs):
    return segments


def assign_speakers_with_face_data(segments, *args, **kwargs):
    return segments


def diarize_transcript_post(segments, *args, **kwargs):
    return segments


async def transcribe_audio(audio_path, *args, **kwargs):
    raise NotImplementedError(
        "Legacy transcribe_audio removed — transcription now runs inside "
        "reframer_audio.AudioIntelligence via the Perceiver."
    )


async def transcribe_audio_subprocess(audio_path, *args, **kwargs):
    raise NotImplementedError(
        "Legacy transcribe_audio_subprocess removed — see reframer_audio."
    )


def retranscribe_job(*args, **kwargs):
    raise NotImplementedError(
        "Standalone retranscription was retired with the engine swap; "
        "re-run analysis to regenerate the transcript."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  transcript_utils
# ═══════════════════════════════════════════════════════════════════════════

def analyze_transcript_energy(*args, **kwargs) -> dict:
    return {}


def correlate_scenes_with_transcript(*args, **kwargs) -> str:
    return ""


def derive_content_guidance(*args, **kwargs) -> str:
    return ""


def compute_filler_density(*args, **kwargs) -> float:
    return 0.0


def detect_filler_words(*args, **kwargs) -> list:
    return []


def detect_emphasis_words(*args, **kwargs) -> list:
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  hot_zone_scorer
# ═══════════════════════════════════════════════════════════════════════════

class HotZone:
    def __init__(self, **kwargs):
        self.start = kwargs.get("start", 0.0)
        self.end = kwargs.get("end", 0.0)
        self.composite_score = kwargs.get("composite_score", 0.0)
        for k, v in kwargs.items():
            setattr(self, k, v)


def score_hot_zones(*args, **kwargs) -> list:
    return []


def score_hot_zones_transcript_only(*args, **kwargs) -> list:
    return []


def format_hot_zones_for_prompt(*args, **kwargs) -> str:
    return ""


def get_coverage_gaps(*args, **kwargs) -> list:
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  face_detector / face_registry / active_speaker
# ═══════════════════════════════════════════════════════════════════════════

ANIME_MODE_DETECTED = False


class FaceInfo:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FrameFaces:
    def __init__(self, timestamp=0.0, faces=None):
        self.timestamp = timestamp
        self.faces = faces or []


def detect_faces_batch(frames, *args, **kwargs) -> list:
    return []


def detect_faces_dense(frames, *args, **kwargs) -> list:
    return []


def classify_gameplay_content(frames, *args, **kwargs):
    return None


class FaceSlot:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class FaceRegistry:
    def __init__(self):
        self.slots = []

    def to_dict(self) -> dict:
        return {"slots": []}


def build_face_registry(face_results=None, *args, **kwargs) -> FaceRegistry:
    return FaceRegistry()


def build_face_registry_with_embeddings(*args, **kwargs) -> FaceRegistry:
    return FaceRegistry()


class SpeakerEvent:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_active_slot_at_time(events, timestamp, *args, **kwargs):
    return None


def map_speakers_to_face_slots(segments, *args, **kwargs) -> dict:
    return {}


def build_vad_presence(*args, **kwargs) -> dict:
    return {}


# ═══════════════════════════════════════════════════════════════════════════
#  render_plan_builder / render_plan_keyframes
# ═══════════════════════════════════════════════════════════════════════════

def build_render_plan(segments=None, *args, **kwargs):
    """Legacy render-plan builder. The reframer bridge now produces RenderPlans
    directly; this returns a minimal valid empty plan for any stray caller."""
    from backend.services.render_plan import RenderPlan
    return RenderPlan(
        source_width=kwargs.get("source_width", 1920),
        source_height=kwargs.get("source_height", 1080),
        target_width=kwargs.get("target_width", 1080),
        target_height=kwargs.get("target_height", 1920),
        total_duration_sec=kwargs.get("total_duration_sec", 0.0),
        fps=kwargs.get("fps", 30.0),
        ops=[],
    )


def keyframes_to_render_plan(*args, **kwargs):
    return build_render_plan(**kwargs)


def render_plan_to_keyframes(*args, **kwargs) -> list:
    return []


def build_keyframes(*args, **kwargs) -> list:
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  gameplay_filters / game_layouts
# ═══════════════════════════════════════════════════════════════════════════

def is_gameplay_layout(layout_mode) -> bool:
    return layout_mode in ("gameplay", "stacked_gameplay")


def get_gameplay_crop_regions(*args, **kwargs):
    return None


def apply_gameplay_filters(*args, **kwargs):
    return None


def build_gameplay_blurfill_filter(*args, **kwargs) -> str:
    return ""


def build_gameplay_wide_zoom_filter(*args, **kwargs) -> str:
    return ""


def get_hud_layout(*args, **kwargs):
    return None


def get_action_center(*args, **kwargs):
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  clip_boundary_snapper / subject_track
# ═══════════════════════════════════════════════════════════════════════════

def snap_all_clips(clips, beats=None, *args, **kwargs):
    return clips


def build_subject_track(scenes=None, *args, **kwargs) -> list:
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  reframe_config
# ═══════════════════════════════════════════════════════════════════════════

class ReframeConfig:
    """Inert stand-in for the deleted reframe_config.ReframeConfig dataclass."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        # Any unknown knob reads as a neutral 0 so callers never KeyError.
        return 0


def get_default_config(*args, **kwargs) -> ReframeConfig:
    return ReframeConfig()


# ═══════════════════════════════════════════════════════════════════════════
#  camera_solver / shot_detector / required_regions / anime_shot_detector
#  / _autoflip_lp / object_detector / saliency_tracker
# ═══════════════════════════════════════════════════════════════════════════

class _PermissiveEnumMeta(type):
    """Metaclass so ``CameraMode.ANYTHING`` resolves to the member name string."""

    def __getattr__(cls, name):
        return name


class CameraMode(metaclass=_PermissiveEnumMeta):
    """Permissive stand-in for the deleted camera_solver.CameraMode enum."""


class ShotCamera:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Shot:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MultiRegionLPResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


CROP_ASPECT = 9 / 16
USE_ANIME_SHOT_DETECTOR = False


def detect_shots(*args, **kwargs) -> list:
    return []


def detect_anime_shots(*args, **kwargs) -> list:
    return []


def solve_all_shots(*args, **kwargs) -> list:
    return []


def get_params_for_content_type(*args, **kwargs) -> dict:
    return {}


def build_required_regions(*args, **kwargs) -> list:
    return []


def promote_preferred_to_required(*args, **kwargs) -> list:
    return []


def solve_multi_region_camera_path(*args, **kwargs):
    return None


def get_detector(*args, **kwargs):
    return None


def _resolve_yolo_device(*args, **kwargs) -> str:
    return "cpu"


def track_saliency_in_frames(*args, **kwargs) -> list:
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  clip_scoring / render_plan_keyframes / transcript_corrector / retranscribe
# ═══════════════════════════════════════════════════════════════════════════

def four_axis_scoring_enabled(*args, **kwargs) -> bool:
    """The reframer bridge populates the 4 axes directly — legacy path off."""
    return False


def keyframes_from_cached_render_plan(*args, **kwargs) -> list:
    return []


def correct_transcript(segments, *args, **kwargs):
    """Legacy transcript corrector — the reframer's TACT pass already cleans
    the transcript, so this is an inert pass-through."""
    return segments


def _adaptive_batch_size(*args, **kwargs) -> int:
    return 8


def _locate_audio_path(*args, **kwargs):
    return None
