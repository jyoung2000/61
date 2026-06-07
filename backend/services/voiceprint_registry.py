"""Voiceprint persistence & cross-job speaker naming (Task 3).

Otter.ai's standout feature: name a voice once and it auto-labels that same
voice by name in *future* recordings ("continuous learning"). ClipAI used to
throw speaker identity away after each job. This module persists a registry
of speaker voiceprints so naming carries across jobs.

Pipeline:
  * After diarization + fusion, a speaker embedding is extracted per
    job-local ``SPEAKER_xx`` and matched against the registry. Matches apply
    the stored name automatically.
  * When the user renames a speaker via the existing rename endpoint, that
    speaker's embedding is captured and ``enroll_or_update``-d — the learning
    loop. The centroid is a running mean, so accuracy improves with use.

Persistence: ``/data/logs/voiceprints.json`` on the same Docker mount that
backs ``user_settings.json``, so the registry survives container rebuilds.

**Graceful degradation is the whole game here.** The embedding *backend*
(pyannote ``pyannote/embedding`` model, ``HF_TOKEN``, torch) may be absent.
When it is, ``extract_speaker_embeddings`` / ``embed_audio_regions`` return
empty and the registry simply stays empty — per-job ``Speaker N`` labels
stand, nothing raises. The registry's own math (cosine match, centroid
update, persistence) is pure-Python and always available.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from typing import Optional

logger = logging.getLogger("clipai.voiceprint_registry")

DEFAULT_MATCH_THRESHOLD = 0.75


def _resolve_data_dir() -> str:
    docker_path = "/data/logs"
    if os.path.isdir(docker_path) and os.access(docker_path, os.W_OK):
        return docker_path
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".clipai",
    )
    os.makedirs(local_path, exist_ok=True)
    return local_path


def _voiceprints_path() -> str:
    return os.path.join(_resolve_data_dir(), "voiceprints.json")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 on degenerate input)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _match_threshold() -> float:
    try:
        from backend.config import settings
        return float(getattr(settings, "VOICEPRINT_MATCH_THRESHOLD", DEFAULT_MATCH_THRESHOLD))
    except Exception:
        return DEFAULT_MATCH_THRESHOLD


class VoiceprintRegistry:
    """A persisted store of named speaker voiceprints.

    Schema (``/data/logs/voiceprints.json``)::

        {voiceprint_id: {
            "name": str,
            "centroid_embedding": [float, ...],
            "sample_count": int,
            "updated_at": float,   # unix epoch
        }}
    """

    def __init__(self, path: Optional[str] = None):
        self._path = path or _voiceprints_path()
        self._data: dict = {}
        self.load()

    # ── persistence ──────────────────────────────────────────────────────
    def load(self) -> dict:
        self._data = {}
        if not os.path.exists(self._path):
            return self._data
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._data = raw
        except Exception as e:
            logger.warning("Failed to load voiceprints from %s: %s", self._path, e)
            self._data = {}
        return self._data

    def save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning("Failed to save voiceprints to %s: %s", self._path, e)
            return False

    # ── matching ─────────────────────────────────────────────────────────
    def match(self, embedding, threshold: Optional[float] = None):
        """Return ``(voiceprint_id, name)`` for the best match above
        ``threshold``, else ``None``. Conservative by design — better to
        leave a speaker as ``Speaker N`` than to mislabel."""
        if not embedding:
            return None
        threshold = threshold if threshold is not None else _match_threshold()
        emb = [float(x) for x in embedding]
        best_id = None
        best_name = ""
        best_sim = -1.0
        for vid, v in self._data.items():
            cent = v.get("centroid_embedding")
            if not cent:
                continue
            sim = _cosine_similarity(emb, [float(x) for x in cent])
            if sim > best_sim:
                best_sim = sim
                best_id = vid
                best_name = v.get("name", "")
        if best_id is not None and best_sim >= threshold:
            return best_id, best_name
        return None

    # ── enrollment / learning ─────────────────────────────────────────────
    def enroll_or_update(self, embedding, name: str,
                         voiceprint_id: Optional[str] = None) -> Optional[str]:
        """Enroll ``name``'s voiceprint, or update an existing one with a
        running-mean centroid: ``new = (centroid*n + emb) / (n + 1)``.

        Target selection: explicit ``voiceprint_id`` → existing entry with
        the same name → embedding match above threshold → otherwise a new
        entry. Returns the voiceprint id, or ``None`` when ``embedding`` is
        empty (no-op).
        """
        if not embedding:
            return None
        emb = [float(x) for x in embedding]
        name = (name or "").strip()

        target_id = voiceprint_id if voiceprint_id in self._data else None
        if target_id is None and name:
            for vid, v in self._data.items():
                if (v.get("name", "") or "").strip().lower() == name.lower():
                    target_id = vid
                    break
        if target_id is None:
            m = self.match(emb)
            if m is not None:
                target_id = m[0]

        now = time.time()
        if target_id is not None and target_id in self._data:
            v = self._data[target_id]
            n = max(1, int(v.get("sample_count", 1)))
            cent = [float(x) for x in (v.get("centroid_embedding") or emb)]
            if len(cent) == len(emb):
                cent = [(c * n + e) / (n + 1) for c, e in zip(cent, emb)]
            else:
                cent = emb
            v["centroid_embedding"] = cent
            v["sample_count"] = n + 1
            if name:
                v["name"] = name
            v["updated_at"] = now
        else:
            target_id = uuid.uuid4().hex
            self._data[target_id] = {
                "name": name,
                "centroid_embedding": emb,
                "sample_count": 1,
                "updated_at": now,
            }
        self.save()
        return target_id

    # ── review / privacy ───────────────────────────────────────────────────
    def list_voiceprints(self) -> list[dict]:
        """Return a UI-friendly summary (no raw embeddings)."""
        out = []
        for vid, v in self._data.items():
            out.append({
                "id": vid,
                "name": v.get("name", ""),
                "sample_count": int(v.get("sample_count", 0)),
                "updated_at": v.get("updated_at", 0),
                "dims": len(v.get("centroid_embedding") or []),
            })
        return out

    def delete(self, voiceprint_id: str) -> bool:
        if voiceprint_id in self._data:
            del self._data[voiceprint_id]
            self.save()
            return True
        return False

    def clear(self) -> int:
        n = len(self._data)
        self._data = {}
        self.save()
        return n


# ── module-level default registry ─────────────────────────────────────────

_registry: Optional[VoiceprintRegistry] = None


def get_registry() -> VoiceprintRegistry:
    global _registry
    if _registry is None:
        _registry = VoiceprintRegistry()
    return _registry


# ── embedding backend (feature-detected, no-op when unavailable) ───────────

def _voiceprint_enabled() -> bool:
    try:
        from backend.config import settings
        return bool(getattr(settings, "VOICEPRINT_ENABLED", True))
    except Exception:
        return True


class SpeakerEmbedder:
    """Thin wrapper over pyannote's embedding model.

    Lazily loads ``pyannote/embedding`` (or the inference helper) when an
    ``HF_TOKEN`` is present. ``available`` is False when torch / pyannote /
    the token are missing, in which case every call returns empty results.
    Unloads after use to respect the 4 GB GTX 1650 budget (matching
    ``NMTTranslator.unload()``).
    """

    def __init__(self):
        self.available = False
        self._model = None

    def try_load(self) -> bool:
        try:
            # Also honor the Settings/config field (HF_AUTH_TOKEN) — the UI and
            # overlay use it, while this read historically only checked the env.
            try:
                from backend.config import settings as _settings
                _cfg_token = (getattr(_settings, "HF_AUTH_TOKEN", "") or "").strip()
            except Exception:
                _cfg_token = ""
            hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
                        or _cfg_token)
            if not hf_token:
                return False
            from pyannote.audio import Inference, Model  # type: ignore
            import torch  # noqa: F401
            model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_token)
            self._model = Inference(model, window="whole")
            try:
                if torch.cuda.is_available():
                    self._model.to(torch.device("cuda"))
            except Exception:
                pass
            self.available = True
        except Exception as e:
            logger.info("Speaker embedding backend unavailable (%s) — voiceprints disabled", e)
            self.available = False
        return self.available

    def embed_region(self, audio_path: str, start_s: float, end_s: float) -> Optional[list]:
        if not self.available or self._model is None:
            return None
        try:
            from pyannote.core import Segment  # type: ignore
            excerpt = Segment(max(0.0, start_s), max(start_s + 0.1, end_s))
            vec = self._model.crop(audio_path, excerpt)
            data = getattr(vec, "data", vec)
            flat = data.reshape(-1).tolist() if hasattr(data, "reshape") else list(data)
            return [float(x) for x in flat]
        except Exception as e:
            logger.debug("embed_region failed (%s)", e)
            return None

    def unload(self) -> None:
        self._model = None
        self.available = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def extract_speaker_embeddings(audio_path: str, speaker_timeline: dict) -> dict:
    """Extract one mean embedding per raw speaker id from the diarization
    timeline. Returns ``{raw_speaker_id: embedding}``.

    No-op (returns ``{}``) when voiceprints are disabled, the backend is
    unavailable, the audio path is missing, or the timeline is empty — so
    callers degrade to per-job ``Speaker N`` labels without any guard.
    """
    if not _voiceprint_enabled() or not speaker_timeline or not audio_path:
        return {}
    if not os.path.exists(audio_path):
        return {}
    embedder = SpeakerEmbedder()
    if not embedder.try_load():
        return {}
    try:
        # Collect contiguous time ranges per speaker from the 200 ms timeline.
        ranges: dict = {}
        for t_ms in sorted(speaker_timeline):
            sid = speaker_timeline[t_ms]
            if not sid:
                continue
            ranges.setdefault(sid, []).append(t_ms)
        out: dict = {}
        for sid, times in ranges.items():
            # Embed up to the first few windows and average them.
            vecs = []
            for t_ms in times[:5]:
                v = embedder.embed_region(audio_path, t_ms / 1000.0, t_ms / 1000.0 + 1.5)
                if v:
                    vecs.append(v)
            if vecs:
                dims = len(vecs[0])
                mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dims)]
                out[sid] = mean
        return out
    except Exception as e:
        logger.info("extract_speaker_embeddings failed (%s) — voiceprints skipped", e)
        return {}
    finally:
        embedder.unload()


def enroll_from_segments(audio_path: str, segments: list, name: str) -> Optional[str]:
    """Capture a voiceprint for ``name`` from the audio regions of the given
    transcript ``segments`` and enroll/update the registry (the learning
    loop fired when a user renames a speaker).

    No-ops (returns ``None``) when voiceprints are disabled, the backend /
    audio are unavailable, or no embedding could be extracted.
    """
    if not _voiceprint_enabled() or not name or not audio_path or not segments:
        return None
    if not os.path.exists(audio_path):
        return None
    embedder = SpeakerEmbedder()
    if not embedder.try_load():
        return None
    try:
        vecs = []
        for seg in segments[:8]:
            start = float(seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0))
            end = float(seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", start))
            v = embedder.embed_region(audio_path, start, end)
            if v:
                vecs.append(v)
        if not vecs:
            return None
        dims = len(vecs[0])
        mean = [sum(v[i] for v in vecs) / len(vecs) for i in range(dims)]
        return get_registry().enroll_or_update(mean, name)
    except Exception as e:
        logger.info("enroll_from_segments failed (%s) — voiceprint not learned", e)
        return None
    finally:
        embedder.unload()


def locate_job_audio(job_id: str) -> Optional[str]:
    """Best-effort lookup of a job's audio file under /data/uploads."""
    import glob
    upload_dir = f"/data/uploads/{job_id}"
    for ext in ("wav", "mp3", "m4a", "aac", "ogg", "flac"):
        matches = glob.glob(f"{upload_dir}/*.{ext}")
        if matches:
            return matches[0]
    extracted = os.path.join(upload_dir, "audio.wav")
    if os.path.isfile(extracted):
        return extracted
    return None


def apply_voiceprint_names(transcript: list, speaker_timeline: dict,
                           audio_path: str, label_map: Optional[dict] = None) -> list:
    """Match each job-local speaker against the registry and apply stored
    names to the transcript. Returns the transcript unchanged on any
    degradation (disabled, no backend, no matches)."""
    if not _voiceprint_enabled():
        return transcript
    embeddings = extract_speaker_embeddings(audio_path, speaker_timeline)
    if not embeddings:
        return transcript
    registry = get_registry()
    # raw_id → display label (Speaker N)
    if label_map is None:
        from backend.services.speaker_fusion import _build_label_map
        label_map = _build_label_map(speaker_timeline)
    # display label → matched name
    rename: dict = {}
    for raw_id, emb in embeddings.items():
        m = registry.match(emb)
        if m is not None:
            label = label_map.get(raw_id)
            if label:
                rename[label] = m[1]
    if not rename:
        return transcript
    out = []
    for seg in transcript:
        if isinstance(seg, dict):
            spk = seg.get("speaker")
            if spk in rename:
                seg = {**seg, "speaker": rename[spk]}
        else:
            spk = getattr(seg, "speaker", None)
            if spk in rename and hasattr(seg, "model_copy"):
                seg = seg.model_copy(update={"speaker": rename[spk]})
        out.append(seg)
    return out
