"""Standalone helpers used by the analysis pipeline.

Extracted from ``pipeline.py`` so unit tests can import them without
pulling in the LLM provider stack (``openai``, ``anthropic``, ``groq``,
``ollama`` ...). The pipeline re-exports these from its module
namespace so existing call sites keep working.

Contents:
  * :func:`_hash_file_sha256` — SHA-256 a file in chunks.
  * :func:`_maybe_use_cached_extraction` — re-analyze cache probe.
  * :func:`_write_extraction_manifest` — sidecar so the cache hit can
    rebuild ``FrameData`` without ffprobe.
  * :func:`_stage_timer` — async ctx manager that logs + records
    per-stage wall-clock time.
  * :func:`_record_pipeline_warning` / :func:`_drain_pipeline_telemetry`
    — accumulators that surface on ``JobResult.timings`` and
    ``JobResult.pipeline_warnings``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time as _time
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# ── Audio preconditioning filter chain (duration-aware) ──────────
#
# Whisper transcript coverage improves when the audio is pre-conditioned
# before ASR: highpass kills rumble, ``afftdn`` (FFT spectral denoise) pulls
# quiet speech out of constant backgrounds, and ``loudnorm`` lifts off-mic
# speech into Whisper's sensitive band. But ``afftdn`` is CPU-bound at only a
# few × realtime — on a 2 h track it alone adds ~15-20 min to the
# "frame+audio extraction" stage. Since fast GPU frame extraction finishes
# long before it, the pipeline looks "stuck at faces" (the next step can't
# start until audio is done) with no VRAM in use. highpass + single-pass
# loudnorm are far cheaper and carry most of the coverage benefit, so on long
# videos we drop ONLY ``afftdn`` and keep the rest.

def build_precondition_filters(
    precondition: bool,
    duration_sec: float,
    denoise_max_min: float = 45.0,
) -> "str | None":
    """Return the ffmpeg ``-af`` chain for audio preconditioning, or ``None``.

    ``None`` means "no ``-af`` filter" (plain extraction). When ``precondition``
    is on this always applies ``highpass`` + ``loudnorm``; the expensive
    ``afftdn`` FFT denoiser is included only when the track is at/under
    ``denoise_max_min`` minutes (``denoise_max_min <= 0`` disables that cap,
    i.e. always denoise).
    """
    if not precondition:
        return None
    # Resample to Whisper's 16 kHz target FIRST so every downstream filter runs
    # on ~3× fewer samples than a 48 kHz source (afftdn/loudnorm cost scales with
    # sample count). This is the most effective speedup available here: ffmpeg
    # audio filters are CPU-only — there is NO CUDA build of afftdn/loudnorm/
    # highpass (ffmpeg GPU accel is video decode/encode/scale only), so the work
    # can't move to the GPU; shrinking the sample count is the lever. Lossless
    # for ASR since Whisper consumes 16 kHz mono regardless.
    filters = ["aresample=16000", "highpass=f=80"]
    dur_min = (duration_sec or 0.0) / 60.0
    if denoise_max_min <= 0 or dur_min <= denoise_max_min:
        filters.append("afftdn=nf=-25")
    filters.append("loudnorm=I=-18:LRA=11:TP=-1.5")
    return ",".join(filters)


# ── Per-job stage-timing accumulator ────────────────────────────
#
# Keyed by job_id so concurrent analyses don't clobber each other;
# flushed onto JobResult.timings / pipeline_warnings at the end of the
# run by ``_drain_pipeline_telemetry``.

_pipeline_timings: dict[str, dict[str, float]] = {}
_pipeline_warnings: dict[str, list[str]] = {}
# Which device/host served each stage — "remote" / "local_gpu" / "cpu" for
# transcription, an Ollama host name for AI stages. Flushed onto
# ``JobResult.stage_locations`` alongside the timings.
_pipeline_stage_locations: dict[str, dict[str, str]] = {}


def _record_stage_location(job_id: str, stage: str, location: str) -> None:
    """Tag ``stage`` with the device/host that served it (fail-soft)."""
    if not job_id or not stage or not location:
        return
    _pipeline_stage_locations.setdefault(job_id, {})[stage] = location


def _drain_stage_locations(job_id: str) -> dict[str, str]:
    """Pop the accumulated stage→location map for ``job_id``."""
    return _pipeline_stage_locations.pop(job_id, {}) or {}


# ── Synthetic / fallback scene-description detection ────────────
#
# Every vision provider has its own way of saying "I couldn't analyze
# this frame": Ollama emits "Frame at <ts> (vision unavailable …)",
# OpenRouter emits "Frame analysis unavailable" or "Frame analysis
# unavailable — API key limit exceeded", the orchestrator-level fallback
# emits "Analysis failed", etc. Without a single source of truth the
# pipeline could either
#   (a) count error placeholders as real scenes (Key Scenes shows
#       junk), or
#   (b) miss legitimate descriptions whose wording happens to overlap
#       with a marker substring.
# This list is the canonical "this is NOT a real description" check.
# Keep all matches lowercase substrings; ``is_synthetic_scene`` does
# the case-insensitive comparison.
_SYNTHETIC_SCENE_MARKERS: tuple[str, ...] = (
    "vision skipped",
    "vision unavailable",
    "vision model crashed",
    "analysis skipped",
    "analysis failed",
    "frame analysis unavailable",
    "frame analysis failed",
    "frame not analyzed",
    "api key limit exceeded",
    "no description",
    "no analysis",
    "model returned no response",
    "model returned empty",
)


def is_synthetic_scene(scene) -> bool:
    """Return True iff ``scene.description`` is an error placeholder.

    Used by the pipeline's ``real_scenes`` filter, the warning logic
    that recommends "switch vision providers" when 0 real descriptions
    came back, and any caller that needs to know whether a scene is
    user-presentable.
    """
    desc = getattr(scene, "description", None)
    if not desc:
        return True
    low = str(desc).strip().lower()
    if not low:
        return True
    if low.startswith("frame at "):
        # Ollama's "Frame at 12.5s (vision unavailable — CLIP on CPU)"
        return True
    return any(marker in low for marker in _SYNTHETIC_SCENE_MARKERS)


def _record_pipeline_warning(job_id: str, message: str) -> None:
    """Append a soft-warning string to the per-job warnings list.

    Surfaced on ``JobResult.pipeline_warnings`` so the UI can show
    non-fatal issues alongside the result. Idempotent: duplicate
    messages are coalesced so the list stays short.
    """
    bag = _pipeline_warnings.setdefault(job_id, [])
    if message and message not in bag:
        bag.append(message)


_COMPUTE_STAGE_LABELS = {
    "frame_extract": "Frame extraction",
    "yolo_world": "Subject detection (YOLO-World)",
    "whisper": "Transcription (Whisper)",
}


def cpu_fallback_stages(compute_summary: dict) -> list:
    """Return human labels of stages in ``compute_summary`` that ran on CPU.

    ``compute_summary`` shape: ``{stage: {"device": "cuda:0"|"cpu", ...}}``.
    Used to warn the user when the GPU was enabled but a heavy stage fell
    back to CPU (~10-30x slower) — instead of the run silently crawling.
    """
    out = []
    for k, v in (compute_summary or {}).items():
        dev = ""
        if isinstance(v, dict):
            dev = str(v.get("device", ""))
        if dev.startswith("cpu"):
            out.append(_COMPUTE_STAGE_LABELS.get(k, k))
    return out


def translation_progress_pct(message: str, *, lo: int = 63, hi: int = 69,
                             default: int = 64) -> int:
    """Map a translation status message onto the translation band of the bar.

    Translation reports per-batch messages like ``"Translating subtitles…
    (310/621)"``. Parse the trailing ``(a/b)`` cue count and scale it into the
    ``lo``-``hi`` band so the main progress bar actually advances during the
    multi-minute pass instead of sitting at a static %. Falls back to
    ``default`` when there's no parseable hint.
    """
    try:
        import re as _re
        m = _re.search(r"\((\d+)\s*/\s*(\d+)\)", message or "")
        if m and int(m.group(2)) > 0:
            frac = min(1.0, max(0.0, int(m.group(1)) / int(m.group(2))))
            return lo + int(frac * max(0, hi - lo))
    except Exception:
        pass
    return default


def resolve_clip_progress(frac, message, last_pct: int):
    """Decide the (pct, label) for one clip-stage progress tick, or ``None`` to
    skip it.

    - ``pct`` maps ``frac`` 0-1 onto the 80-97 % band, forward-only (never below
      ``last_pct`` — the ProgressBar is monotonic).
    - WITH a ``message`` (e.g. "Exporting clip 12/63 …"): always emit, even if
      the % is flat — the changing text is what keeps the activity log and the
      frontend's stuck-timer alive through the long export tail (the bug where a
      63-clip export sat at ~96 % with one static label and tripped the
      "pipeline may be stuck" banner).
    - WITHOUT a message: emit only when the bar actually advances, with a
      band-derived label, so identical updates aren't spammed.

    Returns ``(pct, label)`` or ``None``.
    """
    try:
        f = float(frac or 0)
    except (TypeError, ValueError):
        f = 0.0
    f = max(0.0, min(1.0, f))
    pct = max(80 + int(f * 17), int(last_pct))
    if message:
        return pct, str(message)
    if pct <= last_pct:
        return None
    if f < 0.10:
        label = "Scoring clip candidates from signal density..."
    elif f < 0.55:
        label = "Asking the VLM to discover viral moments..."
    elif f < 0.75:
        label = "Judging clip candidates for hook + flow..."
    elif f < 0.95:
        label = "Exporting top clips..."
    else:
        label = "Finalizing clip detection..."
    return pct, label


def resolve_sample_fps(
    video_duration_s: float,
    *,
    max_samples: int,
    fps_ceiling: float,
    fps_floor: float,
    abs_floor: float = 0.2,
) -> float:
    """Pick the reframer's per-frame sample rate (the dominant analysis cost).

    Cap-respecting: never above ``fps_ceiling``, and never more than
    ``max_samples`` frames total. The comfort floor (``fps_floor``) may only
    RAISE the rate for shorter videos — it must never override the sample cap on
    long ones. The old ``max(fps_floor, …)`` did exactly that, so a 128-min
    video sampled at the 1.2 fps floor (~9 200 frames, 5x the 1 800 cap) and
    face/motion detection ran ~5x longer than intended. ``abs_floor`` is a hard
    minimum so a multi-hour video still gets usable temporal coverage when the
    cap alone would drop below it.
    """
    if video_duration_s <= 0:
        return fps_ceiling
    fps = min(fps_ceiling, max_samples / video_duration_s)
    if fps_floor * video_duration_s <= max_samples:
        fps = max(fps, fps_floor)
    return max(fps, abs_floor)


def _drain_pipeline_telemetry(job_id: str) -> tuple[dict[str, float], list[str]]:
    """Pop the accumulated timings + warnings for ``job_id``.

    Returns ``({stage: seconds}, [warning, ...])``. Safe to call
    multiple times — the second call returns empty.
    """
    timings = _pipeline_timings.pop(job_id, {}) or {}
    warnings = _pipeline_warnings.pop(job_id, []) or []
    return timings, warnings


@asynccontextmanager
async def _stage_timer(job_id: str, stage: str):
    """Log + record wall-clock time for a pipeline stage.

    If a stage runs twice (rare — recovery retries) the cumulative
    time is kept. The UI cares about "where time went", not which
    retry attempt it was.

    Also stamps the stage name into the request context so outbound
    remote calls (Ollama hosts, remote Whisper, the GPU Companion) carry
    ``X-ClipAI-Stage`` for their live activity feeds.
    """
    t0 = _time.monotonic()
    logger.info("[%s] Stage '%s' started", job_id, stage)
    try:
        from backend.services.request_context import set_stage
        set_stage(stage)
    except Exception:
        pass
    try:
        yield
    finally:
        elapsed = _time.monotonic() - t0
        logger.info("[%s] Stage '%s' finished in %.1fs", job_id, stage, elapsed)
        bag = _pipeline_timings.setdefault(job_id, {})
        bag[stage] = round(bag.get(stage, 0.0) + elapsed, 2)


# ── Extraction cache (frame + audio re-use across re-analyze) ──


def _hash_file_sha256(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of ``path`` in 1 MiB chunks. Returns hex digest.

    Returns the empty string when the file can't be read so callers
    don't have to special-case missing files. Hashing a 4 GB video
    takes ~4 s on an SSD; we run it in a thread to keep the event
    loop responsive.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


async def _maybe_use_cached_extraction(
    *,
    job_id: str,
    video_path: str,
    frames_dir: str,
    audio_path: str,
    expected_sha: str,
    precomputed_sha: "str | None" = None,
):
    """Return cached ``(frames, scene_cut_timestamps)`` if the source
    hasn't changed, else ``None``.

    Conservative: requires both the audio file and at least 5 frame
    images to be present, and the source SHA-256 to match the value
    recorded on the job. Falls back to a fresh extract on any
    mismatch — *never* silently uses stale data.

    ``precomputed_sha`` lets the pipeline pass the digest from its
    background hash task (started right after metadata extraction) so the
    source file is hashed exactly once per run; when ``None`` the probe
    hashes internally exactly as before (standalone-caller fallback).
    """
    if not expected_sha:
        return None
    if not (os.path.isdir(frames_dir) and os.path.isfile(audio_path)):
        return None

    try:
        frame_files = sorted(
            f for f in os.listdir(frames_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        )
    except OSError:
        return None
    if len(frame_files) < 5:
        return None

    # Use the pipeline's pre-started hash when available; otherwise compare
    # hashes off the event loop (the legacy path).
    actual_sha = precomputed_sha
    if not actual_sha:
        actual_sha = await asyncio.to_thread(_hash_file_sha256, video_path)
    if not actual_sha or actual_sha != expected_sha:
        return None

    # Reconstruct the FrameData list. Lazy import to avoid pulling
    # the whole ``backend.models`` chain into pure-Python sandboxes.
    from backend.models import FrameData

    sidecar = os.path.join(frames_dir, "manifest.json")
    timestamps: list[float] = []
    scene_cut_timestamps: list[float] = []
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, "r") as f:
                data = json.load(f)
            timestamps = [float(t) for t in (data.get("timestamps") or [])]
            scene_cut_timestamps = [float(t) for t in (data.get("scene_cuts") or [])]
        except Exception:
            timestamps = []
            scene_cut_timestamps = []

    if not timestamps or len(timestamps) != len(frame_files):
        # Fall back to filename-index ordering with a 1 s grid placeholder.
        # We cannot recover scene cuts without a sidecar, so they degrade
        # to empty.
        timestamps = []
        for i, name in enumerate(frame_files):
            base = os.path.splitext(name)[0]
            tail = base.rsplit("_", 1)[-1] if "_" in base else base
            try:
                idx = int(tail)
                timestamps.append(idx * 1.0)
            except ValueError:
                timestamps.append(float(i))

    frames = [
        FrameData(timestamp=ts, path=os.path.join(frames_dir, name))
        for ts, name in zip(timestamps, frame_files)
    ]
    logger.info(
        "[%s] Cache hit: %d frames, audio=%.1f KB, %d scene cuts",
        job_id, len(frames),
        os.path.getsize(audio_path) / 1024.0,
        len(scene_cut_timestamps),
    )
    return frames, scene_cut_timestamps


def _write_extraction_manifest(
    frames_dir: str,
    frames,
    scene_cut_timestamps,
) -> None:
    """Persist a tiny sidecar so future cache hits can rebuild the
    ``FrameData`` list without re-decoding the video.

    Best-effort: any IO error is swallowed since we'll just fall back
    to filename parsing on the next run.
    """
    try:
        manifest = {
            "timestamps": [float(f.timestamp) for f in (frames or [])],
            "scene_cuts": [float(t) for t in (scene_cut_timestamps or [])],
        }
        os.makedirs(frames_dir, exist_ok=True)
        with open(os.path.join(frames_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)
    except Exception:
        pass
