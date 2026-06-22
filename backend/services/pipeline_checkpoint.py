"""Engine checkpoint — persist the reframer's perception + plan so a
failed / interrupted job can RESUME instead of re-running the single most
expensive stage (face/motion detection + Whisper transcription + planning).

The analysis pipeline already caches frame + audio extraction (keyed on the
source SHA-256, see :mod:`pipeline_helpers`). Everything after that — the
``ReframeEngine.analyze()`` call that produces the in-memory
:class:`PerceptionResult` + :class:`RenderPlan` — used to re-run from scratch
on every re-analyze, and there was nothing to resume from when the container
died mid-run. This module serializes those two objects to a per-job
``checkpoint/`` directory right after the engine stage finishes, and reloads
them on the next run when the source + analysis parameters are unchanged.

A reload returns a perception the post-engine stages (bridge / clipper /
evaluator) consume exactly as a freshly-computed one: the int-keyed timeline
dicts (``face_timeline``, ``motion_timeline`` …) are restored to integer keys
(downstream code does ``int(t_ms / 1000)`` on them, which would crash on the
string keys JSON produces). The only field deliberately dropped is
``coverage_ledger`` — a millisecond-resolution structure read only by the
planner (which has already run by checkpoint time); see the note by
``_INT_KEYED_FIELDS``.

Validity is gated on a *signature* (source SHA + sample-fps + aspect ratio +
source language + vocal-separation setting + a planner-flag fingerprint), so a
stale checkpoint is never silently reused after the source or the analysis
settings change.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)

# Bump when the on-disk shape changes so old checkpoints are ignored
# (re-run the engine) instead of deserialized into the wrong fields.
# v2: invalidates checkpoints written before the transcribe-reuses-audio.wav
# fix — those could carry an empty (timed-out) transcript that a resume would
# otherwise keep reusing.
CHECKPOINT_VERSION = 2

_PERCEPTION_FILE = "engine_perception.json"
_PLAN_FILE = "engine_plan.json"
_META_FILE = "engine_meta.json"

# Timeline fields whose keys are integers (time_ms, or track_id for the
# speaker map). JSON stringifies dict keys on write — these are restored to
# ``int`` on load so consumers that do arithmetic on the key keep working.
_INT_KEYED_FIELDS = (
    "face_timeline", "motion_timeline", "motion_hotspot", "speaker_timeline",
    "track_speaker_map", "speech_active", "audio_rms", "audio_events",
    "person_timeline", "saliency_hotspot",
)
# NOTE: ``coverage_ledger`` is deliberately NOT checkpointed. It is a
# millisecond-resolution structure (20ms bins → tens of thousands of
# ``LedgerBin`` dataclasses on a long video) consumed ONLY by the planner,
# which runs *inside* ``engine.analyze()`` — i.e. before this checkpoint is
# written. Nothing after the engine (bridge / summary / clip detection) reads
# it. Serializing + rebuilding all those bins in pure Python on a resume held
# the GIL long enough to starve the asyncio event loop, freezing the 2s
# ``/api/diagnostics/gpu-status`` poll (the "VRAM bar stopped working" report).
# Dropping it makes resume cheap; a restored perception simply has
# ``coverage_ledger=None``, which the model documents as the supported
# "fall back to speech_active" state.


def _numpy_safe_default(obj):
    """JSON fallback for numpy scalars/arrays that slip through detection."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    return str(obj)


def checkpoint_dir(job_id: str) -> str:
    return f"/data/uploads/{job_id}/checkpoint"


def checkpoint_signature(
    *,
    source_sha: str,
    source_language: str,
    sample_fps: float,
    aspect_ratio: str,
    vocal_sep_key: str = "",
    planner_fingerprint: str = "",
) -> dict:
    """Identity of the engine inputs. A reload only happens when the saved
    signature matches the one the current run would produce — otherwise the
    cached perception/plan no longer describe this source/config."""
    return {
        "version": CHECKPOINT_VERSION,
        "source_sha": source_sha or "",
        "source_language": (source_language or "auto").strip().lower() or "auto",
        "sample_fps": round(float(sample_fps or 0.0), 3),
        "aspect_ratio": (aspect_ratio or "9:16").strip(),
        "vocal_sep_key": vocal_sep_key or "",
        "planner_fingerprint": planner_fingerprint or "",
    }


def _signatures_match(saved: dict, expected: dict) -> bool:
    if not isinstance(saved, dict):
        return False
    # An empty source SHA can't be trusted to identify the source — never
    # reuse a checkpoint we can't tie to specific bytes.
    if not expected.get("source_sha"):
        return False
    return all(saved.get(k) == expected.get(k) for k in expected)


# ── Serialization ────────────────────────────────────────────────────


def _serialize_perception(p) -> dict:
    d = {
        "src_w": int(getattr(p, "src_w", 0) or 0),
        "src_h": int(getattr(p, "src_h", 0) or 0),
        "fps": float(getattr(p, "fps", 30.0) or 30.0),
        "duration_ms": int(getattr(p, "duration_ms", 0) or 0),
        "total_frames": int(getattr(p, "total_frames", 0) or 0),
        "scene_cuts": list(getattr(p, "scene_cuts", None) or []),
        "transcript_segments": list(getattr(p, "transcript_segments", None) or []),
        "detected_language": getattr(p, "detected_language", "") or "",
        # coverage_ledger intentionally omitted — see note by _INT_KEYED_FIELDS.
    }
    for field in _INT_KEYED_FIELDS:
        d[field] = getattr(p, field, None) or {}
    return d


def _int_keyed(d) -> dict:
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            out[k] = v
    return out


def _deserialize_perception(d: dict):
    from backend.services.reframer_models import PerceptionResult
    p = PerceptionResult(
        src_w=int(d.get("src_w", 0) or 0),
        src_h=int(d.get("src_h", 0) or 0),
        fps=float(d.get("fps", 30.0) or 30.0),
        duration_ms=int(d.get("duration_ms", 0) or 0),
        total_frames=int(d.get("total_frames", 0) or 0),
        scene_cuts=[int(c) for c in (d.get("scene_cuts") or [])],
        transcript_segments=list(d.get("transcript_segments") or []),
        detected_language=d.get("detected_language", "") or "",
        # coverage_ledger stays at its None default (not checkpointed).
    )
    for field in _INT_KEYED_FIELDS:
        setattr(p, field, _int_keyed(d.get(field)))
    return p


def _atomic_write(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_sync(job_id: str, perception, reframer_plan, signature: dict,
               audio_meta: dict) -> bool:
    from dataclasses import asdict
    d = checkpoint_dir(job_id)
    _atomic_write(
        os.path.join(d, _PERCEPTION_FILE),
        json.dumps(_serialize_perception(perception), default=_numpy_safe_default),
    )
    _atomic_write(
        os.path.join(d, _PLAN_FILE),
        json.dumps(asdict(reframer_plan), default=_numpy_safe_default),
    )
    # Meta last: its presence is the "checkpoint is complete + valid" marker,
    # so a crash between the perception/plan writes never yields a half
    # checkpoint that load() would trust.
    _atomic_write(
        os.path.join(d, _META_FILE),
        json.dumps({"signature": signature, "audio": audio_meta or {}}, indent=2),
    )
    return True


async def save_engine_checkpoint(
    job_id: str,
    perception,
    reframer_plan,
    *,
    signature: dict,
    audio_meta: Optional[dict] = None,
) -> bool:
    """Persist the engine's perception + plan so the next run can resume.

    Best-effort: serialization runs in a worker thread and any failure is
    logged and swallowed (the next run just re-runs the engine).
    """
    import asyncio
    try:
        await asyncio.to_thread(
            _save_sync, job_id, perception, reframer_plan, signature, audio_meta or {})
        logger.info(
            "[%s] Engine checkpoint saved (%d transcript segs, %d face samples) — "
            "a resume will skip detection + transcription",
            job_id,
            len(getattr(perception, "transcript_segments", None) or []),
            sum(1 for v in (getattr(perception, "face_timeline", None) or {}).values() if v),
        )
        return True
    except Exception as exc:  # noqa: BLE001 — checkpointing must never break a run
        logger.warning("[%s] Engine checkpoint save failed (non-fatal): %s", job_id, exc)
        return False


def _load_sync(job_id: str, expected_signature: dict):
    d = checkpoint_dir(job_id)
    meta_path = os.path.join(d, _META_FILE)
    perception_path = os.path.join(d, _PERCEPTION_FILE)
    plan_path = os.path.join(d, _PLAN_FILE)
    if not (os.path.isfile(meta_path) and os.path.isfile(perception_path)
            and os.path.isfile(plan_path)):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not _signatures_match(meta.get("signature") or {}, expected_signature):
        logger.info("[%s] Engine checkpoint present but signature differs — re-running engine", job_id)
        return None

    with open(perception_path, "r", encoding="utf-8") as f:
        perception = _deserialize_perception(json.load(f))

    from backend.services.reframer_models import RenderPlan
    reframer_plan = RenderPlan.load(plan_path)

    audio = meta.get("audio") or {}
    engine_stub = SimpleNamespace(
        _perceiver_audio_device=audio.get("device"),
        _perceiver_audio_model=audio.get("model"),
        _perceiver_audio_model_requested=audio.get("requested"),
        _resumed_from_checkpoint=True,
    )
    return perception, reframer_plan, engine_stub


async def load_engine_checkpoint(job_id: str, *, signature: dict):
    """Return ``(perception, reframer_plan, engine_stub)`` from a valid
    checkpoint, or ``None`` when absent / stale / unreadable.

    ``engine_stub`` is a lightweight stand-in carrying the Whisper device +
    model strings ``_build_compute_summary`` reads, so the Compute card stays
    faithful on a resumed run without a live engine object.
    """
    import asyncio
    try:
        return await asyncio.to_thread(_load_sync, job_id, signature)
    except Exception as exc:  # noqa: BLE001 — a bad checkpoint must fall back to fresh
        logger.warning("[%s] Engine checkpoint load failed (%s) — re-running engine", job_id, exc)
        return None
