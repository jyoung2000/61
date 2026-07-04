"""Post-engine stage checkpoints — durable, signature-gated snapshots of
expensive stages that run AFTER the engine checkpoint (translation, judge
verdicts). The engine checkpoint made detection + transcription resumable;
these make a resume skip the LLM-heavy stages it used to re-pay.

Same discipline as :mod:`pipeline_checkpoint`:
  * atomic tmp+rename+fsync writes (reuses ``_atomic_write``)
  * a stage is only reused when its saved signature matches exactly
  * best-effort — any failure logs and falls back to re-running the stage
  * saves run in a worker thread, never on the pipeline's hot path

Files live next to the engine checkpoint: ``<job>/checkpoint/<stage>.json``
with shape ``{"signature": {...}, "payload": {...}}``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

from backend.services.pipeline_checkpoint import _atomic_write, checkpoint_dir

logger = logging.getLogger(__name__)


def transcript_hash(segments) -> str:
    """Stable identity of a transcript: cue timing + text, nothing volatile.

    Used as the translation signature's input key — it captures the ACTUAL
    text being translated, so a different Whisper model, an edited cue, or a
    different source video all invalidate the cached translation without
    needing the source SHA threaded through the post-processing stage."""
    h = hashlib.sha256()
    for seg in segments or []:
        get = seg.get if isinstance(seg, dict) else lambda k, d=None: getattr(seg, k, d)
        h.update(f"{float(get('start', 0) or 0):.2f}|{float(get('end', 0) or 0):.2f}|"
                 f"{(get('text', '') or '').strip()}\n".encode("utf-8"))
    return h.hexdigest()


def _stage_path(job_id: str, stage: str) -> str:
    return os.path.join(checkpoint_dir(job_id), f"{stage}.json")


def _save_sync(job_id: str, stage: str, payload: dict, signature: dict) -> None:
    from backend.services import fastjson
    _atomic_write(
        _stage_path(job_id, stage),
        fastjson.dumps_bytes({"signature": signature, "payload": payload}),
    )


async def save_stage_checkpoint(job_id: str, stage: str, payload: dict,
                                *, signature: dict) -> bool:
    """Persist a stage's output. Best-effort: failure only means the next
    resume re-runs this stage."""
    import asyncio
    try:
        await asyncio.to_thread(_save_sync, job_id, stage, payload, signature)
        logger.info("[%s] Stage checkpoint saved: %s — a resume will skip this stage",
                    job_id, stage)
        return True
    except Exception as exc:  # noqa: BLE001 — checkpointing must never break a run
        logger.error("[%s] Stage checkpoint save FAILED for %s (%s) — a future "
                     "resume will re-run this stage", job_id, stage, exc)
        return False


def _load_sync(job_id: str, stage: str, expected_signature: dict) -> Optional[dict]:
    path = _stage_path(job_id, stage)
    if not os.path.isfile(path):
        return None
    from backend.services import fastjson
    with open(path, "rb") as f:
        data = fastjson.loads(f.read())
    saved_sig = data.get("signature")
    if not isinstance(saved_sig, dict) or not expected_signature:
        return None
    # Exact-match on every expected key; an empty identity value on the
    # expected side can never be trusted (mirrors the engine-checkpoint rule).
    if any(not v for v in expected_signature.values()):
        return None
    if any(saved_sig.get(k) != v for k, v in expected_signature.items()):
        logger.info("[%s] Stage checkpoint %s present but signature differs — "
                    "re-running stage", job_id, stage)
        return None
    payload = data.get("payload")
    return payload if isinstance(payload, dict) else None


async def load_stage_checkpoint(job_id: str, stage: str,
                                *, signature: dict) -> Optional[dict]:
    """Return the stage payload when a valid checkpoint exists, else None."""
    import asyncio
    try:
        return await asyncio.to_thread(_load_sync, job_id, stage, signature)
    except Exception as exc:  # noqa: BLE001 — a bad checkpoint falls back to fresh
        logger.warning("[%s] Stage checkpoint load failed for %s (%s) — "
                       "re-running stage", job_id, stage, exc)
        return None
