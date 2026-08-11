"""Push job-progress heartbeats to the paired GPU Companion.

The Companion renders a live progress bar for the job it's helping with, but it
only *sees* the pipeline when an AI request (Ollama / Whisper) actually hits its
proxy. During local-only stages — video decode, frame extraction, NVENC
encoding on the SERVER GPU — no request reaches it, so its bar would freeze at
the last value. This module fire-and-forgets the overall progress to the
Companion's ``POST /v1/progress`` (carrying the same ``X-ClipAI-*`` headers the
AI routes use) so the bar tracks the container. No-op when nothing is paired.
"""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("clipai")

# Per-job throttle: (monotonic_ts, last_progress). Skip a send when it's been
# < MIN_INTERVAL_S since the last one AND the percentage hasn't changed.
_last_sent: dict[str, tuple[float, int]] = {}
MIN_INTERVAL_S = 1.5


def heartbeat(progress: int) -> None:
    """Schedule a progress heartbeat for the current job (throttled).

    Safe to call from the pipeline's hot path: it captures the running job
    context, throttles, and dispatches an async fire-and-forget POST. Silently
    does nothing outside a running event loop or a job context."""
    try:
        from backend.services.request_context import current_job_id
        job_id = current_job_id()
        if not job_id:
            return
        now = time.monotonic()
        last = _last_sent.get(job_id)
        if last is not None and (now - last[0]) < MIN_INTERVAL_S and last[1] == int(progress):
            return
        _last_sent[job_id] = (now, int(progress))
        # create_task copies the current context, so the contextvar-based
        # X-ClipAI-* headers resolve correctly inside the task.
        asyncio.create_task(_post())
    except RuntimeError:
        # No running loop (called from a worker thread) — the next in-loop
        # progress update will carry it.
        pass
    except Exception:  # never let telemetry break the pipeline
        pass


def forget(job_id: str) -> None:
    """Drop a finished job's throttle state."""
    _last_sent.pop(job_id, None)


def job_ended(job_id: str) -> None:
    """Tell the Companion a job has ENDED (completed / failed / cancelled /
    deleted) so it clears its active-job display immediately, instead of waiting
    out the 45s staleness timeout. Best-effort; safe to call anywhere."""
    forget(job_id)
    try:
        asyncio.create_task(_post_ended(job_id))
    except RuntimeError:
        # No running loop (worker thread) — fire a throwaway loop just for this.
        try:
            asyncio.run(_post_ended(job_id))
        except Exception:
            pass
    except Exception:
        pass


async def _post_ended(job_id: str) -> None:
    try:
        import httpx
        from backend.services import ollama_registry as reg

        host = reg.companion_host()
        if host is None:
            return
        base = reg.companion_base(host)
        if not base:
            return
        headers = {"X-ClipAI-Job-Id": job_id or "", "X-ClipAI-Job-Ended": "1"}
        token = getattr(host, "token", "") or ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(f"{base}/v1/progress", headers=headers)
    except Exception:
        pass


# Why-are-heartbeats-not-arriving diagnostics: the Companion GUI's per-job
# cards are fed ONLY by these posts, and a silently-failing sender leaves the
# GUI saying "No active job" while a pipeline runs — with nothing in any log
# to explain it. One throttled line per cause per 10 minutes.
_diag_last: dict[str, float] = {}


def _diag(reason: str) -> None:
    now = time.monotonic()
    if now - _diag_last.get(reason, 0.0) >= 600.0:
        _diag_last[reason] = now
        logger.warning("Companion progress heartbeat not delivered: %s "
                       "(the Companion GUI will not show this job)", reason)


async def _post() -> None:
    try:
        import httpx
        from backend.services import ollama_registry as reg
        from backend.services.request_context import clipai_headers

        host = reg.companion_host()
        if host is None:
            _diag("no Companion host in the Ollama host registry")
            return
        base = reg.companion_base(host)
        if not base:
            _diag("registered Companion host has no usable base URL")
            return
        headers = dict(clipai_headers())
        if not headers.get("X-ClipAI-Job-Id"):
            _diag("no job id in the request context (contextvars not propagated)")
            return
        token = getattr(host, "token", "") or ""
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(f"{base}/v1/progress", headers=headers)
        if r.status_code != 200:
            _diag(f"Companion answered HTTP {r.status_code}")
    except Exception as e:
        # Best-effort: a paused/offline Companion (503/timeout) must never
        # affect the job. The next heartbeat retries.
        _diag(f"unreachable: {type(e).__name__}")
