"""Per-job request context propagated onto outbound AI/transcription calls.

The pipeline stamps the running job's identity here (contextvars, so
concurrent jobs never bleed into each other) and every remote call —
Ollama hosts, remote Whisper servers, the GPU Companion proxy — attaches
it as ``X-ClipAI-*`` headers. The Companion uses these headers to render
its live "what is ClipAI doing on my GPU right now" activity feed.

Values are plain informational strings; nothing here is a secret.
"""
from __future__ import annotations

import contextvars

_job_id: contextvars.ContextVar[str] = contextvars.ContextVar("clipai_job_id", default="")
_job_title: contextvars.ContextVar[str] = contextvars.ContextVar("clipai_job_title", default="")
_job_stage: contextvars.ContextVar[str] = contextvars.ContextVar("clipai_job_stage", default="")
_job_progress: contextvars.ContextVar[int] = contextvars.ContextVar("clipai_job_progress", default=-1)


def set_job(job_id: str = "", title: str = "") -> None:
    """Stamp the current job's id/title. Called once at pipeline start."""
    _job_id.set(job_id or "")
    _job_title.set(title or "")


def set_stage(stage: str) -> None:
    """Stamp the currently running pipeline stage (updated per stage)."""
    _job_stage.set(stage or "")


def set_progress(pct: int) -> None:
    """Stamp the job's overall progress (0-100) so the Companion can show a live
    bar for the work it serves. Best-effort; -1 means unknown."""
    try:
        _job_progress.set(max(0, min(100, int(pct))))
    except Exception:
        pass


def clear() -> None:
    _job_id.set("")
    _job_title.set("")
    _job_stage.set("")
    _job_progress.set(-1)


def current_job_id() -> str:
    """The job this task is working for, or "" outside a pipeline run."""
    return _job_id.get()


def _header_safe(value: str, limit: int = 180) -> str:
    """HTTP headers must be latin-1; strip anything that isn't and cap length."""
    cleaned = "".join(ch for ch in (value or "") if 32 <= ord(ch) < 256)
    return cleaned[:limit]


def clipai_headers() -> dict:
    """``X-ClipAI-*`` headers for the current job, or the identity headers
    alone outside a job.

    ``X-ClipAI-Port`` is ALWAYS attached: the Companion combines it with the
    connection's peer IP to learn this server's reachable base URL, which
    powers its self-update download for manually-added (never GUI-paired)
    setups."""
    import os as _os
    headers = {"X-ClipAI-Port": _os.environ.get("CLIPAI_PORT", "1353")}
    # Explicit override for networks where the source IP the Companion sees is
    # NOT the address it can reach us at (custom bridges, macvlan, reverse
    # proxies). Set CLIPAI_PUBLIC_URL to the http(s)://host:port the Companion
    # PC uses to reach ClipAI; the Companion prefers it over peer-IP inference.
    _pub = (_os.environ.get("CLIPAI_PUBLIC_URL", "") or "").strip()
    if _pub.startswith("http://") or _pub.startswith("https://"):
        headers["X-ClipAI-Origin"] = _header_safe(_pub, 200)
    job_id = _job_id.get()
    if job_id:
        headers["X-ClipAI-Job-Id"] = _header_safe(job_id, 80)
    stage = _job_stage.get()
    if stage:
        headers["X-ClipAI-Stage"] = _header_safe(stage, 80)
    title = _job_title.get()
    if title:
        headers["X-ClipAI-Job-Title"] = _header_safe(title)
    pct = _job_progress.get()
    if pct is not None and pct >= 0:
        headers["X-ClipAI-Progress"] = str(int(pct))
    return headers
