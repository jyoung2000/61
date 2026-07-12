"""Automatic model placement on the GPU Companion.

The Ollama registry prefers the Companion host, but only for models that
are actually INSTALLED there — a missing llava:7b or qwen2.5:3b silently
lands summary-vision / clip-judge / SEO calls on the small local card
(spilling to CPU). This module checks the Companion's /api/tags at job
start and triggers background pulls for the models the pipeline uses.
Fire-and-forget + throttled: a pull failure or an offline Companion never
touches the job, and we ask at most once per model per THROTTLE_S.
"""
from __future__ import annotations

import asyncio
import logging
import time

from backend.config import settings

logger = logging.getLogger("clipai.companion_models")

_last_attempt: dict[str, float] = {}
THROTTLE_S = 6 * 3600.0
_tasks: set = set()


def _wanted_models() -> list[str]:
    out = []
    for key in ("OLLAMA_PRIMARY_MODEL", "OLLAMA_EDITORIAL_MODEL",
                "OLLAMA_TRANSLATION_MODEL"):
        m = (getattr(settings, key, "") or "").strip()
        if m and m not in out:
            out.append(m)
    return out


def _names_match(installed: str, wanted: str) -> bool:
    a, b = installed.split(":")[0], wanted.split(":")[0]
    return installed == wanted or (a == b and ":" not in wanted)


async def _ensure(job_id: str) -> None:
    try:
        import httpx
        from backend.services import ollama_registry as reg
        comp = reg.companion_host() if hasattr(reg, "companion_host") else None
        if comp is None:
            return
        base = reg.companion_base(comp)
        if not base:
            return
        headers = reg.auth_headers(comp) if hasattr(reg, "auth_headers") else {}
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{base}/ollama/api/tags", headers=headers)
            if r.status_code != 200:
                return
            installed = [m.get("name", "") for m in (r.json() or {}).get("models", [])]
        now = time.monotonic()
        for model in _wanted_models():
            if any(_names_match(i, model) for i in installed):
                continue
            # Explicit membership check: time.monotonic() starts near ZERO
            # after boot, so a `get(model, 0.0)` default would read as
            # "attempted just now" and silently suppress every pull for the
            # first THROTTLE_S of uptime.
            if model in _last_attempt and now - _last_attempt[model] < THROTTLE_S:
                continue
            _last_attempt[model] = now
            logger.info(
                "[%s] Companion is missing %s — starting a background pull so "
                "vision/editorial stages can run on the big GPU", job_id, model)
            # Long-lived pull; its own task so models download in parallel
            # and a slow pull never blocks anything.
            t = asyncio.create_task(_pull(base, headers, model, job_id))
            _tasks.add(t)
            t.add_done_callback(_tasks.discard)
    except Exception as e:
        logger.debug("[%s] companion model placement skipped: %s", job_id, e)


async def _pull(base: str, headers: dict, model: str, job_id: str) -> None:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3600.0) as client:
            r = await client.post(f"{base}/ollama/api/pull", headers=headers,
                                  json={"model": model, "stream": False})
            if r.status_code == 200:
                logger.info("[%s] Companion pull complete: %s", job_id, model)
            else:
                logger.info("[%s] Companion pull of %s answered HTTP %s",
                            job_id, model, r.status_code)
    except Exception as e:
        logger.debug("[%s] companion pull of %s failed (%s) — the registry "
                     "keeps routing that model locally", job_id, model, e)


def ensure_companion_models(job_id: str) -> None:
    """Fire-and-forget entry point (call at job start)."""
    if not bool(getattr(settings, "COMPANION_AUTOPULL_MODELS", True)):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        t = loop.create_task(_ensure(job_id))
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)
    except Exception:
        pass
