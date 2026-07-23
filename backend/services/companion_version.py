"""Companion app version handshake.

The Companion is a desktop app the user installs on their GPU PC — the
container can't update it, and an old install fails SILENTLY: its proxy
just 404s the routes it doesn't have (the observed run probed
``/v1/vision/health`` every minute, got 404 all job, and face detection
quietly stayed local). This module makes staleness LOUD instead:

  * ``EXPECTED_COMPANION_VERSION`` is the Companion version this container
    ships with (a repo-sync test pins it to ``companion/src-tauri/
    Cargo.toml``, ``tauri.conf.json``, ``package.json`` and the ``RELEASE``
    tag, so a bump can't drift).
  * ``check_companion_version(job_id)`` runs fire-and-forget at job start:
    it reads the paired Companion's ``/v1/health`` ``version`` and, when
    older than expected, records a pipeline warning + broadcasts the same
    ``compute_warning`` event the CPU-fallback warning uses — so it shows
    in the Processing Log with no frontend changes, and names the fix
    (Settings → GPU Companion → download the updated installer).

Fail-soft everywhere: no Companion, an unreachable one, or a parse error
never touches the job.
"""
from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger("clipai.companion_version")

# The Companion version this container was built alongside. Bump together
# with companion/RELEASE (test_companion_version_sync enforces it).
EXPECTED_COMPANION_VERSION = "0.8.0"

_tasks: set = set()
# Warn once per (reported_version) per process — every job start re-checks,
# but an unchanged stale version shouldn't re-log server-side each time.
_warned_versions: set = set()


def parse_version(s) -> tuple:
    """Tolerant semver triple: '0.2.0' / 'v0.2' / 'companion-v0.2.0' → (0,2,0).
    Unparseable / empty → (0, 0, 0)."""
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(s or ""))
    if not m:
        return (0, 0, 0)
    return tuple(int(g or 0) for g in m.groups())


def is_outdated(reported, expected: str = "") -> bool:
    """True when the Companion's reported version predates ``expected``.

    A blank/missing version is treated as outdated — every Companion build
    has reported ``version`` in /v1/health since the first release, so its
    absence means something older (or foreign) is answering."""
    exp = parse_version(expected or EXPECTED_COMPANION_VERSION)
    rep = parse_version(reported)
    if rep == (0, 0, 0):
        return True
    return rep < exp


def outdated_message(reported) -> str:
    rep = str(reported or "unknown")
    return (
        f"GPU Companion app v{rep} is older than this ClipAI build expects "
        f"(v{EXPECTED_COMPANION_VERSION}). Offloads added since then — vision "
        "offload, Whisper pre-warm, speed-profile sync — are unavailable, so "
        "jobs run slower than they should. Update it from Settings → GPU "
        "Companion (Download for Windows/macOS) and reinstall on the GPU PC."
    )


async def _check(job_id: str) -> None:
    try:
        import httpx
        from backend.services import ollama_registry as reg
        comp = reg.companion_host()
        if comp is None:
            return
        base = reg.companion_base(comp)
        if not base:
            return
        headers = reg.auth_headers(comp)
        # Tell the Companion what this container expects; a new-enough build
        # echoes update_available so ITS UI can show the hint too.
        headers = dict(headers or {})
        headers["X-ClipAI-Expected-Companion"] = EXPECTED_COMPANION_VERSION
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base}/v1/health", headers=headers)
        if r.status_code != 200:
            return
        data = r.json() or {}
        if data.get("service") != "clipai-gpu-companion":
            return
        ver = (data.get("version") or "").strip()
        if not is_outdated(ver):
            return
        msg = outdated_message(ver)
        if ver not in _warned_versions:
            _warned_versions.add(ver)
            logger.warning("[%s] %s", job_id, msg)
        try:
            from backend.services.pipeline_helpers import _record_pipeline_warning
            _record_pipeline_warning(job_id, msg)
        except Exception:
            pass
        try:
            from backend.services.pipeline import broadcast_ws
            await broadcast_ws(job_id, {
                "type": "compute_warning",
                "level": "warning",
                "message": msg,
            })
        except Exception:
            pass
    except Exception as e:
        logger.debug("[%s] companion version check skipped: %s", job_id, e)


def check_companion_version(job_id: str) -> None:
    """Fire-and-forget entry point (call at job start)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    try:
        t = loop.create_task(_check(job_id))
        _tasks.add(t)
        t.add_done_callback(_tasks.discard)
    except Exception:
        pass
