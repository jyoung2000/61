"""The COMPLETE-finalization gate must stop stale progress relays from
reverting a finished job back to ``detecting_clips``.

Production symptom: the UI hung at 94 % "Asking the VLM…" / "Generating
summary…" because a queued clipper progress relay committed a detecting_clips
write just after the COMPLETE save, and ``protect_terminal`` (which checks the
status at the relay's load time) couldn't catch it. The fix is a synchronous
finalization gate: once a job enters finalization, _update_progress drops every
non-terminal write regardless of lock/loop scheduling.

pipeline pulls the heavy provider/cv2/torch chain, so we stub ONLY the modules
that are actually missing (never shadowing installed packages → no sibling
test pollution).
"""

import asyncio
import importlib.util
import sys
import types

import pytest


class _AnyModule(types.ModuleType):
    __path__: list = []

    def __getattr__(self, name):
        return type(name, (), {})


def _stub_if_missing(name):
    if name in sys.modules:
        return
    base = name.split(".")[0]
    try:
        if importlib.util.find_spec(base) is not None:
            return
    except Exception:
        pass
    sys.modules[name] = _AnyModule(name)


for _n in ["google", "google.generativeai", "groq", "replicate", "cv2", "torch",
           "faster_whisper", "librosa", "soundfile", "sentencepiece",
           "ctranslate2", "transformers"]:
    _stub_if_missing(_n)
if isinstance(sys.modules.get("google"), _AnyModule):
    sys.modules["google"].generativeai = sys.modules.get(
        "google.generativeai", _AnyModule("google.generativeai"))

from backend.services import pipeline  # noqa: E402
from backend.models import JobStatus  # noqa: E402


def _aiorun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _patch_io(monkeypatch):
    """Record DB writes; silence WS + cancel checks."""
    calls = []

    async def _fake_update(job_id, **kwargs):
        calls.append(kwargs)
        return None

    async def _fake_broadcast(job_id, msg):
        return None

    monkeypatch.setattr(pipeline.database, "update_job_status", _fake_update)
    monkeypatch.setattr(pipeline, "broadcast_ws", _fake_broadcast)
    monkeypatch.setattr(pipeline, "is_cancel_requested", lambda jid: False)
    pipeline._finalizing_jobs.discard("jobX")
    yield calls
    pipeline._finalizing_jobs.discard("jobX")


def test_progress_blocked_during_finalization(_patch_io):
    pipeline._finalizing_jobs.add("jobX")
    # A stale clipper relay tries to write detecting_clips after finalization.
    _aiorun(pipeline._update_progress(
        "jobX", JobStatus.DETECTING_CLIPS, 94, "Asking the VLM..."))
    assert _patch_io == [], "stale non-terminal progress write was NOT blocked"


def test_terminal_write_allowed_during_finalization(_patch_io):
    pipeline._finalizing_jobs.add("jobX")
    # A terminal status must still get through (e.g. a late FAILED).
    _aiorun(pipeline._update_progress(
        "jobX", JobStatus.FAILED, 100, "Failed"))
    assert len(_patch_io) == 1
    assert _patch_io[0].get("status") == JobStatus.FAILED


def test_progress_allowed_when_not_finalizing(_patch_io):
    _aiorun(pipeline._update_progress(
        "jobX", JobStatus.DETECTING_CLIPS, 80, "Detecting clips..."))
    assert len(_patch_io) == 1
