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


# ── _persist_complete_job: direct write survives + forces job_id ──────────

def test_persist_complete_job_writes_and_verifies(tmp_path, monkeypatch):
    """The robust finalizer must persist status=COMPLETE + clips to the
    canonical path even though the DB-merge path was unreliable in prod."""
    import os
    import backend.database as db
    from backend.models import JobResult, JobStatus as _JS, ClipCandidate

    d = str(tmp_path)
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))

    # Seed a detecting_clips job on disk (what finalization starts from).
    seed = JobResult(job_id="J", filename="v.mp4", file_path="p",
                     status=_JS.DETECTING_CLIPS.value)
    _aiorun(db.save_job(seed))

    clip = ClipCandidate(
        id=1, title="t", start_time=0, end_time=5, duration=5,
        viral_score=80, viral_score_reasoning="r", clip_type="x",
        platform="both", suggested_caption="c", hook_text="h",
        why_this_works="w")
    fields = dict(status=_JS.COMPLETE, progress=100, clips=[clip])

    ok = _aiorun(pipeline._persist_complete_job("J", fields))
    assert ok is True
    reloaded = _aiorun(db.load_job("J"))
    assert str(reloaded.status) == str(_JS.COMPLETE)
    assert len(reloaded.clips) == 1


def test_persist_complete_job_forces_job_id(tmp_path, monkeypatch):
    """If the stored job_id is wrong (the suspected prod cause of saves
    landing on the wrong path), the finalizer forces it and still persists."""
    import os
    import backend.database as db
    from backend.models import JobResult, JobStatus as _JS

    d = str(tmp_path)
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    # Stored job carries a DIFFERENT job_id than its directory key.
    seed = JobResult(job_id="WRONG", filename="v.mp4", file_path="p",
                     status=_JS.DETECTING_CLIPS.value)
    os.makedirs(os.path.join(d, "K"), exist_ok=True)
    import json
    with open(os.path.join(d, "K", "job.json"), "w") as f:
        json.dump(seed.model_dump(mode="json"), f)

    ok = _aiorun(pipeline._persist_complete_job("K", dict(status=_JS.COMPLETE, progress=100)))
    assert ok is True
    reloaded = _aiorun(db.load_job("K"))
    assert str(reloaded.status) == str(_JS.COMPLETE)
    assert reloaded.job_id == "K"   # forced to the directory key


def test_persist_complete_job_survives_concurrent_relay(tmp_path, monkeypatch):
    """The finalizer must not lose COMPLETE to a clipper progress relay that
    runs concurrently. Reproduces the prod race: a relay's locked
    read-modify-write (load pre-COMPLETE snapshot → save detecting_clips)
    interleaving with the finalize write. Because both now take the same
    per-job lock, the terminal COMPLETE state always wins."""
    import os
    import backend.database as db
    from backend.models import JobResult, JobStatus as _JS, ClipCandidate

    d = str(tmp_path)
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    _aiorun(db.save_job(JobResult(
        job_id="R", filename="v.mp4", file_path="p",
        status=_JS.DETECTING_CLIPS.value, progress=80)))

    clip = ClipCandidate(
        id=1, title="t", start_time=0, end_time=5, duration=5,
        viral_score=80, viral_score_reasoning="r", clip_type="x",
        platform="both", suggested_caption="c", hook_text="h",
        why_this_works="w")
    fields = dict(status=_JS.COMPLETE, progress=100, clips=[clip])

    async def _race():
        # A stale relay write (detecting_clips, no clips) racing the finalize.
        # protect_terminal mirrors the real ``_update_progress`` relay.
        relay = db.update_job_status(
            "R", status=_JS.DETECTING_CLIPS, progress=94,
            protect_terminal=True)
        persist = pipeline._persist_complete_job("R", fields)
        await asyncio.gather(relay, persist)

    _aiorun(_race())
    reloaded = _aiorun(db.load_job("R"))
    assert str(reloaded.status) == str(_JS.COMPLETE), (
        "concurrent relay reverted the finished job")
    assert len(reloaded.clips) == 1


# ── WS replays current status on connect (recovers a missed COMPLETE) ──────

class _FakeWS:
    def __init__(self):
        self.sent = []
        self._first = True

    async def accept(self):
        pass

    async def send_json(self, payload):
        self.sent.append(payload)

    async def receive_text(self):
        # Disconnect right after the on-connect replay so the endpoint exits.
        from fastapi import WebSocketDisconnect
        raise WebSocketDisconnect()


def test_ws_replays_complete_status_on_connect(tmp_path, monkeypatch):
    import os
    import backend.database as db
    from backend.models import JobResult, JobStatus as _JS
    from backend.routers import ws as ws_router

    d = str(tmp_path)
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    _aiorun(db.save_job(JobResult(
        job_id="C", filename="v.mp4", file_path="p",
        status=_JS.COMPLETE.value, progress=100)))

    fake = _FakeWS()
    _aiorun(ws_router.websocket_job_progress(fake, "C"))
    # The very first frame must announce completion so a reconnecting client
    # that missed the live broadcast flips out of the in-progress overlay.
    assert fake.sent, "no status frame replayed on connect"
    assert fake.sent[0]["type"] == "complete"
    assert fake.sent[0]["status"] == "complete"


def test_ws_replays_inprogress_status_on_connect(tmp_path, monkeypatch):
    import os
    import backend.database as db
    from backend.models import JobResult, JobStatus as _JS
    from backend.routers import ws as ws_router

    d = str(tmp_path)
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    _aiorun(db.save_job(JobResult(
        job_id="P", filename="v.mp4", file_path="p",
        status=_JS.DETECTING_CLIPS.value, progress=80)))

    fake = _FakeWS()
    _aiorun(ws_router.websocket_job_progress(fake, "P"))
    assert fake.sent[0]["type"] == "status"
    assert fake.sent[0]["status"] == "detecting_clips"
