"""The pipeline stall watchdog: a run that emits NO real progress for the
stall window is presumed wedged (a hung native call the cooperative cancel
can never reach), cancelled, and auto-resumed from its checkpoint — the case
the heartbeat-staleness revive can never catch, because the heartbeat keeps
stamping heartbeat_at while the run is wedged (observed: a bulk-import video
whose face loop went silent for 13+ minutes on a corrupt file while the job
looked alive)."""

import asyncio
from types import SimpleNamespace

import pytest

import backend.services.pipeline as P


class _FakeTask:
    def __init__(self):
        self.cancel_calls = 0
    def done(self):
        return False
    def cancel(self):
        self.cancel_calls += 1


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    P._run_tasks.clear()
    P._stall_revive_pending.clear()
    P._stall_fail_pending.clear()
    monkeypatch.delenv("CLIPAI_AUTO_RESUME", raising=False)
    yield
    P._run_tasks.clear()
    P._stall_revive_pending.clear()
    P._stall_fail_pending.clear()


# ── Heartbeat → handler wiring ──────────────────────────────────────────────

def test_heartbeat_fires_stall_handler_once_with_stage(monkeypatch):
    calls = []

    async def fake_handler(job_id, stage, silent_s):
        calls.append((job_id, stage, silent_s))
    monkeypatch.setattr(P, "_handle_stalled_run", fake_handler)

    async def main():
        hb = P._PipelineHeartbeat("j1", interval=999, persist_interval=999,
                                  stall_after_s=0.12, tick=0.03)
        hb.touch("face detection")
        hb.start()
        await asyncio.sleep(0.45)
        hb.stop()
        await asyncio.sleep(0.05)
    asyncio.run(main())

    assert len(calls) == 1, "the watchdog must fire exactly once"
    job_id, stage, silent_s = calls[0]
    assert job_id == "j1" and stage == "face detection"
    assert silent_s >= 0.12


def test_heartbeat_touch_resets_the_stall_clock(monkeypatch):
    calls = []

    async def fake_handler(*a):
        calls.append(a)
    monkeypatch.setattr(P, "_handle_stalled_run", fake_handler)

    async def main():
        hb = P._PipelineHeartbeat("j1", interval=999, persist_interval=999,
                                  stall_after_s=0.25, tick=0.03)
        hb.start()
        for _ in range(8):                 # 0.4s total, touching every 0.05s
            await asyncio.sleep(0.05)
            hb.touch("transcription")
        hb.stop()
    asyncio.run(main())
    assert calls == [], "regular progress must keep the watchdog quiet"


def test_heartbeat_stall_disabled_with_zero(monkeypatch):
    calls = []

    async def fake_handler(*a):
        calls.append(a)
    monkeypatch.setattr(P, "_handle_stalled_run", fake_handler)

    async def main():
        hb = P._PipelineHeartbeat("j1", interval=999, persist_interval=999,
                                  stall_after_s=0, tick=0.03)
        hb.start()
        await asyncio.sleep(0.3)
        hb.stop()
    asyncio.run(main())
    assert calls == []


# ── The stall handler's decisions ───────────────────────────────────────────

def _wire_db(monkeypatch, attempts):
    job = SimpleNamespace(resume_attempts=attempts)
    writes = []

    async def load_job(jid):
        return job

    async def update(jid, **kw):
        writes.append((jid, kw))
    monkeypatch.setattr(P.database, "load_job", load_job)
    monkeypatch.setattr(P.database, "update_job_status", update)
    return writes


def test_stall_handler_requeues_for_resume_with_attempts_left(monkeypatch):
    task = _FakeTask()
    P._run_tasks["j1"] = task
    writes = _wire_db(monkeypatch, attempts=0)

    asyncio.run(P._handle_stalled_run("j1", "face detection", 3600))

    assert task.cancel_calls == 1
    assert "j1" in P._stall_revive_pending and "j1" not in P._stall_fail_pending
    jid, kw = writes[0]
    assert jid == "j1"
    assert kw["status"] == P.JobStatus.QUEUED
    assert kw["resume_attempts"] == 1
    assert "stalled" in kw["progress_message"].lower()
    assert "checkpoint" in kw["progress_message"].lower()


def test_stall_handler_fails_after_the_attempt_cap(monkeypatch):
    task = _FakeTask()
    P._run_tasks["j1"] = task
    writes = _wire_db(monkeypatch, attempts=P._MAX_STALL_RESUME_ATTEMPTS)

    asyncio.run(P._handle_stalled_run("j1", "translation", 7200))

    assert task.cancel_calls == 1
    assert "j1" in P._stall_fail_pending and "j1" not in P._stall_revive_pending
    jid, kw = writes[0]
    assert kw["status"] == P.JobStatus.FAILED
    assert "stalled" in kw["progress_message"].lower()


def test_stall_handler_respects_auto_resume_disabled(monkeypatch):
    monkeypatch.setenv("CLIPAI_AUTO_RESUME", "0")
    task = _FakeTask()
    P._run_tasks["j1"] = task
    writes = _wire_db(monkeypatch, attempts=0)

    asyncio.run(P._handle_stalled_run("j1", "clips", 3600))

    assert "j1" in P._stall_fail_pending
    assert writes[0][1]["status"] == P.JobStatus.FAILED


def test_stall_handler_noop_without_a_live_task(monkeypatch):
    writes = _wire_db(monkeypatch, attempts=0)
    asyncio.run(P._handle_stalled_run("ghost", "x", 3600))
    assert writes == []
    assert not P._stall_revive_pending and not P._stall_fail_pending


# ── User abort ("Cancel import" must stop a run NOW) ────────────────────────

def test_abort_analysis_cancels_the_task_and_flags_a_user_abort():
    """The cooperative flag alone only lands at the next checkpoint — a run
    inside a long stage would keep burning GPU for minutes after Cancel."""
    P._user_abort_pending.clear()
    task = _FakeTask()
    P._run_tasks["job-1"] = task

    assert P.abort_analysis("job-1") is True
    assert task.cancel_calls == 1, "the task itself must be cancelled"
    assert P.is_cancel_requested("job-1"), "cooperative flag is set too"
    # Flagged so the CancelledError handler records CANCELLED instead of
    # re-raising (which would strand the job on PROCESSING).
    assert "job-1" in P._user_abort_pending
    P._user_abort_pending.clear()
    P._cancel_events.pop("job-1", None)


def test_abort_analysis_on_a_job_with_no_live_run_is_false():
    P._user_abort_pending.clear()
    assert P.abort_analysis("ghost") is False
    assert "ghost" not in P._user_abort_pending
    P._cancel_events.pop("ghost", None)


def test_abort_analysis_ignores_an_already_finished_task():
    class _Done(_FakeTask):
        def done(self):
            return True
    P._user_abort_pending.clear()
    t = _Done()
    P._run_tasks["job-2"] = t
    assert P.abort_analysis("job-2") is False
    assert t.cancel_calls == 0
    P._cancel_events.pop("job-2", None)
