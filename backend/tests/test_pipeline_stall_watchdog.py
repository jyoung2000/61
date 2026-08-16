"""The pipeline stall watchdog: a run that emits NO real progress for the
stall window is presumed wedged (a hung native call the cooperative cancel
can never reach), cancelled, and auto-resumed from its checkpoint — the case
the heartbeat-staleness revive can never catch, because the heartbeat keeps
stamping heartbeat_at while the run is wedged (observed: a bulk-import video
whose face loop went silent for 13+ minutes on a corrupt file while the job
looked alive)."""

import asyncio
import time
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


def test_abort_analysis_clears_lingering_stall_flags():
    """A stall verdict whose task.cancel() never landed (the run finished at
    that exact moment) leaves its pending flag behind. Without clearing it, a
    later user cancel of the SAME job would hit the revive branch and
    resurrect the job the user just cancelled."""
    P._user_abort_pending.clear()
    task = _FakeTask()
    P._run_tasks["job-9"] = task
    P._stall_revive_pending.add("job-9")
    P._stall_fail_pending.add("job-9")

    assert P.abort_analysis("job-9") is True
    assert "job-9" not in P._stall_revive_pending, "user intent outranks a stall verdict"
    assert "job-9" not in P._stall_fail_pending
    assert "job-9" in P._user_abort_pending
    P._user_abort_pending.clear()
    P._cancel_events.pop("job-9", None)


# ── has_live_run: the revive guard's in-process ownership signal ────────────

def test_has_live_run_covers_the_gate_wait():
    """A run registers its cancel event BEFORE waiting on the concurrency
    gate and its heartbeat only once admitted — has_live_run must be True in
    BOTH windows, because the gate wait writes no heartbeat and would
    otherwise read as a dead job to the staleness-based revive (which then
    spawns a second concurrent run of the same job)."""
    assert not P.has_live_run("g1")
    # Window 1: queued behind the gate — only the cancel event exists.
    P._cancel_events["g1"] = asyncio.Event()
    assert P.has_live_run("g1")
    P._cancel_events.pop("g1")
    assert not P.has_live_run("g1")
    # Window 2: admitted — the heartbeat is registered.
    P._heartbeats["g1"] = object()
    assert P.has_live_run("g1")
    P._heartbeats.pop("g1")
    assert not P.has_live_run("g1")


def test_spawn_analysis_holds_a_strong_reference(monkeypatch):
    """asyncio only weakly references bare create_task results — an
    unreferenced analysis task can be garbage-collected mid-run. Every
    fire-and-forget spawn (stall revive, Companion import) must ride
    spawn_analysis, which anchors the task until it finishes."""
    ran = []

    async def stub(job_id, resume=False):
        ran.append((job_id, resume))
    monkeypatch.setattr(P, "run_analysis", stub)

    async def main():
        t = P.spawn_analysis("jz", resume=True)
        assert t in P._BACKGROUND_TASKS, "the task must be strongly referenced"
        await t
        await asyncio.sleep(0)          # let the done-callback run
        assert t not in P._BACKGROUND_TASKS, "and released once done"
    asyncio.run(main())
    assert ran == [("jz", True)]


# ── End-to-end: user abort vs a lingering stall-revive flag ─────────────────

def test_user_abort_outranks_a_lingering_revive_flag(monkeypatch):
    """Full run_analysis pass with the inner pipeline hung: the user's hard
    cancel must land the job on CANCELLED — never on the revive branch, even
    when a stale _stall_revive_pending flag is present — and the finally
    cleanup must leave no registry entries or pending flags behind."""
    from types import SimpleNamespace as NS

    writes, spawned = [], []

    async def upd(jid, **kw):
        writes.append((jid, kw))

    async def load(jid):
        return NS(owner_user_id="", filename="v.mp4")

    async def noop(*a, **kw):
        pass

    old_gate = P._analysis_semaphore
    P._analysis_semaphore = None
    monkeypatch.setattr(P.database, "update_job_status", upd)
    monkeypatch.setattr(P.database, "load_job", load)
    monkeypatch.setattr(P, "_update_progress", noop)
    monkeypatch.setattr(P, "broadcast_ws", noop)
    monkeypatch.setattr(P, "_warm_seo_intelligence", lambda jid: None)
    monkeypatch.setattr(P, "spawn_analysis",
                        lambda jid, resume=False: spawned.append((jid, resume)))
    import backend.services.companion_models as CM
    import backend.services.companion_version as CV
    import backend.services.companion_progress as CP
    monkeypatch.setattr(CM, "ensure_companion_models", lambda jid: None)
    monkeypatch.setattr(CV, "check_companion_version", lambda jid: None)
    monkeypatch.setattr(CP, "job_ended", lambda jid: None)

    async def main():
        started = asyncio.Event()

        async def hang_inner(job_id, resume=False):
            started.set()
            await asyncio.Event().wait()        # wedged forever
        monkeypatch.setattr(P, "_run_analysis_inner", hang_inner)

        task = asyncio.create_task(P.run_analysis("jx"))
        await asyncio.wait_for(started.wait(), 5)
        assert P.has_live_run("jx")
        # The lingering flag from a stall verdict whose cancel never landed:
        P._stall_revive_pending.add("jx")
        assert P.abort_analysis("jx") is True
        await asyncio.wait_for(task, 5)
    try:
        asyncio.run(main())
    finally:
        P._analysis_semaphore = old_gate

    final_status = [kw for _, kw in writes if "status" in kw][-1]
    assert final_status["status"] == P.JobStatus.CANCELLED
    assert spawned == [], "a user cancel must never be turned into a resume"
    assert "jx" not in P._stall_revive_pending
    assert "jx" not in P._user_abort_pending
    assert not P.has_live_run("jx"), "finally-cleanup must clear every registry"


# ── Long remote transcription must not read as a stalled run ────────────────

def _ra():
    """reframer_audio, skipped where its native deps aren't installed."""
    pytest.importorskip("cv2")
    import backend.services.reframer_audio as RA
    return RA


def test_remote_whisper_keepalive_touches_the_stall_clock(monkeypatch):
    """The single transcription POST is the longest step in the pipeline and
    emits no progress of its own: full large-v3 at beam 5 decodes at ~0.5-0.7×
    realtime, so 150 min of audio sits inside ONE call for 75-105 minutes.
    With PIPELINE_STALL_MINUTES=60 and no keepalive the watchdog declared the
    run wedged, cancelled it, and resumed from checkpoint — restarting the
    same decode, stalling again, and burning every resume attempt."""
    RA = _ra()
    from backend.services import request_context as RC

    touches = []
    monkeypatch.setattr(P, "set_heartbeat_stage",
                        lambda jid, stage: touches.append((jid, stage)))
    monkeypatch.setattr(RA.RemoteWhisperClient, "_KEEPALIVE_TICK_S", 0.02)
    RC.set_job("job-x", "video.mp4")
    try:
        payload = 150 * 60 * 32000          # 150 min of 16 kHz mono s16 PCM
        stop = RA.RemoteWhisperClient._start_transcribe_keepalive(payload)
        deadline = time.monotonic() + 3.0
        while not touches and time.monotonic() < deadline:
            time.sleep(0.02)
        stop()
    finally:
        RC.clear()

    assert touches, "the stall clock must be refreshed while the request runs"
    job_id, stage = touches[0]
    assert job_id == "job-x"
    # The label names the REAL work, so the UI can never freeze on the
    # PREVIOUS stage's text ("Releasing face detection models…") for an hour.
    assert "transcribing on the Companion GPU" in stage
    assert "150 min audio" in stage


def test_keepalive_is_a_noop_without_a_job_context(monkeypatch):
    """Outside a pipeline run (ad-hoc transcription, tests) there is no job to
    keep alive — no thread, no touches, no raise."""
    RA = _ra()
    from backend.services import request_context as RC

    touches = []
    monkeypatch.setattr(P, "set_heartbeat_stage",
                        lambda jid, stage: touches.append(jid))
    monkeypatch.setattr(RA.RemoteWhisperClient, "_KEEPALIVE_TICK_S", 0.02)
    RC.clear()
    stop = RA.RemoteWhisperClient._start_transcribe_keepalive(32000 * 60)
    time.sleep(0.1)
    stop()
    assert touches == []
