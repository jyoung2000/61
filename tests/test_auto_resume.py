"""Tests for auto-resume of interrupted jobs after a container restart.

When the container goes down mid-analysis, pipeline worker threads die. On the
next startup, jobs that have results are completed and result-less in-progress
jobs are RE-QUEUED to resume (reusing the engine checkpoint + extraction cache)
instead of being marked FAILED — capped so a crash-looping job can't relaunch
itself forever.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import backend.database as database
import backend.main as main


def _run(coro):
    return asyncio.run(coro)


def _job(job_id, status, *, clips=None, summary=None, resume_attempts=0, updated_at=""):
    return SimpleNamespace(
        job_id=job_id, status=status, clips=clips or [], summary=summary,
        resume_attempts=resume_attempts, updated_at=updated_at,
    )


def _install(monkeypatch, jobs, *, cap=3):
    updates = []
    scheduled = []

    async def _list_jobs(**k):
        return list(jobs)

    async def _update(job_id, **kw):
        updates.append((job_id, kw))
        return None

    monkeypatch.setattr(database, "list_jobs", _list_jobs)
    monkeypatch.setattr(database, "update_job_status", _update)
    monkeypatch.setattr(main, "_schedule_resume", lambda jid: scheduled.append(jid))
    monkeypatch.setattr(main, "_MAX_AUTO_RESUME_ATTEMPTS", cap)
    return updates, scheduled


# ── _auto_resume_interrupted_jobs ────────────────────────────────────


def test_resumes_resultless_job_under_cap(monkeypatch):
    updates, scheduled = _install(monkeypatch, [
        _job("a", "transcribing", clips=[], resume_attempts=0)])
    resumed, failed = _run(main._auto_resume_interrupted_jobs())
    assert (resumed, failed) == (1, 0)
    assert scheduled == ["a"]
    # Re-queued (status reset) with the attempt counter bumped + persisted.
    (jid, kw) = updates[0]
    assert jid == "a"
    assert kw["status"] == "queued"
    assert kw["resume_attempts"] == 1


def test_fails_resultless_job_at_cap(monkeypatch):
    updates, scheduled = _install(monkeypatch, [
        _job("b", "translating", clips=[], resume_attempts=3)], cap=3)
    resumed, failed = _run(main._auto_resume_interrupted_jobs())
    assert (resumed, failed) == (0, 1)
    assert scheduled == []  # never re-launched past the cap
    assert updates[0][1]["status"] == "failed"


def test_skips_jobs_with_results(monkeypatch):
    # Has clips → completed by recover_orphaned_jobs' first pass, not here.
    updates, scheduled = _install(monkeypatch, [
        _job("c", "detecting_clips", clips=[{"id": 1}])])
    resumed, failed = _run(main._auto_resume_interrupted_jobs())
    assert (resumed, failed) == (0, 0)
    assert updates == []
    assert scheduled == []


def test_skips_terminal_jobs(monkeypatch):
    updates, scheduled = _install(monkeypatch, [
        _job("d", "complete", clips=[]),
        _job("e", "failed", clips=[]),
    ])
    resumed, failed = _run(main._auto_resume_interrupted_jobs())
    assert (resumed, failed) == (0, 0)
    assert updates == []
    assert scheduled == []


def test_cap_zero_never_resumes(monkeypatch):
    # CLIPAI_MAX_RESUME_ATTEMPTS=0 → resume disabled, interrupted jobs fail.
    updates, scheduled = _install(monkeypatch, [
        _job("f", "analyzing_scenes", clips=[], resume_attempts=0)], cap=0)
    resumed, failed = _run(main._auto_resume_interrupted_jobs())
    assert (resumed, failed) == (0, 1)
    assert scheduled == []


# ── _recover_stale_jobs None-branch (used by the startup passes) ──────


def test_recover_none_fail_branch_never_fails(monkeypatch):
    # fail_stale_s=None → result-less job with no timestamp is left alone
    # (age is None would otherwise read as "stale" and fail it).
    updates, _ = _install(monkeypatch, [_job("g", "transcribing", clips=[])])
    completed, failed = _run(main._recover_stale_jobs(
        complete_stale_s=0, fail_stale_s=None))
    assert (completed, failed) == (0, 0)
    assert updates == []


def test_recover_none_complete_branch_only_fails(monkeypatch):
    updates, _ = _install(monkeypatch, [_job("h", "translating", clips=[])])
    completed, failed = _run(main._recover_stale_jobs(
        complete_stale_s=None, fail_stale_s=0))
    assert (completed, failed) == (0, 1)
    assert updates[0][1]["status"] == "failed"


# ── recover_orphaned_jobs end-to-end ─────────────────────────────────


def test_startup_completes_and_resumes(monkeypatch):
    monkeypatch.setattr(main, "_auto_resume_enabled", lambda: True)
    updates, scheduled = _install(monkeypatch, [
        _job("wr", "detecting_clips", clips=[{"id": 1}]),   # has results → COMPLETE
        _job("rl", "transcribing", clips=[], resume_attempts=0),  # → RESUME
    ])
    _run(main.recover_orphaned_jobs())
    by_id = {jid: kw for jid, kw in updates}
    assert by_id["wr"]["status"] == "complete"
    assert by_id["rl"]["status"] == "queued"
    assert by_id["rl"]["resume_attempts"] == 1
    assert scheduled == ["rl"]


def test_startup_with_auto_resume_disabled_fails_resultless(monkeypatch):
    monkeypatch.setattr(main, "_auto_resume_enabled", lambda: False)
    updates, scheduled = _install(monkeypatch, [
        _job("wr", "detecting_clips", clips=[{"id": 1}]),   # has results → COMPLETE
        _job("rl", "transcribing", clips=[]),               # → FAILED (old behavior)
    ])
    _run(main.recover_orphaned_jobs())
    by_id = {jid: kw for jid, kw in updates}
    assert by_id["wr"]["status"] == "complete"
    assert by_id["rl"]["status"] == "failed"
    assert scheduled == []


# ── Deferred resume drainer ──────────────────────────────────────────


def test_resume_drainer_runs_jobs_sequentially_after_delay(monkeypatch):
    """Resumes are deferred + drained one at a time so each runs in a settled,
    fully-warmed environment (not racing startup warmup)."""
    import backend.services.pipeline as pl

    calls = []

    async def fake_run(jid):
        calls.append(jid)

    monkeypatch.setattr(pl, "run_analysis", fake_run)
    monkeypatch.setenv("CLIPAI_RESUME_DELAY_S", "0")  # no grace delay in the test

    async def go():
        main._resume_queue.clear()
        main._resume_drainer = None
        main._schedule_resume("j1")
        main._schedule_resume("j2")  # shares the one drainer, not a 2nd
        for _ in range(50):
            await asyncio.sleep(0.01)
            if main._resume_drainer and main._resume_drainer.done():
                break

    asyncio.run(go())
    assert calls == ["j1", "j2"]  # ran sequentially, in order


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
