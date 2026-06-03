"""Tests for stuck-job recovery (the 'stuck at translating forever' fix).

A run that died at the translate stage left the job pinned at ``translating``,
because the recovery sets omitted that status. These cover the derived
non-terminal set + the staleness-guarded recovery that heals such jobs without
ever touching an in-flight run.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import backend.database as database
import backend.main as main


def _run(coro):
    return asyncio.run(coro)


def _iso(minutes_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _job(job_id, status, *, clips=None, summary=None, updated_at=""):
    return SimpleNamespace(job_id=job_id, status=status, clips=clips or [],
                           summary=summary, updated_at=updated_at)


def _install(monkeypatch, jobs):
    updates = []

    async def _list_jobs(**k):
        return list(jobs)

    async def _update(job_id, **kw):
        updates.append((job_id, kw))
        return None

    monkeypatch.setattr(database, "list_jobs", _list_jobs)
    monkeypatch.setattr(database, "update_job_status", _update)
    return updates


def test_non_terminal_set_covers_every_running_phase():
    # The regression was a hardcoded set missing these — derive-from-enum fixes it.
    for s in ("queued", "transcribing", "extracting_frames", "analyzing_scenes",
              "generating_summary", "translating", "detecting_clips"):
        assert s in main._NON_TERMINAL_STATUSES
    for s in ("complete", "failed", "cancelled"):
        assert s not in main._NON_TERMINAL_STATUSES


def test_is_stale_threshold():
    fresh = _job("j", "translating", updated_at=_iso(1))
    old = _job("j", "translating", updated_at=_iso(120))
    assert main._is_stale(fresh, 0) is True            # 0 → always stale (startup)
    assert main._is_stale(fresh, 1800) is False         # 1 min < 30 min → active
    assert main._is_stale(old, 1800) is True            # 120 min ≥ 30 min → dead
    assert main._is_stale(_job("j", "translating"), 1800) is True  # no timestamp → stale


def test_completes_stuck_translating_with_results(monkeypatch):
    updates = _install(monkeypatch, [
        _job("a", "translating", clips=[{"id": 1}], updated_at=_iso(200))])
    completed, failed = _run(main._recover_stale_jobs(
        complete_stale_s=1800, fail_stale_s=7200))
    assert (completed, failed) == (1, 0)
    assert updates[0][0] == "a"
    assert updates[0][1]["status"] == "complete"


def test_fails_stuck_translating_without_results(monkeypatch):
    updates = _install(monkeypatch, [
        _job("b", "translating", clips=[], summary=None, updated_at=_iso(200))])
    completed, failed = _run(main._recover_stale_jobs(
        complete_stale_s=1800, fail_stale_s=7200))
    assert (completed, failed) == (0, 1)
    assert updates[0][1]["status"] == "failed"


def test_never_touches_an_active_run(monkeypatch):
    # Recent updated_at → a live worker is advancing it → must be left alone.
    updates = _install(monkeypatch, [
        _job("c", "translating", clips=[{"id": 1}], updated_at=_iso(2))])
    completed, failed = _run(main._recover_stale_jobs(
        complete_stale_s=1800, fail_stale_s=7200))
    assert (completed, failed) == (0, 0)
    assert updates == []


def test_skips_terminal_jobs(monkeypatch):
    updates = _install(monkeypatch, [
        _job("d", "complete", clips=[{"id": 1}], updated_at=_iso(999))])
    completed, failed = _run(main._recover_stale_jobs(
        complete_stale_s=0, fail_stale_s=0))
    assert (completed, failed) == (0, 0)
    assert updates == []


def test_startup_recovers_regardless_of_age(monkeypatch):
    # stale_s=0 (startup): every worker is already dead, so recover even a
    # just-stamped job.
    updates = _install(monkeypatch, [
        _job("e", "translating", clips=[{"id": 1}], updated_at=_iso(0))])
    completed, failed = _run(main._recover_stale_jobs(
        complete_stale_s=0, fail_stale_s=0))
    assert (completed, failed) == (1, 0)
    assert updates[0][1]["status"] == "complete"
