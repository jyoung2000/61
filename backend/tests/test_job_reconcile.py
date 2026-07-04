"""Startup/periodic reconcile: a non-terminal job that actually has results
(clips or summary) must be flipped to COMPLETE, not left "stuck judging clips"
or marked failed.

The reconcile lives in backend.main, which pulls uvicorn/fastapi-heavy imports
not present in the test image. The reconcile's CONTRACT is pure: scan jobs,
and for any in-progress job with clips/summary, update_job_status(complete).
We exercise that contract directly against the database layer (the real
dependency), mirroring backend.main._reconcile_finished_jobs.
"""

import asyncio
import os
import tempfile

import pytest

import backend.database as db
from backend.models import JobResult, JobStatus, ClipCandidate, VideoSummary


def _aiorun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture()
def tmp_job_store(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    return d


_IN_PROGRESS = {
    "analyzing_scenes", "extracting_frames", "generating_summary", "detecting_clips",
}


async def _reconcile_finished_jobs() -> int:
    """Copy of backend.main._reconcile_finished_jobs (kept in sync)."""
    fixed = 0
    for job in await db.list_jobs(include_unowned=True):
        if job.status not in _IN_PROGRESS:
            continue
        if bool(getattr(job, "clips", None)) or getattr(job, "summary", None) is not None:
            await db.update_job_status(
                job.job_id, status="complete", progress=100,
                progress_message="Analysis complete",
            )
            fixed += 1
    return fixed


def _clip():
    return ClipCandidate(
        id=1, title="t", start_time=0, end_time=5, duration=5,
        viral_score=80, viral_score_reasoning="r", clip_type="x",
        platform="both", suggested_caption="c", hook_text="h", why_this_works="w")


def _summary():
    return VideoSummary(overview="o", key_topics=["a"], tone="t",
                        estimated_audience="e", content_category="c")


def test_stuck_judging_job_with_clips_marked_complete(tmp_job_store):
    async def scenario():
        # Mirrors the screenshot: status detecting_clips ("Judging clip
        # candidates…") but the clips are already on disk.
        await db.save_job(JobResult(
            job_id="J", filename="v.mp4", file_path="p",
            status=JobStatus.DETECTING_CLIPS.value, progress=89,
            clips=[_clip().model_dump()]))
        fixed = await _reconcile_finished_jobs()
        return fixed, await db.load_job("J")

    fixed, job = _aiorun(scenario())
    assert fixed == 1
    assert str(job.status) == str(JobStatus.COMPLETE)
    assert job.progress == 100


def test_inprogress_job_with_summary_only_marked_complete(tmp_job_store):
    async def scenario():
        await db.save_job(JobResult(
            job_id="S", filename="v.mp4", file_path="p",
            status=JobStatus.GENERATING_SUMMARY.value, summary=_summary().model_dump()))
        await _reconcile_finished_jobs()
        return await db.load_job("S")

    job = _aiorun(scenario())
    assert str(job.status) == str(JobStatus.COMPLETE)


def test_inprogress_job_without_results_left_untouched(tmp_job_store):
    async def scenario():
        await db.save_job(JobResult(
            job_id="E", filename="v.mp4", file_path="p",
            status=JobStatus.EXTRACTING_FRAMES.value))
        fixed = await _reconcile_finished_jobs()
        return fixed, await db.load_job("E")

    fixed, job = _aiorun(scenario())
    assert fixed == 0
    # Reconcile doesn't touch it (the fail-path handles true orphans).
    assert str(job.status) == str(JobStatus.EXTRACTING_FRAMES)


def test_complete_job_not_reprocessed(tmp_job_store):
    async def scenario():
        await db.save_job(JobResult(
            job_id="C", filename="v.mp4", file_path="p",
            status=JobStatus.COMPLETE.value, clips=[_clip().model_dump()]))
        return await _reconcile_finished_jobs()

    assert _aiorun(scenario()) == 0


# ── Liveness math (heartbeat-aware staleness, fix 5) ─────────────────────

def test_job_age_uses_freshest_of_updated_and_heartbeat():
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from backend.services.job_liveness import job_age_seconds, is_stale

    now = datetime.now(timezone.utc)
    old = (now - timedelta(minutes=45)).isoformat()
    fresh = (now - timedelta(seconds=30)).isoformat()

    # Long silent stage: no progress writes for 45 min, but the heartbeat
    # stamped 30s ago — the job is ALIVE and must not look stale.
    j = SimpleNamespace(updated_at=old, heartbeat_at=fresh)
    age = job_age_seconds(j)
    assert age is not None and age < 60
    assert is_stale(j, 600) is False

    # Dead run: both signals old.
    dead = SimpleNamespace(updated_at=old, heartbeat_at=old)
    assert is_stale(dead, 600) is True

    # Legacy job without the heartbeat field at all.
    legacy = SimpleNamespace(updated_at=fresh)
    assert is_stale(legacy, 600) is False

    # No signals at all → treated as stale (recoverable), never crashes.
    blank = SimpleNamespace(updated_at="", heartbeat_at="")
    assert job_age_seconds(blank) is None
    assert is_stale(blank, 600) is True

    # stale_after_s <= 0 = no staleness requirement (startup semantics).
    assert is_stale(j, 0) is True


def test_heartbeat_at_field_persists(tmp_job_store):
    """heartbeat_at travels through update_job_status kwargs like any field."""
    async def run():
        await db.save_job(JobResult(
            job_id="hb1", filename="v.mp4", file_path="/tmp/v.mp4", status=JobStatus.TRANSCRIBING,
            progress=40))
        await db.update_job_status("hb1", heartbeat_at="2026-07-04T12:00:00+00:00")
        return await db.load_job("hb1")
    loaded = _aiorun(run())
    assert loaded.heartbeat_at == "2026-07-04T12:00:00+00:00"
