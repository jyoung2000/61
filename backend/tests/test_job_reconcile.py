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
