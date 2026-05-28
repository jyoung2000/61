"""Regression tests for the job-persistence read-modify-write race.

Before the fix, ``database.update_job_status`` loaded the job, mutated it,
and saved it in two separately-locked steps. A progress callback relayed
from the clipper worker thread (via ``run_coroutine_threadsafe``) could
load the pre-COMPLETE snapshot and then save it back *after* the COMPLETE
save landed — reverting ``status`` to ``detecting_clips`` and wiping the
just-persisted ``clips`` and ``translated_transcript``.

User-visible symptoms this caused:
  * the pipeline appeared "stuck finalizing clips" (status never reached
    COMPLETE in the persisted record);
  * the clip list read back empty;
  * the translated subtitle track silently reverted to the source
    language (Japanese stayed Japanese even after a successful EN
    translation, because ``translated_transcript`` was clobbered).

The fix makes the whole load→modify→save atomic under the per-job lock and
adds an opt-in ``protect_terminal`` guard so a stale in-progress write can't
drag a finished job backwards — while deliberate restarts (re-analysis,
retranscribe) still work.
"""
import asyncio
import os
import tempfile

import pytest

import backend.database as db
from backend.models import (
    ClipCandidate,
    JobResult,
    JobStatus,
    TranscriptSegment,
)


def _mk_clip() -> ClipCandidate:
    return ClipCandidate(
        id=1, title="t", start_time=0, end_time=5, duration=5,
        viral_score=80, viral_score_reasoning="r", clip_type="x",
        platform="both", suggested_caption="c", hook_text="h",
        why_this_works="w",
    )


def _mk_seg(text="hello") -> TranscriptSegment:
    return TranscriptSegment(start=0, end=1, text=text, speaker="Speaker 1")


@pytest.fixture()
def tmp_job_store(monkeypatch):
    """Redirect the on-disk job store to a throwaway temp dir."""
    d = tempfile.mkdtemp()
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    return d


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_stale_progress_does_not_revert_complete(tmp_job_store):
    """A late guarded progress write must not undo the COMPLETE save."""
    async def scenario():
        job = JobResult(job_id="a", filename="f", file_path="p",
                        status=JobStatus.DETECTING_CLIPS.value)
        await db.save_job(job)

        async def complete():
            await db.update_job_status(
                "a", status=JobStatus.COMPLETE.value, progress=100,
                clips=[_mk_clip()], translated_transcript=[_mk_seg()],
            )

        async def stale_progress():
            # Mirrors the clipper progress relay landing after COMPLETE.
            await db.update_job_status(
                "a", status=JobStatus.DETECTING_CLIPS.value, progress=96,
                progress_message="Finalizing clip detection...",
                protect_terminal=True,
            )

        # Run in both interleavings to cover either landing order.
        await asyncio.gather(stale_progress(), complete())
        return await db.load_job("a")

    result = _run(scenario())
    assert str(result.status) == str(JobStatus.COMPLETE)
    assert len(result.clips) == 1
    assert len(result.translated_transcript) == 1
    assert result.progress == 100


def test_post_complete_field_write_persists(tmp_job_store):
    """The background translation write (status=None) must persist on a
    COMPLETE job — this is what carries the English transcript."""
    async def scenario():
        job = JobResult(job_id="b", filename="f", file_path="p",
                        status=JobStatus.COMPLETE.value, progress=100)
        await db.save_job(job)
        await db.update_job_status(
            "b", translated_transcript=[_mk_seg("english"), _mk_seg("text")],
        )
        return await db.load_job("b")

    result = _run(scenario())
    assert str(result.status) == str(JobStatus.COMPLETE)
    assert len(result.translated_transcript) == 2


def test_reanalysis_can_leave_terminal_state(tmp_job_store):
    """Re-analysis resets a COMPLETE job via protect_terminal=False."""
    async def scenario():
        job = JobResult(job_id="c", filename="f", file_path="p",
                        status=JobStatus.COMPLETE.value, progress=100)
        await db.save_job(job)
        await db.update_job_status(
            "c", status=JobStatus.QUEUED.value, progress=1,
            protect_terminal=False,
        )
        return await db.load_job("c")

    result = _run(scenario())
    assert str(result.status) == str(JobStatus.QUEUED)


def test_retranscribe_default_allows_terminal_transition(tmp_job_store):
    """Retranscribe sets TRANSCRIBING directly (default guard off)."""
    async def scenario():
        job = JobResult(job_id="d", filename="f", file_path="p",
                        status=JobStatus.COMPLETE.value, progress=100)
        await db.save_job(job)
        await db.update_job_status("d", status=JobStatus.TRANSCRIBING.value)
        return await db.load_job("d")

    result = _run(scenario())
    assert str(result.status) == str(JobStatus.TRANSCRIBING)
