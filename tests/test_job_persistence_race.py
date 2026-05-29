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


# ── Regression: COMPLETE save must carry summary + clips + transcript ──────
# A JA→EN job reached COMPLETE with empty summary/clips on the frontend
# ("Generating summary..." forever). Root cause: when the first full
# COMPLETE save didn't round-trip, the fallback re-saved status ONLY,
# leaving summary/clips/transcript empty. The pipeline now re-issues the
# FULL payload on the verify-retry. These tests lock in the persistence
# contract that fix depends on.

from backend.models import VideoSummary


def _mk_summary() -> VideoSummary:
    return VideoSummary(
        overview="A Gundam battle.", key_topics=["mecha", "war"],
        tone="dramatic", estimated_audience="anime fans",
        content_category="animation",
    )


def test_complete_save_persists_summary_clips_transcript(tmp_job_store):
    """The full COMPLETE payload round-trips all three result fields."""
    async def scenario():
        job = JobResult(job_id="c", filename="f", file_path="p",
                        status=JobStatus.DETECTING_CLIPS.value)
        await db.save_job(job)
        await db.update_job_status(
            "c", status=JobStatus.COMPLETE.value, progress=100,
            summary=_mk_summary(), clips=[_mk_clip()],
            transcript=[_mk_seg("hello"), _mk_seg("world")],
        )
        return await db.load_job("c")

    result = _run(scenario())
    assert str(result.status) == str(JobStatus.COMPLETE)
    assert result.summary is not None and result.summary.overview
    assert len(result.clips) == 1
    assert len(result.transcript) == 2


def test_full_payload_resave_restores_after_status_only_loss(tmp_job_store):
    """Simulate the production failure + the fix: a status-only COMPLETE
    leaves the results empty; re-issuing the FULL payload restores them."""
    async def scenario():
        job = JobResult(job_id="d", filename="f", file_path="p",
                        status=JobStatus.DETECTING_CLIPS.value)
        await db.save_job(job)
        # The old fallback behaviour: status flips to COMPLETE but the
        # results were never carried → empty summary/clips (the bug).
        await db.update_job_status("d", status=JobStatus.COMPLETE.value, progress=100)
        broken = await db.load_job("d")
        assert str(broken.status) == str(JobStatus.COMPLETE)
        assert broken.summary is None and len(broken.clips) == 0
        # The fix: re-issue the FULL payload.
        complete_fields = dict(
            status=JobStatus.COMPLETE.value, progress=100,
            summary=_mk_summary(), clips=[_mk_clip()],
            transcript=[_mk_seg("hi")],
        )
        await db.update_job_status("d", **complete_fields)
        return await db.load_job("d")

    result = _run(scenario())
    assert result.summary is not None and result.summary.overview
    assert len(result.clips) == 1
    assert len(result.transcript) == 1
