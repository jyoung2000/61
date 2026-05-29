"""Tests for transcript ↔ NLE-timeline sync persistence.

Covers the segment-cleaning helper the bulk transcript-replace endpoint uses
(coerce / drop blank+backwards / chronological sort) and the database
persistence contract the endpoint relies on (writing the translated track vs
the original track, surviving a reload = container restart).

We deliberately do NOT import backend.routers.jobs here: it pulls the heavy
provider/cv2/torch import chain, and stubbing those modules pollutes
sys.modules for sibling tests in the same pytest process. The endpoint's
logic lives in the dependency-light transcript_sync helper + the database
layer, both imported directly.
"""

import asyncio
import os
import tempfile

import pytest

def _aiorun(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


import backend.database as db
from backend.models import JobResult, JobStatus, TranscriptSegment
from backend.services.transcript_sync import clean_and_sort_segments


@pytest.fixture()
def tmp_job_store(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr(db, "_job_dir", lambda jid: os.path.join(d, jid))
    return d


def _run(coro):
    return _aiorun(coro)


def _seg(s, e, text, speaker="Speaker 1"):
    return TranscriptSegment(start=s, end=e, text=text, speaker=speaker)


def _f(seg, key):
    return seg[key] if isinstance(seg, dict) else getattr(seg, key)


# ── clean_and_sort_segments (the endpoint's transform) ──────────────────────

def test_clean_sorts_and_drops_bad_cues():
    rows = clean_and_sort_segments([
        {"start": 5.0, "end": 6.0, "text": "second", "speaker": "Speaker 1"},
        {"start": 1.0, "end": 2.0, "text": "first", "speaker": "Speaker 1"},
        {"start": 9.0, "end": 9.0, "text": "   ", "speaker": "Speaker 1"},   # blank
        {"start": 9.0, "end": 8.0, "text": "backwards", "speaker": "Speaker 1"},  # bad
    ])
    assert [r["text"] for r in rows] == ["first", "second"]


def test_clean_preserves_words():
    rows = clean_and_sort_segments([{
        "start": 0.0, "end": 2.0, "text": "hello world", "speaker": "Speaker 1",
        "words": [{"start": 0.0, "end": 1.0, "word": "hello"},
                  {"start": 1.0, "end": 2.0, "word": "world"}],
    }])
    assert rows[0]["words"][1]["word"] == "world"


def test_clean_zero_duration_kept():
    # Zero-duration cues are valid (the reference SRT has them); only
    # backwards cues are dropped.
    rows = clean_and_sort_segments([{"start": 3.0, "end": 3.0, "text": "No sir.", "speaker": "Speaker 1"}])
    assert len(rows) == 1


# ── DB persistence contract (bulk replace → survives reload/restart) ─────────

def _mk_job(job_id, *, translated=None, transcript=None):
    return JobResult(
        job_id=job_id, filename="v.mp4", file_path="p",
        status=JobStatus.COMPLETE.value,
        transcript=[s.model_dump() for s in (transcript or [])],
        translated_transcript=[s.model_dump() for s in (translated or [])],
    )


def test_translated_track_replace_persists(tmp_job_store):
    async def scenario():
        await db.save_job(_mk_job(
            "a", translated=[_seg(0, 1, "old")], transcript=[_seg(0, 1, "原文")]))
        rows = clean_and_sort_segments([
            {"start": 1.0, "end": 2.0, "text": "edited translated", "speaker": "Speaker 1"},
        ])
        # Mirror the endpoint: translated target writes translated_transcript.
        await db.update_job_status("a", translated_transcript=rows)
        # Reload = container restart.
        return await db.load_job("a")

    job = _run(scenario())
    assert _f(job.translated_transcript[0], "text") == "edited translated"
    assert _f(job.transcript[0], "text") == "原文"   # original untouched


def test_original_track_replace_persists(tmp_job_store):
    async def scenario():
        await db.save_job(_mk_job("b", transcript=[_seg(0, 1, "orig")]))
        rows = clean_and_sort_segments([
            {"start": 0.0, "end": 1.5, "text": "edited", "speaker": "Speaker 1"},
        ])
        await db.update_job_status("b", transcript=rows)
        return await db.load_job("b")

    job = _run(scenario())
    assert _f(job.transcript[0], "text") == "edited"
    assert _f(job.transcript[0], "end") == 1.5
