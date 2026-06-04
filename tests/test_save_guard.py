"""Tests for the job-save anti-clobber guard.

Production corruption seen: a transcript edit / reverse-sync that loaded the job
microseconds before the pipeline persisted the translation then saved its stale
snapshot back — wiping translated_transcript (→0) and reverting a finished status
(→detecting_clips). The guard in _save_job_unlocked refuses both downgrades.
"""

from __future__ import annotations

import asyncio

import pytest

import backend.database as db
from backend.models import JobResult, TranscriptSegment


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def tmp_jobs(monkeypatch, tmp_path):
    def _dir(job_id):
        d = tmp_path / job_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(db, "_job_dir", _dir)
    monkeypatch.setattr(db, "_job_path", lambda jid: str(tmp_path / jid / "job.json"))
    return tmp_path


def _seg(text, a=0.0, b=1.0):
    return TranscriptSegment(text=text, start=a, end=b, speaker="Speaker 1")


def _job(job_id="j1", status="complete", translated=None):
    return JobResult(job_id=job_id, filename="video.mp4", file_path="/v.mp4",
                     status=status, translated_transcript=translated or [])


def test_save_job_does_not_wipe_translated(tmp_jobs):
    _run(db.save_job(_job(translated=[_seg("Hello")])))
    _run(db.save_job(_job(translated=[])))            # stale snapshot, no translation
    reloaded = _run(db.load_job("j1"))
    assert len(reloaded.translated_transcript) == 1   # kept
    assert reloaded.translated_transcript[0].text == "Hello"


def test_save_job_does_not_revert_terminal_status(tmp_jobs):
    _run(db.save_job(_job(status="complete", translated=[_seg("Hi")])))
    _run(db.save_job(_job(status="detecting_clips", translated=[])))  # stale revert
    reloaded = _run(db.load_job("j1"))
    assert db._status_value(reloaded.status) == "complete"
    assert len(reloaded.translated_transcript) == 1


def test_legit_retranslation_overwrites(tmp_jobs):
    _run(db.save_job(_job(translated=[_seg("Old")])))
    _run(db.save_job(_job(translated=[_seg("New1"), _seg("New2", 1, 2)])))
    reloaded = _run(db.load_job("j1"))
    assert len(reloaded.translated_transcript) == 2   # non-empty write still wins
    assert reloaded.translated_transcript[0].text == "New1"


def test_update_job_status_still_allows_reanalysis_reset(tmp_jobs):
    # Re-analysis moves a COMPLETE job back to QUEUED via update_job_status
    # (protect_terminal=False). The terminal guard is save_job-only, so this
    # legitimate reset must NOT be blocked.
    _run(db.save_job(_job(status="complete")))
    _run(db.update_job_status("j1", status="queued", progress=1,
                              protect_terminal=False))
    reloaded = _run(db.load_job("j1"))
    assert db._status_value(reloaded.status) == "queued"
