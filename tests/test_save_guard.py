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
from backend.models import JobResult, TranscriptSegment, VideoSummary


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


def _summary(overview="A clear overview of the video."):
    return VideoSummary(overview=overview, key_topics=["mecha", "war"],
                        tone="serious", estimated_audience="anime fans",
                        content_category="entertainment")


def test_save_job_does_not_wipe_summary(tmp_jobs):
    # The summary stage persists via update_job_status(summary=...) — a separate DB
    # write that does NOT update the in-memory pipeline job. A later whole-object
    # save_job() of that stale job (summary=None) must NOT clobber the persisted
    # summary, or the Summary tab shows "No summary available" despite generation
    # succeeding (the production symptom).
    j = _job(translated=[_seg("Hello")])
    j.summary = _summary("The real generated summary.")
    _run(db.save_job(j))
    _run(db.save_job(_job(translated=[_seg("Hello")])))   # stale snapshot: summary=None
    reloaded = _run(db.load_job("j1"))
    assert reloaded.summary is not None                   # kept
    assert reloaded.summary.overview == "The real generated summary."


def test_legit_summary_regeneration_overwrites(tmp_jobs):
    # The inverse must NOT be blocked: a re-analysis producing a NEW non-empty
    # summary has to win, exactly like the translated_transcript guard.
    j = _job(translated=[_seg("Hi")])
    j.summary = _summary("Old summary.")
    _run(db.save_job(j))
    j2 = _job(translated=[_seg("Hi")])
    j2.summary = _summary("New, better summary.")
    _run(db.save_job(j2))
    reloaded = _run(db.load_job("j1"))
    assert reloaded.summary.overview == "New, better summary."   # non-empty write wins


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


def _dumps(segs):
    return [s.model_dump() for s in segs]


def test_purity_guard_refuses_half_source_overwrite(tmp_jobs):
    # The recurring bug: a clean English translation (0% source-script) gets
    # clobbered by a half-Japanese re-analyze / Whisper-native fallback. The
    # purity guard in update_job_status must refuse that regression.
    clean = [_seg("Hello there"), _seg("How are you", 1, 2)]
    _run(db.save_job(_job(status="complete", translated=clean)))
    _run(db.update_job_status("j1", subtitle_language="en"))   # non-CJK target
    half_jp = [_seg("宇宙コロニーでの生活に新たな希望を"), _seg("Hello there", 1, 2)]  # 50% CJK
    _run(db.update_job_status("j1", translated_transcript=_dumps(half_jp)))
    reloaded = _run(db.load_job("j1"))
    assert len(reloaded.translated_transcript) == 2
    assert all("宇宙" not in (s.text or "") for s in reloaded.translated_transcript)
    assert reloaded.translated_transcript[0].text == "Hello there"   # clean kept


def test_purity_guard_allows_clean_retranslation_over_dirty(tmp_jobs):
    # The inverse must NOT be blocked: a clean English result replacing an
    # earlier half-Japanese one is an improvement and has to win.
    dirty = [_seg("宇宙コロニー"), _seg("Hi", 1, 2)]
    _run(db.save_job(_job(status="complete", translated=dirty)))
    _run(db.update_job_status("j1", subtitle_language="en"))
    clean = [_seg("In the space colony"), _seg("Hi", 1, 2)]
    _run(db.update_job_status("j1", translated_transcript=_dumps(clean)))
    reloaded = _run(db.load_job("j1"))
    assert reloaded.translated_transcript[0].text == "In the space colony"


def test_purity_guard_allows_cjk_target(tmp_jobs):
    # A →ja translation legitimately contains CJK; the guard must not fire when
    # the TARGET language is itself CJK.
    clean_en = [_seg("Hello"), _seg("World", 1, 2)]
    _run(db.save_job(_job(status="complete", translated=clean_en)))
    _run(db.update_job_status("j1", subtitle_language="ja"))   # CJK target
    ja = [_seg("こんにちは"), _seg("世界", 1, 2)]
    _run(db.update_job_status("j1", translated_transcript=_dumps(ja)))
    reloaded = _run(db.load_job("j1"))
    assert reloaded.translated_transcript[0].text == "こんにちは"
