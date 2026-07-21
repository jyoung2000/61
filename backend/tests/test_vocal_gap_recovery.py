"""Targeted vocal-separation gap recovery: pure logic + the post-COMPLETE
merge path. The heavy pieces (Demucs, remote ASR) are exercised only through
their fail-soft seams — recovery must never be able to break a finished job.
"""

import asyncio
import json

import pytest

from backend.services.vocal_gap_recovery import (
    _JUNK_RE,
    _clip_to_gap,
    _concat_offsets,
    find_coverage_gaps,
    merge_recovered,
)


def test_concat_offsets_track_silence_separators():
    # All spans separated into ONE Demucs pass: each clip's start offset inside
    # the concatenation = sum of prior clip durations + one separator between.
    assert _concat_offsets([20.0, 15.0, 32.0], 1.0) == [0.0, 21.0, 37.0]
    assert _concat_offsets([10.0], 1.0) == [0.0]   # lone span → no separator
    assert _concat_offsets([], 1.0) == []


def _track(*spans):
    return [{"start": float(a), "end": float(b), "text": t}
            for a, b, t in spans]


def test_gap_finder_interior_holes_only():
    segs = _track((0, 4, "a"), (5, 9, "b"), (40, 44, "c"), (44, 300, "d"))
    # 9→40 is a hole; nothing before the first or after the last cue counts.
    gaps = find_coverage_gaps(segs, min_gap_s=8.0, pad_s=2.0)
    assert gaps == [(7.0, 42.0)]


def test_gap_finder_ignores_small_holes_and_respects_caps():
    segs = _track((0, 10, "a"), (14, 20, "b"),      # 4s hole — too small
                  (60, 64, "c"), (200, 204, "d"),   # 136s hole
                  (400, 404, "e"))                  # 196s hole
    gaps = find_coverage_gaps(segs, min_gap_s=8.0, pad_s=0.0,
                              max_spans=8, max_total_s=150.0)
    # Largest-first budget: the 196s hole exceeds 150s alone and is skipped;
    # the 136s hole fits.
    assert gaps == [(64.0, 200.0)]


def test_gap_finder_needs_two_cues():
    assert find_coverage_gaps(_track((0, 5, "only"))) == []
    assert find_coverage_gaps([]) == []


def test_clip_to_gap_strips_the_padded_leadin():
    gap = (95.0, 132.0)  # padded by 2s on each side → real hole 97..130
    inside = _clip_to_gap({"start": 100.0, "end": 104.0, "text": "x"}, gap, 2.0)
    assert inside == {"start": 100.0, "end": 104.0, "text": "x"}
    # A cue entirely inside the lead-in pad re-hears an EXISTING cue — drop.
    assert _clip_to_gap({"start": 95.2, "end": 96.8, "text": "x"}, gap, 2.0) is None
    # Straddling the pad boundary: clipped to the real hole.
    edge = _clip_to_gap({"start": 96.0, "end": 101.0, "text": "x"}, gap, 2.0)
    assert edge["start"] == 97.0 and edge["end"] == 101.0


def test_merge_is_additive_and_drops_boundary_echoes():
    existing = _track((0, 4, "The Colony Summit begins."),
                      (40, 44, "General Septem is waiting."))
    merged, added = merge_recovered(existing, [
        {"start": 10.0, "end": 13.0, "text": "Any comments, Vice Minister?"},
        {"start": 14.0, "end": 17.0, "text": "Any comment, Vice Minister?!"},  # dup of accepted
        {"start": 38.0, "end": 41.0, "text": "General Septem is waiting"},     # echo of neighbor
    ])
    assert added == 1
    assert len(merged) == 3
    # Sorted by start, existing cues untouched.
    assert [m["text"] for m in merged] == [
        "The Colony Summit begins.",
        "Any comments, Vice Minister?",
        "General Septem is waiting.",
    ]


def test_junk_hallucinations_regex():
    assert _JUNK_RE.match("Thank you for watching.")
    assert _JUNK_RE.match("ご視聴ありがとうございました")
    assert not _JUNK_RE.match("Thank you, Lieutenant.")


def test_post_complete_recovery_merges_and_translates(monkeypatch, tmp_path):
    from backend import database
    from backend.models import JobResult, JobStatus, TranscriptSegment
    from backend.services import pipeline as pl

    monkeypatch.setattr(database, "_job_dir",
                        lambda job_id: str(tmp_path / job_id))
    now = "2026-07-21T00:00:00+00:00"
    d = tmp_path / "j-rec"
    d.mkdir()
    src = [TranscriptSegment(start=float(i * 10), end=float(i * 10 + 4),
                             text=f"セリフ{i}", speaker="Speaker 1")
           for i in range(6)]
    tt = [TranscriptSegment(start=s.start, end=s.end,
                            text=f"line {i}", speaker="Speaker 1")
          for i, s in enumerate(src)]
    job = JobResult(job_id="j-rec", filename="v.mp4", file_path="/x/v.mp4",
                    status=JobStatus.COMPLETE, created_at=now, updated_at=now,
                    transcript=src, translated_transcript=tt,
                    subtitle_language="en")
    (d / "job.json").write_text(job.model_dump_json())

    async def _fake_recover(job_id, audio, segments, lang, work_dir):
        return [{"start": 21.0, "end": 24.0, "text": "埋もれた台詞",
                 "speaker": "Speaker 1"}]

    monkeypatch.setattr(
        "backend.services.vocal_gap_recovery.recover_gap_dialogue",
        _fake_recover)

    events = []

    async def _fake_broadcast(job_id, msg):
        events.append(msg)

    monkeypatch.setattr(pl, "broadcast_ws", _fake_broadcast)

    class _Orch:
        async def text_completion(self, prompt, **kw):
            assert "埋もれた台詞" in prompt
            return "The buried line."

    asyncio.run(pl._post_complete_gap_recovery("j-rec", _Orch()))

    saved = json.loads((d / "job.json").read_text())
    src_texts = [s["text"] for s in saved["transcript"]]
    tt_texts = [s["text"] for s in saved["translated_transcript"]]
    assert "埋もれた台詞" in src_texts
    assert "The buried line." in tt_texts
    assert len(saved["transcript"]) == 7
    # Existing cues untouched, order preserved by start time.
    assert src_texts[0] == "セリフ0" and src_texts[-1] == "セリフ5"
    assert any(e.get("task") == "vocal_recovery"
               and e.get("status") == "complete" for e in events)


def test_post_complete_recovery_noop_without_gap_hits(monkeypatch, tmp_path):
    from backend import database
    from backend.models import JobResult, JobStatus, TranscriptSegment
    from backend.services import pipeline as pl

    monkeypatch.setattr(database, "_job_dir",
                        lambda job_id: str(tmp_path / job_id))
    now = "2026-07-21T00:00:00+00:00"
    d = tmp_path / "j-none"
    d.mkdir()
    src = [TranscriptSegment(start=float(i), end=float(i) + 0.9, text=f"t{i}",
                             speaker="Speaker 1") for i in range(6)]
    job = JobResult(job_id="j-none", filename="v.mp4", file_path="/x/v.mp4",
                    status=JobStatus.COMPLETE, created_at=now, updated_at=now,
                    transcript=src)
    before = job.model_dump_json()
    (d / "job.json").write_text(before)

    async def _fake_recover(*a, **k):
        return []

    monkeypatch.setattr(
        "backend.services.vocal_gap_recovery.recover_gap_dialogue",
        _fake_recover)
    asyncio.run(pl._post_complete_gap_recovery("j-none", None))
    assert (d / "job.json").read_text() == before  # nothing rewritten
