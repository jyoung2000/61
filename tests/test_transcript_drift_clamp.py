"""Tests for clamping faster-whisper timestamp drift to the real audio end.

On a repetitive-music AMV, faster-whisper drifts/loops and stamps cues PAST the
audio — the Gundam Wing run produced subtitle cues out to 36:33 on a 24:27
(1467 s) video. Those past-the-end cues were loop-repeats of earlier narration /
lyrics, not unique tail dialogue, so dropping them loses no real content while
fixing the back-third timing. ``clamp_segments_to_duration`` does that, and
clamps any cue that merely overruns the end. These pin that behaviour.
"""

from __future__ import annotations

from backend.services.transcript_dedup import clamp_segments_to_duration


_DUR = 1467.0  # 24:27, the real audio length


def test_drops_drifted_cues_and_clamps_straddler():
    segs = [
        {"start_sec": 10.0, "end_sec": 12.0, "text": "early",
         "words": [{"word": "early", "start": 10.0, "end": 12.0}]},
        # Straddles the end: starts before 1467 s, ends after.
        {"start_sec": 1465.0, "end_sec": 1471.0, "text": "straddle",
         "words": [{"word": "strad", "start": 1465.0, "end": 1466.0},
                   {"word": "dle", "start": 1466.0, "end": 1471.0}]},
        # Pure drift: a looped narration re-emitted past the end.
        {"start_sec": 1500.0, "end_sec": 1510.0, "text": "loop-repeat", "words": []},
        {"start_sec": 2193.0, "end_sec": 2195.0, "text": "loop @ 36:33", "words": []},
    ]
    out, changed = clamp_segments_to_duration(
        segs, _DUR, start_key="start_sec", end_key="end_sec")

    assert changed == 3                              # 2 dropped + 1 clamped
    assert [s["text"] for s in out] == ["early", "straddle"]
    assert out[0]["end_sec"] == 12.0                 # in-range cue untouched
    assert out[1]["end_sec"] == _DUR                 # straddler clamped to the end
    # the straddler's word that ran past the end is clamped too
    assert out[1]["words"][-1]["end"] == _DUR


def test_noop_without_duration():
    segs = [{"start_sec": 0.0, "end_sec": 5.0, "text": "x"}]
    out, changed = clamp_segments_to_duration(
        segs, 0, start_key="start_sec", end_key="end_sec")
    assert changed == 0 and out is segs


def test_in_range_untouched_default_schema():
    segs = [{"start": 0.0, "end": 5.0, "text": "a"},
            {"start": 5.0, "end": 10.0, "text": "b"}]
    out, changed = clamp_segments_to_duration(segs, 100.0)
    assert changed == 0 and len(out) == 2


def test_drops_word_that_starts_past_end_in_straddler():
    segs = [{"start_sec": 1466.0, "end_sec": 1480.0, "text": "tail",
             "words": [{"word": "in", "start": 1466.0, "end": 1466.5},
                       {"word": "past", "start": 1468.0, "end": 1480.0}]}]
    out, changed = clamp_segments_to_duration(
        segs, _DUR, start_key="start_sec", end_key="end_sec")
    assert changed == 1 and len(out) == 1
    # the word starting after the end is dropped; the in-range word is kept
    assert [w["word"] for w in out[0]["words"]] == ["in"]
