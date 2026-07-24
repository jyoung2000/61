"""Export-time subtitle timing invariants matched to YouTube/Netflix:

  * Pass-2 max-duration TRIM — an un-splittable over-long cue (single word or a
    ``[♪ … ♪]`` marker) is trimmed to a readable display duration instead of
    lingering for its whole source window (the ED theme shipped at 99.46s).
  * enforce_min_gap — consecutive cues never ship TOUCHING (0 ms); a small gap
    is guaranteed on every SRT/VTT export (the sentence splitter emits contiguous
    pieces).
"""

import re

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.subtitle_formatter import enforce_readability, enforce_min_gap
from backend.services.srt_generator import generate_srt


def _cue(start, end, text, words=None, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker, words=words)


MAX_DUR_S = 7.0
MARKER_MAX_S = 4.0


# ── Pass-2 max-duration trim (un-splittable cues) ──

def test_single_word_over_long_is_trimmed():
    # "Above?" lingering 11.18s with no word timing and no split point.
    out = enforce_readability(
        [_cue(100.0, 111.18, "Above?")],
        max_cps=17.0, min_duration_ms=833, max_duration_ms=7000,
        allow_split=True, word_timed_split_only=True)
    assert len(out) == 1
    dur = out[0].end - out[0].start
    assert 0.833 - 1e-6 <= dur <= MAX_DUR_S + 1e-6
    assert dur < 4.0            # reading time + linger, well under the 99s-style linger
    assert out[0].text.strip() == "Above?"   # text never altered


def test_ending_theme_marker_capped_to_marker_max():
    # The 99.46s "[♪ Ending theme ♪]" marker -> short fixed hold, start unchanged.
    out = enforce_readability(
        [_cue(1287.42, 1386.88, "[♪ Ending theme ♪]")],
        max_cps=17.0, min_duration_ms=833, max_duration_ms=7000,
        allow_split=True, word_timed_split_only=True)
    assert len(out) == 1
    assert abs(out[0].start - 1287.42) < 1e-6           # in-time preserved
    assert abs((out[0].end - out[0].start) - MARKER_MAX_S) < 1e-3


def test_music_marker_over_max_capped():
    out = enforce_readability(
        [_cue(685.0, 697.46, "[♪ music ♪]")],
        max_cps=17.0, min_duration_ms=833, max_duration_ms=7000,
        allow_split=True, word_timed_split_only=True)
    assert abs((out[0].end - out[0].start) - MARKER_MAX_S) < 1e-3


def test_cue_at_max_duration_not_modified():
    # Exactly 7.0s -> guard is strict > max_dur_s, so it must be untouched.
    out = enforce_readability(
        [_cue(10.0, 17.0, "[♪ music ♪]")],
        max_cps=17.0, min_duration_ms=833, max_duration_ms=7000,
        allow_split=True, word_timed_split_only=True)
    assert abs((out[0].end - out[0].start) - 7.0) < 1e-6


def test_trim_preserves_and_clamps_words():
    words = [WordTimestamp(start=100.0, end=101.0, word="one"),
             WordTimestamp(start=101.0, end=102.0, word="two"),
             WordTimestamp(start=108.0, end=109.0, word="ten")]
    out = enforce_readability(
        [_cue(100.0, 111.18, "one two ten", words=words)],
        max_cps=17.0, min_duration_ms=833, max_duration_ms=7000,
        allow_split=True, word_timed_split_only=True)
    seg = out[0]
    # Every surviving word ends within the trimmed display window.
    for w in (seg.words or []):
        assert w.end <= seg.end + 1e-6


def test_splittable_cue_still_splits_not_clamped():
    # A long multi-sentence cue WITH word timings must still split into pieces,
    # not be clamped to one over-long cue.
    words = []
    t = 0.0
    for wtok in "first sentence here. second sentence here.".split():
        words.append(WordTimestamp(start=t, end=t + 0.9, word=wtok))
        t += 1.0
    out = enforce_readability(
        [_cue(0.0, t, "first sentence here. second sentence here.", words=words)],
        max_cps=6.0, min_duration_ms=833, max_duration_ms=2500,
        allow_split=True, word_timed_split_only=True)
    assert len(out) >= 2


# ── enforce_min_gap ──

def test_enforce_min_gap_opens_touching():
    segs = [_cue(0.0, 2.0, "a"), _cue(2.0, 4.0, "b"), _cue(4.0, 6.0, "c")]
    out = enforce_min_gap(segs, min_gap_s=0.08)
    for i in range(len(out) - 1):
        assert out[i + 1].start - out[i].end >= 0.08 - 1e-6


def test_enforce_min_gap_removes_overlap():
    segs = [_cue(0.0, 3.0, "a"), _cue(2.5, 5.0, "b")]
    out = enforce_min_gap(segs, min_gap_s=0.08)
    assert out[1].start - out[0].end >= 0.08 - 1e-6
    assert out[0].end <= out[1].start


def test_enforce_min_gap_idempotent():
    segs = [_cue(0.0, 2.0, "a"), _cue(2.0, 4.0, "b")]
    once = enforce_min_gap(segs, min_gap_s=0.08)
    times1 = [(s.start, s.end) for s in once]
    twice = enforce_min_gap(once, min_gap_s=0.08)
    times2 = [(s.start, s.end) for s in twice]
    assert times1 == times2


def test_enforce_min_gap_preserves_count_order_text():
    segs = [_cue(0.0, 2.0, "alpha"), _cue(2.0, 4.0, "beta"), _cue(4.0, 6.0, "gamma")]
    out = enforce_min_gap(segs, min_gap_s=0.08)
    assert [s.text for s in out] == ["alpha", "beta", "gamma"]
    assert len(out) == 3


def test_enforce_min_gap_zero_is_noop():
    segs = [_cue(0.0, 2.0, "a"), _cue(2.0, 4.0, "b")]
    out = enforce_min_gap(segs, min_gap_s=0.0)
    assert out[0].end == 2.0 and out[1].start == 2.0


def _parse_gaps(srt: str):
    times = []
    for m in re.finditer(
            r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)", srt):
        g = list(map(int, m.groups()))
        s = g[0]*3600+g[1]*60+g[2]+g[3]/1000
        e = g[4]*3600+g[5]*60+g[6]+g[7]/1000
        times.append((s, e))
    return [times[i+1][0] - times[i][1] for i in range(len(times)-1)]


def test_generate_srt_has_no_touching_cues():
    # Contiguous pieces (left.end == right.start) mimic split_run_on_cues output.
    segs = [_cue(i * 2.0, (i + 1) * 2.0, f"line {i}") for i in range(6)]
    srt = generate_srt(segs, include_speakers=False)
    gaps = _parse_gaps(srt)
    assert gaps and all(g >= 0.079 for g in gaps)   # >= 80ms within 1ms rounding
