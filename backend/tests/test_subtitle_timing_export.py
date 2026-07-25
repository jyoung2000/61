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
    from backend.config import settings
    min_gap_s = float(getattr(settings, "SUBTITLE_MIN_GAP_MS", 42)) / 1000.0
    segs = [_cue(i * 2.0, (i + 1) * 2.0, f"line {i}") for i in range(6)]
    srt = generate_srt(segs, include_speakers=False)
    gaps = _parse_gaps(srt)
    assert gaps and all(g >= min_gap_s - 0.0011 for g in gaps)


# ── Frame quantization (the "authored like YouTube" invariant) ──

def test_frame_quantize_puts_every_cue_on_the_frame_grid():
    from backend.services.subtitle_formatter import quantize_to_frames
    fps = 24000 / 1001   # 23.976 — NTSC-pulldown, the real source rate
    segs = [_cue(1.234, 3.777, "a"), _cue(3.777, 6.111, "b"), _cue(6.5, 9.01, "c")]
    out = quantize_to_frames(segs, fps, min_gap_frames=1)
    for s in out:
        for t in (s.start, s.end):
            frames = t * fps
            assert abs(frames - round(frames)) / fps <= 0.0011


def test_frame_quantize_keeps_one_frame_gap():
    from backend.services.subtitle_formatter import quantize_to_frames
    fps = 24.0
    # Touching cues: quantizing must leave exactly one frame between them.
    segs = [_cue(1.0, 2.0, "a"), _cue(2.0, 3.0, "b"), _cue(3.0, 4.0, "c")]
    out = quantize_to_frames(segs, fps, min_gap_frames=1)
    for i in range(len(out) - 1):
        gap = out[i + 1].start - out[i].end
        assert gap >= (1.0 / fps) - 0.0011
        assert gap <= (1.0 / fps) + 0.0011     # exactly one frame, not two


def test_frame_quantize_is_idempotent():
    from backend.services.subtitle_formatter import quantize_to_frames
    fps = 24000 / 1001
    segs = [_cue(1.234, 3.777, "a"), _cue(3.777, 6.111, "b")]
    once = quantize_to_frames(segs, fps)
    t1 = [(s.start, s.end) for s in once]
    twice = quantize_to_frames(once, fps)
    assert [(s.start, s.end) for s in twice] == t1


def test_frame_quantize_noop_without_fps():
    from backend.services.subtitle_formatter import quantize_to_frames
    segs = [_cue(1.234, 3.777, "a")]
    out = quantize_to_frames(segs, 0.0)
    assert (out[0].start, out[0].end) == (1.234, 3.777)


def test_srt_timestamps_round_not_truncate():
    # 29.988 s must emit as ,988 — the old per-field arithmetic truncated
    # (29.988 % 1 == 0.98799…) and emitted ,987, one ms early.
    from backend.services.srt_generator import _format_srt_time
    assert _format_srt_time(29.988) == "00:00:29,988"
    assert _format_srt_time(0.0) == "00:00:00,000"
    assert _format_srt_time(3661.5) == "01:01:01,500"


def test_generate_srt_frame_aligns_when_fps_given():
    fps = 24000 / 1001
    segs = [_cue(i * 2.0 + 0.137, (i + 1) * 2.0, f"line {i}") for i in range(5)]
    srt = generate_srt(segs, include_speakers=False, fps=fps)
    times = []
    for m in re.finditer(
            r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)", srt):
        g = list(map(int, m.groups()))
        times.append(g[0]*3600+g[1]*60+g[2]+g[3]/1000)
        times.append(g[4]*3600+g[5]*60+g[6]+g[7]/1000)
    assert times
    for t in times:
        frames = t * fps
        assert abs(frames - round(frames)) / fps <= 0.0011


# ── clamp_cue_durations on the persisted dict rows ──

def test_clamp_cue_durations_on_dict_rows():
    from backend.services.subtitle_formatter import clamp_cue_durations
    rows = [{"start": 1287.42, "end": 1386.88, "text": "[♪ Ending theme ♪]"},
            {"start": 1400.0, "end": 1402.0, "text": "normal cue"}]
    clamp_cue_durations(rows)
    assert abs((rows[0]["end"] - rows[0]["start"]) - MARKER_MAX_S) < 1e-3
    assert rows[1]["end"] == 1402.0          # already short — untouched


def test_quantize_to_frames_accepts_dict_rows():
    # Attribute-only access made a dict row read every start as 0.0 and then raise
    # on assignment; the caller's fail-soft wrapper turned that into "return the
    # cues unchanged", silently dropping BOTH quantization and the gap.
    from backend.services.subtitle_formatter import quantize_to_frames
    fps = 24000 / 1001
    rows = [{"start": 1.234, "end": 3.777, "text": "a"},
            {"start": 3.777, "end": 6.111, "text": "b"}]
    out = quantize_to_frames(rows, fps, min_gap_frames=1)
    for r in out:
        for k in ("start", "end"):
            frames = r[k] * fps
            assert abs(frames - round(frames)) / fps <= 0.0011
    assert out[1]["start"] - out[0]["end"] >= (1.0 / fps) - 0.0011


def test_balanced_two_line_prefers_even_legal_split():
    from backend.services.subtitle_formatter import _balanced_two_line
    # 79 chars: the greedy wrapper produced 30/48 (line 2 over budget) because a
    # long word ended line 1 early. No legal 42/42 split exists here, so the
    # helper must decline rather than emit an over-budget line.
    assert _balanced_two_line(
        "They're not intimidated by our intimidation tactics; "
        "attack and shoot them down", 42) is None
    # Where a legal split DOES exist it must be chosen, and be near-even.
    got = _balanced_two_line("alpha beta gamma delta epsilon zeta eta theta", 30)
    assert got is not None
    a, b = got.split("\n")
    assert len(a) <= 30 and len(b) <= 30
    assert abs(len(a) - len(b)) <= 12


def test_enforce_min_gap_on_dict_rows():
    rows = [{"start": 0.0, "end": 2.0, "text": "a"},
            {"start": 2.0, "end": 4.0, "text": "b"}]
    out = enforce_min_gap(rows, min_gap_s=0.042)
    assert out[1]["start"] - out[0]["end"] >= 0.042 - 1e-6
