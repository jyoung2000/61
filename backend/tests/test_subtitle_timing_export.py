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


def _words_for(text, start, end):
    """A plausible per-word timing array for ``text``, 1:1 with its tokens —
    what every cue carries once the pipeline's tiers have run."""
    from backend.services.subtitle_aligner import distribute_cue_window
    return [WordTimestamp(**w)
            for w in distribute_cue_window(text.split(), start, end)]


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


# ── the wrapper must never invent text ────────────────────────────────────

def test_greedy_wrap_does_not_duplicate_a_repeated_word():
    from backend.services.subtitle_formatter import _greedy_wrap
    # The wrap point lands on "the", which also appears near the START of the
    # cue. Locating the tail by VALUE instead of by position rebuilt the last
    # line from that earlier "the", re-emitting a whole clause: one measured cue
    # grew 81 -> 134 characters with "the machine might be fine," repeated.
    text = ("While the machine might be fine, the reckless pilot "
            "who flew it should have died.")
    got = _greedy_wrap(text.split(), 34, 2)
    assert " ".join(got.split()) == text, got
    # Same shape, every common word as the wrap point.
    for filler in ("the", "a", "to", "of", "and"):
        t = f"Alpha {filler} bravo charlie delta echo {filler} foxtrot golf hotel india"
        out = _greedy_wrap(t.split(), 20, 2)
        assert " ".join(out.split()) == t, (filler, out)


def test_readability_never_grows_a_cue_at_a_tight_budget():
    # End-to-end guard on the same defect: the shipped text may be re-split
    # across cues or re-wrapped, but the words must survive unchanged.
    text = ("While the machine might be fine, the reckless pilot "
            "who flew it should have died.")
    for budget in (28, 34, 42):
        out = enforce_readability(
            [TranscriptSegment(start=10.0, end=15.0, text=text,
                               speaker="Speaker 1")],
            max_chars_per_line=budget, max_lines=2)
        flat = " ".join(" ".join((c.text or "").split()) for c in out)
        assert len(flat) <= len(text) + 4, (budget, flat)


def test_unfittable_text_shares_the_overflow_instead_of_stacking_it():
    from backend.services.subtitle_formatter import _hard_wrap_lines
    # 94 chars cannot fit 2 x 34. Folding everything into line 2 gave the worst
    # shape in the file (34 over 62); both lines are over budget either way, so
    # the excess must be SHARED.
    text = ("However, Zechs of Oz was using a new mobile suit "
            "called Cancer to securely capture the Gundam.")
    lines = _hard_wrap_lines(text, 34, 2).split("\n")
    assert len(lines) == 2
    assert " ".join(" ".join(lines).split()) == text
    a, b = (len(l) for l in lines)
    assert min(a, b) / max(a, b) >= 0.8, (a, b)
    # A cue that DOES fit is untouched by the sharing path.
    assert _hard_wrap_lines("alpha bravo charlie delta", 14, 2).split("\n") == [
        "alpha bravo", "charlie delta"]


def test_split_candidates_are_ranked_not_a_single_bet():
    from backend.services.subtitle_formatter import _split_candidates
    # The measured failure: this cue's ONLY match in any category was the
    # conjunction in "report that your" at index 19 of 119. The old single-answer
    # finder returned 19, which strands a sub-minimum piece, so the cue never
    # split and shipped 58 characters over budget.
    text = ("I received a report that your subordinate under Treize Khushrenada "
            "lost three mobile suits during atmospheric re-entry.")
    cands = _split_candidates(text)
    assert len(cands) > 5, cands
    assert 19 in cands                       # the conjunction is still offered…
    # …and a boundary near the centre is offered too, so a rejected top pick
    # falls through to one that actually works.
    assert any(abs(c - len(text) // 2) <= 8 for c in cands), cands
    # Every candidate is a real position inside the text.
    assert all(0 < c < len(text) for c in cands)


def test_over_budget_cue_splits_even_when_cps_is_fine():
    # 96 chars at ~17 CPS: inside the CPS cap, but nearly 30 chars past the
    # 2 x 34 on-screen budget. The CPS early-return used to decline the split
    # while the caller's loop was asking for it precisely because of the budget.
    text = ("Now then, today's agenda is about experimenting with a solidarity "
            "organization between colonies.")
    seg = _cue(100.0, 105.63, text,
               words=_words_for(text, 100.0, 105.63))
    out = enforce_readability([seg], max_chars_per_line=34, max_lines=2,
                              allow_split=True, word_timed_split_only=True)
    assert len(out) >= 2, [c.text for c in out]
    for c in out:
        assert len(" ".join((c.text or "").split())) <= 68, c.text
    # No text invented or lost.
    assert " ".join(" ".join(c.text.split()) for c in out).replace("  ", " ") \
        .count("solidarity") == 1


def test_merge_keeps_both_cues_word_timings():
    from backend.services.subtitle_formatter import _concat_words
    a = _cue(10.0, 10.4, "one two", words=[
        WordTimestamp(start=10.0, end=10.2, word="one"),
        WordTimestamp(start=10.2, end=10.4, word="two")])
    b = _cue(10.45, 10.9, "three four", words=[
        WordTimestamp(start=10.45, end=10.7, word="three"),
        WordTimestamp(start=10.7, end=10.9, word="four")])
    got = _concat_words(a, b, "one two three four")
    assert [w["word"] for w in got] == ["one", "two", "three", "four"]
    assert [w["start"] for w in got] == [10.0, 10.2, 10.45, 10.7]
    # A count that no longer describes the merged text is rejected outright,
    # so the resync pass rebuilds rather than shipping a mismatched array.
    assert _concat_words(a, b, "one two three") == []
    # Non-monotonic input is rejected too — the highlight schedule needs order.
    c = _cue(9.0, 9.4, "zero", words=[
        WordTimestamp(start=9.0, end=9.4, word="zero")])
    assert _concat_words(a, c, "one two zero") == []


def test_readability_returns_cues_whose_word_count_matches_their_text():
    from backend.services.subtitle_formatter import resync_cue_words
    # This is the contract the frontend enforces before it will use word
    # timings at all (activeWordTiming.js requires an exact count match), so a
    # pass that rewrites text without touching words silently kills karaoke.
    text = "One two three four five six seven eight nine ten eleven twelve"
    segs = [_cue(0.0, 4.0, text, words=_words_for(text, 0.0, 4.0)),
            _cue(4.2, 5.0, "short bit", words=_words_for("short bit", 4.2, 5.0))]
    out = enforce_readability(segs, max_chars_per_line=34, max_lines=2,
                             allow_split=True, word_timed_split_only=True)
    for c in out:
        assert len(c.words or []) == len((c.text or "").split()), c.text

    # Stale array from a text rewrite → rebuilt to match.
    seg = _cue(0.0, 2.0, "a rewritten line with more tokens now",
               words=_words_for("original", 0.0, 2.0))
    kept, rebuilt = resync_cue_words([seg])
    assert (kept, rebuilt) == (0, 1)
    assert len(seg.words) == 7
    # Rebuilt rows keep the model's row TYPE, not bare dicts.
    assert all(isinstance(w, WordTimestamp) for w in seg.words)
    # Matching count → real times preserved untouched.
    seg2 = _cue(0.0, 2.0, "two words", words=[
        WordTimestamp(start=0.5, end=0.9, word="two"),
        WordTimestamp(start=1.4, end=1.8, word="words")])
    kept, rebuilt = resync_cue_words([seg2])
    assert (kept, rebuilt) == (1, 0)
    assert [w.start for w in seg2.words] == [0.5, 1.4]


def test_default_line_budget_matches_the_reference_track():
    from backend.config import settings
    # The reference YouTube track's line lengths stop dead at 34; shipping 42
    # is what put our own lines in the 40-44 bucket.
    assert settings.SUBTITLE_MAX_CHARS_PER_LINE == 34


# ── defects found by adversarial review of the word-timing commit ──────────

def test_fabricated_word_rows_are_not_evidence_for_a_time_cut():
    """Rebuilding ``words`` must not promote a word-less cue to "word-timed".

    "Has a words array matching the token count" is what every consumer reads as
    "has real audio times". The resync pass rebuilds that array, so a tier-C cue
    came back looking word-timed — and because the pipeline feeds
    enforce_readability's output back into itself up to four times, the next
    iteration split those cues at guessed midpoints. That is exactly the
    char-proportional time cut ``word_timed_split_only`` exists to forbid; a
    12 s cue cascaded 1 -> 2 -> 4.
    """
    from backend.services.subtitle_formatter import (
        _word_timed_midpoint, _word_gap_split_point,
    )
    text = ("This is a long tier C cue with no word timings at all and it "
            "should stay whole rather than be cut at a guessed midpoint")
    kw = dict(max_cps=17.0, max_chars_per_line=34, max_lines=2,
              min_duration_ms=833, max_duration_ms=7000,
              allow_split=True, word_timed_split_only=True)
    first = enforce_readability(
        [_cue(0.0, 12.0, text, words=None)], **kw)
    assert all(c.words_synthetic for c in first)
    # The split helpers must refuse the fabricated array…
    assert _word_timed_midpoint(first[0], 40) is None
    assert _word_gap_split_point(first[0]) is None
    # …so a second pass cannot cascade.
    again = enforce_readability([c.model_copy(deep=True) for c in first], **kw)
    assert len(again) == len(first), [c.text for c in again]


def test_real_word_times_are_never_marked_synthetic():
    text = "One two three four five"
    out = enforce_readability(
        [_cue(0.0, 2.5, text, words=_words_for(text, 0.0, 2.5))],
        max_chars_per_line=34, max_lines=2,
        allow_split=True, word_timed_split_only=True)
    assert not any(c.words_synthetic for c in out)


def test_over_budget_predicate_is_shared_by_loop_and_splitter():
    """The two used separate expressions and disagreed on the no-legal-wrap
    case: the loop asked for a split, the splitter saw a cue inside the
    character budget and returned it unchanged, and the cue shipped with an
    over-long line — the very failure the budget test was added to prevent."""
    from backend.services.subtitle_formatter import (
        _text_over_budget, _balanced_two_line,
    )
    # Inside 2x34=68 chars, so the arithmetic budget says it fits — but one
    # 33-char token straddles every boundary, so no split leaves BOTH halves
    # under the per-line cap. This is the case the two predicates disagreed on.
    flat = "a counterintelligencereconnaissance apparatus"
    assert len(flat) <= 68
    assert _balanced_two_line(flat, 34) is None      # genuinely unwrappable
    assert _text_over_budget(flat, 34, 2) is True
    # Fits one line -> never over budget.
    assert _text_over_budget("short line", 34, 2) is False
    # Past the box entirely.
    assert _text_over_budget("x" * 80, 34, 2) is True


def test_concat_words_rejects_an_end_overrun():
    """Start-ordering is not a usable schedule.

    The ASS export anchors each word to the PREVIOUS word's end, so an array
    where a word's end overruns the next word's start renders the karaoke out of
    order — measured: word 4 highlighted before word 3, then jumping back.
    """
    from backend.services.subtitle_formatter import _concat_words
    a = _cue(11.5, 12.0, "Not now", words=[
        WordTimestamp(start=11.52, end=11.70, word="Not"),
        WordTimestamp(start=11.70, end=12.28, word="now")])   # end past 12.05
    b = _cue(12.05, 13.5, "and get down", words=[
        WordTimestamp(start=12.10, end=12.22, word="and"),
        WordTimestamp(start=12.22, end=12.60, word="get"),
        WordTimestamp(start=12.60, end=13.50, word="down")])
    assert _concat_words(a, b, "Not now and get down") == []
    # A clean, non-overlapping pair still concatenates.
    a2 = _cue(11.5, 12.0, "Not now", words=[
        WordTimestamp(start=11.52, end=11.70, word="Not"),
        WordTimestamp(start=11.70, end=11.99, word="now")])
    assert len(_concat_words(a2, b, "Not now and get down")) == 5
