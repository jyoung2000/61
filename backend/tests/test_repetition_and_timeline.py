"""Regression tests for the gap-fill hallucination flood + timeline blowup.

A 24-min anime episode produced a transcript that:
  * repeated the same intro narration / OP-lyric / "了解" lines dozens of
    times (gap-fill re-transcribing music with VAD off → Whisper loops), and
  * spanned 46 minutes — the oversize-cue repair extended those hallucinated
    dumps and the duration-splitter spread them across fabricated start times.

These tests pin the two fixes:
  1. drop_repetition_loops removes scattered exact-text repeats.
  2. enforce_readability's oversize repair is bounded by the next cue's start,
     so it can never inflate the timeline past the real content.
"""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.transcript_dedup import (
    collapse_adjacent_duplicates,
    drop_repetition_loops,
    drop_scattered_duplicates,
    collapse_repeated_runs,
)
from backend.services.subtitle_formatter import enforce_readability


# ── repetition-loop filter ───────────────────────────────────────────────

def _d(start, end, text):
    return {"start_sec": start, "end_sec": end, "text": text}


def test_long_line_repeat_kept_once():
    narration = "しかし、地球圏統一連合は正義と平和の名のもとに圧倒的な軍事力をもって各コロニーを制圧していった。"
    segs = [_d(i * 30, i * 30 + 4, narration) for i in range(20)]
    kept, dropped = drop_repetition_loops(segs)
    assert len(kept) == 1          # long sentence → keep exactly one
    assert dropped == 19


def test_short_interjection_tolerated_then_capped():
    segs = [_d(i, i + 1, "了解") for i in range(10)]
    kept, dropped = drop_repetition_loops(segs)
    assert len(kept) == 3          # short line → keep up to 3
    assert dropped == 7


def test_distinct_lines_all_kept():
    segs = [_d(0, 1, "alpha"), _d(2, 3, "beta"), _d(4, 5, "gamma")]
    kept, dropped = drop_repetition_loops(segs)
    assert dropped == 0
    assert len(kept) == 3


def test_blank_segments_passthrough():
    segs = [_d(0, 1, ""), _d(1, 2, "  ")]
    kept, dropped = drop_repetition_loops(segs)
    assert dropped == 0 and len(kept) == 2


# ── scattered-duplicate collapse (short fragments echoed many times) ─────────

def test_scattered_short_fragment_collapsed_to_one():
    # A SHORT fragment the translator echoed 10× — drop_repetition_loops would
    # cap it at 3 (it's under long_block_chars); the scattered pass takes it to 1.
    segs = [_d(i * 5, i * 5 + 2, "real breasts and you're") for i in range(10)]
    kept, dropped = drop_scattered_duplicates(segs)
    assert len(kept) == 1
    assert dropped == 9


def test_scattered_below_threshold_preserved():
    # A genuine interjection at 3× is below the default threshold (4) → untouched.
    segs = [_d(i, i + 1, "Yes.") for i in range(3)]
    kept, dropped = drop_scattered_duplicates(segs)
    assert dropped == 0 and len(kept) == 3


def test_scattered_markers_exempt():
    # Music markers must never be collapsed, however many times they appear.
    segs = [_d(i * 10, i * 10 + 2, "[♪ music ♪]") for i in range(6)]
    kept, dropped = drop_scattered_duplicates(segs)
    assert dropped == 0 and len(kept) == 6


def test_scattered_distinct_lines_all_kept():
    segs = [_d(0, 1, "alpha"), _d(2, 3, "beta"), _d(4, 5, "gamma")]
    kept, dropped = drop_scattered_duplicates(segs)
    assert dropped == 0 and len(kept) == 3


def test_scattered_keeps_first_occurrence_order():
    segs = [
        _d(0, 1, "keeper one"),
        _d(2, 3, "echo"), _d(4, 5, "echo"), _d(6, 7, "echo"),
        _d(8, 9, "echo"), _d(10, 11, "echo"),
        _d(12, 13, "keeper two"),
    ]
    kept, dropped = drop_scattered_duplicates(segs)
    texts = [k["text"] for k in kept]
    assert texts == ["keeper one", "echo", "keeper two"]
    assert dropped == 4


# ── block / repeated-run collapse (a re-transcribed span shows up twice) ─────

def test_repeated_block_dropped_keeps_first():
    block = ["alpha one", "beta two", "gamma three", "delta four"]
    segs = [_d(i, i + 1, t) for i, t in enumerate(block)]                 # 0-3
    segs += [_d(10, 11, "unrelated middle")]                              # 4
    segs += [_d(20 + i, 21 + i, t) for i, t in enumerate(block)]         # 5-8 (repeat)
    kept, dropped = collapse_repeated_runs(segs)
    assert dropped == 4                                  # the whole second block
    assert [k["text"] for k in kept] == block + ["unrelated middle"]


def test_short_run_below_min_kept():
    # A 2-cue coincidental repeat is below min_run (3) → left alone.
    segs = [_d(0, 1, "ok"), _d(1, 2, "sure"), _d(5, 6, "ok"), _d(6, 7, "sure")]
    kept, dropped = collapse_repeated_runs(segs)
    assert dropped == 0 and len(kept) == 4


def test_repeated_block_distinct_content_untouched():
    segs = [_d(i, i + 1, f"line {i}") for i in range(12)]
    kept, dropped = collapse_repeated_runs(segs)
    assert dropped == 0 and len(kept) == 12


def test_markers_break_a_run():
    # Identical markers shouldn't be treated as a repeated run.
    segs = [_d(i * 5, i * 5 + 1, "[♪ music ♪]") for i in range(8)]
    kept, dropped = collapse_repeated_runs(segs)
    assert dropped == 0 and len(kept) == 8


# ── adjacent-duplicate collapse ──────────────────────────────────────────

def _a(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_back_to_back_long_dup_collapsed_and_end_extended():
    # The exact prod symptom: the same long narration cue emitted twice in a
    # row. drop_repetition_loops would keep only one globally, but the adjacent
    # collapse is what merges the timing into a single clean cue.
    line = "アフターコロニー195年、作戦名オペレーション・メテオ。"
    segs = [_a(685.0, 688.0, line), _a(685.0, 691.0, line)]
    kept, dropped = collapse_adjacent_duplicates(segs)
    assert dropped == 1
    assert len(kept) == 1
    assert kept[0]["end"] == 691.0     # stretched to cover the dropped copy


def test_non_adjacent_dup_not_collapsed_here():
    # Two identical cues separated by a different one are NOT adjacent, so this
    # pass leaves them (drop_repetition_loops handles scattered repeats).
    segs = [_a(0, 1, "x"), _a(1, 2, "y"), _a(2, 3, "x")]
    kept, dropped = collapse_adjacent_duplicates(segs)
    assert dropped == 0 and len(kept) == 3


def test_whitespace_insensitive_adjacent_match():
    segs = [_a(0, 2, "ド ー リ ア ンリ"), _a(2, 4, "ドー リアンリ")]
    kept, dropped = collapse_adjacent_duplicates(segs)
    assert dropped == 1 and len(kept) == 1


def test_distinct_adjacent_lines_kept():
    segs = [_a(0, 1, "alpha"), _a(1, 2, "beta")]
    kept, dropped = collapse_adjacent_duplicates(segs)
    assert dropped == 0 and len(kept) == 2


# ── oversize repair is bounded (no timeline inflation) ────────────────────

def _seg(start, end, text):
    return TranscriptSegment(start=start, end=end, text=text, speaker="Speaker 1")


def test_oversize_repair_never_overruns_next_cue():
    # A giant hallucination dump (no word timing) with a tiny duration sitting
    # just before a real cue. The repair must NOT push past the next cue's
    # start, and no resulting cue may start beyond the last real timestamp.
    dump = "い" * 400   # ~400 CJK chars, would "need" ~30s at 13 cps
    segs = [
        _seg(1400.0, 1400.8, dump),     # corrupt giant cue near the end
        _seg(1405.0, 1408.0, "本物の台詞。"),  # the next real cue
    ]
    out = enforce_readability(segs)
    last_end = max(s.end for s in out)
    # The whole video is ~1408s; the timeline must not blow past it by much.
    assert last_end <= 1410.0, f"timeline inflated to {last_end:.1f}s"
    # No cue starts after the last real content.
    assert all(s.start <= 1408.5 for s in out)


def test_oversize_repair_still_splits_corrupt_block_when_gap_exists():
    # A giant block at the very end with plenty of idle time after it — the
    # repair may extend (bounded) and the splitter breaks it into >1 cue.
    dump = "の" * 300
    out = enforce_readability([_seg(10.0, 10.8, dump)])
    assert len(out) >= 2
    # Still bounded by the hard cap (~30s) — not minutes.
    assert max(s.end for s in out) <= 10.8 + 35.0


# ── word-boundary splitting (no mid-word breaks) ─────────────────────────

def test_find_split_point_lands_on_word_boundary_not_mid_word():
    # Regression: text.index(words[mid]) returned the first SUBSTRING match, so
    # the middle word "is" matched inside "th[is]" → "You said th" + "is is …".
    from backend.services.subtitle_formatter import _find_split_point
    t = "You said this is delicious cake"
    idx = _find_split_point(t)
    assert idx is not None
    # No characters lost and both sides are whole words from the original.
    assert (t[:idx] + t[idx:]) == t
    src_words = t.split()
    assert all(w in src_words for w in t[:idx].split())
    assert all(w in src_words for w in t[idx:].split())


# ── greedy phrase merge (fewer, fuller, readable cues) ───────────────────

def _ts(start, end, text, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def test_merge_combines_short_same_speaker_fragments():
    from backend.services.subtitle_formatter import _merge_for_readability
    segs = [_ts(0.0, 1.0, "So am I planning on"),
            _ts(1.1, 2.0, "eating with"),
            _ts(2.1, 3.0, "you today too")]
    out = _merge_for_readability(segs, max_cps=20.0, max_chars_per_line=42,
                                 max_lines=2, max_dur_s=4.5, max_gap_s=1.2)
    assert len(out) == 1
    assert out[0].text == "So am I planning on eating with you today too"
    assert out[0].start == 0.0 and out[0].end == 3.0


def test_merge_stops_at_speaker_change():
    from backend.services.subtitle_formatter import _merge_for_readability
    segs = [_ts(0.0, 1.0, "Hello there", "Speaker 1"),
            _ts(1.1, 2.0, "Hi back", "Speaker 2")]
    out = _merge_for_readability(segs, 20.0, 42, 2, 4.5, 1.2)
    assert len(out) == 2


def test_merge_stops_at_large_gap():
    from backend.services.subtitle_formatter import _merge_for_readability
    segs = [_ts(0.0, 1.0, "First thought"), _ts(5.0, 6.0, "Much later")]
    out = _merge_for_readability(segs, 20.0, 42, 2, 4.5, 1.2)
    assert len(out) == 2


def test_merge_respects_max_duration_and_cps_and_markers():
    from backend.services.subtitle_formatter import _merge_for_readability
    # Would exceed 4.5s → not merged.
    a = _merge_for_readability([_ts(0.0, 3.0, "Aaa"), _ts(3.1, 6.5, "Bbb")],
                               20.0, 42, 2, 4.5, 1.2)
    assert len(a) == 2
    # Merge would read at ~27 CPS — past even the fragment-completion
    # overdraft (20 × 1.15 = 23) → not merged.
    b = _merge_for_readability([_ts(0.0, 0.42, "abcdefghijk"),
                                _ts(0.42, 0.84, "lmnopqrstuv")],
                               20.0, 42, 2, 4.5, 1.2)
    assert len(b) == 2
    # COMPLETING an unfinished fragment may overdraw the cap slightly
    # (22 CPS ≤ 23): a chopped sentence reads worse than a shade-fast cue.
    b2 = _merge_for_readability([_ts(0.0, 0.5, "abcdefghijk"),
                                 _ts(0.5, 1.0, "lmnopqrstuv")],
                                20.0, 42, 2, 4.5, 1.2)
    assert len(b2) == 1
    # A [♪ music ♪] marker never merges into dialogue.
    c = _merge_for_readability([_ts(0.0, 1.0, "[♪ music ♪]"),
                                _ts(1.1, 2.0, "Hello")],
                               20.0, 42, 2, 4.5, 1.2)
    assert len(c) == 2


def test_enforce_readability_merges_choppy_fragments_into_fewer_cues():
    # End-to-end: choppy fragments collapse to a complete, readable caption.
    segs = [_ts(0.0, 1.0, "So am I planning on"),
            _ts(1.1, 2.0, "eating with"),
            _ts(2.1, 3.0, "you today too")]
    out = enforce_readability(segs)
    assert len(out) <= 2
    words = " ".join(p.text.replace("\n", " ") for p in out).split()
    assert words == "So am I planning on eating with you today too".split()


def test_enforce_readability_never_breaks_a_word_on_wordless_cue():
    # Short duration → over-CPS → forces the char-proportional splitter to
    # recurse into the word-boundary fallback (the mid-word bug site). No
    # emitted token may be a broken word fragment (e.g. "th" from "this").
    seg = _seg(0.0, 2.0,
               "You said this is delicious cake and I ate all of it too")
    out = enforce_readability([seg])
    assert len(out) >= 2  # it did split
    tokens = " ".join(p.text.replace("\n", " ") for p in out).split()
    src = set(seg.text.split())
    assert all(tok.strip(".,!?;:") in src for tok in tokens), tokens

