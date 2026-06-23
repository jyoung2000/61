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

