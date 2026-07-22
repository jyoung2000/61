"""Hybrid word-timing projection (subtitle_aligner) + word-timed splitting.

Tier A = Whisper-EN audio projection (real per-word times). Tier B and Tier C
both char-weight-distribute the cue's OWN [start,end] window across its English
tokens (B for multi-token cues, C for the single-token remainder) so no dialogue
cue ships word-less; bracketed non-speech markers are never filled. The LLM text
is always authoritative — Whisper-EN supplies timing only.
"""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.subtitle_aligner import (
    project_word_timings, attach_source_pause_timings, flatten_whisper_words,
    project_hybrid_timings,
)
from backend.services.subtitle_formatter import enforce_readability


def _w(spec):
    return [WordTimestamp(start=s, end=e, word=w) for s, e, w in spec]


def _cue(start, end, text, speaker="Speaker 1", words=None):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker, words=words)


def _whisper_ref():
    # Whisper-EN translation with a 0.7s pause after "everything".
    return [_cue(10.0, 14.0, "i will destroy everything then i will leave", words=_w([
        (10.0, 10.4, "I"), (10.4, 10.9, "will"), (10.9, 11.6, "destroy"),
        (11.6, 12.3, "everything"),
        (13.0, 13.2, "then"), (13.2, 13.5, "I"), (13.5, 13.7, "will"),
        (13.7, 14.0, "leave"),
    ]))]


# ── Tier A: Whisper-EN projection ──

def test_tier_a_projects_and_keeps_llm_text():
    llm = [_cue(10.0, 14.0, "I will destroy everything and then I will leave")]
    cues, n = project_word_timings(llm, flatten_whisper_words(_whisper_ref()))
    assert n == 1
    assert cues[0].text == "I will destroy everything and then I will leave"  # LLM text intact
    assert cues[0].words and len(cues[0].words) == 9
    # Matched anchors carry the real Whisper times.
    by_word = {w.word: (w.start, w.end) for w in cues[0].words}
    assert by_word["destroy"] == (10.9, 11.6)
    assert by_word["leave"][1] == 14.0
    # Monotonic.
    ws = cues[0].words
    for i in range(len(ws) - 1):
        assert ws[i].end <= ws[i + 1].start + 1e-9


def test_tier_a_divergent_wording_interpolates_unmatched():
    # The LLM wording diverges ("annihilate" vs Whisper "destroy") — unmatched
    # tokens must be interpolated, not dropped, and text preserved.
    llm = [_cue(10.0, 14.0, "I shall annihilate it all then depart")]
    cues, n = project_word_timings(llm, flatten_whisper_words(_whisper_ref()),
                                   min_anchor_ratio=0.1)
    assert n == 1
    assert cues[0].text == "I shall annihilate it all then depart"
    assert len(cues[0].words) == len("I shall annihilate it all then depart".split())
    ws = cues[0].words
    for i in range(len(ws) - 1):
        assert ws[i].end <= ws[i + 1].start + 1e-9
    assert ws[0].start >= 10.0 - 1e-9 and ws[-1].end <= 14.0 + 1e-9


def test_tier_a_below_anchor_ratio_keeps_whole():
    # No shared tokens at all → cannot trust projection → cue stays word-less.
    llm = [_cue(10.0, 14.0, "completely different sentence content goes here now")]
    cues, n = project_word_timings(llm, flatten_whisper_words(_whisper_ref()),
                                   min_anchor_ratio=0.5)
    assert n == 0
    assert not cues[0].words


def test_no_whisper_words_is_noop():
    llm = [_cue(10.0, 14.0, "Some text here")]
    cues, n = project_word_timings(llm, [])
    assert n == 0 and not cues[0].words


# ── Tier B: cue-window char-weight distribution ──

def test_tier_b_distributes_cue_window_over_english():
    # English cue is word-less; Tier B distributes the cue's OWN [20,24] window
    # across the English tokens (source words are NO LONGER consulted — mapping
    # English onto Japanese SOV positions put the highlight on the wrong word).
    llm = [_cue(20.0, 24.0, "I will destroy it and then go")]
    cues, n = attach_source_pause_timings(llm, None)
    assert n == 1
    assert cues[0].text == "I will destroy it and then go"  # English intact
    ws = cues[0].words
    assert ws and len(ws) == len("I will destroy it and then go".split())
    for i in range(len(ws) - 1):
        assert ws[i].end <= ws[i + 1].start + 1e-9
    # Times stay within the cue span and span it fully.
    assert ws[0].start >= 20.0 - 1e-9 and ws[-1].end <= 24.0 + 1e-9
    assert abs(ws[0].start - 20.0) < 1e-6 and abs(ws[-1].end - 24.0) < 1e-6


def test_tier_b_is_source_independent():
    # No source words at all no longer blocks Tier B — it fills from the cue's
    # own window (the old positional projection skipped this case).
    llm = [_cue(20.0, 24.0, "English here without source timing")]
    cues, n = attach_source_pause_timings(llm, [_cue(20.0, 24.0, "ジャ", words=None)])
    assert n == 1
    assert cues[0].words and len(cues[0].words) == 5


def test_tier_b_skips_when_window_invalid():
    # Zero-width window → nothing to distribute → left word-less.
    llm = [_cue(20.0, 20.0, "zero width window here")]
    cues, n = attach_source_pause_timings(llm, None)
    assert n == 0 and not cues[0].words


def test_tier_b_skips_bracket_markers():
    # A [♪ music ♪] marker must never get karaoke word timing.
    llm = [_cue(20.0, 24.0, "[♪ music ♪]")]
    cues, n = attach_source_pause_timings(llm, None)
    assert n == 0 and not cues[0].words


# ── A→B→C ladder + idempotency ──

def test_ladder_a_then_b_then_c():
    # cue0 → tier A (whisper match); cue1 → tier B (multi-token, no whisper
    # match); cue2 → tier C (single token: tier B's min_tokens=2 skips it, tier
    # C fills the remainder). source_cues is None — Tier B is source-independent.
    llm = [
        _cue(10.0, 14.0, "I will destroy everything and then I will leave"),
        _cue(20.0, 24.0, "I will destroy it and then go"),
        _cue(30.0, 33.0, "Untranslatable"),
    ]
    tiers = project_hybrid_timings(
        llm, whisper_en_segments=_whisper_ref(), source_cues=None, min_anchor_ratio=0.3)
    assert tiers["tier_a"] == 1
    assert tiers["tier_b"] == 1
    assert tiers["tier_c"] == 1
    assert tiers["total"] == 3
    # Every dialogue cue now carries a timing skeleton (no word-less shipping).
    assert llm[0].words and llm[1].words and llm[2].words


def test_idempotent_does_not_reproject():
    llm = [_cue(10.0, 14.0, "I will destroy everything and then I will leave")]
    project_word_timings(llm, flatten_whisper_words(_whisper_ref()))
    first = [(w.word, w.start, w.end) for w in llm[0].words]
    # Re-running must be a no-op (cue already has words).
    _, n2 = project_word_timings(llm, flatten_whisper_words(_whisper_ref()))
    assert n2 == 0
    assert [(w.word, w.start, w.end) for w in llm[0].words] == first


# ── Integration: projected words let the readability splitter break a run-on ──

def test_projected_words_enable_word_timed_split():
    llm = [_cue(10.0, 14.0, "I will destroy everything and then I will leave")]
    project_word_timings(llm, flatten_whisper_words(_whisper_ref()))
    out = enforce_readability(
        llm, max_cps=8.0, max_duration_ms=2500,
        allow_split=True, word_timed_split_only=True)
    # The run-on splits into 2+ cues at the real pause; text round-trips.
    assert len(out) >= 2
    joined = " ".join(s.text.replace("\n", " ") for s in out).split()
    assert joined == "I will destroy everything and then I will leave".split()
    # Chronological + within the original span.
    for i in range(len(out) - 1):
        assert out[i].start <= out[i].end <= out[i + 1].start + 1e-9


def test_wordless_cue_kept_whole_under_word_timed_split_only():
    # A word-less LLM cue must never be char-proportionally split (tier C).
    cue = _cue(0.0, 10.0, "this is a very long run on line with no word timing at all here")
    out = enforce_readability(
        [cue], max_cps=5.0, max_duration_ms=2000,
        allow_split=True, word_timed_split_only=True)
    assert len(out) == 1  # kept whole, not scrambled
    assert out[0].text.replace("\n", " ") == cue.text


def test_marker_preserved_and_not_projected():
    # A [♪ music ♪] marker has no words and must survive untouched.
    llm = [
        _cue(0.0, 3.0, "[♪ music ♪]"),
        _cue(10.0, 14.0, "I will destroy everything and then I will leave"),
    ]
    project_hybrid_timings(llm, whisper_en_segments=_whisper_ref(), source_cues=None)
    assert llm[0].text == "[♪ music ♪]"
    assert not llm[0].words  # marker never gets per-word (karaoke) timing
    out = enforce_readability(llm, allow_split=True, word_timed_split_only=True)
    assert any(s.text == "[♪ music ♪]" for s in out)


def test_no_whisper_text_leaks_into_output():
    # LLM text uses words the Whisper-EN reference never contained.
    llm = [_cue(10.0, 14.0, "I shall annihilate the whole place then vanish")]
    project_word_timings(llm, flatten_whisper_words(_whisper_ref()), min_anchor_ratio=0.1)
    out = enforce_readability(llm, allow_split=True, word_timed_split_only=True)
    text = " ".join(s.text for s in out).lower()
    # Whisper-only words must not appear.
    assert "destroy" not in text and "leave" not in text and "everything" not in text


# ── Merge pause cap (step 5): no run-on across a large source pause ──

def test_merge_hard_cap_blocks_large_pause():
    from backend.services.subtitle_formatter import _merge_for_readability
    # Two mid-sentence fragments 5s apart. Without a hard cap the 6s sentence
    # bridge would merge them; the 4s hard cap treats the silence as a boundary.
    segs = [_cue(0.0, 1.0, "and it is your"), _cue(6.0, 7.0, "problem to solve")]
    merged_no_cap = _merge_for_readability(
        segs, 20.0, 42, 2, 9.0, 1.2, sentence_gap_s=6.0, hard_max_gap_s=None)
    assert len(merged_no_cap) == 1   # legacy: 6s bridge merges them
    merged_capped = _merge_for_readability(
        segs, 20.0, 42, 2, 9.0, 1.2, sentence_gap_s=6.0, hard_max_gap_s=4.0)
    assert len(merged_capped) == 2   # 5s gap > 4s cap → not merged
