"""Hybrid word-timing projection (subtitle_aligner) + word-timed splitting.

Tier A = Whisper-EN audio projection (real per-word times on lexically-matched
cues). Tier B/C give the rest a timing skeleton: placed on the REAL Whisper-EN
voiced timeline (speech onset/offset + pauses) when a reference word overlaps the
cue, else char-weight-distributed across the cue's OWN [start,end] window (B for
multi-token cues, C for the single-token remainder). No dialogue cue ships
word-less; bracketed non-speech markers are never filled. The LLM text is always
authoritative — Whisper-EN supplies timing only.
"""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.subtitle_aligner import (
    project_word_timings, attach_source_pause_timings, flatten_whisper_words,
    project_hybrid_timings, snap_cue_windows_to_reference,
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


# ── Tier B/C: audio anchoring on the real Whisper-EN voiced timeline ──

def test_tier_b_time_anchors_to_reference_span():
    # The cue WINDOW is padded (9-15) but the real speech (Whisper-EN) is 10-14.
    # With the reference stream, the highlight must sit on the REAL voiced span —
    # NOT sweep uniformly through the 1s of leading/trailing silence (which is
    # exactly how the highlight drifted off the spoken word).
    ref = flatten_whisper_words(_whisper_ref())  # words span 10.0..14.0
    llm = [_cue(9.0, 15.0, "I will destroy everything then leave")]
    cues, n = attach_source_pause_timings(llm, None, whisper_en_words=ref)
    assert n == 1
    ws = cues[0].words
    assert ws and len(ws) == len("I will destroy everything then leave".split())
    # First word onsets at real speech (~10.0), not the padded window (9.0).
    assert ws[0].start >= 10.0 - 1e-6
    # Last word ends at real speech offset (~14.0), not the padded window (15.0).
    assert ws[-1].end <= 14.0 + 1e-6
    # Monotonic and inside the window.
    for i in range(len(ws) - 1):
        assert ws[i].end <= ws[i + 1].start + 1e-9
    assert ws[0].start >= 9.0 - 1e-9 and ws[-1].end <= 15.0 + 1e-9


def test_tier_b_no_overlapping_reference_falls_back_to_window():
    # Reference words are elsewhere (10-14); cue is at 20-24 → no overlap → the
    # plain window distribution (spans the full cue) is used, unchanged.
    ref = flatten_whisper_words(_whisper_ref())
    llm = [_cue(20.0, 24.0, "completely elsewhere in time here")]
    cues, n = attach_source_pause_timings(llm, None, whisper_en_words=ref)
    assert n == 1
    ws = cues[0].words
    assert abs(ws[0].start - 20.0) < 1e-6 and abs(ws[-1].end - 24.0) < 1e-6


def test_distribute_over_reference_monotonic_and_clamped():
    from backend.services.subtitle_aligner import distribute_over_reference
    ref = [(10.0, 10.4), (10.4, 10.9), (11.6, 12.3), (13.0, 13.5)]
    toks = "a bb ccc dddd eeeee".split()
    out = distribute_over_reference(toks, ref, 9.0, 15.0)
    assert len(out) == len(toks)
    assert out[0]["start"] >= 10.0 - 1e-9      # real onset
    assert out[-1]["end"] <= 13.5 + 1e-9       # real offset (last ref end)
    prev = 9.0
    for w in out:
        assert w["start"] >= prev - 1e-9
        assert w["end"] >= w["start"] - 1e-9
        prev = w["end"]


def test_distribute_over_reference_empty_ref_falls_back():
    from backend.services.subtitle_aligner import distribute_over_reference
    toks = "one two three".split()
    out = distribute_over_reference(toks, [], 5.0, 8.0)
    assert len(out) == 3
    assert abs(out[0]["start"] - 5.0) < 1e-6 and abs(out[-1]["end"] - 8.0) < 1e-6


def test_hybrid_reports_ref_anchored_and_places_on_audio():
    # cue0 → tier A (lexical match at 10-14); cue1 → tier B, PADDED window (9-15)
    # overlapping the reference → audio-anchored onto the real 10-14 voiced span.
    llm = [
        _cue(10.0, 14.0, "I will destroy everything and then I will leave"),
        _cue(9.0, 15.0, "wholly different wording that shares the window"),
    ]
    tiers = project_hybrid_timings(
        llm, whisper_en_segments=_whisper_ref(), source_cues=None)
    assert tiers["tier_a"] == 1
    assert tiers["tier_b"] == 1
    assert tiers["ref_anchored"] >= 1
    ws = llm[1].words
    # Audio-anchored: words sit on the real voiced span (10-14), not the padded
    # 9-15 window a uniform char-sweep would have used.
    assert ws[0].start >= 10.0 - 1e-6 and ws[-1].end <= 14.0 + 1e-6


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

# ── Cue-onset snap: pull display window toward real Whisper-EN speech ──

_SNAP_KW = dict(pad_threshold=0.4, max_shift=1.5, lead_in=0.1, min_gap=0.12,
                min_dur_s=0.833, max_cps=17.0)


def _ref_stream():
    # Real speech spans 11.0..13.5 inside a padded window.
    return flatten_whisper_words([_cue(11.0, 13.5, "the speech happens here now", words=_w([
        (11.0, 11.4, "the"), (11.4, 12.0, "speech"), (12.0, 12.6, "happens"),
        (12.6, 13.0, "here"), (13.0, 13.5, "now"),
    ]))])


def test_snap_trims_leading_and_trailing_pad():
    cue = _cue(9.0, 15.0, "the speech happens here now")
    n = snap_cue_windows_to_reference([cue], _ref_stream(), **_SNAP_KW)
    assert n == 1
    # Start pulled toward onset (11.0) minus lead-in, bounded by max_shift.
    assert cue.start >= 9.0 and cue.start <= 11.0
    assert cue.start > 9.0                      # leading pad trimmed
    # End pulled toward offset (13.5), never later than the source end.
    assert cue.end <= 15.0 and cue.end >= 13.5 - 1e-6
    assert cue.end < 15.0                       # trailing pad trimmed
    assert cue.end - cue.start >= 0.833 - 1e-6


def test_snap_is_inward_only():
    # Speech starts BEFORE the window (window is late) -> start must not move
    # earlier (inward-only never expands).
    cue = _cue(12.2, 16.0, "the speech happens here now")
    snap_cue_windows_to_reference([cue], _ref_stream(), **_SNAP_KW)
    assert cue.start >= 12.2 - 1e-6             # never moved earlier than source start


def test_snap_pad_below_threshold_noop():
    cue = _cue(10.9, 13.7, "the speech happens here now")   # <0.4s pad each side
    n = snap_cue_windows_to_reference([cue], _ref_stream(), **_SNAP_KW)
    assert n == 0 and cue.start == 10.9 and cue.end == 13.7


def test_snap_no_ref_overlap_noop():
    cue = _cue(40.0, 44.0, "elsewhere entirely on the timeline")
    n = snap_cue_windows_to_reference([cue], _ref_stream(), **_SNAP_KW)
    assert n == 0 and cue.start == 40.0 and cue.end == 44.0


def test_snap_skips_markers():
    cue = _cue(9.0, 15.0, "[♪ music ♪]")
    n = snap_cue_windows_to_reference([cue], _ref_stream(), **_SNAP_KW)
    assert n == 0 and cue.start == 9.0 and cue.end == 15.0


def test_snap_respects_min_duration():
    # A very short real-speech island in a wide window: trimming both edges must
    # not shrink the cue below min_dur_s.
    ref = flatten_whisper_words([_cue(11.0, 11.3, "hi", words=_w([(11.0, 11.3, "hi")]))])
    cue = _cue(9.0, 15.0, "hi there")
    snap_cue_windows_to_reference([cue], ref, **_SNAP_KW)
    assert cue.end - cue.start >= 0.833 - 1e-6


def test_snap_reclamps_words_into_window():
    cue = _cue(9.0, 15.0, "the speech happens here now", words=_w([
        (9.2, 9.6, "the"), (11.2, 11.6, "speech"), (12.1, 12.5, "happens"),
        (12.6, 13.0, "here"), (14.4, 14.8, "now"),
    ]))
    snap_cue_windows_to_reference([cue], _ref_stream(), **_SNAP_KW)
    prev = cue.start - 1e-9
    for wd in cue.words:
        assert cue.start - 1e-6 <= wd.start <= cue.end + 1e-6
        assert wd.start >= prev - 1e-6          # monotonic
        prev = wd.end


def test_snap_default_off_in_hybrid():
    # HYBRID_CUE_SNAP_ENABLED defaults False -> project_hybrid_timings must leave
    # every cue.start/end untouched (regression fence for the default).
    llm = [_cue(9.0, 15.0, "the speech happens here now")]
    before = [(c.start, c.end) for c in llm]
    tiers = project_hybrid_timings(
        llm, whisper_en_segments=[_cue(11.0, 13.5, "the speech happens here now",
            words=_w([(11.0, 11.4, "the"), (11.4, 12.0, "speech"),
                      (12.0, 12.6, "happens"), (12.6, 13.0, "here"), (13.0, 13.5, "now")]))],
        source_cues=None)
    assert [(c.start, c.end) for c in llm] == before
    assert tiers.get("cue_snapped", 0) == 0


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
