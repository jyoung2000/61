"""Tests for the TACT confidence-gated phantom-hallucination filter.

A ~79%-silent video produced a transcript flooded with invented Whisper cues
that survived every existing filter:

  * short English fragments over silence — "Don't let" (13×), "So nice" (12×),
    "Hmm." (11×) — not in the boilerplate blocklist, no_speech_prob just under
    the 0.7 clamp, too short for the in-segment repetition test, and
  * long Japanese run-ons re-emitted VERBATIM 10-11× each (decoder loops over
    music / quiet regions).

These pin the two fixes:
  1. ``is_low_confidence_phantom`` reuses the ledger's own low-confidence
     signal (word conf < 0.4) to DROP a cue ONLY when its words are
     overwhelmingly low-confidence AND Whisper itself doubted there was speech
     — so genuine quiet speech is preserved.
  2. ``drop_repetition_loops`` collapses the scattered verbatim repeats
     (run UNCONDITIONALLY in transcribe(), not only on the gap-fill pass).
"""

from backend.services.transcript_dedup import (
    is_low_confidence_phantom,
    drop_repetition_loops,
)


def _w(word, conf):
    return {"word": word, "start": 0.0, "end": 0.1, "confidence": conf}


# ── confidence gate: drops phantoms ───────────────────────────────────────

def test_phantom_low_conf_over_silence_is_dropped():
    # "Don't let" over silence: both words barely-there, Whisper unsure.
    words = [_w("Don't", 0.18), _w("let", 0.22)]
    assert is_low_confidence_phantom(words, no_speech_prob=0.62) is True


def test_phantom_single_low_conf_word_dropped():
    # "Hmm." — a single very-low-confidence token over a non-speech chunk.
    assert is_low_confidence_phantom([_w("Hmm.", 0.15)], no_speech_prob=0.55) is True


# ── confidence gate: protects real speech ─────────────────────────────────

def test_confident_words_kept_even_at_moderate_no_speech():
    # Genuine quiet speech: Whisper transcribes confident words even though
    # no_speech_prob is moderate. Must NOT be flagged.
    words = [_w("I", 0.91), _w("really", 0.88), _w("mean", 0.95), _w("it", 0.9)]
    assert is_low_confidence_phantom(words, no_speech_prob=0.6) is False


def test_low_conf_but_below_no_speech_floor_is_kept():
    # Low-confidence words but Whisper was fairly sure there WAS speech →
    # not a silence phantom; leave it for other filters.
    words = [_w("maybe", 0.2), _w("so", 0.25)]
    assert is_low_confidence_phantom(words, no_speech_prob=0.3) is False


def test_mixed_confidence_not_enough_lowconf_fraction_is_kept():
    # Only half the words are low-confidence → below the 0.8 fraction gate.
    words = [_w("the", 0.2), _w("plan", 0.95), _w("is", 0.93), _w("set", 0.9)]
    assert is_low_confidence_phantom(words, no_speech_prob=0.7) is False


def test_no_words_is_not_phantom():
    assert is_low_confidence_phantom([], no_speech_prob=0.9) is False


def test_gapfill_style_low_no_speech_phantom_caught_when_floor_disabled():
    # Gap-fill keeps only segments BELOW its no_speech threshold, so its
    # invented cues have LOW no_speech_prob. With the no_speech floor disabled
    # (min_no_speech=0.0, the new default) the confidence conjunction still
    # flags them — this is the flood the old 0.5 floor could never catch.
    words = [_w("Don't", 0.2), _w("let", 0.18)]
    assert is_low_confidence_phantom(words, 0.1, min_no_speech=0.0) is True
    # With the old 0.5 floor the same gap-fill phantom slipped through:
    assert is_low_confidence_phantom(words, 0.1, min_no_speech=0.5) is False


def test_words_without_confidences_is_not_phantom():
    words = [{"word": "x", "start": 0, "end": 1}]
    assert is_low_confidence_phantom(words, no_speech_prob=0.9) is False


# ── thresholds + schema flexibility ───────────────────────────────────────

def test_custom_thresholds_respected():
    # Both words < 0.4 so the low-confidence fraction gate is satisfied; the
    # avg (0.39) sits between the two ceilings we test.
    words = [_w("a", 0.39), _w("b", 0.39)]
    assert is_low_confidence_phantom(words, 0.6) is True               # default 0.40
    assert is_low_confidence_phantom(words, 0.6, max_avg_conf=0.35) is False


def test_disabled_via_high_no_speech_floor():
    words = [_w("Don't", 0.18), _w("let", 0.22)]
    # A caller can effectively disable the gate with an unreachable floor.
    assert is_low_confidence_phantom(words, 0.62, min_no_speech=1.1) is False


def test_object_words_supported():
    class _Word:
        def __init__(self, confidence):
            self.confidence = confidence

    assert is_low_confidence_phantom(
        [_Word(0.1), _Word(0.2)], no_speech_prob=0.8) is True


# ── repetition-loop collapse on the real transcript patterns ──────────────

def _d(start, text):
    return {"start_sec": start, "end_sec": start + 2, "text": text}


def test_long_japanese_runon_loop_collapsed_to_one():
    # The exact prod symptom: the same long run-on re-emitted 11× across the
    # timeline. A long line that recurs is a loop → keep exactly one.
    runon = ("焦らなくてもいいんだってだって取りそうじゃんちょっとちょうだい"
             "ちょっとダメダメ?絶対ダメ")
    segs = [_d(i * 51, runon) for i in range(11)]
    kept, dropped = drop_repetition_loops(segs)
    assert len(kept) == 1
    assert dropped == 10


def test_short_english_phantom_repeats_capped():
    # "Don't let" 13× — short fragments are capped (the confidence gate removes
    # them upstream; this is the safety net for any that reach the dedup).
    segs = [_d(i * 30, "Don't let") for i in range(13)]
    kept, dropped = drop_repetition_loops(segs)
    assert len(kept) == 3
    assert dropped == 10


def test_real_dialogue_mix_preserved():
    # Distinct real lines must all survive.
    segs = [_d(0, "What's wrong?"), _d(3, "I'm going to do it!"),
            _d(6, "Oh, my god!")]
    kept, dropped = drop_repetition_loops(segs)
    assert dropped == 0 and len(kept) == 3
