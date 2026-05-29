"""Regression tests for slow-speech over-segmentation.

Slow / dramatic narration (e.g. an anime opening monologue) has multi-second
pauses *between* individual words. Whisper emits word timestamps, and the
max-duration splitter used to treat every pause as a split point — shattering
a single sentence into a stream of one-word cues. That destroyed subtitle
readability and per-cue translation quality (a lone word has no phrase to
translate). The ``SUBTITLE_MIN_SPLIT_CHARS`` guard keeps slow speech grouped
into readable phrases instead.
"""

from backend.models import TranscriptSegment, WordTimestamp
from backend.services.subtitle_formatter import enforce_readability


def _slow_sentence_segment():
    """One sentence, ~29 s, words spaced 5-7 s apart (slow narration)."""
    words_spec = [
        (58.0, 58.6, "Humanity"),
        (65.0, 65.5, "born"),
        (67.0, 67.4, "on"),
        (69.0, 69.6, "Earth"),
        (73.0, 73.6, "sought"),
        (77.0, 77.4, "new"),
        (79.0, 79.5, "hope"),
        (82.0, 82.3, "in"),
        (83.0, 83.5, "space"),
        (87.0, 87.8, "colonies."),
    ]
    words = [WordTimestamp(start=s, end=e, word=w) for s, e, w in words_spec]
    text = " ".join(w for _, _, w in words_spec)
    return TranscriptSegment(start=58.0, end=87.8, text=text,
                             speaker="Speaker 1", words=words)


def test_slow_narration_not_shattered_into_single_words():
    seg = _slow_sentence_segment()
    out = enforce_readability([seg], min_split_chars=10)
    # Should NOT degrade to one cue per word (10 words).
    assert len(out) < 6, f"over-segmented into {len(out)} cues"
    # No cue should be a lone word.
    for s in out:
        assert len(s.text.split()) >= 2, f"stranded single-word cue: {s.text!r}"
    # Timing stays monotonic / non-overlapping.
    for i in range(len(out) - 1):
        assert out[i].end <= out[i + 1].start + 1e-6


def test_guard_disabled_restores_legacy_shattering():
    """min_split_chars=0 disables the guard (legacy behaviour) — proves the
    guard is what prevents the shatter."""
    seg = _slow_sentence_segment()
    legacy = enforce_readability([seg], min_split_chars=0)
    guarded = enforce_readability([seg], min_split_chars=10)
    assert len(guarded) < len(legacy)


def test_normal_paced_speech_still_splits():
    """The guard must not stop normal multi-word splitting — a long fast
    sentence should still break into multiple readable cues."""
    # 12 s of dense speech that exceeds the max-duration cap.
    text = ("This is a fairly long sentence that keeps going well past the "
            "maximum subtitle duration and therefore needs to be split into "
            "several readable caption cues for the viewer.")
    words = []
    toks = text.split()
    t = 0.0
    step = 12.0 / len(toks)
    for tok in toks:
        words.append(WordTimestamp(start=round(t, 3), end=round(t + step * 0.8, 3), word=tok))
        t += step
    seg = TranscriptSegment(start=0.0, end=12.0, text=text, speaker="Speaker 1", words=words)
    out = enforce_readability([seg], min_split_chars=10)
    assert len(out) >= 2
    # Every cue carries real phrases, none stranded to a single word.
    assert all(len(s.text.split()) >= 2 for s in out)
