"""Tests that sentence resegmentation honours Whisper's word timestamps.

Whisper-native emits a few long, multi-sentence cues; the pipeline now carries
their word timestamps through so the split lands at accurate boundaries instead
of the char-length proportional guess. These pin that behaviour.
"""

from __future__ import annotations

from backend.models import TranscriptSegment
from backend.services.sentence_segmenter import resegment_by_sentence


# One cue, two sentences with a DELIBERATE mismatch between text length and
# duration: sentence 1 is tiny ("Hi.") but occupies 0–30 s; sentence 2 is long
# but only 30–40 s. A char-length proportional split cuts at ~3 s; the correct
# (word-timed) cut is at 30 s.
_WORDS = [
    {"word": "Hi.", "start": 0.0, "end": 30.0},
    {"word": "This", "start": 30.0, "end": 31.0},
    {"word": "is", "start": 31.0, "end": 32.0},
    {"word": "a", "start": 32.0, "end": 33.0},
    {"word": "much", "start": 33.0, "end": 34.0},
    {"word": "longer", "start": 34.0, "end": 35.0},
    {"word": "sentence", "start": 35.0, "end": 37.0},
    {"word": "here.", "start": 37.0, "end": 40.0},
]
_TEXT = "Hi. This is a much longer sentence here."


def test_resegment_uses_word_timing_when_present():
    seg = TranscriptSegment(text=_TEXT, start=0.0, end=40.0, speaker="S",
                            words=_WORDS)
    out = resegment_by_sentence([seg])
    assert len(out) == 2
    # The second sentence must start at its first word's time (~30 s), NOT the
    # char-proportional ~3 s — this is the whole point of carrying words through.
    assert out[1].start >= 29.0, f"expected ~30s, got {out[1].start}"
    assert out[0].end >= 29.0


def test_resegment_without_words_is_proportional():
    # Same cue, no word timing → the only option is a char-length split, which
    # lands the boundary far too early. Documents the fallback we now avoid for
    # Whisper-native by preserving words.
    seg = TranscriptSegment(text=_TEXT, start=0.0, end=40.0, speaker="S",
                            words=None)
    out = resegment_by_sentence([seg])
    assert len(out) == 2
    assert out[1].start < 15.0, (
        f"proportional split should cut early, got {out[1].start}")
