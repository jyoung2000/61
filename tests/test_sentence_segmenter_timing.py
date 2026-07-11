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


def test_resegment_scrambles_wordless_1to1_cues():
    """Why ``_background_post_processing`` skips resegment on the LLM path.

    The editorial-LLM translator emits clean, one-utterance-per-cue segments
    that inherit the source's accurate (word-timed) start/end — but carry NO
    word timestamps of their own. ``resegment_by_sentence`` MERGES same-speaker
    neighbours into one block then re-splits by sentence; with no words to time
    the cut it falls back to a GLOBAL char-proportional split across the whole
    merged span, erasing the real per-cue timing and the silence between cues.

    Here three equal-length, same-speaker cues have a 56 s silent gap before
    the last one (a scene change at 60 s). Historically resegment merged all
    three into one [0, 62] block and char-split it, erasing the gap and
    pulling the 60 s cue to ~41 s — the "scrambled timing" that made the LLM
    path bypass this step. The merge is now GAP-GATED (same-speaker merges
    never bridge a turn-length silence), so the 56 s gap and the third cue's
    true 60 s start SURVIVE resegmentation. The LLM path's bypass remains a
    belt-and-braces choice, but the scramble it protected against is fixed.
    """
    segs = [
        TranscriptSegment(text="AAAA.", start=0.0, end=2.0, speaker="S", words=None),
        TranscriptSegment(text="BBBB.", start=2.0, end=4.0, speaker="S", words=None),
        TranscriptSegment(text="CCCC.", start=60.0, end=62.0, speaker="S", words=None),
    ]
    # Original timeline has a 56 s gap (4 s → 60 s) before the last cue.
    assert (segs[2].start - segs[1].end) == 56.0

    out = resegment_by_sentence(segs)

    # The 56 s gap survives: the merge refuses to bridge a turn-length
    # silence, so no char-proportional split can smear timing across it.
    gaps = [out[i + 1].start - out[i].end for i in range(len(out) - 1)]
    assert max(gaps) >= 56.0, f"gap-gated merge lost the 56 s silence: {gaps}"

    # And the third utterance keeps its REAL 60 s start.
    third = next(s for s in out if "CCCC" in s.text)
    assert third.start == 60.0, (
        f"the 60 s cue moved to {third.start:.1f}s — merge bridged a "
        "turn-length gap it must never bridge")
