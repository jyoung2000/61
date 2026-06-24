"""Sentence-completion merge — finish a mid-sentence cue into one line.

~42% of translated cues ended mid-sentence on real runs ("and it's your" →
"boobs here."), most within ~6s of their continuation, but the base merge only
bridged 3s so they shipped as two flashes. ``SUBTITLE_SENTENCE_MERGE_GAP_MS``
widens the bridge when the previous cue has no terminal punctuation, while the
duration / 2-line / CPS caps still bound the result.
"""

from backend.services.subtitle_formatter import _merge_for_readability, _ends_sentence
from backend.models import TranscriptSegment as TS


def test_ends_sentence_detection():
    assert _ends_sentence("boobs here.")
    assert _ends_sentence("Really?")
    assert _ends_sentence('He said "go."')      # trailing quote ignored
    assert _ends_sentence("Wait…")
    assert not _ends_sentence("and it's your")
    assert not _ends_sentence("Is it the way")
    assert not _ends_sentence("")


def test_incomplete_cue_bridges_wider_gap():
    # 5s gap: beyond the 3s base bridge, within the 6s sentence bridge. The first
    # cue ends mid-sentence, so the two should merge into one complete line.
    segs = [
        TS(start=10.0, end=11.0, text="and it's your", speaker="Speaker 1"),
        TS(start=16.0, end=17.0, text="boobs here.", speaker="Speaker 1"),
    ]
    out = _merge_for_readability(
        segs, max_cps=20.0, max_chars_per_line=42, max_lines=2,
        max_dur_s=9.0, max_gap_s=3.0, sentence_gap_s=6.0)
    assert len(out) == 1, [s.text for s in out]
    assert out[0].text == "and it's your boobs here."


def test_complete_cue_starts_fresh_across_same_gap():
    # Same 5s gap, but the first cue is a COMPLETE sentence — it should NOT pull
    # the next one in (one-sentence-per-cue is the goal).
    segs = [
        TS(start=10.0, end=11.0, text="I love you.", speaker="Speaker 1"),
        TS(start=16.0, end=17.0, text="Do you know?", speaker="Speaker 1"),
    ]
    out = _merge_for_readability(
        segs, max_cps=20.0, max_chars_per_line=42, max_lines=2,
        max_dur_s=9.0, max_gap_s=3.0, sentence_gap_s=6.0)
    assert len(out) == 2


def test_far_apart_fragments_stay_split():
    # 12s apart — beyond even the sentence bridge (sparse speech). Stays split so
    # we never park one line on screen for 12s.
    segs = [
        TS(start=10.0, end=11.0, text="I've already let", speaker="Speaker 1"),
        TS(start=23.0, end=24.0, text="most of it out.", speaker="Speaker 1"),
    ]
    out = _merge_for_readability(
        segs, max_cps=20.0, max_chars_per_line=42, max_lines=2,
        max_dur_s=9.0, max_gap_s=3.0, sentence_gap_s=6.0)
    assert len(out) == 2


def test_merge_never_crosses_speaker():
    segs = [
        TS(start=10.0, end=11.0, text="and it's your", speaker="Speaker 1"),
        TS(start=13.0, end=14.0, text="boobs here.", speaker="Speaker 2"),
    ]
    out = _merge_for_readability(
        segs, max_cps=20.0, max_chars_per_line=42, max_lines=2,
        max_dur_s=9.0, max_gap_s=3.0, sentence_gap_s=6.0)
    assert len(out) == 2


def test_merge_respects_two_line_budget():
    # Two incomplete halves that together exceed the 2×42 char budget must NOT
    # merge into a 3-line wall.
    long_a = "this is a fairly long first half of a spoken line that runs on"
    long_b = "and here is the long continuation that would overflow two lines"
    segs = [
        TS(start=10.0, end=12.0, text=long_a, speaker="Speaker 1"),
        TS(start=13.0, end=15.0, text=long_b, speaker="Speaker 1"),
    ]
    out = _merge_for_readability(
        segs, max_cps=99.0, max_chars_per_line=42, max_lines=2,
        max_dur_s=9.0, max_gap_s=3.0, sentence_gap_s=6.0)
    assert len(out) == 2
