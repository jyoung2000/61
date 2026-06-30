"""Intra-cue repetition cleanup — repetition WITHIN one translated line that the
cross-cue dedup passes can't see ("I will protect you I will protect you",
"no no no no no"). Conservative: genuine short emphasis is preserved."""

from backend.services.transcript_dedup import (
    _collapse_text_repetition as collapse,
    collapse_intra_cue_repetition,
)


def test_whole_line_phrase_loop_collapses():
    assert collapse("I will protect you I will protect you") == "I will protect you"


def test_phrase_loop_with_punctuation_collapses():
    assert collapse("I will protect you. I will protect you.") == "I will protect you."


def test_phrase_loop_three_times():
    assert collapse("go home go home go home") == "go home"


def test_long_word_run_trims_to_two():
    assert collapse("no no no no no no") == "no no"


def test_legitimate_double_word_preserved():
    assert collapse("No, no.") == "No, no."
    assert collapse("Bye bye") == "Bye bye"
    assert collapse("the the mission is over") == "the the mission is over"


def test_non_repeating_line_untouched():
    assert collapse("We have to go now") == "We have to go now"
    assert collapse("So so good") == "So so good"


def test_empty_and_single_word():
    assert collapse("") == ""
    assert collapse("Hello") == "Hello"


def test_collapse_segments_preserves_markers_and_timing():
    segs = [
        {"text": "Run! Run!", "start": 0.0, "end": 1.0, "speaker": "S1"},
        {"text": "I will go I will go", "start": 1.0, "end": 2.0, "speaker": "S1"},
        {"text": "[♪ music ♪]", "start": 2.0, "end": 3.0, "speaker": ""},
        {"text": "[♪ music ♪]", "start": 3.0, "end": 4.0, "speaker": ""},
    ]
    out, changed = collapse_intra_cue_repetition(segs)
    assert changed == 1
    assert out[0]["text"] == "Run! Run!"          # short emphasis kept
    assert out[1]["text"] == "I will go"           # internal loop collapsed
    assert out[2]["text"] == "[♪ music ♪]"         # markers untouched
    assert out[3]["text"] == "[♪ music ♪]"
    # Timing preserved, cue count preserved (intra-cue edits text only).
    assert len(out) == 4
    assert out[1]["start"] == 1.0 and out[1]["end"] == 2.0


def test_threshold_configurable():
    # With a lower threshold, a 3-run trims; default (4) leaves it.
    assert collapse("ha ha ha", min_word_run=3) == "ha ha"
    assert collapse("ha ha ha") == "ha ha ha"
