"""Near-duplicate variants that escaped the run-43 dedup nets.

Two real defects from the 2026-07-19 Gundam Wing run drove these rules:

  * ``[3:30] … Reporting meteorite impact. The surveillance satellites'
    vision is lacking.`` re-embedded a re-decode of the two PREVIOUS cues —
    the satellite sentence survived because "satellites'" (plural-bare-
    apostrophe) vs "satellite's" (possessive) scored 0.5 < 0.70 without
    plural folding.
  * ``[5:55] Are they trying to escape? There's no way they trying to
    escape? There's no way they could withstand…`` — a 7-word decoder
    overlap-stutter inside ONE cue, invisible to the phrase collapse
    because the unit exceeded max_phrase=6 AND carried a mid-unit "?".
"""

from backend.services.transcript_dedup import (
    _collapse_text_repetition as collapse,
    _content_words,
    drop_repeated_sentences,
)


# ── Content-word folding ─────────────────────────────────────────────


def test_content_words_fold_plural_and_possessive():
    a = _content_words("The surveillance satellite's observation is lacking.")
    b = _content_words("The surveillance satellites' vision is lacking.")
    assert "satellite" in a and "satellite" in b
    # 3-of-4 shared content words → 0.75 ≥ 0.70 similarity.
    assert len(a & b) >= 3


def test_content_words_fold_ies_plural():
    assert "colony" in _content_words("the colonies attacked")
    assert "colony" in _content_words("one colony after another")


def test_content_words_keep_short_and_ss_words():
    # "gas" (len 3) and "boss" (ss) must not be mangled by the s-strip.
    cw = _content_words("the gas boss")
    assert "gas" in cw and "boss" in cw


# ── The run-43 satellite near-dup now drops ──────────────────────────


def test_run43_satellite_variant_drops():
    segs = [
        {"text": "Zechs anomaly. Reporting meteor impact.", "start": 200.0},
        {"text": "The surveillance satellite's observation is lacking.",
         "start": 206.0},
        {"text": "Do you think meteors follow atmospheric entry wave courses? "
                 "The surveillance satellites' vision is lacking.",
         "start": 210.0},
    ]
    out, dropped = drop_repeated_sentences(segs)
    assert dropped == 1
    assert out[2]["text"] == ("Do you think meteors follow atmospheric "
                              "entry wave courses?")
    # The originals are untouched.
    assert out[1]["text"] == ("The surveillance satellite's observation "
                              "is lacking.")


def test_distinct_dialogue_lines_survive():
    # Genuine back-and-forth that shares words must NOT be eaten
    # (dropping real dialogue is worse than keeping a cosmetic dup).
    segs = [
        {"text": "Are we under attack?!", "start": 368.0},
        {"text": "We're under sudden enemy attack!", "start": 378.0},
        {"text": "Enemy attack?! Who's attacking?!", "start": 381.0},
    ]
    out, dropped = drop_repeated_sentences(segs)
    assert dropped == 0
    assert len(out) == 3


# ── The run-43 mid-sentence overlap stutter now collapses ────────────


def test_run43_overlap_stutter_collapses():
    txt = ("Are they trying to escape? There's no way they trying to "
           "escape? There's no way they could withstand that high "
           "temperature… but perhaps they can.")
    assert collapse(txt) == (
        "Are they trying to escape? There's no way they could withstand "
        "that high temperature… but perhaps they can.")


def test_sentence_aligned_restatement_with_tail_preserved():
    # Deliberate emphasis repeats a COMPLETE sentence (unit ends with the
    # terminator) — protected regardless of length. A tail keeps this out
    # of the whole-line periodicity rule, which by long-standing contract
    # collapses exact full-line doubles ("I will protect you. I will
    # protect you." → one copy).
    txt = ("I will destroy every last one of them. "
           "I will destroy every last one of them. Understood?")
    assert collapse(txt) == txt


def test_short_cross_sentence_repeat_still_preserved():
    # The p<5 guard: short units spanning a boundary stay untouched.
    assert collapse("Go home. Go home now.") == "Go home. Go home now."


def test_long_unpunctuated_stutter_collapses():
    # max_phrase now reaches 10-word units with no terminator at all.
    txt = ("he made it all the way to the earth he made it all the way "
           "to the earth")
    assert collapse(txt) == "he made it all the way to the earth"
