"""Adversarially-verified cleanup nets from the run-53 transcript audit:
non-lexical grunt cues ("Nn...", "Nnn...") and a stray leading period
(". And you?").

The grunt fix is layered so it fires regardless of config (the polisher's
filler strip defaults OFF): the SOURCE hallucination gate catches an ASCII
grunt early; tidy_punctuation_artifacts strips an inline/whole-cue grunt on
EVERY translated cue; and sanitize_translated_transcript drops a whole-cue
grunt on the way to persist. All three must leave real words untouched.
"""

from backend.services.hallucination_filter import is_boilerplate_hallucination
from backend.services.translator import tidy_punctuation_artifacts
from backend.services.transcript_sanitize import sanitize_translated_transcript


# ── Source-side hallucination gate (ASCII grunts) ────────────────────

def test_source_gate_drops_ascii_grunts():
    for g in ("Nn...", "Nnn...", "nn", "Mmm", "MM", "mmmm"):
        assert is_boilerplate_hallucination(g), g


def test_source_gate_keeps_real_words():
    for w in ("No", "Now", "Nice", "Man", "Mom", "n", "m", "Mister", "Nine"):
        assert not is_boilerplate_hallucination(w), w


# ── tidy (always-on, every translated cue) ───────────────────────────

def test_tidy_strips_inline_grunt():
    # The 15:37 cue — grunt embedded in real dialogue.
    assert tidy_punctuation_artifacts(
        "That's it! Nnn... Hey, hurry up!") == "That's it! Hey, hurry up!"


def test_tidy_empties_a_whole_cue_grunt():
    # 15:26 standalone — emptied here, then dropped downstream.
    assert tidy_punctuation_artifacts("Nn...") == ""


def test_tidy_leaves_real_words_with_n():
    for w in ("Inn is closed.", "Ann arrived.", "Nine lives."):
        assert tidy_punctuation_artifacts(w) == w


def test_tidy_stray_leading_period_but_keeps_ellipsis():
    assert tidy_punctuation_artifacts(". And you?") == "And you?"
    assert tidy_punctuation_artifacts("... and then we left.") == "... and then we left."
    assert tidy_punctuation_artifacts(".hidden") == ".hidden"


# ── sanitize (always-on, before persist) ─────────────────────────────

def test_sanitize_drops_standalone_grunt_cue():
    segs = [
        {"start": 0.0, "end": 3.0, "text": "You're still just a child."},
        {"start": 3.0, "end": 5.0, "text": "Nn..."},
        {"start": 5.0, "end": 8.0, "text": "Don't move!"},
        {"start": 8.0, "end": 10.0, "text": "Mmm"},
    ]
    out, changed = sanitize_translated_transcript(segs, "en")
    texts = [s["text"] for s in out]
    assert changed
    assert "Nn..." not in texts and "Mmm" not in texts
    assert texts == ["You're still just a child.", "Don't move!"]


def test_sanitize_keeps_real_cues_ending_in_m():
    segs = [{"start": 0.0, "end": 2.0, "text": "I hummed a tune."},
            {"start": 2.0, "end": 4.0, "text": "Mom is home."}]
    out, _ = sanitize_translated_transcript(segs, "en")
    assert [s["text"] for s in out] == ["I hummed a tune.", "Mom is home."]
