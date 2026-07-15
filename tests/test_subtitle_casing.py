"""Deterministic sentence-start casing net (2026-07-15).

When the polish/translate post-edit is skipped or times out the RAW draft ships
(the measured Gundam Wing run: the 12B post-edit timed out → every cue shipped
unpolished). Those cues carry lowercase sentence starts and a bare lowercase
"i". ``fix_subtitle_casing`` restores them WITHOUT a model — and, critically,
WITHOUT capitalizing a continuation cue, because subtitles routinely split one
sentence across two cues ("…my passionate," / "undying feelings").
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.translator import (  # noqa: E402
    fix_subtitle_casing,
    _prev_cue_ends_sentence,
    _cap_first_alpha,
)

EN = "en"


# ── Sentence-start capitalization ───────────────────────────────────────────

def test_first_cue_is_capitalized():
    assert fix_subtitle_casing(["just wild beat communication"], EN) == [
        "Just wild beat communication"]


def test_new_sentence_after_terminator_is_capitalized():
    out = fix_subtitle_casing(["I made it.", "the enemy is here."], EN)
    assert out == ["I made it.", "The enemy is here."]


def test_continuation_after_comma_is_NOT_capitalized():
    # The exact split from the Gundam opening: cue 2 continues cue 1's sentence.
    out = fix_subtitle_casing(
        ["I want to show my passionate,", "undying feelings"], EN)
    assert out == ["I want to show my passionate,", "undying feelings"]


def test_continuation_with_no_end_punctuation_is_NOT_capitalized():
    # First cue capitalizes (it opens the track); the second continues its
    # clause (no terminator) and keeps its lowercase start.
    out = fix_subtitle_casing(["with high hopes, humans", "leave Earth"], EN)
    assert out == ["With high hopes, humans", "leave Earth"]


def test_question_and_bang_start_new_sentences():
    out = fix_subtitle_casing(["what is that?", "look out!", "run away"], EN)
    assert out == ["What is that?", "Look out!", "Run away"]


def test_leading_dash_and_quote_are_skipped():
    out = fix_subtitle_casing(['- yes.', '"the answer."'], EN)
    assert out == ['- Yes.', '"The answer."']


def test_never_lowercases_an_already_capitalized_continuation():
    # The model's own capitalized continuation line is left as-is (we only raise).
    out = fix_subtitle_casing(["I want to show my passionate,", "Undying feelings"], EN)
    assert out == ["I want to show my passionate,", "Undying feelings"]


# ── "I" pronoun ─────────────────────────────────────────────────────────────

def test_lone_i_pronoun_becomes_capital():
    # Cue 1 opens the track (capitalized) AND has its pronoun raised; cue 2 is a
    # continuation (stays lowercase-initial) but its pronoun is still raised.
    out = fix_subtitle_casing(["well, i think so", "and i'll go too"], EN)
    assert out == ["Well, I think so", "and I'll go too"]


def test_i_contractions_capitalized():
    assert fix_subtitle_casing(["i'm here and i've won"], EN) == ["I'm here and I've won"]


def test_ie_abbreviation_not_mangled():
    # "i.e." must stay lowercase — it's not the pronoun.
    assert fix_subtitle_casing(["do it, i.e. now"], EN) == ["Do it, i.e. now"]


def test_i_pronoun_only_for_english_target():
    # Italian "i" is a definite article — never uppercase it.
    assert fix_subtitle_casing(["e i ragazzi vanno"], "it") == ["E i ragazzi vanno"]


# ── Script safety ───────────────────────────────────────────────────────────

def test_cjk_target_is_noop():
    cues = ["こんにちは", "またね"]
    assert fix_subtitle_casing(cues, "ja") == cues


def test_cjk_cue_in_latin_list_is_left_alone():
    # A stray still-source-language cue is never recased and never crashes.
    out = fix_subtitle_casing(["hello there.", "日本語のまま", "next line"], EN)
    assert out[0] == "Hello there."
    assert out[1] == "日本語のまま"
    # The CJK cue ends without a Latin terminator → the following cue is treated
    # as a continuation and left lowercase.
    assert out[2] == "next line"


def test_camelcase_brand_is_preserved():
    assert fix_subtitle_casing(["iPhone sales rose"], EN) == ["iPhone sales rose"]


def test_non_latin_lead_is_left_alone():
    # A Cyrillic-leading cue is not touched (не → leave lead as the model set it).
    assert _cap_first_alpha("недобрый день") == "недобрый день"


def test_multichar_uppercase_lead_is_left_alone():
    # "ß".upper() == "SS" would change length — the net must skip it so the
    # position-preserving invariant (used to keep word timings) always holds.
    assert _cap_first_alpha("ßad example") == "ßad example"


def test_accented_latin_lead_is_capitalized():
    assert _cap_first_alpha("élan vital") == "Élan vital"


# ── Helpers ─────────────────────────────────────────────────────────────────

def test_prev_cue_ends_sentence():
    assert _prev_cue_ends_sentence("I made it.") is True
    assert _prev_cue_ends_sentence('"I made it."') is True     # closer peeled
    assert _prev_cue_ends_sentence("I want to show my passionate,") is False
    assert _prev_cue_ends_sentence("undying feelings") is False
    assert _prev_cue_ends_sentence("") is False


def test_cap_first_alpha_variants():
    assert _cap_first_alpha("hello") == "Hello"
    assert _cap_first_alpha("- hello") == "- Hello"
    assert _cap_first_alpha('"hello"') == '"Hello"'
    assert _cap_first_alpha("♪ just wild") == "♪ Just wild"
    assert _cap_first_alpha("Already good") == "Already good"
    assert _cap_first_alpha("123 go") == "123 Go"


def test_length_and_count_preserved():
    cues = ["one two three.", "four five six", "seven. eight nine"]
    out = fix_subtitle_casing(cues, EN)
    assert len(out) == len(cues)
    for a, b in zip(cues, out):
        assert len(a) == len(b)  # casing never changes length


def test_empty_and_blank_cues():
    assert fix_subtitle_casing(["", "  ", "hello."], EN)[-1] == "Hello."
    assert fix_subtitle_casing([], EN) == []
