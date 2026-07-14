"""Garble detection for translated subtitles (2026-07-14).

translated_24.txt shipped two garble classes that ``_is_untranslated`` misses:
(1) a middot word-salad cue (a small model echoing the glossary's separator
template) and (2) romaji onomatopoeia leaks ("Korikori", "Banzai", "Dame").
These pure detectors flag them for a re-translate, and ``collapse_separator_salad``
is the deterministic net so a "·" pile can never ship.
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.translator import (  # noqa: E402
    garble_reason,
    is_garbled_translation,
    collapse_separator_salad,
    _word_salad_reason,
    _is_mora_reduplication,
    _capitalized_ratio,
)

JA = "ja"
SALAD = ("It's · Really · Huh · That's · Onee-chan · It's · Wow · Seriously · "
         "Are · You're · Like · Amazing · Do · See · You're · Just · Different.")


# ── Word-salad ──────────────────────────────────────────────────────────────

def test_middot_salad_flagged():
    assert _word_salad_reason(SALAD) == "middot-salad"
    assert garble_reason(SALAD, JA) == "middot-salad"


def test_pipe_wordlist_salad_flagged():
    assert _word_salad_reason("Really | Huh | Wow | Amazing | Different | Just") == "wordlist-salad"


def test_normal_prose_not_salad():
    assert _word_salad_reason("We met in New York, London and Paris.") is None
    assert _word_salad_reason("Well — I don't know about that, honestly.") is None
    assert _word_salad_reason("It feels good, doesn't it?") is None
    # A 2-part pipe string is not enough parts to be a salad.
    assert _word_salad_reason("Press A | B") is None


def test_capitalized_ratio_bounds():
    assert _capitalized_ratio("this is a normal sentence here") < 0.2
    assert _capitalized_ratio("Really Huh Wow Amazing Different") >= 0.9
    assert _capitalized_ratio("one two") == 0.0  # <4 tokens


# ── Romaji leak ─────────────────────────────────────────────────────────────

def test_romaji_onomatopoeia_flagged_ja_source():
    for w in ["Banzai", "Dame", "Puncha", "Kuri", "Etchi", "Nonko", "Kamon"]:
        assert garble_reason(w, JA) is not None, w


def test_romaji_reduplication_flagged():
    assert garble_reason("Korikori", JA) == "romaji-reduplication"
    assert _is_mora_reduplication("purupuru") is True
    assert _is_mora_reduplication("dokidoki") is True
    assert _is_mora_reduplication("bonbon") is False   # English allowlist


def test_romaji_negatives_are_name_safe():
    # Valid loanwords, a lone unknown name, and plain English never flag.
    for s in ["Sushi", "Tokyo", "Yuki", "Thank you for the meal.", "See you later."]:
        assert garble_reason(s, JA) is None, s


def test_recurring_name_in_glossary_exempt():
    # A recurring name provided in the glossary is never flagged as romaji.
    assert garble_reason("Sakura", JA, frozenset({"sakura"})) is None


def test_non_japanese_source_never_flags_romaji():
    assert garble_reason("Dame", "es") is None
    assert garble_reason("Banzai", "fr") is None


def test_is_garbled_wrapper():
    assert is_garbled_translation(SALAD, JA) is True
    assert is_garbled_translation("A normal English line.", JA) is False


# ── Deterministic net ───────────────────────────────────────────────────────

def test_collapse_separator_salad_strips_middots():
    out = collapse_separator_salad(SALAD, source="ソースライン")
    assert "·" not in out and "•" not in out
    assert out  # never empty
    # A clean line is returned unchanged.
    assert collapse_separator_salad("A normal line.") == "A normal line."


def test_collapse_falls_back_to_source_when_nothing_usable():
    # A 3-middot salad whose parts all dedup to a single short token collapses
    # to <2 chars → fall back to the honest source rather than a bare glyph.
    assert collapse_separator_salad("A · A · A · A", source="元の文") == "元の文"
