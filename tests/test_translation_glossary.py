"""Content-agnostic recurring-term glossary for translation consistency.

Fixes the observed failure where the same name rendered four different ways in
one pass ("Relena/Lillian/Liliana/Lily") and coined nouns got translated into
ordinary words. Must generalize beyond anime — these tests use cooking, news,
lecture and finance transcripts alongside Japanese.
"""
from backend.services.glossary import (
    extract_recurring_terms,
    build_recurring_terms_block,
    build_translation_glossary_block,
)


def _segs(*texts):
    return [{"text": t} for t in texts]


def test_english_recurring_proper_nouns_general_cooking():
    segs = _segs(
        "Today Chef Tanaka visits Osaka.",
        "Tanaka loves Osaka street food.",
        "In Osaka, Tanaka makes takoyaki.",
    )
    terms = extract_recurring_terms(segs, "en")
    assert "Tanaka" in terms and "Osaka" in terms
    # Common sentence-openers are not treated as names.
    assert "Today" not in terms and "In" not in terms


def test_one_off_names_excluded_by_min_count():
    segs = _segs("Maria went to Berlin.", "Maria stayed home.")
    terms = extract_recurring_terms(segs, "en")
    assert "Maria" in terms       # appears twice
    assert "Berlin" not in terms  # appears once


def test_acronyms_recurring_extracted_general_news():
    segs = _segs(
        "NASA confirmed the launch.",
        "The FBI and NASA disagreed.",
        "NASA released a statement; the FBI declined.",
    )
    terms = extract_recurring_terms(segs, "en")
    assert "NASA" in terms and "FBI" in terms


def test_japanese_katakana_names_extracted():
    segs = _segs(
        "リリーナとゼクスが戦う。",
        "ゼクスはガンダムを追う。",
        "リリーナはガンダムを見た。",
    )
    terms = extract_recurring_terms(segs, "ja")
    assert "リリーナ" in terms and "ゼクス" in terms and "ガンダム" in terms


def test_lecture_domain_terms_general():
    segs = _segs(
        "Schrodinger derived the equation.",
        "The Schrodinger equation is central.",
        "Heisenberg disagreed with Schrodinger.",
    )
    terms = extract_recurring_terms(segs, "en")
    assert "Schrodinger" in terms  # recurs 3x; Heisenberg once → excluded


def test_max_terms_cap_and_ranking():
    segs = _segs(*["Apple Apple Banana" for _ in range(5)])
    terms = extract_recurring_terms(segs, "en", max_terms=1)
    assert terms == ["Apple"]  # most frequent first, capped to 1


def test_block_empty_when_nothing_recurs():
    assert build_recurring_terms_block([]) == ""
    assert build_translation_glossary_block(_segs("Hello there.", "How are you?"), "en") == ""


def test_block_mentions_consistency_and_transliteration():
    block = build_recurring_terms_block(["Relena", "Zechs"], "English")
    assert "Relena" in block and "Zechs" in block
    assert "SAME way" in block
    assert "transliterate" in block.lower()
    # Must not leak any hardcoded franchise vocabulary — only the passed terms.
    assert "Gundam" not in block


def test_handles_pydantic_like_segments():
    class _S:
        def __init__(self, text):
            self.text = text
    segs = [_S("Captain Reynolds spoke."), _S("Reynolds left.")]
    assert "Reynolds" in extract_recurring_terms(segs, "en")
