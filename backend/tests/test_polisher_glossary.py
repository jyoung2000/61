"""The transcript polisher must enforce canonical proper-noun spellings from
the custom-vocabulary glossary.

In the Gundam Wing transcript the same character was spelled both "Darlian"
and "Dorian"; feeding the glossary to the polisher lets it normalise such
phonetic / inconsistent variants across the whole transcript.
"""

from backend.services.transcript_polisher import _build_user_prompt


def _seg(text):
    return {"text": text, "start": 0.0}


def test_glossary_injects_canonical_names_rule():
    prompt = _build_user_prompt(
        [_seg("Mr. Dorian, please comment.")], [], [], "en",
        glossary_terms=["Darlian", "Heero Yuy", "Wufei", "Deathscythe"],
    )
    assert "CANONICAL NAMES" in prompt
    assert "Darlian" in prompt
    assert "Heero Yuy" in prompt
    assert "Wufei" in prompt


def test_no_glossary_no_canonical_rule():
    prompt = _build_user_prompt([_seg("hello")], [], [], "en")
    assert "CANONICAL NAMES" not in prompt


def test_empty_glossary_no_rule():
    prompt = _build_user_prompt([_seg("hello")], [], [], "en", glossary_terms=[])
    assert "CANONICAL NAMES" not in prompt
