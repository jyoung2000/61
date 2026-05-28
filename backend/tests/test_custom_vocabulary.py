"""Acceptance tests for Task 1 — Custom Vocabulary (Whisper biasing).

Covers:
  * save/load round-trip, 300-term cap, 80-char length limit, dedupe
  * build_initial_prompt token ceiling + CJK framing omission
  * whisper_bias_kwargs pass-through: hotwords when supported,
    initial_prompt fallback otherwise, nothing when empty/disabled
"""

import json

import pytest

from backend.services import custom_vocabulary as cv


@pytest.fixture(autouse=True)
def _isolate_vocab_file(tmp_path, monkeypatch):
    """Point the glossary file at a temp path so tests don't touch /data."""
    path = tmp_path / "custom_vocabulary.json"
    monkeypatch.setattr(cv, "_vocabulary_path", lambda: str(path))
    return path


# ── Persistence round-trip + validation ──────────────────────────────────

def test_save_load_round_trip(_isolate_vocab_file):
    terms = ["Heero Yuy", "Zechs Merquise", "Gundam", "OZ", "mobile suit"]
    saved = cv.save_vocabulary(terms)
    assert saved == terms
    assert cv.load_vocabulary() == terms
    # File actually written to the mount-backed path.
    on_disk = json.loads(_isolate_vocab_file.read_text(encoding="utf-8"))
    assert on_disk["terms"] == terms


def test_load_missing_file_returns_empty():
    assert cv.load_vocabulary() == []


def test_cap_at_300_terms():
    terms = [f"term{i}" for i in range(500)]
    saved = cv.save_vocabulary(terms)
    assert len(saved) == cv.MAX_TERMS == 300
    assert saved[0] == "term0"
    assert len(cv.load_vocabulary()) == 300


def test_reject_overlong_terms():
    long_term = "x" * (cv.MAX_TERM_LENGTH + 1)
    ok_term = "y" * cv.MAX_TERM_LENGTH
    saved = cv.save_vocabulary([long_term, ok_term, "short"])
    assert long_term not in saved
    assert ok_term in saved
    assert "short" in saved


def test_dedupe_and_strip():
    saved = cv.save_vocabulary(["  Gundam  ", "gundam", "GUNDAM", "", "  ", "OZ"])
    # Case-insensitive dedupe, first spelling wins, blanks dropped.
    assert saved == ["Gundam", "OZ"]


# ── initial_prompt assembly ───────────────────────────────────────────────

def test_initial_prompt_latin_framing():
    prompt = cv.build_initial_prompt(["Heero Yuy", "Gundam", "OZ"], "en")
    assert prompt.startswith("Glossary:")
    assert "Heero Yuy" in prompt
    assert prompt.endswith(".")


@pytest.mark.parametrize("lang", ["ja", "zh", "ko", "zh-cn", "JA"])
def test_initial_prompt_cjk_omits_latin_framing(lang):
    prompt = cv.build_initial_prompt(["ガンダム", "ヒイロ"], lang)
    assert "Glossary" not in prompt
    assert "ガンダム" in prompt
    # CJK enumeration comma joins the terms.
    assert "、" in prompt


def test_initial_prompt_under_token_ceiling():
    terms = [f"longvocabularyterm{i}" for i in range(400)]
    prompt = cv.build_initial_prompt(terms, "en")
    assert cv._estimate_tokens(prompt) <= cv.PROMPT_TOKEN_CEILING


def test_initial_prompt_empty_for_empty_vocab():
    assert cv.build_initial_prompt([], "en") == ""
    assert cv.build_initial_prompt(["   "], "en") == ""


def test_hotwords_string():
    assert cv.hotwords_string(["Gundam", "OZ"]) == "Gundam OZ"
    assert cv.hotwords_string([]) == ""


# ── whisper_bias_kwargs pass-through ──────────────────────────────────────

def _engine_with_hotwords(audio, *, hotwords=None, initial_prompt=None, **kw):
    return None


def _engine_without_hotwords(audio, *, initial_prompt=None, **kw):
    return None


def test_bias_passes_hotwords_when_supported(_isolate_vocab_file):
    cv.save_vocabulary(["Gundam", "Heero Yuy"])
    kwargs = cv.whisper_bias_kwargs(_engine_with_hotwords, language="en", enabled=True)
    assert kwargs == {"hotwords": "Gundam Heero Yuy"}
    assert "initial_prompt" not in kwargs


def test_bias_falls_back_to_initial_prompt(_isolate_vocab_file):
    cv.save_vocabulary(["Gundam", "Heero Yuy"])
    kwargs = cv.whisper_bias_kwargs(_engine_without_hotwords, language="en", enabled=True)
    assert "hotwords" not in kwargs
    assert kwargs["initial_prompt"].startswith("Glossary:")


def test_bias_empty_when_vocab_empty(_isolate_vocab_file):
    # No glossary saved → nothing passed (preserve current behaviour).
    assert cv.whisper_bias_kwargs(_engine_with_hotwords, "en", enabled=True) == {}


def test_bias_empty_when_disabled(_isolate_vocab_file):
    cv.save_vocabulary(["Gundam"])
    assert cv.whisper_bias_kwargs(_engine_with_hotwords, "en", enabled=False) == {}


def test_bias_cjk_initial_prompt_no_latin_framing(_isolate_vocab_file):
    cv.save_vocabulary(["ガンダム", "ヒイロ"])
    kwargs = cv.whisper_bias_kwargs(_engine_without_hotwords, language="ja", enabled=True)
    assert "Glossary" not in kwargs["initial_prompt"]
