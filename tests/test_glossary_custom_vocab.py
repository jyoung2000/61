"""Custom vocabulary → translation glossary wiring (2026-07-15).

The user's global custom vocabulary already biased Whisper ASR and enforced
spellings in the polisher, but it never reached the LLM TRANSLATION stage — so a
name that Whisper mis-heard, appears once, or is katakana in the source (Zechs vs
"Zeks") was left to the model to guess and drift. `build_translation_glossary_block`
now merges the user's authoritative names (first) with the auto-derived recurring
terms so every translation path pins them consistently.
"""
import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.glossary import (  # noqa: E402
    merge_glossary_terms,
    build_translation_glossary_block,
    build_recurring_terms_block,
)


def _seg(text):
    return {"text": text}


# ── merge_glossary_terms ────────────────────────────────────────────────────

def test_user_terms_rank_first():
    out = merge_glossary_terms(["Zechs", "Relena"], ["Alpha", "Beta"])
    assert out[:2] == ["Zechs", "Relena"]
    assert set(out) == {"Zechs", "Relena", "Alpha", "Beta"}


def test_case_insensitive_dedupe_keeps_user_spelling():
    # The auto-extractor found "gundam"; the user typed "Gundam" — keep the
    # user's canonical casing and don't list it twice.
    out = merge_glossary_terms(["Gundam"], ["gundam", "OZ"])
    assert out == ["Gundam", "OZ"]


def test_cap_prioritizes_user_terms():
    users = [f"User{i}" for i in range(50)]
    autos = [f"Auto{i}" for i in range(50)]
    out = merge_glossary_terms(users, autos, cap=40)
    assert len(out) == 40
    assert all(t.startswith("User") for t in out)   # user terms win the budget


def test_merge_handles_empty_and_blank():
    assert merge_glossary_terms([], []) == []
    assert merge_glossary_terms(["  ", ""], ["Real"]) == ["Real"]
    assert merge_glossary_terms(None, None) == []


# ── build_translation_glossary_block ────────────────────────────────────────

def test_block_includes_user_terms_even_when_nothing_recurs():
    # A single-mention name never reaches the auto-extractor (min_count=2), but a
    # user term still pins it.
    segs = [_seg("Heero appears once.")]
    block = build_translation_glossary_block(
        segs, "en", "English", user_terms=["Heero Yuy"])
    assert "Heero Yuy" in block
    assert "keep these consistent" in block.lower() or "consistent" in block.lower()


def test_block_empty_when_no_user_and_no_recurring():
    segs = [_seg("nothing notable here")]
    assert build_translation_glossary_block(segs, "en", "English", user_terms=[]) == ""


def test_explicit_user_terms_arg_bypasses_autoload():
    # Passing user_terms explicitly must NOT also auto-load the persisted vocab
    # (deterministic for tests / callers that supply their own list).
    segs = [_seg("Relena Relena Relena")]   # recurs → auto term
    block = build_translation_glossary_block(
        segs, "en", "English", user_terms=["Zechs"])
    assert "Zechs" in block
    assert "Relena" in block                # auto term still merged in


def test_block_autoloads_custom_vocab_when_user_terms_none(monkeypatch):
    # user_terms=None (the default from every translation path) → the block
    # auto-loads the persisted custom vocabulary, so callers get it for free.
    import backend.services.glossary as G
    monkeypatch.setattr(G, "load_custom_vocabulary_terms", lambda: ["Zechs Merquise"])
    block = G.build_translation_glossary_block([_seg("mentioned once")], "en", "English")
    assert "Zechs Merquise" in block


def test_recurring_block_still_works_standalone():
    # The lower-level formatter is unchanged (comma-joined, no middot salad).
    block = build_recurring_terms_block(["Zechs", "Relena"], "English")
    assert "Zechs, Relena" in block
    assert "·" not in block
