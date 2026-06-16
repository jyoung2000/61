"""Tests for the numbered-tag context join that replaced the ¶ sentinel (Task 3).

The old `" ¶ "` join was fragile: NLLB routinely dropped/mangled the single
separator, so the code fell back to context-FREE per-segment translation,
throwing away pronoun/gender/idiom resolution. Cues are now wrapped in numbered
tags (`⟦i⟧…⟦/i⟧`) and recovered by index; when re-alignment still fails (tags
lost or block too long) each cue is retried individually WITH prior cues
prepended (translated, then discarded) so context is never silently dropped.
"""

import sys
import types

import pytest


def _install_config_stub(**overrides):
    mod = types.ModuleType("backend.config")
    defaults = dict(
        NMT_NLLB_MODEL="facebook/nllb-200-distilled-1.3B",
        NMT_OPUS_MT_TEMPLATE="Helsinki-NLP/opus-mt-{src}-{tgt}",
        NMT_MAX_OPUS_PAIRS=5, NMT_AUTODOWNLOAD=True, NMT_DEVICE="auto",
    )
    defaults.update(overrides)
    mod.settings = types.SimpleNamespace(**defaults)
    sys.modules["backend.config"] = mod
    return mod


@pytest.fixture()
def N():
    _install_config_stub()
    from backend.services import nmt_translator as N
    return N


# ── _parse_numbered_tags ──────────────────────────────────────────────────

def test_parse_basic(N):
    assert N._parse_numbered_tags("⟦1⟧a⟦/1⟧ ⟦2⟧b⟦/2⟧", 1, 3, 2) == ["a", "b"]


def test_parse_tolerates_mangled_brackets(N):
    # NLLB may render ⟦ as [ or 【 and add whitespace.
    out = N._parse_numbered_tags("[1] alpha [/1]  【2】 beta 【/2】", 1, 3, 2)
    assert out == ["alpha", "beta"]


def test_parse_missing_close_bounded_by_next_open(N):
    assert N._parse_numbered_tags("⟦1⟧a ⟦2⟧b⟦/2⟧", 1, 3, 2) == ["a", "b"]


def test_parse_returns_none_when_index_missing(N):
    # Only cue 1 present; cue 2 was requested → None (caller goes per-cue).
    assert N._parse_numbered_tags("⟦1⟧a⟦/1⟧", 1, 3, 2) is None


def test_parse_returns_none_when_no_tags(N):
    assert N._parse_numbered_tags("just translated prose, no tags", 1, 2, 1) is None


def test_parse_ignores_out_of_range_numbers(N):
    # "[99]" is not a real tag (n_total=1) → stays part of cue 1's text.
    out = N._parse_numbered_tags("⟦1⟧see ref [99] here⟦/1⟧", 1, 2, 1)
    assert out == ["see ref [99] here"]


def test_parse_recovers_only_requested_batch_slice(N):
    # 4 cues total; only the middle two (the batch) are requested.
    block = "⟦1⟧ctxA⟦/1⟧ ⟦2⟧hit1⟦/2⟧ ⟦3⟧hit2⟦/3⟧ ⟦4⟧ctxB⟦/4⟧"
    assert N._parse_numbered_tags(block, 2, 4, 4) == ["hit1", "hit2"]


# ── translate_with_context end-to-end (fake engine) ───────────────────────

def _engine(N):
    tr = N.NMTTranslator()
    tr._loaded = True
    return tr


def _tag_count(N, s):
    return len(N._NMT_TAG_RE.findall(s))


def test_context_join_happy_path(N):
    """Tags survive → one-shot join, cues recovered by index, no per-cue calls."""
    tr = _engine(N)
    calls = []

    def fake(texts, src, tgt, glossary=None):
        calls.append(texts[0])
        return [texts[0].upper()]            # tag-preserving "translation"

    tr.translate_batch = fake
    out = tr.translate_with_context(
        ["he ran", "she sang"], ["dog barks"], ["bird flies"], "en", "en")

    assert out == ["HE RAN", "SHE SANG"]
    assert len(calls) == 1                   # only the single joined call
    assert tr._ctx_join_ok == 1
    assert tr._ctx_join_tag_fail == 0 and tr._ctx_join_too_long == 0


def test_tag_failure_falls_to_per_cue_with_context_retained(N):
    """Join loses tags → per-cue retry; prior context is prepended (then
    discarded) so referential context survives, output is the target only."""
    tr = _engine(N)
    calls = []

    def fake(texts, src, tgt, glossary=None):
        s = texts[0]
        calls.append(s)
        # NLLB drops tags on the long multi-tag join, keeps them on short
        # single-cue minis (the realistic mangling pattern).
        if _tag_count(N, s) > 2:
            return [N._NMT_TAG_RE.sub("", s).upper()]
        return [s.upper()]

    tr.translate_batch = fake
    out = tr.translate_with_context(
        ["he ran", "she sang"], ["dog barks"], ["bird flies"], "en", "en")

    assert out == ["HE RAN", "SHE SANG"]
    assert tr._ctx_join_tag_fail == 1 and tr._ctx_join_ok == 0
    # The per-cue retry for "he ran" tagged it as cue 1 (only the mini does;
    # the join tagged it as cue 2) and prepended the prior context cue…
    mini = next(c for c in calls if "⟦1⟧he ran⟦/1⟧" in c)
    assert "dog barks" in mini
    # …but the discarded context is NOT in the returned cue.
    assert "DOG BARKS" not in out[0]


def test_too_long_block_skips_join_for_per_cue(N, monkeypatch):
    """A block over the safe-chunk cap goes straight to per-cue (never a
    truncated join), and still translates every cue."""
    tr = _engine(N)
    monkeypatch.setattr(N, "_MAX_SRC_CHARS_LATIN", 5)   # force "too long"

    def fake(texts, src, tgt, glossary=None):
        # Tag-preserving so single-cue minis (if attempted) still parse.
        return [texts[0].upper()]

    tr.translate_batch = fake
    out = tr.translate_with_context(["aaa", "bbb"], ["ccc"], [], "en", "en")

    assert out == ["AAA", "BBB"]
    assert tr._ctx_join_too_long == 1
    assert tr._ctx_join_ok == 0 and tr._ctx_join_tag_fail == 0


def test_empty_batch_short_circuits(N):
    tr = _engine(N)
    assert tr.translate_with_context([], ["x"], ["y"], "en", "en") == []
    assert tr.translate_with_context(["", "  "], ["x"], [], "en", "en") == ["", "  "]


def test_single_cue_fallback_is_complete_even_without_tags(N):
    """If even the single-cue mini loses its tag, the plain chunked translation
    still returns a complete (non-empty) result — never the empty string."""
    tr = _engine(N)

    def fake(texts, src, tgt, glossary=None):
        # Always strip tags → join fails AND per-cue mini fails → falls to the
        # bare single-cue translate_batch([cue]) (which has no tags to lose).
        return [N._NMT_TAG_RE.sub("", texts[0]).upper()]

    tr.translate_batch = fake
    out = tr.translate_with_context(["hello world"], ["prior"], [], "en", "en")
    assert out == ["HELLO WORLD"]
