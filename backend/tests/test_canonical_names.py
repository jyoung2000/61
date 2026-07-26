"""Canonical-name resolution for the auto translation glossary.

The Gundam Wing E01 run shipped "Ririna / Leena Dorian", "Zex/Zekus", "Hero
Yuu", "Katō" etc. because the auto glossary locked in Whisper's mis-heard
romaji. ``resolve_canonical_names`` asks one LLM call (anchored on the video
title) to map detected terms to official English names; these tests pin down
its parse robustness, deny-heuristics, fail-softness, caching, and the
glossary-block rendering with user-term precedence.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from backend.services import canonical_names as CN
from backend.services.canonical_names import resolve_canonical_names
from backend.services.glossary import (
    build_recurring_terms_block,
    build_translation_glossary_block,
)

TITLE = "MOBILE SUIT GUNDAM WING Episode 1"
TERMS = ["Ririna", "Zex", "Zekus", "Hero Yuu", "Katō", "Shuttle", "Colony"]


class DummyOrch:
    """Mock orchestrator recording calls; returns a canned response."""

    def __init__(self, response="", exc=None):
        self.response = response
        self.exc = exc
        self.calls = []

    async def text_completion(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.response


class NoJsonModeOrch:
    """Orchestrator variant whose signature rejects json_mode (TypeError)."""

    def __init__(self, response):
        self.response = response
        self.calls = 0

    async def text_completion(self, prompt, max_tokens=4096, timeout=60,
                              job_id="", skip_circuit_breaker=False,
                              model_override=None):
        self.calls += 1
        return self.response


@pytest.fixture(autouse=True)
def _fresh_cache():
    CN.clear_cache()
    yield
    CN.clear_cache()


def _resolve(orch, terms=None, title=TITLE, job_id="", series_hint=""):
    return asyncio.run(resolve_canonical_names(
        terms if terms is not None else list(TERMS), title, orch,
        job_id=job_id, series_hint=series_hint))


# ── Happy path + parse robustness ──


def test_happy_path_maps_names_and_leaves_common_nouns_alone():
    resp = json.dumps({
        "Ririna": "Relena", "Zex": "Zechs", "Zekus": "Zechs",
        "Hero Yuu": "Heero Yuy", "Katō": "Quatre",
        "Shuttle": "Shuttle", "Colony": "Colony",  # identities → dropped
    })
    orch = DummyOrch(resp)
    out = _resolve(orch)
    assert out == {"Ririna": "Relena", "Zex": "Zechs", "Zekus": "Zechs",
                   "Hero Yuu": "Heero Yuy", "Katō": "Quatre"}
    assert "Shuttle" not in out and "Colony" not in out
    # ONE LLM call, json_mode requested, circuit breaker skipped.
    assert len(orch.calls) == 1
    kwargs = orch.calls[0][1]
    assert kwargs.get("json_mode") is True
    assert kwargs.get("skip_circuit_breaker") is True


def test_parses_fenced_and_prose_wrapped_json():
    resp = ("Sure! Here is the mapping:\n```json\n"
            '{"Ririna": "Relena", "Colony": "Colony"}\n```\nHope that helps.')
    out = _resolve(DummyOrch(resp))
    assert out == {"Ririna": "Relena"}


def test_keys_matched_case_insensitively_to_input_surface_form():
    out = _resolve(DummyOrch('{"ririna": "Relena", "ZEX": "Zechs"}'))
    assert out == {"Ririna": "Relena", "Zex": "Zechs"}


def test_non_dict_and_garbage_responses_yield_empty():
    assert _resolve(DummyOrch('["Relena", "Zechs"]')) == {}
    CN.clear_cache()
    assert _resolve(DummyOrch("no json here at all")) == {}
    CN.clear_cache()
    assert _resolve(DummyOrch("")) == {}


def test_non_string_values_dropped():
    resp = '{"Ririna": ["Relena"], "Zex": 5, "Katō": null, "Hero Yuu": "Heero Yuy"}'
    assert _resolve(DummyOrch(resp)) == {"Hero Yuu": "Heero Yuy"}


def test_sentence_like_values_dropped():
    resp = json.dumps({
        "Ririna": "This character is officially called Relena Darlian",
        "Zex": "Zechs",
    })
    assert _resolve(DummyOrch(resp)) == {"Zex": "Zechs"}


def test_invented_keys_dropped():
    # "Wufei" is NOT in the detected terms — the model must not invent names.
    resp = '{"Wufei": "Chang Wufei", "Ririna": "Relena"}'
    assert _resolve(DummyOrch(resp)) == {"Ririna": "Relena"}


def test_wrapper_object_unwrapped():
    resp = '{"mappings": {"Ririna": "Relena"}}'
    assert _resolve(DummyOrch(resp)) == {"Ririna": "Relena"}


# ── Deny-heuristics ──


def test_profane_values_dropped():
    resp = '{"Ririna": "Relena", "Zex": "Fucking Zechs"}'
    assert _resolve(DummyOrch(resp)) == {"Ririna": "Relena"}


def test_duplicate_canonical_allowed_only_for_obvious_variants():
    # Zex + Zekus are variants of "Zechs" → both kept; "Shuttle" mapping to
    # the same value is not a variant → dropped.
    resp = json.dumps({"Zex": "Zechs", "Zekus": "Zechs", "Shuttle": "Zechs"})
    out = _resolve(DummyOrch(resp))
    assert out == {"Zex": "Zechs", "Zekus": "Zechs"}


# ── Fail-soft + gating ──


def test_orchestrator_error_yields_empty():
    assert _resolve(DummyOrch(exc=RuntimeError("provider down"))) == {}


def test_orchestrator_timeout_yields_empty():
    assert _resolve(DummyOrch(exc=asyncio.TimeoutError())) == {}


def test_no_terms_or_too_few_anchorless_terms_makes_no_llm_call():
    # No terms at all → nothing to map, no call, with or without a title.
    orch = DummyOrch('{"Ririna": "Relena"}')
    assert _resolve(orch, terms=[]) == {}
    # No title/hint AND too few terms to fingerprint the work → no call.
    assert _resolve(orch, terms=["Ririna", "Oz"], title="") == {}
    assert orch.calls == []


def test_anchorless_but_distinctive_terms_do_resolve():
    # A generic filename anchors nothing, but >=4 distinctive terms are a
    # fingerprint — the model is asked to identify the work itself (its
    # confidence rule guards against guessing). Regression: a real run
    # passed title "videoplayback.mp4" and every name stayed phonetic.
    orch = DummyOrch('{"Ririna": "Relena"}')
    out = _resolve(orch, title="videoplayback.mp4")
    assert out == {"Ririna": "Relena"}
    assert len(orch.calls) == 1
    # The useless filename must NOT be presented as a title anchor.
    assert "videoplayback" not in orch.calls[0][0]
    assert "identify" in orch.calls[0][0].lower()


def test_series_hint_alone_is_enough_anchor():
    orch = DummyOrch('{"Ririna": "Relena"}')
    out = _resolve(orch, title="", series_hint="Mobile Suit Gundam Wing")
    assert out == {"Ririna": "Relena"}
    assert "Gundam Wing" in orch.calls[0][0]


def test_disabled_via_settings_returns_empty(monkeypatch):
    # The TRANSLATION_CANONICAL_NAMES field lands in config.py via the wiring
    # change; until then the pydantic Settings model rejects unknown fields,
    # so stub the module's settings accessor to flip the gate.
    class _S:
        TRANSLATION_CANONICAL_NAMES = False

    monkeypatch.setattr(CN, "_settings", lambda: _S())
    orch = DummyOrch('{"Ririna": "Relena"}')
    assert _resolve(orch) == {}
    assert orch.calls == []


def test_json_mode_typeerror_falls_back_to_bare_call():
    orch = NoJsonModeOrch('{"Ririna": "Relena"}')
    assert _resolve(orch) == {"Ririna": "Relena"}
    # first attempt (with json_mode) raised TypeError before the coroutine
    # ran, so only the bare retry actually executed.
    assert orch.calls == 1


# ── Caching ──


def test_cache_per_job_id_makes_repeat_calls_free():
    orch = DummyOrch('{"Ririna": "Relena"}')
    first = _resolve(orch, job_id="job-42")
    second = _resolve(orch, job_id="job-42")
    assert first == second == {"Ririna": "Relena"}
    assert len(orch.calls) == 1


def test_prompt_contains_title_and_all_terms():
    orch = DummyOrch("{}")
    _resolve(orch)
    prompt = orch.calls[0][0]
    assert TITLE in prompt
    for t in TERMS:
        assert t in prompt


# ── Glossary block rendering + merge precedence ──


def test_block_renders_arrow_entries_and_canonical_rule():
    block = build_recurring_terms_block(
        ["Ririna", "Colony"], "English",
        canonical_map={"Ririna": "Relena Darlian"})
    assert "Ririna → Relena Darlian" in block
    assert "Colony" in block
    assert "canonical" in block  # instruction line present


def test_block_without_map_is_unchanged_format():
    block = build_recurring_terms_block(["Ririna", "Colony"], "English")
    assert "→" not in block
    assert "canonical" not in block
    assert "Ririna, Colony" in block


def test_user_terms_always_win_over_canonical_rewrites():
    block = build_recurring_terms_block(
        ["Relena Darlian", "Ririna"], "English",
        canonical_map={"Relena Darlian": "Wrong Name",
                       "Ririna": "Relena Darlian"},
        protected_terms=["Relena Darlian"])
    assert "Relena Darlian → Wrong Name" not in block
    assert "Wrong Name" not in block
    assert "Ririna → Relena Darlian" in block


def test_identity_mapping_not_rendered_as_arrow():
    block = build_recurring_terms_block(
        ["Zechs"], "English", canonical_map={"Zechs": "zechs"})
    assert "→" not in block


def test_build_translation_glossary_block_end_to_end_precedence_and_cap():
    segs = [{"text": "Ririna went to the shuttle"},
            {"text": "Ririna spoke to Zex"},
            {"text": "Zex saluted"}]
    block = build_translation_glossary_block(
        segs, "ja", "English", user_terms=["Heero Yuy"],
        canonical_map={"Ririna": "Relena Darlian",
                       "Heero Yuy": "Hero Yu",  # must NOT rewrite user term
                       "Zex": "Zechs"})
    assert "Heero Yuy" in block and "Hero Yu," not in block and "→ Hero Yu" not in block
    assert "Ririna → Relena Darlian" in block
    assert "Zex → Zechs" in block
    # user term ranks first in the merged list
    assert block.index("Heero Yuy") < block.index("Ririna")


def test_cap_preserved_with_canonical_map():
    segs = []
    for i in range(60):
        # each name appears twice so it qualifies as recurring
        segs.append({"text": f"Name{i:02d} met Name{i:02d}"})
    cmap = {f"Name{i:02d}": f"Canon{i:02d}" for i in range(60)}
    block = build_translation_glossary_block(
        segs, "en", "English", max_terms=10, user_terms=[], canonical_map=cmap)
    listed = [e for e in block.splitlines()[1].split(", ") if e.strip()]
    assert len(listed) == 10


# ── Series-hint → ASR bias roster (source-side name fix) ──────────────────

class _RosterOrch:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def text_completion(self, prompt, **kwargs):
        self.calls += 1
        return self.payload


def test_expand_series_hint_caches_and_feeds_asr_roster():
    CN._ROSTER_CACHE.clear()
    orch = _RosterOrch('["Heero Yuy", "Relena Darlian", "Zechs Merquise", "OZ"]')
    from backend.config import settings
    settings.TRANSLATION_SERIES_HINT = "Mobile Suit Gundam Wing"
    settings.CUSTOM_VOCABULARY_ENABLED = True
    out = asyncio.run(CN.expand_series_hint_to_names(
        "Mobile Suit Gundam Wing", orch, job_id="t"))
    assert "Zechs Merquise" in out and "Heero Yuy" in out
    # series_roster_terms() reads the cache for the configured hint.
    assert "Zechs Merquise" in CN.series_roster_terms()
    # Cached — a second expand makes no new LLM call.
    asyncio.run(CN.expand_series_hint_to_names("Mobile Suit Gundam Wing", orch, job_id="t"))
    assert orch.calls == 1
    settings.TRANSLATION_SERIES_HINT = ""
    assert CN.series_roster_terms() == []
    CN._ROSTER_CACHE.clear()


def test_expand_series_hint_failsoft_on_bad_reply():
    CN._ROSTER_CACHE.clear()
    orch = _RosterOrch("I don't know this show, sorry.")
    out = asyncio.run(CN.expand_series_hint_to_names("Some Obscure Show", orch, job_id="t"))
    assert out == []
    CN._ROSTER_CACHE.clear()


def test_vocab_bias_kwargs_includes_series_roster(monkeypatch):
    # The local Whisper bias site must merge the series roster with the glossary.
    CN._ROSTER_CACHE.clear()
    from backend.config import settings
    settings.TRANSLATION_SERIES_HINT = "Mobile Suit Gundam Wing"
    settings.CUSTOM_VOCABULARY_ENABLED = True
    CN._ROSTER_CACHE["mobile suit gundam wing"] = ["Zechs Merquise", "Heero Yuy"]
    import backend.services.custom_vocabulary as CV
    monkeypatch.setattr(CV, "load_vocabulary", lambda: ["MyOwnTerm"])
    from backend.services.reframer_audio import _vocab_bias_kwargs

    def _cb(*a, hotwords=None, **k):
        pass
    out = _vocab_bias_kwargs(_cb, "ja")
    blob = " ".join(str(v) for v in out.values())
    assert "Zechs Merquise" in blob and "MyOwnTerm" in blob
    settings.TRANSLATION_SERIES_HINT = ""
    CN._ROSTER_CACHE.clear()


# ── Roster precision: an ASR garble keeps the leading sound ──

def test_roster_rejects_pairs_with_a_different_initial():
    """A changed first letter means a DIFFERENT name, not a mishearing.

    A real run rewrote "Marina" — the Alliance's salvage ship, used consistently
    three times — into the character "Relena" because they rhyme and Relena was
    more frequent, corrupting three cues ("Alliance's Relena is trying to recover
    that machine"). The same rule rejects three other recorded failures.
    """
    from backend.services.canonical_names import _roster_phonetic_ok
    for wrong, right in [
        ("Marina", "Relena"),                       # ship -> character
        ("Hero Yuu", "Trowa Barton"),               # wrong character entirely
        ("Earth Sphere Alliance", "Zeon"),          # wrong franchise
        ("Operation Meteor", "Operation Endgame"),  # shared word stripped first
    ]:
        allowed, _ = _roster_phonetic_ok(wrong, right)
        assert not allowed, f"{wrong!r} -> {right!r} must be rejected"


def test_roster_still_allows_real_mishearings():
    """Interior-sound garbles are exactly what this pass exists to consolidate."""
    from backend.services.canonical_names import _roster_phonetic_ok
    for wrong, right in [
        ("Zecks", "Zechs"), ("Zexes", "Zechs"), ("Hero-kun", "Heero"),
        ("Airies", "Aries"), ("Dorian", "Darlian"), ("Trois", "Treize"),
    ]:
        allowed, _ = _roster_phonetic_ok(wrong, right)
        assert allowed, f"{wrong!r} -> {right!r} must still be allowed"


# ── Determinism: a resolved name map must survive across runs ──

def test_canonical_map_persists_and_is_reused(tmp_path, monkeypatch):
    """Keying the cache per job made the same video re-ask the model, and the
    answer varied: consecutive runs of one episode resolved 6 names then 0,
    shipping "Relena"/"Zechs" one time and "Lilyana"/"Sixes" the next.
    A content key plus a durable store makes it deterministic."""
    from backend.services import canonical_names as CN
    monkeypatch.setattr(CN, "_persist_path", lambda: str(tmp_path / "cn.json"))
    CN.clear_cache()
    key = CN._content_key("Gundam Wing Episode 1", ["リリーナ", "ゼクス"])
    CN._persist_put(key, {"リリーナ": "Relena", "ゼクス": "Zechs"})
    assert CN._persist_load().get(key) == {"リリーナ": "Relena", "ゼクス": "Zechs"}
    # Same title + terms in any order → same key (so a re-run reuses it).
    assert CN._content_key("Gundam Wing Episode 1", ["ゼクス", "リリーナ"]) == key
    # Different terms → different key (no cross-title bleed).
    assert CN._content_key("Gundam Wing Episode 1", ["ハロ"]) != key


def test_canonical_persist_ignores_empty_results(tmp_path, monkeypatch):
    """A transient provider failure must never be cached forever."""
    from backend.services import canonical_names as CN
    monkeypatch.setattr(CN, "_persist_path", lambda: str(tmp_path / "cn.json"))
    key = CN._content_key("Some Show", ["term"])
    CN._persist_put(key, {})
    assert CN._persist_load().get(key) is None


def test_canonical_persist_is_failsoft_on_corrupt_store(tmp_path, monkeypatch):
    from backend.services import canonical_names as CN
    p = tmp_path / "cn.json"
    p.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(CN, "_persist_path", lambda: str(p))
    assert CN._persist_load() == {}
