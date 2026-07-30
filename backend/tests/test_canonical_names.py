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


_MISSES = [{"wrong": "Dorian", "right": "Darlian"},
           {"wrong": "Aires", "right": "Aries"},
           {"wrong": "Hero Yu", "right": "Heero Yuy"},
           {"wrong": "Hero-kun", "right": "Heero"}]
_MISS_CORPUS = ("Relena Dorian arrived. Dorian spoke. Dorian left. Aires unit. "
                "Aires again. Aires third. Hero Yu piloted. Hero-kun waited. "
                "Hero-kun again.")
_ROSTER = {"Relena Darlian", "Darlian", "Heero Yuy", "Heero", "Aries",
           "Zechs Merquise", "Duo Maxwell"}


def test_a_never_heard_name_is_correctable_when_ground_truth_vouches_for_it():
    """The attestation rule required the RIGHT spelling to already out-appear
    the wrong one — which makes a genuine knowledge correction mathematically
    impossible, because a name the ASR never got right is attested zero times.

    Traced on a real run, these four cleared mining, the ask-list, the
    ordinary-word and near-typo guards and the phonetic check (ratios
    0.67-0.86), then all died on 0-vs-N. They are recoverable only because an
    independent source — the series roster / canonical map / operator
    vocabulary — confirms the right-hand side is a real name.
    """
    from backend.services.canonical_names import _vet_roster_pairs
    got = _vet_roster_pairs(_MISSES, {p["wrong"] for p in _MISSES},
                            corpus=_MISS_CORPUS, known_names=_ROSTER)
    assert got == {"Dorian": "Darlian", "Aires": "Aries",
                   "Hero Yu": "Heero Yuy", "Hero-kun": "Heero"}, got


def test_without_ground_truth_an_unattested_correction_is_still_refused():
    """No corroborating source → the pass stays a CONSISTENCY tool, which is
    the only safe default: attestation alone cannot tell "never heard right"
    from "already right"."""
    from backend.services.canonical_names import _vet_roster_pairs
    got = _vet_roster_pairs(_MISSES, {p["wrong"] for p in _MISSES},
                            corpus=_MISS_CORPUS)
    assert got == {}, got


def test_ground_truth_does_not_unlock_un_corrections():
    """The inverse failure the attestation rule was protecting against.

    A fast-model run UN-corrected names that were already right: Duo→Dewo,
    General Septem→General Septain, Katul→Kattul. Those are invented
    homophones, so they clear the phonetic gate by construction and have zero
    attestation just like a real correction. What separates them is that the
    invented spelling appears in NO ground-truth source — so a roster must not
    launder them through.
    """
    from backend.services.canonical_names import _vet_roster_pairs
    pairs = [{"wrong": "Duo", "right": "Dewo"},
             {"wrong": "General Septem", "right": "General Septain"},
             {"wrong": "Katul", "right": "Kattul"}]
    got = _vet_roster_pairs(
        pairs, {p["wrong"] for p in pairs},
        corpus="This is Duo! General Septem is waiting. This is Katul reporting.",
        known_names=_ROSTER)
    assert got == {}, got


def test_attestation_bonus_still_prefers_a_consolidation():
    """The signal the veto was protecting survives as ranking.

    When ``max_pairs`` truncates, a correction toward the transcript's own
    dominant spelling outranks an unattested one of similar phonetic closeness.
    """
    from backend.services.canonical_names import _vet_roster_pairs
    got = _vet_roster_pairs(
        [{"wrong": "Zeks", "right": "Zechs"},
         {"wrong": "Dorian", "right": "Darlian"}],
        {"Zeks", "Dorian"},
        corpus="Zechs Zechs Zechs Zeks Dorian Dorian", max_pairs=1)
    assert got == {"Zeks": "Zechs"}, got


def test_dropping_the_veto_kept_every_other_guard():
    """The hallucination cases the veto also caught are covered elsewhere."""
    from backend.services.canonical_names import _vet_roster_pairs
    for wrong, right, corpus in [
        ("Marina", "Relena", "Marina ship Marina"),        # phonetic guard
        ("Justlove", "Justice", "Justlove"),               # ordinary-word guard
        ("Septem", "Gneral Septem", "Septem"),             # near-typo guard
    ]:
        got = _vet_roster_pairs([{"wrong": wrong, "right": right}],
                                {wrong}, corpus=corpus)
        assert got == {}, f"{wrong!r} -> {right!r} must still be rejected: {got}"


def test_resolved_names_are_published_where_the_roster_pass_reads_them():
    """``resolve_roster_corrections`` reads its series evidence from
    ``job:<id>``; the resolver caches under a CONTENT key.

    Keying the cache on title+terms for determinism moved the entry out from
    under that reader without updating it, and nothing wrote ``job:<id>``
    afterwards — so ``series_map`` was always empty and, with no operator series
    hint, the roster pass returned before making any LLM call. It had therefore
    never run on a real job. The existing tests missed it because they seed
    ``job:<id>`` by hand.
    """
    CN._CACHE.clear()
    CN._publish_series_evidence("job-abc", {"ゼクス": "Zechs", "リリーナ": "Relena"})
    assert CN._CACHE.get("job:job-abc") == {"ゼクス": "Zechs", "リリーナ": "Relena"}
    # Fail-soft / no-op cases must not create junk entries.
    CN._publish_series_evidence("", {"a": "b"})
    CN._publish_series_evidence("job-empty", {})
    assert "job:" not in CN._CACHE
    assert "job:job-empty" not in CN._CACHE
    CN._CACHE.clear()


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
    assert CN._persist_load().get(key) == {
        "map": {"リリーナ": "Relena", "ゼクス": "Zechs"}, "provisional": False}
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


# ── The durable store must not pin a GUESS ────────────────────────────────

def test_store_is_versioned_so_older_entries_are_retired(tmp_path, monkeypatch):
    """A store written under older rules must be discarded, not reused.

    v1 persisted maps resolved from mined terms alone. A real deployment ended
    up with wrong names on disk ("Aires" for Aries, "Dorian" for Darlian),
    reused on every later run as "reused from the durable store" — and once the
    roster pass began working it ENFORCED them. Without a version the only cure
    was deleting the file by hand.
    """
    store = tmp_path / "canonical_names.json"
    monkeypatch.setattr(CN, "_persist_path", lambda: str(store))
    # An unversioned (v1-shaped) store is ignored wholesale.
    store.write_text(json.dumps({"entries": {"sig:x|y": {"エアリーズ": "Aires"}}}),
                     encoding="utf-8")
    assert CN._persist_load() == {}
    # A current-version store round-trips.
    CN._persist_put("sig:a|b", {"ゼクス": "Zechs"})
    assert CN._persist_load() == {
        "sig:a|b": {"map": {"ゼクス": "Zechs"}, "provisional": False}}
    assert json.loads(store.read_text(encoding="utf-8"))["v"] == CN._PERSIST_VERSION

    # A v2 store predates the provisional flag and only ever held ANCHORED
    # resolutions, so it reads forward as authoritative rather than being
    # thrown away — retiring it would cost the very determinism it bought.
    store.write_text(json.dumps({"v": 2, "entries": {"sig:v2": {"ゼクス": "Zechs"}}}),
                     encoding="utf-8")
    assert CN._persist_load() == {
        "sig:v2": {"map": {"ゼクス": "Zechs"}, "provisional": False}}


def test_terms_only_resolution_is_provisional_not_authoritative(tmp_path, monkeypatch):
    """With no informative title and no series hint the map is a guess.

    A guess must not be PINNED — every later run asks the model again, so a
    better answer always wins. It must not be thrown away either: v2 discarded
    it, and consecutive runs of one episode then shipped "Relena"/"Zechs" and
    "Lilyana"/"Zekus", because the second run's model call came back empty and
    had nothing to fall back on. Stored provisionally, it is a floor.
    """
    store = tmp_path / "canonical_names.json"
    monkeypatch.setattr(CN, "_persist_path", lambda: str(store))
    CN.clear_cache()

    calls = []

    class _Orch:
        async def text_completion(self, prompt, **kw):
            calls.append(prompt)
            return json.dumps({"ドーリアン": "Dorian", "エアリーズ": "Aires",
                               "ゼクス": "Zechs", "リリーナ": "Relena",
                               "ガンダム": "Gundam"})

    terms = ["ドーリアン", "エアリーズ", "ゼクス", "リリーナ", "ガンダム"]
    got = asyncio.run(resolve_canonical_names(
        terms, "videoplayback.mp4", _Orch(), job_id="job-guess"))
    assert got, "a terms-only resolution should still be USED for this run"
    assert len(calls) == 1
    stored = CN._persist_load()
    assert stored and all(e["provisional"] for e in stored.values()), stored

    # A later run with the same content re-asks (the guess does not
    # short-circuit) and the fresh answer replaces it.
    CN.clear_cache()
    asyncio.run(resolve_canonical_names(
        terms, "videoplayback.mp4", _Orch(), job_id="job-guess-2"))
    assert len(calls) == 2, "a provisional map must not short-circuit the model"

    # An INFORMATIVE title is authoritative: stored unmarked, and reused
    # without another call.
    CN.clear_cache()
    got2 = asyncio.run(resolve_canonical_names(
        terms, "MOBILE SUIT GUNDAM WING Episode 1", _Orch(), job_id="job-real"))
    assert got2 and len(calls) == 3
    CN.clear_cache()
    asyncio.run(resolve_canonical_names(
        terms, "MOBILE SUIT GUNDAM WING Episode 1", _Orch(), job_id="job-real-2"))
    assert len(calls) == 3, "an anchored map should be reused, not re-asked"


def test_stored_guess_covers_a_run_whose_model_returns_nothing(tmp_path, monkeypatch):
    """The whole point of keeping the guess: never regress to raw romanization.

    A real run's model call came back with no usable mapping and the episode
    shipped "Zekus"/"Lilyana"/"Hero" where the previous run had shipped the
    official spellings.
    """
    store = tmp_path / "canonical_names.json"
    monkeypatch.setattr(CN, "_persist_path", lambda: str(store))
    CN.clear_cache()
    terms = ["ドーリアン", "エアリーズ", "ゼクス", "リリーナ", "ガンダム"]

    class _Good:
        async def text_completion(self, prompt, **kw):
            return json.dumps({"ゼクス": "Zechs", "リリーナ": "Relena"})

    class _Empty:
        async def text_completion(self, prompt, **kw):
            return "{}"

    first = asyncio.run(resolve_canonical_names(
        terms, "videoplayback.mp4", _Good(), job_id="job-1"))
    assert first == {"ゼクス": "Zechs", "リリーナ": "Relena"}

    CN.clear_cache()
    second = asyncio.run(resolve_canonical_names(
        terms, "videoplayback.mp4", _Empty(), job_id="job-2"))
    assert second == first, "the empty run should fall back, not ship romaji"


# ── A canonical name must SOUND like the katakana it replaces ──────────────

def test_canonical_mapping_must_sound_like_the_katakana():
    """A canonical mapping is a SPELLING of the same name, not a translation.

    Without this the model was free to answer with any name from the series it
    thought it recognised. A real run returned エアリーズ (romaji "earizu" —
    Aries, a mobile suit) as "Peacecraft" (a person's surname) and レン as
    "Heero", and both shipped: the transcript called Relena Darlian "Relena
    Peacecraft" and misattributed dialogue. A ratio alone cannot catch it —
    "marina" scores 0.5 against "Relena" — so the LEADING SOUND has to match.
    """
    from backend.services.canonical_names import _canonical_sounds_plausible as ok
    for term, value in [
        ("リリーナ", "Relena"), ("ゼクス", "Zechs"), ("ガンダム", "Gundam"),
        ("マリーナ", "Marina"), ("ヒイロ", "Heero"), ("デュオ", "Duo"),
        ("トロワ", "Trowa"), ("ガンダニウム", "Gundanium"),
        # Equivalent spellings of one sound must still pass:
        ("エアリーズ", "Aries"),      # vowel-initial either way (e / a)
        ("カトル", "Quatre"),         # k / q
        ("コロニー", "Colony"),       # k / c
        ("ウーフェイ", "Wufei"),       # vowel / w glide
        ("ドーリアン", "Darlian"), ("トレーズ", "Treize"),
    ]:
        assert ok(term, value), f"{term} -> {value} must be allowed"

    for term, value in [
        ("エアリーズ", "Peacecraft"),  # the shipped failure
        ("レン", "Heero"),            # the other shipped failure
        ("マリーナ", "Relena"),        # the recurring ship-to-character swap
        ("ゼクス", "Trowa"), ("ガンダム", "Deathscythe"), ("トロワ", "Quatre"),
    ]:
        assert not ok(term, value), f"{term} -> {value} must be rejected"


def test_latin_terms_are_not_subject_to_the_sound_gate():
    """Terms already in Latin script are covered by the other rules; the kana
    gate must not start rejecting them."""
    from backend.services.canonical_names import _canonical_sounds_plausible as ok
    assert ok("Ririna", "Relena")
    assert ok("Hero Yuu", "Heero Yuy")
    assert ok("Shuttle", "Shuttle")


def test_kana_romaji_covers_the_shapes_names_actually_use():
    from backend.services.canonical_names import _kana_to_romaji as r
    assert r("リリーナ") == "ririna"          # long vowel mark dropped
    assert r("デュオ") == "deyuo"             # small-yu digraph
    assert r("ガンダニウム") == "gandaniumu"   # voiced + n
    assert r("ウーフェイ") == "ufei"           # fe digraph
    assert r("Relena") == "Relena"           # non-kana passes through


# ── Second chance for terms the first resolution left behind ───────────────
# A measured run resolved 9/17 terms (Relena, Zechs, OZ, Gundam…) and left
# ヒイロ / ドーリアン / トロワ unresolved. Those leftovers are exactly the
# ground truth the downstream garble corrector needs: with no "Heero Yuy" in
# the known-names set, "Hero Yu" → "Heero Yuy" is uncorroborated and dies in
# vetting. Once ≥2 names are resolved the work is identified, so re-asking
# about the remainder with those anchors is retrieval, not guessing.

def test_partial_resolution_gets_a_second_chance(tmp_path, monkeypatch):
    store = tmp_path / "cn.json"
    monkeypatch.setattr(CN, "_persist_path", lambda: str(store))
    CN.clear_cache()
    calls = []

    class _Orch:
        async def text_completion(self, prompt, **kw):
            calls.append(prompt)
            if len(calls) == 1:
                # First call: partial — knows the leads, misses the rest.
                return json.dumps({"リリーナ": "Relena", "ゼクス": "Zechs"})
            # Second call: anchors in-context → resolves the leftovers,
            # including one cross-name swap the gate must kill.
            assert "Relena" in prompt and "Zechs" in prompt
            assert "romaji" in prompt
            return json.dumps({"series": "Mobile Suit Gundam Wing",
                               "mappings": {"ヒイロ": "Heero Yuy",
                                            "ドーリアン": "Darlian",
                                            "エアリーズ": "Peacecraft"}})

    terms = ["リリーナ", "ゼクス", "ヒイロ", "ドーリアン", "エアリーズ"]
    got = asyncio.run(resolve_canonical_names(
        terms, "videoplayback.mp4", _Orch(), job_id="job-2nd"))
    assert len(calls) == 2
    assert got["リリーナ"] == "Relena" and got["ゼクス"] == "Zechs"
    assert got["ヒイロ"] == "Heero Yuy"
    assert got["ドーリアン"] == "Darlian"
    # The phonetic gate still rules the second answer: エアリーズ (earizu)
    # does not sound like "Peacecraft", no matter how confident the model is.
    assert "エアリーズ" not in got


def test_no_second_chance_without_anchors(tmp_path, monkeypatch):
    # One resolved name identifies nothing — a second ask would be a guess
    # about a guess, so it must not happen.
    store = tmp_path / "cn.json"
    monkeypatch.setattr(CN, "_persist_path", lambda: str(store))
    CN.clear_cache()
    calls = []

    class _Orch:
        async def text_completion(self, prompt, **kw):
            calls.append(prompt)
            return json.dumps({"ゼクス": "Zechs"})

    terms = ["ゼクス", "ヒイロ", "ドーリアン", "エアリーズ", "リリーナ"]
    got = asyncio.run(resolve_canonical_names(
        terms, "videoplayback.mp4", _Orch(), job_id="job-1anchor"))
    assert len(calls) == 1
    assert got == {"ゼクス": "Zechs"}


def test_second_chance_names_reach_the_roster_corroboration_set(tmp_path, monkeypatch):
    """The whole point: leftovers resolved on the second pass land in the
    per-job series evidence, so the roster garble pass can corroborate
    'Hero Yu' → 'Heero Yuy' even though the transcript never once spelled it
    right (attestation 0-vs-N)."""
    store = tmp_path / "cn.json"
    monkeypatch.setattr(CN, "_persist_path", lambda: str(store))
    CN.clear_cache()

    class _Orch:
        async def text_completion(self, prompt, **kw):
            if "Remaining terms:" not in prompt:
                return json.dumps({"リリーナ": "Relena", "ゼクス": "Zechs"})
            return json.dumps({"series": "Mobile Suit Gundam Wing",
                               "mappings": {"ヒイロ": "Heero Yuy",
                                            "トロワ": "Trowa"}})

    # ≥4 terms: with a generic filename, fewer can't fingerprint the work
    # and the resolver correctly refuses to call the model at all.
    terms = ["リリーナ", "ゼクス", "ヒイロ", "トロワ"]
    asyncio.run(resolve_canonical_names(
        terms, "videoplayback.mp4", _Orch(), job_id="job-evidence"))
    evidence = CN._CACHE.get("job:job-evidence") or {}
    assert evidence.get("ヒイロ") == "Heero Yuy"
    # And the vet accepts the unattested correction once corroborated.
    pairs = [{"wrong": "Hero Yu", "right": "Heero Yuy"}]
    corpus = "Hero Yu said hello. Hero Yu sat down."     # right form attested 0×
    vetted = CN._vet_roster_pairs(
        pairs, {"Hero Yu"}, corpus=corpus,
        known_names=set(evidence.values()))
    assert vetted.get("Hero Yu") == "Heero Yuy"


def test_latin_term_cannot_canonicalize_into_an_ordinary_word():
    """The phonetic gate only bites on katakana sources; a Latin-script term
    could be "canonicalized" into plain English vocabulary — a real run mapped
    "Justlove" → "Justice", turning a mishearing into an ordinary word that
    reads as dialogue. A Latin source may only be RESPELLED (a near-variant)."""
    from backend.services.canonical_names import _sanitize_mapping
    out = _sanitize_mapping(
        {"Justlove": "Justice", "Uing": "Wing", "Relena Dorian": "Relena Darlian"},
        ["Justlove", "Uing", "Relena Dorian"])
    assert "Justlove" not in out, out
    # A genuine respelling into a (list-)ordinary word survives: it keeps
    # nearly the whole spelling.
    assert out.get("Uing") == "Wing"
    # Ordinary-word guard leaves real name-to-name respellings alone.
    assert out.get("Relena Dorian") == "Relena Darlian"


def test_katakana_terms_keep_their_phonetic_respellings():
    from backend.services.canonical_names import _sanitize_mapping
    out = _sanitize_mapping({"ヒイロ": "Heero"}, ["ヒイロ"])
    assert out.get("ヒイロ") == "Heero"


_WIKI_FIXTURE = """
The story follows Heero Yuy, a pilot trained by Doctor J. Along the way Heero Yuy
meets Relena Darlian, who later learns she is Relena Peacecraft. The pilot Zechs
Merquise, brother of Relena Darlian, flies the Tallgeese. Other pilots include
Duo Maxwell, Trowa Barton, Quatre Raberba Winner and Chang Wufei. The mass-produced
Leo and the aerial Aries are fielded by OZ under Treize Khushrenada. Lady Une
serves Treize Khushrenada. The Gundam Deathscythe is flown by Duo Maxwell, and
the Sandrock by Quatre Raberba Winner. Later the Aries and the Leo appear again
when Zechs Merquise attacks. In battle Chang Wufei pilots the Shenlong.
The series aired in March and April.
"""


def test_glossary_mining_finds_cast_and_skips_wiki_prose():
    from backend.services.canonical_names import _mine_glossary_names
    names = _mine_glossary_names(_WIKI_FIXTURE)
    joined = " ".join(names)
    assert "Heero Yuy" in names
    assert "Relena Darlian" in names
    assert "Aries" in names and "Leo" in names
    assert "March" not in joined and "April" not in joined


def test_glossary_snaps_misspelled_values_to_official_forms():
    # The measured knowledge ceiling: the model identifies Gundam Wing every
    # run but ships "Hero", "Dorian", "Aires". Spelling is retrieval.
    from backend.services.canonical_names import (
        _apply_glossary_spellings, _glossary_resolve_terms, _mine_glossary_names)
    names = _mine_glossary_names(_WIKI_FIXTURE)
    snapped, n = _apply_glossary_spellings(
        {"ヒーロ": "Hero", "ドーリアン": "Dorian", "エアリーズ": "Aires",
         "ゼクス": "Zechs"}, names)
    assert snapped["ヒーロ"] == "Heero"
    assert snapped["ドーリアン"] == "Darlian"
    assert snapped["エアリーズ"] == "Aries"
    assert snapped["ゼクス"] == "Zechs"        # nothing close → untouched
    assert n == 3
    # A term the model never answered resolves phonetically from the glossary.
    assert _glossary_resolve_terms(["カトル"], names) == {"カトル": "Quatre"}


def test_vowel_heavy_katakana_passes_the_sound_gate_by_skeleton():
    # earizu vs "Aires" scored 0.18 (under the 0.22 floor) and the gate dropped
    # a correct-but-misspelled answer; the consonant skeleton (rz vs rs) is a
    # class match, which now carries it.
    from backend.services.canonical_names import _canonical_sounds_plausible as ok
    assert ok("エアリーズ", "Aires")
    assert ok("エアリーズ", "Aries")
    assert not ok("エアリーズ", "Peacecraft")   # cross-name swap still dies


def test_series_glossary_uses_durable_cache_without_network(tmp_path, monkeypatch):
    import asyncio, json as _json
    from backend.services import canonical_names as CN
    p = tmp_path / "series_glossaries.json"
    p.write_text(_json.dumps({"gundam wing": ["Heero Yuy", "Relena Darlian"]}))
    monkeypatch.setattr(CN, "_GLOSSARY_STORE_PATH", str(p))
    # No httpx call is possible in this test env — a cache hit must not need one.
    names = asyncio.run(CN._get_series_glossary("Gundam Wing"))
    assert names == ["Heero Yuy", "Relena Darlian"]


def test_glossary_identity_entries_reach_roster_evidence(monkeypatch):
    from backend.services import canonical_names as CN
    CN._cache_put("glossary:jobX", {"Heero Yuy": "Heero Yuy"})
    CN._publish_series_evidence("jobX", {"ゼクス": "Zechs"})
    ev = dict(CN._CACHE.get("job:jobX") or {})
    assert ev.get("ゼクス") == "Zechs"
    assert ev.get("Heero Yuy") == "Heero Yuy"
