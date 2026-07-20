"""Run-44 transcript-quality nets: roster garble correction, CJK source
dedup, and misplaced outro-stub removal.

Driven by the audited gaps vs the official Gundam Wing ep-1 subs:
  * "Ail Reese"/"Gundarium"/"Hero Yui"/"Trowat"/"Katru" shipped because the
    mined glossary only anchors RECURRING terms;
  * the same Japanese line re-decoded at 3:30/3:37 survived source dedup
    (whole-CJK sentences mine as ONE content word → the rule no-oped) and
    diverged lexically in translation;
  * "[6:15] Next time… / [6:18] Preview." — whisper hallucinating the
    eyecatch formula nine minutes into a 24-minute episode.
"""

import asyncio

import pytest

from backend.services.canonical_names import (
    _CACHE,
    _mine_name_candidates,
    _vet_roster_pairs,
    apply_roster_corrections,
    resolve_roster_corrections,
)
from backend.services.transcript_dedup import (
    drop_misplaced_outro_stubs,
    drop_repeated_sentences,
)


# ── Candidate mining ─────────────────────────────────────────────────


def test_mining_finds_garbled_names_and_skips_common_words():
    texts = [
        "We’ll dispatch reinforcements as soon as Ail Reese is ready.",
        "Impressive work, Sex, unique… so sudden.",
        "Based on the strength, it can only be Gundarium alloy.",
        "This is Recorder Trowat, identifying myself.",
        "This is Katru. We destroyed the Captain's Mobile Suit.",
        "Return to the trees and try to contain it on Earth.",
        "The capsule fell. But there's nothing here.",
    ]
    cands = {t for t, _ex in _mine_name_candidates(texts)}
    assert "Ail Reese" in cands
    assert "Gundarium" in cands
    assert "Trowat" in cands
    assert "Katru" in cands
    # "But"/"The" are safelisted; "trees" is lowercase (never a candidate).
    assert "But" not in cands and "The" not in cands
    assert not any("trees" in c.lower() for c in cands)


def test_mining_requires_mid_sentence_for_single_words():
    # "Sex" mid-sentence (after a comma) is a candidate; a word that only
    # ever opens sentences is not.
    texts = ["Impressive work, Sex, unique…", "Perhaps we should go."]
    cands = {t for t, _ in _mine_name_candidates(texts)}
    assert "Sex" in cands
    assert "Perhaps" not in cands


def test_mining_skips_words_also_used_lowercase():
    texts = ["The Cut was deep.", "Please cut the rope."]
    cands = {t for t, _ in _mine_name_candidates(texts)}
    assert "Cut" not in cands


# ── Vetting + apply ──────────────────────────────────────────────────


def test_vetting_rejects_unmined_and_ugly_corrections():
    cands = {"Ail Reese", "Gundarium", "Trowat"}
    pairs = [
        {"wrong": "Ail Reese", "right": "Aries"},
        {"wrong": "Gundarium", "right": "Gundanium"},
        {"wrong": "Trowat", "right": "Trowa"},
        {"wrong": "Zechs", "right": "Sechs"},          # not mined → rejected
        {"wrong": "Gundarium", "right": "gundarium"},  # case-only → rejected
        {"wrong": "Trowat", "right": "a" * 50},        # too long → rejected
        {"wrong": "Ail Reese", "right": "第2話"},       # non-Latin → rejected
    ]
    out = _vet_roster_pairs(pairs, cands)
    assert out == {"Ail Reese": "Aries", "Gundarium": "Gundanium",
                   "Trowat": "Trowa"}


def test_apply_word_boundaries_and_possessives():
    texts = [
        "We’ll dispatch reinforcements as soon as Ail Reese is ready.",
        "It can only be Gundarium alloy.",
        "Miina's carrier will take time.",
        "Gundariums is not a word we touch here: Gundariumsuffix stays.",
    ]
    mapping = {"Ail Reese": "Aries", "Gundarium": "Gundanium",
               "Miina": "Marina"}
    out, n = apply_roster_corrections(texts, mapping)
    assert out[0] == "We’ll dispatch reinforcements as soon as Aries is ready."
    assert out[1] == "It can only be Gundanium alloy."
    assert out[2] == "Marina's carrier will take time."
    # No mid-word replacement.
    assert "Gundariumsuffix stays" in out[3]
    assert n == 3


def test_resolver_requires_series_evidence(monkeypatch):
    # With no cached canonical mapping for the job, the pass refuses to run
    # (no series context → too risky) — no LLM call is made.
    _CACHE.pop("job:test-no-evidence", None)

    class _Boom:
        async def text_completion(self, *a, **k):
            raise AssertionError("must not be called")

    out = asyncio.run(resolve_roster_corrections(
        ["Some Text with Gundarium"], _Boom(), job_id="test-no-evidence"))
    assert out == {}


def test_resolver_end_to_end_with_fake_llm():
    _CACHE["job:test-roster"] = {
        "ガンダム": "Gundam", "ゼクス": "Zechs", "リリーナ": "Relena",
    }

    class _Fake:
        def __init__(self):
            self.prompts = []

        async def text_completion(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return ('[{"wrong": "Gundarium", "right": "Gundanium"},'
                    ' {"wrong": "Nonexistent", "right": "Nope"}]')

    fake = _Fake()
    out = asyncio.run(resolve_roster_corrections(
        ["It can only be Gundarium alloy, Zechs."],
        fake, job_id="test-roster"))
    assert out == {"Gundarium": "Gundanium"}
    # The prompt carried the series evidence.
    assert "Zechs" in fake.prompts[0]
    # The mapping is stashed per job so downstream copy generators (clip SEO
    # titles/descriptions/tags) can re-apply the same corrections.
    from backend.services.canonical_names import roster_corrections_for_job
    assert roster_corrections_for_job("test-roster") == {"Gundarium": "Gundanium"}
    assert roster_corrections_for_job("") == {}
    _CACHE.pop("job:test-roster", None)
    _CACHE.pop("roster:test-roster", None)


# ── CJK source-side sentence dedup ───────────────────────────────────


def test_cjk_redecode_duplicate_drops():
    # The 3:30 re-decode: same utterance decoded twice with tiny variation.
    a = "監視衛星の観測は不十分だと言っている。"
    b = "監視衛星の観測は不十分だと言っています。"
    segs = [
        {"text": a, "start": 200.0},
        {"text": "大気圏突入のウェーブコースに流星が乗ると思うか?" + b,
         "start": 210.0},
    ]
    out, dropped = drop_repeated_sentences(segs)
    assert dropped == 1
    assert b not in out[1]["text"]
    assert "大気圏突入のウェーブコース" in out[1]["text"]


def test_cjk_distinct_sentences_survive():
    segs = [
        {"text": "ラグランジュポイントより移動物体を確認。", "start": 10.0},
        {"text": "大気圏突入まで600秒と推定される。", "start": 14.0},
        {"text": "レーダーには5つの金属反応がある。", "start": 18.0},
    ]
    out, dropped = drop_repeated_sentences(segs)
    assert dropped == 0
    assert len(out) == 3


def test_cjk_short_interjections_exempt():
    segs = [
        {"text": "はい。", "start": 1.0},
        {"text": "はい。", "start": 3.0},
    ]
    _out, dropped = drop_repeated_sentences(segs)
    assert dropped == 0


# ── Misplaced outro stubs ────────────────────────────────────────────


def _stub_segs():
    return [
        {"text": "第1話が始まる。", "start": 10.0, "end": 15.0},
        {"text": "Next time...", "start": 375.0, "end": 377.0},
        {"text": "Preview.", "start": 378.0, "end": 380.0},
        {"text": "次回", "start": 500.0, "end": 502.0},
        {"text": "Real dialogue continues.", "start": 900.0, "end": 903.0},
        {"text": "Next time, on Gundam Wing: the Gundam Deathscythe.",
         "start": 1430.0, "end": 1440.0},
        {"text": "To be continued", "start": 1445.0, "end": 1467.0},
    ]


def test_mid_episode_outro_stubs_drop_but_real_preview_survives():
    out, dropped = drop_misplaced_outro_stubs(_stub_segs())
    texts = [s["text"] for s in out]
    assert dropped == 3
    assert "Next time..." not in texts
    assert "Preview." not in texts
    assert "次回" not in texts
    # In the final 15% of the timeline: legitimate outro content stays.
    assert "To be continued" in texts
    # Sentences that merely START with the formula are never touched.
    assert any("Deathscythe" in t for t in texts)


def test_outro_stub_needs_enough_cues():
    segs = [{"text": "Preview.", "start": 1.0, "end": 2.0}]
    _out, dropped = drop_misplaced_outro_stubs(segs)
    assert dropped == 0


# ── Over-CPS condensation ────────────────────────────────────────────


def test_condense_targets_only_unreadable_cues():
    from backend.services.pipeline import _condense_over_cps_cues

    segs = [
        {"text": "A short readable line.", "start": 0.0, "end": 3.0},
        {"text": "This enormously long narration sentence describes the whole "
                 "operation in exhausting detail nobody can read.",
         "start": 5.0, "end": 6.0},          # ~109 chars in 1s → offender
        {"text": "[♪ music ♪]", "start": 7.0, "end": 8.0},
    ]

    class _Fake:
        def __init__(self):
            self.prompts = []

        async def text_completion(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return '["The narration, condensed."]'

    fake = _Fake()
    out, n = asyncio.run(_condense_over_cps_cues(segs, fake, job_id="t"))
    assert n == 1
    assert out[1]["text"] == "The narration, condensed."
    assert out[0]["text"] == "A short readable line."   # untouched
    assert out[2]["text"] == "[♪ music ♪]"              # markers untouched
    assert len(fake.prompts) == 1 and "max" in fake.prompts[0]


def test_condense_keeps_original_when_llm_fails_to_shrink():
    from backend.services.pipeline import _condense_over_cps_cues

    long_line = ("Another enormously long narration sentence that describes "
                 "the whole operation in exhausting, unreadable detail.")
    segs = [{"text": long_line, "start": 0.0, "end": 1.0}]

    class _Fake:
        async def text_completion(self, prompt, **kwargs):
            # Longer than the original AND an empty string — both rejected.
            return '["' + long_line + ' and even more words on top"]'

    out, n = asyncio.run(_condense_over_cps_cues(segs, _Fake(), job_id="t"))
    assert n == 0
    assert out[0]["text"] == long_line


def test_condense_noop_without_offenders():
    from backend.services.pipeline import _condense_over_cps_cues

    class _Boom:
        async def text_completion(self, *a, **k):
            raise AssertionError("must not be called")

    segs = [{"text": "Fine.", "start": 0.0, "end": 2.0}]
    out, n = asyncio.run(_condense_over_cps_cues(segs, _Boom(), job_id="t"))
    assert n == 0 and out[0]["text"] == "Fine."


# ── Cloud-provider parity: response_format + translation concurrency ─


def test_openrouter_response_format_mapping():
    from backend.services.providers.openrouter_provider import _response_format_for

    schema = {"type": "array", "items": {"type": "string"},
              "minItems": 3, "maxItems": 3}
    rf = _response_format_for(False, schema)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == schema
    assert _response_format_for(True, None) == {"type": "json_object"}
    assert _response_format_for(False, None) is None


def test_openrouter_rejection_detector():
    from backend.services.providers.openrouter_provider import (
        _is_response_format_rejection)

    assert _is_response_format_rejection(
        Exception("400: response_format is not supported for this model"))
    assert _is_response_format_rejection(
        Exception("Invalid schema in json_schema field"))
    assert not _is_response_format_rejection(Exception("429 rate limited"))
    assert not _is_response_format_rejection(Exception("500 server error"))


def test_openrouter_text_complete_accepts_json_kwargs():
    # The orchestrator forwards json_schema/json_mode to openrouter — the
    # signature must accept them (a mismatch would silently fall through to
    # the next provider via the generic except).
    import inspect
    from backend.services.providers.openrouter_provider import OpenRouterProvider

    params = inspect.signature(OpenRouterProvider.text_complete).parameters
    assert "json_mode" in params and "json_schema" in params


def test_translation_plan_cloud_concurrency():
    from backend.services.local_models import translation_plan

    # Ollama keeps the Companion-derived parallelism.
    p = translation_plan("qwen3:4b", 100, is_ollama=True, companion_parallel=1)
    assert p["concurrency"] == 1
    # Cloud small model: fans out to the configured cloud concurrency.
    p = translation_plan("some/cloud-model", 100, is_ollama=False, companion_parallel=1)
    assert p["concurrency"] >= 3
    # Cloud LARGE model: no single-slot VRAM logic either.
    p = translation_plan("llama-3.1-70b", 100, is_ollama=False, companion_parallel=1)
    assert p["concurrency"] >= 3
    # Ollama large model keeps the budget-gated single slot default.
    p = translation_plan("gemma3:12b", 100, is_ollama=True, companion_parallel=1)
    assert p["concurrency"] == 1
