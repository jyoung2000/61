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
    cands = {t for t, _ex, _c in _mine_name_candidates(texts)}
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
    cands = {t for t, _ex, _c in _mine_name_candidates(texts)}
    assert "Sex" in cands
    assert "Perhaps" not in cands


def test_mining_skips_words_also_used_lowercase():
    texts = ["The Cut was deep.", "Please cut the rope."]
    cands = {t for t, _ex, _c in _mine_name_candidates(texts)}
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


# ── Run-45 follow-ups ────────────────────────────────────────────────


def test_block_redecode_99s_later_drops_with_wide_window():
    # Run 45 shipped the 5:45 capsule scene again at 7:24 (99 s later) with
    # different cue boundaries — outside the default 25 s window. The
    # source-side pass runs with window_s=120.
    a = "カプセルが軌道を変更した。自殺行為のつもりか?"
    segs = [
        {"text": a, "start": 345.0, "no_speech_prob": 0.1},
        {"text": "別の台詞がここに入ります、確認します。", "start": 380.0,
         "no_speech_prob": 0.1},
        {"text": a, "start": 444.0, "no_speech_prob": 0.1},
    ]
    out, dropped = drop_repeated_sentences(segs, window_s=120.0)
    assert dropped >= 1
    assert len(out) == 2


def test_sung_cues_exempt_from_wide_window():
    # The OP reprises its opening lines ~50 s later — high no_speech_prob
    # marks them as sung; neither copy may be dropped.
    lyric = "雨に打たれながら燃える想いを伝えたい今夜。"
    segs = [
        {"text": lyric, "start": 30.0, "no_speech_prob": 0.85},
        {"text": lyric, "start": 80.0, "no_speech_prob": 0.85},
    ]
    out, dropped = drop_repeated_sentences(segs, window_s=120.0)
    assert dropped == 0
    assert len(out) == 2


def test_stage_direction_runs_collapse():
    from backend.services.transcript_dedup import collapse_stage_direction_runs

    segs = [
        {"text": "(Heavy breathing)", "start": 604.0, "end": 607.0},
        {"text": "(Exhausted gasps)", "start": 611.0, "end": 614.0},
        {"text": "(More heavy breathing)", "start": 619.0, "end": 622.0},
        {"text": "I'm sorry for worrying you.", "start": 634.0, "end": 637.0},
        {"text": "(A single direction elsewhere stays)", "start": 700.0,
         "end": 703.0},
        {"text": "[♪ music ♪]", "start": 710.0, "end": 715.0},
    ]
    out, dropped = collapse_stage_direction_runs(segs)
    texts = [s["text"] for s in out]
    assert dropped == 2
    assert texts.count("(Heavy breathing)") == 1
    assert "(Exhausted gasps)" not in texts
    # The kept head stretched over the dropped run.
    assert out[0]["end"] == 622.0
    # Lone directions and bracket markers untouched.
    assert "(A single direction elsewhere stays)" in texts
    assert "[♪ music ♪]" in texts


def test_roster_call_uses_json_schema():
    _CACHE["job:test-schema"] = {
        "ガンダム": "Gundam", "ゼクス": "Zechs", "リリーナ": "Relena",
    }

    class _Fake:
        def __init__(self):
            self.kwargs = None

        async def text_completion(self, prompt, **kwargs):
            self.kwargs = kwargs
            return '[{"wrong": "Gundarium", "right": "Gundanium"}]'

    fake = _Fake()
    out = asyncio.run(resolve_roster_corrections(
        ["It can only be Gundarium alloy."], fake, job_id="test-schema"))
    assert out == {"Gundarium": "Gundanium"}
    # Grammar-level array schema, not json_mode (format=json biases toward a
    # bare object and run 45 silently produced zero corrections on that).
    assert "json_schema" in (fake.kwargs or {})
    assert fake.kwargs["json_schema"]["type"] == "array"
    assert "json_mode" not in (fake.kwargs or {})
    _CACHE.pop("job:test-schema", None)
    _CACHE.pop("roster:test-schema", None)


# ── Run-47 overcorrection guards ─────────────────────────────────────


def test_vetting_rejects_unrelated_substitutions():
    # The run-47 rogue mappings: phonetically unrelated "corrections" that
    # swapped in different characters/terms or broke correct ones.
    cands = {"Hero Yuu", "Earth Sphere Alliance", "Operation Meteor",
             "Gundarium", "Ail Reese", "Miina"}
    pairs = [
        {"wrong": "Hero Yuu", "right": "Trowa Barton"},          # wrong character
        {"wrong": "Earth Sphere Alliance", "right": "Zeon"},     # wrong franchise
        {"wrong": "Operation Meteor", "right": "Operation Endgame"},  # shared prefix, unrelated core
        {"wrong": "Gundarium", "right": "Gundanium"},            # real mishearing
        {"wrong": "Ail Reese", "right": "Aries"},                # real mishearing
        {"wrong": "Miina", "right": "Marina"},                   # real mishearing
    ]
    out = _vet_roster_pairs(pairs, cands)
    assert "Hero Yuu" not in out
    assert "Earth Sphere Alliance" not in out
    assert "Operation Meteor" not in out
    assert out["Gundarium"] == "Gundanium"
    assert out["Ail Reese"] == "Aries"
    assert out["Miina"] == "Marina"


def test_vetting_rejects_common_word_and_near_typo_rights():
    # Run-50 corruptions from a degraded model, both of which PASS the
    # phonetic gate: the ending-song title "Justlove" rewritten into the
    # ordinary word "Justice" (corrupting every lyric line it appeared in),
    # and the CORRECT name "General Septem" "corrected" into the typo
    # "Gneral Septem" ('gneral' is one transposition from 'general').
    cands = {"Justlove", "General Septem", "Gundarium", "Hero-kun"}
    pairs = [
        {"wrong": "Justlove", "right": "Justice"},
        {"wrong": "General Septem", "right": "Gneral Septem"},
        {"wrong": "Gundarium", "right": "Gundanium"},
        {"wrong": "Hero-kun", "right": "Heero"},
    ]
    out = _vet_roster_pairs(pairs, cands)
    assert "Justlove" not in out
    assert "General Septem" not in out
    # Real mis-hearings still correct.
    assert out["Gundarium"] == "Gundanium"
    assert out["Hero-kun"] == "Heero"


def test_vetting_frequency_gate_and_cap():
    cands = {f"Tok{i}" for i in range(12)} | {"Deathscythe"}
    pairs = [{"wrong": f"Tok{i}", "right": f"Toc{i}"} for i in range(12)]
    pairs.append({"wrong": "Deathscythe", "right": "Death Syche"})
    out = _vet_roster_pairs(pairs, cands, frequent={"Deathscythe"})
    # Frequent consistent terms are untouchable; fan-out capped at 8.
    assert "Deathscythe" not in out
    assert len(out) <= 8


def test_resolver_never_asks_about_frequent_terms():
    _CACHE["job:test-freq"] = {
        "ガンダム": "Gundam", "ゼクス": "Zechs", "リリーナ": "Relena",
    }
    texts = ["The Operation Meteor plan begins now."] * 5 + [
        "It can only be Gundarium alloy."]

    class _Fake:
        def __init__(self):
            self.prompts = []

        async def text_completion(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return '[{"wrong": "Gundarium", "right": "Gundanium"}]'

    fake = _Fake()
    out = asyncio.run(resolve_roster_corrections(texts, fake, job_id="test-freq"))
    assert out == {"Gundarium": "Gundanium"}
    # "Operation Meteor" appears 5× — a consistent term, never offered.
    assert "Operation Meteor" not in fake.prompts[0]
    _CACHE.pop("job:test-freq", None)
    _CACHE.pop("roster:test-freq", None)


def test_roster_worth_asking_drops_noise_keeps_garbles():
    # Run 48: 50 candidates were offered and the busy 12B corrected NONE —
    # the real garbles were buried under ordinary-word noise. The filter must
    # keep the garbles and drop the chaff so the model's attention lands.
    from backend.services.canonical_names import _roster_worth_asking
    known_words = {"mobile", "suit", "relena", "oz", "colony",
                   "gundam", "shuttle", "miina", "zechs"}

    # Real garbles worth asking about — kept.
    for tok in ("Gundarium", "Katul", "Hero Yuu", "Deathbringer",
                "From Nag Ranch", "Am Wufe", "Leo"):
        assert _roster_worth_asking(tok, known_words), tok

    # All-ordinary phrases and sentence-opening common words — dropped.
    for tok in ("Especially", "Humanity", "Inform", "Combat Log",
                "Colony Summit", "Civilian Shuttle", "Eastern", "Part",
                "Alliance Headquarters"):
        assert not _roster_worth_asking(tok, known_words), tok

    # Phrases that merely CONTAIN an already-correct canonical word — dropped
    # (the term is right; the phrase is just it in context).
    for tok in ("Miss Relena", "Oz's Zechs", "New Mobile Report",
                "Gundam Wing", "Especially Miina", "Zechs Merquise"):
        assert not _roster_worth_asking(tok, known_words), tok


def test_resolver_noise_filter_shrinks_ask_and_reharden_correct_terms():
    # The noise filter also re-hardens the run-47 rogue path: a correct term
    # made of ordinary words ("Operation Meteor") is never offered, so the
    # model can't "correct" it into a hallucinated variant.
    _CACHE["job:test-noise"] = {
        "ガンダム": "Gundam", "ゼクス": "Zechs", "リリーナ": "Relena",
        "コロニー": "Colony", "マリーナ": "Miina",
    }
    texts = [
        "It can only be Gundarium alloy.",
        "Especially Miina's flagship approaches.",   # noise: Especially, known Miina
        "The Colony Summit was inconclusive.",       # noise: all ordinary/known
        "Inform Zechs at once.",                     # noise: contains known Zechs
        "Operation Meteor will proceed.",            # correct term, all-ordinary → not offered
        "The Gundam called Deathbringer.",           # real garble (passes phonetic gate)
    ]

    class _Fake:
        def __init__(self):
            self.prompts = []

        async def text_completion(self, prompt, **kwargs):
            self.prompts.append(prompt)
            return ('[{"wrong": "Gundarium", "right": "Gundanium"},'
                    ' {"wrong": "Deathbringer", "right": "Deathscythe"}]')

    fake = _Fake()
    out = asyncio.run(resolve_roster_corrections(texts, fake, job_id="test-noise"))
    assert out == {"Gundarium": "Gundanium", "Deathbringer": "Deathscythe"}
    prompt = fake.prompts[0]
    # Real garbles were offered…
    assert "Gundarium" in prompt and "Deathbringer" in prompt
    # …but the ordinary-word noise and correct-term phrases were withheld.
    assert "Especially" not in prompt
    assert "Colony Summit" not in prompt
    assert "Operation Meteor" not in prompt
    assert "Inform Zechs" not in prompt
    _CACHE.pop("job:test-noise", None)
    _CACHE.pop("roster:test-noise", None)


def test_reduce_summary_coercion_fills_missing_fields():
    # Run 47: the reduce produced a rich summary but omitted
    # content_category — pydantic rejected the whole dict and the UI got a
    # template overview with face-diagnostic topics. The coercion backstop
    # keeps the content.
    from backend.models import VideoSummary

    data = {"overview": "The video depicts the opening of a mecha conflict.",
            "key_topics": ["Gundam", "Colony", "Alliance"],
            "tone": "dramatic", "estimated_audience": "sci-fi fans"}
    merged = {"tone": "conversational", "estimated_audience": "general viewers",
              "content_category": "video content", "key_topics": []}
    merged.update({k: v for k, v in data.items() if v not in (None, "")})
    vs = VideoSummary(**merged)
    assert vs.overview.startswith("The video depicts")
    assert vs.content_category == "video content"
    assert vs.key_topics == ["Gundam", "Colony", "Alliance"]
