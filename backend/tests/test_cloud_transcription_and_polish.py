"""Phase 4 audit tests — cloud STT providers + subtitle-polish recommendation.

Covers: provider selection/availability, verbose_json → local schema
mapping, fallback-to-local on failure, the polish benchmark scorer, the
curated shortlist shape, and SUBTITLE_POLISH_MODEL precedence.
"""

import sys
import types

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

import pytest

from backend.config import settings  # noqa: E402
from backend.services import cloud_transcription as CT  # noqa: E402


# ── Provider selection ──────────────────────────────────────────────────

def test_provider_defaults_to_local():
    prev = settings.TRANSCRIPTION_PROVIDER
    try:
        settings.TRANSCRIPTION_PROVIDER = "local"
        assert CT.provider_selected() == "local"
        settings.TRANSCRIPTION_PROVIDER = "bogus"
        assert CT.provider_selected() == "local"
        settings.TRANSCRIPTION_PROVIDER = "GROQ"
        assert CT.provider_selected() == "groq"
    finally:
        settings.TRANSCRIPTION_PROVIDER = prev


def test_cloud_available_requires_key():
    prev_p, prev_g, prev_o = (settings.TRANSCRIPTION_PROVIDER,
                              settings.GROQ_API_KEY,
                              getattr(settings, "OPENAI_API_KEY", ""))
    try:
        settings.TRANSCRIPTION_PROVIDER = "groq"
        settings.GROQ_API_KEY = ""
        assert CT.cloud_available() is False
        settings.GROQ_API_KEY = "gsk_test"
        assert CT.cloud_available() is True
        settings.TRANSCRIPTION_PROVIDER = "openai"
        settings.OPENAI_API_KEY = ""
        assert CT.cloud_available() is False
        settings.OPENAI_API_KEY = "sk-test"
        assert CT.cloud_available() is True
    finally:
        settings.TRANSCRIPTION_PROVIDER = prev_p
        settings.GROQ_API_KEY = prev_g
        settings.OPENAI_API_KEY = prev_o


def test_transcribe_cloud_none_when_local():
    prev = settings.TRANSCRIPTION_PROVIDER
    try:
        settings.TRANSCRIPTION_PROVIDER = "local"
        assert CT.transcribe_cloud("/nonexistent.wav") is None
    finally:
        settings.TRANSCRIPTION_PROVIDER = prev


# ── verbose_json mapping ────────────────────────────────────────────────

def test_map_verbose_json_segments_and_words():
    data = {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " Hello there. ",
             "no_speech_prob": 0.02, "avg_logprob": -0.15},
            {"start": 2.0, "end": 4.0, "text": "Second line."},
        ],
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.6},
            {"word": "there.", "start": 0.7, "end": 1.4},
            {"word": "Second", "start": 2.1, "end": 2.6},
            {"word": "line.", "start": 2.7, "end": 3.4},
        ],
    }
    segs = CT._map_verbose_json(data)
    assert len(segs) == 2
    assert segs[0]["text"] == "Hello there."
    assert [w["word"] for w in segs[0]["words"]] == ["Hello", "there."]
    assert [w["word"] for w in segs[1]["words"]] == ["Second", "line."]
    # Local schema fields present
    for s in segs:
        for k in ("start_sec", "end_sec", "text", "words",
                  "is_hallucination", "no_speech_prob", "avg_logprob"):
            assert k in s
    assert segs[0]["no_speech_prob"] == pytest.approx(0.02)


def test_map_json_blob_without_words():
    # gpt-4o-transcribe style: text only — forced alignment re-times later
    data = {"text": "One long transcription blob.", "duration": 12.5}
    segs = CT._map_verbose_json(data)
    assert len(segs) == 1
    assert segs[0]["words"] == []
    assert segs[0]["end_sec"] == pytest.approx(12.5)


def test_map_empty_returns_nothing():
    assert CT._map_verbose_json({}) == []


# ── Polish benchmark scorer ─────────────────────────────────────────────

def test_benchmark_score_perfect():
    from backend.services.polish_benchmark import BENCH_CASES, score_results
    golds = [gold for _src, gold in BENCH_CASES]
    r = score_results(golds)
    assert r["exact_fix_rate"] == 1.0
    assert r["segment_count_ok"] is True
    assert r["format_compliance"] >= 0.9


def test_benchmark_score_unpolished():
    from backend.services.polish_benchmark import BENCH_CASES, score_results
    raws = [src for src, _gold in BENCH_CASES]
    r = score_results(raws)
    assert r["exact_fix_rate"] < 0.3  # returning input fixes ~nothing
    # ...but format compliance is perfect (right count, exact word budget)
    assert r["format_compliance"] == 1.0


def test_benchmark_score_wrong_count_and_verbose():
    from backend.services.polish_benchmark import BENCH_CASES, score_results
    # Model went off the rails: half the segments, each rewritten long
    outs = ["this is a very long paraphrase that blows the word budget "
            "completely out of range"] * (len(BENCH_CASES) // 2)
    r = score_results(outs)
    assert r["segment_count_ok"] is False
    assert r["format_compliance"] < 0.5


# ── Shortlist data shape ────────────────────────────────────────────────

def test_shortlist_is_valid_data():
    from backend.services.providers.openrouter_provider import (
        SUBTITLE_POLISH_SHORTLIST)
    assert len(SUBTITLE_POLISH_SHORTLIST) >= 8
    tiers = set()
    for entry in SUBTITLE_POLISH_SHORTLIST:
        assert entry["id"] and "/" in entry["id"]
        assert entry["tier"] in ("free", "efficient", "premium")
        assert len(entry["rationale"]) > 15
        tiers.add(entry["tier"])
    assert tiers == {"free", "efficient", "premium"}
    ids = [e["id"] for e in SUBTITLE_POLISH_SHORTLIST]
    assert len(set(ids)) == len(ids)
    # free tier ids must actually be free ids
    for e in SUBTITLE_POLISH_SHORTLIST:
        if e["tier"] == "free":
            assert e["id"].endswith(":free")


# ── SUBTITLE_POLISH_MODEL precedence ────────────────────────────────────

def test_polish_model_pin_outranks_translation_default():
    # Stub provider SDKs so pipeline imports cleanly (same pattern as
    # test_polish_uses_translation_model.py)
    for name, attr in (("groq", "AsyncGroq"), ("openai", "AsyncOpenAI"),
                       ("anthropic", "AsyncAnthropic")):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            setattr(mod, attr, object)
            sys.modules[name] = mod
    if "google.generativeai" not in sys.modules:
        g = types.ModuleType("google")
        gg = types.ModuleType("google.generativeai")
        gg.configure = lambda *a, **k: None
        gg.GenerativeModel = object
        g.generativeai = gg
        sys.modules.setdefault("google", g)
        sys.modules["google.generativeai"] = gg

    from backend.services.pipeline import _resolve_polish_model_override
    prev = getattr(settings, "SUBTITLE_POLISH_MODEL", "")
    try:
        settings.SUBTITLE_POLISH_MODEL = "anthropic/claude-haiku-4.5"
        assert _resolve_polish_model_override(None) == "anthropic/claude-haiku-4.5"
        settings.SUBTITLE_POLISH_MODEL = ""
        # falls back to the legacy translation-model path (may be None
        # without an orchestrator — must not raise)
        _resolve_polish_model_override(None)
    finally:
        settings.SUBTITLE_POLISH_MODEL = prev


def test_phase4_flag_defaults():
    assert settings.TRANSCRIPTION_PROVIDER == "local"
    assert settings.GROQ_TRANSCRIBE_MODEL == "whisper-large-v3-turbo"
    assert settings.OPENAI_TRANSCRIBE_MODEL == "whisper-1"
    assert settings.SUBTITLE_POLISH_MODEL == ""
