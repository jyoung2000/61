"""Tests for the optional CPU 'quality mode' translation (Task 5).

``quality`` mode routes the offline PRIMARY translation through a larger Ollama
model on CPU (TRANSLATION_QUALITY_MODEL) via translate_via_llm, with NLLB as the
completeness backstop for any cue the big model leaves in the source language.
``speed`` mode (default) is unaffected — it must never enter this path.
"""

import asyncio
import sys
import types

import pytest


def _stub_provider_sdks():
    if "google.generativeai" not in sys.modules:
        g = types.ModuleType("google")
        gg = types.ModuleType("google.generativeai")
        gg.configure = lambda *a, **k: None
        gg.GenerativeModel = object
        g.generativeai = gg
        sys.modules.setdefault("google", g)
        sys.modules["google.generativeai"] = gg
    if "groq" not in sys.modules:
        gr = types.ModuleType("groq")
        gr.AsyncGroq = object
        sys.modules["groq"] = gr


_stub_provider_sdks()

from backend.models import TranscriptSegment  # noqa: E402
from backend.services import translator as T  # noqa: E402


def _seg(start, end, text, speaker="Speaker 1"):
    return TranscriptSegment(start=start, end=end, text=text, speaker=speaker)


def _source():
    return [
        _seg(0.0, 1.4, "彼は店に行った。"),
        _seg(1.4, 3.2, "彼女は本を買った。"),
        _seg(3.2, 5.0, "彼らは一緒に家に帰った。"),
    ]


# ── gate ──────────────────────────────────────────────────────────────────

def test_quality_mode_inactive_by_default(monkeypatch):
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODE", "speed", raising=False)
    monkeypatch.setattr(T.settings, "OLLAMA_HOST", "http://h", raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODEL", "qwen2.5:7b-instruct", raising=False)
    assert T.translation_quality_mode_active() is False


def test_quality_mode_active_when_configured(monkeypatch):
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODE", "quality", raising=False)
    monkeypatch.setattr(T.settings, "OLLAMA_HOST", "http://h", raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODEL", "qwen2.5:7b-instruct", raising=False)
    assert T.translation_quality_mode_active() is True


def test_quality_mode_inactive_without_host_or_model(monkeypatch):
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODE", "quality", raising=False)
    monkeypatch.setattr(T.settings, "OLLAMA_HOST", "", raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODEL", "qwen2.5:7b-instruct", raising=False)
    assert T.translation_quality_mode_active() is False


# ── routing + backstop ─────────────────────────────────────────────────────

@pytest.fixture()
def _quality_settings(monkeypatch):
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODE", "quality", raising=False)
    monkeypatch.setattr(T.settings, "OLLAMA_HOST", "http://localhost:11434", raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODEL", "qwen2.5:7b-instruct", raising=False)
    monkeypatch.setattr(T.settings, "OFFLINE_TRANSLATION_MTPE_NUM_CTX", 8192, raising=False)
    monkeypatch.setattr(T.settings, "NMT_AUTODOWNLOAD", True, raising=False)


def test_quality_mode_uses_big_model_and_nllb_backstop(monkeypatch, _quality_settings):
    cap = {}

    async def fake_llm(segments, src, tgt, orchestrator, glossary=None, **kw):
        cap["client_model"] = orchestrator._model
        cap["client_ctx"] = orchestrator._num_ctx
        # The big LLM translates 1 + 3 but leaves cue 2 in Japanese.
        return [
            _seg(0.0, 1.4, "He went to the store."),
            _seg(1.4, 3.2, "彼女は本を買った。"),       # left untranslated
            _seg(3.2, 5.0, "They went home together."),
        ]

    async def fake_nmt(sub, src, tgt, glossary=None, **kw):
        cap["backstop_n"] = len(sub)
        return [_seg(s.start, s.end, "She bought a book.") for s in sub]

    monkeypatch.setattr(T, "translate_via_llm", fake_llm)
    monkeypatch.setattr(T, "_translate_via_nmt", fake_nmt)

    out = asyncio.run(T._translate_quality_mode(_source(), "ja", "en", None))

    # Big model + raised context window were used.
    assert cap["client_model"] == "qwen2.5:7b-instruct"
    assert cap["client_ctx"] == 8192
    # NLLB backstop ran on exactly the 1 untranslated cue.
    assert cap["backstop_n"] == 1
    # Every cue is now English, timing preserved.
    assert [o.text for o in out] == [
        "He went to the store.", "She bought a book.", "They went home together."]
    assert out[1].start == 1.4 and out[1].end == 3.2


def test_quality_mode_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr(T.settings, "OLLAMA_HOST", "", raising=False)
    monkeypatch.setattr(T.settings, "TRANSLATION_QUALITY_MODEL", "", raising=False)
    out = asyncio.run(T._translate_quality_mode(_source(), "ja", "en", None))
    assert out is None                                   # caller uses speed chain


def test_quality_mode_returns_none_when_llm_empty(monkeypatch, _quality_settings):
    async def fake_llm(*a, **k):
        return None

    monkeypatch.setattr(T, "translate_via_llm", fake_llm)
    out = asyncio.run(T._translate_quality_mode(_source(), "ja", "en", None))
    assert out is None                                   # fall back to NMT→MTPE


def test_quality_mode_no_backstop_when_complete(monkeypatch, _quality_settings):
    called = {"nmt": 0}

    async def fake_llm(segments, src, tgt, orchestrator, glossary=None, **kw):
        return [_seg(s.start, s.end, "fully english line") for s in segments]

    async def fake_nmt(*a, **k):
        called["nmt"] += 1
        return None

    monkeypatch.setattr(T, "translate_via_llm", fake_llm)
    monkeypatch.setattr(T, "_translate_via_nmt", fake_nmt)
    out = asyncio.run(T._translate_quality_mode(_source(), "ja", "en", None))
    assert len(out) == 3
    assert called["nmt"] == 0                             # nothing left to backstop
