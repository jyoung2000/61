"""Separate Subtitle-Translation model picker: saving an ollama/ translation
model sets OLLAMA_TRANSLATION_MODEL (distinct from the editorial model), and the
UI 'current' resolver reads it back as ollama/<model>."""

import asyncio
import sys
import types


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

from backend.config import settings  # noqa: E402
import backend.routers.settings as S  # noqa: E402


def test_save_sets_ollama_translation_model_only(monkeypatch):
    # No .env writes in the test.
    monkeypatch.setattr(S, "_find_env_file", lambda: None)
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen2.5:3b-instruct", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen2.5:3b", raising=False)

    asyncio.run(S.save_models(S.SaveModelsRequest(
        translation_model="ollama/qwen3:4b-instruct-2507-q4_K_M")))

    # Translation model updated; editorial model untouched (separate pickers).
    assert settings.OLLAMA_TRANSLATION_MODEL == "qwen3:4b-instruct-2507-q4_K_M"
    assert settings.OLLAMA_EDITORIAL_MODEL == "qwen2.5:3b-instruct"


def test_save_editorial_does_not_touch_translation(monkeypatch):
    monkeypatch.setattr(S, "_find_env_file", lambda: None)
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen2.5:3b-instruct", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)

    asyncio.run(S.save_models(S.SaveModelsRequest(
        text_model="ollama/llama3.2:3b")))

    assert settings.OLLAMA_EDITORIAL_MODEL == "llama3.2:3b"
    assert settings.OLLAMA_TRANSLATION_MODEL == "qwen3:4b-instruct-2507-q4_K_M"


def test_current_translation_model_reads_back_on_ollama(monkeypatch):
    monkeypatch.setattr(settings, "AI_FALLBACK_CHAIN", "ollama", raising=False)
    monkeypatch.setattr(settings, "EDITORIAL_AI_SOURCE", "auto", raising=False)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    assert S._current_translation_model() == "ollama/qwen3:4b-instruct-2507-q4_K_M"


def test_save_blank_translation_clears_to_editorial(monkeypatch):
    monkeypatch.setattr(S, "_find_env_file", lambda: None)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    # Empty string = "reuse editorial" → OPENROUTER side cleared; ollama path
    # only changes when an ollama/ id is given, so a bare "" sets the
    # OpenRouter translation model empty (the documented clear semantics).
    asyncio.run(S.save_models(S.SaveModelsRequest(translation_model="")))
    assert settings.OPENROUTER_TRANSLATION_MODEL == ""
