"""The configured local TRANSLATION model must surface in the model dropdown
even before it is pulled (alongside the primary + editorial defaults)."""

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

import pytest  # noqa: E402

from backend.config import settings  # noqa: E402
import backend.routers.settings as S  # noqa: E402


class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


class _FakeClient:
    """Async-context httpx stub returning a fixed /api/tags payload."""

    def __init__(self, payload):
        self._payload = payload

    def __call__(self, *a, **k):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, *a, **k):
        return _FakeResp(self._payload)


def _run_available_models(monkeypatch, installed_names):
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "AI_FALLBACK_CHAIN", "ollama", raising=False)
    monkeypatch.setattr(settings, "EDITORIAL_AI_SOURCE", "auto", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_PRIMARY_MODEL", "moondream:1.8b", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    payload = {"models": [{"name": n} for n in installed_names]}
    monkeypatch.setattr(S.httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload))
    return asyncio.run(S.available_models())


def test_translation_default_listed_when_not_pulled(monkeypatch):
    # Ollama host only has old models pulled — the qwen3 translation default is not.
    out = _run_available_models(monkeypatch, ["qwen2.5:3b-instruct", "moondream:1.8b"])
    text_ids = {m["id"] for m in out["text"]}
    translation_ids = {m["id"] for m in out["translation"]}
    tid = "ollama/qwen3:4b-instruct-2507-q4_K_M"
    # Surfaces in BOTH the text and translation dropdowns despite not being pulled.
    assert tid in text_ids
    assert tid in translation_ids
    # Marked as pulling so the user knows it's downloading.
    entry = next(m for m in out["text"] if m["id"] == tid)
    assert "pulling" in entry["name"].lower()


def test_distinct_translation_model_listed_even_if_editorial_differs(monkeypatch):
    # Translation model differs from editorial — it must still appear.
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "AI_FALLBACK_CHAIN", "ollama", raising=False)
    monkeypatch.setattr(settings, "EDITORIAL_AI_SOURCE", "auto", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_PRIMARY_MODEL", "moondream:1.8b", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen2.5:3b-instruct", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    payload = {"models": [{"name": "qwen2.5:3b-instruct"}]}
    monkeypatch.setattr(S.httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload))
    out = asyncio.run(S.available_models())
    ids = {m["id"] for m in out["translation"]}
    assert "ollama/qwen3:4b-instruct-2507-q4_K_M" in ids


def test_pulled_translation_model_not_duplicated(monkeypatch):
    # When the model IS pulled, it appears once (from /api/tags), not duplicated
    # by the defaults fallback.
    out = _run_available_models(
        monkeypatch, ["qwen3:4b-instruct-2507-q4_K_M", "moondream:1.8b"])
    tid = "ollama/qwen3:4b-instruct-2507-q4_K_M"
    n = sum(1 for m in out["text"] if m["id"] == tid)
    assert n == 1
    # And it's the real (pulled) entry, not the "pulling..." placeholder.
    entry = next(m for m in out["text"] if m["id"] == tid)
    assert "pulling" not in entry["name"].lower()
