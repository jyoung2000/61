"""The manual 'Pull local models' button — POST /providers/ollama/pull and its
status endpoint. The pull itself runs in a background thread; here we stub that
out and check the request-shaping + state reporting."""

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


def _idle_state():
    return {"active": False, "models": [], "done": [], "failed": [], "current": None}


def test_pull_defaults_dedupes_and_starts(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_PRIMARY_MODEL", "moondream:1.8b", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    monkeypatch.setattr(S, "_ollama_pull_state", _idle_state(), raising=False)
    captured = {}
    monkeypatch.setattr(S, "_pull_ollama_models_background",
                        lambda models=None: captured.update(models=models))

    out = asyncio.run(S.pull_ollama_models(S.PullOllamaRequest()))
    assert out["status"] == "started"
    # editorial == translation → listed once; primary also present.
    assert out["models"].count("qwen3:4b-instruct-2507-q4_K_M") == 1
    assert "moondream:1.8b" in out["models"]
    assert captured["models"] == out["models"]


def test_pull_specific_model(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    monkeypatch.setattr(S, "_ollama_pull_state", _idle_state(), raising=False)
    monkeypatch.setattr(S, "_pull_ollama_models_background", lambda models=None: None)
    out = asyncio.run(S.pull_ollama_models(
        S.PullOllamaRequest(model="qwen3:4b-instruct-2507-q4_K_M")))
    assert out["status"] == "started"
    assert out["models"] == ["qwen3:4b-instruct-2507-q4_K_M"]


def test_pull_rejects_when_already_running(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    busy = {"active": True, "models": ["x"], "done": [], "failed": [], "current": "x"}
    monkeypatch.setattr(S, "_ollama_pull_state", busy, raising=False)
    called = {"n": 0}
    monkeypatch.setattr(S, "_pull_ollama_models_background",
                        lambda models=None: called.__setitem__("n", called["n"] + 1))
    out = asyncio.run(S.pull_ollama_models(S.PullOllamaRequest()))
    assert out["status"] == "already_running"
    assert called["n"] == 0  # did not start a second run


def test_pull_no_host_errors(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOST", "", raising=False)
    monkeypatch.setattr(S, "_ollama_pull_state", _idle_state(), raising=False)
    out = asyncio.run(S.pull_ollama_models(S.PullOllamaRequest()))
    assert out["status"] == "error"


def test_pull_status_reports_progress(monkeypatch):
    st = {"active": True, "models": ["a", "b", "c"], "done": ["a"], "failed": ["b"], "current": "c"}
    monkeypatch.setattr(S, "_ollama_pull_state", st, raising=False)
    out = asyncio.run(S.ollama_pull_status())
    assert out["total"] == 3
    assert out["finished"] == 2  # done + failed
    assert out["current"] == "c"
    assert out["active"] is True
