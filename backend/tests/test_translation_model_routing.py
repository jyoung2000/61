"""qwen3-translate-only routing: translation uses OLLAMA_TRANSLATION_MODEL while
editorial/SEO keep the fast editorial model.

- _resolve_translation_model_override picks the dedicated translation model for
  the active provider (and returns None when unset or same as editorial).
- text_completion(model_override=...) temporarily swaps the Ollama editorial
  model for that one call and restores it afterward.
"""

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
from backend.services import pipeline as P  # noqa: E402
from backend.services.ai_orchestrator import AIOrchestrator  # noqa: E402


class _FakeOrch:
    def __init__(self, provider, model):
        self._p, self._m = provider, model

    def get_editorial_model_info(self):
        return {"provider": self._p, "model": self._m, "is_thinking": False}


def test_resolver_picks_ollama_translation_model(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
    orch = _FakeOrch("ollama", "qwen2.5:3b-instruct")
    assert P._resolve_translation_model_override(orch) == "qwen3:4b-instruct-2507-q4_K_M"


def test_resolver_none_when_same_as_editorial(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "qwen2.5:3b-instruct")
    orch = _FakeOrch("ollama", "qwen2.5:3b-instruct")
    assert P._resolve_translation_model_override(orch) is None


def test_resolver_none_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL", "")
    orch = _FakeOrch("ollama", "qwen2.5:3b-instruct")
    assert P._resolve_translation_model_override(orch) is None


def test_resolver_openrouter(monkeypatch):
    monkeypatch.setattr(settings, "OPENROUTER_TRANSLATION_MODEL", "some/xlate-model")
    orch = _FakeOrch("openrouter", "some/editorial-model")
    assert P._resolve_translation_model_override(orch) == "some/xlate-model"


# ── text_completion model_override applies to Ollama ──

class _FakeOllamaProvider:
    provider_name = "ollama"

    def __init__(self):
        self._editorial_model = "qwen2.5:3b-instruct"
        self.seen_models = []

    @property
    def text_model_name(self):
        return self._editorial_model

    async def text_complete(self, prompt, max_tokens=4096, timeout=60):
        # Record the model active at call time so we can assert the override.
        self.seen_models.append(self._editorial_model)
        return "ok"


def test_text_completion_override_swaps_and_restores_ollama(monkeypatch):
    orch = AIOrchestrator()
    prov = _FakeOllamaProvider()
    monkeypatch.setattr(orch, "_get_active_chain", lambda: [prov])
    # Avoid the vision-evict path touching a real client.
    monkeypatch.setattr(orch, "_consecutive_ollama_failures", 1, raising=False)

    out = asyncio.run(orch.text_completion(
        "hi", timeout=5, model_override="qwen3:4b-instruct-2507-q4_K_M"))
    assert out == "ok"
    # The call ran under the override model…
    assert prov.seen_models == ["qwen3:4b-instruct-2507-q4_K_M"]
    # …and the editorial model is restored afterward.
    assert prov._editorial_model == "qwen2.5:3b-instruct"


def test_text_completion_no_override_uses_editorial(monkeypatch):
    orch = AIOrchestrator()
    prov = _FakeOllamaProvider()
    monkeypatch.setattr(orch, "_get_active_chain", lambda: [prov])
    monkeypatch.setattr(orch, "_consecutive_ollama_failures", 1, raising=False)

    out = asyncio.run(orch.text_completion("hi", timeout=5))
    assert out == "ok"
    assert prov.seen_models == ["qwen2.5:3b-instruct"]
