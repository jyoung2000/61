"""Cold-load-aware text timeouts: a non-resident Ollama model gets the
configured extra allowance (a cold weights load is not a hang); a resident
model keeps the caller's tight ceiling so genuinely stuck calls still fail
over fast. This is the fix for 'Text completion timed out after 90s
(model=qwen2.5:3b)' silently pushing polish batches to the paid cloud
fallback on a 4 GB GTX 1650."""

import asyncio
import sys
import types


def _stub_provider_sdks():
    for name in ["google", "google.generativeai", "groq", "openai", "anthropic"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["google"].generativeai = sys.modules["google.generativeai"]
    g = sys.modules["google.generativeai"]
    g.configure = lambda *a, **k: None
    g.GenerativeModel = object
    sys.modules["groq"].AsyncGroq = sys.modules["groq"].Groq = object
    sys.modules["openai"].AsyncOpenAI = sys.modules["openai"].OpenAI = object
    sys.modules["anthropic"].AsyncAnthropic = sys.modules["anthropic"].Anthropic = object


_stub_provider_sdks()

import pytest  # noqa: E402

from backend.config import settings  # noqa: E402
import backend.services.ai_orchestrator as O  # noqa: E402


class _FakeOllama:
    """Minimal provider double: records the timeout it was given."""

    provider_name = "ollama"
    text_model_name = "qwen2.5:3b"
    total_tokens = 0

    def __init__(self, loaded: bool, call_duration: float = 0.0):
        self._loaded = loaded
        self._call_duration = call_duration
        self._editorial_model = "qwen2.5:3b"
        self.seen_timeout = None
        self.residency_checks = 0

    async def is_model_loaded(self, model_name):
        self.residency_checks += 1
        return self._loaded

    async def text_complete(self, prompt, max_tokens=4096, timeout=90, **kw):
        self.seen_timeout = timeout
        if self._call_duration:
            await asyncio.sleep(self._call_duration)
        return "polished"


def _orchestrator_with(provider, monkeypatch):
    monkeypatch.setattr(O, "_build_provider", lambda name: None)
    monkeypatch.setattr(O.AIOrchestrator, "__init__",
                        lambda self, **kw: None)
    orch = O.AIOrchestrator()
    orch._providers = {"ollama": provider}
    orch._circuit_breaker = O._CircuitBreaker()
    orch._unreachable = set()
    orch._billing_dead = set()
    orch._consecutive_ollama_failures = 0
    orch._current_model_override = None
    orch._ws_broadcast = None
    return orch


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_COLD_LOAD_TIMEOUT_EXTRA_S", 240.0,
                        raising=False)
    monkeypatch.setattr(settings, "AI_FALLBACK_CHAIN", "ollama", raising=False)
    yield


def test_cold_model_gets_extended_timeout(monkeypatch):
    provider = _FakeOllama(loaded=False)
    orch = _orchestrator_with(provider, monkeypatch)
    out = asyncio.run(orch.text_completion("hi", timeout=90))
    assert out == "polished"
    assert provider.residency_checks == 1
    assert provider.seen_timeout == 330  # 90 caller + 240 cold allowance


def test_warm_model_keeps_tight_timeout(monkeypatch):
    provider = _FakeOllama(loaded=True)
    orch = _orchestrator_with(provider, monkeypatch)
    asyncio.run(orch.text_completion("hi", timeout=90))
    assert provider.seen_timeout == 90


def test_extension_disabled_when_extra_is_zero(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_COLD_LOAD_TIMEOUT_EXTRA_S", 0.0,
                        raising=False)
    provider = _FakeOllama(loaded=False)
    orch = _orchestrator_with(provider, monkeypatch)
    asyncio.run(orch.text_completion("hi", timeout=90))
    assert provider.seen_timeout == 90
    assert provider.residency_checks == 0  # probe skipped entirely


def test_cold_call_survives_past_the_caller_ceiling(monkeypatch):
    """A cold call that takes LONGER than the caller's ceiling now succeeds
    instead of raising AllProvidersFailed (tiny numbers to stay fast)."""
    monkeypatch.setattr(settings, "OLLAMA_COLD_LOAD_TIMEOUT_EXTRA_S", 5.0,
                        raising=False)
    provider = _FakeOllama(loaded=False, call_duration=0.3)
    orch = _orchestrator_with(provider, monkeypatch)
    out = asyncio.run(orch.text_completion("hi", timeout=0.1))
    assert out == "polished"


def test_warm_model_still_fails_over_fast(monkeypatch):
    """Resident model that hangs past the tight ceiling → the provider
    chain still moves on (raises here, since ollama is the only link)."""
    provider = _FakeOllama(loaded=True, call_duration=0.5)
    orch = _orchestrator_with(provider, monkeypatch)
    with pytest.raises(Exception):
        asyncio.run(orch.text_completion("hi", timeout=0.1))
