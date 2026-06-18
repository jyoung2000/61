"""Tests for abandoning a billing-dead provider (OpenRouter "Key limit
exceeded") for the rest of a job, instead of re-hammering the dead key every
stage — the recurring 403 retry waste in the logs.
"""

import sys
import types
from types import SimpleNamespace

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

import backend.services.ai_orchestrator as O  # noqa: E402
from backend.services.ai_orchestrator import (  # noqa: E402
    AIOrchestrator, _CircuitBreaker, _is_key_limit_error,
)


# ── error classification ──────────────────────────────────────────────────

def test_detects_key_limit_403():
    assert _is_key_limit_error(Exception(
        "Error code: 403 - {'message': 'Key limit exceeded (total limit)'}"))
    assert _is_key_limit_error(Exception(
        "All OpenRouter models failed — google/gemini-3.1-flash-lite: 403 Key limit exceeded"))
    assert _is_key_limit_error(Exception("403 quota exceeded for this key"))


def test_does_not_flag_retryable_errors():
    assert not _is_key_limit_error(Exception("timed out after 60s"))
    assert not _is_key_limit_error(Exception("429 rate limit, too many requests"))
    assert not _is_key_limit_error(Exception("connection refused"))


# ── marking ────────────────────────────────────────────────────────────────

def test_mark_adds_only_on_billing_error():
    fake = SimpleNamespace(_billing_dead=set())
    AIOrchestrator._mark_if_key_limited(fake, "openrouter", Exception("403 Key limit exceeded"))
    assert fake._billing_dead == {"openrouter"}
    # A transient error must NOT disable the provider.
    AIOrchestrator._mark_if_key_limited(fake, "ollama", Exception("stalled"))
    assert "ollama" not in fake._billing_dead


# ── active chain skips a billing-dead provider ────────────────────────────

def _fake_orch(billing_dead):
    return SimpleNamespace(
        _unreachable=set(),
        _billing_dead=set(billing_dead),
        _providers={
            "openrouter": SimpleNamespace(provider_name="openrouter"),
            "ollama": SimpleNamespace(provider_name="ollama"),
        },
        _circuit_breaker=_CircuitBreaker(),
    )


def test_active_chain_skips_billing_dead(monkeypatch):
    monkeypatch.setattr(O, "settings",
                        SimpleNamespace(editorial_provider_chain=["openrouter", "ollama"]))
    chain = AIOrchestrator._get_active_chain(_fake_orch({"openrouter"}))
    assert [p.provider_name for p in chain] == ["ollama"]   # dead key skipped


def test_active_chain_full_when_none_dead(monkeypatch):
    monkeypatch.setattr(O, "settings",
                        SimpleNamespace(editorial_provider_chain=["openrouter", "ollama"]))
    chain = AIOrchestrator._get_active_chain(_fake_orch(set()))
    assert [p.provider_name for p in chain] == ["openrouter", "ollama"]


# ── billing-dead survives the per-stage circuit-breaker reset ─────────────

def test_reset_circuit_breaker_keeps_billing_dead():
    fake = SimpleNamespace(_billing_dead={"openrouter"}, _circuit_breaker=_CircuitBreaker())
    fake._circuit_breaker.record_failure("openrouter")
    AIOrchestrator.reset_circuit_breaker(fake)
    # The breaker is reset, but the hard key-limit disable persists for the job.
    assert "openrouter" in fake._billing_dead
