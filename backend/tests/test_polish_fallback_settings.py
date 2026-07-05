"""Subtitle-polish cloud-fallback settings endpoints: the none/auto/model
dropdown choice, its persistence semantics, and validation."""

import asyncio

import pytest

from backend.config import settings
import backend.routers.settings as S


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(settings, "SUBTITLE_POLISH_CLOUD_FALLBACK", True, raising=False)
    monkeypatch.setattr(settings, "SUBTITLE_POLISH_CLOUD_MODEL", "", raising=False)
    monkeypatch.setattr(settings, "SUBTITLE_POLISH_MODEL", "", raising=False)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "sk-or-test", raising=False)
    persisted = {"n": 0}
    monkeypatch.setattr(S, "_persist_user_settings",
                        lambda: persisted.__setitem__("n", persisted["n"] + 1))
    yield persisted


def test_get_reports_auto_default():
    out = asyncio.run(S.get_polish_fallback())
    assert out["choice"] == "auto"
    assert out["enabled"] is True
    assert out["auto_resolves_to"]  # efficient-tier default resolved
    assert out["openrouter_key_set"] is True
    assert any(o["id"] for o in out["options"])  # shortlist populated


def test_put_none_disables_and_persists(_clean):
    out = asyncio.run(S.put_polish_fallback(S.SavePolishFallbackRequest(choice="none")))
    assert out["choice"] == "none"
    assert settings.SUBTITLE_POLISH_CLOUD_FALLBACK is False
    assert _clean["n"] == 1
    # The polisher's availability gate honors it.
    from backend.services.transcript_polisher import _cloud_polish_available
    assert _cloud_polish_available() is False


def test_put_model_pins_and_reenables(_clean):
    asyncio.run(S.put_polish_fallback(S.SavePolishFallbackRequest(choice="none")))
    out = asyncio.run(S.put_polish_fallback(
        S.SavePolishFallbackRequest(choice="anthropic/claude-haiku-4.5")))
    assert out["choice"] == "anthropic/claude-haiku-4.5"
    assert settings.SUBTITLE_POLISH_CLOUD_FALLBACK is True
    assert settings.SUBTITLE_POLISH_CLOUD_MODEL == "anthropic/claude-haiku-4.5"
    # The polisher resolves to exactly the pinned model.
    from backend.services.transcript_polisher import _resolve_cloud_polish_model
    assert _resolve_cloud_polish_model() == "anthropic/claude-haiku-4.5"


def test_put_auto_clears_pin(_clean):
    asyncio.run(S.put_polish_fallback(
        S.SavePolishFallbackRequest(choice="anthropic/claude-haiku-4.5")))
    out = asyncio.run(S.put_polish_fallback(S.SavePolishFallbackRequest(choice="auto")))
    assert out["choice"] == "auto"
    assert settings.SUBTITLE_POLISH_CLOUD_MODEL == ""
    assert settings.SUBTITLE_POLISH_CLOUD_FALLBACK is True


def test_none_keeps_stored_model_for_reenable(_clean):
    asyncio.run(S.put_polish_fallback(
        S.SavePolishFallbackRequest(choice="google/gemini-2.5-flash")))
    asyncio.run(S.put_polish_fallback(S.SavePolishFallbackRequest(choice="none")))
    # Disabling keeps the pinned model stored (the UI can restore it),
    # but the choice reads back as none.
    assert settings.SUBTITLE_POLISH_CLOUD_MODEL == "google/gemini-2.5-flash"
    out = asyncio.run(S.get_polish_fallback())
    assert out["choice"] == "none"


def test_put_rejects_garbage():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        asyncio.run(S.put_polish_fallback(
            S.SavePolishFallbackRequest(choice="not-a-model-id")))
