"""Toggling Offline Mode OFF must hand primary/editorial back to the cloud.

Regression: selecting Ollama models (or the "Ollama (Local)" toggle) pins
``ollama`` at the FRONT of AI_FALLBACK_CHAIN. provider_status then picks Ollama
as the active provider, so turning Offline Mode off left the UI stuck on the
local engine and never re-showed the cloud / Replicate options. Turning it off
now demotes Ollama to the end of the chain so cloud providers lead again.
"""
import asyncio

import pytest

import backend.routers.settings as S
from backend.config import settings


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _no_disk(monkeypatch):
    # The handler persists to .env / user settings — stub the file writers so the
    # test only exercises the in-memory chain logic.
    monkeypatch.setattr(S, "_persist_user_settings", lambda *a, **k: None)
    monkeypatch.setattr(S, "_find_env_file", lambda *a, **k: None)
    monkeypatch.setattr(S, "_invalidate_status_cache", lambda *a, **k: None)
    yield


def _save(**kw):
    return _run(S.save_self_hosted_settings(S.SaveSelfHostedRequest(**kw)))


def test_turning_offline_off_demotes_pinned_ollama_to_end():
    settings.SELF_HOSTED_MODE = True
    settings.AI_FALLBACK_CHAIN = "ollama,openrouter,gemini,groq"
    _save(self_hosted_mode=False, clip_engine_source="auto", editorial_ai_source="auto")
    # Cloud providers now lead; Ollama survives only as a last-resort fallback.
    assert settings.AI_FALLBACK_CHAIN == "openrouter,gemini,groq,ollama"
    # And editorial/clip genuinely resolve to cloud again.
    assert settings.resolve_ai_source("editorial") == "cloud"
    assert settings.active_provider_chain[0] == "openrouter"


def test_no_transition_when_already_off_leaves_chain_untouched():
    # User deliberately pinned Ollama via the Ollama-Local toggle (offline already
    # off) — saving the offline form for another reason must not fight that.
    settings.SELF_HOSTED_MODE = False
    settings.AI_FALLBACK_CHAIN = "ollama,openrouter,gemini"
    _save(self_hosted_mode=False, clip_engine_source="cloud")
    assert settings.AI_FALLBACK_CHAIN == "ollama,openrouter,gemini"


def test_offline_off_when_ollama_not_first_is_noop():
    settings.SELF_HOSTED_MODE = True
    settings.AI_FALLBACK_CHAIN = "openrouter,ollama,gemini"
    _save(self_hosted_mode=False)
    assert settings.AI_FALLBACK_CHAIN == "openrouter,ollama,gemini"


def test_offline_off_with_only_ollama_keeps_it():
    # No cloud provider to promote — keep Ollama so the chain isn't emptied.
    settings.SELF_HOSTED_MODE = True
    settings.AI_FALLBACK_CHAIN = "ollama"
    _save(self_hosted_mode=False)
    assert settings.AI_FALLBACK_CHAIN == "ollama"


def test_explicit_cloud_override_demotes_pinned_ollama_live():
    # Advanced panel: "Editorial AI → Cloud" must lead with cloud even when
    # Ollama is pinned first in the chain — without persisting a chain change.
    settings.SELF_HOSTED_MODE = False
    settings.AI_FALLBACK_CHAIN = "ollama,openrouter,gemini"
    settings.EDITORIAL_AI_SOURCE = "cloud"
    settings.CLIP_ENGINE_SOURCE = "auto"
    assert settings.active_provider_chain == ["openrouter", "gemini", "ollama"]
    # The persisted chain itself is untouched (override is live-only).
    assert settings.AI_FALLBACK_CHAIN == "ollama,openrouter,gemini"


def test_auto_chain_keeps_ollama_first_for_ollama_local_toggle():
    # Plain "auto" (the Ollama-Local toggle's hybrid mode) keeps Ollama primary
    # with cloud as fallback — explicit-cloud demotion must NOT fire here.
    settings.SELF_HOSTED_MODE = False
    settings.AI_FALLBACK_CHAIN = "ollama,openrouter,gemini"
    settings.EDITORIAL_AI_SOURCE = "auto"
    settings.CLIP_ENGINE_SOURCE = "auto"
    assert settings.active_provider_chain == ["ollama", "openrouter", "gemini"]


def test_turning_offline_on_does_not_touch_chain():
    settings.SELF_HOSTED_MODE = False
    settings.AI_FALLBACK_CHAIN = "openrouter,gemini,groq"
    _save(self_hosted_mode=True, clip_engine_source="auto", editorial_ai_source="auto")
    assert settings.AI_FALLBACK_CHAIN == "openrouter,gemini,groq"
    # active_provider_chain forces Ollama while offline is on.
    assert settings.active_provider_chain == ["ollama"]
