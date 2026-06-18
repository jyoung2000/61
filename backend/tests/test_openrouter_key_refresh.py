"""Test: the OpenRouter provider always uses the LATEST api key — no stale key
cached in the SDK client.

The SDK client caches the api key at construction. If a user saves a fresh key,
the provider must rebuild its client so the new key takes effect (no restart),
and never keep serving requests with the old key.
"""

from types import SimpleNamespace

import backend.services.providers.openrouter_provider as ORP
from backend.services.providers.openrouter_provider import OpenRouterProvider


def test_build_client_binds_the_given_key():
    c = OpenRouterProvider._build_client(SimpleNamespace(), "sk-or-FRESH")
    assert c.api_key == "sk-or-FRESH"
    assert str(c.base_url).startswith("https://openrouter.ai")


def test_client_not_rebuilt_when_key_unchanged(monkeypatch):
    monkeypatch.setattr(ORP.settings, "OPENROUTER_API_KEY", "sk-or-A", raising=False)
    built = []
    fake = SimpleNamespace(
        _client_key="sk-or-A", _client="CLIENT_A",
        _build_client=lambda k: (built.append(k), f"CLIENT::{k}")[1])
    out = OpenRouterProvider.client.fget(fake)
    assert out == "CLIENT_A"          # same client object
    assert built == []                # not rebuilt


def test_client_rebuilds_when_key_changes(monkeypatch):
    monkeypatch.setattr(ORP.settings, "OPENROUTER_API_KEY", "sk-or-OLD", raising=False)
    built = []
    fake = SimpleNamespace(
        _client_key="sk-or-OLD", _client="CLIENT_OLD",
        _build_client=lambda k: (built.append(k), f"CLIENT::{k}")[1])

    # User saves a fresh key.
    monkeypatch.setattr(ORP.settings, "OPENROUTER_API_KEY", "sk-or-FRESH")
    out = OpenRouterProvider.client.fget(fake)

    assert fake._client_key == "sk-or-FRESH"   # tracked key updated
    assert built == ["sk-or-FRESH"]            # rebuilt with the new key
    assert out == "CLIENT::sk-or-FRESH"        # serving the fresh-key client


def test_end_to_end_rebuild_uses_fresh_key_in_sdk(monkeypatch):
    # Real _build_client + property: the SDK client's api_key follows settings.
    monkeypatch.setattr(ORP.settings, "OPENROUTER_API_KEY", "sk-or-OLD", raising=False)
    fake = SimpleNamespace(_client_key="sk-or-OLD")
    fake._build_client = lambda k: OpenRouterProvider._build_client(fake, k)
    fake._client = fake._build_client("sk-or-OLD")
    assert fake._client.api_key == "sk-or-OLD"

    monkeypatch.setattr(ORP.settings, "OPENROUTER_API_KEY", "sk-or-FRESH")
    client = OpenRouterProvider.client.fget(fake)
    assert client.api_key == "sk-or-FRESH"     # no stale key cached
