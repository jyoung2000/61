"""GPU Companion pairing endpoint: registry insert-as-primary, idempotent
re-pair, remote-Whisper auto-fill, and API-key enforcement."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings
import backend.auth as auth
import backend.routers.settings as S
from backend.services import ollama_registry as R


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    R.reset_state()
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_URL", "", raising=False)
    monkeypatch.setattr(settings, "WHISPER_REMOTE_API_KEY", "", raising=False)
    monkeypatch.setattr(S, "_persist_user_settings", lambda: True)
    monkeypatch.setattr(auth, "get_or_create_api_key", lambda: "clipai-key-123")

    async def fake_probe(host, force=False, timeout=None):
        import time
        return R.HostStatus(host_id=host.id, online=True, checked_at=time.time())

    monkeypatch.setattr(R, "probe", fake_probe)
    yield
    R.reset_state()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(S.router)
    return TestClient(app)


def _pair(client, token="tok-abc", **overrides):
    body = {
        "name": "Desktop 4070",
        "url": "http://192.168.1.50:11500",
        "token": token,
        "gpu_name": "NVIDIA GeForce RTX 4070",
        "vram_total_mb": 12282,
    }
    body.update(overrides)
    return client.post(
        "/api/settings/companion-register",
        json=body,
        headers={"Authorization": "Bearer clipai-key-123"},
    )


def test_pairing_requires_api_key(client):
    resp = client.post("/api/settings/companion-register",
                       json={"url": "http://192.168.1.50:11500"})
    assert resp.status_code == 401
    resp = client.post("/api/settings/companion-register",
                       json={"url": "http://192.168.1.50:11500"},
                       headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 403


def test_pairing_registers_primary_and_whisper(client):
    resp = _pair(client)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "paired"
    assert data["ollama_host"]["role"] == "primary"
    assert data["ollama_host"]["url"] == "http://192.168.1.50:11500/ollama"

    hosts = R.get_hosts()
    # Companion first, migrated legacy default preserved as fallback.
    assert hosts[0].url == "http://192.168.1.50:11500/ollama"
    assert hosts[0].token == "tok-abc"
    assert hosts[1].url == "http://ollama:11434"
    # Legacy field synced to the new primary.
    assert settings.OLLAMA_HOST == "http://192.168.1.50:11500/ollama"
    # Remote Whisper auto-filled with the proxy base + token.
    assert settings.WHISPER_REMOTE_URL == "http://192.168.1.50:11500"
    assert settings.WHISPER_REMOTE_API_KEY == "tok-abc"
    # The Companion token is never echoed in the response.
    assert "tok-abc" not in json.dumps(data)


def test_pairing_is_idempotent(client):
    assert _pair(client).status_code == 200
    assert _pair(client, token="tok-rotated", name="Desktop 4070 (new)").status_code == 200
    hosts = R.get_hosts()
    companion_entries = [h for h in hosts if "192.168.1.50" in h.url]
    assert len(companion_entries) == 1
    assert hosts[0].token == "tok-rotated"
    assert hosts[0].name == "Desktop 4070 (new)"


def test_pairing_can_skip_whisper(client):
    resp = _pair(client, register_whisper=False)
    assert resp.status_code == 200
    assert settings.WHISPER_REMOTE_URL == ""


def test_pairing_requires_url(client):
    resp = client.post(
        "/api/settings/companion-register",
        json={"url": "  "},
        headers={"Authorization": "Bearer clipai-key-123"},
    )
    assert resp.status_code == 400
