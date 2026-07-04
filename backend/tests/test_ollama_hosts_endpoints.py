"""Settings endpoints for the multi-host Ollama registry: GET list with
status, PUT reorder/save (token keep/clear semantics), and the server-side
Test probe used by the Add-host dialog."""

import asyncio
import json

import pytest

from backend.config import settings
import backend.routers.settings as S
from backend.services import ollama_registry as R


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    R.reset_state()
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    monkeypatch.setattr(S, "_persist_user_settings", lambda: True)
    yield
    R.reset_state()


def _fake_probe(online=True, models=()):
    async def probe(host, force=False, timeout=None):
        import time
        return R.HostStatus(host_id=host.id, online=online,
                            models=list(models), latency_ms=3.2,
                            checked_at=time.time())
    return probe


def test_get_hosts_reports_migrated_default(monkeypatch):
    monkeypatch.setattr(R, "probe", _fake_probe(models=["moondream:1.8b"]))
    out = asyncio.run(S.get_ollama_hosts())
    assert out["migrated"] is True
    assert len(out["hosts"]) == 1
    assert out["hosts"][0]["name"] == "Default"
    assert out["hosts"][0]["role"] == "primary"
    assert out["hosts"][0]["online"] is True


def test_put_hosts_orders_and_syncs_primary(monkeypatch):
    monkeypatch.setattr(R, "probe", _fake_probe())
    req = S.SaveOllamaHostsRequest(hosts=[
        S.OllamaHostEntry(id="a", name="Desktop 4070",
                          url="http://192.168.1.50:11500/ollama", token="tok"),
        S.OllamaHostEntry(id="b", name="Sidecar", url="http://ollama:11434"),
    ])
    out = asyncio.run(S.put_ollama_hosts(req))
    assert out["status"] == "saved"
    assert out["primary"] == "Desktop 4070"
    assert settings.OLLAMA_HOST == "http://192.168.1.50:11500/ollama"
    stored = json.loads(settings.OLLAMA_HOSTS)
    assert [e["id"] for e in stored] == ["a", "b"]
    assert stored[0]["token"] == "tok"
    # Tokens are never echoed back in the status payload.
    assert all("token" not in h for h in out["hosts"])
    assert out["hosts"][0]["has_token"] is True


def test_put_hosts_null_token_keeps_stored_secret(monkeypatch):
    monkeypatch.setattr(R, "probe", _fake_probe())
    asyncio.run(S.put_ollama_hosts(S.SaveOllamaHostsRequest(hosts=[
        S.OllamaHostEntry(id="a", name="A", url="http://a:1", token="secret"),
    ])))
    # Reorder-style save: token omitted (None) → stored secret preserved.
    asyncio.run(S.put_ollama_hosts(S.SaveOllamaHostsRequest(hosts=[
        S.OllamaHostEntry(id="a", name="A renamed", url="http://a:1", token=None),
    ])))
    stored = json.loads(settings.OLLAMA_HOSTS)
    assert stored[0]["token"] == "secret"
    # Explicit empty string clears it.
    asyncio.run(S.put_ollama_hosts(S.SaveOllamaHostsRequest(hosts=[
        S.OllamaHostEntry(id="a", name="A", url="http://a:1", token=""),
    ])))
    stored = json.loads(settings.OLLAMA_HOSTS)
    assert stored[0]["token"] == ""


def test_put_hosts_empty_url_rows_dropped(monkeypatch):
    monkeypatch.setattr(R, "probe", _fake_probe())
    out = asyncio.run(S.put_ollama_hosts(S.SaveOllamaHostsRequest(hosts=[
        S.OllamaHostEntry(name="blank", url="   "),
        S.OllamaHostEntry(name="real", url="http://x:1"),
    ])))
    assert len(out["hosts"]) == 1
    assert out["hosts"][0]["name"] == "real"


def test_test_endpoint_probes_server_side(monkeypatch):
    monkeypatch.setattr(R, "probe", _fake_probe(models=["llava:7b"]))
    out = asyncio.run(S.test_ollama_host(
        S.TestOllamaHostRequest(url="http://192.168.1.50:11500/ollama", token="t")))
    assert out["online"] is True
    assert out["models"] == ["llava:7b"]


def test_test_endpoint_requires_url():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        asyncio.run(S.test_ollama_host(S.TestOllamaHostRequest(url="  ")))
