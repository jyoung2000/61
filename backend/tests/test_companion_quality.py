"""Remote Companion performance/quality control: the GET/POST proxy that
mirrors the desktop app's "Performance" knob (Ollama speed profile + Whisper
transcription quality) into ClipAI's Settings, with the read-only /v1/health
fallback for Companions predating the /v1/config/quality route."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.settings as S


def _client():
    app = FastAPI()
    app.include_router(S.router)
    return TestClient(app)


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
    def json(self):
        return self._body


class _Client:
    """Fake httpx.AsyncClient recording calls; scripted per (method, path-tail)."""
    routes: dict = {}
    calls: list = []
    def __init__(self, *a, **k): ...
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url, headers=None):
        _Client.calls.append(("GET", url, None))
        return _Client.routes.get(("GET", url.rsplit("/", 1)[-1]), _Resp(200, {}))
    async def post(self, url, headers=None, json=None):
        _Client.calls.append(("POST", url, json))
        return _Client.routes.get(("POST", url.rsplit("/", 1)[-1]), _Resp(200, {}))


@pytest.fixture()
def _companion(monkeypatch):
    from backend.services import ollama_registry as oreg

    class _Host:
        name = "Gaming PC"
        url = "http://192.168.8.12:11500/ollama"
    monkeypatch.setattr(oreg, "companion_host", lambda: _Host())
    monkeypatch.setattr(oreg, "companion_base", lambda h: "http://192.168.8.12:11500")
    monkeypatch.setattr(oreg, "auth_headers", lambda h: {"X-Companion-Token": "t"})
    monkeypatch.setattr(oreg, "join_url", lambda base, path: base + path)
    _Client.routes = {}
    _Client.calls = []
    monkeypatch.setattr(S.httpx, "AsyncClient", _Client)
    return _Client


def test_get_quality_passthrough(_companion):
    _companion.routes[("GET", "quality")] = _Resp(200, {
        "speed_profile": "balanced", "whisper_quality": "balanced",
        "num_parallel": 2, "max_loaded_models": 2, "effective_budget_gb": 9.5,
        "whisper_effective": {"model": "large-v3-turbo", "beam_size": 5, "beam_search": True},
    })
    data = _client().get("/api/providers/companion/quality").json()
    assert data["ok"] is True and data["writable"] is True
    assert data["speed_profile"] == "balanced"
    assert data["whisper_effective"]["beam_size"] == 5
    assert data["companion"] == "Gaming PC"


def test_get_quality_falls_back_to_health_read_only(_companion):
    _companion.routes[("GET", "quality")] = _Resp(404, {})
    _companion.routes[("GET", "health")] = _Resp(200, {
        "speed_profile": "turbo", "whisper_quality": "max",
        "num_parallel": 3, "max_loaded_models": 2, "vram_budget_gb": 11.0,
        "whisper_model_effective": "large-v3", "whisper_beam_size": 5,
    })
    data = _client().get("/api/providers/companion/quality").json()
    assert data["ok"] is True and data["writable"] is False
    assert data["speed_profile"] == "turbo" and data["whisper_quality"] == "max"
    assert data["whisper_effective"] == {"model": "large-v3", "beam_size": 5, "beam_search": True}
    assert "update" in data["note"].lower()


def test_get_quality_no_companion(monkeypatch):
    from backend.services import ollama_registry as oreg
    monkeypatch.setattr(oreg, "companion_host", lambda: None)
    assert _client().get("/api/providers/companion/quality").json()["ok"] is False


def test_set_quality_forwards_both_fields(_companion):
    _companion.routes[("POST", "quality")] = _Resp(200, {
        "speed_profile": "turbo", "whisper_quality": "max", "ollama_restarted": True,
        "whisper_effective": {"model": "large-v3", "beam_size": 5, "beam_search": True},
    })
    r = _client().post("/api/providers/companion/quality",
                       json={"speed_profile": "Turbo", "whisper_quality": "MAX"})
    data = r.json()
    assert data["ok"] is True and data["ollama_restarted"] is True
    post = [c for c in _companion.calls if c[0] == "POST"][0]
    assert post[1].endswith("/v1/config/quality")
    # Values are normalized to lowercase before forwarding.
    assert post[2] == {"speed_profile": "turbo", "whisper_quality": "max"}


def test_set_quality_single_field_only_sends_that_field(_companion):
    _companion.routes[("POST", "quality")] = _Resp(200, {"whisper_quality": "fast"})
    r = _client().post("/api/providers/companion/quality", json={"whisper_quality": "fast"})
    assert r.json()["ok"] is True
    post = [c for c in _companion.calls if c[0] == "POST"][0]
    assert post[2] == {"whisper_quality": "fast"}


def test_set_quality_rejects_invalid_values_without_calling_companion(_companion):
    r = _client().post("/api/providers/companion/quality", json={"speed_profile": "ludicrous"})
    data = r.json()
    assert data["ok"] is False and "invalid speed profile" in data["error"].lower()
    r = _client().post("/api/providers/companion/quality", json={"whisper_quality": "netflix"})
    data = r.json()
    assert data["ok"] is False and "invalid whisper quality" in data["error"].lower()
    # The bad values never reached the Companion.
    assert [c for c in _companion.calls if c[0] == "POST"] == []


def test_set_quality_empty_payload_is_an_error(_companion):
    data = _client().post("/api/providers/companion/quality", json={}).json()
    assert data["ok"] is False and "nothing" in data["error"].lower()
    assert _companion.calls == []


def test_set_quality_old_companion_says_update(_companion):
    _companion.routes[("POST", "quality")] = _Resp(404, {})
    data = _client().post("/api/providers/companion/quality",
                          json={"speed_profile": "eco"}).json()
    assert data["ok"] is False and "update" in data["error"].lower()


def test_set_quality_no_companion(monkeypatch):
    from backend.services import ollama_registry as oreg
    monkeypatch.setattr(oreg, "companion_host", lambda: None)
    data = _client().post("/api/providers/companion/quality",
                          json={"speed_profile": "eco"}).json()
    assert data["ok"] is False


# ── Companion sync must never overwrite user-saved local knobs ──────────────

def test_beam_size_sync_respects_user_set_flag(monkeypatch):
    """The model pin had USER_SET protection; beam did not — an explicit
    Companion quality choice silently overwrote (and persisted) the user's
    Beam Size slider. Pin the guard."""
    import re
    import inspect
    src = inspect.getsource(S)
    # The sync's beam write must be gated on WHISPER_BEAM_USER_SET…
    block = src[src.index("def _sync_companion_whisper"):]
    block = block[:block.index("\n@router", 1)] if "\n@router" in block else block
    assert "WHISPER_BEAM_USER_SET" in block, "beam sync lacks the user-set guard"
    # …and the transcription-settings save must SET the flag.
    save = src[src.index("Save transcription speed/quality settings"):]
    save = save[:save.index("def ", 100)]
    assert "WHISPER_BEAM_USER_SET = True" in save
    assert "WHISPER_BEAM_USER_SET" in S._PERSISTABLE_KEYS
