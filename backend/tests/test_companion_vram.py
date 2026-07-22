"""Remote Companion VRAM control + the model-fit pre-check.

Covers the pure estimator, the pull endpoint's "won't fit / budget warning"
gate (so the picker never hangs on 'pulling' for an impossible model), and the
GET/POST proxy that flips auto-allocate + budget on the Companion remotely."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.routers.settings as S


def _client():
    app = FastAPI()
    app.include_router(S.router)
    return TestClient(app)


# ── estimator ────────────────────────────────────────────────────────────────

def test_estimate_from_disk_size_adds_overhead():
    assert S._estimate_model_vram_gb("whatever", size_gb=9.0) == 9.8


def test_estimate_from_param_count_by_quant():
    # 14B at a 4-bit quant ≈ 14*0.62 + 0.8 ≈ 9.5 GB.
    v = S._estimate_model_vram_gb("qwen2.5:14b")
    assert 9.0 <= v <= 10.0
    # 7B smaller.
    assert S._estimate_model_vram_gb("llama3.1:7b") < v
    # fp16 is far heavier than the 4-bit default.
    assert S._estimate_model_vram_gb("x:14b", quant="fp16") > S._estimate_model_vram_gb("x:14b")


def test_estimate_unknown_returns_zero():
    assert S._estimate_model_vram_gb("mystery-model") == 0.0


# ── pull fit-gate ────────────────────────────────────────────────────────────

@pytest.fixture()
def _pull_env(monkeypatch):
    monkeypatch.setattr(S, "_ollama_pull_state", {"active": False}, raising=False)
    monkeypatch.setattr(S.settings, "OLLAMA_HOST", "http://192.168.8.12:11500/ollama", raising=False)
    started = []
    monkeypatch.setattr(S, "_pull_ollama_models_background", lambda models: started.append(list(models)))
    return started


def _snap(total, budget, gpu="RTX 4070"):
    async def _f():
        return {"host": object(), "total_gb": total, "budget_gb": budget, "gpu": gpu}
    return _f


def test_pull_refuses_model_bigger_than_the_card(_pull_env, monkeypatch):
    monkeypatch.setattr(S, "_companion_vram_snapshot", _snap(total=8.0, budget=7.0))
    r = _client().post("/api/providers/ollama/pull", json={"model": "qwen2.5:32b"})
    data = r.json()
    assert data["status"] == "wont_fit"
    assert data["model"] == "qwen2.5:32b"
    assert "can't run on this GPU" in data["message"]
    # Crucially: the hanging background pull was NEVER started.
    assert _pull_env == []


def test_pull_warns_when_over_budget_but_fits_card(_pull_env, monkeypatch):
    monkeypatch.setattr(S, "_companion_vram_snapshot", _snap(total=12.0, budget=6.0))
    r = _client().post("/api/providers/ollama/pull", json={"model": "qwen2.5:14b"})
    data = r.json()
    assert data["status"] == "started"
    assert data["warnings"] and "budget" in data["warnings"][0].lower()
    # It still pulls (download needs no VRAM).
    assert _pull_env == [["qwen2.5:14b"]]


def test_pull_clean_when_it_fits_the_budget(_pull_env, monkeypatch):
    monkeypatch.setattr(S, "_companion_vram_snapshot", _snap(total=12.0, budget=11.0))
    r = _client().post("/api/providers/ollama/pull", json={"model": "llama3.2:3b"})
    data = r.json()
    assert data["status"] == "started"
    assert data["warnings"] == []
    assert _pull_env == [["llama3.2:3b"]]


def test_pull_gate_skips_when_companion_unknown(_pull_env, monkeypatch):
    # No companion / unknown total → don't block; just pull.
    monkeypatch.setattr(S, "_companion_vram_snapshot", _snap(total=0.0, budget=0.0, gpu=""))
    r = _client().post("/api/providers/ollama/pull", json={"model": "qwen2.5:32b"})
    assert r.json()["status"] == "started"
    assert _pull_env == [["qwen2.5:32b"]]


# ── remote VRAM read/write proxy ─────────────────────────────────────────────

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


def test_set_companion_vram_forwards_payload(_companion):
    _companion.routes[("POST", "vram")] = _Resp(200, {
        "vram_auto": True, "effective_budget_gb": 11.0, "vram_total_mb": 12282})
    r = _client().post("/api/providers/companion/vram", json={"vram_auto": True})
    data = r.json()
    assert data["ok"] is True
    assert data["effective_budget_gb"] == 11.0
    # The write reached the Companion's /v1/config/vram with the payload.
    post = [c for c in _companion.calls if c[0] == "POST"][0]
    assert post[1].endswith("/v1/config/vram")
    assert post[2] == {"vram_auto": True}


def test_set_companion_vram_old_companion_says_update(_companion):
    _companion.routes[("POST", "vram")] = _Resp(404, {})
    r = _client().post("/api/providers/companion/vram", json={"vram_budget_gb": 11})
    data = r.json()
    assert data["ok"] is False and "update" in data["error"].lower()


def test_set_companion_vram_no_companion(monkeypatch):
    from backend.services import ollama_registry as oreg
    monkeypatch.setattr(oreg, "companion_host", lambda: None)
    r = _client().post("/api/providers/companion/vram", json={"vram_auto": True})
    assert r.json()["ok"] is False


def test_get_companion_vram_passthrough(_companion):
    _companion.routes[("GET", "vram")] = _Resp(200, {
        "vram_auto": False, "vram_budget_manual_gb": 9.5, "vram_total_mb": 12282,
        "effective_budget_gb": 9.5})
    data = _client().get("/api/providers/companion/vram").json()
    assert data["ok"] is True and data["writable"] is True
    assert data["vram_budget_manual_gb"] == 9.5


def test_get_companion_vram_falls_back_to_health(_companion):
    _companion.routes[("GET", "vram")] = _Resp(404, {})
    _companion.routes[("GET", "health")] = _Resp(200, {
        "vram_auto": True, "vram_budget_manual_gb": 8.0, "vram_budget_gb": 9.0,
        "vram_total_mb": 12282})
    data = _client().get("/api/providers/companion/vram").json()
    assert data["ok"] is True and data["writable"] is False
    assert data["effective_budget_gb"] == 9.0
