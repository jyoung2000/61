"""Multi-host Ollama registry: ordering, migration, failover walk, cooldown,
503 handling, model-substitution ladder, and the contract that a provider
call lands on host #2 when host #1 is a dead socket."""

import asyncio
import json
import socket
import threading

import httpx
import pytest

from backend.config import settings
from backend.services import ollama_registry as R


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    R.reset_state()
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    yield
    R.reset_state()


def _hosts_json(*entries):
    return json.dumps(list(entries))


def test_routable_hosts_strict_drops_local_when_remote_exists(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "comp", "name": "4070", "url": "http://192.168.8.10:11500/ollama"},
        {"id": "loc", "name": "local", "url": "http://ollama:11434"},
    ), raising=False)
    # Off: both hosts routable.
    monkeypatch.setattr(settings, "GPU_STRICT_REMOTE", False, raising=False)
    assert [h.id for h in R.routable_hosts()] == ["comp", "loc"]
    # On: the local-GPU host is dropped.
    monkeypatch.setattr(settings, "GPU_STRICT_REMOTE", True, raising=False)
    assert [h.id for h in R.routable_hosts()] == ["comp"]


def test_routable_hosts_strict_noop_without_remote(monkeypatch):
    # Strict on but only a local host — must NOT strip it (would break the box).
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "loc", "name": "local", "url": "http://ollama:11434"},
    ), raising=False)
    monkeypatch.setattr(settings, "GPU_STRICT_REMOTE", True, raising=False)
    assert [h.id for h in R.routable_hosts()] == ["loc"]


# ── Parsing, ordering, migration ─────────────────────────────────────


def test_migration_synthesizes_default_from_legacy_host(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "http://ollama:11434", raising=False)
    hosts = R.get_hosts()
    assert len(hosts) == 1
    assert hosts[0].name == "Default"
    assert hosts[0].url == "http://ollama:11434"
    assert hosts[0].enabled is True


def test_no_hosts_when_both_unset(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "", raising=False)
    assert R.get_hosts() == []
    assert R.primary_host() is None


def test_array_order_is_priority(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "a", "name": "Desktop 4070", "url": "http://192.168.1.50:11500/ollama", "token": "tok-a"},
        {"id": "b", "name": "Unraid sidecar", "url": "http://ollama:11434"},
    ), raising=False)
    hosts = R.get_hosts()
    assert [h.id for h in hosts] == ["a", "b"]
    assert R.primary_host().id == "a"
    # URL with base path is normalized without a trailing slash.
    assert R.primary_url() == "http://192.168.1.50:11500/ollama"


def test_disabled_hosts_excluded_from_enabled_list(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "a", "name": "A", "url": "http://a:11434", "enabled": False},
        {"id": "b", "name": "B", "url": "http://b:11434"},
    ), raising=False)
    assert [h.id for h in R.enabled_hosts()] == ["b"]
    assert R.primary_host().id == "b"


def test_bad_json_is_ignored_and_legacy_kicks_in(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "{not json", raising=False)
    hosts = R.get_hosts()
    assert len(hosts) == 1 and hosts[0].id == "default"


def test_save_hosts_syncs_legacy_primary(monkeypatch):
    calls = {"persisted": 0}
    import backend.routers.settings as S
    monkeypatch.setattr(S, "_persist_user_settings",
                        lambda: calls.__setitem__("persisted", calls["persisted"] + 1))
    R.save_hosts([
        R.OllamaHost(id="x", name="X", url="http://x:1/ollama", token="t"),
        R.OllamaHost(id="y", name="Y", url="http://y:2"),
    ])
    assert settings.OLLAMA_HOST == "http://x:1/ollama"
    assert calls["persisted"] == 1
    saved = json.loads(settings.OLLAMA_HOSTS)
    assert [e["id"] for e in saved] == ["x", "y"]


# ── URL joining (base paths) + auth headers ──────────────────────────


def test_join_url_handles_base_path():
    assert R.join_url("http://h:11500/ollama", "/api/tags") == "http://h:11500/ollama/api/tags"
    assert R.join_url("http://h:11434/", "/api/tags") == "http://h:11434/api/tags"
    assert R.join_url("h:11434", "api/tags") == "http://h:11434/api/tags"


def test_headers_for_url_longest_prefix_wins(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "a", "name": "A", "url": "http://h:11500", "token": "root-tok"},
        {"id": "b", "name": "B", "url": "http://h:11500/ollama", "token": "proxy-tok"},
    ), raising=False)
    headers = R.headers_for_url("http://h:11500/ollama/api/tags")
    assert headers["Authorization"] == "Bearer proxy-tok"
    headers = R.headers_for_url("http://h:11500/api/tags")
    assert headers["Authorization"] == "Bearer root-tok"
    assert R.headers_for_url("http://elsewhere:1/api/tags").get("Authorization") is None


def test_clipai_job_headers_attached():
    from backend.services import request_context as ctx
    ctx.set_job("job-42", "Podcast Ep 12")
    ctx.set_stage("transcription")
    try:
        headers = R.auth_headers(R.OllamaHost(id="a", name="A", url="http://a", token="t"))
        assert headers["X-ClipAI-Job-Id"] == "job-42"
        assert headers["X-ClipAI-Stage"] == "transcription"
        assert headers["X-ClipAI-Job-Title"] == "Podcast Ep 12"
        assert headers["Authorization"] == "Bearer t"
    finally:
        ctx.clear()


# ── Probing, cooldown, pick_host ─────────────────────────────────────


def _mk_status(host, online=True, models=None):
    return R.HostStatus(host_id=host.id, online=online, models=models or [],
                        checked_at=__import__("time").time())


def test_pick_host_skips_offline_and_cooldown(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "a", "name": "A", "url": "http://a:11434"},
        {"id": "b", "name": "B", "url": "http://b:11434"},
        {"id": "c", "name": "C", "url": "http://c:11434"},
    ), raising=False)
    hosts = {h.id: h for h in R.get_hosts()}

    async def fake_probe(host, force=False, timeout=None):
        return _mk_status(host, online=(host.id != "a"))

    monkeypatch.setattr(R, "probe", fake_probe)
    R.mark_unhealthy(hosts["b"], "test cooldown")

    picked = asyncio.run(R.pick_host())
    assert picked.id == "c"  # a offline, b cooling down


def test_cooldown_expires(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "a", "name": "A", "url": "http://a:11434"},
    ), raising=False)
    host = R.get_hosts()[0]
    R.mark_unhealthy(host, "boom", cooldown=0.01)
    assert R.in_cooldown(host)
    import time
    time.sleep(0.05)
    assert not R.in_cooldown(host)


def test_pick_host_requires_model(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "a", "name": "A", "url": "http://a:11434"},
        {"id": "b", "name": "B", "url": "http://b:11434"},
    ), raising=False)

    async def fake_probe(host, force=False, timeout=None):
        models = ["qwen2.5:3b-instruct"] if host.id == "b" else ["llava:7b"]
        return _mk_status(host, models=models)

    monkeypatch.setattr(R, "probe", fake_probe)
    picked = asyncio.run(R.pick_host(required_model="qwen2.5:3b-instruct"))
    assert picked.id == "b"


# ── Model-substitution ladder ────────────────────────────────────────


def test_model_ladder_substitutes_best_installed():
    status = R.HostStatus(host_id="x", online=True,
                          models=["llava:7b", "qwen2.5:7b-instruct"])
    model, subbed = R.resolve_model_for_host(status, "moondream:1.8b", "vision")
    assert (model, subbed) == ("llava:7b", True)
    model, subbed = R.resolve_model_for_host(status, "qwen2.5:3b-instruct", "text")
    assert (model, subbed) == ("qwen2.5:7b-instruct", True)


def test_model_ladder_keeps_requested_when_installed():
    status = R.HostStatus(host_id="x", online=True, models=["moondream:1.8b"])
    model, subbed = R.resolve_model_for_host(status, "moondream:1.8b", "vision")
    assert (model, subbed) == ("moondream:1.8b", False)


def test_model_ladder_falls_through_when_nothing_matches():
    status = R.HostStatus(host_id="x", online=True, models=["tinyllama"])
    model, subbed = R.resolve_model_for_host(status, "moondream:1.8b", "vision")
    assert (model, subbed) == ("moondream:1.8b", False)  # legacy 404 path preserved


# ── request_with_failover ────────────────────────────────────────────


def _failover_hosts(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "a", "name": "A", "url": "http://a:11434"},
        {"id": "b", "name": "B", "url": "http://b:11434"},
    ), raising=False)
    return {h.id: h for h in R.get_hosts()}


def test_failover_walks_to_next_host_on_connect_error(monkeypatch):
    hosts = _failover_hosts(monkeypatch)
    monkeypatch.setattr(R, "CONNECT_RETRY_BACKOFF_S", 0.0)
    attempts = []

    async def send(host):
        attempts.append(host.id)
        if host.id == "a":
            raise httpx.ConnectError("dead socket")
        return httpx.Response(200, json={"ok": True},
                              request=httpx.Request("GET", host.url))

    resp, host = asyncio.run(R.request_with_failover(send))
    # Transient connect errors get ONE same-host retry (a LAN blip / mid-restart
    # shouldn't drop work onto a weaker fallback) before failing over to b.
    assert attempts == ["a", "a", "b"]
    assert host.id == "b"
    assert resp.status_code == 200
    # Host a is now cooling down (it failed both attempts).
    assert R.in_cooldown(hosts["a"])
    assert not R.in_cooldown(hosts["b"])


def test_failover_503_retries_once_then_moves_on(monkeypatch):
    _failover_hosts(monkeypatch)
    monkeypatch.setattr(R, "MAX_RETRY_AFTER_S", 0.01)
    attempts = []

    async def send(host):
        attempts.append(host.id)
        if host.id == "a":
            return httpx.Response(503, headers={"Retry-After": "0"},
                                  request=httpx.Request("GET", host.url))
        return httpx.Response(200, request=httpx.Request("GET", host.url))

    resp, host = asyncio.run(R.request_with_failover(send))
    # a tried twice (initial + one Retry-After retry), then b served it.
    assert attempts == ["a", "a", "b"]
    assert host.id == "b"


def test_failover_raises_after_full_pass(monkeypatch):
    _failover_hosts(monkeypatch)

    async def send(host):
        raise httpx.ConnectError("everything is down")

    with pytest.raises(httpx.ConnectError):
        asyncio.run(R.request_with_failover(send))


def test_probe_503_reports_paused(monkeypatch):
    """A Companion returns 503 while sharing is paused — probe() must flag it as
    a distinct `paused` state (not a generic HTTP error) so the UI can tell the
    user to resume sharing rather than reporting the host as broken."""
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
        {"id": "comp", "name": "Companion", "url": "http://desktop:11500/ollama"},
    ), raising=False)
    R.reset_state()
    host = R.get_hosts()[0]

    def _handler(request):
        return httpx.Response(503, headers={"Retry-After": "5"})

    orig_client = httpx.AsyncClient

    def _client(*a, **kw):
        kw.pop("timeout", None)
        return orig_client(transport=httpx.MockTransport(_handler))

    monkeypatch.setattr(R.httpx, "AsyncClient", _client)
    st = asyncio.run(R.probe(host, force=True))
    assert st.online is False
    assert st.paused is True
    assert "503" in st.error and "resume" in st.error.lower()


def test_failover_no_hosts_configured(monkeypatch):
    monkeypatch.setattr(settings, "OLLAMA_HOSTS", "", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_HOST", "", raising=False)

    async def send(host):  # pragma: no cover — never reached
        raise AssertionError

    with pytest.raises(RuntimeError):
        asyncio.run(R.request_with_failover(send))


# ── Contract: provider text call succeeds via host #2 when host #1 is a
#    dead socket ────────────────────────────────────────────────────────


class _MiniOllama(threading.Thread):
    """One-shot fake Ollama /api/chat responder on a real TCP socket."""

    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.requests = []
        self._stop = threading.Event()

    def run(self):
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(2)
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    data += chunk
                # Read the body per Content-Length (best-effort).
                try:
                    head, rest = data.split(b"\r\n\r\n", 1)
                    clen = 0
                    for line in head.split(b"\r\n"):
                        if line.lower().startswith(b"content-length:"):
                            clen = int(line.split(b":", 1)[1].strip())
                    while len(rest) < clen:
                        rest += conn.recv(65536)
                    data = head + b"\r\n\r\n" + rest
                except Exception:
                    pass
                self.requests.append(data.decode("utf-8", "replace"))
                body = json.dumps({
                    "message": {"role": "assistant",
                                "content": "served by fallback"},
                    "done": True, "prompt_eval_count": 1, "eval_count": 1,
                })
                resp = ("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}")
                try:
                    conn.sendall(resp.encode())
                except Exception:
                    pass

    def stop(self):
        self._stop.set()
        try:
            self.sock.close()
        except Exception:
            pass


def _free_dead_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()  # nothing listens here anymore → connection refused
    return port


def test_provider_text_call_fails_over_to_second_host(monkeypatch):
    server = _MiniOllama()
    server.start()
    dead_port = _free_dead_port()
    try:
        monkeypatch.setattr(settings, "OLLAMA_HOSTS", _hosts_json(
            {"id": "dead", "name": "Dead primary",
             "url": f"http://127.0.0.1:{dead_port}"},
            {"id": "live", "name": "Live fallback",
             "url": f"http://127.0.0.1:{server.port}", "token": "fallback-tok"},
        ), raising=False)
        monkeypatch.setattr(settings, "OLLAMA_HOST",
                            f"http://127.0.0.1:{dead_port}", raising=False)
        R.reset_state()

        from backend.services.providers.ollama_provider import OllamaProvider
        provider = OllamaProvider()
        provider._capabilities_detected = True  # skip /api/show probing

        async def _run():
            try:
                return await provider._call_text("hello", max_tokens=8, timeout=10)
            finally:
                await provider.close()

        result = asyncio.run(_run())
        assert result == "served by fallback"
        assert provider._host == f"http://127.0.0.1:{server.port}"
        # The fallback host's bearer token was attached by the event hook.
        assert any("authorization: bearer fallback-tok" in r.lower()
                   for r in server.requests)
    finally:
        server.stop()
