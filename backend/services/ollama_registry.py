"""Multi-host Ollama registry with ordered failover (remote GPU sharing).

ClipAI can talk to more than one Ollama daemon: the Unraid sidecar, a
desktop RTX 4070 shared through the GPU Companion proxy, a Mac on the
LAN. The user orders them in Settings (drag-and-drop); **array order is
priority** — index 0 is the primary, the rest are fallbacks tried in
order.

The registry is the single source of truth for "which Ollama do I call
and with what credentials":

  * ``get_hosts()``            — parse ``settings.OLLAMA_HOSTS`` (JSON array),
                                 synthesizing a one-entry registry from the
                                 legacy ``OLLAMA_HOST`` when unset, so
                                 env-only deployments keep working untouched.
  * ``probe(host)``            — GET ``{url}/api/tags`` with a short timeout,
                                 cached ~10 s per host.
  * ``pick_host()``            — first enabled + online host in order (that
                                 has a required model, when asked).
  * ``request_with_failover()``— run one HTTP request against the host list,
                                 walking to the next host on connect errors /
                                 timeouts (with a ~30 s unhealthy cooldown) and
                                 treating HTTP 503 as "busy — honor Retry-After
                                 once, then move on". After a full pass the
                                 last error is raised so the provider-level
                                 ``AI_FALLBACK_CHAIN`` takes over exactly as
                                 before.

Per-host bearer tokens are attached on every call (the GPU Companion
proxy requires one). Tokens are never logged. Host URLs may carry a base
path (the Companion exposes Ollama at ``…:11500/ollama``) — always join
with :func:`join_url`, never assume a bare host root.

Back-compat contract (zero-regression): ``settings.OLLAMA_HOST`` is kept
in sync with the current primary so the many legacy call sites that read
it directly keep hitting the right daemon.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import httpx

from backend.config import settings
from backend.services.request_context import clipai_headers

logger = logging.getLogger(__name__)

# How long a probe result stays fresh before we re-hit /api/tags.
PROBE_CACHE_TTL = 10.0
# How long a host sits out after a connect error / timeout before it is
# retried. Short on purpose: a rebooting desktop should come back quickly.
UNHEALTHY_COOLDOWN_S = 30.0
# Probe timeout — hosts are on the LAN; anything slower is effectively down.
PROBE_TIMEOUT_S = 3.0
# Cap on how long we honor a host's Retry-After before moving to the next one.
MAX_RETRY_AFTER_S = 15.0

# Model substitution ladders — when the chosen host lacks the configured
# model, fall back to the best model it DOES have instead of failing the
# job. Ordered best-first; the requested model always wins when present.
MODEL_LADDER = {
    "vision": ["qwen2.5vl:7b", "llava:13b", "llava:7b", "qwen2.5vl:3b",
               "qwen2.5-vl:7b", "moondream:1.8b"],
    "text": ["qwen2.5:14b", "qwen2.5:7b-instruct", "qwen2.5:3b-instruct"],
}


@dataclass
class OllamaHost:
    id: str
    name: str
    url: str  # normalized: no trailing slash; may include a base path (/ollama)
    token: str = ""
    enabled: bool = True
    # Advertised by a paired GPU Companion (empty for plain Ollama hosts).
    gpu_name: str = ""
    vram_total_mb: int = 0
    is_companion: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "url": self.url,
                "token": self.token, "enabled": self.enabled,
                "gpu_name": self.gpu_name, "vram_total_mb": self.vram_total_mb,
                "is_companion": self.is_companion}


def model_present(installed: list, requested: str) -> bool:
    """True if ``requested`` matches an installed Ollama tag, tolerating an
    implicit ``:latest`` and an ``ollama/`` prefix on either side."""
    if not requested:
        return False
    req = str(requested).split("/")[-1]
    wanted = {req, req if ":" in req else f"{req}:latest"}
    for m in installed or []:
        mm = str(m).split("/")[-1]
        if mm in wanted or (":" not in req and mm.split(":")[0] == req):
            return True
    return False


@dataclass
class HostStatus:
    host_id: str
    online: bool
    models: list = field(default_factory=list)
    latency_ms: Optional[float] = None
    error: str = ""
    version: str = ""
    checked_at: float = 0.0
    # True when the host answered but is refusing work (HTTP 503) — for the
    # Companion proxy this means sharing is paused / the GPU is momentarily busy.
    paused: bool = False


# ── Module state ─────────────────────────────────────────────────────
_probe_cache: dict[str, HostStatus] = {}
_unhealthy_until: dict[str, float] = {}
# Cache keyed by the raw OLLAMA_HOSTS string so repeated get_hosts() calls
# don't re-parse JSON on the hot path.
_parse_cache: tuple[str, str, list] = ("", "", [])


def _normalize_url(url: str) -> str:
    url = (url or "").strip().rstrip("/")
    if url and "://" not in url:
        url = f"http://{url}"
    return url


# Hostnames that resolve to the ClipAI server's OWN GPU (loopback or the
# docker-compose Ollama sidecar). Anything else — a LAN IP, a Companion
# proxy — is a REMOTE GPU: never evict its models to free the local card,
# and never run the local-VRAM serialization dance on its behalf.
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "0.0.0.0", "::1",
                   "ollama", "clipai-ollama", "host.docker.internal"}


def is_local_gpu_host(url: str) -> bool:
    """Heuristic: does this Ollama host share the server's own GPU?"""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url if "://" in (url or "") else f"http://{url}")
        return (parsed.hostname or "").lower() in LOCAL_HOSTNAMES
    except Exception:
        return True


def join_url(host_url: str, path: str) -> str:
    """Join a host URL (which may include a base path like ``/ollama``)
    with an API path. ``path`` must start with ``/``."""
    if not path.startswith("/"):
        path = "/" + path
    return _normalize_url(host_url) + path


def _parse_hosts_json(raw: str) -> list[OllamaHost]:
    hosts: list[OllamaHost] = []
    try:
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError("OLLAMA_HOSTS must be a JSON array")
    except Exception as e:
        logger.warning("OLLAMA_HOSTS is not valid JSON (%s) — ignoring it", e)
        return []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        url = _normalize_url(str(entry.get("url", "")))
        if not url:
            continue
        hosts.append(OllamaHost(
            id=str(entry.get("id") or uuid.uuid4().hex[:8]),
            name=str(entry.get("name") or url),
            url=url,
            token=str(entry.get("token") or ""),
            enabled=bool(entry.get("enabled", True)),
            gpu_name=str(entry.get("gpu_name") or ""),
            vram_total_mb=int(entry.get("vram_total_mb") or 0),
            is_companion=bool(entry.get("is_companion", False)),
        ))
    return hosts


def get_hosts() -> list[OllamaHost]:
    """The ordered host registry. Index 0 = primary.

    When ``OLLAMA_HOSTS`` is empty and the legacy ``OLLAMA_HOST`` is set, a
    one-entry registry named "Default" is synthesized so env-only / Unraid
    template deployments behave exactly as before this feature existed.
    """
    global _parse_cache
    raw = (getattr(settings, "OLLAMA_HOSTS", "") or "").strip()
    legacy = _normalize_url(getattr(settings, "OLLAMA_HOST", "") or "")
    cached_raw, cached_legacy, cached_hosts = _parse_cache
    if raw == cached_raw and legacy == cached_legacy:
        return list(cached_hosts)

    hosts: list[OllamaHost] = []
    if raw:
        hosts = _parse_hosts_json(raw)
    if not hosts and legacy:
        hosts = [OllamaHost(id="default", name="Default", url=legacy)]
    _parse_cache = (raw, legacy, list(hosts))
    return list(hosts)


def enabled_hosts() -> list[OllamaHost]:
    return [h for h in get_hosts() if h.enabled and h.url]


def companion_host() -> Optional[OllamaHost]:
    """The paired GPU Companion host — the box that serves Ollama AND Whisper
    on one GPU. First enabled ``is_companion`` host; else the first enabled
    non-local host whose URL ends with ``/ollama`` (the Companion proxy shape,
    which covers hosts added manually in the UI). ``None`` if there's no remote
    Companion. This is THE single source of truth for "which GPU also does
    transcription", so remote Whisper needs no separate configuration."""
    hosts = enabled_hosts()
    comp = next((h for h in hosts if h.is_companion), None)
    if comp is not None:
        return comp
    return next((h for h in hosts
                 if not is_local_gpu_host(h.url)
                 and h.url.rstrip("/").endswith("/ollama")), None)


def companion_base(host: OllamaHost) -> str:
    """A Companion's base URL (for ``/v1/audio/transcriptions``, ``/v1/health``)
    — its Ollama URL minus a trailing ``/ollama``."""
    url = (getattr(host, "url", "") or "").rstrip("/")
    if url.endswith("/ollama"):
        url = url[: -len("/ollama")]
    return url.rstrip("/")


def routable_hosts() -> list[OllamaHost]:
    """Hosts eligible to serve a request, in priority order.

    Same as :func:`enabled_hosts`, except when ``GPU_STRICT_REMOTE`` is on AND a
    remote host exists: the local-GPU daemon (the weak on-server card) is
    dropped so work never silently falls back to it — if every remote is down
    the Ollama provider fails and the AI fallback chain goes to the cloud
    instead. With no remote host configured this is a no-op so single-GPU
    deployments are unaffected."""
    hosts = enabled_hosts()
    if getattr(settings, "GPU_STRICT_REMOTE", False):
        remote = [h for h in hosts if not is_local_gpu_host(h.url)]
        if remote:
            return remote
    return hosts


def save_hosts(hosts: list[OllamaHost], persist: bool = True) -> None:
    """Persist a new registry (order = priority) and sync the legacy
    ``settings.OLLAMA_HOST`` to the new primary so untouched call sites
    keep talking to the right daemon."""
    settings.OLLAMA_HOSTS = json.dumps([h.to_dict() for h in hosts])
    primary = next((h for h in hosts if h.enabled and h.url), None)
    if primary:
        settings.OLLAMA_HOST = primary.url
    if persist:
        try:
            from backend.routers.settings import _persist_user_settings
            _persist_user_settings()
        except Exception as e:
            logger.warning("Could not persist Ollama host registry: %s", e)


def primary_host() -> Optional[OllamaHost]:
    hosts = enabled_hosts()
    return hosts[0] if hosts else None


def primary_url() -> str:
    h = primary_host()
    return h.url if h else _normalize_url(getattr(settings, "OLLAMA_HOST", "") or "")


def find_host_for_url(url: str) -> Optional[OllamaHost]:
    """Match a full request URL back to its registered host (longest-prefix
    wins, so ``…:11500/ollama`` beats ``…:11500``)."""
    url = url or ""
    best: Optional[OllamaHost] = None
    for h in get_hosts():
        if h.url and url.startswith(h.url):
            if best is None or len(h.url) > len(best.url):
                best = h
    return best


def auth_headers(host: Optional[OllamaHost]) -> dict:
    """Bearer + ``X-ClipAI-*`` headers for one host. Token never logged."""
    headers = dict(clipai_headers())
    if host is not None and host.token:
        headers["Authorization"] = f"Bearer {host.token}"
    return headers


def headers_for_url(url: str) -> dict:
    """Headers for an arbitrary Ollama request URL — used by legacy call
    sites that build URLs from ``settings.OLLAMA_HOST`` directly."""
    return auth_headers(find_host_for_url(url))


def mark_unhealthy(host: OllamaHost, reason: str,
                   cooldown: float = UNHEALTHY_COOLDOWN_S) -> None:
    _unhealthy_until[host.id] = time.monotonic() + cooldown
    _probe_cache.pop(host.id, None)
    logger.warning("Ollama host '%s' (%s) marked unhealthy for %.0fs: %s",
                   host.name, host.url, cooldown, reason)


def note_success(host: OllamaHost) -> None:
    _unhealthy_until.pop(host.id, None)


def in_cooldown(host: OllamaHost) -> bool:
    return time.monotonic() < _unhealthy_until.get(host.id, 0.0)


def reset_state() -> None:
    """Test helper — drop probe caches and cooldowns."""
    global _parse_cache
    _probe_cache.clear()
    _unhealthy_until.clear()
    _parse_cache = ("", "", [])


async def probe(host: OllamaHost, force: bool = False,
                timeout: float = PROBE_TIMEOUT_S) -> HostStatus:
    """GET ``{url}/api/tags`` (bearer token when set). Cached ~10 s."""
    now = time.time()
    cached = _probe_cache.get(host.id)
    if cached is not None and not force and now - cached.checked_at < PROBE_CACHE_TTL:
        return cached

    status = HostStatus(host_id=host.id, checked_at=now, online=False)
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(join_url(host.url, "/api/tags"),
                                    headers=auth_headers(host))
            status.latency_ms = round((time.monotonic() - t0) * 1000, 1)
            if resp.status_code == 200:
                data = resp.json() or {}
                status.online = True
                status.models = [m.get("name", "") for m in data.get("models", [])
                                 if m.get("name")]
            elif resp.status_code in (401, 403):
                status.error = f"auth rejected (HTTP {resp.status_code}) — check the host token"
            elif resp.status_code == 503:
                # Companion returns 503 while sharing is paused (or the GPU is
                # briefly busy). Surface it as a distinct, actionable state.
                status.paused = True
                status.error = "paused (HTTP 503) — resume GPU sharing in the Companion app"
            else:
                status.error = f"HTTP {resp.status_code}"
    except Exception as e:
        status.error = f"{type(e).__name__}: {str(e)[:120]}"
    _probe_cache[host.id] = status
    return status


def _host_has_model(status: HostStatus, model: str) -> bool:
    try:
        from backend.services.local_models import _ollama_names_match
        return any(_ollama_names_match(m, model) for m in status.models)
    except Exception:
        want = (model or "").lower()
        return any(m.lower() == want or m.lower().split(":")[0] == want.split(":")[0]
                   for m in status.models)


async def pick_host(required_model: Optional[str] = None) -> Optional[OllamaHost]:
    """First enabled + online host in priority order (that has the model,
    when one is required). Every skip is logged with its reason."""
    for host in routable_hosts():
        if in_cooldown(host):
            logger.info("Ollama host '%s' skipped: in failure cooldown", host.name)
            continue
        status = await probe(host)
        if not status.online:
            logger.info("Ollama host '%s' skipped: offline (%s)",
                        host.name, status.error or "no response")
            continue
        if required_model and status.models and not _host_has_model(status, required_model):
            logger.info("Ollama host '%s' skipped: model %r not installed",
                        host.name, required_model)
            continue
        return host
    return None


def resolve_model_for_host(status: HostStatus, requested: str,
                           kind: str = "text") -> tuple[str, bool]:
    """Best model available on a host for a request.

    Returns ``(model, substituted)``. The requested model always wins when
    installed; otherwise the first installed entry of ``MODEL_LADDER[kind]``
    is used. When nothing matches (or the model list is unknown) the
    requested model is returned unchanged so the normal 404 → provider
    fallback path is preserved.
    """
    if not status.models or _host_has_model(status, requested):
        return requested, False
    for candidate in MODEL_LADDER.get(kind, []):
        if _host_has_model(status, candidate):
            logger.warning(
                "Ollama model substitution: host lacks %r — using %r "
                "(best installed %s-ladder model)", requested, candidate, kind)
            return candidate, True
    return requested, False


def _retry_after_seconds(resp: httpx.Response) -> float:
    try:
        return min(MAX_RETRY_AFTER_S, max(0.0, float(resp.headers.get("Retry-After", "1"))))
    except (TypeError, ValueError):
        return 1.0


async def request_with_failover(
    send: Callable[[OllamaHost], Awaitable[httpx.Response]],
    *,
    required_model: Optional[str] = None,
    accept_status: Callable[[httpx.Response], bool] = lambda r: r.status_code < 500,
) -> tuple[httpx.Response, OllamaHost]:
    """Run one logical request against the host list in priority order.

    ``send(host)`` performs the actual HTTP call (the caller builds the URL
    with :func:`join_url` and attaches :func:`auth_headers`). Behavior:

      * connect error / timeout → mark the host unhealthy (~30 s cooldown)
        and retry the SAME request on the next host;
      * HTTP 503 → the host is busy: wait ``Retry-After`` (capped), retry it
        once, then move on to the next host;
      * any other response → returned to the caller (per-host 4xx/5xx
        semantics such as OOM ladders stay caller-owned) unless
        ``accept_status`` rejects it, in which case the next host is tried.

    After one full pass the last connection error (or a RuntimeError) is
    raised so the existing ``AI_FALLBACK_CHAIN`` takes over exactly as today.
    """
    hosts = routable_hosts()
    if not hosts:
        raise RuntimeError("No Ollama hosts configured")

    last_exc: Optional[Exception] = None
    last_resp: Optional[tuple[httpx.Response, OllamaHost]] = None
    for host in hosts:
        if in_cooldown(host):
            logger.info("Ollama failover: skipping '%s' (cooldown)", host.name)
            continue
        if required_model:
            status = _probe_cache.get(host.id)
            if status is None or time.time() - status.checked_at >= PROBE_CACHE_TTL:
                status = await probe(host)
            if status.online and status.models and not _host_has_model(status, required_model):
                logger.info("Ollama failover: skipping '%s' (model %r not installed)",
                            host.name, required_model)
                continue
        for attempt in (0, 1):
            try:
                resp = await send(host)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                    httpx.WriteTimeout, httpx.PoolTimeout, httpx.RemoteProtocolError,
                    OSError) as e:
                last_exc = e
                mark_unhealthy(host, f"{type(e).__name__}: {str(e)[:100]}")
                break  # next host
            if resp.status_code == 503 and attempt == 0:
                wait = _retry_after_seconds(resp)
                logger.info("Ollama host '%s' is busy (503) — retrying once in %.1fs",
                            host.name, wait)
                await asyncio.sleep(wait)
                continue
            if resp.status_code == 503:
                logger.info("Ollama host '%s' still busy after retry — trying next host",
                            host.name)
                last_resp = (resp, host)
                break  # next host
            note_success(host)
            if not accept_status(resp):
                last_resp = (resp, host)
                break  # next host
            return resp, host

    if last_resp is not None:
        return last_resp
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("All configured Ollama hosts are unavailable")


async def registry_status(force: bool = False) -> list[dict]:
    """Probe every host (in parallel) for the Settings UI / provider status."""
    hosts = get_hosts()
    results = await asyncio.gather(*(probe(h, force=force) for h in hosts),
                                   return_exceptions=True)
    out = []
    for idx, (host, st) in enumerate(zip(hosts, results)):
        if isinstance(st, Exception):
            st = HostStatus(host_id=host.id, online=False, error=str(st)[:120])
        out.append({
            "id": host.id,
            "name": host.name,
            "url": host.url,
            "enabled": host.enabled,
            "has_token": bool(host.token),
            "priority": idx,
            "role": "primary" if idx == 0 else f"fallback_{idx}",
            "online": st.online,
            "models": st.models,
            "latency_ms": st.latency_ms,
            "error": st.error,
            "paused": bool(getattr(st, "paused", False)),
            "in_cooldown": in_cooldown(host),
            "gpu_name": host.gpu_name,
            "vram_total_mb": host.vram_total_mb,
            "is_companion": host.is_companion,
        })
    return out
