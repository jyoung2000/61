import asyncio
import logging
import re
import time
from typing import Optional

from backend.config import settings
from backend.models import (
    FrameData, SceneDescription, TranscriptSegment, VideoSummary, ClipCandidate, ClipSEO,
)
from backend.services.providers.base import (
    AIProvider, ProviderError, ProviderRateLimitError, AllProvidersFailedError,
)
from backend.services.providers.openrouter_provider import OpenRouterProvider
from backend.services.providers.anthropic_provider import AnthropicProvider
from backend.services.providers.gemini_provider import GeminiProvider
from backend.services.providers.groq_provider import GroqProvider
from backend.services.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)

# WebSocket broadcast callback type
WsBroadcastCallback = Optional[object]  # Will be a callable


def _is_key_limit_error(exc) -> bool:
    """True for a HARD, non-retryable provider billing error — an OpenRouter
    "Key limit exceeded" 403, out-of-credits, or a quota/billing 403. These do
    NOT recover within a job, so the provider should be abandoned for the rest
    of the run instead of re-tried every stage."""
    s = str(exc).lower()
    if "key limit exceeded" in s:
        return True
    if "insufficient" in s and "credit" in s:
        return True
    if "403" in s and ("limit" in s or "quota" in s or "billing" in s
                       or "credit" in s or "exceeded" in s):
        return True
    return False


class _CircuitBreaker:
    """Marks a provider degraded for 15 min after 3 failures in 10 min."""

    def __init__(self):
        self._failures: dict[str, list[float]] = {}
        self._degraded_until: dict[str, float] = {}

    def is_degraded(self, name: str) -> bool:
        if name in self._degraded_until:
            if time.monotonic() < self._degraded_until[name]:
                return True
            del self._degraded_until[name]
        return False

    def record_failure(self, name: str):
        now = time.monotonic()
        if name not in self._failures:
            self._failures[name] = []
        self._failures[name] = [t for t in self._failures[name] if now - t < 600]
        self._failures[name].append(now)
        failure_count = len(self._failures[name])
        logger.info("Circuit breaker: %s failure %d/3 in 10-min window", name, failure_count)
        if failure_count >= 3:
            self._degraded_until[name] = now + 900  # 15 min
            logger.warning("Circuit breaker: provider %s marked DEGRADED for 15 minutes", name)

    def record_success(self, name: str):
        was_degraded = name in self._degraded_until
        self._failures.pop(name, None)
        self._degraded_until.pop(name, None)
        if was_degraded:
            logger.info("Circuit breaker: provider %s RECOVERED (success after degraded)", name)

    def clear_degraded(self, name: str):
        """Immediately remove degraded status for a provider."""
        was_degraded = name in self._degraded_until
        self._degraded_until.pop(name, None)
        if was_degraded:
            logger.info("Circuit breaker: %s manually un-degraded before critical operation", name)

    def force_reset_all(self):
        """Reset ALL provider states. Used before critical pipeline stages."""
        had_degraded = list(self._degraded_until.keys())
        self._failures.clear()
        self._degraded_until.clear()
        if had_degraded:
            logger.info("Circuit breaker: RESET all states (was degraded: %s)", had_degraded)


def _build_provider(name: str) -> Optional[AIProvider]:
    try:
        if name == "openrouter" and settings.OPENROUTER_API_KEY:
            return OpenRouterProvider()
        elif name == "anthropic" and settings.ANTHROPIC_API_KEY:
            return AnthropicProvider()
        elif name == "gemini" and settings.GEMINI_API_KEY:
            return GeminiProvider()
        elif name == "groq" and settings.GROQ_API_KEY:
            return GroqProvider()
        elif name == "ollama":
            return OllamaProvider()
    except Exception as e:
        logger.warning(f"Failed to initialize provider {name}: {e}")
    return None


_OPENROUTER_MODEL_CACHE: dict[str, bool] = {}


async def _openrouter_model_exists(model_id: str, timeout: float = 8.0) -> bool | None:
    """Return True / False if ``model_id`` is in OpenRouter's catalog,
    or ``None`` when the catalog couldn't be fetched (network error).

    Cached per-process so we don't hit /models on every job start.
    """
    if not model_id:
        return None
    cached = _OPENROUTER_MODEL_CACHE.get(model_id)
    if cached is not None:
        return cached
    try:
        import httpx
        headers = {}
        key = getattr(settings, "OPENROUTER_API_KEY", "") or ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get("https://openrouter.ai/api/v1/models", headers=headers)
            if r.status_code != 200:
                return None
            data = r.json() or {}
            ids = {m.get("id") for m in (data.get("data") or [])}
            for mid in ids:
                if mid:
                    _OPENROUTER_MODEL_CACHE[mid] = True
            exists = model_id in ids
            _OPENROUTER_MODEL_CACHE[model_id] = exists
            return exists
    except Exception as e:
        logger.debug("OpenRouter model-exists probe failed: %s", e)
        return None


def _probe_ollama_reachable(timeout: float = 1.5) -> bool:
    """Cheap synchronous TCP ping of the Ollama daemon(s).

    Returns True iff a TCP connection to ANY enabled host in the Ollama
    registry (falling back to ``settings.OLLAMA_HOST`` when no registry is
    configured) opens within ``timeout`` seconds. Used by the orchestrator
    to skip Ollama on jobs where no daemon is running, instead of waiting
    for the first HTTP call to time out (which can take 30+ seconds and
    cascade through every batch). With fallback hosts configured, a dead
    primary alone no longer disqualifies Ollama — the provider fails over.
    """
    import socket
    from urllib.parse import urlparse

    candidates: list[str] = []
    try:
        from backend.services import ollama_registry
        candidates = [h.url for h in ollama_registry.enabled_hosts()]
    except Exception:
        pass
    if not candidates:
        candidates = [getattr(settings, "OLLAMA_HOST", "") or ""]

    for host_url in candidates:
        try:
            parsed = urlparse(host_url if "://" in host_url else f"http://{host_url}")
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 11434
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (OSError, ValueError):
            continue
    return False


class _GpuClassLease:
    """Serializes GPU model CLASSES (big text vs vision) on a remote card
    that cannot hold both models resident at once.

    When gemma3:12b (~8.3 GB) and llava:7b (~5.5 GB) share a 9.5 GB budget,
    running the translation loop concurrently with clip vision analysis
    CPU-spills the text model — a real run degraded every text call to
    21s-3m26s for 21 minutes and shipped an untranslated transcript. The
    lease keeps the OVERLAP (clips still run during translation) but hands
    the GPU to one model class at a time: same-class calls run concurrently,
    an opposing class waits for the current holders to drain plus a short
    grace window (so back-to-back calls of one class don't ping-pong
    multi-GB model loads)."""

    def __init__(self, grace_s: float = 15.0):
        self._grace = grace_s
        self._cls: str | None = None
        self._holders = 0
        self._last_release = 0.0
        self._cond: asyncio.Condition | None = None

    def _condition(self) -> asyncio.Condition:
        # Created lazily so the lease can be built before a loop exists.
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    async def acquire(self, cls: str) -> None:
        cond = self._condition()
        async with cond:
            while True:
                if self._holders == 0:
                    if self._cls in (None, cls):
                        break
                    remaining = self._grace - (time.monotonic() - self._last_release)
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        pass
                    continue
                if self._cls == cls:
                    break
                await cond.wait()
            self._cls = cls
            self._holders += 1

    async def release(self) -> None:
        cond = self._condition()
        async with cond:
            self._holders = max(0, self._holders - 1)
            self._last_release = time.monotonic()
            cond.notify_all()


def _est_model_gb(model_name: str) -> float:
    """Rough resident-size estimate for an Ollama model from its name.

    Weights ≈ params × 0.62 GB/B (q4_K_M-class quants) + ~1.2 GB runtime
    overhead (KV cache, CUDA context). Checked against observation:
    gemma3:12b-it-q4_K_M ≈ 8.3 GB actual vs 8.6 est; llava:7b ≈ 5-6 GB
    actual vs 5.5 est. Only used for a coarse "do these two co-fit"
    decision, so ±15% is fine. Returns 0 when no parameter count is
    parseable (treated as unknown → assume it fits)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", (model_name or "").lower())
    if not m:
        return 0.0
    try:
        return float(m.group(1)) * 0.62 + 1.2
    except ValueError:
        return 0.0


class AIOrchestrator:
    """
    Tries providers in fallback chain order.
    Circuit breaker: marks provider degraded for 15 min after 3 failures in 10 min.
    """

    # Model downgrade fallback for consecutive Ollama failures
    _SMALLER_MODELS = ["qwen2.5:1.5b-instruct", "qwen2.5:0.5b-instruct", "tinyllama"]
    _FAILURE_THRESHOLD_FOR_DOWNGRADE = 3
    # Class-level defaults so instances constructed without __init__ (test
    # stubs) still work. The lease is deliberately CLASS-level: concurrent
    # jobs share the one physical remote GPU, so their text/vision phases
    # must take turns with each other too.
    _gpu_lease_needed: bool | None = None
    _last_ollama_defrag: float = 0.0
    # Residency observed when the last defrag fired, and the futility window:
    # when a defrag doesn't improve residency the pressure is EXTERNAL
    # (whisper sidecar / desktop apps), and further evict+reload cycles only
    # add churn — suspend them and ride on extended timeouts instead.
    _frac_at_last_defrag: float = -1.0
    _defrag_futile_until: float = 0.0
    _gpu_class_lease = _GpuClassLease()

    def __init__(self, ws_broadcast=None, custom_prompts=None, cancel_check=None):
        self._circuit_breaker = _CircuitBreaker()
        self._ws_broadcast = ws_broadcast
        self._custom_prompts = custom_prompts  # PromptSet or None
        self._cancel_check = cancel_check  # callable that raises on cancel
        self._providers: dict[str, AIProvider] = {}
        self._consecutive_ollama_failures: int = 0
        # Cooldown stamp for the CPU-spill defrag (evict-all + settle) so a
        # burst of concurrent degraded calls triggers ONE defrag, not a storm.
        # (The GPU class lease + the fit-check memo live on the CLASS: one
        # physical remote GPU is shared by every orchestrator instance.)
        self._last_ollama_defrag: float = 0.0
        self._current_model_override: str | None = None
        # Provider names that we know are unreachable for the lifetime
        # of this orchestrator (e.g. Ollama daemon not running, but
        # listed in AI_FALLBACK_CHAIN). Excluded from ``_get_active_chain``
        # so we don't burn 30s on every job trying to hit a dead daemon
        # before falling back to OpenRouter.
        self._unreachable: set[str] = set()
        # Providers that returned a HARD, non-retryable billing error this job
        # (OpenRouter "Key limit exceeded" 403 / out-of-credits). Unlike the
        # circuit breaker (3-strikes, reset between stages) this is sticky for
        # the whole job — a dead key WON'T recover mid-run, so re-attempting it
        # every stage just wastes minutes on 403s before the local fallback.
        # Skipped in ``_get_active_chain`` and NOT cleared by
        # ``reset_circuit_breaker`` (it resets next job — new orchestrator).
        self._billing_dead: set[str] = set()
        for name in settings.editorial_provider_chain:
            p = _build_provider(name)
            if p:
                self._providers[name] = p
        # Best-effort sync reachability probe for Ollama.
        # We do NOT probe cloud providers here — their auth check costs
        # money and the orchestrator does proper fallback on first call.
        if "ollama" in self._providers:
            try:
                if not _probe_ollama_reachable():
                    self._unreachable.add("ollama")
                    logger.warning(
                        "Ollama is in AI_FALLBACK_CHAIN but %s is not "
                        "reachable — skipping for this job.",
                        getattr(settings, "OLLAMA_HOST", "(host unset)"),
                    )
            except Exception as e:
                logger.debug("Ollama reachability probe failed: %s", e)

        # Masked fingerprint (last-4 + length) of every API key, so the active
        # key is confirmable at a glance — that a freshly-saved key, not a
        # cached old one, is in play. Reflects the per-user overlay for this
        # job; the full secret is never logged.
        try:
            _fps = settings.api_key_fingerprints()
            logger.info("Active API keys (last-4): %s",
                        " · ".join(f"{k}={v}" for k, v in _fps.items()))
        except Exception:
            pass

    def _wire_ws_to_providers(self, job_id: str):
        """Pass WebSocket broadcast to providers that support model-level notifications."""
        if not self._ws_broadcast:
            return
        for provider in self._providers.values():
            if hasattr(provider, 'set_ws_broadcast'):
                provider.set_ws_broadcast(self._ws_broadcast, job_id)

    def _get_model_info(self, provider) -> str:
        """Get human-readable model info string for a provider."""
        pname = provider.provider_name
        parts = []
        if hasattr(provider, '_primary_model'):
            parts.append(f"vision={provider._primary_model}")
        if hasattr(provider, '_editorial_model'):
            parts.append(f"text={provider._editorial_model}")
        if parts:
            return f"{pname} ({', '.join(parts)})"
        return pname

    def _get_task_model(self, provider, task: str) -> str:
        """Get the specific model name used for a task (vision/text/summary/clips).

        Returns a string like 'reka/reka-edge via openrouter' for storage
        in provider_used so the frontend can show exactly which model ran.
        """
        pname = provider.provider_name
        if pname == "openrouter":
            if task in ("scenes", "vision", "scene_analysis"):
                model = getattr(provider, '_primary_model', None)
            elif task in ("summary",):
                model = getattr(provider, '_summary_model', None) or getattr(provider, '_editorial_model', None)
            else:  # clips, seo, text
                model = getattr(provider, '_editorial_model', None)
            if model:
                return f"{model} via openrouter"
        elif pname == "ollama":
            if task in ("scenes", "vision", "scene_analysis"):
                model = getattr(provider, '_primary_model', None)
            else:
                model = getattr(provider, '_editorial_model', None)
            if model:
                label = f"{model} via ollama"
                # Multi-host registry: name WHICH machine served the stage
                # (e.g. "… via ollama @ Desktop 4070") so failovers are
                # visible in the job's provider/timing display.
                try:
                    if (getattr(settings, "OLLAMA_HOSTS", "") or "").strip():
                        host_label = getattr(provider, "active_host_label", "")
                        if host_label:
                            label += f" @ {host_label}"
                except Exception:
                    pass
                return label
        return pname

    # Rough cost per 1K tokens by provider (input+output blended average)
    _COST_PER_1K_TOKENS = {
        "openrouter": 0.0002,   # varies by model; free tier = 0
        "anthropic": 0.006,     # Claude Sonnet ~$3/$15 per M tokens blended
        "gemini": 0.0003,       # Gemini Flash is very cheap
        "groq": 0.0001,         # Groq is very cheap
        "ollama": 0.0,          # local, no cost
    }

    async def validate_models(self, job_id: str) -> list[str]:
        """Pre-flight check: verify that EVERY constructed provider is
        reachable, in parallel.

        Previously only the first provider in the chain was validated,
        which meant a degraded primary (e.g. Ollama down) hid the fact
        that the user's OpenRouter key was actually fine. This made
        it look like the whole pipeline was broken when only the first
        link was. We now run a fan-out check so the user gets one
        line of feedback per provider in the chain.

        Returns a list of warning messages for the frontend log. Does
        NOT fail the pipeline — warnings are informational.
        """
        self._wire_ws_to_providers(job_id)
        warnings: list[str] = []
        if not self._providers:
            warnings.append("No AI providers configured — check Settings")
            return warnings

        async def _validate_one(pname: str, provider) -> tuple[str, str | None]:
            """Returns (status_message, warning_or_None) for one provider."""
            model_info = self._get_model_info(provider)
            try:
                if pname == "openrouter":
                    if not settings.OPENROUTER_API_KEY or settings.OPENROUTER_API_KEY in {"", "sk-or-..."}:
                        return ("", "⚠ OpenRouter API key is not set — cloud models will fail")
                    # Probe the TEXT model. Vision-model id is also
                    # validated below so we don't burn 100 batches on a
                    # bad ``OPENROUTER_PRIMARY_MODEL`` (e.g. "gpt-4o"
                    # instead of "openai/gpt-4o").
                    try:
                        await asyncio.wait_for(
                            provider.text_complete("Reply with OK", max_tokens=5, timeout=15),
                            timeout=20,
                        )
                    except asyncio.TimeoutError:
                        return ("", "⚠ OpenRouter text model timed out — may be overloaded. Will retry with fallbacks.")
                    except Exception as e:
                        return ("", f"⚠ OpenRouter text model check failed: {str(e)[:150]}")
                    # ── Vision-model existence check ──
                    # Hits OpenRouter's /api/v1/models (cached) and
                    # confirms the configured vision model is in the
                    # catalog. Quietly skipped on network failure so
                    # validate_models never becomes the reason a job
                    # can't start.
                    try:
                        vision_id = getattr(provider, "_primary_model", None)
                        if vision_id:
                            ok = await _openrouter_model_exists(vision_id)
                            if ok is False:
                                return (
                                    "",
                                    (
                                        f"\u26a0 OpenRouter vision model {vision_id!r} is "
                                        "not in the OpenRouter catalog. Scenes will be empty. "
                                        "Verify Settings \u2192 AI Provider \u2192 Vision."
                                    ),
                                )
                    except Exception:
                        pass
                    return (f"✓ {model_info} — models verified", None)

                elif pname == "ollama":
                    import httpx
                    from backend.services import ollama_registry
                    # Walk the registry in priority order — a dead primary
                    # with a live fallback is a WARNING, not a failure (the
                    # provider fails over per-request).
                    active = await ollama_registry.pick_host()
                    if active is None:
                        return ("", "⚠ No Ollama host is reachable "
                                    "(all registry hosts offline)")
                    _hosts = ollama_registry.enabled_hosts()
                    _host_note = ""
                    if _hosts and active.id != _hosts[0].id:
                        _host_note = (f" — primary '{_hosts[0].name}' is offline; "
                                      f"using fallback '{active.name}'")
                    # ── Vision-model existence check ──
                    # ``/api/show`` returns 404 if the model isn't
                    # pulled. Avoids 30s of batch failures on first
                    # use of a model the user hasn't downloaded.
                    try:
                        vision_id = getattr(provider, "_primary_model", None)
                        if vision_id:
                            # Ollama's /api/show uses the ``model`` key and
                            # expects a bare tag (no ``ollama/`` prefix).
                            bare_id = vision_id
                            while isinstance(bare_id, str) and bare_id.lower().startswith("ollama/"):
                                bare_id = bare_id[len("ollama/"):]
                            async with httpx.AsyncClient(timeout=8.0) as client:
                                ping = await client.post(
                                    ollama_registry.join_url(active.url, "/api/show"),
                                    json={"model": bare_id},
                                    headers=ollama_registry.auth_headers(active),
                                )
                                if ping.status_code == 404:
                                    return (
                                        "",
                                        (
                                            f"\u26a0 Ollama vision model {bare_id!r} is not "
                                            f"pulled. Run `ollama pull {bare_id}` or pick a "
                                            "model that's already downloaded."
                                        ),
                                    )
                    except Exception:
                        pass
                    return (f"✓ {model_info} — Ollama connected "
                            f"(host: {active.name}){_host_note}", None)

                elif pname == "gemini":
                    if not settings.GEMINI_API_KEY:
                        return ("", "⚠ Gemini API key is not set")
                    return (f"✓ {model_info} — key present", None)

                elif pname == "anthropic":
                    if not settings.ANTHROPIC_API_KEY:
                        return ("", "⚠ Anthropic API key is not set")
                    return (f"✓ {model_info} — key present", None)

                elif pname == "groq":
                    if not settings.GROQ_API_KEY:
                        return ("", "⚠ Groq API key is not set")
                    return (f"✓ {model_info} — key present", None)
            except Exception as e:
                return ("", f"⚠ {pname} validation crashed: {str(e)[:150]}")
            return (f"Using {model_info}", None)

        # Notify frontend which models we're about to validate.
        for pname, provider in self._providers.items():
            await self._notify_attempt(job_id, provider, "model validation")

        results = await asyncio.gather(
            *(_validate_one(pname, p) for pname, p in self._providers.items()),
            return_exceptions=False,
        )
        for status_msg, warn_msg in results:
            if status_msg and self._ws_broadcast:
                try:
                    await self._ws_broadcast(job_id, {
                        "type": "status", "message": status_msg,
                    })
                except Exception:
                    pass
            if warn_msg:
                warnings.append(warn_msg)
                if self._ws_broadcast:
                    try:
                        await self._ws_broadcast(job_id, {
                            "type": "status", "message": warn_msg,
                        })
                    except Exception:
                        pass
        return warnings

    def get_total_tokens(self) -> int:
        """Return total tokens used across all provider instances."""
        return sum(p.total_tokens for p in self._providers.values())

    def estimate_cost(self) -> float:
        """Estimate total cost in USD from all provider token usage."""
        total = 0.0
        for name, provider in self._providers.items():
            tokens = provider.total_tokens
            if tokens > 0:
                rate = self._COST_PER_1K_TOKENS.get(name, 0.001)
                total += (tokens / 1000) * rate
        return round(total, 6)

    async def unload_local_models(self):
        """Unload Ollama models from VRAM so GPU is free for other tasks."""
        ollama = self._providers.get("ollama")
        if ollama and hasattr(ollama, "unload_models"):
            await ollama.unload_models()

    def reset_circuit_breaker(self):
        """Reset circuit breaker state before critical pipeline operations.

        Call this before summary generation and clip detection to ensure
        that failures from non-critical steps (transcript correction)
        don't block the core analysis pipeline.
        """
        self._circuit_breaker.force_reset_all()

    def _mark_if_key_limited(self, name: str, exc) -> None:
        """Disable ``name`` for the rest of the job when ``exc`` is a hard
        billing/key-limit error, so we stop re-trying a dead key every stage
        (one warning per provider)."""
        if name not in self._billing_dead and _is_key_limit_error(exc):
            self._billing_dead.add(name)
            logger.warning(
                "Provider '%s' hit a hard key/billing limit (403 'Key limit "
                "exceeded') — disabling it for the rest of this job to stop "
                "re-trying a dead key every stage. Fix the key's credit limit / "
                "add credits, or run local (SELF_HOSTED_MODE / "
                "EDITORIAL_AI_SOURCE=local).", name)

    def get_editorial_model_info(self) -> dict:
        """Return info about the text model that will handle the next text_completion call.

        Used by transcript correction to:
        1. Log which model is polishing the transcript
        2. Set an appropriate timeout based on model type (thinking vs standard)
        """
        chain = self._get_active_chain()
        if not chain:
            return {"provider": "none", "model": "none", "is_thinking": False}
        provider = chain[0]
        return {
            "provider": provider.provider_name,
            "model": provider.text_model_name,
            "is_thinking": provider.is_thinking_model,
        }

    def _get_active_chain(self) -> list[AIProvider]:
        chain = []
        skipped = []
        unreachable = []
        for name in settings.editorial_provider_chain:
            if name in self._unreachable:
                unreachable.append(name)
                continue
            if name in self._billing_dead:
                skipped.append(f"{name} (key-limited)")
                continue
            if name in self._providers and not self._circuit_breaker.is_degraded(name):
                chain.append(self._providers[name])
            elif name in self._providers:
                skipped.append(name)
        chain_names = [p.provider_name for p in chain]
        if skipped or unreachable:
            logger.info(
                "Active provider chain: %s (degraded: %s, unreachable: %s)",
                chain_names, skipped, unreachable,
            )
        else:
            logger.debug("Active provider chain: %s", chain_names)
        return chain

    async def _notify_attempt(self, job_id: str, provider, task: str):
        """Notify the frontend that a provider is being tried for a task.
        `provider` can be a string or an AIProvider instance."""
        if isinstance(provider, str):
            label = provider
        else:
            label = self._get_model_info(provider)
        logger.info("Attempting %s via %s", task, label)
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(job_id, {
                    "type": "status",
                    "message": f"Attempting {task} via {label}...",
                })
            except Exception:
                pass

    async def _notify_fallback(self, job_id: str, from_provider: str, reason: str):
        reason_short = reason[:200] if len(reason) > 200 else reason
        logger.info("Falling back from %s: %s", from_provider, reason_short)
        if self._ws_broadcast:
            try:
                await self._ws_broadcast(job_id, {
                    "type": "fallback",
                    "from_provider": from_provider,
                    "to_provider": "next in chain",
                    "reason": reason_short,
                })
            except Exception:
                pass

    async def analyze_frames(
        self, frames: list[FrameData], job_id: str,
        progress_callback=None,
        content_type=None,
    ) -> tuple[list[SceneDescription], str]:
        """Returns (results, provider_name_used).
        progress_callback(frames_done, total_frames, provider_name) is called per batch.

        ``content_type`` is an optional ``ClipContentType`` (or
        string / None). When provided, the OpenRouter provider uses
        it to route ANIME / GAMEPLAY jobs to Qwen3-VL and leaves
        every other content type on the preset default. See
        ``select_primary_model_for_content`` in openrouter_provider.py.
        """
        self._wire_ws_to_providers(job_id)
        frame_prompt = self._custom_prompts.frame_analysis if self._custom_prompts else None
        for provider in self._get_active_chain():
            if not provider.supports_vision:
                continue
            try:
                await self._notify_attempt(job_id, provider, f"scene analysis ({len(frames)} frames)")

                async def _provider_progress(done, total):
                    if progress_callback:
                        await progress_callback(done, total, provider.provider_name)

                # Phase 2 — content-type-aware vision routing. Only
                # OpenRouter ships the preset system, so this is a
                # targeted call; other providers ignore the kwarg.
                _routing_ct = content_type
                try:
                    if hasattr(provider, "apply_primary_model_override"):
                        provider.apply_primary_model_override(_routing_ct)
                except Exception as err:
                    logger.debug("Vision model override skipped: %s", err)

                # On a budget-limited remote card, take the vision turn on the
                # GPU for the whole frame batch — interleaving vision with the
                # concurrent translation loop's text calls CPU-spills the big
                # text model (the observed 21-minute brown-out). Same-class
                # batches still run concurrently; only text-vs-vision switches
                # serialize.
                _vlease_held = False
                if (provider.provider_name == "ollama"
                        and await self._gpu_class_serialize_needed(provider)):
                    logger.info(
                        "Scene analysis waiting for its GPU turn (a text phase "
                        "holds the remote card)")
                    await self._gpu_class_lease.acquire("vision")
                    _vlease_held = True
                try:
                    t0 = time.monotonic()
                    result = await provider.analyze_frames(
                        frames, custom_prompt=frame_prompt,
                        cancel_check=self._cancel_check,
                        progress_callback=_provider_progress,
                    )
                    elapsed = time.monotonic() - t0
                finally:
                    if _vlease_held:
                        await self._gpu_class_lease.release()
                logger.info("Scene analysis via %s completed in %.1fs (%d scenes)", provider.provider_name, elapsed, len(result))
                # Log subject tracking distribution for debugging
                if result:
                    sx_values = [s.subject_x for s in result]
                    sx_unique = len(set(sx_values))
                    all_default = all(v == 50 for v in sx_values)
                    logger.info(
                        "Subject tracking via %s: %d scenes, %d unique subject_x values, range [%d, %d], mean=%.1f%s",
                        provider.provider_name, len(sx_values), sx_unique,
                        min(sx_values), max(sx_values),
                        sum(sx_values) / len(sx_values),
                        " ⚠ ALL VALUES ARE 50 — model may not have detected subject positions" if all_default else "",
                    )
                self._circuit_breaker.record_success(provider.provider_name)
                return result, self._get_task_model(provider, "scenes")
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(provider.provider_name)
                self._mark_if_key_limited(provider.provider_name, e)
                await self._notify_fallback(job_id, provider.provider_name, str(e))
                continue
        raise AllProvidersFailedError("All vision providers failed")

    async def generate_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        job_id: str,
        tier=None,
        output_language: str = "",
    ) -> tuple[VideoSummary, str]:
        """Returns (summary, provider_name_used).

        If tier specifies map_reduce strategy, splits transcript into chunks,
        summarizes each, then merges. This ensures long videos get full coverage.
        """
        # Callers sometimes pass dicts (e.g., transcript / scenes loaded from
        # the DB before Pydantic re-parse). Re-hydrate so downstream code can
        # use attribute access (s.start, s.timestamp, …) without "'dict'
        # object has no attribute 'start'" crashes.
        transcript = [TranscriptSegment(**s) if isinstance(s, dict) else s for s in transcript]
        scenes = [SceneDescription(**s) if isinstance(s, dict) else s for s in scenes]
        if tier and tier.summary_strategy == "map_reduce" and tier.summary_chunk_minutes > 0:
            return await self._map_reduce_summary(
                transcript, scenes, job_id, tier, output_language=output_language)

        # Force the summary into the user's subtitle language (so it matches the
        # clips), translating the source transcript's understanding as needed —
        # instead of the old hardcoded-English default.
        from backend.services.prompts import (
            summary_language_directive, DEFAULT_SUMMARY_PROMPT,
        )
        _base = (self._custom_prompts.summary if self._custom_prompts else None) \
            or DEFAULT_SUMMARY_PROMPT
        summary_prompt = summary_language_directive(output_language) + _base
        for provider in self._get_active_chain():
            try:
                await self._notify_attempt(job_id, provider, "summary generation")
                t0 = time.monotonic()
                result = await provider.generate_summary(transcript, scenes, cancel_check=self._cancel_check, custom_prompt=summary_prompt)
                elapsed = time.monotonic() - t0
                logger.info("Summary generation via %s completed in %.1fs", provider.provider_name, elapsed)
                self._circuit_breaker.record_success(provider.provider_name)
                return result, self._get_task_model(provider, "summary")
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(provider.provider_name)
                self._mark_if_key_limited(provider.provider_name, e)
                await self._notify_fallback(job_id, provider.provider_name, str(e))
                continue
        raise AllProvidersFailedError("All providers failed for summary generation")

    async def _map_reduce_summary(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        job_id: str,
        tier,
        output_language: str = "",
    ) -> tuple[VideoSummary, str]:
        """Hierarchical map-reduce summary for long videos.

        Map: Split transcript into N-minute chunks, generate mini-summary per chunk.
        Reduce: Feed all mini-summaries into a final summary call.
        """
        from backend.services.providers.base import extract_json, has_real_summary_content
        from backend.services.prompts import summary_language_directive
        _lang_dir = summary_language_directive(output_language)

        chunk_seconds = tier.summary_chunk_minutes * 60
        if not transcript:
            from backend.services.providers.base import build_summary_from_transcript
            fb = build_summary_from_transcript(transcript, scenes)
            return VideoSummary(**fb), "fallback"

        video_end = max(s.end for s in transcript)

        # ── Cap the chunk COUNT on long videos ──
        # Each map chunk is one (sequential, on Ollama) LLM call; a 128-min
        # video at 5-min chunks is ~21 calls ≈ 19 min. Enlarge the chunk span
        # so the count stays under SUMMARY_MAX_CHUNKS — this only ever makes
        # chunks BIGGER (never more granular than the tier), trading some
        # summary granularity for far fewer LLM round trips. 0 disables the cap.
        _max_chunks = int(getattr(settings, "SUMMARY_MAX_CHUNKS", 12) or 0)
        if _max_chunks > 0 and chunk_seconds > 0 and video_end > 0:
            import math as _math
            _would = _math.ceil(video_end / chunk_seconds)
            if _would > _max_chunks:
                chunk_seconds = _math.ceil(video_end / _max_chunks)
                logger.info(
                    "[%s] Summary: %d chunks would exceed cap %d — enlarging chunk "
                    "span to %.0fs (~%d chunks)",
                    job_id, _would, _max_chunks, chunk_seconds,
                    _math.ceil(video_end / chunk_seconds))
        chunk_overlap = 30.0  # 30 second overlap between chunks
        chunks: list[tuple[float, float, list[TranscriptSegment], list[SceneDescription]]] = []
        t = 0.0
        while t < video_end:
            chunk_end = min(t + chunk_seconds, video_end)
            # Extend segment selection by overlap into adjacent chunks
            chunk_segs = [s for s in transcript if s.start >= t - chunk_overlap and s.end <= chunk_end + chunk_overlap]
            chunk_scenes = [s for s in scenes if t - chunk_overlap <= s.timestamp <= chunk_end + chunk_overlap]
            chunks.append((t, chunk_end, chunk_segs, chunk_scenes))
            t = chunk_end

        chain = self._get_active_chain()
        is_ollama = chain and chain[0].provider_name == "ollama"
        max_segs = 20 if is_ollama else 50
        chunk_timeout = 90 if is_ollama else 60
        max_chunk_tokens = 300 if is_ollama else 500

        # Companion-aware model upgrade (mirror of the clip-judge upgrade): on
        # a rig whose remote card holds the big translation model, run the
        # summary on IT rather than the small editorial default. A real run's
        # overview and key topics went sparse because all five map chunks and
        # the reduce ran on qwen2.5:3b in 14 seconds flat, while gemma3:12b
        # sat configured for translation on the same GPU.
        _big_override = None
        if is_ollama:
            try:
                from backend.services import ollama_registry as _oreg_s
                if await _oreg_s.remote_primary_vram_gb_resolved() >= 7.0:
                    _cand = str(getattr(
                        settings, "OLLAMA_TRANSLATION_MODEL", "") or "").strip()
                    _cur = str(getattr(chain[0], "_editorial_model", "") or "")
                    if _cand and _est_model_gb(_cand) > _est_model_gb(_cur):
                        _big_override = _cand
                        logger.info(
                            "[%s] Summary upgraded to the rig's large model %s "
                            "(Companion GPU)", job_id, _cand)
            except Exception:
                _big_override = None

        logger.info(
            "[%s] Map-reduce summary: %d chunks (%.0fs each), ollama=%s",
            job_id, len(chunks), chunk_seconds, is_ollama,
        )

        # Map-chunk concurrency ladder. Chunks are independent, disjoint
        # transcript spans, so running them in parallel is quality-neutral.
        # A local small card must stay serial (VRAM), but a high-VRAM remote
        # Companion handles 2-3 in flight; cloud providers take 6.
        if is_ollama:
            _conc = 1
            try:
                from backend.services import ollama_registry
                _vram_gb = ollama_registry.remote_primary_vram_gb()
                if _vram_gb >= 10:
                    _conc = 3
                elif _vram_gb >= 7:
                    _conc = 2
            except Exception:
                _conc = 1  # local/unknown card: keep the serial legacy behavior
        else:
            _conc = 6
        sem = asyncio.Semaphore(_conc)

        async def _summarize_chunk(idx, start, end, segs, scns):
            async with sem:
                time_label = f"{int(start//60)}:{int(start%60):02d}-{int(end//60)}:{int(end%60):02d}"
                # Even-stride sample when a (possibly enlarged) chunk holds more
                # than max_segs cues, so the summary spans the WHOLE chunk instead
                # of only its opening minutes (head-truncation would drop the back
                # half of a big chunk entirely).
                if len(segs) > max_segs:
                    _step = len(segs) / max_segs
                    _picked = [segs[min(int(k * _step), len(segs) - 1)] for k in range(max_segs)]
                else:
                    _picked = segs
                text = "\n".join(
                    f"[{s.start:.0f}s] {s.speaker}: {s.text}"
                    for s in _picked
                )
                scene_text = "\n".join(
                    f"[{s.timestamp:.0f}s] {s.description[:60 if is_ollama else 100]}"
                    for s in scns[:5 if is_ollama else 10]
                )
                prompt = (
                    _lang_dir +
                    f"Summarize this {time_label} segment in 2-3 sentences. "
                    f"Include: main topic, key content, notable moments. "
                    f"Do not reference or speculate about speakers.\n\n"
                    f"TRANSCRIPT:\n{text}\n\nSCENES:\n{scene_text}\n\n"
                    f"Return a plain text summary (no JSON)."
                )
                try:
                    result = await self.text_completion(
                        prompt, max_tokens=max_chunk_tokens,
                        timeout=chunk_timeout,
                        job_id=job_id, skip_circuit_breaker=True,
                        model_override=_big_override,
                    )
                    # Broadcast chunk progress so the user sees activity
                    if self._ws_broadcast and job_id:
                        try:
                            await self._ws_broadcast(job_id, {
                                "type": "status",
                                "message": f"Summary: chunk {idx + 1}/{len(chunks)} complete...",
                            })
                        except Exception:
                            pass
                    return f"[{time_label}] {result.strip()}"
                except Exception as e:
                    logger.warning("[%s] Chunk %d summary failed: %s", job_id, idx, e)
                    return f"[{time_label}] {segs[0].text[:200] if segs else 'No content'}"

        results = await asyncio.gather(*[
            _summarize_chunk(i, s, e, segs, scns)
            for i, (s, e, segs, scns) in enumerate(chunks)
        ])
        mini_summaries = [r for r in results if r]

        # Reduce phase
        combined = "\n".join(mini_summaries)
        if is_ollama and len(combined) > 2500:
            # Even coverage instead of head-truncation: head-truncating at 2500
            # chars silently dropped the back half of long videos from the
            # reduce. Keep mini-summaries evenly strided across the WHOLE list
            # (mirroring the map phase's even-stride cue sampling above) so the
            # reduce still sees the video's end, while staying under the same
            # 2500-char cap that the small local ctx requires. Chronological
            # order of what's kept is preserved.
            for _keep in range(len(mini_summaries) - 1, 0, -1):
                _stride = len(mini_summaries) / _keep
                _idxs = sorted({min(int(k * _stride), len(mini_summaries) - 1)
                                for k in range(_keep)})
                _cand = "\n".join(mini_summaries[i] for i in _idxs)
                if len(_cand) <= 2500:
                    combined = _cand
                    break
            else:
                # Even a single mini-summary exceeds the cap — hard-truncate it.
                combined = combined[:2500]

        reduce_prompt = (
            _lang_dir +
            f"You have segment-by-segment summaries of a video. "
            f"Combine them into a cohesive summary. "
            f"Do not reference or speculate about speakers unless names are explicitly mentioned. "
            f"Focus on what is discussed, shown, and the key moments.\n\n"
            f"SEGMENT SUMMARIES:\n{combined}\n\n"
            "Return ONLY valid JSON:\n"
            '{"overview": "<2-4 sentence paragraph about the video content>", '
            '"key_topics": ["topic1", "topic2", "topic3"], '
            '"tone": "<1-2 words>", "estimated_audience": "<who would watch>", '
            '"content_category": "<specific category>"}\n'
            "key_topics MUST contain 3-6 specific topics from the video."
        )

        # Grammar-level schema so the model CANNOT omit a required field — a
        # real run produced a rich, correct summary and lost ALL of it to a
        # missing "content_category" (pydantic rejected the dict, the stage
        # fell through two fallbacks, and the UI showed a template overview
        # with face-diagnostic "topics").
        _reduce_schema = {
            "type": "object",
            "required": ["overview", "key_topics", "tone",
                         "estimated_audience", "content_category"],
            "properties": {
                "overview": {"type": "string"},
                "key_topics": {"type": "array", "items": {"type": "string"},
                               "minItems": 3, "maxItems": 6},
                "tone": {"type": "string"},
                "estimated_audience": {"type": "string"},
                "content_category": {"type": "string"},
            },
        }
        for provider in chain:
            try:
                _kw = {}
                if getattr(provider, "provider_name", "") in ("ollama", "openrouter"):
                    _kw["json_schema"] = _reduce_schema
                # Reduce on the upgraded model too — the reduce writes the
                # overview/topics the user actually reads.
                _orig_em = None
                if (_big_override
                        and getattr(provider, "provider_name", "") == "ollama"
                        and hasattr(provider, "_editorial_model")):
                    _orig_em = provider._editorial_model
                    provider._editorial_model = _big_override
                try:
                    raw = await asyncio.wait_for(
                        provider.text_complete(reduce_prompt, max_tokens=1000 if is_ollama else 2000, **_kw),
                        timeout=(240 if _big_override else 120) if is_ollama else 90,
                    )
                finally:
                    if _orig_em is not None:
                        provider._editorial_model = _orig_em
                data = extract_json(raw)
                if has_real_summary_content(data):
                    # Coercion backstop: a summary with real content must
                    # never be discarded over a missing enum-ish field —
                    # fill the trivia with defaults and keep the content.
                    try:
                        return VideoSummary(**data), self._get_task_model(provider, "summary")
                    except Exception:
                        _merged = {
                            "tone": "conversational",
                            "estimated_audience": "general viewers",
                            "content_category": "video content",
                            "key_topics": [],
                        }
                        _merged.update({k: v for k, v in (data or {}).items()
                                        if v not in (None, "")})
                        return (VideoSummary(**_merged),
                                self._get_task_model(provider, "summary"))
            except Exception as e:
                logger.warning("[%s] Reduce summary via %s failed: %s", job_id, provider.provider_name, e)
                continue

        # Fallback: stitch the mini-summaries into a cohesive overview. The
        # reduce JSON contract failed, but the per-chunk summaries are plain
        # prose the small model CAN produce — strip the "[m:ss-m:ss]" time tags
        # and join several so the overview reads as a paragraph, not one chunk.
        import re as _re
        _clean = [_re.sub(r"^\[[0-9:.\-\s]+\]\s*", "", s).strip() for s in mini_summaries]
        _clean = [s for s in _clean if s]
        overview = " ".join(_clean[:8])
        if len(overview) > 900:
            overview = overview[:900].rsplit(" ", 1)[0] + "..."
        # Mine content topics from the chunk summaries (recurring TitleCase
        # terms) — an EMPTY key_topics list fails the pipeline's
        # has_real_summary_content gate and drops this perfectly usable
        # overview for the template fallback (whose "topics" are reframer
        # diagnostics like "adaptive face — 2 face(s)").
        _tc: dict = {}
        for _ms in _clean:
            for _tm in _re.findall(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})?\b", _ms):
                if _tm.split()[0].lower() in ("the", "this", "that", "there",
                                              "then", "with", "from"):
                    continue
                _tc[_tm] = _tc.get(_tm, 0) + 1
        _topics = [t for t, c in sorted(_tc.items(), key=lambda kv: (-kv[1], kv[0]))
                   if c >= 2][:5]
        return VideoSummary(
            overview=overview or " ".join(mini_summaries[:6]),
            key_topics=_topics,
            tone="conversational",
            estimated_audience="general viewers",
            content_category="video content",
        ), "map_reduce_fallback"

    async def detect_viral_clips(
        self,
        transcript: list[TranscriptSegment],
        scenes: list[SceneDescription],
        video_duration: float,
        job_id: str,
        clip_count: Optional[int] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        clip_focus: Optional[str] = None,
        video_summary: Optional[str] = None,
        existing_clips: Optional[str] = None,
        hot_zones=None,
        progress_callback=None,
        tier=None,
        content_type=None,
        chapters=None,
        trend_context: Optional[str] = None,
        sentiment_timeline: Optional[str] = None,
        viral_score_min: int = 0,
        viral_score_max: int = 100,
        min_relevance: int = 0,
    ) -> tuple[list[ClipCandidate], str]:
        """Returns (clips, provider_name_used).

        ``content_type`` (Phase 2) is a ``ClipContentType`` enum value
        used to pick the genre-specific viral-clip prompt. ``chapters``
        (Phase 6), ``trend_context`` (Phase 3), and ``sentiment_timeline``
        (Phase 5) are optional context blocks injected into the prompt
        — providers that don't know what to do with them ignore them.
        """
        # Phase 2 — pick the genre-specific prompt unless the user has
        # an explicit override in custom_prompts. The user's custom
        # prompt always wins so they can fine-tune per deployment.
        custom_user_prompt = self._custom_prompts.viral_clip_detection if self._custom_prompts else None
        # ``PromptSet`` defaults to ``DEFAULT_VIRAL_CLIP_PROMPT``, so a
        # vanilla custom_prompts object would shadow the genre routing.
        # Detect the default-equals case and fall through to genre.
        from backend.services.prompts import (
            DEFAULT_VIRAL_CLIP_PROMPT, get_genre_prompt,
        )
        from backend.services.compat_stubs import (
            finalize_clip_scores, four_axis_scoring_enabled,
        )
        if custom_user_prompt and custom_user_prompt != DEFAULT_VIRAL_CLIP_PROMPT:
            clip_prompt = custom_user_prompt
            logger.info("Clip detection: using user custom prompt")
        else:
            clip_prompt = get_genre_prompt(content_type)
            ct_label = content_type.value if hasattr(content_type, "value") else (content_type or "generic")
            logger.info("Using genre prompt: %s", ct_label)
        # If clip_focus is provided, build an augmented focus prompt that
        # BUILDS ON the viral detection infrastructure rather than replacing it.
        # Enhancement 4 — the focus field accepts a comma- or newline-separated
        # list of queries which the LLM is instructed to treat as OR; each
        # returned clip's clip_focus field names the matching sub-query.
        if clip_focus and clip_focus.strip():
            focus_text = clip_focus.strip()
            focus_sub_queries = [
                q.strip() for q in re.split(r"[,\n]+", focus_text)
                if q.strip()
            ]
            is_multi_query = len(focus_sub_queries) > 1
            if is_multi_query:
                sub_bullets = "\n".join(f"  {i+1}. \"{q}\"" for i, q in enumerate(focus_sub_queries))
                clip_prompt = (
                    f"You are finding clips in a video that match ANY of several user-requested topics.\n\n"
                    f"USER'S FOCUS QUERIES (OR):\n{sub_bullets}\n\n"
                    f"For each clip, set clip_focus to the EXACT sub-query string it matched "
                    f"(one of the options above). If a clip strongly matches multiple sub-queries, "
                    f"pick the best match and note the overlap in viral_score_reasoning.\n\n"
                    f"SEMANTIC EXPANSION — expand each sub-query into synonyms and related concepts "
                    f"before searching. For example, \"fighting\" also covers combat, battle, "
                    f"argument, confrontation, sparring.\n\n"
                    f"RELEVANCE TIERS (apply per sub-query the clip matched):\n"
                    f"  Tier 1 (STRONG — score 80-100): the matched sub-query is the main subject.\n"
                    f"  Tier 2 (MODERATE — score 50-79): the matched sub-query is a significant part.\n"
                    f"  Tier 3 (WEAK — score 20-49): passing mention only (include only if <3 stronger).\n"
                    f"  EXCLUDE: mere keyword hits in passing or negations.\n\n"
                    f"SCORING: Put RELEVANCE (1-100) in 'viral_score' — do NOT use virality here. "
                    f"Also include 'focus_relevance' (1-100) and 'focus_tier' "
                    f"(\"strong\", \"moderate\", or \"weak\").\n\n"
                    f"SCENE & SUBJECT COHERENCE (CRITICAL):\n"
                    f"- The main subject MUST stay in focus throughout each clip\n"
                    f"- Keep the clip within ONE scene / exchange\n"
                    f"- Prefer natural speech boundaries for start/end\n\n"
                    f"BOUNDARY RULES:\n"
                    f"- Start at the beginning of a sentence\n"
                    f"- End at a clean exit point\n"
                    f"- Clips must work standalone without the rest of the video"
                )
                logger.info(
                    "Clip focus MULTI-QUERY mode for job %s: %d sub-queries",
                    job_id, len(focus_sub_queries),
                )
            else:
                clip_prompt = (
                    f"You are finding clips in a video that focus on a specific user-requested topic.\n\n"
                    f"USER'S FOCUS QUERY: \"{focus_text}\"\n\n"
                    f"SEMANTIC EXPANSION — Before searching, expand this query into related concepts:\n"
                    f"Think about synonyms, related terms, sub-topics, and adjacent concepts that someone "
                    f"searching for \"{focus_text}\" would also want to see. For example, if the focus is "
                    f"'fighting', also look for: combat, battle, argument, confrontation, sparring, conflict, "
                    f"physical altercation, self-defense, martial arts, etc.\n\n"
                    f"RELEVANCE TIERS:\n"
                    f"  Tier 1 (STRONG — score 80-100): The segment IS ABOUT '{focus_text}'. "
                    f"The topic is the main subject of discussion or the primary visual action.\n"
                    f"  Tier 2 (MODERATE — score 50-79): The segment discusses '{focus_text}' as a "
                    f"significant part of a broader conversation. Multiple sentences or visual moments relate to it.\n"
                    f"  Tier 3 (WEAK — score 20-49): The topic is mentioned briefly or tangentially. "
                    f"Only include Tier 3 clips if fewer than 3 Tier 1/2 clips exist.\n"
                    f"  EXCLUDE: Segments that merely mention a word related to '{focus_text}' in passing, "
                    f"negations ('I don't like {focus_text}'), or purely metaphorical usage.\n\n"
                    f"COMPOUND QUERIES: If the focus contains both a topic and a mood/quality "
                    f"(e.g., 'funny cooking moments'), prioritize segments matching BOTH aspects. "
                    f"Score clips higher when they combine the topic with the specified mood.\n\n"
                    f"SCORING: Use 'viral_score' to represent RELEVANCE to '{focus_text}' (not virality). "
                    f"A clip with 90 relevance means the segment is deeply, directly about the focus topic. "
                    f"In 'viral_score_reasoning', explain WHY this clip matches the focus query and which "
                    f"relevance tier it falls into.\n\n"
                    f"Additionally include 'focus_relevance' (1-100) and 'focus_tier' (\"strong\", \"moderate\", "
                    f"or \"weak\") in each clip's JSON.\n\n"
                    f"SCENE & SUBJECT COHERENCE (CRITICAL):\n"
                    f"- The main subject MUST stay in focus throughout the entire clip\n"
                    f"- NEVER cut across unrelated scenes or topics — the clip must feel like ONE moment\n"
                    f"- If a clip covers a conversation, keep it within the same exchange\n"
                    f"- The visual setting should remain consistent — don't span across location changes\n"
                    f"- Prefer segments where the camera stays on the main action without jarring cuts\n"
                    f"- If scene descriptions show different settings at different timestamps, do NOT combine them into one clip\n\n"
                    f"BOUNDARY RULES:\n"
                    f"- Start at natural speech boundaries — beginning of a sentence, after a pause, at a speaker change\n"
                    f"- End at natural conclusions — even if focus content extends further, find a clean exit point\n"
                    f"- Must work standalone without context from the full video\n"
                    f"- Prefer clips where the focus topic is introduced within the first 5 seconds"
                )
                logger.info("Clip focus mode active for job %s: '%s'", job_id, focus_text)

        # Bug 2 (clip-focus audit): inject score range and min-relevance
        # as *prompt* constraints so the LLM doesn't waste tokens
        # generating candidates we'll throw away. The post-hoc filter in
        # backend/routers/clips.py still runs as a safety net.
        is_focus_mode = bool(clip_focus and clip_focus.strip())
        try:
            viral_score_min = max(0, min(100, int(viral_score_min)))
            viral_score_max = max(0, min(100, int(viral_score_max)))
            min_relevance = max(0, min(100, int(min_relevance)))
        except (TypeError, ValueError):
            viral_score_min, viral_score_max, min_relevance = 0, 100, 0
        if viral_score_min > 0 or viral_score_max < 100:
            clip_prompt = clip_prompt + (
                f"\n\nSCORE RANGE HARD CONSTRAINT: only return clips whose composite "
                f"virality score falls inside [{viral_score_min}, {viral_score_max}]. "
                f"If you would score a clip below {viral_score_min} or above {viral_score_max}, "
                f"skip it entirely — DO NOT pad the clip count with borderline or off-target clips."
            )
        if is_focus_mode and min_relevance > 0:
            clip_prompt = clip_prompt + (
                f"\n\nRELEVANCE FLOOR: only return clips whose focus_relevance is "
                f">= {min_relevance}. If fewer than the requested count qualify, return "
                f"fewer clips — do not include off-topic filler."
            )

        # Phase 3 / 5 / 6 — append optional context blocks the LLM should
        # use for the four-axis scoring. Each block is fenced so the LLM
        # can clearly tell where the rubric ends and the data begins.
        appended_blocks: list[str] = []
        if trend_context:
            appended_blocks.append(
                "TREND CONTEXT (use this to populate trend_score):\n"
                + trend_context.strip()
            )
        if sentiment_timeline:
            appended_blocks.append(
                "SENTIMENT TIMELINE (audio sentiment moments — use for hook_score "
                "and value_score):\n" + sentiment_timeline.strip()
            )
        if chapters:
            try:
                chapter_lines = []
                for ch in chapters:
                    title = getattr(ch, "title", "") or ""
                    start = float(getattr(ch, "start", 0))
                    end = float(getattr(ch, "end", 0))
                    chapter_lines.append(
                        f"  [{start:.0f}-{end:.0f}s] {title}"
                    )
                if chapter_lines:
                    appended_blocks.append(
                        "CHAPTERS (the video has been pre-segmented into topic "
                        "chapters — try to find at least one strong clip per "
                        "chapter, do not cluster all clips in the first chapter):\n"
                        + "\n".join(chapter_lines)
                    )
            except Exception:
                pass
        if appended_blocks:
            clip_prompt = clip_prompt + "\n\n" + "\n\n".join(appended_blocks)

        # Per-provider timeout prevents any single provider from blocking the
        # fallback chain. Scale timeout based on video duration and provider type.
        vid_minutes = video_duration / 60 if video_duration else 0
        if vid_minutes > 30:
            # Windows processed 2 at a time (or sequentially for Ollama)
            est_windows = max(1, int(video_duration / 600) + 1)
            est_rounds = (est_windows + 1) // 2
            default_timeout = max(600, est_rounds * 330 + 300)
        else:
            default_timeout = 330

        # Shared partial results container. Providers populate this incrementally
        # so even if a timeout fires, we have whatever completed.
        _partial_clips: list = []

        for provider in self._get_active_chain():
            pname = provider.provider_name
            # Compute Ollama timeout: sequential windows need much more time
            if pname == "ollama":
                est_windows = max(1, int(video_duration / 300))
                timeout = max(420, est_windows * 300 + 120)
                logger.info("Ollama clip detection timeout: %ds (%d est. windows)", timeout, est_windows)
            else:
                timeout = default_timeout
            try:
                await self._notify_attempt(job_id, provider, "viral clip detection")
                t0 = time.monotonic()
                result = await asyncio.wait_for(
                    provider.detect_viral_clips(
                        transcript, scenes, video_duration,
                        custom_prompt=clip_prompt, cancel_check=self._cancel_check,
                        clip_count=clip_count, min_duration=min_duration,
                        max_duration=max_duration,
                        video_summary=video_summary,
                        existing_clips=existing_clips,
                        hot_zones=hot_zones,
                        progress_callback=progress_callback,
                        _partial_results=_partial_clips,
                        tier=tier,
                    ),
                    timeout=timeout,
                )
                elapsed = time.monotonic() - t0
                logger.info("Clip detection via %s completed in %.1fs (%d clips)", pname, elapsed, len(result))
                self._circuit_breaker.record_success(pname)
                # Phase 4 — recompose viral_score from the four axes
                # using the genre-aware weights. No-op when the four
                # axes are all zero (legacy clip path).
                if four_axis_scoring_enabled():
                    finalize_clip_scores(result, content_type, focus_mode=is_focus_mode)
                return result, self._get_task_model(provider, "clips")
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - t0
                # Check if partial results were collected before timeout
                if _partial_clips:
                    if hasattr(provider, '_deduplicate_clips'):
                        deduped = provider._deduplicate_clips(_partial_clips)
                    else:
                        deduped = _partial_clips
                    logger.warning(
                        "Clip detection via %s timed out after %ds but recovered %d partial clips",
                        pname, timeout, len(deduped),
                    )
                    if four_axis_scoring_enabled():
                        finalize_clip_scores(deduped, content_type, focus_mode=is_focus_mode)
                    return deduped, f"{self._get_task_model(provider, 'clips')} (partial)"
                logger.warning("Clip detection via %s timed out after %ds", pname, timeout)
                self._circuit_breaker.record_failure(pname)
                await self._notify_fallback(job_id, pname, f"Timed out after {timeout}s")
                continue
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(pname)
                await self._notify_fallback(job_id, pname, str(e))
                continue

        # Even if all providers "failed", check partial results
        if _partial_clips:
            logger.warning(
                "All providers failed but recovered %d partial clips", len(_partial_clips),
            )
            if four_axis_scoring_enabled():
                finalize_clip_scores(_partial_clips, content_type, focus_mode=is_focus_mode)
            return _partial_clips, "partial"
        raise AllProvidersFailedError("All providers failed for viral clip detection")

    async def _gpu_class_serialize_needed(self, provider) -> bool:
        """True when the remote card cannot hold the big text model and the
        vision model resident together, so text and vision phases must take
        turns on the GPU (see ``_GpuClassLease``). Memoized after first probe;
        False (current behavior) whenever the budget or sizes are unknown."""
        if self._gpu_lease_needed is not None:
            return self._gpu_lease_needed
        needed = False
        try:
            if not bool(getattr(settings, "OLLAMA_SERIALIZE_TEXT_VISION", True)):
                self._gpu_lease_needed = False
                return False
            from backend.services import ollama_registry as _oreg
            budget = float(await _oreg.remote_primary_vram_gb_resolved() or 0.0)
            if budget <= 0:
                # No remote budget known — local-only rigs already run the
                # serial pipeline, so there is nothing to serialize.
                self._gpu_lease_needed = False
                return False
            text_gb = max(
                _est_model_gb(getattr(provider, "_editorial_model", "") or ""),
                _est_model_gb(str(getattr(
                    settings, "OLLAMA_TRANSLATION_MODEL", "") or "")),
            )
            vision_gb = _est_model_gb(getattr(provider, "_primary_model", "") or "")
            if text_gb > 0 and vision_gb > 0:
                needed = (text_gb + vision_gb) > budget * 0.95
                if needed:
                    logger.info(
                        "GPU class lease ON: text ~%.1f GB + vision ~%.1f GB "
                        "exceed the %.1f GB remote budget — text and vision "
                        "phases take turns instead of CPU-spilling the text "
                        "model", text_gb, vision_gb, budget)
        except Exception:
            needed = False
        self._gpu_lease_needed = needed
        return needed

    async def _maybe_downgrade_ollama_model(self, provider) -> None:
        """After consecutive Ollama failures, clear VRAM and try a smaller model."""
        self._consecutive_ollama_failures += 1
        if self._consecutive_ollama_failures < self._FAILURE_THRESHOLD_FOR_DOWNGRADE:
            return

        logger.warning(
            "Ollama model failed %d times consecutively. Attempting VRAM clear and model downgrade.",
            self._consecutive_ollama_failures,
        )

        # Clear VRAM — and let the CUDA release settle before anything
        # reloads. The observed brown-out loop was exactly this path firing
        # every ~90s: unload the timed-out model, a queued retry reloads it
        # instantly into a half-released pool, it comes back CPU-spilled,
        # and the next timeout re-arms the loop.
        if hasattr(provider, 'clear_vram'):
            await provider.clear_vram()
            await asyncio.sleep(2.0)

        # Try smaller models — through the provider's own client so the
        # request carries the Companion proxy's auth. A raw unauthenticated
        # client 401s here, which silently disabled the downgrade while the
        # VRAM-clear side effect kept firing.
        for smaller_model in self._SMALLER_MODELS:
            try:
                resp = await provider._client.post(
                    f"{provider._host}/api/show",
                    json={"model": smaller_model},
                    timeout=10.0,
                )
                if resp.status_code == 200:
                    logger.info("Downgrading to smaller model: %s", smaller_model)
                    self._current_model_override = smaller_model
                    self._consecutive_ollama_failures = 0
                    return
                logger.info(
                    "Downgrade probe for %s: HTTP %d — skipping",
                    smaller_model, resp.status_code)
            except Exception:
                continue

    async def text_completion(self, prompt: str, max_tokens: int = 4096, timeout: float = 60, job_id: str = "", skip_circuit_breaker: bool = False, model_override: str | None = None, local_only: bool = False, json_mode: bool = False, json_schema: dict | None = None) -> str:
        """Generic text completion using the configured provider chain.

        Used by transcript correction, translation, and other text-only tasks.
        Falls back through the provider chain on failure.
        Returns the raw text response from the first successful provider.

        Args:
            skip_circuit_breaker: If True, failures are NOT recorded in the
                circuit breaker. Use this for non-critical/optional operations
                (like transcript polishing) that should not degrade the provider
                for subsequent critical operations (summary, clip detection).
            model_override: When set and the active provider is OpenRouter,
                use this model for the call instead of the editorial model
                (restored afterwards). Lets the subtitle translator route
                through a dedicated OPENROUTER_TRANSLATION_MODEL while
                transcript polishing keeps using OPENROUTER_EDITORIAL_MODEL.
                Ignored for non-OpenRouter providers and when the Ollama
                downgrade override is already active.
            local_only: When True, only LOCAL Ollama providers are attempted —
                cloud providers (OpenRouter/Gemini/Groq/Anthropic) are skipped
                for this call. Used by subtitle polish so a slow local/companion
                GPU batch is never silently answered — and billed — by a cloud
                provider. Fail-soft: the caller keeps the raw draft if no local
                provider succeeds.
        """
        # Remember if any provider failed specifically due to upstream
        # rate-limiting (HTTP 429) so the caller (e.g. the subtitle translator)
        # can stop early + surface a visible ``translation_failed`` rather than
        # crawling every batch. Without this the rate-limit nature is lost in
        # the generic AllProvidersFailedError.
        saw_rate_limit = False
        for provider in self._get_active_chain():
            pname = provider.provider_name
            # Strictly-local polish: skip cloud providers so a slow local /
            # companion GPU batch is never silently answered (and billed) by a
            # cloud provider. The caller keeps the raw draft when no local
            # provider succeeds.
            if local_only and pname != "ollama":
                continue
            model_name = provider.text_model_name
            original_model = None
            # Apply model override for Ollama if we've downgraded after failures
            if pname == "ollama" and self._current_model_override:
                model_name = self._current_model_override
                # Temporarily override the provider's text model
                original_model = provider._editorial_model
                provider._editorial_model = self._current_model_override
            elif model_override and pname in ("openrouter", "ollama") and hasattr(provider, "_editorial_model"):
                # Caller-requested model for this call only (the dedicated
                # translation model — OPENROUTER_TRANSLATION_MODEL or
                # OLLAMA_TRANSLATION_MODEL). Lets the subtitle translator route
                # through qwen3 while editorial/SEO keep the fast editorial model.
                # Temporarily swap the provider's text model; restored in the
                # finally below.
                _eff_override = model_override
                # GPU-first: if the requested Ollama model (e.g. qwen3:4b-q4)
                # won't fit this card and would spill to the CPU, substitute a
                # smaller-quant build of the SAME model that runs fully on the
                # GPU. Memoized on the provider; no-op on ample VRAM / cloud.
                if pname == "ollama" and hasattr(provider, "resolve_gpu_fitting_model"):
                    try:
                        _eff_override = await provider.resolve_gpu_fitting_model(model_override)
                    except Exception:
                        _eff_override = model_override
                model_name = _eff_override
                original_model = provider._editorial_model
                provider._editorial_model = _eff_override
            _call_timeout = timeout
            # On a budget-limited remote card, take the text turn on the GPU
            # before calling — a concurrent vision batch would CPU-spill this
            # model otherwise. Waiting here does NOT count against the call's
            # own timeout (that clock starts at the provider call below).
            _lease_held = False
            if pname == "ollama" and await self._gpu_class_serialize_needed(provider):
                await self._gpu_class_lease.acquire("text")
                _lease_held = True
            try:
                # Evict a leftover model (e.g. the vision model) before the text
                # call, but KEEP the text model resident so consecutive text
                # calls in a stage (summary, SEO, render-plan conversion, MTPE)
                # don't each pay a multi-GB reload — the load/unload churn that
                # slowed analysis and showed in the pipeline visualization.
                if pname == "ollama" and hasattr(provider, 'clear_vram') and self._consecutive_ollama_failures == 0 and not self._current_model_override:
                    await provider.clear_vram(except_model=model_name)
                # Cold-load-aware timeout: the caller's ceiling (e.g. the
                # polisher's 90s) exists to catch a STUCK model — but on a
                # small card the same silence during a multi-GB weights load
                # is normal, and treating the two alike is what pushed local
                # polish batches to the paid cloud fallback. When the target
                # model is not resident, grant the configured cold-load
                # allowance on top; warm calls keep the tight ceiling.
                #
                # A resident-but-CPU-SPILLED model (partial VRAM residency)
                # gets the SAME allowance: it generates as slowly as a cold
                # load, and a real brown-out showed the warm 90s ceiling
                # killing calls the GPU host was completing at 21s-3m26s —
                # 72 consecutive "timeouts" while translation shipped
                # nothing. Worse, Ollama never rebalances a split model on
                # its own, so once per cooldown window we also DEFRAG: evict
                # everything (the squatting vision model AND the split
                # target) so this call's generate reloads the target into a
                # clean pool.
                _cold_extra = float(getattr(
                    settings, "OLLAMA_COLD_LOAD_TIMEOUT_EXTRA_S", 240.0) or 0.0)
                if (pname == "ollama" and _cold_extra > 0
                        and hasattr(provider, "model_gpu_fraction")):
                    try:
                        _frac = await provider.model_gpu_fraction(model_name)
                        if _frac is None:
                            _call_timeout = timeout + _cold_extra
                            logger.info(
                                "text_completion: %s is not resident — cold "
                                "load expected; extending timeout %.0fs → %.0fs",
                                model_name, timeout, _call_timeout,
                            )
                        elif _frac < 0.9:
                            _call_timeout = timeout + _cold_extra
                            _now = time.monotonic()
                            if _now < self._defrag_futile_until:
                                # A prior defrag didn't win the VRAM back — the
                                # pressure is external. Keep the extended
                                # timeout, skip the churn.
                                logger.info(
                                    "text_completion: %s is %.0f%% on GPU — "
                                    "external VRAM pressure (defrag suspended); "
                                    "extending timeout %.0fs → %.0fs",
                                    model_name, _frac * 100, timeout, _call_timeout,
                                )
                            elif (_now - self._last_ollama_defrag) >= 120.0:
                                if (self._last_ollama_defrag > 0
                                        and _frac <= self._frac_at_last_defrag + 0.15):
                                    # The last defrag changed nothing (a real
                                    # run sat at exactly 22% across a dozen
                                    # cycles — the squatter was the whisper
                                    # sidecar, outside Ollama's reach). Each
                                    # futile cycle costs an unload + multi-GB
                                    # reload; stop paying it.
                                    self._defrag_futile_until = _now + 900.0
                                    logger.warning(
                                        "text_completion: %s still %.0f%% on GPU "
                                        "after a defrag — VRAM pressure is "
                                        "external (another process holds the "
                                        "GPU); suspending defrags for 15 min, "
                                        "keeping extended timeouts",
                                        model_name, _frac * 100,
                                    )
                                else:
                                    self._last_ollama_defrag = _now
                                    self._frac_at_last_defrag = _frac
                                    logger.warning(
                                        "text_completion: %s is only %.0f%% on GPU "
                                        "(CPU-spilled) — defragmenting: evicting all "
                                        "models so it reloads into a clean pool; "
                                        "timeout %.0fs → %.0fs",
                                        model_name, _frac * 100, timeout, _call_timeout,
                                    )
                                    # Forget the sticky num_gpu rung too: a
                                    # first load under (since-cleared) VRAM
                                    # pressure pinned a run at 22% residency
                                    # end-to-end because every reload reused
                                    # the low remembered rung — the clean pool
                                    # was never actually used.
                                    if hasattr(provider, "reset_gpu_layers_memo"):
                                        try:
                                            provider.reset_gpu_layers_memo(model_name)
                                        except Exception:
                                            pass
                                    if hasattr(provider, "clear_vram"):
                                        await provider.clear_vram()
                                        # CUDA frees asynchronously; reloading into a
                                        # half-released pool is how the 1.8GB/8.3GB
                                        # split kept coming back.
                                        await asyncio.sleep(2.0)
                            else:
                                logger.info(
                                    "text_completion: %s is %.0f%% on GPU — "
                                    "degraded but recently defragmented; "
                                    "extending timeout %.0fs → %.0fs",
                                    model_name, _frac * 100, timeout, _call_timeout,
                                )
                    except Exception:
                        pass
                elif (pname == "ollama" and _cold_extra > 0
                        and hasattr(provider, "is_model_loaded")):
                    try:
                        if not await provider.is_model_loaded(model_name):
                            _call_timeout = timeout + _cold_extra
                            logger.info(
                                "text_completion: %s is not resident — cold "
                                "load expected; extending timeout %.0fs → %.0fs",
                                model_name, timeout, _call_timeout,
                            )
                    except Exception:
                        pass
                logger.info("text_completion attempting via %s model=%s (%d chars prompt)", pname, model_name, len(prompt))
                # JSON constraints: Ollama enforces them at the grammar level
                # (``format``); OpenRouter maps them to OpenAI-style
                # ``response_format`` (strict json_schema / json_object) with
                # an internal downgrade for models that reject it. Other
                # providers keep their plain call — the caller's parser still
                # guards their output. Without this forwarding a cloud-routed
                # translation batch loses the exact-count array enforcement
                # and the parse-miss retry cascade returns. NOTE: this
                # forwarding is also what makes ``json_mode=True`` callers
                # work at all — the parameter used to not exist here, so those
                # calls raised TypeError and silently fell into their
                # except-blocks (the batched cleanup prefill never ran).
                _tc_kw = {}
                if pname in ("ollama", "openrouter"):
                    if json_schema is not None:
                        _tc_kw["json_schema"] = json_schema
                    elif json_mode:
                        _tc_kw["json_mode"] = True
                t0 = time.monotonic()
                result = await asyncio.wait_for(
                    provider.text_complete(prompt, max_tokens=max_tokens, timeout=int(_call_timeout), **_tc_kw),
                    timeout=_call_timeout,
                )
                elapsed = time.monotonic() - t0
                logger.info("text_completion via %s model=%s completed in %.1fs", pname, model_name, elapsed)
                if not skip_circuit_breaker:
                    self._circuit_breaker.record_success(pname)
                # Reset consecutive failure counter on success
                if pname == "ollama":
                    self._consecutive_ollama_failures = 0
                return result
            except asyncio.TimeoutError:
                if not skip_circuit_breaker:
                    self._circuit_breaker.record_failure(pname)
                logger.warning("text_completion via %s model=%s timed out after %.0fs — trying next provider", pname, model_name, _call_timeout)
                if pname == "ollama":
                    await self._maybe_downgrade_ollama_model(provider)
                await self._notify_fallback(job_id, pname, f"Text completion timed out after {_call_timeout:.0f}s (model={model_name})")
                continue
            except Exception as e:
                if not skip_circuit_breaker:
                    self._circuit_breaker.record_failure(pname)
                # A hard billing/key-limit 403 won't recover this job — abandon
                # the provider so later batches/stages skip it instead of eating
                # a 403 round-trip each (the recurring OpenRouter retry waste).
                self._mark_if_key_limited(pname, e)
                if isinstance(e, ProviderRateLimitError) or "429" in str(e) or "rate limit" in str(e).lower():
                    saw_rate_limit = True
                logger.warning("text_completion via %s model=%s failed: %s — trying next provider", pname, model_name, e)
                if pname == "ollama" and ("stalled" in str(e).lower() or "overloaded" in str(e).lower()):
                    await self._maybe_downgrade_ollama_model(provider)
                await self._notify_fallback(job_id, pname, str(e))
                continue
            finally:
                if _lease_held:
                    await self._gpu_class_lease.release()
                # Restore original model if we overrode it
                if original_model is not None:
                    provider._editorial_model = original_model
        if saw_rate_limit:
            # Preserve the rate-limit nature so callers can fail loudly/early.
            raise ProviderRateLimitError("All providers rate-limited (HTTP 429) for text completion")
        raise AllProvidersFailedError("All providers failed for text completion")

    async def generate_seo(
        self,
        clip_title: str,
        clip_transcript: str,
        video_summary: str,
        platform: str,
        job_id: str,
    ) -> tuple[ClipSEO, str]:
        """Returns (seo, provider_name_used)."""
        seo_prompt = self._custom_prompts.seo if self._custom_prompts else None
        for provider in self._get_active_chain():
            try:
                await self._notify_attempt(job_id, provider, "SEO generation")
                t0 = time.monotonic()
                result = await provider.generate_seo(
                    clip_title, clip_transcript, video_summary,
                    platform, cancel_check=self._cancel_check,
                    custom_prompt=seo_prompt,
                )
                elapsed = time.monotonic() - t0
                logger.info("SEO generation via %s completed in %.1fs", provider.provider_name, elapsed)
                self._circuit_breaker.record_success(provider.provider_name)
                return result, self._get_task_model(provider, "seo")
            except (ProviderRateLimitError, ProviderError) as e:
                self._circuit_breaker.record_failure(provider.provider_name)
                self._mark_if_key_limited(provider.provider_name, e)
                await self._notify_fallback(job_id, provider.provider_name, str(e))
                continue
        raise AllProvidersFailedError("All providers failed for SEO generation")
