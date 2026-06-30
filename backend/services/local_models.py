"""Pick the best LOCAL (Ollama) text model for editorial work in Offline Mode.

On a small GPU like the GTX 1650 (4 GB), the editorial AI (summaries, SEO,
clip scoring, transcript polishing, LLM translation) must run on a model that
actually fits in VRAM. Offline Mode auto-selects here instead of using the
configured cloud editorial model / cloud judge fallback.

``rank_local_editorial_models`` is a pure function (easy to test): it filters
out vision/multimodal models, drops anything too large for the card, and ranks
the rest so the most capable model that still fits wins. ``best`` = rank[0],
``second-best`` (the offline fallback) = rank[1].
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

# Families/tags that mean a vision/multimodal model — not for editorial TEXT.
_VISION_HINTS = (
    "llava", "moondream", "bakllava", "minicpm-v", "-vision", "vision",
    "llama3.2-vision", "qwen2-vl", "qwen2.5-vl",
)

# Small light-touch preference among similarly-sized models (newer/stronger
# instruct families first). Anything not listed ranks at 0.
_FAMILY_PREFERENCE = (
    "qwen3", "qwen2.5", "llama3.3", "llama3.2", "llama3.1",
    "gemma2", "mistral", "phi4", "phi3", "qwen2", "llama3",
)


def _parse_params_b(name: str) -> Optional[float]:
    """Parse the parameter count (in billions) from an Ollama model tag.

    "qwen2.5:3b-instruct" -> 3.0 · "llama3.1:8b" -> 8.0 · "phi3:3.8b" -> 3.8.
    Returns None when the tag carries no size (e.g. "model:latest")."""
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*b\b", name.lower())
    if not matches:
        return None
    try:
        # The size tag is the last numeric-b token in the name.
        return float(matches[-1])
    except ValueError:
        return None


def _is_vision(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in _VISION_HINTS)


def _total_vram_gb() -> float:
    """Total CUDA VRAM in GiB (0.0 when no GPU / torch unavailable). Used to
    decide whether a 4B editorial model can actually fit this card."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1_073_741_824
    except Exception:
        pass
    return 0.0


def _effective_editorial_max_params_b() -> float:
    """Resolve the editorial param cap, VRAM-aware.

    A 4B-q4 model OOMs on a 4 GB card during the editorial stage, so on a GPU
    smaller than ``OFFLINE_EDITORIAL_SMALL_GPU_GB`` the cap is lowered to
    ``OFFLINE_EDITORIAL_SMALL_GPU_MAX_PARAMS_B`` (default 3B) so auto-selection
    never picks a model that can't run on the GPU. When VRAM can't be detected
    (0.0) the configured cap is kept unchanged (no false downscoping in CPU /
    headless / non-CUDA environments)."""
    configured = float(getattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0))
    small_gb = float(getattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_GB", 5.5))
    small_cap = float(getattr(settings, "OFFLINE_EDITORIAL_SMALL_GPU_MAX_PARAMS_B", 3.0))
    if small_gb <= 0:
        return configured
    total = _total_vram_gb()
    if 0.0 < total < small_gb:
        eff = min(configured, small_cap)
        if eff < configured:
            logger.info(
                "Editorial model cap lowered to %.1fB (only %.1f GB VRAM < %.1f GB "
                "— a 4B model OOMs and would run on CPU)", eff, total, small_gb)
        return eff
    return configured


def _norm_ollama_name(name: str) -> str:
    """Normalize an Ollama tag for comparison: drop an ``ollama/`` prefix and a
    redundant ``:latest`` suffix, lowercase. Distinct size/quant tags stay
    distinct (``qwen2.5:3b`` != ``qwen2.5:7b``)."""
    n = (name or "").strip().lower()
    if n.startswith("ollama/"):
        n = n[len("ollama/"):]
    if n.endswith(":latest"):
        n = n[: -len(":latest")]
    return n


def _ollama_names_match(a: str, b: str) -> bool:
    """True when two Ollama tags refer to the same model (tolerating an
    ``ollama/`` prefix and a missing ``:latest``)."""
    return bool(a) and bool(b) and _norm_ollama_name(a) == _norm_ollama_name(b)


def _family_rank(name: str) -> int:
    n = name.lower()
    for i, fam in enumerate(_FAMILY_PREFERENCE):
        if fam in n:
            return len(_FAMILY_PREFERENCE) - i
    return 0


def _recency_bonus(name: str) -> int:
    """Tiebreak among same-family, same-size, same-instruct candidates toward
    the newer release. Qwen3's '-2507' instruct line is the multilingual-tuned
    refresh we want auto-selection to prefer over a plain '-instruct' tag."""
    n = name.lower()
    bonus = 0
    if "2507" in n:
        bonus += 2
    if "instruct" in n:
        bonus += 1
    return bonus


def qwen3_translation_options(model_name: str) -> dict:
    """Qwen3-family sampling options for the dedicated translation path.

    Returns ``{}`` for non-Qwen3 models so callers keep their existing sampling.
    Qwen3 repeats without a penalty and we want deterministic subtitle JSON, so
    we apply a low temperature + presence/repetition penalty + tighter top_p
    (all configurable). Applied ONLY on the translation path — not to global
    editorial sampling."""
    if "qwen3" not in (model_name or "").lower():
        return {}
    return {
        "temperature": float(getattr(settings, "QWEN3_TRANSLATION_TEMPERATURE", 0.2)),
        "top_p": float(getattr(settings, "QWEN3_TRANSLATION_TOP_P", 0.8)),
        "repeat_penalty": float(getattr(settings, "QWEN3_TRANSLATION_REPEAT_PENALTY", 1.05)),
        "presence_penalty": float(getattr(settings, "QWEN3_TRANSLATION_PRESENCE_PENALTY", 0.5)),
    }


def rank_local_editorial_models(
    model_names, max_params_b: Optional[float] = None
) -> list[str]:
    """Rank installed local text models best-first for editorial work.

    Vision models are dropped; models larger than ``max_params_b`` (default
    ``OFFLINE_EDITORIAL_MAX_PARAMS_B``, tuned for the 4 GB 1650) are dropped so
    we never pick a model that would spill to CPU. Ranking: instruct/chat-tuned
    first, then larger parameter count, then family preference. Models with no
    parsable size are kept but rank below sized ones.
    """
    if max_params_b is None:
        # VRAM-aware: on a small card (≤ OFFLINE_EDITORIAL_SMALL_GPU_GB) this
        # resolves to the 3B cap so a 4B model that would OOM/CPU isn't picked.
        max_params_b = _effective_editorial_max_params_b()

    candidates = []
    for name in model_names or []:
        if not name or _is_vision(name):
            continue
        params = _parse_params_b(name)
        if params is not None and params > max_params_b:
            continue  # too big for the card — would offload to CPU
        instruct = 1 if ("instruct" in name.lower() or "chat" in name.lower()) else 0
        candidates.append(
            (instruct, params or 0.0, _family_rank(name), _recency_bonus(name), name))

    # Sort best-first: instruct, then bigger, then preferred family, then the
    # newer release (so qwen3:4b-instruct-2507 beats a plain qwen3:4b-instruct).
    candidates.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)
    return [c[4] for c in candidates]


async def list_ollama_models(timeout: float = 4.0) -> list[str]:
    """Return the names of models installed on the configured Ollama host."""
    host = (getattr(settings, "OLLAMA_HOST", "") or "").rstrip("/")
    if not host:
        return []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{host}/api/tags")
            resp.raise_for_status()
            data = resp.json() or {}
            return [m.get("name", "") for m in (data.get("models") or []) if m.get("name")]
    except Exception as e:
        logger.debug("list_ollama_models failed: %s", e)
        return []


async def select_local_editorial_models(
    limit: int = 2, model_names: Optional[list] = None
) -> list[str]:
    """Best (and second-best) local editorial models for Offline Mode.

    Pass ``model_names`` to rank an already-fetched list (no network); omit it
    to query the Ollama host. Falls back to the configured
    ``OLLAMA_EDITORIAL_MODEL`` when nothing rankable is installed.
    """
    names = model_names if model_names is not None else await list_ollama_models()
    ranked = rank_local_editorial_models(names)
    if not ranked:
        fb = (getattr(settings, "OLLAMA_EDITORIAL_MODEL", "") or "").strip()
        return [fb][:limit] if fb else []
    # The explicitly-configured editorial model wins over auto-ranking when it's
    # actually installed — an operator who set OLLAMA_EDITORIAL_MODEL to a
    # specific tag should get it, not whatever the heuristic ranks first.
    configured = (getattr(settings, "OLLAMA_EDITORIAL_MODEL", "") or "").strip()
    if configured:
        match = next((n for n in ranked if _ollama_names_match(n, configured)), None)
        if match and ranked[0] != match:
            ranked = [match] + [n for n in ranked if n != match]
    return ranked[:limit]
