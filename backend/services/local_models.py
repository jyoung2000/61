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


def _family_rank(name: str) -> int:
    n = name.lower()
    for i, fam in enumerate(_FAMILY_PREFERENCE):
        if fam in n:
            return len(_FAMILY_PREFERENCE) - i
    return 0


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
        max_params_b = float(getattr(settings, "OFFLINE_EDITORIAL_MAX_PARAMS_B", 4.0))

    candidates = []
    for name in model_names or []:
        if not name or _is_vision(name):
            continue
        params = _parse_params_b(name)
        if params is not None and params > max_params_b:
            continue  # too big for the card — would offload to CPU
        instruct = 1 if ("instruct" in name.lower() or "chat" in name.lower()) else 0
        candidates.append((instruct, params or 0.0, _family_rank(name), name))

    # Sort best-first: instruct, then bigger, then preferred family.
    candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return [c[3] for c in candidates]


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
    return ranked[:limit]
