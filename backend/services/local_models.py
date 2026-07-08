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
    # A newer Qwen minor refresh (qwen3.5) beats qwen3 within the same size tier
    # — it shares the ~2.5 GB Q4 footprint so the partial-offload ladder handles
    # it identically. Outscores the 2507 bump so a 3.5 build wins when present.
    if "qwen3.5" in n or "qwen-3.5" in n:
        bonus += 4
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
        "temperature": float(getattr(settings, "QWEN3_TRANSLATION_TEMPERATURE", 0.6)),
        "top_p": float(getattr(settings, "QWEN3_TRANSLATION_TOP_P", 0.8)),
        "top_k": int(getattr(settings, "QWEN3_TRANSLATION_TOP_K", 20)),
        "min_p": float(getattr(settings, "QWEN3_TRANSLATION_MIN_P", 0.0)),
        "repeat_penalty": float(getattr(settings, "QWEN3_TRANSLATION_REPEAT_PENALTY", 1.1)),
        "presence_penalty": float(getattr(settings, "QWEN3_TRANSLATION_PRESENCE_PENALTY", 1.2)),
        "frequency_penalty": float(getattr(settings, "QWEN3_TRANSLATION_FREQUENCY_PENALTY", 0.3)),
    }


# Approximate effective bits-per-weight for common GGUF quantizations, used
# only to SIZE a model against VRAM (not for any accuracy claim). Good enough
# to tell a q4 (~4.8 bpw) from a q3 (~3.4 bpw) of the same model.
_QUANT_BPW = {
    "q2_k": 2.6, "q3_k_s": 3.1, "q3_k_m": 3.4, "q3_k_l": 3.7,
    "q4_0": 4.5, "q4_1": 4.9, "q4_k_s": 4.6, "q4_k_m": 4.8,
    "q5_0": 5.5, "q5_1": 5.9, "q5_k_s": 5.5, "q5_k_m": 5.7,
    "q6_k": 6.6, "q8_0": 8.5, "f16": 16.0, "fp16": 16.0, "bf16": 16.0,
}


def _parse_quant(name: str):
    """Return ``(quant_key, bits_per_weight)`` parsed from a GGUF tag, or
    ``(None, None)``. Matches the longest quant token so ``q4_k_m`` wins over
    ``q4``. Tolerates ``-``/``_`` separators and case (``q4_K_M``)."""
    n = (name or "").lower().replace("-", "_")
    best = None
    for k in _QUANT_BPW:
        if k in n and (best is None or len(k) > len(best)):
            best = k
    return (best, _QUANT_BPW[best]) if best else (None, None)


def _model_base_key(name: str) -> str:
    """Identity of a model IGNORING its quant tag, so different quantizations
    of the same weights compare equal.

    ``qwen3:4b-instruct-2507-q4_K_M`` and ``…-q3_K_M`` both key to
    ``qwen3:4b_instruct_2507``."""
    n = _norm_ollama_name(name).replace("-", "_")
    qk, _ = _parse_quant(n)
    if qk:
        idx = n.rfind(qk)
        if idx > 0:
            return n[:idx].rstrip("_")
    return n


def estimate_model_weights_gb(name: str) -> Optional[float]:
    """Rough GiB of GPU memory the WEIGHTS of ``name`` occupy.

    ``params_b * bits_per_weight / 8``. Returns ``None`` when the tag carries no
    parseable size. Quant defaults to q4 (~4.8 bpw) when the tag omits it."""
    params = _parse_params_b(name)
    if params is None:
        return None
    _, bpw = _parse_quant(name)
    if not bpw:
        bpw = 4.8  # assume a q4 build when the tag omits the quant
    return params * bpw / 8.0


def select_gpu_fitting_quant(
    configured: str,
    installed: list,
    *,
    total_vram_gb: float,
    baseline_reserve_gb: float,
    kv_headroom_gb: float,
):
    """Pick a GPU-fitting quantization of ``configured`` when it won't fit.

    Returns ``(chosen_model, reason)`` — ``reason`` is ``None`` (keep the
    configured model) when it already fits, VRAM can't be measured, or no
    smaller-quant build of the SAME weights is installed that fits.

    Fit test: ``weights_gb + kv_headroom_gb <= (total_vram_gb -
    baseline_reserve_gb)``. The reserve accounts for the CUDA context +
    baseline allocation that never frees; the headroom for the KV cache +
    compute graph. Among fitting same-base builds we keep the LARGEST quant
    (highest fidelity that still runs fully on the GPU)."""
    if not configured or total_vram_gb <= 0:
        return (configured, None)
    budget = total_vram_gb - max(0.0, baseline_reserve_gb)
    if budget <= 0:
        return (configured, None)

    def _fits(model_name: str) -> bool:
        w = estimate_model_weights_gb(model_name)
        return w is not None and (w + max(0.0, kv_headroom_gb)) <= budget

    if _fits(configured):
        return (configured, None)  # already runs fully on the GPU

    base = _model_base_key(configured)
    _, cfg_bpw = _parse_quant(configured)
    candidates = []
    for m in installed or []:
        if not m or _model_base_key(m) != base:
            continue
        _, bpw = _parse_quant(m)
        if bpw is None:
            continue
        if cfg_bpw is not None and bpw >= cfg_bpw:
            continue  # not a smaller quant than what's configured
        if _fits(m):
            candidates.append((bpw, m))
    if not candidates:
        return (configured, None)  # nothing smaller that fits — keep configured
    candidates.sort(reverse=True)  # largest fitting quant = best fidelity
    chosen = candidates[0][1]
    reason = (
        f"{configured} (~{estimate_model_weights_gb(configured):.1f}GB) won't fit "
        f"fully on a {total_vram_gb:.1f}GB GPU → using {chosen} "
        f"(~{estimate_model_weights_gb(chosen):.1f}GB, runs on GPU)")
    return (chosen, reason)


def gpu_offload_ladder(model_name: str) -> list[int]:
    """Descending ``num_gpu`` (layer-count) values to try for ``model_name``.

    A ~4B q4 model is a hair too big to fully offload on a 4 GB card: forcing
    all layers (num_gpu=99) OOMs, and the old fallback then dumped the WHOLE
    model onto the CPU — very slow. Most of its layers DO fit, so we step down
    through PARTIAL GPU offload (a layer count < the model's total) before ever
    touching CPU, keeping the bulk of compute on the GPU. The first rung (99)
    lets a roomy card place everything on the GPU; the OOM-driven step-down
    self-tunes to the card, so the same ladder is right for 4 GB and 8 GB+.

    Returns ``[99, 0]`` (all-GPU then CPU — the legacy path) for models that
    already fit fully (< the partial threshold) or when partial offload is off.
    The ladder always ends at ``0`` (CPU) as the last-resort rung."""
    if not getattr(settings, "OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD", True):
        return [99, 0]
    params = _parse_params_b(model_name or "")
    threshold = float(getattr(settings, "OLLAMA_PARTIAL_OFFLOAD_MIN_PARAMS_B", 3.5))
    if params is None or params < threshold:
        return [99, 0]
    start = int(getattr(settings, "OLLAMA_MIDSIZE_GPU_LAYERS_START", 32))
    raw = [99, start, int(start * 0.75), int(start * 0.5), 0]
    ladder: list[int] = []
    for v in raw:
        v = max(0, int(v))
        if v not in ladder:
            ladder.append(v)
    if ladder[-1] != 0:
        ladder.append(0)
    return ladder


def gpu_offload_ladder_for_vram(
    model_name: str,
    total_vram_gb: float,
    *,
    reserve_gb: Optional[float] = None,
    kv_headroom_gb: Optional[float] = None,
) -> list[int]:
    """VRAM-aware ``num_gpu`` ladder — like :func:`gpu_offload_ladder`, but sized
    PROACTIVELY to the card so the FIRST attempt already fits instead of always
    starting at ``num_gpu=99`` and paying a guaranteed OOM + VRAM-clear + retry
    for a model that can't fully offload.

    Decision, given the ACTIVE card's TOTAL VRAM (stable — unlike a free-VRAM
    probe that reads low right after Whisper unloads):

    * ``total_vram_gb <= 0`` (unknown) → the plain ladder (OOM-probe fallback,
      behavior unchanged for un-probed hosts).
    * model fully fits (``weights + kv_headroom <= total - reserve``) → the plain
      ladder: ``99`` first is correct, it loads fully on the GPU in one shot.
    * model does NOT fully fit → DROP the ``99`` rung (a guaranteed OOM) and start
      at the largest partial-offload rung that plausibly fits; if essentially no
      weight budget remains, ``[0]`` (CPU-only) — don't push a doomed GPU load.

    ``num_gpu`` is a layer count; layers-that-fit are approximated as
    proportional to the weight budget, using the same ``OLLAMA_MIDSIZE_GPU_
    LAYERS_START`` scale as the base ladder. The OOM step-down still self-tunes
    from wherever we start, so a slightly-off estimate only costs at most one
    extra step, never a wrong final placement."""
    base = gpu_offload_ladder(model_name)
    if not total_vram_gb or total_vram_gb <= 0:
        return base
    w = estimate_model_weights_gb(model_name)
    if w is None or w <= 0:
        return base
    if reserve_gb is None:
        reserve_gb = float(getattr(settings, "OLLAMA_GPU_BASELINE_RESERVE_GB", 1.2))
    if kv_headroom_gb is None:
        kv_headroom_gb = float(getattr(settings, "OLLAMA_GPU_KV_HEADROOM_GB", 0.55))
    budget = max(0.0, float(total_vram_gb) - max(0.0, reserve_gb))
    # Fully fits → the base ladder (99 first) is exactly right.
    if (w + max(0.0, kv_headroom_gb)) <= budget:
        return base
    # Doesn't fully fit: never START at 99 (guaranteed OOM). With essentially no
    # room for weights beyond the KV headroom, go straight to CPU.
    weight_budget = budget - max(0.0, kv_headroom_gb)
    if weight_budget <= 0.4:
        return [0]
    frac = min(1.0, max(0.0, weight_budget / w))
    start = int(getattr(settings, "OLLAMA_MIDSIZE_GPU_LAYERS_START", 32))
    fit_layers = max(1, int(round(start * frac)))
    trimmed = [x for x in base if 0 < x <= fit_layers]
    if not trimmed:
        trimmed = [fit_layers]
    trimmed.append(0)
    return trimmed


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
        from backend.services import ollama_registry
        _headers = ollama_registry.headers_for_url(host)
    except Exception:
        _headers = {}
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_headers) as client:
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
