"""Per-library CPU thread caps for the pipeline's concurrent stages.

The pipeline now runs several CPU-heavy jobs at once — CTranslate2 NMT
batches, SpeechBrain embeddings, saliency numpy, ffmpeg — and each library
defaults to "use every core." Running two or three of those simultaneously
oversubscribes the host (context-switch and cache thrash) exactly when the
pipeline is busiest, and also starves the API event loop, which is why the
UI went laggy mid-job. Capping each library at roughly half the cores keeps
the COMBINED throughput higher and the box responsive.

``apply_thread_caps()`` is called once at app startup (backend.main). The
env caps only bind libraries imported afterwards; ``torch`` and ``cv2`` get
their runtime setters as well, which work post-import.

Override with ``CLIPAI_CPU_THREAD_CAP``: 0 (default) = auto (half the
cores, min 2), N = exactly N threads per library, -1 = don't touch anything
(restores library defaults).
"""

import logging
import os

logger = logging.getLogger(__name__)

_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_applied: dict = {"cap": None}


def recommended_thread_cap() -> int:
    """The per-library thread cap: explicit override, else half the cores
    (min 2). -1 disables capping entirely."""
    try:
        override = int(os.environ.get("CLIPAI_CPU_THREAD_CAP", "0") or 0)
    except ValueError:
        override = 0
    if override < 0:
        return -1
    if override > 0:
        return override
    return max(2, (os.cpu_count() or 4) // 2)


def apply_thread_caps() -> int:
    """Apply the cap to every threading knob we control. Idempotent.
    Returns the cap applied (or -1 when disabled)."""
    cap = recommended_thread_cap()
    if cap == _applied["cap"]:
        return cap
    if cap < 0:
        logger.info("CPU thread caps disabled (CLIPAI_CPU_THREAD_CAP=-1)")
        _applied["cap"] = cap
        return cap
    for key in _ENV_KEYS:
        # setdefault: an operator's explicit env choice always wins.
        os.environ.setdefault(key, str(cap))
    # Runtime setters — these work even after the library was imported.
    try:
        import torch
        torch.set_num_threads(cap)
    except Exception:
        pass
    try:
        import cv2
        cv2.setNumThreads(cap)
    except Exception:
        pass
    logger.info(
        "CPU thread caps applied: %d threads/library (%d cores; override "
        "via CLIPAI_CPU_THREAD_CAP, -1 disables)", cap, os.cpu_count() or 0)
    _applied["cap"] = cap
    return cap
