"""Per-stage VRAM ledger (audit Phase 3.5).

Tiny helper that snapshots free/total GPU memory at each named pipeline
stage so the job log carries an auditable trail of who held VRAM when —
the 4 GB GTX 1650 budget lives or dies on stage ordering (Whisper
released before NLLB/pyannote, Ollama evicted before each GPU stage).

Usage:
    from backend.services.vram_ledger import snapshot
    snapshot("pre_whisper", job_id)

Each call logs one greppable line:
    VRAM ledger | job=42 stage=pre_whisper free=2101MB total=4096MB
and appends to an in-memory ledger retrievable via ``get_ledger(job_id)``
for the job report. No-ops silently on CPU-only hosts.
"""
from __future__ import annotations

import logging
import subprocess
from collections import defaultdict
from time import time

logger = logging.getLogger(__name__)

# job_id → [{stage, free_mb, total_mb, t}]
_ledgers: dict = defaultdict(list)
_MAX_JOBS = 32  # bound memory: drop oldest jobs past this


def _query_vram() -> tuple[int, int] | None:
    """(free_mb, total_mb) — torch first (no subprocess), nvidia-smi next.

    torch reports for the current CUDA context; nvidia-smi covers the
    whole card (includes CTranslate2/Ollama allocations torch can't see),
    so prefer nvidia-smi when available.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            free, total = out.stdout.strip().split("\n")[0].split(",")
            return int(float(free)), int(float(total))
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            return int(free_b / 1048576), int(total_b / 1048576)
    except Exception:
        pass
    return None


def snapshot(stage: str, job_id=None) -> dict | None:
    """Record + log a VRAM snapshot for ``stage``. Returns the entry."""
    vram = _query_vram()
    if vram is None:
        return None
    free_mb, total_mb = vram
    entry = {"stage": stage, "free_mb": free_mb,
             "total_mb": total_mb, "t": round(time(), 1)}
    key = str(job_id) if job_id is not None else "_global"
    _ledgers[key].append(entry)
    while len(_ledgers) > _MAX_JOBS:
        _ledgers.pop(next(iter(_ledgers)))
    logger.info("VRAM ledger | job=%s stage=%s free=%dMB total=%dMB",
                key, stage, free_mb, total_mb)
    return entry


def get_ledger(job_id) -> list:
    """All snapshots recorded for a job (for the job report)."""
    return list(_ledgers.get(str(job_id), []))
