"""Pre-analysis GPU memory preflight.

Runs at the start of every analysis (and optionally right before the
Whisper transcribe stage) to make sure as much VRAM as possible is
available before the heavy ML stages kick in.

The container is sharing a single GPU with the Ollama container (running
on ``OLLAMA_HOST``). On a small card like the GTX 1650 4 GB, every megabyte
counts — a stray Ollama model resident from a previous run can starve
Whisper of the inference workspace it needs and force a CPU fallback
that's 10-20× slower than realtime.

What this module does:
1. Queries Ollama's ``/api/ps`` to enumerate currently-loaded models.
2. For each model with a non-zero ``size_vram``, POSTs to ``/api/generate``
   with ``keep_alive=0`` — the documented Ollama API for evicting a
   model from VRAM without restarting the daemon.
3. Polls ``/api/ps`` until the models are gone (or a short timeout).
4. Drops the reframer's cached Whisper engine (so a stale fp16 GPU copy
   from a prior job doesn't sit on VRAM through this analysis).
5. Calls ``torch.cuda.empty_cache()`` + GC so PyTorch's cached allocator
   releases anything it was holding.
6. Reports the freed VRAM via ``nvidia-smi`` (system-wide, not the
   container-local view that ``torch.cuda.mem_get_info`` returns).

Subsequent editorial AI calls trigger Ollama to auto-reload models on
demand — no need to restart the Ollama container.

Every step is best-effort: any failure is logged at WARNING and the
preflight returns the freed-VRAM stats we managed to collect. The
pipeline is never blocked by this module.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import subprocess
from typing import Optional

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)


# ── nvidia-smi probe (system-wide VRAM, not container-local) ──────────────

def _nvidia_smi_free_mb() -> Optional[int]:
    """Return GPU 0's free VRAM in MB via nvidia-smi, or None on failure.

    Critical: this sees VRAM held by other processes on the host (the
    Ollama container, another browser tab, etc.) — ``torch.cuda.mem_get_info``
    only sees this container's view, which is misleading when multiple
    containers share one GPU.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        first = result.stdout.strip().splitlines()[0].strip()
        return int(first)
    except Exception:
        return None


def _nvidia_smi_compute_procs() -> list[dict]:
    """List compute processes currently holding VRAM on GPU 0.

    Returns a list of {"pid": int, "name": str, "used_mb": int}. Useful
    for diagnosing what's holding VRAM when the Ollama unload doesn't
    free as much as expected. Returns [] on any nvidia-smi failure.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return []
        procs = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    procs.append({
                        "pid": int(parts[0]),
                        "name": parts[1],
                        "used_mb": int(parts[2]),
                    })
                except ValueError:
                    continue
        return procs
    except Exception:
        return []


# ── Ollama model eviction (the heavy lift) ────────────────────────────────

async def _list_ollama_loaded_models(client: httpx.AsyncClient, host: str) -> list[dict]:
    """Return Ollama's /api/ps payload — currently-loaded models."""
    try:
        resp = await client.get(f"{host}/api/ps", timeout=5.0)
        if resp.status_code != 200:
            return []
        return resp.json().get("models", []) or []
    except Exception as e:
        logger.debug("preflight: /api/ps query failed: %s", e)
        return []


async def _unload_ollama_model(client: httpx.AsyncClient, host: str, model: str) -> bool:
    """POST keep_alive=0 to Ollama to evict ``model`` from VRAM."""
    try:
        resp = await client.post(
            f"{host}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10.0,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.debug("preflight: unload of %s failed: %s", model, e)
        return False


async def _evict_all_ollama_models(host: str, job_id: str) -> dict:
    """Unload every Ollama model with non-zero size_vram. Returns a stats dict."""
    stats = {
        "host": host,
        "models_evicted": [],
        "vram_freed_mb": 0,
        "failed": [],
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        loaded = await _list_ollama_loaded_models(client, host)
        if not loaded:
            logger.info("[%s] preflight: no Ollama models currently loaded", job_id)
            return stats

        for m in loaded:
            name = m.get("name", "")
            size_vram = int(m.get("size_vram", 0) or 0)
            if not name or size_vram <= 0:
                continue
            ok = await _unload_ollama_model(client, host, name)
            if ok:
                stats["models_evicted"].append({
                    "name": name,
                    "vram_mb": size_vram // (1024 * 1024),
                })
                stats["vram_freed_mb"] += size_vram // (1024 * 1024)
                logger.info(
                    "[%s] preflight: evicted %s from VRAM (%.0f MB)",
                    job_id, name, size_vram / 1024 / 1024,
                )
            else:
                stats["failed"].append(name)
                logger.warning("[%s] preflight: failed to evict %s", job_id, name)

        # Poll /api/ps a few times to confirm models actually unloaded.
        # Ollama's eviction is async — keep_alive=0 schedules the unload
        # but doesn't block. Give it up to 5 seconds to complete.
        for _ in range(10):
            still_loaded = [
                m for m in await _list_ollama_loaded_models(client, host)
                if int(m.get("size_vram", 0) or 0) > 0
            ]
            if not still_loaded:
                break
            await asyncio.sleep(0.5)
        else:
            still = [m.get("name", "?") for m in still_loaded]
            logger.warning(
                "[%s] preflight: %d Ollama model(s) still loaded after eviction: %s",
                job_id, len(still_loaded), still,
            )

    return stats


# ── Local torch / Whisper cache cleanup ───────────────────────────────────

def _release_local_torch_vram(job_id: str) -> int:
    """Drop the reframer's cached Whisper engine + flush torch allocator.

    Returns the freed VRAM in MB (best-effort, can be 0 if torch isn't
    installed or no CUDA device is present).
    """
    freed_mb = 0
    try:
        from backend.services.reframer_audio import AudioIntelligence
        if getattr(AudioIntelligence, "_cached_engine", None) is not None:
            AudioIntelligence._cached_engine = None
            AudioIntelligence._cached_model_name = None
            AudioIntelligence._cached_device = None
            logger.info("[%s] preflight: dropped cached faster-whisper engine", job_id)
    except Exception as e:
        logger.debug("[%s] preflight: Whisper cache drop skipped: %s", job_id, e)

    gc.collect()
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            before_free, _ = torch.cuda.mem_get_info()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            after_free, _ = torch.cuda.mem_get_info()
            freed_mb = max(0, (after_free - before_free) // (1024 * 1024))
            logger.debug(
                "[%s] preflight: torch allocator released %d MB",
                job_id, freed_mb,
            )
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[%s] preflight: torch cleanup skipped: %s", job_id, e)

    return freed_mb


# ── Public entry points ───────────────────────────────────────────────────

async def ensure_gpu_free_before_analysis(job_id: str) -> dict:
    """Free as much VRAM as possible before a new analysis starts.

    Idempotent and safe to call multiple times. Skipped entirely when
    ``GPU_FREE_BEFORE_ANALYSIS`` is False. Returns a stats dict the
    caller can log.
    """
    if not getattr(settings, "GPU_FREE_BEFORE_ANALYSIS", True):
        logger.debug("[%s] preflight: GPU_FREE_BEFORE_ANALYSIS is False — skipping", job_id)
        return {"skipped": True}

    before_free_mb = _nvidia_smi_free_mb()
    host = (getattr(settings, "OLLAMA_HOST", "") or "").rstrip("/")
    stats: dict = {
        "before_free_mb": before_free_mb,
        "ollama_host": host,
    }

    if host:
        try:
            ollama_stats = await asyncio.wait_for(
                _evict_all_ollama_models(host, job_id), timeout=20.0,
            )
            stats["ollama"] = ollama_stats
        except asyncio.TimeoutError:
            logger.warning("[%s] preflight: Ollama eviction timed out — continuing", job_id)
            stats["ollama"] = {"timeout": True}
        except Exception as e:
            logger.warning("[%s] preflight: Ollama eviction error: %s", job_id, e)
            stats["ollama"] = {"error": str(e)}
    else:
        logger.debug("[%s] preflight: no OLLAMA_HOST configured — skipping Ollama eviction", job_id)
        stats["ollama"] = {"skipped": True, "reason": "no OLLAMA_HOST"}

    stats["torch_freed_mb"] = _release_local_torch_vram(job_id)

    after_free_mb = _nvidia_smi_free_mb()
    stats["after_free_mb"] = after_free_mb

    if before_free_mb is not None and after_free_mb is not None:
        freed = after_free_mb - before_free_mb
        logger.info(
            "[%s] GPU preflight complete — system VRAM: %d MB free before → %d MB free after (+%d MB)",
            job_id, before_free_mb, after_free_mb, freed,
        )
    else:
        # nvidia-smi unavailable (CPU-only host?) — log what we know.
        logger.info(
            "[%s] GPU preflight complete — Ollama models evicted: %d, torch freed: %d MB",
            job_id,
            len(stats.get("ollama", {}).get("models_evicted", [])),
            stats.get("torch_freed_mb", 0),
        )

    # If anything's still hoarding VRAM, surface the suspect processes so
    # the user can see what to stop. Only log when 'tight' — under 1 GB free.
    if after_free_mb is not None and after_free_mb < 1024:
        procs = _nvidia_smi_compute_procs()
        if procs:
            top = sorted(procs, key=lambda p: p["used_mb"], reverse=True)[:5]
            logger.warning(
                "[%s] GPU still tight (%d MB free) — top VRAM consumers: %s",
                job_id, after_free_mb,
                ", ".join(f"{p['name']}(pid={p['pid']}, {p['used_mb']}MB)" for p in top),
            )

    return stats


async def ensure_gpu_free_before_whisper(job_id: str) -> dict:
    """Lighter-touch preflight for right before the Whisper transcribe stage.

    The full analysis-start preflight already evicted Ollama at the start
    of the run; this call mainly cleans up anything the reframer's perceive
    stage loaded (YOLO-World, SFace, scene tracker) so Whisper gets the
    full available VRAM for its inference workspace.

    Skipped entirely when ``GPU_FREE_BEFORE_WHISPER`` is False.
    """
    if not getattr(settings, "GPU_FREE_BEFORE_WHISPER", True):
        return {"skipped": True}

    before_free_mb = _nvidia_smi_free_mb()
    stats = {"before_free_mb": before_free_mb}

    # Re-evict Ollama in case the editorial AI auto-loaded a model during
    # perceive (it shouldn't — perceive is local-only — but be safe).
    host = (getattr(settings, "OLLAMA_HOST", "") or "").rstrip("/")
    if host:
        try:
            ollama_stats = await asyncio.wait_for(
                _evict_all_ollama_models(host, job_id), timeout=10.0,
            )
            stats["ollama"] = ollama_stats
        except Exception as e:
            logger.debug("[%s] preflight-whisper: Ollama re-eviction skipped: %s", job_id, e)

    stats["torch_freed_mb"] = _release_local_torch_vram(job_id)

    after_free_mb = _nvidia_smi_free_mb()
    stats["after_free_mb"] = after_free_mb
    if before_free_mb is not None and after_free_mb is not None:
        logger.info(
            "[%s] pre-Whisper GPU preflight: %d MB free → %d MB free",
            job_id, before_free_mb, after_free_mb,
        )
    return stats
