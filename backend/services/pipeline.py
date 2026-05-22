import asyncio
import json
import logging
import os
import shutil
import time as _time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx

from backend.config import settings
from backend.models import JobResult, JobStatus, FrameData, VideoSummary
from backend import database
from backend.services.frame_extractor import (
    get_video_metadata,
    extract_frames,
    extract_frames_at_timestamps,
    extract_audio,
    frame_to_base64,
)
from backend.services.vlm_fusion import apply_fusion_to_scene, fusion_enabled
from backend.services.ai_orchestrator import AIOrchestrator
from backend.services.prompts import load_prompts
from backend.services.providers.base import build_summary_from_transcript, has_real_summary_content, AllProvidersFailedError
from backend.services.audio_analyzer import analyze_audio_energy, format_audio_energy_map

logger = logging.getLogger(__name__)

# Project root — the directory that holds clipper_config.json (``/app`` in Docker).
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _log_gpu_memory(job_id: str, label: str):
    """Log current GPU memory state for VRAM debugging."""
    try:
        import torch
        if torch.cuda.is_available():
            free_mb, total_mb = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
            used_mb = total_mb - free_mb
            logger.info(
                "[%s] GPU VRAM [%s]: %.0f MB used / %.0f MB total (%.0f MB free)",
                job_id, label, used_mb, total_mb, free_mb,
            )
    except Exception:
        pass  # Non-critical — don't break pipeline if GPU query fails


async def _release_whisper_vram(job_id: str):
    """Release the reframer's cached Whisper engine so the VLM can use the GPU.

    On a 4GB GTX 1650, faster-whisper occupies ~800MB. The reframer caches
    its faster-whisper engine on ``AudioIntelligence._cached_engine``; dropping
    that reference plus a CUDA cache flush hands the GPU to Ollama's VLM.
    """
    try:
        import gc

        # Step 1: Drop the reframer's cached faster-whisper engine.
        try:
            from backend.services.reframer_audio import AudioIntelligence
            if getattr(AudioIntelligence, "_cached_engine", None) is not None:
                AudioIntelligence._cached_engine = None
                AudioIntelligence._cached_model_name = None
                AudioIntelligence._cached_device = None
                logger.info("[%s] Reframer Whisper engine cache cleared", job_id)
            else:
                logger.debug("[%s] No cached Whisper engine to release", job_id)
        except Exception as e:
            logger.debug("[%s] Whisper cache clear skipped: %s", job_id, e)

        # Step 2: Force Python GC (releases CTranslate2 C++ objects).
        gc.collect()
        gc.collect()

        # Step 3: Force a PyTorch CUDA cache flush.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
                free_mb, total_mb = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                logger.info(
                    "[%s] Whisper VRAM released — %.0f MB free / %.0f MB total "
                    "(PyTorch reserved: %.0fMB)",
                    job_id, free_mb, total_mb, reserved,
                )
                if reserved > 100:
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.empty_cache()
        except ImportError:
            pass
        except Exception as e:
            logger.debug("[%s] PyTorch cleanup skipped: %s", job_id, e)

        gc.collect()
        logger.info("[%s] Whisper unloaded from VRAM — GPU free for the VLM", job_id)

    except Exception as e:
        logger.warning("[%s] VRAM release error: %s", job_id, e)


def release_torch_gpu_memory():
    """Release all torch GPU memory. Safe to call multiple times, even if torch not loaded."""
    try:
        import gc
        import torch
        if not torch.cuda.is_available():
            return
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        # Try allocator reset for stubborn cached memory
        try:
            if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_set_allocator_settings'):
                torch.cuda.memory._set_allocator_settings("")
                gc.collect()
                torch.cuda.empty_cache()
        except Exception:
            pass
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        logger.info("Torch GPU memory released: %.0fMB still reserved", reserved)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Torch GPU release error: %s", e)


async def _trigger_ollama_gpu_rediscovery(job_id: str, provider):
    """After releasing torch VRAM, force Ollama to re-discover GPU.

    Ollama caches GPU state from startup. If discovery failed (timeout) or
    the GPU was full (torch hogging VRAM), all subsequent loads use CPU.
    Loading a model with num_gpu=99 AND a vision request triggers a fresh
    GPU scan for both the LLM and the CLIP vision encoder.
    """
    if not hasattr(provider, '_host'):
        return False
    try:
        host = provider._host
        vision_model = provider._vision_model
        logger.info("[%s] Triggering Ollama GPU re-discovery after VRAM release...", job_id)

        # First clear any CPU-loaded models
        if hasattr(provider, 'clear_vram'):
            await provider.clear_vram()
        await asyncio.sleep(2)

        # Generate a tiny 1x1 test image for vision probe
        import base64 as _b64
        _tiny_img = _b64.b64encode(
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        ).decode()

        # Load vision model with GPU forced AND an image to trigger CLIP GPU allocation
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{host}/api/chat",
                json={
                    "model": vision_model,
                    "messages": [{"role": "user", "content": "test", "images": [_tiny_img]}],
                    "stream": False,
                    "options": {"num_gpu": 99, "num_predict": 1},
                },
                timeout=120,
            )
            if resp.status_code == 200:
                ps = await client.get(f"{host}/api/ps", timeout=10)
                if ps.status_code == 200:
                    for m in ps.json().get("models", []):
                        if m.get("size_vram", 0) > 0:
                            logger.info(
                                "[%s] Ollama GPU re-discovery succeeded — %s on GPU (%.0fMB VRAM)",
                                job_id, m.get("name", ""), m.get("size_vram", 0) / 1024 / 1024,
                            )
                            # Don't clear — leave model loaded so scene analysis uses GPU
                            if hasattr(provider, '_force_cpu'):
                                provider._force_cpu = False
                            return True
                logger.warning("[%s] Ollama GPU re-discovery: model still on CPU", job_id)
                # Unload CPU model so scene analysis can retry on GPU
                if hasattr(provider, 'clear_vram'):
                    await provider.clear_vram()
                return False
            elif resp.status_code == 500:
                # OOM during GPU load — model won't fit. Let scene analysis handle CPU fallback
                logger.warning("[%s] Ollama GPU re-discovery: vision model OOM on GPU (HTTP 500)", job_id)
                if hasattr(provider, 'clear_vram'):
                    await provider.clear_vram()
                return False
            else:
                logger.warning("[%s] Ollama GPU re-discovery failed: HTTP %d", job_id, resp.status_code)
                return False
    except Exception as e:
        logger.warning("[%s] Ollama GPU re-discovery error: %s", job_id, e)
        return False


# Dedicated thread pool for base64 frame encoding so it never competes
# with the default executor or the Whisper transcription pool.
_b64_executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="b64enc")

# ── Pipeline stage timeouts ──────────────────────────────────────────
_METADATA_TIMEOUT = 120            # 2 min for FFprobe metadata
_EXTRACTION_TIMEOUT = 600          # 10 min for frame + audio extraction
_SUMMARY_CLIP_TIMEOUT = 900        # 15 min for summary + clip detection
_B64_ENCODE_TIMEOUT = 300          # 5 min for base64 frame encoding


# Pure-Python helpers live in pipeline_helpers so unit tests can import
# them without pulling in the full LLM provider stack. Re-export the
# public names so existing pipeline.py call sites keep working.
from backend.services.pipeline_helpers import (  # noqa: E402
    _hash_file_sha256,
    _maybe_use_cached_extraction,
    _write_extraction_manifest,
    _stage_timer,
    _record_pipeline_warning,
    _drain_pipeline_telemetry,
    is_synthetic_scene,
)

# ── Pipeline heartbeat — prevents >15s gaps in progress updates ──────
class _PipelineHeartbeat:
    """Emits keepalive messages when no real progress update has been sent."""

    def __init__(self, job_id: str, interval: float = 15.0):
        self.job_id = job_id
        self.interval = interval
        self.last_emit = _time.monotonic()
        self.current_stage = ""
        self.stage_start = _time.monotonic()
        self._task: asyncio.Task | None = None

    def touch(self, stage: str = ""):
        """Call whenever a real progress event is emitted."""
        self.last_emit = _time.monotonic()
        if stage and stage != self.current_stage:
            self.current_stage = stage
            self.stage_start = _time.monotonic()

    async def _run(self):
        """Background loop that checks for staleness every 5 seconds."""
        try:
            while True:
                await asyncio.sleep(5.0)
                elapsed_since_emit = _time.monotonic() - self.last_emit
                if elapsed_since_emit >= self.interval and self.current_stage:
                    stage_elapsed = int(_time.monotonic() - self.stage_start)
                    mins, secs = divmod(stage_elapsed, 60)
                    msg = f"Still processing... ({self.current_stage} \u2014 {mins}m {secs}s elapsed)"
                    # Only broadcast via WebSocket — don't update DB to avoid
                    # overwriting real progress values with heartbeat messages.
                    await broadcast_ws(self.job_id, {
                        "type": "heartbeat",
                        "message": msg,
                    })
                    self.last_emit = _time.monotonic()
        except asyncio.CancelledError:
            pass

    def start(self):
        self._task = asyncio.create_task(self._run())

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None


# Active heartbeats per job
_heartbeats: dict[str, _PipelineHeartbeat] = {}

# Semaphore to limit concurrent analyses
_analysis_semaphore: asyncio.Semaphore | None = None

# WebSocket broadcast registry
_ws_subscribers: dict[str, list] = {}

# Cancellation events — set() means "please cancel"
_cancel_events: dict[str, asyncio.Event] = {}


class CancelledError(Exception):
    """Raised when a job is cancelled by the user."""


def request_cancel(job_id: str):
    """Signal a running job to stop at the next checkpoint."""
    ev = _cancel_events.get(job_id)
    if ev:
        ev.set()
        logger.info(f"Cancellation requested for job {job_id}")


def is_cancel_requested(job_id: str) -> bool:
    ev = _cancel_events.get(job_id)
    return ev.is_set() if ev else False


def _check_cancelled(job_id: str):
    """Raise CancelledError if the job has been cancelled."""
    if is_cancel_requested(job_id):
        raise CancelledError(f"Job {job_id} was cancelled by user")


def get_semaphore() -> asyncio.Semaphore:
    global _analysis_semaphore
    if _analysis_semaphore is None:
        _analysis_semaphore = asyncio.Semaphore(settings.CONCURRENT_ANALYSES)
    return _analysis_semaphore


def register_ws_subscriber(job_id: str, ws):
    if job_id not in _ws_subscribers:
        _ws_subscribers[job_id] = []
    _ws_subscribers[job_id].append(ws)


def unregister_ws_subscriber(job_id: str, ws):
    if job_id in _ws_subscribers:
        _ws_subscribers[job_id] = [w for w in _ws_subscribers[job_id] if w is not ws]
        if not _ws_subscribers[job_id]:
            del _ws_subscribers[job_id]


async def broadcast_ws(job_id: str, message: dict):
    """Broadcast a message to all WebSocket subscribers for a job."""
    from enum import Enum
    # Pre-sanitize: ensure all values are JSON-safe primitives (no Enum remnants)
    safe_message = {}
    for k, v in message.items():
        if isinstance(v, Enum):
            safe_message[k] = str(v.value) if hasattr(v, 'value') else str(v)
        else:
            safe_message[k] = v
    subscribers = _ws_subscribers.get(job_id, [])
    dead = []
    for ws in subscribers:
        try:
            await ws.send_json(safe_message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        unregister_ws_subscriber(job_id, ws)


async def _update_progress(job_id: str, status: str, progress: int, message: str):
    """Update job progress in DB and broadcast via WebSocket.
    If a cancel has been requested, raises CancelledError instead of
    writing a stale progress update that would overwrite the 'cancelled' status."""
    if is_cancel_requested(job_id):
        raise CancelledError(f"Job {job_id} was cancelled by user")
    await database.update_job_status(
        job_id,
        status=status,
        progress=progress,
        progress_message=message,
    )
    await broadcast_ws(job_id, {
        "type": "status",
        "status": status,
        "progress": progress,
        "message": message,
    })
    # Touch heartbeat so it knows we just emitted a real update.
    # Use human-friendly stage names for heartbeat messages.
    hb = _heartbeats.get(job_id)
    if hb:
        _stage_labels = {
            "extracting_frames": "frame extraction",
            "transcribing": "transcription",
            "analyzing_scenes": "scene analysis",
            "generating_summary": "summary generation",
            "detecting_clips": "clip detection",
        }
        stage_label = _stage_labels.get(status, status) if isinstance(status, str) else str(status)
        hb.touch(stage_label)


async def _background_post_processing(job_id: str, transcript: list, orchestrator, job):
    """Run transcript polishing and subtitle translation in background after analysis.

    These are quality-of-life improvements that don't affect clip detection.
    Running them after COMPLETE status saves ~5+ minutes on the critical path.
    """
    # ── Transcript polishing ──
    if settings.AI_TRANSCRIPT_CORRECTION and transcript:
        try:
            from backend.services.compat_stubs import correct_transcript, _adaptive_batch_size
            logger.info("[%s] Background transcript polishing started (%d segments)", job_id, len(transcript))

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "running",
                "message": "Polishing transcript in background...",
            })

            _polish_info = orchestrator.get_text_model_info()
            _batch_size = _adaptive_batch_size(len(transcript))
            _total_batches = -(-len(transcript) // _batch_size)
            _remaining_waves = -(- max(0, _total_batches - 1) // 3)
            _per_batch = 150 if _polish_info.get("is_thinking") else 90
            # Scale timeout with segment count — 955 segments at ~7s/batch of 8 = ~835s
            _estimated_time = (_total_batches * _per_batch) * 1.5
            _correction_timeout = max(180, min(1800, int(_estimated_time) + 60))
            logger.info(
                "[%s] Polishing timeout: %ds (segments=%d, batches=%d, per_batch=%ds)",
                job_id, _correction_timeout, len(transcript), _total_batches, _per_batch,
            )

            # Get Whisper's detected language for the correction prompt
            from backend.services.compat_stubs import _last_detected_language
            whisper_lang = _last_detected_language.get("lang", "")

            # If Whisper used task="translate", the transcript is already English
            # regardless of the source language. Tell the corrector it's English
            # so it doesn't apply Japanese-specific corrections to English text.
            _source = job.language.strip().lower() if job.language else ""
            if not _source:
                _source = whisper_lang
            _whisper_translated = (
                job.subtitle_language
                and job.subtitle_language.strip().lower() == "en"
                and _source and _source != "en"
            )
            correction_lang = "en" if _whisper_translated else whisper_lang

            polished = await asyncio.wait_for(
                correct_transcript(transcript, orchestrator, job_id=job_id, language=correction_lang),
                timeout=_correction_timeout,
            )
            await database.update_job_status(job_id, transcript=list(polished))
            transcript = polished  # Use polished version for translation below

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "complete",
                "message": "Transcript polished",
            })
            logger.info("[%s] Background transcript polishing complete", job_id)
        except Exception as e:
            logger.warning("[%s] Background transcript polishing failed: %s", job_id, e)
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "transcript_polishing",
                "status": "failed",
                "message": f"Polishing skipped: {str(e)[:80]}",
            })

    # ── Subtitle translation ──
    if job.subtitle_language and transcript:
        source_lang = job.language or ""
        target_lang = job.subtitle_language.strip().lower()

        # If source language wasn't detected, try to get it from Whisper
        if not source_lang:
            from backend.services.compat_stubs import _last_detected_language
            source_lang = _last_detected_language.get("lang", "")

        # If Whisper used task="translate", the transcript is already in English.
        # Store it as translated_transcript and skip LLM translation.
        whisper_did_translate = (
            target_lang == "en"
            and source_lang
            and source_lang != "en"
        )

        if whisper_did_translate:
            logger.info(
                "[%s] Whisper native translate produced English text — "
                "storing as translated_transcript (skipping LLM translation)",
                job_id,
            )
            await database.update_job_status(
                job_id,
                translated_transcript=list(transcript),
            )
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "complete",
                "message": "English subtitles ready (Whisper native translation)",
            })
        elif target_lang and target_lang != source_lang:
            from backend.services.translator import translate_segments_with_fallback, SUPPORTED_LANGUAGES
            target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
            source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang) if source_lang else "auto-detected"
            logger.info("[%s] Background subtitle translation: %s → %s (%d segments)",
                        job_id, source_name, target_name, len(transcript))

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "running",
                "message": f"Translating subtitles to {target_name}...",
            })

            # Scale timeout with segment count — allow extra time for model pull + fallback
            _trans_timeout = max(600, len(transcript) * 4)
            try:
                orchestrator.reset_circuit_breaker()
                translated = await asyncio.wait_for(
                    translate_segments_with_fallback(
                        transcript,
                        source_language=source_lang if source_lang else "auto",
                        target_language=target_lang,
                        orchestrator=orchestrator,
                    ),
                    timeout=_trans_timeout,
                )

                changed = sum(1 for t, o in zip(translated, transcript) if t.text != o.text)
                await database.update_job_status(
                    job_id,
                    translated_transcript=list(translated),
                )

                await broadcast_ws(job_id, {
                    "type": "background_task",
                    "task": "subtitle_translation",
                    "status": "complete",
                    "message": f"Subtitles translated to {target_name} ({changed}/{len(translated)} segments)",
                })
                logger.info("[%s] Translated %d/%d segments to %s",
                            job_id, changed, len(translated), target_lang)

            except Exception as e:
                logger.error("[%s] Translation failed: %s", job_id, e)
                await broadcast_ws(job_id, {
                    "type": "background_task",
                    "task": "subtitle_translation",
                    "status": "failed",
                    "message": f"Translation failed: {str(e)[:80]}",
                })


async def run_analysis(job_id: str):
    """Execute the full analysis pipeline for a video job."""
    # Set up cancellation event for this job
    _cancel_events[job_id] = asyncio.Event()
    sem = get_semaphore()

    # Broadcast immediately so the Analysis page shows status while waiting
    # for the semaphore (especially when another analysis is already running)
    await _update_progress(
        job_id, JobStatus.QUEUED, 1,
        "Preparing analysis pipeline...",
    )

    async with sem:
        # Start heartbeat for this job
        hb = _PipelineHeartbeat(job_id, interval=15.0)
        _heartbeats[job_id] = hb
        hb.start()
        # Resolve the job's owner so we can apply their per-user
        # settings (API keys + model picks + whisper choices) over the
        # global config for the lifetime of this analysis. Falls back
        # to a no-op overlay when the job has no owner (legacy data).
        from backend.app.auth.settings_overlay import overlay_user_settings
        _job_for_owner = await database.load_job(job_id)
        _owner_id = getattr(_job_for_owner, "owner_user_id", "") if _job_for_owner else ""
        try:
            async with overlay_user_settings(_owner_id) as _overlay:
                if _overlay:
                    logger.info(
                        "[%s] applied per-user settings overlay: %d keys (owner=%s)",
                        job_id,
                        sum(1 for k, v in _overlay.items() if v),
                        (_owner_id or "")[:8],
                    )
                await _run_analysis_inner(job_id)
        except CancelledError:
            logger.info(f"Job {job_id} cancelled by user")
            await database.update_job_status(
                job_id,
                status=JobStatus.CANCELLED,
                progress_message="Cancelled by user",
            )
            await broadcast_ws(job_id, {
                "type": "cancelled",
                "message": "Job cancelled by user",
            })
        except Exception as e:
            logger.exception(f"Analysis pipeline failed for {job_id}")
            # Preserve last known progress so the frontend can show where it
            # failed instead of the bar collapsing to 0%.
            current = await database.load_job(job_id)
            last_pct = current.progress if current and current.progress else 0
            await database.update_job_status(
                job_id,
                status=JobStatus.FAILED,
                progress=last_pct,
                progress_message=f"Failed at {last_pct}%: {str(e)[:120]}",
                error=str(e),
            )
            await broadcast_ws(job_id, {
                "type": "error",
                "message": f"Analysis failed: {str(e)}",
            })
        finally:
            # Stop heartbeat and clean up
            hb.stop()
            _heartbeats.pop(job_id, None)
            _cancel_events.pop(job_id, None)
            # Even when the job didn't finish cleanly, persist whatever
            # stage timings + warnings we collected so the UI can show
            # where it died. Best-effort: any DB error is logged but
            # never re-raised from the cleanup path.
            try:
                _t, _w = _drain_pipeline_telemetry(job_id)
                if _t or _w:
                    await database.update_job_status(
                        job_id,
                        timings=_t,
                        pipeline_warnings=_w,
                    )
            except Exception as _te:
                logger.info("[%s] telemetry drain failed: %s", job_id, _te)


def _select_dominant_face(faces, last_x=None):
    """Pick the most prominent face from a list of FaceInfo objects.

    Scoring:
      - 35% face size (larger = closer to camera = more important)
      - 25% lip motion (speaking subject is usually the focus)
      - 15% centeredness (cinematographers frame subjects near center)
      - 25% continuity (prefer tracking the same subject as last frame)

    Returns the winning FaceInfo, or None if faces is empty.
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]

    best = None
    best_score = -1.0
    for f in faces:
        # Size score — larger face area = more prominent (max at 5% of frame)
        area = f.width * f.height / 10000.0  # normalize: 100*100=10000
        size_score = min(1.0, area / 0.05)

        # Lip motion score — speaking subjects are the focus
        lip_score = min(1.0, f.lip_aperture / 0.05) if f.lip_aperture > 0 else 0.0

        # Centeredness — subjects near frame center are typically the focus
        center_dist = abs(f.nose_x - 50) / 50.0
        center_score = 1.0 - center_dist

        # Continuity — prefer the subject we were already tracking
        continuity_score = 0.0
        if last_x is not None:
            x_dist = abs(f.nose_x - last_x)
            if x_dist < 15:
                continuity_score = 1.0 - (x_dist / 15.0)

        total = (
            0.35 * size_score
            + 0.25 * lip_score
            + 0.15 * center_score
            + 0.25 * continuity_score
        )
        if total > best_score:
            best_score = total
            best = f

    return best


async def _run_analysis_inner(job_id: str):
    job = await database.load_job(job_id)
    if not job:
        raise RuntimeError(f"Job {job_id} not found")

    video_path = job.file_path
    job_dir = f"/data/uploads/{job_id}"
    frames_dir = os.path.join(job_dir, "frames")
    audio_path = os.path.join(job_dir, "audio.wav")

    # Record pipeline start time for ETA and total duration tracking
    _pipeline_start = _time.monotonic()
    _pipeline_start_iso = datetime.now(timezone.utc).isoformat()
    await database.update_job_status(job_id, analysis_started_at=_pipeline_start_iso)

    # Create a cancel checker bound to this job
    def cancel_check():
        _check_cancelled(job_id)

    custom_prompts = load_prompts()
    orchestrator = AIOrchestrator(
        ws_broadcast=broadcast_ws,
        custom_prompts=custom_prompts,
        cancel_check=cancel_check,
    )

    # ── Pre-flight: validate AI models are reachable ──
    try:
        model_warnings = await orchestrator.validate_models(job_id)
        for w in model_warnings:
            logger.warning("[%s] Model validation: %s", job_id, w)
            # Surface the same warning to the Analysis page banner.
            # ``validate_models`` already prefixes its messages with
            # the warning glyph; normalize to the ``warn:`` prefix the
            # frontend banner expects.
            try:
                msg = w if w.lower().startswith(("warn:", "info:")) else f"warn: {w}"
                _record_pipeline_warning(job_id, msg)
            except Exception:
                pass
    except Exception as e:
        logger.warning("[%s] Model validation failed (non-fatal): %s", job_id, e)

    # ── Pre-flight: record device used for each accelerated stage ──
    # Surfaced on the Analysis page via JobResult.pipeline_warnings so
    # the user can verify "is GPU actually being used for THIS job"
    # without scraping logs. We deliberately produce an INFO-style
    # entry per accelerated stage; the UI tags it differently from
    # actual warnings.
    try:
        _device_lines: list[str] = []
        # Whisper (faster-whisper)
        try:
            from backend.services.compat_stubs import (
                _detect_cuda_available as _whisper_detect_cuda,
            )
            cuda_avail, cuda_count, gpu_name, _ = _whisper_detect_cuda()
            if settings.GPU_ACCELERATION_ENABLED and cuda_avail and cuda_count > 0:
                _device_lines.append(
                    f"info: Whisper will use GPU ({gpu_name or 'CUDA'})"
                )
            else:
                reason = ("disabled by GPU_ACCELERATION_ENABLED=false"
                          if not settings.GPU_ACCELERATION_ENABLED
                          else "no CUDA device detected")
                _device_lines.append(
                    f"info: Whisper will use CPU ({reason})"
                )
        except Exception as _werr:
            _device_lines.append(
                f"info: Whisper device probe failed: {str(_werr)[:120]}"
            )
        # YOLO / object detector
        try:
            from backend.services.compat_stubs import _resolve_yolo_device
            _device_lines.append(
                f"info: Object detector device = {_resolve_yolo_device()}"
            )
        except Exception:
            pass
        # Vision provider — cloud (OpenRouter / Gemini / etc.) doesn't
        # use the local GPU, so make that obvious instead of leaving
        # the user wondering why their RTX 4090 is idle during scene
        # analysis.
        try:
            chain = orchestrator._get_active_chain()
            vision_provider = next(
                (p.provider_name for p in chain if p.supports_vision), None
            )
            if vision_provider == "ollama":
                _device_lines.append(
                    "info: Vision provider = ollama (uses local GPU)"
                )
            elif vision_provider:
                _device_lines.append(
                    f"info: Vision provider = {vision_provider} (cloud — local GPU unused for scene analysis)"
                )
        except Exception:
            pass
        for line in _device_lines:
            _record_pipeline_warning(job_id, line)
    except Exception as e:
        logger.debug("[%s] Device probe skipped: %s", job_id, e)

    def _pipeline_elapsed():
        return _time.monotonic() - _pipeline_start

    _clips_phase_start = [0.0]  # mutable; set when clip detection starts
    # Scene analysis phase-local tracking for accurate ETA
    _scene_phase_start = [0.0]  # set when scene analysis begins
    _scene_recent_timestamps: list[float] = []  # timestamps of recent frame completions

    # Phase timing history — records (end_pct, elapsed_sec) for completed phases.
    # Used to estimate remaining phases more accurately than hardcoded constants.
    _phase_timings: dict[str, float] = {}  # phase_name → elapsed seconds
    _transcription_phase_start = [0.0]  # set when transcription begins

    # Last ETA value for smoothing (avoids jarring jumps)
    _last_eta_value = [0.0]
    _last_eta_time = [0.0]

    def _format_remaining(total_remaining: float) -> str:
        if total_remaining < 60:
            return f" — ~{int(total_remaining)}s remaining"
        m, s = divmod(int(total_remaining), 60)
        return f" — ~{m}m {s}s remaining"

    def _smooth_eta(raw_eta: float) -> float:
        """Smooth ETA to avoid jarring jumps between updates.

        Uses exponential moving average: new_eta = 0.3 * raw + 0.7 * prev.
        Resets if more than 30s have passed since last update (phase change).
        """
        now = _time.monotonic()
        prev = _last_eta_value[0]
        gap = now - _last_eta_time[0]

        if prev <= 0 or gap > 30:
            # First call or phase transition — use raw value
            _last_eta_value[0] = raw_eta
            _last_eta_time[0] = now
            return raw_eta

        # Exponential smoothing — heavily weight previous to reduce jitter
        smoothed = 0.3 * raw_eta + 0.7 * prev
        # Clamp: never increase by more than 20% in a single update
        if smoothed > prev * 1.2 and prev > 30:
            smoothed = prev * 1.05
        _last_eta_value[0] = smoothed
        _last_eta_time[0] = now
        return smoothed

    def _estimate_remaining_phases(current_phase: str) -> float:
        """Estimate time for phases after current_phase using actual timings.

        Phase order: extraction → transcription → scene_analysis → summary → clips → save
        Uses actual measured times for completed phases to estimate remaining ones.
        """
        vid_min = metadata["duration"] / 60 if metadata.get("duration") else 10

        # Rough per-phase estimates as fraction of video duration (minutes)
        # These are defaults; replaced by actual timings when available.
        _default_factors = {
            "extraction": 0.04,       # ~4% of video duration
            "transcription": 0.08,    # ~8% of video duration (GPU)
            "scene_analysis": 0.20,   # ~20% of video duration (Ollama)
            "summary": 0.03,          # ~3% of video duration
            "clips": 0.08,            # ~8% of video duration
            "save": 0.002,            # ~15s regardless
        }
        _phase_order = ["extraction", "transcription", "scene_analysis", "summary", "clips", "save"]

        # Find which phases are after current
        try:
            current_idx = _phase_order.index(current_phase)
        except ValueError:
            current_idx = len(_phase_order)

        remaining_secs = 0.0
        for phase_name in _phase_order[current_idx + 1:]:
            if phase_name in _phase_timings:
                # Use actual timing from a completed phase of similar cost
                remaining_secs += _phase_timings[phase_name]
            else:
                # Estimate from video duration
                remaining_secs += vid_min * 60 * _default_factors.get(phase_name, 0.05)

        return max(15, remaining_secs)

    def _pipeline_eta(current_pct):
        """Estimate remaining time based on progress.

        Uses phase-local estimation during scene analysis (15-62%) and clip
        detection (78-95%) to avoid nonsensical ETA drift from global rate.
        """
        if current_pct <= 2:
            return ""
        elapsed = _pipeline_elapsed()

        # During transcription (15-40% in sequential mode), use transcription-local ETA
        if 15 <= current_pct < 40 and _transcription_phase_start[0] > 0 and _scene_phase_start[0] == 0:
            trans_elapsed = _time.monotonic() - _transcription_phase_start[0]
            if trans_elapsed > 5:
                # Transcription maps to 15-40% range (25 points)
                trans_pct_done = current_pct - 15  # 0-25
                if trans_pct_done > 1:
                    trans_rate = trans_pct_done / trans_elapsed
                    trans_remaining = max(0, (40 - current_pct) / trans_rate)
                    # Add estimate for remaining phases
                    after_trans = _estimate_remaining_phases("transcription")
                    raw_eta = trans_remaining + after_trans
                    return _format_remaining(_smooth_eta(raw_eta))

        # During scene analysis (40-62% in sequential mode, or 15-62% concurrent),
        # use sliding window ETA
        if 15 < current_pct < 62 and _scene_phase_start[0] > 0:
            now = _time.monotonic()
            _scene_recent_timestamps.append(now)
            # Keep last 15 timestamps for sliding window average
            if len(_scene_recent_timestamps) > 15:
                _scene_recent_timestamps[:] = _scene_recent_timestamps[-15:]
            if len(_scene_recent_timestamps) >= 3:
                window = _scene_recent_timestamps
                recent_elapsed = window[-1] - window[0]
                recent_steps = len(window) - 1
                if recent_elapsed > 0:
                    recent_rate = recent_steps / recent_elapsed  # pct-steps per sec
                    # Map current_pct to remaining pct in scene phase
                    phase_remaining_pct = 62 - current_pct
                    # Estimate remaining using recent rate (steps map roughly to pct)
                    scene_remaining = phase_remaining_pct / max(0.001, recent_rate)
                    # Add estimate for remaining phases using actual timings
                    after_scenes = _estimate_remaining_phases("scene_analysis")
                    raw_eta = scene_remaining + after_scenes
                    return _format_remaining(_smooth_eta(raw_eta))

        # During summary (65-75%), use phase-local estimate
        if 65 <= current_pct < 75:
            phase_elapsed = elapsed - sum(_phase_timings.get(p, 0) for p in ["extraction", "transcription", "scene_analysis"])
            if phase_elapsed > 3:
                phase_pct = current_pct - 65  # 0-10
                if phase_pct > 0:
                    phase_rate = phase_pct / max(1, phase_elapsed)
                    summary_remaining = max(0, (75 - current_pct) / phase_rate)
                    after_summary = _estimate_remaining_phases("summary")
                    raw_eta = summary_remaining + after_summary
                    return _format_remaining(_smooth_eta(raw_eta))

        # During clip detection (78-95%), use phase-local ETA
        if 78 <= current_pct <= 95 and _clips_phase_start[0] > 0:
            phase_elapsed = _time.monotonic() - _clips_phase_start[0]
            phase_pct = current_pct - 78  # 0-17 within clip phase
            if phase_pct > 0 and phase_elapsed > 5:
                phase_rate = phase_pct / phase_elapsed
                phase_remaining = max(0, (95 - current_pct) / phase_rate)
                total_remaining = phase_remaining + 15  # ~15s for saving
            else:
                # Not enough data yet — rough estimate from video duration
                vid_min = metadata["duration"] / 60 if metadata.get("duration") else 10
                total_remaining = max(60, vid_min * 8)
            return _format_remaining(_smooth_eta(total_remaining))

        # Default: global pipeline rate (used for transitions, early stages)
        if elapsed <= 0:
            return ""
        rate = current_pct / elapsed
        if rate <= 0:
            return ""
        remaining = max(0, (100 - current_pct) / rate)
        return _format_remaining(_smooth_eta(remaining))

    # Step 1 — Video Metadata (0-5%)
    cancel_check()
    await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 2, "Extracting video metadata...")
    logger.info("[%s] Pipeline started — video: %s", job_id, video_path)
    _log_gpu_memory(job_id, "pipeline start")
    async with _stage_timer(job_id, "metadata"):
        try:
            metadata = await asyncio.wait_for(
                get_video_metadata(video_path), timeout=_METADATA_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Video metadata extraction timed out after {_METADATA_TIMEOUT}s. "
                "The file may be corrupt or on slow storage."
            )
    # Backfill ``width``/``height`` from the legacy ``resolution``
    # string when the probe didn't expose them as first-class fields.
    # Without this, every consumer of ``metadata.get("width", 1920)``
    # silently defaults to 1920x1080 — breaking AutoFlip / reframe
    # pixel math for every non-FullHD source (4K, 720p, vertical
    # phone clips, etc.).
    if not metadata.get("width") or not metadata.get("height"):
        try:
            res = str(metadata.get("resolution") or "")
            if "x" in res:
                w_str, h_str = res.split("x", 1)
                _w = int("".join(ch for ch in w_str if ch.isdigit()) or 0)
                _h = int("".join(ch for ch in h_str if ch.isdigit()) or 0)
                if _w > 0 and _h > 0:
                    metadata["width"] = _w
                    metadata["height"] = _h
        except Exception:
            # Final safety net — leave the existing defaults in place
            # rather than crash the pipeline over a malformed string.
            pass
    await database.update_job_status(
        job_id,
        duration=metadata["duration"],
        resolution=metadata["resolution"],
        fps=metadata["fps"],
        file_size_mb=metadata["file_size_mb"],
    )
    dur_fmt = f"{int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}"
    res = metadata.get("resolution", "?")
    fps_val = metadata.get("fps", 0)
    mb = metadata.get("file_size_mb", 0)

    # Duration tier system — auto-adjusts pipeline based on video length
    from backend.config import get_duration_tier, apply_ollama_overrides
    vid_minutes = metadata["duration"] / 60
    tier = get_duration_tier(metadata["duration"])

    # Detect if Ollama is the primary provider
    _active_chain = orchestrator._get_active_chain()
    _primary_provider = _active_chain[0] if _active_chain else None
    is_ollama_primary = _primary_provider and _primary_provider.provider_name == "ollama"

    if is_ollama_primary:
        tier = apply_ollama_overrides(tier, is_ollama=True)
        logger.info(
            "[%s] Ollama is primary provider — applying overrides: "
            "window=%ds, timeout=%ds, summary=%s, sequential=True",
            job_id, tier.window_duration, tier.per_call_timeout_base, tier.summary_strategy,
        )
        # Warm up models to detect capabilities and VRAM constraints,
        # then immediately unload so Whisper gets exclusive GPU access.
        # Models reload automatically when scene analysis starts.
        # Timeout: skip warmup if it takes too long — models will load lazily.
        try:
            await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 3, "Warming up local AI models...")
            await asyncio.wait_for(_primary_provider.warmup(), timeout=60)
            # Log GPU status after warmup for diagnostics
            if hasattr(_primary_provider, 'log_gpu_status'):
                await _primary_provider.log_gpu_status()
        except asyncio.TimeoutError:
            logger.warning("[%s] Ollama warmup timed out after 60s — skipping (models will load lazily)", job_id)
        except Exception as e:
            logger.warning("[%s] Ollama warmup failed (non-fatal): %s", job_id, e)

        # ── Critical: free GPU for Whisper ──
        # warmup() loaded Ollama models (qwen2.5:3b = 2.3GB) onto the GPU.
        # On a 4GB GPU, this leaves only ~1.5GB for Whisper → silent OOM.
        # Unload now — models reload when pipeline reaches scene analysis.
        try:
            await _update_progress(job_id, JobStatus.EXTRACTING_FRAMES, 4, "Freeing GPU for transcription...")
            await asyncio.wait_for(_primary_provider.unload_models(), timeout=15)
            logger.info("[%s] Ollama models unloaded after warmup — GPU freed for Whisper", job_id)
            await asyncio.sleep(2)  # Let CUDA driver reclaim across containers
        except asyncio.TimeoutError:
            logger.warning("[%s] Ollama model unload timed out after 15s — proceeding anyway", job_id)
        except Exception as e:
            logger.warning("[%s] Failed to unload Ollama after warmup: %s", job_id, e)

    logger.info(
        "[%s] Duration tier: %s (%.1f min) — frame_rate=%ds, summary=%s, "
        "window=%ds, max_clips=%d, max_gaps=%d, ollama=%s",
        job_id, tier.name, vid_minutes, tier.frame_sample_rate,
        tier.summary_strategy, tier.window_duration,
        tier.max_clip_candidates, tier.max_gaps_pass2, is_ollama_primary,
    )

    # Adaptive timeouts based on video duration + provider type + tier
    _EXTRACTION_TIMEOUT = max(600, int(vid_minutes * 60))
    if is_ollama_primary:
        est_windows = max(1, int(metadata["duration"] / tier.window_duration)) if tier.window_duration > 0 else 1
        # Ollama timeouts: generous because local inference is slow but reliable
        _SUMMARY_CLIP_TIMEOUT = max(
            1800,  # Minimum 30 minutes for any video
            est_windows * tier.per_call_timeout_base + 900
        )
        _B64_ENCODE_TIMEOUT = max(300, int(vid_minutes * 10))
        # Parent timeout for transcription+scene: scale with video length.
        # GPU Whisper runs ~10-30x real-time, but CPU fallback (int8 small)
        # can be ~0.5-1x real-time. Use generous multiplier to avoid killing
        # long transcriptions that fell back to CPU.
        _trans_scene_timeout = max(
            1800,  # Minimum 30 minutes
            int(vid_minutes * 150)  # ~2.5min per min of video (covers CPU fallback + vision)
        )
        logger.info(
            "[%s] Ollama-scaled timeouts: extraction=%ds, summary_clip=%ds, "
            "trans_scene=%ds, b64=%ds (%d est. windows, %.1f min video)",
            job_id, _EXTRACTION_TIMEOUT, _SUMMARY_CLIP_TIMEOUT,
            _trans_scene_timeout, _B64_ENCODE_TIMEOUT, est_windows, vid_minutes,
        )
    else:
        _SUMMARY_CLIP_TIMEOUT = max(900, int(vid_minutes * 120))
        _B64_ENCODE_TIMEOUT = max(300, int(vid_minutes * 10))
        # Scale generously to handle CPU Whisper fallback (0.5-1x real-time)
        _trans_scene_timeout = max(3600, int(vid_minutes * 300))
    logger.info(
        "[%s] Adaptive timeouts: extraction=%ds, summary_clip=%ds, b64=%ds (%.1f min video)",
        job_id, _EXTRACTION_TIMEOUT, _SUMMARY_CLIP_TIMEOUT, _B64_ENCODE_TIMEOUT, vid_minutes,
    )

    codec_label = metadata.get("codec_name", "unknown")
    pix_fmt_label = metadata.get("pix_fmt", "")
    codec_info = f" [{codec_label}]" if codec_label else ""
    if pix_fmt_label and pix_fmt_label not in ("yuv420p", "yuvj420p"):
        codec_info += f" ({pix_fmt_label})"
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 5,
        f"Metadata extracted — {res} @ {fps_val}fps, {dur_fmt} duration, {mb:.1f}MB{codec_info}",
    )

    # Disk space pre-check — estimate needed space from video metadata
    disk_usage = shutil.disk_usage("/data")
    # Estimate: audio WAV ~1.8MB/min + frames ~3MB + overhead
    estimated_need_mb = max(50, metadata.get("file_size_mb", 100) * 0.3)
    if disk_usage.free < estimated_need_mb * 1024 * 1024:
        raise RuntimeError(
            f"Insufficient disk space: {disk_usage.free // (1024*1024)}MB free, "
            f"estimated {int(estimated_need_mb)}MB needed. "
            f"Please free space on the /data volume."
        )

    # Broadcast GPU info early so user can see what hardware is available
    try:
        from backend.services.clip_exporter import detect_gpu_capabilities, get_encoder_label
        _gpu = detect_gpu_capabilities()
        _gpu_parts = []
        if _gpu.get("cuda_available"):
            _gpu_parts.append(f"Whisper: CUDA ({_gpu.get('gpu_name', 'GPU')})")
        else:
            # Check if GPU is detected but CUDA isn't available
            _gpu_name = _gpu.get("gpu_name", "")
            if _gpu_name and _gpu_name != "None (CPU only)":
                _gpu_parts.append(f"Whisper: CPU (GPU detected: {_gpu_name} — CUDA runtime not available)")
            else:
                _gpu_parts.append("Whisper: CPU")
        _enc_label = get_encoder_label()
        _gpu_parts.append(f"Encoding: {_enc_label}")
        # Add issues hint if GPU detected but encoder fell back to CPU
        _gpu_issues = _gpu.get("gpu_issues", [])
        if _gpu_issues:
            _gpu_parts.append("(GPU passthrough incomplete — check Settings > Advanced)")
        await broadcast_ws(job_id, {
            "type": "status",
            "status": "processing",
            "progress": 5,
            "message": f"Hardware — {' | '.join(_gpu_parts)}",
        })
    except Exception:
        pass  # Non-critical — don't break pipeline if GPU detection fails

    # Step 2 — Frame + Audio Extraction in parallel (5-15%)
    cancel_check()
    file_mb = metadata.get("file_size_mb", 0)
    duration_min = round(metadata["duration"] / 60, 1)
    size_note = f" ({file_mb:.0f}MB, {duration_min}min)" if file_mb > 50 else ""
    await _update_progress(
        job_id, JobStatus.EXTRACTING_FRAMES, 8,
        f"Extracting frames + audio{size_note}...",
    )

    # Estimate total frames for progress scaling
    _est_frame_rate = settings.FRAME_SAMPLE_RATE
    _est_total_frames = max(50, int(metadata["duration"] / _est_frame_rate)) if metadata["duration"] > 0 else 100

    async def _frame_progress(frames_so_far: int):
        extraction_pct = min(1.0, frames_so_far / _est_total_frames)
        pct = 8 + int(extraction_pct * 6)  # 8% to 14%
        await _update_progress(
            job_id, JobStatus.EXTRACTING_FRAMES, pct,
            f"Extracted {frames_so_far} frames so far{size_note}...",
        )

    # ── Re-analyze cache: if frames + audio already exist on disk AND
    #    the source SHA-256 matches what's recorded on the job, skip
    #    the (expensive) re-extract. Saves 30–60% of total wall time
    #    on iterative re-analysis runs (clip tuning loops).
    #
    # Disable with CLIPAI_FORCE_REEXTRACT=1.
    _force_reextract = os.environ.get("CLIPAI_FORCE_REEXTRACT", "").lower() in ("1", "true", "yes")
    cached_extraction = None
    if not _force_reextract:
        try:
            cached_extraction = await _maybe_use_cached_extraction(
                job_id=job_id,
                video_path=video_path,
                frames_dir=frames_dir,
                audio_path=audio_path,
                expected_sha=getattr(job, "source_sha256", "") or "",
            )
        except Exception as _ce:
            logger.info("[%s] Cache probe failed (%s); falling back to fresh extract", job_id, _ce)

    if cached_extraction is not None:
        frames, scene_cut_timestamps = cached_extraction
        async with _stage_timer(job_id, "frame+audio extraction (cached)"):
            logger.info(
                "[%s] Re-using cached extraction: %d frames, %s audio (SHA matched)",
                job_id, len(frames),
                "with" if os.path.isfile(audio_path) else "missing",
            )
            _record_pipeline_warning(
                job_id,
                "Re-used cached frames + audio (re-analyze without source change)",
            )
    else:
        # Run frame extraction and audio extraction concurrently — both are
        # independent FFmpeg reads of the source video, writing to different outputs.
        async with _stage_timer(job_id, "frame+audio extraction"):
            try:
                extraction_result, _ = await asyncio.wait_for(
                    asyncio.gather(
                        extract_frames(
                            video_path, frames_dir,
                            cancel_check=cancel_check, progress_callback=_frame_progress,
                            video_duration=metadata["duration"],
                            video_codec=metadata.get("codec_name", ""),
                        ),
                        extract_audio(
                            video_path, audio_path,
                            cancel_check=cancel_check,
                            precondition=bool(getattr(settings, "WHISPER_AUDIO_PRECONDITION", True)),
                        ),
                    ),
                    timeout=_EXTRACTION_TIMEOUT,
                )
                frames, scene_cut_timestamps = extraction_result
            except asyncio.TimeoutError:
                logger.error("[%s] Frame+audio extraction timed out after %ds", job_id, _EXTRACTION_TIMEOUT)
                raise RuntimeError(
                    f"Frame and audio extraction timed out after {_EXTRACTION_TIMEOUT // 60} minutes. "
                    "The video file may be very large or the container is under heavy load."
                )
        # Persist a tiny sidecar manifest so future cache hits can
        # reconstruct frame timestamps + scene cuts without ffprobe /
        # ffmpeg re-runs.
        try:
            _write_extraction_manifest(frames_dir, frames, scene_cut_timestamps)
        except Exception:
            pass
        # Hash the source AFTER the first successful extract so re-analyze
        # runs can skip re-extracting. Hashing is best-effort and runs in
        # a thread so it doesn't block the event loop.
        try:
            _sha = await asyncio.to_thread(_hash_file_sha256, video_path)
            if _sha:
                await database.update_job_status(job_id, source_sha256=_sha)
        except Exception as _he:
            logger.info("[%s] source hash failed (%s); cache disabled for next run", job_id, _he)
    total_frames = len(frames)

    # Store scene cut timestamps for shot-boundary-aware tracking
    if scene_cut_timestamps:
        await database.update_job_status(job_id, scene_cut_timestamps=scene_cut_timestamps)
        logger.info("[%s] Stored %d scene cut timestamps for tracking", job_id, len(scene_cut_timestamps))

    # ═══════════════════════════════════════════════════════════════════
    #  ENGINE — ReframeEngine (Perceiver → Planner → Smoother) → Bridge
    # ═══════════════════════════════════════════════════════════════════
    # The reframer engine replaces the legacy face-detection / transcription /
    # camera-solver / clip-scoring stack. ReframeEngine.analyze() runs the
    # full perceive→classify→decide→smooth pipeline; transcription (Whisper)
    # happens inside the Perceiver. The bridge then maps its output onto the
    # Fez RenderPlan / SceneDescription / TranscriptSegment / ClipCandidate
    # contracts the React frontend consumes.
    cancel_check()

    source_width = int(metadata.get("width", 1920) or 1920)
    source_height = int(metadata.get("height", 1080) or 1080)
    video_duration = float(metadata.get("duration", 0.0) or 0.0)
    video_fps = float(metadata.get("fps", 30.0) or 30.0)

    from backend.services.reframer_engine import ReframeEngine
    from backend.services.reframer_bridge import (
        to_fez_render_plan, to_fez_scenes, to_fez_transcript, to_fez_clips,
    )

    _log_gpu_memory(job_id, "pre-reframer")
    _loop = asyncio.get_running_loop()

    def _engine_progress(*pargs):
        """Thread-safe progress relay from the (blocking) reframer engine.

        The Perceiver reports a 0.0-1.0 fraction; map it onto 15-58% of the
        overall pipeline bar and schedule the async update on the event loop.
        """
        frac = 0.0
        if pargs:
            try:
                frac = float(pargs[0])
            except (TypeError, ValueError):
                frac = 0.0
        if frac > 1.5:          # tolerate a 0-100 percentage just in case
            frac /= 100.0
        frac = max(0.0, min(1.0, frac))
        try:
            asyncio.run_coroutine_threadsafe(
                _update_progress(
                    job_id, JobStatus.ANALYZING_SCENES,
                    int(15 + frac * 43),
                    f"Analyzing video — faces, audio, motion ({int(frac * 100)}%)",
                ),
                _loop,
            )
        except Exception:
            pass  # progress is best-effort — never break analysis over it

    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 15,
        "Starting reframer analysis (faces, transcription, motion)...",
    )
    engine = ReframeEngine(video_path, sample_fps=5.0, aspect_ratio="9:16")
    async with _stage_timer(job_id, "reframer_analysis"):
        reframer_plan = await asyncio.to_thread(engine.analyze, _engine_progress)
    perception = engine.perception
    _log_gpu_memory(job_id, "post-reframer")

    # Whisper ran inside the Perceiver — free its VRAM before the VLM stage.
    await _release_whisper_vram(job_id)
    _log_gpu_memory(job_id, "post-whisper-release")

    _n_face_samples = sum(1 for v in (perception.face_timeline or {}).values() if v)
    logger.info(
        "[%s] Reframer analysis complete: %d face samples, %d scene cuts, "
        "%d transcript segments, %d keyframes, %d strategies",
        job_id, _n_face_samples, len(perception.scene_cuts or []),
        len(perception.transcript_segments or []),
        len(reframer_plan.keyframes or []),
        len(reframer_plan.strategy_log or []),
    )

    # ── Bridge — convert reframer output into Fez data contracts ──
    cancel_check()
    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 60,
        "Converting analysis into render plan + scenes...",
    )
    async with _stage_timer(job_id, "bridge_conversion"):
        render_plan = to_fez_render_plan(
            reframer_plan, perception,
            src_w=source_width, src_h=source_height,
            target_w=1080, target_h=1920,
            fps=video_fps, total_duration=video_duration,
        )
        scenes = to_fez_scenes(
            perception, reframer_plan,
            video_path=video_path, frames_dir=frames_dir,
        )
        transcript = to_fez_transcript(
            perception.transcript_segments,
            getattr(perception, "speaker_timeline", None),
        )

    # JobResult has no render_plan field, so persist the plan as a sidecar
    # JSON the /api/jobs/{id}/render_plan endpoint can serve to the NLE editor.
    try:
        with open(os.path.join(job_dir, "render_plan.json"), "w") as _rpf:
            json.dump(render_plan.to_dict(), _rpf)
        logger.info("[%s] render_plan.json written (%d ops)", job_id, len(render_plan.ops))
    except Exception as _rpe:
        logger.warning("[%s] render_plan.json write failed: %s", job_id, _rpe)

    # Seed the speaker-name map so the transcript UI has stable colour keys.
    _speakers = sorted({t.get("speaker", "Speaker 1") for t in transcript})
    speaker_names = {s: s for s in _speakers}

    await database.update_job_status(
        job_id,
        scenes=scenes,
        transcript=transcript,
        speaker_names=speaker_names,
        language=getattr(perception, "detected_language", "") or "",
        default_layout_mode="single",
        scene_cut_timestamps=[round(c / 1000.0, 3) for c in (perception.scene_cuts or [])],
    )
    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 62,
        f"Analysis complete — {len(scenes)} scenes, {len(transcript)} transcript segments",
    )

    # ── VLM summary (kept ai_orchestrator) ──
    cancel_check()
    await _update_progress(
        job_id, JobStatus.GENERATING_SUMMARY, 66, "Generating video summary...",
    )
    summary = None
    async with _stage_timer(job_id, "summary"):
        try:
            _sr = await orchestrator.generate_summary(transcript, scenes, job_id, tier=tier)
            summary = _sr[0] if isinstance(_sr, tuple) else _sr
        except Exception as _se:
            logger.warning(
                "[%s] VLM summary failed (%s) — falling back to transcript summary",
                job_id, _se,
            )
    if summary is None or not has_real_summary_content(summary):
        try:
            summary = build_summary_from_transcript(transcript)
        except Exception:
            summary = VideoSummary(
                overview="Summary unavailable for this video.",
                key_topics=[], tone="neutral",
                estimated_audience="general", content_category="generic",
            )

    # ── Clip detection (reframer clipper) ──
    cancel_check()
    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 80, "Detecting viral clip candidates...",
    )
    clips = []
    async with _stage_timer(job_id, "clip_extraction"):
        try:
            from backend.services.reframer_clipper import ClipExtractor, ClipperConfig
            clipper_config = ClipperConfig.load(
                os.path.join(_PROJECT_ROOT, "clipper_config.json"))
            clip_extractor = ClipExtractor(
                video_path=video_path,
                perception=perception,
                plan=reframer_plan,
                config=clipper_config,
                transcript_segments=perception.transcript_segments,
            )
            raw_clips = await asyncio.to_thread(clip_extractor.run)
            clips = to_fez_clips(raw_clips)
            logger.info("[%s] Clip extraction produced %d clips", job_id, len(clips))
        except Exception as _ce:
            logger.exception("[%s] Clip extraction failed: %s", job_id, _ce)
            clips = []

    # ── Final save ──
    cancel_check()
    _analysis_seconds = round(_time.monotonic() - _pipeline_start, 1)
    await _update_progress(job_id, JobStatus.DETECTING_CLIPS, 98, "Saving results...")
    await database.update_job_status(
        job_id,
        status=JobStatus.COMPLETE,
        progress=100,
        progress_message=f"Analysis complete — {len(clips)} clips, {len(scenes)} scenes",
        summary=summary,
        scenes=scenes,
        transcript=transcript,
        clips=clips,
        analysis_duration_seconds=_analysis_seconds,
        default_layout_mode="single",
    )
    await broadcast_ws(job_id, {
        "type": "complete",
        "message": "Analysis complete",
        "progress": 100,
    })
    logger.info(
        "[%s] Pipeline complete in %.1fs — %d clips, %d scenes, %d transcript segments",
        job_id, _analysis_seconds, len(clips), len(scenes), len(transcript),
    )
