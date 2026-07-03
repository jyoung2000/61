import asyncio
import json
import logging
import os
import shutil
import time as _time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

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

# Project root — the directory that holds the legacy
# ``/app/clipper_config.json`` location. Kept for the migration read
# inside :func:`clipper_config_path`; live writes go to
# ``/data/logs/`` (the mounted location — see _canonical_clipper_config_path).
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clipper_config_path() -> str:
    """Return the on-disk path for ``clipper_config.json``.

    Writes always go to ``/data/logs/clipper_config.json`` — the same
    mount-backed directory ``user_settings.json`` lives in
    (``./data/logs:/data/logs`` in docker-compose.yml), so the file
    survives ``docker compose down``, ``--no-cache`` rebuilds, and
    ``rm -rf clipai`` re-clones. ``/data/clipper_config.json`` and
    ``/app/clipper_config.json`` are checked as legacy read locations
    so existing installs auto-migrate on the next write — neither was
    actually persisted across rebuilds (``/data/`` itself isn't
    mounted; only its child dirs are), which is why users were
    losing the editorial primary + fallback spec on every full
    rebuild even though the file appeared to be on ``/data/``.

    The editorial AI's primary model env-var name
    (``OPENROUTER_EDITORIAL_MODEL``) is in ``user_settings.json``; the
    same primary in spec form plus the fallback spec live here. Both
    files share the same mount now so they persist together.
    """
    canonical = _canonical_clipper_config_path()
    # Pure-read callers want a path that exists. Prefer the canonical
    # mount-backed location; fall back to the legacy paths ONLY when
    # the canonical file hasn't been written yet (so the migration is
    # a one-shot — the next write lands in the canonical location).
    if os.path.exists(canonical):
        return canonical
    ephemeral_path = "/data/clipper_config.json"  # legacy: in-container only
    if os.path.exists(ephemeral_path):
        return ephemeral_path
    legacy_app_path = os.path.join(_PROJECT_ROOT, "clipper_config.json")
    if os.path.exists(legacy_app_path):
        return legacy_app_path
    return canonical


def _canonical_clipper_config_path() -> str:
    """The single source-of-truth write location for clipper_config.json.

    Prefers ``/data/logs/`` (volume-mounted from the host via
    docker-compose) so the file survives rebuilds. Falls back to a
    project-local ``.clipai/`` dir for non-Docker dev runs — matches
    the same resolution ``user_settings.json`` uses in
    ``backend/routers/settings.py:_resolve_data_dir``.
    """
    docker_dir = "/data/logs"
    if os.path.isdir(docker_dir) and os.access(docker_dir, os.W_OK):
        return os.path.join(docker_dir, "clipper_config.json")
    local_dir = os.path.join(_PROJECT_ROOT, ".clipai")
    os.makedirs(local_dir, exist_ok=True)
    return os.path.join(local_dir, "clipper_config.json")


def _gpu_status_message() -> str:
    """Human-readable GPU status line for the live Processing Log.

    The backend logs GPU availability at startup, but the per-job log only ever
    mentioned the GPU on a CPU FALLBACK — so a healthy GPU run looked CPU-silent.
    Broadcast this positively at the stages the user actually watches.
    """
    enabled = bool(getattr(settings, "GPU_ACCELERATION_ENABLED", False))
    name = ""
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            name = _torch.cuda.get_device_name(0)
    except Exception:
        name = ""
    if enabled and name:
        return (f"GPU acceleration active: {name} — video decode, subject detection, "
                "and Whisper transcription run on the GPU.")
    if enabled:
        return ("GPU acceleration is enabled, but no CUDA device is visible — "
                "stages will run on CPU (much slower).")
    return "GPU acceleration is OFF — all stages run on CPU."


def _build_compute_summary(engine, perception) -> dict:
    """Snapshot which device each pipeline stage actually used.

    Read by the frontend's Analysis-page Compute card. Three stages
    covered because each picks GPU vs CPU independently and can
    silently fall to CPU for different reasons:

      * ``frame_extract`` — FFmpeg ``-hwaccel cuda`` for video decode.
        Pulled from ``frame_extractor.LAST_EXTRACTION_LABEL`` which is
        the strategy label of the successful extraction
        (``GPU+scene`` / ``CPU+interval`` / …).
      * ``yolo_world`` — the reframer's scene-aware subject detector.
        Reads ``perception.face_detector._yolo_device`` (set at
        FaceDetector __init__ time via the picker we just instrumented).
      * ``whisper`` — speech transcription via CTranslate2. Reads
        the engine's stashed ``_perceiver_audio_device`` (already
        captured by the engine for GUI display).

    Missing stages are simply omitted — the frontend hides any row
    it doesn't get.

    Shape: ``{stage: {"device": "cuda:0"|"cpu"|..., "detail": str}}``.
    """
    summary: dict = {}

    # Frame extraction
    try:
        from backend.services import frame_extractor
        label = getattr(frame_extractor, "LAST_EXTRACTION_LABEL", None)
        if label:
            on_gpu = label.startswith("GPU")
            summary["frame_extract"] = {
                "device": "cuda:0" if on_gpu else "cpu",
                "detail": f"ffmpeg strategy: {label}",
            }
    except Exception:
        pass

    # YOLO-World subject detector (reframer perceiver)
    try:
        fd = getattr(perception, "face_detector", None)
        yolo_dev = getattr(fd, "_yolo_device", None) if fd is not None else None
        if yolo_dev is not None:
            on_gpu = yolo_dev != "cpu"
            summary["yolo_world"] = {
                "device": "cuda:0" if on_gpu else "cpu",
                "detail": "YOLO-World v2 scene-aware subject detector",
            }
    except Exception:
        pass

    # Whisper
    try:
        whisper_dev = getattr(engine, "_perceiver_audio_device", None)
        if whisper_dev:
            # device_used strings look like "cuda_float16" / "cpu_int8" /
            # "cpu_int8_base" / etc. Split on the first underscore.
            on_gpu = whisper_dev.startswith("cuda")
            compute_type = whisper_dev.split("_", 1)[1] if "_" in whisper_dev else ""
            # Report the EFFECTIVE model that actually loaded (post any VRAM
            # downgrade) so the active config can't disagree with what ran.
            _eff_model = getattr(engine, "_perceiver_audio_model", None)
            _req_model = getattr(engine, "_perceiver_audio_model_requested", None)
            _detail = f"CTranslate2 {compute_type}".strip() if compute_type else "CTranslate2"
            if _eff_model:
                _detail = f"{_eff_model} ({_detail})"
                if _req_model and _req_model != _eff_model:
                    _detail += f" — downgraded from requested '{_req_model}'"
            summary["whisper"] = {
                "device": "cuda:0" if on_gpu else "cpu",
                "model": _eff_model or "",
                "requested_model": _req_model or "",
                "detail": _detail,
            }
    except Exception:
        pass

    return summary


def _vram_snapshot(stage: str, job_id=None):
    """Per-stage VRAM ledger entry (audit Phase 3.5). Never raises."""
    try:
        from backend.services.vram_ledger import snapshot
        snapshot(stage, job_id)
    except Exception:
        pass


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


async def _free_editorial_vram_before_local_clips(job_id: str, orchestrator) -> None:
    """Evict the editorial Ollama LLM before LOCAL clip detection loads its
    vision model, so the two never share a 4 GB GPU.

    In Offline Mode the pipeline runs the editorial AI (polish → translate →
    summary) on a local Ollama LLM (qwen2.5:3b ≈ 2.3 GB) and then detects clips
    with a local Ollama vision model (moondream ≈ 1.7 GB). On a GTX 1650 4 GB
    both can't be resident at once. The editorial stages are finished by the
    time clip detection starts, so unload the editorial model now (keep_alive=0)
    and flush the torch allocator — the vision model then loads into a clean GPU
    instead of racing Ollama's lazy LRU eviction (which can spill to CPU/OOM).

    No-op when clip detection is running in the cloud, or when there's no local
    Ollama model loaded. Best-effort: never blocks the pipeline.
    """
    if settings.resolve_ai_source("clip") != "local":
        return
    try:
        if orchestrator is not None and hasattr(orchestrator, "unload_local_models"):
            await asyncio.wait_for(orchestrator.unload_local_models(), timeout=15)
            logger.info(
                "[%s] Editorial Ollama model unloaded — GPU freed for the local "
                "clip-detection vision model", job_id,
            )
    except asyncio.TimeoutError:
        logger.warning("[%s] Editorial model unload timed out — proceeding", job_id)
    except Exception as e:
        logger.debug("[%s] Editorial model unload skipped: %s", job_id, e)

    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[%s] pre-clip torch cleanup skipped: %s", job_id, e)

    # Let the CUDA driver reclaim the freed VRAM across containers (the app and
    # Ollama share one GPU) before the vision model asks for it.
    await asyncio.sleep(1)


def _whisper_native_translate_segments(video_path: str, source_lang: str,
                                       glossary: dict | None = None,
                                       source_segments: list | None = None) -> list:
    """Run Whisper's native audio→English translate task (offline, no LLM).

    Returns a list of ``TranscriptSegment`` (English, with Whisper's own
    audio-aligned timing), or ``[]`` when unavailable. Synchronous — Whisper is
    blocking, so call it via ``asyncio.to_thread``. Loads the configured Whisper
    model (reusing the cached engine when present; ``try_load`` / the translate
    method handle the GPU→CPU fallback on low-VRAM cards).

    This is the preferred offline path for non-English → English (matches
    repo-60): a single ASR-translate pass that avoids the transcribe-then-
    translate double-error and never touches an LLM. The AI model only polishes
    the result downstream.

    Whisper's translate task does NOT diarize, so each English cue is assigned
    the speaker of the diarized ``source_segments`` cue it overlaps most (both
    are wall-clock timestamps from the same audio); it falls back to
    ``"Speaker 1"`` when no source overlap is found.
    """
    from backend.services.reframer_audio import AudioIntelligence
    from backend.models import TranscriptSegment

    model_name = (getattr(settings, "WHISPER_MODEL", "small") or "small")
    # Was the engine already loaded? If so, try_load reuses it and the translate
    # pass needs only inference VRAM (no reload), so it can stay on the GPU at a
    # lower free-VRAM floor — the whole point of reuse on small cards.
    _was_cached = getattr(AudioIntelligence, "_cached_engine", None) is not None
    ai = AudioIntelligence(model_name=model_name)
    if not ai.try_load():
        logger.warning("Whisper native translate: engine failed to load (model=%s)", model_name)
        return []
    raw = ai.whisper_translate(video_path, source_lang=source_lang, reuse_loaded=_was_cached)
    if not raw:
        return []

    # Force any leaked source-term → target-term glossary fixes into the English
    # output (best-effort; reuses the NMT post-processor). Whisper translated
    # straight from audio, so there is no aligned per-cue source text — we treat
    # the English line as both sides, which only swaps source terms that leaked.
    try:
        from backend.services.nmt_translator import apply_glossary
    except Exception:
        apply_glossary = None

    # Index the diarized source cues (start, end, speaker) so each translated
    # cue can inherit a speaker label by max time-overlap.
    src_spans = []
    for s in (source_segments or []):
        ss = s.get("start") if isinstance(s, dict) else getattr(s, "start", None)
        se = s.get("end") if isinstance(s, dict) else getattr(s, "end", None)
        sp = s.get("speaker") if isinstance(s, dict) else getattr(s, "speaker", None)
        if ss is not None and se is not None and sp:
            src_spans.append((float(ss), float(se), sp))

    def _speaker_for(a, b):
        best, best_ov = "Speaker 1", 0.0
        for (ss, se, sp) in src_spans:
            ov = min(b, se) - max(a, ss)
            if ov > best_ov:
                best_ov, best = ov, sp
        return best

    out: list = []
    for seg in raw:
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        if glossary and apply_glossary:
            try:
                txt = apply_glossary(txt, txt, glossary)
            except Exception:
                pass
        start = float(seg.get("start_sec", seg.get("start", 0.0)) or 0.0)
        end = float(seg.get("end_sec", seg.get("end", 0.0)) or 0.0)
        # Carry Whisper's WORD timestamps through. Whisper-native emits a few
        # long, multi-sentence cues; the downstream sentence resegmentation uses
        # these word times to split them at ACCURATE boundaries (instead of the
        # char-length proportional guess it falls back to with no word timing).
        out.append(TranscriptSegment(
            text=txt, start=start, end=end, speaker=_speaker_for(start, end),
            words=seg.get("words") or None))
    return out


def _gpu_free_vram_gb() -> float:
    """Free CUDA VRAM in GiB (0.0 when there's no GPU / torch is unavailable)."""
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            return free_bytes / 1_073_741_824
    except Exception:
        pass
    return 0.0


def _whisper_engine_cached() -> bool:
    """True when the reframer's Whisper engine is still loaded (so a Whisper-
    native translate can REUSE it with no second load). The analyze stage keeps
    it loaded — instead of releasing it early — exactly when an English translate
    is pending, so this is the signal that reuse is possible."""
    try:
        from backend.services.reframer_audio import AudioIntelligence
        return getattr(AudioIntelligence, "_cached_engine", None) is not None
    except Exception:
        return False


async def _get_whisper_en_timing_reference(
    video_path: str, source_lang: str, source_segments: list, job_id: str,
) -> list:
    """Run a Whisper-native English pass PURELY as a timing reference for the
    hybrid word-timing projection (its TEXT is never used — only ``.words``).

    Gated on VRAM: only runs when the transcription engine is still cached (free
    reuse) or enough VRAM is free; otherwise returns ``[]`` so the caller
    degrades to tier B. Fail-soft on timeout / OOM / any error — never blocks the
    job, never eats the CPU path."""
    if not getattr(settings, "HYBRID_WORD_TIMING_ENABLED", True):
        return []
    if not video_path:
        return []
    cached = _whisper_engine_cached()
    free_gb = _gpu_free_vram_gb()
    floor = float(getattr(settings, "HYBRID_WHISPER_REF_MIN_FREE_GB", 3.0))
    if not cached and free_gb < floor:
        logger.info(
            "[%s] Hybrid timing: skipping Whisper-EN reference (engine not cached, "
            "only %.1f GB free < %.1f GB floor) — degrading to tier B",
            job_id, free_gb, floor)
        return []
    timeout = float(getattr(settings, "HYBRID_WHISPER_REF_TIMEOUT_S", 1800.0))
    try:
        ref = await asyncio.wait_for(
            asyncio.to_thread(
                _whisper_native_translate_segments, video_path, source_lang,
                None, source_segments),
            timeout=timeout,
        )
        logger.info(
            "[%s] Hybrid timing: Whisper-EN reference ready (%d cue(s), reuse=%s)",
            job_id, len(ref or []), cached)
        return ref or []
    except asyncio.TimeoutError:
        logger.warning(
            "[%s] Hybrid timing: Whisper-EN reference timed out (%.0fs) — "
            "degrading to tier B", job_id, timeout)
        return []
    except Exception as e:
        logger.warning(
            "[%s] Hybrid timing: Whisper-EN reference failed (%s) — degrading to "
            "tier B", job_id, e)
        return []


def _timeline_coverage_s(segments) -> float:
    """Total seconds of timeline covered by the cues, merging overlaps.

    Used to compare how much of the audio a translation actually covers. Whisper's
    audio→English *translate* task skips/merges non-speech (especially singing),
    so on music/lyric-heavy videos it can cover far less of the timeline than the
    source transcription did — leaving big untranslated gaps. Comparing coverage
    catches that so we can prefer dense offline NMT instead."""
    spans = []
    for s in (segments or []):
        if isinstance(s, dict):
            a = s.get("start", s.get("start_sec"))
            b = s.get("end", s.get("end_sec"))
        else:
            a = getattr(s, "start", getattr(s, "start_sec", None))
            b = getattr(s, "end", getattr(s, "end_sec", None))
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if b > a:
            spans.append((a, b))
    if not spans:
        return 0.0
    spans.sort()
    total = 0.0
    cur_s, cur_e = spans[0]
    for a, b in spans[1:]:
        if a <= cur_e:
            cur_e = max(cur_e, b)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = a, b
    total += cur_e - cur_s
    return total


def _resolve_translation_model_override(orchestrator) -> Optional[str]:
    """The dedicated translation model for the ACTIVE editorial provider, or
    None when it isn't set or already equals the editorial model.

    Lets the LLM-first translation use OLLAMA_TRANSLATION_MODEL /
    OPENROUTER_TRANSLATION_MODEL (e.g. qwen3:4b-instruct-2507) while editorial /
    SEO / summaries keep the fast editorial model."""
    try:
        info = orchestrator.get_editorial_model_info() if orchestrator else {}
        prov = (info.get("provider") or "").lower()
        editorial = (info.get("model") or "").strip()
    except Exception:
        prov, editorial = "", ""
    if prov == "ollama":
        xlate = (getattr(settings, "OLLAMA_TRANSLATION_MODEL", "") or "").strip()
    elif prov == "openrouter":
        xlate = (getattr(settings, "OPENROUTER_TRANSLATION_MODEL", "") or "").strip()
    else:
        xlate = ""
    if not xlate or xlate == editorial:
        return None
    return xlate


def _resolve_polish_model_override(orchestrator) -> Optional[str]:
    """The model subtitle polishing should run on.

    Polishing belongs to the SUBTITLE pipeline, so by default it runs on the
    dedicated translation model (the multilingual model that also translates the
    subtitles) — making the translation AI the single brain that owns subtitles
    end-to-end, while the editorial model stays reserved for SEO + summaries.
    Returns None (→ editorial model, legacy behavior) when the flag is off or no
    separate translation model is configured."""
    # Explicit polish model (the "Recommended for subtitle polish" picker,
    # audit Phase 4.2) outranks the translation-model default.
    explicit = (getattr(settings, "SUBTITLE_POLISH_MODEL", "") or "").strip()
    if explicit:
        return explicit
    if not getattr(settings, "SUBTITLE_POLISH_USES_TRANSLATION_MODEL", True):
        return None
    return _resolve_translation_model_override(orchestrator)


async def translate_subtitles(segments, source_lang, target_lang, *, video_path=None,
                              glossary=None, orchestrator=None, status_callback=None,
                              job_id=None, whisper_timeout=None, nmt_timeout=None):
    """Translate subtitle segments — the shared translation entry point.

    Tries the editorial LLM first when one is configured and
    ``TRANSLATION_PREFER_LLM`` is set (the default): it translates the source
    text-to-text, 1:1 (every segment keeps its timing + speaker), rendering
    every cue so it never leaves the source language behind. Then, for
    non-English → English, Whisper's native audio→English pass (when
    ``video_path`` is available); otherwise — or when an engine yields nothing /
    still-source-language output — the offline NMT engines (Opus-MT / NLLB). Set
    ``TRANSLATION_PREFER_LLM=False`` for a genuinely offline-only run
    (Whisper-native / NMT only). Returns ``(segments, engine)`` with ``engine``
    in ``{"llm", "whisper", "nmt"}``. Raises ``TranslationFailedError`` (from the
    NMT layer) when the offline fallback can't complete.

    Single source of truth shared by the pipeline and the manual re-translate
    endpoint so they can't drift. When ``job_id`` is given, Whisper VRAM is
    released after a Whisper pass and before a local-NMT load (the low-VRAM
    ordering the 4 GB-GPU pipeline needs).

    EVERY path (LLM / Whisper-native / NMT) runs the per-cue
    ``_llm_cleanup_untranslated`` QA before returning, so no engine can ship a
    track with source-language stragglers (the cleanup previously ran only on
    the NMT path, which let the LLM/Whisper paths leak source-language cues).
    """
    from backend.services.translator import (
        translate_segments_with_fallback, _resolve_translation_engine,
    )

    src = (source_lang or "").lower()
    tgt = (target_lang or "").lower()

    # ── PRIMARY: editorial-LLM translation of the SOURCE transcript ──────────
    # Translate the source text-to-text, 1:1 (every segment keeps its timing +
    # speaker). When a capable LLM is configured — the same orchestrator that
    # already writes the summary / SEO / polish — this is the reliable, COMPLETE
    # path: it translates dialogue, lyrics AND narration and never leaves the
    # source language behind. Whisper's translate task does the opposite (it
    # silently transcribes hard/music segments in the source), which is what
    # produced the half-Japanese "translated" track. Falls through to Whisper-
    # native / offline NMT when no LLM is available or it returns still
    # source-language. ``TRANSLATION_PREFER_LLM`` (default on) gates it.
    #
    # Quality mode (Task 5) routes translation through the offline NMT router's
    # CPU quality path instead, so skip this orchestrator-LLM preemption (and
    # the Whisper-native one below) when it is active.
    from backend.services.translator import translation_quality_mode_active
    _quality_mode = translation_quality_mode_active()
    if _quality_mode:
        logger.info("Translation quality mode active — skipping the LLM-first / "
                    "Whisper-native paths; using the offline CPU quality path")
    if (orchestrator is not None
            and not _quality_mode
            and getattr(settings, "TRANSLATION_PREFER_LLM", True)
            and segments and tgt and tgt != src):
        try:
            from backend.services.translator import (
                translate_via_llm, fraction_untranslated,
            )
            if job_id:
                # The LLM runs via API/Ollama — free the reframer's Whisper VRAM now.
                await _release_whisper_vram(job_id)
            # Route translation through the dedicated translation model
            # (OLLAMA_TRANSLATION_MODEL / OPENROUTER_TRANSLATION_MODEL, e.g.
            # qwen3:4b-instruct-2507) when it differs from the editorial model,
            # so translation gets the higher-quality model while editorial/SEO
            # keep the fast default. None when unset or same as editorial.
            _xlate_model = _resolve_translation_model_override(orchestrator)
            if status_callback:
                try:
                    _msg = ("Translating subtitles with %s…" % _xlate_model
                            if _xlate_model else
                            "Translating subtitles with the editorial model…")
                    await status_callback(_msg)
                except Exception:
                    pass
            _llm = await asyncio.wait_for(
                translate_via_llm(segments, source_lang, target_lang, orchestrator,
                                  glossary=glossary, job_id=job_id or "",
                                  status_callback=status_callback,
                                  model_override=_xlate_model),
                timeout=(nmt_timeout or 1800),
            )
            if _llm:
                _resid = fraction_untranslated(_llm, target_lang)
                if _resid < 0.20:
                    # Universal QA: the LLM can still echo/leave a few hard cues in
                    # the source language. Re-translate those per-cue so the LLM
                    # path can't ship half-source either (this cleanup previously
                    # ran ONLY on the NMT path, so LLM/Whisper leftovers slipped
                    # through). Fail-soft + bounded; a clean draft is a no-op.
                    _llm = await _llm_cleanup_untranslated(
                        _llm, source_lang, target_lang, orchestrator, glossary, job_id)
                    logger.info("Translate via editorial LLM: %d segments → %s (complete)",
                                len(_llm), target_lang)
                    return _llm, "llm"
                logger.warning("LLM translation left %.0f%% in the source language "
                               "— trying other engines.", 100 * _resid)
        except Exception as _le:
            logger.warning("LLM translation unavailable (%s) — falling back", _le)

    _want_whisper = (
        bool(video_path)
        and not _quality_mode
        and getattr(settings, "WHISPER_TRANSLATE_TO_EN", True)
        and tgt == "en" and src not in ("en", "english")
    )
    # Whisper-native is a SECOND full ASR pass — only worth it when it can run on
    # the GPU. On a low-VRAM card (e.g. 4 GB, where VRAM is freed for the next
    # stages) it falls back to CPU and takes ~30 min for a 25-min video, then
    # times out; meanwhile offline NMT (NLLB int8) loads in the freed VRAM and
    # finishes in well under a minute. So gate on free GPU memory and otherwise
    # go straight to NMT — never eat the CPU path.
    _WHISPER_MIN_FREE_GB = float(getattr(settings, "WHISPER_TRANSLATE_MIN_FREE_GB", 4.0))
    _free_gb = _gpu_free_vram_gb() if _want_whisper else 0.0
    # Reuse: when the transcription Whisper model is STILL loaded (the analyze
    # stage deferred its VRAM release for exactly this), the translate pass
    # reuses it with NO second load — so the reload gate doesn't apply. This is
    # what makes Whisper-native →English usable on small (4 GB) cards. The
    # per-pass pre-flight in whisper_translate still picks GPU vs CPU from the
    # free VRAM at that moment, and an OOM there falls back cleanly.
    _engine_cached = _whisper_engine_cached() if _want_whisper else False
    use_whisper = _want_whisper and (_engine_cached or _free_gb >= _WHISPER_MIN_FREE_GB)
    if _want_whisper and not use_whisper:
        logger.info(
            "Whisper-native translate skipped (%.1f GB GPU free < %.1f GB to reload "
            "on GPU, and no cached engine to reuse) — using offline NMT instead.",
            _free_gb, _WHISPER_MIN_FREE_GB,
        )
    elif _want_whisper and _engine_cached and _free_gb < _WHISPER_MIN_FREE_GB:
        logger.info(
            "Whisper-native translate: REUSING the already-loaded model "
            "(%.1f GB free, no second load).", _free_gb)
    if use_whisper:
        if status_callback:
            try:
                await status_callback("Translating audio directly to English (Whisper, offline)…")
            except Exception:
                pass
        try:
            _wt = await asyncio.wait_for(
                asyncio.to_thread(_whisper_native_translate_segments,
                                  video_path, source_lang, glossary, segments),
                timeout=(whisper_timeout or 1800),
            )
        except Exception as _e:
            _wt = None
            logger.warning("Whisper native translate failed (%s) — falling back to offline NMT", _e)
        if _wt:
            # Coverage guard: Whisper's TRANSLATE task skips/merges non-speech
            # (especially singing), so on music/lyric-heavy videos it can emit far
            # less timeline coverage than the source transcription captured —
            # leaving long untranslated gaps that read as "still Japanese". When
            # Whisper-native covers materially less of the audio than the source,
            # discard its sparse output and use dense offline NMT on the full
            # source so every source cue gets a translation.
            _src_cov = _timeline_coverage_s(segments)
            _wt_cov = _timeline_coverage_s(_wt)
            _min_ratio = float(getattr(settings, "WHISPER_TRANSLATE_MIN_COVERAGE", 0.6))
            if _src_cov > 0 and _wt_cov < _min_ratio * _src_cov:
                logger.warning(
                    "Whisper-native translate covered only %.0fs of %.0fs source "
                    "speech (%.0f%% < %.0f%% floor) — discarding sparse output and "
                    "using offline NMT on the full source for complete subtitles.",
                    _wt_cov, _src_cov,
                    (100.0 * _wt_cov / _src_cov) if _src_cov else 0.0,
                    100.0 * _min_ratio,
                )
                _wt = None
        if _wt:
            # Language-purity gate: Whisper's translate task leaves music /
            # narration / hard segments in the SOURCE language on mixed content
            # (a half-Japanese "translated" track). Reject that and use offline
            # NMT on the full source, which translates every cue.
            from backend.services.translator import fraction_untranslated
            _resid = fraction_untranslated(_wt, target_lang)
            if _resid >= 0.20:
                logger.warning(
                    "Whisper-native translate left %.0f%% of cues in the source "
                    "language (mixed output) — using offline NMT instead.",
                    100 * _resid)
                _wt = None
        if _wt:
            if job_id:
                await _release_whisper_vram(job_id)
            # Universal QA: Whisper's translate task leaves the odd hard/music cue
            # in the source language even after the purity gate above (which only
            # rejects a WHOLESALE half-source draft). Re-translate the stragglers
            # per-cue so this path can't ship source either. Fail-soft.
            _wt = await _llm_cleanup_untranslated(
                _wt, source_lang, target_lang, orchestrator, glossary, job_id)
            return _wt, "whisper"

    # Offline NMT path. Free Whisper VRAM first when a local NMT engine will load
    # (so it doesn't stack on top of the reframer's Whisper on a 4 GB card).
    if job_id:
        try:
            if _resolve_translation_engine(src or "auto", tgt) in ("nllb", "opus-mt"):
                await _release_whisper_vram(job_id)
                _vram_snapshot("pre_offline_nmt", job_id)
        except Exception:
            pass
    _nmt_coro = translate_segments_with_fallback(
        segments,
        source_language=source_lang or "auto",
        target_language=target_lang,
        orchestrator=orchestrator,
        glossary=glossary,
        status_callback=status_callback,
    )
    out = (await asyncio.wait_for(_nmt_coro, timeout=nmt_timeout)
           if nmt_timeout else await _nmt_coro)
    # Belt-and-suspenders completeness: the offline NMT (especially FuguMT on
    # colloquial, un-punctuated run-on speech) can leave a chunk of cues in the
    # source language. The offline router is LLM-free by design, but THIS entry
    # point is allowed to use the editorial LLM — so when one is available, clean
    # up only the still-source cues with it instead of shipping half-source subs.
    # Fail-soft + bounded (leftover cues only).
    out = await _llm_cleanup_untranslated(
        out, source_lang, target_lang, orchestrator, glossary, job_id)
    return out, "nmt"


async def _llm_cleanup_untranslated(segments, source_lang, target_lang,
                                    orchestrator, glossary, job_id):
    """Re-translate cues the offline NMT left in the source language, ONE CUE AT
    A TIME via a PLAIN-TEXT LLM call.

    FuguMT/NLLB reliably leave long colloquial run-on cues untranslated, and the
    JSON-array LLM path is brittle with small local models — qwen2.5:3b returned
    unparseable batched output and bailed, so the subtitles shipped ~25%
    Japanese. A per-cue plain-text request ("translate this line; reply with only
    the translation") is the most robust local-model call: no JSON to misparse
    and no cross-cue alignment risk, so it recovers the leftovers FuguMT
    couldn't. Bounded (cue cap + wall-clock budget) and fully fail-soft — any
    failure keeps the existing (source) cue. Returns ``segments`` unchanged when
    there's nothing to fix / no orchestrator / a CJK target."""
    if not orchestrator or not segments:
        return segments
    try:
        from backend.services.translator import (
            _cjk_ratio, _CJK_LANGS, fraction_untranslated, _is_untranslated,
        )
        from backend.models import TranscriptSegment
    except Exception:
        return segments
    _tgt = (target_lang or "").strip().lower().split("-")[0]
    if _tgt in _CJK_LANGS:
        return segments  # a CJK target legitimately contains CJK

    def _txt(s):
        return (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")) or ""

    try:
        # _is_untranslated covers BOTH CJK script and (for Japanese sources)
        # transliterated ROMAJI — the old _cjk_ratio-only filter never even
        # selected romaji cues ("Nametotte ageru kara.") for cleanup, so they
        # shipped in the English track.
        leftover_idx = [i for i, s in enumerate(segments)
                        if _is_untranslated(_txt(s), source_lang)]
        if not leftover_idx:
            return segments

        _cap = int(getattr(settings, "TRANSLATION_LLM_CLEANUP_MAX_CUES", 500) or 0)
        _budget = float(getattr(settings, "TRANSLATION_LLM_CLEANUP_BUDGET_S", 1200) or 0)
        if _cap <= 0:
            return segments  # cleanup disabled
        if len(leftover_idx) > _cap:
            logger.warning(
                "[%s] %d untranslated cue(s) exceed the LLM-cleanup cap (%d) — "
                "skipping; transcript may be too garbled to recover cue-by-cue",
                job_id, len(leftover_idx), _cap)
            return segments

        _tgt_name = "English" if _tgt == "en" else (target_lang or "English")
        # Recurring-name glossary so each cue's re-translation renders names the
        # SAME way the main pass did (consistency — no "Mika"/"Mikka"/"Mikako"
        # drift, no coined name turned into an ordinary word). Same auto-extractor
        # translate_via_llm uses; empty string when nothing recurs.
        _terms_block = ""
        try:
            from backend.services.glossary import build_translation_glossary_block
            _terms_block = build_translation_glossary_block(
                segments, source_lang, _tgt_name)
        except Exception:
            _terms_block = ""
        logger.info(
            "[%s] Offline NMT left %d/%d cue(s) in the source language (%.0f%%) — "
            "per-cue LLM cleanup (budget %.0fs)",
            job_id, len(leftover_idx), len(segments),
            100 * fraction_untranslated(segments, target_lang), _budget)
        # Accurate, visible status so the (multi-minute) cleanup doesn't look idle.
        try:
            await _update_progress(
                job_id, JobStatus.TRANSLATING, 68,
                f"Recovering {len(leftover_idx)} untranslated subtitle(s) with the LLM…",
                heartbeat_label="subtitle translation")
        except Exception:
            pass

        _t0 = _time.monotonic()
        _fixed = 0
        # The same hallucinated/looped run-on repeats across many cues — translate
        # each UNIQUE source text once and reuse it, so the cleanup is fast and
        # consistent. "" caches an attempted-but-failed text so its duplicates
        # are skipped instead of re-tried.
        _cache: dict[str, str] = {}
        for i in leftover_idx:
            cur = segments[i]
            src_text = _txt(cur).strip()
            if not src_text:
                continue
            t = _cache.get(src_text)
            if t is None:  # not attempted yet
                if _budget > 0 and _time.monotonic() - _t0 > _budget:
                    logger.warning("[%s] LLM cleanup budget reached (%d unique done)",
                                   job_id, len(_cache))
                    break
                prompt = (
                    (_terms_block or "")
                    + f"Translate this subtitle line into natural, fluent {_tgt_name}. "
                    f"Reply with ONLY the {_tgt_name} translation — no quotes, no "
                    f"notes, do not repeat the original.\n\n{src_text}")
                # ``except Exception`` only — a real cancel (CancelledError, a
                # BaseException) still propagates and stops the run.
                resp = None
                try:
                    resp = await orchestrator.text_completion(
                        prompt, timeout=60, job_id=job_id or "", skip_circuit_breaker=True)
                except Exception:
                    resp = None
                if resp is None:
                    # Local chain dead (the exact state that shipped romaji
                    # last run) — same cloud safety net polish uses.
                    try:
                        from backend.services.transcript_polisher import (
                            _cloud_polish_completion)
                        resp = await _cloud_polish_completion(prompt, 60)
                    except Exception:
                        resp = None
                if resp is None:
                    _cache[src_text] = ""
                    continue
                t = (resp or "").strip().strip('"').strip()
                low = t.lower()
                for _pref in ("translation:", "english:", "translation -",
                              "english -", "translation —", "english —"):
                    if low.startswith(_pref):
                        t = t[len(_pref):].strip()
                        break
                if not t or _is_untranslated(t, source_lang):
                    # echoed source (CJK OR romaji) / failed → keep existing cue
                    _cache[src_text] = ""
                    continue
                _cache[src_text] = t
            elif t == "":  # previously attempted and failed
                continue
            if glossary:
                for k, v in glossary.items():
                    ks, vs = (k or "").strip(), (v or "").strip()
                    if ks and vs and ks in t:
                        t = t.replace(ks, vs)
            if isinstance(cur, dict):
                segments[i] = {**cur, "text": t}
            else:
                segments[i] = TranscriptSegment(
                    text=t,
                    start=float(getattr(cur, "start", 0.0) or 0.0),
                    end=float(getattr(cur, "end", 0.0) or 0.0),
                    speaker=getattr(cur, "speaker", None) or "Speaker 1")
            _fixed += 1
        logger.info(
            "[%s] LLM cleanup recovered %d/%d cue(s); %.0f%% source-script remaining",
            job_id, _fixed, len(leftover_idx),
            100 * fraction_untranslated(segments, target_lang))
        return segments
    except Exception as _e:  # noqa: BLE001 — cleanup is best-effort
        logger.info("[%s] LLM leftover cleanup skipped (%s)", job_id, _e)
        return segments


# Back-compat alias — historical name. This path is LLM-first now, not
# offline-only.
translate_offline = translate_subtitles


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
        vision_model = provider._primary_model
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
    resolve_sample_fps,
    resolve_clip_progress,
    translation_progress_pct,
)
from backend.services import pipeline_checkpoint  # noqa: E402

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


def set_heartbeat_stage(job_id: str, stage: str) -> None:
    """Re-label the keepalive for a sub-phase that doesn't emit progress.

    The heartbeat's stage label is normally derived from the job STATUS via
    ``_update_progress``. Phases that run AFTER the clip stage but before the
    COMPLETE save (Auto-SEO, caption refresh) keep the status on
    ``detecting_clips`` and emit no progress, so the 15s keepalive kept saying
    "Still processing... (clip detection — Xm elapsed)" for minutes while it was
    actually generating SEO. Calling this once at the top of such a phase makes
    the keepalive name the real work and reset its elapsed clock."""
    hb = _heartbeats.get(job_id)
    if hb:
        hb.touch(stage)


def is_job_analyzing(job_id: str) -> bool:
    """True while ``run_analysis`` is actively processing this job.

    A heartbeat is registered for the whole run and removed in the ``finally``,
    so this is a cheap in-memory signal. Used by the file-serving endpoint to
    avoid kicking off a CPU-heavy browser-preview transcode that would contend
    with the live offline pipeline (which is what stalled the preview player)."""
    return job_id in _heartbeats

# Strong references to in-flight background tasks so the asyncio loop
# doesn't garbage-collect them mid-flight (transcript polish + subtitle
# translation, scheduled after analysis completes).
_background_tasks: set[asyncio.Task] = set()

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
    # Persist the event to the job's durable log BEFORE broadcasting, so the
    # PROCESSING LOG survives tab reloads / new devices / container restarts and
    # is captured even when no client is currently subscribed. Best-effort,
    # synchronous (no await → atomic), and filtered/throttled internally.
    try:
        from backend.services.job_events import append_job_event
        append_job_event(job_id, safe_message)
    except Exception:
        pass
    subscribers = _ws_subscribers.get(job_id, [])
    dead = []
    for ws in subscribers:
        try:
            await ws.send_json(safe_message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        unregister_ws_subscriber(job_id, ws)


PIPELINE_STAGES = [
    {"id": "queue",          "label": "Queue",               "status": "queued",             "start_pct": 0,   "end_pct": 1},
    {"id": "metadata",       "label": "Video Metadata",       "status": "extracting_frames",  "start_pct": 2,   "end_pct": 5},
    {"id": "extraction",     "label": "Frame & Audio",        "status": "extracting_frames",  "start_pct": 5,   "end_pct": 14},
    {"id": "face_detection", "label": "Face Detection",       "status": "analyzing_scenes",   "start_pct": 15,  "end_pct": 42},
    {"id": "transcription",  "label": "Audio Transcription",  "status": "analyzing_scenes",   "start_pct": 42,  "end_pct": 56},
    {"id": "diarization",    "label": "Speaker Detection",    "status": "analyzing_scenes",   "start_pct": 56,  "end_pct": 58},
    {"id": "conversion",     "label": "Scene Conversion",     "status": "analyzing_scenes",   "start_pct": 60,  "end_pct": 62},
    {"id": "summary",        "label": "Video Summary",        "status": "generating_summary", "start_pct": 62,  "end_pct": 76},
    {"id": "translation",    "label": "Subtitle Translation", "status": "translating",        "start_pct": 76,  "end_pct": 80},
    {"id": "clips",          "label": "Clip Detection",       "status": "detecting_clips",    "start_pct": 80,  "end_pct": 98},
    {"id": "saving",         "label": "Saving Results",       "status": "detecting_clips",    "start_pct": 98,  "end_pct": 100},
]


def _resolve_pipeline_stage(status: str, progress: int) -> dict:
    """Return the best-matching PIPELINE_STAGES entry for the given status+progress."""
    best = None
    for stage in PIPELINE_STAGES:
        if stage["status"] == status and stage["start_pct"] <= progress <= stage["end_pct"]:
            best = stage
    if best is None:
        # Fallback: match on status alone, pick the stage whose range includes progress
        # or the closest one
        candidates = [s for s in PIPELINE_STAGES if s["status"] == status]
        if candidates:
            best = min(candidates, key=lambda s: abs((s["start_pct"] + s["end_pct"]) / 2 - progress))
    return best or {}


# Jobs currently in the COMPLETE-finalization window. While a job is here,
# every in-flight / queued progress relay is a no-op so it can NEVER write a
# non-terminal status (e.g. ``detecting_clips``) over the COMPLETE save. This
# is stronger than ``protect_terminal`` (which can be defeated by a relay that
# loaded the pre-COMPLETE snapshot and commits just after the COMPLETE write):
# the gate is a synchronous set-membership check at coroutine entry, so it
# holds regardless of lock/loop scheduling. Production symptom it fixes: the
# UI stuck at 94 % "Asking the VLM…" / "Generating summary…" because the
# persisted status kept reverting to detecting_clips.
_finalizing_jobs: set = set()


async def _persist_complete_job(job_id: str, fields: dict) -> bool:
    """Force-persist the COMPLETE job state and verify it against the raw
    on-disk bytes, retrying a few times.

    Bypasses ``update_job_status`` (which silently no-ops when its internal
    load returns None, and writes to ``job.job_id``-derived path) by writing
    the canonical ``database._job_path(job_id)`` directly with synchronous
    I/O — so there is no ``await`` between the write and the verify read, and
    no concurrent task can interleave. Logs the exact cause (load-None,
    job_id mismatch, serialization error, or a stubborn revert) so a stuck
    job is never silent again.
    """
    import json as _json
    import os as _os
    import tempfile as _tf
    path = database._job_path(job_id)
    for attempt in range(1, 4):
        try:
            # CRITICAL: hold the SAME per-job lock that ``update_job_status``
            # and ``load_job`` take, across the whole load→write→verify cycle.
            # The earlier lock-free version raced and silently lost COMPLETE:
            # a clipper progress relay that passed the finalization gate just
            # before it was set, then ``await``-ed ``update_job_status`` and
            # loaded the *pre-COMPLETE* snapshot under the lock, could save
            # ``detecting_clips`` / 0-clips back AFTER this unlocked write
            # landed — clobbering the finished job with no log line (its
            # ``protect_terminal`` check saw the stale non-terminal snapshot
            # it had already loaded). Production symptom: "Pipeline complete"
            # logged, yet 30 s later the background clip-refresh load saw
            # ``DB job has 0 clips`` and the UI stayed stuck on "judging clip
            # candidates". Taking the lock serializes against that relay so
            # either the relay's detecting_clips write happens first (then we
            # overwrite it with COMPLETE) or it happens after (and its
            # ``protect_terminal`` correctly blocks the now-terminal job).
            async with database._get_lock(job_id):
                job = await database._load_job_unlocked(job_id)
                if job is None:
                    logger.error(
                        "[%s] finalize attempt %d: load returned None "
                        "(path=%s exists=%s) — cannot merge existing fields",
                        job_id, attempt, path, _os.path.exists(path))
                else:
                    _stored_id = getattr(job, "job_id", "")
                    if _stored_id != job_id:
                        logger.error(
                            "[%s] finalize: job.job_id mismatch (stored=%r) — forcing %r",
                            job_id, _stored_id, job_id)
                    job.job_id = job_id
                    for _k, _v in fields.items():
                        if hasattr(job, _k):
                            setattr(job, _k, _v)
                    try:
                        data = job.model_dump(mode="json")
                        content = _json.dumps(
                            data, indent=2,
                            default=getattr(database, "_numpy_safe_default", None))
                    except Exception as _ser:
                        logger.error(
                            "[%s] finalize attempt %d: serialization failed: %r",
                            job_id, attempt, _ser, exc_info=True)
                        raise
                    _os.makedirs(_os.path.dirname(path), exist_ok=True)
                    fd, tmp = _tf.mkstemp(dir=_os.path.dirname(path), suffix=".tmp")
                    try:
                        with _os.fdopen(fd, "w", encoding="utf-8") as _f:
                            _f.write(content)
                        _os.replace(tmp, path)
                    finally:
                        try:
                            if _os.path.exists(tmp):
                                _os.unlink(tmp)
                        except OSError:
                            pass
                # Verify against the raw bytes (no model layer, no await),
                # still under the lock so no writer can interleave.
                disk_status = ""
                disk_clips = 0
                try:
                    with open(path, "r", encoding="utf-8") as _f:
                        _raw = _json.load(_f)
                    disk_status = _raw.get("status", "")
                    disk_clips = len(_raw.get("clips", []) or [])
                except Exception as _re:
                    logger.error("[%s] finalize attempt %d: raw read failed: %r",
                                 job_id, attempt, _re)
            if disk_status == JobStatus.COMPLETE.value:
                # Log success unconditionally (not just on retries) so a
                # future stuck-job report can confirm the write actually
                # landed instead of being silent on the happy path.
                logger.info(
                    "[%s] finalize: COMPLETE persisted on attempt %d (clips=%d)",
                    job_id, attempt, disk_clips)
                return True
            logger.warning(
                "[%s] finalize attempt %d: on-disk status=%r clips=%d (path=%s)",
                job_id, attempt, disk_status, disk_clips, path)
        except Exception as _e:
            logger.error("[%s] finalize attempt %d raised: %r",
                         job_id, attempt, _e, exc_info=True)
        await asyncio.sleep(0.4)
    return False


async def _update_progress(
    job_id: str, status: str, progress: int, message: str,
    protect_terminal: bool = True,
    heartbeat_label: str = "",
):
    """Update job progress in DB and broadcast via WebSocket.
    If a cancel has been requested, raises CancelledError instead of
    writing a stale progress update that would overwrite the 'cancelled' status.

    ``protect_terminal`` defaults True so a progress write that lands after
    the COMPLETE save (e.g. a clipper progress callback relayed from the
    worker thread via ``run_coroutine_threadsafe``) cannot revert the job
    to ``detecting_clips`` and wipe the persisted clips / translated
    transcript. The one deliberate exception is the QUEUED reset at the
    top of ``run_analysis``, which re-runs a finished job and therefore
    passes ``protect_terminal=False``."""
    if is_cancel_requested(job_id):
        raise CancelledError(f"Job {job_id} was cancelled by user")
    # Hard gate: once finalization starts, drop every non-terminal progress
    # write (queued clipper/perceiver relays included) so the COMPLETE status
    # cannot be reverted to detecting_clips.
    _status_str = status.value if hasattr(status, "value") else str(status)
    if job_id in _finalizing_jobs and _status_str not in (
            "complete", "failed", "cancelled"):
        return
    await database.update_job_status(
        job_id,
        status=status,
        progress=progress,
        progress_message=message,
        protect_terminal=protect_terminal,
    )
    status_str = status.value if hasattr(status, 'value') else str(status)
    _stage = _resolve_pipeline_stage(status_str, progress)
    _stage_start = _stage.get("start_pct", 0)
    _stage_end = _stage.get("end_pct", 100)
    _stage_range = max(1, _stage_end - _stage_start)
    _stage_pct = int(min(100, max(0, (progress - _stage_start) / _stage_range * 100)))
    await broadcast_ws(job_id, {
        "type": "status",
        "status": status,
        "progress": progress,
        "message": message,
        "stage_id": _stage.get("id", ""),
        "stage_label": _stage.get("label", ""),
        "stage_pct": _stage_pct,
        "stage_start": _stage_start,
        "stage_end": _stage_end,
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
            "translating": "subtitle translation",
            "detecting_clips": "clip detection",
        }
        # ``heartbeat_label`` lets a caller override the status-derived label so
        # the heartbeat names the actual SUB-phase (e.g. "transcription" while
        # Whisper runs inside the ANALYZING_SCENES engine stage, instead of the
        # misleading "scene analysis"). Falls back to the per-status label.
        stage_label = heartbeat_label or _stage_labels.get(status_str, status_str)
        hb.touch(stage_label)


def _load_job_glossary(job_id: str) -> dict | None:
    """Load the per-job translation glossary if it exists.

    The glossary file is a JSON object mapping source terms to target
    terms, persisted at ``/data/uploads/{job_id}/glossary.json``.
    Returns None if no glossary is set or the file is malformed —
    translation continues without it.
    """
    path = f"/data/uploads/{job_id}/glossary.json"
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Accept both ``{"terms": {...}}`` and ``{...}`` directly.
        if isinstance(data, dict) and "terms" in data and isinstance(data["terms"], dict):
            return {str(k): str(v) for k, v in data["terms"].items() if str(k).strip()}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if str(k).strip() and not isinstance(v, (list, dict))}
    except Exception as e:
        logger.warning("[%s] glossary.json parse failed: %s", job_id, e)
    return None


def _slice_transcript_for_clip(segments: list, start_s: float, end_s: float) -> str:
    """Extract transcript text spoken during a clip's time window.

    Mirrors ``reframer_clipper._slice_transcript`` so the post-translation
    refresh produces the same shape (per-line ``[m:ss] text``) the bridge
    originally fed into caption / hook_text / title. Accepts both dicts
    and Pydantic models because the persisted ``translated_transcript``
    can be either depending on the call site that produced it.
    """
    lines = []
    for seg in segments or []:
        if hasattr(seg, "model_dump"):
            seg = seg.model_dump()
        seg_start = seg.get("start_sec", seg.get("start", 0)) or 0
        seg_end = seg.get("end_sec", seg.get("end", 0)) or 0
        if seg_end > start_s and seg_start < end_s:
            rel_start = max(0, seg_start - start_s)
            m, s = divmod(int(rel_start), 60)
            text = (seg.get("text") or "").strip()
            if text:
                lines.append(f"[{m}:{s:02d}] {text}")
    return "\n".join(lines) if lines else "(no speech in this segment)"


async def _refresh_clips_with_translation(
    job_id: str, translated: list, fallback_clips: Optional[list] = None,
) -> int:
    """Rebuild clip caption / hook_text / title from the translated transcript.

    Returns the number of clips updated. Mirrors the bridge's original
    fallback ladder (judge_title → vlm_hook → transcript) so clips end
    up in the same shape they would have had if translation had run
    BEFORE clip extraction. Only clips with usable VLM provenance are
    re-derived; legacy clips missing ``vlm_hook`` / ``judge_title`` are
    treated as pure-transcript and refreshed unconditionally so the
    Japanese caption gets replaced with the English one either way.

    When the fresh ``database.load_job`` snapshot lacks ``clips`` —
    occasionally seen in logs where the disk file round-trips clean
    in tests but reads back empty post-translation — fall back to
    the caller-supplied ``fallback_clips`` (the ``post_job.clips``
    captured at ``COMPLETE`` time) so the refresh still runs against
    the same dataset the rest of the pipeline used.
    """
    from backend.services.caption_text import strip_cue_timestamps
    job = await database.load_job(job_id)
    if job is None:
        logger.warning(
            "[%s] clip refresh: load_job returned None — skipping (translated=%d)",
            job_id, len(translated or []),
        )
        return 0, []
    source_clips = list(job.clips or [])
    used_fallback = False
    if not source_clips and fallback_clips:
        logger.warning(
            "[%s] clip refresh: DB job has 0 clips — using fallback list (n=%d)",
            job_id, len(fallback_clips),
        )
        source_clips = list(fallback_clips)
        used_fallback = True
    if not source_clips:
        return 0, []

    updated_clips = []
    changed = 0
    for clip in source_clips:
        clip_dict = clip.model_dump() if hasattr(clip, "model_dump") else dict(clip)
        start_s = float(clip_dict.get("start_time", 0.0) or 0.0)
        end_s = float(clip_dict.get("end_time", start_s) or start_s)
        if end_s <= start_s:
            updated_clips.append(clip_dict)
            continue

        new_slice = _slice_transcript_for_clip(translated, start_s, end_s)
        if new_slice == "(no speech in this segment)":
            # No transcript covers this clip — leave it alone.
            updated_clips.append(clip_dict)
            continue

        vlm_hook = (clip_dict.get("vlm_hook") or "").strip()
        vlm_reason = (clip_dict.get("vlm_reason") or "").strip()
        judge_title = (clip_dict.get("judge_title") or "").strip()
        idx = clip_dict.get("id", 0)

        # The slice carries inline ``[m:ss]`` cue markers; strip them from the
        # social-facing title/hook/caption (they are shown on cards and used as
        # on-screen overlays) while leaving the slice itself intact.
        new_title = judge_title or vlm_hook[:80] or (
            " ".join(strip_cue_timestamps(new_slice).split()[:8]) or f"Clip {idx}")
        # Prefer the TRANSLATED slice for the hook — vlm_hook is the VLM's
        # source-language line, so keeping it (as before) left every hook in the
        # source language even after translation. Use the first translated cue;
        # fall back to vlm_hook only when the slice is empty.
        _first_cue = new_slice.split("\n", 1)[0].strip() if new_slice else ""
        new_hook = strip_cue_timestamps(_first_cue)[:120] or vlm_hook or new_title
        new_caption = strip_cue_timestamps(new_slice)[:150] if new_slice else new_title
        new_why = vlm_reason or (
            "Strong audio/visual engagement signals across this window.")
        new_reasoning = vlm_reason or clip_dict.get("viral_score_reasoning", "")

        if (clip_dict.get("title") != new_title
                or clip_dict.get("hook_text") != new_hook
                or clip_dict.get("suggested_caption") != new_caption):
            changed += 1
            clip_dict["title"] = new_title
            clip_dict["hook_text"] = new_hook
            clip_dict["suggested_caption"] = new_caption
            clip_dict["why_this_works"] = new_why
            if new_reasoning:
                clip_dict["viral_score_reasoning"] = new_reasoning
        updated_clips.append(clip_dict)

    # Persist when anything changed OR when we ran against the in-process
    # fallback list — in the fallback case the DB has 0 clips on read, so
    # we need to write the source_clips back regardless of whether any
    # captions were rebuilt (the caller's whole reason for threading the
    # fallback through was to restore the DB state).
    if changed > 0 or used_fallback:
        # Persist ONLY the clips — no status pin. This now runs from
        # ``_run_post_clip_followups`` BEFORE the COMPLETE save (translation was
        # decoupled to run ahead of clip extraction), so the job is not yet
        # terminal; pinning COMPLETE here would prematurely finalize it ahead of
        # the authoritative ``_persist_complete_job`` save. ``update_job_status``
        # merges, so the current (TRANSLATING / DETECTING_CLIPS) status stands.
        await database.update_job_status(job_id, clips=updated_clips)
        logger.info(
            "[%s] Refreshed %d/%d clip captions/hooks/titles from translated transcript"
            "%s",
            job_id, changed, len(updated_clips),
            " (restored from in-process fallback)" if used_fallback else "",
        )
    # Return the rebuilt list so the caller can hand the SAME (target-language)
    # clips straight to Auto-SEO — the DB round-trip reads back 0 clips during
    # post-processing, so without this the SEO step would re-load the STALE
    # source-language fallback and clobber these captions/hooks/titles.
    return changed, updated_clips


async def _auto_generate_clip_seo(
    job_id: str, transcript: list, orchestrator,
    fallback_clips: Optional[list] = None,
    output_language: str = "",
) -> tuple[int, int]:
    """Generate SEO (title / description / tags / platform_tips) for every
    clip in the background after analysis completes.

    Without this, every clip card on the Viral Clips page shows raw
    transcript snippets ("[0:29] We've been waiting...") instead of a
    real SEO-ready title and caption. The manual "Generate SEO" button
    on the ClipSEO page still works for regenerating or targeting
    additional platforms — this helper just seeds the primary-platform
    SEO automatically so the first-load experience matches what a user
    would expect from a clip-generator product.

    Runs the per-platform prompt + the platform-cap validator from
    ``prompts.py`` so the generated copy stays inside the platform's
    actual character limits. Per-clip failures are logged but never
    abort the loop — partial success is better than nothing.

    Returns (generated_count, failed_count).
    """
    from backend.models import ClipSEO
    from backend.services.prompts import (
        load_prompts,
        build_platform_seo_prompt,
        enforce_platform_caps,
        PLATFORM_PROFILES,
    )

    def _resolve(raw: str | None) -> str:
        candidate = (raw or "").strip().lower().replace("-", "_")
        aliases = {
            "youtube_short": "youtube_shorts", "shorts": "youtube_shorts",
            "yt_shorts": "youtube_shorts", "yt": "youtube",
            "youtube_long": "youtube", "youtube_longform": "youtube",
            "ig": "instagram", "instagram_reels": "reels", "ig_reels": "reels",
            "fb": "facebook", "twitter": "x", "li": "linkedin", "both": "tiktok",
        }
        candidate = aliases.get(candidate, candidate)
        return candidate if candidate in PLATFORM_PROFILES else "tiktok"

    def _slice(start_s: float, end_s: float) -> str:
        lines = []
        for s in (transcript or []):
            ss = s.model_dump() if hasattr(s, "model_dump") else s
            seg_start = ss.get("start_sec", ss.get("start", 0)) or 0
            seg_end = ss.get("end_sec", ss.get("end", 0)) or 0
            if seg_end > start_s and seg_start < end_s:
                text = (ss.get("text") or "").strip()
                speaker = (ss.get("speaker") or "").strip()
                if text:
                    lines.append(f"{speaker}: {text}" if speaker else text)
        return "\n".join(lines)

    job = await database.load_job(job_id)
    if job is None:
        logger.info(
            "[%s] Auto-SEO skipped early: job_loaded=False, clip_count=0",
            job_id,
        )
        return (0, 0, list(fallback_clips or []))
    # When the fresh DB snapshot has 0 clips but the caller threaded
    # through the post-COMPLETE list, run against that — otherwise the
    # cards on the Viral Clips page ship as bare transcript snippets
    # for every job whose disk file round-trips empty here. The
    # round-trip works in isolated tests, but at least one production
    # job lost its clips between the COMPLETE save and the SEO load
    # (clipai_logs_20260527_000705.log line ~902).
    source_clips = list(job.clips or [])
    if not source_clips and fallback_clips:
        logger.warning(
            "[%s] Auto-SEO: DB job has 0 clips — using fallback list (n=%d)",
            job_id, len(fallback_clips),
        )
        # Use a local list rather than mutating ``job.clips``. The field
        # is typed ``list[ClipCandidate]`` on JobResult but ``fallback_clips``
        # is plain dicts from ``to_fez_clips``; Pydantic v2 silently accepts
        # the raw-dict assignment today (no ``validate_assignment``), but
        # writing a list[dict] into a typed-list field corrupts the next
        # ``model_dump`` round-trip and emits the
        # ``PydanticSerializationUnexpectedValue`` warning. The trailing
        # ``update_job_status(clips=updated_clips)`` already persists
        # updated_clips directly — no need to touch ``job.clips``.
        source_clips = list(fallback_clips)
    if not source_clips:
        logger.info(
            "[%s] Auto-SEO skipped early: job_loaded=True, clip_count=0",
            job_id,
        )
        return (0, 0, [])
    logger.info(
        "[%s] Auto-SEO starting on %d clips, %d transcript segments",
        job_id, len(source_clips), len(transcript or []),
    )
    # Re-label the keepalive so the multi-minute SEO pass reads as
    # "SEO generation" instead of the stale "clip detection" (the status stays
    # detecting_clips through here — see _run_post_clip_followups).
    set_heartbeat_stage(job_id, "SEO generation")

    video_summary = ""
    if job.summary:
        video_summary = job.summary.overview
        if job.summary.key_topics:
            video_summary += "\nTopics: " + ", ".join(job.summary.key_topics)

    # Cache the per-platform prompt strings so we don't rebuild the
    # ~2 kB template for every clip.
    prompt_cache: dict[str, str] = {}
    base_prompts = load_prompts()

    # Fetch TODAY's live short-form trend brief once (daily-cached, fail-soft) so
    # every clip's SEO — title, caption, tags, hook — reflects what's actually
    # trending on TikTok / YT Shorts right now, not the model's stale guesses.
    _trend_brief = ""
    try:
        from backend.services.trend_brief import get_trend_brief
        _trend_brief = await get_trend_brief("both", "")
    except Exception as _tb_err:
        logger.debug("[%s] live trend brief unavailable (%s) — SEO uses evergreen "
                     "patterns", job_id, _tb_err)

    generated = 0
    failed = 0
    updated_clips = []
    _seo_total = len(source_clips)
    for _seo_idx, clip in enumerate(source_clips, 1):
        # Visible, throttled progress so the activity log shows the SEO pass
        # advancing instead of a silent multi-minute gap (the bar already sits
        # at ~98% post-export; this only updates the message line + keepalive).
        if _seo_idx == 1 or _seo_idx % 8 == 0 or _seo_idx == _seo_total:
            set_heartbeat_stage(job_id, "SEO generation")
            try:
                await broadcast_ws(job_id, {
                    "type": "status",
                    "message": f"Generating SEO for clips… ({_seo_idx}/{_seo_total})",
                })
            except Exception:
                pass
        clip_dict = clip.model_dump() if hasattr(clip, "model_dump") else dict(clip)
        platform = _resolve(clip_dict.get("platform"))

        # Skip clips that already carry SEO for this platform — the
        # ClipSEO page might have generated it before this background
        # job ran, and we shouldn't overwrite user edits.
        existing = (clip_dict.get("seo_by_platform") or {}).get(platform)
        if existing and (existing.get("title") if isinstance(existing, dict)
                         else getattr(existing, "title", "")):
            updated_clips.append(clip_dict)
            continue
        if clip_dict.get("seo_title") and platform == _resolve(clip_dict.get("platform")):
            # Legacy SEO already populated for this platform.
            updated_clips.append(clip_dict)
            continue

        start_s = float(clip_dict.get("start_time", 0.0) or 0.0)
        end_s = float(clip_dict.get("end_time", start_s) or start_s)
        clip_transcript = _slice(start_s, end_s)
        if not clip_transcript:
            clip_transcript = clip_dict.get("suggested_caption") or clip_dict.get("title", "")

        if platform not in prompt_cache:
            prompt_cache[platform] = build_platform_seo_prompt(
                platform, trend_brief=_trend_brief, output_language=output_language)
        custom_prompts = base_prompts.model_copy(
            update={"seo": prompt_cache[platform]})

        # Re-bind the orchestrator with the per-platform prompt for
        # this single call. The constructor is cheap (no I/O) so the
        # per-clip allocation cost is negligible.
        from backend.services.ai_orchestrator import AIOrchestrator
        platform_orch = AIOrchestrator(
            ws_broadcast=broadcast_ws, custom_prompts=custom_prompts,
        )
        try:
            seo, provider = await platform_orch.generate_seo(
                clip_title=clip_dict.get("title", ""),
                clip_transcript=clip_transcript,
                video_summary=video_summary,
                platform=platform,
                job_id=job_id,
            )
            capped = enforce_platform_caps(seo.model_dump(), platform)
            seo_record = ClipSEO(**capped)
            seo_by_plat = dict(clip_dict.get("seo_by_platform") or {})
            seo_by_plat[platform] = seo_record.model_dump()
            clip_dict["seo_by_platform"] = seo_by_plat
            # Mirror into legacy single-platform fields so any reader
            # that still expects ``clip.seo_title`` keeps working.
            clip_dict["seo_title"] = capped.get("title", "")
            clip_dict["seo_description"] = capped.get("description", "")
            clip_dict["seo_tags"] = capped.get("tags", [])
            clip_dict["seo_platform_tips"] = capped.get("platform_tips", "")
            generated += 1
        except Exception as e:
            logger.warning(
                "[%s] Auto-SEO generation failed for clip %s (%s/%s)",
                job_id, clip_dict.get("id"), platform, e,
            )
            failed += 1
        updated_clips.append(clip_dict)

    if generated > 0 or failed > 0:
        await database.update_job_status(job_id, clips=updated_clips)
        logger.info(
            "[%s] Auto-SEO complete: %d generated, %d failed (%d clips total)",
            job_id, generated, failed, len(updated_clips),
        )
    else:
        # Loop ran but every clip was skipped (already had SEO). Log it
        # so "SEO generated for 0 clips" isn't a black box.
        logger.info(
            "[%s] Auto-SEO: all %d clips already had SEO, nothing to generate",
            job_id, len(updated_clips),
        )
    # Return the SEO'd list so the caller persists THESE (target-language
    # caption/hook/title carried from source_clips + the SEO fields) in the
    # COMPLETE save, instead of re-reading 0 clips from the DB.
    return (generated, failed, updated_clips)


async def _polish_transcript_loop(
    job_id: str,
    transcript: list,
    orchestrator,
    correction_lang: str,
    mode: str = "asr",
    model_override: Optional[str] = None,
) -> tuple[list, Optional[dict]]:
    """Run the LLM polish + readability loop on a transcript.

    ``model_override`` pins the polish LLM to a specific model — the subtitle
    pipeline passes the dedicated translation model so polishing runs on the
    translation AI (not the editorial model, which is reserved for SEO +
    summaries). None uses the editorial model.

    Returns ``(polished_models, best_readability_report)``. On any
    failure, returns ``(original_as_models, None)`` so callers can
    treat polishing as best-effort. Used both synchronously from
    ``_run_analysis_inner`` (so the readability splitter sees punctuated
    Japanese / Chinese text) and asynchronously from
    ``_background_post_processing`` if a critical-path run is skipped.

    ``mode='translation'`` post-edits ALREADY-translated subtitles toward
    natural phrasing (preserve meaning + timing, never re-translate into another
    language); the default ``mode='asr'`` is the Whisper-accuracy correction
    profile. (The translation post-edit now runs as a dedicated source-aligned
    pass in ``_background_post_processing``; this loop stays mode-agnostic.)
    """
    from backend.models import TranscriptSegment as _TS

    def _to_models(items):
        out = []
        for t in (items or []):
            if isinstance(t, _TS):
                out.append(t)
            elif isinstance(t, dict):
                try:
                    out.append(_TS(**t))
                except Exception:
                    continue
        return out

    if not transcript or not settings.AI_TRANSCRIPT_CORRECTION:
        return _to_models(transcript), None
    if not getattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True) or orchestrator is None:
        return _to_models(transcript), None

    from backend.services.transcript_polisher import correct_transcript
    from backend.services.compat_stubs import _adaptive_batch_size
    from backend.services.subtitle_formatter import (
        enforce_readability, compute_readability_report,
    )

    polished_models = _to_models(transcript)
    if not polished_models:
        return polished_models, None

    _polish_info = orchestrator.get_editorial_model_info()
    _batch_size = _adaptive_batch_size(len(polished_models))
    _total_batches = -(-len(polished_models) // _batch_size)
    _per_batch = 150 if _polish_info.get("is_thinking") else 90
    # When polishing runs on the (larger) translation model it may sit on CPU on
    # a small card, so its batches are slower — widen the per-batch budget so a
    # slow-but-progressing batch isn't killed and dropped to raw text.
    if model_override:
        _per_batch = max(
            _per_batch,
            int(getattr(settings, "SUBTITLE_POLISH_TRANSLATION_SECONDS_PER_BATCH", 180)),
        )
    _estimated_time = (_total_batches * _per_batch) * 1.5
    _correction_timeout = max(180, min(1800, int(_estimated_time) + 60))
    logger.info(
        "[%s] Polishing timeout: %ds (segments=%d, batches=%d, per_batch=%ds, lang=%s, model=%s)",
        job_id, _correction_timeout, len(polished_models), _total_batches,
        _per_batch, correction_lang or "auto", model_override or "editorial",
    )

    target_score = float(getattr(settings, "TRANSCRIPT_READABILITY_TARGET", 90.0))
    max_passes = int(getattr(settings, "TRANSCRIPT_READABILITY_MAX_PASSES", 3))
    best_models = list(polished_models)
    best_report: Optional[dict] = None
    for _pass in range(1, max_passes + 1):
        try:
            polished_models = await asyncio.wait_for(
                correct_transcript(
                    polished_models,
                    orchestrator,
                    job_id=job_id,
                    language=correction_lang,
                    mode=mode,
                    model_override=model_override,
                ),
                timeout=_correction_timeout,
            )
            polished_models = _to_models(polished_models)
        except Exception as _pe:
            logger.warning(
                "[%s] Polish pass %d failed (%s) — keeping previous text",
                job_id, _pass, _pe,
            )

        if getattr(settings, "SUBTITLE_CPS_ENFORCEMENT", False) and polished_models:
            try:
                polished_models = enforce_readability(
                    list(polished_models),
                    max_cps=float(getattr(settings, "SUBTITLE_MAX_CPS", 20.0)),
                    max_chars_per_line=int(getattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42)),
                    min_duration_ms=int(getattr(settings, "SUBTITLE_MIN_DURATION_MS", 833)),
                    max_duration_ms=int(getattr(settings, "SUBTITLE_MAX_DURATION_MS", 9000)),
                    smart_line_breaks=bool(getattr(settings, "SUBTITLE_SMART_LINE_BREAKS", True)),
                )
            except Exception as _rd_err:
                logger.warning("[%s] Readability enforcement failed (%s) — using unmodified polish",
                               job_id, _rd_err)

        try:
            pass_report = compute_readability_report(list(polished_models))
        except Exception:
            pass_report = None

        if pass_report and (
            best_report is None or pass_report.get("score", 0) > best_report.get("score", 0)
        ):
            best_report = pass_report
            best_models = list(polished_models)

        _score = pass_report.get("score") if pass_report else None
        logger.info(
            "[%s] Polish pass %d/%d → readability %s%s",
            job_id, _pass, max_passes,
            f"{_score:.1f}/100" if _score is not None else "(no score)",
            " ✓ target met" if _score is not None and _score >= target_score else "",
        )
        if _score is not None and _score >= target_score:
            break

    return best_models, best_report


async def _set_translation_status(job_id: str, status: str, reason: Optional[str] = None):
    """Persist a visible, job-level translation outcome + broadcast it (Task 3).

    ``status`` is ``"translated"`` or ``"translation_failed"``. A *planned*
    translation that does not produce target-language output must surface here
    so the job never silently presents as a clean COMPLETE with source-language
    subtitles. Best-effort: never raises into callers.
    """
    try:
        await database.update_job_status(
            job_id,
            translation_status=status,
            translation_error=(reason or None),
        )
    except Exception as _e:
        logger.debug("[%s] translation_status persist skipped: %s", job_id, _e)
    try:
        await broadcast_ws(job_id, {
            "type": "translation_status",
            "state": status,
            "status": status,
            "message": reason or (
                "Subtitles translated" if status == "translated"
                else "Translation did not run — keeping source-language subtitles"),
        })
    except Exception:
        pass


async def _run_post_clip_followups(job_id: str, orchestrator, pp_result: Optional[dict], clips):
    """Clip-dependent finishers that run AFTER clip extraction: rebuild clip
    captions from the translated transcript (when translation succeeded) and
    seed per-clip Auto-SEO from the best-available transcript.

    Split out of ``_background_post_processing`` so subtitle translation can run
    BEFORE clip extraction (and therefore never be skipped when the clip stage
    fails) while the parts that genuinely need clips still run once the clips
    exist. Best-effort throughout — a failure here never aborts the finalize.
    ``clips`` (the in-process list) is threaded as the fallback for both
    finishers because the DB round-trip can read back 0 clips before the
    COMPLETE save lands.
    """
    pp = pp_result or {}
    fallback_clips = list(clips or [])
    # The authoritative clip list to hand back for the COMPLETE save. Starts as
    # the in-process clips and is upgraded to the refreshed / SEO'd list as those
    # steps run, so the final save never depends on the DB reading back clips.
    final_clips = list(fallback_clips)
    target_name = pp.get("target_name") or "the target language"

    # (a) Re-derive clip caption / hook_text / title from the translated
    #     transcript so the Viral Clips cards show the target language.
    # ``seo_input_clips`` is what Auto-SEO (step b) runs on. It MUST be the
    # refreshed (target-language) list, not the original ``fallback_clips`` —
    # the DB reads back 0 clips here, so if SEO fell back to the stale source
    # list it would re-persist Japanese captions over the English refresh.
    seo_input_clips = fallback_clips
    if pp.get("translated") and pp.get("target_transcript"):
        try:
            _refreshed, _refreshed_clips = await _refresh_clips_with_translation(
                job_id, list(pp["target_transcript"]),
                fallback_clips=fallback_clips,
            )
            if _refreshed_clips:
                seo_input_clips = _refreshed_clips
                # Carry the English list forward even if SEO returns nothing,
                # so a skipped/failed SEO pass can't revert the COMPLETE save
                # to the source-language clips.
                final_clips = _refreshed_clips
            if _refreshed > 0:
                await broadcast_ws(job_id, {
                    "type": "clips_refreshed",
                    "message": f"Refreshed {_refreshed} clip captions in {target_name}",
                })
        except Exception as _cr_err:
            logger.warning("[%s] Post-translation clip refresh failed: %s", job_id, _cr_err)

    # (b) Seed per-clip SEO from the final transcript (translated when
    #     available, otherwise the polished source).
    seo_segments = pp.get("seo_transcript")
    if seo_segments is None:
        try:
            _j = await database.load_job(job_id)
            seo_segments = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in (getattr(_j, "transcript", []) or [])
            ]
        except Exception:
            seo_segments = []
    try:
        await broadcast_ws(job_id, {
            "type": "background_task", "task": "auto_seo", "status": "running",
            "message": "Generating SEO titles, captions, and tags for clips...",
        })
        _g, _f, _seo_clips = await _auto_generate_clip_seo(
            job_id, list(seo_segments or []), orchestrator,
            fallback_clips=seo_input_clips,
            output_language=pp.get("output_lang", ""),
        )
        if _seo_clips:
            final_clips = _seo_clips
        await broadcast_ws(job_id, {
            "type": "background_task", "task": "auto_seo", "status": "complete",
            "message": (f"SEO generated for {_g} clips"
                        + (f" ({_f} failed)" if _f else "")),
        })
    except Exception as _seo_err:
        logger.warning("[%s] Auto-SEO seeding failed: %s", job_id, _seo_err, exc_info=True)
        await broadcast_ws(job_id, {
            "type": "background_task", "task": "auto_seo", "status": "failed",
            "message": f"Auto-SEO skipped: {str(_seo_err)[:80]}",
        })
    # Hand the final (target-language caption/hook/title + SEO) list back so the
    # caller can persist it directly — the DB round-trip can't be trusted to
    # have these clips during post-processing.
    return final_clips


async def _background_post_processing(
    job_id: str, transcript: list, orchestrator, job,
    polished_already: bool = False,
):
    """Run subtitle translation + target-language polish (and, when no
    translation is needed or translation fails, the source-language polish
    fallback). Returns an outcome ``dict`` (see ``_result`` below) — it does
    NOT touch clips.

    Awaited on the critical path from ``_run_analysis_inner`` BEFORE clip
    extraction (Task 1), so a clip-stage failure (Replicate 429, etc.) can never
    bypass translation. The clip-dependent finishers (caption refresh + Auto-SEO)
    were split out into ``_run_post_clip_followups``, which the caller runs after
    the clip stage using the returned outcome.

    Reordered to translate-THEN-polish: the reframer's Whisper path runs
    task="transcribe" (source-language output), so the LLM translator is what
    actually produces the target language. The sequence is (a) translate →
    (b) LLM polish on the translated text (lang=target) → (c) sentence
    resegment + readability reflow in the target language → (d) final dedup →
    persist ``translated_transcript`` (no status pin — the job is not terminal
    yet; the COMPLETE save happens in the caller after clips). When no
    translation is needed, the source-language polish runs here only if it
    wasn't already done on the critical path. On any failure the best available
    transcript is kept and a visible ``translation_failed`` status is set — we
    never persist the raw source while labelling it translated, and never ship
    a raw, unpolished transcript.

    Returns ``_result`` with keys: ``will_translate``, ``translated``,
    ``source_transcript`` (for the caller's ``transcript`` field — polished on
    the fallback paths), ``target_transcript`` (translated+polished, for clip
    refresh), ``seo_transcript`` (what Auto-SEO should read), ``target_name``
    and ``failed_reason``.
    """
    logger.info("[%s] _background_post_processing entry (polished_already=%s)",
                job_id, polished_already)

    def _as_dicts(items):
        return [
            t.model_dump() if hasattr(t, "model_dump") else dict(t)
            for t in (items or [])
        ]

    # Outcome contract returned to ``_run_analysis_inner`` so it can (a) adopt
    # the finalized SOURCE transcript for the COMPLETE save (polished on the
    # no-translation / translation-failed paths — never raw), (b) drive the
    # post-clip caption refresh + Auto-SEO from the right transcript, and
    # (c) fail loud when a planned translation produced no output (Task 3).
    _result = {
        "will_translate": False,
        "translated": False,
        "source_transcript": None,   # SOURCE transcript for the `transcript` field
        "target_transcript": None,   # translated+polished transcript (clip refresh)
        "seo_transcript": None,      # transcript Auto-SEO should read from
        "target_name": "",
        "failed_reason": None,
    }

    # ── Resolve source / target languages (used by every branch below) ──
    # An explicit job.subtitle_language always wins; otherwise auto-translate
    # non-English audio to English so the default UX matches expectations
    # (upload Japanese → get English subtitles).
    from backend.services.compat_stubs import _last_detected_language
    target_lang = (job.subtitle_language or "").strip().lower()
    source_lang = (job.language or "").strip().lower()
    if source_lang in ("", "auto"):
        # "auto" must resolve to the language Whisper actually detected — otherwise
        # the LLM gets a vague "translate from the source language" prompt and the
        # CJK purity check can't run, which is how half-Japanese tracks slipped
        # through.
        source_lang = (_last_detected_language.get("lang", "") or "").strip().lower()
    if not target_lang and source_lang and source_lang not in ("en", "english"):
        target_lang = "en"
        logger.info(
            "[%s] Auto-translating subtitles: %s → en (no explicit subtitle_language set)",
            job_id, source_lang,
        )
    logger.info(
        "[%s] subtitle_language=%s → target=%s, source=%s",
        job_id, job.subtitle_language or "(none)", target_lang or "(none)",
        source_lang or "auto",
    )
    _will_translate = bool(target_lang and target_lang != source_lang and transcript)
    _result["will_translate"] = _will_translate
    # The language the SHIPPED subtitles/clips end up in: the target when we
    # translate, else the source. The summary + Auto-SEO are written in THIS
    # language so everything (transcript, clips, summary, SEO) matches.
    _result["output_lang"] = (target_lang if _will_translate else source_lang) or ""

    # ── Transcript polishing — SOURCE language, fallback only ──
    # Runs only when NO translation was applied (when a translation follows,
    # the heavy polish runs on the TRANSLATED text instead) and only when the
    # critical-path polish was skipped (``polished_already`` is False). Defined
    # as a nested helper so BOTH the no-translation path and the
    # translation-failed fall-through can reuse it — the latter matters because
    # a translate-then-polish job skips the source polish on the critical path,
    # so if translation fails we still want the source transcript polished.
    async def _polish_source_if_needed():
        nonlocal transcript
        if polished_already or not settings.AI_TRANSCRIPT_CORRECTION or not transcript:
            return
        try:
            logger.info("[%s] Source-language transcript polish (fallback path, lang=%s)",
                        job_id, source_lang or "auto")
            await broadcast_ws(job_id, {
                "type": "background_task", "task": "transcript_polishing",
                "status": "running",
                "message": "Polishing transcript in background...",
            })
            # Polish in the SOURCE language — no Whisper-translate shortcut to
            # assume around (the reframer always transcribes, never translates).
            polished_models, best_report = await _polish_transcript_loop(
                job_id, transcript, orchestrator, source_lang,
                model_override=_resolve_polish_model_override(orchestrator),
            )
            polished_dicts = [
                p.model_dump() if hasattr(p, "model_dump") else dict(p)
                for p in polished_models
            ]
            if best_report is not None:
                await database.update_job_status(
                    job_id, transcript=polished_dicts, transcript_readability=best_report,
                )
            else:
                await database.update_job_status(job_id, transcript=polished_dicts)
            transcript = polished_models
            await broadcast_ws(job_id, {
                "type": "background_task", "task": "transcript_polishing",
                "status": "complete", "message": "Transcript polished",
            })
        except Exception as e:
            logger.warning("[%s] Background transcript polishing failed: %s",
                           job_id, e, exc_info=True)
            await broadcast_ws(job_id, {
                "type": "background_task", "task": "transcript_polishing",
                "status": "failed", "message": f"Polishing skipped: {str(e)[:80]}",
            })

    async def _dedup_source_transcript():
        """Guaranteed dedup + timing cleanup on the SOURCE transcript (Task 5).

        The translated path runs its own dedup before persisting; this mirror
        runs whenever translation was skipped OR failed, so the shipped
        transcript never carries duplicate / overlapping cues (the
        ``[6:23]``/``[6:23]`` and scattered-loop cases) just because the
        post-translation cleanup didn't get to run. Persists the cleaned
        ``transcript`` field. Best-effort: never raises into the caller.
        """
        nonlocal transcript
        if not transcript:
            return
        try:
            from backend.services.transcript_dedup import (
                collapse_adjacent_duplicates, drop_repetition_loops,
                collapse_overlapping_duplicates, collapse_intra_cue_repetition,
            )
            _src = [
                t.model_dump() if hasattr(t, "model_dump") else dict(t)
                for t in transcript
            ]
            _pre = len(_src)
            _ic = 0
            if getattr(settings, "SUBTITLE_INTRA_CUE_DEDUP_ENABLED", True):
                _src, _ic = collapse_intra_cue_repetition(
                    _src,
                    min_word_run=int(getattr(settings, "SUBTITLE_INTRA_CUE_MIN_WORD_RUN", 4)),
                )
            _src, _a = collapse_adjacent_duplicates(_src)
            _src, _o = collapse_overlapping_duplicates(_src)
            _src, _l = drop_repetition_loops(_src)
            if _a or _o or _l or _ic:
                logger.info(
                    "[%s] Source transcript dedup (translation %s): %d → %d "
                    "(%d adjacent, %d overlapping, %d repetition-loop, %d intra-cue)",
                    job_id, "skipped/failed", _pre, len(_src), _a, _o, _l, _ic,
                )
                transcript = _src
                await database.update_job_status(job_id, transcript=_src)
            else:
                logger.debug("[%s] Source transcript dedup: nothing to remove", job_id)
        except Exception as _dd_err:
            logger.warning("[%s] Source transcript dedup skipped (%s)", job_id, _dd_err)

    # ── Subtitle translation, then polish in the target language ──
    if _will_translate:
        from backend.services.translator import (
            SUPPORTED_LANGUAGES,
            TranslationFailedError, TranslationRateLimitedError,
        )
        target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang) if source_lang else "auto-detected"
        _result["target_name"] = target_name
        logger.info("[%s] Subtitle translation: %s → %s (%d segments)",
                    job_id, source_name, target_name, len(transcript))
        # ── Cross-video contamination trace ──────────────────────────────────
        # Log a sample of the SOURCE cues actually being translated for THIS job,
        # so a "transcript belongs to a different video" report can be pinned to
        # the exact stage: if these source lines already match the wrong video,
        # the leak is upstream (audio/Whisper); if they're correct here but the
        # stored/translated track is wrong, it's downstream. Cheap + safe.
        try:
            def _txt(_c):
                return (_c.get("text") if isinstance(_c, dict) else getattr(_c, "text", "")) or ""
            _src_sample = " | ".join(_txt(c)[:50] for c in (transcript or [])[:4])
            logger.info("[%s] XLATE-TRACE source[ja] sample (n=%d): %s",
                        job_id, len(transcript or []), _src_sample)
        except Exception:
            pass

        # Per-video glossary (Key Name and Phrases) — loaded from disk so
        # the user can drop a JSON file in via the Settings UI without
        # restarting the pipeline.
        glossary = None
        if getattr(settings, "TRANSLATION_GLOSSARY_ENABLED", True):
            try:
                glossary = _load_job_glossary(job_id)
                if glossary:
                    logger.info(
                        "[%s] Loaded glossary with %d terms for translation",
                        job_id, len(glossary),
                    )
            except Exception as _g_err:
                logger.warning("[%s] Glossary load failed (%s) — translating without glossary",
                               job_id, _g_err)

        await broadcast_ws(job_id, {
            "type": "background_task",
            "task": "subtitle_translation",
            "status": "running",
            "message": f"Translating subtitles to {target_name}...",
        })

        # Scale timeout generously with segment count. Offline NMT is LOCAL and
        # FREE, so there's no cost reason to cap it tightly — a timeout here would
        # discard the whole translation and revert the user to source-language
        # subtitles, which we never want. 8 s/segment with a 30-min floor is
        # ~16x the measured NLLB rate even with the per-cue completeness retries,
        # so a real video never times out mid-translation. (The one-time offline
        # model download, when needed, also fits inside this floor.)
        _trans_timeout = max(1800, len(transcript) * 8)
        # Translator calls ``seg.text`` directly, so make sure every
        # row is a TranscriptSegment regardless of upstream shape.
        from backend.models import TranscriptSegment as _TS_for_translate
        from backend.services.audio_analyzer import is_subtitle_marker as _is_marker
        _trans_input = []
        _music_markers = []   # non-speech cues kept verbatim, re-merged after
        for t in (transcript or []):
            _txt = t.get("text", "") if isinstance(t, dict) else getattr(t, "text", "")
            if _is_marker(_txt):
                # Language-neutral marker ("[♪ music ♪]") — don't translate it.
                _music_markers.append(t if isinstance(t, dict) else (
                    t.model_dump() if hasattr(t, "model_dump") else dict(t)))
                continue
            if isinstance(t, _TS_for_translate):
                _trans_input.append(t)
            elif isinstance(t, dict):
                try:
                    _trans_input.append(_TS_for_translate(**t))
                except Exception:
                    pass
        # ── (a0-pre) Strip Whisper repetition-loop hallucinations from the SOURCE
        # BEFORE translating. On music / quiet / repetitive audio Whisper loops and
        # re-emits the same line at many scattered timestamps. The LLM/NMT paths
        # translate each copy 1:1, so the loop reappeared in the TRANSLATED
        # transcript (the "duplicated transcript" report) — and the translated-side
        # loop-drop was gated to the Whisper-native path, so it survived. Dropping
        # it here fixes the duplication at the ROOT for every engine and shrinks the
        # translate workload. ``_music_markers`` were already held out, so a real
        # ``[♪ music ♪]`` is untouched; genuine short interjections are CAPPED (to
        # 3), not erased, by the dropper.
        if _trans_input:
            try:
                from backend.services.transcript_dedup import (
                    collapse_adjacent_duplicates as _cad,
                    collapse_overlapping_duplicates as _cod,
                    drop_repetition_loops as _drl,
                    drop_scattered_duplicates as _dsd,
                    collapse_repeated_runs as _crr,
                )
                _pre_src_dd = len(_trans_input)
                # Block-level first (on the raw sequence): drop a whole run of
                # cues re-transcribed at later timestamps (Whisper double-pass).
                _trans_input, _sr = _crr(_trans_input)
                _trans_input, _sa = _cad(_trans_input)
                _trans_input, _so = _cod(_trans_input)
                _trans_input, _ss = _dsd(_trans_input)
                _trans_input, _sl = _drl(_trans_input)
                if _sa or _so or _sl or _ss or _sr:
                    logger.info(
                        "[%s] Source dedup before translate: %d → %d cue(s) "
                        "(%d run, %d adjacent, %d overlapping, %d repetition-loop, %d scattered)",
                        job_id, _pre_src_dd, len(_trans_input), _sr, _sa, _so, _sl, _ss)
            except Exception as _sdd_err:
                logger.warning("[%s] Pre-translate source dedup skipped (%s)",
                               job_id, _sdd_err)
        # ── (a0) Resegment the SOURCE into one-utterance-per-cue units before
        # translating (Task 5). NMT translates cleaner sentence units far more
        # reliably than run-on blocks, and it keeps source↔target cue counts
        # comparable. Markers were already held out into _music_markers, so this
        # only touches dialogue. The target side resegments again post-translate.
        if getattr(settings, "SENTENCE_SEGMENTATION_ENABLED", True) and _trans_input:
            try:
                from backend.services.sentence_segmenter import resegment_by_sentence
                _pre_src_seg = len(_trans_input)
                _trans_input = resegment_by_sentence(_trans_input)
                if len(_trans_input) != _pre_src_seg:
                    logger.info("[%s] Source resegmentation (pre-translate): %d → %d cues",
                                job_id, _pre_src_seg, len(_trans_input))
            except Exception as _src_seg_err:
                logger.warning("[%s] Source resegmentation skipped (%s)",
                               job_id, _src_seg_err)
        try:
            # ── (a) Translate (LLM-first, then Whisper-native / offline NMT) ──
            logger.info("[%s] Translate START: %s → %s (%d segments)",
                        job_id, source_name, target_name, len(_trans_input))
            orchestrator.reset_circuit_breaker()

            # Surface the one-time offline-model auto-download to the UI over
            # the existing websocket channel.
            async def _nmt_status(msg: str):
                logger.info("[%s] %s", job_id, msg)
                # Push a REAL progress update (not just a background_task ping) so
                # the main bar + progress_message reflect translation progress —
                # otherwise it sat at a static 63 % "Translating…" for the whole
                # multi-minute pass and looked stuck. An optional "(a/b)" hint in
                # the message (per-batch cue counts) is mapped onto the 63→69 %
                # translation band so the bar actually advances; the changing
                # message also keeps the stuck-timer reset. ``except Exception``
                # only — a real cancel (CancelledError, a BaseException) still
                # propagates and stops the translation.
                _pct = translation_progress_pct(msg)
                try:
                    await _update_progress(
                        job_id, JobStatus.TRANSLATING, _pct, msg,
                        heartbeat_label="subtitle translation")
                except Exception:
                    pass
                await broadcast_ws(job_id, {
                    "type": "background_task",
                    "task": "subtitle_translation",
                    "status": "running",
                    "message": msg,
                })

            # ── (a) Translate via the shared translation router ──
            # LLM-first when an editorial model is configured and
            # TRANSLATION_PREFER_LLM is on (the default) — it renders every cue
            # 1:1; then Whisper-native audio→English for →en (single-step, avoids
            # the transcribe-then-translate double-error); then offline NMT as
            # the fallback for other pairs or when the prior engines come back
            # empty / still source-language. The post-edit below polishes the
            # result. Whisper-native gets generous headroom (full ASR pass, may
            # run on CPU); the NMT path keeps the scaled timeout. See
            # translate_subtitles for the VRAM-release ordering on low-VRAM cards.
            translated, _engine_used = await translate_subtitles(
                _trans_input, source_lang, target_lang,
                video_path=getattr(job, "file_path", None),
                glossary=glossary,
                orchestrator=orchestrator,
                status_callback=_nmt_status,
                job_id=job_id,
                whisper_timeout=max(1800, _trans_timeout),
                nmt_timeout=_trans_timeout,
            )
            _used_whisper_native = (_engine_used == "whisper")
            _used_llm = (_engine_used == "llm")
            if _used_whisper_native:
                changed = len(translated)
                logger.info("[%s] Whisper native translate: %d English cues "
                            "(offline, single-step, no LLM)", job_id, len(translated))
            else:
                # Compare against the coerced model list (``_trans_input``) so the
                # count works whether the caller handed us dicts or models.
                changed = sum(
                    1 for t, o in zip(translated, _trans_input)
                    if (getattr(t, "text", "") or "") != (getattr(o, "text", "") or "")
                )

            logger.info("[%s] Translate DONE: %d/%d segments changed → %s",
                        job_id, changed, len(translated), target_lang)
            try:
                _out_sample = " | ".join(
                    ((getattr(t, "text", "") or "") if not isinstance(t, dict)
                     else (t.get("text") or ""))[:50]
                    for t in (translated or [])[:4])
                logger.info("[%s] XLATE-TRACE translated[%s] sample: %s",
                            job_id, target_lang, _out_sample)
            except Exception:
                pass
            # Never relabel the source as translated: if NOTHING changed the
            # translation effectively failed, so bail to the no-translation
            # path (keeps the source transcript + runs SEO on it) rather than
            # persisting Japanese under ``translated_transcript``.
            if changed == 0:
                raise RuntimeError(
                    "translation produced 0 changed segments — keeping source transcript")

            # ── (b-pre) Word-timed resegmentation for the Whisper-native path ──
            # Whisper-native emits a few long, multi-sentence cues but WITH word
            # timestamps. Split them into sentence cues NOW, using that word
            # timing for accurate boundaries, BEFORE the MT post-edit rewrites the
            # text and discards the word timestamps (transcript_polisher clears
            # ``words`` whenever it changes the text). Without this, the post-edit
            # path always falls back to a char-length proportional split — cues
            # land seconds off the audio. (The NMT path is source-aligned 1:1, so
            # it must resegment AFTER its post-edit — step (c) below.)
            _pre_resegmented = False
            if _used_whisper_native and getattr(
                    settings, "SENTENCE_SEGMENTATION_ENABLED", True):
                try:
                    from backend.services.sentence_segmenter import resegment_by_sentence
                    _pre_rs = len(translated)
                    translated = resegment_by_sentence(translated)
                    # The word-timed split rebuilds cue text from the (pre-glossary)
                    # words, so re-apply the glossary to the freshly-split cues.
                    if glossary:
                        try:
                            from backend.services.nmt_translator import apply_glossary as _ag
                            for _t in translated:
                                _txt = getattr(_t, "text", "") or ""
                                _t.text = _ag(_txt, _txt, glossary)
                        except Exception:
                            pass
                    _pre_resegmented = True
                    logger.info(
                        "[%s] Whisper resegmentation (word-timed, pre-polish): %d → %d segments",
                        job_id, _pre_rs, len(translated))
                except Exception as _prs_err:
                    logger.warning(
                        "[%s] Whisper pre-polish resegmentation failed (%s)", job_id, _prs_err)

            # ── (b) AI post-edit on the TRANSLATED text (MT post-editing) ──
            # Offline NMT produced the base translation; the editorial LLM now
            # POLISHES that rough draft toward natural, professional subtitles —
            # closing the quality gap WITHOUT doing the translation itself. This
            # is ONE source-aligned pass (the readability reflow is steps (c)/(c-
            # cont) below), so the model can post-edit each draft line against
            # its ORIGINAL source line (the ground truth for meaning).
            from backend.services.transcript_polisher import correct_transcript as _mtpe
            # Source↔draft alignment holds only for the offline-NMT path (1:1);
            # Whisper-native emits its own cue count, so drop the source ref then.
            _src_texts = [getattr(s, "text", "") or "" for s in _trans_input]
            if _used_whisper_native or len(_src_texts) != len(translated):
                _src_texts = None
            # Respect the master AI-correction switch (the old polish loop gated
            # on it too); when off, keep the raw NMT draft.
            if not getattr(settings, "AI_TRANSCRIPT_CORRECTION", True):
                logger.info("[%s] AI post-edit skipped (AI_TRANSCRIPT_CORRECTION off) — "
                            "keeping NMT draft", job_id)
            elif _used_llm:
                # The editorial LLM TRANSLATED the source directly — its output is
                # already clean, complete target-language text. Re-running the
                # post-edit (which compares each line against the SOURCE) was
                # reverting good translations back to the source language (a perfect
                # 0%-Japanese LLM result came back ~40% Japanese). The LLM
                # translation is final; skip the post-edit.
                logger.info("[%s] AI post-edit skipped — the LLM produced the "
                            "translation directly (no post-edit needed)", job_id)
            elif (not _used_whisper_native
                  and bool(getattr(settings, "OFFLINE_TRANSLATION_MTPE_ENABLED", True))
                  and (getattr(settings, "OLLAMA_HOST", "") or "").strip()
                  and (getattr(settings, "OLLAMA_TRANSLATION_MODEL", "") or "").strip()):
                # The offline NMT path already ran a dedicated MTPE post-edit
                # (NLLB/Opus draft → OLLAMA_TRANSLATION_MODEL) inside
                # translate_segments_with_fallback; don't double-edit it here
                # with the orchestrator's (possibly different) editorial model.
                logger.info("[%s] AI post-edit skipped — offline NMT path already "
                            "MTPE-polished the draft (OLLAMA_TRANSLATION_MODEL)", job_id)
            else:
                logger.info(
                    "[%s] AI post-edit START on translated text (lang=%s, %d segments, "
                    "MT post-editing%s)",
                    job_id, target_lang, len(translated),
                    ", source-aligned" if _src_texts is not None else "",
                )
                try:
                    _pol = await asyncio.wait_for(
                        _mtpe(
                            translated, orchestrator,
                            job_id=job_id, language=target_lang,
                            source_texts=_src_texts, source_language=source_lang,
                            mode="translation",
                            model_override=_resolve_polish_model_override(orchestrator),
                        ),
                        timeout=max(600, len(translated) * 8),
                    )
                    if _pol:
                        # Safety: a post-edit must never REINTRODUCE the source
                        # language. If it raised the source-script fraction it
                        # corrupted the draft — keep the pre-edit translation.
                        from backend.services.translator import fraction_untranslated
                        _before = fraction_untranslated(translated, target_lang)
                        _after = fraction_untranslated(_pol, target_lang)
                        if _after > _before + 0.02:
                            logger.warning(
                                "[%s] AI post-edit reintroduced source language "
                                "(%.0f%% → %.0f%% source-script) — keeping the pre-edit "
                                "translation", job_id, 100 * _before, 100 * _after)
                        else:
                            translated = _pol
                    logger.info("[%s] AI post-edit DONE on translated text: %d segments",
                                job_id, len(translated))
                except Exception as _pol_err:
                    logger.warning("[%s] Post-translation AI post-edit failed (%s) — keeping NMT draft",
                                   job_id, _pol_err)

            # ── (c) Sentence resegmentation in the target language ──
            # Skipped for Whisper-native — already done word-timed in (b-pre),
            # before the post-edit dropped its word timestamps; re-running it on
            # the now-wordless cues would only re-merge/re-split them with the
            # inaccurate proportional fallback.
            #
            # Skipped for the LLM path too. translate_via_llm emits clean,
            # one-utterance-per-cue output that already inherits the SOURCE's
            # word-timed boundaries (the source was sentence-resegmented with
            # Whisper words before translation), but the translated cues carry
            # NO words of their own. resegment_by_sentence would MERGE the
            # same-speaker run into one block then re-split it — and with no
            # words to time the split it falls back to a GLOBAL char-proportional
            # cut across the whole merged span, erasing the real per-cue timing
            # and the silences between cues. That is the "scrambled timing" the
            # LLM track otherwise showed. Its 1:1 cues are final here; the
            # per-cue readability reflow + dedup below still run.
            if _used_llm:
                logger.info(
                    "[%s] Translated resegmentation skipped — LLM cues are already "
                    "1:1 and inherit the source's word-timed boundaries; instead "
                    "projecting real word timings so the splitter can break run-ons",
                    job_id)
                # ── (c-hybrid) Project audio-aligned word timings onto the LLM text ──
                # The LLM gives faithful English but NO timing. Get Whisper's
                # native English word timestamps as a TIMING REFERENCE (its text is
                # discarded) and project them onto the LLM cues via same-language
                # EN↔EN alignment (tier A); fall back to the 1:1 source cue's word
                # pauses (tier B); else keep the cue whole (tier C). Cues that gain
                # word timing can then be split at REAL audio pauses below.
                if getattr(settings, "HYBRID_WORD_TIMING_ENABLED", True):
                    try:
                        from backend.services.subtitle_aligner import project_hybrid_timings
                        from backend.models import TranscriptSegment as _TSh
                        _llm_cues = [
                            t if isinstance(t, _TSh)
                            else _TSh(**(t.model_dump() if hasattr(t, "model_dump") else t))
                            for t in translated
                        ]
                        _whisper_ref = await _get_whisper_en_timing_reference(
                            getattr(job, "file_path", None), source_lang,
                            [s.model_dump() if hasattr(s, "model_dump") else dict(s)
                             for s in _trans_input] if _trans_input else None,
                            job_id)
                        _tiers = project_hybrid_timings(
                            _llm_cues, whisper_en_segments=_whisper_ref,
                            source_cues=list(_trans_input) if _trans_input else None,
                            margin_s=float(getattr(settings, "HYBRID_ALIGN_MARGIN_S", 2.0)),
                            min_anchor_ratio=float(getattr(settings, "HYBRID_MIN_ANCHOR_RATIO", 0.30)),
                        )
                        translated = _llm_cues
                        logger.info(
                            "[%s] Hybrid timing projected: tier A (Whisper-EN)=%d, "
                            "tier B (source-pause)=%d, tier C (kept whole)=%d of %d cue(s)",
                            job_id, _tiers["tier_a"], _tiers["tier_b"],
                            _tiers["tier_c"], _tiers["total"])
                    except Exception as _hy_err:
                        logger.warning(
                            "[%s] Hybrid word-timing projection failed (%s) — keeping "
                            "LLM cues whole", job_id, _hy_err)
            elif getattr(settings, "SENTENCE_SEGMENTATION_ENABLED", True) and not _pre_resegmented:
                try:
                    from backend.services.sentence_segmenter import resegment_by_sentence
                    _pre_seg = len(translated)
                    translated = resegment_by_sentence(
                        [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in translated]
                    )
                    logger.info("[%s] Translated resegmentation: %d → %d segments",
                                job_id, _pre_seg, len(translated))
                except Exception as _rs_err:
                    logger.warning("[%s] Translated resegmentation failed (%s)", job_id, _rs_err)

            # ── (c cont.) Re-enforce readability in the target language ──
            # Translation changes character length dramatically — a CJK →
            # English pass typically doubles the line count. Iterate the
            # readability enforcer until the score plateaus.
            if getattr(settings, "SUBTITLE_CPS_ENFORCEMENT", True):
                try:
                    from backend.services.subtitle_formatter import (
                        enforce_readability, compute_readability_report,
                    )
                    from backend.models import TranscriptSegment as _TSr
                    _enforce_kwargs = dict(
                        max_cps=float(getattr(settings, "SUBTITLE_MAX_CPS", 20.0)),
                        max_chars_per_line=int(getattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42)),
                        min_duration_ms=int(getattr(settings, "SUBTITLE_MIN_DURATION_MS", 833)),
                        max_duration_ms=int(getattr(settings, "SUBTITLE_MAX_DURATION_MS", 9000)),
                        smart_line_breaks=bool(getattr(settings, "SUBTITLE_SMART_LINE_BREAKS", True)),
                        # Splitting is enabled, but on the LLM path it's gated to
                        # cues that now carry PROJECTED word timings (tier A/B):
                        # word_timed_split_only refuses the char-proportional time
                        # cut that scrambled word-less cues, so tier-C cues stay
                        # WHOLE while word-timed cues split at real audio pauses.
                        # Whisper-native + NMT keep their existing behavior.
                        allow_split=True,
                        word_timed_split_only=_used_llm,
                    )
                    _cur = [
                        t if isinstance(t, _TSr)
                        else _TSr(**(t.model_dump() if hasattr(t, "model_dump") else t))
                        for t in translated
                    ]
                    best_translated = list(_cur)
                    best_score = -1.0
                    for _rd_iter in range(4):
                        _cur = enforce_readability(list(_cur), **_enforce_kwargs)
                        try:
                            _sc = float(compute_readability_report(list(_cur)).get("score", 0) or 0)
                        except Exception:
                            _sc = 0.0
                        if _sc > best_score + 0.5:
                            best_score = _sc
                            best_translated = list(_cur)
                        else:
                            # Plateau reached — additional passes would only
                            # shuffle the same violations around.
                            break
                    translated = best_translated
                except Exception as _rd_err:
                    logger.warning("[%s] Post-translation readability enforcement failed (%s)",
                                   job_id, _rd_err)

            # ── (d) Final dedup on the translated, polished transcript ──
            try:
                from backend.services.transcript_dedup import (
                    collapse_adjacent_duplicates, drop_repetition_loops,
                    collapse_overlapping_duplicates, drop_scattered_duplicates,
                    collapse_repeated_runs, collapse_intra_cue_repetition,
                )
                _tl = [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in translated]
                _pre_dd = len(_tl)
                _r = 0
                _ic = 0
                # Clean repetition WITHIN a cue first (a line the translator
                # duplicated against itself), so the cross-cue passes then see
                # the cleaned text.
                if getattr(settings, "SUBTITLE_INTRA_CUE_DEDUP_ENABLED", True):
                    _tl, _ic = collapse_intra_cue_repetition(
                        _tl,
                        min_word_run=int(getattr(settings, "SUBTITLE_INTRA_CUE_MIN_WORD_RUN", 4)),
                    )
                if _used_whisper_native or _used_llm:
                    # Block-level safety net: drop a whole run of cues that
                    # reappears later (a source double-pass that survived into the
                    # translation as a repeated block).
                    _tl, _r = collapse_repeated_runs(_tl)
                _tl, _a = collapse_adjacent_duplicates(_tl)
                _tl, _o = collapse_overlapping_duplicates(_tl)
                # Repetition-loop drop on the TRANSLATED text. The SOURCE is now
                # deduped BEFORE translation (step a0-pre), so a clean source can no
                # longer seed a translated loop; this is the safety net for repeats
                # the translator ITSELF emits — a small editorial model (the LLM
                # path) routinely echoes the same safe phrase for several distinct
                # source lines ("Oh, did you see this?" ×7). Run it for the LLM and
                # Whisper-native paths (markers are held out separately and genuine
                # short interjections are CAPPED, not erased, so a real recurring
                # line survives). Left off for the pure offline-NMT path, whose 1:1
                # output the original AMV/chorus concern targeted.
                _l = 0
                _s = 0
                if _used_whisper_native or _used_llm:
                    # Scattered-dedup FIRST, on the original counts: collapse short
                    # fragments the translator echoed many times ("real breasts and
                    # you're" ×10) to the first occurrence. 4+ verbatim recurrences
                    # = an artifact; genuine 2–3× interjections are preserved. Run
                    # it before the loop-drop, whose short-interjection cap of 3
                    # would otherwise mask the high counts and leave 3× repeats.
                    _tl, _s = drop_scattered_duplicates(_tl)
                    _tl, _l = drop_repetition_loops(_tl)
                translated = _tl
                if _a or _o or _l or _s or _r or _ic:
                    logger.info(
                        "[%s] Translated transcript dedup: %d → %d (%d run, %d adj, %d overlap, %d loop, %d scattered, %d intra-cue)",
                        job_id, _pre_dd, len(translated), _r, _a, _o, _l, _s, _ic)
            except Exception as _dd_err:
                logger.warning("[%s] Translated dedup skipped (%s)", job_id, _dd_err)

            # Re-score readability on the FINAL translated transcript so the UI
            # shows the score of what viewers will actually read (coerced to
            # models — compute_readability_report uses attribute access).
            _tr_readability = None
            try:
                from backend.services.subtitle_formatter import compute_readability_report
                from backend.models import TranscriptSegment as _TSscore
                _score_models = [
                    t if isinstance(t, _TSscore)
                    else _TSscore(**(t.model_dump() if hasattr(t, "model_dump") else t))
                    for t in translated
                ]
                _tr_readability = compute_readability_report(_score_models)
                logger.info(
                    "[%s] Translated transcript readability: grade %s (%.1f/100)",
                    job_id,
                    _tr_readability.get("grade"),
                    _tr_readability.get("score", 0),
                )
            except Exception as _trd_err:
                logger.debug("[%s] Translated readability skipped: %s", job_id, _trd_err)

            # Re-insert the language-neutral music markers we held out of
            # translation so the translated track shows them too.
            _translated_out = [
                t.model_dump() if hasattr(t, "model_dump") else dict(t)
                for t in translated
            ]
            if _music_markers:
                from backend.services.audio_analyzer import merge_markers
                _translated_out = merge_markers(_translated_out, _music_markers)

            # Persist in CHRONOLOGICAL order. Readability splits + re-merged
            # ``[♪ music ♪]`` markers can leave cues out of strict time order,
            # which made the transcript PANEL (it renders the saved order)
            # reshuffle between polls even though the SRT/TXT exports re-sort.
            # Sorting here fixes the data at the source so every consumer — the
            # panel, the exports, and clip subtitle slicing — sees one stable
            # timeline order.
            try:
                _translated_out.sort(key=lambda s: (
                    float((s.get("start") if isinstance(s, dict)
                           else getattr(s, "start", 0)) or 0),
                    float((s.get("end") if isinstance(s, dict)
                           else getattr(s, "end", 0)) or 0)))
            except Exception:
                pass

            # ── Final translation QA (re-translate stragglers, don't just reject) ──
            # The definitive check on the FINAL, polished transcript: whichever
            # engine produced the draft — and after post-edit / resegmentation /
            # readability reflow — detect any cue still in the source language and
            # re-translate it one-by-one with the LLM. This is the QA the earlier
            # steps DON'T do: the transcript step dedups the SOURCE, and the
            # subtitle polisher only refines ALREADY-translated text; neither
            # re-translates source-language stragglers. Idempotent (a clean track
            # has zero leftovers → no-op) and fail-soft, so it can only help.
            try:
                _translated_out = await _llm_cleanup_untranslated(
                    _translated_out, source_lang, target_lang,
                    orchestrator, glossary, job_id)
            except Exception as _qa_err:
                logger.warning("[%s] Final translation QA cleanup skipped (%s)",
                               job_id, _qa_err)

            # ── Final purity gate (the invariant this whole effort enforces) ──
            # A "translation" must NEVER be shipped half-source-language. If the
            # finished track is substantially source-script — a bad NMT/Whisper-
            # native draft slipping through, or a corrupted merge — refuse to
            # persist it: raise so the except below records translation_failed
            # and the (labelled) source transcript is kept, instead of saving
            # Japanese under translated_transcript. Skipped for CJK targets,
            # where CJK output is correct (fraction_untranslated returns 0).
            from backend.services.translator import fraction_untranslated as _frac_unt
            _final_resid = _frac_unt(_translated_out, target_lang)
            if _final_resid > 0.20:
                raise RuntimeError(
                    f"refusing to persist a {100 * _final_resid:.0f}%-source-script "
                    f"translated_transcript (target={target_lang}) — the translation "
                    "did not complete cleanly")

            # ── Persist the translated, polished transcript ──
            # No status pin here: translation now runs BEFORE clip extraction,
            # so the job is NOT terminal yet — the COMPLETE save happens in
            # ``_run_analysis_inner`` after clips. The post-clip caption refresh
            # + Auto-SEO run from ``_run_post_clip_followups`` using ``_result``.
            # Final hygiene pass: drop any duplicate / source-language cue before
            # storing (a clean track is unchanged) so a later interrupted resume
            # has a clean base to work from.
            try:
                from backend.services.transcript_sanitize import (
                    sanitize_translated_transcript, merge_transcript_fragments)
                _san, _san_changed = sanitize_translated_transcript(_translated_out, target_lang)
                if _san_changed:
                    logger.info("[%s] Sanitized translated_transcript before persist: %d → %d",
                                job_id, len(_translated_out), len(_san))
                    _translated_out = _san
                # Fold Whisper's mid-sentence fragment splits ("It's just the" /
                # "number 21.") back into whole utterances so the translate panel
                # + SRT read as sentences, not 2-4-word slivers. Idempotent.
                _merged, _merged_changed = merge_transcript_fragments(_translated_out, target_lang)
                if _merged_changed:
                    logger.info("[%s] Merged translated_transcript fragments: %d → %d cue(s)",
                                job_id, len(_translated_out), len(_merged))
                    _translated_out = _merged
            except Exception:
                pass
            logger.info(
                "[%s] Persisting translated_transcript (%d segments)",
                job_id, len(_translated_out),
            )
            _update_kwargs = {"translated_transcript": _translated_out}
            if _tr_readability is not None:
                _update_kwargs["transcript_readability"] = _tr_readability
            await database.update_job_status(job_id, **_update_kwargs)

            # Push the FINAL (translated) readability to any open Analysis page
            # right away. The readability card reads ``job.transcript_readability``,
            # which is first written with the PRELIMINARY source-language pass
            # (the raw JA transcript, scored with CJK limits → a low grade); the
            # English re-score above is the score of what viewers actually read,
            # but without this live event the open page can keep showing the
            # stale source grade. Trim the per-cue ``violations`` so the socket
            # frame stays small — the card only reads the summary fields.
            if _tr_readability is not None:
                try:
                    _rd_summary = {k: v for k, v in _tr_readability.items()
                                   if k != "violations"}
                    await broadcast_ws(job_id, {
                        "type": "readability",
                        "transcript_readability": _rd_summary,
                    })
                except Exception:
                    pass

            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "complete",
                "message": f"Subtitles translated to {target_name} ({len(_translated_out)} segments)",
            })
            logger.info("[%s] Translate+polish complete: %d %s segments persisted "
                        "(via %s)",
                        job_id, len(_translated_out), target_lang,
                        "Whisper native translate" if _used_whisper_native else "offline NMT")

            # Visible, persisted success state (Task 3) + outcome contract.
            await _set_translation_status(job_id, "translated", None)
            _result.update(
                translated=True,
                source_transcript=_as_dicts(transcript),
                target_transcript=_translated_out,
                seo_transcript=_translated_out,
            )
            return _result

        except Exception as e:
            # Fail loudly + visibly (Task 2/3). A rate-limited / unavailable
            # model gets a specific, actionable message; we NEVER persist the
            # source under translated_transcript (the changed==0 guard above
            # raised, so nothing was relabelled). The pipeline continues to clip
            # extraction + finalization with the polished SOURCE-LANGUAGE
            # transcript (labelled as such, never as a translation), and the
            # failure is surfaced in the job's
            # translation_status + pipeline_warnings and over the websocket.
            if isinstance(e, TranslationRateLimitedError):
                _msg = ("Translation model rate-limited or unavailable — offline "
                        "model (NMT) or a paid/local model is recommended")
                _status_label = "translation_failed (rate-limited)"
            elif isinstance(e, TranslationFailedError):
                _msg = f"Translation did not complete — {str(e)[:120]}"
                _status_label = "translation_failed"
            else:
                _msg = f"Translation failed: {str(e)[:120]}"
                _status_label = "translation_failed"
            logger.error("[%s] %s: %s — keeping the source-language transcript "
                         "(deduped, NOT a translation)",
                         job_id, _status_label, e, exc_info=True)
            try:
                _record_pipeline_warning(job_id, f"Subtitle translation failed: {_msg}")
            except Exception:
                pass
            await broadcast_ws(job_id, {
                "type": "background_task",
                "task": "subtitle_translation",
                "status": "failed",
                "state": "translation_failed",
                "message": _msg,
            })
            # Visible, persisted failure state (Task 3) so a planned-but-missing
            # translation never presents as a clean COMPLETE with source subs.
            await _set_translation_status(job_id, "translation_failed", _msg)
            _result["failed_reason"] = _msg
            # Fall through: polish + dedup still run on the source transcript so
            # the job stays usable (and never ships Japanese labelled English).

    # Reached when no translation was needed OR translation failed above.
    # Polish the source transcript (if the critical path didn't already), run a
    # guaranteed dedup + timing pass (Task 5 — so a translation failure can't
    # leave duplicate cues in the shipped transcript). The polished SOURCE
    # transcript is returned so ``_run_analysis_inner`` persists IT (never raw)
    # and Auto-SEO seeds from it post-clips.
    await _polish_source_if_needed()
    await _dedup_source_transcript()
    _src_final = _as_dicts(transcript)
    _result.update(source_transcript=_src_final, seo_transcript=_src_final)
    return _result


# Reframer/planner env flags that change the engine's perception or plan.
# Folded into the engine-checkpoint signature so flipping any of them
# invalidates a saved checkpoint (the next run re-runs the engine) instead of
# silently resuming with the previous behavior.
_PLANNER_ENV_FLAGS = (
    "CLIPAI_ANIME_ANCHOR", "CLIPAI_MUSIC_BEAT_SNAP", "CLIPAI_GAMEPLAY_TRACKER",
    "CLIPAI_EDITORIAL_PRIOR", "CLIPAI_THIRDS_BIAS", "CLIPAI_GAZE_LEAD_ROOM_V2",
    "REFRAMER_MAX_SAMPLES", "REFRAMER_SAMPLE_FPS", "REFRAMER_MIN_SAMPLE_FPS",
)


def _planner_fingerprint() -> str:
    """Stable fingerprint of the env flags that steer the reframer engine."""
    return ";".join(f"{k}={os.environ.get(k, '')}" for k in _PLANNER_ENV_FLAGS)


async def run_analysis(job_id: str):
    """Execute the full analysis pipeline for a video job."""
    # Clear any stale finalization gate from a previous run so this run's
    # progress writes (and the QUEUED reset below) aren't suppressed.
    _finalizing_jobs.discard(job_id)
    # Set up cancellation event for this job
    _cancel_events[job_id] = asyncio.Event()
    sem = get_semaphore()

    # Broadcast immediately so the Analysis page shows status while waiting
    # for the semaphore (especially when another analysis is already running)
    # protect_terminal=False: this is the deliberate reset that re-runs a
    # job which may currently be COMPLETE/FAILED — it MUST be allowed to
    # move the status back out of a terminal state. Every later progress
    # write keeps the default guard on.
    await _update_progress(
        job_id, JobStatus.QUEUED, 1,
        "Preparing analysis pipeline...",
        protect_terminal=False,
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
            _finalizing_jobs.discard(job_id)
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
    # Snapshot before this run overwrites it — used to keep the Compute card
    # complete when resuming from an engine checkpoint (some rows can't be
    # rebuilt without a live engine / fresh extraction this run).
    _prior_compute_summary = getattr(job, "compute_summary", None) or {}

    # Record pipeline start time for ETA and total duration tracking
    _pipeline_start = _time.monotonic()
    _pipeline_start_iso = datetime.now(timezone.utc).isoformat()
    # Scope the process-wide OpenRouter token tally to THIS job so the cost
    # estimate captures every LLM call (translation, summary, SEO, clip judge),
    # not just the orchestrator's own providers. Jobs run sequentially here.
    try:
        from backend.services.providers.openrouter_provider import OpenRouterProvider
        OpenRouterProvider.reset_session()
    except Exception:
        pass
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

    # ── Local editorial model (Offline primary + cloud key-limit fallback) ──
    # Point the Ollama provider at the best installed model that fits the GPU.
    # In Offline Mode it's the PRIMARY editorial AI; in cloud mode it's the
    # EDITORIAL_LOCAL_FALLBACK safety net (appended to editorial_provider_chain)
    # that finishes summary/SEO/polish/translation + the clip judge locally when
    # the cloud key is exhausted (the OpenRouter "Key limit exceeded" case),
    # instead of failing. Computed once and reused for the judge specs at clip
    # detection. Non-offline runs keep the user's dropdown picks as the primary.
    _local_editorial_models: list[str] = []
    _editorial_is_local = settings.resolve_ai_source("editorial") == "local"
    _editorial_local_fallback = bool(
        settings.EDITORIAL_LOCAL_FALLBACK and (settings.OLLAMA_HOST or "").strip())
    if _editorial_is_local or _editorial_local_fallback:
        try:
            from backend.services.local_models import select_local_editorial_models
            _local_editorial_models = await select_local_editorial_models(limit=2)
            if _local_editorial_models:
                _oll = orchestrator._providers.get("ollama")
                if _oll is not None and getattr(_oll, "_editorial_model", None) is not None:
                    _oll._editorial_model = _local_editorial_models[0]
                    _oll._summary_model = _local_editorial_models[0]
                logger.info(
                    "[%s] Local editorial model '%s' wired as %s (secondary '%s')",
                    job_id, _local_editorial_models[0],
                    "Offline primary" if _editorial_is_local else "cloud key-limit fallback",
                    _local_editorial_models[1] if len(_local_editorial_models) > 1 else "none",
                )
        except Exception as _ed_e:
            logger.debug("[%s] local editorial model select skipped: %s", job_id, _ed_e)

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

    # Surface GPU status to the live Processing Log (positive confirmation, not
    # just the CPU-fallback warning). This early broadcast can race the client's
    # WebSocket connect, so it's re-stated at the reframer stage below where the
    # page is reliably subscribed. A final "Ran on GPU" confirmation also follows
    # after analysis.
    try:
        await broadcast_ws(job_id, {"type": "compute_info", "message": _gpu_status_message()})
    except Exception:
        pass

    # ── GPU preflight: evict any Ollama models still resident in VRAM ──
    # On a low-VRAM card (GTX 1650 4 GB and similar) a stray Ollama
    # model from a previous run starves Whisper of inference workspace
    # and forces a CPU fallback that's 10-20× slower than realtime.
    # Ollama itself keeps running — models auto-reload on demand the
    # next time the editorial AI is invoked, so no restart is needed.
    try:
        from backend.services.gpu_preflight import ensure_gpu_free_before_analysis
        await ensure_gpu_free_before_analysis(job_id)
    except Exception as _pre_err:
        logger.warning("[%s] GPU preflight skipped (error: %s)", job_id, _pre_err)
    _vram_snapshot("analysis_start", job_id)
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
    # ── Early browser-preview build ──
    # Build the scrub-friendly preview proxy NOW (2s GOP + faststart, NVENC
    # when available) instead of lazily after analysis. The lazy path is
    # blocked while the job is analyzing (is_job_analyzing guard in
    # serve_file), which is exactly when the user opens the player — so
    # without this the editor scrubs against the raw long-GOP source for
    # the whole run. Fire-and-forget; failures just mean the lazy path
    # builds it later.
    try:
        from backend.services.browser_preview import ensure_browser_preview

        async def _early_preview(path=video_path, jid=job_id):
            try:
                out = await asyncio.to_thread(ensure_browser_preview, path)
                logger.info("[%s] Early browser preview ready: %s", jid,
                            os.path.basename(out) if out else "(source)")
            except Exception as _bp_err:
                logger.info("[%s] Early browser preview skipped: %s", jid, _bp_err)
        asyncio.create_task(_early_preview())
    except Exception:
        pass

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
            "stage_id": "metadata",
            "stage_label": "Video Metadata",
            "stage_pct": 100,
            "stage_start": 2,
            "stage_end": 5,
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

    async def _audio_progress(fraction: float):
        # Audio extraction runs CONCURRENTLY with frame extraction and is the
        # long pole on long videos (afftdn/loudnorm are CPU-bound). GPU frame
        # extraction finishes first, so without this the UI froze on the last
        # "Extracted N frames" message while ffmpeg ground on — looking stuck
        # at the next step (faces) with no VRAM in use. Reporting live audio
        # progress keeps the message accurate and resets the stuck-timer.
        f = max(0.0, min(1.0, fraction))
        pct = 8 + int(f * 6)  # same 8-14% band as frames
        await _update_progress(
            job_id, JobStatus.EXTRACTING_FRAMES, pct,
            f"Preconditioning audio for transcription… ({int(f * 100)}%)",
        )

    # ── Re-analyze cache: if frames + audio already exist on disk AND
    #    the source SHA-256 matches what's recorded on the job, skip
    #    the (expensive) re-extract. Saves 30–60% of total wall time
    #    on iterative re-analysis runs (clip tuning loops).
    #
    # Disable with CLIPAI_FORCE_REEXTRACT=1.
    _force_reextract = os.environ.get("CLIPAI_FORCE_REEXTRACT", "").lower() in ("1", "true", "yes")
    # Tracked across both extraction branches so the engine checkpoint (saved
    # after the reframer stage) can key on the exact source bytes. On a cache
    # hit this is already the matched SHA; on a fresh extract it's filled in
    # once hashing succeeds below.
    _source_sha256_local = getattr(job, "source_sha256", "") or ""
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
            await _update_progress(
                job_id, JobStatus.EXTRACTING_FRAMES, 14,
                f"Re-using cached extraction — {len(frames)} frames (SHA matched)",
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
                            video_duration=metadata["duration"],
                            progress_callback=_audio_progress,
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
                _source_sha256_local = _sha
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
        to_fez_subject_track, serialize_detection_overlay,
    )

    _log_gpu_memory(job_id, "pre-reframer")
    _loop = asyncio.get_running_loop()
    _perceive_t0 = _time.monotonic()

    def _fmt_eta(secs: float) -> str:
        secs = int(max(0, secs))
        if secs >= 3600:
            return f"{secs // 3600}h {(secs % 3600) // 60}m"
        if secs >= 60:
            return f"{secs // 60}m {secs % 60}s"
        return f"{secs}s"

    _engine_phase = ['faces']  # mutable cell: 'faces' | 'whisper_pending' | 'whisper'
    _whisper_t0 = [0.0]
    _last_emitted_pct = [15]   # monotonic floor for the ANALYZING_SCENES band

    def _engine_progress(*pargs):
        """Thread-safe progress relay from the (blocking) reframer engine.

        The Perceiver reports a 0.0-1.0 fraction; map it onto 15-58% of the
        overall pipeline bar, derive an ETA from the observed rate, and
        schedule the async update on the event loop.

        A second positional arg is an optional PHASE HINT emitted at the
        sub-stages that used to go dark for minutes (post-transcribe
        refinement, speaker diarization). With a hint we label the bar and
        the heartbeat with the real work instead of leaving a stale
        "transcription — Xm elapsed" (or worse, "scene analysis") on screen
        through an 8-minute diarization pass.
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
        hint = str(pargs[1]) if len(pargs) > 1 and pargs[1] else ""

        pct_int = None
        msg = ""
        hb_label = ""
        if hint == 'transcript_refine':
            pct_int = 56
            msg = "Refining transcript (redecode + gap-fill + alignment)..."
            hb_label = "transcript refinement"
            _engine_phase[0] = 'whisper'   # % framework: transcription band done
        elif hint == 'diarization':
            pct_int = 57
            msg = "Speaker diarization + voice matching..."
            hb_label = "speaker diarization"
        else:
            # Phase detection: face detection reports 0→1, then Whisper
            # restarts from 0. Detect the restart by watching for frac≥1
            # then a new call with frac<0.5.
            if _engine_phase[0] == 'faces' and frac >= 0.999:
                _engine_phase[0] = 'whisper_pending'
            elif _engine_phase[0] == 'whisper_pending' and frac < 0.5:
                _engine_phase[0] = 'whisper'
                _whisper_t0[0] = _time.monotonic()

            phase = _engine_phase[0]
            elapsed = _time.monotonic() - _perceive_t0

            # ``hb_label`` names the actual sub-phase so the heartbeat (which
            # fires during the long stretches where the % stalls — e.g.
            # Whisper's post-VAD merge) reads "Still processing...
            # (transcription — Xm)" instead of the misleading "scene analysis".
            if phase == 'faces':
                pct_int = int(15 + frac * 27)  # 15% → 42%
                pct_int = max(15, min(42, pct_int))
                msg = f"Detecting faces + motion analysis ({int(frac * 100)}%)"
                hb_label = "face + motion detection"
                if frac >= 0.02 and elapsed > 15:
                    eta = elapsed * (1.0 - frac) / max(frac, 0.01)
                    msg += f" — {_fmt_eta(eta)} left"
            elif phase == 'whisper_pending':
                pct_int = 42
                msg = "Releasing face detection models — freeing GPU for Whisper..."
                hb_label = "transcription"
            else:  # whisper
                pct_int = int(42 + frac * 14)  # 42% → 56%
                pct_int = max(42, min(56, pct_int))
                w_elapsed = _time.monotonic() - _whisper_t0[0]
                msg = f"Transcribing audio with Whisper ({int(frac * 100)}%)"
                hb_label = "transcription"
                if frac >= 0.02 and w_elapsed > 5:
                    eta = w_elapsed * (1.0 - frac) / max(frac, 0.01)
                    msg += f" — {_fmt_eta(eta)} left"

        # Forward-only: interleaved callbacks (concurrent audio precondition
        # vs face sampling; a late straggler after a phase flip) used to make
        # the bar jump BACKWARDS in the activity log. Clamp to the highest
        # percent already shown; the message still updates so sub-phase text
        # stays truthful.
        pct_int = max(int(pct_int), _last_emitted_pct[0])
        _last_emitted_pct[0] = pct_int

        try:
            asyncio.run_coroutine_threadsafe(
                _update_progress(
                    job_id, JobStatus.ANALYZING_SCENES, pct_int, msg,
                    heartbeat_label=hb_label,
                ),
                _loop,
            )
        except Exception:
            pass  # progress is best-effort — never break analysis over it

    # Adaptive sampling — the Perceiver's face/motion cost scales linearly
    # with the number of frames it analyses (the dominant analysis cost on
    # long videos). Cap the total sample count so a long video stays
    # tractable; short videos keep the higher per-frame detail. The cap,
    # ceiling, and floor are all configurable so the face stage can be sped
    # up (fewer samples) without touching code. Defaults lowered from
    # 3600/5.0/2.0 to 1800/5.0/1.2 — roughly halving the face-detection time
    # on long videos with negligible reframing-accuracy loss.
    _max_samples = max(300, int(getattr(settings, "REFRAMER_MAX_SAMPLES", 1800)))
    _fps_ceiling = float(getattr(settings, "REFRAMER_SAMPLE_FPS", 5.0))
    _fps_floor = float(getattr(settings, "REFRAMER_MIN_SAMPLE_FPS", 1.2))
    # Cap-respecting sample rate — see resolve_sample_fps. The comfort floor
    # only raises the rate for shorter videos; it never blows past _max_samples
    # on long ones (which used to make face detection ~5x slower than intended).
    _sample_fps = resolve_sample_fps(
        video_duration,
        max_samples=_max_samples,
        fps_ceiling=_fps_ceiling,
        fps_floor=_fps_floor,
        abs_floor=float(getattr(settings, "REFRAMER_ABS_MIN_SAMPLE_FPS", 0.2)),
    )
    logger.info(
        "[%s] Reframer sample rate: %.2f fps (%.1f min video, ~%d samples, cap=%d)",
        job_id, _sample_fps, video_duration / 60.0,
        int(_sample_fps * video_duration) if video_duration > 0 else 0, _max_samples,
    )

    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 15,
        "Starting reframer analysis (faces, transcription, motion)...",
        heartbeat_label="face + motion detection",
    )

    # Re-state GPU status now that the page is reliably subscribed (the
    # pipeline-start broadcast can fire before the client's WebSocket connects,
    # dropping it). The reframer + Whisper are the long GPU stages just ahead.
    try:
        await broadcast_ws(job_id, {"type": "compute_info", "message": _gpu_status_message()})
    except Exception:
        pass

    # Second GPU preflight right before reframer/Whisper kicks in. The
    # analysis-start preflight already evicted Ollama; this catches
    # anything the frame-extraction stage may have leaked or the
    # editorial AI may have auto-loaded for a probe call.
    try:
        from backend.services.gpu_preflight import ensure_gpu_free_before_whisper
        await ensure_gpu_free_before_whisper(job_id)
    except Exception as _pre_err:
        logger.warning("[%s] pre-Whisper GPU preflight skipped: %s", job_id, _pre_err)
    _vram_snapshot("pre_whisper", job_id)

    # Per-job JSONL trace of every reframe decision. Lands next to
    # render_plan.json / detection_overlay.json so all the reframer
    # artifacts live together. The engine handles an empty path as
    # "tracing disabled" so callers that don't pass one continue to
    # work unchanged.
    _trace_path = os.path.join(job_dir, "reframe_trace.jsonl")
    # Thread the user-selected source language from the upload page into the
    # engine so Whisper transcribes with the chosen language instead of always
    # auto-detecting. Blank stays "auto" (auto-detect). The engine forwards
    # this to AudioIntelligence.transcribe(language=...), so the log will show
    # ``language=ja`` (or whatever was picked) rather than ``language=auto``.
    _engine_source_lang = (job.language or "auto").strip() or "auto"
    logger.info(
        "[%s] Reframer source language=%s, subtitle target=%s",
        job_id, _engine_source_lang, (job.subtitle_language or "").strip() or "(same as source)",
    )
    # ── Engine checkpoint probe (RESUME) ──
    # The reframer engine (face/motion detection + Whisper transcription +
    # planning) is the single most expensive stage. If a previous run got this
    # far and saved a checkpoint for THIS exact source + analysis config,
    # restore its perception + plan and skip straight to bridge → summary →
    # clips. This is what lets a failed/interrupted job "continue where it left
    # off" instead of re-detecting and re-transcribing from scratch. Bypass
    # with CLIPAI_FORCE_REANALYZE=1.
    _engine_ckpt_signature = pipeline_checkpoint.checkpoint_signature(
        source_sha=_source_sha256_local,
        source_language=_engine_source_lang,
        sample_fps=_sample_fps,
        aspect_ratio="9:16",
        vocal_sep_key=(
            f"{bool(getattr(settings, 'VOCAL_SEPARATION_ENABLED', False))}:"
            f"{getattr(settings, 'VOCAL_SEPARATION_MODEL', 'htdemucs')}"
        ),
        planner_fingerprint=_planner_fingerprint(),
    )
    _force_reanalyze = os.environ.get(
        "CLIPAI_FORCE_REANALYZE", "").lower() in ("1", "true", "yes")
    _resumed_from_checkpoint = False
    engine = None
    reframer_plan = None
    perception = None
    if not _force_reanalyze:
        try:
            _ckpt = await pipeline_checkpoint.load_engine_checkpoint(
                job_id, signature=_engine_ckpt_signature)
        except Exception as _ckpt_probe_err:
            logger.info("[%s] checkpoint probe failed (%s); running engine fresh",
                        job_id, _ckpt_probe_err)
            _ckpt = None
        if _ckpt is not None:
            perception, reframer_plan, engine = _ckpt
            _resumed_from_checkpoint = True
            _record_pipeline_warning(
                job_id,
                "Resumed from a saved checkpoint — reused detection, "
                "transcription, and the reframe plan from a previous run.")
            try:
                await broadcast_ws(job_id, {
                    "type": "checkpoint",
                    "message": (
                        "Resumed from checkpoint — reused saved detection + "
                        "transcription (skipped re-analysis)."),
                })
            except Exception:
                pass
            await _update_progress(
                job_id, JobStatus.ANALYZING_SCENES, 56,
                "Resuming — reusing saved detection + transcription...")
            logger.info(
                "[%s] RESUME: restored engine checkpoint (%d transcript segs) — "
                "skipping detection + transcription", job_id,
                len(getattr(perception, "transcript_segments", None) or []))

    # ── Vocal separation (Demucs) before ASR ──
    # Isolate the vocal stem so dialogue buried under loud music/SFX (which the
    # VAD otherwise hears as no-speech and drops) gets transcribed. Runs here —
    # BEFORE the engine loads Whisper/YOLO — as a subprocess, so it gets the
    # whole GPU and frees it on exit. Any failure (not installed, OOM, codec)
    # returns None and the engine transcribes the raw audio unchanged.
    # Skipped entirely on a resumed run (the transcript is already restored).
    _vocals_path = None
    _vocals_future = None
    _vs_executor = None
    if not _resumed_from_checkpoint and getattr(settings, "VOCAL_SEPARATION_ENABLED", False):
        try:
            from backend.services.vocal_separator import separate_vocals, is_available
            if is_available():
                _vs_kwargs = dict(
                    model=getattr(settings, "VOCAL_SEPARATION_MODEL", "htdemucs"),
                    device=getattr(settings, "VOCAL_SEPARATION_DEVICE", "auto"),
                    segment=int(getattr(settings, "VOCAL_SEPARATION_SEGMENT", 7)),
                    timeout=int(getattr(settings, "VOCAL_SEPARATION_TIMEOUT", 1800)),
                )
                # Concurrency (audit Phase 5.5): Demucs and the visual
                # perception pass use different resources most of the run,
                # so overlap them — the perceiver blocks on the future only
                # when it reaches transcription. Gated to GPUs with >=6 GB
                # total VRAM: on a 4 GB card Demucs-on-GPU + YOLO-on-GPU
                # would fight for the budget, so small cards keep the
                # proven sequential order (whole GPU per stage).
                _vs_concurrent = bool(getattr(
                    settings, "VOCAL_SEPARATION_CONCURRENT", True))
                if _vs_concurrent:
                    try:
                        from backend.services.vram_ledger import _query_vram
                        _vram = _query_vram()
                        if _vram is not None and _vram[1] < 6000:
                            _vs_concurrent = False
                    except Exception:
                        _vs_concurrent = False
                if _vs_concurrent:
                    import concurrent.futures as _cf
                    _vs_executor = _cf.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="vocal-sep")
                    _vocals_future = _vs_executor.submit(
                        separate_vocals, video_path,
                        os.path.join(job_dir, "demucs"), **_vs_kwargs)
                    logger.info(
                        "[%s] Vocal separation started CONCURRENT with "
                        "perception (GPU has headroom)", job_id)
                else:
                    async with _stage_timer(job_id, "vocal_separation"):
                        _vocals_path = await asyncio.to_thread(
                            separate_vocals, video_path,
                            os.path.join(job_dir, "demucs"), **_vs_kwargs)
                    logger.info(
                        "[%s] Vocal separation: %s", job_id,
                        f"transcribing isolated vocals ({os.path.basename(_vocals_path)})"
                        if _vocals_path else "no output — transcribing full audio",
                    )
            else:
                logger.info(
                    "[%s] VOCAL_SEPARATION_ENABLED but demucs not installed "
                    "(pip install demucs) — transcribing full audio", job_id)
        except Exception as _vs_err:
            logger.warning(
                "[%s] Vocal separation failed (%s) — transcribing full audio",
                job_id, _vs_err)
            _vocals_path = None

    if _vocals_future is not None:
        _vs_timeout = int(getattr(settings, "VOCAL_SEPARATION_TIMEOUT", 1800)) + 120

        def _await_vocals():
            # Runs on the engine's worker thread right before transcription
            try:
                return _vocals_future.result(timeout=_vs_timeout)
            except Exception as _e:
                logger.warning(
                    "[%s] Concurrent vocal separation failed (%s) — "
                    "transcribing full audio", job_id, _e)
                return None
        _engine_stem = _await_vocals
    else:
        _engine_stem = _vocals_path

    if not _resumed_from_checkpoint:
        engine = ReframeEngine(video_path, sample_fps=_sample_fps,
                               aspect_ratio="9:16", trace_path=_trace_path,
                               source_language=_engine_source_lang,
                               transcribe_audio_path=_engine_stem)
        async with _stage_timer(job_id, "reframer_analysis"):
            reframer_plan = await asyncio.to_thread(engine.analyze, _engine_progress)
        perception = engine.perception
        _log_gpu_memory(job_id, "post-reframer")
        if _vs_executor is not None:
            _vs_executor.shutdown(wait=False)

        # Checkpoint the (expensive) engine output so a failed/interrupted run
        # can RESUME here next time — skipping detection + transcription —
        # instead of re-running the whole engine. Best-effort: never break the
        # run over a serialization hiccup.
        _ckpt_saved = await pipeline_checkpoint.save_engine_checkpoint(
            job_id, perception, reframer_plan,
            signature=_engine_ckpt_signature,
            audio_meta={
                "device": getattr(engine, "_perceiver_audio_device", None),
                "model": getattr(engine, "_perceiver_audio_model", None),
                "requested": getattr(engine, "_perceiver_audio_model_requested", None),
            },
        )
        # Surface a CHECKPOINT marker in the processing log so the user can see
        # exactly where a restart will pick up — detection + transcription are
        # now saved; a crash after this point resumes here instead of re-running
        # the most expensive ~25 min stage.
        if _ckpt_saved:
            try:
                await broadcast_ws(job_id, {
                    "type": "checkpoint",
                    "message": (
                        "Checkpoint reached — detection + transcription saved. "
                        "If the job restarts, it resumes from here."),
                })
            except Exception:
                pass

    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 57,
        "Finalizing speaker analysis + freeing GPU memory...",
        heartbeat_label="speaker analysis wrap-up",
    )
    # Whisper ran inside the Perceiver. Normally free its VRAM now (before the
    # VLM/summary stage). EXCEPTION: when this job will do a Whisper-native
    # audio→English translate, KEEP the engine loaded so that pass REUSES it
    # (no second load) — the translate step frees the VRAM afterwards, and a
    # defensive release runs right before the summary regardless of path.
    # On a RESUMED run no Whisper was loaded (the transcript came from the
    # checkpoint), so this release/keep dance is a no-op and is skipped.
    if not _resumed_from_checkpoint:
        _det_lang_pp = (getattr(perception, "detected_language", "") or "").strip().lower()
        _pp_tgt = (job.subtitle_language or "").strip().lower()
        _pp_src = (job.language or "").strip().lower() or _det_lang_pp
        if not _pp_tgt and _pp_src and _pp_src not in ("en", "english"):
            _pp_tgt = "en"  # auto-translate non-English → English
        _keep_whisper_for_translate = (
            bool(getattr(settings, "WHISPER_TRANSLATE_TO_EN", True))
            and _pp_tgt == "en"
            and _pp_src not in ("en", "english", "")
            and bool(getattr(perception, "transcript_segments", None))
        )
        if _keep_whisper_for_translate:
            logger.info(
                "[%s] Keeping Whisper engine loaded for native audio→English translate "
                "(reuse — no second load on small GPUs)", job_id)
        else:
            await _release_whisper_vram(job_id)
            _log_gpu_memory(job_id, "post-whisper-release")
            _vram_snapshot("post_whisper_release", job_id)

    _n_face_samples = sum(1 for v in (perception.face_timeline or {}).values() if v)
    logger.info(
        "[%s] Reframer analysis complete: %d face samples, %d scene cuts, "
        "%d transcript segments, %d keyframes, %d strategies",
        job_id, _n_face_samples, len(perception.scene_cuts or []),
        len(perception.transcript_segments or []),
        len(reframer_plan.keyframes or []),
        len(reframer_plan.strategy_log or []),
    )
    # ── Cross-video contamination trace ───────────────────────────────────────
    # Prove the analyzed audio/video belong to THIS job and show the RAW Whisper
    # source: if a later "transcript belongs to a different video" report comes
    # in, this pins whether the source transcript was already wrong at perception
    # time (wrong/stale audio → upstream) or only became wrong downstream.
    try:
        import os as _os
        _vid = f"/data/uploads/{job_id}/video.mp4"
        _aud = f"/data/uploads/{job_id}/audio.wav"
        _vsz = _os.path.getsize(_vid) if _os.path.exists(_vid) else -1
        _asz = _os.path.getsize(_aud) if _os.path.exists(_aud) else -1
        _segs0 = (getattr(perception, "transcript_segments", None) or [])[:4]
        _raw = " | ".join((getattr(s, "text", "") or "")[:50] for s in _segs0)
        logger.info(
            "[%s] PERCEIVE-TRACE video=%.1fMB audio=%.1fMB lang=%s raw_source[0:4]: %s",
            job_id, _vsz / 1e6, _asz / 1e6,
            getattr(perception, "detected_language", "?"), _raw)
    except Exception:
        pass

    # ── Bridge — convert reframer output into Fez data contracts ──
    cancel_check()
    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 60,
        "Converting analysis into render plan + scenes...",
        heartbeat_label="render plan conversion",
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
        subject_track = to_fez_subject_track(perception, reframer_plan)

    # Loud, visible signal when transcription came back empty. Without a
    # transcript the pipeline silently skips subtitle translation AND produces
    # an empty summary ("no transcript was available") — so surface it as a job
    # warning + WS notice instead of letting the run look cleanly COMPLETE. The
    # usual cause is a transcription/audio-extraction failure (see the [AUDIO]
    # error in the logs), not a genuinely silent video.
    if not transcript:
        _record_pipeline_warning(
            job_id,
            "Transcription produced no segments — subtitle translation and the "
            "text summary were skipped. Check the audio track / [AUDIO] log lines.")
        logger.warning(
            "[%s] No transcript segments after analysis — translation + summary "
            "will be skipped (likely a transcription/audio-extraction failure).",
            job_id)
        try:
            await broadcast_ws(job_id, {
                "type": "compute_warning",
                "message": ("No speech transcript was produced — translation and "
                            "summary skipped. Check the source audio."),
            })
        except Exception:
            pass

    # Whisper's detected language — hoisted here so the critical-path polish
    # block below can pass it to the polisher (CJK-specific rules). It was
    # previously only assigned much later, which raised UnboundLocalError in
    # the polish try/except and silently disabled the LLM polish pass.
    _detected_lang = getattr(perception, "detected_language", "") or ""

    # ── Will this job translate? (translate-then-polish decision) ──
    # When a translation pass will follow, we deliberately SKIP the heavy
    # source-language polish / resegmentation / readability reflow below and
    # leave that work for AFTER translation, so the LLM polishes the TARGET
    # language text (English) rather than the source (Japanese). Mirrors the
    # target/source resolution in _background_post_processing, including the
    # "auto-translate non-English → English when no explicit subtitle_language"
    # default, so the two decisions never disagree.
    _bg_target = (job.subtitle_language or "").strip().lower()
    _bg_source = (job.language or "").strip().lower() or _detected_lang.strip().lower()
    if not _bg_target and _bg_source and _bg_source not in ("en", "english"):
        _bg_target = "en"
    _will_translate = bool(_bg_target and _bg_target != _bg_source and transcript)
    logger.info(
        "[%s] Post-processing plan: source=%s target=%s → %s",
        job_id, _bg_source or "auto", _bg_target or "(none)",
        "translate-then-polish in target language" if _will_translate
        else "polish in source language (no translation)",
    )

    # ── Speaker fusion (Task 2): overlap voting + mid-segment splits +
    # word-level regrouping. Replaces to_fez_transcript's basic per-segment
    # majority vote with a proper diarization→transcript merge, so speaker
    # turns follow the diarization timeline instead of acoustic windows.
    # The mouth-motion heuristic remains the documented fallback: an empty
    # speaker_timeline leaves the transcript untouched.
    _speaker_timeline = getattr(perception, "speaker_timeline", None)
    if _speaker_timeline and transcript:
        try:
            from backend.services.speaker_fusion import assign_speakers_from_timeline
            from backend.services.reframer_bridge import _speaker_label_map
            _pre_fusion = len(transcript)
            _label_map = _speaker_label_map(_speaker_timeline)
            _fused = assign_speakers_from_timeline(
                transcript, _speaker_timeline, label_map=_label_map,
            )
            transcript = [
                f.model_dump() if hasattr(f, "model_dump") else dict(f)
                for f in _fused
            ]
            logger.info(
                "[%s] Speaker fusion: %d → %d segments (%d speakers)",
                job_id, _pre_fusion, len(transcript), len(_label_map),
            )
        except Exception as _fusion_err:
            logger.warning(
                "[%s] Speaker fusion failed (%s) — keeping basic attribution",
                job_id, _fusion_err,
            )

    # ── Voiceprint matching (Task 3): auto-apply names learned in prior
    # jobs. No-ops gracefully when the pyannote embedding backend / HF_TOKEN
    # are absent — the per-job "Speaker N" labels simply stand.
    if _speaker_timeline and transcript:
        _vram_snapshot("pre_voiceprint", job_id)
        try:
            from backend.services.voiceprint_registry import apply_voiceprint_names
            from backend.services.reframer_bridge import _speaker_label_map
            transcript = apply_voiceprint_names(
                transcript, _speaker_timeline, video_path,
                label_map=_speaker_label_map(_speaker_timeline),
            )
        except Exception as _vp_err:
            logger.warning(
                "[%s] Voiceprint matching skipped (%s)", job_id, _vp_err)

    # ── Synchronous transcript polish (BEFORE readability) ──
    # Without this, CJK content (Japanese narration, K-drama dialogue)
    # arrives as 30s blocks with zero 。 — the readability splitter falls
    # back to particle-boundary guesses. Running the LLM polisher first
    # gives the splitter actual sentence punctuation to break on, which
    # is the single biggest lever on transcript readability.
    _polished_in_critical_path = False
    if _will_translate:
        # Translate-then-polish: the heavy readability reflow runs in the TARGET
        # language after translation, so we don't spend an LLM pass punctuating
        # Japanese we're about to replace with English. BUT (Task 6) apply a
        # single light SOURCE cleanup (punctuation / casing / filler) first so
        # the translator works from clean input AND the shipped source
        # transcript reads cleanly — both benefit. Fail-soft + gated by
        # TRANSLATION_POLISH_SOURCE_FIRST; the full reflow still runs on the
        # translated text. (_polished_in_critical_path stays False so a
        # translation FAILURE still triggers the full source polish fallback.)
        if (settings.AI_TRANSCRIPT_CORRECTION
                and getattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True)
                and getattr(settings, "TRANSLATION_POLISH_SOURCE_FIRST", True)
                and transcript):
            try:
                cancel_check()
                from backend.services.transcript_polisher import (
                    polish_source_before_translation,
                )
                _src_polished = await polish_source_before_translation(
                    transcript, orchestrator,
                    source_language=(_detected_lang or "").lower(), job_id=job_id,
                )
                if _src_polished and len(_src_polished) == len(transcript):
                    transcript = [
                        p.model_dump() if hasattr(p, "model_dump") else dict(p)
                        for p in _src_polished
                    ]
                    logger.info(
                        "[%s] Light source-language polish applied before translation "
                        "(%d segments) — translator + shipped source both benefit",
                        job_id, len(transcript))
            except Exception as _sp_err:
                logger.warning(
                    "[%s] Pre-translation source polish skipped (%s) — keeping raw "
                    "source", job_id, _sp_err)
        else:
            logger.info(
                "[%s] Skipping source-language polish before translation "
                "(disabled) — heavy polish runs on the translated text", job_id)
        logger.info(
            "[%s] Skipping source-language resegment + reflow "
            "(translation pending — heavy reflow runs on the translated text)",
            job_id,
        )
    elif (
        settings.AI_TRANSCRIPT_CORRECTION
        and getattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True)
        and transcript
    ):
        try:
            cancel_check()
            await _update_progress(
                job_id, JobStatus.ANALYZING_SCENES, 60,
                "Polishing transcript (punctuation + proper-noun fixes)...",
            )
            # Pass Whisper's detected language so the polisher applies
            # CJK-specific rules (insert 。/、, smaller batches, no
            # filler removal) when appropriate.
            _correction_lang = (_detected_lang or "").lower()
            _polished_models, _polish_report = await _polish_transcript_loop(
                job_id, transcript, orchestrator, _correction_lang,
                model_override=_resolve_polish_model_override(orchestrator),
            )
            if _polished_models:
                transcript = [
                    p.model_dump() if hasattr(p, "model_dump") else dict(p)
                    for p in _polished_models
                ]
                _polished_in_critical_path = True
                logger.info(
                    "[%s] Critical-path polish complete: %d segments, readability %s",
                    job_id, len(transcript),
                    f"{_polish_report.get('score', 0):.1f}/100" if _polish_report else "(no score)",
                )
        except Exception as _polish_err:
            logger.warning(
                "[%s] Critical-path polish failed (%s) — falling back to raw transcript",
                job_id, _polish_err,
            )

    # ── Sentence-aware resegmentation (Task 4) ──
    # Merge same-speaker neighbours then re-split at sentence boundaries
    # (using word timestamps), so the now-polished transcript breaks by
    # sentence rather than raw VAD window. Runs after speaker fusion +
    # polish (which adds the punctuation this relies on) and before the
    # readability pass, which enforces duration/CPS on the result. Skipped
    # when a translation will follow — resegmentation happens on the
    # translated English text instead (it relies on punctuation the
    # post-translation polish adds).
    if getattr(settings, "SENTENCE_SEGMENTATION_ENABLED", True) and transcript and not _will_translate:
        try:
            from backend.services.sentence_segmenter import resegment_by_sentence
            _pre_resegment = len(transcript)
            _reseg = resegment_by_sentence(transcript)
            transcript = [
                t.model_dump() if hasattr(t, "model_dump") else dict(t)
                for t in _reseg
            ]
            logger.info(
                "[%s] Sentence resegmentation: %d → %d segments",
                job_id, _pre_resegment, len(transcript),
            )
        except Exception as _reseg_err:
            logger.warning(
                "[%s] Sentence resegmentation failed (%s) — keeping segments",
                job_id, _reseg_err,
            )

    # ── Apply readability rules to the (now-polished) transcript ──
    # Whisper emits one segment per VAD-detected speech window, which on
    # dialogue-dense content (Japanese narration, podcasts) ends up as
    # 30 s blocks of un-broken text — unreadable as subtitles. Run the
    # Netflix/YouTube/TikTok-style enforcer here so the on-screen captions
    # and the transcript panel are both segmented to readable chunks
    # BEFORE translation runs. Translation later applies the enforcer
    # again on its own output to handle character-density changes
    # (CJK → English typically doubles segment length). Skipped when a
    # translation will follow — the reflow runs on the translated English
    # text (post-translation) instead, so we don't reflow source segments
    # we're about to discard.
    if getattr(settings, "SUBTITLE_CPS_ENFORCEMENT", True) and transcript and not _will_translate:
        try:
            from backend.services.subtitle_formatter import (
                enforce_readability, compute_readability_report,
            )
            from backend.models import TranscriptSegment
            _ts_models = [
                t if isinstance(t, TranscriptSegment) else TranscriptSegment(**t)
                for t in transcript
            ]
            # Build the readability kwargs from config so the source-language
            # (no-translate) path honors the same settings as the polish loop
            # and the translation path. Without this it fell through to the
            # function-signature defaults (notably max_duration_ms) and ignored
            # SUBTITLE_MAX_DURATION_MS / SUBTITLE_MAX_CPS / SUBTITLE_MAX_CHARS_PER_LINE
            # / SUBTITLE_MIN_DURATION_MS / SUBTITLE_SMART_LINE_BREAKS — a slow
            # English cue between 4.5s and 9s got split that the configured 9s
            # cap would have kept whole.
            _src_enforce_kwargs = dict(
                max_cps=float(getattr(settings, "SUBTITLE_MAX_CPS", 20.0)),
                max_chars_per_line=int(getattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42)),
                min_duration_ms=int(getattr(settings, "SUBTITLE_MIN_DURATION_MS", 833)),
                max_duration_ms=int(getattr(settings, "SUBTITLE_MAX_DURATION_MS", 9000)),
                smart_line_breaks=bool(getattr(settings, "SUBTITLE_SMART_LINE_BREAKS", True)),
            )
            # Iterate the readability enforcer until the score plateaus.
            # Single-pass leaves cascade artifacts (Pass 2 extends a short
            # segment, Pass 4 caps it back below min_dur, score stays low).
            _readable = _ts_models
            _best_readable = list(_ts_models)
            _best_score = -1.0
            for _ in range(4):
                _readable = enforce_readability(list(_readable), **_src_enforce_kwargs)
                try:
                    _sc = float(compute_readability_report(list(_readable)).get("score", 0) or 0)
                except Exception:
                    _sc = 0.0
                if _sc > _best_score + 0.5:
                    _best_score = _sc
                    _best_readable = list(_readable)
                else:
                    break
            _readable = _best_readable
            transcript = [
                t.model_dump() if hasattr(t, "model_dump") else dict(t)
                for t in _readable
            ]
            logger.info(
                "[%s] Raw transcript reflowed for readability: %d → %d segments",
                job_id, len(_ts_models), len(transcript),
            )
            # Once-per-job timing-provenance summary: how many cues are
            # word-timed vs. char-proportional. Makes timing-quality regressions
            # (e.g. polish wiping word timing) visible at a glance.
            try:
                from backend.services.sentence_segmenter import timing_provenance_report
                _tp = timing_provenance_report(transcript)
                logger.info(
                    "[%s] Cue timing provenance: %d/%d word-timed (%.1f%%), "
                    "%d char-proportional",
                    job_id, _tp["word_timed"], _tp["total"],
                    _tp["pct_word_timed"], _tp["proportional"],
                )
            except Exception:
                pass
        except Exception as _re_err:
            logger.warning(
                "[%s] Raw transcript readability pass failed (%s) — keeping Whisper output as-is",
                job_id, _re_err,
            )

    # ── Music suppression + marking (Task 4) ──
    # In sustained music-only spans (OP/ED themes, insert songs), DROP Whisper's
    # hallucinated lyrics/vocalisations and positively label the span with a
    # "[♪ music ♪]" marker instead. Dialogue OVER music is classified `speech`
    # (not `music`) by the spectral classifier, so real dialogue is untouched.
    # Markers are language-neutral and pass through the translator verbatim.
    # One classify pass feeds both suppression and marking. No-ops gracefully
    # if audio/numpy is missing.
    if getattr(settings, "SUBTITLE_MARK_MUSIC", True) and transcript:
        try:
            _audio_wav = os.path.join(job_dir, "audio.wav")
            if os.path.isfile(_audio_wav):
                from backend.services.audio_analyzer import mark_and_suppress_music
                transcript, _n_suppressed, _n_markers = await mark_and_suppress_music(
                    _audio_wav, transcript,
                    min_seconds=float(getattr(settings, "SUBTITLE_MUSIC_MIN_SEC", 5.0)),
                    suppress=bool(getattr(settings, "SUBTITLE_SUPPRESS_SPEECH_IN_MUSIC", True)),
                    min_overlap_frac=float(getattr(settings, "SUBTITLE_MUSIC_SUPPRESS_OVERLAP", 0.6)),
                    vocalizations_only=bool(getattr(settings, "SUBTITLE_MUSIC_SUPPRESS_VOCALIZATIONS_ONLY", True)),
                )
                if _n_suppressed or _n_markers:
                    logger.info(
                        "[%s] Music suppression+marking: dropped %d hallucinated "
                        "speech cue(s) over music, inserted %d [♪ music ♪] cue(s)",
                        job_id, _n_suppressed, _n_markers,
                    )
        except Exception as _mm_err:
            logger.warning(
                "[%s] Music suppression/marking skipped (%s)", job_id, _mm_err)

    # ── Final de-duplication pass ──
    # The per-segment hallucination filters run inside the Whisper stage, but
    # speaker fusion, sentence resegmentation and the readability reflow can
    # all re-introduce duplicates downstream: back-to-back identical cues
    # (observed as ``[11:25] …`` twice in a row) and scattered repetition-loop
    # hallucinations (the garbled ``ドーリアンリ`` name repeated 8× across the
    # episode). Run BOTH collapses one last time on the fully-assembled
    # transcript — this is the version that gets persisted, translated and
    # exported, so it's the one the user actually sees in the TXT/SRT/VTT.
    if transcript:
        try:
            from backend.services.transcript_dedup import (
                collapse_adjacent_duplicates, drop_repetition_loops,
                collapse_overlapping_duplicates,
            )
            _pre_dedup = len(transcript)
            transcript, _adj = collapse_adjacent_duplicates(transcript)
            # Overlap + similarity collapse catches near-duplicate
            # re-transcriptions that aren't strictly adjacent or identical
            # (the gap-fill pass landing the same line a few hundred ms off
            # the primary cue). TranscriptSegment dicts use start/end keys.
            transcript, _ovl = collapse_overlapping_duplicates(transcript)
            transcript, _loop = drop_repetition_loops(transcript)
            if _adj or _ovl or _loop:
                logger.info(
                    "[%s] Final transcript dedup: %d → %d segments "
                    "(%d adjacent dup, %d overlapping near-dup, %d repetition-loop)",
                    job_id, _pre_dedup, len(transcript), _adj, _ovl, _loop,
                )
        except Exception as _dd_err:
            logger.warning(
                "[%s] Final transcript dedup skipped (%s)", job_id, _dd_err)

    # ── Transcript readability score ──
    # Returns a Netflix-style A-F grade + per-axis sub-scores (CPS,
    # line length, duration, gap). Persisted on the job so the Analysis
    # page can display a readability card next to the reframe report.
    transcript_readability = None
    if transcript:
        try:
            from backend.services.subtitle_formatter import compute_readability_report
            from backend.models import TranscriptSegment
            _r_models = [
                t if isinstance(t, TranscriptSegment) else TranscriptSegment(**t)
                for t in transcript
            ]
            transcript_readability = compute_readability_report(_r_models)
            logger.info(
                "[%s] Transcript readability: grade %s (%.1f/100) — CPS %.1f compliance, "
                "avg %.1f cps / peak %.1f cps over %d segments%s",
                job_id,
                transcript_readability.get("grade"),
                transcript_readability.get("score", 0),
                transcript_readability.get("cps_compliance_pct", 0),
                transcript_readability.get("avg_cps", 0),
                transcript_readability.get("max_cps_observed", 0),
                transcript_readability.get("total_segments", 0),
                " (CJK profile)" if transcript_readability.get("is_cjk") else "",
            )
        except Exception as _rd_err:
            logger.warning(
                "[%s] Readability scoring failed: %s",
                job_id, _rd_err,
            )

    # JobResult has no render_plan field, so persist the plan as a sidecar
    # JSON the /api/jobs/{id}/render_plan endpoint can serve to the NLE editor.
    try:
        with open(os.path.join(job_dir, "render_plan.json"), "w") as _rpf:
            json.dump(render_plan.to_dict(), _rpf)
        logger.info("[%s] render_plan.json written (%d ops)", job_id, len(render_plan.ops))
    except Exception as _rpe:
        logger.warning("[%s] render_plan.json write failed: %s", job_id, _rpe)

    # Confirm the reframer-internal RenderPlan (with per-scene signals)
    # also persisted, plus the per-decision JSONL trace. Surfaces in
    # /api/logs/export so the Claude reviewer has the full picture
    # without needing shell access to the job directory.
    try:
        _scene_signals_count = sum(
            1 for sc in (reframer_plan.scenes or []) if sc.get("signals"))
        _trace_size = (os.path.getsize(_trace_path)
                       if os.path.exists(_trace_path) else 0)
        logger.info(
            "[%s] reframer artifacts: scenes_with_signals=%d "
            "reframe_trace.jsonl=%d bytes",
            job_id, _scene_signals_count, _trace_size)
    except Exception:
        pass

    # ── Detection overlay sidecar for the reframer preview ──
    # Captures the face / subject / motion / speech timelines so the
    # VideoEditor canvas overlay can draw spatial boxes that match the
    # render plan. Served by /api/jobs/{id}/detection_overlay.
    try:
        overlay_data = serialize_detection_overlay(perception, reframer_plan)
        with open(os.path.join(job_dir, "detection_overlay.json"), "w") as _odf:
            json.dump(overlay_data, _odf)
        logger.info(
            "[%s] detection_overlay.json written (%d face samples, %d person samples)",
            job_id,
            len(overlay_data.get("face_timeline", {})),
            len(overlay_data.get("person_timeline", {})),
        )
    except Exception as _ovre:
        logger.warning("[%s] detection_overlay.json write failed: %s", job_id, _ovre)

    # ── Reframe quality grade (A-F, 0-100 score, per-axis sub-scores) ──
    reframe_report = None
    try:
        from backend.services.reframe_evaluator import ReframeEvaluator
        _grade = await asyncio.to_thread(
            ReframeEvaluator(reframer_plan, perception).run
        )
        reframe_report = _grade.to_dict()
        reframe_report.pop("second_scores", None)
        logger.info(
            "[%s] reframe grade: %s (%.0f/100)", job_id,
            reframe_report.get("grade"), reframe_report.get("overall_score", 0),
        )
        if reframe_report:
            from backend.services.reframe_evaluator import ReframeEvaluator
            _summary = ReframeEvaluator.summarise_problems(_grade)
            logger.info(
                "[%s] reframe problems: %s",
                job_id, _summary,
            )
            _high = [p for p in (_grade.problems or []) if p.get('severity') == 'HIGH']
            for _p in _high[:5]:
                logger.warning(
                    "[%s] HIGH %s @ t=%ds: crop_x=%s face_cx=%s "
                    "(kf_bracket: %s→%s)",
                    job_id, _p.get('type'), _p.get('time_sec', 0),
                    _p.get('crop_x', '?'), _p.get('best_face_cx', '?'),
                    _p.get('keyframe_t_ms_before', '?'),
                    _p.get('keyframe_t_ms_after', '?'),
                )
    except Exception as _ge:
        logger.warning("[%s] reframe grade failed: %s", job_id, _ge)

    # Seed the speaker-name map so the transcript UI has stable colour keys.
    _speakers = sorted({t.get("speaker", "Speaker 1") for t in transcript})
    speaker_names = {s: s for s in _speakers}

    # ── Compute summary: what ran on GPU vs CPU ──────────────────
    # Three-stage snapshot so the Analysis page can show "did the GPU
    # actually get used?" without making the user export and grep
    # logs. Each stage decides independently and can silently fall to
    # CPU (different code paths, different reasons), so each gets
    # its own row in the Compute card.
    compute_summary = _build_compute_summary(engine, perception)
    if _resumed_from_checkpoint:
        # On a resumed run the live ``engine`` is a stub and frame extraction
        # was served from cache (no fresh LAST_EXTRACTION_LABEL), so some rows
        # can't be rebuilt this run. Keep the prior run's rows underneath so the
        # Compute card stays complete; anything we did rebuild wins on top.
        compute_summary = {**_prior_compute_summary, **compute_summary}
    logger.info(
        "[%s] compute summary: %s",
        job_id,
        ", ".join(f"{k}={v.get('device')}" for k, v in compute_summary.items()),
    )

    # Positive confirmation of GPU usage in the live log (the cpu_fallback
    # warning below only fires on a FALLBACK, so a healthy GPU run was silent).
    try:
        _gpu_stages = [k for k, v in compute_summary.items()
                       if str(v.get("device", "")).startswith("cuda")]
        if _gpu_stages:
            await broadcast_ws(job_id, {
                "type": "compute_info",
                "message": f"Ran on GPU (cuda): {', '.join(_gpu_stages)}.",
            })
    except Exception:
        pass

    # ── CPU-fallback warning ──
    # When GPU acceleration is enabled the user EXPECTS the GPU. If any heavy
    # stage actually ran on CPU (CUDA-only torch missing, CTranslate2 "CUDA
    # failed", GPU not passed through, etc.) the run is ~10-30x slower and used
    # to look "stuck". Surface it as a visible job warning + WS notice so the
    # UI can tell the user instead of silently crawling.
    try:
        from backend.services.pipeline_helpers import cpu_fallback_stages
        _cpu_stages = cpu_fallback_stages(compute_summary)
        if _cpu_stages and bool(getattr(settings, "GPU_ACCELERATION_ENABLED", False)):
            _names = ", ".join(_cpu_stages)
            _warn = (
                f"Running on CPU: {_names}. The GPU was enabled but couldn't be "
                "used — analysis is much slower (~10–30×). Check that the "
                "container sees the GPU (nvidia-container-toolkit, host driver "
                "≥ R525, and the CUDA torch build)."
            )
            _record_pipeline_warning(job_id, _warn)
            logger.warning("[%s] %s", job_id, _warn)
            try:
                await broadcast_ws(job_id, {
                    "type": "compute_warning",
                    "level": "warning",
                    "message": _warn,
                    "cpu_stages": _cpu_stages,
                })
            except Exception:
                pass
    except Exception as _cpu_warn_err:
        logger.debug("[%s] CPU-fallback warning skipped: %s", job_id, _cpu_warn_err)

    await database.update_job_status(
        job_id,
        scenes=scenes,
        transcript=transcript,
        subject_track=subject_track,
        reframe_report=reframe_report,
        compute_summary=compute_summary,
        transcript_readability=transcript_readability,
        speaker_names=speaker_names,
        language=getattr(perception, "detected_language", "") or "",
        default_layout_mode="single",
        scene_cut_timestamps=[round(c / 1000.0, 3) for c in (perception.scene_cuts or [])],
    )
    _n_segs = len(transcript)
    _n_scenes_val = len(scenes)
    # _detected_lang was hoisted above (right after the bridge conversion).
    _lang_note = f" [{_detected_lang}]" if _detected_lang else ""
    # Keep the in-memory compat_stubs dict in sync so the background
    # post-processing task can use it as a reliable fallback even when
    # job.language is empty (e.g. reprocessed or test jobs).
    if _detected_lang:
        from backend.services.compat_stubs import _last_detected_language
        _last_detected_language["lang"] = _detected_lang
    await _update_progress(
        job_id, JobStatus.ANALYZING_SCENES, 62,
        f"Analysis complete — {_n_scenes_val} scenes, {_n_segs} transcript segments{_lang_note}",
    )

    # ── Subtitle translation + target-language polish (right after transcription
    # + speaker assignment; BEFORE summary + clips) ──
    # Translation runs as early as possible — on the freshly transcribed,
    # speaker-labelled, deduped, music-marked transcript — so the translated
    # subtitles are produced immediately after transcription/diarization and are
    # never blocked or delayed by the summary or clip stages. Offline NMT does
    # the translation; the OpenRouter editorial model only polishes the result
    # for readability (meaning + timing preserved). The clip-dependent finishers
    # (caption refresh + Auto-SEO) still run AFTER clips in
    # ``_run_post_clip_followups``.
    cancel_check()
    if _will_translate:
        await _update_progress(
            job_id, JobStatus.TRANSLATING, 63, "Translating + polishing subtitles...",
        )
    logger.info(
        "[%s] invoking translate+polish (target=%s, source=%s, will_translate=%s)",
        job_id, _bg_target or "(none)", _bg_source or "auto", _will_translate,
    )
    _pp_result = None
    try:
        _pp_job = await database.load_job(job_id) or job
        _pp_result = await _background_post_processing(
            job_id, list(transcript), orchestrator, _pp_job,
            polished_already=_polished_in_critical_path,
        )
    except Exception as _pp_err:
        logger.error(
            "[%s] translate+polish step raised (non-fatal — continuing): %s",
            job_id, _pp_err, exc_info=True,
        )
    # Adopt the SOURCE transcript exactly as post-processing finalized it —
    # polished on the no-translation / translation-failed paths — so the COMPLETE
    # save persists a clean transcript, never raw (Task 2). The translated track
    # is persisted separately as translated_transcript by post-processing.
    if _pp_result and _pp_result.get("source_transcript") is not None:
        transcript = _pp_result["source_transcript"]
    # Fail loud when a PLANNED translation did not produce target output (Task 3).
    if _will_translate:
        if _pp_result is None:
            # Crashed before it could translate OR run its source-language
            # fallback — last-resort polish so we never ship raw, plus a status.
            logger.error(
                "[%s] translate+polish crashed before completing — polishing "
                "source transcript as a last resort", job_id)
            try:
                _src_models, _ = await _polish_transcript_loop(
                    job_id, list(transcript), orchestrator, _bg_source,
                    model_override=_resolve_polish_model_override(orchestrator))
                if _src_models:
                    transcript = [
                        m.model_dump() if hasattr(m, "model_dump") else dict(m)
                        for m in _src_models
                    ]
            except Exception as _lp_err:
                logger.warning("[%s] last-resort source polish failed: %s", job_id, _lp_err)
            await _set_translation_status(
                job_id, "translation_failed",
                "translate+polish step crashed; kept source-language transcript")
        elif not _pp_result.get("translated") and not _pp_result.get("failed_reason"):
            await _set_translation_status(
                job_id, "translation_failed",
                "planned translation produced no target-language output")

    # ── VLM summary (kept ai_orchestrator) — runs AFTER translation now ──
    # Defensive, idempotent: free the Whisper engine before the VLM stage in
    # case we deferred its release above to let a Whisper-native translate reuse
    # it. No-ops when it was already released (the normal, non-reuse path).
    await _release_whisper_vram(job_id)
    cancel_check()
    await _update_progress(
        job_id, JobStatus.GENERATING_SUMMARY, 70, "Generating video summary...",
    )
    summary = None
    # Summarize the TRANSLATED (target-language) transcript when we produced one,
    # so the summary comes out in the output language even on a weak local model.
    # `transcript` was reset to the SOURCE track above (~line 3997), so feeding it
    # to the summary made the offline summary come back in the source language
    # (Japanese) — the cloud models happened to translate-on-the-fly and hid it.
    _summary_transcript = transcript
    if _pp_result and _pp_result.get("translated") and _pp_result.get("target_transcript"):
        _summary_transcript = _pp_result["target_transcript"]
    async with _stage_timer(job_id, "summary"):
        try:
            _sr = await orchestrator.generate_summary(
                _summary_transcript, scenes, job_id, tier=tier,
                output_language=(_pp_result or {}).get("output_lang", ""))
            summary = _sr[0] if isinstance(_sr, tuple) else _sr
        except Exception as _se:
            logger.warning(
                "[%s] VLM summary failed (%s) — falling back to transcript summary",
                job_id, _se,
            )
    summary_dict = summary.model_dump() if hasattr(summary, "model_dump") else summary
    if summary is None or not has_real_summary_content(summary_dict):
        try:
            summary = VideoSummary(**build_summary_from_transcript(_summary_transcript, scenes))
        except Exception:
            summary = VideoSummary(
                overview="Summary unavailable for this video.",
                key_topics=[], tone="neutral",
                estimated_audience="general", content_category="generic",
            )
    # Persist the summary so the post-clip Auto-SEO (which reads job.summary for
    # prompt context) sees it.
    try:
        await database.update_job_status(job_id, summary=summary)
    except Exception as _sum_err:
        logger.debug("[%s] summary persist skipped: %s", job_id, _sum_err)


    # ── Clip detection (reframer clipper) ──
    cancel_check()
    # Offline Mode: hand the GPU from the editorial Ollama LLM (just used for
    # polish/translate/summary) to the local clip-detection vision model. No-op
    # in the cloud. Keeps the 4 GB GTX 1650 from trying to hold both at once.
    await _free_editorial_vram_before_local_clips(job_id, orchestrator)
    await _update_progress(
        job_id, JobStatus.DETECTING_CLIPS, 80, "Detecting viral clip candidates...",
    )
    clips = []
    clip_discovery_mode = None  # "vlm" | "signal_only" | None (set after run)
    async with _stage_timer(job_id, "clip_extraction"):
        try:
            from backend.services.reframer_clipper import ClipExtractor, ClipperConfig
            clipper_config = ClipperConfig.load(clipper_config_path())
            # Replicate cloud GPU is configured via app settings, not the
            # clipper_config.json file — overlay it so the clipper sees it.
            clipper_config.replicate_api_key = settings.REPLICATE_API_KEY
            clipper_config.replicate_model = settings.REPLICATE_MODEL
            clipper_config.replicate_enabled = (
                settings.REPLICATE_ENABLED
                and settings.resolve_ai_source("clip") == "cloud")
            # Clip-scoring editorial judge → local model.
            #  • Offline Mode: local best as primary, second-best as fallback —
            #    instead of the configured judge specs, whose fallback is a cloud
            #    model (a cloud call that would break "no cloud calls").
            #  • Cloud + EDITORIAL_LOCAL_FALLBACK: keep the cloud judge primary but
            #    set the FALLBACK to the local model, so when the cloud key is
            #    exhausted the judge still scores clips locally (otherwise scoring
            #    silently degrades to signal-only and ranks clips 1–100 by audio
            #    /visual heuristics, the symptom on the key-limited run).
            if _local_editorial_models:
                if _editorial_is_local:
                    clipper_config.judge_primary = f"ollama:{_local_editorial_models[0]}"
                    clipper_config.judge_fallback = (
                        f"ollama:{_local_editorial_models[1]}"
                        if len(_local_editorial_models) > 1 else "")
                elif not (clipper_config.judge_fallback or "").startswith("ollama:"):
                    clipper_config.judge_fallback = f"ollama:{_local_editorial_models[0]}"
                logger.info(
                    "[%s] Clip judge → %s%s", job_id,
                    clipper_config.judge_primary or "(configured)",
                    f" → {clipper_config.judge_fallback}" if clipper_config.judge_fallback else " (no fallback)",
                )
            # Clip-generation defaults from Settings > Clip Generation.
            if settings.CLIP_MIN_DURATION:
                clipper_config.min_duration_s = settings.CLIP_MIN_DURATION
            if settings.CLIP_MAX_DURATION:
                clipper_config.max_duration_s = settings.CLIP_MAX_DURATION
            clipper_config.ideal_duration_s = max(
                clipper_config.min_duration_s,
                min(clipper_config.max_duration_s,
                    (clipper_config.min_duration_s + clipper_config.max_duration_s) // 2))
            clipper_config.max_clips = settings.CLIP_COUNT
            if settings.CLIP_PREFERRED_SUBJECTS:
                clipper_config.preferred_subjects = settings.CLIP_PREFERRED_SUBJECTS
            if settings.CLIP_AVOID_SUBJECTS:
                clipper_config.avoid_subjects = settings.CLIP_AVOID_SUBJECTS
            clipper_config.discovery_prompt = settings.CLIP_DISCOVERY_PROMPT
            clip_extractor = ClipExtractor(
                video_path=video_path,
                perception=perception,
                plan=reframer_plan,
                config=clipper_config,
                transcript_segments=perception.transcript_segments,
            )

            # ── Live progress for clip extraction (80 → 97 %) ──
            # Without this callback the pipeline sits at 80 %
            # "Detecting viral clip candidates..." for 2-5 minutes while
            # the clipper runs its CPU signal scan, the VLM discovery
            # passes, the judge loop, and the export step. The UI
            # reports that as "stuck at 80 %" (or, when the manual
            # generate-clips heartbeat at clips.py:1047 is also running,
            # "stuck at 89 %"). The clipper already reports
            # internal progress as a 0-1 fraction via its run()
            # ``on_progress`` callback — we just have to relay it
            # back to the main pipeline.
            _loop = asyncio.get_running_loop()
            _last_clipper_pct = [80]  # mutable cell for the closure

            def _clipper_progress(frac, message=None):
                # resolve_clip_progress maps frac→80-97 %% forward-only and
                # decides whether to emit. A per-clip ``message`` ("Exporting clip
                # k/N …") always emits so the long export tail keeps the activity
                # log + stuck-timer alive; a plain fraction emits only on advance.
                res = resolve_clip_progress(frac, message, _last_clipper_pct[0])
                if res is None:
                    return
                pct, label = res
                _last_clipper_pct[0] = pct
                # Once we're in the export tail, name the keepalive "clip export"
                # so a gap between two slow encodes doesn't read as "Still
                # processing... (clip detection …)" while clips are clearly
                # being written.
                _hb_label = "clip export" if str(label).lstrip().startswith("Exporting") else ""
                try:
                    asyncio.run_coroutine_threadsafe(
                        _update_progress(
                            job_id, JobStatus.DETECTING_CLIPS, pct, label,
                            heartbeat_label=_hb_label,
                        ),
                        _loop,
                    )
                except Exception:
                    pass  # best-effort — never break clip extraction

            def _persist_candidates(final_candidates):
                # Snapshot the RANKED clips to the DB before the long export loop
                # so a container death mid-export (frequent on this box) keeps the
                # clip LIST — the user still sees the clips and can re-export
                # individually — instead of losing all of them and seeing "No
                # clips detected". The final enriched persist overwrites this on a
                # clean finish. Best-effort, runs on the clipper thread.
                try:
                    _snap = to_fez_clips(final_candidates)
                    _snap_dicts = [c.model_dump() if hasattr(c, "model_dump") else c
                                   for c in _snap]
                    if _snap_dicts:
                        asyncio.run_coroutine_threadsafe(
                            database.update_job_status(job_id, clips=_snap_dicts),
                            _loop,
                        )
                        logger.info(
                            "[%s] Snapshotted %d clip candidate(s) before export "
                            "(crash-safe)", job_id, len(_snap_dicts))
                except Exception as _pc_err:
                    logger.debug("[%s] clip candidate snapshot skipped: %s", job_id, _pc_err)

            # Hard cap so clip detection can never hang the job indefinitely
            # (e.g. the VLM can't load + Replicate is rate-limited). The clipper
            # already trips a 429 circuit breaker and always has the signal-based
            # pass; this is the last-resort backstop — on timeout we continue
            # with no VLM clips rather than leaving the UI stuck.
            raw_clips = await asyncio.wait_for(
                asyncio.to_thread(clip_extractor.run, _clipper_progress, _persist_candidates),
                timeout=_SUMMARY_CLIP_TIMEOUT,
            )
            clips = to_fez_clips(raw_clips)
            # Flag the degraded path so the UI can say so: VLM discovery either
            # ran (vlm) or was unavailable/rate-limited → signal-only ranking.
            if clips:
                clip_discovery_mode = (
                    "vlm" if getattr(clip_extractor, "vlm_discovery_used", False)
                    else "signal_only"
                )
            if clip_discovery_mode == "signal_only":
                logger.info(
                    "[%s] Clip detection used signal-only ranking (VLM unavailable "
                    "or rate-limited)", job_id,
                )
            logger.info("[%s] Clip extraction produced %d clips", job_id, len(clips))
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] Clip detection exceeded %ds — continuing without clips "
                "(check VLM VRAM / Replicate rate limits).",
                job_id, _SUMMARY_CLIP_TIMEOUT,
            )
            clips = []
        except Exception as _ce:
            logger.exception("[%s] Clip extraction failed: %s", job_id, _ce)
            clips = []

    # ── Post-clip translation finishers (caption refresh + Auto-SEO) ──
    # These need the clips, so they run here — AFTER extraction. They can never
    # block or skip translation, which already ran above the clip stage. The
    # in-process ``clips`` is threaded as the fallback for the DB-round-trip-empty
    # case. Best-effort: a failure here never aborts the finalize.
    try:
        _final_clips = await _run_post_clip_followups(
            job_id, orchestrator, _pp_result, clips)
        # Prefer the in-process result (target-language caption/hook/title + SEO)
        # — the DB round-trip reads back 0 clips during post-processing, and
        # falling back to it would ship the stale source-language list. Only use
        # the DB reload when the followups returned nothing.
        if _final_clips:
            clips = _final_clips
        else:
            _pp_after = await database.load_job(job_id)
            if _pp_after is not None and getattr(_pp_after, "clips", None):
                clips = [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in _pp_after.clips
                ]
    except Exception as _fu_err:
        logger.warning(
            "[%s] Post-clip follow-ups failed (non-fatal): %s",
            job_id, _fu_err, exc_info=True,
        )

    # ── Final save ──
    cancel_check()
    _analysis_seconds = round(_time.monotonic() - _pipeline_start, 1)

    # ── Aggregate the analysis spend ──
    # Orchestrator tracks per-provider token usage (OpenRouter,
    # Anthropic, Gemini, Groq) and translates that to USD via its
    # _COST_PER_1K_TOKENS table; the clipper carries the Replicate
    # estimate from the VideoLLaMA3 chunks. Both are best-effort
    # estimates — Replicate's real billing lines up within a few
    # cents of the per-chunk multiplier we use.
    _total_cost_usd = 0.0
    _cost_breakdown = {}
    try:
        # Prefer the process-wide OpenRouter session tally for THIS job — it
        # covers translation + summary + SEO + the clip judge, which the
        # orchestrator's own per-provider estimate misses (that's why the cost
        # used to read as Replicate-only). Fall back to the orchestrator estimate
        # for non-OpenRouter setups.
        from backend.services.providers.openrouter_provider import OpenRouterProvider
        _llm_cost = float(OpenRouterProvider.session_cost() or 0.0)
        if _llm_cost <= 0:
            _llm_cost = float(orchestrator.estimate_cost() or 0.0)
        _total_cost_usd += _llm_cost
        if _llm_cost > 0:
            _cost_breakdown["llm"] = round(_llm_cost, 4)
    except Exception:
        pass
    try:
        _rep_cost = float(getattr(clip_extractor, "total_cost_usd", 0.0) or 0.0)
        _total_cost_usd += _rep_cost
        if _rep_cost > 0:
            _cost_breakdown["replicate"] = round(_rep_cost, 4)
    except Exception:
        pass
    _total_cost_usd = round(_total_cost_usd, 4)

    # NOTE: translate + polish (``_background_post_processing``) now runs ABOVE,
    # BEFORE clip extraction — so a clip-stage failure can never bypass it — and
    # the clip-dependent caption refresh + Auto-SEO already ran in
    # ``_run_post_clip_followups`` after the clip stage. The translated track
    # (translated_transcript) + transcript_readability + translation_status are
    # already persisted and are preserved through the COMPLETE save by
    # ``_persist_complete_job``'s load→merge (they are not in _complete_fields).

    # Enter the finalization window: from here on, no progress relay may
    # write a non-terminal status (see _finalizing_jobs / _update_progress).
    _finalizing_jobs.add(job_id)
    await _update_progress(job_id, JobStatus.DETECTING_CLIPS, 98, "Saving results...")

    # Coerce ``to_fez_clips`` dicts → ClipCandidate models before the
    # COMPLETE save. ``update_job_status`` calls ``setattr(job, 'clips',
    # value)`` without ``validate_assignment``, so a raw list[dict] in
    # the typed list[ClipCandidate] field triggers
    # ``PydanticSerializationUnexpectedValue`` during the next
    # ``model_dump`` and — observed in production on a JA→EN job — can
    # leave the persisted ``status`` field stuck on ``detecting_clips``
    # despite this call passing ``status=COMPLETE``. Symptom: UI shows
    # "Finalizing clip detection..." indefinitely because the disk
    # round-trip skipped the status update. Pre-coercion eliminates the
    # warning path entirely.
    from backend.models import ClipCandidate as _ClipModel
    _clip_models = []
    for _c in (clips or []):
        if isinstance(_c, _ClipModel):
            _clip_models.append(_c)
        elif isinstance(_c, dict):
            try:
                _clip_models.append(_ClipModel(**_c))
            except Exception as _coerce_err:
                logger.warning(
                    "[%s] Failed to coerce clip dict to ClipCandidate (%s) — saving raw",
                    job_id, _coerce_err,
                )
                _clip_models.append(_c)
        else:
            _clip_models.append(_c)

    # Coerce scenes + transcript to their models too (clips already are).
    # Raw list[dict] in a typed list[Model] field triggers
    # PydanticSerializationUnexpectedValue on model_dump — and, with numpy
    # values from the reframer (np.float32 precise_x, np.int64 face_count)
    # bypassing SceneDescription's before-validator, can corrupt the round
    # trip so the status reverts. Validating here runs the sanitizers and
    # guarantees a clean, JSON-safe payload.
    from backend.models import SceneDescription as _SceneModel
    from backend.models import TranscriptSegment as _TSegModel

    def _coerce_list(rows, model):
        out = []
        for r in (rows or []):
            if isinstance(r, model):
                out.append(r)
            elif isinstance(r, dict):
                try:
                    out.append(model(**r))
                except Exception:
                    out.append(r)
            else:
                out.append(r)
        return out

    _scene_models = _coerce_list(scenes, _SceneModel)
    _transcript_models = _coerce_list(transcript, _TSegModel)

    # The full COMPLETE payload, kept in one dict so the verify-retry below
    # can re-issue *everything* (not just status) if the first save didn't
    # round-trip. Re-saving status alone — the previous behaviour — left the
    # job COMPLETE but with empty summary/clips/transcript, which the UI
    # renders as a permanent "Generating summary..." spinner with no clips.
    _complete_fields = dict(
        status=JobStatus.COMPLETE,
        progress=100,
        progress_message=f"Analysis complete — {len(_clip_models)} clips, {len(scenes)} scenes",
        summary=summary,
        scenes=_scene_models,
        transcript=_transcript_models,
        clips=_clip_models,
        clip_discovery_mode=clip_discovery_mode,
        analysis_duration_seconds=_analysis_seconds,
        estimated_cost_usd=_total_cost_usd,
        cost_breakdown=_cost_breakdown,
        default_layout_mode="single",
        # Clear the auto-resume attempt counter now the run finished, so a later
        # unrelated re-analyze that gets interrupted starts its budget fresh.
        resume_attempts=0,
    )
    # Robust, self-diagnosing finalization. The plain update_job_status was
    # observed to silently NOT persist the COMPLETE payload (status stayed
    # detecting_clips, clips=0) even with no exception and no concurrent status
    # writer — and even an immediate full-payload retry failed. To remove every
    # failure mode at once we write the job.json DIRECTLY to the canonical path
    # (bypassing the load→merge→save abstraction, the job.job_id-derived save
    # path, and the aiofiles large-write path), force the correct job_id,
    # surface any serialization error, verify against the RAW on-disk bytes,
    # and retry.
    _persisted = await _persist_complete_job(job_id, _complete_fields)
    if not _persisted:
        logger.error(
            "[%s] COMPLETE finalization could NOT persist after retries — "
            "job will appear stuck; see preceding finalize logs for cause", job_id)

    await broadcast_ws(job_id, {
        "type": "complete",
        "status": JobStatus.COMPLETE.value,
        "message": "Analysis complete",
        "progress": 100,
    })
    logger.info(
        "[%s] Pipeline complete in %.1fs — %d clips, %d scenes, %d transcript segments",
        job_id, _analysis_seconds, len(_clip_models), len(scenes), len(transcript),
    )
    # NOTE: translate + polish (``_background_post_processing``) ran on the
    # critical path BEFORE clip extraction (decoupled so a clip-stage failure
    # cannot bypass it), and the clip-dependent caption refresh + Auto-SEO ran
    # right after the clip stage. The translated/polished transcript, refreshed
    # clips and translation_status are already persisted and reflected in (or
    # preserved through) this COMPLETE save. It is no longer a fire-and-forget
    # task (that task was being lost on this deployment, leaving non-English jobs
    # stuck at detecting_clips with an un-translated transcript).
