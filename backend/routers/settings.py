import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
import uuid

import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, Form
from pydantic import BaseModel

from typing import Optional

from backend.config import Settings, settings, get_settings
from backend.services.prompts import (
    PromptSet, load_prompts, save_prompts, get_defaults, MAX_PROMPT_LENGTH,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["settings"])


def _resolve_data_dir() -> str:
    """Find a writable data directory for persisting settings and caches.
    Prefers /data/logs (Docker volume mount), falls back to a local .clipai dir."""
    docker_path = "/data/logs"
    if os.path.isdir(docker_path) and os.access(docker_path, os.W_OK):
        return docker_path
    # Fallback: project-local directory (works outside Docker)
    local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".clipai")
    os.makedirs(local_path, exist_ok=True)
    return local_path


_DATA_DIR = _resolve_data_dir()
MODEL_CACHE_PATH = os.path.join(_DATA_DIR, "model_cache.json")
MODEL_CACHE_TTL = 86400  # 24 hours

# Persistent user settings — saved so they survive container/process restarts.
USER_SETTINGS_PATH = os.path.join(_DATA_DIR, "user_settings.json")

_PLACEHOLDER_KEYS = {"sk-or-...", "sk-ant-...", "AIza...", "gsk_...", "r8_...", ""}

# Keys that are persisted to user_settings.json
_PERSISTABLE_KEYS = [
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
    "HF_AUTH_TOKEN", "REPLICATE_API_KEY",
    # Translation cloud NMT secrets.
    "GOOGLE_TRANSLATE_API_KEY", "DEEPL_API_KEY",
    "OPENROUTER_PRESET", "OPENROUTER_PRIMARY_MODEL", "OPENROUTER_EDITORIAL_MODEL",
    "OPENROUTER_SUMMARY_MODEL", "OPENROUTER_TRANSLATION_MODEL",
    "OLLAMA_PRIMARY_MODEL", "OLLAMA_EDITORIAL_MODEL", "OLLAMA_TRANSLATION_MODEL",
    # Multi-host Ollama registry (JSON array; order = priority). The legacy
    # OLLAMA_HOST field is deliberately NOT persisted — it stays env-owned
    # for env-only deployments and is re-synced from the registry primary at
    # startup when OLLAMA_HOSTS is set (see main.py startup).
    "OLLAMA_HOSTS",
    "WHISPER_MODEL", "WHISPER_MODEL_USER_SET", "WHISPER_BEAM_SIZE",
    "WHISPER_VAD_FILTER", "FRAME_SAMPLE_RATE", "WHISPER_AUTO_UPGRADE",
    # Remote Whisper (OpenAI-compatible server, e.g. the GPU Companion).
    "WHISPER_REMOTE_URL", "WHISPER_REMOTE_API_KEY", "WHISPER_REMOTE_MODEL",
    "GPU_STRICT_REMOTE",
    # Reframer perception sampling (face-detection speed).
    "REFRAMER_MAX_SAMPLES", "REFRAMER_SAMPLE_FPS", "REFRAMER_MIN_SAMPLE_FPS",
    "WHISPER_NO_SPEECH_THRESHOLD",
    "WHISPER_GAP_FILL_ENABLED", "WHISPER_GAP_FILL_MIN_SEC",
    "WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD",
    "CLIP_MIN_DURATION", "CLIP_MAX_DURATION", "CLIP_COUNT",
    "CLIP_PREFERRED_SUBJECTS", "CLIP_AVOID_SUBJECTS", "CLIP_DISCOVERY_PROMPT",
    # VideoLLaMA3 Enhanced Discovery toggles — non-secret, persisted so
    # operator-tuned defaults survive container restarts.
    "VIDEOLLAMA3_ENHANCED", "VIDEOLLAMA3_FPS", "VIDEOLLAMA3_MAX_FRAMES",
    "VIDEOLLAMA3_REFINEMENT_PASS", "VIDEOLLAMA3_KEYFRAME_ANALYSIS",
    "VIDEOLLAMA3_AUDIO_ANNOTATION", "VIDEOLLAMA3_ADAPTIVE_CHUNKS",
    "VIDEOLLAMA3_CHUNK_MIN_S", "VIDEOLLAMA3_CHUNK_MAX_S",
    # Custom vocabulary (Whisper biasing) toggle. The term list itself
    # lives in /data/logs/custom_vocabulary.json (mount-backed); only the
    # enable flag rides user_settings.json.
    "CUSTOM_VOCABULARY_ENABLED",
    # Transcript polishing toggles.
    "TRANSCRIPT_POLISHING_ENABLED", "TRANSCRIPT_POLISHING_BATCH_SIZE",
    "TRANSCRIPT_FILLER_REMOVAL", "TRANSCRIPT_SENTENCE_REPAIR",
    # Dedicated subtitle-polish model (audit Phase 4.2) + cloud STT provider.
    "SUBTITLE_POLISH_MODEL", "SUBTITLE_POLISH_CLOUD_FALLBACK",
    "SUBTITLE_POLISH_CLOUD_MODEL", "SUBTITLE_POLISH_AUTO_GLOSSARY",
    "TRANSCRIPTION_PROVIDER", "OPENAI_API_KEY",
    "GROQ_TRANSCRIBE_MODEL", "OPENAI_TRANSCRIBE_MODEL",
    # Sentence-aware resegmentation toggle.
    "SENTENCE_SEGMENTATION_ENABLED",
    # Voiceprint registry (cross-job speaker naming). The registry itself
    # lives in /data/logs/voiceprints.json (mount-backed); only these knobs
    # ride user_settings.json.
    "VOICEPRINT_ENABLED", "VOICEPRINT_MATCH_THRESHOLD",
    # Subtitle readability + safe-zone toggles.
    "SUBTITLE_CPS_ENFORCEMENT", "SUBTITLE_MAX_CPS", "SUBTITLE_MAX_CHARS_PER_LINE",
    "SUBTITLE_MIN_DURATION_MS", "SUBTITLE_MAX_DURATION_MS",
    "SUBTITLE_SMART_LINE_BREAKS", "SUBTITLE_PLATFORM_SAFE_ZONES",
    "SUBTITLE_PLATFORM_PROFILE", "SUBTITLE_MIN_SPLIT_CHARS",
    "SUBTITLE_MARK_MUSIC", "SUBTITLE_MUSIC_MIN_SEC",
    # Translation engine + glossary toggles.
    "TRANSLATION_ENGINE", "TRANSLATION_CONTEXT_WINDOW",
    "TRANSLATION_GLOSSARY_ENABLED", "NMT_DEVICE",
    # Operator series/show hint — anchors canonical-name + roster correction so
    # mis-heard character/mecha names come out as the official spellings.
    "TRANSLATION_SERIES_HINT",
    # Reference transcript (YouTube captions) + conform mode.
    "TRANSLATION_REFERENCE_SUBTITLES", "TRANSLATION_REFERENCE_MODE",
    # Audio event detection toggles.
    "AUDIO_EVENT_DETECTION", "AUDIO_EVENTS_IN_SUBTITLES",
    "AUDIO_MUSIC_DETECTION",
    "SELF_HOSTED_MODE", "CLIP_ENGINE_SOURCE", "EDITORIAL_AI_SOURCE",
    # Editorial Judge primary + fallback specs — persist via user_settings.json
    # (restored at import) so the Editorial AI Fallback dropdown survives
    # container rebuilds just like the model picks above.
    "EDITORIAL_AI_PRIMARY_SPEC", "EDITORIAL_AI_FALLBACK_SPEC",
    "FFMPEG_PRESET", "FFMPEG_CRF", "FFMPEG_THREADS", "FFMPEG_FASTSTART",
    "GPU_ACCELERATION_ENABLED", "GPU_VENDOR_OVERRIDE",
    "GPU_HWDECODE_ENABLED", "GPU_HEVC_FOR_4K", "GPU_DEVICE_INDEX",
    "GPU_NVENC_PRESET",
    "GPU_FREE_BEFORE_ANALYSIS", "GPU_FREE_BEFORE_WHISPER",
    "AI_FALLBACK_CHAIN",
    # Cloud storage OAuth credentials — entered via the Settings > Cloud
    # Storage UI and persisted so containers without env vars can still
    # connect to Google Drive / Box after the user pastes their credentials.
    "CLIPAI_CLOUD_STORAGE_ENABLED",
    "GOOGLE_DRIVE_CLIENT_ID", "GOOGLE_DRIVE_CLIENT_SECRET", "GOOGLE_DRIVE_REDIRECT_URI",
    "BOX_CLIENT_ID", "BOX_CLIENT_SECRET", "BOX_REDIRECT_URI",
]

# API key fields specifically (used to filter out placeholder values)
_API_KEY_FIELDS = {
    "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY",
    "HF_AUTH_TOKEN", "REPLICATE_API_KEY", "OPENAI_API_KEY",
    "WHISPER_REMOTE_API_KEY",
    # Translation cloud NMT secrets.
    "GOOGLE_TRANSLATE_API_KEY", "DEEPL_API_KEY",
    # Cloud client secrets — same "never overwrite with blank" rule.
    "GOOGLE_DRIVE_CLIENT_SECRET", "BOX_CLIENT_SECRET",
}


def _is_real_value(key: str, val: str) -> bool:
    """Check if a value is a real user-entered value (not empty or a placeholder)."""
    if not val:
        return False
    if val in _PLACEHOLDER_KEYS:
        return False
    return True


def _persist_user_settings() -> bool:
    """Save all user-mutable settings to a JSON file that survives restarts.

    Returns True if settings were persisted successfully, False otherwise.
    The file is stored on a Docker volume mount (/data/logs) so it
    survives container stop/restart/recreate cycles.

    IMPORTANT: Merges with existing file to prevent API key loss.
    If the current in-memory value for an API key is empty but the
    existing file has a real key, the persisted key is preserved.
    This prevents non-key setting changes from wiping out API keys
    that were saved earlier but not restored into memory.
    """
    # Load existing persisted data to preserve API keys we might not have in memory
    existing_data = {}
    try:
        if os.path.exists(USER_SETTINGS_PATH):
            with open(USER_SETTINGS_PATH, "r") as f:
                existing_data = json.load(f)
    except Exception:
        pass

    data = {}
    skipped_keys = []

    # Protect user-set WHISPER_MODEL from being overwritten by runtime auto-upgrades/downgrades.
    # If WHISPER_MODEL_USER_SET is True in the existing file, keep the file's WHISPER_MODEL
    # unless the user explicitly changed it via the save_models endpoint (which sets
    # WHISPER_MODEL_USER_SET = True in the current settings too).
    _whisper_user_set_in_file = existing_data.get("WHISPER_MODEL_USER_SET", False)
    _whisper_model_in_file = existing_data.get("WHISPER_MODEL")

    for key in _PERSISTABLE_KEYS:
        val = getattr(settings, key, "")
        # Non-string types (bool, int) are always persisted
        if isinstance(val, (bool, int)):
            data[key] = val
            continue
        # For API keys: if current in-memory value is empty/placeholder but
        # the existing file has a real key, preserve the persisted key.
        # This prevents non-key setting changes from wiping out saved keys.
        if key in _API_KEY_FIELDS:
            if _is_real_value(key, val):
                data[key] = val
            elif key in existing_data and _is_real_value(key, existing_data[key]):
                data[key] = existing_data[key]
                logger.debug("Preserving %s from existing file (in-memory is empty)", key)
            else:
                skipped_keys.append(key)
            continue

        # Protect WHISPER_MODEL: if the user explicitly set a model in the file
        # but the in-memory value differs (due to auto-upgrade/downgrade), keep
        # the user's saved choice. The auto-upgrade only affects the current session.
        if key == "WHISPER_MODEL" and _whisper_user_set_in_file and _whisper_model_in_file:
            if _is_real_value(key, val) and val != _whisper_model_in_file:
                # In-memory value differs from user's saved choice — check if the
                # current settings object also has WHISPER_MODEL_USER_SET=True
                # (meaning the user just changed it via the UI in this session)
                if not getattr(settings, "WHISPER_MODEL_USER_SET", False):
                    # Auto-change, not user change — preserve the file value
                    data[key] = _whisper_model_in_file
                    logger.info(
                        "Preserving user's saved WHISPER_MODEL='%s' (runtime has '%s' from auto-upgrade/downgrade)",
                        _whisper_model_in_file, val,
                    )
                    continue

        # Skip empty values for non-key settings — BUT preserve a real value
        # already on disk instead of dropping it. ``data`` starts empty and is
        # written wholesale, so without this an unrelated settings save (which
        # may run while a string setting like EDITORIAL_AI_FALLBACK_SPEC is
        # still empty in memory — e.g. before the judge-spec restore populated
        # it, or any import-ordering hiccup) would silently WIPE the persisted
        # value. This is the same "preserve from existing file" guarantee API
        # keys already get, extended to every persisted string setting.
        if not _is_real_value(key, val):
            if key in existing_data and _is_real_value(key, existing_data[key]):
                data[key] = existing_data[key]
            continue
        data[key] = val

    # Log what API keys are being saved (or not)
    for key in _API_KEY_FIELDS:
        if key in data:
            logger.info("Persisting %s: YES (value present, %d chars)", key, len(str(data[key])))
        elif key in skipped_keys:
            logger.debug("Persisting %s: NO (empty in memory and file)", key)

    # Preserve extra backup keys (e.g. _judge_primary / _judge_fallback) so
    # that an unrelated settings save doesn't silently wipe the backups that
    # put_judge_config wrote.  Only keys that start with "_" and are not in
    # _PERSISTABLE_KEYS are preserved — they are owned by other writers.
    for k, v in existing_data.items():
        if k not in _PERSISTABLE_KEYS and k.startswith("_") and k not in data:
            data[k] = v

    try:
        os.makedirs(os.path.dirname(USER_SETTINGS_PATH), exist_ok=True)
        with open(USER_SETTINGS_PATH, "w") as f:
            json.dump(data, f, indent=2)
        # Verify critical values were written
        has_key = "OPENROUTER_API_KEY" in data
        whisper = data.get("WHISPER_MODEL", "?")
        chain = data.get("AI_FALLBACK_CHAIN", "?")
        logger.info(
            "Persisted %d settings to %s (WHISPER_MODEL=%s, API_KEY=%s, CHAIN=%s)",
            len(data), USER_SETTINGS_PATH, whisper,
            f"YES({len(data['OPENROUTER_API_KEY'])}ch)" if has_key else "NO",
            chain,
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to persist user settings to {USER_SETTINGS_PATH}: {e}")
        return False


def _restore_user_settings():
    """Load persisted settings and apply them to the settings object.
    Called once at module import time so saved API keys survive restarts.

    API keys are ALWAYS restored from the persisted file — they take priority
    over environment variables and defaults. This ensures user-entered keys
    survive container recreate cycles where the .env file is lost."""
    if not os.path.exists(USER_SETTINGS_PATH):
        logger.info(f"No persisted settings found at {USER_SETTINGS_PATH}")
        return
    try:
        with open(USER_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        restored = 0

        # Log what's in the file for diagnostics
        api_keys_in_file = [k for k in _API_KEY_FIELDS if k in data and _is_real_value(k, data[k])]
        model_keys_in_file = [
            k for k in ["WHISPER_MODEL", "OPENROUTER_PRIMARY_MODEL", "OPENROUTER_EDITORIAL_MODEL",
                         "AI_FALLBACK_CHAIN", "WHISPER_MODEL_USER_SET"]
            if k in data
        ]
        logger.info(
            "Restoring from %s: %d keys total, API keys: %s, model settings: %s",
            USER_SETTINGS_PATH, len(data),
            api_keys_in_file or "none",
            {k: data[k] for k in model_keys_in_file},
        )

        # Stale-default migration: values persisted when they WERE the shipped
        # default are a snapshot, not a user choice — restoring them pins the
        # old default forever and silently defeats a tuned new default. The
        # observed case: REFRAMER_MAX_SAMPLES=1800 (the old default) persisted
        # on every save, so the 1200-sample speedup never took effect. Any
        # persisted value matching a RETIRED default adopts the new default.
        _RETIRED_DEFAULTS = {
            "REFRAMER_MAX_SAMPLES": (1800, 1500),
            # Subtitle tuning has the same failure mode, and it bites harder
            # because every int/bool is snapshotted unconditionally on save: a
            # settings save made while these were the shipped defaults pins them
            # in /data/logs/user_settings.json (a docker volume, so it survives
            # rebuilds) and silently defeats the new value. Observed: cues shipped
            # at 7.3-7.7s against a 7.0s cap because a persisted 9000 ms was still
            # in force, which also made the over-long-cue trim look broken.
            "SUBTITLE_MAX_DURATION_MS": (9000,),
            # Every prior shipped default for this knob: 20 (original), then
            # 17 (Netflix-strict era). The default is 20 again — 17 turned out
            # to be the MERGE ceiling and vetoed the anti-choppiness pass —
            # and a persisted 17 from the strict era would silently keep it.
            "SUBTITLE_MAX_CPS": (17, 17.0),
            "SUBTITLE_MIN_SPLIT_CHARS": (14,),
            # 42 was the Netflix-spec line budget; the reference track's real
            # wall is 34, and a persisted 42 would keep shipping 40-plus-char
            # lines no matter what the default says.
            "SUBTITLE_MAX_CHARS_PER_LINE": (42,),
            # 8.0 s made gap recovery unable to see the holes it exists for —
            # every measured miss was 2.1-5.7 s. A persisted 8.0 would keep the
            # pass blind on an already-deployed box no matter what ships.
            "VOCAL_GAP_MIN_S": (8.0, 8),
        }
        for _k, _olds in _RETIRED_DEFAULTS.items():
            if _k in data and data.get(_k) in _olds:
                logger.info(
                    "Migrating %s: persisted %s was a prior shipped default — "
                    "adopting the current default %s", _k, data[_k], getattr(settings, _k, None))
                data.pop(_k)

        # One-shot self-heal for the readability/export stack. These booleans
        # are the difference between a professionally formatted subtitle file
        # and a raw dump (unwrapped 90-char lines, uncapped durations, leaked
        # placeholder labels) — a snapshot that pinned any of them OFF re-ships
        # raw output on every download, forever, and the value alone can't
        # distinguish a stale snapshot from a deliberate choice. So this runs
        # ONCE per install (marked in the file, and "_"-prefixed markers are
        # preserved across saves): the pinned False is dropped in favour of the
        # shipped default, and turning the toggle off afterwards sticks.
        _STACK_MARKER = "_READABILITY_STACK_HEALED_V1"
        _STACK_KEYS = ("SUBTITLE_CPS_ENFORCEMENT", "SUBTITLE_SMART_LINE_BREAKS",
                       "SUBTITLE_SPEAKER_LABELS_REQUIRE_NAMES")
        if not data.get(_STACK_MARKER):
            _healed = [k for k in _STACK_KEYS if data.get(k) is False]
            for _k in _healed:
                logger.warning(
                    "Healing %s: a persisted False disabled the subtitle "
                    "readability/export stack (raw unwrapped downloads) — "
                    "restoring the default. Turning it off again in Settings "
                    "will now stick.", _k)
                data.pop(_k)
            data[_STACK_MARKER] = True
            try:
                with open(USER_SETTINGS_PATH, "w") as _fh:
                    json.dump(data, _fh, indent=2)
            except Exception as _heal_err:
                logger.warning("Could not persist readability-stack heal "
                               "marker: %s", _heal_err)

        for key, val in data.items():
            if key not in _PERSISTABLE_KEYS:
                continue
            # Bool/int types: always restore from persisted value
            if isinstance(val, (bool, int)):
                setattr(settings, key, val)
                restored += 1
                continue
            if not _is_real_value(key, val):
                continue
            # For API keys: ALWAYS restore persisted real keys.
            # The persisted value is the most recent user-entered key and
            # should override env defaults, placeholders, and even env vars
            # (the user explicitly saved via the UI after the env was set).
            if key in _API_KEY_FIELDS:
                setattr(settings, key, val)
                logger.info(f"Restored API key: {key} ({len(val)} chars)")
                restored += 1
            else:
                # For model/preset settings: always restore persisted values.
                # The persisted file represents the user's last explicit choice.
                setattr(settings, key, val)
                restored += 1
        logger.info(f"Restored {restored} persisted settings from {USER_SETTINGS_PATH}")

        # Warn about common misconfigurations after restore
        if "openrouter" in (getattr(settings, "AI_FALLBACK_CHAIN", "") or "").lower():
            key = getattr(settings, "OPENROUTER_API_KEY", "")
            if not key or key in _PLACEHOLDER_KEYS:
                logger.warning(
                    "⚠ OpenRouter is in the fallback chain but OPENROUTER_API_KEY is empty! "
                    "Cloud AI models will fail. Enter your API key in Settings > AI Provider."
                )
        whisper = getattr(settings, "WHISPER_MODEL", "small")
        user_set = getattr(settings, "WHISPER_MODEL_USER_SET", False)
        logger.info(
            "Whisper config after restore: model=%s, user_set=%s",
            whisper, user_set,
        )
    except Exception as e:
        logger.warning(f"Failed to restore user settings from {USER_SETTINGS_PATH}: {e}")


# Restore saved settings on module load
_restore_user_settings()


def _backfill_api_keys():
    """Ensure API keys loaded from .env are also persisted to user_settings.json.

    When a user upgrades from an older version that only saved keys to .env,
    the key is loaded by pydantic into memory but may not be in user_settings.json.
    On container recreate, .env is lost. This backfill captures any in-memory keys
    that are missing from the persisted file, preventing key loss on updates.
    """
    if not os.path.exists(USER_SETTINGS_PATH):
        return
    try:
        with open(USER_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        updated = False
        for key in _API_KEY_FIELDS:
            in_memory = getattr(settings, key, "")
            in_file = data.get(key, "")
            if _is_real_value(key, in_memory) and not _is_real_value(key, in_file):
                data[key] = in_memory
                updated = True
                logger.info(
                    "Backfilling %s from memory to user_settings.json (%d chars)",
                    key, len(in_memory),
                )
        if updated:
            with open(USER_SETTINGS_PATH, "w") as f:
                json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning("API key backfill failed (non-fatal): %s", e)


_backfill_api_keys()


# ── One-time migration: add WHISPER_MODEL_USER_SET flag if missing ──
# Older versions didn't persist this flag. If the file has a non-default
# Whisper model but no flag, preserve the model and mark it as user-set
# (the user kept it, whether originally from auto-upgrade or manual choice).
# Previously this reset to "small", which wiped users' model selections.
def _migrate_stale_whisper():
    if not os.path.exists(USER_SETTINGS_PATH):
        return
    try:
        with open(USER_SETTINGS_PATH, "r") as f:
            data = json.load(f)
        whisper = data.get("WHISPER_MODEL")
        user_set = data.get("WHISPER_MODEL_USER_SET")
        if whisper and whisper != "small" and user_set is None:
            # Flag was missing — preserve the model and mark as user-set.
            # The user had this model saved, so it represents their preference
            # regardless of how it was originally selected.
            logger.info(
                "Migration: WHISPER_MODEL='%s' with no WHISPER_MODEL_USER_SET flag — "
                "preserving model and marking as user-set.",
                whisper,
            )
            data["WHISPER_MODEL_USER_SET"] = True
            with open(USER_SETTINGS_PATH, "w") as f:
                json.dump(data, f, indent=2)
            settings.WHISPER_MODEL_USER_SET = True
    except Exception as e:
        logger.warning("Migration check failed (non-fatal): %s", e)

_migrate_stale_whisper()

# Short-lived cache for /api/providers/status (avoid hammering Ollama on rapid re-renders)
_status_cache: dict = {}
_status_cache_ts: float = 0
_STATUS_CACHE_TTL = 30  # seconds

# Separate long-lived cache for the VideoLLaMA2 VRAM availability check.
# The check runs torch.cuda.mem_get_info() which is cheap, but logs at INFO
# on every failure — on a permanently undersized GPU this floods logs every
# few seconds.  Cache the result for 5 minutes so the log line appears at
# most once per restart window instead of hundreds of times per hour.
_vl2_cache_ts: float = 0.0
_vl2_cache_result: bool = False
_VL2_CHECK_TTL = 300.0  # seconds (5 minutes)

# Env var names per provider
_PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "huggingface": "HF_AUTH_TOKEN",
}

# -- Cost estimation for a 10-min video --
# 120 frames (sampled every 5s), each ~765 tokens as image input
# Vision: 120 images * 765 ≈ 92K input tokens, ~12K output tokens
# Text: ~15K input tokens (transcript+scenes), ~8K output tokens (summary+clips)
_VISION_INPUT_TOKENS_10MIN = 100_000
_VISION_OUTPUT_TOKENS_10MIN = 15_000
_TEXT_INPUT_TOKENS_10MIN = 15_000
_TEXT_OUTPUT_TOKENS_10MIN = 8_000

# Speed ratings for known model families (estimated minutes to analyze a 10-min video).
# Vision: frame analysis across ~60 frames. Text: summary + clip detection.
# "speed" = "fast" | "medium" | "slow", "est_minutes" = estimated wall-clock minutes
_MODEL_SPEED_PROFILES = {
    # --- Free auto-router ---
    # quality_score: 1=poor, 2=basic, 3=good, 4=excellent, 5=best
    "openrouter/free": {"speed": "medium", "est_minutes_vision": 5.0, "est_minutes_text": 2.0, "quality": "basic", "quality_score": 2},
    # --- Fast models (under 2 min for 10-min video) ---
    "gemini-2.5-flash": {"speed": "fast", "est_minutes_vision": 1.5, "est_minutes_text": 0.5, "quality": "good", "quality_score": 3},
    "gemini-2.0-flash": {"speed": "fast", "est_minutes_vision": 1.5, "est_minutes_text": 0.5, "quality": "good", "quality_score": 3},
    "gemini-flash": {"speed": "fast", "est_minutes_vision": 1.5, "est_minutes_text": 0.5, "quality": "good", "quality_score": 3},
    "llama-3.1-8b": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 0.5, "quality": "basic", "quality_score": 2},
    "llama-3.3-70b": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "qwen": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "mistral": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 0.7, "quality": "good", "quality_score": 3},
    "deepseek": {"speed": "fast", "est_minutes_vision": 0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    # --- Medium models (2-5 min) ---
    "gemini-2.5-pro": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "gpt-4o-mini": {"speed": "medium", "est_minutes_vision": 2.5, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "gpt-4o": {"speed": "medium", "est_minutes_vision": 3.5, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "claude-haiku": {"speed": "medium", "est_minutes_vision": 2.0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "pixtral": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    # --- Slow models (5+ min) ---
    "claude-sonnet": {"speed": "slow", "est_minutes_vision": 5.0, "est_minutes_text": 2.5, "quality": "excellent", "quality_score": 4},
    "claude-opus": {"speed": "slow", "est_minutes_vision": 8.0, "est_minutes_text": 4.0, "quality": "best", "quality_score": 5},
    "gpt-4-turbo": {"speed": "slow", "est_minutes_vision": 5.0, "est_minutes_text": 2.0, "quality": "excellent", "quality_score": 4},
    "o1": {"speed": "slow", "est_minutes_vision": 6.0, "est_minutes_text": 3.0, "quality": "excellent", "quality_score": 4},
    "o3": {"speed": "slow", "est_minutes_vision": 7.0, "est_minutes_text": 3.5, "quality": "best", "quality_score": 5},
    # --- Vision model families ---
    "qwen2.5-vl": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "qwen3-vl": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "excellent", "quality_score": 4},
    "kimi-vl": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "kimi-k2": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "internvl": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "minicpm-v": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 1.0, "quality": "good", "quality_score": 3},
    "glm-4": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "yi-vision": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "step-3.5": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "mimo": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "deepseek-vl": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    "nemotron": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "phi-4": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "phi-3": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "basic", "quality_score": 2},
    "gemma-3": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "llama-3.2-90b": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "llama-3.2-11b": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.8, "quality": "good", "quality_score": 3},
    "llama-4": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "grok-3": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "excellent", "quality_score": 4},
    "grok-4": {"speed": "medium", "est_minutes_vision": 3.5, "est_minutes_text": 2.0, "quality": "excellent", "quality_score": 4},
    "grok-2": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.5, "quality": "good", "quality_score": 3},
    # --- Models with limited/no vision tracking ---
    "reka-edge": {"speed": "fast", "est_minutes_vision": 2.0, "est_minutes_text": 0.5, "quality": "poor", "quality_score": 1},
    "reka-core": {"speed": "medium", "est_minutes_vision": 3.0, "est_minutes_text": 1.0, "quality": "basic", "quality_score": 2},
    "gemini-2.5-flash-lite": {"speed": "fast", "est_minutes_vision": 1.0, "est_minutes_text": 0.3, "quality": "good", "quality_score": 3},
}

# Free tier models are rate-limited (~20 RPM), multiply time by 3x
_FREE_SPEED_MULTIPLIER = 3.0


def _estimate_speed(model_id: str, role: str, is_free: bool) -> dict:
    """Estimate analysis speed for a model on a 10-minute video.

    Returns {"speed": "fast"|"medium"|"slow", "est_minutes": float, "quality": str}.
    """
    mid_lower = model_id.lower()

    # Try to match against known profiles
    best_match = None
    for pattern, profile in _MODEL_SPEED_PROFILES.items():
        if pattern in mid_lower:
            best_match = profile
            break

    if best_match:
        minutes = best_match.get(f"est_minutes_{role}", best_match.get("est_minutes_text", 2.0))
        if is_free:
            minutes *= _FREE_SPEED_MULTIPLIER
        speed = best_match["speed"]
        if is_free and speed == "fast":
            speed = "medium"
        quality = best_match["quality"]
        quality_score = best_match.get("quality_score", 3)
    else:
        # Unknown model — estimate based on whether it's free
        minutes = 4.0 if is_free else 2.0
        speed = "medium"
        quality = "good"
        quality_score = 3

    # Build display string
    if minutes < 1:
        time_str = f"~{int(minutes * 60)}s"
    elif minutes < 10:
        time_str = f"~{minutes:.1f}min"
    else:
        time_str = f"~{int(minutes)}min"

    return {
        "speed": speed,
        "est_minutes": round(minutes, 1),
        "est_time_display": time_str,
        "quality": quality,
        "quality_score": quality_score,
    }


def _key_is_set(key: str) -> bool:
    return bool(key) and key not in _PLACEHOLDER_KEYS


def _check_videollama2_available() -> bool:
    """Cached wrapper around the VideoLLaMA2 availability probe.

    Result is memoised for _VL2_CHECK_TTL seconds so a permanently undersized
    GPU (e.g. GTX 1650 with 4 GB) does not flood logs every few seconds.
    """
    global _vl2_cache_ts, _vl2_cache_result
    now = time.time()
    if now - _vl2_cache_ts < _VL2_CHECK_TTL:
        return _vl2_cache_result
    result = _check_videollama2_uncached()
    _vl2_cache_ts = now
    _vl2_cache_result = result
    return result


def _check_videollama2_uncached() -> bool:
    """True when VideoLLaMA2 can actually run, and logs why when it can't.

    Requires a CUDA GPU with ~10GB free VRAM and the ``videollama2`` package
    importable — either pip-installed or vendored at
    backend/services/VideoLLaMA2/. ``find_spec`` is used so the heavy package
    is not imported just to render the settings page.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("VideoLLaMA2 check: no CUDA GPU available")
            return False
        free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
        if free_mb < 9000:  # VideoLLaMA2.1-7B-AV (int8) needs ~10GB
            logger.info(
                "VideoLLaMA2 check: only %.0f MB VRAM free (need ~9000) — "
                "free GPU memory or stop other models", free_mb,
            )
            return False
        import importlib.util
        import os as _os
        import sys as _sys
        if importlib.util.find_spec("videollama2") is None:
            vl2_dir = _os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                "services", "VideoLLaMA2",
            )
            if _os.path.isdir(_os.path.join(vl2_dir, "videollama2")) and vl2_dir not in _sys.path:
                _sys.path.insert(0, vl2_dir)
            if importlib.util.find_spec("videollama2") is None:
                logger.info(
                    "VideoLLaMA2 check: 'videollama2' package not installed — "
                    "rebuild the image with --build-arg ENABLE_VIDEOLLAMA2=1"
                )
                return False
        return True
    except Exception as e:
        logger.info("VideoLLaMA2 check failed: %s", e)
        return False


@router.get("/providers/status")
async def provider_status():
    global _status_cache, _status_cache_ts
    now = time.time()
    if _status_cache and now - _status_cache_ts < _STATUS_CACHE_TTL:
        return _status_cache

    from backend.services.providers.openrouter_provider import PRESETS

    statuses = {}

    # Ollama — every registry host, probed in parallel. "connected" when ANY
    # enabled host is online (a dead primary with a live fallback still
    # serves jobs via automatic failover). The legacy top-level fields keep
    # reporting the first online host so existing UI/consumers don't break.
    try:
        from backend.services import ollama_registry
        host_statuses = await ollama_registry.registry_status()
        online = [h for h in host_statuses if h["online"] and h["enabled"]]
        if online:
            statuses["ollama"] = {
                "status": "connected",
                "models_loaded": online[0]["models"],
                "host": online[0]["url"],
                "active_host_name": online[0]["name"],
                "hosts": host_statuses,
            }
        else:
            statuses["ollama"] = {
                "status": "offline",
                "error": (host_statuses[0]["error"] if host_statuses
                          else "no Ollama hosts configured"),
                "hosts": host_statuses,
            }
    except Exception as e:
        statuses["ollama"] = {"status": "offline", "error": str(e)}

    # Companion GPU summary for the "active models" indicator: is a paired
    # Companion serving jobs, and are the selected local models downloaded onto
    # it and ready to use?
    companion_info = None
    try:
        from backend.services import ollama_registry as _oreg
        _hs = statuses.get("ollama", {}).get("hosts") or []
        _comp = next((h for h in _hs if h.get("is_companion")), None)
        if _comp is None:
            _comp = next((h for h in _hs
                          if not _oreg.is_local_gpu_host(h.get("url", ""))
                          and str(h.get("url", "")).rstrip("/").endswith("/ollama")), None)
        if _comp is not None:
            want = []
            for _m in (settings.OLLAMA_PRIMARY_MODEL, settings.OLLAMA_EDITORIAL_MODEL,
                       settings.OLLAMA_TRANSLATION_MODEL):
                if _m and _m not in want:
                    want.append(_m)
            have = _comp.get("models") or []
            ready = [m for m in want if _oreg.model_present(have, m)]
            # Remote Whisper is resolved LIVE from this same Companion host (its
            # GPU serves Ollama AND transcription) — no separate URL to sync.
            _comp_url = _comp.get("url", "")
            _comp_base = _comp_url[:-len("/ollama")] if _comp_url.endswith("/ollama") else _comp_url
            try:
                from backend.services import reframer_audio as _ra
                _whisper_remote_on = (_ra.remote_whisper_configured()
                                      and _ra._remote_whisper_base().rstrip("/") == _comp_base.rstrip("/"))
            except Exception:
                _whisper_remote_on = False
            _now_ms = int(time.time() * 1000)
            if bool(_comp.get("online")):
                _companion_seen.update({"url": _comp_url, "last_online_ms": _now_ms})
            _last_seen = (_companion_seen["last_online_ms"]
                          if _companion_seen.get("url") == _comp_url else 0)
            companion_info = {
                "name": _comp.get("name", ""),
                "gpu_name": _comp.get("gpu_name", ""),
                "url": _comp.get("url", ""),
                "online": bool(_comp.get("online")),
                "is_primary": _comp.get("priority") == 0,
                "models_total": len(want),
                "models_ready": len(ready),
                "missing": [m for m in want if m not in ready],
                "ready": bool(want) and len(ready) == len(want) and bool(_comp.get("online")),
                "whisper_remote": _whisper_remote_on,
                # Live-detection fields so the UI can tell "went offline / expired"
                # from "never paired", and show how long ago it was last seen.
                "latency_ms": _comp.get("latency_ms"),
                "error": _comp.get("error"),
                "paused": bool(_comp.get("paused")),
                "in_cooldown": bool(_comp.get("in_cooldown")),
                "last_seen_ms": _last_seen,
            }
    except Exception as _e:
        logger.debug("companion status calc failed: %s", _e)

    # OpenRouter — always read model IDs from settings (the provider does
    # the same), falling back to preset defaults if settings are empty.
    if _key_is_set(settings.OPENROUTER_API_KEY):
        preset_name = settings.OPENROUTER_PRESET
        preset = PRESETS.get(preset_name, PRESETS["free"])
        vision_model = settings.OPENROUTER_PRIMARY_MODEL or preset["vision"]
        text_model = settings.OPENROUTER_EDITORIAL_MODEL or preset["text"]
        summary_model = settings.OPENROUTER_SUMMARY_MODEL or text_model
        statuses["openrouter"] = {
            "status": "configured",
            "preset": preset_name,
            "vision_model": vision_model,
            "summary_model": summary_model,
            "text_model": text_model,
        }
    else:
        statuses["openrouter"] = {"status": "not_configured"}

    # Anthropic
    if _key_is_set(settings.ANTHROPIC_API_KEY):
        statuses["anthropic"] = {"status": "configured"}
    else:
        statuses["anthropic"] = {"status": "not_configured"}

    # Gemini
    if _key_is_set(settings.GEMINI_API_KEY):
        statuses["gemini"] = {"status": "configured"}
    else:
        statuses["gemini"] = {"status": "not_configured"}

    # Groq
    if _key_is_set(settings.GROQ_API_KEY):
        statuses["groq"] = {"status": "configured"}
    else:
        statuses["groq"] = {"status": "not_configured"}

    # Replicate (cloud GPU for VideoLLaMA)
    if _key_is_set(settings.REPLICATE_API_KEY):
        statuses["replicate"] = {
            "status": "configured",
            "model": settings.REPLICATE_MODEL,
            "videollama3_enhanced": bool(getattr(settings, "VIDEOLLAMA3_ENHANCED", False)),
        }
    else:
        statuses["replicate"] = {"status": "not_configured"}

    # HuggingFace (speaker diarization)
    hf_token = settings.HF_AUTH_TOKEN
    if hf_token and hf_token.strip():
        statuses["huggingface"] = {"status": "configured", "message": "Token set"}
    else:
        statuses["huggingface"] = {"status": "not_configured", "message": "No HF token"}

    # Determine the active provider and models based on fallback chain
    chain = settings.active_provider_chain
    active_provider = None
    active_primary_model = None
    active_editorial_model = None
    active_summary_model = None
    for name in chain:
        info = statuses.get(name, {})
        st = info.get("status", "not_configured")
        if st in ("connected", "configured"):
            active_provider = name
            if name == "openrouter":
                active_primary_model = info.get("vision_model", "")
                active_summary_model = info.get("summary_model", "")
                active_editorial_model = info.get("text_model", "")
            elif name == "ollama":
                active_primary_model = settings.OLLAMA_PRIMARY_MODEL
                active_editorial_model = settings.OLLAMA_EDITORIAL_MODEL
                active_summary_model = settings.OLLAMA_EDITORIAL_MODEL
            elif name == "gemini":
                active_primary_model = "gemini-2.5-flash"
                active_editorial_model = "gemini-2.5-flash"
                active_summary_model = "gemini-2.5-flash"
            elif name == "anthropic":
                active_primary_model = "claude-sonnet-4"
                active_editorial_model = "claude-sonnet-4"
                active_summary_model = "claude-sonnet-4"
            elif name == "groq":
                active_primary_model = ""
                active_editorial_model = "llama-3.1-8b-instant"
                active_summary_model = "llama-3.1-8b-instant"
            break

    _videollama2_ok = _check_videollama2_available()
    # Offline Mode auto-selects the best installed LOCAL editorial model (it
    # overrides the configured/cloud pick at runtime — see the pipeline + the
    # clipper judge). Reflect that here, reusing the Ollama model list already
    # fetched above so the banner/chips show the model that will actually run.
    if settings.resolve_ai_source("editorial") == "local":
        try:
            from backend.services.local_models import select_local_editorial_models
            _ollama_models = (statuses.get("ollama", {}) or {}).get("models_loaded", []) or []
            # Use the SAME selector the pipeline uses so the banner matches what
            # actually runs: it honors an explicit OLLAMA_EDITORIAL_MODEL pick over
            # the small-GPU param cap (the cap reads the LOCAL card, but editorial
            # can run on a paired Companion GPU). Falls back to the ranker's top.
            _picked = await select_local_editorial_models(
                limit=1, model_names=_ollama_models)
            if _picked:
                active_editorial_model = _picked[0]
                active_summary_model = _picked[0]
        except Exception:
            pass
    # Offline Mode (or a per-engine "local" override) routes clip detection to
    # the local Ollama vision model — the pipeline disables Replicate in that
    # case (reframer_clipper.replicate_enabled = … and resolve_ai_source("clip")
    # == "cloud"). Mirror that here so the Active-Models banner + header chips
    # show the LOCAL primary engine instead of the now-inactive cloud one.
    _clip_local = settings.resolve_ai_source("clip") == "local"
    _replicate_on = (
        _key_is_set(settings.REPLICATE_API_KEY)
        and settings.REPLICATE_ENABLED
        and not _clip_local
    )
    statuses["_active"] = {
        "provider": active_provider or "none",
        "transcript_model": settings.WHISPER_MODEL,
        "whisper_beam_size": settings.WHISPER_BEAM_SIZE,
        "whisper_vad_filter": settings.WHISPER_VAD_FILTER,
        # ── Primary AI (video/vision) + Editorial AI (scoring/summary) ──
        "primary_model": active_primary_model or "",
        "editorial_model": active_editorial_model or "",
        "videollama2_available": _videollama2_ok,
        "replicate_available": _replicate_on,
        "replicate_model": (settings.REPLICATE_MODEL
                            if (_key_is_set(settings.REPLICATE_API_KEY) and not _clip_local)
                            else ""),
        "videollama3_enhanced": bool(getattr(settings, "VIDEOLLAMA3_ENHANCED", False)),
        # Resolved clip/editorial sources so the UI can label Offline Mode.
        "clip_source": settings.resolve_ai_source("clip"),
        "editorial_source": settings.resolve_ai_source("editorial"),
        "self_hosted_mode": bool(getattr(settings, "SELF_HOSTED_MODE", False)),
        "primary_type": (
            "replicate" if _replicate_on
            else "videollama2" if _videollama2_ok
            else ("ollama" if active_provider == "ollama" else "cloud")
        ),
        # ── Backward-compat keys — kept so any UI not yet migrated to the
        #    primary/editorial naming keeps rendering. ──
        "vision_model": active_primary_model or "",
        "text_model": active_editorial_model or "",
        "summary_model": active_summary_model or "",
        "preset": settings.OPENROUTER_PRESET if active_provider == "openrouter" else "",
        "fallback_chain": chain,
        "ollama_enabled": "ollama" in chain,
        # Which GPU/host is actually serving Ollama, and the paired-Companion
        # download/readiness summary for the "active models" GPU indicator.
        "ollama_host_name": statuses.get("ollama", {}).get("active_host_name", ""),
        "companion": companion_info,
        "gpu_strict_remote": bool(getattr(settings, "GPU_STRICT_REMOTE", False)),
    }

    _status_cache = statuses
    _status_cache_ts = time.time()
    return statuses


async def _sync_companion_whisper(comp) -> dict:
    """Mirror the paired Companion's transcription quality into THIS container's
    whisper settings. The Companion is the source of truth for quality when it
    runs transcription (the user picks it there); reading its /v1/health and
    writing WHISPER_MODEL / WHISPER_BEAM_SIZE keeps the container's settings,
    logs and any local fallback consistent with what the GPU is actually doing.
    Persists only when a value changed. Returns the effective quality."""
    from backend.services import ollama_registry as _oreg
    out = {"model": "", "beam_size": 0, "quality": ""}
    try:
        base = _oreg.companion_base(comp)
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(_oreg.join_url(base, "/v1/health"),
                                 headers=_oreg.auth_headers(comp))
        if r.status_code != 200:
            return out
        h = r.json() or {}
        model = (h.get("whisper_model_effective") or "").strip()
        beam = int(h.get("whisper_beam_size", 0) or 0)
        quality = (h.get("whisper_quality") or "").strip()
        out = {"model": model, "beam_size": beam, "quality": quality}
        # On quality="auto" the Companion's advertised tier is a SNAPSHOT of
        # this instant's free VRAM, not a user preference — polled right
        # after the editorial LLM loads, a 12 GB card "serves small beam 1".
        # The measured run: main decode ran large-v3-turbo beam 5, then the
        # 14B loaded for polish, the health poll caught that moment, and this
        # sync PERSISTED beam=1 — silently degrading every later decode
        # (recovery passes and future jobs) until someone re-saved settings.
        # Only an EXPLICIT Companion quality choice (fast/balanced/max) is a
        # preference worth mirroring.
        if quality.lower() in ("", "auto"):
            logger.debug(
                "Companion whisper sync skipped: quality=auto reports the "
                "moment's VRAM tier (%s beam %s), not a user choice",
                model or "?", beam or "?")
            return out
        changed = False
        if model and model != getattr(settings, "WHISPER_MODEL", ""):
            if bool(getattr(settings, "WHISPER_MODEL_USER_SET", False)):
                # The user PINNED a model in this container's Settings. That
                # pin is authoritative and rides to the Companion on every
                # transcription request (X-ClipAI-Whisper-Model) — mirroring
                # the Companion's currently-loaded model back over it here
                # silently REVERTED the user's choice (the observed
                # large-v3 → turbo bounce right after saving), and
                # _persist_user_settings then wrote the revert to disk.
                logger.debug(
                    "Companion whisper sync: keeping user-pinned WHISPER_MODEL=%r "
                    "(Companion currently runs %r)",
                    getattr(settings, "WHISPER_MODEL", ""), model)
            else:
                settings.WHISPER_MODEL = model
                changed = True
        if beam > 0 and beam != int(getattr(settings, "WHISPER_BEAM_SIZE", 0) or 0):
            settings.WHISPER_BEAM_SIZE = beam
            changed = True
        if changed:
            _persist_user_settings()
            _invalidate_status_cache()
            logger.info("Synced Whisper settings from Companion: model=%s beam=%s "
                        "(quality=%s)", model, beam, quality)
    except Exception as e:
        logger.debug("companion whisper sync skipped: %s", e)
    return out


@router.get("/providers/companion-status")
async def companion_status():
    """Cheap, fast-pollable liveness of the paired GPU Companion so the UI can
    detect within ~10s when it goes offline / expires — without the heavy
    all-hosts/all-providers work of /providers/status. Probes ONLY the Companion
    host (result cached ~10s in the registry)."""
    from backend.services import ollama_registry as _oreg
    hosts = _oreg.get_hosts()
    comp = _find_companion_host(hosts)
    if comp is None:
        return {"paired": False, "online": False}
    st = await _oreg.probe(comp)
    now_ms = int(time.time() * 1000)
    if st.online:
        _companion_seen.update({"url": comp.url, "last_online_ms": now_ms})
    last_seen = (_companion_seen["last_online_ms"]
                 if _companion_seen.get("url") == comp.url else 0)
    want: list[str] = []
    for _m in (settings.OLLAMA_PRIMARY_MODEL, settings.OLLAMA_EDITORIAL_MODEL,
               settings.OLLAMA_TRANSLATION_MODEL):
        if _m and _m not in want:
            want.append(_m)
    ready = [m for m in want if _oreg.model_present(st.models, m)]
    comp_base = comp.url[:-len("/ollama")] if comp.url.endswith("/ollama") else comp.url
    try:
        from backend.services import reframer_audio as _ra
        _whisper_remote = (_ra.remote_whisper_configured()
                           and _ra._remote_whisper_base().rstrip("/") == comp_base.rstrip("/"))
    except Exception:
        _whisper_remote = False
    # When this Companion serves transcription, mirror its quality (model + beam)
    # into the container's whisper settings so they stay in sync with what the
    # user picked on the Companion.
    _wq = {"model": "", "beam_size": 0, "quality": ""}
    if st.online and _whisper_remote:
        _wq = await _sync_companion_whisper(comp)
    return {
        "paired": True,
        "online": bool(st.online),
        "name": comp.name,
        "gpu_name": comp.gpu_name,
        "url": comp.url,
        "latency_ms": st.latency_ms,
        "error": st.error,
        "paused": bool(getattr(st, "paused", False)),
        "in_cooldown": _oreg.in_cooldown(comp),
        "last_seen_ms": last_seen,
        "now_ms": now_ms,
        "models_total": len(want),
        "models_ready": len(ready),
        "ready": bool(want) and len(ready) == len(want) and bool(st.online),
        "whisper_remote": _whisper_remote,
        "whisper_quality": _wq.get("quality", ""),
        "whisper_model": _wq.get("model", ""),
        "whisper_beam_size": _wq.get("beam_size", 0),
    }


@router.get("/providers/companion-logs")
async def companion_logs():
    """Pull the full diagnostics/log report from every connected Companion so
    the user can download them from the ClipAI export menu — no need to be at
    the Companion PC. Returns one entry per remote host (a home setup usually
    has one). Fail-soft per host so one offline Companion doesn't break the rest."""
    from backend.services import ollama_registry as _oreg
    hosts = [h for h in _oreg.get_hosts() if not _oreg.is_local_gpu_host(h.url)]
    companions = []
    for h in hosts:
        base = _oreg.companion_base(h)
        entry = {"name": h.name or base, "url": base, "ok": False, "text": "", "error": ""}
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.get(_oreg.join_url(base, "/v1/logs"),
                                     headers=_oreg.auth_headers(h))
            if r.status_code == 200:
                entry["ok"] = True
                entry["text"] = r.text
            elif r.status_code in (401, 403):
                entry["error"] = "auth rejected — check the host token"
            elif r.status_code == 404:
                entry["error"] = "this Companion is too old to export logs remotely (update it)"
            else:
                entry["error"] = f"HTTP {r.status_code}"
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:120]}"
        companions.append(entry)
    return {"companions": companions}


class CompanionImportRequest(BaseModel):
    host_id: str
    path: str
    kind: str = "video"          # "video" | "media" | "font"
    size: int = 0                # known file size (from the listing) for % progress
    # Same semantics as the Upload page: empty source = auto-detect, empty
    # target = keep the original language. This import path used to hand-build
    # the JobResult with NO language fields, so a Companion-imported video
    # always ran auto-detect → default-English with no way to choose.
    source_language: str = ""    # ISO 639-1 of the spoken audio
    target_language: str = ""    # ISO 639-1 to translate subtitles into


# In-memory progress for in-flight Companion video imports, keyed by a short
# import_id. The browser polls /companion-files/import-progress to drive a real
# progress bar (video files can be many hundreds of MB over the LAN).
_import_progress: dict = {}


async def _post_import_hb(base: str, token: str, job_id: str, title: str, stage: str, pct: int):
    """Fire-and-forget a progress heartbeat to the Companion so its GUI shows
    the import too (it's the source of the pull, but has no view of ClipAI's
    overall progress otherwise). Best-effort — never affects the import."""
    hdrs = {
        "X-ClipAI-Progress": str(max(0, min(100, int(pct)))),
        "X-ClipAI-Job-Id": job_id,
        "X-ClipAI-Job-Title": title,
        "X-ClipAI-Stage": stage,
    }
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            await c.post(f"{base}/v1/progress", headers=hdrs)
    except Exception:
        pass


async def _companion_download(read_url, params, headers, dest, total, on_done):
    """Download a shared file to ``dest``. Uses PARALLEL HTTP Range segments when
    the Companion supports them — several connections at once, the same trick
    that makes browser uploads fast — else a single 1 MB-chunked stream.
    ``on_done(bytes_so_far)`` drives the progress bar. Returns bytes written."""
    supports_range = False
    if total and total > 8 * 1024 * 1024:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.get(read_url, params=params,
                                headers={**headers, "Range": "bytes=0-0"})
                supports_range = (r.status_code == 206)
        except Exception:
            supports_range = False

    if supports_range:
        SEG = 32 * 1024 * 1024          # 32 MB per segment
        segments = []
        off = 0
        while off < total:
            end = min(off + SEG, total) - 1
            segments.append((off, end))
            off = end + 1
        with open(dest, "wb") as f:     # preallocate the full file
            f.truncate(total)
        fd = os.open(dest, os.O_WRONLY)
        counter = {"n": 0}
        sem = asyncio.Semaphore(6)      # up to 6 connections at once

        async def _seg(start, end):
            async with sem:
                async with httpx.AsyncClient(timeout=None) as c:
                    async with c.stream("GET", read_url, params=params,
                                        headers={**headers, "Range": f"bytes={start}-{end}"}) as resp:
                        if resp.status_code not in (206, 200):
                            raise RuntimeError(f"range {start}-{end}: HTTP {resp.status_code}")
                        pos = start
                        async for chunk in resp.aiter_bytes(1024 * 1024):
                            w = 0
                            while w < len(chunk):
                                w += os.pwrite(fd, chunk[w:], pos + w)
                            pos += len(chunk)
                            counter["n"] += len(chunk)
                            on_done(counter["n"])
        try:
            await asyncio.gather(*[_seg(s, e) for s, e in segments])
        finally:
            os.close(fd)
        return counter["n"]

    # Single-stream fallback (range unsupported / size unknown / small file).
    size = 0
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("GET", read_url, params=params, headers=headers) as resp:
            if resp.status_code != 200:
                body = (await resp.aread())[:200].decode("utf-8", "ignore")
                raise RuntimeError(f"companion read {resp.status_code}: {body}")
            with open(dest, "wb", buffering=4 * 1024 * 1024) as f:
                async for chunk in resp.aiter_bytes(1024 * 1024):
                    f.write(chunk)
                    size += len(chunk)
                    on_done(size)
    return size


def _companion_by_id(host_id: str):
    from backend.services import ollama_registry as _oreg
    for h in _oreg.get_hosts():
        if not _oreg.is_local_gpu_host(h.url) and h.id == host_id:
            return h
    return None


@router.get("/providers/companion-files/roots")
async def companion_file_roots():
    """List each connected Companion's shared folders so the ClipAI web app can
    offer a remote file browser (only shown when a Companion is connected)."""
    from backend.services import ollama_registry as _oreg
    out = []
    for h in _oreg.get_hosts():
        if _oreg.is_local_gpu_host(h.url):
            continue
        base = _oreg.companion_base(h)
        entry = {"host_id": h.id, "name": h.name or base, "online": False, "roots": [], "error": ""}
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                r = await client.get(_oreg.join_url(base, "/v1/files/roots"),
                                     headers=_oreg.auth_headers(h))
            if r.status_code == 200:
                entry["online"] = True
                entry["roots"] = (r.json() or {}).get("roots", [])
            elif r.status_code in (401, 403):
                entry["error"] = "auth rejected — check the host token"
            elif r.status_code == 404:
                entry["error"] = "update this Companion to share folders"
            else:
                entry["error"] = f"HTTP {r.status_code}"
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        out.append(entry)
    return {"companions": out}


@router.post("/providers/companion/force-end-jobs")
async def companion_force_end_jobs():
    """Force-end every active job and free every paired Companion's GPU.

    The remote sibling of the Companion GUI's local "Force end" button,
    driven by the Settings → GPU Companion card. Container-first ordering so
    a cancelled pipeline can't immediately reload the models the Companion
    just evicted:

      1. Every non-terminal analysis job is marked CANCELLED and signalled
         (same semantics as the per-job cancel endpoint's running path).
      2. Every paired Companion gets ``POST /v1/jobs/force-end`` — it clears
         its active-job display, kills the whisper sidecar even mid-decode,
         and evicts all resident Ollama models.

    Fail-soft per job and per Companion; the response reports exactly what
    was ended where."""
    from backend import database as _db
    from backend.models import JobStatus as _JS
    from backend.services import ollama_registry as _oreg
    from backend.services.pipeline import request_cancel as _rc

    _terminal = {_JS.COMPLETE, _JS.FAILED, _JS.CANCELLED}
    cancelled = []
    try:
        jobs = await _db.list_jobs(light=True)
    except Exception:
        jobs = []
    for j in jobs:
        try:
            if j.status in _terminal:
                continue
            await _db.update_job_status(
                j.job_id, status=_JS.CANCELLED,
                progress_message="Force-ended from GPU Companion settings")
            _rc(j.job_id)
            cancelled.append(j.job_id)
        except Exception:
            continue

    companions = []
    for h in _oreg.get_hosts():
        if _oreg.is_local_gpu_host(h.url):
            continue
        base = _oreg.companion_base(h)
        entry = {"host_id": h.id, "name": h.name or base, "ok": False,
                 "ended_job": None, "whisper_stopped": False,
                 "ollama_unloaded": 0, "error": ""}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                r = await client.post(_oreg.join_url(base, "/v1/jobs/force-end"),
                                      headers=_oreg.auth_headers(h))
            if r.status_code == 200:
                d = r.json() or {}
                entry.update(ok=True, ended_job=d.get("ended_job"),
                             whisper_stopped=bool(d.get("whisper_stopped")),
                             ollama_unloaded=int(d.get("ollama_unloaded") or 0))
            elif r.status_code == 404:
                entry["error"] = ("this Companion predates remote force-end — "
                                  "update it from this card first")
            elif r.status_code in (401, 403):
                entry["error"] = "auth rejected — check the host token"
            else:
                entry["error"] = f"HTTP {r.status_code}"
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        companions.append(entry)
    logger.info(
        "Force-end all jobs: %d job(s) cancelled, %d companion(s) signalled",
        len(cancelled), len(companions))
    return {"jobs_cancelled": cancelled, "companions": companions}


@router.get("/providers/companion-files/list")
async def companion_file_list(host_id: str, path: str):
    """Proxy a directory listing from a Companion's shared folder (jailed on the
    Companion side to the folders the user shared)."""
    from backend.services import ollama_registry as _oreg
    h = _companion_by_id(host_id)
    if h is None:
        raise HTTPException(status_code=404, detail="companion not found or not connected")
    base = _oreg.companion_base(h)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(_oreg.join_url(base, "/v1/files/list"),
                                 params={"path": path}, headers=_oreg.auth_headers(h))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"companion unreachable: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:200])
    return r.json()


@router.get("/providers/companion-files/thumb")
async def companion_file_thumb(host_id: str, path: str, v: str = ""):
    """Return a small JPEG thumbnail for a shared image/video, generated with
    ffmpeg reading the file straight off the Companion (auth'd). Cached on disk.
    Returns 204 (no content) on any failure so the browser falls back to an
    icon — never blocks the file browser. Video thumbs can be slow for large
    moov-at-end files (no range seek yet); a 20s cap guards that."""
    import hashlib
    import subprocess
    import urllib.parse
    from fastapi.responses import FileResponse, Response
    from backend.services import ollama_registry as _oreg

    h = _companion_by_id(host_id)
    if h is None:
        return Response(status_code=204)
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    IMAGE = {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}
    VIDEO = {"mp4", "mov", "mkv", "webm", "avi", "m4v", "mpg", "mpeg", "wmv", "flv"}
    if ext not in IMAGE and ext not in VIDEO:
        return Response(status_code=204)

    cache_dir = "/tmp/clipai_companion_thumbs"
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(f"{host_id}|{path}|{v}".encode()).hexdigest()
    out = os.path.join(cache_dir, key + ".jpg")
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return FileResponse(out, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})

    base = _oreg.companion_base(h)
    read_url = _oreg.join_url(base, "/v1/files/read") + "?" + urllib.parse.urlencode({"path": path})
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    tok = getattr(h, "token", "") or ""
    if tok:
        cmd += ["-headers", f"Authorization: Bearer {tok}\r\n"]
    if ext in VIDEO:
        cmd += ["-ss", "1"]          # grab a frame ~1s in (input seek)
    cmd += ["-i", read_url, "-frames:v", "1", "-vf", "scale=360:-2", out]
    try:
        subprocess.run(cmd, capture_output=True, timeout=20)
    except Exception:
        pass
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return FileResponse(out, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})
    return Response(status_code=204)


@router.post("/providers/companion-files/import")
async def companion_file_import(req: CompanionImportRequest):
    """Pull a file from a Companion's shared folder into ClipAI: a video becomes
    a queued analysis job; media/fonts land in the media library / fonts dir.
    Streams the bytes (never loads the whole video into memory)."""
    from backend.services import ollama_registry as _oreg
    h = _companion_by_id(req.host_id)
    if h is None:
        raise HTTPException(status_code=404, detail="companion not found or not connected")
    base = _oreg.companion_base(h)
    read_url = _oreg.join_url(base, "/v1/files/read")
    headers = _oreg.auth_headers(h)
    filename = os.path.basename((req.path or "").replace("\\", "/")) or "import.bin"
    kind = (req.kind or "video").lower()

    async def _stream_to(dest_path: str) -> int:
        size = 0
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("GET", read_url, params={"path": req.path},
                                     headers=headers) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread())[:200].decode("utf-8", "ignore")
                    raise HTTPException(status_code=resp.status_code,
                                        detail=f"companion read failed: {body}")
                with open(dest_path, "wb", buffering=4 * 1024 * 1024) as f:
                    async for chunk in resp.aiter_bytes(1024 * 1024):
                        f.write(chunk)
                        size += len(chunk)
        return size

    if kind == "video":
        # Video files are big — download in the BACKGROUND and report progress so
        # the browser can show a real bar (like a normal upload). The job is only
        # created (and thus queued for analysis) once the file is fully on disk,
        # so the pipeline never grabs a half-downloaded file.
        import_id = uuid.uuid4().hex[:12]
        job_id = str(uuid.uuid4())
        _import_progress[import_id] = {
            "done": 0, "total": int(req.size or 0), "status": "downloading",
            "job_id": None, "error": "", "filename": filename,
        }

        async def _bg_import():
            import shutil as _sh
            from datetime import datetime, timezone
            from backend import database as _db
            from backend.models import JobResult, JobStatus
            job_dir = f"/data/uploads/{job_id}"
            os.makedirs(job_dir, exist_ok=True)
            # Save to the CANONICAL path the pipeline + preview player expect
            # (/data/uploads/<job>/video.<ext>) — not the original filename, or
            # the analysis-page player's /api/files/<job>/video.<ext> 404s
            # ("Failed to load video"). The original name rides on job.filename.
            _ext = (os.path.splitext(filename)[1].lstrip(".").lower() or "mp4")
            dest = os.path.join(job_dir, f"video.{_ext}")
            try:
                _total = int(req.size or 0)
                _comp_token = getattr(h, "token", "") or ""
                _hb = {"t": 0.0, "pct": -1}

                def _on_done(n):
                    _import_progress[import_id]["done"] = n
                    # Throttled heartbeat → the Companion shows "Importing … X%".
                    try:
                        pct = int(n * 100 / _total) if _total else 0
                        now = time.monotonic()
                        if now - _hb["t"] >= 1.5 and pct != _hb["pct"]:
                            _hb["t"] = now
                            _hb["pct"] = pct
                            asyncio.create_task(_post_import_hb(
                                base, _comp_token, job_id,
                                f"Importing {filename}", "downloading to ClipAI", pct))
                    except Exception:
                        pass

                size = await _companion_download(
                    read_url, {"path": req.path}, headers, dest, _total, _on_done)
                if size == 0:
                    raise RuntimeError("imported file is empty")
                # Final 100% so the Companion bar completes before analysis.
                asyncio.create_task(_post_import_hb(
                    base, _comp_token, job_id, f"Importing {filename}", "imported", 100))
                now = datetime.now(timezone.utc).isoformat()
                job = JobResult(
                    job_id=job_id, filename=filename, file_path=dest,
                    file_size_mb=round(size / (1024 * 1024), 2),
                    status=JobStatus.QUEUED, progress=0,
                    progress_message="Imported from Companion, waiting for analysis",
                    created_at=now, updated_at=now,
                    # The user's picks from the import dialog — the pipeline
                    # reads exactly these two fields (job.language drives
                    # Whisper's language hint, job.subtitle_language the
                    # translation target), so setting them here gives the
                    # Companion path full upload-parity.
                    language=(req.source_language or "").strip().lower(),
                    subtitle_language=(req.target_language or "").strip().lower(),
                )
                await _db.save_job(job)
                _import_progress[import_id].update({"status": "complete", "job_id": job_id})
                # START the analysis pipeline — saving a QUEUED job does NOT
                # enqueue it (uploads call run_analysis explicitly); without this
                # the imported video sits at "waiting for analysis" forever.
                try:
                    from backend.services.pipeline import run_analysis
                    asyncio.create_task(run_analysis(job_id))
                except Exception as _an_err:
                    logger.error("Imported job %s: failed to start analysis: %s", job_id, _an_err)
            except Exception as e:
                _sh.rmtree(job_dir, ignore_errors=True)
                _import_progress[import_id].update({"status": "error", "error": str(e)[:200]})

        asyncio.create_task(_bg_import())
        return {"ok": True, "kind": "video", "import_id": import_id,
                "job_id": job_id, "filename": filename}

    if kind == "media":
        from backend.routers.media import (
            UPLOAD_DIR, GLOBAL_LIBRARY_ID, detect_media_type, _load_meta, _save_meta,
        )
        mtype = detect_media_type(filename)
        if not mtype:
            raise HTTPException(status_code=400, detail=f"unsupported media type: {filename}")
        media_dir = os.path.join(UPLOAD_DIR, GLOBAL_LIBRARY_ID, "media")
        os.makedirs(media_dir, exist_ok=True)
        media_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(filename)[1].lower()
        safe = f"{media_id}{ext}"
        dest = os.path.join(media_dir, safe)
        try:
            size = await _stream_to(dest)
        except Exception:
            try:
                os.remove(dest)
            except OSError:
                pass
            raise
        meta = _load_meta(media_dir)
        meta[media_id] = {"original_filename": filename}
        _save_meta(media_dir, meta)
        return {"ok": True, "kind": "media", "id": media_id, "filename": filename,
                "type": mtype, "size": size,
                "url": f"/api/files/{GLOBAL_LIBRARY_ID}/media/{safe}"}

    if kind == "font":
        from backend.routers.fonts import FONTS_DIR
        if not filename.lower().endswith((".ttf", ".otf", ".ttc", ".woff", ".woff2")):
            raise HTTPException(status_code=400, detail="not a font file")
        os.makedirs(FONTS_DIR, exist_ok=True)
        safe = os.path.basename(filename)
        dest = os.path.join(FONTS_DIR, safe)
        try:
            size = await _stream_to(dest)
        except Exception:
            try:
                os.remove(dest)
            except OSError:
                pass
            raise
        try:
            import subprocess as _sp
            _sp.run(["fc-cache", "-f", FONTS_DIR], capture_output=True, timeout=10)
        except Exception:
            pass
        return {"ok": True, "kind": "font", "filename": safe, "size": size}

    raise HTTPException(status_code=400, detail=f"unknown import kind: {kind}")


@router.get("/providers/companion-files/import-progress")
async def companion_import_progress(import_id: str):
    """Poll the progress of a background video import (bytes done + status).
    The browser turns this into a progress bar and navigates when complete."""
    st = _import_progress.get(import_id)
    if st is None:
        raise HTTPException(status_code=404, detail="unknown import")
    # Once terminal, let it be garbage-collected after the client reads it.
    if st.get("status") in ("complete", "error"):
        st = dict(st)
        _import_progress.pop(import_id, None)
    return st


# ── Companion path bookmarks ────────────────────────────────────────────────
# Starred paths in the Companion file browser, persisted server-side (per
# deployment, keyed by Companion host_id) so they survive container restarts
# and follow the user across browsers/devices — unlike localStorage.

_BOOKMARKS_PATH = os.path.join(_DATA_DIR, "companion_bookmarks.json")
_BOOKMARKS_MAX_PER_HOST = 100
_bookmarks_lock = threading.Lock()


def _normalize_companion_path(p: str) -> str:
    r"""Strip Windows "verbatim" prefixes (\\?\C:\..., \\?\UNC\server\...) that
    older Companions leaked from canonicalized listings. The plain form
    resolves identically on the Companion; the prefixed form broke display and
    equality — a legacy \\?\-prefixed bookmark could never be un-starred by a
    client sending the clean path."""
    p = (p or "").strip()
    if p.startswith("\\\\?\\UNC\\"):
        return "\\\\" + p[8:]
    if p.startswith("\\\\?\\"):
        return p[4:]
    return p


def _normalized_marks(marks: list) -> list:
    """Bookmarks with verbatim prefixes stripped, deduped on the clean path."""
    out, seen = [], set()
    for m in marks or []:
        p = _normalize_companion_path(str(m.get("path") or ""))
        if not p or p in seen:
            continue
        seen.add(p)
        out.append({**m, "path": p})
    return out


def _load_bookmarks() -> dict:
    """``{host_id: [{path, name, is_dir, added_ms}, …]}`` — {} when missing/corrupt."""
    try:
        with open(_BOOKMARKS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_bookmarks(data: dict) -> None:
    tmp = _BOOKMARKS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _BOOKMARKS_PATH)


class CompanionBookmarkRequest(BaseModel):
    host_id: str
    path: str
    name: str = ""
    is_dir: bool = True


@router.get("/providers/companion-files/bookmarks")
async def companion_bookmarks_list(host_id: str):
    """The starred paths for one Companion (newest first)."""
    with _bookmarks_lock:
        marks = _normalized_marks(_load_bookmarks().get(host_id, []))
    return {"bookmarks": marks}


@router.post("/providers/companion-files/bookmarks")
async def companion_bookmark_add(req: CompanionBookmarkRequest):
    """Star a path. Idempotent on (host_id, path); newest stars sort first."""
    path = _normalize_companion_path(req.path)
    if not req.host_id or not path:
        raise HTTPException(status_code=400, detail="host_id and path are required")
    name = (req.name or "").strip() or os.path.basename(path.replace("\\", "/").rstrip("/\\")) or path
    with _bookmarks_lock:
        data = _load_bookmarks()
        marks = [m for m in _normalized_marks(data.get(req.host_id, [])) if m["path"] != path]
        marks.insert(0, {"path": path, "name": name, "is_dir": bool(req.is_dir),
                         "added_ms": int(time.time() * 1000)})
        data[req.host_id] = marks[:_BOOKMARKS_MAX_PER_HOST]
        _save_bookmarks(data)
        marks = data[req.host_id]
    return {"ok": True, "bookmarks": marks}


@router.delete("/providers/companion-files/bookmarks")
async def companion_bookmark_remove(host_id: str, path: str):
    """Un-star a path. Removing an unknown path is a no-op, not an error.
    Compared on the normalized path so a clean ``C:\\…`` removes a legacy
    ``\\\\?\\C:\\…`` bookmark too."""
    path = _normalize_companion_path(path)
    with _bookmarks_lock:
        data = _load_bookmarks()
        marks = [m for m in _normalized_marks(data.get(host_id, [])) if m["path"] != path]
        data[host_id] = marks
        _save_bookmarks(data)
    return {"ok": True, "bookmarks": marks}


# ── Bulk folder import (sequential) ─────────────────────────────────────────
# "Import every video in this shared folder": ClipAI pulls + fully analyzes
# them ONE AT A TIME (download → transcribe → translate → clips for video N
# completes before video N+1 even starts downloading), until the folder is
# done or the ClipAI device runs out of disk space. State lives in memory and
# survives the dialog being closed; the browser re-attaches via /active.

_BULK_VIDEO_EXTS = {"mp4", "mov", "mkv", "avi", "webm", "m4v", "mpg", "mpeg", "wmv", "flv"}
_BULK_KEEP_TERMINAL = 8            # finished runs kept for the summary screen
_BULK_DISK_FLOOR = 2 * 1024 ** 3   # always leave ≥ 2 GB free on /data
_bulk_imports: dict = {}           # bulk_id -> state (insertion-ordered)


class CompanionFolderImportRequest(BaseModel):
    host_id: str
    path: str
    # Same semantics as the single-file import: empty source = auto-detect,
    # empty target = keep the original language.
    source_language: str = ""
    target_language: str = ""


class _BulkCancelled(Exception):
    """Raised inside the download progress callback to abort mid-transfer."""


def _bulk_data_root() -> str:
    """Volume the imports land on — what the free-space check must watch."""
    return "/data" if os.path.isdir("/data") else "."


def _bulk_disk_free() -> int:
    import shutil
    try:
        return shutil.disk_usage(_bulk_data_root()).free
    except Exception:
        return 0


# Thin awaitable seams around the heavy pipeline/database modules so the bulk
# runner is unit-testable without importing torch & friends.
async def _bulk_save_job(job) -> None:
    from backend import database as _db
    await _db.save_job(job)


async def _bulk_job_state(job_id: str) -> tuple:
    """(status_value, error) of a job after its analysis run finished."""
    from backend import database as _db
    job = await _db.load_job(job_id)
    if job is None:
        return ("failed", "job vanished from the database")
    status = getattr(job.status, "value", job.status)
    return (str(status), getattr(job, "error", "") or "")


async def _bulk_run_analysis(job_id: str) -> None:
    from backend.services.pipeline import run_analysis
    await run_analysis(job_id)     # never raises: failures land on the job row


def _bulk_request_job_cancel(job_id: str) -> None:
    try:
        from backend.services.pipeline import request_cancel
        request_cancel(job_id)
    except Exception:
        pass


async def _companion_list_videos(h, path: str) -> list:
    """The importable videos directly inside one shared folder (A–Z)."""
    from backend.services import ollama_registry as _oreg
    base = _oreg.companion_base(h)
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(_oreg.join_url(base, "/v1/files/list"),
                                 params={"path": path}, headers=_oreg.auth_headers(h))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"companion unreachable: {e}")
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:200])
    entries = ((r.json() or {}).get("entries")) or []
    vids = [e for e in entries
            if not e.get("is_dir") and (e.get("ext") or "").lower() in _BULK_VIDEO_EXTS]
    vids.sort(key=lambda e: (e.get("name") or "").lower())
    return vids


def _bulk_prune_terminal() -> None:
    done = [k for k, s in _bulk_imports.items() if s.get("status") != "running"]
    for k in done[:-_BULK_KEEP_TERMINAL]:
        _bulk_imports.pop(k, None)


async def _run_bulk_import(bulk_id: str) -> None:
    """The sequential worker: for each queued video — free-space check,
    download from the Companion, create the job, and AWAIT the full analysis
    pipeline before touching the next file. One failed video is recorded and
    skipped over; exhausted disk stops the whole batch."""
    st = _bulk_imports.get(bulk_id)
    if st is None:
        return
    from datetime import datetime, timezone
    from backend.services import ollama_registry as _oreg
    from backend.models import JobResult, JobStatus

    h = _companion_by_id(st["host_id"])
    if h is None:
        st.update(status="error", error="companion not found or not connected")
        return
    base = _oreg.companion_base(h)
    read_url = _oreg.join_url(base, "/v1/files/read")
    headers = _oreg.auth_headers(h)
    comp_token = getattr(h, "token", "") or ""
    total = len(st["items"])

    def _skip_rest(from_idx: int) -> None:
        for later in st["items"][from_idx:]:
            if later["status"] == "queued":
                later["status"] = "skipped"

    for i, item in enumerate(st["items"]):
        if st.get("cancel"):
            st["status"] = "cancelled"
            _skip_rest(i)
            st["current"] = -1
            return
        st["current"] = i

        # Free-space gate: the file itself + pipeline scratch (audio WAV,
        # frames — the pipeline's own pre-check budgets ~30 % of the file
        # size) while always keeping the floor untouched.
        need = int(int(item.get("size") or 0) * 1.5) + _BULK_DISK_FLOOR
        if _bulk_disk_free() < need:
            item["status"] = "no_space"
            item["error"] = "not enough free disk space on the ClipAI device"
            st["status"] = "out_of_space"
            _skip_rest(i + 1)
            st["current"] = -1
            logger.warning("Bulk import %s stopped: out of disk space at %s (%d of %d done)",
                           bulk_id, item.get("name"), st["ok"], total)
            return

        import shutil as _sh
        job_id = str(uuid.uuid4())
        job_dir = os.path.join(_bulk_data_root(), "uploads", job_id)
        os.makedirs(job_dir, exist_ok=True)
        filename = item.get("name") or os.path.basename((item.get("path") or "").replace("\\", "/")) or "import.bin"
        # Canonical pipeline path (video.<ext>) — same as the single import.
        _ext = (os.path.splitext(filename)[1].lstrip(".").lower() or "mp4")
        dest = os.path.join(job_dir, f"video.{_ext}")

        item["status"] = "downloading"
        _total = int(item.get("size") or 0)
        _hb = {"t": 0.0, "pct": -1}

        def _on_done(n, _item=item, _t=_total, _jid=job_id, _fn=filename, _idx=i, _hb=_hb):
            _item["done_bytes"] = n
            if st.get("cancel"):
                raise _BulkCancelled()
            try:
                pct = int(n * 100 / _t) if _t else 0
                now = time.monotonic()
                if now - _hb["t"] >= 1.5 and pct != _hb["pct"]:
                    _hb["t"] = now
                    _hb["pct"] = pct
                    asyncio.create_task(_post_import_hb(
                        base, comp_token, _jid,
                        f"Importing {_fn} ({_idx + 1}/{total})", "downloading to ClipAI", pct))
            except Exception:
                pass

        try:
            size = await _companion_download(
                read_url, {"path": item["path"]}, headers, dest, _total, _on_done)
            if size == 0:
                raise RuntimeError("imported file is empty")
        except _BulkCancelled:
            _sh.rmtree(job_dir, ignore_errors=True)
            item["status"] = "cancelled"
            st["status"] = "cancelled"
            _skip_rest(i + 1)
            st["current"] = -1
            return
        except Exception as e:
            _sh.rmtree(job_dir, ignore_errors=True)
            item["status"] = "failed"
            item["error"] = str(e)[:200]
            st["failed"] += 1
            st["done"] += 1
            continue

        now_iso = datetime.now(timezone.utc).isoformat()
        job = JobResult(
            job_id=job_id, filename=filename, file_path=dest,
            file_size_mb=round(size / (1024 * 1024), 2),
            status=JobStatus.QUEUED, progress=0,
            progress_message=f"Imported from Companion folder ({i + 1} of {total}), waiting for analysis",
            created_at=now_iso, updated_at=now_iso,
            language=(st.get("source_language") or "").strip().lower(),
            subtitle_language=(st.get("target_language") or "").strip().lower(),
        )
        try:
            await _bulk_save_job(job)
        except Exception as e:
            _sh.rmtree(job_dir, ignore_errors=True)
            item["status"] = "failed"
            item["error"] = f"could not create job: {str(e)[:160]}"
            st["failed"] += 1
            st["done"] += 1
            continue
        item["job_id"] = job_id
        item["status"] = "analyzing"

        # THE sequential barrier: run_analysis resolves only when this video's
        # whole pipeline (transcription → translation → clips) has finished.
        try:
            await _bulk_run_analysis(job_id)
            status, err = await _bulk_job_state(job_id)
        except Exception as e:                       # defensive; run_analysis is fail-soft
            status, err = ("failed", str(e)[:200])
        if status == "complete":
            item["status"] = "complete"
            st["ok"] += 1
        elif status == "cancelled" and st.get("cancel"):
            item["status"] = "cancelled"
            st["status"] = "cancelled"
            _skip_rest(i + 1)
            st["current"] = -1
            st["done"] += 1
            return
        else:
            # Individually-cancelled or failed job: record it, keep the batch going.
            item["status"] = "failed"
            item["error"] = (err or f"analysis ended with status {status}")[:200]
            st["failed"] += 1
        st["done"] += 1

    st["current"] = -1
    st["status"] = "cancelled" if st.get("cancel") else "complete"
    logger.info("Bulk import %s finished: %d ok, %d failed of %d",
                bulk_id, st["ok"], st["failed"], total)


@router.post("/providers/companion-files/import-folder")
async def companion_folder_import(req: CompanionFolderImportRequest):
    """Start a sequential bulk import of every video directly inside one
    Companion shared folder. Returns a ``bulk_id`` to poll; the run continues
    server-side even if the browser dialog is closed."""
    h = _companion_by_id(req.host_id)
    if h is None:
        raise HTTPException(status_code=404, detail="companion not found or not connected")
    if any(s.get("status") == "running" for s in _bulk_imports.values()):
        raise HTTPException(status_code=409,
                            detail="a folder import is already running — wait for it to finish or cancel it")
    videos = await _companion_list_videos(h, req.path)
    if not videos:
        raise HTTPException(status_code=400, detail="no videos found in this folder")

    bulk_id = uuid.uuid4().hex[:12]
    folder = (req.path or "").rstrip("/\\")
    st = {
        "bulk_id": bulk_id,
        "host_id": req.host_id,
        "folder": req.path,
        "folder_name": os.path.basename(folder.replace("\\", "/")) or folder,
        "status": "running",
        "error": "",
        "total": len(videos),
        "done": 0, "ok": 0, "failed": 0,
        "current": -1,
        "cancel": False,
        "started_ms": int(time.time() * 1000),
        "source_language": req.source_language,
        "target_language": req.target_language,
        "items": [{
            "name": v.get("name") or "",
            "path": v.get("path") or "",
            "size": int(v.get("size") or 0),
            "status": "queued",
            "job_id": "",
            "error": "",
            "done_bytes": 0,
        } for v in videos],
    }
    _bulk_prune_terminal()
    _bulk_imports[bulk_id] = st
    asyncio.create_task(_run_bulk_import(bulk_id))
    logger.info("Bulk import %s started: %d video(s) from %s", bulk_id, len(videos), req.path)
    return {"ok": True, "bulk_id": bulk_id, "total": len(videos), "folder": req.path}


@router.get("/providers/companion-files/import-folder/progress")
async def companion_folder_import_progress(bulk_id: str):
    """Live state of a bulk import. While a video is analyzing, its item is
    enriched with the job's pipeline progress so the dialog can show one
    honest bar per stage (downloading % → analysis %)."""
    st = _bulk_imports.get(bulk_id)
    if st is None:
        raise HTTPException(status_code=404, detail="unknown bulk import")
    out = {k: v for k, v in st.items() if k != "cancel"}
    out["items"] = [dict(it) for it in st["items"]]
    for it in out["items"]:
        if it["status"] == "analyzing" and it.get("job_id"):
            try:
                from backend import database as _db
                job = await _db.load_job(it["job_id"])
                if job is not None:
                    it["analysis_progress"] = int(getattr(job, "progress", 0) or 0)
                    it["analysis_message"] = getattr(job, "progress_message", "") or ""
            except Exception:
                pass
    return out


@router.post("/providers/companion-files/import-folder/cancel")
async def companion_folder_import_cancel(bulk_id: str):
    """Stop a bulk import: aborts an in-flight download immediately, cancels
    the currently-analyzing job, and skips everything still queued."""
    st = _bulk_imports.get(bulk_id)
    if st is None:
        raise HTTPException(status_code=404, detail="unknown bulk import")
    st["cancel"] = True
    cur = st.get("current", -1)
    if 0 <= cur < len(st["items"]):
        it = st["items"][cur]
        if it.get("status") == "analyzing" and it.get("job_id"):
            _bulk_request_job_cancel(it["job_id"])
    return {"ok": True}


@router.get("/providers/companion-files/import-folder/active")
async def companion_folder_import_active():
    """The currently-running bulk import, if any — lets a reopened dialog
    re-attach to a run started earlier (it keeps going server-side)."""
    for bulk_id, s in reversed(list(_bulk_imports.items())):
        if s.get("status") == "running":
            return {"bulk_id": bulk_id}
    return {"bulk_id": None}


@router.post("/providers/test/{provider_name}")
async def test_provider(provider_name: str):
    """Live-test a provider by making a real API call and returning detailed status."""

    if provider_name == "openrouter":
        return await _test_openrouter()
    elif provider_name == "ollama":
        return await _test_ollama()
    elif provider_name == "anthropic":
        return await _test_anthropic()
    elif provider_name == "gemini":
        return await _test_gemini()
    elif provider_name == "groq":
        return await _test_groq()
    elif provider_name == "replicate":
        return await _test_replicate()
    elif provider_name == "huggingface":
        return await _test_huggingface()
    else:
        return {"status": "error", "message": f"Unknown provider: {provider_name}"}


async def _test_openrouter():
    key = settings.OPENROUTER_API_KEY
    if not _key_is_set(key):
        return {
            "status": "not_configured",
            "message": "OPENROUTER_API_KEY is not set. Add it to your .env file.",
            "help": "Get a free key at https://openrouter.ai/keys",
        }

    # Step 1: Validate key by fetching account info
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Check key validity via auth/key endpoint
            auth_resp = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=headers,
            )
            if auth_resp.status_code == 401:
                return {
                    "status": "invalid_key",
                    "message": "API key is invalid or expired. Check your key at openrouter.ai/keys.",
                }
            if auth_resp.status_code == 403:
                return {
                    "status": "invalid_key",
                    "message": "API key is forbidden. It may have been revoked.",
                }

            key_info = {}
            if auth_resp.status_code == 200:
                key_data = auth_resp.json().get("data", {})
                key_info = {
                    "label": key_data.get("label", ""),
                    "usage_usd": key_data.get("usage", 0),
                    "limit_usd": key_data.get("limit"),
                    "is_free_tier": key_data.get("is_free_tier", False),
                    "rate_limit_rpm": key_data.get("rate_limit", {}).get("requests", None),
                }

            # Step 2: Quick model ping — try multiple models to handle unavailable ones
            from backend.services.providers.openrouter_provider import PRESETS
            preset_name = settings.OPENROUTER_PRESET
            preset = PRESETS.get(preset_name, PRESETS["free"])

            # Build a list of models to try: current text model from settings, then fallbacks
            effective_text = settings.OPENROUTER_EDITORIAL_MODEL or preset["text"]
            test_models = [effective_text]
            for fb in (preset.get("text_fallbacks") or []):
                if fb not in test_models:
                    test_models.append(fb)
            if preset.get("text_fallback") and preset["text_fallback"] not in test_models:
                test_models.append(preset["text_fallback"])

            model_ok = False
            model_error = ""
            tested_model = test_models[0]
            for test_model in test_models:
                tested_model = test_model
                try:
                    test_resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={**headers, "Content-Type": "application/json"},
                        json={
                            "model": test_model,
                            "messages": [{"role": "user", "content": "Say OK"}],
                            "max_tokens": 5,
                        },
                        timeout=20.0,
                    )
                    if test_resp.status_code == 200:
                        model_ok = True
                        break
                    else:
                        try:
                            err = test_resp.json()
                            model_error = err.get("error", {}).get("message", test_resp.text[:200])
                        except Exception:
                            model_error = test_resp.text[:200]
                        logger.info(f"Model test failed for {test_model}: {model_error}")
                except httpx.TimeoutException:
                    model_error = f"Timeout testing {test_model}"
                    logger.info(model_error)

            effective_vision = settings.OPENROUTER_PRIMARY_MODEL or preset["vision"]
            effective_summary = settings.OPENROUTER_SUMMARY_MODEL or effective_text
            return {
                "status": "connected" if model_ok else "key_valid_model_error",
                "message": "API key validated and model responded successfully." if model_ok
                    else f"Key is valid but model test failed: {model_error}. Try refreshing models to find available ones.",
                "preset": preset_name,
                "vision_model": effective_vision,
                "summary_model": effective_summary,
                "text_model": effective_text,
                "tested_model": tested_model,
                "model_test_passed": model_ok,
                **key_info,
            }

    except httpx.TimeoutException:
        return {"status": "timeout", "message": "Connection to OpenRouter timed out. Try again."}
    except Exception as e:
        logger.warning(f"OpenRouter test failed: {e}")
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_ollama():
    from backend.services import ollama_registry
    try:
        active = await ollama_registry.pick_host()
        if active is None:
            return {
                "status": "offline",
                "message": (f"Cannot reach any Ollama host "
                            f"(primary: {settings.OLLAMA_HOST})"),
                "hosts": await ollama_registry.registry_status(),
            }
        status = await ollama_registry.probe(active, force=True)
        if not status.online:
            return {
                "status": "offline",
                "message": (f"Cannot reach Ollama at {active.url}: "
                            f"{status.error or 'no response'}"),
            }
        models = status.models

        has_vision = any(
            "moondream" in m or "llava" in m or "bakllava" in m
            for m in models
        )
        return {
            "status": "connected",
            "message": (f"Ollama connected with {len(models)} model(s) loaded "
                        f"(host: {active.name})."),
            "models": models,
            "has_primary_model": has_vision,
            "host": active.url,
            "host_name": active.name,
            "hosts": await ollama_registry.registry_status(),
        }
    except Exception as e:
        return {
            "status": "offline",
            "message": f"Cannot reach Ollama at {settings.OLLAMA_HOST}: {str(e)[:200]}",
        }


async def _test_anthropic():
    key = settings.ANTHROPIC_API_KEY
    if not _key_is_set(key):
        return {"status": "not_configured", "message": "ANTHROPIC_API_KEY is not set."}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "Say OK"}],
                },
            )
            if resp.status_code == 200:
                return {"status": "connected", "message": "Anthropic API key is valid."}
            elif resp.status_code == 401:
                return {"status": "invalid_key", "message": "API key is invalid."}
            else:
                return {"status": "error", "message": f"Anthropic responded with status {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_gemini():
    key = settings.GEMINI_API_KEY
    if not _key_is_set(key):
        return {"status": "not_configured", "message": "GEMINI_API_KEY is not set."}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={key}",
                json={"contents": [{"parts": [{"text": "Say OK"}]}]},
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                return {"status": "connected", "message": "Gemini API key is valid."}
            elif resp.status_code == 400 and "API_KEY_INVALID" in resp.text:
                return {"status": "invalid_key", "message": "API key is invalid."}
            else:
                return {"status": "error", "message": f"Gemini responded with status {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_groq():
    key = settings.GROQ_API_KEY
    if not _key_is_set(key):
        return {"status": "not_configured", "message": "GROQ_API_KEY is not set."}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": "Say OK"}],
                    "max_tokens": 5,
                },
            )
            if resp.status_code == 200:
                return {"status": "connected", "message": "Groq API key is valid."}
            elif resp.status_code == 401:
                return {"status": "invalid_key", "message": "API key is invalid."}
            else:
                return {"status": "error", "message": f"Groq responded with status {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)[:200]}"}


async def _test_replicate():
    """Test Replicate API connectivity by hitting their /v1/models endpoint."""
    key = settings.REPLICATE_API_KEY
    if not _key_is_set(key):
        return {"status": "not_configured", "message": "No Replicate API key set"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Hit the model version endpoint to verify key + model access
            model_id = settings.REPLICATE_MODEL
            resp = await client.get(
                f"https://api.replicate.com/v1/models/{model_id}",
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                desc = data.get("description", "")[:80]
                return {
                    "status": "connected",
                    "message": f"Replicate connected — model: {model_id} — {desc}",
                }
            elif resp.status_code == 401:
                return {"status": "invalid_key", "message": "Invalid Replicate API token"}
            elif resp.status_code == 404:
                return {
                    "status": "error",
                    "message": f"Model '{model_id}' not found on Replicate. Check the model ID.",
                }
            else:
                return {"status": "error", "message": f"Replicate returned HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Cannot reach Replicate API: {e}"}


async def _test_huggingface():
    token = settings.HF_AUTH_TOKEN
    if not token or not token.strip():
        return {
            "status": "not_configured",
            "message": "HF_AUTH_TOKEN is not set. Add your HuggingFace access token to enable pyannote speaker diarization.",
            "help": "Get a free token at https://huggingface.co/settings/tokens",
        }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token.strip()}"},
            )
            if resp.status_code == 200:
                username = resp.json().get("name", "unknown")
                model_resp = await client.get(
                    "https://huggingface.co/api/models/pyannote/speaker-diarization-3.1",
                    headers={"Authorization": f"Bearer {token.strip()}"},
                )
                if model_resp.status_code == 200:
                    return {
                        "status": "connected",
                        "message": f"Connected as '{username}'. pyannote model access confirmed — neural speaker diarization is enabled.",
                    }
                elif model_resp.status_code == 403:
                    return {
                        "status": "connected",
                        "message": f"Connected as '{username}', but you must accept the pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1 and click 'Agree and access repository'.",
                    }
                else:
                    return {
                        "status": "connected",
                        "message": f"Connected as '{username}'. Could not verify pyannote model access (HTTP {model_resp.status_code}).",
                    }
            elif resp.status_code == 401:
                return {"status": "invalid_key", "message": "Token is invalid or expired."}
            else:
                return {"status": "error", "message": f"HuggingFace API returned HTTP {resp.status_code}."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to reach HuggingFace API: {e}"}


class SaveKeyRequest(BaseModel):
    provider: str
    key: str


def _invalidate_status_cache():
    global _status_cache, _status_cache_ts
    _status_cache = {}
    _status_cache_ts = 0


@router.post("/providers/key")
async def save_provider_key(req: SaveKeyRequest):
    """Save an API key to .env and hot-reload settings.

    Keys are persisted to two locations for redundancy:
    1. user_settings.json on the Docker volume mount (survives container recreate)
    2. .env file inside the container (survives container restart)
    """
    env_var = _PROVIDER_KEY_ENV.get(req.provider)
    if not env_var:
        return {"status": "error", "message": f"Unknown provider: {req.provider}"}

    key_val = req.key.strip()
    if not key_val:
        return {"status": "error", "message": "Key cannot be empty"}

    # Update the settings object in memory
    setattr(settings, env_var, key_val)
    _invalidate_status_cache()

    # If the HuggingFace token changed, reload the diarization pipeline
    if env_var == "HF_AUTH_TOKEN":
        try:
            from backend.services.compat_stubs import reload_diarization
            reload_diarization()
            logger.info("Reloading pyannote diarization pipeline with new HF token")
        except Exception as e:
            logger.warning("Failed to reload diarization pipeline: %s", e)

    # Persist to .env file (backup)
    env_path = _find_env_file()
    if env_path:
        _upsert_env_var(env_path, env_var, key_val)

    # Persist to user_settings.json (primary — on volume mount)
    persisted = _persist_user_settings()
    result = {"status": "saved", "provider": req.provider}
    if not persisted:
        result["warning"] = (
            "Key is active in memory but could not be saved to disk. "
            "It may not survive a container restart."
        )
    return result


class SavePresetRequest(BaseModel):
    preset: str
    vision_model: str = ""
    text_model: str = ""
    summary_model: str = ""


@router.post("/providers/preset")
async def save_preset(req: SavePresetRequest):
    """Save the active preset (and optional custom models) to settings.

    When a known preset is selected (free/efficient/balanced/premium),
    the model IDs in settings are updated to match the preset's defaults.
    This ensures OpenRouterProvider always reads the correct models from
    settings without needing to re-resolve the preset dict at init time.

    When custom models are provided (req.vision_model, req.text_model),
    those override the preset defaults.
    """
    from backend.services.providers.openrouter_provider import PRESETS as _PRESETS

    settings.OPENROUTER_PRESET = req.preset

    # Resolve effective model IDs: explicit overrides > preset defaults
    preset_dict = _PRESETS.get(req.preset, _PRESETS["free"])
    vision_model = req.vision_model or preset_dict["vision"]
    text_model = req.text_model or preset_dict["text"]
    summary_model = req.summary_model or preset_dict.get("summary", text_model)

    settings.OPENROUTER_PRIMARY_MODEL = vision_model
    settings.OPENROUTER_EDITORIAL_MODEL = text_model
    settings.OPENROUTER_SUMMARY_MODEL = summary_model
    _invalidate_status_cache()

    env_path = _find_env_file()
    if env_path:
        _upsert_env_var(env_path, "OPENROUTER_PRESET", req.preset)
        _upsert_env_var(env_path, "OPENROUTER_PRIMARY_MODEL", vision_model)
        _upsert_env_var(env_path, "OPENROUTER_EDITORIAL_MODEL", text_model)
        _upsert_env_var(env_path, "OPENROUTER_SUMMARY_MODEL", summary_model)

    _persist_user_settings()
    logger.info(
        "Preset saved: %s (vision=%s, text=%s, summary=%s)",
        req.preset, vision_model, text_model, summary_model,
    )
    return {"status": "saved", "preset": req.preset}


class ToggleOllamaRequest(BaseModel):
    enabled: bool


def _pre_download_whisper_model(model_name: str):
    """Pre-download a Whisper model in a background thread.

    faster-whisper downloads models from HuggingFace on first use. Without
    pre-downloading, the first transcription attempt triggers a download
    inside the subprocess, which can timeout and fail. This ensures the
    model is cached locally before the user starts a video analysis.
    """
    import threading

    def _do_download():
        try:
            logger.info("Whisper pre-download: downloading '%s' from HuggingFace...", model_name)
            # Import and instantiate on CPU with int8 — minimal resources,
            # just triggers the HuggingFace download to cache
            from faster_whisper import WhisperModel
            m = WhisperModel(model_name, device="cpu", compute_type="int8")
            del m
            import gc
            gc.collect()
            logger.info("Whisper pre-download: '%s' is now cached locally", model_name)
        except Exception as e:
            logger.warning("Whisper pre-download failed for '%s': %s", model_name, e)

    t = threading.Thread(target=_do_download, daemon=True, name=f"whisper-download-{model_name}")
    t.start()


# Progress state for the manual "Pull models" button (one pull run at a time).
# Polled by GET /providers/ollama/pull-status so the UI can show progress and
# reload the dropdown when the pull finishes. ``hosts`` carries per-target
# (container + Companion) progress so downloads to each GPU show separately.
_ollama_pull_state: dict = {
    "active": False, "models": [], "hosts": [],
    # Legacy aggregate fields kept for older consumers.
    "done": [], "failed": [], "current": None, "progress": {},
}


def _pull_ollama_models_background(models: list[str] | None = None):
    """Pull Ollama models onto EVERY enabled host in parallel — the local
    container Ollama AND a paired GPU Companion — in a background thread, with
    per-host progress. Called when Ollama is toggled on or models are saved so
    the models are ready wherever a job might run.
    """
    from backend.services import ollama_registry

    if models is None:
        models = []
        for m in (settings.OLLAMA_PRIMARY_MODEL, settings.OLLAMA_EDITORIAL_MODEL,
                  settings.OLLAMA_TRANSLATION_MODEL):
            if m and m not in models:
                models.append(m)
    if not models:
        return

    hosts = ollama_registry.enabled_hosts()
    if not hosts:
        legacy = (getattr(settings, "OLLAMA_HOST", "") or "").strip()
        if legacy:
            hosts = [ollama_registry.OllamaHost(id="default", name="Local Ollama", url=legacy)]
    # De-dup by normalized URL so a host listed twice isn't pulled twice.
    seen, uniq = set(), []
    for h in hosts:
        if h.url and h.url not in seen:
            seen.add(h.url)
            uniq.append(h)
    hosts = uniq
    if not hosts:
        return

    host_states = []
    for h in hosts:
        host_states.append({
            "name": h.name,
            "gpu_name": h.gpu_name,
            "is_companion": bool(h.is_companion),
            "is_local": ollama_registry.is_local_gpu_host(h.url),
            "url": h.url,
            "current": None,
            "progress": {},
            "done": [],
            "failed": [],
        })
    _ollama_pull_state.update({
        "active": True, "models": list(models), "hosts": host_states,
        "done": [], "failed": [], "current": None, "progress": {},
    })

    def _pull_one_host(host, hs):
        import httpx as _httpx
        import json as _json
        headers = ollama_registry.auth_headers(host)
        label = hs["gpu_name"] or hs["name"]
        # What's already installed on this host, so a model whose equivalent is
        # present (e.g. a friendly "Qwen2.5-14B-Instruct" id whose real tag
        # qwen2.5:14b is pulled) is skipped instead of failing an /api/pull for a
        # non-registry tag.
        installed_here: list = []
        try:
            _tr = _httpx.get(ollama_registry.join_url(host.url, "/api/tags"),
                             headers=headers, timeout=10)
            if _tr.status_code == 200:
                installed_here = [m.get("name", "") for m in (_tr.json() or {}).get("models", [])]
        except Exception:
            installed_here = []
        for model in models:
            hs["current"] = model
            hs["progress"][model] = 0.0
            inst_tag = ollama_registry.resolve_installed_tag(installed_here, model)
            if inst_tag:
                hs["progress"][model] = 100.0
                hs["done"].append(model)
                logger.info("Pull %s → %s: already installed as %s (skipped)",
                            model, label, inst_tag)
                continue
            # Not installed here — pull a REAL registry tag. A friendly display
            # id ("Qwen2.5-14B-Instruct") is rebuilt to "qwen2.5:14b" so the
            # pull hits a real tag instead of 404-ing on the display name.
            pull_tag = ollama_registry.canonical_pull_tag(model)
            try:
                logger.info("Pull %s (as %s) → %s ...", model, pull_tag, label)
                with _httpx.stream(
                    "POST", ollama_registry.join_url(host.url, "/api/pull"),
                    json={"name": pull_tag, "stream": True},
                    headers=headers,
                    timeout=_httpx.Timeout(connect=10, read=1800, write=10, pool=10),
                ) as resp:
                    if resp.status_code != 200:
                        logger.warning("Pull %s → %s: HTTP %d", model, label, resp.status_code)
                        hs["failed"].append(model)
                        continue
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                        except Exception:
                            continue
                        if obj.get("error"):
                            raise RuntimeError(obj["error"])
                        total = obj.get("total") or 0
                        completed = obj.get("completed") or 0
                        if total:
                            hs["progress"][model] = round(completed / total * 100, 1)
                hs["progress"][model] = 100.0
                hs["done"].append(model)
                logger.info("Pull %s → %s: ready", model, label)
            except Exception as exc:
                logger.warning("Pull %s → %s failed (%s)", model, label, exc)
                hs["failed"].append(model)
        hs["current"] = None

    def _run():
        threads = []
        for host, hs in zip(hosts, host_states):
            t = threading.Thread(target=_pull_one_host, args=(host, hs),
                                 daemon=True, name=f"pull-{hs['name']}")
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        _ollama_pull_state["active"] = False

    threading.Thread(target=_run, daemon=True, name="ollama-bg-pull").start()


class ToggleStrictRemoteRequest(BaseModel):
    enabled: bool


@router.post("/providers/ollama/strict-remote")
async def toggle_strict_remote(req: ToggleStrictRemoteRequest):
    """Enable/disable 'use only the remote GPU'. When on, Ollama never falls
    back to the local-GPU daemon and remote Whisper isn't bypassed by a flaky
    health probe — the AI stays on the Companion (or fails over to the cloud)."""
    settings.GPU_STRICT_REMOTE = bool(req.enabled)
    _persist_user_settings()
    _invalidate_status_cache()
    return {"status": "saved", "gpu_strict_remote": settings.GPU_STRICT_REMOTE}


@router.post("/providers/ollama/toggle")
async def toggle_ollama(req: ToggleOllamaRequest):
    """Add or remove Ollama from the fallback chain."""
    chain = [p.strip() for p in settings.AI_FALLBACK_CHAIN.split(",") if p.strip()]
    if req.enabled:
        if "ollama" not in chain:
            chain.insert(0, "ollama")  # Ollama goes FIRST — user wants to use local models
    else:
        chain = [p for p in chain if p != "ollama"]
    settings.AI_FALLBACK_CHAIN = ",".join(chain)
    _invalidate_status_cache()

    env_path = _find_env_file()
    if env_path:
        _upsert_env_var(env_path, "AI_FALLBACK_CHAIN", settings.AI_FALLBACK_CHAIN)

    _persist_user_settings()

    # When Ollama is enabled, pull configured models in the background
    # so they're ready when the user needs them.  The startup pull only
    # fires if Ollama was already in the chain at boot time.
    if req.enabled:
        _pull_ollama_models_background()

    return {
        "status": "saved",
        "ollama_enabled": "ollama" in chain,
        "chain": chain,
    }


class PullOllamaRequest(BaseModel):
    # A single tag, or a list of tags. When both are omitted the configured
    # primary/editorial/translation models are pulled.
    model: Optional[str] = None
    models: Optional[list[str]] = None


# ── Companion VRAM: remote read/set + model-fit pre-check ────────────────────
# The paired Companion exposes GET/POST /v1/config/vram (v0.5.0+). ClipAI proxies
# it so the user can flip auto-allocate + drag the budget from ClipAI's own UI,
# and uses the card total to tell the user up front when a chosen model can't fit
# — instead of leaving the model picker stuck on "pulling…" forever.

def _estimate_model_vram_gb(tag: str, size_gb: float | None = None,
                            param_b: float | None = None,
                            quant: str | None = None) -> float:
    """Rough VRAM (GB) a model needs to load GPU-resident. On-disk size is the
    best proxy for the weights; add ~0.8 GB runtime/KV overhead. With no size,
    estimate from parameter count × a per-billion factor by quant. Returns 0.0
    when nothing is known (caller then skips the fit gate). Mirrors the
    frontend ``estimateModelVramGb`` so UI and API agree."""
    try:
        if size_gb and float(size_gb) > 0:
            return round(float(size_gb) + 0.8, 1)
        b = float(param_b) if param_b else 0.0
        if b <= 0:
            m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", (tag or "").lower())
            b = float(m.group(1)) if m else 0.0
        if b <= 0:
            return 0.0
        q = (quant or "").lower()
        if "q2" in q or "q3" in q:
            per = 0.45
        elif "q5" in q:
            per = 0.75
        elif "q6" in q:
            per = 0.9
        elif "q8" in q or "int8" in q:
            per = 1.1
        elif "f16" in q or "fp16" in q or "bf16" in q or "f32" in q:
            per = 2.1
        else:  # q4 / k-quants / unknown → assume a 4-bit quant
            per = 0.62
        return round(b * per + 0.8, 1)
    except Exception:
        return 0.0


async def _companion_vram_snapshot() -> dict:
    """(effective_budget_gb, total_gb, gpu label, host) for the paired Companion
    from its /v1/health, or zeros when none is reachable."""
    from backend.services import ollama_registry as oreg
    h = oreg.companion_host()
    if h is None:
        return {"host": None, "budget_gb": 0.0, "total_gb": 0.0, "gpu": ""}
    base = oreg.companion_base(h)
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.get(oreg.join_url(base, "/v1/health"), headers=oreg.auth_headers(h))
            if r.status_code == 200:
                j = r.json()
                return {
                    "host": h,
                    "budget_gb": round(float(j.get("vram_budget_gb") or 0.0), 1),
                    "total_gb": round(float(j.get("vram_total_mb") or 0.0) / 1024.0, 1),
                    "gpu": j.get("gpu_name") or getattr(h, "name", "") or "the Companion GPU",
                }
    except Exception as e:
        logger.debug("companion vram snapshot failed: %s", e)
    return {"host": h, "budget_gb": 0.0, "total_gb": 0.0,
            "gpu": getattr(h, "name", "") or "the Companion GPU"}


class CompanionVramRequest(BaseModel):
    vram_auto: bool | None = None
    vram_budget_gb: float | None = None
    vram_buffer_gb: float | None = None


@router.get("/providers/companion/vram")
async def get_companion_vram():
    """Current VRAM controls on the paired Companion (auto toggle + manual
    budget + buffer + card total), for ClipAI's slider. Falls back to /v1/health
    for an older Companion that lacks the dedicated config route."""
    from backend.services import ollama_registry as oreg
    h = oreg.companion_host()
    if h is None:
        return {"ok": False, "error": "No GPU Companion is paired."}
    base = oreg.companion_base(h)
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.get(oreg.join_url(base, "/v1/config/vram"), headers=oreg.auth_headers(h))
            if r.status_code == 200:
                return {"ok": True, "companion": getattr(h, "name", ""), "writable": True, **r.json()}
            if r.status_code == 404:
                rh = await c.get(oreg.join_url(base, "/v1/health"), headers=oreg.auth_headers(h))
                if rh.status_code == 200:
                    j = rh.json()
                    return {
                        "ok": True, "companion": getattr(h, "name", ""), "writable": False,
                        "vram_auto": j.get("vram_auto"),
                        "vram_budget_manual_gb": j.get("vram_budget_manual_gb"),
                        "vram_buffer_gb": j.get("vram_buffer_gb"),
                        "vram_total_mb": j.get("vram_total_mb"),
                        "vram_free_mb": j.get("vram_free_mb"),
                        "effective_budget_gb": j.get("vram_budget_gb"),
                        "note": "Update the GPU Companion app to change VRAM remotely.",
                    }
            return {"ok": False, "error": f"Companion returned HTTP {r.status_code}."}
    except Exception as e:
        return {"ok": False, "error": f"Couldn't reach the Companion: {e}"}


@router.post("/providers/companion/vram")
async def set_companion_vram(req: CompanionVramRequest):
    """Remotely set the Companion's auto-allocate + VRAM budget. Returns the
    fresh effective budget so the UI can immediately re-check model fit."""
    from backend.services import ollama_registry as oreg
    h = oreg.companion_host()
    if h is None:
        return {"ok": False, "error": "No GPU Companion is paired."}
    payload: dict = {}
    if req.vram_auto is not None:
        payload["vram_auto"] = bool(req.vram_auto)
    if req.vram_budget_gb is not None:
        payload["vram_budget_gb"] = max(0.0, float(req.vram_budget_gb))
    if req.vram_buffer_gb is not None:
        payload["vram_buffer_gb"] = max(0.0, float(req.vram_buffer_gb))
    if not payload:
        return {"ok": False, "error": "Nothing to change."}
    base = oreg.companion_base(h)
    try:
        async with httpx.AsyncClient(timeout=25.0) as c:
            r = await c.post(oreg.join_url(base, "/v1/config/vram"),
                             headers=oreg.auth_headers(h), json=payload)
            if r.status_code == 200:
                return {"ok": True, "companion": getattr(h, "name", ""), **r.json()}
            if r.status_code == 404:
                return {"ok": False, "error": "This GPU Companion is too old to set "
                        "VRAM remotely — update the Companion app."}
            return {"ok": False, "error": f"Companion returned HTTP {r.status_code}."}
    except Exception as e:
        return {"ok": False, "error": f"Couldn't reach the Companion: {e}"}


# The paired Companion exposes GET/POST /v1/config/quality (v0.11.9+): the
# desktop app's "Performance" control — the Ollama speed profile (pipeline
# parallelism) and the Whisper transcription quality (beam search + model) —
# mirrored over the LAN so the user can drive it from ClipAI's Settings
# without walking to the GPU PC.

_COMPANION_SPEED_PROFILES = {"auto", "eco", "balanced", "turbo"}
_COMPANION_WHISPER_QUALITIES = {"auto", "fast", "balanced", "max"}


class CompanionQualityRequest(BaseModel):
    speed_profile: str | None = None
    whisper_quality: str | None = None


@router.get("/providers/companion/quality")
async def get_companion_quality():
    """Current performance/quality settings on the paired Companion (speed
    profile + whisper quality + what they resolve to on that GPU), for
    ClipAI's remote control. Falls back to /v1/health (read-only) for an
    older Companion that lacks the dedicated config route."""
    from backend.services import ollama_registry as oreg
    h = oreg.companion_host()
    if h is None:
        return {"ok": False, "error": "No GPU Companion is paired."}
    base = oreg.companion_base(h)
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.get(oreg.join_url(base, "/v1/config/quality"),
                            headers=oreg.auth_headers(h))
            if r.status_code == 200:
                return {"ok": True, "companion": getattr(h, "name", ""), "writable": True, **r.json()}
            if r.status_code == 404:
                rh = await c.get(oreg.join_url(base, "/v1/health"), headers=oreg.auth_headers(h))
                if rh.status_code == 200:
                    j = rh.json()
                    return {
                        "ok": True, "companion": getattr(h, "name", ""), "writable": False,
                        "speed_profile": j.get("speed_profile"),
                        "whisper_quality": j.get("whisper_quality"),
                        "num_parallel": j.get("num_parallel"),
                        "max_loaded_models": j.get("max_loaded_models"),
                        "effective_budget_gb": j.get("vram_budget_gb"),
                        "whisper_effective": {
                            "model": j.get("whisper_model_effective"),
                            "beam_size": j.get("whisper_beam_size"),
                            "beam_search": (j.get("whisper_beam_size") or 0) > 1,
                        },
                        "note": "Update the GPU Companion app to change quality remotely.",
                    }
            return {"ok": False, "error": f"Companion returned HTTP {r.status_code}."}
    except Exception as e:
        return {"ok": False, "error": f"Couldn't reach the Companion: {e}"}


@router.post("/providers/companion/quality")
async def set_companion_quality(req: CompanionQualityRequest):
    """Remotely set the Companion's speed profile / transcription quality.
    The Companion applies them exactly like its local GUI (Ollama restart on
    a speed change, whisper sidecar drop on a quality change) and returns the
    fresh effective decode so the UI can show what actually engaged."""
    from backend.services import ollama_registry as oreg
    h = oreg.companion_host()
    if h is None:
        return {"ok": False, "error": "No GPU Companion is paired."}
    payload: dict = {}
    if req.speed_profile is not None:
        v = req.speed_profile.strip().lower()
        if v not in _COMPANION_SPEED_PROFILES:
            return {"ok": False, "error": f"Invalid speed profile {v!r} "
                    f"(one of: {', '.join(sorted(_COMPANION_SPEED_PROFILES))})."}
        payload["speed_profile"] = v
    if req.whisper_quality is not None:
        v = req.whisper_quality.strip().lower()
        if v not in _COMPANION_WHISPER_QUALITIES:
            return {"ok": False, "error": f"Invalid whisper quality {v!r} "
                    f"(one of: {', '.join(sorted(_COMPANION_WHISPER_QUALITIES))})."}
        payload["whisper_quality"] = v
    if not payload:
        return {"ok": False, "error": "Nothing to change."}
    base = oreg.companion_base(h)
    try:
        async with httpx.AsyncClient(timeout=25.0) as c:
            r = await c.post(oreg.join_url(base, "/v1/config/quality"),
                             headers=oreg.auth_headers(h), json=payload)
            if r.status_code == 200:
                return {"ok": True, "companion": getattr(h, "name", ""), **r.json()}
            if r.status_code == 404:
                return {"ok": False, "error": "This GPU Companion is too old to set "
                        "quality remotely — update the Companion app."}
            return {"ok": False, "error": f"Companion returned HTTP {r.status_code}."}
    except Exception as e:
        return {"ok": False, "error": f"Couldn't reach the Companion: {e}"}


@router.post("/providers/ollama/pull")
async def pull_ollama_models(req: PullOllamaRequest | None = None):
    """Kick off a background pull of one or more Ollama models.

    With no body, pulls the configured primary/editorial/translation models
    (e.g. the qwen3 translation default). Returns immediately — the pull runs in
    a background thread; poll GET /providers/ollama/pull-status for progress.
    Never auto-pulls anything the caller didn't ask for, and refuses to start a
    second run while one is active."""
    if _ollama_pull_state.get("active"):
        return {
            "status": "already_running",
            "message": "A model pull is already in progress.",
            **_ollama_pull_state,
        }
    host = (settings.OLLAMA_HOST or "").strip()
    if not host:
        return {"status": "error", "message": "No Ollama host configured."}

    models: list[str] = []
    if req and req.models:
        models = [m.strip() for m in req.models if m and m.strip()]
    elif req and req.model and req.model.strip():
        models = [req.model.strip()]
    else:
        for m in (settings.OLLAMA_PRIMARY_MODEL, settings.OLLAMA_EDITORIAL_MODEL,
                  settings.OLLAMA_TRANSLATION_MODEL):
            if m and m not in models:
                models.append(m)
    if not models:
        return {"status": "error", "message": "No models to pull."}

    # Fit pre-check against the paired Companion's GPU. A model bigger than the
    # whole card would download but never load onto the GPU — which is what
    # leaves the picker stuck on "pulling…". Refuse it up front with a clear
    # message. A model that fits the card but exceeds the CURRENT budget still
    # pulls (downloading needs no VRAM) but we warn the user to raise the budget
    # (or enable auto-allocate) so it loads on the GPU instead of spilling to CPU.
    snap = await _companion_vram_snapshot()
    total_gb, budget_gb, gpu = snap["total_gb"], snap["budget_gb"], snap["gpu"]
    warnings: list[str] = []
    for m in models:
        need = _estimate_model_vram_gb(m)
        if need <= 0 or total_gb <= 0:
            continue
        if need > total_gb + 0.5:
            return {
                "status": "wont_fit",
                "model": m,
                "need_gb": need,
                "total_gb": total_gb,
                "budget_gb": budget_gb,
                "gpu": gpu,
                "message": (f"{m} needs about {need:.0f} GB of VRAM, but {gpu} only "
                            f"has {total_gb:.0f} GB — it can't run on this GPU. "
                            f"Pick a smaller model."),
            }
        if budget_gb > 0 and need > budget_gb + 0.5:
            warnings.append(
                f"{m} needs about {need:.0f} GB, but the Companion VRAM budget is "
                f"{budget_gb:.0f} GB. Raise the budget (or turn on auto-allocate) so "
                f"it loads on the GPU instead of spilling to CPU.")

    _pull_ollama_models_background(models)
    msg = "Pulling " + ", ".join(models) + " in the background."
    if warnings:
        msg += " " + " ".join(warnings)
    return {
        "status": "started",
        "models": models,
        "warnings": warnings,
        "message": msg,
    }


@router.get("/providers/ollama/pull-status")
async def ollama_pull_status():
    """Report progress of the most recent /providers/ollama/pull run so the UI
    can show per-target progress bars and reload the model list when it
    finishes."""
    st = dict(_ollama_pull_state)
    hosts = st.get("hosts") or []
    n_models = len(st.get("models") or [])
    if hosts:
        st["total"] = n_models * len(hosts)
        st["finished"] = sum(len(h.get("done") or []) + len(h.get("failed") or [])
                             for h in hosts)
        st["current"] = next((h.get("current") for h in hosts if h.get("current")), None)
    else:
        st["total"] = n_models
        st["finished"] = len(st.get("done") or []) + len(st.get("failed") or [])
    return st


def _find_env_file() -> str | None:
    """Find or create the .env file for persisting settings.

    Checks project root, cwd, and common Docker paths. If no .env file
    exists, creates one at the project root so API keys and settings
    can be written as a backup alongside user_settings.json.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(project_root, ".env"),
        ".env",
        "/app/.env",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # No .env found — create one at project root so subsequent writes succeed
    env_path = candidates[0]
    try:
        with open(env_path, "w") as f:
            f.write("# ClipAI settings (auto-created)\n")
        logger.info(f"Created new .env file at {env_path}")
        return env_path
    except Exception as e:
        logger.warning(f"Could not create .env at {env_path}: {e}")
        return None


def _upsert_env_var(env_path: str, var_name: str, value: str):
    """Update or add an env var in a .env file."""
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()

        pattern = re.compile(rf"^{re.escape(var_name)}\s*=")
        found = False
        new_lines = []
        for line in lines:
            if pattern.match(line):
                new_lines.append(f"{var_name}={value}\n")
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f"{var_name}={value}\n")

        with open(env_path, "w") as f:
            f.writelines(new_lines)
    except Exception as e:
        logger.warning(f"Failed to update .env: {e}")


@router.get("/providers/models/recommended")
async def recommended_models():
    """Dynamically discover vision-capable models from OpenRouter and build recommendations."""
    if not _key_is_set(settings.OPENROUTER_API_KEY):
        return {"models": [], "error": "OpenRouter API key not configured"}

    all_models = await _fetch_openrouter_models()
    if all_models is None:
        return {"models": [], "error": "Failed to fetch models from OpenRouter"}

    by_id = {m["id"]: m for m in all_models}

    # Dynamically discover free vision models
    free_vision = []
    free_text = []
    for m in all_models:
        mid = m.get("id", "")
        arch = m.get("architecture", {})
        modality = arch.get("modality", "")
        is_free = ":free" in mid or _is_zero_cost(m)
        has_vision = "image" in modality

        if is_free and has_vision:
            free_vision.append(m)
        if is_free:
            free_text.append(m)

    # Sort free vision by context length (larger = better, likely better model)
    free_vision.sort(key=lambda m: m.get("context_length", 0), reverse=True)
    free_text.sort(key=lambda m: m.get("context_length", 0), reverse=True)

    # Build dynamic free tier from discovered models
    best_free_vision = free_vision[0]["id"] if free_vision else None
    second_free_vision = free_vision[1]["id"] if len(free_vision) > 1 else None
    best_free_text = free_text[0]["id"] if free_text else None

    # Build recommendations: dynamic free tiers + static paid tiers
    curated = []

    # Add dynamic free vision combos
    if best_free_vision and best_free_text:
        curated.append({
            "id": "free-best",
            "label": _short_name(by_id.get(best_free_vision, {})) + " (Free)",
            "desc": "Best available free vision model",
            "vision": best_free_vision,
            "summary": best_free_text,
            "text": best_free_text,
            "tier": "free",
        })
    if second_free_vision and best_free_text:
        curated.append({
            "id": "free-alt",
            "label": _short_name(by_id.get(second_free_vision, {})) + " (Free)",
            "desc": "Alternative free vision model",
            "vision": second_free_vision,
            "summary": best_free_text,
            "text": best_free_text,
            "tier": "free",
        })

    # Add more free vision options (up to 4 total free combos)
    for i, fv in enumerate(free_vision[2:6], start=3):
        curated.append({
            "id": f"free-{i}",
            "label": _short_name(fv) + " (Free)",
            "desc": f"Free vision model #{i}",
            "vision": fv["id"],
            "summary": best_free_text or fv["id"],
            "text": best_free_text or fv["id"],
            "tier": "free",
        })

    # Static paid tiers (these are stable model IDs unlikely to vanish)
    paid_combos = [
        {
            "id": "efficient",
            "label": "Gemini 2.5 Flash",
            "desc": "Fastest paid model — great value",
            "vision": "google/gemini-2.5-flash",
            "summary": "google/gemini-2.5-flash",
            "text": "google/gemini-2.5-flash",
            "tier": "efficient",
        },
        {
            "id": "balanced",
            "label": "Flash + Gemini Pro",
            "desc": "Fast vision & summary, smart clip detection",
            "vision": "google/gemini-2.5-flash",
            "summary": "google/gemini-2.5-flash",
            "text": "google/gemini-2.5-pro",
            "tier": "balanced",
        },
        {
            "id": "premium",
            "label": "Gemini Pro + Claude",
            "desc": "Best quality, highest accuracy",
            "vision": "google/gemini-2.5-pro",
            "summary": "google/gemini-2.5-flash",
            "text": "anthropic/claude-sonnet-4",
            "tier": "premium",
        },
        {
            "id": "gemini-pro-full",
            "label": "Gemini 2.5 Pro (Full)",
            "desc": "Gemini Pro for everything",
            "vision": "google/gemini-2.5-pro",
            "summary": "google/gemini-2.5-pro",
            "text": "google/gemini-2.5-pro",
            "tier": "premium",
        },
    ]
    curated.extend(paid_combos)

    result = []
    for combo in curated:
        v_model = by_id.get(combo["vision"])
        s_model = by_id.get(combo.get("summary", combo["text"]))
        t_model = by_id.get(combo["text"])

        v_cost = _estimate_cost(v_model, "vision") if v_model else 0
        s_cost = _estimate_cost(s_model, "text") if s_model else 0
        t_cost = _estimate_cost(t_model, "text") if t_model else 0
        total_cost = v_cost + s_cost + t_cost

        summary_id = combo.get("summary", combo["text"])
        result.append({
            "id": combo["id"],
            "label": combo["label"],
            "desc": combo["desc"],
            "tier": combo["tier"],
            "vision_model": combo["vision"],
            "summary_model": summary_id,
            "text_model": combo["text"],
            "vision_model_name": v_model.get("name", combo["vision"]) if v_model else combo["vision"],
            "summary_model_name": s_model.get("name", summary_id) if s_model else summary_id,
            "text_model_name": t_model.get("name", combo["text"]) if t_model else combo["text"],
            "cost_per_10min": round(total_cost, 4),
            "cost_per_10min_display": f"${total_cost:.4f}" if total_cost > 0 else "FREE",
            "available": v_model is not None and t_model is not None,
        })

    # Count stats
    total_free_vision = len(free_vision)
    total_free_text = len(free_text)

    return {
        "models": result,
        "free_vision_count": total_free_vision,
        "free_text_count": total_free_text,
    }


# Curated local (Ollama) models per role, with an approximate VRAM footprint
# (GB) for a q4-ish quant + KV cache. Used to recommend models that fit the
# detected GPU. (tag, approx_vram_gb, why)
_LOCAL_MODEL_CATALOG: dict[str, list[tuple]] = {
    "primary": [  # vision / video-frame understanding
        ("moondream:1.8b", 2.0, "Tiny vision model — runs on 2-4 GB GPUs"),
        ("qwen2.5vl:3b", 4.0, "Qwen2.5-VL 3B — image/video-frame understanding"),
        ("llava:7b", 6.0, "LLaVA 7B — balanced vision quality"),
        ("qwen2.5vl:7b", 7.0, "Qwen2.5-VL 7B — best local video/vision"),
        ("llava:13b", 10.0, "LLaVA 13B — highest-quality local vision"),
    ],
    "editorial": [  # SEO, summaries, scoring, polish
        ("qwen2.5:3b-instruct", 3.0, "Fast editorial/SEO on small GPUs"),
        ("llama3.1:8b", 6.5, "Llama 3.1 8B — strong general text"),
        ("qwen2.5:7b-instruct", 6.0, "Balanced editorial quality"),
        ("qwen2.5:14b", 10.0, "Best local editorial quality"),
    ],
    "translation": [  # subtitle translation
        ("qwen2.5:3b-instruct", 3.0, "Fast subtitle translation"),
        ("qwen3:4b-instruct-2507-q4_K_M", 4.0, "Qwen3 4B — great multilingual"),
        ("qwen2.5:7b-instruct", 6.0, "Higher-quality translation"),
        ("gemma2:9b", 7.0, "Gemma 2 9B — strong multilingual"),
    ],
}
# The subtitle-polish role reuses the editorial catalog.
_LOCAL_MODEL_CATALOG["polish"] = _LOCAL_MODEL_CATALOG["editorial"]


async def _effective_local_vram_gb() -> tuple[float, str]:
    """Best local GPU to size recommendations for: a paired Companion's GPU
    when present (it becomes the primary host), else the container's own GPU.
    Returns (vram_gb, label). 0.0 when no GPU is detectable."""
    from backend.services import ollama_registry
    for h in ollama_registry.get_hosts():
        if h.is_companion and h.vram_total_mb:
            return round(h.vram_total_mb / 1024.0, 1), (h.gpu_name or h.name or "Companion GPU")
    try:
        from backend.services.clip_exporter import detect_gpu_capabilities
        info = await asyncio.to_thread(detect_gpu_capabilities, force_redetect=False)
        mb = int(info.get("vram_mb") or 0)
        if mb:
            return round(mb / 1024.0, 1), (info.get("gpu_name") or info.get("name") or "local GPU")
    except Exception as e:
        logger.debug("local GPU detection failed: %s", e)
    return 0.0, ""


@router.get("/providers/models/local-recommended")
async def local_recommended_models():
    """Recommended LOCAL (Ollama) models per role, sized to the detected GPU.

    Each entry is ready to select in a dropdown (``ollama/<tag>`` id) and
    marks whether it fits the GPU's VRAM and whether it's already installed.
    Selecting one and hitting Save downloads it to the container + Companion.
    """
    vram_gb, gpu_label = await _effective_local_vram_gb()

    installed: set[str] = set()
    try:
        from backend.services import ollama_registry
        for h in await ollama_registry.registry_status():
            for m in (h.get("models") or []):
                installed.add(str(m))
    except Exception:
        pass

    from backend.services import ollama_registry as _oreg
    headroom = 0.5  # allow a model to be recommended slightly above budget

    def _role(role: str) -> list[dict]:
        cat = _LOCAL_MODEL_CATALOG.get(role, [])
        fitting = [c for c in cat if (vram_gb <= 0 or c[1] <= vram_gb + headroom)]
        # The largest model that still fits is the headline recommendation.
        best_tag = max(fitting, key=lambda c: c[1])[0] if fitting else None
        out = []
        for tag, gb, why in cat:
            fits = (vram_gb <= 0) or (gb <= vram_gb + headroom)
            out.append({
                "id": f"ollama/{tag}",
                "tag": tag,
                "name": tag,
                "size_gb": gb,
                "why": why,
                "provider": "local",
                "is_free": True,
                "fits": fits,
                "recommended": tag == best_tag,
                "installed": _oreg.model_present(list(installed), tag),
            })
        # Fitting first, then by size descending (bigger = better within budget).
        out.sort(key=lambda m: (not m["fits"], -m["size_gb"]))
        return out

    return {
        "gpu": gpu_label,
        "vram_gb": vram_gb,
        "roles": {r: _role(r) for r in ("primary", "editorial", "translation", "polish")},
    }


@router.get("/providers/models/catalog")
async def models_catalog():
    """Searchable superset for the model dropdowns: the full local (Ollama)
    catalog per role + the full OpenRouter cloud model list. The UI shows the
    curated list by default and searches this when the user types."""
    vram_gb, gpu_label = await _effective_local_vram_gb()
    installed: set[str] = set()
    try:
        from backend.services import ollama_registry
        for h in await ollama_registry.registry_status():
            for m in (h.get("models") or []):
                installed.add(str(m))
    except Exception:
        pass
    from backend.services import ollama_registry as _oreg

    def _local(role: str) -> list[dict]:
        out = []
        for tag, gb, why in _LOCAL_MODEL_CATALOG.get(role, []):
            out.append({
                "id": f"ollama/{tag}", "tag": tag, "name": tag, "size_gb": gb,
                "why": why, "provider": "local",
                "fits": (vram_gb <= 0) or (gb <= vram_gb + 0.5),
                "installed": _oreg.model_present(list(installed), tag),
            })
        return out

    local = {r: _local(r) for r in ("primary", "editorial", "translation", "polish")}

    cloud: list[dict] = []
    if _key_is_set(settings.OPENROUTER_API_KEY):
        all_models = await _fetch_openrouter_models() or []
        for m in all_models:
            mid = m.get("id", "")
            if not mid:
                continue
            arch = m.get("architecture", {}) or {}
            cloud.append({
                "id": mid,
                "name": m.get("name", mid),
                "provider": "openrouter",
                "vision": "image" in (arch.get("modality", "") or ""),
                "is_free": ":free" in mid or _is_zero_cost(m),
            })
    return {"gpu": gpu_label, "vram_gb": vram_gb, "local": local, "cloud": cloud}


POLISH_BENCH_PATH = os.path.join(_DATA_DIR, "polish_benchmark_scores.json")


def _load_polish_scores() -> dict:
    try:
        with open(POLISH_BENCH_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_polish_scores(scores: dict) -> None:
    try:
        with open(POLISH_BENCH_PATH, "w") as f:
            json.dump(scores, f, indent=2)
    except Exception as e:
        logger.warning("Could not persist polish benchmark scores: %s", e)


@router.get("/providers/models/recommended/subtitle-polish")
async def recommended_subtitle_polish_models():
    """Recommended models for the transcript-polish role (audit Phase 4.2).

    Intersects the curated SUBTITLE_POLISH_SHORTLIST (ordered, data-only)
    with the live OpenRouter /models list — availability, current pricing
    and context length — and returns the top pick per tier with a
    one-line rationale and $/1M-token cost. Measured benchmark scores
    (see /providers/models/polish-benchmark) outrank the static ordering.
    """
    if not _key_is_set(settings.OPENROUTER_API_KEY):
        return {"models": [], "error": "OpenRouter API key not configured"}
    all_models = await _fetch_openrouter_models()
    if all_models is None:
        return {"models": [], "error": "Failed to fetch models from OpenRouter"}

    from backend.services.providers.openrouter_provider import (
        SUBTITLE_POLISH_SHORTLIST)
    by_id = {m["id"]: m for m in all_models}
    scores = _load_polish_scores()

    available = []
    for rank, entry in enumerate(SUBTITLE_POLISH_SHORTLIST):
        live = by_id.get(entry["id"])
        if not live:
            continue
        pricing = live.get("pricing", {}) or {}
        try:
            prompt_pm = float(pricing.get("prompt", "0")) * 1_000_000
            completion_pm = float(pricing.get("completion", "0")) * 1_000_000
        except (ValueError, TypeError):
            prompt_pm = completion_pm = 0.0
        bench = scores.get(entry["id"]) or {}
        available.append({
            "id": entry["id"],
            "name": live.get("name", entry["id"]),
            "tier": entry["tier"],
            "rationale": entry["rationale"],
            "context_length": live.get("context_length", 0),
            "prompt_cost_per_1m": round(prompt_pm, 3),
            "completion_cost_per_1m": round(completion_pm, 3),
            "cost_display": ("FREE" if prompt_pm == 0 and completion_pm == 0
                             else f"${prompt_pm:.2f} / ${completion_pm:.2f} per 1M tok"),
            "benchmark": bench or None,
            "_rank": rank,
        })

    # Top pick per tier: measured benchmark winners outrank static order.
    def _sort_key(m):
        b = m.get("benchmark") or {}
        # exact_fix_rate in [0,1]; measured models sort above unmeasured
        return (-(b.get("exact_fix_rate", -1)), m["_rank"])

    top_per_tier = {}
    for tier in ("free", "efficient", "premium"):
        tier_models = sorted(
            (m for m in available if m["tier"] == tier), key=_sort_key)
        if tier_models:
            top_per_tier[tier] = tier_models[0]["id"]

    for m in available:
        m["recommended"] = top_per_tier.get(m["tier"]) == m["id"]
        m.pop("_rank", None)

    return {
        "models": available,
        "top_per_tier": top_per_tier,
        "selected": (getattr(settings, "SUBTITLE_POLISH_MODEL", "") or None),
    }


@router.post("/providers/models/polish-benchmark")
async def run_polish_benchmark(payload: dict):
    """Run the built-in ~20-segment polish benchmark against one model.

    Body: {"model": "<openrouter id>"}. Scores exact-fix rate and format
    compliance via transcript_polisher's own prompt, persists the result
    so the recommendation endpoint can prefer measured winners.
    """
    model_id = (payload or {}).get("model", "").strip()
    if not model_id:
        return {"error": "model is required"}
    try:
        from backend.services.polish_benchmark import run_benchmark
        result = await run_benchmark(model_id)
    except Exception as e:
        logger.warning("Polish benchmark failed for %s: %s", model_id, e)
        return {"error": f"Benchmark failed: {e}", "model": model_id}
    if result.get("error"):
        return result
    scores = _load_polish_scores()
    scores[model_id] = {
        "exact_fix_rate": result["exact_fix_rate"],
        "format_compliance": result["format_compliance"],
        "cases": result["cases"],
        "timestamp": int(time.time()),
    }
    _save_polish_scores(scores)
    return result


def _short_name(model_data: dict) -> str:
    """Extract a short display name from a model's full name."""
    name = model_data.get("name", model_data.get("id", "Unknown"))
    # Remove common suffixes/prefixes for brevity
    for remove in ["(free)", "(Free)", ":free"]:
        name = name.replace(remove, "").strip()
    return name


def _is_zero_cost(model_data: dict) -> bool:
    """Check if a model has zero pricing."""
    pricing = model_data.get("pricing", {})
    try:
        prompt = float(pricing.get("prompt", "1"))
        completion = float(pricing.get("completion", "1"))
        return prompt == 0 and completion == 0
    except (ValueError, TypeError):
        return False


# Ollama vision-capable families (name-prefix match), shared by the model
# browser + the available-models list so both agree on which local models can
# do vision.
_OLLAMA_VISION_FAMILIES = {
    "llava", "moondream", "bakllava", "minicpm-v", "llava-llama3", "llava-phi3",
    "nanollava", "llama3.2-vision", "qwen2.5vl", "qwen2-vl", "qwen2.5-vl",
    "gemma3", "mistral-small3.1",
}


def _ollama_family_is_vision(model_name: str) -> bool:
    fam = (model_name or "").split(":")[0].lower()
    return any(vf in fam for vf in _OLLAMA_VISION_FAMILIES)


async def _ollama_models_all_hosts(timeout: float = 5.0) -> list[dict]:
    """Union of installed Ollama models across ALL enabled registry hosts — the
    local card AND any paired Companion — so a model freshly pulled on the
    Companion is searchable even when it isn't the primary host. (The old path
    queried only ``primary_url()``, hiding companion-only pulls.)

    Each returned dict is a raw ``/api/tags`` model entry plus ``_hosts`` (the
    host names that have it). Fully fail-soft: an offline / unauthorized host is
    skipped. Hosts are probed concurrently and results merged by model name.
    """
    from backend.services import ollama_registry
    hosts = ollama_registry.enabled_hosts()
    if not hosts:
        _u = (ollama_registry.primary_url() or settings.OLLAMA_HOST or "").rstrip("/")
        if not _u:
            return []
        hosts = [ollama_registry.OllamaHost(id="default", name="Default", url=_u)]

    merged: dict[str, dict] = {}

    async def _one(host):
        url = ollama_registry.join_url(host.url, "/api/tags")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=ollama_registry.auth_headers(host))
            if resp.status_code != 200:
                return
            for m in resp.json().get("models", []) or []:
                nm = m.get("name", "")
                if not nm:
                    continue
                existing = merged.get(nm)
                if existing is not None:
                    if host.name not in existing.get("_hosts", []):
                        existing.setdefault("_hosts", []).append(host.name)
                else:
                    m["_hosts"] = [host.name]
                    merged[nm] = m
        except Exception as e:
            logger.debug("Ollama /api/tags fetch failed for host %s (%s)", host.name, e)

    await asyncio.gather(*[_one(h) for h in hosts], return_exceptions=True)
    return list(merged.values())


def _ollama_model_browser_entries(raw_models: list[dict]) -> tuple[list, list]:
    """Shape cross-host Ollama ``/api/tags`` dicts into ModelBrowser rows.

    Returns ``(vision_entries, text_entries)`` in the same shape the OpenRouter
    cache uses (``id``/``name``/``context_length``/``pricing``), so the browser
    can search local models alongside cloud ones. All Ollama models can do text;
    vision families also appear in the vision list."""
    vision, text = [], []
    for m in raw_models:
        name = m.get("name", "")
        if not name:
            continue
        details = m.get("details", {}) or {}
        param_size = details.get("parameter_size", "")
        quant = details.get("quantization_level", "")
        size_gb = round((m.get("size", 0) or 0) / (1024 ** 3), 1)
        hosts = m.get("_hosts") or []
        label_bits = [b for b in (param_size, quant, f"{size_gb}GB" if size_gb else "") if b]
        host_bit = f" · {', '.join(hosts)}" if hosts else ""
        entry = {
            "id": f"ollama/{name}",
            "name": f"{name} (Ollama{host_bit})"
                    + (f" — {' '.join(label_bits)}" if label_bits else ""),
            "provider": "ollama",
            "context_length": 0,   # unknown; browser allows ctx==0
            "pricing": {"prompt": "0", "completion": "0"},
            "is_free": True,
            "_hosts": hosts,
        }
        if _ollama_family_is_vision(name):
            vision.append(entry)
        text.append(entry)
    return vision, text


async def _fetch_openrouter_models() -> list | None:
    """Fetch model list from OpenRouter, using cache if fresh."""
    # Check cache first
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            with open(MODEL_CACHE_PATH, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < MODEL_CACHE_TTL:
                return cache.get("raw_models", [])
        except Exception:
            pass

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
    except Exception as e:
        logger.warning(f"Failed to fetch OpenRouter models: {e}")
        return None

    # Save to cache (include raw models for reuse)
    _save_model_cache(models)
    return models


def _save_model_cache(models: list):
    """Save raw model list to cache file."""
    vision_models = []
    text_models = []
    for m in models:
        model_id = m.get("id", "")
        top_provider = m.get("top_provider", {}) or {}
        model_info = {
            "id": model_id,
            "name": m.get("name", model_id),
            "pricing": m.get("pricing", {}),
            "context_length": m.get("context_length", 0),
            "max_completion_tokens": top_provider.get("max_completion_tokens", 0),
        }
        architecture = m.get("architecture", {})
        modality = architecture.get("modality", "")
        input_modalities = architecture.get("input_modalities", [])
        has_vision = "image" in str(modality).lower() or "image" in [
            str(x).lower() for x in input_modalities
        ]
        if has_vision:
            vision_models.append(model_info)
        text_models.append(model_info)

    cache_data = {
        "timestamp": time.time(),
        "raw_models": models,
        "data": {
            "vision_models": vision_models,
            "text_models": text_models,
            "cached": True,
        },
    }
    os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
    try:
        with open(MODEL_CACHE_PATH, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Failed to save model cache: {e}")


def _estimate_cost(model_data: dict, role: str) -> float:
    """Estimate cost for a 10-minute video based on model pricing."""
    pricing = model_data.get("pricing", {})
    try:
        prompt_price = float(pricing.get("prompt", "0"))  # per token
        completion_price = float(pricing.get("completion", "0"))  # per token
    except (ValueError, TypeError):
        return 0.0

    if role == "vision":
        return (prompt_price * _VISION_INPUT_TOKENS_10MIN +
                completion_price * _VISION_OUTPUT_TOKENS_10MIN)
    else:
        return (prompt_price * _TEXT_INPUT_TOKENS_10MIN +
                completion_price * _TEXT_OUTPUT_TOKENS_10MIN)


async def _openrouter_browser_data() -> dict:
    """The OpenRouter half of the model-browser list (cached 24h)."""
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            with open(MODEL_CACHE_PATH, "r") as f:
                cache = json.load(f)
            if time.time() - cache.get("timestamp", 0) < MODEL_CACHE_TTL:
                cached_data = cache.get("data", {})
                if cached_data:
                    return dict(cached_data)
        except Exception:
            pass
    if not _key_is_set(settings.OPENROUTER_API_KEY):
        return {"vision_models": [], "text_models": [], "cached": False}
    models = await _fetch_openrouter_models()
    if models is None:
        return {"vision_models": [], "text_models": [], "cached": False}
    try:
        with open(MODEL_CACHE_PATH, "r") as f:
            cache = json.load(f)
        return dict(cache.get("data", {"vision_models": [], "text_models": [], "cached": True}))
    except Exception:
        return {"vision_models": [], "text_models": [], "cached": False}


@router.get("/providers/models")
async def list_models():
    """Searchable model list for the browser: the live OpenRouter catalog
    (cached 24h — click Refresh to force) MERGED with local Ollama models pulled
    across ALL registry hosts (the local card AND any paired Companion), so
    newly-added models of either kind are searchable. Ollama models are queried
    fresh each call (no cache), so a just-pulled Companion model appears at once.
    """
    data = await _openrouter_browser_data()
    vision = list(data.get("vision_models", []) or [])
    text = list(data.get("text_models", []) or [])

    # Merge cross-host Ollama models (fresh, uncached) so a Companion pull shows.
    if "ollama" in settings.active_provider_chain:
        try:
            _raw = await _ollama_models_all_hosts()
            _ov, _ot = _ollama_model_browser_entries(_raw)
            _seen_v = {m.get("id") for m in vision}
            _seen_t = {m.get("id") for m in text}
            # Local models first — they're free + immediately usable.
            vision = [m for m in _ov if m.get("id") not in _seen_v] + vision
            text = [m for m in _ot if m.get("id") not in _seen_t] + text
        except Exception as e:
            logger.debug("model browser: Ollama merge skipped (%s)", e)

    return {
        "vision_models": vision,
        "text_models": text,
        "primary_models": vision,
        "editorial_models": text,
        "cached": bool(data.get("cached", False)),
    }


@router.post("/providers/models/refresh")
async def refresh_models():
    """Force-refresh the model list from OpenRouter (clears cache)."""
    # Delete cache to force re-fetch
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            os.remove(MODEL_CACHE_PATH)
        except Exception:
            pass

    if not _key_is_set(settings.OPENROUTER_API_KEY):
        return {"status": "error", "message": "OpenRouter API key not configured"}

    models = await _fetch_openrouter_models()
    if models is None:
        return {"status": "error", "message": "Failed to fetch models from OpenRouter"}

    # Count vision models
    vision_count = 0
    free_vision_count = 0
    for m in models:
        arch = m.get("architecture", {})
        modality = arch.get("modality", "")
        if "image" in modality:
            vision_count += 1
            if ":free" in m.get("id", "") or _is_zero_cost(m):
                free_vision_count += 1

    return {
        "status": "refreshed",
        "total_models": len(models),
        "vision_models": vision_count,
        "free_primary_models": free_vision_count,
        "message": f"Loaded {len(models)} models ({vision_count} with vision, {free_vision_count} free vision)",
    }


# ── Per-Task Model Selection ──────────────────────────────────────

# Whisper model options (always available locally, free)
_WHISPER_MODELS = [
    {"id": "tiny", "name": "Whisper Tiny", "provider": "local", "desc": "Fastest, least accurate (~39M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 1, "quality": "poor"},
    {"id": "base", "name": "Whisper Base", "provider": "local", "desc": "Fast, low accuracy (~74M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 2, "quality": "basic"},
    {"id": "small", "name": "Whisper Small", "provider": "local", "desc": "Good accuracy/speed balance (~244M params, default)", "cost_per_hour": 0, "is_free": True, "quality_score": 3, "quality": "good"},
    {"id": "medium", "name": "Whisper Medium", "provider": "local", "desc": "High accuracy, slower (~769M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 4, "quality": "excellent"},
    {"id": "large-v3", "name": "Whisper Large V3", "provider": "local", "desc": "Best accuracy, needs GPU (~1.5B params)", "cost_per_hour": 0, "is_free": True, "quality_score": 5, "quality": "best"},
    {"id": "large-v3-turbo", "name": "Whisper Large V3 Turbo", "provider": "local", "desc": "Near large-v3 accuracy, 40% faster (~809M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 5, "quality": "best"},
    {"id": "kotoba-tech/kotoba-whisper-v2.0-faster", "name": "Kotoba-Whisper v2.0 (Japanese)", "provider": "local", "desc": "Japanese-specialized Whisper (distilled, ~756M) — often beats large-v3 on clean Japanese; fits a 4 GB GPU. Best for Japanese source audio; downloads from HuggingFace on first use.", "cost_per_hour": 0, "is_free": True, "quality_score": 5, "quality": "best"},
    {"id": "distil-large-v3", "name": "Whisper Distil Large V3", "provider": "local", "desc": "Distilled large-v3, 6x faster, English-optimized (~756M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 4, "quality": "excellent"},
    {"id": "distil-large-v3.5", "name": "Whisper Distil Large V3.5", "provider": "local", "desc": "Distilled large-v3.5 — near large-v3 English WER at ~2x turbo speed; downloads CTranslate2 weights from HuggingFace on first use (falls back to distil-large-v3 if unavailable)", "cost_per_hour": 0, "is_free": True, "quality_score": 4, "quality": "excellent"},
    {"id": "distil-small.en", "name": "Whisper Distil Small (English-only)", "provider": "local", "desc": "Distilled small, English-only — very fast, light on VRAM (~166M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 3, "quality": "good", "english_only": True},
    {"id": "distil-medium.en", "name": "Whisper Distil Medium (English-only)", "provider": "local", "desc": "Distilled medium, English-only — fast, fits a 4 GB GPU (~394M params)", "cost_per_hour": 0, "is_free": True, "quality_score": 4, "quality": "excellent", "english_only": True},
]

# Known models for direct providers (when user has their API key)
# "created" = approximate release Unix timestamp so sorting by recency works
_ANTHROPIC_MODELS = [
    {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "provider": "anthropic", "context_length": 200000, "vision": True, "created": 1747872000, "quality_score": 4, "quality": "excellent"},
    {"id": "anthropic/claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "anthropic", "context_length": 200000, "vision": True, "created": 1727740800, "quality_score": 3, "quality": "good"},
]

_GEMINI_MODELS = [
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "gemini", "context_length": 1000000, "vision": True, "created": 1744502400, "quality_score": 3, "quality": "good"},
    {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "gemini", "context_length": 1000000, "vision": True, "created": 1742860800, "quality_score": 4, "quality": "excellent"},
    {"id": "google/gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "gemini", "context_length": 1000000, "vision": True, "created": 1738886400, "quality_score": 3, "quality": "good"},
]


def _cost_per_hour(model_data: dict, role: str) -> float:
    """Estimate cost for a 1-hour video based on model pricing."""
    return _estimate_cost(model_data, role) * 6  # 6 × 10-min segments


# ── Vision model compatibility for subject tracking ──
# Models need sufficient context for at least 1 image + prompt + output.
_MIN_VISION_CONTEXT = 16000  # Absolute minimum context length

# Models known to be incompatible with subject tracking despite having vision.
_VISION_BLOCKLIST_PATTERNS = [
    "reka/reka-edge",      # 16K context, ~5800 tok/img, can't fit 2 images, 94% default to center
    "reka-edge",           # Catch any variant
    "firellava",           # Poor JSON compliance, no spatial reasoning
    "llava:7b",            # Too small for reliable subject_x
    "nanollava",           # Too small for structured output
]

# ── Subject tracking quality scores ──
# Scoring criteria: can the model output structured JSON with a reliable
# subject_x (0-100) horizontal position? Most 7B+ vision models can.
#   5 = best:  Excellent spatial reasoning + JSON + large context + fast
#   4 = excellent: Strong spatial + reliable JSON + good context
#   3 = good:  Solid spatial awareness + JSON works + adequate context
#   2 = basic: Can estimate positions + JSON mostly works
#   1 = minimal: Marginal capability
_VISION_TRACKING_SCORES: dict[str, int] = {
    # ── Google ──
    "gemini-2.5-pro": 5, "gemini-2.5-flash": 5, "gemini-2.0-flash": 4,
    "gemini-2.5-flash-lite": 3, "gemini-flash": 4, "gemini-3": 5,
    # ── Anthropic ──
    "claude-sonnet-4": 5, "claude-opus": 5, "claude-sonnet": 4, "claude-haiku": 3,
    # ── OpenAI ──
    "gpt-4o": 4, "gpt-4o-mini": 3, "gpt-4-turbo": 4, "o1": 4, "o3": 5, "o4-mini": 4,
    # ── Qwen VL ──
    "qwen2.5-vl-72b": 4, "qwen2.5-vl-32b": 4, "qwen2.5-vl-7b": 3, "qwen2.5-vl-3b": 2,
    "qwen3-vl-32b": 4, "qwen3-vl-8b": 3, "qwen-vl-max": 4, "qwen-vl-plus": 3,
    "qwen-vl": 3, "qwq": 3,
    # ── Mistral / Pixtral ──
    "pixtral-large": 4, "pixtral-12b": 3, "pixtral": 3,
    "mistral-large-3": 4, "mistral-small-3.1": 3, "mistral-small-3": 3, "mistral-medium": 3,
    # ── Meta Llama ──
    "llama-4-maverick": 4, "llama-4-scout": 4,
    "llama-3.2-90b": 4, "llama-3.2-11b": 3, "llama-3.2-3b": 2,
    # ── Google Gemma ──
    "gemma-3-27b": 3, "gemma-3-12b": 3, "gemma-3-4b": 2, "gemma-2": 2,
    # ── DeepSeek ──
    "deepseek-r1": 3, "deepseek-v3": 3, "deepseek-vl2": 3, "deepseek": 3,
    # ── Moonshot / Kimi ──
    "kimi-vl": 3, "kimi-k2.5": 4, "kimi-k2": 3, "moonshot": 3,
    # ── NVIDIA ──
    "nemotron": 3, "nemotron-nano-2-vl": 3, "llama-3.1-nemotron": 3,
    # ── xAI Grok ──
    "grok-3": 4, "grok-2": 3, "grok-4": 4, "grok": 3,
    # ── Zhipu / GLM ──
    "glm-4v": 3, "glm-4.5": 3, "chatglm": 2,
    # ── InternLM / InternVL ──
    "internvl": 3, "internlm": 3,
    # ── Yi (01.AI) ──
    "yi-vision": 3, "yi-vl": 3,
    # ── MiniCPM ──
    "minicpm-v": 3, "minicpm": 2,
    # ── StepFun ──
    "step-3.5": 3,
    # ── Reka (only Core; Edge is blocklisted) ──
    "reka-core": 2,
    # ── Moondream ──
    "moondream": 2,
    # ── Cohere ──
    "command-r-plus": 3, "command-r": 2,
    # ── MiMo ──
    "mimo": 3,
    # ── Microsoft Phi ──
    "phi-4": 3, "phi-3.5-vision": 2, "phi-3-vision": 2,
}


def _vision_tracking_compat(model_id: str, context_length: int) -> tuple[bool, int]:
    """Check if a vision model is compatible with subject tracking.

    Returns (is_compatible, tracking_quality_score 0-5).
    Uses longest pattern match for specificity (e.g. "qwen2.5-vl-72b" wins over "qwen-vl").
    """
    mid_lower = model_id.lower()

    for pattern in _VISION_BLOCKLIST_PATTERNS:
        if pattern in mid_lower:
            return False, 0

    # Skip context check for Ollama local models (context_length=0 means unknown)
    if context_length > 0 and context_length < _MIN_VISION_CONTEXT:
        return False, 0

    # Longest match wins for specificity
    best_score = None
    best_len = 0
    for pattern, score in _VISION_TRACKING_SCORES.items():
        if pattern in mid_lower and len(pattern) > best_len:
            best_score = score
            best_len = len(pattern)
    if best_score is not None:
        return True, best_score

    # Context-based fallback for unknown models — most modern vision models
    # with decent context can do basic spatial estimation.
    if context_length == 0:
        return True, 2  # Ollama local, give benefit of doubt
    if context_length >= 200000:
        return True, 4  # 200K+ → likely capable frontier model
    if context_length >= 32000:
        return True, 3  # 32K+ → modern model, should work well
    if context_length >= 16000:
        return True, 2  # 16K → tight but workable
    return True, 1  # Below 16K but passed blocklist — marginal


def _effective_whisper_info() -> dict:
    """Report the configured vs actually-loaded Whisper model.

    Reads the ``AudioIntelligence`` class attrs WITHOUT importing the heavy
    reframer_audio module (cv2 / torch) when it isn't already loaded — so the
    Settings page can poll this cheaply. ``whisper_model_effective`` is the
    model currently resident in VRAM/RAM, falling back to the last model that
    loaded this process (sticky across the post-job VRAM release); it is
    ``None`` only when nothing has loaded since startup. ``whisper_downgraded``
    is True when the effective model differs from the selected one (the GTX
    1650 auto-downgrade, e.g. large-v3-turbo → medium, or the base fallback).
    """
    import sys as _sys
    selected = getattr(settings, "WHISPER_MODEL", "small")
    user_set = bool(getattr(settings, "WHISPER_MODEL_USER_SET", False))
    effective = None
    loaded_now = False
    mod = _sys.modules.get("backend.services.reframer_audio")
    ai = getattr(mod, "AudioIntelligence", None) if mod else None
    if ai is not None:
        cached = getattr(ai, "_cached_model_name", None)
        last = getattr(ai, "_last_loaded_model_name", None)
        loaded_now = bool(cached)
        effective = cached or last
    return {
        "whisper_model_selected": selected,
        "whisper_model_user_set": user_set,
        "whisper_model_effective": effective,
        "whisper_model_loaded": loaded_now,
        "whisper_downgraded": bool(effective and effective != selected),
    }


@router.get("/providers/whisper/effective")
async def get_whisper_effective():
    """Configured vs actually-loaded Whisper model — cheap, safe to poll.

    Lets the Settings page always show the model that REALLY ran (including a
    silent low-VRAM downgrade) instead of only the requested value."""
    return _effective_whisper_info()


def _current_translation_model() -> str:
    """Resolve the translation-model id for the UI dropdown, provider-aware.

    Mirrors how ``current_vision`` / ``current_text`` are resolved in
    ``available_models`` so an Ollama translation pick reads back as
    ``ollama/<model>`` instead of blank. Returns ``''`` to mean "reuse the
    editorial model" — the dropdown's default option.

    ``OLLAMA_TRANSLATION_MODEL`` has a non-empty default (``qwen2.5:3b``) in
    config.py, so it is surfaced ONLY when the Ollama path is the active one.
    This preserves the existing "blank == reuse editorial" semantics for
    OpenRouter-only users (who must never see the Ollama default leak in).
    """
    chain = settings.active_provider_chain
    ollama_is_primary = bool(chain and chain[0] == "ollama")
    ollama_active = ollama_is_primary or (
        "ollama" in chain and not _key_is_set(settings.OPENROUTER_API_KEY)
    )
    ollama_tm = (settings.OLLAMA_TRANSLATION_MODEL or "").strip()
    openrouter_tm = (settings.OPENROUTER_TRANSLATION_MODEL or "").strip()

    # On the Ollama path, surface the dedicated Ollama translation model
    # prefixed so the dropdown selects the matching "ollama/<model>" option.
    if ollama_active and ollama_tm:
        return f"ollama/{ollama_tm}"
    # Otherwise prefer an explicit OpenRouter translation pick. On the
    # OpenRouter path the Ollama default is intentionally NOT surfaced, keeping
    # "blank == reuse editorial" intact for OpenRouter-only users.
    if openrouter_tm:
        return openrouter_tm
    return ""


# Model-id markers for reasoning / "thinking" models. Their chain-of-thought
# output breaks the strict JSON-array the batch translator parses, so they are
# unusable for subtitle translation (the '…-1.2b-thinking:free' the user hit).
_REASONING_ID_MARKERS = (
    "thinking", "reasoning", "deepseek-r1", "qwq", ":r1", "-r1-", "/r1",
)


def _is_translation_capable(model_id: str, output_modalities=None) -> bool:
    """True when a model is usable for SUBTITLE TRANSLATION.

    OpenRouter's catalog is all text-I/O chat models, so the classes that
    cannot be used for the batch JSON translation task — and that should not
    appear in the Translation AI dropdown — are:
      * generators that also OUTPUT image/audio (not a text-translation tool), and
      * reasoning / 'thinking' models, whose chain-of-thought breaks the strict
        JSON-array the translator parses.
    Editorial polishing keeps the full text list; this only governs translation.
    """
    mid = (model_id or "").lower()
    if output_modalities:
        om = [str(x).lower() for x in output_modalities]
        if "image" in om or "audio" in om:
            return False
    if any(tok in mid for tok in _REASONING_ID_MARKERS):
        return False
    leaf = mid.rsplit("/", 1)[-1]
    if leaf in ("o1", "o3") or leaf.startswith(("o1-", "o3-", "o4-")):
        return False
    return True


@router.get("/providers/models/available")
async def available_models():
    """Return all available models grouped by task (transcript, vision, text).
    Each list is sorted: free/cheapest first. Filtered by capability."""

    transcript = list(_WHISPER_MODELS)
    vision = []
    text = []
    # Translation AI dropdown: only models that can actually translate
    # (text-output chat models, no reasoning/'thinking' or image/audio gens).
    translation = []

    # Always add the OpenRouter free auto-router at the top
    if _key_is_set(settings.OPENROUTER_API_KEY):
        _auto_speed = _estimate_speed("openrouter/free", "vision", True)
        _auto = {
            "id": "openrouter/free", "name": "Free Auto-Router",
            "provider": "openrouter", "cost_per_hour": 0, "is_free": True,
            "context_length": 0, "created": int(time.time()),
            "desc": "Auto-routes to best available free model",
            **_auto_speed,
        }
        vision.append(dict(_auto))
        _auto_text = {**_auto, **_estimate_speed("openrouter/free", "text", True)}
        text.append(_auto_text)
        translation.append(dict(_auto_text))

    # Fetch OpenRouter models
    all_models = await _fetch_openrouter_models() if _key_is_set(settings.OPENROUTER_API_KEY) else None

    if all_models:
        for m in all_models:
            mid = m.get("id", "")
            if mid == "openrouter/free":
                continue  # already added above
            name = m.get("name", mid)
            arch = m.get("architecture", {})
            modality = arch.get("modality", "")
            has_vision = "image" in modality
            is_free = ":free" in mid or _is_zero_cost(m)
            ctx = m.get("context_length", 0)
            created = m.get("created", 0)  # Unix timestamp

            entry_base = {
                "id": mid, "name": name, "provider": "openrouter",
                "is_free": is_free, "context_length": ctx, "created": created,
            }

            if has_vision:
                compatible, tracking_score = _vision_tracking_compat(mid, ctx)
                if not compatible:
                    continue  # Skip models that can't do subject tracking
                v_cost = _cost_per_hour(m, "vision")
                v_speed = _estimate_speed(mid, "vision", is_free)
                # Use tracking-specific score when it's higher
                if tracking_score > v_speed.get("quality_score", 0):
                    v_speed["quality_score"] = tracking_score
                    v_speed["quality"] = {1: "minimal", 2: "basic", 3: "good", 4: "excellent", 5: "best"}.get(tracking_score, "good")
                vision.append({**entry_base, "cost_per_hour": round(v_cost, 4),
                               "desc": f"{'FREE' if is_free else f'~${v_cost:.3f}/hr'} — {ctx:,} ctx",
                               "tracking_score": tracking_score,
                               **v_speed})

            t_cost = _cost_per_hour(m, "text")
            t_speed = _estimate_speed(mid, "text", is_free)
            _t_entry = {**entry_base, "cost_per_hour": round(t_cost, 4),
                        "desc": f"{'FREE' if is_free else f'~${t_cost:.3f}/hr'} — {ctx:,} ctx",
                        **t_speed}
            text.append(_t_entry)
            if _is_translation_capable(mid, arch.get("output_modalities")):
                translation.append(dict(_t_entry))

    # Add direct provider models if keys are set
    if _key_is_set(settings.ANTHROPIC_API_KEY):
        for m in _ANTHROPIC_MODELS:
            mid = m["id"]
            entry = {**m, "cost_per_hour": 0, "is_free": False,
                     "desc": f"Direct Anthropic API — {m['context_length']:,} ctx"}
            if m.get("vision"):
                vision.append({**entry, **_estimate_speed(mid, "vision", False)})
            _t = {**entry, **_estimate_speed(mid, "text", False)}
            text.append(_t)
            if _is_translation_capable(mid):
                translation.append(dict(_t))

    if _key_is_set(settings.GEMINI_API_KEY):
        for m in _GEMINI_MODELS:
            mid = m["id"]
            entry = {**m, "cost_per_hour": 0, "is_free": False,
                     "desc": f"Direct Gemini API — {m['context_length']:,} ctx"}
            if m.get("vision"):
                vision.append({**entry, **_estimate_speed(mid, "vision", False)})
            _t = {**entry, **_estimate_speed(mid, "text", False)}
            text.append(_t)
            if _is_translation_capable(mid):
                translation.append(dict(_t))

    # Add Ollama local models if Ollama is in the chain and reachable
    _VISION_FAMILIES = {"llava", "moondream", "bakllava", "minicpm-v", "llava-llama3", "llava-phi3", "nanollava"}
    if "ollama" in settings.active_provider_chain:
        _ollama_seen_ids: set[str] = set()
        try:
            from backend.services import ollama_registry  # noqa: F401  (used below)
            # Enumerate models across ALL enabled registry hosts (local + any
            # paired Companion), not just the primary — so a model pulled on the
            # Companion is selectable even when it isn't the primary host.
            for m in await _ollama_models_all_hosts():
                model_name = m.get("name", "")
                if not model_name:
                    continue
                model_family = model_name.split(":")[0].lower()
                has_vision = _ollama_family_is_vision(model_name)

                size_bytes = m.get("size", 0)
                size_gb = round(size_bytes / (1024**3), 1) if size_bytes else 0
                details = m.get("details", {}) or {}
                param_size = details.get("parameter_size", "")
                quant = details.get("quantization_level", "")
                _hosts = m.get("_hosts") or []

                desc_parts = ["LOCAL", "FREE"]
                if param_size:
                    desc_parts.append(param_size)
                if quant:
                    desc_parts.append(quant)
                if size_gb:
                    desc_parts.append(f"{size_gb}GB")
                if _hosts:
                    desc_parts.append("on " + ", ".join(_hosts))
                desc = " — ".join(desc_parts)

                _host_bit = f" · {', '.join(_hosts)}" if _hosts else ""
                entry = {
                    "id": f"ollama/{model_name}",
                    "name": f"{model_name} (Ollama{_host_bit})",
                    "provider": "ollama",
                    "is_free": True,
                    "cost_per_hour": 0,
                    "context_length": 0,
                    "created": int(time.time()),  # Sort to top as "newest"
                    "desc": desc,
                    "speed": "balanced",
                    "est_time_display": "varies by GPU",
                    "quality_score": 3,
                    "quality": "good",
                    # Surface the on-disk (≈VRAM weights) size + params/quant so
                    # the dropdown can show how much VRAM the model needs and
                    # whether it fits the user's GPU.
                    "size_gb": size_gb or 0,
                    "param_size": param_size or "",
                    "quant": quant or "",
                }

                _ollama_seen_ids.add(f"ollama/{model_name}")
                if has_vision:
                    compatible, tracking_score = _vision_tracking_compat(f"ollama/{model_name}", 0)
                    if compatible:
                        entry["tracking_score"] = tracking_score
                        if tracking_score > entry.get("quality_score", 0):
                            entry["quality_score"] = tracking_score
                            entry["quality"] = {1: "minimal", 2: "basic", 3: "good", 4: "excellent", 5: "best"}.get(tracking_score, "good")
                        vision.append(entry)
                # All models can do text
                text.append(entry)
                if _is_translation_capable(entry["id"]):
                    translation.append(dict(entry))
        except Exception as e:
            logger.warning("Failed to fetch Ollama models for available list: %s", e)

        # Always show the configured default models even if they haven't been
        # pulled yet (e.g. Ollama was just toggled on and pulls are in progress).
        # This lets the user select them in the dropdown immediately.
        _defaults = [
            (settings.OLLAMA_PRIMARY_MODEL, True),   # (model_name, is_vision)
            (settings.OLLAMA_EDITORIAL_MODEL, False),
            # The dedicated translation model must surface in the dropdown too —
            # otherwise a configured-but-not-yet-pulled translation model (e.g.
            # qwen3:4b-instruct-2507-q4_K_M) is invisible until it's pulled.
            (settings.OLLAMA_TRANSLATION_MODEL, False),
        ]
        # Installed model names (from /api/tags above) so a configured model that
        # is really PRESENT under a different spelling — a friendly
        # "Qwen2.5-14B-Instruct" or an -instruct/quant tag vs the installed
        # "qwen2.5:14b" — shows as READY instead of being stuck on "pulling…".
        _installed_names = [i[len("ollama/"):] for i in _ollama_seen_ids
                            if i.startswith("ollama/")]
        for _def_name, _def_is_vision in _defaults:
            if not _def_name:
                continue
            _def_id = f"ollama/{_def_name}"
            if _def_id in _ollama_seen_ids:
                continue  # Already listed from /api/tags
            try:
                _resolved = ollama_registry.resolve_installed_tag(_installed_names, _def_name)
            except Exception:
                _resolved = None
            _def_family = _def_name.split(":")[0].lower()
            _is_vision = _def_is_vision or any(vf in _def_family for vf in _VISION_FAMILIES)
            if _resolved:
                _def_label = f"{_def_name} (Ollama Local — installed as {_resolved})"
                _def_desc = f"LOCAL — FREE — installed as {_resolved}"
            else:
                _def_label = f"{_def_name} (Ollama Local — pulling...)"
                _def_desc = "LOCAL — FREE — downloading..."
            _def_entry = {
                "id": _def_id,
                "name": _def_label,
                "provider": "ollama",
                "is_free": True,
                "cost_per_hour": 0,
                "context_length": 0,
                "created": int(time.time()),
                "desc": _def_desc,
                "speed": "balanced",
                "est_time_display": "varies by GPU",
                "quality_score": 3,
                "quality": "good",
            }
            if _is_vision:
                vision.append(_def_entry)
            text.append(_def_entry)
            if _is_translation_capable(_def_id):
                translation.append(dict(_def_entry))
            _ollama_seen_ids.add(_def_id)

    # Sort: free first, then newer + cheaper towards the top
    # Within free models: newest first.  Within paid: newest first, then cheapest.
    def _sort_key(m):
        is_paid = 0 if m.get("is_free") else 1
        newest_first = -(m.get("created", 0))  # negate so newer = smaller = first
        cost = m.get("cost_per_hour", 999)
        return (is_paid, newest_first, cost)

    vision.sort(key=_sort_key)
    text.sort(key=_sort_key)
    translation.sort(key=_sort_key)

    # Limit to top 100 per category to avoid overwhelming the UI
    # Return current models based on which provider is primary.
    # If Ollama is first in chain OR is in the chain and has models configured,
    # return Ollama models so the UI shows what the user actually selected.
    chain = settings.active_provider_chain
    ollama_is_primary = chain and chain[0] == "ollama"
    ollama_models_set = (
        "ollama" in chain
        and settings.OLLAMA_PRIMARY_MODEL
        and settings.OLLAMA_EDITORIAL_MODEL
    )
    if ollama_is_primary or (ollama_models_set and not _key_is_set(settings.OPENROUTER_API_KEY)):
        current_vision = f"ollama/{settings.OLLAMA_PRIMARY_MODEL}"
        current_text = f"ollama/{settings.OLLAMA_EDITORIAL_MODEL}"
    else:
        current_vision = settings.OPENROUTER_PRIMARY_MODEL
        current_text = settings.OPENROUTER_EDITORIAL_MODEL

    return {
        "transcript": transcript,
        "vision": vision[:100],
        "text": text[:100],
        # Translation AI dropdown — text models minus reasoning/'thinking' and
        # image/audio generators that can't be used for subtitle translation.
        "translation": translation[:100],
        "current": {
            "transcript_model": settings.WHISPER_MODEL,
            "vision_model": current_vision,
            "text_model": current_text,
            # Dedicated subtitle-translation model. Blank means "use the
            # editorial model" — the frontend renders that as the default
            # option in the Translation AI dropdown. Resolved provider-aware
            # so an Ollama pick reads back as "ollama/<model>" (not blank),
            # exactly like vision_model / text_model above.
            "translation_model": _current_translation_model(),
            # Configured vs actually-loaded Whisper model so the Settings page
            # shows what really ran (incl. a low-VRAM downgrade), not only the
            # requested value.
            **_effective_whisper_info(),
        },
    }


class SaveModelsRequest(BaseModel):
    transcript_model: str = ""
    vision_model: str = ""
    text_model: str = ""
    # Dedicated OpenRouter subtitle-translation model (separate from the
    # editorial/text model used for transcript polishing). Optional so an
    # absent field (None) is left untouched while an explicit "" clears it
    # back to the editorial fallback.
    translation_model: Optional[str] = None


@router.post("/providers/models/save")
async def save_models(req: SaveModelsRequest):
    """Save per-task model selections to settings and .env."""
    env_path = _find_env_file()

    if req.transcript_model:
        old_model = settings.WHISPER_MODEL
        settings.WHISPER_MODEL = req.transcript_model
        settings.WHISPER_MODEL_USER_SET = True  # Mark as explicitly chosen by user
        if env_path:
            _upsert_env_var(env_path, "WHISPER_MODEL", req.transcript_model)
            _upsert_env_var(env_path, "WHISPER_MODEL_USER_SET", "true")
        # Force reload if model changed — without this, the cached
        # AudioIntelligence engine holds the OLD model and the next job reuses
        # it. Invalidate the class-level cache so the next transcription loads
        # the new selection without a container restart (Task 3). Guarded via
        # sys.modules so we don't trigger the heavy reframer_audio import here
        # if it isn't already loaded (nothing to invalidate in that case).
        if req.transcript_model != old_model:
            from backend.services.compat_stubs import reload_model as reload_whisper
            reload_whisper()
            import sys as _sys
            _ra = _sys.modules.get("backend.services.reframer_audio")
            _ai = getattr(_ra, "AudioIntelligence", None) if _ra else None
            if _ai is not None and hasattr(_ai, "invalidate_cache"):
                _ai.invalidate_cache(reason=f"model changed {old_model}→{req.transcript_model}")
            logger.info(
                "Whisper model changed: '%s' → '%s' — cache invalidated, "
                "triggering background download",
                old_model, req.transcript_model,
            )
            # Pre-download the new model in background so it's cached before
            # the user starts a video analysis. Without this, the first
            # transcription attempt downloads the model inside the subprocess,
            # which can timeout and fail.
            _pre_download_whisper_model(req.transcript_model)

    if req.vision_model:
        if req.vision_model.startswith("ollama/"):
            # Strip the "ollama/" prefix to get the raw model name
            ollama_model = req.vision_model[len("ollama/"):]
            settings.OLLAMA_PRIMARY_MODEL = ollama_model
            if env_path:
                _upsert_env_var(env_path, "OLLAMA_PRIMARY_MODEL", ollama_model)
        else:
            settings.OPENROUTER_PRIMARY_MODEL = req.vision_model
            settings.OPENROUTER_PRESET = "custom"
            if env_path:
                _upsert_env_var(env_path, "OPENROUTER_PRIMARY_MODEL", req.vision_model)
                _upsert_env_var(env_path, "OPENROUTER_PRESET", "custom")

    if req.text_model:
        if req.text_model.startswith("ollama/"):
            ollama_model = req.text_model[len("ollama/"):]
            settings.OLLAMA_EDITORIAL_MODEL = ollama_model
            if env_path:
                _upsert_env_var(env_path, "OLLAMA_EDITORIAL_MODEL", ollama_model)
        else:
            settings.OPENROUTER_EDITORIAL_MODEL = req.text_model
            settings.OPENROUTER_SUMMARY_MODEL = req.text_model
            settings.OPENROUTER_PRESET = "custom"
            if env_path:
                _upsert_env_var(env_path, "OPENROUTER_EDITORIAL_MODEL", req.text_model)
                _upsert_env_var(env_path, "OPENROUTER_SUMMARY_MODEL", req.text_model)
                _upsert_env_var(env_path, "OPENROUTER_PRESET", "custom")

    # Dedicated subtitle-translation model, kept distinct from the editorial
    # model so SEO/summaries and translation can run different local models.
    # An ``ollama/`` pick sets OLLAMA_TRANSLATION_MODEL (the local-mode picker);
    # anything else sets OPENROUTER_TRANSLATION_MODEL. An explicit "" clears the
    # OpenRouter pick back to "reuse editorial".
    if req.translation_model is not None:
        _tm = (req.translation_model or "").strip()
        if _tm.startswith("ollama/"):
            _tm = _tm[len("ollama/"):]
            settings.OLLAMA_TRANSLATION_MODEL = _tm
            if env_path:
                _upsert_env_var(env_path, "OLLAMA_TRANSLATION_MODEL", _tm)
        else:
            settings.OPENROUTER_TRANSLATION_MODEL = _tm
            if env_path:
                _upsert_env_var(env_path, "OPENROUTER_TRANSLATION_MODEL", _tm)

    # If the user selected Ollama models, ensure Ollama is in the fallback chain
    # so it actually gets used for analysis. Put it first since that's the user's intent.
    has_ollama_models = (
        (req.vision_model and req.vision_model.startswith("ollama/"))
        or (req.text_model and req.text_model.startswith("ollama/"))
    )
    has_openrouter_models = (
        (req.vision_model and not req.vision_model.startswith("ollama/") and req.vision_model)
        or (req.text_model and not req.text_model.startswith("ollama/") and req.text_model)
    )

    if has_ollama_models:
        chain = [p.strip() for p in settings.AI_FALLBACK_CHAIN.split(",") if p.strip()]
        if "ollama" not in chain:
            chain.insert(0, "ollama")
            settings.AI_FALLBACK_CHAIN = ",".join(chain)
            if env_path:
                _upsert_env_var(env_path, "AI_FALLBACK_CHAIN", settings.AI_FALLBACK_CHAIN)
            logger.info("Auto-enabled Ollama in fallback chain (user selected Ollama models)")
        elif chain[0] != "ollama":
            # Move Ollama to front — user clearly wants local models as primary
            chain = ["ollama"] + [p for p in chain if p != "ollama"]
            settings.AI_FALLBACK_CHAIN = ",".join(chain)
            if env_path:
                _upsert_env_var(env_path, "AI_FALLBACK_CHAIN", settings.AI_FALLBACK_CHAIN)
            logger.info("Moved Ollama to front of fallback chain (user selected Ollama models)")
    elif has_openrouter_models:
        # User selected OpenRouter models — ensure OpenRouter is in the chain
        # and move it to the front so it's the primary provider
        chain = [p.strip() for p in settings.AI_FALLBACK_CHAIN.split(",") if p.strip()]
        if "openrouter" not in chain:
            chain.insert(0, "openrouter")
            settings.AI_FALLBACK_CHAIN = ",".join(chain)
            if env_path:
                _upsert_env_var(env_path, "AI_FALLBACK_CHAIN", settings.AI_FALLBACK_CHAIN)
            logger.info("Auto-enabled OpenRouter in fallback chain (user selected OpenRouter models)")
        elif chain[0] != "openrouter":
            chain = ["openrouter"] + [p for p in chain if p != "openrouter"]
            settings.AI_FALLBACK_CHAIN = ",".join(chain)
            if env_path:
                _upsert_env_var(env_path, "AI_FALLBACK_CHAIN", settings.AI_FALLBACK_CHAIN)
            logger.info("Moved OpenRouter to front of fallback chain (user selected OpenRouter models)")

    _invalidate_status_cache()
    _persist_user_settings()

    # Pull every selected Ollama model (vision + editorial + translation) to
    # the active host in the background. When a GPU Companion is paired it IS
    # the primary host, so this downloads the models onto the desktop GPU and
    # /providers/ollama/pull-status reports per-model progress + readiness.
    pull_models = []
    for _sel in (req.vision_model, req.text_model, req.translation_model):
        if _sel and _sel.startswith("ollama/"):
            _m = _sel[len("ollama/"):].strip()
            if _m and _m not in pull_models:
                pull_models.append(_m)
    if pull_models:
        _pull_ollama_models_background(pull_models)

    # Return the currently active models.
    # If the user just saved Ollama models, reflect those regardless of chain order.
    # This prevents the UI from reverting to OpenRouter models when Ollama is enabled
    # but not the first provider in the chain.
    chain = settings.active_provider_chain
    has_ollama_models = (
        (req.vision_model and req.vision_model.startswith("ollama/"))
        or (req.text_model and req.text_model.startswith("ollama/"))
    )
    use_ollama = has_ollama_models or (chain and chain[0] == "ollama")
    if use_ollama:
        return {
            "status": "saved",
            "transcript_model": settings.WHISPER_MODEL,
            "vision_model": f"ollama/{settings.OLLAMA_PRIMARY_MODEL}",
            "text_model": f"ollama/{settings.OLLAMA_EDITORIAL_MODEL}",
            "translation_model": _current_translation_model(),
        }
    return {
        "status": "saved",
        "transcript_model": settings.WHISPER_MODEL,
        "vision_model": settings.OPENROUTER_PRIMARY_MODEL,
        "text_model": settings.OPENROUTER_EDITORIAL_MODEL,
        "translation_model": _current_translation_model(),
    }


# ── Transcription Settings ────────────────────────────────────────


class SaveTranscriptionSettingsRequest(BaseModel):
    beam_size: Optional[int] = None      # 1-5
    vad_filter: Optional[bool] = None
    frame_sample_rate: Optional[int] = None  # 5-30 seconds
    # ── Speech-coverage knobs (default-tuned for recall over precision) ──
    no_speech_threshold: Optional[float] = None       # 0.0-1.0
    gap_fill_enabled: Optional[bool] = None
    gap_fill_min_sec: Optional[float] = None          # 0.5-10.0
    gap_fill_no_speech_threshold: Optional[float] = None  # 0.0-1.0
    # Sentence-aware resegmentation toggle (Task 4).
    sentence_segmentation_enabled: Optional[bool] = None
    # Operator hint naming the show/film (e.g. "Mobile Suit Gundam Wing") so
    # mis-heard character/mecha names resolve to their official spellings.
    series_hint: Optional[str] = None
    # Reference transcript (YouTube captions) to conform the subtitle track to,
    # and how (adopt = words+timing+segmentation; timing = snap timing only).
    reference_subtitles: Optional[str] = None
    reference_mode: Optional[str] = None


@router.get("/transcription/settings")
async def get_transcription_settings():
    """Return current transcription speed/quality settings."""
    return {
        "whisper_model": settings.WHISPER_MODEL,
        "beam_size": settings.WHISPER_BEAM_SIZE,
        "vad_filter": settings.WHISPER_VAD_FILTER,
        "frame_sample_rate": settings.FRAME_SAMPLE_RATE,
        "no_speech_threshold": float(getattr(
            settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.4)),
        "gap_fill_enabled": bool(getattr(
            settings, "WHISPER_GAP_FILL_ENABLED", True)),
        "gap_fill_min_sec": float(getattr(
            settings, "WHISPER_GAP_FILL_MIN_SEC", 1.5)),
        "gap_fill_no_speech_threshold": float(getattr(
            settings, "WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD", 0.25)),
        "sentence_segmentation_enabled": bool(getattr(
            settings, "SENTENCE_SEGMENTATION_ENABLED", True)),
        "series_hint": str(getattr(settings, "TRANSLATION_SERIES_HINT", "") or ""),
        "reference_subtitles": str(getattr(settings, "TRANSLATION_REFERENCE_SUBTITLES", "") or ""),
        "reference_mode": str(getattr(settings, "TRANSLATION_REFERENCE_MODE", "adopt") or "adopt"),
    }


@router.post("/transcription/settings")
async def save_transcription_settings(req: SaveTranscriptionSettingsRequest):
    """Save transcription speed/quality settings."""
    env_path = _find_env_file()

    if req.beam_size is not None:
        clamped = max(1, min(5, req.beam_size))
        settings.WHISPER_BEAM_SIZE = clamped
        if env_path:
            _upsert_env_var(env_path, "WHISPER_BEAM_SIZE", str(clamped))

    if req.vad_filter is not None:
        settings.WHISPER_VAD_FILTER = req.vad_filter
        if env_path:
            _upsert_env_var(env_path, "WHISPER_VAD_FILTER", str(req.vad_filter))

    if req.frame_sample_rate is not None:
        clamped = max(5, min(30, req.frame_sample_rate))
        settings.FRAME_SAMPLE_RATE = clamped
        if env_path:
            _upsert_env_var(env_path, "FRAME_SAMPLE_RATE", str(clamped))

    if req.no_speech_threshold is not None:
        clamped = max(0.0, min(1.0, float(req.no_speech_threshold)))
        settings.WHISPER_NO_SPEECH_THRESHOLD = clamped
        if env_path:
            _upsert_env_var(env_path, "WHISPER_NO_SPEECH_THRESHOLD", str(clamped))

    if req.gap_fill_enabled is not None:
        settings.WHISPER_GAP_FILL_ENABLED = bool(req.gap_fill_enabled)
        if env_path:
            _upsert_env_var(env_path, "WHISPER_GAP_FILL_ENABLED",
                            str(bool(req.gap_fill_enabled)))

    if req.gap_fill_min_sec is not None:
        clamped = max(0.5, min(10.0, float(req.gap_fill_min_sec)))
        settings.WHISPER_GAP_FILL_MIN_SEC = clamped
        if env_path:
            _upsert_env_var(env_path, "WHISPER_GAP_FILL_MIN_SEC", str(clamped))

    if req.gap_fill_no_speech_threshold is not None:
        clamped = max(0.0, min(1.0, float(req.gap_fill_no_speech_threshold)))
        settings.WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD = clamped
        if env_path:
            _upsert_env_var(env_path, "WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD",
                            str(clamped))

    if req.sentence_segmentation_enabled is not None:
        settings.SENTENCE_SEGMENTATION_ENABLED = bool(req.sentence_segmentation_enabled)

    if req.series_hint is not None:
        settings.TRANSLATION_SERIES_HINT = str(req.series_hint).strip()[:200]

    if req.reference_subtitles is not None:
        # Bounded but roomy — a 25-min episode's captions are ~30-60 KB.
        settings.TRANSLATION_REFERENCE_SUBTITLES = str(req.reference_subtitles)[:400_000]

    if req.reference_mode is not None:
        _rm = str(req.reference_mode).strip().lower()
        settings.TRANSLATION_REFERENCE_MODE = _rm if _rm in ("adopt", "timing") else "adopt"

    _invalidate_status_cache()
    _persist_user_settings()
    return {
        "status": "saved",
        "beam_size": settings.WHISPER_BEAM_SIZE,
        "vad_filter": settings.WHISPER_VAD_FILTER,
        "frame_sample_rate": settings.FRAME_SAMPLE_RATE,
        "no_speech_threshold": float(getattr(
            settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.4)),
        "gap_fill_enabled": bool(getattr(
            settings, "WHISPER_GAP_FILL_ENABLED", True)),
        "gap_fill_min_sec": float(getattr(
            settings, "WHISPER_GAP_FILL_MIN_SEC", 1.5)),
        "gap_fill_no_speech_threshold": float(getattr(
            settings, "WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD", 0.25)),
        "sentence_segmentation_enabled": bool(getattr(
            settings, "SENTENCE_SEGMENTATION_ENABLED", True)),
        "series_hint": str(getattr(settings, "TRANSLATION_SERIES_HINT", "") or ""),
        "reference_subtitles": str(getattr(settings, "TRANSLATION_REFERENCE_SUBTITLES", "") or ""),
        "reference_mode": str(getattr(settings, "TRANSLATION_REFERENCE_MODE", "adopt") or "adopt"),
    }


# ── Custom Vocabulary (Whisper biasing) ──────────────────────────

class SaveVocabularyRequest(BaseModel):
    terms: Optional[list[str]] = None
    enabled: Optional[bool] = None


@router.get("/settings/vocabulary")
async def get_vocabulary():
    """Return the persisted custom-vocabulary glossary + enable flag."""
    from backend.services.custom_vocabulary import load_vocabulary, MAX_TERMS
    terms = load_vocabulary()
    return {
        "terms": terms,
        "enabled": bool(getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True)),
        "count": len(terms),
        "max": MAX_TERMS,
    }


@router.put("/settings/vocabulary")
async def put_vocabulary(req: SaveVocabularyRequest):
    """Validate + persist the custom-vocabulary glossary and enable flag.

    The term list is written to /data/logs/custom_vocabulary.json
    (mount-backed); the enable flag is persisted to user_settings.json.
    """
    from backend.services.custom_vocabulary import (
        load_vocabulary, save_vocabulary, MAX_TERMS,
    )
    if req.terms is not None:
        terms = save_vocabulary(req.terms)
    else:
        terms = load_vocabulary()
    if req.enabled is not None:
        settings.CUSTOM_VOCABULARY_ENABLED = bool(req.enabled)
        _persist_user_settings()
    return {
        "status": "saved",
        "terms": terms,
        "enabled": bool(getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True)),
        "count": len(terms),
        "max": MAX_TERMS,
    }


# ── Multi-host Ollama registry (remote GPU sharing) ──────────────
#
# Array order IS priority: index 0 = primary, the rest = ordered
# fallbacks. The Settings UI reorders with drag-and-drop and persists
# immediately through PUT. All probing happens server-side — the
# browser must never probe a host directly (CORS / mixed content).


class OllamaHostEntry(BaseModel):
    id: Optional[str] = None
    name: str = ""
    url: str
    # None = keep the stored token for this id (so edits don't force
    # re-entering secrets); "" = explicitly clear it.
    token: Optional[str] = None
    enabled: bool = True


class SaveOllamaHostsRequest(BaseModel):
    hosts: list[OllamaHostEntry]


class TestOllamaHostRequest(BaseModel):
    url: str
    token: str = ""


@router.get("/settings/ollama-hosts")
async def get_ollama_hosts():
    """The ordered host registry with live per-host status (server-probed)."""
    from backend.services import ollama_registry
    return {
        "hosts": await ollama_registry.registry_status(),
        "legacy_host": settings.OLLAMA_HOST,
        "migrated": not bool((settings.OLLAMA_HOSTS or "").strip()),
    }


@router.put("/settings/ollama-hosts")
async def put_ollama_hosts(req: SaveOllamaHostsRequest):
    """Replace the registry (order = priority) and persist immediately.

    Called on every drag-and-drop reorder, add, edit, toggle, and delete.
    Tokens: a ``null`` token keeps the stored secret for that host id; an
    empty string clears it. Tokens are never echoed back — the status
    payload only reports ``has_token``.
    """
    import uuid as _uuid
    from backend.services import ollama_registry
    _prev = {h.id: h for h in ollama_registry.get_hosts()}
    hosts = []
    for entry in req.hosts:
        url = (entry.url or "").strip()
        if not url:
            continue
        host_id = (entry.id or "").strip() or _uuid.uuid4().hex[:8]
        token = entry.token
        if token is None:
            token = _prev[host_id].token if host_id in _prev else ""
        # Server-owned metadata SURVIVES a UI save. The request model has no
        # is_companion/gpu fields, so rebuilding hosts from it alone silently
        # stripped the companion flag and its advertised GPU on every
        # drag-reorder — after which only a URL heuristic kept pairing,
        # Whisper routing, and the eviction exemption working.
        _old = _prev.get(host_id)
        hosts.append(ollama_registry.OllamaHost(
            id=host_id,
            name=(entry.name or "").strip() or url,
            url=url,
            token=token,
            enabled=bool(entry.enabled),
            gpu_name=_old.gpu_name if _old else "",
            vram_total_mb=_old.vram_total_mb if _old else 0,
            is_companion=_old.is_companion if _old else False,
        ))
    ollama_registry.save_hosts(hosts)
    _invalidate_status_cache()
    logger.info(
        "Ollama host registry saved: %s",
        " > ".join(f"{h.name}{'' if h.enabled else ' (disabled)'}" for h in hosts)
        or "(empty)",
    )
    return {
        "status": "saved",
        "hosts": await ollama_registry.registry_status(force=True),
        "primary": hosts[0].name if hosts else None,
    }


@router.get("/settings/ollama-hosts/{host_id}/token")
async def reveal_ollama_host_token(host_id: str):
    """Return a host's stored bearer token so the user can view/copy it and
    verify it matches the Companion's token."""
    from backend.services import ollama_registry
    host = next((h for h in ollama_registry.get_hosts() if h.id == host_id), None)
    if host is None:
        raise HTTPException(status_code=404, detail="host not found")
    return {"id": host.id, "token": host.token or ""}


@router.post("/settings/ollama-hosts/test")
async def test_ollama_host(req: TestOllamaHostRequest):
    """Server-side probe of one host URL (+ optional token) for the Add/Edit
    dialog's Test button. Returns online/offline, models, version, and a plain
    diagnosis of WHY it failed (wrong path / bad token / unreachable), plus a
    suggested URL when the entry looks like a GPU Companion missing /ollama."""
    from backend.services import ollama_registry
    from urllib.parse import urlparse
    url = (req.url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    async def _probe(u: str):
        cand = ollama_registry.OllamaHost(id="__test__", name="test", url=u, token=req.token or "")
        st = await ollama_registry.probe(cand, force=True)
        ollama_registry._probe_cache.pop("__test__", None)
        return cand, st

    candidate, status = await _probe(url)

    if not status.online:
        # A GPU Companion serves Ollama under /ollama. If the user pasted the
        # bare proxy address (no path), retry there and suggest it.
        norm = ollama_registry._normalize_url(url)
        path = urlparse(norm).path.rstrip("/")
        note = ""
        if not path:
            alt = norm + "/ollama"
            _, alt_status = await _probe(alt)
            if alt_status.online:
                return {
                    "online": True, "models": alt_status.models,
                    "latency_ms": alt_status.latency_ms, "version": "",
                    "error": "", "suggested_url": alt,
                    "note": f"This is a GPU Companion — reachable at {alt}. "
                            f"Use that URL (with /ollama).",
                }
        err = status.error or ""
        low = err.lower()
        if "auth rejected" in low or "401" in err or "403" in err:
            note = "Reached the host, but the bearer token was rejected — copy the exact token shown in the Companion."
        elif "http 404" in low:
            note = "Reached the host, but there's no Ollama API at this path. For a GPU Companion, add /ollama to the URL."
        elif any(k in low for k in ("timeout", "connect", "refused", "unreachable", "name or service")):
            note = ("Could not connect. Check the machine is on and the IP/port are right, "
                    "and that the Windows firewall allows inbound TCP on this port (Private network).")
        return {
            "online": False, "models": [], "latency_ms": status.latency_ms,
            "version": "", "error": err, "note": note,
        }

    version = ""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            vresp = await client.get(
                ollama_registry.join_url(url, "/api/version"),
                headers=ollama_registry.auth_headers(candidate))
            if vresp.status_code == 200:
                version = (vresp.json() or {}).get("version", "")
    except Exception:
        pass
    return {
        "online": status.online,
        "models": status.models,
        "latency_ms": status.latency_ms,
        "version": version,
        "error": status.error,
        "note": "",
    }


# ── GPU Companion pairing ────────────────────────────────────────


class CompanionRegisterRequest(BaseModel):
    """Sent by the GPU Companion's "Connect to ClipAI" screen."""
    name: str = "GPU Companion"
    # Full proxy base URL as reachable FROM the ClipAI server, e.g.
    # http://192.168.1.50:11500 — the Companion detects its own LAN IP.
    url: str
    token: str = ""
    gpu_name: str = ""
    vram_total_mb: int = 0
    version: str = ""
    # Register the Companion's whisper endpoint too (default on).
    register_whisper: bool = True




# ── Remote-Whisper resolution ─────────────────────────────────────────────
#
# Remote Whisper is no longer a separate setting: the paired GPU Companion in
# the Ollama host registry serves Ollama AND transcription on the same GPU with
# the same token, so reframer_audio resolves the endpoint live from
# ollama_registry.companion_host(). WHISPER_REMOTE_URL remains only as an
# optional fallback for a truly separate third-party server.

# Last time the paired Companion was seen online (ms epoch), so the UI can show
# "offline — last seen 3m ago" and distinguish an expired/gone Companion from
# one that was never paired.
_companion_seen: dict = {"url": "", "last_online_ms": 0}


def _find_companion_host(hosts=None):
    """The paired GPU Companion host (delegates to the registry's single source
    of truth). ``hosts`` arg kept for backward-compat; ignored."""
    from backend.services import ollama_registry as _oreg
    return _oreg.companion_host()


async def _companion_whisper_capable(base: str, token: str,
                                     timeout: float = 4.0):
    """Probe a Companion's ``/v1/health`` for Whisper capability. Returns True
    when it can serve transcription, False when it explicitly cannot, or None
    when the probe is inconclusive (network error / not a Companion)."""
    import httpx as _httpx
    url = f"{base.rstrip('/')}/v1/health"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with _httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            data = resp.json() or {}
            if data.get("service") == "clipai-gpu-companion":
                return bool((data.get("backends") or {}).get("whisper"))
    except Exception:
        return None
    return None


def _tiny_wav_bytes(seconds: float = 1.0, freq: float = 440.0, rate: int = 16000) -> bytes:
    """A short 16kHz mono WAV (a tone) to exercise a remote Whisper server with
    a real transcription request. Pure stdlib — no numpy/ffmpeg."""
    import io as _io, wave as _wave, struct as _struct, math as _math
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        frames = bytearray()
        for i in range(n):
            frames += _struct.pack("<h", int(2500 * _math.sin(2 * _math.pi * freq * (i / rate))))
        w.writeframes(bytes(frames))
    return buf.getvalue()


async def _verify_remote_whisper_transcribe(base: str, key: str, model: str,
                                            budget_s: float = 75.0) -> dict:
    """REAL round-trip: POST a tiny WAV to {base}/v1/audio/transcriptions and
    confirm the remote server actually transcribes it. A /v1/health 200 only
    proves the proxy answers — the sidecar starts on demand, so this is the only
    check that proves transcription will run on the remote GPU.

    The Companion serializes GPU transcription: a concurrent decode gets an
    honest 503 + ``Retry-After: 20`` — an INVITATION to retry, not a failure.
    A measured test clicked right after "Force end all jobs" (which stops the
    whisper sidecar) raced the restart, took the first 503 as terminal and
    reported "Whisper: failed — runs on this server" for a perfectly healthy
    Companion. So 503 (busy) and transient transport errors are retried
    within ``budget_s``, honoring Retry-After; a 502 (sidecar mid-restart)
    gets the same patience. Definitive answers (200/401/403/404) return
    immediately. Returns {reachable, transcribed, busy, detail, error}."""
    import asyncio as _asyncio
    import time as _time
    import httpx as _httpx
    out = {"reachable": False, "transcribed": False, "busy": False,
           "detail": "", "error": ""}
    base = (base or "").rstrip("/")
    if not base:
        out["error"] = "no remote Whisper URL"
        return out
    if "://" not in base:
        base = f"http://{base}"
    if base.endswith("/v1"):
        base = base[:-3]
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    url = f"{base}/v1/audio/transcriptions"
    data = {"response_format": "json"}
    if model:
        data["model"] = model
        # Sync the selected model so the Companion warms the same GPU model here.
        headers["X-ClipAI-Whisper-Model"] = model
    # Short connect (unreachable → fail fast); long read — a cold sidecar may
    # take tens of seconds to load its model on the first request.
    timeout = _httpx.Timeout(120.0, connect=5.0)
    deadline = _time.monotonic() + max(0.0, budget_s)
    attempt = 0
    while True:
        attempt += 1
        retry_after = 4.0
        try:
            # Fresh file tuple per attempt — the previous attempt consumed it.
            files = {"file": ("verify.wav", _tiny_wav_bytes(), "audio/wav")}
            async with _httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, files=files, data=data)
            out["reachable"] = True
            if r.status_code == 200:
                out["transcribed"] = True
                out["busy"] = False
                out["error"] = ""
                out["detail"] = (
                    "transcribed a test clip on the remote GPU"
                    + (f" (after waiting out a busy GPU, attempt {attempt})"
                       if attempt > 1 else ""))
                return out
            if r.status_code in (401, 403):
                out["error"] = "auth rejected — check the host access token"
                return out
            if r.status_code == 404:
                out["error"] = ("server has no /v1/audio/transcriptions "
                                "(no Whisper backend)")
                return out
            if r.status_code == 503:
                out["busy"] = True
                out["error"] = ("GPU busy with another transcription for the "
                                f"whole {budget_s:.0f}s test window — a job is "
                                "likely mid-decode; re-test when it finishes")
                try:
                    retry_after = min(20.0, max(
                        3.0, float(r.headers.get("Retry-After", 4))))
                except (TypeError, ValueError):
                    retry_after = 4.0
            elif r.status_code == 502:
                # Companion answers 502 while the sidecar restarts (e.g. right
                # after "Force end all jobs" stops it) — retry through it.
                out["error"] = ("Whisper sidecar was restarting for the whole "
                                f"{budget_s:.0f}s test window — re-test, and check "
                                "the Companion log if it persists")
            else:
                body = (r.text or "")[:160]
                out["error"] = f"HTTP {r.status_code}{': ' + body if body else ''}"
                return out
        except (_httpx.ConnectError, _httpx.ConnectTimeout) as e:
            # Nothing listening / host unroutable — retrying cannot help and
            # would hold the Settings dialog for the whole budget. Fail fast
            # (old behavior). Only BUSY (503) and mid-restart (502) answers
            # from a live Companion earn the retry patience.
            out["error"] = f"{type(e).__name__}: {str(e)[:140]}"
            return out
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {str(e)[:140]}"
        if _time.monotonic() + retry_after > deadline:
            return out
        await _asyncio.sleep(retry_after)


@router.post("/settings/companion-register")
async def companion_register(req: CompanionRegisterRequest):
    """Pair a GPU Companion: add its Ollama proxy to the registry AS PRIMARY
    and point remote Whisper at it.

    Deliberately UNAUTHENTICATED, like the rest of /api/settings/*: the
    identical action has always been available with no key one endpoint over
    (PUT /api/settings/ollama-hosts adds/reorders hosts), so the bearer
    requirement here protected nothing — it only sent first-run users hunting
    for an API key the setup flow never showed them. LAN-trust is the
    existing settings security model; a Bearer header, if sent by an older
    Companion, is simply ignored.

    Idempotent: re-pairing the same URL updates the existing entry (and
    re-promotes it to primary) instead of duplicating it.
    """
    from backend.services import ollama_registry
    base = (req.url or "").strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=400, detail="url is required")
    if "://" not in base:
        base = f"http://{base}"
    ollama_url = f"{base}/ollama"

    hosts = ollama_registry.get_hosts()
    # Identity is the Companion's own persistent token, NOT the URL. The URL
    # is a DHCP lease: when the Windows box renews to a new address, the
    # URL-keyed lookup missed, a SECOND host entry was created, and the stale
    # ghost kept the token/GPU info while transcription pointed at a dead IP —
    # the user had to delete and re-add the Companion by hand every time.
    # The token survives reboots and IP changes (companion.json), so a
    # token match is the same machine announcing a new address: REBIND it.
    existing = None
    if req.token:
        existing = next(
            (h for h in hosts
             if h.is_companion and h.token and h.token == req.token), None)
    if existing is None:
        existing = next((h for h in hosts if h.url == ollama_url), None)
    if existing is not None:
        hosts.remove(existing)
        if existing.url != ollama_url:
            logger.info(
                "GPU Companion '%s' moved: %s → %s — rebinding the existing "
                "host entry (same pairing token)",
                existing.name, existing.url, ollama_url)
            existing.url = ollama_url
        # Same token under ANOTHER entry too = ghost from a pre-rebind
        # double-pair; absorb it instead of racing it for primary.
        if req.token:
            hosts = [h for h in hosts
                     if not (h.token == req.token and h.id != existing.id)]
        existing.name = (req.name or existing.name).strip() or existing.name
        if req.token:
            existing.token = req.token
        existing.enabled = True
        entry = existing
    else:
        import uuid as _uuid
        entry = ollama_registry.OllamaHost(
            id=_uuid.uuid4().hex[:8],
            name=(req.name or "GPU Companion").strip(),
            url=ollama_url,
            token=req.token or "",
            enabled=True,
        )
    # Persist the advertised GPU so the "active models" indicator can name it,
    # and mark this host as a Companion (drives sync/readiness UI + keeps its
    # models from being evicted to free the local card).
    entry.is_companion = True
    if req.gpu_name:
        entry.gpu_name = req.gpu_name.strip()
    if req.vram_total_mb:
        entry.vram_total_mb = int(req.vram_total_mb)
    hosts.insert(0, entry)  # Companion becomes the PRIMARY
    ollama_registry.save_hosts(hosts)

    # Remote Whisper needs no separate setting: the host we just stored (with
    # its token) IS the transcription endpoint — reframer_audio resolves it live
    # from the registry. We only probe capability to shape the response message.
    whisper_registered = bool(req.register_whisper)
    if not whisper_registered:
        whisper_registered = bool(await _companion_whisper_capable(base, req.token or ""))
    try:
        from backend.services import reframer_audio as _ra
        _ra._REMOTE_HEALTH_CACHE.update({"checked_at": 0.0, "url": ""})
    except Exception:
        pass

    _persist_user_settings()
    _invalidate_status_cache()
    logger.info(
        "GPU Companion paired: '%s' (%s, gpu=%s, %d MB) — registered as "
        "primary Ollama host%s",
        entry.name, base, req.gpu_name or "?", req.vram_total_mb,
        " + remote Whisper" if whisper_registered else "",
    )
    return {
        "status": "paired",
        "ollama_host": {"id": entry.id, "name": entry.name, "url": entry.url,
                        "role": "primary"},
        "whisper_remote_url": base if whisper_registered else "",
        "hosts": await ollama_registry.registry_status(force=True),
    }


@router.post("/settings/companion-verify")
async def companion_verify():
    """Round-trip proof that work actually EXECUTES on the paired Companion —
    not just that its token authenticates. Runs a 1-token sentinel generation
    for each selected local model against the Companion's Ollama proxy and
    checks its Whisper capability, so "paired" means "a job ran on that GPU".

    Browser-called (from Settings), so it is unauthenticated like the sibling
    /settings/ollama-hosts/test and /providers/status endpoints — requiring the
    API key here 401'd the webapp's fetch and bounced it to the login screen.
    """
    from backend.services import ollama_registry as _oreg
    import httpx as _httpx
    hosts = _oreg.get_hosts()
    comp = next((h for h in hosts if h.is_companion), None)
    if comp is None:
        comp = next((h for h in hosts
                     if not _oreg.is_local_gpu_host(h.url)
                     and h.url.rstrip("/").endswith("/ollama")), None)
    if comp is None:
        raise HTTPException(status_code=404, detail="No paired GPU Companion found")

    want: list[str] = []
    for _m in (settings.OLLAMA_PRIMARY_MODEL, settings.OLLAMA_EDITORIAL_MODEL,
               settings.OLLAMA_TRANSLATION_MODEL):
        if _m and _m not in want:
            want.append(_m)

    gen_url = _oreg.join_url(comp.url, "/api/generate")
    headers = _oreg.auth_headers(comp)
    # What's actually installed on the Companion, so a configured
    # "qwen2.5:14b-instruct" resolves to an installed "qwen2.5:14b" (same model)
    # instead of a false "model not installed".
    try:
        _status = await _oreg.probe(comp)
        installed = list(_status.models or [])
    except Exception:
        installed = []
    results = []
    # Short connect so an unreachable Companion fails fast; long read so a cold
    # model still has time to load and answer the 1-token sentinel.
    _timeout = _httpx.Timeout(90.0, connect=5.0)
    async with _httpx.AsyncClient(timeout=_timeout) as client:
        for _m in want:
            item = {"model": _m, "ok": False, "detail": ""}
            call_model = _oreg.resolve_installed_tag(installed, _m) or _m
            try:
                r = await client.post(gen_url, headers=headers, json={
                    "model": call_model, "prompt": "ping", "stream": False,
                    "options": {"num_predict": 1},
                })
                if r.status_code == 200:
                    item["ok"] = True
                    item["detail"] = ("generated on the Companion GPU"
                                      if call_model == _m
                                      else f"generated on the Companion GPU (installed as {call_model})")
                elif r.status_code in (401, 403):
                    item["detail"] = "auth rejected — check the host access token"
                elif r.status_code == 404:
                    item["detail"] = "model not installed on the Companion"
                else:
                    item["detail"] = f"HTTP {r.status_code}"
            except Exception as e:
                item["detail"] = f"{type(e).__name__}: {str(e)[:120]}"
            results.append(item)

    comp_base = comp.url[:-len("/ollama")] if comp.url.endswith("/ollama") else comp.url
    try:
        from backend.services import reframer_audio as _ra
        whisper_cfg = bool(_ra.remote_whisper_configured())
    except Exception:
        whisper_cfg = False
    # Real transcription round-trip against the Companion — the only proof that
    # transcription will actually run on its GPU (a /v1/health 200 doesn't cut
    # it: the sidecar starts on demand and may still fail to transcribe).
    try:
        from backend.services.reframer_audio import remote_whisper_pick_model
        _wmodel = remote_whisper_pick_model(None)
    except Exception:
        _wmodel = ""  # let the remote server auto-pick its tier
    whisper = await _verify_remote_whisper_transcribe(comp_base, comp.token or "", _wmodel)
    whisper["configured"] = whisper_cfg
    ollama_ok = bool(want) and all(x["ok"] for x in results)
    return {
        # "verified" means the LLM path works AND transcription actually ran on
        # the Companion — a green light to route jobs there.
        "verified": ollama_ok and whisper.get("transcribed", False),
        "ollama_ok": ollama_ok,
        "host": {"name": comp.name, "url": comp.url, "gpu_name": comp.gpu_name},
        "models": results,
        "whisper": whisper,
    }


# ── Subtitle-polish cloud fallback (AI Providers card) ───────────
#
# When the LOCAL polish chain fails a batch (Ollama cold-load timeout /
# circuit breaker), the polisher can retry that batch via OpenRouter —
# which bills real money on an otherwise-local setup. This pair of
# endpoints exposes that choice as a single dropdown: "none" (strictly
# local, never spend), "auto" (best efficient-tier model), or a pinned
# OpenRouter model id. Persisted via user_settings.json like every other
# provider pick (SUBTITLE_POLISH_CLOUD_FALLBACK / _CLOUD_MODEL are
# already in the allow-list).


class SavePolishFallbackRequest(BaseModel):
    # "none" → fallback disabled; "auto" → enabled with the efficient-tier
    # default; anything else → enabled with that OpenRouter model id.
    choice: str


def _polish_fallback_choice() -> str:
    if not bool(getattr(settings, "SUBTITLE_POLISH_CLOUD_FALLBACK", True)):
        return "none"
    model = (getattr(settings, "SUBTITLE_POLISH_CLOUD_MODEL", "") or "").strip()
    return model if model else "auto"


@router.get("/settings/polish-fallback")
async def get_polish_fallback():
    """Current cloud-polish-fallback choice + the dropdown's model options."""
    from backend.services.transcript_polisher import _resolve_cloud_polish_model
    options = []
    try:
        from backend.services.providers.openrouter_provider import (
            SUBTITLE_POLISH_SHORTLIST)
        options = [
            {"id": e["id"], "tier": e.get("tier", ""),
             "rationale": e.get("rationale", "")}
            for e in SUBTITLE_POLISH_SHORTLIST
        ]
    except Exception:
        pass
    return {
        "choice": _polish_fallback_choice(),
        "enabled": bool(getattr(settings, "SUBTITLE_POLISH_CLOUD_FALLBACK", True)),
        "model": (getattr(settings, "SUBTITLE_POLISH_CLOUD_MODEL", "") or ""),
        # What "auto" resolves to right now, so the dropdown can say so.
        "auto_resolves_to": _resolve_cloud_polish_model(),
        "openrouter_key_set": _key_is_set(settings.OPENROUTER_API_KEY),
        "options": options,
    }


@router.put("/settings/polish-fallback")
async def put_polish_fallback(req: SavePolishFallbackRequest):
    """Persist the cloud-polish-fallback choice (none / auto / model id)."""
    choice = (req.choice or "").strip()
    if choice.lower() in ("none", "off", "disabled"):
        settings.SUBTITLE_POLISH_CLOUD_FALLBACK = False
        # Keep the stored model so re-enabling later restores the pick.
    elif choice.lower() == "auto" or choice == "":
        settings.SUBTITLE_POLISH_CLOUD_FALLBACK = True
        settings.SUBTITLE_POLISH_CLOUD_MODEL = ""
    else:
        if "/" not in choice:
            raise HTTPException(
                status_code=400,
                detail=f"{choice!r} is not an OpenRouter model id "
                       "(expected provider/model), 'auto', or 'none'")
        settings.SUBTITLE_POLISH_CLOUD_FALLBACK = True
        settings.SUBTITLE_POLISH_CLOUD_MODEL = choice
    _persist_user_settings()
    logger.info("Subtitle polish cloud fallback set to %r", _polish_fallback_choice())
    return await get_polish_fallback()


# ── Voiceprint registry (cross-job speaker naming) ───────────────

@router.get("/settings/voiceprints")
async def get_voiceprints():
    """List enrolled voiceprints (names, sample counts) for review."""
    from backend.services.voiceprint_registry import get_registry
    return {
        "enabled": bool(getattr(settings, "VOICEPRINT_ENABLED", True)),
        "match_threshold": float(getattr(settings, "VOICEPRINT_MATCH_THRESHOLD", 0.75)),
        "voiceprints": get_registry().list_voiceprints(),
    }


class SaveVoiceprintSettingsRequest(BaseModel):
    enabled: Optional[bool] = None
    match_threshold: Optional[float] = None


@router.put("/settings/voiceprints")
async def put_voiceprint_settings(req: SaveVoiceprintSettingsRequest):
    """Toggle the voiceprint registry + tune the cosine match threshold."""
    if req.enabled is not None:
        settings.VOICEPRINT_ENABLED = bool(req.enabled)
    if req.match_threshold is not None:
        settings.VOICEPRINT_MATCH_THRESHOLD = max(0.0, min(1.0, float(req.match_threshold)))
    _persist_user_settings()
    from backend.services.voiceprint_registry import get_registry
    return {
        "status": "saved",
        "enabled": bool(getattr(settings, "VOICEPRINT_ENABLED", True)),
        "match_threshold": float(getattr(settings, "VOICEPRINT_MATCH_THRESHOLD", 0.75)),
        "voiceprints": get_registry().list_voiceprints(),
    }


@router.delete("/settings/voiceprints/{voiceprint_id}")
async def delete_voiceprint(voiceprint_id: str):
    """Forget a single enrolled voice (privacy)."""
    from backend.services.voiceprint_registry import get_registry
    deleted = get_registry().delete(voiceprint_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Voiceprint not found")
    return {"status": "deleted", "id": voiceprint_id}


@router.delete("/settings/voiceprints")
async def clear_voiceprints():
    """Forget all enrolled voices (privacy)."""
    from backend.services.voiceprint_registry import get_registry
    n = get_registry().clear()
    return {"status": "cleared", "removed": n}


# ── Clip Generation Settings (Primary AI / VideoLLaMA3) ───────────

class SaveClipGenerationRequest(BaseModel):
    min_duration: Optional[int] = None
    max_duration: Optional[int] = None
    clip_count: Optional[int] = None
    preferred_subjects: Optional[str] = None
    avoid_subjects: Optional[str] = None
    discovery_prompt: Optional[str] = None  # "" = revert to the built-in default
    # VideoLLaMA3 Enhanced Discovery toggles (exposed in the Settings UI).
    videollama3_enhanced: Optional[bool] = None
    videollama3_refinement_pass: Optional[bool] = None
    videollama3_keyframe_analysis: Optional[bool] = None
    videollama3_fps: Optional[int] = None


def _clip_generation_state() -> dict:
    """Current clip-generation settings plus the built-in defaults the
    UI needs to drive its reset-to-default button."""
    from backend.services.reframer_clipper import DEFAULT_DISCOVERY_PROMPT
    return {
        "min_duration": settings.CLIP_MIN_DURATION,
        "max_duration": settings.CLIP_MAX_DURATION,
        "clip_count": settings.CLIP_COUNT,
        "preferred_subjects": settings.CLIP_PREFERRED_SUBJECTS,
        "avoid_subjects": settings.CLIP_AVOID_SUBJECTS,
        "discovery_prompt": settings.CLIP_DISCOVERY_PROMPT,
        "default_discovery_prompt": DEFAULT_DISCOVERY_PROMPT,
        # VideoLLaMA3 Enhanced Discovery — surfaced to the UI so toggles
        # render the current backend state on first paint.
        "videollama3_enhanced": bool(getattr(settings, "VIDEOLLAMA3_ENHANCED", False)),
        "videollama3_refinement_pass": bool(getattr(settings, "VIDEOLLAMA3_REFINEMENT_PASS", True)),
        "videollama3_keyframe_analysis": bool(getattr(settings, "VIDEOLLAMA3_KEYFRAME_ANALYSIS", True)),
        "videollama3_audio_annotation": bool(getattr(settings, "VIDEOLLAMA3_AUDIO_ANNOTATION", True)),
        "videollama3_adaptive_chunks": bool(getattr(settings, "VIDEOLLAMA3_ADAPTIVE_CHUNKS", True)),
        "videollama3_fps": int(getattr(settings, "VIDEOLLAMA3_FPS", 2)),
        "videollama3_max_frames": int(getattr(settings, "VIDEOLLAMA3_MAX_FRAMES", 128)),
        "videollama3_chunk_min_s": int(getattr(settings, "VIDEOLLAMA3_CHUNK_MIN_S", 120)),
        "videollama3_chunk_max_s": int(getattr(settings, "VIDEOLLAMA3_CHUNK_MAX_S", 900)),
        "defaults": {
            "min_duration": 60,
            "max_duration": 300,
            "clip_count": 0,
            "preferred_subjects": "",
            "avoid_subjects": "",
            "videollama3_enhanced": True,
            "videollama3_refinement_pass": True,
            "videollama3_keyframe_analysis": True,
            "videollama3_fps": 2,
        },
    }


@router.get("/clip-generation/settings")
async def get_clip_generation_settings():
    """Return the clip-generation (Primary AI) settings + their defaults."""
    return _clip_generation_state()


@router.post("/clip-generation/settings")
async def save_clip_generation_settings(req: SaveClipGenerationRequest):
    """Save clip-generation settings. They are overlaid onto the clipper
    config so the upload pipeline and the regenerate path both honor them.
    An empty discovery_prompt reverts VideoLLaMA3 to its built-in prompt."""
    if req.min_duration is not None:
        settings.CLIP_MIN_DURATION = max(5, min(1800, int(req.min_duration)))
    if req.max_duration is not None:
        settings.CLIP_MAX_DURATION = max(5, min(3600, int(req.max_duration)))
    if settings.CLIP_MIN_DURATION > settings.CLIP_MAX_DURATION:
        settings.CLIP_MIN_DURATION, settings.CLIP_MAX_DURATION = (
            settings.CLIP_MAX_DURATION, settings.CLIP_MIN_DURATION)
    if req.clip_count is not None:
        settings.CLIP_COUNT = max(0, min(100, int(req.clip_count)))
    if req.preferred_subjects is not None:
        settings.CLIP_PREFERRED_SUBJECTS = req.preferred_subjects.strip()
    if req.avoid_subjects is not None:
        settings.CLIP_AVOID_SUBJECTS = req.avoid_subjects.strip()
    if req.discovery_prompt is not None:
        settings.CLIP_DISCOVERY_PROMPT = req.discovery_prompt.strip()
    if req.videollama3_enhanced is not None:
        settings.VIDEOLLAMA3_ENHANCED = bool(req.videollama3_enhanced)
    if req.videollama3_refinement_pass is not None:
        settings.VIDEOLLAMA3_REFINEMENT_PASS = bool(req.videollama3_refinement_pass)
    if req.videollama3_keyframe_analysis is not None:
        settings.VIDEOLLAMA3_KEYFRAME_ANALYSIS = bool(req.videollama3_keyframe_analysis)
    if req.videollama3_fps is not None:
        settings.VIDEOLLAMA3_FPS = max(1, min(4, int(req.videollama3_fps)))
    _invalidate_status_cache()
    _persist_user_settings()
    return {"status": "saved", **_clip_generation_state()}


# ── Self-Hosted Mode (route the analysis pipeline to local AI) ────

class SaveSelfHostedRequest(BaseModel):
    self_hosted_mode: Optional[bool] = None
    clip_engine_source: Optional[str] = None     # auto | local | cloud
    editorial_ai_source: Optional[str] = None    # auto | local | cloud


def _self_hosted_state() -> dict:
    """Current self-hosted settings plus each engine's resolved source.

    ``stages`` resolves the four user-facing pipeline stages the Offline-Mode
    toggle controls (clip detection, transcription, translation, polishing) so
    the Settings UI can show an accurate Local/Cloud badge per stage.
    """
    return {
        "self_hosted_mode": settings.SELF_HOSTED_MODE,
        "clip_engine_source": settings.CLIP_ENGINE_SOURCE,
        "editorial_ai_source": settings.EDITORIAL_AI_SOURCE,
        "resolved": {
            "clip_engine": settings.resolve_ai_source("clip"),
            "editorial_ai": settings.resolve_ai_source("editorial"),
        },
        "stages": {
            "clip_detection": settings.resolve_stage_source("clip_detection"),
            "transcription": settings.resolve_stage_source("transcription"),
            "translation": settings.resolve_stage_source("translation"),
            "polishing": settings.resolve_stage_source("polishing"),
        },
        "ollama_host": getattr(settings, "OLLAMA_HOST", "") or "",
    }


@router.get("/self-hosted/settings")
async def get_self_hosted_settings():
    """Self-hosted-mode settings + the resolved local/cloud source per engine."""
    return _self_hosted_state()


@router.post("/self-hosted/settings")
async def save_self_hosted_settings(req: SaveSelfHostedRequest):
    """Save self-hosted-mode settings. Routing is read live via
    settings.resolve_ai_source(), so no restart is needed."""
    _allowed = {"auto", "local", "cloud"}
    was_self_hosted = bool(settings.SELF_HOSTED_MODE)
    if req.self_hosted_mode is not None:
        settings.SELF_HOSTED_MODE = bool(req.self_hosted_mode)
    if req.clip_engine_source in _allowed:
        settings.CLIP_ENGINE_SOURCE = req.clip_engine_source
    if req.editorial_ai_source in _allowed:
        settings.EDITORIAL_AI_SOURCE = req.editorial_ai_source

    # Turning Offline Mode OFF must genuinely hand primary/editorial back to the
    # cloud. Selecting Ollama models (Save Models) or the "Ollama (Local)" toggle
    # pins ``ollama`` at the FRONT of AI_FALLBACK_CHAIN, and provider_status picks
    # the first reachable provider in that chain — so Ollama would stay the active
    # provider even though editorial now resolves to "cloud", and the cloud /
    # Replicate engines would never reappear in the UI. On the on→off transition,
    # demote Ollama to the END of the chain: configured cloud providers lead again
    # while Ollama remains a last-resort fallback. (Turning it back ON re-routes to
    # Ollama via active_provider_chain, which forces ["ollama"].)
    if was_self_hosted and not settings.SELF_HOSTED_MODE:
        chain = [p.strip() for p in settings.AI_FALLBACK_CHAIN.split(",") if p.strip()]
        if len(chain) > 1 and chain[0] == "ollama":
            chain = [p for p in chain if p != "ollama"] + ["ollama"]
            settings.AI_FALLBACK_CHAIN = ",".join(chain)
            env_path = _find_env_file()
            if env_path:
                _upsert_env_var(env_path, "AI_FALLBACK_CHAIN", settings.AI_FALLBACK_CHAIN)
            logger.info("Offline Mode off — demoted Ollama to end of fallback chain: %s",
                        settings.AI_FALLBACK_CHAIN)

    _invalidate_status_cache()
    _persist_user_settings()
    return {"status": "saved", **_self_hosted_state()}


# ── Encoding Settings ─────────────────────────────────────────────

_VALID_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"}


class SaveEncodingSettingsRequest(BaseModel):
    preset: Optional[str] = None
    crf: Optional[int] = None
    threads: Optional[int] = None
    faststart: Optional[bool] = None


@router.get("/encoding/settings")
async def get_encoding_settings():
    """Return current FFmpeg encoding settings."""
    return {
        "preset": settings.FFMPEG_PRESET,
        "crf": settings.FFMPEG_CRF,
        "threads": settings.FFMPEG_THREADS,
        "faststart": settings.FFMPEG_FASTSTART,
    }


@router.post("/encoding/settings")
async def save_encoding_settings(req: SaveEncodingSettingsRequest):
    """Save FFmpeg encoding settings."""
    env_path = _find_env_file()

    if req.preset is not None and req.preset in _VALID_PRESETS:
        settings.FFMPEG_PRESET = req.preset
        if env_path:
            _upsert_env_var(env_path, "FFMPEG_PRESET", req.preset)

    if req.crf is not None:
        clamped = max(0, min(51, req.crf))
        settings.FFMPEG_CRF = clamped
        if env_path:
            _upsert_env_var(env_path, "FFMPEG_CRF", str(clamped))

    if req.threads is not None:
        clamped = max(0, min(32, req.threads))
        settings.FFMPEG_THREADS = clamped
        if env_path:
            _upsert_env_var(env_path, "FFMPEG_THREADS", str(clamped))

    if req.faststart is not None:
        settings.FFMPEG_FASTSTART = req.faststart
        if env_path:
            _upsert_env_var(env_path, "FFMPEG_FASTSTART", str(req.faststart))

    _persist_user_settings()
    return {
        "status": "saved",
        "preset": settings.FFMPEG_PRESET,
        "crf": settings.FFMPEG_CRF,
        "threads": settings.FFMPEG_THREADS,
        "faststart": settings.FFMPEG_FASTSTART,
    }


# ── GPU Hardware Acceleration ────────────────────────────────────


@router.get("/gpu-acceleration")
async def get_gpu_acceleration():
    """Return current GPU acceleration toggle state and detected GPU info.

    When enabled, runs GPU detection and returns full hardware details.
    When disabled, returns minimal info with vendor='none'.
    """
    from backend.services.clip_exporter import detect_gpu_capabilities

    # Use cached detection result if available — the test-encode can fail
    # transiently when Ollama is using the GPU. Only force re-detect when
    # explicitly requested (POST toggle endpoint uses force_redetect=True).
    gpu_info = await asyncio.to_thread(detect_gpu_capabilities, force_redetect=False)

    return {
        "enabled": settings.GPU_ACCELERATION_ENABLED,
        "vendor_override": settings.GPU_VENDOR_OVERRIDE,
        "hwdecode_enabled": settings.GPU_HWDECODE_ENABLED,
        "hevc_for_4k": settings.GPU_HEVC_FOR_4K,
        "gpu_device_index": settings.GPU_DEVICE_INDEX,
        "detected": {
            "vendor": gpu_info["vendor"],
            "gpu_name": gpu_info.get("gpu_name", "Unknown"),
            "encoder": gpu_info["encoder"],
            "hevc_encoder": gpu_info.get("hevc_encoder"),
            "decoder": gpu_info["decoder"],
            "hwaccel": gpu_info["hwaccel"],
            "capabilities": gpu_info.get("capabilities", []),
            "vram_mb": gpu_info.get("vram_mb", 0),
            "driver_version": gpu_info.get("driver_version", ""),
            "cuda_available": gpu_info.get("cuda_available", False),
            "whisper_device": gpu_info.get("whisper_device", "cpu"),
            "gpus": gpu_info.get("gpus", []),
            "gpu_issues": gpu_info.get("gpu_issues", []),
        },
    }


class GpuAccelerationRequest(BaseModel):
    enabled: bool
    vendor_override: Optional[str] = None


@router.post("/gpu-acceleration")
async def set_gpu_acceleration(req: GpuAccelerationRequest):
    """Toggle GPU acceleration on/off and optionally set vendor override.

    When toggled ON: clears cached GPU info, re-scans for available GPUs,
    runs test-encodes to confirm the encoder works, returns full GPU details.
    When toggled OFF: clears cache, returns CPU fallback info.
    """
    from backend.services.clip_exporter import detect_gpu_capabilities, _gpu_info_cache_clear
    from backend.services.compat_stubs import reload_model as reload_whisper_model

    settings.GPU_ACCELERATION_ENABLED = req.enabled
    if req.vendor_override is not None:
        settings.GPU_VENDOR_OVERRIDE = req.vendor_override

    _persist_user_settings()

    # Force re-detection so the response includes fresh GPU info
    _gpu_info_cache_clear()
    # Reload Whisper model so it moves between CPU/CUDA to match the toggle
    reload_whisper_model()
    # Run in thread to avoid blocking the event loop (subprocess calls inside).
    gpu_info = await asyncio.to_thread(detect_gpu_capabilities, force_redetect=True)

    return {
        "status": "saved",
        "enabled": settings.GPU_ACCELERATION_ENABLED,
        "detected": {
            "vendor": gpu_info["vendor"],
            "gpu_name": gpu_info.get("gpu_name", "Unknown"),
            "encoder": gpu_info["encoder"],
            "decoder": gpu_info["decoder"],
            "hwaccel": gpu_info["hwaccel"],
            "vram_mb": gpu_info.get("vram_mb", 0),
            "driver_version": gpu_info.get("driver_version", ""),
            "cuda_available": gpu_info.get("cuda_available", False),
            "whisper_device": gpu_info.get("whisper_device", "cpu"),
            "gpus": gpu_info.get("gpus", []),
            "gpu_issues": gpu_info.get("gpu_issues", []),
        },
    }


# ── GPU Preflight (manual trigger + status) ──────────────────────


@router.get("/gpu/preflight-status")
async def get_gpu_preflight_status():
    """Return current VRAM state + which Ollama models are loaded.

    Use this from the Settings UI to verify the preflight is doing what
    you expect, or to diagnose why analysis is slow.
    """
    import httpx as _httpx
    from backend.services.gpu_preflight import (
        _nvidia_smi_free_mb, _nvidia_smi_compute_procs,
    )

    free_mb = _nvidia_smi_free_mb()
    procs = _nvidia_smi_compute_procs()
    ollama_models: list[dict] = []
    host = (settings.OLLAMA_HOST or "").rstrip("/")
    if host:
        try:
            from backend.services import ollama_registry as _oreg
            _ps_url = _oreg.join_url(host, "/api/ps")
            async with _httpx.AsyncClient(
                    timeout=5.0, headers=_oreg.headers_for_url(_ps_url)) as client:
                resp = await client.get(_ps_url)
                if resp.status_code == 200:
                    for m in resp.json().get("models", []) or []:
                        size_vram = int(m.get("size_vram", 0) or 0)
                        ollama_models.append({
                            "name": m.get("name", ""),
                            "vram_mb": size_vram // (1024 * 1024),
                            "on_gpu": size_vram > 0,
                        })
        except Exception:
            pass

    return {
        "free_vram_mb": free_mb,
        "compute_procs": procs,
        "ollama_models": ollama_models,
        "settings": {
            "gpu_free_before_analysis": bool(settings.GPU_FREE_BEFORE_ANALYSIS),
            "gpu_free_before_whisper": bool(settings.GPU_FREE_BEFORE_WHISPER),
        },
    }


@router.post("/gpu/free-now")
async def free_gpu_now():
    """Manually trigger the same preflight the pipeline runs.

    Useful for the user to free VRAM on demand from the Settings UI
    without having to start (and cancel) an analysis. Always runs
    regardless of the ``GPU_FREE_BEFORE_ANALYSIS`` flag.
    """
    from backend.services.gpu_preflight import (
        _evict_all_ollama_models, _release_local_torch_vram,
        _nvidia_smi_free_mb,
    )

    before = _nvidia_smi_free_mb()
    # Evict only from a LOCAL-GPU Ollama host. When the primary is a remote
    # Companion, its models live on the Companion's card — unloading them frees
    # nothing on this server and just forces a cold reload on the next request.
    host = (settings.OLLAMA_HOST or "").rstrip("/")
    try:
        from backend.services import ollama_registry as _oreg
        _local = [h.url for h in _oreg.enabled_hosts()
                  if _oreg.is_local_gpu_host(h.url)]
        host = _local[0] if _local else ""
    except Exception:
        pass
    ollama_stats: dict = {}
    if host:
        try:
            ollama_stats = await _evict_all_ollama_models(host, "manual")
        except Exception as e:
            ollama_stats = {"error": str(e)}
    else:
        ollama_stats = {"skipped": True,
                        "reason": "Ollama host is a remote GPU — its VRAM is not this server's"}
    torch_freed = _release_local_torch_vram("manual")
    after = _nvidia_smi_free_mb()
    return {
        "status": "ok",
        "before_free_mb": before,
        "after_free_mb": after,
        "torch_freed_mb": torch_freed,
        "ollama": ollama_stats,
    }


# ── Client GPU (Browser) Report ──────────────────────────────────


class ClientGpuReport(BaseModel):
    """Reported by the browser after GPU detection."""
    webgpu_supported: bool = False
    webcodec_supported: bool = False
    gpu_name: str = ""
    gpu_vendor: str = ""
    estimated_vram_mb: int = 0
    has_fp16: bool = False
    whisper_capable: bool = False
    h264_hardware_encode: bool = False
    hevc_hardware_encode: bool = False
    h264_hw_encode: bool = False
    hevc_hw_encode: bool = False
    client_whisper_enabled: bool = False
    client_encoding_enabled: bool = False
    gpu_index: str = "0"
    gpu_backend: str = ""


@router.post("/client-gpu-report")
async def report_client_gpu(req: ClientGpuReport):
    """Store client GPU capabilities so the pipeline can decide where to process.

    The server uses this to skip server-side transcription if the client will
    handle it, or to prepare server-side fallback if the client can't.

    IMPORTANT: this describes the CLIENT's (browser / phone) GPU, which is
    irrelevant to where the SERVER runs analysis. It must NOT change the
    server's own GPU targeting:
      * ``GPU_DEVICE_INDEX`` (which physical CUDA device the server's FFmpeg /
        pipeline use) is never set from here — a phone reporting "Adreno 740,
        index 0" used to overwrite it and mis-point the server.
      * Server GPU acceleration is only AUTO-ENABLED, never reconfigured, and
        only when the SERVER itself actually has an NVIDIA GPU (verified via
        nvidia-smi / device nodes), not merely because the client claims one.
    Client capabilities (whisper_capable, encode flags, etc.) are still stored
    so the client-side offload decision works.
    """
    # NOTE: deliberately do NOT touch settings.GPU_DEVICE_INDEX here — see
    # docstring. The server picks its own device; the client's index is
    # meaningless server-side and persisting it corrupted GPU selection.
    if req.gpu_index and req.gpu_index != (settings.GPU_DEVICE_INDEX or ""):
        logger.info(
            "Client reported GPU index %s (%s) — ignored for server device "
            "targeting (server keeps GPU_DEVICE_INDEX=%r)",
            req.gpu_index, req.gpu_name, settings.GPU_DEVICE_INDEX,
        )

    # Auto-enable server GPU acceleration ONLY when the SERVER actually has an
    # NVIDIA GPU — never on the strength of the client's claim alone (a phone
    # reporting Adreno/Qualcomm must not flip the server into NVIDIA mode).
    # Probe the SERVER directly: nvidia-smi first, then /dev/nvidia* nodes.
    _server_has_nvidia = False
    try:
        import subprocess as _sp
        _smi = _sp.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        _server_has_nvidia = _smi.returncode == 0 and "GPU" in (_smi.stdout or "")
    except Exception:
        _server_has_nvidia = False
    if not _server_has_nvidia:
        try:
            import glob as _glob
            _server_has_nvidia = bool(_glob.glob("/dev/nvidia[0-9]*"))
        except Exception:
            _server_has_nvidia = False

    if _server_has_nvidia and not settings.GPU_ACCELERATION_ENABLED:
        logger.info(
            "Server has an NVIDIA GPU — auto-enabling server GPU acceleration "
            "(client report was the trigger, not the source)",
        )
        settings.GPU_ACCELERATION_ENABLED = True
        settings.GPU_VENDOR_OVERRIDE = "nvidia"
        _persist_user_settings()
        # Force GPU re-detection and reload Whisper model for CUDA
        try:
            from backend.services.clip_exporter import _gpu_info_cache_clear
            _gpu_info_cache_clear()
            from backend.services.compat_stubs import reload_model as reload_whisper_model
            reload_whisper_model()
        except Exception:
            pass

    logger.info(
        "Client GPU report: webgpu=%s gpu=%s vendor=%s whisper_capable=%s "
        "client_whisper=%s client_encoding=%s gpu_index=%s",
        req.webgpu_supported, req.gpu_name, req.gpu_vendor, req.whisper_capable,
        req.client_whisper_enabled, req.client_encoding_enabled, req.gpu_index,
    )
    return {"status": "received"}


# ── GPU QA & Validation ──────────────────────────────────────────────


@router.get("/gpu-qa")
async def gpu_qa_validation():
    """Run comprehensive GPU QA validation.

    Checks that GPU acceleration is properly configured and actually
    being used for both Whisper transcription and FFmpeg video encoding.
    Returns a structured report with pass/fail checks, warnings, and
    actionable recommendations.
    """
    import subprocess as _subprocess

    from backend.services.clip_exporter import (
        detect_gpu_capabilities,
        _gpu_encode_args,
        _gpu_decode_args,
    )
    from backend.services.compat_stubs import whisper_device_info

    checks = []
    warnings = []
    errors = []

    # ── 1. GPU Detection ──
    gpu_info = await asyncio.to_thread(detect_gpu_capabilities, force_redetect=True)
    gpu_vendor = gpu_info.get("vendor", "none")
    gpu_name = gpu_info.get("gpu_name", "Unknown")

    if gpu_vendor != "none":
        checks.append({
            "name": "GPU detected",
            "status": "pass",
            "detail": f"{gpu_name} (vendor: {gpu_vendor})",
        })
    else:
        checks.append({
            "name": "GPU detected",
            "status": "fail",
            "detail": "No GPU detected by FFmpeg/system probes",
        })
        errors.append("No GPU hardware detected — GPU acceleration cannot work")

    # ── 2. GPU Acceleration Setting ──
    if settings.GPU_ACCELERATION_ENABLED:
        checks.append({
            "name": "GPU acceleration enabled",
            "status": "pass",
            "detail": f"Enabled (vendor_override={settings.GPU_VENDOR_OVERRIDE or 'auto'})",
        })
    else:
        checks.append({
            "name": "GPU acceleration enabled",
            "status": "fail",
            "detail": "GPU acceleration is disabled in settings",
        })
        errors.append(
            "GPU acceleration is disabled — enable it in Settings > GPU Acceleration"
        )

    # ── 3. CUDA Runtime (for Whisper) ──
    cuda_available = False
    cuda_device_count = 0
    try:
        from backend.services.compat_stubs import _detect_cuda_available
        cuda_available, cuda_device_count, _ = _detect_cuda_available()
    except Exception:
        pass

    if cuda_available and cuda_device_count > 0:
        checks.append({
            "name": "CUDA runtime available",
            "status": "pass",
            "detail": f"{cuda_device_count} CUDA device(s) found",
        })
    else:
        checks.append({
            "name": "CUDA runtime available",
            "status": "warn" if gpu_vendor != "none" else "fail",
            "detail": "CUDA runtime not available (ctranslate2/torch cannot use GPU)",
        })
        if gpu_vendor == "nvidia":
            warnings.append(
                "NVIDIA GPU detected but CUDA runtime is not available. "
                "Ensure CUDA toolkit is installed and container has --gpus all."
            )

    # ── 4. Whisper Model GPU Status ──
    whisper_dev = whisper_device_info.get("device", "cpu")
    whisper_idx = whisper_device_info.get("device_index", 0)
    whisper_compute = whisper_device_info.get("compute_type", "int8")

    if whisper_dev == "cuda":
        checks.append({
            "name": "Whisper using GPU",
            "status": "pass",
            "detail": f"device=cuda:{whisper_idx}, compute_type={whisper_compute}",
        })
    else:
        status = "fail" if settings.GPU_ACCELERATION_ENABLED and cuda_available else "warn"
        checks.append({
            "name": "Whisper using GPU",
            "status": status,
            "detail": f"Whisper running on CPU ({whisper_compute})",
        })
        if status == "fail":
            errors.append(
                "GPU is enabled and CUDA is available, but Whisper is running on CPU. "
                "Try toggling GPU acceleration off and on to reload the model."
            )

    # ── 5. Whisper GPU Verification (live check) ──
    if whisper_dev == "cuda":
        verification_results = []
        try:
            import ctranslate2
            ct2_count = ctranslate2.get_cuda_device_count()
            if ct2_count > 0:
                verification_results.append(f"ctranslate2: {ct2_count} CUDA device(s)")
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated(whisper_idx)
                verification_results.append(
                    f"torch: {mem / 1024 / 1024:.1f}MB allocated on device {whisper_idx}"
                )
        except Exception:
            pass

        try:
            smi = _subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                our_pid = str(os.getpid())
                for line in smi.stdout.strip().split("\n"):
                    if our_pid in line:
                        verification_results.append(f"nvidia-smi: PID {our_pid} using GPU")
                        break
        except Exception:
            pass

        if verification_results:
            checks.append({
                "name": "Whisper GPU verification",
                "status": "pass",
                "detail": "; ".join(verification_results),
            })
        else:
            checks.append({
                "name": "Whisper GPU verification",
                "status": "warn",
                "detail": "Could not independently verify GPU memory usage",
            })
            warnings.append(
                "Whisper reports device=cuda but live GPU verification could not confirm usage"
            )

    # ── 6. FFmpeg GPU Encoder ──
    encoder = gpu_info.get("encoder", "")
    if encoder and encoder != "libx264":
        checks.append({
            "name": "FFmpeg GPU encoder available",
            "status": "pass",
            "detail": f"Encoder: {encoder}",
        })
    elif settings.GPU_ACCELERATION_ENABLED and gpu_vendor != "none":
        checks.append({
            "name": "FFmpeg GPU encoder available",
            "status": "fail",
            "detail": "GPU detected but no hardware encoder found by FFmpeg",
        })
        errors.append(
            "FFmpeg cannot find a GPU encoder. Ensure FFmpeg is built with NVENC/VAAPI/QSV support."
        )
    else:
        checks.append({
            "name": "FFmpeg GPU encoder available",
            "status": "info",
            "detail": "Using software encoder (libx264)",
        })

    # ── 7. FFmpeg GPU Decoder / HW Decode ──
    if settings.GPU_HWDECODE_ENABLED:
        hwaccel = gpu_info.get("hwaccel", "")
        if hwaccel:
            checks.append({
                "name": "FFmpeg GPU decoder available",
                "status": "pass",
                "detail": f"hwaccel: {hwaccel}",
            })
        else:
            checks.append({
                "name": "FFmpeg GPU decoder available",
                "status": "warn",
                "detail": "Hardware decode enabled but no hwaccel method detected",
            })
    else:
        checks.append({
            "name": "FFmpeg GPU decoder available",
            "status": "info",
            "detail": "Hardware decode disabled in settings",
        })

    # ── 8. Test Encode (quick NVENC/VAAPI probe) ──
    if settings.GPU_ACCELERATION_ENABLED and encoder and encoder != "libx264":
        try:
            # Provide a minimal quality preset dict for the test
            test_preset = {"crf": 23, "preset": "fast"}
            encode_args = await asyncio.to_thread(_gpu_encode_args, test_preset, "1080p")
            if encode_args:
                # Run a minimal test encode to verify GPU encoder actually works
                test_cmd = [
                    "ffmpeg", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=64x64:d=0.1:r=1",
                    *encode_args, "-frames:v", "1",
                    "-f", "null", "-",
                ]
                proc = await asyncio.create_subprocess_exec(
                    *test_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=15)
                if proc.returncode == 0:
                    checks.append({
                        "name": "GPU test encode",
                        "status": "pass",
                        "detail": f"Test encode succeeded with {encoder}",
                    })
                else:
                    stderr_text = stderr_bytes.decode(errors="replace")[-300:]
                    checks.append({
                        "name": "GPU test encode",
                        "status": "fail",
                        "detail": f"Test encode failed: {stderr_text}",
                    })
                    errors.append(
                        f"GPU encoder '{encoder}' failed test encode. "
                        "The GPU driver or FFmpeg build may not support this encoder."
                    )
            else:
                checks.append({
                    "name": "GPU test encode",
                    "status": "warn",
                    "detail": "No encode args returned — encoder may not be configured",
                })
        except asyncio.TimeoutError:
            checks.append({
                "name": "GPU test encode",
                "status": "warn",
                "detail": "Test encode timed out (15s)",
            })
        except Exception as exc:
            checks.append({
                "name": "GPU test encode",
                "status": "warn",
                "detail": f"Test encode error: {exc}",
            })

    # ── 9. Device Index Consistency ──
    configured_idx = (settings.GPU_DEVICE_INDEX or "0").strip()
    if whisper_dev == "cuda" and str(whisper_idx) != configured_idx:
        warnings.append(
            f"Whisper is on CUDA device {whisper_idx} but GPU_DEVICE_INDEX is '{configured_idx}'. "
            "Toggle GPU off/on to apply the new device index."
        )

    # ── Summary ──
    pass_count = sum(1 for c in checks if c["status"] == "pass")
    fail_count = sum(1 for c in checks if c["status"] == "fail")
    warn_count = sum(1 for c in checks if c["status"] == "warn")

    overall = "pass"
    if fail_count > 0:
        overall = "fail"
    elif warn_count > 0:
        overall = "warn"

    return {
        "overall": overall,
        "summary": f"{pass_count} passed, {fail_count} failed, {warn_count} warnings",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "config": {
            "gpu_acceleration_enabled": settings.GPU_ACCELERATION_ENABLED,
            "gpu_vendor_override": settings.GPU_VENDOR_OVERRIDE,
            "gpu_hwdecode_enabled": settings.GPU_HWDECODE_ENABLED,
            "gpu_hevc_for_4k": settings.GPU_HEVC_FOR_4K,
            "gpu_device_index": settings.GPU_DEVICE_INDEX,
        },
        "whisper": {
            "device": whisper_dev,
            "compute_type": whisper_compute,
            "device_index": whisper_idx,
            "gpu_name": whisper_device_info.get("gpu_name", ""),
        },
        "ffmpeg": {
            "vendor": gpu_vendor,
            "gpu_name": gpu_name,
            "encoder": encoder,
            "hevc_encoder": gpu_info.get("hevc_encoder", ""),
            "decoder": gpu_info.get("decoder", ""),
            "hwaccel": gpu_info.get("hwaccel", ""),
        },
    }


# ── Prompt Management ──────────────────────────────────────────────


class SavePromptsRequest(BaseModel):
    frame_analysis: Optional[str] = None
    viral_clip_detection: Optional[str] = None
    summary: Optional[str] = None
    seo: Optional[str] = None


@router.get("/prompts")
async def get_prompts():
    """Return current custom prompts and defaults."""
    current = load_prompts()
    defaults = get_defaults()
    return {
        "current": current.model_dump(),
        "defaults": defaults.model_dump(),
    }


@router.post("/prompts")
async def update_prompts(req: SavePromptsRequest):
    """Save custom prompts. Pass null/empty to reset a prompt to default."""
    current = load_prompts()
    defaults = get_defaults()

    if req.frame_analysis is not None:
        text = req.frame_analysis.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"Frame analysis prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.frame_analysis = text if text else defaults.frame_analysis

    if req.viral_clip_detection is not None:
        text = req.viral_clip_detection.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"Viral clip detection prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.viral_clip_detection = text if text else defaults.viral_clip_detection

    if req.summary is not None:
        text = req.summary.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"Summary prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.summary = text if text else defaults.summary

    if req.seo is not None:
        text = req.seo.strip()
        if len(text) > MAX_PROMPT_LENGTH:
            return {
                "status": "error",
                "message": f"SEO prompt exceeds {MAX_PROMPT_LENGTH} characters",
            }
        current.seo = text if text else defaults.seo

    save_prompts(current)
    return {"status": "saved", "prompts": current.model_dump()}


@router.post("/prompts/reset")
async def reset_prompts():
    """Reset all prompts to defaults."""
    defaults = get_defaults()
    save_prompts(defaults)
    return {"status": "reset", "prompts": defaults.model_dump()}


# ═══════════════════════════════════════════════════════════════
# Site Customisation — title, favicon, logo
# ═══════════════════════════════════════════════════════════════

SITE_CONFIG_PATH = os.path.join(_DATA_DIR, "site_config.json")
SITE_UPLOADS_DIR = os.path.join(_DATA_DIR, "site_uploads")


def _load_site_config() -> dict:
    if os.path.exists(SITE_CONFIG_PATH):
        try:
            with open(SITE_CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_site_config(cfg: dict):
    os.makedirs(os.path.dirname(SITE_CONFIG_PATH), exist_ok=True)
    with open(SITE_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


@router.get("/site-config")
async def get_site_config():
    """Return site customisation (title, favicon URL, logo URL)."""
    return _load_site_config()


@router.post("/site-config")
async def update_site_config(
    title: Optional[str] = Form(None),
    favicon: Optional[UploadFile] = File(None),
    logo: Optional[UploadFile] = File(None),
    remove_favicon: Optional[str] = Form(None),
    remove_logo: Optional[str] = Form(None),
):
    """Update site title, favicon, and/or logo."""
    cfg = _load_site_config()
    os.makedirs(SITE_UPLOADS_DIR, exist_ok=True)

    if title is not None:
        cfg["title"] = title.strip()

    if remove_favicon == "true":
        old = cfg.pop("favicon", None)
        if old:
            old_path = os.path.join(SITE_UPLOADS_DIR, os.path.basename(old))
            if os.path.isfile(old_path):
                os.remove(old_path)
    elif favicon and favicon.filename:
        ext = os.path.splitext(favicon.filename)[1].lower() or ".ico"
        fname = f"favicon-{uuid.uuid4().hex[:8]}{ext}"
        fpath = os.path.join(SITE_UPLOADS_DIR, fname)
        content = await favicon.read()
        with open(fpath, "wb") as f:
            f.write(content)
        cfg["favicon"] = fname

    if remove_logo == "true":
        old = cfg.pop("logo", None)
        if old:
            old_path = os.path.join(SITE_UPLOADS_DIR, os.path.basename(old))
            if os.path.isfile(old_path):
                os.remove(old_path)
    elif logo and logo.filename:
        ext = os.path.splitext(logo.filename)[1].lower() or ".png"
        fname = f"logo-{uuid.uuid4().hex[:8]}{ext}"
        fpath = os.path.join(SITE_UPLOADS_DIR, fname)
        content = await logo.read()
        with open(fpath, "wb") as f:
            f.write(content)
        cfg["logo"] = fname

    _save_site_config(cfg)
    return {"status": "saved", **cfg}


@router.get("/site-uploads/{filename}")
async def serve_site_upload(filename: str):
    """Serve uploaded site assets (favicon, logo)."""
    safe = os.path.basename(filename)
    fpath = os.path.join(SITE_UPLOADS_DIR, safe)
    if not os.path.isfile(fpath):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": "not found"})
    from fastapi.responses import FileResponse
    return FileResponse(fpath)


# ═══════════════════════════════════════════════════════════════
# UI State Sync — persists frontend localStorage to the server
# so settings stay consistent across browsers.
# ═══════════════════════════════════════════════════════════════

UI_STATE_PATH = os.path.join(_DATA_DIR, "ui_state.json")


def _load_ui_state() -> dict:
    if os.path.exists(UI_STATE_PATH):
        try:
            with open(UI_STATE_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_ui_state(state: dict):
    os.makedirs(os.path.dirname(UI_STATE_PATH), exist_ok=True)
    with open(UI_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


@router.get("/ui-state")
async def get_ui_state():
    """Return all persisted frontend UI state (settings, segments, etc.)."""
    return _load_ui_state()


@router.put("/ui-state")
async def put_ui_state(request: Request):
    """Merge incoming UI state into the persisted file.

    Accepts a JSON object of localStorage key→value pairs.  Values that
    are ``null`` delete the key from the persisted state so that
    localStorage.removeItem propagates to other browsers.
    """
    incoming = await request.json()
    state = _load_ui_state()
    for key, value in incoming.items():
        if value is None:
            state.pop(key, None)
        else:
            state[key] = value
    _save_ui_state(state)
    return {"status": "saved", "keys": len(state)}


# ─── Subtitle Quality + Translation Engine settings ───────────────────────


def _subtitle_quality_state() -> dict:
    """Return the current subtitle / translation / audio-event settings."""
    return {
        "subtitle_cps_enforcement": bool(getattr(settings, "SUBTITLE_CPS_ENFORCEMENT", True)),
        "subtitle_max_cps": float(getattr(settings, "SUBTITLE_MAX_CPS", 20.0)),
        "subtitle_max_chars_per_line": int(getattr(settings, "SUBTITLE_MAX_CHARS_PER_LINE", 42)),
        "subtitle_min_duration_ms": int(getattr(settings, "SUBTITLE_MIN_DURATION_MS", 833)),
        "subtitle_max_duration_ms": int(getattr(settings, "SUBTITLE_MAX_DURATION_MS", 9000)),
        "subtitle_smart_line_breaks": bool(getattr(settings, "SUBTITLE_SMART_LINE_BREAKS", True)),
        "subtitle_platform_safe_zones": bool(getattr(settings, "SUBTITLE_PLATFORM_SAFE_ZONES", True)),
        "subtitle_platform_profile": str(getattr(settings, "SUBTITLE_PLATFORM_PROFILE", "") or ""),
        "transcript_polishing_enabled": bool(getattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True)),
        "transcript_filler_removal": bool(getattr(settings, "TRANSCRIPT_FILLER_REMOVAL", True)),
        "transcript_sentence_repair": bool(getattr(settings, "TRANSCRIPT_SENTENCE_REPAIR", True)),
        "translation_engine": str(getattr(settings, "TRANSLATION_ENGINE", "auto")),
        "translation_context_window": int(getattr(settings, "TRANSLATION_CONTEXT_WINDOW", 5)),
        "translation_glossary_enabled": bool(getattr(settings, "TRANSLATION_GLOSSARY_ENABLED", True)),
        "nmt_device": str(getattr(settings, "NMT_DEVICE", "auto")),
        "audio_event_detection": bool(getattr(settings, "AUDIO_EVENT_DETECTION", True)),
        "audio_events_in_subtitles": bool(getattr(settings, "AUDIO_EVENTS_IN_SUBTITLES", False)),
        "audio_music_detection": bool(getattr(settings, "AUDIO_MUSIC_DETECTION", True)),
        "google_translate_configured": bool(
            (getattr(settings, "GOOGLE_TRANSLATE_API_KEY", "") or "").strip()
        ),
        "deepl_configured": bool(
            (getattr(settings, "DEEPL_API_KEY", "") or "").strip()
        ),
        # Dedicated subtitle-polish model + cloud STT provider (Phase 4)
        "subtitle_polish_model": str(getattr(settings, "SUBTITLE_POLISH_MODEL", "") or ""),
        "transcription_provider": str(getattr(settings, "TRANSCRIPTION_PROVIDER", "local") or "local"),
        "openai_configured": bool(
            (getattr(settings, "OPENAI_API_KEY", "") or "").strip()
        ),
        "groq_configured": bool(
            (getattr(settings, "GROQ_API_KEY", "") or "").strip()
        ),
    }


class SaveSubtitleQualityRequest(BaseModel):
    subtitle_cps_enforcement: Optional[bool] = None
    subtitle_max_cps: Optional[float] = None
    subtitle_max_chars_per_line: Optional[int] = None
    subtitle_min_duration_ms: Optional[int] = None
    subtitle_max_duration_ms: Optional[int] = None
    subtitle_smart_line_breaks: Optional[bool] = None
    subtitle_platform_safe_zones: Optional[bool] = None
    subtitle_platform_profile: Optional[str] = None
    transcript_polishing_enabled: Optional[bool] = None
    transcript_filler_removal: Optional[bool] = None
    transcript_sentence_repair: Optional[bool] = None
    translation_engine: Optional[str] = None
    translation_context_window: Optional[int] = None
    translation_glossary_enabled: Optional[bool] = None
    nmt_device: Optional[str] = None
    audio_event_detection: Optional[bool] = None
    audio_events_in_subtitles: Optional[bool] = None
    audio_music_detection: Optional[bool] = None
    google_translate_api_key: Optional[str] = None
    deepl_api_key: Optional[str] = None
    subtitle_polish_model: Optional[str] = None
    transcription_provider: Optional[str] = None
    openai_api_key: Optional[str] = None


_VALID_TRANSLATION_ENGINES = {"auto", "llm", "nllb", "opus-mt", "fugumt", "google", "deepl", "whisper"}
_VALID_NMT_DEVICES = {"auto", "cpu", "cuda"}
_VALID_PLATFORM_PROFILES = {"", "tiktok", "reels", "shorts", "horizontal", "square"}


@router.get("/subtitle-quality/settings")
async def get_subtitle_quality():
    """Return the subtitle / translation / audio-event settings."""
    return _subtitle_quality_state()


@router.post("/subtitle-quality/settings")
async def save_subtitle_quality(req: SaveSubtitleQualityRequest):
    """Save subtitle / translation / audio-event settings."""
    if req.subtitle_cps_enforcement is not None:
        settings.SUBTITLE_CPS_ENFORCEMENT = bool(req.subtitle_cps_enforcement)
    if req.subtitle_max_cps is not None:
        settings.SUBTITLE_MAX_CPS = max(5.0, min(30.0, float(req.subtitle_max_cps)))
    if req.subtitle_max_chars_per_line is not None:
        settings.SUBTITLE_MAX_CHARS_PER_LINE = max(20, min(80, int(req.subtitle_max_chars_per_line)))
    if req.subtitle_min_duration_ms is not None:
        settings.SUBTITLE_MIN_DURATION_MS = max(100, min(5000, int(req.subtitle_min_duration_ms)))
    if req.subtitle_max_duration_ms is not None:
        settings.SUBTITLE_MAX_DURATION_MS = max(1000, min(15000, int(req.subtitle_max_duration_ms)))
    if settings.SUBTITLE_MIN_DURATION_MS > settings.SUBTITLE_MAX_DURATION_MS:
        settings.SUBTITLE_MIN_DURATION_MS, settings.SUBTITLE_MAX_DURATION_MS = (
            settings.SUBTITLE_MAX_DURATION_MS, settings.SUBTITLE_MIN_DURATION_MS,
        )
    if req.subtitle_smart_line_breaks is not None:
        settings.SUBTITLE_SMART_LINE_BREAKS = bool(req.subtitle_smart_line_breaks)
    if req.subtitle_platform_safe_zones is not None:
        settings.SUBTITLE_PLATFORM_SAFE_ZONES = bool(req.subtitle_platform_safe_zones)
    if req.subtitle_platform_profile is not None:
        profile = (req.subtitle_platform_profile or "").strip().lower()
        if profile in _VALID_PLATFORM_PROFILES:
            settings.SUBTITLE_PLATFORM_PROFILE = profile
    if req.transcript_polishing_enabled is not None:
        settings.TRANSCRIPT_POLISHING_ENABLED = bool(req.transcript_polishing_enabled)
    if req.transcript_filler_removal is not None:
        settings.TRANSCRIPT_FILLER_REMOVAL = bool(req.transcript_filler_removal)
    if req.transcript_sentence_repair is not None:
        settings.TRANSCRIPT_SENTENCE_REPAIR = bool(req.transcript_sentence_repair)
    if req.translation_engine is not None:
        engine = (req.translation_engine or "").strip().lower()
        if engine in _VALID_TRANSLATION_ENGINES:
            settings.TRANSLATION_ENGINE = engine
    if req.translation_context_window is not None:
        settings.TRANSLATION_CONTEXT_WINDOW = max(0, min(20, int(req.translation_context_window)))
    if req.translation_glossary_enabled is not None:
        settings.TRANSLATION_GLOSSARY_ENABLED = bool(req.translation_glossary_enabled)
    if req.nmt_device is not None:
        dev = (req.nmt_device or "").strip().lower()
        if dev in _VALID_NMT_DEVICES:
            settings.NMT_DEVICE = dev
    if req.audio_event_detection is not None:
        settings.AUDIO_EVENT_DETECTION = bool(req.audio_event_detection)
    if req.audio_events_in_subtitles is not None:
        settings.AUDIO_EVENTS_IN_SUBTITLES = bool(req.audio_events_in_subtitles)
    if req.audio_music_detection is not None:
        settings.AUDIO_MUSIC_DETECTION = bool(req.audio_music_detection)
    # API keys — never persist blanks (mirrors the other API-key handlers).
    if req.google_translate_api_key is not None and req.google_translate_api_key.strip():
        settings.GOOGLE_TRANSLATE_API_KEY = req.google_translate_api_key.strip()
    if req.deepl_api_key is not None and req.deepl_api_key.strip():
        settings.DEEPL_API_KEY = req.deepl_api_key.strip()
    if req.subtitle_polish_model is not None:
        # Blank explicitly clears the pin (back to translation-model default)
        settings.SUBTITLE_POLISH_MODEL = req.subtitle_polish_model.strip()
    if req.transcription_provider is not None:
        prov = (req.transcription_provider or "").strip().lower()
        if prov in ("local", "groq", "openai"):
            settings.TRANSCRIPTION_PROVIDER = prov
    if req.openai_api_key is not None and req.openai_api_key.strip():
        settings.OPENAI_API_KEY = req.openai_api_key.strip()
    _invalidate_status_cache()
    _persist_user_settings()
    return {"status": "saved", **_subtitle_quality_state()}


# ─── Per-job glossary (KNP) endpoint ──────────────────────────────────────


class SaveGlossaryRequest(BaseModel):
    terms: dict = {}


@router.post("/jobs/{job_id}/glossary")
async def save_job_glossary(job_id: str, req: SaveGlossaryRequest):
    """Persist a per-job Key Name and Phrases (KNP) glossary.

    The translator looks for ``/data/uploads/{job_id}/glossary.json``
    at translation time and pins these terms in the prompt so they get
    translated consistently across all batches.
    """
    if not isinstance(req.terms, dict):
        return {"status": "error", "message": "terms must be an object"}
    safe_terms: dict[str, str] = {}
    for k, v in req.terms.items():
        ks = str(k or "").strip()
        vs = str(v or "").strip()
        if ks and vs and len(ks) <= 200 and len(vs) <= 200:
            safe_terms[ks] = vs
    job_dir = f"/data/uploads/{job_id}"
    if not os.path.isdir(job_dir):
        # Fall back to local repo path for tests / non-Docker dev.
        job_dir = os.path.join(_DATA_DIR, "uploads", job_id)
        os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "glossary.json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"terms": safe_terms}, fh, ensure_ascii=False, indent=2)
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "saved", "term_count": len(safe_terms), "path": path}


@router.get("/jobs/{job_id}/glossary")
async def get_job_glossary(job_id: str):
    """Return the per-job glossary if it exists."""
    candidates = [
        f"/data/uploads/{job_id}/glossary.json",
        os.path.join(_DATA_DIR, "uploads", job_id, "glossary.json"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            terms = data.get("terms", data) if isinstance(data, dict) else {}
            if not isinstance(terms, dict):
                terms = {}
            return {"terms": terms, "term_count": len(terms)}
        except Exception:
            continue
    return {"terms": {}, "term_count": 0}


# ─── NMT model download endpoint (on-demand) ──────────────────────────────


class DownloadNMTRequest(BaseModel):
    engine: str = "nllb"   # "nllb" | "opus-mt"
    source: Optional[str] = None
    target: Optional[str] = None


@router.post("/translation/download-model")
async def download_nmt_model(req: DownloadNMTRequest):
    """Download + convert an NMT model on demand. NEVER triggered at
    startup — only when the user explicitly clicks the download button
    in the Settings UI. The conversion can take several minutes and
    requires several hundred MB to a few GB of disk."""
    engine = (req.engine or "nllb").lower()
    if engine == "nllb":
        try:
            from backend.services.nmt_translator import ensure_nllb_downloaded
            path = await asyncio.to_thread(ensure_nllb_downloaded)
            return {"status": "ok", "engine": "nllb", "path": path}
        except Exception as e:
            return {"status": "error", "engine": "nllb", "message": str(e)}
    if engine in ("opus-mt", "fugumt"):
        if not req.source or not req.target:
            return {"status": "error", "message": f"source + target language codes required for {engine}"}
        try:
            from backend.config import settings as _s
            from backend.services.nmt_translator import ensure_opus_mt_downloaded
            _kw = ({} if engine == "opus-mt"
                   else {"subdir": "fugumt",
                         "model_template": getattr(_s, "NMT_FUGUMT_TEMPLATE",
                                                   "staka/fugumt-{src}-{tgt}")})
            path = await asyncio.to_thread(
                lambda: ensure_opus_mt_downloaded(req.source, req.target, **_kw))
            return {"status": "ok", "engine": engine, "path": path}
        except Exception as e:
            return {"status": "error", "engine": engine, "message": str(e)}
    return {"status": "error", "message": f"unknown engine: {engine}"}


# ── Editorial Judge config + model picker ─────────────────────────────
# The clipper's editorial judge can use any provider the user has
# already connected in Settings → Providers. The endpoints below let
# the UI render a dropdown of valid (vision-capable) editorial models
# and persist the user's pick to clipper_config.json.

# Known vision-capable Ollama tags. Ollama doesn't expose a clean
# "supports_vision" flag in /api/tags, so we recognise families by
# name. Pull requests welcome to extend this when new vision models
# ship.
_OLLAMA_VISION_TAGS = (
    "llava", "moondream", "bakllava", "minicpm-v",
    "llama3.2-vision", "llama4", "llama-4",
    "qwen2.5vl", "qwen-vl", "qwen2-vl", "qwen2.5-vl",
    "gemma3", "gemma-3", "internvl", "pixtral",
    "phi3.5-vision", "phi-3-vision",
)


def _clipper_config_path() -> str:
    """Locate clipper_config.json — delegates to the shared helper that
    prefers the mount-backed ``/data/clipper_config.json`` over the
    legacy ``/app/`` copy that doesn't survive container rebuilds."""
    from backend.services.pipeline import clipper_config_path
    return clipper_config_path()


def _read_clipper_config() -> dict:
    path = _clipper_config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f) or {}
    except Exception as e:
        logger.warning(f"Could not read clipper_config.json: {e}")
        return {}


def _write_clipper_config_fields(updates: dict) -> dict:
    """Merge ``updates`` into clipper_config.json on disk.

    Reads from wherever the file lives today (transparently migrates
    from the legacy ``/data/clipper_config.json`` ephemeral location
    or ``/app/clipper_config.json`` if that's where the read lands),
    applies the updates, and ALWAYS writes to the mount-backed
    canonical location (``/data/logs/clipper_config.json``) so the
    next ``docker compose down`` + ``rm -rf clipai`` + rebuild
    doesn't lose the user's saved judge primary / fallback. Only
    ``/data/logs/`` is mounted from the host in docker-compose.yml
    (``./data/logs:/data/logs``); ``/data/`` itself isn't, so the
    earlier ``/data/clipper_config.json`` write path was actually
    ephemeral. Atomic via temp file + rename.
    """
    from backend.services.pipeline import _canonical_clipper_config_path
    cfg = _read_clipper_config()  # reads from wherever it lives today
    cfg.update(updates)
    # Always write to the mount-backed canonical location regardless
    # of where the read came from. This auto-migrates anything that
    # was previously sitting on the ephemeral path.
    write_path = _canonical_clipper_config_path()
    tmp = write_path + ".tmp"
    os.makedirs(os.path.dirname(write_path) or ".", exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2)
    os.replace(tmp, write_path)
    return cfg


async def _editorial_models_openrouter() -> list:
    """Return vision-capable OpenRouter models for the judge dropdown."""
    raw = await _fetch_openrouter_models()
    if not raw:
        return []
    out = []
    for m in raw:
        mid = m.get("id", "")
        if not mid:
            continue
        arch = m.get("architecture", {}) or {}
        modality = str(arch.get("modality", "")).lower()
        input_mods = [str(x).lower() for x in arch.get("input_modalities", []) or []]
        has_vision = "image" in modality or "image" in input_mods
        if not has_vision:
            continue
        out.append({
            "id": mid,
            "label": m.get("name") or mid,
            "spec": f"openrouter:{mid}",
        })
    out.sort(key=lambda x: x["label"].lower())
    return out


def _editorial_models_gemini() -> list:
    """Hardcoded list of vision-capable Gemini models."""
    ids = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]
    return [{"id": i, "label": i, "spec": f"gemini:{i}"} for i in ids]


def _editorial_models_anthropic() -> list:
    """All current Claude models are vision-capable."""
    ids = [
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ]
    return [{"id": i, "label": i, "spec": f"anthropic:{i}"} for i in ids]


def _editorial_models_groq() -> list:
    """Groq vision-capable models (the llama-3.2-vision family)."""
    ids = [
        "llama-3.2-90b-vision-preview",
        "llama-3.2-11b-vision-preview",
    ]
    return [{"id": i, "label": i, "spec": f"groq:{i}"} for i in ids]


async def _editorial_models_ollama() -> list:
    """Query Ollama for installed models, filter to known vision tags."""
    try:
        from backend.services import ollama_registry
        _tags_url = f"{ollama_registry.primary_url() or settings.OLLAMA_HOST}/api/tags"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_tags_url,
                                    headers=ollama_registry.headers_for_url(_tags_url))
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.debug(f"Ollama tag list unavailable: {e}")
        return []
    out = []
    for m in data.get("models", []) or []:
        name = m.get("name", "")
        if not name:
            continue
        lower = name.lower()
        if not any(tag in lower for tag in _OLLAMA_VISION_TAGS):
            continue
        out.append({"id": name, "label": name, "spec": f"ollama:{name}"})
    out.sort(key=lambda x: x["label"].lower())
    return out


@router.get("/clipper/editorial-models")
async def list_editorial_models():
    """Return the dropdown data for the Editorial Judge picker.

    For each backend, reports whether the user has it connected
    (an API key set, or Ollama reachable) and lists the
    vision-capable models they can pick from. UI uses this to
    populate the primary + fallback selectors.
    """
    providers = []

    providers.append({
        "backend": "gemini",
        "label": "Google Gemini",
        "configured": _key_is_set(settings.GEMINI_API_KEY),
        "models": _editorial_models_gemini()
            if _key_is_set(settings.GEMINI_API_KEY) else [],
    })

    or_configured = _key_is_set(settings.OPENROUTER_API_KEY)
    providers.append({
        "backend": "openrouter",
        "label": "OpenRouter",
        "configured": or_configured,
        "models": (await _editorial_models_openrouter()) if or_configured else [],
    })

    providers.append({
        "backend": "anthropic",
        "label": "Anthropic",
        "configured": _key_is_set(settings.ANTHROPIC_API_KEY),
        "models": _editorial_models_anthropic()
            if _key_is_set(settings.ANTHROPIC_API_KEY) else [],
    })

    providers.append({
        "backend": "groq",
        "label": "Groq",
        "configured": _key_is_set(settings.GROQ_API_KEY),
        "models": _editorial_models_groq()
            if _key_is_set(settings.GROQ_API_KEY) else [],
    })

    ollama_models = await _editorial_models_ollama()
    providers.append({
        "backend": "ollama",
        "label": "Ollama (local)",
        "configured": bool(ollama_models),
        "models": ollama_models,
    })

    return {"providers": providers}


class JudgeConfigRequest(BaseModel):
    primary: Optional[str] = None    # spec "<backend>:<model>" or ""
    fallback: Optional[str] = None   # spec "<backend>:<model>" or ""


@router.get("/clipper/judge-config")
async def get_judge_config():
    """Return the currently saved editorial judge specs.

    Prefers the values persisted in user_settings.json (restored at module
    import — the same bulletproof path the primary/editorial model dropdowns
    use), falling back to clipper_config.json for installs saved before the
    spec was mirrored into settings. This is what makes the Editorial AI
    Fallback dropdown survive a no-cache container rebuild.
    """
    cfg = _read_clipper_config()
    _setting_fb = (getattr(settings, "EDITORIAL_AI_FALLBACK_SPEC", "") or "").strip()
    _cfg_fb = (cfg.get("judge_fallback", "") or "")
    primary = (getattr(settings, "EDITORIAL_AI_PRIMARY_SPEC", "") or "").strip() \
        or (cfg.get("judge_primary", "") or "")
    fallback = _setting_fb or _cfg_fb
    logger.info(
        "judge-config GET: fallback=%r (setting=%r, clipper_config=%r), primary=%r",
        fallback, _setting_fb, _cfg_fb, primary,
    )
    return {"primary": primary, "fallback": fallback}


@router.put("/clipper/judge-config")
async def put_judge_config(req: JudgeConfigRequest):
    """Save the editorial judge spec choices to clipper_config.json.

    Accepts ``primary`` and ``fallback`` as "<backend>:<model>" specs
    or empty strings. Empty primary disables the judge entirely; empty
    fallback disables the safety net but keeps the primary.
    """
    logger.info(
        "judge-config PUT received: primary=%r fallback=%r",
        req.primary, req.fallback,
    )
    updates = {}
    if req.primary is not None:
        updates["judge_primary"] = req.primary.strip()
    if req.fallback is not None:
        updates["judge_fallback"] = req.fallback.strip()
    cfg = _write_clipper_config_fields(updates)

    # Mirror into user_settings.json (via _PERSISTABLE_KEYS) — the same path
    # the primary/editorial dropdowns use — so the Editorial AI Fallback
    # survives a no-cache container rebuild without depending on the
    # clipper_config.json restore alone.
    if req.primary is not None:
        settings.EDITORIAL_AI_PRIMARY_SPEC = req.primary.strip()
    if req.fallback is not None:
        settings.EDITORIAL_AI_FALLBACK_SPEC = req.fallback.strip()
    _persist_user_settings()
    logger.info(
        "judge-config PUT persisted: EDITORIAL_AI_FALLBACK_SPEC=%r (clipper_config judge_fallback=%r)",
        getattr(settings, "EDITORIAL_AI_FALLBACK_SPEC", ""),
        cfg.get("judge_fallback", ""),
    )

    # Back up judge specs to user_settings.json so they survive any scenario
    # where clipper_config.json is lost (stale volume, first-time mount, etc.).
    # A startup handler reads these back and restores clipper_config.json.
    try:
        existing_us: dict = {}
        if os.path.exists(USER_SETTINGS_PATH):
            with open(USER_SETTINGS_PATH, "r") as _f:
                existing_us = json.load(_f)
        changed_us = False
        if req.primary is not None:
            _p = req.primary.strip()
            existing_us["_judge_primary"] = _p
            # Write the persistable spec key directly too, and REMOVE it when
            # cleared — _persist_user_settings preserves on-empty (to avoid an
            # unrelated save wiping it), so an explicit clear must delete the
            # key here or the old value would resurrect on restore.
            if _p:
                existing_us["EDITORIAL_AI_PRIMARY_SPEC"] = _p
            else:
                existing_us.pop("EDITORIAL_AI_PRIMARY_SPEC", None)
            changed_us = True
        if req.fallback is not None:
            _fb = req.fallback.strip()
            existing_us["_judge_fallback"] = _fb
            if _fb:
                existing_us["EDITORIAL_AI_FALLBACK_SPEC"] = _fb
            else:
                existing_us.pop("EDITORIAL_AI_FALLBACK_SPEC", None)
            changed_us = True
        if changed_us:
            tmp_us = USER_SETTINGS_PATH + ".tmp"
            os.makedirs(os.path.dirname(USER_SETTINGS_PATH), exist_ok=True)
            with open(tmp_us, "w") as _f:
                json.dump(existing_us, _f, indent=2)
            os.replace(tmp_us, USER_SETTINGS_PATH)
    except Exception as _e:
        logger.warning("Failed to back up judge specs to user_settings.json: %s", _e)

    return {
        "primary": cfg.get("judge_primary", "") or "",
        "fallback": cfg.get("judge_fallback", "") or "",
    }
