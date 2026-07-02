"""Optional cloud speech-to-text providers (audit Phase 4.1).

``TRANSCRIPTION_PROVIDER=local|groq|openai`` selects where the primary
transcription pass runs. Cloud output is mapped to the exact segment
schema the local faster-whisper path produces, so everything downstream
(hallucination filter → forced alignment → polish → formatter) is
provider-agnostic and quality stays uniform.

Providers:
  * groq   — ``whisper-large-v3-turbo`` via the OpenAI-compatible audio
             endpoint; very fast, word timestamps supported.
  * openai — ``whisper-1`` (word timestamps via verbose_json) or
             ``gpt-4o-transcribe`` (no word timestamps — the forced
             aligner re-times words afterwards).

The custom-vocabulary system is respected by injecting the glossary as
the provider ``prompt`` (both APIs accept a bias prompt).

Any failure returns None — the caller falls back to local Whisper.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT_S = 300  # generous — a 30-min upload on a slow link


def provider_selected() -> str:
    """The configured transcription provider ('local' when unset/bad)."""
    p = (getattr(settings, "TRANSCRIPTION_PROVIDER", "local") or "local")
    p = p.strip().lower()
    return p if p in ("local", "groq", "openai") else "local"


def cloud_available() -> bool:
    """True when a non-local provider is configured AND has an API key."""
    p = provider_selected()
    if p == "groq":
        return bool((settings.GROQ_API_KEY or "").strip())
    if p == "openai":
        return bool((getattr(settings, "OPENAI_API_KEY", "") or "").strip())
    return False


def _vocab_prompt(language: str) -> Optional[str]:
    """Custom-vocabulary bias prompt for the provider, or None."""
    try:
        from backend.services.custom_vocabulary import (
            build_initial_prompt, load_vocabulary)
        if not bool(getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True)):
            return None
        terms = load_vocabulary()
        if not terms:
            return None
        return build_initial_prompt(terms, language=language or "en") or None
    except Exception:
        return None


def _map_verbose_json(data: dict) -> list:
    """OpenAI-style verbose_json → local segment schema."""
    segments = []
    api_words = data.get("words") or []
    wi = 0
    for seg in data.get("segments") or []:
        s0 = float(seg.get("start", 0.0))
        s1 = float(seg.get("end", s0))
        text = (seg.get("text") or "").strip()
        words = []
        # words come as a flat list; attribute them to segments by time
        while wi < len(api_words):
            w = api_words[wi]
            ws = float(w.get("start", 0.0))
            if ws >= s1 - 1e-3 and wi < len(api_words) - 1:
                break
            if ws >= s0 - 0.05:
                words.append({
                    "word": (w.get("word") or "").strip(),
                    "start": round(ws, 3),
                    "end": round(float(w.get("end", ws)), 3),
                    # cloud APIs don't return per-word confidence — use a
                    # neutral value the TACT phantom filter treats as fine
                    "confidence": 0.9,
                })
            wi += 1
            if ws >= s1 - 1e-3:
                break
        segments.append({
            "start_sec": round(s0, 3),
            "end_sec": round(s1, 3),
            "text": text,
            "words": words,
            "is_hallucination": False,
            "no_speech_prob": round(float(seg.get("no_speech_prob", 0.0) or 0.0), 3),
            "avg_logprob": round(float(seg.get("avg_logprob", 0.0) or 0.0), 3),
            "source": "cloud",
        })
    if not segments and (data.get("text") or "").strip():
        # json (non-verbose) response, e.g. gpt-4o-transcribe — one blob;
        # the forced aligner + formatter re-segment downstream.
        dur = float(data.get("duration", 0.0) or 0.0)
        segments.append({
            "start_sec": 0.0,
            "end_sec": round(dur, 3) if dur else 0.0,
            "text": data["text"].strip(),
            "words": [],
            "is_hallucination": False,
            "no_speech_prob": 0.0,
            "avg_logprob": 0.0,
            "source": "cloud",
        })
    return segments


def transcribe_cloud(audio_path: str, language: Optional[str] = None) -> Optional[dict]:
    """Transcribe via the configured cloud provider.

    Returns {'segments', 'language', 'provider', 'model'} in the local
    schema, or None on any failure (caller falls back to local).
    """
    provider = provider_selected()
    if provider == "local" or not cloud_available():
        return None
    if not os.path.exists(audio_path):
        return None

    if provider == "groq":
        base = "https://api.groq.com/openai/v1"
        key = settings.GROQ_API_KEY.strip()
        model = (settings.GROQ_TRANSCRIBE_MODEL or "whisper-large-v3-turbo").strip()
    else:
        base = "https://api.openai.com/v1"
        key = getattr(settings, "OPENAI_API_KEY", "").strip()
        model = (getattr(settings, "OPENAI_TRANSCRIBE_MODEL", "whisper-1")
                 or "whisper-1").strip()

    # gpt-4o-transcribe only supports json/text; whisper models support
    # verbose_json with word timestamps.
    wants_words = not model.startswith("gpt-4o")
    data = {
        "model": model,
        "response_format": "verbose_json" if wants_words else "json",
    }
    if wants_words:
        data["timestamp_granularities[]"] = ["word", "segment"]
    if language and language not in ("auto", ""):
        data["language"] = language
    prompt = _vocab_prompt(language or "en")
    if prompt:
        data["prompt"] = prompt

    try:
        import httpx
        with open(audio_path, "rb") as fh:
            files = {"file": (os.path.basename(audio_path), fh, "audio/wav")}
            resp = httpx.post(
                f"{base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {key}"},
                data=data, files=files, timeout=_TIMEOUT_S)
        if resp.status_code != 200:
            logger.warning(
                "Cloud transcription (%s/%s) HTTP %s: %s — falling back to local",
                provider, model, resp.status_code, resp.text[:200])
            return None
        payload = resp.json()
    except Exception as e:
        logger.warning(
            "Cloud transcription (%s/%s) failed: %s — falling back to local",
            provider, model, e)
        return None

    segments = _map_verbose_json(payload)
    if not segments:
        logger.warning("Cloud transcription (%s/%s) returned no segments — "
                       "falling back to local", provider, model)
        return None
    logger.info("Cloud transcription (%s/%s): %d segments, language=%s",
                provider, model, len(segments),
                payload.get("language", language or "auto"))
    return {
        "segments": segments,
        "language": payload.get("language", language or "unknown"),
        "provider": provider,
        "model": model,
    }
