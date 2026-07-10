"""ClipAI GPU Companion — faster-whisper sidecar (Windows/NVIDIA).

A minimal OpenAI-compatible transcription server, packaged standalone
with PyInstaller (see whisper-server.spec) and bundled as a Companion
resource. It binds 127.0.0.1 only — the Companion's authenticated proxy
is the sole LAN surface.

Endpoints:
  GET  /health                     → {"status": "ok", "model": ..., "device": ...}
  POST /v1/audio/transcriptions    → OpenAI schema; response_format
                                     verbose_json returns segments + word
                                     timestamps (what ClipAI requires).

Configuration (env, set by the Companion per the VRAM budget):
  WHISPER_MODEL       model tag (large-v3-turbo / medium / small / ...)
  WHISPER_COMPUTE     ctranslate2 compute type (float16 / int8_float16)
  WHISPER_PORT        listen port (default 11510)
  WHISPER_MODELS_DIR  download/cache dir for model weights
  WHISPER_BATCH_SIZE  BatchedInferencePipeline batch size (default 16)

Optional decode-tuning env (parity with the ClipAI backend's local path;
unset = the engine's own defaults, i.e. the historical behavior). Every
value is feature-detected against the installed faster-whisper before use,
and a per-request form field overrides the env:
  WHISPER_BEAM                     beam size (historical default 5)
  WHISPER_VAD_ONSET                Silero speech threshold (0..1)
  WHISPER_VAD_MIN_SILENCE_MS       VAD min silence between segments
  WHISPER_VAD_SPEECH_PAD_MS        VAD padding around speech
  WHISPER_NO_SPEECH_THRESHOLD      no-speech skip threshold
  WHISPER_COND_PREV                condition_on_previous_text (0/1, default 1)
  WHISPER_NO_REPEAT_NGRAM          no_repeat_ngram_size
  WHISPER_HALLUCINATION_SILENCE_S  silence-gap hallucination guard (seconds)
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("whisper-sidecar")

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
PORT = int(os.environ.get("WHISPER_PORT", "11510"))
MODELS_DIR = os.environ.get("WHISPER_MODELS_DIR", "")
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "16"))

app = FastAPI(title="ClipAI Companion Whisper Sidecar")

_model = None
_batched = None
_model_lock = threading.Lock()
_device_used = "unknown"


def _parse_optional_bool(value):
    """Parse an optional boolean form field. ``None``/blank → ``None``
    (leave the engine default untouched)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return text in ("1", "true", "yes", "on")


def _env_float(name):
    """Optional float env var; unset / blank / unparsable → ``None``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("ignoring unparsable env %s=%r", name, raw)
        return None


def _env_int(name):
    """Optional int env var; unset / blank / unparsable → ``None``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        logger.warning("ignoring unparsable env %s=%r", name, raw)
        return None


def _env_tuning() -> dict:
    """Decode-tuning defaults from the environment (set by the Companion at
    launch — see src-tauri/src/sidecar.rs). This is the zero-protocol-risk
    parity channel: we launch this binary ourselves, so env vars can't 400 on
    a strict server the way extra multipart fields could.

    Every key is ``None`` when its env var is unset/invalid, which preserves
    the engine's own defaults exactly (current behavior). A per-request form
    field still wins over the env value (see ``transcribe``). ``WHISPER_COND_PREV``
    accepts 0/1; unset keeps faster-whisper's default (1/true).
    """
    cond_prev = _env_int("WHISPER_COND_PREV")
    return {
        "beam_size": _env_int("WHISPER_BEAM"),
        "vad_threshold": _env_float("WHISPER_VAD_ONSET"),
        "vad_min_silence_ms": _env_int("WHISPER_VAD_MIN_SILENCE_MS"),
        "vad_speech_pad_ms": _env_int("WHISPER_VAD_SPEECH_PAD_MS"),
        "no_speech_threshold": _env_float("WHISPER_NO_SPEECH_THRESHOLD"),
        "condition_on_previous_text": (None if cond_prev is None
                                       else bool(cond_prev)),
        "no_repeat_ngram_size": _env_int("WHISPER_NO_REPEAT_NGRAM"),
        "hallucination_silence_threshold": _env_float(
            "WHISPER_HALLUCINATION_SILENCE_S"),
    }


def _tuned_transcribe_kwargs(
    transcribe_callable,
    *,
    beam_size=None,
    vad_threshold=None,
    vad_min_silence_ms=None,
    vad_speech_pad_ms=None,
    no_speech_threshold=None,
    condition_on_previous_text=None,
    no_repeat_ngram_size=None,
    log_prob_threshold=None,
    compression_ratio_threshold=None,
    hallucination_silence_threshold=None,
) -> dict:
    """Tuning kwargs for a transcribe call, filtered to what the installed
    faster-whisper build accepts (the same feature-detection ClipAI's
    ``_vocab_bias_kwargs`` uses — only explicit named parameters count, a
    ``**kwargs`` catch-all does NOT, so an older build never raises on an
    unknown kwarg).

    Every argument is optional; ``None`` means "leave the loaded build's own
    default in place", which preserves the sidecar's historical behavior for
    clients (speaches / whisper.cpp compatibility) that never send the field.
    """
    import inspect
    try:
        params = inspect.signature(transcribe_callable).parameters
    except (TypeError, ValueError):
        return {}

    desired = {}
    if beam_size is not None:
        desired["beam_size"] = max(1, int(beam_size))
    if no_speech_threshold is not None:
        desired["no_speech_threshold"] = float(no_speech_threshold)
    if condition_on_previous_text is not None:
        desired["condition_on_previous_text"] = bool(condition_on_previous_text)
    if no_repeat_ngram_size is not None:
        desired["no_repeat_ngram_size"] = max(0, int(no_repeat_ngram_size))
    if log_prob_threshold is not None:
        desired["log_prob_threshold"] = float(log_prob_threshold)
    if compression_ratio_threshold is not None:
        desired["compression_ratio_threshold"] = float(compression_ratio_threshold)
    if hallucination_silence_threshold is not None:
        desired["hallucination_silence_threshold"] = float(
            hallucination_silence_threshold)

    vad_parameters = {}
    if vad_threshold is not None and 0.0 < float(vad_threshold) < 1.0:
        vad_parameters["threshold"] = float(vad_threshold)
    if vad_min_silence_ms is not None:
        vad_parameters["min_silence_duration_ms"] = max(0, int(vad_min_silence_ms))
    if vad_speech_pad_ms is not None:
        vad_parameters["speech_pad_ms"] = max(0, int(vad_speech_pad_ms))
    if vad_parameters:
        desired["vad_parameters"] = vad_parameters

    return {k: v for k, v in desired.items() if k in params}


def _load_model():
    """Load once, GPU first with a CPU fallback so a driver hiccup never
    kills the sidecar outright."""
    global _model, _device_used
    with _model_lock:
        if _model is not None:
            return _model
        from faster_whisper import WhisperModel
        kwargs = {}
        if MODELS_DIR:
            os.makedirs(MODELS_DIR, exist_ok=True)
            kwargs["download_root"] = MODELS_DIR
        for device, compute in (("cuda", COMPUTE), ("cpu", "int8")):
            try:
                logger.info("loading %s on %s (%s)...", MODEL_NAME, device, compute)
                _model = WhisperModel(MODEL_NAME, device=device,
                                      compute_type=compute, **kwargs)
                _device_used = f"{device}_{compute}"
                logger.info("model ready on %s", _device_used)
                return _model
            except Exception as e:  # noqa: BLE001 — fall to the next tier
                logger.warning("load on %s failed: %s", device, e)
        raise RuntimeError("could not load the whisper model on GPU or CPU")


def _load_batched(engine):
    """Wrap the loaded model in a BatchedInferencePipeline (once). Returns
    ``None`` when the installed faster-whisper doesn't ship it — callers
    fall back to the sequential ``engine.transcribe``."""
    global _batched
    with _model_lock:
        if _batched is not None:
            return _batched
        try:
            from faster_whisper import BatchedInferencePipeline
            _batched = BatchedInferencePipeline(model=engine)
            logger.info("batched inference pipeline ready (batch_size=%d)", BATCH_SIZE)
        except Exception as e:  # noqa: BLE001 — sequential fallback
            logger.info("batched inference unavailable (%s) — sequential decode", e)
            _batched = None
        return _batched


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "compute": COMPUTE,
        "device": _device_used,
        "loaded": _model is not None,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(""),
    language: str = Form(""),
    prompt: str = Form(""),
    response_format: str = Form("verbose_json"),
    temperature: float = Form(0.0),
    # ── Optional decode-tuning fields (ClipAI parity) ──
    # All default to None so a client that never sends them (speaches /
    # whisper.cpp compatibility) gets the sidecar's historical behavior:
    # beam_size=5, vad_filter=True with faster-whisper's default VAD and
    # decoding parameters. ClipAI's RemoteWhisperEngine sends the same tuned
    # values its local path computes so the two paths decode identically.
    beam_size: Optional[float] = Form(None),
    vad_threshold: Optional[float] = Form(None),
    vad_min_silence_ms: Optional[float] = Form(None),
    vad_speech_pad_ms: Optional[float] = Form(None),
    no_speech_threshold: Optional[float] = Form(None),
    condition_on_previous_text: Optional[str] = Form(None),
    no_repeat_ngram_size: Optional[float] = Form(None),
    log_prob_threshold: Optional[float] = Form(None),
    compression_ratio_threshold: Optional[float] = Form(None),
    hallucination_silence_threshold: Optional[float] = Form(None),
):
    """OpenAI-compatible transcription. The ``model`` form field is
    accepted for schema compatibility but the loaded model serves every
    request — the Companion restarts this sidecar to change tiers."""
    engine = _load_model()

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        audio_path = tmp.name

    try:
        base_kwargs = {
            "vad_filter": True,
            "word_timestamps": True,
        }
        if language and language not in ("auto", ""):
            base_kwargs["language"] = language
        if prompt:
            base_kwargs["initial_prompt"] = prompt
        if temperature:
            base_kwargs["temperature"] = temperature

        # Precedence: per-request form field > launch env (_env_tuning) >
        # engine default. beam_size keeps its historical hardcoded 5 as the
        # last resort so a bare request decodes exactly as before.
        env = _env_tuning()
        _cond_prev = _parse_optional_bool(condition_on_previous_text)
        _beam = beam_size if beam_size is not None else env["beam_size"]
        tuning = dict(
            beam_size=5 if _beam is None else _beam,
            vad_threshold=(vad_threshold if vad_threshold is not None
                           else env["vad_threshold"]),
            vad_min_silence_ms=(vad_min_silence_ms
                                if vad_min_silence_ms is not None
                                else env["vad_min_silence_ms"]),
            vad_speech_pad_ms=(vad_speech_pad_ms
                               if vad_speech_pad_ms is not None
                               else env["vad_speech_pad_ms"]),
            no_speech_threshold=(no_speech_threshold
                                 if no_speech_threshold is not None
                                 else env["no_speech_threshold"]),
            condition_on_previous_text=(
                _cond_prev if _cond_prev is not None
                else env["condition_on_previous_text"]),
            no_repeat_ngram_size=(no_repeat_ngram_size
                                  if no_repeat_ngram_size is not None
                                  else env["no_repeat_ngram_size"]),
            # Form-only (no env channel): log-prob / compression-ratio floors.
            log_prob_threshold=log_prob_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            hallucination_silence_threshold=(
                hallucination_silence_threshold
                if hallucination_silence_threshold is not None
                else env["hallucination_silence_threshold"]),
        )

        # Batched decode first (large speedup on long uploads), sequential
        # fallback on any batched failure so a request never dies on it.
        segments_iter = info = None
        batched = _load_batched(engine)
        if batched is not None:
            try:
                segments_iter, info = batched.transcribe(
                    audio_path, batch_size=BATCH_SIZE, **base_kwargs,
                    **_tuned_transcribe_kwargs(batched.transcribe, **tuning))
            except Exception as e:  # noqa: BLE001 — sequential fallback
                logger.warning("batched decode failed (%s) — sequential fallback",
                               str(e)[:200])
                segments_iter = info = None
        if segments_iter is None:
            segments_iter, info = engine.transcribe(
                audio_path, **base_kwargs,
                **_tuned_transcribe_kwargs(engine.transcribe, **tuning))

        segments = []
        words = []
        full_text = io.StringIO()
        for i, seg in enumerate(segments_iter):
            text = seg.text
            full_text.write(text)
            segments.append({
                "id": i,
                "seek": getattr(seg, "seek", 0),
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": text,
                "tokens": [],
                "temperature": temperature,
                "avg_logprob": round(float(getattr(seg, "avg_logprob", 0.0) or 0.0), 4),
                "compression_ratio": round(float(getattr(seg, "compression_ratio", 0.0) or 0.0), 4),
                "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0) or 0.0), 4),
            })
            for w in (getattr(seg, "words", None) or []):
                words.append({
                    "word": w.word.strip(),
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                })

        if response_format in ("text",):
            return PlainTextResponse(full_text.getvalue().strip())
        if response_format in ("json",):
            return JSONResponse({"text": full_text.getvalue().strip()})
        # verbose_json (default) — what ClipAI asks for.
        return JSONResponse({
            "task": "transcribe",
            "language": getattr(info, "language", language or "en"),
            "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
            "text": full_text.getvalue().strip(),
            "segments": segments,
            "words": words,
        })
    except Exception as e:  # noqa: BLE001
        logger.exception("transcription failed")
        return JSONResponse({"error": {"message": str(e)[:500]}}, status_code=500)
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


if __name__ == "__main__":
    logger.info("whisper sidecar starting: model=%s compute=%s port=%d",
                MODEL_NAME, COMPUTE, PORT)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
