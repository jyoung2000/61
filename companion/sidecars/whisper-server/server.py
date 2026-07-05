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
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
import threading

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("whisper-sidecar")

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
PORT = int(os.environ.get("WHISPER_PORT", "11510"))
MODELS_DIR = os.environ.get("WHISPER_MODELS_DIR", "")

app = FastAPI(title="ClipAI Companion Whisper Sidecar")

_model = None
_model_lock = threading.Lock()
_device_used = "unknown"


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
        kwargs = {
            "beam_size": 5,
            "vad_filter": True,
            "word_timestamps": True,
        }
        if language and language not in ("auto", ""):
            kwargs["language"] = language
        if prompt:
            kwargs["initial_prompt"] = prompt
        if temperature:
            kwargs["temperature"] = temperature

        segments_iter, info = engine.transcribe(audio_path, **kwargs)

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
