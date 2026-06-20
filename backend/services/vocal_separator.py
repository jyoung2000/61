"""Vocal separation (Demucs) — isolate the vocal stem before ASR.

Dialogue buried under loud music / SFX comes back from Whisper as *no_speech*:
the VAD hears the music bed, not the speech, so whole sections land in the
transcript as silence (confirmed on the Gundam Wing episode — battle / action
scenes with loud BGM dropped out entirely). Running source separation first
and transcribing ONLY the isolated vocal stem recovers that dialogue.

Design choices that matter on a small GPU (the 4 GB GTX 1650 this targets):

* **Subprocess, not in-process.** Demucs runs as ``python -m demucs`` in its
  own process, so *all* of its GPU memory is released the instant it exits —
  it can use the whole card before Whisper loads, then hand off a clean track.
  An in-process separator would fight Whisper for VRAM.
* **`--two-stems vocals`** — only split vocals vs. accompaniment (faster, less
  memory than a full 4-stem split).
* **`--segment`** caps the chunk length so htdemucs fits in <4 GB on CUDA; we
  fall back to CPU automatically if the GPU pass fails (OOM / no CUDA).
* **Self-healing.** Any failure (Demucs not installed, OOM, codec, timeout)
  returns ``None`` and the caller transcribes the original audio unchanged —
  separation can never break a job, only improve it.

The returned file is a 16 kHz mono WAV ready to feed straight to Whisper.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger("clipai.vocal_separator")


def is_available() -> bool:
    """True when the ``demucs`` package is importable in this interpreter."""
    try:
        return importlib.util.find_spec("demucs") is not None
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _device_order(device: str) -> list[str]:
    """Resolve the requested device into an ordered try-list.

    ``cpu`` → only CPU; ``cuda`` → CUDA then CPU fallback; ``auto`` → CUDA
    first when a CUDA device is visible, else CPU. The CPU fallback means a
    GPU OOM (common on 4 GB cards) still yields a vocal track, just slower.
    """
    device = (device or "auto").strip().lower()
    if device == "cpu":
        return ["cpu"]
    if device == "cuda":
        return ["cuda", "cpu"]
    return ["cuda", "cpu"] if _cuda_available() else ["cpu"]


def _demucs_cmd(input_path: str, out_dir: str, model: str, device: str,
                segment: Optional[int]) -> list[str]:
    """Build the ``python -m demucs`` command line (pure — unit tested)."""
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "--two-stems", "vocals",
        "-o", out_dir,
        "--device", device,
    ]
    # htdemucs is trained on ~7.8 s windows; a small --segment keeps the CUDA
    # workspace inside a 4 GB budget. Only meaningful on CUDA, harmless on CPU.
    if segment and device == "cuda":
        cmd += ["--segment", str(int(segment))]
    cmd.append(input_path)
    return cmd


def _vocals_output_path(out_dir: str, model: str, input_path: str) -> str:
    """Where Demucs writes the vocal stem: ``<out>/<model>/<stem>/vocals.wav``."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(out_dir, model, stem, "vocals.wav")


def _run(cmd: list[str], timeout: int) -> tuple[int, bytes]:
    """Run a subprocess, returning (returncode, stderr-tail). -1 on timeout."""
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return p.returncode, (p.stderr or b"")[-400:]
    except subprocess.TimeoutExpired:
        return -1, b"timeout"
    except Exception as e:  # pragma: no cover - defensive
        return -2, str(e).encode()[-400:]


def separate_vocals(
    source_path: str,
    work_dir: str,
    *,
    model: str = "htdemucs",
    device: str = "auto",
    segment: int = 7,
    sr: int = 16000,
    timeout: int = 1800,
) -> Optional[str]:
    """Isolate the vocal stem of ``source_path`` for transcription.

    Extracts a high-quality stereo WAV, runs Demucs ``--two-stems vocals``
    (CUDA with a CPU fallback), then resamples the vocal stem to ``sr`` Hz
    mono. Returns the path to that WAV, or ``None`` on any failure (the caller
    must treat ``None`` as "separation unavailable — use the original audio").
    """
    if not is_available():
        logger.info("Demucs not installed — skipping vocal separation")
        return None
    if not source_path or not os.path.exists(source_path):
        logger.info("Vocal separation: source missing (%s)", source_path)
        return None

    try:
        os.makedirs(work_dir, exist_ok=True)
    except Exception as e:
        logger.warning("Vocal separation: cannot create work dir (%s)", e)
        return None

    # 1. High-quality stereo extract for the separator (Demucs is trained on
    #    44.1 kHz stereo; feeding it the 16 kHz mono ASR copy hurts quality).
    src_wav = os.path.join(work_dir, "demucs_input.wav")
    rc, err = _run([
        "ffmpeg", "-y", "-i", source_path,
        "-vn", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", src_wav,
    ], timeout=min(timeout, 600))
    if rc != 0 or not os.path.exists(src_wav):
        logger.warning("Vocal separation: extract failed (rc=%s) %s", rc, err[-200:])
        return None

    # 2. Demucs (CUDA → CPU fallback). The subprocess frees all its GPU memory
    #    on exit, so Whisper can load the full card afterwards.
    raw_vocals = None
    for dev in _device_order(device):
        cmd = _demucs_cmd(src_wav, work_dir, model, dev, segment)
        logger.info("Vocal separation: running Demucs (%s, model=%s)", dev, model)
        rc, err = _run(cmd, timeout=timeout)
        cand = _vocals_output_path(work_dir, model, src_wav)
        if rc == 0 and os.path.exists(cand):
            raw_vocals = cand
            break
        logger.warning("Vocal separation: Demucs failed on %s (rc=%s) %s",
                       dev, rc, err[-200:])
    if not raw_vocals:
        return None

    # 3. Down-mix / resample the vocal stem to what Whisper wants (16 kHz mono).
    out_wav = os.path.join(work_dir, "vocals_16k.wav")
    rc, err = _run([
        "ffmpeg", "-y", "-i", raw_vocals,
        "-ac", "1", "-ar", str(int(sr)), "-c:a", "pcm_s16le", out_wav,
    ], timeout=300)
    if rc != 0 or not os.path.exists(out_wav):
        logger.warning("Vocal separation: vocal resample failed (rc=%s) %s", rc, err[-200:])
        return None

    logger.info("Vocal separation: vocal stem ready (%s)", os.path.basename(out_wav))
    return out_wav
