"""Server-side filmstrip sprite + waveform peaks generation.

Professional web NLEs (Premiere Web, Descript, Veed) do NOT thumbnail a
long video by seeking a hidden ``<video>`` in the browser — that pins the
main thread and produces nothing the first time a source is opened. They
serve a precomputed *sprite sheet* (one JPEG holding N evenly-spaced tiles,
like a YouTube storyboard) plus a *peaks array* for the audio waveform, and
the client just slices the sheet and draws the peaks.

This module produces both, cheaply and once, from a job's source video:

  * :func:`generate_sprite` — a single ``fps→scale→tile`` FFmpeg pass writes
    ``sprite.jpg`` (a grid of tiles) and ``sprite.json`` (the manifest the
    editor uses to map a scrub time to a tile rectangle).
  * :func:`generate_peaks` — FFmpeg PCM decode + NumPy min/max binning writes
    ``peaks.json`` (~2000 interleaved min/max pairs, a few KB).

Both artifacts land in the job dir (``/data/uploads/{job_id}/``) alongside
the other editor sidecars (``render_plan.json``, ``detection_overlay.json``)
and are served by :mod:`backend.routers.filmstrip`.

Everything here is best-effort: any failure returns ``None`` and the editor
falls back to its existing client-side generation, so nothing regresses.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# ── Sprite geometry ────────────────────────────────────────────────────────
# One 160px-wide tile per sample, laid out in a 12-column grid. The sample
# interval scales with duration so a 30s clip and a 2h film both land on a
# single ~1920px-wide sheet (a 2h source → 12×50 ≈ 1920×4500, ~400KB).
_TILE_W = 160
_COLS = 12
_MAX_ROWS = 64            # JPEG dim ceiling is 65535px; 64*~90 keeps us safe
_MIN_INTERVAL = 2.0       # never finer than one tile / 2s (short clips)

# ── Waveform peaks ─────────────────────────────────────────────────────────
_PEAKS_N = 2000           # min/max pairs across the whole timeline
_PEAKS_SR = 16000         # matches the Whisper audio.wav sample rate


def _run(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def _probe_dims_duration(source_path: str) -> Optional[tuple[int, int, float]]:
    """Return ``(width, height, duration_seconds)`` via ffprobe, or ``None``."""
    try:
        proc = _run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height:format=duration",
                "-of", "json", source_path,
            ],
            timeout=30,
        )
        if proc.returncode != 0:
            logger.warning("ffprobe failed: %s", proc.stderr[:200].decode(errors="replace"))
            return None
        data = json.loads(proc.stdout or b"{}")
        stream = (data.get("streams") or [{}])[0]
        w = int(stream.get("width") or 0)
        h = int(stream.get("height") or 0)
        dur = float((data.get("format") or {}).get("duration") or 0.0)
        if w <= 0 or h <= 0 or dur <= 0:
            return None
        return w, h, dur
    except Exception as e:  # noqa: BLE001 — best-effort probe
        logger.warning("ffprobe crashed: %s", e)
        return None


def _sprite_layout(src_w: int, src_h: int, duration: float) -> dict:
    """Compute the adaptive sprite grid for a source of the given dims/length."""
    # Aim for <= _COLS * _MAX_ROWS tiles; scale the interval so long videos
    # stay on one sheet while short ones keep fine granularity.
    interval = max(_MIN_INTERVAL, math.ceil(duration / (_COLS * (_MAX_ROWS - 1))))
    count = max(1, math.ceil(duration / interval))
    rows = max(1, math.ceil(count / _COLS))
    # Even tile height matching the source aspect (yuv420 needs even dims).
    tile_h = max(2, int(round(_TILE_W * src_h / src_w / 2.0)) * 2)
    return {
        "interval": float(interval),
        "cols": _COLS,
        "rows": rows,
        "tileW": _TILE_W,
        "tileH": tile_h,
        "count": count,
        "duration": round(duration, 3),
        "sheet": "sprite.jpg",
        "sheetW": _TILE_W * _COLS,
        "sheetH": tile_h * rows,
    }


def generate_sprite(source_path: str, out_dir: str) -> Optional[dict]:
    """Build ``sprite.jpg`` + ``sprite.json`` for a source video.

    Returns the manifest dict on success, ``None`` on any failure. Safe to
    call repeatedly — it overwrites in place.
    """
    if not source_path or not os.path.isfile(source_path):
        return None
    dims = _probe_dims_duration(source_path)
    if not dims:
        return None
    src_w, src_h, duration = dims
    layout = _sprite_layout(src_w, src_h, duration)

    os.makedirs(out_dir, exist_ok=True)
    sprite_path = os.path.join(out_dir, "sprite.jpg")
    manifest_path = os.path.join(out_dir, "sprite.json")
    tmp_path = sprite_path + ".tmp"

    vf = (
        f"fps=1/{layout['interval']:g},"
        f"scale={layout['tileW']}:{layout['tileH']},"
        f"tile={layout['cols']}x{layout['rows']}"
    )

    def _sprite_cmd(fast: bool) -> list:
        # A single decode-once pass emits the whole packed sheet; -frames:v 1
        # keeps the first (and only, for our grid size) tiled output frame.
        # FAST path: GPU-assisted decode (falls back to SW automatically) +
        # -skip_frame nokey so only keyframes are fully decoded. That turns a
        # full 2-hour decode into a keyframe scan — dramatically faster, and
        # keyframe-accurate timestamps are plenty for a scrub filmstrip.
        pre = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if fast:
            pre += ["-hwaccel", "auto", "-skip_frame", "nokey"]
        return pre + ["-i", source_path, "-vf", vf, "-frames:v", "1",
                      "-q:v", "5", "-an", tmp_path]

    # A full uniform-sample decode of a long source is the cost here; generous
    # ceiling so 2h films finish, short clips return instantly.
    proc = _run(_sprite_cmd(fast=True), timeout=1800)
    if proc.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        # Fast path unsupported on this box (no HW decoder / codec quirk) —
        # retry with a plain software full-decode.
        logger.info("sprite fast path failed (rc=%s), retrying software", proc.returncode)
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        proc = _run(_sprite_cmd(fast=False), timeout=1800)
    if proc.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) == 0:
        logger.warning(
            "sprite ffmpeg failed (rc=%s): %s",
            proc.returncode, proc.stderr[:300].decode(errors="replace"),
        )
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None

    os.replace(tmp_path, sprite_path)
    try:
        with open(manifest_path, "w") as f:
            json.dump(layout, f)
    except OSError as e:
        logger.warning("sprite manifest write failed: %s", e)
        return None
    logger.info(
        "sprite written: %d tiles (%dx%d grid, %dx%d px) → %s",
        layout["count"], layout["cols"], layout["rows"],
        layout["sheetW"], layout["sheetH"], sprite_path,
    )
    return layout


def generate_peaks(audio_path: str, out_dir: str, n_peaks: int = _PEAKS_N) -> Optional[dict]:
    """Build ``peaks.json`` (interleaved min/max, ~n_peaks pairs).

    ``audio_path`` may be the job's pre-extracted ``audio.wav`` (fast, no
    re-decode of the video) or the source video itself (lazy fallback path).
    Returns the payload dict on success, ``None`` on failure.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return None
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        logger.warning("numpy unavailable for peaks: %s", e)
        return None

    # Reuse the pipeline's PCM idiom (audio_analyzer._extract_pcm_mono):
    # 16 kHz mono s16le on stdout.
    proc = _run(
        [
            "ffmpeg", "-v", "error", "-i", audio_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(_PEAKS_SR), "-ac", "1",
            "-f", "s16le", "-",
        ],
        timeout=900,
    )
    if proc.returncode != 0 or not proc.stdout:
        logger.warning(
            "peaks ffmpeg failed (rc=%s): %s",
            proc.returncode, proc.stderr[:200].decode(errors="replace"),
        )
        return None

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        return None
    spp = max(1, pcm.size // n_peaks)          # samples per peak bin
    usable = (pcm.size // spp) * spp
    if usable == 0:
        return None
    bins = pcm[:usable].reshape(-1, spp)
    mins = (bins.min(axis=1).astype(np.float32) / 32768.0)
    maxs = (bins.max(axis=1).astype(np.float32) / 32768.0)
    data = np.empty(mins.size * 2, dtype=np.float32)
    data[0::2] = mins
    data[1::2] = maxs

    payload = {
        "version": 2,
        "channels": 1,
        "sample_rate": _PEAKS_SR,
        "samples_per_pixel": int(spp),
        "bits": 16,
        "length": int(mins.size),
        "duration": round(pcm.size / _PEAKS_SR, 3),
        "data": [round(float(x), 4) for x in data],
    }
    os.makedirs(out_dir, exist_ok=True)
    peaks_path = os.path.join(out_dir, "peaks.json")
    try:
        with open(peaks_path, "w") as f:
            json.dump(payload, f)
    except OSError as e:
        logger.warning("peaks write failed: %s", e)
        return None
    logger.info("peaks written: %d bins → %s", mins.size, peaks_path)
    return payload
