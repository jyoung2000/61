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
  * :func:`generate_sprite_coarse` — a SEEK-SAMPLED low-tile-count sprite for
    long videos. The fine sprite above decodes (at least the keyframes of)
    the whole file — minutes on a 2-hour source — while this one input-seeks
    to ~48 spots and grabs one frame each, finishing in seconds. It writes
    the same ``sprite.jpg``/``sprite.json`` names with ``"coarse": true`` so
    the editor shows a usable filmstrip immediately and transparently
    upgrades when the fine sheet replaces it.
  * :func:`generate_peaks` — PCM min/max binning writes ``peaks.json``
    (~2000 interleaved min/max pairs, a few KB). The pipeline's own 16 kHz
    mono ``audio.wav`` is binned via a zero-copy memmap; anything else is
    streamed out of FFmpeg in chunks — neither path holds the full PCM
    (≈250 MB for 2 h) in RAM.

Manifests carry a ``"v"`` stamp (write time) so the frontend can cache-bust
``sprite.jpg`` when a coarse sheet is upgraded to the fine one — the sprite
image itself is served with a long max-age.

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
import time
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

# ── Coarse (seek-sampled) sprite ───────────────────────────────────────────
_COARSE_TILES = 48        # enough for a readable strip, cheap to sample
_COARSE_MIN_DURATION = 300.0   # short sources: the fine pass is fast anyway
_COARSE_WORKERS = 4       # parallel single-frame seeks

# ── Waveform peaks ─────────────────────────────────────────────────────────
_PEAKS_N = 2000           # min/max pairs across the whole timeline
_PEAKS_SR = 16000         # matches the Whisper audio.wav sample rate
_PEAKS_CHUNK = 4 * 1024 * 1024   # streaming read granularity (bytes)


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
    # The tmp name must KEEP a .jpg extension — ffmpeg infers the output
    # muxer from it, and "sprite.jpg.tmp" made every sprite attempt fail
    # with "Unable to choose an output format" (rc=234), so long videos
    # got no scrub filmstrip at all.
    tmp_path = os.path.join(out_dir, "sprite.tmp.jpg")

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
    layout["v"] = int(time.time())
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


def generate_sprite_coarse(source_path: str, out_dir: str,
                           tiles: int = _COARSE_TILES) -> Optional[dict]:
    """Build a quick seek-sampled ``sprite.jpg`` + ``sprite.json`` for a LONG
    source, so the editor has a filmstrip in seconds while the fine sheet
    (which needs a whole-file keyframe scan — minutes on a 2-hour video)
    generates behind it.

    Input-seeking (``-ss`` before ``-i``) reads only a few MB around each
    sample point, so N samples cost seconds regardless of file length. The
    manifest is marked ``"coarse": true``; the frontend polls and swaps in
    the fine sheet when its un-marked manifest (with a newer ``"v"``)
    appears.

    Returns the manifest on success, ``None`` when skipped (short source,
    fine sprite already present) or failed. Never raises.
    """
    try:
        if not source_path or not os.path.isfile(source_path):
            return None
        dims = _probe_dims_duration(source_path)
        if not dims:
            return None
        src_w, src_h, duration = dims
        if duration < _COARSE_MIN_DURATION:
            return None      # fine pass is fast enough — don't do double work
        # A finished (fine) sprite already serves this job; don't clobber it.
        manifest_path = os.path.join(out_dir, "sprite.json")
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path) as f:
                    if not json.load(f).get("coarse"):
                        return None
            except Exception:
                pass

        count = max(8, min(int(tiles), 96))
        interval = duration / count
        cols = _COLS
        rows = max(1, math.ceil(count / cols))
        tile_h = max(2, int(round(_TILE_W * src_h / src_w / 2.0)) * 2)

        os.makedirs(out_dir, exist_ok=True)
        tile_dir = os.path.join(out_dir, ".sprite_tiles")
        os.makedirs(tile_dir, exist_ok=True)

        def _grab(i: int) -> Optional[str]:
            t = (i + 0.5) * interval
            path = os.path.join(tile_dir, f"tile_{i:03d}.jpg")
            proc = _run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-ss", f"{t:.3f}", "-i", source_path,
                 "-frames:v", "1", "-vf", f"scale={_TILE_W}:{tile_h}",
                 "-q:v", "6", "-an", path],
                timeout=60,
            )
            if proc.returncode == 0 and os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
            return None

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=_COARSE_WORKERS) as pool:
            grabbed = list(pool.map(_grab, range(count)))
        ok = sum(1 for g in grabbed if g)
        if ok < count * 0.5:
            logger.warning("coarse sprite: only %d/%d samples decoded — skipping", ok, count)
            _rmtree_quiet(tile_dir)
            return None

        # Compose the grid with Pillow (already a runtime dependency). A
        # missing tile leaves its cell black — visually fine for a strip
        # that lives only until the fine sheet replaces it.
        from PIL import Image
        sheet = Image.new("RGB", (_TILE_W * cols, tile_h * rows), (12, 12, 14))
        for i, path in enumerate(grabbed):
            if not path:
                continue
            try:
                with Image.open(path) as im:
                    sheet.paste(im.convert("RGB").resize((_TILE_W, tile_h)),
                                ((i % cols) * _TILE_W, (i // cols) * tile_h))
            except Exception:
                continue
        sprite_path = os.path.join(out_dir, "sprite.jpg")
        tmp_path = os.path.join(out_dir, "sprite.tmp.jpg")
        sheet.save(tmp_path, "JPEG", quality=70)
        os.replace(tmp_path, sprite_path)
        _rmtree_quiet(tile_dir)

        layout = {
            "interval": float(interval),
            "cols": cols,
            "rows": rows,
            "tileW": _TILE_W,
            "tileH": tile_h,
            "count": count,
            "duration": round(duration, 3),
            "sheet": "sprite.jpg",
            "sheetW": _TILE_W * cols,
            "sheetH": tile_h * rows,
            "coarse": True,
            "v": int(time.time()),
        }
        with open(manifest_path, "w") as f:
            json.dump(layout, f)
        logger.info("coarse sprite written: %d/%d tiles in seek-sample mode → %s",
                    ok, count, sprite_path)
        return layout
    except Exception as e:  # noqa: BLE001 — strictly best-effort
        logger.warning("coarse sprite failed: %s", e)
        return None


def _rmtree_quiet(path: str) -> None:
    try:
        import shutil
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _wav_pcm16_memmap(path: str):
    """``(memmap_int16, channels, sample_rate)`` for a PCM-16 RIFF WAV, else
    ``None``. Parses the chunk list directly (the ``wave`` module hides the
    ``data`` offset) so the pipeline's big ``audio.wav`` can be binned
    zero-copy — no decode subprocess, no 250 MB buffer."""
    try:
        import numpy as np
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if fh.read(4) != b"RIFF":
                return None
            fh.seek(8)
            if fh.read(4) != b"WAVE":
                return None
            pos, channels, rate, fmt_ok, data_off, data_len = 12, 0, 0, False, 0, 0
            while pos + 8 <= size:
                fh.seek(pos)
                hdr = fh.read(8)
                if len(hdr) < 8:
                    break
                cid = hdr[:4]
                clen = int.from_bytes(hdr[4:8], "little")
                if cid == b"fmt ":
                    fmt = fh.read(min(clen, 16))
                    if len(fmt) >= 16:
                        audio_fmt = int.from_bytes(fmt[0:2], "little")
                        channels = int.from_bytes(fmt[2:4], "little")
                        rate = int.from_bytes(fmt[4:8], "little")
                        bits = int.from_bytes(fmt[14:16], "little")
                        fmt_ok = (audio_fmt == 1 and bits == 16 and channels >= 1)
                elif cid == b"data":
                    data_off = pos + 8
                    data_len = min(clen, size - data_off)
                pos += 8 + clen + (clen & 1)     # chunks are word-aligned
            # Mono only: a strided every-Nth-sample view of a multichannel
            # memmap silently COPIES on reshape — exactly the RAM spike this
            # path exists to avoid. Multichannel falls to the streaming path.
            if not (fmt_ok and channels == 1 and data_off and data_len >= 2
                    and rate > 0):
                return None
            n = data_len // 2
            return (np.memmap(path, dtype=np.int16, mode="r",
                              offset=data_off, shape=(n,)),
                    channels, rate)
    except Exception:  # noqa: BLE001 — fall back to the ffmpeg path
        return None


def generate_peaks(audio_path: str, out_dir: str, n_peaks: int = _PEAKS_N) -> Optional[dict]:
    """Build ``peaks.json`` (interleaved min/max, ~n_peaks pairs).

    ``audio_path`` may be the job's pre-extracted ``audio.wav`` (binned via a
    zero-copy memmap — no subprocess at all) or the source video itself
    (FFmpeg decode STREAMED in chunks; the full PCM of a 2-hour source is
    ~250 MB and must never sit in RAM at once). Returns the payload dict on
    success, ``None`` on failure.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return None
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        logger.warning("numpy unavailable for peaks: %s", e)
        return None

    mins_l: list = []
    maxs_l: list = []
    total_samples = 0
    rate = _PEAKS_SR

    wav = _wav_pcm16_memmap(audio_path) if audio_path.lower().endswith(".wav") else None
    if wav is not None:
        pcm, channels, rate = wav
        if channels > 1:                     # interleaved → first channel
            pcm = pcm[::channels]
        total_samples = int(pcm.size)
        if total_samples == 0:
            return None
        spp = max(1, total_samples // n_peaks)
        usable = (total_samples // spp) * spp
        if usable == 0:
            return None
        bins = pcm[:usable].reshape(-1, spp)
        mins_arr = bins.min(axis=1).astype(np.float32) / 32768.0
        maxs_arr = bins.max(axis=1).astype(np.float32) / 32768.0
    else:
        # Streaming decode: probe the duration to pick the bin size up front,
        # then fold each chunk into min/max bins as it arrives.
        try:
            probe = _run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", audio_path],
                timeout=30,
            )
            est_dur = float((probe.stdout or b"0").strip() or 0)
        except Exception:
            est_dur = 0.0
        est_samples = int(est_dur * _PEAKS_SR)
        spp = max(1, est_samples // n_peaks) if est_samples else _PEAKS_SR // 10
        carry = b""
        proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", audio_path,
             "-vn", "-acodec", "pcm_s16le",
             "-ar", str(_PEAKS_SR), "-ac", "1",
             "-f", "s16le", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        try:
            while True:
                chunk = proc.stdout.read(_PEAKS_CHUNK)
                if not chunk:
                    break
                buf = carry + chunk
                n_full = (len(buf) // 2 // spp) * spp     # whole bins only
                if n_full:
                    arr = np.frombuffer(buf[:n_full * 2], dtype=np.int16)
                    total_samples += arr.size
                    b = arr.reshape(-1, spp)
                    mins_l.append(b.min(axis=1))
                    maxs_l.append(b.max(axis=1))
                    carry = buf[n_full * 2:]
                else:
                    carry = buf
            if len(carry) >= 2:                            # trailing partial bin
                arr = np.frombuffer(carry[:(len(carry) // 2) * 2], dtype=np.int16)
                if arr.size:
                    total_samples += arr.size
                    mins_l.append(np.array([arr.min()], dtype=np.int16))
                    maxs_l.append(np.array([arr.max()], dtype=np.int16))
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            rc = proc.wait(timeout=900)
        if rc != 0 and total_samples == 0:
            logger.warning("peaks ffmpeg failed (rc=%s)", rc)
            return None
        if total_samples == 0:
            return None
        mins_arr = np.concatenate(mins_l).astype(np.float32) / 32768.0
        maxs_arr = np.concatenate(maxs_l).astype(np.float32) / 32768.0
        rate = _PEAKS_SR

    data = np.empty(mins_arr.size * 2, dtype=np.float32)
    data[0::2] = mins_arr
    data[1::2] = maxs_arr

    payload = {
        "version": 2,
        "channels": 1,
        "sample_rate": int(rate),
        "samples_per_pixel": int(spp),
        "bits": 16,
        "length": int(mins_arr.size),
        "duration": round(total_samples / float(rate), 3),
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
    logger.info("peaks written: %d bins → %s", mins_arr.size, peaks_path)
    return payload
