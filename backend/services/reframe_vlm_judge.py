"""VLM spot-check — a human-proxy framing score for the reframe report.

The evaluator's metrics are geometric proxies (face center inside crop,
safe-area insets). This module samples a handful of frames from the exported
clips WITH the crop applied and asks the local vision model the question a
human reviewer would: "is the main subject well-framed?" The yes-rate becomes
``vlm_framing_pct`` on the reframe report — a calibration signal for the
proxy metrics, not a replacement.

Fail-soft by design: no Ollama, no vision model, ffmpeg failure, or fewer
than 4 usable answers → returns None and the report simply omits the score.
Cost-bounded: ``REFRAME_VLM_FRAMES`` frames (default 12), one short
generation each, hard wall-clock budget.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import subprocess
import tempfile
import time as _time
from typing import Optional

from backend.config import settings

logger = logging.getLogger("clipai.reframe_vlm_judge")

_PROMPT = (
    "This image is a vertical crop taken from a video for social media. "
    "Judge the framing: is the main subject (the person, face, or focus of "
    "the action) clearly visible inside the crop and not awkwardly cut off "
    "at the left or right edge? Answer with exactly one word: YES or NO."
)


def _sample_times(clips: list, max_frames: int) -> list[float]:
    """Evenly spread sample timestamps across the exported clips."""
    windows = []
    for c in clips or []:
        start = float((c.get("start_time") if isinstance(c, dict)
                       else getattr(c, "start_time", 0)) or 0)
        end = float((c.get("end_time") if isinstance(c, dict)
                     else getattr(c, "end_time", 0)) or 0)
        if end - start >= 2.0:
            windows.append((start, end))
    if not windows:
        return []
    total = sum(e - s for s, e in windows)
    times: list[float] = []
    for s, e in windows:
        n = max(1, round((e - s) / total * max_frames))
        for i in range(n):
            times.append(s + (e - s) * (i + 0.5) / n)
    times.sort()
    return times[:max_frames]


def _extract_cropped_frame(video_path: str, t: float, crop_w: int, crop_h: int,
                           x: int, y: int, out_path: str) -> bool:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{max(0.0, t):.3f}", "-i", video_path,
             "-vf", f"crop={crop_w}:{crop_h}:{x}:{y},scale=360:-2",
             "-frames:v", "1", "-q:v", "5", out_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30, check=False,
        )
        return proc.returncode == 0 and os.path.getsize(out_path) > 0
    except Exception:
        return False


async def _ask_vlm(client, model: str, image_b64: str) -> Optional[bool]:
    """One YES/NO framing judgment. None on failure/ambiguity."""
    try:
        from backend.services import ollama_registry
        _url = f"{settings.OLLAMA_HOST.rstrip('/')}/api/generate"
        resp = await client.post(
            _url,
            headers=ollama_registry.headers_for_url(_url),
            json={
                "model": model,
                "prompt": _PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 8},
            },
            timeout=60.0,
        )
        if resp.status_code != 200:
            return None
        answer = (resp.json().get("response") or "").strip().upper()
        if answer.startswith("YES"):
            return True
        if answer.startswith("NO"):
            return False
        return None
    except Exception:
        return None


async def vlm_spotcheck_framing(video_path: str, plan, clips: list,
                                max_frames: Optional[int] = None,
                                budget_s: float = 300.0) -> Optional[float]:
    """Return the %% of sampled cropped frames the VLM judges well-framed.

    None when the check can't run (no flag/host/model/frames) or produced
    fewer than 4 usable answers — callers treat None as "not measured".
    """
    if not bool(getattr(settings, "REFRAME_VLM_SPOTCHECK", True)):
        return None
    host = (getattr(settings, "OLLAMA_HOST", "") or "").strip()
    model = (getattr(settings, "OLLAMA_PRIMARY_MODEL", "") or "").strip()
    if not host or not model:
        return None
    n_frames = int(max_frames if max_frames is not None
                   else getattr(settings, "REFRAME_VLM_FRAMES", 12))
    if n_frames <= 0:
        return None

    times = _sample_times(clips, n_frames)
    if not times:
        return None

    try:
        from backend.services.reframer_models import clamp_x, interpolate_x
    except Exception:
        return None
    keyframes = getattr(plan, "keyframes", None) or []
    crop_w = int(getattr(plan, "crop_w", 0) or 0)
    crop_h = int(getattr(plan, "crop_h", 0) or 0)
    crop_y = int(getattr(plan, "crop_y", 0) or 0)
    max_x = getattr(plan, "max_x", 0)
    if crop_w <= 0 or crop_h <= 0:
        return None

    import httpx
    t0 = _time.monotonic()
    yes = no = 0
    async with httpx.AsyncClient() as client:
        with tempfile.TemporaryDirectory(prefix="vlm_spot_") as tmpdir:
            for i, t in enumerate(times):
                if _time.monotonic() - t0 > budget_s:
                    logger.info("VLM spot-check budget reached after %d frame(s)",
                                yes + no)
                    break
                x = clamp_x(interpolate_x(keyframes, int(t * 1000)), max_x)
                frame_path = os.path.join(tmpdir, f"f_{i:03d}.jpg")
                ok = await asyncio.to_thread(
                    _extract_cropped_frame, video_path, t,
                    crop_w, crop_h, int(x), crop_y, frame_path)
                if not ok:
                    continue
                with open(frame_path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("ascii")
                verdict = await _ask_vlm(client, model, b64)
                if verdict is True:
                    yes += 1
                elif verdict is False:
                    no += 1

    answered = yes + no
    if answered < 4:
        return None
    pct = round(yes / answered * 100.0, 1)
    logger.info("VLM spot-check: %d/%d sampled frames judged well-framed "
                "(%.1f%%, model=%s)", yes, answered, pct, model)
    return pct
