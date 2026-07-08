"""Editor filmstrip sprite + waveform peaks endpoints.

Serves the precomputed scrubbing assets the NLE timeline uses instead of
seeking a hidden ``<video>`` in the browser:

  * ``GET /api/jobs/{job_id}/filmstrip.json`` — sprite manifest
  * ``GET /api/jobs/{job_id}/filmstrip.jpg``  — sprite sheet image
  * ``GET /api/jobs/{job_id}/waveform.json``  — audio peaks array

Newly-analyzed jobs have these written at analysis time (see
:mod:`backend.services.pipeline`). For jobs analyzed before this shipped —
or if precompute failed — the first request kicks off a one-time background
build and returns 404; the editor falls back to client-side generation
meanwhile and picks up the server assets on a later open. Nothing here ever
blocks a request on a cold multi-minute decode.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from email.utils import format_datetime, parsedate_to_datetime

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, Response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["filmstrip"])

_JOB_ROOT = "/data/uploads"

# Jobs currently generating an asset, so concurrent editor requests don't
# spawn duplicate FFmpeg passes. Keyed by "{job_id}:sprite" / "{job_id}:peaks".
_GENERATING: set[str] = set()


def _safe_job_id(job_id: str) -> str:
    if not job_id or not all(c.isalnum() or c in "-_" for c in job_id):
        raise HTTPException(status_code=400, detail="invalid job_id")
    return job_id


def _job_dir(job_id: str) -> str:
    return os.path.join(_JOB_ROOT, job_id)


def _source_video(job_id: str) -> str | None:
    """Best-effort source path for a job (``video.<ext>`` in the job dir)."""
    job_dir = _job_dir(job_id)
    if not os.path.isdir(job_dir):
        return None
    for name in os.listdir(job_dir):
        if name.startswith("video."):
            return os.path.join(job_dir, name)
    return None


def _kick_background(key: str, fn, *args) -> None:
    """Run a best-effort generation once, in a worker thread, deduped by key."""
    if key in _GENERATING:
        return
    _GENERATING.add(key)

    async def _runner():
        try:
            await asyncio.to_thread(fn, *args)
        except Exception as e:  # noqa: BLE001 — best effort
            logger.warning("filmstrip background gen failed (%s): %s", key, e)
        finally:
            _GENERATING.discard(key)

    try:
        asyncio.get_running_loop().create_task(_runner())
    except RuntimeError:
        _GENERATING.discard(key)


def _conditional_file(path: str, media_type: str, if_modified_since: str | None):
    """FileResponse with 304 handling + aggressive immutable caching."""
    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    if if_modified_since:
        try:
            if parsedate_to_datetime(if_modified_since) >= mtime:
                return Response(status_code=304)
        except Exception:
            pass
    headers = {
        "Cache-Control": "public, max-age=2592000",  # 30 days; content is content-addressed by job
        "Last-Modified": format_datetime(mtime, usegmt=True),
        "X-Content-Type-Options": "nosniff",
    }
    return FileResponse(path=path, media_type=media_type, headers=headers)


@router.get("/jobs/{job_id}/filmstrip.json")
async def get_filmstrip_manifest(
    job_id: str,
    if_modified_since: str = Header(None, alias="If-Modified-Since"),
):
    job_id = _safe_job_id(job_id)
    manifest = os.path.join(_job_dir(job_id), "sprite.json")
    sprite = os.path.join(_job_dir(job_id), "sprite.jpg")
    if os.path.isfile(manifest) and os.path.isfile(sprite):
        return _conditional_file(manifest, "application/json", if_modified_since)
    # Not ready — build in the background, tell the editor to fall back.
    src = _source_video(job_id)
    if src:
        from backend.services.filmstrip_generator import generate_sprite
        _kick_background(f"{job_id}:sprite", generate_sprite, src, _job_dir(job_id))
    raise HTTPException(status_code=404, detail="sprite not ready")


@router.get("/jobs/{job_id}/filmstrip.jpg")
async def get_filmstrip_sprite(
    job_id: str,
    if_modified_since: str = Header(None, alias="If-Modified-Since"),
):
    job_id = _safe_job_id(job_id)
    sprite = os.path.join(_job_dir(job_id), "sprite.jpg")
    if os.path.isfile(sprite):
        return _conditional_file(sprite, "image/jpeg", if_modified_since)
    raise HTTPException(status_code=404, detail="sprite not ready")


@router.get("/jobs/{job_id}/waveform.json")
async def get_waveform_peaks(
    job_id: str,
    if_modified_since: str = Header(None, alias="If-Modified-Since"),
):
    job_id = _safe_job_id(job_id)
    peaks = os.path.join(_job_dir(job_id), "peaks.json")
    if os.path.isfile(peaks):
        return _conditional_file(peaks, "application/json", if_modified_since)
    # Not ready — prefer the pre-extracted audio.wav, fall back to the source.
    audio = os.path.join(_job_dir(job_id), "audio.wav")
    audio_src = audio if os.path.isfile(audio) else _source_video(job_id)
    if audio_src:
        from backend.services.filmstrip_generator import generate_peaks
        _kick_background(f"{job_id}:peaks", generate_peaks, audio_src, _job_dir(job_id))
    raise HTTPException(status_code=404, detail="peaks not ready")
