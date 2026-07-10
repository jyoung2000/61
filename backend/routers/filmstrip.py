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


def _prepare_assets(job_id: str) -> None:
    """One-time editor prep for a job whose assets predate this feature.

    New jobs get faststart at ingest + sprite/peaks at analysis. Jobs analyzed
    before this shipped have none of it, so the FIRST time the editor opens one
    we (in the background, best-effort) (1) relocate the moov atom so the video
    streams from the first bytes, (2) build the thumbnail sprite, (3) build the
    waveform peaks. Everything is idempotent and skips work already done, so
    re-opening a fully-prepped job is a cheap set of existence checks.
    """
    job_dir = _job_dir(job_id)
    src = _source_video(job_id)
    if not src:
        return

    # 1) COARSE thumbnail sprite first — seek-sampled, seconds even on a
    #    2-hour source, and it must not wait behind the faststart remux of a
    #    multi-GB file. The editor gets a filmstrip almost immediately.
    _had_sprite = os.path.isfile(os.path.join(job_dir, "sprite.jpg"))
    if not _had_sprite:
        try:
            from backend.services.filmstrip_generator import generate_sprite_coarse
            generate_sprite_coarse(src, job_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] lazy coarse sprite failed: %s", job_id, e)

    # 2) Faststart the source in place — the single biggest video-load win for
    #    a long, non-faststart file (moov currently trails mdat → the browser
    #    must pull the whole file before it can play or seek).
    try:
        from backend.config import settings
        if getattr(settings, "FFMPEG_FASTSTART", True):
            from backend.services.faststart import ensure_faststart
            ensure_faststart(src)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] lazy faststart failed: %s", job_id, e)

    # 3) FINE sprite — the whole-file keyframe scan (minutes on long
    #    sources); replaces the coarse sheet, the editor swaps it in live.
    if not _had_sprite:
        try:
            from backend.services.filmstrip_generator import generate_sprite
            generate_sprite(src, job_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] lazy sprite failed: %s", job_id, e)
    else:
        # A sprite exists but may be a leftover COARSE sheet whose fine pass
        # died (container restart mid-generation). Finish the upgrade.
        try:
            import json as _json
            with open(os.path.join(job_dir, "sprite.json")) as f:
                if _json.load(f).get("coarse"):
                    from backend.services.filmstrip_generator import generate_sprite
                    generate_sprite(src, job_dir)
        except Exception:
            pass

    # 4) Waveform peaks (prefer the pre-extracted audio.wav over re-decoding).
    if not os.path.isfile(os.path.join(job_dir, "peaks.json")):
        try:
            from backend.services.filmstrip_generator import generate_peaks
            audio = os.path.join(job_dir, "audio.wav")
            generate_peaks(audio if os.path.isfile(audio) else src, job_dir)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] lazy peaks failed: %s", job_id, e)


def _kick_prep(job_id: str) -> None:
    """Kick a one-time background prep for a job, deduped so concurrent editor
    requests (filmstrip.json + waveform.json fire together) share one run."""
    key = f"{job_id}:prep"
    if key in _GENERATING:
        return
    _GENERATING.add(key)

    async def _runner():
        try:
            await asyncio.to_thread(_prepare_assets, job_id)
        except Exception as e:  # noqa: BLE001 — best effort
            logger.warning("editor asset prep failed (%s): %s", key, e)
        finally:
            _GENERATING.discard(key)

    try:
        asyncio.get_running_loop().create_task(_runner())
    except RuntimeError:
        _GENERATING.discard(key)


def _conditional_file(path: str, media_type: str, if_modified_since: str | None,
                      cache_control: str = "public, max-age=2592000"):
    """FileResponse with 304 handling. The default Cache-Control is a long
    max-age (the sprite image is cache-busted by ``?v=`` from the manifest);
    the MANIFEST itself must pass ``no-cache`` so the browser revalidates and
    picks up the coarse→fine sprite upgrade — a 30-day max-age there froze
    the first sheet the client ever saw."""
    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    if if_modified_since:
        try:
            if parsedate_to_datetime(if_modified_since) >= mtime:
                return Response(status_code=304)
        except Exception:
            pass
    headers = {
        "Cache-Control": cache_control,
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
        # A served coarse manifest means the fine pass may still be pending —
        # make sure it's (re)kicked even though assets "exist".
        try:
            import json as _json
            with open(manifest) as f:
                if _json.load(f).get("coarse"):
                    _kick_prep(job_id)
        except Exception:
            pass
        return _conditional_file(manifest, "application/json",
                                 if_modified_since, cache_control="no-cache")
    # Not ready — prep faststart+sprite+peaks in the background, tell the editor
    # to fall back to client-side generation meanwhile.
    _kick_prep(job_id)
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
    # Not ready — prep faststart+sprite+peaks in the background (shares one run
    # with the sprite request via the dedupe key).
    _kick_prep(job_id)
    raise HTTPException(status_code=404, detail="peaks not ready")
