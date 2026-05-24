"""API endpoint that serves the per-job detection overlay JSON.

The reframer pipeline writes ``detection_overlay.json`` under the
job's upload directory (see ``backend.services.pipeline``). The React
``ReframePreview`` component fetches it once per job to paint face,
subject, motion, and speech indicators on top of the source video.
"""

import json
import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend import database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["detection_overlay"])


def _overlay_path(job_id: str) -> str:
    return os.path.join("/data/uploads", job_id, "detection_overlay.json")


def _sparse_filter(timeline: dict, stride: int) -> dict:
    if not isinstance(timeline, dict) or stride <= 1:
        return timeline
    keys = sorted(timeline.keys(), key=lambda k: int(k))
    return {k: timeline[k] for i, k in enumerate(keys) if i % stride == 0}


@router.get("/jobs/{job_id}/detection_overlay")
async def get_detection_overlay(
    job_id: str,
    sparse: bool = Query(False, description="Downsample timelines for a smaller payload"),
):
    """Return the detection overlay JSON for a finished job.

    Pass ``sparse=true`` to keep every fourth sample — handy for an
    initial render on slow connections; the client can re-fetch the
    full payload when the user pauses on a frame.
    """
    # Make sure the job actually exists so we don't leak filesystem layout.
    job = await database.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    path = _overlay_path(job_id)
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail="Detection overlay not available for this job",
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[%s] detection_overlay read failed: %s", job_id, e)
        raise HTTPException(
            status_code=500,
            detail="Detection overlay sidecar is corrupt",
        )

    if sparse:
        for key in (
            "face_timeline", "person_timeline", "motion_timeline",
            "speech_active", "saliency_hotspot",
        ):
            if key in data:
                data[key] = _sparse_filter(data[key], 4)

    return data
