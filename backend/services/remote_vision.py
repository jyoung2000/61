"""Remote vision inference on the GPU Companion (face-detection offload).

The FACES stage is the pipeline's largest block and runs YOLO-World on the
small server GPU. The Companion's vision sidecar (``/v1/vision/detect``
behind the same authenticated proxy whisper uses) runs the identical model
on the big card, ~2-3× faster. This client is deliberately conservative:

  * OFF unless the Companion answers ``/v1/vision/health`` (cached probe) —
    an older Companion 404s and everything silently stays local.
  * Every call falls back to LOCAL inference on any error, and a breaker
    disables the remote for the rest of the run after
    ``REMOTE_VISION_BREAKER_FAILS`` consecutive failures — a mid-run
    Companion crash degrades to exactly today's behavior.
  * The response is mapped into a tiny ultralytics-shaped shim (``.boxes``
    with ``.cls``/``.conf``/``.xyxy``), so downstream consumers are
    byte-identical between local and remote paths.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Optional

import numpy as np

from backend.config import settings

logger = logging.getLogger("clipai.remote_vision")


class _Box:
    __slots__ = ("cls", "conf", "xyxy")

    def __init__(self, cls_id: int, conf: float, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.asarray(xyxy, dtype=np.float32)]


class _Boxes(list):
    """Iterable of _Box; truthiness matches ultralytics (empty → falsy)."""


class _Result:
    __slots__ = ("boxes",)

    def __init__(self, boxes: _Boxes):
        self.boxes = boxes if boxes else None


class RemoteVisionDetector:
    """One per FaceDetector run. Not thread-safe beyond the GIL (the
    perceiver detects on a single consumer thread)."""

    def __init__(self):
        self._fails = 0
        self._disabled = False
        self._healthy_at = 0.0
        self._healthy = False
        self._client = None

    # ── availability ──────────────────────────────────────────────────
    def _base(self) -> str:
        try:
            from backend.services.reframer_audio import _remote_whisper_base
            return _remote_whisper_base() or ""
        except Exception:
            return ""

    def _headers(self) -> dict:
        h = {}
        try:
            from backend.services.reframer_audio import _remote_whisper_token
            tok = _remote_whisper_token()
            if tok:
                h["Authorization"] = f"Bearer {tok}"
        except Exception:
            pass
        return h

    def available(self) -> bool:
        if self._disabled or not bool(getattr(settings, "REMOTE_VISION_ENABLED", True)):
            return False
        base = self._base()
        if not base:
            return False
        now = time.monotonic()
        if now - self._healthy_at < 60.0:
            return self._healthy
        self._healthy_at = now
        try:
            import httpx
            r = httpx.get(f"{base}/v1/vision/health",
                          headers=self._headers(), timeout=3.0)
            self._healthy = (r.status_code == 200)
            if self._healthy:
                logger.info("Remote vision sidecar available at %s — "
                            "face detection offloads to the Companion GPU", base)
        except Exception:
            self._healthy = False
        return self._healthy

    def _fail(self, why: str) -> None:
        self._fails += 1
        limit = int(getattr(settings, "REMOTE_VISION_BREAKER_FAILS", 5))
        if self._fails >= limit and not self._disabled:
            self._disabled = True
            logger.warning(
                "Remote vision disabled for this run after %d consecutive "
                "failures (%s) — face detection continues LOCALLY", self._fails, why)

    # ── inference ─────────────────────────────────────────────────────
    def predict(self, frame_bgr, classes: list, conf: float = 0.25,
                max_det: int = 20) -> Optional[list]:
        """Remote detect one frame. Returns ultralytics-shaped results, or
        ``None`` (caller runs local inference) on ANY problem."""
        try:
            import cv2
            import httpx
            ok, jpg = cv2.imencode(".jpg", frame_bgr,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 87])
            if not ok:
                return None
            if self._client is None:
                self._client = httpx.Client(
                    timeout=float(getattr(settings, "REMOTE_VISION_TIMEOUT_S", 10.0)))
            r = self._client.post(
                f"{self._base()}/v1/vision/detect",
                headers=self._headers(),
                json={
                    "image_b64": base64.b64encode(jpg.tobytes()).decode("ascii"),
                    "classes": list(classes or []),
                    "conf": float(conf),
                    "max_det": int(max_det),
                })
            if r.status_code != 200:
                self._fail(f"HTTP {r.status_code}")
                return None
            payload = r.json() or {}
            boxes = _Boxes()
            for b in payload.get("boxes") or []:
                try:
                    boxes.append(_Box(int(b["cls"]), float(b["conf"]),
                                      [float(x) for x in b["xyxy"]]))
                except (KeyError, TypeError, ValueError):
                    continue
            self._fails = 0
            return [_Result(boxes)]
        except Exception as e:
            self._fail(f"{type(e).__name__}: {str(e)[:80]}")
            return None

    def close(self) -> None:
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._client = None
