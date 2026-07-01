"""Optional MediaPipe Face Landmarker for a true lips-based mouth-open signal.

The baseline mouth-motion signal is either a YuNet 5-point MAR *delta* or a
pixel-diff on the lower-third face ROI. Both are proxies: the pixel-diff in
particular fires on head bob, beard and lighting change, which corrupts
"who is talking" and therefore speaker selection.

MediaPipe's Face Landmarker returns 478 dense points including the inner-lip
contour, so we can measure an actual **mouth aspect ratio** (vertical lip gap /
mouth width) — a direct open/close signal. It runs on CPU (~2-3 ms/face) and
never touches the GPU budget.

This module is entirely optional and defensive:
  * import of ``mediapipe`` is lazy and wrapped;
  * the ``.task`` model is looked up at a configurable path (or a couple of
    conventional locations) and, if missing, the detector disables itself;
  * every public call returns ``None`` on any failure, so the perceiver simply
    falls back to the existing MAR/pixel-diff path.

Gated behind ``settings.REFRAMER_MEDIAPIPE_MAR`` at the call site.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Inner-lip landmark indices in the 478-point FaceMesh topology.
_UPPER_LIP = 13     # center of the upper inner lip
_LOWER_LIP = 14     # center of the lower inner lip
_MOUTH_LEFT = 78    # left mouth corner (inner)
_MOUTH_RIGHT = 308  # right mouth corner (inner)


def _candidate_model_paths() -> list:
    configured = ""
    try:
        from backend.config import settings
        configured = str(getattr(settings, "REFRAMER_MEDIAPIPE_MODEL_PATH", "") or "")
    except Exception:
        configured = ""
    env = os.environ.get("CLIPAI_MEDIAPIPE_FACE_LANDMARKER", "")
    paths = [configured, env,
             "/data/models/face_landmarker.task",
             os.path.join(os.path.dirname(__file__), "..", "..",
                          "data", "models", "face_landmarker.task")]
    return [p for p in paths if p]


class _MediaPipeMouth:
    """Lazy singleton wrapper around a MediaPipe FaceLandmarker."""

    _instance: "Optional[_MediaPipeMouth]" = None
    _disabled: bool = False

    def __init__(self) -> None:
        self._landmarker = None
        self._mp = None
        self._ready = False
        self._init_landmarker()

    @classmethod
    def get(cls) -> "Optional[_MediaPipeMouth]":
        if cls._disabled:
            return None
        if cls._instance is None:
            try:
                cls._instance = _MediaPipeMouth()
            except Exception as exc:  # pragma: no cover - env-specific
                logger.info("MediaPipe landmarker unavailable: %s", exc)
                cls._disabled = True
                return None
        return cls._instance if cls._instance._ready else None

    def _init_landmarker(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:
            logger.info("mediapipe not importable (%s); MAR upgrade disabled", exc)
            _MediaPipeMouth._disabled = True
            return
        model_path = next((p for p in _candidate_model_paths() if os.path.exists(p)), None)
        if not model_path:
            logger.info(
                "face_landmarker.task not found in %s; MAR upgrade disabled",
                _candidate_model_paths())
            _MediaPipeMouth._disabled = True
            return
        try:
            base = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
            )
            self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            self._ready = True
        except Exception as exc:  # pragma: no cover - env-specific
            logger.info("Failed to create MediaPipe FaceLandmarker: %s", exc)
            _MediaPipeMouth._disabled = True

    def mouth_open_ratio(self, face_bgr) -> Optional[float]:
        """Return mouth-open ratio (lip gap / mouth width) for a face crop.

        ``face_bgr`` is a tight BGR crop around one face. Returns ``None`` when
        no landmarks are produced.
        """
        if not self._ready or self._landmarker is None:
            return None
        try:
            import numpy as np
            mp = self._mp
            rgb = face_bgr[:, :, ::-1].copy()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self._landmarker.detect(mp_image)
            if not res.face_landmarks:
                return None
            lm = res.face_landmarks[0]
            up = lm[_UPPER_LIP]
            lo = lm[_LOWER_LIP]
            ml = lm[_MOUTH_LEFT]
            mr = lm[_MOUTH_RIGHT]
            gap = abs(lo.y - up.y)
            width = abs(mr.x - ml.x)
            if width <= 1e-6:
                return None
            return float(gap / width)  # normalized MAR, ~0 closed, ~0.5+ open
        except Exception:
            return None


def mouth_open_ratio(face_bgr) -> Optional[float]:
    """Module-level convenience: mouth-open ratio or ``None`` if unavailable."""
    inst = _MediaPipeMouth.get()
    if inst is None:
        return None
    return inst.mouth_open_ratio(face_bgr)
