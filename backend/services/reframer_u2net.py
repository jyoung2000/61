"""Optional u2netp learned salient-object model for no-face framing.

Spectral residual predicts pixels that *stand out* (edges, logos, text). What
the reframer actually wants on a faceless frame is *where a human looks* — a
different, learnable thing. U²-Netₚ (``u2netp``) is a 4.7 MB salient-object
model that outputs a clean subject **mask**; its centroid is a far better crop
target than a spectral edge peak.

It runs on CPU through the OpenCV DNN backend (``cv2.dnn.readNetFromONNX``), so
it needs no extra Python dependency beyond the opencv the project already
ships, and never touches the 4 GB GPU budget.

Entirely optional and defensive: gated behind ``settings.REFRAMER_U2NET_SALIENCY``
with the model path in ``settings.REFRAMER_U2NET_MODEL_PATH``. Any failure
(missing model, bad file, inference error) disables the path and the caller
falls back to the fused spectral stack.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)


class _U2NetSaliency:
    _instance: "Optional[_U2NetSaliency]" = None
    _disabled: bool = False

    def __init__(self, model_path: str) -> None:
        self._net = None
        self._input = 320  # u2netp native input size
        if cv2 is None:
            raise RuntimeError("cv2 unavailable")
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._net = net

    @classmethod
    def get(cls) -> "Optional[_U2NetSaliency]":
        if cls._disabled:
            return None
        if cls._instance is not None:
            return cls._instance
        try:
            from backend.config import settings
            path = str(getattr(settings, "REFRAMER_U2NET_MODEL_PATH", "") or "")
        except Exception:
            path = ""
        if not path:
            path = os.environ.get("CLIPAI_U2NET_MODEL_PATH", "")
        if not path:
            # Auto-discover the image-baked copy (audit Phase 5.4 — the
            # Dockerfiles now pre-bake u2netp.onnx so learned saliency
            # works out of the box instead of requiring a manual path).
            for candidate in (
                os.path.join(os.path.dirname(__file__), "..", "models", "u2netp.onnx"),
                "/app/backend/models/u2netp.onnx",
                "/data/models/u2netp.onnx",
            ):
                candidate = os.path.abspath(candidate)
                if os.path.exists(candidate):
                    path = candidate
                    break
        if not path or not os.path.exists(path):
            logger.info("u2netp model not found (path=%r); learned saliency disabled", path)
            cls._disabled = True
            return None
        try:
            cls._instance = _U2NetSaliency(path)
            return cls._instance
        except Exception as exc:  # pragma: no cover - env-specific
            logger.info("Failed to load u2netp: %s", exc)
            cls._disabled = True
            return None

    def saliency(self, bgr, out_size: int) -> Optional[np.ndarray]:
        """Return a normalized [0,1] saliency map resized to ``out_size``."""
        if self._net is None or cv2 is None:
            return None
        try:
            blob = cv2.dnn.blobFromImage(
                bgr, scalefactor=1.0 / 255.0,
                size=(self._input, self._input),
                mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                swapRB=True, crop=False)
            self._net.setInput(blob)
            out = self._net.forward()
            # u2netp returns (1,1,H,W); squeeze and normalize.
            m = np.array(out).squeeze().astype(np.float32)
            if m.ndim != 2:
                return None
            mn, mx = float(m.min()), float(m.max())
            if mx - mn < 1e-6:
                return None
            m = (m - mn) / (mx - mn)
            return cv2.resize(m, (out_size, out_size), interpolation=cv2.INTER_AREA)
        except Exception:
            return None


def u2net_saliency(bgr, out_size: int) -> Optional[np.ndarray]:
    """Module-level convenience: [0,1] saliency map or ``None`` if unavailable."""
    inst = _U2NetSaliency.get()
    if inst is None:
        return None
    return inst.saliency(bgr, out_size)
