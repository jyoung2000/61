"""Optional u2netp learned salient-object model for no-face framing.

Spectral residual predicts pixels that *stand out* (edges, logos, text). What
the reframer actually wants on a faceless frame is *where a human looks* — a
different, learnable thing. U²-Netₚ (``u2netp``) is a 4.7 MB salient-object
model that outputs a clean subject **mask**; its centroid is a far better crop
target than a spectral edge peak.

Device policy (``REFRAMER_U2NET_DEVICE``: auto | cuda | cpu):
``auto`` runs the model on the GPU through onnxruntime's
CUDAExecutionProvider when the GPU wheel is installed and free VRAM clears
``REFRAMER_U2NET_GPU_MIN_FREE_MB`` — the profiled anime run spent the bulk
of its 612 s per-sample "other" time on these CPU forwards (~0.4-0.5 s each;
~5-15 ms on the GPU), and with remote Whisper/Ollama the local card is idle
during the face loop anyway. Any CUDA load/inference failure permanently
drops to the classic CPU path: OpenCV DNN (``cv2.dnn.readNetFromONNX``),
which needs no dependency beyond the opencv the project already ships.

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
        self._ort = None
        self._ort_input = ""
        self._input = 320  # u2netp native input size
        self._model_path = model_path
        self.device = "cpu"
        if cv2 is None:
            raise RuntimeError("cv2 unavailable")
        if self._should_try_cuda():
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in (ort.get_available_providers() or []):
                    sess = ort.InferenceSession(
                        model_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
                    # Only claim CUDA if the session actually bound it (ORT
                    # silently falls back to CPU when the CUDA EP can't init).
                    if "CUDAExecutionProvider" in sess.get_providers():
                        self._ort = sess
                        self._ort_input = sess.get_inputs()[0].name
                        self.device = "cuda"
                        logger.info(
                            "u2netp saliency device: cuda "
                            "(onnxruntime CUDAExecutionProvider)")
            except Exception as exc:
                logger.info(
                    "u2netp CUDA path unavailable (%s) — using OpenCV CPU", exc)
        if self._ort is None:
            self._load_cpu_net()
            logger.info("u2netp saliency device: cpu (OpenCV DNN)")

    @staticmethod
    def _should_try_cuda() -> bool:
        """Device policy: 'cuda' forces the attempt, 'cpu' forbids it, 'auto'
        (default) tries CUDA when free VRAM clears the floor (u2netp is a
        4.7 MB model — the workspace is small, but a card already pinned by
        Whisper shouldn't take even that)."""
        try:
            from backend.config import settings
            policy = str(getattr(settings, "REFRAMER_U2NET_DEVICE", "auto") or "auto").lower()
        except Exception:
            policy = "auto"
        if policy == "cpu":
            return False
        if policy in ("cuda", "gpu"):
            return True
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
            floor = 300.0
            try:
                from backend.config import settings
                floor = float(getattr(settings, "REFRAMER_U2NET_GPU_MIN_FREE_MB", 300))
            except Exception:
                pass
            return free_mb >= floor
        except Exception:
            # No torch to ask — let onnxruntime try; failure falls back to CPU.
            return True

    def _load_cpu_net(self) -> None:
        net = cv2.dnn.readNetFromONNX(self._model_path)
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
        if (self._net is None and self._ort is None) or cv2 is None:
            return None
        try:
            blob = cv2.dnn.blobFromImage(
                bgr, scalefactor=1.0 / 255.0,
                size=(self._input, self._input),
                mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                swapRB=True, crop=False)
            if self._ort is not None:
                try:
                    out = self._ort.run(None, {self._ort_input: blob})[0]
                except Exception as exc:
                    # A mid-run CUDA fault (OOM, driver hiccup) permanently
                    # drops this process to the CPU path — same outputs.
                    logger.warning(
                        "u2netp CUDA inference failed (%s) — falling back to "
                        "OpenCV CPU for the rest of the run", exc)
                    self._ort = None
                    self.device = "cpu"
                    if self._net is None:
                        self._load_cpu_net()
                    self._net.setInput(blob)
                    out = self._net.forward()
            else:
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
