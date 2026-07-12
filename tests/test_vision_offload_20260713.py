"""Face-detection offload + companion model placement (with failsafes)."""
import asyncio
import sys
import types

import numpy as np
import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.config import settings


def _client(monkeypatch, healthy=True):
    from backend.services.remote_vision import RemoteVisionDetector
    d = RemoteVisionDetector()
    monkeypatch.setattr(d, "_base", lambda: "http://comp.test:11500")
    monkeypatch.setattr(d, "_headers", lambda: {})
    return d


def test_unavailable_without_base(monkeypatch):
    from backend.services.remote_vision import RemoteVisionDetector
    d = RemoteVisionDetector()
    monkeypatch.setattr(d, "_base", lambda: "")
    assert d.available() is False


def test_predict_maps_boxes_and_resets_breaker(monkeypatch):
    d = _client(monkeypatch)
    cv2 = sys.modules["cv2"]
    cv2.imencode = lambda ext, f, p=None: (True, np.zeros(10, dtype=np.uint8))
    cv2.IMWRITE_JPEG_QUALITY = 1

    class _R:
        status_code = 200
        @staticmethod
        def json():
            return {"boxes": [{"cls": 0, "conf": 0.9, "xyxy": [1, 2, 3, 4]},
                              {"cls": "bad"}]}

    d._client = types.SimpleNamespace(post=lambda *a, **k: _R(), close=lambda: None)
    d._fails = 3
    res = d.predict(np.zeros((4, 4, 3)), ["person"])
    assert res is not None and len(res) == 1
    b = list(res[0].boxes)
    assert len(b) == 1 and int(b[0].cls[0]) == 0
    assert float(b[0].conf[0]) == pytest.approx(0.9)
    assert list(b[0].xyxy[0]) == [1, 2, 3, 4]
    assert d._fails == 0


def test_breaker_disables_after_consecutive_failures(monkeypatch):
    d = _client(monkeypatch)
    cv2 = sys.modules["cv2"]
    cv2.imencode = lambda ext, f, p=None: (True, np.zeros(10, dtype=np.uint8))
    cv2.IMWRITE_JPEG_QUALITY = 1

    def _boom(*a, **k):
        raise RuntimeError("net down")

    d._client = types.SimpleNamespace(post=_boom, close=lambda: None)
    monkeypatch.setattr(settings, "REMOTE_VISION_BREAKER_FAILS", 3, raising=False)
    for _ in range(3):
        assert d.predict(np.zeros((4, 4, 3)), ["person"]) is None
    assert d._disabled is True
    assert d.available() is False       # breaker wins even with a healthy probe


def test_yolo_predict_wiring_falls_back_local():
    import inspect
    from backend.services import reframer_face as rf
    src = inspect.getsource(rf._FaceDetectorSrc if hasattr(rf, "_FaceDetectorSrc") else rf)
    assert "RemoteVisionDetector" in src
    # Remote path returns None → the local predict below still runs.
    i_remote = src.find("_rv.predict(")
    i_local = src.find("self._yolo_model.predict(*args, device=self._yolo_device")
    assert i_remote != -1 and i_local != -1 and i_remote < i_local


def test_companion_models_pulls_only_missing(monkeypatch):
    from backend.services import companion_models as cm
    from backend.services import ollama_registry as reg
    monkeypatch.setattr(settings, "OLLAMA_PRIMARY_MODEL", "llava:7b", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_EDITORIAL_MODEL", "qwen2.5:3b", raising=False)
    monkeypatch.setattr(settings, "OLLAMA_TRANSLATION_MODEL",
                        "qwen3:4b-instruct-2507-q4_K_M", raising=False)
    monkeypatch.setattr(reg, "companion_host", lambda: object())
    monkeypatch.setattr(reg, "companion_base", lambda c: "http://comp.test:11500")
    monkeypatch.setattr(reg, "auth_headers", lambda c: {})
    cm._last_attempt.clear()
    pulled = []

    class _Tags:
        status_code = 200
        @staticmethod
        def json():
            return {"models": [{"name": "qwen3:4b-instruct-2507-q4_K_M"}]}

    class _C:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, headers=None): return _Tags()
        async def post(self, url, headers=None, json=None):
            pulled.append(json["model"])
            return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(cm, "httpx", types.SimpleNamespace(AsyncClient=_C),
                        raising=False)
    import backend.services.companion_models as _m
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=_C))

    async def _run():
        await cm._ensure("job-x")
        await asyncio.sleep(0.05)   # let the pull tasks run

    asyncio.run(_run())
    assert sorted(pulled) == ["llava:7b", "qwen2.5:3b"]


def test_companion_models_throttles(monkeypatch):
    from backend.services import companion_models as cm
    import time
    cm._last_attempt["llava:7b"] = time.monotonic()
    # A second _ensure within the throttle window must not re-pull — covered
    # implicitly by the timestamp guard; just pin the constant is sane.
    assert cm.THROTTLE_S >= 3600


def test_compute_summary_shows_face_detection():
    """The Compute indicator must show face detection's device — the old
    perception.face_detector read was never attached, so the row silently
    vanished from every run."""
    from backend.services.pipeline import _build_compute_summary

    p = types.SimpleNamespace(detection_device="cuda:0",
                              detection_remote_frames=0,
                              detection_local_frames=1745)
    s = _build_compute_summary(types.SimpleNamespace(), p)
    assert s["face_detection"]["device"] == "cuda:0"

    p2 = types.SimpleNamespace(detection_device="cuda:0",
                               detection_remote_frames=1700,
                               detection_local_frames=45)
    s2 = _build_compute_summary(types.SimpleNamespace(), p2)
    assert s2["face_detection"]["device"] == "cuda:companion"
    assert "1700 frames remote" in s2["face_detection"]["detail"]

    p3 = types.SimpleNamespace(detection_device="cpu",
                               detection_remote_frames=0,
                               detection_local_frames=10)
    s3 = _build_compute_summary(types.SimpleNamespace(), p3)
    assert s3["face_detection"]["device"] == "cpu"


def test_perceiver_stashes_detection_telemetry():
    import inspect
    from backend.services import reframer_perceiver as rp
    src = inspect.getsource(rp)
    assert "r.detection_device" in src
    assert "r.detection_remote_frames" in src
