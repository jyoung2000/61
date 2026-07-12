"""Vision inference sidecar for the ClipAI GPU Companion.

Serves YOLO-World detection on the Companion GPU behind the proxy
(``/v1/vision/detect``), so ClipAI's FACES stage — its largest block —
runs on the big card instead of the server's small one. Same packaging
pattern as whisper-server (PyInstaller, localhost-only, started lazily,
stopped on idle by the Companion).

Request (JSON):  {image_b64, classes: [..], conf, max_det}
Response (JSON): {boxes: [{cls, conf, xyxy: [x1,y1,x2,y2]}, ...]}

Class vocabularies are cached: set_classes() re-encodes CLIP text only
when the vocabulary actually changes, exactly like ClipAI's local path.
"""
import base64
import io
import logging
import os
import threading

from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vision-server")

VISION_MODEL = os.environ.get("VISION_MODEL", "yolov8s-worldv2.pt")
PORT = int(os.environ.get("VISION_PORT", "11511"))

app = FastAPI()
_lock = threading.Lock()
_state = {"model": None, "classes": None, "device": "cpu"}


class DetectRequest(BaseModel):
    image_b64: str
    classes: list[str] = []
    conf: float = 0.25
    max_det: int = 20


def _model():
    if _state["model"] is None:
        from ultralytics import YOLO
        import torch
        _state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("loading %s on %s", VISION_MODEL, _state["device"])
        m = YOLO(VISION_MODEL)
        m.to(_state["device"])
        _state["model"] = m
    return _state["model"]


@app.get("/health")
def health():
    return {"ok": True, "model": VISION_MODEL, "device": _state["device"],
            "loaded": _state["model"] is not None}


@app.post("/v1/vision/detect")
def detect(req: DetectRequest):
    import numpy as np
    import cv2

    raw = base64.b64decode(req.image_b64)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"boxes": [], "error": "bad image"}
    with _lock:
        m = _model()
        if req.classes and req.classes != _state["classes"]:
            m.set_classes(list(req.classes))
            _state["classes"] = list(req.classes)
        results = m.predict(frame, device=_state["device"], verbose=False,
                            conf=float(req.conf), max_det=int(req.max_det))
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            try:
                boxes.append({
                    "cls": int(b.cls[0]),
                    "conf": float(b.conf[0]),
                    "xyxy": [float(x) for x in b.xyxy[0].tolist()],
                })
            except Exception:
                continue
    return {"boxes": boxes}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
