"""ClipAI Reframer — Tiered Face Detector.

Extracted from clipai_reframer.py (jyoung2000/60) for the Fez engine
transplant. The original Tkinter GUI is not part of this module.
"""

import cv2
import numpy as np
import json
import subprocess
import threading
import os
import sys
import math
import logging
import time as _time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Callable, Dict
from pathlib import Path

from backend.services.reframer_models import (
    ReframeLogger, get_logger, reset_logger, RenderPlan,
    interpolate_x, clamp_x, _face_overlaps_person,
    LedgerBin, CoverageLedger, PerceptionResult, SceneSignals, AdaptiveParams,
)

logger = logging.getLogger("clipai.reframer_face")


def _pick_yolo_device():
    """Choose the device for YOLO-World inference.

    Returns ``0`` (first CUDA GPU) when a GPU with enough free VRAM is
    present, else ``'cpu'``. GPU inference is ~20x faster than CPU for the
    per-frame subject pass and produces identical detections, so it is the
    single biggest analysis-speed win on a capable GPU.

    Override with the ``CLIPAI_REFRAMER_YOLO_DEVICE`` env var
    (``cpu`` | ``cuda`` | ``auto``). The 6.5GB free-VRAM gate keeps small
    cards (e.g. a 4GB GTX 1650) on CPU so YOLO never starves the VLM stage.
    """
    forced = os.environ.get("CLIPAI_REFRAMER_YOLO_DEVICE", "auto").strip().lower()
    if forced == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
            if forced in ("cuda", "gpu", "0") or free_mb >= 6500:
                return 0
    except Exception:
        pass
    return "cpu"


class FaceDetector:
    """
    Tiered face detector — prioritizes ACTUAL face detectors over person detectors.

    Tier 1: OpenCV DNN ResNet-10 SSD — real face detector, high accuracy
            Auto-downloads ~5MB model on first run
    Tier 2: YOLO-assisted DNN — uses YOLO to find people, then runs DNN
            face detection within each person region (catches angled faces)
    Tier 3: Haar cascade + skin/eye validation — always available fallback

    IMPORTANT: YOLO detects PERSONS not faces. Using person bboxes as face
    bboxes produced oversized boxes covering hats/shoulders/mics. YOLO is
    now only used as a region hint for the real face detector.
    """

    def __init__(self, model_dir: str = None, confidence: float = 0.30):
        self.confidence = confidence
        self.tier = 'none'
        self._yolo_model = None
        self._yolo_device = _pick_yolo_device()  # 0 (GPU) or 'cpu'
        self._dnn_net = None
        self._haar_face = None
        self._haar_eye = None
        # Spatial cache: remembered nonhuman-subject positions.
        # When YOLO detects a toy/character/robot, we cache its bbox.
        # In subsequent frames where YOLO doesn't detect it (intermittent
        # detection is normal), the cache still rejects faces at that
        # position.  Cleared on scene cuts via clear_nonhuman_cache().
        self._nonhuman_cache = []       # list of (x1, y1, x2, y2)

        if model_dir is None:
            try:
                # This file lives in backend/services/; the model weights
                # (YuNet, SFace, YOLO-World) ship in backend/models/ — so
                # resolve one directory above the services package.
                model_dir = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))
            except NameError:
                model_dir = os.getcwd()
        self.model_dir = os.path.join(model_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)

        log = get_logger()

        # ── Tier 1: OpenCV YuNet face detector (330KB ONNX, built into OpenCV) ──
        self._yunet = None
        yunet_path = os.path.join(self.model_dir, "face_detection_yunet_2023mar.onnx")
        try:
            if not os.path.exists(yunet_path) or os.path.getsize(yunet_path) < 50000:
                log.log_stage('PERCEIVE', 'Downloading YuNet face model (330KB)...')
                import urllib.request
                yunet_urls = [
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                ]
                for url in yunet_urls:
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'ClipAI-Reframer/1.0'})
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = resp.read()
                            if len(data) > 50000:
                                with open(yunet_path, 'wb') as f:
                                    f.write(data)
                                log.log_stage('PERCEIVE', f'YuNet downloaded: {len(data)//1024}KB')
                                break
                    except Exception:
                        continue

            if os.path.exists(yunet_path) and os.path.getsize(yunet_path) > 50000:
                self._yunet = cv2.FaceDetectorYN.create(yunet_path, '', (640, 360),
                                                         score_threshold=self.confidence)
                self.tier = 'yunet'
                log.log_stage('PERCEIVE', 'Face detector: YuNet (primary)')
            else:
                raise ValueError("YuNet model download failed")
        except Exception as e:
            log.log_stage('PERCEIVE', f'YuNet unavailable: {str(e)[:80]}')

        # ── Try loading YOLO-World as scene-aware subject detector ──
        # YOLO-World detects ANY object by text prompt — not just "person".
        # This handles anime characters with helmets, mecha, vehicles,
        # explosions, animals, and any other subject a human editor would
        # recognize as "the thing to keep in frame."
        _saved_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        try:
            from ultralytics import YOLO

            # Prefer YOLO-World (open-vocabulary) over yolov8n (person-only)
            world_path = os.path.join(self.model_dir, "yolov8s-worldv2.pt")
            if not os.path.exists(world_path):
                world_path = "yolov8s-worldv2.pt"
            # Fallback to yolov8n if World model not available
            if not os.path.exists(world_path):
                world_path = os.path.join(self.model_dir, "yolov8n.pt")
                if not os.path.exists(world_path):
                    world_path = "yolov8n.pt"

            # Hide the GPU from YOLO only when CPU is the chosen device. With
            # a GPU selected, leave CUDA visible so .predict(device=0) works.
            if self._yolo_device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self._yolo_model = YOLO(world_path)

            # Set broad scene-understanding classes for YOLO-World
            # These cover anime, live-action, sports, and general content.
            # The model detects whichever classes are present — no penalty
            # for including classes that aren't in the frame.
            is_world_model = 'world' in str(world_path).lower()
            if is_world_model and hasattr(self._yolo_model, 'set_classes'):
                self._yolo_classes = [
                    # Priority 1: People and faces (highest value for reframing)
                    "person", "head", "face", "character",
                    # Priority 2: Action subjects (anime/gaming/sci-fi)
                    "robot", "mecha", "vehicle", "car",
                    # Priority 3: Other subjects
                    "animal", "bird",
                    # Priority 4: Background surfaces with face-like content
                    # (suppresses false face detections on artwork/posters)
                    "painting", "poster", "picture", "screen",
                ]
                try:
                    self._yolo_model.set_classes(self._yolo_classes)
                    log.log_stage('PERCEIVE',
                        f'YOLO-World loaded with {len(self._yolo_classes)} classes: '
                        f'{", ".join(self._yolo_classes[:6])}...')
                except Exception as e:
                    log.log_stage('PERCEIVE',
                        f'YOLO-World set_classes failed: {e}. Using default COCO classes.')
                    is_world_model = False
            else:
                is_world_model = False

            # Test inference — confirms the chosen device works and triggers
            # the CPU fallback now (not mid-run) if the GPU path is broken.
            test = np.zeros((64, 64, 3), dtype=np.uint8)
            self._yolo_predict(test, verbose=False)
            _yolo_dev_label = 'GPU' if self._yolo_device != 'cpu' else 'CPU'

            if is_world_model:
                log.log_stage('PERCEIVE',
                    f'YOLO-World v2 loaded as scene-aware subject detector ({_yolo_dev_label})')
            elif self.tier == 'yunet':
                log.log_stage('PERCEIVE',
                    f'YOLO loaded as person-detection helper for YuNet ({_yolo_dev_label})')
            else:
                self.tier = 'yolo_only'
                log.log_stage('PERCEIVE',
                    f'Face detector: YOLO + Haar combined ({_yolo_dev_label})')
        except Exception as e:
            log.log_stage('PERCEIVE', f'YOLO unavailable: {str(e)[:100]}')
        finally:
            # Restore CUDA_VISIBLE_DEVICES so Whisper can still use GPU
            if _saved_cuda_visible is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = _saved_cuda_visible
            else:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)

        # ── Haar fallback (always works) ──
        if self.tier == 'none':
            self.tier = 'haar'
            log.log_stage('PERCEIVE', 'Face detector: Haar cascade (fallback)')
        self._haar_face = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        self._haar_eye = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

        # ── SFace face recognizer (37MB ONNX, for track identity matching) ──
        # Used by track consolidation to match faces across scene cuts and
        # detection dropouts by appearance (embedding similarity) rather than
        # just spatial position. Ships with OpenCV >= 4.5.4.
        self._sface = None
        sface_path = os.path.join(self.model_dir, "face_recognition_sface_2021dec.onnx")
        try:
            if not os.path.exists(sface_path) or os.path.getsize(sface_path) < 100000:
                log.log_stage('PERCEIVE', 'Downloading SFace recognition model (37MB)...')
                import urllib.request
                sface_urls = [
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                ]
                for url in sface_urls:
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'ClipAI-Reframer/1.0'})
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            data = resp.read()
                            if len(data) > 100000:
                                with open(sface_path, 'wb') as f:
                                    f.write(data)
                                log.log_stage('PERCEIVE', f'SFace downloaded: {len(data)//1024}KB')
                                break
                    except Exception:
                        continue

            if os.path.exists(sface_path) and os.path.getsize(sface_path) > 100000:
                self._sface = cv2.FaceRecognizerSF.create(sface_path, '')
                log.log_stage('PERCEIVE', 'SFace face recognizer loaded')
        except Exception as e:
            log.log_stage('PERCEIVE', f'SFace unavailable (track identity disabled): {str(e)[:80]}')

    def _yolo_predict(self, *args, **kwargs):
        """Run YOLO ``.predict()`` on the configured device.

        If a GPU inference call fails (e.g. a torchvision NMS CUDA backend
        mismatch), permanently fall back to CPU for the rest of this run so
        analysis degrades gracefully instead of crashing.
        """
        kwargs.pop('device', None)
        try:
            return self._yolo_predict(*args, device=self._yolo_device, **kwargs)
        except Exception as e:
            if self._yolo_device != 'cpu':
                logger.warning(
                    "YOLO GPU inference failed (%s) — falling back to CPU",
                    str(e)[:140],
                )
                self._yolo_device = 'cpu'
                return self._yolo_predict(*args, device='cpu', **kwargs)
            raise

    def compute_embedding(self, frame_bgr, face_dict: dict) -> Optional[np.ndarray]:
        """Compute a 128-dim face embedding for identity matching.

        Uses SFace (FaceRecognizerSF) with YuNet-format detection array
        for face alignment. Returns None if SFace is unavailable or the
        face region is too small for reliable embedding."""
        if self._sface is None:
            return None

        try:
            fx, fy, fw, fh = face_dict['x'], face_dict['y'], face_dict['w'], face_dict['h']

            # Need minimum face size for reliable embedding
            if fw < 20 or fh < 20:
                return None

            # Build YuNet-format detection array for alignCrop:
            # [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #  nose_x, nose_y, right_mouth_x, right_mouth_y,
            #  left_mouth_x, left_mouth_y, confidence]
            rm = face_dict.get('right_mouth', (fx + fw * 3 // 4, fy + fh * 4 // 5))
            lm = face_dict.get('left_mouth', (fx + fw // 4, fy + fh * 4 // 5))
            nose = face_dict.get('nose', (fx + fw // 2, fy + fh * 2 // 3))

            det_array = np.array([[
                fx, fy, fw, fh,
                fx + fw * 3 // 10, fy + fh * 3 // 10,  # right eye estimate
                fx + fw * 7 // 10, fy + fh * 3 // 10,  # left eye estimate
                nose[0], nose[1],
                rm[0], rm[1],
                lm[0], lm[1],
                face_dict.get('confidence', 0.9)
            ]], dtype=np.float32)

            aligned = self._sface.alignCrop(frame_bgr, det_array[0])
            embedding = self._sface.feature(aligned)
            return embedding.flatten()  # (1,128) → (128,)
        except Exception:
            return None

    def detect(self, frame_bgr, min_confidence: float = None) -> List[dict]:
        """Detect faces by unioning every available detector.

        No early-stop — full-frame YuNet finds large/lit faces; YOLO-assisted
        YuNet recovers small/occluded faces; tiled YuNet picks up faces in
        wide group shots. Results are merged and deduplicated. Better recall
        on panel/group shots at the cost of extra detection passes per frame.
        """
        conf = min_confidence or self.confidence
        try:
            if self.tier == 'yunet':
                # ── Single YOLO inference, split by class ──
                # person_bboxes  → positive gate (keep faces on humans)
                # nonhuman_bboxes → negative gate (reject faces on toys/figurines)
                if self._yolo_model is not None:
                    person_bboxes, nonhuman_bboxes_raw = \
                        self._get_split_subject_bboxes(frame_bgr)
                    # Update spatial cache and get effective nonhuman zones
                    # (current detection + cache from recent frames where
                    # YOLO detected the toy but not this frame)
                    self._update_nonhuman_cache(nonhuman_bboxes_raw)
                    nonhuman_bboxes = self._get_effective_nonhuman_bboxes(
                        nonhuman_bboxes_raw)
                else:
                    person_bboxes, nonhuman_bboxes = [], []

                all_faces = []
                # 1. Full-frame YuNet (cheapest, best for prominent faces)
                all_faces.extend(self._detect_yunet(frame_bgr, conf))
                # 2. YOLO-assisted YuNet (only inside person bboxes)
                if person_bboxes:
                    all_faces.extend(
                        self._detect_yolo_assisted_yunet(frame_bgr, conf, person_bboxes))
                # 3. Tiled YuNet (recovers faces in wide group shots)
                all_faces.extend(self._detect_yunet_tiled(frame_bgr, conf))

                # ── 4a. Positive gate: keep faces inside person bboxes ──
                if person_bboxes:
                    all_faces = self._gate_by_person_bboxes(all_faces, person_bboxes)
                elif nonhuman_bboxes and all_faces:
                    # YOLO found non-human subjects (toy, character) but
                    # NO persons.  Every face in this frame is very likely
                    # on the figurine/toy — reject them all rather than
                    # risk tracking a non-human.
                    all_faces = []

                # ── 4b. Negative gate: reject faces on non-human subjects ──
                # Belt-and-suspenders: even if a face survived the positive
                # gate (e.g. YOLO misclassified the figurine as "person"),
                # reject it if it's also inside a toy/character bbox.
                if nonhuman_bboxes and all_faces:
                    all_faces = self._reject_nonhuman_faces(
                        all_faces, nonhuman_bboxes)

                # ── 4c. Live-action face validation ──
                # Landmark geometry + skin/sharpness liveness check.
                # Catches false positives on background artwork, posters,
                # logos, and framed photos that survived YOLO gating.
                if all_faces:
                    all_faces = self._validate_live_action_faces(
                        all_faces, frame_bgr, person_bboxes)

                # 5. Haar fallback — but NOT if we explicitly rejected
                #    faces due to nonhuman subjects (would re-find them)
                if not all_faces and not nonhuman_bboxes:
                    return self._detect_haar(frame_bgr)
                return self._dedupe_faces(all_faces) if all_faces else []
            elif self.tier == 'dnn':
                faces = self._detect_dnn(frame_bgr, conf)
                if not faces and self._yolo_model is not None:
                    faces = self._detect_yolo_assisted_dnn(frame_bgr, conf)
                if not faces:
                    faces = self._detect_haar(frame_bgr)
                return faces
            elif self.tier == 'yolo_only':
                # No DNN, no YuNet — but YOLO is available. Union YOLO-assisted
                # Haar with plain Haar so we don't lose hits from either path.
                all_faces = []
                all_faces.extend(self._detect_haar(frame_bgr))
                if self._yolo_model is not None:
                    all_faces.extend(self._detect_yolo_assisted_haar(frame_bgr, conf))
                if not all_faces:
                    return self._detect_haar_relaxed(frame_bgr)
                return self._dedupe_faces(all_faces)
            else:
                # Pure Haar fallback — no DNN, no YuNet, no YOLO
                faces = self._detect_haar(frame_bgr)
                if faces:
                    return faces
                return self._detect_haar_relaxed(frame_bgr)
        except Exception:
            return self._detect_haar(frame_bgr)

    def _detect_yunet(self, frame_bgr, conf) -> List[dict]:
        """YuNet face detection — fast, accurate, handles all angles."""
        h, w = frame_bgr.shape[:2]
        self._yunet.setInputSize((w, h))
        self._yunet.setScoreThreshold(conf)
        _, raw = self._yunet.detect(frame_bgr)

        if raw is None:
            return []

        faces = []
        for det in raw:
            fx, fy, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            score = float(det[-1])

            # Loose size floor — keep small faces from panel shots, group framings,
            # and distant subjects. Absolute minimum 18px on the longer side, plus
            # 1.5% of frame width to scale with resolution.
            min_dim = max(18, int(w * 0.015))
            if fw < min_dim or fh < min_dim:
                continue
            # Aspect ratio sanity (kept generous for tilted/profile heads)
            aspect = fw / max(1, fh)
            if aspect < 0.4 or aspect > 2.5:
                continue

            faces.append({
                'x': max(0, fx), 'y': max(0, fy),
                'w': fw, 'h': fh,
                'cx': fx + fw // 2, 'cy': fy + fh // 2,
                'area': fw * fh,
                'confidence': round(score, 3),
                'source': 'yunet',
                # YuNet landmarks — eyes + nose + mouth for validation & MAR
                'right_eye': (int(det[4]), int(det[5])),
                'left_eye': (int(det[6]), int(det[7])),
                'right_mouth': (int(det[10]), int(det[11])),
                'left_mouth': (int(det[12]), int(det[13])),
                'nose': (int(det[8]), int(det[9])),
            })
        return faces

    # Classes that represent actual human bodies — only these validate
    # face detections in the person-overlap gate.  Everything else
    # (character, toy, microphone, robot …) is tracked for subject
    # display but must NOT let cartoon/figurine faces pass the gate.
    _HUMAN_BODY_CLASSES = {"person", "head", "child"}

    # Classes whose bounding boxes should SUPPRESS face detections.
    # If YuNet finds a "face" inside a toy/character/robot bbox, that
    # face is almost certainly non-human (figurine, artwork, mascot).
    _NONHUMAN_FACE_CLASSES = {
        "toy", "character", "robot", "mecha",
        "animal", "bird", "dog", "cat",
        # Background surfaces that can trigger false face detections
        "painting", "poster", "picture", "screen",
        "frame", "artwork", "display",
    }

    def clear_nonhuman_cache(self):
        """Reset the spatial cache of nonhuman subject positions.

        Called by the analyzer on scene cuts — a nonhuman subject in scene A
        probably isn't in the same position in scene B."""
        self._nonhuman_cache = []

    def _update_nonhuman_cache(self, nonhuman_bboxes):
        """Merge new YOLO detections into the spatial cache.

        When YOLO detects a toy/figurine this frame, update the cache.
        The cache persists until the next scene cut (clear_nonhuman_cache)
        because static props like figurines don't disappear between frames.
        No frame-count expiry — YOLO only detects "toy" ~50% of the time,
        but the toy is there 100% of the time within a scene."""
        if nonhuman_bboxes:
            for nb in nonhuman_bboxes:
                nx = (nb[0] + nb[2]) // 2
                ny = (nb[1] + nb[3]) // 2
                # Check if near an existing cached entry (YOLO jitter ~few px)
                merged = False
                for i, cb in enumerate(self._nonhuman_cache):
                    cx_c = (cb[0] + cb[2]) // 2
                    cy_c = (cb[1] + cb[3]) // 2
                    if abs(nx - cx_c) < 80 and abs(ny - cy_c) < 80:
                        self._nonhuman_cache[i] = nb  # update with latest
                        merged = True
                        break
                if not merged:
                    self._nonhuman_cache.append(nb)

    def _get_effective_nonhuman_bboxes(self, nonhuman_bboxes):
        """Return ALL known nonhuman zones: current detection + cache.

        Always returns the union so that even if YOLO only detects the
        toy in this frame, we also suppress faces at cached positions
        from earlier frames (and vice versa)."""
        if nonhuman_bboxes and self._nonhuman_cache:
            # Union of current + cache (cache may already include current
            # via _update_nonhuman_cache, but duplicates don't matter
            # for the negative gate — it just checks containment)
            return list(nonhuman_bboxes) + list(self._nonhuman_cache)
        return nonhuman_bboxes if nonhuman_bboxes else \
            (list(self._nonhuman_cache) if self._nonhuman_cache else [])

    def _get_split_subject_bboxes(self, frame_bgr):
        """Run YOLO once, return (person_bboxes, nonhuman_bboxes).

        Single inference, two lists:
          person_bboxes   – classes in _HUMAN_BODY_CLASSES  (for positive gate)
          nonhuman_bboxes – classes in _NONHUMAN_FACE_CLASSES (for negative gate)

        Avoids double inference that separate person_only=True / False calls
        would incur.
        """
        if self._yolo_model is None:
            return [], []
        try:
            has_world_classes = hasattr(self, '_yolo_classes') and self._yolo_classes
            kwargs = dict(verbose=False, conf=0.3, max_det=15, device='cpu')
            if not has_world_classes:
                kwargs['classes'] = [0]  # person only for standard YOLO
            results = self._yolo_predict(frame_bgr, **kwargs)
        except Exception:
            return [], []
        person_boxes = []
        nonhuman_boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x2 - x1 < 30 or y2 - y1 < 30:
                    continue
                if has_world_classes:
                    cls_id = int(box.cls[0])
                    cls_name = (self._yolo_classes[cls_id]
                                if cls_id < len(self._yolo_classes) else "")
                    if cls_name in self._HUMAN_BODY_CLASSES:
                        person_boxes.append((x1, y1, x2, y2))
                    elif cls_name in self._NONHUMAN_FACE_CLASSES:
                        nonhuman_boxes.append((x1, y1, x2, y2))
                    # other classes (microphone, book, …) → neither list
                else:
                    # Standard YOLO (class 0 = person)
                    person_boxes.append((x1, y1, x2, y2))
        return person_boxes, nonhuman_boxes

    def _get_person_bboxes(self, frame_bgr,
                           person_only: bool = False) -> List[Tuple[int, int, int, int]]:
        """Run YOLO/YOLO-World and return subject bboxes as (x1, y1, x2, y2).

        With YOLO-World: detects all set classes (person, character, robot, etc.)
        With yolov8n: detects person only (class 0)

        Parameters
        ----------
        person_only : bool
            When True, return only bboxes whose YOLO-World class is in
            _HUMAN_BODY_CLASSES (person, head).  Used by the face gate so
            that cartoon figurines / toys / characters detected by
            YOLO-World don't accidentally validate YuNet face hits on
            non-human objects.  When False (default), return all detected
            subject bboxes (used for subject tracking and display).
        """
        if self._yolo_model is None:
            return []
        try:
            # YOLO-World: detect all set classes (no class filter)
            # yolov8n: detect person only (class 0)
            has_world_classes = hasattr(self, '_yolo_classes') and self._yolo_classes
            kwargs = dict(verbose=False, conf=0.3, max_det=15, device='cpu')
            if not has_world_classes:
                kwargs['classes'] = [0]  # person only for standard YOLO
            results = self._yolo_predict(frame_bgr, **kwargs)
        except Exception:
            return []
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if x2 - x1 < 30 or y2 - y1 < 30:
                    continue
                # When person_only is requested and we have YOLO-World
                # class names, skip non-human classes (toy, character, etc.)
                if person_only and has_world_classes:
                    cls_id = int(box.cls[0])
                    cls_name = (self._yolo_classes[cls_id]
                                if cls_id < len(self._yolo_classes) else "")
                    if cls_name not in self._HUMAN_BODY_CLASSES:
                        continue
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _gate_by_person_bboxes(self, faces: List[dict],
                                person_bboxes: List[Tuple[int, int, int, int]]) -> List[dict]:
        """Keep only faces that look like heads attached to bodies.

        A real human head sits in the UPPER portion of a person bbox — roughly
        the top 35% from a slightly-extended top edge. A toy/figurine that
        happens to sit beside a person (e.g. on a table next to a seated
        guest) might fall *inside* the person's bbox at chest or knee level
        but won't be in the head zone.

        Gate logic per face:
          1. Face center must be horizontally inside the person bbox.
          2. Face center must vertically fall in the head zone:
             from 10% above the bbox top (slack for under-framed heads)
             to 35% down from the bbox top.
          3. Face area must be plausible relative to the person bbox —
             no more than 40% of the bbox area (rejects detections where
             the entire bbox is treated as one giant face).

        If YOLO found no persons, we keep all faces (the caller skips the
        gate in that case)."""
        if not person_bboxes:
            return faces
        kept = []
        for f in faces:
            cx, cy = f['cx'], f['cy']
            face_area = f.get('area', f['w'] * f['h'])
            for px1, py1, px2, py2 in person_bboxes:
                ph = py2 - py1
                pw = px2 - px1
                # Horizontal: center inside the bbox
                if not (px1 <= cx <= px2):
                    continue
                # Vertical: head zone is upper ~35% of bbox, with 10% slack on top
                top_slack = int(ph * 0.10)
                head_zone_top = py1 - top_slack
                head_zone_bottom = py1 + int(ph * 0.35)
                if not (head_zone_top <= cy <= head_zone_bottom):
                    continue
                # Sanity: face must not be too large relative to person bbox
                # (rejects bbox-sized phantom detections)
                bbox_area = max(1, pw * ph)
                if face_area > bbox_area * 0.40:
                    continue
                kept.append(f)
                break
        return kept

    def _reject_nonhuman_faces(self, faces: List[dict],
                                nonhuman_bboxes: List[Tuple[int, int, int, int]]) -> List[dict]:
        """Negative gate: remove faces whose center falls inside a non-human
        subject bbox (toy, figurine, character, robot, animal).

        This is the complement to _gate_by_person_bboxes (positive gate).
        The positive gate keeps faces that ARE inside person bboxes.
        This negative gate removes faces that ARE inside non-human bboxes.

        Together they handle all failure modes:
          - Positive gate alone fails when YOLO misses the person → all
            faces pass, including figurine faces.
          - Negative gate catches figurine faces even when positive gate
            was skipped (no person detected).
          - When YOLO misclassifies a figurine as "person", the positive
            gate passes the figurine face, but the negative gate catches
            it because the figurine is ALSO detected as "toy"/"character".

        Uses a generous margin (25% of bbox width/height) because YOLO
        bboxes don't always tightly wrap the subject."""
        if not nonhuman_bboxes or not faces:
            return faces
        kept = []
        for f in faces:
            fcx, fcy = f['cx'], f['cy']
            in_nonhuman = False
            for nx1, ny1, nx2, ny2 in nonhuman_bboxes:
                # Generous margin — YOLO bboxes are approximate
                nw, nh = nx2 - nx1, ny2 - ny1
                mx = int(nw * 0.25)
                my = int(nh * 0.25)
                if (nx1 - mx <= fcx <= nx2 + mx and
                        ny1 - my <= fcy <= ny2 + my):
                    in_nonhuman = True
                    break
            if not in_nonhuman:
                kept.append(f)
        return kept

    # ── Live-action face validation ──
    # Background artwork, posters, logos, and framed photos can trigger
    # YuNet false positives.  These methods use geometric and photometric
    # cues to distinguish real 3D human faces from flat images.

    def _validate_face_landmarks(self, face: dict) -> float:
        """Score landmark geometry plausibility (0.0 = implausible, 1.0 = perfect).

        Real human faces have specific geometric relationships between
        landmarks.  Faces on flat artwork viewed at an angle, logos, or
        other non-face patterns produce distorted landmark geometry that
        this function penalizes.

        Returns a 0-1 plausibility score.  Faces without landmarks get 1.0
        (benefit of the doubt — Haar/DNN detections have no landmarks)."""
        if 'right_eye' not in face or 'left_eye' not in face:
            return 1.0  # no landmarks to validate

        try:
            re = face['right_eye']
            le = face['left_eye']
            nose = face.get('nose')
            rm = face.get('right_mouth')
            lm = face.get('left_mouth')
            fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']

            if fw < 10 or fh < 10:
                return 0.5  # too small to validate

            score = 1.0

            # ── Check 1: Inter-eye distance proportional to face width ──
            # Real faces: IED is typically 30-70% of face width.
            # Distorted artwork/logos often produce very wide or very narrow IED.
            ied = math.sqrt((re[0] - le[0])**2 + (re[1] - le[1])**2)
            ied_ratio = ied / max(1, fw)
            if ied_ratio < 0.15 or ied_ratio > 0.85:
                score -= 0.4  # major distortion
            elif ied_ratio < 0.25 or ied_ratio > 0.75:
                score -= 0.15

            # ── Check 2: Eyes roughly horizontal ──
            # Real faces can be tilted but eyes should not differ by > 40% of face height.
            eye_dy = abs(re[1] - le[1])
            if eye_dy > fh * 0.40:
                score -= 0.3
            elif eye_dy > fh * 0.25:
                score -= 0.1

            # ── Check 3: Eyes in upper portion of face bbox ──
            # Eyes should be in the top 55% of the face (above midpoint).
            eye_avg_y = (re[1] + le[1]) / 2
            eye_rel_y = (eye_avg_y - fy) / max(1, fh)
            if eye_rel_y > 0.60 or eye_rel_y < 0.05:
                score -= 0.3  # eyes too low or above face bbox
            elif eye_rel_y > 0.50:
                score -= 0.1

            # ── Check 4: Nose below eyes, above mouth ──
            if nose:
                nose_y = nose[1]
                if nose_y < eye_avg_y:
                    score -= 0.25  # nose above eyes = implausible
                if rm and lm:
                    mouth_avg_y = (rm[1] + lm[1]) / 2
                    if nose_y > mouth_avg_y:
                        score -= 0.2  # nose below mouth = implausible

            # ── Check 5: Landmarks inside face bounding box ──
            # All landmarks should be within or very near the bbox.
            # Flat artwork viewed at angles produces out-of-bbox landmarks.
            margin_x = int(fw * 0.15)
            margin_y = int(fh * 0.15)
            for lm_name in ['right_eye', 'left_eye', 'nose', 'right_mouth', 'left_mouth']:
                pt = face.get(lm_name)
                if pt is None:
                    continue
                if (pt[0] < fx - margin_x or pt[0] > fx + fw + margin_x or
                        pt[1] < fy - margin_y or pt[1] > fy + fh + margin_y):
                    score -= 0.15

            # ── Check 6: Mouth width proportional to face ──
            if rm and lm:
                mouth_w = math.sqrt((rm[0] - lm[0])**2 + (rm[1] - lm[1])**2)
                mouth_ratio = mouth_w / max(1, fw)
                if mouth_ratio > 0.95 or mouth_ratio < 0.10:
                    score -= 0.2

            return max(0.0, min(1.0, score))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return 0.8  # error → mild penalty

    def _face_liveness_check(self, frame_bgr, face: dict) -> float:
        """Photometric liveness score for a face detection (0.0 = fake, 1.0 = real).

        Distinguishes real 3D human faces from flat images (posters, logos,
        framed photos, artwork) using:
          1. Skin-tone presence (YCrCb range check)
          2. Laplacian variance (sharpness diversity — real faces have both
             smooth skin and textured features; flat prints may differ)
          3. Color variance (natural faces have gradual hue/saturation
             gradients; artificial images may have flat or extreme colors)

        Returns a 0-1 score.  Called only for faces that survived other gates."""
        try:
            fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']
            h, w = frame_bgr.shape[:2]

            # Extract face ROI with slight padding
            pad = max(2, int(min(fw, fh) * 0.05))
            y1 = max(0, fy - pad)
            y2 = min(h, fy + fh + pad)
            x1 = max(0, fx - pad)
            x2 = min(w, fx + fw + pad)
            roi = frame_bgr[y1:y2, x1:x2]

            if roi.size == 0 or roi.shape[0] < 8 or roi.shape[1] < 8:
                return 0.8  # too small to analyze

            score = 1.0

            # ── Skin tone check ──
            ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
            # Expanded skin range to handle diverse skin tones and lighting
            lower_skin = np.array([0, 125, 65], dtype=np.uint8)
            upper_skin = np.array([255, 195, 145], dtype=np.uint8)
            skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
            skin_pct = np.count_nonzero(skin_mask) / max(1, skin_mask.size)

            if skin_pct < 0.05:
                score -= 0.5  # no skin tones at all → likely not a real face
            elif skin_pct < 0.10:
                score -= 0.25

            # ── Laplacian variance (focus/depth cue) ──
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Resize to consistent analysis size to normalize variance
            analyze_size = (64, 64)
            gray_n = cv2.resize(gray_roi, analyze_size, interpolation=cv2.INTER_LINEAR)
            laplacian = cv2.Laplacian(gray_n, cv2.CV_64F)
            lap_var = float(laplacian.var())

            # Real faces typically have Laplacian variance between 50-2000.
            # Flat printed faces may have very low (blurry poster) or very high
            # (halftone/sharp edge pattern) variance.  Extremely low suggests
            # an out-of-focus background surface.
            if lap_var < 15:
                score -= 0.3  # very blurry → likely background
            elif lap_var < 30:
                score -= 0.15

            # ── Color uniformity check ──
            # Real faces have natural color variation (cheeks, lips, forehead).
            # Logos and artwork may be very uniform or very saturated.
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            sat_mean = float(np.mean(hsv[:, :, 1]))
            sat_std = float(np.std(hsv[:, :, 1]))
            val_std = float(np.std(hsv[:, :, 2]))

            # Very low saturation std → flat color (logo, graphic)
            if sat_std < 8 and sat_mean > 100:
                score -= 0.2

            # Very low brightness variance → uniform flat surface
            if val_std < 10:
                score -= 0.2

            return max(0.0, min(1.0, score))
        except Exception:
            return 0.8  # error → mild penalty

    def _validate_live_action_faces(self, faces: List[dict], frame_bgr,
                                      person_bboxes: List[Tuple[int, int, int, int]]) -> List[dict]:
        """Apply live-action validation to face detections.

        Faces that fail landmark geometry or liveness checks get their
        confidence reduced.  Faces below a minimum confidence after
        penalties are removed.

        This is the final quality gate for live-action content — it catches
        false positives that survived YOLO person gating (e.g. faces on
        background artwork that happen to be inside a large person bbox,
        or faces detected when no person bboxes were found).

        Faces that ARE inside a person bbox head zone get a lighter check
        (they're already validated by body proximity).  Faces that are NOT
        inside any person bbox get the full check battery."""
        if not faces:
            return faces

        # Determine which faces are person-gated (inside a person bbox head zone)
        person_gated = set()
        if person_bboxes:
            for i, f in enumerate(faces):
                cx, cy = f['cx'], f['cy']
                for px1, py1, px2, py2 in person_bboxes:
                    ph = py2 - py1
                    top_slack = int(ph * 0.10)
                    head_zone_top = py1 - top_slack
                    head_zone_bottom = py1 + int(ph * 0.40)
                    if px1 <= cx <= px2 and head_zone_top <= cy <= head_zone_bottom:
                        person_gated.add(i)
                        break

        # Compute median face area of person-gated faces for size consistency
        gated_areas = [faces[i].get('area', 0) for i in person_gated]
        median_area = sorted(gated_areas)[len(gated_areas) // 2] if gated_areas else 0

        validated = []
        for i, face in enumerate(faces):
            is_gated = i in person_gated
            conf = face.get('confidence', 0.5)

            # ── Landmark geometry check ──
            landmark_score = self._validate_face_landmarks(face)
            if landmark_score < 0.3:
                # Very implausible geometry → reject outright
                continue
            elif landmark_score < 0.6:
                # Moderate implausibility → penalize confidence
                conf *= (0.4 + landmark_score * 0.6)

            # ── Liveness check (heavier for ungated faces) ──
            if not is_gated:
                liveness = self._face_liveness_check(frame_bgr, face)
                if liveness < 0.3:
                    # Almost certainly not a real face → reject
                    continue
                elif liveness < 0.6:
                    conf *= (0.3 + liveness * 0.7)

                # ── Size consistency check ──
                # Ungated faces much smaller than gated faces are likely
                # on background surfaces (posters, artwork at distance).
                if median_area > 0:
                    face_area = face.get('area', 0)
                    size_ratio = face_area / max(1, median_area)
                    if size_ratio < 0.15:
                        # Tiny compared to real faces → very likely background
                        conf *= 0.3
                    elif size_ratio < 0.30:
                        conf *= 0.6
            else:
                # Person-gated faces: lighter liveness check (skin only)
                liveness = self._face_liveness_check(frame_bgr, face)
                if liveness < 0.2:
                    conf *= 0.5

            # Apply updated confidence
            face = dict(face)  # don't mutate original
            face['confidence'] = round(max(0.01, conf), 3)

            # Minimum confidence threshold after validation
            if face['confidence'] >= 0.12:
                validated.append(face)

        return validated

    def _detect_yolo_assisted_yunet(self, frame_bgr, conf,
                                     person_bboxes: Optional[List] = None) -> List[dict]:
        """Use YOLO person bboxes to run YuNet on each person crop.
        Recall booster for panel/group shots where YuNet at full scale misses
        small or partially-occluded faces. The full-frame YuNet pass catches
        big, well-lit faces; this catches the rest.

        person_bboxes can be passed in to avoid redundant YOLO inference;
        otherwise it's computed here."""
        if self._yunet is None:
            return []
        if person_bboxes is None:
            person_bboxes = self._get_person_bboxes(frame_bgr, person_only=True)
        if not person_bboxes:
            return []
        h, w = frame_bgr.shape[:2]

        faces = []
        # Per-crop threshold: the person crop already constrains where a face
        # can plausibly be, so we tolerate slightly weaker scores than full
        # frame — but not so low that we hallucinate faces in clothing/hands.
        crop_thresh = max(0.25, conf - 0.05)

        for px1, py1, px2, py2 in person_bboxes:
            pw, ph = px2 - px1, py2 - py1

            # Crop the person region with a 10% margin for context.
            margin = int(max(pw, ph) * 0.1)
            rx1 = max(0, px1 - margin)
            ry1 = max(0, py1 - margin)
            rx2 = min(w, px2 + margin)
            ry2 = min(h, py2 + margin)
            person_crop = frame_bgr[ry1:ry2, rx1:rx2]
            if person_crop.size == 0:
                continue
            ch, cw = person_crop.shape[:2]

            # Upscale small crops so YuNet has enough pixels to work with.
            scale = 1.0
            if max(cw, ch) < 320:
                scale = 320 / max(cw, ch)
                person_crop = cv2.resize(
                    person_crop, (int(cw * scale), int(ch * scale)),
                    interpolation=cv2.INTER_LINEAR)
                ch, cw = person_crop.shape[:2]

            try:
                self._yunet.setInputSize((cw, ch))
                self._yunet.setScoreThreshold(crop_thresh)
                _, raw = self._yunet.detect(person_crop)
            except Exception:
                continue

            if raw is None:
                continue

            for det in raw:
                fx, fy, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                score = float(det[-1])
                # Map back to full-frame coordinates, accounting for upscale
                inv = 1.0 / scale
                abs_x = rx1 + int(fx * inv)
                abs_y = ry1 + int(fy * inv)
                abs_w = int(fw * inv)
                abs_h = int(fh * inv)
                if abs_w < 12 or abs_h < 12:
                    continue
                aspect = abs_w / max(1, abs_h)
                if aspect < 0.4 or aspect > 2.5:
                    continue
                faces.append({
                    'x': max(0, abs_x), 'y': max(0, abs_y),
                    'w': abs_w, 'h': abs_h,
                    'cx': abs_x + abs_w // 2,
                    'cy': abs_y + abs_h // 2,
                    'area': abs_w * abs_h,
                    'confidence': round(score, 3),
                    'source': 'yunet_yolo',
                    # Map landmarks back to full-frame coords
                    'right_eye': (rx1 + int(det[4] * inv), ry1 + int(det[5] * inv)),
                    'left_eye': (rx1 + int(det[6] * inv), ry1 + int(det[7] * inv)),
                    'right_mouth': (rx1 + int(det[10] * inv), ry1 + int(det[11] * inv)),
                    'left_mouth': (rx1 + int(det[12] * inv), ry1 + int(det[13] * inv)),
                    'nose': (rx1 + int(det[8] * inv), ry1 + int(det[9] * inv)),
                })

        return faces

    def _detect_yunet_tiled(self, frame_bgr, conf) -> List[dict]:
        """Tile the frame into overlapping quadrants and run YuNet on each.
        Increases effective resolution per-tile so small faces in wide shots
        are detected. Tiles overlap by 20% so faces near tile boundaries
        aren't cut. Skipped on small frames where it would add no benefit."""
        if self._yunet is None:
            return []
        h, w = frame_bgr.shape[:2]
        # Tile only when there are enough pixels to gain anything. At small
        # detection resolutions a tile would be lower-res than the full frame.
        if w < 960 or h < 540:
            return []

        # 2x2 tile grid with 20% overlap
        tw = int(w * 0.6)
        th = int(h * 0.6)
        # tile origins for 2x2 layout
        origins = [
            (0, 0),
            (w - tw, 0),
            (0, h - th),
            (w - tw, h - th),
        ]

        # Tiled threshold: tile edges produce odd patterns that can match
        # face features at low scores. Stay above the full-frame threshold.
        tile_thresh = max(0.32, conf + 0.02)

        faces = []
        for ox, oy in origins:
            tile = frame_bgr[oy:oy + th, ox:ox + tw]
            if tile.size == 0:
                continue
            try:
                self._yunet.setInputSize((tw, th))
                self._yunet.setScoreThreshold(tile_thresh)
                _, raw = self._yunet.detect(tile)
            except Exception:
                continue
            if raw is None:
                continue
            min_dim = max(15, int(w * 0.012))
            for det in raw:
                fx, fy, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                score = float(det[-1])
                if fw < min_dim or fh < min_dim:
                    continue
                aspect = fw / max(1, fh)
                if aspect < 0.4 or aspect > 2.5:
                    continue
                abs_x = ox + fx
                abs_y = oy + fy
                faces.append({
                    'x': max(0, abs_x), 'y': max(0, abs_y),
                    'w': fw, 'h': fh,
                    'cx': abs_x + fw // 2, 'cy': abs_y + fh // 2,
                    'area': fw * fh,
                    'confidence': round(score, 3),
                    'source': 'yunet_tile',
                    # Map landmarks to full-frame coords
                    'right_eye': (ox + int(det[4]), oy + int(det[5])),
                    'left_eye': (ox + int(det[6]), oy + int(det[7])),
                    'right_mouth': (ox + int(det[10]), oy + int(det[11])),
                    'left_mouth': (ox + int(det[12]), oy + int(det[13])),
                    'nose': (ox + int(det[8]), oy + int(det[9])),
                })

        return faces

    def _detect_dnn(self, frame_bgr, conf) -> List[dict]:
        """DNN SSD face detection — the primary detector."""
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < conf:
                continue
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            fw, fh = x2 - x1, y2 - y1
            if fw < w * 0.03 or fh < h * 0.03:
                continue
            aspect = fw / max(1, fh)
            if aspect < 0.5 or aspect > 2.0:
                continue
            faces.append({
                'x': x1, 'y': y1, 'w': fw, 'h': fh,
                'cx': x1 + fw // 2, 'cy': y1 + fh // 2,
                'area': fw * fh, 'confidence': round(confidence, 3),
            })
        return faces

    def _detect_yolo_assisted_dnn(self, frame_bgr, conf) -> List[dict]:
        """Use YOLO to find people, then run DNN face detector on each person region.
        Catches faces that DNN misses at full-frame scale (angled, small, occluded)."""
        h, w = frame_bgr.shape[:2]
        results = self._yolo_predict(
            frame_bgr, verbose=False, conf=0.4, classes=[0], max_det=10,
            device='cpu')

        faces = []
        for r in results:
            for box in r.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                pw, ph = px2 - px1, py2 - py1
                if pw < 30 or ph < 30:
                    continue

                # Expand person region slightly for DNN context
                margin = int(max(pw, ph) * 0.1)
                rx1 = max(0, px1 - margin)
                ry1 = max(0, py1 - margin)
                rx2 = min(w, px2 + margin)
                ry2 = min(h, py2 + margin)

                person_crop = frame_bgr[ry1:ry2, rx1:rx2]
                if person_crop.size == 0:
                    continue

                # Run DNN on the person crop
                blob = cv2.dnn.blobFromImage(
                    person_crop, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
                self._dnn_net.setInput(blob)
                detections = self._dnn_net.forward()

                ch, cw = person_crop.shape[:2]
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence < conf:
                        continue
                    fx1 = max(0, int(detections[0, 0, i, 3] * cw))
                    fy1 = max(0, int(detections[0, 0, i, 4] * ch))
                    fx2 = min(cw, int(detections[0, 0, i, 5] * cw))
                    fy2 = min(ch, int(detections[0, 0, i, 6] * ch))
                    fw, fh = fx2 - fx1, fy2 - fy1
                    if fw < 15 or fh < 15:
                        continue

                    # Map back to full frame coordinates
                    abs_x = rx1 + fx1
                    abs_y = ry1 + fy1
                    faces.append({
                        'x': abs_x, 'y': abs_y, 'w': fw, 'h': fh,
                        'cx': abs_x + fw // 2, 'cy': abs_y + fh // 2,
                        'area': fw * fh, 'confidence': round(confidence, 3),
                    })

        return self._dedupe_faces(faces)

    def _detect_yolo_assisted_haar(self, frame_bgr, conf) -> List[dict]:
        """Use YOLO to find people, then run relaxed Haar inside each person's head region."""
        h, w = frame_bgr.shape[:2]
        results = self._yolo_predict(
            frame_bgr, verbose=False, conf=0.3, classes=[0], max_det=10,
            device='cpu')

        all_faces = []
        for r in results:
            for box in r.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                pw, ph = px2 - px1, py2 - py1
                if pw < 30 or ph < 30:
                    continue

                # Search for face in upper 50% of person bbox (not 40%)
                # and expand horizontally for people at angles
                margin_x = int(pw * 0.15)
                head_x1 = max(0, px1 - margin_x)
                head_x2 = min(w, px2 + margin_x)
                head_y1 = max(0, py1)
                head_y2 = min(h, py1 + int(ph * 0.50))

                head_region = frame_bgr[head_y1:head_y2, head_x1:head_x2]
                if head_region.size == 0:
                    continue

                gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                min_face = max(12, int(pw * 0.15))

                # Relaxed params: scaleFactor=1.15 (finer scan), minNeighbors=3 (more detections)
                raw = self._haar_face.detectMultiScale(
                    eq, scaleFactor=1.15, minNeighbors=3,
                    minSize=(min_face, min_face))

                for (fx, fy, fw, fh) in (raw if len(raw) > 0 else []):
                    abs_x = head_x1 + fx
                    abs_y = head_y1 + fy

                    # Basic skin check on the detected region
                    roi = frame_bgr[abs_y:abs_y+fh, abs_x:abs_x+fw]
                    if roi.size > 0 and not self._has_skin_region(roi):
                        continue

                    all_faces.append({
                        'x': abs_x, 'y': abs_y, 'w': int(fw), 'h': int(fh),
                        'cx': abs_x + fw // 2, 'cy': abs_y + fh // 2,
                        'area': int(fw * fh), 'confidence': 0.6,
                    })

        return all_faces

    def _detect_haar_relaxed(self, frame_bgr) -> List[dict]:
        """Full-frame Haar with relaxed parameters — catches more faces at the
        cost of slightly more false positives. Skin check filters the worst."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)

        min_face = max(25, int(w * 0.05))
        max_face = int(w * 0.80)

        # More permissive: scaleFactor=1.15, minNeighbors=3
        raw = self._haar_face.detectMultiScale(
            eq, scaleFactor=1.15, minNeighbors=3,
            minSize=(min_face, min_face), maxSize=(max_face, max_face),
            flags=cv2.CASCADE_SCALE_IMAGE)

        faces = []
        for (fx, fy, fw, fh) in (raw if len(raw) > 0 else []):
            aspect = fw / max(1, fh)
            if aspect < 0.6 or aspect > 1.5:
                continue
            # Skip watermark zone
            if fy + fh > h * 0.90 and fh < h * 0.15:
                continue
            # Skin check
            roi = frame_bgr[fy:fy+fh, fx:fx+fw]
            if roi.size > 0 and not self._has_skin_region(roi):
                continue

            faces.append({
                'x': int(fx), 'y': int(fy), 'w': int(fw), 'h': int(fh),
                'cx': int(fx + fw // 2), 'cy': int(fy + fh // 2),
                'area': int(fw * fh), 'confidence': 0.45,
            })
        return faces

    def _detect_haar(self, frame_bgr) -> List[dict]:
        """Haar cascade with skin tone validation. Eye check is advisory, not mandatory."""
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)

        min_face = max(30, int(w * 0.06))
        max_face = int(w * 0.80)

        raw = self._haar_face.detectMultiScale(
            eq, scaleFactor=1.15, minNeighbors=4,
            minSize=(min_face, min_face), maxSize=(max_face, max_face),
            flags=cv2.CASCADE_SCALE_IMAGE)

        faces = []
        for (fx, fy, fw, fh) in (raw if len(raw) > 0 else []):
            aspect = fw / max(1, fh)
            if aspect < 0.6 or aspect > 1.5:
                continue
            if fy + fh > h * 0.90 and fh < h * 0.15:
                continue
            # Skin tone is mandatory
            roi = frame_bgr[fy:fy+fh, fx:fx+fw]
            if roi.size > 0 and not self._has_skin_region(roi):
                continue

            # Eye check: if eyes found → higher confidence, if not → still accept
            conf = 0.5
            if fw > min_face * 1.3:
                eye_roi = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
                if eye_roi.size > 0:
                    eyes = self._haar_eye.detectMultiScale(
                        eye_roi, scaleFactor=1.15, minNeighbors=2,
                        minSize=(int(fw * 0.08), int(fw * 0.08)),
                        maxSize=(int(fw * 0.45), int(fw * 0.45)))
                    conf = 0.7 if len(eyes) >= 1 else 0.4

            faces.append({
                'x': int(fx), 'y': int(fy), 'w': int(fw), 'h': int(fh),
                'cx': int(fx + fw // 2), 'cy': int(fy + fh // 2),
                'area': int(fw * fh), 'confidence': conf,
            })
        return faces

    def _has_skin_region(self, roi_bgr) -> bool:
        """Check if a region contains skin-like colors."""
        try:
            ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
            lower = np.array([0, 130, 70], dtype=np.uint8)
            upper = np.array([255, 180, 135], dtype=np.uint8)
            mask = cv2.inRange(ycrcb, lower, upper)
            return np.count_nonzero(mask) / max(1, mask.size) > 0.10
        except Exception:
            return True

    def _write_embedded_prototxt(self, path):
        """Try multiple sources for the ResNet-10 SSD deploy.prototxt.
        This config file must match the caffemodel layer names exactly."""
        log = get_logger()
        import urllib.request

        urls = [
            "https://raw.githubusercontent.com/sr6033/face-detection-with-OpenCV-and-DNN/master/deploy.prototxt.txt",
            "https://raw.githubusercontent.com/keyurr2/face-detection/master/deploy.prototxt",
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
            "https://raw.githubusercontent.com/LZQthePlane/Face-detection-base-on-ResnetSSD/master/deploy.prototxt",
            "https://raw.githubusercontent.com/thegopieffect/computer_vision/master/CAFFE_DNN/deploy.prototxt.txt",
        ]

        for url in urls:
            try:
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                resp = urllib.request.urlopen(req, timeout=10)
                data = resp.read()
                # Validate: must contain Caffe layer definitions
                if len(data) > 5000 and b'Convolution' in data and b'detection_out' in data:
                    with open(path, 'wb') as f:
                        f.write(data)
                    log.log_stage('PERCEIVE', f'Prototxt downloaded from {url.split("/")[4]}')
                    return True
            except Exception:
                continue

        log.log_stage('PERCEIVE', 'All prototxt download URLs failed — DNN tier unavailable')
        return False

    def _dedupe_faces(self, faces: List[dict]) -> List[dict]:
        """Remove overlapping face detections, keeping highest confidence.

        Source priority for tie-breaking: full-frame YuNet > tiled YuNet >
        YOLO-assisted YuNet. Full-frame is the most reliable; the other two
        are recall boosters that occasionally hallucinate.

        Corroboration boost: when full-frame YuNet AND another detector
        agree on the same face, that's strong evidence — bump confidence.
        When tiled YuNet and YOLO-assisted YuNet agree without full-frame,
        DON'T boost — both share the small-face/edge-effect failure mode
        and can produce correlated false positives in busy backgrounds."""
        if len(faces) <= 1:
            return faces
        source_priority = {
            'yunet': 3,        # full-frame, primary
            'yunet_tile': 2,   # tiled
            'yunet_yolo': 1,   # person-cropped
        }

        def sort_key(f):
            return (f.get('confidence', 0),
                    source_priority.get(f.get('source', ''), 0))
        faces = sorted(faces, key=sort_key, reverse=True)

        keep = []
        for f in faces:
            is_dup = False
            for k in keep:
                # Intersection-over-min-area
                ox = max(0, min(f['x']+f['w'], k['x']+k['w']) - max(f['x'], k['x']))
                oy = max(0, min(f['y']+f['h'], k['y']+k['h']) - max(f['y'], k['y']))
                overlap = ox * oy
                min_area = min(f['area'], k['area'])
                # Tightened from 0.30 → 0.25: more aggressive merging cuts
                # down on near-duplicate ghosts that survived previous dedupe
                if min_area > 0 and overlap / min_area > 0.25:
                    is_dup = True
                    # Only boost confidence when full-frame YuNet corroborates.
                    # tile+yolo agreement alone can be a shared hallucination.
                    if (f.get('source') != k.get('source')
                            and (f.get('source') == 'yunet'
                                 or k.get('source') == 'yunet')):
                        k['confidence'] = round(min(0.99, k['confidence'] + 0.05), 3)
                    break
            if not is_dup:
                keep.append(f)
        return keep


