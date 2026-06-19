"""Tests: the reframer subsamples the heavy YOLO-World subject detector
(runs it every Nth frame, carrying subject bboxes forward) WITHOUT thinning
the per-frame YuNet face pass or motion. This is what cut the ~17-min face
stage roughly in half. A scene cut forces a fresh YOLO pass.
"""

import sys
import types

# reframer_face imports cv2 at module load; stub it (no OpenCV in CI).
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.reframer_face import FaceDetector


def _make_detector(stride):
    """A FaceDetector with the real detect()/stride logic but stubbed,
    call-counting sub-detectors (no models / cv2 needed)."""
    fd = FaceDetector.__new__(FaceDetector)
    fd.tier = "yunet"
    fd.confidence = 0.30
    fd._yolo_model = object()          # truthy → YOLO path active
    fd._yolo_stride = stride
    fd._yolo_det_count = 0
    fd._cached_person_bboxes = []
    fd._cached_nonhuman_raw = []
    fd._nonhuman_cache = []

    counts = {"yolo": 0, "yunet": 0, "tiled": 0}

    def _split(_frame):
        counts["yolo"] += 1
        return [], []                  # (person_bboxes, nonhuman_raw)

    def _yunet(_frame, _conf):
        counts["yunet"] += 1
        return []

    def _tiled(_frame, _conf):
        counts["tiled"] += 1
        return []

    fd._get_split_subject_bboxes = _split
    fd._detect_yunet = _yunet
    fd._detect_yunet_tiled = _tiled
    fd._detect_haar = lambda _f: []
    fd._update_nonhuman_cache = lambda _b: None
    fd._get_effective_nonhuman_bboxes = lambda _b: []
    return fd, counts


def test_yolo_runs_every_nth_frame_but_yunet_every_frame():
    fd, counts = _make_detector(stride=2)
    for _ in range(10):
        fd.detect(object())
    # YuNet face pass runs on EVERY frame (framing density unchanged)…
    assert counts["yunet"] == 10
    assert counts["tiled"] == 10
    # …while the heavy YOLO detector runs only every 2nd frame.
    assert counts["yolo"] == 5


def test_stride_one_restores_every_frame_yolo():
    fd, counts = _make_detector(stride=1)
    for _ in range(6):
        fd.detect(object())
    assert counts["yolo"] == 6
    assert counts["yunet"] == 6


def test_scene_cut_forces_fresh_yolo_pass():
    fd, counts = _make_detector(stride=3)
    fd.detect(object())          # frame 0 → YOLO runs (count 1)
    fd.detect(object())          # frame 1 → carried forward
    assert counts["yolo"] == 1
    fd.clear_nonhuman_cache()    # scene cut → reset counter + drop carried bboxes
    assert fd._cached_person_bboxes == []
    fd.detect(object())          # first frame of new scene → YOLO runs again
    assert counts["yolo"] == 2


def test_carry_forward_reuses_cached_person_bboxes():
    fd, counts = _make_detector(stride=2)
    # YOLO frame returns a person bbox; the skipped frame must reuse it.
    seen_person = []

    def _split(_frame):
        counts["yolo"] += 1
        return [(1, 2, 3, 4)], []

    # capture what the gate receives by stubbing the gate to record it
    def _gate(faces, person_bboxes):
        seen_person.append(list(person_bboxes))
        return faces

    fd._get_split_subject_bboxes = _split
    fd._gate_by_person_bboxes = _gate
    # make YuNet return a face so the positive gate actually runs
    fd._detect_yunet = lambda _f, _c: [{"x": 0, "y": 0, "w": 5, "h": 5}]
    fd._detect_yunet_tiled = lambda _f, _c: []
    fd._detect_yolo_assisted_yunet = lambda _f, _c, _p: []
    fd._validate_live_action_faces = lambda faces, _f, _p: faces
    fd._dedupe_faces = lambda faces: faces

    fd.detect(object())          # YOLO frame → person bbox detected
    fd.detect(object())          # skip frame → must reuse the cached bbox
    assert counts["yolo"] == 1
    assert seen_person == [[(1, 2, 3, 4)], [(1, 2, 3, 4)]]


def test_default_stride_is_two():
    from backend.config import settings
    assert settings.REFRAMER_YOLO_STRIDE == 2
