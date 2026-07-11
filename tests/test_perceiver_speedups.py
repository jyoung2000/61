"""Perceiver hot-path fixes from the 128-min run profile (0.72 s/sample).

  * The seek-vs-grab threshold is now SAMPLE-RATE-AWARE: at 0.23 fps sampling
    the old flat 60-frame threshold made every one of 1800 samples take a
    hard CAP_PROP_POS_FRAMES seek (keyframe rewind + NVDEC flush) and the
    LK gap-bridging branch never ran — the cause of 1-2-sample face-track
    fragments.
  * The no-face branch reuses the subject boxes detect()'s own YOLO pass
    already computed for the frame — no second identical inference.
  * The YOLO-assisted crop pass skips person bboxes whose head zone is
    already covered by a dominant full-frame face (with the corroboration
    bump preserved).
"""

import sys
import types

import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))


# ─────────────────────────────────────────────────────────────────────────────
# Seek threshold math (mirrors the perceiver's inline computation)
# ─────────────────────────────────────────────────────────────────────────────

def _seek_gap_for(sample_times_ms, fps, floor=60, cap=450):
    seek_gap = floor
    if len(sample_times_ms) >= 2:
        diffs = sorted(b - a for a, b in zip(sample_times_ms, sample_times_ms[1:]))
        stride = int(round((diffs[len(diffs) // 2] / 1000.0) * max(fps, 1e-6)))
        seek_gap = min(max(floor, int(2.5 * stride)), cap)
    return seek_gap


def test_long_video_stride_stays_on_grab_path():
    # 128-min run: 4265 ms stride at 30 fps = 128-frame gaps. The old flat
    # 60 threshold seeked every sample; the stride-aware one must not.
    times = list(range(0, 7_680_000, 4265))
    gap_frames = int(4.265 * 30)
    assert gap_frames > 60                     # old behavior: hard seek
    assert _seek_gap_for(times, 30.0) >= gap_frames   # new: grab path


def test_dense_sampling_keeps_the_floor():
    times = list(range(0, 60_000, 800))        # 1.25 fps sampling
    assert _seek_gap_for(times, 30.0) == 60    # floor unchanged


def test_huge_strides_still_seek():
    times = list(range(0, 7_680_000, 60_000))  # one sample per minute
    sg = _seek_gap_for(times, 30.0)
    assert sg == 450                           # capped — 1800-frame gap seeks
    assert int(60.0 * 30) > sg


# ─────────────────────────────────────────────────────────────────────────────
# _uncovered_person_bboxes geometry (no cv2 needed — pure python method)
# ─────────────────────────────────────────────────────────────────────────────

def _fd():
    from backend.services.reframer_face import FaceDetector
    return FaceDetector.__new__(FaceDetector)


def test_covered_person_is_skipped_and_face_bumped():
    fd = _fd()
    person = (100, 100, 300, 500)              # ph=400, head zone 60..240
    face = {"cx": 200, "cy": 150, "h": 120, "confidence": 0.6}
    out = fd._uncovered_person_bboxes([person], [face])
    assert out == []                           # covered → crop pass skipped
    assert face["confidence"] == pytest.approx(0.65)   # corroboration bump


def test_small_face_does_not_cover():
    fd = _fd()
    person = (100, 100, 300, 500)              # zone height ~180
    face = {"cx": 200, "cy": 150, "h": 40, "confidence": 0.6}   # < 50% zone
    out = fd._uncovered_person_bboxes([person], [face])
    assert out == [person]                     # crop pass still runs


def test_face_outside_head_zone_does_not_cover():
    fd = _fd()
    person = (100, 100, 300, 500)
    face = {"cx": 200, "cy": 400, "h": 150, "confidence": 0.6}  # chest level
    assert fd._uncovered_person_bboxes([person], [face]) == [person]


def test_no_faces_returns_all_persons():
    fd = _fd()
    persons = [(0, 0, 100, 200), (200, 0, 300, 200)]
    assert fd._uncovered_person_bboxes(persons, []) == persons
