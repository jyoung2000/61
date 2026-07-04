"""Perceiver performance changes (Tasks 5-7 of the speed pass).

  * Task 5 — fail-soft hardware-accelerated VideoCapture open.
  * Task 6 — cached saliency meshgrids (identical numerics).
  * Task 7 — opt-in ``REFRAMER_FIX_PREV_FRAME_MOTION`` (flag off = legacy
    behavior byte-identical; flag on = 2nd+ faces get real motion and the
    previous-frame reference updates on zero-face samples too).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from backend.config import settings
from backend.services import reframer_perceiver as rp


class _StubLog:
    def __init__(self):
        self.lines = []

    def log_stage(self, stage, msg, **kw):
        self.lines.append(msg)

    def log_error(self, stage, msg, **kw):
        self.lines.append(msg)


def _bare_perceiver(path="/nonexistent/video.mp4") -> rp.Perceiver:
    """A Perceiver without running __init__ (no model downloads)."""
    p = rp.Perceiver.__new__(rp.Perceiver)
    p.path = path
    p._prev_frame_for_face_motion = None
    return p


# ── Task 5: hw-accel capture with sw fallback ────────────────────────


class _FakeCap:
    """Stands in for cv2.VideoCapture in both constructor signatures."""

    def __init__(self, opened=True, read_ok=True):
        self._opened = opened
        self._read_ok = read_ok
        self.released = False
        self._pos = 0

    def isOpened(self):
        return self._opened

    def read(self):
        if self._read_ok:
            self._pos += 1
            return True, np.zeros((4, 4, 3), np.uint8)
        return False, None

    def set(self, prop, val):
        self._pos = int(val)
        return True

    def get(self, prop):
        return self._pos

    def release(self):
        self.released = True


def test_hw_open_failure_falls_back_to_software(monkeypatch):
    """The list-param (hw) constructor signature fails → plain constructor."""
    made = []

    def _fake_video_capture(path, *args):
        if args:  # hw signature: (path, CAP_FFMPEG, [params])
            raise cv2.error("no hwaccel in this build")
        cap = _FakeCap()
        made.append(("sw", cap))
        return cap

    monkeypatch.setattr(rp.cv2, "VideoCapture", _fake_video_capture)
    monkeypatch.setattr(settings, "REFRAMER_CV2_HWACCEL", True, raising=False)
    p = _bare_perceiver()
    log = _StubLog()
    cap = p._open_capture(log)
    assert p._decode_path == "sw"
    assert made and cap is made[0][1]
    assert any("decode=sw" in ln for ln in log.lines)


def test_hw_open_unreadable_capture_falls_back(monkeypatch):
    """hw capture opens but its first read fails → released + sw reopen."""
    hw_caps = []

    def _fake_video_capture(path, *args):
        if args:
            cap = _FakeCap(opened=True, read_ok=False)
            hw_caps.append(cap)
            return cap
        return _FakeCap()

    monkeypatch.setattr(rp.cv2, "VideoCapture", _fake_video_capture)
    monkeypatch.setattr(settings, "REFRAMER_CV2_HWACCEL", True, raising=False)
    p = _bare_perceiver()
    cap = p._open_capture(_StubLog())
    assert p._decode_path == "sw"
    assert hw_caps and hw_caps[0].released
    assert cap.isOpened()


def test_hw_open_success_rewinds_to_frame_zero(monkeypatch):
    def _fake_video_capture(path, *args):
        return _FakeCap()

    monkeypatch.setattr(rp.cv2, "VideoCapture", _fake_video_capture)
    monkeypatch.setattr(settings, "REFRAMER_CV2_HWACCEL", True, raising=False)
    p = _bare_perceiver()
    log = _StubLog()
    cap = p._open_capture(log)
    assert p._decode_path == "hw(any)"
    # The probe read advanced position; the winner must be rewound to 0 so
    # the sampling loop sees exactly the same frames as the sw path.
    assert int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == 0
    assert any("decode=hw(any)" in ln for ln in log.lines)


def test_hwaccel_setting_off_uses_plain_constructor(monkeypatch):
    calls = []

    def _fake_video_capture(path, *args):
        calls.append(args)
        return _FakeCap()

    monkeypatch.setattr(rp.cv2, "VideoCapture", _fake_video_capture)
    monkeypatch.setattr(settings, "REFRAMER_CV2_HWACCEL", False, raising=False)
    p = _bare_perceiver()
    p._open_capture(_StubLog())
    assert calls == [()]  # only the plain (path,) call
    assert p._decode_path == "sw"


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None,
                    reason="ffmpeg not installed")
def test_open_capture_parity_with_plain_capture(tmp_path):
    """Whichever decode path _open_capture picks on this machine, the capture
    the sampling loop consumes must be indistinguishable from the plain
    software one: same metadata (⇒ identical sample_times_ms, which derive
    only from duration/fps) and identical decoded frames."""
    import subprocess
    src = str(tmp_path / "clip.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc=duration=2:size=128x96:rate=10",
         "-pix_fmt", "yuv420p", src],
        check=True, capture_output=True)

    p = _bare_perceiver(src)
    cap = p._open_capture(_StubLog())
    ref = cv2.VideoCapture(src)
    try:
        for prop in (cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_WIDTH,
                     cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_COUNT):
            assert cap.get(prop) == ref.get(prop)
        for _ in range(5):
            ok_a, fa = cap.read()
            ok_b, fb = ref.read()
            assert ok_a == ok_b
            if ok_a:
                assert np.array_equal(fa, fb)
    finally:
        cap.release()
        ref.release()


# ── Task 6: cached saliency meshgrids ────────────────────────────────


def test_sal_meshgrid_cached_and_identical():
    p = _bare_perceiver()
    yy1, xx1 = p._sal_meshgrid(64)
    yy2, xx2 = p._sal_meshgrid(64)
    assert yy1 is yy2 and xx1 is xx2  # cached, no reallocation
    yy_ref, xx_ref = np.mgrid[0:64, 0:64]
    assert np.array_equal(yy1, yy_ref) and np.array_equal(xx1, xx_ref)
    # A different sal_size rebuilds.
    yy3, _ = p._sal_meshgrid(32)
    assert yy3.shape == (32, 32)


def test_speaker_bump_identical_with_cached_meshgrid():
    """The Gaussian bump computed from the cached grid equals the one from a
    fresh np.mgrid — the exact expression used in run()'s speaker-bump."""
    p = _bare_perceiver()
    sal_size, sp_x, sp_y = 64, 20, 40
    sigma = sal_size * 0.10

    yy, xx = p._sal_meshgrid(sal_size)
    bump_cached = np.exp(-((xx - sp_x) ** 2 + (yy - sp_y) ** 2)
                         / (2.0 * sigma * sigma)).astype(np.float32)
    yy_f, xx_f = np.mgrid[0:sal_size, 0:sal_size]
    bump_fresh = np.exp(-((xx_f - sp_x) ** 2 + (yy_f - sp_y) ** 2)
                        / (2.0 * sigma * sigma)).astype(np.float32)
    assert np.array_equal(bump_cached, bump_fresh)


# ── Task 7: opt-in prev-frame motion fix ─────────────────────────────


def _perceiver_for_faces(faces_by_call):
    """Perceiver stub whose face detector returns canned detections."""
    p = _bare_perceiver()
    it = iter(faces_by_call)
    p.face_detector = SimpleNamespace(
        detect=lambda frame: next(it),
        compute_embedding=lambda frame, face: None,
    )
    return p


def _frames_with_moving_faces():
    """Two 100x100 frames where BOTH face regions change between frames."""
    f1 = np.zeros((100, 100), np.uint8)
    f2 = np.zeros((100, 100), np.uint8)
    f2[10:40, 10:40] = 200   # face A region changes
    f2[60:90, 60:90] = 120   # face B region changes
    return f1, f2


_TWO_FACES = [
    {"x": 10, "y": 10, "w": 30, "h": 30, "confidence": 0.9},
    {"x": 60, "y": 60, "w": 30, "h": 30, "confidence": 0.9},
]


def _detect_two_frames(p, f1, f2):
    bgr1 = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
    bgr2 = cv2.cvtColor(f2, cv2.COLOR_GRAY2BGR)
    p._detect_faces_fast(f1, bgr1, 20, 80, 100, 100, 1.0)
    return p._detect_faces_fast(f2, bgr2, 20, 80, 100, 100, 1.0)


def test_flag_off_second_face_motion_is_zero(monkeypatch):
    """Legacy behavior locked in: with the flag OFF the 2nd face compares the
    current frame against itself → motion == 0 even though its pixels moved."""
    monkeypatch.setattr(settings, "REFRAMER_FIX_PREV_FRAME_MOTION", False,
                        raising=False)
    f1, f2 = _frames_with_moving_faces()
    p = _perceiver_for_faces([list(_TWO_FACES), list(_TWO_FACES)])
    out = _detect_two_frames(p, f1, f2)
    assert len(out) == 2
    assert out[0]["motion"] > 0.0        # 1st face sees the real prev frame
    assert out[1]["motion"] == 0.0       # 2nd face: prev == current (the bug)
    assert out[1]["mouth_motion"] == 0.0


def test_flag_on_both_faces_get_nonzero_motion(monkeypatch):
    monkeypatch.setattr(settings, "REFRAMER_FIX_PREV_FRAME_MOTION", True,
                        raising=False)
    f1, f2 = _frames_with_moving_faces()
    p = _perceiver_for_faces([list(_TWO_FACES), list(_TWO_FACES)])
    out = _detect_two_frames(p, f1, f2)
    assert len(out) == 2
    assert out[0]["motion"] > 0.0
    assert out[1]["motion"] > 0.0        # fixed: real prev frame for ALL faces
    assert out[1]["mouth_motion"] > 0.0


def test_flag_on_zero_face_frames_update_prev_reference(monkeypatch):
    monkeypatch.setattr(settings, "REFRAMER_FIX_PREV_FRAME_MOTION", True,
                        raising=False)
    f1, _ = _frames_with_moving_faces()
    p = _perceiver_for_faces([[]])
    bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
    p._detect_faces_fast(f1, bgr, 20, 80, 100, 100, 1.0)
    assert p._prev_frame_for_face_motion is f1  # updated on the 0-face path


def test_flag_off_zero_face_frames_leave_prev_reference(monkeypatch):
    monkeypatch.setattr(settings, "REFRAMER_FIX_PREV_FRAME_MOTION", False,
                        raising=False)
    f1, _ = _frames_with_moving_faces()
    p = _perceiver_for_faces([[]])
    bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR)
    p._detect_faces_fast(f1, bgr, 20, 80, 100, 100, 1.0)
    assert p._prev_frame_for_face_motion is None  # legacy: never updated
