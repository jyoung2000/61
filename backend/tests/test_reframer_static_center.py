"""Test: a static graphic (no face, no person, no motion) centers instead of
chasing the spectral-saliency EDGE.

A centered GUNDAM logo title card was cropping to the far left because, with no
face/person, the planner followed a single spectral-saliency peak that latches
onto the highest-contrast edge (the logo's wing-tip). On motion-less frames the
crop now centers; frames with motion still follow the salient action.
"""

import sys
import types

# reframer_planner pulls cv2 (via reframer_models); stub it so the pure helper
# imports without OpenCV in this environment.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.reframer_planner import _saliency_or_static_center_x  # noqa: E402

# 9:16 crop of a 640x360 source: crop_w ~= 203, max_x = 640-203 = 437,
# centered crop-x = 218 (matches the Inspector's "/ 437").
CROP_W, MAX_X = 203, 437
CENTER = MAX_X // 2  # 218


def test_static_graphic_centers_ignoring_edge_saliency():
    # Saliency peak on the logo's left edge, no motion → center, not the edge.
    assert _saliency_or_static_center_x(128, False, CROP_W, MAX_X) == CENTER


def test_far_left_saliency_peak_still_centers_when_static():
    # The exact symptom (peak near the left wing) → centered, not x=27.
    out = _saliency_or_static_center_x(30, False, CROP_W, MAX_X)
    assert out == CENTER
    assert out != 0


def test_motion_frame_still_follows_saliency():
    # With motion present, follow the salient action (peak - crop_w//2).
    assert _saliency_or_static_center_x(420, True, CROP_W, MAX_X) == 420 - CROP_W // 2


def test_disabled_restores_pure_saliency_framing():
    # center_static=False → old behavior even on static frames (clamps to 0).
    out = _saliency_or_static_center_x(30, False, CROP_W, MAX_X, center_static=False)
    assert out == max(0, 30 - CROP_W // 2) == 0
    assert out != CENTER


def test_centered_peak_unchanged_either_way():
    # A peak that's already centered stays centered regardless of motion.
    sal_cx = CENTER + CROP_W // 2          # peak whose crop-x == center
    assert _saliency_or_static_center_x(sal_cx, True, CROP_W, MAX_X) == CENTER
    assert _saliency_or_static_center_x(sal_cx, False, CROP_W, MAX_X) == CENTER
