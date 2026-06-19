"""Test: the cached-RenderPlan parser reconstructs the planner's subject track.

The analysis pipeline persists the reframer's smooth per-frame camera track as
a Fez ``RenderPlan`` (``TRACKING_CROP`` ops carrying ``motion_path`` keypoints)
to ``render_plan.json``. ``render_plan_keyframes.keyframes_from_cached_render_plan``
turns that back into the ``(relative_time_sec, subject_x_pct)`` keyframes the
FFmpeg crop pipeline consumes, so the export follows the planner exactly instead
of re-deriving a coarser cluster-snap crop.

These tests build a real ``RenderPlan`` with the production dataclasses,
serialize it exactly as the pipeline does (``to_dict``), and verify the parse
round-trips the crop CENTRE position and seeds clip-relative boundaries.
"""

import sys
import types

# render_plan is stdlib-only, but stub cv2 defensively in case an __init__
# shim pulls it in this environment.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from backend.services.render_plan import (  # noqa: E402
    RenderPlan, RenderOp, RenderOpKind, Rect, MotionKeypoint,
)
from backend.services.render_plan_keyframes import (  # noqa: E402
    keyframes_from_cached_render_plan,
)

SRC_W, SRC_H = 1920, 1080
CROP_W = 608                      # 9:16 of full height
CW = CROP_W / SRC_W               # normalized crop width (~0.3167)


def _rect_for_left(left_px: int) -> Rect:
    """Build a normalized crop rect whose LEFT edge is ``left_px`` source px."""
    return Rect(x=left_px / SRC_W, y=0.0, w=CW, h=1.0)


def _expected_center_pct(left_px: int) -> int:
    return round((left_px + CROP_W / 2) / SRC_W * 100)


def _plan_dict():
    """One 10s TRACKING_CROP op; crop left pans 0 -> 656 -> 328 px."""
    op = RenderOp(
        kind=RenderOpKind.TRACKING_CROP,
        start_sec=0.0,
        end_sec=10.0,
        primary_rect=_rect_for_left(0),
        motion_path=[
            MotionKeypoint(t=0.0, rect=_rect_for_left(0)),    # center ~16%
            MotionKeypoint(t=5.0, rect=_rect_for_left(656)),  # center  50%
            MotionKeypoint(t=10.0, rect=_rect_for_left(328)),  # center ~33%
        ],
    )
    plan = RenderPlan(
        source_width=SRC_W, source_height=SRC_H,
        target_width=1080, target_height=1920,
        total_duration_sec=10.0, fps=30.0, ops=[op],
    )
    return plan.to_dict()


def test_full_clip_round_trips_crop_centers():
    kfs = keyframes_from_cached_render_plan(_plan_dict(), clip_start=0.0, clip_end=10.0)
    # Times preserved, positions are the crop CENTRE as a 0-100 percentage.
    assert kfs[0] == (0.0, _expected_center_pct(0))      # (0.0, 16)
    assert (5.0, _expected_center_pct(656)) in kfs        # (5.0, 50)
    assert kfs[-1] == (10.0, _expected_center_pct(328))   # (10.0, 33)
    # It must be a genuine dynamic track, not a collapsed static value.
    assert len({x for _, x in kfs}) >= 3


def test_sub_clip_is_rebased_and_boundary_seeded():
    # Clip [2.5, 7.5] contains only the interior t=5 keypoint; the parser must
    # seed t=0 and t=clip_dur by interpolating the planner's path at the edges.
    kfs = keyframes_from_cached_render_plan(_plan_dict(), clip_start=2.5, clip_end=7.5)
    assert kfs[0][0] == 0.0                       # rebased to clip-relative
    assert kfs[-1][0] == 5.0                      # clip_dur = 7.5 - 2.5
    # Seed at clip_start=2.5 = halfway between center 16% (t0) and 50% (t5).
    assert kfs[0][1] == round((_expected_center_pct(0) + _expected_center_pct(656)) / 2)
    # The interior real keypoint (abs t=5 -> rel t=2.5) keeps its 50% center.
    assert (2.5, _expected_center_pct(656)) in kfs


def test_failsoft_on_missing_or_malformed_plan():
    assert keyframes_from_cached_render_plan(None, 0.0, 10.0) == []
    assert keyframes_from_cached_render_plan({}, 0.0, 10.0) == []
    assert keyframes_from_cached_render_plan({"ops": []}, 0.0, 10.0) == []
    # Structurally broken ops must not raise.
    assert keyframes_from_cached_render_plan({"ops": [{"motion_path": "nope"}]}, 0.0, 10.0) == []


def test_static_op_holds_its_center():
    # An op with no motion_path falls back to its primary_rect centre, held
    # across the window (a static crop position).
    op = RenderOp(
        kind=RenderOpKind.CROP, start_sec=0.0, end_sec=4.0,
        primary_rect=_rect_for_left(656), motion_path=[],
    )
    plan = RenderPlan(
        source_width=SRC_W, source_height=SRC_H,
        target_width=1080, target_height=1920,
        total_duration_sec=4.0, fps=30.0, ops=[op],
    )
    kfs = keyframes_from_cached_render_plan(plan.to_dict(), 0.0, 4.0)
    assert kfs[0] == (0.0, _expected_center_pct(656))
    assert len({x for _, x in kfs}) == 1   # held static at 50%
