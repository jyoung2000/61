"""Export-quality fixes from the run-54 export audit:

1. LOOP: overlapping editor segments were emitted into BOTH speed-timeline
   entries, so the export re-played the overlap. The timeline builder now
   clamps each segment to the already-covered position.
2. STUTTER: active-word exports forced ``-r 30`` on any source — an uneven
   4:5 pull-down on 23.976fps (every 4th frame doubled). The rate is now an
   exact 2x of the probed source fps, and only for sub-30fps sources.
3. SWAY: dense keyframe tracks (the preview's own camera path) were
   downsampled then STEP-interpolated (~0.5s stair-jumps), and sparse
   backend keyframes smoothstep-panned across whole inter-keyframe gaps for
   jumps under the old threshold. Dense sets now interpolate linearly;
   every backend position change becomes hold-then-snap (threshold=1).
4. SEO: the "(no speech in this segment)" prompt placeholder could become a
   clip title, and a failed/empty LLM SEO left cards with no title. Both
   are now placeholder-proofed with a deterministic fallback ladder.
5. Sidecar: the per-export SEO sidecar is CSV (headers + one row).
"""

import csv
import io

from backend.services.clip_exporter import (
    _active_word_output_rate,
    _build_crop_x_expr,
    _build_speed_timeline,
    _insert_snap_transitions,
)
from backend.services.clip_seo_sidecar import (
    format_clip_seo_csv,
    sidecar_path_for,
)
from backend.services.pipeline import (
    _fallback_seo_title,
    _clip_visual_context,
    _is_timestamp_title,
)
from backend.services.reframer_bridge import to_fez_clips


# ── 1. Speed timeline: overlapping segments can't duplicate content ──────

def test_speed_timeline_clamps_overlapping_segments():
    # Two segments overlap 9.5-10.5 (drag drift). Without the clamp the
    # overlap appeared in both entries → re-played in the export.
    segs = [
        {"start": 100.0, "end": 110.5, "speed": 2.0},
        {"start": 109.5, "end": 120.0, "speed": 1.0},
    ]
    tl = _build_speed_timeline(segs, 20.0, 1.0, 1.0, clip_start=100.0)
    for a, b in zip(tl, tl[1:]):
        assert b["start"] >= a["end"] - 1e-9, (a, b)
    # Full coverage is preserved.
    assert tl[0]["start"] == 0.0
    assert abs(tl[-1]["end"] - 20.0) < 1e-9


def test_speed_timeline_untouched_when_no_overlap():
    segs = [
        {"start": 100.0, "end": 105.0, "speed": 2.0},
        {"start": 105.0, "end": 110.0, "speed": 0.5},
    ]
    tl = _build_speed_timeline(segs, 20.0, 1.0, 1.0, clip_start=100.0)
    assert [(t["start"], t["end"]) for t in tl] == [
        (0.0, 5.0), (5.0, 10.0), (10.0, 20.0)]


# ── 2. Active-word output rate: exact 2x, never a judder resample ────────

def test_active_word_rate_doubles_cinematic_sources():
    assert _active_word_output_rate(23.976) == "47.952"
    assert _active_word_output_rate(25.0) == "50.000"


def test_active_word_rate_leaves_fast_or_unknown_sources_alone():
    assert _active_word_output_rate(30.0) is None
    assert _active_word_output_rate(59.94) is None
    assert _active_word_output_rate(None) is None


# ── 3. Crop interpolation ────────────────────────────────────────────────

def test_dense_keyframes_use_linear_interpolation():
    # 120 keyframes at 0.25s — a dense camera track. Old behavior: forced
    # step after downsampling (stair-jumps). Now: piecewise linear.
    kfs = [(i * 0.25, 30 + (i % 40)) for i in range(120)]
    expr = _build_crop_x_expr(kfs, max_offset=800, src_w=1920, crop_w=606)
    assert "clip((t-" in expr           # linear ramp segments
    assert "st(0" not in expr           # no smoothstep register ops
    # Depth-limited: still bounded under the ffmpeg nesting limit.
    assert expr.count("if(") <= 80


def test_sparse_step_mode_still_steps():
    kfs = [(0.0, 30), (5.0, 70), (10.0, 30)]
    expr = _build_crop_x_expr(kfs, max_offset=800, src_w=1920, crop_w=606,
                              step_mode=True)
    assert "clip((t-" not in expr
    assert "st(0" not in expr


def test_snap_transitions_hold_then_settle_at_trigger():
    # A 10-unit move over 4s used to smoothstep-pan the whole gap (sway).
    # Now: hold → one deliberate pan that SETTLES exactly at the trigger.
    kfs = [(0.0, 40), (4.0, 50)]
    out = _insert_snap_transitions(kfs, jump_threshold=1)
    assert len(out) == 3
    hold_end, settle = out[1], out[2]
    assert hold_end[1] == 40 and settle == (4.0, 50)
    move = settle[0] - hold_end[0]
    assert 0.4 <= move <= 0.9                    # deliberate, human-paced


def test_snap_transitions_duration_scales_with_distance():
    small = _insert_snap_transitions([(0.0, 47), (4.0, 53)], jump_threshold=1)
    large = _insert_snap_transitions([(0.0, 41), (4.0, 59)], jump_threshold=1)
    small_move = small[-1][0] - small[1][0]
    large_move = large[-1][0] - large[1][0]
    assert large_move > small_move               # bigger pan = slower move
    assert large_move <= 0.9 + 1e-9              # but always bounded


def test_big_jumps_become_editorial_cuts_not_sweeps():
    # The user-reported case: 34 → 72 (38 units). A human editor CUTS to the
    # new framing — a whip-pan across 38% of the frame loses the viewer.
    out = _insert_snap_transitions([(0.0, 34), (6.0, 72)], jump_threshold=1)
    assert len(out) == 3
    hold_end, cut = out[1], out[2]
    assert hold_end[1] == 34 and cut == (6.0, 72)
    assert (cut[0] - hold_end[0]) <= 0.002       # instant cut, no sweep


def test_snap_transitions_preserve_scene_cut_pairs():
    # 1ms scene-cut pairs are hard cuts — never stretched into eases.
    kfs = [(0.0, 30), (5.0, 30), (5.001, 70), (9.0, 70)]
    out = _insert_snap_transitions(kfs, jump_threshold=1)
    assert (5.001, 70) in out                    # the cut edge survives intact


def test_ping_pong_bounce_is_suppressed():
    from backend.services.clip_exporter import _suppress_ping_pong
    # A→B→A within 1.2s: the bounce (B) is dropped; the hold carries through.
    kfs = [(0.0, 40), (5.0, 60), (6.2, 41), (12.0, 41)]
    out = _suppress_ping_pong(kfs)
    assert (5.0, 60) not in out
    # A real move that STAYS is untouched.
    kfs2 = [(0.0, 40), (5.0, 60), (11.0, 60)]
    assert _suppress_ping_pong(kfs2) == kfs2
    # Scene-cut pairs are never treated as bounces.
    kfs3 = [(0.0, 40), (5.0, 40), (5.001, 60), (5.8, 40)]
    assert (5.001, 60) in _suppress_ping_pong(kfs3)


def test_crop_expr_uses_ease_out_curve():
    # Sparse smoothstep mode now emits the ease-out cubic (3-3p+p²) form the
    # preview player uses, not the symmetric smoothstep (3-2p).
    expr = _build_crop_x_expr([(0.0, 30), (5.0, 70), (10.0, 30)],
                              max_offset=800, src_w=1920, crop_w=606)
    assert "(3-3*ld(0)+ld(0)*ld(0))" in expr
    assert "(3-2*" not in expr


# ── 4. SEO title guarantees ──────────────────────────────────────────────

def test_to_fez_clips_never_titles_with_placeholder():
    clips = to_fez_clips([{
        "start_s": 10.0, "end_s": 40.0, "duration_s": 30.0,
        "transcript_slice": "(no speech in this segment)",
        "composite_score": 0.5, "source": "signal",
    }])
    assert len(clips) == 1
    assert "no speech" not in clips[0]["title"].lower()
    assert "no speech" not in (clips[0]["suggested_caption"] or "").lower()
    assert "no speech" not in (clips[0]["hook_text"] or "").lower()


class _JobStub:
    filename = "MOBILE SUIT GUNDAM WING Episode 1.mp4"


def test_timestamp_titles_are_flagged_generic():
    assert _is_timestamp_title("3:20")
    assert _is_timestamp_title("highlight at 10:05")
    assert _is_timestamp_title("Clip @ 1:02-1:30")
    assert not _is_timestamp_title("Zechs strikes first")
    assert not _is_timestamp_title("The Gundam falls at dawn")


def test_fallback_seo_title_is_descriptive_never_a_timestamp():
    job = _JobStub()
    # Real clip title survives.
    assert _fallback_seo_title(
        {"title": "Zechs strikes first", "start_time": 65}, "", job) == "Zechs strikes first"
    # No-speech clip → describe from the VLM's on-screen hook.
    t = _fallback_seo_title(
        {"title": "Clip 5", "start_time": 500,
         "vlm_hook": "A Gundam pierces the Alliance line"}, "", job)
    assert t == "A Gundam pierces the Alliance line"
    # Only the editorial reason available → its first sentence.
    t = _fallback_seo_title(
        {"title": "(no speech in this segment)", "start_time": 500,
         "vlm_reason": "Zechs corners the enemy fighter over Eurasia. Then more."},
        "", job)
    assert t == "Zechs corners the enemy fighter over Eurasia"
    # Nothing clip-specific → the video's opening subject sentence (topical,
    # still descriptive — NOT a timestamp).
    t = _fallback_seo_title(
        {"title": "Clip 3", "start_time": 605}, "Humanity fled to space colonies. War follows.", job)
    assert t == "Humanity fled to space colonies"
    # Absolutely nothing → the source title, never a bare timestamp.
    class _Empty:
        filename = ""
    t = _fallback_seo_title({"start_time": 0}, "", _Empty())
    assert t == "Untitled clip"
    # Every branch is timestamp-free.
    for cd, summ in (({"start_time": 500, "vlm_hook": "A Gundam pierces the line"}, ""),
                     ({"start_time": 605}, "Humanity fled to space."),
                     ({"start_time": 0}, "")):
        assert not _is_timestamp_title(_fallback_seo_title(cd, summ, job))


def test_visual_context_grounds_no_speech_clips():
    ctx = _clip_visual_context({
        "vlm_hook": "A Gundam pierces the Alliance line",
        "vlm_reason": "It transforms mid-dive and downs two Aries.",
        "why_this_works": "Instant spectacle.",
    })
    assert "A Gundam pierces the Alliance line" in ctx
    assert "downs two Aries" in ctx
    # A placeholder-only clip yields no misleading context.
    assert _clip_visual_context({"vlm_hook": "(no speech in this segment)"}) == ""


# ── 5. CSV sidecar ───────────────────────────────────────────────────────

def _sample_job_and_clip():
    job = {
        "job_id": "job-1", "filename": "episode1.mp4",
        "clips": [], "summary": None,
    }
    clip = {
        "id": 1, "title": "Zechs vs the Gundam",
        "seo_title": "Zechs vs the Gundam — the first clash",
        "viral_score": 88, "hook_score": 90, "flow_score": 85,
        "value_score": 80, "trend_score": 75,
        "viral_score_reasoning": "Strong open, clean arc.",
        "platform": "tiktok",
        "suggested_caption": "The first clash.",
        "hook_text": "It was definitely a Gundam.",
        "seo_tags": ["gundam", "anime edit"],
        "seo_description": "The moment the war changes.",
        "seo_platform_tips": "Post 6-9pm.",
        "why_this_works": "Instant stakes.",
        "start_time": 600.0, "end_time": 640.0,
        "seo_by_platform": {
            "tiktok": {"title": "T-title", "description": "T-desc",
                       "tags": ["t1", "t2"], "platform_tips": "tips",
                       "primary_keyword": "gundam wing", "hook": "hook!",
                       "keywords": ["mecha", "90s anime"]},
        },
    }
    transcript = [
        {"start": 601.0, "end": 604.0, "text": "It was definitely a Gundam.",
         "speaker": "Speaker 2"},
    ]
    return job, clip, transcript


def test_csv_sidecar_headers_and_row():
    job, clip, transcript = _sample_job_and_clip()
    out = format_clip_seo_csv(
        job, clip, "[1080P] Zechs vs the Gundam.mp4",
        export_info={"start": 600.0, "end": 640.0, "export_quality": "1080p",
                     "aspect_ratio": "9:16", "subtitles_enabled": True,
                     "job_id": "job-1"},
        transcript=transcript,
    )
    rows = list(csv.reader(io.StringIO(out)))
    assert len(rows) == 2, "exactly one header row + one data row"
    header, data = rows
    assert len(header) == len(data)
    rec = dict(zip(header, data))
    # Core fields present with headers.
    assert rec["clip_file"] == "[1080P] Zechs vs the Gundam.mp4"
    assert rec["seo_title"] == "Zechs vs the Gundam — the first clash"
    assert rec["viral_score"] == "88"
    assert rec["tags"] == "gundam; anime edit"
    assert rec["hashtags"] == "#gundam #animeedit"
    assert rec["export_quality"] == "1080p"
    assert rec["aspect_ratio"] == "9:16"
    assert rec["subtitles_enabled"] == "yes"
    assert rec["clip_start"] == "10:00" and rec["clip_end"] == "10:40"
    # Per-platform flattening.
    assert rec["tiktok_title"] == "T-title"
    assert rec["tiktok_tags"] == "t1; t2"
    assert rec["tiktok_primary_keyword"] == "gundam wing"
    # Captions joined single-line.
    assert "It was definitely a Gundam." in rec["captions_transcript"]
    assert "\n" not in rec["captions_transcript"]


def test_sidecar_path_is_csv():
    assert sidecar_path_for("/x/[1080P] Title.mp4") == "/x/[1080P] Title.csv"
