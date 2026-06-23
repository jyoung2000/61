"""Tests for the per-clip SEO sidecar (.txt) written next to every export.

When a clip is exported to MP4, ClipAI also drops a human-readable ``.txt``
carrying the clip's viral score, title, suggested caption, hashtags,
recommended platform, per-platform SEO and the clip's captions. These pin
the formatter (defensive against dict vs pydantic, range-filtered captions)
and the best-effort writer.
"""

import os
import tempfile

from backend.services import clip_seo_sidecar as cs


def _job():
    clip = {
        "id": 3,
        "title": "The ONE Morning Habit",
        "viral_score": 87,
        "viral_score_composite": 84,
        "viral_score_reasoning": "Bold claim opens cold; tight payoff.",
        "hook_score": 90, "flow_score": 82, "value_score": 88, "trend_score": 76,
        "platform": "both",
        "suggested_caption": "You're doing mornings wrong",
        "hook_text": "You're doing mornings wrong.",
        "why_this_works": "Pattern interrupt + payoff.",
        "seo_title": "The ONE Habit That Changed My Mornings",
        "seo_description": "A 60-second routine that fixed my focus.",
        "seo_tags": ["morning routine", "#habits", "productivity"],
        "seo_platform_tips": "Post 7-9am.",
        "seo_by_platform": {
            "tiktok": {"title": "Mornings, fixed", "description": "Try this",
                       "tags": ["fyp"], "platform_tips": "Hook in 0.5s"},
        },
    }
    job = {
        "job_id": "abc123", "filename": "podcast_ep12.mp4", "clips": [clip],
        "transcript": [
            {"start": 9.0, "end": 12.0, "text": "Before the clip", "speaker": "Host"},
            {"start": 60.5, "end": 63.0, "text": "You're doing mornings wrong.", "speaker": "Host"},
            {"start": 63.2, "end": 67.0, "text": "Here's what I changed.", "speaker": "Host"},
            {"start": 200.0, "end": 202.0, "text": "After the clip", "speaker": "Host"},
        ],
    }
    return job, clip


_EXPORT = {"start": 60.0, "end": 95.0, "export_quality": "1080p",
           "aspect_ratio": "9:16", "subtitles_enabled": True, "job_id": "abc123"}


def test_sidecar_path_swaps_extension():
    assert cs.sidecar_path_for("/x/[1080P] A.mp4") == "/x/[1080P] A.txt"
    assert cs.sidecar_path_for("/x/clip.MP4") == "/x/clip.txt"


def test_format_contains_all_seo_sections():
    job, clip = _job()
    txt = cs.format_clip_seo_text(job, clip, "[1080P] The ONE Morning Habit.mp4",
                                  export_info=_EXPORT, transcript=job["transcript"])
    for needle in ("VIRAL SCORE", "Overall: 87 / 100", "Composite: 84 / 100",
                   "Hook 90", "RECOMMENDED PLATFORM", "TikTok & YouTube Shorts",
                   "CAPTION (social post)", "HOOK", "TAGS / HASHTAGS",
                   "DESCRIPTION", "PLATFORM TIPS", "WHY THIS WORKS",
                   "PER-PLATFORM SEO", "EXPORT DETAILS", "CAPTIONS / TRANSCRIPT",
                   "The ONE Habit That Changed My Mornings"):
        assert needle in txt, f"missing {needle!r}"


def test_tags_are_hashtagified_and_deduped_of_hashes():
    job, clip = _job()
    txt = cs.format_clip_seo_text(job, clip, "c.mp4", export_info=_EXPORT,
                                  transcript=job["transcript"])
    # Raw tags listed without leading '#', and a hashtag line with collapsed tokens.
    assert "morning routine, habits, productivity" in txt
    assert "#morningroutine #habits #productivity" in txt


def test_captions_are_range_filtered_and_clip_relative():
    job, clip = _job()
    txt = cs.format_clip_seo_text(job, clip, "c.mp4", export_info=_EXPORT,
                                  transcript=job["transcript"])
    # In-window lines present, out-of-window lines absent.
    assert "[0:00] Host: You're doing mornings wrong." in txt
    assert "[0:03] Host: Here's what I changed." in txt
    assert "Before the clip" not in txt
    assert "After the clip" not in txt


def test_platform_label_mapping():
    assert cs._platform_label("youtube_shorts") == "YouTube Shorts"
    assert cs._platform_label("x") == "X (Twitter)"
    assert cs._platform_label("unknown_thing") == "unknown_thing"
    assert cs._platform_label("") == ""


def test_write_creates_file_and_returns_path():
    job, _clip = _job()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "[1080P] The ONE Morning Habit.mp4")
        path = cs.write_clip_seo_sidecar(job, 3, out, export_info=_EXPORT)
        assert path == os.path.join(d, "[1080P] The ONE Morning Habit.txt")
        assert os.path.isfile(path)
        assert "VIRAL SCORE" in open(path, encoding="utf-8").read()


def test_write_is_best_effort_for_unknown_clip():
    """A clip id with no matching candidate still writes export details +
    captions rather than failing the export."""
    job, _clip = _job()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "mystery.mp4")
        path = cs.write_clip_seo_sidecar(job, 999, out, export_info=_EXPORT)
        assert path and os.path.isfile(path)
        body = open(path, encoding="utf-8").read()
        assert "EXPORT DETAILS" in body
        # Captions still resolve from the job transcript within the window.
        assert "You're doing mornings wrong." in body


def test_falls_back_to_translated_transcript_for_captions():
    job, _clip = _job()
    job["translated_transcript"] = [
        {"start": 61.0, "end": 64.0, "text": "TRANSLATED LINE", "speaker": "Host"},
    ]
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "x.mp4")
        path = cs.write_clip_seo_sidecar(job, 3, out, export_info=_EXPORT)
        body = open(path, encoding="utf-8").read()
        assert "TRANSLATED LINE" in body
        # The source line should NOT appear — translated takes precedence.
        assert "Here's what I changed." not in body
