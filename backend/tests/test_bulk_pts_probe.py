"""Exact frame timestamps from the PRIMARY extraction pass.

The extraction filter graphs now end in ``showinfo``, so the pts_time of
every written frame is parsed from the extraction run's own stderr — no
second ffmpeg decode over the extracted JPEGs (the old ``_bulk_probe_pts``),
and no ≤240-frame cap on exact timestamps.
"""

import asyncio
import os
import shutil
import subprocess
import tempfile

import pytest


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


SHOWINFO_BLOB = b"""ffmpeg version 6.0 Copyright (c) 2000-2023
[Parsed_showinfo_3 @ 0x55d000] config in time_base: 1/15360, frame_rate: 30/1
[Parsed_showinfo_3 @ 0x55d000] n:   0 pts:      0 pts_time:0       duration:512
[Parsed_showinfo_3 @ 0x55d000] n:   1 pts: 184320 pts_time:12.04   duration:512
[Parsed_showinfo_3 @ 0x55d000] n:   2 pts: 368640 pts_time:24.5    duration:512
[Parsed_showinfo_3 @ 0x55d000] color_range:tv color_space:bt709
frame=    3 fps=0.0 q=-0.0 Lsize=N/A time=00:00:24.50 bitrate=N/A speed= 341x
"""


def test_parse_showinfo_pts_ordered_values():
    import backend.services.frame_extractor as fe
    assert fe._parse_showinfo_pts(SHOWINFO_BLOB) == [0.0, 12.04, 24.5]


def test_parse_showinfo_pts_empty_and_noise():
    import backend.services.frame_extractor as fe
    assert fe._parse_showinfo_pts(b"") == []
    assert fe._parse_showinfo_pts(b"stall: 0 frames produced") == []
    # A pts_time on a non-showinfo line must not be picked up.
    assert fe._parse_showinfo_pts(b"[warn] pts_time:9.9 out of order") == []


def test_filters_end_with_showinfo_after_format():
    import backend.services.frame_extractor as fe
    for f in (fe._build_scene_filter(10), fe._build_interval_filter(10)):
        assert f.endswith("format=pix_fmts=yuvj420p,showinfo")


def test_error_extraction_ignores_showinfo_noise():
    """_extract_ffmpeg_error keyword matching must still surface real errors
    when the stderr tail is full of showinfo lines."""
    import backend.services.frame_extractor as fe
    blob = SHOWINFO_BLOB + b"\n/data/in.mp4: No space left on device\n"
    assert "No space left on device" in fe._extract_ffmpeg_error(blob)
    # And showinfo-only stderr yields no false error lines (falls back to tail).
    tail = fe._extract_ffmpeg_error(SHOWINFO_BLOB)
    assert "pts_time" in tail  # tail fallback, not a keyword match


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")
def test_extract_frames_assigns_exact_pts_from_extraction_pass():
    """End-to-end: extract_frames on a synthetic source gets its timestamps
    from the extraction pass's showinfo output (no second decode)."""
    import backend.services.frame_extractor as fe

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", "testsrc=duration=10:size=160x120:rate=10",
             "-pix_fmt", "yuv420p", src],
            check=True, capture_output=True,
        )
        out_dir = os.path.join(tmp, "frames")
        frames, _cuts = asyncio.run(fe.extract_frames(
            src, out_dir, sample_rate=2, video_duration=10.0,
        ))
        assert frames
        ts = [f.timestamp for f in frames]
        assert ts == sorted(ts)
        # showinfo pts land on the true 2s select grid (±1 frame @10fps).
        for t in ts:
            assert abs(t - round(t / 2.0) * 2.0) <= 0.15
