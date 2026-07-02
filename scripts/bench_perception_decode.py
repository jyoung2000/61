#!/usr/bin/env python3
"""Benchmark perception decode strategies on a real video (audit Phase 5.1).

Measures wall time to sample a video at the perception rate (default
5 fps) and downscale to detection resolution, using three strategies:

  seek   — CAP_PROP_POS_FRAMES for every gap > 5 (the OLD behavior:
           at 5 fps sampling of 30 fps video this seeks EVERY sample,
           and each seek re-decodes from the previous keyframe)
  grab   — sequential cap.grab() for gaps up to REFRAMER_SEEK_GAP_FRAMES
           (the NEW default: decode-skip without color conversion)
  ffmpeg — piped ffmpeg decode (-vf fps=...,scale=...) → rawvideo pipe;
           decode+downscale happen once inside ffmpeg. Add --hwaccel to
           try NVDEC.

Run on the deployment box (needs OpenCV + ffmpeg + a fixture):

    python scripts/bench_perception_decode.py /path/to/30min_1080p.mp4
    python scripts/bench_perception_decode.py video.mp4 --hwaccel --minutes 5

Report the numbers in CHANGES.md / the tuning issue; if the ffmpeg pipe
wins consistently on your content, that's the vote for making it the
default sampler.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def bench_opencv(path: str, sample_fps: float, det_w: int, det_h: int,
                 seek_gap: int, max_seconds: float) -> tuple[int, float]:
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur_ms = min(max_seconds * 1000, total / fps * 1000)
    interval_ms = 1000.0 / sample_fps
    times = [int(i * interval_ms) for i in range(int(dur_ms / interval_ms))]

    n = 0
    t0 = time.monotonic()
    for time_ms in times:
        target = int(time_ms / 1000.0 * fps)
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        gap = target - pos
        if gap < 0 or gap > seek_gap:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(target, total - 1))
        elif gap > 1:
            for _ in range(gap - 1):
                cap.grab()
        ret, frame = cap.read()
        if not ret:
            break
        cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        n += 1
    cap.release()
    return n, time.monotonic() - t0


def bench_ffmpeg_pipe(path: str, sample_fps: float, det_w: int, det_h: int,
                      max_seconds: float, hwaccel: bool) -> tuple[int, float]:
    import numpy as np
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if hwaccel:
        cmd += ["-hwaccel", "cuda"]
    cmd += [
        "-t", str(max_seconds), "-i", path,
        "-vf", f"fps={sample_fps},scale={det_w}:{det_h}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = det_w * det_h * 3
    n = 0
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=frame_bytes * 4)
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        np.frombuffer(buf, dtype=np.uint8).reshape(det_h, det_w, 3)
        n += 1
    proc.wait()
    return n, time.monotonic() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--sample-fps", type=float, default=5.0)
    ap.add_argument("--det-w", type=int, default=640)
    ap.add_argument("--det-h", type=int, default=360)
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="cap the benchmark to the first N minutes")
    ap.add_argument("--hwaccel", action="store_true",
                    help="add -hwaccel cuda to the ffmpeg pipe run")
    args = ap.parse_args()
    max_s = args.minutes * 60

    print(f"video={args.video} sample_fps={args.sample_fps} "
          f"det={args.det_w}x{args.det_h} window={args.minutes}min\n")

    results = {}
    for label, seek_gap in (("seek (old, gap>5)", 5),
                            ("grab (new, gap>60)", 60)):
        n, dt = bench_opencv(args.video, args.sample_fps, args.det_w,
                             args.det_h, seek_gap, max_s)
        results[label] = (n, dt)
        print(f"{label:24s}  {n:5d} frames in {dt:7.2f}s "
              f"({n / max(dt, 1e-9):6.1f} fps)")

    try:
        n, dt = bench_ffmpeg_pipe(args.video, args.sample_fps, args.det_w,
                                  args.det_h, max_s, args.hwaccel)
        label = "ffmpeg pipe" + (" +nvdec" if args.hwaccel else "")
        results[label] = (n, dt)
        print(f"{label:24s}  {n:5d} frames in {dt:7.2f}s "
              f"({n / max(dt, 1e-9):6.1f} fps)")
    except FileNotFoundError:
        print("ffmpeg pipe             skipped (ffmpeg not on PATH)")

    best = min(results.items(), key=lambda kv: kv[1][1])
    print(f"\nfastest: {best[0]} ({best[1][1]:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
