#!/usr/bin/env python3
"""Client-export ↔ server-export frame parity harness (audit Phase 1.6).

Renders of the SAME timeline through the two export paths (browser canvas
via ExportEngine, server FFmpeg via clip_exporter) are compared frame by
frame:

  * SSIM per sampled frame (global structural similarity)
  * subtitle bounding-box position/size deltas (bright-pixel detection in
    the caption band) so font-metric drift between canvas and libass is
    measured, not eyeballed

Usage:
    python scripts/parity_harness.py client.mp4 server.mp4 \
        --timestamps 1.0 2.5 5.0 [--ssim-threshold 0.90] \
        [--caption-band 0.70 1.0] [--json report.json]

Requires: ffmpeg on PATH, numpy. Designed to run on the deployment host
(the Unraid box) or any dev machine — CI here only checks the feature
matrix; pixel checks need real renders.

Exit code 0 = all sampled frames pass the SSIM threshold; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def extract_frame(video: Path, ts: float, out_png: Path) -> bool:
    """Extract a single frame at ``ts`` seconds as PNG via ffmpeg."""
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{ts:.3f}", "-i", str(video),
        "-frames:v", "1", "-y", str(out_png),
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and out_png.exists()


def load_png_gray(path: Path) -> np.ndarray:
    """Load a PNG as float grayscale without cv2/PIL (ffmpeg → rawvideo)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    w, h = (int(x) for x in probe.stdout.strip().split(","))
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True)
    arr = np.frombuffer(raw.stdout, dtype=np.uint8)[: w * h]
    return arr.reshape(h, w).astype(np.float64)


def _uniform_filter(img: np.ndarray, size: int = 8) -> np.ndarray:
    """Box filter via cumulative sums (no scipy dependency)."""
    pad = size // 2
    padded = np.pad(img, pad, mode="reflect")
    cs = padded.cumsum(0).cumsum(1)
    cs = np.pad(cs, ((1, 0), (1, 0)))
    h, w = img.shape
    tot = (cs[size:size + h, size:size + w]
           - cs[:h, size:size + w]
           - cs[size:size + h, :w]
           + cs[:h, :w])
    return tot / (size * size)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM with 8x8 uniform windows (Wang et al. simplified)."""
    if a.shape != b.shape:
        # Resize b to a via nearest-neighbor sampling
        ys = (np.arange(a.shape[0]) * b.shape[0] / a.shape[0]).astype(int)
        xs = (np.arange(a.shape[1]) * b.shape[1] / a.shape[1]).astype(int)
        b = b[ys][:, xs]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = _uniform_filter(a), _uniform_filter(b)
    var_a = _uniform_filter(a * a) - mu_a ** 2
    var_b = _uniform_filter(b * b) - mu_b ** 2
    cov = _uniform_filter(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def subtitle_bbox(gray: np.ndarray, band: tuple[float, float]) -> dict | None:
    """Bright-pixel bounding box inside the caption band (normalized).

    Subtitles are high-luminance text over darker video; a 200+ threshold
    inside the configured band isolates them well enough to measure
    position/size drift between renderers.
    """
    h, w = gray.shape
    y0, y1 = int(h * band[0]), int(h * band[1])
    region = gray[y0:y1]
    mask = region >= 200
    if mask.sum() < 20:  # no visible caption
        return None
    ys, xs = np.nonzero(mask)
    return {
        "x": float(xs.min() / w), "y": float((y0 + ys.min()) / h),
        "w": float((xs.max() - xs.min()) / w),
        "h": float((ys.max() - ys.min()) / h),
    }


def compare(client: Path, server: Path, timestamps: list[float],
            threshold: float, band: tuple[float, float]) -> dict:
    results = []
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for ts in timestamps:
            fa, fb = tdir / f"c_{ts}.png", tdir / f"s_{ts}.png"
            if not (extract_frame(client, ts, fa) and extract_frame(server, ts, fb)):
                results.append({"ts": ts, "error": "frame extraction failed"})
                continue
            ga, gb = load_png_gray(fa), load_png_gray(fb)
            entry = {"ts": ts, "ssim": round(ssim(ga, gb), 4)}
            ba, bb = subtitle_bbox(ga, band), subtitle_bbox(gb, band)
            if ba and bb:
                entry["subtitle_delta"] = {
                    "dx": round(abs(ba["x"] - bb["x"]), 4),
                    "dy": round(abs(ba["y"] - bb["y"]), 4),
                    "dw": round(abs(ba["w"] - bb["w"]), 4),
                    "dh": round(abs(ba["h"] - bb["h"]), 4),
                }
            elif ba or bb:
                entry["subtitle_delta"] = "caption visible in only one render"
            results.append(entry)
    passed = all("ssim" in r and r["ssim"] >= threshold for r in results)
    return {"passed": passed, "threshold": threshold, "frames": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("client", type=Path, help="client (canvas) export video")
    ap.add_argument("server", type=Path, help="server (FFmpeg) export video")
    ap.add_argument("--timestamps", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--ssim-threshold", type=float, default=0.90)
    ap.add_argument("--caption-band", type=float, nargs=2, default=[0.70, 1.0],
                    metavar=("Y0", "Y1"),
                    help="normalized vertical band to search for subtitles")
    ap.add_argument("--json", type=Path, help="write full report to this path")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg/ffprobe not found on PATH", file=sys.stderr)
        return 2

    report = compare(args.client, args.server, args.timestamps,
                     args.ssim_threshold, tuple(args.caption_band))
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
