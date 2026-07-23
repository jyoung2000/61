#!/usr/bin/env bash
# ── Run the ClipAI vision (face-detection) sidecar FROM SOURCE ─────────────
# Linux/macOS companion machines. Same idea as run.ps1: create a venv, install
# torch + YOLO-World, serve /v1/vision/detect on 127.0.0.1:11511 so the ClipAI
# GPU Companion offloads face detection onto THIS GPU. Leave it running while
# ClipAI analyses. Requires Python 3.11 on PATH.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if [ ! -d .venv ]; then
  echo "Creating virtual environment (.venv)..."
  python3 -m venv .venv
fi
py=".venv/bin/python"

echo "Installing dependencies (first run only)..."
"$py" -m pip install --upgrade pip
# On Linux the default PyPI torch wheel already ships CUDA; on macOS it's CPU/MPS.
"$py" -m pip install torch torchvision
"$py" -m pip install "fastapi==0.111.0" "uvicorn==0.30.0" "ultralytics>=8.2,<8.4" opencv-python-headless numpy

export VISION_PORT=11511
echo
echo "Starting vision sidecar on http://127.0.0.1:11511  (Ctrl+C to stop)"
echo "Keep this running — the Companion will show 'Vision offload running'."
exec "$py" server.py
