#requires -Version 5
# ── Run the ClipAI vision (face-detection) sidecar FROM SOURCE ─────────────
# For when the packaged binary / GitHub-release asset isn't available (e.g.
# CI is disabled). One-time: creates a local venv and installs the CUDA torch
# + YOLO-World dependencies. Then it serves /v1/vision/detect on
# 127.0.0.1:11511 — the ClipAI GPU Companion detects that and ClipAI offloads
# its heaviest analysis stage (face/subject detection) onto THIS GPU.
#
# Usage (on the GPU machine that runs the Companion):
#   powershell -ExecutionPolicy Bypass -File run.ps1
# Leave the window open while ClipAI is analysing. First launch downloads
# torch + the YOLO-World weight (~a few hundred MB); later launches are fast.
#
# Requires Python 3.11 on PATH (https://www.python.org/downloads/).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$venv = Join-Path $here ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv $venv
}
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "Installing dependencies (first run only)..."
& $py -m pip install --upgrade pip
# CUDA torch FIRST from the PyTorch CUDA index so YOLO-World runs on the GPU —
# the default PyPI torch wheel is CPU-only and would be no faster than the
# ClipAI server. cu121 matches recent NVIDIA drivers; adjust if yours differ.
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
& $py -m pip install "fastapi==0.111.0" "uvicorn==0.30.0" "ultralytics>=8.2,<8.4" opencv-python-headless numpy

$env:VISION_PORT = "11511"
Write-Host ""
Write-Host "Starting vision sidecar on http://127.0.0.1:11511  (Ctrl+C to stop)"
Write-Host "Keep this window open — the Companion will show 'Vision offload running'."
& $py (Join-Path $here "server.py")
