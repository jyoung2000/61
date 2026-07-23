# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Windows vision sidecar (face-detection offload).
#
# Build (on Windows x64, Python 3.11, on a machine with the CUDA-enabled
# torch wheel installed):
#   pip install -r requirements.txt
#   pyinstaller vision-server.spec
# Output: dist/vision-server/vision-server.exe (one-dir build — one-file
# would unpack the multi-hundred-MB torch/CUDA DLLs to %TEMP% on every
# start). CI zips dist/vision-server/* into vision-server-x64.zip and
# attaches it to the companion release; the app fetches + extracts it into
# app-data/vision-bin on demand (Settings → GPU Companion → Download).
#
# This is a heavy build: ultralytics pulls in torch + torchvision (the CUDA
# runtime DLLs ride along inside the torch wheel, so the frozen exe runs
# YOLO-World on the GPU without a separate CUDA toolkit install), plus
# OpenCV for JPEG decode. Missing any of these makes the frozen exe die at
# import time on a clean machine.

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules

# torch/torchvision ship the CUDA + cuDNN DLLs; cv2 decodes the posted
# frame; ultralytics carries the YOLO(-World) model plumbing.
binaries = (
    collect_dynamic_libs("torch")
    + collect_dynamic_libs("torchvision")
    + collect_dynamic_libs("cv2")
)
# ultralytics + torch ship yaml configs / default weights metadata that the
# loader reads at runtime; collect them so a frozen exe finds them.
datas = (
    collect_data_files("ultralytics")
    + collect_data_files("torch")
    + collect_data_files("torchvision")
)

# ultralytics lazily imports many submodules by string; PyInstaller's static
# analysis misses them, so pull the whole package in.
hiddenimports = (
    collect_submodules("ultralytics")
    + [
        "torch",
        "torchvision",
        "cv2",
        "numpy",
        "PIL",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
    ]
)

a = Analysis(
    ["server.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tensorflow", "matplotlib", "faster_whisper", "ctranslate2"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="vision-server",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="vision-server",
)
