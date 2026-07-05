# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Windows whisper sidecar.
#
# Build (on Windows x64, Python 3.11):
#   pip install -r requirements.txt
#   pyinstaller whisper-server.spec
# Output: dist/whisper-server/whisper-server.exe (one-dir build — one-file
# would unpack the multi-hundred-MB CUDA DLLs to %TEMP% on every start).
# CI copies dist/whisper-server/* into companion/src-tauri/sidecar/
# before `tauri build` so it ships inside the installer as a resource.

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# ctranslate2 ships the CUDA/cuDNN DLLs; av (PyAV) decodes the uploaded
# audio; onnxruntime runs faster-whisper's Silero VAD. Missing any of
# these makes the frozen exe die at import time on a clean machine.
binaries = (
    collect_dynamic_libs("ctranslate2")
    + collect_dynamic_libs("av")
    + collect_dynamic_libs("onnxruntime")
)
datas = (
    collect_data_files("faster_whisper")
    + collect_data_files("onnxruntime")
)

a = Analysis(
    ["server.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "faster_whisper",
        "ctranslate2",
        "av",
        "onnxruntime",
        "tokenizers",
        "huggingface_hub",
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "tensorflow", "matplotlib", "PIL", "cv2"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="whisper-server",
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
    name="whisper-server",
)
