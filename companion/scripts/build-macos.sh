#!/usr/bin/env bash
# Build the ClipAI GPU Companion macOS installer (.dmg) natively on a Mac.
#
# macOS apps cannot be cross-compiled from Linux (the .app bundle, framework
# linking, and .dmg creation all require macOS), so this is the supported way
# to produce the Mac installer without GitHub Actions. Run it on your Mac:
#
#   cd companion && ./scripts/build-macos.sh
#
# Output: a universal (Apple Silicon + Intel) .dmg under
#   src-tauri/target/universal-apple-darwin/release/bundle/dmg/
# and a manifest.json next to it. Install it directly, or copy both files
# into your ClipAI server's cache dir so the Settings download button lights
# up for macOS too (see the printed instructions at the end).
#
# Prereqs (installed for you if missing, except Xcode CLT):
#   * Xcode Command Line Tools:  xcode-select --install
#   * Rust (rustup):             https://rustup.rs
#   * Node.js 18+:               https://nodejs.org
set -euo pipefail

cd "$(dirname "$0")/.."   # → companion/
echo "==> Building ClipAI GPU Companion (.dmg) in $(pwd)"

if [ "$(uname -s)" != "Darwin" ]; then
  echo "ERROR: this must run on macOS. On Linux/Windows use the Docker/native path instead." >&2
  exit 1
fi

command -v cargo >/dev/null 2>&1 || { echo "ERROR: Rust not found — install from https://rustup.rs" >&2; exit 1; }
command -v npm   >/dev/null 2>&1 || { echo "ERROR: Node.js/npm not found — install from https://nodejs.org" >&2; exit 1; }

# Universal binary needs both arch targets.
echo "==> Ensuring Rust targets (aarch64 + x86_64 apple-darwin)"
rustup target add aarch64-apple-darwin x86_64-apple-darwin >/dev/null

# ── Optional: build the whisper.cpp sidecar (Metal) so remote transcription
# works on the Mac too. Skip with NO_WHISPER=1. Fail-soft: the app still
# builds (and shares Ollama) if this can't complete.
if [ "${NO_WHISPER:-0}" != "1" ]; then
  echo "==> Building whisper.cpp server (Metal) — set NO_WHISPER=1 to skip"
  (
    set -e
    if ! command -v cmake >/dev/null 2>&1; then
      echo "   cmake not found — skipping sidecar (install with: brew install cmake)"; exit 0
    fi
    [ -d whisper.cpp ] || git clone --depth 1 --branch v1.7.4 https://github.com/ggerganov/whisper.cpp
    cmake -S whisper.cpp -B whisper.cpp/build \
      -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
      -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_TESTS=OFF
    cmake --build whisper.cpp/build --config Release -j "$(sysctl -n hw.ncpu)"
    mkdir -p src-tauri/sidecar
    cp whisper.cpp/build/bin/whisper-server src-tauri/sidecar/whisper-server 2>/dev/null \
      || cp whisper.cpp/build/bin/server src-tauri/sidecar/whisper-server
    chmod +x src-tauri/sidecar/whisper-server
    echo "   whisper sidecar built."
  ) || echo "   whisper sidecar build failed — continuing without it (Ollama sharing still works)."
fi
# The bundler needs at least one file under src-tauri/sidecar.
mkdir -p src-tauri/sidecar
[ -e src-tauri/sidecar/README.md ] || echo "sidecar staging" > src-tauri/sidecar/README.md

echo "==> Installing frontend dependencies"
npm install --no-audit --no-fund

echo "==> Generating icons"
npx tauri icon app-icon.png >/dev/null 2>&1 || true

echo "==> Building the universal .dmg (this takes a few minutes)"
npx tauri build --target universal-apple-darwin --bundles dmg

DMG=$(ls src-tauri/target/universal-apple-darwin/release/bundle/dmg/*.dmg 2>/dev/null | head -1 || true)
if [ -z "$DMG" ]; then
  echo "ERROR: no .dmg was produced — check the tauri build output above." >&2
  exit 1
fi

OUT_DIR=$(dirname "$DMG")
python3 scripts/make_local_manifest.py "$OUT_DIR" || true

echo
echo "======================================================================"
echo " SUCCESS — macOS installer built:"
echo "   $DMG"
echo
echo " It is UNSIGNED. On first launch: right-click the app → Open, or run"
echo '   xattr -dr com.apple.quarantine "/Applications/ClipAI GPU Companion.app"'
echo
echo " To install now: open the .dmg and drag the app to Applications."
echo
echo " To make it downloadable from ClipAI → Settings → GPU Companion,"
echo " copy the .dmg into your ClipAI server's cache dir (file presence is"
echo " all that's needed — the macOS button lights up automatically):"
echo "   scp \"$DMG\" \\"
echo "     <user>@<clipai-host>:/mnt/user/appdata/clipai/data/companion-cache/"
echo "======================================================================"
