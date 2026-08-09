# ClipAI GPU Companion

Share a desktop GPU (Windows or macOS) with a ClipAI server over the LAN.
The Companion runs **Ollama** (vision/text models) and a **Whisper**
transcription sidecar on your desktop's GPU and exposes both through **one
authenticated port** (default `11500`). ClipAI — typically running on an
Unraid box with a small GPU — sends prompts, frames, and extracted audio
here; **source video never leaves the server**.

```
ClipAI (Unraid, GTX 1650)                Desktop (RTX 4070 / Apple Silicon)
┌──────────────────────┐    LAN     ┌───────────────────────────────────┐
│ pipeline / FFmpeg    │ ─────────► │ Companion proxy :11500 (bearer)   │
│ NVENC stays local    │            │  ├─ /ollama/*  → Ollama (localhost)│
│                      │ ◄───────── │  ├─ /v1/audio/transcriptions      │
│                      │            │  │      → faster-whisper / whisper.cpp
└──────────────────────┘            │  └─ /v1/health                    │
                                    └───────────────────────────────────┘
```

## Install

Grab the installer from **ClipAI → Settings → AI Providers → GPU Companion**
(served by your own ClipAI container), or from the
[GitHub Releases](../../../releases) page (`companion-v*` tags):

* **Windows**: `ClipAI GPU Companion_*.exe` (wizard installer; an `.msi` is
  also published). No command line needed.
* **macOS**: `ClipAI GPU Companion_*.dmg` — drag to Applications.
  Builds are **unsigned**: on first launch right-click the app → **Open**,
  or run `xattr -dr com.apple.quarantine "/Applications/ClipAI GPU Companion.app"`.

## First run

The setup wizard walks through everything:

1. **Ollama** — detected if already installed; otherwise one-click install
   (`winget install Ollama.Ollama` on Windows, `brew install ollama` on
   macOS) with a fallback button to [ollama.com/download](https://ollama.com/download).
   The Companion runs Ollama bound to `127.0.0.1` only — the proxy is the
   sole LAN surface.
2. **Model pulls** — recommendations sized to your GPU memory.
3. **Access token** — generated on first run; every request must carry it.
4. **Pairing** — paste your ClipAI URL + API key (ClipAI Settings → API).
   ClipAI then adds this machine as its **primary** Ollama host and points
   remote Whisper at it. Manual setup works too: add
   `http://<this-machine>:11500/ollama` (+ token) in ClipAI's *Ollama Hosts*
   card and `http://<this-machine>:11500` under *Remote Whisper*.

## Running in the background (no terminal, no window)

The Companion is a background GPU service, not a program you keep open:

* **No terminal, ever.** It is a normal desktop app — double-click it, or let
  it start at login. The release build has no console, and every child process
  it launches (Ollama, whisper.cpp, `nvidia-smi`, `winget`) is spawned with
  `CREATE_NO_WINDOW` so nothing flashes a console window either.
* **Closing the window keeps it sharing.** The X button hides the dashboard to
  the tray/menu bar; the proxy, Ollama and the whisper sidecar keep serving.
  Losing the window some other way — a WebView crash, "End task" on the window,
  Cmd-Q — is refused the same way, so a mid-job Companion can't vanish and take
  a running ClipAI job down with it. **Quit** in the tray menu is the only way
  to stop sharing (it also stops the managed Ollama/whisper children).
* **Starts at login, straight to the tray.** Registration is automatic and
  passes `--hidden`, so after a reboot the GPU is back online with no window on
  screen and nobody at the desk. Launching it yourself always opens the
  dashboard.

The tray menu has **Open Dashboard**, **Pause sharing**, and **Quit**.

## VRAM budget

The dashboard slider ("GPU memory ClipAI may use") is a **soft budget**:

* Whisper tier — ≥6 GB → `large-v3-turbo` (fp16); ≥3 GB → `medium` (int8);
  below → `small` (int8). macOS uses the matching whisper.cpp quants.
* Ollama — restarted with `OLLAMA_GPU_OVERHEAD = total − budget`, plus
  `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`, and a configurable
  `OLLAMA_KEEP_ALIVE` (default 10 m) so models unload when idle.

One transcription runs at a time; a second concurrent request gets an
honest `503 + Retry-After`, which ClipAI treats as "busy — try the next
host".

## Firewall

Allow inbound **TCP 11500** on **Private** networks. The Windows installer
does not add a rule automatically; Windows will prompt on first use —
accept for Private networks only.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| ClipAI shows the host offline | Firewall blocking TCP 11500, or the Companion is paused (tray menu). |
| `401/403` in ClipAI logs | Token mismatch — re-pair, or copy the token from the dashboard into ClipAI's host entry. |
| `cudnn_ops64_9.dll not found` / sidecar dies instantly | NVIDIA driver too old for CUDA 12 — update to R550+ from nvidia.com. |
| Port 11500 already in use | Another service owns it; stop it or change the port in `companion.json` (app config dir) and re-pair. |
| Ollama not detected after install | Log out/in (PATH refresh) or click **Re-detect**; verify `ollama --version` in a terminal. |
| Transcription slow on first request | The sidecar downloads the model on first use — subsequent runs are warm. |
| macOS "app is damaged / can't be opened" | Unsigned build quarantine — right-click → Open, or the `xattr` command above. |

## Building from source

```bash
cd companion
npm install
npx tauri icon app-icon.png   # generate platform icon formats (once)
npm run tauri dev             # dev build
npm run tauri build           # installer for the current OS
```

The Windows whisper sidecar is built separately (see
`sidecars/whisper-server/whisper-server.spec`) and dropped into
`src-tauri/sidecar/` before `tauri build`; CI
(`.github/workflows/companion-release.yml`) automates all of it on tag
`companion-v*`. No model weights or Ollama binaries are ever committed to
this repository.

### Cross-building the Windows installer from Linux (MSVC, via cargo-xwin)

The ClipAI Docker image does this automatically (its `companion-builder`
stage, `--build-arg COMPANION_BUILD_FROM_SOURCE=1`) so the Settings
download button works before any GitHub release exists. Manually:

```bash
sudo apt install nsis clang llvm lld pkg-config libgtk-3-dev libayatana-appindicator3-dev
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin --locked
cd companion && npm install && npx tauri icon app-icon.png
XWIN_ACCEPT_LICENSE=1 npx tauri build --runner cargo-xwin \
  --target x86_64-pc-windows-msvc --bundles nsis
# → src-tauri/target/x86_64-pc-windows-msvc/release/bundle/nsis/*-setup.exe
```

**Use MSVC, not MinGW.** Tauri/WebView2 only supports the MSVC toolchain
on Windows; a `x86_64-pc-windows-gnu` (MinGW) build compiles but opens to
a blank white webview and exits. `cargo-xwin` provides the MSVC target on
Linux (it fetches the MSVC CRT + Windows SDK from Microsoft at build
time — `XWIN_ACCEPT_LICENSE=1` accepts the EULA). The `libgtk-3-dev` /
`libayatana-appindicator3-dev` packages satisfy a host-side probe
tauri-cli runs for the tray library, regardless of the Windows target.

This fallback build cannot include the PyInstaller whisper sidecar
(PyInstaller doesn't cross-compile). The Companion detects that,
reports `backends.whisper=false`, and pairing leaves ClipAI's
transcription local — GPU-shared Ollama works fully either way.

### Building the macOS installer (natively, on a Mac)

macOS apps can't be built from Linux at all. On a Mac, run:

```bash
cd companion && ./scripts/build-macos.sh
# → src-tauri/target/universal-apple-darwin/release/bundle/dmg/*.dmg
```

It builds a universal (Apple Silicon + Intel) `.dmg`, optionally with the
whisper.cpp Metal sidecar (`NO_WHISPER=1` to skip). To make it downloadable
from ClipAI, copy the `.dmg` into the server's `data/companion-cache/`
folder — the Settings card serves whatever installer files are present, so
the macOS button lights up with no manifest step.
