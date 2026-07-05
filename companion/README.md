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

The app lives in the tray/menu bar. Closing the window keeps it running;
the tray menu has **Open**, **Pause sharing**, and **Quit**. "Launch at
login" is a checkbox on the dashboard.

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
