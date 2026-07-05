# Remote GPU sharing

ClipAI's biggest bottleneck on a small card (a 4 GB GTX 1650) is not raw
speed — it is VRAM forcing tiny models and serialized stage execution
(Whisper must be unloaded before the vision model fits). Pointing the AI
stages at a bigger GPU on your LAN removes both constraints at once and
frees the local card for what it is genuinely good at: FFmpeg NVENC/NVDEC
encode and decode, which always stay on the ClipAI server (shipping raw
video across the LAN for encode is a net loss).

Two independent features make this work, usable together or separately:

* **Multi-host Ollama registry** — save several Ollama hosts, drag-and-drop
  to reorder them; the top host is the primary and the rest are ordered
  fallbacks with automatic failover (connection errors put a host in a
  ~30 s cooldown and the request retries on the next one; a busy host's
  `503 + Retry-After` is honored once). If every host fails, the normal
  cloud fallback chain (`AI_FALLBACK_CHAIN`) takes over exactly as before.
* **Remote Whisper** — send transcription to any OpenAI-compatible
  `/v1/audio/transcriptions` server. Only the already-extracted 16 kHz WAV
  is uploaded, never the source video. If the server drops mid-job, the
  local Whisper ladder takes over — a job never fails because the remote
  end vanished.

With neither configured, ClipAI behaves exactly as before.

---

## Option A — one-click: the GPU Companion

1. **Download** the Companion from *Settings → AI Providers → GPU
   Companion* (Windows `.exe` or macOS `.dmg`, served by your own ClipAI
   container). Where the installer comes from, in order:
   * the runtime cache (`/config/companion-cache`, filled by "Check for
     updates"),
   * installers baked into the Docker image — a published `companion-v*`
     GitHub Release when one exists, otherwise a **Windows installer
     cross-built from source during the image build** (Ollama sharing
     fully works; the Whisper sidecar ships with official releases, so
     transcription stays on the server until then),
   * a labeled redirect to GitHub Releases.
2. **Install and open it.** The setup wizard detects or installs Ollama
   (winget/brew — from ollama.com, never bundled), offers model pulls
   sized to your GPU, and shows the access token.
3. **Pair**: paste your ClipAI URL (e.g. `http://tower.local:1353`) and
   your ClipAI API key. ClipAI registers the desktop as its **primary**
   Ollama host and points remote Whisper at it. Done — the next analysis
   uses the desktop GPU, and the job appears live in the Companion's
   activity feed.

The Companion's dashboard has a "GPU memory ClipAI may use" slider (a
soft budget): it picks the Whisper tier and restarts Ollama with
`OLLAMA_GPU_OVERHEAD = total − budget` so your games keep their VRAM.
See `companion/README.md` for details and troubleshooting.

## Option B — manual: plain Ollama + any Whisper server

No Companion needed. On the desktop:

* Install [Ollama](https://ollama.com/download) and let it listen on the
  LAN (`OLLAMA_HOST=0.0.0.0`), or keep it local and front it with any
  reverse proxy you trust.
* Run any OpenAI-compatible Whisper server —
  [speaches](https://github.com/speaches-ai/speaches),
  [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice),
  or whisper.cpp's server started with
  `--inference-path /v1/audio/transcriptions`.

Then either use the Settings UI (*Ollama Hosts* card → Add host; *Remote
Whisper* card under Transcription), or configure by environment:

```bash
# Ordered host registry — index 0 is the primary. Tokens are optional
# (the Companion proxy requires one; plain Ollama has none).
OLLAMA_HOSTS='[
  {"id":"desktop","name":"Desktop 4070","url":"http://192.168.1.50:11434","token":"","enabled":true},
  {"id":"sidecar","name":"Unraid sidecar","url":"http://ollama:11434","token":"","enabled":true}
]'

# Remote Whisper (OpenAI-compatible). Blank model = auto ladder
# (large-v3-turbo for English/auto, large-v3 for pinned non-English).
WHISPER_REMOTE_URL=http://192.168.1.50:9000
WHISPER_REMOTE_API_KEY=
WHISPER_REMOTE_MODEL=
```

Host URLs may include a base path (the Companion exposes Ollama at
`http://<desktop>:11500/ollama`).

## Recommended models by VRAM tier

| VRAM | Primary AI (vision) | Editorial AI | Translation | Whisper |
| --- | --- | --- | --- | --- |
| 4 GB (GTX 1650) | `moondream:1.8b` | `qwen2.5:3b-instruct` | `qwen3:4b-instruct-2507-q4_K_M` (partial offload) | `small` / `large-v3-turbo` int8 |
| 8 GB | `llava:7b` | `qwen2.5:7b-instruct` | `qwen2.5:7b-instruct` | `large-v3-turbo` int8 |
| 12 GB (RTX 4070) | `llava:13b` | `qwen2.5:14b` | `qwen2.5:14b` | `large-v3-turbo` fp16 |
| 16 GB+ | `llava:13b` / `qwen2.5-vl:7b` | `qwen2.5:14b` | `qwen2.5:14b` | `large-v3` fp16 |

If a host lacks the configured model, ClipAI substitutes the best model
that host does have (vision: `llava:13b → llava:7b → qwen2.5-vl:7b →
moondream:1.8b`; text: `qwen2.5:14b → qwen2.5:7b-instruct →
qwen2.5:3b-instruct`) and logs the swap rather than failing the job.

## What you should see

* *Settings → AI Providers* shows every host with a live status dot; the
  job's provider readout names the host that served each stage
  (`qwen2.5:14b via ollama @ Desktop 4070`).
* With remote Whisper healthy, the transcription stage is tagged
  `remote`, the model auto-upgrades to `large-v3-turbo`, and the log
  shows "Whisper VRAM release skipped — transcription ran on the remote
  server".
* Kill the primary mid-job → the request retries on the next host and
  the job completes; drag another host to the top → the next job uses it
  first, no restart needed.

## Troubleshooting

* **Host shows offline** — firewall on the desktop (allow inbound TCP
  11500 for the Companion, 11434 for plain Ollama, on Private networks);
  or the Companion's sharing is paused (tray menu).
* **Desktop asleep** — set the desktop to never sleep while sharing, or
  enable Wake-on-LAN; a sleeping host just fails over to the next one.
* **401/403 from a host** — bearer token mismatch. Re-pair the Companion
  or re-enter the token in the host's Edit dialog.
* **`cudnn_ops64_9.dll` errors on the desktop** — NVIDIA driver too old
  for the CUDA 12 / cuDNN 9 stack; update to R550+.
* **Remote transcription never used** — check *Settings → Transcription →
  Remote Whisper → Test*; the health probe must pass at job start, and
  `WHISPER_REMOTE_URL` must be reachable **from the ClipAI container**
  (use the desktop's LAN IP, not `localhost`).
* **Everything falls back to cloud/local** — check the per-host reasons
  in the backend log: every skipped host logs why (offline, cooldown,
  missing model).
