# Inflect Studio

A local, Windows‑first desktop app that **clones a voice from a video/audio
clip** and **synthesizes speech in that voice** — with a unique **Inflection
Editor** that lets you highlight any span of text and change *how it is spoken*
(8 emotion sliders, a natural‑language delivery description, speed, and pauses).

Everything runs on your machine. The only network use is the first‑time model
download from Hugging Face.

```
Import MP4 ─▶ isolate + auto‑pick the cleanest speech ─▶ Voice Profile
                                                              │
Type a script ─▶ highlight a phrase ─▶ make it "angry 0.7, slower" ─▶ │
                                                              ▼
        IndexTTS‑2 / Chatterbox / Fish ─▶ crossfade + ‑16 LUFS ─▶ WAV / MP3
```

---

## Features

- **Zero‑shot voice cloning** from MP4/MKV/MOV/MP3/WAV/M4A/FLAC — no training.
  ffmpeg extraction → optional Demucs vocal isolation → silero‑VAD picks the
  best 10–20 s clip → reusable **Voice Profile**.
- **Inflection Editor**: highlight text and set 8 emotion dims, a free‑form
  delivery (e.g. *"whispering, almost crying"*), emotion strength, speed
  (0.5–1.5×) and trailing pauses. Styled spans show as colored underlines;
  unstyled text uses the document default.
- **Three engines, one workflow**
  - **Draft — Chatterbox**: fast auditioning.
  - **Final — IndexTTS‑2**: high‑quality, emotion‑vector / emo‑text / emotion‑
    reference control.
  - **Hybrid — Performance Transfer** *(optional, Phase 6)*: Fish S2 performs the
    line expressively, then IndexTTS‑2 reproduces that performance **in your
    cloned voice**.
- **Per‑segment caching** — editing one highlighted phrase only re‑renders that
  phrase. **Engine‑batched** rendering keeps GPU model swaps to a minimum.
- **Seamless assembly** — 15 ms equal‑power crossfades, pause insertion,
  loudness‑normalized to **‑16 LUFS** with a ‑1 dBTP peak limit.
- **Timeline** waveform with colored segment regions, click‑to‑seek, per‑segment
  re‑render. **Projects** save to `.inflect` JSON; persistent **Voice Library**.

---

## Requirements

- **OS**: Windows 10/11 (first‑class). Linux/macOS work for the app and
  Chatterbox; IndexTTS‑2/Fish are best on NVIDIA CUDA.
- **Python**: 3.10 or 3.11.
- **GPU**: NVIDIA with CUDA, **12 GB VRAM** recommended (built and tuned for an
  RTX 4070). CPU works for Chatterbox only (IndexTTS‑2 on CPU is very slow).
- **ffmpeg** and **ffprobe** on your `PATH` (https://ffmpeg.org/download.html).
  These are **not** pip‑installable.

---

## Install

### Quick start (recommended)

```bat
:: Windows
git clone <your-fork-url> inflect-studio
cd inflect-studio
run.bat
```

```bash
# Linux / macOS
git clone <your-fork-url> inflect-studio
cd inflect-studio
./run.sh
```

`run.bat` / `run.sh` create a virtual environment, install a CUDA build of
PyTorch (Windows/Linux) followed by `requirements.txt`, then launch the app.

### Manual install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate

# 1) Install a CUDA build of PyTorch FIRST (pick the CUDA your driver supports):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2) Then everything else:
pip install -r requirements.txt

# 3) Run:
python -m inflect
```

> IndexTTS‑2 (`indextts`) and, for Phase 6, Fish Speech may need to be installed
> from source if no wheel matches your platform:
> ```bash
> pip install git+https://github.com/index-tts/index-tts.git
> pip install git+https://github.com/fishaudio/fish-speech.git   # optional
> ```

---

## First run & model downloads

Models download automatically the first time you use each engine and are cached
under your data home (see below):

| Engine | Hugging Face | When |
|---|---|---|
| Chatterbox | `ResembleAI/chatterbox` (via `chatterbox-tts`) | first Draft render |
| IndexTTS‑2 | `IndexTeam/IndexTTS-2` | first Final render |
| Fish S2 *(optional)* | `fishaudio/s2-pro` | first Hybrid/Fish render |

The **first** render of a session is slow (weights load into VRAM, and IndexTTS‑2
warms up CUDA kernels). Subsequent renders are fast and cached.

**Data home** (models, voice library, project cache, logs):

- Windows: `%APPDATA%\InflectStudio\inflect-studio`
- Linux/macOS: `~/.inflect-studio`
- Override with the `INFLECT_HOME` environment variable.

---

## Using it

1. **Import a voice** — *Voice Library ▸ ＋ Import from video/audio…*. Pick a
   file, optionally enable **Demucs** isolation (auto‑suggested for noisy/music
   sources), **Analyze**, audition/trim one of the three candidate clips, tick
   *"I have permission to clone this voice."*, name it, **Save**.
2. **Pick the voice** in the toolbar (or the library list).
3. **Type your script** in the center editor.
4. **Inflect a phrase** — select text, set emotion sliders / *Describe delivery*
   / speed / pause in the right **Inspector**, then **Apply to selection**.
   *Set as default* changes the unstyled delivery for the whole document.
5. **Choose a mode** — *Draft* (fast) or *Final* (quality) in the toolbar; or set
   a per‑span engine (including **Hybrid**) in the Inspector.
6. **▶ Synthesize All** — watch progress; **⏹ Cancel** stops between segments.
7. **Timeline** — click to seek, Space to play/pause; right‑click a segment to
   re‑render just it.
8. **Export** — WAV or MP3.

### Hybrid "Performance Transfer"

Set a span's engine to **Hybrid**. Fish renders the line with maximal
expression (its default voice), then IndexTTS‑2 re‑voices that exact performance
in **your** cloned timbre. Use **Audition performance (stage 1)** to hear Fish's
take before committing. A mixed document needs at most three model swaps.

---

## Troubleshooting

- **"ffmpeg not found"** — Install ffmpeg and add it to `PATH`, or set its path
  in *Settings*. Importing voices and MP3 export need it.
- **CUDA out of memory (OOM)** — Close other GPU apps (browsers, games). Use
  shorter segments, switch to **Draft**, keep **fp16** enabled (Settings), and
  avoid running Demucs and a TTS engine back‑to‑back without letting one unload.
  Only one engine is ever resident; the status bar shows live VRAM.
- **First render is very slow** — Expected: weights load + CUDA warm‑up. It is
  fast afterwards, and unchanged segments come from cache.
- **No CUDA / CPU only** — Chatterbox runs on CPU. IndexTTS‑2 will warn and be
  very slow; Fish is impractical on CPU.
- **No sound / wrong device** — Choose an output device in *Settings*.
- **Demucs download is large** — It is optional; leave isolation off for clean
  sources, or comment `demucs` out of `requirements.txt`.
- **Something failed** — The error dialog has **Copy diagnostics** (message +
  traceback + recent log). Logs live at `<data home>/logs/inflect.log`.

---

## Projects, library & cache

- **Projects**: `*.inflect` (versioned JSON — text, spans, default delivery,
  selected voice + engine).
- **Voice Library**: `<data home>/voices/<uuid>/reference.wav` + `meta.json`,
  indexed by `voices/index.json`.
- **Render cache**: `<data home>/project_cache/<hash>.wav` (delete to force a
  clean re‑render).

---

## Development

```bash
pip install pytest numpy soundfile pyloudnorm scipy
pytest                      # 150+ unit tests (model, segmenter, assembly, pipeline)
QT_QPA_PLATFORM=offscreen pytest tests/test_gui_smoke.py   # headless Qt smoke
```

The core (document model, segmenter, project IO, assembly, pipeline
orchestration, engine‑parameter mappings) is engine‑agnostic and fully unit
tested with a fake engine — no GPU required.

---

## Licensing & consent

- You are responsible for having the right to clone any voice you import; the
  import wizard requires you to confirm consent.
- **Fish Speech** weights are research/personal‑use; commercial use needs a
  license from Fish Audio. IndexTTS‑2 and Chatterbox have their own licenses —
  review them before commercial use.
