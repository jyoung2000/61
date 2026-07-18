# Speed audit: 24-min video, 46-min pipeline → 15–20 min target

Five parallel deep audits (transcription/translation, perception/summary,
reframer subsystem, clip export/Companion, cross-cutting orchestration) over
the full backend. Every finding below was verified against the code at the
cited `file:line`. Findings are grouped by risk, not by subsystem: Tier 1 is
config-only with identical output, Tier 2 is small code changes with
byte-identical output (scheduling + redundancy removal), Tier 3 is the small
set of explicit near-neutral trades. Savings ranges overlap — they do not
simply sum; expected end states are at the bottom.

## Where the 46 minutes goes (24-min video, 4 GB GTX 1650, local everything)

| Segment | Est. wall | Why |
|---|---|---|
| Metadata + Ollama warmup dance | ~1.5 min | warmup/unload blocks critical path (`pipeline.py:4552-4599`) |
| Perception face/YOLO loop | ~7–9 min | 1,800 samples — same budget as a 128-min video (`pipeline_helpers.py:250-274`) |
| Local Whisper (after faces) | ~7–10 min | serialized behind the face loop on local GPU (`reframer_perceiver.py:940-968`) |
| Transcript chain + repair/bridge | ~3 min | already overlapped internally |
| Translation + polish passes | ~8–12 min | 3 full-transcript LLM passes + double recovery loop |
| Hidden 2nd Whisper pass (timing ref) | ~8–10 min | full re-decode, text discarded (`pipeline.py:549-616`) |
| Summary (serial on "local" editorial) | ~2–4 min | serialized ahead of clips (`pipeline.py:6253-6260`) |
| Clip candidate export | ~6–20 min | full libx264 re-encode per clip; stream-copy path exists but is OFF (`config.py:789`) |
| Auto-SEO followups | ~2–6 min | blocks COMPLETE (`pipeline.py:6497-6499`) |

The pipeline already contains most of the overlap machinery (extraction ∥
perceive, remote-Whisper ∥ faces, diarization ∥ faces, repair+bridge ∥
transcript chain). What remains is: three gates that don't recognize a
high-VRAM remote Companion as lifting the 4 GB constraint, one false
dependency (translation → clip detection), two redundant full passes, and
two dormant fast paths that ship disabled.

---

## Tier 1 — config flips, zero quality change

### 1.1 Turn on clip-export stream copy (biggest single win)
- `config.py:789` `CLIP_EXPORT_STREAM_COPY: bool = False`; export at
  `reframer_clipper.py:3940-4040`.
- Candidate clips written during `clip_extraction` are plain time-cuts —
  no reframe, no subtitles (the docstring says so) — yet each one takes a
  full `libx264 -preset veryfast -crf 21` re-encode because both the
  stream-copy flag and `GPU_ACCELERATION_ENABLED` (`config.py:883`) default
  off. These MP4s are review artifacts only: the final user export re-reads
  the original source (`routers/clips.py:379-381`), so stream copy is
  bit-identical-to-source (it *removes* a generation loss). Sole caveat:
  `-c copy` snaps preview starts to the previous keyframe (~1–2 s early),
  a cosmetic on previews only — `config.py:784-788` documents exactly this.
- If frame-accurate previews are required instead, enable
  `GPU_ACCELERATION_ENABLED=true` so `_export_clip` uses NVENC (~5–10×).
- Also: `CLIP_EXPORT_STREAM_COPY` and `CLIP_EXPORT_CONCURRENCY` are absent
  from `.env.example` — add them.
- **Saves ~6–20 min** depending on clip count.

### 1.2 Run Whisper on the Companion
- `.env.example:51` `WHISPER_REMOTE_URL`. The faces ∥ Whisper overlap is
  already implemented and byte-identical (`reframer_perceiver.py:199-249`)
  but only engages for remote Whisper; local Whisper strictly follows the
  face loop (`:940-968`). Attaching the Companion removes the entire local
  Whisper leg from the critical path, at same-or-better model quality
  (large-v3 vs the local `WHISPER_MODEL="small"` default, `config.py:229`).
- **Saves ~7–10 min.**

### 1.3 Stop model-eviction thrash between stages
- Companion idle reaper: `companion/src-tauri/src/sidecar.rs:586-651` frees
  the whole GPU after 45 s of AI-quiet time; a single slow clip encode
  (up to 180 s allowed, `reframer_clipper.py:4015`) exceeds it, so the
  post-clip SEO/judge calls pay cold reloads. Raise `gpu_idle_free_sec`
  to ≥300 (and/or treat "job fresh" as heartbeat-within-5-min at
  `sidecar.rs:625`); optionally emit a heartbeat per started clip.
- Ollama text calls send no `keep_alive` (`providers/ollama_provider.py:1604-1633`)
  so the 5-min server default evicts the translation model across long gaps
  (vision path already pins with `OLLAMA_VISION_KEEP_ALIVE=-1`, `:1452`).
  Add `"keep_alive": "30m"` (env-configurable) to the text payload.
  Explicit evictions (`keep_alive:0`, `clear_vram`) still govern handoffs.
- Companion managed keep_alive 10m → 30m (`companion/src-tauri/src/ollama.rs:400-433`).
- **Saves ~1–3 min + removes run-to-run variance and a timeout-cascade failure mode.**

### 1.4 Don't block the pipeline start on Ollama warmup for remote hosts
- `pipeline.py:4552-4599`: up to 60 s warmup + 15 s unload + 2 s sleep on
  the critical path before extraction. For a remote primary host,
  fire-and-forget (the unload is already skipped at `:4594`).
- **Saves 30–77 s.**

---

## Tier 2 — small code changes, byte-identical final output

### 2.1 Windowed second-Whisper timing pass instead of a full re-decode
- `pipeline.py:549-616` (`_get_whisper_en_timing_reference`), invoked at
  `:3403-3428` on the LLM-translate path: after translation, Whisper's
  native translate task re-decodes the **entire video** and keeps only
  `.words` — the text is discarded. Because the analyze stage deliberately
  keeps the engine cached (`_whisper_engine_cached`, `:537-546`), the pass
  always runs locally (~8–10 min for a 24-min video).
- Only cues that will actually be split (over-CPS/over-length, typically
  10–20%) need tier-A word alignment; the rest already use tier-B source
  word timestamps. Decode only those cue windows via `clip_timestamps` —
  the mechanism already exists in the redecode pass
  (`reframer_audio.py:2905-2913`).
- **Saves ~6–9 min.** Identical translated text; identical split decisions
  for every cue that gets the tier-A treatment.

### 2.2 Overlap translation+polish with clip detection (false dependency)
- Translation (`_background_post_processing` awaited at `pipeline.py:6154`)
  fully precedes clip detection (`:6274`), but clip detection reads the RAW
  `perception.transcript_segments` (`:6327`) — not the translated track.
  Only `_run_post_clip_followups` (`:6498`) truly needs translation output.
- Start `_background_post_processing` as a task after the plan join
  (`:5947`), run clip detection concurrently, join the task before the
  followups. Gate on a remote/high-VRAM Ollama host (same pattern as the
  Demucs VRAM gate at `:5231-5236`) so 4 GB local rigs keep the proven
  serial order.
- **Saves ~5–8 min on Companion rigs.**

### 2.3 Defer the source polish and the translation post-edit off the critical path
- Source polish: `pipeline.py:5662-5688` → `transcript_polisher.py:1203-1249`
  runs a full batched LLM pass over ALL source cues *before* translation,
  despite the code's own comment at `:5652` about not spending an LLM pass
  punctuating text that's about to be replaced. The translated track does
  not depend on it (the post-edit already repairs meaning against source).
  Defer it to the background tail after the translated track persists, or
  default `TRANSLATION_POLISH_SOURCE_FIRST=false` when translation is
  pending (`config.py:1084`). **Saves 3–8 min.**
- Post-edit (pass 3): `pipeline.py:3271-3281`, budget up to 720 s
  (`SUBTITLE_POLISH_MAX_S`, `config.py:977`), sometimes on an auto-upsized
  model (`transcript_polisher.py:1374-1386`). Keep the pass (it's real
  quality for 4B-translated tracks) but run it in the background after the
  translated track first persists — same final bytes, up to 12 min less
  perceived wall. The `TRANSLATION_SKIP_POSTEDIT_LARGE_MODEL` gate
  (`pipeline.py:3198`) already covers 12B+. **Saves ~5 min typical.**

### 2.4 Deduplicate the untranslated-cue recovery
- `translator.py:985-1008`: translate_via_llm's internal completeness loop
  makes up to 3 passes with recursive halving (worst case 2N−1 calls) over
  stubborn cues that `_llm_cleanup_untranslated` (`pipeline.py:1105-1349`)
  — guaranteed to run on every path (`:784-787`) and strictly more robust —
  retries again anyway. Cap the internal loop to one pass when called from
  `translate_subtitles`. **Saves 2–6 min on hard content, 0 on clean.**

### 2.5 Fix the three gates that don't recognize the Companion
- Summary ∥ clips: `pipeline.py:6254` keys on `_editorial_is_local`
  (`:4151`) — true even when Ollama runs on a 12 GB Companion. Add the
  `remote_vram_gb`/`is_local_gpu_host` check the codebase already uses at
  `:4534`/`:5575` and `config.py:1874-1905`. **Saves 2–4 min.**
- Summary map concurrency: `ai_orchestrator.py:754` hardcodes
  `Semaphore(1 if is_ollama else 3)`, ignoring the remote-VRAM ladder the
  vision path uses (`config.py:1891-1896`). Size 2–3 on ≥7 GB hosts; raise
  cloud 3→6. Chunks are disjoint spans — identical outputs, interleaved
  wall time. **Saves 2–4 min.**
- Editorial VRAM free before clips: `pipeline.py:6268` → `:388` unloads the
  editorial model that Auto-SEO needs minutes later — a needless 60–150 s
  cold reload on remote ≥8 GB hosts. Skip when remote. **Saves 1–2.5 min.**

### 2.6 Local Whisper ∥ face loop on ≥6 GB local cards
- Extend the existing remote-overlap branch (`reframer_perceiver.py:199-249`)
  to local CUDA Whisper when free VRAM ≥ ~6 GB (YOLO-World small is
  ~150 MB, `reframer_face.py:42-48`; turbo ~1.5–2 GB), keeping the
  sequential order on 4 GB cards. Pure scheduling; same calls, same join
  point. **Saves ~3–7 min for local-GPU users without a Companion.**

### 2.7 Move Auto-SEO after COMPLETE
- `pipeline.py:6497-6499` awaits `_run_post_clip_followups` (per-clip,
  per-platform LLM calls) before the COMPLETE save. SEO copy is metadata on
  already-final clips; the late-delivery plumbing already exists
  (WS `clips_refreshed`, `:2645,2665-2680`, manual regenerate path). Persist
  COMPLETE first, run followups as a task that re-persists. Keep the cheap
  caption refresh synchronous if target-language cards should be present at
  COMPLETE. **Saves 2–6 min perceived.**

### 2.8 Batch export concurrency (user-facing path, not the 46-min run)
- `routers/agent.py:538-552` exports strictly serially; the pipeline side
  already established 3 concurrent NVENC sessions as safe
  (`config.py:768-771`). Semaphore + gather. **~2–3× on batch exports.**

### 2.9 Bridge thumbnails from already-extracted frames
- `reframer_bridge.py:394-431,490-505`: ~1 ffmpeg spawn per scene for
  UI-only thumbnails; nearest frame from `frames_dir` is already on disk.
  **Saves 0.5–2 min of overlapped CPU.**

---

## Tier 3 — explicit near-neutral trades (opt-in, flagged honestly)

These are NOT byte-identical; each has a documented, small effect.

- **Perception sample budget** — `REFRAMER_MIN_SAMPLE_FPS` 1.2→0.8 and/or
  `REFRAMER_MAX_SAMPLES` 1800→1000–1200 (`config.py:586-594`). A 24-min
  video currently gets the same 1,800-sample budget as a 128-min video
  (1.25 fps). Precedent: the cap was previously halved 3600→1800 with
  "negligible reframing-accuracy loss," and keyframes pass through multiple
  smoothing layers. **Saves ~7–12 min** — the decisive knob for local-only
  rigs. Quality-neutral alternative worth building: skip the detector union
  when frame-diff vs the previous sample is ~zero (static scene) and carry
  detections forward.
- **`WHISPER_REMOTE_PREFER_ACCURACY=false`** (`config.py:508`): turbo vs
  large-v3; marginal English WER delta, **saves ~3–5 min** of remote
  transcription. Sibling: `WHISPER_PREFER_DISTIL_ENGLISH=true`
  (`config.py:234`) for English-only libraries.
- **`REFRAME_VLM_SPOTCHECK=false`** or async (`config.py:912-913`,
  `reframe_vlm_judge.py:106-165`): 12 serial extract→VLM iterations, up to
  300 s, producing a calibration-only metric never blended into the grade
  (`pipeline.py:6441-6459`). Async-ing it is quality-identical;
  disabling only drops the reported `vlm_framing_pct`. **Saves 1–3 min.**
- **u2netp stride-2 + tiled-YuNet skip after N confirmed-empty samples**
  (`reframer_perceiver.py:713-726`, `reframer_face.py:612-763`): near-
  identical after two smoothing layers; **saves 2–7 min on faceless-heavy
  content** (gameplay/anime), ~0 on talking heads.
- **Vocal separation** (only if you enabled it; default off,
  `config.py:526`): full-track htdemucs is the single largest lever when
  on — gate it on measured speech coverage / music-heavy probes, or use
  two-stem `mdx_extra_q` on the CPU-fallback path. The stem feeds only
  Whisper (`reframer_perceiver.py:951-963`), so the bar is "vocals
  intelligible to ASR." **Saves up to 10–30 min on the serial/CPU path.**
- **Do not touch:** `WHISPER_BEAM_SIZE=5` (measured 5–8% recall cost),
  `WHISPER_REDECODE_ENABLED`, `SUBTITLE_FORCED_ALIGN`,
  `REFRAMER_PROBLEM_REPAIR`, `REFRAMER_YOLO_STRIDE` beyond 2 (violates the
  1.5 s staleness rule at `reframer_perceiver.py:317-322`).

---

## Bugs found along the way (quality, not speed)

- **Summary reduce truncation**: `ai_orchestrator.py:812-813` clips the
  reduce input to `combined[:2500]` for Ollama — with up to 12 chunk
  mini-summaries this silently drops the back half of long videos. Raise
  the cap or stride-sample, mirroring the map's even-stride fix at
  `:763-767`.
- **Duplicate config field**: `TRANSLATION_LLM_CLEANUP_CONCURRENCY` is
  declared twice (`config.py:1374` and `:1403`; the second wins). Also,
  cleanup concurrency hardcodes 3 (`pipeline.py:1089,1268`) instead of the
  `companion_num_parallel`-derived width the main fan-out uses
  (`translator.py:534-554`).

## Verified non-issues (don't re-audit)

Per-clip source re-decode (all export paths use `-ss` before `-i`);
redundant subtitle burn passes (single filter-graph invocation);
GPU preflight evicting Companion models (filtered to local hosts,
`gpu_preflight.py:272-274,358-360`); Companion Ollama env tuning
(`OLLAMA_NUM_PARALLEL`/`FLASH_ATTENTION`/`KV_CACHE_TYPE=q8_0` already set,
`ollama.rs:400-433`); repair pass (bounded, parallel, overlapped);
engine post-smoothers (ms-scale); stage timeouts at
`pipeline.py:1463-1466` (ceilings, not waits); CPU thread caps
(`cpu_threads.py`, `proc_priority.py` — well-tuned).

---

## Expected outcomes

- **Tier 1 only** (config flips, ~an afternoon incl. verification):
  46 → **~24–30 min**. Stream copy + Companion Whisper do the lifting.
- **Tier 1 + Tier 2** (the scheduling/redundancy code changes):
  46 → **~15–20 min** with byte-identical deliverables. The orchestration
  audit's overlapped schedule for a Companion rig lands ≈19–20 min;
  the translation-stage redundancy removals (2.1, 2.3, 2.4) are what pull
  the local-only case into range.
- **Local-only 4 GB rig, no Companion**: Tier 1.1 + 2.1 + 2.3 + 2.4 + 2.6
  plus the Tier 3 sample-budget knob reaches ~18–24 min; the Companion is
  the difference-maker below that.

## Suggested rollout order

1. `CLIP_EXPORT_STREAM_COPY=true`, Companion Whisper URL, keep-alive fixes
   (1.1–1.4) — measure with the existing `_stage_timer` lines.
2. 2.1 (windowed timing pass) and 2.3/2.4 (translation redundancy) — the
   translation stack drops from 5 passes to 2 on the critical path.
3. 2.2 + 2.5 (the Companion-aware gates and the translation ∥ clips
   overlap) — gate every overlap on the existing `remote_vram_gb` signal.
4. 2.7 (SEO after COMPLETE), then the Tier 3 knobs per deployment taste.

Every stage is already instrumented (`_stage_timer`) — before/after
comparison is one grep of the job log.
