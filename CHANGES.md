# ClipAI — Every exported clip now ships an SEO ``.txt`` next to the MP4

Exporting/downloading a clip now also gives you a human-readable ``.txt`` with
everything you'd paste into a social upload form: **viral score** (+ the
hook/flow/value/trend breakdown and reasoning), **title** (and SEO title),
**suggested caption**, **hashtags**, **recommended platform**, **per-platform
SEO** (title/description/tags/tips for each target), description, "why this
works", the **export details**, and the clip's **captions/transcript** (range-
filtered, clip-relative timestamps).

How it works:
- **`backend/services/clip_seo_sidecar.py`** (new): pure-stdlib formatter +
  best-effort writer. Reads every field defensively (pydantic model *or*
  persisted dict), filters captions to the clip window, and renders a tidy
  sectioned report. A sidecar problem never fails the actual export.
- **`clip_exporter.export_clip`** drops `[QUALITY] Title.txt` next to the MP4
  after each render (gated by `CLIP_EXPORT_SEO_SIDECAR`, default on), so **all**
  export paths — UI, API, agent, batch, share — get it for free.
- **Auto-download**: the export's WebSocket `export_complete` now carries a
  `seo_url`; `useEncodingManager` fetches and saves it right after the clip,
  so a single Export click lands both `[1080P] Title.mp4` and
  `[1080P] Title.txt`. The share-page URL rewriter is applied to it too.
- **Re-download**: a new `GET /api/jobs/{job}/clips/{clip}/seo.txt` builds the
  file *fresh* from the live job (so it reflects SEO edits made after export and
  works for clips exported before this shipped). Companion "SEO .txt" links sit
  next to every "Download clip" affordance — the ClipSEO toolbar, the Analysis
  **Exported Clips** list, and the Logs **Exported** list.

---

# ClipAI — Transcript-tab edits now reach the subtitle elements (and back), even with a translation

Editing a line in the Transcript tab (text, speaker, split/merge, delete, insert)
didn't change the subtitle elements on the video — and timeline edits didn't always
land in the transcript either. Root cause: when a translation exists, the tab shows
the **translated** transcript by default, but every edit endpoint blindly wrote to
the **source** transcript. The two tracks have *different segmentation* (one source
cue → several translated cues), so a translated-list index lands on a different
source cue or out of range — the edit silently hit the wrong line or 404'd, so it
"didn't take."

Fix — every single-track edit now declares which track it targets, and the backend
honours it:
- **`backend/routers/jobs.py`**: new `_resolve_transcript_rows(job, target)` /
  `_persist_transcript_rows(...)` helpers pick and save the right list. `target=
  "translated"` edits `translated_transcript` *only when one exists*, else falls
  back to the source. Wired through PUT (update), DELETE, POST (insert) and
  bulk-update-speaker; each response echoes the resolved `target`.
- **`TranscriptViewer.jsx`**: derives `editTarget = (hasTranslation &&
  !showingOriginal) ? 'translated' : 'original'` and sends it on all five edit
  calls, so it always edits exactly the track it's displaying.
- **`VideoEditor.jsx`**: new `transcriptTarget` prop is sent on the forward-sync
  PUT (timeline → transcript) so element edits hit the displayed track too;
  reverse-sync already keys `transcriptIndex` off the same track.
- **`Analysis.jsx`** passes a `transcriptTarget` that flips together with
  `activeTranscript`; **`ClipSEO.jsx`** passes one mirroring its `stableTranscript`.

Net effect: text/speaker/timing/add/delete stay in lockstep both ways, on the
original *and* the translated track, on desktop and mobile.

---

# ClipAI — No false "Reconnecting to container…" during analysis: health probe decoupled from Ollama

The connection banner showed "Reconnecting to container…" mid-analysis even though
the container was reachable and the pipeline was progressing. `useConnectionStatus`
pings every 8 s with a 5 s `AbortSignal` and flips to "reconnecting" after 2
consecutive failures — but it pinged `GET /api/providers/status`, which `await`s a
5 s Ollama `/api/tags` call. During offline processing Ollama is busy serving the
pipeline, so that call is slow; the frontend's 5 s timeout cancels the request
*before its 30 s cache is populated*, so the next poll misses the cache and times
out too → two consecutive failures → false banner.

Fix: a dependency-free liveness probe.
- **`GET /api/health`** (new, `main.py`): returns `{"status":"ok"}` instantly — no
  provider/Ollama/disk work. It only reflects whether the web server's event loop
  is responsive, which is what "is the container reachable" actually means.
- **`useConnectionStatus`** now pings `/api/health` instead of
  `/api/providers/status`. Provider/Ollama health is a separate concern (still shown
  in the models/diagnostics panels).

(The heavy pipeline work is offloaded to threads, so the event loop stays responsive
and the trivial probe returns well under the 5 s timeout even mid-export.)

---

# ClipAI — Preview player loads during analysis (no more blocking browser-preview transcode)

The in-page preview player wouldn't load while an offline job was processing. Root
cause: `GET /api/files/{id}/video.*` (`main.py:serve_file`) `await`ed
`ensure_browser_preview()` for the source video on EVERY request — which re-ran
`ffprobe` per Range request and, for non-browser-friendly sources, kicked off a
full (up to 30-min) ffmpeg transcode. During offline analysis that work contended
with the saturated pipeline, so the `<video>` request hung and the player never
loaded.

Fix — the request path never blocks on ffprobe/ffmpeg now:
- **`browser_preview.cached_browser_preview()`** (new): non-blocking resolution —
  returns the cached `browser_preview.v2.mp4` if fresh, the raw source if we've
  already determined none is needed (a new `.none` sentinel, written once so
  scrubbing an already-compatible MP4 no longer spawns an ffprobe per seek), or
  `None` when unresolved.
- **`serve_file`** serves the cached preview when ready, otherwise streams the RAW
  source immediately (range requests still work), and only schedules preview
  generation **in the background** — and only when `pipeline.is_job_analyzing(id)`
  is False, so the transcode can't steal CPU/GPU from the live pipeline.
- Background generation (`_schedule_browser_preview`) is deduped per source and
  holds strong task refs.

Net: the player loads instantly during analysis (raw source); the polished
browser preview is still produced afterward for incompatible/oversize sources.
New unit tests in `tests/test_browser_preview_nonblocking.py` (added to CI).

---

# ClipAI — Durable PROCESSING LOG: survives tab reload, new device, container restart

The Analysis page's PROCESSING LOG was built purely from live WebSocket messages
into local React state (`activityLog`), with no fetch of history on mount — so a
tab reload, opening the job on another device, or a container restart reset it to
empty and only showed FUTURE events. The full pipeline journey was lost.

Now every user-facing event is persisted server-side and re-fetched on load:
- **`services/job_events.py`** (new, stdlib-only): `broadcast_ws` appends each event
  append-only to `/data/uploads/{job_id}/events.jsonl` — cheap one-line writes that
  never rewrite the multi-MB `job.json`, captured even when no client is connected.
  Keepalive `heartbeat`/`pong` pings are skipped and rapid same-stage `status` ticks
  are throttled (8 s) so the durable log carries meaningful events, not noise; a
  one-time compaction caps file growth on very long runs.
- **`GET /api/jobs/{id}/events`** (new, same owner/admin auth as the other job
  routes) returns the most-recent events in chronological order.
- **Analysis.jsx** hydrates `activityLog` from that endpoint on mount and merges it
  with any live entries (time-ordered, de-duped via `eventToLogEntry` +
  `mergeLogEntries`), so the log renders in full on a fresh tab/device and keeps
  streaming live afterward.
- **PIPELINE PROGRESS tracker is durable too.** Each persisted status event carries
  `stage_id` + `ts`, so the same hydration fetch rebuilds the multi-stage bar's
  state (`reconstructStageState`): the active stage, per-stage durations, pipeline
  start, and elapsed — instead of resetting to a blank bar on refresh. (Stage
  CHANGES are never throttled server-side, so every stage's first event survives and
  the transitions are exact.) The elapsed timer is gated on a non-terminal, loaded
  job so a refreshed COMPLETE job shows a frozen total, not a runaway count.

New unit tests in `tests/test_job_events_persistence.py` (added to CI).

---

# ClipAI — Audio preconditioning runs at 16 kHz (GPU not applicable): ~3× faster offline

Answering "would audio preconditioning speed up on the GPU?": **no — ffmpeg's audio
filters (`highpass`/`afftdn`/`loudnorm`) are CPU-only; there is no CUDA build of
them (ffmpeg GPU accel is video decode/encode/scale).** A torch/torchaudio GPU
rewrite is possible but not worth the VRAM contention on a 4 GB GTX 1650 that
Whisper needs. The real, GPU-free win: the preconditioning filtergraph ran at the
SOURCE rate (e.g. 48 kHz) and only resampled to 16 kHz at output, so `afftdn`/
`loudnorm` chewed ~3× more samples than necessary. `build_precondition_filters` now
prepends `aresample=16000` so the whole chain runs at Whisper's 16 kHz target —
lossless for ASR (Whisper consumes 16 kHz mono regardless) and ~3× fewer samples,
stacking with the earlier "drop afftdn on long videos" change. Tests updated.

---

# ClipAI — "Stuck at faces, no VRAM" was the audio-precondition pass: drop afftdn on long videos + live audio progress

A 128-min offline run looked frozen at the face step with the VRAM bar reading
"No models loaded". The log told the real story: the `frame+audio extraction`
stage runs frame extraction and audio extraction CONCURRENTLY (`asyncio.gather`)
and only advances when BOTH finish. GPU frame extraction completed in ~7 min
(339 frames at 05:03:02), but there was no "Audio extraction complete" line after
it — the audio-preconditioning ffmpeg pass (`highpass,afftdn=nf=-25,loudnorm`) was
still grinding. `afftdn` is an FFT spectral denoiser that runs CPU-bound at only a
few × realtime, so on a 2 h track it alone adds ~15-20 min. The faces/perceiver
step can't start until the stage finishes, so the run *looks* stuck at faces, and
"No models loaded" is actually CORRECT — Ollama was unloaded to free the 4 GB GTX
1650 for Whisper, Whisper hasn't loaded yet, and ffmpeg audio filtering is CPU, not
a GPU model. Nothing was broken; the audio precondition was just the silent long pole.

Fixes:
- **Drop only `afftdn` on long videos** (`pipeline_helpers.build_precondition_filters`,
  used by both `frame_extractor.extract_audio` and the `reframer_audio` fallback).
  highpass + single-pass loudnorm are far cheaper and carry most of the coverage
  benefit, so tracks over `WHISPER_PRECONDITION_DENOISE_MAX_MIN` minutes (default
  45; 0 disables the cap) keep `highpass,loudnorm` and skip the expensive denoise.
  Cuts the 2 h audio pass from ~15-20 min toward ~3-5 min.
- **Live audio progress** (`extract_audio` now takes `video_duration` +
  `progress_callback`; `_run_subprocess_with_file_progress` polls the growing WAV
  size against the expected PCM byte count). The pipeline's `_audio_progress` pushes
  "Preconditioning audio for transcription… N%", so once GPU frames finish the UI no
  longer freezes on the last "Extracted N frames" message — it shows the audio pass
  advancing, which also keeps the stuck-timer reset. Accurate messaging for the
  phase that previously looked dead.

New unit tests in `tests/test_audio_precondition_chain.py` (added to CI).

---

# ClipAI — Transcript tab: clearer blue "now-playing" highlight + smoother mobile autoscroll

The transcript tab already tracks playback (TranscriptViewer computes the active
line from the player's currentTime, highlights it, and auto-centers it via a
manual-override-aware lerp; both the Analysis tab — side-by-side inline player on
desktop, stacked on mobile — and the ClipSEO pages feed it a live currentTime). Two
polish fixes so it reads clearly and scrolls right on both desktop and mobile:
- The active line's highlight was a faint 18% blue tint; bumped to a clear 32% blue
  with a 4px ``#0a84ff`` left bar so the currently-spoken line is obvious.
- Added ``WebkitOverflowScrolling: touch`` to the transcript scroll container for
  momentum scrolling on iOS. (The container already had ``overscroll-behavior:
  contain`` + ``touch-action: pan-y``, so manual scrolling and the auto-center
  coexist on touch devices.)
Both changes live in the shared ``TranscriptViewer``, so they apply on the Analysis
transcript tab and the SEO pages at once.

---

# ClipAI — Much faster clip export: GPU encode + parallel + optional stream-copy

Clip export was the single longest tail of the run — ~48 min for 63 clips in the
latest log. Cause: ``reframer_clipper._export_clip`` re-encoded every candidate
clip on the CPU (``libx264 -preset fast -crf 18``) even though the box has NVENC,
and exported them one at a time. The candidate clips are plain time-cuts of the
source (no reframing/subtitles burned in — that happens later on export), so the
heavy CPU re-encode was pure waste.

Speedups (fastest first; ``_export_clip`` falls through each):
- **GPU encode (default).** Reuses the exporter's ``_gpu_encode_args`` (NVENC/
  VAAPI/QSV, respecting the GPU toggle) at ``CLIP_EXPORT_CRF`` (21, visually
  transparent for review clips; 18 was overkill) — frame-accurate and ~5-10×
  faster than CPU libx264. Falls back to ``libx264 -preset veryfast`` if the GPU
  encode fails. ``-ss``/``-to`` stay before ``-i`` (fast keyframe seek).
- **Parallel export.** ``CLIP_EXPORT_CONCURRENCY`` (default 2) runs several
  independent clip encodes at once via a thread pool; progress is reported on
  completion and stays serialized, so the "Exporting clips… (k/N)" status and the
  stuck-timer still behave. Clips are re-sorted to chronological order after.
- **Stream-copy (opt-in, ``CLIP_EXPORT_STREAM_COPY``).** No transcode at all —
  remux the bytes; the whole export phase drops from many minutes to seconds. The
  cut snaps to the nearest keyframe (a clip may start a second or two early), so
  it's off by default and ideal when you just want fast previews to review.

Net for that 63-clip run: ~48 min → roughly 5-10 min on GPU (×~2 with the default
concurrency), or seconds with stream-copy. New config: ``CLIP_EXPORT_CRF``,
``CLIP_EXPORT_CONCURRENCY``, ``CLIP_EXPORT_STREAM_COPY``. (Separately, fewer clips
also helps linearly — set ``CLIP_COUNT`` if 63 candidates is more than you need.)

---

# ClipAI — Readable captions: merge choppy fragments + recurring-name glossary in cleanup

Follow-up to the fresh run (now fully translated + phantom-free): the transcript
still read choppily — strings of 2-3 word cues ("So am I planning on" / "eating
with" / "you today too") because VAD over-segments slow/paused speech and the
translator emits one cue per fragment. Short cues also fail the readability grade's
duration sub-score (sub-833ms flashes), capping it at B.

A — Greedy phrase merge (the YouTube/Netflix look). New
``subtitle_formatter._merge_for_readability`` runs BEFORE the splitter in
``enforce_readability``: it combines consecutive SAME-SPEAKER cues into the longest
caption that still satisfies every limit — ≤ 2 lines × 42 chars, ≤ 4.5 s, and
≤ the CPS cap (Netflix 17/21 Latin, 13 CJK) — bridging only small gaps (continuous
speech, ≤ ``SUBTITLE_MERGE_MAX_GAP_MS`` = 1200 ms), never a real pause, a speaker
change, or a ``[♪ music ♪]`` marker. So fragments become complete phrases and only
genuinely-long cues are then split at clause boundaries. This lifts the duration +
gap sub-scores (fewer, fuller cues) toward an A and reads far more naturally, the
same on offline and cloud (shared formatter). New config:
``SUBTITLE_MERGE_MAX_GAP_MS``.

B — Recurring-name glossary in the per-cue cleanup. The auto recurring-terms
glossary (``glossary.extract_recurring_terms``) was already fed to the main
``translate_via_llm`` pass; it now also primes the per-cue LLM cleanup
(``_llm_cleanup_untranslated``) so any re-translated stragglers render names the
SAME way (no "Mika"/"Mikka"/"Mikako" drift, no coined name flattened to an ordinary
word). (One-off ASR mis-hearings of a name still need a bigger model or a
user-supplied glossary.json — the recurrence-based glossary only pins names that
actually repeat.)

Regression tests for the merge + word-boundary split added to
``backend/tests/test_repetition_and_timeline.py``.

---

# ClipAI — Subtitle splitter no longer breaks mid-word ("You said th" / "is is …")

A fresh run came back fully translated and phantom-free (the gap-fill + QA fixes
landed), but the readability splitter produced mid-word cue breaks like
``[4:40] You said th`` / ``[4:43] is is delicious cake``. Cause: in
``subtitle_formatter._find_split_point``, the Latin word-boundary fallback used
``text.index(words[mid])`` — which returns the first SUBSTRING match. For
"You said this is delicious cake" the middle word "is" matched INSIDE "th[is]",
so the split landed at column 11 (mid-word). It now splits on the actual
whitespace position nearest the centre, so every emitted piece is whole-worded.
Regression tests added (in CI). This mainly bites LLM-translated cues, which have
no word-level timing and therefore fall through to this text-only split path.

---

# ClipAI — Translation QA on every path: no subtitle is left in the source language

Subtitles were still shipping with source-language cues. The cause was a hole in
where the cleanup ran, not a missing capability:

- The per-cue re-translate QA (`_llm_cleanup_untranslated`) ran **only on the
  offline-NMT path**. The latest run took the **editorial-LLM path**
  (`fraction_untranslated < 0.20` → returned early), so it shipped the residual
  source-language cues *without* cleanup. The Whisper-native path had the same hole.
- The final purity gate only **rejected** a wholesale (>20%) half-source draft
  (keeping the labelled source transcript) — it never **fixed** the ≤20% stragglers.

Why the earlier steps don't cover this:
- **Transcript step** dedups/cleans the *source* transcript (repetition-loop +
  phantom filters). It doesn't translate, so it can't guarantee target-language.
- **Subtitle polishing** (MT post-edit) only *refines already-translated* text and
  is explicitly forbidden from reintroducing the source language — it assumes its
  input is already translated, so it doesn't re-translate source-language cues.

Fix — the QA now runs on the FINAL transcript and on every engine path:
- `_llm_cleanup_untranslated` is now invoked on **all three** `translate_subtitles`
  paths (LLM / Whisper-native / NMT), not just NMT — so no engine can ship
  source-language stragglers.
- A **final QA pass** runs on the polished, deduped, resegmented transcript right
  before the purity gate: it detects any cue still in the source script and
  re-translates it one-by-one with the LLM (idempotent + fail-soft; a clean track
  is a no-op). The existing purity gate remains the backstop.

Combined with the gap-fill flood fix (below — which removes the hallucinated
run-ons at the source, before they ever reach translation), the persisted track is
fully target-language with no repeated/hallucinated lines. (Offline runs without
any LLM are unchanged: the cleanup no-ops without an orchestrator and the purity
gate still guards.)

---

# ClipAI — Phantom flood traced to gap-fill: cap it on mostly-silent video + make the confidence gate actually fire

The latest run's transcript still showed the phantom flood (repeated "Don't let" /
"So nice" / "Hmm." and verbatim Japanese run-ons, many left untranslated). The log
pinned the real cause — and why the earlier phantom filter didn't catch it:

- **Gap-fill was the flood.** On this ~70%-silent (128-min) video the gap-fill pass
  re-transcribed **5326 s (89 min) of "gaps" with VAD off**, adding **653 segments**
  to a 133-segment main pass — almost all hallucinated cues over music/silence. The
  untranslated Japanese run-ons in the output were exactly these (FuguMT can't
  translate looped run-ons, so they shipped in source form).
- **The confidence gate could never fire on them.** Gap-fill only keeps segments
  BELOW its `no_speech_threshold` (0.25), so every gap-fill cue has a LOW
  `no_speech_prob` — but the gate required `no_speech_prob ≥ 0.50`. Result: only
  ~4 cues quarantined on a video drowning in them. And the gate was only wired into
  the main loop, never applied to gap-fill output.

Fixes:
- **`WHISPER_GAP_FILL_MAX_FRACTION` (new, default 0.6):** when uncovered gaps exceed
  this fraction of the video, the content is mostly non-speech and `_gap_fill_pass`
  now SKIPS — the VAD main pass is far more reliable there. Speech-heavy videos
  (small gap fraction, e.g. the Gundam case gap-fill exists for) are unaffected.
  This alone removes the 653-cue flood here (786 → ~133 segments).
- **`WHISPER_PHANTOM_MIN_NO_SPEECH` default 0.50 → 0.0:** the low-confidence
  conjunction (avg < 0.40 AND ≥80% of words < 0.4) is the real discriminator and
  must not be gated behind a HIGH `no_speech_prob` it will never see on gap-fill /
  VAD-kept cues. Real speech essentially never has 80%+ of its words below 0.4
  confidence, so it's preserved.
- **Confidence gate now applied to gap-fill output** (`_gap_fill_pass`), where the
  hallucinations actually live, using the same `is_low_confidence_phantom` helper
  with the no_speech requirement off.

Net: the gap-fill hallucination flood is removed at the source on mostly-silent
content, and on normal content low-confidence gap-fill phantoms are dropped — so the
transcript tracks the real (sparse) dialogue ~1:1 and far fewer cues reach
translation. New regression test in `tests/test_phantom_hallucination_filter.py`.

> Note: this run's audio was extracted before the deploy (old `afftdn` chain, reused
> on resume), so the 16 kHz-resample/afftdn-skip speedups will only show on a fresh
> upload.

---

# ClipAI — Phantom-hallucination removal: TACT confidence gate + unconditional loop dedup (1:1 transcript)

A ~79 %-silent JA source produced a transcript flooded with invented cues that
translated through unchanged, so the output read nothing like 1:1: short English
phantoms over silence — "Don't let" (13×), "So nice" (12×), "Hmm." (11×) — plus
long Japanese run-ons re-emitted VERBATIM 10–11× each (decoder loops over the
quiet stretches). Counting cue bodies in the shipped transcript: **551 of 1806
cues (30 %) were scattered loop-repeats**, and the short English ones weren't
repeats of real dialogue at all — they were silence phantoms.

**Is TACT being used?** Yes — the Temporal Audio Coverage ledger (20 ms bins:
covered_speech / low_confidence / covered_silence / quarantined …) is built every
run and logged ("TACT coverage: … covered_silence 79.2 %, low_confidence 6.6 %,
covered_speech 14.2 %"), and it drives the reframer. **But until now it was a
reporting/perception layer only** — its silence/low-confidence classification was
never fed back to DROP transcript cues. The cues handed to translation were
filtered solely by the exact-match boilerplate blocklist, a `no_speech_prob>0.7`
clamp, and an in-segment repetition test — none of which catch a short,
low-confidence, repeated fragment sitting in silence. That gap is the phantom flood.

Fix — make TACT actually filter the transcript:
- **Confidence-gated phantom filter (new `transcript_dedup.is_low_confidence_phantom`,
  wired as hallucination check #4 in `reframer_audio.transcribe`).** Reuses the
  ledger's OWN low-confidence signal (word conf < 0.4): a cue is dropped when its
  words are overwhelmingly low-confidence (avg < `WHISPER_PHANTOM_MAX_AVG_CONF`
  0.40 AND ≥ `WHISPER_PHANTOM_MIN_LOWCONF_FRAC` 0.80 of words under 0.4) **AND**
  Whisper itself doubted there was speech (`no_speech_prob` ≥
  `WHISPER_PHANTOM_MIN_NO_SPEECH` 0.50). All three must hold, so genuine quiet
  speech — confident words even at a moderate no_speech_prob — is preserved.
  Gated by `WHISPER_PHANTOM_FILTER_ENABLED` (default on).
- **Repetition-loop dedup now runs UNCONDITIONALLY** (was only invoked when the
  gap-fill pass produced segments). The MAIN pass loops too, and on a mostly-silent
  video gap-fill may add nothing yet the primary transcript still carries the
  verbatim repeats. `drop_repetition_loops` keeps the earliest occurrence (one for
  long lines, ≤3 for short interjections; bracketed `[♪ music ♪]` markers exempt).
  Verified on the real transcript: 1806 → 1255 cues (551 loop-repeats removed),
  each long run-on collapsed 11→1.

Net: the long Japanese lines appear once each, the short English silence-phantoms
are removed before they're ever translated (also saving translation budget), and
the transcript tracks the audio 1:1. New unit tests in
`tests/test_phantom_hallucination_filter.py` (+ `backend/tests/test_repetition_and_timeline.py`
added to CI) pin both behaviours.

---

# ClipAI — Offline translation completeness: per-cue plain-text LLM cleanup of NMT leftovers

A 2 h JA→EN run still shipped ~25 % of cues in Japanese. The logs pinned it
exactly: the local LLM ran (`text_completion via ollama qwen2.5:3b … completed
in 128.9s`) but its batched JSON-array output was unparseable
(`editorial model returned no usable output`), so `translate_via_llm` bailed →
fell to FuguMT, which left `185/742 cue(s) untranslated (recovered 0/185)` on the
long colloquial run-ons. The previous leftover-cleanup also used the JSON-array
path, so it bailed the same way.

Fix: the NMT-leftover cleanup (`pipeline._llm_cleanup_untranslated`) now
re-translates each still-source cue **one at a time with a PLAIN-TEXT request**
("translate this line; reply with only the translation"). That's the most robust
local-model call — no JSON to misparse and no cross-cue alignment risk — so it
recovers the cues FuguMT couldn't. It dedups identical run-ons (they repeat
across many timestamps) so it's fast + consistent, is bounded by a cue cap +
wall-clock budget, surfaces an accurate "Recovering N untranslated subtitle(s)…"
status, and is fully fail-soft (any failure keeps the existing cue). New knobs:
`TRANSLATION_LLM_CLEANUP_MAX_CUES` (default 500; 0 disables),
`TRANSLATION_LLM_CLEANUP_BUDGET_S` (default 1200).

Separate ASR-quality issue (not translation): on this ~79 %-silent source Whisper
also emits short phantom repeats ("Don't let", "So nice") that translate through
unchanged — the hallucination blocklist catches fixed phrases but not these.
**Resolved by the TACT confidence gate + unconditional loop dedup — see the entry above.**

---

# ClipAI — Accurate live messaging during Whisper + translation (no false "stuck" banner)

The long Whisper and translation phases showed the red "No progress update —
pipeline may be stuck" banner even though the run was alive (just slow), and the
messaging was inaccurate (the heartbeat said "scene analysis" while Whisper ran;
translation sat at a static 63 % "Translating…").

Root cause: the frontend `heartbeat` handler's comment claimed to "reset the
stuck timer" but never did — so the 15 s heartbeats (and the per-batch
`background_task` translation pings) never reset it, and the timer climbed to the
300 s alarm. And the heartbeat label was derived from the coarse JobStatus, not
the actual sub-phase.

Fixes:
- **Frontend (`Analysis.jsx`):** a flowing heartbeat means the event loop is
  alive and the run is NOT stuck — so `heartbeat` AND running `background_task`
  messages now reset the stuck-timer. The "may be stuck" banner now only fires
  when heartbeats actually STOP (event loop blocked / process dead).
- **Backend (`pipeline.py`):** `_update_progress` takes a `heartbeat_label`
  override so the heartbeat names the real sub-phase — "transcription" while
  Whisper runs inside the ANALYZING_SCENES stage (was the misleading "scene
  analysis"), "face + motion detection" during face analysis.
- **Backend translation progress:** the per-batch translation status now pushes
  a REAL progress update (not just a `background_task` ping) — `translation_progress_pct`
  (extracted + unit-tested) maps the "(a/b)" cue count onto the 63→69 % band, so
  the bar advances and the message reads "Translating subtitles… (310/621)"
  instead of a static label.

---

# ClipAI — Clip export reports per-clip progress (no more false "pipeline may be stuck" banner)

Exporting many reframed clips (e.g. 63 clips × ~45-135 s each ≈ 47 min) is the
longest tail of a run, but the UI sat at ~96 % with a single static "Exporting
top clips…" the whole time and tripped the red "No progress update for 36m — the
pipeline may be stuck" banner. The clipper already called its progress callback
per clip, but the pipeline relay (a) compressed export into ~94→97 % (≈3 integer
ticks across 60+ clips) and (b) only emitted on an integer-% increase with a
static label — and the frontend's stuck-timer keys off `progress_message`
changes, which never changed.

Fix: the export loop now reports a per-clip MESSAGE ("Exporting clip k/N …"),
and the progress relay (`resolve_clip_progress`, extracted + unit-tested) always
forwards a message even when the % is flat (while staying monotonic). The
changing text resets the stuck-timer every clip (~45-135 s), so the alarming
300 s "may be stuck" banner no longer fires during a healthy long export, and
each tick also stamps `updated_at` so backend stale-job recovery sees liveness.
No frontend change needed.

---

# ClipAI — Offline translation no longer ships half-Japanese (LLM timeout + NMT leftover cleanup)

The offline JA→EN translation left ~18% of lines in Japanese. From the logs:
the editorial LLM (qwen2.5:3b) hit its per-batch timeout — `Text completion
timed out after 90s` — on the FIRST batch, so `translate_via_llm` bailed
entirely and fell back to FuguMT, which then left 119/661 colloquial run-on cues
untranslated (`completeness pass recovered 0/119`). The LLM path is the better,
COMPLETE one (it has its own 3-pass source-script retry); it just needed to not
get killed mid-answer.

Two fixes (both, per request):
- **Let the local LLM finish (`translator.py`).** The per-batch translation
  timeout is now a generous, configurable CEILING
  (`TRANSLATION_LLM_TIMEOUT_FLOOR`=180 s, `TRANSLATION_LLM_SECONDS_PER_SEGMENT`=12,
  was `max(60, 5×batch)`) — raising a ceiling never slows the fast path, it only
  stops a slow local model being killed and dropped to NMT. Batches are also
  smaller on Ollama (`TRANSLATION_LLM_BATCH`, auto=8) so each produces a short
  JSON array quickly + reliably.
- **Harden the NMT fallback (`pipeline.py`).** The offline router is LLM-free by
  design, but `translate_subtitles` (LLM-first → NMT) now runs a fail-soft
  leftover cleanup: after NMT, any cue still in source script is re-translated
  with the editorial LLM (leftovers only, bounded, never raises). So even when
  the run falls to FuguMT, the shipped subtitles aren't half-source.

Combined with the previous hallucination-blocklist additions, the translated
track should now be complete English without the "see you next time" / おわり
phantom repeats.

---

# ClipAI — Long videos: face detection no longer 5x over-samples + drop "see you next time"/"おわり" hallucinations

Two issues from a 128-min offline run (GTX 1650):

**Slowness — face/motion detection sampled 5x more than its own cap.** The log
showed `Reframer sample rate: 1.20 fps (128.0 min video, ~9212 samples,
cap=1800)` and `reframer_analysis finished in 3352.0s` (~56 min). The sample
cap (1800) was being overridden by the min-fps *floor* (1.2): for any video
longer than ~25 min, `max(floor, …)` sampled at the floor instead of honoring
the cap, so a 2 h video analysed ~9 200 frames instead of 1 800 — ~5x the
intended work. Fixed: `resolve_sample_fps` (extracted + unit-tested) now treats
the floor as a comfort minimum that only applies while it stays within the cap;
long videos honor the cap (128 min → 1 800 samples → ~12 min instead of ~56).
A configurable `REFRAMER_ABS_MIN_SAMPLE_FPS` (default 0.2) keeps multi-hour
videos from under-sampling.

**Transcript/translation quality — "see you next time" / "おわり" hallucinations.**
The video is ~79% silence, and Whisper hallucinated phantom end-of-segment
phrases over the quiet stretches — "See you next time", "I'll see you next
time", and the bare Japanese 終わり/おわり ("the end") — which then dominated the
translated subtitles. Those weren't in the multilingual hallucination blocklist;
added them (and the JA また次回/また来週 family). Exact-cue match only, so real
dialogue (e.g. おわりにしましょう) is untouched.

NOTE: this run ALSO had the offline NMT (FuguMT) leave ~18% of cues
untranslated (long un-punctuated run-on lines) after the local LLM (qwen2.5:3b)
hit its 90 s per-chunk timeout — that half-Japanese output is a separate,
model-capability issue still being addressed.

---

# ClipAI — Transcription no longer times out on long videos (the real "translation didn't run" cause)

A 2 h video came back with `0 transcript segments`, an empty summary ("no
transcript was available"), and no subtitle translation. Root cause from the
logs: the reframer's `transcribe()` RE-EXTRACTED audio from the video with the
preconditioning chain (`highpass + afftdn denoise + loudnorm`) under a hardcoded
**240 s** timeout — but that exact chain takes ~10 min on a 2 h file (the main
pipeline's own `audio.wav` extraction logged 624 s). So it timed out, Whisper
got no audio, and with no transcript the pipeline silently skipped translation
and produced an empty summary. Translation was never the problem — there was
simply nothing to translate (mode-independent: same in offline and cloud).

Fixes (`backend/services/reframer_audio.py`):
- `transcribe()` now **reuses the pipeline's already-extracted `audio.wav`**
  (same job dir, identical preconditioning chain) instead of re-extracting —
  eliminating ~10 min of duplicate ffmpeg work AND the timeout entirely.
- The fallback extraction (when no `audio.wav` exists) now uses a
  **duration-scaled timeout** (floor 10 min, ~realtime + slack) and catches
  `TimeoutExpired` so a slow/odd codec falls back to a plain copy instead of
  killing the whole transcript.

Supporting changes:
- `pipeline.py` now emits a visible job warning + WS notice when analysis
  produces **0 transcript segments**, so a transcription failure can't masquerade
  as a clean COMPLETE with an empty summary again.
- `pipeline_checkpoint.py` `CHECKPOINT_VERSION` bumped to 2 so the empty-transcript
  checkpoint the failed run cached is invalidated — the next analyze/resume
  re-runs the engine and transcribes correctly.

Verify on the host: re-analyse the 2 h video; the log should show
`[AUDIO] Reusing pre-extracted audio.wav … skipping redundant re-extraction`,
a non-zero `transcript segments` count, a populated summary, and — for a
non-English source — the subtitle translation running.

---

# ClipAI — Auto-resume now defers + serializes its runs (transcript/translation completes)

Follow-up to the resume feature: a revived job could finish WITHOUT its
transcript polish or translation. Root cause was timing, not the resume logic
(which reuses the exact same post-engine code path). Auto-resume fired its
`run_analysis` as a fire-and-forget task *during* container startup, so the
resumed pipeline raced the Whisper/GPU preload and the model + settings/auth
warmup. The AI stages — transcript polish, subtitle translation, summary — then
hit GPU/model contention (and, for cloud, keys/overlay not yet applied) that a
normal user-triggered run never sees, so they failed-soft and the job completed
with a raw/untranslated transcript.

Fix (`backend/main.py`): resumes are now **queued and drained by a single
background task** that (a) waits a grace period for startup to settle
(`CLIPAI_RESUME_DELAY_S`, default 30s) and (b) runs each resumed job to
completion **sequentially** — never two heavy runs at once on the shared GPU.
A revived job therefore executes in the same fully-warmed environment a
continuous run does and completes every step (polish → translation → summary →
clips → COMPLETE) identically. New test:
`tests/test_auto_resume.py::test_resume_drainer_runs_jobs_sequentially_after_delay`.

Verify on the host: analyse a non-English video, restart the container
mid-run, and confirm the log shows `Auto-resume: starting deferred run …` after
the grace delay, the `subtitle_translation … complete` background-task event,
and a populated translated transcript on the finished job.

---

# ClipAI — Resume failed/interrupted jobs (engine checkpoint + auto-resume on restart)

When the container went down mid-analysis (restart, crash, OOM), the worker
thread died and the job could only be RE-ANALYSED FROM SCRATCH — re-running the
single most expensive stage (face/motion detection + Whisper transcription +
the reframe planner) every time. The frame+audio extraction was already cached,
but nothing after it was. This adds a true resume.

**1. Engine checkpoint (`backend/services/pipeline_checkpoint.py`).** Right after
the reframer engine finishes, its in-memory `PerceptionResult` + `RenderPlan`
are serialized to a per-job `checkpoint/` dir. On the next run, if a checkpoint
exists for THIS exact source + analysis config, the pipeline restores it and
skips straight to bridge → summary → clip detection — no re-detection, no
re-transcription. The round-trip is faithful for everything the post-engine
stages read: the int-keyed timeline dicts (`face_timeline`, `motion_timeline`,
…) are restored to **integer** keys (downstream does `int(t_ms / 1000)` on them
and would crash on JSON's string keys). The millisecond-resolution
`coverage_ledger` is deliberately NOT checkpointed — it's read only by the
planner (already run by checkpoint time), and serializing/rebuilding its
tens-of-thousands of 20ms bins in pure Python on resume held the GIL long
enough to starve the event loop and freeze the live `/diagnostics/gpu-status`
VRAM poll. Dropping it keeps the checkpoint tiny and resume non-blocking.

Reuse is gated on a *signature* — source SHA-256, sample-fps, aspect ratio,
source language, vocal-separation setting, and a fingerprint of the reframer
env flags — so a stale checkpoint is never used after the source or analysis
settings change. Bypass with `CLIPAI_FORCE_REANALYZE=1`.

**2. Auto-resume on container restart (`backend/main.py`).** Startup recovery
now: (a) completes jobs that already have results, then (b) **re-queues**
result-less interrupted jobs to RESUME from the checkpoint (continue where they
left off) instead of marking them FAILED. A persisted `resume_attempts` counter
caps this at `CLIPAI_MAX_RESUME_ATTEMPTS` (default 3) so a job that crashes the
container can't relaunch itself forever — past the cap it's marked FAILED.
Disable the whole behavior with `CLIPAI_AUTO_RESUME=0` (reverts to the old
fail-and-wait-for-manual-re-analyse flow). The counter resets on COMPLETE and
on a user-triggered re-analysis.

New env knobs: `CLIPAI_AUTO_RESUME` (default on), `CLIPAI_MAX_RESUME_ATTEMPTS`
(default 3), `CLIPAI_FORCE_REANALYZE` (force a clean engine re-run).

Verify on the Unraid host (no GPU/weights/media here, so this is unit-tested +
code-traced only): analyse a video, `docker compose restart app` mid-run, and
confirm the log shows `RESUME: restored engine checkpoint … skipping detection
+ transcription` and the job finishes without re-running Whisper. New tests:
`backend/tests/test_engine_checkpoint.py` (serialization round-trip, int-key
restoration, signature gating) and `tests/test_auto_resume.py` (startup
complete/resume/fail routing + attempt cap).

---

# ClipAI — Transcription recall on music-heavy content (vocal separation + preconditioning)

Diagnosed from a real run's `clipai_logs_*.txt` on the Gundam Wing episode:
whole dialogue sections (e.g. 2:55–4:27, 10:08–11:08) came back as **silence**.
The log showed `covered_silence: 64.7%`, `0 hallucinations quarantined`, and
music suppression dropping only **3** cues — so the loss was NOT over-zealous
music suppression. The audio is dialogue under a loud music/SFX bed that
Whisper's VAD hears as no-speech and drops, even on the vad-off gap-fill pass.

Three stacked fixes (priority order):

1. **Vocal separation before ASR (`vocal_separator.py`).** Run Demucs
   `--two-stems vocals` as a *subprocess* (so all its GPU memory frees on exit)
   BEFORE Whisper loads, then transcribe only the isolated vocal stem. CUDA
   with a small `--segment` to fit the 4 GB GTX 1650, automatic CPU fallback,
   self-healing to the original audio on any failure. Threaded through
   `ReframeEngine → Perceiver → AudioIntelligence.transcribe(audio_path_override=)`.
   Gated by `VOCAL_SEPARATION_ENABLED` (default on, **no-op until
   `pip install demucs`**; htdemucs downloads ~80 MB on first run → needs net).

2. **Audio preconditioning now reaches the ASR.** `WHISPER_AUDIO_PRECONDITION`
   (highpass + afftdn denoise + loudnorm) was only applied to the
   diarization/music copy; the reframer transcribe path re-extracted RAW audio.
   Mirrored the exact chain into `transcribe()`'s own extraction (with a raw
   fallback) so faint speech is recovered as documented.

3. **Music suppression no longer deletes real dialogue.** Suppression inside a
   "music" span now drops only non-lexical vocalisations (`ああああ`/`lalala`),
   keeping lexically-diverse dialogue when the spectral classifier mislabels a
   loud-BGM scene as music (`SUBTITLE_MUSIC_SUPPRESS_VOCALIZATIONS_ONLY`).

Proper-noun errors (Darlian→"Dorian", "Hatsune Miku") are the **empty Custom
Vocabulary glossary**, not a bug — populate Settings → Custom Vocabulary.

Verify on the Unraid host (no GPU/weights/media here, so this is unit-tested +
code-traced only). After rebuild + `pip install demucs`, the
`clipai_logs_*.txt` should show: a `vocal_separation` stage, `[AUDIO] Using
pre-separated vocal track`, a higher `covered_speech` / lower
`covered_silence`, and the 2:55–4:27 / 10:08–11:08 dialogue present. New tests:
`backend/tests/test_vocal_separator.py`, extended `test_music_marking.py`.

---

# ClipAI — Offline Mode: one switch runs the whole pipeline on the local GPU

Settings → AI Provider now has an **Offline Mode (Local GPU)** toggle. One
switch routes all four stages a user thinks about — clip detection,
transcription, translation, and polishing — onto the local GPU (the GTX 1650)
with no cloud calls, and a four-stage status grid shows where each stage
actually runs (LOCAL / CLOUD), updated live from the backend.

It drives the existing self-hosted machinery rather than a parallel knob:
turning it on sets `SELF_HOSTED_MODE` and resets the per-engine overrides to
Auto, so clip detection runs on the local Ollama vision model (Replicate
disabled), the editorial chain collapses to `["ollama"]` (polishing +
LLM-first translation, then offline Whisper/NMT), and Whisper transcription
stays local as always. A collapsible "Advanced overrides" block keeps the
per-engine Local/Cloud/Auto dropdowns for power users. The control moved out
of the Prompts tab into AI Provider where the user asked for it.

Sequencing for a 4 GB card: the pipeline already evicts Ollama before
Whisper and releases Whisper VRAM before the editorial LLM. Added the missing
hand-off — `_free_editorial_vram_before_local_clips` unloads the editorial
Ollama model (keep_alive=0) and flushes the torch allocator right before
local clip detection loads its vision model, so the editorial LLM and the
vision model never have to share the GPU. No-op in the cloud.

New: `Settings.resolve_stage_source(stage)` maps the four user-facing stages
onto the two engine knobs; `/api/self-hosted/settings` now returns a `stages`
block. Covered by `tests/test_offline_mode_settings.py`.

---

# ClipAI — Fix the real reason translations were half-Japanese: source="auto"

The LLM translation logged "0% still source-script" yet the track was ~40%
Japanese. Root cause: the job's source language is "auto", and the language-
purity check only counted CJK for KNOWN CJK source codes — so for "auto" it
returned 0, the gate was effectively OFF, and the LLM (told to translate from
"the source language") left long narration / lyrics untranslated, which then
sailed through.

Fixes:
- The purity check is now CONTENT-based and TARGET-aware (`fraction_untranslated`):
  any CJK in the output of a non-CJK-target translation counts as untranslated,
  regardless of how the source was declared. So it works for "auto".
- `translate_via_llm` runs a completeness cleanup: after the main pass it
  re-translates any cue still in CJK script (up to 3 passes), so nothing is left
  in the source language.
- Resolve source "auto" → the language Whisper detected, so the LLM prompt is
  precise ("Japanese → English") and the gate has a concrete target.

Net: the translated track is fully target-language — the gate can no longer be
silently disabled by an "auto" source, and any stray untranslated cue is caught
and re-translated.

---

# ClipAI — Stop the post-edit from reverting LLM translations back to Japanese

The LLM translation (previous change) works: logs show `LLM translation: 116/116
segments -> en (0% still source-script)` — a perfect, complete English draft. But
the persisted track came back ~40% Japanese. Cause: the MT post-edit ran on it
"source-aligned" — comparing each English line against the Japanese SOURCE — and
reverted chunks (long narration, song lyrics) back to the source language.

Fix:
- Skip the post-edit entirely when the engine is the LLM (_used_llm): the LLM
  translated the source directly, so its output is already final, clean target
  text — there is nothing to "post-edit toward the source".
- For the NMT / Whisper paths that still post-edit, add a safety net: if the
  post-edit RAISES the source-script fraction (i.e. it reintroduced the source
  language), discard it and keep the pre-edit translation.

Net: the perfect LLM translation is persisted intact — fully English, every cue.

---

# ClipAI — Fundamental rethink: translate with the LLM, not Whisper's translate task

The translated track kept coming back HALF JAPANESE — dialogue in English, but
song lyrics, narration and hard segments left in the source language. Root cause
is architectural: translation was hardcoded "offline-only, never an LLM", forcing
Whisper's task='translate' as the primary path. On mixed/music content Whisper's
translate task silently TRANSCRIBES (source language) instead of translating, so
the output is a JP/EN mix. The coverage guard only checked timeline coverage, not
whether the result was actually English, so it passed the mix through.

Rethink — translate the SOURCE transcript text-to-text with the editorial LLM
(the same orchestrator already used for summary / SEO / polish):
- New translate_via_llm(): batches the source cues, asks the model to translate
  EVERY numbered line (lyrics + narration included) into a JSON array, maps the
  result 1:1 back onto the cues (timing + speaker preserved), applies the
  glossary, and splits-and-retries on any batch the model mangles. It bails
  cleanly (returns None) if the model is unusable, so offline users fall back.
- translate_offline now tries the LLM FIRST (TRANSLATION_PREFER_LLM, default on),
  then Whisper-native, then offline NMT.
- Language-purity gate (fraction_source_script): any "translation" still >20%
  source-script (CJK) is REJECTED — applied to both the LLM result and the
  Whisper-native result — so a half-translated track is never persisted; it
  falls through to a complete engine instead.

Net: every cue is translated, 1:1, with the high-quality model you already have
configured — no more Japanese left in the English track.

---

# ClipAI — Stop stale saves from wiping the translation (the real "not translated" bug)

Direct from a corrupted job on disk: `translated_transcript` was 0 (the pipeline
had persisted 279 English segments), the source was a doubled 217-segment copy,
and status had reverted to `detecting_clips` — all AFTER the run finished. A
whole-object `save_job` (a transcript-segment edit / reverse-sync that loaded the
job microseconds before the translation landed, then wrote its stale snapshot
back) clobbered the good data. Every display-side fix was futile because the
English was being destroyed on disk.

Fix — an anti-clobber guard in the central write (`_save_job_unlocked`):
- **Never wipe a non-empty `translated_transcript` with an empty one.** A real
  re-translation writes a NEW non-empty value, which still wins; only the
  empty-wipe (always a stale snapshot) is blocked. Applies to every write.
- **Never revert a terminal status** (complete/failed/cancelled) to a
  non-terminal one — but only on the `save_job` whole-object path, so a
  deliberate re-analysis reset via `update_job_status` is still allowed.

Both re-read the current on-disk copy under the per-job lock and refuse the
downgrade. Tests cover wipe-protection, terminal-status protection, legit
re-translation overwrite, and re-analysis still resetting status.

---

# ClipAI — Reliable translated-transcript loading (lightweight endpoint)

Across incognito + multiple devices the Analysis page kept showing the SOURCE
(Japanese) transcript even though the backend had persisted the English
(`translated_transcript`). Root cause: the UI reads the FULL job over the network
to get the translation, and that payload is large enough that over a tunnel the
fetch didn't reliably complete — so `translated_transcript` never reached any
client and `activeTranscript` fell back to the source.

Fix: a dedicated `GET /jobs/{id}/transcripts` endpoint that returns ONLY the
source + translated transcripts (+ status) — a tiny response that always lands.
The Analysis page pulls it on load, on every pipeline status change, and on a
short timer while running, and merges the result in. The English now appears as
soon as it's persisted, regardless of how big the rest of the job is.

---

# ClipAI — Whisper subtitle cues land on the audio (word-timed resegmentation)

Audit follow-up on the Whisper-native path. Whisper's translate task emits a few
LONG, multi-sentence cues but with per-word timestamps. The pipeline split those
into sentence cues only AFTER the MT post-edit — which clears word timestamps
when it rewrites text — so the split always fell back to a char-length
PROPORTIONAL guess. On a 40 s cue that puts internal sentence boundaries several
seconds off the audio (a short sentence with long dictation gets a tiny slice; a
long sentence spoken fast gets too much).

Fix — carry the word timestamps through and split BEFORE polishing:
- `_whisper_native_translate_segments` now passes Whisper's `words` through.
- The Whisper-native path resegments by sentence **before** the post-edit, using
  those word times for accurate boundaries (the glossary is re-applied to the
  split cues, since the word-timed split rebuilds text from the pre-glossary
  words). The post-edit then polishes the already-correctly-timed short cues
  (it preserves start/end), and the post-edit-stage resegmentation is skipped for
  this path. The NMT path (source-aligned 1:1) is unchanged.

Tests: word-timed vs proportional split (boundary at 30 s vs <15 s), and a
pipeline test that a long Whisper cue splits word-timed before polish.

---

# ClipAI — Stop dropping legitimately-repeated subtitle cues + steadier diarizer threshold

Two tuning fixes from comparing against repo-60 (which never mangles translated
transcripts because it does **no** post-translation processing — it saves the raw
cues).

**Don't loop-drop NMT output.** The translated transcript runs a
`drop_repetition_loops` dedup that removes Whisper's hallucinated repetition loops
(it loops on music/silence). But it drops **any** recurring long line across the
whole timeline — which on a song/AMV deletes legitimately repeated chorus and
narration lines (kept only the first of 6, in testing). Offline NMT translates the
source 1:1 and never hallucinates loops, so its repeats are real: the loop-drop is
now **gated to the Whisper-native path only**. NMT-translated choruses survive.

**ECAPA diarizer threshold 0.55 → 0.70.** 0.55 over-split noisy / music-heavy
audio straight into the 8-speaker cap. 0.70 is a steadier default for ECAPA
agglomerative clustering (configurable via `LOCAL_DIARIZER_THRESHOLD`).

---

# ClipAI — Fix: jobs stuck at "translating" forever (interrupted-run recovery)

Root-caused from a real run: a job re-analysed after completing died at the
translate stage and then **span at "translating" forever** in the UI — no clips,
source-language subtitles. The first run had finished cleanly (clips + 348
translated segments verified on disk); a second run re-entered, set status
`translating` (the only place that status is written), and never finished.

The bug: both recovery paths used a **hardcoded** in-progress set
(`analyzing_scenes, extracting_frames, generating_summary, detecting_clips`) that
**omitted `translating`** (and `transcribing`/`queued`). So a job stuck at those
statuses was neither reconciled-to-complete (it has results) nor failed — it just
spun.

Fix:
- Derive the non-terminal set from the `JobStatus` enum, so it can **never drift
  out of sync** again (covers every running phase).
- Staleness-guarded recovery (`_recover_stale_jobs`): a non-terminal job that no
  live worker is advancing (old `updated_at`) is flipped to **COMPLETE** when it
  has results, or **FAILED** when it doesn't. Startup runs it unconditionally
  (workers are dead); the periodic self-heal (every 10 min) uses 30 min / 2 h
  thresholds so it **never touches an in-flight run**. Stuck jobs now recover
  without a restart.

---

# ClipAI — Coverage guard: don't let sparse Whisper-translate leave Japanese gaps

Follow-up to the Whisper-translate reuse work. On a music/lyric-heavy video
(e.g. an anime AMV), Whisper's audio→English *translate* task skips/merges the
singing, so it emitted only ~35 cues where the source transcription had captured
122 — leaving long stretches with no English subtitle (they read as "still
Japanese"). Because the reuse change made Whisper-native run on the 4 GB card
instead of NLLB, those gaps surfaced in real output.

Fix: a **coverage guard**. After a Whisper-native pass, compare how much of the
audio timeline it covers against the source transcript; when it covers less than
`WHISPER_TRANSLATE_MIN_COVERAGE` (default 0.6) of the source, discard the sparse
output and translate the **full source via offline NMT** instead — so every
source cue gets an English line. Dialogue-heavy videos (where Whisper-native
covers ~all the speech) are unaffected and still use the higher-accuracy
single-pass translate. Threshold is configurable.

---

# ClipAI — Local audio diarization without a HF token + Whisper-translate reuse

Two local-first quality wins that need no external API and no token.

**Speaker diarization without a HF_TOKEN.** Without a token, pyannote's gated
model can't load and diarization fell back to a *visual* left/right mouth-motion
heuristic — useless for off-screen or audio-only voices, and it collapses
same-side speakers into one. There's now a real LOCAL audio diarizer
(`local_diarizer.py`): SpeechBrain ECAPA-TDNN speaker embeddings on Whisper's
speech segments, clustered (scipy, cosine) into who-spoke-when, emitting the
same `{time_ms: "SPEAKER_xx"}` timeline pyannote does — so speaker fusion and
the "Speaker N" labelling work unchanged. The ECAPA checkpoint is a PUBLIC
model (no token) fetched once (~80 MB) into `/data/models`, then fully offline.
Order is now: pyannote (if a token is set) → **local ECAPA** → visual heuristic.
It runs on CPU by default to avoid GPU contention during perception. Quality:
pyannote 3.1 (token) > local ECAPA > visual-only.

**Whisper native →English translate reuses the loaded model (works on 4 GB).**
Whisper's audio→English translate is the best LOCAL English path but was skipped
whenever <4 GB VRAM was free, because it would *reload* a second model — so on a
4 GB card it never ran. The analyze stage now KEEPS the transcription Whisper
engine loaded when an English translate is pending (instead of releasing it
early), the translate gate REUSES that resident model (no second load, so the
4 GB reload gate no longer applies), and the per-pass pre-flight uses a lower
free-VRAM floor when reusing (1.5 GB vs 3.0) so it can stay on the GPU — with a
clean OOM→CPU fallback and a defensive, idempotent VRAM release before the VLM
summary. Net: high-quality local Whisper translation for English targets even on
small GPUs; non-English targets still use NLLB + the AI post-edit.

---

# ClipAI — Translate to ANY language, completely (no cap, no source-language leftovers)

Subtitle translation must work — and finish — for whatever language the user
picks, with the transcript in the *target* language, not the source.

**Any target language, no "unsupported" stop.** The offline NLLB engine covers
~200 languages, but the ISO→Flores map (and the friendly-name
`SUPPORTED_LANGUAGES` the `translate-subtitles` endpoint validated against) only
listed 23 — so picking, say, Czech/Greek/Hebrew/Persian was rejected outright,
and an auto-detected *source* outside the 23 silently failed. Both maps now cover
~100 common languages (kept in sync), the endpoint guard accepts any
Flores-capable target (capability, not a short name list), and the Upload
language dropdown was widened to match.

**No source-language leftovers for any pair.** The completeness retry was
CJK-only — it caught Japanese left in an English track but not, e.g., English
left in a German track. It now flags an untranslated cue for ANY pair (output
still in a CJK source script when the target isn't CJK, OR a substantial cue the
engine echoed back unchanged) and retries it per-cue (which chunks long inputs).
The offline path has no early-stop cap; the rate-limit/consecutive-failure bail
only ever lived in the unused LLM translator, never in the NMT path. Net: the
persisted `translated_transcript` (what the UI shows when a target is set) is
fully in the user's chosen language.

---

# ClipAI — Complete offline translation + AI post-edit, and no phantom speaker color

Three fixes from a `ja→en` run whose subtitles came back half-Japanese with a
duplicate speaker swatch.

**Duplicate speaker colors.** Non-speech cues (`[♪ music ♪]`) are emitted with
an EMPTY speaker. The transcript-derived speaker list (`Analysis.jsx`,
`TranscriptViewer.jsx`) included that empty string as a distinct "speaker", so it
rendered as a phantom, unnamed swatch whose palette-by-position color collided
with a real speaker's — the two identical oranges. Both speaker lists now skip
blank/whitespace speakers (and de-dupe after trim).

**Translation left long cues untranslated.** The log said `70/78 changed → en`,
but the 8 misses were the long run-on cues (Whisper loops) — and those dominate
by volume, so the output read as mostly Japanese once the readability splitter
fanned them out. NLLB was handed each long cue whole, exceeding the fixed
256-token decode cap, and a failed cue silently kept its source text. Now the
offline engine is made *complete*:

  * `NMTTranslator`/`OpusMTTranslator` CHUNK over-long source cues at sentence →
    clause → hard boundaries before translating, and scale the decode budget
    with input length (was a fixed 256) — so a long run-on can't truncate into
    untranslated source.
  * `translate_with_context` skips the fragile context-join (which loses its `¶`
    markers when long) and goes per-segment for long batches.
  * A **completeness pass** in `_translate_via_nmt` detects any cue still in the
    source script (non-CJK target) and retries it per-cue (which chunks), so the
    offline engine never returns source-language text. Still offline-only — the
    AI is never called to translate.

**AI polishing now closes the quality gap (MTPE).** Per request, the offline NMT
still does the base translation, but the editorial LLM step is now machine-
translation *post-editing* instead of "readability-only": it aggressively
rewrites the rough NMT draft into natural, professional subtitles while
preserving meaning, line count, and timing. It runs as ONE dedicated, source-
aligned pass (the readability reflow stays in the steps after it), so each draft
line is shown next to its ORIGINAL source line as ground truth — letting the
model repair mistranslations without translating from scratch. The translation-
mode length guards were loosened (no word-count clamp) so fluent rewrites aren't
rejected. Gated on `AI_TRANSCRIPT_CORRECTION` as before.

---

# ClipAI — Offline NMT uses the GPU when there's room (4 GB card), CPU fallback

A `ja→en` run finally translated offline via NLLB end-to-end (tokenizer fix
landed: `saved tokenizer file … → using NMTTranslator … for ja→en`), but it
picked CPU and crawled: the device heuristic looked at *total* VRAM (`≤4 GB →
cpu`) instead of what's *free*. Translation runs after Whisper's VRAM is
released, so a 4 GB GTX 1650 has ~2.6 GB free and NLLB-600M int8 needs only
~1 GB. NMT_DEVICE=auto now decides on **free VRAM** (`_NLLB_CUDA_MIN_FREE_GB`,
1.8 GB) and uses the GPU when the model + workspace genuinely fit — and a CUDA
load error (OOM) now **retries on CPU** instead of dropping to the LLM, so cuda
is never fatal. Net: much faster offline translation on small cards, same safety.

---

# ClipAI — Run translation right after transcription/speakers (before summary)

Per request, subtitle translation now runs **immediately after transcription +
speaker assignment**, before the VLM summary and clip stages — instead of after
the summary. In `_run_analysis_inner` the translate+polish block was moved above
the summary block (new order: analyze → **translate+polish** → summary → clips →
post-clip finishers). It operates on the freshly transcribed, speaker-labelled,
deduped, music-marked transcript; the summary then runs on the polished source
transcript, and the clip-dependent finishers (caption refresh + Auto-SEO) still
run after clips. Progress stays monotonic (analysis 62% → translate 63% →
summary 70% → clips 80%); the summary is persisted right after it is generated so
post-clip Auto-SEO still sees it. Translation already ran before clips (so a
clip-stage failure can't skip it); this moves it earlier still.

---

# ClipAI — Offline NMT: save the SentencePiece tokenizer with the converted model

The torch.load fix let the NLLB convert succeed (log: `converted to int8 … kept
599.4 MB`), but translation *still* fell back to a stalling Ollama path. Cause:
CTranslate2's converter writes `model.bin` + the CT2 vocab but **not** the
SentencePiece tokenizer the runtime needs, so `NMTTranslator._model_files_present`
was False → `pick_local_engine("ja","en")` returned None → `NMT: no local model …
unsupported — caller will fall back to the LLM` (the `:free` model was correctly
skipped, then `qwen2.5:3b` limped through 59/78 segments with JSON parse errors).

Fix:
- The convert now fetches the tokenizer alongside the model — `sentencepiece.bpe.model`
  for NLLB, `source.spm`/`target.spm`/`vocab.json` for Opus-MT — reusing the HF
  snapshot the convert already pulled (a copy, not a re-download).
- `ensure_nllb_downloaded` **fails loud** if the tokenizer (or `ctranslate2` /
  `sentencepiece`) is still missing, instead of silently falling back to the LLM.
- **Repair shortcut:** an existing `model.bin` with no tokenizer (the broken
  state already on the user's `/data`) is fixed by fetching just the ~5 MB
  tokenizer — no 2.4 GB re-convert. So the next run self-heals and translates
  offline via NLLB (Ollama is never reached).

---

# ClipAI — Translation AI dropdown filter + translation-readability polish

Two follow-up requests after the offline-NMT fix:

**Translation AI dropdown — only models that can translate.** The dropdown was
fed the full text-model list, which (on OpenRouter's catalog) includes reasoning
/ "thinking" models whose chain-of-thought breaks the strict JSON-array the
batch translator parses (the `…-1.2b-thinking:free` the user hit) and
image/audio generators. `available_models()` now also returns a dedicated
`translation` list — text models minus those two classes (`_is_translation_capable`)
— and the Translation AI dropdown uses it. Against the live 343-model catalog
this excludes 39 (28 reasoning, 11 image/audio gens) and keeps every standard
instruct translator; ids that merely contain an `o` (e.g. `grok-2`, `claude-opus`)
are not false-matched by the o1/o3/o4 rule.

**Offline-first, OpenRouter polishes readability only.** Offline NMT already
translates first; the OpenRouter editorial model only polishes the result. That
polish now runs in a new **`mode="translation"`** profile: a readability-only
persona (`_SYSTEM_PROMPT_TRANSLATION`) that fixes punctuation / capitalisation /
spacing / obvious MT grammar glitches but **must not re-translate, change
meaning, reorder, merge/split, or change timing**. It replaces the old
ASR-correction framing (which would "phonetically correct" already-correct
translated words), and the tight length band + word-count guard are forced on so
the LLM can't drift into paraphrase. Timing and segment count were already
preserved (those fields aren't in the LLM's output); this protects *meaning* too.

---

# ClipAI — Offline translation reliability + transcription quality redesign (TACT)

A 24.5-min Japanese anime episode (Gundam Wing) with subtitle target = English
came back with: subtitles still in Japanese, the opening narration repeated at
six separated timestamps, hallucinated vocalisations over the music bed,
mistimed cues, and mangled song lyrics. Three stacked failures, fixed in
priority order. Full design rationale: `docs/redesign-transcription-translation.md`.

> Branch note: this work was developed on `claude/sharp-lamport-JBXmK` (the
> session's designated branch), which is identical to the task's stated base
> `fix/translate-step-reliability`.

## Task 1 (P0) — Fix the NMT convert bug so offline translation can succeed

**Root cause.** `nmt_translator._convert_with_cleanup` did
`os.makedirs(target_dir, exist_ok=True)` and then
`converter.convert(target_dir, quantization="int8", force=False)`. CTranslate2's
`TransformersConverter.convert` *refuses* a pre-existing output dir unless
`force=True` — and the `makedirs` had just created it. So **every** offline NMT
download raised, the cleanup ran only after the failure, and the caller fell
back to the LLM. Offline NMT had never succeeded on any branch.

**Fix.** Convert into a **fresh sibling temp dir** on the same volume (a path the
converter creates itself, so `force` is irrelevant), then promote it onto
`target_dir` with an atomic `os.replace` (clearing any stale/partial target
first). On any convert/download failure, remove the temp dir **and** any partial
`target_dir`, so a retry is never blocked by a stale directory and a half-written
model is never visible as "present". The HF-intermediate-cache redirect+cleanup,
the disk-space guard, and the Opus-MT pair cap are all retained.

**Network / disk.** The converter pulls NLLB-200 from `huggingface.co` (must be
reachable) — a ~2.5 GB transient full-precision download into a redirected HF
cache that is deleted afterwards, leaving the ~600 MB int8 CT2 model under
`/data/models/nllb/…`. Documented in the function docstring.

**Follow-up — second blocker found in the 2026-06-02 log.** With the dir bug
fixed, the convert reached model loading and hit a *different* deterministic
failure: `transformers ≥ 4.50` refuses `torch.load` of a `pytorch_model.bin` on
`torch < 2.6` (CVE-2025-32434) via `check_torch_load_is_safe()` — and NLLB-200 /
Opus-MT ship **only** `.bin` (no safetensors), while this repo pins
`torch==2.5.1+cu121` (torch 2.6 has **no** cu121 wheel, so bumping it would force
a whole cu121→cu124 CUDA-base migration). Result: the convert raised *“…require
users to upgrade torch to at least v2.6… does not apply when loading files with
safetensors”* and fell back to the LLM every time. Fix: `_allow_trusted_torch_load()`
temporarily neutralises that over-cautious version guard **only around our
convert** of trusted, HTTPS-fetched official HF checkpoints (loaded with
`weights_only=True` — exactly the case the guard itself skips when the caller
opts out), then restores it. No torch bump, no CUDA-base change; a no-op on
torch ≥ 2.6 and harmless if a future transformers renames the symbol.

## Task 2 (P0) — Offline NMT the guaranteed default; never grind a free model

- Offline-NMT-as-default was already correct in
  `translator._resolve_translation_engine` (auto → NLLB when no cloud keys +
  `NMT_AUTODOWNLOAD`); the convert bug was what stopped it from ever running.
- **`:free` model policy.** Before touching the orchestrator LLM, the router now
  detects whether the model the LLM path would actually call is a `:free`
  OpenRouter model (the translation override, else the orchestrator's active
  OpenRouter model, else settings). If so it **skips the OpenRouter LLM entirely**
  — no probe, no 10-minute 429 storm — and falls through to a local Ollama model
  if one is configured, otherwise raises `TranslationFailedError` **fast** with an
  actionable reason (use offline NMT / a paid or local model).
- The retained untranslated transcript is no longer logged as "clean"; it is the
  **source-language** transcript (deduped, explicitly *not* a translation). The
  fail-loud `translation_failed` status + reason are unchanged.

## Task 3 — Kill repetition-loop generation at the source (transcription)

`condition_on_previous_text=True` on the main Whisper pass drove the
repetition-loop pathology through the musical opening. New `_decoding_kwargs()`
helper (signature-filtered like `_vocab_bias_kwargs`, so an older faster-whisper
build never sees an unknown kwarg) now supplies, on the main + native-translate
passes:

- `condition_on_previous_text` defaulting **OFF** (`WHISPER_CONDITION_ON_PREVIOUS_TEXT`);
- `no_repeat_ngram_size=3`, `compression_ratio_threshold=2.4`,
  `log_prob_threshold=-1.0`, `repetition_penalty=1.1`, and a `temperature`
  fallback ladder — so degenerate/looped output is **rejected by the decoder**
  (re-decoded hotter) instead of emitted.

The gap-fill pass forces `condition_on_previous_text=False` regardless of the
global default (it lives over music/quiet regions where priming is harmful).

## Task 4 — Music-aware suppression + fuzzy hallucination quarantine

- **Music suppression.** `audio_analyzer.mark_and_suppress_music` classifies the
  audio **once**, drops Whisper "speech" cues that sit ≥ 60 %
  (`SUBTITLE_MUSIC_SUPPRESS_OVERLAP`) inside a sustained music-only span
  (≥ `SUBTITLE_MUSIC_MIN_SEC`) — the hallucinated lyrics/vocalisations — then
  positively labels the span `[♪ music ♪]`. BGM-under-dialogue is classified
  `speech` (not `music`) by the spectral classifier, so real dialogue over music
  is untouched. The pipeline now suppresses, then marks.
- **Fuzzy repetition quarantine.** `transcript_dedup.drop_repetition_loops` now
  catches near-identical long blocks (> 24 normalised chars), not just exact
  ones: (a) an **exact normalised-key** match with unbounded timeline reach —
  `_normalize_text` strips Unicode punctuation + whitespace, so the narration
  re-emitted at six separated timestamps with only punctuation/spacing
  differences collapses to one (the case the old exact-only filter missed); plus
  (b) a **length-gated fuzzy match (≥ 0.9)** over a bounded recent window for
  genuine ASR character drift. The length gate (`min/max ≥ 0.9`) means a
  distinct *longer* line that merely *contains* a shorter kept line is never
  dropped, and the window keeps the pass O(n) instead of O(n²) on long episodes.
  Short interjections keep the exact cap of 3; `[♪ music ♪]` markers are exempt.

## Task 5 — Otter-parity segmentation + glossary

- The dialogue-only source (markers held out) is passed through
  `sentence_segmenter.resegment_by_sentence` **before** translation, so the NMT
  sees clean one-utterance-per-cue units aligned to word timestamps rather than
  run-on blocks; the target side still resegments post-translation.
- Per-noun glossary enforcement already flows into the NMT path
  (`apply_glossary`) and is left in place.
- TACT heterogeneous-engine consensus (a Parakeet/Canary second pass) is left as
  documented future work — explicitly optional, VRAM-risky on the 1650, and not
  verifiable in this environment.

## New / changed settings

`WHISPER_CONDITION_ON_PREVIOUS_TEXT` (False), `WHISPER_NO_REPEAT_NGRAM_SIZE` (3),
`WHISPER_COMPRESSION_RATIO_THRESHOLD` (2.4), `WHISPER_LOG_PROB_THRESHOLD` (-1.0),
`WHISPER_REPETITION_PENALTY` (1.1), `WHISPER_TEMPERATURE_FALLBACK` (ladder),
`SUBTITLE_SUPPRESS_SPEECH_IN_MUSIC` (True), `SUBTITLE_MUSIC_SUPPRESS_OVERLAP` (0.6).

## Verification

**Done here** — 16 dependency-light unit tests
(`tests/test_transcription_translation_redesign.py`, all green):

- Convert fix with a converter mock faithful to CTranslate2's `force=False`
  refusal — the fresh-target case **fails against the old code, passes against
  the fix**; stale-target replacement; failure cleanup + retry-unblocked.
- `:free` detection + the fast-skip path: `text_completion` is **never called**
  (no 429 grind); fail-fast when no Ollama; fall-through to Ollama when set.
- `_decoding_kwargs` defaults (cond_prev OFF), signature-filtering for old
  builds, `**kwargs`-only safety, and the gap-fill override.
- Fuzzy quarantine: six near-identical narration copies → one, dialogue + 3×
  short interjection kept; music suppression: hallucinated lyrics dropped,
  marker + dialogue kept; source resegmentation: run-on → one-utterance cues.

**Must be run on the Unraid host** (this sandbox has no GPU, no
faster-whisper/CTranslate2 weights, and no media, so a full pipeline run +
fresh `clipai_logs_*.txt` cannot be produced here):

1. Build/deploy on the branch, preserving `/data` (so the NMT model persists)
   and `.env`. First `ja→en` job: expect log lines
   `NMT: neutralised transformers' torch<2.6 torch.load guard …`,
   `NMT: downloading + converting NLLB …`, `NMT: … converted to int8 at
   /data/models/nllb/…`, `NMT: using NMTTranslator (…) for ja→en`, and **no**
   `convert failed (… upgrade torch to at least v2.6 …)` and **no**
   `text_completion attempting via openrouter` for translation. Re-run: expect
   `NLLB … already downloaded` (no re-download, no convert error).
2. Re-run the Gundam Wing video (target=English) and confirm against the
   reference transcript: opening narration appears **once** at its real time; the
   song renders as `[♪ music ♪]` (no looped lyric fragments); no cues over
   silent/music-only spans; one clean utterance per cue. Capture the log + a
   side-by-side of the new output vs the reference.

---

# ClipAI — Make translate-then-polish actually run (decouple from clip extraction)

A user picked **English** as the translate-to language for a 24.5-min Japanese
video, but the subtitles came back in the original Japanese — untranslated and
unpolished. The pipeline log for job `5ee727f1` proved the target was captured
correctly (`detected_language=ja`, `subtitle_language=en`, plan logged as
`translate-then-polish in target language`) and that the source-language polish
was then **skipped** "in anticipation" of translation — but the translation it
deferred to **never ran**. Clip extraction hit repeated Replicate `429`
(`rate limit … less than $5.0 in credit`) on the VideoLLaMA3 calls, and because
the critical-path translate+polish call sat **after** the clip stage, the clip
failure bypassed it. The job still finalized `complete`, with the raw Japanese
transcript — the worst-case outcome (neither translated nor polished).

All changes on this branch (`fix/translate-step-reliability`).

## Root cause

In `backend/services/pipeline.py`, `_run_analysis_inner` logged the plan, **skipped**
the source-language polish/resegment/reflow when a translation was planned, ran
the clip-extraction stage, and only **then** invoked the critical-path
`await _background_post_processing(...)`. So any failure/early-return in the clip
region (the `429`s) reached the outer finalize before translation ever ran.
Meanwhile the source polish had already been skipped — so the transcript was
neither translated nor polished.

## Task 1 — Run translate-then-polish BEFORE clip extraction (decoupled)

- `_run_analysis_inner` now invokes `_background_post_processing(...)`
  **immediately after summary generation and before the clip-detection stage**.
  Subtitle translation no longer depends on clip detection succeeding.
- The clip-dependent work (caption refresh + Auto-SEO) was split out of
  `_background_post_processing` into a new module-level
  **`_run_post_clip_followups(...)`** that runs *after* the clip stage — so the
  parts that genuinely need clips still run once clips exist, while translation
  runs ahead of them.
- The clip stage is wrapped so a Replicate `429` (or any clipper exception)
  **degrades to zero clips and continues** — it can never abort or skip the
  remainder of `_run_analysis_inner`. The post-clip follow-ups are likewise
  best-effort and never abort the finalize.
- `_background_post_processing` is still **awaited on the critical path** (the
  translated+polished transcript is persisted before the COMPLETE save). The
  `_background_post_processing entry`, `Translation engine resolved`,
  `Translate START` log lines now fire on every translating job because the
  call is reached unconditionally.

## Task 2 — Never skip source polish unless translation actually runs

- `_background_post_processing` now **returns an outcome contract**
  (`will_translate`, `translated`, `source_transcript`, `target_transcript`,
  `seo_transcript`, `target_name`, `failed_reason`). `_run_analysis_inner`
  **adopts `source_transcript`** for the COMPLETE save, so the persisted
  `transcript` field is the *polished* source on the no-translation /
  translation-failed paths — never the raw text. (Previously the COMPLETE save
  re-persisted `_run_analysis_inner`'s own raw local `transcript`, silently
  clobbering the polished source the fallback had just written.)
- On translation failure the existing `_polish_source_if_needed()` +
  `_dedup_source_transcript()` fallback runs (polish + readability-enforce +
  dedup in the **source** language). A new **last-resort polish** in
  `_run_analysis_inner` covers the (near-impossible) case where
  `_background_post_processing` itself crashes — so there is **no code path
  where the source polish is skipped but translation does not run**. End state:
  the final transcript is always either (a) translated+polished in the target
  language, or (b) polished in the source language. Never raw + unpolished.

## Task 3 — Fail loud when a planned translation does not happen

- New `JobResult.translation_status` (`None` / `"translated"` /
  `"translation_failed"`) + `translation_error` (short reason) in
  `backend/models.py`, persisted on the job and surfaced over the existing
  websocket channel via the new `_set_translation_status(...)` helper
  (`{"type":"translation_status","state":...}`).
- `_background_post_processing` sets `"translated"` on success and
  `"translation_failed"` + reason on failure. `_run_analysis_inner` adds a
  **fail-loud backstop**: if a translation was *planned* (`subtitle_language` ≠
  source) but produced no target output, it sets `translation_failed` with a
  reason — a planned-but-missing translation can never silently present as a
  clean COMPLETE with source subtitles.
- Explicit call-site logging: `"[job] invoking translate+polish (target=…,
  source=…, will_translate=…)"` is logged immediately before the call, so the
  "it just didn't run" failure is visible in the log next time.

## Task 4 — Status clarity

- New `JobStatus.TRANSLATING = "translating"`. The translating progress update
  now uses this **distinct status** (at 77 %) instead of reusing
  `DETECTING_CLIPS` at 97 %, so a stuck/failed translation is no longer mistaken
  for clip detection in diagnostics. Wired through `PIPELINE_STAGES` (a new
  `translation` stage, 76–80 %), the heartbeat stage labels, the frontend
  `Dashboard` status badge + cancellable set, and the `PipelineTracker`
  (`STAGE_ORDER` / `STAGE_LABELS` / `STAGE_WEIGHTS` — the teal `translation`
  colour was already defined). Non-terminal, so `Analysis.jsx`'s `isProcessing`
  treats it correctly and the `_finalizing_jobs` gate still blocks only
  terminal reverts.

## Files changed

- `backend/models.py` — `JobStatus.TRANSLATING`; `translation_status` +
  `translation_error` fields.
- `backend/services/pipeline.py` — reorder translate before clips; split out
  `_run_post_clip_followups`; `_background_post_processing` returns an outcome
  contract, drops the premature `status=COMPLETE` pin, sets `translation_status`;
  `_set_translation_status` helper; fail-loud backstop + last-resort source
  polish; `_refresh_clips_with_translation` no longer pins COMPLETE (it now runs
  before finalization); `TRANSLATING` PIPELINE_STAGE + heartbeat label.
- `frontend/src/pages/Dashboard.jsx`, `frontend/src/components/PipelineTracker.jsx`
  — render the new `translating` status / `translation` stage.
- `tests/test_translate_step_reliability.py` — new regression tests.

## Verification

Run from the repo root in this environment (the heavy ML deps are lazy-imported,
so `backend.services.pipeline` imports with only `pydantic`/`httpx`/SDK shims):

- `python -m pytest tests/test_translate_step_reliability.py` → **5 passed**.
  Covers: translation success sets `translation_status="translated"` and
  **never** pins `status=COMPLETE`; a `TranslationRateLimitedError` (the `429`
  case) falls back to the polished **source** transcript, sets
  `translation_failed` + reason, and never relabels source as translated; the
  no-translation path returns the source for the COMPLETE save;
  `_run_post_clip_followups` refreshes captions **only** when a translation
  happened and always seeds Auto-SEO from the right transcript;
  `_set_translation_status` persists the field + broadcasts.
- `python -m py_compile backend/services/pipeline.py backend/models.py` → OK.
- `npx esbuild src/pages/Dashboard.jsx src/components/PipelineTracker.jsx
  --loader:.jsx=jsx` (from `frontend/`) → parses clean.
- `tests/test_job_persistence_race.py` → **6 passed** in isolation (no
  regression to the COMPLETE-finalize logic). The only suite failures are
  pre-existing and environment-only: tests importing `backend.services.object_detector`
  (a module that does not exist in this checkout) and `fastapi`/`cv2`/`torch`-backed
  modules that aren't installed here; plus the suite's pre-existing
  `HOME`-set-at-import cross-file ordering fragility (each affected file passes
  alone).

### Acceptance criteria → how to confirm on the live Unraid run

> The live Unraid build/deploy + the actual Japanese-video run (with a real
> Replicate `429` induced by low credit) was **not run from this environment** —
> there is no Unraid host, GPU, Replicate credential or test video here. Build
> and deploy with the standard nohup command on `fix/translate-step-reliability`
> (preserve `/data` and `.env`), then capture a fresh `clipai_logs_*.txt` and
> confirm:

1. **Translation runs to completion** — the log shows, in order:
   `Post-processing plan: source=ja target=en → translate-then-polish` →
   `invoking translate+polish (target=en, source=ja, will_translate=True)` →
   `_background_post_processing entry` → `Translation engine resolved: …` →
   `Translate START: Japanese → English (N segments)` → `Translate DONE` →
   `Polish START on translated text (lang=en)` →
   `Translated transcript readability: grade …` →
   `Persisting translated_transcript (N segments)` → then clip detection → COMPLETE.
2. **Clip failure does not block translation** — induce the Replicate `429`
   (low credit). The log shows `Clip extraction failed: …429…` **after** the
   `Translate DONE` / `Persisting translated_transcript` lines, the job
   completes with `clips=0` (degraded) and the **English** translated transcript
   intact. (Proven structurally + at the unit level by the tests above.)
3. **Translation genuinely can't run** → the transcript is still polished in the
   **source** language (`Source-language transcript polish (fallback path …)` /
   `Source transcript dedup …`), and the job carries a visible
   `translation_status="translation_failed"` with a reason (also broadcast as
   `{"type":"translation_status","state":"translation_failed"}`).
4. **No skip-without-translate window** — the source polish is skipped only when
   `_will_translate` is true, and on that branch `_background_post_processing`
   is always reached (translation runs, or its source fallback / the last-resort
   polish does).

### Before / after transcript (illustrative of the target transformation)

> Illustrative shape, not a capture from a live model run (no NMT/LLM weights or
> test video in this environment):

```
BEFORE (raw source, what the buggy path shipped):
  [00:03] 今日はいい天気ですね。
  [00:07] 散歩に行きましょう。

AFTER (translated + polished, target=en — what this fix ships):
  [00:03] It's such nice weather today.
  [00:07] Let's go for a walk.
```


# ClipAI — Whisper model: show the real one + let the user change it

Two Settings fixes for the Whisper transcription model: (1) the GUI now shows
the model that **actually loaded** (not just the configured value — the 4 GB
GTX 1650 can auto-downgrade), and (2) the read-only "Whisper Model" row is now a
**dropdown** whose pick is persisted, marked as a deliberate user choice, and
applied to the next transcription without a container restart.

All changes on this branch (`claude/inspiring-maxwell-GG90I`).

### Task 1 — Expose the effective (actually-loaded) Whisper model

- `backend/services/reframer_audio.py`: new **sticky** class attrs
  `AudioIntelligence._last_loaded_model_name` / `_last_loaded_device`, stamped by
  `_record_loaded(...)` at every real load (GPU tier success, base fallback,
  cache reuse, CPU reload). Unlike `_cached_model_name` — which
  `_release_whisper_vram()` nulls after a job — these survive the VRAM release so
  the GUI can always report what ran.
- `backend/routers/settings.py`: new helper `_effective_whisper_info()` +
  endpoint **`GET /api/providers/whisper/effective`** (and the same fields added
  to `GET /api/providers/models/available` → `current`). Returns:
  `whisper_model_selected`, `whisper_model_user_set`, `whisper_model_effective`
  (cached → sticky → `null`), `whisper_model_loaded`, and `whisper_downgraded`
  (`effective != selected`). Reads the `AudioIntelligence` class attrs out of
  `sys.modules` **without** importing the heavy reframer_audio module, so it's
  cheap to poll.

### Task 2 — Real dropdown instead of a read-only label

- `frontend/src/pages/Settings.jsx`: the Analysis-Settings "Whisper Model" row is
  now a `<select>` over the valid faster-whisper models (`tiny`, `base`, `small`,
  `medium`, `large-v3`, `large-v3-turbo`, `distil-small.en`, `distil-medium.en`,
  `distil-large-v3`, with 4 GB-oriented VRAM hints; `.en` variants labelled
  English-only). Below it, an effective-model line shows
  *"Selected: large-v3-turbo · Running: medium (downgraded — not enough VRAM…)"*
  when they differ, just *"Running: <model>"* when they match, "Not loaded yet"
  before the first job, and a ↻ refresh button. `backend/routers/settings.py`
  `_WHISPER_MODELS` gained the two distil `.en` entries (`english_only: true`).

### Task 3 — Persist the override and actually apply it

- `_perUserModelPatch('transcript')` now returns
  `{ WHISPER_MODEL: id, WHISPER_MODEL_USER_SET: true }`; the picker saves
  immediately (global `/providers/models/save` **and** the per-user patch).
- `backend/routers/auth.py`: added `WHISPER_MODEL_USER_SET` to the per-user
  `PUT /me/settings` allow-list so the flag actually persists.
- `backend/routers/settings.py` `save_models`: on a model change it now
  **invalidates** `AudioIntelligence` (new `invalidate_cache()` classmethod,
  sys.modules-guarded) so the next job loads the new selection without a restart
  (the old `reload_model()` was a no-op). The cache is also model-name-keyed, so
  a new pick reloads regardless.

### Task 4 — Honest downgrade behavior (and stop overwriting a user pick)

- `backend/main.py`: the startup Whisper auto-upgrade now skips when
  `WHISPER_MODEL_USER_SET` is true — a deliberate `small`/`medium`/`base` pick is
  no longer silently bumped to `large-v3-turbo` on a GPU box. Auto-upgrade still
  applies for the default (non-user-set) case.
- `reframer_audio.py` already records the real loaded model in `_cached_model_name`
  for both the downgrade and base-fallback paths; the new sticky fields mirror it
  for display so the GUI reflects reality.

### Tests

- `backend/tests/test_whisper_effective_endpoint.py` — effective/selected/
  downgraded/sticky reporting + the distil `.en` model list. (5 tests.)

```
$ python3 -m pytest backend/tests/test_whisper_effective_endpoint.py -q
5 passed
```

> The Settings dropdown + the GTX-1650 selected-vs-downgraded screenshot must be
> captured on the GPU box (no GPU/CUDA/faster-whisper here). The frontend JSX was
> verified to parse/transform with esbuild; the backend logic is unit-tested.

---

# ClipAI — Automatic + reliable offline translation, model lifecycle, Whisper fix

Make offline NMT translation the automatic default (no manual Settings download),
fail loudly instead of silently shipping the untranslated source, clean up model
downloads so disk/VRAM don't fill, honor the user-selected Whisper model, and
guarantee a clean, de-duplicated transcript even when translation fails.

All changes on this branch (`claude/inspiring-maxwell-GG90I`).

## New / changed settings (`backend/config.py`)

| Setting | Default | Purpose |
| --- | --- | --- |
| `NMT_AUTODOWNLOAD` | `True` | Auto-download + convert the offline NMT model the first time a translation needs it — no manual Settings step. The LLM is only used if the download itself fails. |
| `NMT_MAX_OPUS_PAIRS` | `5` | LRU cap on per-pair Opus-MT model dirs; least-recently-used pairs beyond the cap are pruned after a new download. NLLB-200 is a single model and is never counted against this cap. |
| `NMT_DEVICE` | `auto` (unchanged) | **Behavior change:** when `auto` and the GPU has ≤ 4 GB total VRAM (e.g. GTX 1650) the NMT engine resolves to **CPU** (OOM-safe on a card Whisper + Ollama already share). Larger GPUs keep CUDA. Set `cuda`/`cpu` to force. |

### Task 1 — Offline NMT auto-downloads on demand (no manual Settings step)

- `backend/services/translator.py`
  - `_resolve_translation_engine()`: with `TRANSLATION_ENGINE=auto` and **no cloud
    keys**, when no local model is on disk it now resolves to **`nllb`** (or
    `opus-mt` for pairs outside NLLB's Flores map) instead of falling straight to
    `llm`. The NMT path downloads it on first use; the LLM is the last resort only
    when `NMT_AUTODOWNLOAD` is off or the download fails.
  - `_translate_via_nmt()`: when no local engine exists and auto-download is on, it
    downloads + converts in a worker thread (`asyncio.to_thread(auto_download_for_pair,…)`),
    broadcasts **"Downloading offline translation model (one-time)…"** over the
    websocket `background_task` channel, re-resolves the local engine, and translates
    offline. Only a *download* failure falls back to the LLM, logged distinctly.
- `backend/services/nmt_translator.py`: new `auto_download_for_pair(src, tgt, prefer="nllb")`
  — NLLB-200 preferred (one model, 200 languages); Opus-MT only when explicitly
  requested or when the pair isn't in NLLB's Flores map.
- **faster-whisper** keeps its own on-first-use model download (unchanged).
- **pyannote diarization** is the only model needing a manual token (`HF_AUTH_TOKEN`);
  absent → `try_load()` returns `False` and the pipeline degrades to the existing
  spatial pseudo-diarization / mouth-motion ↔ audio-energy speaker mapping. Never
  blocks. (Documented; no code change.)

### Task 2 — No more silent fall-through to a rate-limited free model

- `backend/services/translator.py`: new `TranslationFailedError` /
  `TranslationRateLimitedError`. `translate_segments()` detects sustained 429s
  (`ProviderRateLimitError`) and **aborts after 2 rate-limited attempts** instead of
  crawling every batch; the counter **resets on success** so a recovered transient
  429 never trips it. A `:free` translation model logs a one-line warning (not blocked).
- `backend/services/ai_orchestrator.py`: `text_completion()` raises
  `ProviderRateLimitError` (not generic `AllProvidersFailedError`) when every provider
  failed with 429, so callers can fail fast + loud.
- `backend/services/pipeline.py`: translation failure now ends in a visible
  **`translation_failed`** state — a `pipeline_warnings` entry plus a `background_task`
  `failed` broadcast (`state: "translation_failed"`) with an actionable message. The
  source is never persisted under `translated_transcript` when translation didn't
  complete.

### Task 3 — Model lifecycle: download, cache, clean up (`nmt_translator.py`)

- **HF intermediate-cache cleanup:** `_hf_cache_redirect()` points HF cache env at a
  throwaway dir on the **same volume** as the models for the convert, then deletes it
  — only the ~600 MB int8 CT2 model survives. **Bytes reclaimed are logged.**
- **Disk-space guard** before download (~6 GB NLLB, ~3 GB per Opus pair) → clear failure
  instead of a half-written dir.
- **Partial-convert cleanup:** target dir removed if `convert()` raises (clean retry).
- **Opus-MT LRU cap** (`NMT_MAX_OPUS_PAIRS`, `.last_used` markers).
- **VRAM:** `NMT_DEVICE=auto` → CPU on ≤ 4 GB GPUs; confirmed (already correct) that
  `NMTTranslator.unload()` + `empty_cache()` run in a `finally` after translation,
  `_release_whisper_vram()` runs before the NMT load, and NMT is unloaded before polish.

### Task 4 — Honor the user-selected Whisper model (`large-v3-turbo` → `medium` mismatch)

- `reframer_audio.py`: `AudioIntelligence` records `requested_model_name`, logs the
  requested model up front, honors it (falls to **CPU** rather than swapping size when
  GPU VRAM is short — logged), and logs the genuine last-resort change as an explicit
  **`DOWNGRADE: requested <model> … → loading base on CPU`** with free-VRAM numbers.
- `reframer_engine.py` + `pipeline.py`: the **effective** loaded model is surfaced in
  the compute summary (`summary["whisper"]["model"]`/`requested_model`) so the active
  config can't disagree with what ran.
- `settings_overlay.py`: a per-user `WHISPER_MODEL` that differs from the global pin is
  logged loudly (`global=… → per-user=…`). `WHISPER_AUTO_UPGRADE` (non-user-set)
  unchanged.

### Task 5 — Clean timing/dedup even when translation fails (`pipeline.py`)

- New `_dedup_source_transcript()` runs `collapse_adjacent_duplicates` →
  `collapse_overlapping_duplicates` → `drop_repetition_loops` on the **source**
  transcript whenever translation is skipped or fails, mirroring the post-translation
  dedup, and persists the cleaned transcript (removed-count logged). Verified
  `collapse_overlapping_duplicates` collapses identical / near-identical-timestamp
  duplicates (the `[6:23]`/`[6:23]`, `[2:30]`/`[2:30]` cases).

### Tests (dependency-light — no ctranslate2 / torch / GPU)

- `backend/tests/test_nmt_autodownload_and_cleanup.py` — disk guard, HF-cache
  redirect + cleanup, partial-convert cleanup, Opus-MT LRU cap, `auto_download_for_pair`
  routing.
- `backend/tests/test_translation_fail_loud.py` — rate-limit detection, exception
  hierarchy, fast abort on sustained 429s, no-abort on a recovered transient 429.

```
$ python3 -m pytest backend/tests/test_nmt_autodownload_and_cleanup.py \
                    backend/tests/test_translation_fail_loud.py -q
12 passed
```

> The Unraid build/deploy + the two Japanese-video runs (empty `/data/models`, then
> cached) must be run on the GPU box — this dev container has no GPU / CUDA /
> ctranslate2 / faster-whisper / `/data`, so the heavy ML paths can't run here. The
> logic above is unit-tested; the six acceptance criteria are confirmable from a fresh
> `clipai_logs_*.txt` on the box.

---

# ClipAI — Transcription / Translation / Polish fixes

Branch base: `claude/otter-transcription-parity-mTjAh` (identical commit to the
session branch this was developed on). All changes are surgical and leave the
reframer camera / clip pipeline untouched.

## New config key

| Key | Default | Meaning |
| --- | --- | --- |
| `OPENROUTER_TRANSLATION_MODEL` | `""` (blank) | Dedicated OpenRouter model for **subtitle translation**. Blank → falls back to `OPENROUTER_EDITORIAL_MODEL`. Mirrors the existing `OLLAMA_TRANSLATION_MODEL`. |
| `WHISPER_GAP_FILL_MAX_SEC` | `8.0` | Hard ceiling on a single gap-fill segment's length. Longer cues are split at word boundaries so run-on music/narration can't collapse into one timestamp. `0` disables the split. |

---

## Task 1 — Translate, then polish, in the target language (core fix)

**`backend/services/pipeline.py`**

- **`_run_analysis_inner`** now decides up-front whether a translation will
  follow (`_will_translate`, computed from `job.subtitle_language` /
  `job.language` / Whisper's detected language, including the
  "auto-translate non-English → English" default). When it will translate, the
  **source-language** critical-path work is skipped — the heavy LLM polish, the
  sentence resegmentation and the readability reflow no longer run on the
  Japanese text. Music marking + dedup + the readability score still run, so the
  source artifact is still clean. When no translation is needed, behaviour is
  unchanged (full source polish, as before).
- **`_background_post_processing`** was reordered to **translate → polish →
  resegment → reflow → dedup**:
  - (a) translate the source segments to the target language,
  - (b) LLM polish on the **translated** text with `correction_lang = target_lang`
    (the old `_whisper_translated` assumption — that Whisper had already
    translated — was removed; the reframer always runs `task="transcribe"`),
  - (c) sentence resegmentation + readability reflow in the target language,
  - (d) final dedup (adjacent + overlap + repetition-loop),
  - then persist `translated_transcript` + `transcript_readability`, refresh clip
    captions, broadcast the `background_task` websocket events, and seed SEO.
  - If translation produces **0 changed segments** (or throws), it bails to the
    source path and keeps the source transcript — it never persists Japanese
    under `translated_transcript`.
- **Reliability:** the step is no longer a fire-and-forget `asyncio.create_task`
  scheduled after COMPLETE (which was being lost on this deployment — the job
  stalled at `detecting_clips` with the Japanese transcript and zero translation
  in the log). It is now **awaited on the critical path, before the COMPLETE
  save**, so the translated + polished transcript is part of the finalize. The
  clips are reloaded afterwards so the single COMPLETE save carries the
  translated captions; `translated_transcript` / `transcript_readability` are
  preserved by `_persist_complete_job`'s load→merge.
- Explicit `logger.info` lines (all tagged with `job_id`) now mark: entry,
  before/after translate (with the model used), before/after polish (`lang=…`),
  resegment/readability/dedup counts, and the final persist.

## Task 2 — Gap-fill timing + duplicate fixes

**`backend/services/reframer_audio.py`**

- `_gap_fill_pass` now **splits over-long cues** (> `WHISPER_GAP_FILL_MAX_SEC`,
  default 8 s) at word boundaries — and at any inter-word silence ≥ 1 s — so a
  whole OP song / minutes of narration can't collapse into one run-on segment.
  Each emitted sub-cue carries correct **absolute** start/end times.
- The `_inside_gap` overlap guard is applied on **both** the `clip_timestamps`
  path and the full-file fallback path (segments outside a real gap are dropped
  and counted).
- A summary line logs **segments kept / dropped-outside-gap / dropped-as-
  duplicate / max segment length**.
- `_cross_validate_segments` now runs a second, **non-adjacent** collapse pass.

**`backend/services/transcript_dedup.py`**

- New `collapse_overlapping_duplicates()` removes near-duplicates by **timestamp
  overlap + text similarity** (whitespace-insensitive exact, containment, word-
  Jaccard, or CJK char-bigram), not just adjacency — collapsing the repeated
  `作戦名オペレーション・メテオ`-style re-transcriptions. Used in the gap-fill pass,
  `_cross_validate_segments`, the pipeline's final transcript dedup, and the
  post-translation dedup. The existing `collapse_adjacent_duplicates` /
  `drop_repetition_loops` are unchanged (their tests still pass).

## Task 3 — Source-language dropdown → Whisper

**`backend/services/pipeline.py`**

- `ReframeEngine(...)` is now constructed with
  `source_language=(job.language or "auto")`, so the user's upload-page language
  pick reaches `AudioIntelligence.transcribe(language=…)` and the log shows
  `language=ja` (or whatever was picked) instead of always `auto`. A log line
  confirms both the source language and the `subtitle_language` target.

## Task 4 — Dedicated OpenRouter translation model + dropdown

- **`backend/config.py`** — added `OPENROUTER_TRANSLATION_MODEL` (blank → editorial).
- **`backend/services/ai_orchestrator.py`** — `text_completion(...)` takes an
  optional `model_override`; when set and the active provider is OpenRouter, it
  temporarily swaps `provider._editorial_model` (restored in `finally`), mirroring
  the Ollama override pattern.
- **`backend/services/translator.py`** — `translate_segments` accepts
  `model_override`; `translate_segments_with_fallback` passes
  `settings.OPENROUTER_TRANSLATION_MODEL or None` into the probe + full LLM
  translation calls and logs the model used. Polishing keeps the editorial model.
- **`backend/routers/settings.py`** — `OPENROUTER_TRANSLATION_MODEL` added to the
  persisted-keys list, the `SaveModelsRequest` (`translation_model`), the
  `/providers/models/save` writer (+ `.env` upsert) and the
  `/providers/models/available` + save `current` blocks.
- **`backend/app/auth/settings_overlay.py`** + **`backend/routers/auth.py`** —
  `OPENROUTER_TRANSLATION_MODEL` added to the per-user overlay / PUT allow-lists.
- **`frontend/src/pages/Settings.jsx`** — new **"Translation AI (OpenRouter)"**
  dropdown under the Editorial AI controls, populated from the OpenRouter
  catalog, wired through `pendingModels.translation_model` /
  `currentModels.translation_model` with the same buffer-then-save pattern and
  POSTed to `OPENROUTER_TRANSLATION_MODEL`. The **"Editorial AI"** control was
  relabelled/clarified as the **transcript-polishing** model so the two are
  clearly distinct.

## Task 5 — Sanity

- `TRANSLATION_ENGINE='auto'` with only OpenRouter configured still resolves to
  the `llm` engine (`_resolve_translation_engine`), and that path now honours the
  translation-model override.
- No `HF_TOKEN` requirement was added; reframer camera/clip logic is untouched.

---

## Verification

- All changed Python modules compile (`python -m py_compile`).
- `frontend/src/pages/Settings.jsx` transpiles cleanly via `esbuild`.
- The existing `transcript_dedup` regression cases pass, and the new
  `collapse_overlapping_duplicates` + gap-fill split logic were unit-checked
  (run-on cues split to ≤ 8 s with accurate times; overlapping CJK/Latin
  near-duplicates collapsed; non-overlapping distinct lines untouched).

> Note: the live Unraid build/deploy + fresh `clipai_logs_*.txt` capture was
> not run from this environment (no access to the Unraid host). The acceptance
> criteria are satisfied by the code paths above; run the standard nohup build
> on the branch and upload the Japanese test video to capture the before/after
> transcript + the `language=ja → translate ja→en → polish lang=en → persisted
> COMPLETE` log sequence.

---

# Offline NMT translation — finish & verify NLLB-200 + Opus-MT (CTranslate2)

Branch: `claude/clipai-offline-nmt-GsHqe`. Finishes wiring the **offline**
neural-MT path that already existed so subtitle translation can run fully
local (no OpenRouter/LLM call) on the Unraid GTX 1650 box. No new engine was
added and the reframer camera/clip pipeline is untouched. This builds on the
translate→polish reordering above — the offline engine plugs in at the existing
`translate_segments_with_fallback` call, so the flow stays **translate (now
possibly offline NMT) → LLM polish in target language → resegment/readability →
dedup**. Transcript *polishing* still uses the editorial LLM; only the
*translation* step changes engine.

## New config key

| Key | Default | Meaning |
| --- | --- | --- |
| `NMT_DEVICE` | `auto` | Device policy for local NMT (`auto` \| `cpu` \| `cuda`). `auto` → CUDA when available (int8_float16), else CPU (int8). **`cpu` is the safe choice on 4 GB GPUs** (GTX 1650) where Whisper + Ollama already compete for VRAM — a 24-min video still translates in ~a couple of minutes on CPU and can't OOM. Opus-MT always runs on CPU regardless. |

## New dependency

- **`transformers>=4.40,<5`** added to `backend/requirements.txt`. It powers
  `ctranslate2.converters.TransformersConverter`, which the Settings
  "Download model" button uses to fetch + convert NLLB / Opus-MT to the int8
  CTranslate2 format. Without it the download endpoint raised
  `RuntimeError("ctranslate2 with the transformers converter is required …")`.
- **`ctranslate2>=4.0,<5`** now pinned explicitly (was pulled in transitively by
  faster-whisper) so the int8 inference + converter are guaranteed present.
- The conversion is a **one-time, on-demand download — NEVER at startup**. The
  converted int8 model is cached under the models dir (Docker `/data/models`,
  i.e. `/data/models/nllb/...` and `/data/models/opus-mt/{src}-{tgt}/...`) so it
  survives a container rebuild as long as `/data` is preserved.

## Files touched

- **`backend/requirements.txt`** — added `transformers>=4.40,<5`, explicit
  `ctranslate2>=4.0,<5` pin (kept `sentencepiece`). A fresh
  `pip install -r backend/requirements.txt` now imports
  `from ctranslate2.converters import TransformersConverter` cleanly.
- **`backend/config.py`** — new `NMT_DEVICE` setting (`auto|cpu|cuda`, default
  `auto`) with a doc comment about the 4 GB-GPU CPU recommendation.
- **`backend/services/nmt_translator.py`** —
  - `NMTTranslator.__init__` now reads `NMT_DEVICE` from settings (explicit
    `device=` arg still wins) instead of hard-coding `"auto"`.
  - `NMTTranslator.load()` logs a **"Whisper VRAM freed before NLLB load — N MB
    free"** line (+ a defensive `empty_cache()`) when resolving to CUDA, so the
    OOM-avoidance ordering is visible in the log. Keeps `int8_float16` on CUDA /
    `int8` on CPU. Opus-MT still forces CPU (unchanged). `unload()` +
    `torch.cuda.empty_cache()` in the `finally` block is unchanged.
- **`backend/services/translator.py`** —
  - `_resolve_translation_engine` now logs **"NMT: no local model for ja→en,
    falling back to LLM"** when `auto` finds no downloaded model (so it's obvious
    why the offline path didn't run), and logs probe failures.
  - `_translate_via_nmt` now names the **exact engine + model** per job
    (`NMT: using NMTTranslator (facebook/nllb-200-distilled-600M) for ja→en …`
    or `OpusMTTranslator (Helsinki-NLP/opus-mt-ja-en) …`) and logs the no-model
    fall-through.
- **`backend/services/pipeline.py`** — before the `translate_segments_with_fallback`
  call in `_background_post_processing`, when the resolved engine is a local NMT
  (`nllb`/`opus-mt`) it re-runs `_release_whisper_vram(job_id)` defensively and
  logs **"Local NMT engine '…' selected — freeing Whisper VRAM before NMT
  load"**. (Whisper VRAM is already freed during analysis at the post-reframer
  stage; this makes the ordering explicit and robust to future reordering.)
- **`backend/routers/settings.py`** — `NMT_DEVICE` added to the persisted-keys
  list, to `_subtitle_quality_state()` (`nmt_device`), to
  `SaveSubtitleQualityRequest`, and validated against `{auto,cpu,cuda}` in the
  save handler. (`TRANSLATION_ENGINE` was already persisted + saved; the
  `/api/translation/download-model` route already matches the frontend path —
  `router` prefix `/api` + `/translation/download-model`.)
- **`frontend/src/components/SubtitleQualitySettings.jsx`** —
  - The single hard-coded `downloadNLLB()` (always `{engine:'nllb'}`) is replaced
    by an engine-aware `downloadModel(engine)`: **Download Opus-MT (src→tgt)**
    passes `{engine:'opus-mt', source, target}`, **Download NLLB-200** passes
    `{engine:'nllb'}`. Both surface the endpoint's `path`/`message` (success +
    error) instead of failing silently.
  - Added two ISO-code language `<select>`s (the same codes as the Upload page)
    to pick the Opus-MT pair to pre-download. **Default target = `en`**, default
    source = `ja`. The Opus-MT button is disabled when source == target.
  - Added a **Local NMT device** `<select>` (`auto|cpu|cuda`) wired to
    `nmt_device` → `NMT_DEVICE`, persisted with the rest of the form.

## Offline-translation usage (download → select engine → run)

1. **Download a model** (one-time): Settings → Subtitle Quality → Translation
   Engine. For NLLB click **Download NLLB-200 (~600 MB)**. For Opus-MT pick the
   source→target pair (e.g. Japanese → English) and click **Download Opus-MT**.
   The button shows `… ready ✓ (/data/models/…)` on success.
2. **Select the engine**: set **Engine** to `nllb`, `opus-mt`, or `auto`
   (auto prefers Opus-MT, then NLLB, before the LLM). Optionally set **Local NMT
   device** to `cpu` on a 4 GB GPU. Click **Save** — `TRANSLATION_ENGINE` +
   `NMT_DEVICE` persist to `user_settings.json` and survive a container restart.
3. **Run** the Japanese test video with source=Japanese, target=English. With a
   model downloaded the log shows `NMT: using NMTTranslator (…nllb…) for ja→en`
   (or `OpusMTTranslator (…opus-mt-ja-en…)`) — **no OpenRouter/LLM call for
   translation** — followed by the LLM polish pass on the English text. With no
   model downloaded, `auto` logs `NMT: no local model for ja→en, falling back to
   LLM` and uses the LLM path as before.

## Verification

- All changed Python modules compile (`python -m py_compile`).
- The download endpoint returns `{status:"ok", path:…}` for both `nllb` and an
  `opus-mt` ja→en pair once `transformers` + `ctranslate2` are installed; on a
  stock container (before this change) it returned the converter `RuntimeError`.

> Note: the live Unraid build/deploy + `pip install` + fresh `clipai_logs_*.txt`
> capture (pre-download NLLB from Settings, run the Japanese test video offline)
> was not run from this environment (no access to the Unraid host). Run the
> standard nohup build on `claude/clipai-offline-nmt-GsHqe`, preserving `/data`
> (so downloaded NMT models persist) and `.env`, then capture the
> `NMT: using …` / `Whisper VRAM freed before NMT load` / `Polish START on
> translated text (lang=en)` log sequence and the before/after transcript.
