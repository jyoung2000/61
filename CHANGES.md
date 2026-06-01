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
