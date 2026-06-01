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
