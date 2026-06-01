# Translate → polish ordering, gap-fill timing, source-language dropdown, OpenRouter translation model

This branch — `fix/translate-polish-order-and-model-dropdowns` — fixes the
ClipAI subtitle pipeline so non-English audio (Japanese, Korean, Chinese, …)
actually ships English subtitles when the user picks English on the upload
page, and so the polish + readability passes run on the language viewers
will read instead of on the source-language transcript.

## Task 1 — Translate, then polish, in the target language

When a target subtitle language differs from the detected source, the
critical-path polish + sentence resegmentation + readability enforcement
in `_run_analysis_inner` is now **skipped** and the heavy work runs in
the background **after** translation. The previous flow polished
Japanese on the critical path, translated to English, and never polished
the English — which is why the OP "transcript came out Japanese" and
the polish ran on text the user was about to throw away.

`_background_post_processing` now runs in this order whenever
translation applies:

1. Translate source → target via the orchestrator chain.
2. **LLM polish on the translated text** (`correction_lang = target_lang`).
3. Sentence resegmentation in the target language.
4. Readability enforcement in the target language.
5. Final dedup (adjacent + repetition-loop + overlap-aware).
6. Persist `translated_transcript` + readability + reassert `COMPLETE`.

The bg-task scheduling now logs entry, exit, and any exception via
`add_done_callback`, so a crash inside the bg coroutine surfaces in the
log instead of being swallowed by the task GC. The previous code
relied on `_background_tasks.discard` alone, which let a raised
exception disappear silently — the "job stalled at detecting_clips"
symptom in `clipai_logs_dbc7b25b`.

**Files touched:** `backend/services/pipeline.py`

## Task 2 — Fix gap-fill timing and duplicates

* `_gap_fill_pass` keeps **VAD on** by default (`WHISPER_GAP_FILL_VAD=true`)
  so a 700 s music region can no longer collapse into one 707 s
  run-on segment.
* Every emitted gap-fill segment is now bounded by
  `WHISPER_GAP_FILL_MAX_SEG_SEC` (default 8 s). When Whisper hands us
  anything longer, it's split at word boundaries (or uniformly when no
  word grid is available) so each cue stays inside the readability cap.
* A new in-pass duplicate guard catches the same line emitted twice in
  one gap run (CJK gap-fill over music often produces this).
* A new `collapse_overlapping_duplicates` helper in
  `backend/services/transcript_dedup.py` catches **non-adjacent** dups
  that share a timestamp window — the
  `作戦名オペレーション・メテオ` pair the old adjacent-only filter missed.
  Normalization strips both ASCII and CJK punctuation so
  `'作戦名 オペレーション メテオ'` and `'作戦名 オペレーション・メテオ'`
  collapse correctly.
* `_cross_validate_segments` chains its existing adjacent pass through
  the new overlap-aware pass and logs counts removed.
* Final pipeline dedup also runs the overlap pass on the assembled
  transcript before persist, plus a mirror dedup on the polished
  translated transcript.
* `_gap_fill_pass` now logs a one-line summary: kept count,
  dropped-outside-gap, dropped-as-duplicate, hallucination /
  no-speech / repetition drops, long runs split, and the maximum
  emitted segment length.

**Files touched:** `backend/services/reframer_audio.py`,
`backend/services/transcript_dedup.py`, `backend/services/pipeline.py`,
`backend/config.py` (new defaults `WHISPER_GAP_FILL_VAD`,
`WHISPER_GAP_FILL_MAX_SEG_SEC`).

## Task 3 — Thread the source-language dropdown into Whisper

`ReframeEngine(...)` is now constructed with
`source_language=(job.language or "auto")` so the upload page's
language dropdown actually reaches Whisper. Previously the engine
defaulted to `"auto"` regardless of the user's pick. A log line at
construction time records the resolved value next to `job.language`
and `job.subtitle_language` so the language plumbing is visible in
the standard log export.

**Files touched:** `backend/services/pipeline.py`.

## Task 4 — Dedicated OpenRouter translation model + dropdown

* **New config key `OPENROUTER_TRANSLATION_MODEL`** in `backend/config.py`.
  Blank ⇒ fall back to `OPENROUTER_EDITORIAL_MODEL` at translate time
  (preserves the legacy behaviour for upgrades).
* `AIOrchestrator.text_completion(...)` accepts an optional
  `model_override: str | None`. When set AND the active provider is
  OpenRouter, the provider's `_editorial_model` is temporarily swapped
  in `try` / restored in `finally`. Mirrors the existing Ollama
  downgrade-override pattern.
* `translator.translate_segments(...)` reads
  `settings.OPENROUTER_TRANSLATION_MODEL` and passes it as
  `model_override` on every batch, so polishing keeps the editorial
  model while translation routes through the user's choice. The
  translation model used is logged on the first batch.
* `backend/routers/settings.py` persists `OPENROUTER_TRANSLATION_MODEL`
  via `_PERSISTABLE_KEYS` (survives container restarts via
  `/data/logs/user_settings.json`). The `/api/providers/models/save`
  endpoint accepts a new `translation_model` field with `Optional[str]`
  sentinel — `None` ⇒ "not touched", `""` ⇒ "explicitly cleared",
  any other string ⇒ that model. The
  `/api/providers/models/available` GET response surfaces the saved
  pick under `current.translation_model`.
* Per-user settings overlay (`backend/app/auth/settings_overlay.py`)
  and the `/api/auth/me/settings` PUT allow-list both include
  `OPENROUTER_TRANSLATION_MODEL` so the per-user model picks survive
  the same.
* `frontend/src/pages/Settings.jsx`:
  * New "Translation AI (OpenRouter)" `ModelDropdown` under the
    Editorial AI Fallback, populated from `availableModels.editorial`.
  * Re-labels Editorial AI as
    **"Editorial AI (transcript polishing)"** with a description
    that makes the polish / translate split explicit.
  * Buffer-then-save state pattern matches the other model controls:
    `pendingModels.translation_model` /
    `currentModels.translation_model`, included in
    `modelsHaveChanges` and `handleSaveAllModels`. Blank ⇒ clears
    the per-user override too.

## Task 5 — Regression guards

`TRANSLATION_ENGINE='auto'` still resolves to the `llm` engine when
only OpenRouter is configured (`_resolve_translation_engine` →
falls through DeepL/Google/NMT to LLM). The orchestrator route
now consults the new override; no changes to the existing decision
tree. No changes to reframer camera/clip logic. No requirement on
`HF_TOKEN` introduced.

## New config keys

```
OPENROUTER_TRANSLATION_MODEL: str = ""        # dedicated translation model
WHISPER_GAP_FILL_VAD: bool = True              # VAD on for the gap pass
WHISPER_GAP_FILL_MAX_SEG_SEC: float = 8.0      # split runs longer than this
```

## Files touched

* `backend/config.py`
* `backend/services/ai_orchestrator.py`
* `backend/services/pipeline.py`
* `backend/services/reframer_audio.py`
* `backend/services/transcript_dedup.py`
* `backend/services/translator.py`
* `backend/app/auth/settings_overlay.py`
* `backend/routers/auth.py`
* `backend/routers/jobs.py`
* `backend/routers/settings.py`
* `frontend/src/pages/Settings.jsx`
