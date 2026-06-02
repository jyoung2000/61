# Redesign: offline translation reliability + transcription quality (TACT)

Implementation notes for the stacked fix covering offline NMT translation and
music-heavy transcription quality. Written **before** the code changes so the
intent, the confirmed root causes, and the chosen approach are on the record.

Context: ClipAI on Unraid, GTX 1650 (4 GB VRAM). A 24.5-min Japanese anime
episode (Gundam Wing) with subtitle target = English came back with: subtitles
still in Japanese, the opening narration block repeated at six widely-separated
timestamps, hallucinated vocalisations over the music bed, mistimed cues, and
mangled song lyrics. Three stacked failures, fixed in priority order.

## Confirmed root causes (verified against the code on this branch)

1. **Offline NMT is bricked by a deterministic convert bug.**
   `nmt_translator._convert_with_cleanup` did
   `os.makedirs(target_dir, exist_ok=True)` and then
   `converter.convert(target_dir, quantization="int8", force=False)`.
   CTranslate2's `TransformersConverter.convert` *refuses* a pre-existing
   output directory unless `force=True` — and the `makedirs` had just created
   it. So the convert raised every time, the `rmtree` cleanup ran only after
   the failure, and the caller fell back to the LLM. Offline NMT therefore
   never succeeded on any branch.

2. **The LLM fallback grinds on a free, rate-limited model.**
   The configured editorial fallback / translation override is a `:free`
   OpenRouter model (`qwen/qwen3-next-80b-a3b-instruct:free`). Free endpoints
   are rate-limited upstream, so the LLM path spent ~10 min in 429s before
   surfacing `translation_failed`. A `:free` warning existed but was passive.

3. **The source transcription is degenerate on music-heavy content.**
   The main Whisper pass ran with `condition_on_previous_text=True`, which
   drives Whisper's repetition-loop pathology through the musical opening (the
   narration re-emits at 0:48 / 1:31 / 2:48 / 5:50 / 7:25 / 9:51). The
   repetition-loop filter was exact-text only, so near-identical copies
   survived. Whisper also hallucinated lyrics/vocalisations over the music bed,
   which the CoverageLedger quarantine didn't catch (plausible text + low
   `no_speech_prob`).

## What already existed on this branch (kept, not rebuilt)

- Offline-NMT-as-default routing in `translator._resolve_translation_engine`
  (auto → NLLB when no cloud keys + autodownload). The convert bug is what
  stopped it from ever working; once fixed, this routing is correct.
- Fail-loud `translation_status` (`translated` / `translation_failed`) with a
  persisted reason; the untranslated result is kept as the source transcript.
- Music marking (`audio_analyzer.detect_music_markers` / `merge_markers`) that
  inserts `[♪ music ♪]` cues into speech gaps.
- `transcript_dedup`: exact `drop_repetition_loops`, `collapse_adjacent_*`, and
  fuzzy `collapse_overlapping_duplicates` (overlap + similarity).
- Sentence-aware, word-timestamp-aligned resegmentation
  (`sentence_segmenter.resegment_by_sentence`), run post-translation.
- Per-job glossary loading + NMT `apply_glossary` post-processing.
- Gap-fill pass already uses `condition_on_previous_text=False`.

## Changes (by task)

### Task 1 — Fix the NMT convert bug (P0)
`_convert_with_cleanup`: convert into a **fresh temp dir** on the same volume,
then atomically `os.replace` it onto `target_dir` on success (clearing any
stale/partial `target_dir` first). On any failure, remove the temp dir **and**
any partial `target_dir` so a retry is never blocked by a stale directory.
A half-written model is never visible as "present". HF-cache redirect, disk
guard, and Opus-pair cap are retained. Network need (huggingface.co) and the
~2.5 GB transient HF download → ~600 MB int8 residual are documented in the
docstring.

### Task 2 — Offline NMT the guaranteed default; never grind a free model (P0)
- Offline-default routing verified (see above).
- New `:free` policy in `translate_segments_with_fallback`: before touching the
  orchestrator LLM, detect whether the model the LLM path would actually call
  is a `:free` OpenRouter model (translation override, else the orchestrator's
  active OpenRouter model, else settings). If so, **skip the OpenRouter LLM
  entirely** — fall through to the local Ollama translation model if one is
  configured, otherwise raise `TranslationFailedError` fast with an actionable
  reason (recommend offline NMT or a paid/local model). No 429 storm.
- Stop calling the retained untranslated transcript "clean" — it is the
  *source-language* transcript; relabel the log line accordingly.

### Task 3 — Kill repetition-loop generation at the source
- `condition_on_previous_text` now defaults to **False** for transcription
  (new `WHISPER_CONDITION_ON_PREVIOUS_TEXT` setting), applied to the main pass
  and the range/clip pass; the gap-fill pass already had it off.
- Anti-repetition decoding params added via a signature-introspecting helper
  (`_decoding_kwargs`, mirroring `_vocab_bias_kwargs`) so we only pass kwargs
  the installed faster-whisper build accepts: `no_repeat_ngram_size=3`,
  `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
  `repetition_penalty`, and a `temperature` fallback ladder. Degenerate/looped
  output is rejected by the decoder instead of emitted.

### Task 4 — Music-aware suppression + fuzzy hallucination quarantine
- `audio_analyzer.suppress_speech_in_music_spans`: in sustained music-only
  spans (≥ `SUBTITLE_MUSIC_MIN_SEC`), drop Whisper "speech" segments that sit
  ≥ `SUBTITLE_MUSIC_SUPPRESS_OVERLAP` (0.6) inside the span — these are the
  hallucinated lyrics/vocalisations. The span is positively labelled with the
  `[♪ music ♪]` marker instead. BGM-under-dialogue is unaffected because the
  spectral classifier labels those windows "speech", not "music". The pipeline
  classifies events once, suppresses, then marks.
- `transcript_dedup.drop_repetition_loops` extended: long blocks (>24 norm
  chars) are now clustered by **similarity ≥ 0.9** (not exact), so a
  near-identical narration block recurring at separated timestamps keeps only
  the first occurrence and the rest are dropped (the six-timestamp loop).
  `_text_similarity` strips punctuation before the CJK bigram compare. Short
  interjections keep the exact-match cap of 3.

### Task 5 — Otter-parity segmentation + glossary
- Resegment the **source** (dialogue-only, markers held out) by sentence with
  word-timestamp alignment *before* translation, so NMT sees clean
  one-utterance units; the target side already resegments post-translation.
- Glossary (proper nouns) already flows into the NMT path; left as-is.
- TACT heterogeneous-engine consensus (Parakeet/Canary second pass) is left as
  documented future work — explicitly optional, VRAM-risky on the 1650, and not
  verifiable in this environment.

## Verification constraints (honest scope)

This ephemeral container has **no GPU, no faster-whisper/ctranslate2 weights,
and no Gundam Wing media**, so a full pipeline run + fresh `clipai_logs_*.txt`
cannot be produced here. Verification is therefore:
- Unit tests (pytest) for every pure-logic change: the convert temp-dir/rename
  (with a mocked converter, incl. the pre-existing-dir case that used to fail),
  fuzzy `drop_repetition_loops`, music-span suppression, and `:free` detection.
- A code-level trace of the decoding-param and pipeline wiring on the device
  paths that need a GPU/model to execute.
The build/deploy + real-video verification must be run on the Unraid host; the
steps and expected log lines are recorded in `CHANGES.md`.
