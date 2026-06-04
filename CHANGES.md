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
