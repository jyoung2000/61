from dataclasses import dataclass, replace
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    # Primary provider
    AI_PROVIDER: str = "openrouter"

    # OpenRouter (uses OpenAI-compatible API — set your key from openrouter.ai)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_PRESET: str = "balanced"  # free | efficient | balanced | premium | custom
    OPENROUTER_PRIMARY_MODEL: str = "google/gemini-2.5-flash"
    OPENROUTER_EDITORIAL_MODEL: str = "google/gemini-2.5-pro"
    OPENROUTER_SUMMARY_MODEL: str = "google/gemini-2.5-flash"
    # Dedicated subtitle-translation model. Blank → fall back to
    # OPENROUTER_EDITORIAL_MODEL (mirrors OLLAMA_TRANSLATION_MODEL's intent).
    # Lets users pick a separate, translation-strong OpenRouter model for
    # the subtitle translation pass while keeping the editorial model for
    # transcript polishing.
    OPENROUTER_TRANSLATION_MODEL: str = ""

    # Direct provider keys
    ANTHROPIC_API_KEY: str = ""
    # Per-provider model picks — populated by the per-user settings
    # overlay (``/api/auth/me/settings``) so each user can run a
    # different model on the same backend without restarting.
    ANTHROPIC_MODEL: str = ""             # blank → provider's MODEL constant default
    GEMINI_API_KEY: str = ""
    GEMINI_EDITORIAL_MODEL: str = ""           # blank → "gemini-2.0-flash"
    GEMINI_VIDEO_MODEL: str = ""          # blank → "gemini-2.5-flash"
    GEMINI_USE_NATIVE_VIDEO: bool = False
    GROQ_API_KEY: str = ""
    GROQ_EDITORIAL_MODEL: str = ""             # blank → "llama-3.3-70b-versatile"
    OPENAI_API_KEY: str = ""                   # for TRANSCRIPTION_PROVIDER=openai
    # ── Cloud transcription (audit Phase 4.1) ──
    # local (faster-whisper, default) | groq | openai. Cloud output flows
    # through the SAME post chain (hallucination filter → forced alignment
    # → polish → formatter) and falls back to local automatically on any
    # API failure. Custom vocabulary is injected as the provider prompt.
    TRANSCRIPTION_PROVIDER: str = "local"
    GROQ_TRANSCRIBE_MODEL: str = "whisper-large-v3-turbo"
    OPENAI_TRANSCRIBE_MODEL: str = "whisper-1"  # or gpt-4o-transcribe (no word timestamps — forced alignment re-times)
    # ── Subtitle polish model (audit Phase 4.2) ──
    # Explicit model for transcript polishing (e.g. an OpenRouter id from
    # the "Recommended for subtitle polish" picker). Takes precedence over
    # SUBTITLE_POLISH_USES_TRANSLATION_MODEL. Blank = legacy behavior.
    SUBTITLE_POLISH_MODEL: str = ""

    # Ollama — Primary AI (video/vision) + Editorial AI (scoring/summary).
    # Defaults are sized to fit a 4GB GTX 1650 so the same image runs
    # everywhere: moondream:1.8b (~1.8GB) for Primary AI and
    # qwen2.5:3b-instruct (~1.8GB) for Editorial AI. The pipeline frees
    # Whisper VRAM before the VLM stage, so each gets the GPU in turn.
    # On 6GB+ GPUs switch Primary AI to llava:7b (or larger) in Settings.
    OLLAMA_HOST: str = "http://ollama:11434"
    OLLAMA_PRIMARY_MODEL: str = "moondream:1.8b"
    # EDITORIAL model (summaries, SEO, clip scoring, transcript polish). Kept at
    # the 3B class because it must run on the GPU for the per-clip SEO / summary
    # passes to be fast: a 4B-q4 model does NOT fit a 4 GB GTX 1650 (it OOMs and
    # falls back to CPU, turning 63-clip SEO into a 30+ min crawl). qwen2.5:3b
    # (~1.8 GB) fits in the VRAM freed after Whisper.
    OLLAMA_EDITORIAL_MODEL: str = "qwen2.5:3b-instruct"
    # Dedicated model for subtitle TRANSLATION (and offline MTPE). Qwen3-4B-
    # Instruct-2507 is a NON-thinking, multilingual-tuned model used ONLY for the
    # translation pass (where quality matters most), routed via model_override so
    # editorial/SEO keep using the fast qwen2.5 above. On a 4 GB card the 4B runs
    # on CPU (translation is slower but higher quality); on ≥6-8 GB it fits the
    # GPU. The exact Ollama tag may vary by quant (…-q4_K_M / -q8_0 / -fp16) or a
    # user Modelfile — overridable via env; never hardcode a tag deeper in code.
    OLLAMA_TRANSLATION_MODEL: str = "qwen3:4b-instruct-2507-q4_K_M"
    # ── Qwen3 translation sampling ──
    # Qwen3 is prone to repetition without a presence/repetition penalty, and for
    # deterministic subtitle JSON we want LOW temperature. These apply ONLY to the
    # Qwen3 family on the dedicated translation path (see
    # translator._translate_batch_via_ollama / local_models.qwen3_translation_options)
    # — they do NOT change global editorial sampling (summaries/descriptions still
    # use their higher-diversity settings). Qwen3's official non-thinking sampling
    # guidance: temperature ~0.7 for chat, but subtitle translation wants
    # determinism, so we default lower.
    # Qwen3's model card warns that greedy / very-low-temperature decoding makes
    # the model fall into ENDLESS REPETITION. For non-thinking models it
    # recommends temperature≈0.7, top_p=0.8, top_k=20, min_p=0, plus a
    # presence_penalty up to ~1.5 to break loops. The old near-greedy temp=0.2
    # was itself a cause of the repeating subtitle lines, so we move to the
    # card's anti-repetition profile (still deterministic enough for faithful
    # subtitles at 0.6). presence/frequency penalties suppress token- and
    # phrase-level loops at the source.
    QWEN3_TRANSLATION_TEMPERATURE: float = 0.6      # was 0.2 — Qwen3 loops near-greedy
    QWEN3_TRANSLATION_TOP_P: float = 0.8
    QWEN3_TRANSLATION_TOP_K: int = 20               # Qwen3 non-thinking default
    QWEN3_TRANSLATION_MIN_P: float = 0.0
    QWEN3_TRANSLATION_REPEAT_PENALTY: float = 1.1   # was 1.05
    # presence_penalty toward the high end can cause occasional language mixing
    # (source tokens leaking back). The pipeline already reverts a post-edit that
    # raises the source-script fraction (see pipeline.py fraction_untranslated
    # guard). If language-mixing regressions appear, lower this to ~0.8 BEFORE
    # touching temperature.
    QWEN3_TRANSLATION_PRESENCE_PENALTY: float = 1.2  # was 0.5 — card allows up to 1.5
    QWEN3_TRANSLATION_FREQUENCY_PENALTY: float = 0.3  # token-level loop suppression

    # ── Partial GPU offload for the 4B translation model on a small card ──
    # A 4B-q4 model's weights are ~2.5 GB — most of its layers DO fit a 4 GB
    # GTX 1650, only a few don't. The old behavior forced ALL layers onto the
    # GPU (num_gpu=99); that OOMs, and the fallback then dumped the WHOLE model
    # onto the CPU (very slow — the user's complaint). With partial offload we
    # step DOWN through a layer-count ladder (num_gpu=32 → 24 → 16) so most of
    # the model runs on the GPU and only a few layers spill to CPU — much faster
    # than full CPU. The first rung (99) lets a roomy card place everything on
    # GPU; the OOM-driven step-down self-tunes to whatever the card can hold, so
    # the same ladder is correct for 4 GB and 8 GB+ cards. Set False to restore
    # the old all-GPU-or-all-CPU behavior.
    OLLAMA_SMALL_GPU_PARTIAL_OFFLOAD: bool = True
    # Only models at/above this size get the partial ladder; smaller models that
    # already fit fully keep the simple [all-GPU, CPU] path.
    OLLAMA_PARTIAL_OFFLOAD_MIN_PARAMS_B: float = 3.5
    # First partial rung (layer count) tried after the all-GPU attempt OOMs.
    # qwen3:4b has ~36 layers; 32 keeps ~89% on GPU. Lowered automatically by
    # the ladder (×0.75, ×0.5) if 32 still OOMs.
    OLLAMA_MIDSIZE_GPU_LAYERS_START: int = 32

    # VideoLLaMA2 — optional local audio-visual model for Primary AI.
    # Requires ~10GB VRAM (RTX 4070+); on smaller GPUs the engine falls
    # back to the Ollama vision model, then cloud, then signal-only scoring.
    VIDEOLLAMA2_ENABLED: bool = True
    VIDEOLLAMA2_MODEL: str = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    VIDEOLLAMA2_QUANTIZE: str = "int8"

    # Replicate — cloud GPU inference for VideoLLaMA.
    # When set, the clipper sends video chunks to Replicate's hosted
    # VideoLLaMA3-7B instead of requiring a local 10GB+ VRAM GPU.
    # This is the recommended path for GTX 1650 / low-VRAM systems.
    REPLICATE_API_KEY: str = ""
    REPLICATE_MODEL: str = "lucataco/videollama3-7b"
    REPLICATE_ENABLED: bool = True   # user toggle — disable to skip Replicate even if key is set

    # VideoLLaMA3 Enhanced Discovery — multi-pass, image-mode, system prompt.
    # When True, the clipper uses ``ReplicateDiscoveryV3`` (system prompt +
    # adaptive chunking + audio annotations + optional refinement pass +
    # optional keyframe image analysis) instead of the flat single-pass
    # ``ReplicateDiscovery``. Every enhancement degrades independently if
    # the Replicate cog wrapper rejects a parameter or the call fails, so
    # the pipeline never crashes from the new code path.
    VIDEOLLAMA3_ENHANCED: bool = True
    VIDEOLLAMA3_FPS: int = 2                # frames/sec for V3 temporal sampling (1-4)
    VIDEOLLAMA3_MAX_FRAMES: int = 128       # max frames per chunk (V3 supports up to 180)
    VIDEOLLAMA3_REFINEMENT_PASS: bool = True  # enable coarse→refined two-pass discovery
    VIDEOLLAMA3_KEYFRAME_ANALYSIS: bool = True  # use V3 image mode for peak-signal keyframes
    VIDEOLLAMA3_AUDIO_ANNOTATION: bool = True   # annotate prompts with signal timeline audio data
    VIDEOLLAMA3_ADAPTIVE_CHUNKS: bool = True    # use signal-aware chunk sizing instead of rigid 10min
    VIDEOLLAMA3_CHUNK_MIN_S: int = 120          # minimum adaptive chunk (seconds)
    VIDEOLLAMA3_CHUNK_MAX_S: int = 900          # maximum adaptive chunk (seconds)

    # Fallback chain (ollama excluded by default — user can enable it in Settings)
    AI_FALLBACK_CHAIN: str = "openrouter,gemini,groq"

    # Analysis settings
    WHISPER_MODEL: str = "small"  # Auto-upgraded to large-v3-turbo when GPU detected
    WHISPER_MODEL_USER_SET: bool = False  # True when user explicitly chose a model in UI
    # Opt-in: auto-upgrade targets distil-large-v3 instead of
    # large-v3-turbo. distil is English-focused — near large-v3 English
    # WER at ~2x turbo speed — so only enable on English-only libraries.
    WHISPER_PREFER_DISTIL_ENGLISH: bool = False
    WHISPER_BEAM_SIZE: int = 5    # beam search for better accuracy (was 1/greedy)
    WHISPER_VAD_FILTER: bool = True    # skip silence — major speedup
    # ── Phase 1 coverage boosters ──
    # These default to ON because they improve first-pass transcript
    # coverage on ~every content type (podcasts, interviews, vlogs,
    # cinematic, music videos, anime) with negligible quality cost.
    # All four are per-user overridable via the settings overlay so
    # users who want the legacy behavior can disable them.
    #
    # Apply FFmpeg ``highpass=80, afftdn=-25, loudnorm=-18 LUFS`` before
    # Whisper sees the audio. Adds ~3-6s to extraction time on a
    # typical clip but recovers 2-5 percentage points of coverage on
    # mixed / quiet / noisy source material — this is the single
    # biggest reason Whisper drops faint speech.
    WHISPER_AUDIO_PRECONDITION: bool = True
    # The afftdn (FFT spectral denoiser) step of the preconditioning chain is
    # CPU-bound at only a few × realtime — on a 2 h track it alone adds ~15-20
    # min to the "frame+audio extraction" stage. Because fast GPU frame
    # extraction finishes long before it, the run LOOKS stuck at the next step
    # (faces) with no VRAM in use while ffmpeg grinds the audio. highpass +
    # single-pass loudnorm are far cheaper and carry most of the coverage win,
    # so on long videos we drop ONLY afftdn. Skip afftdn when the track exceeds
    # this many minutes; 0 disables the cap (always denoise). No effect when
    # WHISPER_AUDIO_PRECONDITION is off.
    WHISPER_PRECONDITION_DENOISE_MAX_MIN: int = 45
    # Silero VAD onset sensitivity. 0.15 (faster-whisper default) is
    # conservative and drops whispered / soft speech on realistic
    # content. 0.10 matches silero's published default and catches the
    # faint speech the 0.15 threshold was rejecting. CJK / anime paths
    # still override this to 0.08 for even more sensitivity.
    WHISPER_VAD_ONSET: float = 0.10
    # ``no_speech_threshold`` — Whisper's self-reported confidence that
    # a chunk is NOT speech. 0.8 (previous hard-coded default) threw
    # out a lot of valid quiet speech; 0.5 catches it while the worker
    # hallucination filter cleans up any false positives. Lowered to
    # 0.4 to recover more quiet / soft / off-mic speech that the
    # comparison to YouTube's official subtitle track showed missing
    # from the medium-model output.
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.4
    # ── Gap-fill second pass ──
    # After the main Whisper pass + Coverage Ledger build, scan for
    # runs of uncovered audio ≥ ``WHISPER_GAP_FILL_MIN_SEC`` seconds
    # and re-transcribe just those regions with the VAD filter
    # disabled and a much lower no-speech threshold. Catches:
    #   * quiet / whispered speech the main VAD dropped
    #   * sung lyrics in opening / ending themes
    #   * brief utterances under the VAD's 300 ms silence break
    # Gap-fill segments are tagged ``source='gap_fill'`` and routed
    # through the same hallucination filter; the only difference is
    # they're allowed to live in regions the main pass left empty.
    WHISPER_GAP_FILL_ENABLED: bool = True
    WHISPER_GAP_FILL_MIN_SEC: float = 1.5
    WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD: float = 0.25
    # Skip gap-fill entirely when the uncovered gaps exceed this fraction of the
    # video — i.e. the content is mostly NON-speech (music / silence), where
    # re-transcribing with VAD off invents far more than it recovers. On a
    # ~70%-silent anime the gap-fill pass re-transcribed 89 min of a 128-min
    # video and added 653 mostly-hallucinated cues (repeated run-ons + short
    # phantoms) that flooded the transcript; the VAD main pass is far more
    # reliable on such material. Speech-heavy videos (small gap fraction) are
    # unaffected — they still get gap-fill. 0 disables the cap.
    WHISPER_GAP_FILL_MAX_FRACTION: float = 0.6
    # Hard ceiling on the duration of any single gap-fill segment. With
    # ``vad_filter=False`` Whisper can collapse a whole OP song / minutes of
    # narration into one run-on cue stamped at a single timestamp. Any
    # gap-fill segment longer than this is split at word boundaries into
    # sub-segments ≤ this length so timestamps stay accurate and the
    # subtitle track never shows a giant block. 0 disables the split.
    WHISPER_GAP_FILL_MAX_SEC: float = 8.0
    # ── Anti-repetition / anti-hallucination decoding (Task 3) ──
    # condition_on_previous_text feeds each window the previous window's text
    # as a prompt. On music / singing / sparse-speech content (anime OP/ED,
    # insert songs) this DRIVES Whisper's repetition-loop pathology — the
    # opening narration re-emitted at six separated timestamps on the Gundam
    # Wing episode. Default it OFF; the small coherence loss on dialogue-dense
    # video is worth never generating the loop. Per-user overridable.
    WHISPER_CONDITION_ON_PREVIOUS_TEXT: bool = False
    # Block verbatim n-gram loops in the decoder (0 disables).
    WHISPER_NO_REPEAT_NGRAM_SIZE: int = 3
    # A segment whose gzip compression ratio exceeds this is degenerate/looped
    # → Whisper re-decodes it at the next temperature in the fallback ladder.
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    # Average log-probability floor; below it a segment is re-decoded hotter.
    WHISPER_LOG_PROB_THRESHOLD: float = -1.0
    # Token-level repetition penalty (>1 discourages loops). Dropped silently
    # on faster-whisper builds that don't expose it.
    WHISPER_REPETITION_PENALTY: float = 1.1
    # Temperature fallback ladder — on a degenerate/low-confidence segment
    # Whisper retries at the next temperature instead of emitting the loop.
    WHISPER_TEMPERATURE_FALLBACK: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    # ── VAD tuning + hallucination hardening (audit Phase 3.2) ──
    # min_silence 300ms catches intra-sentence pauses; speech_pad 150ms
    # (down from 200) tightens cue in/out points without clipping onsets.
    WHISPER_VAD_MIN_SILENCE_MS: int = 300
    WHISPER_VAD_SPEECH_PAD_MS: int = 150
    # faster-whisper's own silence-gap hallucination guard (skips segments
    # that follow ≥ this many seconds of silence when the decode is shaky).
    # Feature-detected in _decoding_kwargs; dropped on older builds.
    WHISPER_HALLUCINATION_SILENCE_S: float = 2.0
    # ── Two-pass difficult-segment redecode (audit Phase 3.4) ──
    # Segments flagged by the hallucination filter, with avg_logprob below
    # the threshold, or with degenerate word timestamps (batched-inference
    # word-timing bug) are re-decoded individually with beam_size=8 and
    # patience, bounded to MAX_FRAC of segments (worst first).
    WHISPER_REDECODE_ENABLED: bool = True
    WHISPER_REDECODE_LOGPROB: float = -0.8
    WHISPER_REDECODE_MAX_FRAC: float = 0.10
    WHISPER_REDECODE_BEAM: int = 8
    # ── CTC forced-alignment refinement (audit Phase 3.1) ──
    # Whisper word timestamps drift 50-200ms; a CTC forced aligner
    # (torchaudio wav2vec2, <1GB, or the ctc-forced-aligner package when
    # installed) re-aligns words against the audio so cues snap within
    # ~1 frame of speech onset/offset. No-op when no aligner backend is
    # available. GPU is used only when >1.5GB VRAM is free, else CPU.
    SUBTITLE_FORCED_ALIGN: bool = True
    # ── TACT phantom-hallucination filter (confidence-gated) ──
    # Whisper invents short, low-confidence cues over silence / music — the
    # "Don't let", "So nice", "Hmm." fragments (and repeated verbatim run-ons)
    # seen flooding a mostly-silent video — that slip past BOTH the exact-match
    # boilerplate blocklist AND the no_speech_prob>0.7 clamp. The Temporal
    # Audio Coverage (TACT) ledger already classifies a word as
    # ``low_confidence`` below 0.4; this gate feeds that SAME signal back to
    # DROP a cue when its words are OVERWHELMINGLY low-confidence (avg below
    # ``MAX_AVG_CONF`` and at least ``MIN_LOWCONF_FRAC`` of words under 0.4).
    # That low-confidence conjunction is the real discriminator — real speech
    # essentially never has 80%+ of its words below 0.4 confidence, so it is
    # preserved while the invented cues are dropped.
    #
    # ``MIN_NO_SPEECH`` defaults to 0.0 (the no_speech_prob condition is OFF).
    # It MUST NOT default high: the gap-fill pass only keeps segments with
    # no_speech_prob below its 0.25 threshold, so requiring a HIGH no_speech_prob
    # made this gate unable to fire on the gap-fill hallucinations that are the
    # actual flood (an earlier 0.50 default quarantined only ~4 cues on a video
    # drowning in them). Raise it only if you want to additionally require
    # Whisper-self-doubt on top of the confidence signal.
    WHISPER_PHANTOM_FILTER_ENABLED: bool = True
    WHISPER_PHANTOM_MAX_AVG_CONF: float = 0.40
    WHISPER_PHANTOM_MIN_LOWCONF_FRAC: float = 0.80
    WHISPER_PHANTOM_MIN_NO_SPEECH: float = 0.0
    # Auto-upgrade the Whisper model tier when the detected GPU has
    # spare VRAM. Existing logic already jumped ``small → large-v3-turbo``
    # on ≥6 GB cards; this flag extends the ladder so mid-tier GPUs
    # get ``small → medium`` on ≥2.5 GB as well.
    WHISPER_AUTO_UPGRADE: bool = True
    # ── Vocal separation (Demucs) before ASR ──
    # Isolate the vocal stem before Whisper so dialogue buried under loud
    # music / SFX (which the VAD otherwise hears as no-speech and drops) is
    # transcribed. Runs as a subprocess BEFORE Whisper loads, so it gets the
    # whole GPU and frees it on exit. Self-heals: if demucs isn't installed or
    # the pass fails, the job transcribes the original audio unchanged.
    # OPT-IN (default OFF): on a Japanese OP/ED with an English-loanword chorus
    # the dry vocal stem can flip Whisper's auto language detection to 'en'
    # (transcribe() now re-detects on the ORIGINAL audio to guard against this,
    # but separation also amplifies song vocals and is unvalidated as a blanket
    # default). Enable it deliberately for music-heavy material and compare.
    VOCAL_SEPARATION_ENABLED: bool = False
    VOCAL_SEPARATION_MODEL: str = "htdemucs"   # demucs model name
    VOCAL_SEPARATION_DEVICE: str = "auto"       # auto | cuda | cpu
    VOCAL_SEPARATION_SEGMENT: int = 7           # demucs --segment (CUDA VRAM cap)
    VOCAL_SEPARATION_TIMEOUT: int = 1800        # seconds, hard ceiling per pass
    FRAME_SAMPLE_RATE: int = 10        # seconds between frames (lower=more detail, slower)
    MAX_CLIP_CANDIDATES: int = 12

    # ── Reframer perception sampling (face/motion detection speed) ──
    # The Perceiver's face-detection cost scales with the number of frames
    # sampled. These cap the total samples on long videos. Lowering
    # REFRAMER_MAX_SAMPLES (or REFRAMER_MIN_SAMPLE_FPS) directly speeds up
    # the dominant analysis stage at a small reframing-accuracy cost.
    REFRAMER_MAX_SAMPLES: int = 1800       # total face/motion samples cap
    REFRAMER_SAMPLE_FPS: float = 5.0       # ceiling fps (short videos)
    REFRAMER_MIN_SAMPLE_FPS: float = 1.2   # floor fps (MEDIUM videos — applied
                                           # only while it stays within the cap)
    # Hard floor for VERY long videos where even the cap-respecting rate falls
    # below MIN_SAMPLE_FPS. Keeps a multi-hour video from under-sampling while
    # still bounding total samples. The MIN_SAMPLE_FPS floor no longer overrides
    # the sample cap (a 2 h video used to sample ~9 200 frames vs the 1 800 cap,
    # making face detection ~5x slower) — see resolve_sample_fps.
    REFRAMER_ABS_MIN_SAMPLE_FPS: float = 0.2
    # The heavy YOLO-World open-vocab subject detector (the ~0.5s/frame cost
    # behind the 17-min face stage) runs every Nth sampled frame, carrying its
    # subject bboxes forward in between. YuNet faces + motion still run EVERY
    # frame, so framing density is unchanged; subjects don't teleport in one
    # ~0.8s sample. 2 ≈ halves the YOLO cost; 1 restores every-frame detection.
    REFRAMER_YOLO_STRIDE: int = 2
    # On a STATIC frame with no face, no person, AND no motion (a title card /
    # logo / credits), spectral saliency latches onto the highest-contrast EDGE
    # (e.g. a centered logo's wing-tip) and mis-frames the crop off to the side.
    # When True, center the crop on such static graphics instead of chasing the
    # saliency edge. Frames with motion or a person/face are unaffected (they
    # still track the subject). Set False to restore pure saliency framing.
    REFRAMER_CENTER_STATIC_GRAPHICS: bool = True
    # ── Reframer crop zoom (face-size normalization) ──
    # During export the crop can be punched IN to drive the detected face
    # toward a fixed fraction of the frame (``_compute_zoom_factor`` — target
    # ~15% of crop width, up to 1.15x). On a 16:9 → 9:16 reframe this ENLARGES
    # any subject whose face is naturally smaller than that target, which makes
    # talking heads look "too big" and amplifies any tracking error (a slightly
    # off-center subject looks much worse when it is also blown up). The
    # reference reframer applies no punch-in at all: it renders the full-height
    # 9:16 slice and lets the subject sit at its natural size. Default off for
    # reference-parity framing; set True to restore face-size normalization.
    REFRAMER_FACE_SIZE_ZOOM: bool = False

    # ══════════════════════════════════════════════════════════════════
    #  Reframer "human camera operator" upgrades (2026 reframing rework)
    #  Each flag gates one improvement so the tuned baseline path stays
    #  reachable. Pure-code, low-risk items default ON; anything that
    #  needs an extra model download or is architecturally invasive
    #  defaults OFF with graceful fallback to the baseline behavior.
    # ══════════════════════════════════════════════════════════════════
    # One-Euro filter replacing the fixed-band velocity-adaptive EMA on the
    # planner target-x (Casiez 2012). Adapts its cutoff to subject speed:
    # near-zero lag on fast moves, heavy smoothing when slow — the direct
    # "smooth AND snappy" lever. Set False to restore the fixed-band EMA.
    REFRAMER_ONE_EURO_FILTER: bool = True
    REFRAMER_ONE_EURO_MINCUTOFF: float = 1.0   # Hz — lower = smoother when slow
    REFRAMER_ONE_EURO_BETA: float = 0.02       # speed coefficient — higher = snappier
    REFRAMER_ONE_EURO_DCUTOFF: float = 1.0     # Hz — derivative smoothing cutoff
    # Non-causal Savitzky-Golay pass over the whole target-x trajectory after
    # the keyframe smoother (we render offline, so lookahead is free). Kills
    # residual jitter without the reactive 6-pass keyframe surgery. Skips
    # cuts and _centering-flagged keyframes. Set False to disable.
    REFRAMER_SAVGOL_SMOOTHING: bool = True
    REFRAMER_SAVGOL_WINDOW_MS: int = 1200      # smoothing window (ms) on the path
    REFRAMER_SAVGOL_POLYORDER: int = 2         # polynomial order (2 = quadratic)
    # Fused saliency stack (spectral residual + center-prior + motion energy +
    # skin-color prior + temporal persistence) replacing the raw per-frame
    # spectral peak, plus temporal EMA smoothing of the saliency hotspot so it
    # stops wandering. Only affects faceless/subjectless frames. False =
    # baseline raw spectral peak.
    REFRAMER_SALIENCY_STACK: bool = True
    # Anticipatory lead-room: bias the crop ahead of a moving subject by a
    # fraction of crop_w * v̂ (velocity from finite diff), and add gaze-based
    # lead room via the existing gaze_estimator. Removes the "crop chasing the
    # subject" lag. False = no lead-room bias.
    REFRAMER_LEAD_ROOM: bool = True
    REFRAMER_LEAD_ROOM_MAX_FRAC: float = 0.10  # cap: fraction of crop_w
    # Eye-line framing anchor: frame on the eye-midpoint x (from YuNet
    # landmarks) instead of the raw bbox center — a steadier, more human
    # anchor. False = bbox-center anchor.
    REFRAMER_EYE_LINE_ANCHOR: bool = True
    # Velocity-augmented overlay: emit per-box velocity in the detection
    # overlay so the preview can advect boxes between sparse samples (optical-
    # flow-style propagation) for smooth 24-30fps preview tracking.
    REFRAMER_OVERLAY_VELOCITY: bool = True
    # ── Model-dependent / invasive items (default OFF, graceful fallback) ──
    # MediaPipe Face Landmarker (478-pt) for a true lips-based MAR mouth-open
    # signal. Needs the mediapipe wheel + face_landmarker.task model. Falls
    # back to the YuNet 5-point MAR when unavailable.
    REFRAMER_MEDIAPIPE_MAR: bool = False
    REFRAMER_MEDIAPIPE_MODEL_PATH: str = ""    # path to face_landmarker.task (blank = auto-discover)
    # u2netp learned salient-object model (4.7 MB ONNX via cv2.dnn) as the
    # no-face saliency source, spectral stack as fallback. Default ON — but it
    # is a graceful no-op (falls back to the fused spectral stack) until the
    # model file is present at REFRAMER_U2NET_MODEL_PATH.
    REFRAMER_U2NET_SALIENCY: bool = True
    REFRAMER_U2NET_MODEL_PATH: str = ""        # path to u2netp.onnx (blank ⇒ falls back to spectral)
    # L1-optimal camera path (Grundmann 2011) — decomposes the target path into
    # static holds + constant-velocity pans via an LP (scipy.optimize.linprog).
    # Replaces the reactive smoother output. Default ON.
    REFRAMER_L1_PATH: bool = True
    REFRAMER_L1_WEIGHTS: str = "1,10,100"      # w1,w2,w3 for |D1|,|D2|,|D3|
    # Deadband/hysteresis applied to the L1 targets: subject motion under
    # this fraction of crop width does NOT move the camera (holds stay
    # truly static instead of micro-drifting after the L1 solve).
    REFRAMER_L1_DEADBAND_FRAC: float = 0.025
    # Saccade behavior: displacements above this fraction of crop width
    # become a hard cut instead of a whip-pan (human editors cut, not pan,
    # for large re-frames). Applies in the Planner's pan/cut decision and
    # when keyframes are rebuilt from the L1 path.
    REFRAMER_SACCADE_CUT_FRAC: float = 0.38
    # Vertical composition: when the target is WIDER than the source
    # (16:9/4:3 outputs, letterboxed sources) place the subject eye-line
    # at ~1/3 from the crop top instead of blind vertical centering.
    # No-op when the crop uses full source height.
    REFRAMER_VERTICAL_EYELINE: bool = True
    # Minimum face-to-crop-edge margin as a fraction of crop height.
    REFRAMER_HEADROOM_MIN_FRAC: float = 0.08
    # Minimum keypoint density (Hz) for the export motion path. Eased
    # keyframe transitions are sampled at this rate so the FFmpeg
    # piecewise-LINEAR x expression can't produce visible velocity steps.
    REFRAMER_EXPORT_KEYPOINT_HZ: float = 10.0
    # Adaptive tiled face detection: run the (expensive) 2x2 tiled YuNet
    # pass only when the largest face found so far is smaller than
    # REFRAMER_TILED_MIN_FACE_FRAC of frame height (or no face at all).
    # Set to false to restore the always-tiled behavior.
    REFRAMER_TILED_ADAPTIVE: bool = True
    REFRAMER_TILED_MIN_FACE_FRAC: float = 0.04
    # Motivated zoom — time-varying push-in/pull-out via the dormant
    # motivated_zoom planner, adding a per-keyframe `scale` term rendered as a
    # time-varying ffmpeg crop. Default ON — a no-op unless the plan actually
    # carries a non-trivial scale (a detected held-speaker push-in).
    REFRAMER_MOTIVATED_ZOOM: bool = True
    REFRAMER_MOTIVATED_ZOOM_MAX: float = 1.15  # max push-in scale

    # ── Clip generation (Primary AI / VideoLLaMA3) defaults ──
    # Exposed in Settings > Clip Generation and overlaid onto the clipper
    # config so the upload pipeline + the regenerate path both honor them.
    CLIP_MIN_DURATION: int = 60        # seconds — shortest clip
    CLIP_MAX_DURATION: int = 300       # seconds — longest clip
    CLIP_COUNT: int = 0                # 0 = auto (scales with video length)
    # ── Clip export speed ──
    # The candidate-clip exporter used to re-encode every clip with CPU libx264
    # at -crf 18 — the single longest tail of the pipeline (~45-135 s/clip ×
    # dozens of clips = the bulk of a long run). It now uses the GPU encoder
    # (NVENC/VAAPI/QSV per the GPU toggle) at this quality, ~5-10× faster with no
    # meaningful loss for review clips, and exports several at once. CRF/CQ 21 is
    # visually transparent for these previews (18 was overkill). Concurrency runs
    # N clip encodes in parallel (independent ffmpeg jobs; the small 9:16 frames
    # are cheap) — 0/1 = sequential.
    CLIP_EXPORT_CRF: int = 21
    CLIP_EXPORT_CONCURRENCY: int = 2
    # Opt-in: skip re-encoding candidate clips entirely and just remux the bytes
    # (``-c copy``). Near-instant (the whole export phase drops from many minutes
    # to seconds), but the cut snaps to the nearest keyframe, so a clip may begin
    # a second or two before its intended moment. Great when you just need fast
    # previews to review; leave off when you need frame-accurate starts.
    CLIP_EXPORT_STREAM_COPY: bool = False
    # When a clip is exported, also drop a human-readable ``.txt`` next to the
    # MP4 (and auto-download it in the UI) carrying the clip's viral score,
    # title, suggested caption, hashtags, recommended platform, per-platform
    # SEO copy and captions — everything you'd paste into a social upload form.
    CLIP_EXPORT_SEO_SIDECAR: bool = True
    CLIP_PREFERRED_SUBJECTS: str = ""  # topics to prioritize, comma-separated
    CLIP_AVOID_SUBJECTS: str = ""      # topics to skip, comma-separated
    CLIP_DISCOVERY_PROMPT: str = ""    # custom VideoLLaMA3 prompt; "" = built-in default

    # ── Live trend brief (titles/tags/captions/hooks that work TODAY) ──
    # Social platforms change day-by-day and an LLM's training data is stale, so
    # ClipAI fetches a LIVE short-form trend brief (current hashtags / sounds /
    # hook formats / topics for TikTok + YouTube Shorts) once per day and injects
    # it into the clip judge + SEO generation. Three tiers, fail-soft: a web-
    # search model (primary) → Google Trends (free fallback, needs pytrends) →
    # the static lexicon (last resort). Refreshed daily; cached so it's one
    # cheap fetch/day reused across every clip. Set False to use the static
    # lexicon only (no live calls, no cost).
    LIVE_TRENDS_ENABLED: bool = True
    # OpenRouter web-search model for tier 1. ``perplexity/sonar`` has built-in
    # web search; alternatively append ``:online`` to any model (e.g.
    # ``google/gemini-2.5-flash:online``) to enable OpenRouter's web plugin.
    LIVE_TRENDS_MODEL: str = "perplexity/sonar"
    LIVE_TRENDS_REGION: str = "united_states"   # Google Trends region (tier 2)
    LIVE_TRENDS_CACHE_HOURS: int = 24           # refresh cadence (trends move daily)

    # ── Self-hosted mode — route the analysis pipeline to local AI ──
    # The master toggle flips every "auto" engine local; a per-engine
    # override ("local" / "cloud") wins over the master when set.
    SELF_HOSTED_MODE: bool = False
    CLIP_ENGINE_SOURCE: str = "auto"    # auto | local | cloud
    EDITORIAL_AI_SOURCE: str = "auto"   # auto | local | cloud
    # Offline Mode auto-selects the best installed local (Ollama) text model for
    # the editorial AI instead of using the configured cloud model / cloud judge
    # fallback. Models larger than this many billion params are skipped so we
    # never pick one that would spill off the 4 GB 1650 onto the CPU. The dropdown
    # picks still apply when Offline Mode is off.
    OFFLINE_EDITORIAL_MAX_PARAMS_B: float = 4.0
    # VRAM-aware editorial cap. A 4B-q4 model's weights (~2.5 GB) + its CUDA
    # compute buffer (~2.5 GB) exceed the ~2.5 GB free on a 4 GB card during the
    # editorial stage, so it OOMs and runs on CPU — making per-clip SEO/summaries
    # crawl and stall the job. On cards with LESS total VRAM than
    # OFFLINE_EDITORIAL_SMALL_GPU_GB, the editorial auto-selection cap is lowered
    # to OFFLINE_EDITORIAL_SMALL_GPU_MAX_PARAMS_B so it picks a model that fits
    # the GPU (e.g. qwen2.5:3b). Translation can still use a larger model via
    # OLLAMA_TRANSLATION_MODEL (a single pass, not per-clip). Set the GB floor to
    # 0 to disable the VRAM-aware downscope.
    OFFLINE_EDITORIAL_SMALL_GPU_GB: float = 5.5
    OFFLINE_EDITORIAL_SMALL_GPU_MAX_PARAMS_B: float = 3.0
    # When the cloud editorial chain is exhausted (e.g. an OpenRouter "Key limit
    # exceeded" 403), fall back to the local Ollama model for editorial tasks
    # (summary/SEO/polish/translation + the clip-scoring judge) so a run still
    # finishes on the GPU instead of failing. Only adds the fallback when an
    # Ollama host is configured; set False to keep cloud-only behavior.
    EDITORIAL_LOCAL_FALLBACK: bool = True

    # ── Editorial Judge specs ("<backend>:<model>") ──
    # The Editorial AI + Editorial AI Fallback dropdowns. Persisted here in
    # user_settings.json via _PERSISTABLE_KEYS (restored at module import) so
    # they survive container rebuilds exactly like the primary/editorial model
    # picks above — instead of relying solely on clipper_config.json + the
    # async startup restore, which could leave the fallback looking unsaved.
    EDITORIAL_AI_PRIMARY_SPEC: str = ""
    EDITORIAL_AI_FALLBACK_SPEC: str = ""
    # Adaptive frame extraction
    MIN_FRAMES: int = 30               # minimum for any video
    FRAMES_PER_MINUTE: float = 6       # target density (first 30 min; diminishes for longer videos)
    CONCURRENT_ANALYSES: int = 2
    AUTO_ANALYZE: bool = True
    SUBJECT_TRACKING_ENABLED: bool = True
    DENSE_FACE_SAMPLE_RATE: float = 0.5  # seconds between dense face detection frames (0.5 = 2fps)

    # FFmpeg encoding settings
    FFMPEG_PRESET: str = "fast"       # ultrafast|superfast|veryfast|faster|fast|medium|slow
    FFMPEG_CRF: int = 23             # 0-51, lower=better quality, 23=default
    FFMPEG_THREADS: int = 4          # Limit threads to control memory usage (0=auto risks OOM)
    FFMPEG_FASTSTART: bool = True    # -movflags +faststart for web streaming

    # GPU Hardware Acceleration — user toggle persisted to user_settings.json
    # Auto-enabled at startup when NVIDIA GPU is detected (see main.py)
    GPU_ACCELERATION_ENABLED: bool = False   # Toggle in Settings > Advanced
    GPU_VENDOR_OVERRIDE: str = ""            # Empty = auto-detect; "nvidia", "intel", "amd", "apple" to force
    GPU_HWDECODE_ENABLED: bool = True        # Use GPU for video decoding (NVDEC/DXVA2/VAAPI/VideoToolbox)
    GPU_HEVC_FOR_4K: bool = True             # Use HEVC encoder for 4K exports when available
    GPU_DEVICE_INDEX: str = ""               # SERVER FFmpeg CUDA device index ("0" etc.). Empty = auto. NEVER set from the browser/phone client report — the client's GPU is irrelevant server-side. With NVIDIA_VISIBLE_DEVICES pinned to the 1650's UUID, index 0 in-container is always the 1650.

    # Pre-analysis GPU memory preflight.
    # When True, the pipeline unloads any Ollama models still resident in
    # VRAM (via the Ollama daemon's keep_alive=0 API) before the first
    # frame extraction starts. Ollama itself keeps running — models
    # auto-reload on the next editorial AI request. Critical on low-VRAM
    # systems (GTX 1650 4 GB and similar) where stray Ollama models from
    # a previous run starve Whisper of the workspace it needs and force a
    # CPU fallback that's 10-20× slower than realtime.
    GPU_FREE_BEFORE_ANALYSIS: bool = True
    # Per-stage GPU preflight: also free VRAM right before the Whisper
    # transcribe stage, so anything that loaded during perceive (YOLO,
    # SFace, scene tracker) doesn't compete with Whisper for the
    # inference workspace. Lighter touch than the analysis-start preflight.
    GPU_FREE_BEFORE_WHISPER: bool = True

    # ── Custom Vocabulary (Whisper biasing) — Otter.ai-style accuracy lever ──
    # When enabled and the user has supplied glossary terms (see
    # backend/services/custom_vocabulary.py + /data/logs/custom_vocabulary.json),
    # those terms bias the Whisper decoder so jargon / names / acronyms /
    # brand terms transcribe correctly at the source. Empty glossary →
    # nothing is passed and Whisper behaves exactly as before.
    CUSTOM_VOCABULARY_ENABLED: bool = True

    # AI transcript post-correction
    AI_TRANSCRIPT_CORRECTION: bool = True  # Use LLM to fix proper nouns, punctuation, fillers

    # ── Transcript Polishing (real implementation) ──
    # When True, the post-analysis pipeline routes the transcript through
    # backend/services/transcript_polisher.py — a batched LLM corrector
    # that fixes proper nouns, punctuation, fillers, and sentence
    # boundaries. When False, the inert compat_stubs.correct_transcript
    # pass-through is used (the legacy behavior).
    TRANSCRIPT_POLISHING_ENABLED: bool = True
    TRANSCRIPT_POLISHING_BATCH_SIZE: int = 15   # segments per LLM call
    # Subtitle polishing is part of the SUBTITLE pipeline, not the editorial
    # pipeline. When True (default), the transcript/subtitle polish + MT
    # post-edit run on the dedicated translation model
    # (``OLLAMA_TRANSLATION_MODEL`` / ``OPENROUTER_TRANSLATION_MODEL`` — e.g.
    # qwen3:4b-instruct-2507, the multilingual model that also does the
    # translation) so the translation AI is the single brain that owns the
    # subtitles end-to-end (translate → polish). The editorial model
    # (``OLLAMA_EDITORIAL_MODEL`` — qwen2.5:3b) is then reserved for SEO +
    # summaries only. Falls back to the editorial model automatically when no
    # separate translation model is configured (e.g. cloud mode where they're
    # the same). Set False to restore polishing on the editorial model.
    SUBTITLE_POLISH_USES_TRANSLATION_MODEL: bool = True
    # Clean repetition INSIDE a single cue (the cross-cue dedup passes only see
    # whole-cue duplicates). Catches a small LLM duplicating its own output in
    # one line ("I will protect you I will protect you") or a stutter loop
    # ("no no no no no"). Conservative: a repeated unit must be ≥2 words, and a
    # single-word run must hit the threshold below before it's trimmed, so
    # genuine emphasis ("No, no.") is preserved.
    SUBTITLE_INTRA_CUE_DEDUP_ENABLED: bool = True
    # Minimum number of identical back-to-back words before a run is trimmed.
    # Netflix: "if a word/phrase is repeated twice in a row, translate it only
    # once." 3 keeps a genuine double ("No, no.") but collapses a 3×+ loop
    # ("no no no" → "no no").
    SUBTITLE_INTRA_CUE_MIN_WORD_RUN: int = 3
    # Per-batch time budget (seconds) for the polish loop when it runs on the
    # translation model. The translation model (qwen3:4b) is larger than the
    # editorial model and may run on CPU on a 4 GB card, so its batches are
    # slower — give the polish timeout more headroom than the editorial estimate
    # so a slow-but-progressing CPU batch isn't killed and dropped to raw text.
    SUBTITLE_POLISH_TRANSLATION_SECONDS_PER_BATCH: int = 180
    # Task 6 — transcription polish parity. When a job translates, the heavy
    # readability polish runs on the TARGET text after translation; by default
    # the SOURCE transcript was left raw before translation. Cloud transcripts
    # read cleaner partly because the source is effectively polished first, so
    # apply a single light source-language cleanup (punctuation / casing /
    # filler) BEFORE translation: the translator then works from clean input AND
    # the shipped source transcript reads cleanly. One extra LLM pass per
    # translate job (uses the local editorial model when offline); set False to
    # restore the pure translate-then-polish behavior. The full readability
    # reflow still runs on the translated text.
    TRANSLATION_POLISH_SOURCE_FIRST: bool = True
    TRANSCRIPT_FILLER_REMOVAL: bool = False     # remove um, uh, like, you know
    TRANSCRIPT_SENTENCE_REPAIR: bool = True     # fix run-on/fragmented sentences
    # Default ON: tells the polisher to keep every spoken word and only
    # add punctuation / sentence breaks for readability. Matches the
    # "the transcript should match the spoken audio" guideline — flip
    # off if you want the polisher to actually delete fillers /
    # restructure.
    TRANSCRIPT_PRESERVE_WORDS: bool = True
    # When the polish changes a segment's text it used to NULL the segment's
    # word-level timestamps (they no longer aligned char-for-char). Because the
    # polish adds punctuation to almost every segment, that wiped nearly all
    # word timing — so the downstream sentence segmenter fell back to its
    # char-proportional "if not words" branch and timed every sentence by
    # character length (a uniform-speech-rate assumption that drifts hundreds
    # of ms on slow / drawn-out speech). When True, RE-MAP the original word
    # timestamps onto the edited text by token alignment (polish is ~1:1 word
    # count by design) and carry the original start/end, only dropping a word's
    # timing when it has no counterpart. Set False to restore the old
    # null-the-words behavior.
    POLISH_REMAP_WORD_TIMESTAMPS: bool = True
    # Minimum fraction of edited tokens that must align to an original word for
    # the remapped timing to be trusted. Below this the segment keeps words=[]
    # (the polish rewrote too much to trust positional alignment).
    POLISH_REMAP_MIN_CONFIDENCE: float = 0.5
    # Deterministic, non-LLM punctuation restore fallback. Local-mode readability
    # otherwise depends entirely on the ≤4B editorial model inserting sentence
    # terminators — and when that model is unavailable, errors, or simply leaves
    # a segment with no terminator, the resegmenter has nothing to split on. When
    # True, such segments are passed through a lightweight punctuation restorer
    # (the optional ``deepmultilingualpunctuation`` model for Latin scripts; a
    # rule-based terminator for CJK) so readability does not hinge on the LLM.
    # The dependency is lazily imported and OPTIONAL — if it isn't installed the
    # fallback skips gracefully (Latin text is left unchanged; CJK still gets its
    # rule-based terminator). Set False to disable entirely.
    PUNCTUATION_RESTORE_FALLBACK_ENABLED: bool = True
    # Re-polish loop: keep iterating until the readability report scores
    # the transcript ≥ this target (0-100). Capped at 3 passes so a
    # poorly-segmented source can't loop forever.
    TRANSCRIPT_READABILITY_TARGET: float = 90.0
    TRANSCRIPT_READABILITY_MAX_PASSES: int = 3

    # ── Sentence-aware resegmentation (Task 4) ──
    # Merge same-speaker neighbours then re-split at sentence boundaries
    # (using Whisper word timestamps) so transcripts/subtitles break by
    # sentence instead of by raw VAD window. Runs after speaker fusion and
    # before the readability pass. When False, the raw segments flow
    # straight to readability enforcement (the legacy behaviour).
    SENTENCE_SEGMENTATION_ENABLED: bool = True
    # Pause-based (acoustic) sentence splitting. Local-mode readability depends
    # on a ≤4B model inserting sentence punctuation, which it does poorly
    # (especially CJK), so the terminator-based resegmenter falls back to
    # particle/char guessing. When True, a word-timed block that has NO sentence
    # terminators — or a terminator group that runs longer than
    # SENTENCE_SPLIT_MAX_CUE_MS — is split on inter-word silence gaps instead
    # (language-agnostic, no model), using the word timestamps that exist before
    # polish. Explicit terminators are still respected when present. Set False to
    # restore pure terminator/char-based splitting.
    SENTENCE_SPLIT_PAUSE_ENABLED: bool = True
    # Inter-word silence (ms) that counts as a sentence/clause boundary for the
    # pause-based splitter. ~350-450ms is a deliberate pause without firing on
    # ordinary dialogue rhythm.
    SENTENCE_SPLIT_PAUSE_MS: int = 400
    # Max cue length (ms) for the pause-based splitter: a word-timed group longer
    # than this is broken at its largest qualifying pause even when no terminator
    # is present, so a punctuation-less block can't survive as one wall-of-text
    # cue.
    SENTENCE_SPLIT_MAX_CUE_MS: int = 8000

    # ── Subtitle Readability + Safe Zones ──
    # Enforces Netflix-style CPS / line-length / duration limits and
    # platform-specific safe-zone margins for TikTok / Reels / Shorts.
    # Wired into srt_generator + ass_generator as a preprocessing step.
    SUBTITLE_CPS_ENFORCEMENT: bool = True       # enforce reading speed limits
    # Netflix uses language-specific reading-speed limits — up to 17 cps adult
    # (13 cps kids) for most languages; 20 is the looser English-USA value. The
    # output here is translated (usually non-English), so 17 is the correct
    # target. ``_cps()`` separately down-weights CJK glyphs, so CJK content lands
    # near Netflix's ~13 cps CJK-equivalent without a second knob.
    SUBTITLE_MAX_CPS: float = 17.0             # Netflix language-specific adult limit
    # Keep a cue WHOLE up to ``SUBTITLE_MAX_CPS × this`` and only split above it.
    # The reading-speed cap alone shatters every merged sentence right back into
    # 2-3 word flashes (a 193→174 merge re-exploded to 500+ cues), which is the
    # opposite of the goal. At 1.5 a cue is kept intact up to 30 cps and only the
    # genuinely unreadable ones split; Pass 0.7 still EXTENDS cues into idle time
    # toward the strict 20 cps, so lines read at the proper speed where there's
    # room and only the cramped ones run fast. 1.0 restores strict Netflix CPS;
    # raise toward 2.0 for even fuller (faster-reading) lines.
    SUBTITLE_SPLIT_CPS_TOLERANCE: float = 1.5
    SUBTITLE_MAX_CHARS_PER_LINE: int = 42       # Netflix Latin standard
    SUBTITLE_MIN_DURATION_MS: int = 833         # 5/6 second (Netflix minimum)
    SUBTITLE_MAX_DURATION_MS: int = 7000        # Netflix maximum per event (7s).
                                                # The phrase-merge still combines
                                                # slow fragments up to this cap;
                                                # fragments whose combined span
                                                # exceeds 7s stay separate (a
                                                # single caption must not sit on
                                                # screen longer than 7s per spec).
                                                # Clause-level resegmentation +
                                                # the 2-line/CPS budget keep
                                                # normal-pace cues ~4-5s.
    SUBTITLE_SMART_LINE_BREAKS: bool = True     # linguistic boundary breaks
    # Minimum characters a split piece may carry. Stops the duration
    # splitter from shattering slow / dramatic narration (Whisper detects
    # multi-second pauses *between* words) into unreadable one-word cues,
    # which also wrecks per-cue translation. 0 disables the guard.
    SUBTITLE_MIN_SPLIT_CHARS: int = 14
    # Before splitting an over-fast (CPS > cap) cue into 2-3 word flashes, first
    # STRETCH its on-screen time into the idle gap after it (bounded by the next
    # cue + the max display duration — never overruns a neighbour). Translated
    # CJK→EN cues are often longer than the source window that timed them, so the
    # naive fix is to shatter them; borrowing the (usually ample) silence after a
    # line keeps it a single readable phrase instead. The biggest lever against
    # choppiness on paused / sparse-speech videos. Splitting still runs for cues
    # that are STILL over the cap after stretching.
    SUBTITLE_EXTEND_BEFORE_SPLIT: bool = True
    # Greedy phrase-merge: combine consecutive same-speaker cues into one fuller
    # cue (up to the duration / 2-line / CPS limits) when the gap between them is
    # at most this many ms. VAD over-segments slow / paused speech, so the
    # translator emits many 2-3 word fragments ("So am I" / "planning on" /
    # "eating with you"); merging them into complete phrases is what makes the
    # transcript read like YouTube/Netflix captions instead of a flicker of short
    # lines, and is the biggest lever on the duration sub-score (no more sub-833ms
    # flashes). Only bridges SMALL gaps (continuous speech) — never a real pause,
    # a speaker change, or a [♪ music ♪] marker. 0 disables the merge.
    SUBTITLE_MERGE_MAX_GAP_MS: int = 3000
    # When the previous cue ends MID-SENTENCE (no terminal punctuation), widen
    # the merge bridge to this gap so the thought is completed into one line
    # ("and it's your" + "boobs here." → "and it's your boobs here.") instead of
    # trailing off across two flashes. ~42% of translated cues ended mid-sentence
    # on real runs, most within ~6s of their continuation. Still bounded by the
    # 2-line / max-duration / CPS caps, so parts that are genuinely far apart
    # (sparse speech) stay split. 0 disables the wider bridge.
    SUBTITLE_SENTENCE_MERGE_GAP_MS: int = 6000
    # Hard ceiling (ms) on the gap the phrase-merge will EVER bridge, even for a
    # mid-sentence continuation. Diarization labels everything "Speaker 1" on
    # single-speaker content, so the sentence-merge bridge (6 s) could glue cues
    # across a genuine pause into a run-on. This caps that: a silence larger than
    # this is treated as a real boundary and never merged across. 0 disables the
    # cap (legacy behavior).
    SUBTITLE_MERGE_MAX_PAUSE_MS: int = 4000
    SUBTITLE_PLATFORM_SAFE_ZONES: bool = True   # per-platform margin profiles
    SUBTITLE_PLATFORM_PROFILE: str = ""         # "" | tiktok | reels | shorts | horizontal | square

    # ── Translation Engine ──
    # ``auto`` picks the best available OFFLINE engine. For non-English → English
    # jobs the pipeline first tries Whisper's native translate task (see
    # WHISPER_TRANSLATE_TO_EN); otherwise: DeepL > Google (both key-gated) >
    # Opus-MT > NLLB. Translation NEVER uses an LLM — the AI model only polishes
    # the already-translated text for readability.
    TRANSLATION_ENGINE: str = "auto"            # auto | nllb | opus-mt | fugumt | google | deepl | whisper
    # Prefer Whisper's native audio→English translate task for non-English →
    # English jobs (matches repo-60's offline approach). It is a single-step,
    # fully-offline ASR-translate pass that avoids the transcribe-then-translate
    # double-error and never touches an LLM. When it yields no output (or the
    # target language is not English), translation falls back to the offline NMT
    # engines below.
    # Translate the SOURCE transcript text-to-text with the editorial LLM (the
    # same orchestrator used for summary/SEO/polish) as the PRIMARY path. It's
    # the reliable, complete translation — it renders every cue (dialogue,
    # lyrics, narration) and never leaves the source language, unlike Whisper's
    # translate task which transcribes hard/music segments in the source and
    # produced half-translated tracks. Falls back to Whisper-native / offline NMT
    # when no LLM is configured or it comes back still source-language.
    TRANSLATION_PREFER_LLM: bool = True
    # OPT-IN second pass (default OFF). On the LLM-first path the first call did
    # the translation, so the MTPE post-edit is (correctly) skipped — leaving the
    # offline default as a single 4B pass with no editorial refinement. When this
    # is on AND the active engine is local, run ONE self-refinement pass over the
    # already-translated cues (NOT a re-translate): the model rewrites its own
    # target lines against the aligned source purely for fluency/de-stutter,
    # reusing correct_transcript(mode="translation") and its "never raise the
    # source-script fraction" revert guard. COST: a second full pass over every
    # cue — on a 4 GB card the 4B is already partial-offloaded to CPU, so this
    # ~doubles an already-slow inference. Suitable for an unattended batch clip,
    # never interactive. Leave OFF unless you accept the wall-clock cost; the
    # deterministic Tasks 1/2/4/5 carry the quality at zero added latency.
    TRANSLATION_LLM_REFINE_PASS: bool = False
    # Detect transliterated-Japanese (romaji) cues the model spelled out
    # phonetically instead of translating ("Katte ippai tsukuritaku naru toko").
    # These carry no CJK script, so the CJK-only completeness check scored them
    # 0% and shipped them in the English track. Only applied when the SOURCE is
    # Japanese (romaji patterns overlap with open-syllable Romance languages).
    TRANSLATION_ROMAJI_DETECT_ENABLED: bool = True
    # Fraction of a line's word tokens that must look like Japanese mora before
    # it's treated as untranslated romaji (0.6 cleanly separated real romaji from
    # English on the audited output).
    TRANSLATION_ROMAJI_DETECT_THRESHOLD: float = 0.6
    # Per-batch timeout for LLM subtitle translation (a CEILING — never slows the
    # fast path; a fast GPU batch returns in seconds regardless). A small model
    # on a low-VRAM GPU needs far more than the old 5 s/segment / 60 s floor; too
    # low and it gets killed mid-answer and falls back to the weaker offline NMT,
    # leaving cues in the source language. Sized for the worst case: a 4B
    # translation model (qwen3:4b-instruct-2507) running on CPU on a 4 GB card,
    # where a CJK→EN batch can take a few minutes — at 180 s those batches were
    # killed, tripping the >20% residual check and silently demoting the whole
    # translation to NMT. 300 s floor / 20 s per segment gives CPU batches room
    # to finish so the chosen translation model is actually the one that runs.
    TRANSLATION_LLM_SECONDS_PER_SEGMENT: float = 20.0
    TRANSLATION_LLM_TIMEOUT_FLOOR: float = 300.0
    # LLM translation batch size. Smaller batches on a local model generate a
    # shorter JSON array faster + more reliably (less timeout risk). 0 = auto
    # (8 for Ollama, 18 for cloud).
    TRANSLATION_LLM_BATCH: int = 0
    # After the offline NMT (FuguMT/NLLB) runs, any cue it left in the source
    # language is re-translated ONE AT A TIME with a plain-text LLM call (robust
    # where the batched JSON path fails on small local models). Cap the number of
    # such cues (0 disables the cleanup) and bound its wall-clock time so a
    # hopelessly-garbled transcript can't run for hours.
    TRANSLATION_LLM_CLEANUP_MAX_CUES: int = 500
    TRANSLATION_LLM_CLEANUP_BUDGET_S: float = 1200.0
    WHISPER_TRANSLATE_TO_EN: bool = True
    # Whisper-native translate is a second full ASR pass; it's only worth it when
    # it can run on the GPU. Below this much FREE VRAM it would fall back to CPU
    # (~30 min for a 25-min video), so the pipeline skips it and uses offline NMT
    # (NLLB int8, which loads in the freed VRAM and finishes in seconds) instead.
    WHISPER_TRANSLATE_MIN_FREE_GB: float = 4.0
    # Whisper's translate task skips/merges non-speech (esp. singing), so on
    # music/lyric-heavy videos it can cover far less of the audio than the source
    # transcription did. When the Whisper-native English track covers less than
    # this fraction of the source-transcript timeline, discard it and use dense
    # offline NMT on the full source instead (every source cue gets translated).
    WHISPER_TRANSLATE_MIN_COVERAGE: float = 0.6
    # ── Hybrid word-timed line splitting (JA→EN) ──
    # The editorial LLM produces the authoritative English TEXT but no timing;
    # Whisper's translate task produces audio-aligned English WORD timestamps but
    # weaker text. When True, after the LLM translation we run a Whisper-native
    # English pass purely as a TIMING REFERENCE and project those word times onto
    # the LLM text (same-language EN↔EN monotonic alignment). The projected
    # per-word times let the readability splitter break run-on LLM cues at real
    # audio pauses. Whisper-EN text NEVER enters the output (text stays 100% LLM).
    # Degradation ladder: A=Whisper-EN projection, B=source-JA word-pause timings,
    # C=keep the cue whole (never char-proportional-time a word-less cue).
    HYBRID_WORD_TIMING_ENABLED: bool = True
    # Minimum fraction of an LLM cue's words that must align to a Whisper-EN word
    # for the projection to be trusted (else the cue falls to tier B/C).
    HYBRID_MIN_ANCHOR_RATIO: float = 0.30
    # Time margin (s) around an LLM cue when gathering Whisper-EN candidate words
    # (the two translations drift, so allow slack at the edges).
    HYBRID_ALIGN_MARGIN_S: float = 2.0
    # The Whisper-EN timing pass is a second ASR pass; only run it when the engine
    # is still cached (free reuse) or this much VRAM is free. Otherwise degrade to
    # tier B. Reuses WHISPER_TRANSLATE_MIN_FREE_GB's intent at a lower floor since
    # this is inference-only on a (possibly) already-warm model.
    HYBRID_WHISPER_REF_MIN_FREE_GB: float = 3.0
    # Per-pass ceiling (s) for the Whisper-EN timing reference; on timeout we
    # degrade to tier B rather than block the job.
    HYBRID_WHISPER_REF_TIMEOUT_S: float = 1800.0
    TRANSLATION_CONTEXT_WINDOW: int = 5         # segments before/after for context
    TRANSLATION_GLOSSARY_ENABLED: bool = True   # per-video KNP glossary support
    # Auto-derive a per-video glossary of recurring proper nouns from the source
    # transcript and feed it to the translator so recurring names render
    # consistently (no "Relena/Lillian/Liliana" drift) and coined nouns are
    # transliterated, not translated into ordinary words. Content-agnostic; works
    # in any source language. Set False to disable the auto glossary.
    TRANSLATION_AUTO_GLOSSARY: bool = True
    # Push the LLM translator toward natural, idiomatic English (dub/localization
    # phrasing) instead of a structurally-literal rendering — while preserving
    # the exact meaning. Set False to revert to the plain faithful style.
    TRANSLATION_IDIOMATIC: bool = True
    GOOGLE_TRANSLATE_API_KEY: str = ""          # for Google Cloud Translation v3
    DEEPL_API_KEY: str = ""                     # for DeepL API
    # NMT model identifiers — downloaded on demand (NOT at startup).
    # Default to NLLB-200-distilled-1.3B: markedly more fluent than the 600M
    # and the int8 CT2 copy (~1.3-1.5 GB) still fits the freed VRAM on a 4 GB
    # GTX 1650 (translation runs AFTER analysis releases Whisper's VRAM); it
    # falls back to CPU int8 if VRAM is short. On an even smaller card, pin back
    # to the lighter model with the env override
    # ``NMT_NLLB_MODEL=facebook/nllb-200-distilled-600M``.
    NMT_NLLB_MODEL: str = "facebook/nllb-200-distilled-1.3B"
    NMT_OPUS_MT_TEMPLATE: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    # FuguMT (staka/fugumt-ja-en): a JParaCrawl-trained, Japanese-specialised
    # Marian model — markedly better ja↔en than the NLLB/Opus generalists on
    # everyday vocabulary and idioms. Loads through the Opus-MT machinery
    # (CTranslate2 int8, ~300 MB) in its own cache dir. Selected via
    # TRANSLATION_ENGINE=fugumt; it only covers ja↔en, so the router falls back
    # to NLLB for any other pair.
    NMT_FUGUMT_TEMPLATE: str = "staka/fugumt-{src}-{tgt}"
    # With TRANSLATION_ENGINE=auto, prefer FuguMT for Japanese↔English (it beats
    # the NLLB/Opus generalists on everyday JA vocabulary). Other pairs still
    # resolve to NLLB/Opus. Set False to keep auto on NLLB for ja↔en too.
    NMT_PREFER_FUGUMT_JA_EN: bool = True
    # After offline NMT, auto-unify recurring proper nouns the small model spelled
    # several ways ("Doria"/"Dorian", "Zechs"/"Zex") to the dominant spelling.
    # Fully automatic (no glossary), output-only, conservative — makes names
    # CONSISTENT (not necessarily the official spelling, which offline models
    # don't know). Set False to keep the raw NMT spellings.
    NMT_AUTO_NAME_CONSISTENCY: bool = True
    # Auto-download the offline NMT model the first time a translation needs
    # it (no manual Settings step). When ``auto`` resolves to a local engine
    # but nothing is on disk yet, the translator fetches + converts NLLB-200
    # (a one-time int8 download: ~1.3-1.5 GB for the 1.3B default, ~600 MB for
    # the 600M), then translates fully offline. The LLM path is only used as a
    # last resort if the download itself fails.
    NMT_AUTODOWNLOAD: bool = True
    # Each Opus-MT language pair is a separate model dir. Cap how many pairs
    # we keep on disk; least-recently-used pair dirs beyond the cap are pruned
    # after a new download so translating many directions can't fill the disk.
    # NLLB-200 is a single model covering 200 languages, so it stays the
    # preferred auto-download and is never counted against this cap.
    NMT_MAX_OPUS_PAIRS: int = 5
    # NMT device policy. ``auto`` → CUDA when available (int8_float16), else CPU
    # (int8) — EXCEPT on a small GPU (total VRAM ≤ 4 GB, e.g. the GTX 1650)
    # where Whisper + Ollama already compete for VRAM: there ``auto`` resolves
    # to CPU, the safe choice that avoids OOM (a 24-min video still translates
    # in a couple of minutes on CPU). Set ``cuda`` to force the GPU anyway, or
    # ``cpu`` to force CPU everywhere. Opus-MT always runs on CPU.
    NMT_DEVICE: str = "auto"                    # auto | cpu | cuda

    # ── Offline translation MTPE (machine-translation post-editing) ──
    # After the offline NMT engine (NLLB / Opus-MT) produces a fluent DRAFT,
    # post-edit it with the dedicated local translation model
    # (``OLLAMA_TRANSLATION_MODEL``, e.g. qwen2.5:3b) acting as an MT
    # post-editor: fix fluency / honorifics / idioms / glossary consistency
    # against the source — NOT a cold re-translation. Cue count + timing are
    # preserved; a bad/short response falls back to the raw NLLB draft.
    # OPT-IN (default OFF): without a canonical-name glossary, a tiny local
    # model (qwen2.5:3b on a 4 GB card) MANGLES proper nouns during the
    # post-edit ("Darlian" → "Liliana Doria", "Quatre" → "Catur"), and the raw
    # NLLB-1.3B draft is a dedicated translator that renders names more
    # faithfully. Enable it once a Custom Vocabulary glossary is populated (the
    # post-edit then keeps those spellings canonical) for the fluency gain.
    OFFLINE_TRANSLATION_MTPE_ENABLED: bool = False
    # Context window for the MTPE pass. Raised from the old 4096 toward 8192 so
    # longer videos keep real surrounding context (qwen2.5 supports it). This is
    # the CPU-rung context; GPU attempts use the smaller GPU ctx below so the KV
    # cache fits in VRAM.
    OFFLINE_TRANSLATION_MTPE_NUM_CTX: int = 8192
    # ── Translation context window ON the GPU ──
    # The KV cache scales linearly with num_ctx and is the single biggest VRAM
    # cost after the weights: for a 4B model the KV cache at 8192 ctx is ~1.2 GB,
    # which pushes the model past a 4 GB card even with partial layer offload. A
    # subtitle batch only needs ~2K tokens, so on GPU attempts we cap the context
    # to this (KV cache drops to ~0.3 GB) — letting qwen3:4b actually fit and run
    # on the GPU. The full (large) context above is still used on the CPU rung,
    # where the KV cache lives in plentiful system RAM. Raise only if a single
    # batch's prompt approaches this size.
    OLLAMA_TRANSLATION_GPU_NUM_CTX: int = 2048
    # num_batch for translation GPU attempts. A smaller batch shrinks the compute
    # graph buffer (another VRAM cost) at a small throughput cost — worth it to
    # keep the model on the GPU on a 4 GB card.
    OLLAMA_TRANSLATION_GPU_NUM_BATCH: int = 128

    # ── Translation quality mode (Task 5) ──
    # ``speed`` (default): the offline NLLB-draft → MTPE chain above — fast and
    # GPU-friendly on a 4 GB card. ``quality``: route the offline PRIMARY
    # translation through a larger Ollama model on CPU. You can't fit a 7-8B at
    # GPU speed on a GTX 1650, but it runs on CPU (slow, higher quality); NLLB
    # is the completeness backstop for any cue the big model leaves in the
    # source language. ``quality`` only takes effect when an Ollama host +
    # ``TRANSLATION_QUALITY_MODEL`` are configured; otherwise it transparently
    # uses the speed chain. When ``quality`` is set, the pipeline also skips its
    # LLM-first / Whisper-native preemptions so this CPU path is the one that
    # runs. ``speed`` mode timing is unchanged.
    TRANSLATION_QUALITY_MODE: str = "speed"          # speed | quality
    TRANSLATION_QUALITY_MODEL: str = "qwen2.5:7b-instruct"

    # ── Music marking ──
    # Insert a "[♪ music ♪]" marker cue over sustained music regions (OP/ED
    # themes, insert songs) instead of letting Whisper hallucinate lyrics or
    # leave a silent gap. The viewer sees that music is playing; lyrics are
    # NOT transcribed/translated. Markers are language-neutral and pass
    # through the translator verbatim.
    SUBTITLE_MARK_MUSIC: bool = True
    SUBTITLE_MUSIC_MIN_SEC: float = 5.0         # only mark sustained music
    # In a sustained music-only span, drop Whisper "speech" cues (hallucinated
    # lyrics / vocalisations) so the span is positively labelled music instead.
    # Dialogue OVER music is classified `speech` (not `music`), so it is not in
    # a music span and is untouched. A cue is suppressed when this fraction of
    # its timespan falls inside a music span.
    SUBTITLE_SUPPRESS_SPEECH_IN_MUSIC: bool = True
    SUBTITLE_MUSIC_SUPPRESS_OVERLAP: float = 0.6
    # The spectral classifier can mislabel dialogue over a loud orchestral /
    # action cue as ``music``; blanket suppression then deletes a whole spoken
    # section (observed as minute-long holes vs the reference subtitle track).
    # With this True, suppression inside a music span drops ONLY sung
    # vocalisations / onomatopoeia (``ああああ``, ``lalala``) and keeps
    # lexically-diverse real dialogue. A genuine OP/ED song is still marked
    # ``[♪ music ♪]`` (its marker survives because no real dialogue remains in
    # the span). Set False to restore blanket suppression of every cue.
    SUBTITLE_MUSIC_SUPPRESS_VOCALIZATIONS_ONLY: bool = True

    # ── Audio Analysis ──
    AUDIO_EVENT_DETECTION: bool = True          # spectral audio event classification
    AUDIO_EVENTS_IN_SUBTITLES: bool = False     # inject [applause], [music] into subtitle track
    AUDIO_MUSIC_DETECTION: bool = True          # detect sustained harmonic content

    # ── Voiceprint registry (Task 3) — Otter.ai-style cross-job naming ──
    # When enabled, a speaker embedding is extracted per diarized speaker
    # and matched against a persisted registry (/data/logs/voiceprints.json);
    # matches auto-apply the stored name. Renaming a speaker enrolls/updates
    # that voice (running-mean centroid → continuous learning). No-ops
    # gracefully when the pyannote embedding model / HF_TOKEN are absent.
    VOICEPRINT_ENABLED: bool = True
    VOICEPRINT_MATCH_THRESHOLD: float = 0.75  # cosine; conservative on purpose

    # Speaker diarization (pyannote)
    DIARIZATION_ENABLED: bool = True  # Use pyannote for real speaker diarization
    DIARIZATION_MIN_SPEAKERS: int = 1
    DIARIZATION_MAX_SPEAKERS: int = 0  # 0 = unlimited (pyannote auto-detects)
    HF_AUTH_TOKEN: str = ""  # HuggingFace token for pyannote model access

    # Local audio diarization (SpeechBrain ECAPA) — the no-HF-token fallback used
    # when pyannote can't load. Real audio diarization (not just the visual
    # left/right heuristic), fully offline after a one-time ~80 MB model fetch.
    LOCAL_DIARIZATION_ENABLED: bool = True
    LOCAL_DIARIZER_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"
    # "auto" runs ECAPA on the GPU when one is free (it's a tiny ~80 MB model,
    # and embedding 100+ cues on CPU costs MINUTES — 7 min on a 24-min video —
    # vs seconds on the GPU), falling back to CPU when CUDA is absent/contended.
    # Force "cpu" or "cuda" to override. The perception models are released
    # before diarization, so the GPU is free by the time this runs.
    LOCAL_DIARIZER_DEVICE: str = "auto"
    # Cosine-distance threshold for splitting speakers (agglomerative clustering).
    # Higher = more merging = fewer speakers. 0.55 over-split music/noisy audio
    # (an AMV clustered into the 8-speaker cap); 0.70 is a steadier default for
    # ECAPA AHC. Raise toward 0.8 to merge more, lower to separate more.
    LOCAL_DIARIZER_THRESHOLD: float = 0.70

    # Camera solver (per-shot AutoFlip-style crop planning)
    CLIPAI_CAMERA_SOLVER: str = "on"  # "on" | "off" — env CLIPAI_CAMERA_SOLVER

    # Content-type routing for solver tuning (off by default until tested)
    CLIPAI_CONTENT_ROUTING: str = "off"  # "on" | "off" — env CLIPAI_CONTENT_ROUTING
    CLIPAI_CONTENT_TYPES_ENABLED: str = ""  # comma-separated: "talking_head,stream" — empty = all

    # ── Cloud storage integration (Google Drive + Box) ───────────────────────
    # Feature flag — default on, flip to "false" in the Unraid template to
    # kill the entire feature if something goes wrong in production.
    CLIPAI_CLOUD_STORAGE_ENABLED: bool = True

    # Fernet key (32 url-safe base64 bytes) used to encrypt stored OAuth
    # tokens at rest. If missing, a loud warning is logged and an ephemeral
    # key is generated (which means every container restart drops stored
    # cloud sessions — set this in your Unraid template).
    CLIPAI_TOKEN_ENC_KEY: str = ""

    # Google Drive OAuth client credentials.
    GOOGLE_DRIVE_CLIENT_ID: str = ""
    GOOGLE_DRIVE_CLIENT_SECRET: str = ""
    GOOGLE_DRIVE_REDIRECT_URI: str = "http://localhost:8000/api/cloud/google_drive/callback"

    # Box OAuth client credentials.
    BOX_CLIENT_ID: str = ""
    BOX_CLIENT_SECRET: str = ""
    BOX_REDIRECT_URI: str = "http://localhost:8000/api/cloud/box/callback"

    def api_key_fingerprints(self) -> dict:
        """Last-4 fingerprint (+ length) of every external API key, for
        at-a-glance confirmation of WHICH key is active — e.g. that a freshly
        saved key, not a cached old one, is in play. NEVER returns the full
        secret. Reads the live settings, so when logged per-job (after the
        per-user overlay) it reflects the key actually used for that job."""
        def _fp(v: str) -> str:
            v = (v or "").strip()
            if not v:
                return "(unset)"
            return f"…{v[-4:]} ({len(v)} chars)" if len(v) > 4 else "**** (set)"
        return {
            "OpenRouter":      _fp(self.OPENROUTER_API_KEY),
            "Anthropic":       _fp(self.ANTHROPIC_API_KEY),
            "Gemini":          _fp(self.GEMINI_API_KEY),
            "Groq":            _fp(self.GROQ_API_KEY),
            "Replicate":       _fp(self.REPLICATE_API_KEY),
            "GoogleTranslate": _fp(self.GOOGLE_TRANSLATE_API_KEY),
            "DeepL":           _fp(self.DEEPL_API_KEY),
            "HuggingFace":     _fp(self.HF_AUTH_TOKEN),
        }

    @property
    def active_provider_chain(self) -> list[str]:
        # Self-hosted / local editorial routes every editorial LLM task to local
        # Ollama ONLY — no cloud fallback (Offline Mode must make no cloud calls).
        if self.resolve_ai_source("editorial") == "local":
            return ["ollama"]
        chain = [p.strip() for p in self.AI_FALLBACK_CHAIN.split(",") if p.strip()]
        # An explicit "Editorial AI → Cloud" override must put a cloud provider
        # first even if a local Ollama is pinned at the front of the chain (from
        # selecting Ollama models or the Ollama-Local toggle). Otherwise
        # provider_status keeps Ollama as the active editorial provider and the
        # cloud / Replicate engines never reappear. Demote Ollama to a trailing
        # fallback. (A plain "auto" chain is left as-is so the Ollama-Local toggle
        # keeps its intended local-primary-with-cloud-fallback behavior.)
        if self.EDITORIAL_AI_SOURCE == "cloud" and "ollama" in chain:
            cloud = [p for p in chain if p != "ollama"]
            if cloud:
                return cloud + ["ollama"]
        return chain

    @property
    def editorial_provider_chain(self) -> list[str]:
        """Chain the AI orchestrator actually executes for editorial tasks.

        Same as ``active_provider_chain`` but, in cloud mode with
        ``EDITORIAL_LOCAL_FALLBACK`` on and an Ollama host configured, appends a
        local Ollama last-resort fallback so summary / SEO / polish / translation
        survive a cloud key-limit by finishing on the GPU. The displayed
        ``active_provider_chain`` stays cloud-only so the status banner and the
        "Ollama (Local)" toggle aren't muddied by the safety-net entry."""
        chain = self.active_provider_chain
        if (self.EDITORIAL_LOCAL_FALLBACK
                and self.resolve_ai_source("editorial") == "cloud"
                and (self.OLLAMA_HOST or "").strip()
                and "ollama" not in chain):
            return chain + ["ollama"]
        return chain

    def resolve_ai_source(self, engine: str) -> str:
        """Return 'local' or 'cloud' for an engine ('clip' | 'editorial'),
        honoring its per-engine override or, when 'auto', SELF_HOSTED_MODE."""
        override = {
            "clip": self.CLIP_ENGINE_SOURCE,
            "editorial": self.EDITORIAL_AI_SOURCE,
        }.get(engine, "auto")
        if override in ("local", "cloud"):
            return override
        return "local" if self.SELF_HOSTED_MODE else "cloud"

    def resolve_stage_source(self, stage: str) -> str:
        """Return 'local' or 'cloud' for a USER-FACING pipeline stage.

        The Offline-Mode toggle in Settings → AI Provider speaks in terms of
        the four stages a user recognises; this maps each onto the two
        underlying engine knobs ``resolve_ai_source`` already drives:

          * ``transcription`` — always local. Whisper is the only transcriber
            (cloud LLM providers don't do speech-to-text); it runs on the GPU
            when VRAM is free, CPU otherwise.
          * ``clip_detection`` — the clip engine (Primary AI): local Ollama
            vision vs cloud VideoLLaMA3 on Replicate.
          * ``translation`` / ``polishing`` — the editorial AI. In self-hosted
            mode the editorial chain is local Ollama, so both run on the GPU
            (translation is LLM-first locally, then offline Whisper/NMT).
        """
        if stage == "transcription":
            return "local"
        if stage in ("clip", "clip_detection"):
            return self.resolve_ai_source("clip")
        if stage in ("translation", "polishing"):
            return self.resolve_ai_source("editorial")
        return self.resolve_ai_source(stage)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


# ── Duration Tier System ─────────────────────────────────────────────────


@dataclass
class VideoDurationTier:
    """Pipeline configuration that auto-adjusts based on video duration."""
    name: str
    max_minutes: float
    frame_sample_rate: int           # seconds between frame samples
    summary_strategy: str            # "single" | "map_reduce"
    summary_chunk_minutes: int       # chunk size for map_reduce (0 = N/A)
    window_duration: float           # clip detection window size in seconds
    window_overlap: float            # overlap between windows in seconds
    max_clip_candidates: int         # max clips to return
    max_gaps_pass2: int              # max coverage gaps to scan in pass 2
    hot_zone_top_n: int              # how many hot zones to send to AI
    per_call_timeout_base: int       # base timeout per AI call in seconds
    vision_batch_concurrency: int    # concurrent vision API batches


DURATION_TIERS = [
    VideoDurationTier("short",     15,   8, "single",      0,    0,    0, 12,  4, 15, 120, 2),
    VideoDurationTier("medium",    60,  12, "single",      0,  480,   90, 20,  6, 20, 180, 2),
    VideoDurationTier("long",     180,  20, "map_reduce", 10,  600,  120, 40,  8, 25, 240, 3),
    VideoDurationTier("marathon", 9999, 35, "map_reduce", 15,  600,  120, 60, 12, 30, 300, 3),
]


def get_duration_tier(duration_seconds: float) -> VideoDurationTier:
    """Return the appropriate tier configuration for a given video duration."""
    minutes = duration_seconds / 60
    for tier in DURATION_TIERS:
        if minutes <= tier.max_minutes:
            return tier
    return DURATION_TIERS[-1]


@dataclass
class OllamaTierOverrides:
    """Overrides applied when Ollama is the active provider.

    Ollama (local AI) has much smaller context windows (4K-8K tokens),
    slower inference (~12 tok/s on GTX 1650), and limited VRAM (4GB).
    Windows must be smaller, processing must be sequential, and timeouts
    must be much longer than cloud providers.
    """
    window_duration: float           # smaller windows (4-5 min) to fit context
    window_overlap: float
    per_call_timeout_base: int       # 300-480s for local 7B inference
    summary_strategy: str            # always map_reduce (context too small for single-pass)
    summary_chunk_minutes: int       # smaller chunks (5-10 min) for tiny context
    vision_batch_concurrency: int    # 1 for 4GB VRAM (never concurrent)
    max_transcript_chars: int        # per-window transcript budget (chars)
    max_scene_chars: int             # per-window scene budget (chars)
    sequential_windows: bool         # True = one window at a time (VRAM safety)


OLLAMA_TIER_OVERRIDES = {
    #                           window  overlap  timeout  summary   chunk  conc  tx_ch  sc_ch  seq
    "short":    OllamaTierOverrides(180,  20, 240, "single",       0, 1, 2500,  800, True),
    "medium":   OllamaTierOverrides(180,  30, 300, "map_reduce",   5, 1, 2000,  600, True),
    "long":     OllamaTierOverrides(180,  30, 360, "map_reduce",   5, 1, 1800,  500, True),
    "marathon": OllamaTierOverrides(180,  30, 420, "map_reduce",   5, 1, 1500,  400, True),
}

# Model recommendations based on available VRAM
OLLAMA_VRAM_PROFILES = {
    "4gb": {
        "vision": "moondream:1.8b",
        "text": "qwen2.5:3b-instruct",
        "translation": "qwen2.5:3b",
        "notes": "GTX 1650 / 4GB — fastest models that fit in VRAM",
    },
    "6gb": {
        "vision": "llava:7b-v1.6-q4_0",
        "text": "qwen2.5:7b-instruct-q4_0",
        "translation": "qwen2.5:3b",
        "notes": "RTX 2060 / 6GB — good balance of speed and quality",
    },
    "8gb+": {
        "vision": "llava:13b-v1.6-q4_0",
        "text": "llama3.1:8b-instruct-q4_0",
        "translation": "qwen2.5:7b",
        "notes": "RTX 3060+ / 8GB+ — highest quality local models",
    },
}


def apply_ollama_overrides(tier: VideoDurationTier, is_ollama: bool) -> VideoDurationTier:
    """Return a modified tier with Ollama-appropriate settings if Ollama is active."""
    if not is_ollama:
        return tier
    overrides = OLLAMA_TIER_OVERRIDES.get(tier.name, OLLAMA_TIER_OVERRIDES["medium"])
    return replace(
        tier,
        window_duration=overrides.window_duration,
        window_overlap=overrides.window_overlap,
        per_call_timeout_base=overrides.per_call_timeout_base,
        summary_strategy=overrides.summary_strategy,
        summary_chunk_minutes=overrides.summary_chunk_minutes if overrides.summary_chunk_minutes else tier.summary_chunk_minutes,
        vision_batch_concurrency=overrides.vision_batch_concurrency,
    )
