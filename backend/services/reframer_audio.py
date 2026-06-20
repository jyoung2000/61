"""ClipAI Reframer — Audio Intelligence (TACT transcription).

Extracted from clipai_reframer.py (jyoung2000/60) for the Fez engine
transplant. The original Tkinter GUI is not part of this module.
"""

import cv2
import numpy as np
import json
import subprocess
import threading
import os
import sys
import math
import logging
import time as _time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Callable, Dict
from pathlib import Path

from backend.services.reframer_models import (
    ReframeLogger, get_logger, reset_logger, RenderPlan,
    interpolate_x, clamp_x, _face_overlaps_person,
    LedgerBin, CoverageLedger, PerceptionResult, SceneSignals, AdaptiveParams,
)
from backend.config import settings

logger = logging.getLogger("clipai.reframer_audio")


from backend.services.hallucination_filter import (
    is_boilerplate_hallucination as _is_boilerplate_hallucination,
)


def _vocab_bias_kwargs(transcribe_callable, language: str) -> dict:
    """Build the custom-vocabulary biasing kwargs for a Whisper transcribe call.

    Delegates to ``custom_vocabulary.whisper_bias_kwargs``, which feature-
    detects ``hotwords`` support (preferred) and falls back to
    ``initial_prompt`` otherwise. Returns ``{}`` — preserving the current
    no-prompt behaviour exactly — when the feature is disabled or the
    glossary is empty.
    """
    try:
        from backend.services.custom_vocabulary import whisper_bias_kwargs
        return whisper_bias_kwargs(
            transcribe_callable,
            language=language,
            enabled=bool(getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True)),
        )
    except Exception as e:
        logger.warning("Custom vocabulary biasing skipped (%s)", e)
        return {}


def _decoding_kwargs(transcribe_callable, condition_on_previous_text=None) -> dict:
    """Anti-repetition / anti-hallucination decoding kwargs for a Whisper
    transcribe call, filtered to those the installed faster-whisper build
    accepts (mirrors ``_vocab_bias_kwargs``' feature-detection so an older
    build never raises on an unknown kwarg).

    ``condition_on_previous_text`` defaults to the
    ``WHISPER_CONDITION_ON_PREVIOUS_TEXT`` setting; pass an explicit value to
    force it (the gap-fill pass forces ``False`` over music/quiet regions
    regardless of the global default).

    These make the decoder REJECT looped / degenerate output instead of
    emitting it (Task 3):
      * ``condition_on_previous_text`` defaults OFF — it is the primary driver
        of the looped-narration hallucination on music / singing.
      * ``no_repeat_ngram_size`` blocks verbatim n-gram loops within a window.
      * ``compression_ratio_threshold`` + ``log_prob_threshold`` trip Whisper's
        ``temperature`` fallback so a degenerate segment is re-decoded hotter
        rather than kept.
      * ``repetition_penalty`` discourages token-level loops.
    Only kwargs that appear as explicit named parameters of the callable are
    returned (a ``**kwargs`` catch-all does NOT count — same rule as the vocab
    biasing helper), so passing the result is always safe.
    """
    import inspect
    cond = (bool(getattr(settings, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False))
            if condition_on_previous_text is None
            else bool(condition_on_previous_text))
    desired = {
        "condition_on_previous_text": cond,
        "no_repeat_ngram_size": int(getattr(
            settings, "WHISPER_NO_REPEAT_NGRAM_SIZE", 3)),
        "compression_ratio_threshold": float(getattr(
            settings, "WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4)),
        "log_prob_threshold": float(getattr(
            settings, "WHISPER_LOG_PROB_THRESHOLD", -1.0)),
        "repetition_penalty": float(getattr(
            settings, "WHISPER_REPETITION_PENALTY", 1.1)),
        "temperature": tuple(getattr(
            settings, "WHISPER_TEMPERATURE_FALLBACK",
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))),
    }
    try:
        params = inspect.signature(transcribe_callable).parameters
    except (TypeError, ValueError):
        # Can't introspect — pass nothing extra rather than risk an unknown
        # kwarg. The caller still sets condition_on_previous_text explicitly.
        return {}
    return {k: v for k, v in desired.items() if k in params}


def _cross_validate_segments(segments: list) -> list:
    """Remove cross-segment artefacts the per-segment TACT filter misses.

    Drops:
      - Exact text duplicates of the immediately preceding segment.
      - Segments that start before the previous segment ended (temporal
        overlap by more than 100 ms).
      - Segments with >80% word overlap with the previous segment (a
        common Whisper hallucination pattern on noisy audio).

    Lightweight pure-Python pass — no feature flag, only removes clearly
    invalid output.
    """
    if not segments:
        return segments
    cleaned: list = []
    prev_text = ""
    prev_end = 0.0
    for seg in segments:
        text = (seg.get("text", "") or "").strip()
        start = seg.get("start_sec", seg.get("start", 0)) or 0
        end = seg.get("end_sec", seg.get("end", 0)) or 0

        if text and text == prev_text:
            continue
        if cleaned and start < prev_end - 0.1:
            continue
        if prev_text and text:
            # Strip punctuation so "fine," and "fine" compare equal.
            import string as _str
            _tbl = str.maketrans("", "", _str.punctuation)
            prev_words = set(prev_text.lower().translate(_tbl).split())
            curr_words = set(text.lower().translate(_tbl).split())
            if prev_words and len(prev_words & curr_words) / len(prev_words) > 0.8:
                continue

        cleaned.append(seg)
        prev_text = text
        prev_end = float(end) or prev_end

    dropped = len(segments) - len(cleaned)

    # Second pass: collapse NON-adjacent near-duplicates — segments that
    # overlap in time and carry similar (not just identical) text. The
    # adjacency-only logic above misses re-transcriptions that land a few
    # hundred ms apart with the same line (the repeated proper-noun cue the
    # gap-fill pass produces over music). Operates on start_sec/end_sec keys.
    from backend.services.transcript_dedup import collapse_overlapping_duplicates
    cleaned, overlap_dropped = collapse_overlapping_duplicates(
        cleaned, start_key="start_sec", end_key="end_sec")
    dropped += overlap_dropped

    if dropped > 0:
        logger.info(
            "cross-segment validation removed %d duplicate/overlapping segments "
            "(%d adjacent, %d overlapping near-duplicate)",
            dropped, dropped - overlap_dropped, overlap_dropped,
        )
    return cleaned


def _drop_repetition_loops(segments: list) -> list:
    """Remove Whisper repetition-loop hallucinations.

    When the gap-fill pass re-transcribes music / quiet regions with VAD off,
    Whisper loops and emits the SAME text over and over, scattered across the
    timeline (so the adjacent-only ``_cross_validate_segments`` misses them).
    Real dialogue almost never repeats verbatim many times across an episode,
    so an exact-text segment that recurs beyond a small cap is a hallucination.

    Keeps the earliest occurrences (1 for long lines, up to 3 for short
    interjections like "了解") and drops the rest. Operates on the segment
    dicts (``text`` + ``start_sec``/``end_sec``); order-preserving.
    """
    from backend.services.transcript_dedup import drop_repetition_loops
    out, dropped = drop_repetition_loops(segments, text_key="text")
    if dropped > 0:
        logger.info(
            "repetition-loop filter removed %d duplicated hallucination segment(s)",
            dropped,
        )
    return out


class AudioIntelligence:
    """
    Whisper-based audio analysis for reframing intelligence.

    Model selection (speed vs accuracy tradeoff):
      - 'base'           : ~10x realtime on CPU, ~50x on GPU. Good enough for speech detection.
      - 'small'           : ~5x realtime on CPU, ~30x on GPU. Better accuracy.
      - 'large-v3-turbo'  : ~1x realtime on CPU, ~10x on GPU. Best accuracy, slowest.

    Default is 'small' — good balance of speed and accuracy for reframing.
    The model is cached between runs to avoid re-downloading.

    Requires: pip install faster-whisper
    """

    # Class-level model cache — survives across Perceiver instances
    _cached_engine = None
    _cached_model_name = None
    _cached_device = None
    # Sticky record of the model/device that ACTUALLY loaded most recently.
    # Unlike ``_cached_model_name`` (which ``_release_whisper_vram`` nulls when
    # it frees the engine after a job), these survive the VRAM release so the
    # Settings page can always report the real effective model that ran — even
    # after the engine has been unloaded. Set wherever a real load succeeds.
    _last_loaded_model_name = None
    _last_loaded_device = None

    @classmethod
    def _record_loaded(cls, model_name, device):
        """Stamp the model/device that just loaded (sticky, for the GUI)."""
        cls._last_loaded_model_name = model_name
        cls._last_loaded_device = device

    @classmethod
    def invalidate_cache(cls, reason: str = ""):
        """Drop the cached engine so the next job loads a freshly-selected
        model. Safe to call from anywhere (e.g. when the user changes
        WHISPER_MODEL in Settings) — frees the old engine's VRAM too. Leaves
        the sticky ``_last_loaded_*`` record intact for display."""
        had = cls._cached_model_name
        cls._cached_engine = None
        cls._cached_model_name = None
        cls._cached_device = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if had:
            logger.info("AudioIntelligence cache invalidated (was '%s')%s",
                        had, f" — {reason}" if reason else "")

    def __init__(self, model_name: str = 'small'):
        self.available = False
        self.engine = None
        self.model_name = model_name
        # The model the caller asked for, preserved across any VRAM downgrade
        # so the pipeline can report requested-vs-effective accurately (Task 4).
        self.requested_model_name = model_name
        self.device_used = 'unknown'
        # try_load() picks an appropriate batched-inference batch size
        # based on the GPU's total VRAM (4 → batch=4, larger → batch=16).
        # Surfaced as ``self._batch_size`` so ``transcribe()`` uses it.
        self._batch_size = 16

    def try_load(self) -> bool:
        """Load faster-whisper with GPU → CPU fallback.
        Uses cached model if same model was already loaded."""
        log = get_logger()

        try:
            from faster_whisper import WhisperModel
            log.log_stage('AUDIO', 'faster-whisper imported OK')
        except ImportError:
            log.log_stage('AUDIO',
                'faster-whisper not installed. Run: pip install faster-whisper')
            return False

        # State the requested model up front so the log makes any later
        # downgrade unambiguous — and so "what was asked for" vs "what loaded"
        # can be compared at a glance (Task 4).
        log.log_stage('AUDIO',
            f'Whisper model requested: {self.requested_model_name}')

        # Check if we already have this model loaded (skip re-download)
        if (AudioIntelligence._cached_engine is not None
                and AudioIntelligence._cached_model_name == self.model_name):
            self.engine = AudioIntelligence._cached_engine
            self.device_used = AudioIntelligence._cached_device
            self.available = True
            AudioIntelligence._record_loaded(self.model_name, self.device_used)
            # Re-derive batch size from the cached device tier so a reused
            # GPU engine doesn't suddenly run with the wrong workspace.
            if self.device_used and self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _total_gb = (_torch.cuda.mem_get_info()[1]
                                     / 1_073_741_824)
                        self._batch_size = 4 if _total_gb < 5.5 else 16
                except Exception:
                    self._batch_size = 16
            else:
                self._batch_size = 16
            log.log_stage('AUDIO',
                f'Whisper {self.model_name} already loaded ({self.device_used}, '
                f'batch={self._batch_size}) — using cached')
            return True

        # ── Diagnose CUDA ──
        cuda_available = False
        cuda_reason = "unknown"

        try:
            import torch
            torch_cuda = torch.cuda.is_available()
            torch_build = 'CUDA' if torch_cuda else 'CPU-ONLY'
            log.log_stage('AUDIO', f'PyTorch {torch.__version__} ({torch_build})')
            if torch_cuda:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_mem / 1073741824
                log.log_stage('AUDIO', f'GPU: {gpu_name} ({vram_gb:.1f} GB)')
            else:
                log.log_stage('AUDIO',
                    '*** PyTorch is CPU-only. Batched transcription will fail. ***\n'
                    '  FIX: pip install torch --index-url https://download.pytorch.org/whl/cu121')
        except Exception:
            pass

        try:
            import ctranslate2
            try:
                cuda_types = ctranslate2.get_supported_compute_types('cuda')
                if cuda_types:
                    cuda_available = True
                    cuda_reason = "ctranslate2 CUDA OK"
                    log.log_stage('AUDIO', f'CTranslate2 CUDA: {cuda_types}')
                else:
                    cuda_reason = "ctranslate2 no CUDA"
                    log.log_stage('AUDIO',
                        'CTranslate2 has no CUDA support. Fix: pip install ctranslate2 --force-reinstall')
            except Exception as e:
                cuda_reason = f"ctranslate2 error: {e}"
                log.log_stage('AUDIO', f'CTranslate2 CUDA check failed: {e}')
        except ImportError:
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_available = True
                    cuda_reason = "torch CUDA"
            except Exception:
                pass

        log.log_stage('AUDIO', f'CUDA: available={cuda_available}, reason={cuda_reason}')

        # ── VRAM budgets per (model, compute_type) ──
        # First number = load footprint, second = peak workspace under
        # the chosen batched-inference settings. ``int8_float16`` keeps
        # weights in int8 (half the fp16 footprint) but does matmul in
        # fp16 — same speed as float16 on modern GPUs, half the VRAM.
        # Empirical floors from real GTX 1650 / RTX 4060 / RTX 4090 runs.
        _vram_load_gb = {
            ('large',         'float16'):       3.0,
            ('large',         'int8_float16'):  1.6,
            ('large-v2',      'float16'):       3.0,
            ('large-v2',      'int8_float16'):  1.6,
            ('large-v3',      'float16'):       3.0,
            ('large-v3',      'int8_float16'):  1.6,
            # large-v3-turbo = large-v3's full encoder + a pruned 4-layer decoder
            # (~809M params), so its footprint sits between medium and large-v3.
            # int8_float16 (~1.0 GB load) fits a 4 GB GTX 1650 alongside the
            # batched-inference workspace; a runtime OOM still falls back to CPU.
            ('large-v3-turbo','float16'):       1.8,
            ('large-v3-turbo','int8_float16'):  1.0,
            ('medium',        'float16'):       1.6,
            ('medium',        'int8_float16'):  0.85,
            ('small',         'float16'):       1.0,
            ('small',         'int8_float16'):  0.55,
            ('base',          'float16'):       0.4,
            ('base',          'int8_float16'):  0.25,
            ('tiny',          'float16'):       0.2,
            ('tiny',          'int8_float16'):  0.15,
        }
        _gpu_free_gb = None
        _gpu_total_gb = None
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _free_bytes, _total_bytes = _torch.cuda.mem_get_info()
                _gpu_free_gb = _free_bytes / 1_073_741_824
                _gpu_total_gb = _total_bytes / 1_073_741_824
        except Exception:
            pass

        # On small GPUs (≤4 GB cards like the GTX 1650) prefer
        # ``int8_float16`` over plain ``float16`` so models from medium
        # upward actually fit. Larger cards stick with fp16 for maximum
        # decoder throughput — the int8 path is a touch slower per token.
        _small_gpu = (_gpu_total_gb is not None and _gpu_total_gb < 5.5)
        if _small_gpu:
            self._batch_size = 4   # smaller workspace footprint
            _tiers = [
                ('CUDA int8_float16', 'cuda', 'int8_float16'),
                ('CUDA float16',      'cuda', 'float16'),
                ('CUDA int8',         'cuda', 'int8'),
                ('CPU int8',          'cpu',  'int8'),
            ]
        else:
            self._batch_size = 16
            _tiers = [
                ('CUDA fp16',         'cuda', 'float16'),
                ('CUDA int8_float16', 'cuda', 'int8_float16'),
                ('CUDA int8',         'cuda', 'int8'),
                ('CPU int8',          'cpu',  'int8'),
            ]

        # ── Try the chosen ordered list of (device, compute_type) tiers ──
        for tier, device, compute in _tiers:
            if device == 'cuda' and not cuda_available:
                continue
            # Skip the CUDA tier up front when there's no headroom —
            # otherwise the load succeeds, sits in VRAM, and the later
            # batched-inference workspace check forces a CPU fallback
            # while the GPU copy lingers.
            if device == 'cuda' and _gpu_free_gb is not None:
                load_gb = _vram_load_gb.get(
                    (self.model_name, compute), 1.5 if compute != 'int8' else 0.8)
                # Workspace is roughly 1.0× model size under batch=16,
                # 0.4× under batch=4. Add a small constant for ctranslate2
                # scratch buffers so we don't squeak through and OOM mid-run.
                _ws_mult = 0.45 if self._batch_size <= 4 else 1.0
                _budget = load_gb + load_gb * _ws_mult + 0.25
                if _gpu_free_gb < _budget:
                    log.log_stage('AUDIO',
                        f'Skipping {tier}: only {_gpu_free_gb:.2f} GB VRAM free '
                        f'(need ~{_budget:.2f} GB for {self.model_name} '
                        f'@ batch={self._batch_size})')
                    continue
            try:
                log.log_stage('AUDIO',
                    f'Loading {self.model_name} on {tier} (batch={self._batch_size})...')
                self.engine = WhisperModel(
                    self.model_name, device=device, compute_type=compute)
                self.available = True
                self.device_used = f'{device}_{compute}'
                # Cache for reuse
                AudioIntelligence._cached_engine = self.engine
                AudioIntelligence._cached_model_name = self.model_name
                AudioIntelligence._cached_device = self.device_used
                AudioIntelligence._record_loaded(self.model_name, self.device_used)
                log.log_stage('AUDIO', f'Whisper ready: {tier}')
                # The user's requested model loaded — honor it (Task 4). If we
                # only fit it by falling to CPU while a GPU was available, say so
                # explicitly: the model is unchanged, but it'll be slower.
                if device == 'cpu' and cuda_available:
                    log.log_stage('AUDIO',
                        f'NOTE: requested {self.requested_model_name} loaded on CPU '
                        f'(only {_gpu_free_gb if _gpu_free_gb is not None else 0:.2f} GB VRAM free) '
                        f'— honoring your model choice; transcription will be slower')
                return True
            except Exception as e:
                log.log_stage('AUDIO', f'{tier} FAILED: {type(e).__name__}: {str(e)[:150]}')
                # Free anything ctranslate2 mapped before raising — otherwise
                # the next tier picks up the same fragmented allocator state.
                try:
                    if getattr(self, 'engine', None) is not None:
                        del self.engine
                        self.engine = None
                    import gc as _gc
                    _gc.collect()
                    if device == 'cuda':
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                except Exception:
                    pass

        # Last resort: try 'base' model on CPU. This is the ONLY path that
        # actually changes the model the user asked for, so log it as an
        # explicit, accurate downgrade (Task 4) — never silently report a model
        # other than what loaded.
        if self.model_name != 'base':
            try:
                _free_txt = (f'{_gpu_free_gb:.2f} GB VRAM free'
                             if _gpu_free_gb is not None else 'no GPU')
                log.log_stage('AUDIO',
                    f'DOWNGRADE: requested {self.requested_model_name} would not '
                    f'load on any tier ({_free_txt}) → loading base on CPU')
                self.engine = WhisperModel('base', device='cpu', compute_type='int8')
                self.available = True
                self.model_name = 'base'
                self.device_used = 'cpu_int8_base'
                AudioIntelligence._cached_engine = self.engine
                AudioIntelligence._cached_model_name = 'base'
                AudioIntelligence._cached_device = self.device_used
                AudioIntelligence._record_loaded('base', self.device_used)
                log.log_stage('AUDIO',
                    f'Whisper ready: CPU int8 base (effective model=base, '
                    f'requested={self.requested_model_name})')
                return True
            except Exception as e:
                log.log_stage('AUDIO', f'Base model FAILED: {e}')

        log.log_stage('AUDIO', 'ALL Whisper tiers failed')
        return False

    def transcribe(self, video_path: str, duration_ms: int,
                    language: str = 'auto',
                    on_progress: Callable = None) -> dict:
        """Transcribe the video's audio track.
        Optimized for speed: beam_size=1 (greedy), larger batch_size."""
        if not self.available:
            return {'speech_active': {}, 'segments': [], 'language': ''}

        log = get_logger()
        log.start_timer('transcribe')

        whisper_lang = None if language in ('auto', '', None) else language

        try:
            import tempfile
            audio_path = tempfile.mktemp(suffix='.wav')
            # Audio preconditioning for the ASR input. This path re-extracts
            # the audio straight from the video for Whisper, so the denoise /
            # loudness-normalise chain frame_extractor.extract_audio applies to
            # the diarization/music copy was NOT reaching the transcriber —
            # faint dialogue under music/SFX was being dropped by VAD as a
            # result. Mirror the SAME chain here, gated by the documented
            # WHISPER_AUDIO_PRECONDITION setting:
            #   highpass=f=80    — strip AC hum / rumble below 80 Hz
            #   afftdn=nf=-25    — adaptive FFT broadband denoise
            #   loudnorm=...     — bring quiet speech up to a consistent level
            _precondition = bool(getattr(settings, "WHISPER_AUDIO_PRECONDITION", True))
            log.log_stage('AUDIO',
                f'Extracting audio from {os.path.basename(video_path)}'
                f' (preconditioning {"on" if _precondition else "off"})...')
            _extract_cmd = ['ffmpeg', '-y', '-i', video_path, '-vn']
            if _precondition:
                _extract_cmd += ['-af',
                    'highpass=f=80,afftdn=nf=-25,loudnorm=I=-18:LRA=11:TP=-1.5']
            _extract_cmd += ['-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                             audio_path]
            result = subprocess.run(_extract_cmd, capture_output=True, timeout=240)

            if not os.path.exists(audio_path):
                log.log_error('AUDIO', f'Audio extraction failed (exit={result.returncode})')
                # Preconditioning can fail on exotic codecs / filter builds;
                # retry once with a plain copy so a filter error never costs us
                # the whole transcript.
                if _precondition:
                    log.log_stage('AUDIO',
                        'Preconditioned extract failed — retrying without filters')
                    subprocess.run([
                        'ffmpeg', '-y', '-i', video_path,
                        '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                        audio_path,
                    ], capture_output=True, timeout=180)
                if not os.path.exists(audio_path):
                    return {'speech_active': {}, 'segments': [], 'language': ''}

            audio_size = os.path.getsize(audio_path)
            log.log_stage('AUDIO', f'Audio extracted: {audio_size/1048576:.1f} MB')

            # Pre-flight: the GPU model is already loaded — only the
            # batched-inference workspace is left to allocate. Reload on
            # CPU only when the remaining headroom is below that workspace
            # footprint (load_gb × ws_mult + scratch). Keeps us on GPU on
            # tight cards like the GTX 1650 while still aborting cleanly
            # when the budget really is too small.
            if self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _free_bytes, _ = _torch.cuda.mem_get_info()
                        _free_gb = _free_bytes / 1_073_741_824
                        _compute = self.device_used.split('_', 1)[1] if '_' in self.device_used else 'float16'
                        # Rough workspace budgets — match the load-time
                        # numbers in try_load() so the two checks agree.
                        _ws_load = {
                            ('large',    'float16'):      3.0,
                            ('large',    'int8_float16'): 1.6,
                            ('large-v2', 'float16'):      3.0,
                            ('large-v2', 'int8_float16'): 1.6,
                            ('large-v3', 'float16'):      3.0,
                            ('large-v3', 'int8_float16'): 1.6,
                            ('large-v3-turbo', 'float16'):      1.8,
                            ('large-v3-turbo', 'int8_float16'): 1.0,
                            ('medium',   'float16'):      1.6,
                            ('medium',   'int8_float16'): 0.85,
                            ('small',    'float16'):      1.0,
                            ('small',    'int8_float16'): 0.55,
                            ('base',     'float16'):      0.4,
                            ('base',     'int8_float16'): 0.25,
                            ('tiny',     'float16'):      0.2,
                            ('tiny',     'int8_float16'): 0.15,
                        }.get((self.model_name, _compute), 1.5)
                        _ws_mult = 0.45 if self._batch_size <= 4 else 1.0
                        _needed = _ws_load * _ws_mult + 0.15
                        if _free_gb < _needed:
                            log.log_stage('AUDIO',
                                f'Only {_free_gb:.2f} GB VRAM free '
                                f'(need ~{_needed:.2f} GB workspace for '
                                f'{self.model_name} {_compute} @ batch={self._batch_size}) '
                                '— using CPU to avoid OOM')
                            self._reload_on_cpu()
                except Exception:
                    pass

            log.log_stage('AUDIO', f'Transcribing with {self.model_name} ({self.device_used})...'
                           f' language={whisper_lang or "auto"} batch={self._batch_size}')

            # Transcription parameters:
            #   beam_size=5 (better accuracy — captures ~5-8% more words than greedy)
            #   batch_size=self._batch_size (4 on ≤4GB GPUs, 16 elsewhere)
            #   vad min_silence=300ms (catches brief pauses within sentences)
            #   no_speech_threshold from settings.WHISPER_NO_SPEECH_THRESHOLD
            #       (lower = less likely to skip quiet speech)
            #   anti-repetition decoding via _decoding_kwargs(): cond_prev OFF
            #       by default + no_repeat_ngram_size / compression-ratio +
            #       log-prob thresholds + temperature fallback (rejects loops)
            #   word_timestamps=True (per-word timing for subtitle + reframing)
            _ns_threshold = float(getattr(
                settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.4))
            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=self.engine)
                _bias = _vocab_bias_kwargs(batched.transcribe, language)
                _decode = _decoding_kwargs(batched.transcribe)
                segments_iter, info = batched.transcribe(
                    audio_path, batch_size=self._batch_size,
                    language=whisper_lang,
                    beam_size=5, vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 200,
                    },
                    word_timestamps=True,
                    no_speech_threshold=_ns_threshold,
                    **_decode,
                    **_bias,
                )
                log.log_stage('AUDIO',
                    f'Using batched inference (batch=16, beam=5, '
                    f'no_speech_thresh={_ns_threshold}, '
                    f'cond_prev={_decode.get("condition_on_previous_text")}, '
                    f'no_repeat_ngram={_decode.get("no_repeat_ngram_size")})')
            except Exception as e:
                err_str = str(e)
                log.log_stage('AUDIO', f'Batched inference failed: {err_str[:120]}')
                # CUDA out-of-memory or any other CUDA error — reload on CPU
                # so the sequential fallback below doesn't hit the same wall.
                if self.device_used.startswith('cuda') and (
                    'out of memory' in err_str.lower()
                    or 'cublas' in err_str.lower()
                    or 'cuda' in err_str.lower()
                ):
                    log.log_stage('AUDIO',
                        'CUDA out of memory — reloading Whisper on CPU')
                    self._reload_on_cpu()
                log.log_stage('AUDIO', 'Falling back to sequential transcription (slower)')
                _bias = _vocab_bias_kwargs(self.engine.transcribe, language)
                _decode = _decoding_kwargs(self.engine.transcribe)
                segments_iter, info = self.engine.transcribe(
                    audio_path, language=whisper_lang,
                    beam_size=5, vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 200,
                    },
                    word_timestamps=True,
                    no_speech_threshold=_ns_threshold,
                    **_decode,
                    **_bias,
                )

            duration_sec = float(getattr(info, 'duration', 0)) or duration_ms / 1000
            language = getattr(info, 'language', 'unknown')

            segments = []
            speech_active = {}
            last_log_pct = 0
            transcribe_start = _time.monotonic()

            # ── TACT: Hallucination detection constants ──
            # Boilerplate phrases live in the module-level
            # _BOILERPLATE_HALLUCINATIONS (multilingual). See _is_boilerplate_hallucination.

            for seg in segments_iter:
                start_ms = int(seg.start * 1000)
                end_ms = int(seg.end * 1000)

                text = seg.text.strip()
                no_speech_prob = getattr(seg, 'no_speech_prob', 0.0) or 0.0

                # ── TACT: Hallucination filter ──
                is_hallucination = False

                # 1. Boilerplate blocklist (multilingual)
                if _is_boilerplate_hallucination(text):
                    is_hallucination = True

                # 2. High no_speech_prob + text present → likely hallucination
                if no_speech_prob > 0.7 and text:
                    is_hallucination = True

                # 3. Repetition detection: same phrase repeating 3+ times
                if text and len(text) > 20:
                    words_list = text.lower().split()
                    if len(words_list) >= 6:
                        # Check for repeating patterns of 2-4 words
                        for plen in range(2, 5):
                            if len(words_list) >= plen * 3:
                                pattern = tuple(words_list[:plen])
                                repeats = 0
                                for i in range(0, len(words_list) - plen + 1, plen):
                                    if tuple(words_list[i:i + plen]) == pattern:
                                        repeats += 1
                                if repeats >= 3:
                                    is_hallucination = True
                                    break

                # Extract word-level timestamps if available
                words = []
                if hasattr(seg, 'words') and seg.words:
                    for w in seg.words:
                        word_conf = getattr(w, 'probability', None)
                        if word_conf is None:
                            word_conf = getattr(w, 'confidence', 1.0) or 1.0
                        words.append({
                            'word': w.word.strip() if hasattr(w, 'word') else str(w).strip(),
                            'start': round(getattr(w, 'start', seg.start), 3),
                            'end': round(getattr(w, 'end', seg.end), 3),
                            'confidence': round(float(word_conf), 3),
                        })

                seg_entry = {
                    'start_sec': round(seg.start, 3),
                    'end_sec': round(seg.end, 3),
                    'text': text,
                    'words': words,
                    'is_hallucination': is_hallucination,
                    'no_speech_prob': round(no_speech_prob, 3),
                }
                segments.append(seg_entry)

                # Align to 100ms grid for speech_active (backwards compat)
                if not is_hallucination:
                    aligned_start = (start_ms // 100) * 100
                    aligned_end = end_ms + 100
                    for t in range(aligned_start, aligned_end, 100):
                        speech_active[t] = True

                if on_progress and duration_sec > 0:
                    on_progress(min(1.0, seg.end / duration_sec))

                if duration_sec > 0:
                    pct = int(seg.end / duration_sec * 100)
                    if pct >= last_log_pct + 25:
                        elapsed_so_far = _time.monotonic() - transcribe_start
                        remaining = (elapsed_so_far / max(0.01, seg.end / duration_sec)) - elapsed_so_far
                        log.log_stage('AUDIO',
                            f'  Transcribing {pct}% — {len(segments)} segments, '
                            f'{elapsed_so_far:.1f}s elapsed, ~{remaining:.0f}s remaining')
                        last_log_pct = pct

            # ── Gap-fill pass: re-transcribe uncovered runs ──
            # The VAD filter + ``no_speech_threshold`` on the main pass
            # silently drop quiet / soft / off-mic / sung speech. The
            # YouTube vs ClipAI comparison on the GUNDAM Wing episode 1
            # showed ClipAI's transcript starting at ~1:59 while YouTube
            # had dialogue / lyrics from 0:26 — 90 seconds of audio
            # that VAD classified as silence. Find the uncovered runs
            # in the segments list and re-transcribe just those
            # regions with no VAD and a very low no-speech threshold
            # so quiet speech gets a second chance. Gap-fill segments
            # are tagged ``source='gap_fill'`` and routed through the
            # same hallucination filter — the relaxed thresholds only
            # apply inside the gaps where the main pass already
            # produced nothing.
            if (getattr(settings, "WHISPER_GAP_FILL_ENABLED", True)
                    and segments and duration_sec > 0):
                try:
                    gap_segments = self._gap_fill_pass(
                        audio_path, segments, duration_sec, whisper_lang, log)
                    if gap_segments:
                        # Merge gap-fill segments and re-sort by start time
                        # so the Coverage Ledger / per-segment loops see
                        # them in temporal order.
                        segments.extend(gap_segments)
                        segments.sort(key=lambda s: s.get('start_sec', 0))
                        # Drop repetition-loop hallucinations the gap-fill
                        # pass produces over music / quiet regions (the same
                        # line emitted dozens of times across the timeline).
                        _pre_dedup = len(segments)
                        segments = _drop_repetition_loops(segments)
                        log.log_stage('AUDIO',
                            f'Gap-fill added {len(gap_segments)} segments '
                            f'(total {_pre_dedup}, {len(segments)} after '
                            f'repetition-loop filter)')
                except Exception as gf_err:
                    log.log_stage('AUDIO',
                        f'Gap-fill pass failed (non-fatal): {gf_err}')

            # ── Clamp timestamp drift to the real audio end ──
            # faster-whisper drifts/loops on repetitive music and can stamp cues
            # PAST the audio (e.g. cues at 35:51 on a 24:27 video), mis-timing
            # the back third of the subtitles. Those past-the-end cues are
            # loop-repeats of earlier lines, not real tail dialogue, so drop them
            # (and clamp any cue that merely overruns the end). Clamp to the
            # larger of info.duration and the caller's media duration so a short
            # info.duration can never truncate legitimate tail speech; a clean
            # run with no drift is a no-op.
            try:
                from backend.services.transcript_dedup import clamp_segments_to_duration
                _clamp_limit = max(float(duration_sec), (duration_ms or 0) / 1000.0)
                segments, _drifted = clamp_segments_to_duration(
                    segments, _clamp_limit, start_key='start_sec', end_key='end_sec')
                if _drifted:
                    log.log_stage('AUDIO',
                        f'Timestamp-drift clamp @ {_clamp_limit:.0f}s audio end: '
                        f'dropped/clamped {_drifted} cue(s) past the end')
            except Exception as _clamp_err:
                log.log_stage('AUDIO',
                    f'Timestamp-drift clamp skipped (non-fatal): {_clamp_err}')

            # ── TACT: Build Coverage Ledger ──
            ledger = CoverageLedger(bin_width_ms=20, duration_ms=int(duration_sec * 1000))

            # Initialize all bins as uncovered
            for t in range(0, int(duration_sec * 1000), 20):
                ledger.bins[t] = LedgerBin(status='uncovered')

            # Claim bins from segments
            hallucinated_count = 0
            low_conf_count = 0
            for seg_entry in segments:
                seg_start = int(seg_entry['start_sec'] * 1000)
                seg_end = int(seg_entry['end_sec'] * 1000)

                if seg_entry.get('is_hallucination'):
                    # Quarantine hallucinated bins
                    hallucinated_count += 1
                    for t in range(seg_start, seg_end, 20):
                        if t in ledger.bins:
                            ledger.bins[t] = LedgerBin(
                                status='quarantined',
                                text=seg_entry['text'][:30],
                                source='whisper_primary',
                                confidence=0.0,
                            )
                    continue

                # Word-level bin claiming (more precise)
                if seg_entry.get('words'):
                    for w in seg_entry['words']:
                        w_start = int(w['start'] * 1000)
                        w_end = int(w['end'] * 1000)
                        conf = w.get('confidence', 1.0)
                        status = 'covered_speech' if conf >= 0.4 else 'low_confidence'
                        if conf < 0.4:
                            low_conf_count += 1
                        for t in range(w_start, w_end, 20):
                            if t in ledger.bins:
                                ledger.bins[t] = LedgerBin(
                                    status=status,
                                    text=w.get('word', ''),
                                    source='whisper_primary',
                                    confidence=conf,
                                )
                else:
                    # Segment-level claiming (no word timestamps)
                    for t in range(seg_start, seg_end, 20):
                        if t in ledger.bins:
                            ledger.bins[t] = LedgerBin(
                                status='covered_speech',
                                text=seg_entry['text'][:20],
                                source='whisper_primary',
                                confidence=0.7,
                            )

            # Label remaining uncovered bins as silence
            silence_count = 0
            for t, b in ledger.bins.items():
                if b.status == 'uncovered':
                    ledger.bins[t] = LedgerBin(
                        status='covered_silence',
                        source='inferred',
                        confidence=0.8,
                        event_type='silence',
                    )
                    silence_count += 1

            # Build audio_events from high no_speech_prob segments
            # (lightweight event inference — YAMNet integration is Phase 4)
            audio_events = {}
            for seg_entry in segments:
                if seg_entry.get('no_speech_prob', 0) > 0.6 and not seg_entry.get('is_hallucination'):
                    seg_start = int(seg_entry['start_sec'] * 1000)
                    seg_end = int(seg_entry['end_sec'] * 1000)
                    for t in range(seg_start, seg_end, 100):
                        audio_events[t] = 'noise'
                        # Update ledger bin event_type
                        bin_key = (t // 20) * 20
                        if bin_key in ledger.bins:
                            ledger.bins[bin_key].event_type = 'noise'

            report = ledger.coverage_report()

            try:
                os.remove(audio_path)
            except Exception:
                pass

            elapsed = log.stop_timer('transcribe')
            speed_ratio = duration_sec / max(0.01, elapsed)

            # Filter hallucinations from the segment list for downstream consumers
            clean_segments = [s for s in segments if not s.get('is_hallucination')]

            # ── Cross-segment hallucination validation ──
            # Remove exact duplicates, temporal overlaps, and >80% word-overlap
            # repeats that slip past the per-segment TACT filter above.
            clean_segments = _cross_validate_segments(clean_segments)

            log.log_stage('AUDIO',
                f'Transcription complete: {len(clean_segments)} segments '
                f'({hallucinated_count} hallucinations quarantined, '
                f'{low_conf_count} low-confidence words), '
                f'language={language}, {elapsed:.1f}s ({speed_ratio:.1f}x realtime)',
                speech_seconds=round(len(speech_active) * 0.1, 1),
                coverage=f'{report["coverage_ratio"]:.1%}')

            if on_progress:
                on_progress(1.0)

            return {
                'speech_active': speech_active,
                'segments': clean_segments,
                'language': language,
                'coverage_ledger': ledger,
                'audio_events': audio_events,
            }

        except Exception as e:
            log.log_error('AUDIO', f'Transcription failed: {e}')
            return {'speech_active': {}, 'segments': [], 'language': ''}

    def _gap_fill_pass(
        self, audio_path: str, primary_segments: list,
        duration_sec: float, whisper_lang, log,
    ) -> list:
        """Re-transcribe runs of audio the main pass left uncovered.

        Whisper-medium with VAD enabled silently drops:
          * soft / off-mic / whispered speech (VAD ``no_speech_prob``
            crosses threshold)
          * sung audio in opening / ending themes (Whisper trained
            mostly on spoken audio; high no_speech_prob on lyrics)
          * brief utterances under the 300 ms VAD silence break

        This helper finds runs of source audio ≥
        ``WHISPER_GAP_FILL_MIN_SEC`` seconds where the main pass
        produced no segment, then re-transcribes just those runs
        with:
          * ``vad_filter=False`` — VAD already said this was silence,
            don't ask it again
          * ``no_speech_threshold`` from
            ``WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD`` (default 0.25)
          * ``condition_on_previous_text=False`` — no priming from
            the (probably very different) main pass

        Each resulting segment is run through the same hallucination
        filter the main pass uses (boilerplate blocklist, repetition
        detection, no_speech_prob clamp) before being merged. Returns
        the list of clean gap-fill segment dicts ready to merge into
        the main ``segments`` list.
        """
        min_gap = float(getattr(settings, "WHISPER_GAP_FILL_MIN_SEC", 3.0))
        gap_ns_thresh = float(getattr(
            settings, "WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD", 0.25))

        # Build the list of (start_sec, end_sec) gaps to re-transcribe.
        # Only consider clean (non-hallucination) segments as 'covered'.
        covered = sorted(
            (float(s['start_sec']), float(s['end_sec']))
            for s in primary_segments
            if not s.get('is_hallucination')
        )
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in covered:
            if start - cursor >= min_gap:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if duration_sec - cursor >= min_gap:
            gaps.append((cursor, duration_sec))

        if not gaps:
            log.log_stage('AUDIO',
                'Gap-fill: no gaps ≥ %.1fs — skipping' % min_gap)
            return []

        total_gap_sec = sum(e - s for s, e in gaps)
        log.log_stage('AUDIO',
            f'Gap-fill: re-transcribing {len(gaps)} run(s) '
            f'totalling {total_gap_sec:.1f}s '
            f'(threshold={gap_ns_thresh})')

        # ``clip_timestamps`` accepts a flat list of seconds; pairs are
        # interpreted as (start, end). faster-whisper transcribes ONLY
        # those ranges and skips everything else, so the gap-fill pass
        # doesn't waste compute on the already-covered audio.
        clip_ts: list[float] = []
        for s, e in gaps:
            clip_ts.append(round(s, 3))
            clip_ts.append(round(e, 3))

        # Gap-fill lives in music / quiet regions, so force priming OFF
        # regardless of the global condition_on_previous_text default, and add
        # the same anti-repetition decoding guards the main pass uses.
        _gap_decode = _decoding_kwargs(
            self.engine.transcribe, condition_on_previous_text=False)
        try:
            segs_iter, _info = self.engine.transcribe(
                audio_path, language=whisper_lang,
                beam_size=5, vad_filter=False,
                word_timestamps=True,
                no_speech_threshold=gap_ns_thresh,
                clip_timestamps=clip_ts,
                **_gap_decode,
            )
        except TypeError:
            # Older faster-whisper builds don't accept clip_timestamps.
            # Fall back to a full-file second pass with the relaxed
            # threshold and dedupe overlaps below.
            log.log_stage('AUDIO',
                'Gap-fill: clip_timestamps unsupported — full-file fallback')
            segs_iter, _info = self.engine.transcribe(
                audio_path, language=whisper_lang,
                beam_size=5, vad_filter=False,
                word_timestamps=True,
                no_speech_threshold=gap_ns_thresh,
                **_gap_decode,
            )

        # Same boilerplate / repetition filter the main pass uses
        # (multilingual — see module-level _is_boilerplate_hallucination).

        # Build a fast O(log n) membership check for "is this gap-fill
        # segment actually inside a real gap?" — used to drop any
        # segment that leaked into already-covered territory on the
        # full-file fallback path.
        from bisect import bisect_right
        gap_starts = [g[0] for g in gaps]

        def _inside_gap(start_sec: float, end_sec: float) -> bool:
            i = bisect_right(gap_starts, start_sec) - 1
            if i < 0:
                return False
            gs, ge = gaps[i]
            # Allow segments that mostly overlap the gap (≥50 %).
            seg_len = max(0.001, end_sec - start_sec)
            overlap = max(0.0, min(end_sec, ge) - max(start_sec, gs))
            return overlap / seg_len >= 0.5

        # Hard ceiling on gap-fill segment length. With ``vad_filter=False``
        # Whisper can fold a whole OP song / minutes of narration into ONE
        # run-on cue stamped at a single timestamp; splitting at word
        # boundaries (and at inter-word silences) keeps each emitted cue
        # short and time-accurate.
        max_gap_seg = float(getattr(settings, "WHISPER_GAP_FILL_MAX_SEC", 8.0))
        GAP_BREAK = 1.0  # seconds of silence that forces a split

        def _word_dict(w):
            word_conf = (getattr(w, 'probability', None)
                         or getattr(w, 'confidence', 1.0) or 1.0)
            return {
                'word': (w.word.strip() if hasattr(w, 'word')
                         else str(w).strip()),
                'start': round(float(getattr(w, 'start', 0.0) or 0.0), 3),
                'end': round(float(getattr(w, 'end', 0.0) or 0.0), 3),
                'confidence': round(float(word_conf), 3),
            }

        out: list[dict] = []
        dropped_outside = 0
        for seg in segs_iter:
            text = (seg.text or '').strip()
            if not text:
                continue
            no_speech_prob = float(getattr(seg, 'no_speech_prob', 0.0) or 0.0)

            # Drop common Whisper hallucinations (multilingual)
            if _is_boilerplate_hallucination(text):
                continue
            # Very high no_speech confidence is true silence
            if no_speech_prob > 0.85:
                continue
            # Repetition check (lighter than the main pass — gap-fill
            # already lives in low-confidence territory)
            if len(text) > 20:
                words_list = text.lower().split()
                if len(words_list) >= 6:
                    for plen in range(2, 5):
                        if len(words_list) >= plen * 3:
                            pattern = tuple(words_list[:plen])
                            repeats = sum(
                                1 for i in range(0, len(words_list) - plen + 1, plen)
                                if tuple(words_list[i:i + plen]) == pattern
                            )
                            if repeats >= 3:
                                text = None
                                break
                    if text is None:
                        continue

            seg_start = float(seg.start)
            seg_end = float(seg.end)
            # Overlap guard — applied on BOTH the clip_timestamps path and the
            # full-file fallback path, so a segment that leaked into
            # already-covered territory is dropped regardless of which
            # transcribe call produced it.
            if not _inside_gap(seg_start, seg_end):
                dropped_outside += 1
                continue

            raw_words = list(seg.words) if (hasattr(seg, 'words') and seg.words) else []

            # ── Split over-long run-on cues into ≤ max_gap_seg sub-cues ──
            pieces: list[list] = []
            if max_gap_seg > 0 and (seg_end - seg_start) > max_gap_seg and raw_words:
                cur: list = []
                cur_start = None
                prev_end = None
                for w in raw_words:
                    w_start = float(getattr(w, 'start', seg_start) or seg_start)
                    w_end = float(getattr(w, 'end', w_start) or w_start)
                    if cur and (
                        (w_end - cur_start > max_gap_seg)
                        or (prev_end is not None and w_start - prev_end >= GAP_BREAK)
                    ):
                        pieces.append(cur)
                        cur = []
                        cur_start = None
                    if cur_start is None:
                        cur_start = w_start
                    cur.append(w)
                    prev_end = w_end
                if cur:
                    pieces.append(cur)

            if len(pieces) > 1:
                for ch in pieces:
                    # Reconstruct chunk text from the RAW word tokens so the
                    # original spacing (latin) / non-spacing (CJK) survives.
                    c_text = ''.join(
                        (w.word if hasattr(w, 'word') else str(w)) for w in ch
                    ).strip()
                    if not c_text:
                        continue
                    c_start = round(float(getattr(ch[0], 'start', seg_start) or seg_start), 3)
                    c_end = round(float(getattr(ch[-1], 'end', c_start) or c_start), 3)
                    out.append({
                        'start_sec': c_start,
                        'end_sec': c_end,
                        'text': c_text,
                        'words': [_word_dict(w) for w in ch],
                        'is_hallucination': False,
                        'no_speech_prob': round(no_speech_prob, 3),
                        'source': 'gap_fill',
                    })
            else:
                out.append({
                    'start_sec': round(seg_start, 3),
                    'end_sec': round(seg_end, 3),
                    'text': text,
                    'words': [_word_dict(w) for w in raw_words],
                    'is_hallucination': False,
                    'no_speech_prob': round(no_speech_prob, 3),
                    'source': 'gap_fill',
                })

        # Collapse near-duplicate re-transcriptions (overlapping in time AND
        # similar in text — e.g. the repeated 作戦名オペレーション・メテオ) that the
        # adjacent-only filters miss.
        from backend.services.transcript_dedup import collapse_overlapping_duplicates
        out, dropped_dup = collapse_overlapping_duplicates(
            out, start_key='start_sec', end_key='end_sec')

        max_seg_len = max((s['end_sec'] - s['start_sec'] for s in out), default=0.0)
        log.log_stage('AUDIO',
            f'Gap-fill output: {len(out)} segment(s) kept, '
            f'{dropped_outside} dropped-outside-gap, '
            f'{dropped_dup} dropped-as-duplicate, '
            f'max segment length {max_seg_len:.1f}s')
        return out

    def _reload_on_cpu(self) -> None:
        """Reload Whisper on CPU int8 — used when CUDA runs out of memory.
        Low-VRAM GPUs (the GTX 1650 = 4 GB, for instance) can't fit Whisper
        medium under batched inference; CPU is slower but always completes."""
        # The previously loaded GPU engine is referenced from BOTH ``self``
        # and the class-level cache. Dropping only ``self.engine`` leaves
        # the cache alive, so ``empty_cache()`` finds nothing to free and
        # the GPU model lingers for the rest of the process. Clear both
        # before flushing the allocator.
        try:
            import gc as _gc
            AudioIntelligence._cached_engine = None
            AudioIntelligence._cached_model_name = None
            AudioIntelligence._cached_device = None
            if hasattr(self, 'engine'):
                try:
                    del self.engine
                except Exception:
                    pass
                self.engine = None
            _gc.collect()
            _gc.collect()
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass
        from faster_whisper import WhisperModel
        self.engine = WhisperModel(self.model_name, device='cpu', compute_type='int8')
        self.device_used = 'cpu_int8'
        # Re-cache the CPU engine so the next Perceiver in this process
        # doesn't load yet another copy on top of the GPU one we just freed.
        AudioIntelligence._cached_engine = self.engine
        AudioIntelligence._cached_model_name = self.model_name
        AudioIntelligence._cached_device = self.device_used
        AudioIntelligence._record_loaded(self.model_name, self.device_used)
        log.log_stage('AUDIO', f'Whisper reloaded on CPU int8 ({self.model_name})')

    def whisper_translate(self, video_path: str, source_lang: str = None,
                          on_progress=None, reuse_loaded: bool = False) -> List[dict]:
        """Direct audio→English translation via Whisper's native translate task.

        For non-English → English, this single-step pass (Whisper run with
        ``task="translate"``) is the preferred OFFLINE path: it avoids the
        transcribe-then-translate double-error, needs no separate NMT model
        download, and never touches an LLM (the AI model only polishes the
        result afterward). The pipeline calls this first for English targets
        and falls back to the offline NMT engines in
        ``backend/services/translator.py`` only when it returns ``[]``.

        Returns a list of ``{start, end, text}`` dicts (English, with Whisper's
        own audio-aligned timing), or ``[]`` when the engine is unavailable.
        """
        log = get_logger()

        if not self.engine:
            log.log_stage('TRANSLATE',
                'Whisper not loaded — cannot use native translate')
            return []

        # Extract audio
        import subprocess as _sp
        import tempfile as _tf
        audio_path = os.path.join(_tf.gettempdir(), 'clipai_translate_audio.wav')
        try:
            _sp.run(['ffmpeg', '-y', '-i', video_path,
                     '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                     audio_path],
                    capture_output=True, timeout=120)
        except Exception as e:
            log.log_error('TRANSLATE', f'Audio extraction failed: {e}')
            return []

        log.log_stage('TRANSLATE',
            f'Whisper native translate: {source_lang or "auto"} → en '
            f'(direct audio→English, single-step)')

        try:
            # Use the same model but with task="translate"
            whisper_lang = source_lang if source_lang and source_lang != 'auto' else None

            # Pre-flight VRAM check — mirrors transcribe(). On low-VRAM GPUs,
            # batched translate OOMs just as readily as batched transcribe.
            # When REUSING an already-loaded model, no reload headroom is needed
            # (the weights are already resident) — only inference activations —
            # so allow a lower free-VRAM floor before dropping to the slow CPU
            # path. This is what lets a 4 GB card translate on the GPU via reuse;
            # a genuine OOM below still falls back to CPU cleanly.
            if self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _free_bytes, _ = _torch.cuda.mem_get_info()
                        _free_gb = _free_bytes / 1_073_741_824
                        _floor = 1.5 if reuse_loaded else 3.0
                        if _free_gb < _floor:
                            log.log_stage('TRANSLATE',
                                f'Only {_free_gb:.1f} GB VRAM free (< {_floor:.1f}) '
                                '— using CPU for translation')
                            self._reload_on_cpu()
                except Exception:
                    pass

            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=self.engine)
                _decode = _decoding_kwargs(batched.transcribe)
                segments_iter, info = batched.transcribe(
                    audio_path, batch_size=self._batch_size,
                    language=whisper_lang,
                    task='translate',  # ← the key difference
                    beam_size=5, vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 200,
                    },
                    word_timestamps=True,
                    no_speech_threshold=float(getattr(
                        settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.4)),
                    **_decode,
                )
            except Exception as e:
                err_str = str(e)
                if self.device_used.startswith('cuda') and (
                    'out of memory' in err_str.lower()
                    or 'cublas' in err_str.lower()
                    or 'cuda' in err_str.lower()
                ):
                    log.log_stage('TRANSLATE',
                        'CUDA out of memory — reloading Whisper on CPU')
                    self._reload_on_cpu()
                # Fallback to sequential
                segments_iter, info = self.engine.transcribe(
                    audio_path, language=whisper_lang,
                    task='translate',
                    beam_size=5, vad_filter=True,
                    word_timestamps=True,
                    **_decoding_kwargs(self.engine.transcribe),
                )

            segments = []
            for seg in segments_iter:
                words = []
                if hasattr(seg, 'words') and seg.words:
                    for w in seg.words:
                        words.append({
                            'word': w.word.strip() if hasattr(w, 'word') else str(w).strip(),
                            'start': round(getattr(w, 'start', seg.start), 3),
                            'end': round(getattr(w, 'end', seg.end), 3),
                        })
                segments.append({
                    'start_sec': round(seg.start, 3),
                    'end_sec': round(seg.end, 3),
                    'text': seg.text.strip(),
                    'words': words,
                })

                if on_progress and info.duration > 0:
                    on_progress(min(1.0, seg.end / info.duration))

            # Clamp drift to the real audio end (task='translate' loops on music
            # too) so the English cues never run past the video.
            try:
                from backend.services.transcript_dedup import clamp_segments_to_duration
                if getattr(info, 'duration', 0):
                    segments, _drifted = clamp_segments_to_duration(
                        segments, float(info.duration),
                        start_key='start_sec', end_key='end_sec')
                    if _drifted:
                        log.log_stage('TRANSLATE',
                            f'Timestamp-drift clamp @ {info.duration:.0f}s: '
                            f'dropped/clamped {_drifted} cue(s)')
            except Exception:
                pass

            log.log_stage('TRANSLATE',
                f'Whisper translate complete: {len(segments)} segments '
                f'(direct {source_lang or "auto"} → en)')

            return segments

        except Exception as e:
            log.log_error('TRANSLATE', f'Whisper translate failed: {e}')
            return []
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass


