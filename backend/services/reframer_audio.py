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

logger = logging.getLogger("clipai.reframer_audio")


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

    def __init__(self, model_name: str = 'small'):
        self.available = False
        self.engine = None
        self.model_name = model_name
        self.device_used = 'unknown'

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

        # Check if we already have this model loaded (skip re-download)
        if (AudioIntelligence._cached_engine is not None
                and AudioIntelligence._cached_model_name == self.model_name):
            self.engine = AudioIntelligence._cached_engine
            self.device_used = AudioIntelligence._cached_device
            self.available = True
            log.log_stage('AUDIO',
                f'Whisper {self.model_name} already loaded ({self.device_used}) — using cached')
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

        # ── Try loading (CUDA float16 → CUDA int8 → CPU int8) ──
        for tier, device, compute in [
            ('CUDA fp16', 'cuda', 'float16'),
            ('CUDA int8', 'cuda', 'int8'),
            ('CPU int8', 'cpu', 'int8'),
        ]:
            if device == 'cuda' and not cuda_available:
                continue
            try:
                log.log_stage('AUDIO', f'Loading {self.model_name} on {tier}...')
                self.engine = WhisperModel(
                    self.model_name, device=device, compute_type=compute)
                self.available = True
                self.device_used = f'{device}_{compute}'
                # Cache for reuse
                AudioIntelligence._cached_engine = self.engine
                AudioIntelligence._cached_model_name = self.model_name
                AudioIntelligence._cached_device = self.device_used
                log.log_stage('AUDIO', f'Whisper ready: {tier}')
                return True
            except Exception as e:
                log.log_stage('AUDIO', f'{tier} FAILED: {type(e).__name__}: {str(e)[:150]}')

        # Last resort: try 'base' model on CPU
        if self.model_name != 'base':
            try:
                log.log_stage('AUDIO', 'Trying base model on CPU...')
                self.engine = WhisperModel('base', device='cpu', compute_type='int8')
                self.available = True
                self.model_name = 'base'
                self.device_used = 'cpu_int8_base'
                AudioIntelligence._cached_engine = self.engine
                AudioIntelligence._cached_model_name = 'base'
                AudioIntelligence._cached_device = self.device_used
                log.log_stage('AUDIO', 'Whisper ready: CPU int8 base (lowest quality)')
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
            log.log_stage('AUDIO', f'Extracting audio from {os.path.basename(video_path)}...')
            result = subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                audio_path
            ], capture_output=True, timeout=120)

            if not os.path.exists(audio_path):
                log.log_error('AUDIO', f'Audio extraction failed (exit={result.returncode})')
                return {'speech_active': {}, 'segments': [], 'language': ''}

            audio_size = os.path.getsize(audio_path)
            log.log_stage('AUDIO', f'Audio extracted: {audio_size/1048576:.1f} MB')

            # Pre-flight: low-VRAM GPUs (the GTX 1650 = 4 GB, etc.) can't fit
            # Whisper medium under batched inference. Reload on CPU before
            # transcribing so we don't burn an attempt on a doomed CUDA call.
            if self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _free_bytes, _ = _torch.cuda.mem_get_info()
                        _free_gb = _free_bytes / 1_073_741_824
                        _needed = {'large': 5.5, 'large-v2': 5.5, 'large-v3': 5.5,
                                   'medium': 3.0, 'small': 1.8,
                                   'base': 1.0, 'tiny': 0.6,
                                   }.get(self.model_name, 3.0)
                        if _free_gb < _needed:
                            log.log_stage('AUDIO',
                                f'Only {_free_gb:.1f} GB VRAM free '
                                f'(need ~{_needed:.1f} GB for {self.model_name}) '
                                '— using CPU to avoid OOM')
                            self._reload_on_cpu()
                except Exception:
                    pass

            log.log_stage('AUDIO', f'Transcribing with {self.model_name} ({self.device_used})...'
                           f' language={whisper_lang or "auto"}')

            # Transcription parameters:
            #   beam_size=5 (better accuracy — captures ~5-8% more words than greedy)
            #   batch_size=16 (reduced from 24 to compensate for beam memory)
            #   vad min_silence=300ms (catches brief pauses within sentences)
            #   no_speech_threshold=0.5 (lower = less likely to skip quiet speech)
            #   condition_on_previous_text=True (improves coherence across segments)
            #   word_timestamps=True (per-word timing for subtitle + reframing)
            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=self.engine)
                segments_iter, info = batched.transcribe(
                    audio_path, batch_size=16,
                    language=whisper_lang,
                    beam_size=5, vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 200,
                    },
                    word_timestamps=True,
                    condition_on_previous_text=True,
                    no_speech_threshold=0.5,
                )
                log.log_stage('AUDIO', 'Using batched inference (batch=16, beam=5)')
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
                segments_iter, info = self.engine.transcribe(
                    audio_path, language=whisper_lang,
                    beam_size=5, vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 200,
                    },
                    word_timestamps=True,
                    condition_on_previous_text=True,
                    no_speech_threshold=0.5,
                )

            duration_sec = float(getattr(info, 'duration', 0)) or duration_ms / 1000
            language = getattr(info, 'language', 'unknown')

            segments = []
            speech_active = {}
            last_log_pct = 0
            transcribe_start = _time.monotonic()

            # ── TACT: Hallucination detection constants ──
            _BOILERPLATE = {
                'thank you for watching', 'thanks for watching',
                'please subscribe', 'like and subscribe',
                "don't forget to subscribe", 'see you in the next video',
                'bye bye', 'thanks for listening', 'music playing',
                'music', 'applause', 'subtitles by', 'captions by',
                'thank you', 'thanks', 'the end',
            }

            for seg in segments_iter:
                start_ms = int(seg.start * 1000)
                end_ms = int(seg.end * 1000)

                text = seg.text.strip()
                no_speech_prob = getattr(seg, 'no_speech_prob', 0.0) or 0.0

                # ── TACT: Hallucination filter ──
                is_hallucination = False

                # 1. Boilerplate blocklist
                if text.lower().rstrip('.!,') in _BOILERPLATE:
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

    def _reload_on_cpu(self) -> None:
        """Reload Whisper on CPU int8 — used when CUDA runs out of memory.
        Low-VRAM GPUs (the GTX 1650 = 4 GB, for instance) can't fit Whisper
        medium under batched inference; CPU is slower but always completes."""
        try:
            import torch as _torch
            if hasattr(self, 'engine'):
                del self.engine
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass
        from faster_whisper import WhisperModel
        self.engine = WhisperModel(self.model_name, device='cpu', compute_type='int8')
        self.device_used = 'cpu_int8'
        AudioIntelligence._cached_device = self.device_used
        log.log_stage('AUDIO', f'Whisper reloaded on CPU int8 ({self.model_name})')

    def whisper_translate(self, video_path: str, source_lang: str = None,
                          on_progress=None) -> List[dict]:
        """Direct audio→English translation using Whisper's native translate task.

        This bypasses the two-step error amplification problem:
          BAD:  ja audio → Whisper(ja text) → Google(en text)  [errors × errors]
          GOOD: ja audio → Whisper(en text directly)           [single step]

        Whisper's translate task is trained on multilingual audio→English pairs.
        It handles proper nouns better because it hears the audio directly
        rather than trying to transcribe Japanese text first (where names
        like ゼクス become "Z/X" and then Google translates that literally).

        Returns: [{start_sec, end_sec, text, words}] in English
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
            if self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _free_bytes, _ = _torch.cuda.mem_get_info()
                        _free_gb = _free_bytes / 1_073_741_824
                        if _free_gb < 3.0:
                            log.log_stage('TRANSLATE',
                                f'Only {_free_gb:.1f} GB VRAM free '
                                '— using CPU for translation')
                            self._reload_on_cpu()
                except Exception:
                    pass

            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=self.engine)
                segments_iter, info = batched.transcribe(
                    audio_path, batch_size=16,
                    language=whisper_lang,
                    task='translate',  # ← the key difference
                    beam_size=5, vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 200,
                    },
                    word_timestamps=True,
                    condition_on_previous_text=True,
                    no_speech_threshold=0.5,
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
                    condition_on_previous_text=True,
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


