#!/usr/bin/env python3
"""
ClipAI Clipper — AI-Driven Clip Extraction for ClipAI Reframer

Two-pass hotspot architecture:
  Pass 1 (CPU):  Score every 30s window using signals already computed by the
                 Reframer's Perceiver (faces, motion, speech, audio RMS, etc.)
  Pass 2 (GPU):  Send only the top-N hottest chunks to VideoLLaMA2.1-7B-AV for
                 deep audio-visual understanding (laughter, applause, energy).

Cloud editorial judge (optional):
  Gemini via Google AI  —or—  any model via OpenRouter  —or—  none (signal-only).

The VLM and cloud judge analyze the video provided (source or reframed export)
to find the most engaging clip-worthy moments automatically after analysis.

Usage:
  Launched from the "Find Clips" button in the ClipAI Reframer GUI.
  Can also be imported:
    from clipai_clipper import ClipExtractor, ClipperConfig
"""

import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time as _time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Callable
from pathlib import Path

logger = logging.getLogger("clipai.clipper")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClipCandidate:
    """A candidate clip moment discovered by VLM or algorithmic signals."""
    start_s: float = 0.0
    end_s: float = 0.0
    duration_s: float = 0.0
    source: str = ""                 # "vlm_discovery", "signal_peak", "algo_window"
    composite_score: float = 0.0     # 0-1 ranking score
    signal_score: float = 0.0       # raw signal density
    vlm_reason: str = ""            # why VLM flagged this moment
    vlm_hook: str = ""              # suggested hook/opening line
    judge_scores: Optional[dict] = None   # from cloud editorial judge
    judge_verdict: str = ""         # keep / trim / skip
    judge_title: str = ""           # suggested title from judge
    transcript_slice: str = ""      # text spoken during this clip
    crop_x_offset: int = 0          # per-clip crop X adjustment (pixels)


@dataclass
class ClipperConfig:
    """User-configurable clipper settings, persisted to clipper_config.json."""
    # ── VideoLLaMA2 ──
    videollama2_enabled: bool = True
    videollama2_model: str = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    videollama2_quantize: str = "int8"   # "int8", "fp16"
    chunk_duration_s: int = 600          # 10 min chunks
    max_vlm_chunks: int = 6             # caps GPU time for long videos

    # ── Cloud Editorial Judge ──
    cloud_backend: str = "none"          # "none", "google_ai", "openrouter"
    google_ai_key: str = ""
    google_ai_model: str = "gemini-2.0-flash"
    openrouter_key: str = ""
    openrouter_model: str = "google/gemini-2.0-flash"

    # ── Clip Preferences ──
    platforms: list = field(default_factory=lambda: ["tiktok", "reels", "shorts"])
    max_clips: int = 0                   # 0 = auto (scales with video length)
    min_duration_s: int = 15
    max_duration_s: int = 300            # 5 minutes
    ideal_duration_s: int = 60           # 1 minute

    # ── Optional Content Preferences ──
    preferred_subjects: str = ""         # e.g. "funny moments, hot takes, drama"
    avoid_subjects: str = ""             # e.g. "sponsor segments, dead air"

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def effective_max_clips(self, video_duration_s: float) -> int:
        """Compute actual max_clips. 0 = auto: ~1 clip per 2 minutes."""
        if self.max_clips > 0:
            return self.max_clips
        # Auto: roughly 1 clip per 2 minutes, minimum 3, no cap
        return max(3, int(video_duration_s / 120))

    @classmethod
    def load(cls, path: str) -> 'ClipperConfig':
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            cfg = cls()
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            # Backward compat: old configs had "platform" (str) not "platforms" (list)
            if 'platform' in data and 'platforms' not in data:
                cfg.platforms = [data['platform']]
            return cfg
        except Exception:
            return cls()


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL TIMELINE — CPU-side scoring of per-second "interestingness"
# ═══════════════════════════════════════════════════════════════════════════

class SignalTimeline:
    """Per-second signal arrays derived from the Reframer's PerceptionResult.

    These are cheap to compute (mostly just reshaping existing data) and
    provide the coarse filter for the two-pass architecture: we score every
    30-second window, rank them, and send only the hottest to the VLM.
    """

    def __init__(self, duration_s: int):
        self.duration_s = duration_s
        n = max(1, duration_s)
        self.speech_density = [0.0] * n    # words per second in this window
        self.sentiment = [0.0] * n         # -1 to +1 emotional valence
        self.rms_energy = [0.0] * n        # 0-1 normalized audio energy
        self.hook_score = [0.0] * n        # 0-1 viral language score
        self.motion_magnitude = [0.0] * n  # avg frame-to-frame motion
        self.speaker_turn_rate = [0.0] * n # speaker changes per second
        self.question_density = [0.0] * n  # questions per second
        self.face_count = [0.0] * n        # faces visible per second
        self.scene_cut_density = [0.0] * n # cuts per second in window

    @classmethod
    def from_perception(cls, perception) -> 'SignalTimeline':
        """Build signal timeline from an existing PerceptionResult.

        Almost everything is already computed — we just reshape from
        time_ms → per-second arrays. No new ML inference needed.
        """
        duration_s = max(1, perception.duration_ms // 1000)
        st = cls(duration_s)
        n = st.duration_s

        # ── Face count per second ──
        if perception.face_timeline:
            for t_ms, faces in perception.face_timeline.items():
                sec = min(int(t_ms / 1000), n - 1)
                st.face_count[sec] = max(st.face_count[sec], len(faces))

        # ── Motion magnitude per second ──
        if perception.motion_timeline:
            counts = [0] * n
            for t_ms, mag in perception.motion_timeline.items():
                sec = min(int(t_ms / 1000), n - 1)
                st.motion_magnitude[sec] += mag
                counts[sec] += 1
            for i in range(n):
                if counts[i] > 0:
                    st.motion_magnitude[i] /= counts[i]

        # ── RMS energy per second ──
        if perception.audio_rms:
            counts = [0] * n
            for t_ms, rms in perception.audio_rms.items():
                sec = min(int(t_ms / 1000), n - 1)
                st.rms_energy[sec] += rms
                counts[sec] += 1
            for i in range(n):
                if counts[i] > 0:
                    st.rms_energy[i] /= counts[i]
            # Normalize to 0-1
            peak = max(st.rms_energy) if max(st.rms_energy) > 0 else 1.0
            st.rms_energy = [v / peak for v in st.rms_energy]

        # ── Speech density (words per second from transcript) ──
        if perception.transcript_segments:
            for seg in perception.transcript_segments:
                s_start = int(seg.get('start', 0))
                s_end = int(seg.get('end', s_start + 1))
                text = seg.get('text', '')
                word_count = len(text.split())
                span = max(1, s_end - s_start)
                wps = word_count / span
                for sec in range(max(0, s_start), min(n, s_end)):
                    st.speech_density[sec] = wps

        # ── Hook score (viral language patterns) ──
        hook_patterns = [
            r'\b(never|always|worst|best|insane|crazy|wild|unbelievable)\b',
            r'\b(secret|hack|trick|tip|game.?changer|mind.?blow)\b',
            r'\b(you won\'?t believe|wait for it|here\'?s the thing)\b',
            r'\b(hot take|controversial|unpopular opinion)\b',
            r'\b(finally|breaking|just happened|right now)\b',
            r'\b(this is why|the reason|nobody talks about)\b',
        ]
        if perception.transcript_segments:
            for seg in perception.transcript_segments:
                text = seg.get('text', '').lower()
                s_start = int(seg.get('start', 0))
                s_end = int(seg.get('end', s_start + 1))
                score = 0.0
                for pattern in hook_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        score += 0.2
                score = min(1.0, score)
                if score > 0:
                    for sec in range(max(0, s_start), min(n, s_end)):
                        st.hook_score[sec] = max(st.hook_score[sec], score)

        # ── Question density ──
        if perception.transcript_segments:
            for seg in perception.transcript_segments:
                text = seg.get('text', '')
                q_count = text.count('?')
                if q_count > 0:
                    s_start = int(seg.get('start', 0))
                    s_end = int(seg.get('end', s_start + 1))
                    span = max(1, s_end - s_start)
                    qps = q_count / span
                    for sec in range(max(0, s_start), min(n, s_end)):
                        st.question_density[sec] = qps

        # ── Speaker turn rate ──
        if perception.speaker_timeline:
            prev_speaker = None
            for t_ms in sorted(perception.speaker_timeline.keys()):
                spk = perception.speaker_timeline[t_ms]
                sec = min(int(t_ms / 1000), n - 1)
                if prev_speaker is not None and spk != prev_speaker:
                    st.speaker_turn_rate[sec] += 1.0
                prev_speaker = spk

        # ── Scene cut density ──
        if perception.scene_cuts:
            for cut_ms in perception.scene_cuts:
                sec = min(int(cut_ms / 1000), n - 1)
                st.scene_cut_density[sec] = 1.0

        # ── Sentiment (keyword heuristic — no extra model needed) ──
        pos_words = {'amazing', 'love', 'awesome', 'great', 'incredible',
                     'beautiful', 'perfect', 'fantastic', 'wonderful', 'brilliant',
                     'hilarious', 'excited', 'happy', 'best', 'favorite'}
        neg_words = {'terrible', 'horrible', 'awful', 'hate', 'worst',
                     'disgusting', 'angry', 'furious', 'stupid', 'pathetic',
                     'disaster', 'failure', 'depressing', 'sad', 'annoyed'}
        if perception.transcript_segments:
            for seg in perception.transcript_segments:
                words = set(seg.get('text', '').lower().split())
                pos = len(words & pos_words)
                neg = len(words & neg_words)
                if pos > 0 or neg > 0:
                    val = (pos - neg) / max(1, pos + neg)
                    s_start = int(seg.get('start', 0))
                    s_end = int(seg.get('end', s_start + 1))
                    for sec in range(max(0, s_start), min(n, s_end)):
                        st.sentiment[sec] = val

        return st


def score_chunk_signals(st: SignalTimeline, start_s: int, end_s: int) -> float:
    """Score a chunk's overall 'interestingness' from pre-computed signals.

    Uses adaptive weighting: when text-based signals (hooks, sentiment,
    questions) are absent (e.g. non-English content, no speech), their
    weight is redistributed to visual/audio signals so the total score
    range stays meaningful regardless of language.
    """
    n = st.duration_s
    start = max(0, start_s)
    end = min(n, end_s)
    if start >= end:
        return 0.0

    span = end - start
    components = {}

    # ── Visual / audio signals (always available) ──

    # Face count — more faces = more interesting
    face_vals = st.face_count[start:end]
    if face_vals:
        max_faces = max(face_vals)
        avg_faces = sum(face_vals) / span
        components['faces'] = min(1.0, avg_faces / 2.0) * 0.5 + min(1.0, max_faces / 3.0) * 0.5

    # RMS energy peaks — audio excitement
    rms_vals = st.rms_energy[start:end]
    if rms_vals:
        components['rms'] = max(rms_vals) * 0.6 + (sum(rms_vals) / span) * 0.4

    # Motion magnitude — visual dynamism (use variance, not just average)
    motion_vals = st.motion_magnitude[start:end]
    if motion_vals:
        avg_m = sum(motion_vals) / span
        max_m = max(motion_vals)
        components['motion'] = min(1.0, (avg_m / 10.0) * 0.4 + (max_m / 20.0) * 0.6)

    # Scene cut density — more cuts = more dynamic editing
    cut_vals = st.scene_cut_density[start:end]
    if cut_vals:
        n_cuts = sum(cut_vals)
        components['cuts'] = min(1.0, n_cuts / max(1, span / 10))  # ~1 cut per 10s = 1.0

    # Speaker turn rate — conversation dynamics
    turn_vals = st.speaker_turn_rate[start:end]
    if turn_vals:
        components['turns'] = min(1.0, sum(turn_vals) / span * 2)

    # ── Text-based signals (may be absent for non-English) ──

    # Speech density
    speech_sum = sum(st.speech_density[start:end])
    if speech_sum > 0:
        components['speech'] = min(1.0, speech_sum / span / 3.0)

    # Sentiment peaks
    sentiment_vals = st.sentiment[start:end]
    sent_max = max(abs(v) for v in sentiment_vals) if sentiment_vals else 0
    if sent_max > 0:
        components['sentiment'] = sent_max

    # Hook phrases
    hook_vals = st.hook_score[start:end]
    hook_max = max(hook_vals) if hook_vals else 0
    if hook_max > 0:
        components['hooks'] = hook_max

    # Question density
    q_sum = sum(st.question_density[start:end])
    if q_sum > 0:
        components['questions'] = min(1.0, q_sum / span)

    # ── Adaptive weighting ──
    # Base weights for each signal category
    weights = {
        'faces':     0.20,
        'rms':       0.15,
        'motion':    0.10,
        'cuts':      0.05,
        'turns':     0.10,
        'speech':    0.10,
        'sentiment': 0.10,
        'hooks':     0.10,
        'questions': 0.10,
    }

    # Compute which signals are available
    available = {k for k in components if components[k] > 0}
    unavailable_weight = sum(w for k, w in weights.items()
                            if k not in available and k in ('sentiment', 'hooks', 'questions', 'speech'))

    # Redistribute unavailable text-signal weight to visual/audio signals
    visual_keys = [k for k in ('faces', 'rms', 'motion', 'cuts', 'turns')
                   if k in available]
    if visual_keys and unavailable_weight > 0:
        bonus_each = unavailable_weight / len(visual_keys)
        for k in visual_keys:
            weights[k] += bonus_each

    # Compute final score
    score = 0.0
    for k, w in weights.items():
        if k in components:
            score += w * components[k]

    return score


# ═══════════════════════════════════════════════════════════════════════════
#  VideoLLaMA2 DISCOVERY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class VideoLLaMA2Discovery:
    """VideoLLaMA2.1-7B-AV for clip-worthy moment discovery.

    Uses the audio-visual model to deeply understand video chunks:
    - BEATs audio encoder detects laughter, applause, music swells
    - SigLIP vision encoder understands visual context
    - Qwen2 language model reasons about viral potential

    INT8 quantization fits within RTX 4070's 12GB VRAM.
    """

    MODEL_ID = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"

    def __init__(self, model_id: str = None, quantize: str = "int8"):
        self.model_id = model_id or self.MODEL_ID
        self.quantize = quantize
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False

    def is_available(self) -> bool:
        """True only when VideoLLaMA2 can realistically run on THIS machine.

        Two gates:
          1. The GPU has ~9GB+ free VRAM. The 7B-AV model is a fixed ~7-8GB
             (INT8) cost — it cannot be shrunk by chunking the input. On a
             4GB card (e.g. GTX 1650) this returns False so the clipper goes
             straight to the Ollama / cloud / signal fallback instead of
             attempting a load that would OOM.
          2. The ``videollama2`` package is importable — pip-installed, or a
             repo vendored at backend/services/VideoLLaMA2/ (added to
             sys.path on demand). find_spec is used so the heavy package is
             not imported just for this probe.

        This lets one container image self-adapt: it uses VideoLLaMA2 on a
        big GPU and silently falls back on a small one.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("VideoLLaMA2 skipped: no CUDA GPU on this host")
                return False
            free_mb = torch.cuda.mem_get_info()[0] / 1024 / 1024
            if free_mb < 9000:
                logger.info(
                    "VideoLLaMA2 skipped: %.0f MB VRAM free (need ~9000) — "
                    "the 7B model will not fit; using Ollama/cloud/signal fallback",
                    free_mb,
                )
                return False
        except Exception as e:
            logger.info("VideoLLaMA2 VRAM probe failed (%s) — skipping", e)
            return False

        import importlib.util
        if importlib.util.find_spec("videollama2") is not None:
            return True
        vl2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VideoLLaMA2")
        if os.path.isdir(os.path.join(vl2_dir, "videollama2")):
            if vl2_dir not in sys.path:
                sys.path.insert(0, vl2_dir)
            return importlib.util.find_spec("videollama2") is not None
        logger.info("VideoLLaMA2 skipped: 'videollama2' package not installed")
        return False

    def load(self):
        """Load model with INT8 quantization for RTX 4070."""
        if self._loaded:
            return

        logger.info(f"Loading VideoLLaMA2: {self.model_id} ({self.quantize})")

        try:
            from videollama2 import model_init
            from videollama2.utils import disable_torch_init
        except ImportError:
            # Try path-based import
            vl2_dir = os.path.join(os.path.dirname(__file__), "VideoLLaMA2")
            if os.path.exists(vl2_dir) and vl2_dir not in sys.path:
                sys.path.insert(0, vl2_dir)
            from videollama2 import model_init
            from videollama2.utils import disable_torch_init

        disable_torch_init()

        load_kwargs = {}
        if self.quantize == "int8":
            load_kwargs['load_8bit'] = True
        elif self.quantize == "int4":
            load_kwargs['load_4bit'] = True

        self.model, self.processor, self.tokenizer = model_init(
            self.model_id, **load_kwargs
        )
        self._loaded = True
        logger.info("VideoLLaMA2 loaded successfully")

    def discover_clips(self, video_path: str, transcript_segments: list,
                       signal_timeline: SignalTimeline,
                       chunk_duration_s: int = 600,
                       max_vlm_chunks: int = 6,
                       preferred_subjects: str = "",
                       avoid_subjects: str = "",
                       platforms: list = None,
                       on_progress: Callable = None) -> List[ClipCandidate]:
        """Process video hotspots via VLM, return discovered clip candidates.

        Two-pass architecture:
        1. signal_timeline (from Pass 1) has already scored every second
        2. We divide the video into chunks and rank by signal density
        3. Only the top max_vlm_chunks are sent to VideoLLaMA2
        4. Short videos (≤ max_vlm_chunks chunks) send ALL chunks

        Args:
            video_path: Path to video to analyze for clips
            transcript_segments: Whisper transcript [{start, end, text}]
            signal_timeline: SignalTimeline from Pass 1 (pre-computed)
            chunk_duration_s: Chunk size in seconds (default 600 = 10 min)
            max_vlm_chunks: Max chunks to send to VLM (default 6)
            preferred_subjects: Optional user preference for clip content
            avoid_subjects: Optional user preference for what to skip
            on_progress: Callback(float 0-1)
        """
        if not self._loaded:
            self.load()

        import torch

        duration_s = _get_video_duration(video_path)
        n_total_chunks = max(1, math.ceil(duration_s / chunk_duration_s))

        # ── Rank chunks by signal density ──
        chunk_scores = []
        for i in range(n_total_chunks):
            start_s = int(i * chunk_duration_s)
            end_s = int(min((i + 1) * chunk_duration_s, duration_s))
            score = score_chunk_signals(signal_timeline, start_s, end_s)
            chunk_scores.append((i, start_s, end_s, score))

        chunk_scores.sort(key=lambda x: x[3], reverse=True)

        n_to_process = min(n_total_chunks, max_vlm_chunks)
        selected_chunks = chunk_scores[:n_to_process]
        selected_chunks.sort(key=lambda x: x[1])  # re-sort by time

        logger.info(
            f"VideoLLaMA2: {n_total_chunks} total chunks, "
            f"processing top {n_to_process} by signal density"
        )
        for idx, (i, s, e, score) in enumerate(selected_chunks):
            logger.info(
                f"  Chunk {i+1}/{n_total_chunks} "
                f"({_fmt_time(s)}–{_fmt_time(e)}) "
                f"signal_score={score:.3f}"
            )

        all_candidates = []

        for idx, (chunk_idx, start_s, end_s, sig_score) in enumerate(selected_chunks):
            if on_progress:
                on_progress(idx / max(1, n_to_process))

            chunk_path = None
            try:
                chunk_path = _extract_chunk(video_path, start_s, end_s)

                transcript_slice = _slice_transcript(
                    transcript_segments, start_s, end_s)

                prompt = self._build_discovery_prompt(
                    start_s, end_s, transcript_slice,
                    preferred_subjects, avoid_subjects, platforms)

                from videollama2 import mm_infer

                modal_tensor = self.processor['video'](chunk_path)
                output = mm_infer(
                    modal_tensor,
                    prompt,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    do_sample=False,
                    modal='video'
                )

                candidates = _parse_vlm_response(output, start_s, end_s)
                for c in candidates:
                    c.signal_score = sig_score
                all_candidates.extend(candidates)

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    logger.warning(f"OOM on chunk {chunk_idx+1}, skipping")
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                logger.warning(f"VLM error on chunk {chunk_idx+1}: {e}")
            finally:
                if chunk_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass
                torch.cuda.empty_cache()

        if on_progress:
            on_progress(1.0)

        return all_candidates

    def _build_discovery_prompt(self, start_s, end_s, transcript_slice,
                                preferred_subjects="", avoid_subjects="",
                                platforms=None):
        """Prompt VideoLLaMA2 to find viral clip moments."""
        platform_str = _format_platforms(platforms)
        pref_line = ""
        if preferred_subjects.strip():
            pref_line = f"\nPRIORITIZE moments with: {preferred_subjects.strip()}"
        avoid_line = ""
        if avoid_subjects.strip():
            avoid_line = f"\nAVOID: {avoid_subjects.strip()}"

        return f"""You are a viral video editor analyzing a video segment from {_fmt_time(start_s)} to {_fmt_time(end_s)}.

TRANSCRIPT FOR THIS SEGMENT:
{transcript_slice}

Watch and listen carefully to this segment. Identify the 2-3 most compelling moments that would make strong standalone short-form clips (15-60 seconds) for {platform_str}.

Look for:
- Emotional peaks (laughter, surprise, anger, excitement)
- Strong opinions or hot takes
- Funny or unexpected moments
- "Aha" revelations or surprising facts
- Confrontation or debate
- Music/audio energy spikes
- Visual moments that would stop someone from scrolling
{pref_line}{avoid_line}

For each moment, respond in this exact JSON format:
[
  {{"timestamp": "MM:SS", "duration": 30, "reason": "one sentence why this is clip-worthy", "hook": "suggested opening line for the clip"}},
  ...
]

Timestamps are relative to the START of this segment ({_fmt_time(start_s)}).
Respond ONLY with the JSON array, no other text."""

    def unload(self):
        """Free GPU memory after discovery pass."""
        if self.model is not None:
            logger.info("Unloading VideoLLaMA2 from GPU")
            del self.model
            del self.processor
            del self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            self._loaded = False
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
#  OLLAMA FALLBACK DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

class OllamaDiscovery:
    """Fallback VLM discovery using a local Ollama instance with a vision model.

    Used when VideoLLaMA2 is not installed or fails to load. Sends keyframes
    (not video) to a vision-capable Ollama model (e.g. llava:7b).
    """

    def __init__(self, model: str = "llava:7b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip('/')

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.host}/api/tags", method='GET')
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            models = [m.get('name', '') for m in data.get('models', [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def discover_clips(self, video_path: str, transcript_segments: list,
                       signal_timeline: SignalTimeline,
                       chunk_duration_s: int = 600,
                       max_vlm_chunks: int = 6,
                       preferred_subjects: str = "",
                       avoid_subjects: str = "",
                       platforms: list = None,
                       on_progress: Callable = None) -> List[ClipCandidate]:
        """Extract keyframes from top chunks and send to Ollama vision model."""
        import base64

        platform_str = _format_platforms(platforms)
        duration_s = _get_video_duration(video_path)
        n_total = max(1, math.ceil(duration_s / chunk_duration_s))

        chunk_scores = []
        for i in range(n_total):
            s = int(i * chunk_duration_s)
            e = int(min((i + 1) * chunk_duration_s, duration_s))
            score = score_chunk_signals(signal_timeline, s, e)
            chunk_scores.append((i, s, e, score))

        chunk_scores.sort(key=lambda x: x[3], reverse=True)
        n_proc = min(n_total, max_vlm_chunks)
        selected = sorted(chunk_scores[:n_proc], key=lambda x: x[1])

        all_candidates = []
        for idx, (ci, start_s, end_s, sig_score) in enumerate(selected):
            if on_progress:
                on_progress(idx / max(1, n_proc))

            # Extract 4 keyframes from this chunk
            frames_b64 = _extract_keyframes_b64(video_path, start_s, end_s, n_frames=4)
            transcript_slice = _slice_transcript(transcript_segments, start_s, end_s)

            pref = ""
            if preferred_subjects.strip():
                pref = f"\nPRIORITIZE: {preferred_subjects}"
            avoid = ""
            if avoid_subjects.strip():
                avoid = f"\nAVOID: {avoid_subjects}"

            prompt = (
                f"You are a viral video editor. This is a segment from "
                f"{_fmt_time(start_s)} to {_fmt_time(end_s)}.\n\n"
                f"TRANSCRIPT:\n{transcript_slice}\n\n"
                f"These keyframes show the visual content. Identify 2-3 "
                f"clip-worthy moments (15-60s) for {platform_str}.\n"
                f"{pref}{avoid}\n\n"
                f"Respond ONLY with JSON:\n"
                f'[{{"timestamp":"MM:SS","duration":30,"reason":"...","hook":"..."}}]'
                f"\nTimestamps relative to {_fmt_time(start_s)}."
            )

            try:
                import urllib.request
                payload = json.dumps({
                    "model": self.model,
                    "prompt": prompt,
                    "images": frames_b64,
                    "stream": False,
                    "options": {"temperature": 0.3}
                }).encode()
                req = urllib.request.Request(
                    f"{self.host}/api/generate",
                    data=payload,
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )
                resp = urllib.request.urlopen(req, timeout=120)
                result = json.loads(resp.read().decode())
                output = result.get('response', '')
                candidates = _parse_vlm_response(output, start_s, end_s)
                for c in candidates:
                    c.signal_score = sig_score
                    c.source = "ollama_discovery"
                all_candidates.extend(candidates)
            except Exception as e:
                logger.warning(f"Ollama error on chunk {ci+1}: {e}")

        if on_progress:
            on_progress(1.0)
        return all_candidates


# ═══════════════════════════════════════════════════════════════════════════
#  CLOUD EDITORIAL JUDGES
# ═══════════════════════════════════════════════════════════════════════════

class GeminiJudge:
    """Google AI Gemini editorial judge.

    Rates clip candidates using vision (keyframes) + transcript context.
    Provides scores, title suggestions, and trim recommendations.
    """

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    AVAILABLE_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def judge(self, candidate: ClipCandidate, transcript_slice: str,
              keyframes_b64: List[str], signal_summary: str,
              preferred_subjects: str = "",
              avoid_subjects: str = "") -> dict:
        """Rate a clip candidate. Returns dict with scores + verdict."""
        import urllib.request

        url = self.ENDPOINT.format(model=self.model) + f"?key={self.api_key}"

        prompt = _build_judge_prompt(
            candidate, transcript_slice, signal_summary,
            preferred_subjects, avoid_subjects)

        parts = [{"text": prompt}]
        for img_b64 in keyframes_b64[:6]:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64
                }
            })

        payload = json.dumps({
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024
            }
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            resp = urllib.request.urlopen(req, timeout=45)
            data = json.loads(resp.read().decode())
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _parse_judge_response(text)
        except Exception as e:
            logger.warning(f"Gemini judge error: {e}")
            return {"error": str(e)}


class OpenRouterJudge:
    """OpenRouter editorial judge — works with any model.

    Falls back to text-only if the model doesn't support vision.
    """

    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-exp:free"):
        self.api_key = api_key
        self.model = model

    def judge(self, candidate: ClipCandidate, transcript_slice: str,
              keyframes_b64: List[str], signal_summary: str,
              preferred_subjects: str = "",
              avoid_subjects: str = "") -> dict:
        """Rate a clip candidate via OpenRouter."""
        import urllib.request

        prompt = _build_judge_prompt(
            candidate, transcript_slice, signal_summary,
            preferred_subjects, avoid_subjects)

        content = [{"type": "text", "text": prompt}]
        for img_b64 in keyframes_b64[:6]:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
            "max_tokens": 1024
        }).encode()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(
            self.ENDPOINT, data=payload, headers=headers, method='POST'
        )

        try:
            resp = urllib.request.urlopen(req, timeout=45)
            data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            return _parse_judge_response(text)
        except urllib.error.HTTPError as e:
            # Vision-blind fallback: if 400/422, retry text-only
            if e.code in (400, 422):
                logger.info(f"Model {self.model} may not support vision, retrying text-only")
                return self._judge_text_only(prompt)
            logger.warning(f"OpenRouter judge error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.warning(f"OpenRouter judge error: {e}")
            return {"error": str(e)}

    def _judge_text_only(self, prompt: str) -> dict:
        """Retry without images for vision-blind models."""
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024
        }).encode()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(
            self.ENDPOINT, data=payload, headers=headers, method='POST'
        )

        try:
            resp = urllib.request.urlopen(req, timeout=45)
            data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            return _parse_judge_response(text)
        except Exception as e:
            logger.warning(f"OpenRouter text-only judge error: {e}")
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  API KEY TESTING
# ═══════════════════════════════════════════════════════════════════════════

def test_google_ai_key(api_key: str, model: str = "gemini-2.0-flash") -> dict:
    """Send a minimal request to the Google AI API to verify the key works.

    Returns:
        {"ok": True, "model": ..., "response": ...} on success
        {"ok": False, "error": ...} on failure
    """
    import urllib.request, urllib.error

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    payload = json.dumps({
        "contents": [{"parts": [{"text": "Reply with exactly: KEY_VALID"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 32}
    }).encode()

    req = urllib.request.Request(
        url, data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return {"ok": True, "model": model, "response": text}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:
            pass
        if e.code == 400:
            return {"ok": False, "error": f"Invalid API key or bad request.\n{body}"}
        elif e.code == 403:
            return {"ok": False, "error": f"API key forbidden — check key permissions.\n{body}"}
        elif e.code == 404:
            return {"ok": False, "error": f"Model '{model}' not found — try a different model."}
        elif e.code == 429:
            return {"ok": False, "error": "Rate limited — key is valid but quota exceeded."}
        else:
            return {"ok": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def test_openrouter_key(api_key: str, model: str = "google/gemini-2.0-flash-exp:free") -> dict:
    """Send a minimal request to OpenRouter to verify the key and model work.

    Returns:
        {"ok": True, "model": ..., "response": ...} on success
        {"ok": False, "error": ...} on failure
    """
    import urllib.request, urllib.error

    url = "https://openrouter.ai/api/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: KEY_VALID"}],
        "temperature": 0,
        "max_tokens": 32
    }).encode()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"].strip()
        actual_model = data.get("model", model)
        return {"ok": True, "model": actual_model, "response": text}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:300]
        except Exception:
            pass
        if e.code == 401:
            return {"ok": False, "error": f"Invalid API key.\n{body}"}
        elif e.code == 402:
            return {"ok": False, "error": "Insufficient credits on OpenRouter."}
        elif e.code == 404:
            return {"ok": False, "error": f"Model '{model}' not found on OpenRouter."}
        elif e.code == 429:
            return {"ok": False, "error": "Rate limited — key is valid but quota exceeded."}
        else:
            return {"ok": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL FETCHING — pull real model lists from APIs
# ═══════════════════════════════════════════════════════════════════════════

def fetch_openrouter_models(api_key: str = "") -> list:
    """Fetch available models from OpenRouter's /api/v1/models endpoint.

    Returns a list of model ID strings sorted by name, or a hardcoded
    fallback list if the API call fails.
    """
    import urllib.request, urllib.error

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers=headers, method='GET'
    )

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        models = data.get("data", [])

        # Extract model IDs, filter to text/chat models, sort
        model_ids = []
        for m in models:
            mid = m.get("id", "")
            if not mid:
                continue
            model_ids.append(mid)

        model_ids.sort()
        return model_ids if model_ids else _openrouter_fallback()
    except Exception as e:
        logger.warning(f"Could not fetch OpenRouter models: {e}")
        return _openrouter_fallback()


def _openrouter_fallback() -> list:
    """Hardcoded fallback list of popular OpenRouter models."""
    return [
        "anthropic/claude-sonnet-4",
        "google/gemini-2.0-flash",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "meta-llama/llama-4-maverick",
        "meta-llama/llama-4-scout",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4.1",
        "openai/gpt-4.1-mini",
        "qwen/qwen3-235b-a22b",
    ]


def fetch_google_ai_models(api_key: str) -> list:
    """Fetch available Gemini models from the Google AI API.

    Returns a list of model name strings (e.g. 'gemini-2.0-flash'),
    or a hardcoded fallback if the API call fails.
    """
    import urllib.request, urllib.error

    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

    req = urllib.request.Request(url, method='GET')

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        models_raw = data.get("models", [])

        model_names = []
        for m in models_raw:
            name = m.get("name", "")  # e.g. "models/gemini-2.0-flash"
            # Only include generateContent-capable models
            methods = m.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            # Strip "models/" prefix
            short = name.replace("models/", "") if name.startswith("models/") else name
            if short:
                model_names.append(short)

        model_names.sort()
        return model_names if model_names else _google_ai_fallback()
    except Exception as e:
        logger.warning(f"Could not fetch Google AI models: {e}")
        return _google_ai_fallback()


def _google_ai_fallback() -> list:
    """Hardcoded fallback list of Gemini models."""
    return [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_time(s: float) -> str:
    """Format seconds to H:MM:SS or M:SS."""
    total = int(s)
    m, sec = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


_PLATFORM_LABELS = {
    "tiktok": "TikTok",
    "reels": "Instagram Reels",
    "shorts": "YouTube Shorts",
    "youtube": "YouTube",
}


def _format_platforms(platforms: list = None) -> str:
    """Format platform list into a human-readable string for prompts."""
    if not platforms:
        return "TikTok/Reels/Shorts"
    labels = [_PLATFORM_LABELS.get(p, p) for p in platforms]
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _get_video_duration(path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries',
             'format=duration', '-of', 'csv=p=0', path],
            capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_chunk(video_path: str, start_s: float, end_s: float) -> str:
    """FFmpeg extract chunk to temp MP4, keeping audio for AV models."""
    fd, chunk_path = tempfile.mkstemp(suffix=".mp4", prefix="clipai_chunk_")
    os.close(fd)
    subprocess.run([
        'ffmpeg', '-y', '-ss', str(start_s), '-to', str(end_s),
        '-i', video_path, '-c:v', 'copy', '-c:a', 'aac',
        chunk_path
    ], capture_output=True, timeout=120)
    return chunk_path


def _extract_keyframes_b64(video_path: str, start_s: float, end_s: float,
                           n_frames: int = 4) -> List[str]:
    """Extract evenly-spaced keyframes as base64 JPEG strings."""
    import base64
    try:
        import cv2
    except ImportError:
        return []

    frames_b64 = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = end_s - start_s
    interval = max(1, duration / (n_frames + 1))

    for i in range(n_frames):
        t = start_s + interval * (i + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            # Resize for API efficiency (max 512px wide)
            h, w = frame.shape[:2]
            if w > 512:
                scale = 512 / w
                frame = cv2.resize(frame, (512, int(h * scale)))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frames_b64.append(base64.b64encode(buf).decode('ascii'))

    cap.release()
    return frames_b64


def _slice_transcript(segments: list, start_s: float, end_s: float) -> str:
    """Extract transcript text for a time window."""
    lines = []
    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        if seg_end > start_s and seg_start < end_s:
            rel_start = max(0, seg_start - start_s)
            m, s = divmod(int(rel_start), 60)
            lines.append(f"[{m}:{s:02d}] {seg.get('text', '').strip()}")
    return '\n'.join(lines) if lines else "(no speech in this segment)"


def _parse_vlm_response(output: str, chunk_start_s: float,
                        chunk_end_s: float) -> List[ClipCandidate]:
    """Parse JSON timestamps from VLM response into ClipCandidates."""
    candidates = []
    text = output.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    items = None
    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
            except json.JSONDecodeError:
                pass

    if not items:
        logger.warning(f"Could not parse VLM response: {text[:200]}")
        return candidates

    for item in items:
        try:
            ts = str(item.get('timestamp', '0:00'))
            parts = ts.split(':')
            if len(parts) == 2:
                relative_s = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                relative_s = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                relative_s = 0

            abs_start = chunk_start_s + relative_s
            duration = min(int(item.get('duration', 30)), 90)
            abs_end = min(abs_start + duration, chunk_end_s)

            candidates.append(ClipCandidate(
                start_s=abs_start,
                end_s=abs_end,
                duration_s=abs_end - abs_start,
                source="vlm_discovery",
                vlm_reason=str(item.get('reason', '')),
                vlm_hook=str(item.get('hook', '')),
            ))
        except (ValueError, KeyError, IndexError):
            continue

    return candidates


def _build_judge_prompt(candidate: ClipCandidate, transcript_slice: str,
                        signal_summary: str, preferred_subjects: str = "",
                        avoid_subjects: str = "") -> str:
    """Build the editorial judge prompt."""
    pref = ""
    if preferred_subjects.strip():
        pref = f"\nUser prefers clips about: {preferred_subjects}"
    avoid = ""
    if avoid_subjects.strip():
        avoid = f"\nUser wants to avoid: {avoid_subjects}"

    return f"""You are an expert viral content editor judging a potential short-form clip.

CLIP: {_fmt_time(candidate.start_s)} → {_fmt_time(candidate.end_s)} ({candidate.duration_s:.0f}s)

TRANSCRIPT:
{transcript_slice}

SIGNAL DATA:
{signal_summary}

VLM NOTES: {candidate.vlm_reason}
SUGGESTED HOOK: {candidate.vlm_hook}
{pref}{avoid}

The keyframes below show what the viewer will see (this is the reframed vertical version).

Rate this clip on a 1-10 scale for each:
- hook: Does the opening grab attention in the first 3 seconds?
- payoff: Does the clip deliver on its promise?
- retention: Will viewers watch to the end?
- shareability: Will viewers share/save this?
- standalone: Does it work without context?

Then give:
- verdict: "keep", "trim", or "skip"
- trim_suggestion: if "trim", suggest new start/end as "MM:SS-MM:SS"
- title: a short, punchy title for this clip (max 10 words)
- explanation: one sentence explaining your verdict

Respond ONLY with JSON:
{{"hook":N,"payoff":N,"retention":N,"shareability":N,"standalone":N,"verdict":"...","trim_suggestion":"...","title":"...","explanation":"..."}}"""


def _parse_judge_response(text: str) -> dict:
    """Parse cloud judge JSON response."""
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"error": f"Could not parse judge response: {text[:200]}"}


# ═══════════════════════════════════════════════════════════════════════════
#  ALGORITHMIC CANDIDATE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_algo_candidates(signal_timeline: SignalTimeline,
                             config: ClipperConfig) -> List[ClipCandidate]:
    """Generate clip candidates from signal peaks using sliding windows.

    Key improvements over naive sliding window:
    1. Duration bias: clips near ideal_duration get a score bonus
    2. Many window sizes for diverse clip lengths
    3. Dense overlap for thorough coverage
    """
    candidates = []
    n = signal_timeline.duration_s
    if n < config.min_duration_s:
        return candidates

    video_dur = float(n)
    max_clips = config.effective_max_clips(video_dur)
    ideal = config.ideal_duration_s

    # Generate window sizes: every 15s from min to max
    window_sizes = set()
    step_size = 15
    w = config.min_duration_s
    while w <= min(config.max_duration_s, n):
        window_sizes.add(int(w))
        w += step_size
    if config.ideal_duration_s <= n:
        window_sizes.add(config.ideal_duration_s)

    for win_s in sorted(window_sizes):
        if win_s > n:
            continue
        step = max(1, win_s // 4)
        for start in range(0, n - win_s + 1, step):
            end = start + win_s
            score = score_chunk_signals(signal_timeline, start, end)

            if score > 0.001:
                # Duration preference: bonus for clips near ideal duration
                # Gaussian-like falloff from ideal, σ = ideal/2
                dur_dist = abs(win_s - ideal) / max(1, ideal * 0.5)
                dur_bonus = math.exp(-0.5 * dur_dist * dur_dist)
                # Blend: 70% signal score + 30% duration preference
                adjusted_score = score * 0.7 + score * dur_bonus * 0.3

                candidates.append(ClipCandidate(
                    start_s=float(start),
                    end_s=float(end),
                    duration_s=float(win_s),
                    source="signal_peak",
                    signal_score=score,
                    composite_score=adjusted_score,
                ))

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates[:max_clips * 30]


def select_diverse_clips(candidates: List[ClipCandidate],
                         max_clips: int,
                         video_duration_s: float,
                         min_gap_s: float = 30.0) -> List[ClipCandidate]:
    """Select clips that are spread across the video timeline.

    Uses greedy selection with a temporal proximity penalty:
    each selected clip reduces the score of nearby candidates,
    ensuring coverage of the full video instead of clustering
    in one hot region.

    Args:
        candidates: Pre-sorted by composite_score descending
        max_clips: Maximum number of clips to select
        video_duration_s: Total video duration in seconds
        min_gap_s: Minimum desired gap between clip centers (soft)
    """
    if not candidates or max_clips <= 0:
        return []

    selected = []
    remaining = list(candidates)

    for _ in range(max_clips):
        if not remaining:
            break

        # Pick the highest-scoring remaining candidate
        best = remaining[0]
        selected.append(best)
        best_center = (best.start_s + best.end_s) / 2

        # Penalize remaining candidates near the selected one
        new_remaining = []
        for c in remaining[1:]:
            c_center = (c.start_s + c.end_s) / 2
            dist = abs(c_center - best_center)

            # Temporal penalty: candidates closer than min_gap_s get penalized
            if dist < min_gap_s:
                # Stronger penalty for closer candidates
                penalty = 1.0 - (dist / min_gap_s) * 0.7  # up to 70% penalty
                c.composite_score *= (1.0 - penalty)

            # Also skip candidates that heavily overlap with the selected one
            overlap_start = max(c.start_s, best.start_s)
            overlap_end = min(c.end_s, best.end_s)
            overlap = max(0, overlap_end - overlap_start)
            shorter = min(c.duration_s, best.duration_s)
            if shorter > 0 and overlap / shorter > 0.8:
                continue  # skip >80% overlap

            new_remaining.append(c)

        # Re-sort after score adjustments
        new_remaining.sort(key=lambda c: c.composite_score, reverse=True)
        remaining = new_remaining

    return selected


def deduplicate_candidates(candidates: List[ClipCandidate],
                           iou_threshold: float = 0.5) -> List[ClipCandidate]:
    """Remove overlapping candidates, keeping higher-scored ones."""
    if not candidates:
        return []

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    kept = []

    for c in candidates:
        overlaps = False
        for k in kept:
            # Compute temporal IoU
            overlap_start = max(c.start_s, k.start_s)
            overlap_end = min(c.end_s, k.end_s)
            overlap = max(0, overlap_end - overlap_start)
            union = (c.end_s - c.start_s) + (k.end_s - k.start_s) - overlap
            iou = overlap / max(1, union)
            if iou > iou_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(c)

    return kept


def refine_boundaries(candidates: List[ClipCandidate],
                      signal_timeline: SignalTimeline,
                      perception) -> List[ClipCandidate]:
    """Snap clip boundaries to natural breakpoints (scene cuts, silence).

    Adjusts start/end by up to 2 seconds to align with:
    - Scene cuts (avoid cutting mid-shot)
    - Speech gaps (avoid cutting mid-word)
    - Low-motion moments (cleaner transitions)
    """
    if not perception:
        return candidates

    scene_cuts_s = set()
    if perception.scene_cuts:
        scene_cuts_s = {int(ms / 1000) for ms in perception.scene_cuts}

    for c in candidates:
        # Try to snap start to a nearby scene cut
        for offset in range(-2, 3):
            t = int(c.start_s) + offset
            if t in scene_cuts_s and t >= 0:
                c.start_s = float(t)
                break

        # Try to snap end to a nearby scene cut
        for offset in range(-2, 3):
            t = int(c.end_s) + offset
            if t in scene_cuts_s and t <= signal_timeline.duration_s:
                c.end_s = float(t)
                break

        # Recalculate duration
        c.duration_s = c.end_s - c.start_s

    return candidates


# ═══════════════════════════════════════════════════════════════════════════
#  CLIP EXTRACTOR — Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class ClipExtractor:
    """Full clip extraction pipeline orchestrator.

    Two-pass hotspot architecture:
    - Pass 1 (CPU): Signal scan scores every 30s window
    - Pass 2 (GPU): VideoLLaMA2 processes top-N hottest chunks
    - Cloud judge (optional): rates each candidate
    - Export: FFmpeg extracts final clips from the video
    """

    def __init__(self, video_path: str, perception, plan,
                 config: ClipperConfig,
                 transcript_segments: list = None):
        """
        Args:
            video_path: Path to the video to analyze for clips.
                        Can be the source video (auto after analysis) or
                        a reframed export (manual re-run after export).
            perception: PerceptionResult from the Reframer's analysis.
            plan: RenderPlan from the Reframer's planner.
            config: ClipperConfig with user settings.
            transcript_segments: Override transcript (uses perception's if None).
        """
        self.video_path = video_path
        self.perception = perception
        self.plan = plan
        self.config = config
        self.transcript = transcript_segments or (
            perception.transcript_segments if perception else [])
        self.clips: List[ClipCandidate] = []

    def run(self, on_progress: Callable = None) -> List[ClipCandidate]:
        """Execute the full clip extraction pipeline."""
        logger.info(f"Starting clip extraction: {self.video_path}")
        logger.info(f"Config: VLM={self.config.videollama2_enabled}, "
                     f"cloud={self.config.cloud_backend}, "
                     f"max_clips={self.config.max_clips} (0=auto)")

        # ── PASS 1: SIGNAL SCAN (CPU) ─────────────────────────
        if on_progress:
            on_progress(0.05)

        logger.info("Pass 1: Building signal timeline from perception data...")
        signals = SignalTimeline.from_perception(self.perception)
        logger.info(f"Signal timeline: {signals.duration_s}s, "
                     f"speech coverage: {sum(1 for v in signals.speech_density if v > 0)}"
                     f"/{signals.duration_s}s")

        # ── ALGORITHMIC CANDIDATES ────────────────────────────
        if on_progress:
            on_progress(0.10)

        logger.info("Generating algorithmic candidates from signal peaks...")
        algo_candidates = generate_algo_candidates(signals, self.config)
        logger.info(f"Algorithmic candidates: {len(algo_candidates)}")

        # Score algo candidates
        for c in algo_candidates:
            c.composite_score = c.signal_score
            c.transcript_slice = _slice_transcript(
                self.transcript, c.start_s, c.end_s)

        # ── PASS 2: VLM DISCOVERY (GPU, capped) ──────────────
        vlm_candidates = []

        if self.config.videollama2_enabled:
            vlm = VideoLLaMA2Discovery(
                model_id=self.config.videollama2_model,
                quantize=self.config.videollama2_quantize)

            if vlm.is_available():
                try:
                    logger.info("Pass 2: VideoLLaMA2 discovery...")
                    vlm.load()
                    vlm_candidates = vlm.discover_clips(
                        self.video_path,
                        self.transcript,
                        signal_timeline=signals,
                        chunk_duration_s=self.config.chunk_duration_s,
                        max_vlm_chunks=self.config.max_vlm_chunks,
                        preferred_subjects=self.config.preferred_subjects,
                        avoid_subjects=self.config.avoid_subjects,
                        platforms=self.config.platforms,
                        on_progress=lambda p: on_progress(
                            0.15 + p * 0.40) if on_progress else None
                    )
                    vlm.unload()  # FREE GPU before cloud calls
                    logger.info(f"VLM discovered {len(vlm_candidates)} candidates")
                except Exception as e:
                    logger.warning(f"VideoLLaMA2 failed: {e}")
                    try:
                        vlm.unload()
                    except Exception:
                        pass
            else:
                logger.info("VideoLLaMA2 not available, trying Ollama fallback...")

            # Ollama fallback
            if not vlm_candidates:
                ollama = OllamaDiscovery()
                if ollama.is_available():
                    try:
                        vlm_candidates = ollama.discover_clips(
                            self.video_path,
                            self.transcript,
                            signal_timeline=signals,
                            chunk_duration_s=self.config.chunk_duration_s,
                            max_vlm_chunks=self.config.max_vlm_chunks,
                            preferred_subjects=self.config.preferred_subjects,
                            avoid_subjects=self.config.avoid_subjects,
                            platforms=self.config.platforms,
                            on_progress=lambda p: on_progress(
                                0.15 + p * 0.40) if on_progress else None
                        )
                        logger.info(f"Ollama discovered {len(vlm_candidates)} candidates")
                    except Exception as e:
                        logger.warning(f"Ollama fallback failed: {e}")

        if on_progress:
            on_progress(0.55)

        # Enrich VLM candidates with transcript
        for c in vlm_candidates:
            c.transcript_slice = _slice_transcript(
                self.transcript, c.start_s, c.end_s)
            # VLM candidates get a score boost
            c.composite_score = max(c.signal_score, 0.3) + 0.2

        # ── MERGE + DEDUPLICATE ───────────────────────────────
        all_candidates = algo_candidates + vlm_candidates
        all_candidates = deduplicate_candidates(all_candidates, iou_threshold=0.7)
        logger.info(f"After dedup: {len(all_candidates)} candidates")

        # Filter by duration constraints
        all_candidates = [
            c for c in all_candidates
            if self.config.min_duration_s <= c.duration_s <= self.config.max_duration_s
        ]

        video_dur = _get_video_duration(self.video_path)
        eff_max = self.config.effective_max_clips(video_dur)

        # Select diverse clips spread across the video timeline
        # min_gap scales with video length: longer videos = larger gaps
        min_gap = max(30, video_dur / (eff_max * 2))
        all_candidates.sort(key=lambda c: c.composite_score, reverse=True)
        judge_candidates = select_diverse_clips(
            all_candidates, eff_max * 3, video_dur, min_gap_s=min_gap)
        logger.info(f"After diversity selection: {len(judge_candidates)} candidates "
                     f"(target {eff_max} clips, min_gap={min_gap:.0f}s)")

        if on_progress:
            on_progress(0.60)

        # ── CLOUD EDITORIAL JUDGE (if configured) ─────────────
        if self.config.cloud_backend != "none" and judge_candidates:
            logger.info(f"Running cloud editorial judge "
                         f"({self.config.cloud_backend}) on "
                         f"{len(judge_candidates)} candidates...")

            judge = self._create_judge()
            if judge:
                self._run_editorial_judge(
                    judge, judge_candidates,
                    lambda p: on_progress(0.60 + p * 0.25) if on_progress else None
                )

        if on_progress:
            on_progress(0.85)

        # ── BOUNDARY REFINEMENT ───────────────────────────────
        judge_candidates = refine_boundaries(
            judge_candidates, signals, self.perception)

        # ── FINAL RANK + EXPORT ───────────────────────────────
        self.clips = self._final_rank_and_export(
            judge_candidates,
            lambda p: on_progress(0.85 + p * 0.15) if on_progress else None
        )

        if on_progress:
            on_progress(1.0)

        logger.info(f"Clip extraction complete: {len(self.clips)} clips exported")
        return self.clips

    def _create_judge(self):
        """Create the appropriate cloud judge based on config."""
        if self.config.cloud_backend == "google_ai":
            if not self.config.google_ai_key:
                logger.warning("Google AI key not set, skipping judge")
                return None
            return GeminiJudge(self.config.google_ai_key, self.config.google_ai_model)
        elif self.config.cloud_backend == "openrouter":
            if not self.config.openrouter_key:
                logger.warning("OpenRouter key not set, skipping judge")
                return None
            return OpenRouterJudge(self.config.openrouter_key, self.config.openrouter_model)
        return None

    def _run_editorial_judge(self, judge, candidates: List[ClipCandidate],
                             on_progress: Callable = None):
        """Send each candidate to the cloud judge for scoring."""
        total = len(candidates)
        for idx, c in enumerate(candidates):
            if on_progress:
                on_progress(idx / max(1, total))

            # Extract keyframes from the video for the judge
            keyframes = _extract_keyframes_b64(
                self.video_path, c.start_s, c.end_s, n_frames=4)

            signal_summary = (
                f"signal_score={c.signal_score:.3f}, "
                f"source={c.source}, "
                f"duration={c.duration_s:.0f}s"
            )

            result = judge.judge(
                c, c.transcript_slice, keyframes, signal_summary,
                self.config.preferred_subjects, self.config.avoid_subjects)

            if 'error' not in result:
                c.judge_scores = {
                    k: result.get(k, 0)
                    for k in ['hook', 'payoff', 'retention',
                              'shareability', 'standalone']
                }
                c.judge_verdict = result.get('verdict', 'keep')
                c.judge_title = result.get('title', '')

                # Update composite score with judge input
                if c.judge_scores:
                    judge_avg = sum(
                        v for v in c.judge_scores.values()
                        if isinstance(v, (int, float))
                    ) / max(1, len(c.judge_scores))
                    c.composite_score = (
                        0.4 * c.composite_score +
                        0.6 * (judge_avg / 10.0)
                    )

                # Apply trim suggestion
                trim = result.get('trim_suggestion', '')
                if trim and c.judge_verdict == 'trim':
                    try:
                        parts = trim.split('-')
                        if len(parts) == 2:
                            new_start = _parse_mmss(parts[0])
                            new_end = _parse_mmss(parts[1])
                            if new_end > new_start:
                                c.start_s = new_start
                                c.end_s = new_end
                                c.duration_s = new_end - new_start
                    except Exception:
                        pass

            _time.sleep(0.5)  # rate limiting

        if on_progress:
            on_progress(1.0)

    def _final_rank_and_export(self, candidates: List[ClipCandidate],
                               on_progress: Callable = None) -> List[ClipCandidate]:
        """Final ranking and FFmpeg clip extraction."""
        # Remove judge-skipped candidates
        keepers = [c for c in candidates if c.judge_verdict != 'skip']
        if not keepers:
            keepers = candidates

        video_dur = _get_video_duration(self.video_path)
        eff_max = self.config.effective_max_clips(video_dur)

        # Final diversity selection — spread clips across timeline
        keepers.sort(key=lambda c: c.composite_score, reverse=True)
        min_gap = max(20, video_dur / (eff_max * 2))
        final = select_diverse_clips(keepers, eff_max, video_dur, min_gap_s=min_gap)

        # Sort final clips by time for sequential output
        final.sort(key=lambda c: c.start_s)

        if not final:
            return []

        # Create output directory next to the video
        out_dir = os.path.join(os.path.dirname(self.video_path), "clips")
        os.makedirs(out_dir, exist_ok=True)

        # Export each clip
        exported = []
        for idx, c in enumerate(final):
            if on_progress:
                on_progress(idx / max(1, len(final)))

            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            # Use judge title or VLM hook for filename
            slug = ""
            if c.judge_title:
                slug = re.sub(r'[^\w\s-]', '', c.judge_title)
                slug = re.sub(r'[\s]+', '_', slug).strip('_')[:40]
            elif c.vlm_hook:
                slug = re.sub(r'[^\w\s-]', '', c.vlm_hook[:30])
                slug = re.sub(r'[\s]+', '_', slug).strip('_')

            if slug:
                clip_name = f"{base_name}_clip{idx+1}_{slug}.mp4"
            else:
                clip_name = f"{base_name}_clip{idx+1}.mp4"

            clip_path = os.path.join(out_dir, clip_name)

            ok = _export_clip(self.video_path, clip_path, c.start_s, c.end_s)
            if ok:
                exported.append(c)
                logger.info(
                    f"Exported clip {idx+1}/{len(final)}: "
                    f"{_fmt_time(c.start_s)}–{_fmt_time(c.end_s)} "
                    f"({c.duration_s:.0f}s) → {clip_name}")

        # Write manifest
        manifest = {
            "source_video": self.video_path,
            "clips": [
                {
                    "index": i + 1,
                    "start": _fmt_time(c.start_s),
                    "end": _fmt_time(c.end_s),
                    "duration_s": round(c.duration_s, 1),
                    "score": round(c.composite_score, 3),
                    "source": c.source,
                    "vlm_reason": c.vlm_reason,
                    "vlm_hook": c.vlm_hook,
                    "judge_verdict": c.judge_verdict,
                    "judge_title": c.judge_title,
                    "judge_scores": c.judge_scores,
                    "transcript": c.transcript_slice[:500],
                }
                for i, c in enumerate(exported)
            ]
        }
        manifest_path = os.path.join(out_dir, "clips_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        if on_progress:
            on_progress(1.0)

        return exported


def _export_clip(video_path: str, output_path: str,
                 start_s: float, end_s: float) -> bool:
    """Extract a clip from the reframed video using FFmpeg."""
    try:
        result = subprocess.run([
            'ffmpeg', '-y',
            '-ss', f"{start_s:.2f}",
            '-to', f"{end_s:.2f}",
            '-i', video_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            output_path
        ], capture_output=True, timeout=120)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        logger.warning(f"FFmpeg clip export failed: {e}")
        return False


def _parse_mmss(s: str) -> float:
    """Parse 'MM:SS' or 'H:MM:SS' to seconds."""
    parts = s.strip().split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  SETTINGS DIALOG (Tkinter modal)
# ═══════════════════════════════════════════════════════════════════════════

