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

import concurrent.futures as _cf
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

    # ── Cloud Editorial Judge (legacy: clipper-private keys) ──
    # Retained so existing clipper_config.json files continue to work
    # untouched. New code path uses ``judge_primary`` / ``judge_fallback``
    # below, which pulls keys from the app-wide settings (managed by the
    # Settings page) instead of duplicating them here.
    cloud_backend: str = "none"          # "none", "google_ai", "openrouter"
    google_ai_key: str = ""
    google_ai_model: str = "gemini-2.0-flash"
    openrouter_key: str = ""
    openrouter_model: str = "google/gemini-2.0-flash"

    # ── Editorial Judge (new spec format) ──
    # Spec format: "<backend>:<model>" where backend is one of
    # gemini | openrouter | anthropic | groq | ollama. Empty string =
    # no judge (or, for ``judge_fallback``, no backup). When set,
    # these win over the legacy ``cloud_backend`` fields and the API
    # key for the named provider is read from settings (the app-wide
    # provider key managed in the Settings page).
    judge_primary: str = ""
    judge_fallback: str = ""

    # ── Clip Preferences ──
    platforms: list = field(default_factory=lambda: ["tiktok", "reels", "shorts"])
    max_clips: int = 0                   # 0 = auto (scales with video length)
    min_duration_s: int = 60             # 1 minute
    max_duration_s: int = 300            # 5 minutes
    ideal_duration_s: int = 150          # 2.5 minutes

    # ── Optional Content Preferences ──
    preferred_subjects: str = ""         # e.g. "funny moments, hot takes, drama"
    avoid_subjects: str = ""             # e.g. "sponsor segments, dead air"

    # ── Editable VideoLLaMA discovery prompt ("" = built-in default) ──
    discovery_prompt: str = ""

    # ── Replicate (cloud GPU) ──
    replicate_api_key: str = ""
    replicate_model: str = "lucataco/videollama3-7b"
    replicate_enabled: bool = True
    # Coarse-pass rate-limit resilience (see ReplicateDiscoveryV3): on a 429 the
    # tripped chunks resume SEQUENTIALLY after a backoff instead of abandoning the
    # rest of the video. retries=0 restores the old give-up behavior; workers=1
    # goes sequential from the start (gentlest on a low Replicate tier).
    replicate_chunk_workers: int = 3
    replicate_rate_limit_retries: int = 2
    replicate_rate_limit_backoff_s: float = 20.0

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
                s_start = int(seg.get('start_sec', seg.get('start', 0)))
                s_end = int(seg.get('end_sec', seg.get('end', s_start + 1)))
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
                s_start = int(seg.get('start_sec', seg.get('start', 0)))
                s_end = int(seg.get('end_sec', seg.get('end', s_start + 1)))
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
                    s_start = int(seg.get('start_sec', seg.get('start', 0)))
                    s_end = int(seg.get('end_sec', seg.get('end', s_start + 1)))
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
                    s_start = int(seg.get('start_sec', seg.get('start', 0)))
                    s_end = int(seg.get('end_sec', seg.get('end', s_start + 1)))
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
                    # idx+1 = position in the selected batch (1..n_to_process);
                    # chunk_idx+1 = position in the full timeline.
                    logger.warning(
                        f"OOM on chunk {idx+1}/{n_to_process} "
                        f"[timeline {chunk_idx+1}], skipping")
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                logger.warning(
                    f"VLM error on chunk {idx+1}/{n_to_process} "
                    f"[timeline {chunk_idx+1}]: {e}")
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

Watch and listen carefully to this segment. Identify the 2-3 most compelling moments that would make strong standalone short-form clips (1-5 minutes) for {platform_str}.

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
  {{"timestamp": "MM:SS", "duration": 150, "reason": "one sentence why this is clip-worthy", "hook": "suggested opening line for the clip"}},
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
#  REPLICATE CLOUD GPU DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_DISCOVERY_PROMPT = """You are a viral video editor analyzing a video segment from {start} to {end}.

TRANSCRIPT FOR THIS SEGMENT:
{transcript}

Watch and listen carefully to this segment. Identify the 2-3 most compelling moments that would make strong standalone short-form clips, each {min_duration}-{max_duration} seconds long (ideally about {ideal_duration}s), for {platforms}.

Look for:
- Emotional peaks (laughter, surprise, anger, excitement)
- Strong opinions or hot takes
- Funny or unexpected moments
- "Aha" revelations or surprising facts
- Confrontation or debate
- Music/audio energy spikes
- Visual moments that would stop someone from scrolling
{preferred}{avoid}

For each moment, respond in this exact JSON format:
[
  {"timestamp": "MM:SS", "duration": {ideal_duration}, "reason": "one sentence why this is clip-worthy", "hook": "suggested opening line for the clip"},
  ...
]

Timestamps are relative to the START of this segment ({start}).
Respond ONLY with the JSON array, no other text."""


class ReplicateDiscovery:
    """Cloud GPU VideoLLaMA via Replicate API.

    Sends video chunks to Replicate's hosted VideoLLaMA3-7B model.
    No local GPU required — ideal for GTX 1650 and other low-VRAM systems.
    Costs ~$0.02 per chunk (~23s processing time per chunk).

    Per-instance ``total_cost_usd`` accumulates an estimate of every
    call routed through this object so the pipeline can surface the
    real Replicate spend on the analysis summary card.
    """

    def __init__(self, api_key: str, model_id: str = "lucataco/videollama3-7b"):
        self.api_key = api_key
        self.model_id = model_id
        self.total_cost_usd: float = 0.0

    def is_available(self) -> bool:
        """True when a Replicate API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def discover_clips(self, video_path: str, transcript_segments: list,
                       signal_timeline: 'SignalTimeline',
                       chunk_duration_s: int = 600,
                       max_vlm_chunks: int = 6,
                       preferred_subjects: str = "",
                       avoid_subjects: str = "",
                       platforms: list = None,
                       min_dur_s: int = 60,
                       max_dur_s: int = 300,
                       ideal_dur_s: int = 150,
                       discovery_prompt: str = "",
                       on_progress: Callable = None) -> List['ClipCandidate']:
        """Process video hotspots via Replicate cloud GPU.

        Same two-pass architecture as VideoLLaMA2Discovery:
        1. signal_timeline ranks chunks by signal density
        2. Top max_vlm_chunks are uploaded to Replicate for VLM analysis
        """
        import replicate as replicate_sdk

        os.environ["REPLICATE_API_TOKEN"] = self.api_key

        # Resolve a version pin for community models. The Replicate SDK's
        # "owner/model" form (no ":") hits POST /v1/models/owner/model/predictions,
        # which only works for Replicate-curated *official* models. Community
        # models (like lucataco/videollama3-7b) live under /v1/predictions and
        # require "owner/model:VERSION_ID". Resolve the latest version once
        # per discovery call so users can keep the friendly "owner/model"
        # default in Settings.
        model_ref = self.model_id
        if ":" not in model_ref:
            try:
                model_obj = replicate_sdk.models.get(model_ref)
                version_obj = getattr(model_obj, "latest_version", None)
                if version_obj is not None and getattr(version_obj, "id", None):
                    model_ref = f"{model_ref}:{version_obj.id}"
                    logger.info(
                        f"Replicate: resolved {self.model_id} → "
                        f"{model_ref[:len(self.model_id) + 13]}…"
                    )
            except Exception as e:
                logger.warning(
                    f"Replicate: could not resolve latest version for "
                    f"{self.model_id}: {e} — call may 404"
                )

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
            f"Replicate VLM: {n_total_chunks} total chunks, "
            f"processing top {n_to_process} by signal density"
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
                    preferred_subjects, avoid_subjects, platforms,
                    min_dur_s, max_dur_s, ideal_dur_s,
                    template=discovery_prompt)

                # Upload video chunk to Replicate. idx+1 = position in the
                # selected batch; chunk_idx+1 = position in the full timeline.
                logger.info(
                    f"Replicate: sending chunk {idx+1}/{n_to_process} "
                    f"[timeline {chunk_idx+1}] "
                    f"({_fmt_time(start_s)}–{_fmt_time(end_s)}) to {self.model_id}"
                )

                with open(chunk_path, "rb") as f:
                    output = replicate_sdk.run(
                        model_ref,
                        input={
                            "video": f,
                            "prompt": prompt,
                            "max_new_tokens": 1024,
                            "temperature": 0.1,
                        },
                        use_file_output=False,
                    )

                # Replicate returns an iterator for streaming models
                if hasattr(output, '__iter__') and not isinstance(output, (str, dict)):
                    response_text = "".join(str(chunk) for chunk in output)
                else:
                    response_text = str(output)

                logger.info(
                    f"Replicate chunk {idx+1}/{n_to_process} "
                    f"[timeline {chunk_idx+1}] response: {response_text[:200]}...")

                candidates = _parse_vlm_response(response_text, start_s, end_s)
                for c in candidates:
                    c.signal_score = sig_score
                all_candidates.extend(candidates)

            except Exception as e:
                logger.warning(
                    f"Replicate error on chunk {idx+1}/{n_to_process} "
                    f"[timeline {chunk_idx+1}]: {e}")
            finally:
                if chunk_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

        if on_progress:
            on_progress(1.0)

        return all_candidates

    def _build_discovery_prompt(self, start_s, end_s, transcript_slice,
                                preferred_subjects="", avoid_subjects="",
                                platforms=None, min_dur_s=60, max_dur_s=300,
                                ideal_dur_s=150, template=""):
        """Render the VideoLLaMA discovery prompt.

        Uses the operator-supplied ``template`` when set, else the
        built-in DEFAULT_DISCOVERY_PROMPT. Placeholders are substituted
        literally (str.replace) so a custom template can never raise.
        """
        platform_str = _format_platforms(platforms)
        pref_line = (f"\nPRIORITIZE moments with: {preferred_subjects.strip()}"
                     if preferred_subjects and preferred_subjects.strip() else "")
        avoid_line = (f"\nAVOID: {avoid_subjects.strip()}"
                      if avoid_subjects and avoid_subjects.strip() else "")
        tpl = template.strip() if (template and template.strip()) else DEFAULT_DISCOVERY_PROMPT
        # Structural placeholders first, free-text content last so a
        # transcript / subject string can't be re-substituted.
        for key, val in (
            ("{start}", _fmt_time(start_s)),
            ("{end}", _fmt_time(end_s)),
            ("{platforms}", platform_str),
            ("{min_duration}", str(min_dur_s)),
            ("{max_duration}", str(max_dur_s)),
            ("{ideal_duration}", str(ideal_dur_s)),
            ("{preferred}", pref_line),
            ("{avoid}", avoid_line),
            ("{transcript}", transcript_slice or "(no speech in this segment)"),
        ):
            tpl = tpl.replace(key, val)
        return tpl


# ═══════════════════════════════════════════════════════════════════════════
#  VIDEOLLAMA3 ENHANCED DISCOVERY (Replicate)
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_V3 = """You are an expert viral video editor and content strategist. Your job is to identify the exact moments in video content that will generate maximum engagement on short-form platforms (TikTok, Instagram Reels, YouTube Shorts).

EVALUATION CRITERIA (score each moment mentally):
- Hook strength: Would this stop a scroller in the first 2 seconds?
- Emotional intensity: Does this provoke laughter, shock, awe, anger, or tears?
- Shareability: Would someone tag a friend or repost this?
- Completeness: Does this moment have a natural beginning, peak, and resolution?
- Audio-visual sync: Are the strongest visual and audio moments aligned?

OUTPUT FORMAT: Respond ONLY with a JSON array. No preamble, no markdown, no explanation.
Each element: {"timestamp": "MM:SS", "duration": <int seconds>, "reason": "<one sentence>", "hook": "<opening line>", "confidence": <0.0-1.0>, "category": "<emotional_peak|hot_take|funny|revelation|confrontation|visual_wow>"}
Timestamps are relative to the START of the provided segment."""


REFINEMENT_PROMPT_V3 = """You previously identified a potentially viral moment near {timestamp} in this video segment.

Watch this focused clip carefully. Provide a DETAILED assessment:

1. EXACT optimal start and end timestamps (to the second) for maximum impact
2. The single strongest "hook" frame or moment that should open the clip
3. Whether the moment has a satisfying conclusion or needs the clip trimmed
4. A confidence score (0.0-1.0) now that you've seen it in detail
5. A viral-optimized title (max 60 chars)
6. 3 hashtag suggestions

TRANSCRIPT:
{transcript}
{audio_annotations}

Respond ONLY with JSON:
{{"start": "MM:SS", "end": "MM:SS", "hook_timestamp": "MM:SS", "needs_trim": true/false, "trim_suggestion": "MM:SS-MM:SS", "confidence": 0.0-1.0, "title": "...", "hashtags": ["...", "...", "..."], "reason": "..."}}"""


def _is_rate_limited_error(exc) -> bool:
    """True when a Replicate error is an upstream rate-limit / throttle (429).

    These can't be retried away within a job (the account's per-minute budget is
    exhausted), so the discovery should stop hammering Replicate and fall back to
    the signal-based clips instead of grinding every chunk through 429 retries.
    """
    s = str(exc).lower()
    return (
        "429" in s
        or "too many requests" in s
        or "rate limit" in s
        or "rate-limit" in s
        or "throttl" in s
        or "reduced to" in s          # "...rate limit ... is reduced to N requests..."
        or "quota" in s
    )


class ReplicateDiscoveryV3:
    """VideoLLaMA3-enhanced cloud GPU discovery.

    Wraps Replicate's ``lucataco/videollama3-7b`` with five upgrades over
    the base ``ReplicateDiscovery`` class. Every upgrade has independent
    error handling so a single failed feature never crashes the pipeline:

    1. System + user prompt separation — gives V3's instruction tuning the
       persona/criteria persistently while per-segment context varies.
    2. Audio annotations — V3 has no audio branch, so per-window summaries
       of speech density, RMS energy, face count and motion are injected
       into each prompt so the model knows about events it can't hear.
    3. Adaptive chunk sizing — chunk boundaries are placed at signal
       valleys (natural scene breaks) instead of a rigid 10-minute grid,
       keeping signal-dense regions together for better cross-boundary
       reasoning.
    4. fps / max_frames control — temporal sampling density is scaled by
       chunk signal density. Falls back to the base 4-param call if the
       cog wrapper rejects the new fields.
    5. Two-pass refinement — top candidates by confidence are re-queried
       on a tight sub-clip for surgical timestamp/title precision (mirrors
       the editorial-judge pattern but on V3 itself).

    Method signature matches ``ReplicateDiscovery.discover_clips`` exactly
    so the calling code in ``ClipExtractor.run`` doesn't change.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "lucataco/videollama3-7b",
        fps: int = 2,
        max_frames: int = 128,
        refinement_enabled: bool = True,
        keyframe_analysis: bool = True,
        audio_annotation: bool = True,
        adaptive_chunks: bool = True,
        chunk_min_s: int = 120,
        chunk_max_s: int = 900,
        chunk_workers: int = 3,
        rate_limit_retries: int = 2,
        rate_limit_backoff_s: float = 20.0,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.total_cost_usd: float = 0.0
        self.fps = max(1, min(4, int(fps)))
        self.max_frames = max(16, min(180, int(max_frames)))
        self.refinement_enabled = bool(refinement_enabled)
        self.keyframe_analysis = bool(keyframe_analysis)
        self.audio_annotation = bool(audio_annotation)
        self.adaptive_chunks = bool(adaptive_chunks)
        self.chunk_min_s = max(30, int(chunk_min_s))
        self.chunk_max_s = max(self.chunk_min_s, int(chunk_max_s))
        # Coarse-pass rate-limit resilience: the first wave runs at
        # ``chunk_workers`` concurrency, then any chunk Replicate 429'd is resumed
        # SEQUENTIALLY after a growing backoff, up to ``rate_limit_retries`` times
        # (retries=0 restores the old give-up-immediately behavior).
        self.chunk_workers = max(1, int(chunk_workers))
        self.rate_limit_retries = max(0, int(rate_limit_retries))
        self.rate_limit_backoff_s = max(0.0, float(rate_limit_backoff_s))

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    # ── Public entry point — identical signature to ReplicateDiscovery ──

    def discover_clips(
        self,
        video_path: str,
        transcript_segments: list,
        signal_timeline: 'SignalTimeline',
        chunk_duration_s: int = 600,
        max_vlm_chunks: int = 6,
        preferred_subjects: str = "",
        avoid_subjects: str = "",
        platforms: list = None,
        min_dur_s: int = 60,
        max_dur_s: int = 300,
        ideal_dur_s: int = 150,
        discovery_prompt: str = "",
        on_progress: Callable = None,
    ) -> List['ClipCandidate']:
        import replicate as replicate_sdk

        os.environ["REPLICATE_API_TOKEN"] = self.api_key
        t_start_total = _time.time()
        # Circuit breaker: once Replicate returns a 429/throttle, stop sending
        # the remaining chunks (and skip the refine/keyframe passes) so a
        # rate-limited account doesn't hang clip detection for many minutes —
        # the caller falls back to the signal-based clips instead.
        self._rate_limited = threading.Event()

        model_ref = self._resolve_model_version(replicate_sdk)
        duration_s = _get_video_duration(video_path)

        # ── Step 1: build chunks (adaptive or rigid) ──
        chunks = self._build_chunks(signal_timeline, duration_s, chunk_duration_s)

        # ── Step 2: rank chunks by signal density, keep top-N ──
        ranked = sorted(
            (
                (i, s, e, score_chunk_signals(signal_timeline, s, e))
                for i, (s, e) in enumerate(chunks)
            ),
            key=lambda x: x[3],
            reverse=True,
        )
        n_to_process = min(len(ranked), max_vlm_chunks)
        selected = sorted(ranked[:n_to_process], key=lambda x: x[1])

        logger.info(
            "VideoLLaMA3-V3: %s chunking produced %d chunks (%s)",
            "adaptive" if self.adaptive_chunks else "rigid",
            len(chunks),
            ", ".join(f"{_fmt_time(s)}-{_fmt_time(e)}" for s, e in chunks[:8])
            + ("…" if len(chunks) > 8 else ""),
        )
        logger.info(
            "VideoLLaMA3-V3: sending top %d/%d chunks to Replicate (%s)",
            n_to_process, len(chunks), self.model_id,
        )

        # ── Step 3: coarse pass ──
        # The first wave runs chunks concurrently (fast on healthy accounts). On
        # a Replicate 429 the tripped chunks are RESUMED sequentially after a
        # backoff (see _coarse_pass_with_resume) instead of abandoning the rest
        # of the video — so a low-tier / rate-limited account still gets
        # full-video VLM coverage rather than only the first chunk or two.
        def _dispatch_chunk(args):
            idx, chunk_idx, start_s, end_s, sig_score = args
            return self._coarse_pass_chunk(
                replicate_sdk, model_ref, video_path, signal_timeline,
                transcript_segments, start_s, end_s, sig_score,
                preferred_subjects, avoid_subjects, platforms,
                min_dur_s, max_dur_s, ideal_dur_s, discovery_prompt,
                chunk_idx=idx + 1, n_total=n_to_process,
                timeline_idx=chunk_idx + 1, n_timeline=len(chunks),
            )

        chunk_args = [
            (idx, chunk_idx, start_s, end_s, sig_score)
            for idx, (chunk_idx, start_s, end_s, sig_score) in enumerate(selected)
        ]

        if on_progress:
            on_progress(0.0)

        all_candidates: List[ClipCandidate] = self._coarse_pass_with_resume(
            chunk_args, _dispatch_chunk, n_to_process, on_progress)

        coarse_count = len(all_candidates)

        # ── Step 4: refinement pass on top candidates ──
        # Skip when rate-limited — more Replicate calls would just 429 too.
        refined_count = 0
        if self.refinement_enabled and all_candidates and not self._rate_limited.is_set():
            try:
                refined_count = self._refinement_pass(
                    replicate_sdk, model_ref, video_path, signal_timeline,
                    transcript_segments, all_candidates,
                )
            except Exception as e:
                logger.warning("VideoLLaMA3-V3: refinement pass failed (%s) — using coarse results", e)

        # ── Step 5: optional keyframe image analysis on top candidates ──
        if self.keyframe_analysis and all_candidates and not self._rate_limited.is_set():
            try:
                self._keyframe_pass(
                    replicate_sdk, model_ref, video_path, signal_timeline,
                    transcript_segments, all_candidates,
                )
            except Exception as e:
                logger.warning("VideoLLaMA3-V3: keyframe analysis failed (%s) — skipping", e)

        if on_progress:
            on_progress(1.0)

        elapsed = _time.time() - t_start_total
        # Replicate cost ≈ $0.011 per coarse chunk + $0.005 per refine + keyframe
        est_cost = (
            n_to_process * 0.011
            + refined_count * 0.005
            + (min(3, len(all_candidates)) * 0.003 if self.keyframe_analysis else 0)
        )
        # Persist the running spend so the pipeline can sum it with the
        # OpenRouter / Anthropic / Gemini totals for the analysis card.
        try:
            self.total_cost_usd = float(self.total_cost_usd or 0.0) + float(est_cost)
        except Exception:
            pass
        logger.info(
            "VideoLLaMA3-V3: total discovery: %d chunks, %d candidates (%d coarse, %d refined) — %.1fs, ~$%.2f",
            n_to_process, len(all_candidates), coarse_count, refined_count, elapsed, est_cost,
        )
        return all_candidates

    # ── Replicate version pinning (same logic as base class) ──

    def _resolve_model_version(self, replicate_sdk) -> str:
        model_ref = self.model_id
        if ":" in model_ref:
            return model_ref
        try:
            model_obj = replicate_sdk.models.get(model_ref)
            version_obj = getattr(model_obj, "latest_version", None)
            if version_obj is not None and getattr(version_obj, "id", None):
                model_ref = f"{model_ref}:{version_obj.id}"
                logger.info(
                    "VideoLLaMA3-V3: resolved %s → %s…",
                    self.model_id, model_ref[:len(self.model_id) + 13],
                )
        except Exception as e:
            logger.warning(
                "VideoLLaMA3-V3: could not resolve latest version for %s: %s — call may 404",
                self.model_id, e,
            )
        return model_ref

    # ── Chunking: adaptive (signal-valley splits) or rigid fallback ──

    def _build_chunks(self, signal_timeline, duration_s: float, rigid_chunk_s: int):
        """Return List[(start_s, end_s)] covering the whole video."""
        if not self.adaptive_chunks:
            return self._rigid_chunks(duration_s, rigid_chunk_s)
        try:
            chunks = self._compute_adaptive_chunks(
                signal_timeline, duration_s,
                min_chunk_s=self.chunk_min_s,
                max_chunk_s=self.chunk_max_s,
            )
            if not chunks:
                raise ValueError("adaptive chunker returned no chunks")
            return chunks
        except Exception as e:
            logger.warning(
                "VideoLLaMA3-V3: adaptive chunking failed (%s) — falling back to rigid %ds chunks",
                e, rigid_chunk_s,
            )
            return self._rigid_chunks(duration_s, rigid_chunk_s)

    @staticmethod
    def _rigid_chunks(duration_s: float, chunk_s: int):
        n = max(1, math.ceil(duration_s / max(1, chunk_s)))
        return [
            (int(i * chunk_s), int(min((i + 1) * chunk_s, duration_s)))
            for i in range(n)
        ]

    def _compute_adaptive_chunks(self, signal_timeline, duration_s, min_chunk_s=120, max_chunk_s=900):
        """Create variable-length chunks based on signal density.

        Dense-signal regions get shorter chunks (more VLM attention per
        second). Sparse-signal regions get longer chunks (less wasted API
        calls). Signal valleys are preferred as chunk edges so a "scene
        break" doesn't fall mid-chunk and lose context.
        """
        duration_s = int(duration_s)
        if duration_s <= max_chunk_s:
            return [(0, duration_s)]

        composite = [
            score_chunk_signals(signal_timeline, s, s + 1)
            for s in range(duration_s)
        ]
        # Wider smoothing kernel so per-second jitter doesn't manufacture
        # local minima on every frame.
        kernel = 30
        smoothed = []
        for i in range(len(composite)):
            window = composite[max(0, i - kernel):i + kernel]
            smoothed.append(sum(window) / max(1, len(window)))

        # Only flag genuine valleys — points strictly lower than their
        # immediate neighbors AND meaningfully lower than the surrounding
        # window. Without the significance gate a plateau or noise floor
        # becomes a candidate at every position, forcing the chunker into
        # minimum-sized chunks on flat / sparse-signal content.
        avg_signal = sum(smoothed) / max(1, len(smoothed)) if smoothed else 0
        significance_floor = max(1e-3, avg_signal * 0.05)
        split_candidates = []
        for i in range(kernel, len(smoothed) - kernel):
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                wide = smoothed[i - kernel:i + kernel + 1]
                if not wide:
                    continue
                wide_avg = sum(wide) / len(wide)
                # Require a real dip: at least 5% below the local window
                # average (or below the global noise floor).
                if smoothed[i] + significance_floor <= wide_avg:
                    split_candidates.append(i)

        chunks = []
        prev = 0
        for sp in split_candidates:
            chunk_len = sp - prev
            if chunk_len < min_chunk_s:
                continue
            if chunk_len <= max_chunk_s:
                chunks.append((prev, sp))
                prev = sp
            else:
                while sp - prev > max_chunk_s:
                    chunks.append((prev, prev + max_chunk_s))
                    prev = prev + max_chunk_s
                if sp - prev >= min_chunk_s:
                    chunks.append((prev, sp))
                    prev = sp

        remaining = duration_s - prev
        if remaining > 0:
            if remaining < min_chunk_s and chunks:
                last_start, _ = chunks.pop()
                chunks.append((last_start, duration_s))
            else:
                while duration_s - prev > max_chunk_s:
                    chunks.append((prev, prev + max_chunk_s))
                    prev = prev + max_chunk_s
                if duration_s > prev:
                    chunks.append((prev, duration_s))

        return chunks or [(0, duration_s)]

    # ── Per-chunk fps scaling ──

    def _compute_chunk_fps(self, signal_timeline, start_s, end_s, max_fps=4):
        score = score_chunk_signals(signal_timeline, start_s, end_s)
        if score > 0.7:
            return min(max_fps, self.fps + 1)
        if score < 0.3:
            return max(1, self.fps - 1)
        return self.fps

    # ── Audio annotations (V3 can't hear, so we describe what it can't) ──

    def _build_audio_annotation(self, signal_timeline, start_s, end_s):
        if not self.audio_annotation or signal_timeline is None:
            return ""
        # Signal arrays — handle both attribute names ('audio_rms' is the
        # spec name but the actual SignalTimeline class stores RMS as
        # 'rms_energy'). getattr keeps either path working.
        speech = list(getattr(signal_timeline, "speech_density", []))
        rms = list(getattr(signal_timeline, "audio_rms", None)
                   or getattr(signal_timeline, "rms_energy", []))
        faces = list(getattr(signal_timeline, "face_count", []))
        motion = list(getattr(signal_timeline, "motion_magnitude", []))

        n = max(len(speech), len(rms), len(faces), len(motion))
        if n == 0:
            return ""

        start_s_i = max(0, int(start_s))
        end_s_i = min(n, int(end_s))
        if end_s_i <= start_s_i:
            return ""

        window_s = 15
        annotations = []
        for ws in range(start_s_i, end_s_i, window_s):
            we = min(ws + window_s, end_s_i)
            sp = speech[ws:we]
            rm = rms[ws:we]
            fc = faces[ws:we]
            mo = motion[ws:we]

            avg_speech = sum(sp) / max(1, len(sp)) if sp else 0
            avg_rms = sum(rm) / max(1, len(rm)) if rm else 0
            avg_faces = sum(fc) / max(1, len(fc)) if fc else 0
            avg_motion = sum(mo) / max(1, len(mo)) if mo else 0

            notable = []
            if avg_speech > 0.5:
                notable.append("active speech")
            elif avg_speech > 0.0:
                notable.append("sparse speech")
            if avg_rms > 0.7:
                notable.append("loud audio (possible laughter/applause/music)")
            elif avg_rms > 0.4:
                notable.append("moderate audio")
            if avg_faces >= 2:
                notable.append(f"{int(avg_faces)} faces visible")
            # motion_magnitude in SignalTimeline is normalised by frame count
            # but not 0-1; >0.6 of the local average is a reasonable spike.
            if avg_motion > 0.6:
                notable.append("high motion")

            if notable:
                rel_start = ws - start_s_i
                rel_end = we - start_s_i
                annotations.append(
                    f"  {_fmt_time(rel_start)}-{_fmt_time(rel_end)}: {', '.join(notable)}"
                )

        if not annotations:
            return ""

        return (
            "\n\nAUDIO/SIGNAL ANNOTATIONS (you cannot hear audio — "
            "these are machine-detected audio and motion signals):\n"
            + "\n".join(annotations)
        )

    # ── Coarse pass: one Replicate call per selected chunk ──

    def _coarse_pass_chunk(
        self, replicate_sdk, model_ref, video_path, signal_timeline,
        transcript_segments, start_s, end_s, sig_score,
        preferred_subjects, avoid_subjects, platforms,
        min_dur_s, max_dur_s, ideal_dur_s, discovery_prompt,
        chunk_idx, n_total,
        timeline_idx=None, n_timeline=None,
    ):
        # Circuit breaker tripped by an earlier chunk's 429 — skip all work
        # (no fps probe, no upload, no Replicate call) and fall back to signals.
        if getattr(self, "_rate_limited", None) is not None and self._rate_limited.is_set():
            logger.info(
                "VideoLLaMA3-V3: coarse pass chunk %d/%d skipped (Replicate rate-limited) "
                "— will resume after a backoff",
                chunk_idx, n_total,
            )
            return None
        chunk_path = None
        chunk_fps = self._compute_chunk_fps(signal_timeline, start_s, end_s)
        t_start = _time.time()
        timeline_suffix = (
            f" [timeline {timeline_idx}/{n_timeline}]"
            if timeline_idx is not None and n_timeline is not None else ""
        )
        try:
            chunk_path = _extract_chunk(video_path, start_s, end_s)
            transcript_slice = _slice_transcript(transcript_segments, start_s, end_s)
            audio_annotation = self._build_audio_annotation(signal_timeline, start_s, end_s)

            segment_prompt = self._build_segment_prompt(
                start_s, end_s, transcript_slice, audio_annotation,
                preferred_subjects, avoid_subjects, platforms,
                min_dur_s, max_dur_s, ideal_dur_s, discovery_prompt,
            )
            full_prompt = f"[SYSTEM]\n{SYSTEM_PROMPT_V3}\n\n[USER]\n{segment_prompt}"

            logger.info(
                "VideoLLaMA3-V3: coarse pass chunk %d/%d%s (%s-%s, fps=%d, max_frames=%d)",
                chunk_idx, n_total, timeline_suffix,
                _fmt_time(start_s), _fmt_time(end_s),
                chunk_fps, self.max_frames,
            )

            response_text = self._call_replicate_video(
                replicate_sdk, model_ref, chunk_path, full_prompt,
                fps=chunk_fps, max_frames=self.max_frames, max_new_tokens=2048,
            )

            candidates = _parse_vlm_response(response_text, start_s, end_s)
            if not candidates and "[SYSTEM]" in full_prompt:
                # JSON parsing failed — retry once with a flat prompt in
                # case the SYSTEM/USER tags confused the model.
                logger.info(
                    "VideoLLaMA3-V3: coarse pass chunk %d/%d%s returned no candidates — retrying flat prompt",
                    chunk_idx, n_total, timeline_suffix,
                )
                response_text = self._call_replicate_video(
                    replicate_sdk, model_ref, chunk_path, segment_prompt,
                    fps=chunk_fps, max_frames=self.max_frames, max_new_tokens=2048,
                )
                candidates = _parse_vlm_response(response_text, start_s, end_s)

            for c in candidates:
                c.signal_score = sig_score
                c.source = "vlm_discovery_v3"

            elapsed = _time.time() - t_start
            logger.info(
                "VideoLLaMA3-V3: coarse pass chunk %d/%d%s — %.1fs — %d candidates",
                chunk_idx, n_total, timeline_suffix, elapsed, len(candidates),
            )
            return candidates

        except Exception as e:
            if _is_rate_limited_error(e) and getattr(self, "_rate_limited", None) is not None:
                if not self._rate_limited.is_set():
                    self._rate_limited.set()
                    logger.warning(
                        "VideoLLaMA3-V3: Replicate rate-limited (429) on chunk %d/%d%s — "
                        "remaining chunks will resume sequentially after a backoff.",
                        chunk_idx, n_total, timeline_suffix,
                    )
                return None  # retryable — resumed by _coarse_pass_with_resume
            logger.warning(
                "VideoLLaMA3-V3: coarse pass chunk %d/%d%s failed: %s",
                chunk_idx, n_total, timeline_suffix, e,
            )
            return []
        finally:
            if chunk_path and os.path.exists(chunk_path):
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

    def _coarse_pass_with_resume(self, chunk_args, run_chunk, n_to_process,
                                 on_progress=None):
        """Run coarse-pass chunks with Replicate rate-limit resilience.

        ``run_chunk(args)`` returns a list of candidates, or ``None`` when the
        chunk was rate-limited / skipped (retryable). The first wave runs
        concurrently (``self.chunk_workers``); any chunk that comes back ``None``
        is RESUMED in up to ``self.rate_limit_retries`` follow-up waves that run
        SEQUENTIALLY (1 worker) after a growing backoff, clearing the circuit
        breaker each round — so a low-tier account still covers the whole video
        instead of abandoning most of it on the first 429. Returns the flat
        candidate list. Pure orchestration (no Replicate/video I/O of its own),
        so it's unit-testable with a fake ``run_chunk``."""
        all_candidates: list = []
        _done = [0]
        _lock = threading.Lock()

        def _wave(args_list, workers):
            skipped = []
            if not args_list:
                return skipped
            with _cf.ThreadPoolExecutor(
                    max_workers=max(1, min(len(args_list), workers))) as _ex:
                _futs = {_ex.submit(run_chunk, a): a for a in args_list}
                for _fut in _cf.as_completed(_futs):
                    try:
                        _res = _fut.result()
                    except Exception as _e:
                        logger.warning("VideoLLaMA3-V3: chunk dispatch raised: %s", _e)
                        _res = []
                    if _res is None:           # rate-limited / skipped → retry later
                        skipped.append(_futs[_fut])
                        continue
                    all_candidates.extend(_res)
                    with _lock:
                        _done[0] += 1
                        _d = _done[0]
                    if on_progress:
                        on_progress((min(_d, n_to_process) / max(1, n_to_process)) * 0.75)
            return skipped

        pending = _wave(chunk_args, max(1, int(getattr(self, "chunk_workers", 3))))
        _retries = max(0, int(getattr(self, "rate_limit_retries", 0)))
        _backoff_base = max(0.0, float(getattr(self, "rate_limit_backoff_s", 0.0)))
        _attempt = 0
        while pending and _attempt < _retries:
            _attempt += 1
            _backoff = _backoff_base * _attempt
            logger.info(
                "VideoLLaMA3-V3: Replicate rate-limited — backing off %.0fs, then "
                "resuming %d chunk(s) sequentially (resume %d/%d).",
                _backoff, len(pending), _attempt, _retries,
            )
            if getattr(self, "_rate_limited", None) is not None:
                self._rate_limited.clear()      # let the resumed chunks actually run
            if _backoff > 0:
                _time.sleep(_backoff)
            pending = _wave(pending, 1)          # sequential resume avoids re-tripping

        if getattr(self, "_rate_limited", None) is not None:
            # Leave the breaker tripped only if we truly gave up (so the
            # refine/keyframe passes still skip); otherwise clear it.
            if pending:
                self._rate_limited.set()
            else:
                self._rate_limited.clear()
        if pending:
            logger.warning(
                "VideoLLaMA3-V3: %d chunk(s) still rate-limited after %d resume "
                "attempt(s) — those spans fall back to signal-based clips.",
                len(pending), _retries,
            )
        return all_candidates

    def _build_segment_prompt(
        self, start_s, end_s, transcript_slice, audio_annotation,
        preferred_subjects, avoid_subjects, platforms,
        min_dur_s, max_dur_s, ideal_dur_s, discovery_prompt,
    ):
        """Per-chunk user message. Mirrors the placeholder substitution of
        the base ``ReplicateDiscovery`` so a user's custom prompt template
        keeps working under V3, but appends the audio annotations."""
        platform_str = _format_platforms(platforms)
        pref_line = (f"\nPRIORITIZE moments with: {preferred_subjects.strip()}"
                     if preferred_subjects and preferred_subjects.strip() else "")
        avoid_line = (f"\nAVOID: {avoid_subjects.strip()}"
                      if avoid_subjects and avoid_subjects.strip() else "")
        tpl = discovery_prompt.strip() if (discovery_prompt and discovery_prompt.strip()) else DEFAULT_DISCOVERY_PROMPT
        for key, val in (
            ("{start}", _fmt_time(start_s)),
            ("{end}", _fmt_time(end_s)),
            ("{platforms}", platform_str),
            ("{min_duration}", str(min_dur_s)),
            ("{max_duration}", str(max_dur_s)),
            ("{ideal_duration}", str(ideal_dur_s)),
            ("{preferred}", pref_line),
            ("{avoid}", avoid_line),
            ("{transcript}", transcript_slice or "(no speech in this segment)"),
        ):
            tpl = tpl.replace(key, val)
        return tpl + (audio_annotation or "")

    # ── Replicate transport — wraps fps/max_frames params with safe fallback ──

    def _call_replicate_video(
        self, replicate_sdk, model_ref, chunk_path, prompt,
        fps=None, max_frames=None, max_new_tokens=2048,
    ):
        """Run the model with V3-specific extras; on bad-input errors fall
        back to the legacy 4-param call so an unsupported field never
        crashes the pipeline."""
        base_input = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,
        }
        extras = {}
        if fps is not None:
            extras["fps"] = int(fps)
        if max_frames is not None:
            extras["max_frames"] = int(max_frames)
        # top_p is also V3-only; include in extras so the same fallback
        # rescues us if the cog rejects it.
        extras["top_p"] = 0.9

        try:
            with open(chunk_path, "rb") as f:
                output = replicate_sdk.run(
                    model_ref,
                    input={**base_input, "video": f, **extras},
                    use_file_output=False,
                )
            return self._collect_output(output)
        except Exception as e:
            msg = str(e).lower()
            # Replicate raises ReplicateError("Invalid input: ...") for
            # unknown fields. Retry once without the V3 extras.
            if "invalid" in msg or "input" in msg or "unknown" in msg or "unexpected" in msg:
                logger.warning(
                    "VideoLLaMA3-V3: Replicate rejected V3 params (%s) — retrying with base 4-param call",
                    e,
                )
                with open(chunk_path, "rb") as f:
                    output = replicate_sdk.run(
                        model_ref,
                        input={**base_input, "video": f},
                        use_file_output=False,
                    )
                return self._collect_output(output)
            raise

    @staticmethod
    def _collect_output(output) -> str:
        if hasattr(output, "__iter__") and not isinstance(output, (str, bytes, dict)):
            return "".join(str(chunk) for chunk in output)
        return str(output)

    # ── Refinement pass: re-query top candidates with tight sub-clips ──

    def _refinement_pass(
        self, replicate_sdk, model_ref, video_path, signal_timeline,
        transcript_segments, candidates,
    ):
        # Pick top 3 by confidence (fallback: signal_score) so we don't burn
        # API calls on every coarse candidate.
        scored = sorted(
            candidates,
            key=lambda c: (
                _candidate_confidence(c),
                getattr(c, "signal_score", 0.0),
            ),
            reverse=True,
        )
        top = scored[:3]
        if not top:
            return 0

        t_start = _time.time()
        refined = 0
        for c in top:
            sub_start = max(0.0, c.start_s - 5.0)
            sub_end = c.end_s + 5.0
            # Cap sub-clip to a sane 60s upper bound so we don't re-upload
            # a long segment when the coarse pass over-estimated duration.
            if sub_end - sub_start > 60:
                center = (c.start_s + c.end_s) / 2
                sub_start = max(0.0, center - 30.0)
                sub_end = sub_start + 60.0

            chunk_path = None
            try:
                chunk_path = _extract_chunk(video_path, sub_start, sub_end)
                transcript_slice = _slice_transcript(transcript_segments, sub_start, sub_end)
                audio_annotation = self._build_audio_annotation(signal_timeline, sub_start, sub_end)

                user_prompt = REFINEMENT_PROMPT_V3.format(
                    timestamp=_fmt_time(c.start_s),
                    transcript=transcript_slice,
                    audio_annotations=audio_annotation,
                )
                full_prompt = f"[SYSTEM]\n{SYSTEM_PROMPT_V3}\n\n[USER]\n{user_prompt}"

                response_text = self._call_replicate_video(
                    replicate_sdk, model_ref, chunk_path, full_prompt,
                    fps=self.fps, max_frames=min(self.max_frames, 64),
                    max_new_tokens=1024,
                )

                if self._apply_refinement(c, response_text, sub_start, sub_end):
                    refined += 1
            except Exception as e:
                logger.warning(
                    "VideoLLaMA3-V3: refinement on candidate @%s failed: %s",
                    _fmt_time(c.start_s), e,
                )
            finally:
                if chunk_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except OSError:
                        pass

        elapsed = _time.time() - t_start
        logger.info(
            "VideoLLaMA3-V3: refinement pass on %d top candidates — %.1fs total (%d refined)",
            len(top), elapsed, refined,
        )
        return refined

    @staticmethod
    def _apply_refinement(candidate, response_text, sub_start, sub_end):
        text = response_text.strip()
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return False
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return False
        if not isinstance(data, dict):
            return False

        changed = False
        # Refined absolute timestamps (relative to sub_start)
        try:
            new_rel_start = _parse_mmss(str(data.get("start", "")))
            new_rel_end = _parse_mmss(str(data.get("end", "")))
            if new_rel_end > new_rel_start > 0:
                abs_start = sub_start + new_rel_start
                abs_end = sub_start + new_rel_end
                # Sanity: keep within the sub-clip window
                abs_start = max(sub_start, min(abs_start, sub_end))
                abs_end = max(abs_start, min(abs_end, sub_end))
                if abs_end - abs_start >= 5:
                    candidate.start_s = float(abs_start)
                    candidate.end_s = float(abs_end)
                    candidate.duration_s = float(abs_end - abs_start)
                    changed = True
        except Exception:
            pass

        title = data.get("title")
        if title and isinstance(title, str):
            candidate.judge_title = title.strip()[:60]
            changed = True

        reason = data.get("reason")
        if reason and isinstance(reason, str):
            candidate.vlm_reason = reason.strip()
            changed = True

        # Stash the refined confidence on judge_scores so downstream
        # ranking can use it without breaking the existing schema.
        try:
            conf = float(data.get("confidence", 0))
            if conf > 0:
                candidate.judge_scores = candidate.judge_scores or {}
                candidate.judge_scores["v3_confidence"] = conf
                changed = True
        except (TypeError, ValueError):
            pass

        hashtags = data.get("hashtags")
        if isinstance(hashtags, list) and hashtags:
            candidate.judge_scores = candidate.judge_scores or {}
            candidate.judge_scores["v3_hashtags"] = [
                str(h).strip() for h in hashtags[:3] if str(h).strip()
            ]
            changed = True

        return changed

    # ── Keyframe image-mode analysis (optional) ──

    def _keyframe_pass(
        self, replicate_sdk, model_ref, video_path, signal_timeline,
        transcript_segments, candidates,
    ):
        # Only run on the top 3 candidates to keep cost predictable.
        scored = sorted(
            candidates,
            key=lambda c: (
                _candidate_confidence(c),
                getattr(c, "signal_score", 0.0),
            ),
            reverse=True,
        )
        for c in scored[:3]:
            try:
                # Sample 2 keyframes from the candidate window.
                frames_b64 = _extract_keyframes_b64(
                    video_path, c.start_s, c.end_s, n_frames=2,
                )
                if not frames_b64:
                    continue
                transcript_slice = _slice_transcript(transcript_segments, c.start_s, c.end_s)
                audio_annotation = self._build_audio_annotation(signal_timeline, c.start_s, c.end_s)

                prompt = (
                    f"[SYSTEM]\n{SYSTEM_PROMPT_V3}\n\n"
                    f"[USER]\nThese are keyframes from a potential viral moment at "
                    f"{_fmt_time(c.start_s)}-{_fmt_time(c.end_s)}.\n\n"
                    f"TRANSCRIPT: {transcript_slice}\n"
                    f"{audio_annotation}\n\n"
                    "Analyze what makes this moment visually compelling. "
                    "What is the strongest single frame for a thumbnail? "
                    "Is there visual payoff (reaction, reveal, transformation)?\n\n"
                    'Respond with JSON: {"visual_score": 0.0-1.0, '
                    '"best_thumbnail_frame": 1-2, '
                    '"visual_hook": "what the viewer sees that stops the scroll", '
                    '"has_visual_payoff": true/false}'
                )

                # Try image-mode first (V3 supports it natively, but the cog
                # wrapper may not expose `image` as an input). On rejection,
                # skip silently.
                try:
                    output = replicate_sdk.run(
                        model_ref,
                        input={
                            "image": f"data:image/jpeg;base64,{frames_b64[0]}",
                            "prompt": prompt,
                            "max_new_tokens": 512,
                            "temperature": 0.1,
                        },
                        use_file_output=False,
                    )
                except Exception as e:
                    logger.warning(
                        "VideoLLaMA3-V3: image-mode rejected by cog wrapper (%s) — skipping keyframe analysis",
                        e,
                    )
                    return

                text = self._collect_output(output).strip()
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if not match:
                        continue
                    try:
                        data = json.loads(match.group())
                    except json.JSONDecodeError:
                        continue
                if not isinstance(data, dict):
                    continue

                try:
                    visual_score = float(data.get("visual_score", 0))
                except (TypeError, ValueError):
                    visual_score = 0
                visual_hook = str(data.get("visual_hook", "")).strip()
                has_payoff = bool(data.get("has_visual_payoff", False))

                c.judge_scores = c.judge_scores or {}
                c.judge_scores["v3_visual_score"] = visual_score
                c.judge_scores["v3_has_visual_payoff"] = has_payoff
                if visual_hook and not c.vlm_hook:
                    c.vlm_hook = visual_hook
                logger.info(
                    "VideoLLaMA3-V3: keyframe analysis on candidate @%s — visual_score=%.2f",
                    _fmt_time(c.start_s), visual_score,
                )
            except Exception as e:
                logger.warning(
                    "VideoLLaMA3-V3: keyframe analysis on candidate @%s failed: %s",
                    _fmt_time(getattr(c, "start_s", 0)), e,
                )


def _candidate_confidence(c) -> float:
    """Best available confidence signal for ranking V3 candidates."""
    js = getattr(c, "judge_scores", None) or {}
    if "v3_confidence" in js:
        try:
            return float(js["v3_confidence"])
        except (TypeError, ValueError):
            pass
    return float(getattr(c, "composite_score", 0) or getattr(c, "signal_score", 0) or 0)


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
                f"clip-worthy moments (1-5 min) for {platform_str}.\n"
                f"{pref}{avoid}\n\n"
                f"Respond ONLY with JSON:\n"
                f'[{{"timestamp":"MM:SS","duration":150,"reason":"...","hook":"..."}}]'
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


class AnthropicJudge:
    """Anthropic Claude editorial judge (Messages API, vision-capable)."""

    ENDPOINT = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-5"):
        self.api_key = api_key
        self.model = model

    def judge(self, candidate: ClipCandidate, transcript_slice: str,
              keyframes_b64: List[str], signal_summary: str,
              preferred_subjects: str = "",
              avoid_subjects: str = "") -> dict:
        import urllib.request

        prompt = _build_judge_prompt(
            candidate, transcript_slice, signal_summary,
            preferred_subjects, avoid_subjects)

        content = []
        for img_b64 in keyframes_b64[:6]:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            })
        content.append({"type": "text", "text": prompt})

        payload = json.dumps({
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": content}],
        }).encode()

        req = urllib.request.Request(
            self.ENDPOINT, data=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.VERSION,
                "Content-Type": "application/json",
            },
            method='POST',
        )

        try:
            resp = urllib.request.urlopen(req, timeout=45)
            data = json.loads(resp.read().decode())
            blocks = data.get("content") or []
            text = next(
                (b.get("text", "") for b in blocks if b.get("type") == "text"),
                "")
            return _parse_judge_response(text)
        except Exception as e:
            logger.warning(f"Anthropic judge error: {e}")
            return {"error": str(e)}


class GroqJudge:
    """Groq editorial judge (OpenAI-compatible Chat Completions)."""

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str = "llama-3.2-90b-vision-preview"):
        self.api_key = api_key
        self.model = model

    def judge(self, candidate: ClipCandidate, transcript_slice: str,
              keyframes_b64: List[str], signal_summary: str,
              preferred_subjects: str = "",
              avoid_subjects: str = "") -> dict:
        import urllib.request

        prompt = _build_judge_prompt(
            candidate, transcript_slice, signal_summary,
            preferred_subjects, avoid_subjects)

        # Groq vision quirk: their llama-3.2 vision endpoint accepts at
        # most ONE image per request. We send the middle keyframe (it
        # tends to be the most representative); text-only models drop
        # all images automatically below.
        content = [{"type": "text", "text": prompt}]
        if keyframes_b64:
            mid = keyframes_b64[len(keyframes_b64) // 2]
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{mid}"},
            })

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }).encode()

        req = urllib.request.Request(
            self.ENDPOINT, data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method='POST',
        )

        try:
            resp = urllib.request.urlopen(req, timeout=45)
            data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            return _parse_judge_response(text)
        except urllib.error.HTTPError as e:
            if e.code in (400, 422):
                logger.info(
                    f"Groq model {self.model} rejected images, "
                    f"retrying text-only")
                return self._judge_text_only(prompt)
            logger.warning(f"Groq judge error: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.warning(f"Groq judge error: {e}")
            return {"error": str(e)}

    def _judge_text_only(self, prompt: str) -> dict:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
        }).encode()
        req = urllib.request.Request(
            self.ENDPOINT, data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method='POST',
        )
        try:
            resp = urllib.request.urlopen(req, timeout=45)
            data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"]
            return _parse_judge_response(text)
        except Exception as e:
            logger.warning(f"Groq text-only judge error: {e}")
            return {"error": str(e)}


class OllamaJudge:
    """Ollama editorial judge — local models via /api/chat.

    Vision is supported for multimodal models (llava, llama3.2-vision,
    moondream, qwen2.5vl, etc.) by passing the ``images`` field on the
    user message. Non-vision models simply ignore it.
    """

    def __init__(self, host: str, model: str = "llama3.2-vision:11b"):
        self.host = host.rstrip("/")
        self.model = model

    def judge(self, candidate: ClipCandidate, transcript_slice: str,
              keyframes_b64: List[str], signal_summary: str,
              preferred_subjects: str = "",
              avoid_subjects: str = "") -> dict:
        import urllib.request

        prompt = _build_judge_prompt(
            candidate, transcript_slice, signal_summary,
            preferred_subjects, avoid_subjects)

        msg = {"role": "user", "content": prompt}
        if keyframes_b64:
            msg["images"] = keyframes_b64[:4]

        payload = json.dumps({
            "model": self.model,
            "messages": [msg],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        }).encode()

        url = f"{self.host}/api/chat"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method='POST',
        )

        try:
            # Local inference can be slow; Ollama needs the longer ceiling.
            resp = urllib.request.urlopen(req, timeout=120)
            data = json.loads(resp.read().decode())
            text = (data.get("message") or {}).get("content", "")
            return _parse_judge_response(text)
        except Exception as e:
            logger.warning(f"Ollama judge error: {e}")
            return {"error": str(e)}


def _build_judge_from_spec(spec: str):
    """Construct a judge from a "<backend>:<model>" spec string.

    Reads the API key for the named provider from the app-wide
    settings (the same place the Settings page writes to). Returns
    ``None`` if the spec is empty, malformed, or names a provider
    whose key isn't configured.
    """
    if not spec or ':' not in spec:
        return None
    backend, model = spec.split(':', 1)
    backend = backend.strip().lower()
    model = model.strip()
    if not backend or not model:
        return None

    # Lazy import to avoid pulling pydantic_settings at module import
    # time (this module is imported in places where the app settings
    # aren't yet initialised, e.g. tests).
    try:
        from backend.config import settings
    except Exception as e:
        logger.warning(f"Editorial judge: cannot import settings ({e})")
        return None

    if backend == "gemini" or backend == "google_ai":
        key = settings.GEMINI_API_KEY
        if not key:
            logger.warning(f"Judge spec '{spec}' needs GEMINI_API_KEY, not set")
            return None
        return GeminiJudge(key, model)
    if backend == "openrouter":
        key = settings.OPENROUTER_API_KEY
        if not key:
            logger.warning(f"Judge spec '{spec}' needs OPENROUTER_API_KEY, not set")
            return None
        return OpenRouterJudge(key, model)
    if backend == "anthropic":
        key = settings.ANTHROPIC_API_KEY
        if not key:
            logger.warning(f"Judge spec '{spec}' needs ANTHROPIC_API_KEY, not set")
            return None
        return AnthropicJudge(key, model)
    if backend == "groq":
        key = settings.GROQ_API_KEY
        if not key:
            logger.warning(f"Judge spec '{spec}' needs GROQ_API_KEY, not set")
            return None
        return GroqJudge(key, model)
    if backend == "ollama":
        host = settings.OLLAMA_HOST
        if not host:
            logger.warning(f"Judge spec '{spec}' needs OLLAMA_HOST, not set")
            return None
        return OllamaJudge(host, model)

    logger.warning(f"Unknown editorial judge backend: {backend}")
    return None


class FallbackJudge:
    """Wraps a primary judge with a backup judge.

    Behaviour:
    * Calls primary first; on success, returns its result and resets
      the failure counter.
    * On a *transient* primary failure (timeout, 429, 5xx, network) OR a
      cloud key-limit / billing / quota error, falls through to the fallback
      (a local judge has no key limit) and returns whatever it produces.
    * On a *permanent* primary failure (auth, model-not-found, JSON parse),
      returns the primary error unchanged — there is no point burning
      fallback budget on a genuine misconfiguration.
    * If the primary has failed transiently ``STICKY_AFTER`` times in
      a row, subsequent candidates skip the primary entirely and call
      the fallback directly for the rest of the batch. This keeps
      total wall-clock bounded when the primary is fully down — we
      pay the primary's timeout once per (STICKY_AFTER) candidates,
      not once per candidate. The counter resets on the next primary
      success.
    """

    STICKY_AFTER = 3
    TRANSIENT_MARKERS = (
        "429", "500", "502", "503", "504",
        "timeout", "timed out",
        "connection", "remote disconnected",
        "rate limit", "overloaded", "unavailable",
        # A cloud key-limit / billing / quota / out-of-credits error means the
        # CLOUD primary is exhausted for the run — fall through to the (usually
        # LOCAL) fallback judge, which has no such limit. Without these, a
        # `403 Key limit exceeded` was misclassified as a permanent
        # misconfiguration, the local fallback never ran, the judge scored ZERO
        # candidates, and clip selection collapsed to a couple of raw-signal
        # picks instead of a judged spread across the episode.
        "key limit", "quota", "insufficient", "billing",
        "out of credit", "credits", "402", "payment required",
    )

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self._primary_consecutive_failures = 0

    @classmethod
    def _is_transient(cls, err: str) -> bool:
        e = (err or "").lower()
        return any(m in e for m in cls.TRANSIENT_MARKERS)

    def judge(self, *args, **kwargs) -> dict:
        skip_primary = (
            self._primary_consecutive_failures >= self.STICKY_AFTER
        )

        if not skip_primary:
            result = self.primary.judge(*args, **kwargs)
            if 'error' not in result:
                self._primary_consecutive_failures = 0
                return result

            err = str(result.get('error', ''))
            if not self._is_transient(err):
                # Permanent — auth, model-not-found, etc. Don't fall
                # back; the user should fix the primary config.
                return result

            self._primary_consecutive_failures += 1
            logger.info(
                "Primary judge transient failure (%s) — trying fallback",
                err[:120])
        else:
            logger.debug(
                "Primary judge sticky-skipped (%d consecutive failures), "
                "using fallback directly",
                self._primary_consecutive_failures)

        fb = self.fallback.judge(*args, **kwargs)
        return fb


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
        seg_start = seg.get('start_sec', seg.get('start', 0))
        seg_end = seg.get('end_sec', seg.get('end', 0))
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

    # Today's live trend brief (daily-cached; warmed by the SEO stage / a prior
    # job today). A moment that rides a CURRENT trend/format is more shareable,
    # so let the judge factor it in. Empty (no nudge) when the cache is cold.
    trend = ""
    try:
        from backend.services.trend_brief import read_cached_brief
        _tb = read_cached_brief("both", "")
        if _tb:
            trend = ("\n\nTODAY'S SHORT-FORM TREND CONTEXT (favor moments that fit a "
                     "CURRENT trend/format, but never reward an off-topic match):\n"
                     + _tb.strip())
    except Exception:
        pass

    return f"""You are an expert viral content editor judging a potential short-form clip.{trend}

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


def _select_clip_pool(candidates, eff_max):
    """Build the candidate pool for the final diversity selection.

    Judge-approved clips (verdict != 'skip') come first. When the judge keeps
    fewer than ``eff_max``, backfill the pool with the best-scored *skipped*
    candidates so the diversity pass can still spread real moments across the
    whole video instead of returning a few bunched in one region. The judge
    blends its verdict into ``composite_score`` (keepers score higher), so the
    score-sorted selection downstream keeps the strongest clips on top — this
    only fills the remaining slots. If the judge skipped everything, fall back
    to all candidates. Pure (no video / FFmpeg) so it's unit-testable."""
    keepers = [c for c in candidates if getattr(c, "judge_verdict", "") != "skip"]
    if not keepers:
        return list(candidates)
    if len(keepers) < eff_max:
        _skipped = sorted(
            (c for c in candidates if getattr(c, "judge_verdict", "") == "skip"),
            key=lambda c: getattr(c, "composite_score", 0.0), reverse=True)
        keepers = keepers + _skipped[: max(0, eff_max * 3 - len(keepers))]
    return keepers


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
    # Work on a LOCAL score copy so we never mutate the candidates'
    # ``composite_score``. This function is called twice on the same objects
    # (pre-judge pool + final selection); persistently penalising the shared
    # score corrupted the second pass.
    remaining = [[c, float(getattr(c, "composite_score", 0.0) or 0.0)]
                 for c in candidates]

    for _ in range(max_clips):
        if not remaining:
            break

        # Pick the highest-scoring remaining candidate
        best = remaining[0][0]
        selected.append(best)
        best_center = (best.start_s + best.end_s) / 2

        # Penalize remaining candidates near the selected one (local score only)
        new_remaining = []
        for c, sc in remaining[1:]:
            c_center = (c.start_s + c.end_s) / 2
            dist = abs(c_center - best_center)

            # Temporal penalty: candidates closer than min_gap_s get penalized
            if dist < min_gap_s:
                # Stronger penalty for closer candidates
                penalty = 1.0 - (dist / min_gap_s) * 0.7  # up to 70% penalty
                sc = sc * (1.0 - penalty)

            # Also skip candidates that heavily overlap with the selected one
            overlap_start = max(c.start_s, best.start_s)
            overlap_end = min(c.end_s, best.end_s)
            overlap = max(0, overlap_end - overlap_start)
            shorter = min(c.duration_s, best.duration_s)
            if shorter > 0 and overlap / shorter > 0.8:
                continue  # skip >80% overlap

            new_remaining.append([c, sc])

        # Re-sort after score adjustments
        new_remaining.sort(key=lambda t: t[1], reverse=True)
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
        # Running estimate of the Replicate / cloud spend incurred by
        # this extraction. Read by the pipeline after run() completes
        # to populate the job's analysis-cost card.
        self.total_cost_usd: float = 0.0

    def run(self, on_progress: Callable = None,
            on_candidates: Callable = None) -> List[ClipCandidate]:
        """Execute the full clip extraction pipeline.

        ``on_candidates(final_candidates)`` fires once the clips are RANKED but
        BEFORE the (long, crash-prone) export loop, so the caller can persist the
        clip list immediately — a container death mid-export then keeps the clips
        instead of losing all of them.
        """
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

        # Priority 1: Replicate cloud GPU (no local VRAM needed)
        if self.config.replicate_enabled and self.config.replicate_api_key:
            # When VIDEOLLAMA3_ENHANCED is set, use the multi-pass V3
            # discovery (system prompt + adaptive chunks + audio
            # annotations + refinement + image keyframe analysis).
            # Otherwise preserve the classic single-pass behavior.
            try:
                from backend.config import settings as _app_settings
            except Exception:
                _app_settings = None
            use_v3_enhanced = bool(getattr(_app_settings, "VIDEOLLAMA3_ENHANCED", False))
            if use_v3_enhanced:
                rep = ReplicateDiscoveryV3(
                    api_key=self.config.replicate_api_key,
                    model_id=self.config.replicate_model,
                    fps=getattr(_app_settings, "VIDEOLLAMA3_FPS", 2),
                    max_frames=getattr(_app_settings, "VIDEOLLAMA3_MAX_FRAMES", 128),
                    refinement_enabled=getattr(_app_settings, "VIDEOLLAMA3_REFINEMENT_PASS", True),
                    keyframe_analysis=getattr(_app_settings, "VIDEOLLAMA3_KEYFRAME_ANALYSIS", True),
                    audio_annotation=getattr(_app_settings, "VIDEOLLAMA3_AUDIO_ANNOTATION", True),
                    adaptive_chunks=getattr(_app_settings, "VIDEOLLAMA3_ADAPTIVE_CHUNKS", True),
                    chunk_min_s=getattr(_app_settings, "VIDEOLLAMA3_CHUNK_MIN_S", 120),
                    chunk_max_s=getattr(_app_settings, "VIDEOLLAMA3_CHUNK_MAX_S", 900),
                    chunk_workers=getattr(self.config, "replicate_chunk_workers", 3),
                    rate_limit_retries=getattr(self.config, "replicate_rate_limit_retries", 2),
                    rate_limit_backoff_s=getattr(self.config, "replicate_rate_limit_backoff_s", 20.0),
                )
            else:
                rep = ReplicateDiscovery(
                    api_key=self.config.replicate_api_key,
                    model_id=self.config.replicate_model)
            if rep.is_available():
                try:
                    logger.info("Pass 2: Replicate cloud VideoLLaMA discovery...")
                    vlm_candidates = rep.discover_clips(
                        self.video_path,
                        self.transcript,
                        signal_timeline=signals,
                        chunk_duration_s=self.config.chunk_duration_s,
                        max_vlm_chunks=self.config.max_vlm_chunks,
                        preferred_subjects=self.config.preferred_subjects,
                        avoid_subjects=self.config.avoid_subjects,
                        platforms=self.config.platforms,
                        min_dur_s=self.config.min_duration_s,
                        max_dur_s=self.config.max_duration_s,
                        ideal_dur_s=self.config.ideal_duration_s,
                        discovery_prompt=self.config.discovery_prompt,
                        on_progress=lambda p: on_progress(
                            0.15 + p * 0.40) if on_progress else None
                    )
                    logger.info(f"Replicate discovered {len(vlm_candidates)} candidates")
                except Exception as e:
                    logger.warning(f"Replicate VLM failed: {e} — falling back to local")
                # Surface the running cost estimate (V3 + V2 both
                # populate ``total_cost_usd`` after a run, V2 stays at
                # 0.0 since it doesn't compute it).
                self.total_cost_usd = float(
                    getattr(rep, "total_cost_usd", 0.0) or 0.0
                )

        # Priority 2: Local VideoLLaMA2 (needs ≥10GB VRAM)
        if not vlm_candidates and self.config.videollama2_enabled:
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

        # Record whether VLM discovery actually contributed, so the pipeline /
        # UI can flag the degraded "signal-only" mode (VLM unavailable on this
        # GPU, or the cloud VLM was rate-limited).
        self.vlm_discovery_used = bool(vlm_candidates)

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
        judge_configured = (
            self.config.cloud_backend != "none"
            or bool(self.config.judge_primary)
        )
        if judge_configured and judge_candidates:
            judge = self._create_judge()
            if judge:
                judge_label = (
                    self.config.judge_primary
                    or f"legacy:{self.config.cloud_backend}"
                )
                if self.config.judge_fallback:
                    judge_label += f" → {self.config.judge_fallback}"
                logger.info(
                    f"Running cloud editorial judge ({judge_label}) on "
                    f"{len(judge_candidates)} candidates...")
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
        # Forward an optional per-clip message (m) through to the pipeline's
        # progress relay so the long export phase reports "Exporting clip k/N".
        self.clips = self._final_rank_and_export(
            judge_candidates,
            lambda p, m=None: on_progress(0.85 + p * 0.15, m) if on_progress else None,
            on_ranked=on_candidates,
        )

        if on_progress:
            on_progress(1.0)

        logger.info(f"Clip extraction complete: {len(self.clips)} clips exported")
        return self.clips

    def _create_judge(self):
        """Build the editorial judge from config.

        Priority order:
        1. New spec format (``judge_primary`` / ``judge_fallback``)
           using app-wide settings keys. If both are set, returns a
           ``FallbackJudge`` wrapper. If only primary is set, returns
           that judge directly.
        2. Legacy ``cloud_backend`` + clipper-private keys (the old
           pre-fallback config). Preserved so existing on-disk
           configs keep working without a migration.
        3. ``None`` — pipeline falls back to signal-only scoring.
        """
        primary = _build_judge_from_spec(self.config.judge_primary)

        if primary is None and self.config.cloud_backend != "none":
            primary = self._create_legacy_judge()

        if primary is None:
            return None

        fallback = _build_judge_from_spec(self.config.judge_fallback)
        if fallback is not None:
            logger.info(
                "Editorial judge using fallback chain: primary=%s fallback=%s",
                self.config.judge_primary or f"legacy:{self.config.cloud_backend}",
                self.config.judge_fallback)
            return FallbackJudge(primary, fallback)

        return primary

    def _create_legacy_judge(self):
        """Build a judge from the pre-fallback config fields."""
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
        """Send each candidate to the cloud judge for scoring.

        Safety nets:
        * Per-candidate exception wrap so one bad call can't kill the loop.
        * Circuit breaker: if total failures >= MAX_TOTAL_FAILURES, cancel
          remaining futures — remaining clips fall back to signal-only scores.
        * Total time budget: BUDGET_S — caps worst-case stall time.
        * Concurrency cap (MAX_JUDGE_WORKERS) acts as a soft rate-limiter;
          replaces the old sequential 0.5s sleep between calls.
        """
        total = len(candidates)
        MAX_JUDGE_WORKERS = 5
        MAX_TOTAL_FAILURES = 3
        BUDGET_S = 240
        loop_start = _time.monotonic()
        failure_lock = threading.Lock()
        failure_count = [0]
        done_count = [0]

        def _judge_one(args):
            idx, c = args
            with failure_lock:
                if failure_count[0] >= MAX_TOTAL_FAILURES:
                    return idx, c, None  # circuit breaker tripped
            if _time.monotonic() - loop_start > BUDGET_S:
                return idx, c, None  # time budget exhausted

            keyframes = _extract_keyframes_b64(
                self.video_path, c.start_s, c.end_s, n_frames=4)
            signal_summary = (
                f"signal_score={c.signal_score:.3f}, "
                f"source={c.source}, "
                f"duration={c.duration_s:.0f}s"
            )
            try:
                result = judge.judge(
                    c, c.transcript_slice, keyframes, signal_summary,
                    self.config.preferred_subjects, self.config.avoid_subjects)
            except Exception as e:
                logger.warning("Editorial judge raised on candidate %d: %s", idx, e)
                result = {"error": str(e)}

            with failure_lock:
                if 'error' in result:
                    failure_count[0] += 1
                else:
                    failure_count[0] = 0  # reset on success
            return idx, c, result

        with _cf.ThreadPoolExecutor(max_workers=MAX_JUDGE_WORKERS) as executor:
            futures = {
                executor.submit(_judge_one, (idx, c)): idx
                for idx, c in enumerate(candidates)
            }
            for fut in _cf.as_completed(futures):
                with failure_lock:
                    _done = done_count[0] = done_count[0] + 1
                    _failures = failure_count[0]
                if on_progress:
                    on_progress(_done / max(1, total))

                if _failures >= MAX_TOTAL_FAILURES:
                    logger.warning(
                        "Editorial judge: %d total failures — cancelling "
                        "remaining candidates, falling back to signal-only scores",
                        _failures)
                    for pending in futures:
                        pending.cancel()
                    break

                elapsed = _time.monotonic() - loop_start
                if elapsed > BUDGET_S:
                    logger.warning(
                        "Editorial judge time budget exceeded (%ds) after "
                        "%d/%d candidates — cancelling remainder",
                        int(elapsed), _done, total)
                    for pending in futures:
                        pending.cancel()
                    break

                try:
                    idx, c, result = fut.result()
                except Exception:
                    continue

                if result is None or 'error' in result:
                    continue

                c.judge_scores = {
                    k: result.get(k, 0)
                    for k in ['hook', 'payoff', 'retention',
                              'shareability', 'standalone']
                }
                c.judge_verdict = result.get('verdict', 'keep')
                c.judge_title = result.get('title', '')

                if c.judge_scores:
                    judge_avg = sum(
                        v for v in c.judge_scores.values()
                        if isinstance(v, (int, float))
                    ) / max(1, len(c.judge_scores))
                    c.composite_score = (
                        0.4 * c.composite_score +
                        0.6 * (judge_avg / 10.0)
                    )

                trim = result.get('trim_suggestion', '')
                if trim and c.judge_verdict == 'trim':
                    try:
                        parts = trim.split('-')
                        if len(parts) == 2:
                            new_start = _parse_mmss(parts[0])
                            new_end = _parse_mmss(parts[1])
                            # A trim must SHRINK the clip, not RELOCATE it. Reject a
                            # suggestion that doesn't substantially overlap the
                            # original candidate — otherwise a model that returns the
                            # same early "highlight" window for many candidates
                            # collapses them ALL onto the opening (the clip-clustering
                            # bug: every clip ends up in the first ~2 minutes).
                            if new_end > new_start:
                                _ov = max(0.0, min(new_end, c.end_s) - max(new_start, c.start_s))
                                if _ov >= 0.5 * (new_end - new_start):
                                    c.start_s = new_start
                                    c.end_s = new_end
                                    c.duration_s = new_end - new_start
                                else:
                                    logger.debug(
                                        "Judge trim %s ignored — relocates clip away "
                                        "from its %.0f-%.0fs window",
                                        trim, c.start_s, c.end_s)
                    except Exception:
                        pass

        if on_progress:
            on_progress(1.0)

    def _final_rank_and_export(self, candidates: List[ClipCandidate],
                               on_progress: Callable = None,
                               on_ranked: Callable = None) -> List[ClipCandidate]:
        """Final ranking and FFmpeg clip extraction."""
        video_dur = _get_video_duration(self.video_path)
        eff_max = self.config.effective_max_clips(video_dur)

        # Judge-approved clips first, backfilled toward the target with the
        # best-scored skipped moments when the judge kept too few (it's often
        # over-strict on visual/action beats), so we get MORE clips spread
        # across the episode rather than a handful bunched together.
        keepers = _select_clip_pool(candidates, eff_max)

        # Final diversity selection — spread clips across timeline
        keepers.sort(key=lambda c: c.composite_score, reverse=True)
        min_gap = max(20, video_dur / (eff_max * 2))
        final = select_diverse_clips(keepers, eff_max, video_dur, min_gap_s=min_gap)

        # Diagnostics: why this many clips, and whether they spread across the
        # episode or bunch up. Shows the judge verdict mix, the keeper pool's
        # time-span vs the video, and the chosen clip start times — so a
        # "few clips clustered in the opening" run is explainable at a glance.
        try:
            _vc = {}
            for c in candidates:
                _vc[getattr(c, "judge_verdict", "") or "(none)"] = \
                    _vc.get(getattr(c, "judge_verdict", "") or "(none)", 0) + 1
            _kc = [(c.start_s + c.end_s) / 2 for c in keepers]
            _span = (max(_kc) - min(_kc)) if _kc else 0
            logger.info(
                "Final clip selection: video=%.0fs eff_max=%d · candidates=%d "
                "verdicts=%s · keeper pool=%d spanning %.0f-%.0fs (%.0f%% of video) "
                "· selected=%d at %s",
                video_dur, eff_max, len(candidates), _vc, len(keepers),
                (min(_kc) if _kc else 0), (max(_kc) if _kc else 0),
                (100.0 * _span / video_dur) if video_dur else 0,
                len(final), [f"{int(c.start_s)}s" for c in final])
        except Exception:
            pass

        # Sort final clips by time for sequential output
        final.sort(key=lambda c: c.start_s)

        if not final:
            return []

        # Hand the RANKED clips to the caller before the export loop starts. The
        # export tail is the longest, most crash-prone phase (N FFmpeg encodes);
        # persisting the list here means a container death mid-export keeps the
        # clips (the user sees them + can re-export) instead of losing all N.
        if on_ranked:
            try:
                on_ranked(list(final))
            except Exception:
                pass

        # Create output directory next to the video
        out_dir = os.path.join(os.path.dirname(self.video_path), "clips")
        os.makedirs(out_dir, exist_ok=True)

        # Export each clip (GPU-encoded, several at a time). Report a per-clip
        # MESSAGE (not just a fraction): exporting N clips is the longest tail of
        # the run, and a single static "Exporting top clips…" let the UI's
        # stuck-timer trip the "pipeline may be stuck" banner. A changing
        # "Exporting clip k/N …" keeps the activity log + stuck-timer alive.
        _n_final = len(final)

        def _clip_name_for(idx: int, c) -> str:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            slug = ""
            if c.judge_title:
                slug = re.sub(r'[^\w\s-]', '', c.judge_title)
                slug = re.sub(r'[\s]+', '_', slug).strip('_')[:40]
            elif c.vlm_hook:
                slug = re.sub(r'[^\w\s-]', '', c.vlm_hook[:30])
                slug = re.sub(r'[\s]+', '_', slug).strip('_')
            return (f"{base_name}_clip{idx+1}_{slug}.mp4" if slug
                    else f"{base_name}_clip{idx+1}.mp4")

        # (idx, clip, clip_name, clip_path)
        tasks = [(idx, c, _clip_name_for(idx, c)) for idx, c in enumerate(final)]
        tasks = [(idx, c, cn, os.path.join(out_dir, cn)) for idx, c, cn in tasks]

        try:
            from backend.config import settings as _cs
            _conc = max(1, int(getattr(_cs, "CLIP_EXPORT_CONCURRENCY", 2) or 1))
        except Exception:
            _conc = 2
        _conc = max(1, min(_conc, _n_final))

        _ok: list[tuple[int, object]] = []   # (idx, clip) for successful exports

        def _log_ok(idx, c, cn):
            logger.info(
                f"Exported clip {idx+1}/{_n_final}: "
                f"{_fmt_time(c.start_s)}–{_fmt_time(c.end_s)} "
                f"({c.duration_s:.0f}s) → {cn}")

        if _conc <= 1:
            for idx, c, cn, cp in tasks:
                if on_progress:
                    on_progress(idx / max(1, _n_final),
                                f"Exporting clip {idx + 1}/{_n_final}…")
                if _export_clip(self.video_path, cp, c.start_s, c.end_s):
                    _ok.append((idx, c))
                    _log_ok(idx, c, cn)
        else:
            # Export several clips at once — they're independent ffmpeg jobs and
            # the small 9:16 frames are cheap, so the GPU/CPU sit idle between
            # sequential encodes. as_completed yields on THIS thread, so
            # on_progress is still called serially (no interleaving).
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _done = 0
            logger.info("Clip export: %d clips, %d at a time (GPU encode)",
                        _n_final, _conc)
            with ThreadPoolExecutor(max_workers=_conc) as _ex:
                _futs = {
                    _ex.submit(_export_clip, self.video_path, cp, c.start_s, c.end_s):
                        (idx, c, cn)
                    for idx, c, cn, cp in tasks
                }
                for _fut in as_completed(_futs):
                    idx, c, cn = _futs[_fut]
                    _done += 1
                    if on_progress:
                        on_progress(_done / max(1, _n_final),
                                    f"Exporting clips… ({_done}/{_n_final})")
                    try:
                        if _fut.result():
                            _ok.append((idx, c))
                            _log_ok(idx, c, cn)
                    except Exception as _ee:
                        logger.warning("clip %d export errored: %s", idx + 1, _ee)

        # Restore chronological order (parallel completion is out of order).
        _ok.sort(key=lambda t: t[0])
        exported = [c for _i, c in _ok]

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
    """Extract a candidate clip — a plain time-cut of the source video (no
    reframing/subtitles are burned here; those are applied later on export).

    Clip export was the longest tail of the pipeline: the old
    ``libx264 -preset fast -crf 18`` re-encoded every clip on the CPU
    (~45-135 s each × dozens of clips). Speedups, fastest first:

      * ``CLIP_EXPORT_STREAM_COPY`` (opt-in): no re-encode at all — remux the
        bytes (``-c copy``). Near-instant, but the cut snaps to the nearest
        keyframe so a clip can begin a second or two early.
      * GPU encoder (NVENC/VAAPI/QSV, per the GPU toggle): frame-accurate and
        ~5-10× faster than CPU libx264. This is the default.
      * Fast CPU encode (``libx264 -preset veryfast``): the fallback.

    ``-ss``/``-to`` stay BEFORE ``-i`` for fast keyframe seek, so only the clip
    span is decoded. Each option falls through to the next if it fails."""
    try:
        from backend.config import settings
    except Exception:
        settings = None
    _crf = int(getattr(settings, "CLIP_EXPORT_CRF", 21) if settings else 21)
    _stream_copy = bool(getattr(settings, "CLIP_EXPORT_STREAM_COPY", False) if settings else False)
    base = ['ffmpeg', '-y', '-ss', f"{start_s:.2f}", '-to', f"{end_s:.2f}",
            '-i', video_path]
    tail = ['-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart', output_path]

    cmds: list[list] = []
    # Fastest (opt-in): stream-copy — no transcode, keyframe-approximate start.
    if _stream_copy:
        cmds.append(base + ['-c', 'copy', '-movflags', '+faststart', output_path])
    # Preferred: GPU encoder (respects GPU_ACCELERATION_ENABLED; returns libx264
    # itself when the toggle is off or no GPU encoder is present).
    try:
        from backend.services.clip_exporter import _gpu_encode_args
        # preset only affects the libx264 fallback inside the helper (NVENC uses
        # its own p5); "veryfast" keeps the CPU path quick when the GPU is off.
        _vargs = _gpu_encode_args({"preset": "veryfast", "crf": _crf}, "1080p")
        if _vargs:
            cmds.append(base + _vargs + tail)
    except Exception:
        pass
    # Fallback (and the path when GPU detection is unavailable): fast CPU encode.
    # veryfast + crf 21 is far quicker than the old fast + crf 18 and plenty for
    # review clips.
    _cpu = base + ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(_crf),
                   '-pix_fmt', 'yuv420p'] + tail
    if not cmds or cmds[-1] != _cpu:
        cmds.append(_cpu)

    for attempt, cmd in enumerate(cmds):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            if result.returncode == 0 and os.path.exists(output_path) \
                    and os.path.getsize(output_path) > 0:
                return True
            logger.warning(
                "clip export attempt %d/%d failed (rc=%s): %s",
                attempt + 1, len(cmds), result.returncode,
                (result.stderr or b"")[-300:].decode("utf-8", "replace"))
        except Exception as e:
            logger.warning("clip export attempt %d/%d error: %s",
                           attempt + 1, len(cmds), e)
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

