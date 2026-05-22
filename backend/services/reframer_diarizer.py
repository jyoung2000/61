"""ClipAI Reframer — Speaker Diarizer.

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

logger = logging.getLogger("clipai.reframer_diarizer")


class SpeakerDiarizer:
    """
    Audio-based speaker diarization using pyannote.audio.

    Extracts the audio track from the video, runs speaker segmentation,
    and produces a timeline of who is speaking when (SPEAKER_00, SPEAKER_01, etc.)

    Requires: pip install pyannote.audio torch torchaudio
    Falls back gracefully if unavailable — the reframer still works,
    just without audio-driven speaker identification.
    """

    def __init__(self):
        self.available = False
        self.pipeline = None

    def try_load(self) -> bool:
        """Attempt to load the diarization pipeline."""
        log = get_logger()
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
            import torch

            # Use the pretrained pipeline (requires accepting HF terms)
            # Falls back to a simpler approach if HF token isn't set
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
            if hf_token:
                self.pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token)
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                self.available = True
                log.log_stage('PERCEIVE', 'Speaker diarization: pyannote 3.1 loaded (GPU)'
                              if torch.cuda.is_available()
                              else 'Speaker diarization: pyannote 3.1 loaded (CPU)')
            else:
                log.log_stage('PERCEIVE',
                    'Speaker diarization: HF_TOKEN not set. '
                    'Set HF_TOKEN env var with a Hugging Face token that has '
                    'accepted pyannote/speaker-diarization-3.1 terms. '
                    'Falling back to mouth-motion heuristic.')
                return False

        except ImportError:
            log.log_stage('PERCEIVE',
                'Speaker diarization unavailable (pip install pyannote.audio). '
                'Using mouth-motion heuristic instead.')
        except Exception as e:
            log.log_stage('PERCEIVE', f'Speaker diarization failed to load: {str(e)[:150]}')
        return self.available

    def diarize(self, video_path: str, duration_ms: int) -> Dict[int, str]:
        """Run diarization on the video's audio track.
        Returns {time_ms: speaker_id} at 200ms resolution."""
        if not self.available:
            return {}

        log = get_logger()
        log.start_timer('diarize')

        try:
            # Extract audio to temp WAV
            import tempfile
            audio_path = tempfile.mktemp(suffix='.wav')
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                audio_path
            ], capture_output=True, timeout=120)

            if not os.path.exists(audio_path):
                log.log_error('PERCEIVE', 'Audio extraction failed')
                return {}

            # Run diarization
            diarization = self.pipeline(audio_path)

            # Convert to timeline at 200ms resolution
            timeline = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                for t in range(start_ms, end_ms, 200):
                    timeline[t] = speaker

            # Cleanup
            try:
                os.remove(audio_path)
            except Exception:
                pass

            elapsed = log.stop_timer('diarize')
            n_speakers = len(set(timeline.values()))
            log.log_stage('PERCEIVE', f'Diarization complete: {n_speakers} speakers found',
                           elapsed_sec=round(elapsed, 2),
                           segments=len(timeline))
            return timeline

        except Exception as e:
            log.log_error('PERCEIVE', f'Diarization failed: {e}')
            return {}


