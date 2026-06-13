"""Orchestrate MP4/MP3/WAV → analysis for the import wizard.

Glue over :mod:`extract`, :mod:`isolate`, :mod:`vad`. Runs on a worker thread
(it shells out to ffmpeg and may load Demucs), reporting stage progress. Kept
separate from the dialog so the heavy logic is reusable and the UI stays thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Settings
from . import vad
from .extract import extract_reference_pair, probe_duration
from .isolate import isolate_vocals, should_suggest_isolation, spectral_flatness
from .vad import ClipCandidate


@dataclass
class IngestAnalysis:
    wav24_path: Path
    wav44_path: Path
    sample_rate: int
    duration_s: float
    speech_ratio: float
    suggest_isolation: bool
    candidates: list[ClipCandidate]
    audio24: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.float32))
    audio44: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, np.float32))
    sample_rate_44: int = 44_100
    isolated: bool = False


def analyze_source(
    input_path: str | Path,
    work_dir: str | Path,
    settings: Settings,
    isolate: bool = False,
    progress=None,
) -> IngestAnalysis:
    """Extract, optionally isolate vocals, run VAD and rank candidate clips."""
    import soundfile as sf

    def report(msg: str, frac: float = -1.0) -> None:
        if progress:
            progress(msg, frac)

    report("Extracting audio…")
    wav24, wav44 = extract_reference_pair(input_path, work_dir, settings.ffmpeg_path)
    audio24, sr = sf.read(str(wav24), dtype="float32", always_2d=False)
    audio24 = np.asarray(audio24, dtype=np.float32).reshape(-1)

    isolated = False
    if isolate:
        from ..models.model_manager import resolve_device

        report("Isolating vocals with Demucs…")
        device = resolve_device(settings.use_cuda)
        audio24, sr = isolate_vocals(audio24, sr, device=device)
        sf.write(str(wav24), audio24, sr)
        isolated = True

    report("Detecting speech and scoring clips…")
    analysis = vad.analyze(audio24, sr, min_s=10.0, max_s=20.0, n=3)
    flat = spectral_flatness(audio24)
    suggest = should_suggest_isolation(
        analysis.speech_ratio, flat, speech_threshold=settings.auto_isolate_threshold
    )

    audio44, sr44 = sf.read(str(wav44), dtype="float32", always_2d=False)
    audio44 = np.asarray(audio44, dtype=np.float32).reshape(-1)

    duration = probe_duration(input_path, settings.ffprobe_path) or (
        audio24.size / sr if sr else 0.0
    )

    return IngestAnalysis(
        wav24_path=wav24,
        wav44_path=wav44,
        sample_rate=int(sr),
        duration_s=float(duration),
        speech_ratio=float(analysis.speech_ratio),
        suggest_isolation=bool(suggest) and not isolated,
        candidates=list(analysis.candidates),
        audio24=audio24,
        audio44=audio44,
        sample_rate_44=int(sr44),
        isolated=isolated,
    )


def slice_seconds(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Return ``audio[start_s:end_s]`` (clamped) as float32."""
    a = int(max(0.0, start_s) * sr)
    b = int(min(end_s, audio.size / sr if sr else 0.0) * sr)
    if b <= a:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(audio[a:b], dtype=np.float32)
