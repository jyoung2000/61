"""Voice-activity detection + best-clip selection (silero-vad).

The ML part (silero) is a thin wrapper; the *scoring* that ranks candidate
windows is pure numpy so it is unit-testable without the model. A good cloning
clip is continuous speech, with consistent level and no clipping -- exactly the
three factors :func:`score_window` multiplies together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Sample rate silero-vad expects. We resample for detection but score on the
# original-rate audio.
VAD_SR = 16_000


@dataclass
class SpeechRegion:
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class ClipCandidate:
    start_s: float
    end_s: float
    score: float
    speech_ratio: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class SpeechAnalysis:
    total_duration_s: float
    speech_ratio: float  # speech seconds / total seconds
    regions: list[SpeechRegion]
    candidates: list[ClipCandidate]


# --------------------------------------------------------------------------- #
# Pure scoring (unit-tested)
# --------------------------------------------------------------------------- #
def _frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if audio.size < frame:
        return np.array([float(np.sqrt(np.mean(audio**2) + 1e-12))], dtype=np.float32)
    n = 1 + (audio.size - frame) // hop
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = audio[i * hop : i * hop + frame]
        out[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return out


def rms_consistency(audio: np.ndarray, sr: int) -> float:
    """1.0 for a perfectly steady level, → 0 as the level swings around.

    Computed as ``1 / (1 + coefficient_of_variation)`` over per-frame RMS of the
    audible frames only (near-silent frames are excluded so a pause inside the
    window does not look like an inconsistency on its own).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    frame = max(256, int(0.04 * sr))  # ~40 ms
    hop = max(128, frame // 2)
    rms = _frame_rms(audio, frame, hop)
    audible = rms[rms > rms.max() * 0.1] if rms.max() > 0 else rms
    if audible.size == 0 or audible.mean() <= 0:
        return 0.0
    cv = float(audible.std() / (audible.mean() + 1e-9))
    return float(1.0 / (1.0 + cv))


def clipping_fraction(audio: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples at/above the clipping threshold."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


def score_window(audio_window: np.ndarray, sr: int, speech_fraction: float) -> float:
    """Combine speech continuity × level consistency × (1 − clipping)."""
    speech_fraction = float(max(0.0, min(1.0, speech_fraction)))
    consistency = rms_consistency(audio_window, sr)
    clip_pen = 1.0 - clipping_fraction(audio_window)
    return float(speech_fraction * consistency * clip_pen)


def _coverage_fraction(
    regions: list[SpeechRegion], start_s: float, end_s: float
) -> float:
    """Fraction of ``[start_s, end_s)`` covered by speech regions."""
    span = end_s - start_s
    if span <= 0:
        return 0.0
    covered = 0.0
    for r in regions:
        lo = max(start_s, r.start_s)
        hi = min(end_s, r.end_s)
        if hi > lo:
            covered += hi - lo
    return min(1.0, covered / span)


def pick_candidates(
    regions: list[SpeechRegion],
    audio: np.ndarray,
    sr: int,
    *,
    min_s: float = 10.0,
    max_s: float = 20.0,
    n: int = 3,
    hop_s: float = 1.0,
) -> list[ClipCandidate]:
    """Rank candidate clips and return the top ``n`` non-overlapping windows.

    Tries a few window lengths between ``min_s`` and ``max_s`` (longer clips give
    the cloner more material), slides each across the file, scores every position
    and greedily selects the best non-overlapping windows.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    total_s = audio.size / sr if sr else 0.0
    if total_s <= 0 or not regions:
        return []

    lengths = sorted({min_s, (min_s + max_s) / 2.0, max_s})
    lengths = [length for length in lengths if length <= total_s] or [min(total_s, max_s)]

    scored: list[ClipCandidate] = []
    for win_s in lengths:
        start = 0.0
        while start + win_s <= total_s + 1e-6:
            end = start + win_s
            cov = _coverage_fraction(regions, start, end)
            if cov > 0.0:
                seg = audio[int(start * sr) : int(end * sr)]
                scored.append(
                    ClipCandidate(start, end, score_window(seg, sr, cov), cov)
                )
            start += hop_s

    scored.sort(key=lambda c: c.score, reverse=True)
    chosen: list[ClipCandidate] = []
    for cand in scored:
        if all(_no_overlap(cand, c) for c in chosen):
            chosen.append(cand)
        if len(chosen) >= n:
            break
    return chosen


def _no_overlap(a: ClipCandidate, b: ClipCandidate) -> bool:
    return a.end_s <= b.start_s or b.end_s <= a.start_s


def speech_ratio(regions: list[SpeechRegion], total_s: float) -> float:
    if total_s <= 0:
        return 0.0
    return min(1.0, sum(r.duration for r in regions) / total_s)


# --------------------------------------------------------------------------- #
# silero-vad wrapper (not unit-tested -- requires the model)
# --------------------------------------------------------------------------- #
_vad_model = None


def _load_vad():  # pragma: no cover - requires torch + model download
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad

            _vad_model = load_silero_vad()
        except Exception as exc:  # fall back to torch.hub for older installs
            try:
                import torch

                _vad_model, _ = torch.hub.load(
                    "snakers4/silero-vad", "silero_vad", trust_repo=True
                )
            except Exception:
                raise RuntimeError(
                    "silero-vad is not available. Install it with "
                    "`pip install silero-vad`."
                ) from exc
    return _vad_model


def detect_speech_regions(
    audio: np.ndarray, sr: int
) -> list[SpeechRegion]:  # pragma: no cover - requires model
    """Run silero-vad on ``audio`` (any rate) and return speech regions in sec."""
    import torch

    from ..synth.assemble import resample

    model = _load_vad()
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    wav16 = resample(mono, sr, VAD_SR)
    tensor = torch.from_numpy(wav16)
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(
        tensor, model, sampling_rate=VAD_SR, return_seconds=True
    )
    return [SpeechRegion(float(s["start"]), float(s["end"])) for s in stamps]


def analyze(
    audio: np.ndarray, sr: int, *, min_s: float = 10.0, max_s: float = 20.0, n: int = 3
) -> SpeechAnalysis:  # pragma: no cover - requires model
    """Full analysis: detect speech, compute ratio, rank candidate clips."""
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    total_s = mono.size / sr if sr else 0.0
    regions = detect_speech_regions(mono, sr)
    return SpeechAnalysis(
        total_duration_s=total_s,
        speech_ratio=speech_ratio(regions, total_s),
        regions=regions,
        candidates=pick_candidates(
            regions, mono, sr, min_s=min_s, max_s=max_s, n=n
        ),
    )
