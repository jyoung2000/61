"""Assemble per-segment wavs into one seamless, normalized track.

Pure numpy/scipy/pyloudnorm so it is fully unit-testable without a GPU. Steps
(spec §8):

* resample every segment to a common rate,
* concatenate with 15 ms **equal-power** crossfades,
* insert explicit ``pause_after_ms`` silences (a pause suppresses the crossfade
  at that join -- the gap is intentional),
* loudness-normalize the final mix to -16 LUFS and true-peak limit to -1 dBTP.

Equal-power (rather than linear) crossfades keep perceived loudness steady
across the join between two *uncorrelated* TTS segments.
"""

from __future__ import annotations

import numpy as np

try:  # pyloudnorm is a core dep, but keep assembly importable if it is missing.
    import pyloudnorm as _pyln
except Exception:  # pragma: no cover - exercised only on broken installs
    _pyln = None


def silence(duration_ms: float, sample_rate: int) -> np.ndarray:
    """Return ``duration_ms`` of digital silence as float32."""
    n = int(round(max(0.0, duration_ms) * sample_rate / 1000.0))
    return np.zeros(n, dtype=np.float32)


def equal_power_crossfade(a: np.ndarray, b: np.ndarray, n_samples: int) -> np.ndarray:
    """Join ``a`` and ``b`` overlapping the last/first ``n_samples`` equal-power.

    The fade uses quarter-cosine curves so ``fade_out**2 + fade_in**2 == 1``
    everywhere, preserving power across the overlap. ``n_samples`` is clamped to
    the available length of either side; if it collapses to zero the two arrays
    are simply concatenated.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = int(min(n_samples, len(a), len(b)))
    if n <= 0:
        return np.concatenate([a, b])
    # Sample the curves at bin centers so the overlap is symmetric.
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    fade_out = np.cos(t * (np.pi / 2.0))
    fade_in = np.sin(t * (np.pi / 2.0))
    overlap = a[-n:] * fade_out + b[:n] * fade_in
    return np.concatenate([a[:-n], overlap.astype(np.float32), b[n:]])


def assemble_segments(
    segments: list[np.ndarray],
    pauses_ms: list[int] | None,
    sample_rate: int,
    crossfade_ms: float = 15.0,
) -> np.ndarray:
    """Concatenate ``segments`` with crossfades + explicit pauses.

    ``pauses_ms[i]`` is the silence inserted *after* segment ``i``. A non-zero
    pause replaces the crossfade at that join (you asked for a gap, so we give a
    clean gap rather than blending across it). Empty inputs yield empty audio.
    """
    segs = [np.asarray(s, dtype=np.float32).reshape(-1) for s in segments if len(s)]
    if not segs:
        return np.zeros(0, dtype=np.float32)
    if pauses_ms is None:
        pauses_ms = [0] * len(segments)

    crossfade_n = int(round(crossfade_ms * sample_rate / 1000.0))
    out = segs[0]
    seg_idx = 0  # index into the original `segments` for pause lookup
    # Map filtered segs back to original indices for correct pause association.
    orig_indices = [i for i, s in enumerate(segments) if len(s)]

    for k in range(1, len(segs)):
        prev_orig = orig_indices[k - 1]
        pause = pauses_ms[prev_orig] if prev_orig < len(pauses_ms) else 0
        if pause > 0:
            out = np.concatenate([out, silence(pause, sample_rate), segs[k]])
        else:
            out = equal_power_crossfade(out, segs[k], crossfade_n)
        seg_idx = k

    # Trailing pause after the final segment, if the user set one.
    last_orig = orig_indices[-1]
    if last_orig < len(pauses_ms) and pauses_ms[last_orig] > 0:
        out = np.concatenate([out, silence(pauses_ms[last_orig], sample_rate)])

    return out.astype(np.float32)


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target_lufs: float = -16.0
) -> np.ndarray:
    """Integrated-loudness normalize to ``target_lufs`` (ITU-R BS.1770).

    Falls back to a no-op for silence / very short clips where integrated
    loudness is undefined (``-inf``), and degrades to peak normalization if
    pyloudnorm is unavailable.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    if _pyln is None:  # pragma: no cover
        peak = float(np.max(np.abs(audio)))
        return audio if peak == 0 else (audio * (0.5 / peak)).astype(np.float32)

    meter = _pyln.Meter(sample_rate)
    try:
        loudness = meter.integrated_loudness(audio.astype(np.float64))
    except Exception:  # pragma: no cover - short/degenerate input
        return audio
    if not np.isfinite(loudness):
        return audio
    out = _pyln.normalize.loudness(audio.astype(np.float64), loudness, target_lufs)
    return out.astype(np.float32)


def true_peak_limit(audio: np.ndarray, dbtp: float = -1.0) -> np.ndarray:
    """Scale ``audio`` down so its sample peak does not exceed ``dbtp``.

    A simple peak clamp (oversampled true-peak is overkill here and the spec
    accepts this). Never boosts -- only attenuates an over-hot signal.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    limit = float(10.0 ** (dbtp / 20.0))
    peak = float(np.max(np.abs(audio)))
    if peak > limit and peak > 0:
        audio = audio * (limit / peak)
    return audio.astype(np.float32)


def master(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -16.0,
    true_peak_dbtp: float = -1.0,
) -> np.ndarray:
    """Loudness-normalize then true-peak limit the final mix."""
    audio = normalize_loudness(audio, sample_rate, target_lufs)
    return true_peak_limit(audio, true_peak_dbtp)


def resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample mono audio from ``src_sr`` to ``dst_sr``.

    Prefers ``scipy.signal.resample_poly`` (good quality, no extra deps); if
    scipy is unavailable falls back to linear interpolation.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_sr == dst_sr or audio.size == 0:
        return audio
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(src_sr), int(dst_sr))
        up = int(dst_sr) // g
        down = int(src_sr) // g
        return resample_poly(audio, up, down).astype(np.float32)
    except Exception:  # pragma: no cover - scipy missing
        n_out = int(round(audio.size * dst_sr / src_sr))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def ensure_rate(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Convenience wrapper used by the pipeline to unify segment rates."""
    return resample(audio, src_sr, target_sr) if src_sr != target_sr else audio
