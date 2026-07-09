"""Speech-coverage audit + voice-gated gap recovery.

"Transcribe the entire video" cannot mean "emit a cue for every second" — most
of a long video is music / silence / non-verbal sound, and forcing Whisper to
decode those regions only INVENTS text (the "Wow! Thank you!" hallucinations
over a moan or a musical outro). The achievable, correct goal is: **miss none
of the actual SPEECH, and leave genuine silence blank.**

To do that we need a source of truth for where speech actually is that is
INDEPENDENT of the Whisper decode — a voice-activity map. Silero VAD (bundled
with faster-whisper) gives exactly that from the raw waveform. We then:

  * measure coverage: how much of the voice-active audio the transcript covers,
    so the pipeline can honestly report "97% of speech covered; the rest is
    music/silence" instead of leaving the user to guess whether a blank stretch
    was missed or simply had no words; and
  * find the voice-active spans the transcript missed (NOT the silent ones) so
    the caller can re-transcribe only those — recovering soft / breathy / off-mic
    dialogue without flooding silence with hallucinations.

The interval math is pure and unit-tested. The VAD call is guarded: if
faster-whisper's VAD isn't importable the map is empty and the caller simply
keeps its existing behavior.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

Interval = Tuple[float, float]


# ── Pure interval helpers (no audio, unit-testable) ────────────────────────

def merge_intervals(intervals: List[Interval]) -> List[Interval]:
    """Sort and merge overlapping/touching intervals. Drops empty/negative."""
    cleaned = sorted(
        (float(a), float(b)) for a, b in intervals
        if b is not None and a is not None and float(b) > float(a))
    if not cleaned:
        return []
    out: List[List[float]] = [list(cleaned[0])]
    for a, b in cleaned[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _subtract(region: Interval, covered_sorted: List[Interval]) -> List[Interval]:
    """Parts of ``region`` not overlapped by the (merged, sorted) covered list."""
    a, b = region
    parts: List[Interval] = []
    cur = a
    for cs, ce in covered_sorted:
        if ce <= cur:
            continue
        if cs >= b:
            break
        if cs > cur:
            parts.append((cur, min(cs, b)))
        cur = max(cur, ce)
        if cur >= b:
            break
    if cur < b:
        parts.append((cur, b))
    return parts


def uncovered_voice_gaps(voice_regions: List[Interval],
                         covered_regions: List[Interval],
                         min_gap_s: float = 1.5) -> List[Interval]:
    """Voice-active spans NOT overlapped by any covered (transcribed) span,
    keeping only gaps at least ``min_gap_s`` long (short gaps between words are
    normal and re-transcribing them just re-emits the same cue)."""
    voice = merge_intervals(voice_regions)
    covered = merge_intervals(covered_regions)
    gaps: List[Interval] = []
    for v in voice:
        for g in _subtract(v, covered):
            if (g[1] - g[0]) >= min_gap_s:
                gaps.append(g)
    return gaps


def coverage_stats(voice_regions: List[Interval],
                   covered_regions: List[Interval]) -> Dict[str, float]:
    """Coverage of the voice-active audio by the transcript.

    Returns voice_sec (total speech time per VAD), covered_voice_sec,
    uncovered_voice_sec, coverage_ratio (0..1), and gap_count (uncovered
    voice runs ≥ 1.5s). A ratio near 1.0 means the transcript captured
    essentially all the speech; the untranscribed remainder of the runtime is
    genuine silence/music.
    """
    voice = merge_intervals(voice_regions)
    covered = merge_intervals(covered_regions)
    voice_sec = sum(b - a for a, b in voice)
    uncovered = 0.0
    for v in voice:
        for g in _subtract(v, covered):
            uncovered += (g[1] - g[0])
    covered_voice = max(0.0, voice_sec - uncovered)
    ratio = (covered_voice / voice_sec) if voice_sec > 0 else 1.0
    gap_count = len(uncovered_voice_gaps(voice_regions, covered_regions, 1.5))
    return {
        "voice_sec": voice_sec,
        "covered_voice_sec": covered_voice,
        "uncovered_voice_sec": uncovered,
        "coverage_ratio": ratio,
        "gap_count": float(gap_count),
    }


# ── Voice-activity map (Silero VAD via faster-whisper) ─────────────────────

def voice_activity_regions(audio_path: str,
                           threshold: float = 0.35,
                           min_speech_ms: int = 200,
                           min_silence_ms: int = 300,
                           speech_pad_ms: int = 200) -> List[Interval]:
    """Speech regions ``[(start_s, end_s), …]`` from a mono WAV, via the Silero
    VAD bundled with faster-whisper. Independent of any Whisper decode, so it
    catches soft/breathy speech a transcription pass dropped. A slightly
    relaxed default ``threshold`` (0.35 vs Silero's 0.5) favours recall — we
    want to FIND speech to check it was transcribed, not to gate decoding.

    Returns ``[]`` on any failure (VAD unavailable, decode error) so the caller
    keeps its existing behavior.
    """
    try:
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import get_speech_timestamps, VadOptions
    except Exception as e:
        logger.info("Voice-activity map unavailable (%s)", e)
        return []
    try:
        audio = decode_audio(audio_path, sampling_rate=16000)
    except Exception as e:
        logger.warning("Voice-activity: could not decode %s (%s)", audio_path, e)
        return []
    # VadOptions fields have shifted across faster-whisper versions — build it
    # defensively and fall back to defaults rather than lose the map.
    try:
        opts = VadOptions(threshold=threshold,
                          min_speech_duration_ms=min_speech_ms,
                          min_silence_duration_ms=min_silence_ms,
                          speech_pad_ms=speech_pad_ms)
    except Exception:
        try:
            opts = VadOptions()
        except Exception:
            opts = None
    try:
        ts = (get_speech_timestamps(audio, opts, sampling_rate=16000)
              if opts is not None
              else get_speech_timestamps(audio, sampling_rate=16000))
    except TypeError:
        # Older signature without sampling_rate kw / positional options.
        try:
            ts = get_speech_timestamps(audio)
        except Exception as e:
            logger.warning("Voice-activity detection failed (%s)", e)
            return []
    except Exception as e:
        logger.warning("Voice-activity detection failed (%s)", e)
        return []
    regions: List[Interval] = []
    for t in ts or []:
        try:
            regions.append((float(t["start"]) / 16000.0,
                            float(t["end"]) / 16000.0))
        except (KeyError, TypeError, ValueError):
            continue
    return merge_intervals(regions)
