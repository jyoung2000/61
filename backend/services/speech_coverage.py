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


def overlaps_voice(voice_regions: List[Interval],
                   start_s: float, end_s: float,
                   min_overlap_s: float = 0.2,
                   min_overlap_frac: float = 0.3) -> bool:
    """True when the span ``[start_s, end_s]`` overlaps the voice-activity map
    by at least ``max(min_overlap_s, min_overlap_frac × span)`` seconds.

    Used by the TACT phantom filter to distinguish an invented cue over
    silence/music (drop it) from a low-confidence decode of REAL speech
    (Silero heard voice there — redecode it instead of dropping). The
    absolute floor keeps a trivial brush against a voice region from
    counting; the fractional floor scales the requirement up for long cues.
    """
    try:
        a, b = float(start_s), float(end_s)
    except (TypeError, ValueError):
        return False
    if b <= a:
        return False
    overlap = 0.0
    for vs, ve in merge_intervals(voice_regions):
        overlap += max(0.0, min(b, ve) - max(a, vs))
    return overlap >= max(min_overlap_s, min_overlap_frac * (b - a))


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


# ── Voice attestation: no subtitle over silent audio ───────────────────────

# One VAD pass per audio file per process. The map is consulted at several
# boundaries of the same job (inline relisten, persist-time attestation,
# post-COMPLETE recovery) and each fresh pass decodes the full track — the
# cache makes every consult after the first free. Keyed on (path, mtime,
# size) so a re-extracted file re-computes. Failures are never cached: VAD
# can become available mid-process (model download finishing).
_VAD_CACHE: Dict[tuple, List[Interval]] = {}
_VAD_CACHE_MAX = 8


def voice_activity_regions_cached(audio_path: str, **kwargs) -> List[Interval]:
    """``voice_activity_regions`` behind a per-file cache (see above)."""
    import os
    try:
        st = os.stat(audio_path)
        key = (os.path.abspath(audio_path), st.st_mtime_ns, st.st_size,
               tuple(sorted(kwargs.items())))
    except OSError:
        return []
    hit = _VAD_CACHE.get(key)
    if hit is not None:
        return list(hit)
    regions = voice_activity_regions(audio_path, **kwargs)
    if regions:
        while len(_VAD_CACHE) >= _VAD_CACHE_MAX:
            _VAD_CACHE.pop(next(iter(_VAD_CACHE)))
        _VAD_CACHE[key] = list(regions)
    return list(regions)


def attest_cues_to_voice(rows: list, audio_path: str,
                         min_overlap_s: float = 0.15,
                         ) -> tuple[list, list]:
    """Drop SPEECH cues that overlap no VAD-detected voice at all.

    The final audio-truth gate before subtitles are persisted: a measured
    run shipped an 11-cue block at 0:00-0:09 of the timeline — over pure
    silence before the opening theme — because a translation-stage defect
    manufactured cues with degenerate times and the formatter dutifully
    packed them at the head. Wherever such a cue comes from (mistimed
    recovery, orphan LLM output, decode hallucination over music), the one
    property it cannot fake is voiced audio under its window, so that is
    what is attested here.

    Deliberately recall-biased for REAL speech: the bar is a small ABSOLUTE
    overlap (no fractional requirement — readability extension legitimately
    stretches a cue well past its voiced audio), and bracketed markers
    ("[♪ Opening theme ♪]", "[Music]") are exempt because they annotate
    music on purpose. Zero-width and time-less cues have no audio under
    them by definition and are dropped. Fail-soft: no VAD map → rows
    returned unchanged.

    Returns ``(kept_rows, dropped_samples)``.
    """
    if not rows:
        return rows, []
    regions = voice_activity_regions_cached(audio_path)
    if not regions:
        return rows, []
    try:
        from backend.services.audio_analyzer import is_subtitle_marker
    except Exception:
        def is_subtitle_marker(_t: str) -> bool:
            return False
    kept, dropped = [], []
    for r in rows:
        if isinstance(r, dict):
            txt = (r.get("text") or "").strip()
            a, b = r.get("start"), r.get("end")
        else:
            txt = (getattr(r, "text", "") or "").strip()
            a, b = getattr(r, "start", None), getattr(r, "end", None)
        if not txt or is_subtitle_marker(txt):
            kept.append(r)
            continue
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            a, b = 0.0, 0.0
        if b > a and overlaps_voice(regions, a, b,
                                    min_overlap_s=min_overlap_s,
                                    min_overlap_frac=0.0):
            kept.append(r)
            continue
        dropped.append(f"{a:.2f}-{b:.2f}s {txt[:40]!r}")
    if not dropped:
        return rows, []
    return kept, dropped
