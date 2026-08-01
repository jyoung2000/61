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
                         margin_s: float = 0.5,
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

    Deliberately recall-biased for REAL speech — the very next measured run
    proved a naive gate too eager: it dropped the episode's whispered
    signature line ("I'll kill you") and battle dialogue under loud BGM,
    speech Silero cannot hear. Three recall levers:

      * the map is decoded at a LOWER threshold (0.25) than the coverage
        default, so whispers count as voice;
      * the cue window is widened by ``margin_s`` before the overlap test
        (onset bias and readability extension move cue edges off the voice);
      * a cue carrying MEASURED word rows (``words`` present and not
        ``words_synthetic``) is kept even with zero VAD overlap — those
        times came from a decode or CTC alignment against real audio, which
        is stronger evidence than Silero's opinion of a whisper. Phantom
        cues never carry measured words: their word rows are synthetic
        projections, or gone entirely.

    The bar is a small ABSOLUTE overlap (no fractional requirement), and
    bracketed markers ("[♪ Opening theme ♪]") are exempt because they
    annotate music on purpose. Zero-width and time-less cues have no audio
    under them by definition and are dropped regardless of evidence.
    Fail-soft: no VAD map → rows returned unchanged.

    Returns ``(kept_rows, dropped_samples)``.
    """
    if not rows:
        return rows, []
    regions = voice_activity_regions_cached(audio_path, threshold=0.25)
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
            words = r.get("words")
            synthetic = r.get("words_synthetic")
        else:
            txt = (getattr(r, "text", "") or "").strip()
            a, b = getattr(r, "start", None), getattr(r, "end", None)
            words = getattr(r, "words", None)
            synthetic = getattr(r, "words_synthetic", None)
        if not txt or is_subtitle_marker(txt):
            kept.append(r)
            continue
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            a, b = 0.0, 0.0
        if b > a:
            if overlaps_voice(regions, a - margin_s, b + margin_s,
                              min_overlap_s=min_overlap_s,
                              min_overlap_frac=0.0):
                kept.append(r)
                continue
            if words and not synthetic:
                kept.append(r)
                continue
        dropped.append(f"{a:.2f}-{b:.2f}s {txt[:40]!r}")
    if not dropped:
        return rows, []
    return kept, dropped


def snap_cues_to_voice_onsets(rows: list, audio_path: str,
                              max_shift_s: float = 2.0,
                              min_shift_s: float = 0.15,
                              threshold: float = 0.5,
                              ) -> tuple[list, list]:
    """Pull a cue that starts in SILENCE forward onto the next voice onset.

    Measured against a professional reference, ClipAI's timing error was
    almost perfectly one-sided: of the cues off by more than a second, 20
    were EARLY and 1 was late, clustered into four windows totalling 12% of
    the runtime (one line landed 26 seconds before its audio). Random jitter
    is symmetric; a one-sided error is a systematic offset, and an offset is
    correctable against the one signal that cannot drift — where the voice
    actually starts.

    Conservative by construction, because a wrong snap is worse than a small
    lead:
      * only cues whose start lies in SILENCE are candidates — a cue already
        sitting on voice is left exactly where it is;
      * cues carrying MEASURED word rows are skipped: their times came from
        a decode or CTC alignment against this same audio and outrank a VAD
        region boundary;
      * the shift is capped at ``max_shift_s`` and never crosses into the
        previous cue's window or past the cue's own end;
      * shifts under ``min_shift_s`` are not worth the churn.

    Only the START moves; the end is left alone, so a snapped cue simply
    gets shorter and can never overlap its neighbour. Fail-soft: no VAD map
    → rows returned unchanged. Returns ``(rows, shifted_samples)``."""
    if not rows:
        return rows, []
    # A CONFIDENT map, not the recall-biased one the attestation gate uses.
    # The two passes ask opposite questions. The gate asks "could there be a
    # voice here?" and must say yes to a whisper, so it decodes at 0.25 — and
    # at that sensitivity nearly the whole track reads as voiced. Feeding the
    # same map to this pass made it inert: a measured run found ZERO cues
    # starting in silence and logged nothing at all. This pass asks "is this
    # definitely silence?", which needs the strict default. Risk stays bounded
    # by the guards below: measured word rows always win, and the shift is
    # capped.
    regions = voice_activity_regions_cached(audio_path, threshold=threshold)
    if not regions:
        return rows, []
    try:
        from backend.services.audio_analyzer import is_subtitle_marker
    except Exception:
        def is_subtitle_marker(_t: str) -> bool:
            return False

    def _g(r, k, d=None):
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    def _set(r, k, v):
        if isinstance(r, dict):
            r[k] = v
        else:
            setattr(r, k, v)

    merged = merge_intervals(regions)
    shifted = []
    prev_end = 0.0
    # Instrumentation. Two measured runs produced no log line at all, which
    # left "found nothing" and "wired wrong" indistinguishable — the pass
    # has to say what it looked at even when it changes nothing.
    n = {"examined": 0, "marker": 0, "measured": 0, "on_voice": 0,
         "no_onset": 0, "out_of_range": 0, "blocked": 0}
    for r in rows:
        try:
            a = float(_g(r, "start", 0.0) or 0.0)
            b = float(_g(r, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        txt = (_g(r, "text", "") or "").strip()
        if b <= a or not txt or is_subtitle_marker(txt):
            n["marker"] += 1
            prev_end = max(prev_end, b)
            continue
        n["examined"] += 1
        words = _g(r, "words", None)
        if words and not _g(r, "words_synthetic", None):
            n["measured"] += 1
            prev_end = max(prev_end, b)      # measured times outrank the VAD
            continue
        # In silence? (inside any voiced region → already anchored)
        if any(vs <= a <= ve for vs, ve in merged):
            n["on_voice"] += 1
            prev_end = max(prev_end, b)
            continue
        nxt = None
        for vs, _ve in merged:
            if vs > a:
                nxt = vs
                break
        if nxt is None:
            n["no_onset"] += 1
            prev_end = max(prev_end, b)
            continue
        shift = nxt - a
        if shift < min_shift_s or shift > max_shift_s:
            n["out_of_range"] += 1
            prev_end = max(prev_end, b)
            continue
        new_a = min(nxt, b - 0.2)            # never collapse the cue
        if new_a <= a or new_a < prev_end:
            n["blocked"] += 1
            prev_end = max(prev_end, b)
            continue
        _set(r, "start", round(new_a, 3))
        shifted.append(f"{a:.2f}->{new_a:.2f}s {txt[:32]!r}")
        prev_end = max(prev_end, b)
    logger.info(
        "voice-onset snap: %d speech cue(s) examined against %d voice "
        "region(s) — %d shifted; skipped %d already on voice, %d "
        "audio-measured, %d with no onset ahead, %d beyond the %.1fs cap, "
        "%d blocked by a neighbour",
        n["examined"], len(merged), len(shifted), n["on_voice"],
        n["measured"], n["no_onset"], n["out_of_range"], max_shift_s,
        n["blocked"])
    return rows, shifted
