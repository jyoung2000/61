"""Audio energy analysis for viral moment detection.

Extracts loudness contour from audio using FFmpeg's ebur128 filter,
identifies volume spikes (laughter, applause, excitement), and generates
an energy map for the clip detection prompt.

Also provides ``classify_audio_events()`` — a spectral classifier built
on FFmpeg + scipy (no ML model) that tags each window of audio as
``speech``, ``music``, ``laughter``, ``applause``, ``silence``, or
``noise``. The output integrates with the subtitle pipeline (the
[applause] / [music] / [laughter] annotations) and the SignalTimeline.
"""

import asyncio
import logging
import os
import re
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


async def analyze_audio_energy(
    audio_path: str,
    window_seconds: float = 2.0,
    spike_threshold_db: float = 6.0,
    max_moments: int = 25,
) -> list[dict]:
    """Analyze audio for energy spikes using FFmpeg loudness metering.

    Returns a list of {timestamp, loudness_db, delta_db, type} dicts for high-energy moments.
    """
    # Use FFmpeg's astats filter to get per-window RMS levels
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-af", f"asegment=timestamps=0,astats=metadata=1:reset={int(window_seconds * 100)}",
        "-f", "null", "-",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            logger.warning("Audio analysis process did not exit after kill — force continuing")
        logger.warning("Audio analysis timed out")
        return []

    # Parse RMS levels from FFmpeg stderr output
    output = stderr.decode(errors='replace')
    rms_values = []
    current_time = 0.0

    for line in output.split('\n'):
        # Look for RMS level patterns in astats output
        rms_match = re.search(r'RMS level dB:\s*(-?\d+\.?\d*)', line)
        if rms_match:
            rms_db = float(rms_match.group(1))
            rms_values.append((current_time, rms_db))
            current_time += window_seconds

    if not rms_values:
        # Fallback: use volumedetect for overall stats
        logger.info("No per-window RMS data, using volumedetect fallback")
        return []

    # Calculate baseline loudness (median)
    sorted_rms = sorted(v[1] for v in rms_values if v[1] > -60)  # Ignore silence
    if not sorted_rms:
        return []

    baseline = sorted_rms[len(sorted_rms) // 2]

    # Find energy spikes above threshold
    moments = []
    for timestamp, rms_db in rms_values:
        if rms_db - baseline > spike_threshold_db:
            spike_type = "volume_spike"
            if rms_db - baseline > spike_threshold_db * 2:
                spike_type = "extreme_spike"
            moments.append({
                "timestamp": round(timestamp, 1),
                "loudness_db": round(rms_db, 1),
                "delta_db": round(rms_db - baseline, 1),
                "type": spike_type,
                "sentiment": classify_audio_sentiment(rms_db, rms_db - baseline, spike_type),
            })

    # Also detect sudden silence-to-loud transitions (reveals, drops)
    for i in range(1, len(rms_values)):
        prev_rms = rms_values[i - 1][1]
        curr_rms = rms_values[i][1]
        if prev_rms < baseline - 10 and curr_rms > baseline + spike_threshold_db:
            moments.append({
                "timestamp": round(rms_values[i][0], 1),
                "loudness_db": round(curr_rms, 1),
                "delta_db": round(curr_rms - prev_rms, 1),
                "type": "silence_to_loud",
                "sentiment": "applause",
            })

    # Sort by delta_db (most dramatic first) and cap
    moments.sort(key=lambda m: m["delta_db"], reverse=True)
    capped = moments[:max_moments]
    # Re-sort by timestamp for chronological output
    capped.sort(key=lambda m: m["timestamp"])

    logger.info("Audio energy analysis: %d spikes detected (baseline=%.1f dB)", len(capped), baseline)
    return capped


def classify_audio_sentiment(loudness_db: float, delta_db: float, spike_type: str) -> str:
    """Classify an audio moment into a coarse sentiment tag.

    Phase 5 of the OpusClip parity gap. We deliberately avoid a new
    ML model — these tags come from rule-based thresholds on the
    existing FFmpeg loudness signal. They feed the LLM via the
    SENTIMENT TIMELINE block and the hot-zone scorer's audio
    component.

    Tags returned:
      - laughter: brief sustained burst above baseline
      - cheering: high-energy spike with strong delta
      - shouting: extreme spike, very loud
      - applause: silence → loud transition
      - silence: handled by the gap detector (not from this function)
      - neutral: anything we can't confidently classify
    """
    if spike_type == "silence_to_loud":
        return "applause"
    if spike_type == "extreme_spike":
        if loudness_db > -8:
            return "shouting"
        return "cheering"
    if spike_type == "volume_spike":
        if delta_db > 12:
            return "cheering"
        if delta_db > 8:
            return "laughter"
    return "neutral"


def format_sentiment_timeline(moments: list[dict], top_n: int = 20) -> str:
    """Format an audio sentiment timeline for injection into the clip prompt.

    Returns an empty string when no moments carry sentiment tags so
    the orchestrator can skip the SENTIMENT TIMELINE block entirely
    and the LLM never sees an empty header.
    """
    if not moments:
        return ""
    tagged = [m for m in moments if m.get("sentiment") and m.get("sentiment") != "neutral"]
    if not tagged:
        return ""
    tagged.sort(key=lambda m: m.get("delta_db", 0), reverse=True)
    top = tagged[:top_n]
    top.sort(key=lambda m: m.get("timestamp", 0))
    lines = []
    for m in top:
        ts = float(m.get("timestamp", 0))
        sentiment = m.get("sentiment", "neutral")
        delta = m.get("delta_db", 0)
        lines.append(f"  [{ts:.1f}s] {sentiment} (+{delta:.0f}dB)")
    return "\n".join(lines)


def format_audio_energy_map(moments: list[dict]) -> str:
    """Format audio energy moments for injection into clip detection prompt."""
    if not moments:
        return ""

    lines = []
    for m in moments:
        type_label = {
            "volume_spike": "LOUD",
            "extreme_spike": "VERY LOUD",
            "silence_to_loud": "SILENCE->LOUD",
        }.get(m["type"], "SPIKE")

        lines.append(f"[{m['timestamp']:.0f}s] {type_label} (+{m['delta_db']:.0f}dB)")

    return (
        "\n\nAUDIO ENERGY SPIKES (detected from audio waveform — these are real volume peaks, "
        "not transcript guesses. Clips containing these moments tend to be more engaging):\n"
        + "\n".join(lines)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Spectral Audio Event Classification (no ML model required)
# ═══════════════════════════════════════════════════════════════════════════

# Reference frequency bands (Hz) used for the rule-based classifier.
_FUNDAMENTAL_BAND = (60, 250)        # voiced speech fundamental, bass notes
_FORMANT_BAND    = (250, 3500)       # vowel formants — speech intelligibility
_PRESENCE_BAND   = (3500, 8000)      # consonants, applause / cymbal energy
_AIR_BAND        = (8000, 16000)     # very high frequencies — applause hash


def _extract_pcm_mono(
    audio_path: str,
    sample_rate: int = 16000,
) -> tuple[bytes, int]:
    """Decode the input to 16 kHz mono PCM via FFmpeg. Returns (raw_bytes,
    sample_rate). On failure returns ``(b'', 0)``."""
    import subprocess
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", audio_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(sample_rate), "-ac", "1",
                "-f", "s16le", "-",
            ],
            capture_output=True, timeout=600,
        )
        if proc.returncode != 0:
            logger.warning("FFmpeg PCM extract failed: %s", proc.stderr[:200].decode(errors="replace"))
            return b"", 0
        return proc.stdout, sample_rate
    except Exception as e:
        logger.warning("FFmpeg PCM extract crashed: %s", e)
        return b"", 0


def _classify_window(
    samples,
    sample_rate: int,
    rms_threshold_silence: float,
) -> tuple[str, float]:
    """Classify a single 1-second window. Returns (label, confidence).

    Pure-Python rule-based classifier — no ML model. Uses RMS + a few
    spectral descriptors (centroid, flatness, band-energy ratios).
    """
    try:
        import numpy as np
    except Exception:
        return ("noise", 0.0)
    if len(samples) == 0:
        return ("silence", 1.0)

    arr = samples.astype("float32") / 32768.0
    rms = float((arr * arr).mean() ** 0.5)
    if rms < rms_threshold_silence:
        return ("silence", min(1.0, (rms_threshold_silence - rms) * 20))

    # Spectrum via real FFT.
    n = len(arr)
    # Hann window to suppress spectral leakage.
    win = arr * np.hanning(n)
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spec2 = spec * spec
    total_power = float(spec2.sum()) or 1e-9

    def _band_power(lo, hi):
        mask = (freqs >= lo) & (freqs < hi)
        return float(spec2[mask].sum())

    fund = _band_power(*_FUNDAMENTAL_BAND) / total_power
    formant = _band_power(*_FORMANT_BAND) / total_power
    presence = _band_power(*_PRESENCE_BAND) / total_power
    air = _band_power(*_AIR_BAND) / total_power

    # Spectral centroid (Hz) — speech ≈ 1-3 kHz, music varies, applause is high.
    centroid = float((freqs * spec).sum() / max(1e-9, spec.sum()))
    # Spectral flatness (geometric mean / arithmetic mean) — noise/applause is high (~0.3+),
    # tonal music/speech is low (~0.05).
    nonzero = spec[spec > 1e-9]
    if len(nonzero) > 1:
        log_mean = float(np.log(nonzero).mean())
        arith_mean = float(nonzero.mean())
        flatness = float(np.exp(log_mean) / arith_mean) if arith_mean > 0 else 0.0
    else:
        flatness = 0.0

    # ── Rule-based decisions ──
    # Applause / crowd noise: very flat spectrum, high presence/air energy.
    if flatness > 0.35 and (presence + air) > 0.30 and centroid > 2500:
        return ("applause", min(1.0, flatness * 2))

    # Music: tonal (low flatness), strong formant + fundamental energy,
    # centroid moderate.
    if flatness < 0.15 and (fund + formant) > 0.55 and 500 < centroid < 4000:
        return ("music", min(1.0, 1.0 - flatness * 3))

    # Speech: balanced formant energy, centroid ≈ 1-2.5 kHz, moderate flatness.
    if 0.05 < flatness < 0.25 and formant > 0.35 and 800 < centroid < 3000:
        return ("speech", 0.7)

    # Laughter: high-energy spike, very broad spectrum, centroid >2 kHz,
    # presence > formant. Tends to be loud.
    if rms > 0.15 and presence > formant * 0.8 and centroid > 1500 and flatness > 0.20:
        return ("laughter", 0.6)

    return ("noise", 0.4)


async def classify_audio_events(
    audio_path: str,
    window_seconds: float = 1.0,
) -> list[dict]:
    """Classify audio events using spectral analysis (no ML model).

    Returns a list of ``{"timestamp": float, "duration": float, "type":
    str, "confidence": float}`` dicts spanning the entire audio. Adjacent
    same-type windows are merged into a single event.

    Types: ``speech``, ``music``, ``laughter``, ``applause``,
    ``silence``, ``noise``.
    """
    try:
        import numpy as np
    except Exception as e:
        logger.warning("classify_audio_events: numpy unavailable (%s) — returning empty", e)
        return []

    raw, sr = await asyncio.to_thread(_extract_pcm_mono, audio_path, 16000)
    if not raw or sr <= 0:
        return []

    samples = np.frombuffer(raw, dtype="<i2")
    if samples.size == 0:
        return []

    # Estimate a per-clip silence threshold: the 10th-percentile RMS.
    win_samples = max(1, int(window_seconds * sr))
    if samples.size < win_samples:
        return []

    # Pre-compute per-window RMS to derive silence floor.
    n_windows = samples.size // win_samples
    rms_arr = np.empty(n_windows, dtype="float32")
    for i in range(n_windows):
        chunk = samples[i * win_samples:(i + 1) * win_samples].astype("float32") / 32768.0
        rms_arr[i] = float((chunk * chunk).mean() ** 0.5)
    if rms_arr.size == 0:
        return []
    silence_floor = float(max(0.005, np.percentile(rms_arr, 10) * 1.5))

    raw_events: list[dict] = []
    for i in range(n_windows):
        chunk = samples[i * win_samples:(i + 1) * win_samples]
        label, conf = _classify_window(chunk, sr, silence_floor)
        raw_events.append({
            "timestamp": round(i * window_seconds, 3),
            "duration": window_seconds,
            "type": label,
            "confidence": round(conf, 3),
        })

    # Merge adjacent same-type events.
    merged: list[dict] = []
    for ev in raw_events:
        if merged and merged[-1]["type"] == ev["type"]:
            merged[-1]["duration"] = round(
                merged[-1]["duration"] + ev["duration"], 3
            )
            # Keep the higher of the two confidences.
            merged[-1]["confidence"] = round(
                max(merged[-1]["confidence"], ev["confidence"]), 3
            )
        else:
            merged.append(dict(ev))

    # Drop micro-events shorter than 0.5s of non-speech (e.g. a single
    # noisy frame inside a long speech block).
    cleaned: list[dict] = []
    for ev in merged:
        if ev["type"] != "speech" and ev["duration"] < 0.5:
            # Re-tag as the surrounding speech if possible.
            if cleaned and cleaned[-1]["type"] == "speech":
                cleaned[-1]["duration"] = round(cleaned[-1]["duration"] + ev["duration"], 3)
                continue
        cleaned.append(ev)

    logger.info(
        "classify_audio_events: %d events across %.1fs",
        len(cleaned), n_windows * window_seconds,
    )
    return cleaned


# Default marker for sustained music regions (OP/ED themes, insert songs).
# A unicode music note bracketed so it reads as a caption cue, not dialogue.
MUSIC_MARKER = "[♪ music ♪]"


# Bare ASR music tags. Whisper emits "[Music]" / "[music]" / "[MUSIC]" of its
# own accord, and because they are bracketed they satisfy is_subtitle_marker,
# get held out of translation, and are re-inserted verbatim — which is exactly
# why a measured run shipped two cues reading "[Music]" as if they were
# dialogue, one of them 2.3s after a legitimate marker saying the same thing.
# They carry no information the styled marker does not.
_BARE_MUSIC_TAG_RE = re.compile(r"^\[\s*music\s*\]$", re.IGNORECASE)


def normalize_music_marker(text: str) -> str:
    """Fold a bare ASR ``[Music]`` tag onto :data:`MUSIC_MARKER`.

    Returns ``text`` unchanged for everything else, including markers that
    already carry the note. Pure; safe to apply to any cue."""
    return MUSIC_MARKER if _BARE_MUSIC_TAG_RE.match((text or "").strip()) else text


def is_subtitle_marker(text: str) -> bool:
    """True when ``text`` is a bracketed non-speech caption marker (music /
    applause / laughter) rather than translatable dialogue. Used to keep
    such cues verbatim through the translator."""
    t = (text or "").strip()
    if not t:
        return False
    if not (t.startswith("[") and t.endswith("]")):
        return False
    inner = t[1:-1]
    # A marker has no sentence-like content — just a short tag (+ the note).
    return ("♪" in t) or (len(inner.split()) <= 2 and inner.replace(" ", "").isalpha())


def merge_markers(transcript: list, markers: list) -> list:
    """Merge non-speech ``markers`` into ``transcript`` (both lists of dicts),
    returned sorted by start time. Markers that overlap an existing segment
    are dropped so they never collide with dialogue. Pure / deterministic."""
    if not markers:
        return transcript

    def _span(seg):
        if isinstance(seg, dict):
            return (float(seg.get("start", seg.get("start_sec", 0)) or 0),
                    float(seg.get("end", seg.get("end_sec", 0)) or 0))
        return (float(getattr(seg, "start", 0) or 0), float(getattr(seg, "end", 0) or 0))

    spans = sorted(_span(s) for s in (transcript or []))

    def _overlaps(s, e):
        for ws, we in spans:
            if we <= s:
                continue
            if ws >= e:
                break
            if min(we, e) > max(ws, s):
                return True
        return False

    out = list(transcript or [])
    for m in markers:
        ms, me = _span(m)
        if me <= ms or _overlaps(ms, me):
            continue
        out.append(m)
    out.sort(key=lambda s: _span(s)[0])
    return out


async def detect_music_markers(
    audio_path: str,
    transcript: list,
    min_seconds: float = 5.0,
    label: str = MUSIC_MARKER,
) -> list:
    """Classify the audio and return ``[{start,end,text}]`` marker cues for
    sustained music regions (>= ``min_seconds``) that don't overlap speech.

    No-ops to ``[]`` when numpy/ffmpeg are unavailable or the audio can't be
    read — so the pipeline degrades to "no markers" without raising.
    """
    try:
        events = await classify_audio_events(audio_path)
    except Exception as e:
        logger.info("detect_music_markers: classify failed (%s) — no markers", e)
        return []
    music = [e for e in events
             if e.get("type") == "music" and float(e.get("duration", 0)) >= min_seconds]
    if not music:
        return []
    return _markers_from_music_events(music, transcript, min_seconds, label)


def _markers_from_music_events(music_events: list, transcript: list,
                               min_seconds: float, label: str) -> list:
    """Build ``[♪ music ♪]`` marker cues from sustained-music events.

    Wraps ``build_non_speech_subtitle_events`` (which emits ``"[music]"``) and
    relabels to the ♪ form. Shared by ``detect_music_markers`` and
    ``mark_and_suppress_music`` so the marker shape is identical."""
    raw = build_non_speech_subtitle_events(
        music_events, transcript, min_event_s=min_seconds)
    for ev in raw:
        ev["text"] = label
        ev["speaker"] = ""
    return raw


def _music_spans_from_events(events: list, min_seconds: float) -> list:
    """Absolute ``(start, end)`` spans for sustained music events."""
    spans = []
    for e in events or []:
        if e.get("type") != "music":
            continue
        dur = float(e.get("duration", 0) or 0)
        if dur < min_seconds:
            continue
        start = float(e.get("timestamp", e.get("start", 0)) or 0)
        spans.append((start, start + dur))
    return spans


def _is_nonlexical_vocalization(text: str) -> bool:
    """True when ``text`` is sung / hummed filler or onomatopoeia
    (``ああああ``, ``lalala``, ``mmmm``, ``ーーー``) rather than real dialogue.

    Music-span suppression uses this so it removes only the vocalisations
    Whisper invents over a song bed — never lexically-diverse real dialogue
    that merely overlaps a span the spectral classifier *mislabelled*
    ``music`` (e.g. dialogue over a loud orchestral / action cue). A
    minute-long spoken section always has many distinct characters and is
    therefore always kept; losing it is far worse than leaving one stray
    sung line. Pure / deterministic; script-agnostic (the distinct-character
    test works for CJK syllabaries and elongated latin vowels alike)."""
    import re as _re
    t = _re.sub(r"[\s\W_]+", "", str(text or ""), flags=_re.UNICODE).lower()
    if len(t) < 4:
        # Too short to call sung filler — keep real short words ("はい",
        # "ok", "go"). Suppression targets the long song bed, not these.
        return False
    distinct = len(set(t))
    if distinct <= 2:
        # ≤2 morae stretched over ≥4 positions: "ああああ" / "lalala" /
        # "ーーー" / "mmmm" — the classic hallucinated-lyric signature.
        return True
    # A single character dominating the cue ("aaaargh", "naaaaa").
    most = max(t.count(c) for c in set(t))
    return most / len(t) >= 0.7


def suppress_speech_in_music_spans(
    transcript: list,
    music_spans: list,
    min_overlap_frac: float = 0.6,
    vocalizations_only: bool = True,
) -> tuple[list, list]:
    """Drop transcribed speech cues sitting inside sustained music-only spans.

    Whisper hallucinates lyrics / vocalisations (the ``ああああ`` / fake-lyric
    cues) over a song bed; this removes any cue whose timespan is ≥
    ``min_overlap_frac`` inside a music span so the span can be positively
    labelled ``[♪ music ♪]`` instead. Bracketed markers are always kept.

    When ``vocalizations_only`` is True (default), a cue inside a music span
    is dropped ONLY if it reads as a sung vocalisation
    (:func:`_is_nonlexical_vocalization`). The spectral classifier is not
    perfect — a loud orchestral / action cue under dialogue can be mislabelled
    ``music`` — so lexically-diverse real dialogue is kept even when it
    overlaps a "music" span; otherwise whole spoken sections disappear from
    the transcript. The ``[♪ music ♪]`` marker for such a span is then dropped
    downstream by ``merge_markers`` (it overlaps the surviving dialogue), so a
    true OP/ED song still gets marked while real dialogue survives. Set
    ``vocalizations_only=False`` to restore blanket suppression.

    Returns ``(kept, suppressed)``. Pure / deterministic."""
    if not transcript or not music_spans:
        return list(transcript or []), []
    spans = sorted((float(s), float(e)) for s, e in music_spans if float(e) > float(s))
    if not spans:
        return list(transcript), []

    def _span(seg):
        if isinstance(seg, dict):
            return (float(seg.get("start", seg.get("start_sec", 0)) or 0),
                    float(seg.get("end", seg.get("end_sec", 0)) or 0))
        return (float(getattr(seg, "start", 0) or 0),
                float(getattr(seg, "end", 0) or 0))

    def _txt(seg):
        return seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", "")

    def _music_overlap_frac(s, e):
        seg_len = max(1e-6, e - s)
        ov = 0.0
        for ms, me in spans:
            if me <= s:
                continue
            if ms >= e:
                break
            ov += max(0.0, min(e, me) - max(s, ms))
        return ov / seg_len

    kept, suppressed = [], []
    for seg in transcript:
        txt = _txt(seg)
        if is_subtitle_marker(txt):
            kept.append(seg)
            continue
        s, e = _span(seg)
        inside_music = e > s and _music_overlap_frac(s, e) >= min_overlap_frac
        if inside_music and (
            not vocalizations_only or _is_nonlexical_vocalization(txt)
        ):
            suppressed.append(seg)
        else:
            kept.append(seg)
    return kept, suppressed


async def mark_and_suppress_music(
    audio_path: str,
    transcript: list,
    min_seconds: float = 5.0,
    suppress: bool = True,
    min_overlap_frac: float = 0.6,
    label: str = MUSIC_MARKER,
    vocalizations_only: bool = True,
) -> tuple[list, int, int]:
    """Classify the audio ONCE, then suppress hallucinated speech in sustained
    music-only spans and insert ``[♪ music ♪]`` markers over them.

    Returns ``(new_transcript, n_suppressed, n_markers)``. Degrades to a no-op
    ``(transcript, 0, 0)`` when audio/numpy is unavailable. The single classify
    pass is shared between suppression and marking (Task 4)."""
    try:
        events = await classify_audio_events(audio_path)
    except Exception as e:
        logger.info("mark_and_suppress_music: classify failed (%s) — no-op", e)
        return transcript, 0, 0
    music_events = [e for e in events
                    if e.get("type") == "music"
                    and float(e.get("duration", 0) or 0) >= min_seconds]
    if not music_events:
        return transcript, 0, 0
    n_suppressed = 0
    if suppress:
        spans = _music_spans_from_events(music_events, min_seconds)
        transcript, dropped = suppress_speech_in_music_spans(
            transcript, spans, min_overlap_frac=min_overlap_frac,
            vocalizations_only=vocalizations_only)
        n_suppressed = len(dropped)
    # Build markers AFTER suppression so they fill the now-cleared song spans
    # (a marker that overlapped a hallucinated lyric would have been dropped).
    markers = _markers_from_music_events(music_events, transcript, min_seconds, label)
    if markers:
        transcript = merge_markers(transcript, markers)
    return transcript, n_suppressed, len(markers)


def build_non_speech_subtitle_events(
    audio_events: list[dict],
    transcript_segments: list,
    min_gap_s: float = 0.6,
    min_event_s: float = 0.8,
) -> list[dict]:
    """Translate non-speech audio events into bracketed subtitle events.

    Produces ``[{"start": s, "end": s, "text": "[applause]"}]`` dicts
    that the export pipeline can merge with the speech transcript when
    ``AUDIO_EVENTS_IN_SUBTITLES`` is enabled. Only emits events that fall
    entirely inside gaps in the speech transcript, to avoid overlapping
    actual dialogue.
    """
    if not audio_events:
        return []

    # Build a list of speech windows for quick gap testing.
    speech_windows = []
    for seg in transcript_segments or []:
        start = getattr(seg, "start", getattr(seg, "start_sec", None))
        end = getattr(seg, "end", getattr(seg, "end_sec", None))
        if isinstance(seg, dict):
            start = seg.get("start", seg.get("start_sec"))
            end = seg.get("end", seg.get("end_sec"))
        if start is None or end is None:
            continue
        speech_windows.append((float(start), float(end)))
    speech_windows.sort()

    def _overlaps_speech(s: float, e: float) -> bool:
        for ws, we in speech_windows:
            if we + min_gap_s < s:
                continue
            if ws - min_gap_s > e:
                break
            if min(we, e) > max(ws, s):
                return True
        return False

    label_map = {
        "applause": "[applause]",
        "laughter": "[laughter]",
        "music": "[music]",
    }
    out: list[dict] = []
    for ev in audio_events:
        label = label_map.get(ev.get("type"))
        if not label:
            continue
        if ev.get("duration", 0) < min_event_s:
            continue
        start = float(ev.get("timestamp", 0))
        end = start + float(ev.get("duration", 0))
        if _overlaps_speech(start, end):
            continue
        out.append({"start": start, "end": end, "text": label})
    return out


# One spectral classification per audio file per process. ``classify_audio_events``
# decodes and analyses the whole track, and by the time the translated transcript
# is finalized the source path has already paid for it — so the theme collapse
# gets its music spans free. Keyed on (path, mtime, size) so a re-extracted file
# re-classifies; failures are never cached.
_MUSIC_SPAN_CACHE: dict = {}
_MUSIC_SPAN_CACHE_MAX = 8


def _spans_of_type(events: list, kind: str) -> list:
    """Absolute ``(start, end)`` spans for every event of ``kind``."""
    out = []
    for e in events or []:
        if e.get("type") != kind:
            continue
        start = float(e.get("timestamp", e.get("start", 0)) or 0)
        out.append((start, start + float(e.get("duration", 0) or 0)))
    return out


def _covered_seconds(spans: list, a: float, b: float) -> float:
    """Seconds of ``[a, b)`` covered by ``spans`` (which may overlap)."""
    total, cur_a, cur_b = 0.0, None, None
    for s, e in sorted(spans):
        s, e = max(s, a), min(e, b)
        if e <= s:
            continue
        if cur_b is not None and s <= cur_b:
            cur_b = max(cur_b, e)
        else:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = s, e
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def _bridge_spans(spans: list, bridge_s: float, blockers: list = (),
                  blocker_tol_s: float = 2.0) -> list:
    """Merge spans separated by no more than ``bridge_s`` into one.

    The classifier works in short windows — a measured run produced 699
    events across 1467 s, about two seconds each — so a minute-long theme
    arrives as thirty separate fragments, and any "is this span long
    enough?" test applied to the raw events answers no every time. Worse,
    a SUNG theme keeps flipping between the music and speech labels
    because the vocal IS voice, so the fragments are not even contiguous.
    Bridging across the short gaps is what turns fragments back into the
    region a human would point at.

    The default tolerance is deliberately modest. On a reference episode the
    opening theme ends about twenty seconds before the opening narration
    begins, so a bridge that reached that far would weld a real, captioned
    narration onto the theme and delete it. Tolerate a sung phrase, not a
    scene.

    ``blockers`` closes the hole that tolerance alone leaves. A measured run
    bridged fragments of BACKGROUND SCORE playing under the opening narration
    into a single 142-209 s "music" span, and the theme collapse then deleted
    sixty-five seconds of real, professionally-captioned dialogue. Duration is
    not evidence of a theme when the gaps between the fragments are full of
    speech. A gap holding more than ``blocker_tol_s`` seconds of blocker is
    not bridged.

    This is deliberately strict, and the cost is understood: a SUNG theme
    flips to the speech label on every vocal phrase, and phrases run longer
    than the tolerance, so a sung theme will usually fail to assemble here.
    That is the correct trade. Across every measured run the audio pass has
    never once located a sung theme — the spans it produced sat under
    dialogue — while it has destroyed a narration scene. What survives this
    gate is an instrumental bed, which is unambiguous; a sung theme falls
    through to the text pass, which finds it reliably."""
    out: list = []
    for a, b in sorted((float(x[0]), float(x[1])) for x in spans if x[1] > x[0]):
        if out and a - out[-1][1] <= bridge_s and (
                not blockers
                or _covered_seconds(blockers, out[-1][1], a) <= blocker_tol_s):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


async def music_spans_cached(audio_path: str, min_seconds: float = 5.0,
                             bridge_s: float = 8.0,
                             min_music_frac: float = 0.6) -> list:
    """Sustained music-only ``(start, end)`` spans, cached per audio file.

    Dialogue over a score classifies as ``speech``, so a span returned here is
    positive evidence of music WITHOUT speech — which is what makes it usable
    as ground truth for "these cues are sung". Adjacent fragments are bridged
    (see ``_bridge_spans``) BEFORE the duration filter, because the raw events
    are seconds long and no theme would ever clear a meaningful bar without
    that.

    Two tests must both pass, because bridging alone proved far too generous.
    Speech events BLOCK a bridge, so score under a scene cannot be welded into
    one long span; and the bridged result must still be ``min_music_frac``
    actual music by coverage, so a span assembled out of thin fragments spread
    across a dialogue scene is rejected even when no single gap was wide
    enough to block it. Returns ``[]`` on any failure so every caller degrades
    to its previous behaviour."""
    import os as _os
    try:
        st = _os.stat(audio_path)
        key = (_os.path.abspath(audio_path), st.st_mtime_ns, st.st_size)
    except OSError:
        return []
    hit = _MUSIC_SPAN_CACHE.get(key)
    if hit is None:
        try:
            events = await classify_audio_events(audio_path)
        except Exception as e:
            logger.info("music_spans_cached: classify failed (%s) — no spans", e)
            return []
        hit = (_music_spans_from_events(events, 0.0),
               _spans_of_type(events, "speech"))
        if hit[0]:
            while len(_MUSIC_SPAN_CACHE) >= _MUSIC_SPAN_CACHE_MAX:
                _MUSIC_SPAN_CACHE.pop(next(iter(_MUSIC_SPAN_CACHE)))
            _MUSIC_SPAN_CACHE[key] = (list(hit[0]), list(hit[1]))
    music, speech = hit
    merged = _bridge_spans(music, bridge_s, blockers=speech)
    long_enough = [s for s in merged if (s[1] - s[0]) >= min_seconds]
    out = [s for s in long_enough
           if _covered_seconds(music, s[0], s[1]) >= min_music_frac * (s[1] - s[0])]
    logger.info(
        "music spans: %d raw event span(s) → %d bridged (%d speech blocker(s)) "
        "→ %d over %.0fs → %d at least %.0f%% music%s",
        len(music), len(merged), len(speech), len(long_enough), min_seconds,
        len(out), min_music_frac * 100.0,
        (" — " + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in out[:8]))
        if out else "")
    return out
