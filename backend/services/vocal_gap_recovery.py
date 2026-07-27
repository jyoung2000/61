"""Targeted vocal-separation gap recovery — fill music-buried dropouts.

Dialogue under loud BGM/SFX comes back from Whisper as *no_speech*, so whole
scenes drop out of the transcript (confirmed: the Gundam Wing ep-1 press
conference — a reporter barrage over crowd noise — shipped as a 25-second
hole in every run). Full-track separation before ASR would fix it but delays
transcription start and kills the early-translation overlap, so it stays off.

This module is the surgical version:

  1. Find COVERAGE GAPS in the finished transcript (uncovered timeline holes
     between the first and last cue, bounded in count and total seconds).
  2. Slice ONLY those spans out of the already-extracted job audio.
  3. Demucs each slice on the CPU (a few tens of seconds of audio — never
     touches either GPU while it may still be busy).
  4. Re-transcribe the isolated vocal stems (remote Companion whisper when
     paired, else the local engine path the caller provides).
  5. Return recovered cues clipped to their gap so they can be merged
     ADDITIVELY — existing cues are never modified or replaced.

Runs post-COMPLETE (like the deferred SEO) so the analysis time the user
sees is untouched; the transcript refreshes in place when recovery lands.
Fail-soft everywhere: no Demucs, no gaps, ASR failure, or garbage output
just means "no recovered cues"."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from difflib import SequenceMatcher
from typing import Optional

logger = logging.getLogger("clipai.vocal_gap_recovery")


def _seg_bounds(seg) -> Optional[tuple[float, float]]:
    if isinstance(seg, dict):
        a, b = seg.get("start"), seg.get("end")
    else:
        a, b = getattr(seg, "start", None), getattr(seg, "end", None)
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    return (a, b) if b > a else None


def _seg_text(seg) -> str:
    t = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
    return (t or "").strip()


def _overlaps_voice(gap: tuple[float, float], voice: list, pad_s: float,
                    min_voice_s: float = 0.4) -> bool:
    """True when the gap's UNPADDED interior contains at least ``min_voice_s``
    of VAD-detected speech.

    This is the signal the selector was missing. Holes are computed against our
    own transcript, so "no cue here" also covers every second of theme music —
    and a measured run spent its whole 228-second budget on song and title
    spans, ran a Demucs separation over them, and correctly reported no new
    dialogue. Meanwhile the five spans that DID hold speech were never
    considered. Silero VAD answers the question the transcript cannot: is there
    a voice in this hole at all."""
    lo, hi = gap[0] + pad_s, gap[1] - pad_s
    if hi <= lo:
        return False
    total = 0.0
    for v in voice or []:
        try:
            vs, ve = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            continue
        total += max(0.0, min(ve, hi) - max(vs, lo))
        if total >= min_voice_s:
            return True
    return False


def find_coverage_gaps(
    segments,
    *,
    min_gap_s: float = 8.0,
    pad_s: float = 2.0,
    max_spans: int = 8,
    max_total_s: float = 240.0,
    max_span_s: float = 45.0,
    voice_regions: list | None = None,
) -> list[tuple[float, float]]:
    """Uncovered timeline holes between the first and last cue, ``min_gap_s`` ≤
    length ≤ ``max_span_s``.

    Interior only — silence before the first or after the last cue is
    normally logos/credits, not buried dialogue. A hole LONGER than
    ``max_span_s`` is skipped: music-buried DIALOGUE arrives as short holes,
    whereas a continuous 45s+ hole is a non-speech scene (music / action) —
    Demucs-separating minutes of it costs many post-COMPLETE minutes and
    recovers nothing. Spans are padded by ``pad_s`` on each side (Whisper
    needs lead-in context), largest ELIGIBLE first, capped at ``max_spans``
    and ``max_total_s`` recovered seconds so a pathological transcript can't
    schedule half the episode."""
    spans = sorted(b for s in (segments or []) if (b := _seg_bounds(s)))
    if len(spans) < 2:
        return []
    gaps: list[tuple[float, float]] = []
    cover_end = spans[0][1]
    for a, b in spans[1:]:
        hole = a - cover_end
        if hole >= min_gap_s and (max_span_s <= 0 or hole <= max_span_s):
            gaps.append((max(0.0, cover_end - pad_s), a + pad_s))
        cover_end = max(cover_end, b)
    if voice_regions:
        # Keep only holes that actually contain a voice. Without this the
        # budget goes to song and title spans (which have no dialogue by
        # definition) and the real misses never get considered.
        gaps = [g for g in gaps if _overlaps_voice(g, voice_regions, pad_s)]
    # SMALLEST first. Largest-first contradicted this module's own premise —
    # music-buried dialogue arrives as SHORT holes, so the big spans it
    # preferred are the least likely to contain speech, and they consumed the
    # budget before the short ones were reached. Measured: the real misses on a
    # reference episode were 2.1-5.7 s while the selector spent 228 s on spans
    # of 14-25 s that held only song.
    gaps.sort(key=lambda g: g[1] - g[0])
    picked: list[tuple[float, float]] = []
    total = 0.0
    for g in gaps[: max(0, max_spans)]:
        if total + (g[1] - g[0]) > max_total_s:
            continue
        picked.append(g)
        total += g[1] - g[0]
    picked.sort()
    return picked


def _clip_to_gap(seg: dict, gap: tuple[float, float], pad_s: float) -> Optional[dict]:
    """Keep a recovered cue only where it lands INSIDE the (unpadded) gap —
    the padded lead-in/out overlaps existing cues and would double them."""
    b = _seg_bounds(seg)
    if b is None:
        return None
    lo, hi = gap[0] + pad_s, gap[1] - pad_s
    if b[1] <= lo or b[0] >= hi:
        return None
    out = dict(seg)
    out["start"], out["end"] = max(b[0], lo), min(b[1], hi)
    return out if out["end"] - out["start"] >= 0.3 else None


def _similar(a: str, b: str) -> float:
    a, b = (a or "").lower().strip(), (b or "").lower().strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def merge_recovered(existing: list, recovered: list[dict]) -> tuple[list, int]:
    """Insert recovered cues into ``existing`` (returned sorted by start).

    Additive only: existing cues are untouched. A recovered cue is dropped
    when it text-matches (≥0.7) a temporal neighbor — Whisper re-hearing the
    padded boundary — or another recovered cue already accepted."""
    out = list(existing or [])
    added = 0
    accepted: list[dict] = []
    for r in sorted(recovered or [], key=lambda s: s.get("start", 0.0)):
        rt = _seg_text(r)
        if not rt:
            continue
        rb = _seg_bounds(r)
        if rb is None:
            continue
        dup = False
        for e in out:
            eb = _seg_bounds(e)
            if eb is None or abs(eb[0] - rb[0]) > 20.0:
                continue
            if _similar(rt, _seg_text(e)) >= 0.7:
                dup = True
                break
        if not dup:
            for a in accepted:
                if _similar(rt, _seg_text(a)) >= 0.8:
                    dup = True
                    break
        if dup:
            continue
        accepted.append(r)
        added += 1
    out.extend(accepted)

    def _key(s):
        b = _seg_bounds(s)
        return b[0] if b else 0.0

    out.sort(key=_key)
    return out, added


# Whisper hallucination staples that show up on near-silent vocal stems.
_JUNK_RE = re.compile(
    r"^(?:thank you(?: for watching)?|thanks for watching|please subscribe"
    r"|ご視聴ありがとうございました|お疲れ様でした)[.!\s]*$",
    re.IGNORECASE,
)


def _slice_wav(audio_path: str, out_path: str, start: float, end: float) -> bool:
    try:
        rc = subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
             "-i", audio_path, "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", out_path],
            capture_output=True, timeout=120,
        ).returncode
        return rc == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception:
        return False


def _wav_duration(path: str) -> float:
    """Seconds of audio in a WAV (ffprobe); 0.0 on any failure."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=30)
        return float((r.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def _make_silence(out_path: str, sec: float, sr: int = 16000) -> bool:
    """A ``sec``-second 16 kHz mono PCM silence clip (the concat separator)."""
    try:
        rc = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-t", f"{max(0.1, sec):.2f}",
             "-i", f"anullsrc=r={sr}:cl=mono", "-c:a", "pcm_s16le", out_path],
            capture_output=True, timeout=60).returncode
        return rc == 0 and os.path.exists(out_path)
    except Exception:
        return False


def _concat_wavs(inputs: list[str], sep_path: str, out_path: str,
                 work_dir: str) -> bool:
    """Concatenate ``inputs`` into ``out_path`` with ``sep_path`` between each.

    All clips share params (16 kHz mono PCM), so the concat demuxer with
    ``-c copy`` is exact and cheap. Returns False on any failure."""
    try:
        listfile = os.path.join(work_dir, "gaps_concat_list.txt")
        lines = []
        for i, p in enumerate(inputs):
            if i:
                lines.append(f"file '{sep_path}'")
            lines.append(f"file '{p}'")
        with open(listfile, "w") as f:
            f.write("\n".join(lines) + "\n")
        rc = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
             "-c", "copy", out_path],
            capture_output=True, timeout=300).returncode
        return rc == 0 and os.path.exists(out_path)
    except Exception:
        return False


def _concat_offsets(durations: list[float], sep_s: float) -> list[float]:
    """Start offset of each clip inside a silence-separated concatenation
    (pure — unit tested). Clip i starts after all prior clips + separators."""
    offs, cur = [], 0.0
    for d in durations:
        offs.append(cur)
        cur += d + sep_s
    return offs


def _default_speaker(segments, at_s: float) -> str:
    """Nearest existing cue's speaker so recovered lines blend in."""
    best, best_d = "Speaker 1", float("inf")
    for s in segments or []:
        b = _seg_bounds(s)
        if b is None:
            continue
        d = min(abs(b[0] - at_s), abs(b[1] - at_s))
        if d < best_d:
            spk = (s.get("speaker") if isinstance(s, dict)
                   else getattr(s, "speaker", None)) or "Speaker 1"
            best, best_d = spk, d
    return best


async def recover_gap_dialogue(
    job_id: str,
    audio_path: str,
    segments: list,
    source_lang: str,
    work_dir: str,
) -> list[dict]:
    """Run the full recovery for one job; returns recovered SOURCE-language
    cues (possibly empty). Never raises."""
    try:
        from backend.config import settings
        from backend.services import vocal_separator

        if not bool(getattr(settings, "VOCAL_GAP_RECOVERY_ENABLED", True)):
            return []
        if not audio_path or not os.path.exists(audio_path):
            logger.info("[%s] gap recovery: job audio missing — skipping", job_id)
            return []
        if not vocal_separator.is_available():
            logger.info("[%s] gap recovery: demucs not installed — skipping", job_id)
            return []

        # Music markers ("[♪ music ♪]") are NOT coverage — they are the exact
        # spans where buried dialogue lives. Compute holes against speech
        # cues only.
        try:
            from backend.services.audio_analyzer import is_subtitle_marker
            _speech = [s for s in (segments or [])
                       if not is_subtitle_marker(_seg_text(s))]
        except Exception:
            _speech = list(segments or [])

        # VAD voice regions, straight from the audio and independent of any
        # Whisper decode — the one signal that can tell a hole holding buried
        # speech from a hole holding music. Computed here rather than reused
        # because the producers upstream collapse the region list to a scalar
        # and never persist it. Fail-soft: no VAD → unfiltered behaviour.
        _voice: list = []
        if bool(getattr(settings, "VOCAL_GAP_REQUIRE_VOICE", True)):
            try:
                from backend.services.speech_coverage import voice_activity_regions
                _voice = voice_activity_regions(audio_path, speech_pad_ms=0) or []
                logger.info("[%s] gap recovery: VAD found %d voice region(s) to "
                            "screen candidate holes against", job_id, len(_voice))
            except Exception as _vad_e:
                logger.info("[%s] gap recovery: VAD unavailable (%s) — selecting "
                            "holes without a voice check", job_id, _vad_e)

        pad = float(getattr(settings, "VOCAL_GAP_PAD_S", 2.0))
        gaps = find_coverage_gaps(
            _speech,
            min_gap_s=float(getattr(settings, "VOCAL_GAP_MIN_S", 8.0)),
            pad_s=pad,
            max_spans=int(getattr(settings, "VOCAL_GAP_MAX_SPANS", 8)),
            max_total_s=float(getattr(settings, "VOCAL_GAP_MAX_TOTAL_S", 240.0)),
            max_span_s=float(getattr(settings, "VOCAL_GAP_MAX_SPAN_S", 45.0)),
            voice_regions=_voice,
        )
        if not gaps:
            logger.info("[%s] gap recovery: no coverage gaps ≥ threshold", job_id)
            return []
        logger.info(
            "[%s] gap recovery: %d gap(s), %.0fs total — %s", job_id, len(gaps),
            sum(b - a for a, b in gaps),
            ", ".join(f"{int(a) // 60}:{int(a) % 60:02d}-{int(b) // 60}:{int(b) % 60:02d}"
                      for a, b in gaps))

        os.makedirs(work_dir, exist_ok=True)
        # ONE Demucs pass for ALL spans. On CPU the cost is dominated by
        # subprocess start + model load (~25-35s), NOT the few seconds of audio
        # per span — separating each span in its own subprocess paid that fixed
        # cost once PER GAP (a measured 11-minute post-COMPLETE tail across 13
        # gaps that recovered nothing). Instead: slice every span, concatenate
        # them into ONE track with a short silence separator, separate that once,
        # then slice the vocal stem back per span for ASR. Same audio processed,
        # a single model load.
        SEP_S = 1.0
        raws: list[str] = []
        planned: list[tuple[tuple[float, float], float]] = []  # (gap, duration)
        for i, gap in enumerate(gaps):
            raw = os.path.join(work_dir, f"gap{i}.wav")
            if not await asyncio.to_thread(_slice_wav, audio_path, raw, gap[0], gap[1]):
                continue
            dur = await asyncio.to_thread(_wav_duration, raw)
            if dur <= 0.2:
                continue
            raws.append(raw)
            planned.append((gap, dur))
        if not raws:
            logger.info("[%s] gap recovery: no usable span audio", job_id)
            return []

        sep_wav = os.path.join(work_dir, "sep_silence.wav")
        concat_wav = os.path.join(work_dir, "gaps_concat.wav")
        if not (await asyncio.to_thread(_make_silence, sep_wav, SEP_S)
                and await asyncio.to_thread(
                    _concat_wavs, raws, sep_wav, concat_wav, work_dir)):
            logger.info("[%s] gap recovery: span concatenation failed — skipping",
                        job_id)
            return []

        total_s = sum(d for _, d in planned) + SEP_S * max(0, len(planned) - 1)
        # CPU on purpose: recovery may overlap SEO's GPU work, and one pass over
        # a couple of concatenated minutes stays comfortably bounded.
        vocals = await asyncio.to_thread(
            vocal_separator.separate_vocals, concat_wav,
            os.path.join(work_dir, "sep"),
            model=str(getattr(settings, "VOCAL_SEPARATION_MODEL", "htdemucs")),
            device=str(getattr(settings, "VOCAL_GAP_DEVICE", "cpu")),
            segment=int(getattr(settings, "VOCAL_SEPARATION_SEGMENT", 7)),
            timeout=int(120 + total_s * 4),
        )
        if not vocals:
            logger.info("[%s] gap recovery: separation unavailable", job_id)
            return []

        offsets = _concat_offsets([d for _, d in planned], SEP_S)
        recovered: list[dict] = []
        for (gap, dur), off in zip(planned, offsets):
            stem = os.path.join(work_dir, f"stem_{int(gap[0])}.wav")
            if not await asyncio.to_thread(_slice_wav, vocals, stem, off, off + dur):
                continue
            segs = await asyncio.to_thread(_transcribe_stem, stem, source_lang)
            for s in segs:
                s["start"] = float(s.get("start", 0.0)) + gap[0]
                s["end"] = float(s.get("end", 0.0)) + gap[0]
                s = _clip_to_gap(s, gap, pad)
                if s is None:
                    continue
                txt = _seg_text(s)
                if not txt or _JUNK_RE.match(txt):
                    continue
                if float(s.get("no_speech_prob", 0.0) or 0.0) > 0.85:
                    continue
                s["speaker"] = _default_speaker(segments, s["start"])
                s["text"] = txt
                recovered.append(s)
        if recovered:
            logger.info(
                "[%s] gap recovery: %d cue(s) recovered from buried audio",
                job_id, len(recovered))
        else:
            logger.info("[%s] gap recovery: separation found no new dialogue", job_id)
        return recovered
    except Exception as e:
        logger.warning("[%s] gap recovery skipped (%s)", job_id, e)
        return []


def _transcribe_stem(wav_path: str, source_lang: str) -> list[dict]:
    """ASR one vocal stem → list of {start,end,text,no_speech_prob} dicts.
    Remote Companion whisper when configured; [] on any failure."""
    try:
        from backend.services.reframer_audio import (
            RemoteWhisperEngine, remote_whisper_configured)
        if not remote_whisper_configured():
            return []
        res = RemoteWhisperEngine().transcribe_wav(
            wav_path, language=(source_lang or None))
        segs = (res or {}).get("segments") or []
        out = []
        for s in segs:
            d = dict(s) if isinstance(s, dict) else {
                "start": getattr(s, "start", 0.0),
                "end": getattr(s, "end", 0.0),
                "text": getattr(s, "text", ""),
                "no_speech_prob": getattr(s, "no_speech_prob", 0.0),
            }
            out.append(d)
        return out
    except Exception as e:
        logger.info("gap recovery ASR failed (%s)", e)
        return []
