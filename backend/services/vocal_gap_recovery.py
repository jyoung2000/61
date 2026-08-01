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


# Longest pause treated as "still the same utterance run" when grouping VAD
# regions. A breath, not a beat of silence — see ``_under_transcribed_spans``.
_VOICE_BRIDGE_S = 0.6


def _chunk(lo: float, hi: float, max_span_s: float) -> list[tuple[float, float]]:
    """``[lo, hi)`` cut into pieces no longer than ``max_span_s``."""
    if max_span_s <= 0 or hi - lo <= max_span_s:
        return [(lo, hi)]
    out, t = [], lo
    while t < hi:
        out.append((t, min(t + max_span_s, hi)))
        t += max_span_s
    return out


def _voiced_subspans(
    gap: tuple[float, float], voice: list, pad_s: float,
    max_span_s: float, min_gap_s: float,
) -> list[tuple[float, float]]:
    """Split an over-long hole into the parts VAD says carry a voice.

    A 45s+ hole used to be dropped whole, on the reasoning that a continuous
    hole that long is a non-speech scene. VAD makes that testable, and on a
    reference episode it was wrong: the 70-second hole before the ending theme
    held two lines of dialogue and the "to be continued" card. Dropping the
    hole dropped them. Keeping the voiced parts costs Demucs only the seconds
    that actually contain speech."""
    lo, hi = gap
    out: list[tuple[float, float]] = []
    for v in sorted(voice or [], key=lambda r: float(r[0])):
        try:
            vs, ve = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            continue
        s, e = max(vs, lo + pad_s), min(ve, hi - pad_s)
        if e - s <= 0:
            continue
        if out and s - out[-1][1] < min_gap_s:
            out[-1] = (out[-1][0], e)          # bridge neighbouring utterances
        else:
            out.append((s, e))
    return [(max(0.0, s - pad_s), e + pad_s) for s, e in out
            if max_span_s <= 0 or (e - s) <= max_span_s]


def _under_transcribed_spans(
    spans: list[tuple[float, float]],
    segments,
    voice: list,
    *,
    pad_s: float,
    min_gap_s: float,
    max_span_s: float,
    density_ratio: float,
) -> list[tuple[float, float]]:
    """Voiced runs the transcript nominally COVERS but barely transcribes.

    ``find_coverage_gaps`` only ever saw holes, and most of a real track's
    missing dialogue is not a hole. Measured on a reference episode: a
    17-second press scrum where the reference has nine lines came back as two
    cues totalling fifteen characters — 0.9 characters per second against the
    track's own median of 9.0. There is no gap there to find, so the recovery
    pass never looked, and it was the single largest content miss in the file.

    Density is measured against the track's OWN median rather than an absolute
    floor, so the test travels across languages and speaking rates."""
    if not voice or density_ratio <= 0:
        return []
    rows = [(b, _seg_text(s)) for s in (segments or []) if (b := _seg_bounds(s))]
    rows = [(b, t) for b, t in rows if t]
    if len(rows) < 8:
        return []

    def _chars_in(lo: float, hi: float) -> int:
        return sum(len(t) for (a, b), t in rows if a < hi and b > lo)

    # Median density over the cues we DO have, so the yardstick is this
    # track's own speaking rate.
    per_cue = sorted(len(t) / max(0.25, b - a) for (a, b), t in rows)
    median = per_cue[len(per_cue) // 2]
    if median <= 0:
        return []
    floor = median * density_ratio

    # Bridge only across BREATH-length pauses. Bridging across ``min_gap_s``
    # welded a 17-second scrum to the scene after it into one 48-second run,
    # which then tripped the span cap and was dropped — the detector found the
    # exact span it was built for and threw it away.
    runs: list[list[float]] = []
    for v in sorted(voice, key=lambda r: float(r[0])):
        try:
            vs, ve = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            continue
        if runs and vs - runs[-1][1] < _VOICE_BRIDGE_S:
            runs[-1][1] = max(runs[-1][1], ve)
        else:
            runs.append([vs, ve])

    out: list[tuple[float, float]] = []
    for lo, hi in runs:
        if hi - lo < min_gap_s:
            continue
        if _chars_in(lo, hi) / (hi - lo) >= floor:
            continue
        # A long run is CHUNKED, not skipped: the cap exists to bound one
        # Demucs call, not to decide whether speech is there.
        for c_lo, c_hi in _chunk(lo, hi, max_span_s):
            cand = (max(0.0, c_lo - pad_s), c_hi + pad_s)
            # Don't re-list something the hole scan already picked up.
            if any(cand[0] < b and cand[1] > a for a, b in spans):
                continue
            out.append(cand)
    return out


def find_coverage_gaps(
    segments,
    *,
    min_gap_s: float = 8.0,
    pad_s: float = 2.0,
    max_spans: int = 8,
    max_total_s: float = 240.0,
    max_span_s: float = 45.0,
    voice_regions: list | None = None,
    density_ratio: float = 0.0,
) -> list[tuple[float, float]]:
    """Spans worth re-transcribing: uncovered timeline holes, plus — when VAD
    regions are supplied — voiced runs the transcript barely transcribes.

    Interior only — silence before the first or after the last cue is
    normally logos/credits, not buried dialogue. A hole longer than
    ``max_span_s`` is not dropped outright any more: with VAD it is reduced to
    the parts that carry a voice, because "long" turned out to be a bad proxy
    for "no speech". Spans are padded by ``pad_s`` on each side (Whisper needs
    lead-in context), smallest ELIGIBLE first, capped at ``max_spans`` and
    ``max_total_s`` recovered seconds so a pathological transcript can't
    schedule half the episode."""
    spans = sorted(b for s in (segments or []) if (b := _seg_bounds(s)))
    if len(spans) < 2:
        return []
    gaps: list[tuple[float, float]] = []
    oversized: list[tuple[float, float]] = []
    cover_end = spans[0][1]
    for a, b in spans[1:]:
        hole = a - cover_end
        if hole >= min_gap_s:
            if max_span_s <= 0 or hole <= max_span_s:
                gaps.append((max(0.0, cover_end - pad_s), a + pad_s))
            else:
                oversized.append((max(0.0, cover_end - pad_s), a + pad_s))
        cover_end = max(cover_end, b)
    if voice_regions:
        # Keep only holes that actually contain a voice. Without this the
        # budget goes to song and title spans (which have no dialogue by
        # definition) and the real misses never get considered.
        gaps = [g for g in gaps if _overlaps_voice(g, voice_regions, pad_s)]
        for g in oversized:
            gaps.extend(_voiced_subspans(
                g, voice_regions, pad_s, max_span_s, min_gap_s))
        gaps.extend(_under_transcribed_spans(
            gaps, segments, voice_regions, pad_s=pad_s, min_gap_s=min_gap_s,
            max_span_s=max_span_s, density_ratio=density_ratio))
        gaps.sort()
    # SMALLEST first. Largest-first contradicted this module's own premise —
    # music-buried dialogue arrives as SHORT holes, so the big spans it
    # preferred are the least likely to contain speech, and they consumed the
    # budget before the short ones were reached. Measured: the real misses on a
    # reference episode were 2.1-5.7 s while the selector spent 228 s on spans
    # of 14-25 s that held only song.
    gaps.sort(key=lambda g: g[1] - g[0])
    if voice_regions:
        # Rank by EXPECTED YIELD — voiced seconds inside the span that no cue
        # currently covers — rather than by span length. Smallest-first was a
        # proxy adopted when long spans reliably meant song; VAD screens for
        # that directly now, and the proxy had started costing content: with
        # both detectors feeding it, the candidate list came to 458 s against
        # a 240 s budget and the two largest real misses (a five-line exchange
        # and a whole reaction beat) were evicted in favour of shorter spans
        # holding nothing. Ranking by the quantity we are trying to recover
        # spends the same budget on the most missing dialogue.
        _cov = sorted(b for s in (segments or []) if (b := _seg_bounds(s)))

        def _yield_s(g: tuple[float, float]) -> float:
            lo, hi = g[0] + pad_s, g[1] - pad_s
            got = 0.0
            for v in voice_regions:
                try:
                    vs, ve = float(v[0]), float(v[1])
                except (TypeError, ValueError, IndexError):
                    continue
                s0, e0 = max(vs, lo), min(ve, hi)
                if e0 <= s0:
                    continue
                covered = sum(max(0.0, min(cb, e0) - max(ca, s0))
                              for ca, cb in _cov if ca < e0 and cb > s0)
                got += max(0.0, (e0 - s0) - covered)
            return got

        gaps.sort(key=lambda g: (-_yield_s(g), g[1] - g[0]))
    picked: list[tuple[float, float]] = []
    total = 0.0
    for g in gaps[: max(0, max_spans)]:
        if total + (g[1] - g[0]) > max_total_s:
            continue
        picked.append(g)
        total += g[1] - g[0]
    picked.sort()
    return picked


def _clip_to_gap(seg: dict, gap: tuple[float, float], pad_s: float,
                 existing: Optional[list] = None) -> Optional[dict]:
    """Keep a recovered cue where it lands inside the gap.

    The pad's PURPOSE is to avoid double-captioning audio an existing cue
    already covers (the padded lead-in/out overlaps those cues) — so a decode
    that only touches the pad is tested against the existing cues DIRECTLY
    instead of being discarded by geometry. Blanket-culling by geometry threw
    away real recovered dialogue: a measured run decoded speech in all 15
    spans ("了解", "ゼクス…") and culled 37/37 segments as "outside-gap",
    which is precisely the dialogue the pass exists to recover."""
    b = _seg_bounds(seg)
    if b is None:
        return None
    lo, hi = gap[0] + pad_s, gap[1] - pad_s
    if b[1] > lo and b[0] < hi:
        # Overlaps the gap interior — the unambiguous keep.
        out = dict(seg)
        out["start"], out["end"] = max(b[0], lo), min(b[1], hi)
        return out if out["end"] - out["start"] >= 0.3 else None
    if b[1] <= gap[0] or b[0] >= gap[1]:
        return None                     # outside even the padded window
    # Pad-only decode: keep it unless an existing cue already covers that
    # audio (≥ 0.2 s overlap) — then it is a boundary re-hearing, not a find.
    for e in (existing or []):
        eb = _seg_bounds(e)
        if eb is None:
            continue
        if min(eb[1], b[1]) - max(eb[0], b[0]) > 0.2:
            return None
    out = dict(seg)
    out["start"], out["end"] = max(b[0], gap[0]), min(b[1], gap[1])
    return out if out["end"] - out["start"] >= 0.3 else None


def _repair_stem_times(segs: list, stem_dur: float,
                       pad_s: float = 0.0,
                       voiced: Optional[list] = None) -> tuple[list, int]:
    """Give decoded segments usable STEM-RELATIVE times when the decode
    returned degenerate ones.

    A measured run culled 86/86 recovered segments across two passes as
    "no-times": the relisten decode heard real dialogue in every span
    (連合本部に察知されていた, - 観戦の破片だろうがな! - 了解!) but each
    segment came back with start == end, so `_seg_bounds` rejected it and
    both known transcript holes stayed open. The TEXT is the recovery's
    whole value and the stem is only a few seconds wide — approximate
    placement beats discarding the line every time.

    Redistribution fires ONLY when NO segment carries a usable time (the
    measured failure mode). A mixed list keeps its exact-timed segments
    verbatim — Whisper routinely emits one zero-length trailing artifact
    beside well-timed segments, and rewriting the good ones for it would
    drift real lines by seconds (and with them the nearest-cue speaker
    guess). The lone artifact still dies in the per-segment no-times cull.

    The window distributed is the PAD-TRIMMED interior: the stem includes
    ``pad_s`` of lead-in/out that overlaps existing cues by construction,
    and a short first/last segment placed wholly inside a pad would be
    culled as a boundary re-hearing — dropping exactly the line the repair
    exists to save. Weighted by text length (the char-proportional model
    used everywhere else in the pipeline).

    ``voiced`` (optional) is the span's VAD voice intervals in STEM-RELATIVE
    seconds. When provided, the char-weight timeline is distributed across
    the CONCATENATED voiced intervals instead of the flat interior, so a
    repaired cue — and the active-word skeleton later derived from it —
    sits on audible speech rather than straddling silence. Fail-soft: too
    little voiced audio (< 0.3 s inside the interior) falls back to the
    flat interior. Returns ``(segs, n_repaired)``."""
    if not segs or stem_dur <= 0.2:
        return segs, 0

    def _valid(s) -> bool:
        b = _seg_bounds(s)
        return (b is not None and b[0] >= -0.5
                and b[1] <= stem_dur + 5.0 and b[1] - b[0] >= 0.05)

    if any(_valid(s) for s in segs):
        return segs, 0
    inset = min(max(0.0, pad_s), stem_dur / 4.0)
    lo, hi = inset, stem_dur - inset
    if hi - lo <= 0.2:
        lo, hi = 0.0, stem_dur

    # Anchor to VAD voice intervals (clipped to the interior, merged) when
    # they cover enough audio to be trustworthy; otherwise the flat interior.
    clipped: list[tuple[float, float]] = []
    for v in voiced or []:
        try:
            vs, ve = max(float(v[0]), lo), min(float(v[1]), hi)
        except (TypeError, ValueError, IndexError):
            continue
        if ve - vs >= 0.1:
            clipped.append((vs, ve))
    spans: list[list[float]] = []
    for vs, ve in sorted(clipped):
        if spans and vs <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], ve)
        else:
            spans.append([vs, ve])
    voiced_total = sum(ve - vs for vs, ve in spans)
    if voiced_total < 0.3:
        spans = [[lo, hi]]
        voiced_total = hi - lo

    def _at(t: float) -> float:
        """Position ``t`` on the concatenated voiced timeline → stem time."""
        for vs, ve in spans:
            if t <= (ve - vs) + 1e-9:
                return vs + t
            t -= ve - vs
        return spans[-1][1]

    weights = [max(1, len(_seg_text(s))) for s in segs]
    total = float(sum(weights))
    out = []
    cursor = 0.0
    for s, w in zip(segs, weights):
        d = dict(s)
        nxt = min(voiced_total, cursor + voiced_total * (w / total))
        d["start"] = round(_at(cursor), 3)
        d["end"] = round(max(_at(nxt), d["start"]), 3)
        cursor = nxt
        out.append(d)
    return out, len(out)


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
    *,
    separate: bool = True,
) -> list[dict]:
    """Run the full recovery for one job; returns recovered SOURCE-language
    cues (possibly empty). Never raises.

    ``separate=False`` is the CHEAP tier: re-ASR the selected spans straight
    from the job audio with no Demucs pass. Most missing dialogue is not buried
    under music at all — it is ordinary speech Whisper's VAD dropped, and a
    plain second listen at a lower threshold gets it back. Measured on a
    reference episode, of nine missing exchanges only the crowd-noise press
    scrum actually needed separation. That matters because it decides WHERE the
    pass can run: separation is minutes of CPU and has to stay post-COMPLETE,
    while a second listen is one warm Whisper call and can run inside the job,
    so its lines reach the subtitle file the user actually downloads."""
    try:
        from backend.config import settings
        from backend.services import vocal_separator

        if not bool(getattr(settings, "VOCAL_GAP_RECOVERY_ENABLED", True)):
            return []
        if not audio_path or not os.path.exists(audio_path):
            logger.info("[%s] gap recovery: job audio missing — skipping", job_id)
            return []
        if separate and not vocal_separator.is_available():
            logger.info("[%s] gap recovery: demucs not installed — skipping", job_id)
            return []
        tag = "gap recovery" if separate else "gap re-listen"

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
            density_ratio=float(getattr(
                settings, "VOCAL_GAP_DENSITY_RATIO", 0.35)),
        )
        if not gaps:
            logger.info("[%s] %s: no coverage gaps ≥ threshold", job_id, tag)
            return []
        logger.info(
            "[%s] %s: %d span(s), %.0fs total — %s", job_id, tag, len(gaps),
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
            logger.info("[%s] %s: no usable span audio", job_id, tag)
            return []

        sep_wav = os.path.join(work_dir, "sep_silence.wav")
        concat_wav = os.path.join(work_dir, "gaps_concat.wav")
        if not (await asyncio.to_thread(_make_silence, sep_wav, SEP_S)
                and await asyncio.to_thread(
                    _concat_wavs, raws, sep_wav, concat_wav, work_dir)):
            logger.info("[%s] %s: span concatenation failed — skipping", job_id, tag)
            return []

        total_s = sum(d for _, d in planned) + SEP_S * max(0, len(planned) - 1)
        if separate:
            # CPU on purpose: recovery may overlap SEO's GPU work, and one pass
            # over a couple of concatenated minutes stays comfortably bounded.
            vocals = await asyncio.to_thread(
                vocal_separator.separate_vocals, concat_wav,
                os.path.join(work_dir, "sep"),
                model=str(getattr(settings, "VOCAL_SEPARATION_MODEL", "htdemucs")),
                device=str(getattr(settings, "VOCAL_GAP_DEVICE", "cpu")),
                segment=int(getattr(settings, "VOCAL_SEPARATION_SEGMENT", 7)),
                timeout=int(120 + total_s * 4),
            )
            if not vocals:
                logger.info("[%s] %s: separation unavailable", job_id, tag)
                return []
        else:
            # Cheap tier: listen again to the ORIGINAL audio. No separation, so
            # the concatenated track is already what we want to transcribe.
            vocals = concat_wav

        offsets = _concat_offsets([d for _, d in planned], SEP_S)
        recovered: list[dict] = []
        # Failure accounting so an infrastructure failure can never masquerade
        # as silence again. A run whose 15 stems all errored into a cold
        # sidecar logged the same "found no new dialogue" as genuine quiet.
        # Culls are broken down BY REASON with samples: the very next run
        # decoded 44 segments across 15 spans and culled every one of them
        # behind a single opaque counter — undiagnosable from the log.
        n_failed = n_empty = n_heard = n_times_repaired = 0
        n_cull_clip = n_cull_junk = n_cull_nospeech = n_cull_notimes = 0
        _cull_samples: list[str] = []
        # The FIRST stem pays the sidecar's cold-start (measured 11 s of model
        # load, during which every request errors); later stems hit it warm.
        _patience = float(getattr(
            settings, "VOCAL_GAP_ASR_WARMUP_PATIENCE_S", 90.0))
        # Per-SPAN ledger: which span produced what, and where each decoded
        # segment went. The aggregate counters below say "44 culled" but not
        # WHICH hole stayed open or why — the two persistently-missing spans
        # (a measured 6:58-7:08 and 11:41-11:58) were undiagnosable without
        # a per-span verdict line.
        _span_rows: list[str] = []

        def _mmss(t: float) -> str:
            return f"{int(t // 60)}:{t % 60:04.1f}"

        for (gap, dur), off in zip(planned, offsets):
            _span_lbl = f"{_mmss(gap[0])}-{_mmss(gap[1])}"
            stem = os.path.join(work_dir, f"stem_{int(gap[0])}.wav")
            if not await asyncio.to_thread(_slice_wav, vocals, stem, off, off + dur):
                _span_rows.append(f"{_span_lbl}=slice-failed")
                continue
            segs = await asyncio.to_thread(
                _transcribe_stem, stem, source_lang, _patience)
            _patience = 20.0    # warm now; keep a small cushion per stem
            if segs is None:
                n_failed += 1
                _span_rows.append(f"{_span_lbl}=ASR-FAILED")
                continue
            if not segs:
                n_empty += 1
                _span_rows.append(f"{_span_lbl}=empty")
                continue
            n_heard += 1
            # Degenerate decode times (start == end) get the stem window
            # distributed across them instead of a guaranteed "no-times" cull
            # — the measured failure mode was EVERY relisten segment arriving
            # timeless, which silently kept both known transcript holes open.
            # Stem-relative VAD intervals for this span: repaired times land
            # on audible speech, not distributed across silence.
            _stem_voice = []
            for _v in _voice:
                try:
                    _vs = max(float(_v[0]), gap[0]) - gap[0]
                    _ve = min(float(_v[1]), gap[1]) - gap[0]
                except (TypeError, ValueError, IndexError):
                    continue
                if _ve > _vs:
                    _stem_voice.append((_vs, _ve))
            segs, _sp_repaired = _repair_stem_times(
                segs, dur, pad_s=pad, voiced=_stem_voice)
            if _sp_repaired:
                n_times_repaired += _sp_repaired
            _sp_kept = _sp_clip = _sp_junk = _sp_nospeech = _sp_notimes = 0
            for s in segs:
                _raw_txt = _seg_text(s)
                s["start"] = float(s.get("start", 0.0)) + gap[0]
                s["end"] = float(s.get("end", 0.0)) + gap[0]
                _b0 = _seg_bounds(s)
                if _b0 is None:
                    # Decoded text with no usable timestamps — a DIFFERENT
                    # failure from landing outside the span, and previously
                    # indistinguishable from it in the log.
                    n_cull_notimes += 1
                    _sp_notimes += 1
                    if len(_cull_samples) < 4:
                        _cull_samples.append(f"no-times:{_raw_txt[:40]!r}")
                    continue
                s = _clip_to_gap(s, gap, pad, existing=_speech)
                if s is None:
                    n_cull_clip += 1
                    _sp_clip += 1
                    if len(_cull_samples) < 4:
                        # Carry the decoded bounds: whether the cull was a
                        # boundary re-hearing or a mis-clocked decode is
                        # readable straight off the numbers.
                        _cull_samples.append(
                            f"clip[{_b0[0]:.1f}-{_b0[1]:.1f} vs span "
                            f"{gap[0]:.1f}-{gap[1]:.1f}]:{_raw_txt[:40]!r}")
                    continue
                txt = _seg_text(s)
                if not txt or _JUNK_RE.match(txt):
                    n_cull_junk += 1
                    _sp_junk += 1
                    if len(_cull_samples) < 4:
                        _cull_samples.append(f"junk:{txt[:40]!r}")
                    continue
                if float(s.get("no_speech_prob", 0.0) or 0.0) > 0.85:
                    n_cull_nospeech += 1
                    _sp_nospeech += 1
                    if len(_cull_samples) < 4:
                        _cull_samples.append(f"nospeech:{txt[:40]!r}")
                    continue
                s["speaker"] = _default_speaker(segments, s["start"])
                s["text"] = txt
                recovered.append(s)
                _sp_kept += 1
            _sp_bits = [f"{len(segs)} dec", f"{_sp_kept} kept"]
            if _sp_clip:
                _sp_bits.append(f"{_sp_clip} outside-gap")
            if _sp_junk:
                _sp_bits.append(f"{_sp_junk} junk")
            if _sp_nospeech:
                _sp_bits.append(f"{_sp_nospeech} no-speech")
            if _sp_notimes:
                _sp_bits.append(f"{_sp_notimes} no-times")
            if _sp_repaired:
                _sp_bits.append(f"{_sp_repaired} times-repaired")
            _span_rows.append(f"{_span_lbl}=" + "/".join(_sp_bits))
        if _span_rows:
            logger.info("[%s] %s: per-span outcomes: %s",
                        job_id, tag, "; ".join(_span_rows))
        n_culled = n_cull_clip + n_cull_junk + n_cull_nospeech + n_cull_notimes
        _cull_detail = (
            f"{n_culled} culled ({n_cull_clip} outside-gap, {n_cull_junk} junk, "
            f"{n_cull_nospeech} no-speech, {n_cull_notimes} no-times"
            + (f"; e.g. {'; '.join(_cull_samples)}" if _cull_samples else "") + ")"
            + (f"; {n_times_repaired} degenerate decode time(s) repaired"
               if n_times_repaired else ""))
        if recovered:
            logger.info(
                "[%s] %s: %d cue(s) recovered from %s "
                "(%d/%d span(s) heard speech, %d empty, %d ASR-failed, %s)",
                job_id, tag, len(recovered),
                "buried audio" if separate else "a second listen",
                n_heard, len(planned), n_empty, n_failed, _cull_detail)
        else:
            logger.info(
                "[%s] %s: found no new dialogue "
                "(%d span(s): %d decoded empty, %d ASR-FAILED, %s)%s",
                job_id, tag, len(planned), n_empty, n_failed, _cull_detail,
                " — the failures mean this verdict is NOT trustworthy"
                if n_failed else "")
        return recovered
    except Exception as e:
        logger.warning("[%s] gap recovery skipped (%s)", job_id, e)
        return []


# Decode gates for the SECOND listen. These spans were selected precisely
# because the tuned first pass dropped them, so re-sending the identical
# thresholds reproduces the identical silence. Silero already vetted that a
# voice is in the slice; the junk regex and the no_speech cull downstream
# handle what a laxer decode lets through.
_RELISTEN_TUNING = {
    "vad_threshold": 0.05,
    "no_speech_threshold": 0.80,
    "log_prob_threshold": -2.0,
    "condition_on_previous_text": "false",
    "hallucination_silence_threshold": 4.0,
}


def _transcribe_stem(wav_path: str, source_lang: str,
                     patience_s: float = 0.0) -> Optional[list[dict]]:
    """ASR one vocal stem → list of {start,end,text,no_speech_prob} dicts.

    Remote Companion whisper when configured. Returns ``None`` when the
    TRANSPORT failed (every attempt errored) and ``[]`` when the decode
    succeeded but heard nothing — the caller must not confuse the two: a
    measured run fired 15 stems into a sidecar that was still loading its
    model, every request failed inside the 11-second warmup, and the pass
    reported "found no new dialogue" over spans with plainly audible speech.
    ``patience_s`` re-tries across exactly that window; the first stem pays
    it once and the rest hit a warm sidecar."""
    try:
        import time as _t
        from backend.services.reframer_audio import (
            RemoteWhisperEngine, remote_whisper_configured)
        if not remote_whisper_configured():
            return None
        deadline = _t.monotonic() + max(0.0, float(patience_s or 0.0))
        attempt = 0
        while True:
            attempt += 1
            res = None
            try:
                res = RemoteWhisperEngine().transcribe_wav(
                    wav_path, language=(source_lang or None),
                    tuning_overrides=_RELISTEN_TUNING)
            except Exception as e:
                logger.info("gap recovery ASR attempt %d failed (%s)", attempt, e)
            if res is not None:
                break
            if _t.monotonic() >= deadline:
                return None
            _t.sleep(4.0)
        return _normalize_stem_segments((res or {}).get("segments") or [])
    except Exception as e:
        logger.info("gap recovery ASR failed (%s)", e)
        return None


def _normalize_stem_segments(segs: list) -> list[dict]:
    """Remote stem decodes → CLEAN ``{start,end,text,no_speech_prob}`` dicts.

    The old ``dict(s)`` copy carried EVERY remote key through the recovery:
    the remote mapper times segments as ``start_sec``/``end_sec`` (so
    ``_seg_bounds``, which reads ``start``/``end``, saw nothing and the
    repair redistributed 100% of real decode times), and the per-word rows
    stay STEM-RELATIVE forever — the repair and the ``+gap[0]`` shift only
    rewrite segment times. A measured run merged those cues and the
    sentence resegmenter's word-timed path then re-timed them to the word
    rows — 0-13 s of the video — planting the whole recovery at the head
    of the timeline over the opening theme.

    So: prefer ``start``/``end``, fall back to ``start_sec``/``end_sec``
    (the real decode times, stem-relative — exactly what the caller
    shifts), and carry NOTHING else. Word rows die here on purpose:
    recovered cues get their timing from the repair/clip machinery, and
    downstream tiers rebuild word timing against real audio."""
    out = []
    for s in segs:
        if isinstance(s, dict):
            start = s.get("start", s.get("start_sec", 0.0))
            end = s.get("end", s.get("end_sec", 0.0))
            text = s.get("text", "")
            nsp = s.get("no_speech_prob", 0.0)
        else:
            start = getattr(s, "start", getattr(s, "start_sec", 0.0))
            end = getattr(s, "end", getattr(s, "end_sec", 0.0))
            text = getattr(s, "text", "")
            nsp = getattr(s, "no_speech_prob", 0.0)
        try:
            start = float(start or 0.0)
            end = float(end or 0.0)
        except (TypeError, ValueError):
            start = end = 0.0
        out.append({"start": start, "end": end, "text": text or "",
                    "no_speech_prob": nsp})
    return out
