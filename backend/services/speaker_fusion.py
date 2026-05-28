"""Diarization → transcript fusion (Task 2).

The pyannote diarizer (``reframer_diarizer.SpeakerDiarizer``) emits a
``{time_ms: "SPEAKER_00"}`` timeline at 200 ms resolution. Whisper emits
transcript segments at VAD-detected acoustic-window boundaries. Historically
the merge between the two was an inert stub (``compat_stubs``), so speaker
labels were unreliable.

This module maps the diarization timeline onto transcript segments with:

  * **Overlap voting** — each segment is assigned the speaker who owns the
    most time across its span (bin-overlap, not point sampling).
  * **Turn-taking continuity** — on a tie or < 50 % coverage the segment
    inherits the previous segment's speaker rather than guessing.
  * **Mid-segment splitting** — a segment that spans a speaker changeover is
    split into per-speaker pieces. When Whisper word timestamps are present
    (they are — ``word_timestamps=True`` in reframer_audio), each word is
    attributed to its timeline speaker and consecutive same-speaker words are
    re-grouped into clean speaker turns. Without words, a coarse 60/30
    leading/tail split is used.

Graceful degradation: an empty ``speaker_timeline`` (no ``HF_TOKEN`` / pyannote
unavailable, or the mouth-motion fallback produced nothing) returns the
segments unchanged — the mouth-motion heuristic stands as the documented
fallback.
"""

from __future__ import annotations

import bisect
import logging
from typing import Optional

from backend.models import TranscriptSegment, WordTimestamp

logger = logging.getLogger("clipai.speaker_fusion")

# pyannote's native diarization grid.
DEFAULT_RESOLUTION_MS = 200
# Minimum coverage (overlap / segment-duration) below which we don't trust
# the timeline enough to relabel — inherit the previous speaker instead.
MIN_COVERAGE = 0.5
# Word-level runs shorter than this are merged into their neighbour so a
# single mis-attributed word doesn't fragment a turn.
MIN_RUN_MS = 400


def _build_label_map(speaker_timeline: dict) -> dict:
    """Map raw speaker ids → stable "Speaker N" labels in first-appearance
    order, matching reframer_bridge._speaker_label_map."""
    mapping: dict = {}
    n = 0
    for t in sorted(speaker_timeline):
        sid = speaker_timeline[t]
        if sid and sid not in mapping:
            n += 1
            mapping[sid] = f"Speaker {n}"
    return mapping


def _w_get(word, key, default=None):
    if isinstance(word, dict):
        return word.get(key, default)
    return getattr(word, key, default)


def _speaker_at(t_ms: float, keys: list[int], timeline: dict,
                resolution_ms: int) -> Optional[str]:
    """Return the timeline speaker whose bin contains ``t_ms`` (or ``None``
    when ``t_ms`` falls in an uncovered gap)."""
    if not keys:
        return None
    idx = bisect.bisect_right(keys, t_ms) - 1
    if idx < 0:
        return None
    k = keys[idx]
    if t_ms < k + resolution_ms:
        return timeline.get(k)
    return None


def _overlap_votes(start_ms: float, end_ms: float, keys: list[int],
                   timeline: dict, resolution_ms: int) -> tuple[dict, float]:
    """Sum, per speaker, the bin-overlap duration with ``[start_ms, end_ms]``.

    Returns ``(votes, total_overlap_ms)``.
    """
    votes: dict = {}
    total = 0.0
    if not keys or end_ms <= start_ms:
        return votes, total
    # Only scan bins that could overlap the span.
    lo = bisect.bisect_right(keys, start_ms - resolution_ms)
    for i in range(max(0, lo), len(keys)):
        k = keys[i]
        if k >= end_ms:
            break
        sid = timeline.get(k)
        if not sid:
            continue
        overlap = min(end_ms, k + resolution_ms) - max(start_ms, k)
        if overlap > 0:
            votes[sid] = votes.get(sid, 0.0) + overlap
            total += overlap
    return votes, total


def _majority(votes: dict) -> tuple[Optional[str], bool]:
    """Return ``(top_speaker, is_tie)``."""
    if not votes:
        return None, False
    ordered = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    top_sid, top_v = ordered[0]
    is_tie = len(ordered) > 1 and abs(ordered[1][1] - top_v) < 1e-6
    return top_sid, is_tie


def _new_segment(template: TranscriptSegment, start: float, end: float,
                 text: str, speaker: str, words: Optional[list]) -> TranscriptSegment:
    return TranscriptSegment(
        start=round(float(start), 3),
        end=round(float(end), 3),
        text=text,
        speaker=speaker,
        words=words or None,
        confidence=getattr(template, "confidence", None),
        avg_logprob=getattr(template, "avg_logprob", None),
        no_speech_prob=getattr(template, "no_speech_prob", None),
    )


def _word_runs(seg: TranscriptSegment, keys: list[int], timeline: dict,
               label_map: dict, resolution_ms: int,
               fallback_speaker: str) -> list[dict]:
    """Attribute each word to its timeline speaker and group consecutive
    same-speaker words into runs. Returns a list of ``{speaker, words}``."""
    words = seg.words or []
    attributed: list[tuple] = []  # (word, sid_or_None)
    for w in words:
        ws = float(_w_get(w, "start", seg.start) or seg.start)
        we = float(_w_get(w, "end", ws) or ws)
        mid_ms = (ws + we) / 2.0 * 1000.0
        sid = _speaker_at(mid_ms, keys, timeline, resolution_ms)
        attributed.append((w, sid))

    # Forward-fill gaps from the previous attributed word.
    last = None
    for i, (w, sid) in enumerate(attributed):
        if sid is not None:
            last = sid
        elif last is not None:
            attributed[i] = (w, last)
    # Back-fill any leading Nones from the first known speaker.
    first_known = next((sid for _, sid in attributed if sid is not None), None)
    if first_known is not None:
        for i, (w, sid) in enumerate(attributed):
            if sid is None:
                attributed[i] = (w, first_known)
            else:
                break

    runs: list[dict] = []
    for w, sid in attributed:
        label = label_map.get(sid, fallback_speaker) if sid else fallback_speaker
        if runs and runs[-1]["speaker"] == label:
            runs[-1]["words"].append(w)
        else:
            runs.append({"speaker": label, "words": [w]})

    # Merge runs shorter than MIN_RUN_MS into the previous run (or the next
    # one for a short leading run) to avoid single-word fragmentation.
    def _run_dur_ms(run):
        ws = run["words"]
        s = float(_w_get(ws[0], "start", seg.start) or seg.start)
        e = float(_w_get(ws[-1], "end", s) or s)
        return (e - s) * 1000.0

    merged: list[dict] = []
    for run in runs:
        if merged and _run_dur_ms(run) < MIN_RUN_MS:
            merged[-1]["words"].extend(run["words"])
        else:
            merged.append(run)
    # A short leading run absorbs into the second.
    if len(merged) > 1 and _run_dur_ms(merged[0]) < MIN_RUN_MS:
        merged[1]["words"] = merged[0]["words"] + merged[1]["words"]
        merged[1]["speaker"] = merged[1]["speaker"]
        merged = merged[1:]
    return merged


def _coarse_split(seg: TranscriptSegment, keys: list[int], timeline: dict,
                  label_map: dict, resolution_ms: int) -> Optional[list[TranscriptSegment]]:
    """No-words fallback: split a segment that spans a changeover where the
    leading speaker owns ≥60 % and a different single speaker owns ≥30 % of
    the tail. Text is divided proportionally by time."""
    start_ms = seg.start * 1000.0
    end_ms = seg.end * 1000.0
    dur = end_ms - start_ms
    if dur <= 0:
        return None
    # Contiguous leading speaker from the segment start.
    lead_sid = _speaker_at(start_ms + resolution_ms / 2.0, keys, timeline, resolution_ms)
    if not lead_sid:
        return None
    change_ms = None
    t = start_ms
    while t < end_ms:
        sid = _speaker_at(t + resolution_ms / 2.0, keys, timeline, resolution_ms)
        if sid and sid != lead_sid:
            change_ms = t
            break
        t += resolution_ms
    if change_ms is None:
        return None
    lead_frac = (change_ms - start_ms) / dur
    tail_votes, _ = _overlap_votes(change_ms, end_ms, keys, timeline, resolution_ms)
    tail_sid, _ = _majority(tail_votes)
    tail_frac = (end_ms - change_ms) / dur
    if lead_frac < 0.6 or tail_frac < 0.3 or not tail_sid or tail_sid == lead_sid:
        return None
    # Proportional text split.
    text = seg.text.strip()
    ratio = (change_ms - start_ms) / dur
    cut = max(1, min(len(text) - 1, int(round(len(text) * ratio))))
    # Snap to a word boundary near the cut.
    space = text.rfind(" ", 0, cut)
    if space > 0:
        cut = space
    left_text = text[:cut].strip()
    right_text = text[cut:].strip()
    if not left_text or not right_text:
        return None
    mid_s = change_ms / 1000.0
    return [
        _new_segment(seg, seg.start, mid_s, left_text,
                     label_map.get(lead_sid, seg.speaker), None),
        _new_segment(seg, mid_s, seg.end, right_text,
                     label_map.get(tail_sid, seg.speaker), None),
    ]


def assign_speakers_from_timeline(
    segments: list,
    speaker_timeline: Optional[dict],
    label_map: Optional[dict] = None,
    resolution_ms: int = DEFAULT_RESOLUTION_MS,
) -> list:
    """Attribute transcript ``segments`` to diarized speakers.

    ``segments`` is a list of :class:`TranscriptSegment` (dicts are coerced).
    ``speaker_timeline`` is ``{time_ms: raw_speaker_id}``. An empty timeline
    returns the segments unchanged (mouth-motion fallback stands).
    """
    if not speaker_timeline:
        return segments
    # Coerce dicts → TranscriptSegment.
    segs: list[TranscriptSegment] = [
        TranscriptSegment(**s) if isinstance(s, dict) else s for s in (segments or [])
    ]
    if not segs:
        return segments

    label_map = label_map or _build_label_map(speaker_timeline)
    keys = sorted(int(k) for k in speaker_timeline.keys())
    # Normalize timeline keys to ints for lookup.
    timeline = {int(k): v for k, v in speaker_timeline.items()}

    out: list[TranscriptSegment] = []
    prev_speaker: Optional[str] = None

    for seg in segs:
        start_ms = seg.start * 1000.0
        end_ms = seg.end * 1000.0
        duration_ms = max(1e-6, end_ms - start_ms)
        votes, total = _overlap_votes(start_ms, end_ms, keys, timeline, resolution_ms)
        coverage = total / duration_ms
        top_sid, is_tie = _majority(votes)

        # Decide the fallback (single-speaker) label for this segment.
        if not top_sid or is_tie or coverage < MIN_COVERAGE:
            fallback = prev_speaker or (
                label_map.get(top_sid) if top_sid else None) or seg.speaker or "Speaker 1"
        else:
            fallback = label_map.get(top_sid, seg.speaker or "Speaker 1")

        produced: list[TranscriptSegment]
        if seg.words:
            runs = _word_runs(seg, keys, timeline, label_map, resolution_ms, fallback)
            if len(runs) <= 1:
                produced = [_new_segment(
                    seg, seg.start, seg.end, seg.text, fallback, seg.words)]
            else:
                produced = []
                for run in runs:
                    rwords = run["words"]
                    ws = float(_w_get(rwords[0], "start", seg.start) or seg.start)
                    we = float(_w_get(rwords[-1], "end", ws) or ws)
                    rtext = " ".join(
                        str(_w_get(w, "word", "")).strip() for w in rwords
                    ).strip()
                    if not rtext:
                        continue
                    produced.append(_new_segment(
                        seg, ws, we, rtext, run["speaker"], rwords))
                if not produced:
                    produced = [_new_segment(
                        seg, seg.start, seg.end, seg.text, fallback, seg.words)]
        else:
            split = _coarse_split(seg, keys, timeline, label_map, resolution_ms)
            if split:
                produced = split
            else:
                produced = [_new_segment(
                    seg, seg.start, seg.end, seg.text, fallback, None)]

        out.extend(produced)
        if out:
            prev_speaker = out[-1].speaker

    return out
