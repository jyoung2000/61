"""Sentence-aware resegmentation (Task 4).

Whisper emits one segment per VAD-detected acoustic window, which on
dialogue-dense content produces the "30 s unbroken block" the pipeline
comments complain about. Otter.ai instead breaks transcripts at
sentence / clause boundaries. This module does the same:

  1. Merge adjacent same-speaker segments (never across a speaker
     boundary — that would glue two people's turns together).
  2. Re-split each merged block at sentence-final punctuation
     (``. ! ? 。 ！ ？ …``), using Whisper word timestamps to assign an
     accurate ``start`` / ``end`` to each sentence-segment.
  3. Language-aware: CJK uses ``。！？`` and joins words without spaces;
     Latin scripts join on spaces.

It does **not** enforce duration/CPS itself — over-long sentences are
handed off to ``subtitle_formatter.enforce_readability`` downstream
(which the pipeline already runs right after this step).

Runs after speaker fusion (Task 2) and before the readability pass.
Gated behind ``SENTENCE_SEGMENTATION_ENABLED`` (default True).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from backend.models import TranscriptSegment

logger = logging.getLogger("clipai.sentence_segmenter")

# Sentence terminators (Latin + CJK).
_SENT_END = set(".!?。！？…")
# Trailing characters that can follow a terminator without ending the check.
_CLOSERS = "\"')]}」』）”’"


def _is_cjk(text: str) -> bool:
    """Reuse the subtitle formatter's CJK heuristic when available."""
    try:
        from backend.services.subtitle_formatter import _is_cjk as _impl
        return _impl(text)
    except Exception:
        if not text:
            return False
        cjk = sum(
            1 for ch in text
            if 0x3040 <= ord(ch) <= 0x30FF
            or 0x4E00 <= ord(ch) <= 0x9FFF
            or 0xAC00 <= ord(ch) <= 0xD7A3
        )
        total = sum(1 for ch in text if not ch.isspace())
        return total > 0 and cjk / total >= 0.30


def _ends_sentence(word_text: str) -> bool:
    s = (word_text or "").rstrip()
    # Strip trailing closing quotes / brackets before checking.
    while s and s[-1] in _CLOSERS:
        s = s[:-1]
    return bool(s) and s[-1] in _SENT_END


def _join_words(words: list, is_cjk: bool) -> str:
    parts = [str(_w_get(w, "word", "")).strip() for w in words]
    parts = [p for p in parts if p]
    if is_cjk:
        return "".join(parts)
    return " ".join(parts)


def _w_get(word, key, default=None):
    if isinstance(word, dict):
        return word.get(key, default)
    return getattr(word, key, default)


def _new_segment(template: TranscriptSegment, start: float, end: float,
                 text: str, words: Optional[list]) -> TranscriptSegment:
    return TranscriptSegment(
        start=round(float(start), 3),
        end=round(float(end), 3),
        text=text,
        speaker=template.speaker,
        words=words or None,
        confidence=getattr(template, "confidence", None),
        avg_logprob=getattr(template, "avg_logprob", None),
        no_speech_prob=getattr(template, "no_speech_prob", None),
    )


def _split_text_sentences(text: str, is_cjk: bool) -> list[str]:
    """Split raw text into sentences (no word timing available)."""
    text = (text or "").strip()
    if not text:
        return []
    if is_cjk:
        # Keep the terminator with the preceding sentence.
        parts = re.split(r"(?<=[。！？…])", text)
    else:
        parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _split_segment_by_sentence(seg: TranscriptSegment) -> list[TranscriptSegment]:
    is_cjk = _is_cjk(seg.text or "")
    words = seg.words or []

    if not words:
        # No word timing — split text and distribute duration by char length.
        sentences = _split_text_sentences(seg.text, is_cjk)
        if len(sentences) <= 1:
            return [seg]
        total_chars = sum(len(s) for s in sentences) or 1
        duration = max(0.0, seg.end - seg.start)
        out: list[TranscriptSegment] = []
        cursor = seg.start
        for i, sent in enumerate(sentences):
            frac = len(sent) / total_chars
            end = seg.end if i == len(sentences) - 1 else cursor + duration * frac
            out.append(_new_segment(seg, cursor, max(cursor, end), sent, None))
            cursor = end
        return out

    # Word-timed path — group words into sentences.
    sentences: list[list] = []
    cur: list = []
    for w in words:
        cur.append(w)
        if _ends_sentence(_w_get(w, "word", "")):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    if len(sentences) <= 1:
        return [seg]

    out = []
    for sw in sentences:
        text = _join_words(sw, is_cjk)
        if not text:
            continue
        start = float(_w_get(sw[0], "start", seg.start) or seg.start)
        end = float(_w_get(sw[-1], "end", start) or start)
        out.append(_new_segment(seg, start, max(start, end), text, sw))
    return out or [seg]


def resegment_by_sentence(segments: list) -> list:
    """Merge same-speaker neighbours, then re-split at sentence boundaries.

    Accepts a list of :class:`TranscriptSegment` (dicts are coerced) and
    returns sentence-aligned segments with monotonic, non-overlapping
    timing. Returns the input unchanged when there is nothing to do.
    """
    if not segments:
        return segments
    segs: list[TranscriptSegment] = [
        TranscriptSegment(**s) if isinstance(s, dict) else s for s in segments
    ]

    # 1. Merge adjacent same-speaker segments (never across speakers).
    merged: list[TranscriptSegment] = []
    for s in segs:
        if merged and (merged[-1].speaker or "") == (s.speaker or ""):
            prev = merged[-1]
            is_cjk = _is_cjk((prev.text or "") + (s.text or ""))
            joiner = "" if is_cjk else " "
            new_text = ((prev.text or "").strip() + joiner + (s.text or "").strip()).strip()
            new_words = (prev.words or []) + (s.words or [])
            merged[-1] = _new_segment(
                prev, prev.start, s.end, new_text, new_words or None)
        else:
            merged.append(s)

    # 2. Re-split each merged block by sentence.
    out: list[TranscriptSegment] = []
    for s in merged:
        out.extend(_split_segment_by_sentence(s))
    return out
