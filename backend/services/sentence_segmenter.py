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


def _pause_split_params() -> tuple[bool, float, float, float]:
    """Resolve (enabled, pause_s, max_cue_s, turn_pause_s) from config.

    ``turn_pause_s`` is the LONGER silence treated as a speaker-turn boundary:
    terminator-carrying cues are additionally split there (and same-speaker
    merges are blocked across it) so two people's lines that Whisper welded
    into one cue translate as two independent utterances."""
    try:
        from backend.config import settings as _s
        enabled = bool(getattr(_s, "SENTENCE_SPLIT_PAUSE_ENABLED", True))
        pause_s = float(getattr(_s, "SENTENCE_SPLIT_PAUSE_MS", 400)) / 1000.0
        max_cue_s = float(getattr(_s, "SENTENCE_SPLIT_MAX_CUE_MS", 8000)) / 1000.0
        turn_pause_s = float(getattr(_s, "SENTENCE_SPLIT_TURN_PAUSE_MS", 700)) / 1000.0
    except Exception:
        enabled, pause_s, max_cue_s, turn_pause_s = True, 0.4, 8.0, 0.7
    return enabled, pause_s, max_cue_s, turn_pause_s


def _group_has_terminator(group: list) -> bool:
    return any(_ends_sentence(_w_get(w, "word", "")) for w in group)


def _split_words_by_pause(group: list, pause_s: float, max_cue_s: float) -> list[list]:
    """Split a list of word-timed tokens at inter-word silence gaps.

    Language-agnostic and model-free: a new cue begins when the previous word
    ends a sentence (explicit terminator — still respected), when the silence
    before the next word is at least ``pause_s``, or when the running cue would
    exceed ``max_cue_s``. Words missing timing never force a split."""
    if len(group) < 2:
        return [group]
    out: list[list] = []
    cur: list = [group[0]]
    for w in group[1:]:
        prev = cur[-1]
        prev_end = _w_get(prev, "end", None)
        w_start = _w_get(w, "start", None)
        cur_start = _w_get(cur[0], "start", None)
        gap = None
        if prev_end is not None and w_start is not None:
            gap = float(w_start) - float(prev_end)
        cur_dur = None
        if cur_start is not None and prev_end is not None:
            cur_dur = float(prev_end) - float(cur_start)
        terminator = _ends_sentence(_w_get(prev, "word", ""))
        too_long = cur_dur is not None and cur_dur >= max_cue_s
        big_pause = gap is not None and gap >= pause_s
        if terminator or big_pause or too_long:
            out.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        out.append(cur)
    return out


def _split_segment_by_sentence(seg: TranscriptSegment) -> list[TranscriptSegment]:
    is_cjk = _is_cjk(seg.text or "")
    words = seg.words or []

    # PARTIAL word arrays must never erase text: after a polish whose word
    # remap fell below confidence, a merged block can carry words for only
    # one of its source cues — the word-timed path below rebuilds text
    # exclusively from the words, silently DROPPING the words-less
    # neighbour's text before translation. When the words cover well under
    # the full text, fall back to the text path (char-proportional timing —
    # the pre-existing fallback) instead of losing content.
    if words:
        _joined_chars = sum(
            len(str(_w_get(w, "word", "") or "").strip()) for w in words)
        _text_chars = len((seg.text or "").replace(" ", ""))
        if _text_chars > 0 and _joined_chars < 0.7 * _text_chars:
            words = []

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

    # Word-timed path — group words into sentences at terminators.
    sentences: list[list] = []
    cur: list = []
    for w in words:
        cur.append(w)
        if _ends_sentence(_w_get(w, "word", "")):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)

    # Pause-based refinement: local-mode (and CJK) transcripts often arrive with
    # NO terminators, so terminator grouping leaves the whole block as one cue.
    # Split such groups — and any terminator group that still runs too long — on
    # acoustic silence using the word timestamps. Word-accurate boundaries with
    # no model. Well-punctuated, normal-length sentences are left untouched.
    pause_enabled, pause_s, max_cue_s, turn_pause_s = _pause_split_params()
    if pause_enabled:
        refined: list[list] = []
        for grp in sentences:
            if not grp:
                continue
            grp_start = _w_get(grp[0], "start", None)
            grp_end = _w_get(grp[-1], "end", None)
            grp_dur = (float(grp_end) - float(grp_start)
                       if grp_start is not None and grp_end is not None else 0.0)
            if not _group_has_terminator(grp) or grp_dur > max_cue_s:
                refined.extend(_split_words_by_pause(grp, pause_s, max_cue_s))
            else:
                # Terminator-carrying cues were previously exempt from pause
                # splitting entirely — but the deterministic punctuation
                # restorer appends terminators to EVERY CJK cue, which
                # disarmed the splitter on exactly the two-speaker welded
                # cues it exists for. Split these too, at the LONGER
                # turn-pause threshold only: word timestamps must show a
                # real ≥turn_pause_s silence, so well-punctuated
                # single-utterance cues are untouched.
                refined.extend(_split_words_by_pause(grp, turn_pause_s, max_cue_s))
        sentences = refined

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


def timing_provenance_report(segments: list) -> dict:
    """Summarise how each segment's timing was derived: from per-word
    timestamps (accurate) vs. char-proportional fallback (a uniform-rate guess).

    A segment counts as word-timed when it carries a non-empty ``words`` array;
    otherwise its start/end were distributed by character length. Surfacing this
    once per job makes timing-quality regressions (e.g. polish wiping word
    timing) visible without digging through the transcript."""
    total = 0
    word_timed = 0
    for s in segments or []:
        words = (s.get("words") if isinstance(s, dict)
                 else getattr(s, "words", None))
        text = (s.get("text") if isinstance(s, dict)
                else getattr(s, "text", "")) or ""
        if not text.strip():
            continue
        total += 1
        if words:
            word_timed += 1
    proportional = total - word_timed
    pct = (word_timed / total * 100.0) if total else 100.0
    return {
        "total": total,
        "word_timed": word_timed,
        "proportional": proportional,
        "pct_word_timed": round(pct, 1),
    }


def _trim_boundary_overlap(prev_text: str, next_text: str) -> tuple[str, int]:
    """Drop text from the head of ``next_text`` that duplicates the tail of
    ``prev_text``.

    Whisper's overlapping decode windows re-emit the boundary words in BOTH
    neighbouring cues; merging the cues verbatim then doubles that text
    ("端っこから食べれるうん端っこから食べれるうん"). Finds the longest
    suffix of ``prev_text`` that is also a prefix of ``next_text`` (minimum 4
    characters, capped at 60 so a pathological cue can't go quadratic; Latin
    overlaps must end on a word boundary) and returns
    ``(trimmed_next_text, overlap_chars)``. A short echo ("はい" after "はい")
    is below the minimum and survives — this only removes decode-window
    duplication, not genuine repetition.
    """
    a = (prev_text or "").strip()
    b = (next_text or "").strip()
    if not a or not b:
        return b, 0
    max_n = min(len(a), len(b), 60)
    for n in range(max_n, 3, -1):
        if a[-n:] != b[:n]:
            continue
        rest = b[n:]
        # Latin scripts: never split a word — the char after the overlap (and
        # the char before it in prev) must be a boundary.
        if rest and rest[0].isalnum() and b[n - 1].isalnum() \
                and not _is_cjk(b[:n]):
            continue
        return rest.lstrip(), n
    return b, 0


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

    # 1. Merge adjacent same-speaker segments (never across speakers, and
    #    never across a turn-length silence — a real ≥turn-pause gap between
    #    same-speaker cues is a natural cue boundary; welding across it is
    #    what produced the incoherent multi-utterance lines).
    _, _, _, _turn_pause_s = _pause_split_params()
    from backend.config import settings
    _max_merge_s = float(getattr(settings, "SENTENCE_MERGE_MAX_CUE_S", 12.0))
    _max_merge_chars = int(getattr(settings, "SENTENCE_MERGE_MAX_CHARS", 280))
    merged: list[TranscriptSegment] = []
    for s in segs:
        _gap_ok = (not merged
                   or (float(s.start or 0) - float(merged[-1].end or 0))
                   < _turn_pause_s)
        # Never merge past a duration/length ceiling: the re-split below relies
        # on sentence terminators, and unpunctuated ASR (raw Japanese cues
        # before polish) can't be re-split — an uncapped same-speaker chain
        # with <0.7 s gaps produced 30-40 s paragraph cues that survived all
        # the way into the export (950 → 488 pre-translate collapse). Capping
        # the merge keeps worst-case cues subtitle-sized even when the
        # sentence splitter has nothing to split on.
        if merged and _gap_ok and (merged[-1].speaker or "") == (s.speaker or ""):
            _cand_dur = float(s.end or 0) - float(merged[-1].start or 0)
            _cand_len = len((merged[-1].text or "")) + len((s.text or ""))
            if _cand_dur > _max_merge_s or _cand_len > _max_merge_chars:
                merged.append(s)
                continue
        if merged and _gap_ok and (merged[-1].speaker or "") == (s.speaker or ""):
            prev = merged[-1]
            is_cjk = _is_cjk((prev.text or "") + (s.text or ""))
            joiner = "" if is_cjk else " "
            # Whisper's overlapping decode windows repeat the boundary text in
            # both cues — trim the duplicate before welding, or the merged cue
            # ships it twice.
            _s_text, _overlap = _trim_boundary_overlap(prev.text or "", s.text or "")
            new_text = ((prev.text or "").strip() + joiner + _s_text).strip()
            # Carry words ONLY when both sides have them: a merged block with
            # a PARTIAL word array would rebuild its text exclusively from
            # the words, silently dropping the words-less side's text. After an
            # overlap trim the word array no longer matches the text — drop it
            # rather than let a words-based rebuild resurrect the duplicate.
            new_words = ((prev.words or []) + (s.words or [])
                         if (prev.words and s.words and not _overlap) else None)
            merged[-1] = _new_segment(
                prev, prev.start, s.end, new_text, new_words)
        else:
            merged.append(s)

    # 2. Re-split each merged block by sentence.
    out: list[TranscriptSegment] = []
    for s in merged:
        out.extend(_split_segment_by_sentence(s))
    return out
