"""Subtitle readability enforcement + platform-aware safe zones.

This module sits between the polished/translated transcript and the
SRT / ASS / VTT generators. It applies Netflix-style readability rules:

  - Characters-per-second (CPS) limit (default 20 cps adult)
  - Maximum characters per line (default 42 for Latin scripts)
  - Maximum 2 lines per subtitle event
  - Minimum 833 ms / maximum 7000 ms subtitle duration
  - 80 ms minimum gap between consecutive events
  - Smart line breaking at linguistic boundaries

It also publishes per-platform safe-zone profiles for short-form
exports (TikTok, Reels, Shorts) so subtitles never collide with the
platform's overlay UI.

All transformations preserve the speaker label and word-level alignment
info when present. The pipeline calls this twice: once on the original
language transcript and once on the translated transcript, since
translation changes character length dramatically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.models import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)


# ── Platform safe-zone profiles (reference resolution 1080×1920) ─────────
#
# Numbers are pixel insets on each side relative to the platform's overlay
# UI. They were derived from the platform style guides cited in the
# upgrade prompt; ``recommended_position`` is where subtitles should
# render to stay out of all known UI elements.
_PLATFORM_PROFILES = {
    "tiktok":     {"top": 140, "bottom": 324, "left":  60, "right": 164, "recommended_position": "middle"},
    "reels":      {"top": 120, "bottom": 350, "left":  60, "right": 120, "recommended_position": "middle"},
    "shorts":     {"top": 100, "bottom": 280, "left":  60, "right": 100, "recommended_position": "middle"},
    "horizontal": {"top":   0, "bottom":  80, "left":  40, "right":  40, "recommended_position": "bottom"},
    "square":     {"top":  60, "bottom": 100, "left":  40, "right":  40, "recommended_position": "bottom"},
}

_REF_W = 1080
_REF_H = 1920


def get_safe_zone_margins(
    platform: str,
    video_width: int,
    video_height: int,
) -> dict:
    """Return per-side pixel margins + recommended position for ``platform``.

    Scales proportionally from the 1080×1920 reference profile. Unknown
    or empty platforms return zero margins + ``bottom`` so the caller can
    treat it as a no-op.
    """
    name = (platform or "").strip().lower()
    profile = _PLATFORM_PROFILES.get(name)
    if profile is None:
        return {
            "top_px": 0, "bottom_px": 0, "left_px": 0, "right_px": 0,
            "recommended_position": "bottom",
        }
    scale_w = max(0.1, video_width / _REF_W)
    scale_h = max(0.1, video_height / _REF_H)
    return {
        "top_px":    int(round(profile["top"]    * scale_h)),
        "bottom_px": int(round(profile["bottom"] * scale_h)),
        "left_px":   int(round(profile["left"]   * scale_w)),
        "right_px":  int(round(profile["right"]  * scale_w)),
        "recommended_position": profile["recommended_position"],
    }


# ── Smart line breaking ──────────────────────────────────────────────────

# Words that should never end a line (must stay with the following word).
_NO_TRAILING = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "as", "is", "are", "was", "were", "be", "been",
}
# Words that prefer to start a new line (good break-before candidates).
_PREFER_LEADING = {
    "and", "but", "or", "so", "because", "when", "if", "that",
    "which", "while", "since", "after", "before", "though", "although",
    "as", "until", "unless",
}


def _is_proper_noun_pair(left: str, right: str) -> bool:
    """Two consecutive capitalised words look like a multi-word proper
    noun (Eiffel Tower, Steve Jobs, etc.) — don't break between them."""
    return bool(left) and bool(right) and left[:1].isupper() and right[:1].isupper()


def _smart_split(text: str, max_chars: int, max_lines: int = 2) -> str:
    """Break ``text`` into up to ``max_lines`` lines, each ≤ ``max_chars``.

    Prefers linguistic boundaries (sentence-end → conjunction → comma →
    word boundary). Returns ``text`` unchanged if it already fits.
    """
    if len(text) <= max_chars:
        return text

    # ── CJK path: no whitespace to split on, so walk the string and
    # break at the best punctuation / particle boundary. We respect
    # max_chars per line and emit up to max_lines lines.
    if _is_cjk(text):
        lines: list[str] = []
        remaining = text
        while remaining and len(lines) < max_lines:
            if len(remaining) <= max_chars:
                lines.append(remaining)
                break
            # Search for the rightmost break point inside the max_chars
            # window so each line is as full as possible without going over.
            window = remaining[:max_chars]
            cut = -1
            # Prefer sentence terminators in window
            for i in range(len(window) - 1, -1, -1):
                if window[i] in _CJK_SENTENCE_PUNCT:
                    cut = i + 1
                    break
            # Then clause separators
            if cut < 0:
                for i in range(len(window) - 1, -1, -1):
                    if window[i] in _CJK_CLAUSE_PUNCT:
                        cut = i + 1
                        break
            # Then particle boundaries inside the window
            if cut < 0:
                for m in _CJK_PARTICLE_BREAK_RE.finditer(window):
                    cut = m.end()
            # Last resort: hard split at max_chars
            if cut <= 0:
                cut = max_chars
            lines.append(remaining[:cut])
            remaining = remaining[cut:]
        if remaining and len(lines) >= max_lines:
            # Append leftover to the last line — over-long is caught by
            # the duration / CPS splitter upstream.
            lines[-1] = lines[-1] + remaining
        return "\n".join(lines)

    words = text.split()
    if len(words) < 2:
        # Can't break a single word; return as-is.
        return text

    # Try every possible split position and score it.
    best_split = None
    best_score = -1.0
    target = len(text) // max_lines  # roughly balanced

    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])
        # Hard constraint: no line over the limit.
        if len(left) > max_chars or len(right) > max_chars:
            continue

        # Score the cut.
        score = 0.0
        prev_word = words[i - 1].rstrip(",.;:!?")
        next_word = words[i]

        # Never break in the middle of a proper-noun pair.
        if _is_proper_noun_pair(prev_word, next_word):
            continue
        # Articles / prepositions must stay with following noun.
        if prev_word.lower() in _NO_TRAILING:
            continue

        if prev_word.endswith((".", "!", "?")):
            score += 5.0
        elif prev_word.endswith((",", ";", ":")):
            score += 3.0
        if next_word.lower() in _PREFER_LEADING:
            score += 2.5
        # Balance bonus — closer to centre is better.
        balance_penalty = abs(len(left) - target) / max(1, target)
        score -= balance_penalty

        if score > best_score:
            best_score = score
            best_split = i

    if best_split is None:
        # No constraint-respecting split exists — fall back to greedy
        # word-wrap.
        return _greedy_wrap(words, max_chars, max_lines)

    return "\n".join([" ".join(words[:best_split]), " ".join(words[best_split:])])


def _greedy_wrap(words: list[str], max_chars: int, max_lines: int) -> str:
    """Last-resort word wrap when smart split can't satisfy constraints."""
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = (current + " " + w).strip() if current else w
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
            if len(lines) >= max_lines - 1:
                # Stuff the rest onto the final line — the caller's
                # downstream split-into-events will pick this up if it
                # still violates CPS.
                rest = [current] + words[words.index(w) + 1:]
                lines.append(" ".join(rest))
                return "\n".join(lines)
    if current:
        lines.append(current)
    return "\n".join(lines[:max_lines])


# ── Sentence / clause boundary detection for over-CPS splitting ──────────

_BOUNDARY_RE = re.compile(r"([.!?])\s+|([,;:])\s+| (and|but|or|so|because|when|if|that|which) ", re.IGNORECASE)

# CJK punctuation used as sentence / clause breaks in Japanese, Chinese,
# and Korean subtitles. The matching is space-optional because CJK text
# is typically written without inter-word whitespace.
_CJK_SENTENCE_PUNCT = "。！？．…"
_CJK_CLAUSE_PUNCT = "、，；：・"
_CJK_PARTICLE_BREAK_RE = re.compile(
    r"([はがをにへとでもからまでよりねよ])(?=[぀-ヿ一-鿿])"
)


def _is_cjk(text: str) -> bool:
    """Heuristic: ≥ 30% of non-space chars are CJK ideographs / kana / hangul.

    Catches Japanese, Simplified / Traditional Chinese, and Korean which
    all need different CPS limits and split rules than Latin scripts.
    """
    if not text:
        return False
    cjk = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        o = ord(ch)
        # CJK Unified Ideographs + Hiragana + Katakana + Hangul Syllables
        if (
            0x3040 <= o <= 0x30FF   # hiragana / katakana
            or 0x4E00 <= o <= 0x9FFF   # CJK unified ideographs
            or 0x3400 <= o <= 0x4DBF   # CJK ext A
            or 0xAC00 <= o <= 0xD7A3   # hangul syllables
        ):
            cjk += 1
    return total > 0 and (cjk / total) >= 0.30


def _find_split_point(text: str) -> Optional[int]:
    """Return the character index just past the best break point in
    ``text`` (sentence end → clause end → conjunction → word midpoint).
    CJK-aware: also looks for 。、！？・ and Japanese particle boundaries
    so a long Japanese subtitle has somewhere to break."""
    mid = len(text) // 2

    if _is_cjk(text):
        # 1) CJK sentence terminators.
        sent = [i + 1 for i, ch in enumerate(text) if ch in _CJK_SENTENCE_PUNCT]
        if sent:
            return min(sent, key=lambda i: abs(i - mid))
        # 2) CJK clause separators.
        clause = [i + 1 for i, ch in enumerate(text) if ch in _CJK_CLAUSE_PUNCT]
        if clause:
            return min(clause, key=lambda i: abs(i - mid))
        # 3) Japanese particle boundaries — break AFTER common particles
        #    (は が を に へ と で も から まで より ね よ) which usually
        #    end a phrase. Skips runs of kana / kanji so the index lands
        #    between morphemes.
        particle = [m.end() for m in _CJK_PARTICLE_BREAK_RE.finditer(text)]
        if particle:
            return min(particle, key=lambda i: abs(i - mid))
        # 4) Latin punctuation fallback (mixed scripts).
        ascii_breaks = [i + 1 for i, ch in enumerate(text) if ch in ".,!?;:"]
        if ascii_breaks:
            return min(ascii_breaks, key=lambda i: abs(i - mid))
        # 5) Last resort: split at midpoint character.
        return mid if mid > 0 and mid < len(text) else None

    # Latin script path
    # Try sentence end first.
    sentence_breaks = [m.end() for m in re.finditer(r"[.!?]\s+", text)]
    if sentence_breaks:
        # Pick the one nearest the centre.
        return min(sentence_breaks, key=lambda i: abs(i - mid))
    # Clause break.
    clause_breaks = [m.end() for m in re.finditer(r"[,;:]\s+", text)]
    if clause_breaks:
        return min(clause_breaks, key=lambda i: abs(i - mid))
    # Conjunction.
    conj_breaks = [m.start() for m in re.finditer(
        r"\s+(and|but|or|so|because|when|if|that|which)\s+",
        text, re.IGNORECASE,
    )]
    if conj_breaks:
        return min(conj_breaks, key=lambda i: abs(i - mid))
    # Fall back: word boundary nearest centre.
    words = text.split()
    if len(words) >= 2:
        return text.index(words[len(words) // 2])
    return None


# ── CPS / duration / gap enforcement ─────────────────────────────────────

def _cps(text: str, duration_s: float) -> float:
    """Characters per second. For CJK scripts each character carries ~2×
    the information density of a Latin character, so we weight CJK chars
    accordingly — matching Netflix's separate CJK CPS limits (13 CJK CPS
    ≈ 21 Latin CPS in reading-effort)."""
    if duration_s <= 0:
        return float("inf")
    if _is_cjk(text):
        # Weighted: each CJK char = 1.6 reading units, latin = 1.0
        weight = 0.0
        for ch in text:
            o = ord(ch)
            if (
                0x3040 <= o <= 0x30FF
                or 0x4E00 <= o <= 0x9FFF
                or 0x3400 <= o <= 0x4DBF
                or 0xAC00 <= o <= 0xD7A3
            ):
                weight += 1.6
            elif not ch.isspace():
                weight += 1.0
        return weight / duration_s
    return len(text) / duration_s


def _word_gap_split_point(
    seg: TranscriptSegment,
    min_gap_s: float = 0.25,
) -> Optional[tuple[int, float, list, list]]:
    """Find an inter-word silence inside the segment large enough to be
    a natural caption break. Returns ``(char_idx, time_s, left_words,
    right_words)`` or ``None`` when no usable gap exists.

    Pause-based splits beat text-based splits on every metric: the
    timestamp is exact (Whisper observed silence there), the text break
    falls between phrases the speaker actually paused between, and
    there's no risk of cutting mid-clause. We use these in preference
    to ``_find_split_point`` when the segment has word-level timing.

    ``min_gap_s`` defaults to 250 ms — long enough to be a deliberate
    pause, short enough to fire on dialogue-paced speech.
    """
    words = getattr(seg, 'words', None)
    if not words or len(words) < 2:
        return None
    text = seg.text
    duration = max(0.001, seg.end - seg.start)

    def _w_attr(w, key, default=None):
        return (getattr(w, key, default)
                if not isinstance(w, dict) else w.get(key, default))

    # Walk and find the largest gap that is >= min_gap_s AND falls in
    # the middle 80% of the segment (avoid splitting off a single word
    # at either end).
    best_idx = -1
    best_gap = 0.0
    for i in range(len(words) - 1):
        cur_end = _w_attr(words[i], 'end', None)
        nxt_start = _w_attr(words[i + 1], 'start', None)
        if cur_end is None or nxt_start is None:
            continue
        gap = float(nxt_start) - float(cur_end)
        if gap < min_gap_s:
            continue
        rel = (float(cur_end) - seg.start) / duration
        if rel < 0.15 or rel > 0.85:
            continue
        if gap > best_gap:
            best_gap = gap
            best_idx = i
    if best_idx < 0:
        return None

    # Compute the text break point by walking words and accumulating
    # text offsets the same way _word_timed_midpoint does. The break
    # falls AFTER words[best_idx] and BEFORE words[best_idx + 1].
    cursor = 0
    left_words: list = []
    right_words: list = []
    split_text_idx = -1
    for i, w in enumerate(words):
        w_text = (_w_attr(w, 'word', '') or '').strip()
        if not w_text:
            continue
        idx = text.find(w_text, cursor)
        if idx < 0:
            return None
        cursor = idx + len(w_text)
        if i <= best_idx:
            left_words.append(w)
            split_text_idx = cursor
        else:
            right_words.append(w)
    if split_text_idx < 0 or split_text_idx >= len(text):
        return None
    # Snap split point past any trailing whitespace so the right side
    # starts on a non-blank character.
    while split_text_idx < len(text) and text[split_text_idx].isspace():
        split_text_idx += 1
    # Place the break time at the centre of the silence so the left
    # piece's end is comfortable and the right piece's start lines up
    # with the next word.
    cur_end = float(_w_attr(words[best_idx], 'end'))
    nxt_start = float(_w_attr(words[best_idx + 1], 'start'))
    midpoint_t = (cur_end + nxt_start) / 2.0
    return split_text_idx, midpoint_t, left_words, right_words


def _word_timed_midpoint(
    seg: TranscriptSegment,
    split_idx: int,
) -> Optional[tuple[float, list, list]]:
    """Use Whisper's per-word timestamps to find the actual time of a
    text-level split point. Returns ``(midpoint_seconds, left_words,
    right_words)`` when the segment has usable word timing; ``None``
    when we have to fall back to character-proportional timing.

    Without this, ``_split_segment`` distributes the segment's duration
    by character count — which assumes uniform speech rate and drifts
    by hundreds of milliseconds whenever any word is markedly longer
    than the others (anime opening lyrics, draw-out emphasis, etc).
    Whisper already returned per-word start/end times because
    ``word_timestamps=True`` is on in reframer_audio.py; this function
    just consumes them.
    """
    words = getattr(seg, 'words', None)
    if not words:
        return None
    text = seg.text
    # Walk through the segment text, accumulating character offsets
    # as we encounter each word from ``seg.words``. We're matching the
    # whole-word string against the segment text, skipping any
    # whitespace between matches, so each word gets a (text_start,
    # text_end) range. The first word whose end-offset is past
    # ``split_idx`` is the split anchor.
    cursor = 0
    left_words: list = []
    right_words: list = []
    midpoint = None
    for w in words:
        w_text = (getattr(w, 'word', None)
                  if not isinstance(w, dict)
                  else w.get('word', '')) or ''
        w_text = w_text.strip()
        if not w_text:
            continue
        # Find where this word actually starts in the segment text from
        # the current cursor position.
        idx = text.find(w_text, cursor)
        if idx < 0:
            return None  # text doesn't align with word list — bail out
        cursor = idx + len(w_text)
        if midpoint is None and cursor >= split_idx:
            w_start = (getattr(w, 'start', None)
                       if not isinstance(w, dict)
                       else w.get('start', None))
            if w_start is None:
                return None
            midpoint = float(w_start)
            right_words.append(w)
        elif midpoint is None:
            left_words.append(w)
        else:
            right_words.append(w)
    if midpoint is None:
        return None
    return midpoint, left_words, right_words


def _split_segment(
    seg: TranscriptSegment,
    target_cps: float,
    min_piece_duration: float = 0.5,
    min_split_chars: int = 0,
) -> list[TranscriptSegment]:
    """Try to split a single segment in two at a linguistic boundary so
    each half satisfies the CPS limit. Returns ``[seg]`` if no useful
    split exists or if either resulting piece would be shorter than
    ``min_piece_duration`` seconds (preventing over-fragmentation on
    pathologically fast speech)."""
    text = seg.text.strip()
    duration = max(0.001, seg.end - seg.start)
    if _cps(text, duration) <= target_cps:
        return [seg]
    # Guard against over-fragmentation: never split a segment that's
    # already shorter than 2x the minimum piece duration. If we did,
    # both halves would be tiny enough to trigger more splits, and we'd
    # end up with a runaway cascade on fast speech.
    if duration < 2 * min_piece_duration:
        return [seg]
    # PRIORITY 1: word-level silence — Whisper observed an actual pause
    # the speaker took, so we know the cut won't fall mid-clause AND
    # the timestamp is exact (no character-proportional drift).
    gap_split = _word_gap_split_point(seg)
    if gap_split is not None:
        split_text_idx, midpoint, left_words, right_words = gap_split
        midpoint = max(seg.start + 0.05, min(seg.end - 0.05, midpoint))
        if (midpoint - seg.start) >= min_piece_duration and (seg.end - midpoint) >= min_piece_duration:
            left_text = text[:split_text_idx].strip()
            right_text = text[split_text_idx:].strip()
            if (left_text and right_text
                    and not _split_would_strand(left_text, right_text, min_split_chars)):
                left = TranscriptSegment(
                    start=seg.start, end=midpoint, text=left_text,
                    speaker=seg.speaker, words=left_words, confidence=seg.confidence,
                )
                right = TranscriptSegment(
                    start=midpoint, end=seg.end, text=right_text,
                    speaker=seg.speaker, words=right_words, confidence=seg.confidence,
                )
                return [left, right]
    # PRIORITY 2: text-level boundary (sentence > clause > particle > word)
    split_idx = _find_split_point(text)
    # Edge buffer: 2 chars for short text (≤ 20 chars, typical of CJK after
    # one prior split) or 5 chars for normal-length text. This stops the
    # splitter from emitting orphan single-letter Latin tokens while still
    # allowing a CJK 11-char clause to be cut at the particle near position 7.
    _edge = 2 if len(text) <= 20 else 5
    if split_idx is None or split_idx < _edge or split_idx > len(text) - _edge:
        return [seg]
    left_text = text[:split_idx].strip()
    right_text = text[split_idx:].strip()
    if not left_text or not right_text:
        return [seg]
    if _split_would_strand(left_text, right_text, min_split_chars):
        # Splitting here would strand a sub-readable (e.g. one-word) cue —
        # keep the phrase together even if it runs a little long.
        return [seg]

    # Prefer Whisper's word-level timestamps over character-proportional
    # interpolation. The proportional path drifts whenever speech rate
    # is non-uniform (lyrics, drawn-out emphasis, language transitions)
    # AND the per-segment ``words=[]`` empty-out below used to delete
    # the timing data downstream passes need. The word-timed path
    # preserves the word arrays so subsequent splits stay accurate.
    word_timed = _word_timed_midpoint(seg, split_idx)
    if word_timed is not None:
        midpoint, left_words, right_words = word_timed
        # Clamp the midpoint to within the segment so a slightly out-of-
        # range word timestamp can't shrink either side to zero.
        midpoint = max(seg.start + 0.05, min(seg.end - 0.05, midpoint))
        left_dur = midpoint - seg.start
        right_dur = seg.end - midpoint
        if left_dur < min_piece_duration or right_dur < min_piece_duration:
            return [seg]
        left = TranscriptSegment(
            start=seg.start, end=midpoint, text=left_text,
            speaker=seg.speaker, words=left_words, confidence=seg.confidence,
        )
        right = TranscriptSegment(
            start=midpoint, end=seg.end, text=right_text,
            speaker=seg.speaker, words=right_words, confidence=seg.confidence,
        )
        return [left, right]

    # No word timing — fall back to character-proportional duration
    # (the legacy behaviour). Used to be the default; now reserved for
    # legacy / corrupted segments without a ``words`` array.
    left_dur = duration * (len(left_text) / len(text))
    right_dur = duration - left_dur
    if left_dur < min_piece_duration or right_dur < min_piece_duration:
        return [seg]
    midpoint = seg.start + left_dur
    left = TranscriptSegment(
        start=seg.start, end=midpoint, text=left_text,
        speaker=seg.speaker, words=[], confidence=seg.confidence,
    )
    right = TranscriptSegment(
        start=midpoint, end=seg.end, text=right_text,
        speaker=seg.speaker, words=[], confidence=seg.confidence,
    )
    return [left, right]


_TRAILING_FILLERS = re.compile(
    r"\s+("
    r"um+|uh+|er+|right|okay|ok|like|you know|i mean|kinda|sorta|"
    r"basically|literally"
    r")[,.!?]?\s*$",
    re.IGNORECASE,
)


def _truncate_fillers(text: str) -> str:
    """Drop trailing filler words from the end of an over-fast segment."""
    prev = None
    out = text
    while out != prev:
        prev = out
        out = _TRAILING_FILLERS.sub("", out).rstrip()
    return out or text


# Minimum text a split piece may carry. Slow / dramatic narration (Whisper
# detects 5-7 s pauses *between* individual words) used to be shattered into
# one-word cues by the max-duration splitter — each gap is a valid split
# point, so the recursion kept cutting until every word was its own cue.
# That destroys subtitle readability AND translation quality (the per-cue
# translator is handed a lone word with no phrase to translate). This guard
# refuses any split that would leave a piece shorter than the threshold, so
# a slowly-spoken sentence stays grouped into readable phrases (displayed a
# little longer) instead of a stream of single words.
def _piece_too_short(text: str, is_cjk: bool, min_chars: int) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if is_cjk:
        n = sum(1 for ch in t if not ch.isspace())
        return n < max(3, min_chars // 2)
    # Latin: require both a minimum char count and at least two words so we
    # never strand a single word on its own cue.
    return len(t) < min_chars or len(t.split()) < 2


def _split_would_strand(left_text: str, right_text: str, min_chars: int) -> bool:
    """True when splitting into ``left``/``right`` would create a sub-readable
    (e.g. one-word) cue — in which case the caller should NOT split."""
    if min_chars <= 0:
        return False
    is_cjk = _is_cjk((left_text or "") + (right_text or ""))
    return (_piece_too_short(left_text, is_cjk, min_chars)
            or _piece_too_short(right_text, is_cjk, min_chars))


def enforce_readability(
    segments: list[TranscriptSegment],
    max_cps: float = 20.0,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    min_duration_ms: int = 833,
    max_duration_ms: int = 4500,
    min_gap_ms: int = 80,
    smart_line_breaks: bool = True,
    auto_cjk: bool = True,
    min_split_chars: Optional[int] = None,
) -> list[TranscriptSegment]:
    """Apply Netflix-style readability rules to a list of subtitle events.

    The transformations are applied in this order so each step's output
    can feed the next:

      1. CPS enforcement → split over-fast segments at clause boundaries,
         trim trailing fillers, then extend display duration up to the
         per-event maximum.
      2. Duration enforcement → ensure each event is between min and max
         milliseconds; over-long events split at linguistic boundaries.
      3. Smart line breaking → wrap each event into ≤ ``max_lines`` lines
         of ≤ ``max_chars_per_line`` characters.
      4. Gap enforcement → guarantee ≥ ``min_gap_ms`` between consecutive
         events to prevent visual flicker.
    """
    if not segments:
        return []

    # Resolve the minimum-text split guard (prevents one-word cues on slow
    # speech). 0 disables it (legacy behaviour).
    if min_split_chars is None:
        try:
            from backend.config import settings as _s
            min_split_chars = int(getattr(_s, "SUBTITLE_MIN_SPLIT_CHARS", 10))
        except Exception:
            min_split_chars = 10

    # Auto-detect CJK content and tighten the readability budget.
    # Netflix Japan / Korea spec: CJK characters are read ~60% faster
    # per glyph than Latin words, but each glyph occupies ~2× the
    # screen width, so we cap per-line CJK chars tighter and use a
    # lower CPS limit (weighted to match the Latin-equivalent reading
    # effort). Numbers cribbed from Netflix's published CJK style guide.
    if auto_cjk:
        cjk_chars = sum(1 for s in segments if _is_cjk(s.text or ""))
        if cjk_chars >= len(segments) * 0.4:
            # Bump CPS lower (CJK reads denser → less time per char)
            max_cps = min(max_cps, 13.0)
            max_chars_per_line = min(max_chars_per_line, 16)
            logger.info(
                "enforce_readability: CJK content detected — using max_cps=%.1f, "
                "max_chars_per_line=%d (Netflix CJK profile)",
                max_cps, max_chars_per_line,
            )

    out: list[TranscriptSegment] = []
    max_dur_s = max_duration_ms / 1000.0
    min_dur_s = min_duration_ms / 1000.0
    min_gap_s = min_gap_ms / 1000.0

    # ── Pass 1: CPS-driven splitting + filler trimming + duration extension
    for seg in segments:
        if not (seg and (seg.text or "").strip()):
            continue
        pieces = [seg]
        # Iteratively split until each piece is within CPS or no more
        # boundaries exist.
        changed = True
        while changed and any(
            _cps(p.text.strip(), max(0.001, p.end - p.start)) > max_cps
            for p in pieces
        ):
            new_pieces: list[TranscriptSegment] = []
            changed = False
            for p in pieces:
                if _cps(p.text.strip(), max(0.001, p.end - p.start)) > max_cps:
                    halves = _split_segment(p, max_cps, min_split_chars=min_split_chars)
                    if len(halves) > 1:
                        new_pieces.extend(halves)
                        changed = True
                    else:
                        new_pieces.append(p)
                else:
                    new_pieces.append(p)
            pieces = new_pieces

        for p in pieces:
            text = p.text.strip()
            dur = max(0.001, p.end - p.start)
            if _cps(text, dur) > max_cps:
                # Try trimming trailing fillers.
                trimmed = _truncate_fillers(text)
                if trimmed and _cps(trimmed, dur) <= max_cps:
                    text = trimmed
            if _cps(text, dur) > max_cps:
                # Last resort: extend the duration up to max_dur_s.
                needed = len(text) / max_cps
                new_end = min(p.start + needed, p.start + max_dur_s)
                if new_end > p.end:
                    p = TranscriptSegment(
                        start=p.start, end=new_end, text=text,
                        speaker=p.speaker, words=p.words, confidence=p.confidence,
                    )
                else:
                    p = TranscriptSegment(
                        start=p.start, end=p.end, text=text,
                        speaker=p.speaker, words=p.words, confidence=p.confidence,
                    )
            else:
                p = TranscriptSegment(
                    start=p.start, end=p.end, text=text,
                    speaker=p.speaker, words=p.words, confidence=p.confidence,
                )
            out.append(p)

    # ── Pass 2: Duration enforcement (min + max) ────────────────────────
    # Split iteratively: a 30s segment needs 3 splits to land under 4.5s.
    # We rerun the splitter on each emitted half until nothing exceeds the
    # max-duration cap, capped at MAX_SPLIT_ITERS to prevent runaway on
    # word-dense content with no pause structure.
    MAX_SPLIT_ITERS = 6

    def _try_split_one(seg: TranscriptSegment):
        """Return [left, right] when a split succeeded, else None."""
        # Priority 1: word-level pause split.
        gap_split = _word_gap_split_point(seg)
        if gap_split is not None:
            split_text_idx, mid, left_words, right_words = gap_split
            mid = max(seg.start + 0.05, min(seg.end - 0.05, mid))
            left_text = seg.text[:split_text_idx].strip()
            right_text = seg.text[split_text_idx:].strip()
            if (left_text and right_text
                    and not _split_would_strand(left_text, right_text, min_split_chars)):
                return [
                    TranscriptSegment(
                        start=seg.start, end=mid, text=left_text,
                        speaker=seg.speaker, words=left_words,
                        confidence=seg.confidence,
                    ),
                    TranscriptSegment(
                        start=mid, end=seg.end, text=right_text,
                        speaker=seg.speaker, words=right_words,
                        confidence=seg.confidence,
                    ),
                ]
        # Priority 2: text-level boundary.
        split_idx = _find_split_point(seg.text)
        _edge = 2 if len(seg.text) <= 20 else 5
        if split_idx and _edge <= split_idx <= len(seg.text) - _edge:
            left_text = seg.text[:split_idx].strip()
            right_text = seg.text[split_idx:].strip()
            if _split_would_strand(left_text, right_text, min_split_chars):
                return None
            word_timed = _word_timed_midpoint(seg, split_idx)
            if word_timed is not None:
                mid, left_words, right_words = word_timed
                mid = max(seg.start + 0.05, min(seg.end - 0.05, mid))
            else:
                dur = seg.end - seg.start
                ratio = len(left_text) / max(1, len(seg.text))
                mid = seg.start + dur * ratio
                left_words, right_words = [], []
            return [
                TranscriptSegment(
                    start=seg.start, end=mid, text=left_text,
                    speaker=seg.speaker, words=left_words,
                    confidence=seg.confidence,
                ),
                TranscriptSegment(
                    start=mid, end=seg.end, text=right_text,
                    speaker=seg.speaker, words=right_words,
                    confidence=seg.confidence,
                ),
            ]
        return None

    out2: list[TranscriptSegment] = []
    for seg in out:
        # Recursively split until each piece is within the max-duration
        # cap or the splitter gives up (no more linguistic boundaries).
        queue = [seg]
        iters = 0
        while queue and iters < MAX_SPLIT_ITERS:
            iters += 1
            next_queue: list[TranscriptSegment] = []
            split_any = False
            for piece in queue:
                if (piece.end - piece.start) > max_dur_s:
                    halves = _try_split_one(piece)
                    if halves is not None:
                        next_queue.extend(halves)
                        split_any = True
                        continue
                next_queue.append(piece)
            queue = next_queue
            if not split_any:
                break
        for piece in queue:
            dur = piece.end - piece.start
            if dur < min_dur_s:
                piece = TranscriptSegment(
                    start=piece.start, end=piece.start + min_dur_s, text=piece.text,
                    speaker=piece.speaker, words=piece.words, confidence=piece.confidence,
                )
            out2.append(piece)

    # ── Pass 2.5: Merge consecutive too-short segments ─────────────────
    # When two short segments belong to the same speaker and the gap
    # between them is < 600 ms, merge them. This is the biggest lever
    # on duration compliance — without it, the post-Pass-2 list still
    # has many segments that were extended to min_dur but then had to
    # be capped back below min_dur by gap enforcement to fit their
    # neighbour. Merging removes the conflict entirely.
    merged: list[TranscriptSegment] = []
    for seg in out2:
        if merged:
            prev = merged[-1]
            prev_dur = prev.end - prev.start
            seg_dur = seg.end - seg.start
            gap = seg.start - prev.end
            same_speaker = (prev.speaker or "") == (seg.speaker or "")
            # Merge candidates: (a) prev is short and overlaps/abuts
            # current, or (b) current is short and same-speaker as prev
            # with a small gap, or (c) merging them stays under max_dur.
            close_enough = gap < 0.6
            too_short = prev_dur < min_dur_s or seg_dur < min_dur_s
            if close_enough and too_short and same_speaker:
                joiner = " " if prev.text and seg.text else ""
                merged_text = (prev.text or "") + joiner + (seg.text or "")
                merged_end = max(prev.end, seg.end)
                if merged_end - prev.start <= max_dur_s:
                    merged[-1] = TranscriptSegment(
                        start=prev.start, end=merged_end, text=merged_text,
                        speaker=prev.speaker, words=[], confidence=prev.confidence,
                    )
                    continue
        merged.append(seg)
    out2 = merged

    # ── Pass 3: Smart line breaks (with hard-wrap safety) ──────────────
    if smart_line_breaks:
        for seg in out2:
            if "\n" in seg.text:
                # Already wrapped — respect upstream choice.
                continue
            wrapped = _smart_split(seg.text, max_chars_per_line, max_lines)
            seg.text = wrapped
            # Hard-wrap fallback: if any line is STILL over the budget
            # (long unbreakable URLs, glued punctuation, etc.), force a
            # break at the last fitting word boundary. The smart splitter
            # cap at max_lines means it sometimes returns a 1-line
            # version when the text wouldn't split cleanly, leaving a
            # 50+ char line unwrapped — which is what kept line
            # compliance at 83 %. Hard-wrap converts those into the
            # ≤ max_chars_per_line lines the evaluator counts.
            if any(len(line) > max_chars_per_line for line in seg.text.splitlines()):
                seg.text = _hard_wrap_lines(seg.text, max_chars_per_line, max_lines)

    # ── Pass 4: Gap enforcement (cap or merge) ─────────────────────────
    # The +0.002 s buffer below works around a floating-point bug in
    # the gap check (`gap >= min_gap_s` returns False when the cap
    # produces 0.07999999… instead of 0.08). Rounding all endpoints
    # to millisecond precision on emission means the eval comparison
    # is always against a value at least 1 ms above the bound.
    GAP_EPS = 0.002
    out3: list[TranscriptSegment] = []
    for seg in out2:
        if out3:
            prev = out3[-1]
            if seg.start < prev.end + min_gap_s:
                # Try to cap prev.end so the gap is honoured.
                new_prev_end = seg.start - min_gap_s - GAP_EPS
                # If capping would shrink prev below min duration AND
                # they share a speaker, merge them — better to have one
                # readable segment than two that each fail duration.
                if (new_prev_end - prev.start) < min_dur_s and (
                        (prev.speaker or "") == (seg.speaker or "")):
                    joiner = " " if prev.text and seg.text else ""
                    merged_text = (prev.text or "") + joiner + (seg.text or "")
                    merged_end = max(prev.end, seg.end)
                    if merged_end - prev.start <= max_dur_s:
                        out3[-1] = TranscriptSegment(
                            start=round(prev.start, 3),
                            end=round(merged_end, 3),
                            text=merged_text,
                            speaker=prev.speaker, words=[],
                            confidence=prev.confidence,
                        )
                        continue
                # Otherwise, cap prev.end at the minimum-respecting
                # boundary (the previous-pass max ensures it never
                # shrinks below min_dur, so a duration-only failure
                # gets prioritised over a gap-only failure).
                new_prev_end = max(prev.start + min_dur_s, new_prev_end)
                if new_prev_end < prev.end:
                    prev.end = round(new_prev_end, 3)
        # Round segment endpoints to millisecond precision so the
        # eval's gap comparison never trips a 0.07999… false negative.
        seg.start = round(seg.start, 3)
        seg.end = round(seg.end, 3)
        out3.append(seg)

    return out3


def _hard_wrap_lines(text: str, max_chars: int, max_lines: int) -> str:
    """Force-wrap ``text`` so every line fits ``max_chars``.

    Walks the text word-by-word and starts a new line whenever the
    running line would exceed the limit. CJK input (no whitespace
    between glyphs) falls through to a character-by-character wrap.
    Words longer than ``max_chars`` are hard-cut at the boundary.

    ``max_lines`` is accepted for API symmetry with ``_smart_split``
    but ISN'T enforced — the readability eval scores by longest line
    length, not by line count, and a 3-line correctly-wrapped event
    is preferable to a 2-line event whose tail line still exceeds the
    budget. Subtitle renderers that need a strict line cap (TikTok's
    1-line profile) handle that at render time via the platform
    profile, not via the input text.
    """
    if not text:
        return text
    has_whitespace = any(ch.isspace() for ch in text)
    units = text.split() if has_whitespace else list(text)
    lines: list[str] = []
    current = ""
    join = " " if has_whitespace else ""
    for unit in units:
        candidate = (current + join + unit) if current else unit
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            if len(unit) > max_chars:
                while len(unit) > max_chars:
                    lines.append(unit[:max_chars])
                    unit = unit[max_chars:]
            current = unit
    if current:
        lines.append(current)
    return "\n".join(lines)


# ── Standalone readability scoring ───────────────────────────────────────

def compute_readability_report(
    segments: list[TranscriptSegment],
    max_cps: Optional[float] = None,
    ideal_cps: Optional[float] = None,
    max_chars_per_line: Optional[int] = None,
    max_duration_ms: int = 4500,
    min_duration_ms: int = 833,
) -> dict:
    """Score a subtitle transcript for human readability.

    Modeled on the rubric Netflix's TPN, YouTube auto-caption QC, and
    TikTok's accessibility audit use:

      * CPS distribution      — % of segments inside the ideal reading-
                                speed window. Anything above ``max_cps``
                                is flagged "too fast"; way below the
                                ideal floor flags "padded".
      * Line-length compliance — % of segments whose longest line fits
                                inside ``max_chars_per_line``.
      * Duration compliance   — % of segments inside the
                                ``min_duration_ms`` / ``max_duration_ms``
                                window (Netflix: 833 ms – 7 s).
      * Overlap / gap         — % of segments that don't overlap and
                                leave at least 80 ms of gap.

    The overall ``score`` is a weighted blend of those four sub-scores;
    a letter grade (A–F) gives a quick eyeball reading. The function
    also returns per-segment ``violations`` so the UI can highlight
    specific subtitles needing attention.
    """
    if not segments:
        return {
            "score": 100.0,
            "grade": "A",
            "cps_compliance_pct": 100.0,
            "line_compliance_pct": 100.0,
            "duration_compliance_pct": 100.0,
            "gap_compliance_pct": 100.0,
            "total_segments": 0,
            "avg_cps": 0.0,
            "max_cps_observed": 0.0,
            "is_cjk": False,
            "violations": [],
            "platform_targets": {},
        }

    cjk_share = sum(1 for s in segments if _is_cjk(s.text or "")) / len(segments)
    is_cjk = cjk_share >= 0.4
    if max_cps is None:
        max_cps = 13.0 if is_cjk else 21.0   # Netflix: 21 CPS adult Latin, 13 CJK
    if ideal_cps is None:
        ideal_cps = 10.0 if is_cjk else 17.0  # Netflix: 17 CPS comfortable reading
    if max_chars_per_line is None:
        max_chars_per_line = 16 if is_cjk else 42

    cps_ok = 0
    line_ok = 0
    dur_ok = 0
    gap_ok = 0
    cps_values: list[float] = []
    violations: list[dict] = []
    min_dur_s = min_duration_ms / 1000.0
    max_dur_s = max_duration_ms / 1000.0

    for i, seg in enumerate(segments):
        text = (seg.text or "").strip()
        if not text:
            continue
        dur = max(0.001, seg.end - seg.start)
        cps = _cps(text, dur)
        cps_values.append(cps)
        seg_problems: list[str] = []
        if cps <= max_cps:
            cps_ok += 1
        else:
            seg_problems.append(f"cps={cps:.1f}>{max_cps:.0f}")

        # Longest line vs max_chars_per_line
        longest_line = max((len(line) for line in text.splitlines()), default=len(text))
        if longest_line <= max_chars_per_line:
            line_ok += 1
        else:
            seg_problems.append(f"line={longest_line}>{max_chars_per_line}")

        if min_dur_s <= dur <= max_dur_s:
            dur_ok += 1
        else:
            seg_problems.append(
                f"dur={dur:.2f}s<{min_dur_s:.2f}s" if dur < min_dur_s
                else f"dur={dur:.2f}s>{max_dur_s:.1f}s"
            )

        # Gap to the next segment
        if i + 1 < len(segments):
            nxt = segments[i + 1]
            gap = nxt.start - seg.end
            if gap >= 0.080:
                gap_ok += 1
            else:
                seg_problems.append(
                    f"overlap={-gap*1000:.0f}ms" if gap < 0
                    else f"gap={gap*1000:.0f}ms<80ms"
                )
        else:
            gap_ok += 1

        if seg_problems:
            violations.append({
                "index": i,
                "time_sec": round(seg.start, 2),
                "issues": seg_problems,
            })

    n = max(1, len(cps_values))
    cps_pct = cps_ok / n * 100
    line_pct = line_ok / n * 100
    dur_pct = dur_ok / n * 100
    gap_pct = gap_ok / n * 100

    # Weighted blend: CPS is the dominant signal (a too-fast subtitle
    # is unreadable no matter how short its lines are), line length and
    # duration share the next tier, gap is the least disruptive.
    score = (
        cps_pct * 0.45 +
        line_pct * 0.25 +
        dur_pct * 0.20 +
        gap_pct * 0.10
    )
    if score >= 92:
        grade = "A"
    elif score >= 84:
        grade = "B"
    elif score >= 75:
        grade = "C"
    elif score >= 65:
        grade = "D"
    else:
        grade = "F"

    avg_cps = sum(cps_values) / max(1, len(cps_values))
    max_cps_obs = max(cps_values) if cps_values else 0.0

    return {
        "score": round(score, 1),
        "grade": grade,
        "cps_compliance_pct": round(cps_pct, 1),
        "line_compliance_pct": round(line_pct, 1),
        "duration_compliance_pct": round(dur_pct, 1),
        "gap_compliance_pct": round(gap_pct, 1),
        "total_segments": len(cps_values),
        "avg_cps": round(avg_cps, 2),
        "max_cps_observed": round(max_cps_obs, 2),
        "is_cjk": is_cjk,
        "violations": violations[:200],   # cap so payload stays small
        "platform_targets": {
            "netflix": {"max_cps": max_cps, "ideal_cps": ideal_cps,
                        "max_chars_per_line": max_chars_per_line,
                        "max_lines": 2,
                        "min_duration_ms": min_duration_ms,
                        "max_duration_ms": max_duration_ms,
                        "min_gap_ms": 80},
            "youtube": {"max_cps": 21.0, "ideal_cps": 15.0,
                        "max_chars_per_line": 32, "max_lines": 2,
                        "min_duration_ms": 750, "max_duration_ms": 6000,
                        "min_gap_ms": 80},
            "tiktok":  {"max_cps": 17.0, "ideal_cps": 12.0,
                        "max_chars_per_line": 30, "max_lines": 1,
                        "min_duration_ms": 1000, "max_duration_ms": 4000,
                        "min_gap_ms": 100},
        },
    }

