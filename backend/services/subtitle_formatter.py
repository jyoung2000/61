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


def _split_segment(
    seg: TranscriptSegment,
    target_cps: float,
    min_piece_duration: float = 0.5,
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
    split_idx = _find_split_point(text)
    if split_idx is None or split_idx < 5 or split_idx > len(text) - 5:
        return [seg]
    left_text = text[:split_idx].strip()
    right_text = text[split_idx:].strip()
    if not left_text or not right_text:
        return [seg]
    # Split duration proportionally to character count.
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


def enforce_readability(
    segments: list[TranscriptSegment],
    max_cps: float = 20.0,
    max_chars_per_line: int = 42,
    max_lines: int = 2,
    min_duration_ms: int = 833,
    max_duration_ms: int = 7000,
    min_gap_ms: int = 80,
    smart_line_breaks: bool = True,
    auto_cjk: bool = True,
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
                    halves = _split_segment(p, max_cps)
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
    out2: list[TranscriptSegment] = []
    for seg in out:
        dur = seg.end - seg.start
        if dur > max_dur_s:
            # Try to split.
            split_idx = _find_split_point(seg.text)
            if split_idx and 5 < split_idx < len(seg.text) - 5:
                left_text = seg.text[:split_idx].strip()
                right_text = seg.text[split_idx:].strip()
                ratio = len(left_text) / max(1, len(seg.text))
                mid = seg.start + dur * ratio
                out2.append(TranscriptSegment(
                    start=seg.start, end=mid, text=left_text,
                    speaker=seg.speaker, words=[], confidence=seg.confidence,
                ))
                out2.append(TranscriptSegment(
                    start=mid, end=seg.end, text=right_text,
                    speaker=seg.speaker, words=[], confidence=seg.confidence,
                ))
                continue
        if dur < min_dur_s:
            seg = TranscriptSegment(
                start=seg.start, end=seg.start + min_dur_s, text=seg.text,
                speaker=seg.speaker, words=seg.words, confidence=seg.confidence,
            )
        out2.append(seg)

    # ── Pass 3: Smart line breaks ───────────────────────────────────────
    if smart_line_breaks:
        for seg in out2:
            if "\n" in seg.text:
                # Already wrapped — respect upstream choice.
                continue
            wrapped = _smart_split(seg.text, max_chars_per_line, max_lines)
            if wrapped != seg.text:
                seg.text = wrapped

    # ── Pass 4: Gap enforcement (truncate overlap, then nudge by min_gap)
    out3: list[TranscriptSegment] = []
    for seg in out2:
        if out3:
            prev = out3[-1]
            if seg.start < prev.end + min_gap_s:
                # Cap prev.end so the gap is honoured. Never push prev
                # past seg.start so we don't crush the new segment.
                new_prev_end = max(prev.start + min_dur_s, seg.start - min_gap_s)
                if new_prev_end < prev.end:
                    prev.end = new_prev_end
        out3.append(seg)

    return out3


# ── Standalone readability scoring ───────────────────────────────────────

def compute_readability_report(
    segments: list[TranscriptSegment],
    max_cps: Optional[float] = None,
    ideal_cps: Optional[float] = None,
    max_chars_per_line: Optional[int] = None,
    max_duration_ms: int = 7000,
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

