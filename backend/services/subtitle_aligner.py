"""Hybrid word-timing projection for the JA→EN subtitle path.

The editorial LLM produces the authoritative English subtitle TEXT but carries
no timing. Whisper's ``translate`` task produces audio-aligned English with
per-word timestamps but weaker, sparser text. This module marries the two:

  * **Tier A — Whisper-EN projection** (``project_word_timings``): align the LLM
    cue text to the Whisper-EN word stream with a *same-language* (EN↔EN)
    monotonic match, then carry the matched Whisper word's start/end onto the
    LLM word and interpolate the rest. The Whisper text is used ONLY as a timing
    skeleton — it never enters the output (which stays 100% LLM).

  * **Tier B — source-pause projection** (``attach_source_pause_timings``): when
    no usable Whisper-EN anchors exist, fall back to the 1:1 SOURCE cue's word
    timestamps (Japanese), mapping the English tokens onto the source word
    timeline by position so the inter-word *pauses* (the real audio silences)
    land between the right English tokens. Split times then derive from real
    audio, even though the per-word text is approximate.

  * **Tier C — keep whole**: a cue that gets neither stays word-less and the
    readability splitter is told to leave it intact (never char-proportionally
    re-time a word-less cue — that scrambled timing on the LLM path).

Both projection functions return ``(cues, n_projected)`` and attach
``WordTimestamp`` objects whose ``.word`` text is the LLM's own tokens, so the
existing word-pause splitter in ``subtitle_formatter`` (which matches word
surfaces against the cue text) works unchanged. Monotonic, clamped to the cue
span, idempotent (a cue that already has words is left untouched).
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Optional

from backend.models import TranscriptSegment, WordTimestamp

logger = logging.getLogger(__name__)

# Tokens for matching: alphanumerics only, lowercased (apostrophes folded out so
# "don't"/"dont" align). Punctuation is dropped for the match but the original
# surface is kept for the emitted word text.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _norm(token: str) -> str:
    return "".join(_WORD_RE.findall((token or "").lower()))


def _cue_attr(cue, key, default=None):
    if isinstance(cue, dict):
        return cue.get(key, default)
    return getattr(cue, key, default)


def _word_attr(w, key, default=None):
    if isinstance(w, dict):
        return w.get(key, default)
    return getattr(w, key, default)


def _cue_text(cue) -> str:
    return (_cue_attr(cue, "text", "") or "")


def _cue_has_words(cue) -> bool:
    return bool(_cue_attr(cue, "words", None))


def _set_words(cue, words: list):
    """Attach ``words`` to a cue (mutating it). Pydantic doesn't validate on
    assignment, so build the proper type rather than leaving dicts behind."""
    typed = [
        w if isinstance(w, WordTimestamp)
        else WordTimestamp(start=float(w["start"]), end=float(w["end"]), word=w["word"])
        for w in words
    ]
    if isinstance(cue, dict):
        cue["words"] = [w.model_dump() for w in typed]
    else:
        cue.words = typed


def flatten_whisper_words(whisper_en_segments) -> list[dict]:
    """Flatten Whisper-EN segments' ``.words`` into one time-ordered stream of
    ``{"word", "start", "end"}`` dicts. Tolerant of dict / object shapes and of
    a segment-level ``words`` that may be None."""
    stream: list[dict] = []
    for seg in (whisper_en_segments or []):
        for w in (_cue_attr(seg, "words", None) or []):
            surf = (_word_attr(w, "word", "") or "").strip()
            st = _word_attr(w, "start", None)
            en = _word_attr(w, "end", None)
            if surf and st is not None and en is not None:
                stream.append({"word": surf, "start": float(st), "end": float(en)})
    stream.sort(key=lambda d: (d["start"], d["end"]))
    return stream


def _tokenize(text: str) -> list[str]:
    """Whitespace tokens with at least one alphanumeric char (keeps surface)."""
    return [t for t in (text or "").split() if _norm(t)]


def _interpolate_and_clamp(
    surfaces: list[str], times: list[Optional[tuple[float, float]]],
    lo: float, hi: float,
) -> list[dict]:
    """Fill ``None`` timing slots by linear interpolation between anchors, clamp
    into ``[lo, hi]``, and enforce monotonic non-negative spans. Returns a list
    of ``{word,start,end}`` dicts aligned 1:1 with ``surfaces``."""
    n = len(surfaces)
    out: list[dict] = [None] * n  # type: ignore
    # Place anchors first.
    for i, t in enumerate(times):
        if t is not None:
            out[i] = {"word": surfaces[i], "start": t[0], "end": t[1]}
    # Interpolate runs of None between anchors (or the cue edges).
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        left_t = out[i - 1]["end"] if i > 0 and out[i - 1] else lo
        right_t = out[j]["start"] if j < n and out[j] else hi
        if right_t < left_t:
            right_t = left_t
        span = right_t - left_t
        count = j - i
        for k in range(count):
            a = left_t + span * (k / (count + 1))
            b = left_t + span * ((k + 1) / (count + 1))
            out[i + k] = {"word": surfaces[i + k], "start": a, "end": b}
        i = j
    # Clamp + monotonic.
    prev_end = lo
    for w in out:
        s = max(lo, min(hi, float(w["start"])))
        e = max(s, min(hi, float(w["end"])))
        if s < prev_end:
            s = prev_end
        if e < s:
            e = s
        w["start"], w["end"] = round(s, 3), round(e, 3)
        prev_end = e
    return out


def _align_anchors(en_tokens: list[str], ref_tokens: list[str]) -> dict[int, int]:
    """Monotonic same-language alignment of ``en_tokens`` → ``ref_tokens``.

    Returns a map ``{en_index: ref_index}`` for confidently-matched tokens, using
    the equal blocks of a difflib opcode diff over the normalized tokens. Order-
    preserving by construction (difflib matches are monotonic)."""
    a = [_norm(t) for t in en_tokens]
    b = [_norm(t) for t in ref_tokens]
    matches: dict[int, int] = {}
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                matches[i1 + k] = j1 + k
    return matches


def project_word_timings(
    llm_cues: list,
    whisper_en_words: list[dict],
    *,
    margin_s: float = 2.0,
    min_anchor_ratio: float = 0.30,
) -> tuple[list, int]:
    """Tier A: project Whisper-EN word timings onto the LLM cue text.

    ``llm_cues`` carry authoritative English text + approximate ``start``/``end``
    (inherited from their source sentence) but no ``words``. ``whisper_en_words``
    is the flat, time-ordered Whisper-EN stream (see ``flatten_whisper_words``).

    For each word-less cue: gather Whisper-EN words overlapping ``[start,end]``
    ± ``margin_s``, monotonically align the cue's tokens to them, assign matched
    tokens the Whisper word's time and interpolate the rest. The cue is only
    given ``words`` when at least ``min_anchor_ratio`` of its tokens matched;
    otherwise it's left word-less for tier B/C. Idempotent: cues that already
    have words are skipped. Returns ``(llm_cues, n_projected)``."""
    if not whisper_en_words:
        return llm_cues, 0
    n_projected = 0
    for cue in (llm_cues or []):
        if _cue_has_words(cue):
            continue
        text = _cue_text(cue)
        en_tokens = _tokenize(text)
        if len(en_tokens) < 2:
            continue
        start = float(_cue_attr(cue, "start", 0.0) or 0.0)
        end = float(_cue_attr(cue, "end", start) or start)
        lo, hi = start - margin_s, end + margin_s
        cand = [w for w in whisper_en_words if w["end"] >= lo and w["start"] <= hi]
        if len(cand) < 1:
            continue
        ref_tokens = [w["word"] for w in cand]
        matches = _align_anchors(en_tokens, ref_tokens)
        if len(matches) < max(1, int(round(min_anchor_ratio * len(en_tokens)))):
            continue
        times: list[Optional[tuple[float, float]]] = []
        for i in range(len(en_tokens)):
            if i in matches:
                w = cand[matches[i]]
                times.append((w["start"], w["end"]))
            else:
                times.append(None)
        # Interpolate within the projected anchor span (clamp to the cue itself).
        words = _interpolate_and_clamp(en_tokens, times, start, end)
        _set_words(cue, words)
        n_projected += 1
    return llm_cues, n_projected


def attach_source_pause_timings(
    llm_cues: list,
    source_cues: list,
    *,
    min_tokens: int = 2,
) -> tuple[list, int]:
    """Tier B: give still-word-less LLM cues a timing skeleton from the 1:1
    SOURCE cue's word timestamps.

    The LLM cues are 1:1 with the source sentences (translate_via_llm preserves
    order + count). Map each English token onto the source word timeline by
    position so the source's inter-word *pauses* fall between the right English
    tokens — the split times then come from real audio silence even though the
    English per-word text is approximate. Only fills cues that don't already have
    words. Returns ``(llm_cues, n_attached)``."""
    if not source_cues or len(source_cues) != len(llm_cues):
        return llm_cues, 0
    n_attached = 0
    for cue, src in zip(llm_cues, source_cues):
        if _cue_has_words(cue):
            continue
        en_tokens = _tokenize(_cue_text(cue))
        if len(en_tokens) < min_tokens:
            continue
        src_words = [
            (float(_word_attr(w, "start")), float(_word_attr(w, "end")))
            for w in (_cue_attr(src, "words", None) or [])
            if _word_attr(w, "start", None) is not None
            and _word_attr(w, "end", None) is not None
        ]
        if len(src_words) < 2:
            continue
        start = float(_cue_attr(cue, "start", src_words[0][0]) or src_words[0][0])
        end = float(_cue_attr(cue, "end", src_words[-1][1]) or src_words[-1][1])
        n, m = len(en_tokens), len(src_words)
        times: list[Optional[tuple[float, float]]] = []
        for i in range(n):
            frac = (i / (n - 1)) if n > 1 else 0.0
            j = min(m - 1, max(0, round(frac * (m - 1))))
            times.append((src_words[j][0], src_words[j][1]))
        words = _interpolate_and_clamp(en_tokens, times, start, end)
        _set_words(cue, words)
        n_attached += 1
    return llm_cues, n_attached


def project_hybrid_timings(
    llm_cues: list,
    whisper_en_segments=None,
    source_cues=None,
    *,
    margin_s: float = 2.0,
    min_anchor_ratio: float = 0.30,
) -> dict:
    """Run the A→B ladder over ``llm_cues`` (mutating them) and report which tier
    each cue ended on. Tier C cues are simply left word-less. Returns a summary
    dict ``{"tier_a", "tier_b", "tier_c", "total"}`` for one-line logging."""
    total = len(llm_cues or [])
    whisper_words = flatten_whisper_words(whisper_en_segments)
    _, n_a = project_word_timings(
        llm_cues, whisper_words, margin_s=margin_s, min_anchor_ratio=min_anchor_ratio)
    n_b = 0
    if source_cues is not None:
        _, n_b = attach_source_pause_timings(llm_cues, source_cues)
    n_c = sum(1 for c in (llm_cues or []) if not _cue_has_words(c))
    return {"tier_a": n_a, "tier_b": n_b, "tier_c": n_c, "total": total}
