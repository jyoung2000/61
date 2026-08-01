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


def _is_marker(text: str) -> bool:
    """A bracketed non-speech caption (``[♪ music ♪]``, ``[applause]``). These
    must never receive per-word timing — karaoke-highlighting a music glyph or
    stage direction is wrong — so both fill tiers skip them."""
    t = (text or "").strip()
    return t.startswith("[") and t.endswith("]")


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


def distribute_cue_window(tokens: list[str], start: float, end: float) -> list[dict]:
    """Char-weight distribution of a cue's [start,end] window across its
    DISPLAYED tokens — the deterministic, translation-correct highlight model.

    Each token gets a share of the window proportional to its length (longer
    words read longer), so the highlight always advances left-to-right inside
    the cue's own audio-aligned span. This is what the export fallback and the
    frontend fallback compute too, so the three stay in lock-step. Returns
    ``{word,start,end}`` dicts 1:1 with ``tokens``."""
    n = len(tokens)
    if n == 0:
        return []
    dur = max(0.0, end - start)
    total = sum(max(1, len(t)) for t in tokens)
    out, cur = [], start
    for i, t in enumerate(tokens):
        share = (max(1, len(t)) / total) if total else (1.0 / n)
        w_end = end if i == n - 1 else min(end, cur + dur * share)
        out.append({"word": t, "start": round(cur, 3), "end": round(w_end, 3)})
        cur = w_end
    return out


def snap_cue_windows_to_reference(
    cues: list,
    whisper_en_words: list,
    *,
    pad_threshold: float = 0.4,
    max_shift: float = 1.5,
    lead_in: float = 0.1,
    min_gap: float = 0.12,
    min_dur_s: float = 0.833,
    max_cps: float = 17.0,
) -> int:
    """Pull each cue's DISPLAY [start,end] toward the real speech the Whisper-EN
    reference shows, so the subtitle appears/disappears WITH the words instead of
    on the padded source (Japanese) sentence window.

    INWARD-ONLY: the start may only move LATER (trim leading padding) and the end
    only EARLIER (trim trailing padding) — never past the real onset/offset, never
    beyond the source edges. Because both edges only move inward, the operation
    can never create an overlap or a new merge/split (the gap to each neighbour can
    only grow), so it is cue-count- and readability-neutral by construction. Only
    trims padding larger than ``pad_threshold``; caps each edge's movement at
    ``max_shift``; keeps a ``lead_in`` before the real onset; and refuses any trim
    that would drop the visible duration below ``min_dur_s`` or below the text's
    reading time at ``max_cps`` (so it never manufactures a duration/CPS
    violation). Bracket markers and word-less/empty cues are skipped. Mutates
    ``cues`` in place; returns the number of cues whose window changed."""
    n_snapped = 0
    # Does a later readability pass extend a too-fast cue's END into the idle time
    # before the next cue? If so, the trim below can be bounded by that reachable
    # end rather than by the cue's current one (see the `room_end` note).
    try:
        from backend.config import settings as _xs
        _extend_after = bool(getattr(_xs, "SUBTITLE_EXTEND_BEFORE_SPLIT", True))
    except Exception:
        _extend_after = True

    cue_list = list(cues or [])
    for i, cue in enumerate(cue_list):
        text = _cue_text(cue)
        if _is_marker(text) or not text.strip():
            continue
        s0 = _cue_attr(cue, "start", None)
        e0 = _cue_attr(cue, "end", None)
        if s0 is None or e0 is None:
            continue
        s0, e0 = float(s0), float(e0)
        if e0 - s0 <= 0:
            continue
        prev_end = None
        if i > 0:
            _pe = _cue_attr(cue_list[i - 1], "end", None)
            prev_end = float(_pe) if _pe is not None else None
        next_start = None
        if i + 1 < len(cue_list):
            _ns = _cue_attr(cue_list[i + 1], "start", None)
            next_start = float(_ns) if _ns is not None else None
        ref = _ref_words_in_window(whisper_en_words, s0, e0)
        if not ref:
            continue
        # Drop the PREVIOUS speaker's tail. _ref_words_in_window selects by
        # overlap, so a word that begins before this cue and merely spills across
        # its start becomes ref[0] — making onset <= s0, so the leading-pad test
        # below is false and the whole in-trim silently no-ops. That is the common
        # case on back-to-back dialogue (over half the cues in a real export
        # touched their neighbour), i.e. it defeated the snap exactly where it was
        # needed. A word starting before s0 whose bulk lies in the previous cue's
        # span is that cue's, not this one's.
        if prev_end is not None:
            ref = [(ws, we) for (ws, we) in ref
                   if not (ws < s0
                           and (min(we, prev_end) - ws) >= 0.5 * max(1e-6, we - ws))]
            if not ref:
                continue
        onset, offset = ref[0][0], ref[-1][1]
        new_s, new_e = s0, e0
        # IN: trim leading padding (move start later, bounded, never past onset).
        if onset - s0 > pad_threshold:
            new_s = min(onset - lead_in, s0 + max_shift)
            new_s = max(new_s, s0)               # inward only
        # OUT: trim trailing padding (move end earlier, bounded, never past offset).
        if e0 - offset > pad_threshold:
            new_e = max(offset + lead_in, e0 - max_shift)
            new_e = min(new_e, e0)               # inward only
        # Never leave less than the min duration or the text's reading time.
        # The budget is measured against the end this cue can actually REACH, not
        # its current end: the readability pass that runs after this one extends a
        # too-fast cue into the idle time before the next cue (and never moves a
        # start back). Bounding by e0 instead made every cue already sitting at the
        # CPS budget an unconditional no-op however much leading padding it had —
        # which, with cues deliberately kept whole well above the strict CPS cap,
        # silently excluded a whole band of them.
        required = max(min_dur_s, len(text.strip()) / max_cps if max_cps > 0 else 0.0)
        room_end = e0
        if _extend_after and next_start is not None:
            room_end = max(e0, next_start - max(min_gap, 0.0))
        if room_end - new_s < required:
            new_e = min(e0, new_s + required)
            if new_e - new_s < required:
                new_s = max(s0, new_e - required)
        if abs(new_s - s0) < 1e-3 and abs(new_e - e0) < 1e-3:
            continue
        new_s, new_e = round(new_s, 3), round(new_e, 3)
        if new_e <= new_s:
            continue
        # Re-clamp existing word timings into the corrected window (monotonic).
        words = _cue_attr(cue, "words", None)
        if words:
            clamped = []
            prev = new_s
            for w in words:
                ws = float(_word_attr(w, "start", new_s) or new_s)
                we = float(_word_attr(w, "end", ws) or ws)
                ws = max(new_s, min(new_e, ws))
                we = max(ws, min(new_e, we))
                if ws < prev:
                    ws = prev
                if we < ws:
                    we = ws
                clamped.append({"word": _word_attr(w, "word", ""),
                                "start": round(ws, 3), "end": round(we, 3)})
                prev = we
            _set_words(cue, clamped)
        if isinstance(cue, dict):
            cue["start"], cue["end"] = new_s, new_e
        else:
            cue.start, cue.end = new_s, new_e
        n_snapped += 1
    return n_snapped


def _ref_words_in_window(whisper_en_words: list, lo: float, hi: float) -> list:
    """Whisper-EN words whose span overlaps ``[lo, hi]``, time-ordered.

    These are the REAL speech onsets/offsets inside the cue — the audio the
    highlight should track — even when their *text* never lexically matched the
    LLM tokens (tier A's requirement)."""
    out = []
    for w in whisper_en_words or []:
        try:
            s = float(w["start"])
            e = float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e >= lo and s <= hi:
            out.append((s, e))
    out.sort()
    return out


def distribute_over_reference(
    tokens: list[str], ref_spans: list, lo: float, hi: float
) -> list[dict]:
    """Place ``tokens`` on the REAL speech timeline given by ``ref_spans``
    (Whisper-EN word ``(start, end)`` pairs overlapping the cue window).

    Whisper-EN already decoded the whole audio, so its word onsets mark where
    speech actually is inside ``[lo, hi]``. Instead of sweeping the highlight at
    a uniform char-rate across the padded window (silence and all), lay the LLM
    tokens down by char-weight but map each token boundary onto the reference
    onset grid — so the highlight starts at real speech onset, ends at real
    speech offset, and a genuine inter-word pause lands between the right two
    tokens. Both translations are the SAME English utterance in the SAME order,
    so token-position ≈ reference-position is a sound anchor (unlike the old JA
    source-word mapping, which had a different word order). Times are clamped to
    ``[lo, hi]`` and monotonic. Falls back to :func:`distribute_cue_window` when
    the reference is unusable. Returns ``{word,start,end}`` dicts 1:1 with
    ``tokens``."""
    n = len(tokens)
    if n == 0:
        return []
    spans = [(s, e) for (s, e) in (ref_spans or []) if e >= s]
    if not spans:
        return distribute_cue_window(tokens, lo, hi)
    ref_lo = max(lo, spans[0][0])
    ref_hi = min(hi, max(e for _, e in spans))
    if ref_hi <= ref_lo:
        return distribute_cue_window(tokens, lo, hi)
    # Content→time anchors: the reference word onsets inside the span, plus the
    # closing offset so the last token reaches real speech end. Evenly spaced in
    # "content" (word index), then interpolated by the tokens' char-weight
    # cumulative fraction.
    anchors = sorted({s for s, _ in spans if ref_lo <= s <= ref_hi} | {ref_lo, ref_hi})
    m = len(anchors)
    weights = [max(1, len(t)) for t in tokens]
    total = float(sum(weights)) or 1.0
    cum = [0.0]
    for wgt in weights:
        cum.append(cum[-1] + wgt / total)

    def _map(frac: float) -> float:
        frac = min(1.0, max(0.0, frac))
        if m <= 1:
            return ref_lo + (ref_hi - ref_lo) * frac
        pos = frac * (m - 1)
        k = int(pos)
        if k >= m - 1:
            return anchors[-1]
        t = pos - k
        return anchors[k] * (1.0 - t) + anchors[k + 1] * t

    out, prev_end = [], ref_lo
    for i, tok in enumerate(tokens):
        s = max(prev_end, min(hi, _map(cum[i])))
        e = max(s, min(hi, _map(cum[i + 1])))
        if i == n - 1:
            e = max(s, min(hi, ref_hi))
        out.append({"word": tok, "start": round(s, 3), "end": round(e, 3)})
        prev_end = e
    return out


def attach_source_pause_timings(
    llm_cues: list,
    source_cues: list,
    *,
    min_tokens: int = 2,
    whisper_en_words: Optional[list] = None,
) -> tuple[list, int]:
    """Tier B: give still-word-less LLM cues a highlight skeleton.

    When ``whisper_en_words`` (the flat, time-ordered Whisper-EN stream) is
    supplied and a cue's window overlaps real reference words, the LLM tokens are
    placed on that REAL voiced timeline (:func:`distribute_over_reference`) so the
    highlight tracks actual speech onset/offset even without a lexical match —
    the audio-locked path that turns most tier-B cues into real timing. When no
    reference word overlaps (or none was supplied), it falls back to
    char-weight-distributing the cue's own [start,end] window across its English
    tokens — which still advances the highlight left-to-right at a uniform rate
    and matches the export + preview fallback exactly.

    (The very old Tier B mapped each English token onto the JAPANESE source word
    at the same POSITION — but Japanese is SOV with a different word count/order,
    so the real audio pauses landed under the wrong English words. The Whisper-EN
    reference is already in English order, so position-based anchoring is sound.)
    ``source_cues`` is kept in the signature for call-site compatibility but no
    longer used. Only fills cues that don't already have words."""
    _ = source_cues  # retained for API stability; no longer consulted
    n_attached = 0
    for cue in (llm_cues or []):
        if _cue_has_words(cue) or _is_marker(_cue_text(cue)):
            continue
        en_tokens = _tokenize(_cue_text(cue))
        if len(en_tokens) < min_tokens:
            continue
        start = _cue_attr(cue, "start", None)
        end = _cue_attr(cue, "end", None)
        if start is None or end is None or float(end) <= float(start):
            continue
        lo, hi = float(start), float(end)
        ref = _ref_words_in_window(whisper_en_words, lo, hi) if whisper_en_words else []
        if ref:
            _set_words(cue, distribute_over_reference(en_tokens, ref, lo, hi))
        else:
            _set_words(cue, distribute_cue_window(en_tokens, lo, hi))
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
    """Run the A→B→C ladder over ``llm_cues`` (mutating them) and report which
    tier each cue ended on. Tier A projects real Whisper-EN audio times onto
    lexically-matched cues. Tier B/C give the rest a timing skeleton — placed on
    the REAL Whisper-EN voiced timeline (speech onset/offset + pauses) when a
    reference word overlaps the cue and ``HYBRID_REF_TIME_ANCHOR`` is on, else
    char-weight-distributed across the cue's own window (B = multi-token cues,
    C = whatever remains, e.g. single-token cues). No dialogue cue ships
    word-less; bracketed non-speech markers are never filled. Returns a summary
    dict ``{"tier_a", "tier_b", "tier_c", "ref_anchored", "total"}`` for
    one-line logging (``ref_anchored`` = B/C cues that landed on real audio)."""
    total = len(llm_cues or [])
    whisper_words = flatten_whisper_words(whisper_en_segments)
    try:
        from backend.config import settings as _s
        _anchor = bool(getattr(_s, "HYBRID_REF_TIME_ANCHOR", True))
    except Exception:
        _anchor = True
    _ref_stream = whisper_words if _anchor else None
    _, n_a = project_word_timings(
        llm_cues, whisper_words, margin_s=margin_s, min_anchor_ratio=min_anchor_ratio)
    # Snapshot, among the cues tier A left word-less (i.e. the tier-B/C pool),
    # how many overlap a real Whisper-EN word — the ones B/C will place on the
    # audio timeline rather than the uniform window fallback.
    ref_anchored = 0
    if _ref_stream:
        for cue in (llm_cues or []):
            if _cue_has_words(cue) or _is_marker(_cue_text(cue)):
                continue
            s = _cue_attr(cue, "start", None)
            e = _cue_attr(cue, "end", None)
            if s is None or e is None or float(e) <= float(s):
                continue
            if _ref_words_in_window(_ref_stream, float(s), float(e)):
                ref_anchored += 1
    _, n_b = attach_source_pause_timings(
        llm_cues, source_cues, whisper_en_words=_ref_stream)
    # Tier C: fill ANY remaining word-less cue (single-token, or an edge the
    # tiers above skipped) — on the real reference timeline when a Whisper-EN
    # word overlaps its window, else by char-weight-distributing the window — so
    # NO translated cue ships word-less and highlighting always has a real,
    # cue-aligned skeleton (word-less cues were the systematically mis-aligned
    # case: preview/export both fell back to a pure synthetic estimate).
    n_c = 0
    for cue in (llm_cues or []):
        if _cue_has_words(cue) or _is_marker(_cue_text(cue)):
            continue
        toks = _tokenize(_cue_text(cue))
        s = _cue_attr(cue, "start", None)
        e = _cue_attr(cue, "end", None)
        if not toks or s is None or e is None or float(e) <= float(s):
            continue
        lo, hi = float(s), float(e)
        ref = _ref_words_in_window(_ref_stream, lo, hi) if _ref_stream else []
        if ref:
            _set_words(cue, distribute_over_reference(toks, ref, lo, hi))
        else:
            _set_words(cue, distribute_cue_window(toks, lo, hi))
        n_c += 1
    # Cue-onset snap (final step, gated OFF by default): pull each cue's DISPLAY
    # window toward the real Whisper-EN speech so the subtitle appears/disappears
    # WITH the words, cutting the per-cue start variance. Uses the FULL Whisper-EN
    # stream (independent of HYBRID_REF_TIME_ANCHOR) and only when enabled.
    n_snapped = 0
    try:
        from backend.config import settings as _s3
        _snap_on = bool(getattr(_s3, "HYBRID_CUE_SNAP_ENABLED", False))
    except Exception:
        _s3, _snap_on = None, False
    if _snap_on and whisper_words:
        try:
            n_snapped = snap_cue_windows_to_reference(
                llm_cues, whisper_words,
                pad_threshold=float(getattr(_s3, "HYBRID_CUE_SNAP_PAD_THRESHOLD_S", 0.4)),
                max_shift=float(getattr(_s3, "HYBRID_CUE_SNAP_MAX_SHIFT_S", 1.5)),
                lead_in=float(getattr(_s3, "HYBRID_CUE_SNAP_LEAD_IN_S", 0.1)),
                min_gap=float(getattr(_s3, "HYBRID_CUE_SNAP_MIN_GAP_S", 0.12)),
                min_dur_s=float(getattr(_s3, "SUBTITLE_MIN_DURATION_MS", 833)) / 1000.0,
                max_cps=float(getattr(_s3, "SUBTITLE_MAX_CPS", 17.0)),
            )
        except Exception:
            n_snapped = 0
    return {"tier_a": n_a, "tier_b": n_b, "tier_c": n_c,
            "ref_anchored": ref_anchored, "cue_snapped": n_snapped, "total": total}


def restore_translation_windows(translated: list, source: list,
                                max_drift_s: float = 15.0) -> dict:
    """1:1 window attestation at the translation boundary.

    A translated cue's time window is INHERITED from its source cue — no
    stage between translation and formatting may legally rewrite it. A
    measured run reached the formatter with ~48 cues whose windows had
    collapsed; the formatter packed them at 0:00 over silence and the
    persist-time voice gate then had to kill them, losing the LINES along
    with the phantoms. Restoring the source window instead keeps each line
    at its true audio position.

    Mutates ``translated`` in place wherever a cue's window is degenerate
    (< 0.05 s wide) or drifted more than ``max_drift_s`` from its source
    cue's window, restoring the source window and clearing any word rows
    built for the wrong window (tier C keeps the cue whole downstream).
    Only applies when the lists are 1:1 (equal length) — every other shape
    is returned untouched.

    Returns ``{"restored": n, "samples": [...], "source_degenerate": m}``;
    ``source_degenerate`` counts cues whose SOURCE window is itself
    unusable (nothing to restore from — corruption is upstream of
    translation, worth its own log line at the call site)."""
    out = {"restored": 0, "samples": [], "source_degenerate": 0}
    if not translated or not source or len(translated) != len(source):
        return out
    for t, s in zip(translated, source):
        try:
            sa = float(_cue_attr(s, "start", 0.0) or 0.0)
            sb = float(_cue_attr(s, "end", 0.0) or 0.0)
            ta = float(_cue_attr(t, "start", 0.0) or 0.0)
            tb = float(_cue_attr(t, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if sb - sa < 0.05:
            out["source_degenerate"] += 1
            continue                      # no source truth to restore from
        degenerate = (tb - ta) < 0.05
        drifted = (abs(ta - sa) > max_drift_s or abs(tb - sb) > max_drift_s)
        if not degenerate and not drifted:
            continue
        if isinstance(t, dict):
            t["start"], t["end"] = sa, sb
            if t.get("words"):
                t["words"] = None
            if "words_synthetic" in t:
                t["words_synthetic"] = None
        else:
            t.start, t.end = sa, sb
            if getattr(t, "words", None):
                t.words = None
            if hasattr(t, "words_synthetic"):
                t.words_synthetic = None
        out["restored"] += 1
        if len(out["samples"]) < 6:
            out["samples"].append(
                f"{ta:.2f}-{tb:.2f}s -> {sa:.2f}-{sb:.2f}s "
                f"{_cue_text(t)[:32]!r}")
    return out


def _is_anchored(cue) -> bool:
    """True when a cue's times were MEASURED against audio — a decode or a CTC
    alignment attached real word rows. Synthetic (char-weight) word arrays are
    projections, not measurements, and do not anchor anything."""
    return bool(_cue_attr(cue, "words", None)) and not _cue_attr(
        cue, "words_synthetic", None)


def enforce_anchor_brackets(cues: list, min_room_s: float = 0.4) -> dict:
    """Pull un-anchored cues back inside the audio-anchored cues around them.

    Timing that came from a projection tier rather than a measurement can
    drift, and a measured run's drift was one-sided: 20 cues more than a
    second EARLY against 1 late, gathered into four windows covering 12% of
    the runtime. Whole RUNS of consecutive cues drifted together — 228 of 311
    cues had CTC word times and held their place; the other 83 kept projected
    times and slid, one run leading its audio by nine seconds.

    A cue with measured word rows is an ANCHOR. Any run of un-anchored cues
    between two anchors must lie inside that bracket: if it starts before the
    left anchor ends, or ends after the right anchor starts, the run is
    re-timed across the bracket in proportion to its cues' text lengths — the
    same char-weight model the projection tiers use, but pinned at both ends
    so the error cannot accumulate. A run already inside its bracket is left
    untouched: this repairs violations, it does not re-time the world.

    Runs before the first anchor or after the last have only one side to pin
    and are left alone. Mutates in place; returns ``{"runs": n, "cues": m,
    "samples": [...]}``."""
    out = {"runs": 0, "cues": 0, "samples": []}
    n = len(cues or [])
    if n < 3:
        return out
    anchors = [i for i in range(n) if _is_anchored(cues[i])]
    if len(anchors) < 2:
        return out

    def _num(cue, key, default=0.0):
        try:
            return float(_cue_attr(cue, key, default) or default)
        except (TypeError, ValueError):
            return default

    for a, b in zip(anchors, anchors[1:]):
        run = list(range(a + 1, b))
        if not run:
            continue
        lo = _num(cues[a], "end")
        hi = _num(cues[b], "start")
        if hi - lo < min_room_s:
            continue                       # no room to place anything
        first_s = _num(cues[run[0]], "start")
        last_e = _num(cues[run[-1]], "end", _num(cues[run[-1]], "start"))
        if first_s >= lo - 1e-6 and last_e <= hi + 1e-6:
            continue                       # already inside its bracket
        weights = [max(1, len(_cue_text(cues[i]).strip())) for i in run]
        total = float(sum(weights))
        span = hi - lo
        cursor = lo
        for i, w in zip(run, weights):
            piece = span * (w / total)
            new_s = round(cursor, 3)
            new_e = round(min(hi, cursor + piece), 3)
            if new_e > new_s:
                if len(out["samples"]) < 6:
                    out["samples"].append(
                        f"{_num(cues[i], 'start'):.2f}->{new_s:.2f}s "
                        f"{_cue_text(cues[i])[:28]!r}")
                if isinstance(cues[i], dict):
                    cues[i]["start"], cues[i]["end"] = new_s, new_e
                else:
                    cues[i].start, cues[i].end = new_s, new_e
                out["cues"] += 1
            cursor = new_e
        out["runs"] += 1
    return out
