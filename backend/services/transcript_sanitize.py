"""Defensive cleanup for a translated transcript corrupted by an interrupted run.

When the container restarts mid-pipeline (GPU drops, OOM) and the job resumes —
or a flaky websocket drives repeated relay saves — the stored
``translated_transcript`` can end up a UNION of the source + translated tracks
with cues duplicated many times over (observed: 584 cues, 37% still Japanese,
hot lines repeated 10×). The pipeline itself persists a clean track; this is
damage that happens AFTER, in the resume/relay path.

``sanitize_translated_transcript`` repairs that on the way out (and is applied
before persist as hygiene): it drops cues still in the source script (CJK when
the target is non-CJK — an English "translated" track must not contain Japanese)
and collapses duplication, while leaving a clean track untouched. Pure stdlib so
it stays unit-testable and cheap.

Duplication has two sources, handled together: the resume/relay UNION above
(same blob many times), AND Whisper repetition/hallucination on non-speech audio
(music, moans, silence) that emits the same line at several timestamps and is
carried verbatim through the 1:1 translation. A SUBSTANTIAL line collapses to a
single occurrence; SHORT lines (which can legitimately recur) keep a couple;
markers ("[♪ music ♪]") are never deduped.

``merge_transcript_fragments`` is a SEPARATE, opt-in pass (the sanitizer never
calls it) that folds Whisper's mid-sentence splits back into whole utterances.
Whisper segments on acoustic pauses, so one spoken sentence arrives as 2-4 cues
of a few words ("It's just the" / "number 21."). It merges a cue into the next
ONLY when the current text looks unfinished (no sentence-final punctuation) and
the two are clearly one utterance: same speaker, a small time gap, result under
a readable length/duration. It is a strict FIXED POINT (merging twice == once),
which is what makes it safe to apply once on read for a terminal job without the
"lines move around while I'm reading" churn an earlier non-idempotent re-flow
caused. Space-joining is wrong for CJK, so CJK targets are returned untouched.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_CJK_TARGETS ={"ja", "ko", "zh", "zh-cn", "zh-tw", "yue"}
# A whole cue that is only a repeated grunt letter + trailing dots ("Nn...",
# "Nnn...", "Mmm") is non-lexical mumble filler Whisper emits and the 1:1
# translation carries through — YouTube omits these. ≥2 of the same letter so
# a legitimate lone "n" survives and no real word can match.
_GRUNT_CUE_RE = re.compile(r"(?:n{2,}|m{2,})[.…!?\s]*$", re.IGNORECASE)
# Duplicate-cue policy. A SUBSTANTIAL line (a real sentence/phrase) should appear
# once: a verbatim repeat far apart is almost always Whisper repetition /
# hallucination on non-speech audio (music, moans, silence), faithfully carried
# through the 1:1 translation — the "same English line shows up at 0:14 and 3:16"
# symptom. SHORT lines ("Yes.", "Okay?", "No no") can legitimately recur, so a
# couple of copies are allowed there. Markers ("[♪ music ♪]") are never deduped —
# music genuinely plays at several points.
_MAX_DUPLICATES = 2
# A normalized cue this long (chars) is treated as substantial → collapsed to a
# single occurrence rather than the 2-copy allowance.
_SUBSTANTIAL_DEDUP_CHARS = 16
# When a SUBSTANTIAL line recurs at least this many times it isn't "a duplicate
# to trim to one" — it's a Whisper hallucination LOOP (the same full sentence
# emitted at a dozen-plus timestamps over non-speech audio: moans, music,
# silence), carried 1:1 through translation. Keeping even one places a wrong line
# at a wrong time, so drop EVERY occurrence. 6+ verbatim repeats of a real
# sentence is vanishingly rare in genuine dialogue. Observed: a 4-line block
# repeated 16× across 64:00–114:00 of a mostly-non-speech video.
_GROSS_REPEAT_DROP = 6


def _cjk_ratio(text: str) -> float:
    t = text or ""
    cjk = base = 0
    for c in t:
        if ("぀" <= c <= "ヿ") or ("㐀" <= c <= "鿿") or ("가" <= c <= "힣") or ("ｦ" <= c <= "ﾟ"):
            cjk += 1
            base += 1
        elif c.isalpha():
            base += 1
    return (cjk / base) if base else 0.0


def _get(seg, key, default=None):
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def _is_marker(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith("[") and t.endswith("]")


# ── Fragment-merge tuning ────────────────────────────────────────────────────
# A cue that does NOT end with sentence-final punctuation is treated as an
# unfinished fragment and folded into the following cue when they're clearly one
# utterance. The caps keep a merged cue subtitle-readable and bound any
# over-merge of a run of unpunctuated lines.
_FRAG_GAP_MAX_S = 2.5     # max silence between two fragments of one utterance
_FRAG_LEN_MAX = 100       # max merged characters (≈ 2 subtitle lines)
_FRAG_DUR_MAX_S = 8.0     # max merged on-screen duration
_SENTENCE_FINAL = ".!?…。！？"      # a cue ending here is a complete thought
_TRAILING_CLOSERS = "\"'”’`)]》」』"  # peel these before checking the last char


def _ends_complete(text: str) -> bool:
    """True if ``text`` ends a sentence (so it should NOT absorb the next cue).

    Trailing quotes/brackets are peeled first so ``He left."`` still reads as
    complete. A comma/semicolon/word ending is NOT complete → a continuation."""
    t = (text or "").rstrip()
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1].rstrip()
    return (not t) or (t[-1] in _SENTENCE_FINAL)


def _as_rows(segments):
    rows = []
    for r in (segments or []):
        if hasattr(r, "model_dump"):
            rows.append(r.model_dump(mode="json"))
        elif isinstance(r, dict):
            rows.append(r)
        else:
            rows.append(dict(r))
    return rows


def merge_transcript_fragments(segments, target_lang: str = "en"):
    """Fold Whisper's mid-sentence fragment splits into whole utterances.

    Returns ``(rows, changed)`` (plain dicts). A cue is extended with the
    following cue(s) only while ALL hold: the running text is unfinished (no
    sentence-final punctuation), same speaker, gap ≤ ``_FRAG_GAP_MAX_S``,
    neither side a ``[marker]``, and the result stays within the length /
    duration caps. Complete lines ("How old are you now?") never absorb the
    next line. Strict fixed point. Fail-soft: returns the input on any error."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        # Space-joining is wrong for CJK (no inter-word spaces) — leave as-is.
        if tgt in _CJK_TARGETS or len(rows) < 2:
            return rows, False

        out = []
        i = 0
        n = len(rows)
        while i < n:
            base = rows[i]
            raw = base.get("text") or ""
            # A complete or marker cue never starts a merge — pass through byte
            # for byte so unchanged cues stay identical (no spurious churn).
            if _is_marker(raw.strip()) or _ends_complete(raw):
                out.append(dict(base))
                i += 1
                continue
            norm = " ".join(raw.split())
            start = float(base.get("start") or 0.0)
            end = float(base.get("end") or start)
            spk = base.get("speaker")
            words = list(base.get("words") or [])  # keep per-word timing aligned
            j = i + 1
            absorbed = False
            while j < n and not _ends_complete(norm) and not _is_marker(norm):
                nxt = rows[j]
                nxt_text = " ".join((nxt.get("text") or "").split())
                if not nxt_text or _is_marker(nxt_text):
                    break
                if nxt.get("speaker") != spk:
                    break
                ns = float(nxt.get("start") or 0.0)
                ne = float(nxt.get("end") or ns)
                if ns - end > _FRAG_GAP_MAX_S:
                    break
                joiner = "" if norm.endswith("-") else " "
                cand = norm + joiner + nxt_text
                if len(cand) > _FRAG_LEN_MAX:
                    break
                if ne - start > _FRAG_DUR_MAX_S:
                    break
                norm, end, j, absorbed = cand, ne, j + 1, True
                words += list(nxt.get("words") or [])
            if absorbed:
                merged = dict(base)
                merged["text"], merged["start"], merged["end"] = norm, start, end
                # Span the per-word timestamps across all folded fragments so a
                # merged cue's karaoke highlight stays correct (stale partial
                # ``words`` from only the first fragment would mis-highlight).
                merged["words"] = words or None
                out.append(merged)
                i = j
            else:
                out.append(dict(base))
                i += 1
        return out, (len(out) != len(rows))
    except Exception:
        return _as_rows(segments), False


# ── Fragment repair the plain merge can't reach ──────────────────────────────
# ``merge_transcript_fragments`` only folds cues that DON'T end in sentence-final
# punctuation, so three common over-splits slip past it and land on their own
# cue — hurting readability AND giving the karaoke highlight a junk target:
#   1. a bare-punctuation cue (".", "…", "?") — nothing to read, nothing to speak;
#   2. a title abbreviation left dangling ("Mr." / "Lt." on its own, then the
#      name on the next cue) — ``_ends_complete`` sees the "." and calls it done;
#   3. an ellipsis BRIDGE where one word is split across two cues by a dramatic
#      pause ("Am Wu…" then "…Fey."), both halves reading as complete.
# This pass repairs exactly those, repartitioning ``words`` so the merged cue's
# per-word timing stays aligned. Same caps / same-speaker / gap discipline as the
# plain merge; strict fixed point; fail-soft.
_ABBREV_TRAILING = {
    "mr.", "mrs.", "ms.", "dr.", "lt.", "sgt.", "capt.", "cpt.", "col.", "gen.",
    "sr.", "jr.", "st.", "vs.", "mt.", "prof.", "rev.", "gov.", "sen.", "rep.",
    "cmdr.", "adm.", "maj.", "pvt.", "cpl.",
}
_ELLIPSIS_TAIL_RE = re.compile(r"(?:\.\.\.|…)\s*$")
_ELLIPSIS_HEAD_RE = re.compile(r"^\s*(?:\.\.\.|…)")


def _is_bare_punct(text: str) -> bool:
    """A non-empty cue with no letter/number/CJK char — pure punctuation/symbols
    (a lone ".", "…", "?", "-"). Markers are never bare punctuation."""
    t = (text or "").strip()
    if not t or _is_marker(t):
        return False
    return re.search(r"[^\W_]", t, re.UNICODE) is None


def _ends_with_abbrev(text: str) -> bool:
    t = (text or "").rstrip()
    while t and t[-1] in _TRAILING_CLOSERS:
        t = t[:-1].rstrip()
    last = t.split()[-1].lower() if t.split() else ""
    return last in _ABBREV_TRAILING


def _ellipsis_bridge(a: str, b: str) -> bool:
    return bool(_ELLIPSIS_TAIL_RE.search(a or "")) and bool(_ELLIPSIS_HEAD_RE.search(b or ""))


def repair_fragment_cues(segments, target_lang: str = "en"):
    """Repair the three over-splits the plain fragment merge leaves behind
    (bare-punctuation cue, dangling title abbreviation, ellipsis word-bridge).

    Returns ``(rows, changed)`` (plain dicts). CJK targets pass through (space
    joins are wrong there). Fail-soft: returns the input on any error."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        if tgt in _CJK_TARGETS or len(rows) < 2:
            return rows, False

        out: list = []
        i = 0
        n = len(rows)
        while i < n:
            base = rows[i]
            raw = (base.get("text") or "")
            norm = " ".join(raw.split())

            # (1) Bare-punctuation cue → drop it, folding its on-screen time into
            # the previous cue so the timeline stays gap-free (or the next, if it
            # leads the track).
            if _is_bare_punct(norm):
                this_end = float(base.get("end") or base.get("start") or 0.0)
                if out:
                    prev = out[-1]
                    prev["end"] = max(float(prev.get("end") or 0.0), this_end)
                    i += 1
                    continue
                if i + 1 < n:
                    nxt = rows[i + 1]
                    nxt = dict(nxt)
                    nxt["start"] = min(float(nxt.get("start") or 0.0),
                                       float(base.get("start") or 0.0))
                    rows[i + 1] = nxt
                    i += 1
                    continue
                # lone bare-punct cue with no neighbor — nothing to fold into
                out.append(dict(base))
                i += 1
                continue

            # (2)/(3) Absorb the next cue while the running text dangles on a title
            # abbreviation OR bridges an ellipsis-split word — same speaker, small
            # gap, within the readable length/duration caps.
            start = float(base.get("start") or 0.0)
            end = float(base.get("end") or start)
            spk = base.get("speaker")
            words = list(base.get("words") or [])
            j = i + 1
            absorbed = False
            while j < n and not _is_marker(norm):
                nxt = rows[j]
                nxt_text = " ".join((nxt.get("text") or "").split())
                if not nxt_text or _is_marker(nxt_text) or _is_bare_punct(nxt_text):
                    break
                if nxt.get("speaker") != spk:
                    break
                bridge = _ellipsis_bridge(norm, nxt_text)
                if not (_ends_with_abbrev(norm) or bridge):
                    break
                ns = float(nxt.get("start") or 0.0)
                ne = float(nxt.get("end") or ns)
                if ns - end > _FRAG_GAP_MAX_S:
                    break
                if bridge:
                    # Collapse the doubled ellipsis at the seam into one.
                    left = _ELLIPSIS_TAIL_RE.sub("…", norm)
                    right = _ELLIPSIS_HEAD_RE.sub("", nxt_text).lstrip()
                    cand = left + right
                else:
                    cand = norm + " " + nxt_text
                if len(cand) > _FRAG_LEN_MAX:
                    break
                if ne - start > _FRAG_DUR_MAX_S:
                    break
                norm, end, j, absorbed = cand, ne, j + 1, True
                words += list(nxt.get("words") or [])

            if absorbed:
                merged = dict(base)
                merged["text"], merged["start"], merged["end"] = norm, start, end
                # Rebuilt cue may no longer be 1:1 with the concatenated word list
                # (the ellipsis seam drops a token's text but not its timing); keep
                # ``words`` only when the count still matches so the karaoke branch
                # stays valid, else let the proportional fallback re-fill.
                merged["words"] = words if (words and len(words) == len(norm.split())) else None
                out.append(merged)
                i = j
            else:
                out.append(dict(base))
                i += 1
        return out, (len(out) != len(rows))
    except Exception:
        return _as_rows(segments), False


# ── Run-on cue split (YouTube-style one-thought-per-cue) ────────────────────
# Official subs put ONE sentence/thought per cue; Whisper+1:1 translation can
# cram 3-4 sentences into an 11-second cue ("The capsule has altered its
# course. Does it have suicidal tendencies? If it burns out… I suppose that's
# about right."), which reads far worse than the reference. Split such cues at
# sentence boundaries, allocating time by character share.
_RUNON_MAX_CHARS = 50          # a cue past ~one 42-char line is a run-on (fallback)
_RUNON_MIN_PIECE_S = 1.0       # never create a cue shorter than this
_RUNON_MAX_PIECES = 6          # a cue never explodes into confetti (fallback)
_RUNON_MIN_PIECE_CHARS = 16    # never orphan a sub-readable fragment onto its own cue
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"'‘“(\[]?[A-Z0-9])")
# Strong intra-sentence clause boundaries — split AFTER the punctuation, at the
# following whitespace, so tokens are never cut in half (the word-partition
# invariant stays exact). Deliberately NOT conjunction WORDS ("and"/"that"/…):
# a punctuation-only rule keeps 1:1 word partitions intact and avoids
# over-fragmenting mid-phrase.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:—–])\s+")


def _split_sentences_abbrev_safe(text: str) -> list[str]:
    """Sentence-split, but NEVER break after a title abbreviation ("Mr." / "Dr."
    / "Lt." …). ``_SENT_SPLIT_RE`` treats the period in "Mr." as a full stop and
    would orphan the title onto its own cue ("Mr." | "Darlian") — the exact
    honorific over-split seen against YouTube. Re-joins any piece whose
    predecessor ends in a known abbreviation."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    out: list[str] = []
    for p in parts:
        if out and _ends_with_abbrev(out[-1]):
            out[-1] = (out[-1] + " " + p).strip()
        else:
            out.append(p)
    return out


def _clause_units(sentence: str) -> list[str]:
    """Break one sentence into clause units at strong punctuation boundaries.
    Concatenates (with single spaces) back to the input, so token counts are
    preserved. Returns ``[sentence]`` when there is no clause boundary."""
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence) if p.strip()]
    return parts or [sentence]


def _pack_units(units: list[str], budget: int, min_chars: int) -> list[str]:
    """Greedily pack clause units into ≤ ``budget``-char pieces, never closing a
    piece while it is still shorter than ``min_chars`` (so a 7-char clause like
    "M Plan," is absorbed into its neighbour instead of orphaned onto its own
    cue). A single over-budget unit becomes its own piece."""
    pieces: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) >= min_chars and (len(cur) + 1 + len(u) > budget):
            pieces.append(cur)
            cur = u
        else:
            cur = (cur + " " + u).strip()
    if cur:
        if pieces and len(cur) < min_chars:
            pieces[-1] = (pieces[-1] + " " + cur).strip()
        else:
            pieces.append(cur)
    return pieces


# ── Missing-terminator repair (translator dropped the sentence period) ──────
# The 1:1 LLM translation frequently welds two English sentences into one cue
# WITHOUT the period between them ("feelings in the air tonight Holding your
# wet shoulder", "our objective, Erase everything"). With no terminator the
# run-on splitter below sees a single "sentence" and leaves the weld intact, so
# the shipped cue reads as two thoughts crammed together — the exact opposite of
# YouTube's one-thought-per-cue pacing. The signature of such a weld is precise:
# a COMMON (lowercase-initial) word, ending in a letter or comma, immediately
# followed by a Title-cased word that starts a fresh sentence. We restore the
# dropped terminator so the splitter can do its job.
#
# Precision guards keep proper nouns mid-sentence from being mistaken for a new
# sentence (the false-positive that would manufacture wrong breaks):
#   * left word must be lower-initial — skips proper-noun PHRASES where every
#     token is capitalised ("Mobile Suit", "Earth Sphere", "After Colony").
#   * left word must not be an article / preposition / possessive / conjunction /
#     object-pronoun / title — these introduce a capitalised name, not a new
#     sentence ("a Gundam", "as Aries", "from Oz", "General Septem", "call
#     myself Trois").
#   * the capitalised word must be Title-case (upper then lower), never ALL-CAPS
#     ("OZ", "AX-GY") and never a bare "I"/"I'm" — those are not sentence starts
#     here.
#   * ≥2 words on each side, so interjections ("Hey", "Yeah") aren't orphaned
#     onto their own one-word cue.
# Terminator is ATTACHED to the preceding token (comma dropped), so the
# whitespace token count is unchanged and 1:1 word timings stay aligned.
_CAPS_LEFT_STOP = frozenset({
    # articles / determiners / demonstratives / quantifiers
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each",
    "every", "no", "another", "such", "both", "either", "neither", "one",
    "all", "most", "many", "few", "several",
    # possessives
    "my", "your", "his", "her", "its", "our", "their", "whose",
    # prepositions
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto",
    "over", "under", "about", "as", "like", "near", "off", "per", "via",
    "through", "toward", "towards", "upon", "within", "without", "against",
    "between", "among", "amongst", "across", "behind", "beside", "beyond",
    # conjunctions / continuations that expect more to follow
    "and", "or", "but", "nor", "yet", "so", "than", "then", "if", "when",
    "while", "because", "although", "though", "unless", "until", "whether",
    # linking verbs — a capital after a copula is a predicate NAME, not a new
    # sentence ("This is Duo", "I am Heero", "he was Zechs").
    "is", "am", "are", "was", "were", "be", "been", "being", "become",
    "becomes", "became", "called", "named",
    # object / reflexive pronouns (a name often follows "call myself X")
    "me", "us", "him", "them", "it", "myself", "yourself", "himself",
    "herself", "itself", "ourselves", "yourselves", "themselves",
    # titles / honorifics — a capitalised name follows, not a new sentence
    "mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord", "general",
    "captain", "colonel", "major", "lieutenant", "lt", "sergeant", "sgt",
    "commander", "admiral", "professor", "prof", "king", "queen", "prince",
    "princess", "president", "chief", "officer", "agent", "saint", "st",
})
_TERMINATORS = ".!?…"


def _looks_sentence_start(tok: str) -> bool:
    """A Title-cased token that plausibly begins a new sentence — upper-then-lower
    (so ALL-CAPS acronyms and bare ``I``/``I'm`` are excluded)."""
    if len(tok) < 2:
        return False
    # Strip a leading opening quote/bracket so '"Turn' still qualifies.
    core = tok.lstrip("\"'‘“([")
    return len(core) >= 2 and core[0].isupper() and core[1].islower()


_POSSESSIVE_RE = re.compile(r"['’]s$")


def _mid_cap_core(tok: str) -> str:
    """The bare Title-cased word inside a token (leading quotes/brackets and
    trailing punctuation/possessive stripped), lowercased — or '' if the token
    isn't Title-cased (upper-then-lower). ALL-CAPS ("OZ") and bare "I" return ''."""
    core = tok.strip("\"'‘“([)]}.,!?…;:")
    core = _POSSESSIVE_RE.sub("", core)
    if len(core) < 2 or not core[0].isupper() or not any(c.islower() for c in core[1:]):
        return ""
    return core.lower()


def collect_proper_nouns(rows) -> frozenset:
    """Proper-noun cores mined from the transcript itself, so a capitalised word
    that begins a *sentence* (common word) is told apart from a capitalised word
    that names an *entity* (proper noun) without any external dictionary.

    A word is judged a proper noun when it appears Title-cased in a NON
    sentence-initial position at least twice — recurring mid-sentence capitals
    ("on Earth", "Oz's fleet", "Mobile Suit") are names; a one-off sentence-start
    common word ("…tonight Holding…") is not. Fail-soft: '' set on any error."""
    try:
        from collections import Counter
        midcap: Counter = Counter()
        for r in rows or []:
            txt = (r.get("text") if isinstance(r, dict) else getattr(r, "text", "")) or ""
            toks = txt.split()
            prev_term = True  # first token counts as sentence-initial
            for tok in toks:
                if not tok:
                    continue
                core = _mid_cap_core(tok)
                if core and not prev_term:
                    midcap[core] += 1
                prev_term = tok[-1] in _TERMINATORS
        return frozenset(w for w, c in midcap.items() if c >= 2)
    except Exception:
        return frozenset()


def insert_missing_sentence_breaks(text: str, proper_nouns=frozenset()) -> str:
    """Restore a dropped sentence terminator at a high-confidence weld.

    ``proper_nouns`` (lowercased cores, e.g. from :func:`collect_proper_nouns`)
    suppresses splitting before a recognised entity name — the difference between
    a real weld ("…rebellion Failed to…") and a proper-noun object of a verb
    ("…recover Rio…", "…enter Earth's…"). Returns ``text`` unchanged when no
    confident boundary is found. Pure string op; token count is preserved so
    downstream 1:1 word partitions stay valid."""
    t = (text or "").strip()
    if not t or len(t) < 12:
        return text
    toks = t.split()
    if len(toks) < 4:
        return text
    changed = False
    for i in range(len(toks) - 1):
        # ≥2 words on each side of the boundary between tok[i] and tok[i+1].
        if i < 1 or (len(toks) - (i + 1)) < 2:
            continue
        left, right = toks[i], toks[i + 1]
        if not left or not left[0].islower():
            continue
        # Left must end in a letter or a comma (not already terminated / a
        # clause colon-semicolon we leave alone).
        tail = left[-1]
        if tail in _TERMINATORS or tail in ":;":
            continue
        core = left.rstrip(",")
        if not core or not core[-1].isalpha():
            continue
        if core.lower() in _CAPS_LEFT_STOP:
            continue
        if not _looks_sentence_start(right):
            continue
        # The capitalised word is a recurring entity name, not a sentence start.
        if _mid_cap_core(right) in proper_nouns:
            continue
        toks[i] = core + "."
        changed = True
    return " ".join(toks) if changed else text


def split_run_on_cues(segments, target_lang: str = "en"):
    """Split run-on cues into YouTube-style one-thought-per-cue pieces.

    Returns ``(rows, changed)`` (plain dicts). A cue is split when it is a
    non-CJK, non-``[marker]`` cue at least ``2 × _RUNON_MIN_PIECE_S`` long AND
    it either runs past one subtitle line (``TRANSCRIPT_RUNON_MAX_CHARS``) or
    carries ≥ 2 finished sentences. Splitting rules:

      * **One sentence per piece** — two finished thoughts are never welded into
        one cue (this is also what keeps a second pass a no-op: each emitted
        piece is a single ≤-line sentence and re-fires nothing).
      * **Clause splitting** — a single sentence longer than one line is broken
        at strong clause punctuation (``, ; : — –``), packed back up to the
        line budget. Only done when the cue's words are 1:1 with its text (tier
        A/B) so each piece can carry exact word timings; word-less (tier C)
        cues stay sentence-only, timed char-proportionally.
      * **Timing** — piece boundaries land on the real word start when word
        timings are present (no char-proportional drift at clause cuts),
        otherwise by character share. Word timestamps are partitioned into
        their piece by token count. Every piece keeps ≥ ``_RUNON_MIN_PIECE_S``.

    Fail-soft: returns the input unchanged on any error."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        if tgt in _CJK_TARGETS or not rows:
            return rows, False
        try:
            from backend.config import settings as _s
            max_chars = int(getattr(_s, "TRANSCRIPT_RUNON_MAX_CHARS", _RUNON_MAX_CHARS))
            max_pieces = int(getattr(_s, "TRANSCRIPT_RUNON_MAX_PIECES", _RUNON_MAX_PIECES))
            clause_split = bool(getattr(_s, "TRANSCRIPT_RUNON_CLAUSE_SPLIT", True))
            caps_split = bool(getattr(_s, "TRANSCRIPT_RUNON_CAPS_SPLIT", True))
        except Exception:
            max_chars, max_pieces, clause_split = _RUNON_MAX_CHARS, _RUNON_MAX_PIECES, True
            caps_split = True
        # Mine entity names ONCE from the whole track so the weld repair below
        # never mistakes a proper-noun object ("recover Rio") for a new sentence.
        proper_nouns = collect_proper_nouns(rows) if caps_split else frozenset()
        out = []
        changed = False
        for seg in rows:
            text = (seg.get("text") or "").strip()
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
            dur = end - start
            if _is_marker(text) or dur < 2 * _RUNON_MIN_PIECE_S or not text:
                out.append(seg)
                continue
            # Restore any terminator the translator dropped between two welded
            # sentences ("…tonight Holding…"), so the split below can see both
            # thoughts. The period attaches to an existing token, so the word
            # partition stays 1:1; propagate the repair onto the row either way.
            if caps_split:
                _repaired = insert_missing_sentence_breaks(text, proper_nouns)
                if _repaired != text:
                    text = _repaired
                    seg = {**seg, "text": text}
                    changed = True
            sentences = _split_sentences_abbrev_safe(text)
            # A run-on is a cue that spills past one line OR carries ≥2 finished
            # thoughts (official subs give each its own cue regardless of length).
            if not sentences or not (len(text) > max_chars or len(sentences) >= 2):
                out.append(seg)
                continue

            words = list(seg.get("words") or [])
            token_partition = bool(words) and len(text.split()) == len(words)

            # Build pieces: one sentence per piece; a long single sentence is
            # clause-split (only when word-timed, to avoid scrambling tier-C
            # karaoke). Never weld two sentences together.
            pieces: list[str] = []
            for s in sentences:
                if clause_split and token_partition and len(s) > max_chars:
                    units = _clause_units(s)
                else:
                    units = [s]
                pieces.extend(_pack_units(units, max_chars, _RUNON_MIN_PIECE_CHARS))
            if len(pieces) < 2:
                out.append(seg)
                continue

            # Confetti cap: rein in clause-fragment explosion, but NEVER weld two
            # COMPLETE sentences — welding whole thoughts would (a) re-manufacture
            # a run-on and (b) re-split on the next pass (the welded piece carries
            # ≥2 sentence marks), breaking idempotency. So only merge a piece whose
            # LEFT is an unfinished clause fragment; when every piece is already a
            # whole sentence we stop and let each keep its own cue (one thought per
            # cue, regardless of count). The min-duration floor is handled below in
            # time allocation, so it no longer forces a sentence weld here.
            while len(pieces) > max_pieces:
                cand = [i for i in range(len(pieces) - 1) if not _ends_complete(pieces[i])]
                if not cand:
                    break
                j = min(cand, key=lambda i: len(pieces[i]) + len(pieces[i + 1]))
                pieces[j] = (pieces[j] + " " + pieces[j + 1]).strip()
                del pieces[j + 1]

            piece_tokens = [len(p.split()) for p in pieces]
            token_partition = bool(words) and sum(piece_tokens) == len(words)

            # Word-timed contiguous boundaries when 1:1 words exist: each cut
            # lands on the real start of the next piece's first word.
            bounds = None
            if token_partition:
                cand = [start]
                cum = 0
                ok = True
                for k in range(len(pieces) - 1):
                    cum += piece_tokens[k]
                    w = words[cum]
                    ws = w.get("start") if isinstance(w, dict) else getattr(w, "start", None)
                    if ws is None or ws != ws:   # `ws != ws` rejects NaN
                        ok = False
                        break
                    cand.append(min(end, max(cand[-1], float(ws))))
                if ok:
                    cand.append(end)
                    bounds = cand

            total_chars = sum(len(p) for p in pieces) or 1
            n = len(pieces)
            # Feasible per-piece floor: honor _RUNON_MIN_PIECE_S when the cue is
            # long enough to give every piece that much, else fall back to an
            # equal share (dur / n) so a dense multi-thought cue still yields
            # short-but-nonzero cues instead of a zero-length one. n·floor ≤ dur
            # by construction, so every window below is non-empty.
            floor = min(_RUNON_MIN_PIECE_S, dur / n) if n else 0.0
            w_off = 0
            t = start
            for k, p in enumerate(pieces):
                tail = n - 1 - k
                if k == n - 1:
                    p_end = end
                else:
                    if bounds is not None:
                        p_end = bounds[k + 1]
                    else:
                        p_end = t + dur * (len(p) / total_chars)
                    # Reserve the floor for every remaining piece, then honor
                    # this piece's own floor — so no cue is starved and the last
                    # piece (fixed to ``end``) still clears the floor.
                    p_end = min(p_end, end - tail * floor)
                    p_end = max(p_end, t + floor)
                p_end = min(end, max(p_end, t))
                piece_row = dict(seg)
                piece_row["text"] = p
                piece_row["start"], piece_row["end"] = round(t, 3), round(p_end, 3)
                if token_partition:
                    piece_row["words"] = words[w_off:w_off + piece_tokens[k]] or None
                    w_off += piece_tokens[k]
                else:
                    piece_row["words"] = None
                out.append(piece_row)
                t = p_end
            changed = True
        return out, changed
    except Exception:
        return _as_rows(segments), False


_THEME_OPEN_LABEL = "[♪ Opening theme ♪]"
_THEME_END_LABEL = "[♪ Ending theme ♪]"
_PREVIEW_RE = re.compile(
    r"\b(next (?:episode|time)|next,?\s+on|to be continued|preview|deathscythe)\b", re.I)
_CHORUS_MIN_CHARS = 12
_LYRIC_MAX_CHARS = 48
_THEME_MIN_SPAN_S = 12.0
# Hard ceiling on ONE theme marker's span. The reference episode's whole OP
# lyric block is 66s and its ED block 56s; a measured failure minted a
# 124-second "ending theme" that had swallowed a full dialogue scene before
# chaining the real lyrics on. A marker this long is out of distribution —
# refuse it and ship the cues instead: a leaked lyric line is recoverable,
# deleted dialogue is not.
_THEME_MARKER_MAX_SPAN_S = 110.0
# Through-composed OP/ED themes (verses all distinct → no repeated chorus) are
# caught by a repetition-INDEPENDENT signal instead: a long, contiguous,
# single-speaker, proper-noun-free run in the head/tail window. Real dialogue
# turn-takes and names people/places within a few lines, so these thresholds
# (deliberately higher than the chorus path's) keep it off narration/monologue.
# NOTE: no per-cue length cap here — translated lyric lines are often long full
# sentences; the length cap made this a no-op on real themes.
_THEME_RUN_MIN_CUES = 6
_THEME_RUN_MIN_SPAN_S = 25.0


def _song_norm(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def _sent_split(text: str) -> list:
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+", (text or "").strip()) if p.strip()]


def _proper_noun_count(text: str) -> int:
    """Capitalized, mid-sentence tokens — a preview/dialogue signal (a sung
    lyric line rarely names characters). Sentence-initial caps don't count."""
    n, sent_start = 0, True
    for tok in (text or "").split():
        w = tok.strip(".,!?;:\"'()[]…—–“”’")
        if not sent_start and len(w) > 1 and w[:1].isupper() and not w.isupper():
            n += 1
        sent_start = bool(tok) and tok[-1] in ".!?…"
    return n


# Sung-hook mining: a TitleCase phrase repeated across cues of one window is
# the song's HOOK ("Just Love", "Wild Wing"), not a person — but the raw
# proper-noun counter scored it 2, which flagged every verse containing it as
# a "preview" and shipped the ending theme's verses as dialogue.
_HOOK_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")

# Capitalized tokens that are never character names — interjections and
# lyric filler the ASR capitalizes when it glues sung fragments into one cue.
# Exempted from the proper-noun count INSIDE theme windows only.
_LYRIC_CAP_STOP = frozenset("""
uh uhh ah ahh oh ohh ha hey yeah yea la na naa ooh oooh whoa woah huh hmm mm
mmm wow ts tonight baby love again right there
""".split())


def _ends_terminal(text: str) -> bool:
    """Does the cue end like a finished spoken sentence? Sung verse lines are
    translated as unpunctuated fragments; dialogue closes with .!?…"""
    t = (text or "").rstrip().rstrip('"\'’”)')
    return bool(t) and t[-1] in ".!?…。！？"


def _longest_theme_run(rows, idxs, is_preview, pn_fn=None, soft=False):
    """Longest run of consecutive (within ``idxs``) cues that read as a sung
    theme: same speaker (when the cue is labelled), no proper nouns, and not a
    next-episode preview. A speaker change, proper nouns, or a preview cue end
    the run — so real dialogue (which turn-takes and names people/places) can
    never be absorbed. No per-cue length cap: translated lyric lines are often
    long full sentences. Returns the row indices of the best run.

    ``soft=True`` lets a cue with EXACTLY ONE capitalized token CONTINUE (not
    start) a run: ASR garble routinely capitalizes an interjection mid-lyric
    ("cool the heat Uh", "Protecting your gaze Right"), and the strict rule
    broke a measured 10-cue opening theme into fragments too short to
    collapse. The caller must re-vet a soft run (strict majority + terminal-
    punctuation scarcity) before acting on it."""
    pn = pn_fn or _proper_noun_count
    best: list = []
    cur: list = []
    cur_spk = None
    soft_idx: set = set()

    def _trimmed(run: list) -> list:
        # A soft cue may only be INTERIOR: lyrics tolerate a stray capital
        # mid-song, but a run must open and close on strictly-clean cues —
        # otherwise a single-name narration line adjacent to the song rides
        # along as its trailing edge.
        out = list(run)
        while out and out[-1] in soft_idx:
            out.pop()
        return out

    for i in idxs:
        txt = (rows[i].get("text") or "").strip()
        spk = rows[i].get("speaker")
        n_pn = pn(txt) if txt else 99
        strict_ok = bool(txt) and not is_preview(txt) and n_pn == 0
        soft_ok = (soft and bool(txt) and not is_preview(txt) and n_pn == 1)
        theme_like = strict_ok or (soft_ok and bool(cur))
        same_spk = cur_spk is None or spk in (None, "") or spk == cur_spk
        if theme_like and (not cur or same_spk):
            if not cur:
                cur_spk = spk
            if not strict_ok:
                soft_idx.add(i)
            cur.append(i)
        else:
            t = _trimmed(cur)
            if len(t) > len(best):
                best = t
            if strict_ok:
                cur, cur_spk = [i], spk
            else:
                cur, cur_spk = [], None
    t = _trimmed(cur)
    if len(t) > len(best):
        best = t
    return best


def _theme_anchor_start(starts: list) -> float:
    """The time a theme marker should claim, given its run's cue starts.

    A run whose earliest member sits at ~0:00 while its NEXT member starts
    several seconds later is anchored by a hallucination: Whisper emits
    phantom lyric repeats at exactly 0.000, they legitimately join the run
    (they ARE chorus text), and the marker inherits the zero. The second
    cue's start is the run's real onset in that shape. A genuinely-early
    theme (first two cues adjacent) keeps its first start."""
    ss = sorted(float(s) for s in starts)
    if len(ss) >= 2 and ss[0] < 1.0 and ss[1] - ss[0] > 8.0:
        return ss[1]
    return ss[0] if ss else 0.0


def source_chorus_spans(source_rows, head_s: float = 120.0,
                        tail_s: float = 240.0,
                        min_span_s: float = 20.0) -> list:
    """Theme time-spans detected on the SOURCE track. ``[(start, end), …]``.

    The translated-side chorus detection is at the mercy of how the LLM
    phrased the lyrics THAT run: the same episode's ending theme collapsed
    on runs whose translation left lyrics unpunctuated and shipped 10-14
    lyric cues as dialogue on runs where it punctuated them as sentences —
    with an identical source track underneath both. The Whisper-JA text is
    the phrasing-independent evidence: a sung theme repeats its chorus in
    the SOURCE, so repeated normalized source lines inside the opening/
    closing window mark the song no matter what the translation did.
    Whisper's own JA hallucination loops also repeat, so a span needs TWO
    distinct repeated lines (a loop repeats one) plus a real duration.
    Returned spans feed ``collapse_theme_by_music_spans`` alongside music
    spans. Empty on any failure or when the source shows no repetition."""
    try:
        rows = _as_rows(source_rows)
        if len(rows) < 6:
            return []

        def _st(r):
            return float(r.get("start") or 0.0)

        def _en(r):
            return float(r.get("end") or _st(r))

        first_t = min(_st(r) for r in rows)
        last_t = max(_en(r) for r in rows)
        out = []
        for w0, w1 in ((first_t, first_t + head_s),
                       (max(0.0, last_t - tail_s), last_t)):
            idxs = [i for i, r in enumerate(rows)
                    if (r.get("text") or "").strip()
                    and not _is_marker((r.get("text") or "").strip())
                    and w0 - 0.01 <= _st(r) <= w1 + 0.01]
            if len(idxs) < 4:
                continue
            counts: dict = {}
            for i in idxs:
                # CJK has no inter-sentence whitespace for _sent_split to
                # cut on — count the whole cue AND its punctuation-split
                # pieces (Whisper glues the hook onto different verse text
                # each repeat, so the full-cue key never matches twice; the
                # PIECE does). Each cue contributes a key at most once.
                txt = (rows[i].get("text") or "").strip()
                keys = {_song_norm(txt)}
                for piece in re.split(r"[。！？!?…♪、]+", txt):
                    keys.add(_song_norm(piece.strip()))
                for ns in keys:
                    if len(ns) >= 6:
                        counts[ns] = counts.get(ns, 0) + 1
                # (the hits pass below rebuilds the same keys per cue)
            chorus = {ns for ns, c in counts.items() if c >= 2}
            if not chorus:
                logger.info("source chorus: window %.0f-%.0fs — no repeated "
                            "line among %d cue(s)", w0, w1, len(idxs))
                continue
            if len(chorus) < 2:
                # ONE repeated line can be the song's hook — or a Whisper
                # loop. A loop repeats its one line MANY times over a run of
                # nothing else; a hook recurs 2-3 times among DISTINCT verse
                # lines. The first cut of this gate demanded two distinct
                # repeats and matched nothing on a real ED whose verses each
                # transcribe slightly differently on every pass — only the
                # hook is stable enough to repeat verbatim.
                only = next(iter(chorus))
                if (counts[only] > 3 or len(idxs) < 5
                        or len(counts) < max(4, int(0.6 * len(idxs)))):
                    logger.info(
                        "source chorus: window %.0f-%.0fs — single repeated "
                        "line rejected (%d occurrence(s), %d distinct of %d)",
                        w0, w1, counts[only], len(counts), len(idxs))
                    continue
            def _cue_keys(i: int) -> set:
                txt = (rows[i].get("text") or "").strip()
                ks = {_song_norm(txt)}
                for piece in re.split(r"[。！？!?…♪、]+", txt):
                    ks.add(_song_norm(piece.strip()))
                return ks

            hits = [i for i in idxs if _cue_keys(i) & chorus]
            if len(hits) < (3 if len(chorus) >= 2 else 2):
                continue
            a = min(_st(rows[i]) for i in hits)
            b = max(_en(rows[i]) for i in hits)
            # A single hook's occurrences bound a NARROWER span than a full
            # two-line chorus — accept the theme-minimum instead of the
            # chorus minimum, and let the collapse eat only what lies inside.
            if b - a >= (min_span_s if len(chorus) >= 2 else 12.0):
                logger.info(
                    "source chorus: window %.0f-%.0fs — %d repeated line(s), "
                    "%d hit cue(s) spanning %.0f-%.0fs",
                    w0, w1, len(chorus), len(hits), a, b)
                out.append((round(a, 3), round(b, 3)))
        return out
    except Exception:
        return []


def collapse_song_choruses(segments, target_lang: str = "en"):
    """Collapse a sung OPENING/ENDING theme — mis-transcribed as duplicated,
    garbled dialogue — into a single ``[♪ … theme ♪]`` marker, the way official
    subtitles do.

    The signal is chorus REPETITION inside the head/tail windows: a normalized
    sentence that recurs ≥2× is a chorus line (spoken dialogue does not repeat
    whole sentences within a couple of minutes). The collapsed run is anchored
    to the actual chorus cues, so it can never eat surrounding dialogue, and a
    next-episode preview narrated over the ending theme (flagged by preview
    keywords or ≥2 proper nouns) is preserved. Returns ``(rows, changed)``;
    CJK targets, short transcripts, and non-repeating windows are passed
    through unchanged. Fail-soft."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        if tgt in _CJK_TARGETS or len(rows) < 6:
            return rows, False
        try:
            from backend.config import settings as _s
            if not getattr(_s, "TRANSCRIPT_MARK_THEME_SONGS", True):
                return rows, False
            head_s = float(getattr(_s, "TRANSCRIPT_THEME_HEAD_S", 150.0))
            tail_s = float(getattr(_s, "TRANSCRIPT_THEME_TAIL_S", 210.0))
            min_rep = int(getattr(_s, "TRANSCRIPT_THEME_MIN_REPEATS", 2))
        except Exception:
            head_s, tail_s, min_rep = 150.0, 210.0, 2

        def _st(r):
            return float(r.get("start") or 0.0)

        def _en(r):
            return float(r.get("end") or _st(r))

        first_t = _st(rows[0])
        last_t = max(_en(r) for r in rows)
        windows = [(first_t, first_t + head_s, _THEME_OPEN_LABEL),
                   (last_t - tail_s, last_t, _THEME_END_LABEL)]

        def _is_preview_raw(txt):
            return bool(_PREVIEW_RE.search(txt)) or _proper_noun_count(txt) >= 2

        try:
            from backend.config import settings as _gs
            max_gap = float(getattr(_gs, "TRANSCRIPT_THEME_MAX_GAP_S", 10.0))
        except Exception:
            max_gap = 10.0

        def _groups(cand: list) -> list:
            """Split window indices into TIME-CONTIGUOUS groups (gap ≤ max_gap).

            This is the load-bearing constraint the collapse was missing: a
            real sung theme is one continuous block of audio, so no legitimate
            chorus run contains a minute of silence. Without it, chorus lines
            at 22:38-23:07 bounded a run that reached back across a 60-second
            gap and absorbed a whole scene of dialogue starting at 21:34 —
            including "I'll kill you.", the episode's signature line. Nineteen
            cues shipped as one four-second marker."""
            out, cur = [], []
            for i in cand:
                if cur and _st(rows[i]) - _en(rows[cur[-1]]) > max_gap:
                    out.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                out.append(cur)
            return out

        drop = set()
        marker_at = {}
        runs_at: dict = {}   # marker key -> indices its runs dropped (for veto)
        for w0, w1, label in windows:
            idxs = [i for i, r in enumerate(rows)
                    if (r.get("text") or "").strip()
                    and not _is_marker((r.get("text") or "").strip())
                    and w0 - 0.01 <= _st(r) <= w1 + 0.01]
            if len(idxs) < 4:
                continue
            # Sung HOOKS: a TitleCase phrase recurring across this window's
            # LYRIC-SHAPED cues is the song's refrain title ("Just Love"), not
            # a person — but the raw counter scored it 2 proper nouns, which
            # flagged every verse carrying it as a "preview" and shipped the
            # ending theme's verses as dialogue. Qualification is strict so a
            # character named in ordinary dialogue can never become a hook:
            # the phrase must recur in ≥2 cues that are UNPUNCTUATED and hold
            # no OTHER capitalized content (interjection stoplist applied),
            # and must never appear in a punctuated (dialogue-shaped) cue of
            # the window.
            def _neutral_caps(txt: str) -> str:
                """Lowercase stoplisted interjection tokens IN PLACE — deleting
                them shifted sentence-position tracking (a name right after a
                cue-initial "Oh" became sentence-initial and stopped counting)
                and dropped terminal periods riding on the deleted token."""
                out = []
                for t in (txt or "").split():
                    core = t.strip(".,!?;:\"'()[]…—–“”’")
                    out.append(t.lower() if core.lower() in _LYRIC_CAP_STOP
                               else t)
                return " ".join(out)

            _hook_qual: dict = {}
            _hook_dialog: set = set()
            for _i in idxs:
                _t = (rows[_i].get("text") or "").strip()
                for _m in set(_HOOK_RE.findall(_t)):
                    if len(_m) <= 2:
                        continue
                    if _ends_terminal(_t):
                        _hook_dialog.add(_m)
                        continue
                    _rest = re.sub(r"\b" + re.escape(_m) + r"\b", " ", _t)
                    if _proper_noun_count(_neutral_caps(_rest)) == 0:
                        _hook_qual[_m] = _hook_qual.get(_m, 0) + 1
            hooks = {h for h, c in _hook_qual.items()
                     if c >= 2 and h not in _hook_dialog}

            def _strip_hooks(txt: str) -> str:
                for h in hooks:
                    txt = re.sub(r"\b" + re.escape(h) + r"\b", h.lower(), txt)
                return txt

            def _pn(txt: str) -> int:
                """Proper nouns that count as DIALOGUE evidence: hooks are the
                song's refrain, and stoplisted interjections are ASR garble —
                fragment-gluing capitalizes them mid-cue ("cool the heat Uh",
                "Protecting your gaze Right"). Both are NEUTRALIZED (lower-
                cased in place), never deleted, so token positions and the
                terminal punctuation they carry stay intact for the counter."""
                return _proper_noun_count(_neutral_caps(_strip_hooks(txt)))

            def _is_preview(txt):
                return bool(_PREVIEW_RE.search(txt)) or _pn(txt) >= 2

            def _absorbable(i: int) -> bool:
                """A cue that can ride along INSIDE/AROUND a detected song
                block: no proper nouns (hooks exempt), not a preview,
                lyric-length. Extension uses a slightly looser length cap than
                the in-run test — translated lyric lines are often full
                sentences."""
                txt = (rows[i].get("text") or "").strip()
                return (not _is_preview(txt)
                        and _pn(txt) == 0
                        and len(txt) <= max(_LYRIC_MAX_CHARS, 60))

            def _vet_soft_run(trun: list) -> bool:
                """A soft (pn≤1-tolerant) run must still LOOK sung before it
                may collapse: mostly strictly-clean cues, and mostly WITHOUT
                terminal punctuation. Verse translations arrive as
                unpunctuated fragments; dialogue closes its sentences — a
                measured school-chatter run passed every other test and died
                only here (5/7 cues ended in .!?)."""
                if not trun:
                    return False
                # Songs don't name the cast: a run whose cues collectively
                # carry 3+ proper nouns is narration, whatever its shape.
                # The soft path tolerates pn≤1 PER CUE, so a 10-cue run
                # could smuggle 10 names through — a measured run's chain
                # absorbed the PROLOGUE ("who opposed Alliance", "sent
                # disguised weapons to Earth"…) and its ED soft run ate the
                # next-episode preview ("The Gundam sank to the ocean",
                # "Union sent Marley…"); the two-signal tier had ALREADY
                # rejected the same window for exactly this signal ("4/9
                # cues carry proper nouns") but the text pass never read
                # it. Measured true lyric runs carry 0-1 total.
                total_pn = sum(
                    _pn((rows[i].get("text") or "").strip()) for i in trun)
                # Longer runs get the stricter bar: measured true lyric runs
                # carry 0 total, and a wrongly-vetoed song merely leaks
                # lyrics (the lesser evil) while a wrongly-passed run
                # deletes narration.
                if total_pn >= (2 if len(trun) >= 5 else 3):
                    logger.info(
                        "theme collapse: soft run %.1f-%.1fs (%d cues) "
                        "rejected — %d proper noun(s) across the run; "
                        "songs don't name the cast",
                        min(_st(rows[i]) for i in trun),
                        max(_en(rows[i]) for i in trun),
                        len(trun), total_pn)
                    return False
                strict = sum(
                    1 for i in trun
                    if _pn((rows[i].get("text") or "").strip()) == 0)
                punct = sum(
                    1 for i in trun
                    if _ends_terminal((rows[i].get("text") or "").strip()))
                return (strict >= max(3, len(trun) // 2 + 1)
                        and punct <= len(trun) * 0.34)

            _w_markers: list = []
            # Chorus lines are identified over the WHOLE window (a chorus and
            # its reprise are often separated by a verse or a narrated
            # preview), but a collapse run must live inside ONE contiguous
            # group — repetition says "this window holds a song"; contiguity
            # says "these particular cues are it".
            counts = {}
            for i in idxs:
                for s in _sent_split(rows[i].get("text") or ""):
                    ns = _song_norm(s)
                    if len(ns) >= _CHORUS_MIN_CHARS:
                        counts[ns] = counts.get(ns, 0) + 1
            chorus = {ns for ns, c in counts.items() if c >= 2}
            if len(chorus) < min_rep:
                # No repeated chorus, but a THROUGH-COMPOSED theme (an OP with
                # all-distinct verse lines, or an ED with only a short hook) is
                # a long single-speaker, proper-noun-free run. Per contiguous
                # group, so the run can no longer bridge a scene of dialogue.
                # Soft mode + vetting: one garbled interjection-capital per cue
                # is tolerated, but the run must stay strict-majority and
                # mostly unpunctuated to collapse.
                for grp in _groups(idxs):
                    trun = _longest_theme_run(rows, grp, _is_preview,
                                              pn_fn=_pn, soft=True)
                    if len(trun) >= _THEME_RUN_MIN_CUES and _vet_soft_run(trun):
                        # A SOFT run carries no repetition evidence — only
                        # shape — and in the CLOSING window that shape also
                        # fits a quiet dialogue scene: a measured run
                        # collapsed the birthday scene (13 mostly-
                        # unpunctuated cues, "I'll kill you." included)
                        # as the "ending theme", then chained the real
                        # lyrics onto it — a 124-second marker over 33
                        # cues. Structure disambiguates: the ED is the
                        # LAST content before the next-episode preview, so
                        # a soft run followed by more NON-preview content
                        # in the window is a scene, not the song.
                        if label == _THEME_END_LABEL:
                            _after = [
                                j for j in idxs
                                if j > max(trun) and j not in drop
                                and not (rows[j].get("text") or "")
                                .strip().startswith("[")]
                            _blockers = [
                                j for j in _after
                                if not _is_preview(
                                    (rows[j].get("text") or "").strip())]
                            if _blockers:
                                logger.info(
                                    "theme collapse: END soft run %.1f-%.1fs "
                                    "(%d cues) vetoed — %d non-preview cue(s) "
                                    "follow it (e.g. %r); the ending theme is "
                                    "the last content before the preview",
                                    min(_st(rows[i]) for i in trun),
                                    max(_en(rows[i]) for i in trun),
                                    len(trun), len(_blockers),
                                    (rows[_blockers[0]].get("text") or "")[:40])
                                continue
                        tm_start = _theme_anchor_start(
                            [_st(rows[i]) for i in trun])
                        tm_end = max(_en(rows[i]) for i in trun)
                        if tm_end - tm_start >= _THEME_RUN_MIN_SPAN_S:
                            drop.update(trun)
                            marker_at[min(trun)] = (tm_start, tm_end, label)
                            runs_at.setdefault(min(trun), set()).update(trun)
                            _w_markers.append(min(trun))
                            logger.info(
                                "theme collapse: %s soft run collapsed "
                                "%.1f-%.1fs (%d cues) — absorbed e.g. %s",
                                label, tm_start, tm_end, len(trun),
                                "; ".join(
                                    repr((rows[i].get("text") or "")[:28])
                                    for i in trun[:4]))
            else:
              for grp in _groups(idxs):
                g_hits = [i for i in grp
                          if not _is_preview((rows[i].get("text") or "").strip())
                          and any(_song_norm(s) in chorus
                                  for s in _sent_split(rows[i].get("text") or ""))]
                if len(g_hits) < 2:
                    # A group with at most one chorus-shaped cue is a scene of
                    # dialogue, even when the window as a whole holds a song.
                    # This is the exact guard the collapse was missing: chorus
                    # hits at 22:38-23:07 used to bound a run that reached back
                    # across a 60-second gap and ate 19 dialogue cues from
                    # 21:34 — including "I'll kill you.", the episode's
                    # signature line.
                    continue
                lo, hi = min(g_hits), max(g_hits)
                run = []
                for i in grp:
                    if not (lo <= i <= hi):
                        continue
                    txt = (rows[i].get("text") or "").strip()
                    if _is_preview(txt):
                        continue   # preserve preview/dialogue interleaved in the run
                    is_chorus = any(_song_norm(s) in chorus for s in _sent_split(txt))
                    lyric_like = (_pn(txt) == 0
                                  and len(txt) <= _LYRIC_MAX_CHARS)
                    if is_chorus or lyric_like:
                        run.append(i)
                if len(run) < 2:
                    continue
                # Extend over the song's own head/tail INSIDE the group: the
                # verse lines before the first chorus repeat and after the last
                # one are part of the same continuous song, and leaving them
                # made half the ending theme collapse while the other half
                # shipped as dialogue. Proper nouns / previews still stop the
                # walk, so the next-episode narration over the outro survives.
                g_lo, g_hi = grp.index(min(run)), grp.index(max(run))
                _chorus_lo, _chorus_hi = grp[g_lo], grp[g_hi]
                _chorus_n = len(run)
                j = g_lo - 1
                while j >= 0 and _absorbable(grp[j]):
                    run.append(grp[j])
                    j -= 1
                j = g_hi + 1
                while j < len(grp) and _absorbable(grp[j]):
                    run.append(grp[j])
                    j += 1
                m_start = _theme_anchor_start([_st(rows[i]) for i in run])
                m_end = max(_en(rows[i]) for i in run)
                if m_end - m_start < _THEME_MIN_SPAN_S:
                    continue
                # Where the marker lands has oscillated between runs — 0.000s
                # on some, a correct ~30s on others — and the mechanism has
                # only ever been reproduced on synthetic input, never observed.
                # The walk below extends the chorus run backwards through
                # ``_absorbable`` cues, so the marker inherits the START of
                # whatever it absorbed. Log the extension and the cues that
                # caused it: an anchor 25s adrift of its chorus is the
                # signature, and until this line exists the evidence to fix it
                # does not survive the run.
                _absorbed = sorted(i for i in run
                                   if i < _chorus_lo or i > _chorus_hi)
                if _absorbed:
                    _chorus_start = _st(rows[_chorus_lo])
                    logger.info(
                        "theme collapse: %s run extended %d→%d cue(s), anchor "
                        "%.3fs (chorus starts %.3fs, drift %+.3fs) — absorbed %s",
                        label, _chorus_n, len(run), m_start,
                        _chorus_start, m_start - _chorus_start,
                        "; ".join(
                            f"{_st(rows[i]):.2f}s {(rows[i].get('text') or '')[:28]!r}"
                            for i in _absorbed[:4]))
                drop.update(run)
                # A second window re-collapsing the same run must never
                # SHRINK a marker the first window's chain already extended
                # (head and tail windows overlap on short inputs).
                _prev = marker_at.get(min(run))
                if _prev:
                    m_start = min(m_start, _prev[0])
                    m_end = max(m_end, _prev[1])
                marker_at[min(run)] = (m_start, m_end, label)
                runs_at.setdefault(min(run), set()).update(run)
                _w_markers.append(min(run))

            # ── Markers-only policy: chain trailing VERSE blocks into the
            # marker. The chorus path collapses the repeats, but the verses
            # that follow after an instrumental bridge are the SAME song — a
            # measured run kept its "[♪ Ending theme ♪]" marker AND shipped
            # five verse cues 24s later as dialogue. A relaxed lyric run that
            # OPENS a group within 60s of a marker's end is absorbed and the
            # marker extended; previews and dialogue still break the run, and
            # the soft-run vetting (strict majority + unpunctuated majority)
            # applies in full.
            def _vet_tiny_tail(trun: list) -> bool:
                """A TWO-cue trailing block may still be the song's last
                verse: a measured run kept its "[♪ Ending theme ♪]" marker
                and shipped exactly two lyric cues 24s later ("I don't think
                I should say…", "Jump Suits") — under the 3-cue chain
                minimum. Two cues carry less evidence, so every one must
                look sung: NO terminal punctuation anywhere (dialogue closes
                its sentences), at most one stray capital per cue, and ≥4s
                of combined dwell. Only ever consulted for blocks that open
                within the chain's 60s window after a theme marker."""
                if len(trun) != 2:
                    return False
                texts = [(rows[i].get("text") or "").strip() for i in trun]
                if any(_ends_terminal(t) for t in texts):
                    return False
                if any(_pn(t) > 1 for t in texts):
                    return False
                span = (max(_en(rows[i]) for i in trun)
                        - min(_st(rows[i]) for i in trun))
                return span >= 4.0

            for mk in list(_w_markers):
                # ENDING themes only. The chain exists for one measured case:
                # the ED's last verse after an instrumental bridge. In the
                # OPENING window it has now eaten the prologue narration on
                # two measured runs ("Humanity sought new hope…", 1:52-2:14,
                # absorbed as "verse blocks" 19s after the song actually
                # ended) — an OP ends hard and cuts to the episode; there is
                # no post-bridge verse to chain.
                if marker_at[mk][2] != _THEME_END_LABEL:
                    continue
                _more = True
                while _more:
                    _more = False
                    m_s, m_e, lab = marker_at[mk]
                    for grp in _groups(idxs):
                        g_live = [i for i in grp if i not in drop]
                        if not g_live:
                            continue
                        g0 = _st(rows[g_live[0]])
                        if not (m_e - 0.01 <= g0 <= m_e + 60.0):
                            continue
                        trun = _longest_theme_run(rows, g_live, _is_preview,
                                                  pn_fn=_pn, soft=True)
                        if not (trun and trun[0] == g_live[0]
                                and len(trun) >= 3):
                            # Soft (one-stray-capital) cues may CONTINUE a
                            # run but never open one, so a two-cue tail
                            # whose cues are both pn==1 lyric garble yields
                            # an empty run — or the longest run lands on an
                            # unrelated clean cue DEEPER in the group (a
                            # preview sentence), which is just as useless
                            # here: the chain only ever absorbs from the
                            # group's front. Either way, collect the leading
                            # lyric-tolerable cues directly; the tiny-tail
                            # vet holds the bar.
                            _lead: list = []
                            for i in g_live:
                                _t = (rows[i].get("text") or "").strip()
                                if (not _t or _is_preview(_t) or _pn(_t) > 1
                                        or len(_t) > max(_LYRIC_MAX_CHARS, 60)):
                                    break
                                _lead.append(i)
                                if len(_lead) > 2:
                                    break
                            if len(_lead) == 2:
                                trun = _lead
                        _ok = (len(trun) >= 3 and _vet_soft_run(trun)
                               and (max(_en(rows[i]) for i in trun)
                                    - min(_st(rows[i]) for i in trun)) >= 6.0
                               ) or _vet_tiny_tail(trun)
                        if _ok and trun and trun[0] == g_live[0]:
                            _new_end = max(_en(rows[i]) for i in trun)
                            if _new_end - m_s > _THEME_MARKER_MAX_SPAN_S:
                                logger.info(
                                    "theme collapse: chain stopped — extending "
                                    "%s to %.1fs would span %.0fs (cap %.0fs)",
                                    lab, _new_end, _new_end - m_s,
                                    _THEME_MARKER_MAX_SPAN_S)
                                continue
                            logger.info(
                                "theme collapse: chain absorbed %d cue(s) "
                                "%.1f-%.1fs into %s — e.g. %s",
                                len(trun), _st(rows[trun[0]]), _new_end, lab,
                                "; ".join(
                                    repr((rows[i].get("text") or "")[:28])
                                    for i in trun[:4]))
                            drop.update(trun)
                            marker_at[mk] = (m_s, _new_end, lab)
                            runs_at.setdefault(mk, set()).update(trun)
                            _more = True
                            break

        # Refuse out-of-distribution markers at the door: restore their cues
        # instead of shipping a marker that plainly swallowed a scene. The
        # reference episode's longest lyric block is 66s; the measured failure
        # was a 124-second "ending theme" holding a whole dialogue scene
        # ("I'll kill you." included) plus the chained real lyrics.
        for _k in list(marker_at):
            _s0, _e0, _lab = marker_at[_k]
            if _e0 - _s0 > _THEME_MARKER_MAX_SPAN_S:
                _restore = runs_at.get(_k, set())
                logger.warning(
                    "theme collapse: VETO %s %.1f-%.1fs — %.0fs span exceeds "
                    "the %.0fs cap; restoring %d absorbed cue(s)",
                    _lab, _s0, _e0, _e0 - _s0, _THEME_MARKER_MAX_SPAN_S,
                    len(_restore))
                drop.difference_update(_restore)
                del marker_at[_k]
        if not drop and not marker_at:
            return rows, False
        out = []
        for i, r in enumerate(rows):
            if i in marker_at:
                s, e, lab = marker_at[i]
                base = dict(r)
                base.update({"text": lab, "start": round(s, 3), "end": round(e, 3),
                             "words": None})
                out.append(base)
            if i in drop:
                continue
            out.append(dict(r))
        return out, True
    except Exception as e:
        # Fail-soft, but never silently: a bug in here disables the WHOLE
        # theme collapse and the transcript simply ships the sung lyrics as
        # dialogue — a failure mode indistinguishable, from the outside, from
        # "no theme was detected". A bare NameError hid behind this handler
        # once already.
        logger.warning("theme collapse failed (%s: %s) — themes stay as cues",
                       type(e).__name__, e)
        return _as_rows(segments), False


def sanitize_translated_transcript(segments, target_lang: str = "en"):
    """Return ``(cleaned_rows, changed)``.

    ``cleaned_rows`` are plain dicts. ``changed`` is True when anything was
    dropped/reordered, so callers can persist the repair only when needed.
    Fail-soft: on any error the original rows are returned unchanged.
    """
    try:
        rows = list(segments or [])
        if not rows:
            return rows, False
        tgt = (target_lang or "").strip().lower().split("-")[0]
        # A CJK target legitimately contains CJK — only de-dup there, never drop.
        drop_source_script = tgt not in _CJK_TARGETS

        # Pre-count normalized keys so a SUBSTANTIAL line that recurs ≥
        # _GROSS_REPEAT_DROP times can be recognized as a hallucination LOOP and
        # dropped entirely (not just trimmed to one). Markers + source-script
        # relapses are excluded from the count exactly as from the keep logic.
        totals: dict[str, int] = {}
        for seg in rows:
            text = (_get(seg, "text", "") or "").strip()
            if not text or _is_marker(text):
                continue
            if drop_source_script and _cjk_ratio(text) > 0.30:
                continue
            key = " ".join(text.lower().split())
            if len(key) >= _SUBSTANTIAL_DEDUP_CHARS:
                totals[key] = totals.get(key, 0) + 1

        kept = []
        counts: dict[str, int] = {}
        for seg in rows:
            text = (_get(seg, "text", "") or "").strip()
            if not text:
                continue
            # Standalone non-lexical grunt cue ("Nn...", "Mmm") — always-on
            # drop (the polisher's filler strip is off by default). The
            # ``fullmatch`` keeps it to a WHOLE-cue grunt, never a real cue
            # that merely ends in "…mm".
            if not _is_marker(text) and _GRUNT_CUE_RE.fullmatch(text):
                continue
            if drop_source_script and not _is_marker(text) and _cjk_ratio(text) > 0.30:
                continue  # source-language relapse — not part of a translation
            if not _is_marker(text):
                key = " ".join(text.lower().split())
                # A substantial line repeated many times over is a Whisper
                # hallucination loop — drop every copy, not just the excess.
                if len(key) >= _SUBSTANTIAL_DEDUP_CHARS and totals.get(key, 0) >= _GROSS_REPEAT_DROP:
                    continue
                n = counts.get(key, 0)
                # Substantial lines collapse to one; short lines keep up to 2.
                cap = 1 if len(key) >= _SUBSTANTIAL_DEDUP_CHARS else _MAX_DUPLICATES
                if n >= cap:
                    continue  # duplicate artifact (Whisper repetition / corruption)
                counts[key] = n + 1
            kept.append(seg.model_dump(mode="json") if hasattr(seg, "model_dump")
                       else dict(seg))

        kept.sort(key=lambda s: (float(s.get("start") or 0), float(s.get("end") or 0)))
        changed = (len(kept) != len(rows))
        if not changed:
            # Length unchanged but order might have been fixed — detect a reorder.
            for a, b in zip(kept, rows):
                bt = (_get(b, "text", "") or "")
                if (a.get("text") or "") != bt:
                    changed = True
                    break
        return kept, changed
    except Exception:
        return list(segments or []), False


# ── Degenerate-window sweep (pre-translation) ───────────────────────────────

def repair_degenerate_cue_windows(rows: list, dup_ratio: float = 0.6,
                                  ) -> tuple[list, list, list]:
    """Last net before translation: no source cue may carry an unusable
    time window.

    A measured run reached translation with 20 source cues whose windows
    had collapsed; the translated cues inherited them and the formatter
    packed the block at 0:00 over the opening theme. The upstream causes
    get fixed where they live, but this sweep guarantees the INVARIANT:

      * a degenerate cue (window < 0.05 s) whose text near-duplicates
        (``dup_ratio`` similarity) another, validly-timed cue is an ECHO —
        its content already ships at the right time — so it is dropped;
      * a degenerate cue with UNIQUE text is real content whose window was
        lost — it is re-timed into the silence between its list
        neighbours (position in the list is the one ordering signal a
        window-less cue still carries), or dropped when the neighbours
        leave no room.

    Pure stdlib, order-preserving, mutates windows in place. Returns
    ``(rows, dropped_samples, repaired_samples)``."""
    from difflib import SequenceMatcher

    def _get(r, k, d=None):
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    def _win(r):
        try:
            a = float(_get(r, "start", 0.0) or 0.0)
            b = float(_get(r, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0, 0.0
        return a, b

    def _set_win(r, a, b):
        if isinstance(r, dict):
            r["start"], r["end"] = a, b
        else:
            r.start, r.end = a, b

    valid_idx = []
    degen_idx = []
    for i, r in enumerate(rows or []):
        a, b = _win(r)
        (valid_idx if b - a >= 0.05 else degen_idx).append(i)
    if not degen_idx:
        return rows, [], []

    valid_texts = [
        ((_get(rows[i], "text", "") or "").strip().lower(), i)
        for i in valid_idx
    ]
    dropped, repaired = [], []
    drop_set = set()
    for i in degen_idx:
        r = rows[i]
        txt = ((_get(r, "text", "") or "").strip())
        low = txt.lower()
        is_echo = False
        for vt, _vi in valid_texts:
            if not vt or abs(len(vt) - len(low)) > max(10, len(low)):
                continue
            if SequenceMatcher(None, low, vt).ratio() >= dup_ratio:
                is_echo = True
                break
        if is_echo or not txt:
            drop_set.add(i)
            dropped.append(f"echo:{txt[:36]!r}")
            continue
        # Unique text: re-time into the gap between the nearest validly
        # timed neighbours on each side.
        prev_end = 0.0
        for j in range(i - 1, -1, -1):
            if j in drop_set:
                continue
            a, b = _win(rows[j])
            if b - a >= 0.05:
                prev_end = b
                break
        next_start = None
        for j in range(i + 1, len(rows)):
            a, b = _win(rows[j])
            if b - a >= 0.05:
                next_start = a
                break
        room = (next_start - prev_end) if next_start is not None else 4.0
        if room < 0.4:
            drop_set.add(i)
            dropped.append(f"no-room:{txt[:36]!r}")
            continue
        a = prev_end + min(0.15, room * 0.1)
        b = min(a + max(0.6, min(3.0, room * 0.6)),
                (next_start - 0.05) if next_start is not None else a + 3.0)
        _set_win(r, round(a, 3), round(b, 3))
        repaired.append(f"{txt[:36]!r} -> {a:.2f}-{b:.2f}s")
    if drop_set:
        rows = [r for i, r in enumerate(rows) if i not in drop_set]
    return rows, dropped, repaired


# ── Echo suppression (translated track) ────────────────────────────────────

def _echo_norm(text: str) -> str:
    """Comparison form: lowercase, punctuation-free, whitespace-collapsed."""
    t = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return " ".join(t.split())


_ECHO_STOPWORDS = frozenset("""
a an and are as at be been being but by can do does for from had has have he
her him his how i if in is it its me my not of on or our out she so than that
the their them then there they this to too us was we were what when which who
will with would you your
""".split())


def _echo_stems(text: str) -> set:
    """Content words, lightly stemmed. Two independent translations of one
    line share their CONTENT, not their surface form ("Reporting meteor
    strikes." vs "Reported as falling meteorites"), so the comparison has to
    see through inflection and function words."""
    out = set()
    for w in _echo_norm(text).split():
        if w in _ECHO_STOPWORDS or len(w) < 3:
            continue
        for suf in ("ings", "ing", "ies", "ied", "es", "ed", "s"):
            if len(w) - len(suf) >= 3 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def _echo_similarity(a: str, b: str) -> float:
    """How much two cues say the same thing: the better of surface
    similarity and stemmed content-word containment."""
    from difflib import SequenceMatcher
    surface = SequenceMatcher(None, _echo_norm(a), _echo_norm(b)).ratio()
    sa, sb = _echo_stems(a), _echo_stems(b)
    if not sa or not sb:
        return surface
    # Prefix-tolerant match: crude suffix stripping still leaves related forms
    # apart ("meteor" vs "meteorit"), and those pairs are exactly the ones two
    # independent translations of one line produce.
    # Containment on a ONE-word set is meaningless: "Father, what's that?"
    # reduces to {father} and then scores 1.0 against every other line
    # mentioning a father. Two content words minimum before containment may
    # override the surface measure.
    #
    # …and the two cues must LOOK somewhat alike as well. Run over the
    # PROFESSIONAL reference track this pass deleted 8 of its 347 cues, four
    # at containment exactly 1.00, every one a short reply whose content words
    # happen to recur in the longer line before it — the ordinary shape of
    # dialogue, not duplication. Two independent translations of one line
    # share wording as well as vocabulary, so a surface floor separates them:
    # it takes the reference's false positives from 8 to 6 while the
    # four-rendering meteor cluster this pass exists to collapse still
    # collapses to one. Raising the stem minimum instead scored better on the
    # reference and lost that collapse, which is the wrong trade.
    if min(len(sa), len(sb)) < 2 or surface < _ECHO_SURFACE_FLOOR:
        return surface
    small, large = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    hits = 0
    for w in small:
        if any(w == o or (len(w) >= 4 and len(o) >= 4
                          and (w.startswith(o[:4]) and (w.startswith(o) or o.startswith(w))))
               for o in large):
            hits += 1
    containment = hits / float(len(small))
    return max(surface, containment)



def _unique_proper_nouns(loser: str, keeper: str) -> bool:
    """True when ``loser`` carries a capitalized word ``keeper`` does not —
    a name, place or designation that would be lost by dropping it."""
    def _caps(t):
        # MID-SENTENCE capitals only. A sentence-initial capital is grammar,
        # not a name — counting it made every ordinary line look like it
        # carried a proper noun and disarmed the suppressor completely.
        out = set()
        for m in re.finditer(r"\b[A-Z][a-zA-Z'\-]{2,}", t or ""):
            k = m.start() - 1
            while k >= 0 and (t[k].isspace() or t[k] in "\"'("):
                k -= 1
            if k < 0 or t[k] in ".!?…:;":
                continue                  # opens a sentence — not evidence
            out.add(m.group(0).lower())
        return out
    return bool(_caps(loser) - _caps(keeper))


# Shortest cue that may be judged an echo. The professional reference ships
# "Good morning!" twice in one exchange and "Fire! Fire!!" back to back — at
# this length a repeat is dialogue, not duplication.
_ECHO_MIN_CHARS = 16
# How alike two cues must LOOK before shared vocabulary is allowed to convict
# them. Fitted to the empty band between the two measured populations, not
# guessed: at 0.45 the professional reference loses 6 cues instead of 8 and the
# meteor cluster still collapses; below it the reference damage climbs with no
# gain, above it genuine duplicate renderings start surviving.
_ECHO_SURFACE_FLOOR = 0.45


def suppress_echo_cues(rows: list, window_s: float = 12.0,
                       ratio: float = 0.66,
                       tail_window_s: float = 120.0,
                       tail_s: float = 150.0) -> tuple[list, list]:
    """Drop cues that re-say a nearby cue's content in different words.

    Independent decodes of the same audio (the main pass, the second listen,
    and the post-COMPLETE recovery) each translate separately, so one line of
    dialogue can ship two to four times. A measured run emitted the same
    meteor report four times across 3:22-3:34 where the professional
    reference has ONE line, and parked two cues repeating "Please leave it
    to me!" across eleven seconds that hold no dialogue at all.

    Semantic, not exact: the duplicates never match character-for-character
    because each was translated independently. Cues are compared to their
    neighbours inside ``window_s`` on a punctuation-free similarity ratio.

    Guards, each protecting a real subtitle pattern:
      * SHORT cues (< 12 chars normalized) are never echoes — professional
        tracks legitimately repeat exclamations ("Fire! Fire!!", "Enemy
        attack! Enemy attack!");
      * a cue is never compared across a SPEAKER change (two characters
        saying the same thing is drama, not duplication);
      * markers are exempt.

    The survivor is the cue with MEASURED word rows (audio-anchored) or,
    failing that, the longest text — the most complete rendering. Windows
    are left alone: the dropped cue's span belongs to the silence the
    reference also leaves empty.

    The last ``tail_s`` of an episode gets ``tail_window_s`` instead. The
    next-episode preview is read once over the ending theme, but the theme
    collapse and the recovery passes each leave their own copy behind and
    those copies land a minute or more apart — a measured run shipped the
    preview twice, at 22:59 and again at 24:11, with the episode title card
    between them. Twelve seconds cannot see that far. The wide window is
    confined to the tail because only there is a distant near-repeat more
    likely to be residue than drama; mid-episode, a callback line is real.
    Returns ``(rows, dropped_samples)``."""
    def _g(r, k, d=None):
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    n = len(rows or [])
    if n < 2:
        return rows, []
    norm, drop = [], set()
    for r in rows:
        norm.append(_echo_norm(_g(r, "text", "") or ""))
    track_end = 0.0
    for r in rows:
        try:
            track_end = max(track_end, float(_g(r, "end", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    tail_from = max(0.0, track_end - tail_s)
    for i in range(n):
        if i in drop or len(norm[i]) < _ECHO_MIN_CHARS:
            continue
        ti = (_g(rows[i], "text", "") or "").strip()
        if not ti or ti.startswith("["):
            continue
        try:
            ei = float(_g(rows[i], "end", 0.0) or 0.0)
            si = float(_g(rows[i], "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        win = max(window_s, tail_window_s) if si >= tail_from else window_s
        for j in range(i + 1, n):
            if j in drop or len(norm[j]) < _ECHO_MIN_CHARS:
                continue
            tj = (_g(rows[j], "text", "") or "").strip()
            if not tj or tj.startswith("["):
                continue
            try:
                sj = float(_g(rows[j], "start", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if sj - ei > win:
                break                     # rows are time-ordered; done with i
            if (_g(rows[i], "speaker", "") or "") != (_g(rows[j], "speaker", "") or ""):
                continue
            if _echo_similarity(_g(rows[i], "text", "") or "",
                                _g(rows[j], "text", "") or "") < ratio:
                continue
            # Same line twice. Keep the audio-anchored copy, else the fuller.
            wi = bool(_g(rows[i], "words", None)) and not _g(rows[i], "words_synthetic", None)
            wj = bool(_g(rows[j], "words", None)) and not _g(rows[j], "words_synthetic", None)
            if wj and not wi:
                loser, keeper = i, j
            elif wi and not wj:
                loser, keeper = j, i
            else:
                loser, keeper = (j, i) if len(norm[i]) >= len(norm[j]) else (i, j)
            # A cue that carries a NAME the survivor does not is not a
            # duplicate — it elaborates. The professional reference uses
            # exactly this beat ("My name..." then "My name is Relena
            # Darlian."), and an early build of this pass deleted the
            # heroine's surname because the fragment before it scored as a
            # match. Information the survivor lacks is never redundant.
            if _unique_proper_nouns(_g(rows[loser], "text", "") or "",
                                    _g(rows[keeper], "text", "") or ""):
                continue
            drop.add(loser)
            if loser == i:
                break                     # i is gone; move on
    if not drop:
        return rows, []
    samples = []
    for i in sorted(drop):
        try:
            a = float(_g(rows[i], "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            a = 0.0
        samples.append(f"{a:.2f}s {(_g(rows[i], 'text', '') or '')[:40]!r}")
    return [r for i, r in enumerate(rows) if i not in drop], samples


# ── Junk-cue filter ────────────────────────────────────────────────────────

# Roleplay stage-direction markup ("*Grunt* *grunt*"). Never valid subtitle
# text — a professional track writes "[grunts]" or nothing at all.
_ASTERISK_MARKUP_RE = re.compile(r"^\s*(?:\*[^*]+\*\s*)+$")
# Translation-note prose the model emits ABOUT its work instead of doing it.
# Substring-anchored on the note phrases themselves, so real dialogue that
# merely contains the word "translation" is untouched.
_META_NOTE_RE = re.compile(
    r"(?i)\bcheck(?:\s+the)?\s+translation\b|\btranslation\s+note\b"
    r"|^\(?\s*TN\s*:|^\s*note\s*:\s"
    # The model narrating its own copy-editing as if it were dialogue — a
    # measured run shipped "Misspelled: 'Capturing' corrected." as a cue.
    # Anchored to the leading "Misspell…" so a character SAYING the word
    # mid-sentence ("You misspelled my name!") is untouched.
    r"|^\s*misspell\w*\b"
    # The model talking ABOUT its translation task instead of translating —
    # a measured run shipped "(No translation needed as it's just an
    # interjection)" and "…all translations are accurate and follow the
    # rules provided." as subtitle cues. No character ever says these; the
    # word "translation(s)" beside accuracy/rules/needed vocabulary is the
    # model's contract language leaking through a batch reply.
    r"|\bno\s+translation\s+(?:needed|necessary|required)\b"
    r"|\btranslations?\s+(?:are|is)\s+accurate\b"
    r"|\bfollow\s+the\s+rules\s+provided\b"
    r"|\bas\s+it'?s\s+just\s+an\s+interjection\b")
# A cue that IS a bare title/episode card readout and nothing else. At most
# one short word or a number may follow ("Title Strange.", "Episode 1",
# "Episode") — a sentence that merely STARTS near these words has more
# after it and falls through.
_TITLE_STUB_RE = re.compile(
    r"(?i)^(?:(?:title|episode)(?:\s+(?:\w{1,12}|\d+))?"
    # Eyecatch card readouts ("Part 1" — a measured run shipped "Part" and
    # "1" as two dialogue cues). Only number-ish continuations, so a real
    # imperative like "Part ways." is never touched.
    r"|part(?:\s+(?:\d+|one|two|three|i{1,3}))?)\s*[.!?…]?$")
# A whole cue that is the model INTRODUCING its answer rather than the
# answer ("Here's the translated subtitle line in English:") — a measured
# run shipped exactly that as cue text, with the actual translation in the
# next cue. Anchored to the announcement shape; dialogue never says this.
_PREAMBLE_CUE_RE = re.compile(
    r"(?i)^here'?s (?:the|your)[^.!?]{0,60}"
    r"(?:translat|subtitle|english)[^.!?]{0,20}:?\s*$")
# OP/ED credit-card readouts Whisper lifts off the screen and the
# translator dutifully renders ("Lyrics by…", "…Composition and…",
# "…arrangement by Initial composition Miku" — all shipped on a measured
# run). Nobody SAYS these; they are 作詞/作曲/編曲 cards under the song.
_CREDITS_RE = re.compile(
    r"(?i)\b(?:lyrics|music|vocals|arrangement|screenplay|storyboard)\s+by\b"
    r"|\bcomposition\s+and\b|\bcomposed\s+and\s+arranged\b"
    r"|\binitial\s+composition\b")
# Structured-output crumbs glued to a cue's tail ("…now. }`[") — the words
# are real, the brackets are the batch reply's JSON leaking through.
_CRUMB_TAIL_RE = re.compile(r"[\s{}\[\]`]*[{}\[\]`]+\s*$")
# The 次回予告 card readout and its stutter: "Next time preview Part 1
# Part 2 Part 2 Part 2" shipped as a cue. The reference's own "Next
# Episode" / "Next, on Gundam Wing, Episode 2." carry no "preview" and no
# Part-run, and survive.
_PREVIEW_STUB_RE = re.compile(
    r"(?i)^next\s+(?:time|episode)\s+preview\b|(?:\bpart\s+\d+\s*){3,}$")

# ``key:value`` tokens — subtitle-file metadata, never spoken English. The
# right side must start alphanumeric and carry no further colon, so a clock
# time ("10:30") and a ratio ("3:1") are both excluded by the left side
# needing to be a word.
_KV_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*:[A-Za-z0-9][A-Za-z0-9_.\-]*$")
_BOILERPLATE_RE = re.compile(
    r"https?://|www\.|\bamara\.org\b|\bopensubtitles\b|\bsubtitles?\s+by\b"
    r"|\bsync(?:ed|hronized)?\s+(?:and\s+corrected\s+)?by\b"
    r"|\btranscri(?:bed|ption)\s+by\b|\bcorrected\s+by\b",
    re.IGNORECASE)
# Laughter, which professional tracks annotate ("[laughs]") rather than
# transcribe, and pure vocalizations. Matched against the normalized form.
_LAUGH_TOKEN_RE = re.compile(r"^(?:h+[ae]+)+h*$|^lol$")
_VOCALIZATION = {
    "uh", "um", "umm", "er", "err", "mm", "mmm", "hmm", "hm", "ah", "aah",
    "ahh", "oh", "ooh", "eh", "ehh", "gah", "guh", "ugh", "argh", "agh",
    "wah", "hmph", "tsk", "grunt", "groan", "sigh",
}


# Stage-direction prose the ASR/LLM emits when it narrates the soundtrack
# instead of transcribing it. Whisper writes non-speech events as parenthesized
# annotations ("(音楽)", "(効果音)"); those use FULL-WIDTH parentheses, so
# ``is_subtitle_marker`` — which requires square brackets — never held them out
# of translation, and they came back as English prose and shipped as dialogue.
_ANNOTATION_PARENS = (("(", ")"), ("（", "）"), ("〈", "〉"), ("《", "》"))
_ANNOTATION_WORDS = frozenset({
    "dialogue", "dialog", "music", "sound", "effect", "effects", "sfx",
    "silence", "noise", "audio", "bgm", "narration", "subtitle", "subtitles",
})
_ANNOTATION_VERBS = frozenset({
    "start", "starts", "started", "starting", "begin", "begins", "beginning",
    "end", "ends", "ended", "ending", "stop", "stops", "resume", "resumes",
    "continue", "continues", "fade", "fades", "playing", "plays", "over",
})
# Qualifiers the translator puts in FRONT of a soundtrack noun when it is
# narrating structure rather than transcribing speech ("More dialogue", "Final
# line", "Another music cue"). Admitted only alongside a real annotation word,
# so "more" and "final" on their own convict nothing.
_ANNOTATION_QUALIFIERS = frozenset({
    "more", "final", "last", "first", "next", "another", "further",
    "additional", "the", "a", "an", "some", "no",
})
# The same prose welded onto the END of a real line — "…in space colonies
# Dialogue end." A measured run shipped exactly that, and because the cue also
# carries genuine dialogue neither the whole-cue test nor the leading-tag strip
# could touch it. Anchored to the end and requiring the soundtrack noun, so an
# ordinary sentence that happens to finish on "…the music ends" is not at risk
# unless it stands alone as a tag (needs the noun AND a structural verb AND no
# terminal punctuation before it).
_TRAILING_ANNOTATION_RE = re.compile(
    r"(?<=[a-z0-9\)\]])\s+"
    r"(?:more\s+|final\s+|last\s+|another\s+)?"
    r"(?:dialogue|dialog|music|narration|sound\s+effects?|sfx)\s+"
    r"(?:start|starts|started|begin|begins|end|ends|ended|over|resumes?)\s*[.!?…]*\s*$",
    re.IGNORECASE)
# A parenthesized tag hung off the END of a real line — "Brrr (sound effect)".
# Capped at four words for the same reason the whole-cue test is: parentheses
# are also a legitimate convention for whispered or aside speech, and a
# whispered sentence is dialogue.
_TRAILING_PAREN_TAG_RE = re.compile(
    r"\s*[\(（]\s*(?:\S+\s+){0,3}\S+\s*[\)）]\s*[.!?…]*\s*$")
_ANNOTATION_PREFIX_RE = re.compile(
    r"^\s*[\(（]\s*([^)）]{1,24})\s*[\)）]\s*(?=\S)")


def _paren_wrapped(t: str):
    for lo, hi in _ANNOTATION_PARENS:
        if t.startswith(lo) and t.endswith(hi) and len(t) > 2:
            return t[1:-1].strip()
    return None


def looks_like_annotation_artifact(text: str) -> bool:
    """True when the WHOLE cue is a soundtrack annotation, not speech.

    Two shapes, both measured on a real run: a fully parenthesized tag
    ("(Alarm sound)", "(音楽)"), and bare stage-direction prose the translator
    produced from one ("Dialogue end", "Music starts", "Sound effect"). The
    prose form is why a text-only filter is needed at all — by the time it
    reaches the subtitle track it has no punctuation, no brackets and no other
    tell, and ``looks_like_asr_boilerplate`` correctly declines to convict it.

    Deliberately narrow: the prose branch fires only when EVERY word comes from
    a closed two-part vocabulary (a soundtrack noun plus an optional
    start/stop verb) and the cue is at most six words. "Music to my ears" and
    "The sound of it" both survive, because ``to``/``my``/``ears``/``the``/``of``
    are in neither set. The cap is six, not three, because a measured run fused
    TWO annotations into one cue — "Dialogue end. Sound effect" — which sat
    over the 3-word cap here, survived the junk filter, and was then cut into
    two separate junk cues by the run-on splitter downstream. All-words-in-
    vocabulary is the load-bearing test; the cap only bounds the scan. Pure
    and deterministic."""
    t = (text or "").strip()
    if not t or t.startswith("["):
        # Square brackets are the caption-marker namespace, and "[♪ music ♪]"
        # would otherwise convict on the word test. Markers are normalized and
        # deduped by their own pass; this one must not reach into them.
        return False
    inner = _paren_wrapped(t)
    if inner is not None and "(" not in inner:
        # Short tags only. Parentheses are also a legitimate subtitle
        # convention for whispered or aside speech, and a whispered SENTENCE
        # is real dialogue — "(He whispers something and leaves)" must not go
        # the same way as "(Alarm sound)". CJK annotations are exempt from the
        # cap: "(音楽)" has no spaces to count and is never whispered speech.
        _w = [w for w in re.split(r"\s+", inner) if w]
        if len(_w) <= 4 or not any(ch.isascii() and ch.isalpha() for ch in inner):
            return True
    words = [w for w in re.split(r"[\s\W_]+", t.lower()) if w]
    if not words or len(words) > 6:
        return False
    if not any(w in _ANNOTATION_WORDS for w in words):
        return False
    return all(w in _ANNOTATION_WORDS or w in _ANNOTATION_VERBS
               or w in _ANNOTATION_QUALIFIERS for w in words)


def strip_trailing_annotation(text: str) -> str:
    """Remove a soundtrack annotation welded onto the END of a real line.

    The translator narrates structure as well as speech, and when it does so
    mid-cue neither the whole-cue test nor the leading-tag strip can reach it:
    a measured run shipped a line of genuine narration with "Dialogue end."
    fused onto its tail. The dialogue is real and stays; only the tag goes.
    Returns ``text`` unchanged when stripping would empty the cue."""
    t = (text or "").strip()
    if not t:
        return text
    out = _TRAILING_ANNOTATION_RE.sub("", t).strip()
    if out == t:
        out = _TRAILING_PAREN_TAG_RE.sub("", t).strip()
    return out if out and out != t else text


def strip_annotation_prefix(text: str) -> str:
    """Remove a leading ``(tag)`` from a cue that also carries real dialogue.

    The roster resolver can mint junk mappings — one run turned a katakana
    token into the word "Emotion" — after which cues shipped as
    "(Emotion) Come on! Hurry up!". The dialogue is real and must be kept; only
    the tag goes. A cue that is NOTHING but a tag is left alone here and
    convicted by :func:`looks_like_annotation_artifact` instead, so the two
    never fight over the same cue."""
    t = (text or "").strip()
    if not t or _paren_wrapped(t) is not None:
        return text
    out = _ANNOTATION_PREFIX_RE.sub("", t, count=1)
    return out.strip() if out.strip() else text


def looks_like_asr_boilerplate(text: str) -> bool:
    """True when ``text`` is subtitle-file metadata the ASR hallucinated.

    Whisper trained on scraped subtitle corpora and it reproduces their
    headers and credits when the audio underneath is thin. A measured run
    shipped ``] sync:20 plain:no-commentary The`` as a visible subtitle at
    3:30 — a string that appears nowhere in this codebase or its logs and
    that no character says.

    Two independent signatures, either sufficient:
      * a credit line or a URL — ``subtitles by``, ``synced and corrected
        by``, ``amara.org``, any ``http``;
      * two or more ``word:value`` tokens. English dialogue does not carry
        two colon-joined tokens in one cue; a subtitle header always does.
        One is not enough, because a single one can survive a mis-split.

    Pure and deterministic. Narrow on purpose: a cue that merely contains a
    colon, a clock time, or one stray bracket is left alone."""
    t = (text or "").strip()
    if not t:
        return False
    if _BOILERPLATE_RE.search(t):
        return True
    return sum(1 for w in t.split() if _KV_TOKEN_RE.match(w)) >= 2


def drop_junk_cues(rows: list, vocalization_max_dwell_s: float = 2.5,
                   ) -> tuple[list, list]:
    """Remove cues that carry no information a viewer could use.

    Deliberately NARROW. A measured run's stray cues included "*Grunt*
    *grunt*" held 4.5 s and "Ha ha" held 5.8 s over a battle beat whose
    three real lines were missing — but the same run also shipped "Hey!",
    "Huh?" and "What?", every one of which the professional reference
    ALSO captions. Dropping by shortness would have thrown away real
    dialogue, so only two classes go:

      * asterisk roleplay markup, which is never subtitle text; and
      * laughter / bare vocalizations ("Ah", "Hmph") that sit on screen
        longer than ``vocalization_max_dwell_s`` — a real interjection is
        brief, so a multi-second one is decode residue filling a hole.

    Real words keep their place regardless of length. Returns
    ``(rows, dropped_samples)``."""
    def _g(r, k, d=None):
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    kept, dropped = [], []
    for r in rows or []:
        txt = (_g(r, "text", "") or "").strip()
        if not txt:
            kept.append(r)
            continue
        # A junk tag at either end never costs the cue its dialogue.
        _stripped = strip_trailing_annotation(strip_annotation_prefix(txt))
        if _stripped != txt:
            txt = _stripped
            if isinstance(r, dict):
                r["text"] = txt
            else:
                setattr(r, "text", txt)
        if looks_like_annotation_artifact(txt):
            dropped.append(f"annotation:{txt[:32]!r}")
            continue
        # Bracketed markers are caption furniture, exempt from every test
        # below. They are checked for annotation prose FIRST, though: the
        # bracket is what let "[Music]" through every filter in the chain.
        if txt.startswith("["):
            kept.append(r)
            continue
        # Crumb strip AFTER the marker exemption, so a marker's closing "]"
        # can never be eaten; a non-marker cue ending in stray JSON/fence
        # characters keeps its words and loses the crumbs.
        _decrumbed = _CRUMB_TAIL_RE.sub("", txt).strip()
        if _decrumbed != txt:
            if not _decrumbed:
                dropped.append(f"crumb:{txt[:32]!r}")
                continue
            txt = _decrumbed
            if isinstance(r, dict):
                r["text"] = txt
            else:
                setattr(r, "text", txt)
        try:
            dur = float(_g(r, "end", 0.0) or 0.0) - float(_g(r, "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            dur = 0.0
        if _ASTERISK_MARKUP_RE.match(txt):
            dropped.append(f"markup:{txt[:32]!r}")
            continue
        if looks_like_asr_boilerplate(txt):
            dropped.append(f"boilerplate:{txt[:40]!r}")
            continue
        # Meta leaks: translation-note prose the LLM emitted ABOUT its work
        # ("Mrs. Ifc, check translation.") and bare title/episode stubs the
        # ASR reads off an on-screen card ("Title Strange.", "Episode 1") —
        # a measured run shipped all three as subtitles. Both patterns are
        # anchored tightly: "Next Episode" and "Next, on Gundam Wing,
        # Episode 2." in the professional reference start with real words
        # and survive; a cue that IS only a title stub does not.
        if _META_NOTE_RE.search(txt):
            dropped.append(f"meta:{txt[:40]!r}")
            continue
        if _PREAMBLE_CUE_RE.match(txt):
            dropped.append(f"preamble:{txt[:40]!r}")
            continue
        if _CREDITS_RE.search(txt):
            dropped.append(f"credits:{txt[:40]!r}")
            continue
        if _TITLE_STUB_RE.match(txt):
            dropped.append(f"title-stub:{txt[:32]!r}")
            continue
        if _PREVIEW_STUB_RE.search(txt):
            dropped.append(f"preview-stub:{txt[:40]!r}")
            continue
        # A single Latin letter is never a subtitle (except "I"): a measured
        # run held a bare "S" on screen for 2.5 seconds — decode residue from
        # a music sting. CJK is exempt (one character is a real word there).
        _bare = txt.rstrip(".!?…")
        if (len(_bare) == 1 and _bare[0].isascii()
                and (_bare[0].isdigit()
                     or (_bare[0].isalpha() and _bare[0].upper() != "I"))):
            # A single Latin letter or digit is never a subtitle (except
            # "I"): measured runs held a bare "S" for 2.5s and a bare "1"
            # (the second half of a "Part 1" eyecatch) as dialogue cues.
            dropped.append(f"letter:{txt!r}")
            continue
        norm = _echo_norm(txt)
        if norm and dur > vocalization_max_dwell_s:
            words = norm.split()

            def _is_vocal(w: str) -> bool:
                if w in _VOCALIZATION:
                    return True
                # Elongated screams: "aaah", "aaahh", "nooo" are the same
                # vocalizations stretched — collapse letter runs both ways
                # before giving up (a measured run shipped "AAAH" for 3.8s
                # and "AAAHH" for 3.9s past the exact-match set).
                w2 = re.sub(r"(.)\1{2,}", r"\1\1", w)   # 3+ → 2
                w1 = re.sub(r"(.)\1+", r"\1", w)        # 2+ → 1
                return w2 in _VOCALIZATION or w1 in _VOCALIZATION

            _laugh = words and all(_LAUGH_TOKEN_RE.match(w) for w in words)
            if _laugh or (len(words) <= 2 and all(_is_vocal(w) for w in words)):
                dropped.append(f"vocalization:{txt[:32]!r} ({dur:.1f}s)")
                continue
        kept.append(r)
    return (kept, dropped) if dropped else (rows, [])


# ── Audio-keyed theme collapse ─────────────────────────────────────────────

def collapse_theme_by_music_spans(segments, music_spans: list,
                                  target_lang: str = "en",
                                  head_s: float = 120.0,
                                  tail_s: float = 240.0,
                                  min_span_s: float = 20.0,
                                  min_cues: int = 3,
                                  raw_music: list = None):
    """Collapse a sung theme using the AUDIO, not the translation's wording.

    ``collapse_song_choruses`` infers a theme from the text: chorus repetition
    plus punctuation and capitalization heuristics. That inference depends on
    how the LLM happened to render the lyrics on a given run, and it is not
    stable — across measured runs the same episode's opening theme collapsed
    correctly three times and shipped as dialogue three times, with nothing
    changed but the translation's phrasing. Text is the wrong evidence for a
    question the audio already answers.

    The spectral classifier labels sustained MUSIC-ONLY regions; dialogue over
    a score is classified speech, not music, so a long music span is positive
    evidence that whatever cues sit inside it are sung, not spoken. Any run of
    at least ``min_cues`` cues lying inside such a span, in the opening or
    closing window, becomes one ``[♪ … theme ♪]`` marker regardless of what
    the words say.

    Preview narration over the ending theme is preserved: a preview cue splits
    the run rather than being swallowed by it. Deterministic and fail-soft —
    no spans (classifier unavailable, no sustained music) returns the rows
    untouched, and the text-based pass still gets its turn afterwards.

    ``raw_music`` (unbridged classifier fragments, from
    ``raw_music_spans_cached``) enables the TWO-SIGNAL tier. A sung theme
    defeats the strong tier structurally — the classifier reads sung vocals
    as speech, so the theme's own voice blocks its bridge and no OP/ED span
    ever qualifies (measured: 205 speech blockers; the only qualified spans
    were mid-episode instrumentals). The weak tier collapses a run only when
    two independently-unreliable signals AGREE: a dense, contiguous run of
    lyric-shaped cues in the opening/closing window (text — which alone
    flip-flops with the LLM's punctuation habits: the same ED collapsed on
    one run and shipped 14 lyric cues as dialogue on the next), whose span
    the raw music fragments still cover ≥30% (audio — which alone would weld
    score under dialogue, the measured 142-209s failure). Dialogue survives
    both ways: a scene's proper nouns and pauses fail the run test, and
    score under speech classifies as speech so its coverage stays low.
    Returns ``(rows, changed)``."""
    try:
        tgt = (target_lang or "").strip().lower().split("-")[0]
        rows = _as_rows(segments)
        if tgt in _CJK_TARGETS or len(rows) < 4:
            return rows, False
        if not music_spans and not raw_music:
            return rows, False
        music_spans = music_spans or []
        try:
            from backend.config import settings as _s
            if not getattr(_s, "TRANSCRIPT_MARK_THEME_SONGS", True):
                return rows, False
        except Exception:
            pass

        def _st(r):
            return float(r.get("start") or 0.0)

        def _en(r):
            return float(r.get("end") or 0.0)

        track_end = max((_en(r) for r in rows), default=0.0)
        spans = []
        for sp in music_spans:
            try:
                a, b = float(sp[0]), float(sp[1])
            except (TypeError, ValueError, IndexError):
                continue
            if b - a < min_span_s:
                continue
            # Only the opening and closing windows. A sustained music cue in
            # the middle of an episode is score under a scene, not a theme.
            # ``head_s`` is two minutes, not three: on the reference episode
            # the opening theme is done by 1:32 and the narration that follows
            # is captioned dialogue. A span that STARTS after two minutes is
            # score under a scene however long it runs, and a measured run
            # deleted sixty-five seconds of that narration when this window
            # reached far enough to admit it.
            if a <= head_s or a >= max(0.0, track_end - tail_s):
                spans.append((a, b))
        if not spans and not raw_music:
            return rows, False

        drop, markers = set(), []
        for a, b in sorted(spans):
            inside = []
            for i, r in enumerate(rows):
                if i in drop:
                    continue
                txt = (r.get("text") or "").strip()
                if not txt or txt.startswith("["):
                    continue
                mid = (_st(r) + _en(r)) / 2.0
                if a - 0.5 <= mid <= b + 0.5:
                    inside.append(i)
            if len(inside) < min_cues:
                continue
            # A next-episode preview narrated over the ending theme is real
            # content — cut the run at it instead of swallowing it.
            run = []
            for i in inside:
                if _PREVIEW_RE.search((rows[i].get("text") or "")):
                    break
                run.append(i)
            if len(run) < min_cues:
                continue
            m_start = min(_st(rows[i]) for i in run)
            m_end = max(_en(rows[i]) for i in run)
            if m_end - m_start < _THEME_MIN_SPAN_S:
                continue
            label = (_THEME_OPEN_LABEL if m_start <= head_s
                     else _THEME_END_LABEL)
            drop.update(run)
            markers.append({"start": round(m_start, 3), "end": round(m_end, 3),
                            "text": label, "speaker": ""})

        # ── Two-signal tier ──────────────────────────────────────────────
        raw = []
        for sp in (raw_music or []):
            try:
                raw.append((float(sp[0]), float(sp[1])))
            except (TypeError, ValueError, IndexError):
                continue

        def _raw_cover(a: float, b: float) -> float:
            got = 0.0
            for x, y in raw:
                got += max(0.0, min(b, y) - max(a, x))
            return got / (b - a) if b > a else 0.0

        if raw:
            first_t = min((_st(r) for r in rows), default=0.0)
            for w0, w1, label in ((first_t, first_t + head_s,
                                   _THEME_OPEN_LABEL),
                                  (max(0.0, track_end - tail_s), track_end,
                                   _THEME_END_LABEL)):
                cand = []
                for i, r in enumerate(rows):
                    if i in drop:
                        continue
                    txt = (r.get("text") or "").strip()
                    if (not txt or txt.startswith("[")
                            or _PREVIEW_RE.search(txt)
                            or _proper_noun_count(txt) > 2
                            or len(txt) > 60):
                        continue
                    mid = (_st(r) + _en(r)) / 2.0
                    if w0 - 0.01 <= mid <= w1 + 0.01:
                        cand.append(i)
                groups, cur = [], []
                for i in cand:
                    if cur and _st(rows[i]) - _en(rows[cur[-1]]) > 10.0:
                        groups.append(cur)
                        cur = []
                    cur.append(i)
                if cur:
                    groups.append(cur)
                for grp in groups:
                    # Stricter than the strong tier on every text axis,
                    # because the audio here is only fragments: at least 5
                    # cues, a continuous span, a proper-noun-free majority
                    # (a dialogue scene names people; a song does not), and
                    # ≥30% of the span still classified music despite the
                    # vocals. Punctuation is deliberately NOT consulted —
                    # the LLM's habit of punctuating lyrics as sentences is
                    # exactly the instability this tier exists to survive.
                    if len(grp) < max(min_cues, 5):
                        continue
                    g_start = min(_st(rows[i]) for i in grp)
                    g_end = max(_en(rows[i]) for i in grp)
                    if g_end - g_start < min_span_s:
                        continue
                    _pn0 = sum(
                        1 for i in grp
                        if _proper_noun_count(
                            (rows[i].get("text") or "").strip()) == 0)
                    if _pn0 < 0.6 * len(grp):
                        logger.info(
                            "theme collapse (two-signal): %s group %.1f-%.1fs "
                            "rejected — %d/%d cues carry proper nouns",
                            label, g_start, g_end,
                            len(grp) - _pn0, len(grp))
                        continue
                    cover = _raw_cover(g_start, g_end)
                    # A LONG lyric-shaped run earns a lower coverage floor
                    # in the CLOSING window: a vocal-dense ED classifies as
                    # speech nearly wall-to-wall (the vocals ARE the song),
                    # so its instrumental intro/bridges may be all the music
                    # the classifier ever sees — a measured run shipped a
                    # 14-cue, 55-second ED as dialogue with every text test
                    # passing and only the 30% floor in the way. Dialogue
                    # scenes don't produce 8+ contiguous pn-free cues over
                    # 40+ seconds with ANY music coverage in the tail.
                    _floor = (0.15 if (label == _THEME_END_LABEL
                                       and len(grp) >= 8
                                       and g_end - g_start >= 40.0)
                              else 0.30)
                    if cover < _floor:
                        logger.info(
                            "theme collapse (two-signal): %s group %.1f-%.1fs "
                            "rejected — %d cue(s), music fragments cover "
                            "%.0f%% (< %.0f%% floor)",
                            label, g_start, g_end, len(grp),
                            cover * 100.0, _floor * 100.0)
                        continue
                    logger.info(
                        "theme collapse (two-signal): %s — %d lyric-shaped "
                        "cue(s) %.1f-%.1fs, music fragments cover %.0f%%",
                        label, len(grp), g_start, g_end, cover * 100.0)
                    drop.update(grp)
                    markers.append({"start": round(g_start, 3),
                                    "end": round(g_end, 3),
                                    "text": label, "speaker": ""})
        if not markers:
            return rows, False
        out = [r for i, r in enumerate(rows) if i not in drop] + markers
        out.sort(key=lambda r: (float(r.get("start") or 0.0),
                                float(r.get("end") or 0.0)))
        return dedupe_theme_markers(out), True
    except Exception:
        return _as_rows(segments), False


def drop_recovered_near_theme_markers(rows: list, before_s: float = 45.0,
                                      after_s: float = 90.0) -> tuple:
    """Drop RECOVERED cues that land inside a collapsed song's region.

    The post-COMPLETE recovery merges into the shipped track AFTER the theme
    collapse made its one whole-track decision — so a cue it recovers inside
    the opening/ending song walks past every lyric guard and ships. A
    measured run shipped the ED's credit-card readouts and a stray lyric
    fragment exactly this way. The theme marker is the collapse's surviving
    testimony of WHERE the song is; a recovered cue within its window is
    lyric/credit residue by construction — with one exception: preview-shaped
    cues (next-episode narration is genuinely spoken over the ending theme
    and the reference captions it). Scoped by provenance: only rows the
    recovery stamped ``recovered`` are touched. ``(rows, dropped_samples)``."""
    try:
        markers = []
        for r in rows or []:
            txt = ((r.get("text") if isinstance(r, dict)
                    else getattr(r, "text", "")) or "").strip()
            # THEME markers only. A bare mid-episode "[♪ music ♪]" marks
            # score under a scene — real dialogue lives right beside it, and
            # a measured run's guard around one dropped a recovered line at
            # 682s that was plausibly genuine dialogue. Only the opening/
            # ending labels testify to a SONG region.
            if _is_marker(txt) and "♪" in txt and "theme" in txt.lower():
                try:
                    markers.append(float(
                        (r.get("start") if isinstance(r, dict)
                         else getattr(r, "start", 0.0)) or 0.0))
                except (TypeError, ValueError):
                    continue
        if not markers:
            return list(rows or []), []
        kept, dropped = [], []
        for r in rows or []:
            is_rec = bool(r.get("recovered") if isinstance(r, dict)
                          else getattr(r, "recovered", False))
            txt = ((r.get("text") if isinstance(r, dict)
                    else getattr(r, "text", "")) or "").strip()
            if not is_rec or _is_marker(txt):
                kept.append(r)
                continue
            try:
                s = float((r.get("start") if isinstance(r, dict)
                           else getattr(r, "start", 0.0)) or 0.0)
            except (TypeError, ValueError):
                kept.append(r)
                continue
            in_window = any(m - before_s <= s <= m + after_s for m in markers)
            if in_window and not (_PREVIEW_RE.search(txt)
                                  or _proper_noun_count(txt) >= 2):
                dropped.append(f"{s:.1f}s {txt[:36]!r}")
                continue
            kept.append(r)
        return (kept, dropped) if dropped else (list(rows or []), [])
    except Exception:
        return list(rows or []), []


def clamp_theme_marker_starts(segments, first_onset,
                              max_start_s: float = 1.0) -> tuple:
    """Pin a ZERO-ANCHORED theme marker to the first real audio onset.

    Backstop for exactly one pathology: Whisper hallucinating a lyric line
    at 0:00.000, which then becomes the collapse run's earliest member and
    drags the marker to zero. A hallucinated cue cannot conjure sound, so a
    marker at ~0:00 moves up to the classifier's first onset.

    Deliberately restricted to markers starting under ``max_start_s`` — on
    its first production outing an unrestricted version MOVED A CORRECT
    MARKER: the text pass had anchored the opening theme at 30.342s (right)
    and the classifier's first onset was 61.0s (its silence floor slept
    through the song's quiet intro), so the clamp shoved the marker half a
    minute late. The cue evidence is MORE accurate than the onset whenever
    the cue evidence is sane; the onset only outranks a cue that claims the
    impossible. ``(rows, notes)``; no onset is a no-op."""
    rows = _as_rows(segments)
    notes: list = []
    try:
        onset = float(first_onset) if first_onset is not None else None
    except (TypeError, ValueError):
        onset = None
    if onset is None or onset <= 1.0:
        return rows, notes
    changed = False
    for r in rows:
        txt = (r.get("text") or "").strip()
        if not (_is_marker(txt) and "♪" in txt):
            continue
        try:
            s = float(r.get("start") or 0.0)
            e = float(r.get("end") or s)
        except (TypeError, ValueError):
            continue
        if s >= max_start_s or s >= onset - 1.0:
            continue
        dur = max(e - s, 1.5)
        r["start"] = round(onset, 3)
        r["end"] = round(max(e, onset + min(dur, 4.0)), 3)
        notes.append(f"{txt} {s:.3f}s → {onset:.3f}s (first audio onset)")
        changed = True
    if changed:
        rows.sort(key=lambda r: (float(r.get("start") or 0.0),
                                 float(r.get("end") or 0.0)))
    return rows, notes


def normalize_markers(rows: list, adjacent_window_s: float = 20.0) -> tuple:
    """Fold bare ASR music tags onto the styled marker, then drop a marker that
    merely repeats the one before it.

    Whisper writes ``[Music]`` on its own. Because it is bracketed it satisfies
    ``is_subtitle_marker``, is held out of translation, and is re-inserted
    verbatim — so a measured run shipped a cue reading ``[Music]`` 2.3 seconds
    after a cue reading ``[♪ music ♪]``, two spellings of the same fact, one of
    them untranslated-looking. Folding first is what makes the dedup possible:
    before normalization the two strings share no useful similarity, and no
    ratio threshold that catches them is safe on real dialogue.

    The adjacency test is deliberate — two music markers far apart in an
    episode are two different musical passages and both belong. Returns
    ``(rows, notes)``."""
    try:
        from backend.services.audio_analyzer import normalize_music_marker
    except Exception:
        return rows, []

    def _g(r, k, d=None):
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    def _set(r, k, v):
        if isinstance(r, dict):
            r[k] = v
        else:
            setattr(r, k, v)

    kept, notes, last_label, last_end = [], [], None, None
    for r in rows or []:
        txt = (_g(r, "text", "") or "").strip()
        norm = normalize_music_marker(txt)
        if norm != txt:
            notes.append(f"folded:{txt[:24]!r}")
            _set(r, "text", norm)
            txt = norm
        if not txt.startswith("["):
            last_label = None
            kept.append(r)
            continue
        try:
            st = float(_g(r, "start", 0.0) or 0.0)
            en = float(_g(r, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            kept.append(r)
            continue
        if (txt == last_label and last_end is not None
                and st - last_end <= adjacent_window_s):
            notes.append(f"adjacent:{txt[:24]!r}@{st:.2f}s")
            last_end = max(last_end, en)
            continue
        last_label, last_end = txt, en
        kept.append(r)
    return (kept, notes) if notes else (rows, [])


def dedupe_theme_markers(rows: list) -> list:
    """Keep only the FIRST marker of each theme label.

    The audio-keyed collapse and the text-keyed one both emit theme markers,
    and they do not agree on where the theme is. A measured run shipped
    ``[♪ Opening theme ♪]`` twice — once at 0:30 from the text pass, which had
    it right, and again at 2:22 from the audio pass, which did not. An episode
    has one opening theme and one ending theme, so a second marker carrying
    the same label is by construction the wrong one.

    Earliest wins for both labels: the opening theme is the first music in the
    episode, and the ending theme's marker is placed at the run's start, so a
    later duplicate is always a stray. Pure; order-preserving."""
    seen, out = set(), []
    for r in rows or []:
        txt = (r.get("text") if isinstance(r, dict)
               else getattr(r, "text", "") or "")
        label = (txt or "").strip()
        if label in (_THEME_OPEN_LABEL, _THEME_END_LABEL):
            if label in seen:
                continue
            seen.add(label)
        out.append(r)
    return out


# ── Repetition bursts ──────────────────────────────────────────────────────

def drop_repetition_bursts(rows: list, min_run: int = 3,
                           max_cue_s: float = 0.833,
                           lookback_s: float = 240.0,
                           containment: float = 0.5,
                           ) -> tuple[list, list]:
    """Delete a packed run of sub-minimum cues that re-states earlier content.

    ``suppress_echo_cues`` compares a cue to its NEIGHBOURS inside a twelve
    second window, which is the right scope for two decodes of the same
    moment. It cannot see the other failure shape: a measured run shipped six
    consecutive cues across seven seconds at 3:30 that re-stated the opening
    narration from more than a minute earlier, each one a quarter to
    two-thirds of a second long. Content that already aired, re-emitted in a
    burst too fast to read, is decode residue however far back the original
    sits.

    The signature is the conjunction, and every part of it is load-bearing:
      * at least ``min_run`` CONSECUTIVE cues, each under ``max_cue_s`` — real
        dialogue is not delivered as a stream of sub-minimum flashes;
      * the run's content is largely (``containment``) already present in the
        cues BEFORE it, within ``lookback_s`` — measured on stemmed CONTENT
        words, so the shared vocabulary of ordinary dialogue does not count;
      * the run carries at least five distinct content stems, so a rapid
        exchange of interjections can never be convicted on thin evidence.

    Either half alone is innocent. A few short cues in a row happen in rapid
    exchanges, and a line can legitimately echo an earlier one ("So it WAS a
    Gundam"). Only together do they identify the burst. Markers are never
    part of a run. Returns ``(rows, dropped_samples)``."""
    def _g(r, k, d=None):
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    n = len(rows or [])
    if n < min_run + 1:
        return rows, []

    def _dur(r):
        try:
            return float(_g(r, "end", 0.0) or 0.0) - float(_g(r, "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _short(i):
        txt = (_g(rows[i], "text", "") or "").strip()
        if not txt or txt.startswith("["):
            return False
        return 0.0 < _dur(rows[i]) < max_cue_s

    drop, dropped = set(), []
    i = 0
    while i < n:
        if not _short(i):
            i += 1
            continue
        j = i
        while j + 1 < n and _short(j + 1):
            j += 1
        run = list(range(i, j + 1))
        i = j + 1
        if len(run) < min_run:
            continue
        try:
            run_start = float(_g(rows[run[0]], "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        burst_stems = set()
        for k in run:
            burst_stems |= _echo_stems(_g(rows[k], "text", "") or "")
        if len(burst_stems) < 5:
            # Too little content to convict on. A three-cue run of
            # interjections ("Yes, sir." / "What?!" / "Hurry!") is a rapid
            # exchange, and the reference captions those — the burst has to
            # carry real content words before its overlap means anything.
            continue
        prior_stems = set()
        for k in range(0, run[0]):
            try:
                s0 = float(_g(rows[k], "start", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if run_start - s0 > lookback_s:
                continue
            prior_stems |= _echo_stems(_g(rows[k], "text", "") or "")
        if not prior_stems:
            continue
        hits = sum(
            1 for w in burst_stems
            if any(w == o or (len(w) >= 4 and len(o) >= 4
                              and (w.startswith(o) or o.startswith(w)))
                   for o in prior_stems))
        if hits / float(len(burst_stems)) < containment:
            continue
        drop.update(run)
        dropped.append(
            f"{run_start:.1f}s x{len(run)} "
            f"{(_g(rows[run[0]], 'text', '') or '')[:36]!r}")
    if not drop:
        return rows, []
    return [r for k, r in enumerate(rows) if k not in drop], dropped


def drop_restatement_cues(rows: list, window_cues: int = 5,
                          window_s: float = 30.0,
                          containment: float = 0.6,
                          min_stems: int = 3,
                          recovered_only: bool = True) -> tuple[list, list]:
    """Drop a RECOVERED cue whose content the surrounding cues already carry.

    The third repetition shape, and the one both existing passes miss.
    ``suppress_echo_cues`` needs surface similarity above 0.66 between a PAIR of
    cues; ``drop_repetition_bursts`` needs three consecutive cues under 833 ms.
    A measured run closed on ten cues across seventeen seconds where the
    professional reference has six, four of them restating one idea in four
    different phrasings — durations from 0.334 s to 2.293 s, so no run of three
    short cues existed, and each pairwise similarity sat below the echo gate
    because the translator had reworded rather than repeated.

    Content containment sees it: the cue's stemmed content words are compared
    against the UNION of the preceding ``window_cues`` cues, so four
    reformulations convict where no two of them would.

    ``recovered_only`` is what makes this safe, and it is on by default. Run
    against a whole track the test does not discriminate: on the professional
    reference it deletes 4 genuine lines while catching 4 restatements on the
    measured run — a one-to-one trade against the very track we are trying to
    match, and no threshold tested moves it (at zero-tolerance it catches 0 and
    still costs 1). Restricted to cues stamped by ``merge_recovered``, the same
    test is principled rather than statistical: gap recovery exists to fill
    holes, so a recovered cue that says only what the track already said is
    residue by construction. Reference-style tracks carry no recovered cues and
    are therefore untouchable by this pass.

    Guards, each protecting a real subtitle pattern:
      * a cue needs ``min_stems`` distinct content stems, so "Fire! Fire!!"
        (one stem) and every short exclamation are exempt by construction;
      * a speaker change blocks the comparison — two characters landing on the
        same point is drama;
      * a cue carrying a proper noun the window lacks is elaborating, not
        restating;
      * markers are never touched.

    Returns ``(rows, dropped_samples)``."""
    def _g(r, k, d=None):
        return r.get(k, d) if isinstance(r, dict) else getattr(r, k, d)

    n = len(rows or [])
    if n < 2:
        return rows, []
    stems = [_echo_stems(_g(r, "text", "") or "") for r in rows]
    drop, dropped = set(), []
    for i in range(1, n):
        txt = (_g(rows[i], "text", "") or "").strip()
        if not txt or txt.startswith("[") or len(stems[i]) < min_stems:
            continue
        if recovered_only and not _g(rows[i], "recovered", False):
            continue
        try:
            si = float(_g(rows[i], "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        window: set = set()
        seen = 0
        for j in range(i - 1, -1, -1):
            if j in drop:
                continue
            tj = (_g(rows[j], "text", "") or "").strip()
            if not tj or tj.startswith("["):
                continue
            try:
                ej = float(_g(rows[j], "end", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if si - ej > window_s:
                break
            if (_g(rows[j], "speaker", "") or "") != (_g(rows[i], "speaker", "") or ""):
                continue
            window |= stems[j]
            seen += 1
            if seen >= window_cues:
                break
        if not window:
            continue
        hits = sum(
            1 for w in stems[i]
            if any(w == o or (len(w) >= 4 and len(o) >= 4
                              and (w.startswith(o) or o.startswith(w)))
                   for o in window))
        if hits / float(len(stems[i])) < containment:
            continue
        # Information the window does not already have is never redundant.
        prior_txt = " ".join(
            (_g(rows[j], "text", "") or "")
            for j in range(max(0, i - window_cues * 2), i) if j not in drop)
        if _unique_proper_nouns(txt, prior_txt):
            continue
        drop.add(i)
        dropped.append(f"{si:.1f}s {txt[:40]!r}")
    if not drop:
        return rows, []
    return [r for k, r in enumerate(rows) if k not in drop], dropped
