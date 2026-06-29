"""AI-powered transcript polishing — proper noun + punctuation + filler cleanup.

This replaces the inert ``compat_stubs.correct_transcript`` pass-through.
It batches transcript segments through the editorial LLM, asking it to:

  - Fix proper nouns (capitalisation, consistent spelling across segments)
  - Fix punctuation (add missing periods, commas, question marks)
  - Remove filler words (um, uh, like, you know, basically, literally)
  - Merge / split sentence fragments that span segment boundaries
  - Fix homophones (their/there/they're, your/you're, its/it's)

Timing (``start_sec`` / ``end_sec``) and speaker labels are preserved
byte-for-byte — only the text changes. Failures degrade silently: any
batch that the LLM bungles is dropped back to the original text rather
than letting the entire job crash.

The module is language-aware:
  - ``en`` / Latin scripts → full clean-up
  - ``ja`` / ``ko`` / ``zh`` (CJK) → punctuation kept minimal, particles
    untouched, smaller batches because CJK text is dense per token
"""

import asyncio
import json
import logging
import re
from typing import Any, Callable, Iterable, Optional

from backend.config import settings

logger = logging.getLogger(__name__)


# ── System prompt: persona + invariants ───────────────────────────────────
_SYSTEM_PROMPT = (
    "You are a transcript accuracy editor. You receive Whisper ASR output "
    "from a SMALLER model (e.g. medium) that the user wants polished to "
    "the accuracy of a LARGER model (e.g. large-v3) WITHOUT re-running "
    "transcription. Whisper-medium's typical failure modes:\n"
    "  - Phonetic mis-hears on short common words (e.g. 'Aiken' for "
    "'Okay', 'areas' for 'Aries')\n"
    "  - Mis-spelled proper nouns, especially when the name is unfamiliar "
    "to the model's training data ('Dorian' for 'Darlian', 'Sex' for "
    "'Zechs', 'Aaron' for 'Heero')\n"
    "  - Wrong homophones (their/there, your/you're)\n"
    "  - Punctuation omitted or misplaced\n\n"
    "Your job is to correct these specific errors using surrounding "
    "context, while preserving the speaker's meaning, tone, sentence "
    "structure, and word count EXACTLY.\n\n"
    "STRICT RULES (must follow every time):\n"
    "1. Return EXACTLY the same number of segments you receive. Never "
    "merge or split segments — only edit the text inside each.\n"
    "2. Word count per segment must stay within ±15 % of the input. If "
    "you can't find a high-confidence correction for a token, KEEP THE "
    "ORIGINAL — do not paraphrase, summarise, translate, or compress.\n"
    "3. Substitutions are only allowed when the new word is BOTH "
    "phonetically similar to the original AND clearly correct given the "
    "context (surrounding segments, recurring proper nouns, the show's "
    "setting).\n"
    "4. Never change timing — those fields are not in your output.\n"
    "5. Return ONLY a JSON array of strings, no preamble, no markdown.\n"
    "6. The array length must equal the input segment count.\n"
)

# Translation-polish persona — machine-translation post-editing (MTPE). A
# separate OFFLINE engine (NLLB / Opus-MT) produces the base translation; that
# raw output is often stilted or grammatically rough. The editorial model's job
# here is to POLISH that draft toward natural, professional subtitles — closing
# the quality gap WITHOUT doing the translation itself. It is aggressive on
# fluency/grammar/word-choice but strict on meaning, line count, and timing.
# When the ORIGINAL source line is supplied alongside the draft, the source is
# the ground truth for meaning (so mistranslations can be repaired).
_SYSTEM_PROMPT_TRANSLATION = (
    "You are a professional subtitle localization editor doing machine-"
    "translation post-editing (MTPE). You receive subtitle lines that an OFFLINE "
    "translation engine has already translated into the target language. That raw "
    "draft is often stilted, awkward, or grammatically rough. Rewrite each line "
    "so it reads like a professional human subtitler wrote it — natural, "
    "idiomatic, fluent target language — while preserving the original meaning.\n\n"
    "When the ORIGINAL source line is provided next to the draft, treat the "
    "source as the ground truth for MEANING and use it to repair mistranslations, "
    "dropped words, wrong pronouns/subjects, and garbled names in the draft. You "
    "are EDITING the draft, not translating from scratch.\n\n"
    "You MAY (and should): rephrase awkward machine-translation wording into "
    "natural target language; fix grammar, agreement, articles, tense, and word "
    "order; choose idiomatic vocabulary; fix punctuation and capitalisation; "
    "correct proper nouns; tighten verbose phrasing to subtitle length.\n"
    "You MUST NOT: re-translate into a different language; change the meaning of "
    "a line; add information not in the source/draft; merge or split lines; "
    "reorder lines; or change timing.\n\n"
    "STRICT RULES (must follow every time):\n"
    "1. Return EXACTLY the same number of lines you receive, in order — one "
    "rewritten line per input segment. Never merge or split.\n"
    "2. Each output line must be in the TARGET language only — never leave "
    "source-language words (except proper nouns with no target form).\n"
    "3. Preserve the meaning of every line. When unsure, keep the draft.\n"
    "4. Never change timing — those fields are not in your output.\n"
    "5. Return ONLY a JSON array of strings, no preamble, no markdown.\n"
    "6. The array length must equal the input segment count.\n"
)

# Friendly target-language names for the MTPE prompt (ISO 639-1 → English name).
_LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
    "th": "Thai", "uk": "Ukrainian", "sv": "Swedish", "id": "Indonesian",
    "ms": "Malay", "tl": "Filipino",
}


def _lang_name(code: str) -> str:
    """Friendly language name for prompts ('en' → 'English')."""
    c = (code or "").strip().lower().split("-")[0]
    return _LANG_NAMES.get(c, code or "the target language")

# ── Filler-word patterns by language ──────────────────────────────────────
_EN_FILLERS = re.compile(
    r"\b("
    r"um|uh|umm|uhh|er|erm|mmm|mhm|"
    r"like|you know|i mean|kinda|sorta|"
    r"basically|literally|honestly|actually|"
    r"sort of|kind of"
    r")\b[,]?\s*",
    re.IGNORECASE,
)


# Languages where Western punctuation insertion and filler removal would
# corrupt the text. These get a light-touch polishing path.
_CJK_LANGS = {"ja", "ko", "zh", "zh-cn", "zh-tw", "yue"}


def _coerce_segment(seg) -> dict:
    """Return a dict view of a TranscriptSegment or dict. The polisher
    accepts either shape so it can drop into both the reframer pipeline
    (which passes dicts) and the model-based pipeline (which passes
    Pydantic objects).

    Bare strings are coerced to a stub segment with the string as text
    so a polish pass that returned raw strings instead of segments
    doesn't crash the readability scoring downstream. The compat path
    that previously crashed with ``'str' object has no attribute
    'get'`` (background polishing background task on the 12:48:49 run)
    came from exactly this case.
    """
    if isinstance(seg, dict):
        return seg
    if isinstance(seg, str):
        return {
            "start": 0.0,
            "end": 0.0,
            "text": seg,
            "speaker": "",
            "_obj": seg,
        }
    # Pydantic / dataclass
    return {
        "start": getattr(seg, "start", getattr(seg, "start_sec", 0.0)),
        "end": getattr(seg, "end", getattr(seg, "end_sec", 0.0)),
        "text": getattr(seg, "text", "") or "",
        "speaker": getattr(seg, "speaker", "") or "",
        "_obj": seg,  # round-trip the original object for re-emission
    }


# Punctuation that may be attached to a word surface without being part of the
# spoken token (added/moved by the polish). Stripped for alignment comparison,
# kept on the emitted surface so sentence terminators survive for the segmenter.
_REMAP_STRIP = " \t\r\n.,!?;:…。、！？．・「」『』（）()\"'”’‘“-—–"


def _w_get_attr(word, key, default=None):
    """Read ``key`` off a word that may be a dict or an object."""
    if isinstance(word, dict):
        return word.get(key, default)
    return getattr(word, key, default)


def _remap_words_onto_text(orig_words, new_text, seg_start, seg_end):
    """Re-map original per-word timestamps onto polished ``new_text``.

    The polish is ~1:1 word count by design (it adds punctuation / fixes the odd
    word), so we can carry each original word's start/end onto the matching token
    in the edited text by walking both in order. Tokens with no original
    counterpart (a substituted or inserted word) get their timing interpolated
    between the surrounding anchors so the reconstructed text stays complete and
    the timestamps stay monotonic.

    Returns ``(words, confidence)`` where ``words`` is a list of
    ``{"start","end","word"}`` dicts that, joined, reproduce ``new_text``; or
    ``(None, 0.0)`` when there's nothing to map. ``confidence`` is the fraction
    of original words that found a home in the edited text — the caller drops the
    timing (keeps ``words=[]``) when it's too low to trust.
    """
    txt = new_text or ""
    if not txt.strip() or not orig_words:
        return None, 0.0
    # Extract usable (surface, start, end) triples from the original words.
    o: list[tuple[str, float, float]] = []
    for w in orig_words:
        surf = (_w_get_attr(w, "word", "") or "").strip()
        st = _w_get_attr(w, "start", None)
        en = _w_get_attr(w, "end", None)
        if surf and st is not None and en is not None:
            o.append((surf, float(st), float(en)))
    if not o:
        return None, 0.0

    try:
        from backend.services.subtitle_formatter import _is_cjk
        is_cjk = _is_cjk(txt)
    except Exception:
        is_cjk = False

    lower = txt.lower()
    cursor = 0
    out: list[dict] = []
    matched = 0
    for surf, st, en in o:
        key = surf.strip(_REMAP_STRIP)
        if not key:
            continue
        idx = lower.find(key.lower(), cursor)
        if idx < 0:
            # Original word's surface isn't in the edited text (substituted /
            # dropped) — skip it; its span is absorbed by a neighbour below.
            continue
        end = idx + len(key)
        # Absorb trailing attached punctuation (no intervening space) so a
        # polish-added terminator rides on this token (the segmenter needs it).
        while end < len(txt) and not txt[end].isspace() and txt[end] in _REMAP_STRIP:
            end += 1
        # Any text skipped between the cursor and this match is inserted /
        # substituted content with no source timing — emit it as a token now so
        # the reconstructed text is complete; its timing is interpolated later.
        if idx > cursor:
            gap = txt[cursor:idx].strip()
            if gap:
                out.append({"word": gap, "start": None, "end": None})
        out.append({"word": txt[idx:end], "start": st, "end": en})
        cursor = end
        matched += 1
    if not out:
        return None, 0.0
    # Trailing remainder (e.g. a final added clause) keeps the text whole.
    if cursor < len(txt):
        rest = txt[cursor:].strip()
        if rest:
            out.append({"word": rest, "start": None, "end": None})

    # ── Interpolate timing for tokens that had no original counterpart ──
    lo = float(seg_start) if seg_start is not None else o[0][1]
    hi = float(seg_end) if seg_end is not None else o[-1][2]
    n = len(out)
    i = 0
    while i < n:
        if out[i]["start"] is not None:
            i += 1
            continue
        # Maximal run [i, j) of un-timed tokens.
        j = i
        while j < n and out[j]["start"] is None:
            j += 1
        left_t = out[i - 1]["end"] if i > 0 else lo
        right_t = out[j]["start"] if j < n else hi
        if right_t < left_t:
            right_t = left_t
        span = right_t - left_t
        count = j - i
        for k in range(count):
            a = left_t + span * (k / (count + 1))
            b = left_t + span * ((k + 1) / (count + 1))
            out[i + k]["start"] = a
            out[i + k]["end"] = b
        i = j

    # Clamp into the segment and enforce monotonic, non-negative spans.
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

    confidence = matched / max(1, len(o))
    return out, confidence


def _coerce_word_objs(words):
    """Coerce a list of ``{"start","end","word"}`` dicts to ``WordTimestamp``
    objects so a Pydantic segment stays well-typed. Returns the input unchanged
    on any failure (downstream code reads both shapes)."""
    if not words:
        return words
    try:
        from backend.models import WordTimestamp
        return [
            w if not isinstance(w, dict)
            else WordTimestamp(start=w["start"], end=w["end"], word=w["word"])
            for w in words
        ]
    except Exception:
        return words


def _maybe_remap_words(orig, new_text):
    """Return the word list to attach to a polished segment whose text changed.

    Honors ``POLISH_REMAP_WORD_TIMESTAMPS`` / ``POLISH_REMAP_MIN_CONFIDENCE``.
    Returns ``[]`` (the legacy null-the-words behavior) when remapping is
    disabled, unavailable, or low-confidence."""
    if not bool(getattr(settings, "POLISH_REMAP_WORD_TIMESTAMPS", True)):
        return []
    orig_words = _w_get_attr(orig, "words", None)
    if not orig_words:
        return []
    seg_start = _w_get_attr(orig, "start", _w_get_attr(orig, "start_sec", None))
    seg_end = _w_get_attr(orig, "end", _w_get_attr(orig, "end_sec", None))
    try:
        remapped, conf = _remap_words_onto_text(orig_words, new_text, seg_start, seg_end)
    except Exception:
        return []
    min_conf = float(getattr(settings, "POLISH_REMAP_MIN_CONFIDENCE", 0.5))
    if not remapped or conf < min_conf:
        return []
    return remapped


def _emit_segment(orig, new_text: str):
    """Apply ``new_text`` back to a segment without mutating timing/speaker.
    Round-trips both dict and Pydantic shapes.

    When the text changed, word-level timestamps no longer align char-for-char.
    Rather than nulling them (which wiped nearly all word timing because the
    polish punctuates almost every segment, forcing the downstream segmenter
    onto its char-proportional fallback), RE-MAP the original timestamps onto
    the edited text — keeping ``words=[]`` only when alignment confidence is too
    low. Gated by ``POLISH_REMAP_WORD_TIMESTAMPS``."""
    if isinstance(orig, dict):
        # Some pipelines use ``text``, others might use both ``text`` and
        # carry word-level timestamps. We only rewrite text.
        out = dict(orig)
        out["text"] = new_text
        # Re-map (or, on low confidence, clear) word timestamps when text changed.
        if "words" in out and out.get("text", "") != orig.get("text", ""):
            out["words"] = _maybe_remap_words(orig, new_text)
        return out
    obj = orig
    text_changed = (getattr(obj, "text", None) != new_text)
    new_words = _maybe_remap_words(obj, new_text) if text_changed else None
    # Try to use ``model_copy`` (Pydantic v2) or fall back to attribute set.
    try:
        update = {"text": new_text}
        if text_changed:
            # ``model_copy(update=...)`` does NOT validate, so coerce the
            # remapped dict words into the model's word type to keep the
            # segment well-typed (model_dump / downstream attribute access).
            update["words"] = _coerce_word_objs(new_words)
        new = obj.model_copy(update=update)
        return new
    except Exception:
        try:
            setattr(obj, "text", new_text)
            if text_changed and hasattr(obj, "words"):
                setattr(obj, "words", new_words)
        except Exception:
            pass
        return obj


def _light_filler_strip(text: str, language: str) -> str:
    """Pure-Python filler/whitespace cleanup — used as a baseline pass even
    when the LLM is unavailable, and as a guard inside the LLM path so the
    model isn't asked to remove obvious fillers it might miss."""
    if not text:
        return text
    if not settings.TRANSCRIPT_FILLER_REMOVAL:
        return text
    if language.lower() in _CJK_LANGS:
        # Filler removal in CJK is risky — leave the text alone.
        return text
    cleaned = _EN_FILLERS.sub("", text)
    # Collapse "  ," "  ." artefacts the regex may leave behind.
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # Capitalise the first letter if the original had it.
    if text[:1].isupper() and cleaned[:1].islower():
        cleaned = cleaned[:1].upper() + cleaned[1:]
    return cleaned


def _build_user_prompt(
    batch: list[dict],
    context_before: list[dict],
    context_after: list[dict],
    language: str,
    glossary_terms: Optional[list[str]] = None,
    source_texts: Optional[list[str]] = None,
    mode: str = "asr",
) -> str:
    """Assemble the per-batch user message with surrounding context.

    ``mode='translation'`` post-edits an already-translated draft toward natural,
    professional subtitles (aggressive on fluency, strict on meaning + timing);
    when ``source_texts`` is given, each draft line is shown next to its ORIGINAL
    source line so mistranslations can be repaired. ``mode='asr'`` (default) is
    the Whisper-accuracy correction profile.
    """
    lang_hint = (
        f"\nLanguage: {language}\n"
        if language and language not in ("", "auto", "unknown")
        else ""
    )

    preserve = getattr(settings, "TRANSCRIPT_PRESERVE_WORDS", True)
    rules: list[str] = []
    # Custom-vocabulary glossary → authoritative proper-noun spellings.
    # This is what fixes inconsistent / phonetically-wrong names (e.g.
    # "Dorian" → "Darlian", "Wing Zero" variants) across the whole
    # transcript, not just within a batch.
    if glossary_terms:
        _terms = ", ".join(str(t).strip() for t in glossary_terms if str(t).strip())
        if _terms:
            rules.append(
                "CANONICAL NAMES (highest priority): these are the correct, "
                "authoritative spellings of proper nouns / jargon in this "
                "content — " + _terms + ". Whenever a segment contains a "
                "phonetically-similar or inconsistently-spelled variant of "
                "one of these, replace it with the EXACT canonical spelling. "
                "Never introduce one of these names where the audio clearly "
                "says something else."
            )
    if mode == "translation":
        # MT post-editing profile. A separate OFFLINE engine produced the base
        # translation; the LLM rewrites that rough draft into natural,
        # professional subtitles WITHOUT doing the translation itself —
        # aggressive on fluency/grammar/word-choice, strict on meaning, count,
        # and timing. When the source line is supplied it is the ground truth.
        tgt_name = _lang_name(language)
        rules.append(
            "PRIMARY OBJECTIVE: post-edit each machine-translated subtitle line "
            "into natural, fluent, idiomatic " + tgt_name + " that reads like "
            "professional human subtitles. Aggressively fix stilted machine-"
            "translation phrasing, grammar, agreement, tense, articles, word "
            "choice, and word order — while preserving the original meaning."
        )
        if source_texts is not None:
            rules.append(
                'Each segment gives the ORIGINAL source line ("source") and the '
                'machine-translation draft ("text"). Use the source as the ground '
                "truth for MEANING: repair mistranslations, restore dropped "
                "meaning, fix wrong subjects/pronouns, and correct names. Edit "
                "the draft — do NOT translate from scratch — and output ONLY the "
                "polished " + tgt_name + " line (never the source text)."
            )
        rules.append(
            "Do NOT re-translate into a different language, change the meaning, "
            "add information, merge or split lines, reorder, or change timing. "
            "Keep exactly one line per input segment."
        )
        if language.lower() in _CJK_LANGS:
            rules.append(
                "Target is Japanese/Korean/Chinese: use 。 and 、 punctuation; "
                "do NOT add Western punctuation; leave particles untouched."
            )
        else:
            rules.append(
                "Output must be entirely in " + tgt_name + " — never leave "
                "source-language words untranslated (except proper nouns that "
                "have no accepted " + tgt_name + " form)."
            )
    elif preserve:
        # Accuracy-focused profile: the polisher's job is to bridge the
        # Whisper-medium-to-Whisper-large quality gap by correcting the
        # specific kinds of errors a smaller ASR model makes — phonetic
        # mis-hears, mis-spelled proper nouns, dropped homophones —
        # WITHOUT rewriting, paraphrasing, or restructuring. The model
        # may substitute individual words when (and ONLY when) the
        # substitution is grounded in BOTH (a) phonetic similarity to
        # the input token and (b) the surrounding context segments.
        # Word count must stay within ±15 % so the model can't
        # silently compress / expand.
        rules.append(
            "PRIMARY OBJECTIVE: Correct individual mis-transcribed words "
            "(phonetic errors, mis-spelled proper nouns, wrong homophones) "
            "to bridge the Whisper-medium → Whisper-large accuracy gap. "
            "DO NOT rewrite, paraphrase, summarise, compress, or "
            "restructure. Sentence structure and word count must remain "
            "essentially unchanged."
        )
        rules.append(
            "WORD COUNT RULE: Output word count must stay within ±15 % of "
            "the input. If you can't find a high-confidence correction, "
            "keep the original word verbatim — false 'corrections' are "
            "worse than missed errors."
        )
        rules.append(
            "Fix recurring proper nouns by consistent spelling across the "
            "batch: when the same person / place / organisation is "
            "transcribed two different ways, pick the spelling that "
            "fits the surrounding context (use the CONTEXT blocks above "
            "and the show's setting from earlier batches). Examples of "
            "the kind of error to catch: 'Dorian' → 'Darlian', "
            "'Aiken' → 'Okay', 'Sex' → 'Zechs', 'areas' → 'Aries', "
            "'Aaron' → 'Heero' — i.e. phonetic ASR confusions on names "
            "and short common words."
        )
        rules.append(
            "You MAY add or correct punctuation, capitalisation, and "
            "sentence boundaries to make the text readable."
        )
        rules.append(
            "You MAY split a long run-on into multiple sentences by "
            "inserting punctuation, but every WORD (after the targeted "
            "phonetic corrections above) must remain in the same order."
        )
        if language.lower() in _CJK_LANGS:
            rules.append(
                "Japanese/Korean/Chinese: insert 。 at obvious sentence "
                "ends and 、 at clause breaks where appropriate. Leave "
                "particles untouched. Do NOT remove or substitute kana. "
                "Phonetic corrections are limited to mis-spelled proper "
                "nouns (katakana name spellings) — do NOT 'correct' "
                "native words."
            )
        else:
            rules.append("Fix homophones: their/there/they're, your/you're, its/it's.")
            rules.append(
                "Standardise spellings of proper nouns across all "
                "segments in this batch — but ONLY when the same entity "
                "is referenced multiple times AND the context makes the "
                "correct spelling unambiguous."
            )
        rules.append(
            "WHEN IN DOUBT, KEEP THE ORIGINAL TEXT. The cost of an "
            "incorrect 'correction' is far higher than leaving a "
            "Whisper-medium artefact in place."
        )
    else:
        if settings.TRANSCRIPT_FILLER_REMOVAL and language.lower() not in _CJK_LANGS:
            rules.append("Delete filler words: um, uh, like, you know, basically, literally, kinda, sorta.")
        if settings.TRANSCRIPT_SENTENCE_REPAIR:
            rules.append(
                "Repair fragmented sentences: if the segment is a fragment that "
                "obviously continues from the previous one in CONTEXT, keep it as "
                "a fragment but match casing/punctuation appropriately."
            )
        if language.lower() in _CJK_LANGS:
            rules.append("Japanese/Korean/Chinese: do NOT add Western punctuation. Leave particles untouched. Do NOT change sentence-final particles.")
        else:
            rules.append("Add proper punctuation: periods, commas, question marks, capitalisation.")
            rules.append("Fix homophones: their/there/they're, your/you're, its/it's.")
            rules.append("Standardise spellings of proper nouns across all segments in this batch.")

    rules_block = "\n".join(f"  - {r}" for r in rules)

    def _ctx_block(label: str, items: list[dict]) -> str:
        if not items:
            return ""
        lines = [f"{label} (do NOT translate or edit — context only):"]
        for s in items:
            t = (s.get("text") or "").strip()
            if not t:
                continue
            start = s.get("start", s.get("start_sec", 0))
            lines.append(f"  [{float(start):.1f}s] {t}")
        return "\n".join(lines) + "\n\n"

    batch_items = []
    for i, s in enumerate(batch):
        t = (s.get("text") or "").strip()
        item = {"index": i, "text": t}
        # MTPE with source reference: show the original line so the model edits
        # the draft against the source meaning instead of translating blind.
        if mode == "translation" and source_texts is not None and i < len(source_texts):
            src_line = (source_texts[i] or "").strip()
            if src_line:
                item["source"] = src_line
        batch_items.append(item)

    prompt = (
        f"{lang_hint}"
        f"{_ctx_block('PREVIOUS CONTEXT', context_before)}"
        f"{_ctx_block('FOLLOWING CONTEXT', context_after)}"
        f"OPERATIONS TO APPLY:\n{rules_block}\n\n"
        f"SEGMENTS TO POLISH (return EXACTLY {len(batch_items)} strings, one per segment, in order):\n"
        f"{json.dumps(batch_items, ensure_ascii=False, indent=2)}\n\n"
        f"Output ONLY a JSON array of {len(batch_items)} polished strings. "
        f"No markdown, no preamble, no explanation."
    )
    return prompt


# A leaked-structure signature: the model echoed the prompt's input objects
# (``{"index": i, "text": ..., "source": ...}``) instead of a flat string array.
# Used to reject any "polished" line that is really a stringified dict so it can
# never reach the subtitles (the '{'index': 0, 'text': ...}' + Japanese 'source'
# leak).
_LEAKED_STRUCT_RE = re.compile(
    r"\{\s*['\"]index['\"]|['\"]text['\"]\s*:\s*['\"]|['\"]source['\"]\s*:\s*['\"]")


def _coerce_polished_item(x) -> Optional[str]:
    """One response element → a clean polished string, or ``None`` if it can't
    be recovered.

    Small models sometimes ECHO the prompt's input objects
    (``{"index": i, "text": ..., "source": ...}``) instead of returning a flat
    array of strings. Extract the ``text`` field rather than ``str()``-ing the
    whole dict — which dumped ``{'index': 0, 'text': ...}`` and the source
    language straight into the subtitles."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for key in ("text", "polished", "line", "translation", "output"):
            v = x.get(key)
            if isinstance(v, str):
                return v
        return None
    return None


def _parse_polished_response(response: str, expected: int) -> Optional[list[Optional[str]]]:
    """Parse the LLM's JSON array response into ``expected`` slots.

    Returns ``None`` only when the response can't be parsed as a JSON array of
    the expected length (alignment is ambiguous → caller keeps the whole batch
    raw). Otherwise returns a list of length ``expected`` whose elements are the
    clean polished string, or ``None`` for any single element that fails the
    existing guards (un-coercible to a string, or still looks like a leaked
    input object). Per-element ``None`` lets the caller keep the ORIGINAL for
    just that index instead of reverting the whole batch — one malformed line in
    a 20-segment batch no longer discards the other 19.

    Tolerates a model that echoes the input objects instead of a flat string
    array by extracting their ``text``. The guards themselves are unchanged;
    only their scope narrowed from whole-batch to per-element."""
    text = (response or "").strip()
    # Strip markdown fences if the model added them.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Last-resort: extract the first [ ... ] block.
    if not text.startswith("["):
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        text = match.group()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != expected:
        # Length mismatch — positional alignment is unreliable, so keep the
        # whole batch raw (unchanged behavior for this case).
        return None
    out: list[Optional[str]] = []
    for x in data:
        s = _coerce_polished_item(x)
        if s is None:
            out.append(None)        # unrecoverable element → keep original here
            continue
        # A line that still looks like a stringified input object (the model
        # returned the dict as a string) must never ship — keep original here.
        if _LEAKED_STRUCT_RE.search(s):
            out.append(None)
            continue
        out.append(s)
    return out


async def _polish_batch(
    orchestrator,
    batch: list[dict],
    context_before: list[dict],
    context_after: list[dict],
    language: str,
    timeout: float,
    glossary_terms: Optional[list[str]] = None,
    source_texts: Optional[list[str]] = None,
    mode: str = "asr",
) -> Optional[list[Optional[str]]]:
    """Polish a single batch via the editorial LLM.

    Returns ``None`` when the whole batch is unusable (LLM error, or a response
    that can't be parsed as a JSON array of the right length). Otherwise returns
    a list of length ``len(batch)`` where each element is the polished string or
    ``None`` for an individual line that failed the parse-level guards (the
    caller keeps the original for just those indices)."""
    user_prompt = _build_user_prompt(
        batch, context_before, context_after, language, glossary_terms,
        source_texts=source_texts, mode=mode)
    system_prompt = _SYSTEM_PROMPT_TRANSLATION if mode == "translation" else _SYSTEM_PROMPT
    full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
    try:
        response = await orchestrator.text_completion(full_prompt, timeout=timeout)
    except Exception as e:
        logger.warning("transcript polishing: LLM call failed: %s", e)
        return None
    polished = _parse_polished_response(response, expected=len(batch))
    if polished is None:
        logger.warning(
            "transcript polishing: bad response shape (expected %d items)",
            len(batch),
        )
    elif any(p is None for p in polished):
        logger.info(
            "transcript polishing: salvaged %d/%d lines (kept original for %d "
            "that failed per-line guards)",
            sum(1 for p in polished if p is not None), len(polished),
            sum(1 for p in polished if p is None),
        )
    return polished


async def polish_source_before_translation(
    segments: Iterable,
    orchestrator=None,
    source_language: str = "",
    job_id: str = "",
) -> list:
    """Light SOURCE-language cleanup (punctuation / casing / filler) applied
    BEFORE translation — the offline transcription-polish parity move (Task 6).

    Cloud transcripts read cleaner partly because the source is effectively
    polished first; offline left the source raw before translation (the heavy
    readability polish only ran on the TARGET text afterward). A single
    ``mode='asr'`` pass here means the translator works from clean input AND the
    shipped source transcript reads cleanly. Uses the orchestrator's editorial
    model — the local model when offline.

    Gated by ``TRANSLATION_POLISH_SOURCE_FIRST`` (and the master
    ``TRANSCRIPT_POLISHING_ENABLED`` / ``AI_TRANSCRIPT_CORRECTION`` switches).
    Fail-soft: returns the segments unchanged on any error or when disabled, so
    it can never break the translate path. Cue count + timing are preserved by
    ``correct_transcript``'s round-trip.
    """
    seg_list = list(segments) if segments else []
    if not seg_list:
        return seg_list
    if not bool(getattr(settings, "TRANSLATION_POLISH_SOURCE_FIRST", True)):
        return seg_list
    if (orchestrator is None
            or not getattr(settings, "TRANSCRIPT_POLISHING_ENABLED", True)
            or not getattr(settings, "AI_TRANSCRIPT_CORRECTION", True)):
        return seg_list
    try:
        polished = await correct_transcript(
            seg_list, orchestrator, job_id=job_id,
            language=(source_language or ""), mode="asr",
        )
    except Exception as e:
        logger.warning(
            "source-before-translation polish failed (%s) — keeping raw source", e)
        return seg_list
    # Defensive: never change the cue count on this path (translation aligns 1:1).
    if not polished or len(polished) != len(seg_list):
        logger.warning(
            "source-before-translation polish changed cue count (%s≠%d) — keeping "
            "raw source", len(polished) if polished else 0, len(seg_list))
        return seg_list
    return polished


async def correct_transcript(
    segments: Iterable,
    orchestrator=None,
    job_id: str = "",
    language: str = "",
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int], Any]] = None,
    timeout_per_batch: float = 90.0,
    glossary_terms: Optional[list[str]] = None,
    source_texts: Optional[list[str]] = None,
    source_language: str = "",
    mode: str = "asr",
) -> list:
    """Polish a transcript using the editorial LLM in batches.

    ``mode='translation'`` post-edits an already-translated draft toward natural,
    professional subtitles (aggressive on fluency, strict on meaning + timing,
    never re-translates into another language). When ``source_texts`` is given
    (aligned 1:1 with ``segments``), each draft line is shown next to its
    ORIGINAL source line so the model can repair mistranslations against the
    source. ``mode='asr'`` (default) is the Whisper-accuracy correction profile.

    Drop-in replacement for ``compat_stubs.correct_transcript`` — same
    signature, same return shape. When ``TRANSCRIPT_POLISHING_ENABLED``
    is False or the orchestrator is missing, returns the segments
    unchanged (the legacy behavior).

    ``glossary_terms`` supplies authoritative proper-noun spellings (the
    custom-vocabulary glossary). When None, it is auto-loaded from the
    persisted glossary so the polisher enforces consistent names without
    every caller having to thread it through.
    """
    seg_list = list(segments) if segments else []
    if not seg_list:
        return seg_list

    if not settings.TRANSCRIPT_POLISHING_ENABLED or orchestrator is None:
        return seg_list

    # Auto-load the custom-vocabulary glossary as the canonical name list.
    if glossary_terms is None and getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True):
        try:
            from backend.services.custom_vocabulary import load_vocabulary
            glossary_terms = load_vocabulary()
        except Exception:
            glossary_terms = None

    if batch_size is None:
        batch_size = max(1, int(getattr(settings, "TRANSCRIPT_POLISHING_BATCH_SIZE", 15)))
    # CJK: smaller batches because dense glyphs make the model lose count.
    if (language or "").lower() in _CJK_LANGS:
        batch_size = min(batch_size, 8)

    # Coerce into dicts for prompt construction while keeping the original
    # objects around for round-tripping.
    views = [(_coerce_segment(s), s) for s in seg_list]
    batches = []
    for start in range(0, len(views), batch_size):
        batches.append(views[start:start + batch_size])

    polished_out: list = []
    total = len(batches)
    # Length tolerance by profile. ASR/preserve modes keep the tight band +
    # word-count clamp so the model can't paraphrase. MT post-editing
    # (mode='translation') is SUPPOSED to rephrase the rough draft into fluent
    # subtitles, so it gets a generous band and NO word-count clamp — clamping
    # would reject exactly the fluent rewrites we want; only gross runaway
    # (likely hallucination) is rejected.
    preserve_mode = getattr(settings, "TRANSCRIPT_PRESERVE_WORDS", True)
    len_max_ratio = 1.5 if preserve_mode else 3.0
    len_min_ratio = 0.6 if preserve_mode else 0.3
    for idx, batch_pairs in enumerate(batches):
        batch = [pair[0] for pair in batch_pairs]
        # Sliding 3-segment context windows (separate from the
        # TRANSLATION_CONTEXT_WINDOW which controls the translator).
        ctx_before_start = max(0, idx * batch_size - 3)
        ctx_before = [v[0] for v in views[ctx_before_start: idx * batch_size]]
        ctx_after_start = (idx + 1) * batch_size
        ctx_after = [v[0] for v in views[ctx_after_start: ctx_after_start + 3]]
        # Source reference for MTPE, sliced to match this batch (aligned 1:1
        # with ``segments``). None when no source was threaded in.
        batch_src = None
        if source_texts is not None:
            _bs = idx * batch_size
            batch_src = source_texts[_bs: _bs + len(batch)]

        polished_texts = await _polish_batch(
            orchestrator, batch, ctx_before, ctx_after,
            language=language, timeout=timeout_per_batch,
            glossary_terms=glossary_terms, source_texts=batch_src, mode=mode,
        )

        for i, (view, orig_obj) in enumerate(batch_pairs):
            orig_full = view.get("text", "")
            if polished_texts is None:
                # Whole batch failed (LLM error / unparseable). Keep the MT draft
                # for translation mode; for ASR fall back to a light-touch filler
                # strip rather than nothing.
                new_text = (orig_full if mode == "translation"
                            else _light_filler_strip(orig_full, language))
                polished_out.append(_emit_segment(orig_obj, new_text))
                continue
            if polished_texts[i] is None:
                # This specific line failed the per-line parse guards — keep its
                # ORIGINAL text + words verbatim while the rest of the batch is
                # polished. Re-emit the original object unchanged so word timing
                # is preserved.
                polished_out.append(orig_obj)
                continue
            cand = polished_texts[i].strip()
            orig_text = orig_full.strip()
            if mode == "translation":
                # Accept fluent rewrites (longer OR terser); reject only empty
                # output or gross runaway. The 80-char floor lets a short, badly
                # truncated draft be expanded to a correct full line.
                if cand and len(cand) <= max(80, len(orig_text) * 3):
                    new_text = cand
                else:
                    new_text = orig_full
            else:
                orig_len = len(orig_text)
                # Length-band guard
                if orig_len > 0 and (
                    len(cand) > orig_len * len_max_ratio
                    or len(cand) < orig_len * len_min_ratio
                ):
                    new_text = orig_full
                # Word-count guard (preserve mode only) — catches a model that
                # kept text length but substituted multi-word paraphrases.
                elif preserve_mode and orig_text:
                    orig_words = orig_text.split()
                    cand_words = cand.split()
                    if orig_words and (
                        len(cand_words) > len(orig_words) * 1.30
                        or len(cand_words) < len(orig_words) * 0.70
                    ):
                        new_text = orig_full
                    else:
                        new_text = cand if cand else orig_full
                else:
                    new_text = cand if cand else orig_full
            polished_out.append(_emit_segment(orig_obj, new_text))

        if progress_callback:
            try:
                res = progress_callback(int(((idx + 1) / total) * 100))
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    changed = sum(
        1 for old, new in zip(seg_list, polished_out)
        if (_coerce_segment(old).get("text") or "") != (_coerce_segment(new).get("text") or "")
    )
    logger.info(
        "transcript polishing: %d/%d segments polished (job=%s lang=%s)",
        changed, len(polished_out), (job_id or "")[:8], language or "auto",
    )
    return polished_out
