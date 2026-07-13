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
import time
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
    "Aim for BROADCAST subtitle quality (Netflix / professional YouTube): every "
    "line must read as fluent, natural target language a native speaker would "
    "say — never word-for-word or machine-literal.\n\n"
    "STRICT RULES (must follow every time):\n"
    "1. Return EXACTLY the same number of lines you receive, in order — one "
    "rewritten line per input segment. Never merge or split.\n"
    "2. Each output line must be in the TARGET language only — never leave "
    "source-language words (except proper nouns with no target form).\n"
    "3. Preserve the meaning of every line. When unsure, keep the draft.\n"
    "4. Never change timing — those fields are not in your output.\n"
    "5. NEVER repeat a phrase within a line ('I will go I will go') and NEVER "
    "duplicate the previous line's text — if a line would just echo its "
    "neighbour, rephrase it to carry the line's own meaning.\n"
    "6. No stutter or filler runs ('no no no no', 'the the') — write the clean, "
    "natural phrasing a professional subtitler would use.\n"
    "7. Each line should be a coherent, complete thought, not a choppy fragment.\n"
    "8. Return ONLY a JSON array of strings, no preamble, no markdown.\n"
    "9. The array length must equal the input segment count.\n"
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
        # ASR confidence rides along so the polish prompt can mark lines the
        # model may rewrite more freely vs. lines it must preserve.
        "avg_logprob": getattr(seg, "avg_logprob", None),
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
    _any_low_conf = False
    for i, s in enumerate(batch):
        t = (s.get("text") or "").strip()
        item = {"index": i, "text": t}
        # MTPE with source reference: show the original line so the model edits
        # the draft against the source meaning instead of translating blind.
        if mode == "translation" and source_texts is not None and i < len(source_texts):
            src_line = (source_texts[i] or "").strip()
            if src_line:
                item["source"] = src_line
        # Confidence marks: a low Whisper avg_logprob means the ASR was
        # GUESSING ("Hand kimchi", "mandarind fruit") — tell the model which
        # lines it may rewrite toward contextual sense vs. which it must
        # preserve near-verbatim. Threshold matches the redecode trigger.
        try:
            _lp = s.get("avg_logprob")
            if _lp is not None and float(_lp) < float(
                    getattr(settings, "WHISPER_REDECODE_LOGPROB", -0.8)):
                item["asr_confidence"] = "low"
                _any_low_conf = True
        except (TypeError, ValueError):
            pass
        batch_items.append(item)

    conf_note = ""
    if _any_low_conf:
        conf_note = (
            'Segments marked "asr_confidence": "low" are unreliable speech '
            "recognition — if such a line reads as nonsense, rewrite it into "
            "what was most plausibly said given the surrounding context. "
            "Unmarked lines are reliable: preserve their wording.\n\n")

    prompt = (
        f"{lang_hint}"
        f"{_ctx_block('PREVIOUS CONTEXT', context_before)}"
        f"{_ctx_block('FOLLOWING CONTEXT', context_after)}"
        f"OPERATIONS TO APPLY:\n{rules_block}\n\n"
        f"{conf_note}"
        f"SEGMENTS TO POLISH (one output object per segment):\n"
        f"{json.dumps(batch_items, ensure_ascii=False, indent=2)}\n\n"
        f"Output ONLY a JSON array of {len(batch_items)} objects, one per "
        f"segment, each shaped {{\"index\": <that segment's index>, "
        f"\"text\": \"<polished text>\"}}. Include every index from 0 to "
        f"{len(batch_items) - 1} exactly once, in order. "
        f"The editing rules NEVER justify changing the number of outputs: "
        f"when adjacent inputs are duplicates or fragments of one thought, "
        f"still return one object per index (complete the thought at its "
        f"first index; give the other index its own remaining content) — "
        f"never omit or merge indices. "
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


# Bare (unquoted) object keys — the observed qwen failure shape is
# ``{ index: 3, text: "…" }``, which json.loads rejects outright even though
# the payload is otherwise fine. Quoting just the identifier-shaped keys
# recovers it. Runs only after strict parsing failed, so it can't corrupt a
# valid response.
_BARE_KEY_RE = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)')
_TRAILING_COMMA_RE = re.compile(r',\s*([}\]])')

# One indexed pair inside a broken array: index first, then the text string
# (with escapes). Used as the last-resort extractor when even the repaired
# text won't parse (e.g. a truncated tail cuts the array mid-object).
_INDEXED_PAIR_RE = re.compile(
    r'["\']?index["\']?\s*:\s*(\d+)\s*,\s*["\']?text["\']?\s*:\s*"((?:[^"\\]|\\.)*)"')


def _repair_json(text: str):
    """Best-effort re-parse of near-JSON: quote bare keys, drop trailing
    commas. Returns the parsed value or ``None``."""
    repaired = _BARE_KEY_RE.sub(r'\1"\2"\3', text)
    repaired = _TRAILING_COMMA_RE.sub(r'\1', repaired)
    if repaired == text:
        return None
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _regex_salvage_indexed(text: str, expected: int) -> Optional[list[Optional[str]]]:
    """Extract ``{"index": N, "text": "…"}`` pairs from an unparseable
    response. Each pair's text is decoded as a JSON string (escapes resolved);
    pairs that fail keep their cue's original text via the ``None`` slot."""
    quoted = _BARE_KEY_RE.sub(r'\1"\2"\3', text)
    pairs = []
    for m in _INDEXED_PAIR_RE.finditer(quoted):
        try:
            pairs.append((int(m.group(1)), json.loads('"' + m.group(2) + '"')))
        except (ValueError, json.JSONDecodeError):
            continue
    if not pairs:
        return None
    return _slots_from_indexed(pairs, expected)


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
    # Strip any <think>…</think> reasoning blocks first. The default local model
    # (Qwen3-4B-Instruct-2507) is NON-thinking, but a mis-tagged / swapped model
    # could emit them; an unclosed block (truncated mid-think) would otherwise
    # break the JSON-array extraction below.
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<think>.*$", "", text)
    text = text.strip()
    # Strip markdown fences if the model added them.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    # Last-resort: extract the first [ ... ] block. A bare JSON OBJECT
    # ({"0": ..., "1": ...} or {"segments": [...]}) passes through to the
    # dict-salvage path below instead of being rejected here.
    if not text.startswith("[") and not text.startswith("{"):
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return None
        text = match.group()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = _repair_json(text)
        if data is None:
            # Final salvage: pull whatever {"index": N, "text": "..."} pairs
            # exist out of the broken response before discarding the whole
            # generation (each discard costs a halve-and-retry round trip).
            return _regex_salvage_indexed(text, expected)
    # Index-keyed salvage: the prompt asks for {"index": i, "text": ...}
    # objects precisely because small models LOSE COUNT on flat arrays (the
    # observed 15-failures-per-run "bad response shape" waste — each one a
    # full generation discarded). With explicit indices, a response that
    # merged or dropped lines still salvages every index it did return;
    # missing indices keep their original text. Exact mapping — no positional
    # guessing — so this cannot mis-assign a polish to the wrong cue.
    if isinstance(data, dict):
        inner = data.get("segments")
        if isinstance(inner, list):
            data = inner
        else:
            return _slots_from_indexed(list(data.items()), expected)
    if not isinstance(data, list):
        return None
    if data and all(isinstance(x, dict)
                    and ("index" in x or "i" in x) for x in data):
        slots = _slots_from_indexed(
            [(x.get("index", x.get("i")), x) for x in data], expected)
        if slots is not None:
            return slots
    if len(data) != expected:
        # Un-indexed length mismatch — positional alignment is unreliable, so
        # keep the whole batch raw (unchanged behavior for this case).
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


def _slots_from_indexed(pairs, expected: int) -> Optional[list[Optional[str]]]:
    """Build ``expected`` slots from (index, value) pairs. Indices outside
    range and un-coercible values are skipped (their cues keep the original
    text). Returns ``None`` when NOTHING salvages, so the caller treats the
    batch as failed exactly as before."""
    slots: list[Optional[str]] = [None] * expected
    filled = 0
    for raw_idx, val in pairs:
        try:
            i = int(raw_idx)
        except (TypeError, ValueError):
            continue
        if not (0 <= i < expected):
            continue
        s = val if isinstance(val, str) else _coerce_polished_item(val)
        if s is None or _LEAKED_STRUCT_RE.search(s):
            continue
        slots[i] = s
        filled += 1
    return slots if filled else None


# ── Cloud fallback for the polish LLM (Netflix-quality reliability) ────────
# The observed failure mode on the GTX 1650 box: an Ollama-only provider
# chain where the local model cold-loads (partial offload) slower than the
# batch timeout → 3 strikes → circuit breaker degrades Ollama for 15 min →
# EVERY polish batch fails and the raw draft ships unpolished. When the user
# has configured an OpenRouter key, that is explicit cloud intent — use it
# as the safety net so polish quality never silently drops to zero.

_cloud_polish_provider = None  # cached OpenRouterProvider


def _cloud_polish_available() -> bool:
    if not bool(getattr(settings, "SUBTITLE_POLISH_CLOUD_FALLBACK", True)):
        return False
    return bool((getattr(settings, "OPENROUTER_API_KEY", "") or "").strip())


def _resolve_cloud_polish_model() -> str:
    """The OpenRouter model the cloud path polishes on.

    Order: SUBTITLE_POLISH_MODEL (when it's an OpenRouter-style id) →
    SUBTITLE_POLISH_CLOUD_MODEL → the first "efficient"-tier entry of the
    curated shortlist (strong constrained editing at low cost).
    """
    pinned = (getattr(settings, "SUBTITLE_POLISH_MODEL", "") or "").strip()
    if pinned and "/" in pinned:
        return pinned
    explicit = (getattr(settings, "SUBTITLE_POLISH_CLOUD_MODEL", "") or "").strip()
    if explicit and "/" in explicit:
        return explicit
    try:
        from backend.services.providers.openrouter_provider import (
            SUBTITLE_POLISH_SHORTLIST)
        for entry in SUBTITLE_POLISH_SHORTLIST:
            if entry.get("tier") == "efficient":
                return entry["id"]
    except Exception:
        pass
    return "google/gemini-2.5-flash"


async def _cloud_polish_completion(full_prompt: str, timeout: float) -> Optional[str]:
    """Direct OpenRouter completion for polish, bypassing the provider chain.

    Used (a) as the fallback when the local chain fails, and (b) as the
    primary path when SUBTITLE_POLISH_MODEL pins an OpenRouter model but the
    active chain is Ollama-only. Returns None on any failure — callers keep
    their existing fail-soft behavior.
    """
    global _cloud_polish_provider
    if not _cloud_polish_available():
        return None
    model = _resolve_cloud_polish_model()
    try:
        if _cloud_polish_provider is None:
            from backend.services.providers.openrouter_provider import (
                OpenRouterProvider)
            _cloud_polish_provider = OpenRouterProvider()
        prov = _cloud_polish_provider
        prev_model = prov._editorial_model
        try:
            prov._editorial_model = model
            result = await asyncio.wait_for(
                prov.text_complete(full_prompt, timeout=int(timeout)),
                timeout=timeout,
            )
        finally:
            prov._editorial_model = prev_model
        logger.info("polish cloud fallback succeeded via OpenRouter %s", model)
        _note_cloud_fallback_used(model)
        return result
    except Exception as e:
        logger.warning("polish cloud fallback via %s failed: %s", model, e)
        return None


# Jobs already warned about cloud polish spend (one warning per job — the
# fallback fires per batch and can run dozens of times).
_cloud_fallback_warned_jobs: set[str] = set()


def _note_cloud_fallback_used(model: str) -> None:
    """Surface the polish cloud fallback in the job's pipeline warnings.

    The fallback is a deliberate safety net, but it spends real money on a
    setup the user believes is fully local (observed: 31 silent Haiku calls
    ≈ $0.009 on an 'all-Ollama' job after the local model cold-load blew
    the 90s text timeout). Warn once per job, with the off switch named.
    """
    try:
        from backend.services.request_context import current_job_id
        job_id = current_job_id()
        if not job_id or job_id in _cloud_fallback_warned_jobs:
            return
        _cloud_fallback_warned_jobs.add(job_id)
        from backend.services.pipeline_helpers import _record_pipeline_warning
        _record_pipeline_warning(
            job_id,
            (f"Local polish model timed out — polish batches fell back to "
             f"OpenRouter ({model}), which bills a small cloud cost. For a "
             "strictly-local run, disable 'cloud polish fallback' in "
             "Settings → Subtitles, or fix the local timeout (see the "
             "estimated cost on this job for the actual spend)."),
        )
    except Exception:
        pass


async def _evict_model(model: str) -> None:
    """Unload ``model`` from the primary Ollama host (keep_alive=0).

    Used when the polish pass abandons an auto-upgraded model after batch 0:
    without an explicit eviction Ollama keeps the (usually partially-offloaded)
    big model resident for its keep_alive window, and the light model the pass
    downshifted to gets starved into partial offload beside it. Fail-soft —
    any error just leaves Ollama's own keep_alive to clean up eventually."""
    try:
        import httpx
        from backend.services import ollama_registry as reg
        host = await reg.pick_host(required_model=model)
        if host is None:
            host = reg.primary_host()
        if host is None:
            return
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                reg.join_url(host.url, "/api/generate"),
                headers=reg.auth_headers(host),
                json={"model": model, "keep_alive": 0},
            )
        logger.info(
            "transcript polishing: evicted abandoned upgrade model %s from "
            "'%s' (HTTP %s) so the downshifted model gets the whole GPU",
            model, host.name or host.url, r.status_code)
    except Exception as e:
        logger.debug("transcript polishing: eviction of %s skipped (%s)",
                     model, e)


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
    model_override: Optional[str] = None,
    cloud_direct: bool = False,
    local_only: bool = False,
) -> Optional[list[Optional[str]]]:
    """Polish a single batch via the polish LLM.

    ``local_only`` keeps the batch on the local / companion GPU: the
    orchestrator call skips cloud providers and the OpenRouter direct
    fallback is not attempted, so a slow local batch is never silently
    answered (and billed) by the cloud — the caller keeps the raw draft.

    ``model_override`` routes the call to a specific model (the dedicated
    translation model for subtitle polishing) instead of the editorial model;
    None keeps the editorial model.

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
    response = None
    if cloud_direct:
        # SUBTITLE_POLISH_MODEL pins an OpenRouter model — go straight to the
        # cloud instead of feeding an OpenRouter id to an Ollama-only chain
        # (which fails every batch and then falls back here anyway).
        response = await _cloud_polish_completion(full_prompt, timeout)
        if response is None:
            return None
    else:
        try:
            response = await orchestrator.text_completion(
                full_prompt, timeout=timeout, model_override=model_override,
                skip_circuit_breaker=True, local_only=local_only)
        except Exception as e:
            logger.warning("transcript polishing: LLM call failed: %s", e)
            if local_only:
                # Strictly-local polish (default): the batch runs on the local /
                # companion GPU and must NOT fall back to a paid cloud provider.
                # Keep the raw draft for this batch instead of billing the cloud.
                return None
            # Local chain exhausted (timeout / circuit breaker / offline
            # chain with Ollama degraded). Cloud safety net — quality must
            # not silently drop to an unpolished draft.
            response = await _cloud_polish_completion(full_prompt, timeout)
            if response is None:
                return None
    polished = _parse_polished_response(response, expected=len(batch))
    if polished is None:
        # Diagnosable failure: the old message ("expected 15 items") gave no
        # signal whether the model lost count, merged lines, returned broken
        # JSON, or had its prompt truncated by the context window — include
        # what actually came back.
        _head = re.sub(r"\s+", " ", (response or ""))[:120]
        logger.warning(
            "transcript polishing: bad response shape (expected %d items, "
            "response_head=%r)", len(batch), _head,
        )
    elif any(p is None for p in polished):
        logger.info(
            "transcript polishing: salvaged %d/%d lines (kept original for %d "
            "that failed per-line guards)",
            sum(1 for p in polished if p is not None), len(polished),
            sum(1 for p in polished if p is None),
        )
    return polished


# ── Deterministic (non-LLM) punctuation-restore fallback ───────────────────
# When the local editorial model is unavailable / errors / leaves a segment
# without a terminator, restore sentence punctuation deterministically so the
# resegmenter still has boundaries to split on. The Latin restorer is the
# optional ``deepmultilingualpunctuation`` model (lazily loaded, cached); CJK
# uses a rule-based terminator since that model doesn't cover CJK scripts.

_TERMINATORS = ".?!…。！？"
_TERM_CLOSERS = "\"')]}」』）”’‘“"

_PUNCT_MODEL = None
_PUNCT_MODEL_UNAVAILABLE = False


def _ends_with_terminator(text: str) -> bool:
    s = (text or "").rstrip()
    while s and s[-1] in _TERM_CLOSERS:
        s = s[:-1]
    return bool(s) and s[-1] in _TERMINATORS


def _get_punct_model():
    """Lazily load + cache the optional Latin punctuation model. Returns None
    (once) when the dependency isn't installed or fails to load."""
    global _PUNCT_MODEL, _PUNCT_MODEL_UNAVAILABLE
    if _PUNCT_MODEL is not None or _PUNCT_MODEL_UNAVAILABLE:
        return _PUNCT_MODEL
    try:
        from deepmultilingualpunctuation import PunctuationModel
        _PUNCT_MODEL = PunctuationModel()
    except Exception as e:  # ImportError or model-load failure
        logger.info(
            "punctuation-restore fallback: optional dependency unavailable "
            "(%s) — Latin text left unchanged", e)
        _PUNCT_MODEL_UNAVAILABLE = True
    return _PUNCT_MODEL


def _restore_text_punctuation(text: str, language: str) -> str:
    """Add sentence punctuation to ``text`` deterministically. Returns ``text``
    unchanged when nothing can be done (so it's always fail-soft)."""
    t = (text or "")
    if not t.strip():
        return text
    if (language or "").lower() in _CJK_LANGS or _is_probably_cjk(t):
        # Rule-based: ensure a sentence terminator at the end. The pause-based
        # resegmenter handles internal CJK boundaries acoustically.
        stripped = t.rstrip()
        if stripped and stripped[-1] not in _TERMINATORS:
            return stripped + "。"
        return text
    model = _get_punct_model()
    if model is None:
        return text
    try:
        restored = model.restore_punctuation(t)
        return restored if restored and restored.strip() else text
    except Exception as e:
        logger.debug("punctuation-restore fallback failed (%s) — keeping text", e)
        return text


def _is_probably_cjk(text: str) -> bool:
    try:
        from backend.services.subtitle_formatter import _is_cjk
        return _is_cjk(text)
    except Exception:
        return False


def restore_punctuation_fallback(segments, language: str = "") -> list:
    """Apply the deterministic punctuation restorer to any segment that lacks a
    sentence terminator. Gated by ``PUNCTUATION_RESTORE_FALLBACK_ENABLED``.

    Fail-soft: returns the input unchanged when disabled or on any error.
    Timing / speaker are preserved and word timestamps are re-mapped onto the
    restored text via ``_emit_segment``."""
    seg_list = list(segments) if segments else []
    if not seg_list:
        return seg_list
    if not bool(getattr(settings, "PUNCTUATION_RESTORE_FALLBACK_ENABLED", True)):
        return seg_list
    # Neighbour-aware CJK terminators. The old per-cue rule stamped '。' on
    # EVERY unterminated CJK cue — including mid-utterance Whisper fragments
    # whose continuation starts 0.1-0.3 s later. That fake sentence stop (a)
    # made the NMT decode each fragment as a finished sentence (the
    # fragmentary translations in the measured run) and (b) disarmed the
    # pause-based resegmenter, whose split gate treats a terminator as
    # authoritative. Append the terminator only when the cue plausibly ENDS
    # an utterance: a turn-length gap to the next cue, a speaker change, or
    # being the last cue. Only machine-APPENDED punctuation is affected —
    # Whisper's and the LLM's own punctuation is never touched, and the
    # Latin restorer (model-based, context-aware) keeps its behavior.
    _turn_gap_s = 0.7
    try:
        _turn_gap_s = float(getattr(
            settings, "SENTENCE_SPLIT_TURN_PAUSE_MS", 700)) / 1000.0
    except Exception:
        pass

    def _utterance_ends_here(idx: int, view: dict) -> bool:
        if idx >= len(seg_list) - 1:
            return True
        try:
            nxt = _coerce_segment(seg_list[idx + 1])
            if (nxt.get("speaker") or "") != (view.get("speaker") or ""):
                return True
            gap = float(nxt.get("start") or 0.0) - float(view.get("end") or 0.0)
            return gap >= _turn_gap_s
        except Exception:
            return True   # unknown neighbour — keep the legacy behavior

    out: list = []
    n_restored = 0
    for idx, seg in enumerate(seg_list):
        try:
            view = _coerce_segment(seg)
            text = (view.get("text") or "")
            if text.strip() and not _ends_with_terminator(text):
                _cjk_cue = ((language or "").lower() in _CJK_LANGS
                            or _is_probably_cjk(text))
                if not _cjk_cue or _utterance_ends_here(idx, view):
                    new_text = _restore_text_punctuation(text, language)
                    if new_text and new_text != text:
                        out.append(_emit_segment(seg, new_text))
                        n_restored += 1
                        continue
        except Exception:
            pass
        out.append(seg)
    if n_restored:
        logger.info(
            "punctuation-restore fallback: added terminators to %d/%d segment(s)",
            n_restored, len(seg_list))
    return out


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


def derive_entity_glossary(texts: Iterable[str], max_terms: int = 16,
                           min_count: int = 3) -> list[str]:
    """Auto-derive a canonical proper-noun list from the transcript itself.

    Netflix-grade subs render a name the SAME way in every cue; small-model
    drafts drift ("Zeks"/"Zecks"/"Zeck", "Riley"/"Rilina"/"Lilina"). This
    clusters recurring capitalized tokens by fuzzy similarity (same initial,
    difflib ratio ≥ 0.75), picks the most frequent variant as canonical, and
    returns the canonicals so the polish prompt can pin them. Data-only
    heuristic — no LLM call. Sentence-initial words only count when they
    also appear capitalized mid-sentence (drops This/The/Yes noise).
    """
    import difflib
    from collections import Counter

    mid_counts: Counter = Counter()
    all_counts: Counter = Counter()
    word_re = re.compile(r"[A-Za-z][a-z]+(?:-[A-Za-z][a-z]+)?")
    for text in texts:
        tokens = re.findall(r"\S+", text or "")
        for pos, raw in enumerate(tokens):
            m = word_re.fullmatch(raw.strip('.,!?;:"()[]—-'))
            if not m:
                continue
            w = m.group(0)
            if not w[0].isupper() or len(w) < 3:
                continue
            all_counts[w] += 1
            if pos > 0:
                mid_counts[w] += 1

    # Candidates must recur AND appear mid-sentence at least once.
    cands = [w for w, c in all_counts.items()
             if c >= min_count and mid_counts.get(w, 0) >= 1]
    cands.sort(key=lambda w: -all_counts[w])

    canonicals: list[str] = []
    used: set = set()
    for w in cands:
        if w in used:
            continue
        cluster = [w]
        for other in cands:
            if other in used or other == w:
                continue
            if other[0].lower() != w[0].lower():
                continue
            if difflib.SequenceMatcher(None, w.lower(), other.lower()).ratio() >= 0.75:
                cluster.append(other)
        for c in cluster:
            used.add(c)
        # Canonical = most frequent variant; only worth pinning when the
        # name actually recurs (a cluster total under min_count is noise).
        if sum(all_counts[c] for c in cluster) >= min_count:
            canonicals.append(max(cluster, key=lambda c: all_counts[c]))
        if len(canonicals) >= max_terms:
            break
    return canonicals


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
    model_override: Optional[str] = None,
    local_only: Optional[bool] = None,
) -> list:
    """Polish a transcript using the polish LLM in batches.

    ``local_only`` keeps every batch on the local / companion GPU (no cloud
    provider, no OpenRouter fallback). ``None`` reads the default from
    ``SUBTITLE_POLISH_LOCAL_ONLY`` (on by default) so polish never silently
    bills a cloud provider for a run the user believes is fully local.

    ``model_override`` pins every batch to a specific model — used so subtitle
    polishing runs on the dedicated translation model (the multilingual model
    that also translates) rather than the editorial model, which stays reserved
    for SEO + summaries. None keeps the editorial model (legacy behavior).

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

    # Strictly-local polish by default: keep the polish LLM on the local /
    # companion GPU and never fall back to a paid cloud provider for what the
    # user runs as a local job.
    if local_only is None:
        local_only = bool(getattr(settings, "SUBTITLE_POLISH_LOCAL_ONLY", True))

    # Translation-polish quality routing: when the PRIMARY Ollama host is a paired
    # Companion GPU, upsize the polish model to the best one installed there
    # (e.g. qwen2.5:14b) so the English track reads far more naturally. Only fires
    # for the translation post-edit on a local Ollama model; no-op on a local-only
    # card or a cloud (``vendor/model``) override. The pre-upgrade model is kept
    # so the batch loop can DOWNSHIFT back to it when the upgraded model's
    # measured throughput would blow the polish time budget.
    _base_model_override = model_override
    if (mode == "translation" and model_override
            and "/" not in str(model_override)
            and bool(getattr(settings, "OLLAMA_TRANSLATION_POLISH_AUTO", True))):
        try:
            from backend.services.translator import resolve_translation_polish_model
            _upgraded = await resolve_translation_polish_model(model_override)
            if _upgraded and _upgraded != model_override:
                logger.info(
                    "transcript polishing: routing translation polish to %s on the "
                    "Companion GPU (was %s)", _upgraded, model_override)
                model_override = _upgraded
        except Exception:
            pass

    if not settings.TRANSCRIPT_POLISHING_ENABLED or orchestrator is None:
        # No LLM polish available — readability would otherwise hinge entirely
        # on the model. Still restore sentence terminators deterministically so
        # the resegmenter has boundaries to work with.
        return restore_punctuation_fallback(seg_list, language=language)

    # Auto-load the custom-vocabulary glossary as the canonical name list.
    if glossary_terms is None and getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True):
        try:
            from backend.services.custom_vocabulary import load_vocabulary
            glossary_terms = load_vocabulary()
        except Exception:
            glossary_terms = None

    # Merge an auto-derived proper-noun list so recurring names render
    # consistently even without a user-maintained vocabulary. The custom
    # vocabulary (user-authored) always wins order-wise.
    if bool(getattr(settings, "SUBTITLE_POLISH_AUTO_GLOSSARY", True)):
        try:
            auto_terms = derive_entity_glossary(
                (_coerce_segment(x).get("text", "") for x in seg_list))
            existing = {t.lower() for t in (glossary_terms or [])}
            merged = list(glossary_terms or []) + [
                t for t in auto_terms if t.lower() not in existing]
            if merged:
                glossary_terms = merged
        except Exception:
            pass

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
    # An OpenRouter-style pinned polish model on a local-only chain would
    # fail per batch before falling back — route it straight to the cloud.
    # Suppressed under local_only: strictly-local polish never touches the cloud.
    cloud_direct = bool(model_override and "/" in model_override
                        and not local_only and _cloud_polish_available())

    # ── Batch execution: bounded concurrency + wall-clock budget ──
    # The observed failure this replaces: 851 translated cues → 57 batches run
    # STRICTLY SEQUENTIALLY against qwen2.5:14b on the Companion = 43 minutes
    # of pipeline time for a polish that is an enhancement, not a requirement.
    # Now:
    #   * batch 0 runs alone (existing cold-load logic, and it calibrates
    #     per-batch latency);
    #   * remaining batches run SUBTITLE_POLISH_CONCURRENCY at a time (the
    #     Companion's Ollama serves parallel requests — resolve_speed gives a
    #     4070 num_parallel 3-4), identical outputs, ~Nx the throughput;
    #   * SUBTITLE_POLISH_MAX_S caps the whole pass — when the budget runs
    #     out, remaining batches keep their draft text (exactly what a failed
    #     batch already does) instead of holding the pipeline hostage;
    #   * if batch-0 latency projects a blown budget AND the model was the
    #     auto-upgraded Companion model, remaining batches downshift to the
    #     original (smaller, several-times-faster) model — polish coverage
    #     stays near-100% instead of being budget-truncated;
    #   * 5 consecutive whole-batch failures aborts the rest (drafts kept) so
    #     a broken model can't burn GPU-minutes producing garbage.
    _conc = max(1, int(getattr(settings, "SUBTITLE_POLISH_CONCURRENCY", 3)))
    _budget_s = float(getattr(settings, "SUBTITLE_POLISH_MAX_S", 600.0))
    _t_start = time.monotonic()
    _deadline = (_t_start + _budget_s) if _budget_s > 0 else None
    _state = {"fail_streak": 0, "aborted": False, "skipped": 0, "done": 0,
              "model": model_override}
    batch_results: list[Optional[list[Optional[str]]]] = [None] * len(batches)

    # Context-window size. The neighbor blocks are the fat in each prompt
    # (~half of a ~6000-char batch), and the observed run polished only
    # 211/930 cues (23%) in the 600 s budget because every call carried
    # ±3 neighbors on TOP of the per-line source ref. When the batch is
    # source-aligned, each line already has its own ground-truth source
    # line, so a wide neighbor window is redundant — shrink it to
    # SUBTITLE_POLISH_CONTEXT_ALIGNED (default 1) to ~halve the prompt and
    # roughly double throughput. Non-aligned (ASR) mode keeps the wider
    # window since it has no per-line reference.
    _ctx_n = (int(getattr(settings, "SUBTITLE_POLISH_CONTEXT_ALIGNED", 1))
              if source_texts is not None
              else int(getattr(settings, "SUBTITLE_POLISH_CONTEXT", 3)))
    _ctx_n = max(0, _ctx_n)

    def _ctx_for(idx: int, batch_len: int):
        ctx_before_start = max(0, idx * batch_size - _ctx_n)
        ctx_before = ([v[0] for v in views[ctx_before_start: idx * batch_size]]
                      if _ctx_n else [])
        ctx_after_start = (idx + 1) * batch_size
        ctx_after = ([v[0] for v in views[ctx_after_start: ctx_after_start + _ctx_n]]
                     if _ctx_n else [])
        batch_src = None
        if source_texts is not None:
            _bs = idx * batch_size
            batch_src = source_texts[_bs: _bs + batch_len]
        return ctx_before, ctx_after, batch_src

    async def _run_one(idx: int, timeout_s: float) -> None:
        if _state["aborted"]:
            _state["skipped"] += 1
            return
        if _deadline is not None and time.monotonic() >= _deadline:
            if not _state["aborted"]:
                _state["aborted"] = True
                logger.warning(
                    "transcript polishing: %.0fs budget exhausted after %d/%d "
                    "batches — remaining cues keep their draft text "
                    "(SUBTITLE_POLISH_MAX_S)", _budget_s, _state["done"],
                    len(batches))
            _state["skipped"] += 1
            return
        batch = [pair[0] for pair in batches[idx]]
        ctx_before, ctx_after, batch_src = _ctx_for(idx, len(batch))
        result = await _polish_batch(
            orchestrator, batch, ctx_before, ctx_after,
            language=language, timeout=timeout_s,
            glossary_terms=glossary_terms, source_texts=batch_src, mode=mode,
            model_override=_state["model"], cloud_direct=cloud_direct,
            local_only=local_only,
        )
        if (result is None and len(batch) >= 4 and not _state["aborted"]
                and (_deadline is None or time.monotonic() < _deadline)):
            # Halve-and-retry, once: half-size arrays are dramatically more
            # count-reliable for small models AND halve the prompt (so a
            # context-window truncation stops eating the JSON contract).
            # translate_via_llm already uses this exact recovery. A half that
            # still fails keeps its drafts; both halves failing counts as ONE
            # failure for the breaker.
            mid = len(batch) // 2
            halves: list[list[Optional[str]]] = []
            for lo, hi in ((0, mid), (mid, len(batch))):
                if _deadline is not None and time.monotonic() >= _deadline:
                    halves.append([None] * (hi - lo))
                    continue
                sub_src = batch_src[lo:hi] if batch_src is not None else None
                # Retries run after the main call already paid any cold-load,
                # so they use the BASE timeout even when the main call ran
                # with the scaled first-batch timeout.
                half = await _polish_batch(
                    orchestrator, batch[lo:hi], ctx_before, ctx_after,
                    language=language, timeout=timeout_per_batch,
                    glossary_terms=glossary_terms, source_texts=sub_src,
                    mode=mode, model_override=_state["model"],
                    cloud_direct=cloud_direct, local_only=local_only,
                )
                halves.append(half if half is not None
                              else [None] * (hi - lo))
            merged = halves[0] + halves[1]
            if any(x is not None for x in merged):
                logger.info(
                    "transcript polishing: halve-and-retry recovered %d/%d "
                    "lines of a failed batch",
                    sum(1 for x in merged if x is not None), len(merged))
                result = merged
        batch_results[idx] = result
        _state["done"] += 1
        if result is None:
            _state["fail_streak"] += 1
            if _state["fail_streak"] >= 5 and not _state["aborted"]:
                _state["aborted"] = True
                logger.warning(
                    "transcript polishing: 5 consecutive batch failures — "
                    "aborting the remaining batches (drafts kept)")
        else:
            _state["fail_streak"] = 0
        if progress_callback:
            try:
                res = progress_callback(
                    int((_state["done"] / max(1, len(batches))) * 100))
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    # Batch 0 alone: cold-load timeout scaling (on a 4 GB card the FIRST batch
    # often pays a multi-minute Ollama partial-offload model load; a flat 90s
    # timeout used to strike the circuit breaker and kill polish for the whole
    # job) + latency calibration for the downshift decision.
    _t0 = time.monotonic()
    await _run_one(0, min(300.0, timeout_per_batch * 3))
    _first_latency = time.monotonic() - _t0
    if (len(batches) > 1 and _deadline is not None
            and _state["model"] and _state["model"] != _base_model_override):
        _remaining_s = _deadline - time.monotonic()
        _projected_s = _first_latency * (len(batches) - 1) / _conc
        if _projected_s > _remaining_s:
            logger.info(
                "transcript polishing: %s runs %.0fs/batch — projected %.0fs "
                "exceeds the %.0fs budget; downshifting remaining batches to "
                "%s so every cue still gets polished",
                _state["model"], _first_latency, _projected_s, _remaining_s,
                _base_model_override)
            _too_slow = _state["model"]
            _state["model"] = _base_model_override
            # EVICT the abandoned upgrade immediately (keep_alive=0). A model
            # that was too slow for batch 0 was almost certainly partially
            # offloaded — and Ollama keeps it resident (keep_alive ~10 min,
            # max_loaded 2), starving the light model we just downshifted to
            # into partial offload TOO. The observed run: 14b at 90 s/batch →
            # downshift → the 4b still crawled at ~45 s/batch beside the
            # resident 14b, and the budget died at 31/62 batches. Fire-and-
            # forget: eviction failing only means today's (slow) behavior.
            asyncio.ensure_future(_evict_model(_too_slow))

    if len(batches) > 1:
        _sem = asyncio.Semaphore(_conc)

        async def _guarded(idx: int) -> None:
            async with _sem:
                await _run_one(idx, timeout_per_batch)

        # ── Worst-first ordering ──────────────────────────────────────────
        # The budget routinely runs out before every batch is polished
        # (observed: 35/62). Left-to-right, that means the LAST third of the
        # track ships raw — but "needs polish" isn't positional. Rank each
        # remaining batch by how many of its cues Whisper was UNSURE about
        # (avg_logprob below the redecode threshold = the "Hand kimchi",
        # romaji-fragment, word-salad lines), and polish the neediest batches
        # FIRST. Context windows are still read from the full ``views`` array
        # by absolute index, so reordering EXECUTION doesn't change any
        # batch's neighbors — only which cues win the budget. Ties keep
        # natural order for stable context/logs.
        _lp_thr = float(getattr(settings, "WHISPER_REDECODE_LOGPROB", -0.8))

        def _neediness(idx: int) -> int:
            score = 0
            for pair in batches[idx]:
                v = pair[0]
                try:
                    lp = v.get("avg_logprob")
                    if lp is not None and float(lp) < _lp_thr:
                        score += 1
                        continue
                except (TypeError, ValueError):
                    pass
                # Fallback signal when confidence is absent: a very short cue
                # or one still carrying source-script glyphs is likelier junk.
                t = (v.get("text") or "").strip()
                if len(t) <= 3:
                    score += 1
            return score

        _rest = list(range(1, len(batches)))
        if bool(getattr(settings, "SUBTITLE_POLISH_WORST_FIRST", True)):
            _scored = [(idx, _neediness(idx)) for idx in _rest]
            if any(s for _, s in _scored):
                _rest = [idx for idx, _ in
                         sorted(_scored, key=lambda p: (-p[1], p[0]))]
                logger.info(
                    "transcript polishing: worst-first order — %d batch(es) "
                    "carry low-confidence cues, polishing those before the "
                    "budget runs out", sum(1 for _, s in _scored if s))

        await asyncio.gather(*(_guarded(i) for i in _rest),
                             return_exceptions=True)

    for idx, batch_pairs in enumerate(batches):
        polished_texts = batch_results[idx]
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

    changed = sum(
        1 for old, new in zip(seg_list, polished_out)
        if (_coerce_segment(old).get("text") or "") != (_coerce_segment(new).get("text") or "")
    )
    logger.info(
        "transcript polishing: %d/%d segments polished (job=%s lang=%s)",
        changed, len(polished_out), (job_id or "")[:8], language or "auto",
    )
    # Safety net: the ≤4B model often misses terminators on some segments
    # (especially CJK). Restore them deterministically so the resegmenter isn't
    # left guessing on un-punctuated cues. No-op for already-terminated cues.
    polished_out = restore_punctuation_fallback(polished_out, language=language)
    return polished_out
