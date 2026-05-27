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


def _emit_segment(orig, new_text: str):
    """Apply ``new_text`` back to a segment without mutating timing/speaker.
    Round-trips both dict and Pydantic shapes."""
    if isinstance(orig, dict):
        # Some pipelines use ``text``, others might use both ``text`` and
        # carry word-level timestamps. We only rewrite text.
        out = dict(orig)
        out["text"] = new_text
        # Strip word timestamps when text changed — they no longer align.
        if "words" in out and out.get("text", "") != orig.get("text", ""):
            out["words"] = []
        return out
    obj = orig
    # Try to use ``model_copy`` (Pydantic v2) or fall back to attribute set.
    try:
        new = obj.model_copy(update={"text": new_text, "words": []})
        return new
    except Exception:
        try:
            setattr(obj, "text", new_text)
            if hasattr(obj, "words"):
                setattr(obj, "words", [])
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
) -> str:
    """Assemble the per-batch user message with surrounding context."""
    lang_hint = (
        f"\nLanguage: {language}\n"
        if language and language not in ("", "auto", "unknown")
        else ""
    )

    preserve = getattr(settings, "TRANSCRIPT_PRESERVE_WORDS", True)
    rules: list[str] = []
    if preserve:
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
        batch_items.append({"index": i, "text": t})

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


def _parse_polished_response(response: str, expected: int) -> Optional[list[str]]:
    """Parse the LLM's JSON array response. Returns None on parse failure
    or on length mismatch."""
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
    if not isinstance(data, list):
        return None
    if len(data) != expected:
        return None
    return [str(x) if x is not None else "" for x in data]


async def _polish_batch(
    orchestrator,
    batch: list[dict],
    context_before: list[dict],
    context_after: list[dict],
    language: str,
    timeout: float,
) -> Optional[list[str]]:
    """Polish a single batch via the editorial LLM. Returns None on failure
    so the caller can keep the originals."""
    user_prompt = _build_user_prompt(batch, context_before, context_after, language)
    full_prompt = f"[SYSTEM]\n{_SYSTEM_PROMPT}\n\n[USER]\n{user_prompt}"
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
    return polished


async def correct_transcript(
    segments: Iterable,
    orchestrator=None,
    job_id: str = "",
    language: str = "",
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int], Any]] = None,
    timeout_per_batch: float = 90.0,
) -> list:
    """Polish a transcript using the editorial LLM in batches.

    Drop-in replacement for ``compat_stubs.correct_transcript`` — same
    signature, same return shape. When ``TRANSCRIPT_POLISHING_ENABLED``
    is False or the orchestrator is missing, returns the segments
    unchanged (the legacy behavior).
    """
    seg_list = list(segments) if segments else []
    if not seg_list:
        return seg_list

    if not settings.TRANSCRIPT_POLISHING_ENABLED or orchestrator is None:
        return seg_list

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
    for idx, batch_pairs in enumerate(batches):
        batch = [pair[0] for pair in batch_pairs]
        # Sliding 3-segment context windows (separate from the
        # TRANSLATION_CONTEXT_WINDOW which controls the translator).
        ctx_before_start = max(0, idx * batch_size - 3)
        ctx_before = [v[0] for v in views[ctx_before_start: idx * batch_size]]
        ctx_after_start = (idx + 1) * batch_size
        ctx_after = [v[0] for v in views[ctx_after_start: ctx_after_start + 3]]

        polished_texts = await _polish_batch(
            orchestrator, batch, ctx_before, ctx_after,
            language=language, timeout=timeout_per_batch,
        )

        # Length tolerance depends on the polish profile. In the
        # default ``preserve=True`` (accuracy-focused) path the prompt
        # explicitly forbids restructuring and bounds word count at
        # ±15 %; tighten the rejection thresholds to match so the
        # model can't silently drift into paraphrase territory. In
        # the ``preserve=False`` (full editing) path the model is
        # allowed to drop fillers and merge fragments so leave the
        # legacy looser bounds.
        preserve_mode = getattr(settings, "TRANSCRIPT_PRESERVE_WORDS", True)
        len_max_ratio = 1.5 if preserve_mode else 3.0
        len_min_ratio = 0.6 if preserve_mode else 0.3
        for i, (view, orig_obj) in enumerate(batch_pairs):
            if polished_texts is None:
                # Fall back: light-touch filler strip rather than nothing.
                new_text = _light_filler_strip(view.get("text", ""), language)
            else:
                cand = polished_texts[i].strip()
                orig_text = view.get("text", "").strip()
                orig_len = len(orig_text)
                # Length-band guard
                if orig_len > 0 and (
                    len(cand) > orig_len * len_max_ratio
                    or len(cand) < orig_len * len_min_ratio
                ):
                    new_text = view.get("text", "")
                # Word-count guard (preserve mode only) — catches a model
                # that kept text length but substituted multi-word
                # paraphrases. Compare token counts via simple whitespace
                # split; tolerant enough to handle the prompt's ±15 %
                # allowance plus one or two punctuation-driven shifts.
                elif preserve_mode and orig_text:
                    orig_words = orig_text.split()
                    cand_words = cand.split()
                    if orig_words and (
                        len(cand_words) > len(orig_words) * 1.30
                        or len(cand_words) < len(orig_words) * 0.70
                    ):
                        new_text = view.get("text", "")
                    else:
                        new_text = cand if cand else view.get("text", "")
                else:
                    new_text = cand if cand else view.get("text", "")
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
