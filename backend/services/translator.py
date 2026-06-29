"""Subtitle translation service.

Translates TranscriptSegment[] from source language to target language
using the configured AI provider. Preserves timing, speaker labels, and
generates proportional word timestamps for translated text.

If the primary provider fails (e.g. vision-only model), falls back to
a dedicated Ollama translation model (OLLAMA_TRANSLATION_MODEL).
"""

import asyncio
import json
import logging
from typing import Optional

import httpx

from backend.config import settings
from backend.models import TranscriptSegment, WordTimestamp
from backend.services.ai_orchestrator import AIOrchestrator
from backend.services.providers.base import ProviderRateLimitError

logger = logging.getLogger(__name__)


class TranslationFailedError(RuntimeError):
    """Translation did not actually complete (so the source must NOT be
    relabelled as translated). Carries a user-facing message."""


# ── Language-purity detection ────────────────────────────────────────────────
# Whisper's task='translate' silently leaves music / narration / hard segments
# in the SOURCE language on mixed content (the "translated" track comes back half
# Japanese). We detect that by measuring how much of the output is still in the
# source SCRIPT, and reject it — so a broken translation falls through to a
# complete one instead of being persisted as-is.

_CJK_SOURCES = {"ja", "japanese", "zh", "chinese", "zh-cn", "zh-tw", "ko", "korean", "yue"}


def _cjk_ratio(text: str) -> float:
    """Share of a line's letters that are Hiragana / Katakana / Han / Hangul."""
    t = text or ""
    cjk = 0
    base = 0
    for c in t:
        is_cjk = (
            "぀" <= c <= "ヿ"     # hiragana + katakana
            or "㐀" <= c <= "鿿"  # CJK ideographs
            or "가" <= c <= "힣"  # hangul
            or "ｦ" <= c <= "ﾟ"  # half-width katakana
        )
        if is_cjk:
            cjk += 1
            base += 1
        elif c.isalpha():
            base += 1
    return (cjk / base) if base else 0.0


def _seg_text(s):
    return (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")) or ""


def fraction_untranslated(segments, target_language: str) -> float:
    """Fraction of cues still written in CJK script.

    Meaningful only when translating TO a non-CJK target (English etc.), where
    any CJK left in the output is untranslated source — and crucially this is
    judged from the OUTPUT TEXT, not the declared source language, so it still
    works when the source was detected as ``auto`` (which previously disabled the
    check and let half-Japanese tracks through). For CJK targets, CJK output is
    correct, so it returns 0."""
    if (target_language or "").lower() in _CJK_LANGS:
        return 0.0
    rows = list(segments or [])
    if not rows:
        return 0.0
    n = sum(1 for s in rows if _cjk_ratio(_seg_text(s)) > 0.30)
    return n / len(rows)


# Back-compat alias (some call sites pass the source language; the check is
# content-based now, so the language argument is only used to skip CJK targets).
def fraction_source_script(segments, language: str) -> float:
    return fraction_untranslated(segments, "en")


# ── LLM (editorial-model) translation ────────────────────────────────────────

_LLM_LANG_NAMES = {
    "ja": "Japanese", "en": "English", "ko": "Korean", "zh": "Chinese",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic", "hi": "Hindi",
    "nl": "Dutch", "tr": "Turkish", "pl": "Polish", "th": "Thai",
    "vi": "Vietnamese", "id": "Indonesian",
}


def _parse_json_array(response: str, expected: int) -> Optional[list[str]]:
    """Parse the LLM's ``["...", "..."]`` reply into exactly ``expected`` strings."""
    import re
    text = (response or "").strip()
    # Strip any <think>…</think> reasoning blocks before parsing. The default
    # local model (Qwen3-4B-Instruct-2507) is NON-thinking and emits none, but a
    # mis-tagged / swapped model could — and an unclosed block (truncated
    # mid-think) would otherwise poison the array extraction below.
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<think>.*$", "", text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", text)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    if not text.startswith("["):
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return None
        text = m.group()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or len(data) != expected:
        return None
    # A small model sometimes ECHOES the prompt's input objects
    # (``{"index": i, "text": ...}``) instead of a flat string array. Extract
    # their ``text`` rather than ``str()``-ing the dict (which leaks
    # ``{'index': 0, 'text': ...}`` into the subtitles); reject anything that
    # still can't be reduced to a clean string so the caller keeps the source.
    _LEAK = re.compile(r"\{\s*['\"]index['\"]|['\"]text['\"]\s*:\s*['\"]")
    out: list[str] = []
    for x in data:
        if x is None:
            out.append("")
        elif isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            v = next((x[k] for k in ("text", "translation", "line", "output")
                      if isinstance(x.get(k), str)), None)
            if v is None:
                return None
            out.append(v)
        else:
            return None
    if any(_LEAK.search(s) for s in out):
        return None
    return out


async def translate_via_llm(
    segments: list,
    source_language: str,
    target_language: str,
    orchestrator: AIOrchestrator,
    glossary: dict | None = None,
    job_id: str = "",
    status_callback=None,
) -> Optional[list[TranscriptSegment]]:
    """Translate the SOURCE transcript text-to-text with the editorial LLM,
    1:1 — every segment, same timing + speaker. The reliable, COMPLETE path:
    unlike Whisper's translate task it never leaves lyrics / narration in the
    source language. Returns ``None`` when no orchestrator is available."""
    if not orchestrator or not segments:
        return None

    _sl = (source_language or "").strip().lower()
    src_name = (_LLM_LANG_NAMES.get(_sl)
                or ("the source language" if _sl in ("", "auto") else source_language))
    tgt_name = _LLM_LANG_NAMES.get((target_language or "").lower(), target_language or "English")

    def _txt(s):
        return (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")) or ""

    # Auto-derive a glossary of recurring proper nouns from the WHOLE transcript
    # (not per-batch) so names render consistently and coined nouns are
    # transliterated, not translated into ordinary words. Content-agnostic.
    _auto_terms = ""
    if getattr(settings, "TRANSLATION_AUTO_GLOSSARY", True):
        try:
            from backend.services.glossary import build_translation_glossary_block
            _auto_terms = build_translation_glossary_block(segments, source_language, tgt_name)
            if _auto_terms:
                logger.info("LLM translate: attached recurring-terms glossary "
                            "(%d term chars) for consistency", len(_auto_terms))
        except Exception as _g_e:
            logger.debug("auto-glossary skipped: %s", _g_e)

    async def _call(batch) -> Optional[list[str]]:
        lines = [_txt(s) for s in batch]
        numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
        prompt = (
            f"You are a professional {src_name}->{tgt_name} subtitle translator.\n"
            f"Translate EVERY one of the {len(lines)} numbered subtitle lines below into "
            f"natural, fluent {tgt_name}.\n"
            "Rules:\n"
            f"- Translate ALL lines, including song lyrics, narration and exclamations. "
            f"NEVER leave a line in {src_name}.\n"
            "- Preserve honorifics (-san, -kun, -chan, -sama) and proper nouns "
            "(names of people, places, organizations, products); transliterate "
            "names rather than translating them into ordinary words.\n"
            "- Translate faithfully: never summarise, merge meaning, or invent "
            "words to fill a gap. If unsure, translate as literally as possible.\n"
            "- Exactly one output per input line; never merge, split, add or drop lines.\n"
            f"- Output ONLY a JSON array of exactly {len(lines)} {tgt_name} strings, in order. "
            "No commentary, no numbering.\n\n"
            f"{_auto_terms}"
            f"Lines:\n{numbered}"
        )
        try:
            # Per-batch timeout is a CEILING (raising it never slows the fast
            # path — it only stops a slow LOCAL model being killed mid-answer and
            # falling back to the weaker offline NMT). A small editorial model on
            # a 4 GB GPU needs well over the old 5 s/segment / 60 s floor: a
            # first batch also pays the model load. Generous + configurable.
            _per_seg = float(getattr(settings, "TRANSLATION_LLM_SECONDS_PER_SEGMENT", 12.0))
            _floor = float(getattr(settings, "TRANSLATION_LLM_TIMEOUT_FLOOR", 180.0))
            resp = await orchestrator.text_completion(
                prompt, timeout=max(_floor, len(batch) * _per_seg), job_id=job_id)
        except Exception as e:
            logger.warning("LLM translate: call failed (%s)", e)
            return None
        return _parse_json_array(resp, expected=len(batch))

    async def _translate_batch(batch) -> list[str]:
        out = await _call(batch)
        if out is not None:
            return out
        if len(batch) <= 1:                       # keep the source rather than drop it
            return [_txt(s) for s in batch]
        mid = len(batch) // 2                      # split on failure and recurse
        return (await _translate_batch(batch[:mid])) + (await _translate_batch(batch[mid:]))

    # Smaller batches on a local model: a 3B model on a small GPU generates a
    # short JSON array far faster + more reliably than an 18-line one, so each
    # batch is much less likely to hit the timeout (the failure that dropped the
    # whole LLM path to FuguMT and left ~18% of cues in Japanese). Cloud models
    # keep the larger batch. Configurable via TRANSLATION_LLM_BATCH.
    try:
        _chain = orchestrator._get_active_chain() if orchestrator else []
        _is_ollama = bool(_chain) and getattr(_chain[0], "provider_name", "") == "ollama"
    except Exception:
        _is_ollama = False
    BATCH = int(getattr(settings, "TRANSLATION_LLM_BATCH", 8 if _is_ollama else 18) or
                (8 if _is_ollama else 18))
    total = len(segments)
    out_segs: list[TranscriptSegment] = []
    _any_ok = False
    for i in range(0, total, BATCH):
        batch = segments[i: i + BATCH]
        direct = await _call(batch)
        if direct is not None:
            translations = direct
            _any_ok = True
        elif not _any_ok and i == 0:
            # The very first batch failing outright means the editorial model
            # isn't usable here — bail so the caller falls back cleanly instead
            # of "translating" every line to itself.
            logger.info("LLM translate: editorial model returned no usable output "
                        "— deferring to other translation engines.")
            return None
        else:
            translations = await _translate_batch(batch)
        for seg, tr in zip(batch, translations):
            txt = (tr or "").strip() or _txt(seg)
            if glossary:
                for k, v in glossary.items():
                    ks, vs = (k or "").strip(), (v or "").strip()
                    if ks and vs and ks in txt:
                        txt = txt.replace(ks, vs)
            start = float(seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0) or 0.0)
            spk = (seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)) or "Speaker 1"
            out_segs.append(TranscriptSegment(text=txt, start=start, end=end, speaker=spk))
        if status_callback:
            try:
                r = status_callback(f"Translating subtitles… ({min(i + BATCH, total)}/{total})")
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

    # ── Completeness cleanup ────────────────────────────────────────────────
    # The model occasionally echoes a hard line (long narration, song lyrics)
    # untranslated inside an otherwise-valid array — and with a vague/auto source
    # it does so more often. Re-translate any cue still in CJK script (only when
    # the target is non-CJK), up to a couple of passes, so NOTHING is left in the
    # source language.
    if (target_language or "").lower() not in _CJK_LANGS:
        for _pass in range(3):
            idxs = [i for i, s in enumerate(out_segs) if _cjk_ratio(s.text or "") > 0.30]
            if not idxs:
                break
            logger.info("LLM translate: re-translating %d cue(s) still in source "
                        "script (pass %d)", len(idxs), _pass + 1)
            redo = await _translate_batch([out_segs[i] for i in idxs])
            for i, tr in zip(idxs, redo):
                t = (tr or "").strip()
                if t and _cjk_ratio(t) <= 0.30:
                    if glossary:
                        for k, v in glossary.items():
                            ks, vs = (k or "").strip(), (v or "").strip()
                            if ks and vs and ks in t:
                                t = t.replace(ks, vs)
                    cur = out_segs[i]
                    out_segs[i] = TranscriptSegment(
                        text=t, start=cur.start, end=cur.end, speaker=cur.speaker)

    logger.info("LLM translation: %d/%d segments → %s (%.0f%% still source-script)",
                len(out_segs), total, target_language,
                100 * fraction_untranslated(out_segs, target_language))
    return out_segs


class TranslationRateLimitedError(TranslationFailedError):
    """The LLM translation model is being rate-limited upstream (sustained
    HTTP 429 / "add your own key"). We stop early instead of crawling through
    every batch, and surface a visible ``translation_failed`` state."""


def _looks_rate_limited(exc: Exception) -> bool:
    """True when ``exc`` indicates upstream rate-limiting (HTTP 429)."""
    if isinstance(exc, ProviderRateLimitError):
        return True
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "rate-limit" in s or "too many requests" in s


def _is_free_openrouter_model(spec: str) -> bool:
    """True when ``spec`` names a FREE OpenRouter model (``…:free``).

    Accepts bare ids (``vendor/model:free``) and provider-prefixed specs
    (``openrouter:vendor/model:free``)."""
    s = (spec or "").strip().lower()
    return bool(s) and s.endswith(":free")


def _llm_translation_model_is_free(orchestrator, model_override) -> tuple[bool, str]:
    """Best-effort: would the LLM translation path call a FREE OpenRouter model?

    Returns ``(is_free, label)``. Free OpenRouter endpoints are rate-limited
    upstream and only burn a job in 429s, so the router skips them (Task 2).
    An explicit translation override wins; otherwise we inspect the
    orchestrator's active provider chain (the model it would actually use),
    then fall back to the relevant settings. We treat the model as NOT free
    whenever the first provider the chain would try is not OpenRouter (a paid
    OpenAI/Anthropic key, or local Ollama) — so the skip only fires when a free
    OpenRouter endpoint is genuinely what the translator would hit."""
    if model_override:
        return (_is_free_openrouter_model(model_override), model_override)
    try:
        chain = orchestrator._get_active_chain()
    except Exception:
        chain = None
    if chain:
        for provider in chain:
            pname = getattr(provider, "provider_name", "")
            model = getattr(provider, "text_model_name", "") or ""
            if pname == "openrouter":
                return (_is_free_openrouter_model(model), model)
            # First provider isn't OpenRouter — not the free-grind case.
            return (False, model)
    for spec in (
        getattr(settings, "OPENROUTER_TRANSLATION_MODEL", ""),
        getattr(settings, "OPENROUTER_EDITORIAL_MODEL", ""),
        getattr(settings, "EDITORIAL_AI_FALLBACK_SPEC", ""),
    ):
        s = (spec or "").strip()
        if s:
            return (_is_free_openrouter_model(s), s)
    return (False, "")

# Friendly names for every language the offline NMT engine can target. Kept in
# sync with nmt_translator._FLORES_CODES (same key set) so the translate-subtitles
# endpoint accepts — and the pipeline names — any of them. The user can pick any
# of these as a subtitle target; translation is never capped to a short list.
SUPPORTED_LANGUAGES = {
    # Western European
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ca": "Catalan",
    "gl": "Galician", "eu": "Basque", "ga": "Irish", "cy": "Welsh",
    "is": "Icelandic", "lb": "Luxembourgish", "mt": "Maltese",
    # Nordic
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "nb": "Norwegian Bokmål",
    "nn": "Norwegian Nynorsk", "fi": "Finnish",
    # Slavic / Baltic / other Eastern European
    "ru": "Russian", "uk": "Ukrainian", "pl": "Polish", "cs": "Czech",
    "sk": "Slovak", "sl": "Slovenian", "hr": "Croatian", "sr": "Serbian",
    "bs": "Bosnian", "bg": "Bulgarian", "mk": "Macedonian", "be": "Belarusian",
    "ro": "Romanian", "hu": "Hungarian", "et": "Estonian", "lv": "Latvian",
    "lt": "Lithuanian", "sq": "Albanian", "el": "Greek",
    # Middle East / Caucasus / Central Asia
    "ar": "Arabic", "he": "Hebrew", "fa": "Persian", "tr": "Turkish",
    "az": "Azerbaijani", "kk": "Kazakh", "ky": "Kyrgyz", "uz": "Uzbek",
    "tg": "Tajik", "hy": "Armenian", "ka": "Georgian", "ku": "Kurdish",
    "ps": "Pashto",
    # South Asia
    "hi": "Hindi", "bn": "Bengali", "ur": "Urdu", "pa": "Punjabi",
    "gu": "Gujarati", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "ne": "Nepali", "si": "Sinhala",
    "or": "Odia", "as": "Assamese",
    # East / Southeast Asia
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)", "yue": "Cantonese",
    "vi": "Vietnamese", "th": "Thai", "id": "Indonesian", "ms": "Malay",
    "tl": "Filipino", "my": "Burmese", "km": "Khmer", "lo": "Lao",
    "jv": "Javanese", "su": "Sundanese", "mn": "Mongolian",
    # Africa
    "sw": "Swahili", "am": "Amharic", "ha": "Hausa", "yo": "Yoruba",
    "ig": "Igbo", "zu": "Zulu", "xh": "Xhosa", "sn": "Shona",
    "so": "Somali", "af": "Afrikaans", "mg": "Malagasy", "ny": "Chichewa",
    "st": "Sesotho",
}

TRANSLATION_PROMPT = """Translate the following subtitle segments from {source_lang} to {target_lang}.

These are scripted dialogue lines from a film / TV / streaming production
intended to display as on-screen subtitles. Treat them the way a
professional subtitler would: faithful to meaning and character voice,
not flowery, and timed for reading speed.

CRITICAL RULES:
1. Translate naturally — produce fluent {target_lang}, not word-for-word translation
2. Preserve the meaning, tone, and speaker intent — match each line's register
   (formal/casual/military/intimate) to the {target_lang} equivalent
3. Preserve dialogue rhythm — short utterances stay short, sentence-final
   particles become tonal markers, NOT extra words
4. Keep translations concise — each subtitle must be readable in 2-3 seconds.
   When in doubt, prefer the shorter / cleaner phrasing
5. Preserve proper nouns (names of people, brands, ranks like "Lieutenant",
   organization names like "OZ" / "Alliance") UNLESS they have standard
   {target_lang} translations
6. Match published subtitle conventions — no narrator additions, no
   parenthetical stage directions, no expanded descriptions of what's
   visible on screen
7. Return EXACTLY {count} translated strings as a JSON array
8. If a segment is very short (e.g., "Yeah", "Okay"), use the natural
   {target_lang} equivalent — do NOT expand short utterances
9. DO NOT leave any words in {source_lang} unless they are proper nouns
10. DO NOT romanize — output must be in {target_lang} script
{extra_rules}
{context_section}Segments to translate:
{segments_json}

Return ONLY a JSON array of {count} translated strings. No markdown, no explanation.
Example: ["Translated one.", "Translated two."]"""


# Language-pair specific rules
_PAIR_RULES = {
    ("ja", "en"): (
        "9. Japanese honorifics: translate -san as Mr./Ms., -sensei as Professor/Teacher, "
        "-sama as a respectful form, -kun/-chan can be dropped\n"
        "10. Sentence-final particles (よ, ね, な, わ) convey nuance — "
        "reflect them in tone rather than translating literally\n"
        "11. Japanese often omits the subject — infer it from context"
    ),
    ("ko", "en"): (
        "9. Korean honorifics: translate 님 (-nim) as Mr./Ms., 선생님 as Teacher/Professor\n"
        "10. Respect the speech level (formal/informal) in English word choice"
    ),
    ("zh", "en"): (
        "9. Chinese idioms (成语 chéngyǔ): translate the meaning, not the characters\n"
        "10. Measure words can be dropped in English"
    ),
}

# CJK language codes for batch size reduction
_CJK_LANGS = {"ja", "ko", "zh", "zh-cn", "zh-tw"}


def _get_pair_rules(source: str, target: str) -> str:
    """Get language-pair-specific translation rules."""
    key = (source.lower(), target.lower())
    return _PAIR_RULES.get(key, "")


def _generate_proportional_word_timestamps(
    text: str, start: float, end: float
) -> list[WordTimestamp]:
    """Generate proportional word timestamps for translated text.

    Since we don't have actual word-level alignment for translations,
    distribute the segment duration proportionally across words
    based on character count.
    """
    words = text.split()
    if not words:
        return []

    duration = end - start
    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        word_dur = duration / len(words)
        return [
            WordTimestamp(
                start=round(start + i * word_dur, 3),
                end=round(start + (i + 1) * word_dur, 3),
                word=w,
            )
            for i, w in enumerate(words)
        ]

    current = start
    result = []
    for w in words:
        char_ratio = len(w) / total_chars
        word_dur = duration * char_ratio
        result.append(WordTimestamp(
            start=round(current, 3),
            end=round(current + word_dur, 3),
            word=w,
        ))
        current += word_dur
    return result


def _parse_translation_response(response: str) -> list:
    """Parse a JSON array from an LLM translation response."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    return json.loads(cleaned)


async def _ensure_ollama_model(model: str) -> bool:
    """Check if the Ollama model exists locally; pull it if not."""
    host = settings.OLLAMA_HOST
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(f"{host}/api/show", json={"model": model})
            if resp.status_code == 200:
                return True
    except Exception:
        pass

    # Model not found — pull it
    logger.info("Translation model %s not found locally — pulling from Ollama registry...", model)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=1800, write=10, pool=10)) as client:
            resp = await client.post(
                f"{host}/api/pull",
                json={"name": model},
            )
            if resp.status_code == 200:
                logger.info("Successfully pulled translation model: %s", model)
                return True
            else:
                logger.warning("Failed to pull translation model %s: HTTP %d", model, resp.status_code)
                return False
    except Exception as e:
        logger.warning("Failed to pull translation model %s: %s", model, e)
        return False


async def _translate_batch_via_ollama(
    prompt: str,
    model: str,
    timeout: float = 180.0,
    num_ctx: int = 4096,
) -> str:
    """Call Ollama chat API directly with a dedicated translation model.

    ``num_ctx`` is configurable so the MTPE post-edit pass can use a larger
    context window (8192) and keep real surrounding context on long videos.
    """
    host = settings.OLLAMA_HOST
    options = {
        "num_ctx": int(num_ctx),
        "temperature": 0.3,
        "num_predict": 4096,
    }
    # Qwen3 family: apply low-temperature + presence/repetition penalties for
    # deterministic subtitle JSON (Qwen3 repeats without a penalty). Gated to the
    # Qwen3 family — non-Qwen3 models keep the default 0.3 temperature.
    try:
        from backend.services.local_models import qwen3_translation_options
        options.update(qwen3_translation_options(model))
    except Exception:
        pass
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0)) as client:
        resp = await client.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": options,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


class _OllamaMTPEClient:
    """Minimal orchestrator-shaped client so ``transcript_polisher.
    correct_transcript`` can post-edit the offline NMT draft with the DEDICATED
    local translation model (``OLLAMA_TRANSLATION_MODEL``) at a larger context
    window — reusing the polisher's MTPE persona, batching, strict JSON-array +
    count check, and timing-preserving round-trip without the full
    ``AIOrchestrator`` / its (possibly cloud) editorial model."""

    def __init__(self, model: str, num_ctx: int):
        self._model = model
        self._num_ctx = int(num_ctx)

    async def text_completion(self, prompt: str, timeout: float = 90.0, **_kwargs) -> str:
        return await _translate_batch_via_ollama(
            prompt, self._model, timeout=timeout, num_ctx=self._num_ctx)


def translation_quality_mode_active() -> bool:
    """True when ``TRANSLATION_QUALITY_MODE=quality`` AND an Ollama host +
    quality model are configured (so the CPU quality path can actually run).

    Imported by the pipeline so it can skip its LLM-first / Whisper-native
    preemptions and let the offline router's quality path be the one that runs.
    """
    if (getattr(settings, "TRANSLATION_QUALITY_MODE", "speed") or "speed").lower() != "quality":
        return False
    host = (getattr(settings, "OLLAMA_HOST", "") or "").strip()
    model = (getattr(settings, "TRANSLATION_QUALITY_MODEL", "") or "").strip()
    return bool(host and model)


async def _translate_quality_mode(
    segments: list[TranscriptSegment],
    source_language: str,
    target_language: str,
    glossary: dict | None,
    status_callback=None,
) -> list[TranscriptSegment] | None:
    """Task 5 'quality' mode: translate the source with a larger Ollama model
    on CPU (``TRANSLATION_QUALITY_MODEL``, e.g. qwen2.5:7b-instruct) via
    ``translate_via_llm``, then backstop any cue it left in the source language
    with offline NLLB.

    Returns ``None`` when the quality model isn't configured or produced nothing
    usable, so the caller falls back to the speed NMT→MTPE chain. ``speed`` mode
    never calls this, so its timing is unchanged.
    """
    host = (getattr(settings, "OLLAMA_HOST", "") or "").strip()
    model = (getattr(settings, "TRANSLATION_QUALITY_MODEL", "") or "").strip()
    if not host or not model:
        logger.info("Translation quality mode requested but no Ollama host / "
                    "quality model configured — using the speed (NMT→MTPE) chain")
        return None
    # Honest about the cost: a 7-8B model can't run at GPU speed on a 4 GB card,
    # so Ollama places it largely on CPU.
    logger.warning(
        "Translation QUALITY mode: translating via %s. A 7-8B model can't fit at "
        "GPU speed on a 4 GB card, so Ollama runs it largely on CPU — expect this "
        "to be SUBSTANTIALLY slower than speed mode (minutes to tens of minutes on "
        "long videos) in exchange for higher quality.", model)
    if status_callback:
        try:
            r = status_callback(f"Translating with the CPU quality model {model} (slow)…")
            if hasattr(r, "__await__"):
                await r
        except Exception:
            pass
    num_ctx = int(getattr(settings, "OFFLINE_TRANSLATION_MTPE_NUM_CTX", 8192))
    client = _OllamaMTPEClient(model, num_ctx)
    try:
        llm_out = await translate_via_llm(
            segments, source_language, target_language, client,
            glossary=glossary, status_callback=status_callback)
    except Exception as e:
        logger.warning("Translation quality mode: LLM call failed (%s) — falling "
                       "back to the NMT→MTPE chain", e)
        return None
    if not llm_out:
        logger.warning("Translation quality mode: %s produced no usable output — "
                       "falling back to the NMT→MTPE chain", model)
        return None

    # ── NLLB completeness backstop ──────────────────────────────────────────
    # Fill any cue the big LLM left in the source script with offline NLLB so
    # quality mode is never LESS complete than the speed chain.
    if (target_language or "").lower() not in _CJK_LANGS:
        leftover = [i for i, s in enumerate(llm_out)
                    if _cjk_ratio(_seg_text(s)) > 0.30]
        if leftover:
            logger.info(
                "Translation quality mode: NLLB backstop filling %d/%d cue(s) the "
                "quality LLM left in the source language", len(leftover), len(llm_out))
            sub = [segments[i] for i in leftover]
            try:
                nllb_out = await _translate_via_nmt(
                    sub, source_language, target_language, glossary=glossary,
                    autodownload=bool(getattr(settings, "NMT_AUTODOWNLOAD", True)),
                    engine_pref="nllb",
                )
            except Exception as e:
                logger.warning("Translation quality mode: NLLB backstop failed (%s)", e)
                nllb_out = None
            if nllb_out and len(nllb_out) == len(sub):
                for k, i in enumerate(leftover):
                    llm_out[i] = nllb_out[k]
    logger.info("Translation quality mode: %d cue(s) translated via %s (+ NLLB "
                "backstop)", len(llm_out), model)
    return llm_out


async def mtpe_postedit_offline(
    nmt_segments: list[TranscriptSegment],
    source_segments: list | None,
    source_language: str,
    target_language: str,
    glossary: dict | None = None,
    status_callback=None,
) -> list[TranscriptSegment]:
    """MTPE post-edit of the offline NLLB / Opus-MT draft (Task 4).

    Feeds (source cue, NMT draft, surrounding context, glossary) to the local
    editorial model via the existing MTPE persona in ``transcript_polisher``,
    asking it to fix fluency / honorifics / idioms / glossary consistency —
    explicitly NOT to re-translate from scratch and NOT to alter timing or cue
    count. Small models post-edit far better than they translate cold, so this
    is the offline-quality parity move over the raw NLLB draft.

    Gated by ``OFFLINE_TRANSLATION_MTPE_ENABLED`` (default on) and only runs
    when an Ollama host + ``OLLAMA_TRANSLATION_MODEL`` are configured. Fail-soft:
    returns the raw NMT draft unchanged on any error, a bad/short response, a
    count mismatch, or if the post-edit would reintroduce the source language.
    """
    if not nmt_segments:
        return nmt_segments
    if not bool(getattr(settings, "OFFLINE_TRANSLATION_MTPE_ENABLED", True)):
        return nmt_segments
    host = (getattr(settings, "OLLAMA_HOST", "") or "").strip()
    model = (getattr(settings, "OLLAMA_TRANSLATION_MODEL", "") or "").strip()
    if not host or not model:
        logger.info("Offline MTPE skipped — no Ollama host/translation model "
                    "configured (keeping raw NMT draft)")
        return nmt_segments
    try:
        from backend.services.transcript_polisher import correct_transcript
    except Exception as e:
        logger.warning("Offline MTPE unavailable (%s) — keeping raw NMT draft", e)
        return nmt_segments

    num_ctx = int(getattr(settings, "OFFLINE_TRANSLATION_MTPE_NUM_CTX", 8192))
    # Source↔draft 1:1 alignment lets the model repair mistranslations against
    # the source; the offline NMT path is 1:1, so this normally holds.
    src_texts = [
        (s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")) or ""
        for s in (source_segments or [])
    ]
    if len(src_texts) != len(nmt_segments):
        src_texts = None
    # Per-video glossary target terms become canonical spellings the MTPE keeps.
    glossary_terms = sorted({
        (v or "").strip() for v in (glossary or {}).values() if (v or "").strip()
    }) or None

    if status_callback:
        try:
            r = status_callback("Post-editing the translation (offline MTPE)…")
            if hasattr(r, "__await__"):
                await r
        except Exception:
            pass

    logger.info(
        "Offline MTPE: post-editing %d cue(s) with %s (num_ctx=%d%s)",
        len(nmt_segments), model, num_ctx,
        ", source-aligned" if src_texts is not None else "",
    )
    client = _OllamaMTPEClient(model, num_ctx)
    try:
        polished = await correct_transcript(
            nmt_segments, client,
            language=target_language,
            source_texts=src_texts,
            source_language=source_language,
            glossary_terms=glossary_terms,
            mode="translation",
        )
    except Exception as e:
        logger.warning("Offline MTPE failed (%s) — keeping raw NMT draft", e)
        return nmt_segments

    # Strict shape: count must be preserved or we keep the raw draft (fail-soft).
    if not polished or len(polished) != len(nmt_segments):
        logger.warning(
            "Offline MTPE returned %d cue(s) (expected %d) — keeping raw NMT draft",
            len(polished) if polished else 0, len(nmt_segments))
        return nmt_segments
    # Safety: a post-edit must never REINTRODUCE the source language.
    try:
        before = fraction_untranslated(nmt_segments, target_language)
        after = fraction_untranslated(polished, target_language)
        if after > before + 0.02:
            logger.warning(
                "Offline MTPE reintroduced source language (%.0f%% → %.0f%% "
                "source-script) — keeping raw NMT draft", 100 * before, 100 * after)
            return nmt_segments
    except Exception:
        pass
    _changed = sum(
        1 for a, b in zip(nmt_segments, polished)
        if (getattr(a, "text", "") or "") != (getattr(b, "text", "") or ""))
    logger.info("Offline MTPE: refined %d/%d cue(s) (count + timing preserved)",
                _changed, len(polished))
    return polished


def _apply_batch_translations(
    batch: list[TranscriptSegment],
    translations: list,
) -> list[TranscriptSegment]:
    """Convert raw translation strings + original segments into translated TranscriptSegments."""
    result = []
    for i, trans_text in enumerate(translations):
        orig = batch[i]
        if isinstance(trans_text, str) and trans_text.strip():
            trans_words = _generate_proportional_word_timestamps(
                trans_text.strip(), orig.start, orig.end
            )
            result.append(TranscriptSegment(
                start=orig.start,
                end=orig.end,
                text=trans_text.strip(),
                speaker=orig.speaker,
                words=trans_words,
                confidence=orig.confidence,
            ))
        else:
            result.append(orig)
    return result


def _idiomatic_rule() -> str:
    """Extra directive pushing the model toward natural, idiomatic target-language
    phrasing (toggle: ``TRANSLATION_IDIOMATIC``). Returns '' when disabled.

    Kept placeholder-free so it survives ``TRANSLATION_PROMPT.format()`` as a
    literal value, and pairs with the existing 'preserve meaning / no additions'
    rules so 'idiomatic' never licenses embellishment."""
    if not bool(getattr(settings, "TRANSLATION_IDIOMATIC", True)):
        return ""
    return (
        "\nIDIOMATIC PHRASING (top priority on wording): render each line the way "
        "a professional dub / localization writer would actually say it in the "
        "target language. Recast the source sentence structure into natural, "
        "idiomatic phrasing — never a word-order calque of the source. Preserve "
        "the exact meaning and every piece of information; do NOT add, omit, "
        "soften, or embellish content. Natural wording, faithful substance."
    )


def _format_glossary_block(glossary: dict | None) -> str:
    """Format a per-video glossary as a prompt block."""
    if not glossary or not isinstance(glossary, dict):
        return ""
    pairs = [(str(k), str(v)) for k, v in glossary.items() if str(k).strip() and str(v).strip()]
    if not pairs:
        return ""
    lines = ["GLOSSARY (always use these exact translations for these terms):"]
    for src, tgt in pairs:
        lines.append(f"  {src} → {tgt}")
    return "\n".join(lines) + "\n\n"


def _format_context_block(label: str, items: list[TranscriptSegment]) -> str:
    """Format a context window as a prompt block."""
    if not items:
        return ""
    lines = [f"{label} (do NOT translate — context only):"]
    for s in items:
        lines.append(f"  [{s.start:.0f}s] {s.text}")
    return "\n".join(lines) + "\n"


async def translate_segments(
    segments: list[TranscriptSegment],
    source_language: str,
    target_language: str,
    orchestrator: AIOrchestrator,
    batch_size: int = 25,
    progress_callback=None,
    context_window: int | None = None,
    glossary: dict | None = None,
    model_override: str | None = None,
) -> list[TranscriptSegment]:
    """Translate transcript segments to the target language.

    Returns new TranscriptSegment[] with translated text and
    proportional word timestamps. Original timing is preserved.

    ``context_window`` controls the number of surrounding segments included
    as reference-only context (default: TRANSLATION_CONTEXT_WINDOW setting).
    ``glossary`` is a {source_term: target_term} dict whose entries are
    pinned in the prompt so proper nouns and recurring terms get
    translated consistently across batches.
    """
    if source_language == target_language:
        return segments

    source_name = SUPPORTED_LANGUAGES.get(source_language, source_language)
    target_name = SUPPORTED_LANGUAGES.get(target_language, target_language)

    # If source is "auto" or unknown, try to detect from segment text
    if source_language in ("auto", "") and segments:
        source_name = "the original language"

    # Smaller batches for CJK languages — denser text, LLMs lose count easily
    if source_language.lower() in _CJK_LANGS or target_language.lower() in _CJK_LANGS:
        batch_size = min(batch_size, 10)
        logger.info("Using smaller batch size (%d) for CJK translation", batch_size)

    extra_rules = _get_pair_rules(source_language, target_language) + _idiomatic_rule()
    if context_window is None:
        context_window = int(getattr(settings, "TRANSLATION_CONTEXT_WINDOW", 5))
    context_window = max(0, min(20, int(context_window)))
    use_glossary = bool(getattr(settings, "TRANSLATION_GLOSSARY_ENABLED", True))
    glossary_block = _format_glossary_block(glossary) if use_glossary else ""
    # Auto-derived recurring proper-noun glossary (content-agnostic, built once
    # from all segments) — keeps names consistent across batches.
    auto_terms_block = ""
    if bool(getattr(settings, "TRANSLATION_AUTO_GLOSSARY", True)):
        try:
            from backend.services.glossary import build_translation_glossary_block
            auto_terms_block = build_translation_glossary_block(
                segments, source_language, target_name)
        except Exception:
            auto_terms_block = ""

    translated = []
    total_batches = (len(segments) + batch_size - 1) // batch_size
    consecutive_failures = 0
    MAX_CONSECUTIVE_BATCH_FAILURES = 3
    # A free / rate-limited model won't recover within a job — once we see a
    # couple of clear HTTP 429s, stop instead of crawling through every batch
    # (each 429 retry adds backoff, so a 1400-segment job would take hours).
    rate_limited_hits = 0
    MAX_RATE_LIMITED_HITS = 2

    for batch_idx, batch_start in enumerate(range(0, len(segments), batch_size)):
        batch = segments[batch_start : batch_start + batch_size]

        # Include surrounding segments as context (not to be translated)
        context_before = segments[max(0, batch_start - context_window): batch_start]
        ctx_after_start = batch_start + len(batch)
        context_after = segments[ctx_after_start: ctx_after_start + context_window]

        context_section = (
            _format_context_block("PREVIOUS CONTEXT", context_before)
            + _format_context_block("FOLLOWING CONTEXT", context_after)
        )
        if context_section:
            context_section += "\n"

        seg_texts = [{"index": i, "text": seg.text} for i, seg in enumerate(batch)]
        prompt = glossary_block + auto_terms_block + TRANSLATION_PROMPT.format(
            source_lang=source_name,
            target_lang=target_name,
            count=len(batch),
            segments_json=json.dumps(seg_texts, ensure_ascii=False, indent=2),
            extra_rules=extra_rules,
            context_section=context_section,
        )

        batch_success = False
        for attempt in range(2):  # 2 attempts per batch
            try:
                response = await orchestrator.text_completion(
                    prompt, timeout=120, model_override=model_override)
                translations = _parse_translation_response(response)

                if isinstance(translations, list):
                    if len(translations) == len(batch):
                        translated.extend(_apply_batch_translations(batch, translations))
                        batch_success = True
                        break
                    elif len(translations) > 0 and len(translations) < len(batch):
                        logger.warning(
                            "Translation batch %d: got %d/%d translations — applying partial",
                            batch_idx, len(translations), len(batch),
                        )
                        partial = _apply_batch_translations(batch[:len(translations)], translations)
                        partial.extend(batch[len(translations):])
                        translated.extend(partial)
                        batch_success = True
                        break
                    else:
                        logger.warning("Translation batch %d attempt %d: wrong count (got %d, expected %d)",
                                       batch_idx, attempt + 1,
                                       len(translations), len(batch))
                else:
                    logger.warning("Translation batch %d attempt %d: invalid response type",
                                   batch_idx, attempt + 1)
            except Exception as e:
                if _looks_rate_limited(e):
                    rate_limited_hits += 1
                logger.warning("Translation batch %d attempt %d failed: %s",
                               batch_idx, attempt + 1, e)

        # Bail loudly on SUSTAINED upstream rate-limiting (consecutive 429s
        # with no success in between) rather than grinding through hundreds of
        # slow, 429-throttled batches — each 429 carries backoff (Task 2). The
        # counter resets on any success, so a transient-but-recovered 429
        # never trips this.
        if not batch_success and rate_limited_hits >= MAX_RATE_LIMITED_HITS:
            raise TranslationRateLimitedError(
                "Translation model rate-limited upstream (HTTP 429) — using an "
                "offline model (NMT) or a paid/local model is recommended"
            )

        if not batch_success:
            translated.extend(batch)  # Keep originals for this batch
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_BATCH_FAILURES:
                logger.error(
                    "Translation: %d consecutive batch failures — aborting remaining batches",
                    consecutive_failures,
                )
                remaining_start = batch_start + batch_size
                translated.extend(segments[remaining_start:])
                break
        else:
            consecutive_failures = 0
            rate_limited_hits = 0

        if progress_callback:
            pct = int(((batch_idx + 1) / total_batches) * 100)
            await progress_callback(pct)

    logger.info(
        "Translated %d segments from %s to %s",
        len(translated), source_name, target_name,
    )
    return translated


async def _translate_via_nmt(
    segments: list[TranscriptSegment],
    source_language: str,
    target_language: str,
    glossary: dict | None,
    progress_callback=None,
    autodownload: bool = True,
    engine_pref: str = "nllb",
    status_callback=None,
) -> list[TranscriptSegment] | None:
    """Translate offline via local NMT (Opus-MT preferred, NLLB fallback).

    When no local model is on disk and ``autodownload`` is on, fetch +
    convert one on demand (one-time) and then translate — so offline
    translation is the automatic default with no manual Settings step.
    Returns ``None`` only when the pair is unsupported OR the auto-download
    failed, so the caller can fall back to the LLM and log that distinctly.
    """
    try:
        from backend.services.nmt_translator import (
            pick_local_engine, auto_download_for_pair,
            NMTTranslator, OpusMTTranslator,
            iso_to_flores, _flores_is_cjk, _looks_cjk,
        )
    except Exception as e:
        logger.warning("NMT module unavailable: %s", e)
        return None
    engine = None
    # FuguMT (Japanese-specialised Marian model) when explicitly requested —
    # loaded through the Opus-MT machinery in its own cache dir. FuguMT is
    # ja↔en only, so anything else (or a load failure) falls through to the
    # standard NLLB path below.
    if engine_pref == "fugumt":
        try:
            from backend.services.nmt_translator import get_marian_variant
            engine = await asyncio.to_thread(
                get_marian_variant, source_language, target_language, "fugumt",
                getattr(settings, "NMT_FUGUMT_TEMPLATE", "staka/fugumt-{src}-{tgt}"),
                bool(autodownload),
            )
        except Exception as _fm_err:
            logger.warning(
                "FuguMT unavailable for %s→%s (%s) — falling back to NLLB",
                source_language, target_language, _fm_err)
            engine = None
    if engine is None:
        engine = pick_local_engine(source_language, target_language)
    if engine is None and autodownload:
        # Auto-download the offline model on demand (Task 1). NLLB-200 is the
        # preferred single-model download; Opus-MT only when explicitly asked.
        prefer = "opus-mt" if engine_pref == "opus-mt" else "nllb"
        logger.info(
            "NMT: no local %s model for %s→%s — auto-downloading (one-time) before translating",
            prefer, source_language, target_language,
        )
        if status_callback:
            try:
                res = status_callback("Downloading offline translation model (one-time)…")
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                pass
        try:
            engine = await asyncio.to_thread(
                auto_download_for_pair, source_language, target_language, prefer,
            )
        except Exception as dl_err:
            # Network / disk / converter failure — this is the ONLY case where
            # we fall back to the LLM, and we log it distinctly (Task 1).
            logger.error(
                "NMT: auto-download of offline model for %s→%s FAILED (%s) — "
                "falling back to the LLM translation path",
                source_language, target_language, dl_err,
            )
            return None
    if engine is None:
        logger.info(
            "NMT: no local model for %s→%s (auto-download off or unsupported) "
            "— caller will fall back to LLM",
            source_language, target_language,
        )
        return None
    # Name the exact engine + model used for this job so the log makes it
    # unambiguous which offline backend ran (and that no LLM was called).
    if isinstance(engine, NMTTranslator):
        model_label = engine.model_id
    else:
        model_label = getattr(engine, "hf_repo", None) or "Helsinki-NLP/opus-mt-{}-{}".format(
            getattr(engine, "source", source_language),
            getattr(engine, "target", target_language),
        )
    logger.info(
        "NMT: using %s (%s) for %s→%s (%d segments)",
        type(engine).__name__, model_label, source_language, target_language, len(segments),
    )
    context_window = max(0, min(20, int(getattr(settings, "TRANSLATION_CONTEXT_WINDOW", 5))))
    out: list[TranscriptSegment] = []
    try:
        # Batch in groups of 16 to keep memory + decode time bounded.
        BATCH = 16
        for start in range(0, len(segments), BATCH):
            batch = segments[start: start + BATCH]
            ctx_before = [
                s.text for s in segments[max(0, start - context_window): start]
            ]
            ctx_after = [
                s.text for s in segments[start + len(batch): start + len(batch) + context_window]
            ]
            texts = [s.text for s in batch]
            if isinstance(engine, NMTTranslator):
                translations = engine.translate_with_context(
                    texts, ctx_before, ctx_after,
                    source_language, target_language, glossary=glossary,
                )
            else:
                translations = engine.translate_batch(texts, glossary=glossary)
            out.extend(_apply_batch_translations(batch, translations))
            _done = min(start + len(batch), len(segments))
            # Granular progress in the processing log (mirrors the LLM path at
            # translate_segments) so an offline-NMT translation shows
            # "Translating subtitles… (N/M)" rather than only a generic
            # "Still processing…" heartbeat — the LLM-failed→NMT-fallback case
            # otherwise displayed no per-batch progress at all.
            if status_callback:
                try:
                    res = status_callback(f"Translating subtitles… ({_done}/{len(segments)})")
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass
            if progress_callback:
                pct = int((_done / max(1, len(segments))) * 100)
                try:
                    res = progress_callback(pct)
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass

        # ── Completeness pass: re-translate any cue the batched pass left
        # UNtranslated, for ANY target language (offline-only; the AI is never
        # called here). A long run-on can exceed the decoder and come back as
        # source text, and for some pairs NLLB simply echoes a cue. We retry
        # those in ISOLATION — the per-cue path chunks long inputs, so it
        # succeeds where the batched / context-joined pass did not — so no
        # source-language text is ever left behind regardless of the target.
        _src_l = (source_language or "").strip().lower().split("-")[0]
        _tgt_l = (target_language or "").strip().lower().split("-")[0]
        _tgt_is_cjk = _flores_is_cjk(iso_to_flores(target_language))

        def _untranslated(translated_text: str, source_text: str) -> bool:
            t = (translated_text or "").strip()
            s = (source_text or "").strip()
            if not s:
                return False
            if not t:
                return True
            # Output still in a CJK source script while the target isn't CJK.
            if not _tgt_is_cjk and _looks_cjk(t):
                return True
            # Unchanged from source (NMT dropped / echoed it). Only flag
            # substantial cues so trivial tokens (names, numbers, "OK") aren't
            # needlessly retried.
            if t == s and len(s) >= 8 and len(s.split()) >= 2:
                return True
            return False

        if _src_l and _tgt_l and _src_l != _tgt_l:
            leftover = [i for i, t in enumerate(out)
                        if _untranslated(getattr(t, "text", ""),
                                         getattr(segments[i], "text", ""))]
            if leftover:
                logger.info(
                    "NMT: completeness pass — %d/%d cue(s) untranslated, retrying per-cue",
                    len(leftover), len(out),
                )
                recovered = 0
                for i in leftover:
                    orig = segments[i]
                    src_text = getattr(orig, "text", "") or ""
                    if not src_text.strip():
                        continue
                    try:
                        if isinstance(engine, NMTTranslator):
                            retry = engine.translate_batch(
                                [src_text], source_language, target_language, glossary=glossary)
                        else:
                            retry = engine.translate_batch([src_text], glossary=glossary)
                    except Exception as _re:
                        logger.debug("NMT: completeness retry failed for cue %d (%s)", i, _re)
                        continue
                    new_text = (retry[0] if retry else "") or ""
                    if new_text.strip() and not _untranslated(new_text, src_text):
                        out[i] = _apply_batch_translations([orig], [new_text])[0]
                        recovered += 1
                logger.info("NMT: completeness pass recovered %d/%d cue(s)",
                            recovered, len(leftover))
        # Context-join telemetry (Task 3): how the numbered-tag context join
        # fared across the job — clean tag re-alignment vs. the per-cue-with-
        # context fallback (tag loss or block-too-long). Tag failure should be
        # the exception; even when it falls to per-cue, context is retained.
        if isinstance(engine, NMTTranslator):
            _ok = getattr(engine, "_ctx_join_ok", 0)
            _tf = getattr(engine, "_ctx_join_tag_fail", 0)
            _tl = getattr(engine, "_ctx_join_too_long", 0)
            _attempted = _ok + _tf
            if _ok or _tf or _tl:
                _rate = (100.0 * _ok / _attempted) if _attempted else 0.0
                logger.info(
                    "NMT: context-join — %d ok / %d tag-fail (%.0f%% success when "
                    "attempted), %d too-long → per-cue-with-context",
                    _ok, _tf, _rate, _tl,
                )
    finally:
        try:
            engine.unload()
        except Exception:
            pass
    changed = sum(1 for t, o in zip(out, segments) if t.text != o.text)
    if changed == 0:
        logger.warning("NMT: 0 segments changed — returning None so caller can fall back")
        return None
    logger.info("NMT: translated %d/%d segments via local engine", changed, len(out))
    return out


def _is_ja_en_pair(source: str, target: str) -> bool:
    """True for a Japanese↔English pair (either direction), tolerant of ISO /
    Flores / language-name spellings (ja/jpn/japanese, en/eng/english)."""
    def _norm(x):
        return (x or "").strip().lower().split("-")[0].split("_")[0]
    s, t = _norm(source), _norm(target)
    s_ja, t_ja = s in ("ja", "jpn", "jp", "japanese"), t in ("ja", "jpn", "jp", "japanese")
    s_en, t_en = s in ("en", "eng", "english"), t in ("en", "eng", "english")
    return (s_ja and t_en) or (s_en and t_ja)


def _resolve_translation_engine(source: str, target: str) -> str:
    """Pick the translation engine to use based on settings + availability.

    Returns one of: ``"deepl"``, ``"google"``, ``"nllb"``, ``"opus-mt"``,
    ``"fugumt"``. Translation is offline-only and never resolves to an LLM. The
    router in ``translate_segments_with_fallback`` consults this when
    ``TRANSLATION_ENGINE=auto``.
    """
    requested = (getattr(settings, "TRANSLATION_ENGINE", "auto") or "auto").lower()
    # Concrete engines bypass auto-selection. ``llm``/``whisper`` are no longer
    # text-translation engines here — translation is offline-only and the AI
    # model only polishes — so they fall through to AUTO, which always resolves
    # to an offline NMT engine. (Whisper's native audio→English path is handled
    # upstream in the pipeline before this text router is ever consulted.)
    if requested in ("deepl", "google", "opus-mt", "nllb", "fugumt"):
        return requested
    # AUTO selection — offline-first, never an LLM.
    if (getattr(settings, "DEEPL_API_KEY", "") or "").strip():
        return "deepl"
    if (getattr(settings, "GOOGLE_TRANSLATE_API_KEY", "") or "").strip():
        return "google"
    # Japanese↔English: FuguMT (a JA-specialist Marian model) is markedly more
    # accurate than the NLLB/Opus generalists on everyday vocabulary, so prefer
    # it for this pair offline. Auto-downloads on first use and falls back to
    # NLLB if it can't load. Disable via NMT_PREFER_FUGUMT_JA_EN.
    if (bool(getattr(settings, "NMT_PREFER_FUGUMT_JA_EN", True))
            and _is_ja_en_pair(source, target)):
        return "fugumt"
    try:
        from backend.services.nmt_translator import pick_local_engine, iso_to_flores
        engine = pick_local_engine(source, target)
        if engine is not None:
            return "opus-mt" if engine.__class__.__name__ == "OpusMTTranslator" else "nllb"
        # No local model on disk yet. Offline NMT is the only translator, so
        # resolve to the best NMT engine regardless of the auto-download flag:
        # the NMT path fetches + converts it on first use when downloads are on,
        # or fails with a clear NMT error when they're off (never an LLM).
        engine_name = "nllb" if (iso_to_flores(source) and iso_to_flores(target)) else "opus-mt"
        if bool(getattr(settings, "NMT_AUTODOWNLOAD", True)):
            logger.info(
                "NMT: no local model for %s→%s yet — will auto-download %s "
                "(one-time, then fully offline)",
                source, target, engine_name,
            )
        else:
            logger.warning(
                "NMT: no local model for %s→%s and auto-download is off — the NMT "
                "engine will fail unless a model is already present "
                "(translation is offline-only, no LLM fallback)",
                source, target,
            )
        return engine_name
    except Exception as e:
        logger.warning("NMT: engine probe failed for %s→%s (%s)", source, target, e)
    # Probe failed entirely — still resolve to NLLB so the NMT path surfaces a
    # concrete, actionable error rather than silently relabelling the source.
    return "nllb"


async def translate_segments_with_fallback(
    segments: list[TranscriptSegment],
    source_language: str,
    target_language: str,
    orchestrator: AIOrchestrator,
    batch_size: int = 25,
    progress_callback=None,
    glossary: dict | None = None,
    status_callback=None,
) -> list[TranscriptSegment]:
    """Translate segments with OFFLINE engines only — never an LLM.

    This is the text-based router. For non-English → English, the pipeline
    runs Whisper's native audio→English translate task *before* calling this
    (see ``WHISPER_TRANSLATE_TO_EN``); this router handles every other case.

    Decision tree (driven by TRANSLATION_ENGINE):
      - ``auto``    → DeepL → Google (both key-gated) → Opus-MT / NLLB
                      (auto-downloaded offline default)
      - ``deepl``   → DeepL API (requires key)
      - ``google``  → Google Cloud Translation v3 (requires key)
      - ``opus-mt`` → Opus-MT local (auto-downloaded if missing)
      - ``nllb``    → NLLB-200 local (auto-downloaded if missing)

    The AI model never translates here; it only polishes the translated text
    downstream. ``orchestrator``/``batch_size`` are accepted for call-site
    compatibility but unused.

    ``status_callback(msg)`` (optional, sync or async) surfaces coarse status
    such as the one-time offline-model download to the websocket.

    Raises ``TranslationFailedError`` when offline translation can't complete,
    so the caller never relabels the untranslated source as a translation.
    """
    if source_language == target_language:
        return segments

    engine = _resolve_translation_engine(source_language, target_language)
    logger.info("Translation engine resolved: %s (requested=%s)",
                engine, getattr(settings, "TRANSLATION_ENGINE", "auto"))

    # Translation is OFFLINE-ONLY and never uses an LLM: the AI model's sole
    # job is to polish the already-translated text for readability. For
    # non-English → English the pipeline runs Whisper's native translate task
    # before reaching this router; everything else routes to the local NMT
    # engines (Opus-MT / NLLB), with the key-gated cloud services (DeepL,
    # Google) used only when the user has explicitly configured an API key.

    # ── Cloud NMT engines (DeepL, Google) — key-gated, never the default ──
    if engine == "deepl":
        try:
            out = await _translate_via_deepl(segments, source_language, target_language, glossary)
            if out is not None:
                return out
        except Exception as e:
            logger.warning("DeepL translation failed: %s — falling back", e)
    if engine == "google":
        try:
            out = await _translate_via_google(segments, source_language, target_language, glossary)
            if out is not None:
                return out
        except Exception as e:
            logger.warning("Google Translate failed: %s — falling back", e)

    # ── Local NMT engines (Opus-MT, NLLB) — offline default, auto-downloaded ──
    _nmt_error: Exception | None = None
    if engine in ("opus-mt", "nllb", "fugumt"):
        # Quality mode (Task 5): try the larger CPU LLM + NLLB backstop first.
        # If it's not configured / produced nothing, fall through to the speed
        # NMT→MTPE chain below. Speed mode never enters this branch.
        if translation_quality_mode_active():
            try:
                q = await _translate_quality_mode(
                    segments, source_language, target_language,
                    glossary, status_callback)
                if q is not None:
                    return q
            except Exception as e:
                logger.warning("Translation quality mode errored (%s) — using the "
                               "speed NMT→MTPE chain", e)
        try:
            out = await _translate_via_nmt(
                segments, source_language, target_language,
                glossary=glossary, progress_callback=progress_callback,
                autodownload=bool(getattr(settings, "NMT_AUTODOWNLOAD", True)),
                engine_pref=engine,
                status_callback=status_callback,
            )
            if out is not None:
                # MTPE post-edit (Task 4): the NLLB / Opus-MT draft is fluent but
                # rough — polish it with the dedicated local translation model
                # (OLLAMA_TRANSLATION_MODEL) as an MT post-editor. Fully fail-soft
                # (returns the raw draft on any error), so a MTPE blow-up can
                # never turn a successful NMT translation into a hard failure.
                try:
                    out = await mtpe_postedit_offline(
                        out, segments, source_language, target_language,
                        glossary=glossary, status_callback=status_callback,
                    )
                except Exception as _mt_e:
                    logger.warning(
                        "Offline MTPE pass errored (%s) — keeping raw NMT draft", _mt_e)
                # Auto name-consistency: small offline models spell recurring
                # proper nouns several ways ("Doria"/"Dorian", "Zechs"/"Zex");
                # unify the rare variants to the dominant spelling. Automatic
                # (no glossary needed) + output-only, so it can't break the
                # translation. Makes names consistent, not necessarily official.
                if bool(getattr(settings, "NMT_AUTO_NAME_CONSISTENCY", True)):
                    try:
                        from backend.services.name_consistency import (
                            unify_proper_noun_variants,
                        )
                        out, _n_fixed = unify_proper_noun_variants(out)
                        if _n_fixed:
                            logger.info(
                                "Name consistency: unified %d proper-noun variant(s) "
                                "to their dominant spelling", _n_fixed)
                    except Exception as _nc_e:
                        logger.warning(
                            "Name-consistency pass skipped (%s)", _nc_e)
                return out
        except Exception as e:
            _nmt_error = e
            logger.error("NMT translation failed: %s", e)

    # Offline translation is the only path — there is NO LLM fallback. The AI
    # model only polishes the translated text downstream; it never translates.
    # Fail loudly with an actionable reason so the caller keeps the (labelled)
    # source transcript instead of silently relabelling it as a translation.
    raise TranslationFailedError(
        f"Offline translation failed for {source_language}→{target_language} "
        f"(engine={engine}"
        + (f": {_nmt_error}" if _nmt_error else "")
        + "). Translation is offline-only and never uses an LLM. Ensure the NMT "
        "model can be downloaded/loaded (TRANSLATION_ENGINE=auto with "
        "NMT_AUTODOWNLOAD on), or for X→English enable WHISPER_TRANSLATE_TO_EN."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Cloud NMT engines — DeepL + Google Cloud Translation v3
# ═══════════════════════════════════════════════════════════════════════════

# DeepL's supported source/target codes (a subset of ISO 639-1 with a few
# regional variants). Used to short-circuit ``auto`` selection so we don't
# attempt a pair DeepL can't handle.
_DEEPL_LANGS = {
    "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hu", "id", "it", "ja", "ko", "lt", "lv", "nb", "nl", "pl", "pt",
    "ro", "ru", "sk", "sl", "sv", "tr", "uk", "zh",
}


def _iso_to_deepl_lang(code: str, *, is_target: bool) -> str | None:
    if not code:
        return None
    c = code.lower().split("-")[0]
    if c not in _DEEPL_LANGS:
        return None
    if c == "en" and is_target:
        return "EN-US"
    if c == "pt" and is_target:
        return "PT-BR"
    if c == "zh" and is_target:
        return "ZH-HANS"
    return c.upper()


async def _translate_via_deepl(
    segments: list[TranscriptSegment],
    source_language: str,
    target_language: str,
    glossary: dict | None,
) -> list[TranscriptSegment] | None:
    """Translate via DeepL's REST API. Returns None when the key is missing
    or the pair is unsupported."""
    key = (getattr(settings, "DEEPL_API_KEY", "") or "").strip()
    if not key:
        return None
    src = _iso_to_deepl_lang(source_language, is_target=False)
    tgt = _iso_to_deepl_lang(target_language, is_target=True)
    if not src or not tgt:
        logger.info("DeepL: %s→%s is not in DeepL's supported pairs",
                    source_language, target_language)
        return None
    endpoints = [
        "https://api.deepl.com/v2/translate",
        "https://api-free.deepl.com/v2/translate",
    ]
    headers = {"Authorization": f"DeepL-Auth-Key {key}"}

    out: list[TranscriptSegment] = []
    CHUNK = 50
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for start in range(0, len(segments), CHUNK):
            batch = segments[start: start + CHUNK]
            texts = [s.text for s in batch]
            payload = {
                "text": texts,
                "source_lang": src,
                "target_lang": tgt,
                "preserve_formatting": "1",
            }
            translations: list[str] | None = None
            last_err: str | None = None
            for url in endpoints:
                try:
                    resp = await client.post(url, data=payload, headers=headers)
                    if resp.status_code == 403:
                        last_err = f"HTTP 403 at {url}"
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    translations = [t.get("text", "") for t in data.get("translations", [])]
                    if len(translations) != len(texts):
                        translations = None
                        last_err = f"DeepL returned wrong count"
                        continue
                    break
                except Exception as e:
                    last_err = str(e)
                    continue
            if translations is None:
                logger.warning("DeepL chunk failed: %s", last_err or "unknown")
                return None
            if glossary:
                for i, src_text in enumerate(texts):
                    new = translations[i]
                    for k, v in glossary.items():
                        ks, vs = (k or "").strip(), (v or "").strip()
                        if ks and vs and ks in src_text and ks in new:
                            new = new.replace(ks, vs)
                    translations[i] = new
            out.extend(_apply_batch_translations(batch, translations))
    logger.info("DeepL: translated %d segments %s→%s", len(out), src, tgt)
    return out


async def _translate_via_google(
    segments: list[TranscriptSegment],
    source_language: str,
    target_language: str,
    glossary: dict | None,
) -> list[TranscriptSegment] | None:
    """Translate via Google Cloud Translation v2 REST API. Returns None
    when the key is missing."""
    key = (getattr(settings, "GOOGLE_TRANSLATE_API_KEY", "") or "").strip()
    if not key:
        return None
    url = f"https://translation.googleapis.com/language/translate/v2?key={key}"
    out: list[TranscriptSegment] = []
    CHUNK = 100
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for start in range(0, len(segments), CHUNK):
            batch = segments[start: start + CHUNK]
            texts = [s.text for s in batch]
            payload = {
                "q": texts,
                "target": target_language.split("-")[0],
                "format": "text",
            }
            if source_language and source_language not in ("auto", ""):
                payload["source"] = source_language.split("-")[0]
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                items = data.get("data", {}).get("translations", [])
                translations = [it.get("translatedText", "") for it in items]
                if len(translations) != len(texts):
                    logger.warning("Google Translate returned %d for %d inputs",
                                   len(translations), len(texts))
                    return None
            except Exception as e:
                logger.warning("Google Translate chunk failed: %s", e)
                return None
            if glossary:
                for i, src_text in enumerate(texts):
                    new = translations[i]
                    for k, v in glossary.items():
                        ks, vs = (k or "").strip(), (v or "").strip()
                        if ks and vs and ks in src_text and ks in new:
                            new = new.replace(ks, vs)
                    translations[i] = new
            out.extend(_apply_batch_translations(batch, translations))
    logger.info("Google Translate: translated %d segments to %s", len(out), target_language)
    return out

