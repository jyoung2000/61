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

SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
    "ko": "Korean", "zh": "Chinese (Simplified)", "ar": "Arabic",
    "hi": "Hindi", "nl": "Dutch", "pl": "Polish", "tr": "Turkish",
    "vi": "Vietnamese", "th": "Thai", "uk": "Ukrainian", "sv": "Swedish",
    "id": "Indonesian", "ms": "Malay", "tl": "Filipino",
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
) -> str:
    """Call Ollama chat API directly with a dedicated translation model."""
    host = settings.OLLAMA_HOST
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0)) as client:
        resp = await client.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_ctx": 4096,
                    "temperature": 0.3,
                    "num_predict": 4096,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


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

    extra_rules = _get_pair_rules(source_language, target_language)
    if context_window is None:
        context_window = int(getattr(settings, "TRANSLATION_CONTEXT_WINDOW", 5))
    context_window = max(0, min(20, int(context_window)))
    use_glossary = bool(getattr(settings, "TRANSLATION_GLOSSARY_ENABLED", True))
    glossary_block = _format_glossary_block(glossary) if use_glossary else ""

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
        prompt = glossary_block + TRANSLATION_PROMPT.format(
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
        model_label = "Helsinki-NLP/opus-mt-{}-{}".format(
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
            if progress_callback:
                pct = int(((start + len(batch)) / max(1, len(segments))) * 100)
                try:
                    res = progress_callback(pct)
                    if hasattr(res, "__await__"):
                        await res
                except Exception:
                    pass

        # ── Completeness pass: re-translate any cue still in the SOURCE script ──
        # A long run-on can exceed the decoder and come back untranslated (the
        # engine keeps the source for a failed cue). For a non-CJK target, detect
        # leftover source-script cues and retry them in ISOLATION — the per-cue
        # path chunks long inputs, so it succeeds where the batched / context-
        # joined pass did not. Offline-only; the AI is never called here.
        if not _flores_is_cjk(iso_to_flores(target_language)):
            leftover = [i for i, t in enumerate(out)
                        if _looks_cjk(getattr(t, "text", "") or "")]
            if leftover:
                logger.info(
                    "NMT: completeness pass — %d/%d cue(s) still in source script, "
                    "retrying per-cue", len(leftover), len(out),
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
                    if new_text.strip() and not _looks_cjk(new_text):
                        out[i] = _apply_batch_translations([orig], [new_text])[0]
                        recovered += 1
                logger.info("NMT: completeness pass recovered %d/%d cue(s)",
                            recovered, len(leftover))
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


def _resolve_translation_engine(source: str, target: str) -> str:
    """Pick the translation engine to use based on settings + availability.

    Returns one of: ``"deepl"``, ``"google"``, ``"nllb"``, ``"opus-mt"``.
    Translation is offline-only and never resolves to an LLM. The router in
    ``translate_segments_with_fallback`` consults this when
    ``TRANSLATION_ENGINE=auto``.
    """
    requested = (getattr(settings, "TRANSLATION_ENGINE", "auto") or "auto").lower()
    # Concrete engines bypass auto-selection. ``llm``/``whisper`` are no longer
    # text-translation engines here — translation is offline-only and the AI
    # model only polishes — so they fall through to AUTO, which always resolves
    # to an offline NMT engine. (Whisper's native audio→English path is handled
    # upstream in the pipeline before this text router is ever consulted.)
    if requested in ("deepl", "google", "opus-mt", "nllb"):
        return requested
    # AUTO selection — offline-first, never an LLM.
    if (getattr(settings, "DEEPL_API_KEY", "") or "").strip():
        return "deepl"
    if (getattr(settings, "GOOGLE_TRANSLATE_API_KEY", "") or "").strip():
        return "google"
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
    if engine in ("opus-mt", "nllb"):
        try:
            out = await _translate_via_nmt(
                segments, source_language, target_language,
                glossary=glossary, progress_callback=progress_callback,
                autodownload=bool(getattr(settings, "NMT_AUTODOWNLOAD", True)),
                engine_pref=engine,
                status_callback=status_callback,
            )
            if out is not None:
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

