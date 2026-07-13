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
import re
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


# ── Romaji (transliterated Japanese) detection ───────────────────────────────
# A model sometimes echoes a hard Japanese line PHONETICALLY in Latin letters
# ("Katte ippai tsukuritaku naru toko aru ne") instead of translating it. That
# is untranslated source, but it carries NO CJK script, so the _cjk_ratio check
# scores it 0% and it ships in the "English" track. A Japanese romaji word
# decomposes into open (consonant+vowel) mora with optional gemination / long
# vowels; Latin words with L/Q/V/X, consonant clusters, or final consonants
# (other than 'n') fail the pattern — which is what separates romaji from English.
_MORA_WORD = re.compile(
    r"^(?:"
    r"(?:[kgsztdnhbpmr]?[kgsztdnhbpmr])?"                       # optional gemination
    r"(?:ky|gy|sh|ch|ts|ny|hy|by|py|my|ry|dy|[kgsztdnhfbpmyrwj])?"
    r"[aeiouāēīōū]"
    r"|n)+$",
    re.IGNORECASE,
)
# Short romaji tokens that are ALSO ordinary English words — never count these as
# Japanese evidence on their own (keeps English lines from being flagged).
_ROMAJI_AMBIG = {
    "a", "i", "o", "no", "to", "na", "ka", "me", "he", "we", "so", "re",
    "in", "on", "an", "at", "it", "is", "be", "as", "or", "up", "us",
}
_JA_SOURCE = {"ja", "jpn", "japanese", "ja-jp"}


# Distinctive Japanese function words / honorifics in romaji. These are
# near-zero-frequency in English text, so 2+ hits is strong evidence even
# when the mora ratio is diluted by a half-translated (mixed) line.
_ROMAJI_HINTS = re.compile(
    r"\b(?:desu|masu|kudasai|arigatou?|gomen(?:asai)?|sensei|senpai|"
    r"onee|onii|chan|kun|sama|kawaii|sugoi|hontou?|nani|chotto|dame|"
    r"yatta|daijoubu|itadakimasu|oishii|kimochi|urusai|baka|"
    r"n[' ]?da(?:yo|ne)?|ndesho|mashita|shite(?:ru)?|nakute|kedo)\b",
    re.IGNORECASE,
)


def _romaji_token_stats(toks: list) -> float:
    if len(toks) < 3:
        return 0.0
    jp = 0
    for t in toks:
        if t.lower() in _ROMAJI_AMBIG:
            continue
        if len(t) >= 3 and _MORA_WORD.match(t):
            jp += 1
    return jp / len(toks)


def _romaji_ja_ratio(text: str) -> float:
    """Share of word tokens that look like Japanese romaji (mora-only words).

    ~0 for English, high for a transliterated-Japanese line. MIXED lines —
    a romaji run welded to an English half ("... kara puru Repeating it
    all") — dilute the full-line ratio below any safe threshold, so this
    also scores the first and last 8-token windows and returns the MAX:
    a line that starts or ends with a solid romaji run is still flagged.
    """
    toks = [t for t in re.findall(r"[A-Za-zāēīōū]+", text or "") if len(t) >= 2]
    if len(toks) < 3:
        return 0.0
    full = _romaji_token_stats(toks)
    if len(toks) <= 8:
        return full
    return max(full,
               _romaji_token_stats(toks[:8]),
               _romaji_token_stats(toks[-8:]))


def _is_untranslated(text: str, source_language: str = "") -> bool:
    """True when a line is still in the source language — CJK script (any source)
    OR, for a Japanese source, transliterated romaji."""
    if _cjk_ratio(text) > 0.30:
        return True
    src = (source_language or "").lower()
    if (getattr(settings, "TRANSLATION_ROMAJI_DETECT_ENABLED", True)
            and (src in _JA_SOURCE or src in ("", "auto", "unknown"))):
        # For auto/unknown sources the mora pattern alone could false-flag
        # simple English ("see you"), so an unknown source needs BOTH a
        # slightly higher ratio and no other signal — while distinctive
        # romaji function words (desu/masu/kudasai/-chan…) count as strong
        # corroboration at a lower ratio for any source.
        thr = float(getattr(settings, "TRANSLATION_ROMAJI_DETECT_THRESHOLD", 0.6))
        if src not in _JA_SOURCE:
            thr = max(thr, 0.75)
        ratio = _romaji_ja_ratio(text)
        if ratio >= thr:
            return True
        hints = len(_ROMAJI_HINTS.findall(text or ""))
        if hints >= 2 and ratio >= thr * 0.6:
            return True
    return False


def fraction_untranslated(segments, target_language: str, source_language: str = "") -> float:
    """Fraction of cues still written in the source language.

    Meaningful only when translating TO a non-CJK target (English etc.), where
    any source-language text left in the output is untranslated — and crucially
    this is judged from the OUTPUT TEXT, not the declared source language, so it
    still works when the source was detected as ``auto``. Detects CJK script for
    any source and, when ``source_language`` is Japanese, transliterated romaji
    too (a line the model spelled out phonetically instead of translating). For
    CJK targets, CJK output is correct, so it returns 0."""
    if (target_language or "").lower() in _CJK_LANGS:
        return 0.0
    rows = list(segments or [])
    if not rows:
        return 0.0
    n = sum(1 for s in rows if _is_untranslated(_seg_text(s), source_language))
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


_PREAMBLE_RE = re.compile(
    r"^(?:sure[,!.]?\s*|okay[,!.]?\s*|certainly[,!.]?\s*)?"
    r"here(?:'s| is)\s+(?:the\s+|your\s+)?(?:english\s+)?translation\s*[:\-—]*\s*",
    re.IGNORECASE,
)


def strip_llm_preamble(text: str) -> str:
    """Remove a chatty assistant preamble from a translated line.

    Shipped artifact this kills (2026-07-03 21:32 run, cue 37:23):
    ``"Sure, here is the translation:\\n\\nNothing really matters."`` — the
    model prefixed its answer instead of answering bare. Also strips plain
    ``Translation:`` / ``English:`` label prefixes.
    """
    t = (text or "").strip()
    t2 = _PREAMBLE_RE.sub("", t).strip()
    low = t2.lower()
    for pref in ("translation:", "english:", "translation -", "english -",
                 "translation —", "english —"):
        if low.startswith(pref):
            t2 = t2[len(pref):].strip()
            break
    # Never strip a line down to nothing — keep the original then.
    return t2 if t2 else t


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
    return [strip_llm_preamble(s) for s in out]


async def _translation_batch_concurrency() -> int:
    """How many LLM translation batches to run at once.

    Returns 1 (sequential — the safe default) UNLESS the PRIMARY Ollama host is a
    paired Companion advertising more than one parallel slot in its Speed profile
    (Turbo). The advertised ``num_parallel`` from ``/v1/health`` already encodes
    the profile (Turbo → several, Eco → 1), so >1 means "the Companion GPU can
    handle concurrent requests." Capped by ``TRANSLATION_PARALLEL_MAX``. A local
    card or a cloud primary stays sequential (returns 1). Fully fail-soft."""
    if not bool(getattr(settings, "TRANSLATION_PARALLEL_BATCHES", True)):
        return 1
    try:
        from backend.services import ollama_registry as _oreg
        host = _oreg.primary_host()
        if host is None or _oreg.is_local_gpu_host(host.url):
            return 1
        n = await _oreg.companion_num_parallel(host)
        cap = int(getattr(settings, "TRANSLATION_PARALLEL_MAX", 4) or 4)
        return max(1, min(cap, int(n or 1)))
    except Exception:
        return 1


# ── Deterministic per-line output sanitizers (LLM translation failsafes) ────
# Small local translators occasionally emit screenplay formatting ("MECA: ...",
# "Mechanoid: ...") for plain dialogue, and sometimes free-run past the source
# line into a paragraph of invented continuation (a hallucination-loop source
# region translated "creatively"). Both are deterministic to detect against the
# source line, so fix them here instead of hoping a later LLM pass does.

# A speaker-label token: one capitalized word (or an ALL-CAPS tag) + colon.
_SPK_LABEL = r"(?:[A-Z][\w'’.-]{1,20}|[A-Z]{2,8})"
_SPK_LEAD_RE = re.compile(rf"^\s*{_SPK_LABEL}:\s+")
_SPK_MID_RE = re.compile(rf"([.!?…]\s+){_SPK_LABEL}:\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def strip_invented_speaker_labels(text: str, source: str) -> str:
    """Remove screenplay-style ``Name:`` labels the model invented.

    Fires when the SOURCE line contains no colon (ASCII or fullwidth) — a
    colon construct in the translation of a colon-free source is model-added
    formatting, not content. A source colon only PROTECTS the label when the
    source also contains Latin script: a purely CJK source line can never
    legitimately yield a Latin ``MECA:`` prefix, so a Japanese ``：`` (or a
    Whisper-emitted ``名前:``) must not shield the invented English label —
    the observed leak kept "MECA:" through four runs because of exactly that.
    Leading labels and labels re-appearing after sentence punctuation are
    both stripped. Fail-soft: never empties a cue.
    """
    if not text:
        return text
    src = source or ""
    if (":" in src or "：" in src) and re.search(r"[A-Za-z]", src):
        return text
    out = _SPK_LEAD_RE.sub("", text)
    out = _SPK_MID_RE.sub(r"\1", out)
    out = out.strip()
    return out if out else text


def clamp_runaway_translation(text: str, source: str) -> str:
    """Cut a translation that ballooned far past its source line.

    A subtitle translation legitimately expands ~2–2.5× when going CJK→Latin;
    beyond ``TRANSLATION_MAX_EXPANSION_RATIO`` (default 4×, with a
    ``TRANSLATION_MAX_EXPANSION_CHARS`` floor of 200 so short lines are never
    touched) the tail is a model free-run, not a translation — the observed
    failure shipped a ~1400-char paragraph in a 5-second cue. Whole sentences
    are kept up to the cap; the first sentence always survives."""
    if not text:
        return text
    src_len = len((source or "").strip())
    try:
        floor = int(getattr(settings, "TRANSLATION_MAX_EXPANSION_CHARS", 200))
        ratio = float(getattr(settings, "TRANSLATION_MAX_EXPANSION_RATIO", 4.0))
    except (TypeError, ValueError):
        floor, ratio = 200, 4.0
    if ratio <= 0:      # explicit off switch
        return text
    cap = max(floor, int(src_len * ratio))
    if len(text) <= cap:
        return text
    kept: list[str] = []
    used = 0
    for sent in _SENT_SPLIT_RE.split(text.strip()):
        if kept and used + len(sent) + 1 > cap:
            break
        kept.append(sent)
        used += len(sent) + 1
    out = " ".join(kept).strip()
    if len(out) > cap:
        # The FIRST "sentence" alone blew the cap (an unbroken run-on) — hard
        # cut at the last word boundary under the cap.
        cut = out[:cap]
        sp = cut.rfind(" ")
        out = (cut[:sp] if sp > cap // 2 else cut).rstrip()
    if not out:
        out = text[:cap].rstrip()
    logger.warning(
        "Translation runaway clamped: %d chars for a %d-char source line "
        "(cap %d) — kept %d chars of whole sentences",
        len(text), src_len, cap, len(out))
    return out


async def translate_via_llm(
    segments: list,
    source_language: str,
    target_language: str,
    orchestrator: AIOrchestrator,
    glossary: dict | None = None,
    job_id: str = "",
    status_callback=None,
    model_override: str | None = None,
) -> Optional[list[TranscriptSegment]]:
    """Translate the SOURCE transcript text-to-text with the editorial LLM,
    1:1 — every segment, same timing + speaker. The reliable, COMPLETE path:
    unlike Whisper's translate task it never leaves lyrics / narration in the
    source language. Returns ``None`` when no orchestrator is available.

    ``model_override`` routes the translation calls through a dedicated model
    (e.g. OLLAMA_TRANSLATION_MODEL / OPENROUTER_TRANSLATION_MODEL) so translation
    can use a higher-quality model while editorial/SEO keep the fast default."""
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

    # Surrounding SOURCE lines shown to the model for continuity — reference
    # only, never re-translated or emitted. Kept short so local models at
    # ctx=2048 don't truncate the batch itself.
    _ctx_on = bool(getattr(settings, "TRANSLATION_LLM_CONTEXT", True))
    _ctx_before_n = max(0, int(getattr(settings, "TRANSLATION_LLM_CONTEXT_BEFORE", 2)))
    _ctx_after_n = max(0, int(getattr(settings, "TRANSLATION_LLM_CONTEXT_AFTER", 1)))

    def _context_block(before, after) -> str:
        if not _ctx_on:
            return ""
        b = [(_txt(s) or "").strip() for s in (before or [])]
        a = [(_txt(s) or "").strip() for s in (after or [])]
        b = [x for x in b if x]
        a = [x for x in a if x]
        if not b and not a:
            return ""
        parts = [
            "Surrounding dialogue for CONTINUITY (pronouns, gender, formality, "
            "tense). Reference only — do NOT translate or output these:\n"
        ]
        for x in b:
            parts.append(f"(before) {x}\n")
        for x in a:
            parts.append(f"(after) {x}\n")
        parts.append("\n")
        return "".join(parts)

    async def _call(batch, ctx_before=None, ctx_after=None) -> Optional[list[str]]:
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
            "- Write BROADCAST-quality subtitles (Netflix / professional YouTube): "
            "natural, fluent, idiomatic — never word-for-word or machine-literal.\n"
            "- NEVER repeat a phrase within a line and NEVER duplicate the previous "
            "line; no stutter or filler runs ('no no no no'). Each line is a "
            "clean, complete thought.\n"
            "- Exactly one output per input line; never merge, split, add or drop lines.\n"
            f"- Output ONLY a JSON array of exactly {len(lines)} {tgt_name} strings, in order. "
            "No commentary, no numbering.\n\n"
            f"{_auto_terms}"
            f"{_context_block(ctx_before, ctx_after)}"
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
                prompt, timeout=max(_floor, len(batch) * _per_seg), job_id=job_id,
                model_override=model_override)
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

    def _build_segs(batch, translations) -> list:
        """Apply translations + glossary onto a batch, preserving timing/speaker."""
        built = []
        for seg, tr in zip(batch, translations):
            txt = (tr or "").strip() or _txt(seg)
            # Deterministic failsafes against small-model output artifacts:
            # invented "Name:" screenplay labels and free-run continuations
            # that balloon one cue into a paragraph (both verified against
            # the source line, so real content is never touched).
            _src_line = _txt(seg)
            txt = strip_invented_speaker_labels(txt, _src_line)
            txt = clamp_runaway_translation(txt, _src_line)
            if glossary:
                for k, v in glossary.items():
                    ks, vs = (k or "").strip(), (v or "").strip()
                    if ks and vs and ks in txt:
                        txt = txt.replace(ks, vs)
            start = float(seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0) or 0.0)
            end = float(seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0) or 0.0)
            spk = (seg.get("speaker") if isinstance(seg, dict) else getattr(seg, "speaker", None)) or "Speaker 1"
            built.append(TranscriptSegment(text=txt, start=start, end=end, speaker=spk))
        return built

    async def _emit_status(done_segs: int):
        if status_callback:
            try:
                r = status_callback(f"Translating subtitles… ({min(done_segs, total)}/{total})")
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

    def _ctx_for(start, blen):
        cb = segments[max(0, start - _ctx_before_n): start] if _ctx_before_n else []
        ca = segments[start + blen: start + blen + _ctx_after_n] if _ctx_after_n else []
        return cb, ca

    # Non-first batch: direct call, else split-and-recurse (never bails).
    async def _process_one(start) -> list:
        batch = segments[start: start + BATCH]
        cb, ca = _ctx_for(start, len(batch))
        direct = await _call(batch, cb, ca)
        translations = direct if direct is not None else await _translate_batch(batch)
        return _build_segs(batch, translations)

    batch_starts = list(range(0, total, BATCH))
    # ── Validate the model on batch 0 FIRST: a first-batch outright failure means
    #    the editorial model isn't usable here — bail so the caller falls back
    #    cleanly instead of "translating" every line to itself. (Preserved from
    #    the sequential path; also gates the parallel fan-out below.) ──
    b0 = batch_starts[0]
    batch0 = segments[b0: b0 + BATCH]
    _cb0, _ca0 = _ctx_for(b0, len(batch0))
    direct0 = await _call(batch0, _cb0, _ca0)
    if direct0 is None:
        logger.info("LLM translate: editorial model returned no usable output "
                    "— deferring to other translation engines.")
        return None
    results: list = [None] * len(batch_starts)
    results[0] = _build_segs(batch0, direct0)
    _done = {"n": 1}
    await _emit_status(min(BATCH, total))

    rest = list(range(1, len(batch_starts)))
    _parallel = await _translation_batch_concurrency()
    if _parallel > 1 and len(rest) > 1:
        # Fan the remaining batches out across the Companion GPU's advertised
        # parallel slots (Turbo). Batches are independent (each prompt is built
        # only from its own lines + the once-computed glossary), so order is
        # restored by index afterward. Concurrency is bounded to num_parallel, so
        # total in-flight requests never exceed what the Companion is sized for.
        logger.info("LLM translate: parallelizing %d batches × %d Companion GPU "
                    "slots (Turbo)", len(rest), _parallel)
        _sem = asyncio.Semaphore(_parallel)

        async def _guarded(idx: int):
            async with _sem:
                segs = await _process_one(batch_starts[idx])
                results[idx] = segs
                _done["n"] += 1
                await _emit_status(_done["n"] * BATCH)

        await asyncio.gather(*[_guarded(i) for i in rest])
    else:
        for idx in rest:
            results[idx] = await _process_one(batch_starts[idx])
            _done["n"] += 1
            await _emit_status(_done["n"] * BATCH)

    for segs in results:
        out_segs.extend(segs or [])

    # ── Completeness cleanup ────────────────────────────────────────────────
    # The model occasionally echoes a hard line (long narration, song lyrics)
    # untranslated inside an otherwise-valid array — and with a vague/auto source
    # it does so more often. Re-translate any cue still in the source language
    # (CJK script OR, for a Japanese source, transliterated romaji — "Katte ippai
    # tsukuritaku naru toko" left in Latin letters), up to a couple of passes, so
    # NOTHING is left in the source language. (Runs only when the target is
    # non-CJK.)
    if (target_language or "").lower() not in _CJK_LANGS:
        for _pass in range(3):
            idxs = [i for i, s in enumerate(out_segs)
                    if _is_untranslated(s.text or "", source_language)]
            if not idxs:
                break
            logger.info("LLM translate: re-translating %d cue(s) still in the "
                        "source language (pass %d)", len(idxs), _pass + 1)
            redo = await _translate_batch([out_segs[i] for i in idxs])
            for i, tr in zip(idxs, redo):
                t = (tr or "").strip()
                # Accept the redo only if it's no longer source-language.
                if t and not _is_untranslated(t, source_language):
                    if glossary:
                        for k, v in glossary.items():
                            ks, vs = (k or "").strip(), (v or "").strip()
                            if ks and vs and ks in t:
                                t = t.replace(ks, vs)
                    cur = out_segs[i]
                    out_segs[i] = TranscriptSegment(
                        text=t, start=cur.start, end=cur.end, speaker=cur.speaker)

    logger.info("LLM translation: %d/%d segments → %s (%.0f%% still source-language)",
                len(out_segs), total, target_language,
                100 * fraction_untranslated(out_segs, target_language, source_language))

    # ── OPT-IN self-refinement pass (Task 3) ────────────────────────────────
    # A second LOCAL pass that post-edits the model's OWN output for fluency /
    # de-stutter (not a re-translate). Default off — it ~doubles an already-slow
    # CPU-offloaded 4B inference, so it's a batch-quality lever, not interactive.
    if (getattr(settings, "TRANSLATION_LLM_REFINE_PASS", False)
            and _is_ollama and out_segs):
        try:
            from backend.services.transcript_polisher import correct_transcript
            src_texts = [_txt(s) for s in segments]
            if len(src_texts) != len(out_segs):
                src_texts = None       # alignment lost → skip the source ref
            before = fraction_untranslated(out_segs, target_language, source_language)
            if status_callback:
                try:
                    r = status_callback("Refining translation for fluency (local)…")
                    if asyncio.iscoroutine(r):
                        await r
                except Exception:
                    pass
            refined = await correct_transcript(
                out_segs, orchestrator,
                language=target_language, source_texts=src_texts,
                source_language=source_language, mode="translation",
                model_override=model_override,
            )
            # Guard: never accept a refine that REINTRODUCES the source language
            # or changes the cue count (same protection as the MTPE post-edit).
            if refined and len(refined) == len(out_segs):
                after = fraction_untranslated(refined, target_language, source_language)
                if after <= before + 0.02:
                    out_segs = [
                        r if isinstance(r, TranscriptSegment)
                        else TranscriptSegment(
                            text=(getattr(r, "text", "") or _seg_text(r)),
                            start=getattr(r, "start", 0.0), end=getattr(r, "end", 0.0),
                            speaker=getattr(r, "speaker", "") or "Speaker 1")
                        for r in refined
                    ]
                    logger.info("LLM translate: self-refinement pass applied "
                                "(%d cues)", len(out_segs))
                else:
                    logger.warning("LLM translate: self-refinement raised source-"
                                   "script fraction (%.0f%%→%.0f%%) — keeping the "
                                   "pre-refine translation", 100 * before, 100 * after)
        except Exception as _ref_e:
            logger.warning("LLM translate: self-refinement pass failed (%s) — "
                           "keeping the pre-refine translation", _ref_e)

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
    from backend.services import ollama_registry
    active = await ollama_registry.pick_host()
    host = active.url if active else settings.OLLAMA_HOST
    _headers = ollama_registry.auth_headers(active)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(f"{host}/api/show", json={"model": model},
                                     headers=_headers)
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
                headers=_headers,
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


# OOM patterns Ollama / llama.cpp emit when a model doesn't fit in VRAM.
_OLLAMA_OOM_PATTERNS = (
    "out of memory", "cudamalloc", "ggml_assert", "failed to allocate cuda",
    "cuda error", "no available", "unable to allocate",
)

# Remembers the GPU layer count that last loaded successfully for a model, so
# every subsequent batch starts at the known-good rung instead of re-OOMing at
# num_gpu=99 and reloading the model each time (each reload costs ~10-15s).
_GPU_LAYERS_GOOD: dict[str, int] = {}


def _is_ollama_oom(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in _OLLAMA_OOM_PATTERNS)


def _translation_gpu_ladder(model: str, total_vram_gb: float = 0.0) -> list[int]:
    """Partial-GPU-offload ladder for the translation model, starting from the
    last rung that loaded successfully (so we don't re-OOM at 99 every batch).

    ``total_vram_gb`` (the ACTIVE card's TOTAL VRAM) lets the ladder start at a
    rung that already FITS instead of a guaranteed-OOM ``99`` for a model too big
    to fully offload. ``0`` (unknown) keeps the plain OOM-probe ladder."""
    try:
        from backend.services.local_models import (
            gpu_offload_ladder_for_vram, _ollama_names_match,
        )
        ladder = gpu_offload_ladder_for_vram(model, total_vram_gb)
    except Exception:
        return [99, 0]
    good = None
    for k, v in _GPU_LAYERS_GOOD.items():
        try:
            if _ollama_names_match(k, model):
                good = v
                break
        except Exception:
            pass
    if good is not None:
        # Start at the known-good rung; keep lower rungs (and CPU) as fallback.
        trimmed = [x for x in ladder if x <= good or x == 0]
        if good not in trimmed:
            trimmed = [good] + trimmed
        return trimmed or [good, 0]
    return ladder


async def _translate_batch_via_ollama(
    prompt: str,
    model: str,
    timeout: float = 180.0,
    num_ctx: int = 4096,
) -> str:
    """Call Ollama chat API directly with a dedicated translation model.

    ``num_ctx`` is configurable so the MTPE post-edit pass can use a larger
    context window (8192) and keep real surrounding context on long videos.

    Forces GPU placement via an explicit ``num_gpu`` and, on a small card where
    the 4B model can't fully offload, steps DOWN through a partial-offload ladder
    (most layers on GPU, a few on CPU) before falling back to full CPU — so the
    translation runs on the GPU instead of crawling on the CPU. Without an
    explicit num_gpu, Ollama's auto-scheduler frequently placed the model on the
    CPU after Whisper released VRAM.
    """
    from backend.services import ollama_registry
    _active = await ollama_registry.pick_host(required_model=model)
    if _active is None:
        # No host has the model (or none probed online) — fall back to the
        # priority-order primary and let the per-request handling decide.
        _active = next(iter(ollama_registry.enabled_hosts()), None)
    host = _active.url if _active else settings.OLLAMA_HOST
    _headers = ollama_registry.auth_headers(_active)
    base_options = {
        "num_ctx": int(num_ctx),
        "temperature": 0.3,
        "num_predict": 4096,
    }
    # Qwen3 family: apply low-temperature + presence/repetition penalties for
    # deterministic subtitle JSON (Qwen3 repeats without a penalty). Gated to the
    # Qwen3 family — non-Qwen3 models keep the default 0.3 temperature.
    try:
        from backend.services.local_models import qwen3_translation_options
        base_options.update(qwen3_translation_options(model))
    except Exception:
        pass

    # On GPU rungs cap the context (and shrink the batch) so the KV cache +
    # compute buffer fit in VRAM — the large MTPE context is only affordable on
    # the CPU rung, where the KV cache lives in system RAM.
    gpu_ctx = min(int(num_ctx),
                  int(getattr(settings, "OLLAMA_TRANSLATION_GPU_NUM_CTX", 2048)))
    gpu_batch = int(getattr(settings, "OLLAMA_TRANSLATION_GPU_NUM_BATCH", 128))

    # Proactive VRAM fit: size the starting rung to the card that will run this.
    # Remote (Companion) host → its advertised total; local host → torch total;
    # unknown → 0 (the ladder then keeps its plain OOM-probe fallback).
    _total_vram_gb = 0.0
    try:
        if _active is not None:
            if ollama_registry.is_local_gpu_host(_active.url):
                from backend.services.local_models import _total_vram_gb as _tv
                _total_vram_gb = _tv()
            elif getattr(_active, "vram_total_mb", 0):
                _total_vram_gb = _active.vram_total_mb / 1024.0
    except Exception:
        _total_vram_gb = 0.0

    # PROMPT-AWARE ctx floor: Ollama head-truncates any prompt over num_ctx —
    # losing the instruction contract first, which is how a 15-item batch comes
    # back with the wrong count. When the prompt itself needs more than the
    # flat GPU cap, raise the GPU-rung ctx toward what the prompt requires
    # (bounded by the host's card; the OOM ladder below still protects VRAM).
    _required_ctx = len(prompt) // 3 + 1024  # ~3 chars/token floor for EN/CJK
    if _required_ctx > gpu_ctx:
        _ceiling = 8192 if _total_vram_gb >= 10.0 else 4096
        _raised = min(int(num_ctx),
                      min(((_required_ctx + 1023) // 1024) * 1024, _ceiling))
        if _raised > gpu_ctx:
            logger.info(
                "translation num_ctx raised %d → %d for a %d-char prompt "
                "(host VRAM %.1f GB) — head truncation would break the "
                "JSON contract", gpu_ctx, _raised, len(prompt), _total_vram_gb)
            gpu_ctx = _raised

    ladder = _translation_gpu_ladder(model, _total_vram_gb)
    last_status_err: Optional[Exception] = None
    _host_hops = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15.0)) as client:
        i = 0
        while i < len(ladder):
            n_gpu = ladder[i]
            is_last = i == len(ladder) - 1
            options = dict(base_options)
            # num_gpu=99 → all layers on GPU (Ollama caps at the real count);
            # a smaller positive value → that many layers on GPU, rest on CPU;
            # 0 → CPU-only (the last-resort rung).
            options["num_gpu"] = int(n_gpu)
            if n_gpu > 0:
                # GPU attempt: small context + batch so it fits VRAM.
                options["num_ctx"] = gpu_ctx
                options["num_batch"] = gpu_batch
            else:
                # CPU rung: use the full configured context (RAM is plentiful).
                options["num_ctx"] = int(num_ctx)
            try:
                resp = await client.post(
                    f"{host}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": options,
                    },
                    headers=_headers,
                )
                if resp.status_code == 500 and _is_ollama_oom(resp.text) and not is_last:
                    logger.warning(
                        "Ollama translation OOM at num_gpu=%s — stepping down to %s "
                        "(partial GPU offload)", n_gpu, ladder[i + 1])
                    i += 1
                    continue
                if resp.status_code == 503 and _host_hops == 0:
                    # A saturated Companion proxy asks us to wait — honor
                    # Retry-After once, then the failover path takes over.
                    try:
                        _wait = min(15.0, max(0.5, float(
                            resp.headers.get("Retry-After", "2"))))
                    except (TypeError, ValueError):
                        _wait = 2.0
                    logger.info("Ollama translation host busy (503) — retrying in %.1fs", _wait)
                    await asyncio.sleep(_wait)
                    _host_hops += 1
                    continue
                resp.raise_for_status()
                data = resp.json()
                # Remember the rung that worked so later batches skip the OOM dance.
                _GPU_LAYERS_GOOD[model] = int(n_gpu)
                if n_gpu == 0:
                    logger.info("Ollama translation ran on CPU (model=%s) — GPU "
                                "offload exhausted; expect slower throughput", model)
                else:
                    logger.debug("Ollama translation ran with num_gpu=%s (model=%s)",
                                 n_gpu, model)
                return data.get("message", {}).get("content", "")
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                # Host down mid-translation — mark it unhealthy and restart
                # the ladder on the next registry host.
                if _active is not None:
                    ollama_registry.mark_unhealthy(
                        _active, f"{type(e).__name__}: {str(e)[:100]}")
                _next = await ollama_registry.pick_host()
                if _host_hops < 4 and _next is not None and _next.url != host:
                    logger.warning(
                        "Ollama translation failover: %s unreachable — switching "
                        "to '%s' (%s)", host, _next.name, _next.url)
                    _active = _next
                    host = _next.url
                    _headers = ollama_registry.auth_headers(_active)
                    _host_hops += 1
                    i = 0
                    continue
                raise
            except httpx.HTTPStatusError as e:
                last_status_err = e
                body = ""
                try:
                    body = e.response.text if e.response is not None else ""
                except Exception:
                    body = ""
                if _is_ollama_oom(body) and not is_last:
                    logger.warning(
                        "Ollama translation OOM at num_gpu=%s — stepping down to %s",
                        n_gpu, ladder[i + 1])
                    i += 1
                    continue
                raise
    if last_status_err is not None:
        raise last_status_err
    raise RuntimeError("Ollama translation produced no response")


# Preference order for auto-selecting the translation-POLISH model on a paired
# GPU host — larger / more-multilingual first. Matched loosely (name prefix), so
# any installed quant/tag of these families qualifies. The first entry that is
# BOTH installed on the Companion AND fits its VRAM wins.
_POLISH_MODEL_PREFERENCE = [
    "qwen2.5:14b-instruct", "qwen2.5:14b",
    "qwen2.5:7b-instruct", "qwen2.5:7b",
    "qwen3:8b", "gemma2:9b-instruct", "llama3.1:8b-instruct",
    "qwen2.5:3b-instruct",
]


async def _companion_vram_budget_gb() -> float:
    """The Companion's live Ollama VRAM budget (GB) from its /v1/health, or
    0.0 when unknown. Cached ~60 s — the budget only moves when the user
    changes speed settings or the Whisper sidecar starts/stops."""
    import time as _t
    global _COMPANION_BUDGET_CACHE
    now = _t.monotonic()
    cached_at, cached_val = _COMPANION_BUDGET_CACHE
    if now - cached_at < 60.0:
        return cached_val
    val = 0.0
    try:
        from backend.services.reframer_audio import (
            _remote_whisper_base, _remote_whisper_token,
        )
        base = _remote_whisper_base()
        if base:
            headers = {}
            tok = _remote_whisper_token()
            if tok:
                headers["Authorization"] = f"Bearer {tok}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{base}/v1/health", headers=headers)
                if r.status_code == 200:
                    val = float((r.json() or {}).get("vram_budget_gb", 0) or 0)
    except Exception:
        val = 0.0
    _COMPANION_BUDGET_CACHE = (now, val)
    return val


_COMPANION_BUDGET_CACHE: tuple[float, float] = (-1e9, 0.0)


async def resolve_translation_polish_model(fallback: str) -> str:
    """Pick the model for the translation-POLISH (MTPE) pass.

    An explicit ``OLLAMA_TRANSLATION_POLISH_MODEL`` wins. Otherwise, when a
    paired Companion (remote GPU host) is available, auto-pick the largest
    suitable instruct model already installed on it — the polish reads far more
    naturally from a 14B than the light 4B translation model, and the Companion's
    big card runs it at GPU speed (the host registry routes the call there via
    ``pick_host(required_model=...)``). Falls back to ``fallback`` (the
    translation model) when auto is off, there's no Companion, nothing suitable
    is installed, or on any error — so a small-card-only deployment is unchanged.
    """
    pinned = (getattr(settings, "OLLAMA_TRANSLATION_POLISH_MODEL", "") or "").strip()
    if pinned:
        return pinned
    if not bool(getattr(settings, "OLLAMA_TRANSLATION_POLISH_AUTO", True)):
        return fallback
    try:
        from backend.services import ollama_registry as _oreg
        # Only auto-upgrade when the PRIMARY Ollama host is a remote GPU (the
        # paired Companion). Both routers — the orchestrator (uses primary_url())
        # and the offline MTPE path (pick_host prefers primary) — then land the
        # bigger model on that card. If the primary is the weak local card, an
        # upsize would spill to CPU and be slower, not better — so keep the light
        # model there.
        host = _oreg.primary_host()
        if host is None or _oreg.is_local_gpu_host(host.url):
            return fallback
        status = await _oreg.probe(host)
        if not status.online or not status.models:
            return fallback
        vram_gb = ((getattr(host, "vram_total_mb", 0) or 0) / 1024.0)
        # The registry rarely knows the card size (probe is /api/tags only),
        # but the Companion's /v1/health reports its LIVE Ollama VRAM budget
        # (card minus the resident Whisper sidecar + overhead). The observed
        # failure: 12 GB card, but budget was 7 GB — the auto-picked 14b
        # partial-offloaded at 90 s/batch and starved the pass. Prefer the
        # live budget; keep total-VRAM as fallback.
        try:
            budget_gb = await _companion_vram_budget_gb()
            if budget_gb and budget_gb > 0:
                vram_gb = budget_gb
        except Exception:
            pass
        from backend.services.local_models import (
            _ollama_names_match, estimate_model_weights_gb,
        )
        for pref in _POLISH_MODEL_PREFERENCE:
            match = next((inst for inst in status.models
                          if _ollama_names_match(inst, pref)), None)
            if not match:
                continue
            # Respect the host's VRAM when it's known: skip a model that won't
            # fit fully (it would spill to CPU and be slower, not better).
            # +1.5 GB covers KV cache (num_parallel × num_ctx) + runtime.
            w = estimate_model_weights_gb(match)
            if vram_gb > 0:
                if w is not None and (w + 1.5) > vram_gb:
                    logger.info(
                        "Translation polish: skipping %s — %.1f GB weights "
                        "won't fit the Companion's %.1f GB Ollama budget "
                        "(would partial-offload and run slower, not better)",
                        match, w or 0.0, vram_gb)
                    continue
            elif w is not None and w > float(getattr(
                    settings, "TRANSLATION_POLISH_UNKNOWN_VRAM_MAX_GB", 6.0)):
                # Budget unknown: be conservative — an oversized pick costs
                # minutes (partial offload), an undersized one costs nothing.
                continue
            if _ollama_names_match(match, fallback):
                return fallback  # best available IS the light model — no change
            logger.info(
                "Translation polish: auto-selected %s on GPU host '%s' for higher "
                "quality (fallback was %s)", match, host.name, fallback)
            return match
    except Exception as e:
        logger.debug("translation-polish model auto-select skipped (%s)", e)
    return fallback


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
                    if _is_untranslated(_seg_text(s), source_language)]
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

    # Route the polish to the paired Companion GPU with a larger model when one
    # is available (higher translation quality). When it auto-upgrades, the model
    # is already confirmed installed on that host, so skip the local availability
    # pre-check below and let the host registry route the call there.
    _auto_polish = await resolve_translation_polish_model(model)
    _polish_upgraded = (_auto_polish or "").strip() != (model or "").strip()
    model = _auto_polish

    # ── Model-availability pre-check + one-line diagnostics ──
    # Confirm the configured translation model is actually pulled on the Ollama
    # host. If it's missing, log an actionable `ollama pull` line and fall back to
    # the raw NMT draft (the completeness backstop) instead of erroring inside the
    # request path. We never auto-pull silently here. A transient list failure is
    # non-fatal — we proceed and let the per-batch fail-soft handle any error.
    # Skipped when the polish model was auto-upgraded (already verified resident
    # on the Companion, which may not be the primary host this check inspects).
    if not _polish_upgraded:
        try:
            from backend.services.local_models import list_ollama_models, _ollama_names_match
            installed = await list_ollama_models()
            if installed and not any(_ollama_names_match(m, model) for m in installed):
                logger.warning(
                    "Local translation model %r is not installed on the Ollama host "
                    "(%s) — falling back to the offline NMT draft. To enable the "
                    "Qwen3 MTPE pass, run:  ollama pull %s",
                    model, host, model)
                return nmt_segments
        except Exception as _avail_e:
            logger.debug("Translation model availability check skipped (%s)", _avail_e)
    logger.info(
        "Local translation engine: model=%s num_ctx=%d host=%s device=ollama-auto "
        "(GPU after Whisper release, CPU fallback on OOM)",
        model, num_ctx, host)
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
        # Batch size trades memory for CT2 worker parallelism — the engines
        # now decode a whole batch in one call across inter_threads workers,
        # so bigger batches directly cut wall time on multi-core hosts.
        BATCH = max(4, int(getattr(settings, "NMT_BATCH_SIZE", 32)))
        for start in range(0, len(segments), BATCH):
            batch = segments[start: start + BATCH]
            ctx_before = [
                s.text for s in segments[max(0, start - context_window): start]
            ]
            ctx_after = [
                s.text for s in segments[start + len(batch): start + len(batch) + context_window]
            ]
            texts = [s.text for s in batch]
            if hasattr(engine, "translate_with_context"):
                # NLLB *and* Opus-MT/FuguMT: surrounding source cues resolve
                # dropped subjects/pronouns (Japanese omits them constantly —
                # isolated-cue decoding was the largest coherence gap on the
                # default ja→en path). Each engine guards recovery with
                # numbered tags and falls back per-cue to the isolated
                # translation, so output is never worse than context-free.
                # to_thread: the CT2 decode is seconds of pure CPU per batch —
                # it must not block the event loop (heartbeats, websockets).
                translations = await asyncio.to_thread(
                    engine.translate_with_context,
                    texts, ctx_before, ctx_after,
                    source_language, target_language, glossary=glossary,
                )
            else:
                translations = await asyncio.to_thread(
                    engine.translate_batch, texts, glossary=glossary)
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
    # Normalize to ISO 639-1 first: whisper.cpp reports full language names
    # ("japanese"), and every downstream consumer — the engine router, HF
    # model-id templates, the Flores map, the CJK checks — keys on ISO codes.
    # An unnormalized name previously built the invalid repo id
    # "staka/fugumt-japanese-en" and hard-failed the whole translation.
    from backend.services.language_codes import normalize_lang_code
    source_language = normalize_lang_code(source_language)
    target_language = normalize_lang_code(target_language)
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

