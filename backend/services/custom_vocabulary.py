"""Custom vocabulary (Whisper biasing) — Otter.ai-style accuracy lever.

ClipAI historically passed no biasing prompt to Whisper, so jargon, names,
acronyms, and brand/product terms mis-transcribed and the downstream
polisher had to *guess* corrections. This module lets the user supply a
glossary of terms that bias the ASR at the source.

Two biasing strategies are exposed:

  * ``hotwords_string`` — a space-joined term list for faster-whisper's
    ``hotwords=`` kwarg (recent faster-whisper). This biases the decoder
    without the hallucination risk of a free-text prompt and is preferred
    when the installed engine supports it.
  * ``build_initial_prompt`` — a natural-language ``initial_prompt`` for
    older faster-whisper builds that lack ``hotwords``. Kept short (under
    ~200 tokens) because Whisper echoes / hallucinates long prompts — the
    diagnostics module detects this prompt-echo behaviour. For CJK
    languages the Latin "Glossary:" framing is dropped (it would echo as
    literal output), and the terms are emitted with CJK delimiters.

Persistence: the glossary lives at ``/data/logs/custom_vocabulary.json``
on the same Docker volume mount that backs ``user_settings.json``, so it
survives ``docker compose down`` + rebuild without any extra restore step.
The ``CUSTOM_VOCABULARY_ENABLED`` toggle rides ``user_settings.json`` via
``_PERSISTABLE_KEYS``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger("clipai.custom_vocabulary")

# ── Limits (Otter caps at ~200 on Pro; we allow a little more) ──
MAX_TERMS = 300
MAX_TERM_LENGTH = 80
# Whisper ignores / echo-hallucinates long initial prompts. Keep the
# assembled prompt comfortably under this estimated-token ceiling.
PROMPT_TOKEN_CEILING = 200

# ISO-639-1 prefixes for CJK languages, which need delimiter-only framing.
_CJK_LANGS = {"ja", "zh", "ko", "yue"}


def _resolve_data_dir() -> str:
    """Find a writable data directory for the glossary file.

    Prefers ``/data/logs`` (the Docker volume mount that also backs
    ``user_settings.json``); falls back to a project-local ``.clipai``
    directory when running outside the container.
    """
    docker_path = "/data/logs"
    if os.path.isdir(docker_path) and os.access(docker_path, os.W_OK):
        return docker_path
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".clipai",
    )
    os.makedirs(local_path, exist_ok=True)
    return local_path


def _vocabulary_path() -> str:
    return os.path.join(_resolve_data_dir(), "custom_vocabulary.json")


def clean_terms(terms: list[str]) -> list[str]:
    """Strip, dedupe (case-insensitive, order-preserving), drop terms
    longer than ``MAX_TERM_LENGTH`` chars, and cap at ``MAX_TERMS``."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in terms or []:
        term = str(raw or "").strip()
        if not term:
            continue
        if len(term) > MAX_TERM_LENGTH:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= MAX_TERMS:
            break
    return out


def load_vocabulary() -> list[str]:
    """Load the persisted glossary terms. Returns an empty list when the
    file is missing or unreadable (so the pipeline degrades to current
    no-prompt behaviour)."""
    path = _vocabulary_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        terms = data.get("terms", []) if isinstance(data, dict) else data
        return clean_terms(list(terms or []))
    except Exception as e:
        logger.warning("Failed to load custom vocabulary from %s: %s", path, e)
        return []


def save_vocabulary(terms: list[str]) -> list[str]:
    """Persist the (cleaned) glossary to the mount-backed JSON file.

    Returns the cleaned term list that was actually written so callers
    can echo the validated state back to the UI.
    """
    cleaned = clean_terms(terms)
    path = _vocabulary_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"terms": cleaned}, f, ensure_ascii=False, indent=2)
        logger.info("Persisted %d custom vocabulary terms to %s", len(cleaned), path)
    except Exception as e:
        logger.warning("Failed to save custom vocabulary to %s: %s", path, e)
    return cleaned


def _estimate_tokens(text: str) -> int:
    """Rough upper-bound token estimate for an English/CJK mixed string.

    Whisper's tokenizer is BPE; ~4 chars/token is a decent average, but
    we take the max of the char-based and whitespace-based estimates so
    we never *under*-count and blow past the ceiling.
    """
    if not text:
        return 0
    char_estimate = (len(text) + 3) // 4
    word_estimate = len(text.split())
    return max(char_estimate, word_estimate)


def _is_cjk_language(language: Optional[str]) -> bool:
    if not language:
        return False
    return language.strip().lower().split("-")[0] in _CJK_LANGS


def build_initial_prompt(terms: list[str], language: Optional[str] = None) -> str:
    """Assemble a Whisper ``initial_prompt`` biasing string from ``terms``.

    Non-CJK: a natural-language sentence, e.g.
        ``"Glossary: Heero Yuy, Zechs Merquise, Gundam, OZ, mobile suit."``
    CJK (ja/zh/ko/yue): the Latin "Glossary:" framing is dropped (Whisper
    would echo it as literal output); terms are joined with the CJK
    enumeration comma "、".

    Terms are dropped from the tail until the assembled prompt is under
    ``PROMPT_TOKEN_CEILING`` estimated tokens.
    """
    cleaned = clean_terms(terms)
    if not cleaned:
        return ""

    is_cjk = _is_cjk_language(language)

    def _assemble(items: list[str]) -> str:
        if is_cjk:
            return "、".join(items)
        return "Glossary: " + ", ".join(items) + "."

    # Trim from the tail until under the token ceiling.
    items = list(cleaned)
    prompt = _assemble(items)
    while items and _estimate_tokens(prompt) > PROMPT_TOKEN_CEILING:
        items.pop()
        prompt = _assemble(items)
    return prompt if items else ""


def hotwords_string(terms: list[str]) -> str:
    """Space-joined term list for faster-whisper's ``hotwords=`` kwarg.

    Unlike ``initial_prompt``, hotwords bias the decoder without being
    decoded into output, so there is no prompt-echo / hallucination risk.
    Still trimmed to ``MAX_TERMS`` via ``clean_terms``.
    """
    return " ".join(clean_terms(terms))


def _callable_supports(transcribe_callable, kwarg: str) -> bool:
    """Feature-detect whether ``transcribe_callable`` accepts ``kwarg``."""
    import inspect
    try:
        params = inspect.signature(transcribe_callable).parameters
    except (TypeError, ValueError):
        return False
    # ``**kwargs`` catch-alls don't count as explicit support.
    return kwarg in params


def whisper_bias_kwargs(
    transcribe_callable,
    language: Optional[str] = None,
    enabled: bool = True,
    terms: Optional[list[str]] = None,
) -> dict:
    """Build the biasing kwargs to splat into a Whisper ``transcribe()`` call.

    Returns ``{"hotwords": "..."}`` when ``transcribe_callable`` accepts a
    ``hotwords`` kwarg (preferred — biases without hallucination risk),
    otherwise ``{"initial_prompt": "..."}`` as a fallback. Returns an empty
    dict — preserving the current no-prompt behaviour exactly — when the
    feature is disabled or the glossary is empty.

    ``terms`` is loaded from the persisted glossary when not supplied.
    """
    try:
        if not enabled:
            return {}
        vocab = load_vocabulary() if terms is None else clean_terms(terms)
        if not vocab:
            return {}
        lang = None if language in ("auto", "", None) else language
        if _callable_supports(transcribe_callable, "hotwords"):
            hw = hotwords_string(vocab)
            return {"hotwords": hw} if hw else {}
        prompt = build_initial_prompt(vocab, lang)
        return {"initial_prompt": prompt} if prompt else {}
    except Exception as e:
        logger.warning("Custom vocabulary biasing skipped (%s)", e)
        return {}
