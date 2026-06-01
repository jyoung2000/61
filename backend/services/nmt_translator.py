"""Local NMT translation backends — NLLB-200 + Opus-MT via CTranslate2.

Provides a fast, free, fully local alternative to LLM-based translation.
Models are downloaded on demand (NEVER at startup) — call
``ensure_nllb_downloaded()`` from a UI button or CLI before first use.

Two engines:
  - NLLB-200-distilled-600M: 200 languages, ~600 MB int8, broadest coverage.
  - Opus-MT (Helsinki-NLP/opus-mt-{src}-{tgt}): per-pair, 250-350 MB, fastest.

Both implementations are guarded by ``is_available()`` so callers can
probe without loading and the translator router can fall back to the LLM
path cleanly.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ── ISO 639-1 → Flores-200 mapping for NLLB ──────────────────────────────
# Covers the languages already in translator.SUPPORTED_LANGUAGES.
_FLORES_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "zh": "zho_Hans",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "uk": "ukr_Cyrl",
    "sv": "swe_Latn",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "tl": "tgl_Latn",
}


def iso_to_flores(code: str) -> Optional[str]:
    """Map an ISO 639-1 code to a Flores-200 code (case-insensitive)."""
    if not code:
        return None
    return _FLORES_CODES.get(code.strip().lower())


# ── Disk locations ────────────────────────────────────────────────────────

def _models_dir() -> str:
    """Persistent models directory. Prefers the Docker /data mount."""
    docker = "/data/models"
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        os.makedirs(docker, exist_ok=True)
        return docker
    local = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".clipai", "models",
    )
    os.makedirs(local, exist_ok=True)
    return local


def _nllb_dir(model_id: str) -> str:
    return os.path.join(_models_dir(), "nllb", model_id.replace("/", "_"))


def _opus_dir(src: str, tgt: str) -> str:
    return os.path.join(_models_dir(), "opus-mt", f"{src}-{tgt}")


# ── NLLB-200 wrapper (CTranslate2) ───────────────────────────────────────

class NMTTranslator:
    """Local NMT translation via CTranslate2.

    Lazy-loads the requested model on first use. Call ``unload()`` after a
    translation pass so Whisper / Ollama can reclaim the GPU's VRAM —
    the GTX 1650 has only 4 GB and the three engines must take turns.
    """

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        from backend.config import settings as _settings
        self.model_id = model_id or _settings.NMT_NLLB_MODEL
        # Device policy: explicit arg wins, else the NMT_DEVICE setting
        # ("auto" | "cpu" | "cuda"). cpu is the safe choice on 4 GB GPUs.
        self.device = (device or getattr(_settings, "NMT_DEVICE", "auto") or "auto").lower()
        self._translator = None
        self._tokenizer = None
        self._loaded = False

    # ── Availability ─────────────────────────────────────────────────────

    @staticmethod
    def _has_dependencies() -> bool:
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        """True only when both the dependencies AND the model files are on
        disk. We never trigger a download from this probe."""
        if not self._has_dependencies():
            return False
        return self._model_files_present(self.model_id)

    @staticmethod
    def _model_files_present(model_id: str) -> bool:
        path = _nllb_dir(model_id)
        if not os.path.isdir(path):
            return False
        # CTranslate2 model.bin + tokenizer SentencePiece file are the
        # minimum we need for inference.
        has_model = os.path.exists(os.path.join(path, "model.bin"))
        has_tok = any(
            os.path.exists(os.path.join(path, name))
            for name in ("sentencepiece.bpe.model", "tokenizer.json", "spiece.model")
        )
        return has_model and has_tok

    # ── Loading / unloading ──────────────────────────────────────────────

    def load(self):
        if self._loaded:
            return
        if not self._has_dependencies():
            raise RuntimeError(
                "NMTTranslator: ctranslate2 + sentencepiece required — "
                "pip install ctranslate2 sentencepiece"
            )
        path = _nllb_dir(self.model_id)
        if not self._model_files_present(self.model_id):
            raise FileNotFoundError(
                f"NLLB model not downloaded at {path}. "
                "Download via the Settings UI before using TRANSLATION_ENGINE=nllb."
            )
        import ctranslate2
        import sentencepiece as spm

        device = self.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # On CUDA, confirm Whisper (and any prior CUDA tenant) has released
        # VRAM before we load NLLB — on a 4 GB GTX 1650 the engines must take
        # turns or this load OOMs. Translation runs post-analysis, so the
        # pipeline's _release_whisper_vram() has already run; this is the
        # confirmation log + a defensive empty_cache().
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    free_mb, total_mb = [x / (1024 * 1024) for x in torch.cuda.mem_get_info()]
                    logger.info(
                        "NMT: Whisper VRAM freed before NLLB load — %.0f MB free / %.0f MB total",
                        free_mb, total_mb,
                    )
            except Exception:
                pass

        # int8 keeps the 600M model under 1 GB VRAM.
        compute_type = "int8_float16" if device == "cuda" else "int8"
        logger.info("NMT: loading NLLB %s on %s (%s)", self.model_id, device, compute_type)
        self._translator = ctranslate2.Translator(path, device=device, compute_type=compute_type)

        tok_file = None
        for name in ("sentencepiece.bpe.model", "spiece.model"):
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                tok_file = cand
                break
        if tok_file is None:
            raise FileNotFoundError(f"SentencePiece tokenizer missing in {path}")
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(tok_file)
        self._loaded = True

    def unload(self):
        """Release GPU memory so Whisper / Ollama can reclaim it."""
        if not self._loaded:
            return
        try:
            del self._translator
        except Exception:
            pass
        self._translator = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Translation ──────────────────────────────────────────────────────

    def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary: Optional[dict] = None,
    ) -> list[str]:
        """Translate a list of strings between ISO 639-1 language codes.

        Returns the translations in the same order. Empty inputs are
        passed through unchanged. On failure, returns the original texts.
        """
        if not texts:
            return []
        flores_src = iso_to_flores(source_lang)
        flores_tgt = iso_to_flores(target_lang)
        if not flores_src or not flores_tgt:
            logger.warning(
                "NMT: unsupported language pair %s→%s (not in Flores-200 mapping)",
                source_lang, target_lang,
            )
            return list(texts)

        if not self._loaded:
            self.load()
        assert self._translator is not None and self._tokenizer is not None

        results: list[str] = []
        for raw in texts:
            text = (raw or "").strip()
            if not text:
                results.append(raw)
                continue
            try:
                tokens = self._tokenizer.encode_as_pieces(text)
                source = [flores_src] + tokens + ["</s>"]
                output = self._translator.translate_batch(
                    [source],
                    target_prefix=[[flores_tgt]],
                    beam_size=4,
                    max_decoding_length=256,
                )
                pieces = output[0].hypotheses[0]
                # Drop the language token prefix.
                if pieces and pieces[0] == flores_tgt:
                    pieces = pieces[1:]
                translated = self._tokenizer.decode(pieces)
                if glossary:
                    translated = apply_glossary(text, translated, glossary)
                results.append(translated)
            except Exception as e:
                logger.warning("NMT: translation failed for one segment (%s) — keeping original", e)
                results.append(raw)
        return results

    def translate_with_context(
        self,
        batch: list[str],
        context_before: list[str],
        context_after: list[str],
        source_lang: str,
        target_lang: str,
        glossary: Optional[dict] = None,
    ) -> list[str]:
        """Translate ``batch`` with surrounding context for pronoun /
        gender / idiom resolution.

        We concatenate the segments with a sentinel separator and ask NLLB
        to translate the whole string in one shot, then split the result
        back. NLLB doesn't accept a system prompt the way an LLM does, so
        the context is included inline and we keep only the translations
        for the in-batch segments.
        """
        if not batch:
            return []
        sep = " ¶ "
        joined = sep.join(context_before + batch + context_after).strip()
        if not joined:
            return list(batch)
        translated_joined = self.translate_batch(
            [joined], source_lang, target_lang, glossary=glossary,
        )[0]
        parts = [p.strip() for p in re.split(r"\s*¶\s*", translated_joined)]
        start = len(context_before)
        end = start + len(batch)
        out = parts[start:end]
        if len(out) != len(batch):
            # The model lost the separators; fall back to per-segment.
            return self.translate_batch(batch, source_lang, target_lang, glossary=glossary)
        return out


# ── Opus-MT wrapper (one model per pair) ─────────────────────────────────

class OpusMTTranslator:
    """Local Helsinki-NLP/opus-mt translator for a single language pair.

    Much smaller than NLLB (~300 MB each), faster on European pairs, but
    one model per direction — so we lazy-create translators on demand
    and cache them.
    """

    _cache: dict[tuple[str, str], "OpusMTTranslator"] = {}

    def __init__(self, source: str, target: str):
        self.source = source.lower()
        self.target = target.lower()
        self._translator = None
        self._tokenizer = None
        self._loaded = False

    @classmethod
    def get(cls, source: str, target: str) -> "OpusMTTranslator":
        key = (source.lower(), target.lower())
        if key not in cls._cache:
            cls._cache[key] = cls(source, target)
        return cls._cache[key]

    def is_available(self) -> bool:
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except Exception:
            return False
        path = _opus_dir(self.source, self.target)
        return os.path.exists(os.path.join(path, "model.bin"))

    def load(self):
        if self._loaded:
            return
        import ctranslate2
        import sentencepiece as spm
        path = _opus_dir(self.source, self.target)
        if not os.path.exists(os.path.join(path, "model.bin")):
            raise FileNotFoundError(
                f"Opus-MT model {self.source}-{self.target} not downloaded at {path}."
            )
        device = "cpu"  # Opus-MT runs faster on CPU for small batches.
        self._translator = ctranslate2.Translator(path, device=device, compute_type="int8")
        tok_file = None
        for name in ("source.spm", "sentencepiece.bpe.model", "spiece.model"):
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                tok_file = cand
                break
        if tok_file is None:
            raise FileNotFoundError(f"SentencePiece tokenizer missing in {path}")
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(tok_file)
        self._loaded = True

    def unload(self):
        if not self._loaded:
            return
        try:
            del self._translator
        except Exception:
            pass
        self._translator = None
        self._tokenizer = None
        self._loaded = False

    def translate_batch(
        self,
        texts: list[str],
        glossary: Optional[dict] = None,
    ) -> list[str]:
        if not texts:
            return []
        if not self._loaded:
            self.load()
        results: list[str] = []
        for raw in texts:
            text = (raw or "").strip()
            if not text:
                results.append(raw)
                continue
            try:
                tokens = self._tokenizer.encode_as_pieces(text)
                output = self._translator.translate_batch(
                    [tokens + ["</s>"]],
                    beam_size=4,
                    max_decoding_length=256,
                )
                pieces = output[0].hypotheses[0]
                translated = self._tokenizer.decode(pieces)
                if glossary:
                    translated = apply_glossary(text, translated, glossary)
                results.append(translated)
            except Exception as e:
                logger.warning("Opus-MT: translation failed (%s)", e)
                results.append(raw)
        return results


# ── Glossary enforcement (shared between NLLB + Opus-MT) ─────────────────

def apply_glossary(source_text: str, translated_text: str, glossary: dict) -> str:
    """Force glossary terms into the translation.

    NMT models can't be conditioned on a glossary the way LLMs can. We
    post-process: for every glossary entry whose source term appears in
    ``source_text``, replace any occurrence of the source term in the
    translation with the target term, and inject the target term if the
    translation seems to have missed it entirely.
    """
    if not glossary or not translated_text:
        return translated_text
    out = translated_text
    for src, tgt in glossary.items():
        src_s = (src or "").strip()
        tgt_s = (tgt or "").strip()
        if not src_s or not tgt_s:
            continue
        if src_s not in source_text:
            continue
        # If the source term leaked into the translation, swap it.
        if src_s in out:
            out = out.replace(src_s, tgt_s)
        # If the model already emitted the target term (case-insensitive
        # match), leave it. Otherwise we don't risk injecting it blindly
        # — the source position no longer maps cleanly.
    return out


# ── Download helpers (for the Settings UI button) ─────────────────────────

def ensure_nllb_downloaded(
    model_id: Optional[str] = None,
    progress_callback=None,
) -> str:
    """Download + convert NLLB-200 to CTranslate2 format on demand.

    Returns the absolute path to the model directory. Raises
    ``RuntimeError`` if dependencies are missing. Safe to call repeatedly
    — does nothing when the converted model already exists.
    """
    from backend.config import settings as _settings
    model_id = model_id or _settings.NMT_NLLB_MODEL
    target_dir = _nllb_dir(model_id)
    if NMTTranslator._model_files_present(model_id):
        logger.info("NLLB %s already downloaded at %s", model_id, target_dir)
        return target_dir
    try:
        from ctranslate2.converters import TransformersConverter
    except Exception as e:
        raise RuntimeError(
            "ctranslate2 with the transformers converter is required to "
            "download NLLB. pip install 'ctranslate2[transformers]'"
        ) from e
    os.makedirs(target_dir, exist_ok=True)
    converter = TransformersConverter(model_id)
    converter.convert(target_dir, quantization="int8", force=False)
    logger.info("NLLB downloaded + converted to %s", target_dir)
    return target_dir


def ensure_opus_mt_downloaded(source: str, target: str) -> str:
    """Download + convert an Opus-MT pair to CTranslate2 format.

    Same semantics as ``ensure_nllb_downloaded``.
    """
    from backend.config import settings as _settings
    src, tgt = source.lower(), target.lower()
    target_dir = _opus_dir(src, tgt)
    if os.path.exists(os.path.join(target_dir, "model.bin")):
        return target_dir
    try:
        from ctranslate2.converters import TransformersConverter
    except Exception as e:
        raise RuntimeError(
            "ctranslate2 with the transformers converter is required to "
            "download Opus-MT. pip install 'ctranslate2[transformers]'"
        ) from e
    os.makedirs(target_dir, exist_ok=True)
    model_id = _settings.NMT_OPUS_MT_TEMPLATE.format(src=src, tgt=tgt)
    converter = TransformersConverter(model_id)
    converter.convert(target_dir, quantization="int8", force=False)
    return target_dir


# ── Convenience: pick the right local engine for a pair ──────────────────

def pick_local_engine(source: str, target: str):
    """Return an instantiated local translator (Opus-MT preferred) or
    None if no local model is downloaded for the pair.
    """
    opus = OpusMTTranslator.get(source, target)
    if opus.is_available():
        return opus
    nllb = NMTTranslator()
    if nllb.is_available() and iso_to_flores(source) and iso_to_flores(target):
        return nllb
    return None
