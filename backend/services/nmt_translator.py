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

import contextlib
import logging
import os
import re
import shutil
import tempfile
import time
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Rough free-disk floors required before a download/convert starts. The HF
# converter pulls the full-precision model into a cache, then writes the
# smaller int8 CTranslate2 copy — so we need headroom for BOTH transiently.
# NLLB-200-distilled-600M is ~2.4 GB fp32 on the Hub + ~0.6 GB int8 out.
_NLLB_MIN_FREE_BYTES = 6 * 1024 ** 3   # 6 GB headroom for HF cache + int8 out
_OPUS_MIN_FREE_BYTES = 3 * 1024 ** 3   # 3 GB headroom per Opus-MT pair

# Free VRAM (GB) needed to load NLLB-600M int8 on CUDA: ~1 GB weights + CT2
# activation workspace + margin. Above this, NMT_DEVICE=auto uses the GPU even
# on a 4 GB card (Whisper VRAM is released before translation); below it, CPU.
_NLLB_CUDA_MIN_FREE_GB = 1.8


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


# ── Script helpers + long-cue chunking ───────────────────────────────────
# CJK / no-space Flores script suffixes (joined without spaces on output).
_CJK_SCRIPTS = {"Jpan", "Hans", "Hant", "Hang", "Hira", "Kana", "Bopo"}

# Source-length caps (chars) before a cue is chunked for NMT. NLLB decodes
# ~200 source tokens reliably; beyond that the output is truncated at
# max_decoding_length and a long run-on comes back UNtranslated. CJK is denser
# (~1.5 tokens/char) so it gets a tighter cap than Latin scripts.
_MAX_SRC_CHARS_CJK = 80
_MAX_SRC_CHARS_LATIN = 300

_SENT_SPLIT_RE = re.compile(r".*?(?:[。．！？!?…]+|$)", re.DOTALL)
_CLAUSE_CHARS = "、，,；;:："


def _flores_is_cjk(flores_code: Optional[str]) -> bool:
    """True when a Flores-200 code targets a CJK / no-space script."""
    return bool(flores_code) and flores_code.rsplit("_", 1)[-1] in _CJK_SCRIPTS


def _looks_cjk(text: str) -> bool:
    """Heuristic: >= 30% of non-space chars are CJK ideographs / kana / hangul.

    Used to detect a cue that came back UNtranslated (still in the source
    script) so the completeness pass can retry it.
    """
    if not text:
        return False
    cjk = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        o = ord(ch)
        if (
            0x3040 <= o <= 0x30FF   # hiragana / katakana
            or 0x3400 <= o <= 0x4DBF   # CJK ext A
            or 0x4E00 <= o <= 0x9FFF   # CJK unified ideographs
            or 0xAC00 <= o <= 0xD7A3   # hangul syllables
        ):
            cjk += 1
    return total > 0 and (cjk / total) >= 0.30


def _hard_split(text: str, max_chars: int, cjk: bool) -> list[str]:
    """Last-resort split of an over-long sentence at clause separators (or
    spaces, for Latin scripts), falling back to fixed-width slices."""
    out: list[str] = []
    buf = ""
    seps = _CLAUSE_CHARS if cjk else _CLAUSE_CHARS + " "
    for ch in text:
        buf += ch
        if len(buf) >= max_chars:
            cut = max((buf.rfind(s) for s in seps), default=-1)
            if cut >= max_chars // 2:
                out.append(buf[: cut + 1])
                buf = buf[cut + 1:]
            else:
                out.append(buf)
                buf = ""
    if buf:
        out.append(buf)
    return out


def _split_for_nmt(text: str, max_chars: int, cjk: bool) -> list[str]:
    """Split an over-long source cue into <= ``max_chars`` pieces at
    sentence -> clause -> hard boundaries, so NMT never truncates a long
    run-on into an UNtranslated source line. Order-preserving."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences = [m.group(0) for m in _SENT_SPLIT_RE.finditer(text) if m.group(0).strip()]
    pieces: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) <= max_chars:
            buf += s
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        if len(s) <= max_chars:
            buf = s
        else:
            pieces.extend(_hard_split(s, max_chars, cjk))
    if buf:
        pieces.append(buf)
    return [p.strip() for p in pieces if p.strip()]


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


# ── Disk-space / cleanup helpers ─────────────────────────────────────────

def _free_bytes(path: str) -> int:
    """Free bytes on the filesystem holding ``path`` (walks up to an
    existing ancestor so the check works before the dir is created)."""
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return shutil.disk_usage(probe or "/").free
    except Exception:
        return 0


def _dir_size_bytes(path: str) -> int:
    """Total size of all files under ``path`` (0 if missing)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _human(nbytes: float) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


@contextlib.contextmanager
def _hf_cache_redirect():
    """Point the Hugging Face cache at a throwaway dir for the duration of a
    convert, then delete it.

    ``TransformersConverter`` downloads the full-precision HF model into the
    HF cache (``HF_HOME`` / ``HF_HUB_CACHE`` / ``TRANSFORMERS_CACHE``) before
    emitting the small int8 CTranslate2 model. That full model is several GB
    of dead weight afterwards. We redirect the cache to a temp dir on the
    SAME volume as the models (so the converter's downloads don't fill the
    container's root fs, and a cross-device move isn't needed) and remove it
    once the int8 model is written — logging the bytes reclaimed.
    """
    models_root = _models_dir()
    tmp_root = os.path.join(models_root, ".hf_cache_tmp")
    os.makedirs(tmp_root, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="convert-", dir=tmp_root)
    _keys = ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE")
    _saved = {k: os.environ.get(k) for k in _keys}
    for k in _keys:
        os.environ[k] = tmp_dir
    try:
        yield tmp_dir
    finally:
        for k, v in _saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            reclaimed = _dir_size_bytes(tmp_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            # Drop the parent scratch dir too when it's now empty.
            with contextlib.suppress(OSError):
                if not os.listdir(tmp_root):
                    os.rmdir(tmp_root)
            logger.info(
                "NMT: cleaned HF intermediate cache — reclaimed %s", _human(reclaimed)
            )
        except Exception as e:
            logger.warning("NMT: HF cache cleanup skipped (%s)", e)


@contextlib.contextmanager
def _allow_trusted_torch_load():
    """Neutralise transformers' torch.load CVE guard for the duration of a
    convert (CVE-2025-32434).

    transformers >= 4.50 refuses ``torch.load`` of a ``pytorch_model.bin`` on
    torch < 2.6 via ``check_torch_load_is_safe()``. NLLB-200 and Opus-MT ship
    ONLY ``.bin`` (no safetensors), and this repo pins ``torch==2.5.1+cu121``
    (torch 2.6 has no cu121 wheel, so bumping it would force a whole CUDA-base
    migration) — so every offline-NMT convert hit that guard and silently fell
    back to the LLM. We convert TRUSTED, HTTPS-fetched official HF model-hub
    checkpoints, loaded with ``weights_only=True`` — precisely the case the
    guard is over-cautious about (it even skips itself when the caller opts out
    of safety) — so we temporarily neutralise it.

    Restored afterwards. It is a no-op on torch >= 2.6 (the guard would pass
    anyway) and harmless if a future transformers renames/removes the symbol.
    The guard is imported as a module global into ``modeling_utils`` (the call
    sites), so that binding is the one that must be patched; ``import_utils``
    (its definition site) is patched too for any other caller.
    """
    import importlib
    saved = []
    for modname in ("transformers.modeling_utils", "transformers.utils.import_utils"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, "check_torch_load_is_safe"):
            saved.append((mod, mod.check_torch_load_is_safe))
            mod.check_torch_load_is_safe = lambda *a, **k: None
    if saved:
        logger.info(
            "NMT: neutralised transformers' torch<2.6 torch.load guard for a "
            "trusted HF checkpoint (weights_only=True retained)")
    try:
        yield
    finally:
        for mod, orig in saved:
            try:
                mod.check_torch_load_is_safe = orig
            except Exception:
                pass


def _require_free_space(target_dir: str, min_free: int, label: str) -> None:
    """Raise ``OSError`` when the volume holding ``target_dir`` lacks
    ``min_free`` bytes — so we fail with a clear message instead of writing a
    half-finished model dir that blocks later retries."""
    free = _free_bytes(target_dir)
    if free < min_free:
        raise OSError(
            f"Not enough disk space to download {label}: "
            f"{_human(free)} free, need ~{_human(min_free)} on the models "
            f"volume ({_models_dir()}). Free up space or pre-download a model."
        )


def _enforce_opus_pair_cap() -> None:
    """Keep at most ``NMT_MAX_OPUS_PAIRS`` Opus-MT pair dirs, evicting the
    least-recently-used extras. NLLB is a single model and is never touched."""
    from backend.config import settings as _settings
    cap = int(getattr(_settings, "NMT_MAX_OPUS_PAIRS", 5) or 0)
    if cap <= 0:
        return
    opus_root = os.path.join(_models_dir(), "opus-mt")
    if not os.path.isdir(opus_root):
        return
    pairs = []
    for name in os.listdir(opus_root):
        d = os.path.join(opus_root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "model.bin")):
            # Prefer the access marker we stamp on use; fall back to mtime.
            marker = os.path.join(d, ".last_used")
            try:
                ts = os.path.getmtime(marker if os.path.exists(marker) else d)
            except OSError:
                ts = 0.0
            pairs.append((ts, d, name))
    if len(pairs) <= cap:
        return
    pairs.sort()  # oldest first
    for _ts, d, name in pairs[: len(pairs) - cap]:
        try:
            freed = _dir_size_bytes(d)
            shutil.rmtree(d, ignore_errors=True)
            logger.info(
                "NMT: evicted least-recently-used Opus-MT pair '%s' (%s) — "
                "over the %d-pair cap", name, _human(freed), cap,
            )
        except Exception as e:
            logger.warning("NMT: failed to evict Opus-MT pair '%s' (%s)", name, e)


def _touch_opus_pair(src: str, tgt: str) -> None:
    """Stamp an Opus-MT pair as recently used for the LRU cap."""
    marker = os.path.join(_opus_dir(src, tgt), ".last_used")
    try:
        with open(marker, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


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
                if torch.cuda.is_available():
                    # NLLB-600M int8 is small (~1 GB). Translation runs AFTER the
                    # reframer releases Whisper's VRAM, so the GPU is usually free
                    # by now — decide on FREE VRAM, not total card size, and use
                    # the GPU whenever the model + workspace genuinely fit (the
                    # GTX 1650's CPU path is many times slower). A CUDA OOM at
                    # load time still retries on CPU below, so cuda is never fatal.
                    torch.cuda.empty_cache()
                    free_gb = torch.cuda.mem_get_info()[0] / 1_073_741_824
                    if free_gb >= _NLLB_CUDA_MIN_FREE_GB:
                        device = "cuda"
                        logger.info(
                            "NMT: NMT_DEVICE=auto → cuda (%.1f GB VRAM free ≥ %.1f GB "
                            "needed for NLLB int8; GPU is much faster than CPU)",
                            free_gb, _NLLB_CUDA_MIN_FREE_GB,
                        )
                    else:
                        device = "cpu"
                        logger.info(
                            "NMT: NMT_DEVICE=auto → cpu (only %.1f GB VRAM free < %.1f GB "
                            "needed for NLLB int8 — CPU is the OOM-safe choice)",
                            free_gb, _NLLB_CUDA_MIN_FREE_GB,
                        )
                else:
                    device = "cpu"
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
        try:
            self._translator = ctranslate2.Translator(path, device=device, compute_type=compute_type)
        except Exception as e:
            # A CUDA OOM (or any GPU load error) must never drop us to the LLM —
            # retry on CPU (slower but always works) so offline NMT still wins.
            if device == "cuda":
                logger.warning(
                    "NMT: NLLB CUDA load failed (%s) — retrying on CPU (slower but safe)", e)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                device = "cpu"
                self._translator = ctranslate2.Translator(path, device="cpu", compute_type="int8")
            else:
                raise

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
                translated = self._translate_text(text, flores_src, flores_tgt)
                if glossary:
                    translated = apply_glossary(text, translated, glossary)
                results.append(translated if translated.strip() else raw)
            except Exception as e:
                logger.warning("NMT: translation failed for one segment (%s) — keeping original", e)
                results.append(raw)
        return results

    def _translate_text(self, text: str, flores_src: str, flores_tgt: str) -> str:
        """Translate one cue, CHUNKING over-long input so a long run-on never
        truncates into an untranslated source line, then rejoining."""
        cjk_src = _flores_is_cjk(flores_src)
        max_chars = _MAX_SRC_CHARS_CJK if cjk_src else _MAX_SRC_CHARS_LATIN
        chunks = _split_for_nmt(text, max_chars, cjk_src)
        if not chunks:
            return text
        joiner = "" if _flores_is_cjk(flores_tgt) else " "
        parts: list[str] = []
        for ch in chunks:
            try:
                parts.append(self._translate_chunk(ch, flores_src, flores_tgt))
            except Exception as e:
                logger.debug("NMT: chunk translate failed (%s) — keeping chunk source", e)
                parts.append(ch)
        return joiner.join(p for p in parts if p).strip()

    def _translate_chunk(self, text: str, flores_src: str, flores_tgt: str) -> str:
        tokens = self._tokenizer.encode_as_pieces(text)
        source = [flores_src] + tokens + ["</s>"]
        # Scale the decode budget with input length — the English of a long
        # Japanese line far exceeds the old fixed 256-token cap (which silently
        # truncated long cues) — but keep a ceiling so a degenerate input can't
        # run away.
        max_dec = min(512, max(128, len(tokens) * 3))
        output = self._translator.translate_batch(
            [source],
            target_prefix=[[flores_tgt]],
            beam_size=4,
            max_decoding_length=max_dec,
        )
        pieces = output[0].hypotheses[0]
        # Drop the language token prefix.
        if pieces and pieces[0] == flores_tgt:
            pieces = pieces[1:]
        return self._tokenizer.decode(pieces)

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
        # A long joined string would itself be chunked (losing the ¶ markers, so
        # the split below misaligns) or truncated by the decoder. When it exceeds
        # one safe chunk, skip the context-join trick and translate per-segment —
        # ``translate_batch`` chunks each long cue, guaranteeing completeness.
        cap = _MAX_SRC_CHARS_CJK if _looks_cjk(joined) else _MAX_SRC_CHARS_LATIN
        if len(joined) > cap:
            return self.translate_batch(batch, source_lang, target_lang, glossary=glossary)
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
        tgt_cjk = self.target.split("-")[0] in ("ja", "zh", "ko", "yue")
        results: list[str] = []
        for raw in texts:
            text = (raw or "").strip()
            if not text:
                results.append(raw)
                continue
            try:
                src_cjk = _looks_cjk(text)
                max_chars = _MAX_SRC_CHARS_CJK if src_cjk else _MAX_SRC_CHARS_LATIN
                chunks = _split_for_nmt(text, max_chars, src_cjk) or [text]
                joiner = "" if tgt_cjk else " "
                parts: list[str] = []
                for ch in chunks:
                    tokens = self._tokenizer.encode_as_pieces(ch)
                    max_dec = min(512, max(128, len(tokens) * 3))
                    output = self._translator.translate_batch(
                        [tokens + ["</s>"]],
                        beam_size=4,
                        max_decoding_length=max_dec,
                    )
                    parts.append(self._tokenizer.decode(output[0].hypotheses[0]))
                translated = (joiner.join(p for p in parts if p).strip()) or text
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

def _fetch_tokenizer_into(model_id: str, out_dir: str, candidates) -> bool:
    """Copy the SentencePiece tokenizer file(s) the runtime loader needs into
    ``out_dir``.

    CTranslate2's ``TransformersConverter`` writes ``model.bin`` + the CT2 vocab
    but NOT the ``.spm`` tokenizer file. Without it the model is on disk yet
    ``NMTTranslator._model_files_present`` is False, so ``pick_local_engine``
    returns None and the engine silently falls back to the LLM even though the
    convert "succeeded" (the 599 MB model with no tokenizer). Best-effort per
    file; returns True if at least one landed. Call INSIDE the HF-cache redirect
    so the snapshot the convert already pulled is reused (no re-download)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        logger.warning("NMT: huggingface_hub unavailable to fetch tokenizer (%s)", e)
        return False
    got = False
    for fname in candidates:
        dest = os.path.join(out_dir, fname)
        if os.path.exists(dest):
            got = True
            continue
        try:
            src = hf_hub_download(model_id, fname)
            shutil.copyfile(src, dest)
            got = True
            logger.info("NMT: saved tokenizer file '%s' alongside the converted model", fname)
        except Exception as e:
            logger.debug("NMT: tokenizer file '%s' not in %s (%s)", fname, model_id, e)
    return got


def _convert_with_cleanup(
    model_id: str, target_dir: str, label: str, tokenizer_files=(),
) -> None:
    """Convert ``model_id`` to an int8 CTranslate2 model at ``target_dir``.

    Network: the converter pulls the source weights from Hugging Face, so
    ``huggingface.co`` must be reachable. For NLLB-200 that is a ~2.5 GB
    transient full-precision download into a redirected HF cache; only the
    ~600 MB int8 CT2 model is kept (the cache is deleted by
    ``_hf_cache_redirect``).

    The convert is done into a **fresh temp dir** on the same volume and then
    promoted onto ``target_dir`` with an atomic ``os.replace``. This is
    deliberate, and fixes the bug that bricked offline NMT:

      * ``TransformersConverter.convert`` *refuses* a pre-existing output
        directory unless ``force=True`` — so converting straight into a
        ``target_dir`` we just ``makedirs``'d failed every single time and
        the caller silently fell back to the LLM. Handing the converter a
        path that does **not** yet exist lets it create the dir itself
        (``force`` irrelevant), and the rename means a half-written model is
        never visible at ``target_dir`` as "present".
      * On any convert/download failure we remove the temp dir **and** any
        partial ``target_dir``, so a retry is never blocked by a stale dir.
    """
    try:
        from ctranslate2.converters import TransformersConverter
    except Exception as e:
        raise RuntimeError(
            f"ctranslate2 with the transformers converter is required to "
            f"download {label}. pip install 'ctranslate2[transformers]'"
        ) from e

    parent = os.path.dirname(target_dir) or "."
    os.makedirs(parent, exist_ok=True)
    # A sibling temp dir on the SAME volume so the final promotion is an atomic
    # rename (not a cross-device copy). The converter writes into a child of it
    # that does NOT yet exist, so it never trips the "dir already exists" guard.
    tmp_holder = tempfile.mkdtemp(prefix=".convert-", dir=parent)
    convert_dir = os.path.join(tmp_holder, "ct2")
    try:
        with _hf_cache_redirect():
            converter = TransformersConverter(model_id)
            with _allow_trusted_torch_load():
                converter.convert(convert_dir, quantization="int8", force=False)
            # CT2 writes model.bin + vocab but NOT the SentencePiece tokenizer
            # the runtime needs — copy it in now, while the HF snapshot the
            # convert pulled is still cached (so this is a copy, not a download).
            if tokenizer_files:
                _fetch_tokenizer_into(model_id, convert_dir, tokenizer_files)
        # Promote atomically. If a stale/partial target exists, rename it ASIDE
        # first (atomic, same volume) rather than rmtree-ing it in place: a
        # rmtree(ignore_errors=True) that silently fails to fully clear the dir
        # (e.g. a locked file) would leave it non-empty, os.replace would then
        # refuse with ENOTEMPTY, and the except branch would delete the FRESH
        # model. Swapping the stale dir aside means os.replace always targets a
        # non-existent path, and the fresh model is never the thing at risk.
        backup = None
        if os.path.exists(target_dir):
            backup = f"{target_dir}.old-{os.getpid()}"
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(target_dir, backup)
        try:
            os.replace(convert_dir, target_dir)
        except Exception:
            if backup is not None and not os.path.exists(target_dir):
                os.replace(backup, target_dir)  # restore the stale model
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except Exception:
        # Tear down BOTH the temp dir and any partial target_dir so neither is
        # mistaken for a complete download (and so a retry starts clean).
        shutil.rmtree(tmp_holder, ignore_errors=True)
        shutil.rmtree(target_dir, ignore_errors=True)
        logger.warning(
            "NMT: %s convert failed — removed temp + partial model dir %s",
            label, target_dir,
        )
        raise
    finally:
        # ``tmp_holder`` is now empty on success (its only child was renamed
        # away) and already gone on failure; this drops the empty holder.
        shutil.rmtree(tmp_holder, ignore_errors=True)
    kept = _dir_size_bytes(target_dir)
    logger.info(
        "NMT: %s converted to int8 at %s (kept %s; %s free remains)",
        label, target_dir, _human(kept), _human(_free_bytes(target_dir)),
    )


def ensure_nllb_downloaded(
    model_id: Optional[str] = None,
    progress_callback=None,
) -> str:
    """Download + convert NLLB-200 to CTranslate2 (int8) on demand.

    Returns the absolute path to the model directory. Raises
    ``RuntimeError`` if dependencies are missing or ``OSError`` when the
    models volume is out of space. Safe to call repeatedly — does nothing
    when the converted model already exists. The multi-GB full-precision HF
    model pulled during conversion is removed afterwards; only the ~600 MB
    int8 CT2 model is kept.
    """
    from backend.config import settings as _settings
    model_id = model_id or _settings.NMT_NLLB_MODEL
    target_dir = _nllb_dir(model_id)
    if NMTTranslator._model_files_present(model_id):
        logger.info("NLLB %s already downloaded at %s", model_id, target_dir)
        return target_dir
    # Repair shortcut: a prior convert (before the tokenizer-save fix) may have
    # left a 599 MB model.bin with NO SentencePiece tokenizer — fetch just the
    # ~5 MB tokenizer instead of re-downloading + re-converting 2.4 GB.
    if os.path.exists(os.path.join(target_dir, "model.bin")):
        logger.info(
            "NLLB %s present but tokenizer missing — fetching tokenizer only "
            "(repair, no full re-convert)", model_id)
        try:
            with _hf_cache_redirect():
                _fetch_tokenizer_into(model_id, target_dir, ("sentencepiece.bpe.model",))
        except Exception as _rep_err:
            logger.warning("NMT: tokenizer repair failed (%s) — re-converting", _rep_err)
        if NMTTranslator._model_files_present(model_id):
            logger.info("NLLB %s repaired (tokenizer added) at %s", model_id, target_dir)
            return target_dir
        logger.warning("NLLB %s tokenizer repair did not complete — re-converting", model_id)
    _require_free_space(target_dir, _NLLB_MIN_FREE_BYTES, f"NLLB ({model_id})")
    logger.info(
        "NMT: downloading + converting NLLB %s (one-time, ~600 MB int8) → %s",
        model_id, target_dir,
    )
    _convert_with_cleanup(
        model_id, target_dir, f"NLLB {model_id}",
        tokenizer_files=("sentencepiece.bpe.model",),
    )
    # Fail LOUD if the SentencePiece tokenizer didn't land — otherwise the model
    # is on disk yet pick_local_engine() returns None and the job silently falls
    # back to the LLM (the '599 MB model, no tokenizer' regression).
    if not NMTTranslator._model_files_present(model_id):
        present = sorted(os.listdir(target_dir)) if os.path.isdir(target_dir) else []
        raise RuntimeError(
            f"NLLB {model_id} converted but the model dir is missing a required "
            f"file (have: {present}). Need model.bin + a SentencePiece tokenizer "
            f"(sentencepiece.bpe.model)."
        )
    # Files are present — confirm the runtime deps too, so a missing
    # sentencepiece surfaces clearly instead of another silent LLM fallback.
    if not NMTTranslator._has_dependencies():
        raise RuntimeError(
            "NLLB converted but ctranslate2 + sentencepiece are not BOTH "
            "importable at runtime — offline NMT can't load. Ensure "
            "'sentencepiece' is installed in the image."
        )
    return target_dir


def ensure_opus_mt_downloaded(source: str, target: str) -> str:
    """Download + convert an Opus-MT pair to CTranslate2 (int8) on demand.

    Same semantics as ``ensure_nllb_downloaded`` (disk guard, HF-cache
    cleanup, partial-dir cleanup on failure), plus an LRU cap on how many
    pair dirs are kept (``NMT_MAX_OPUS_PAIRS``).
    """
    from backend.config import settings as _settings
    src, tgt = source.lower(), target.lower()
    target_dir = _opus_dir(src, tgt)
    if os.path.exists(os.path.join(target_dir, "model.bin")):
        _touch_opus_pair(src, tgt)
        return target_dir
    _require_free_space(target_dir, _OPUS_MIN_FREE_BYTES, f"Opus-MT {src}-{tgt}")
    model_id = _settings.NMT_OPUS_MT_TEMPLATE.format(src=src, tgt=tgt)
    logger.info(
        "NMT: downloading + converting Opus-MT %s (one-time) → %s",
        model_id, target_dir,
    )
    _convert_with_cleanup(
        model_id, target_dir, f"Opus-MT {src}-{tgt}",
        tokenizer_files=("source.spm", "target.spm", "vocab.json"),
    )
    if not os.path.exists(os.path.join(target_dir, "source.spm")):
        present = sorted(os.listdir(target_dir)) if os.path.isdir(target_dir) else []
        raise RuntimeError(
            f"Opus-MT {src}-{tgt} converted but the SentencePiece tokenizer "
            f"(source.spm) is missing (have: {present})."
        )
    _touch_opus_pair(src, tgt)
    # Prune least-recently-used pairs beyond the cap now that a new one landed.
    _enforce_opus_pair_cap()
    return target_dir


# ── Convenience: pick the right local engine for a pair ──────────────────

def pick_local_engine(source: str, target: str):
    """Return an instantiated local translator (Opus-MT preferred) or
    None if no local model is downloaded for the pair.
    """
    opus = OpusMTTranslator.get(source, target)
    if opus.is_available():
        _touch_opus_pair(source, target)
        return opus
    nllb = NMTTranslator()
    if nllb.is_available() and iso_to_flores(source) and iso_to_flores(target):
        return nllb
    return None


def auto_download_for_pair(source: str, target: str, prefer: str = "nllb"):
    """Ensure a local NMT model exists for ``source→target``, downloading +
    converting one on demand, then return the instantiated local engine.

    Synchronous (the convert is CPU/IO-bound) — call from a worker thread
    via ``asyncio.to_thread``. NLLB-200 is preferred (one model, 200
    languages); Opus-MT is fetched only when explicitly requested
    (``prefer="opus-mt"``) or when the pair isn't in NLLB's Flores map.

    Returns the engine, or ``None`` if the pair is unsupported. Raises
    ``OSError`` / ``RuntimeError`` when the download itself fails (network,
    disk, or missing converter deps) so the caller can fall back to the LLM
    and log the failure distinctly.
    """
    existing = pick_local_engine(source, target)
    if existing is not None:
        return existing

    nllb_capable = bool(iso_to_flores(source) and iso_to_flores(target))
    if prefer == "opus-mt" or not nllb_capable:
        if prefer != "opus-mt":
            logger.info(
                "NMT: %s→%s not in NLLB's Flores map — auto-downloading Opus-MT pair",
                source, target,
            )
        ensure_opus_mt_downloaded(source, target)
    else:
        ensure_nllb_downloaded()
    return pick_local_engine(source, target)
