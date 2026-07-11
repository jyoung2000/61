"""Local NMT translation backends — NLLB-200 + Opus-MT via CTranslate2.

Provides a fast, free, fully local alternative to LLM-based translation.
Models are downloaded on demand (NEVER at startup) — call
``ensure_nllb_downloaded()`` from a UI button or CLI before first use.

Two engines:
  - NLLB-200-distilled: 200 languages, broadest coverage. Default is the
    1.3B distilled (~1.3-1.5 GB int8, markedly more fluent); the 600M
    (~600 MB int8) is a lighter env-pinned fallback for smaller cards.
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
# smaller int8 CTranslate2 copy — so we need headroom for BOTH transiently
# (the fp32 cache is deleted by ``_hf_cache_redirect`` once the int8 is out).
#   * NLLB-200-distilled-600M: ~2.4 GB fp32 on the Hub + ~0.6 GB int8 out.
#   * NLLB-200-distilled-1.3B: ~5.5 GB fp32 on the Hub + ~1.3-1.5 GB int8 out.
_NLLB_MIN_FREE_BYTES = 8 * 1024 ** 3        # 1.3B (default): fp32 cache + int8 out
_NLLB_600M_MIN_FREE_BYTES = 6 * 1024 ** 3   # 600M (env-pinned fallback)
_OPUS_MIN_FREE_BYTES = 3 * 1024 ** 3        # 3 GB headroom per Opus-MT pair

# Free VRAM (GB) needed to load NLLB int8 on CUDA: weights + CT2 activation
# workspace + margin. Above this, NMT_DEVICE=auto uses the GPU even on a 4 GB
# card (Whisper VRAM is released before translation); below it, CPU. Scaled by
# model size — the 1.3B int8 (~1.3-1.5 GB) needs more headroom than the 600M.
_NLLB_CUDA_MIN_FREE_GB = 1.8           # 600M: ~1 GB weights + workspace + margin
_NLLB_1P3B_CUDA_MIN_FREE_GB = 2.4      # 1.3B: ~1.5 GB weights + workspace + margin


def _is_nllb_1p3b(model_id: Optional[str]) -> bool:
    """True for the 1.3B-distilled checkpoint (larger disk + VRAM footprint)."""
    return "1.3b" in (model_id or "").lower()


def _nllb_min_free_bytes(model_id: Optional[str]) -> int:
    """Disk-headroom floor for downloading/converting ``model_id``."""
    return _NLLB_MIN_FREE_BYTES if _is_nllb_1p3b(model_id) else _NLLB_600M_MIN_FREE_BYTES


def _nllb_cuda_min_free_gb(model_id: Optional[str]) -> float:
    """Free-VRAM floor for loading ``model_id`` int8 on CUDA."""
    return _NLLB_1P3B_CUDA_MIN_FREE_GB if _is_nllb_1p3b(model_id) else _NLLB_CUDA_MIN_FREE_GB


# ── ISO 639-1 → Flores-200 mapping for NLLB ──────────────────────────────
# NLLB-200 covers ~200 languages; this maps the common ISO 639-1 codes (plus a
# few regional aliases) to their Flores-200 codes so the user can pick ANY of
# these as a subtitle target — and so an auto-detected SOURCE in any of them
# still translates — instead of failing with "unsupported pair". Keep in sync
# with translator.SUPPORTED_LANGUAGES (same key set).
_FLORES_CODES = {
    # ── Western European ──
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "ca": "cat_Latn",
    "gl": "glg_Latn", "eu": "eus_Latn", "ga": "gle_Latn", "cy": "cym_Latn",
    "is": "isl_Latn", "lb": "ltz_Latn", "mt": "mlt_Latn",
    # ── Nordic ──
    "sv": "swe_Latn", "da": "dan_Latn", "no": "nob_Latn", "nb": "nob_Latn",
    "nn": "nno_Latn", "fi": "fin_Latn",
    # ── Slavic / Baltic / other Eastern European ──
    "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "pl": "pol_Latn", "cs": "ces_Latn",
    "sk": "slk_Latn", "sl": "slv_Latn", "hr": "hrv_Latn", "sr": "srp_Cyrl",
    "bs": "bos_Latn", "bg": "bul_Cyrl", "mk": "mkd_Cyrl", "be": "bel_Cyrl",
    "ro": "ron_Latn", "hu": "hun_Latn", "et": "est_Latn", "lv": "lvs_Latn",
    "lt": "lit_Latn", "sq": "als_Latn", "el": "ell_Grek",
    # ── Middle East / Caucasus / Central Asia ──
    "ar": "arb_Arab", "he": "heb_Hebr", "iw": "heb_Hebr", "fa": "pes_Arab",
    "tr": "tur_Latn", "az": "azj_Latn", "kk": "kaz_Cyrl", "ky": "kir_Cyrl",
    "uz": "uzn_Latn", "tg": "tgk_Cyrl", "hy": "hye_Armn", "ka": "kat_Geor",
    "ku": "kmr_Latn", "ps": "pbt_Arab",
    # ── South Asia ──
    "hi": "hin_Deva", "bn": "ben_Beng", "ur": "urd_Arab", "pa": "pan_Guru",
    "gu": "guj_Gujr", "mr": "mar_Deva", "ta": "tam_Taml", "te": "tel_Telu",
    "kn": "kan_Knda", "ml": "mal_Mlym", "ne": "npi_Deva", "si": "sin_Sinh",
    "or": "ory_Orya", "as": "asm_Beng",
    # ── East / Southeast Asia ──
    "ja": "jpn_Jpan", "ko": "kor_Hang",
    "zh": "zho_Hans", "zh-cn": "zho_Hans", "zh-hans": "zho_Hans",
    "zh-tw": "zho_Hant", "zh-hk": "zho_Hant", "zh-hant": "zho_Hant",
    "yue": "yue_Hant",
    "vi": "vie_Latn", "th": "tha_Thai", "id": "ind_Latn", "ms": "zsm_Latn",
    "tl": "tgl_Latn", "fil": "tgl_Latn", "my": "mya_Mymr", "km": "khm_Khmr",
    "lo": "lao_Laoo", "jv": "jav_Latn", "su": "sun_Latn", "mn": "khk_Cyrl",
    # ── Africa ──
    "sw": "swh_Latn", "am": "amh_Ethi", "ha": "hau_Latn", "yo": "yor_Latn",
    "ig": "ibo_Latn", "zu": "zul_Latn", "xh": "xho_Latn", "sn": "sna_Latn",
    "so": "som_Latn", "af": "afr_Latn", "mg": "plt_Latn", "ny": "nya_Latn",
    "st": "sot_Latn",
}


def iso_to_flores(code: str) -> Optional[str]:
    """Map an ISO 639-1 code to a Flores-200 code (case-insensitive).

    Tolerates full language names ("japanese") and 639-2/3 codes ("jpn") —
    whisper.cpp reports full names, and an unnormalized name must not knock
    the pair out of NLLB's map.
    """
    if not code:
        return None
    from backend.services.language_codes import normalize_lang_code
    return _FLORES_CODES.get(normalize_lang_code(code))


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


# ── Anti-repetition decode kwargs (NLLB / Opus-MT) ───────────────────────
# NLLB-distilled + Opus-MT loop on run-on cues (the same degenerate-repeat
# failure the Whisper path already guards via reframer_audio._decoding_kwargs).
# These CTranslate2 kwargs make the decoder reject verbatim n-gram loops and
# penalise token-level repetition. CTranslate2 >= 4.0 (pinned in requirements)
# supports both.
_CT2_ANTI_REPEAT = {
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
}
# Cached subset of the above the installed build accepts. ``None`` until first
# detected; set to ``{}`` if a call ever proves them unsupported (self-heal).
_CT2_DECODE_KWARGS: Optional[dict] = None


def _ct2_decode_kwargs() -> dict:
    """Anti-repetition decode kwargs to pass to CTranslate2's
    ``Translator.translate_batch``, feature-detected once (mirrors
    ``reframer_audio._decoding_kwargs``' intent so an unexpectedly-old build
    never raises on an unknown kwarg).

    ``translate_batch`` is a pybind11 binding whose signature
    ``inspect.signature`` usually CANNOT read, so we also scan the docstring
    (pybind11 embeds the call signature there). On the pinned >= 4.0 build both
    kwargs are present; when neither source is readable (e.g. ``python -OO``
    strips docstrings) we still pass them — ``_ct2_translate_batch`` self-heals
    by dropping them if a call ever rejects them.
    """
    global _CT2_DECODE_KWARGS
    if _CT2_DECODE_KWARGS is not None:
        return dict(_CT2_DECODE_KWARGS)
    try:
        import inspect
        import ctranslate2
        fn = ctranslate2.Translator.translate_batch
        try:
            names = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            names = set()
        doc = fn.__doc__ or ""
        detected = {k: v for k, v in _CT2_ANTI_REPEAT.items()
                    if k in names or k in doc}
        if detected:
            _CT2_DECODE_KWARGS = dict(detected)
            return dict(detected)
    except Exception:
        # ctranslate2 not importable here; the real call path imports it first.
        return dict(_CT2_ANTI_REPEAT)
    # Signature + docstring both unreadable — pass anyway (build is >= 4.0) and
    # let the call-site self-heal demote to {} if it's genuinely unsupported.
    return dict(_CT2_ANTI_REPEAT)


def _ct2_translate_batch(translator, source_list, **kwargs):
    """``translator.translate_batch`` with anti-repetition kwargs injected.

    Self-heals on an unexpectedly-old CTranslate2 build: if the call raises a
    ``TypeError`` about an unexpected keyword, the anti-repeat kwargs are
    dropped (cached as unsupported) and the call is retried once — so the
    decode still runs, just without the loop guard, instead of failing.
    """
    global _CT2_DECODE_KWARGS
    extra = _ct2_decode_kwargs()
    try:
        return translator.translate_batch(source_list, **kwargs, **extra)
    except TypeError as e:
        msg = str(e)
        if extra and ("repetition_penalty" in msg or "no_repeat_ngram_size" in msg
                      or "unexpected keyword" in msg):
            logger.warning(
                "NMT: CTranslate2 rejected anti-repetition kwargs (%s) — retrying "
                "without them (older build); loop guard disabled", e)
            _CT2_DECODE_KWARGS = {}
            return translator.translate_batch(source_list, **kwargs)
        raise


# ── Numbered-tag context join (survives NMT mangling) ────────────────────
# Each cue in a context block is wrapped ⟦i⟧…⟦/i⟧. Numbered tags are far more
# robust to NMT mangling than a single bare separator: every cue is recoverable
# independently by index, and the parser tolerates bracket substitution
# (⟦→[ /【/〔) and stray whitespace the model introduces.
_NMT_TAG_RE = re.compile(r"[⟦\[【〔]\s*(/?)\s*(\d{1,3})\s*[⟧\]】〕]")


def _wrap_numbered_tag(n: int, text: str) -> str:
    """Wrap ``text`` in the numbered cue tag ``⟦n⟧…⟦/n⟧``."""
    return f"⟦{n}⟧{text}⟦/{n}⟧"


def _parse_numbered_tags(
    text: str, lo: int, hi: int, n_total: int,
) -> Optional[list[str]]:
    """Recover the cue texts for 1-based indices ``lo..hi-1`` from a translated
    block wrapped in numbered tags.

    Returns ``None`` (caller falls back to the per-cue path) if any requested
    index is missing or came back empty. Only tags numbered ``1..n_total`` are
    treated as delimiters, so a stray bracketed number inside a cue's own text
    (e.g. "[12]") can't masquerade as a tag.
    """
    tokens = [
        (m.start(), m.end(), bool(m.group(1)), int(m.group(2)))
        for m in _NMT_TAG_RE.finditer(text)
        if 1 <= int(m.group(2)) <= n_total
    ]
    if not tokens:
        return None
    result: dict[int, str] = {}
    for ti, (_s, e, is_close, num) in enumerate(tokens):
        if is_close or num in result:
            continue
        seg_start = e
        seg_end = len(text)
        for (s2, _e2, is_close2, num2) in tokens[ti + 1:]:
            if not is_close2:        # the next OPENING tag bounds this cue
                seg_end = s2
                break
            if num2 == num:          # matching CLOSING tag bounds this cue
                seg_end = s2
                break
            # a stray closing tag for another index → keep scanning
        result[num] = text[seg_start:seg_end].strip()
    out: list[str] = []
    for i in range(lo, hi):
        v = result.get(i, "")
        if not v:
            return None
        out.append(v)
    return out


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


def _opus_dir(src: str, tgt: str, subdir: str = "opus-mt") -> str:
    return os.path.join(_models_dir(), subdir, f"{src}-{tgt}")


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
        # Context-join telemetry (Task 3): how often the numbered-tag join
        # re-aligned cleanly vs. fell to the per-cue-with-context path.
        self._ctx_join_ok = 0
        self._ctx_join_tag_fail = 0
        self._ctx_join_too_long = 0

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
        min_free_gb = _nllb_cuda_min_free_gb(self.model_id)
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    # NLLB int8 is small (600M ~1 GB, 1.3B ~1.3-1.5 GB).
                    # Translation runs AFTER the reframer releases Whisper's VRAM,
                    # so the GPU is usually free by now — decide on FREE VRAM, not
                    # total card size, against THIS model's footprint, and use the
                    # GPU whenever the model + workspace genuinely fit (the GTX
                    # 1650's CPU path is many times slower). A CUDA OOM at load
                    # time still retries on CPU below, so cuda is never fatal.
                    torch.cuda.empty_cache()
                    free_gb = torch.cuda.mem_get_info()[0] / 1_073_741_824
                    if free_gb >= min_free_gb:
                        device = "cuda"
                        logger.info(
                            "NMT: NMT_DEVICE=auto → cuda (%.1f GB VRAM free ≥ %.1f GB "
                            "needed for %s int8; GPU is much faster than CPU)",
                            free_gb, min_free_gb, self.model_id,
                        )
                    else:
                        device = "cpu"
                        logger.info(
                            "NMT: NMT_DEVICE=auto → cpu (only %.1f GB VRAM free < %.1f GB "
                            "needed for %s int8 — CPU is the OOM-safe choice)",
                            free_gb, min_free_gb, self.model_id,
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
        output = _ct2_translate_batch(
            self._translator,
            [source],
            target_prefix=[[flores_tgt]],
            beam_size=5,
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

        NLLB doesn't accept a system prompt the way an LLM does, so the context
        cues are translated inline alongside the batch and only the in-batch
        translations are kept. Each cue is wrapped in a numbered tag
        (``⟦i⟧…⟦/i⟧``) and the joined block is translated in one shot, then the
        batch cues are recovered by tag index — far more robust to NMT mangling
        than the old single ``¶`` separator (which NLLB routinely dropped,
        collapsing the whole batch back to context-free per-segment).

        When the tagged block is too long for one safe decode chunk, or tag
        re-alignment fails, we DON'T drop context: each cue is retried
        individually with 1-2 prior cues prepended (translated, then discarded)
        so isolated retries keep their referential context.
        """
        if not batch:
            return []
        if not any((c or "").strip() for c in batch):
            return list(batch)

        all_cues = list(context_before) + list(batch) + list(context_after)
        n_total = len(all_cues)
        start = len(context_before)            # 0-based offset of batch
        lo = start + 1                         # 1-based tag index of first batch cue
        hi = start + len(batch) + 1            # exclusive upper bound

        tagged = " ".join(
            _wrap_numbered_tag(i + 1, (c or "").strip())
            for i, c in enumerate(all_cues)
        )
        # Only context-join when the tagged block fits one safe decode chunk —
        # otherwise it would itself be chunked/truncated (losing tags). The tags
        # are part of the decoded source, so they count against the cap.
        cap = _MAX_SRC_CHARS_CJK if _looks_cjk(tagged) else _MAX_SRC_CHARS_LATIN
        if len(tagged) <= cap:
            try:
                translated_joined = self.translate_batch(
                    [tagged], source_lang, target_lang, glossary=glossary,
                )[0]
                recovered = _parse_numbered_tags(translated_joined, lo, hi, n_total)
            except Exception as e:
                logger.debug("NMT context-join: translate/parse error (%s)", e)
                recovered = None
            if recovered is not None and len(recovered) == len(batch):
                self._ctx_join_ok += 1
                logger.debug(
                    "NMT context-join: tag re-alignment OK (%d cues; ok=%d)",
                    len(batch), self._ctx_join_ok)
                return recovered
            self._ctx_join_tag_fail += 1
            logger.info(
                "NMT context-join: tag re-alignment FAILED (%d cues) — retrying "
                "per-cue with context (tag-fail=%d, ok=%d)",
                len(batch), self._ctx_join_tag_fail, self._ctx_join_ok)
        else:
            self._ctx_join_too_long += 1
            logger.debug(
                "NMT context-join: block too long for one safe chunk (%d chars > "
                "%d) — per-cue with context (too-long=%d)",
                len(tagged), cap, self._ctx_join_too_long)
        return self._translate_batch_with_per_cue_context(
            batch, context_before, source_lang, target_lang, glossary)

    def _translate_batch_with_per_cue_context(
        self, batch, context_before, source_lang, target_lang, glossary,
    ) -> list[str]:
        """Translate each cue individually, prepending 1-2 prior SOURCE cues as
        referential context (translated, then discarded). Guarantees one output
        per input cue and never drops context the way the old context-free
        fallback did."""
        out: list[str] = []
        prior = list(context_before)
        for cue in batch:
            out.append(self._translate_one_with_context(
                cue, prior, source_lang, target_lang, glossary))
            prior.append(cue)
        return out

    def _translate_one_with_context(
        self, cue, prior_cues, source_lang, target_lang, glossary,
    ) -> str:
        """Translate a single cue with up to 2 prior cues prepended for
        pronoun/gender/subject resolution. The context is translated but
        DISCARDED — only the tagged target cue is kept. Falls back to a plain
        chunked single-cue translation (always complete) if the tag is lost."""
        cue_s = (cue or "").strip()
        if not cue_s:
            return cue
        ctx = [c for c in prior_cues if (c or "").strip()][-2:]
        if ctx:
            mini = f"{' '.join(ctx)} {_wrap_numbered_tag(1, cue_s)}"
            cap = _MAX_SRC_CHARS_CJK if _looks_cjk(mini) else _MAX_SRC_CHARS_LATIN
            if len(mini) <= cap:
                try:
                    tj = self.translate_batch(
                        [mini], source_lang, target_lang, glossary=glossary)[0]
                    rec = _parse_numbered_tags(tj, 1, 2, 1)
                    if rec and rec[0].strip():
                        return rec[0]
                except Exception as e:
                    logger.debug("NMT per-cue context retry failed (%s)", e)
        tb = self.translate_batch(
            [cue_s], source_lang, target_lang, glossary=glossary)
        return tb[0] if (tb and (tb[0] or "").strip()) else cue


# ── Opus-MT wrapper (one model per pair) ─────────────────────────────────

class OpusMTTranslator:
    """Local Helsinki-NLP/opus-mt translator for a single language pair.

    Much smaller than NLLB (~300 MB each), faster on European pairs, but
    one model per direction — so we lazy-create translators on demand
    and cache them.
    """

    _cache: dict[tuple, "OpusMTTranslator"] = {}

    def __init__(self, source: str, target: str, *,
                 subdir: str = "opus-mt", model_template: Optional[str] = None):
        # Normalize to ISO 639-1 — source/target are substituted into HF repo
        # ids ("staka/fugumt-{src}-{tgt}"), where a full name like "japanese"
        # produces an invalid repo and a hard download failure.
        from backend.services.language_codes import normalize_lang_code
        self.source = normalize_lang_code(source)
        self.target = normalize_lang_code(target)
        # ``subdir`` + ``model_template`` let a Marian-format *variant* (e.g.
        # FuguMT, staka/fugumt-ja-en — a Japanese-specialised translator) reuse
        # this exact CTranslate2/SentencePiece loader in its own cache dir.
        self.subdir = subdir
        self.model_template = model_template
        self._translator = None
        self._tokenizer = None
        self._loaded = False

    @classmethod
    def get(cls, source: str, target: str, *,
            subdir: str = "opus-mt", model_template: Optional[str] = None) -> "OpusMTTranslator":
        from backend.services.language_codes import normalize_lang_code
        key = (normalize_lang_code(source), normalize_lang_code(target), subdir)
        if key not in cls._cache:
            cls._cache[key] = cls(source, target, subdir=subdir, model_template=model_template)
        return cls._cache[key]

    @property
    def hf_repo(self) -> str:
        """The HF repo id this variant loads from (for logging/labels)."""
        from backend.config import settings as _s
        tmpl = self.model_template or _s.NMT_OPUS_MT_TEMPLATE
        return tmpl.format(src=self.source, tgt=self.target)

    def is_available(self) -> bool:
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401
        except Exception:
            return False
        path = _opus_dir(self.source, self.target, self.subdir)
        return os.path.exists(os.path.join(path, "model.bin"))

    def load(self):
        if self._loaded:
            return
        import ctranslate2
        import sentencepiece as spm
        path = _opus_dir(self.source, self.target, self.subdir)
        if not os.path.exists(os.path.join(path, "model.bin")):
            raise FileNotFoundError(
                f"Opus-MT model {self.source}-{self.target} not downloaded at {path}."
            )
        device = "cpu"  # Opus-MT runs faster on CPU for small batches.
        # Worker parallelism: translate_batch() now sends REAL multi-example
        # batches, and inter_threads workers decode sub-batches concurrently —
        # this is where the per-cue → batched rewrite's wall-time win comes
        # from on a multi-core host. 0 = auto (half the cores, capped at 4,
        # so the pipeline's other stages keep breathing room).
        from backend.config import settings as _s
        inter = int(getattr(_s, "NMT_CT2_INTER_THREADS", 0) or 0)
        if inter <= 0:
            inter = max(1, min(4, (os.cpu_count() or 4) // 2))
        intra = int(getattr(_s, "NMT_CT2_INTRA_THREADS", 0) or 0)
        kwargs = {"inter_threads": inter}
        if intra > 0:
            kwargs["intra_threads"] = intra
        try:
            self._translator = ctranslate2.Translator(
                path, device=device, compute_type="int8", **kwargs)
        except TypeError:
            # Very old CT2 without the threading kwargs — load plain.
            self._translator = ctranslate2.Translator(
                path, device=device, compute_type="int8")
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

    def translate_with_context(
        self,
        batch: list[str],
        context_before: list[str],
        context_after: list[str],
        source_lang: str = "",
        target_lang: str = "",
        glossary: Optional[dict] = None,
    ) -> list[str]:
        """Per-cue contextual translation for the Marian family (FuguMT/Opus).

        The AUTO-selected ja→en engine (FuguMT) previously translated every
        cue in total isolation, while the context machinery only reached
        NLLB — and Japanese drops subjects/pronouns, so isolated-cue decoding
        is exactly what produces wrong-subject / wrong-gender / tense-flipped
        lines. Mirror of ``NMTTranslator._translate_one_with_context``: up to
        2 prior SOURCE cues are prepended, the target cue rides in a numbered
        tag, and the context translation is DISCARDED. When the tag doesn't
        survive round-trip (or the mini-block exceeds the safe decode chunk),
        the cue falls back to today's isolated translation — output is never
        worse than context-free. ``source_lang``/``target_lang`` are accepted
        for signature parity with NLLB and ignored (the pair is fixed).

        ``context_after`` is unused (Marian decodes left-to-right and the
        prior-cue window is where the referential payoff is).
        """
        if not batch:
            return []
        prior = [c for c in (context_before or []) if (c or "").strip()]
        # Pass 1: build every cue's context mini-block up front. ``prior``
        # grows with the batch's own earlier SOURCE cues, so each mini is
        # byte-identical to what the old serial loop produced — only the CT2
        # calls are batched (2 calls total instead of up to 2 per cue), which
        # is where the multi-core speedup comes from.
        cues: list[str] = []
        minis: dict[int, str] = {}
        for idx, cue in enumerate(batch):
            cue_s = (cue or "").strip()
            cues.append(cue_s)
            if cue_s:
                ctx = prior[-2:]
                if ctx:
                    mini = f"{' '.join(ctx)} {_wrap_numbered_tag(1, cue_s)}"
                    cap = (_MAX_SRC_CHARS_CJK if _looks_cjk(mini)
                           else _MAX_SRC_CHARS_LATIN)
                    if len(mini) <= cap:
                        minis[idx] = mini
            prior.append(cue_s)
        out: list[Optional[str]] = [None] * len(batch)
        # One batched call for all context blocks…
        if minis:
            keys = list(minis)
            try:
                tjs = self.translate_batch([minis[k] for k in keys],
                                           glossary=glossary)
            except Exception as e:
                logger.debug("OpusMT batched context translation failed (%s)", e)
                tjs = [""] * len(keys)
            for k, tj in zip(keys, tjs):
                rec = _parse_numbered_tags(tj or "", 1, 2, 1)
                if rec and (rec[0] or "").strip():
                    out[k] = rec[0]
        # …and one batched call for the isolated fallbacks (tag didn't
        # survive, block over the cap, or no context yet) — output is never
        # worse than context-free, exactly as before.
        rest = [i for i, c in enumerate(cues) if out[i] is None and c]
        if rest:
            tb = self.translate_batch([cues[i] for i in rest],
                                      glossary=glossary)
            for i, t in zip(rest, tb or []):
                out[i] = t if (t or "").strip() else batch[i]
        for i in range(len(batch)):
            if out[i] is None:
                out[i] = batch[i]      # empty cues pass through untouched
        return out

    def translate_batch(
        self,
        texts: list[str],
        glossary: Optional[dict] = None,
    ) -> list[str]:
        """Translate ``texts`` in REAL CT2 batches.

        The old loop sent one example per ``translate_batch`` call, so a
        1,000-cue transcript paid ~1,000 sequential decodes on a single
        worker. All chunks are now encoded up front and decoded in one call
        per decode-length bucket, letting CT2's inter/intra-thread workers
        run examples concurrently. Per-example inputs, beam size and decode
        caps are unchanged (bucket caps only round UP, never truncate), so
        outputs match the serial path; any batch-level failure falls back to
        per-chunk decoding for just that bucket.
        """
        if not texts:
            return []
        if not self._loaded:
            self.load()
        tgt_cjk = self.target.split("-")[0] in ("ja", "zh", "ko", "yue")
        # Pass 1: chunk + encode everything.
        plan: list[Optional[list[int]]] = []   # per text: global chunk ids
        chunk_tokens: list[list[str]] = []
        chunk_maxdec: list[int] = []
        for raw in texts:
            text = (raw or "").strip()
            if not text:
                plan.append(None)
                continue
            try:
                src_cjk = _looks_cjk(text)
                max_chars = _MAX_SRC_CHARS_CJK if src_cjk else _MAX_SRC_CHARS_LATIN
                chunks = _split_for_nmt(text, max_chars, src_cjk) or [text]
                ids = []
                for ch in chunks:
                    tokens = self._tokenizer.encode_as_pieces(ch)
                    ids.append(len(chunk_tokens))
                    chunk_tokens.append(tokens + ["</s>"])
                    chunk_maxdec.append(min(512, max(128, len(tokens) * 3)))
                plan.append(ids)
            except Exception as e:
                logger.warning("Opus-MT: encode failed (%s)", e)
                plan.append(None)
        # Pass 2: decode, bucketed by decode-length cap (rounded up to 128s
        # so a handful of calls covers the batch; a cap can only grow, and
        # the anti-repetition kwargs still guard runaway decodes).
        decoded: dict[int, str] = {}
        buckets: dict[int, list[int]] = {}
        for i, md in enumerate(chunk_maxdec):
            buckets.setdefault(((md + 127) // 128) * 128, []).append(i)
        for cap, idxs in sorted(buckets.items()):
            try:
                outs = _ct2_translate_batch(
                    self._translator,
                    [chunk_tokens[i] for i in idxs],
                    beam_size=5,
                    max_decoding_length=cap,
                    max_batch_size=8,
                )
                for i, o in zip(idxs, outs):
                    decoded[i] = self._tokenizer.decode(o.hypotheses[0])
            except Exception as e:
                logger.warning(
                    "Opus-MT: batched decode failed (%s) — retrying that "
                    "bucket per chunk", e)
                for i in idxs:
                    try:
                        o = _ct2_translate_batch(
                            self._translator, [chunk_tokens[i]],
                            beam_size=5, max_decoding_length=chunk_maxdec[i])
                        decoded[i] = self._tokenizer.decode(o[0].hypotheses[0])
                    except Exception as e2:
                        logger.warning("Opus-MT: translation failed (%s)", e2)
        # Pass 3: reassemble per input text.
        results: list[str] = []
        joiner = "" if tgt_cjk else " "
        for raw, ids in zip(texts, plan):
            if ids is None:
                results.append(raw)
                continue
            if any(i not in decoded for i in ids):
                results.append(raw)     # a chunk failed → keep the original
                continue
            text = (raw or "").strip()
            translated = (joiner.join(
                decoded[i] for i in ids if decoded[i]).strip()) or text
            if glossary:
                translated = apply_glossary(text, translated, glossary)
            results.append(translated)
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
    ``huggingface.co`` must be reachable. For NLLB-200 that is a transient
    full-precision download into a redirected HF cache (~2.5 GB for the 600M,
    ~5.5 GB for the 1.3B); only the int8 CT2 model is kept (~600 MB / ~1.3-1.5
    GB respectively — the fp32 cache is deleted by ``_hf_cache_redirect``).

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
    model pulled during conversion is removed afterwards; only the int8 CT2
    model is kept (~600 MB for the 600M, ~1.3-1.5 GB for the 1.3B default).
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
    _require_free_space(target_dir, _nllb_min_free_bytes(model_id), f"NLLB ({model_id})")
    _int8_hint = "~1.3-1.5 GB" if _is_nllb_1p3b(model_id) else "~600 MB"
    logger.info(
        "NMT: downloading + converting NLLB %s (one-time, %s int8) → %s",
        model_id, _int8_hint, target_dir,
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


def ensure_opus_mt_downloaded(source: str, target: str, *,
                              subdir: str = "opus-mt",
                              model_template: Optional[str] = None) -> str:
    """Download + convert an Opus-MT (or Marian variant) pair to CTranslate2
    (int8) on demand.

    Same semantics as ``ensure_nllb_downloaded`` (disk guard, HF-cache
    cleanup, partial-dir cleanup on failure), plus an LRU cap on how many
    pair dirs are kept (``NMT_MAX_OPUS_PAIRS``). ``subdir`` + ``model_template``
    select a Marian variant (e.g. FuguMT) into its own cache dir; the LRU cap
    only applies to the shared ``opus-mt`` subdir.
    """
    from backend.config import settings as _settings
    from backend.services.language_codes import normalize_lang_code
    src, tgt = normalize_lang_code(source), normalize_lang_code(target)
    target_dir = _opus_dir(src, tgt, subdir)
    if os.path.exists(os.path.join(target_dir, "model.bin")):
        if subdir == "opus-mt":
            _touch_opus_pair(src, tgt)
        return target_dir
    _require_free_space(target_dir, _OPUS_MIN_FREE_BYTES, f"{subdir} {src}-{tgt}")
    model_id = (model_template or _settings.NMT_OPUS_MT_TEMPLATE).format(src=src, tgt=tgt)
    logger.info(
        "NMT: downloading + converting %s %s (one-time) → %s",
        subdir, model_id, target_dir,
    )
    _convert_with_cleanup(
        model_id, target_dir, f"{subdir} {src}-{tgt}",
        tokenizer_files=("source.spm", "target.spm", "vocab.json"),
    )
    if not os.path.exists(os.path.join(target_dir, "source.spm")):
        present = sorted(os.listdir(target_dir)) if os.path.isdir(target_dir) else []
        raise RuntimeError(
            f"{subdir} {src}-{tgt} converted but the SentencePiece tokenizer "
            f"(source.spm) is missing (have: {present})."
        )
    if subdir == "opus-mt":
        _touch_opus_pair(src, tgt)
        # Prune least-recently-used pairs beyond the cap now that a new one landed.
        _enforce_opus_pair_cap()
    return target_dir


def get_marian_variant(source: str, target: str, subdir: str,
                       model_template: str, autodownload: bool = True):
    """Get (downloading if needed) a Marian/Opus-format model under its own
    ``subdir`` + HF ``model_template``.

    Exposes FuguMT (``staka/fugumt-ja-en``) — a JParaCrawl-trained,
    Japanese-specialised translator — through the same CTranslate2 loader as
    Opus-MT. Returns an ``OpusMTTranslator`` ready to translate, or ``None``
    when the variant has no model for the pair and can't be downloaded (the
    caller then falls back to NLLB).
    """
    t = OpusMTTranslator.get(source, target, subdir=subdir, model_template=model_template)
    if t.is_available():
        return t
    if not autodownload:
        return None
    ensure_opus_mt_downloaded(source, target, subdir=subdir, model_template=model_template)
    return t if t.is_available() else None


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
