"""ClipAI Reframer — Audio Intelligence (TACT transcription).

Extracted from clipai_reframer.py (jyoung2000/60) for the Fez engine
transplant. The original Tkinter GUI is not part of this module.
"""

import cv2
import numpy as np
import json
import subprocess
import threading
import os
import sys
import math
import logging
import time as _time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Callable, Dict
from pathlib import Path

from backend.services.reframer_models import (
    ReframeLogger, get_logger, reset_logger, RenderPlan,
    interpolate_x, clamp_x, _face_overlaps_person,
    LedgerBin, CoverageLedger, PerceptionResult, SceneSignals, AdaptiveParams,
)
from backend.config import settings

logger = logging.getLogger("clipai.reframer_audio")


from backend.services.hallucination_filter import (
    is_boilerplate_hallucination as _is_boilerplate_hallucination,
)


# Languages whose native script is CJK — the wrong-script hallucination gate
# only applies to these (Latin-script languages can't distinguish "wrong
# script" from normal text).
_CJK_SCRIPT_LANGS = {"ja", "zh", "ko", "japanese", "chinese", "korean",
                     "yue", "zh-cn", "zh-tw", "ja-jp", "ko-kr"}

# The languages Whisper (large-v3 / large-v3-turbo, and thus the Companion's
# whisper.cpp) can decode. The spoken-language classifier (VoxLingua107) covers
# 107 languages — a few of which Whisper does NOT support — so we only PIN a
# detected language that is in this set; anything else falls back to Whisper's
# own auto-detect rather than forcing a code the decoder would reject.
_WHISPER_LANGUAGE_CODES = frozenset({
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
})


def _wrong_script_for_language(text: str, language) -> bool:
    """True when a cue's script contradicts the PINNED source language.

    With ``language='ja'`` faster-whisper is forced to decode Japanese, so a
    segment that comes back as a long run of pure-Latin prose ("See you next
    time in the video.") is a hallucination over music/silence, not speech.
    Conservative on purpose: 4+ words, ≥60% Latin letters, <10% CJK — short
    interjections, loanwords, and mixed romaji fragments inside real speech
    never trip it. No-op when the language isn't pinned CJK.
    """
    lang = (str(language or "")).strip().lower()
    if lang not in _CJK_SCRIPT_LANGS:
        return False
    t = (text or "").strip()
    if not t or len(t.split()) < 4:
        return False
    latin = cjk = letters = 0
    for ch in t:
        o = ord(ch)
        if ("a" <= ch.lower() <= "z"):
            latin += 1
            letters += 1
        elif (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF
              or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF):
            cjk += 1
            letters += 1
    if letters == 0:
        return False
    return (latin / letters) >= 0.60 and (cjk / letters) < 0.10


def _collapse_repeated_phrases(text: str) -> str:
    """Collapse an immediate repeat-loop WITHIN a single cue's text.

    whisper.cpp — especially the pruned ``large-v3-turbo`` decoder — loops a
    short phrase over music / breathy audio inside ONE segment:
    ``じゃあ いっか。じゃあ いっか。じゃあ いっか。`` or
    ``お疲れ様でした。お疲れ様でした。``. The segment-level dedup
    (``_cross_validate_segments`` / ``_drop_repetition_loops``) acts only
    BETWEEN segments, so a loop packed into one cue survives. This keeps the
    first occurrence of each run of identical sentences. Conservative: a
    single non-repeated sentence is untouched, and a 2× whitespace repeat
    ("very very") is preserved — only 3+ identical tokens in a row collapse.
    """
    if not text or not text.strip():
        return text
    import re as _re
    # 1) Sentence level — drop consecutive identical sentences (CJK 。！？ or
    #    Latin .!? terminated, or a trailing unterminated tail). Whitespace is
    #    ignored for the equality test only; the kept copy is verbatim.
    sentences = _re.findall(r"[^。．！？!?\n]*[。．！？!?\n]|[^。．！？!?\n]+", text)
    if len(sentences) > 1:
        kept = []
        prev_core = None
        for s in sentences:
            core = _re.sub(r"\s+", "", s).rstrip("。．！？!?\n")
            if core and core == prev_core:
                continue
            kept.append(s)
            if core:
                prev_core = core
        text = "".join(kept)
    # 2) Word level — collapse a run of 3+ identical whitespace tokens
    #    ("un un un un" → "un un"); 2× emphasis survives.
    if " " in text:
        toks = text.split(" ")
        out: list = []
        run = 0
        for tk in toks:
            if out and tk == out[-1] and tk.strip():
                run += 1
                if run >= 2:
                    continue
            else:
                run = 0
            out.append(tk)
        text = " ".join(out)
    return text.strip()


def _vocab_bias_kwargs(transcribe_callable, language: str) -> dict:
    """Build the custom-vocabulary biasing kwargs for a Whisper transcribe call.

    Delegates to ``custom_vocabulary.whisper_bias_kwargs``, which feature-
    detects ``hotwords`` support (preferred) and falls back to
    ``initial_prompt`` otherwise. Returns ``{}`` — preserving the current
    no-prompt behaviour exactly — when the feature is disabled or the
    glossary is empty.
    """
    try:
        from backend.services.custom_vocabulary import whisper_bias_kwargs
        return whisper_bias_kwargs(
            transcribe_callable,
            language=language,
            enabled=bool(getattr(settings, "CUSTOM_VOCABULARY_ENABLED", True)),
        )
    except Exception as e:
        logger.warning("Custom vocabulary biasing skipped (%s)", e)
        return {}


def _decoding_kwargs(transcribe_callable, condition_on_previous_text=None) -> dict:
    """Anti-repetition / anti-hallucination decoding kwargs for a Whisper
    transcribe call, filtered to those the installed faster-whisper build
    accepts (mirrors ``_vocab_bias_kwargs``' feature-detection so an older
    build never raises on an unknown kwarg).

    ``condition_on_previous_text`` defaults to the
    ``WHISPER_CONDITION_ON_PREVIOUS_TEXT`` setting; pass an explicit value to
    force it (the gap-fill pass forces ``False`` over music/quiet regions
    regardless of the global default).

    These make the decoder REJECT looped / degenerate output instead of
    emitting it (Task 3):
      * ``condition_on_previous_text`` defaults OFF — it is the primary driver
        of the looped-narration hallucination on music / singing.
      * ``no_repeat_ngram_size`` blocks verbatim n-gram loops within a window.
      * ``compression_ratio_threshold`` + ``log_prob_threshold`` trip Whisper's
        ``temperature`` fallback so a degenerate segment is re-decoded hotter
        rather than kept.
      * ``repetition_penalty`` discourages token-level loops.
    Only kwargs that appear as explicit named parameters of the callable are
    returned (a ``**kwargs`` catch-all does NOT count — same rule as the vocab
    biasing helper), so passing the result is always safe.
    """
    import inspect
    cond = (bool(getattr(settings, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False))
            if condition_on_previous_text is None
            else bool(condition_on_previous_text))
    desired = {
        "condition_on_previous_text": cond,
        "no_repeat_ngram_size": int(getattr(
            settings, "WHISPER_NO_REPEAT_NGRAM_SIZE", 3)),
        "compression_ratio_threshold": float(getattr(
            settings, "WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4)),
        "log_prob_threshold": float(getattr(
            settings, "WHISPER_LOG_PROB_THRESHOLD", -1.0)),
        "repetition_penalty": float(getattr(
            settings, "WHISPER_REPETITION_PENALTY", 1.1)),
        "temperature": tuple(getattr(
            settings, "WHISPER_TEMPERATURE_FALLBACK",
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))),
        # faster-whisper's silence-gap hallucination guard: a segment that
        # follows ≥ this many seconds of silence is skipped when the decode
        # is shaky. Second line of defense stays the TACT filter.
        "hallucination_silence_threshold": float(getattr(
            settings, "WHISPER_HALLUCINATION_SILENCE_S", 2.0)),
    }
    try:
        params = inspect.signature(transcribe_callable).parameters
    except (TypeError, ValueError):
        # Can't introspect — pass nothing extra rather than risk an unknown
        # kwarg. The caller still sets condition_on_previous_text explicitly.
        return {}
    return {k: v for k, v in desired.items() if k in params}


def _detect_language_multiwindow(engine, video_path: str, duration_ms: int):
    """Vote on the spoken language across several spread-out windows.

    A single first-window probe (Whisper's default) mis-reads intros, logos,
    music, and silence — a Japanese video was detected as ``en``, which then
    skipped translation and let Whisper hallucinate English over the JA audio.
    We decode ~30s at a few points across the runtime and SUM the per-window
    language probabilities so the language actually spoken wins.

    Returns ``(language, score, detail)`` or ``(None, 0.0, detail)`` when
    nothing is confident enough to override Whisper's own detection. Best
    effort: any failure yields ``(None, …)`` so the caller keeps auto-detect.
    """
    import tempfile
    dur_s = max(1, int((duration_ms or 0) / 1000))
    # Skip the very start (intro/logo/silence); spread probes across the body.
    if dur_s <= 90:
        starts = [max(0, dur_s // 3)]
    else:
        starts = [int(dur_s * f) for f in (0.15, 0.4, 0.65, 0.85)]
    votes: dict = {}
    probed = 0
    for start in starts:
        wav = tempfile.mktemp(suffix='.wav')
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-ss', str(start), '-t', '30', '-i', video_path,
                 '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', wav],
                capture_output=True, timeout=90)
            if not os.path.exists(wav) or os.path.getsize(wav) < 2000:
                continue
            _seg, _info = engine.transcribe(wav, language=None, vad_filter=True)
            lang = getattr(_info, 'language', None)
            prob = float(getattr(_info, 'language_probability', 0.0) or 0.0)
            probed += 1
            if lang and prob >= 0.5:
                votes[lang] = votes.get(lang, 0.0) + prob
        except Exception:
            continue
        finally:
            try:
                os.remove(wav)
            except Exception:
                pass
    if not votes:
        return None, 0.0, f'no confident language over {probed} window(s)'
    lang, score = max(votes.items(), key=lambda kv: kv[1])
    detail = ', '.join(f'{k}:{v:.2f}' for k, v in sorted(votes.items(), key=lambda kv: -kv[1]))
    return lang, score, detail


def _vad_parameters() -> dict:
    """VAD tuning shared by the batched and sequential transcribe calls.

    min_silence 300ms catches brief intra-sentence pauses; speech_pad
    150ms keeps onsets while tightening cue boundaries (audit Phase 3.2).
    """
    params = {
        "min_silence_duration_ms": int(getattr(
            settings, "WHISPER_VAD_MIN_SILENCE_MS", 300)),
        "speech_pad_ms": int(getattr(
            settings, "WHISPER_VAD_SPEECH_PAD_MS", 150)),
    }
    # Silero speech-onset probability threshold. The 0.5 default drops soft /
    # whispered / distant speech — a common cause of sparse transcripts. Wiring
    # WHISPER_VAD_ONSET (previously dead config) lets a lower value recover it.
    try:
        onset = float(getattr(settings, "WHISPER_VAD_ONSET", 0) or 0)
        if 0.0 < onset < 1.0:
            params["threshold"] = onset
    except Exception:
        pass
    return params


def _beam_size() -> int:
    """Honor WHISPER_BEAM_SIZE (was dead config — the transcribe calls hardcoded
    5). Higher beams (8-10) capture a few % more words at more VRAM/time."""
    try:
        return max(1, int(getattr(settings, "WHISPER_BEAM_SIZE", 5)))
    except Exception:
        return 5


def _wait_for_sibling_audio(final_path: str, part_path: str,
                            duration_ms: int = 0,
                            grace_s: float = 20.0,
                            poll_s: float = 2.0,
                            sleep=None) -> bool:
    """Wait for the pipeline's shared ``audio.wav`` to finish extracting.

    The pipeline's ``extract_audio`` writes atomically: ffmpeg streams into
    ``audio.wav.part.wav`` and the final path appears only via os.replace()
    when complete. So the transcription thread can safely WAIT instead of
    re-running the whole preconditioning chain (highpass+afftdn+loudnorm —
    minutes of duplicated CPU on a long video):

      * final file exists            → ready (True)
      * ``.part.wav`` marker exists  → extraction in flight; poll until the
                                       marker disappears (bounded by a
                                       duration-scaled cap)
      * neither appears in ``grace_s`` → no pipeline extraction is running
                                       (standalone engine use) → False

    Returns True when the final file is ready to reuse. Never raises.
    """
    _sleep = sleep or _time.sleep
    dur_s = max(0, int((duration_ms or 0) / 1000))
    # Cap mirrors extract_audio's own duration-scaled timeout, plus slack.
    cap_s = max(600.0, dur_s + 180.0)
    waited = 0.0
    saw_part = False
    try:
        while waited <= cap_s:
            if os.path.isfile(final_path) and os.path.getsize(final_path) > 1024:
                return True
            if os.path.exists(part_path):
                saw_part = True
            elif saw_part:
                # Marker vanished but no final file → extraction failed or was
                # cleaned up; fall back to self-extraction immediately.
                return (os.path.isfile(final_path)
                        and os.path.getsize(final_path) > 1024)
            elif waited >= grace_s:
                # Never saw an extraction start — nobody is producing the file.
                return False
            _sleep(poll_s)
            waited += poll_s
    except Exception:
        pass
    return os.path.isfile(final_path) and os.path.getsize(final_path) > 1024


def _gap_boost_af_args() -> list:
    """FFmpeg ``-af`` arguments that lift quiet / off-mic speech in a gap
    slice before the retry decode: high-pass out the rumble, denoise, then
    speechnorm the level up. The gaps being recovered are exactly the spans
    the main decode already dropped as too faint, so a boosted retry is what
    gives them a real second chance. Returns ``[]`` when
    ``SPEECH_GAP_BOOST_ENABLED`` is off (slice extracted unchanged, the
    pre-boost behavior)."""
    if not bool(getattr(settings, "SPEECH_GAP_BOOST_ENABLED", True)):
        return []
    chain = (getattr(settings, "SPEECH_GAP_BOOST_FILTER", "") or "").strip()
    if not chain:
        chain = "highpass=f=80,afftdn=nf=-25,speechnorm=e=6.25:r=0.0001:l=1"
    return ["-af", chain]


def _remote_tuning_fields() -> dict:
    """Decode-tuning form fields for the remote transcription POST — the SAME
    tuned values the local path computes (``_vad_parameters()``,
    ``WHISPER_NO_SPEECH_THRESHOLD``, the ``_decoding_kwargs()`` thresholds,
    ``_beam_size()``), so the Companion sidecar no longer runs faster-whisper
    defaults (Silero threshold 0.5, min_silence 2000 ms, no_speech 0.6,
    condition_on_previous_text=True) that drop quiet speech and skip long
    stretches the local path would have kept.

    The sidecar treats every field as optional, and other OpenAI-compatible
    servers (speaches, whisper.cpp) ignore unknown multipart fields, so
    sending them is backward compatible. ``WHISPER_REMOTE_SEND_TUNING=False``
    turns them off for a strict server that rejects extra fields.
    """
    if not bool(getattr(settings, "WHISPER_REMOTE_SEND_TUNING", True)):
        return {}
    try:
        vad = _vad_parameters()
        fields = {
            "beam_size": str(_beam_size()),
            "vad_min_silence_ms": str(int(vad.get("min_silence_duration_ms", 300))),
            "vad_speech_pad_ms": str(int(vad.get("speech_pad_ms", 150))),
            "no_speech_threshold": str(float(getattr(
                settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.4))),
            "condition_on_previous_text": (
                "true" if bool(getattr(
                    settings, "WHISPER_CONDITION_ON_PREVIOUS_TEXT", False))
                else "false"),
            "no_repeat_ngram_size": str(int(getattr(
                settings, "WHISPER_NO_REPEAT_NGRAM_SIZE", 3))),
            "log_prob_threshold": str(float(getattr(
                settings, "WHISPER_LOG_PROB_THRESHOLD", -1.0))),
            "compression_ratio_threshold": str(float(getattr(
                settings, "WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4))),
            "hallucination_silence_threshold": str(float(getattr(
                settings, "WHISPER_HALLUCINATION_SILENCE_S", 2.0))),
        }
        if "threshold" in vad:
            fields["vad_threshold"] = str(float(vad["threshold"]))
        return fields
    except Exception as e:
        logger.warning("Remote tuning fields skipped (%s)", e)
        return {}


def _words_degenerate(words: list, start_sec: float, end_sec: float) -> bool:
    """True when a segment's word timestamps are unusable — missing for a
    multi-word text, non-monotonic, or collapsed to zero-width spans.
    Batched inference on some faster-whisper builds mis-times words; these
    segments get re-decoded sequentially.
    """
    if not words:
        return False  # no words at all is handled separately (may be legit)
    prev_end = None
    zero_width = 0
    for w in words:
        ws, we = w.get('start'), w.get('end')
        if ws is None or we is None or we < ws:
            return True
        if prev_end is not None and ws < prev_end - 0.25:
            return True  # went backwards past tolerance
        if we - ws < 1e-4:
            zero_width += 1
        prev_end = we
    return zero_width >= max(2, len(words) // 2)


def _cross_validate_segments(segments: list) -> list:
    """Remove cross-segment artefacts the per-segment TACT filter misses.

    Drops:
      - Exact text duplicates of the immediately preceding segment.
      - Segments that start before the previous segment ended (temporal
        overlap by more than 100 ms).
      - Segments with >80% word overlap with the previous segment (a
        common Whisper hallucination pattern on noisy audio).

    Lightweight pure-Python pass — no feature flag, only removes clearly
    invalid output.
    """
    if not segments:
        return segments
    cleaned: list = []
    prev_text = ""
    prev_end = 0.0
    for seg in segments:
        text = (seg.get("text", "") or "").strip()
        # Collapse an in-cue repeat-loop ("じゃあ いっか。じゃあ いっか。じゃあ いっか。")
        # BEFORE the between-segment dedup, so the stored cue and every
        # downstream stage (polish / formatter / SRT) get the clean text.
        _collapsed = _collapse_repeated_phrases(text)
        if _collapsed != text:
            text = _collapsed
            seg["text"] = _collapsed
        start = seg.get("start_sec", seg.get("start", 0)) or 0
        end = seg.get("end_sec", seg.get("end", 0)) or 0

        if text and text == prev_text:
            continue
        if cleaned and start < prev_end - 0.1:
            continue
        if prev_text and text:
            # Strip punctuation so "fine," and "fine" compare equal.
            import string as _str
            _tbl = str.maketrans("", "", _str.punctuation)
            prev_words = set(prev_text.lower().translate(_tbl).split())
            curr_words = set(text.lower().translate(_tbl).split())
            if prev_words and len(prev_words & curr_words) / len(prev_words) > 0.8:
                continue

        cleaned.append(seg)
        prev_text = text
        prev_end = float(end) or prev_end

    dropped = len(segments) - len(cleaned)

    # Second pass: collapse NON-adjacent near-duplicates — segments that
    # overlap in time and carry similar (not just identical) text. The
    # adjacency-only logic above misses re-transcriptions that land a few
    # hundred ms apart with the same line (the repeated proper-noun cue the
    # gap-fill pass produces over music). Operates on start_sec/end_sec keys.
    from backend.services.transcript_dedup import collapse_overlapping_duplicates
    cleaned, overlap_dropped = collapse_overlapping_duplicates(
        cleaned, start_key="start_sec", end_key="end_sec")
    dropped += overlap_dropped

    if dropped > 0:
        logger.info(
            "cross-segment validation removed %d duplicate/overlapping segments "
            "(%d adjacent, %d overlapping near-duplicate)",
            dropped, dropped - overlap_dropped, overlap_dropped,
        )
    return cleaned


def _drop_repetition_loops(segments: list) -> list:
    """Remove Whisper repetition-loop hallucinations.

    When the gap-fill pass re-transcribes music / quiet regions with VAD off,
    Whisper loops and emits the SAME text over and over, scattered across the
    timeline (so the adjacent-only ``_cross_validate_segments`` misses them).
    Real dialogue almost never repeats verbatim many times across an episode,
    so an exact-text segment that recurs beyond a small cap is a hallucination.

    Keeps the earliest occurrences (1 for long lines, up to 3 for short
    interjections like "了解") and drops the rest. Operates on the segment
    dicts (``text`` + ``start_sec``/``end_sec``); order-preserving.
    """
    from backend.services.transcript_dedup import drop_repetition_loops
    out, dropped = drop_repetition_loops(segments, text_key="text")
    if dropped > 0:
        logger.info(
            "repetition-loop filter removed %d duplicated hallucination segment(s)",
            dropped,
        )
    return out


# ── Remote Whisper (OpenAI-compatible server, e.g. the GPU Companion) ──────

_REMOTE_HEALTH_CACHE = {"checked_at": 0.0, "healthy": False, "url": ""}
_REMOTE_HEALTH_TTL_S = 30.0


def _companion_whisper() -> tuple[str, str]:
    """(base_url, token) of the paired GPU Companion from the Ollama host
    registry — the SAME GPU that serves Ollama also does transcription, so
    remote Whisper needs no separate URL/token. Empty base when none paired."""
    try:
        from backend.services import ollama_registry as _oreg
        h = _oreg.companion_host()
        if h is not None:
            return _oreg.companion_base(h), (h.token or "").strip()
    except Exception:
        pass
    return "", ""


def remote_whisper_configured() -> bool:
    """True when transcription can be offloaded remotely: a paired Companion in
    the Ollama host registry, OR an explicit third-party WHISPER_REMOTE_URL."""
    base, _ = _companion_whisper()
    if base:
        return True
    return bool((getattr(settings, "WHISPER_REMOTE_URL", "") or "").strip())


def _remote_whisper_base() -> str:
    # Prefer the paired Companion (its GPU already serves Ollama); fall back to
    # a manually-set third-party WHISPER_REMOTE_URL.
    base, _ = _companion_whisper()
    url = (base or (getattr(settings, "WHISPER_REMOTE_URL", "") or "").strip()).rstrip("/")
    if url and "://" not in url:
        url = f"http://{url}"
    # Accept either the bare server root or a URL that already ends in /v1.
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _remote_whisper_token() -> str:
    """Bearer token for remote Whisper — the Companion host's token when paired,
    else the third-party WHISPER_REMOTE_API_KEY."""
    _, token = _companion_whisper()
    if token:
        return token
    return (getattr(settings, "WHISPER_REMOTE_API_KEY", "") or "").strip()


_translate_cap_cache: dict = {"base": "", "ok": False, "at": 0.0}


def remote_whisper_supports_translate() -> bool:
    """True when the remote whisper host advertises the audio→English translate
    task on ``/v1/health`` (``whisper_translate: true``). Cached ~30 s. Defaults
    to False (safe: callers keep the local translate path) so a host that can't
    translate — or an old Companion that doesn't advertise it — never silently
    produces a worse result."""
    import time as _t
    base = _remote_whisper_base()
    if not base:
        return False
    now = _t.monotonic()
    if (_translate_cap_cache["base"] == base
            and now - _translate_cap_cache["at"] < 30.0):
        return bool(_translate_cap_cache["ok"])
    ok = False
    try:
        import httpx as _httpx
        headers = {}
        tok = _remote_whisper_token()
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        r = _httpx.get(f"{base}/v1/health", headers=headers, timeout=4.0)
        if r.status_code == 200:
            ok = bool((r.json() or {}).get("whisper_translate", False))
    except Exception:
        ok = False
    _translate_cap_cache.update({"base": base, "ok": ok, "at": now})
    return ok


def remote_whisper_healthy(force: bool = False) -> bool:
    """Cheap health probe of the remote transcription server (cached ~30 s).

    Servers differ: the GPU Companion serves ``/v1/health``, speaches serves
    ``/health``, whisper.cpp answers on ``/``. Any HTTP response (even 404)
    proves the server is up — connection errors are the only failure.
    """
    if not remote_whisper_configured():
        return False
    base = _remote_whisper_base()
    now = _time.monotonic()
    if (not force and _REMOTE_HEALTH_CACHE["url"] == base
            and now - _REMOTE_HEALTH_CACHE["checked_at"] < _REMOTE_HEALTH_TTL_S):
        return _REMOTE_HEALTH_CACHE["healthy"]
    healthy = False
    try:
        import httpx
        headers = {}
        key = _remote_whisper_token()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=3.0, headers=headers) as client:
            # Prefer /v1/health: a ClipAI Companion reports whether its Whisper
            # backend is actually present, so we never route transcription to a
            # Companion that has no sidecar (which would just time out and fall
            # back to local). Its verdict is authoritative.
            try:
                r = client.get(f"{base}/v1/health")
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except Exception:
                        data = {}
                    if isinstance(data, dict) and data.get("service") == "clipai-gpu-companion":
                        healthy = bool((data.get("backends") or {}).get("whisper"))
                        _REMOTE_HEALTH_CACHE.update(
                            {"checked_at": now, "healthy": healthy, "url": base})
                        return healthy
                    healthy = True  # some other server exposes /v1/health
            except httpx.HTTPError:
                pass
            # Third-party servers (speaches /health, whisper.cpp on /): any HTTP
            # response proves the server is up — connection errors are failure.
            if not healthy:
                for path in ("/health", "/"):
                    try:
                        client.get(f"{base}{path}")
                        healthy = True
                        break
                    except httpx.HTTPError:
                        continue
    except Exception:
        healthy = False
    _REMOTE_HEALTH_CACHE.update(
        {"checked_at": now, "healthy": healthy, "url": base})
    return healthy


def remote_whisper_pick_model(language: Optional[str]) -> str:
    """Model to request from the remote server.

    ``WHISPER_REMOTE_MODEL`` wins when set; a user-pinned local model
    (WHISPER_MODEL_USER_SET) is honored next; otherwise the auto ladder
    picks by accuracy. With ``WHISPER_REMOTE_PREFER_ACCURACY`` (the
    default) every job — English/auto included — requests full large-v3,
    since the Companion GPU can afford it. Set that flag off to opt into
    speed: English/auto then drop to the pruned large-v3-turbo decoder
    while pinned non-English stays on large-v3.
    """
    configured = (getattr(settings, "WHISPER_REMOTE_MODEL", "") or "").strip()
    if configured:
        return configured
    if bool(getattr(settings, "WHISPER_MODEL_USER_SET", False)):
        pinned = (getattr(settings, "WHISPER_MODEL", "") or "").strip()
        if pinned:
            return pinned
    lang = (language or "auto").strip().lower()
    if lang in ("", "auto", "en", "english"):
        # The Companion GPU can afford full large-v3 for the accuracy win; only
        # drop to the pruned turbo decoder when the user opts for speed.
        if bool(getattr(settings, "WHISPER_REMOTE_PREFER_ACCURACY", True)):
            return "large-v3"
        return "large-v3-turbo"
    return "large-v3"


def remote_whisper_warm() -> None:
    """Ask the Companion to start its whisper sidecar NOW, in the background.

    Called the moment remote Whisper is SELECTED for a job — several minutes
    before the WAV upload — so the ~35 s of server spawn + model load happens
    while frame extraction / language ID run, instead of delaying the decode.
    Fire-and-forget on a daemon thread (this runs in sync perceiver code);
    best-effort in every direction: a 404 from an older Companion, pause
    (503), or any transport error changes nothing — the decode request's own
    ensure_running remains the fallback.
    """
    if not remote_whisper_configured():
        return

    def _fire():
        try:
            import httpx
            headers = {}
            token = _remote_whisper_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            _model = remote_whisper_pick_model(None)
            if _model:
                headers["X-ClipAI-Whisper-Model"] = _model
            r = httpx.post(f"{_remote_whisper_base()}/v1/sidecar/warm",
                           headers=headers, timeout=5.0)
            if r.status_code in (200, 202):
                logger.info(
                    "Companion whisper sidecar pre-warm requested "
                    "(model=%s) — loads while extraction runs", _model)
        except Exception:
            pass

    import threading
    threading.Thread(target=_fire, name="clipai-whisper-warm",
                     daemon=True).start()


def remote_whisper_release() -> bool:
    """Ask the Companion to shut its whisper sidecar down NOW (free VRAM).

    Called right after a job's transcription stage completes: the next
    pipeline phase (translation/polish/SEO) runs Ollama on the SAME Companion
    GPU, and a resident whisper server (~3-4 GB) can force the LLM to spill
    layers to CPU on an 8 GB card — several-times-slower generation until the
    Companion's 15-minute idle reaper fires. Eager release costs nothing:
    any later whisper request (native-translate timing pass, a new job)
    cold-restarts the sidecar automatically.

    Best-effort in every direction: unknown route on an older Companion
    (404), a decode in flight (409), or any transport error just returns
    False and nothing changes — the idle reaper remains the backstop.
    """
    if not remote_whisper_configured():
        return False
    try:
        import httpx
        headers = {}
        token = _remote_whisper_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        r = httpx.post(f"{_remote_whisper_base()}/v1/sidecar/release",
                       headers=headers, timeout=5.0)
        if r.status_code == 200:
            logger.info("Companion whisper sidecar released (VRAM freed for the LLM phase)")
            return True
        return False
    except Exception:
        return False


class RemoteWhisperEngine:
    """OpenAI-compatible remote transcription client.

    POSTs the already-extracted WAV (never the source video) to
    ``{WHISPER_REMOTE_URL}/v1/audio/transcriptions`` with
    ``response_format=verbose_json`` + word/segment timestamp
    granularities and maps the response into the exact segment schema the
    local faster-whisper path produces, so everything downstream (TACT
    filters → forced alignment → polish → SRT/ASS) is byte-compatible.

    Works with the GPU Companion, speaches, whisper-asr-webservice, and
    whisper.cpp server (``--inference-path /v1/audio/transcriptions``).
    """

    # Base request timeout. The REAL per-request timeout scales with the
    # audio duration (see _timeout_for) — a fixed 600 s ceiling silently
    # killed every whole-file decode longer than ~40 min of audio: the
    # observed 128-min video needs ~25-30 min on turbo (more on full
    # large-v3), the client abandoned the request at 10 min, the retries
    # collided with the Companion still decoding the abandoned job, and the
    # whole pipeline continued WITHOUT a transcript.
    TIMEOUT_S = 600

    def __init__(self, model: str = ""):
        self.base = _remote_whisper_base()
        self.model = model
        self.api_key = _remote_whisper_token()

    @staticmethod
    def _timeout_for(wav_bytes: bytes) -> float:
        """Request timeout scaled to the payload's audio duration.

        16 kHz mono s16 PCM ⇒ 32,000 bytes/second. Budget one×realtime of
        decode per audio second (full large-v3 beam-5 runs ~0.5-0.7×RT on a
        mid-range GPU; turbo ~0.25) plus a base for upload + model load,
        clamped to a hard cap. Every knob has a config override."""
        base = float(getattr(settings, "WHISPER_REMOTE_TIMEOUT_BASE_S", 600.0))
        per_min = float(getattr(settings, "WHISPER_REMOTE_TIMEOUT_PER_AUDIO_MIN_S", 60.0))
        cap = float(getattr(settings, "WHISPER_REMOTE_TIMEOUT_MAX_S", 10800.0))
        audio_min = len(wav_bytes) / 32000.0 / 60.0
        return max(base, min(cap, base + audio_min * per_min))

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            from backend.services.request_context import clipai_headers
            headers.update(clipai_headers())
        except Exception:
            pass
        return headers

    # Reliability knobs for the LAN upload to the Companion.
    UPLOAD_RETRIES = 3          # transient-failure attempts per payload
    CHUNK_SECONDS = 600         # fallback chunk size (10 min) when a whole upload fails
    MIN_CHUNK_SECONDS = 120     # don't chunk finer than this

    def _post_wav(self, wav_bytes: bytes, filename: str, data: dict,
                  headers: dict, timeout_s: Optional[float] = None) -> tuple:
        """ONE upload attempt with an EXACT Content-Length — the body is bytes
        in memory, so it can never race a file that's still being written (the
        "Too much data for declared Content-Length" failure). Returns
        (ok, payload, retryable, detail, retry_after_s)."""
        import httpx
        try:
            files = {"file": (filename, wav_bytes, "audio/wav")}
            resp = httpx.post(
                f"{self.base}/v1/audio/transcriptions",
                headers=headers, data=data, files=files,
                timeout=timeout_s if timeout_s else self.TIMEOUT_S)
        except Exception as e:  # network / connection reset / timeout
            return (False, None, True, f"{type(e).__name__}: {str(e)[:120]}", 0.0)
        if resp.status_code == 200:
            try:
                return (True, resp.json(), False, "", 0.0)
            except Exception as e:
                return (False, None, True, f"bad JSON: {str(e)[:80]}", 0.0)
        if resp.status_code == 503:  # busy / paused — honor Retry-After
            try:
                wait = min(30.0, max(0.5, float(resp.headers.get("Retry-After", "5"))))
            except (TypeError, ValueError):
                wait = 5.0
            return (False, None, True, "busy/paused (503)", wait)
        if resp.status_code in (401, 403, 404):  # auth / no endpoint — don't retry
            return (False, None, False, f"HTTP {resp.status_code}", 0.0)
        # 413 (too large), 5xx, etc. — retryable; chunking may get it through.
        return (False, None, True, f"HTTP {resp.status_code}: {resp.text[:120]}", 0.0)

    def _transcribe_payload(self, wav_bytes: bytes, filename: str, data: dict,
                            headers: dict, language, model,
                            offset_s: float = 0.0) -> Optional[dict]:
        """Upload one WAV payload with retries + backoff; map to the local
        schema and shift timestamps by ``offset_s`` (for chunked uploads).

        The request timeout scales with the payload's audio duration, and a
        503-busy answer waits WITHOUT consuming an upload attempt (bounded by
        its own patience budget): the GPU finishing another decode is not an
        upload failure, and burning the 3 attempts on instant 503s is exactly
        how a long video ended transcript-less."""
        from backend.services.cloud_transcription import _map_verbose_json
        last = ""
        timeout_s = self._timeout_for(wav_bytes)
        busy_budget_s = float(getattr(settings, "WHISPER_REMOTE_BUSY_WAIT_S", 600.0))
        busy_waited = 0.0
        attempt = 0
        while attempt < self.UPLOAD_RETRIES:
            attempt += 1
            ok, payload, retryable, detail, retry_after = self._post_wav(
                wav_bytes, filename, data, headers, timeout_s=timeout_s)
            if ok:
                segs = _map_verbose_json(payload) or []
                if offset_s:
                    for s in segs:
                        for k in ("start", "end"):
                            if isinstance(s.get(k), (int, float)):
                                s[k] = s[k] + offset_s
                        for w in (s.get("words") or []):
                            for k in ("start", "end"):
                                if isinstance(w.get(k), (int, float)):
                                    w[k] = w[k] + offset_s
                return {"segments": segs,
                        "language": payload.get("language", language or "unknown")}
            last = detail
            if not retryable:
                break
            if "503" in detail and busy_waited < busy_budget_s:
                # Busy is patience, not failure: the GPU is finishing another
                # decode (possibly one WE abandoned on a timeout). Wait it out
                # on its own budget and DON'T consume an upload attempt.
                wait = retry_after if retry_after > 0 else 10.0
                busy_waited += wait
                attempt -= 1
                logger.info(
                    "Remote Whisper busy (503) — waiting %.0fs (%.0f/%.0fs busy "
                    "budget) without consuming a retry", wait, busy_waited,
                    busy_budget_s)
                _time.sleep(wait)
                continue
            if attempt < self.UPLOAD_RETRIES:
                wait = retry_after if retry_after > 0 else min(8.0, 1.5 * attempt)
                logger.info("Remote Whisper upload attempt %d/%d failed (%s) — "
                            "retrying in %.1fs", attempt, self.UPLOAD_RETRIES, detail, wait)
                _time.sleep(wait)
        logger.warning("Remote Whisper upload failed after %d attempts: %s",
                       self.UPLOAD_RETRIES, last)
        return None

    def _iter_wav_chunks(self, audio_path: str, chunk_seconds: int):
        """Yield (offset_s, wav_bytes, label) time slices of a PCM WAV, each a
        complete standalone WAV. Raises if the file isn't a readable PCM WAV."""
        import wave, io
        with wave.open(audio_path, "rb") as w:
            nch, sw, fr = w.getnchannels(), w.getsampwidth(), w.getframerate()
            nframes = w.getnframes()
            step = max(1, int(chunk_seconds * fr))
            pos = 0
            while pos < nframes:
                w.setpos(pos)
                frames = w.readframes(min(step, nframes - pos))
                if not frames:
                    break
                buf = io.BytesIO()
                with wave.open(buf, "wb") as out:
                    out.setnchannels(nch)
                    out.setsampwidth(sw)
                    out.setframerate(fr)
                    out.writeframes(frames)
                yield (pos / float(fr), buf.getvalue(),
                       f"{pos // fr}s-{(pos + step) // fr}s")
                pos += step

    def _transcribe_chunked(self, audio_path: str, data: dict, headers: dict,
                            language, model) -> Optional[dict]:
        """Fallback when a whole-file upload keeps failing: split the WAV into
        smaller time chunks the Companion can accept, transcribe each (retried),
        and merge with correct timestamps. Reliability path for any speed
        setting / flaky link. Quality impact is negligible — Whisper already
        works in ~30 s internal windows, so a 10-min chunk boundary only touches
        one edge window."""
        try:
            base = os.path.basename(audio_path)
            merged: list = []
            lang_seen = language or "unknown"
            n = 0
            for offset_s, wav_bytes, label in self._iter_wav_chunks(
                    audio_path, self.CHUNK_SECONDS):
                n += 1
                logger.info("Remote Whisper chunk %d (%s, %.1f MB) → Companion",
                            n, label, len(wav_bytes) / (1024 * 1024))
                part = self._transcribe_payload(
                    wav_bytes, f"chunk{n}_{base}", data, headers,
                    language, model, offset_s=offset_s)
                if part is None:
                    logger.warning("Remote Whisper chunk %d failed after retries "
                                   "— aborting chunked upload", n)
                    return None
                merged.extend(part["segments"])
                if part.get("language") and part["language"] != "unknown":
                    lang_seen = part["language"]
            if not merged:
                return None
            logger.info("Remote Whisper chunked upload OK: %d chunks → %d segments",
                        n, len(merged))
            return {"segments": merged, "language": lang_seen}
        except Exception as e:
            logger.warning("Remote Whisper chunked upload could not run (%s) — "
                           "falling back to local", e)
            return None

    def transcribe_wav(self, audio_path: str,
                       language: Optional[str] = None,
                       translate: bool = False) -> Optional[dict]:
        """Returns ``{'segments', 'language', 'provider', 'model'}`` in the
        local schema (same contract as ``cloud_transcription.transcribe_cloud``)
        or ``None`` on any failure — the caller falls back to local.

        ``translate=True`` requests Whisper's native audio→English translate
        task (whisper.cpp honors the ``translate`` form field) so the primary
        whisper GPU — the paired Companion — can produce the English timing
        reference instead of the slow local card.

        Reliable transfer to the Companion: the body is sent with an EXACT
        Content-Length (bytes snapshot, never a growing file handle), transient
        failures are retried with backoff, and if the whole upload still can't
        get through, it's re-sent as smaller time chunks that are merged back."""
        from backend.services.cloud_transcription import _vocab_prompt

        model = self.model or remote_whisper_pick_model(language)
        data = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
        }
        # Decode-tuning parity with the local path: the Companion sidecar
        # honors these; other servers ignore the extra multipart fields.
        data.update(_remote_tuning_fields())
        if translate:
            # whisper.cpp server reads this to run task=translate (→ English).
            data["translate"] = "true"
        if language and language not in ("auto", ""):
            data["language"] = language
        prompt = _vocab_prompt(language or "en")
        if prompt:
            data["prompt"] = prompt

        # Sync the SELECTED model to the Companion so it loads the same family on
        # its GPU (the Companion picks its whisper.cpp model from this header).
        headers = self._headers()
        if model:
            headers["X-ClipAI-Whisper-Model"] = model

        # Snapshot the file to bytes ONCE — an exact Content-Length that can't
        # race a still-being-written WAV.
        try:
            with open(audio_path, "rb") as fh:
                wav_bytes = fh.read()
        except Exception as e:
            logger.warning("Remote Whisper could not read %s (%s) — local fallback",
                           audio_path, e)
            return None

        logger.info(
            "Remote Whisper upload: %.1f MB (%.0f min audio) — request timeout %.0fs",
            len(wav_bytes) / (1024 * 1024), len(wav_bytes) / 32000.0 / 60.0,
            self._timeout_for(wav_bytes))
        result = self._transcribe_payload(
            wav_bytes, os.path.basename(audio_path), data, headers, language, model)
        if result is None:
            # Whole-file upload failed after retries — try smaller chunks.
            logger.info("Remote Whisper whole-file upload failed — retrying in "
                        "%d s chunks", self.CHUNK_SECONDS)
            result = self._transcribe_chunked(audio_path, data, headers, language, model)
        if result is None:
            return None

        segments = result["segments"]
        if not segments:
            logger.warning("Remote Whisper returned no segments — falling back to local")
            return None
        for entry in segments:
            entry["source"] = "remote"
        logger.info("Remote Whisper (%s): %d segments, language=%s, model=%s",
                    self.base, len(segments), result.get("language", "auto"), model)
        return {
            "segments": segments,
            "language": result.get("language", language or "unknown"),
            "provider": "remote",
            "model": model,
        }


class AudioIntelligence:
    """
    Whisper-based audio analysis for reframing intelligence.

    Model selection (speed vs accuracy tradeoff):
      - 'base'           : ~10x realtime on CPU, ~50x on GPU. Good enough for speech detection.
      - 'small'           : ~5x realtime on CPU, ~30x on GPU. Better accuracy.
      - 'large-v3-turbo'  : ~1x realtime on CPU, ~10x on GPU. Best accuracy, slowest.

    Default is 'small' — good balance of speed and accuracy for reframing.
    The model is cached between runs to avoid re-downloading.

    Requires: pip install faster-whisper
    """

    # Class-level model cache — survives across Perceiver instances
    _cached_engine = None
    _cached_model_name = None
    _cached_device = None
    # Sticky record of the model/device that ACTUALLY loaded most recently.
    # Unlike ``_cached_model_name`` (which ``_release_whisper_vram`` nulls when
    # it frees the engine after a job), these survive the VRAM release so the
    # Settings page can always report the real effective model that ran — even
    # after the engine has been unloaded. Set wherever a real load succeeds.
    _last_loaded_model_name = None
    _last_loaded_device = None

    # Load-footprint (GB) per (model, compute_type). Mirrors the ``_vram_load_gb``
    # table used at load time; kept at class scope so the diagnostics panel can
    # estimate Whisper's GPU residency. faster-whisper is CTranslate2, whose VRAM
    # lives OUTSIDE torch's allocator — so torch.cuda.memory_reserved() can't see
    # it and the live gauge needs either nvidia-smi (measured) or this estimate.
    _VRAM_LOAD_GB = {
        ('large', 'float16'): 3.0, ('large', 'int8_float16'): 1.6,
        ('large-v2', 'float16'): 3.0, ('large-v2', 'int8_float16'): 1.6,
        ('large-v3', 'float16'): 3.0, ('large-v3', 'int8_float16'): 1.6,
        ('large-v3-turbo', 'float16'): 1.8, ('large-v3-turbo', 'int8_float16'): 1.0,
        ('distil-large-v3', 'float16'): 1.6, ('distil-large-v3', 'int8_float16'): 0.9,
        ('distil-large-v3.5', 'float16'): 1.6, ('distil-large-v3.5', 'int8_float16'): 0.9,
        ('kotoba-tech/kotoba-whisper-v2.0-faster', 'float16'): 1.8,
        ('kotoba-tech/kotoba-whisper-v2.0-faster', 'int8_float16'): 1.0,
        ('medium', 'float16'): 1.6, ('medium', 'int8_float16'): 0.85,
        ('small', 'float16'): 1.0, ('small', 'int8_float16'): 0.55,
        ('base', 'float16'): 0.4, ('base', 'int8_float16'): 0.25,
        ('tiny', 'float16'): 0.2,
    }

    @classmethod
    def _record_loaded(cls, model_name, device):
        """Stamp the model/device that just loaded (sticky, for the GUI)."""
        cls._last_loaded_model_name = model_name
        cls._last_loaded_device = device

    @classmethod
    def gpu_residency_estimate(cls) -> dict:
        """Best-effort Whisper GPU residency for the live VRAM gauge.

        Returns ``{loaded, model, device, on_gpu, est_bytes}``. ``est_bytes`` is
        a load-footprint estimate (from ``_VRAM_LOAD_GB``) used as a fallback
        when nvidia-smi can't be reached from the app container (so CTranslate2
        Whisper VRAM would otherwise be invisible). ``loaded`` reflects whether
        an engine is currently held in memory (``_cached_engine``); the sticky
        ``_last_loaded_*`` fields are used only for labelling when idle.
        """
        engine_loaded = cls._cached_engine is not None
        model = cls._cached_model_name or cls._last_loaded_model_name or ""
        device = (cls._cached_device or cls._last_loaded_device or "")
        on_gpu = engine_loaded and str(device).startswith("cuda")
        est_bytes = 0
        if on_gpu:
            compute = device.split("_", 1)[1] if "_" in device else "float16"
            gb = cls._VRAM_LOAD_GB.get((model, compute))
            if gb is None:
                # Unknown pairing — bias by whether weights are int8-quantized.
                gb = 1.6 if "int8" in compute else 3.0
            est_bytes = int(gb * 1024 ** 3)
        return {
            "loaded": engine_loaded,
            "model": model,
            "device": device,
            "on_gpu": on_gpu,
            "est_bytes": est_bytes,
        }

    @classmethod
    def invalidate_cache(cls, reason: str = ""):
        """Drop the cached engine so the next job loads a freshly-selected
        model. Safe to call from anywhere (e.g. when the user changes
        WHISPER_MODEL in Settings) — frees the old engine's VRAM too. Leaves
        the sticky ``_last_loaded_*`` record intact for display."""
        had = cls._cached_model_name
        cls._cached_engine = None
        cls._cached_model_name = None
        cls._cached_device = None
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if had:
            logger.info("AudioIntelligence cache invalidated (was '%s')%s",
                        had, f" — {reason}" if reason else "")

    def __init__(self, model_name: str = 'small'):
        self.available = False
        self.engine = None
        self.model_name = model_name
        # The model the caller asked for, preserved across any VRAM downgrade
        # so the pipeline can report requested-vs-effective accurately (Task 4).
        self.requested_model_name = model_name
        self.device_used = 'unknown'
        # try_load() picks an appropriate batched-inference batch size
        # based on the GPU's total VRAM (4 → batch=4, larger → batch=16).
        # Surfaced as ``self._batch_size`` so ``transcribe()`` uses it.
        self._batch_size = 16
        # Set when a remote decode ANSWERED but was rejected (filtered to
        # nothing / degenerate coverage). Re-decoding the same audio on the
        # same server is deterministic garbage, so try_load() then skips the
        # remote selection and any retry goes straight to the local ladder.
        # Instance-scoped: a new job gets a fresh AudioIntelligence.
        self._remote_decode_rejected = False

    def try_load(self, force_local: bool = False) -> bool:
        """Load a transcription engine: remote → local CUDA ladder → CPU.

        Selection order: a configured + healthy remote OpenAI-compatible
        server wins (``WHISPER_REMOTE_URL`` — nothing loads on the local
        GPU at all); otherwise the local faster-whisper CUDA→CPU ladder
        runs unchanged. ``force_local=True`` skips the remote path — used
        by callers that need the real local engine (whisper_translate's
        native audio→English task, redecode) and by the mid-stage remote
        failure fallback.

        Uses the cached local model if the same model was already loaded.
        """
        log = get_logger()

        if (not force_local and remote_whisper_configured()
                and getattr(self, '_remote_decode_rejected', False)):
            # This instance already got a 200-OK decode from the remote server
            # and REJECTED it (looped/degenerate/filtered-to-nothing). The
            # same audio would deterministically produce the same garbage —
            # skip straight to the local ladder instead of paying a second
            # full remote decode + gap-recovery pass. Applies even in strict
            # mode: strict guards against a flaky HEALTH PROBE, not against
            # affirmative evidence the decode itself is bad.
            log.log_stage('AUDIO',
                'Remote Whisper skipped: its previous decode of this video '
                'was rejected as degenerate — using the local ladder')
        elif not force_local and remote_whisper_configured():
            _strict = bool(getattr(settings, "GPU_STRICT_REMOTE", False))
            _healthy = remote_whisper_healthy()
            # In strict mode we select the remote server even when the quick
            # health probe fails: that probe can false-negative (cold sidecar,
            # slow /v1/health) and must not silently push transcription onto the
            # local GPU. The real request still falls back at transcribe-time if
            # the remote genuinely can't serve it.
            if _healthy or _strict:
                self.engine = RemoteWhisperEngine()
                self.available = True
                self.device_used = 'remote'
                AudioIntelligence._record_loaded(
                    remote_whisper_pick_model(None), 'remote')
                log.log_stage('AUDIO',
                    f'Remote Whisper selected: {_remote_whisper_base()} '
                    '(local GPU stays free)'
                    + ('' if _healthy else
                       ' — strict mode: health probe failed but not falling back to the local GPU'))
                # Pre-warm the Companion's sidecar NOW: language ID + the
                # audio prep run for minutes before the WAV lands, and the
                # observed cold path cost the decode ~35 s of server spawn +
                # model load. Fire-and-forget on a daemon thread; an older
                # Companion answers 404 and nothing changes.
                remote_whisper_warm()
                return True
            log.log_stage('AUDIO',
                f'Remote Whisper configured ({_remote_whisper_base()}) but '
                'health probe failed — using the local ladder')

        try:
            from faster_whisper import WhisperModel
            log.log_stage('AUDIO', 'faster-whisper imported OK')
        except ImportError:
            log.log_stage('AUDIO',
                'faster-whisper not installed. Run: pip install faster-whisper')
            return False

        # State the requested model up front so the log makes any later
        # downgrade unambiguous — and so "what was asked for" vs "what loaded"
        # can be compared at a glance (Task 4).
        log.log_stage('AUDIO',
            f'Whisper model requested: {self.requested_model_name}')

        # Check if we already have this model loaded (skip re-download)
        if (AudioIntelligence._cached_engine is not None
                and AudioIntelligence._cached_model_name == self.model_name):
            self.engine = AudioIntelligence._cached_engine
            self.device_used = AudioIntelligence._cached_device
            self.available = True
            AudioIntelligence._record_loaded(self.model_name, self.device_used)
            # Re-derive batch size from the cached device tier so a reused
            # GPU engine doesn't suddenly run with the wrong workspace.
            if self.device_used and self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _total_gb = (_torch.cuda.mem_get_info()[1]
                                     / 1_073_741_824)
                        self._batch_size = 4 if _total_gb < 5.5 else 16
                except Exception:
                    self._batch_size = 16
            else:
                self._batch_size = 16
            log.log_stage('AUDIO',
                f'Whisper {self.model_name} already loaded ({self.device_used}, '
                f'batch={self._batch_size}) — using cached')
            return True

        # ── Diagnose CUDA ──
        cuda_available = False
        cuda_reason = "unknown"

        try:
            import torch
            torch_cuda = torch.cuda.is_available()
            torch_build = 'CUDA' if torch_cuda else 'CPU-ONLY'
            log.log_stage('AUDIO', f'PyTorch {torch.__version__} ({torch_build})')
            if torch_cuda:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_mem / 1073741824
                log.log_stage('AUDIO', f'GPU: {gpu_name} ({vram_gb:.1f} GB)')
            else:
                log.log_stage('AUDIO',
                    '*** PyTorch is CPU-only. Batched transcription will fail. ***\n'
                    '  FIX: pip install torch --index-url https://download.pytorch.org/whl/cu121')
        except Exception:
            pass

        try:
            import ctranslate2
            try:
                cuda_types = ctranslate2.get_supported_compute_types('cuda')
                if cuda_types:
                    cuda_available = True
                    cuda_reason = "ctranslate2 CUDA OK"
                    log.log_stage('AUDIO', f'CTranslate2 CUDA: {cuda_types}')
                else:
                    cuda_reason = "ctranslate2 no CUDA"
                    log.log_stage('AUDIO',
                        'CTranslate2 has no CUDA support. Fix: pip install ctranslate2 --force-reinstall')
            except Exception as e:
                cuda_reason = f"ctranslate2 error: {e}"
                log.log_stage('AUDIO', f'CTranslate2 CUDA check failed: {e}')
        except ImportError:
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_available = True
                    cuda_reason = "torch CUDA"
            except Exception:
                pass

        log.log_stage('AUDIO', f'CUDA: available={cuda_available}, reason={cuda_reason}')

        # ── VRAM budgets per (model, compute_type) ──
        # First number = load footprint, second = peak workspace under
        # the chosen batched-inference settings. ``int8_float16`` keeps
        # weights in int8 (half the fp16 footprint) but does matmul in
        # fp16 — same speed as float16 on modern GPUs, half the VRAM.
        # Empirical floors from real GTX 1650 / RTX 4060 / RTX 4090 runs.
        _vram_load_gb = {
            ('large',         'float16'):       3.0,
            ('large',         'int8_float16'):  1.6,
            ('large-v2',      'float16'):       3.0,
            ('large-v2',      'int8_float16'):  1.6,
            ('large-v3',      'float16'):       3.0,
            ('large-v3',      'int8_float16'):  1.6,
            # large-v3-turbo = large-v3's full encoder + a pruned 4-layer decoder
            # (~809M params), so its footprint sits between medium and large-v3.
            # int8_float16 (~1.0 GB load) fits a 4 GB GTX 1650 alongside the
            # batched-inference workspace; a runtime OOM still falls back to CPU.
            ('large-v3-turbo','float16'):       1.8,
            ('large-v3-turbo','int8_float16'):  1.0,
            # distil-large-v3 / v3.5 — full encoder + 2-layer decoder (~756M).
            # Near-large-v3 English WER at ~2x turbo speed; footprint one
            # notch under turbo. English-focused — auto-preferred only when
            # the pipeline language is explicitly English.
            ('distil-large-v3','float16'):      1.6,
            ('distil-large-v3','int8_float16'): 0.9,
            ('distil-large-v3.5','float16'):      1.6,
            ('distil-large-v3.5','int8_float16'): 0.9,
            # Kotoba-Whisper v2.0 (Japanese-specialized, distilled ~756M) — same
            # footprint class as distil/turbo.
            ('kotoba-tech/kotoba-whisper-v2.0-faster', 'float16'):      1.8,
            ('kotoba-tech/kotoba-whisper-v2.0-faster', 'int8_float16'): 1.0,
            ('medium',        'float16'):       1.6,
            ('medium',        'int8_float16'):  0.85,
            ('small',         'float16'):       1.0,
            ('small',         'int8_float16'):  0.55,
            ('base',          'float16'):       0.4,
            ('base',          'int8_float16'):  0.25,
            ('tiny',          'float16'):       0.2,
            ('tiny',          'int8_float16'):  0.15,
        }
        _gpu_free_gb = None
        _gpu_total_gb = None
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _free_bytes, _total_bytes = _torch.cuda.mem_get_info()
                _gpu_free_gb = _free_bytes / 1_073_741_824
                _gpu_total_gb = _total_bytes / 1_073_741_824
        except Exception:
            pass

        # On small GPUs (≤4 GB cards like the GTX 1650) prefer
        # ``int8_float16`` over plain ``float16`` so models from medium
        # upward actually fit. Larger cards stick with fp16 for maximum
        # decoder throughput — the int8 path is a touch slower per token.
        _small_gpu = (_gpu_total_gb is not None and _gpu_total_gb < 5.5)
        if _small_gpu:
            self._batch_size = 4   # smaller workspace footprint
            _tiers = [
                ('CUDA int8_float16', 'cuda', 'int8_float16'),
                ('CUDA float16',      'cuda', 'float16'),
                ('CUDA int8',         'cuda', 'int8'),
                ('CPU int8',          'cpu',  'int8'),
            ]
        else:
            self._batch_size = 16
            _tiers = [
                ('CUDA fp16',         'cuda', 'float16'),
                ('CUDA int8_float16', 'cuda', 'int8_float16'),
                ('CUDA int8',         'cuda', 'int8'),
                ('CPU int8',          'cpu',  'int8'),
            ]

        # ── Try the chosen ordered list of (device, compute_type) tiers ──
        for tier, device, compute in _tiers:
            if device == 'cuda' and not cuda_available:
                continue
            # Skip the CUDA tier up front when there's no headroom —
            # otherwise the load succeeds, sits in VRAM, and the later
            # batched-inference workspace check forces a CPU fallback
            # while the GPU copy lingers.
            if device == 'cuda' and _gpu_free_gb is not None:
                load_gb = _vram_load_gb.get(
                    (self.model_name, compute), 1.5 if compute != 'int8' else 0.8)
                # Workspace is roughly 1.0× model size under batch=16,
                # 0.4× under batch=4. Add a small constant for ctranslate2
                # scratch buffers so we don't squeak through and OOM mid-run.
                _ws_mult = 0.45 if self._batch_size <= 4 else 1.0
                _budget = load_gb + load_gb * _ws_mult + 0.25
                if _gpu_free_gb < _budget:
                    log.log_stage('AUDIO',
                        f'Skipping {tier}: only {_gpu_free_gb:.2f} GB VRAM free '
                        f'(need ~{_budget:.2f} GB for {self.model_name} '
                        f'@ batch={self._batch_size})')
                    continue
            try:
                log.log_stage('AUDIO',
                    f'Loading {self.model_name} on {tier} (batch={self._batch_size})...')
                self.engine = WhisperModel(
                    self.model_name, device=device, compute_type=compute)
                self.available = True
                self.device_used = f'{device}_{compute}'
                # Cache for reuse
                AudioIntelligence._cached_engine = self.engine
                AudioIntelligence._cached_model_name = self.model_name
                AudioIntelligence._cached_device = self.device_used
                AudioIntelligence._record_loaded(self.model_name, self.device_used)
                log.log_stage('AUDIO', f'Whisper ready: {tier}')
                # The user's requested model loaded — honor it (Task 4). If we
                # only fit it by falling to CPU while a GPU was available, say so
                # explicitly: the model is unchanged, but it'll be slower.
                if device == 'cpu' and cuda_available:
                    log.log_stage('AUDIO',
                        f'NOTE: requested {self.requested_model_name} loaded on CPU '
                        f'(only {_gpu_free_gb if _gpu_free_gb is not None else 0:.2f} GB VRAM free) '
                        f'— honoring your model choice; transcription will be slower')
                return True
            except Exception as e:
                log.log_stage('AUDIO', f'{tier} FAILED: {type(e).__name__}: {str(e)[:150]}')
                # Free anything ctranslate2 mapped before raising — otherwise
                # the next tier picks up the same fragmented allocator state.
                try:
                    if getattr(self, 'engine', None) is not None:
                        del self.engine
                        self.engine = None
                    import gc as _gc
                    _gc.collect()
                    if device == 'cuda':
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                except Exception:
                    pass

        # Last resort: try 'base' model on CPU. This is the ONLY path that
        # actually changes the model the user asked for, so log it as an
        # explicit, accurate downgrade (Task 4) — never silently report a model
        # other than what loaded.
        if self.model_name != 'base':
            try:
                _free_txt = (f'{_gpu_free_gb:.2f} GB VRAM free'
                             if _gpu_free_gb is not None else 'no GPU')
                log.log_stage('AUDIO',
                    f'DOWNGRADE: requested {self.requested_model_name} would not '
                    f'load on any tier ({_free_txt}) → loading base on CPU')
                self.engine = WhisperModel('base', device='cpu', compute_type='int8')
                self.available = True
                self.model_name = 'base'
                self.device_used = 'cpu_int8_base'
                AudioIntelligence._cached_engine = self.engine
                AudioIntelligence._cached_model_name = 'base'
                AudioIntelligence._cached_device = self.device_used
                AudioIntelligence._record_loaded('base', self.device_used)
                log.log_stage('AUDIO',
                    f'Whisper ready: CPU int8 base (effective model=base, '
                    f'requested={self.requested_model_name})')
                return True
            except Exception as e:
                log.log_stage('AUDIO', f'Base model FAILED: {e}')

        log.log_stage('AUDIO', 'ALL Whisper tiers failed')
        return False

    def transcribe(self, video_path: str, duration_ms: int,
                    language: str = 'auto',
                    on_progress: Callable = None,
                    audio_path_override: Optional[str] = None,
                    remote_only: bool = False) -> dict:
        """Transcribe the video's audio track.
        Optimized for speed: beam_size=1 (greedy), larger batch_size.

        ``remote_only`` (used by the perceiver's CONCURRENT transcription that
        overlaps face detection): never fall back to the LOCAL CUDA/CPU engine —
        that would contend with YOLO on the same card. On any remote failure it
        returns ``{'_remote_failed': True, ...}`` so the caller can run the
        normal (local-capable) pass SEQUENTIALLY, after the local GPU is free."""
        if not self.available:
            # A configured cloud provider can still transcribe without a
            # local faster-whisper install.
            try:
                from backend.services.cloud_transcription import (
                    provider_selected, cloud_available)
                _cloud_ok = provider_selected() != 'local' and cloud_available()
            except Exception:
                _cloud_ok = False
            if not _cloud_ok:
                return {'speech_active': {}, 'segments': [], 'language': ''}

        log = get_logger()
        log.start_timer('transcribe')

        whisper_lang = None if language in ('auto', '', None) else language

        try:
            import tempfile
            _own_audio = False
            if audio_path_override and os.path.exists(audio_path_override):
                # A pre-separated vocal stem (already 16 kHz mono) was supplied
                # by the vocal-separation stage — transcribe it directly instead
                # of re-extracting from the video. The caller owns the file.
                audio_path = audio_path_override
                log.log_stage('AUDIO',
                    f'Using pre-separated vocal track '
                    f'({os.path.basename(audio_path)}) — skipping extraction')
            else:
                # Prefer the pipeline's already-extracted ``audio.wav`` (same job
                # dir) over a SECOND extraction. The main pipeline writes
                # audio.wav with the EXACT same preconditioning chain (both gated
                # by WHISPER_AUDIO_PRECONDITION), so re-extracting here just
                # duplicated ~10 min of ffmpeg work — and under the old fixed
                # 240 s timeout it TIMED OUT on long videos (the denoise+loudnorm
                # chain runs for minutes on a 2 h file), returning an EMPTY
                # transcript. With no transcript the pipeline then silently
                # skipped subtitle translation and produced an empty summary —
                # the "translation didn't run" / "no transcript available"
                # symptom. Reuse the shared file when present; extract only as a
                # fallback, with a duration-scaled timeout.
                _precondition = bool(getattr(settings, "WHISPER_AUDIO_PRECONDITION", True))
                _sibling_wav = os.path.join(
                    os.path.dirname(video_path) or ".", "audio.wav")
                # extract_audio writes atomically (``.part.wav`` → rename), so
                # the final file is COMPLETE whenever it exists — and when the
                # pipeline's extraction is still in flight (the extraction↔
                # perceive overlap), we WAIT for it instead of burning a second
                # full preconditioning pass on the same audio.
                _sibling_ready = _wait_for_sibling_audio(
                    _sibling_wav, _sibling_wav + ".part.wav", duration_ms,
                    grace_s=float(getattr(
                        settings, "PIPELINE_SHARED_AUDIO_GRACE_S", 10.0)))
                if _sibling_ready:
                    audio_path = _sibling_wav
                    _own_audio = False  # shared file — must NOT be deleted below
                    log.log_stage('AUDIO',
                        f'Reusing pre-extracted audio.wav '
                        f'({os.path.getsize(audio_path) / 1048576:.1f} MB) — '
                        'skipping redundant re-extraction')
                else:
                    audio_path = tempfile.mktemp(suffix='.wav')
                    _own_audio = True
                    log.log_stage('AUDIO',
                        f'Extracting audio from {os.path.basename(video_path)}'
                        f' (preconditioning {"on" if _precondition else "off"})...')
                    _extract_cmd = ['ffmpeg', '-y', '-i', video_path, '-vn']
                    if _precondition:
                        # Duration-aware chain — same rule as the main
                        # frame_extractor path. The cap default is now 0
                        # (always denoise — coverage over speed); set a
                        # positive minute count to drop afftdn on long tracks.
                        from backend.services.pipeline_helpers import build_precondition_filters
                        _af = build_precondition_filters(
                            True, (duration_ms or 0) / 1000.0,
                            float(getattr(settings, "WHISPER_PRECONDITION_DENOISE_MAX_MIN", 0) or 0))
                        if _af:
                            _extract_cmd += ['-af', _af]
                    _extract_cmd += ['-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le',
                                     audio_path]
                    # Scale the timeout with duration: a fixed 240 s cap
                    # guaranteed failure on long videos. Floor 10 min; otherwise
                    # allow ~realtime + slack (extraction runs well under that).
                    _dur_s = int((duration_ms or 0) / 1000)
                    _extract_timeout = max(600, _dur_s + 120)
                    result = None
                    try:
                        result = subprocess.run(_extract_cmd, capture_output=True,
                                                timeout=_extract_timeout)
                    except subprocess.TimeoutExpired:
                        log.log_error('AUDIO',
                            f'Audio extraction timed out after {_extract_timeout}s')

                    if not os.path.exists(audio_path):
                        log.log_error(
                            'AUDIO',
                            'Audio extraction failed'
                            + (f' (exit={result.returncode})' if result is not None
                               else ' (timeout)'))
                        # Preconditioning can fail on exotic codecs / filter builds
                        # (or be slow); retry once with a plain copy so a filter
                        # error never costs us the whole transcript.
                        if _precondition:
                            log.log_stage('AUDIO',
                                'Preconditioned extract failed — retrying without filters')
                            try:
                                subprocess.run([
                                    'ffmpeg', '-y', '-i', video_path,
                                    '-vn', '-ac', '1', '-ar', '16000',
                                    '-c:a', 'pcm_s16le', audio_path,
                                ], capture_output=True, timeout=max(600, _dur_s + 120))
                            except subprocess.TimeoutExpired:
                                log.log_error('AUDIO', 'Plain extraction also timed out')
                        if not os.path.exists(audio_path):
                            return {'speech_active': {}, 'segments': [], 'language': ''}

            audio_size = os.path.getsize(audio_path)
            log.log_stage('AUDIO', f'Audio extracted: {audio_size/1048576:.1f} MB')

            # ── Robust spoken-language pin (auto jobs, >= 2 min) ──
            # Whisper's own language detection — and the Companion's whisper.cpp
            # large-v3-turbo especially — mis-reads breathy / sparse-dialogue /
            # music-heavy audio: a Japanese video was repeatedly detected as
            # English and hallucinated into English cues, most of which the
            # no-speech / phantom filters then dropped (so the transcript read as
            # garbage English AND skipped most of the runtime). Identify the
            # language straight from the audio with a dedicated classifier
            # (VoxLingua107 on the server CPU — no Whisper decode, no GPU
            # contention) and PIN it so every path (remote / cloud / local)
            # forces the right language. Best-effort: on any failure we keep
            # Whisper's own auto-detect (the multi-window Whisper vote below
            # still runs for the local path when this leaves it unset).
            if whisper_lang is None and (duration_ms or 0) >= 120000:
                try:
                    from backend.services.spoken_language_id import (
                        identify_spoken_language)
                    _slang, _sdetail = identify_spoken_language(
                        video_path, duration_ms)
                    if _slang and _slang in _WHISPER_LANGUAGE_CODES:
                        whisper_lang = _slang
                        log.log_stage('AUDIO',
                            f'Spoken language identified: {whisper_lang} '
                            f'({_sdetail}) — pinned before transcription, '
                            'overrides Whisper auto-detect (which mis-reads '
                            'breathy/music-heavy audio)')
                    elif _slang:
                        # Detected a language Whisper can't decode — don't force
                        # a code the decoder would reject; keep auto-detect.
                        log.log_stage('AUDIO',
                            f'Spoken language identified as {_slang} ({_sdetail}) '
                            'but Whisper has no decoder for it — using auto-detect')
                    else:
                        log.log_stage('AUDIO',
                            f'Spoken-language ID inconclusive ({_sdetail}) — '
                            'falling back to Whisper auto-detect')
                except Exception as _sl_err:
                    log.log_stage('AUDIO',
                        f'Spoken-language ID failed ({_sl_err}) — '
                        'using Whisper auto-detect')

            # ── Remote Whisper (LAN server, e.g. the GPU Companion) ──
            # The audio (never the video) goes to the OpenAI-compatible
            # server picked in try_load(); output flows through the SAME
            # post chain as cloud STT (hallucination filter → clamp →
            # forced alignment) so downstream is schema-identical. Any
            # mid-stage failure reloads the LOCAL ladder and continues —
            # a job never fails solely because the remote server dropped.
            if self.device_used == 'remote':
                _remote_engine = (self.engine
                                  if isinstance(self.engine, RemoteWhisperEngine)
                                  else RemoteWhisperEngine())
                # whisper_lang was already pinned above by the dedicated
                # spoken-language classifier (VoxLingua107) when the job is auto
                # — so the companion request forces the right language instead of
                # letting whisper.cpp's flaky single-pass auto-detect decide.
                _remote_model = _remote_engine.model or remote_whisper_pick_model(whisper_lang)
                log.log_stage('AUDIO',
                    f'Remote Whisper: {_remote_engine.base} '
                    f'model={_remote_model} language={whisper_lang or "auto"}')
                _remote_engine.model = _remote_model
                _remote = _remote_engine.transcribe_wav(audio_path, whisper_lang)
                if _remote:
                    _result = self._finalize_cloud_transcription(
                        _remote, audio_path, duration_ms, log, on_progress,
                        pinned_language=whisper_lang)
                    if _result is not None:
                        # Voice-gated coverage audit + gap recovery: measure how
                        # much of the ACTUAL speech (Silero VAD) the transcript
                        # covers and re-transcribe only the voice-active runs the
                        # companion missed — never silence. Fully guarded: any
                        # failure returns the un-recovered result untouched.
                        _result = self._audit_and_recover_speech(
                            _result, audio_path, duration_ms, whisper_lang,
                            _remote_engine, log)
                        # Degenerate-decode gate: a transcript that covers
                        # almost NONE of the VAD-confirmed speech on a talky
                        # video is a failed decode (looping whisper.cpp, wrong
                        # audio, truncated upload) — reject it so the local
                        # fallback produces a real transcript instead of the
                        # job completing with 0 segments.
                        if not self._remote_transcript_acceptable(_result, log):
                            _result = None
                    if _result is not None:
                        _result['transcription_provider'] = 'remote'
                        _result['transcription_location'] = 'remote'
                        # Report the REMOTE model as the effective model so
                        # the compute summary shows what actually ran.
                        self.model_name = _remote_model
                        AudioIntelligence._record_loaded(_remote_model, 'remote')
                        if _own_audio:
                            try:
                                os.remove(audio_path)
                            except Exception:
                                pass
                        return _result
                    # The server ANSWERED (200 OK) but its transcript was
                    # rejected — filtered to nothing, or degenerate per the
                    # coverage gate. Re-sending the same audio to the same
                    # server is deterministic: it re-produces the same garbage
                    # and pays the same gap-recovery pass again. Remember the
                    # rejection so any retry on this instance (the perceiver's
                    # sequential pass after remote_only) goes straight to the
                    # local ladder. Transport failures (no response at all) do
                    # NOT set this — a flaky network deserves a second try.
                    self._remote_decode_rejected = True
                if remote_only:
                    # Concurrent-with-faces path: do NOT load local Whisper here
                    # (it would fight YOLO for the local card). Signal the caller
                    # to run the normal, local-capable pass sequentially once the
                    # local GPU is free.
                    log.log_stage('AUDIO',
                        'Remote Whisper failed — remote_only set; deferring the '
                        'local fallback to a sequential pass (keeps the local GPU '
                        'for face detection)')
                    if _own_audio:
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
                    return {'speech_active': {}, 'segments': [], 'language': '',
                            '_remote_failed': True}
                log.log_stage('AUDIO',
                    'Remote Whisper failed mid-stage — falling back to the '
                    'local CUDA→CPU ladder (job continues)')
                self.available = False
                self.engine = None
                self.device_used = 'unknown'
                if not self.try_load(force_local=True):
                    log.log_stage('AUDIO',
                        'Local Whisper fallback also unavailable — no transcript')
                    if _own_audio:
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass
                    return {'speech_active': {}, 'segments': [], 'language': ''}

            # ── Cloud transcription provider (audit Phase 4.1) ──
            # TRANSCRIPTION_PROVIDER=groq|openai runs the primary pass in
            # the cloud; output is mapped to the local segment schema and
            # flows through the SAME post chain (hallucination filter →
            # repetition/clamp → forced alignment → downstream polish +
            # formatter). Any failure falls straight through to local.
            try:
                from backend.services.cloud_transcription import (
                    provider_selected, cloud_available, transcribe_cloud)
                if provider_selected() != 'local':
                    if cloud_available():
                        log.log_stage('AUDIO',
                            f'Cloud transcription: provider={provider_selected()}')
                        _cloud = transcribe_cloud(audio_path, whisper_lang)
                        if _cloud:
                            _result = self._finalize_cloud_transcription(
                                _cloud, audio_path, duration_ms, log, on_progress,
                                pinned_language=whisper_lang)
                            if _result is not None:
                                if _own_audio:
                                    try:
                                        os.remove(audio_path)
                                    except Exception:
                                        pass
                                return _result
                        log.log_stage('AUDIO',
                            'Cloud transcription failed — falling back to local Whisper')
                    else:
                        log.log_stage('AUDIO',
                            f'TRANSCRIPTION_PROVIDER={provider_selected()} set but no '
                            'API key configured — using local Whisper')
            except Exception as _ct_err:
                log.log_stage('AUDIO',
                    f'Cloud transcription error ({_ct_err}) — using local Whisper')

            # ── Robust language detection for the auto path ──
            # Whisper's built-in detection reads only the FIRST window, which
            # mis-reads intros / logos / music / silence — a Japanese video was
            # detected as 'en', which then skipped translation and let Whisper
            # hallucinate English over the JA audio. For any auto job (stem or
            # not), vote across several spread-out windows of the ORIGINAL video
            # audio and force the winner. Best-effort; falls back to auto-detect.
            # Skipped for short clips (< 2 min), where the first-window guess is
            # reliable and the extra probes aren't worth the time.
            if whisper_lang is None and (duration_ms or 0) >= 120000:
                try:
                    _lang, _score, _detail = _detect_language_multiwindow(
                        self.engine, video_path, duration_ms)
                    if _lang:
                        whisper_lang = _lang
                        log.log_stage('AUDIO',
                            f'Language by multi-window vote: {whisper_lang} '
                            f'(score {_score:.2f}; {_detail}) — overrides the '
                            'single-window auto-detect (intros/music misread as en)')
                except Exception as _ld_err:
                    log.log_stage('AUDIO',
                        f'Multi-window language detect failed ({_ld_err}) — '
                        'using single-window auto-detect')

            # Pre-flight: the GPU model is already loaded — only the
            # batched-inference workspace is left to allocate. Reload on
            # CPU only when the remaining headroom is below that workspace
            # footprint (load_gb × ws_mult + scratch). Keeps us on GPU on
            # tight cards like the GTX 1650 while still aborting cleanly
            # when the budget really is too small.
            if self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _free_bytes, _ = _torch.cuda.mem_get_info()
                        _free_gb = _free_bytes / 1_073_741_824
                        _compute = self.device_used.split('_', 1)[1] if '_' in self.device_used else 'float16'
                        # Rough workspace budgets — match the load-time
                        # numbers in try_load() so the two checks agree.
                        _ws_load = {
                            ('large',    'float16'):      3.0,
                            ('large',    'int8_float16'): 1.6,
                            ('large-v2', 'float16'):      3.0,
                            ('large-v2', 'int8_float16'): 1.6,
                            ('large-v3', 'float16'):      3.0,
                            ('large-v3', 'int8_float16'): 1.6,
                            ('large-v3-turbo', 'float16'):      1.8,
                            ('large-v3-turbo', 'int8_float16'): 1.0,
                            ('distil-large-v3', 'float16'):      1.6,
                            ('distil-large-v3', 'int8_float16'): 0.9,
                            ('distil-large-v3.5', 'float16'):      1.6,
                            ('distil-large-v3.5', 'int8_float16'): 0.9,
                            ('kotoba-tech/kotoba-whisper-v2.0-faster', 'float16'):      1.8,
                            ('kotoba-tech/kotoba-whisper-v2.0-faster', 'int8_float16'): 1.0,
                            ('medium',   'float16'):      1.6,
                            ('medium',   'int8_float16'): 0.85,
                            ('small',    'float16'):      1.0,
                            ('small',    'int8_float16'): 0.55,
                            ('base',     'float16'):      0.4,
                            ('base',     'int8_float16'): 0.25,
                            ('tiny',     'float16'):      0.2,
                            ('tiny',     'int8_float16'): 0.15,
                        }.get((self.model_name, _compute), 1.5)
                        _ws_mult = 0.45 if self._batch_size <= 4 else 1.0
                        _needed = _ws_load * _ws_mult + 0.15
                        if _free_gb < _needed:
                            log.log_stage('AUDIO',
                                f'Only {_free_gb:.2f} GB VRAM free '
                                f'(need ~{_needed:.2f} GB workspace for '
                                f'{self.model_name} {_compute} @ batch={self._batch_size}) '
                                '— using CPU to avoid OOM')
                            self._reload_on_cpu()
                except Exception:
                    pass

            log.log_stage('AUDIO', f'Transcribing with {self.model_name} ({self.device_used})...'
                           f' language={whisper_lang or "auto"} batch={self._batch_size}')

            # Transcription parameters:
            #   beam_size=5 (better accuracy — captures ~5-8% more words than greedy)
            #   batch_size=self._batch_size (4 on ≤4GB GPUs, 16 elsewhere)
            #   vad min_silence=300ms (catches brief pauses within sentences)
            #   no_speech_threshold from settings.WHISPER_NO_SPEECH_THRESHOLD
            #       (lower = less likely to skip quiet speech)
            #   anti-repetition decoding via _decoding_kwargs(): cond_prev OFF
            #       by default + no_repeat_ngram_size / compression-ratio +
            #       log-prob thresholds + temperature fallback (rejects loops)
            #   word_timestamps=True (per-word timing for subtitle + reframing)
            _ns_threshold = float(getattr(
                settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.4))
            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=self.engine)
                _bias = _vocab_bias_kwargs(batched.transcribe, language)
                _decode = _decoding_kwargs(batched.transcribe)
                segments_iter, info = batched.transcribe(
                    audio_path, batch_size=self._batch_size,
                    language=whisper_lang,
                    beam_size=_beam_size(), vad_filter=True,
                    vad_parameters=_vad_parameters(),
                    word_timestamps=True,
                    no_speech_threshold=_ns_threshold,
                    **_decode,
                    **_bias,
                )
                log.log_stage('AUDIO',
                    f'Using batched inference (batch={self._batch_size}, beam=5, '
                    f'no_speech_thresh={_ns_threshold}, '
                    f'cond_prev={_decode.get("condition_on_previous_text")}, '
                    f'no_repeat_ngram={_decode.get("no_repeat_ngram_size")})')
            except Exception as e:
                err_str = str(e)
                log.log_stage('AUDIO', f'Batched inference failed: {err_str[:120]}')
                # CUDA out-of-memory or any other CUDA error — reload on CPU
                # so the sequential fallback below doesn't hit the same wall.
                if self.device_used.startswith('cuda') and (
                    'out of memory' in err_str.lower()
                    or 'cublas' in err_str.lower()
                    or 'cuda' in err_str.lower()
                ):
                    log.log_stage('AUDIO',
                        'CUDA out of memory — reloading Whisper on CPU')
                    self._reload_on_cpu()
                log.log_stage('AUDIO', 'Falling back to sequential transcription (slower)')
                _bias = _vocab_bias_kwargs(self.engine.transcribe, language)
                _decode = _decoding_kwargs(self.engine.transcribe)
                segments_iter, info = self.engine.transcribe(
                    audio_path, language=whisper_lang,
                    beam_size=_beam_size(), vad_filter=True,
                    vad_parameters=_vad_parameters(),
                    word_timestamps=True,
                    no_speech_threshold=_ns_threshold,
                    **_decode,
                    **_bias,
                )

            duration_sec = float(getattr(info, 'duration', 0)) or duration_ms / 1000
            language = getattr(info, 'language', 'unknown')

            segments = []
            speech_active = {}
            last_log_pct = 0
            transcribe_start = _time.monotonic()

            # ── TACT: Hallucination detection constants ──
            # Boilerplate phrases live in the module-level
            # _BOILERPLATE_HALLUCINATIONS (multilingual). See _is_boilerplate_hallucination.
            # The confidence-gated phantom filter (check #4 below) reuses the
            # ledger's own low-confidence signal to drop invented cues over
            # silence; resolve its thresholds once here, not per-segment.
            from backend.services.transcript_dedup import is_low_confidence_phantom
            _phantom_on = bool(getattr(settings, "WHISPER_PHANTOM_FILTER_ENABLED", True))
            _phantom_max_avg = float(getattr(settings, "WHISPER_PHANTOM_MAX_AVG_CONF", 0.40))
            _phantom_min_frac = float(getattr(settings, "WHISPER_PHANTOM_MIN_LOWCONF_FRAC", 0.80))
            _phantom_min_ns = float(getattr(settings, "WHISPER_PHANTOM_MIN_NO_SPEECH", 0.50))

            for seg in segments_iter:
                start_ms = int(seg.start * 1000)
                end_ms = int(seg.end * 1000)

                text = seg.text.strip()
                no_speech_prob = getattr(seg, 'no_speech_prob', 0.0) or 0.0

                # Extract word-level timestamps if available. Done BEFORE the
                # hallucination filter so the TACT confidence gate (check #4)
                # can read per-word confidences; also consumed downstream.
                words = []
                if hasattr(seg, 'words') and seg.words:
                    for w in seg.words:
                        word_conf = getattr(w, 'probability', None)
                        if word_conf is None:
                            word_conf = getattr(w, 'confidence', 1.0) or 1.0
                        words.append({
                            'word': w.word.strip() if hasattr(w, 'word') else str(w).strip(),
                            'start': round(getattr(w, 'start', seg.start), 3),
                            'end': round(getattr(w, 'end', seg.end), 3),
                            'confidence': round(float(word_conf), 3),
                        })

                # ── TACT: Hallucination filter ──
                is_hallucination = False

                # 1. Boilerplate blocklist (multilingual)
                if _is_boilerplate_hallucination(text):
                    is_hallucination = True

                # 2. High no_speech_prob + text present → likely hallucination
                if no_speech_prob > 0.7 and text:
                    is_hallucination = True

                # 3. Repetition detection: same phrase repeating 3+ times
                if text and len(text) > 20:
                    words_list = text.lower().split()
                    if len(words_list) >= 6:
                        # Check for repeating patterns of 2-4 words
                        for plen in range(2, 5):
                            if len(words_list) >= plen * 3:
                                pattern = tuple(words_list[:plen])
                                repeats = 0
                                for i in range(0, len(words_list) - plen + 1, plen):
                                    if tuple(words_list[i:i + plen]) == pattern:
                                        repeats += 1
                                if repeats >= 3:
                                    is_hallucination = True
                                    break

                # 3b. Wrong-script gate (language pinned to a CJK source):
                #     decoding is forced to the selected language, so a cue
                #     that comes back as a long run of pure-Latin prose is a
                #     hallucination, not speech ("See you next time in the
                #     video." over a musical outro). Conservative: needs a
                #     pinned CJK language, 4+ words, ≥60% Latin letters and
                #     <10% expected-script characters. Short interjections
                #     and mixed lines (loanwords, romaji fragments inside
                #     real speech) are untouched.
                if (not is_hallucination and text
                        and getattr(settings, "WHISPER_SCRIPT_FILTER", True)
                        and _wrong_script_for_language(text, whisper_lang)):
                    is_hallucination = True

                # 4. TACT confidence gate — the key phantom filter.
                #    Whisper invents short, low-confidence cues over the long
                #    silent / musical stretches (this material was ~79% silence)
                #    that survive every check above: they aren't boilerplate,
                #    their no_speech_prob sits just under 0.7, and they're too
                #    short for the in-segment repetition test ("Don't let",
                #    "So nice", "Hmm."). Reuse the ledger's own low-confidence
                #    signal to drop them — but only with corroboration, so real
                #    quiet speech is kept (see is_low_confidence_phantom).
                _phantom_hit = False
                if (not is_hallucination and text and words and _phantom_on
                        and is_low_confidence_phantom(
                            words, no_speech_prob,
                            max_avg_conf=_phantom_max_avg,
                            min_lowconf_frac=_phantom_min_frac,
                            min_no_speech=_phantom_min_ns)):
                    is_hallucination = True
                    _phantom_hit = True

                seg_entry = {
                    'start_sec': round(seg.start, 3),
                    'end_sec': round(seg.end, 3),
                    'text': text,
                    'words': words,
                    'is_hallucination': is_hallucination,
                    'no_speech_prob': round(no_speech_prob, 3),
                    'avg_logprob': round(float(getattr(
                        seg, 'avg_logprob', 0.0) or 0.0), 3),
                }
                if _phantom_hit:
                    # Remember WHY it was quarantined: a low-confidence phantom
                    # (not boilerplate/repetition) may still be REAL quiet
                    # speech — the VAD-confirm pass below checks the voice map
                    # and routes it to the redecode queue instead of dropping.
                    seg_entry['phantom'] = True
                segments.append(seg_entry)

                # Align to 100ms grid for speech_active (backwards compat)
                if not is_hallucination:
                    aligned_start = (start_ms // 100) * 100
                    aligned_end = end_ms + 100
                    for t in range(aligned_start, aligned_end, 100):
                        speech_active[t] = True

                if on_progress and duration_sec > 0:
                    on_progress(min(1.0, seg.end / duration_sec))

                if duration_sec > 0:
                    pct = int(seg.end / duration_sec * 100)
                    if pct >= last_log_pct + 25:
                        elapsed_so_far = _time.monotonic() - transcribe_start
                        remaining = (elapsed_so_far / max(0.01, seg.end / duration_sec)) - elapsed_so_far
                        log.log_stage('AUDIO',
                            f'  Transcribing {pct}% — {len(segments)} segments, '
                            f'{elapsed_so_far:.1f}s elapsed, ~{remaining:.0f}s remaining')
                        last_log_pct = pct

            # Phase hint: the per-segment loop above tracks audio time, so a
            # video whose speech ends early leaves the bar frozen at e.g. 79%
            # while redecode + gap-fill + alignment run for minutes. Tell the
            # pipeline the transcription band is done and refinement started.
            if on_progress:
                try:
                    on_progress(1.0, 'transcript_refine')
                except TypeError:
                    on_progress(1.0)   # legacy single-arg callback
            # ── Filter/coverage tension: VAD-confirm phantom-flagged cues ──
            # The TACT phantom gate protects against invented cues, but it
            # also fires on REAL quiet speech (low word confidence is exactly
            # what soft/off-mic dialogue looks like). Before those cues are
            # dropped, check the independent Silero voice map: a phantom
            # whose span overlaps confirmed voice is routed to the
            # low-confidence redecode queue (with a raised budget) instead of
            # being discarded — the redecode either rescues real speech or
            # confirms the drop. Non-fatal: any failure leaves the phantom
            # verdicts unchanged.
            if (getattr(settings, "WHISPER_REDECODE_ENABLED", True)
                    and bool(getattr(settings,
                                     "WHISPER_PHANTOM_VAD_RESCUE_ENABLED", True))
                    and any(s.get('phantom') for s in segments)):
                try:
                    from backend.services import speech_coverage as SC
                    _voice_map = SC.voice_activity_regions(audio_path)
                    _n_confirmed = 0
                    if _voice_map:
                        for s in segments:
                            if s.get('phantom') and SC.overlaps_voice(
                                    _voice_map, s.get('start_sec', 0),
                                    s.get('end_sec', 0)):
                                s['vad_confirmed_voice'] = True
                                _n_confirmed += 1
                    if _n_confirmed:
                        log.log_stage('AUDIO',
                            f'TACT phantom filter: {_n_confirmed} low-confidence '
                            'cue(s) overlap VAD-confirmed voice — routing to the '
                            'redecode queue instead of dropping')
                except Exception as _vr_err:
                    log.log_stage('AUDIO',
                        f'Phantom VAD-confirm pass skipped (non-fatal): {_vr_err}')

            # ── Two-pass difficult-segment redecode (audit Phase 3.4) ──
            # Hallucination-flagged / low-logprob / degenerate-word-timing
            # segments get one focused sequential re-decode (beam 8,
            # patience) before the polish LLM ever sees them. Bounded to
            # ≤10% of segments, worst first. Non-fatal on any error.
            if (getattr(settings, "WHISPER_REDECODE_ENABLED", True)
                    and segments):
                try:
                    n_redecoded = self._redecode_difficult_segments(
                        audio_path, segments, whisper_lang, log)
                    if n_redecoded:
                        log.log_stage('AUDIO',
                            f'Difficult-segment redecode: {n_redecoded} '
                            'segment(s) re-decoded with beam=8')
                except Exception as _rd_err:
                    log.log_stage('AUDIO',
                        f'Difficult-segment redecode skipped (non-fatal): {_rd_err}')

            # ── Gap-fill pass: re-transcribe uncovered runs ──
            # The VAD filter + ``no_speech_threshold`` on the main pass
            # silently drop quiet / soft / off-mic / sung speech. The
            # YouTube vs ClipAI comparison on the GUNDAM Wing episode 1
            # showed ClipAI's transcript starting at ~1:59 while YouTube
            # had dialogue / lyrics from 0:26 — 90 seconds of audio
            # that VAD classified as silence. Find the uncovered runs
            # in the segments list and re-transcribe just those
            # regions with no VAD and a very low no-speech threshold
            # so quiet speech gets a second chance. Gap-fill segments
            # are tagged ``source='gap_fill'`` and routed through the
            # same hallucination filter — the relaxed thresholds only
            # apply inside the gaps where the main pass already
            # produced nothing.
            if (getattr(settings, "WHISPER_GAP_FILL_ENABLED", True)
                    and segments and duration_sec > 0):
                try:
                    gap_segments = self._gap_fill_pass(
                        audio_path, segments, duration_sec, whisper_lang, log)
                    if gap_segments:
                        # Merge gap-fill segments and re-sort by start time
                        # so the Coverage Ledger / per-segment loops see
                        # them in temporal order.
                        segments.extend(gap_segments)
                        segments.sort(key=lambda s: s.get('start_sec', 0))
                        log.log_stage('AUDIO',
                            f'Gap-fill added {len(gap_segments)} segments '
                            f'(total {len(segments)})')
                except Exception as gf_err:
                    log.log_stage('AUDIO',
                        f'Gap-fill pass failed (non-fatal): {gf_err}')

            # ── Drop repetition-loop hallucinations (BOTH passes) ──
            # Whisper loops over music / quiet regions and re-emits the SAME
            # line repeatedly, scattered across the timeline — the repeated
            # verbatim Japanese run-ons (seen 10-11× each) and the short English
            # fragments on the mostly-silent material. Run UNCONDITIONALLY, not
            # just when gap-fill produced output: the MAIN pass loops too, and
            # on a ~79%-silent video gap-fill may add nothing yet the primary
            # transcript still carries the repeats. Order-preserving; keeps the
            # earliest occurrence (bracketed markers and a few short
            # interjections are exempt — see drop_repetition_loops).
            if segments:
                _pre_dedup = len(segments)
                segments = _drop_repetition_loops(segments)
                if len(segments) != _pre_dedup:
                    log.log_stage('AUDIO',
                        f'Repetition-loop filter: {_pre_dedup} → {len(segments)} '
                        f'segments ({_pre_dedup - len(segments)} loop repeats dropped)')

            # ── Clamp timestamp drift to the real audio end ──
            # faster-whisper drifts/loops on repetitive music and can stamp cues
            # PAST the audio (e.g. cues at 35:51 on a 24:27 video), mis-timing
            # the back third of the subtitles. Those past-the-end cues are
            # loop-repeats of earlier lines, not real tail dialogue, so drop them
            # (and clamp any cue that merely overruns the end). Clamp to the
            # larger of info.duration and the caller's media duration so a short
            # info.duration can never truncate legitimate tail speech; a clean
            # run with no drift is a no-op.
            try:
                from backend.services.transcript_dedup import clamp_segments_to_duration
                _clamp_limit = max(float(duration_sec), (duration_ms or 0) / 1000.0)
                segments, _drifted = clamp_segments_to_duration(
                    segments, _clamp_limit, start_key='start_sec', end_key='end_sec')
                if _drifted:
                    log.log_stage('AUDIO',
                        f'Timestamp-drift clamp @ {_clamp_limit:.0f}s audio end: '
                        f'dropped/clamped {_drifted} cue(s) past the end')
            except Exception as _clamp_err:
                log.log_stage('AUDIO',
                    f'Timestamp-drift clamp skipped (non-fatal): {_clamp_err}')

            # ── TACT: Build Coverage Ledger ──
            ledger = CoverageLedger(bin_width_ms=20, duration_ms=int(duration_sec * 1000))

            # Initialize all bins as uncovered
            for t in range(0, int(duration_sec * 1000), 20):
                ledger.bins[t] = LedgerBin(status='uncovered')

            # Claim bins from segments
            hallucinated_count = 0
            low_conf_count = 0
            for seg_entry in segments:
                seg_start = int(seg_entry['start_sec'] * 1000)
                seg_end = int(seg_entry['end_sec'] * 1000)

                if seg_entry.get('is_hallucination'):
                    # Quarantine hallucinated bins
                    hallucinated_count += 1
                    for t in range(seg_start, seg_end, 20):
                        if t in ledger.bins:
                            ledger.bins[t] = LedgerBin(
                                status='quarantined',
                                text=seg_entry['text'][:30],
                                source='whisper_primary',
                                confidence=0.0,
                            )
                    continue

                # Word-level bin claiming (more precise)
                if seg_entry.get('words'):
                    for w in seg_entry['words']:
                        w_start = int(w['start'] * 1000)
                        w_end = int(w['end'] * 1000)
                        conf = w.get('confidence', 1.0)
                        status = 'covered_speech' if conf >= 0.4 else 'low_confidence'
                        if conf < 0.4:
                            low_conf_count += 1
                        for t in range(w_start, w_end, 20):
                            if t in ledger.bins:
                                ledger.bins[t] = LedgerBin(
                                    status=status,
                                    text=w.get('word', ''),
                                    source='whisper_primary',
                                    confidence=conf,
                                )
                else:
                    # Segment-level claiming (no word timestamps)
                    for t in range(seg_start, seg_end, 20):
                        if t in ledger.bins:
                            ledger.bins[t] = LedgerBin(
                                status='covered_speech',
                                text=seg_entry['text'][:20],
                                source='whisper_primary',
                                confidence=0.7,
                            )

            # Label remaining uncovered bins as silence
            silence_count = 0
            for t, b in ledger.bins.items():
                if b.status == 'uncovered':
                    ledger.bins[t] = LedgerBin(
                        status='covered_silence',
                        source='inferred',
                        confidence=0.8,
                        event_type='silence',
                    )
                    silence_count += 1

            # Build audio_events from high no_speech_prob segments
            # (lightweight event inference — YAMNet integration is Phase 4)
            audio_events = {}
            for seg_entry in segments:
                if seg_entry.get('no_speech_prob', 0) > 0.6 and not seg_entry.get('is_hallucination'):
                    seg_start = int(seg_entry['start_sec'] * 1000)
                    seg_end = int(seg_entry['end_sec'] * 1000)
                    for t in range(seg_start, seg_end, 100):
                        audio_events[t] = 'noise'
                        # Update ledger bin event_type
                        bin_key = (t // 20) * 20
                        if bin_key in ledger.bins:
                            ledger.bins[bin_key].event_type = 'noise'

            report = ledger.coverage_report()

            # Only delete the temp WAV we extracted ourselves; a supplied
            # vocal-stem override is owned (and cleaned up) by the caller.
            if _own_audio:
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

            elapsed = log.stop_timer('transcribe')
            speed_ratio = duration_sec / max(0.01, elapsed)

            # Filter hallucinations from the segment list for downstream consumers
            clean_segments = [s for s in segments if not s.get('is_hallucination')]

            # ── Cross-segment hallucination validation ──
            # Remove exact duplicates, temporal overlaps, and >80% word-overlap
            # repeats that slip past the per-segment TACT filter above.
            clean_segments = _cross_validate_segments(clean_segments)

            log.log_stage('AUDIO',
                f'Transcription complete: {len(clean_segments)} segments '
                f'({hallucinated_count} hallucinations quarantined, '
                f'{low_conf_count} low-confidence words), '
                f'language={language}, {elapsed:.1f}s ({speed_ratio:.1f}x realtime)',
                speech_seconds=round(len(speech_active) * 0.1, 1),
                coverage=f'{report["coverage_ratio"]:.1%}')

            # ── CTC forced-alignment refinement (audit Phase 3.1) ──
            # Snap word (and cue) boundaries to actual speech onset/offset;
            # Whisper's own word times drift 50-200 ms. Fail-safe no-op when
            # no aligner backend is available. Runs while the decoded audio
            # is still on disk; GPU only when >1.5 GB VRAM is free.
            try:
                from backend.services.forced_aligner import refine_word_timestamps
                _fa_stats = refine_word_timestamps(
                    audio_path, clean_segments, language)
                if _fa_stats.get('segments_aligned'):
                    log.log_stage('AUDIO',
                        f"Forced alignment ({_fa_stats['backend']}): "
                        f"{_fa_stats['segments_aligned']} segments / "
                        f"{_fa_stats['words_aligned']} words refined, "
                        f"mean shift {_fa_stats['mean_shift_ms']}ms")
            except Exception as _fa_err:
                log.log_stage('AUDIO',
                    f'Forced alignment skipped (non-fatal): {_fa_err}')

            if on_progress:
                on_progress(1.0)

            return {
                'speech_active': speech_active,
                'segments': clean_segments,
                'language': language,
                'coverage_ledger': ledger,
                'audio_events': audio_events,
            }

        except Exception as e:
            log.log_error('AUDIO', f'Transcription failed: {e}')
            return {'speech_active': {}, 'segments': [], 'language': ''}

    def _finalize_cloud_transcription(self, cloud: dict, audio_path: str,
                                      duration_ms: int, log, on_progress,
                                      pinned_language: Optional[str] = None) -> Optional[dict]:
        """Run cloud STT output through the same post chain as local.

        Applies the hallucination filter, repetition-loop drop, end clamp
        and forced alignment, then builds speech_active + a coverage
        ledger in the local result schema. Returns None if everything was
        filtered out (caller falls back to local).

        ``pinned_language`` is the language the JOB pinned (the caller's
        ``whisper_lang``), NOT the script the provider happened to detect.
        The wrong-script gate only fires when the job actually pinned a
        language — on an AUTO job it stays off so genuine minority-language
        speech (English lines inside a mostly-JA video) is never dropped.
        """
        segments = cloud.get('segments') or []
        language = cloud.get('language') or 'unknown'
        duration_sec = (duration_ms or 0) / 1000.0
        if not duration_sec and segments:
            duration_sec = max(s['end_sec'] for s in segments)

        # Hallucination filter — the SAME stack as the local loop, so the
        # companion / cloud path is no longer weaker: boilerplate + no_speech
        # clamp + wrong-script CJK gate + the TACT low-confidence phantom gate.
        _phantom_on = bool(getattr(settings, "WHISPER_PHANTOM_FILTER_ENABLED", True))
        _phantom_max_avg = float(getattr(settings, "WHISPER_PHANTOM_MAX_AVG_CONF", 0.40))
        _phantom_min_frac = float(getattr(settings, "WHISPER_PHANTOM_MIN_LOWCONF_FRAC", 0.80))
        _phantom_min_ns = float(getattr(settings, "WHISPER_PHANTOM_MIN_NO_SPEECH", 0.50))
        _script_on = bool(getattr(settings, "WHISPER_SCRIPT_FILTER", True))
        _ns_drop = float(getattr(settings, "WHISPER_CLOUD_NO_SPEECH_DROP", 0.7))
        try:
            from backend.services.transcript_dedup import is_low_confidence_phantom as _phantom
        except Exception:
            _phantom = None
        for entry in segments:
            text = entry.get('text') or ''
            ns = float(entry.get('no_speech_prob', 0.0) or 0.0)
            words = entry.get('words') or []
            if _is_boilerplate_hallucination(text):
                entry['is_hallucination'] = True
            elif ns > _ns_drop and text:
                entry['is_hallucination'] = True
            elif (text and _script_on and pinned_language
                  and _wrong_script_for_language(text, pinned_language)):
                # Long pure-Latin prose forced onto a PINNED CJK job is a
                # phantom ("See you next time." over a JA musical outro).
                # Gate on the job's PIN, not the provider-detected language —
                # on an auto job the pin is None and this stays off so real
                # English speech inside a JA video survives.
                entry['is_hallucination'] = True
            elif (text and words and _phantom_on and _phantom is not None
                  and _phantom(words, ns, max_avg_conf=_phantom_max_avg,
                               min_lowconf_frac=_phantom_min_frac,
                               min_no_speech=_phantom_min_ns)):
                # Short low-confidence cues invented over silence/music that
                # slip past the no_speech clamp (schema-flexible; no-op when
                # the provider returns no per-word confidence).
                entry['is_hallucination'] = True

        try:
            segments = _drop_repetition_loops(segments)
        except Exception:
            pass
        try:
            from backend.services.transcript_dedup import clamp_segments_to_duration
            if duration_sec:
                segments, _ = clamp_segments_to_duration(
                    segments, duration_sec, start_key='start_sec', end_key='end_sec')
        except Exception:
            pass
        if not segments:
            return None

        # Forced alignment — same refinement as local (also gives word
        # timing to providers that return none, e.g. gpt-4o-transcribe)
        try:
            from backend.services.forced_aligner import refine_word_timestamps
            _fa = refine_word_timestamps(audio_path, segments, language)
            if _fa.get('segments_aligned'):
                log.log_stage('AUDIO',
                    f"Forced alignment ({_fa['backend']}): "
                    f"{_fa['segments_aligned']} cloud segments refined")
        except Exception:
            pass

        segments = _cross_validate_segments(
            [s for s in segments if not s.get('is_hallucination')])
        if not segments:
            # EVERYTHING the provider returned was filtered as hallucinated /
            # looped / drifted. Whether that is a FAILED decode or simply a
            # (near-)speech-free video depends on evidence the decode can't
            # give us — so ask Silero. With substantial VAD-confirmed speech
            # present, an empty transcript is a failed decode: return None so
            # the caller runs the local fallback (the 128-min run where
            # whisper.cpp looped — 257 segments, 254 verbatim repeats —
            # shipped COMPLETE with no transcript precisely because this case
            # fell through as success). With little or no detected speech,
            # empty IS the correct transcript: return a valid empty result so
            # legitimately speech-free content (a music video, a screencast
            # with no narration) doesn't pay a redundant local decode.
            _voice_sec = None
            try:
                from backend.services import speech_coverage as SC
                _voice = SC.voice_activity_regions(audio_path)
                if _voice:  # [] is ambiguous (no speech OR VAD unavailable)
                    _voice_sec = sum(b - a for a, b in _voice)
            except Exception:
                _voice_sec = None
            _min_voice = float(getattr(
                settings, "REMOTE_TRANSCRIPT_MIN_VOICE_S", 120.0))
            if _voice_sec is not None and _voice_sec < _min_voice:
                log.log_stage('AUDIO',
                    f'Cloud transcription empty after filtering, and Silero '
                    f'found only {_voice_sec:.0f}s of speech in the audio — '
                    'treating as legitimately speech-free (no local re-decode)')
                _ledger = CoverageLedger(bin_width_ms=20,
                                         duration_ms=int(duration_sec * 1000))
                for _t in range(0, int(duration_sec * 1000), 20):
                    _ledger.bins[_t] = LedgerBin(status='uncovered')
                if on_progress:
                    on_progress(1.0)
                return {
                    'speech_active': {},
                    'segments': [],
                    'language': language,
                    'coverage_ledger': _ledger,
                    'audio_events': [],
                    'transcription_provider': cloud.get('provider'),
                    'no_speech_evidence': True,
                }
            log.log_stage('AUDIO',
                'Cloud transcription REJECTED: every returned segment was '
                'filtered as hallucinated/looped'
                + (f' while Silero detected {_voice_sec:.0f}s of speech'
                   if _voice_sec is not None else '')
                + ' — treating the remote decode as failed so the local '
                  'fallback can run')
            return None

        speech_active = {}
        for entry in segments:
            s0 = (int(entry['start_sec'] * 1000) // 100) * 100
            s1 = int(entry['end_sec'] * 1000) + 100
            for t in range(s0, s1, 100):
                speech_active[t] = True

        ledger = CoverageLedger(bin_width_ms=20,
                                duration_ms=int(duration_sec * 1000))
        for t in range(0, int(duration_sec * 1000), 20):
            ledger.bins[t] = LedgerBin(status='uncovered')
        for entry in segments:
            for w in entry.get('words') or [{'start': entry['start_sec'],
                                             'end': entry['end_sec'],
                                             'confidence': 0.9}]:
                for t in range(int(w['start'] * 1000), int(w['end'] * 1000), 20):
                    if t in ledger.bins:
                        ledger.bins[t] = LedgerBin(
                            status='covered_speech',
                            source=f"cloud_{cloud.get('provider', '')}",
                            confidence=w.get('confidence', 0.9))

        log.log_stage('AUDIO',
            f"Cloud transcription complete ({cloud.get('provider')}/"
            f"{cloud.get('model')}): {len(segments)} segments, "
            f"language={language}")
        if on_progress:
            on_progress(1.0)
        return {
            'speech_active': speech_active,
            'segments': segments,
            'language': language,
            'coverage_ledger': ledger,
            'audio_events': [],
            'transcription_provider': cloud.get('provider'),
        }

    def _remote_transcript_acceptable(self, result: dict, log) -> bool:
        """Sanity gate on a remote/companion transcript AFTER filtering + the
        coverage audit: is this a plausible transcript of the video, or a
        failed decode wearing a 200 OK?

        Rejects (→ caller falls back to local Whisper) when:
          * the filtered transcript is EMPTY (unless finalize marked it
            ``no_speech_evidence`` — VAD confirmed the audio holds no
            substantial speech, so empty IS the correct transcript), or
          * the independent Silero voice map found substantial speech
            (≥ REMOTE_TRANSCRIPT_MIN_VOICE_S) and the transcript covers less
            than REMOTE_TRANSCRIPT_MIN_COVERAGE of it — the signature of a
            looped/garbage decode (the whisper.cpp context-loop run: 1725 s of
            speech, 0% covered after 254 verbatim repeats were filtered).

        Accepts when no coverage evidence exists (audit disabled / VAD
        unavailable) and the transcript is non-empty — no evidence, no
        rejection. Never raises.
        """
        try:
            if not (result or {}).get('segments'):
                if (result or {}).get('no_speech_evidence'):
                    # Finalize already checked Silero and found (almost) no
                    # speech — an empty transcript is CORRECT for this audio.
                    log.log_stage('AUDIO',
                        'Remote transcript is empty but VAD confirms the audio '
                        'is (near-)speech-free — accepting the empty transcript')
                    return True
                log.log_stage('AUDIO',
                    'Remote transcript REJECTED: 0 segments after filtering')
                return False
            st = (result or {}).get('speech_coverage') or {}
            voice_sec = float(st.get('voice_sec', 0.0) or 0.0)
            ratio = float(st.get('coverage_ratio', 1.0) or 0.0)
            covered = float(st.get('covered_voice_sec', 0.0) or 0.0)
            min_cov = float(getattr(
                settings, "REMOTE_TRANSCRIPT_MIN_COVERAGE", 0.15))
            min_voice = float(getattr(
                settings, "REMOTE_TRANSCRIPT_MIN_VOICE_S", 120.0))
            min_covered = float(getattr(
                settings, "REMOTE_TRANSCRIPT_MIN_COVERED_S", 30.0))
            # BOTH the ratio and the absolute floor must fail: Silero counts
            # singing/shouts as voice, and the wrong-script/no-speech filters
            # intentionally drop lyric cues — so a concert VOD with 90 s of
            # real MC dialogue against 700 s of sung "voice" has a tiny RATIO
            # yet a perfectly good transcript. A genuine garbage decode has
            # both: almost no ratio AND almost no absolute covered speech.
            if voice_sec >= min_voice and ratio < min_cov and covered < min_covered:
                logger.warning(
                    "Remote transcript rejected as degenerate: covers only "
                    "%.0f%% (%.0fs) of %.0fs VAD-detected speech "
                    "(floors: %.0f%% / %.0fs)",
                    ratio * 100, covered, voice_sec, min_cov * 100, min_covered)
                log.log_stage('AUDIO',
                    f'Remote transcript REJECTED: covers only {ratio:.0%} '
                    f'({covered:.0f}s) of {voice_sec:.0f}s of detected speech '
                    '— degenerate decode; falling back to local Whisper')
                return False
        except Exception:
            return True  # gate must never break a working transcript
        return True

    def _audit_and_recover_speech(self, result: dict, audio_path: str,
                                  duration_ms: int, whisper_lang,
                                  remote_engine, log) -> dict:
        """Measure speech coverage against an independent voice-activity map and
        recover the voice-active runs the companion missed.

        'Transcribe the whole video' means 'miss none of the SPEECH' — silence
        must stay blank (forcing it invents cues). We take a Silero VAD map
        (bundled with faster-whisper, so it's independent of the companion's
        decode), report how much of the actual speech the transcript covers, and
        re-transcribe ONLY the voice-active gaps via the companion. Fully
        guarded: any failure returns ``result`` unchanged.
        """
        try:
            if not bool(getattr(settings, "SPEECH_COVERAGE_AUDIT_ENABLED", True)):
                return result
            from backend.services import speech_coverage as SC
            voice = SC.voice_activity_regions(audio_path)
            if not voice:
                return result  # VAD unavailable — keep behavior unchanged

            def _covered(segs):
                out = []
                for s in segs or []:
                    try:
                        a = float(s.get('start_sec', s.get('start', 0)) or 0)
                        b = float(s.get('end_sec', s.get('end', 0)) or 0)
                        if b > a:
                            out.append((a, b))
                    except (TypeError, ValueError):
                        continue
                return out

            segs = result.get('segments') or []
            st = SC.coverage_stats(voice, _covered(segs))
            result['speech_coverage'] = st

            # ── Voice-gated gap recovery ──
            # When it can't run over real gaps, SAY WHY: the observed
            # 84%-coverage run left 272 s of speech unrecovered with zero log
            # evidence of the skip.
            recovered = 0
            gaps = SC.uncovered_voice_gaps(
                voice, _covered(segs),
                min_gap_s=float(getattr(settings, "SPEECH_GAP_MIN_SEC", 1.2)))
            _rec_enabled = bool(getattr(settings, "SPEECH_GAP_RECOVERY_ENABLED", True))
            if gaps and not _rec_enabled:
                log.log_stage('AUDIO',
                    f'Gap recovery skipped ({len(gaps)} voice-active gap(s) '
                    'uncovered): SPEECH_GAP_RECOVERY_ENABLED is off')
            elif gaps and remote_engine is None:
                log.log_stage('AUDIO',
                    f'Gap recovery skipped ({len(gaps)} voice-active gap(s) '
                    'uncovered): no remote engine handle on this path')
            if _rec_enabled and remote_engine is not None:
                if gaps:
                    new_segs = self._recover_voice_gaps_remote(
                        gaps, audio_path, whisper_lang, remote_engine, log)
                    if new_segs:
                        merged = _cross_validate_segments(
                            sorted(segs + new_segs,
                                   key=lambda s: float(s.get('start_sec', 0) or 0)))
                        recovered = max(0, len(merged) - len(segs))
                        result['segments'] = merged
                        # Rebuild speech_active from the merged set.
                        sa = {}
                        for entry in merged:
                            s0 = (int(float(entry.get('start_sec', 0) or 0) * 1000) // 100) * 100
                            s1 = int(float(entry.get('end_sec', 0) or 0) * 1000) + 100
                            for t in range(s0, s1, 100):
                                sa[t] = True
                        result['speech_active'] = sa
                        # Rebuild the coverage ledger too — the TACT planner reads
                        # it as the covered-speech map, so a recovered run must be
                        # marked 'covered_speech' or the planner would redundantly
                        # gap-fill a region we just recovered.
                        _dur_s = (duration_ms or 0) / 1000.0
                        if not _dur_s and merged:
                            _dur_s = max(float(s.get('end_sec', 0) or 0) for s in merged)
                        if _dur_s > 0:
                            led = CoverageLedger(bin_width_ms=20,
                                                 duration_ms=int(_dur_s * 1000))
                            for t in range(0, int(_dur_s * 1000), 20):
                                led.bins[t] = LedgerBin(status='uncovered')
                            for entry in merged:
                                for w in entry.get('words') or [
                                        {'start': entry.get('start_sec', 0),
                                         'end': entry.get('end_sec', 0),
                                         'confidence': 0.9}]:
                                    for t in range(int(float(w.get('start', 0) or 0) * 1000),
                                                   int(float(w.get('end', 0) or 0) * 1000), 20):
                                        if t in led.bins:
                                            led.bins[t] = LedgerBin(
                                                status='covered_speech',
                                                source=entry.get('source', 'remote'),
                                                confidence=w.get('confidence', 0.9))
                            result['coverage_ledger'] = led
                        # Re-measure coverage after recovery.
                        st = SC.coverage_stats(voice, _covered(merged))
                        result['speech_coverage'] = st

            # Honest tail: only claim the remainder is music/silence when the
            # transcript actually covered (nearly) all the speech. A low ratio
            # is a WARNING — the transcript is likely incomplete or the decode
            # failed, and saying "the rest is silence" at 0% hid a dead run.
            _ratio = float(st.get('coverage_ratio', 0.0) or 0.0)
            if _ratio >= 0.85:
                _tail = (". The untranscribed remainder of the runtime is "
                         "music/silence with no speech.")
            elif _ratio >= 0.5:
                _tail = (". Coverage is below target — some speech may be "
                         "missing from the transcript.")
            else:
                _tail = (". LOW COVERAGE: most detected speech is NOT in the "
                         "transcript — the decode likely failed or degraded.")
                logger.warning(
                    "Speech coverage only %.0f%% of %.0fs detected speech — "
                    "transcript likely incomplete", _ratio * 100,
                    float(st.get('voice_sec', 0.0) or 0.0))
            log.log_stage('AUDIO',
                f"Speech coverage: {st['coverage_ratio']:.0%} of voice-active "
                f"audio transcribed ({st['covered_voice_sec']:.0f}s of "
                f"{st['voice_sec']:.0f}s speech; "
                f"{int(st['gap_count'])} gap(s), {st['uncovered_voice_sec']:.0f}s "
                f"left)"
                + (f" — recovered {recovered} missed cue(s)" if recovered else "")
                + _tail)
        except Exception as e:
            log.log_stage('AUDIO', f'Speech-coverage audit skipped ({e})')
        return result

    def _recover_voice_gaps_remote(self, gaps, audio_path, whisper_lang,
                                   remote_engine, log) -> list:
        """Re-transcribe voice-active gap runs via the companion and return the
        clean, offset segments to merge. Bounded by count + total duration; each
        slice runs through the SAME hallucination filter as the main pass. Any
        per-gap failure is skipped, never fatal."""
        import tempfile
        max_runs = int(getattr(settings, "SPEECH_GAP_RECOVERY_MAX_RUNS", 40))
        max_total = float(getattr(settings, "SPEECH_GAP_RECOVERY_MAX_TOTAL_SEC", 900.0))
        # Longest gaps first — that's where the most missed speech is.
        ordered = sorted(gaps, key=lambda g: (g[1] - g[0]), reverse=True)
        picked, total = [], 0.0
        for g in ordered:
            if len(picked) >= max_runs or total >= max_total:
                break
            picked.append(g)
            total += (g[1] - g[0])
        if not picked:
            return []
        picked.sort(key=lambda g: g[0])
        log.log_stage('AUDIO',
            f'Gap recovery: re-transcribing {len(picked)} voice-active run(s) '
            f'({total:.0f}s) the companion missed')
        _ns_drop = float(getattr(settings, "WHISPER_CLOUD_NO_SPEECH_DROP", 0.7))
        out = []
        # Failure accounting: per-slice errors were silently swallowed, which
        # made a dead/cold sidecar look identical to "the gaps were silence".
        # Consecutive remote failures with zero successes abort the loop early
        # (each failed slice costs a full upload timeout) and get reported.
        _remote_fails = 0
        _remote_ok = 0
        for (a, b) in picked:
            if _remote_ok == 0 and _remote_fails >= 3:
                log.log_stage('AUDIO',
                    f'Gap recovery aborted: first {_remote_fails} slice '
                    'uploads all failed (remote engine unreachable?) — '
                    f'skipping the remaining {len(picked) - _remote_fails} gap(s)')
                break
            wav = tempfile.mktemp(suffix='.wav')
            try:
                # Small pad so a word straddling the boundary isn't clipped.
                ss = max(0.0, a - 0.20)
                dur = (b - a) + 0.40
                # Boost quiet/off-mic speech before the retry decode — these
                # spans were dropped as too faint the first time, so lift them
                # (highpass → denoise → speechnorm) for the second chance.
                boost = _gap_boost_af_args()
                subprocess.run(
                    ['ffmpeg', '-y', '-ss', f'{ss:.3f}', '-t', f'{dur:.3f}',
                     '-i', audio_path, '-vn'] + boost + ['-ac', '1',
                     '-ar', '16000', '-c:a', 'pcm_s16le', wav],
                    capture_output=True, timeout=120)
                if boost and (not os.path.exists(wav)
                              or os.path.getsize(wav) < 2000):
                    # The boost chain can fail on an ffmpeg build without
                    # afftdn/speechnorm — retry the slice unfiltered rather
                    # than lose the gap entirely.
                    subprocess.run(
                        ['ffmpeg', '-y', '-ss', f'{ss:.3f}', '-t', f'{dur:.3f}',
                         '-i', audio_path, '-vn', '-ac', '1', '-ar', '16000',
                         '-c:a', 'pcm_s16le', wav],
                        capture_output=True, timeout=120)
                if not os.path.exists(wav) or os.path.getsize(wav) < 2000:
                    continue
                part = remote_engine.transcribe_wav(wav, whisper_lang)
                if not part:
                    _remote_fails += 1
                    continue
                _remote_ok += 1
                for seg in (part or {}).get('segments') or []:
                    text = (seg.get('text') or '').strip()
                    if not text or _is_boilerplate_hallucination(text):
                        continue
                    ns = float(seg.get('no_speech_prob', 0.0) or 0.0)
                    if ns > _ns_drop:
                        continue
                    if (whisper_lang and _wrong_script_for_language(text, whisper_lang)):
                        continue
                    # Shift slice-relative timestamps back to absolute video time.
                    seg['start_sec'] = round(float(seg.get('start_sec', 0) or 0) + ss, 3)
                    seg['end_sec'] = round(float(seg.get('end_sec', 0) or 0) + ss, 3)
                    for w in seg.get('words') or []:
                        for k in ('start', 'end'):
                            if isinstance(w.get(k), (int, float)):
                                w[k] = round(w[k] + ss, 3)
                    seg['source'] = 'gap_recovery'
                    seg['text'] = _collapse_repeated_phrases(text)
                    out.append(seg)
            except Exception:
                _remote_fails += 1
                continue
            finally:
                try:
                    os.remove(wav)
                except Exception:
                    pass
        log.log_stage('AUDIO',
            f'Gap recovery finished: {len(out)} cue(s) recovered from '
            f'{_remote_ok}/{len(picked)} slice(s)'
            + (f' ({_remote_fails} failed)' if _remote_fails else ''))
        return out

    def _redecode_difficult_segments(
        self, audio_path: str, segments: list, whisper_lang, log,
    ) -> int:
        """Second-chance decode for the hardest segments (audit Phase 3.4).

        Candidates, worst first, bounded to WHISPER_REDECODE_MAX_FRAC of
        the transcript:
          * hallucination-flagged segments (the filter may be reacting to a
            bad decode, not bad audio),
          * avg_logprob below WHISPER_REDECODE_LOGPROB,
          * degenerate word timestamps (the batched-inference word-timing
            failure mode) — these are re-timed by a sequential decode.

        Each candidate is re-decoded individually via ``clip_timestamps``
        with beam_size=WHISPER_REDECODE_BEAM and patience>1. The original
        entry is replaced in place only when the redecode is measurably
        better (higher avg_logprob and not itself a hallucination).
        Returns the number of segments replaced.
        """
        import inspect
        if self.engine is None:
            return 0
        params = {}
        try:
            params = inspect.signature(self.engine.transcribe).parameters
        except (TypeError, ValueError):
            return 0
        if 'clip_timestamps' not in params:
            return 0  # faster-whisper too old for windowed redecode

        logprob_floor = float(getattr(settings, 'WHISPER_REDECODE_LOGPROB', -0.8))
        max_frac = float(getattr(settings, 'WHISPER_REDECODE_MAX_FRAC', 0.10))
        vad_max_frac = float(getattr(
            settings, 'WHISPER_REDECODE_VAD_MAX_FRAC', 0.25))
        beam = int(getattr(settings, 'WHISPER_REDECODE_BEAM', 8))

        def _score(entry):
            # Lower = worse = redecode first
            return entry.get('avg_logprob', 0.0)

        candidates = []
        for i, entry in enumerate(segments):
            degenerate = _words_degenerate(
                entry.get('words') or [], entry['start_sec'], entry['end_sec'])
            if (entry.get('is_hallucination')
                    or entry.get('avg_logprob', 0.0) < logprob_floor
                    or degenerate):
                candidates.append((i, degenerate))
        if not candidates:
            return 0
        candidates.sort(key=lambda c: _score(segments[c[0]]))
        # VAD-confirmed phantom cues (the voice map says there IS speech under
        # them) get their own, larger budget: the redecode is their only path
        # back into the transcript, so they must never be crowded out by — or
        # crowd out — the ordinary low-logprob candidates.
        vad_confirmed = [c for c in candidates
                         if segments[c[0]].get('vad_confirmed_voice')]
        regular = [c for c in candidates
                   if not segments[c[0]].get('vad_confirmed_voice')]
        budget = max(1, int(len(segments) * max_frac))
        vad_budget = max(1, int(len(segments) * vad_max_frac)) if vad_confirmed else 0
        candidates = regular[:budget] + vad_confirmed[:vad_budget]
        candidates.sort(key=lambda c: _score(segments[c[0]]))

        _decode = _decoding_kwargs(self.engine.transcribe)
        # The redecode is the LAST chance before the polish LLM — spend
        # more search on it than the main pass.
        extra = {'beam_size': beam}
        if 'patience' in params:
            extra['patience'] = 1.5

        replaced = 0
        for idx, degenerate in candidates:
            entry = segments[idx]
            w_start = max(0.0, entry['start_sec'] - 0.2)
            w_end = entry['end_sec'] + 0.2
            if w_end - w_start < 0.15:
                continue
            try:
                seg_iter, _info = self.engine.transcribe(
                    audio_path,
                    language=whisper_lang,
                    vad_filter=False,
                    word_timestamps=True,
                    clip_timestamps=[w_start, w_end],
                    **extra,
                    **_decode,
                )
                new_segs = list(seg_iter)
            except Exception:
                continue
            if not new_segs:
                continue
            text = ' '.join(sg.text.strip() for sg in new_segs).strip()
            if not text or _is_boilerplate_hallucination(text):
                continue
            new_logprob = min(float(getattr(sg, 'avg_logprob', 0.0) or 0.0)
                              for sg in new_segs)
            # Keep the redecode when it is measurably more confident, or
            # when the original words were unusable (any valid re-timing
            # beats degenerate timestamps).
            if not degenerate and new_logprob <= entry.get('avg_logprob', 0.0) + 0.05:
                continue
            words = []
            for sg in new_segs:
                if getattr(sg, 'words', None):
                    for w in sg.words:
                        conf = getattr(w, 'probability', None)
                        if conf is None:
                            conf = getattr(w, 'confidence', 1.0) or 1.0
                        words.append({
                            'word': w.word.strip(),
                            'start': round(float(w.start), 3),
                            'end': round(float(w.end), 3),
                            'confidence': round(float(conf), 3),
                        })
            entry['text'] = text
            if words:
                entry['words'] = words
                entry['start_sec'] = round(words[0]['start'], 3)
                entry['end_sec'] = round(words[-1]['end'], 3)
            entry['avg_logprob'] = round(new_logprob, 3)
            entry['is_hallucination'] = False
            entry.pop('phantom', None)
            entry['redecoded'] = True
            replaced += 1
        return replaced

    def _gap_fill_pass(
        self, audio_path: str, primary_segments: list,
        duration_sec: float, whisper_lang, log,
    ) -> list:
        """Re-transcribe runs of audio the main pass left uncovered.

        Whisper-medium with VAD enabled silently drops:
          * soft / off-mic / whispered speech (VAD ``no_speech_prob``
            crosses threshold)
          * sung audio in opening / ending themes (Whisper trained
            mostly on spoken audio; high no_speech_prob on lyrics)
          * brief utterances under the 300 ms VAD silence break

        This helper finds runs of source audio ≥
        ``WHISPER_GAP_FILL_MIN_SEC`` seconds where the main pass
        produced no segment, then re-transcribes just those runs
        with:
          * ``vad_filter=False`` — VAD already said this was silence,
            don't ask it again
          * ``no_speech_threshold`` from
            ``WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD`` (default 0.25)
          * ``condition_on_previous_text=False`` — no priming from
            the (probably very different) main pass

        Each resulting segment is run through the same hallucination
        filter the main pass uses (boilerplate blocklist, repetition
        detection, no_speech_prob clamp) before being merged. Returns
        the list of clean gap-fill segment dicts ready to merge into
        the main ``segments`` list.
        """
        min_gap = float(getattr(settings, "WHISPER_GAP_FILL_MIN_SEC", 3.0))
        gap_ns_thresh = float(getattr(
            settings, "WHISPER_GAP_FILL_NO_SPEECH_THRESHOLD", 0.25))

        # Build the list of (start_sec, end_sec) gaps to re-transcribe.
        # Only consider clean (non-hallucination) segments as 'covered'.
        covered = sorted(
            (float(s['start_sec']), float(s['end_sec']))
            for s in primary_segments
            if not s.get('is_hallucination')
        )
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in covered:
            if start - cursor >= min_gap:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if duration_sec - cursor >= min_gap:
            gaps.append((cursor, duration_sec))

        if not gaps:
            log.log_stage('AUDIO',
                'Gap-fill: no gaps ≥ %.1fs — skipping' % min_gap)
            return []

        total_gap_sec = sum(e - s for s, e in gaps)

        # Mostly-non-speech guard: when the uncovered gaps dominate the video,
        # the content is music / silence and re-transcribing it with VAD off
        # invents far more than it recovers (on a ~70%-silent anime this added
        # 653 mostly-hallucinated cues). Trust the VAD main pass instead.
        _max_frac = float(getattr(settings, "WHISPER_GAP_FILL_MAX_FRACTION", 0.6))
        if _max_frac > 0 and duration_sec > 0 and (total_gap_sec / duration_sec) > _max_frac:
            log.log_stage('AUDIO',
                f'Gap-fill: gaps are {total_gap_sec / duration_sec:.0%} of the video '
                f'(> {_max_frac:.0%} cap) — content is mostly non-speech '
                '(music/silence). Skipping gap-fill to avoid flooding the '
                'transcript with hallucinated cues; the VAD main pass is more '
                'reliable on this material.')
            return []

        log.log_stage('AUDIO',
            f'Gap-fill: re-transcribing {len(gaps)} run(s) '
            f'totalling {total_gap_sec:.1f}s '
            f'(threshold={gap_ns_thresh})')

        # ``clip_timestamps`` accepts a flat list of seconds; pairs are
        # interpreted as (start, end). faster-whisper transcribes ONLY
        # those ranges and skips everything else, so the gap-fill pass
        # doesn't waste compute on the already-covered audio.
        clip_ts: list[float] = []
        for s, e in gaps:
            clip_ts.append(round(s, 3))
            clip_ts.append(round(e, 3))

        # Gap-fill lives in music / quiet regions, so force priming OFF
        # regardless of the global condition_on_previous_text default, and add
        # the same anti-repetition decoding guards the main pass uses.
        _gap_decode = _decoding_kwargs(
            self.engine.transcribe, condition_on_previous_text=False)
        try:
            segs_iter, _info = self.engine.transcribe(
                audio_path, language=whisper_lang,
                beam_size=5, vad_filter=False,
                word_timestamps=True,
                no_speech_threshold=gap_ns_thresh,
                clip_timestamps=clip_ts,
                **_gap_decode,
            )
        except TypeError:
            # Older faster-whisper builds don't accept clip_timestamps.
            # Fall back to a full-file second pass with the relaxed
            # threshold and dedupe overlaps below.
            log.log_stage('AUDIO',
                'Gap-fill: clip_timestamps unsupported — full-file fallback')
            segs_iter, _info = self.engine.transcribe(
                audio_path, language=whisper_lang,
                beam_size=5, vad_filter=False,
                word_timestamps=True,
                no_speech_threshold=gap_ns_thresh,
                **_gap_decode,
            )

        # Same boilerplate / repetition filter the main pass uses
        # (multilingual — see module-level _is_boilerplate_hallucination).

        # Build a fast O(log n) membership check for "is this gap-fill
        # segment actually inside a real gap?" — used to drop any
        # segment that leaked into already-covered territory on the
        # full-file fallback path.
        from bisect import bisect_right
        gap_starts = [g[0] for g in gaps]

        def _inside_gap(start_sec: float, end_sec: float) -> bool:
            i = bisect_right(gap_starts, start_sec) - 1
            if i < 0:
                return False
            gs, ge = gaps[i]
            # Allow segments that mostly overlap the gap (≥50 %).
            seg_len = max(0.001, end_sec - start_sec)
            overlap = max(0.0, min(end_sec, ge) - max(start_sec, gs))
            return overlap / seg_len >= 0.5

        # Hard ceiling on gap-fill segment length. With ``vad_filter=False``
        # Whisper can fold a whole OP song / minutes of narration into ONE
        # run-on cue stamped at a single timestamp; splitting at word
        # boundaries (and at inter-word silences) keeps each emitted cue
        # short and time-accurate.
        max_gap_seg = float(getattr(settings, "WHISPER_GAP_FILL_MAX_SEC", 8.0))
        GAP_BREAK = 1.0  # seconds of silence that forces a split

        def _word_dict(w):
            word_conf = (getattr(w, 'probability', None)
                         or getattr(w, 'confidence', 1.0) or 1.0)
            return {
                'word': (w.word.strip() if hasattr(w, 'word')
                         else str(w).strip()),
                'start': round(float(getattr(w, 'start', 0.0) or 0.0), 3),
                'end': round(float(getattr(w, 'end', 0.0) or 0.0), 3),
                'confidence': round(float(word_conf), 3),
            }

        out: list[dict] = []
        dropped_outside = 0
        for seg in segs_iter:
            text = (seg.text or '').strip()
            if not text:
                continue
            no_speech_prob = float(getattr(seg, 'no_speech_prob', 0.0) or 0.0)

            # Drop common Whisper hallucinations (multilingual)
            if _is_boilerplate_hallucination(text):
                continue
            # Very high no_speech confidence is true silence
            if no_speech_prob > 0.85:
                continue
            # Repetition check (lighter than the main pass — gap-fill
            # already lives in low-confidence territory)
            if len(text) > 20:
                words_list = text.lower().split()
                if len(words_list) >= 6:
                    for plen in range(2, 5):
                        if len(words_list) >= plen * 3:
                            pattern = tuple(words_list[:plen])
                            repeats = sum(
                                1 for i in range(0, len(words_list) - plen + 1, plen)
                                if tuple(words_list[i:i + plen]) == pattern
                            )
                            if repeats >= 3:
                                text = None
                                break
                    if text is None:
                        continue

            seg_start = float(seg.start)
            seg_end = float(seg.end)
            # Overlap guard — applied on BOTH the clip_timestamps path and the
            # full-file fallback path, so a segment that leaked into
            # already-covered territory is dropped regardless of which
            # transcribe call produced it.
            if not _inside_gap(seg_start, seg_end):
                dropped_outside += 1
                continue

            raw_words = list(seg.words) if (hasattr(seg, 'words') and seg.words) else []

            # ── Split over-long run-on cues into ≤ max_gap_seg sub-cues ──
            pieces: list[list] = []
            if max_gap_seg > 0 and (seg_end - seg_start) > max_gap_seg and raw_words:
                cur: list = []
                cur_start = None
                prev_end = None
                for w in raw_words:
                    w_start = float(getattr(w, 'start', seg_start) or seg_start)
                    w_end = float(getattr(w, 'end', w_start) or w_start)
                    if cur and (
                        (w_end - cur_start > max_gap_seg)
                        or (prev_end is not None and w_start - prev_end >= GAP_BREAK)
                    ):
                        pieces.append(cur)
                        cur = []
                        cur_start = None
                    if cur_start is None:
                        cur_start = w_start
                    cur.append(w)
                    prev_end = w_end
                if cur:
                    pieces.append(cur)

            if len(pieces) > 1:
                for ch in pieces:
                    # Reconstruct chunk text from the RAW word tokens so the
                    # original spacing (latin) / non-spacing (CJK) survives.
                    c_text = ''.join(
                        (w.word if hasattr(w, 'word') else str(w)) for w in ch
                    ).strip()
                    if not c_text:
                        continue
                    c_start = round(float(getattr(ch[0], 'start', seg_start) or seg_start), 3)
                    c_end = round(float(getattr(ch[-1], 'end', c_start) or c_start), 3)
                    out.append({
                        'start_sec': c_start,
                        'end_sec': c_end,
                        'text': c_text,
                        'words': [_word_dict(w) for w in ch],
                        'is_hallucination': False,
                        'no_speech_prob': round(no_speech_prob, 3),
                        'source': 'gap_fill',
                    })
            else:
                out.append({
                    'start_sec': round(seg_start, 3),
                    'end_sec': round(seg_end, 3),
                    'text': text,
                    'words': [_word_dict(w) for w in raw_words],
                    'is_hallucination': False,
                    'no_speech_prob': round(no_speech_prob, 3),
                    'source': 'gap_fill',
                })

        # Drop low-confidence phantom cues the gap-fill pass invents over
        # silence / music. Gap-fill only keeps segments BELOW its no_speech
        # threshold, so these phantoms have LOW no_speech_prob and the main
        # per-segment no_speech clamp can't catch them — the word-confidence
        # signal is the discriminator. Same gate the main pass uses, with the
        # no_speech requirement off (min_no_speech=0.0) since it doesn't apply
        # here. This is what removes the gap-fill flood on quiet/musical content.
        if bool(getattr(settings, "WHISPER_PHANTOM_FILTER_ENABLED", True)) and out:
            from backend.services.transcript_dedup import is_low_confidence_phantom
            _ph_max = float(getattr(settings, "WHISPER_PHANTOM_MAX_AVG_CONF", 0.40))
            _ph_frac = float(getattr(settings, "WHISPER_PHANTOM_MIN_LOWCONF_FRAC", 0.80))
            _kept = [s for s in out if not is_low_confidence_phantom(
                s.get('words') or [], s.get('no_speech_prob', 0.0),
                max_avg_conf=_ph_max, min_lowconf_frac=_ph_frac, min_no_speech=0.0)]
            _dropped_ph = len(out) - len(_kept)
            if _dropped_ph:
                log.log_stage('AUDIO',
                    f'Gap-fill: dropped {_dropped_ph} low-confidence phantom cue(s)')
                out = _kept

        # Collapse near-duplicate re-transcriptions (overlapping in time AND
        # similar in text — e.g. the repeated 作戦名オペレーション・メテオ) that the
        # adjacent-only filters miss.
        from backend.services.transcript_dedup import collapse_overlapping_duplicates
        out, dropped_dup = collapse_overlapping_duplicates(
            out, start_key='start_sec', end_key='end_sec')

        max_seg_len = max((s['end_sec'] - s['start_sec'] for s in out), default=0.0)
        log.log_stage('AUDIO',
            f'Gap-fill output: {len(out)} segment(s) kept, '
            f'{dropped_outside} dropped-outside-gap, '
            f'{dropped_dup} dropped-as-duplicate, '
            f'max segment length {max_seg_len:.1f}s')
        return out

    def _reload_on_cpu(self) -> None:
        """Reload Whisper on CPU int8 — used when CUDA runs out of memory.
        Low-VRAM GPUs (the GTX 1650 = 4 GB, for instance) can't fit Whisper
        medium under batched inference; CPU is slower but always completes."""
        # The previously loaded GPU engine is referenced from BOTH ``self``
        # and the class-level cache. Dropping only ``self.engine`` leaves
        # the cache alive, so ``empty_cache()`` finds nothing to free and
        # the GPU model lingers for the rest of the process. Clear both
        # before flushing the allocator.
        try:
            import gc as _gc
            AudioIntelligence._cached_engine = None
            AudioIntelligence._cached_model_name = None
            AudioIntelligence._cached_device = None
            if hasattr(self, 'engine'):
                try:
                    del self.engine
                except Exception:
                    pass
                self.engine = None
            _gc.collect()
            _gc.collect()
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
                _torch.cuda.empty_cache()
        except Exception:
            pass
        from faster_whisper import WhisperModel
        self.engine = WhisperModel(self.model_name, device='cpu', compute_type='int8')
        self.device_used = 'cpu_int8'
        # Re-cache the CPU engine so the next Perceiver in this process
        # doesn't load yet another copy on top of the GPU one we just freed.
        AudioIntelligence._cached_engine = self.engine
        AudioIntelligence._cached_model_name = self.model_name
        AudioIntelligence._cached_device = self.device_used
        AudioIntelligence._record_loaded(self.model_name, self.device_used)
        log.log_stage('AUDIO', f'Whisper reloaded on CPU int8 ({self.model_name})')

    def _remote_translate(self, video_path: str, source_lang: str, log) -> List[dict]:
        """Run audio→English translate on the PRIMARY remote whisper GPU (the
        paired Companion). Extracts the WAV locally and uploads it with
        ``translate=true``. Returns local-schema segments WITH word timestamps,
        or ``[]`` to fall back to the local engine — e.g. when the remote host
        can't return word-level timing, which tier-A projection requires."""
        import subprocess as _sp
        import tempfile as _tf
        audio_path = os.path.join(_tf.gettempdir(), 'clipai_remote_translate.wav')
        try:
            _sp.run(['ffmpeg', '-y', '-i', video_path,
                     '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                     audio_path],
                    capture_output=True, timeout=180)
        except Exception as e:
            log.log_error('TRANSLATE', f'Remote translate audio extract failed: {e}')
            return []
        try:
            log.log_stage('TRANSLATE',
                f'Whisper native translate on the Companion GPU: '
                f'{source_lang or "auto"} → en (remote)')
            engine = RemoteWhisperEngine(model="")
            whisper_lang = source_lang if source_lang and source_lang != 'auto' else None
            result = engine.transcribe_wav(audio_path, language=whisper_lang, translate=True)
            if not result:
                return []
            segs = result.get('segments') or []
            # Tier-A projection needs word timestamps. If the host returned none,
            # fall back to the local engine (which always produces them).
            has_words = any(
                (s.get('words') if isinstance(s, dict) else None) for s in segs)
            if not has_words:
                log.log_stage('TRANSLATE',
                    'Remote translate returned no word timestamps — local '
                    'fallback for tier-A timing')
                return []
            out = []
            for s in segs:
                if not isinstance(s, dict):
                    continue
                out.append({
                    'start_sec': round(float(s.get('start_sec', s.get('start', 0.0))), 3),
                    'end_sec': round(float(s.get('end_sec', s.get('end', 0.0))), 3),
                    'text': (s.get('text') or '').strip(),
                    'words': s.get('words') or [],
                })
            log.log_stage('TRANSLATE',
                f'Remote translate complete on the Companion: {len(out)} segments')
            return out
        except Exception as e:
            log.log_error('TRANSLATE', f'Remote translate failed: {e}')
            return []
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass

    def whisper_translate(self, video_path: str, source_lang: str = None,
                          on_progress=None, reuse_loaded: bool = False) -> List[dict]:
        """Direct audio→English translation via Whisper's native translate task.

        For non-English → English, this single-step pass (Whisper run with
        ``task="translate"``) is the preferred OFFLINE path: it avoids the
        transcribe-then-translate double-error, needs no separate NMT model
        download, and never touches an LLM (the AI model only polishes the
        result afterward). The pipeline calls this first for English targets
        and falls back to the offline NMT engines in
        ``backend/services/translator.py`` only when it returns ``[]``.

        Returns a list of ``{start, end, text}`` dicts (English, with Whisper's
        own audio-aligned timing), or ``[]`` when the engine is unavailable.
        """
        log = get_logger()

        # PRIMARY-GPU routing: when the paired Companion serves whisper AND it
        # advertises the translate task, run audio→English on THAT GPU (e.g. the
        # 4070) — the whole point of pairing it, and what lets long videos reach
        # tier-A timing without the slow local pass. Falls back to the local
        # engine below when the remote path yields no word-timestamped segments.
        if self.device_used == 'remote' or isinstance(self.engine, RemoteWhisperEngine):
            if remote_whisper_configured() and remote_whisper_supports_translate():
                remote_segs = self._remote_translate(video_path, source_lang, log)
                if remote_segs:
                    return remote_segs
                log.log_stage('TRANSLATE',
                    'Remote translate unusable — falling back to the local engine')
            else:
                # No remote translate capability — the native translate task
                # needs a REAL local faster-whisper engine (batched decode with
                # task="translate"). Swap in the local ladder just for this pass.
                log.log_stage('TRANSLATE',
                    'Remote Whisper active — loading the LOCAL engine for the '
                    'native translate task')
            self.engine = None
            self.available = False
            self.device_used = 'unknown'
            if not self.try_load(force_local=True):
                log.log_stage('TRANSLATE',
                    'Local Whisper unavailable — cannot use native translate')
                return []

        if not self.engine:
            log.log_stage('TRANSLATE',
                'Whisper not loaded — cannot use native translate')
            return []

        # Extract audio
        import subprocess as _sp
        import tempfile as _tf
        audio_path = os.path.join(_tf.gettempdir(), 'clipai_translate_audio.wav')
        try:
            _sp.run(['ffmpeg', '-y', '-i', video_path,
                     '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                     audio_path],
                    capture_output=True, timeout=120)
        except Exception as e:
            log.log_error('TRANSLATE', f'Audio extraction failed: {e}')
            return []

        log.log_stage('TRANSLATE',
            f'Whisper native translate: {source_lang or "auto"} → en '
            f'(direct audio→English, single-step)')

        try:
            # Use the same model but with task="translate"
            whisper_lang = source_lang if source_lang and source_lang != 'auto' else None

            # Pre-flight VRAM check — mirrors transcribe(). On low-VRAM GPUs,
            # batched translate OOMs just as readily as batched transcribe.
            # When REUSING an already-loaded model, no reload headroom is needed
            # (the weights are already resident) — only inference activations —
            # so allow a lower free-VRAM floor before dropping to the slow CPU
            # path. This is what lets a 4 GB card translate on the GPU via reuse;
            # a genuine OOM below still falls back to CPU cleanly.
            if self.device_used.startswith('cuda'):
                try:
                    import torch as _torch
                    if _torch.cuda.is_available():
                        _free_bytes, _ = _torch.cuda.mem_get_info()
                        _free_gb = _free_bytes / 1_073_741_824
                        _floor = 1.5 if reuse_loaded else 3.0
                        if _free_gb < _floor:
                            log.log_stage('TRANSLATE',
                                f'Only {_free_gb:.1f} GB VRAM free (< {_floor:.1f}) '
                                '— using CPU for translation')
                            self._reload_on_cpu()
                except Exception:
                    pass

            try:
                from faster_whisper import BatchedInferencePipeline
                batched = BatchedInferencePipeline(model=self.engine)
                _decode = _decoding_kwargs(batched.transcribe)
                segments_iter, info = batched.transcribe(
                    audio_path, batch_size=self._batch_size,
                    language=whisper_lang,
                    task='translate',  # ← the key difference
                    beam_size=_beam_size(), vad_filter=True,
                    vad_parameters=_vad_parameters(),
                    word_timestamps=True,
                    no_speech_threshold=float(getattr(
                        settings, "WHISPER_NO_SPEECH_THRESHOLD", 0.4)),
                    **_decode,
                )
            except Exception as e:
                err_str = str(e)
                if self.device_used.startswith('cuda') and (
                    'out of memory' in err_str.lower()
                    or 'cublas' in err_str.lower()
                    or 'cuda' in err_str.lower()
                ):
                    log.log_stage('TRANSLATE',
                        'CUDA out of memory — reloading Whisper on CPU')
                    self._reload_on_cpu()
                # Fallback to sequential
                segments_iter, info = self.engine.transcribe(
                    audio_path, language=whisper_lang,
                    task='translate',
                    beam_size=_beam_size(), vad_filter=True,
                    word_timestamps=True,
                    **_decoding_kwargs(self.engine.transcribe),
                )

            segments = []
            for seg in segments_iter:
                words = []
                if hasattr(seg, 'words') and seg.words:
                    for w in seg.words:
                        words.append({
                            'word': w.word.strip() if hasattr(w, 'word') else str(w).strip(),
                            'start': round(getattr(w, 'start', seg.start), 3),
                            'end': round(getattr(w, 'end', seg.end), 3),
                        })
                segments.append({
                    'start_sec': round(seg.start, 3),
                    'end_sec': round(seg.end, 3),
                    'text': seg.text.strip(),
                    'words': words,
                })

                if on_progress and info.duration > 0:
                    on_progress(min(1.0, seg.end / info.duration))

            # Clamp drift to the real audio end (task='translate' loops on music
            # too) so the English cues never run past the video.
            try:
                from backend.services.transcript_dedup import clamp_segments_to_duration
                if getattr(info, 'duration', 0):
                    segments, _drifted = clamp_segments_to_duration(
                        segments, float(info.duration),
                        start_key='start_sec', end_key='end_sec')
                    if _drifted:
                        log.log_stage('TRANSLATE',
                            f'Timestamp-drift clamp @ {info.duration:.0f}s: '
                            f'dropped/clamped {_drifted} cue(s)')
            except Exception:
                pass

            log.log_stage('TRANSLATE',
                f'Whisper translate complete: {len(segments)} segments '
                f'(direct {source_lang or "auto"} → en)')

            return segments

        except Exception as e:
            log.log_error('TRANSLATE', f'Whisper translate failed: {e}')
            return []
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass


