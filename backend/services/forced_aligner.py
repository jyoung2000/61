"""CTC forced-alignment refinement for Whisper word timestamps.

Whisper word timestamps drift 50-200 ms — the visible difference from
Netflix-grade cueing. This module re-aligns each segment's words against
the audio with a CTC forced aligner and snaps word (and cue) boundaries
to actual speech onset/offset.

Backends, tried in order:
  1. ``ctc_forced_aligner`` package (multilingual, int8 wav2vec2) when
     installed.
  2. ``torchaudio`` bundled wav2vec2 CTC (WAV2VEC2_ASR_BASE_960H,
     ~360 MB, English only) via ``torchaudio.functional.forced_align``.

Both run in <1 GB VRAM; the GPU is only used when >1.5 GB is free (a
GTX 1650 mid-pipeline usually isn't), else CPU — alignment is cheap
compared to decoding.

Everything is fail-safe: any missing dependency, model download failure
or per-segment error leaves the original Whisper timestamps untouched.

Gated by ``SUBTITLE_FORCED_ALIGN`` (default ON).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)

# Maximum |shift| we accept from the aligner. A shift larger than this
# means the aligner mis-anchored (music, crosstalk) — keep Whisper's time.
_MAX_SHIFT_S = 0.6


def _pick_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            if free_b / 1_073_741_824 > 1.5:
                return "cuda"
    except Exception:
        pass
    return "cpu"


_NON_ALPHA = re.compile(r"[^a-z']+")


def _normalize_en(word: str) -> str:
    return _NON_ALPHA.sub("", word.lower())


class _TorchaudioAligner:
    """English CTC aligner on torchaudio's bundled wav2vec2."""

    def __init__(self, device: str):
        import torch  # noqa: F401
        import torchaudio
        from torchaudio.functional import forced_align  # noqa: F401
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().to(device).eval()
        self.labels = bundle.get_labels()
        self.sample_rate = bundle.sample_rate
        self.device = device
        self.dictionary = {c.lower(): i for i, c in enumerate(self.labels)}
        self.blank_id = 0

    def align_words(self, waveform, words: list[str]) -> Optional[list[tuple[float, float]]]:
        """Return [(start_s, end_s)] per word within the waveform, or None."""
        import torch
        from torchaudio.functional import forced_align, merge_tokens

        tokens = []
        spans = []  # (first_token_idx, n_tokens) per word
        for w in words:
            norm = _normalize_en(w)
            ids = [self.dictionary.get(c) for c in norm]
            ids = [i for i in ids if i is not None]
            if not ids:
                return None  # unalignable token (number/symbol) — bail
            spans.append((len(tokens), len(ids)))
            tokens.extend(ids)
            tokens.append(self.dictionary.get("|"))
        if tokens and tokens[-1] == self.dictionary.get("|"):
            tokens = tokens[:-1]
            spans[-1] = (spans[-1][0], spans[-1][1])

        with torch.inference_mode():
            emission, _ = self.model(waveform.to(self.device))
            targets = torch.tensor([tokens], dtype=torch.int32,
                                   device=emission.device)
            aligned, scores = forced_align(
                torch.log_softmax(emission, dim=-1), targets,
                blank=self.blank_id)
            token_spans = merge_tokens(aligned[0], scores[0])

        # merge_tokens gives per-token frame spans (blanks removed).
        # Frames → seconds: frame duration = waveform_len / n_frames / sr
        n_frames = emission.shape[1]
        frame_s = waveform.shape[1] / n_frames / self.sample_rate
        # Non-separator tokens in order correspond to our `tokens` list
        # entries that aren't "|"
        sep_id = self.dictionary.get("|")
        tok_times = []  # (start_s, end_s) per non-sep token
        ti = 0
        for span in token_spans:
            if span.token == sep_id:
                continue
            tok_times.append((span.start * frame_s, span.end * frame_s))
            ti += 1

        # Map back to words: spans hold indices into the tokens list
        # counting separators, so recompute per-word token counts.
        out = []
        cursor = 0
        for w in words:
            n = len([c for c in _normalize_en(w)
                     if self.dictionary.get(c) is not None])
            if cursor + n > len(tok_times) or n == 0:
                return None
            out.append((tok_times[cursor][0], tok_times[cursor + n - 1][1]))
            cursor += n
        return out


_aligner_cache: dict = {}


def _get_backend(language: str, device: str):
    """Resolve an aligner backend or None. Cached per (backend, device)."""
    # 1. ctc-forced-aligner package (multilingual)
    key = ("ctc_fa", device)
    if key not in _aligner_cache:
        try:
            from ctc_forced_aligner import (  # noqa: F401
                load_alignment_model, generate_emissions,
                get_alignments, get_spans, preprocess_text,
                postprocess_results,
            )
            _aligner_cache[key] = "ctc_fa"
        except Exception:
            _aligner_cache[key] = None
    if _aligner_cache[key]:
        return _aligner_cache[key]

    # 2. torchaudio wav2vec2 — English only
    if not (language or "").startswith("en"):
        return None
    key = ("torchaudio", device)
    if key not in _aligner_cache:
        try:
            _aligner_cache[key] = _TorchaudioAligner(device)
        except Exception as e:
            logger.info("Forced alignment unavailable (torchaudio: %s)", e)
            _aligner_cache[key] = None
    return _aligner_cache[key]


def align_translated_cues(audio_path: str, cues: list) -> dict:
    """Force-align ENGLISH subtitle cues against the audio, in place.

    ``refine_word_timestamps`` below is written for the reframer's segment shape
    (``start_sec``/``end_sec`` dicts) and for REFINING timings that are already
    roughly right — it rejects a whole segment when any word moves more than
    ``_MAX_SHIFT_S``. Neither fits the shipped subtitle track:

      * subtitle cues are ``TranscriptSegment``-shaped (``start``/``end``);
      * only about half of them carry real audio word times. The rest are
        distributed across the cue window by character width, so their words are
        routinely more than 0.6 s from the truth — exactly the cues that most
        need aligning are the ones the refine guard throws out.

    So this entry point keeps the cue WINDOW as the trusted anchor (it came from
    the ASR/timing tiers) and places the words inside it from the audio, with no
    prior-shift veto. It also pulls the cue's own start back to its first voiced
    word, which is what stops a cue appearing before the speech it captions —
    bounded by ``max_cue_shift_s`` so a mis-anchored cue can't wander.

    Returns stats; never raises. On any failure the caller's timings stand."""
    stats = {"enabled": False, "backend": None, "cues_aligned": 0,
             "words_aligned": 0, "mean_shift_ms": 0.0, "starts_tightened": 0,
             "ends_extended": 0}
    if not bool(getattr(settings, "SUBTITLE_FORCED_ALIGN", True)):
        return stats
    if not cues or not audio_path:
        return stats

    max_cue_shift = float(getattr(settings, "SUBTITLE_ALIGN_MAX_CUE_SHIFT_S", 0.75))
    min_dur_s = float(getattr(settings, "SUBTITLE_MIN_DURATION_MS", 833)) / 1000.0
    device = _pick_device()
    # Time the backend acquisition separately from the per-cue work: the first
    # call after an image rebuild DOWNLOADS the ~360 MB wav2vec2 checkpoint
    # with no log output at all — a measured run sat 5 minutes inside "Cue
    # alignment" on a step that takes ~25 s warm, and nothing in the log said
    # why. (TORCH_HOME now points into the persisted model volume so this
    # should happen at most once per host; the log proves it either way.)
    import time as _time
    _t0 = _time.monotonic()
    try:
        backend = _get_backend("en", device)
    except Exception as e:
        logger.info("Cue alignment skipped: %s", e)
        return stats
    _load_s = _time.monotonic() - _t0
    if _load_s > 15.0:
        logger.warning(
            "Cue alignment: backend took %.0f s to become ready — the wav2vec2 "
            "checkpoint was (re)downloaded. Persist TORCH_HOME "
            "(%s) across container rebuilds to avoid this stall.",
            _load_s, os.environ.get("TORCH_HOME", "~/.cache/torch"))
    if backend is None or backend == "ctc_fa":
        return stats

    try:
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != backend.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, backend.sample_rate)
            sr = backend.sample_rate
    except Exception as e:
        logger.info("Cue alignment: audio load failed (%s)", e)
        return stats

    stats["enabled"] = True
    stats["backend"] = f"torchaudio/{device}"
    pad = 0.30
    total_shift = 0.0

    def _get(c, k, d=None):
        return (c.get(k, d) if isinstance(c, dict) else getattr(c, k, d))

    def _set(c, k, v):
        if isinstance(c, dict):
            c[k] = v
        else:
            setattr(c, k, v)

    # Each cue's successor start, so the end-extension below can borrow only
    # genuinely idle time. Computed over a sorted view — the caller's list is
    # normally chronological, but nothing here should depend on it.
    def _f(c):
        try:
            return float(_get(c, "start", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    _order = sorted(range(len(cues)), key=lambda k: _f(cues[k]))
    _next_start = {}
    for _k, _idx in enumerate(_order):
        _next_start[_idx] = (_f(cues[_order[_k + 1]])
                             if _k + 1 < len(_order) else None)

    for _i_cue, cue in enumerate(cues):
        text = str(_get(cue, "text", "") or "")
        tokens = text.split()
        if not tokens or not all(_normalize_en(t) for t in tokens):
            continue
        try:
            c_s = float(_get(cue, "start", 0.0) or 0.0)
            c_e = float(_get(cue, "end", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if c_e - c_s < 0.10:
            continue
        s0 = max(0.0, c_s - pad)
        a, b = int(s0 * sr), int((c_e + pad) * sr)
        if a >= waveform.shape[1] or b - a < sr // 10:
            continue
        try:
            spans = backend.align_words(
                waveform[:, a:min(b, waveform.shape[1])], tokens)
        except Exception:
            spans = None
        if not spans or len(spans) != len(tokens):
            continue
        # The cue window itself can be WRONG — tier B/C placement squeezes
        # some cues well under their real voiced extent, and the guillotined
        # end is what makes a subtitle vanish while its line is still being
        # spoken (and fakes an over-CPS reading that the splitter then
        # "fixes"). When the aligner hears the last word running past the
        # cue's end, extend the end to the voiced extent — bounded by the
        # padded window it actually listened to and by the next cue's start,
        # so it can only ever borrow idle time.
        e_eff = c_e
        _last_we = s0 + float(spans[-1][1])
        if _last_we > c_e + 0.04:
            _nxt = _next_start.get(_i_cue)
            # Stop ONE frame short of the successor, not two: the 0.084 bound
            # doubled the track's median inter-cue gap (0.042 → 0.083) because
            # every extended end parked 2 frames early. 0.043 ≈ one 23.976-fps
            # frame + ε, the reference track's own cadence.
            _room = (_nxt - 0.043) if _nxt is not None else _last_we + 0.15
            e_eff = max(c_e, min(_last_we + 0.10, c_e + pad, _room))
        rows = []
        prev_end = None
        ok = True
        for tok, (ws, we) in zip(tokens, spans):
            # Clamp back into the (possibly extended) cue. Alignment runs over
            # a padded window, so a row could otherwise end up to ``pad`` past
            # the cue's own end — which crosses the one-frame inter-cue gap
            # into the NEXT cue, makes a later merge produce a non-monotonic
            # array, and lets the burn-in extend the cue beyond what the SRT
            # says.
            n_s = round(min(max(s0 + ws, c_s), e_eff), 3)
            n_e = round(min(max(s0 + we, c_s), e_eff), 3)
            if n_e <= n_s or (prev_end is not None and n_s < prev_end - 0.05):
                ok = False       # non-monotonic → the aligner lost the thread
                break
            rows.append({"word": tok, "start": n_s, "end": n_e})
            prev_end = n_e
        if not ok:
            continue
        if e_eff > c_e + 0.02:
            _set(cue, "end", round(e_eff, 3))
            c_e = e_eff
            stats["ends_extended"] += 1
        # Accumulate how far the words moved, for the log line.
        for old, new in zip(_iter_word_starts(cue), rows):
            total_shift += abs(new["start"] - old)
        _set(cue, "words", _as_word_rows(cue, rows))
        # Tighten the cue onto its own speech: pull the start up to the first
        # voiced word (never push it later than the audio), bounded so a
        # mis-anchor cannot move the cue far.
        # Never tighten a cue below the minimum display duration. A 0.10 s floor
        # only guaranteed start < end: a 1.0 s cue whose speech began 0.70 s in
        # became a 0.30 s cue, and the readability pass then restored the minimum
        # by pushing the END out — leaving the cue on screen 0.53 s LONGER than
        # the window the timing tiers set, the opposite of the intent. Skip the
        # tightening rather than shrink past the floor.
        new_start = rows[0]["start"]
        cand = min(new_start, c_e - min_dur_s)
        if c_s < new_start - 0.02 and (new_start - c_s) <= max_cue_shift \
                and cand > c_s + 0.02:
            _set(cue, "start", round(cand, 3))
            stats["starts_tightened"] += 1
        stats["cues_aligned"] += 1
        stats["words_aligned"] += len(rows)

    if stats["words_aligned"]:
        stats["mean_shift_ms"] = round(
            total_shift / stats["words_aligned"] * 1000, 1)
    stats["elapsed_s"] = round(_time.monotonic() - _t0, 1)
    return stats


def _iter_word_starts(cue) -> list:
    raw = (cue.get("words") if isinstance(cue, dict) else getattr(cue, "words", None)) or []
    out = []
    for w in raw:
        v = w.get("start") if isinstance(w, dict) else getattr(w, "start", None)
        out.append(float(v) if v is not None else 0.0)
    return out


def _as_word_rows(cue, rows: list):
    """Word rows in whatever shape ``cue`` already stores — a dict row keeps
    dicts, a pydantic segment keeps ``WordTimestamp`` (assignment is not
    validated, so bare dicts there would break every ``w.end`` consumer)."""
    if isinstance(cue, dict):
        return rows
    try:
        from backend.models import WordTimestamp
        return [WordTimestamp(**r) for r in rows]
    except Exception:
        return rows


def refine_word_timestamps(audio_path: str, segments: list,
                           language: str = "en") -> dict:
    """Refine ``segments[i]['words']`` in place against the audio.

    Returns stats: {'enabled', 'backend', 'segments_aligned',
    'words_aligned', 'mean_shift_ms'}. Never raises; on any failure the
    original timestamps are left untouched.
    """
    stats = {"enabled": False, "backend": None,
             "segments_aligned": 0, "words_aligned": 0,
             "mean_shift_ms": 0.0}
    if not bool(getattr(settings, "SUBTITLE_FORCED_ALIGN", True)):
        return stats
    if not segments:
        return stats

    device = _pick_device()
    backend = None
    try:
        backend = _get_backend(language, device)
    except Exception as e:
        logger.info("Forced alignment skipped: %s", e)
    if backend is None:
        return stats
    if backend == "ctc_fa":
        # The package's high-level API operates on the whole file; its
        # integration is heavier and it duplicates what the torchaudio
        # path does — defer to per-segment torchaudio when English, else
        # skip (multilingual support tracked as follow-up).
        backend = _get_backend("en", device) if (language or "").startswith("en") else None
        if backend in (None, "ctc_fa"):
            return stats

    stats["enabled"] = True
    stats["backend"] = f"torchaudio/{device}"

    try:
        import torch
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != backend.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, backend.sample_rate)
            sr = backend.sample_rate
    except Exception as e:
        logger.info("Forced alignment: audio load failed (%s)", e)
        return stats

    total_shift = 0.0
    pad = 0.25  # context around the Whisper segment window
    for entry in segments:
        words = entry.get("words") or []
        if not words or entry.get("is_hallucination"):
            continue
        texts = [w.get("word", "") for w in words]
        if not all(_normalize_en(t) for t in texts):
            continue
        s0 = max(0.0, float(entry["start_sec"]) - pad)
        s1 = float(entry["end_sec"]) + pad
        a, b = int(s0 * sr), int(s1 * sr)
        if b - a < sr // 10 or a >= waveform.shape[1]:
            continue
        chunk = waveform[:, a:min(b, waveform.shape[1])]
        try:
            spans = backend.align_words(chunk, texts)
        except Exception:
            spans = None
        if not spans or len(spans) != len(words):
            continue
        seg_shift = 0.0
        ok = True
        for w, (ws, we) in zip(words, spans):
            new_s = round(s0 + ws, 3)
            new_e = round(s0 + we, 3)
            if abs(new_s - w["start"]) > _MAX_SHIFT_S or new_e <= new_s:
                ok = False
                break
            seg_shift += abs(new_s - w["start"])
        if not ok:
            continue
        for w, (ws, we) in zip(words, spans):
            w["start"] = round(s0 + ws, 3)
            w["end"] = round(s0 + we, 3)
        entry["start_sec"] = round(words[0]["start"], 3)
        entry["end_sec"] = round(max(words[-1]["end"], words[0]["start"] + 0.05), 3)
        entry["forced_aligned"] = True
        stats["segments_aligned"] += 1
        stats["words_aligned"] += len(words)
        total_shift += seg_shift

    if stats["words_aligned"]:
        stats["mean_shift_ms"] = round(
            total_shift / stats["words_aligned"] * 1000, 1)
    return stats
