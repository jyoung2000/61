"""Spoken-language identification via SpeechBrain VoxLingua107.

Whisper's own language detection — and the paired GPU Companion's whisper.cpp
``large-v3-turbo`` especially — mis-reads breathy / sparse-dialogue / music-heavy
audio. A 128-minute Japanese video was repeatedly auto-detected as ``english``
and then hallucinated into English cues over the JA audio: forcing the wrong
language makes Whisper invent Latin-script text, most of which the downstream
no-speech / phantom filters then drop, so the transcript both reads as garbage
English AND skips most of the runtime.

The fix is to identify the SPOKEN language WITHOUT relying on a Whisper decode.
VoxLingua107 is a dedicated ECAPA-TDNN language classifier (107 languages) that
reads the language straight from the audio embedding, so it is far more robust
on hard audio than a transcription-decode guess. We classify several spread-out
windows and take a confidence-weighted majority.

Everything heavy (torch / torchaudio / speechbrain) is imported lazily and every
stage is guarded: if the model or its deps are unavailable, ``identify`` returns
``(None, reason)`` and the caller keeps Whisper's own auto-detect — no
regression, the pipeline never fails because language ID was unavailable.

SpeechBrain is already a project dependency (the local ECAPA diarizer uses the
same stack), and this mirrors that module's proven load pattern.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Pure helpers (unit-testable without torch / speechbrain) ───────────────

# A few full-name fallbacks in case a build returns the language name instead
# of the ISO code before the colon. VoxLingua107 normally returns "ja: Japanese".
_NAME_TO_ISO = {
    "japanese": "ja", "english": "en", "chinese": "zh", "mandarin": "zh",
    "cantonese": "yue", "korean": "ko", "spanish": "es", "french": "fr",
    "german": "de", "italian": "it", "portuguese": "pt", "russian": "ru",
    "arabic": "ar", "hindi": "hi", "thai": "th", "vietnamese": "vi",
    "indonesian": "id", "dutch": "nl", "polish": "pl", "turkish": "tr",
}


def iso_from_label(label) -> Optional[str]:
    """Map a VoxLingua107 label ("ja: Japanese", "en", "Japanese") to an ISO-639
    code. Returns None when nothing usable can be extracted."""
    if not label:
        return None
    s = str(label).strip()
    code = s.split(":", 1)[0].strip().lower() if ":" in s else s.strip().lower()
    if not code:
        return None
    if code in _NAME_TO_ISO:
        return _NAME_TO_ISO[code]
    # VoxLingua107 codes are ISO-639-1 (2-letter) with a couple of 3-letter
    # exceptions (e.g. "yue"); keep 3-letter codes we know, else take 2.
    if code in ("yue",):
        return code
    return code[:2] if len(code) >= 2 else None


def window_starts(duration_s: int, n_windows: int) -> List[int]:
    """Start offsets (seconds) for the probe windows: skip the intro/logo and
    spread the rest across the body of the video, where dialogue actually is."""
    dur = max(1, int(duration_s or 0))
    if dur <= 90:
        return [max(0, dur // 3)]
    n = max(2, int(n_windows))
    span_lo, span_hi = 0.08, 0.92  # skip the first/last ~8%
    return [int(dur * (span_lo + (span_hi - span_lo) * i / (n - 1)))
            for i in range(n)]


def vote(observations: List[Tuple[Optional[str], float]]) -> Tuple[Optional[str], str]:
    """Confidence-weighted majority over per-window ``(iso, confidence)`` pairs.

    Noise / music windows scatter low-confidence votes across languages while
    real dialogue windows concentrate high-confidence votes on the spoken
    language, so the true language wins even when several windows miss speech.
    Returns ``(iso, detail)`` or ``(None, detail)``.
    """
    votes: dict = {}
    for iso, conf in observations:
        if not iso:
            continue
        votes[iso] = votes.get(iso, 0.0) + max(0.0, float(conf or 0.0))
    if not votes:
        return None, f"no confident windows ({len(observations)} probed)"
    top_iso, _ = max(votes.items(), key=lambda kv: kv[1])
    detail = ", ".join(f"{k}:{v:.2f}"
                       for k, v in sorted(votes.items(), key=lambda kv: -kv[1]))
    return top_iso, f"{detail} over {len(observations)} window(s)"


# ── Model-backed identifier ────────────────────────────────────────────────

class SpokenLanguageIdentifier:
    """Multi-window spoken-language ID with a cached VoxLingua107 model."""

    _model = None  # class-cached EncoderClassifier (loaded once per process)
    WINDOW_SECONDS = 30  # wide enough to catch dialogue in sparse-speech content

    def __init__(self):
        from backend.config import settings as _settings
        self.model_id = getattr(
            _settings, "SPOKEN_LANGUAGE_ID_MODEL",
            "speechbrain/lang-id-voxlingua107-ecapa")
        self.n_windows = int(getattr(_settings, "SPOKEN_LANGUAGE_ID_WINDOWS", 8))

    @staticmethod
    def _has_dependencies() -> bool:
        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            import speechbrain  # noqa: F401
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        from backend.config import settings as _settings
        if not getattr(_settings, "SPOKEN_LANGUAGE_ID_ENABLED", True):
            return False
        return self._has_dependencies()

    def _savedir(self) -> str:
        """Persistent dir for the fetched model (shares the models mount with
        the NMT / ECAPA models so it's cached across container restarts)."""
        try:
            from backend.services.nmt_translator import _models_dir
            return os.path.join(_models_dir(), "voxlingua")
        except Exception:
            d = os.path.join(tempfile.gettempdir(), "clipai_voxlingua")
            os.makedirs(d, exist_ok=True)
            return d

    def _load_model(self):
        if SpokenLanguageIdentifier._model is not None:
            return SpokenLanguageIdentifier._model
        # EncoderClassifier moved across speechbrain versions (pretrained →
        # inference.classifiers in 1.0, with a top-level re-export). Try each
        # known location so a version skew never costs us language ID.
        EncoderClassifier = None
        for _imp in (
            lambda: __import__("speechbrain.inference.classifiers",
                               fromlist=["EncoderClassifier"]).EncoderClassifier,
            lambda: __import__("speechbrain.inference",
                               fromlist=["EncoderClassifier"]).EncoderClassifier,
            lambda: __import__("speechbrain.pretrained",
                               fromlist=["EncoderClassifier"]).EncoderClassifier,
        ):
            try:
                EncoderClassifier = _imp()
                break
            except Exception:
                continue
        if EncoderClassifier is None:
            raise ImportError("speechbrain EncoderClassifier not found")
        savedir = self._savedir()
        os.makedirs(savedir, exist_ok=True)
        # CPU only: language ID runs during the concurrent perceive stage while
        # the local GPU is busy with face detection — ECAPA on a 20s window is
        # fast on CPU and must never contend for VRAM.
        logger.info("Spoken-language ID: loading VoxLingua107 (%s) on cpu "
                    "(one-time fetch → %s)", self.model_id, savedir)
        model = EncoderClassifier.from_hparams(
            source=self.model_id, savedir=savedir, run_opts={"device": "cpu"})
        SpokenLanguageIdentifier._model = model
        return model

    def _classify_window(self, model, wav_path: str) -> Tuple[Optional[str], float]:
        """Classify one 16 kHz mono WAV → ``(iso, confidence in [0,1])``."""
        import math
        try:
            signal = model.load_audio(wav_path)
        except Exception:
            # Version-robust fallback — the WAV is already 16 kHz mono PCM.
            import torchaudio
            _wav, _sr = torchaudio.load(wav_path)
            signal = _wav.mean(dim=0) if _wav.shape[0] > 1 else _wav.squeeze(0)
        # classify_batch → (out_prob, score, index, text_lab); score is the
        # top class log-likelihood, text_lab like ['ja: Japanese'].
        out = model.classify_batch(signal)
        text_lab = out[3]
        score = out[1]
        label = text_lab[0] if isinstance(text_lab, (list, tuple)) else text_lab
        iso = iso_from_label(label)
        try:
            conf = math.exp(float(score))  # log-likelihood → (0, 1] confidence
        except (TypeError, ValueError, OverflowError):
            conf = 1.0
        return iso, max(0.0, min(1.0, conf))

    def identify(self, video_path: str, duration_ms: int) -> Tuple[Optional[str], str]:
        """Detect the spoken language across several windows of ``video_path``.

        Returns ``(iso, detail)`` or ``(None, reason)``. Best-effort: any
        failure (missing deps, model fetch, ffmpeg, inference) yields
        ``(None, ...)`` so the caller keeps Whisper's auto-detect.
        """
        if not self.is_available():
            return None, "spoken-language ID unavailable (deps/config)"
        try:
            model = self._load_model()
        except Exception as e:  # model fetch / load failure
            return None, f"model load failed: {type(e).__name__}: {str(e)[:120]}"

        dur_s = max(1, int((duration_ms or 0) / 1000))
        observations: List[Tuple[Optional[str], float]] = []
        for start in window_starts(dur_s, self.n_windows):
            wav = tempfile.mktemp(suffix=".wav")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(start), "-t",
                     str(self.WINDOW_SECONDS), "-i", video_path, "-vn",
                     "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav],
                    capture_output=True, timeout=90)
                if not os.path.exists(wav) or os.path.getsize(wav) < 2000:
                    continue
                observations.append(self._classify_window(model, wav))
            except Exception:
                continue
            finally:
                try:
                    os.remove(wav)
                except Exception:
                    pass

        return vote(observations)


def identify_spoken_language(video_path: str, duration_ms: int) -> Tuple[Optional[str], str]:
    """Module-level convenience wrapper. Never raises."""
    try:
        return SpokenLanguageIdentifier().identify(video_path, duration_ms)
    except Exception as e:
        return None, f"spoken-language ID error: {type(e).__name__}: {str(e)[:120]}"
