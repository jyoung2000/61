"""Fish Speech / OpenAudio S2 adapter (Phase 6) — expressive Final engine.

Fish has no numeric emotion vector; it takes free-form inline ``[tag]``
instructions embedded in the text (e.g. ``[whisper in small voice]``,
``[excited and fast]``). The headline of this adapter is therefore
:func:`inflection_to_tags`, which turns our numeric :class:`Inflection` into a
Fish tag string. That mapping is pure and unit-tested.

The actual inference binding is version-specific (the published fish-speech
inference API has changed across releases and is not pinned in the README), so
it is isolated in :meth:`FishEngine._infer`. If the expected entrypoint is not
importable the engine raises :class:`EngineNotAvailable` with install guidance —
the :class:`TTSEngine` contract and everything above it stay unchanged either
way, exactly as the spec requires.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np

from ..config import Config
from ..document.spans import EMOTIONS, Inflection
from ..synth.assemble import time_stretch
from .engine_base import (
    EngineNotAvailable,
    EngineOOM,
    SynthRequest,
    TTSEngine,
    to_mono_float32,
)

log = logging.getLogger("inflect.engine.fish")

FISH_REPO_ID = "fishaudio/s2-pro"

LICENSE_NOTICE = (
    "Fish Speech weights are released for research / personal use. Commercial "
    "use requires a license from Fish Audio (https://fish.audio)."
)

# Per-emotion adjective tiers: (mild < 0.5, moderate < 0.8, intense >= 0.8).
_EMOTION_WORDS: dict[str, tuple[str, str, str]] = {
    "happy": ("a little pleased", "happy", "elated"),
    "angry": ("slightly annoyed", "angry", "furious"),
    "sad": ("a little down", "sad", "heartbroken"),
    "afraid": ("uneasy", "afraid", "terrified"),
    "disgusted": ("displeased", "disgusted", "repulsed"),
    "melancholic": ("pensive", "melancholic", "mournful"),
    "surprised": ("mildly surprised", "surprised", "astonished"),
    "calm": ("calm", "calm and steady", "serene"),
}

# Only dims at or above this strength contribute a tag word.
_EMOTION_TAG_THRESHOLD = 0.35


def _emotion_word(name: str, value: float) -> str:
    mild, base, intense = _EMOTION_WORDS[name]
    if value < 0.5:
        return mild
    if value < 0.8:
        return base
    return intense


def _speed_word(speed: float) -> str | None:
    if speed >= 1.15:
        return "fast"
    if speed >= 1.05:
        return "a little fast"
    if speed <= 0.85:
        return "slow"
    if speed <= 0.95:
        return "a little slow"
    return None


def inflection_to_tags(inflection: Inflection) -> str:
    """Translate an :class:`Inflection` into a Fish ``[tag]`` string ("" if none).

    * ``emo_text`` is preferred verbatim as the tag content.
    * Otherwise the top 1–2 emotion dims ≥ 0.35 become adjective phrases scaled
      by magnitude ("slightly annoyed" → "angry" → "furious").
    * A speed word is appended when speed ≠ 1.0.
    """
    phrases: list[str] = []
    if inflection.emo_text and inflection.emo_text.strip():
        phrases.append(inflection.emo_text.strip())
    else:
        vec = inflection.emotion_vector or []
        ranked = sorted(
            ((EMOTIONS[i], v) for i, v in enumerate(vec) if v >= _EMOTION_TAG_THRESHOLD),
            key=lambda p: p[1],
            reverse=True,
        )[:2]
        phrases.extend(_emotion_word(name, value) for name, value in ranked)

    speed = _speed_word(inflection.speed)
    if speed:
        phrases.append(speed)

    if not phrases:
        return ""
    return "[" + " and ".join(phrases) + "]"


def apply_tags(text: str, inflection: Inflection) -> str:
    """Prepend the inflection's tag to the segment text (Fish reads inline tags)."""
    tag = inflection_to_tags(inflection)
    return f"{tag} {text}".strip() if tag else text


class FishEngine(TTSEngine):
    name = "fish"
    display_name = "Fish S2 (Final)"
    supports_cpu = False  # 4B model — CPU is impractically slow
    sample_rate = 44_100  # provisional; refined from the first render
    license_notice = LICENSE_NOTICE

    def __init__(self, config: Config, device: str = "cuda") -> None:
        super().__init__(device=device)
        self.config = config
        self._engine = None

    @property
    def model_dir(self) -> Path:
        return self.config.paths.fish_dir

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        self._engine = self._load_inference()
        self._loaded = True

    def unload(self) -> None:
        self._engine = None
        self._loaded = False
        gc.collect()
        _empty_cache()

    # -- synthesis ---------------------------------------------------------
    def synthesize(self, request: SynthRequest) -> np.ndarray:
        if not self._loaded:
            self.load()
        job = request.job
        text = apply_tags(job.text, job.inflection)
        # An empty speaker_wav means "Fish default voice" (used by hybrid stage 1,
        # where we only want the performance, not the cloned timbre).
        speaker = request.speaker_wav or None
        try:
            audio, sr = self._infer(text, speaker)
            self.sample_rate = int(sr)
        except Exception as exc:
            if _is_oom(exc):
                raise EngineOOM(
                    "GPU ran out of memory in Fish. Fish S2 is a 4B model — close "
                    "other GPU apps or shorten the segment."
                ) from exc
            raise
        audio = to_mono_float32(audio)
        # Tags steer tempo loosely; keep the precise time-stretch fallback.
        if abs(job.inflection.speed - 1.0) > 1e-3:
            audio = time_stretch(audio, job.inflection.speed)
        _empty_cache()
        return audio

    # -- version-specific binding -----------------------------------------
    def _load_inference(self):
        """Construct Fish's inference engine. Isolated; confirm vs installed version.

        The published fish-speech inference API differs by release. This method
        is the single integration point: wire it to whatever the installed
        version exposes (an inference-engine class or the tools.* entrypoints),
        downloading ``fishaudio/s2-pro`` into :pyattr:`model_dir` on first run.
        """
        try:
            import torch  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise EngineNotAvailable("PyTorch is required for Fish.") from exc

        self._ensure_weights()
        try:
            # Preferred: the high-level inference engine shipped with recent
            # fish-speech / openaudio releases.
            from fish_speech.inference_engine import TTSInferenceEngine  # type: ignore

            return TTSInferenceEngine.from_pretrained(  # type: ignore[attr-defined]
                str(self.model_dir),
                device=self.device,
                half=bool(self.config.settings.use_fp16 and self.device == "cuda"),
            )
        except Exception as exc:
            raise EngineNotAvailable(
                "Fish Speech is not installed, or its inference API differs from "
                "this build. Install it from source "
                "(pip install git+https://github.com/fishaudio/fish-speech.git) "
                "and wire FishEngine._load_inference/_infer to the installed "
                "inference entrypoint. " + LICENSE_NOTICE
            ) from exc

    def _infer(self, text: str, speaker_wav: str | None) -> tuple[np.ndarray, int]:
        """Run one Fish synthesis. Returns (mono audio, sample_rate).

        Kept tiny and version-specific. Adapt the call below to the inference
        object returned by :meth:`_load_inference` for your installed version.
        """
        engine = self._engine
        if engine is None:  # pragma: no cover
            raise EngineNotAvailable("Fish inference engine is not loaded.")
        # Most fish-speech inference engines expose a generate-style call taking
        # the text and an optional reference wav for zero-shot cloning.
        result = engine.synthesize(text=text, reference_audio=speaker_wav)  # type: ignore[attr-defined]
        audio, sr = _coerce_result(result, getattr(engine, "sample_rate", self.sample_rate))
        return audio, sr

    def _ensure_weights(self) -> None:
        if any(self.model_dir.glob("**/*")):
            return
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover
            raise EngineNotAvailable("huggingface_hub is required to fetch Fish weights.") from exc
        self.model_dir.mkdir(parents=True, exist_ok=True)
        log.info("Downloading Fish weights %s …", FISH_REPO_ID)
        snapshot_download(
            repo_id=FISH_REPO_ID, local_dir=str(self.model_dir), local_dir_use_symlinks=False
        )


def _coerce_result(result, default_sr: int) -> tuple[np.ndarray, int]:
    """Accept the variety of shapes fish-speech versions return."""
    if isinstance(result, tuple) and len(result) == 2:
        audio, sr = result
        return np.asarray(audio, dtype=np.float32), int(sr)
    return np.asarray(result, dtype=np.float32), int(default_sr)


def _is_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or exc.__class__.__name__ == "OutOfMemoryError"


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
