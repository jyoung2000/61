"""Chatterbox adapter -- the fast "Draft" preview engine.

Chatterbox has no numeric emotion vector; it exposes ``exaggeration`` and
``cfg_weight``. We derive ``exaggeration`` from the magnitude of the inflection's
emotion vector (energetic emotions push it higher) so Draft mode still tracks the
user's intent, and surface ``cfg_weight`` as an advanced/engine param.

The vector→exaggeration mapping is a pure function so it is unit-tested without
loading the model.
"""

from __future__ import annotations

import gc

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

# Emotions that read as high-energy / expressive get more weight when deriving
# Chatterbox's single "exaggeration" knob. Index order matches EMOTIONS.
_ENERGY_WEIGHTS = {
    "happy": 1.0,
    "angry": 1.0,
    "sad": 0.5,
    "afraid": 0.9,
    "disgusted": 0.8,
    "melancholic": 0.4,
    "surprised": 1.0,
    "calm": 0.1,
}
_ENERGY = np.array([_ENERGY_WEIGHTS[e] for e in EMOTIONS], dtype=np.float32)

NEUTRAL_EXAGGERATION = 0.5
MIN_EXAGGERATION = 0.25
MAX_EXAGGERATION = 1.5


def emotion_to_exaggeration(inflection: Inflection) -> float:
    """Map an inflection's emotion vector onto Chatterbox's exaggeration.

    Neutral (no vector) → 0.5. Otherwise a weighted blend of the active emotion
    dims, scaled so a fully-saturated energetic emotion approaches the max.
    """
    vec = inflection.emotion_vector
    if not vec or not any(v > 1e-6 for v in vec):
        return NEUTRAL_EXAGGERATION
    v = np.asarray(vec, dtype=np.float32)
    # Weighted average of active dims, biased by how energetic they are.
    weighted = float(np.sum(v * _ENERGY) / (np.sum(v) + 1e-6))
    intensity = float(np.max(v))  # how strongly any emotion is felt
    exaggeration = NEUTRAL_EXAGGERATION + weighted * intensity
    return float(min(MAX_EXAGGERATION, max(MIN_EXAGGERATION, exaggeration)))


class ChatterboxEngine(TTSEngine):
    name = "chatterbox"
    display_name = "Chatterbox (Draft)"
    supports_cpu = True
    sample_rate = 24_000

    def __init__(self, config: Config, device: str = "cuda") -> None:
        super().__init__(device=device)
        self.config = config
        self.model = None

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from chatterbox.tts import ChatterboxTTS
        except Exception as exc:
            raise EngineNotAvailable(
                "Chatterbox is not installed. Run `pip install chatterbox-tts`."
            ) from exc
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        # Chatterbox exposes its native rate as `.sr`.
        self.sample_rate = int(getattr(self.model, "sr", 24_000))
        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self._loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def synthesize(self, request: SynthRequest) -> np.ndarray:
        if not self._loaded:
            self.load()
        job = request.job
        inflection = job.inflection
        exaggeration = emotion_to_exaggeration(inflection)
        cfg_weight = float(
            job.engine_params.get("cfg_weight", self.config.settings.chatterbox_cfg_weight)
        )
        try:
            wav = self.model.generate(
                job.text,
                audio_prompt_path=request.speaker_wav,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        except Exception as exc:  # translate CUDA OOM into a friendly error
            if _is_oom(exc):
                raise EngineOOM(
                    "GPU ran out of memory while drafting. Close other GPU apps "
                    "or try a shorter segment."
                ) from exc
            raise
        audio = to_mono_float32(wav)
        if abs(inflection.speed - 1.0) > 1e-3:
            audio = time_stretch(audio, inflection.speed)
        _empty_cache()
        return audio


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
