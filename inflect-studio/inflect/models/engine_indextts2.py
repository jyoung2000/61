"""IndexTTS-2 adapter -- the high-quality "Final" engine.

Wraps ``indextts.infer_v2.IndexTTS2`` whose real inference signature is::

    infer(spk_audio_prompt, text, output_path,
          emo_audio_prompt=None, emo_alpha=1.0, emo_vector=None,
          use_emo_text=False, emo_text=None, use_random=False, verbose=True)

Notes that shaped this adapter:
* ``emo_vector`` order is exactly our :data:`EMOTIONS` order, so it maps 1:1.
* When both a vector and emo_text are present we prefer the vector (deterministic)
  per the spec, and log a notice.
* ``infer`` writes a wav to ``output_path``; we read it back to learn the native
  sample rate rather than hard-coding it.
* There is no documented speed/duration argument, so the ``speed`` knob is applied
  as a librosa phase-vocoder time-stretch after synthesis.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..document.spans import Inflection
from ..synth.assemble import time_stretch
from .engine_base import (
    EngineNotAvailable,
    EngineOOM,
    SynthRequest,
    TTSEngine,
    to_mono_float32,
)
from .model_manager import download_indextts2, indextts2_weights_present

log = logging.getLogger("inflect.engine.indextts2")


@dataclass
class EmotionArgs:
    """The subset of ``infer`` kwargs that express an inflection's emotion."""

    emo_vector: list[float] | None = None
    emo_audio_prompt: str | None = None
    use_emo_text: bool = False
    emo_text: str | None = None
    emo_alpha: float = 1.0


def resolve_emotion(inflection: Inflection) -> EmotionArgs:
    """Translate an :class:`Inflection` into IndexTTS-2 emotion kwargs.

    Precedence (deterministic first): explicit emotion vector → emotion reference
    audio → emotion text. Only one channel is used at a time to avoid undefined
    combinations in the underlying model.
    """
    has_vector = bool(inflection.emotion_vector) and any(
        v > 1e-6 for v in (inflection.emotion_vector or [])
    )
    args = EmotionArgs(emo_alpha=float(inflection.emo_alpha))
    if has_vector:
        if inflection.emo_text:
            log.info("Both emotion vector and emo_text set; using the vector.")
        args.emo_vector = list(inflection.emotion_vector or [])
    elif inflection.emo_audio:
        args.emo_audio_prompt = inflection.emo_audio
    elif inflection.emo_text:
        args.use_emo_text = True
        args.emo_text = inflection.emo_text
    return args


class IndexTTS2Engine(TTSEngine):
    name = "indextts2"
    display_name = "IndexTTS-2 (Final)"
    supports_cpu = True  # technically yes, but extremely slow -- UI warns
    sample_rate = 22_050  # provisional; overwritten from the first render

    def __init__(self, config: Config, device: str = "cuda") -> None:
        super().__init__(device=device)
        self.config = config
        self.model = None
        self._tmp_dir = config.paths.tmp

    @property
    def model_dir(self) -> Path:
        return self.config.paths.indextts2_dir

    def ensure_weights(self, progress=None) -> None:
        if not indextts2_weights_present(self.model_dir):
            download_indextts2(self.model_dir, progress)

    def load(self) -> None:
        if self._loaded:
            return
        try:
            from indextts.infer_v2 import IndexTTS2
        except Exception as exc:
            raise EngineNotAvailable(
                "IndexTTS-2 is not installed. Install the `indextts` package or "
                "`pip install git+https://github.com/index-tts/index-tts.git`."
            ) from exc
        self.ensure_weights()
        cfg_path = self.model_dir / "config.yaml"
        use_cuda = self.device == "cuda"
        self.model = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(self.model_dir),
            use_fp16=bool(self.config.settings.use_fp16 and use_cuda),
            use_cuda_kernel=bool(self.config.settings.use_cuda_kernel and use_cuda),
        )
        self._loaded = True

    def unload(self) -> None:
        self.model = None
        self._loaded = False
        gc.collect()
        _empty_cache()

    def synthesize(self, request: SynthRequest) -> np.ndarray:
        if not self._loaded:
            self.load()
        job = request.job
        emo = resolve_emotion(job.inflection)

        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=str(self._tmp_dir), delete=False
        ) as tmp:
            out_path = Path(tmp.name)
        try:
            self.model.infer(
                spk_audio_prompt=request.speaker_wav,
                text=job.text,
                output_path=str(out_path),
                emo_audio_prompt=emo.emo_audio_prompt,
                emo_alpha=emo.emo_alpha,
                emo_vector=emo.emo_vector,
                use_emo_text=emo.use_emo_text,
                emo_text=emo.emo_text,
                verbose=False,
            )
            audio, sr = _read_wav(out_path)
            self.sample_rate = int(sr)
        except Exception as exc:
            if _is_oom(exc):
                raise EngineOOM(
                    "GPU ran out of memory during Final render. Close other GPU "
                    "apps, shorten the segment, or switch to Draft mode."
                ) from exc
            raise
        finally:
            out_path.unlink(missing_ok=True)
            _empty_cache()

        audio = to_mono_float32(audio)
        if abs(job.inflection.speed - 1.0) > 1e-3:
            audio = time_stretch(audio, job.inflection.speed)
        return audio


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32), int(sr)


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
