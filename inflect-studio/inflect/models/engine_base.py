"""Abstract base class every TTS engine adapter implements.

The contract is deliberately tiny so the rest of the app (worker, pipeline,
assembly, UI) never has to know which engine produced a segment:

* :meth:`load` / :meth:`unload` manage the (large) model in VRAM. Only one
  engine is ever loaded at a time -- see :mod:`inflect.models.model_manager`.
* :meth:`synthesize` renders a single :class:`SegmentJob` to a mono ``float32``
  numpy array at :attr:`sample_rate`. Engines resample/normalize is left to the
  assembler so each engine can emit at its own native rate.

Nothing in this module imports torch or any engine SDK, so it stays importable
(and the ABC stays unit-testable with fakes) on machines without a GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..document.segmenter import SegmentJob


class EngineError(RuntimeError):
    """Base class for recoverable engine failures surfaced to the UI."""


class EngineNotAvailable(EngineError):
    """The engine's package/weights are not installed or importable."""


class EngineOOM(EngineError):
    """CUDA ran out of memory while rendering a segment."""


@dataclass
class SynthRequest:
    """Everything an engine needs to render one segment.

    Resolved by the pipeline from a :class:`SegmentJob` + the voice library, so
    engines never touch the profile store directly.
    """

    job: SegmentJob
    speaker_wav: str  # path to the voice profile's reference wav (timbre)
    device: str = "cuda"  # "cuda" | "cpu"


class TTSEngine(ABC):
    """Common interface for Chatterbox / IndexTTS-2 / Fish adapters."""

    #: Stable short name, e.g. "chatterbox" / "indextts2" / "fish".
    name: str = "base"
    #: Native output sample rate in Hz. Set after :meth:`load` if it varies.
    sample_rate: int = 24_000
    #: Human-readable label for the UI.
    display_name: str = "Base Engine"
    #: Whether this engine can run (slowly) on CPU.
    supports_cpu: bool = False

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights into memory/VRAM. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Free weights + VRAM. Idempotent. Must be safe to call when unloaded."""

    def is_loaded(self) -> bool:
        return self._loaded

    # -- synthesis ---------------------------------------------------------
    @abstractmethod
    def synthesize(self, request: SynthRequest) -> np.ndarray:
        """Render one segment to mono float32 at :attr:`sample_rate`.

        Implementations must:
          * call ``torch.cuda.empty_cache()`` after the forward pass,
          * translate :class:`EngineOOM` from CUDA OOM errors,
          * never block on the GUI thread (callers run this on a worker).
        """

    # -- context manager sugar --------------------------------------------
    def __enter__(self) -> "TTSEngine":
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Coerce arbitrary engine output to a contiguous mono float32 vector."""
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        # (channels, n) or (n, channels) -> average to mono.
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    arr = np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)
    # Guard against NaN/Inf leaking from a half-precision forward pass.
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
