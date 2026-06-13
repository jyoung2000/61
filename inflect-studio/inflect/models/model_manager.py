"""Single owner of the currently-loaded TTS engine + weight downloads + VRAM.

The hard rule on a 12 GB card: exactly one heavy model resident at a time. Every
engine swap goes through :meth:`ModelManager.get_engine`, which unloads the
previous engine and reclaims VRAM (``del`` + ``empty_cache`` + ``gc``) before
loading the next. Demucs isolation also asks the manager to free the current
engine first via :meth:`unload_current`.

Torch is imported lazily inside functions so this module imports on a machine
with no CUDA / no torch (the UI can still start and show a friendly message).
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from .engine_base import EngineNotAvailable, TTSEngine

ProgressCb = Callable[[str, float], None]  # (message, fraction 0..1 or -1 indeterminate)


# --------------------------------------------------------------------------- #
# VRAM / device helpers (all torch access guarded)
# --------------------------------------------------------------------------- #
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(prefer_cuda: bool = True) -> str:
    return "cuda" if (prefer_cuda and cuda_available()) else "cpu"


@dataclass
class VramSnapshot:
    allocated_mb: float
    reserved_mb: float
    total_mb: float

    @property
    def available(self) -> bool:
        return self.total_mb > 0

    def __str__(self) -> str:
        if not self.available:
            return "VRAM: n/a"
        return f"VRAM: {self.allocated_mb:.0f} / {self.total_mb:.0f} MB"


def vram_snapshot() -> VramSnapshot:
    try:
        import torch

        if not torch.cuda.is_available():
            return VramSnapshot(0, 0, 0)
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        return VramSnapshot(
            allocated_mb=torch.cuda.memory_allocated(dev) / 1024**2,
            reserved_mb=torch.cuda.memory_reserved(dev) / 1024**2,
            total_mb=props.total_memory / 1024**2,
        )
    except Exception:
        return VramSnapshot(0, 0, 0)


def empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Weight presence / download
# --------------------------------------------------------------------------- #
def indextts2_weights_present(model_dir: Path) -> bool:
    return (Path(model_dir) / "config.yaml").exists()


def fish_weights_present(model_dir: Path) -> bool:
    p = Path(model_dir)
    return p.exists() and any(p.glob("**/*.pth"))


def download_indextts2(model_dir: Path, progress: ProgressCb | None = None) -> Path:
    """Download IndexTeam/IndexTTS-2 weights into ``model_dir`` (first run)."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if indextts2_weights_present(model_dir):
        return model_dir
    if progress:
        progress("Downloading IndexTTS-2 weights (first run, several GB)…", -1)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - hub missing
        raise EngineNotAvailable(
            "huggingface_hub is required to download model weights."
        ) from exc
    snapshot_download(
        repo_id="IndexTeam/IndexTTS-2",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    if progress:
        progress("IndexTTS-2 weights ready.", 1.0)
    return model_dir


# --------------------------------------------------------------------------- #
# Engine registry + swap
# --------------------------------------------------------------------------- #
def _build_engine(name: str, config: Config, device: str) -> TTSEngine:
    # Imported lazily so an engine's SDK is only needed when actually selected.
    if name == "chatterbox":
        from .engine_chatterbox import ChatterboxEngine

        return ChatterboxEngine(config=config, device=device)
    if name == "indextts2":
        from .engine_indextts2 import IndexTTS2Engine

        return IndexTTS2Engine(config=config, device=device)
    if name == "fish":
        from .engine_fish import FishEngine

        return FishEngine(config=config, device=device)
    raise EngineNotAvailable(f"Unknown engine: {name!r}")


class ModelManager:
    """Owns the one resident engine and serializes swaps."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine: TTSEngine | None = None

    @property
    def current_engine_name(self) -> str | None:
        return self._engine.name if self._engine else None

    def get_engine(self, name: str, device: str | None = None) -> TTSEngine:
        """Return a loaded engine ``name``, swapping out any other engine first."""
        device = device or resolve_device(self.config.settings.use_cuda)
        if self._engine is not None and self._engine.name == name:
            if not self._engine.is_loaded():
                self._engine.load()
            return self._engine
        self.unload_current()
        engine = _build_engine(name, self.config, device)
        engine.load()
        self._engine = engine
        return engine

    def unload_current(self) -> None:
        """Unload + free the resident engine (also called before Demucs runs)."""
        if self._engine is not None:
            try:
                self._engine.unload()
            finally:
                self._engine = None
        gc.collect()
        empty_cuda_cache()

    def vram(self) -> VramSnapshot:
        return vram_snapshot()
