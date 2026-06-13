"""Central configuration: data paths, model directories and user settings.

This is the *only* place global, process-wide state is allowed to live (per the
spec's "no global state outside config.py" rule). Everything here is plain
stdlib + JSON so it imports with zero heavy dependencies and is test-friendly.

The data root can be relocated with the ``INFLECT_HOME`` environment variable,
which also lets tests point at a throwaway tmp dir.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

APP_NAME = "Inflect Studio"
APP_ORG = "InflectStudio"
APP_ID = "inflect-studio"

# Canonical audio rate used for the extracted voice reference and as a fallback
# assembly rate. Engines may output at their own native rate; assembly resamples
# everything to the chosen canonical rate at mix time.
REFERENCE_SR = 24_000

# Underline colors for inflection spans (RGB hex). Length must match
# spans.NUM_SPAN_COLORS. Chosen to be distinguishable on a dark background.
SPAN_COLORS: list[str] = [
    "#ff6b6b",  # red
    "#feca57",  # amber
    "#1dd1a1",  # green
    "#54a0ff",  # blue
    "#5f27cd",  # purple
    "#ff9ff3",  # pink
    "#00d2d3",  # cyan
    "#ee5253",  # crimson
]


def _default_home() -> Path:
    env = os.environ.get("INFLECT_HOME")
    if env:
        return Path(env).expanduser()
    # Windows-first per the spec, but degrade gracefully on other platforms.
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / APP_ORG / APP_ID
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / APP_ID
    return Path.home() / ".inflect-studio"


@dataclass
class Paths:
    """All filesystem locations the app uses, derived from a single root."""

    home: Path

    @property
    def voices(self) -> Path:
        return self.home / "voices"

    @property
    def voices_index(self) -> Path:
        return self.voices / "index.json"

    @property
    def models(self) -> Path:
        return self.home / "models"

    @property
    def indextts2_dir(self) -> Path:
        return self.models / "indextts2"

    @property
    def chatterbox_dir(self) -> Path:
        return self.models / "chatterbox"

    @property
    def fish_dir(self) -> Path:
        return self.models / "fish"

    @property
    def demucs_dir(self) -> Path:
        return self.models / "demucs"

    @property
    def cache(self) -> Path:
        return self.home / "project_cache"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def tmp(self) -> Path:
        return self.home / "tmp"

    @property
    def settings_file(self) -> Path:
        return self.home / "settings.json"

    def ensure(self) -> "Paths":
        for p in (
            self.home,
            self.voices,
            self.models,
            self.indextts2_dir,
            self.chatterbox_dir,
            self.fish_dir,
            self.demucs_dir,
            self.cache,
            self.logs,
            self.tmp,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass
class Settings:
    """User-tweakable settings, persisted as JSON alongside the data root."""

    ffmpeg_path: str = "ffmpeg"  # resolved from PATH unless overridden
    ffprobe_path: str = "ffprobe"
    output_device: int | None = None  # sounddevice device index, None = default
    default_engine: str = "indextts2"  # "chatterbox" (Draft) | "indextts2" (Final)
    theme: str = "dark"
    use_cuda: bool = True
    use_fp16: bool = True
    # IndexTTS-2's custom BigVGAN CUDA kernel is faster but requires a separate
    # compile step; off by default so a fresh install never crashes on load.
    use_cuda_kernel: bool = False
    crossfade_ms: int = 15
    target_lufs: float = -16.0
    true_peak_dbtp: float = -1.0
    # Per-engine advanced params surfaced in the settings dialog.
    chatterbox_cfg_weight: float = 0.5
    auto_isolate_threshold: float = 0.6  # VAD speech-ratio below which we suggest demucs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Config:
    """Bundle of :class:`Paths` + :class:`Settings` passed around the app."""

    paths: Paths
    settings: Settings = field(default_factory=Settings)

    def save_settings(self) -> None:
        self.paths.home.mkdir(parents=True, exist_ok=True)
        self.paths.settings_file.write_text(
            json.dumps(self.settings.to_dict(), indent=2), "utf-8"
        )

    @classmethod
    def load(cls, home: str | Path | None = None) -> "Config":
        root = Path(home).expanduser() if home else _default_home()
        paths = Paths(root)
        settings = Settings()
        sf = paths.settings_file
        if sf.exists():
            try:
                settings = Settings.from_dict(json.loads(sf.read_text("utf-8")))
            except (json.JSONDecodeError, OSError, TypeError):
                # Corrupt settings should never block startup; fall back to defaults.
                settings = Settings()
        return cls(paths=paths, settings=settings)


# Process-wide singleton, created lazily so importing this module is cheap and
# does not touch the filesystem until something actually needs a path.
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.load()
        _config.paths.ensure()
    return _config


def set_config(config: Config) -> None:
    """Override the singleton (used by tests and by an explicit home switch)."""
    global _config
    _config = config
