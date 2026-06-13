"""Rotating file + console logging, plus a helper to grab recent log text.

Per-segment synth timings and VRAM snapshots are logged at INFO by the pipeline,
so ``logs/inflect.log`` doubles as the diagnostics the error dialog can copy.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import Paths

_CONFIGURED = False
_LOG_PATH: Path | None = None


def setup_logging(paths: Paths, level: int = logging.INFO) -> Path:
    """Configure the root 'inflect' logger once. Returns the log file path."""
    global _CONFIGURED, _LOG_PATH
    paths.logs.mkdir(parents=True, exist_ok=True)
    log_path = paths.logs / "inflect.log"
    _LOG_PATH = log_path
    if _CONFIGURED:
        return log_path

    logger = logging.getLogger("inflect")
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"
    )
    file_handler = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    _CONFIGURED = True
    logger.info("Logging started → %s", log_path)
    return log_path


def recent_log_text(max_lines: int = 200) -> str:
    """Return the tail of the log file for the 'copy diagnostics' button."""
    if _LOG_PATH is None or not _LOG_PATH.exists():
        return "No log file available."
    try:
        lines = _LOG_PATH.read_text("utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Could not read log: {exc}"
    return "\n".join(lines[-max_lines:])
