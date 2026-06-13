"""QApplication bootstrap: dark Fusion theme, logging, startup checks."""

from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QMessageBox

from .config import get_config
from .ingest.extract import ffmpeg_available
from .logging_setup import setup_logging

log = logging.getLogger("inflect.app")


def _apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    base = QColor("#1e1e22")
    alt = QColor("#26262b")
    text = QColor("#e6e6e6")
    highlight = QColor("#3b82f6")
    palette.setColor(QPalette.ColorRole.Window, QColor("#1b1b1f"))
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, alt)
    palette.setColor(QPalette.ColorRole.ToolTipBase, alt)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, alt)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#7a7a82"))
    disabled = QColor("#6a6a72")
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled)
    app.setPalette(palette)


def build_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Inflect Studio")
    app.setOrganizationName("InflectStudio")
    _apply_dark_theme(app)
    return app


def run() -> int:
    config = get_config()
    setup_logging(config.paths)
    app = build_app()

    # Import here so a missing optional dep surfaces as a dialog, not a crash.
    from .ui.main_window import MainWindow

    window = MainWindow(config)

    if not ffmpeg_available(config.settings.ffmpeg_path):
        QMessageBox.warning(
            window,
            "ffmpeg not found",
            "ffmpeg was not found on your PATH. Importing voices and exporting "
            "MP3 will not work until you install ffmpeg "
            "(https://ffmpeg.org/download.html) or set its path in Settings.",
        )

    window.show()
    return app.exec()
