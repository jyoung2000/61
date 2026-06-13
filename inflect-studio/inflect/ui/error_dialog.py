"""Error dialog with a 'Copy diagnostics' button.

Bundles the short message, the full traceback and the tail of the log file so a
user can paste a complete report when something goes wrong (OOM, missing
ffmpeg, a slow first render that timed out, …).
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..logging_setup import recent_log_text


def show_error(parent: QWidget | None, title: str, message: str, details: str = "") -> None:
    dlg = _ErrorDialog(parent, title, message, details)
    dlg.exec()


class _ErrorDialog(QDialog):
    def __init__(self, parent, title: str, message: str, details: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(560)
        self._diagnostics = self._compose(message, details)

        root = QVBoxLayout(self)
        headline = QLabel(message)
        headline.setWordWrap(True)
        headline.setStyleSheet("font-weight:600;")
        root.addWidget(headline)

        self._detail_view = QPlainTextEdit()
        self._detail_view.setReadOnly(True)
        self._detail_view.setPlainText(self._diagnostics)
        self._detail_view.setMaximumHeight(280)
        root.addWidget(self._detail_view)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_btn = QPushButton("Copy diagnostics")
        copy_btn.clicked.connect(self._copy)
        buttons.addButton(copy_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

    def _compose(self, message: str, details: str) -> str:
        parts = [f"Error: {message}", ""]
        if details:
            parts += ["--- details ---", details, ""]
        parts += ["--- recent log ---", recent_log_text()]
        return "\n".join(parts)

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._diagnostics)
