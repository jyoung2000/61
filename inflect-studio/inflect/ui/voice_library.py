"""Left-dock Voice Library: profile cards, import button, preview play.

Cards are plain widgets hosted in a QListWidget so selection/keyboard work for
free. The dock owns no audio playback itself -- it emits signals the main window
wires to the shared player and the ingest wizard.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..ingest.profile import VoiceLibrary, VoiceProfile


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class _ProfileCard(QWidget):
    previewClicked = Signal(str)  # profile id

    def __init__(self, profile: VoiceProfile, parent=None) -> None:
        super().__init__(parent)
        self.profile = profile
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)

        text = QVBoxLayout()
        name = QLabel(profile.name)
        name.setStyleSheet("font-weight:600;")
        meta = QLabel(f"{_fmt_duration(profile.duration_s)} · {profile.source_filename or 'imported'}")
        meta.setStyleSheet("color:#9aa; font-size:11px;")
        text.addWidget(name)
        text.addWidget(meta)
        row.addLayout(text, 1)

        play = QPushButton("▶")
        play.setFixedWidth(34)
        play.setToolTip("Preview reference audio")
        play.clicked.connect(lambda: self.previewClicked.emit(self.profile.id))
        row.addWidget(play)


class VoiceLibraryDock(QWidget):
    importRequested = Signal()
    profileSelected = Signal(str)  # profile id (or "" when cleared)
    previewRequested = Signal(str)  # profile id

    def __init__(self, library: VoiceLibrary, parent=None) -> None:
        super().__init__(parent)
        self._library = library
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        self._import_btn = QPushButton("＋  Import from video / audio…")
        self._import_btn.clicked.connect(lambda: self.importRequested.emit())
        root.addWidget(self._import_btn)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.currentItemChanged.connect(self._on_selection)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        root.addWidget(self._list, 1)

    # -- data --------------------------------------------------------------
    def refresh(self, select_id: str | None = None) -> None:
        self._list.clear()
        for profile in self._library.list_profiles():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, profile.id)
            card = _ProfileCard(profile)
            card.previewClicked.connect(lambda pid: self.previewRequested.emit(pid))
            item.setSizeHint(card.sizeHint())
            self._list.addItem(item)
            self._list.setItemWidget(item, card)
            if select_id and profile.id == select_id:
                self._list.setCurrentItem(item)

    def current_profile_id(self) -> str | None:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def select_profile(self, profile_id: str) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == profile_id:
                self._list.setCurrentItem(item)
                return

    # -- events ------------------------------------------------------------
    def _on_selection(self, current, _previous) -> None:  # noqa: ANN001
        self.profileSelected.emit(current.data(Qt.ItemDataRole.UserRole) if current else "")

    def _on_context_menu(self, pos) -> None:  # noqa: ANN001
        item = self._list.itemAt(pos)
        if item is None:
            return
        profile_id = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        rename = menu.addAction("Rename…")
        delete = menu.addAction("Delete")
        chosen = menu.exec(self._list.mapToGlobal(pos))
        if chosen == rename:
            self._rename(profile_id)
        elif chosen == delete:
            self._delete(profile_id)

    def _rename(self, profile_id: str) -> None:
        try:
            current = self._library.get(profile_id)
        except Exception:
            return
        new_name, ok = QInputDialog.getText(self, "Rename voice", "Name:", text=current.name)
        if ok and new_name.strip():
            self._library.rename(profile_id, new_name.strip())
            self.refresh(select_id=profile_id)

    def _delete(self, profile_id: str) -> None:
        confirm = QMessageBox.question(
            self,
            "Delete voice",
            "Delete this voice profile and its reference audio? This cannot be undone.",
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._library.delete(profile_id)
            self.refresh()
            self.profileSelected.emit("")
