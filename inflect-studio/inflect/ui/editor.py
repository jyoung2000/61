"""InflectionTextEdit — a QTextEdit that renders inflection spans as underlines.

Crucially, span styling is *derived* from the :class:`Document` model on every
change and drawn with ``setExtraSelections`` — it is never baked into the text
document's own character formatting. That keeps the text and the styling model
cleanly separated (copy/paste stays plain text, save/load is just the model).

The editor keeps the model's span offsets in sync with edits by listening to the
document's ``contentsChange(position, removed, added)`` signal and calling
:meth:`Document.remap_for_edit`.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QMenu, QTextEdit

from ..document.spans import Document, Inflection, InflectionSpan
from .format import color_for_inflection, color_hex, summarize_inflection


class InflectionTextEdit(QTextEdit):
    #: Emitted when the caret/selection moves (inspector should refresh).
    selectionContextChanged = Signal()
    #: Emitted when the span model changes (re-render may be needed).
    spansChanged = Signal()
    #: Right-click actions routed to the main window.
    applyInflectionRequested = Signal()
    clearInflectionRequested = Signal()
    insertPauseRequested = Signal()
    previewSelectionRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc = Document()
        self._syncing = False
        self.setAcceptRichText(False)
        self.setMouseTracking(True)
        self.setPlaceholderText("Type the script here. Highlight any part to change how it's spoken.")
        self.document().contentsChange.connect(self._on_contents_change)
        self.cursorPositionChanged.connect(self._emit_context)
        self.selectionChanged.connect(self._emit_context)

    # -- model wiring ------------------------------------------------------
    @property
    def model(self) -> Document:
        return self._doc

    def set_document_model(self, document: Document) -> None:
        """Replace the backing model and repaint (used on project load/new)."""
        self._doc = document
        self._syncing = True
        try:
            self.setPlainText(document.text)
        finally:
            self._syncing = False
        self.refresh_highlights()
        self._emit_context()

    def _on_contents_change(self, position: int, removed: int, added: int) -> None:
        if self._syncing:
            return
        self._doc.text = self.toPlainText()
        self._doc.remap_for_edit(position, removed, added)
        self.refresh_highlights()
        self.spansChanged.emit()

    # -- selection / context ----------------------------------------------
    def selection_range(self) -> tuple[int, int]:
        cur = self.textCursor()
        return cur.selectionStart(), cur.selectionEnd()

    def has_selection(self) -> bool:
        start, end = self.selection_range()
        return end > start

    def effective_inflection(self) -> Inflection:
        """The inflection in force at the current caret/selection start."""
        start, end = self.selection_range()
        span = self._doc.span_at(start)
        return span.inflection if span else self._doc.default_inflection

    def current_span(self) -> InflectionSpan | None:
        start, _ = self.selection_range()
        return self._doc.span_at(start)

    def _emit_context(self) -> None:
        self.selectionContextChanged.emit()

    # -- mutations (called by inspector / menu) ----------------------------
    def apply_inflection_to_selection(self, inflection: Inflection) -> None:
        start, end = self.selection_range()
        if end <= start:
            return
        self._doc.apply_inflection(start, end, inflection, color_for_inflection(inflection))
        self.refresh_highlights()
        self.spansChanged.emit()
        self._emit_context()

    def clear_selection_inflection(self) -> None:
        start, end = self.selection_range()
        if end <= start:
            span = self.current_span()
            if span is None:
                return
            start, end = span.start, span.end
        self._doc.clear_inflection(start, end)
        self.refresh_highlights()
        self.spansChanged.emit()
        self._emit_context()

    def insert_pause_at_caret(self, pause_ms: int) -> None:
        start, _ = self.selection_range()
        self._doc.set_pause_after(start, pause_ms)
        self.refresh_highlights()
        self.spansChanged.emit()
        self._emit_context()

    def set_default_inflection(self, inflection: Inflection) -> None:
        self._doc.default_inflection = inflection
        self.spansChanged.emit()

    # -- rendering ---------------------------------------------------------
    def refresh_highlights(self) -> None:
        """Rebuild the colored underlines from the span model."""
        selections: list[QTextEdit.ExtraSelection] = []
        text_len = len(self._doc.text)
        for span in self._doc.sorted_spans():
            if span.start >= span.end or span.start >= text_len:
                continue
            sel = QTextEdit.ExtraSelection()
            cursor = QTextCursor(self.document())
            cursor.setPosition(min(span.start, text_len))
            cursor.setPosition(min(span.end, text_len), QTextCursor.MoveMode.KeepAnchor)
            fmt = QTextCharFormat()
            color = QColor(color_hex(span.color_idx))
            fmt.setUnderlineColor(color)
            # Wavy underline for emotional spans, solid for plain delivery tweaks.
            style = (
                QTextCharFormat.UnderlineStyle.WaveUnderline
                if span.inflection.emotion_vector
                else QTextCharFormat.UnderlineStyle.SingleUnderline
            )
            fmt.setUnderlineStyle(style)
            fmt.setFontUnderline(True)
            sel.cursor = cursor
            sel.format = fmt
            selections.append(sel)
        self.setExtraSelections(selections)

    # -- hover tooltip -----------------------------------------------------
    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ToolTip:
            cursor = self.cursorForPosition(event.pos())
            pos = cursor.position()
            span = self._doc.span_at(pos)
            if span is not None:
                self.setToolTip(summarize_inflection(span.inflection))
            else:
                self.setToolTip("")
        return super().event(event)

    # -- context menu ------------------------------------------------------
    def contextMenuEvent(self, event) -> None:  # noqa: ANN001
        menu: QMenu = self.createStandardContextMenu()
        menu.addSeparator()
        has_sel = self.has_selection()

        apply_act = menu.addAction("Apply inflection to selection")
        apply_act.setEnabled(has_sel)
        apply_act.triggered.connect(lambda: self.applyInflectionRequested.emit())

        clear_act = menu.addAction("Clear inflection")
        clear_act.setEnabled(has_sel or self.current_span() is not None)
        clear_act.triggered.connect(lambda: self.clearInflectionRequested.emit())

        pause_act = menu.addAction("Insert pause…")
        pause_act.triggered.connect(lambda: self.insertPauseRequested.emit())

        preview_act = menu.addAction("Preview this segment")
        preview_act.setEnabled(has_sel or self.current_span() is not None)
        preview_act.triggered.connect(lambda: self.previewSelectionRequested.emit())

        menu.exec(event.globalPos())
