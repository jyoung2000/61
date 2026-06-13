"""Right-dock Inspector: edit the inflection for the current selection.

Reads the editor's effective inflection into its widgets, and builds a fresh
:class:`Inflection` from the widgets when the user applies it. Includes the
per-span engine selector and (Phase 6) the hybrid "audition performance" button;
both degrade harmlessly when those engines are not installed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..document.spans import EMOTIONS, Inflection
from .format import color_hex

# engine combo label <-> model value
_ENGINE_ITEMS = [
    ("Default (toolbar)", None),
    ("Chatterbox (Draft)", "chatterbox"),
    ("IndexTTS-2 (Final)", "indextts2"),
    ("Fish (Final)", "fish"),
    ("Hybrid: Fish → IndexTTS-2", "hybrid"),
]


class EmotionBars(QWidget):
    """Tiny bar chart of the 8 emotion dims, colored by the span palette."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._values = [0.0] * len(EMOTIONS)
        self.setMinimumHeight(70)

    def set_values(self, values: list[float]) -> None:
        self._values = list(values) + [0.0] * (len(EMOTIONS) - len(values))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: ANN001
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        w = self.width()
        h = self.height()
        n = len(EMOTIONS)
        gap = 4
        bar_w = max(2, (w - gap * (n + 1)) / n)
        painter.fillRect(self.rect(), QColor("#1e1e22"))
        for i, v in enumerate(self._values):
            x = gap + i * (bar_w + gap)
            bar_h = max(1.0, v * (h - 6))
            painter.fillRect(
                int(x), int(h - bar_h - 2), int(bar_w), int(bar_h), QColor(color_hex(i))
            )
        painter.end()


class Inspector(QWidget):
    applyRequested = Signal(object)  # Inflection
    setDefaultRequested = Signal(object)  # Inflection
    previewRequested = Signal(object)  # Inflection
    auditionPerformanceRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._loading = False
        self._sliders: list[QSlider] = []
        self._slider_labels: list[QLabel] = []
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        self._scope_label = QLabel("Document default")
        self._scope_label.setStyleSheet("color:#9aa; font-style:italic;")
        root.addWidget(self._scope_label)

        # Emotion sliders
        emo_group = QGroupBox("Emotion")
        grid = QGridLayout(emo_group)
        grid.setVerticalSpacing(2)
        for i, name in enumerate(EMOTIONS):
            label = QLabel(name.capitalize())
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(0)
            value = QLabel("0")
            value.setFixedWidth(28)
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            slider.valueChanged.connect(
                lambda v, lbl=value: lbl.setText(str(v))
            )
            slider.valueChanged.connect(self._on_live_change)
            row = i
            grid.addWidget(label, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value, row, 2)
            self._sliders.append(slider)
            self._slider_labels.append(value)
        root.addWidget(emo_group)

        self._bars = EmotionBars()
        root.addWidget(self._bars)

        # Free-form delivery + numeric knobs
        form_group = QGroupBox("Delivery")
        form = QFormLayout(form_group)
        self._emo_text = QLineEdit()
        self._emo_text.setPlaceholderText('e.g. "whispering, almost crying"')
        self._emo_text.textChanged.connect(self._on_live_change)
        form.addRow("Describe", self._emo_text)

        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0.0, 1.0)
        self._alpha.setSingleStep(0.05)
        self._alpha.setValue(0.8)
        form.addRow("Emotion strength (α)", self._alpha)

        self._speed = QDoubleSpinBox()
        self._speed.setRange(0.5, 1.5)
        self._speed.setSingleStep(0.05)
        self._speed.setValue(1.0)
        form.addRow("Speed ×", self._speed)

        self._pause = QSpinBox()
        self._pause.setRange(0, 10_000)
        self._pause.setSingleStep(50)
        self._pause.setSuffix(" ms")
        form.addRow("Pause after", self._pause)

        self._engine = QComboBox()
        for label, _value in _ENGINE_ITEMS:
            self._engine.addItem(label)
        self._engine.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow("Engine", self._engine)

        root.addWidget(form_group)

        # Hybrid audition (only meaningful for engine == hybrid)
        self._audition_btn = QPushButton("Audition performance (stage 1)…")
        self._audition_btn.clicked.connect(lambda: self.auditionPerformanceRequested.emit())
        self._audition_btn.setVisible(False)
        root.addWidget(self._audition_btn)

        # Action buttons
        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply to selection")
        self._apply_btn.clicked.connect(lambda: self.applyRequested.emit(self.current_inflection()))
        self._default_btn = QPushButton("Set as default")
        self._default_btn.clicked.connect(
            lambda: self.setDefaultRequested.emit(self.current_inflection())
        )
        btn_row.addWidget(self._apply_btn)
        btn_row.addWidget(self._default_btn)
        root.addLayout(btn_row)

        self._preview_btn = QPushButton("▶ Preview this segment")
        self._preview_btn.clicked.connect(
            lambda: self.previewRequested.emit(self.current_inflection())
        )
        root.addWidget(self._preview_btn)

        root.addStretch(1)

    # -- state in/out ------------------------------------------------------
    def set_context(self, inflection: Inflection, *, is_default: bool, has_selection: bool) -> None:
        """Load an inflection into the widgets and adjust button availability."""
        self._loading = True
        try:
            vec = inflection.emotion_vector or [0.0] * len(EMOTIONS)
            for i, slider in enumerate(self._sliders):
                v = int(round(vec[i] * 100)) if i < len(vec) else 0
                slider.setValue(v)
                self._slider_labels[i].setText(str(v))
            self._emo_text.setText(inflection.emo_text or "")
            self._alpha.setValue(float(inflection.emo_alpha))
            self._speed.setValue(float(inflection.speed))
            self._pause.setValue(int(inflection.pause_after_ms))
            self._set_engine_value(inflection.engine)
            self._bars.set_values([v / 100.0 for v in self._slider_values()])
        finally:
            self._loading = False

        self._scope_label.setText(
            "Document default" if is_default else ("Selection" if has_selection else "Caret span")
        )
        self._apply_btn.setEnabled(has_selection)
        self._audition_btn.setVisible(self._engine_value() == "hybrid")

    def current_inflection(self) -> Inflection:
        vec = [v / 100.0 for v in self._slider_values()]
        has_emotion = any(v > 1e-6 for v in vec)
        return Inflection(
            emotion_vector=vec if has_emotion else None,
            emo_text=self._emo_text.text().strip() or None,
            emo_alpha=float(self._alpha.value()),
            speed=float(self._speed.value()),
            pause_after_ms=int(self._pause.value()),
            engine=self._engine_value(),
        )

    # -- helpers -----------------------------------------------------------
    def _slider_values(self) -> list[int]:
        return [s.value() for s in self._sliders]

    def _engine_value(self) -> str | None:
        return _ENGINE_ITEMS[self._engine.currentIndex()][1]

    def _set_engine_value(self, value: str | None) -> None:
        for i, (_label, v) in enumerate(_ENGINE_ITEMS):
            if v == value:
                self._engine.setCurrentIndex(i)
                return
        self._engine.setCurrentIndex(0)

    def _on_live_change(self, *_: object) -> None:
        if self._loading:
            return
        self._bars.set_values([v / 100.0 for v in self._slider_values()])

    def _on_engine_changed(self, *_: object) -> None:
        self._audition_btn.setVisible(self._engine_value() == "hybrid")
