"""Bottom-dock Timeline: pyqtgraph waveform + segment regions + playhead.

Shows the assembled mix as a downsampled waveform, overlays a colored band per
segment (matching the editor underline colors), draws a movable playhead, lets
the user click to seek, and offers a per-segment "re-render" context action.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QMenu, QVBoxLayout, QWidget

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - pyqtgraph optional at import time
    pg = None

from ..synth.pipeline import RenderResult
from .format import color_hex

# Max points to draw -- waveforms are downsampled to keep the plot snappy.
_MAX_POINTS = 6000


class TimelineDock(QWidget):
    seekRequested = Signal(float)  # seconds
    rerenderSegmentRequested = Signal(int)  # seg_id

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._result: RenderResult | None = None
        self._seg_bounds: list[tuple[int, float, float, int]] = []  # seg_id,start,end,color_idx
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        if pg is None:
            root.addWidget(QLabel("pyqtgraph is not installed — waveform unavailable."))
            self._plot = None
            return
        pg.setConfigOptions(antialias=True, background="#161619", foreground="#aaa")
        self._plot = pg.PlotWidget()
        self._plot.setMenuEnabled(False)
        self._plot.showGrid(x=True, y=False, alpha=0.15)
        self._plot.setMouseEnabled(x=True, y=False)
        self._plot.setYRange(-1.0, 1.0)
        self._plot.getPlotItem().hideAxis("left")
        self._curve = self._plot.plot([], [], pen=pg.mkPen("#6cf", width=1))
        self._playhead = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen("#fff", width=2)
        )
        self._plot.addItem(self._playhead)
        self._plot.scene().sigMouseClicked.connect(self._on_click)
        root.addWidget(self._plot)

    # -- data --------------------------------------------------------------
    def set_result(self, result: RenderResult) -> None:
        self._result = result
        if self._plot is None:
            return
        self._clear_regions()
        audio = result.audio
        sr = result.sample_rate or 1
        if audio.size == 0:
            self._curve.setData([], [])
            return
        # Downsample by min/max envelope for an honest-looking waveform.
        xs, ys = _envelope(audio, sr, _MAX_POINTS)
        self._curve.setData(xs, ys)
        self._plot.setXRange(0, audio.size / sr, padding=0.01)

        self._seg_bounds = []
        for (seg_id, start, end), seg in zip(result.segment_offsets(), result.segments):
            color = color_hex(_seg_color_idx(seg))
            region = pg.LinearRegionItem(
                values=(start, end),
                brush=pg.mkBrush(_brush_rgba(color, 48)),
                movable=False,
            )
            region.setZValue(-10)
            self._plot.addItem(region)
            self._seg_bounds.append((seg_id, start, end, _seg_color_idx(seg)))

    def set_position(self, seconds: float) -> None:
        if self._plot is not None:
            self._playhead.setPos(seconds)

    # -- interaction -------------------------------------------------------
    def _on_click(self, event) -> None:  # noqa: ANN001
        if self._plot is None or self._result is None:
            return
        vb = self._plot.getPlotItem().vb
        point = vb.mapSceneToView(event.scenePos())
        seconds = float(point.x())
        if event.button() == Qt.MouseButton.RightButton:
            seg_id = self._segment_at(seconds)
            if seg_id is not None:
                menu = QMenu(self)
                act = menu.addAction(f"Re-render segment {seg_id}")
                if menu.exec(event.screenPos().toPoint()) == act:
                    self.rerenderSegmentRequested.emit(seg_id)
            return
        self.seekRequested.emit(max(0.0, seconds))

    def _segment_at(self, seconds: float) -> int | None:
        for seg_id, start, end, _c in self._seg_bounds:
            if start <= seconds < end:
                return seg_id
        return None

    def _clear_regions(self) -> None:
        if self._plot is None:
            return
        for item in list(self._plot.items()):
            if isinstance(item, pg.LinearRegionItem):
                self._plot.removeItem(item)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _seg_color_idx(seg) -> int:  # noqa: ANN001
    # Color the band by engine when no per-emotion color is available.
    return {"chatterbox": 3, "indextts2": 2, "fish": 4, "hybrid": 5}.get(seg.engine, 3)


def _brush_rgba(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    """Parse ``#rrggbb`` into an ``(r, g, b, alpha)`` tuple for a translucent brush."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, int(max(0, min(255, alpha))))


def _envelope(audio: np.ndarray, sr: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max envelope downsample, returning x (seconds) and y arrays."""
    n = audio.size
    if n <= max_points:
        xs = np.arange(n) / sr
        return xs, audio
    bucket = int(np.ceil(n / (max_points // 2)))
    usable = (n // bucket) * bucket
    trimmed = audio[:usable].reshape(-1, bucket)
    mins = trimmed.min(axis=1)
    maxs = trimmed.max(axis=1)
    ys = np.empty(mins.size * 2, dtype=np.float32)
    ys[0::2] = mins
    ys[1::2] = maxs
    centers = (np.arange(mins.size) * bucket) / sr
    xs = np.repeat(centers, 2)
    return xs, ys
