"""QThread workers: synthesis + generic long tasks (ingest, downloads).

The GUI must never block, so every long operation runs on a worker object moved
onto a :class:`QThread`. Workers communicate purely through signals (queued
across the thread boundary), report ``segment i/N`` progress, and support
cancellation that takes effect between segments.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable

from PySide6.QtCore import QObject, QThread, Signal

from ..document.spans import Document
from .pipeline import PipelineCancelled, RenderResult, SynthesisPipeline


class SynthWorker(QObject):
    """Renders a whole document on a background thread."""

    progress = Signal(int, int, str)  # done, total, message
    finished = Signal(object)  # RenderResult
    failed = Signal(str, str)  # short message, full diagnostics
    cancelled = Signal()

    def __init__(
        self,
        pipeline: SynthesisPipeline,
        document: Document,
        engine: str,
        engine_params: dict | None = None,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.document = document
        self.engine = engine
        self.engine_params = engine_params or {}
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        """Slot connected to ``QThread.started``."""
        try:
            result: RenderResult = self.pipeline.render(
                self.document,
                self.engine,
                self.engine_params,
                should_cancel=lambda: self._cancel,
                progress=lambda d, t, m: self.progress.emit(d, t, m),
            )
            if self._cancel:
                self.cancelled.emit()
            else:
                self.finished.emit(result)
        except PipelineCancelled:
            self.cancelled.emit()
        except Exception as exc:  # surface a friendly message + copyable details
            self.failed.emit(str(exc), traceback.format_exc())


class TaskWorker(QObject):
    """Runs an arbitrary callable that accepts a ``progress(msg, frac)`` cb.

    Used by the ingest wizard (extract/isolate/analyze) and model downloads.
    """

    progress = Signal(str, float)  # message, fraction (-1 == indeterminate)
    finished = Signal(object)  # whatever the callable returns
    failed = Signal(str, str)

    def __init__(self, fn: Callable[..., object], *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(
                *self._args,
                progress=lambda msg, frac=-1.0: self.progress.emit(msg, frac),
                **self._kwargs,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc), traceback.format_exc())


def start_worker(worker: QObject) -> QThread:
    """Move ``worker`` onto a fresh thread, wire run/cleanup, and start it.

    The worker must expose a ``run()`` slot and one or more terminal signals
    (``finished``/``failed``/``cancelled``). The thread quits when any terminal
    signal fires and both are deleted afterwards. Returns the thread so the
    caller can keep a reference (otherwise it would be garbage collected).
    """
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def _quit(*_: object) -> None:
        thread.quit()

    for sig_name in ("finished", "failed", "cancelled"):
        sig = getattr(worker, sig_name, None)
        if sig is not None:
            sig.connect(_quit)

    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
