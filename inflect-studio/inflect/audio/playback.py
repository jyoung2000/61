"""Non-blocking audio playback via sounddevice with a position callback.

A single :class:`AudioPlayer` owns the current mix. Playback runs on
sounddevice's own callback thread; a small QTimer on the GUI thread polls the
frame cursor and emits :pyattr:`position_changed` so the timeline can move a
playhead without us emitting Qt signals from the audio thread.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal


class AudioPlayer(QObject):
    position_changed = Signal(float)  # current position, seconds
    state_changed = Signal(bool)  # True == playing
    finished = Signal()

    def __init__(self, device: int | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._audio = np.zeros(0, dtype=np.float32)
        self._sr = 24_000
        self._device = device
        self._frame = 0
        self._lock = threading.Lock()
        self._stream = None
        self._playing = False

        self._timer = QTimer(self)
        self._timer.setInterval(40)  # ~25 fps playhead
        self._timer.timeout.connect(self._tick)

    # -- configuration -----------------------------------------------------
    def set_device(self, device: int | None) -> None:
        self._device = device

    def set_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Load a new mix, stopping any current playback."""
        self.stop()
        with self._lock:
            self._audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            self._sr = int(sample_rate)
            self._frame = 0

    @property
    def duration_s(self) -> float:
        return len(self._audio) / self._sr if self._sr else 0.0

    @property
    def is_playing(self) -> bool:
        return self._playing

    # -- transport ---------------------------------------------------------
    def play(self, start_s: float | None = None) -> None:
        if self._audio.size == 0:
            return
        try:
            import sounddevice as sd
        except Exception:
            return
        if start_s is not None:
            self.seek(start_s)
        if self._playing:
            return

        def _callback(outdata, frames, time_info, status):  # noqa: ANN001
            with self._lock:
                start = self._frame
                end = min(start + frames, len(self._audio))
                chunk = self._audio[start:end]
                self._frame = end
            n = len(chunk)
            outdata[:n, 0] = chunk
            if n < frames:
                outdata[n:, 0] = 0.0
                raise sd.CallbackStop()

        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=_callback,
            finished_callback=self._on_stream_finished,
        )
        self._stream.start()
        self._playing = True
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        self._stop_stream()
        self._playing = False
        self._timer.stop()
        self.state_changed.emit(False)
        self.position_changed.emit(self.position_s)

    def stop(self) -> None:
        self._stop_stream()
        self._playing = False
        self._timer.stop()
        with self._lock:
            self._frame = 0
        self.state_changed.emit(False)
        self.position_changed.emit(0.0)

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._frame = int(max(0.0, min(seconds, self.duration_s)) * self._sr)
        self.position_changed.emit(self.position_s)

    @property
    def position_s(self) -> float:
        with self._lock:
            return self._frame / self._sr if self._sr else 0.0

    # -- internals ---------------------------------------------------------
    def _stop_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _on_stream_finished(self) -> None:
        # Called on the audio thread when playback runs off the end.
        self._playing = False

    def _tick(self) -> None:
        self.position_changed.emit(self.position_s)
        if not self._playing:
            self._timer.stop()
            self.state_changed.emit(False)
            if self._frame >= len(self._audio) and len(self._audio) > 0:
                self.finished.emit()
