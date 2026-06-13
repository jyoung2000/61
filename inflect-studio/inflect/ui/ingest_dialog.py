"""Import wizard: video/audio file → analyzed candidate clips → Voice Profile.

Flow: pick a file → (optional Demucs) → analysis runs on a worker thread with a
progress readout → up to three scored candidate clips are auditioned and trimmed
→ name + required consent checkbox → save into the Voice Library.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..audio.playback import AudioPlayer
from ..config import Config
from ..ingest.extract import ffmpeg_available
from ..ingest.profile import VoiceLibrary, VoiceProfile
from ..ingest.source import IngestAnalysis, analyze_source, slice_seconds
from ..synth.worker import TaskWorker, start_worker

_MEDIA_FILTER = "Media (*.mp4 *.mkv *.mov *.mp3 *.wav *.m4a *.flac *.webm);;All files (*)"


class _CandidateRow(QWidget):
    """One scored candidate: radio + play + trim start/end."""

    playClicked = Signal(object)  # _CandidateRow

    def __init__(self, index: int, start_s: float, end_s: float, score: float, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 2, 2, 2)
        self.radio = QRadioButton(f"Clip {index + 1}  ·  score {score:.2f}")
        row.addWidget(self.radio, 1)

        self.start = QDoubleSpinBox()
        self.start.setRange(0, 1e6)
        self.start.setDecimals(2)
        self.start.setSuffix(" s")
        self.start.setValue(round(start_s, 2))
        self.end = QDoubleSpinBox()
        self.end.setRange(0, 1e6)
        self.end.setDecimals(2)
        self.end.setSuffix(" s")
        self.end.setValue(round(end_s, 2))
        row.addWidget(QLabel("from"))
        row.addWidget(self.start)
        row.addWidget(QLabel("to"))
        row.addWidget(self.end)

        play = QPushButton("▶")
        play.setFixedWidth(32)
        play.clicked.connect(lambda: self.playClicked.emit(self))
        row.addWidget(play)


class IngestDialog(QDialog):
    def __init__(self, library: VoiceLibrary, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import voice from video / audio")
        self.setMinimumWidth(560)
        self._library = library
        self._config = config
        self._analysis: IngestAnalysis | None = None
        self._rows: list[_CandidateRow] = []
        self._thread = None
        self._player = AudioPlayer(device=config.settings.output_device, parent=self)
        self.created_profile: VoiceProfile | None = None
        self._work_dir = config.paths.tmp / f"ingest_{uuid.uuid4().hex[:8]}"
        self._build()

    # -- ui ----------------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)

        file_row = QHBoxLayout()
        self._file_label = QLabel("No file selected.")
        self._file_label.setStyleSheet("color:#9aa;")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        file_row.addWidget(self._file_label, 1)
        file_row.addWidget(browse)
        root.addLayout(file_row)

        self._isolate = QCheckBox("Isolate vocals with Demucs (recommended for noisy/music sources)")
        root.addWidget(self._isolate)

        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self._start_analysis)
        root.addWidget(self._analyze_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._status = QLabel("")
        self._status.setStyleSheet("color:#9aa;")
        root.addWidget(self._progress)
        root.addWidget(self._status)

        self._candidates_box = QGroupBox("Candidate clips")
        self._candidates_layout = QVBoxLayout(self._candidates_box)
        self._candidates_box.setVisible(False)
        self._group = QButtonGroup(self)
        root.addWidget(self._candidates_box)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Voice name"))
        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Narrator (Jane)")
        self._name.textChanged.connect(self._update_save_enabled)
        name_row.addWidget(self._name, 1)
        root.addLayout(name_row)

        self._consent = QCheckBox("I have permission to clone this voice.")
        self._consent.stateChanged.connect(self._update_save_enabled)
        root.addWidget(self._consent)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self._save_btn = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        self._save_btn.setEnabled(False)
        self._buttons.accepted.connect(self._save)
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        if not ffmpeg_available(self._config.settings.ffmpeg_path):
            self._status.setText("⚠ ffmpeg was not found on PATH — extraction will fail.")

    # -- file pick ---------------------------------------------------------
    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose media file", "", _MEDIA_FILTER)
        if path:
            self._input_path = Path(path)
            self._file_label.setText(self._input_path.name)
            self._analyze_btn.setEnabled(True)
            if not self._name.text().strip():
                self._name.setText(self._input_path.stem)

    # -- analysis ----------------------------------------------------------
    def _start_analysis(self) -> None:
        self._set_busy(True, "Analyzing…")
        worker = TaskWorker(
            analyze_source,
            str(self._input_path),
            str(self._work_dir),
            self._config.settings,
            isolate=self._isolate.isChecked(),
        )
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_analysis_done)
        worker.failed.connect(self._on_analysis_failed)
        self._thread = start_worker(worker)

    def _on_progress(self, message: str, _frac: float) -> None:
        self._status.setText(message)

    def _on_analysis_failed(self, message: str, _details: str) -> None:
        self._set_busy(False, f"Analysis failed: {message}")

    def _on_analysis_done(self, analysis: IngestAnalysis) -> None:
        self._analysis = analysis
        self._set_busy(False, "")
        msg = f"Speech ratio {analysis.speech_ratio:.0%}, {len(analysis.candidates)} candidate(s)."
        if analysis.suggest_isolation and not analysis.isolated:
            msg += "  Tip: this looks noisy — try Demucs isolation."
            self._isolate.setChecked(True)
        self._status.setText(msg)
        self._populate_candidates(analysis)

    def _populate_candidates(self, analysis: IngestAnalysis) -> None:
        # Clear old rows.
        for row in self._rows:
            row.setParent(None)
        self._rows.clear()
        for btn in list(self._group.buttons()):
            self._group.removeButton(btn)

        for i, cand in enumerate(analysis.candidates):
            row = _CandidateRow(i, cand.start_s, cand.end_s, cand.score)
            row.playClicked.connect(self._play_candidate)
            self._group.addButton(row.radio, i)
            row.radio.toggled.connect(self._update_save_enabled)
            self._candidates_layout.addWidget(row)
            self._rows.append(row)
        if self._rows:
            self._rows[0].radio.setChecked(True)
        self._candidates_box.setVisible(bool(self._rows))
        self._update_save_enabled()

    def _play_candidate(self, row: _CandidateRow) -> None:
        if self._analysis is None:
            return
        clip = slice_seconds(
            self._analysis.audio44,
            self._analysis.sample_rate_44,
            row.start.value(),
            row.end.value(),
        )
        if clip.size:
            self._player.set_audio(clip, self._analysis.sample_rate_44)
            self._player.play()

    # -- save --------------------------------------------------------------
    def _selected_row(self) -> _CandidateRow | None:
        for row in self._rows:
            if row.radio.isChecked():
                return row
        return None

    def _update_save_enabled(self, *_: object) -> None:
        ready = (
            self._analysis is not None
            and self._selected_row() is not None
            and bool(self._name.text().strip())
            and self._consent.isChecked()
        )
        self._save_btn.setEnabled(ready)

    def _save(self) -> None:
        import soundfile as sf

        row = self._selected_row()
        if self._analysis is None or row is None:
            return
        start_s, end_s = row.start.value(), row.end.value()
        ref24 = slice_seconds(self._analysis.audio24, self._analysis.sample_rate, start_s, end_s)
        aud44 = slice_seconds(
            self._analysis.audio44, self._analysis.sample_rate_44, start_s, end_s
        )
        if ref24.size == 0:
            self._status.setText("Selected clip is empty — adjust the trim.")
            return

        self._work_dir.mkdir(parents=True, exist_ok=True)
        ref_path = self._work_dir / "clip_ref.wav"
        aud_path = self._work_dir / "clip_audition.wav"
        sf.write(str(ref_path), ref24, self._analysis.sample_rate)
        sf.write(str(aud_path), aud44, self._analysis.sample_rate_44)

        self.created_profile = self._library.add(
            self._name.text().strip(),
            ref_path,
            source_filename=self._input_path.name,
            duration_s=end_s - start_s,
            sample_rate=self._analysis.sample_rate,
            consent=self._consent.isChecked(),
            audition_wav=aud_path,
        )
        self.accept()

    # -- helpers -----------------------------------------------------------
    def _set_busy(self, busy: bool, status: str) -> None:
        self._progress.setVisible(busy)
        self._progress.setRange(0, 0 if busy else 1)  # 0,0 == indeterminate
        self._analyze_btn.setEnabled(not busy and hasattr(self, "_input_path"))
        if status:
            self._status.setText(status)
