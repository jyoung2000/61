"""Main window: docks + toolbar + menus wired to the pipeline and player.

Owns the long-lived services (model manager, synthesis pipeline, audio player)
and routes every long operation onto a worker thread so the UI never blocks.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolBar,
)

from ..config import Config
from ..document.project_io import load_project, save_project
from ..document.segmenter import SegmentJob, segment_document
from ..document.spans import Document, Inflection
from ..models.model_manager import ModelManager
from ..synth.pipeline import RenderResult, SynthesisPipeline
from ..synth.worker import SynthWorker, TaskWorker, start_worker
from .editor import InflectionTextEdit
from .inspector import Inspector
from .timeline import TimelineDock
from .voice_library import VoiceLibraryDock

log = logging.getLogger("inflect.ui")

_ENGINE_CHOICES = [("Draft — Chatterbox", "chatterbox"), ("Final — IndexTTS-2", "indextts2")]


def _render_segment_task(pipeline, document, job, force=False, progress=None):
    """Worker entry point for single-segment preview / re-render."""
    return pipeline.render_one(document, job, force=force)


def _render_performance_task(pipeline, document, job, progress=None):
    """Worker entry point for the hybrid stage-1 'audition performance' button."""
    return pipeline.render_performance(document, job)


class MainWindow(QMainWindow):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.setWindowTitle("Inflect Studio")
        self.resize(1280, 820)

        # Services
        self.model_manager = ModelManager(config)
        self.voice_library = self.config_voice_library()
        self.pipeline = SynthesisPipeline(config, self.model_manager, self.voice_library)
        from ..audio.playback import AudioPlayer

        self.player = AudioPlayer(device=config.settings.output_device, parent=self)

        # State
        self.current_engine = config.settings.default_engine
        self.project_path: Path | None = None
        self.last_result: RenderResult | None = None
        self._synth_thread = None
        self._synth_worker: SynthWorker | None = None
        self._task_thread = None
        self._syncing_voice = False

        self._build_ui()
        self._wire()
        self._refresh_inspector()
        self._start_vram_timer()

    def config_voice_library(self):
        from ..ingest.profile import VoiceLibrary

        return VoiceLibrary(self.config.paths.voices)

    # -- construction ------------------------------------------------------
    def _build_ui(self) -> None:
        self.editor = InflectionTextEdit()
        self.setCentralWidget(self.editor)

        # Left dock: voice library
        self.library_dock = VoiceLibraryDock(self.voice_library)
        left = QDockWidget("Voice Library", self)
        left.setWidget(self.library_dock)
        left.setObjectName("voice_library")
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left)

        # Right dock: inspector
        self.inspector = Inspector()
        right = QDockWidget("Inflection", self)
        right.setWidget(self.inspector)
        right.setObjectName("inspector")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right)

        # Bottom dock: timeline
        self.timeline = TimelineDock()
        bottom = QDockWidget("Timeline", self)
        bottom.setWidget(self.timeline)
        bottom.setObjectName("timeline")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom)
        self._timeline_dock = bottom

        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        tb.addWidget(QLabel(" Voice: "))
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(180)
        tb.addWidget(self.voice_combo)
        self._reload_voice_combo()

        tb.addSeparator()
        tb.addWidget(QLabel(" Mode: "))
        self.engine_combo = QComboBox()
        for label, value in _ENGINE_CHOICES:
            self.engine_combo.addItem(label, value)
        idx = next((i for i, (_l, v) in enumerate(_ENGINE_CHOICES) if v == self.current_engine), 1)
        self.engine_combo.setCurrentIndex(idx)
        tb.addWidget(self.engine_combo)

        tb.addSeparator()
        self.synth_btn = QPushButton("▶ Synthesize All")
        tb.addWidget(self.synth_btn)
        self.cancel_btn = QPushButton("⏹ Cancel")
        self.cancel_btn.setEnabled(False)
        tb.addWidget(self.cancel_btn)
        self.play_btn = QPushButton("⏯ Play")
        tb.addWidget(self.play_btn)
        self.export_btn = QPushButton("Export…")
        tb.addWidget(self.export_btn)

    def _build_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        self._add_action(file_menu, "New", self._new_project, QKeySequence.StandardKey.New)
        self._add_action(file_menu, "Open…", self._open_project, QKeySequence.StandardKey.Open)
        self._add_action(file_menu, "Save", self._save_project, QKeySequence.StandardKey.Save)
        self._add_action(file_menu, "Save As…", self._save_project_as, QKeySequence.StandardKey.SaveAs)
        file_menu.addSeparator()
        self._add_action(file_menu, "Import voice…", self._import_voice)
        self._add_action(file_menu, "Export audio…", self._export)
        file_menu.addSeparator()
        self._add_action(file_menu, "Settings…", self._open_settings)
        self._add_action(file_menu, "Quit", self.close, QKeySequence.StandardKey.Quit)

        help_menu = menubar.addMenu("&Help")
        self._add_action(help_menu, "Open log folder", self._open_log_folder)
        self._add_action(help_menu, "About", self._about)

    def _build_statusbar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.statusBar().addWidget(self.progress)
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.vram_label = QLabel("VRAM: n/a")
        self.statusBar().addPermanentWidget(self.vram_label)

    def _add_action(self, menu, text, slot, shortcut=None) -> QAction:
        act = QAction(text, self)
        act.triggered.connect(slot)
        if shortcut is not None:
            act.setShortcut(shortcut)
        menu.addAction(act)
        return act

    # -- wiring ------------------------------------------------------------
    def _wire(self) -> None:
        # Toolbar
        self.synth_btn.clicked.connect(self._synthesize_all)
        self.cancel_btn.clicked.connect(self._cancel_synth)
        self.play_btn.clicked.connect(self.player.toggle)
        self.export_btn.clicked.connect(self._export)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_combo_changed)

        # Library dock
        self.library_dock.importRequested.connect(self._import_voice)
        self.library_dock.profileSelected.connect(self._on_voice_selected)
        self.library_dock.previewRequested.connect(self._preview_profile)

        # Editor
        self.editor.selectionContextChanged.connect(self._refresh_inspector)
        self.editor.applyInflectionRequested.connect(
            lambda: self.editor.apply_inflection_to_selection(self.inspector.current_inflection())
        )
        self.editor.clearInflectionRequested.connect(self.editor.clear_selection_inflection)
        self.editor.insertPauseRequested.connect(self._insert_pause)
        self.editor.previewSelectionRequested.connect(
            lambda: self._preview_selection(self.inspector.current_inflection())
        )

        # Inspector
        self.inspector.applyRequested.connect(self.editor.apply_inflection_to_selection)
        self.inspector.setDefaultRequested.connect(self.editor.set_default_inflection)
        self.inspector.previewRequested.connect(self._preview_selection)
        self.inspector.auditionPerformanceRequested.connect(self._audition_performance)

        # Timeline + player
        self.timeline.seekRequested.connect(self.player.seek)
        self.timeline.rerenderSegmentRequested.connect(self._rerender_segment)
        self.player.position_changed.connect(self.timeline.set_position)

        # Space toggles playback when the timeline has focus.
        sc = QShortcut(QKeySequence(Qt.Key.Key_Space), self._timeline_dock)
        sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        sc.activated.connect(self.player.toggle)

    # -- voice management --------------------------------------------------
    def _reload_voice_combo(self, select_id: str | None = None) -> None:
        self._syncing_voice = True
        try:
            self.voice_combo.clear()
            self.voice_combo.addItem("— none —", None)
            for profile in self.voice_library.list_profiles():
                self.voice_combo.addItem(profile.name, profile.id)
            if select_id:
                for i in range(self.voice_combo.count()):
                    if self.voice_combo.itemData(i) == select_id:
                        self.voice_combo.setCurrentIndex(i)
                        break
        finally:
            self._syncing_voice = False

    def _on_voice_combo_changed(self, _idx: int) -> None:
        if self._syncing_voice:
            return
        pid = self.voice_combo.currentData()
        self.editor.model.voice_profile_id = pid
        self._syncing_voice = True
        try:
            if pid:
                self.library_dock.select_profile(pid)
        finally:
            self._syncing_voice = False

    def _on_voice_selected(self, profile_id: str) -> None:
        pid = profile_id or None
        self.editor.model.voice_profile_id = pid
        if not self._syncing_voice:
            self._reload_voice_combo(select_id=pid)

    def _import_voice(self) -> None:
        from .ingest_dialog import IngestDialog

        dlg = IngestDialog(self.voice_library, self.config, self)
        if dlg.exec() and dlg.created_profile is not None:
            self.library_dock.refresh(select_id=dlg.created_profile.id)
            self._reload_voice_combo(select_id=dlg.created_profile.id)
            self.editor.model.voice_profile_id = dlg.created_profile.id
            self.status_label.setText(f"Imported voice '{dlg.created_profile.name}'.")

    def _preview_profile(self, profile_id: str) -> None:
        if not profile_id:
            return
        import soundfile as sf

        path = self.voice_library.audition_path(profile_id)
        if not path.exists():
            path = self.voice_library.reference_path(profile_id)
        if not path.exists():
            return
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        self.player.set_audio(audio, sr)
        self.player.play()

    # -- inspector ---------------------------------------------------------
    def _refresh_inspector(self) -> None:
        inflection = self.editor.effective_inflection()
        is_default = self.editor.current_span() is None
        self.inspector.set_context(
            inflection, is_default=is_default, has_selection=self.editor.has_selection()
        )

    def _insert_pause(self) -> None:
        ms, ok = QInputDialog.getInt(self, "Insert pause", "Pause (ms):", 300, 0, 10_000, 50)
        if ok:
            self.editor.insert_pause_at_caret(ms)

    # -- synthesis ---------------------------------------------------------
    def _engine_params(self, engine: str | None = None) -> dict:
        engine = engine or self.current_engine
        if engine == "chatterbox":
            return {"cfg_weight": self.config.settings.chatterbox_cfg_weight}
        return {}

    def _on_engine_changed(self, _idx: int) -> None:
        self.current_engine = self.engine_combo.currentData()
        self.status_label.setText(f"Engine: {self.current_engine}")

    def _synthesize_all(self) -> None:
        if not self.editor.model.voice_profile_id:
            QMessageBox.warning(self, "No voice", "Select or import a voice profile first.")
            return
        if not self.editor.model.text.strip():
            QMessageBox.information(self, "Nothing to synthesize", "The script is empty.")
            return
        document = copy.deepcopy(self.editor.model)
        worker = SynthWorker(
            self.pipeline, document, self.current_engine, self._engine_params()
        )
        worker.progress.connect(self._on_synth_progress)
        worker.finished.connect(self._on_synth_finished)
        worker.failed.connect(self._on_task_failed)
        worker.cancelled.connect(self._on_synth_cancelled)
        self._synth_worker = worker
        self._set_synth_busy(True)
        self._synth_thread = start_worker(worker)

    def _cancel_synth(self) -> None:
        if self._synth_worker is not None:
            self._synth_worker.cancel()
            self.status_label.setText("Cancelling…")

    def _on_synth_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(message)

    def _on_synth_finished(self, result: RenderResult) -> None:
        self.last_result = result
        self._set_synth_busy(False)
        self.timeline.set_result(result)
        self.player.set_audio(result.audio, result.sample_rate)
        self.status_label.setText(
            f"Done — {len(result.segments)} segment(s), {result.duration_s:.1f}s."
        )

    def _on_synth_cancelled(self) -> None:
        self._set_synth_busy(False)
        self.status_label.setText("Cancelled.")

    def _set_synth_busy(self, busy: bool) -> None:
        self.synth_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.progress.setVisible(busy)

    def _preview_selection(self, inflection: Inflection) -> None:
        if not self.editor.model.voice_profile_id:
            QMessageBox.warning(self, "No voice", "Select or import a voice profile first.")
            return
        start, end = self.editor.selection_range()
        if end <= start:
            span = self.editor.current_span()
            if span is None:
                self.status_label.setText("Select some text to preview.")
                return
            start, end = span.start, span.end
        text = self.editor.model.text[start:end].strip()
        if not text:
            return
        engine = inflection.engine or self.current_engine
        job = SegmentJob(
            seg_id=0,
            text=text,
            inflection=inflection,
            voice_profile_id=self.editor.model.voice_profile_id,
            engine=engine,
            engine_params=self._engine_params(engine),
            char_start=start,
            char_end=end,
        )
        self._run_segment_task(job, then_play=True)

    def _audition_performance(self) -> None:
        """Render and play only the Fish stage-1 performance for the selection."""
        if not self.editor.model.voice_profile_id:
            QMessageBox.warning(self, "No voice", "Select or import a voice profile first.")
            return
        start, end = self.editor.selection_range()
        if end <= start:
            span = self.editor.current_span()
            if span is None:
                self.status_label.setText("Select some text to audition.")
                return
            start, end = span.start, span.end
        text = self.editor.model.text[start:end].strip()
        if not text:
            return
        inflection = self.inspector.current_inflection()
        job = SegmentJob(
            seg_id=0,
            text=text,
            inflection=inflection,
            voice_profile_id=self.editor.model.voice_profile_id,
            engine="hybrid",
            char_start=start,
            char_end=end,
        )
        document = copy.deepcopy(self.editor.model)
        worker = TaskWorker(_render_performance_task, self.pipeline, document, job)
        worker.progress.connect(lambda m, _f: self.status_label.setText(m))
        worker.finished.connect(self._on_segment_preview_ready)
        worker.failed.connect(self._on_task_failed)
        self.status_label.setText("Rendering Fish performance (stage 1)…")
        self._task_thread = start_worker(worker)

    def _rerender_segment(self, seg_id: int) -> None:
        jobs = segment_document(self.editor.model, self.current_engine, self._engine_params())
        job = next((j for j in jobs if j.seg_id == seg_id), None)
        if job is None:
            return
        # Invalidate just this segment's cache, then re-render everything (the
        # rest is served from cache, so only this segment is recomputed).
        cache = self.pipeline.cache_dir / f"{job.hash}.wav"
        if cache.exists():
            cache.unlink(missing_ok=True)
        self._synthesize_all()

    def _run_segment_task(self, job: SegmentJob, then_play: bool) -> None:
        document = copy.deepcopy(self.editor.model)
        worker = TaskWorker(_render_segment_task, self.pipeline, document, job)
        worker.progress.connect(lambda m, _f: self.status_label.setText(m))
        if then_play:
            worker.finished.connect(self._on_segment_preview_ready)
        worker.failed.connect(self._on_task_failed)
        self.status_label.setText("Rendering preview…")
        self._task_thread = start_worker(worker)

    def _on_segment_preview_ready(self, seg) -> None:  # noqa: ANN001
        self.player.set_audio(seg.audio, seg.sample_rate)
        self.player.play()
        self.status_label.setText(f"Preview ready ({seg.duration_s:.1f}s).")

    # -- project io --------------------------------------------------------
    def _new_project(self) -> None:
        pid = self.editor.model.voice_profile_id
        self.editor.set_document_model(Document(voice_profile_id=pid))
        self.last_result = None
        self.project_path = None
        self.status_label.setText("New project.")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "", "Inflect projects (*.inflect)"
        )
        if not path:
            return
        try:
            pf = load_project(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self.editor.set_document_model(pf.document)
        self.current_engine = pf.engine
        self._sync_engine_combo()
        if pf.document.voice_profile_id:
            self._reload_voice_combo(select_id=pf.document.voice_profile_id)
            self.library_dock.select_profile(pf.document.voice_profile_id)
        self.project_path = Path(path)
        self.status_label.setText(f"Opened {Path(path).name}.")

    def _save_project(self) -> None:
        if self.project_path is None:
            self._save_project_as()
            return
        save_project(
            self.project_path, self.editor.model, self.current_engine, self._engine_params()
        )
        self.status_label.setText(f"Saved {self.project_path.name}.")

    def _save_project_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "", "Inflect projects (*.inflect)"
        )
        if not path:
            return
        pf = save_project(path, self.editor.model, self.current_engine, self._engine_params())
        self.project_path = pf.path
        self.status_label.setText(f"Saved {pf.path.name}.")

    def _sync_engine_combo(self) -> None:
        for i in range(self.engine_combo.count()):
            if self.engine_combo.itemData(i) == self.current_engine:
                self.engine_combo.setCurrentIndex(i)
                return

    # -- export ------------------------------------------------------------
    def _export(self) -> None:
        if self.last_result is None or self.last_result.audio.size == 0:
            QMessageBox.information(self, "Nothing to export", "Synthesize the script first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export audio", "", "WAV (*.wav);;MP3 (*.mp3)"
        )
        if not path:
            return
        try:
            self._write_export(Path(path))
            self.status_label.setText(f"Exported {Path(path).name}.")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _write_export(self, path: Path) -> None:
        import soundfile as sf

        from ..ingest.extract import encode_output

        result = self.last_result
        assert result is not None
        if path.suffix.lower() == ".mp3":
            tmp = self.config.paths.tmp / "export_tmp.wav"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(tmp), result.audio, result.sample_rate)
            encode_output(tmp, path, fmt="mp3", ffmpeg_path=self.config.settings.ffmpeg_path)
        else:
            sf.write(str(path), result.audio, result.sample_rate)

    # -- settings / help ---------------------------------------------------
    def _open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        dlg = SettingsDialog(self.config, self)
        if dlg.exec():
            self.player.set_device(self.config.settings.output_device)
            self.status_label.setText("Settings saved.")

    def _open_log_folder(self) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.config.paths.logs)))

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "Inflect Studio",
            "Inflect Studio — local voice cloning + inflection-aware TTS.\n"
            "Engines: Chatterbox (Draft), IndexTTS-2 (Final), Fish (optional).",
        )

    # -- errors / vram -----------------------------------------------------
    def _on_task_failed(self, message: str, details: str) -> None:
        self._set_synth_busy(False)
        from .error_dialog import show_error

        show_error(self, "Operation failed", message, details)
        self.status_label.setText(f"Error: {message}")

    def _start_vram_timer(self) -> None:
        self._vram_timer = QTimer(self)
        self._vram_timer.setInterval(1500)
        self._vram_timer.timeout.connect(self._poll_vram)
        self._vram_timer.start()

    def _poll_vram(self) -> None:
        self.vram_label.setText(str(self.model_manager.vram()))

    def closeEvent(self, event) -> None:  # noqa: ANN001
        try:
            self.player.stop()
            self.model_manager.unload_current()
        finally:
            super().closeEvent(event)
