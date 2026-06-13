"""Settings dialog: ffmpeg path, output device, engine defaults, mastering.

Writes back into :class:`Config.settings` and persists via ``save_settings``.
Model directories are shown read-only (they derive from the data home) with a
button to reveal the folder.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..config import Config


def _query_output_devices() -> list[tuple[int, str]]:
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        return [
            (i, d["name"])
            for i, d in enumerate(devices)
            if d.get("max_output_channels", 0) > 0
        ]
    except Exception:
        return []


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        self._config = config
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        s = self._config.settings

        # ffmpeg
        self._ffmpeg = QLineEdit(s.ffmpeg_path)
        form.addRow("ffmpeg path", self._ffmpeg)
        self._ffprobe = QLineEdit(s.ffprobe_path)
        form.addRow("ffprobe path", self._ffprobe)

        # output device
        self._device = QComboBox()
        self._device.addItem("System default", None)
        for idx, name in _query_output_devices():
            self._device.addItem(f"{idx}: {name}", idx)
        self._select_device(s.output_device)
        form.addRow("Output device", self._device)

        # engine defaults
        self._engine = QComboBox()
        self._engine.addItem("IndexTTS-2 (Final)", "indextts2")
        self._engine.addItem("Chatterbox (Draft)", "chatterbox")
        self._engine.setCurrentIndex(0 if s.default_engine == "indextts2" else 1)
        form.addRow("Default engine", self._engine)

        self._cuda = QCheckBox("Use CUDA when available")
        self._cuda.setChecked(s.use_cuda)
        form.addRow("", self._cuda)
        self._fp16 = QCheckBox("Use fp16 (recommended on 12 GB cards)")
        self._fp16.setChecked(s.use_fp16)
        form.addRow("", self._fp16)
        self._cuda_kernel = QCheckBox("IndexTTS-2 CUDA kernel (faster; needs a compiled kernel)")
        self._cuda_kernel.setChecked(s.use_cuda_kernel)
        form.addRow("", self._cuda_kernel)

        self._cfg_weight = QDoubleSpinBox()
        self._cfg_weight.setRange(0.0, 1.0)
        self._cfg_weight.setSingleStep(0.05)
        self._cfg_weight.setValue(s.chatterbox_cfg_weight)
        form.addRow("Chatterbox cfg_weight", self._cfg_weight)

        # mastering
        self._crossfade = QSpinBox()
        self._crossfade.setRange(0, 100)
        self._crossfade.setSuffix(" ms")
        self._crossfade.setValue(s.crossfade_ms)
        form.addRow("Crossfade", self._crossfade)

        self._lufs = QDoubleSpinBox()
        self._lufs.setRange(-30.0, -6.0)
        self._lufs.setValue(s.target_lufs)
        form.addRow("Target loudness (LUFS)", self._lufs)

        root.addLayout(form)

        self._paths_label = QLabel(f"Models & data: {self._config.paths.home}")
        self._paths_label.setStyleSheet("color:#9aa; font-size:11px;")
        self._paths_label.setWordWrap(True)
        root.addWidget(self._paths_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _select_device(self, device: int | None) -> None:
        for i in range(self._device.count()):
            if self._device.itemData(i) == device:
                self._device.setCurrentIndex(i)
                return

    def _save(self) -> None:
        s = self._config.settings
        s.ffmpeg_path = self._ffmpeg.text().strip() or "ffmpeg"
        s.ffprobe_path = self._ffprobe.text().strip() or "ffprobe"
        s.output_device = self._device.currentData()
        s.default_engine = self._engine.currentData()
        s.use_cuda = self._cuda.isChecked()
        s.use_fp16 = self._fp16.isChecked()
        s.use_cuda_kernel = self._cuda_kernel.isChecked()
        s.chatterbox_cfg_weight = float(self._cfg_weight.value())
        s.crossfade_ms = int(self._crossfade.value())
        s.target_lufs = float(self._lufs.value())
        self._config.save_settings()
        self.accept()
