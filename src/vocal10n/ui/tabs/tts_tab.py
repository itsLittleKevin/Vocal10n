"""TTS settings tab for Section B."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.constants import Language, ModelStatus, TTSSource
from vocal10n.state import SystemState
from vocal10n.tts.audio_output import list_output_devices
from vocal10n.ui.utils.combobox_styling import ArrowComboBox
from vocal10n.ui.widgets.param_slider import ParamSlider


_STATUS_COLORS = {
    ModelStatus.UNLOADED: ("#8892a4", "Disconnected"),
    ModelStatus.LOADING: ("#f0c030", "Connecting..."),
    ModelStatus.LOADED: ("#40c040", "Connected"),
    ModelStatus.UNLOADING: ("#f0c030", "Disconnecting..."),
    ModelStatus.ERROR: ("#e04040", "Error"),
}


class TTSTab(QWidget):
    """Settings tab for GPT-SoVITS TTS module."""

    start_server_requested = Signal()
    stop_server_requested = Signal()
    reference_changed = Signal(str, str, str)  # path, text, language
    output_device_changed = Signal(int)  # device index (-1 = default)

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── TTS Toggles ───────────────────────────────────────────────
        toggle_box = QGroupBox("TTS Output")
        tgl = QVBoxLayout(toggle_box)

        self._source_cb = QCheckBox("Speak Source Text (original language)")
        self._source_cb.setChecked(self._state.tts_source_enabled)
        self._source_cb.toggled.connect(self._on_source_toggled)
        tgl.addWidget(self._source_cb)

        self._target_cb = QCheckBox("Speak Target Text (translated)")
        self._target_cb.setChecked(self._state.tts_target_enabled)
        self._target_cb.toggled.connect(self._on_target_toggled)
        tgl.addWidget(self._target_cb)

        root.addWidget(toggle_box)

        # ── Server Status ─────────────────────────────────────────────
        srv_box = QGroupBox("GPT-SoVITS Server")
        srv_lay = QHBoxLayout(srv_box)

        self._status_label = QLabel("Status: Disconnected")
        srv_lay.addWidget(self._status_label)
        srv_lay.addStretch()

        self._start_btn = QPushButton("Start")
        self._start_btn.setFixedHeight(36)
        self._start_btn.clicked.connect(self._on_start_clicked)
        srv_lay.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setFixedHeight(36)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        srv_lay.addWidget(self._stop_btn)

        root.addWidget(srv_box)

        # Listen for state changes
        self._state.tts_status_changed.connect(self._on_status_changed)

        # ── Output Device ─────────────────────────────────────────────
        dev_box = QGroupBox("Audio Output")
        dev_lay = QHBoxLayout(dev_box)
        dev_lay.addWidget(QLabel("Device:"))
        self._device_combo = ArrowComboBox()
        self._populate_devices()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        dev_lay.addWidget(self._device_combo, stretch=1)
        refresh_btn = QPushButton("\u27f3")
        refresh_btn.setFixedSize(36, 36)
        refresh_btn.clicked.connect(self._populate_devices)
        dev_lay.addWidget(refresh_btn)
        root.addWidget(dev_box)

        # ── Reference Audio ───────────────────────────────────────────
        ref_box = QGroupBox("Reference Audio (Voice Clone)")
        ref_lay = QVBoxLayout(ref_box)

        # Audio file row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._ref_path_edit = QLineEdit()
        self._ref_path_edit.setReadOnly(True)
        self._ref_path_edit.setPlaceholderText("Select a reference .wav file")
        saved_path = self._cfg.get("tts.ref_audio_path", "")
        if saved_path:
            self._ref_path_edit.setText(saved_path)
        file_row.addWidget(self._ref_path_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedHeight(36)
        browse_btn.clicked.connect(self._browse_ref_audio)
        file_row.addWidget(browse_btn)
        ref_lay.addLayout(file_row)

        # Reference text row
        txt_row = QHBoxLayout()
        txt_row.addWidget(QLabel("Text:"))
        self._ref_text_edit = QLineEdit()
        self._ref_text_edit.setPlaceholderText("Transcript of reference audio")
        saved_text = self._cfg.get("tts.ref_audio_text", "")
        self._ref_text_edit.setText(saved_text)
        self._ref_text_edit.textChanged.connect(self._on_ref_changed)
        txt_row.addWidget(self._ref_text_edit, stretch=1)
        ref_lay.addLayout(txt_row)

        # Reference language row
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self._ref_lang_combo = ArrowComboBox()
        self._ref_lang_combo.addItems([lang.value for lang in Language])
        saved_lang = self._cfg.get("tts.ref_audio_lang", "English")
        idx = self._ref_lang_combo.findText(saved_lang)
        if idx >= 0:
            self._ref_lang_combo.setCurrentIndex(idx)
        self._ref_lang_combo.currentTextChanged.connect(self._on_ref_changed)
        lang_row.addWidget(self._ref_lang_combo, stretch=1)
        ref_lay.addLayout(lang_row)

        root.addWidget(ref_box)

        # ── TTS Parameters ────────────────────────────────────────────
        param_box = QGroupBox("Synthesis Parameters")
        pl = QVBoxLayout(param_box)

        self._speed_slider = ParamSlider(
            label="Speed",
            minimum=0.5,
            maximum=2.0,
            default=self._cfg.get("tts.speed_factor", 1.0),
            step=0.01,
            tooltip="Playback speed factor (1.0 = normal)",
        )
        pl.addWidget(self._speed_slider)

        self._top_k_slider = ParamSlider(
            label="Top-K",
            minimum=1,
            maximum=50,
            default=self._cfg.get("tts.top_k", 15),
            step=1,
            tooltip="Top-K sampling for token selection",
        )
        pl.addWidget(self._top_k_slider)

        self._top_p_slider = ParamSlider(
            label="Top-P",
            minimum=0.1,
            maximum=1.0,
            default=self._cfg.get("tts.top_p", 1.0),
            step=0.01,
            tooltip="Nucleus sampling probability",
        )
        pl.addWidget(self._top_p_slider)

        self._temp_slider = ParamSlider(
            label="Temperature",
            minimum=0.1,
            maximum=2.0,
            default=self._cfg.get("tts.temperature", 1.0),
            step=0.01,
            tooltip="Sampling temperature",
        )
        pl.addWidget(self._temp_slider)

        root.addWidget(param_box)
        root.addStretch()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_source_toggled(self, checked: bool) -> None:
        self._state.tts_source_enabled = checked
        self._cfg.set("tts.source_enabled", checked)

    def _on_target_toggled(self, checked: bool) -> None:
        self._state.tts_target_enabled = checked
        self._cfg.set("tts.target_enabled", checked)

    def _on_start_clicked(self) -> None:
        self.start_server_requested.emit()

    def _on_stop_clicked(self) -> None:
        self.stop_server_requested.emit()

    def _on_status_changed(self, status: ModelStatus) -> None:
        color, text = _STATUS_COLORS.get(status, ("#8892a4", "Unknown"))
        self._status_label.setText(f"Status: <span style='color:{color}'>{text}</span>")

        is_connected = status == ModelStatus.LOADED
        is_idle = status in (ModelStatus.UNLOADED, ModelStatus.ERROR)
        self._start_btn.setEnabled(is_idle)
        self._stop_btn.setEnabled(is_connected)

    def _browse_ref_audio(self) -> None:
        start_dir = str(Path.cwd() / "reference_audio")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Audio",
            start_dir,
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)",
        )
        if path:
            self._ref_path_edit.setText(path)
            self._cfg.set("tts.ref_audio_path", path)
            self._on_ref_changed()

    def _on_ref_changed(self) -> None:
        path = self._ref_path_edit.text()
        text = self._ref_text_edit.text()
        lang = self._ref_lang_combo.currentText()
        self._cfg.set("tts.ref_audio_text", text)
        self._cfg.set("tts.ref_audio_lang", lang)
        self.reference_changed.emit(path, text, lang)

    def _on_device_changed(self, index: int) -> None:
        data = self._device_combo.currentData()
        device_idx = data if data is not None else -1
        self.output_device_changed.emit(device_idx)

    def _populate_devices(self) -> None:
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._device_combo.addItem("(Default)", None)
        for dev in list_output_devices():
            self._device_combo.addItem(dev["name"], dev["index"])
        self._device_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selected_device_index(self) -> int | None:
        return self._device_combo.currentData()

    def get_tts_params(self) -> dict:
        """Return current TTS parameter values."""
        return {
            "speed_factor": self._speed_slider.value(),
            "top_k": int(self._top_k_slider.value()),
            "top_p": self._top_p_slider.value(),
            "temperature": self._temp_slider.value(),
        }
