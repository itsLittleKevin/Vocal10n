"""Qwen3-TTS settings sub-tab."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
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
from vocal10n.constants import Language, ModelStatus
from vocal10n.state import SystemState
from vocal10n.ui.utils.combobox_styling import ArrowComboBox
from vocal10n.ui.widgets.param_slider import ParamSlider


_STATUS_COLORS = {
    ModelStatus.UNLOADED: ("#8892a4", "Unloaded"),
    ModelStatus.LOADING: ("#f0c030", "Loading..."),
    ModelStatus.LOADED: ("#40c040", "Ready"),
    ModelStatus.UNLOADING: ("#f0c030", "Unloading..."),
    ModelStatus.ERROR: ("#e04040", "Error"),
}

# Available Qwen3-TTS models
_MODELS = [
    ("0.6B Base (voice clone, ~1.5GB VRAM)", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
    ("0.6B CustomVoice (built-in voices)", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
    ("1.7B Base (voice clone, ~3.5GB VRAM)", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
    ("1.7B CustomVoice (built-in voices)", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
    ("1.7B VoiceDesign (describe voice)", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
]


class Qwen3TTSTab(QWidget):
    """Settings sub-tab for Qwen3-TTS."""

    load_requested = Signal()
    unload_requested = Signal()
    reference_changed = Signal(str, str, str)  # path, text, language
    output_device_changed = Signal(int)

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Model Selection ───────────────────────────────────────────
        model_box = QGroupBox("Model")
        ml = QVBoxLayout(model_box)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = ArrowComboBox()
        saved_model = self._cfg.get("tts_qwen3.model_name", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        for display_name, model_id in _MODELS:
            self._model_combo.addItem(display_name, model_id)
        # Select saved model
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == saved_model:
                self._model_combo.setCurrentIndex(i)
                break
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo, stretch=1)
        ml.addLayout(model_row)

        # Local path override
        local_row = QHBoxLayout()
        local_row.addWidget(QLabel("Local path:"))
        self._local_path_edit = QLineEdit()
        self._local_path_edit.setPlaceholderText("(optional) Local model directory")
        self._local_path_edit.setText(self._cfg.get("tts_qwen3.local_model_path", ""))
        self._local_path_edit.textChanged.connect(
            lambda t: self._cfg.set("tts_qwen3.local_model_path", t)
        )
        local_row.addWidget(self._local_path_edit, stretch=1)
        ml.addLayout(local_row)

        root.addWidget(model_box)

        # ── Status + Load/Unload ──────────────────────────────────────
        status_box = QGroupBox("Qwen3-TTS Engine")
        sl = QHBoxLayout(status_box)

        self._status_label = QLabel("Status: Unloaded")
        sl.addWidget(self._status_label)
        sl.addStretch()

        self._load_btn = QPushButton("Load")
        self._load_btn.setFixedHeight(36)
        self._load_btn.clicked.connect(self._on_load_clicked)
        sl.addWidget(self._load_btn)

        self._unload_btn = QPushButton("Unload")
        self._unload_btn.setFixedHeight(36)
        self._unload_btn.setEnabled(False)
        self._unload_btn.clicked.connect(self._on_unload_clicked)
        sl.addWidget(self._unload_btn)

        root.addWidget(status_box)

        # Listen for state changes
        self._state.tts_status_changed.connect(self._on_status_changed)

        # ── Output Device ─────────────────────────────────────────────
        from vocal10n.tts.audio_output import list_output_devices

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

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self._ref_path_edit = QLineEdit()
        self._ref_path_edit.setReadOnly(True)
        self._ref_path_edit.setPlaceholderText("Select a reference .wav file (3s+ recommended)")
        saved_path = self._cfg.get("tts_qwen3.ref_audio_path", "")
        if saved_path:
            self._ref_path_edit.setText(saved_path)
        file_row.addWidget(self._ref_path_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedHeight(36)
        browse_btn.clicked.connect(self._browse_ref_audio)
        file_row.addWidget(browse_btn)
        ref_lay.addLayout(file_row)

        txt_row = QHBoxLayout()
        txt_row.addWidget(QLabel("Text:"))
        self._ref_text_edit = QLineEdit()
        self._ref_text_edit.setPlaceholderText("Transcript of reference audio")
        saved_text = self._cfg.get("tts_qwen3.ref_audio_text", "")
        self._ref_text_edit.setText(saved_text)
        self._ref_text_edit.textChanged.connect(self._on_ref_changed)
        txt_row.addWidget(self._ref_text_edit, stretch=1)
        ref_lay.addLayout(txt_row)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:"))
        self._ref_lang_combo = ArrowComboBox()
        self._ref_lang_combo.addItems([lang.value for lang in Language])
        saved_lang = self._cfg.get("tts_qwen3.ref_audio_lang", "English")
        idx = self._ref_lang_combo.findText(saved_lang)
        if idx >= 0:
            self._ref_lang_combo.setCurrentIndex(idx)
        self._ref_lang_combo.currentTextChanged.connect(self._on_ref_changed)
        lang_row.addWidget(self._ref_lang_combo, stretch=1)
        ref_lay.addLayout(lang_row)

        root.addWidget(ref_box)

        # ── Synthesis Parameters ──────────────────────────────────────
        param_box = QGroupBox("Synthesis Parameters")
        pl = QVBoxLayout(param_box)

        self._top_k_slider = ParamSlider(
            label="Top-K",
            minimum=1,
            maximum=100,
            default=self._cfg.get("tts_qwen3.top_k", 50),
            step=1,
            tooltip="Top-K sampling for token selection",
        )
        pl.addWidget(self._top_k_slider)

        self._top_p_slider = ParamSlider(
            label="Top-P",
            minimum=0.1,
            maximum=1.0,
            default=self._cfg.get("tts_qwen3.top_p", 0.9),
            step=0.01,
            tooltip="Nucleus sampling probability",
        )
        pl.addWidget(self._top_p_slider)

        self._temp_slider = ParamSlider(
            label="Temperature",
            minimum=0.1,
            maximum=2.0,
            default=self._cfg.get("tts_qwen3.temperature", 1.0),
            step=0.01,
            tooltip="Sampling temperature (higher = more expressive)",
        )
        pl.addWidget(self._temp_slider)

        root.addWidget(param_box)

        # ── Flash Attention toggle ────────────────────────────────────
        adv_box = QGroupBox("Advanced")
        al = QVBoxLayout(adv_box)
        self._flash_attn_cb = QCheckBox("Use FlashAttention 2 (reduces VRAM)")
        self._flash_attn_cb.setChecked(self._cfg.get("tts_qwen3.use_flash_attn", True))
        self._flash_attn_cb.toggled.connect(
            lambda v: self._cfg.set("tts_qwen3.use_flash_attn", v)
        )
        al.addWidget(self._flash_attn_cb)
        root.addWidget(adv_box)

        root.addStretch()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_model_changed(self, index: int) -> None:
        model_id = self._model_combo.currentData()
        if model_id:
            self._cfg.set("tts_qwen3.model_name", model_id)

    def _on_load_clicked(self) -> None:
        self.load_requested.emit()

    def _on_unload_clicked(self) -> None:
        self.unload_requested.emit()

    def _on_status_changed(self, status: ModelStatus) -> None:
        color, text = _STATUS_COLORS.get(status, ("#8892a4", "Unknown"))
        
        # Show error message if available
        if status == ModelStatus.ERROR:
            # Try to get error from controller (this is a bit hacky, but works)
            try:
                from vocal10n.ui.main_window import MainWindow
                main_win = self.window()
                if hasattr(main_win, '_qwen3_tts_ctrl'):
                    error_msg = main_win._qwen3_tts_ctrl.get_error_message()
                    if error_msg:
                        text = f"Error: {error_msg[:50]}..."  # Truncate long errors
            except Exception:
                pass
        
        self._status_label.setText(f"Status: <span style='color:{color}'>{text}</span>")
        is_loaded = status == ModelStatus.LOADED
        is_idle = status in (ModelStatus.UNLOADED, ModelStatus.ERROR)
        self._load_btn.setEnabled(is_idle)
        self._unload_btn.setEnabled(is_loaded)

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
            self._cfg.set("tts_qwen3.ref_audio_path", path)
            self._on_ref_changed()

    def _on_ref_changed(self) -> None:
        path = self._ref_path_edit.text()
        text = self._ref_text_edit.text()
        lang = self._ref_lang_combo.currentText()
        self._cfg.set("tts_qwen3.ref_audio_text", text)
        self._cfg.set("tts_qwen3.ref_audio_lang", lang)
        self.reference_changed.emit(path, text, lang)

    def _on_device_changed(self, index: int) -> None:
        data = self._device_combo.currentData()
        device_idx = data if data is not None else -1
        self.output_device_changed.emit(device_idx)

    def _populate_devices(self) -> None:
        from PySide6.QtCore import Qt
        from vocal10n.tts.audio_output import list_output_devices

        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._device_combo.addItem("(Default)", None)
        for dev in list_output_devices():
            self._device_combo.addItem(dev["name"], dev["index"])
            idx = self._device_combo.count() - 1
            self._device_combo.setItemData(idx, dev["name"], Qt.ToolTipRole)
        self._device_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tts_params(self) -> dict:
        return {
            "top_k": int(self._top_k_slider.value()),
            "top_p": self._top_p_slider.value(),
            "temperature": self._temp_slider.value(),
        }
