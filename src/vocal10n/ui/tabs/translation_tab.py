"""Translation settings tab for Section B."""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.constants import Language, ModelStatus
from vocal10n.state import SystemState
from vocal10n.ui.widgets.model_selector import ModelSelector
from vocal10n.ui.widgets.param_slider import ParamSlider


class TranslationTab(QWidget):
    """Settings tab for the LLM translation module."""

    target_language_changed = Signal(str)  # emits language name, e.g. "English"
    load_requested = Signal(str)
    unload_requested = Signal()

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Enable toggle ─────────────────────────────────────────────
        self._enable_cb = QCheckBox("Enable Translation (LLM)")
        self._enable_cb.setChecked(self._state.llm_enabled)
        self._enable_cb.toggled.connect(self._on_enable)
        root.addWidget(self._enable_cb)

        # ── Backend selector ──────────────────────────────────────────
        backend_box = QGroupBox("Backend")
        bb_lay = QHBoxLayout(backend_box)
        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["Local GGUF", "OpenAI API"])
        cur_backend = self._cfg.get("translation.backend", "local")
        if cur_backend == "api":
            self._backend_combo.setCurrentIndex(1)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        bb_lay.addWidget(QLabel("Backend:"))
        bb_lay.addWidget(self._backend_combo, stretch=1)
        root.addWidget(backend_box)

        # ── Model selector (local mode) ─────────────────────────────
        models = ["Qwen3-4B-Q4_K_M"]
        self._model_sel = ModelSelector(label="LLM Model", items=models)
        root.addWidget(self._model_sel)

        # Wire model selector status to state
        self._state.llm_status_changed.connect(self._model_sel.set_status)

        # Forward model selector signals to tab-level signals
        self._model_sel.load_requested.connect(self.load_requested)
        self._model_sel.unload_requested.connect(self.unload_requested)

        # ── API settings (API mode) ─────────────────────────────────
        self._api_group = QGroupBox("API Connection")
        al = QVBoxLayout(self._api_group)

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self._api_url_edit = QLineEdit()
        self._api_url_edit.setPlaceholderText("http://localhost:1234/v1")
        self._api_url_edit.setText(
            self._cfg.get("translation.api_url", "http://localhost:1234/v1"),
        )
        url_row.addWidget(self._api_url_edit, stretch=1)
        al.addLayout(url_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._api_model_edit = QLineEdit()
        self._api_model_edit.setPlaceholderText("(auto-detect from server)")
        self._api_model_edit.setText(
            self._cfg.get("translation.api_model", ""),
        )
        model_row.addWidget(self._api_model_edit, stretch=1)
        al.addLayout(model_row)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setPlaceholderText("(optional, for cloud APIs)")
        self._api_key_edit.setEchoMode(QLineEdit.Password)
        self._api_key_edit.setText(
            self._cfg.get("translation.api_key", ""),
        )
        key_row.addWidget(self._api_key_edit, stretch=1)
        al.addLayout(key_row)

        btn_row = QHBoxLayout()
        self._api_status = QLabel("● Disconnected")
        self._api_status.setStyleSheet("color: #8892a4;")
        btn_row.addWidget(self._api_status)
        btn_row.addStretch()
        self._api_connect_btn = QPushButton("Connect")
        self._api_connect_btn.setProperty("accent", True)
        self._api_connect_btn.setFixedWidth(80)
        self._api_connect_btn.clicked.connect(self._on_api_connect)
        btn_row.addWidget(self._api_connect_btn)
        self._api_disconnect_btn = QPushButton("Disconnect")
        self._api_disconnect_btn.setFixedWidth(80)
        self._api_disconnect_btn.setEnabled(False)
        self._api_disconnect_btn.clicked.connect(self._on_api_disconnect)
        btn_row.addWidget(self._api_disconnect_btn)
        al.addLayout(btn_row)

        root.addWidget(self._api_group)
        self._api_group.hide()  # local mode by default

        # Wire API status feedback
        self._state.llm_status_changed.connect(self._on_api_status_changed)

        # Apply initial backend visibility
        if cur_backend == "api":
            self._model_sel.hide()
            self._api_group.show()

        # ── Target language ───────────────────────────────────────────
        lang_box = QGroupBox("Target Language")
        lb_lay = QHBoxLayout(lang_box)
        self._target_combo = QComboBox()
        self._target_combo.addItems(["English", "Chinese"])
        cur = self._cfg.get("translation.target_language", "English")
        idx = self._target_combo.findText(cur)
        if idx >= 0:
            self._target_combo.setCurrentIndex(idx)
        self._target_combo.currentTextChanged.connect(self._on_target_lang)
        lb_lay.addWidget(QLabel("Translate to:"))
        lb_lay.addWidget(self._target_combo, stretch=1)
        root.addWidget(lang_box)

        # ── Generation parameters ─────────────────────────────────────
        gen_box = QGroupBox("Generation Parameters")
        gl = QVBoxLayout(gen_box)
        gl.setSpacing(6)

        self._ctx_slider = ParamSlider(
            "Context Length",
            minimum=64, maximum=512,
            default=self._cfg.get("translation.n_ctx", 128),
            step=32,
            tooltip="LLM context window size. 128 is enough for translation.\n"
                    "Larger uses more VRAM.",
        )
        self._ctx_slider.value_changed.connect(
            lambda v: self._cfg.set("translation.n_ctx", int(v))
        )
        gl.addWidget(self._ctx_slider)

        self._tokens_slider = ParamSlider(
            "Max Tokens",
            minimum=16, maximum=256,
            default=self._cfg.get("translation.max_tokens", 64),
            step=8,
            tooltip="Maximum tokens in translation output.\n"
                    "64 is usually sufficient for single sentences.",
        )
        self._tokens_slider.value_changed.connect(
            lambda v: self._cfg.set("translation.max_tokens", int(v))
        )
        gl.addWidget(self._tokens_slider)

        self._temp_slider = ParamSlider(
            "Temperature",
            minimum=0.0, maximum=1.0,
            default=self._cfg.get("translation.temperature", 0.0),
            step=0.05,
            tooltip="0.0 = deterministic (fastest, best for translation).\n"
                    "Higher = more creative output.",
        )
        self._temp_slider.value_changed.connect(
            lambda v: self._cfg.set("translation.temperature", v)
        )
        gl.addWidget(self._temp_slider)

        root.addWidget(gen_box)

        # ── Prompt template ───────────────────────────────────────────
        prompt_box = QGroupBox("Prompt Template")
        pl = QVBoxLayout(prompt_box)
        pl.setSpacing(4)

        prompt_info = QLabel(
            "Variables: {target_lang}, {text}, {source_lang}\n"
            "The model also receives stop tokens to prevent over-generation."
        )
        prompt_info.setProperty("dim", True)
        prompt_info.setWordWrap(True)
        pl.addWidget(prompt_info)

        self._prompt_edit = QPlainTextEdit()
        self._prompt_edit.setMaximumHeight(90)
        default_tmpl = self._cfg.get(
            "translation.prompt_template",
            "Fix any speech recognition errors and translate to {target_lang}:\n{text}",
        )
        self._prompt_edit.setPlainText(default_tmpl)
        self._prompt_edit.textChanged.connect(self._on_prompt_changed)
        pl.addWidget(self._prompt_edit)

        root.addWidget(prompt_box)

        # ── Batching tuning ───────────────────────────────────────────
        batch_box = QGroupBox("Batching / Latency")
        bl = QVBoxLayout(batch_box)
        bl.setSpacing(6)

        self._debounce_slider = ParamSlider(
            "Pending Debounce",
            minimum=50, maximum=500,
            default=self._cfg.get("pipeline.translation_debounce_ms", 150),
            step=25, suffix=" ms",
            tooltip="How long to wait for pending text to settle before translating.\n"
                    "Lower = more responsive, Higher = fewer LLM calls.",
        )
        self._debounce_slider.value_changed.connect(
            lambda v: self._cfg.set("pipeline.translation_debounce_ms", int(v))
        )
        bl.addWidget(self._debounce_slider)

        self._batch_slider = ParamSlider(
            "Confirmed Batch Delay",
            minimum=200, maximum=2000,
            default=self._cfg.get("pipeline.confirmed_batch_delay_ms", 800),
            step=100, suffix=" ms",
            tooltip="How long to accumulate confirmed fragments before batching.\n"
                    "Sentence/clause punctuation triggers immediate translation.",
        )
        self._batch_slider.value_changed.connect(
            lambda v: self._cfg.set("pipeline.confirmed_batch_delay_ms", int(v))
        )
        bl.addWidget(self._batch_slider)

        root.addWidget(batch_box)

        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_enable(self, checked: bool) -> None:
        self._state.llm_enabled = checked
        self._cfg.set("pipeline.enable_translation", checked)

    @Slot(str)
    def _on_target_lang(self, lang: str) -> None:
        self._cfg.set("translation.target_language", lang)
        self.target_language_changed.emit(lang)

    @Slot()
    def _on_prompt_changed(self) -> None:
        self._cfg.set(
            "translation.prompt_template",
            self._prompt_edit.toPlainText(),
        )

    @Slot(int)
    def _on_backend_changed(self, index: int) -> None:
        backend = "api" if index == 1 else "local"
        self._cfg.set("translation.backend", backend)
        is_api = index == 1
        self._model_sel.setVisible(not is_api)
        self._api_group.setVisible(is_api)
        self._ctx_slider.setVisible(not is_api)

    @Slot()
    def _on_api_connect(self) -> None:
        self._cfg.set("translation.api_url", self._api_url_edit.text())
        self._cfg.set("translation.api_model", self._api_model_edit.text())
        self._cfg.set("translation.api_key", self._api_key_edit.text())
        self.load_requested.emit("api")

    @Slot()
    def _on_api_disconnect(self) -> None:
        self.unload_requested.emit()

    @Slot(ModelStatus)
    def _on_api_status_changed(self, status: ModelStatus) -> None:
        if self._backend_combo.currentIndex() != 1:
            return
        _colours = {
            ModelStatus.UNLOADED: ("#8892a4", "Disconnected"),
            ModelStatus.LOADING: ("#f0c030", "Connecting..."),
            ModelStatus.LOADED: ("#0f9b8e", "Connected"),
            ModelStatus.UNLOADING: ("#f0c030", "Disconnecting..."),
            ModelStatus.ERROR: ("#e94560", "Error"),
        }
        colour, label = _colours.get(status, ("#8892a4", "Unknown"))
        self._api_status.setStyleSheet(f"color: {colour};")
        self._api_status.setText(f"● {label}")
        is_idle = status in (ModelStatus.UNLOADED, ModelStatus.ERROR)
        is_loaded = status == ModelStatus.LOADED
        self._api_connect_btn.setEnabled(is_idle)
        self._api_disconnect_btn.setEnabled(is_loaded)
        self._api_url_edit.setEnabled(is_idle)
        self._api_model_edit.setEnabled(is_idle)
        self._api_key_edit.setEnabled(is_idle)
