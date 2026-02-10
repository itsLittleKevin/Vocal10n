"""Translation settings tab for Section B."""

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
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

        # ── Model selector ────────────────────────────────────────────
        models = ["Qwen3-4B-Q4_K_M"]
        self._model_sel = ModelSelector(label="LLM Model", items=models)
        root.addWidget(self._model_sel)

        # Wire model selector status to state
        self._state.llm_status_changed.connect(self._model_sel.set_status)

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
