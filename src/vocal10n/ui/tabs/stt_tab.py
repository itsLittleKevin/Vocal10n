"""STT settings tab for Section B."""

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.constants import Language, ModelStatus
from vocal10n.state import SystemState
from vocal10n.ui.utils.combobox_styling import ArrowComboBox
from vocal10n.ui.widgets.model_selector import ModelSelector
from vocal10n.ui.widgets.param_slider import ParamSlider


class STTTab(QWidget):
    """Settings tab for the Speech-to-Text module."""

    term_files_changed = Signal(list)  # emits list[str] of paths

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Enable toggle ─────────────────────────────────────────────
        self._enable_cb = QCheckBox("Enable STT (Speech-to-Text)")
        self._enable_cb.setChecked(self._state.stt_enabled)
        self._enable_cb.toggled.connect(self._on_enable)
        root.addWidget(self._enable_cb)

        # ── Model selector ────────────────────────────────────────────
        models = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
        self._model_sel = ModelSelector(label="Whisper Model", items=models)
        # Pre-select current config value
        current = self._cfg.get("stt.model_size", "large-v3-turbo")
        idx = self._model_sel._combo.findText(current)
        if idx >= 0:
            self._model_sel._combo.setCurrentIndex(idx)
        root.addWidget(self._model_sel)

        # Wire model selector status to state
        self._state.stt_status_changed.connect(self._model_sel.set_status)

        # ── Language ──────────────────────────────────────────────────
        lang_box = QGroupBox("Language")
        lb_lay = QHBoxLayout(lang_box)
        self._lang_combo = ArrowComboBox()
        for lang in Language:
            self._lang_combo.addItem(lang.display_name, lang.value)
        # Set current
        cur_lang = self._cfg.get("stt.language") or "auto"
        li = self._lang_combo.findData(cur_lang)
        if li >= 0:
            self._lang_combo.setCurrentIndex(li)
        self._lang_combo.currentIndexChanged.connect(self._on_language)
        lb_lay.addWidget(QLabel("Source language:"))
        lb_lay.addWidget(self._lang_combo, stretch=1)
        root.addWidget(lang_box)

        # ── Latency tuning ────────────────────────────────────────────
        tune_box = QGroupBox("Latency Tuning")
        tl = QVBoxLayout(tune_box)
        tl.setSpacing(6)

        self._window_slider = ParamSlider(
            "Window Size",
            minimum=2.0, maximum=15.0,
            default=self._cfg.get("stt.window_seconds", 6.5),
            step=0.5, suffix=" s",
            tooltip="Sliding window of audio sent to Whisper. "
                    "Smaller = lower latency, Larger = better context.",
        )
        self._window_slider.value_changed.connect(lambda v: self._cfg.set("stt.window_seconds", v))
        tl.addWidget(self._window_slider)

        self._confirm_slider = ParamSlider(
            "Confirm Threshold",
            minimum=0.1, maximum=2.0,
            default=self._cfg.get("stt.confirm_threshold", 0.3),
            step=0.05, suffix=" s",
            tooltip="How long a segment must be stable before confirming. "
                    "Lower = faster but riskier, Higher = more stable.",
        )
        self._confirm_slider.value_changed.connect(lambda v: self._cfg.set("stt.confirm_threshold", v))
        tl.addWidget(self._confirm_slider)

        self._age_slider = ParamSlider(
            "Max Segment Age",
            minimum=1.0, maximum=5.0,
            default=self._cfg.get("stt.max_segment_age", 2.0),
            step=0.5, suffix=" s",
            tooltip="Force-confirm segments older than this. "
                    "Lower = faster but keeps Whisper mistakes.",
        )
        self._age_slider.value_changed.connect(lambda v: self._cfg.set("stt.max_segment_age", v))
        tl.addWidget(self._age_slider)

        self._beam_slider = ParamSlider(
            "Beam Size",
            minimum=1, maximum=5,
            default=self._cfg.get("stt.beam_size", 1),
            step=1,
            tooltip="1 = greedy (fastest), higher = more accurate but slower.",
        )
        self._beam_slider.value_changed.connect(lambda v: self._cfg.set("stt.beam_size", int(v)))
        tl.addWidget(self._beam_slider)

        root.addWidget(tune_box)

        # ── Recognition context note ─────────────────────────────────
        ctx_note = QLabel(
            "<b>Recognition Context:</b> Term files for phonetic correction "
            "and Whisper initial_prompt are managed in the "
            "<b>Knowledge Base</b> tab."
        )
        ctx_note.setWordWrap(True)
        ctx_note.setStyleSheet(
            "color: #8892a4; font-size: 12px; padding: 8px; "
            "border: 1px solid #3a3f4b; border-radius: 4px; margin-top: 4px;"
        )
        root.addWidget(ctx_note)

        # ── Info ──────────────────────────────────────────────────────
        info = QLabel(
            "Settings apply immediately — no model reload needed.\n"
            "Presets:  Low latency (Win=5, Thr=0.5, Age=1.5, Beam=1)  |  "
            "Quality (Win=10, Thr=1.0, Age=3.5, Beam=3)"
        )
        info.setProperty("dim", True)
        info.setWordWrap(True)
        root.addWidget(info)

        root.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_enable(self, checked: bool) -> None:
        self._state.stt_enabled = checked
        self._cfg.set("pipeline.enable_stt", checked)

    @Slot(int)
    def _on_language(self, _idx: int) -> None:
        code = self._lang_combo.currentData()
        lang_val = None if code == "auto" else code
        self._cfg.set("stt.language", lang_val)
        try:
            self._state.source_language = Language(code)
        except ValueError:
            pass
