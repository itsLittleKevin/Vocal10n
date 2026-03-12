"""TTS container tab — holds sub-tabs for GPT-SoVITS and Qwen3-TTS.

Provides shared TTS output toggles at the top and a QTabWidget with
engine-specific sub-tabs below.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.constants import Language
from vocal10n.state import SystemState
from vocal10n.ui.tabs.qwen3_tts_tab import Qwen3TTSTab
from vocal10n.ui.tabs.tts_tab import TTSTab


# Map display names to Language enum
_LANG_FROM_DISPLAY = {
    "English": Language.ENGLISH,
    "Chinese": Language.CHINESE,
    "Auto-detect": Language.AUTO,
    "Auto": Language.AUTO,
}


class TTSContainerTab(QWidget):
    """Top-level TTS tab containing shared controls and engine sub-tabs."""

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Shared TTS Toggles ────────────────────────────────────────
        toggle_box = QGroupBox("TTS Output")
        tgl = QVBoxLayout(toggle_box)

        # Source row with checkbox and language status
        source_row = QHBoxLayout()
        self._source_cb = QCheckBox("Speak Source Text (original language)")
        self._source_cb.setChecked(self._state.tts_source_enabled)
        self._source_cb.toggled.connect(self._on_source_toggled)
        source_row.addWidget(self._source_cb)
        source_row.addStretch()
        self._source_lang_label = QLabel()
        self._source_lang_label.setProperty("dim", True)
        source_row.addWidget(self._source_lang_label)
        tgl.addLayout(source_row)

        # Target row with checkbox and language status
        target_row = QHBoxLayout()
        self._target_cb = QCheckBox("Speak Target Text (translated)")
        self._target_cb.setChecked(self._state.tts_target_enabled)
        self._target_cb.toggled.connect(self._on_target_toggled)
        target_row.addWidget(self._target_cb)
        target_row.addStretch()
        self._target_lang_label = QLabel()
        self._target_lang_label.setProperty("dim", True)
        target_row.addWidget(self._target_lang_label)
        tgl.addLayout(target_row)

        # Initialize language labels
        self._update_language_labels()

        # Listen for language changes
        self._state.source_language_changed.connect(self._update_language_labels)
        self._state.target_language_changed.connect(self._update_language_labels)

        root.addWidget(toggle_box)

        # ── Engine Sub-tabs ───────────────────────────────────────────
        self._sub_tabs = QTabWidget()

        # GPT-SoVITS sub-tab
        self.gptsovits_tab = TTSTab(state)
        gptsovits_scroll = QScrollArea()
        gptsovits_scroll.setWidgetResizable(True)
        gptsovits_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        gptsovits_scroll.setFrameShape(QScrollArea.NoFrame)
        gptsovits_scroll.setWidget(self.gptsovits_tab)
        self._sub_tabs.addTab(gptsovits_scroll, "GPT-SoVITS")

        # Qwen3-TTS sub-tab
        self.qwen3_tab = Qwen3TTSTab(state)
        qwen3_scroll = QScrollArea()
        qwen3_scroll.setWidgetResizable(True)
        qwen3_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        qwen3_scroll.setFrameShape(QScrollArea.NoFrame)
        qwen3_scroll.setWidget(self.qwen3_tab)
        self._sub_tabs.addTab(qwen3_scroll, "Qwen3-TTS")

        root.addWidget(self._sub_tabs, stretch=1)

    # ------------------------------------------------------------------
    # Shared toggle handlers
    # ------------------------------------------------------------------

    def _on_source_toggled(self, checked: bool) -> None:
        self._state.tts_source_enabled = checked
        self._cfg.set("tts.source_enabled", checked)

    def _on_target_toggled(self, checked: bool) -> None:
        self._state.tts_target_enabled = checked
        self._cfg.set("tts.target_enabled", checked)

    def _update_language_labels(self) -> None:
        """Update the language status labels based on current state."""
        source_lang = self._state.source_language
        target_lang = self._state.target_language

        # Get display names
        source_name = source_lang.display_name
        target_name = target_lang.display_name

        self._source_lang_label.setText(f"Speaking: {source_name}")
        self._target_lang_label.setText(f"Speaking: {target_name}")

    def set_target_language_from_string(self, lang_name: str) -> None:
        """Set target language from a display name string (e.g., 'English', 'Chinese')."""
        lang = _LANG_FROM_DISPLAY.get(lang_name)
        if lang:
            self._state.target_language = lang
