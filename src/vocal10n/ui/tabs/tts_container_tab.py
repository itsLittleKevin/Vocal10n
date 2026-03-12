"""TTS container tab — holds sub-tabs for GPT-SoVITS and Qwen3-TTS.

Provides shared TTS output toggles at the top and a QTabWidget with
engine-specific sub-tabs below.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.state import SystemState
from vocal10n.ui.tabs.qwen3_tts_tab import Qwen3TTSTab
from vocal10n.ui.tabs.tts_tab import TTSTab


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

        self._source_cb = QCheckBox("Speak Source Text (original language)")
        self._source_cb.setChecked(self._state.tts_source_enabled)
        self._source_cb.toggled.connect(self._on_source_toggled)
        tgl.addWidget(self._source_cb)

        self._target_cb = QCheckBox("Speak Target Text (translated)")
        self._target_cb.setChecked(self._state.tts_target_enabled)
        self._target_cb.toggled.connect(self._on_target_toggled)
        tgl.addWidget(self._target_cb)

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
