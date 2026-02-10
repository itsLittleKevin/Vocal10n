"""Section B — bottom portion of the main window.

A QTabWidget that will host settings tabs for each pipeline component.
Placeholder widgets are replaced as each phase is implemented.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from vocal10n.state import SystemState
from vocal10n.ui.tabs.stt_tab import STTTab
from vocal10n.ui.tabs.translation_tab import TranslationTab
from vocal10n.ui.tabs.tts_tab import TTSTab


def _placeholder(text: str) -> QWidget:
    """Return a centred-label placeholder tab body."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #8892a4; font-size: 14px;")
    lay.addWidget(lbl)
    lay.addStretch()
    return w


def _scrollable(widget: QWidget) -> QScrollArea:
    """Wrap *widget* in a QScrollArea with vertical scrollbar."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setWidget(widget)
    return area


class SectionB(QTabWidget):
    """Bottom tab container.  Placeholder tabs are swapped out as each
    phase implements real content."""

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)

        # Phase 3: real STT tab
        self.stt_tab = STTTab(state)
        self.addTab(_scrollable(self.stt_tab), "STT")

        # Phase 4: real Translation tab
        self.translation_tab = TranslationTab(state)
        self.addTab(_scrollable(self.translation_tab), "Translation")

        # Phase 5: real TTS tab
        self.tts_tab = TTSTab(state)
        self.addTab(_scrollable(self.tts_tab), "TTS")

        self.addTab(_placeholder("Output settings will appear here (Phase 6)"), "Output")
        self.addTab(_placeholder("OBS overlay settings will appear here (Phase 7)"), "OBS")
        self.addTab(
            _placeholder("Training tools will appear here (Phase 8)"), "Training"
        )
