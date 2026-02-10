"""Section B — bottom portion of the main window.

A QTabWidget that will host settings tabs for each pipeline component.
Placeholder widgets are replaced as each phase is implemented.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QScrollArea, QTabWidget, QVBoxLayout, QWidget

from vocal10n.state import SystemState
from vocal10n.ui.tabs.kb_tab import KnowledgeBaseTab
from vocal10n.ui.tabs.obs_tab import OBSTab
from vocal10n.ui.tabs.output_tab import OutputTab
from vocal10n.ui.tabs.stt_tab import STTTab
from vocal10n.ui.tabs.training_tab import TrainingTab
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

        # Phase 6: real Output tab
        self.output_tab = OutputTab(state)
        self.addTab(_scrollable(self.output_tab), "Output")

        # Phase 7: real OBS tab
        self.obs_tab = OBSTab(state)
        self.addTab(_scrollable(self.obs_tab), "OBS")

        # Knowledge Base tab (glossary management + RAG)
        self.kb_tab = KnowledgeBaseTab(state)
        self.addTab(_scrollable(self.kb_tab), "Knowledge Base")

        # Phase 8: training tab
        self.training_tab = TrainingTab(state)
        self.addTab(_scrollable(self.training_tab), "Training")
