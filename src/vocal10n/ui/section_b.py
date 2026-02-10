"""Section B — bottom portion of the main window.

A QTabWidget that will host settings tabs for each pipeline component.
Placeholder widgets are replaced as each phase is implemented.
"""

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from vocal10n.state import SystemState
from vocal10n.ui.tabs.stt_tab import STTTab


def _placeholder(text: str) -> QWidget:
    """Return a centred-label placeholder tab body."""
    w = QWidget()
    lay = QVBoxLayout(w)
    lbl = QLabel(text)
    lbl.setStyleSheet("color: #8892a4; font-size: 14px;")
    lay.addWidget(lbl)
    lay.addStretch()
    return w


class SectionB(QTabWidget):
    """Bottom tab container.  Placeholder tabs are swapped out as each
    phase implements real content."""

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)

        # Phase 3: real STT tab
        self.stt_tab = STTTab(state)
        self.addTab(self.stt_tab, "STT")

        # Placeholders — will be replaced in Phases 4-8
        self.addTab(
            _placeholder("Translation settings will appear here (Phase 4)"),
            "Translation",
        )
        self.addTab(_placeholder("TTS settings will appear here (Phase 5)"), "TTS")
        self.addTab(_placeholder("Output settings will appear here (Phase 6)"), "Output")
        self.addTab(_placeholder("OBS overlay settings will appear here (Phase 7)"), "OBS")
        self.addTab(
            _placeholder("Training tools will appear here (Phase 8)"), "Training"
        )
