"""Section B — bottom portion of the main window.

A QTabWidget that will host settings tabs for each pipeline component.
Phase 2 creates placeholder widgets; real tabs arrive in later phases.
"""

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget


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

    def __init__(self, parent=None):
        super().__init__(parent)

        # Placeholders — will be replaced in Phases 3-8
        self.addTab(_placeholder("STT settings will appear here (Phase 3)"), "STT")
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
