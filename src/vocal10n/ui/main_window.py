"""Main application window — vertical A / B split layout."""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from vocal10n.state import SystemState
from vocal10n.ui.section_a import SectionA
from vocal10n.ui.section_b import SectionB
from vocal10n.utils.gpu import get_gpu_monitor


class MainWindow(QMainWindow):
    """Vocal10n main window.

    Layout::

        ┌───────────────────────────────┐
        │  Section A  (streams + stats) │  ≈ 55 %
        ├───────────────────────────────┤
        │  Section B  (settings tabs)   │  ≈ 45 %
        └───────────────────────────────┘
        │           status bar          │
    """

    def __init__(self, state: SystemState, parent=None):
        super().__init__(parent)
        self._state = state

        self.setWindowTitle("Vocal10n — Real-time Speech Translation")
        self.resize(1280, 800)

        # ── Central widget with splitter ──────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        self.section_a = SectionA()
        self.section_b = SectionB()

        splitter.addWidget(self.section_a)
        splitter.addWidget(self.section_b)
        splitter.setStretchFactor(0, 55)
        splitter.setStretchFactor(1, 45)

        self.setCentralWidget(splitter)

        # ── Status bar ────────────────────────────────────────────────
        sb = QStatusBar()
        self._sb_gpu = QLabel("GPU: —")
        self._sb_vram = QLabel("VRAM: —")
        self._sb_latency = QLabel("Latency: —")
        for w in (self._sb_gpu, self._sb_vram, self._sb_latency):
            sb.addPermanentWidget(w)
        self.setStatusBar(sb)

        # ── Wire state signals → UI ──────────────────────────────────
        self._connect_state()

        # ── Periodic GPU poll (every 3 s) ─────────────────────────────
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._poll_gpu)
        self._gpu_timer.start(3000)
        self._poll_gpu()  # immediate first read

    # ------------------------------------------------------------------
    # State ↔ UI wiring
    # ------------------------------------------------------------------

    def _connect_state(self) -> None:
        s = self._state

        # Module status → A2 indicators
        s.stt_status_changed.connect(self.section_a.on_stt_status)
        s.llm_status_changed.connect(self.section_a.on_llm_status)
        s.tts_status_changed.connect(self.section_a.on_tts_status)

        # Live text → stream panels
        s.current_stt_text_changed.connect(self.section_a.stt_panel.set_text)
        s.current_translation_changed.connect(self.section_a.translation_panel.set_text)

    # ------------------------------------------------------------------
    # GPU polling
    # ------------------------------------------------------------------

    def _poll_gpu(self) -> None:
        info = get_gpu_monitor().query()
        if info.name:
            self._sb_gpu.setText(f"GPU: {info.name}")
            self._sb_vram.setText(
                f"VRAM: {info.vram_used_mb:.0f}/{info.vram_total_mb:.0f} MB"
            )
            self.section_a.update_gpu(
                info.name, info.vram_used_mb, info.vram_total_mb, info.gpu_util_pct,
            )
