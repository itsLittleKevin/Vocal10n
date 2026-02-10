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

from vocal10n.constants import ModelStatus
from vocal10n.llm.controller import LLMController
from vocal10n.pipeline.latency import LatencyTracker
from vocal10n.state import SystemState
from vocal10n.stt.controller import STTController
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
        self.section_b = SectionB(state)

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

        # ── STT controller ────────────────────────────────────────────
        self._latency = LatencyTracker()
        self._stt_ctrl = STTController(state, self._latency, parent=self)

        # Connect STT tab load/unload → controller
        stt_tab = self.section_b.stt_tab
        stt_tab._model_sel.load_requested.connect(self._stt_ctrl.load_model)
        stt_tab._model_sel.unload_requested.connect(self._stt_ctrl.unload_model)
        stt_tab.term_files_changed.connect(self._stt_ctrl.update_term_files)
        # ── LLM controller ────────────────────────────────────────────
        self._llm_ctrl = LLMController(state, self._latency, parent=self)

        # Connect Translation tab load/unload → controller
        trans_tab = self.section_b.translation_tab
        trans_tab.load_requested.connect(self._llm_ctrl.load_model)
        trans_tab.unload_requested.connect(self._llm_ctrl.unload_model)
        trans_tab.target_language_changed.connect(self._llm_ctrl.set_target_language)

        # Manual input: source panel text → LLM translation
        self.section_a.stt_accumulated_panel.text_submitted.connect(
            self._llm_ctrl.translate_manual_text
        )

        # Connect latency tracker → Section A display
        self._latency.stats_updated.connect(self._on_latency_stats)

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

        # Module status → manual input mode toggle
        s.stt_status_changed.connect(self._update_manual_mode)
        s.llm_status_changed.connect(self._update_manual_mode)

        # Live text → stream panels (upper live panels)
        s.current_stt_text_changed.connect(self.section_a.stt_live_panel.set_text)
        s.current_translation_changed.connect(self.section_a.translation_live_panel.set_text)

        # Accumulated text → lower panels
        s.accumulated_stt_text_changed.connect(self.section_a.stt_accumulated_panel.set_text)
        s.accumulated_translation_changed.connect(self.section_a.translation_accumulated_panel.set_text)

    # ------------------------------------------------------------------
    # Manual input mode
    # ------------------------------------------------------------------

    def _update_manual_mode(self, *_) -> None:
        """Enable editable source panel when LLM loaded but STT is not."""
        stt_loaded = self._state.stt_status == ModelStatus.LOADED
        llm_loaded = self._state.llm_status == ModelStatus.LOADED
        self.section_a.stt_accumulated_panel.set_editable(llm_loaded and not stt_loaded)

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

    # ------------------------------------------------------------------
    # Latency display
    # ------------------------------------------------------------------

    def _on_latency_stats(self) -> None:
        stats = self._latency.get_all_stats()
        for component in ("stt", "translation", "tts", "total"):
            s = stats.get(component)
            if s and s.count > 0:
                self.section_a.update_latency(component, s.current_ms, s.avg_5s_ms)
                if component == "total":
                    self._sb_latency.setText(f"Latency: {s.current_ms:.0f} ms")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._llm_ctrl.shutdown()
        self._stt_ctrl.shutdown()
        super().closeEvent(event)
