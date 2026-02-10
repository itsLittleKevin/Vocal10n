"""Section A — top portion of the main window.

Layout (horizontal):
  A1 (≈70 %)  Two side-by-side columns, each split vertically:
               • Left column  — Source speech
                   - Upper (30%): Live STT (ephemeral, no punctuation)
                   - Lower (70%): Accumulated STT (corrected, with punctuation)
               • Right column — Translation
                   - Upper (30%): Live translation (ephemeral)
                   - Lower (70%): Accumulated translation (full history)
  A2 (≈30 %)  Status / metrics column:
               • Module status indicators (STT, LLM, TTS)
               • Latency stats
               • GPU VRAM bar
"""

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from vocal10n.constants import ModelStatus
from vocal10n.ui.widgets.stream_text import StreamText


# ── Tiny status-dot helper ────────────────────────────────────────────
_STATUS_COLOURS = {
    ModelStatus.UNLOADED:  "#8892a4",
    ModelStatus.LOADING:   "#f0c030",
    ModelStatus.LOADED:    "#0f9b8e",
    ModelStatus.UNLOADING: "#f0c030",
    ModelStatus.ERROR:     "#e94560",
}


def _dot(status: ModelStatus) -> str:
    c = _STATUS_COLOURS.get(status, "#8892a4")
    return f'<span style="color:{c}; font-size:16px;">●</span>'


def _text_column(live_title: str, live_placeholder: str,
                 acc_title: str, acc_placeholder: str) -> tuple[QSplitter, StreamText, StreamText]:
    """Build a vertical splitter with live (30%) + accumulated (70%) panels."""
    splitter = QSplitter(Qt.Vertical)
    live = StreamText(title=live_title, placeholder=live_placeholder)
    accumulated = StreamText(title=acc_title, placeholder=acc_placeholder)
    splitter.addWidget(live)
    splitter.addWidget(accumulated)
    splitter.setStretchFactor(0, 30)
    splitter.setStretchFactor(1, 70)
    return splitter, live, accumulated


# ======================================================================
# Section A
# ======================================================================

class SectionA(QWidget):
    """Top section: text streams + metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)

        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 0)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)

        # ── A1: text streams ──────────────────────────────────────────
        a1 = QWidget()
        a1_lay = QHBoxLayout(a1)
        a1_lay.setContentsMargins(0, 0, 0, 0)
        a1_lay.setSpacing(4)

        # Left column — Source speech (live + accumulated)
        stt_col, self.stt_live_panel, self.stt_accumulated_panel = _text_column(
            live_title="Live STT",
            live_placeholder="Real-time speech recognition…",
            acc_title="Source Speech (corrected)",
            acc_placeholder="Accumulated corrected text will appear here…",
        )
        a1_lay.addWidget(stt_col)

        # Right column — Translation (live + accumulated)
        trans_col, self.translation_live_panel, self.translation_accumulated_panel = _text_column(
            live_title="Live Translation",
            live_placeholder="Real-time translation preview…",
            acc_title="Translation (full)",
            acc_placeholder="Accumulated translated text will appear here…",
        )
        a1_lay.addWidget(trans_col)

        # Keep backward-compat aliases for existing wiring
        self.stt_panel = self.stt_live_panel
        self.translation_panel = self.translation_live_panel

        splitter.addWidget(a1)

        # ── A2: metrics / status ──────────────────────────────────────
        a2 = QWidget()
        a2.setMinimumWidth(200)
        a2.setMaximumWidth(320)
        a2_lay = QVBoxLayout(a2)
        a2_lay.setContentsMargins(0, 0, 0, 0)
        a2_lay.setSpacing(6)

        # -- Module status --
        status_box = QGroupBox("Modules")
        sb_lay = QVBoxLayout(status_box)
        sb_lay.setSpacing(4)

        self._stt_status_label = QLabel()
        self._llm_status_label = QLabel()
        self._tts_status_label = QLabel()
        for lbl in (self._stt_status_label, self._llm_status_label, self._tts_status_label):
            lbl.setTextFormat(Qt.RichText)
            sb_lay.addWidget(lbl)

        self._set_module_row(self._stt_status_label, "STT", ModelStatus.UNLOADED)
        self._set_module_row(self._llm_status_label, "LLM", ModelStatus.UNLOADED)
        self._set_module_row(self._tts_status_label, "TTS", ModelStatus.UNLOADED)
        a2_lay.addWidget(status_box)

        # -- Latency --
        latency_box = QGroupBox("Latency")
        ll = QVBoxLayout(latency_box)
        ll.setSpacing(2)
        self._latency_labels: dict[str, QLabel] = {}
        for key in ("STT", "Translation", "TTS", "Total"):
            lbl = QLabel(f"{key}: —")
            lbl.setProperty("dim", True)
            ll.addWidget(lbl)
            self._latency_labels[key.lower()] = lbl
        a2_lay.addWidget(latency_box)

        # -- GPU --
        gpu_box = QGroupBox("GPU")
        gl = QVBoxLayout(gpu_box)
        gl.setSpacing(4)
        self._gpu_name_label = QLabel("—")
        self._gpu_name_label.setProperty("dim", True)
        gl.addWidget(self._gpu_name_label)

        self._vram_bar = QProgressBar()
        self._vram_bar.setRange(0, 100)
        self._vram_bar.setValue(0)
        self._vram_bar.setTextVisible(True)
        self._vram_bar.setFormat("VRAM: %p%")
        gl.addWidget(self._vram_bar)

        self._vram_detail = QLabel("— / — MB")
        self._vram_detail.setProperty("dim", True)
        gl.addWidget(self._vram_detail)
        a2_lay.addWidget(gpu_box)

        a2_lay.addStretch()
        splitter.addWidget(a2)

        # Splitter proportions  ≈ 70 / 30
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Slots (connected by MainWindow / app)
    # ------------------------------------------------------------------

    @Slot(ModelStatus)
    def on_stt_status(self, s: ModelStatus) -> None:
        self._set_module_row(self._stt_status_label, "STT", s)

    @Slot(ModelStatus)
    def on_llm_status(self, s: ModelStatus) -> None:
        self._set_module_row(self._llm_status_label, "LLM", s)

    @Slot(ModelStatus)
    def on_tts_status(self, s: ModelStatus) -> None:
        self._set_module_row(self._tts_status_label, "TTS", s)

    def update_latency(self, component: str, current_ms: float, avg_ms: float) -> None:
        lbl = self._latency_labels.get(component)
        if lbl:
            name = component.capitalize()
            lbl.setText(f"{name}: {current_ms:.0f} ms  (avg {avg_ms:.0f})")
            lbl.setProperty("dim", False)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def update_gpu(self, name: str, used_mb: float, total_mb: float, util_pct: float) -> None:
        self._gpu_name_label.setText(name or "—")
        pct = int(used_mb / total_mb * 100) if total_mb else 0
        self._vram_bar.setValue(pct)
        self._vram_detail.setText(f"{used_mb:.0f} / {total_mb:.0f} MB")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_module_row(label: QLabel, name: str, status: ModelStatus) -> None:
        label.setText(f"{_dot(status)}  {name}: {status.value}")
