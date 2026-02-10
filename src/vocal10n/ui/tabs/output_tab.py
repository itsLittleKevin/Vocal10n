"""Output settings tab for Section B."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.state import SystemState

# Default output directory (project root / output)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"


class OutputTab(QWidget):
    """Settings tab for pipeline file output (SRT, TXT, WAV)."""

    # Emitted whenever any output setting changes
    settings_changed = Signal()
    # Manual session control
    start_session_requested = Signal()
    stop_session_requested = Signal()

    def __init__(self, state: SystemState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()
        self._session_active = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Session Control ───────────────────────────────────────────
        session_box = QGroupBox("Session Control")
        scl = QVBoxLayout(session_box)

        scl.addWidget(QLabel(
            "<i>Sessions auto-start when STT begins. "
            "Use the button below to manually start/stop a recording session.</i>"
        ))

        btn_row = QHBoxLayout()
        self._session_btn = QPushButton("Start Session")
        self._session_btn.setFixedHeight(36)
        self._session_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 0 24px; }"
        )
        self._session_btn.clicked.connect(self._on_session_btn)
        btn_row.addWidget(self._session_btn)
        btn_row.addStretch()
        scl.addLayout(btn_row)

        self._session_label = QLabel("No active session")
        scl.addWidget(self._session_label)

        root.addWidget(session_box)

        # ── Output Directory ──────────────────────────────────────────
        dir_box = QGroupBox("Output Directory")
        dir_lay = QHBoxLayout(dir_box)

        self._dir_edit = QLineEdit()
        self._dir_edit.setReadOnly(True)
        saved_dir = self._cfg.get("output.directory", str(_DEFAULT_OUTPUT_DIR))
        self._dir_edit.setText(saved_dir)
        dir_lay.addWidget(self._dir_edit, stretch=1)

        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedHeight(36)
        browse_btn.clicked.connect(self._browse_dir)
        dir_lay.addWidget(browse_btn)

        root.addWidget(dir_box)

        # ── Source Language Outputs ───────────────────────────────────
        src_box = QGroupBox("Source Language Outputs")
        sl = QVBoxLayout(src_box)

        self._src_txt_cb = QCheckBox(
            "Save source TXT (corrected transcript with punctuation)"
        )
        self._src_txt_cb.setChecked(self._cfg.get("output.save_source_txt", False))
        self._src_txt_cb.toggled.connect(
            lambda c: self._on_toggle("output.save_source_txt", c)
        )
        sl.addWidget(self._src_txt_cb)

        self._src_srt_cb = QCheckBox(
            "Save source SRT (corrected transcript, no punctuation, space-separated)"
        )
        self._src_srt_cb.setChecked(self._cfg.get("output.save_source_srt", False))
        self._src_srt_cb.toggled.connect(
            lambda c: self._on_toggle("output.save_source_srt", c)
        )
        sl.addWidget(self._src_srt_cb)

        root.addWidget(src_box)

        # ── Target Language Outputs ───────────────────────────────────
        tgt_box = QGroupBox("Target Language Outputs")
        tl = QVBoxLayout(tgt_box)

        self._tgt_txt_cb = QCheckBox(
            "Save target TXT (translated text with punctuation)"
        )
        self._tgt_txt_cb.setChecked(self._cfg.get("output.save_target_txt", False))
        self._tgt_txt_cb.toggled.connect(
            lambda c: self._on_toggle("output.save_target_txt", c)
        )
        tl.addWidget(self._tgt_txt_cb)

        self._tgt_srt_cb = QCheckBox(
            "Save target SRT (translated text, no punctuation, space-separated)"
        )
        self._tgt_srt_cb.setChecked(self._cfg.get("output.save_target_srt", False))
        self._tgt_srt_cb.toggled.connect(
            lambda c: self._on_toggle("output.save_target_srt", c)
        )
        tl.addWidget(self._tgt_srt_cb)

        root.addWidget(tgt_box)

        # ── Audio Output ─────────────────────────────────────────────
        wav_box = QGroupBox("Audio Recording")
        wl = QVBoxLayout(wav_box)

        self._wav_cb = QCheckBox(
            "Save WAV audio file (for review and training)"
        )
        self._wav_cb.setChecked(self._cfg.get("output.save_wav", False))
        self._wav_cb.toggled.connect(
            lambda c: self._on_toggle("output.save_wav", c)
        )
        wl.addWidget(self._wav_cb)

        wl.addWidget(QLabel(
            "<i>Raw microphone audio saved as 16-bit PCM WAV (16 kHz mono).</i>"
        ))

        root.addWidget(wav_box)

        # ── Info ──────────────────────────────────────────────────────
        root.addWidget(QLabel(
            "<i>Files are saved to output/subtitles/ with timestamped names.</i>"
        ))
        root.addStretch()

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _browse_dir(self) -> None:
        current = self._dir_edit.text() or str(_DEFAULT_OUTPUT_DIR)
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", current
        )
        if chosen:
            self._dir_edit.setText(chosen)
            self._cfg.set("output.directory", chosen)
            self.settings_changed.emit()

    def _on_toggle(self, key: str, checked: bool) -> None:
        self._cfg.set(key, checked)
        self.settings_changed.emit()

    def _on_session_btn(self) -> None:
        if self._session_active:
            self.stop_session_requested.emit()
        else:
            self.start_session_requested.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session_info(self, text: str) -> None:
        """Update the session info label (called by coordinator)."""
        self._session_label.setText(text)

    def set_session_active(self, active: bool) -> None:
        """Update button state to reflect session status."""
        self._session_active = active
        if active:
            self._session_btn.setText("Stop Session")
            self._session_label.setText("Session active — recording outputs")
        else:
            self._session_btn.setText("Start Session")
            self._session_label.setText("No active session")
