"""Training placeholder tab for Section B.

This tab will host model fine-tuning and training tools in a future phase.
Currently displays a placeholder with planned feature descriptions.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from vocal10n.state import SystemState


class TrainingTab(QWidget):
    """Training tools placeholder tab."""

    def __init__(self, state: SystemState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel("Training Tools")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0;")
        root.addWidget(header)

        # ── Planned Features ──────────────────────────────────────────
        planned_box = QGroupBox("Planned Features")
        pl = QVBoxLayout(planned_box)

        features = [
            "STT fine-tuning — Adapt FasterWhisper to domain-specific vocabulary",
            "TTS voice training — Fine-tune GPT-SoVITS with custom reference audio",
            "Translation tuning — Optimize LLM translation with parallel corpora",
            "Training data management — Organize and export recorded WAV/SRT pairs",
        ]

        for feat in features:
            lbl = QLabel(f"  •  {feat}")
            lbl.setStyleSheet("color: #8892a4; font-size: 13px;")
            lbl.setWordWrap(True)
            pl.addWidget(lbl)

        root.addWidget(planned_box)

        # ── Status ────────────────────────────────────────────────────
        status = QLabel(
            "<i>Training functionality will be implemented in a future update.</i>"
        )
        status.setStyleSheet("color: #6272a4; font-size: 12px; margin-top: 8px;")
        root.addWidget(status)

        root.addStretch()
