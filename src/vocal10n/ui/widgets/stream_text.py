"""Streaming text display widget for live STT / translation output."""

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QTextCursor, QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class StreamText(QFrame):
    """A text panel that shows streaming content with a header and clear button.

    Supports two display modes:
    - *pending* text (gray italic) — partial / in-progress recognition
    - *confirmed* text (normal) — finalised segments appended to history
    """

    # Colours (match theme.qss palette)
    _PENDING_COLOR = QColor("#8892a4")
    _CONFIRMED_COLOR = QColor("#e0e0e0")

    def __init__(self, title: str = "", placeholder: str = "", parent=None):
        super().__init__(parent)
        self._confirmed_lines: list[str] = []

        # --- layout ---
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Header row
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        header.addWidget(self._title_label)
        header.addStretch()
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedSize(52, 22)
        self._clear_btn.setStyleSheet("font-size: 11px;")
        self._clear_btn.clicked.connect(self.clear)
        header.addWidget(self._clear_btn)
        root.addLayout(header)

        # Text area
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setPlaceholderText(placeholder)
        self._text.setAcceptRichText(False)
        root.addWidget(self._text)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @Slot(str)
    def set_pending(self, text: str) -> None:
        """Show *pending* (partial) text below confirmed history."""
        self._render(pending=text)

    @Slot(str)
    def append_confirmed(self, text: str) -> None:
        """Append a finalised segment and clear any pending text."""
        text = text.strip()
        if text:
            self._confirmed_lines.append(text)
            # Keep rolling buffer
            if len(self._confirmed_lines) > 200:
                self._confirmed_lines = self._confirmed_lines[-200:]
        self._render(pending="")

    @Slot(str)
    def set_text(self, text: str) -> None:
        """Replace entire contents (for simple single-stream display)."""
        self._text.setPlainText(text)
        self._scroll_to_bottom()

    @Slot()
    def clear(self) -> None:
        self._confirmed_lines.clear()
        self._text.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _render(self, pending: str = "") -> None:
        """Re-render confirmed history + pending tail."""
        self._text.clear()
        cursor: QTextCursor = self._text.textCursor()

        # Confirmed
        if self._confirmed_lines:
            fmt_confirmed = cursor.charFormat()
            fmt_confirmed.setForeground(self._CONFIRMED_COLOR)
            fmt_confirmed.setFontItalic(False)
            cursor.setCharFormat(fmt_confirmed)
            cursor.insertText("\n".join(self._confirmed_lines))

        # Pending
        if pending:
            if self._confirmed_lines:
                cursor.insertText("\n")
            fmt_pending = cursor.charFormat()
            fmt_pending.setForeground(self._PENDING_COLOR)
            fmt_pending.setFontItalic(True)
            cursor.setCharFormat(fmt_pending)
            cursor.insertText(pending)

        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        sb = self._text.verticalScrollBar()
        sb.setValue(sb.maximum())
