"""Drag-and-drop term file list widget for STT context / phonetic correction."""

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class _FileItemWidget(QWidget):
    """Single row: icon + name + size + delete button."""

    remove_requested = Signal(str)  # emits file path

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self._path = file_path
        p = Path(file_path)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)

        # File info
        name_lbl = QLabel(f"<b>{p.name}</b>")
        name_lbl.setToolTip(str(p))
        lay.addWidget(name_lbl)

        # Size
        try:
            size = p.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            else:
                size_str = f"{size / 1024:.1f} KB"
        except OSError:
            size_str = "?"
        size_lbl = QLabel(size_str)
        size_lbl.setProperty("dim", True)
        size_lbl.setFixedWidth(60)
        lay.addWidget(size_lbl)

        # Line count
        try:
            lines = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
            terms_lbl = QLabel(f"{lines} terms")
            terms_lbl.setProperty("dim", True)
            terms_lbl.setFixedWidth(70)
            lay.addWidget(terms_lbl)
        except Exception:
            pass

        lay.addStretch()

        # Delete button
        del_btn = QPushButton("x")
        del_btn.setMinimumWidth(36)
        del_btn.setFixedHeight(24)
        del_btn.setStyleSheet(
            "font-size: 13px; font-weight: bold; "
            "padding: 2px; border-radius: 3px;color: #e94560;"
        )
        del_btn.setToolTip("Remove this file")
        del_btn.clicked.connect(lambda: self.remove_requested.emit(self._path))
        lay.addWidget(del_btn)

    @property
    def file_path(self) -> str:
        return self._path


class TermFileList(QFrame):
    """Widget showing loaded term files with drag/drop + add/remove.

    Emits :pyattr:`files_changed` whenever the list changes.
    Each term file should have one term per line (plain text, UTF-8).
    """

    files_changed = Signal(list)  # emits list[str] of paths

    def __init__(self, title: str = "Term Files", parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._files: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"<b>{title}</b>"))
        hdr.addStretch()
        add_btn = QPushButton("+ Add File")
        add_btn.setFixedHeight(24)
        add_btn.clicked.connect(self._browse)
        hdr.addWidget(add_btn)
        root.addLayout(hdr)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        self._list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._list.setMinimumHeight(60)
        self._list.setMaximumHeight(180)
        root.addWidget(self._list)

        # Placeholder
        self._placeholder = QLabel(
            "Drag and drop term files here\n"
            "(one term per line, plain text UTF-8)"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setProperty("dim", True)
        self._placeholder.setStyleSheet("color: #8892a4; font-style: italic; padding: 12px;")
        root.addWidget(self._placeholder)

        self._update_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_file(self, path: str) -> None:
        """Add a term file to the list."""
        path = str(Path(path).resolve())
        if path in self._files:
            return
        if not Path(path).is_file():
            return
        self._files.append(path)
        self._add_row(path)
        self._update_visibility()
        self.files_changed.emit(list(self._files))

    def remove_file(self, path: str) -> None:
        """Remove a term file from the list."""
        if path not in self._files:
            return
        self._files.remove(path)
        self._rebuild_list()
        self._update_visibility()
        self.files_changed.emit(list(self._files))

    @property
    def files(self) -> list[str]:
        return list(self._files)

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and Path(path).is_file() and Path(path).suffix in (".txt", ".csv", ".tsv", ""):
                self.add_file(path)
        event.acceptProposedAction()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _browse(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Term Files", "",
            "Text Files (*.txt *.csv *.tsv);;All Files (*)",
        )
        for p in paths:
            self.add_file(p)

    def _add_row(self, path: str) -> None:
        item = QListWidgetItem()
        widget = _FileItemWidget(path)
        widget.remove_requested.connect(self.remove_file)
        item.setSizeHint(widget.sizeHint())
        self._list.addItem(item)
        self._list.setItemWidget(item, widget)

    def _rebuild_list(self) -> None:
        self._list.clear()
        for path in self._files:
            self._add_row(path)

    def _update_visibility(self) -> None:
        has_files = len(self._files) > 0
        self._list.setVisible(has_files)
        self._placeholder.setVisible(not has_files)
