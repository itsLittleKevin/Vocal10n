"""Filter list editor widget for STT hallucination filters."""

from pathlib import Path

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class FilterListEditor(QFrame):
    """Widget for editing STT filter phrases.

    Shows current filters from filters.txt, allows add/remove/edit.
    Emits :pyattr:`filters_changed` when the list is modified.
    """

    filters_changed = Signal(list)  # emits list[str] of filter phrases

    def __init__(self, filters_path: Path, parent=None):
        super().__init__(parent)
        self._filters_path = filters_path
        self._filters: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Header with buttons
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("Hallucination Filters")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        header.addWidget(title)

        header.addStretch()

        add_btn = QPushButton("+ Add")
        add_btn.setToolTip("Add a new filter phrase")
        add_btn.clicked.connect(self._add_filter)
        header.addWidget(add_btn)

        reload_btn = QPushButton("Reload")
        reload_btn.setToolTip("Reload filters from file")
        reload_btn.clicked.connect(self._reload_filters)
        header.addWidget(reload_btn)

        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save changes to filters.txt")
        save_btn.clicked.connect(self._save_filters)
        header.addWidget(save_btn)

        root.addLayout(header)

        # Filter list
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.MultiSelection)
        self._list.itemDoubleClicked.connect(self._edit_filter)
        root.addWidget(self._list)

        # Bottom buttons
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.setToolTip("Remove selected filter phrases")
        remove_btn.clicked.connect(self._remove_selected)
        bottom.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setToolTip("Remove all filters")
        clear_btn.clicked.connect(self._clear_all)
        bottom.addWidget(clear_btn)

        bottom.addStretch()

        info = QLabel("Double-click to edit")
        info.setProperty("dim", True)
        info.setStyleSheet("font-size: 11px; color: #8892a4;")
        bottom.addWidget(info)

        root.addLayout(bottom)

        # Load initial filters
        self._reload_filters()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _reload_filters(self) -> None:
        """Load filters from the file."""
        try:
            if self._filters_path.exists():
                lines = self._filters_path.read_text(encoding="utf-8").splitlines()
                self._filters = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if stripped.startswith("PHRASE:"):
                        self._filters.append(stripped[7:].strip())
                    else:
                        # Keep REGEX: lines and plain phrases as-is
                        self._filters.append(stripped)
            else:
                self._filters = []
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load filters: {e}")
            self._filters = []

        self._update_list()

    @Slot()
    def _save_filters(self) -> None:
        """Save current filters back to the file, preserving comment header."""
        try:
            # Read existing to preserve leading comment block
            header_lines: list[str] = []
            if self._filters_path.exists():
                for line in self._filters_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line.strip():
                        header_lines.append(line)
                    else:
                        break

            # Build output
            out_lines = list(header_lines)
            for f in self._filters:
                out_lines.append(f)

            self._filters_path.parent.mkdir(parents=True, exist_ok=True)
            self._filters_path.write_text(
                "\n".join(out_lines) + "\n", encoding="utf-8"
            )
            QMessageBox.information(
                self, "Saved",
                "Filters saved. They will be applied on next STT restart.",
            )
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save filters: {e}")

    @Slot()
    def _add_filter(self) -> None:
        """Add a new filter phrase via input dialog."""
        text, ok = QInputDialog.getText(
            self,
            "Add Filter",
            "Enter filter phrase (or REGEX:pattern for regex):",
        )
        if ok and text.strip():
            self._filters.append(text.strip())
            self._update_list()
            self.filters_changed.emit(self._filters)

    @Slot(QListWidgetItem)
    def _edit_filter(self, item: QListWidgetItem) -> None:
        """Edit an existing filter via input dialog."""
        current_text = item.text()
        new_text, ok = QInputDialog.getText(
            self, "Edit Filter", "Edit filter phrase:", text=current_text
        )
        if ok and new_text.strip() and new_text != current_text:
            idx = self._list.row(item)
            self._filters[idx] = new_text.strip()
            self._update_list()
            self.filters_changed.emit(self._filters)

    @Slot()
    def _remove_selected(self) -> None:
        """Remove selected filters from the list."""
        selected = self._list.selectedItems()
        if not selected:
            return
        reply = QMessageBox.question(
            self,
            "Confirm Remove",
            f"Remove {len(selected)} selected filter(s)?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            indices = sorted(
                [self._list.row(item) for item in selected], reverse=True
            )
            for idx in indices:
                del self._filters[idx]
            self._update_list()
            self.filters_changed.emit(self._filters)

    @Slot()
    def _clear_all(self) -> None:
        """Clear all filters."""
        reply = QMessageBox.question(
            self,
            "Confirm Clear",
            "Remove all filters? This cannot be undone without reloading.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._filters.clear()
            self._update_list()
            self.filters_changed.emit(self._filters)

    def _update_list(self) -> None:
        """Refresh the QListWidget from ``self._filters``."""
        self._list.clear()
        for f in self._filters:
            item = QListWidgetItem(f)
            item.setToolTip("Double-click to edit")
            self._list.addItem(item)
