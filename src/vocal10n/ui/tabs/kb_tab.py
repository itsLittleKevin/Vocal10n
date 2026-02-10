"""Knowledge Base tab for Section B.

Manages two parallel systems:

1. **Translation Glossary** (``knowledge_base/*.txt``)
   Format: ``source_term|translation`` per line.
   Used by the Corrector for fuzzy pinyin matching + translation prompt
   injection.  Switches to RAG vector search when term count exceeds
   threshold.

2. **STT Recognition Context** (``stt_terms/*.txt``)
   Format: one plain term per line.
   Fed into Whisper's ``initial_prompt`` to bias recognition toward
   domain-specific vocabulary.

Both systems present a consistent file-table + inline-editor UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.state import SystemState
from vocal10n.ui.widgets.param_slider import ParamSlider

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_KB_DIR = _PROJECT_ROOT / "knowledge_base"
_STT_TERMS_DIR = _PROJECT_ROOT / "stt_terms"


class KnowledgeBaseTab(QWidget):
    """Manages translation glossary and STT recognition term files."""

    glossary_changed = Signal()  # emitted when glossaries are modified
    term_files_changed = Signal(list)  # emits list[str] of STT term file paths

    def __init__(self, state: SystemState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        # Track current file for each system
        self._gloss_current: Path | None = None
        self._stt_current: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel("Knowledge Base")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0;")
        root.addWidget(header)

        desc = QLabel(
            "Two systems work together to improve accuracy:<br>"
            "<b>Translation Glossary</b> — fixes STT errors and guides "
            "translation (post-recognition).<br>"
            "<b>STT Recognition Terms</b> — biases Whisper to hear "
            "domain words correctly (during recognition)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8892a4; font-size: 12px; margin-bottom: 4px;")
        root.addWidget(desc)

        # ══════════════════════════════════════════════════════════════
        # SECTION A: Translation Glossary  (knowledge_base/*.txt)
        # ══════════════════════════════════════════════════════════════
        gloss_box = QGroupBox(
            "Translation Glossary  —  knowledge_base/*.txt"
        )
        gl = QVBoxLayout(gloss_box)

        gloss_info = QLabel(
            "Format: <b>source_term|translation</b> per line. "
            "Used after STT to correct errors and inject hints into "
            "the LLM translation prompt."
        )
        gloss_info.setWordWrap(True)
        gloss_info.setStyleSheet("color: #8892a4; font-size: 11px;")
        gl.addWidget(gloss_info)

        # File table
        self._gloss_table = QTableWidget(0, 3)
        self._gloss_table.setHorizontalHeaderLabels(["File", "Terms", "Status"])
        self._gloss_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._gloss_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._gloss_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self._gloss_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._gloss_table.setSelectionMode(QTableWidget.SingleSelection)
        self._gloss_table.setMaximumHeight(120)
        self._gloss_table.verticalHeader().setVisible(False)
        self._gloss_table.currentCellChanged.connect(self._on_gloss_file_selected)
        gl.addWidget(self._gloss_table)

        # Buttons row
        gbtn = QHBoxLayout()
        self._btn_gloss_new = QPushButton("New")
        self._btn_gloss_new.setFixedWidth(60)
        self._btn_gloss_new.clicked.connect(self._on_gloss_new)
        gbtn.addWidget(self._btn_gloss_new)

        self._btn_gloss_import = QPushButton("Import")
        self._btn_gloss_import.setFixedWidth(70)
        self._btn_gloss_import.clicked.connect(self._on_gloss_import)
        gbtn.addWidget(self._btn_gloss_import)

        self._btn_gloss_delete = QPushButton("Delete")
        self._btn_gloss_delete.setFixedWidth(80)
        self._btn_gloss_delete.setEnabled(False)
        self._btn_gloss_delete.clicked.connect(self._on_gloss_delete)
        gbtn.addWidget(self._btn_gloss_delete)

        gbtn.addStretch()

        self._rag_label = QLabel("RAG: inactive")
        self._rag_label.setStyleSheet("color: #8892a4; font-size: 11px;")
        gbtn.addWidget(self._rag_label)
        gl.addLayout(gbtn)

        # Editor
        self._gloss_editor = QTextEdit()
        self._gloss_editor.setPlaceholderText(
            "# Example glossary\n"
            "# source_term|preferred_translation\n"
            "科技小院|Science and Technology Backyard\n"
            "中国农业大学|China Agricultural University"
        )
        self._gloss_editor.setMinimumHeight(140)
        self._gloss_editor.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;"
        )
        gl.addWidget(self._gloss_editor)

        # Save / Revert / count
        ge_row = QHBoxLayout()
        self._btn_gloss_save = QPushButton("Save")
        self._btn_gloss_save.setProperty("accent", True)
        self._btn_gloss_save.setFixedWidth(80)
        self._btn_gloss_save.setEnabled(False)
        self._btn_gloss_save.clicked.connect(self._on_gloss_save)
        ge_row.addWidget(self._btn_gloss_save)

        self._btn_gloss_revert = QPushButton("Revert")
        self._btn_gloss_revert.setFixedWidth(80)
        self._btn_gloss_revert.setEnabled(False)
        self._btn_gloss_revert.clicked.connect(self._on_gloss_revert)
        ge_row.addWidget(self._btn_gloss_revert)

        ge_row.addStretch()

        self._gloss_count = QLabel("")
        self._gloss_count.setStyleSheet("color: #8892a4; font-size: 11px;")
        ge_row.addWidget(self._gloss_count)
        gl.addLayout(ge_row)

        # RAG threshold
        rag_row = QHBoxLayout()
        rag_row.addWidget(QLabel("RAG threshold:"))
        self._thresh_edit = QLineEdit()
        self._thresh_edit.setFixedWidth(60)
        self._thresh_edit.setText(
            str(self._cfg.get("translation.rag_threshold", 100))
        )
        self._thresh_edit.editingFinished.connect(self._on_threshold_changed)
        rag_row.addWidget(self._thresh_edit)
        rag_row.addWidget(QLabel(
            "terms  (auto-switches from pinyin scan to vector search)"
        ))
        rag_row.addStretch()
        gl.addLayout(rag_row)

        root.addWidget(gloss_box)

        # ══════════════════════════════════════════════════════════════
        # SECTION B: STT Recognition Terms  (stt_terms/*.txt)
        # ══════════════════════════════════════════════════════════════
        stt_box = QGroupBox(
            "STT Recognition Terms  —  stt_terms/*.txt"
        )
        sl = QVBoxLayout(stt_box)

        stt_info = QLabel(
            "Format: <b>one term per line</b> (plain text, no translations). "
            "Fed into Whisper's initial_prompt to bias recognition toward "
            "these words during speech-to-text."
        )
        stt_info.setWordWrap(True)
        stt_info.setStyleSheet("color: #8892a4; font-size: 11px;")
        sl.addWidget(stt_info)

        # File table
        self._stt_table = QTableWidget(0, 2)
        self._stt_table.setHorizontalHeaderLabels(["File", "Terms"])
        self._stt_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._stt_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self._stt_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._stt_table.setSelectionMode(QTableWidget.SingleSelection)
        self._stt_table.setMaximumHeight(100)
        self._stt_table.verticalHeader().setVisible(False)
        self._stt_table.currentCellChanged.connect(self._on_stt_file_selected)
        sl.addWidget(self._stt_table)

        # Buttons
        sbtn = QHBoxLayout()
        self._btn_stt_new = QPushButton("New")
        self._btn_stt_new.setFixedWidth(60)
        self._btn_stt_new.clicked.connect(self._on_stt_new)
        sbtn.addWidget(self._btn_stt_new)

        self._btn_stt_import = QPushButton("Import")
        self._btn_stt_import.setFixedWidth(70)
        self._btn_stt_import.clicked.connect(self._on_stt_import)
        sbtn.addWidget(self._btn_stt_import)

        self._btn_stt_delete = QPushButton("Delete")
        self._btn_stt_delete.setFixedWidth(80)
        self._btn_stt_delete.setEnabled(False)
        self._btn_stt_delete.clicked.connect(self._on_stt_delete)
        sbtn.addWidget(self._btn_stt_delete)

        sbtn.addStretch()

        self._stt_total_label = QLabel("")
        self._stt_total_label.setStyleSheet("color: #8892a4; font-size: 11px;")
        sbtn.addWidget(self._stt_total_label)
        sl.addLayout(sbtn)

        # Editor
        self._stt_editor = QTextEdit()
        self._stt_editor.setPlaceholderText(
            "# One term per line\n"
            "# These words bias Whisper recognition\n"
            "科技小院\n"
            "中国农业大学\n"
            "曲周"
        )
        self._stt_editor.setMinimumHeight(120)
        self._stt_editor.setStyleSheet(
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;"
        )
        sl.addWidget(self._stt_editor)

        # Save / Revert / count
        se_row = QHBoxLayout()
        self._btn_stt_save = QPushButton("Save")
        self._btn_stt_save.setProperty("accent", True)
        self._btn_stt_save.setFixedWidth(80)
        self._btn_stt_save.setEnabled(False)
        self._btn_stt_save.clicked.connect(self._on_stt_save)
        se_row.addWidget(self._btn_stt_save)

        self._btn_stt_revert = QPushButton("Revert")
        self._btn_stt_revert.setFixedWidth(80)
        self._btn_stt_revert.setEnabled(False)
        self._btn_stt_revert.clicked.connect(self._on_stt_revert)
        se_row.addWidget(self._btn_stt_revert)

        se_row.addStretch()

        self._stt_count = QLabel("")
        self._stt_count.setStyleSheet("color: #8892a4; font-size: 11px;")
        se_row.addWidget(self._stt_count)
        sl.addLayout(se_row)

        # Capacity slider
        cap_row = QHBoxLayout()
        self._capacity_slider = ParamSlider(
            "Initial Prompt Capacity",
            minimum=50, maximum=500,
            default=self._cfg.get("stt.initial_prompt_capacity", 200),
            step=25,
            tooltip="Max terms to include in Whisper's initial_prompt.\n"
                    "Larger = better recognition, but higher latency.",
        )
        self._capacity_slider.value_changed.connect(
            lambda v: self._cfg.set("stt.initial_prompt_capacity", int(v))
        )
        cap_row.addWidget(self._capacity_slider)
        sl.addLayout(cap_row)

        root.addWidget(stt_box)

        root.addStretch()

        # Initial scan
        self._scan_gloss_files()
        self._scan_stt_files()

    # ==================================================================
    # GLOSSARY file management
    # ==================================================================

    def _scan_gloss_files(self) -> None:
        """Scan knowledge_base/ and populate the glossary file table."""
        self._gloss_table.setRowCount(0)
        _KB_DIR.mkdir(parents=True, exist_ok=True)

        total_terms = 0
        for f in sorted(_KB_DIR.glob("*.txt")):
            row = self._gloss_table.rowCount()
            self._gloss_table.insertRow(row)
            self._gloss_table.setItem(row, 0, QTableWidgetItem(f.name))

            tc = self._count_terms(f)
            total_terms += tc
            ci = QTableWidgetItem(str(tc))
            ci.setTextAlignment(Qt.AlignCenter)
            self._gloss_table.setItem(row, 1, ci)

            si = QTableWidgetItem("Loaded")
            si.setTextAlignment(Qt.AlignCenter)
            self._gloss_table.setItem(row, 2, si)

        # Update RAG status
        threshold = self._cfg.get("translation.rag_threshold", 100)
        if total_terms >= threshold:
            self._rag_label.setText(
                f"RAG: active ({total_terms} ≥ {threshold})"
            )
            self._rag_label.setStyleSheet("color: #0f9b8e; font-size: 11px;")
        else:
            self._rag_label.setText(
                f"RAG: inactive ({total_terms} < {threshold})"
            )
            self._rag_label.setStyleSheet("color: #8892a4; font-size: 11px;")

    @Slot(int, int, int, int)
    def _on_gloss_file_selected(
        self, row: int, _c: int, _pr: int, _pc: int
    ) -> None:
        if row < 0:
            self._gloss_current = None
            self._gloss_editor.clear()
            self._btn_gloss_save.setEnabled(False)
            self._btn_gloss_revert.setEnabled(False)
            self._btn_gloss_delete.setEnabled(False)
            return

        name = self._gloss_table.item(row, 0).text()
        path = _KB_DIR / name
        self._gloss_current = path
        self._btn_gloss_delete.setEnabled(True)

        if path.exists():
            self._gloss_editor.setPlainText(
                path.read_text(encoding="utf-8")
            )
            self._btn_gloss_save.setEnabled(True)
            self._btn_gloss_revert.setEnabled(True)
            self._update_gloss_count()

    @Slot()
    def _on_gloss_new(self) -> None:
        _KB_DIR.mkdir(parents=True, exist_ok=True)
        i = 1
        while (_KB_DIR / f"glossary_{i:02d}.txt").exists():
            i += 1
        path = _KB_DIR / f"glossary_{i:02d}.txt"
        path.write_text(
            "# New Glossary\n"
            "# Format: source_term|preferred_translation\n"
            "#\n",
            encoding="utf-8",
        )
        self._scan_gloss_files()
        for row in range(self._gloss_table.rowCount()):
            if self._gloss_table.item(row, 0).text() == path.name:
                self._gloss_table.selectRow(row)
                break
        self.glossary_changed.emit()

    @Slot()
    def _on_gloss_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Glossary File", "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return
        src = Path(path)
        dest = _KB_DIR / src.name
        if dest.exists():
            ret = QMessageBox.question(
                self, "File Exists",
                f"{dest.name} already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        self._scan_gloss_files()
        self.glossary_changed.emit()

    @Slot()
    def _on_gloss_delete(self) -> None:
        if not self._gloss_current or not self._gloss_current.exists():
            return
        ret = QMessageBox.question(
            self, "Delete Glossary",
            f"Delete {self._gloss_current.name}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self._gloss_current.unlink()
        self._gloss_current = None
        self._gloss_editor.clear()
        self._scan_gloss_files()
        self.glossary_changed.emit()

    @Slot()
    def _on_gloss_save(self) -> None:
        if not self._gloss_current:
            return
        self._gloss_current.write_text(
            self._gloss_editor.toPlainText(), encoding="utf-8"
        )
        self._scan_gloss_files()
        self._update_gloss_count()
        self.glossary_changed.emit()
        logger.info("Glossary saved: %s", self._gloss_current.name)

    @Slot()
    def _on_gloss_revert(self) -> None:
        if self._gloss_current and self._gloss_current.exists():
            self._gloss_editor.setPlainText(
                self._gloss_current.read_text(encoding="utf-8")
            )
            self._update_gloss_count()

    def _update_gloss_count(self) -> None:
        text = self._gloss_editor.toPlainText()
        count = sum(
            1 for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        self._gloss_count.setText(f"{count} terms")

    @Slot()
    def _on_threshold_changed(self) -> None:
        try:
            val = int(self._thresh_edit.text())
            if val < 1:
                val = 1
            self._cfg.set("translation.rag_threshold", val)
            self._scan_gloss_files()
        except ValueError:
            pass

    # ==================================================================
    # STT TERM file management
    # ==================================================================

    def _scan_stt_files(self) -> None:
        """Scan stt_terms/ and populate the STT term file table."""
        self._stt_table.setRowCount(0)
        _STT_TERMS_DIR.mkdir(parents=True, exist_ok=True)

        total = 0
        for f in sorted(_STT_TERMS_DIR.glob("*.txt")):
            row = self._stt_table.rowCount()
            self._stt_table.insertRow(row)
            self._stt_table.setItem(row, 0, QTableWidgetItem(f.name))

            tc = self._count_terms(f)
            total += tc
            ci = QTableWidgetItem(str(tc))
            ci.setTextAlignment(Qt.AlignCenter)
            self._stt_table.setItem(row, 1, ci)

        self._stt_total_label.setText(f"Total: {total} terms")
        self._emit_stt_files()

    @Slot(int, int, int, int)
    def _on_stt_file_selected(
        self, row: int, _c: int, _pr: int, _pc: int
    ) -> None:
        if row < 0:
            self._stt_current = None
            self._stt_editor.clear()
            self._btn_stt_save.setEnabled(False)
            self._btn_stt_revert.setEnabled(False)
            self._btn_stt_delete.setEnabled(False)
            return

        name = self._stt_table.item(row, 0).text()
        path = _STT_TERMS_DIR / name
        self._stt_current = path
        self._btn_stt_delete.setEnabled(True)

        if path.exists():
            self._stt_editor.setPlainText(
                path.read_text(encoding="utf-8")
            )
            self._btn_stt_save.setEnabled(True)
            self._btn_stt_revert.setEnabled(True)
            self._update_stt_count()

    @Slot()
    def _on_stt_new(self) -> None:
        _STT_TERMS_DIR.mkdir(parents=True, exist_ok=True)
        i = 1
        while (_STT_TERMS_DIR / f"terms_{i:02d}.txt").exists():
            i += 1
        path = _STT_TERMS_DIR / f"terms_{i:02d}.txt"
        path.write_text(
            "# STT Recognition Terms\n"
            "# One term per line — biases Whisper toward these words\n"
            "#\n",
            encoding="utf-8",
        )
        self._scan_stt_files()
        for row in range(self._stt_table.rowCount()):
            if self._stt_table.item(row, 0).text() == path.name:
                self._stt_table.selectRow(row)
                break

    @Slot()
    def _on_stt_import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Term File", "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return
        src = Path(path)
        dest = _STT_TERMS_DIR / src.name
        if dest.exists():
            ret = QMessageBox.question(
                self, "File Exists",
                f"{dest.name} already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if ret != QMessageBox.Yes:
                return
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        self._scan_stt_files()

    @Slot()
    def _on_stt_delete(self) -> None:
        if not self._stt_current or not self._stt_current.exists():
            return
        ret = QMessageBox.question(
            self, "Delete Term File",
            f"Delete {self._stt_current.name}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self._stt_current.unlink()
        self._stt_current = None
        self._stt_editor.clear()
        self._scan_stt_files()

    @Slot()
    def _on_stt_save(self) -> None:
        if not self._stt_current:
            return
        self._stt_current.write_text(
            self._stt_editor.toPlainText(), encoding="utf-8"
        )
        self._scan_stt_files()
        self._update_stt_count()
        logger.info("STT terms saved: %s", self._stt_current.name)

    @Slot()
    def _on_stt_revert(self) -> None:
        if self._stt_current and self._stt_current.exists():
            self._stt_editor.setPlainText(
                self._stt_current.read_text(encoding="utf-8")
            )
            self._update_stt_count()

    def _update_stt_count(self) -> None:
        text = self._stt_editor.toPlainText()
        count = sum(
            1 for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        self._stt_count.setText(f"{count} terms")

    def _emit_stt_files(self) -> None:
        """Emit all STT term file paths."""
        paths = [
            str(f) for f in sorted(_STT_TERMS_DIR.glob("*.txt"))
            if f.is_file()
        ]
        self.term_files_changed.emit(paths)

    # ==================================================================
    # Shared
    # ==================================================================

    @staticmethod
    def _count_terms(path: Path) -> int:
        """Count non-comment, non-empty lines."""
        count = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    count += 1
        except Exception:
            pass
        return count
