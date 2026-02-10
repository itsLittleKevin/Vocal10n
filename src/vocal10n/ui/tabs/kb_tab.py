"""Knowledge Base tab for Section B.

Manages glossary files in ``knowledge_base/`` that feed into the
STT corrector and translation prompt.  Also hosts STT recognition
context (term files for phonetic correction + Whisper initial_prompt).
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
from vocal10n.ui.widgets.term_file_list import TermFileList

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_KB_DIR = _PROJECT_ROOT / "knowledge_base"


class KnowledgeBaseTab(QWidget):
    """Manages glossary files, RAG settings, and STT recognition context."""

    glossary_changed = Signal()  # emitted when glossaries are modified
    term_files_changed = Signal(list)  # emits list[str] of STT term file paths

    def __init__(self, state: SystemState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()
        self._current_file: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Header ────────────────────────────────────────────────────
        header = QLabel("Knowledge Base")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #e0e0e0;")
        root.addWidget(header)

        desc = QLabel(
            "Glossary files improve STT correction and translation accuracy. "
            "Terms are fuzzy-matched against speech output and injected as "
            "hints into the translation prompt."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #8892a4; font-size: 12px; margin-bottom: 4px;")
        root.addWidget(desc)

        # ── Glossary Files ────────────────────────────────────────────
        files_box = QGroupBox("Glossary Files")
        fl = QVBoxLayout(files_box)

        # File list table
        self._file_table = QTableWidget(0, 3)
        self._file_table.setHorizontalHeaderLabels(["File", "Terms", "Status"])
        self._file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._file_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._file_table.setSelectionMode(QTableWidget.SingleSelection)
        self._file_table.setMaximumHeight(150)
        self._file_table.verticalHeader().setVisible(False)
        self._file_table.currentCellChanged.connect(self._on_file_selected)
        fl.addWidget(self._file_table)

        # File action buttons
        btn_row = QHBoxLayout()
        self._btn_new = QPushButton("New Glossary")
        self._btn_new.setFixedWidth(110)
        self._btn_new.clicked.connect(self._on_new_glossary)
        btn_row.addWidget(self._btn_new)

        self._btn_import = QPushButton("Import File")
        self._btn_import.setFixedWidth(100)
        self._btn_import.clicked.connect(self._on_import_file)
        btn_row.addWidget(self._btn_import)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.setFixedWidth(70)
        self._btn_delete.setEnabled(False)
        self._btn_delete.clicked.connect(self._on_delete_file)
        btn_row.addWidget(self._btn_delete)

        btn_row.addStretch()

        # RAG status
        self._rag_label = QLabel("RAG: inactive")
        self._rag_label.setStyleSheet("color: #8892a4; font-size: 11px;")
        btn_row.addWidget(self._rag_label)
        fl.addLayout(btn_row)

        root.addWidget(files_box)

        # ── Term Editor ───────────────────────────────────────────────
        editor_box = QGroupBox("Term Editor")
        el = QVBoxLayout(editor_box)

        format_info = QLabel(
            "Format: <b>source_term|translation</b> (one per line). "
            "Lines starting with <b>#</b> are comments."
        )
        format_info.setWordWrap(True)
        format_info.setStyleSheet("color: #8892a4; font-size: 11px;")
        el.addWidget(format_info)

        self._editor = QTextEdit()
        self._editor.setPlaceholderText(
            "# Example glossary\n"
            "# source_term|preferred_translation\n"
            "科技小院|Science and Technology Backyard\n"
            "中国农业大学|China Agricultural University"
        )
        self._editor.setMinimumHeight(200)
        self._editor.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;")
        el.addWidget(self._editor)

        # Editor buttons
        ed_btn_row = QHBoxLayout()
        self._btn_save = QPushButton("Save")
        self._btn_save.setProperty("accent", True)
        self._btn_save.setFixedWidth(80)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        ed_btn_row.addWidget(self._btn_save)

        self._btn_revert = QPushButton("Revert")
        self._btn_revert.setFixedWidth(80)
        self._btn_revert.setEnabled(False)
        self._btn_revert.clicked.connect(self._on_revert)
        ed_btn_row.addWidget(self._btn_revert)

        ed_btn_row.addStretch()

        self._term_count_label = QLabel("")
        self._term_count_label.setStyleSheet("color: #8892a4; font-size: 11px;")
        ed_btn_row.addWidget(self._term_count_label)
        el.addLayout(ed_btn_row)

        root.addWidget(editor_box)

        # ── Quick Add ─────────────────────────────────────────────────
        quick_box = QGroupBox("Quick Add Term")
        ql = QHBoxLayout(quick_box)
        ql.addWidget(QLabel("Term:"))
        self._quick_term = QLineEdit()
        self._quick_term.setPlaceholderText("source term")
        ql.addWidget(self._quick_term, stretch=2)
        ql.addWidget(QLabel("Translation:"))
        self._quick_trans = QLineEdit()
        self._quick_trans.setPlaceholderText("preferred translation")
        ql.addWidget(self._quick_trans, stretch=2)
        self._btn_quick_add = QPushButton("Add")
        self._btn_quick_add.setFixedWidth(60)
        self._btn_quick_add.clicked.connect(self._on_quick_add)
        ql.addWidget(self._btn_quick_add)
        root.addWidget(quick_box)

        # ── RAG Settings ─────────────────────────────────────────────
        rag_box = QGroupBox("Vector Retrieval (RAG)")
        rl = QVBoxLayout(rag_box)

        rag_desc = QLabel(
            "When glossary exceeds the threshold below, the system automatically "
            "switches from pinyin matching to vector-based retrieval (FAISS + MiniLM). "
            "Requires: pip install sentence-transformers faiss-cpu"
        )
        rag_desc.setWordWrap(True)
        rag_desc.setStyleSheet("color: #8892a4; font-size: 11px;")
        rl.addWidget(rag_desc)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("RAG threshold:"))
        self._thresh_edit = QLineEdit()
        self._thresh_edit.setFixedWidth(60)
        self._thresh_edit.setText(str(self._cfg.get("translation.rag_threshold", 100)))
        self._thresh_edit.editingFinished.connect(self._on_threshold_changed)
        thresh_row.addWidget(self._thresh_edit)
        thresh_row.addWidget(QLabel("terms"))
        thresh_row.addStretch()
        rl.addLayout(thresh_row)

        root.addWidget(rag_box)

        # ── STT Recognition Context ──────────────────────────────────
        ctx_box = QGroupBox("STT Recognition Context (Term Files)")
        cl = QVBoxLayout(ctx_box)
        cl.setSpacing(4)

        ctx_info = QLabel(
            "Term files improve STT recognition accuracy. Terms are used for:\n"
            "• Phonetic correction (fuzzy pinyin matching against STT output)\n"
            "• Whisper initial_prompt context (biases Whisper toward these terms)\n"
            "Format: one term per line, plain text, UTF-8."
        )
        ctx_info.setWordWrap(True)
        ctx_info.setStyleSheet("color: #8892a4; font-size: 11px;")
        cl.addWidget(ctx_info)

        self._term_list = TermFileList(title="Loaded Term Files")
        self._term_list.files_changed.connect(self._on_term_files_changed)
        cl.addWidget(self._term_list)

        # Pre-load existing term files
        for default_file in ("config/context_gaming.txt",):
            fp = _PROJECT_ROOT / default_file
            if fp.exists():
                self._term_list.add_file(str(fp))

        # Capacity + status row
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

        self._term_status = QLabel("Loaded: 0 terms")
        self._term_status.setMinimumWidth(120)
        self._term_status.setStyleSheet("color: #8892a4; font-size: 11px;")
        cap_row.addWidget(self._term_status)
        cl.addLayout(cap_row)

        root.addWidget(ctx_box)

        root.addStretch()

        # Initial scan
        self._scan_files()

    # ------------------------------------------------------------------
    # File management
    # ------------------------------------------------------------------

    def _scan_files(self) -> None:
        """Scan knowledge_base/ and populate the file table."""
        self._file_table.setRowCount(0)
        _KB_DIR.mkdir(parents=True, exist_ok=True)

        total_terms = 0
        for f in sorted(_KB_DIR.glob("*.txt")):
            row = self._file_table.rowCount()
            self._file_table.insertRow(row)

            self._file_table.setItem(row, 0, QTableWidgetItem(f.name))

            term_count = self._count_terms(f)
            total_terms += term_count
            count_item = QTableWidgetItem(str(term_count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self._file_table.setItem(row, 1, count_item)

            status_item = QTableWidgetItem("Loaded")
            status_item.setTextAlignment(Qt.AlignCenter)
            self._file_table.setItem(row, 2, status_item)

        # Update RAG status
        threshold = self._cfg.get("translation.rag_threshold", 100)
        if total_terms >= threshold:
            self._rag_label.setText(f"RAG: active ({total_terms} terms ≥ {threshold} threshold)")
            self._rag_label.setStyleSheet("color: #0f9b8e; font-size: 11px;")
        else:
            self._rag_label.setText(f"RAG: inactive ({total_terms} terms < {threshold} threshold)")
            self._rag_label.setStyleSheet("color: #8892a4; font-size: 11px;")

    @staticmethod
    def _count_terms(path: Path) -> int:
        """Count non-comment, non-empty lines in a glossary file."""
        count = 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    count += 1
        except Exception:
            pass
        return count

    @Slot(int, int, int, int)
    def _on_file_selected(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row < 0:
            self._current_file = None
            self._editor.clear()
            self._btn_save.setEnabled(False)
            self._btn_revert.setEnabled(False)
            self._btn_delete.setEnabled(False)
            return

        name = self._file_table.item(row, 0).text()
        path = _KB_DIR / name
        self._current_file = path
        self._btn_delete.setEnabled(True)

        if path.exists():
            self._editor.setPlainText(path.read_text(encoding="utf-8"))
            self._btn_save.setEnabled(True)
            self._btn_revert.setEnabled(True)
            self._update_term_count()

    @Slot()
    def _on_new_glossary(self) -> None:
        """Create a new empty glossary file."""
        _KB_DIR.mkdir(parents=True, exist_ok=True)
        # Find next available name
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
        self._scan_files()
        # Select the new file
        for row in range(self._file_table.rowCount()):
            if self._file_table.item(row, 0).text() == path.name:
                self._file_table.selectRow(row)
                break
        self.glossary_changed.emit()

    @Slot()
    def _on_import_file(self) -> None:
        """Import an external glossary text file."""
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
        self._scan_files()
        self.glossary_changed.emit()

    @Slot()
    def _on_delete_file(self) -> None:
        """Delete the selected glossary file."""
        if not self._current_file or not self._current_file.exists():
            return
        ret = QMessageBox.question(
            self, "Delete Glossary",
            f"Delete {self._current_file.name}?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if ret != QMessageBox.Yes:
            return
        self._current_file.unlink()
        self._current_file = None
        self._editor.clear()
        self._scan_files()
        self.glossary_changed.emit()

    @Slot()
    def _on_save(self) -> None:
        """Save the editor content to the current file."""
        if not self._current_file:
            return
        self._current_file.write_text(
            self._editor.toPlainText(), encoding="utf-8",
        )
        self._scan_files()
        self._update_term_count()
        self.glossary_changed.emit()
        logger.info("Glossary saved: %s", self._current_file.name)

    @Slot()
    def _on_revert(self) -> None:
        """Reload file content, discarding editor changes."""
        if self._current_file and self._current_file.exists():
            self._editor.setPlainText(
                self._current_file.read_text(encoding="utf-8"),
            )
            self._update_term_count()

    @Slot()
    def _on_quick_add(self) -> None:
        """Add a term to the currently selected glossary file."""
        term = self._quick_term.text().strip()
        if not term:
            return
        trans = self._quick_trans.text().strip()
        line = f"{term}|{trans}" if trans else term

        if self._current_file and self._current_file.exists():
            # Append to file and update editor
            content = self._current_file.read_text(encoding="utf-8")
            if not content.endswith("\n"):
                content += "\n"
            content += line + "\n"
            self._current_file.write_text(content, encoding="utf-8")
            self._editor.setPlainText(content)
        else:
            # Append to editor only
            cursor = self._editor.textCursor()
            cursor.movePosition(cursor.End)
            text = self._editor.toPlainText()
            if text and not text.endswith("\n"):
                cursor.insertText("\n")
            cursor.insertText(line + "\n")

        self._quick_term.clear()
        self._quick_trans.clear()
        self._scan_files()
        self._update_term_count()
        self.glossary_changed.emit()

    @Slot()
    def _on_threshold_changed(self) -> None:
        """Update RAG threshold from UI."""
        try:
            val = int(self._thresh_edit.text())
            if val < 1:
                val = 1
            self._cfg.set("translation.rag_threshold", val)
            self._scan_files()  # Refresh RAG status display
        except ValueError:
            pass

    def _update_term_count(self) -> None:
        """Update the term count label from editor content."""
        text = self._editor.toPlainText()
        count = sum(
            1 for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        self._term_count_label.setText(f"{count} terms")

    @Slot(list)
    def _on_term_files_changed(self, paths: list[str]) -> None:
        """Handle STT term file list changes."""
        total_terms = 0
        for p in paths:
            try:
                pp = Path(p)
                if pp.exists():
                    total_terms += sum(
                        1 for line in pp.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )
            except Exception:
                pass
        self._term_status.setText(f"Loaded: {total_terms} terms")
        self.term_files_changed.emit(paths)
