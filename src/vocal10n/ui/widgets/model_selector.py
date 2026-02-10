"""Model selector widget: dropdown + load / unload buttons + status indicator."""

from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from vocal10n.constants import ModelStatus

# Status colour map
_STATUS_COLOURS = {
    ModelStatus.UNLOADED:  "#8892a4",
    ModelStatus.LOADING:   "#f0c030",
    ModelStatus.LOADED:    "#0f9b8e",
    ModelStatus.UNLOADING: "#f0c030",
    ModelStatus.ERROR:     "#e94560",
}


class ModelSelector(QWidget):
    """Dropdown for model selection with Load / Unload controls.

    Signals
    -------
    load_requested(str)
        Emitted when the user clicks **Load**, carrying the selected item text.
    unload_requested()
        Emitted when the user clicks **Unload**.
    """

    load_requested = Signal(str)
    unload_requested = Signal()

    def __init__(
        self,
        label: str = "Model",
        items: Optional[list[str]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._status = ModelStatus.UNLOADED

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # Label
        self._label = QLabel(label)
        self._label.setStyleSheet("font-weight: bold;")
        root.addWidget(self._label)

        # Row: combo + status dot + buttons
        row = QHBoxLayout()
        row.setSpacing(6)

        self._combo = QComboBox()
        if items:
            self._combo.addItems(items)
        self._combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        row.addWidget(self._combo, stretch=1)

        # Status indicator (coloured dot via unicode)
        self._status_label = QLabel("●")
        self._status_label.setFixedWidth(20)
        self._status_label.setAlignment(Qt.AlignCenter)
        self._apply_status_style()
        row.addWidget(self._status_label)

        self._load_btn = QPushButton("Load")
        self._load_btn.setProperty("accent", True)
        self._load_btn.setMinimumWidth(70)
        self._load_btn.setFixedHeight(28)
        self._load_btn.clicked.connect(self._on_load)
        row.addWidget(self._load_btn)

        self._unload_btn = QPushButton("Unload")
        self._unload_btn.setMinimumWidth(70)
        self._unload_btn.setFixedHeight(28)
        self._unload_btn.setEnabled(False)
        self._unload_btn.clicked.connect(self._on_unload)
        row.addWidget(self._unload_btn)

        root.addLayout(row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def selected(self) -> str:
        return self._combo.currentText()

    def set_items(self, items: list[str]) -> None:
        self._combo.clear()
        self._combo.addItems(items)

    @Slot(ModelStatus)
    def set_status(self, status: ModelStatus) -> None:
        self._status = status
        self._apply_status_style()
        is_idle = status in (ModelStatus.UNLOADED, ModelStatus.ERROR)
        is_loaded = status == ModelStatus.LOADED
        self._load_btn.setEnabled(is_idle)
        self._unload_btn.setEnabled(is_loaded)
        self._combo.setEnabled(is_idle)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_status_style(self) -> None:
        colour = _STATUS_COLOURS.get(self._status, "#8892a4")
        self._status_label.setStyleSheet(f"color: {colour}; font-size: 16px;")
        self._status_label.setToolTip(self._status.value)

    def _on_load(self) -> None:
        self.load_requested.emit(self._combo.currentText())

    def _on_unload(self) -> None:
        self.unload_requested.emit()
