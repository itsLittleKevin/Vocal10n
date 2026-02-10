"""Parameter slider with value label, tooltip info, and reset button."""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ParamSlider(QWidget):
    """Horizontal slider with current-value display and a reset-to-default button.

    Parameters are expressed as *real* floats.  Internally the QSlider
    works with integers, so we multiply by ``1 / step``.
    """

    value_changed = Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        default: float,
        step: float = 1.0,
        suffix: str = "",
        tooltip: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._default = default
        self._step = step
        self._suffix = suffix

        self._multiplier = round(1.0 / step) if step < 1.0 else 1
        self._int_step = int(step) if step >= 1.0 else 1

        # --- layout ---
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(2)

        # Top row: label + value + reset
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        if tooltip:
            self._label.setToolTip(tooltip)
        top.addWidget(self._label)
        top.addStretch()

        self._value_label = QLabel()
        self._value_label.setStyleSheet("color: #0f9b8e;")
        top.addWidget(self._value_label)

        self._reset_btn = QPushButton("↺")
        self._reset_btn.setFixedSize(22, 22)
        self._reset_btn.setToolTip(f"Reset to {self._fmt(default)}")
        self._reset_btn.clicked.connect(self.reset)
        top.addWidget(self._reset_btn)
        root.addLayout(top)

        # Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(int(minimum * self._multiplier))
        self._slider.setMaximum(int(maximum * self._multiplier))
        self._slider.setSingleStep(self._int_step if step >= 1.0 else 1)
        self._slider.setValue(int(default * self._multiplier))
        self._slider.valueChanged.connect(self._on_slider_changed)
        root.addWidget(self._slider)

        self._update_label()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def value(self) -> float:
        return self._slider.value() / self._multiplier

    def set_value(self, v: float) -> None:
        self._slider.setValue(int(v * self._multiplier))

    def reset(self) -> None:
        self.set_value(self._default)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_slider_changed(self, _raw: int) -> None:
        self._update_label()
        self.value_changed.emit(self.value)

    def _update_label(self) -> None:
        self._value_label.setText(self._fmt(self.value))

    def _fmt(self, v: float) -> str:
        # Show integers without decimals
        if self._step >= 1.0:
            return f"{int(v)}{self._suffix}"
        decimals = max(0, len(str(self._step).rstrip("0").split(".")[-1]))
        return f"{v:.{decimals}f}{self._suffix}"
