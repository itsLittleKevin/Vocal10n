"""Helper utilities for QComboBox styling."""

from PySide6.QtWidgets import QComboBox
from PySide6.QtGui import QPainter, QFont
from PySide6.QtCore import Qt, QRect


class ArrowComboBox(QComboBox):
    """QComboBox with a visible Unicode down arrow (▼) in dropdown button."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(28)

    def paintEvent(self, event):
        """Override paint to draw Unicode arrow in dropdown area."""
        # First call parent paint
        super().paintEvent(event)
        
        # Draw the down arrow (▼) in the dropdown button area
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get dropdown button rect (right side of combo box)
        dropdown_width = 40
        dropdown_rect = QRect(
            self.width() - dropdown_width,
            0,
            dropdown_width,
            self.height()
        )
        
        # Draw down arrow symbol
        font = self.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(self.palette().buttonText().color())
        painter.drawText(
            dropdown_rect,
            Qt.AlignCenter,
            "▼"
        )


def style_combo_with_arrow(combo: QComboBox) -> None:
    """Enhance any QComboBox with better spacing and visual indicator.
    
    For standard QComboBox, this just improves styling. For better results,
    use ArrowComboBox instead.
    """
    combo.setMinimumHeight(28)
    combo.setStyleSheet(
        combo.styleSheet() + """
        QComboBox {
            padding-right: 40px;
        }
        QComboBox::drop-down {
            width: 40px;
        }
        """
    )

