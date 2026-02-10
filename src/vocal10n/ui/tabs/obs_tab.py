"""OBS subtitle styling tab for Section B.

Controls:
- Source / Target subtitle enable toggles
- Font family and size for each language line
- Color pickers for each line
- Live preview (type text, see styled result)
- Server URL display
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from vocal10n.config import get_config
from vocal10n.state import SystemState


class OBSTab(QWidget):
    """OBS overlay settings tab."""

    settings_changed = Signal()

    def __init__(self, state: SystemState, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # ── Server Info ───────────────────────────────────────────────
        server_box = QGroupBox("OBS Browser Source")
        sl = QVBoxLayout(server_box)

        host = self._cfg.get("obs.host", "127.0.0.1")
        port = int(self._cfg.get("obs.port", 5124))
        url = f"http://{host}:{port}/overlay.html"

        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("URL:"))
        self._url_edit = QLineEdit(url)
        self._url_edit.setReadOnly(True)
        url_row.addWidget(self._url_edit, stretch=1)
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(60)
        copy_btn.clicked.connect(self._copy_url)
        url_row.addWidget(copy_btn)
        sl.addLayout(url_row)

        sl.addWidget(QLabel(
            "<i>Add this URL as a Browser Source in OBS Studio.</i>"
        ))

        root.addWidget(server_box)

        # ── Source Subtitle ───────────────────────────────────────────
        src_box = QGroupBox("Source Language Subtitle")
        src_lay = QVBoxLayout(src_box)

        self._src_enable = QCheckBox("Enable source subtitle (top line)")
        self._src_enable.setChecked(
            self._cfg.get("obs.enable_source_subtitle", True)
        )
        self._src_enable.toggled.connect(
            lambda c: self._on_toggle("obs.enable_source_subtitle", c)
        )
        src_lay.addWidget(self._src_enable)

        # Font family
        font_row_src = QHBoxLayout()
        font_row_src.addWidget(QLabel("Font:"))
        self._src_font_combo = self._make_font_combo(
            self._cfg.get("obs.font_family_source", "Noto Sans SC")
        )
        self._src_font_combo.currentTextChanged.connect(
            lambda t: self._on_font("obs.font_family_source", t)
        )
        font_row_src.addWidget(self._src_font_combo, stretch=1)
        src_lay.addLayout(font_row_src)

        # Font size
        size_row_src = QHBoxLayout()
        size_row_src.addWidget(QLabel("Size:"))
        self._src_size_spin = QSpinBox()
        self._src_size_spin.setRange(12, 120)
        self._src_size_spin.setValue(
            int(self._cfg.get("obs.font_size_source", 28))
        )
        self._src_size_spin.setSuffix(" px")
        self._src_size_spin.valueChanged.connect(
            lambda v: self._on_size("obs.font_size_source", v)
        )
        size_row_src.addWidget(self._src_size_spin)
        size_row_src.addStretch()

        # Color
        size_row_src.addWidget(QLabel("Color:"))
        self._src_color_btn = QPushButton()
        self._src_color = self._cfg.get("obs.color_source", "#FFFFFF")
        self._src_color_btn.setFixedSize(36, 28)
        self._update_color_btn(self._src_color_btn, self._src_color)
        self._src_color_btn.clicked.connect(
            lambda: self._pick_color("source")
        )
        size_row_src.addWidget(self._src_color_btn)

        src_lay.addLayout(size_row_src)

        # Stroke
        stroke_row_src = QHBoxLayout()
        stroke_row_src.addWidget(QLabel("Stroke:"))
        self._src_stroke_w_spin = QSpinBox()
        self._src_stroke_w_spin.setRange(0, 20)
        self._src_stroke_w_spin.setValue(int(self._cfg.get("obs.stroke_width_source", 2)))
        self._src_stroke_w_spin.setSuffix(" px")
        self._src_stroke_w_spin.valueChanged.connect(
            lambda v: self._on_size("obs.stroke_width_source", v)
        )
        stroke_row_src.addWidget(self._src_stroke_w_spin)
        stroke_row_src.addWidget(QLabel("Color:"))
        self._src_stroke_color_btn = QPushButton()
        self._src_stroke_color = self._cfg.get("obs.stroke_color_source", "#000000")
        self._src_stroke_color_btn.setFixedSize(36, 28)
        self._update_color_btn(self._src_stroke_color_btn, self._src_stroke_color)
        self._src_stroke_color_btn.clicked.connect(lambda: self._pick_color("stroke_source"))
        stroke_row_src.addWidget(self._src_stroke_color_btn)
        stroke_row_src.addStretch()
        src_lay.addLayout(stroke_row_src)

        # Shadow
        shadow_row_src = QHBoxLayout()
        shadow_row_src.addWidget(QLabel("Shadow:"))
        self._src_shadow_color_btn = QPushButton()
        self._src_shadow_color = self._cfg.get("obs.shadow_color_source", "#000000")
        self._src_shadow_color_btn.setFixedSize(36, 28)
        self._update_color_btn(self._src_shadow_color_btn, self._src_shadow_color)
        self._src_shadow_color_btn.clicked.connect(lambda: self._pick_color("shadow_source"))
        shadow_row_src.addWidget(self._src_shadow_color_btn)
        shadow_row_src.addWidget(QLabel("X:"))
        self._src_shadow_x = QSpinBox()
        self._src_shadow_x.setRange(-50, 50)
        self._src_shadow_x.setValue(int(self._cfg.get("obs.shadow_x_source", 0)))
        self._src_shadow_x.setSuffix(" px")
        self._src_shadow_x.valueChanged.connect(lambda v: self._on_size("obs.shadow_x_source", v))
        shadow_row_src.addWidget(self._src_shadow_x)
        shadow_row_src.addWidget(QLabel("Y:"))
        self._src_shadow_y = QSpinBox()
        self._src_shadow_y.setRange(-50, 50)
        self._src_shadow_y.setValue(int(self._cfg.get("obs.shadow_y_source", 2)))
        self._src_shadow_y.setSuffix(" px")
        self._src_shadow_y.valueChanged.connect(lambda v: self._on_size("obs.shadow_y_source", v))
        shadow_row_src.addWidget(self._src_shadow_y)
        shadow_row_src.addWidget(QLabel("Blur:"))
        self._src_shadow_blur = QSpinBox()
        self._src_shadow_blur.setRange(0, 50)
        self._src_shadow_blur.setValue(int(self._cfg.get("obs.shadow_blur_source", 4)))
        self._src_shadow_blur.setSuffix(" px")
        self._src_shadow_blur.valueChanged.connect(lambda v: self._on_size("obs.shadow_blur_source", v))
        shadow_row_src.addWidget(self._src_shadow_blur)
        shadow_row_src.addWidget(QLabel("Opacity:"))
        self._src_shadow_opacity = QSpinBox()
        self._src_shadow_opacity.setRange(0, 100)
        self._src_shadow_opacity.setValue(int(self._cfg.get("obs.shadow_opacity_source", 80)))
        self._src_shadow_opacity.setSuffix(" %")
        self._src_shadow_opacity.valueChanged.connect(lambda v: self._on_size("obs.shadow_opacity_source", v))
        shadow_row_src.addWidget(self._src_shadow_opacity)
        src_lay.addLayout(shadow_row_src)

        root.addWidget(src_box)

        # ── Target Subtitle ───────────────────────────────────────────
        tgt_box = QGroupBox("Target Language Subtitle")
        tgt_lay = QVBoxLayout(tgt_box)

        self._tgt_enable = QCheckBox("Enable target subtitle (bottom line)")
        self._tgt_enable.setChecked(
            self._cfg.get("obs.enable_target_subtitle", True)
        )
        self._tgt_enable.toggled.connect(
            lambda c: self._on_toggle("obs.enable_target_subtitle", c)
        )
        tgt_lay.addWidget(self._tgt_enable)

        # Font family
        font_row_tgt = QHBoxLayout()
        font_row_tgt.addWidget(QLabel("Font:"))
        self._tgt_font_combo = self._make_font_combo(
            self._cfg.get("obs.font_family_target", "Noto Sans")
        )
        self._tgt_font_combo.currentTextChanged.connect(
            lambda t: self._on_font("obs.font_family_target", t)
        )
        font_row_tgt.addWidget(self._tgt_font_combo, stretch=1)
        tgt_lay.addLayout(font_row_tgt)

        # Font size
        size_row_tgt = QHBoxLayout()
        size_row_tgt.addWidget(QLabel("Size:"))
        self._tgt_size_spin = QSpinBox()
        self._tgt_size_spin.setRange(12, 120)
        self._tgt_size_spin.setValue(
            int(self._cfg.get("obs.font_size_target", 28))
        )
        self._tgt_size_spin.setSuffix(" px")
        self._tgt_size_spin.valueChanged.connect(
            lambda v: self._on_size("obs.font_size_target", v)
        )
        size_row_tgt.addWidget(self._tgt_size_spin)
        size_row_tgt.addStretch()

        # Color
        size_row_tgt.addWidget(QLabel("Color:"))
        self._tgt_color_btn = QPushButton()
        self._tgt_color = self._cfg.get("obs.color_target", "#FFE066")
        self._tgt_color_btn.setFixedSize(36, 28)
        self._update_color_btn(self._tgt_color_btn, self._tgt_color)
        self._tgt_color_btn.clicked.connect(
            lambda: self._pick_color("target")
        )
        size_row_tgt.addWidget(self._tgt_color_btn)

        tgt_lay.addLayout(size_row_tgt)

        # Stroke
        stroke_row_tgt = QHBoxLayout()
        stroke_row_tgt.addWidget(QLabel("Stroke:"))
        self._tgt_stroke_w_spin = QSpinBox()
        self._tgt_stroke_w_spin.setRange(0, 20)
        self._tgt_stroke_w_spin.setValue(int(self._cfg.get("obs.stroke_width_target", 2)))
        self._tgt_stroke_w_spin.setSuffix(" px")
        self._tgt_stroke_w_spin.valueChanged.connect(
            lambda v: self._on_size("obs.stroke_width_target", v)
        )
        stroke_row_tgt.addWidget(self._tgt_stroke_w_spin)
        stroke_row_tgt.addWidget(QLabel("Color:"))
        self._tgt_stroke_color_btn = QPushButton()
        self._tgt_stroke_color = self._cfg.get("obs.stroke_color_target", "#000000")
        self._tgt_stroke_color_btn.setFixedSize(36, 28)
        self._update_color_btn(self._tgt_stroke_color_btn, self._tgt_stroke_color)
        self._tgt_stroke_color_btn.clicked.connect(lambda: self._pick_color("stroke_target"))
        stroke_row_tgt.addWidget(self._tgt_stroke_color_btn)
        stroke_row_tgt.addStretch()
        tgt_lay.addLayout(stroke_row_tgt)

        # Shadow
        shadow_row_tgt = QHBoxLayout()
        shadow_row_tgt.addWidget(QLabel("Shadow:"))
        self._tgt_shadow_color_btn = QPushButton()
        self._tgt_shadow_color = self._cfg.get("obs.shadow_color_target", "#000000")
        self._tgt_shadow_color_btn.setFixedSize(36, 28)
        self._update_color_btn(self._tgt_shadow_color_btn, self._tgt_shadow_color)
        self._tgt_shadow_color_btn.clicked.connect(lambda: self._pick_color("shadow_target"))
        shadow_row_tgt.addWidget(self._tgt_shadow_color_btn)
        shadow_row_tgt.addWidget(QLabel("X:"))
        self._tgt_shadow_x = QSpinBox()
        self._tgt_shadow_x.setRange(-50, 50)
        self._tgt_shadow_x.setValue(int(self._cfg.get("obs.shadow_x_target", 0)))
        self._tgt_shadow_x.setSuffix(" px")
        self._tgt_shadow_x.valueChanged.connect(lambda v: self._on_size("obs.shadow_x_target", v))
        shadow_row_tgt.addWidget(self._tgt_shadow_x)
        shadow_row_tgt.addWidget(QLabel("Y:"))
        self._tgt_shadow_y = QSpinBox()
        self._tgt_shadow_y.setRange(-50, 50)
        self._tgt_shadow_y.setValue(int(self._cfg.get("obs.shadow_y_target", 2)))
        self._tgt_shadow_y.setSuffix(" px")
        self._tgt_shadow_y.valueChanged.connect(lambda v: self._on_size("obs.shadow_y_target", v))
        shadow_row_tgt.addWidget(self._tgt_shadow_y)
        shadow_row_tgt.addWidget(QLabel("Blur:"))
        self._tgt_shadow_blur = QSpinBox()
        self._tgt_shadow_blur.setRange(0, 50)
        self._tgt_shadow_blur.setValue(int(self._cfg.get("obs.shadow_blur_target", 4)))
        self._tgt_shadow_blur.setSuffix(" px")
        self._tgt_shadow_blur.valueChanged.connect(lambda v: self._on_size("obs.shadow_blur_target", v))
        shadow_row_tgt.addWidget(self._tgt_shadow_blur)
        shadow_row_tgt.addWidget(QLabel("Opacity:"))
        self._tgt_shadow_opacity = QSpinBox()
        self._tgt_shadow_opacity.setRange(0, 100)
        self._tgt_shadow_opacity.setValue(int(self._cfg.get("obs.shadow_opacity_target", 80)))
        self._tgt_shadow_opacity.setSuffix(" %")
        self._tgt_shadow_opacity.valueChanged.connect(lambda v: self._on_size("obs.shadow_opacity_target", v))
        shadow_row_tgt.addWidget(self._tgt_shadow_opacity)
        tgt_lay.addLayout(shadow_row_tgt)

        root.addWidget(tgt_box)

        # ── Preview ───────────────────────────────────────────────────
        preview_box = QGroupBox("Preview")
        pl = QVBoxLayout(preview_box)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Source:"))
        self._preview_src_input = QLineEdit("你好世界")
        self._preview_src_input.textChanged.connect(self._update_preview)
        input_row.addWidget(self._preview_src_input, stretch=1)
        pl.addLayout(input_row)

        input_row2 = QHBoxLayout()
        input_row2.addWidget(QLabel("Target:"))
        self._preview_tgt_input = QLineEdit("Hello World")
        self._preview_tgt_input.textChanged.connect(self._update_preview)
        input_row2.addWidget(self._preview_tgt_input, stretch=1)
        pl.addLayout(input_row2)

        # Preview display area
        self._preview_widget = QWidget()
        self._preview_widget.setMinimumHeight(100)
        self._preview_widget.setStyleSheet(
            "background: #1a1a2e; border-radius: 6px; padding: 12px;"
        )
        pv_lay = QVBoxLayout(self._preview_widget)
        pv_lay.setAlignment(Qt.AlignCenter)

        self._preview_src_label = QLabel("你好世界")
        self._preview_src_label.setAlignment(Qt.AlignCenter)
        self._preview_src_label.setWordWrap(True)
        pv_lay.addWidget(self._preview_src_label)

        self._preview_tgt_label = QLabel("Hello World")
        self._preview_tgt_label.setAlignment(Qt.AlignCenter)
        self._preview_tgt_label.setWordWrap(True)
        pv_lay.addWidget(self._preview_tgt_label)

        pl.addWidget(self._preview_widget)
        root.addWidget(preview_box)

        root.addStretch()

        # Initial preview update
        self._update_preview()

    # ------------------------------------------------------------------
    # Font combo helper
    # ------------------------------------------------------------------

    def _make_font_combo(self, current: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        families = sorted(set(QFontDatabase.families()))
        combo.addItems(families)
        idx = combo.findText(current, Qt.MatchFixedString)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setCurrentText(current)
        return combo

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_toggle(self, key: str, checked: bool) -> None:
        self._cfg.set(key, checked)
        self._update_preview()
        self.settings_changed.emit()

    def _on_font(self, key: str, family: str) -> None:
        self._cfg.set(key, family)
        self._update_preview()
        self.settings_changed.emit()

    def _on_size(self, key: str, value: int) -> None:
        self._cfg.set(key, value)
        self._update_preview()
        self.settings_changed.emit()

    def _pick_color(self, which: str) -> None:
        _COLOR_MAP = {
            "source": ("_src_color", "obs.color_source", "_src_color_btn"),
            "target": ("_tgt_color", "obs.color_target", "_tgt_color_btn"),
            "stroke_source": ("_src_stroke_color", "obs.stroke_color_source", "_src_stroke_color_btn"),
            "stroke_target": ("_tgt_stroke_color", "obs.stroke_color_target", "_tgt_stroke_color_btn"),
            "shadow_source": ("_src_shadow_color", "obs.shadow_color_source", "_src_shadow_color_btn"),
            "shadow_target": ("_tgt_shadow_color", "obs.shadow_color_target", "_tgt_shadow_color_btn"),
        }
        attr, cfg_key, btn_attr = _COLOR_MAP[which]
        initial = getattr(self, attr)

        from PySide6.QtGui import QColor
        color = QColorDialog.getColor(QColor(initial), self, f"Pick {which} color")
        if not color.isValid():
            return

        hex_color = color.name()
        setattr(self, attr, hex_color)
        self._cfg.set(cfg_key, hex_color)
        self._update_color_btn(getattr(self, btn_attr), hex_color)

        self._update_preview()
        self.settings_changed.emit()

    def _copy_url(self) -> None:
        from PySide6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(self._url_edit.text())

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _update_preview(self) -> None:
        src_text = self._preview_src_input.text()
        tgt_text = self._preview_tgt_input.text()

        src_font = self._cfg.get("obs.font_family_source", "Noto Sans SC")
        tgt_font = self._cfg.get("obs.font_family_target", "Noto Sans")
        src_size = int(self._cfg.get("obs.font_size_source", 28))
        tgt_size = int(self._cfg.get("obs.font_size_target", 28))
        src_color = self._cfg.get("obs.color_source", "#FFFFFF")
        tgt_color = self._cfg.get("obs.color_target", "#FFE066")
        src_enabled = self._cfg.get("obs.enable_source_subtitle", True)
        tgt_enabled = self._cfg.get("obs.enable_target_subtitle", True)

        # Clamp preview font size for readability in the small preview box
        preview_src_size = min(src_size, 36)
        preview_tgt_size = min(tgt_size, 36)

        src_shadow_css = self._combined_preview_shadow("source")
        tgt_shadow_css = self._combined_preview_shadow("target")

        self._preview_src_label.setText(src_text if src_enabled else "")
        self._preview_src_label.setVisible(src_enabled)
        self._preview_src_label.setStyleSheet(
            f"color: {src_color}; font-family: '{src_font}'; "
            f"font-size: {preview_src_size}px; font-weight: 600; "
            f"text-shadow: {src_shadow_css};"
        )

        self._preview_tgt_label.setText(tgt_text if tgt_enabled else "")
        self._preview_tgt_label.setVisible(tgt_enabled)
        self._preview_tgt_label.setStyleSheet(
            f"color: {tgt_color}; font-family: '{tgt_font}'; "
            f"font-size: {preview_tgt_size}px; font-weight: 600; "
            f"text-shadow: {tgt_shadow_css};"
        )

    def _combined_preview_shadow(self, which: str) -> str:
        """Build combined outline + drop-shadow for Qt preview."""
        cfg = self._cfg
        parts: list[str] = []

        # Outline via multi-directional 0-blur shadows
        stroke_w = int(cfg.get(f"obs.stroke_width_{which}", 2))
        stroke_c = cfg.get(f"obs.stroke_color_{which}", "#000000")
        if stroke_w > 0:
            for dx, dy in [
                (-1, -1), (0, -1), (1, -1),
                (-1,  0),          (1,  0),
                (-1,  1), (0,  1), (1,  1),
            ]:
                parts.append(
                    f"{dx * stroke_w}px {dy * stroke_w}px 0 {stroke_c}"
                )

        # Drop shadow
        x = int(cfg.get(f"obs.shadow_x_{which}", 0))
        y = int(cfg.get(f"obs.shadow_y_{which}", 2))
        blur = int(cfg.get(f"obs.shadow_blur_{which}", 4))
        opacity = int(cfg.get(f"obs.shadow_opacity_{which}", 80))
        color = cfg.get(f"obs.shadow_color_{which}", "#000000")
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        a = opacity / 100.0
        parts.append(f"{x}px {y}px {blur}px rgba({r},{g},{b},{a:.2f})")

        return ", ".join(parts)

    @staticmethod
    def _update_color_btn(btn: QPushButton, hex_color: str) -> None:
        btn.setStyleSheet(
            f"background-color: {hex_color}; border: 1px solid #555; border-radius: 3px;"
        )
