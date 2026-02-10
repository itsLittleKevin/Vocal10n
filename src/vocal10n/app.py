"""Vocal10n application entry point.

Creates the QApplication, loads the theme, instantiates global state,
and shows the main window.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from vocal10n.state import SystemState
from vocal10n.ui.main_window import MainWindow
from vocal10n.utils.logger import setup_logging


_THEME_PATH = Path(__file__).resolve().parent / "ui" / "styles" / "theme.qss"


def main() -> int:
    setup_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Vocal10n")
    app.setOrganizationName("Vocal10n")

    # Load QSS theme
    if _THEME_PATH.exists():
        app.setStyleSheet(_THEME_PATH.read_text(encoding="utf-8"))

    state = SystemState()
    window = MainWindow(state)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
