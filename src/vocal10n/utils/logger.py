"""Logging configuration for Vocal10n."""

import logging
import sys
from typing import Optional

from vocal10n.config import get_config

_initialised = False


def setup_logging(level: Optional[str] = None) -> None:
    """Initialise root logger. Safe to call multiple times (idempotent)."""
    global _initialised
    if _initialised:
        return
    _initialised = True

    if level is None:
        level = get_config().get("logging.level", "INFO")

    numeric = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    root = logging.getLogger()
    root.setLevel(numeric)
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "httpcore", "httpx", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
