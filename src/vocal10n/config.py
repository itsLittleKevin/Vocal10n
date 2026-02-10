"""YAML configuration loader/saver for Vocal10n."""

import copy
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class Config:
    """Thread-safe, nested-key configuration backed by YAML files."""

    def __init__(self, path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._path = Path(path) if path else _DEFAULT_CONFIG_PATH.resolve()
        self.load()

    # -- I/O ----------------------------------------------------------------

    def load(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path else self._path
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            with self._lock:
                self._data = raw
                self._path = p

    def save(self, path: Optional[Path] = None) -> None:
        p = Path(path) if path else self._path
        p.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            snapshot = copy.deepcopy(self._data)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # -- Access -------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value using dot-separated keys, e.g. ``stt.model_size``."""
        with self._lock:
            node = self._data
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    return default
            return node

    def set(self, key: str, value: Any) -> None:
        """Set a value using dot-separated keys."""
        with self._lock:
            parts = key.split(".")
            node = self._data
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value

    def section(self, name: str) -> Dict[str, Any]:
        """Return a *copy* of a top-level section dict."""
        with self._lock:
            return copy.deepcopy(self._data.get(name, {}))

    @property
    def data(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_config: Optional[Config] = None


def get_config(path: Optional[Path] = None) -> Config:
    global _config
    if _config is None:
        _config = Config(path)
    return _config
