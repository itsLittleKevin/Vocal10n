"""Latency tracker for the Vocal10n pipeline.

Tracks current, 5-second rolling average, and overall average latency
for STT, Translation, TTS, and Total end-to-end.

Ported from prebuild's ``latency_tracker.py``.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from PySide6.QtCore import QObject, Signal


@dataclass
class LatencyStats:
    current_ms: float = 0.0
    avg_5s_ms: float = 0.0
    avg_overall_ms: float = 0.0
    count: int = 0


class LatencyTracker(QObject):
    """Pipeline latency tracker with Qt signal for UI updates."""

    stats_updated = Signal()  # emitted after any record_*() call

    def __init__(self, window_seconds: float = 5.0, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._window = window_seconds
        self._lock = threading.Lock()

        # Per-component storage: deque of (timestamp, ms) + running totals
        self._components = ("stt", "translation", "tts", "total")
        self._samples: Dict[str, Deque[tuple]] = {c: deque() for c in self._components}
        self._totals: Dict[str, float] = {c: 0.0 for c in self._components}
        self._counts: Dict[str, int] = {c: 0 for c in self._components}
        self._current: Dict[str, float] = {c: 0.0 for c in self._components}

        # For total latency bookmarks
        self._speech_start: Optional[float] = None

    # -- Recording ----------------------------------------------------------

    def _record(self, component: str, latency_ms: float) -> None:
        now = time.time()
        with self._lock:
            self._current[component] = latency_ms
            self._samples[component].append((now, latency_ms))
            self._totals[component] += latency_ms
            self._counts[component] += 1
            self._prune(component, now)
        self.stats_updated.emit()

    def record_stt(self, latency_ms: float) -> None:
        self._record("stt", latency_ms)

    def record_translation(self, latency_ms: float) -> None:
        self._record("translation", latency_ms)

    def record_tts(self, latency_ms: float) -> None:
        self._record("tts", latency_ms)

    def record_total(self, latency_ms: float) -> None:
        self._record("total", latency_ms)

    def mark_speech_start(self) -> None:
        self._speech_start = time.time()

    def mark_audio_output(self) -> None:
        if self._speech_start is not None:
            self.record_total((time.time() - self._speech_start) * 1000)
            self._speech_start = None

    # -- Queries ------------------------------------------------------------

    def get_stats(self, component: str) -> LatencyStats:
        with self._lock:
            self._prune(component, time.time())
            samples = self._samples[component]
            count = self._counts[component]
            return LatencyStats(
                current_ms=self._current[component],
                avg_5s_ms=self._window_avg(samples),
                avg_overall_ms=self._totals[component] / count if count else 0.0,
                count=count,
            )

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        return {c: self.get_stats(c) for c in self._components}

    # -- Helpers ------------------------------------------------------------

    def _prune(self, component: str, now: float) -> None:
        cutoff = now - self._window
        dq = self._samples[component]
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    @staticmethod
    def _window_avg(samples: Deque[tuple]) -> float:
        if not samples:
            return 0.0
        return sum(s[1] for s in samples) / len(samples)

    def reset(self) -> None:
        with self._lock:
            for c in self._components:
                self._samples[c].clear()
                self._totals[c] = 0.0
                self._counts[c] = 0
                self._current[c] = 0.0
            self._speech_start = None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_tracker: Optional[LatencyTracker] = None


def get_latency_tracker() -> LatencyTracker:
    global _tracker
    if _tracker is None:
        _tracker = LatencyTracker()
    return _tracker
