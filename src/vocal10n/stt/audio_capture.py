"""Microphone audio capture with ring-buffer.

Uses ``sounddevice.InputStream`` to capture audio in a background callback,
buffered via a queue and flushed into a contiguous numpy array by a
processing thread.
"""

import logging
import queue
import threading

import numpy as np
import sounddevice as sd

from vocal10n.config import get_config

logger = logging.getLogger(__name__)


class AudioCapture:
    """Captures audio from the default (or configured) microphone.

    Audio is stored in a growing numpy array (float32, mono, 16 kHz).
    Callers retrieve slices via :meth:`get_recent` or :meth:`get_from_offset`.
    """

    def __init__(self):
        cfg = get_config().section("stt")
        self._sample_rate: int = cfg.get("sample_rate", 16000)
        self._channels: int = cfg.get("channels", 1)
        self._chunk_size: int = int(cfg.get("chunk_duration", 0.2) * self._sample_rate)

        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray] = queue.Queue()

        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._buffer = np.array([], dtype=np.float32)
        self._running = True

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=self._chunk_size,
            callback=self._callback,
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()
        logger.info("Audio capture started (sr=%d)", self._sample_rate)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Audio capture stopped (%.1f s buffered)", self.duration)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def duration(self) -> float:
        """Total buffered duration in seconds."""
        with self._lock:
            return len(self._buffer) / self._sample_rate

    def get_recent(self, seconds: float) -> tuple[np.ndarray, float]:
        """Return the last *seconds* of audio and the offset (in seconds)
        at which that slice starts within the full buffer."""
        with self._lock:
            n = int(seconds * self._sample_rate)
            total = len(self._buffer)
            if total <= n:
                return self._buffer.copy(), 0.0
            start = total - n
            return self._buffer[start:].copy(), start / self._sample_rate

    def get_from_offset(self, offset_s: float) -> np.ndarray:
        """Audio from *offset_s* to the end of the buffer."""
        with self._lock:
            idx = int(offset_s * self._sample_rate)
            if idx >= len(self._buffer):
                return np.array([], dtype=np.float32)
            return self._buffer[idx:].copy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _callback(self, indata: np.ndarray, _frames, _time, status) -> None:
        if status:
            logger.debug("Audio status: %s", status)
        self._queue.put(indata.copy())

    def _flush_loop(self) -> None:
        while self._running or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.1)
                flat = data.flatten()
                with self._lock:
                    self._buffer = np.concatenate([self._buffer, flat])
            except queue.Empty:
                continue
