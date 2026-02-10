"""TTS request queue with worker thread.

Manages synthesis requests to avoid overlapping and handles backpressure.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

from vocal10n.tts.client import GPTSoVITSClient

logger = logging.getLogger(__name__)


class TTSQueue:
    """Queue for TTS requests with background worker."""

    def __init__(self, client: GPTSoVITSClient, max_size: int = 10):
        self.client = client
        self.max_size = max_size
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._processing = False
        self._audio_callback: Optional[Callable[[dict[str, Any]], None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, audio_callback: Callable[[dict[str, Any]], None]) -> None:
        """Start the background worker.

        Args:
            audio_callback: Called with synthesis result dict (contains 'audio_data').
        """
        if self._worker_thread and self._worker_thread.is_alive():
            logger.warning("TTSQueue already running")
            return

        self._audio_callback = audio_callback
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("TTS queue worker started")

    def stop(self) -> None:
        """Stop the background worker."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None
        logger.info("TTS queue worker stopped")

    # ------------------------------------------------------------------
    # Enqueue / clear
    # ------------------------------------------------------------------

    def enqueue(self, text: str, language: str = "en") -> bool:
        """Add text to synthesis queue.

        Returns:
            True if queued, False if queue full.
        """
        if self._queue.full():
            logger.warning("TTS queue full, dropping request")
            return False

        try:
            self._queue.put_nowait({
                "text": text,
                "language": language,
                "timestamp": time.time(),
            })
            return True
        except queue.Full:
            return False

    def clear(self) -> None:
        """Discard all pending requests."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def is_processing(self) -> bool:
        return self._processing

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        logger.info("TTS worker loop running")
        while not self._stop_event.is_set():
            try:
                request = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Catch-up: if severely behind, skip to latest
            if self._queue.qsize() >= 8:
                logger.warning("TTS catch-up: queue has %d items, skipping to latest", self._queue.qsize() + 1)
                while self._queue.qsize() > 0:
                    try:
                        request = self._queue.get_nowait()
                    except queue.Empty:
                        break

            # Skip stale requests (older than 30s)
            age = time.time() - request["timestamp"]
            if age > 30.0:
                logger.warning("Dropping stale TTS request (age %.1fs)", age)
                continue

            logger.info("TTS processing: '%s...' (queue: %d)", request["text"][:30], self._queue.qsize())

            self._processing = True
            t0 = time.time()
            result = self.client.synthesize(request["text"], request["language"], streaming=False)
            dt = time.time() - t0
            self._processing = False

            if result["status"] == "success" and self._audio_callback:
                logger.info("TTS synthesis completed in %.1fs, duration=%.1fms", dt, result.get("duration_ms", 0))
                self._audio_callback(result)
            elif result["status"] == "error":
                logger.error("TTS synthesis failed (%.1fs): %s", dt, result.get("message"))
