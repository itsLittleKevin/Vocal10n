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

    def __init__(self, client: GPTSoVITSClient, max_size: int = 10,
                 max_pending: int = 3):
        self.client = client
        self.max_size = max_size
        self.max_pending = max_pending  # Drop oldest when queue exceeds this
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._processing = False
        self._audio_callback: Optional[Callable[[dict[str, Any]], None]] = None
        self._stream_callback: Optional[Callable] = None

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

        Prunes oldest items when queue exceeds ``max_pending`` to keep
        the pipeline responsive during fast speech.

        Returns:
            True if queued, False if queue full.
        """
        # Prune: if queue >= max_pending, drop oldest items to make room
        while self._queue.qsize() >= self.max_pending:
            try:
                dropped = self._queue.get_nowait()
                logger.info("TTS queue pruned stale item: '%s...'",
                            dropped.get("text", "")[:30])
            except queue.Empty:
                break

        try:
            self._queue.put_nowait({
                "text": text,
                "language": language,
                "timestamp": time.time(),
                "enqueue_time": time.time(),
            })
            return True
        except queue.Full:
            logger.warning("TTS queue full after pruning, dropping request")
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

            # Drop stale requests (older than 8s) to prevent memory waste
            age = time.time() - request["timestamp"]
            if age > 8.0:
                logger.warning("Dropping stale TTS request (age %.1fs): '%s...'",
                               age, request["text"][:30])
                continue

            logger.info("TTS processing: '%s...' (queue: %d)", request["text"][:30], self._queue.qsize())

            self._processing = True
            t0 = time.time()
            try:
                result = self.client.synthesize(request["text"], request["language"], streaming=True)
            except Exception as e:
                logger.exception("TTS synthesize exception: %s", e)
                self._processing = False
                continue
            dt = time.time() - t0

            status = result.get("status", "unknown")
            if status == "success" and self._audio_callback:
                # Override request_start with enqueue_time so TTFA measures
                # text arrival → first audio chunk (the real user-perceived latency)
                enqueue_time = request.get("enqueue_time")
                if enqueue_time:
                    result["request_start"] = enqueue_time
                if result.get("streaming"):
                    logger.info("TTS streaming started in %.1fms (HTTP response)", result.get("latency_ms", 0))
                    self._audio_callback(result)
                else:
                    audio_size = len(result.get("audio_data", b""))
                    logger.info("TTS synthesis completed in %.1fs, duration=%.1fms, audio=%d bytes", 
                                dt, result.get("duration_ms", 0), audio_size)
                    self._audio_callback(result)
            elif status == "error":
                logger.error("TTS synthesis failed (%.1fs): %s", dt, result.get("message"))
            else:
                logger.warning("TTS unexpected result status: %s, keys: %s", status, list(result.keys()))
            self._processing = False
