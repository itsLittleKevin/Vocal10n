"""Pipeline coordinator — session lifecycle and status façade.

The heavy lifting (model loading, event wiring, inference) lives in
the per-component controllers (STTController, LLMController,
TTSController).  This module provides a thin convenience layer for:

* Starting / stopping a recording session
* Querying aggregate pipeline status
* Flushing pending buffers on session stop
* Owning the FileWriter for SRT/TXT/WAV output
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import QObject, Signal, Slot

from vocal10n.config import get_config
from vocal10n.constants import EventType, ModelStatus
from vocal10n.pipeline.events import Event, TextEvent, TranslationEvent, get_dispatcher
from vocal10n.pipeline.file_writer import FileWriter
from vocal10n.pipeline.latency import LatencyTracker
from vocal10n.state import SystemState

if TYPE_CHECKING:
    from vocal10n.stt.audio_capture import AudioCapture

logger = logging.getLogger(__name__)

# Default output directory
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_OUTPUT_DIR = _PROJECT_ROOT / "output"


class PipelineCoordinator(QObject):
    """Owns session lifecycle and file output.

    Signals:
        session_started  — emitted when a recording session begins
        session_stopped  — emitted when a session ends
    """

    session_started = Signal()
    session_stopped = Signal()

    def __init__(
        self,
        state: SystemState,
        latency: LatencyTracker,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._state = state
        self._latency = latency
        self._cfg = get_config()
        self._dispatcher = get_dispatcher()

        self._running = False
        self._session_start: Optional[float] = None
        self._wav_offset_s: float = 0.0  # audio buffer offset at session start

        # File writer (created on first session start)
        self._writer: Optional[FileWriter] = None

        # Optional reference to AudioCapture for WAV recording
        self._audio_capture: Optional[AudioCapture] = None

    # ------------------------------------------------------------------
    # External wiring
    # ------------------------------------------------------------------

    def set_audio_capture(self, capture: AudioCapture) -> None:
        """Set a reference to AudioCapture for WAV recording."""
        self._audio_capture = capture

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @Slot()
    def start_session(self) -> None:
        """Begin a recording / translation session."""
        if self._running:
            logger.info("Session already running")
            return

        self._session_start = time.time()
        self._running = True

        # Create file writer if any output is enabled
        if self._any_output_enabled():
            output_dir = Path(
                self._cfg.get("output.directory", str(_OUTPUT_DIR))
            )
            self._writer = FileWriter(output_dir)
            self._writer.start_session()

            # Subscribe writer to pipeline events
            self._dispatcher.subscribe(
                EventType.STT_CONFIRMED, self._on_stt_for_writer
            )
            self._dispatcher.subscribe(
                EventType.TRANSLATION_CONFIRMED, self._on_translation_for_writer
            )
            logger.info("File writer started — output dir: %s", output_dir)

        self._dispatcher.publish(Event(
            type=EventType.PIPELINE_READY,
            data={"session_start": self._session_start},
            source="coordinator",
        ))

        self.session_started.emit()
        logger.info("Pipeline session started")

        # Record current audio buffer position for WAV (only capture from now)
        if self._audio_capture:
            self._wav_offset_s = self._audio_capture.duration

    @Slot()
    def stop_session(self) -> None:
        """End the current session and flush outputs."""
        if not self._running:
            return

        self._running = False

        # Unsubscribe writer from events
        self._dispatcher.unsubscribe(
            EventType.STT_CONFIRMED, self._on_stt_for_writer
        )
        self._dispatcher.unsubscribe(
            EventType.TRANSLATION_CONFIRMED, self._on_translation_for_writer
        )

        # Save WAV if enabled and audio capture is available
        if (
            self._writer
            and self._cfg.get("output.save_wav", False)
            and self._audio_capture
        ):
            buf = self._audio_capture.get_from_offset(self._wav_offset_s)
            if len(buf) > 0:
                self._writer.write_wav(buf, self._audio_capture.sample_rate)
                logger.info(
                    "WAV audio queued — %.1fs at %d Hz",
                    len(buf) / self._audio_capture.sample_rate,
                    self._audio_capture.sample_rate,
                )

        # Finalize file outputs
        if self._writer:
            self._writer.end_session()
            self._writer = None

        self.session_stopped.emit()
        duration = time.time() - (self._session_start or time.time())
        logger.info("Pipeline session stopped (%.1fs)", duration)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return aggregate pipeline status."""
        return {
            "running": self._running,
            "stt": self._state.stt_status.value,
            "llm": self._state.llm_status.value,
            "tts": self._state.tts_status.value,
            "session_duration_s": (
                time.time() - self._session_start if self._session_start else 0
            ),
        }

    # ------------------------------------------------------------------
    # Output config helpers
    # ------------------------------------------------------------------

    def _any_output_enabled(self) -> bool:
        return any(
            self._cfg.get(k, False)
            for k in (
                "output.save_source_srt",
                "output.save_target_srt",
                "output.save_source_txt",
                "output.save_target_txt",
                "output.save_wav",
            )
        )

    # ------------------------------------------------------------------
    # Event → FileWriter bridges
    # ------------------------------------------------------------------

    def _on_stt_for_writer(self, event: TextEvent) -> None:
        if not self._writer:
            return
        if self._cfg.get("output.save_source_txt", False):
            self._writer.write_source_line(event.text)
        if self._cfg.get("output.save_source_srt", False):
            self._writer.write_source_srt_entry(event.text)

    def _on_translation_for_writer(self, event: TranslationEvent) -> None:
        if not self._writer:
            return
        if self._cfg.get("output.save_target_srt", False):
            self._writer.write_target_srt_entry(event.translated_text)
        if self._cfg.get("output.save_target_txt", False):
            self._writer.write_target_line(event.translated_text)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self.stop_session()
