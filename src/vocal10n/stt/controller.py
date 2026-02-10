"""STT controller — orchestrates audio capture, engine, filters, and worker.

Owns the lifecycle: load → start → stop → unload.
Bridges between UI controls (signals/slots) and the STT pipeline.
"""

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Slot

from vocal10n.config import get_config
from vocal10n.constants import EventType, ModelStatus
from vocal10n.pipeline.events import get_dispatcher
from vocal10n.pipeline.latency import LatencyTracker
from vocal10n.state import SystemState
from vocal10n.stt.audio_capture import AudioCapture
from vocal10n.stt.engine import STTEngine
from vocal10n.stt.filters import STTFilters
from vocal10n.stt.transcript import TranscriptManager
from vocal10n.stt.worker import STTWorker

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/vocal10n/stt → project root


class STTController(QObject):
    """High-level STT lifecycle manager.

    Connects:
    - ``state.stt_enabled_changed`` → start / stop capture & worker
    - ``ModelSelector.load_requested`` → load engine
    - ``ModelSelector.unload_requested`` → unload engine
    - ``STT_CONFIRMED`` / ``STT_PENDING`` events → ``state.current_stt_text``
    """

    def __init__(self, state: SystemState, latency: LatencyTracker, parent=None):
        super().__init__(parent)
        self._state = state
        self._latency = latency

        cfg = get_config()

        # Sub-components
        self._capture = AudioCapture()
        self._engine = STTEngine()
        self._filters = STTFilters()
        self._transcript = TranscriptManager(get_dispatcher(), latency, self._filters)
        self._worker: STTWorker | None = None

        # Load filter + phonetic files if they exist
        filters_path = _PROJECT_ROOT / "config" / "filters.txt"
        if filters_path.exists():
            self._filters.load_filter_file(filters_path)

        context_path = _PROJECT_ROOT / "config" / "context_gaming.txt"
        if context_path.exists():
            self._filters.build_phonetic_index(context_path)

        # Term files → phonetic index + initial_prompt for Whisper
        self._term_files: list[str] = []
        self._initial_prompt: str = ""

        # Wire state signals
        self._state.stt_enabled_changed.connect(self._on_enabled)

        # Wire pipeline events → state (for UI display)
        dispatcher = get_dispatcher()
        dispatcher.subscribe(EventType.STT_CONFIRMED, self._on_stt_confirmed)
        dispatcher.subscribe(EventType.STT_PENDING, self._on_stt_pending)

    # ------------------------------------------------------------------
    # Term file management
    # ------------------------------------------------------------------

    @Slot(list)
    def update_term_files(self, paths: list[str]) -> None:
        """Rebuild phonetic index and initial_prompt from term files."""
        self._term_files = list(paths)

        # Rebuild phonetic index from all term files
        all_terms: list[str] = []
        for p in paths:
            pp = Path(p)
            if pp.exists():
                self._filters.build_phonetic_index(pp)
                try:
                    for line in pp.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line:
                            all_terms.append(line)
                except Exception:
                    pass

        # Build initial_prompt for Whisper (helps bias recognition)
        cfg = get_config()
        capacity = cfg.get("stt.initial_prompt_capacity", 200)
        if all_terms:
            self._initial_prompt = "，".join(all_terms[:capacity])
        else:
            self._initial_prompt = ""
        logger.info("Term files updated: %d files, %d terms, %d in prompt, prompt=%d chars",
                     len(paths), len(all_terms), min(len(all_terms), capacity), len(self._initial_prompt))

    # ------------------------------------------------------------------
    # Model lifecycle (called from STT tab)
    # ------------------------------------------------------------------

    @Slot(str)
    def load_model(self, model_name: str) -> None:
        """Load the Whisper model (runs in main thread — fast for cached)."""
        if self._engine.is_loaded:
            return
        self._state.stt_status = ModelStatus.LOADING
        get_config().set("stt.model_size", model_name)
        try:
            self._engine.load()
            self._state.stt_status = ModelStatus.LOADED
            logger.info("STT model loaded: %s", model_name)
        except Exception as e:
            self._state.stt_status = ModelStatus.ERROR
            logger.exception("Failed to load STT model: %s", e)

    @Slot()
    def unload_model(self) -> None:
        """Stop capture + worker, then unload Whisper model."""
        self._stop_pipeline()
        self._state.stt_status = ModelStatus.UNLOADING
        try:
            self._engine.unload()
            self._state.stt_status = ModelStatus.UNLOADED
        except Exception as e:
            self._state.stt_status = ModelStatus.ERROR
            logger.exception("Failed to unload STT model: %s", e)

    # ------------------------------------------------------------------
    # Pipeline start / stop
    # ------------------------------------------------------------------

    def _start_pipeline(self) -> None:
        if not self._engine.is_loaded:
            logger.warning("Cannot start STT — model not loaded")
            return
        if self._worker is not None:
            return  # already running

        self._transcript.start_session()
        self._capture.start()
        self._worker = STTWorker(self._capture, self._engine, self._transcript,
                                 initial_prompt=self._initial_prompt)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.begin()

        get_dispatcher().publish_text(EventType.STT_STARTED, "", source="stt")
        logger.info("STT pipeline started")

    def _stop_pipeline(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        self._capture.stop()
        self._state.current_stt_text = ""
        get_dispatcher().publish_text(EventType.STT_STOPPED, "", source="stt")
        logger.info("STT pipeline stopped")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_enabled(self, enabled: bool) -> None:
        if enabled:
            self._start_pipeline()
        else:
            self._stop_pipeline()

    @Slot(str)
    def _on_worker_error(self, msg: str) -> None:
        logger.error("STT worker error: %s", msg)
        self._state.stt_status = ModelStatus.ERROR
        self._stop_pipeline()

    # ------------------------------------------------------------------
    # Event handlers (pipeline events → state)
    # ------------------------------------------------------------------

    def _on_stt_confirmed(self, event) -> None:
        self._state.current_stt_text = event.text
        # Update accumulated panel with full transcript (with punctuation)
        self._state.accumulated_stt_text = self._transcript.full_transcript

    def _on_stt_pending(self, event) -> None:
        if event.text:
            self._state.current_stt_text = event.text

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown — stop everything."""
        self._stop_pipeline()
        if self._engine.is_loaded:
            self._engine.unload()
