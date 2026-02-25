"""LLM controller — orchestrates model lifecycle and translation pipeline.

Bridges between UI controls (signals/slots) and the LLM components.
Mirrors the STTController pattern.
"""

import logging
import threading
import time

from PySide6.QtCore import QObject, Slot

from vocal10n.config import get_config
from vocal10n.constants import EventType, ModelStatus
from vocal10n.llm.engine import LLMEngine
from vocal10n.llm.translator import LLMTranslator
from vocal10n.pipeline.events import get_dispatcher
from vocal10n.pipeline.latency import LatencyTracker
from vocal10n.state import SystemState

logger = logging.getLogger(__name__)


class LLMController(QObject):
    """High-level LLM lifecycle manager.

    Connects:
    - ``state.llm_enabled_changed`` → subscribe/unsubscribe STT events
    - ``TranslationTab.load_requested`` → load engine
    - ``TranslationTab.unload_requested`` → unload engine
    - ``TRANSLATION_CONFIRMED`` / ``TRANSLATION_PENDING`` → ``state``
    """

    def __init__(self, state: SystemState, latency: LatencyTracker, parent=None):
        super().__init__(parent)
        self._state = state
        self._latency = latency

        self._engine = None  # Created on load (LLMEngine or LLMApiBackend)
        self._translator: LLMTranslator | None = None

        # Wire state signals
        self._state.llm_enabled_changed.connect(self._on_enabled)

        # Wire pipeline events → state (for UI display)
        dispatcher = get_dispatcher()
        dispatcher.subscribe(EventType.TRANSLATION_CONFIRMED, self._on_translation_confirmed)
        dispatcher.subscribe(EventType.TRANSLATION_PENDING, self._on_translation_pending)

    # ------------------------------------------------------------------
    # Model lifecycle (called from Translation tab)
    # ------------------------------------------------------------------

    @Slot(str)
    def load_model(self, model_name: str) -> None:
        """Load the LLM backend (local GGUF or OpenAI-compatible API)."""
        if self._engine and self._engine.is_loaded:
            return

        cfg = get_config()
        backend = cfg.get("translation.backend", "local")

        if backend == "api":
            from vocal10n.llm.api_backend import LLMApiBackend
            self._engine = LLMApiBackend()
        else:
            self._engine = LLMEngine()

        self._state.llm_status = ModelStatus.LOADING
        try:
            self._engine.load()
            # Create translator once engine is ready
            self._translator = LLMTranslator(
                self._engine, get_dispatcher(), self._latency,
            )
            self._state.llm_status = ModelStatus.LOADED
            logger.info("LLM backend loaded: %s (%s)", backend, model_name)
        except Exception as e:
            self._engine = None
            self._state.llm_status = ModelStatus.ERROR
            logger.exception("Failed to load LLM backend: %s", e)

    @Slot()
    def unload_model(self) -> None:
        """Stop translation + unload LLM backend."""
        self._unsubscribe_stt()
        translator, self._translator = self._translator, None
        if translator:
            translator.reset()
        self._state.llm_status = ModelStatus.UNLOADING
        try:
            engine, self._engine = self._engine, None
            if engine:
                engine.unload()
            self._state.llm_status = ModelStatus.UNLOADED
        except Exception as e:
            self._state.llm_status = ModelStatus.ERROR
            logger.exception("Failed to unload LLM backend: %s", e)

    # ------------------------------------------------------------------
    # Enable / disable (subscribe to STT events)
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_enabled(self, enabled: bool) -> None:
        if enabled:
            self._subscribe_stt()
        else:
            self._unsubscribe_stt()

    def _subscribe_stt(self) -> None:
        """Start listening to STT events for translation."""
        if not self._engine or not self._engine.is_loaded or self._translator is None:
            logger.warning("Cannot enable translation — model not loaded")
            return
        dispatcher = get_dispatcher()
        dispatcher.subscribe(EventType.STT_CONFIRMED, self._translator.on_stt_confirmed)
        dispatcher.subscribe(EventType.STT_PENDING, self._translator.on_stt_pending)
        logger.info("Translation enabled — listening to STT events")

    def _unsubscribe_stt(self) -> None:
        """Stop listening to STT events."""
        if self._translator is None:
            return
        dispatcher = get_dispatcher()
        dispatcher.unsubscribe(EventType.STT_CONFIRMED, self._translator.on_stt_confirmed)
        dispatcher.unsubscribe(EventType.STT_PENDING, self._translator.on_stt_pending)
        self._state.current_translation = ""
        logger.info("Translation disabled")

    # ------------------------------------------------------------------
    # Target language
    # ------------------------------------------------------------------

    @Slot(str)
    def set_target_language(self, language: str) -> None:
        if self._translator:
            self._translator.target_language = language
        get_config().set("translation.target_language", language)

    # ------------------------------------------------------------------
    # Manual translation (no STT, user types source text)
    # ------------------------------------------------------------------

    @Slot(str)
    def translate_manual_text(self, text: str) -> None:
        """Translate manually entered text (standalone translator mode)."""
        if not self._engine or not self._engine.is_loaded:
            return
        t = threading.Thread(
            target=self._do_manual_translate,
            args=(text,),
            daemon=True,
        )
        t.start()

    def _do_manual_translate(self, text: str) -> None:
        cfg = get_config()
        target_lang = cfg.get("translation.target_language", "English")
        try:
            t0 = time.perf_counter()
            result = self._engine.translate(text, target_lang)
            dt_ms = (time.perf_counter() - t0) * 1000
            if result:
                self._state.current_translation = result
                self._state.accumulated_translation = result
                self._latency.record_translation(dt_ms)
        except Exception:
            logger.exception("Manual translation failed")

    # ------------------------------------------------------------------
    # Event handlers (pipeline events → state)
    # ------------------------------------------------------------------

    def _on_translation_confirmed(self, event) -> None:
        self._state.current_translation = event.translated_text
        if self._translator:
            self._state.accumulated_translation = self._translator.accumulated_text

    def _on_translation_pending(self, event) -> None:
        if event.translated_text:
            self._state.current_translation = event.translated_text

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Graceful shutdown — stop everything."""
        self._unsubscribe_stt()
        if self._translator:
            self._translator.reset()
        if self._engine and self._engine.is_loaded:
            self._engine.unload()
