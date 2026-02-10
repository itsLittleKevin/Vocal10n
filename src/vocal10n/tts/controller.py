"""TTS controller — lifecycle and pipeline integration.

Owns GPT-SoVITS server management and audio playback queuing.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Slot

from vocal10n.config import get_config
from vocal10n.constants import EventType, ModelStatus
from vocal10n.pipeline.events import TranslationEvent, get_dispatcher
from vocal10n.pipeline.latency import LatencyTracker
from vocal10n.state import SystemState
from vocal10n.tts.audio_output import AudioPlayer
from vocal10n.tts.client import GPTSoVITSClient, TTSConfig
from vocal10n.tts.queue import TTSQueue
from vocal10n.tts.server_manager import GPTSoVITSServer

logger = logging.getLogger(__name__)

# Default reference audio (relative to project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REF_AUDIO = _PROJECT_ROOT / "reference_audio" / "audio_03.wav"
_DEFAULT_REF_TEXT = "I'm really sorry you were put on hold, but unfortunately we've had heavy call volume today. But I'm back now, and I can definitely check rates and availability for you."


class TTSController(QObject):
    """Manages TTS lifecycle: server start/stop, synthesis queue, and playback."""

    def __init__(self, state: SystemState, latency: Optional[LatencyTracker] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()
        self._latency = latency

        # TTS components (lazy init)
        self._server: Optional[GPTSoVITSServer] = None
        self._client: Optional[GPTSoVITSClient] = None
        self._queue: Optional[TTSQueue] = None
        self._player: Optional[AudioPlayer] = None
        self._pending_device: Optional[int] = None  # stored until player is created

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    @Slot()
    def start_server(self) -> None:
        """Start the GPT-SoVITS API server subprocess."""
        if self._state.tts_status == ModelStatus.LOADED:
            logger.info("TTS server already running")
            return

        self._state.tts_status = ModelStatus.LOADING
        logger.info("Starting GPT-SoVITS server...")

        try:
            host = self._cfg.get("tts.api_host", "127.0.0.1")
            port = self._cfg.get("tts.api_port", 9880)

            self._server = GPTSoVITSServer(host=host, port=port)
            if not self._server.start(wait_ready=True):
                raise RuntimeError("Server did not become ready")

            # Initialize client
            self._client = GPTSoVITSClient(self._build_config())
            self._client.check_health()

            # Initialize player with any device the user already selected
            self._player = AudioPlayer(device_index=self._pending_device)
            self._player.start()  # Start dedicated playback thread

            # Initialize queue
            self._queue = TTSQueue(self._client, max_size=10)
            self._queue.start(self._on_audio_ready)

            # Subscribe to translation events for auto-playback
            dispatcher = get_dispatcher()
            dispatcher.subscribe(EventType.TRANSLATION_CONFIRMED, self._on_translation)

            # Warmup TTS in background (first request loads models into GPU memory)
            # Status stays LOADING until warmup completes
            def do_warmup():
                logger.info("Warming up TTS model (background)...")
                if self._client.warmup():
                    self._state.tts_status = ModelStatus.LOADED
                    logger.info("TTS server ready (warmup complete)")
                else:
                    logger.warning("TTS warmup failed, but server is available")
                    self._state.tts_status = ModelStatus.LOADED

            threading.Thread(target=do_warmup, daemon=True).start()
            logger.info("TTS server starting warmup...")

        except Exception as e:
            logger.exception("Failed to start TTS server: %s", e)
            self._state.tts_status = ModelStatus.ERROR

    @Slot()
    def stop_server(self) -> None:
        """Stop the GPT-SoVITS server and cleanup."""
        self._state.tts_status = ModelStatus.UNLOADING
        logger.info("Stopping TTS server...")

        try:
            # Unsubscribe from translation events
            dispatcher = get_dispatcher()
            dispatcher.unsubscribe(EventType.TRANSLATION_CONFIRMED, self._on_translation)

            if self._queue:
                self._queue.stop()
                self._queue = None

            if self._player:
                self._player.stop()
                self._player = None

            if self._server:
                self._server.stop()
                self._server = None

            self._client = None
            self._state.tts_status = ModelStatus.UNLOADED
            logger.info("TTS server stopped")

        except Exception as e:
            logger.exception("Error stopping TTS server: %s", e)
            self._state.tts_status = ModelStatus.ERROR

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text: str, language: str) -> bool:
        """Queue text for synthesis.

        Args:
            text: Text to synthesize.
            language: Language code (en, zh, ja, etc.).

        Returns:
            True if queued successfully.
        """
        if self._state.tts_status != ModelStatus.LOADED:
            logger.warning("TTS not ready, cannot synthesize")
            return False

        if not self._queue:
            return False

        return self._queue.enqueue(text, language)

    def speak_source(self, text: str) -> bool:
        """Speak source text if enabled."""
        if not self._state.tts_source_enabled:
            return False
        lang = self._state.source_language.value.lower()[:2]
        return self.synthesize(text, lang)

    def speak_target(self, text: str) -> bool:
        """Speak target (translated) text if enabled."""
        if not self._state.tts_target_enabled:
            return False
        lang = self._state.target_language.value.lower()[:2]
        return self.synthesize(text, lang)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_translation(self, event: TranslationEvent) -> None:
        """Handle translation event — speak the translated text."""
        if not event.translated_text:
            return
        logger.debug("Translation event received: tts_target_enabled=%s, status=%s",
                     self._state.tts_target_enabled, self._state.tts_status)
        if self.speak_target(event.translated_text):
            logger.info("Queued TTS for: '%s'", event.translated_text[:50])
        else:
            logger.debug("TTS not queued (speak_target returned False)")

    # ------------------------------------------------------------------
    # Reference audio
    # ------------------------------------------------------------------

    @Slot(str, str, str)
    def set_reference_audio(self, path: str, text: str, language: str) -> None:
        """Update reference audio for voice cloning."""
        if self._client:
            lang_code = language.lower()[:2] if language else "en"
            self._client.set_reference_audio(path, text, lang_code)
        # Save to config
        self._cfg.set("tts.ref_audio_path", path)
        self._cfg.set("tts.ref_audio_text", text)
        self._cfg.set("tts.ref_audio_lang", language)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_config(self) -> TTSConfig:
        """Build TTSConfig from application config."""
        # Use default reference audio if none configured
        ref_path = self._cfg.get("tts.ref_audio_path", "")
        ref_text = self._cfg.get("tts.ref_audio_text", "")
        if not ref_path and _DEFAULT_REF_AUDIO.exists():
            ref_path = str(_DEFAULT_REF_AUDIO)
            ref_text = _DEFAULT_REF_TEXT
            logger.info("Using default reference audio: %s", ref_path)

        return TTSConfig(
            api_host=self._cfg.get("tts.api_host", "127.0.0.1"),
            api_port=self._cfg.get("tts.api_port", 9880),
            ref_audio_path=ref_path,
            ref_audio_text=ref_text,
            ref_audio_lang=self._cfg.get("tts.ref_audio_lang", "en"),
            output_lang=self._cfg.get("tts.output_lang", "en"),
            speed_factor=self._cfg.get("tts.speed_factor", 1.2),
            top_k=self._cfg.get("tts.top_k", 5),
            top_p=self._cfg.get("tts.top_p", 0.7),
            temperature=self._cfg.get("tts.temperature", 0.5),
        )

    @Slot(int)
    def set_output_device(self, device_index: int) -> None:
        """Change the audio output device."""
        idx = device_index if device_index >= 0 else None
        self._pending_device = idx
        if self._player:
            self._player.set_device(idx)
        logger.info("Audio output device changed to: %s", idx)

    def _on_audio_ready(self, result: dict) -> None:
        """Called by queue worker when audio is ready (streaming or full)."""
        if result.get("streaming") and self._player:
            generator = result.get("audio_generator")
            if generator:
                # 2-tier pipeline: blocks during chunk accumulation (synthesis),
                # then enqueues audio for the dedicated playback thread and
                # returns immediately so the queue worker can synthesise the
                # next item while current audio plays.
                request_start = result.get("request_start")
                self._player.play_stream(
                    generator,
                    request_start=request_start,
                    on_ttfa=self._record_tts_latency,
                )
            else:
                logger.warning("Streaming result but no audio_generator")
        else:
            latency_ms = result.get("latency_ms", 0)
            if latency_ms > 0:
                self._record_tts_latency(latency_ms)
            audio_data = result.get("audio_data")
            if audio_data and self._player:
                logger.info("Playing TTS audio: %d bytes, duration=%.1fms",
                            len(audio_data), result.get("duration_ms", 0))
                success = self._player.play(audio_data, blocking=False)
                if not success:
                    logger.error("Audio playback failed")
            else:
                logger.warning("Audio callback: no audio_data or no player")

    def _record_tts_latency(self, latency_ms: float) -> None:
        """Record TTS latency and compute total pipeline latency."""
        if not self._latency:
            return
        self._latency.record_tts(latency_ms)
        stt_stats = self._latency.get_stats("stt")
        trans_stats = self._latency.get_stats("translation")
        if stt_stats.count > 0 and trans_stats.count > 0:
            total = stt_stats.current_ms + trans_stats.current_ms + latency_ms
            self._latency.record_total(total)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop everything on application exit."""
        self.stop_server()
