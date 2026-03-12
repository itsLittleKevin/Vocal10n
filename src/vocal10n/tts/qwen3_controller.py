"""Qwen3-TTS controller — lifecycle and pipeline integration.

Loads the Qwen3-TTS model directly (no subprocess), manages synthesis
queue and audio playback.
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
from vocal10n.tts.qwen3_client import Qwen3TTSClient, Qwen3TTSConfig
from vocal10n.tts.queue import TTSQueue

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_REF_AUDIO = _PROJECT_ROOT / "reference_audio" / "audio_03.wav"
_DEFAULT_REF_TEXT = "I'm really sorry you were put on hold, but unfortunately we've had heavy call volume today. But I'm back now, and I can definitely check rates and availability for you."


class Qwen3TTSController(QObject):
    """Manages Qwen3-TTS lifecycle: model load/unload, synthesis queue, playback."""

    def __init__(self, state: SystemState, latency: Optional[LatencyTracker] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._state = state
        self._cfg = get_config()
        self._latency = latency

        self._client: Optional[Qwen3TTSClient] = None
        self._queue: Optional[TTSQueue] = None
        self._player: Optional[AudioPlayer] = None
        self._pending_device: Optional[int] = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    @Slot()
    def load_model(self) -> None:
        """Load the Qwen3-TTS model."""
        if self._state.tts_status == ModelStatus.LOADED:
            logger.info("Qwen3-TTS already loaded")
            return

        self._state.tts_status = ModelStatus.LOADING
        logger.info("Loading Qwen3-TTS model...")

        def _do_load():
            try:
                self._client = Qwen3TTSClient(self._build_config())

                if not self._client.load_model():
                    raise RuntimeError("Model load returned False")

                # Initialize player
                self._player = AudioPlayer(device_index=self._pending_device)
                self._player.start()

                # Initialize queue — reuse the same TTSQueue with duck-typed client
                max_pending = self._cfg.get("pipeline.tts_queue_max_pending", 3)
                self._queue = TTSQueue(self._client, max_size=10,
                                       max_pending=max_pending)
                self._queue.start(self._on_audio_ready)

                # Subscribe to translation events
                dispatcher = get_dispatcher()
                dispatcher.subscribe(EventType.TRANSLATION_CONFIRMED, self._on_translation)

                # Warmup
                logger.info("Warming up Qwen3-TTS model...")
                if self._client.warmup():
                    self._state.tts_status = ModelStatus.LOADED
                    logger.info("Qwen3-TTS ready (warmup complete)")
                else:
                    logger.warning("Qwen3-TTS warmup failed, but model is loaded")
                    self._state.tts_status = ModelStatus.LOADED

            except Exception as e:
                logger.exception("Failed to load Qwen3-TTS: %s", e)
                self._state.tts_status = ModelStatus.ERROR

        threading.Thread(target=_do_load, daemon=True).start()

    @Slot()
    def unload_model(self) -> None:
        """Unload the Qwen3-TTS model and cleanup."""
        self._state.tts_status = ModelStatus.UNLOADING
        logger.info("Unloading Qwen3-TTS model...")

        try:
            try:
                dispatcher = get_dispatcher()
                dispatcher.unsubscribe(EventType.TRANSLATION_CONFIRMED, self._on_translation)
            except Exception:
                pass

            q, self._queue = self._queue, None
            p, self._player = self._player, None
            c, self._client = self._client, None

            if q:
                q.stop()
            if p:
                p.stop()
            if c:
                c.unload_model()

            self._state.tts_status = ModelStatus.UNLOADED
            logger.info("Qwen3-TTS model unloaded")

        except Exception as e:
            logger.exception("Error unloading Qwen3-TTS: %s", e)
            self._state.tts_status = ModelStatus.ERROR

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(self, text: str, language: str) -> bool:
        if self._state.tts_status != ModelStatus.LOADED:
            logger.warning("Qwen3-TTS not ready, cannot synthesize")
            return False
        if not self._queue:
            return False
        return self._queue.enqueue(text, language)

    def speak_source(self, text: str) -> bool:
        if not self._state.tts_source_enabled:
            return False
        lang = self._state.source_language.value.lower()[:2]
        return self.synthesize(text, lang)

    def speak_target(self, text: str) -> bool:
        if not self._state.tts_target_enabled:
            return False
        lang = self._state.target_language.value.lower()[:2]
        return self.synthesize(text, lang)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_translation(self, event: TranslationEvent) -> None:
        if not event.translated_text:
            return
        if self.speak_target(event.translated_text):
            logger.info("Queued Qwen3-TTS for: '%s'", event.translated_text[:50])

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Clean shutdown — called on application exit."""
        self.unload_model()

    # ------------------------------------------------------------------
    # Reference audio
    # ------------------------------------------------------------------

    @Slot(str, str, str)
    def set_reference_audio(self, path: str, text: str, language: str) -> None:
        if self._client:
            lang_code = language.lower()[:2] if language else "en"
            self._client.set_reference_audio(path, text, lang_code)
        self._cfg.set("tts_qwen3.ref_audio_path", path)
        self._cfg.set("tts_qwen3.ref_audio_text", text)
        self._cfg.set("tts_qwen3.ref_audio_lang", language)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_config(self) -> Qwen3TTSConfig:
        ref_path = self._cfg.get("tts_qwen3.ref_audio_path", "")
        ref_text = self._cfg.get("tts_qwen3.ref_audio_text", "")
        if not ref_path and _DEFAULT_REF_AUDIO.exists():
            ref_path = str(_DEFAULT_REF_AUDIO)
            ref_text = _DEFAULT_REF_TEXT

        return Qwen3TTSConfig(
            model_name=self._cfg.get("tts_qwen3.model_name", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            local_model_path=self._cfg.get("tts_qwen3.local_model_path", ""),
            local_tokenizer_path=self._cfg.get("tts_qwen3.local_tokenizer_path", ""),
            device=self._cfg.get("tts_qwen3.device", "cuda:0"),
            dtype=self._cfg.get("tts_qwen3.dtype", "bfloat16"),
            use_flash_attn=self._cfg.get("tts_qwen3.use_flash_attn", True),
            ref_audio_path=ref_path,
            ref_audio_text=ref_text,
            ref_audio_lang=self._cfg.get("tts_qwen3.ref_audio_lang", "en"),
            output_lang=self._cfg.get("tts_qwen3.output_lang", "en"),
            speed_factor=self._cfg.get("tts_qwen3.speed_factor", 1.2),
            top_k=self._cfg.get("tts_qwen3.top_k", 50),
            top_p=self._cfg.get("tts_qwen3.top_p", 0.9),
            temperature=self._cfg.get("tts_qwen3.temperature", 1.0),
        )

    @Slot(int)
    def set_output_device(self, device_index: int) -> None:
        idx = device_index if device_index >= 0 else None
        self._pending_device = idx
        if self._player:
            self._player.set_device(idx)
        logger.info("Audio output device changed to: %s", idx)

    def _on_audio_ready(self, result: dict) -> None:
        """Called by queue worker when audio is ready."""
        latency_ms = result.get("latency_ms", 0)
        if latency_ms > 0:
            self._record_tts_latency(latency_ms)
        audio_data = result.get("audio_data")
        if audio_data and self._player:
            logger.info("Playing Qwen3-TTS audio: %d bytes, duration=%.1fms",
                        len(audio_data), result.get("duration_ms", 0))
            self._player.play(audio_data, blocking=False)
        else:
            logger.warning("Qwen3-TTS audio callback: no audio_data or no player")

    def _record_tts_latency(self, latency_ms: float) -> None:
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

    def get_error_message(self) -> str | None:
        """Get the last error message from the client."""
        if self._client:
            status = self._client.get_status()
            return status.get("last_error")
        return None
