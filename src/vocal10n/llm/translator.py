"""Translation orchestrator — batching, debouncing, event wiring.

Ported from the prebuild's ``TranslationConsumer`` in
``pipeline_coordinator.py``.

Responsibilities:
- Subscribe to STT_CONFIRMED / STT_PENDING events.
- Batch confirmed fragments (wait for sentence end, clause + enough
  chars, max age, or silence timeout).
- Debounce pending text (150 ms).
- Clean transcription artefacts (stuttering, double punctuation).
- Drive :class:`LLMEngine.translate` and publish
  TRANSLATION_CONFIRMED / TRANSLATION_PENDING events.
"""

import logging
import re
import threading
import time

from vocal10n.config import get_config
from vocal10n.constants import EventType
from vocal10n.llm.corrector import Corrector
from vocal10n.llm.engine import LLMEngine
from vocal10n.pipeline.events import EventDispatcher, TextEvent
from vocal10n.pipeline.latency import LatencyTracker

logger = logging.getLogger(__name__)


class LLMTranslator:
    """Consumes STT events and produces translation events.

    All LLM calls go through the engine's internal lock, so this class
    only needs to guard its own mutable state (buffers / timers).
    """

    def __init__(
        self,
        engine: LLMEngine,
        dispatcher: EventDispatcher,
        latency: LatencyTracker,
    ) -> None:
        self._engine = engine
        self._dispatcher = dispatcher
        self._latency = latency

        cfg = get_config()
        self._target_lang: str = cfg.get("translation.target_language", "English")

        # Corrector — glossary-based STT correction hints
        self._corrector = Corrector()
        self._load_glossaries()

        # Debouncing for pending text
        self._pending_timer: threading.Timer | None = None
        self._pending_text: str = ""
        self._debounce_s: float = cfg.get("pipeline.translation_debounce_ms", 150) / 1000

        # Batching for confirmed text
        self._confirmed_buffer: list[str] = []
        self._confirmed_timer: threading.Timer | None = None
        self._batch_delay_s: float = cfg.get("pipeline.confirmed_batch_delay_ms", 800) / 1000
        self._buffer_start_t: float | None = None
        self._max_buffer_age: float = cfg.get("pipeline.max_buffer_age_s", 2.0)
        self._min_clause_chars: int = cfg.get("pipeline.min_clause_chars", 8)

        # Dedup
        self._last_confirmed_src: str = ""
        self._last_pending_src: str = ""

        # Context window — last N confirmed translations for coherence
        self._context_window: list[str] = []
        self._context_window_size: int = cfg.get("translation.context_window_size", 2)

        # Accumulated translation text
        self._accumulated: list[str] = []

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def target_language(self) -> str:
        return self._target_lang

    @target_language.setter
    def target_language(self, value: str) -> None:
        self._target_lang = value
        # Clear dedup caches so retranslation is allowed
        self._last_confirmed_src = ""
        self._last_pending_src = ""

    @property
    def accumulated_text(self) -> str:
        return " ".join(self._accumulated)

    def reset(self) -> None:
        """Clear all buffers (new session)."""
        with self._lock:
            self._confirmed_buffer.clear()
            self._buffer_start_t = None
            if self._confirmed_timer:
                self._confirmed_timer.cancel()
                self._confirmed_timer = None
            if self._pending_timer:
                self._pending_timer.cancel()
                self._pending_timer = None
            self._last_confirmed_src = ""
            self._last_pending_src = ""
            self._context_window.clear()
            self._accumulated.clear()

    def _load_glossaries(self) -> None:
        """Load glossary files from knowledge_base/ directory."""
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[3]
        kb_dir = project_root / "knowledge_base"
        if kb_dir.is_dir():
            count = self._corrector.load_glossary_dir(kb_dir)
            if count:
                logger.info("Corrector loaded %d glossary terms", count)

    # ------------------------------------------------------------------
    # Event handlers — called by dispatcher in arbitrary threads
    # ------------------------------------------------------------------

    def on_stt_confirmed(self, event: TextEvent) -> None:
        """Handle a confirmed STT segment."""
        text = event.text.strip()
        if not text:
            return

        with self._lock:
            if not self._confirmed_buffer:
                self._buffer_start_t = time.time()
            self._confirmed_buffer.append(text)
            combined = " ".join(self._confirmed_buffer)

            # 1) Sentence-ending punctuation → translate now
            if self._is_sentence_end(text):
                self._flush_and_translate(combined, "sentence")
                return

            # 2) Clause punctuation + enough content → early cut
            if self._is_clause_end(text) and self._char_count(combined) >= self._min_clause_chars:
                self._flush_and_translate(combined, "clause")
                return

            # 3) Buffer too old → force translate
            age = time.time() - self._buffer_start_t if self._buffer_start_t else 0
            if age >= self._max_buffer_age:
                self._flush_and_translate(combined, "max-age")
                return

            # 4) Otherwise wait for more (silence timer)
            if self._confirmed_timer:
                self._confirmed_timer.cancel()
            self._confirmed_timer = threading.Timer(
                self._batch_delay_s,
                self._on_confirmed_timeout,
            )
            self._confirmed_timer.daemon = True
            self._confirmed_timer.start()

    def on_stt_pending(self, event: TextEvent) -> None:
        """Handle pending (unconfirmed) STT text with debouncing."""
        text = event.text.strip()
        if not text:
            return

        with self._lock:
            self._pending_text = text
            if self._pending_timer:
                self._pending_timer.cancel()
            self._pending_timer = threading.Timer(
                self._debounce_s,
                self._on_pending_timeout,
            )
            self._pending_timer.daemon = True
            self._pending_timer.start()

    # ------------------------------------------------------------------
    # Internal — translation dispatch
    # ------------------------------------------------------------------

    def _flush_and_translate(self, combined: str, reason: str) -> None:
        """Must be called with self._lock held."""
        if self._confirmed_timer:
            self._confirmed_timer.cancel()
            self._confirmed_timer = None
        self._confirmed_buffer.clear()
        self._buffer_start_t = None

        if combined:
            t = threading.Thread(
                target=self._do_translate,
                args=(combined, False, reason),
                daemon=True,
            )
            t.start()

    def _on_confirmed_timeout(self) -> None:
        with self._lock:
            if not self._confirmed_buffer:
                return
            combined = " ".join(self._confirmed_buffer)
            self._confirmed_buffer.clear()
            self._confirmed_timer = None
            self._buffer_start_t = None
        if combined:
            self._do_translate(combined, False, "timeout")

    def _on_pending_timeout(self) -> None:
        with self._lock:
            text = self._pending_text
        if text:
            self._do_translate(text, True, "pending")

    def _do_translate(self, text: str, is_pending: bool, reason: str) -> None:
        """Run the actual translation (called from background thread)."""
        # Clean transcription artefacts
        cleaned = self._clean_transcription(text)
        if not cleaned:
            return

        # Dedup
        if is_pending and cleaned == self._last_pending_src:
            return
        if not is_pending and cleaned == self._last_confirmed_src:
            return

        if not self._engine.is_loaded:
            return

        try:
            t0 = time.perf_counter()
            # Build context from recent confirmed translations
            context = ""
            if not is_pending and self._context_window:
                context = " ".join(self._context_window[-self._context_window_size:])

            # Build glossary hint for domain-specific terms
            glossary_hint = self._corrector.build_glossary_hint(cleaned)
            if glossary_hint:
                context = f"{glossary_hint}\n{context}" if context else glossary_hint

            translation = self._engine.translate(cleaned, self._target_lang,
                                                 context=context)
            dt_ms = (time.perf_counter() - t0) * 1000

            if not translation:
                return

            # Track
            if is_pending:
                self._last_pending_src = cleaned
            else:
                self._last_confirmed_src = cleaned
                self._accumulated.append(translation)
                self._context_window.append(translation)
                # Keep context window bounded
                if len(self._context_window) > self._context_window_size * 2:
                    self._context_window = self._context_window[-self._context_window_size:]
                self._latency.record_translation(dt_ms)

            # Publish event
            event_type = (
                EventType.TRANSLATION_PENDING if is_pending
                else EventType.TRANSLATION_CONFIRMED
            )
            self._dispatcher.publish_translation(
                event_type=event_type,
                source_text=cleaned,
                translated_text=translation,
                target_language=self._target_lang,
                latency_ms=dt_ms,
                is_from_pending=is_pending,
            )

            logger.info(
                "Translation (%s, %.0f ms): '%s' → '%s'",
                reason, dt_ms,
                cleaned[:40], translation[:40],
            )

        except Exception:
            logger.exception("Translation failed")
            self._dispatcher.publish_text(
                EventType.TRANSLATION_ERROR,
                f"Translation error for: {cleaned[:40]}",
                source="translation",
            )

    # ------------------------------------------------------------------
    # Text cleaning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_transcription(text: str) -> str:
        """Remove stuttering, double punctuation, and filler artefacts."""
        if not text:
            return text
        # Ellipsis
        text = re.sub(r"[。.]{2,}", "", text)
        # Stuttering: char + punct + same char
        text = re.sub(r"(.)[，,、。](\1)", r"\2", text)
        # Multi-punctuation
        text = re.sub(r"[，,]{2,}", "，", text)
        text = re.sub(r"[。]{2,}", "。", text)
        # Leading/trailing junk
        text = re.sub(r"^[，,。.、；;\s]+", "", text)
        text = re.sub(r"[，,、；;\s]+$", "", text)
        return text.strip()

    @staticmethod
    def _is_sentence_end(text: str) -> bool:
        return bool(text) and text.strip()[-1] in "。！？.!?"

    @staticmethod
    def _is_clause_end(text: str) -> bool:
        return bool(text) and text.strip()[-1] in "，,；;：:"

    @staticmethod
    def _char_count(text: str) -> int:
        return len(re.sub(r"[\s，,。.！!？?；;：:、]", "", text))
