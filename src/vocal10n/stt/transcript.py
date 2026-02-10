"""Segment management and confirmation logic for STT output.

Ported from the prebuild's ``TranscriptManager.add_segment`` /
``update_pending`` / ``_transcription_loop`` logic.

Responsibilities:
- Accept raw Whisper segments, apply filters, add to confirmed history.
- Track pending (unconfirmed) text for live display.
- Publish events (``STT_CONFIRMED`` / ``STT_PENDING``) via the event
  dispatcher so downstream modules (translation, TTS, OBS) react.
- Maintain a rolling history for duplicate detection.
"""

import logging
import time
from dataclasses import dataclass, field

from vocal10n.config import get_config
from vocal10n.constants import EventType
from vocal10n.pipeline.events import EventDispatcher
from vocal10n.pipeline.latency import LatencyTracker
from vocal10n.stt.filters import STTFilters

logger = logging.getLogger(__name__)


@dataclass
class ConfirmedSegment:
    """A finalised STT segment."""

    text: str          # with punctuation
    text_clean: str    # stripped of punctuation
    start: float
    end: float


class TranscriptManager:
    """Accepts Whisper segments, filters them, and publishes events.

    This class owns the *confirmed history* and *pending text*.  The
    incremental transcription worker (:class:`STTWorker`) calls
    :meth:`confirm` and :meth:`set_pending` as new segments arrive.
    """

    def __init__(
        self,
        dispatcher: EventDispatcher,
        latency: LatencyTracker,
        filters: STTFilters,
    ) -> None:
        self._dispatcher = dispatcher
        self._latency = latency
        self._filters = filters

        cfg = get_config()
        self._use_simplified: bool = cfg.get("stt.use_simplified_chinese", True)
        self._min_confidence: float = cfg.get("stt.min_confidence_threshold", -2.0)
        self._max_no_speech: float = cfg.get("stt.max_no_speech_prob", 0.8)

        self.segments: list[ConfirmedSegment] = []
        self.pending_text: str = ""

        # Lightweight history dicts for duplicate detection
        self._history: list[dict] = []

        # Session timing
        self._session_start: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        self.segments.clear()
        self._history.clear()
        self.pending_text = ""
        self._session_start = time.time()

    # ------------------------------------------------------------------
    # Confirm a segment (called by worker)
    # ------------------------------------------------------------------

    def confirm(
        self,
        text: str,
        start: float,
        end: float,
        confidence: float = 0.0,
        no_speech_prob: float = 0.0,
        word_confidences: dict[str, float] | None = None,
        is_final: bool = False,
    ) -> bool:
        """Process and accept a segment.  Returns ``True`` if accepted."""

        # -- Pre-processing ---
        if self._use_simplified:
            text = self._filters.to_simplified(text)
        text = self._filters.phonetic_correct(text, word_confidences)

        # -- Confidence gate ---
        if confidence < self._min_confidence or no_speech_prob > self._max_no_speech:
            logger.debug("Filtered (confidence): '%s' conf=%.2f nsp=%.2f", text.strip(), confidence, no_speech_prob)
            return False

        # -- Filter gates ---
        if self._filters.is_repetitive(text):
            logger.debug("Filtered (repetitive): '%s'", text.strip())
            return False
        if self._filters.contains_filtered(text, confidence, no_speech_prob):
            logger.debug("Filtered (phrase): '%s'", text.strip())
            return False
        if self._filters.is_duplicate(text, start, end, self._history):
            logger.debug("Filtered (duplicate): '%s'", text.strip())
            return False

        # Short-phrase consecutive dedup:
        # If a very short phrase (≤4 chars) was already confirmed recently, skip.
        text_clean_check = self._filters.strip_punctuation(text).strip()
        if text_clean_check and len(text_clean_check) <= 4 and self.segments:
            for prev in self.segments[-3:]:
                if prev.text_clean == text_clean_check:
                    logger.debug("Filtered (short repeat): '%s'", text.strip())
                    return False

        # -- Pause-based punctuation upgrade ---
        if self.segments:
            prev = self.segments[-1]
            gap = start - prev.end
            prev_is_cn = self._filters.is_chinese(prev.text)
            max_sent = 40 if prev_is_cn else 80
            sent_len = self._current_sentence_length()
            if gap > 0.5 or (gap > 0.3 and sent_len > max_sent):
                self._upgrade_previous_to_period()

        # -- Punctuate ---
        text_punct = self._filters.ensure_punctuation(text, is_final=is_final)
        text_clean = self._filters.strip_punctuation(text)

        seg = ConfirmedSegment(text=text_punct, text_clean=text_clean, start=start, end=end)
        self.segments.append(seg)
        self._history.append({"text": text_punct, "start": start, "end": end})
        if len(self._history) > 20:
            self._history = self._history[-20:]

        # -- Publish event ---
        self._dispatcher.publish_text(
            EventType.STT_CONFIRMED,
            text_clean,
            language="",
            start_time=start,
            end_time=end,
            is_final=is_final,
            source="stt",
        )

        # -- Latency ---
        if self._session_start:
            wall = time.time()
            audio_wall = self._session_start + end
            latency_ms = (wall - audio_wall) * 1000
            if latency_ms > 0:
                self._latency.record_stt(latency_ms)

        return True

    # ------------------------------------------------------------------
    # Pending text
    # ------------------------------------------------------------------

    def set_pending(self, text: str, word_confidences: dict[str, float] | None = None) -> None:
        """Update the current pending (unconfirmed) text."""
        if self._use_simplified:
            text = self._filters.to_simplified(text)
        text = self._filters.phonetic_correct(text, word_confidences)
        self.pending_text = text.strip()

        self._dispatcher.publish_text(
            EventType.STT_PENDING,
            self.pending_text,
            source="stt",
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def live_text(self) -> str:
        """Pending if available, else last confirmed segment."""
        if self.pending_text:
            return self.pending_text
        if self.segments:
            return self.segments[-1].text_clean
        return ""

    @property
    def full_transcript(self) -> str:
        return "".join(s.text for s in self.segments)

    def clear(self) -> None:
        self.segments.clear()
        self._history.clear()
        self.pending_text = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_sentence_length(self) -> int:
        total = 0
        for seg in reversed(self.segments):
            if seg.text and seg.text[-1] in "。！？.!?":
                break
            total += len(seg.text.rstrip("，,"))
        return total

    def _upgrade_previous_to_period(self) -> None:
        if not self.segments:
            return
        prev = self.segments[-1]
        if prev.text and prev.text[-1] in "，,":
            ch = "。" if self._filters.is_chinese(prev.text) else "."
            self.segments[-1] = ConfirmedSegment(
                text=prev.text[:-1] + ch,
                text_clean=prev.text_clean,
                start=prev.start,
                end=prev.end,
            )
