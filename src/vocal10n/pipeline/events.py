"""Pub/sub event dispatcher for the Vocal10n pipeline.

Ported from the prebuild's ``event_dispatcher.py`` with simplifications:
- Removed async path (PySide6 uses signals instead).
- Kept thread-safe synchronous dispatch.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from vocal10n.constants import EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event data classes
# ---------------------------------------------------------------------------

@dataclass
class Event:
    type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    event_id: str = field(default_factory=lambda: f"{time.time_ns()}")
    origin_timestamp: Optional[float] = None


@dataclass
class TextEvent(Event):
    text: str = ""
    language: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    is_final: bool = False


@dataclass
class TranslationEvent(Event):
    source_text: str = ""
    source_language: str = ""
    translated_text: str = ""
    target_language: str = ""
    latency_ms: float = 0.0
    is_from_pending: bool = False


@dataclass
class AudioEvent(Event):
    audio_data: bytes = b""
    sample_rate: int = 32000
    duration_ms: float = 0.0
    is_chunk: bool = False


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class EventDispatcher:
    """Thread-safe pub/sub event system."""

    def __init__(self, max_history: int = 100):
        self._subscribers: Dict[EventType, List[Callable]] = {
            et: [] for et in EventType
        }
        self._history: deque = deque(maxlen=max_history)
        self._lock = threading.RLock()

    # -- Subscribe / unsubscribe -------------------------------------------

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        with self._lock:
            if callback not in self._subscribers[event_type]:
                self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        with self._lock:
            subs = self._subscribers[event_type]
            if callback in subs:
                subs.remove(callback)

    # -- Publish ------------------------------------------------------------

    def publish(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            subscribers = self._subscribers.get(event.type, []).copy()
        for cb in subscribers:
            try:
                cb(event)
            except Exception:
                logger.exception("Error in event subscriber %s", cb.__name__)

    # -- Convenience publishers ---------------------------------------------

    def publish_text(
        self,
        event_type: EventType,
        text: str,
        language: str = "",
        start_time: float = 0.0,
        end_time: float = 0.0,
        is_final: bool = False,
        source: str = "stt",
    ) -> TextEvent:
        event = TextEvent(
            type=event_type,
            data={"text": text, "language": language},
            source=source,
            text=text,
            language=language,
            start_time=start_time,
            end_time=end_time,
            is_final=is_final,
            origin_timestamp=(
                time.time() - (end_time - start_time) if end_time > 0 else time.time()
            ),
        )
        self.publish(event)
        return event

    def publish_translation(
        self,
        event_type: EventType,
        source_text: str,
        translated_text: str,
        source_language: str = "",
        target_language: str = "",
        latency_ms: float = 0.0,
        is_from_pending: bool = False,
        origin_timestamp: Optional[float] = None,
        source: str = "translation",
    ) -> TranslationEvent:
        event = TranslationEvent(
            type=event_type,
            data={
                "source_text": source_text,
                "translated_text": translated_text,
                "latency_ms": latency_ms,
            },
            source=source,
            source_text=source_text,
            source_language=source_language,
            translated_text=translated_text,
            target_language=target_language,
            latency_ms=latency_ms,
            is_from_pending=is_from_pending,
            origin_timestamp=origin_timestamp,
        )
        self.publish(event)
        return event

    def publish_audio(
        self,
        event_type: EventType,
        audio_data: bytes,
        sample_rate: int = 32000,
        duration_ms: float = 0.0,
        is_chunk: bool = False,
        source: str = "tts",
    ) -> AudioEvent:
        event = AudioEvent(
            type=event_type,
            data={"size": len(audio_data), "duration_ms": duration_ms},
            source=source,
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            is_chunk=is_chunk,
        )
        self.publish(event)
        return event

    # -- History ------------------------------------------------------------

    def get_recent_events(
        self, count: int = 10, event_type: Optional[EventType] = None
    ) -> List[Event]:
        with self._lock:
            events = list(self._history)
        if event_type:
            events = [e for e in events if e.type == event_type]
        return events[-count:]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_dispatcher: Optional[EventDispatcher] = None


def get_dispatcher() -> EventDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = EventDispatcher()
    return _dispatcher
