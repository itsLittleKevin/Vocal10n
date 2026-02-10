"""Shared constants and enumerations for Vocal10n."""

from enum import Enum


class EventType(Enum):
    """Event types in the pipeline."""
    # STT Events
    STT_PENDING = "stt.pending"
    STT_CONFIRMED = "stt.confirmed"
    STT_STARTED = "stt.started"
    STT_STOPPED = "stt.stopped"

    # Translation Events
    TRANSLATION_PENDING = "translation.pending"
    TRANSLATION_CONFIRMED = "translation.confirmed"
    TRANSLATION_ERROR = "translation.error"

    # TTS Events
    TTS_STARTED = "tts.started"
    TTS_CHUNK = "tts.chunk"
    TTS_COMPLETED = "tts.completed"
    TTS_ERROR = "tts.error"

    # Pipeline Events
    PIPELINE_READY = "pipeline.ready"
    PIPELINE_ERROR = "pipeline.error"
    LATENCY_REPORT = "pipeline.latency"


class ModelStatus(Enum):
    """Status of a model component."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


class Language(Enum):
    """Supported languages."""
    AUTO = "auto"
    CHINESE = "zh"
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"

    @property
    def display_name(self) -> str:
        names = {
            "auto": "Auto-detect",
            "zh": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
        }
        return names.get(self.value, self.value)


class TTSSource(Enum):
    """Which text stream feeds TTS."""
    CONFIRMED = "confirmed"
    PENDING = "pending"
    BOTH = "both"
