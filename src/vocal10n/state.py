"""Thread-safe global state for Vocal10n, with Qt signals for UI binding."""

import threading
from typing import Optional

from PySide6.QtCore import QObject, Signal

from vocal10n.constants import Language, ModelStatus


class SystemState(QObject):
    """Centralised, observable application state.

    Every public attribute change emits the corresponding Qt signal so that
    UI widgets can connect directly without polling.
    """

    # -- Signals (emitted on state change) ----------------------------------
    stt_status_changed = Signal(ModelStatus)
    llm_status_changed = Signal(ModelStatus)
    tts_status_changed = Signal(ModelStatus)

    stt_enabled_changed = Signal(bool)
    llm_enabled_changed = Signal(bool)
    tts_enabled_changed = Signal(bool)
    tts_source_enabled_changed = Signal(bool)
    tts_target_enabled_changed = Signal(bool)

    source_language_changed = Signal(Language)
    target_language_changed = Signal(Language)

    current_stt_text_changed = Signal(str)
    accumulated_stt_text_changed = Signal(str)
    current_translation_changed = Signal(str)
    accumulated_translation_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._lock = threading.RLock()

        # Model status
        self._stt_status = ModelStatus.UNLOADED
        self._llm_status = ModelStatus.UNLOADED
        self._tts_status = ModelStatus.UNLOADED

        # Component toggles
        self._stt_enabled = False
        self._llm_enabled = False
        self._tts_enabled = False
        self._tts_source_enabled = False
        self._tts_target_enabled = True  # Default: speak translated text

        # Languages
        self._source_language = Language.AUTO
        self._target_language = Language.ENGLISH

        # Live text (updated by pipeline workers)
        self._current_stt_text = ""
        self._accumulated_stt_text = ""
        self._current_translation = ""
        self._accumulated_translation = ""

    # -- Properties with signal emission ------------------------------------

    @property
    def stt_status(self) -> ModelStatus:
        return self._stt_status

    @stt_status.setter
    def stt_status(self, value: ModelStatus) -> None:
        with self._lock:
            if self._stt_status != value:
                self._stt_status = value
                self.stt_status_changed.emit(value)

    @property
    def llm_status(self) -> ModelStatus:
        return self._llm_status

    @llm_status.setter
    def llm_status(self, value: ModelStatus) -> None:
        with self._lock:
            if self._llm_status != value:
                self._llm_status = value
                self.llm_status_changed.emit(value)

    @property
    def tts_status(self) -> ModelStatus:
        return self._tts_status

    @tts_status.setter
    def tts_status(self, value: ModelStatus) -> None:
        with self._lock:
            if self._tts_status != value:
                self._tts_status = value
                self.tts_status_changed.emit(value)

    @property
    def stt_enabled(self) -> bool:
        return self._stt_enabled

    @stt_enabled.setter
    def stt_enabled(self, value: bool) -> None:
        with self._lock:
            if self._stt_enabled != value:
                self._stt_enabled = value
                self.stt_enabled_changed.emit(value)

    @property
    def llm_enabled(self) -> bool:
        return self._llm_enabled

    @llm_enabled.setter
    def llm_enabled(self, value: bool) -> None:
        with self._lock:
            if self._llm_enabled != value:
                self._llm_enabled = value
                self.llm_enabled_changed.emit(value)

    @property
    def tts_enabled(self) -> bool:
        return self._tts_enabled

    @tts_enabled.setter
    def tts_enabled(self, value: bool) -> None:
        with self._lock:
            if self._tts_enabled != value:
                self._tts_enabled = value
                self.tts_enabled_changed.emit(value)

    @property
    def tts_source_enabled(self) -> bool:
        return self._tts_source_enabled

    @tts_source_enabled.setter
    def tts_source_enabled(self, value: bool) -> None:
        with self._lock:
            if self._tts_source_enabled != value:
                self._tts_source_enabled = value
                self.tts_source_enabled_changed.emit(value)

    @property
    def tts_target_enabled(self) -> bool:
        return self._tts_target_enabled

    @tts_target_enabled.setter
    def tts_target_enabled(self, value: bool) -> None:
        with self._lock:
            if self._tts_target_enabled != value:
                self._tts_target_enabled = value
                self.tts_target_enabled_changed.emit(value)

    @property
    def source_language(self) -> Language:
        return self._source_language

    @source_language.setter
    def source_language(self, value: Language) -> None:
        with self._lock:
            if self._source_language != value:
                self._source_language = value
                self.source_language_changed.emit(value)

    @property
    def target_language(self) -> Language:
        return self._target_language

    @target_language.setter
    def target_language(self, value: Language) -> None:
        with self._lock:
            if self._target_language != value:
                self._target_language = value
                self.target_language_changed.emit(value)

    @property
    def current_stt_text(self) -> str:
        return self._current_stt_text

    @current_stt_text.setter
    def current_stt_text(self, value: str) -> None:
        with self._lock:
            self._current_stt_text = value
            self.current_stt_text_changed.emit(value)

    @property
    def accumulated_stt_text(self) -> str:
        return self._accumulated_stt_text

    @accumulated_stt_text.setter
    def accumulated_stt_text(self, value: str) -> None:
        with self._lock:
            self._accumulated_stt_text = value
            self.accumulated_stt_text_changed.emit(value)

    @property
    def current_translation(self) -> str:
        return self._current_translation

    @current_translation.setter
    def current_translation(self, value: str) -> None:
        with self._lock:
            self._current_translation = value
            self.current_translation_changed.emit(value)

    @property
    def accumulated_translation(self) -> str:
        return self._accumulated_translation

    @accumulated_translation.setter
    def accumulated_translation(self, value: str) -> None:
        with self._lock:
            self._accumulated_translation = value
            self.accumulated_translation_changed.emit(value)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_state: Optional[SystemState] = None


def get_state() -> SystemState:
    global _state
    if _state is None:
        _state = SystemState()
    return _state
