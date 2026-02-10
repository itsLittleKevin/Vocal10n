"""FasterWhisper model wrapper — load / unload / transcribe.

Ported from the prebuild's ``STTEngine`` in ``voice_memo.py``.
"""

import gc
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vocal10n.config import get_config

logger = logging.getLogger(__name__)

# Suppress verbose faster_whisper VAD/processing logs
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


@dataclass
class SegmentResult:
    """One Whisper segment with metadata."""

    text: str
    start: float  # seconds (relative to submitted audio)
    end: float
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    word_confidences: dict[str, float] = field(default_factory=dict)


class STTEngine:
    """Thin wrapper around ``faster_whisper.WhisperModel``.

    All public methods are designed to be called from a worker thread.
    """

    def __init__(self) -> None:
        self._model: Any = None  # WhisperModel (lazy import)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the Whisper model onto the device specified in config."""
        if self._model is not None:
            return

        cfg = get_config().section("stt")
        model_size = cfg.get("model_size", "large-v3-turbo")
        device = cfg.get("device", "cuda")
        compute_type = cfg.get("compute_type", "int8_float16")

        logger.info("Loading Whisper model %s (device=%s, compute=%s)", model_size, device, compute_type)

        from faster_whisper import WhisperModel  # heavy import — deferred

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        logger.info("Whisper model loaded")

    def unload(self) -> None:
        """Unload model and release VRAM."""
        if self._model is None:
            return
        del self._model
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Whisper model unloaded — VRAM freed")

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str = "",
    ) -> list[SegmentResult]:
        """Run Whisper on *audio* (float32 mono at 16 kHz).

        Returns a list of :class:`SegmentResult`.
        """
        if self._model is None:
            return []

        cfg = get_config().section("stt")
        language = cfg.get("language")  # None → auto-detect
        beam_size = cfg.get("beam_size", 1)

        segments_iter, _info = self._model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=beam_size,
            best_of=1,
            temperature=0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=400),
            word_timestamps=True,
            condition_on_previous_text=False,
            repetition_penalty=1.1,
            initial_prompt=initial_prompt or None,
        )

        results: list[SegmentResult] = []
        for seg in segments_iter:
            # Extract per-word confidence
            wc: dict[str, float] = {}
            if seg.words:
                for w in seg.words:
                    t = w.word.strip()
                    if t:
                        wc[t] = w.probability

            results.append(
                SegmentResult(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    avg_logprob=seg.avg_logprob,
                    no_speech_prob=seg.no_speech_prob,
                    word_confidences=wc,
                )
            )

        return results
