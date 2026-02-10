"""STT worker thread — incremental transcription loop.

Runs in a ``QThread`` so it can emit Qt signals safely.  The core
algorithm is ported from the prebuild's
``VoiceMemoApp._transcription_loop``:

    1. Only transcribe audio from the last confirmed endpoint onward.
    2. Segments older than ``confirm_threshold`` are confirmed.
    3. A *best-of cache* keeps the highest-confidence version of each
       pending time-bucket; the best version is used at confirm time.
"""

import logging
import time

from PySide6.QtCore import QThread, Signal

from vocal10n.config import get_config
from vocal10n.stt.audio_capture import AudioCapture
from vocal10n.stt.engine import STTEngine
from vocal10n.stt.transcript import TranscriptManager

logger = logging.getLogger(__name__)

_CACHE_BUCKET = 0.5  # seconds


class STTWorker(QThread):
    """Background thread running the incremental transcription loop."""

    error_occurred = Signal(str)

    def __init__(
        self,
        capture: AudioCapture,
        engine: STTEngine,
        transcript: TranscriptManager,
        initial_prompt: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._capture = capture
        self._engine = engine
        self._transcript = transcript
        self._initial_prompt = initial_prompt
        self._running = False

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """Start the worker (call from main thread)."""
        self._running = True
        self.start()

    def stop(self) -> None:
        """Request graceful stop and wait."""
        self._running = False
        self.wait(5000)

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self) -> None:  # noqa: C901 — intentionally complex, ported loop
        cfg = get_config()
        min_dur = cfg.get("stt.min_transcribe_duration", 0.3)
        min_conf = cfg.get("stt.min_confidence_threshold", -2.0)
        max_nsp = cfg.get("stt.max_no_speech_prob", 0.8)
        max_seg_age = cfg.get("stt.max_segment_age", 2.0)

        last_confirmed_end: float = 0.0
        last_transcribe_t: float = 0.0
        pending_cache: dict[float, dict] = {}

        try:
            while self._running:
                confirm_threshold = cfg.get("stt.confirm_threshold", 0.3)
                cur_dur = self._capture.duration
                if cur_dur < min_dur:
                    self.msleep(100)
                    continue
                if cur_dur - last_transcribe_t < 0.3:
                    self.msleep(100)
                    continue

                audio = self._capture.get_from_offset(last_confirmed_end)
                last_transcribe_t = cur_dur
                if len(audio) < self._capture.sample_rate * 0.3:
                    self.msleep(100)
                    continue

                segments = self._engine.transcribe(audio, initial_prompt=self._initial_prompt)
                if not segments:
                    self.msleep(100)
                    continue

                # Offset → absolute
                for s in segments:
                    s.start += last_confirmed_end
                    s.end += last_confirmed_end

                confirm_cutoff = cur_dur - confirm_threshold
                stable = [s for s in segments if s.end > last_confirmed_end + 0.1 and s.end < confirm_cutoff]
                pending = [s for s in segments if s.end > last_confirmed_end + 0.1 and s.end >= confirm_cutoff]

                # Update pending cache
                for s in pending:
                    bk = round(s.start / _CACHE_BUCKET) * _CACHE_BUCKET
                    old = pending_cache.get(bk)
                    if old is None or s.avg_logprob > old.get("confidence", -999):
                        pending_cache[bk] = {
                            "text": s.text, "start": s.start, "end": s.end,
                            "confidence": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                            "word_confidences": s.word_confidences,
                        }

                # Confirm stable segments
                for s in stable:
                    if s.end <= last_confirmed_end + 0.1:
                        continue
                    bk = round(s.start / _CACHE_BUCKET) * _CACHE_BUCKET
                    cached = pending_cache.get(bk)
                    # Use cached if better confidence and close in time
                    seg_data = {
                        "text": s.text, "start": s.start, "end": s.end,
                        "confidence": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
                        "word_confidences": s.word_confidences,
                    }
                    if cached and cached.get("confidence", -999) > s.avg_logprob:
                        if abs(cached["start"] - s.start) < _CACHE_BUCKET * 2:
                            seg_data = cached

                    if seg_data["confidence"] < min_conf or seg_data["no_speech_prob"] > max_nsp:
                        continue

                    accepted = self._transcript.confirm(
                        seg_data["text"].strip(),
                        seg_data["start"],
                        seg_data["end"],
                        confidence=seg_data["confidence"],
                        no_speech_prob=seg_data["no_speech_prob"],
                        word_confidences=seg_data.get("word_confidences"),
                    )
                    if accepted:
                        last_confirmed_end = seg_data["end"]
                        pending_cache = {
                            k: v for k, v in pending_cache.items()
                            if k >= last_confirmed_end - _CACHE_BUCKET
                        }

                # Force-confirm pending segments longer than max_segment_age.
                # We check segment *duration* (end - start), NOT whether the
                # segment ended a long time ago.  A long segment whose end is
                # still near cur_dur would never be caught by an end-age check
                # because Whisper keeps extending it while the speaker talks.
                if max_seg_age > 0:
                    for s in pending:
                        if s.end <= last_confirmed_end + 0.1:
                            continue
                        if (s.end - s.start) > max_seg_age:
                            if s.avg_logprob < min_conf or s.no_speech_prob > max_nsp:
                                continue
                            accepted = self._transcript.confirm(
                                s.text.strip(), s.start, s.end,
                                confidence=s.avg_logprob,
                                no_speech_prob=s.no_speech_prob,
                                word_confidences=s.word_confidences,
                            )
                            if accepted:
                                last_confirmed_end = s.end

                # Update pending display
                if pending:
                    valid = [
                        s for s in pending
                        if s.avg_logprob >= min_conf
                        and s.no_speech_prob <= max_nsp
                        and not self._transcript._filters.is_repetitive(s.text)
                        and not self._transcript._filters.contains_filtered(s.text)
                    ]
                    if valid:
                        all_wc: dict[str, float] = {}
                        for s in valid:
                            all_wc.update(s.word_confidences)
                        self._transcript.set_pending(
                            " ".join(s.text.strip() for s in valid), all_wc,
                        )
                    else:
                        self._transcript.set_pending("")
                else:
                    self._transcript.set_pending("")

                self.msleep(100)

            # ── Final flush ───────────────────────────────────────────
            audio = self._capture.get_from_offset(last_confirmed_end)
            if len(audio) > self._capture.sample_rate * 0.3:
                segs = self._engine.transcribe(audio, initial_prompt=self._initial_prompt)
                for s in segs:
                    s.start += last_confirmed_end
                    s.end += last_confirmed_end
                remaining = [s for s in segs if s.end > last_confirmed_end + 0.1]
                for i, s in enumerate(remaining):
                    if s.avg_logprob < min_conf or s.no_speech_prob > max_nsp:
                        continue
                    if self._transcript._filters.is_repetitive(s.text):
                        continue
                    if self._transcript._filters.contains_filtered(s.text):
                        continue
                    self._transcript.confirm(
                        s.text.strip(), s.start, s.end,
                        confidence=s.avg_logprob,
                        no_speech_prob=s.no_speech_prob,
                        word_confidences=s.word_confidences,
                        is_final=(i == len(remaining) - 1),
                    )
                    last_confirmed_end = s.end
            self._transcript.set_pending("")

        except Exception as e:
            logger.exception("STT worker error")
            self.error_occurred.emit(str(e))
