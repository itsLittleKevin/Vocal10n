"""Async file writer for SRT / TXT / WAV output.

Writes happen on a background thread so the pipeline never blocks on
disk I/O.  Each session creates timestamped files under ``output/subtitles/``:

    {stamp}_source.srt   — corrected source, punctuation stripped, space-separated
    {stamp}_target.srt   — translation, punctuation stripped, space-separated
    {stamp}_source.txt   — corrected source, with punctuation
    {stamp}_target.txt   — translation, with punctuation
    {stamp}.wav          — raw microphone audio (16 kHz mono)
"""

from __future__ import annotations

import logging
import queue
import re
import struct
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Punctuation characters to strip for SRT output
# Covers CJK, fullwidth, and standard Western punctuation
_PUNCT_RE = re.compile(
    r"[，。！？、；：""''（）【】《》「」『』〈〉"
    r",.!?;:\"'()\[\]{}<>—–\-…·•/\\@#$%^&*_+=|~`]"
)


def _strip_punctuation(text: str) -> str:
    """Remove punctuation and normalise whitespace for SRT output."""
    cleaned = _PUNCT_RE.sub(" ", text)
    return " ".join(cleaned.split())


def _srt_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS,mmm`` for SRT."""
    td = timedelta(seconds=max(0, seconds))
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class FileWriter:
    """Background file writer for pipeline output.

    Supports separate source/target SRT (punctuation-stripped) and TXT
    (with punctuation), plus WAV audio recording.
    """

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._queue: queue.Queue[Optional[tuple]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Session state — file paths
        self._src_srt_path: Optional[Path] = None
        self._tgt_srt_path: Optional[Path] = None
        self._src_txt_path: Optional[Path] = None
        self._tgt_txt_path: Optional[Path] = None
        self._wav_path: Optional[Path] = None
        self._session_start: float = 0.0
        self._srt_index: int = 0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self) -> None:
        """Create output files and start background writer thread."""
        sub_dir = self._output_dir / "subtitles"
        sub_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = self._output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._src_srt_path = sub_dir / f"{stamp}_source.srt"
        self._tgt_srt_path = sub_dir / f"{stamp}_target.srt"
        self._src_txt_path = sub_dir / f"{stamp}_source.txt"
        self._tgt_txt_path = sub_dir / f"{stamp}_target.txt"
        self._wav_path = audio_dir / f"{stamp}.wav"
        self._session_start = time.time()
        self._srt_index = 0

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="file-writer"
        )
        self._thread.start()
        logger.info("File writer session started: %s", stamp)

    def end_session(self) -> None:
        """Flush remaining writes and close."""
        if not self._thread:
            return
        self._stop_event.set()
        self._queue.put(None)  # poison pill
        self._thread.join(timeout=5.0)
        self._thread = None
        logger.info("File writer session ended — %d SRT entries", self._srt_index)

    # ------------------------------------------------------------------
    # Public write API (thread-safe, non-blocking)
    # ------------------------------------------------------------------

    def write_source_srt_entry(
        self,
        text: str,
        duration_s: float = 3.0,
    ) -> None:
        """Queue a source-language SRT entry (punctuation will be stripped)."""
        elapsed = time.time() - self._session_start
        self._queue.put(("src_srt", text, elapsed, duration_s))

    def write_target_srt_entry(
        self,
        text: str,
        duration_s: float = 3.0,
    ) -> None:
        """Queue a target-language SRT entry (punctuation will be stripped)."""
        elapsed = time.time() - self._session_start
        self._queue.put(("tgt_srt", text, elapsed, duration_s))

    def write_source_line(self, text: str) -> None:
        """Queue a line of source text for the source TXT file (with punctuation)."""
        self._queue.put(("src_txt", text))

    def write_target_line(self, text: str) -> None:
        """Queue a line of target text for the target TXT file (with punctuation)."""
        self._queue.put(("tgt_txt", text))

    def write_wav(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        """Queue raw audio data (float32 mono) for WAV output."""
        self._queue.put(("wav", audio, sample_rate))

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            self._dispatch(item)

        # Drain remaining items
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    self._dispatch(item)
            except queue.Empty:
                break

    def _dispatch(self, item: tuple) -> None:
        try:
            kind = item[0]
            if kind == "src_srt":
                self._do_write_srt(self._src_srt_path, item[1], item[2], item[3])
            elif kind == "tgt_srt":
                self._do_write_srt(self._tgt_srt_path, item[1], item[2], item[3])
            elif kind == "src_txt":
                self._do_append(self._src_txt_path, item[1])
            elif kind == "tgt_txt":
                self._do_append(self._tgt_txt_path, item[1])
            elif kind == "wav":
                self._do_write_wav(item[1], item[2])
        except Exception:
            logger.exception("File writer error")

    # ------------------------------------------------------------------
    # Disk I/O helpers
    # ------------------------------------------------------------------

    def _do_write_srt(
        self,
        path: Optional[Path],
        text: str,
        elapsed_s: float,
        duration_s: float,
    ) -> None:
        if not path:
            return
        self._srt_index += 1
        start_ts = _srt_timestamp(elapsed_s)
        end_ts = _srt_timestamp(elapsed_s + duration_s)
        clean = _strip_punctuation(text)

        block = (
            f"{self._srt_index}\n"
            f"{start_ts} --> {end_ts}\n"
            f"{clean}\n\n"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)

    def _do_append(self, path: Optional[Path], text: str) -> None:
        if not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")

    def _do_write_wav(self, audio: np.ndarray, sample_rate: int) -> None:
        """Write float32 mono audio as 16-bit PCM WAV."""
        if not self._wav_path:
            return
        # Convert float32 [-1, 1] → int16
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        raw = pcm.tobytes()

        num_channels = 1
        sample_width = 2  # 16-bit
        byte_rate = sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width
        data_size = len(raw)

        with open(self._wav_path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))           # chunk size
            f.write(struct.pack("<H", 1))            # PCM format
            f.write(struct.pack("<H", num_channels))
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", byte_rate))
            f.write(struct.pack("<H", block_align))
            f.write(struct.pack("<H", sample_width * 8))
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(raw)
