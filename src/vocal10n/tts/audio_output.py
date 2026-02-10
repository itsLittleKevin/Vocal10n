"""Audio playback with device selection.

Uses sounddevice for low-latency playback of synthesized audio.
Two-tier pipeline (matching legacy architecture):
  Tier 1 — caller accumulates streaming chunks into a full audio buffer
  Tier 2 — dedicated playback thread plays buffers sequentially via sd.play()
Synthesis of item N+1 overlaps with playback of item N.
"""

from __future__ import annotations

import io
import logging
import queue
import struct
import threading
import time as _time
import wave
from typing import Any, Callable, Generator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def list_output_devices() -> list[dict[str, Any]]:
    """Return list of available audio output devices.

    Each dict contains: index, name, channels, sample_rate.

    Windows reports the same physical device under multiple host APIs
    (MME, DirectSound, WASAPI, WDM-KS).  MME truncates names to ~31
    characters.  We deduplicate by:
      1. Collecting all output devices.
      2. Skipping any device whose name is a prefix of a longer device
         name (i.e. the MME-truncated variant).
      3. Among devices with the same full name, keeping only the first.
    """
    raw: list[dict[str, Any]] = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            raw.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_output_channels"],
                "sample_rate": int(dev["default_samplerate"]),
            })

    # Collect all names to detect truncated prefixes
    all_names = [d["name"] for d in raw]

    def is_truncated_prefix(name: str) -> bool:
        """True if *name* looks like an MME-truncated version of a longer name."""
        if len(name) < 31:
            return False  # Only MME names hit the ~31-char limit
        for other in all_names:
            if other != name and other.startswith(name):
                return True
        return False

    seen: set[str] = set()
    devices: list[dict[str, Any]] = []
    for d in raw:
        name = d["name"]
        if is_truncated_prefix(name):
            continue
        if name in seen:
            continue
        seen.add(name)
        devices.append(d)
    return devices


def get_default_output_device() -> Optional[dict[str, Any]]:
    """Return the default output device info, or None if unavailable."""
    try:
        idx = sd.default.device[1]
        if idx is None:
            return None
        dev = sd.query_devices(idx)
        return {
            "index": idx,
            "name": dev["name"],
            "channels": dev["max_output_channels"],
            "sample_rate": int(dev["default_samplerate"]),
        }
    except Exception:
        return None


class AudioPlayer:
    """Plays audio through a selected output device.

    Uses a dedicated playback thread with its own queue so that
    synthesis of the *next* item can overlap with playback of the
    *current* item (2-tier pipeline, matching legacy architecture).
    """

    def __init__(self, device_index: Optional[int] = None):
        self.device_index = device_index
        # Playback queue: holds (samples_f32, sample_rate, on_done_callback) tuples
        self._play_queue: queue.Queue[Optional[tuple]] = queue.Queue()
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the dedicated playback thread."""
        if self._playback_thread and self._playback_thread.is_alive():
            return
        self._stop_event.clear()
        self._playback_thread = threading.Thread(
            target=self._playback_loop, daemon=True, name="tts-playback"
        )
        self._playback_thread.start()
        logger.info("Audio playback thread started")

    def set_device(self, device_index: Optional[int]) -> None:
        """Change output device."""
        self.device_index = device_index

    def play(self, audio_data: bytes, blocking: bool = False) -> bool:
        """Play WAV audio data.

        Args:
            audio_data: Raw WAV bytes.
            blocking: If True, block until playback finishes.

        Returns:
            True if playback started successfully.
        """
        try:
            samples, sample_rate = self._decode_wav(audio_data)
        except Exception as e:
            logger.error("Failed to decode WAV audio: %s", e)
            return False

        if blocking:
            try:
                sd.play(samples, samplerate=sample_rate, device=self.device_index)
                sd.wait()
                return True
            except sd.PortAudioError as e:
                logger.error("Audio playback failed: %s", e)
                return False

        # Non-blocking: enqueue for the playback thread
        self._play_queue.put((samples, sample_rate, None))
        return True

    def play_stream(
        self,
        chunk_generator: Generator[bytes, None, None],
        request_start: Optional[float] = None,
        on_ttfa: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """Accumulate streaming chunks into a buffer, then enqueue for playback.

        This method BLOCKS while accumulating HTTP chunks (synthesis time),
        but returns as soon as the audio is enqueued for playback.
        The dedicated playback thread handles sd.play() + sd.wait().

        This means synthesis of item N+1 starts while item N is still playing.

        Args:
            chunk_generator: Yields raw audio bytes (first chunk has WAV header).
            request_start: Epoch when the text entered the TTS queue.
            on_ttfa: Callback(ttfa_ms) invoked when audio is enqueued for playback.

        Returns:
            Dict with 'success'.
        """
        sample_rate = 32000
        channels = 1
        sample_width = 2
        header_parsed = False
        pcm_chunks: list[bytes] = []

        try:
            for chunk in chunk_generator:
                if not chunk:
                    continue

                if not header_parsed:
                    if len(chunk) >= 44 and chunk[:4] == b"RIFF":
                        channels = struct.unpack_from("<H", chunk, 22)[0]
                        sample_rate = struct.unpack_from("<I", chunk, 24)[0]
                        sample_width = struct.unpack_from("<H", chunk, 34)[0] // 8
                        pcm_data = chunk[44:]
                        header_parsed = True
                    else:
                        pcm_data = chunk
                        header_parsed = True
                else:
                    pcm_data = chunk

                if pcm_data:
                    pcm_chunks.append(pcm_data)

            if not pcm_chunks:
                logger.warning("Streaming playback: no audio data received")
                return {"success": False}

            # Build numpy buffer
            all_pcm = b"".join(pcm_chunks)
            dtype = np.int16 if sample_width == 2 else np.int32
            samples = np.frombuffer(all_pcm, dtype=dtype)
            if channels > 1:
                samples = samples.reshape(-1, channels)
            samples_f32 = samples.astype(np.float32) / (
                32768 if sample_width == 2 else 2147483648
            )

            duration_s = len(samples) / sample_rate

            # Report TTFA — time from text enqueue to audio ready for playback
            if request_start is not None:
                ttfa_ms = (_time.time() - request_start) * 1000
                logger.info(
                    "TTS synth done: %.0fms since enqueue, %.1fs audio, queueing for playback",
                    ttfa_ms, duration_s,
                )
                if on_ttfa:
                    try:
                        on_ttfa(ttfa_ms)
                    except Exception as e:
                        logger.error("on_ttfa callback error: %s", e)
            else:
                logger.info("TTS synth done: %.1fs audio, queueing for playback", duration_s)

            # Enqueue for playback thread — returns immediately
            self._play_queue.put((samples_f32, sample_rate, None))
            return {"success": True}

        except Exception as e:
            logger.error("Stream accumulation error: %s", e)
            return {"success": False}

    # ------------------------------------------------------------------
    # Playback thread (Tier 2)
    # ------------------------------------------------------------------

    def _playback_loop(self) -> None:
        """Dedicated playback thread: dequeue audio → sd.play() → sd.wait()."""
        logger.info("Playback loop running")
        while not self._stop_event.is_set():
            try:
                item = self._play_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                break  # Poison pill

            samples_f32, sample_rate, on_done = item
            try:
                sd.play(samples_f32, samplerate=sample_rate, device=self.device_index)
                sd.wait()
            except sd.PortAudioError as e:
                logger.error("Audio playback failed: %s", e)
            except Exception as e:
                logger.error("Playback error: %s", e)

            if on_done:
                try:
                    on_done()
                except Exception:
                    pass

        logger.info("Playback loop stopped")

    def stop(self) -> None:
        """Stop playback thread and discard pending audio."""
        self._stop_event.set()
        # Drain the queue
        while not self._play_queue.empty():
            try:
                self._play_queue.get_nowait()
            except queue.Empty:
                break
        # Send poison pill in case thread is blocking on get()
        self._play_queue.put(None)
        sd.stop()
        if self._playback_thread:
            self._playback_thread.join(timeout=3.0)
            self._playback_thread = None

    def _decode_wav(self, audio_data: bytes) -> tuple[np.ndarray, int]:
        """Decode WAV bytes to numpy array and sample rate."""
        with io.BytesIO(audio_data) as buf:
            with wave.open(buf, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

        # Determine dtype
        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        samples = np.frombuffer(raw, dtype=dtype)

        # Reshape for multi-channel
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels)

        # Normalize to float32 [-1, 1]
        if dtype == np.uint8:
            samples = (samples.astype(np.float32) - 128) / 128
        elif dtype == np.int16:
            samples = samples.astype(np.float32) / 32768
        elif dtype == np.int32:
            samples = samples.astype(np.float32) / 2147483648

        return samples, sample_rate
