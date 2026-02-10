"""Audio playback with device selection.

Uses sounddevice for low-latency playback of synthesized audio.
"""

from __future__ import annotations

import io
import logging
import struct
import wave
from typing import Any, Generator, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def list_output_devices() -> list[dict[str, Any]]:
    """Return list of available audio output devices.

    Each dict contains: index, name, channels, sample_rate.
    Deduplicates by keeping only the first device per unique name
    (typically the Windows DirectSound / WASAPI default).
    """
    seen_names: set[str] = set()
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            name = dev["name"]
            if name in seen_names:
                continue
            seen_names.add(name)
            devices.append({
                "index": idx,
                "name": name,
                "channels": dev["max_output_channels"],
                "sample_rate": int(dev["default_samplerate"]),
            })
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
    """Plays WAV audio data through a selected output device."""

    def __init__(self, device_index: Optional[int] = None):
        """Initialize player.

        Args:
            device_index: Output device index (None = default).
        """
        self.device_index = device_index

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

        try:
            logger.debug("Playing audio: shape=%s, sr=%d, device=%s", samples.shape, sample_rate, self.device_index)
            sd.play(samples, samplerate=sample_rate, device=self.device_index)
            if blocking:
                sd.wait()
            logger.info("Audio playback started successfully")
            return True
        except sd.PortAudioError as e:
            logger.error("Audio playback failed: %s", e)
            return False

    def play_stream(
        self,
        chunk_generator: Generator[bytes, None, None],
        request_start: Optional[float] = None,
    ) -> dict:
        """Play streaming audio chunks as they arrive.

        The first chunk must contain a WAV header (44 bytes) followed by PCM data.
        Subsequent chunks are raw PCM data.

        Args:
            chunk_generator: Yields raw audio bytes (first chunk has WAV header).
            request_start: Epoch time when the TTS request was initiated.

        Returns:
            Dict with 'success' (bool) and 'ttfa_ms' (time-to-first-audio in ms).
        """
        sample_rate = 32000
        channels = 1
        sample_width = 2  # int16
        header_parsed = False
        stream = None
        ttfa_ms: Optional[float] = None
        import time as _time

        try:
            for chunk in chunk_generator:
                if not chunk:
                    continue

                if not header_parsed:
                    # Parse WAV header from first chunk
                    if len(chunk) >= 44 and chunk[:4] == b"RIFF":
                        channels = struct.unpack_from("<H", chunk, 22)[0]
                        sample_rate = struct.unpack_from("<I", chunk, 24)[0]
                        sample_width = struct.unpack_from("<H", chunk, 34)[0] // 8
                        pcm_data = chunk[44:]  # Skip WAV header
                        header_parsed = True
                        logger.debug("Stream WAV header: sr=%d, ch=%d, sw=%d", sample_rate, channels, sample_width)
                    else:
                        pcm_data = chunk
                        header_parsed = True
                else:
                    pcm_data = chunk

                if not pcm_data:
                    continue

                # Create output stream on first PCM data
                if stream is None:
                    # Use a large blocksize to buffer ~0.25s of audio,
                    # preventing underrun stutters when HTTP chunks arrive slowly
                    blocksize = int(sample_rate * 0.25)
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=channels,
                        dtype="int16" if sample_width == 2 else "int32",
                        device=self.device_index,
                        blocksize=blocksize,
                    )
                    stream.start()
                    if request_start is not None:
                        ttfa_ms = (_time.time() - request_start) * 1000
                        logger.info("Streaming audio playback started (sr=%d, TTFA=%.0fms)", sample_rate, ttfa_ms)
                    else:
                        logger.info("Streaming audio playback started (sr=%d)", sample_rate)

                # Convert bytes to numpy and write
                dtype = np.int16 if sample_width == 2 else np.int32
                samples = np.frombuffer(pcm_data, dtype=dtype)
                if channels > 1:
                    samples = samples.reshape(-1, channels)
                else:
                    samples = samples.reshape(-1, 1)
                stream.write(samples)

            if stream:
                stream.stop()
                stream.close()
            return {"success": True, "ttfa_ms": ttfa_ms}

        except Exception as e:
            logger.error("Streaming playback error: %s", e)
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
            return {"success": False, "ttfa_ms": ttfa_ms}

    def stop(self) -> None:
        """Stop any ongoing playback."""
        sd.stop()

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
