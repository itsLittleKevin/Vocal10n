"""Audio playback with device selection.

Uses sounddevice for low-latency playback of synthesized audio.
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Any, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def list_output_devices() -> list[dict[str, Any]]:
    """Return list of available audio output devices.

    Each dict contains: index, name, channels, sample_rate.
    """
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] > 0:
            devices.append({
                "index": idx,
                "name": dev["name"],
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
