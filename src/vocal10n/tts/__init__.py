"""TTS module — GPT-SoVITS client and audio playback."""

from vocal10n.tts.audio_output import AudioPlayer, get_default_output_device, list_output_devices
from vocal10n.tts.client import GPTSoVITSClient, TTSConfig
from vocal10n.tts.controller import TTSController
from vocal10n.tts.queue import TTSQueue
from vocal10n.tts.server_manager import GPTSoVITSServer

__all__ = [
    "AudioPlayer",
    "GPTSoVITSClient",
    "GPTSoVITSServer",
    "TTSConfig",
    "TTSController",
    "TTSQueue",
    "get_default_output_device",
    "list_output_devices",
]
