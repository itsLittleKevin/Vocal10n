"""TTS module — GPT-SoVITS and Qwen3-TTS clients and audio playback."""

from vocal10n.tts.audio_output import AudioPlayer, get_default_output_device, list_output_devices
from vocal10n.tts.client import GPTSoVITSClient, TTSConfig
from vocal10n.tts.controller import TTSController
from vocal10n.tts.qwen3_client import Qwen3TTSClient, Qwen3TTSConfig
from vocal10n.tts.qwen3_controller import Qwen3TTSController
from vocal10n.tts.queue import TTSQueue
from vocal10n.tts.server_manager import GPTSoVITSServer

__all__ = [
    "AudioPlayer",
    "GPTSoVITSClient",
    "GPTSoVITSServer",
    "Qwen3TTSClient",
    "Qwen3TTSConfig",
    "Qwen3TTSController",
    "TTSConfig",
    "TTSController",
    "TTSQueue",
    "get_default_output_device",
    "list_output_devices",
]
