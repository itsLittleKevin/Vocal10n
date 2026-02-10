"""GPT-SoVITS HTTP client.

Provides sync and async methods against the GPT-SoVITS /tts endpoint.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """Configuration for TTS synthesis."""

    api_host: str = "127.0.0.1"
    api_port: int = 9880
    api_timeout: int = 30

    # Reference audio
    ref_audio_path: str = ""
    ref_audio_text: str = ""
    ref_audio_lang: str = "en"

    # Output settings
    output_lang: str = "en"
    streaming_mode: int = 0  # 0=off, 1=best, 2=medium, 3=fast
    speed_factor: float = 1.0

    # Quality settings
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1

    @property
    def api_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"


class GPTSoVITSClient:
    """HTTP client for GPT-SoVITS TTS API."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._is_ready = False
        self._last_error: Optional[str] = None

    @property
    def api_url(self) -> str:
        return self.config.api_url

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ------------------------------------------------------------------
    # Health & validation
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Check if TTS API is accessible."""
        try:
            resp = requests.get(
                f"{self.api_url}/tts",
                params={
                    "text": "ping",
                    "text_lang": "en",
                    "ref_audio_path": "x",
                    "prompt_lang": "en",
                },
                timeout=2,
            )
            self._is_ready = resp.status_code in (200, 400, 422)
            return self._is_ready
        except requests.RequestException as e:
            logger.warning("TTS health check failed: %s", e)
            self._is_ready = False
            return False

    def validate_reference_audio(self) -> dict[str, Any]:
        """Validate reference audio file."""
        ref_path = Path(self.config.ref_audio_path)

        if not ref_path.exists():
            return {"valid": False, "error": f"Not found: {ref_path}", "path": str(ref_path)}

        size_mb = ref_path.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            return {"valid": False, "error": f"Too large: {size_mb:.1f}MB", "path": str(ref_path)}

        try:
            with wave.open(str(ref_path), "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
                return {
                    "valid": True,
                    "path": str(ref_path),
                    "duration_seconds": duration,
                    "sample_rate": wf.getframerate(),
                    "channels": wf.getnchannels(),
                }
        except wave.Error:
            return {"valid": True, "path": str(ref_path), "warning": "Non-WAV format"}

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        text_lang: Optional[str] = None,
        streaming: bool = False,
    ) -> dict[str, Any]:
        """Synthesize speech from text (synchronous).

        Args:
            text: Text to synthesize.
            text_lang: Language code (default from config).
            streaming: Return streaming generator if True.

        Returns:
            Dict with 'status' and either 'audio_data' or 'error'.
        """
        if not text.strip():
            return {"status": "error", "message": "Empty text"}

        text_lang = text_lang or self.config.output_lang
        start_time = time.time()

        payload = self._build_payload(text, text_lang, streaming)

        try:
            resp = requests.post(
                f"{self.api_url}/tts",
                json=payload,
                timeout=self.config.api_timeout,
                stream=streaming,
            )
            latency_ms = (time.time() - start_time) * 1000

            if resp.status_code == 200:
                if streaming:
                    return self._handle_streaming(resp, latency_ms)
                return self._handle_full_response(resp, payload, latency_ms)

            return self._handle_error_response(resp)

        except requests.exceptions.Timeout:
            self._last_error = "TTS request timed out"
            return {"status": "error", "message": self._last_error}
        except requests.exceptions.ConnectionError:
            self._last_error = "Cannot connect to TTS server"
            return {"status": "error", "message": self._last_error}
        except Exception as e:
            self._last_error = str(e)
            logger.exception("TTS synthesis failed")
            return {"status": "error", "message": str(e)}

    def _build_payload(self, text: str, text_lang: str, streaming: bool) -> dict[str, Any]:
        cfg = self.config
        return {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": cfg.ref_audio_path,
            "prompt_text": cfg.ref_audio_text,
            "prompt_lang": cfg.ref_audio_lang.lower(),
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "temperature": cfg.temperature,
            "text_split_method": cfg.text_split_method,
            "batch_size": cfg.batch_size,
            "speed_factor": cfg.speed_factor,
            "streaming_mode": cfg.streaming_mode if streaming else 0,
            "media_type": "wav",
        }

    def _handle_full_response(
        self, resp: requests.Response, payload: dict, latency_ms: float
    ) -> dict[str, Any]:
        audio_data = resp.content
        duration_ms = self._get_audio_duration_ms(audio_data)
        return {
            "status": "success",
            "audio_data": audio_data,
            "sample_rate": 32000,
            "duration_ms": duration_ms,
            "latency_ms": latency_ms,
            "text": payload["text"],
            "language": payload["text_lang"],
        }

    def _handle_streaming(self, resp: requests.Response, latency_ms: float) -> dict[str, Any]:
        def audio_generator():
            for chunk in resp.iter_content(chunk_size=4096):
                if chunk:
                    yield chunk

        return {
            "status": "success",
            "audio_generator": audio_generator(),
            "latency_ms": latency_ms,
            "streaming": True,
        }

    def _handle_error_response(self, resp: requests.Response) -> dict[str, Any]:
        try:
            error_msg = resp.json().get("message", "Unknown error")
        except Exception:
            error_msg = resp.text[:200]
        self._last_error = error_msg
        logger.error("TTS API error %d: %s", resp.status_code, error_msg)
        return {"status": "error", "message": error_msg, "status_code": resp.status_code}

    def _get_audio_duration_ms(self, audio_data: bytes) -> float:
        """Calculate audio duration from WAV data."""
        try:
            with io.BytesIO(audio_data) as buf:
                with wave.open(buf, "rb") as wf:
                    return (wf.getnframes() / wf.getframerate()) * 1000
        except Exception:
            # Estimate: 16-bit mono @ 32kHz
            return (len(audio_data) / (32000 * 2)) * 1000

    # ------------------------------------------------------------------
    # Reference audio helpers
    # ------------------------------------------------------------------

    def set_reference_audio(self, audio_path: str, text: str = "", language: str = "en") -> bool:
        """Update reference audio for voice cloning."""
        self.config.ref_audio_path = audio_path
        self.config.ref_audio_text = text
        self.config.ref_audio_lang = language

        validation = self.validate_reference_audio()
        if not validation.get("valid", False):
            logger.error("Invalid reference audio: %s", validation.get("error"))
            return False
        return True

    def set_target_language(self, language: str) -> None:
        """Update target output language."""
        self.config.output_lang = language

    def get_status(self) -> dict[str, Any]:
        """Return current client status."""
        return {
            "api_url": self.api_url,
            "is_ready": self._is_ready,
            "last_error": self._last_error,
            "reference_audio": self.validate_reference_audio(),
            "output_language": self.config.output_lang,
            "streaming_mode": self.config.streaming_mode,
        }
