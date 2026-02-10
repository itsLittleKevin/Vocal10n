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
    api_timeout: int = 120  # GPT-SoVITS first request can be slow (model warm-up)

    # Reference audio
    ref_audio_path: str = ""
    ref_audio_text: str = ""
    ref_audio_lang: str = "en"

    # Output settings
    output_lang: str = "en"
    streaming_mode: int = 3  # 0=off, 1=best, 2=medium, 3=fast (lowest first-byte latency)
    speed_factor: float = 1.2

    # Quality settings
    top_k: int = 5
    top_p: float = 0.7
    temperature: float = 0.5
    text_split_method: str = "cut0"
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

    def warmup(self) -> bool:
        """Perform a warmup synthesis to load models into memory.

        Returns:
            True if warmup succeeded.
        """
        logger.info("TTS warmup: synthesizing test phrase...")
        t0 = time.time()
        result = self.synthesize("Hello.", "en", streaming=False)
        dt = time.time() - t0
        if result["status"] == "success":
            logger.info("TTS warmup completed in %.1fs", dt)
            return True
        else:
            logger.warning("TTS warmup failed (%.1fs): %s", dt, result.get("message"))
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
        logger.info("TTS request: text='%s...', lang=%s, ref=%s", 
                    text[:30], text_lang, payload.get("ref_audio_path"))

        try:
            logger.info("TTS calling API: POST %s/tts (timeout=%ds)", self.api_url, self.config.api_timeout)
            resp = requests.post(
                f"{self.api_url}/tts",
                json=payload,
                timeout=self.config.api_timeout,
                stream=streaming,
            )
            latency_ms = (time.time() - start_time) * 1000
            logger.info("TTS API response: status=%d, latency=%.1fms", resp.status_code, latency_ms)

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
        # Ensure ref_audio_path is absolute
        ref_path = cfg.ref_audio_path
        if ref_path and not Path(ref_path).is_absolute():
            # Resolve relative to project root
            project_root = Path(__file__).resolve().parents[3]
            ref_path = str(project_root / ref_path)

        return {
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_path,
            "prompt_text": cfg.ref_audio_text,
            "prompt_lang": cfg.ref_audio_lang.lower(),
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "temperature": cfg.temperature,
            "text_split_method": cfg.text_split_method,
            "batch_size": cfg.batch_size,
            "speed_factor": cfg.speed_factor,
            "streaming_mode": cfg.streaming_mode,
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
            "request_start": time.time() - latency_ms / 1000,
            "streaming": True,
        }

    def _handle_error_response(self, resp: requests.Response) -> dict[str, Any]:
        try:
            data = resp.json()
            error_msg = data.get("message", "Unknown error")
            exception = data.get("Exception", "")
            if exception:
                error_msg = f"{error_msg}: {exception}"
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
