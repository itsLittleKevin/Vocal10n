"""Qwen3-TTS client — local model inference.

Uses the qwen-tts Python package to load and run Qwen3-TTS models directly,
no subprocess server needed.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Qwen3TTSConfig:
    """Configuration for Qwen3-TTS synthesis."""

    # Model selection
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    tokenizer_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    local_model_path: str = ""  # If set, load from local directory
    local_tokenizer_path: str = ""

    # Device / precision
    device: str = "cuda:0"
    dtype: str = "bfloat16"  # "bfloat16" or "float16"
    use_flash_attn: bool = True

    # Reference audio (for voice cloning)
    ref_audio_path: str = ""
    ref_audio_text: str = ""
    ref_audio_lang: str = "en"

    # Output settings
    output_lang: str = "en"
    speed_factor: float = 1.2

    # Generation settings
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 1.0
    max_new_tokens: int = 2048


class Qwen3TTSClient:
    """Client for Qwen3-TTS local inference."""

    # Language code mapping (Vocal10n short codes → Qwen3-TTS names)
    _LANG_MAP = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }

    def __init__(self, config: Optional[Qwen3TTSConfig] = None):
        self.config = config or Qwen3TTSConfig()
        self._model = None
        self._voice_clone_prompt = None  # Cached prompt for reuse
        self._is_ready = False
        self._last_error: Optional[str] = None
        self._sample_rate: int = 24000  # Will be set after first synthesis

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the Qwen3-TTS model into GPU memory."""
        try:
            import torch
            try:
                from qwen_tts import Qwen3TTSModel
            except ImportError as e:
                logger.error("Failed to import qwen_tts: %s. Please install qwen-tts package.", e)
                return False

            cfg = self.config
            model_path = cfg.local_model_path or cfg.model_name
            dtype = getattr(torch, cfg.dtype, torch.bfloat16)
            attn_impl = "flash_attention_2" if cfg.use_flash_attn else "eager"

            logger.info("Loading Qwen3-TTS model: %s (dtype=%s, attn=%s)",
                        model_path, cfg.dtype, attn_impl)
            t0 = time.time()

            self._model = Qwen3TTSModel.from_pretrained(
                model_path,
                device_map=cfg.device,
                dtype=dtype,
                attn_implementation=attn_impl,
            )

            dt = time.time() - t0
            logger.info("Qwen3-TTS model loaded in %.1fs", dt)
            self._is_ready = True
            return True

        except Exception as e:
            self._last_error = str(e)
            logger.exception("Failed to load Qwen3-TTS model: %s", e)
            self._is_ready = False
            return False

    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        self._model = None
        self._voice_clone_prompt = None
        self._is_ready = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Qwen3-TTS model unloaded")

    # ------------------------------------------------------------------
    # Health & validation
    # ------------------------------------------------------------------

    def check_health(self) -> bool:
        """Check if model is loaded and ready."""
        self._is_ready = self._model is not None
        return self._is_ready

    def warmup(self) -> bool:
        """Perform a warmup synthesis."""
        logger.info("Qwen3-TTS warmup: synthesizing test phrase...")
        t0 = time.time()
        result = self.synthesize("Hello.", "en")
        dt = time.time() - t0
        if result["status"] == "success":
            logger.info("Qwen3-TTS warmup completed in %.1fs", dt)
            return True
        else:
            logger.warning("Qwen3-TTS warmup failed (%.1fs): %s", dt, result.get("message"))
            return False

    def validate_reference_audio(self) -> dict[str, Any]:
        """Validate reference audio file."""
        ref_path = Path(self.config.ref_audio_path)

        if not self.config.ref_audio_path:
            return {"valid": False, "error": "No reference audio configured", "path": ""}

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
        """Synthesize speech from text.

        Uses voice cloning if reference audio is configured,
        otherwise falls back to custom voice.

        Args:
            text: Text to synthesize.
            text_lang: Language code (default from config).
            streaming: Ignored for Qwen3-TTS (always non-streaming).

        Returns:
            Dict with 'status' and 'audio_data' or 'message'.
        """
        if not text.strip():
            return {"status": "error", "message": "Empty text"}

        if not self._model:
            return {"status": "error", "message": "Model not loaded"}

        text_lang = text_lang or self.config.output_lang
        qwen_lang = self._LANG_MAP.get(text_lang.lower()[:2], "Auto")
        start_time = time.time()

        try:
            ref_path = self._resolve_ref_path()

            if ref_path and Path(ref_path).exists():
                result = self._synthesize_clone(text, qwen_lang, ref_path)
            else:
                result = self._synthesize_custom(text, qwen_lang)

            latency_ms = (time.time() - start_time) * 1000
            result["latency_ms"] = latency_ms
            result["text"] = text
            result["language"] = text_lang
            return result

        except Exception as e:
            self._last_error = str(e)
            logger.exception("Qwen3-TTS synthesis failed")
            return {"status": "error", "message": str(e)}

    def _synthesize_clone(self, text: str, language: str, ref_path: str) -> dict[str, Any]:
        """Synthesize using voice cloning."""
        # Build or reuse cached clone prompt
        if self._voice_clone_prompt is None:
            logger.info("Building voice clone prompt from: %s", ref_path)
            self._voice_clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=self.config.ref_audio_text or "",
                x_vector_only_mode=not bool(self.config.ref_audio_text),
            )

        wavs, sr = self._model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=self._voice_clone_prompt,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )
        self._sample_rate = sr
        return self._wav_to_result(wavs[0], sr)

    def _synthesize_custom(self, text: str, language: str) -> dict[str, Any]:
        """Synthesize using built-in custom voices."""
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker="Ryan" if language == "English" else "Vivian",
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )
        self._sample_rate = sr
        return self._wav_to_result(wavs[0], sr)

    def _wav_to_result(self, wav_array, sample_rate: int) -> dict[str, Any]:
        """Convert numpy waveform to WAV bytes result dict."""
        import numpy as np

        # Convert float32 array to 16-bit PCM WAV bytes
        wav_array = np.clip(wav_array, -1.0, 1.0)
        pcm = (wav_array * 32767).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

        audio_data = buf.getvalue()
        duration_ms = (len(pcm) / sample_rate) * 1000

        return {
            "status": "success",
            "audio_data": audio_data,
            "sample_rate": sample_rate,
            "duration_ms": duration_ms,
        }

    def _resolve_ref_path(self) -> str:
        """Resolve reference audio path (handle relative paths)."""
        ref_path = self.config.ref_audio_path
        if ref_path and not Path(ref_path).is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            ref_path = str(project_root / ref_path)
        return ref_path

    # ------------------------------------------------------------------
    # Reference audio helpers
    # ------------------------------------------------------------------

    def set_reference_audio(self, audio_path: str, text: str = "", language: str = "en") -> bool:
        """Update reference audio for voice cloning."""
        self.config.ref_audio_path = audio_path
        self.config.ref_audio_text = text
        self.config.ref_audio_lang = language
        # Invalidate cached clone prompt so it's rebuilt on next synthesis
        self._voice_clone_prompt = None

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
            "model": self.config.model_name,
            "is_ready": self._is_ready,
            "last_error": self._last_error,
            "reference_audio": self.validate_reference_audio(),
            "output_language": self.config.output_lang,
        }
