"""OpenAI-compatible API backend for LLM translation.

Allows using remote LLM servers (LM Studio, ollama, vLLM, etc.)
or cloud APIs (OpenAI, DeepSeek, etc.) as the translation engine.

Implements the same interface as ``LLMEngine`` so the controller
and translator can use either interchangeably.
"""

import logging
import re
import threading
import time

import requests

from vocal10n.config import get_config

logger = logging.getLogger(__name__)


class LLMApiBackend:
    """OpenAI-compatible chat completion backend.

    Thread-safety: all API calls go through ``_lock``.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._lock = threading.Lock()
        self._inference_times: list[float] = []
        self._session: requests.Session | None = None
        self._api_url: str = ""
        self._api_model: str = ""

    # ------------------------------------------------------------------
    # Lifecycle (same interface as LLMEngine)
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, model_path: str | None = None) -> None:
        """Validate API connection and mark as ready."""
        if self._loaded:
            return

        cfg = get_config().section("translation")
        self._api_url = cfg.get("api_url", "http://localhost:1234/v1").rstrip("/")
        self._api_model = cfg.get("api_model", "")
        api_key = cfg.get("api_key", "")

        # Ensure URL ends with /v1
        if not self._api_url.endswith("/v1"):
            self._api_url = self._api_url.rstrip("/") + "/v1"

        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"
        self._session.headers["Content-Type"] = "application/json"

        # Test connection by listing models
        try:
            resp = self._session.get(f"{self._api_url}/models", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            model_ids = [m.get("id", "") for m in models]
            logger.info("API connected. Available models: %s", model_ids)
            if not self._api_model and model_ids:
                self._api_model = model_ids[0]
                logger.info("Auto-selected model: %s", self._api_model)
        except Exception as e:
            logger.warning("Could not list models (non-fatal): %s", e)

        self._loaded = True
        logger.info(
            "API backend ready: %s (model: %s)", self._api_url, self._api_model,
        )
        self._warmup()

    def _warmup(self) -> None:
        """Run a quick test call to verify everything works."""
        t0 = time.perf_counter()
        try:
            self._call_chat(
                system="You are a translator.",
                user="Translate to English: 你好",
                max_tokens=8,
            )
        except Exception as e:
            logger.warning("API warmup failed (non-fatal): %s", e)
        dt = (time.perf_counter() - t0) * 1000
        logger.info("API warmup done in %.0f ms", dt)

    def unload(self) -> None:
        """Close session and mark as disconnected."""
        if self._session:
            self._session.close()
            self._session = None
        self._loaded = False
        self._inference_times.clear()
        logger.info("API backend disconnected")

    # ------------------------------------------------------------------
    # Translation (same interface as LLMEngine)
    # ------------------------------------------------------------------

    def translate(self, text: str, target_language: str) -> str:
        """Translate *text* to *target_language* via API.

        Raises ``RuntimeError`` if not connected.
        """
        if not self._loaded:
            raise RuntimeError("API backend not connected")

        cfg = get_config().section("translation")

        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        source_lang = "Chinese" if has_chinese else "English"

        template = cfg.get(
            "prompt_template",
            "Fix any speech recognition errors and translate to {target_lang}:\n{text}",
        )
        user_content = template.format(
            target_lang=target_language,
            text=text,
            source_lang=source_lang,
        )

        system_msg = (
            "You are a professional translator. "
            "Output only the corrected translation, nothing else."
        )

        t0 = time.perf_counter()
        with self._lock:
            result = self._call_chat(
                system=system_msg,
                user=user_content,
                max_tokens=cfg.get("max_tokens", 64),
                temperature=cfg.get("temperature", 0.0),
            )
        dt = (time.perf_counter() - t0) * 1000

        self._inference_times.append(dt)
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-100:]

        logger.debug("API translate %.0f ms: '%s' → '%s'", dt, text[:40], result[:40])
        return result

    @property
    def avg_latency_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> str:
        """Make a chat completion request and extract the reply."""
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if self._api_model:
            payload["model"] = self._api_model

        cfg = get_config().section("translation")
        timeout = cfg.get("api_timeout", 10)

        resp = self._session.post(
            f"{self._api_url}/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return self._extract_translation(content)

    @staticmethod
    def _extract_translation(text: str) -> str:
        """Pick first non-meta line from response."""
        meta_prefixes = ("translate", "translation:", "source:", "target:")
        for line in text.split("\n"):
            line = line.strip()
            if line and not any(line.lower().startswith(p) for p in meta_prefixes):
                return line
        return text.split("\n")[0].strip() if text else ""
