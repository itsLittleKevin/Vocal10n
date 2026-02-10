"""LLM model wrapper — load / unload / translate via llama-cpp-python.

Ported from the prebuild's ``OptimizedModelLoader`` in
``realtime_translator.py``.
"""

import gc
import logging
import re
import threading
import time

from vocal10n.config import get_config

logger = logging.getLogger(__name__)


class LLMEngine:
    """Thin wrapper around ``llama_cpp.Llama`` for Qwen3-4B GGUF.

    Thread-safety: all inference goes through ``_lock`` since
    llama-cpp-python is **not** thread-safe.
    """

    def __init__(self) -> None:
        self._model = None  # Llama instance (lazy import)
        self._lock = threading.Lock()
        self._inference_times: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: str | None = None) -> None:
        """Load the GGUF model onto the GPU."""
        if self._model is not None:
            return

        cfg = get_config().section("translation")
        if model_path is None:
            model_path = cfg.get("model_path", "models/llm/Qwen3-4B-Instruct-2507.Q4_K_M.gguf")

        # Resolve relative path from project root
        from pathlib import Path
        p = Path(model_path)
        if not p.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            p = project_root / p
        model_path = str(p)

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading LLM model: %s", model_path)

        from llama_cpp import Llama  # heavy import — deferred

        n_gpu_layers = cfg.get("n_gpu_layers", -1)
        self._model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            main_gpu=0,
            n_ctx=cfg.get("n_ctx", 128),
            n_batch=cfg.get("n_batch", 8),
            n_threads=cfg.get("n_threads", 4),
            n_threads_batch=cfg.get("n_threads", 4),
            use_mlock=False,
            use_mmap=True,
            flash_attn=True,
            verbose=False,
        )
        logger.info("LLM model loaded (n_gpu_layers=%s)", n_gpu_layers)

        # Warmup
        self._warmup()

    def _warmup(self) -> None:
        """Run a quick inference to warm up CUDA kernels."""
        if self._model is None:
            return
        t0 = time.perf_counter()
        try:
            self._model(
                prompt=(
                    "<|im_start|>system\nYou are a translator.<|im_end|>\n"
                    "<|im_start|>user\nTranslate to English:\n你好<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
                max_tokens=8,
                temperature=0.0,
                top_k=1,
                stop=["<|im_end|>", "\n\n"],
                echo=False,
            )
        except Exception as e:
            logger.warning("Warmup failed (non-fatal): %s", e)
        dt = (time.perf_counter() - t0) * 1000
        logger.info("LLM warmup done in %.0f ms", dt)

    def unload(self) -> None:
        """Unload model and release VRAM."""
        if self._model is None:
            return
        del self._model
        self._model = None
        self._inference_times.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("LLM model unloaded — VRAM freed")

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate(self, text: str, target_language: str) -> str:
        """Translate *text* to *target_language*.  Returns translated string.

        Raises ``RuntimeError`` if model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("LLM model not loaded")

        prompt = self._build_prompt(text, target_language)

        cfg = get_config().section("translation")

        t0 = time.perf_counter()
        with self._lock:
            response = self._model(
                prompt=prompt,
                max_tokens=cfg.get("max_tokens", 64),
                temperature=cfg.get("temperature", 0.0),
                top_k=cfg.get("top_k", 1),
                top_p=cfg.get("top_p", 1.0),
                repeat_penalty=1.0,
                stop=["<|im_end|>", "\n\n"],
                echo=False,
            )
        dt = (time.perf_counter() - t0) * 1000

        # Track latency
        self._inference_times.append(dt)
        if len(self._inference_times) > 100:
            self._inference_times = self._inference_times[-100:]

        result = self._extract_translation(response)
        logger.debug("LLM translate %.0f ms: '%s' → '%s'", dt, text[:40], result[:40])
        return result

    @property
    def avg_latency_ms(self) -> float:
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(text: str, target_language: str) -> str:
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
        source_lang = "Chinese" if has_chinese else "English"

        cfg = get_config()
        template = cfg.get(
            "translation.prompt_template",
            "Fix any speech recognition errors and translate to {target_lang}:\n{text}",
        )
        user_content = template.format(
            target_lang=target_language,
            text=text,
            source_lang=source_lang,
        )
        # Qwen3 ChatML format (non-thinking Instruct model)
        return (
            "<|im_start|>system\n"
            "You are a professional translator. "
            "Output only the corrected translation, nothing else.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_content}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    @staticmethod
    def _extract_translation(response) -> str:
        if isinstance(response, dict) and "choices" in response:
            text = response["choices"][0]["text"].strip()
        else:
            text = str(response).strip()

        meta_prefixes = ("translate", "translation:", "source:", "target:")
        for line in text.split("\n"):
            line = line.strip()
            if line and not any(line.lower().startswith(p) for p in meta_prefixes):
                return line

        return text.split("\n")[0].strip() if text else ""
