"""GPT-SoVITS subprocess manager.

Launches the GPT-SoVITS API server as a separate process using venv_tts.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
# server_manager.py is at src/vocal10n/tts/server_manager.py
# parents[0]=tts, [1]=vocal10n, [2]=src, [3]=project_root (Vocal10n)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_SOVITS = _PROJECT_ROOT / "vendor" / "GPT-SoVITS"
_VENV_TTS = _PROJECT_ROOT / "venvs" / "venv_tts"


class GPTSoVITSServer:
    """Manages GPT-SoVITS API subprocess lifecycle."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9880,
        sovits_path: Optional[Path] = None,
        venv_path: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.sovits_path = Path(sovits_path) if sovits_path else _VENDOR_SOVITS
        self.venv_path = Path(venv_path) if venv_path else _VENV_TTS
        self._process: Optional[subprocess.Popen] = None
        self._startup_timeout = 60  # seconds

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """Check if the subprocess is still running."""
        if self._process is None:
            return False
        return self._process.poll() is None

    def _get_python_exe(self) -> Path:
        """Return python executable path for venv_tts."""
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    def start(self, wait_ready: bool = True) -> bool:
        """Start the GPT-SoVITS API server.

        Args:
            wait_ready: If True, block until the server responds to health check.

        Returns:
            True if server started successfully.
        """
        if self.is_running:
            logger.info("GPT-SoVITS server already running")
            return True

        python_exe = self._get_python_exe()
        if not python_exe.exists():
            logger.error("venv_tts python not found at %s", python_exe)
            return False

        api_script = self.sovits_path / "api_v2.py"
        if not api_script.exists():
            logger.error("api_v2.py not found at %s", api_script)
            return False

        cmd = [
            str(python_exe),
            str(api_script),
            "-a", self.host,
            "-p", str(self.port),
        ]

        logger.info("Starting GPT-SoVITS server: %s", " ".join(cmd))
        logger.info("Working directory: %s", self.sovits_path)

        # GPT-SoVITS needs both directories in PYTHONPATH:
        # - Root for tools/, GPT_SoVITS/ imports
        # - GPT_SoVITS/ subdir for AR/, BigVGAN/, module/, etc.
        env = dict(os.environ)
        gpt_sovits_subdir = self.sovits_path / "GPT_SoVITS"
        env["PYTHONPATH"] = f"{self.sovits_path};{gpt_sovits_subdir}"

        try:
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.sovits_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            logger.exception("Failed to start GPT-SoVITS: %s", e)
            return False

        if wait_ready:
            return self._wait_for_ready()
        return True

    def _wait_for_ready(self) -> bool:
        """Poll health endpoint until server responds or timeout."""
        deadline = time.time() + self._startup_timeout
        logger.info("Waiting for GPT-SoVITS to become ready...")

        while time.time() < deadline:
            if not self.is_running:
                logger.error("GPT-SoVITS process exited unexpectedly")
                return False

            if self.health_check():
                logger.info("GPT-SoVITS server ready at %s", self.api_url)
                return True

            time.sleep(1.0)

        logger.error("GPT-SoVITS startup timeout after %ds", self._startup_timeout)
        return False

    def health_check(self) -> bool:
        """Return True if server is responding."""
        try:
            # A GET to /tts with missing params returns 400, but means alive
            resp = requests.get(
                f"{self.api_url}/tts",
                params={"text": "ping", "text_lang": "en", "ref_audio_path": "x", "prompt_lang": "en"},
                timeout=2,
            )
            return resp.status_code in (200, 400, 422)
        except requests.RequestException:
            return False

    def stop(self) -> None:
        """Terminate the GPT-SoVITS subprocess."""
        if self._process is None:
            return

        logger.info("Stopping GPT-SoVITS server...")

        # Try graceful shutdown via API
        try:
            requests.get(f"{self.api_url}/control", params={"command": "exit"}, timeout=2)
        except requests.RequestException:
            pass

        # Give it a moment to exit
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing GPT-SoVITS process")
            self._process.kill()
            self._process.wait()

        self._process = None
        logger.info("GPT-SoVITS server stopped")

    def restart(self) -> bool:
        """Restart the server."""
        self.stop()
        return self.start(wait_ready=True)
