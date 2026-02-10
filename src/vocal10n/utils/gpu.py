"""GPU / VRAM monitoring via pynvml."""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pynvml  # provided by nvidia-ml-py

    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


@dataclass
class GPUInfo:
    gpu_util_pct: float = 0.0      # 0–100
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_util_pct: float = 0.0     # 0–100
    temperature_c: int = 0
    name: str = ""


class GPUMonitor:
    """Lightweight wrapper around pynvml for a single GPU."""

    def __init__(self, device_index: int = 0):
        self._index = device_index
        self._handle = None
        self._initialised = False

        if _NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._initialised = True
            except pynvml.NVMLError:
                logger.warning("Failed to initialise NVML for GPU %d", device_index)

    def query(self) -> GPUInfo:
        if not self._initialised or self._handle is None:
            return GPUInfo()
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode()
            total_mb = mem.total / (1024 * 1024)
            used_mb = mem.used / (1024 * 1024)
            return GPUInfo(
                gpu_util_pct=util.gpu,
                vram_used_mb=used_mb,
                vram_total_mb=total_mb,
                vram_util_pct=(used_mb / total_mb * 100) if total_mb else 0.0,
                temperature_c=temp,
                name=name,
            )
        except pynvml.NVMLError:
            logger.debug("NVML query failed", exc_info=True)
            return GPUInfo()

    def shutdown(self) -> None:
        if self._initialised:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._initialised = False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor(device_index: int = 0) -> GPUMonitor:
    global _monitor
    if _monitor is None:
        _monitor = GPUMonitor(device_index)
    return _monitor
