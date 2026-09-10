"""Auto-detects best available device: NPU > DirectML GPU > CUDA > CPU."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    NPU = "npu"
    DIRECTML = "directml"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass
class DeviceInfo:
    device_type: DeviceType
    device_name: str
    memory_gb: float
    supports_float16: bool
    # Torch device string used for pipeline `.to()` calls.
    # NPU inference runs through ONNX Runtime, so torch stays on CPU.
    torch_device: str


class DeviceManager:
    def __init__(self, force_device: Optional[str] = None) -> None:
        """
        Args:
            force_device: One of "npu", "directml", "cuda", "cpu" to skip auto-detection.
        """
        self._force = force_device
        self._cached: Optional[DeviceInfo] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> DeviceInfo:
        """Return (cached) best available device info."""
        if self._cached:
            return self._cached

        if self._force:
            probe = {
                "npu": self._try_npu,
                "directml": self._try_directml,
                "cuda": self._try_cuda,
                "mps": self._try_mps,
                "cpu": lambda: self._cpu_fallback(),
            }.get(self._force.lower())
            if probe:
                result = probe()
                if result:
                    self._cached = result
                    return result
            logger.warning("Forced device '%s' unavailable, falling back to auto-detect", self._force)

        for probe in (self._try_npu, self._try_directml, self._try_cuda, self._try_mps):
            result = probe()
            if result:
                self._cached = result
                logger.info("Selected device: %s (%s)", result.device_name, result.device_type.value)
                return result

        self._cached = self._cpu_fallback()
        logger.info("Falling back to CPU")
        return self._cached

    # ------------------------------------------------------------------
    # Private probes
    # ------------------------------------------------------------------

    @staticmethod
    def _try_npu() -> Optional[DeviceInfo]:
        try:
            import onnxruntime as ort

            if "VitisAIExecutionProvider" in ort.get_available_providers():
                logger.info("AMD Ryzen AI NPU detected (VitisAIExecutionProvider)")
                return DeviceInfo(
                    device_type=DeviceType.NPU,
                    device_name="AMD Ryzen AI NPU",
                    memory_gb=2.0,
                    supports_float16=True,
                    torch_device="cpu",  # ONNX handles NPU; PyTorch runs on CPU
                )
        except ImportError:
            pass
        return None

    @staticmethod
    def _try_directml() -> Optional[DeviceInfo]:
        try:
            import torch_directml

            name = torch_directml.device_name(0)
            logger.info("DirectML GPU detected: %s", name)
            return DeviceInfo(
                device_type=DeviceType.DIRECTML,
                device_name=name,
                memory_gb=17.0,  # AMD Radeon 760M: 2 GB dedicated + 15 GB shared
                supports_float16=True,
                torch_device="privateuseone",  # DirectML device string in PyTorch
            )
        except (ImportError, Exception):
            pass
        return None

    @staticmethod
    def _try_cuda() -> Optional[DeviceInfo]:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info("CUDA device detected: %s (%.1f GB)", name, mem_gb)
                return DeviceInfo(
                    device_type=DeviceType.CUDA,
                    device_name=name,
                    memory_gb=mem_gb,
                    supports_float16=True,
                    torch_device="cuda",
                )
        except ImportError:
            pass
        return None

    @staticmethod
    def _try_mps() -> Optional[DeviceInfo]:
        try:
            import torch
            if torch.backends.mps.is_available():
                import platform
                chip = platform.processor() or "Apple Silicon"
                logger.info("Apple MPS detected: %s", chip)
                return DeviceInfo(
                    device_type=DeviceType.MPS,
                    device_name=f"Apple MPS ({chip})",
                    memory_gb=16.0,  # unified memory; conservative estimate
                    supports_float16=True,
                    torch_device="mps",
                )
        except (ImportError, AttributeError):
            pass
        return None

    @staticmethod
    def _cpu_fallback() -> DeviceInfo:
        import multiprocessing

        cores = multiprocessing.cpu_count()
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_name=f"CPU ({cores} cores)",
            memory_gb=32.0,
            supports_float16=False,
            torch_device="cpu",
        )
