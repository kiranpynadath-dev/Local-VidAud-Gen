"""Utilities: logging, progress tracking, memory management."""
from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from tqdm import tqdm


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, handlers=handlers, force=True)


class MemoryManager:
    """Context manager that runs GC + clears GPU cache on exit."""

    def __enter__(self) -> "MemoryManager":
        return self

    def __exit__(self, *_) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class ProgressTracker:
    def __init__(self, total: int, desc: str = "Processing", unit: str = "step") -> None:
        self._bar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def set_description(self, desc: str) -> None:
        self._bar.set_description(desc)

    def close(self) -> None:
        self._bar.close()

    def __enter__(self) -> "ProgressTracker":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def check_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_video_duration(path: Path) -> float:
    """Return duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    raise RuntimeError(f"Could not determine duration for {path}")


def get_memory_info() -> dict:
    info: dict = {}
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_used_gb"] = round(mem.used / 1024**3, 2)
        info["ram_total_gb"] = round(mem.total / 1024**3, 2)
        info["ram_available_gb"] = round(mem.available / 1024**3, 2)
    except ImportError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            info["gpu_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            info["gpu_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
    except ImportError:
        pass

    return info


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
