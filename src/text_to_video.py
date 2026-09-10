"""Text-to-video generation — free, local, no API key.

Models used:
  fast / medium : damo-vilab/text-to-video-ms-1.7b  (~3.4 GB, Apache 2.0)
  high          : THUDM/CogVideoX-2b                 (~9 GB,   Apache 2.0)

Both run on DirectML (AMD Radeon 760M) or CPU with float16 + model offloading.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from .device_manager import DeviceInfo, DeviceManager, DeviceType
from .utils import MemoryManager, ensure_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "fast": {
        "model_id":            "damo-vilab/text-to-video-ms-1.7b",
        "engine":              "modelscope",
        "num_frames":          16,
        "num_inference_steps": 25,
        "height":              256,
        "width":               256,
        "fps":                 8,
    },
    "medium": {
        "model_id":            "damo-vilab/text-to-video-ms-1.7b",
        "engine":              "modelscope",
        "num_frames":          24,
        "num_inference_steps": 40,
        "height":              256,
        "width":               256,
        "fps":                 8,
    },
    "high": {
        "model_id":            "THUDM/CogVideoX-2b",
        "engine":              "cogvideox",
        "num_frames":          49,
        "num_inference_steps": 50,
        "height":              480,
        "width":               720,
        "fps":                 8,
    },
}


# ---------------------------------------------------------------------------
# TextToVideoGenerator
# ---------------------------------------------------------------------------

class TextToVideoGenerator:
    def __init__(
        self,
        quality: str = "medium",
        device_manager: Optional[DeviceManager] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.quality        = quality
        self.device_manager = device_manager or DeviceManager()
        self.cache_dir      = cache_dir
        self._pipeline      = None
        self._current_model: Optional[str] = None
        self._device_info: Optional[DeviceInfo] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        output_path: Union[str, Path],
        negative_prompt: str = "blurry, low quality, distorted, watermark",
        seed: Optional[int] = None,
    ) -> Path:
        """Generate a video clip from a text prompt.

        Args:
            prompt:          Describe what you want to see.
            negative_prompt: What to avoid (CFG guidance).
            seed:            For reproducibility.

        Returns:
            Path to the written .mp4 file.
        """
        preset = PRESETS[self.quality]
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        if self._pipeline is None or self._current_model != preset["model_id"]:
            self._load_pipeline(preset)

        import torch
        generator = torch.manual_seed(seed) if seed is not None else None

        logger.info(
            "T2V: '%s…' | model=%s | %dx%d | %d frames | %d steps",
            prompt[:60], preset["engine"],
            preset["width"], preset["height"],
            preset["num_frames"], preset["num_inference_steps"],
        )

        with MemoryManager():
            if preset["engine"] == "cogvideox":
                frames = self._run_cogvideox(prompt, negative_prompt, preset, generator)
            else:
                frames = self._run_modelscope(prompt, negative_prompt, preset, generator)

        _frames_to_mp4(frames, output_path, fps=preset["fps"])
        logger.info("T2V saved → %s", output_path)
        return output_path

    def unload(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        MemoryManager().__exit__(None, None, None)
        logger.info("T2V pipeline unloaded")

    # ------------------------------------------------------------------
    # Internal — pipeline loading
    # ------------------------------------------------------------------

    def _load_pipeline(self, preset: dict) -> None:
        import torch

        self._device_info = self.device_manager.detect()
        dtype = torch.float16 if self._device_info.supports_float16 else torch.float32

        kw: dict = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
        if self.cache_dir:
            kw["cache_dir"] = str(self.cache_dir)

        model_id = preset["model_id"]
        logger.info("Loading T2V model: %s (dtype=%s)", model_id, dtype)

        if preset["engine"] == "cogvideox":
            from diffusers import CogVideoXPipeline
            pipe = CogVideoXPipeline.from_pretrained(model_id, **kw)
        else:
            # ModelScope text-to-video uses the standard DiffusionPipeline
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(model_id, **kw)

        pipe.enable_attention_slicing()

        if self._device_info.device_type == DeviceType.DIRECTML:
            import torch_directml
            pipe = pipe.to(torch_directml.device())
        elif self._device_info.device_type == DeviceType.CUDA:
            pipe = pipe.to("cuda")
            pipe.enable_model_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()

        self._pipeline = pipe
        self._current_model = model_id
        logger.info("T2V model ready on %s", self._device_info.device_name)

    # ------------------------------------------------------------------
    # Internal — inference
    # ------------------------------------------------------------------

    def _run_modelscope(self, prompt, neg_prompt, preset, generator):
        result = self._pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_frames=preset["num_frames"],
            height=preset["height"],
            width=preset["width"],
            num_inference_steps=preset["num_inference_steps"],
            generator=generator,
        )
        # ModelScope returns .frames as list[list[PIL.Image]]
        frames = result.frames[0] if hasattr(result.frames[0], "__len__") else result.frames
        return frames

    def _run_cogvideox(self, prompt, neg_prompt, preset, generator):
        import torch
        result = self._pipeline(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_frames=preset["num_frames"],
            height=preset["height"],
            width=preset["width"],
            num_inference_steps=preset["num_inference_steps"],
            generator=generator,
            guidance_scale=6.0,
        )
        # CogVideoX returns .frames[0] as a list of PIL Images
        return result.frames[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frames_to_mp4(frames, output_path: Path, fps: int = 8) -> None:
    import imageio
    import numpy as np
    from PIL import Image

    numpy_frames = []
    for f in frames:
        if isinstance(f, Image.Image):
            numpy_frames.append(np.array(f))
        else:
            numpy_frames.append(np.array(f))

    with imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    ) as w:
        for frame in numpy_frames:
            w.append_data(frame)
