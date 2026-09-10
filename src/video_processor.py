"""Stable Video Diffusion pipeline with DirectML / CUDA / CPU support."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .device_manager import DeviceInfo, DeviceManager, DeviceType
from .utils import MemoryManager, ensure_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------

QUALITY_PRESETS: dict[str, dict] = {
    "fast": {
        "num_frames": 14,
        "num_inference_steps": 10,
        "decode_chunk_size": 2,
        "motion_bucket_id": 127,
        "noise_aug_strength": 0.02,
        "fps": 6,
        "width": 576,
        "height": 320,
    },
    "medium": {
        "num_frames": 25,
        "num_inference_steps": 20,
        "decode_chunk_size": 4,
        "motion_bucket_id": 100,
        "noise_aug_strength": 0.02,
        "fps": 12,
        "width": 1024,
        "height": 576,
    },
    "high": {
        "num_frames": 25,
        "num_inference_steps": 30,
        "decode_chunk_size": 8,
        "motion_bucket_id": 80,
        "noise_aug_strength": 0.01,
        "fps": 24,
        "width": 1024,
        "height": 576,
    },
}

SVD_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------


class VideoProcessor:
    def __init__(
        self,
        device_manager: Optional[DeviceManager] = None,
        quality: str = "medium",
        model_id: str = SVD_MODEL_ID,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.device_manager = device_manager or DeviceManager()
        self.quality = quality
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._pipeline = None
        self._device_info: Optional[DeviceInfo] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        import torch
        from diffusers import StableVideoDiffusionPipeline

        self._device_info = self.device_manager.detect()
        dtype = torch.float16 if self._device_info.supports_float16 else torch.float32

        logger.info("Loading SVD model '%s' (dtype=%s) on %s", self.model_id, dtype, self._device_info.device_name)

        kwargs: dict = dict(
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)

        pipe = StableVideoDiffusionPipeline.from_pretrained(self.model_id, **kwargs)
        pipe.enable_attention_slicing()

        if self._device_info.device_type == DeviceType.DIRECTML:
            import torch_directml
            pipe = pipe.to(torch_directml.device())
        elif self._device_info.device_type == DeviceType.CUDA:
            pipe = pipe.to("cuda")
            pipe.enable_model_cpu_offload()
        elif self._device_info.device_type == DeviceType.MPS:
            pipe = pipe.to("mps")
        else:
            # CPU or NPU (NPU runs UNet via ONNX; PyTorch pipeline on CPU)
            pipe.enable_model_cpu_offload()

        self._pipeline = pipe
        logger.info("Model loaded successfully")

    def generate_from_image(
        self,
        image: Union[str, Path, Image.Image],
        output_path: Union[str, Path],
        seed: Optional[int] = None,
    ) -> Path:
        """Generate a video clip from a single conditioning image.

        Returns:
            Path to the written .mp4 file.
        """
        if self._pipeline is None:
            self.load_model()

        import torch

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        preset = QUALITY_PRESETS[self.quality]
        image = image.resize((preset["width"], preset["height"]), Image.LANCZOS)

        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        generator = torch.manual_seed(seed) if seed is not None else None

        logger.info(
            "Generating %d frames @ %d steps (quality=%s)", preset["num_frames"], preset["num_inference_steps"], self.quality
        )

        with MemoryManager():
            result = self._pipeline(
                image,
                num_frames=preset["num_frames"],
                num_inference_steps=preset["num_inference_steps"],
                decode_chunk_size=preset["decode_chunk_size"],
                motion_bucket_id=preset["motion_bucket_id"],
                noise_aug_strength=preset["noise_aug_strength"],
                generator=generator,
            )

        frames_to_video(result.frames[0], output_path, fps=preset["fps"])
        logger.info("Video saved → %s", output_path)
        return output_path

    def generate_from_video(
        self,
        video_path: Union[str, Path],
        output_path: Union[str, Path],
        seed: Optional[int] = None,
    ) -> Path:
        """Use the first frame of an existing video as the conditioning image."""
        first_frame = extract_first_frame(Path(video_path))
        return self.generate_from_image(first_frame, output_path, seed=seed)

    def unload_model(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        MemoryManager().__exit__(None, None, None)
        logger.info("Model unloaded, memory released")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def frames_to_video(frames: list, output_path: Path, fps: int = 10) -> None:
    """Write a list of PIL Images to an mp4 via imageio."""
    import imageio
    import numpy as np

    numpy_frames = [np.array(f) for f in frames]
    writer_kwargs = {"fps": fps, "codec": "libx264", "quality": 8, "pixelformat": "yuv420p"}

    with imageio.get_writer(str(output_path), **writer_kwargs) as writer:
        for frame in numpy_frames:
            writer.append_data(frame)


def extract_first_frame(video_path: Path) -> Image.Image:
    """Return the first frame of a video as a PIL Image."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read video: {video_path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
