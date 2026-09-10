"""Text-prompt video editing — two modes:

  SDEdit (img2img)       — broad style / scene changes ("make it rainy", "turn to night")
  InstructPix2Pix        — surgical instruction edits ("replace the car with a bicycle")

Both work frame-by-frame with temporal consistency via frame-to-frame seeding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .device_manager import DeviceInfo, DeviceManager, DeviceType
from .utils import MemoryManager, ensure_dir

logger = logging.getLogger(__name__)

# SD 1.5 is small (~4 GB), fast, DirectML-compatible
SD_MODEL   = "runwayml/stable-diffusion-v1-5"
IPX2P_MODEL = "timbrooks/instruct-pix2pix"


class VideoEditor:
    """Edit a video using a text prompt, frame by frame.

    Usage::

        editor = VideoEditor()
        out = editor.edit(
            video_path="raw.mp4",
            prompt="make the background a starry night sky",
            output_path="edited.mp4",
            mode="sdedit",   # or "instruct"
            strength=0.55,
        )
    """

    def __init__(
        self,
        device_manager: Optional[DeviceManager] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.device_manager = device_manager or DeviceManager()
        self.cache_dir      = cache_dir
        self._pipeline      = None
        self._loaded_mode: Optional[str] = None
        self._device_info: Optional[DeviceInfo] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def edit(
        self,
        video_path: Union[str, Path],
        prompt: str,
        output_path: Union[str, Path],
        mode: str = "sdedit",
        negative_prompt: str = "blurry, low quality, artifacts, watermark",
        strength: float = 0.55,
        guidance_scale: float = 7.5,
        steps: int = 20,
        max_frames: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> Path:
        """Apply a text edit to every frame of a video.

        Args:
            mode:      'sdedit'   — style/scene changes via img2img noise injection.
                       'instruct' — instruction-following via InstructPix2Pix.
            strength:  For sdedit: 0.0 = no change, 1.0 = ignore original.
                       For instruct: image_guidance_scale (1.0–2.5 typical).
            max_frames: Process only the first N frames (faster for testing).
        """
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        frames, fps, (w, h) = _load_video_frames(Path(video_path))
        if max_frames:
            frames = frames[:max_frames]

        logger.info(
            "Editing %d frames | mode=%s | strength=%.2f | prompt='%s…'",
            len(frames), mode, strength, prompt[:50],
        )

        self._ensure_loaded(mode)

        edited: list[Image.Image] = []
        for i, frame in enumerate(frames):
            logger.debug("  frame %d/%d", i + 1, len(frames))
            frame_seed = (seed or 0) + i  # deterministic per-frame seed for temporal coherence
            out_frame = self._process_frame(
                frame, prompt, negative_prompt, mode,
                strength, guidance_scale, steps, frame_seed,
            )
            edited.append(out_frame.resize((w, h), Image.LANCZOS))

        _frames_to_mp4(edited, output_path, fps=fps)
        logger.info("Edited video saved → %s", output_path)
        return output_path

    def unload(self) -> None:
        if self._pipeline:
            del self._pipeline
            self._pipeline = None
        MemoryManager().__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self, mode: str) -> None:
        if self._pipeline and self._loaded_mode == mode:
            return

        import torch
        self._device_info = self.device_manager.detect()
        dtype = torch.float16 if self._device_info.supports_float16 else torch.float32
        kw: dict = dict(torch_dtype=dtype, low_cpu_mem_usage=True, safety_checker=None)
        if self.cache_dir:
            kw["cache_dir"] = str(self.cache_dir)

        if mode == "instruct":
            from diffusers import StableDiffusionInstructPix2PixPipeline
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IPX2P_MODEL, **kw)
            logger.info("Loaded InstructPix2Pix pipeline")
        else:
            from diffusers import StableDiffusionImg2ImgPipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL, **kw)
            logger.info("Loaded SDEdit img2img pipeline")

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
            pipe.enable_model_cpu_offload()

        self._pipeline    = pipe
        self._loaded_mode = mode

    def _process_frame(
        self,
        frame: Image.Image,
        prompt: str,
        negative_prompt: str,
        mode: str,
        strength: float,
        guidance_scale: float,
        steps: int,
        seed: int,
    ) -> Image.Image:
        import torch
        generator = torch.manual_seed(seed)

        # Work at a resolution SD handles well
        w, h = _round64(frame.width), _round64(frame.height)
        frame_resized = frame.resize((w, h), Image.LANCZOS)

        if mode == "instruct":
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=frame_resized,
                num_inference_steps=steps,
                image_guidance_scale=max(1.0, strength * 2.5),
                guidance_scale=guidance_scale,
                generator=generator,
            )
        else:
            result = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=frame_resized,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        return result.images[0]


# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def _load_video_frames(video_path: Path) -> tuple[list[Image.Image], float, tuple[int, int]]:
    """Return (frames, fps, (width, height))."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    logger.info("Loaded %d frames @ %.1f fps (%dx%d)", len(frames), fps, w, h)
    return frames, fps, (w, h)


def _frames_to_mp4(frames: list[Image.Image], output_path: Path, fps: float) -> None:
    import imageio, numpy as np
    with imageio.get_writer(
        str(output_path), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    ) as w:
        for f in frames:
            w.append_data(np.array(f))


def _round64(n: int) -> int:
    """Round to nearest multiple of 64 (SD requirement)."""
    return max(64, (n // 64) * 64)
