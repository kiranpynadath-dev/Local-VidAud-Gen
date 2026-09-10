"""VideoAudioGenerator — unified high-level API."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .audio_generator import AudioGenerator
from .batch_processor import BatchProcessor
from .device_manager import DeviceManager
from .sync import sync_audio_video
from .utils import MemoryManager, ensure_dir, setup_logging
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class VideoAudioGenerator:
    """One-stop API for local video + audio generation.

    Usage::

        gen = VideoAudioGenerator(quality="medium")
        result = gen.generate(
            image="photo.jpg",
            text="Welcome to our product demo.",
            output_dir="output/",
        )
        print(result["final"])  # path to synced .mp4
    """

    def __init__(
        self,
        quality: str = "medium",
        device: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        output_dir: Union[str, Path] = "output",
        model_cache_dir: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ) -> None:
        """
        Args:
            quality: 'fast' | 'medium' | 'high'
            device: Force device — 'npu' | 'directml' | 'cuda' | 'cpu' | None (auto).
            elevenlabs_api_key: If None, reads ELEVENLABS_API_KEY env var.
            output_dir: Root directory for generated files.
            model_cache_dir: Where to cache Hugging Face models.
            log_level: Python logging level string.
        """
        setup_logging(log_level)

        self.output_dir = ensure_dir(Path(output_dir))
        self.quality = quality

        self._device_manager = DeviceManager(force_device=device)
        self._video_processor = VideoProcessor(
            device_manager=self._device_manager,
            quality=quality,
            cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        )
        self._audio_generator: Optional[AudioGenerator] = None
        self._elevenlabs_key = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY", "")

    # ------------------------------------------------------------------
    # Main public methods
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Union[str, Path, Image.Image],
        text: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        voice: str = "rachel",
        seed: Optional[int] = None,
        loop_audio: bool = True,
        filename_prefix: str = "output",
    ) -> dict[str, Optional[Path]]:
        """Generate a video from an image + optional TTS audio, then sync.

        Returns:
            dict with keys: 'video', 'audio', 'final'
                - 'video'  — raw generated .mp4 (no audio)
                - 'audio'  — TTS .mp3 (None if text not provided)
                - 'final'  — synced .mp4 (equals 'video' if no audio)
        """
        out = ensure_dir(Path(output_dir) if output_dir else self.output_dir)

        video_path = out / f"{filename_prefix}_video.mp4"
        audio_path: Optional[Path] = None
        final_path: Optional[Path] = None

        # ---- Video generation ----------------------------------------
        logger.info("=== Video Generation ===")
        video_path = self._video_processor.generate_from_image(image, video_path, seed=seed)

        # ---- Audio generation ----------------------------------------
        if text:
            logger.info("=== Audio Generation ===")
            audio_path = out / f"{filename_prefix}_audio.mp3"
            audio_path = self._get_audio_generator().generate_speech(
                text, audio_path, voice=voice
            )

        # ---- Sync ----------------------------------------------------
        if audio_path:
            logger.info("=== Sync ===")
            final_path = out / f"{filename_prefix}_final.mp4"
            sync_audio_video(video_path, audio_path, final_path, loop_audio=loop_audio)
        else:
            final_path = video_path

        logger.info("Done. Final output: %s", final_path)
        return {"video": video_path, "audio": audio_path, "final": final_path}

    def generate_video_only(
        self,
        image: Union[str, Path, Image.Image],
        output_path: Union[str, Path],
        seed: Optional[int] = None,
    ) -> Path:
        return self._video_processor.generate_from_image(image, output_path, seed=seed)

    def generate_audio_only(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = "rachel",
        **kwargs,
    ) -> Path:
        return self._get_audio_generator().generate_speech(text, output_path, voice=voice, **kwargs)

    def sync(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs,
    ) -> Path:
        return sync_audio_video(video_path, audio_path, output_path, **kwargs)

    def create_batch_processor(self, max_workers: int = 1) -> BatchProcessor:
        return BatchProcessor(max_workers=max_workers)

    def device_info(self) -> dict:
        info = self._device_manager.detect()
        return {
            "device": info.device_name,
            "type": info.device_type.value,
            "memory_gb": info.memory_gb,
            "float16": info.supports_float16,
            "torch_device": info.torch_device,
        }

    def unload(self) -> None:
        """Release all loaded models and free memory."""
        self._video_processor.unload_model()
        self._audio_generator = None
        MemoryManager().__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_audio_generator(self) -> AudioGenerator:
        if self._audio_generator is None:
            if not self._elevenlabs_key:
                raise ValueError(
                    "ElevenLabs API key required for audio generation. "
                    "Set ELEVENLABS_API_KEY or pass elevenlabs_api_key= to VideoAudioGenerator."
                )
            self._audio_generator = AudioGenerator(api_key=self._elevenlabs_key)
        return self._audio_generator
