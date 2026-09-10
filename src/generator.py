"""VideoAudioGenerator — unified high-level API."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .batch_processor import BatchProcessor
from .device_manager import DeviceManager
from .sync import sync_audio_video
from .text_to_video import TextToVideoGenerator
from .tts_generator import TTSGenerator
from .utils import MemoryManager, ensure_dir, setup_logging, upscale_video
from .video_editor import VideoEditor
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class VideoAudioGenerator:
    """One-stop API for local video + audio generation.

    Audio uses Kokoro TTS (free, local, no API key).
    Video uses Stable Video Diffusion (SVD-XT, HuggingFace).

    Usage::

        gen = VideoAudioGenerator(quality="medium")
        result = gen.generate(
            image="photo.jpg",
            text="Welcome to our product demo.",
        )
        print(result["final"])  # → output/output_final.mp4
    """

    def __init__(
        self,
        quality: str = "medium",
        device: Optional[str] = None,
        output_dir: Union[str, Path] = "output",
        model_cache_dir: Optional[Union[str, Path]] = None,
        tts_model_dir: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ) -> None:
        """
        Args:
            quality:       'fast' | 'medium' | 'high'
            device:        Force device — 'npu' | 'directml' | 'cuda' | 'cpu' | None (auto).
            output_dir:    Root directory for generated files.
            model_cache_dir: Where to cache SVD model files (HuggingFace).
            tts_model_dir:  Where to store Kokoro TTS model files.
                            Defaults to models/kokoro/ inside the project.
            log_level:     Python logging level string.
        """
        setup_logging(log_level)

        self.output_dir = ensure_dir(Path(output_dir))
        self.quality    = quality

        self._device_manager = DeviceManager(force_device=device)
        self._video_processor = VideoProcessor(
            device_manager=self._device_manager,
            quality=quality,
            cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        )
        self._tts: Optional[TTSGenerator] = None
        self._tts_model_dir = Path(tts_model_dir) if tts_model_dir else None
        self._t2v: Optional[TextToVideoGenerator] = None
        self._editor: Optional[VideoEditor] = None

    # ------------------------------------------------------------------
    # Main public methods
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Union[str, Path, Image.Image],
        text: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        voice: str = "heart",
        speed: float = 1.0,
        seed: Optional[int] = None,
        loop_audio: bool = True,
        filename_prefix: str = "output",
    ) -> dict[str, Optional[Path]]:
        """Generate a video from an image + optional TTS audio, then sync.

        Returns:
            dict with keys 'video', 'audio', 'final'
        """
        out        = ensure_dir(Path(output_dir) if output_dir else self.output_dir)
        video_path = out / f"{filename_prefix}_video.mp4"
        audio_path: Optional[Path] = None
        final_path: Optional[Path] = None

        # ── Video ──────────────────────────────────────────────────────
        logger.info("=== Video Generation ===")
        video_path = self._video_processor.generate_from_image(image, video_path, seed=seed)
        if self.quality in ("medium", "high"):
            logger.info("=== Upscaling to 1080p 60fps ===")
            video_path = upscale_video(video_path, video_path)

        # ── Audio (Kokoro TTS) ─────────────────────────────────────────
        if text:
            logger.info("=== Audio Generation (Kokoro TTS) ===")
            audio_path = out / f"{filename_prefix}_audio.mp3"
            audio_path = self._get_tts().generate_speech_chunked(
                text, audio_path, voice=voice, speed=speed
            )

        # ── Sync ───────────────────────────────────────────────────────
        if audio_path:
            logger.info("=== A/V Sync ===")
            final_path = out / f"{filename_prefix}_final.mp4"
            sync_audio_video(video_path, audio_path, final_path, loop_audio=loop_audio)
        else:
            final_path = video_path

        logger.info("Done → %s", final_path)
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
        voice: str = "heart",
        speed: float = 1.0,
    ) -> Path:
        return self._get_tts().generate_speech_chunked(
            text, output_path, voice=voice, speed=speed
        )

    def sync(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        **kwargs,
    ) -> Path:
        return sync_audio_video(video_path, audio_path, output_path, **kwargs)

    # ------------------------------------------------------------------
    # Text → Video
    # ------------------------------------------------------------------

    def generate_from_text(
        self,
        prompt: str,
        output_dir: Optional[Union[str, Path]] = None,
        negative_prompt: str = "blurry, low quality, distorted, watermark",
        text: Optional[str] = None,
        voice: str = "heart",
        speed: float = 1.0,
        seed: Optional[int] = None,
        loop_audio: bool = True,
        filename_prefix: str = "t2v",
    ) -> dict[str, Optional[Path]]:
        """Generate a video from a text description, then optionally add TTS audio.

        Args:
            prompt:   Visual description for the video ("a sunset over the ocean").
            text:     Script for TTS voiceover. If None, video is silent.
        """
        out        = ensure_dir(Path(output_dir) if output_dir else self.output_dir)
        video_path = out / f"{filename_prefix}_video.mp4"
        audio_path: Optional[Path] = None
        final_path: Optional[Path] = None

        logger.info("=== Text-to-Video Generation ===")
        if self._t2v is None:
            self._t2v = TextToVideoGenerator(
                quality=self.quality,
                device_manager=self._device_manager,
                cache_dir=Path(self._video_processor.cache_dir) if self._video_processor.cache_dir else None,
            )
        video_path = self._t2v.generate(
            prompt=prompt,
            output_path=video_path,
            negative_prompt=negative_prompt,
            seed=seed,
        )
        if self.quality in ("medium", "high"):
            logger.info("=== Upscaling to 1080p 60fps ===")
            video_path = upscale_video(video_path, video_path)

        if text:
            logger.info("=== TTS Audio ===")
            audio_path = out / f"{filename_prefix}_audio.mp3"
            audio_path = self._get_tts().generate_speech_chunked(
                text, audio_path, voice=voice, speed=speed
            )
            logger.info("=== A/V Sync ===")
            final_path = out / f"{filename_prefix}_final.mp4"
            sync_audio_video(video_path, audio_path, final_path, loop_audio=loop_audio)
        else:
            final_path = video_path

        logger.info("Done → %s", final_path)
        return {"video": video_path, "audio": audio_path, "final": final_path}

    # ------------------------------------------------------------------
    # Video editing with text prompt
    # ------------------------------------------------------------------

    def edit_video(
        self,
        video_path: Union[str, Path],
        prompt: str,
        output_dir: Optional[Union[str, Path]] = None,
        mode: str = "sdedit",
        negative_prompt: str = "blurry, low quality, artifacts",
        strength: float = 0.55,
        steps: int = 20,
        max_frames: Optional[int] = None,
        seed: Optional[int] = 42,
        filename_prefix: str = "edit",
    ) -> Path:
        """Edit an existing video using a text prompt.

        Args:
            mode:       'sdedit'   — style/atmosphere changes.
                        'instruct' — instruction-style edits ("replace X with Y").
            strength:   How strongly to apply the edit (0.0–1.0).
            max_frames: Process only first N frames (useful for quick tests).
        """
        out = ensure_dir(Path(output_dir) if output_dir else self.output_dir)
        output_path = out / f"{filename_prefix}_edited.mp4"

        if self._editor is None:
            self._editor = VideoEditor(
                device_manager=self._device_manager,
                cache_dir=Path(self._video_processor.cache_dir) if self._video_processor.cache_dir else None,
            )

        logger.info("=== Video Edit | mode=%s | strength=%.2f ===", mode, strength)
        return self._editor.edit(
            video_path=video_path,
            prompt=prompt,
            output_path=output_path,
            mode=mode,
            negative_prompt=negative_prompt,
            strength=strength,
            steps=steps,
            max_frames=max_frames,
            seed=seed,
        )

    def create_batch_processor(self, max_workers: int = 1) -> BatchProcessor:
        return BatchProcessor(max_workers=max_workers)

    def device_info(self) -> dict:
        info = self._device_manager.detect()
        return {
            "device": info.device_name,
            "type":   info.device_type.value,
            "memory_gb": info.memory_gb,
            "float16":   info.supports_float16,
            "torch_device": info.torch_device,
        }

    def list_voices(self) -> list[dict]:
        return self._get_tts().list_voices()

    def unload(self) -> None:
        """Release all loaded models and free memory."""
        self._video_processor.unload_model()
        if self._tts:
            self._tts.unload()
            self._tts = None
        if self._t2v:
            self._t2v.unload()
            self._t2v = None
        if self._editor:
            self._editor.unload()
            self._editor = None
        MemoryManager().__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_tts(self) -> TTSGenerator:
        if self._tts is None:
            self._tts = TTSGenerator(model_dir=self._tts_model_dir)
        return self._tts
