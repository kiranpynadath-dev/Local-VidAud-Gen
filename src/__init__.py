"""Local Video and Audio Generation System for AMD Ryzen AI hardware."""
__version__ = "1.0.0"

from .generator import VideoAudioGenerator
from .device_manager import DeviceManager, DeviceType
from .video_processor import VideoProcessor, QUALITY_PRESETS
from .text_to_video import TextToVideoGenerator
from .video_editor import VideoEditor
from .tts_generator import TTSGenerator, VOICES as TTS_VOICES
from .batch_processor import BatchProcessor
from .sync import sync_audio_video

__all__ = [
    "VideoAudioGenerator",
    "DeviceManager",
    "DeviceType",
    "VideoProcessor",
    "QUALITY_PRESETS",
    "TextToVideoGenerator",
    "VideoEditor",
    "TTSGenerator",
    "TTS_VOICES",
    "BatchProcessor",
    "sync_audio_video",
]
