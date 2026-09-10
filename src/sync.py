"""FFmpeg-based audio/video synchronization utilities."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

from .utils import check_ffmpeg, get_video_duration

logger = logging.getLogger(__name__)


def sync_audio_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    loop_audio: bool = True,
    audio_volume: float = 1.0,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> Path:
    """Mux video and audio, trimming or looping audio to match video duration.

    Args:
        loop_audio: If True and audio is shorter than video, loop it to fill.
                    If False, remaining video plays silently.
        audio_volume: Multiplier (1.0 = unchanged, 0.5 = quieter).
        video_codec: 'copy' preserves original video quality with no re-encode.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg/ffprobe not found on PATH. See TROUBLESHOOTING.md.")

    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_duration = get_video_duration(video_path)
    logger.info("Syncing video (%.1fs) + audio → %s", video_duration, output_path)

    # Build filter chain for audio
    audio_filters = []
    if loop_audio:
        # aloop=-1 loops indefinitely; atrim caps to video length
        audio_filters.append(f"aloop=loop=-1:size=2147483647")
    audio_filters.append(f"atrim=duration={video_duration:.3f}")
    if audio_volume != 1.0:
        audio_filters.append(f"volume={audio_volume}")
    audio_filter = ",".join(audio_filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-filter_complex", f"[1:a]{audio_filter}[a]",
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", video_codec,
        "-c:a", audio_codec,
        "-b:a", audio_bitrate,
        "-shortest",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    logger.info("Synced output saved → %s", output_path)
    return output_path


def add_subtitles(
    video_path: Union[str, Path],
    srt_path: Union[str, Path],
    output_path: Union[str, Path],
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
) -> Path:
    """Burn SRT subtitles into video (re-encodes video stream)."""
    video_path = Path(video_path)
    srt_path = Path(srt_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Escape Windows-style paths for ffmpeg filter syntax
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")

    sub_filter = (
        f"subtitles={srt_escaped}:force_style='"
        f"FontSize={font_size},PrimaryColour=&H{_hex_color(font_color)}&,"
        f"OutlineColour=&H{_hex_color(outline_color)}&,Outline=2'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", sub_filter,
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subtitle burn failed:\n{result.stderr}")
    return output_path


def extract_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    codec: str = "mp3",
) -> Path:
    """Extract audio track from a video file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-acodec", codec,
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extract failed:\n{result.stderr}")
    return output_path


def trim_video(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    start: float = 0.0,
    duration: Optional[float] = None,
    end: Optional[float] = None,
) -> Path:
    """Trim video between [start, start+duration] or [start, end]."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path), "-ss", str(start)]
    if duration:
        cmd += ["-t", str(duration)]
    elif end:
        cmd += ["-t", str(end - start)]
    cmd += ["-c", "copy", str(output_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed:\n{result.stderr}")
    return output_path


def _hex_color(name: str) -> str:
    """Convert simple color name to ffmpeg ASS color hex (BBGGRR, no alpha)."""
    colors = {
        "white": "FFFFFF",
        "black": "000000",
        "yellow": "00FFFF",
        "red": "0000FF",
        "blue": "FF0000",
        "green": "00FF00",
    }
    return colors.get(name.lower(), "FFFFFF")
