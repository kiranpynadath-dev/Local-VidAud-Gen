"""ElevenLabs text-to-speech integration with duration querying."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# Default voice IDs from ElevenLabs (public voices)
VOICE_PRESETS = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "clyde": "2EiwWnXFnvU5JabPnv8n",
    "domi": "AZnzlk1XvdvUeBnXmlld",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "sam": "yoZ06aMxZJJ28mfd3POQ",
}


class AudioGenerator:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY env var or pass api_key."
            )
        self._client = None

    # ------------------------------------------------------------------
    # Lazy client
    # ------------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            from elevenlabs.client import ElevenLabs

            self._client = ElevenLabs(api_key=self.api_key)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_voices(self) -> list[dict]:
        """Return list of {id, name, labels} for available voices."""
        resp = self.client.voices.get_all()
        return [
            {"id": v.voice_id, "name": v.name, "labels": v.labels or {}}
            for v in resp.voices
        ]

    def generate_speech(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = "rachel",
        model_id: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True,
    ) -> Path:
        """Convert text to speech and write to output_path (.mp3).

        Args:
            voice: Preset name (see VOICE_PRESETS) or raw voice_id string.
            model_id: ElevenLabs model. Use 'eleven_turbo_v2' for faster generation.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        voice_id = VOICE_PRESETS.get(voice.lower(), voice)

        logger.info("Generating speech: %d chars, voice=%s, model=%s", len(text), voice_id, model_id)

        from elevenlabs import VoiceSettings

        audio_iter = self.client.generate(
            text=text,
            voice=voice_id,
            model=model_id,
            voice_settings=VoiceSettings(
                stability=stability,
                similarity_boost=similarity_boost,
                style=style,
                use_speaker_boost=use_speaker_boost,
            ),
        )

        with open(output_path, "wb") as f:
            for chunk in audio_iter:
                if chunk:
                    f.write(chunk)

        duration = get_audio_duration(output_path)
        logger.info("Audio saved → %s (%.1fs)", output_path, duration)
        return output_path

    def generate_speech_to_match_duration(
        self,
        text: str,
        output_path: Union[str, Path],
        target_duration: float,
        voice: str = "rachel",
        **kwargs,
    ) -> Path:
        """Generate speech then trim/pad to match target_duration seconds."""
        raw_path = Path(output_path).with_suffix(".raw.mp3")
        self.generate_speech(text, raw_path, voice=voice, **kwargs)

        actual = get_audio_duration(raw_path)
        output_path = Path(output_path)

        if abs(actual - target_duration) < 0.5:
            raw_path.rename(output_path)
            return output_path

        _adjust_audio_duration(raw_path, output_path, target_duration)
        raw_path.unlink(missing_ok=True)
        return output_path


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


def get_audio_duration(path: Union[str, Path]) -> float:
    """Return audio duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if "duration" in stream:
            return float(stream["duration"])
    raise RuntimeError(f"Could not read duration from {path}")


def _adjust_audio_duration(src: Path, dst: Path, target: float) -> None:
    """Trim or pad (with silence) audio to target duration using ffmpeg."""
    actual = get_audio_duration(src)
    if actual > target:
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-t", str(target),
            "-acodec", "copy", str(dst),
        ]
    else:
        # Pad with silence
        pad = target - actual
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-af", f"apad=pad_dur={pad:.3f}",
            "-t", str(target),
            str(dst),
        ]
    subprocess.run(cmd, check=True, capture_output=True)
