"""Local TTS using Kokoro — free, offline, no API key required.

Model: hexgrad/Kokoro-82M (MIT license)
Package: kokoro (official) — uses KPipeline, auto-downloads model + voices
Quality: near-commercial, real-time on CPU / fast on GPU
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice catalogue  (friendly_name → Kokoro voice ID)
# ---------------------------------------------------------------------------

VOICES: dict[str, str] = {
    # American English — Female
    "heart":    "af_heart",
    "bella":    "af_bella",
    "sarah":    "af_sarah",
    "sky":      "af_sky",
    "nicole":   "af_nicole",
    # American English — Male
    "adam":     "am_adam",
    "michael":  "am_michael",
    "puck":     "am_puck",
    "echo":     "am_echo",
    # British English — Female
    "emma":     "bf_emma",
    "isabella": "bf_isabella",
    # British English — Male
    "george":   "bm_george",
    "lewis":    "bm_lewis",
}

DEFAULT_VOICE = "heart"

def _lang_code(voice_id: str) -> str:
    """Derive KPipeline lang_code from voice ID prefix."""
    return "b" if voice_id.startswith("b") else "a"


# ---------------------------------------------------------------------------
# TTSGenerator
# ---------------------------------------------------------------------------

class TTSGenerator:
    """Local text-to-speech via the official kokoro package (KPipeline)."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,  # kept for API compat, unused
        default_voice: str = DEFAULT_VOICE,
    ) -> None:
        self.default_voice = default_voice
        self._pipelines: dict[str, object] = {}  # lang_code → KPipeline

    # ------------------------------------------------------------------
    # Setup (kept for backward compat — kokoro auto-downloads on first use)
    # ------------------------------------------------------------------

    def download_models(self) -> None:
        """Trigger a model download by running a short synthesis."""
        logger.info("Pre-loading Kokoro model (first run downloads ~300 MB) …")
        self._get_pipeline("a")
        logger.info("Kokoro ready")

    def _get_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            try:
                from kokoro import KPipeline
            except ImportError as exc:
                raise RuntimeError(
                    "kokoro not installed. Run: pip install kokoro"
                ) from exc
            logger.info("Loading KPipeline lang_code=%s …", lang_code)
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
            logger.info("KPipeline(%s) ready", lang_code)
        return self._pipelines[lang_code]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_speech(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang: str = "en-us",  # kept for compat — derived from voice ID
    ) -> Path:
        import numpy as np
        voice_id = VOICES.get(voice.lower(), voice)
        pipeline  = self._get_pipeline(_lang_code(voice_id))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("TTS: %d chars, voice=%s, speed=%.1f", len(text), voice_id, speed)
        chunks = []
        for _, _, audio in pipeline(text, voice=voice_id, speed=speed):
            chunks.append(audio)

        combined = np.concatenate(chunks) if chunks else np.zeros(24000, dtype=np.float32)
        _save_audio(combined, 24000, output_path)
        logger.info("TTS saved → %s", output_path)
        return output_path

    def generate_speech_chunked(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang: str = "en-us",
        chunk_size: int = 300,  # kept for compat — KPipeline handles splitting
    ) -> Path:
        # KPipeline handles long text natively; just call generate_speech
        return self.generate_speech(text, output_path, voice=voice, speed=speed)

    def list_voices(self) -> list[dict]:
        rows = []
        for name, vid in VOICES.items():
            lang   = "British" if vid.startswith("b") else "American"
            gender = "Female" if "_f" in vid else "Male"
            rows.append({"id": vid, "name": name.capitalize(), "lang": lang, "gender": gender})
        return rows

    def unload(self) -> None:
        self._pipelines.clear()
        logger.info("Kokoro TTS unloaded")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_audio(samples, sample_rate: int, output_path: Path) -> None:
    if output_path.suffix.lower() == ".mp3":
        wav_tmp = output_path.with_suffix(".wav")
        _write_wav(wav_tmp, samples, sample_rate)
        _wav_to_mp3(wav_tmp, output_path)
        wav_tmp.unlink(missing_ok=True)
    else:
        _write_wav(output_path, samples, sample_rate)


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    try:
        import soundfile as sf
        sf.write(str(path), samples, sample_rate)
    except ImportError:
        from scipy.io import wavfile
        import numpy as np
        wavfile.write(str(path), sample_rate, (samples * 32767).astype(np.int16))


def _wav_to_mp3(wav: Path, mp3: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-q:a", "2", str(mp3)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        import shutil
        shutil.copy(wav, mp3)
        logger.warning("ffmpeg unavailable; audio saved as WAV at %s", mp3)


def _split_sentences(text: str, max_chars: int) -> list[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks
