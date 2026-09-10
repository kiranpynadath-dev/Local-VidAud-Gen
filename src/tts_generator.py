"""Local TTS using Kokoro ONNX — free, offline, no API key required.

Model: hexgrad/Kokoro-82M (~310 MB, MIT license)
Quality: near-commercial, real-time on CPU
Setup: auto-downloads on first use via huggingface_hub
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
    "heart":    "af_heart",     # warm, friendly (recommended default)
    "bella":    "af_bella",     # expressive, emotive
    "sarah":    "af_sarah",     # professional, clear
    "sky":      "af_sky",       # bright, energetic
    "nicole":   "af_nicole",    # neutral, natural
    # American English — Male
    "adam":     "am_adam",      # deep, authoritative
    "michael":  "am_michael",   # casual, warm
    "puck":     "am_puck",      # energetic, upbeat
    "echo":     "am_echo",      # smooth, radio-style
    # British English — Female
    "emma":     "bf_emma",      # clear, professional
    "isabella": "bf_isabella",  # warm, British
    # British English — Male
    "george":   "bm_george",    # authoritative, British
    "lewis":    "bm_lewis",     # casual, British
}

DEFAULT_VOICE  = "heart"
MODEL_DIR      = Path("models/kokoro")
HF_REPO        = "hexgrad/Kokoro-82M"


# ---------------------------------------------------------------------------
# TTSGenerator
# ---------------------------------------------------------------------------

class TTSGenerator:
    """Local text-to-speech via Kokoro ONNX."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        default_voice: str = DEFAULT_VOICE,
    ) -> None:
        self.model_dir   = Path(model_dir) if model_dir else MODEL_DIR
        self.default_voice = default_voice
        self._kokoro = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _find_repo_files(self) -> tuple[str, str]:
        """Return (onnx_filename, voices_filename) by listing the HF repo live."""
        from huggingface_hub import list_repo_files
        files = list(list_repo_files(HF_REPO))
        onnx = next((f for f in files if f.endswith(".onnx") and "kokoro" in f), None)
        voices = next((f for f in files if "voices" in f and f.endswith(".bin")), None)
        if not onnx:
            raise RuntimeError(f"No .onnx model found in {HF_REPO}. Files: {files}")
        if not voices:
            raise RuntimeError(f"No voices.bin found in {HF_REPO}. Files: {files}")
        return onnx, voices

    def download_models(self) -> None:
        """Download Kokoro model + voice files from Hugging Face (once only)."""
        from huggingface_hub import hf_hub_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        onnx_file, voices_file = self._find_repo_files()
        logger.info("Kokoro repo files: model=%s  voices=%s", onnx_file, voices_file)

        for filename, size in ((onnx_file, "~310 MB"), (voices_file, "~2 MB")):
            dest = self.model_dir / Path(filename).name
            if not dest.exists():
                logger.info("Downloading %s (%s) …", filename, size)
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename=filename,
                    local_dir=str(self.model_dir),
                )
                logger.info("Saved → %s", dest)
            else:
                logger.debug("Already exists: %s", dest)

    def _ensure_loaded(self) -> None:
        if self._kokoro is not None:
            return
        # Find whichever model file is present locally, or download
        existing_onnx = next(self.model_dir.glob("*.onnx"), None) if self.model_dir.exists() else None
        existing_voices = next(self.model_dir.glob("voices*.bin"), None) if self.model_dir.exists() else None
        if not existing_onnx or not existing_voices:
            logger.info("Kokoro model not found locally — downloading …")
            self.download_models()
            existing_onnx = next(self.model_dir.glob("*.onnx"), None)
            existing_voices = next(self.model_dir.glob("voices*.bin"), None)
        try:
            from kokoro_onnx import Kokoro
            self._kokoro = Kokoro(str(existing_onnx), str(existing_voices))
            logger.info("Kokoro TTS loaded: %s", existing_onnx.name)
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx not installed. Run: pip install kokoro-onnx soundfile"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_speech(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang: str = "en-us",
    ) -> Path:
        """Synthesise text → MP3/WAV file.

        Args:
            voice: Friendly name (see VOICES) or raw Kokoro voice ID like 'af_heart'.
            speed: Playback speed multiplier (0.5–2.0).
            lang:  Language code — 'en-us', 'en-gb', etc.

        Returns:
            Path to the written audio file.
        """
        self._ensure_loaded()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        voice_id = VOICES.get(voice.lower(), voice)   # accept name or raw ID
        logger.info("TTS: %d chars, voice=%s, speed=%.1f", len(text), voice_id, speed)

        samples, sample_rate = self._kokoro.create(
            text, voice=voice_id, speed=speed, lang=lang
        )

        # Write WAV then convert to MP3 with ffmpeg (or keep WAV as fallback)
        if output_path.suffix.lower() == ".mp3":
            wav_tmp = output_path.with_suffix(".wav")
            _write_wav(wav_tmp, samples, sample_rate)
            _wav_to_mp3(wav_tmp, output_path)
            wav_tmp.unlink(missing_ok=True)
        else:
            _write_wav(output_path, samples, sample_rate)

        logger.info("TTS saved → %s", output_path)
        return output_path

    def generate_speech_chunked(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang: str = "en-us",
        chunk_size: int = 300,
    ) -> Path:
        """Split long text into chunks and concatenate — avoids max-token limits."""
        import numpy as np

        self._ensure_loaded()
        voice_id = VOICES.get(voice.lower(), voice)
        chunks   = _split_sentences(text, chunk_size)
        all_samples: list = []
        rate = 24000  # Kokoro default

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            logger.debug("Chunk %d/%d: %d chars", i + 1, len(chunks), len(chunk))
            samples, rate = self._kokoro.create(
                chunk, voice=voice_id, speed=speed, lang=lang
            )
            all_samples.append(samples)

        combined = np.concatenate(all_samples) if all_samples else np.zeros(rate, dtype=np.float32)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix.lower() == ".mp3":
            wav_tmp = output_path.with_suffix(".wav")
            _write_wav(wav_tmp, combined, rate)
            _wav_to_mp3(wav_tmp, output_path)
            wav_tmp.unlink(missing_ok=True)
        else:
            _write_wav(output_path, combined, rate)

        logger.info("Chunked TTS saved → %s", output_path)
        return output_path

    def list_voices(self) -> list[dict]:
        rows = []
        for name, vid in VOICES.items():
            lang = "British" if vid.startswith("b") else "American"
            gender = "Female" if "_f" in vid else "Male"
            rows.append({"id": vid, "name": name.capitalize(), "lang": lang, "gender": gender})
        return rows

    def unload(self) -> None:
        self._kokoro = None
        logger.info("Kokoro TTS unloaded")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_wav(path: Path, samples, sample_rate: int) -> None:
    try:
        import soundfile as sf
        sf.write(str(path), samples, sample_rate)
    except ImportError:
        # scipy fallback
        from scipy.io import wavfile
        import numpy as np
        wavfile.write(str(path), sample_rate, (samples * 32767).astype(np.int16))


def _wav_to_mp3(wav: Path, mp3: Path) -> None:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav), "-q:a", "2", str(mp3)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # ffmpeg not available — keep WAV with .mp3 extension as last resort
        import shutil
        shutil.copy(wav, mp3)
        logger.warning("ffmpeg unavailable; audio saved as uncompressed WAV at %s", mp3)


def _split_sentences(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries to stay under Kokoro's token limit."""
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
