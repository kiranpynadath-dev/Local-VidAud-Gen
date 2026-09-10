"""Local/cloud TTS — tries kokoro (offline) first, falls back to edge-tts (free, online).

Primary:  kokoro (KPipeline) — fully offline, MIT, ~300 MB model download once
Fallback: edge-tts            — free, no API key, needs internet, zero install issues
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice catalogue
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

# edge-tts voice mapping (fallback)
_EDGE_VOICES: dict[str, str] = {
    "af_heart":    "en-US-JennyNeural",
    "af_bella":    "en-US-AriaNeural",
    "af_sarah":    "en-US-SaraNeural",
    "af_sky":      "en-US-NancyNeural",
    "af_nicole":   "en-US-MichelleNeural",
    "am_adam":     "en-US-GuyNeural",
    "am_michael":  "en-US-ChristopherNeural",
    "am_puck":     "en-US-EricNeural",
    "am_echo":     "en-US-RogerNeural",
    "bf_emma":     "en-GB-SoniaNeural",
    "bf_isabella": "en-GB-LibbyNeural",
    "bm_george":   "en-GB-RyanNeural",
    "bm_lewis":    "en-GB-ThomasNeural",
}

DEFAULT_VOICE = "heart"


def _lang_code(voice_id: str) -> str:
    return "b" if voice_id.startswith("b") else "a"


def _kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401
        return True
    except ImportError:
        return False


def _edge_available() -> bool:
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# TTSGenerator
# ---------------------------------------------------------------------------

class TTSGenerator:
    """TTS with automatic backend selection: kokoro (offline) → edge-tts (online)."""

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        default_voice: str = DEFAULT_VOICE,
    ) -> None:
        self.default_voice = default_voice
        self._pipelines: dict[str, object] = {}
        self._backend: Optional[str] = None

    # ------------------------------------------------------------------

    def download_models(self) -> None:
        backend = self._select_backend()
        if backend == "kokoro":
            logger.info("Pre-loading Kokoro model (~300 MB first run) …")
            self._get_kokoro_pipeline("a")
            logger.info("Kokoro ready")
        else:
            logger.info("Using edge-tts backend (online, no download needed)")

    def _select_backend(self) -> str:
        if self._backend:
            return self._backend
        if _kokoro_available():
            self._backend = "kokoro"
            logger.info("TTS backend: kokoro (offline)")
        elif _edge_available():
            self._backend = "edge"
            logger.info("TTS backend: edge-tts (online)")
        else:
            raise RuntimeError(
                "No TTS backend found. Run:  pip install kokoro  OR  pip install edge-tts"
            )
        return self._backend

    def _get_kokoro_pipeline(self, lang_code: str):
        if lang_code not in self._pipelines:
            from kokoro import KPipeline
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code)
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
        lang: str = "en-us",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice_id = VOICES.get(voice.lower(), voice)
        backend = self._select_backend()

        if backend == "kokoro":
            return self._generate_kokoro(text, output_path, voice_id, speed)
        else:
            return self._generate_edge(text, output_path, voice_id, speed)

    def generate_speech_chunked(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        lang: str = "en-us",
        chunk_size: int = 300,
    ) -> Path:
        return self.generate_speech(text, output_path, voice=voice, speed=speed)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _generate_kokoro(self, text, output_path, voice_id, speed) -> Path:
        import numpy as np
        pipeline = self._get_kokoro_pipeline(_lang_code(voice_id))
        chunks = [audio for _, _, audio in pipeline(text, voice=voice_id, speed=speed)]
        combined = np.concatenate(chunks) if chunks else np.zeros(24000, dtype=np.float32)
        _save_audio(combined, 24000, output_path)
        logger.info("Kokoro TTS → %s", output_path)
        return output_path

    def _generate_edge(self, text, output_path, voice_id, speed) -> Path:
        import edge_tts, tempfile, shutil
        edge_voice = _EDGE_VOICES.get(voice_id, "en-US-JennyNeural")
        rate = f"+{int((speed-1)*100)}%" if speed >= 1 else f"{int((speed-1)*100)}%"

        # edge-tts is async — run in event loop
        async def _run():
            comm = edge_tts.Communicate(text, voice=edge_voice, rate=rate)
            tmp = output_path.with_suffix(".tmp.mp3")
            await comm.save(str(tmp))
            return tmp

        tmp = asyncio.get_event_loop().run_until_complete(_run())

        if output_path.suffix.lower() == ".mp3":
            shutil.move(str(tmp), str(output_path))
        else:
            # convert to wav
            subprocess.run(["ffmpeg", "-y", "-i", str(tmp), str(output_path)],
                           capture_output=True)
            tmp.unlink(missing_ok=True)

        logger.info("edge-tts → %s", output_path)
        return output_path

    def list_voices(self) -> list[dict]:
        rows = []
        for name, vid in VOICES.items():
            lang   = "British" if vid.startswith("b") else "American"
            gender = "Female" if "_f" in vid else "Male"
            rows.append({"id": vid, "name": name.capitalize(), "lang": lang, "gender": gender})
        return rows

    def unload(self) -> None:
        self._pipelines.clear()
        self._backend = None


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
        logger.warning("ffmpeg unavailable; kept WAV at %s", mp3)


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
