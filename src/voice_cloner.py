"""Voice cloning via Chatterbox TTS (Resemble AI) — Python 3.13 compatible."""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_model = None  # lazy singleton


def _load_model(device: str = "cuda"):
    global _model
    if _model is not None:
        return _model
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise RuntimeError(
            "chatterbox-tts not installed. Run:  pip install chatterbox-tts  then restart the server."
        )
    logger.info("Loading Chatterbox TTS model…")
    _model = ChatterboxTTS.from_pretrained(device=device)
    logger.info("Chatterbox TTS ready")
    return _model


def clone_voice(
    text: str,
    reference_audio: Union[str, Path],
    output_path: Union[str, Path],
    language: str = "en",  # kept for API compat, Chatterbox is English-only
    device: str = "cuda",
) -> Path:
    """Synthesise text in the voice from reference_audio (WAV/MP3, 6-10s ideal)."""
    import torch
    reference_audio = Path(reference_audio)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        device = "cpu"

    # Chatterbox needs WAV input
    ref_wav = reference_audio
    if reference_audio.suffix.lower() != ".wav":
        ref_wav = reference_audio.with_suffix(".tmp.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(reference_audio), str(ref_wav)],
            capture_output=True, check=True,
        )

    model = _load_model(device)
    wav = model.generate(text, audio_prompt_path=str(ref_wav))

    if ref_wav != reference_audio:
        ref_wav.unlink(missing_ok=True)

    # Save output
    wav_out = output_path.with_suffix(".wav")
    import torchaudio
    torchaudio.save(str(wav_out), wav, model.sr)

    if output_path.suffix.lower() == ".mp3":
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_out), "-q:a", "2", str(output_path)],
            capture_output=True,
        )
        wav_out.unlink(missing_ok=True)
    else:
        wav_out.rename(output_path)

    logger.info("Cloned voice → %s", output_path)
    return output_path


def unload():
    global _model
    if _model is not None:
        del _model
        _model = None
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
