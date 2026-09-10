"""Voice cloning via Coqui XTTS v2 — clone any voice from a 6-10s sample."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_xtts = None  # lazy singleton


def _load_xtts(device: str = "cuda"):
    global _xtts
    if _xtts is not None:
        return _xtts
    try:
        from TTS.api import TTS
    except ImportError as e:
        raise RuntimeError(
            f"Coqui TTS not importable ({e}). Run:  pip install TTS  then restart the server."
        )
    except Exception as e:
        raise RuntimeError(f"TTS import error: {e}")
    logger.info("Loading XTTS v2 model (~1.8 GB first run)…")
    _xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    logger.info("XTTS v2 ready")
    return _xtts


def clone_voice(
    text: str,
    reference_audio: Union[str, Path],
    output_path: Union[str, Path],
    language: str = "en",
    device: str = "cuda",
) -> Path:
    """Synthesise text in the voice from reference_audio (WAV/MP3, 6-10s ideal)."""
    import torch, shutil, subprocess
    reference_audio = Path(reference_audio)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # XTTS needs WAV input
    ref_wav = reference_audio
    if reference_audio.suffix.lower() != ".wav":
        ref_wav = reference_audio.with_suffix(".tmp.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(reference_audio), str(ref_wav)],
            capture_output=True, check=True,
        )

    if not torch.cuda.is_available():
        device = "cpu"

    tts = _load_xtts(device)

    wav_out = output_path.with_suffix(".wav")
    tts.tts_to_file(
        text=text,
        speaker_wav=str(ref_wav),
        language=language,
        file_path=str(wav_out),
    )

    if ref_wav != reference_audio:
        ref_wav.unlink(missing_ok=True)

    # Convert to MP3
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
    global _xtts
    if _xtts is not None:
        del _xtts
        _xtts = None
        try:
            import torch, gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
