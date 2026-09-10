"""Voice cloning:
- English: Chatterbox TTS (Python 3.13 compatible)
- Indian languages: edge-tts + OpenVoice v2 tone color conversion
"""
from __future__ import annotations
import logging
import subprocess
import asyncio
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

_chatterbox = None   # English cloning
_ov_converter = None  # OpenVoice v2 tone converter

# Indian language → edge-tts voice (neutral base for conversion)
_INDIAN_BASE_VOICES = {
    "hi": "hi-IN-SwaraNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "bn": "bn-IN-TanishaaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "pa": "pa-IN-OjasNeural",
    "ur": "ur-IN-GulNeural",
}


def _load_chatterbox(device: str):
    global _chatterbox
    if _chatterbox is not None:
        return _chatterbox
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        raise RuntimeError("chatterbox-tts not installed. Run: pip install chatterbox-tts")
    logger.info("Loading Chatterbox TTS…")
    _chatterbox = ChatterboxTTS.from_pretrained(device=device)
    return _chatterbox


def _openvoice_available() -> bool:
    try:
        import openvoice  # noqa: F401
        return True
    except ImportError:
        return False


def _load_openvoice(device: str):
    global _ov_converter
    if _ov_converter is not None:
        return _ov_converter
    from openvoice.api import ToneColorConverter
    ckpt = Path("checkpoints_v2/converter")
    if not ckpt.exists():
        logger.info("Downloading OpenVoice v2 checkpoints (~200 MB)…")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="myshell-ai/OpenVoiceV2",
                          local_dir="checkpoints_v2", ignore_patterns=["*.git*"])
    logger.info("Loading OpenVoice v2 tone converter…")
    _ov_converter = ToneColorConverter(str(ckpt / "config.json"), device=device)
    _ov_converter.load_ckpt(str(ckpt / "checkpoint.pth"))
    return _ov_converter


def _edge_tts_sync(text: str, voice: str, out_path: Path) -> None:
    """Generate audio via edge-tts (handles running event loop)."""
    import edge_tts

    async def _run():
        comm = edge_tts.Communicate(text, voice=voice)
        await comm.save(str(out_path))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            ex.submit(lambda: asyncio.run(_run())).result()
    else:
        asyncio.run(_run())


def clone_voice(
    text: str,
    reference_audio: Union[str, Path],
    output_path: Union[str, Path],
    language: str = "en",
    device: str = "cuda",
) -> Path:
    """Synthesise text in the voice from reference_audio.

    English:        Chatterbox TTS (direct voice cloning)
    Indian/other:   edge-tts base TTS + OpenVoice v2 tone color conversion
    """
    import torch
    reference_audio = Path(reference_audio)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        device = "cpu"

    lang = language.lower().split("-")[0]  # "hi-IN" → "hi"

    if lang == "en":
        return _clone_english(text, reference_audio, output_path, device)
    else:
        return _clone_indian(text, reference_audio, output_path, lang, device)


def _clone_english(text, reference_audio, output_path, device) -> Path:
    """Chatterbox direct voice clone."""
    ref_wav = _to_wav(reference_audio)
    model = _load_chatterbox(device)
    wav = model.generate(text, audio_prompt_path=str(ref_wav))
    if ref_wav != reference_audio:
        ref_wav.unlink(missing_ok=True)
    import torchaudio
    wav_out = output_path.with_suffix(".wav")
    torchaudio.save(str(wav_out), wav, model.sr)
    return _to_mp3_if_needed(wav_out, output_path)


def _clone_indian(text, reference_audio, output_path, lang, device) -> Path:
    """edge-tts base + OpenVoice v2 tone color conversion.
    Falls back to plain edge-tts if OpenVoice is not installed."""
    base_voice = _INDIAN_BASE_VOICES.get(lang, "hi-IN-SwaraNeural")
    base_mp3 = output_path.with_name(output_path.stem + "_base.mp3")

    # Step 1: generate base audio in target language
    _edge_tts_sync(text, base_voice, base_mp3)

    if not _openvoice_available():
        logger.warning("OpenVoice not installed — returning plain edge-tts audio (no voice cloning)")
        return _to_mp3_if_needed(base_mp3.with_suffix(".wav") if base_mp3.suffix != ".mp3" else base_mp3,
                                 output_path) if base_mp3 != output_path else base_mp3.rename(output_path) or output_path

    from openvoice import se_extractor
    base_wav = output_path.with_name(output_path.stem + "_base.wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(base_mp3), str(base_wav)],
                   capture_output=True, check=True)

    # Step 2: apply reference voice tone
    converter = _load_openvoice(device)
    ref_wav = _to_wav(reference_audio)

    src_se, _ = se_extractor.get_se(str(base_wav), converter, vad=False)
    tgt_se, _ = se_extractor.get_se(str(ref_wav), converter, vad=True)

    wav_out = output_path.with_suffix(".wav")
    converter.convert(
        audio_src_path=str(base_wav),
        src_se=src_se,
        tgt_se=tgt_se,
        output_path=str(wav_out),
        tau=0.3,
    )

    for f in (base_mp3, base_wav):
        f.unlink(missing_ok=True)
    if ref_wav != reference_audio:
        ref_wav.unlink(missing_ok=True)

    return _to_mp3_if_needed(wav_out, output_path)


def _to_wav(src: Path) -> Path:
    if src.suffix.lower() == ".wav":
        return src
    dst = src.with_suffix(".tmp.wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(src), str(dst)],
                   capture_output=True, check=True)
    return dst


def _to_mp3_if_needed(wav_out: Path, output_path: Path) -> Path:
    if output_path.suffix.lower() == ".mp3":
        subprocess.run(["ffmpeg", "-y", "-i", str(wav_out), "-q:a", "2", str(output_path)],
                       capture_output=True)
        wav_out.unlink(missing_ok=True)
        logger.info("Cloned voice → %s", output_path)
        return output_path
    wav_out.rename(output_path)
    logger.info("Cloned voice → %s", output_path)
    return output_path


def unload():
    global _chatterbox, _ov_converter
    for attr in ("_chatterbox", "_ov_converter"):
        obj = globals().get(attr)
        if obj is not None:
            del obj
    _chatterbox = None
    _ov_converter = None
    try:
        import torch, gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
