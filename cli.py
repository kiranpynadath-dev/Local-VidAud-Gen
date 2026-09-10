#!/usr/bin/env python3
"""CLI for Local Video and Audio Generation System.

Usage examples:
    python cli.py generate --image photo.jpg --text "Hello world" --quality medium
    python cli.py generate --image photo.jpg --no-audio --quality fast --seed 42
    python cli.py audio --text "Hello" --voice rachel --output audio/out.mp3
    python cli.py sync --video raw.mp4 --audio speech.mp3 --output final.mp4
    python cli.py batch --input-dir images/ --text "Product demo" --output-dir out/
    python cli.py device-info
    python cli.py npu export --model-cache ./models
    python cli.py npu quantize --onnx-dir ./onnx_models
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def cmd_generate(args: argparse.Namespace) -> None:
    from src.generator import VideoAudioGenerator

    gen = VideoAudioGenerator(
        quality=args.quality,
        device=args.device,
        output_dir=args.output_dir,
        model_cache_dir=args.model_cache,
        log_level=args.log_level,
    )

    print(f"Device: {gen.device_info()}")

    result = gen.generate(
        image=Path(args.image),
        text=args.text if not args.no_audio else None,
        voice=args.voice,
        seed=args.seed,
        loop_audio=not args.no_loop,
        filename_prefix=args.prefix,
    )

    print("\n=== Output ===")
    for k, v in result.items():
        if v:
            print(f"  {k}: {v}")


def cmd_audio(args: argparse.Namespace) -> None:
    from src.audio_generator import AudioGenerator

    gen = AudioGenerator()
    out = gen.generate_speech(
        text=args.text,
        output_path=Path(args.output),
        voice=args.voice,
        model_id=args.model,
        stability=args.stability,
        similarity_boost=args.similarity,
    )
    print(f"Audio saved: {out}")


def cmd_sync(args: argparse.Namespace) -> None:
    from src.sync import sync_audio_video

    out = sync_audio_video(
        video_path=Path(args.video),
        audio_path=Path(args.audio),
        output_path=Path(args.output),
        loop_audio=not args.no_loop,
    )
    print(f"Synced output: {out}")


def cmd_batch(args: argparse.Namespace) -> None:
    from src.generator import VideoAudioGenerator
    from src.batch_processor import BatchProcessor

    gen = VideoAudioGenerator(quality=args.quality, device=args.device, log_level=args.log_level)
    proc = gen.create_batch_processor(max_workers=args.workers)
    output_dir = Path(args.output_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in image_exts])

    if not images:
        print(f"No images found in {args.input_dir}")
        sys.exit(1)

    print(f"Queuing {len(images)} images …")
    for img in images:
        out_path = output_dir / f"{img.stem}_final.mp4"
        proc.add_job(
            gen.generate,
            img,
            text=args.text,
            output_dir=output_dir,
            filename_prefix=img.stem,
            job_id=img.stem,
        )

    def on_done(job):
        status = "✓" if job.status.value == "done" else "✗"
        print(f"  {status} {job.job_id} ({job.elapsed:.1f}s)" if job.elapsed else f"  {status} {job.job_id}")

    result = proc.run(progress_callback=on_done)
    print(f"\n{result.summary()}")


def cmd_device_info(args: argparse.Namespace) -> None:
    from src.device_manager import DeviceManager

    dm = DeviceManager()
    info = dm.detect()
    data = {
        "device": info.device_name,
        "type": info.device_type.value,
        "memory_gb": info.memory_gb,
        "float16": info.supports_float16,
        "torch_device": info.torch_device,
    }
    print(json.dumps(data, indent=2))


def cmd_npu_export(args: argparse.Namespace) -> None:
    from src.npu_generator import NPUVideoGenerator

    npu = NPUVideoGenerator(onnx_dir=Path(args.onnx_dir), cache_dir=args.model_cache)
    path = npu.export_onnx()
    print(f"ONNX model exported: {path}")


def cmd_npu_quantize(args: argparse.Namespace) -> None:
    from src.npu_generator import NPUVideoGenerator

    npu = NPUVideoGenerator(onnx_dir=Path(args.onnx_dir))
    path = npu.quantize(precision=args.precision)
    print(f"Quantized model: {path}")


def cmd_npu_generate(args: argparse.Namespace) -> None:
    from src.npu_generator import NPUVideoGenerator

    npu = NPUVideoGenerator(
        onnx_dir=Path(args.onnx_dir),
        quality=args.quality,
        cache_dir=args.model_cache,
    )
    npu.load_session(quantized=not args.no_quantized, precision=args.precision)
    out = npu.generate_from_image(
        image=Path(args.image),
        output_path=Path(args.output),
        seed=args.seed,
    )
    print(f"NPU video: {out}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vidaudgen",
        description="Local Video & Audio Generation System (AMD Ryzen AI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- generate ----
    p_gen = sub.add_parser("generate", help="Generate video + optional TTS audio from an image")
    p_gen.add_argument("--image", required=True, help="Input image path")
    p_gen.add_argument("--text", default=None, help="Script text for TTS (omit for silent video)")
    p_gen.add_argument("--no-audio", action="store_true", help="Skip audio generation")
    p_gen.add_argument("--voice", default="rachel", help="ElevenLabs voice name or ID")
    p_gen.add_argument("--quality", default="medium", choices=["fast", "medium", "high"])
    p_gen.add_argument("--device", default=None, choices=["npu", "directml", "cuda", "cpu"])
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--no-loop", action="store_true", help="Don't loop audio")
    p_gen.add_argument("--output-dir", default="output")
    p_gen.add_argument("--prefix", default="output")
    p_gen.add_argument("--model-cache", default=None)
    p_gen.set_defaults(func=cmd_generate)

    # ---- audio ----
    p_aud = sub.add_parser("audio", help="Generate TTS audio only")
    p_aud.add_argument("--text", required=True)
    p_aud.add_argument("--output", required=True)
    p_aud.add_argument("--voice", default="rachel")
    p_aud.add_argument("--model", default="eleven_multilingual_v2")
    p_aud.add_argument("--stability", type=float, default=0.5)
    p_aud.add_argument("--similarity", type=float, default=0.75)
    p_aud.set_defaults(func=cmd_audio)

    # ---- sync ----
    p_sync = sub.add_parser("sync", help="Mux existing video + audio with FFmpeg")
    p_sync.add_argument("--video", required=True)
    p_sync.add_argument("--audio", required=True)
    p_sync.add_argument("--output", required=True)
    p_sync.add_argument("--no-loop", action="store_true")
    p_sync.set_defaults(func=cmd_sync)

    # ---- batch ----
    p_bat = sub.add_parser("batch", help="Batch-process a directory of images")
    p_bat.add_argument("--input-dir", required=True)
    p_bat.add_argument("--output-dir", default="output")
    p_bat.add_argument("--text", default=None)
    p_bat.add_argument("--quality", default="medium", choices=["fast", "medium", "high"])
    p_bat.add_argument("--device", default=None, choices=["npu", "directml", "cuda", "cpu"])
    p_bat.add_argument("--workers", type=int, default=1)
    p_bat.set_defaults(func=cmd_batch)

    # ---- device-info ----
    p_dev = sub.add_parser("device-info", help="Show detected hardware and device info")
    p_dev.set_defaults(func=cmd_device_info)

    # ---- npu ----
    p_npu = sub.add_parser("npu", help="NPU-specific commands")
    npu_sub = p_npu.add_subparsers(dest="npu_command", required=True)

    p_npu_exp = npu_sub.add_parser("export", help="Export SVD UNet to ONNX")
    p_npu_exp.add_argument("--onnx-dir", default="onnx_models")
    p_npu_exp.add_argument("--model-cache", default=None)
    p_npu_exp.set_defaults(func=cmd_npu_export)

    p_npu_quant = npu_sub.add_parser("quantize", help="Quantize ONNX UNet for NPU")
    p_npu_quant.add_argument("--onnx-dir", default="onnx_models")
    p_npu_quant.add_argument("--precision", default="int8", choices=["int8", "int4"])
    p_npu_quant.set_defaults(func=cmd_npu_quantize)

    p_npu_gen = npu_sub.add_parser("generate", help="Generate video via NPU inference")
    p_npu_gen.add_argument("--image", required=True)
    p_npu_gen.add_argument("--output", required=True)
    p_npu_gen.add_argument("--onnx-dir", default="onnx_models")
    p_npu_gen.add_argument("--quality", default="medium", choices=["fast", "medium", "high"])
    p_npu_gen.add_argument("--precision", default="int8")
    p_npu_gen.add_argument("--no-quantized", action="store_true")
    p_npu_gen.add_argument("--model-cache", default=None)
    p_npu_gen.add_argument("--seed", type=int, default=None)
    p_npu_gen.set_defaults(func=cmd_npu_generate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))
    args.func(args)


if __name__ == "__main__":
    main()
