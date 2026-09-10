"""Example: NPU-accelerated video generation via ONNX Runtime + AMD VitisAI.

One-time setup (run once, then skip export/quantize steps):
    python examples/npu_optimized.py --setup

Inference only (after setup):
    python examples/npu_optimized.py --image photo.jpg --output output/npu_out.mp4
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.npu_generator import NPUVideoGenerator
from src.device_manager import DeviceManager, DeviceType

ONNX_DIR = Path("onnx_models")
MODEL_CACHE = Path("D:/models/huggingface")  # adjust to your cache location


def setup():
    print("=== NPU Setup: Export + Quantize ===")
    npu = NPUVideoGenerator(onnx_dir=ONNX_DIR, cache_dir=MODEL_CACHE)

    print("Step 1/2: Exporting SVD UNet to ONNX (requires ~30 min + 12 GB RAM) …")
    unet_path = npu.export_onnx()
    print(f"  Exported: {unet_path}")

    print("Step 2/2: Quantizing to INT8 for NPU (requires ~15 min) …")
    q_path = npu.quantize(precision="int8")
    print(f"  Quantized: {q_path}")

    print("\nSetup complete. Run without --setup to generate videos.")


def generate(image_path: Path, output_path: Path, quality: str = "medium"):
    # Verify NPU is available
    dm = DeviceManager()
    info = dm.detect()
    if info.device_type != DeviceType.NPU:
        print(f"WARNING: NPU not detected (got {info.device_type.value}). Falling back to CPU ONNX.")

    npu = NPUVideoGenerator(
        onnx_dir=ONNX_DIR,
        quality=quality,
        cache_dir=MODEL_CACHE,
    )

    print(f"Loading quantized ONNX session …")
    npu.load_session(quantized=True, precision="int8")

    print(f"Generating video from {image_path} …")
    out = npu.generate_from_image(image=image_path, output_path=output_path, seed=42)

    npu.unload_session()
    print(f"NPU video saved: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Run one-time export + quantize")
    parser.add_argument("--image", default="examples/sample_input.jpg")
    parser.add_argument("--output", default="output/npu_video.mp4")
    parser.add_argument("--quality", default="medium", choices=["fast", "medium", "high"])
    args = parser.parse_args()

    if args.setup:
        setup()
    else:
        generate(Path(args.image), Path(args.output), quality=args.quality)
