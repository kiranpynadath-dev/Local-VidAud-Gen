"""NPU-optimized video generation via ONNX Runtime + AMD VitisAI.

Workflow (one-time setup):
1. Export SVD UNet to ONNX format with `NPUVideoGenerator.export_onnx()`.
2. Quantize to INT8 with `NPUVideoGenerator.quantize()` (2-3x faster on NPU).
3. Run inference with `NPUVideoGenerator.generate_from_image()`.

The VAE encoder/decoder remain on CPU (float32) since they are I/O-bound
and VitisAI EP is optimized for the compute-heavy UNet attention blocks.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)

# VitisAI config file — set VITISAI_EP_JSON_CONFIG env var to override.
DEFAULT_VITISAI_CONFIG = "vaip_config.json"


class NPUVideoGenerator:
    def __init__(
        self,
        onnx_dir: Union[str, Path],
        quality: str = "medium",
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            onnx_dir: Directory where exported/quantized ONNX models are stored.
            quality: 'fast' | 'medium' | 'high'  (controls frames + steps).
            cache_dir: Hugging Face model cache directory.
        """
        self.onnx_dir = Path(onnx_dir)
        self.quality = quality
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._session: Optional[object] = None  # ort.InferenceSession

    # ------------------------------------------------------------------
    # One-time setup helpers
    # ------------------------------------------------------------------

    def export_onnx(self, model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt") -> Path:
        """Export the SVD UNet to ONNX. Only needed once per model.

        Returns:
            Path to the exported unet.onnx file.
        """
        import torch
        from diffusers import StableVideoDiffusionPipeline

        self.onnx_dir.mkdir(parents=True, exist_ok=True)
        unet_path = self.onnx_dir / "unet.onnx"

        if unet_path.exists():
            logger.info("ONNX model already exists at %s, skipping export", unet_path)
            return unet_path

        logger.info("Loading SVD pipeline for ONNX export …")
        kw = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
        if self.cache_dir:
            kw["cache_dir"] = str(self.cache_dir)
        pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, **kw)
        unet = pipe.unet.eval()

        # Build dummy inputs matching SVD UNet signature
        B, F, C, H, W = 1, 8, 4, 40, 72
        dummy_sample = torch.zeros(B * 2, F, C, H, W)
        dummy_ts = torch.tensor([1.0])
        dummy_enc = torch.zeros(B * 2, 1, unet.config.cross_attention_dim)
        dummy_af = torch.zeros(B * 2, 1)

        logger.info("Exporting UNet to %s …", unet_path)
        torch.onnx.export(
            unet,
            (dummy_sample, dummy_ts, dummy_enc, dummy_af),
            str(unet_path),
            input_names=["sample", "timestep", "encoder_hidden_states", "added_time_ids"],
            output_names=["out_sample"],
            dynamic_axes={
                "sample": {0: "batch"},
                "encoder_hidden_states": {0: "batch"},
                "added_time_ids": {0: "batch"},
            },
            opset_version=17,
        )
        logger.info("ONNX export complete → %s", unet_path)
        del pipe, unet
        return unet_path

    def quantize(self, precision: str = "int8") -> Path:
        """Quantize the exported ONNX UNet for NPU execution.

        Uses AMD Olive (preferred) or onnxruntime quantization as fallback.
        Requires running once before NPU inference.
        """
        unet_path = self.onnx_dir / "unet.onnx"
        quantized_path = self.onnx_dir / f"unet_{precision}.onnx"

        if quantized_path.exists():
            logger.info("Quantized model already exists at %s", quantized_path)
            return quantized_path

        if not unet_path.exists():
            raise FileNotFoundError(f"Run export_onnx() first. Expected: {unet_path}")

        # Try AMD Olive first (preferred for VitisAI-optimized quantization)
        try:
            return self._quantize_with_olive(unet_path, quantized_path, precision)
        except ImportError:
            logger.warning("AMD Olive not available, falling back to onnxruntime quantization")
            return self._quantize_with_ort(unet_path, quantized_path, precision)

    def _quantize_with_olive(self, src: Path, dst: Path, precision: str) -> Path:
        """Quantize using AMD Olive framework."""
        from olive.workflows import run as olive_run  # type: ignore

        config = {
            "input_model": {"type": "OnnxModel", "config": {"model_path": str(src)}},
            "systems": {
                "local_system": {"type": "LocalSystem", "config": {"accelerators": [{"device": "npu", "execution_providers": ["VitisAIExecutionProvider"]}]}}
            },
            "passes": {
                "quantization": {
                    "type": "VitisAIQuantization" if precision == "int8" else "OnnxDynamicQuantization",
                    "config": {"quant_mode": "static" if precision == "int8" else "dynamic"},
                }
            },
            "output_dir": str(dst.parent),
        }
        olive_run(config)
        # Olive writes to output_dir; find the output
        candidates = list(dst.parent.glob("*quantized*.onnx"))
        if not candidates:
            raise RuntimeError("Olive quantization produced no output ONNX file")
        candidates[0].rename(dst)
        logger.info("Olive quantization complete → %s", dst)
        return dst

    def _quantize_with_ort(self, src: Path, dst: Path, precision: str) -> Path:
        """Quantize using onnxruntime's built-in quantization."""
        if precision == "int8":
            from onnxruntime.quantization import QuantType, quantize_static  # type: ignore
            quantize_static(str(src), str(dst), quant_type=QuantType.QInt8)
        else:
            from onnxruntime.quantization import quantize_dynamic  # type: ignore
            quantize_dynamic(str(src), str(dst))
        logger.info("ORT quantization complete → %s", dst)
        return dst

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load_session(self, quantized: bool = True, precision: str = "int8") -> None:
        """Load ONNX Runtime inference session with VitisAI execution provider."""
        import onnxruntime as ort

        model_path = self.onnx_dir / (f"unet_{precision}.onnx" if quantized else "unet.onnx")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run export_onnx() and quantize() first, or pass quantized=False."
            )

        vitisai_config = os.environ.get("VITISAI_EP_JSON_CONFIG", DEFAULT_VITISAI_CONFIG)
        providers = [
            ("VitisAIExecutionProvider", {"config_file": vitisai_config}),
            "CPUExecutionProvider",  # fallback
        ]
        available = ort.get_available_providers()
        if "VitisAIExecutionProvider" not in available:
            logger.warning("VitisAI EP not available — falling back to CPU EP only")
            providers = ["CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.enable_mem_pattern = True
        so.enable_cpu_mem_arena = True
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(str(model_path), sess_options=so, providers=providers)
        logger.info("ONNX session loaded from %s", model_path)

    def generate_from_image(
        self,
        image: Union[str, Path, Image.Image],
        output_path: Union[str, Path],
        seed: Optional[int] = None,
    ) -> Path:
        """Generate video using NPU-accelerated UNet inference.

        Falls back to CPU for VAE encode/decode.
        """
        if self._session is None:
            self.load_session()

        # Import pipeline pieces individually to keep VAE on CPU
        import numpy as np
        import torch
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.schedulers import EulerDiscreteScheduler

        from .video_processor import QUALITY_PRESETS, frames_to_video
        from .utils import ensure_dir

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        preset = QUALITY_PRESETS[self.quality]
        image = image.resize((preset["width"], preset["height"]), Image.LANCZOS)
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        kw: dict = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True)
        if self.cache_dir:
            kw["cache_dir"] = str(self.cache_dir)

        logger.info("Loading VAE and image encoder on CPU for NPU pipeline …")
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", **kw
        )
        pipe.enable_model_cpu_offload()

        # Override UNet forward with ONNX session call
        original_unet = pipe.unet

        def onnx_unet_forward(sample, timestep, encoder_hidden_states, added_time_ids, **_):
            inputs = {
                "sample": sample.numpy().astype(np.float32),
                "timestep": np.array([float(timestep)], dtype=np.float32),
                "encoder_hidden_states": encoder_hidden_states.numpy().astype(np.float32),
                "added_time_ids": added_time_ids.numpy().astype(np.float32),
            }
            out = self._session.run(["out_sample"], inputs)[0]

            class _Wrapper:
                def __init__(self, sample):
                    self.sample = torch.from_numpy(sample)

            return _Wrapper(out)

        pipe.unet.forward = onnx_unet_forward  # type: ignore

        rng = torch.manual_seed(seed) if seed is not None else None
        logger.info("Running NPU-accelerated inference (quality=%s) …", self.quality)

        result = pipe(
            image,
            num_frames=preset["num_frames"],
            num_inference_steps=preset["num_inference_steps"],
            decode_chunk_size=preset["decode_chunk_size"],
            motion_bucket_id=preset["motion_bucket_id"],
            noise_aug_strength=preset["noise_aug_strength"],
            generator=rng,
        )

        frames_to_video(result.frames[0], output_path, fps=preset["fps"])
        logger.info("NPU video saved → %s", output_path)
        return output_path

    def unload_session(self) -> None:
        self._session = None
        logger.info("ONNX session released")
