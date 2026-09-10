# Local Video & Audio Generation System

Generate AI videos and speech locally on AMD Ryzen AI hardware — no cloud required.

| Component | Technology |
|---|---|
| Video generation | Stable Video Diffusion (SVD-XT) |
| Text-to-speech | ElevenLabs API |
| GPU acceleration | DirectML (AMD Radeon 760M) |
| NPU acceleration | AMD Ryzen AI + VitisAI EP (ONNX Runtime) |
| A/V sync | FFmpeg |

## Hardware requirements

- AMD Ryzen 5 PRO 230 (12-core) or equivalent
- 32 GB RAM (minimum 16 GB)
- AMD Radeon 760M (DirectX 12 / D3D12)
- AMD Ryzen AI NPU with Vitis AI Runtime drivers *(optional, for 2-3× speedup)*
- Windows 11 Pro

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/kiranpynadath-dev/local-vidaud-gen
cd Local-VidAud-Gen

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install PyTorch (CPU base — required by torch-directml)
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env — add your ELEVENLABS_API_KEY

# 6. Verify device detection
python cli.py device-info

# 7. Generate your first video
python cli.py generate --image photo.jpg --text "Hello world" --quality fast
```

See [QUICKSTART.md](QUICKSTART.md) for more examples and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues.

## Project structure

```
Local-VidAud-Gen/
├── src/
│   ├── generator.py        # Unified high-level API
│   ├── video_processor.py  # SVD pipeline (DirectML / CUDA / CPU)
│   ├── audio_generator.py  # ElevenLabs TTS
│   ├── sync.py             # FFmpeg A/V sync
│   ├── npu_generator.py    # NPU-optimized ONNX path
│   ├── batch_processor.py  # Queue-based batch jobs
│   ├── device_manager.py   # Auto-detect NPU/GPU/CPU
│   └── utils.py            # Logging, memory, progress
├── examples/
│   ├── basic_video_gen.py
│   ├── audio_video_sync.py
│   ├── batch_processing.py
│   └── npu_optimized.py
├── cli.py                  # CLI entry point
├── requirements.txt
├── .env.example
├── setup.py
├── QUICKSTART.md
└── TROUBLESHOOTING.md
```

## Python API

```python
from src.generator import VideoAudioGenerator

gen = VideoAudioGenerator(quality="medium")   # auto-detects device

result = gen.generate(
    image="photo.jpg",
    text="Your script here.",
    voice="rachel",
    output_dir="output/",
)
print(result["final"])   # → output/output_final.mp4
```

## CLI reference

```
python cli.py generate      --image photo.jpg --text "Script" --quality medium
python cli.py audio         --text "Script" --output audio/out.mp3
python cli.py sync          --video raw.mp4 --audio speech.mp3 --output final.mp4
python cli.py batch         --input-dir images/ --text "Script" --output-dir out/
python cli.py device-info
python cli.py npu export    --onnx-dir onnx_models/
python cli.py npu quantize  --onnx-dir onnx_models/ --precision int8
python cli.py npu generate  --image photo.jpg --output npu_video.mp4
```

## Quality presets

| Preset | Frames | Steps | Resolution | FPS | Approx. time (DirectML) |
|--------|--------|-------|------------|-----|--------------------------|
| fast   | 14     | 10    | 576×320    | 6   | ~3 min                   |
| medium | 25     | 20    | 1024×576   | 12  | ~8 min                   |
| high   | 25     | 30    | 1024×576   | 24  | ~15 min                  |

## Device selection priority

```
NPU (VitisAI) → DirectML GPU → CUDA → CPU
```

Force a device: `--device directml` or set `FORCE_DEVICE=directml` in `.env`.

## NPU acceleration

NPU inference gives 2-3× faster generation on AMD Ryzen AI NPUs.  
One-time setup (export + quantize, ~45 min):

```bash
python cli.py npu export --onnx-dir onnx_models/
python cli.py npu quantize --onnx-dir onnx_models/ --precision int8
```

Then generate:

```bash
python cli.py npu generate --image photo.jpg --output video.mp4
```

## License

MIT
