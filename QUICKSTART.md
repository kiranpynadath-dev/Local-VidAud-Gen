# Quickstart Guide

## Prerequisites

- Python 3.10 or 3.11 (3.12 not yet fully supported by torch-directml)
- Windows 11 (DirectML) or Linux (CUDA/CPU)
- FFmpeg installed and on PATH
- ElevenLabs account (free tier works for testing)

---

## Installation

### Step 1 — Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python --version   # verify 3.10 or 3.11
```

### Step 2 — PyTorch (install before requirements.txt)

AMD GPU on Windows requires the **CPU-only** PyTorch wheel as the base;
`torch-directml` wraps it with DirectML at runtime.

```powershell
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

### Step 3 — All other dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — FFmpeg

```powershell
winget install FFmpeg
# or download from https://ffmpeg.org/download.html and add to PATH
ffmpeg -version   # verify
```

### Step 5 — Environment

```powershell
copy .env.example .env
notepad .env      # add ELEVENLABS_API_KEY
```

### Step 6 — Verify

```powershell
python cli.py device-info
```

Expected output on AMD Radeon 760M:
```json
{
  "device": "AMD Radeon(TM) 760M Graphics",
  "type": "directml",
  "memory_gb": 17.0,
  "float16": true,
  "torch_device": "privateuseone"
}
```

---

## Usage examples

### Generate a video from an image (silent)

```powershell
python cli.py generate --image photo.jpg --no-audio --quality fast --seed 42
# Output: output/output_video.mp4
```

### Generate a video with voiceover

```powershell
python cli.py generate `
  --image photo.jpg `
  --text "Welcome to our product demo. This was made locally." `
  --voice rachel `
  --quality medium `
  --output-dir output/demo `
  --prefix product_demo
# Output: output/demo/product_demo_final.mp4
```

### Generate audio only

```powershell
python cli.py audio `
  --text "Hello, this is a test of the ElevenLabs voice API." `
  --voice adam `
  --output output/test_audio.mp3
```

### Sync existing video + audio

```powershell
python cli.py sync `
  --video raw_video.mp4 `
  --audio voiceover.mp3 `
  --output final.mp4
```

### Batch process a folder of images

```powershell
python cli.py batch `
  --input-dir images/ `
  --text "Product showcase" `
  --quality fast `
  --output-dir output/batch/
```

---

## Python API

### Minimal example

```python
from src.generator import VideoAudioGenerator

gen = VideoAudioGenerator(quality="fast")
result = gen.generate(image="photo.jpg", text="Hello world.")
print(result["final"])
```

### Full example with error handling

```python
import logging
from pathlib import Path
from src.generator import VideoAudioGenerator

logging.basicConfig(level=logging.INFO)

try:
    gen = VideoAudioGenerator(
        quality="medium",
        output_dir="output",
        log_level="INFO",
    )

    print("Device:", gen.device_info())

    result = gen.generate(
        image=Path("photo.jpg"),
        text="Your script goes here.",
        voice="rachel",
        seed=42,
        loop_audio=True,
        filename_prefix="my_video",
    )

    print("Final video:", result["final"])

except ValueError as e:
    print(f"Configuration error: {e}")
except RuntimeError as e:
    print(f"Generation failed: {e}")
finally:
    gen.unload()   # release VRAM / RAM
```

### Custom device

```python
# Force DirectML GPU
gen = VideoAudioGenerator(quality="medium", device="directml")

# Force CPU (slowest, no GPU needed)
gen = VideoAudioGenerator(quality="fast", device="cpu")
```

### Batch jobs in Python

```python
from src.generator import VideoAudioGenerator
from pathlib import Path

gen = VideoAudioGenerator(quality="fast")
proc = gen.create_batch_processor(max_workers=1)

for img in Path("images/").glob("*.jpg"):
    proc.add_job(
        gen.generate,
        img,
        text="Product showcase",
        output_dir=Path("output/batch"),
        filename_prefix=img.stem,
    )

result = proc.run()
print(result.summary())
```

---

## NPU Quickstart (AMD Ryzen AI)

> Skip this section if you don't have Ryzen AI drivers installed.

### One-time setup (~45 minutes, downloads ~10 GB)

```powershell
# Export SVD UNet to ONNX
python cli.py npu export --onnx-dir onnx_models/

# Quantize to INT8 for NPU
python cli.py npu quantize --onnx-dir onnx_models/ --precision int8
```

### Generate with NPU

```powershell
python cli.py npu generate `
  --image photo.jpg `
  --output output/npu_video.mp4 `
  --quality medium
```

Expected speedup: **2-3× faster** than DirectML GPU for the UNet inference step.

---

## ElevenLabs voices

| Name    | Voice ID                   | Style       |
|---------|----------------------------|-------------|
| rachel  | 21m00Tcm4TlvDq8ikWAM       | Calm, clear |
| adam    | pNInz6obpgDQGcFmaJgB       | Deep, male  |
| josh    | TxGEqnHWrfWFTfGW9XjX       | Casual      |
| bella   | EXAVITQu4vr4xnSDxMaL       | Warm        |
| elli    | MF3mGyEYCl7XYWbV9V6O       | Expressive  |

List all available voices:
```python
from src.audio_generator import AudioGenerator
for v in AudioGenerator().list_voices():
    print(v["name"], v["id"])
```
