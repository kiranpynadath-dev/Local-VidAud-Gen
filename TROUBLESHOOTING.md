# Troubleshooting Guide

---

## Device detection

### DirectML not detected

**Symptom:** `device-info` returns `"type": "cpu"` on Windows with AMD GPU.

**Fixes:**
1. Verify torch-directml is installed: `python -c "import torch_directml; print(torch_directml.device_name(0))"`
2. Install PyTorch CPU wheels first, then torch-directml:
   ```
   pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
   pip install torch-directml==0.2.4.dev230426
   ```
3. Verify DirectX 12 support: `dxdiag` → Display tab → Feature Level = 12_0 or higher.
4. Update AMD GPU drivers from https://www.amd.com/en/support.

---

### NPU not detected

**Symptom:** `"VitisAIExecutionProvider"` missing from `onnxruntime.get_available_providers()`.

**Fixes:**
1. Install AMD Ryzen AI SDK: https://ryzenai.docs.amd.com/en/latest/inst.html
2. Uninstall the standard `onnxruntime-directml` and replace with AMD's custom ONNX Runtime build.
3. Verify Vitis AI drivers: Device Manager → System devices → AMD AI NPU.
4. Verify: `python -c "import onnxruntime as ort; print(ort.get_available_providers())"`

---

## Memory errors

### `OutOfMemoryError` during video generation

**Fixes (try in order):**

1. **Reduce quality preset:**
   ```
   python cli.py generate --quality fast ...
   ```

2. **Force CPU offloading** — already enabled automatically, but verify by checking logs for `enable_model_cpu_offload`.

3. **Reduce decode_chunk_size** — lower values use less VRAM at the cost of speed. Override in `src/video_processor.py` `QUALITY_PRESETS`.

4. **Free other GPU applications** — close browser tabs, other ML apps, games.

5. **Use NPU path** — ONNX Runtime NPU inference has a smaller VRAM footprint.

6. **Set Windows VRAM** — AMD Radeon 760M uses shared system RAM. Increase GPU memory allocation:
   BIOS → UMA Frame Buffer Size → 4GB (or higher).

---

### Python crashes with `Segmentation fault` or `Access violation`

- Caused by DirectML + float16 mismatch on some AMD driver versions.
- Try `device="cpu"` to rule out DirectML as the cause.
- Update AMD GPU drivers.

---

## FFmpeg errors

### `ffmpeg: command not found`

Install FFmpeg and add to PATH:
```powershell
winget install FFmpeg
# Restart terminal after install
ffmpeg -version
```

### `ffmpeg failed: ... moov atom not found`

The input video is corrupt or incomplete.  
Re-run video generation to produce a fresh file.

### Audio/video out of sync

The generated video is shorter than the audio. Enable `loop_audio`:
```
python cli.py sync --video raw.mp4 --audio speech.mp3 --output final.mp4
# (loop_audio=True is the default)
```

Or trim the audio:
```
ffmpeg -i speech.mp3 -t <video_duration> trimmed.mp3
```

---

## Model download issues

### `401 Unauthorized` when downloading SVD model

SVD requires accepting a license on Hugging Face:
1. Visit https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
2. Click "Access repository" and accept the license.
3. Generate a token: https://huggingface.co/settings/tokens
4. Add to `.env`: `HF_TOKEN=your_token_here`
5. Or login via CLI: `huggingface-cli login`

### Slow download / interrupted download

- Set a local model cache to avoid re-downloading:
  ```
  HF_HOME=D:/models/huggingface
  ```
- The SVD model is ~8 GB. Ensure you have space.
- Resume interrupted downloads: diffusers caches partial shards automatically.

---

## ElevenLabs errors

### `ValueError: ElevenLabs API key not found`

Add your key to `.env`:
```
ELEVENLABS_API_KEY=your_key_here
```
Or pass it directly:
```python
AudioGenerator(api_key="your_key_here")
```

### `401 Unauthorized` from ElevenLabs

- API key is invalid or expired. Regenerate at https://elevenlabs.io/app/settings/api-keys.

### `429 Too Many Requests`

- Free tier has monthly character limits and rate limits.
- Add `time.sleep(1)` between requests or upgrade your ElevenLabs plan.

---

## NPU (ONNX / VitisAI) issues

### `FileNotFoundError: unet.onnx`

Run the export step first:
```
python cli.py npu export --onnx-dir onnx_models/
```
This takes ~30 minutes and requires ~12 GB RAM.

### `VitisAIExecutionProvider` is listed but gives wrong results

- Ensure `VITISAI_EP_JSON_CONFIG` points to the correct `vaip_config.json` from the Ryzen AI SDK.
- Try loading without quantization first (`--no-quantized`) to confirm the base ONNX is correct.

### Quantization produces `NaN` outputs

- The INT8 calibration dataset might be too small. Increase `num_calibration_data` in the Olive config.
- Try INT4 instead: `python cli.py npu quantize --precision int4`.

---

## General debugging

### Enable DEBUG logging

```powershell
python cli.py --log-level DEBUG generate --image photo.jpg ...
```

### Check memory usage during generation

```python
from src.utils import get_memory_info
import time

while True:
    print(get_memory_info())
    time.sleep(2)
```

### Validate installation

```python
# Run this to check all imports work:
from src.device_manager import DeviceManager
from src.video_processor import VideoProcessor
from src.audio_generator import AudioGenerator
from src.sync import sync_audio_video
from src.batch_processor import BatchProcessor
from src.npu_generator import NPUVideoGenerator
print("All imports OK")
print(DeviceManager().detect())
```

---

## Performance tips

| Situation | Recommendation |
|-----------|---------------|
| First run (model not cached) | Allow 15-30 min for ~8 GB download |
| Repeated runs | Models load from disk cache in ~60s |
| Low on RAM (<16 GB free) | Use `quality=fast`, close other apps |
| Batch processing | Set `max_workers=1` to avoid OOM |
| NPU first use | Run export + quantize once (setup ~45 min) |
| Best quality | Use `quality=high` + `seed=42` for reproducibility |
