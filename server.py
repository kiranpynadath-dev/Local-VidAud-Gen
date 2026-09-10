#!/usr/bin/env python3
"""
FastAPI server for the Local VidGen AI web UI.

Start with:
    python server.py

Then open http://localhost:8000 in your browser.

Extra dependencies (add to venv before running):
    pip install fastapi uvicorn python-multipart
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path so `from src.xxx import` works
# regardless of what directory the user runs the script from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import logging
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="LocalVid Gen API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store  {job_id: {...}}
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)  # one GPU job at a time
_current_gen = None  # global generator ref — unloaded before each new job

# ---------------------------------------------------------------------------
# Serve UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path("index.html")
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found. Place it next to server.py.</h2>", status_code=404)


@app.get("/api/health")
async def api_health():
    return {"status": "ok"}


@app.get("/api/output/{filename}")
async def serve_output(filename: str, request: Request):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    ext = path.suffix.lower()
    media_map = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".zip": "application/zip"}
    media_type = media_map.get(ext, "application/octet-stream")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header and ext == ".mp4":
        # Parse "bytes=start-end"
        range_val = range_header.replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end   = int(parts[1]) if parts[1] else file_size - 1
        end   = min(end, file_size - 1)
        chunk = end - start + 1

        def _iter():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = chunk
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            _iter(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk),
            },
        )

    return FileResponse(
        str(path), media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.post("/api/enrich-image")
async def api_enrich_image(file: UploadFile = File(...)):
    """Receive an image, return a 16:9 1920×1080 enriched version."""
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {ext}")
    src = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    dst = OUTPUT_DIR / f"{src.stem}_16x9.jpg"
    try:
        from src.image_utils import enrich_to_16x9
        enrich_to_16x9(src, dst)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return FileResponse(str(dst), media_type="image/jpeg",
                        headers={"Content-Disposition": f'attachment; filename="{dst.name}"'})


@app.get("/api/device-info")
async def api_device_info():
    try:
        from src.device_manager import DeviceManager
        info = DeviceManager().detect()
        return {
            "device": info.device_name,
            "type": info.device_type.value,
            "memory_gb": info.memory_gb,
            "float16": info.supports_float16,
            "torch_device": info.torch_device,
        }
    except Exception as exc:
        return {"device": "Unknown", "type": "cpu", "error": str(exc)}


@app.get("/api/voices")
async def api_voices():
    from src.tts_generator import VOICES
    voices = [{"id": vid, "name": name.capitalize()} for name, vid in VOICES.items()]
    return {"voices": voices}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": filename, "path": str(dest)}


class GenerateRequest(BaseModel):
    image_filename: str
    text: Optional[str] = None
    voice: str = "heart"          # Kokoro voice name or ID (e.g. "heart", "af_heart")
    speed: float = 1.0
    quality: str = "medium"
    seed: Optional[int] = None
    loop_audio: bool = True


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    image_path = UPLOAD_DIR / req.image_filename
    if not image_path.exists():
        raise HTTPException(status_code=400, detail=f"Uploaded image not found: {req.image_filename}")

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "pending", "progress": 0, "result": None, "error": None, "log": []}
    asyncio.create_task(_run_generation(job_id, req, image_path))
    return {"job_id": job_id}


# ── Text-to-video ─────────────────────────────────────────────────────────

class TextToVideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = "blurry, low quality, distorted, watermark"
    text: Optional[str] = None          # TTS script (voiceover)
    voice: str = "heart"
    speed: float = 1.0
    quality: str = "medium"
    seed: Optional[int] = None
    loop_audio: bool = True


@app.post("/api/generate-text")
async def api_generate_text(req: TextToVideoRequest):
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "pending", "progress": 0, "result": None, "error": None, "log": []}
    asyncio.create_task(_run_t2v(job_id, req))
    return {"job_id": job_id}


async def _run_t2v(job_id: str, req: TextToVideoRequest) -> None:
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["log"].append("Starting text-to-video generation …")

    def _gen():
        global _current_gen
        if _current_gen is not None:
            try: _current_gen.unload()
            except Exception: pass
            _current_gen = None
        from src.generator import VideoAudioGenerator
        gen = VideoAudioGenerator(quality=req.quality, output_dir=str(OUTPUT_DIR))
        _current_gen = gen
        try:
            result = gen.generate_from_text(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                text=req.text or None,
                voice=req.voice,
                speed=req.speed,
                seed=req.seed,
                loop_audio=req.loop_audio,
                filename_prefix=job_id,
            )
        finally:
            gen.unload()
            _current_gen = None
        return result

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _gen)
        _jobs[job_id].update({
            "status": "done", "progress": 100,
            "result": {
                "video": result["video"].name if result.get("video") else None,
                "audio": result["audio"].name if result.get("audio") else None,
                "final": result["final"].name if result.get("final") else None,
            },
        })
    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "error": str(exc)})
        logger.error("T2V job %s failed: %s", job_id, exc)


# ── Video editing ──────────────────────────────────────────────────────────

@app.post("/api/upload-video")
async def api_upload_video(file: UploadFile = File(...)):
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = Path(file.filename or "video").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported video type: {ext}")
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"filename": filename}


class EditVideoRequest(BaseModel):
    video_filename: str
    prompt: str
    negative_prompt: str = "blurry, low quality, artifacts"
    mode: str = "sdedit"        # "sdedit" or "instruct"
    strength: float = 0.55
    steps: int = 20
    max_frames: Optional[int] = 30   # limit for speed; None = full video
    seed: int = 42


@app.post("/api/edit-video")
async def api_edit_video(req: EditVideoRequest):
    video_path = UPLOAD_DIR / req.video_filename
    if not video_path.exists():
        raise HTTPException(status_code=400, detail=f"Video not found: {req.video_filename}")

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "pending", "progress": 0, "result": None, "error": None, "log": []}
    asyncio.create_task(_run_edit(job_id, req, video_path))
    return {"job_id": job_id}


async def _run_edit(job_id: str, req: EditVideoRequest, video_path: Path) -> None:
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["log"].append(f"Editing video (mode={req.mode}) …")

    def _edit():
        global _current_gen
        if _current_gen is not None:
            try: _current_gen.unload()
            except Exception: pass
            _current_gen = None
        from src.generator import VideoAudioGenerator
        gen = VideoAudioGenerator(quality="medium", output_dir=str(OUTPUT_DIR))
        _current_gen = gen
        try:
            out = gen.edit_video(
                video_path=video_path,
                prompt=req.prompt,
                mode=req.mode,
                negative_prompt=req.negative_prompt,
                strength=req.strength,
                steps=req.steps,
                max_frames=req.max_frames,
                seed=req.seed,
                filename_prefix=job_id,
            )
        finally:
            gen.unload()
            _current_gen = None
        return out

    loop = asyncio.get_event_loop()
    try:
        out_path = await loop.run_in_executor(_executor, _edit)
        _jobs[job_id].update({
            "status": "done", "progress": 100,
            "result": {"final": out_path.name, "video": out_path.name, "audio": None},
        })
    except Exception as exc:
        _jobs[job_id].update({"status": "failed", "error": str(exc)})
        logger.error("Edit job %s failed: %s", job_id, exc)


@app.get("/api/job/{job_id}")
async def api_job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


# ---------------------------------------------------------------------------
# Background generation task
# ---------------------------------------------------------------------------

async def _run_generation(job_id: str, req: GenerateRequest, image_path: Path) -> None:
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["progress"] = 5
    _jobs[job_id]["log"].append("Starting generation …")

    def _generate():
        global _current_gen
        if _current_gen is not None:
            try: _current_gen.unload()
            except Exception: pass
            _current_gen = None
        from src.generator import VideoAudioGenerator
        gen = VideoAudioGenerator(
            quality=req.quality,
            output_dir=str(OUTPUT_DIR),
            log_level="INFO",
        )
        _current_gen = gen
        try:
            result = gen.generate(
                image=image_path,
                text=req.text if req.text else None,
                voice=req.voice,
                speed=req.speed,
                seed=req.seed,
                loop_audio=req.loop_audio,
                filename_prefix=job_id,
            )
        finally:
            gen.unload()
            _current_gen = None
        return result

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(_executor, _generate)
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["result"] = {
            "video": result["video"].name if result.get("video") else None,
            "audio": result["audio"].name if result.get("audio") else None,
            "final": result["final"].name if result.get("final") else None,
        }
        _jobs[job_id]["log"].append("Generation complete.")
    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
        _jobs[job_id]["log"].append(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import threading, webbrowser, time

    PORT = int(os.environ.get("PORT", 8000))
    URL  = f"http://localhost:{PORT}"

    print(f"\n{'='*50}")
    print(f"  LocalVid AI — server starting on {URL}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*50}\n")

    # Open browser after a short delay so the server is up first
    def _open_browser():
        time.sleep(1.8)
        webbrowser.open(URL)

    threading.Thread(target=_open_browser, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False, log_level="info")
