"""Example: Full pipeline — image → SVD video + ElevenLabs TTS → synced mp4."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.generator import VideoAudioGenerator

SCRIPT = """
Welcome to our product demonstration.
This video was generated entirely on-device using AMD Ryzen AI hardware,
Stable Video Diffusion, and ElevenLabs text-to-speech.
No cloud rendering required.
"""

gen = VideoAudioGenerator(
    quality="medium",
    output_dir="output/demo",
)

result = gen.generate(
    image=Path("examples/sample_input.jpg"),
    text=SCRIPT.strip(),
    voice="rachel",          # or try "adam", "josh", "bella"
    loop_audio=True,         # loop TTS audio to match video length
    filename_prefix="demo",
    seed=42,
)

print("\nGeneration complete:")
print(f"  Video : {result['video']}")
print(f"  Audio : {result['audio']}")
print(f"  Final : {result['final']}")
