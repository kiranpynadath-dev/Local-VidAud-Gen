"""Example: Generate a silent video from a single image."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.generator import VideoAudioGenerator

# Auto-detects best device (NPU > DirectML > CPU)
gen = VideoAudioGenerator(quality="medium")

print("Device:", gen.device_info())

result = gen.generate_video_only(
    image=Path("examples/sample_input.jpg"),
    output_path=Path("output/basic_video.mp4"),
    seed=42,
)

print(f"Video generated: {result}")
