"""Example: Batch-process a folder of images into videos with TTS audio."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.generator import VideoAudioGenerator
from src.batch_processor import BatchProcessor

SCRIPTS = {
    "product_a": "Introducing Product A — the future of performance.",
    "product_b": "Product B delivers reliability you can count on.",
    "product_c": "Experience Product C, designed for professionals.",
}

gen = VideoAudioGenerator(quality="fast", output_dir="output/batch")
proc = gen.create_batch_processor(max_workers=1)  # sequential for GPU safety

input_dir = Path("examples/batch_inputs")
output_dir = Path("output/batch")
output_dir.mkdir(parents=True, exist_ok=True)

for name, script in SCRIPTS.items():
    image_path = input_dir / f"{name}.jpg"
    if not image_path.exists():
        print(f"Skipping {name} — image not found at {image_path}")
        continue

    proc.add_job(
        gen.generate,
        image_path,
        text=script,
        output_dir=output_dir,
        filename_prefix=name,
        job_id=name,
    )

print(f"Processing {len(proc._jobs)} jobs …\n")


def on_complete(job):
    icon = "✓" if job.status.value == "done" else "✗"
    elapsed = f"{job.elapsed:.1f}s" if job.elapsed else "?"
    print(f"  {icon} {job.job_id:20s} [{elapsed}]")
    if job.error:
        print(f"    Error: {job.error}")


batch_result = proc.run(progress_callback=on_complete)

print(f"\n{batch_result.summary()}")
for job in batch_result.successful:
    if job.result:
        print(f"  → {job.result.get('final', 'unknown')}")
