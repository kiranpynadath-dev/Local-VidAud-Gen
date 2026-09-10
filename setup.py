from setuptools import setup, find_packages

setup(
    name="local-vidaud-gen",
    version="1.0.0",
    description="Local Video & Audio Generation for AMD Ryzen AI hardware",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "diffusers>=0.27.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "tqdm>=4.60.0",
        "imageio>=2.30.0",
        "imageio-ffmpeg>=0.4.0",
        "opencv-python>=4.8.0",
        "ffmpeg-python>=0.2.0",
        "elevenlabs>=1.0.0",
        "python-dotenv>=1.0.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "directml": ["torch-directml>=0.2.0"],
        "npu": ["onnxruntime-directml>=1.17.0", "onnx>=1.15.0", "olive-ai>=0.5.0"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "vidaudgen=cli:main",
        ]
    },
)
