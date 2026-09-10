#!/usr/bin/env bash
# LocalVid AI — installer for macOS (Apple Silicon & Intel)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ================================================"
echo "   LocalVid AI  -  macOS Installation"
echo "  ================================================"
echo ""

# ── Python ────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found."
    echo "  Install via: brew install python@3.11"
    echo "  Or download from https://python.org"
    exit 1
fi
PY_VER=$(python3 --version)
echo "[OK] $PY_VER"

# Warn if Python is too old
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "[WARN] Python 3.10+ is recommended. You have Python 3.${PY_MINOR}."
fi

# ── Homebrew ──────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    echo "[INFO] Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "[OK] Homebrew found"
fi

# ── FFmpeg ────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "[INFO] Installing FFmpeg via Homebrew..."
    brew install ffmpeg
else
    echo "[OK] FFmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
fi

# ── Virtual environment ───────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "[OK] Virtual environment created"
else
    echo "[OK] .venv already exists"
fi

source .venv/bin/activate

# ── Upgrade pip ───────────────────────────────────────────────
pip install --upgrade pip --quiet

# ── PyTorch (MPS support built-in on Apple Silicon) ──────────
echo ""
echo "Installing PyTorch (MPS acceleration for Apple Silicon)..."
echo "This may take 5-10 minutes..."

# Detect Apple Silicon vs Intel
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "[INFO] Apple Silicon detected — PyTorch MPS will be used"
    pip install torch==2.2.2 torchvision==0.17.2 --quiet
else
    echo "[INFO] Intel Mac detected — running on CPU (MPS unavailable)"
    pip install torch==2.2.2 torchvision==0.17.2 --quiet
fi
echo "[OK] PyTorch installed"

# ── Mac-specific requirements ─────────────────────────────────
echo ""
echo "Installing dependencies (may take 10-20 min on first run)..."
pip install -r requirements_mac.txt --quiet
echo "[OK] Dependencies installed"

# ── Pre-download Kokoro TTS model (~312 MB) ───────────────────
echo ""
echo "Pre-downloading Kokoro TTS model (~312 MB, one time only)..."
python3 -c "from src.tts_generator import TTSGenerator; TTSGenerator().download_models()" && \
    echo "[OK] Kokoro TTS model ready" || \
    echo "[WARN] Kokoro model download failed — will retry on first use"

# ── .env (optional — no API key required) ────────────────────
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo "# LocalVid AI config (no API keys required for local TTS)" > .env
        echo "# Optional: HF_TOKEN=hf_xxx  (only if HuggingFace model download requires login)" >> .env
    fi
    echo "[OK] .env created (no API keys required)"
fi

# ── Verify MPS ────────────────────────────────────────────────
echo ""
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    python3 -c "
import torch
if torch.backends.mps.is_available():
    print('[OK] Apple MPS (Metal) acceleration is available')
else:
    print('[WARN] MPS not available — will use CPU (requires macOS 12.3+ and Apple Silicon)')
" 2>/dev/null || echo "[WARN] Could not verify MPS status"
fi

echo ""
echo "  ================================================"
echo "   Installation complete!"
echo ""
echo "   Run:  bash start.sh"
echo "   Then open http://localhost:8000"
echo ""
if [ "$(uname -m)" = "arm64" ]; then
    echo "   Hardware: Apple Silicon MPS will be used"
else
    echo "   Hardware: Intel Mac — CPU mode (slower)"
fi
echo "  ================================================"
echo ""
