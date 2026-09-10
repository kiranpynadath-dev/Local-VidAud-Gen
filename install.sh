#!/usr/bin/env bash
# LocalVid AI — installer for Linux/macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "  ================================================"
echo "   LocalVid AI  -  Installation"
echo "  ================================================"
echo ""

# ── Python ────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Install Python 3.10+."
    exit 1
fi
PY_VER=$(python3 --version)
echo "[OK] $PY_VER"

# ── FFmpeg ────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    echo "[INFO] FFmpeg not found. Installing..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y ffmpeg
    elif command -v brew &>/dev/null; then
        brew install ffmpeg
    else
        echo "[WARN] Could not auto-install FFmpeg. Install manually: https://ffmpeg.org"
    fi
else
    echo "[OK] FFmpeg found"
fi

# ── Venv ──────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "[OK] Virtual environment ready"

# ── PyTorch ───────────────────────────────────────────────────
echo ""
echo "Installing PyTorch..."
pip install torch==2.2.2 torchvision==0.17.2 --quiet

# ── Requirements ──────────────────────────────────────────────
echo "Installing dependencies (may take 10-20 min)..."
pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed"

# ── .env ──────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "[ACTION] .env created. Add your ELEVENLABS_API_KEY to .env before running."
fi

echo ""
echo "  ================================================"
echo "   Done! Run:  bash start.sh"
echo "  ================================================"
echo ""
