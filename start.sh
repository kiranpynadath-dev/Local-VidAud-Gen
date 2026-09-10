#!/usr/bin/env bash
# LocalVid AI — startup script for Linux/macOS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Check venv ────────────────────────────────────────────────
if [ ! -f ".venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found. Run install.sh first."
    exit 1
fi

if [ ! -f ".env" ]; then
    echo "[INFO] No .env found — using defaults (no API key required for local TTS)."
fi

source .venv/bin/activate

echo ""
echo "  Starting LocalVid AI server..."
echo "  Open http://localhost:8000 in your browser"
echo "  Press Ctrl+C to stop."
echo ""

python server.py
