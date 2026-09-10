@echo off
setlocal enabledelayedexpansion
title LocalVid AI — Installer

echo.
echo  ================================================
echo   LocalVid AI  -  First-Time Installation
echo  ================================================
echo.

:: ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo  Install Python 3.10 or 3.11 from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Python %PY_VER% found

:: ── Check FFmpeg ──────────────────────────────────────────────
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  [INFO] FFmpeg not found. Attempting install via winget...
    winget install --id Gyan.FFmpeg -e --silent
    if %errorlevel% neq 0 (
        echo  [WARN] winget install failed. Download FFmpeg manually:
        echo        https://ffmpeg.org/download.html
        echo        Then add it to your PATH and re-run this script.
    ) else (
        echo  [OK] FFmpeg installed via winget
    )
) else (
    echo  [OK] FFmpeg found
)

:: ── Create virtual environment ────────────────────────────────
echo.
echo  Creating virtual environment (.venv)...
if exist .venv (
    echo  [INFO] .venv already exists, skipping creation
) else (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created
)

call .venv\Scripts\activate.bat

:: ── PyTorch CPU (base for torch-directml) ────────────────────
echo.
echo  Installing PyTorch CPU base (required for AMD DirectML)...
echo  This may take 5-10 minutes...
pip install torch==2.2.2 torchvision==0.17.2 ^
    --index-url https://download.pytorch.org/whl/cpu ^
    --quiet
if %errorlevel% neq 0 (
    echo  [ERROR] PyTorch install failed. Check your internet connection.
    pause
    exit /b 1
)
echo  [OK] PyTorch installed

:: ── All other requirements ────────────────────────────────────
echo.
echo  Installing remaining dependencies...
echo  This may take 10-20 minutes on first run...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo  [WARN] Some packages may have failed. Check output above.
)
echo  [OK] Dependencies installed

:: ── Copy .env ─────────────────────────────────────────────────
echo.
if not exist .env (
    copy .env.example .env >nul
    echo  [ACTION] .env file created from template.
    echo.
    echo  *** IMPORTANT ***
    echo  Open .env in a text editor and set your ELEVENLABS_API_KEY.
    echo  Get a free key at: https://elevenlabs.io
    echo  ****************
) else (
    echo  [OK] .env already exists
)

:: ── Done ──────────────────────────────────────────────────────
echo.
echo  ================================================
echo   Installation complete!
echo.
echo   Next steps:
echo   1. Edit .env  — add your ELEVENLABS_API_KEY
echo   2. Run start.bat to launch the app
echo  ================================================
echo.
pause
