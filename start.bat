@echo off
setlocal
title LocalVid AI

:: ── Check venv ────────────────────────────────────────────────
if not exist .venv\Scripts\activate.bat (
    echo  [ERROR] Virtual environment not found.
    echo  Run install.bat first.
    pause
    exit /b 1
)

:: ── Check .env ────────────────────────────────────────────────
if not exist .env (
    echo  [ERROR] .env file not found.
    echo  Run install.bat first, then edit .env with your API key.
    pause
    exit /b 1
)

:: ── Activate and start ────────────────────────────────────────
call .venv\Scripts\activate.bat

echo.
echo  Starting LocalVid AI server...
echo  The browser will open automatically.
echo  Press Ctrl+C to stop.
echo.

python server.py

:: If we get here, server stopped
echo.
echo  Server stopped.
pause
