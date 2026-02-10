@echo off
REM Quick Start Guide for Emotion Recognition MVP (Windows)

setlocal enabledelayedexpansion

echo ================================
echo Emotion Recognition MVP Setup
echo ================================
echo.

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo X Docker is not installed. Please install Docker Desktop first:
    echo   https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo [OK] Docker found
echo.

REM Create checkpoints directory if it doesn't exist
if not exist "checkpoints" (
    mkdir checkpoints
    echo [OK] Created checkpoints directory
)

REM Check for checkpoint
if not exist "checkpoints\best.pt" (
    echo [WARNING] No checkpoint found at .\checkpoints\best.pt
    echo The system will run in MOCK MODE for testing.
    echo To use real predictions, place your trained model checkpoint at:
    echo   .\checkpoints\best.pt
    echo.
    set /p continue="Continue with mock mode? [y/n] "
    if /i not "!continue!"=="y" (
        echo Aborted.
        exit /b 1
    )
)

echo.
echo Starting services...
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:8080
echo   vLLM:     http://localhost:8001 (optional)
echo.
echo To stop, press Ctrl+C
echo.

REM Build and run
docker compose up --build
