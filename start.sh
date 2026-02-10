#!/bin/bash
# Quick Start Guide for Emotion Recognition MVP

echo "================================"
echo "Emotion Recognition MVP Setup"
echo "================================"
echo ""

if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
    echo "✓ WSL environment detected"
    case "$(pwd)" in
        /mnt/*)
            echo "⚠ Project is under /mnt/* (Windows filesystem)."
            echo "  For faster training and file I/O, keep project in Linux filesystem (e.g. /home/<user>/...)."
            ;;
    esac
    echo ""
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Engine/Desktop first:"
    echo "   https://docs.docker.com/engine/install/"
    exit 1
fi

echo "✓ Docker found"

# Create checkpoints directory if it doesn't exist
if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
    echo "✓ Created checkpoints directory"
fi

# Check for checkpoint
if [ ! -f "checkpoints/best.pt" ]; then
    echo "⚠ WARNING: No checkpoint found at ./checkpoints/best.pt"
    echo "  The system will run in MOCK MODE (random predictions) for testing."
    echo "  To use real predictions, place your trained model checkpoint at:"
    echo "    ./checkpoints/best.pt"
    echo ""
    read -p "Continue with mock mode? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Starting services..."
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:8080"
echo "  vLLM:     http://localhost:8001 (optional)"
echo ""
echo "To stop, press Ctrl+C"
echo ""

# Build and run
docker compose up --build
