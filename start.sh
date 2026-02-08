#!/bin/bash
# Quick Start Guide for Emotion Recognition MVP

echo "================================"
echo "Emotion Recognition MVP Setup"
echo "================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first:"
    echo "   https://www.docker.com/products/docker-desktop"
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
