#!/bin/bash

# Start the Python API service.

echo "Starting National Risk Intelligence Backend API..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "apps/api" ]; then
    echo "apps/api directory not found. Run this script from the project root."
    exit 1
fi

cd apps/api

if [ ! -d ".venv" ]; then
    echo "Creating apps/api virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -e . >/dev/null

export AI_INFERENCE_URL="${AI_INFERENCE_URL:-http://localhost:8100}"

echo "Starting Python API server on http://localhost:8000"
echo "HTTP endpoints are served from apps/api"
echo "AI inference upstream: ${AI_INFERENCE_URL}"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn api_service.main:app --host 0.0.0.0 --port 8000 --reload
