#!/bin/bash

# Start the contract-aligned AI inference service.

set -e

echo "Starting Governance Intel AI inference service..."
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

cd apps/ai

if [ ! -d ".venv" ]; then
    echo "Creating apps/ai virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -e .

echo "AI inference API: http://localhost:8100"
echo ""

python3 -m uvicorn ai.serving.api.main:app --host 0.0.0.0 --port 8100 --reload
