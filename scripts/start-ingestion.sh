#!/bin/bash

set -e

echo "Starting Governance Intel ingestion service..."
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

cd apps/ingestion

if [ ! -d ".venv" ]; then
    echo "Creating apps/ingestion virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -e .

echo "Ingestion API: http://localhost:8300"
echo ""

python3 -m uvicorn ingestion_service.main:app --host 0.0.0.0 --port 8300 --reload
