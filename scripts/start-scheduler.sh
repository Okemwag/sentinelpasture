#!/bin/bash

set -e

echo "Starting Governance Intel scheduler service..."
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

cd apps/scheduler

if [ ! -d ".venv" ]; then
    echo "Creating apps/scheduler virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -e .

echo "Scheduler API: http://localhost:8200"
echo ""

python3 -m uvicorn scheduler_service.main:app --host 0.0.0.0 --port 8200 --reload
