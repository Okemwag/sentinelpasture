#!/bin/bash

# Start Backend API Server
# This script starts the FastAPI backend with mock data

echo "🚀 Starting National Risk Intelligence Backend API..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9 or higher."
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "backend" ]; then
    echo "❌ backend directory not found. Please run this script from the project root."
    exit 1
fi

# Install dependencies if needed
if [ ! -f "backend/.venv/bin/activate" ]; then
    echo "📦 Installing backend dependencies..."
    cd backend
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# Start the backend
echo "✅ Starting FastAPI server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd backend
source .venv/bin/activate 2>/dev/null || true
python3 main.py
