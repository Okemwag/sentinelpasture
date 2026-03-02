#!/bin/bash

# Start the Go API service.

echo "Starting National Risk Intelligence Backend API..."
echo ""

# Check if Go is installed
if ! command -v go &> /dev/null; then
    echo "Go is not installed. Please install Go 1.22 or higher."
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "apps/api" ]; then
    echo "apps/api directory not found. Run this script from the project root."
    exit 1
fi

echo "Starting Go API server on http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd apps/api
API_ADDR=:8000 GOCACHE=/tmp/go-build go run ./cmd/api
