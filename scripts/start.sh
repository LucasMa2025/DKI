#!/bin/bash
# DKI System Startup Script for Linux/Mac
# ========================================

set -e

echo "========================================"
echo "  DKI - Dynamic KV Injection System"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    exit 1
fi

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
echo "Working directory: $(pwd)"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "[WARNING] Virtual environment not found. Using system Python."
fi

# Initialize database if needed
mkdir -p data
mkdir -p experiment_results

if [ ! -f "data/dki.db" ]; then
    echo "Initializing database..."
    if command -v sqlite3 &> /dev/null; then
        sqlite3 data/dki.db < scripts/init_db.sql 2>/dev/null || true
    fi
fi

# Set environment variables
export DKI_CONFIG_PATH="config/config.yaml"
export PYTHONPATH="$(pwd)"

# Parse arguments
MODE="${1:-web}"

echo ""
echo "Starting in $MODE mode..."
echo ""

case "$MODE" in
    web)
        echo "Starting Web UI at http://localhost:8080"
        python3 -m dki.web.app
        ;;
    api)
        echo "Starting API server at http://localhost:8080"
        uvicorn dki.web.app:create_app --factory --host 0.0.0.0 --port 8080
        ;;
    experiment)
        echo "Running experiments..."
        python3 -m dki.experiment.runner
        ;;
    generate-data)
        echo "Generating experiment data..."
        python3 -m dki.experiment.data_generator
        ;;
    *)
        echo "Usage: $0 [web|api|experiment|generate-data]"
        exit 1
        ;;
esac
