#!/bin/bash
# DKI System Setup Script for Linux/Mac
# ======================================

set -e

echo "========================================"
echo "  DKI System Setup"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Set working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
echo "Working directory: $(pwd)"

# Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "[3/5] Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "[4/5] Creating directories..."
mkdir -p data
mkdir -p experiment_results
mkdir -p logs

# Initialize database
echo ""
echo "[5/5] Initializing database..."
python3 -c "from dki.database.connection import DatabaseManager; DatabaseManager()"
echo "Database initialized."

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To start the system:"
echo "  ./scripts/start.sh web        - Start Web UI"
echo "  ./scripts/start.sh api        - Start API server"
echo "  ./scripts/start.sh experiment - Run experiments"
echo ""
