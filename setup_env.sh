#!/bin/bash
# Hyperlane Development Environment Setup
# This script initializes the development environment with all dependencies

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

echo "========================================"
echo "Hyperlane Development Environment Setup"
echo "========================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "[1] Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[2] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "     Virtual environment created"
else
    echo "[2] Virtual environment exists"
fi

# Activate virtual environment
echo "[3] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "     Virtual environment activated"

# Upgrade pip
echo "[4] Upgrading pip, setuptools, wheel..."
pip install --quiet --upgrade pip setuptools wheel
echo "     Updated"

# Install dependencies
echo "[5] Installing project dependencies..."
pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
echo "     Dependencies installed"

# Install hyperlane_client package in editable mode
echo "[6] Installing hyperlane_client package..."
cd "$PROJECT_ROOT/hyperlane_client"
pip install --quiet -e .
cd "$PROJECT_ROOT"
echo "     Package installed"

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run tests: python3 tests.py"
echo "  3. Build C++ worker: bash build.sh (requires CUDA, gRPC, ONNX)"
echo ""
echo "Virtual environment location: $VENV_DIR"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"
echo ""
