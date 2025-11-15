#!/bin/bash
# Build script for Hyperlane project
# Builds both C++ worker and Python client components

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_DIR="$PROJECT_ROOT/hyperlane_worker"
CLIENT_DIR="$PROJECT_ROOT/hyperlane_client"
PYBIND_DIR="$CLIENT_DIR/pybind"

echo "======================================"
echo "Hyperlane Build Script"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check dependencies
echo -e "${YELLOW}[1] Checking dependencies...${NC}"

command -v cmake &> /dev/null || { echo -e "${RED}cmake not found${NC}"; exit 1; }
command -v python3 &> /dev/null || { echo -e "${RED}python3 not found${NC}"; exit 1; }
command -v protoc &> /dev/null || { echo -e "${RED}protoc not found${NC}"; exit 1; }

echo -e "${GREEN}✓ Dependencies OK${NC}"

# Build proto
echo -e "${YELLOW}[2] Generating gRPC stubs from proto...${NC}"
cd "$PROJECT_ROOT"
python3 "$CLIENT_DIR/generate_grpc.py"
echo -e "${GREEN}✓ gRPC stubs generated${NC}"

# Build C++ worker
echo -e "${YELLOW}[3] Building hyperlane_worker (C++/CUDA)...${NC}"
mkdir -p "$WORKER_DIR/build"
cd "$WORKER_DIR/build"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-march=native -O3"

cmake --build . -j$(nproc)
echo -e "${GREEN}✓ hyperlane_worker built${NC}"

# Build pybind11 extension
echo -e "${YELLOW}[4] Building pybind11 TensorSocket extension...${NC}"
mkdir -p "$PYBIND_DIR/build"
cd "$PYBIND_DIR/build"

python3 -m pip install pybind11 &> /dev/null || true

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(python3 -c 'import sys; print(sys.executable)')"

cmake --build . -j$(nproc)
echo -e "${GREEN}✓ pybind11 extension built${NC}"

# Install Python package
echo -e "${YELLOW}[5] Installing hyperlane Python package...${NC}"
cd "$CLIENT_DIR"

# Copy built extension to package
cp "$PYBIND_DIR/build"/*.so "$CLIENT_DIR/hyperlane_client/" 2>/dev/null || true

python3 -m pip install -e .
echo -e "${GREEN}✓ Python package installed${NC}"

echo ""
echo -e "${GREEN}======================================"
echo "Build Complete!"
echo "=====================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Start a worker: $WORKER_DIR/build/hyperlane_worker [port]"
echo "  2. Use the client: python3 -c 'from hyperlane_client import *'"
echo ""
