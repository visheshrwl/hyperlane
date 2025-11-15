#!/bin/bash

# Build and test wheels locally for multiple Python versions
# Usage: bash build_wheels_local.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HYPERLANE_CLIENT_DIR="$PROJECT_ROOT/hyperlane_client"
DIST_DIR="$PROJECT_ROOT/dist"
PYTHON_VERSIONS=("3.9" "3.10" "3.11" "3.12")

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          🔨 Building Wheels for Multiple Python Versions       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Cleanup old builds
echo -e "${BLUE}[1/7] Cleaning up old builds...${NC}"
cd "$HYPERLANE_CLIENT_DIR"
rm -rf dist/ build/ *.egg-info/ __pycache__
echo -e "${GREEN}✓ Cleanup complete${NC}\n"

# Check Python installation
echo -e "${BLUE}[2/7] Checking Python installations...${NC}"
for py_version in "${PYTHON_VERSIONS[@]}"; do
    if command -v python$py_version &> /dev/null; then
        actual_version=$(python$py_version --version 2>&1 | awk '{print $2}')
        echo -e "${GREEN}✓${NC} python$py_version: $actual_version"
    else
        echo -e "${YELLOW}⚠${NC} python$py_version: NOT FOUND (will skip)"
    fi
done
echo ""

# Install build dependencies
echo -e "${BLUE}[3/7] Installing build dependencies...${NC}"
python3 -m pip install --upgrade pip setuptools wheel build > /dev/null 2>&1
python3 -m pip install cmake ninja pybind11 twine > /dev/null 2>&1
echo -e "${GREEN}✓ Build dependencies installed${NC}\n"

# Create distribution directory
mkdir -p "$DIST_DIR"

# Build wheels for each Python version
echo -e "${BLUE}[4/7] Building wheels...${NC}"
WHEELS_BUILT=()
for py_version in "${PYTHON_VERSIONS[@]}"; do
    if ! command -v python$py_version &> /dev/null; then
        echo -e "${YELLOW}⊘${NC} Skipping Python $py_version (not installed)"
        continue
    fi
    
    echo -e "\n${BLUE}Building for Python $py_version...${NC}"
    
    # Create venv for each Python version
    venv_dir="$PROJECT_ROOT/.venv_build_py$py_version"
    rm -rf "$venv_dir"
    python$py_version -m venv "$venv_dir"
    source "$venv_dir/bin/activate"
    
    # Install build tools
    python -m pip install --upgrade pip setuptools wheel build > /dev/null 2>&1
    python -m pip install cmake ninja pybind11 > /dev/null 2>&1
    python -m pip install -r "$PROJECT_ROOT/requirements.txt" > /dev/null 2>&1
    
    # Build wheel
    cd "$HYPERLANE_CLIENT_DIR"
    if python -m build --wheel -C='--build-option=--verbose' 2>&1 | grep -q "Successfully"; then
        wheel_file=$(ls -t dist/hyperlane-*.whl 2>/dev/null | head -1)
        if [ -n "$wheel_file" ]; then
            cp "$wheel_file" "$DIST_DIR/"
            WHEELS_BUILT+=("$wheel_file")
            echo -e "${GREEN}✓${NC} Built: $(basename $wheel_file)"
        fi
    else
        echo -e "${RED}✗${NC} Failed to build for Python $py_version"
    fi
    
    # Clean up venv
    deactivate 2>/dev/null || true
    rm -rf "$venv_dir"
done

if [ ${#WHEELS_BUILT[@]} -eq 0 ]; then
    echo -e "${RED}✗ No wheels built successfully!${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Built ${#WHEELS_BUILT[@]} wheel(s)${NC}\n"

# Test wheels
echo -e "${BLUE}[5/7] Testing wheels...${NC}"
TESTS_PASSED=0
TESTS_FAILED=0

for py_version in "${PYTHON_VERSIONS[@]}"; do
    if ! command -v python$py_version &> /dev/null; then
        continue
    fi
    
    echo -e "\n${BLUE}Testing with Python $py_version...${NC}"
    
    # Create test venv
    test_venv_dir="$PROJECT_ROOT/.venv_test_py$py_version"
    rm -rf "$test_venv_dir"
    python$py_version -m venv "$test_venv_dir"
    source "$test_venv_dir/bin/activate"
    
    # Find matching wheel
    wheel_file=$(ls -t "$DIST_DIR"/hyperlane-*-cp${py_version//./}*.whl 2>/dev/null | head -1)
    
    if [ -z "$wheel_file" ]; then
        echo -e "${YELLOW}⊘${NC} No wheel found for Python $py_version"
        deactivate
        continue
    fi
    
    # Install dependencies
    python -m pip install --upgrade pip > /dev/null 2>&1
    python -m pip install -r "$PROJECT_ROOT/requirements.txt" > /dev/null 2>&1
    
    # Install wheel
    python -m pip install "$wheel_file" > /dev/null 2>&1
    
    # Test imports
    if python << 'TESTEOF'
import sys
try:
    from hyperlane_client import DiscoveryManager, AutoDistributedModel
    from hyperlane_client.orchestrator import KnapsackPartitioner
    from hyperlane_client.grpc_client import WorkerClient
    print("✓ All imports successful")
    sys.exit(0)
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
TESTEOF
    then
        echo -e "${GREEN}✓${NC} Imports passed for Python $py_version"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗${NC} Imports failed for Python $py_version"
        ((TESTS_FAILED++))
    fi
    
    # Clean up test venv
    deactivate 2>/dev/null || true
    rm -rf "$test_venv_dir"
done

echo -e "\n${GREEN}✓ Tests passed: $TESTS_PASSED / $(($TESTS_PASSED + $TESTS_FAILED))${NC}\n"

# Verify wheel contents
echo -e "${BLUE}[6/7] Verifying wheel contents...${NC}"
python3 << 'VERIFYEOF'
import zipfile
import glob
from pathlib import Path

wheels = sorted(glob.glob('dist/hyperlane-*.whl'))
print(f"\n📦 Found {len(wheels)} wheel(s)\n")

for wheel_path in wheels:
    print(f"Wheel: {Path(wheel_path).name}")
    with zipfile.ZipFile(wheel_path, 'r') as zf:
        # Count file types
        py_files = [n for n in zf.namelist() if n.endswith('.py')]
        so_files = [n for n in zf.namelist() if n.endswith('.so')]
        dist_info = [n for n in zf.namelist() if '.dist-info' in n]
        
        print(f"  ├─ Python files: {len(py_files)}")
        print(f"  ├─ Binary extensions: {len(so_files)}")
        if so_files:
            for so in so_files[:3]:  # Show first 3
                print(f"  │  └─ {so}")
        print(f"  └─ Metadata files: {len(dist_info)}\n")

print(f"✓ Wheel verification complete")
VERIFYEOF

# Summary
echo -e "${BLUE}[7/7] Summary${NC}"
echo ""
echo -e "${GREEN}Build Complete!${NC}"
echo ""
echo "Wheels location: $DIST_DIR"
echo "Wheels built:"
ls -lh "$DIST_DIR"/hyperlane-*.whl
echo ""
echo "Next steps:"
echo "  1. Test locally: pip install $DIST_DIR/hyperlane-*.whl"
echo "  2. Upload to PyPI: twine upload $DIST_DIR/*.whl"
echo ""

# Cleanup temp environments
echo -e "${BLUE}Cleaning up temporary environments...${NC}"
rm -rf "$PROJECT_ROOT"/.venv_build_py*
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    ✓ Build Process Complete                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
