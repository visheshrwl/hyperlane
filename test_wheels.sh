#!/bin/bash

# Test wheels on multiple Python versions
# Usage: bash test_wheels.sh [wheel_directory]

set -e

WHEEL_DIR="${1:-.}"
PYTHON_VERSIONS=("3.9" "3.10" "3.11" "3.12")

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           🧪 Testing Wheels on Multiple Python Versions        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Find all wheels
WHEELS=$(find "$WHEEL_DIR" -name "hyperlane-*.whl" 2>/dev/null || echo "")

if [ -z "$WHEELS" ]; then
    echo -e "${RED}✗ No wheels found in $WHEEL_DIR${NC}"
    exit 1
fi

WHEEL_COUNT=$(echo "$WHEELS" | wc -l)
echo -e "${BLUE}Found $WHEEL_COUNT wheel(s):${NC}"
echo "$WHEELS" | sed 's/^/  /'
echo ""

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test each wheel
while IFS= read -r wheel_file; do
    wheel_name=$(basename "$wheel_file")
    
    # Extract Python version from wheel name
    # Format: hyperlane-0.1.0-cp39-cp39-linux_x86_64.whl
    if [[ $wheel_name =~ cp([0-9]+) ]]; then
        wheel_py_version="${BASH_REMATCH[1]}"
        wheel_py_version="${wheel_py_version:0:1}.${wheel_py_version:1}"
    else
        echo -e "${YELLOW}⊘ Could not determine Python version from $wheel_name${NC}"
        continue
    fi
    
    echo -e "${BLUE}Testing: $wheel_name${NC}"
    echo -e "  Target Python: $wheel_py_version"
    
    if ! command -v python$wheel_py_version &> /dev/null; then
        echo -e "  ${YELLOW}⊘ Python $wheel_py_version not found (skipping)${NC}\n"
        continue
    fi
    
    # Create isolated test environment
    test_dir="/tmp/hyperlane_test_$$_$RANDOM"
    mkdir -p "$test_dir"
    cd "$test_dir"
    
    # Create venv
    python$wheel_py_version -m venv venv
    source venv/bin/activate
    
    # Install dependencies and wheel
    python -m pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    # Install requirements
    python -m pip install \
        grpcio \
        zeroconf \
        torch \
        transformers \
        onnx \
        onnxruntime \
        > /dev/null 2>&1 || true
    
    # Install the wheel
    python -m pip install "$wheel_file" > /dev/null 2>&1
    
    # Run tests
    ((TOTAL_TESTS++))
    
    echo -e "  ${BLUE}Running import tests...${NC}"
    
    test_failed=0
    
    # Test 1: Basic imports
    if python << 'TESTEOF'
from hyperlane_client import DiscoveryManager, AutoDistributedModel
print("    ✓ Main imports successful")
TESTEOF
    then
        :
    else
        echo -e "    ${RED}✗ Failed to import main modules${NC}"
        test_failed=1
    fi
    
    # Test 2: Module imports
    if python << 'TESTEOF'
from hyperlane_client.discovery import DiscoveryManager
from hyperlane_client.orchestrator import AutoDistributedModel, KnapsackPartitioner
from hyperlane_client.grpc_client import WorkerClient
print("    ✓ Sub-module imports successful")
TESTEOF
    then
        :
    else
        echo -e "    ${RED}✗ Failed to import sub-modules${NC}"
        test_failed=1
    fi
    
    # Test 3: Instantiation
    if python << 'TESTEOF'
from hyperlane_client import DiscoveryManager
from hyperlane_client.orchestrator import KnapsackPartitioner

dm = DiscoveryManager()
print("    ✓ DiscoveryManager instantiated")

partitioner = KnapsackPartitioner()
print("    ✓ KnapsackPartitioner instantiated")
TESTEOF
    then
        :
    else
        echo -e "    ${RED}✗ Failed to instantiate classes${NC}"
        test_failed=1
    fi
    
    # Test 4: Basic functionality
    if python << 'TESTEOF'
from hyperlane_client.orchestrator import KnapsackPartitioner

partitioner = KnapsackPartitioner()

# Test data
layers = [
    {'name': 'layer1', 'size_bytes': 1024 * 1024 * 500},
    {'name': 'layer2', 'size_bytes': 1024 * 1024 * 500},
]
workers = [
    {'name': 'worker1', 'vram_bytes': 1024 * 1024 * 1024 * 24},
]

partition = partitioner.partition(layers, workers)
assert len(partition) == len(workers)
print("    ✓ Partitioning logic works")
TESTEOF
    then
        :
    else
        echo -e "    ${RED}✗ Failed functionality tests${NC}"
        test_failed=1
    fi
    
    # Clean up
    deactivate 2>/dev/null || true
    cd /
    rm -rf "$test_dir"
    
    # Record results
    if [ $test_failed -eq 0 ]; then
        echo -e "  ${GREEN}✓ All tests passed for Python $wheel_py_version${NC}\n"
        ((PASSED_TESTS++))
    else
        echo -e "  ${RED}✗ Some tests failed for Python $wheel_py_version${NC}\n"
        ((FAILED_TESTS++))
    fi
    
done <<< "$WHEELS"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                       Test Summary                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "Total wheels tested: $WHEEL_COUNT"
echo -e "Test cases passed:  ${GREEN}$PASSED_TESTS${NC}/$TOTAL_TESTS"
echo -e "Test cases failed:  $([ $FAILED_TESTS -eq 0 ] && echo -e "${GREEN}$FAILED_TESTS${NC}" || echo -e "${RED}$FAILED_TESTS${NC}")"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit 1
fi
