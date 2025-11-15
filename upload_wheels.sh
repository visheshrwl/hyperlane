#!/bin/bash

# Upload wheels to PyPI
# Usage: bash upload_wheels.sh [test|prod] [wheel_directory]

set -e

ENVIRONMENT="${1:-test}"
WHEEL_DIR="${2:-.}"

if [ "$ENVIRONMENT" != "test" ] && [ "$ENVIRONMENT" != "prod" ]; then
    echo "Usage: bash upload_wheels.sh [test|prod] [wheel_directory]"
    echo ""
    echo "Environment options:"
    echo "  test  - Upload to TestPyPI (recommended for first test)"
    echo "  prod  - Upload to PyPI (production)"
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              📦 Uploading Wheels to PyPI                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Target: ${ENVIRONMENT^^} PyPI${NC}"
echo ""

# Find wheels
WHEELS=$(find "$WHEEL_DIR" -name "hyperlane-*.whl" 2>/dev/null || echo "")

if [ -z "$WHEELS" ]; then
    echo -e "${RED}✗ No wheels found in $WHEEL_DIR${NC}"
    exit 1
fi

WHEEL_COUNT=$(echo "$WHEELS" | wc -l)
echo -e "${GREEN}✓ Found $WHEEL_COUNT wheel(s):${NC}"
echo "$WHEELS" | while read -r wheel; do
    echo "  └─ $(basename "$wheel")"
done
echo ""

# Check twine installation
if ! command -v twine &> /dev/null; then
    echo -e "${YELLOW}Installing twine...${NC}"
    python3 -m pip install twine --quiet
fi

# Verify wheels with twine
echo -e "${BLUE}Verifying wheels...${NC}"
if twine check "$WHEEL_DIR"/hyperlane-*.whl; then
    echo -e "${GREEN}✓ All wheels are valid${NC}\n"
else
    echo -e "${RED}✗ Wheel validation failed${NC}"
    exit 1
fi

# Upload to appropriate repository
if [ "$ENVIRONMENT" = "test" ]; then
    REPO="testpypi"
    REPO_URL="https://test.pypi.org/project/hyperlane/"
    TOKEN_VAR="TESTPYPI_API_TOKEN"
else
    REPO="pypi"
    REPO_URL="https://pypi.org/project/hyperlane/"
    TOKEN_VAR="PYPI_API_TOKEN"
fi

echo -e "${BLUE}Checking credentials...${NC}"

# Check if token is set
if [ -z "${!TOKEN_VAR}" ]; then
    echo -e "${YELLOW}Token not found in environment variable: $TOKEN_VAR${NC}"
    echo ""
    echo "To set up authentication:"
    echo "  1. Get your API token from https://$([ "$ENVIRONMENT" = "test" ] && echo "test." || echo "")pypi.org/account/#api-tokens"
    echo "  2. Create ~/.pypirc with:"
    echo ""
    echo "    [distutils]"
    echo "    index-servers = testpypi pypi"
    echo ""
    echo "    [testpypi]"
    echo "    repository = https://test.pypi.org/legacy/"
    echo "    username = __token__"
    echo "    password = pypi-YOUR_TOKEN_HERE"
    echo ""
    echo "    [pypi]"
    echo "    repository = https://upload.pypi.org/legacy/"
    echo "    username = __token__"
    echo "    password = pypi-YOUR_TOKEN_HERE"
    echo ""
    echo "  OR use environment variable:"
    echo "    export $TOKEN_VAR='pypi-YOUR_TOKEN_HERE'"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ Credentials found${NC}\n"

# Upload
echo -e "${BLUE}Uploading to ${ENVIRONMENT^^} PyPI...${NC}"
echo ""

if [ "$ENVIRONMENT" = "test" ]; then
    if twine upload --repository testpypi "$WHEEL_DIR"/hyperlane-*.whl --skip-existing; then
        echo ""
        echo -e "${GREEN}✓ Upload to TestPyPI successful!${NC}"
        echo ""
        echo "View your package at:"
        echo "  $REPO_URL"
        echo ""
        echo "Install with:"
        echo "  pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0"
    else
        echo -e "${RED}✗ Upload failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ Uploading to PRODUCTION PyPI!${NC}"
    echo "This cannot be undone. Press Ctrl+C to cancel, or wait 5 seconds..."
    sleep 5
    echo ""
    
    if twine upload "$WHEEL_DIR"/hyperlane-*.whl --skip-existing; then
        echo ""
        echo -e "${GREEN}✓ Upload to PyPI successful!${NC}"
        echo ""
        echo "View your package at:"
        echo "  $REPO_URL"
        echo ""
        echo "Install with:"
        echo "  pip install hyperlane==0.1.0"
    else
        echo -e "${RED}✗ Upload failed${NC}"
        exit 1
    fi
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  ✓ Upload Complete                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
