# PyPI Upload Quick Start Guide

## Step 1: Prepare Your Package

### Modify `setup.py` to Remove C++ Build Requirement

The current `setup.py` tries to build C++ extensions, which will fail for most users. Let's make it optional:

```python
# hyperlane_client/setup.py - UPDATED VERSION

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
req_file = Path(__file__).parent.parent / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "grpcio>=1.50.0",
        "grpcio-tools>=1.50.0",
        "zeroconf>=0.60.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.14.0",  # CPU version by default
    ]

# Replace onnxruntime-gpu with CPU version if present
requirements = [r for r in requirements if 'onnxruntime-gpu' not in r]
if 'onnxruntime>=1.14.0' not in requirements:
    requirements.append('onnxruntime>=1.14.0')

setup(
    name="hyperlane",
    version="0.1.0",
    description="Distributed GPU inference engine for heterogeneous prosumer hardware",
    author="Hyperlane Team",
    author_email="contact@hyperlane.dev",
    url="https://github.com/yourusername/hyperlane",
    long_description=open(Path(__file__).parent.parent / "README.md").read(),
    long_description_content_type="text/markdown",
    
    packages=find_packages(),
    python_requires=">=3.9",
    
    install_requires=requirements,
    
    extras_require={
        "gpu": ["onnxruntime-gpu>=1.14.0"],  # Optional GPU support
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "pylint>=2.15.0",
            "twine>=4.0.0",
        ],
    },
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    
    keywords="distributed-inference GPU LLM pipeline-parallelism",
    
    project_urls={
        "Documentation": "https://github.com/yourusername/hyperlane/blob/main/README.md",
        "Source": "https://github.com/yourusername/hyperlane",
        "Bug Reports": "https://github.com/yourusername/hyperlane/issues",
    },
    
    # Don't build C++ extension - users can build separately
    # ext_modules=[],
    
    entry_points={
        "console_scripts": [
            "hyperlane-discover=hyperlane_client.discovery:main",
        ],
    },
)
```

### Create License File

```bash
cd /home/herb/Desktop/gpusharing
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Hyperlane Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
```

### Create MANIFEST.in

```bash
cat > /home/herb/Desktop/gpusharing/MANIFEST.in << 'EOF'
include README.md
include LICENSE
include requirements.txt
include Makefile
include .github/workflows/ci.yml
recursive-include hyperlane_client *.py
recursive-include proto *.proto
EOF
```

### Create pyproject.toml

```bash
cat > /home/herb/Desktop/gpusharing/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hyperlane"
version = "0.1.0"
description = "Distributed GPU inference engine for heterogeneous prosumer hardware"
readme = "README.md"
requires-python = ">=3.9"
authors = [
    {name = "Hyperlane Team", email = "contact@hyperlane.dev"},
]
license = {text = "MIT"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]
keywords = ["distributed-inference", "gpu", "llm", "pipeline-parallelism"]
EOF
```

---

## Step 2: Build Distribution Locally

```bash
cd /home/herb/Desktop/gpusharing/hyperlane_client

# Install build tools
pip install --upgrade setuptools wheel twine

# Build source distribution and wheel
python setup.py sdist bdist_wheel

# Check what was created
ls -lh dist/
# Expected output:
#   hyperlane-0.1.0.tar.gz (source)
#   hyperlane-0.1.0-py3-none-any.whl (wheel)
```

---

## Step 3: Test Installation Locally

```bash
# Test in a fresh virtual environment
python3 -m venv test_env
source test_env/bin/activate

# Install from your local build
pip install /home/herb/Desktop/gpusharing/hyperlane_client/dist/hyperlane-0.1.0-py3-none-any.whl

# Verify it works
python -c "from hyperlane_client import DiscoveryManager, AutoDistributedModel; print('✓ Import successful')"

# Test discovery
python -c "dm = DiscoveryManager(); print(f'✓ DiscoveryManager works')"

# Deactivate
deactivate
```

---

## Step 4: Set Up PyPI Credentials

### Create `.pypirc` file

```bash
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your TestPyPI token

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your PyPI token
EOF

chmod 600 ~/.pypirc
```

### Get PyPI Tokens

1. **TestPyPI**: https://test.pypi.org/account/
   - Create account if needed
   - Go to Settings → API tokens
   - Create new token for "Entire account"
   - Copy token (starts with `pypi-`)

2. **PyPI**: https://pypi.org/account/
   - Create account if needed
   - Go to Settings → API tokens
   - Create new token for "Entire account"
   - Copy token

---

## Step 5: Upload to TestPyPI (Recommended First!)

```bash
cd /home/herb/Desktop/gpusharing/hyperlane_client

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Expected output:
# Uploading hyperlane-0.1.0.tar.gz
# Uploading hyperlane-0.1.0-py3-none-any.whl
# View at: https://test.pypi.org/project/hyperlane/0.1.0/
```

### Test TestPyPI Installation

```bash
# In a fresh venv
python3 -m venv testpypi_env
source testpypi_env/bin/activate

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0

# Test
python -c "from hyperlane_client import DiscoveryManager; print('✓ Works!')"

deactivate
```

---

## Step 6: Upload to Production PyPI

Only do this after successful TestPyPI test!

```bash
cd /home/herb/Desktop/gpusharing/hyperlane_client

# Upload to PyPI
twine upload dist/*

# Expected output:
# Uploading hyperlane-0.1.0.tar.gz
# Uploading hyperlane-0.1.0-py3-none-any.whl
# View at: https://pypi.org/project/hyperlane/0.1.0/
```

### Test PyPI Installation

```bash
# In a fresh venv
python3 -m venv pypi_env
source pypi_env/bin/activate

# Install from PyPI (no -i flag needed, it's the default)
pip install hyperlane==0.1.0

# Test
python -c "from hyperlane_client import DiscoveryManager, AutoDistributedModel; print('✓ PyPI installation works!')"

# Clean up
deactivate
```

---

## Step 7: Announce It!

Once on PyPI, users can install with:

```bash
# Basic installation (CPU version - works anywhere)
pip install hyperlane

# With GPU support (requires CUDA 11.8+)
pip install hyperlane[gpu]

# For development
pip install hyperlane[dev]

# All together
pip install hyperlane[gpu,dev]
```

---

## Verification Checklist

Before uploading, verify all of these:

```bash
✓ setup.py parses without errors
  python setup.py check

✓ All imports work
  python -c "from hyperlane_client import *"

✓ Package builds cleanly
  python setup.py sdist bdist_wheel

✓ Wheel is correct size (10-50MB typical)
  ls -lh dist/*.whl

✓ No build artifacts left in source
  python setup.py clean --all

✓ LICENSE file exists
  [ -f LICENSE ] && echo "✓"

✓ README exists and parses
  python -c "import email.parser; print(len(open('README.md').read())) > 100"

✓ TestPyPI install works
  pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0

✓ Imports work from installed package
  python -c "import hyperlane_client"
```

---

## Troubleshooting Common Issues

### Issue: "Invalid Distribution"

**Cause:** Missing files or bad `setup.py`

**Fix:**
```bash
python setup.py check
twine check dist/*
```

### Issue: "Conflicts with existing release"

**Cause:** Version 0.1.0 already exists on PyPI

**Fix:**
```bash
# Increment version in setup.py
# 0.1.0 → 0.1.1

# Rebuild
rm -rf dist/ build/ *.egg-info/
python setup.py sdist bdist_wheel

# Re-upload
twine upload dist/*
```

### Issue: "403 Invalid credentials"

**Cause:** Wrong token in `.pypirc`

**Fix:**
```bash
# Verify token format (starts with pypi-)
grep "password" ~/.pypirc

# Regenerate token from PyPI website
# Update ~/.pypirc
```

### Issue: "ERROR: onnxruntime-gpu requires CUDA"

**Cause:** User has no GPU

**Fix:** Already handled! Users should install base package:
```bash
pip install hyperlane  # Gets CPU version

# If they have GPU later:
pip install hyperlane[gpu]
```

---

## After Upload

### What Users Can Do (✅ Day 1)

```python
from hyperlane_client import DiscoveryManager, AutoDistributedModel

# Auto-discover workers on LAN
dm = DiscoveryManager()
workers = dm.get_available_workers()

# Load and shard models
model = AutoDistributedModel.from_pretrained("meta-llama/Llama-2-7b")

# Partition across discovered workers
model.shard(workers)
```

### What Users Cannot Do Yet (❌ Needs C++ Build)

```bash
# C++ worker not included in pip package
# Users need separate build:
git clone https://github.com/yourusername/hyperlane
cd hyperlane
bash build.sh  # Builds C++ worker locally
```

---

## Long-Term: Binary Wheels (Optional)

If you want users to install everything with `pip install hyperlane[gpu]`:

1. Set up GitHub Actions with CUDA container
2. Use `cibuildwheel` to build multi-platform wheels
3. Upload wheels to PyPI
4. Estimated effort: 4-6 weeks

For now, CPU-only Python package is a great MVP!

---

## Commands Summary

```bash
# Prepare
cd /home/herb/Desktop/gpusharing/hyperlane_client
pip install setuptools wheel twine

# Build
python setup.py sdist bdist_wheel

# Test locally
pip install dist/hyperlane-0.1.0-py3-none-any.whl

# Test on TestPyPI
twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0

# Production
twine upload dist/*
```

That's it! You're ready to share with the world.
