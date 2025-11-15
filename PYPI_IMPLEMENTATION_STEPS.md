# Step-by-Step: Make Your Package PyPI-Ready

This document shows **exact code** you need to change. Copy-paste ready.

---

## Step 1: Create LICENSE File

Create `/home/herb/Desktop/gpusharing/LICENSE`:

```
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
```

---

## Step 2: Create MANIFEST.in

Create `/home/herb/Desktop/gpusharing/MANIFEST.in`:

```
include README.md
include LICENSE
include requirements.txt
include Makefile
include setup_env.sh
recursive-include hyperlane_client *.py
recursive-include proto *.proto
```

---

## Step 3: Create pyproject.toml

Create `/home/herb/Desktop/gpusharing/pyproject.toml`:

```toml
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

keywords = ["distributed-inference", "gpu", "llm", "pipeline-parallelism"]

classifiers = [
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
]

dependencies = [
    "grpcio>=1.50.0",
    "grpcio-tools>=1.50.0",
    "zeroconf>=0.60.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "onnx>=1.12.0",
    "onnxruntime>=1.14.0",
]

[project.optional-dependencies]
gpu = ["onnxruntime-gpu>=1.14.0"]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
    "pylint>=2.15.0",
    "twine>=4.0.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/hyperlane"
Documentation = "https://github.com/yourusername/hyperlane/blob/main/README.md"
Repository = "https://github.com/yourusername/hyperlane.git"
Issues = "https://github.com/yourusername/hyperlane/issues"
```

---

## Step 4: Update setup.py

Replace the entire `/home/herb/Desktop/gpusharing/hyperlane_client/setup.py` with:

```python
from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from parent directory
req_file = Path(__file__).parent.parent / "requirements.txt"

# Build requirements list
install_requires = [
    "grpcio>=1.50.0",
    "grpcio-tools>=1.50.0",
    "zeroconf>=0.60.0",
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "onnx>=1.12.0",
    "onnxruntime>=1.14.0",  # CPU version by default
]

# Ensure onnxruntime-gpu is NOT in the base requirements
install_requires = [r for r in install_requires if 'onnxruntime-gpu' not in r]

# Add CPU version if not already present
if not any('onnxruntime' in r for r in install_requires):
    install_requires.append('onnxruntime>=1.14.0')

# Read long description
readme_file = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text()

setup(
    name="hyperlane",
    version="0.1.0",
    description="Distributed GPU inference engine for heterogeneous prosumer hardware",
    author="Hyperlane Team",
    author_email="contact@hyperlane.dev",
    url="https://github.com/yourusername/hyperlane",
    
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    license="MIT",
    
    packages=find_packages(),
    
    python_requires=">=3.9",
    
    install_requires=install_requires,
    
    extras_require={
        "gpu": ["onnxruntime-gpu>=1.14.0"],  # Optional GPU acceleration
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
    
    keywords=[
        "distributed-inference",
        "gpu",
        "llm",
        "pipeline-parallelism",
        "inference-engine",
        "model-serving",
    ],
    
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/hyperlane/issues",
        "Documentation": "https://github.com/yourusername/hyperlane/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/hyperlane",
    },
    
    entry_points={
        "console_scripts": [
            "hyperlane-discover=hyperlane_client.discovery:main",
        ],
    },
    
    # NOTE: C++ extension excluded from PyPI package
    # Users can build it separately with: bash build.sh
    # See DEVELOPMENT.md for instructions
)
```

---

## Step 5: Update README.md

Add this Installation section at the top (after the title):

```markdown
## Installation

### Quick Install (CPU)

```bash
pip install hyperlane
```

This installs the Python orchestration layer, which works on any system without GPU hardware.

### With GPU Support

If you have NVIDIA GPU with CUDA 11.8+:

```bash
pip install hyperlane[gpu]
```

This includes GPU-optimized ONNX Runtime for inference.

### Development Version

To get the full system including C++ worker:

```bash
git clone https://github.com/yourusername/hyperlane
cd hyperlane
bash build.sh  # Compiles C++ worker (requires CUDA, gRPC, ONNX)
source venv/bin/activate
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for complete setup instructions.

---

## Quick Start

```python
from hyperlane_client import DiscoveryManager, AutoDistributedModel

# Auto-discover workers on LAN
discovery = DiscoveryManager()
workers = discovery.get_available_workers()

# Load and shard model
model = AutoDistributedModel.from_pretrained("meta-llama/Llama-2-7b-hf")
model.shard(workers)

# Run inference
output = model.generate(prompt="Hello", max_length=50)
```
```

---

## Step 6: Build the Distribution

```bash
cd /home/herb/Desktop/gpusharing/hyperlane_client

# Install build tools
pip install --upgrade setuptools wheel twine

# Clean old builds
rm -rf dist/ build/ *.egg-info/

# Build source distribution + wheel
python setup.py sdist bdist_wheel

# Verify builds
ls -lh dist/
# Should show:
#   hyperlane-0.1.0.tar.gz (source, ~1-2MB)
#   hyperlane-0.1.0-py3-none-any.whl (wheel, ~1-2MB)
```

---

## Step 7: Test Locally

```bash
# Create test environment
python3 -m venv test_env
source test_env/bin/activate

# Install from wheel
pip install /home/herb/Desktop/gpusharing/hyperlane_client/dist/hyperlane-0.1.0-py3-none-any.whl

# Test it works
python << 'EOF'
from hyperlane_client import DiscoveryManager, AutoDistributedModel
print("✓ DiscoveryManager imported")
print("✓ AutoDistributedModel imported")

dm = DiscoveryManager()
print("✓ DiscoveryManager instantiated")

print("\nAll imports and basic functionality working!")
EOF

# Cleanup
deactivate
rm -rf test_env
```

---

## Step 8: Create PyPI Credentials

### For TestPyPI:

1. Go to: https://test.pypi.org/account/
2. Create account if needed
3. Go to: Account Settings → API tokens
4. Click: "Add API token"
5. Copy the token (starts with `pypi-`)

### For PyPI:

1. Go to: https://pypi.org/account/
2. Create account if needed
3. Go to: Account Settings → API tokens
4. Click: "Add API token"
5. Copy the token (starts with `pypi-`)

### Setup `.pypirc`:

```bash
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE
EOF

chmod 600 ~/.pypirc
```

Replace `YOUR_TESTPYPI_TOKEN_HERE` and `YOUR_PYPI_TOKEN_HERE` with actual tokens.

---

## Step 9: Upload to TestPyPI

```bash
cd /home/herb/Desktop/gpusharing/hyperlane_client

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Expected output:
# Uploading hyperlane-0.1.0.tar.gz
# Uploading hyperlane-0.1.0-py3-none-any.whl
# https://test.pypi.org/project/hyperlane/0.1.0/
```

### Test TestPyPI Installation

```bash
# New environment
python3 -m venv test_pypi_env
source test_pypi_env/bin/activate

# Install from TestPyPI
pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0

# Test
python -c "from hyperlane_client import DiscoveryManager; print('✓ Works!')"

deactivate
```

**If this works, proceed to Step 10. If it fails, fix the issues and redo steps 6-9.**

---

## Step 10: Upload to Production PyPI

```bash
cd /home/herb/Desktop/gpusharing/hyperlane_client

# Upload to PyPI (official)
twine upload dist/*

# Expected output:
# Uploading hyperlane-0.1.0.tar.gz
# Uploading hyperlane-0.1.0-py3-none-any.whl
# https://pypi.org/project/hyperlane/0.1.0/
```

**🎉 Your package is now on PyPI!**

---

## Step 11: Verify PyPI Installation

```bash
# New environment
python3 -m venv verify_env
source verify_env/bin/activate

# Install from PyPI (default, no -i flag needed)
pip install hyperlane

# Test all imports
python << 'EOF'
from hyperlane_client import DiscoveryManager, AutoDistributedModel
print("✓ Basic imports work")

dm = DiscoveryManager()
print("✓ DiscoveryManager works")

# Test discovery (may not find workers, but API should work)
try:
    workers = dm.get_available_workers()
    print(f"✓ Found {len(workers)} workers")
except Exception as e:
    print(f"✓ API callable (error expected without actual workers): {e}")

print("\nPyPI installation verified successfully!")
EOF

deactivate
```

---

## Step 12: Announce It!

Now users can install with:

```bash
# Basic installation
pip install hyperlane

# With GPU support
pip install hyperlane[gpu]

# For development
pip install hyperlane[dev]
```

---

## Verification Checklist

Before uploading to PyPI, verify:

```bash
# ✓ setup.py is valid
python /home/herb/Desktop/gpusharing/hyperlane_client/setup.py check

# ✓ Built packages are valid
twine check /home/herb/Desktop/gpusharing/hyperlane_client/dist/*

# ✓ All required files exist
[ -f /home/herb/Desktop/gpusharing/LICENSE ] && echo "✓ LICENSE"
[ -f /home/herb/Desktop/gpusharing/MANIFEST.in ] && echo "✓ MANIFEST.in"
[ -f /home/herb/Desktop/gpusharing/pyproject.toml ] && echo "✓ pyproject.toml"
[ -f /home/herb/Desktop/gpusharing/README.md ] && echo "✓ README.md"

# ✓ Wheel is reasonable size (50-100MB typical)
ls -lh /home/herb/Desktop/gpusharing/hyperlane_client/dist/*.whl

# ✓ Source distribution is small (<5MB)
ls -lh /home/herb/Desktop/gpusharing/hyperlane_client/dist/*.tar.gz

# ✓ Package installs and imports work
pip install /home/herb/Desktop/gpusharing/hyperlane_client/dist/*.whl
python -c "from hyperlane_client import *; print('✓')"
```

---

## Troubleshooting

### "Invalid Distribution"

```bash
# Check what's wrong
twine check dist/*

# Common fixes:
# 1. Missing long_description
# 2. Invalid README markdown
# 3. Missing LICENSE
```

### "Conflicts with existing release"

```bash
# Version already exists. Increment:
# 0.1.0 → 0.1.1

# Update setup.py version
# Rebuild
rm -rf dist/ build/ *.egg-info/
python setup.py sdist bdist_wheel

# Re-upload
twine upload dist/*
```

### "403 Invalid credentials"

```bash
# Wrong token in ~/.pypirc
# Get new token from:
# - TestPyPI: https://test.pypi.org/account/#api-tokens
# - PyPI: https://pypi.org/account/#api-tokens

# Update ~/.pypirc with correct token
```

### "onnxruntime-gpu requires CUDA"

```bash
# This happens if user tries to install [gpu] extra without CUDA
# They should install base: pip install hyperlane
# And build from source for GPU support: bash build.sh
```

---

## Next Steps

After uploading, consider:

1. ✅ Share PyPI link with users
2. ✅ Update GitHub README with `pip install hyperlane`
3. ✅ Create GitHub releases for each PyPI version
4. ✅ Set up CI/CD to auto-publish on release
5. (Later) Build binary wheels for common GPU configurations
6. (Later) Docker image for complete deployment

---

## Timeline

- **Step 1-5**: 30 minutes (create files, update setup.py)
- **Step 6-7**: 15 minutes (build and test locally)
- **Step 8**: 10 minutes (create PyPI credentials)
- **Step 9-10**: 5 minutes (upload to TestPyPI, then PyPI)
- **Step 11-12**: 10 minutes (verify and announce)

**Total: ~70 minutes to go from code to PyPI!**

---

## Example: Full Command Sequence

```bash
# 1. Create files
cat > /home/herb/Desktop/gpusharing/LICENSE << 'EOF'
MIT License...
EOF

cat > /home/herb/Desktop/gpusharing/MANIFEST.in << 'EOF'
include README.md
...
EOF

cat > /home/herb/Desktop/gpusharing/pyproject.toml << 'EOF'
[build-system]
...
EOF

# 2. Update setup.py (use editor or provided code)

# 3. Build
cd /home/herb/Desktop/gpusharing/hyperlane_client
pip install setuptools wheel twine
python setup.py sdist bdist_wheel

# 4. Test locally
python3 -m venv test_env
source test_env/bin/activate
pip install dist/*.whl
python -c "from hyperlane_client import *; print('✓')"
deactivate

# 5. Upload to TestPyPI
twine upload --repository testpypi dist/*

# 6. Test from TestPyPI
python3 -m venv test_pypi_env
source test_pypi_env/bin/activate
pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0
python -c "from hyperlane_client import *; print('✓')"
deactivate

# 7. Upload to PyPI
twine upload dist/*

# 8. Verify
python3 -m venv verify_env
source verify_env/bin/activate
pip install hyperlane
python -c "from hyperlane_client import *; print('✓')"
deactivate

echo "🎉 Done! Package is now on PyPI!"
```

That's it. You're done!
