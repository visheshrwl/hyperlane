# Building and Distributing Hyperlane Wheels

Complete guide for building wheels for multiple Python versions and uploading to PyPI.

## Overview

This guide covers:
- Building wheels locally for Python 3.9, 3.10, 3.11, 3.12
- Testing wheels on each Python version
- Uploading to TestPyPI (staging)
- Uploading to PyPI (production)
- GitHub Actions CI/CD for automated building and testing

## Quick Start

### 1. Build Wheels Locally

```bash
cd /home/herb/Desktop/gpusharing
bash build_wheels_local.sh
```

This will:
- ✅ Build wheels for all available Python versions
- ✅ Test each wheel
- ✅ Verify contents
- ✅ Output to `dist/` directory

**Expected output:**
```
dist/hyperlane-0.1.0-cp39-cp39-linux_x86_64.whl
dist/hyperlane-0.1.0-cp310-cp310-linux_x86_64.whl
dist/hyperlane-0.1.0-cp311-cp311-linux_x86_64.whl
dist/hyperlane-0.1.0-cp312-cp312-linux_x86_64.whl
```

### 2. Test Wheels

```bash
bash test_wheels.sh dist/
```

This tests:
- ✅ Import all modules
- ✅ Instantiate classes
- ✅ Run basic functionality
- ✅ On each Python version

### 3. Upload to TestPyPI (Recommended First)

```bash
# Set up token (one time)
export TESTPYPI_API_TOKEN="pypi-YOUR_TOKEN_HERE"

# Upload
bash upload_wheels.sh test dist/
```

Test installation:
```bash
pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0
python -c "from hyperlane_client import *; print('✓ Works!')"
```

### 4. Upload to Production PyPI

```bash
# Set up token (one time)
export PYPI_API_TOKEN="pypi-YOUR_TOKEN_HERE"

# Upload
bash upload_wheels.sh prod dist/
```

Users can now install:
```bash
pip install hyperlane==0.1.0
```

---

## Prerequisites

### System Requirements

```bash
# Ubuntu 20.04+ with:
- Python 3.9, 3.10, 3.11, 3.12
- CMake 3.18+
- C++ compiler (gcc)
- pip and setuptools

# Install missing Python versions (if needed):
sudo apt update
sudo apt install python3.9 python3.10 python3.11 python3.12
```

### PyPI Tokens

1. **Create TestPyPI account** (if needed):
   - https://test.pypi.org/account/register/

2. **Get TestPyPI token**:
   - Go to: https://test.pypi.org/account/#api-tokens
   - Click "Add API token"
   - Name it "hyperlane-test"
   - Copy token (starts with `pypi-`)

3. **Get PyPI token**:
   - Go to: https://pypi.org/account/#api-tokens
   - Click "Add API token"
   - Name it "hyperlane"
   - Copy token (starts with `pypi-`)

4. **Configure credentials** (one-time setup):

   Option A: Environment variables (temporary)
   ```bash
   export TESTPYPI_API_TOKEN="pypi-YOUR_TOKEN_HERE"
   export PYPI_API_TOKEN="pypi-YOUR_TOKEN_HERE"
   ```

   Option B: ~/.pypirc file (persistent)
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

---

## Detailed Workflows

### Building Wheels (Local)

**File:** `build_wheels_local.sh`

What it does:
1. Cleans old builds
2. Checks Python installations
3. Creates isolated venvs for each Python version
4. Installs dependencies in each venv
5. Builds wheel with that Python version
6. Tests wheel imports
7. Verifies wheel contents
8. Outputs to `dist/`

**Usage:**
```bash
bash build_wheels_local.sh
```

**Output:**
- `dist/hyperlane-0.1.0-cp{39,310,311,312}-cp{39,310,311,312}-linux_x86_64.whl`
- Each wheel is platform + Python-version specific
- File size: ~100MB each (includes torch, transformers)

### Testing Wheels (Local)

**File:** `test_wheels.sh`

What it tests:
1. Extracts Python version from wheel filename
2. Creates isolated venv for that version
3. Installs wheel
4. Runs import tests
5. Tests instantiation
6. Tests functionality
7. Cleans up venv

**Usage:**
```bash
bash test_wheels.sh dist/
```

**Sample output:**
```
Testing: hyperlane-0.1.0-cp310-cp310-linux_x86_64.whl
  Target Python: 3.10
  Running import tests...
    ✓ Main imports successful
    ✓ Sub-module imports successful
    ✓ DiscoveryManager instantiated
    ✓ KnapsackPartitioner instantiated
    ✓ Partitioning logic works
  ✓ All tests passed for Python 3.10
```

### Uploading Wheels

**File:** `upload_wheels.sh`

What it does:
1. Finds wheels in directory
2. Verifies wheels with twine
3. Checks credentials
4. Uploads to appropriate repository
5. Provides installation instructions

**Usage:**
```bash
# Upload to TestPyPI (staging)
bash upload_wheels.sh test dist/

# Upload to production PyPI
bash upload_wheels.sh prod dist/
```

---

## GitHub Actions CI/CD

**File:** `.github/workflows/build-wheel.yml`

Automated workflow that:

### 1. Build Job
- **Runs on:** Ubuntu 20.04
- **For each Python version:** 3.9, 3.10, 3.11, 3.12
- **Steps:**
  - Checkout code
  - Set up Python
  - Install build dependencies
  - Build wheel
  - Upload to artifacts
  - Verify wheel contents

### 2. Test Job
- **Depends on:** Build job
- **For each Python version:** 3.9, 3.10, 3.11, 3.12
- **Steps:**
  - Download wheel artifact
  - Install wheel
  - Test imports
  - Test functionality
  - Run unit tests

### 3. Verification Job
- **Checks:** All wheels together
- **Verifies:**
  - Wheel structure
  - Binary extensions
  - Python package structure
  - Compatibility tags

### 4. Publishing Job
- **Only runs on:** Git tags (e.g., `v0.1.0`)
- **Publishes to:**
  - TestPyPI (if tag contains `-rc` or `-beta`)
  - PyPI (for release tags)

### 5. Coverage Job
- **Runs:** Code coverage analysis
- **Reports:** To Codecov
- **Artifacts:** HTML coverage report

**Triggers:**
- `push` to main/develop
- `push` tags (v*)
- `pull_request` to main
- Manual `workflow_dispatch`

**How to use:**
```bash
# CI/CD runs automatically on:
git push origin main
git tag v0.1.0
git push origin v0.1.0
```

---

## Complete Workflow

### Step 1: Local Development & Testing

```bash
# Build wheels
cd /home/herb/Desktop/gpusharing
bash build_wheels_local.sh

# Test wheels
bash test_wheels.sh dist/
```

### Step 2: Upload to TestPyPI

```bash
# Set token
export TESTPYPI_API_TOKEN="pypi-YOUR_TOKEN"

# Upload
bash upload_wheels.sh test dist/

# Test installation
pip install -i https://test.pypi.org/simple/ hyperlane==0.1.0

# Verify
python -c "from hyperlane_client import *; print('✓')"
```

### Step 3: GitHub Push & CI/CD

```bash
# Push code (triggers GitHub Actions)
git add .
git commit -m "Release v0.1.0"
git push origin main

# Create release tag (triggers wheel build and test in CI)
git tag v0.1.0
git push origin v0.1.0

# GitHub Actions will:
# 1. Build wheels for all Python versions
# 2. Test wheels
# 3. Publish to PyPI automatically
```

### Step 4: Users Install

```bash
# Users on your team can now:
pip install hyperlane==0.1.0

# Or with GPU support:
pip install "hyperlane[gpu]==0.1.0"
```

---

## Troubleshooting

### Issue: "CMake not found"

**Solution:**
```bash
sudo apt install cmake
```

### Issue: "Python X.X not found"

**Solution:**
```bash
# Check available versions
ls /usr/bin/python*

# Install missing version
sudo apt install python3.X python3.X-venv

# Or skip that version in build_wheels_local.sh
```

### Issue: "Wheel is invalid"

**Solution:**
```bash
# Check with twine
twine check dist/*.whl

# Check zipfile contents
unzip -l dist/hyperlane-*.whl | head -20
```

### Issue: "TestPyPI token expired"

**Solution:**
1. Go to: https://test.pypi.org/account/#api-tokens
2. Revoke old token
3. Create new token
4. Update environment variable or ~/.pypirc

### Issue: "Module not found when testing wheel"

**Solution:**
```bash
# Make sure requirements are installed
pip install grpcio zeroconf torch transformers onnx onnxruntime

# Then test again
pip install dist/hyperlane-*.whl
python -c "from hyperlane_client import *"
```

---

## Performance Notes

### Wheel Sizes

Expected sizes (including dependencies):
- `hyperlane-0.1.0-cp39-cp39-linux_x86_64.whl`: ~100MB
  - Includes: torch, transformers, dependencies

### Build Times

Expected times on standard Ubuntu 20.04:
- Build one wheel: 5-10 minutes
- Build all 4 wheels: 20-40 minutes
- Test all wheels: 10-15 minutes
- Upload: < 1 minute

### Compatibility

The wheels are specific to:
- ✅ Linux x86_64 architecture
- ✅ Python version (separate wheel per version)
- ❌ Not compatible with Windows/macOS
- ❌ Not compatible with ARM architecture

For cross-platform support, would need additional workflows.

---

## Version Management

### Updating Version

Update in `hyperlane_client/setup.py`:
```python
setup(
    name="hyperlane",
    version="0.2.0",  # Update here
    ...
)
```

### Release Tagging

```bash
# Update version in setup.py
# Commit
git add hyperlane_client/setup.py
git commit -m "Bump version to 0.2.0"

# Tag
git tag v0.2.0

# Push (triggers CI/CD)
git push origin main
git push origin v0.2.0
```

---

## Security

### Token Safety

- ⚠️ Never commit tokens to git
- ✅ Use environment variables
- ✅ Use GitHub Secrets for CI/CD
- ✅ Regenerate tokens if leaked

### GitHub Secrets Setup

1. Go to: Repository Settings → Secrets and variables → Actions
2. Add new secrets:
   - `TESTPYPI_API_TOKEN`
   - `PYPI_API_TOKEN`

3. Reference in workflow: `${{ secrets.PYPI_API_TOKEN }}`

---

## References

- [PyPI Documentation](https://pypi.org/help/)
- [TestPyPI Documentation](https://test.pypi.org/)
- [setuptools Documentation](https://setuptools.pypa.io/)
- [wheel Documentation](https://wheel.readthedocs.io/)
- [twine Documentation](https://twine.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

## Summary

| Task | Command | Time |
|------|---------|------|
| Build wheels | `bash build_wheels_local.sh` | 20-40 min |
| Test wheels | `bash test_wheels.sh dist/` | 10-15 min |
| Upload to TestPyPI | `bash upload_wheels.sh test dist/` | <1 min |
| Upload to PyPI | `bash upload_wheels.sh prod dist/` | <1 min |
| CI/CD (automatic) | `git tag v0.1.0 && git push` | 30-50 min |

**Total time for release:** ~1-2 hours (mostly automated by CI/CD on subsequent releases)
