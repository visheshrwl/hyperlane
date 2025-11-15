from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from pathlib import Path
import sys
import os
import platform
import subprocess

# Read requirements
req_file = Path(__file__).parent.parent / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        raw_reqs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        # Use CPU version of onnxruntime for wider compatibility
        requirements = [r if 'onnxruntime' not in r or 'gpu' not in r else 'onnxruntime>=1.14.0' for r in raw_reqs]
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

# Remove GPU version if present
requirements = [r for r in requirements if 'onnxruntime-gpu' not in r]
if not any('onnxruntime' in r for r in requirements):
    requirements.append('onnxruntime>=1.14.0')


class CMakeBuildExt(build_ext):
    """Custom build extension using CMake for C++ extension"""
    
    def build_extension(self, ext):
        """Build CMake extension"""
        if ext.name != 'hyperlane_client.hyperlane_tensor_socket':
            return super().build_extension(ext)
        
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_build_dir = os.path.join(self.build_temp, 'cmake_build')
        
        # Create build directory
        os.makedirs(cmake_build_dir, exist_ok=True)
        os.makedirs(ext_dir, exist_ok=True)
        
        # CMake configure
        cmake_args = [
            f'-DCMAKE_BUILD_TYPE=Release',
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}',
            f'-DCMAKE_INSTALL_PREFIX={ext_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
        ]
        
        # Platform-specific optimizations
        if platform.system() == 'Linux':
            cmake_args.append('-DCMAKE_CXX_FLAGS=-O3 -march=native')
        elif platform.system() == 'Darwin':
            cmake_args.append('-DCMAKE_CXX_FLAGS=-O3 -march=native')
        
        print(f"CMake configure arguments: {cmake_args}")
        
        try:
            subprocess.check_call(
                ['cmake', os.path.dirname(__file__) + '/pybind', *cmake_args],
                cwd=cmake_build_dir
            )
            subprocess.check_call(
                ['cmake', '--build', '.', '--config', 'Release'],
                cwd=cmake_build_dir
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n⚠️  WARNING: Could not build C++ extension: {e}")
            print("The Python-only version will be used instead.")
            print("To use GPU features, manually build with: bash build.sh\n")


def build_ext_modules():
    """Create extension modules for building"""
    try:
        import pybind11
        pybind11_include = pybind11.get_cmake_dir()
    except ImportError:
        print("pybind11 not found. C++ extension will be skipped.")
        return []
    
    extensions = [
        Extension(
            'hyperlane_client.hyperlane_tensor_socket',
            sources=['pybind/tensor_socket.cpp'],
            include_dirs=[str(Path(__file__).parent / 'pybind')],
            language='c++',
            extra_compile_args=['-O3', '-march=native'] if platform.system() != 'Windows' else [],
        )
    ]
    return extensions


# Only try to build extensions if CMake is available
ext_modules = []
try:
    import subprocess
    subprocess.check_call(['cmake', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ext_modules = build_ext_modules()
    print(f"Building {len(ext_modules)} C++ extension(s)")
except (subprocess.CalledProcessError, FileNotFoundError):
    print("CMake not found. Skipping C++ extension build.")
    ext_modules = []


setup(
    name="hyperlane",
    version="0.1.0",
    description="Distributed GPU inference engine for heterogeneous prosumer hardware",
    author="Hyperlane Team",
    author_email="contact@hyperlane.dev",
    url="https://github.com/yourusername/hyperlane",
    long_description=open(Path(__file__).parent.parent / "README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    
    packages=find_packages(),
    
    package_data={
        "hyperlane_client": ["*.so", "*.pyd", "*.dylib"],  # Include compiled extensions
    },
    
    python_requires=">=3.9",
    
    install_requires=requirements,
    
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
    
    ext_modules=ext_modules,
    cmdclass={'build_ext': CMakeBuildExt},
    zip_safe=False,  # Don't zip - allows .so files to be found
)
