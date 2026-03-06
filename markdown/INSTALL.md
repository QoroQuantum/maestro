# Installation Guide

## Quick Start

### Python Package (Recommended)

Install the Python bindings via pip. This automatically handles most dependencies.

```bash
pip install qoro-maestro
```

**Prerequisites:**

- C++ Compiler (GCC/Clang)
- CMake 3.15+
- System libraries: `libfftw3-dev`, `libboost-all-dev` (Ubuntu/Debian)

### C++ Library

To build the C++ library and executable:

```bash
git clone https://github.com/QoroQuantum/maestro.git
cd maestro
./build.sh
```

The `build.sh` script automatically downloads and builds required dependencies (Eigen, QCSim, etc.) locally.

---

## Detailed Instructions

### System Requirements

Install these packages before building from source:

**Ubuntu/Debian:**

```bash
sudo apt-get install build-essential cmake libfftw3-dev libboost-all-dev libopenblas-dev git curl
```

**Fedora/RHEL:**

```bash
sudo dnf install gcc-c++ cmake fftw-devel boost-devel openblas-devel git curl
```

**macOS:**

```bash
brew install cmake fftw boost openblas
```

### Advanced Build Options

#### Enable Qiskit Aer Support

Qiskit Aer support is optional. To enable it:

1. **Install BLAS** (e.g., `libopenblas-dev`).
2. **Provide Aer Source** and install:

   ```bash
   export AER_INCLUDE_DIR=/path/to/qiskit-aer/src
   pip install qoro-maestro
   ```

   Or using `build.sh`:

   ```bash
   export AER_INCLUDE_DIR=/path/to/qiskit-aer/src
   ./build.sh
   ```

#### Custom Dependency Paths

If you have dependencies installed in non-standard locations, you can override the automatic fetching:

```bash
export EIGEN5_INCLUDE_DIR=/path/to/eigen
export BOOST_ROOT=/path/to/boost
export QCSIM_INCLUDE_DIR=/path/to/QCSim/QCSim
pip install qoro-maestro
```

## Troubleshooting

- **`maestro.so` not found**: Add the installation path to your library path:

  ```bash
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  ```

- **Build fails on dependencies**: Ensure you have the `-dev` or `-devel` packages installed (e.g., `libfftw3-dev`), as the build process requires header files.
