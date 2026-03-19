# Installation Guide

## Quick Start

### Python Package (Recommended)

Install the Python bindings via pip. Pre-built wheels are available — **no compiler or build tools required**.

```bash
pip install qoro-maestro
```

**Supported platforms (pre-built wheels):**

| Platform | Architecture | Python |
|----------|-------------|--------|
| Linux | x86_64 | 3.10, 3.11, 3.12 |
| macOS | arm64 (Apple Silicon) | 3.10, 3.11, 3.12 |
| Windows | AMD64 | 3.10, 3.11, 3.12 |

That's it — you're ready to `import maestro` and run quantum simulations. See [TUTORIAL.md](https://github.com/QoroQuantum/maestro/blob/main/markdown/TUTORIAL.md) for usage examples.

### C++ Library

To build the C++ library and executable:

```bash
git clone https://github.com/QoroQuantum/maestro.git
cd maestro
./build.sh
```

The `build.sh` script automatically downloads and builds required dependencies (Eigen, QCSim, etc.) locally.

---

## Building from Source

If you need to build from source (e.g., for a platform without pre-built wheels, or to enable optional features like Qiskit Aer), follow these instructions.

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

**Windows:**

Windows builds use [vcpkg](https://vcpkg.io/) for dependency management. Required packages:

```
boost-json boost-program-options boost-container boost-serialization openblas
```

### Building the Python Package from Source

From the root of the Maestro repository:

```bash
pip install .
```

This will compile the C++ core and install the `maestro` Python package.

### Advanced Build Options

#### Enable Qiskit Aer Support

Qiskit Aer support is optional. To enable it:

1. **Install BLAS** (e.g., `libopenblas-dev`).
2. **Provide Aer Source** and install:

   ```bash
   export AER_INCLUDE_DIR=/path/to/qiskit-aer/src
   pip install .
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
pip install .
```

## Troubleshooting

- **`maestro.so` not found**: Add the installation path to your library path:

  ```bash
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  ```

- **Build fails on dependencies**: Ensure you have the `-dev` or `-devel` packages installed (e.g., `libfftw3-dev`), as the build process requires header files.

- **Windows build issues**: Make sure vcpkg packages are installed for `x64-windows` and the vcpkg toolchain file is passed to CMake.
