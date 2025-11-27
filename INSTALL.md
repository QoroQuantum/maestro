# Installation Guide

This guide provides instructions on how to build and install the Maestro library.

## Prerequisites

Ensure you have the following tools installed on your system:

- **CMake** (version 3.10 or higher)
- **C++ Compiler** (supporting C++17, e.g., GCC, Clang, MSVC)
- **Git**
- **curl**
- **tar**
- **OpenMP** (optional but recommended for parallelization)

## Dependencies

Maestro relies on several external libraries. The provided `build.sh` script attempts to download and build these dependencies automatically if they are not found.

- **Eigen 5** (Linear algebra)
- **Boost** (version 1.89.0 or compatible, specifically `json`, `container`, `serialization`)
- **QCSim** (Quantum Circuit Simulator)
- **nlohmann/json** (JSON for Modern C++)
- **Qiskit Aer** (Optional, for Aer backend support)

## Building with `build.sh`

The easiest way to build Maestro is using the provided `build.sh` script. This script handles dependency management and invokes CMake.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/QoroQuantum/maestro.git
    cd maestro
    ```

2.  **Run the build script:**
    ```bash
    chmod +x build.sh
    ./build.sh
    ```

    This script will:
    - Create a `build` directory.
    - Download and extract Eigen and Boost if not present.
    - Build Boost libraries.
    - Clone QCSim.
    - Clone Qiskit Aer and nlohmann/json (unless `NO_QISKIT_AER` is set).
    - Set necessary environment variables.
    - Run `cmake` and `make`.

## Manual Build / Custom Paths

If you prefer to manage dependencies manually or use system-installed libraries, you can run CMake directly. You must set the following environment variables to point to your installations:

- `EIGEN5_INCLUDE_DIR`: Path to Eigen 5 include directory.
- `BOOST_ROOT`: Root directory of the Boost installation.
- `QCSIM_INCLUDE_DIR`: Path to QCSim include directory.
- `JSON_INCLUDE_DIR`: Path to nlohmann/json include directory (required if Aer is enabled).
- `AER_INCLUDE_DIR`: Path to Qiskit Aer source directory (required if Aer is enabled).

### Example Manual Build

```bash
mkdir build
cd build
cmake ..
make
```

### Build Options

- `-DNO_QISKIT_AER=ON`: Disable Qiskit Aer support. This removes the dependency on Qiskit Aer and nlohmann/json.
  ```bash
  cmake -DNO_QISKIT_AER=ON ..
  ```

- `-DCMAKE_BUILD_TYPE=Debug`: Build with debug information.
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Debug ..
  ```

## Installation

To install the library to your system (default `/usr/local/lib` on Unix):

```bash
cd build
sudo make install
```
