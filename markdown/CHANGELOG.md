# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cost estimation class with benchmarking and logging for ML-based predictions
- GPLv3 plugin linking exception for proprietary plugin support
- QuEST simulator initialization in maestro lib/exe

### Changed
- Documentation restructured and updated
- Improvements to cost class for sampling circuits with mid-circuit measurements

## [0.2.5] - 2026-03-18

### Added
- **Windows wheel support** — pre-built wheels for Windows AMD64 (Python 3.10, 3.11, 3.12) published to PyPI
- vcpkg-based dependency management for Windows CI builds
- Dry-run Windows CI workflow

### Fixed
- Eigen type mismatch between Windows and Linux (`Eigen::Index` → `long long int`)
- OpenMP configuration on Windows
- Compile warnings in MPS simulator and QCSim

## [0.2.4] - 2026-03-12

### Changed
- Split Python version builds into separate CI matrix elements for faster Windows builds
- Code styling and minor CI workflow improvements

## [0.2.3] - 2026-03-07

### Added
- Exposed setting the initial qubits map from the Python API
- MPS swap cost optimization with improved heuristics

### Fixed
- Linux wheel repair (`auditwheel`) patching for `libmaestro` linkage
- rpath configuration (`$ORIGIN`) for reliable library resolution in wheels

## [0.2.2] - 2026-03-01

### Fixed
- Bundled excluded `libmaestro` shared library with Python wheel
- macOS build issues

## [0.2.1] - 2026-02-25

### Fixed
- PyPI package metadata issue
- Version bump for initial stable release

### Added
- **Python bindings** via nanobind with `QuantumCircuit` model and GPU support
- **PyPI distribution** with `scikit-build-core` and `cibuildwheel` for Linux & macOS
- **Pauli propagator simulator** — CPU and GPU implementations with non-Clifford gate decomposition
- **Extended stabilizer simulator** with support for > 64 qubits
- **QuEST simulator** integration with factory function and tests
- **GPU stabilizer simulator** exposed through the library API
- **Rydberg atom array** simulation examples (adiabatic preparation, phase diagrams, spatial correlations)
- MPS parameters passthrough in the `SimpleExecute` flow
- Expectation value estimation via Python (`SimpleEstimate`)
- Maestro executable support for new simulators (Pauli propagator, extended stabilizer, QuEST)
- Double-precision toggle from Python bindings
- ccache support for faster rebuilds
- Doxygen documentation GitHub Actions workflow

### Changed
- Refactored GitHub Actions CI workflow for better dependency caching
- Build system configured for PyPI distribution (rpath, install targets)
- Improved GPU precision handling with placeholder params

### Fixed
- macOS OpenMP/install build issues
- Linux GPU compilation errors
- Circuit distribution crash when no distributor is present in disconnected networks
- Build warnings suppressed for third-party dependencies

## [0.1.0] - 2026-01-18

### Added
- **Core simulation engine** with multi-backend support (QCSim statevector, MPS tensor network, Qiskit Aer, Clifford)
- **Composite simulator** for automatic circuit distribution across backends
- **GPU tensor network simulator** with statevector and stabilizer support
- Maestro shared library (`libmaestro`) with `SimpleExecute` JSON/QASM API
- Maestro standalone executable with command-line interface
- QASM 2.0 parser (string → circuit conversion) with standard gate support
- Expectation value estimation framework
- Network-based job scheduling and execution
- OpenMP parallelization with SIMD (AVX2/FMA) acceleration
- Boost serialization support
- Doxygen API documentation with GitHub Pages deployment
- GitHub Actions CI (Ubuntu build + test)
- Pre-commit hooks with clang-format code formatting
- `CITATION.cff`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `INSTALL.md`

[Unreleased]: https://github.com/QoroQuantum/maestro/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/QoroQuantum/maestro/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/QoroQuantum/maestro/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/QoroQuantum/maestro/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/QoroQuantum/maestro/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/QoroQuantum/maestro/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/QoroQuantum/maestro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/QoroQuantum/maestro/releases/tag/v0.1.0
