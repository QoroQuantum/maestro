# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.10] - 2026-04-16

### Added
- **Noise simulation API** — `NoiseModel` class for per-qubit Pauli noise channels (depolarizing, dephasing, bit-flip, custom)
- **`noisy_estimate`** — analytical Pauli channel damping with zero simulation overhead for fast ansatz noise screening
- **`noisy_estimate_montecarlo`** — gate-by-gate Monte Carlo noisy estimation for accurate depth-dependent noise simulation
- **`noisy_execute`** — Monte Carlo shot-based noisy circuit execution with batched noise realizations
- Noise model helpers: `set_depolarizing`, `set_dephasing`, `set_bit_flip`, `set_pauli_channel`, `set_all_depolarizing`
- Comprehensive noise test suite (25 tests covering NoiseModel, analytical damping, Monte Carlo estimation, and execution)
- Noise simulation section in TUTORIAL.md with usage examples and parameter reference

### Changed
- Default `lookahead_depth` changed from `20` to `-1` (auto-tuning) to prevent hangs on larger MPS circuits

### Fixed
- MPS swap optimization hang on larger circuits caused by hardcoded `lookahead_depth=20` bypassing the C++ auto-tuning heuristic

## [0.2.7] - 2026-03-27

### Added
- **Inner product** — compute ⟨ψ₁|ψ₂⟩ between two circuits via `ProjectOnZero` (`maestro.inner_product(circ_1, circ_2)` and `qc.inner_product(other)`)
- Python documentation for `inner_product` in `python.dox`
- Comprehensive test suite for inner product (9 tests covering identical, orthogonal, parametric, MPS, and phase-difference cases)

## [0.2.6] - 2026-03-25

### Added
- **`ProjectOnZero`** — efficient ⟨0|ψ⟩ projection for all simulator backends (QCSim MPS, GPU MPS, statevector, composite)
- `ExecuteOnHostProjectOnZero` network-level method with qubit remapping support
- `ExecuteOnHostAmplitudes` with proper qubit remapping for statevector extraction
- **Mirror fidelity** — `maestro.mirror_fidelity()` and `qc.mirror_fidelity()` with shot-based and exact statevector modes
- **Statevector access** — `maestro.get_statevector()` and `qc.get_statevector()` Python bindings
- **Probability access** — `maestro.get_probabilities()` Python binding
- MPS swap optimization with lookahead heuristics and initial qubit mapping
- Classically controlled gate support for MPS lookahead swap optimization
- Improved QCSim MPS sampling for partial qubit measurements
- ProjectOnZero C++ tests across all simulator backends

### Fixed
- Linux GPU library compilation (missing include)
- MPS optimization test adjusted from 20 to 12 qubits for stability

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

[Unreleased]: https://github.com/QoroQuantum/maestro/compare/v0.2.10...HEAD
[0.2.10]: https://github.com/QoroQuantum/maestro/compare/v0.2.7...v0.2.10
[0.2.7]: https://github.com/QoroQuantum/maestro/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/QoroQuantum/maestro/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/QoroQuantum/maestro/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/QoroQuantum/maestro/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/QoroQuantum/maestro/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/QoroQuantum/maestro/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/QoroQuantum/maestro/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/QoroQuantum/maestro/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/QoroQuantum/maestro/releases/tag/v0.1.0
