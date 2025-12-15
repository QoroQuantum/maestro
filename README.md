# Maestro: The Interface to Quantum Circuit Simulation

[![Built and tested on Ubuntu](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml)

Maestro is a unified interface and intelligent orchestration layer for quantum circuit simulation. It automates the complexity of selecting and configuring simulators, enabling researchers and developers to execute quantum circuits efficiently across CPUs, GPUs, and distributed HPC environments without manual tuning.

## Key Features

Maestro addresses the fragmentation of the current simulator ecosystem by providing a single entry point to various simulation methods.

- Unified Abstraction Layer: Write your circuit once (e.g., in Qiskit) and Maestro compiles it to the native format of the target backend.
- Intelligent Prediction Engine: Automatically analyzes circuit features (gate density, entanglement, locality) to predict and select the fastest simulation backend for your specific workload.
- High-Performance Optimizations: Transparently applies multi-threading, multi-processing, and optimized state sampling to increase throughput.
- GPU Acceleration: Integrated support for GPU-accelerated Statevector and custom Matrix Product State (MPS) execution.
- Distributed Quantum Computing (DQC): Supports p-block simulation (Composite mode) to simulate distributed quantum networks and break the memory ceiling of monolithic simulations.
- Backend-agnostic: Allows new simulators to be added easily

## Architecture Overview

The Maestro pipeline consists of:

1. Circuit ingestion (Qiskit/QASM)
2. Conversion to Maestro’s Intermediate Representation
3. Feature extraction (gate density, entanglement locality, structure)
4. Prediction engine (runtime estimation and backend routing)
5. Execution on one of the supported backends:
   - CPU statevector
   - GPU statevector
   - CPU/GPU MPS
   - Tensor networks
   - Clifford/stabilizer
   - Composite p-block distributed simulation

## Simulation Backends

Maestro integrates or wraps the following:

- CPU statevector (Qiskit Aer, QCSim, custom implementations)
- GPU statevector (NVIDIA cuStateVec)
- CPU MPS (multiple libraries)
- GPU MPS (custom CUDA implementation)
- Tensor network simulators
- Stabilizer/Clifford simulators
- p-block composite simulation for DQC

Each backend is accessed through a C++ adapter that maps Maestro’s IR to the simulator’s native API.

## Automatic Backend Selection

Maestro includes a prediction engine that:

- extracts structural features from the circuit
- uses a regression model trained on benchmark data
- estimates relative runtimes across all backends
- selects the backend expected to run fastest on the current hardware

The model normalizes performance features to reduce hardware dependence and can be recalibrated on installation.

## Documentation

- [Installation Guide](INSTALL.md): Detailed build and installation instructions.
- [Tutorial](TUTORIAL.md): Usage examples and API overview.

### API Documentation

To generate the API documentation using Doxygen:

```bash
# Ensure Doxygen is installed
cd build
cmake ..
make doc
```

The documentation will be generated in `docs/html/index.html`.

## Building Maestro

Quick start:

```bash
chmod +x build.sh
./build.sh
```

For detailed instructions, see [INSTALL.md](INSTALL.md).

## Citation

An Article detailing Maestro will be published shorty. This reference can be used for citation.

```latex
@article{bertomeu2025maestro,
  title={Maestro: Intelligent Execution for Quantum Circuit Simulation},
  author={Bertomeu, Oriol and Ghayas, Hamzah and Roman, Adrian and DiAdamo, Stephen},
  organization={Qoro Quantum},
  year={2025}
}
```

## License

This project is licensed under the GNU General Public License v3.0.

You may copy, distribute, and modify this software under the terms of the GPL-3.0 license.
A copy of the license text is available in the LICENSE file and at:

<https://www.gnu.org/licenses/gpl-3.0.en.html>
