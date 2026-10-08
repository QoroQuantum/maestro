# Maestro

[![Built and tested on Ubuntu](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml)

A unified interface for quantum circuit simulation. Write your circuit once — Maestro picks the best backend and runs it on CPU, GPU, or distributed HPC.

## Features

- **One API, many backends** — compile from Qiskit / QASM to any supported simulator
- **Automatic backend selection** — a prediction engine analyzes your circuit and routes it to the fastest backend
- **CPU simulation** — statevector, MPS, MPO, Pauli propagation, path integral, Clifford/stabilizer, extended stabilizer, density matrix
- **GPU acceleration** — statevector (cuStateVec), MPS (custom CUDA and cuTensorNet), MPO (cuTensorNet, cuBlas and custrom CUDA) tensor network (cuTensorNet), Pauli propagation (cuPauliProp), density matrix (cuDensityMatrix)
- **Distributed simulation** — p-block composite mode for distributed quantum computing
- **Expectation values** — direct observable estimation (Pauli strings) for VQA workflows
- **Performance optimizations** — automatic multi-threading, multi-processing, and optimized sampling

## Quick Start

```bash
pip install qoro-maestro  # Linux, macOS & Windows
```

Or build from source:

```bash
chmod +x build.sh
./build.sh
```

For detailed build instructions, see [INSTALL.md](https://github.com/QoroQuantum/maestro/blob/main/markdown/INSTALL.md).

Composer integration is optional and defaults to off. To build Maestro with
Composer's network and execution estimators, point CMake at the Composer source
root (the directory containing `composer/`):

```bash
cmake -S . -B build -DMAESTRO_ENABLE_COMPOSER=ON -DCOMPOSER_INCLUDE_DIR=../composer
cmake --build build --config Release
```

An enabled build also searches the sibling `../composer` and `../../composer`
locations. An explicit `COMPOSER_INCLUDE_DIR` supports any checkout layout.
Composer's own CMake build and Visual Studio solution enable this integration
automatically.

## How It Works

```
Qiskit / QASM circuit
        ↓
Maestro Intermediate Representation
        ↓
Feature extraction  →  Prediction engine  →  Backend selection
        ↓
Execution (CPU / GPU / Distributed)
```

1. **Ingest** — accepts circuits from Qiskit or QASM
2. **Convert** — compiles to Maestro's intermediate representation
3. **Analyze** — extracts features (gate density, entanglement, locality)
4. **Route** — prediction engine estimates runtimes and selects the fastest backend
5. **Execute** — runs on the chosen backend with automatic performance tuning

## Backends

| Type | Backends |
|------|----------|
| CPU | Statevector (Aer, QCSim), MPS, Pauli propagation, Clifford/stabilizer |
| GPU | Statevector (cuStateVec), MPS (CUDA), tensor network, Pauli propagation |
| Distributed | p-block composite simulation |

Each backend is accessed through a C++ adapter that maps Maestro's IR to the simulator's native API.

QCSim has separate implementations for statevector, density matrix, MPS, MPO,
stabilizer, extended stabilizer, tensor network, Pauli propagation, and path
integral simulation in `Simulators/QCSim`. The GPU implementations in
`Simulators/Gpu` cover statevector, density matrix, MPS, MPO, tensor network,
and Pauli propagation. Each owns only its native backend; MPS and MPO share
the routing and bond-dimension bookkeeping in a tensor-chain base.

Select the method with `SimulatorsFactory::CreateSimulator` or
`CreateSimulatorUnique`. These factories add the gate-fusion adapter where
appropriate; internal composite children use the immediate statevector backend.
The method is fixed at construction. Replaying the same `method` configuration
is supported; select a different method by creating another simulator.

## Documentation

| Resource | Link |
|----------|------|
| Installation | [INSTALL.md](https://github.com/QoroQuantum/maestro/blob/main/markdown/INSTALL.md) |
| Tutorial & API | [TUTORIAL.md](https://github.com/QoroQuantum/maestro/blob/main/markdown/TUTORIAL.md) |
| Python examples | [maestro-examples](https://github.com/QoroQuantum/maestro-examples) |

To generate API docs with Doxygen:

```bash
cd build
cmake ..
make doc
# Opens at docs/html/index.html
```

## Citation

```latex
@article{bertomeu2025maestro,
  title={Maestro: Intelligent Execution for Quantum Circuit Simulation},
  author={Bertomeu, Oriol and Ghayas, Hamzah and Roman, Adrian and DiAdamo, Stephen},
  organization={Qoro Quantum},
  year={2025}
}
```

## License

GPL-3.0 — see [LICENSE](./LICENSE) or <https://www.gnu.org/licenses/gpl-3.0.en.html>.
