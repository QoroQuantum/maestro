# Maestro

[![Built and tested on Ubuntu](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml)

A unified interface for quantum circuit simulation. Write your circuit once — Maestro picks the best backend and runs it on CPU, GPU, or distributed HPC.

## Features

- **One API, many backends** — compile from Qiskit / QASM to any supported simulator
- **Automatic backend selection** — a prediction engine analyzes your circuit and routes it to the fastest backend
- **CPU simulation** — statevector, MPS, Pauli propagation, Clifford/stabilizer
- **GPU acceleration** — statevector (cuStateVec), MPS (custom CUDA), tensor network, Pauli propagation
- **Distributed simulation** — p-block composite mode for distributed quantum computing
- **Expectation values** — direct observable estimation (Pauli strings) for VQA workflows
- **Performance optimizations** — automatic multi-threading, multi-processing, and optimized sampling

## Quick Start

```bash
pip install qoro-maestro  # Linux & macOS
```

Or build from source:

```bash
chmod +x build.sh
./build.sh
```

For detailed build instructions, see [INSTALL.md](https://github.com/QoroQuantum/maestro/blob/main/markdown/INSTALL.md).

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

## Documentation

| Resource | Link |
|----------|------|
| Installation | [INSTALL.md](https://github.com/QoroQuantum/maestro/blob/main/markdown/INSTALL.md) |
| Tutorial & API | [TUTORIAL.md](https://github.com/QoroQuantum/maestro/blob/main/markdown/TUTORIAL.md) |
| Python examples | [examples/](https://github.com/QoroQuantum/maestro/tree/main/examples) |

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
