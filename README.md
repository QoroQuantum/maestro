# Maestro

[![Build and test](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml)
[![PyPI](https://img.shields.io/pypi/v/qoro-maestro)](https://pypi.org/project/qoro-maestro/)
[![Python](https://img.shields.io/pypi/pyversions/qoro-maestro)](https://pypi.org/project/qoro-maestro/)
[![Docs](https://img.shields.io/badge/docs-qoroquantum.github.io-blue)](https://qoroquantum.github.io/maestro/)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-green)](./LICENSE)

**One API for simulating quantum circuits with any method: statevector, MPS, stabilizer, Pauli propagation, density matrix and more. Runs on a laptop CPU, NVIDIA GPUs, or an MPI cluster.**

Write a circuit once, then change one config field to switch from a 30-qubit statevector to a 100-qubit matrix product state, or from CPU to GPU. Already using Qiskit or PennyLane? Change one import and keep your code.

```bash
pip install qoro-maestro
```

## Quick start

```python
import maestro
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

result = qc.execute(shots=1000)
print(result["counts"])  # about {"00": 500, "11": 500}
```

The default is a CPU statevector. To run the same circuit with another method, pass a `SimulatorConfig`:

```python
config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.MatrixProductState)
result = qc.execute(shots=1000, config=config)
```

Requires Python 3.10+. Pre-built wheels are available for Linux, macOS and Windows.

## Already using Qiskit or PennyLane?

**Qiskit:** [`qiskit-maestro`](https://github.com/QoroQuantum/qiskit-maestro) provides a backend, Sampler and Estimator you can use in place of Aer:

```bash
pip install qiskit-maestro
```

```python
from qiskit_maestro import MaestroBackend, MaestroSampler, MaestroEstimator

backend = MaestroBackend()      # was AerSimulator()
sampler = MaestroSampler()      # was StatevectorSampler()
estimator = MaestroEstimator()  # was StatevectorEstimator()
```

On 120–200 qubit MPS workloads, `MaestroEstimator` is up to [7× faster than Aer's `EstimatorV2`](https://qoroquantum.github.io/maestro/benchmarks.html).

**PennyLane:** [`pennylane-maestro`](https://github.com/QoroQuantum/pennylane-maestro) adds a Maestro device, including adjoint gradients for variational algorithms:

```bash
pip install pennylane-maestro
```

```python
import pennylane as qml

dev = qml.device("maestro.qubit", wires=2)  # was "default.qubit"
```

Maestro can also load OpenQASM 2.0 programs directly.

## Features

- **Many simulation methods behind one API:** statevector, density matrix, matrix product state (MPS), matrix product operator (MPO), tensor network, stabilizer, extended stabilizer, Pauli propagation and path integral. See [Choosing a simulation method](https://qoroquantum.github.io/maestro/user-guide/choosing-a-method.html) for which method suits which circuits.
- **GPU and cluster scaling:** NVIDIA GPU backends built on cuQuantum, a statevector split across the GPUs in one node, or across nodes with MPI.
- **Realistic noise:** Pauli and coherent errors, T1/T2 relaxation, crosstalk, readout errors and time-correlated noise. Noise runs as Monte Carlo trajectories, or as exact channels on density-matrix and MPO backends.
- **Beyond sampling:** expectation values of Pauli observables, statevectors and probabilities, mid-circuit measurement with classical control, and [Stim/Sinter integration](https://qoroquantum.github.io/maestro/user-guide/qec-sinter.html) for error-correction studies.
- **C/C++ and JSON interfaces:** link `libmaestro` directly, or send batch jobs through the [native JSON request API](docs/native-request-api.md). Neither needs Python.

## Running on GPUs

The pip package runs on the CPU. GPU backends come from a separate plugin, which you can access in three ways:

- **Qoro cloud:** run on our GPUs with free starter credits.
- **Free academic licence:** run the plugin on your own hardware.
- **Enterprise site licence:** run it on your organization's infrastructure.

See [Get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html) for details. Once the plugin is installed, select the GPU in the config:

```python
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.Statevector,
)
result = qc.execute(shots=1000, config=config)
```

| Backend | Methods |
|---|---|
| `QCSim` (CPU, default) | All methods listed above |
| `Gpu` | Statevector, density matrix, MPS, MPO, tensor network, Pauli propagation |
| `DistributedGpu` | Statevector split across the GPUs in one node (Linux) |
| `DistributedMpiGpu` | Statevector split across MPI ranks and nodes (Linux, CUDA-aware MPI) |

Optional Qiskit Aer and QuEST backends are also available. See [Scaling: multi-GPU and MPI](https://qoroquantum.github.io/maestro/user-guide/scaling.html) for a complete MPI example.

## Related projects

| Repository | What it is | Install |
|---|---|---|
| [qiskit-maestro](https://github.com/QoroQuantum/qiskit-maestro) | Qiskit provider: `MaestroBackend`, `MaestroSampler` and `MaestroEstimator` | `pip install qiskit-maestro` |
| [pennylane-maestro](https://github.com/QoroQuantum/pennylane-maestro) | PennyLane device `maestro.qubit`, with adjoint gradients | `pip install pennylane-maestro` |
| [maestro-demos](https://github.com/QoroQuantum/maestro-demos) | Example simulations written with Maestro | — |

## Documentation

Full documentation: **[qoroquantum.github.io/maestro](https://qoroquantum.github.io/maestro/)**

| | |
|---|---|
| **Get started** | [First circuit](https://qoroquantum.github.io/maestro/getting-started/first-circuit.html) · [Coming from Qiskit Aer](https://qoroquantum.github.io/maestro/getting-started/coming-from-qiskit.html) · [PennyLane](https://qoroquantum.github.io/maestro/getting-started/pennylane.html) · [GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html) |
| **Guides** | [Building circuits](https://qoroquantum.github.io/maestro/user-guide/building-circuits.html) · [Choosing a method](https://qoroquantum.github.io/maestro/user-guide/choosing-a-method.html) · [Sampling and shots](https://qoroquantum.github.io/maestro/user-guide/sampling-and-shots.html) · [Expectation values](https://qoroquantum.github.io/maestro/user-guide/expectation-values.html) · [Noise and QEC](https://qoroquantum.github.io/maestro/user-guide/noise-and-qec.html) · [Scaling](https://qoroquantum.github.io/maestro/user-guide/scaling.html) |
| **Reference** | [Python API](https://qoroquantum.github.io/maestro/api/index.html) · [C++ API](https://qoroquantum.github.io/maestro/cpp/annotated.html) · [Native JSON API](docs/native-request-api.md) · [Benchmarks](https://qoroquantum.github.io/maestro/benchmarks.html) |
| **C++ and source builds** | [Install from source](markdown/INSTALL.md) · [C++ tutorial](markdown/TUTORIAL.md) |
| **More** | [Examples](https://github.com/QoroQuantum/maestro-demos) · [Changelog](markdown/CHANGELOG.md) · [Contributing](markdown/CONTRIBUTING.md) |

## Citation

If you use Maestro in your research, please cite:

```bibtex
@article{bertomeu2025maestro,
  title={Maestro: Intelligent Execution for Quantum Circuit Simulation},
  author={Bertomeu, Oriol and Ghayas, Hamzah and Roman, Adrian and DiAdamo, Stephen},
  organization={Qoro Quantum},
  year={2025}
}
```

## License

Maestro is licensed under [GPL-3.0](./LICENSE). The GPU plugin is licensed separately; see [Get GPU access](https://qoroquantum.github.io/maestro/getting-started/gpu-access.html).
