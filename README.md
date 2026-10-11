# Maestro

[![Built and tested on Ubuntu](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/QoroQuantum/maestro/actions/workflows/cmake-multi-platform.yml)

A unified interface for quantum circuit simulation, from your laptop to GPU workstations and MPI clusters. Build circuits in Python, import Qiskit circuits or OpenQASM 2.0, and run them through a common simulation API. Maestro also provides C/C++ interfaces and a native JSON request API.

## What Maestro Can Do

- **Simulate ideal and noisy circuits** — sample measurement outcomes, access states and probabilities, and run circuits with mid-circuit measurements, resets, and classical control.
- **Choose the right simulation method** — statevector, density matrix, matrix product state (MPS), matrix product operator (MPO), tensor network, stabilizer, extended stabilizer, Pauli propagation, and path integral. Select a backend explicitly or use Maestro's prediction-based backend selection in network execution.
- **Accelerate on GPUs and clusters** — use NVIDIA GPU backends, distribute a statevector across local GPUs, or scale across nodes with MPI.
- **Model hardware noise** — Pauli and coherent errors, T1/T2 relaxation, crosstalk, readout errors, and time-correlated noise, with Monte Carlo execution and exact channel evolution on supported density-matrix and MPO backends.
- **Study algorithms and error correction** — compute Pauli expectation values, inner products, and mirror fidelity; use Stim/Sinter integration for quantum error correction sampling.
- **Tune performance** — gate fusion, optimized shot sampling, MPS/MPO swap routing, configurable tensor truncation, and parallel execution.

## Quick Start

Requires Python 3.10 or newer. Pre-built wheels are available for Linux, macOS, and Windows.

```bash
pip install qoro-maestro
```

```python
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure([(0, 0), (1, 1)])

result = qc.execute(shots=1024)
print(result["counts"])  # Approximately {"00": 512, "11": 512}
```

The default is CPU statevector simulation. Reuse the same circuit with a different backend by passing a `maestro.SimulatorConfig` to `execute()`.

For source builds and the C++ library, see the [installation guide](https://qoroquantum.github.io/maestro/da/d03/md_markdown_2INSTALL.html).

## CPU, GPU, and Distributed Execution

| Execution | Supported methods |
|-----------|-------------------|
| CPU (QCSim; optional Qiskit Aer integration) | Statevector, density matrix, MPS, MPO, tensor network, stabilizer, extended stabilizer, Pauli propagation, path integral; availability varies by backend |
| NVIDIA GPU | Statevector, density matrix, MPS, MPO, tensor network, Pauli propagation, using CUDA and NVIDIA cuQuantum libraries |
| Local multi-GPU | Statevector distributed across GPUs within one node (`DistributedGpu`) |
| MPI + GPU | Statevector distributed across MPI ranks and GPUs on one or multiple nodes (`DistributedMpiGpu`) |

Maestro also supports p-block composite simulation and an optional QuEST statevector backend.

With the GPU plugin installed, select a single GPU using:

```python
import maestro

config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.Statevector,
    gpu_device=0,
)
result = qc.execute(shots=1024, config=config)
```

For larger statevectors, `DistributedGpu` spreads the state across local GPU memory. `DistributedMpiGpu` extends this to MPI clusters, allowing simulations beyond a single GPU's memory capacity. Both distributed backends currently support **statevector simulation only**.

GPU execution requires the corresponding acceleration plugin and compatible NVIDIA hardware and CUDA libraries. Distributed GPU plugins are Linux-only; MPI execution additionally requires CUDA-aware MPI. Run the same program on every rank, with matching simulation calls, using a launcher such as `mpirun`. See the [GPU, multi-GPU, and MPI guide](https://qoroquantum.github.io/maestro/dc/d01/py_hpc.html) for plugin setup, licensing where required, device selection, and a complete MPI example.

## Documentation

Browse the [Doxygen documentation](https://qoroquantum.github.io/maestro/).

- [Installation and source builds](https://qoroquantum.github.io/maestro/da/d03/md_markdown_2INSTALL.html)
- [Python quick start and configuration](https://qoroquantum.github.io/maestro/d2/d02/py_quickstart.html) · [Circuit building](https://qoroquantum.github.io/maestro/d2/d00/py_circuits.html) · [Simulation backends](https://qoroquantum.github.io/maestro/dc/d03/py_backends.html)
- [Noise modeling](https://qoroquantum.github.io/maestro/d5/d03/py_noise.html) · [Observables and algorithms](https://qoroquantum.github.io/maestro/d4/d01/py_algorithms.html) · [QEC with Stim/Sinter](https://qoroquantum.github.io/maestro/d6/d02/py_sinter.html)
- [GPU and MPI execution](https://qoroquantum.github.io/maestro/dc/d01/py_hpc.html)
- [C++ tutorial](https://qoroquantum.github.io/maestro/da/d03/md_markdown_2TUTORIAL.html) · [C++ API reference](https://qoroquantum.github.io/maestro/annotated.html)
- [Examples](https://github.com/QoroQuantum/maestro-demos) · [Release notes](https://qoroquantum.github.io/maestro/da/d02/md_markdown_2CHANGELOG.html)

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
