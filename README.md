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

For source builds and the C++ library, see the [installation guide](markdown/INSTALL.md).

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

GPU execution requires the corresponding acceleration plugin and compatible NVIDIA hardware and CUDA libraries. Distributed GPU plugins are Linux-only; MPI execution additionally requires CUDA-aware MPI. Run the same program on every rank, with matching simulation calls, using a launcher such as `mpirun`. See the [GPU, multi-GPU, and MPI guide](docs/python/hpc.dox) for plugin setup, licensing where required, device selection, and a complete MPI example.

## Documentation

- [Installation and source builds](markdown/INSTALL.md)
- [Python quick start and configuration](docs/python/quickstart.dox) · [Circuit building](docs/python/circuits.dox) · [Simulation backends](docs/python/backends.dox)
- [Noise modeling](docs/python/noise.dox) · [Observables and algorithms](docs/python/algorithms.dox) · [QEC with Stim/Sinter](docs/python/sinter.dox)
- [GPU and MPI execution](docs/python/hpc.dox)
- [C++ tutorial](markdown/TUTORIAL.md) · [Native JSON request API](docs/native-request-api.md)
- [Examples](examples) · [Release notes](markdown/CHANGELOG.md)

<details>
<summary>Optional Composer integration</summary>

Composer's network and execution estimators are disabled by default. Enable them by pointing CMake at the Composer source root (the directory containing `composer/`):

```bash
cmake -S . -B build -DMAESTRO_ENABLE_COMPOSER=ON -DCOMPOSER_INCLUDE_DIR=../composer
cmake --build build --config Release
```

An enabled build also searches `../composer` and `../../composer`. Composer's own CMake build and Visual Studio solution enable this integration automatically.

</details>

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
