# Maestro C++ Tutorial & Developer Guide

This tutorial demonstrates how to use the Maestro C++ library to simulate quantum circuits.

## Introduction

Maestro provides a unified C/C++ interface to various quantum simulation backends. You can define circuits using OpenQASM 2.0 or circuit factory objects and execute them on the optimal simulator backend.

## Basic Usage

The standard C++ workflow involves:

1. Initializing the Maestro library handle.
2. Creating a simulator instance.
3. Defining a circuit (OpenQASM 2.0 string or IR).
4. Executing the circuit.
5. Processing the results.
6. Cleaning up allocated handles.

### Step-by-Step Example

Below is a complete, standalone C++ example:

```cpp
#include <iostream>
#include <string>
#include <vector>
#include "maestrolib/Interface.h"

void PrintResults(const char* jsonResult) {
    if (!jsonResult) {
        std::cout << "No results returned." << std::endl;
        return;
    }
    std::cout << "Simulation Results: " << jsonResult << std::endl;
}

int main() {
    // 1. Initialize Maestro singleton engine
    void* maestro = GetMaestroObject();
    if (!maestro) {
        std::cerr << "Failed to initialize Maestro." << std::endl;
        return 1;
    }

    // 2. Create a Simulator instance (2 qubits)
    unsigned long int simHandle = CreateSimpleSimulator(2);
    if (simHandle == 0) {
        std::cerr << "Failed to create simulator." << std::endl;
        return 1;
    }

    // 3. Define a Bell State circuit in OpenQASM 2.0
    const char* qasmCircuit =
        "OPENQASM 2.0;\n"
        "include \"qelib1.inc\";\n"
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "measure q -> c;\n";

    // 4. Configure Execution
    const char* config = "{\"shots\": 1024}";

    // 5. Execute the Circuit
    char* result = SimpleExecute(simHandle, qasmCircuit, config);

    // 6. Process Results
    PrintResults(result);

    // 7. Cleanup memory
    FreeResult(result);
    DestroySimpleSimulator(simHandle);

    return 0;
}
```

---

## Compiling Your C++ Application

To compile against the Maestro shared library:

```bash
g++ -std=c++17 -o maestro_example example.cpp \
    -I/path/to/maestro \
    -L/path/to/maestro/build/lib -lmaestro \
    -Wl,-rpath,/path/to/maestro/build/lib
```

---

## Advanced C++ Usage

### Manual Simulator Control

You can specify exact simulator backends and simulation methods:

```cpp
#include "Simulators/Factory.h"
#include "Simulators/State.h"

// Backend types: 0: Statevector, 1: MPS, 2: Stabilizer, 3: TensorNetwork
unsigned long int simHandle = CreateSimulator(0, 0);
```

### Expectation Values in C++

```cpp
const char* observables = "ZZ;XX";
char* estimate = SimpleEstimate(simHandle, qasmCircuit, observables);
std::cout << "Expectation Values: " << estimate << std::endl;
FreeResult(estimate);
```

### GPU and QuEST Backends (C++)

```cpp
#include "Simulators/Factory.h"

// Load optional acceleration libraries
bool questReady = Simulators::SimulatorsFactory::InitQuestLibrary();
bool gpuReady = Simulators::SimulatorsFactory::InitGpuLibrary();

// Switch simulator to GPU Statevector
RemoveAllOptimizationSimulatorsAndAdd(
    simHandle,
    static_cast<int>(Simulators::SimulatorType::kGpuSim),
    static_cast<int>(Simulators::SimulationType::kStatevector)
);
```

---

## Python SDK

For Python development, please see the comprehensive **Python User Guide** (`python_guide`):

- **Getting Started:** Installation, `SimulatorConfig`, and quick execution (`py_quickstart`).
- **Circuit Construction:** Programmatic `QuantumCircuit` builder and gate catalog (`py_circuits`).
- **Backend Selection:** Statevector, MPS, Stabilizer, and Tensor Networks (`py_backends`).
- **Observables & Metrics:** Expectation values, mirror fidelity, and time evolution (`py_algorithms`).
- **Noise Modeling:** Hardware relaxation, CPTP channels, and Monte Carlo simulation (`py_noise`).
- **HPC Accelerators:** GPU acceleration and QuEST MPI clusters (`py_hpc`).

### Selecting a GPU per simulator

On Linux with glibc, Maestro loads an isolated GPU plugin instance with
`dlmopen` for each requested CUDA-visible device ordinal. Instances on the same
device share that initialized library. Loading and initialization are lazy;
creating a CPU network no longer initializes the GPU plugin.

Configure a C++ GPU simulator before `Initialize()`:

```cpp
auto sim = Simulators::SimulatorsFactory::CreateSimulator(
    Simulators::SimulatorType::kGpuSim,
    Simulators::SimulationType::kStatevector);
sim->Configure("gpu_device", "0");
sim->AllocateQubits(2);
sim->Initialize();
```

For a simple network, call `network.Configure("gpu_device", "0")` before
`network.CreateSimulator(...)`. The setting is retained by network jobs,
recreated simulators and clones. The JSON configuration accepted by
`SimpleExecute` and `SimpleEstimate` also accepts `"gpu_device": 0`.

In Python:

```python
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.Statevector,
    gpu_device=0,
)
result = maestro.simple_execute(qasm, shots=100, config=config)
```

Use a separate configuration with `gpu_device=1` for a second visible GPU.
`None` (the default) uses device 0, or the default selected by
`maestro.select_gpu_device(device)`. Changing that default affects future
simulators only. Device ordinals follow CUDA visibility, including
`CUDA_VISIBLE_DEVICES`; they are not necessarily physical `nvidia-smi` indices.
An initialized simulator cannot change device until it is cleared and recreated.
Reapplying its current device is allowed. Explicit invalid or unavailable devices
raise an error instead of silently selecting another GPU.

`init_gpu()` remains an optional warm-up of the default device and returns true
when it is already initialized. `is_gpu_available()` probes and initializes the
default device; `get_gpu_device_count()` can load the plugin for discovery but
does not initialize simulator resources. GPU plugins must export `SetGpuDevice`,
`GetGpuDeviceCount`, and the CUDA runtime's `cudaGetDevice`/`cudaSetDevice` symbols
(the latter can come from a linked shared CUDA runtime).

Library instances remain cached for reuse. glibc has a finite namespace limit
(documented as 16 total, including the application's base namespace), and plugin
dependencies consume additional resources. Cross-context density-matrix/MPO
overlap is rejected; clone operations retain their source context. Legacy
synchronous simulator estimators inherit the network's device through a scoped
thread-local default; custom estimators creating simulators on additional threads
must propagate configuration to those threads explicitly.

GPU loader regression tests need no GPU:

```sh
cmake --build build --target gpu_registry_tests gpu_device_tests
ctest --test-dir build -R 'gpu_registry|gpu_device_configuration' --output-on-failure
```

To test two isolated real-plugin namespaces on a single GPU:

```sh
build/bin/gpu_registry_tests /path/to/libmaestro_gpu_simulators.so --real
```

This checks namespace compatibility, worker-thread use and cloning on GPU 0.
When two GPUs are visible, it also exercises separate instances on devices 0
and 1. Distinct physical-device placement still requires such a machine.

### GPU SVD algorithm selection

GPU MPS, MPO and tensor-network simulators accept these boolean configuration
keys through `Configure(key, value)`:

| Backend | Standard SVD | Jacobi SVD | GESVDP | Randomized SVD |
| --- | --- | --- | --- | --- |
| MPS | `matrix_product_state_use_gesvd` | `matrix_product_state_use_gesvdj` | `matrix_product_state_use_gesvdp` | `matrix_product_state_use_gesvdr` |
| MPO | `matrix_product_operator_use_gesvd` | `matrix_product_operator_use_gesvdj` | `matrix_product_operator_use_gesvdp` | `matrix_product_operator_use_gesvdr` |
| Tensor network | `tensor_network_use_gesvd` | `tensor_network_use_gesvdj` | `tensor_network_use_gesvdp` | `tensor_network_use_gesvdr` |

Values are `true`/`false` or `1`/`0`. Enabling an algorithm disables the other two
for that backend; disabling the active algorithm restores the default GESVD.
These settings can be applied before initialization or changed on a live
simulator. Configuration replay, network recreation and supported clone
operations preserve the latest selection.

```cpp
sim->Configure("matrix_product_state_use_gesvdp", "true");
// Later, switch the same MPS simulator to randomized SVD:
sim->Configure("matrix_product_state_use_gesvdr", "true");
```

Python exposes the same choices as `SimulatorConfig` boolean properties:

```python
config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.MatrixProductState,
    gpu_device=0,
)
config.mps_use_gesvdp = True
```

The corresponding GPU wrapper classes expose `SetGesvdP`, `GetGesvdP`,
`SetGesvdR` and `GetGesvdR`, alongside the existing Jacobi methods. MPS and MPO
also expose `GetLastSvdAlgo()`: `-1` before the first split, then `0` for GESVD,
`1` for GESVDJ, `2` for GESVDP or `3` for GESVDR. This reports the algorithm
actually used, rather than just the configured flag. The tensor-network plugin
does not currently export that diagnostic.

GESVDR has different truncation semantics: in MPS/MPO discarded-weight mode,
the plugin uses the bond-dimension cap and a numerical-rank floor instead of
the discarded-weight cutoff. The current tensor-network plugin uses GESVD as
its internal fallback when GESVDR is requested. Maestro preserves the plugin's
behavior; it does not reinterpret the cutoff or override that fallback.

P/R selection and the algorithm diagnostic require a plugin exporting the new
API. An older plugin remains loadable, but an explicit unsupported request
fails rather than calling a missing function.
