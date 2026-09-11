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

### Sampling-order regression tests

Sampling results follow the order of the requested qubit list: result bit `i`
corresponds to `qubits[i]`, including when sampling a subset. Both packed integer
and vector results use this convention.

```sh
cmake --build build --target sampling_order_tests -j2
ctest --test-dir build -R '^sampling_order$' --output-on-failure
```

The tests cover every three-qubit basis state, all full permutations, reordered
subsets, one and multiple shots, and identity/nonidentity internal mappings.
QCSim and composite tests always run. Aer tests are included when Aer is enabled,
including both MPS sampling algorithms. Configure `AER_INCLUDE_DIR` to use the Aer
fork checked out by `build.sh`. GPU coverage is a separate optional test:

```sh
ctest --test-dir build -R '^sampling_order_gpu$' --output-on-failure
```

It needs only one GPU and skips if no GPU/plugin is available. CUDA discovery
errors remain failures. These tests do not require Python bindings.

### Selecting a GPU per simulator

On Linux, Maestro lazily loads one process-wide GPU library singleton with
`dlopen`. All simulators share its handle and API table. `InitLib` runs once,
and `FreeLib` runs when the singleton is released at process shutdown. Creating
a CPU network does not initialize the GPU plugin.

Before native simulator initialization, Maestro calls the plugin's
`SetGpuDevice` with the configured CUDA-visible device ordinal. Selection and
initialization are serialized by Maestro. After initialization, the GPU plugin
owns device activation, state, cloning and cleanup for each native simulator;
Maestro does not switch CUDA devices around ordinary simulator calls.

Configure a C++ GPU simulator before `Initialize()`:

```cpp
auto sim = Simulators::SimulatorsFactory::CreateSimulator(
    Simulators::SimulatorType::kGpuSim,
    Simulators::SimulationType::kStatevector);
sim->Configure("gpu_device", "0");
sim->AllocateQubits(2);
sim->Initialize();
assert(sim->GetGpuDevice() == 0); // Native placement; -1 before initialization.
```

The low-level wrappers also accept a device when constructed, for example
`GpuLibStateVectorSim(library, 1)`. Pauli and stabilizer wrappers capture this
selection even though their native object is created later by `CreateSimulator`.
`GetGpuDevice()` queries the native object, including clones. Each instance stays
on one GPU; selecting a device does not distribute an instance across GPUs.

For a simple network, call `network.Configure("gpu_device", "0")` before
`network.CreateSimulator(...)`; `network.GetGpuDevice()` reports the native
placement after creation. `network.GetLastGpuDevice()` reports the device used
by the last execution, even if the network recreates its original simulator
afterward. The setting is retained by network jobs,
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
assert result["gpu_device"] == 0
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
does not initialize simulator resources. A return value of -1 reports a CUDA
discovery error. Execution and expectation results include `gpu_device` when
the chosen simulator runs on a GPU (also in the JSON API). GPU plugins must export `SetGpuDevice`
and `GetGpuDeviceCount`, plus the per-object GPU ID queries (such as
`GetStateVectorGpuId` and `MPSGetGpuId`), alongside
`InitLib`, `FreeLib` and their simulator APIs. Maestro no longer requires the
CUDA runtime's `cudaGetDevice`/`cudaSetDevice` symbols.

The plugin must allow `SetGpuDevice` after `InitLib` and retain the selected
device in each newly initialized native simulator. It must initialize resources
for each device as needed and handle any cross-device operation restrictions
itself. A simulator's device cannot be inferred from the shared library pointer.
Legacy synchronous simulator estimators inherit the network's device through a
scoped thread-local default; custom estimators creating simulators on additional
threads must propagate configuration to those threads explicitly.

GPU loader regression tests need no GPU:

```sh
cmake --build build --target gpu_registry_tests gpu_device_tests
ctest --test-dir build -R 'gpu_registry|gpu_device_configuration' --output-on-failure
```

To test singleton ownership and plugin-managed device handling:

```sh
build/bin/gpu_registry_tests /path/to/libmaestro_gpu_simulators.so --real
```

This checks singleton identity, worker-thread use, interleaved states and cloning.
When two GPUs are visible, it also exercises separate instances on devices 0
and 1. Distinct physical-device placement still requires such a machine.

To compare interleaved simulator instances on exactly two real GPUs against
independent QCSim CPU statevectors (Linux):

```sh
cmake -S . -B build -DCOMPILE_TESTS=ON
cmake --build build --target gpu_multi_device_tests -j2
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/path/to/gpu/plugin/directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} ctest --test-dir build -R '^gpu_multi_device$' --output-on-failure -V
```

Use the directory containing the real `libmaestro_gpu_simulators.so`, not
`build/mock_gpu`. Device ordinals 0 and 1 refer to the two GPUs exposed by
`CUDA_VISIBLE_DEVICES`; change that variable to select another physical pair.
No third GPU is used. Existing build dependency settings still apply.

The test covers statevector, density matrix, MPS, MPO, tensor network and Pauli
propagator backends, one pair at a time. It applies different three-qubit
entangling circuits to the two live instances, reverses update order between
rounds, and changes the process default to the other device. After each update,
both instances are compared with their CPU references using all 64 Pauli
expectation values (including phase-sensitive X/Y observables), with an absolute
tolerance of `1e-6` in double precision. It also checks that destroying one
instance leaves its peer usable.

Success prints `Two-GPU CPU-statevector comparisons passed` and exits 0.
With zero or one visible GPU (or no discoverable GPU plugin), CTest reports the
test as skipped, not failed. On two GPUs, initialization errors, missing backend
support or a numerical mismatch fail; mismatch messages identify the backend,
device, round and observable. Direct execution of `build/bin/gpu_multi_device_tests`
uses exit code 77 for a skip and 1 for a failure. Prefer CTest to handle skips
automatically.

### GPU SVD algorithm selection

GPU MPS, MPO and tensor-network simulators accept these boolean configuration
keys through `Configure(key, value)`:

| Backend | Standard SVD | Jacobi SVD | GESVDP | Randomized SVD |
| --- | --- | --- | --- | --- |
| MPS | `matrix_product_state_use_gesvd` | `matrix_product_state_use_gesvdj` | `matrix_product_state_use_gesvdp` | `matrix_product_state_use_gesvdr` |
| MPO | `matrix_product_operator_use_gesvd` | `matrix_product_operator_use_gesvdj` | `matrix_product_operator_use_gesvdp` | `matrix_product_operator_use_gesvdr` |
| Tensor network | `tensor_network_use_gesvd` | `tensor_network_use_gesvdj` | `tensor_network_use_gesvdp` | `tensor_network_use_gesvdr` |

The current GPU plugin defaults to GESVDP for MPS, MPO and tensor networks.
Maestro inherits the installed plugin's default unless an algorithm is selected.
Values are `true`/`false` or `1`/`0`. Enabling an algorithm disables the other
selectors for that backend. Clearing the active GESVDJ, GESVDP or GESVDR flag
selects plain GESVD; it does not restore the construction default. Enable
`*_use_gesvdp` explicitly to return to GESVDP. Setting `*_use_gesvd` to `true`
clears all three native flags; setting it to `false` leaves the live selection
unchanged.
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
