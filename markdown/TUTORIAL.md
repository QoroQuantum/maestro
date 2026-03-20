# Maestro Tutorial

This tutorial demonstrates how to use the Maestro library to simulate quantum circuits in C++.

## Introduction

Maestro provides a unified C interface to various quantum simulation backends. You can define circuits using QASM or a JSON format and execute them on the most appropriate simulator.

## Basic Usage

The core workflow involves:

1. Initializing the Maestro library.
2. Creating a simulator instance.
3. Defining a circuit (QASM string).
4. Executing the circuit.
5. Parsing the results.
6. Cleaning up.

### Step-by-Step Example

Below is a complete C++ example. You can also find this in `examples/basic_simulation.cpp`.

```cpp
#include <iostream>
#include <string>
#include <vector>
#include "maestrolib/Interface.h"

// Helper to parse the JSON result (using a simple string search for demonstration)
// In a real app, use a JSON library like nlohmann/json or boost::json
void PrintResults(const char* jsonResult) {
    if (!jsonResult) {
        std::cout << "No results returned." << std::endl;
        return;
    }
    std::cout << "Simulation Results: " << jsonResult << std::endl;
}

int main() {
    // 1. Initialize Maestro
    // Get the singleton instance of the Maestro engine
    void* maestro = GetMaestroObject();
    if (!maestro) {
        std::cerr << "Failed to initialize Maestro." << std::endl;
        return 1;
    }

    // 2. Create a Simulator
    // Create a simple simulator for a small number of qubits.
    // This returns a handle (ID) to the simulator.
    unsigned long int simHandle = CreateSimpleSimulator(2);
    if (simHandle == 0) {
        std::cerr << "Failed to create simulator." << std::endl;
        return 1;
    }

    std::cout << "Simulator created with handle: " << simHandle << std::endl;

    // 3. Define a Circuit
    // We'll use a simple Bell State circuit in OpenQASM 2.0 format.
    const char* qasmCircuit =
        "OPENQASM 2.0;\n"
        "include \"qelib1.inc\";\n"
        "qreg q[2];\n"
        "creg c[2];\n"
        "h q[0];\n"
        "cx q[0], q[1];\n"
        "measure q -> c;\n";

    // 4. Configure Execution
    // Configuration is passed as a JSON string.
    // Here we request 1024 shots.
    const char* config = "{\"shots\": 1024}";

    // 5. Execute the Circuit
    // SimpleExecute takes the simulator handle, circuit string, and config.
    // It returns a JSON string with the results.
    char* result = SimpleExecute(simHandle, qasmCircuit, config);

    // 6. Process Results
    PrintResults(result);

    // 7. Cleanup
    // Free the result string memory
    FreeResult(result);

    // Destroy the simulator instance
    DestroySimpleSimulator(simHandle);

    return 0;
}
```

## Compiling the Example

To compile this example, you need to link against the `maestro` library.

Assuming you have built Maestro in the `build` directory:

```bash
g++ -std=c++17 -o maestro_example example.cpp \
    -I. \
    -L./build/lib -lmaestro \
    -Wl,-rpath,./build/lib
```

*Note: You may need to adjust include paths (`-I`) depending on where `maestrolib/Interface.h` is located relative to your source file.*

## Advanced Usage

### Manual Simulator Control

Instead of `SimpleExecute`, you can manually control the simulator state. See `examples/advanced_simulation.cpp` for a complete runnable example.

```cpp
// Create a specific simulator type (e.g., Statevector)
// See Simulators::SimulatorType enum for values (mapped to int)
// 0: Statevector, 1: MPS, etc. (Check source for exact mapping)
unsigned long int simHandle = CreateSimulator(0, 0);
void* sim = GetSimulator(simHandle);

// Apply gates directly
ApplyH(sim, 0);           // H on qubit 0
ApplyCX(sim, 0, 1);       // CX 0 -> 1

// Measure
unsigned long long int outcomes = MeasureNoCollapse(sim);
std::cout << "Measurement outcome: " << outcomes << std::endl;

// Clean up
DestroySimulator(simHandle);
```

### Configuration Options

The `jsonConfig` string in `SimpleExecute` supports various keys:

- `shots`: Number of execution shots (integer).
- `matrix_product_state_max_bond_dimension`: Max bond dimension for MPS (string/int).
- `matrix_product_state_truncation_threshold`: Truncation threshold for MPS (string/double).
- `mps_sample_measure_algorithm`: Algorithm for MPS sampling (string).

```json
{
  "shots": 1000,
  "matrix_product_state_max_bond_dimension": "100"
}
```
### Expectation Values

Maestro allows calculating the expectation values of observables without needing to perform manual sampling. This is particularly useful for Variational Quantum Algorithms.

```cpp
const char* observables = "ZZ;XX;YY";
char* result = SimpleEstimate(simHandle, qasmCircuit, observables, config);
// Result contains "expectation_values" array
```

### GPU and QuEST Backends (C++)

Maestro supports **GPU** and **QuEST** as dynamically-loaded simulation backends. On Linux, `GetMaestroObject()` automatically attempts to load the GPU library. You can also initialise these backends explicitly via the `SimulatorsFactory`.

> **Note:** The GPU backend is **not included** in the open-source version of Maestro. Contact [Qoro Quantum](https://qoroquantum.de) for access.

#### Initialisation

```cpp
#include "Simulators/Factory.h"

// GPU (Linux only) — automatically called by GetMaestroObject()
#ifdef __linux__
bool gpuReady = Simulators::SimulatorsFactory::InitGpuLibrary();
#endif

// QuEST (all platforms)
bool questReady = Simulators::SimulatorsFactory::InitQuestLibrary();
```

#### Using GPU via the C Interface

The `SimulatorType` and `SimulationType` enums are mapped to `int` values in the C interface. Use `RemoveAllOptimizationSimulatorsAndAdd` to switch the backend of an existing `SimpleSimulator` handle:

```cpp
#include "maestrolib/Interface.h"
#include "Simulators/State.h"  // for enum definitions

void* maestro = GetMaestroObject();
unsigned long int simHandle = CreateSimpleSimulator(2);

// Switch to GPU + Statevector
RemoveAllOptimizationSimulatorsAndAdd(
    simHandle,
    static_cast<int>(Simulators::SimulatorType::kGpuSim),
    static_cast<int>(Simulators::SimulationType::kStatevector)
);

const char* qasm =
    "OPENQASM 2.0;\n"
    "include \"qelib1.inc\";\n"
    "qreg q[2];\n"
    "creg c[2];\n"
    "h q[0];\n"
    "cx q[0], q[1];\n"
    "measure q -> c;\n";

char* result = SimpleExecute(simHandle, qasm, "{\"shots\": 1024}");
PrintResults(result);

FreeResult(result);
DestroySimpleSimulator(simHandle);
```

#### Using QuEST via the C Interface

```cpp
#include "maestrolib/Interface.h"
#include "Simulators/Factory.h"
#include "Simulators/State.h"

void* maestro = GetMaestroObject();

// Initialise QuEST (required before first use)
Simulators::SimulatorsFactory::InitQuestLibrary();

unsigned long int simHandle = CreateSimpleSimulator(2);

// Switch to QuEST + Statevector (QuEST only supports Statevector)
RemoveAllOptimizationSimulatorsAndAdd(
    simHandle,
    static_cast<int>(Simulators::SimulatorType::kQuestSim),
    static_cast<int>(Simulators::SimulationType::kStatevector)
);

const char* qasm =
    "OPENQASM 2.0;\n"
    "include \"qelib1.inc\";\n"
    "qreg q[2];\n"
    "creg c[2];\n"
    "h q[0];\n"
    "cx q[0], q[1];\n"
    "measure q -> c;\n";

char* result = SimpleExecute(simHandle, qasm, "{\"shots\": 1024}");
PrintResults(result);

FreeResult(result);
DestroySimpleSimulator(simHandle);
```

#### Supported Simulation Types

| SimulationType | GPU | QuEST |
|----------------|-----|-------|
| `kStatevector` | ✅ | ✅ |
| `kMatrixProductState` | ✅ | ❌ |
| `kTensorNetwork` | ✅ | ❌ |
| `kPauliPropagator` | ✅ | ❌ |
| `kStabilizer` | ❌ | ❌ |
| `kExtendedStabilizer` | ❌ | ❌ |

## Python

Maestro provides Python bindings for ease of use, allowing you to integrate its high-performance simulation capabilities into your Python-based quantum workflows.

### Installation

Install directly from PyPI (pre-built wheels for Linux, macOS, and Windows):

```bash
pip install qoro-maestro
```

Or build from source from the root of the Maestro repository:

```bash
pip install .
```

This will compile the C++ core and install the `maestro` Python package.

### Convenience API

The easiest way to use Maestro in Python is through the `simple_execute` and `simple_estimate` functions.

```python
import maestro

qasm_circuit = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

# 1. Execute and get counts
# You can specify simulator_type and simulation_type if needed
result = maestro.simple_execute(qasm_circuit, shots=1024)
print(f"Simulator: {result['simulator']}")  # int (SimulatorType enum value)
print(f"Method: {result['method']}")          # int (SimulationType enum value)
print(f"Counts: {result['counts']}")

# 2. Estimate expectation values
observables = "ZZ;XX;YY"
estimate = maestro.simple_estimate(qasm_circuit, observables)
print(f"Expectation Values: {estimate['expectation_values']}")
```

### QuantumCircuit Model

The `QuantumCircuit` class provides a Pythonic, object-oriented API for building
and executing quantum circuits directly in Python — no QASM strings required.
This is the **recommended** approach for most Python workflows.

#### Creating a Circuit

```python
import maestro

# Access the QuantumCircuit class from the circuits submodule
QuantumCircuit = maestro.circuits.QuantumCircuit

qc = QuantumCircuit()
```

#### Adding Gates

The `QuantumCircuit` supports a comprehensive gate set. Qubits are referenced
by integer index (0-based) and are automatically allocated as needed.

**Single-Qubit Gates (Non-Parametric)**

| Method       | Gate             | Description                  |
|--------------|------------------|------------------------------|
| `qc.x(q)`    | Pauli-X          | Bit flip                     |
| `qc.y(q)`    | Pauli-Y          | Bit + phase flip             |
| `qc.z(q)`    | Pauli-Z          | Phase flip                   |
| `qc.h(q)`    | Hadamard         | Creates superposition        |
| `qc.s(q)`    | S Gate           | π/2 phase                    |
| `qc.sdg(q)`  | S† Gate          | −π/2 phase                   |
| `qc.t(q)`    | T Gate           | π/4 phase                    |
| `qc.tdg(q)`  | T† Gate          | −π/4 phase                   |
| `qc.sx(q)`   | √X Gate          | Square root of X             |
| `qc.sxdg(q)` | √X† Gate        | Adjoint of √X                |
| `qc.k(q)`    | K Gate           | K gate                       |

**Single-Qubit Gates (Parametric)**

| Method                       | Gate    | Parameters          |
|------------------------------|---------|---------------------|
| `qc.p(q, λ)`                | Phase   | `λ` (lambda)        |
| `qc.rx(q, θ)`               | Rx      | `θ` (theta)         |
| `qc.ry(q, θ)`               | Ry      | `θ` (theta)         |
| `qc.rz(q, θ)`               | Rz      | `θ` (theta)         |
| `qc.u(q, θ, φ, λ)`         | U       | `θ`, `φ`, `λ`       |

**Two-Qubit Gates**

| Method              | Gate    | Description                     |
|---------------------|---------|----------------------------------|
| `qc.cx(c, t)`      | CNOT    | Controlled-X (control, target)   |
| `qc.cy(c, t)`      | CY      | Controlled-Y                     |
| `qc.cz(c, t)`      | CZ      | Controlled-Z                     |
| `qc.ch(c, t)`      | CH      | Controlled-Hadamard              |
| `qc.csx(c, t)`     | CSX     | Controlled-√X                    |
| `qc.csxdg(c, t)`   | CSX†    | Controlled-√X†                   |
| `qc.swap(a, b)`    | SWAP    | Swaps two qubits                 |

**Controlled Parametric Gates**

| Method                          | Gate | Parameters                    |
|---------------------------------|------|-------------------------------|
| `qc.cp(c, t, λ)`               | CP   | `λ` (lambda)                  |
| `qc.crx(c, t, θ)`              | CRx  | `θ` (theta)                   |
| `qc.cry(c, t, θ)`              | CRy  | `θ` (theta)                   |
| `qc.crz(c, t, θ)`              | CRz  | `θ` (theta)                   |
| `qc.cu(c, t, θ, φ, λ, γ)`     | CU   | `θ`, `φ`, `λ`, `γ`            |

**Three-Qubit Gates**

| Method                    | Gate     | Description                              |
|---------------------------|----------|------------------------------------------|
| `qc.ccx(c1, c2, t)`      | Toffoli  | Controlled-Controlled-X (CCX)            |
| `qc.cswap(c, a, b)`      | Fredkin  | Controlled-SWAP                          |

#### Measurements

Add measurements by providing a list of `(qubit_index, classical_bit_index)` pairs:

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)

# Measure qubit 0 → classical bit 0, qubit 1 → classical bit 1
qc.measure([(0, 0), (1, 1)])
```

You can also remap the classical bit assignments. For example, to store the
result of qubit 0 in classical bit 1 and vice versa:

```python
qc.measure([(0, 1), (1, 0)])
```

#### Executing a Circuit

The `execute` method runs the circuit for a given number of shots and returns
a dictionary with measurement counts:

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure([(0, 0), (1, 1)])

result = qc.execute(shots=1024)

print(result["counts"])        # e.g. {"00": 512, "11": 512}
print(result["time_taken"])    # Execution time in seconds
print(result["simulator"])     # Simulator type used (int)
print(result["method"])        # Simulation method used (int)
```

**Execution Parameters:**

| Parameter                    | Type   | Default                              | Description                            |
|------------------------------|--------|--------------------------------------|----------------------------------------|
| `simulator_type`             | enum   | `SimulatorType.QCSim`                | Simulator backend                      |
| `simulation_type`            | enum   | `SimulationType.Statevector`         | Simulation method                      |
| `shots`                      | int    | `1024`                               | Number of measurement shots            |
| `max_bond_dimension`         | int    | `2`                                  | Max bond dimension (MPS only)          |
| `singular_value_threshold`   | float  | `1e-8`                               | Truncation threshold (MPS only)        |

**Example with MPS backend:**

```python
result = qc.execute(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
    singular_value_threshold=1e-6,
    shots=2048
)
```

#### Estimating Expectation Values

The `estimate` method computes expectation values of Pauli observables without
sampling. This is ideal for variational algorithms and Hamiltonian simulation.

Observables can be specified as either a semicolon-separated string or a list of
strings. Each observable is a Pauli string (e.g., `"ZZ"`, `"XII"`, `"IYZ"`).

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
# Bell state: (|00⟩ + |11⟩) / √2

# Using a list of Pauli strings
result = qc.estimate(observables=["XX", "ZZ", "IZ"])
print(result["expectation_values"])  # [1.0, 1.0, 0.0]

# Using a semicolon-separated string
result = qc.estimate(observables="XX;ZZ;IZ")
print(result["expectation_values"])  # [1.0, 1.0, 0.0]
```

**Estimation Parameters:**

| Parameter                    | Type          | Default                              | Description                            |
|------------------------------|---------------|--------------------------------------|----------------------------------------|
| `observables`                | str or list   | *(required)*                         | Pauli observables to measure           |
| `simulator_type`             | enum          | `SimulatorType.QCSim`                | Simulator backend                      |
| `simulation_type`            | enum          | `SimulationType.Statevector`         | Simulation method                      |
| `max_bond_dimension`         | int           | `2`                                  | Max bond dimension (MPS only)          |
| `singular_value_threshold`   | float         | `1e-8`                               | Truncation threshold (MPS only)        |

> **Note:** The `estimate` method does not require measurements to be added to
> the circuit. The number of qubits is automatically inferred from the circuit
> operations and observable lengths.

#### Full Example: Bell State

```python
import maestro

QuantumCircuit = maestro.circuits.QuantumCircuit

# Build a Bell state circuit
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)

# --- Sampling ---
qc.measure([(0, 0), (1, 1)])
result = qc.execute(shots=1000)
print("Counts:", result["counts"])
# Expected: ~{"00": 500, "11": 500}

# --- Expectation Values ---
qc_est = QuantumCircuit()
qc_est.h(0)
qc_est.cx(0, 1)

est = qc_est.estimate(observables=["XX", "ZZ", "ZI", "IZ"])
print("⟨XX⟩ =", est["expectation_values"][0])  # 1.0
print("⟨ZZ⟩ =", est["expectation_values"][1])  # 1.0
print("⟨ZI⟩ =", est["expectation_values"][2])  # 0.0
print("⟨IZ⟩ =", est["expectation_values"][3])  # 0.0
```

### Manual Control API

For more granular control over the simulation lifecycle, you can use the `Maestro` class directly.

```python
from maestro import Maestro, SimulatorType, SimulationType

# Initialize Maestro
m = Maestro()

# Create a simulator handle
# Defaults to QCSim and MatrixProductState
sim_handle = m.create_simulator(SimulatorType.QCSim, SimulationType.Statevector)

# Get the raw simulator object
sim = m.get_simulator(sim_handle)

# Cleanup
m.destroy_simulator(sim_handle)
```

> **Note:** The `Maestro` class provides handle-based lifecycle management.
> For most Python workflows, the `QuantumCircuit` API or the `simple_execute` /
> `simple_estimate` convenience functions are the recommended approach.

### Probability Access

Get the full probability distribution after executing a circuit (without sampling):

```python
import maestro

QuantumCircuit = maestro.circuits.QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)

probs = maestro.get_probabilities(qc)
print(probs)  # [0.5, 0.0, 0.0, 0.5] for a Bell state
```

### Mirror Fidelity

Mirror fidelity measures how well a circuit "undoes itself". It works by
running the circuit forward, then appending the adjoint (inverse) of every gate
in reverse order, and computing the probability of returning to the all-zeros
state P(|0…0⟩). A value of 1.0 means the circuit perfectly undoes itself.

This is useful for benchmarking simulator accuracy and for characterising
noise when using approximate simulation methods (e.g. MPS with low bond
dimension).

By default, `mirror_fidelity` uses shot-based sampling (1024 shots).
For exact results on small circuits, pass `full_amplitude=True` to use
the full statevector instead:

#### Module-Level Function

```python
import maestro
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.rx(0, 3.14159 / 4)

fidelity = maestro.mirror_fidelity(qc)
print(f"Mirror fidelity: {fidelity:.4f}")  # ~1.0
```

#### Circuit Method

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.s(0)

fidelity = qc.mirror_fidelity()
print(f"Mirror fidelity: {fidelity:.4f}")  # ~1.0
```

#### With a Specific Backend

```python
fidelity = qc.mirror_fidelity(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=16,
)
```

#### With More Shots

```python
fidelity = qc.mirror_fidelity(shots=10000)  # tighter estimate
```

#### Exact Mode (Small Circuits Only)

For exact results without statistical noise, use `full_amplitude=True`.
This extracts the full statevector, so it only works for small qubit counts:

```python
fidelity = qc.mirror_fidelity(full_amplitude=True)
print(f"Mirror fidelity: {fidelity:.10f}")  # 1.0000000000

#### Mirror Fidelity Parameters

| Parameter                    | Type   | Default                              | Description                            |
|------------------------------|--------|--------------------------------------|----------------------------------------|
| `simulator_type`             | enum   | `SimulatorType.QCSim`                | Simulator backend                      |
| `simulation_type`            | enum   | `SimulationType.Statevector`         | Simulation method                      |
| `shots`                      | int    | `1024`                               | Number of measurement shots            |
| `max_bond_dimension`         | int    | `None`                               | Max bond dimension (MPS only)          |
| `singular_value_threshold`   | float  | `None`                               | Truncation threshold (MPS only)        |
| `use_double_precision`       | bool   | `False`                              | Use 64-bit precision (GPU only)        |
| `full_amplitude`             | bool   | `False`                              | Use exact statevector (small circuits) |

> **Note:** Measurements in the original circuit are automatically skipped
> when building the mirror circuit — only unitary gate operations are mirrored.

### Examples

You can find several complete examples in the `examples/` directory:

- `examples/python_example_1.py`: Basic simulation and sampling.
- `examples/python_example_2.py`: Advanced simulation with manual gate application.
- `examples/python_example_3.py`: Working with expectation values and observables.

---

## QuEST and GPU Execution (Dynamic Backends)

Maestro supports **QuEST** and **GPU** simulation backends as **optional, dynamically-loaded libraries**. Unlike the built-in CPU backends (QCSim, MPS, etc.) which are always available, these backends are packaged as separate shared libraries (`libmaestroquest.so`, `libmaestro_gpu_simulators.so`) and loaded at runtime only when explicitly requested.

This design means:

- The core `maestro` package works without QuEST or GPU support installed.
- You must **initialize** a dynamic backend before using it.
- If the shared library is not found on the system, initialization will gracefully return `False` rather than crashing.

### Initializing Dynamic Backends

Before using QuEST or GPU simulators, you **must** call the corresponding initialization function. This loads the shared library into memory and registers the backend with Maestro's simulator factory.

```python
import maestro

# --- QuEST Backend ---
quest_ok = maestro.init_quest()
print(f"QuEST initialized: {quest_ok}")  # True if library found and loaded

# --- GPU Backend ---
gpu_ok = maestro.init_gpu()
print(f"GPU initialized: {gpu_ok}")  # True if library found and loaded
```

> **Note:** Initialization only needs to be done **once** per process. Subsequent calls are safe but redundant.

### Checking Availability

You can check whether a dynamic backend has been successfully loaded at any point:

```python
import maestro

print(f"QuEST available: {maestro.is_quest_available()}")
print(f"GPU available:   {maestro.is_gpu_available()}")
```

These return `False` if `init_quest()` or `init_gpu()` has not been called, or if the initialization failed (e.g., shared library not found).

### Running Circuits with QuEST

QuEST provides an alternative statevector simulation engine. Once initialized, you can select it via `SimulatorType.QuestSim`. QuEST also natively supports **MPI-distributed statevector** simulation, enabling execution across multiple nodes for larger qubit counts.

> **Limitation:** QuEST only supports the **Statevector** simulation type. Attempting to use it with MPS, Stabilizer, or other simulation types will raise an error.

#### Using the QuantumCircuit API

```python
import maestro

# Initialize QuEST (required before first use)
if not maestro.init_quest():
    raise RuntimeError("QuEST library not available")

QuantumCircuit = maestro.circuits.QuantumCircuit

# Build a circuit
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure([(0, 0), (1, 1)])

# Execute on QuEST
result = qc.execute(
    simulator_type=maestro.SimulatorType.QuestSim,
    simulation_type=maestro.SimulationType.Statevector,
    shots=1024
)

print(f"Counts: {result['counts']}")
print(f"Time:   {result['time_taken']:.4f}s")
```

#### Using the Convenience API

```python
import maestro

maestro.init_quest()

qasm_circuit = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

# Execute
result = maestro.simple_execute(
    qasm_circuit,
    simulator_type=maestro.SimulatorType.QuestSim,
    shots=1024
)
print(f"Counts: {result['counts']}")

# Estimate expectation values
estimate = maestro.simple_estimate(
    qasm_circuit,
    observables="ZZ;XX",
    simulator_type=maestro.SimulatorType.QuestSim
)
print(f"Expectation values: {estimate['expectation_values']}")
```

### Running Circuits on GPU

The GPU backend provides CUDA-accelerated simulation. Once initialized, select it via `SimulatorType.Gpu`.

> **Note:** The GPU backend is **not included** in the open-source version of Maestro. Contact [Qoro Quantum](https://qoroquantum.de) for access.

The GPU backend supports multiple simulation types:

| Simulation Type     | GPU Support |
|---------------------|-------------|
| Statevector         | ✅          |
| MatrixProductState  | ✅          |
| TensorNetwork       | ✅          |
| PauliPropagator     | ✅          |
| Stabilizer          | ❌          |
| ExtendedStabilizer  | ❌          |

```python
import maestro

# Initialize GPU (required before first use)
if not maestro.init_gpu():
    raise RuntimeError("GPU library not available — is CUDA installed?")

QuantumCircuit = maestro.circuits.QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure([(0, 0), (1, 1)])

# Execute on GPU with statevector
result = qc.execute(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.Statevector,
    shots=2048
)
print(f"GPU Counts: {result['counts']}")

# Execute on GPU with MPS
result_mps = qc.execute(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
    shots=2048
)
print(f"GPU MPS Counts: {result_mps['counts']}")
```

### Defensive Initialization Pattern

For scripts that should work regardless of backend availability, use a guard pattern:

```python
import maestro

QuantumCircuit = maestro.circuits.QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure([(0, 0), (1, 1)])

# Try QuEST, fall back to default CPU
if maestro.init_quest():
    result = qc.execute(simulator_type=maestro.SimulatorType.QuestSim)
    print("Ran on QuEST")
elif maestro.init_gpu():
    result = qc.execute(simulator_type=maestro.SimulatorType.Gpu)
    print("Ran on GPU")
else:
    result = qc.execute()  # Default: QCSim Statevector
    print("Ran on CPU (QCSim)")

print(f"Counts: {result['counts']}")
```

### API Reference

| Function                    | Returns | Description                                                  |
|-----------------------------|---------|--------------------------------------------------------------|
| `maestro.init_quest()`      | `bool`  | Load the QuEST shared library. Returns `True` on success.    |
| `maestro.is_quest_available()` | `bool` | Check if QuEST has been loaded and is ready to use.         |
| `maestro.init_gpu()`        | `bool`  | Load the GPU shared library. Returns `True` on success.      |
| `maestro.is_gpu_available()`| `bool`  | Check if the GPU backend has been loaded and is ready.        |

| Simulator Type               | Enum Value                        | Dynamic? | Backend Library                   |
|-------------------------------|-----------------------------------|----------|-----------------------------------|
| QCSim (default CPU)           | `SimulatorType.QCSim`             | No       | Built-in                          |
| Composite QCSim               | `SimulatorType.CompositeQCSim`    | No       | Built-in                          |
| QuEST                         | `SimulatorType.QuestSim`          | **Yes**  | `libmaestroquest.so` / `.dylib`   |
| GPU                           | `SimulatorType.Gpu`               | **Yes**  | `libmaestro_gpu_simulators.so`   |

### Building with Dynamic Backend Support

The QuEST and GPU shared libraries are **not** part of the default `pip install qoro-maestro` package. To use them:

- **QuEST:** Build the QuEST integration library and ensure `libmaestroquest.so` (or `.dylib` on macOS) is on your library path.
- **GPU:** The GPU backend is not included in the open-source release. Contact [Qoro Quantum](https://qoroquantum.de) for access to the GPU libraries.

See `INSTALL.md` for detailed build instructions.
