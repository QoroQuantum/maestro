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
| `kPathIntegral` | ❌ | ❌ |

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

#### SimulatorConfig

All execution and estimation methods accept a `SimulatorConfig` object
that bundles every simulator parameter into one reusable value. Create
a config once and pass it to every call:

```python
import maestro

config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)

result = qc.execute(config=config, shots=1024)
fidelity = qc.mirror_fidelity(config=config)
```

**SimulatorConfig Parameters:**

| Parameter                    | Type           | Default                       | Description                            |
|------------------------------|----------------|-------------------------------|----------------------------------------|
| `simulator_type`             | `SimulatorType`| `QCSim`                       | Simulator backend                      |
| `simulation_type`            | `SimulationType`| `Statevector`                | Simulation method                      |
| `max_bond_dimension`         | int or None    | `None`                        | Max bond dimension (MPS only)          |
| `singular_value_threshold`   | float or None  | `None`                        | Truncation threshold (MPS only)        |
| `use_double_precision`       | bool           | `False`                       | Use 64-bit precision (GPU only)        |
| `disable_optimized_swapping` | bool           | `False`                       | Disable MPS swap optimization          |
| `lookahead_depth`            | int            | `-1`                          | Swap optimization lookahead depth      |
| `mps_measure_no_collapse`    | bool           | `True`                        | Use probability-based MPS sampling     |

When no config is passed, a default `SimulatorConfig()` is used
(QCSim + Statevector with all defaults). Config fields can also be
modified after construction:

```python
config = maestro.SimulatorConfig()
config.simulation_type = maestro.SimulationType.MatrixProductState
config.max_bond_dimension = 128
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

| Parameter | Type             | Default              | Description                    |
|-----------|------------------|----------------------|--------------------------------|
| `config`  | `SimulatorConfig`| `SimulatorConfig()`  | Simulator configuration        |
| `shots`   | int              | `1024`               | Number of measurement shots    |

**Example with MPS backend:**

```python
mps_config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
    singular_value_threshold=1e-6,
)
result = qc.execute(config=mps_config, shots=2048)
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

| Parameter     | Type              | Default              | Description                      |
|---------------|-------------------|----------------------|----------------------------------|
| `observables` | str or list       | *(required)*         | Pauli observables to measure     |
| `config`      | `SimulatorConfig` | `SimulatorConfig()`  | Simulator configuration          |

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
mps_config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QCSim,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=16,
)
fidelity = qc.mirror_fidelity(config=mps_config)
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
```

#### Mirror Fidelity Parameters

| Parameter        | Type              | Default             | Description                            |
|------------------|-------------------|---------------------|----------------------------------------|
| `config`         | `SimulatorConfig` | `SimulatorConfig()` | Simulator configuration                |
| `shots`          | int               | `1024`              | Number of measurement shots            |
| `full_amplitude` | bool              | `False`             | Use exact statevector (small circuits) |

> **Note:** Measurements in the original circuit are automatically skipped
> when building the mirror circuit — only unitary gate operations are mirrored.

### Examples

You can find several complete examples in the `examples/` directory:

- `examples/python_example_1.py`: Basic simulation and sampling.
- `examples/python_example_2.py`: Advanced simulation with manual gate application.
- `examples/python_example_3.py`: Working with expectation values and observables.

### Noise Simulation

Maestro provides three approaches to noisy simulation, each targeting a
different trade-off between speed and accuracy:

| Function | Overhead | Accuracy | Best for |
|----------|----------|----------|----------|
| `noisy_estimate` | Zero | Per-qubit Pauli damping | Fast ansatz screening |
| `noisy_estimate_montecarlo` | N × noiseless | Gate-by-gate accurate | Training with realistic noise |
| `noisy_execute` | N × noiseless | Gate-by-gate + shot noise | Shot-based workflows |

#### Creating a Noise Model

The `NoiseModel` class defines per-qubit Pauli noise channels. Each qubit can
have independent X, Y, and Z error probabilities:

```python
import maestro

nm = maestro.NoiseModel()

# Depolarizing noise: equal probability of X, Y, Z errors
# p is the total error probability, split equally as p/3 for each Pauli
nm.set_depolarizing(qubit=0, p=0.01)

# Set the same depolarizing rate on all qubits at once
nm.set_all_depolarizing(num_qubits=5, p=0.005)

# Dephasing noise: only Z errors (T2 relaxation)
nm.set_dephasing(qubit=1, p=0.02)

# Bit-flip noise: only X errors (T1 relaxation)
nm.set_bit_flip(qubit=2, p=0.01)

# Custom Pauli channel: specify px, py, pz independently
nm.set_pauli_channel(qubit=3, px=0.005, py=0.002, pz=0.01)
```

#### Analytical Noisy Estimation (Zero Overhead)

`noisy_estimate` runs a single noiseless simulation and analytically damps
each expectation value based on the noise model. This is exact for a
single-layer Pauli channel and provides a fast first-order approximation:

```python
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)

nm = maestro.NoiseModel()
nm.set_all_depolarizing(2, 0.05)

result = maestro.noisy_estimate(qc, ['ZZ', 'XX', 'YY'], nm)

print(result['expectation_values'])        # Noise-attenuated values
print(result['ideal_expectation_values'])  # Noiseless reference
print(result['time_taken'])                # Same as noiseless
```

This approach is ideal for quickly screening which ansatze are noise-resilient.
It runs at exactly the same speed as noiseless estimation.

> **Note:** The analytical approach applies a uniform per-qubit damping factor
> to the observable. It does not capture depth-dependent noise accumulation —
> a 100-layer circuit and a 1-layer circuit with the same qubits get the same
> damping. For depth-accurate noise, use `noisy_estimate_montecarlo`.

#### Gate-by-gate Monte Carlo Estimation (Accurate)

`noisy_estimate_montecarlo` injects random Pauli errors after every gate,
runs noiseless estimation on each noisy circuit, and averages the results.
This captures how noise compounds through the circuit depth:

```python
result = maestro.noisy_estimate_montecarlo(
    qc, ['ZZ', 'XX'], nm,
    noise_realizations=200,  # More samples → less variance
    seed=42                  # For reproducibility
)

print(result['expectation_values'])        # Gate-by-gate accurate
print(result['ideal_expectation_values'])  # Noiseless reference
print(result['noise_realizations'])        # 200
```

This is the recommended approach for QML training with realistic noise.
The cost is `noise_realizations × noiseless_time`, which is typically much
faster than density-matrix simulation (e.g., Qiskit fake backends).

Works with any backend — statevector, MPS, or Pauli propagation:

```python
mps_config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)
result = maestro.noisy_estimate_montecarlo(
    qc, ['ZZ'], nm,
    noise_realizations=100,
    config=mps_config,
    seed=42
)
```

#### Monte Carlo Noisy Execution (Shot-based)

`noisy_execute` is the shot-based counterpart: it injects gate-by-gate noise
and returns measurement counts rather than expectation values:

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

nm = maestro.NoiseModel()
nm.set_all_depolarizing(2, 0.05)

result = maestro.noisy_execute(
    qc, nm,
    shots=1024,
    noise_realizations=64,  # Independent noise samples
    seed=42
)
print(result['counts'])  # Aggregated counts across all realizations
```

#### Noise Simulation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_realizations` | int | `100` (MC estimate) / `64` (execute) | Independent noise samples |
| `seed` | int | `None` (random) | RNG seed for reproducibility |
| `config` | `SimulatorConfig` | `SimulatorConfig()` | Simulator configuration |

> **Tip:** For QML training workflows where you need noise-aware gradients,
> `noisy_estimate_montecarlo` with 50–200 realizations gives a good
> speed-accuracy trade-off. Use `noisy_estimate` for rapid screening.

#### Coherent Noise Simulation

Coherent noise models systematic calibration errors — e.g. over/under-rotation
of gates — rather than stochastic Pauli errors. Instead of randomly inserting
X, Y, Z gates, coherent noise injects **deterministic rotation gates**
(Rx, Ry, Rz) after every gate in the circuit.

The key difference from Pauli noise:
- **Pauli noise** averages out over shots (each shot sees a different random error)
- **Coherent noise** is deterministic within a single noise realization — all
  shots from one circuit see the same systematic error. The ± sign of the
  rotation is sampled per-realization.

This is important for QEC research where coherent errors behave
fundamentally differently from stochastic ones.

##### Configuring Coherent Noise

The `NoiseModel` class supports coherent noise alongside Pauli noise.
Both can coexist on the same model:

```python
import maestro

nm = maestro.NoiseModel()

# From a depolarizing probability: ε = 2·arcsin(√p)
# This gives the same per-gate infidelity as DEPOLARIZE1(p)
nm.set_coherent_depolarizing(qubit=0, p=0.01)

# Apply to all qubits at once
nm.set_all_coherent_depolarizing(num_qubits=5, p=0.001)

# Convenience alias for set_all_coherent_depolarizing
nm.set_coherent_strength(num_qubits=5, p=0.001)

# Per-axis control: explicit rotation angles (radians)
nm.set_coherent_rotation(qubit=0, rx=0.01, ry=0.0, rz=0.05)

# Dephasing-like: Z-axis rotation only
nm.set_coherent_dephasing(qubit=1, p=0.02)

# Bit-flip-like: X-axis rotation only
nm.set_coherent_bit_flip(qubit=2, p=0.01)

# Check if coherent noise is configured
print(nm.has_coherent())  # True
```

##### Coherent Estimate (Expectation Values)

`coherent_estimate` is the coherent analogue of `noisy_estimate_montecarlo`.
It injects rotation noise, runs expectation value estimation, and averages
over multiple sign realizations:

```python
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)

nm = maestro.NoiseModel()
nm.set_all_coherent_depolarizing(2, 0.005)

result = maestro.coherent_estimate(
    qc, ['ZZ', 'XX', 'YY'], nm,
    noise_realizations=200,
    seed=42
)

print(result['expectation_values'])        # Coherent-noisy values
print(result['ideal_expectation_values'])  # Noiseless reference
print(result['noise_type'])                # 'coherent'
```

##### Coherent Execute (Shot-based)

`coherent_execute` is the shot-based counterpart. Each noise realization
gets a deterministic rotation pattern, and shots are distributed across
realizations:

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

nm = maestro.NoiseModel()
nm.set_coherent_strength(2, 0.001)

result = maestro.coherent_execute(
    qc, nm,
    shots=1024,
    noise_realizations=64,
    seed=42
)
print(result['counts'])           # Aggregated counts
print(result['noise_type'])       # 'coherent'
```

> **Note:** Coherent noise uses rotation gates, so it requires a simulation
> backend that supports continuous rotations — **MPS**, **Statevector**, or
> **TensorNetwork**. Stabilizer simulation will not work.

##### Coherent Noise API Reference

| Method | Description |
|--------|-------------|
| `nm.set_coherent_depolarizing(q, p)` | Rz angle from depolarizing probability |
| `nm.set_coherent_dephasing(q, p)` | Rz rotation from dephasing probability |
| `nm.set_coherent_bit_flip(q, p)` | Rx rotation from bit-flip probability |
| `nm.set_coherent_rotation(q, rx, ry, rz)` | Explicit per-axis angles (radians) |
| `nm.set_all_coherent_depolarizing(n, p)` | Uniform coherent noise on all qubits |
| `nm.set_all_coherent_dephasing(n, p)` | Uniform coherent dephasing on all qubits |
| `nm.set_coherent_strength(n, p)` | Alias for `set_all_coherent_depolarizing` |
| `nm.has_coherent()` | Check if any coherent noise is configured |
| `maestro.coherent_estimate(...)` | Expectation values with coherent noise |
| `maestro.coherent_execute(...)` | Shot-based execution with coherent noise |

#### T1 Amplitude Damping

T1 amplitude damping models energy relaxation — the spontaneous decay from |1⟩
to |0⟩ over time. Maestro implements this using the **quantum trajectory method**:
after each gate, with probability γ, the qubit is probabilistically reset to |0⟩.

```python
nm = maestro.NoiseModel()

# Set per-gate decay probability directly
nm.set_t1(qubit=0, gamma=0.001)

# Set uniform T1 on all qubits
nm.set_all_t1(num_qubits=5, gamma=0.0003)

# Compute gamma from physical time constants (most accurate)
# gamma = 1 - exp(-gate_time / T1)
nm.set_t1_from_time(qubit=0, gate_time_s=500e-9, t1_time_s=200e-6)

# Check if T1 is configured
print(nm.has_t1())  # True
```

> **Note:** T1 damping is asymmetric — it only decays |1⟩ → |0⟩, never the
> reverse. This is physically correct: thermal relaxation at millikelvin
> temperatures overwhelmingly favours the ground state.

#### ZZ Crosstalk

ZZ crosstalk models parasitic coupling between neighbouring qubits. When a gate
acts on qubit q1, the spectator qubit q2 accumulates an unwanted Rz rotation.
This is one of the dominant error sources on superconducting devices.

```python
nm = maestro.NoiseModel()

# Set symmetric ZZ coupling between qubits 0 and 1
# After a gate on q0, Rz(0.005) is applied on q1, and vice versa
nm.set_crosstalk(q1=0, q2=1, strength=0.005)

# Set nearest-neighbour crosstalk on a chain
for i in range(n_qubits - 1):
    nm.set_crosstalk(i, i + 1, 0.008)

print(nm.has_crosstalk())  # True
```

> **Tip:** Crosstalk strength values can be estimated from simultaneous
> randomised benchmarking (sim-RB) data on IBM devices.

#### Combined Noise Simulation

The `full_noise_execute` and `full_noise_estimate` functions apply **all configured
noise layers** in a single call. The noise is injected per gate in physical order:

1. **Coherent over-rotations** (systematic Rx/Ry/Rz)
2. **ZZ crosstalk** (Rz on spectator neighbours)
3. **T1 amplitude damping** (probabilistic reset to |0⟩)
4. **Pauli noise** (stochastic X/Y/Z errors)

This enables realistic device-level simulation with a single noise model:

```python
from maestro.circuits import QuantumCircuit

# Build a noise model matching your hardware
nm = maestro.NoiseModel()

# Layer 1: Coherent gate over-rotations (from GST or calibration data)
nm.set_all_coherent_depolarizing(n_qubits, 0.005)

# Layer 2: ZZ crosstalk between neighbours
for i in range(n_qubits - 1):
    nm.set_crosstalk(i, i + 1, 0.008)

# Layer 3: T1 amplitude damping (from backend properties)
for q in range(n_qubits):
    nm.set_t1_from_time(q, gate_time_s=500e-9, t1_time_s=200e-6)

# Layer 4: Pauli (incoherent) noise (from RB data)
nm.set_all_depolarizing(n_qubits, 0.005)

# Check if any noise is configured
print(nm.has_any())  # True
```

##### Combined Execute (Shot-based)

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

result = qc.full_noise_execute(
    nm,
    shots=1024,
    noise_realizations=64,
    seed=42
)
print(result['counts'])       # Aggregated counts
print(result['noise_type'])   # 'combined'
```

##### Combined Estimate (Expectation Values)

```python
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)

result = qc.full_noise_estimate(
    ['ZZ', 'XX'], nm,
    noise_realizations=100,
    seed=42
)
print(result['expectation_values'])        # Noisy values
print(result['ideal_expectation_values'])  # Noiseless reference
print(result['noise_type'])                # 'combined'
```

##### Combined Noise API Reference

| Method | Description |
|--------|-------------|
| `nm.set_t1(q, gamma)` | Per-gate T1 decay probability |
| `nm.set_all_t1(n, gamma)` | Uniform T1 on all qubits |
| `nm.set_t1_from_time(q, gate_time_s, t1_time_s)` | T1 from physical time constants |
| `nm.has_t1()` | Check if T1 is configured |
| `nm.set_crosstalk(q1, q2, strength)` | Symmetric ZZ coupling |
| `nm.has_crosstalk()` | Check if crosstalk is configured |
| `nm.has_any()` | Check if any noise type is configured |
| `maestro.full_noise_execute(...)` | Shot-based combined execution |
| `maestro.full_noise_estimate(...)` | Combined expectation values |

##### All Noise Simulation Functions

| Function | Noise Type | Overhead | Best for |
|----------|-----------|----------|----------|
| `noisy_estimate` | Pauli | Zero | Fast ansatz screening |
| `noisy_estimate_montecarlo` | Pauli | N × noiseless | Training with realistic noise |
| `noisy_execute` | Pauli | N × noiseless | Shot-based Pauli noise |
| `coherent_estimate` | Coherent | N × noiseless | Coherent error analysis |
| `coherent_execute` | Coherent | N × noiseless | Shot-based coherent noise |
| `full_noise_execute` | **All layers** | N × noiseless | Realistic device simulation |
| `full_noise_estimate` | **All layers** | N × noiseless | Hardware-accurate expectation values |

> **Tip:** All noise functions are also available as bound methods on `QuantumCircuit`:
> `qc.noisy_execute(nm)`, `qc.noisy_estimate(['ZZ'], nm)`,
> `qc.noisy_estimate_montecarlo(['ZZ'], nm)`,
> `qc.coherent_execute(nm)`, `qc.coherent_estimate(['ZZ'], nm)`,
> `qc.full_noise_execute(nm)`, `qc.full_noise_estimate(['ZZ'], nm)`.

---

### Path Integral Simulation

The **Path Integral** backend computes the probability of a single output
state without building the full statevector. Instead of propagating all 2^n
amplitudes, it traces only the Pauli paths that contribute to the target
state.

This is particularly useful for:
- **QUBO/HUBO validation** — check if a known optimal solution has high probability
- **Single-state queries at scale** — probe P(|target⟩) for circuits where
  statevector/MPS would be too slow
- **Circuits with few branching gates** — cost scales as O(2^b) where b is
  the number of H/Rx/Ry gates, not O(2^n)

#### Using `qc.prob()` (Recommended)

The simplest way to query a single state probability:

```python
from maestro.circuits import QuantumCircuit

qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)

result = qc.prob('11')
print(result['probability'])   # 0.5
print(result['amplitude'])     # (0.707+0j)
print(result['target_state'])  # '11'
print(result['time_taken'])    # seconds
```

The `prob()` method returns a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `probability` | `float` | \|⟨target\|U\|0⟩\|² |
| `amplitude` | `complex` | ⟨target\|U\|0⟩ |
| `target_state` | `str` | The queried bitstring |
| `time_taken` | `float` | Computation time in seconds |

#### Module-Level `state_probability()`

Alternatively, use the module-level function which also accepts QASM strings:

```python
import maestro
from maestro.circuits import QuantumCircuit

# From a QuantumCircuit
qc = QuantumCircuit()
qc.h(0)
qc.cx(0, 1)
result = maestro.state_probability(qc, '11')

# From a QASM string
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""
result = maestro.state_probability(qasm, '11')
```

#### Via SimulatorConfig

Path Integral is also available as a `SimulationType` for `execute()` and
`estimate()`:

```python
config = maestro.SimulatorConfig(
    simulation_type=maestro.SimulationType.PathIntegral
)
result = qc.execute(config=config, shots=1024)
result = qc.estimate(['ZZ'], config=config)
```

#### Performance Characteristics

Path Integral cost depends on **branching gates** (H, Rx, Ry), not qubit count:

| Circuit Type | Branching | PI Cost | vs Statevector |
|---|---|---|---|
| Mostly Rz/T/CX (few H) | O(1) | **Constant** | 1000x+ faster at 20+ qubits |
| QAOA (H+Rx on all qubits) | O(n) | O(2^2n) | Slower than SV |
| Warm-start (Rx on k qubits) | O(k) | O(2^k) | Fast if k << n |

> **When to use Path Integral:**
> - Querying P(|target⟩) for a specific bitstring
> - Circuits dominated by diagonal gates (Rz, T, S, CZ, CX)
> - QUBO validation where you know the expected optimal solution
>
> **When NOT to use Path Integral:**
> - Full statevector/counts needed (use Statevector or MPS)
> - QAOA circuits with H/Rx on every qubit (branching = 2n)

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
quest_config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QuestSim,
)
result = qc.execute(config=quest_config, shots=1024)

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

quest_config = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.QuestSim,
)

# Execute
result = maestro.simple_execute(qasm_circuit, config=quest_config, shots=1024)
print(f"Counts: {result['counts']}")

# Estimate expectation values
estimate = maestro.simple_estimate(
    qasm_circuit, observables="ZZ;XX", config=quest_config
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
gpu_sv = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
)
result = qc.execute(config=gpu_sv, shots=2048)
print(f"GPU Counts: {result['counts']}")

# Execute on GPU with MPS
gpu_mps = maestro.SimulatorConfig(
    simulator_type=maestro.SimulatorType.Gpu,
    simulation_type=maestro.SimulationType.MatrixProductState,
    max_bond_dimension=64,
)
result_mps = qc.execute(config=gpu_mps, shots=2048)
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
    config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QuestSim)
    print("Ran on QuEST")
elif maestro.init_gpu():
    config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.Gpu)
    print("Ran on GPU")
else:
    config = maestro.SimulatorConfig()  # Default: QCSim Statevector
    print("Ran on CPU (QCSim)")

result = qc.execute(config=config)
print(f"Counts: {result['counts']}")
```

### API Reference

| Function                    | Returns | Description                                                  |
|-----------------------------|---------|--------------------------------------------------------------|
| `maestro.init_quest()`      | `bool`  | Load the QuEST shared library. Returns `True` on success.    |
| `maestro.is_quest_available()` | `bool` | Check if QuEST has been loaded and is ready to use.         |
| `maestro.init_gpu()`        | `bool`  | Load the GPU shared library. Returns `True` on success.      |
| `maestro.is_gpu_available()`| `bool`  | Check if the GPU backend has been loaded and is ready.        |
| `maestro.NoiseModel()`      | object  | Create a noise model for configuring per-qubit Pauli and coherent noise channels. |
| `maestro.noisy_estimate()`  | `dict`  | Analytical noisy estimation (zero overhead, per-qubit damping). |
| `maestro.noisy_estimate_montecarlo()` | `dict` | Gate-by-gate Monte Carlo noisy estimation (accurate). |
| `maestro.noisy_execute()`   | `dict`  | Monte Carlo noisy execution with shot-based sampling.        |
| `maestro.coherent_estimate()` | `dict` | Coherent noise estimation (rotation errors, averaged over sign realizations). |
| `maestro.coherent_execute()` | `dict`  | Coherent noise execution with shot-based sampling.           |
| `maestro.full_noise_execute()` | `dict` | Combined noise execution (coherent + crosstalk + T1 + Pauli). |
| `maestro.full_noise_estimate()` | `dict` | Combined noise estimation (all layers, expectation values). |
| `maestro.state_probability()` | `dict` | Compute P(\|target⟩) via path integral (circuit or QASM input). |
| `qc.prob(target_state)`     | `dict`  | Compute P(\|target⟩) via path integral (bound method).       |

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
