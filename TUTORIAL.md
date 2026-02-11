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

## Python

Maestro provides Python bindings for ease of use, allowing you to integrate its high-performance simulation capabilities into your Python-based quantum workflows.

### Installation

To install the Python bindings, run the following command from the root of the Maestro repository:

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
print(f"Simulator: {result['simulator']}")
print(f"Method: {result['method']}")
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

| Method     | Gate             | Description                  |
|------------|------------------|------------------------------|
| `qc.x(q)`  | Pauli-X          | Bit flip                     |
| `qc.y(q)`  | Pauli-Y          | Bit + phase flip             |
| `qc.z(q)`  | Pauli-Z          | Phase flip                   |
| `qc.h(q)`  | Hadamard         | Creates superposition        |
| `qc.s(q)`  | S Gate           | π/2 phase                    |
| `qc.sdg(q)` | S† Gate         | −π/2 phase                   |
| `qc.t(q)`  | T Gate           | π/4 phase                    |
| `qc.tdg(q)` | T† Gate         | −π/4 phase                   |
| `qc.sx(q)` | √X Gate          | Square root of X             |

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
| `qc.swap(a, b)`    | SWAP    | Swaps two qubits                 |

**Controlled Parametric Gates**

| Method                  | Gate | Parameters          |
|-------------------------|------|---------------------|
| `qc.cp(c, t, λ)`       | CP   | `λ` (lambda)        |
| `qc.crx(c, t, θ)`      | CRx  | `θ` (theta)         |
| `qc.cry(c, t, θ)`      | CRy  | `θ` (theta)         |
| `qc.crz(c, t, θ)`      | CRz  | `θ` (theta)         |

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

For more granular control, you can use the `Maestro` and `ISimulator` classes directly.

```python
from maestro import Maestro, SimulatorType, SimulationType

# Initialize Maestro
m = Maestro()

# Create a simulator handle
# Defaults to QCSim and MatrixProductState
sim_handle = m.CreateSimulator(SimulatorType.QCSim, SimulationType.Statevector)

# Get the simulator object
sim = m.GetSimulator(sim_handle)

# Apply gates manually
sim.ApplyH(0)
sim.ApplyCX(0, 1)

# Sample counts
counts = sim.SampleCounts([0, 1], shots=1000)
print(f"Counts: {counts}")

# Cleanup
m.DestroySimulator(sim_handle)
```

### Examples

You can find several complete examples in the `examples/` directory:

- `examples/python_example_1.py`: Basic simulation and sampling.
- `examples/python_example_2.py`: Advanced simulation with manual gate application.
- `examples/python_example_3.py`: Working with expectation values and observables.
