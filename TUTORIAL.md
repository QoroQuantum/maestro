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
