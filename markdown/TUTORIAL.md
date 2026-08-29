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
