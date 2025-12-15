/**
 * @file basic_simulation.cpp
 * @brief Basic usage example of the Maestro library using the C interface.
 *
 * This example demonstrates:
 * 1. Initializing the Maestro library.
 * 2. Creating a simple simulator.
 * 3. Defining a quantum circuit using OpenQASM 2.0.
 * 4. Executing the circuit.
 * 5. Parsing and printing the results.
 * 6. Cleaning up resources.
 */

#include "maestrolib/Interface.h"
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Helper function to print the JSON result.
 *
 * @param jsonResult The JSON string returned by the simulator.
 */
void PrintResults(const char *jsonResult) {
  if (!jsonResult) {
    std::cout << "No results returned." << std::endl;
    return;
  }
  std::cout << "Simulation Results: " << jsonResult << std::endl;
}

int main() {
  // 1. Initialize Maestro
  // Get the singleton instance of the Maestro engine
  void *maestro = GetMaestroObject();
  if (!maestro) {
    std::cerr << "Failed to initialize Maestro." << std::endl;
    return 1;
  }

  // 2. Create a Simulator
  // Create a simple simulator for 2 qubits.
  // This returns a handle (ID) to the simulator.
  unsigned long int simHandle = CreateSimpleSimulator(2);
  if (simHandle == 0) {
    std::cerr << "Failed to create simulator." << std::endl;
    return 1;
  }

  std::cout << "Simulator created with handle: " << simHandle << std::endl;

  // 3. Define a Circuit
  // We'll use a simple Bell State circuit in OpenQASM 2.0 format.
  const char *qasmCircuit =
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
  const char *config = "{\"shots\": 1024}";

  // 5. Execute the Circuit
  // SimpleExecute takes the simulator handle, circuit string, and config.
  // It returns a JSON string with the results.
  char *result = SimpleExecute(simHandle, qasmCircuit, config);

  // 6. Process Results
  PrintResults(result);

  // 7. Cleanup
  // Free the result string memory
  FreeResult(result);

  // Destroy the simulator instance
  DestroySimpleSimulator(simHandle);

  return 0;
}
