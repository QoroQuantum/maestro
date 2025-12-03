/**
 * @file advanced_simulation.cpp
 * @brief Advanced usage example of the Maestro library using the C interface
 * for manual control.
 *
 * This example demonstrates:
 * 1. Creating a specific simulator type (Statevector).
 * 2. Manually applying gates (H, CX).
 * 3. Performing a measurement without collapsing the state (if supported) or
 * standard measurement.
 * 4. Cleaning up resources.
 */

#include "maestrolib/Interface.h"
#include <iostream>
#include <vector>

int main() {
  // 1. Initialize Maestro
  void *maestro = GetMaestroObject();
  if (!maestro) {
    std::cerr << "Failed to initialize Maestro." << std::endl;
    return 1;
  }

  // 2. Create a Simulator
  // Create a Statevector simulator.
  // Simulators::SimulatorType::Statevector is typically 0 (check
  // Simulators/Simulator.h or documentation).
  // Simulators::SimulationType::QiskitAer is typically 0.
  // We use 0, 0 here for demonstration.
  unsigned long int simHandle = CreateSimulator(0, 0);
  if (simHandle == 0) {
    std::cerr << "Failed to create simulator." << std::endl;
    return 1;
  }

  void *sim = GetSimulator(simHandle);
  if (!sim) {
    std::cerr << "Failed to get simulator instance." << std::endl;
    DestroySimulator(simHandle);
    return 1;
  }

  std::cout << "Simulator created with handle: " << simHandle << std::endl;

  // 3. Apply Gates Manually
  // We will create a Bell state: |00> -> H(0) -> |+0> -> CX(0,1) -> (|00> +
  // |11>) / sqrt(2)

  // Apply Hadamard on qubit 0
  ApplyH(sim, 0);
  std::cout << "Applied H on qubit 0" << std::endl;

  // Apply CNOT with control 0 and target 1
  ApplyCX(sim, 0, 1);
  std::cout << "Applied CX on 0 -> 1" << std::endl;

  // 4. Measure
  // MeasureNoCollapse returns the outcome without collapsing the state vector
  // (useful for debugging/inspection) Note: This might not be supported by all
  // backends.
  unsigned long long int outcome = MeasureNoCollapse(sim);
  std::cout << "Measurement outcome (no collapse): " << outcome << std::endl;

  // Standard Measure (collapses state)
  // We need to specify which qubits to measure.
  unsigned long int qubits[] = {0, 1};
  unsigned long long int measurement = Measure(sim, qubits, 2);
  std::cout << "Measurement outcome (collapsed): " << measurement << std::endl;

  // 5. Cleanup
  DestroySimulator(simHandle);

  return 0;
}
