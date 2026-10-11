#pragma once

#include "Circuit/Circuit.h"
#include "Simulators/Interfaces/Simulator.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace MaestroExecution
{
struct SimulatorConfig;

// Inputs must outlive the context; circuit must be non-null. Transforms replace
// the circuit with a copy when editing it so callers can safely reuse the source.
struct TransformContext
{
    std::shared_ptr<Circuits::Circuit<double>> &circuit;
    std::vector<std::string> &observables;
    Simulators::SimulatorType simulator_type;
    const SimulatorConfig &config;

    // Indicates whether parity reduction was applied to the circuit.
    bool auto_reduced = false;
    std::vector<double> expectation_factors{};

    // Effective register widths include observables and the one-qubit minimum.
    size_t qubits_before = 0;
    size_t qubits_after = 0;
    // Passes dispatched in order, including passes that conservatively no-op.
    // Diagnostics are reset on each Run.
    std::vector<std::string> applied_transforms{};
};
} // namespace MaestroExecution
