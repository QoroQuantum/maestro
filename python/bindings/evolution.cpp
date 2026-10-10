#include "core.h"
#include "runtime.h"

namespace maestro_bindings
{

// Core Incremental Time Evolution Logic
// Uses SaveState/RestoreState to avoid re-simulating from scratch at each
// measurement point. Instead of building a fresh circuit with k Trotter steps
// per measurement, this creates one simulator, evolves forward incrementally,
// and checkpoints the MPS state for expectation value computation.
//
// Cost: O(total_steps) instead of O(Σ step_i) ≈ O(n_points × avg_step).
nb::dict incremental_evolve_core(std::shared_ptr<Circuits::Circuit<double>> init_circuit, std::shared_ptr<Circuits::Circuit<double>> trotter_step,
                                 const std::vector<int> &measure_at_steps, const std::vector<std::string> &paulis, const SimulatorConfig &config)
{
    if (!init_circuit)
        throw nb::value_error("init_circuit is null.");
    if (!trotter_step)
        throw nb::value_error("trotter_step is null.");
    if (measure_at_steps.empty())
        throw nb::value_error("measure_at_steps must not be empty.");

    // Determine qubit count from circuits and observables
    int num_qubits = std::max(1, static_cast<int>(init_circuit->GetMaxQubitIndex()) + 1);
    num_qubits = std::max(num_qubits, static_cast<int>(trotter_step->GetMaxQubitIndex()) + 1);
    for (const auto &p : paulis)
        num_qubits = std::max(num_qubits, (int)p.length());

    ScopedSimulator sim(num_qubits);
    if (sim.handle == 0)
        throw std::runtime_error("incremental_evolve: failed to create simulator handle.");

    // Sort measurement steps for sequential processing
    std::vector<int> sorted_steps = measure_at_steps;
    std::sort(sorted_steps.begin(), sorted_steps.end());

    // Results storage: one vector of expectation values per measurement point
    nb::list all_expectations;
    nb::list steps_measured;
    nb::list bond_dim_evolution;
    nb::list times_per_step;

    auto network = ConfigureNetwork(sim.handle, config);
    if (!network)
        throw std::runtime_error("incremental_evolve: failed to configure network.");

    // Simulator and simulation types need to be set explicitely.
    // ConfigureNetwork returns a dummy QCSim MPS simulator, backend must be set
    // after Network at runtime
    network->CreateSimulator(config.simulator_type, config.simulation_type);
    auto simulator = network->GetSimulator();
    if (!simulator)
        throw std::runtime_error("incremental_evolve: requested simulator/simulation type is not "
                                 "available.");

    // Disable circuit optimization to preserve gate ordering
    network->GetController()->SetOptimizeCircuit(false);

    Circuits::OperationState opState;
    opState.AllocateBits(num_qubits);

    size_t current_max_bond_dim = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Execute initial circuit (non-measurement gates) — release GIL
    {
        nb::gil_scoped_release release;
        init_circuit->ExecuteNonMeasurements(simulator, opState, &current_max_bond_dim);
    }

    int current_step = 0;
    for (int target_step : sorted_steps)
    {
        auto start_step = std::chrono::high_resolution_clock::now();
        int delta = target_step - current_step;
        if (delta < 0)
            continue; // duplicate or out-of-order (shouldn't happen after sort)

        // Apply delta trotter steps — release GIL for heavy computation
        {
            nb::gil_scoped_release release;
            for (int s = 0; s < delta; ++s)
            {
                trotter_step->ExecuteNonMeasurements(simulator, opState, &current_max_bond_dim);
            }
        }
        current_step = target_step;

        // Share tensor contractions across the observables at this state.
        std::vector<double> values;
        {
            nb::gil_scoped_release release;
            values = simulator->ExpectationValues(paulis);
        }
        nb::list step_exp;
        for (double value : values)
            step_exp.append(value);

        auto end_step = std::chrono::high_resolution_clock::now();

        all_expectations.append(step_exp);
        steps_measured.append(target_step);
        bond_dim_evolution.append(current_max_bond_dim);
        times_per_step.append(std::chrono::duration<double>(end_step - start_step).count());
    }

    auto end = std::chrono::high_resolution_clock::now();

    nb::dict py_result;
    py_result["expectation_values"] = all_expectations;
    py_result["steps"] = steps_measured;
    py_result["time_taken"] = std::chrono::duration<double>(end - start).count();
    py_result["time_per_step"] = times_per_step;
    py_result["simulator"] = (int)config.simulator_type;
    py_result["method"] = (int)config.simulation_type;
    if (network->GetGpuDevice() >= 0)
        py_result["gpu_device"] = network->GetGpuDevice();

    py_result["dynamic_bond_dims"] = bond_dim_evolution;
    if (current_max_bond_dim > 0)
    {
        py_result["max_bond_dim_reached"] = current_max_bond_dim;
    }

    return py_result;
}

} // namespace maestro_bindings
