#include "Execution/IncrementalEvolution.h"
#include "core.h"
#include "runtime.h"

namespace maestro_bindings
{

// Each observation reads the current simulator state. Both the quantum and
// classical registers continue into the next step on the same trajectory.
nb::dict incremental_evolve_core(std::shared_ptr<Circuits::Circuit<double>> init_circuit, std::shared_ptr<Circuits::Circuit<double>> trotter_step,
                                 const std::vector<int> &measure_at_steps, const std::vector<std::string> &paulis, const SimulatorConfig &config)
{
    if (!init_circuit)
        throw nb::value_error("init_circuit is null.");
    if (!trotter_step)
        throw nb::value_error("trotter_step is null.");
    if (measure_at_steps.empty())
        throw nb::value_error("measure_at_steps must not be empty.");

    // Determine qubit count from circuits and observables. The simulator runs
    // the circuits index by index, so the register holds every one of them.
    size_t width = std::max(CircuitWidth(*init_circuit), CircuitWidth(*trotter_step));
    for (const auto &p : paulis)
        width = std::max(width, p.length());
    const int num_qubits = RegisterQubits(width);

    // Evolution drives the simulator itself: resolve the one backend it runs
    // and allocate exactly its register, without the network's entanglement
    // qubit. That is the requested pair, or the first listed candidate whose
    // register fits (automatic selection without candidates allows only the
    // requested pair). Probing several candidates already configured the
    // winner's network, register included: adopt it rather than allocate the
    // register again.
    const size_t register_qubits = static_cast<size_t>(num_qubits);
    unsigned long int probed = 0;
    auto resolved = ResolveCandidateBackend(config, register_qubits, probed);
    resolved.fixed_backend = true;
    ScopedSimulator sim(num_qubits, probed);
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

    auto network =
        probed ? static_cast<Maestro *>(GetMaestroObject())->GetSimpleSimulator(sim.handle) : ConfigureNetwork(sim.handle, resolved, register_qubits);
    if (!network)
        throw std::runtime_error("incremental_evolve: failed to configure network.");
    auto simulator = network->GetSimulator();
    if (!simulator)
        throw std::runtime_error("incremental_evolve: requested simulator/simulation type is not "
                                 "available.");

    // Disable circuit optimization to preserve gate ordering
    network->GetController()->SetOptimizeCircuit(false);

    Circuits::OperationState opState(std::max(init_circuit->GetMaxCbitIndex(), trotter_step->GetMaxCbitIndex()) + 1);
    const MaestroExecution::IncrementalEvolution evolution(init_circuit, trotter_step, std::max(0, sorted_steps.back()));

    size_t current_max_bond_dim = 0;

    auto start = std::chrono::high_resolution_clock::now();

    {
        nb::gil_scoped_release release;
        evolution.Start(simulator, opState, current_max_bond_dim);
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
            for (int s = current_step; s < target_step; ++s)
                evolution.Step(s, simulator, opState, current_max_bond_dim);
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
    // The backend evolution drove, which a candidate list may have chosen.
    py_result["simulator"] = simulator->GetType();
    py_result["method"] = simulator->GetSimulationType();
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
