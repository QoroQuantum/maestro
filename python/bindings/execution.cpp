#include "Execution/Estimation.h"
#include "Execution/Sampling.h"
#include "core.h"
#include "runtime.h"
#include <numeric>

namespace maestro_bindings
{

// Helper to parse observables from String (";" sep) or List[str]
std::vector<std::string> ParseObservables(const nb::object &observables)
{
    std::vector<std::string> paulis;

    if (nb::isinstance<nb::str>(observables))
    {
        std::string obsStr = nb::cast<std::string>(observables);
        std::stringstream ss(obsStr);
        std::string item;
        while (std::getline(ss, item, ';'))
        {
            // Trim whitespace if necessary, usually safe to skip empty
            if (!item.empty())
                paulis.push_back(item);
        }
    }
    else if (nb::isinstance<nb::list>(observables))
    {
        paulis = nb::cast<std::vector<std::string>>(observables);
    }
    else
    {
        throw nb::type_error("Observables must be a ';'-separated string or a list of strings.");
    }
    return paulis;
}

// Core Execution Logic
nb::dict execute_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");

    if (shots < 0)
        throw nb::value_error("shots must be nonnegative");
    MaestroExecution::SamplingResult result;
    {
        nb::gil_scoped_release release;
        result = MaestroExecution::SampleCircuit(circuit, config, static_cast<size_t>(shots));
    }
    const auto &raw_results = result.counts;

    // Process results back in Python land
    nb::dict counts;
    for (const auto &pair : raw_results)
    {
        // Optimization: Pre-allocate string to avoid repeated reallocation
        const auto &bool_vec = pair.first;
        std::string bitstring(bool_vec.size(), '0');
        for (size_t i = 0; i < bool_vec.size(); ++i)
        {
            if (bool_vec[i])
                bitstring[i] = '1';
        }
        counts[bitstring.c_str()] = pair.second;
    }

    nb::dict py_result;
    py_result["counts"] = counts;
    py_result["time_taken"] = result.time_taken;
    py_result["simulator"] = static_cast<int>(result.simulator);
    py_result["method"] = static_cast<int>(result.method);
    py_result["sampling_components"] = result.components;
    py_result["largest_component_qubits"] = result.largest_component;
    if (result.gpu_device >= 0)
        py_result["gpu_device"] = result.gpu_device;
    if (result.max_bond_dimension)
        py_result["max_bond_dim_reached"] = result.max_bond_dimension;

    return py_result;
}

// Core Estimation Logic
nb::dict estimate_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const std::vector<std::string> &observables, const SimulatorConfig &config,
                       MaestroExecution::EstimationCache *cache)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");

    MaestroExecution::EstimationResult result;
    {
        nb::gil_scoped_release release;
        result = MaestroExecution::EstimateCircuit(circuit, observables, config, cache);
    }
    const auto &plan = result.plan;
    nb::dict out;
    out["expectation_values"] = nb::cast(result.values);
    out["auto_reduced"] = plan.auto_reduced;
    out["expectation_error_bounds"] = nb::cast(plan.error_bounds);
    out["bounded_observables"] = plan.bounded_observables;
    out["qubits_before"] = plan.qubits_before;
    out["qubits_after"] = plan.qubits_after;
    out["clusters_count"] = plan.groups.size();
    out["largest_cluster_qubits"] = plan.qubits_after;
    out["mean_cluster_qubits"] = plan.groups.empty() ? 0.0 : static_cast<double>(plan.total_cluster_qubits) / plan.groups.size();
    out["reduction_time"] = plan.reduction_time;
    out["time_taken"] = result.time_taken;
    out["simulator"] = static_cast<int>(result.simulator);
    out["method"] = static_cast<int>(result.method);
    nb::list methods;
    for (auto method : result.cluster_methods)
        methods.append(static_cast<int>(method));
    out["cluster_methods"] = methods;
    // GPU estimates resolved without simulation retain -1 (no device used).
    if (result.simulator == Simulators::SimulatorType::kGpuSim || result.gpu_device >= 0)
        out["gpu_device"] = result.gpu_device;
    if (result.max_bond_dimension)
        out["max_bond_dim_reached"] = result.max_bond_dimension;
    return out;
}

// Core Statevector Logic
std::vector<std::complex<double>> statevector_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");

    int num_qubits = std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
    ScopedSimulator sim(num_qubits);
    if (sim.handle == 0)
        throw std::runtime_error("Failed to create simulator handle.");

    auto network = ConfigureNetwork(sim.handle, config);
    if (!network)
        throw std::runtime_error("Failed to configure network.");

    std::vector<std::complex<double>> amplitudes;
    {
        nb::gil_scoped_release release;
        amplitudes = network->ExecuteOnHostAmplitudes(circuit, 0);
    }
    return amplitudes;
}

} // namespace maestro_bindings
