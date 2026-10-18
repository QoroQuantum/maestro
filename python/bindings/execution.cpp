#include "Execution/Transforms/ObservableClustering.h"
#include "Execution/Transforms/TransformPipeline.h"
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

    int num_qubits = std::max(1, static_cast<int>(circuit->GetMaxQubitIndex()) + 1);
    ScopedSimulator sim(num_qubits);
    if (sim.handle == 0)
        throw std::runtime_error("Failed to create simulator handle.");

    auto network = ConfigureNetwork(sim.handle, config);
    if (!network)
        throw std::runtime_error("Failed to configure network.");

    Network::INetwork<double>::ExecuteResults raw_results;

    // Release GIL for heavy computation
    auto start = std::chrono::high_resolution_clock::now();
    {
        nb::gil_scoped_release release;
        raw_results = network->RepeatedExecuteOnHost(circuit, 0, (size_t)shots);
    }
    auto end = std::chrono::high_resolution_clock::now();

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
    py_result["time_taken"] = std::chrono::duration<double>(end - start).count();
    py_result["simulator"] = (int)network->GetLastSimulatorType();
    py_result["method"] = (int)network->GetLastSimulationType();
    if (network->GetLastGpuDevice() >= 0)
        py_result["gpu_device"] = network->GetLastGpuDevice();

    size_t max_bond_dim = network->GetCurrentMaxBondDimension();
    if (max_bond_dim > 0)
        py_result["max_bond_dim_reached"] = max_bond_dim;

    return py_result;
}

// Core Estimation Logic
nb::dict estimate_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const std::vector<std::string> &observables, const SimulatorConfig &config)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");

    // Partition observables into groups with disjoint lightcones; each group is
    // reduced and simulated on a register sized to its own cone.
    std::vector<std::vector<size_t>> groups;
    auto cluster_start = std::chrono::high_resolution_clock::now();
    const bool cluster = config.enable_causal_cone_reduction && config.enable_observable_clustering && observables.size() > 1 &&
                         !Simulators::IsDistributedGpuSimulator(config.simulator_type) &&
                         MaestroExecution::BenefitsFromObservableClustering(config.simulator_type, config.simulation_type);
    if (cluster)
    {
        nb::gil_scoped_release release;
        groups = MaestroExecution::ClusterObservables(circuit, observables, config.max_cluster_qubits);
    }
    else
    {
        groups.emplace_back(observables.size());
        std::iota(groups.back().begin(), groups.back().end(), size_t{0});
    }
    auto reduction_duration = std::chrono::high_resolution_clock::now() - cluster_start;
    std::chrono::high_resolution_clock::duration execution_duration{};

    std::vector<double> expectations(observables.size(), 0.0);
    bool auto_reduced = false;
    size_t qubits_before = 0;
    size_t largest_cluster_qubits = 0;
    size_t total_cluster_qubits = 0;
    size_t max_bond_dim = 0;
    int simulator_type = 0, method = 0, gpu_device = -1;
    bool first = true;

    for (const auto &group : groups)
    {
        std::vector<std::string> paulis;
        paulis.reserve(group.size());
        for (auto index : group)
            paulis.push_back(observables[index]);

        auto cluster_circuit = circuit;
        MaestroExecution::TransformContext ctx{cluster_circuit, paulis, config.simulator_type, config};
        auto transform_start = std::chrono::high_resolution_clock::now();
        {
            nb::gil_scoped_release release;
            MaestroExecution::TransformPipeline::Run(ctx);
        }
        reduction_duration += std::chrono::high_resolution_clock::now() - transform_start;

        int num_qubits = static_cast<int>(ctx.circuit->GetMaxQubitIndex()) + 1;
        for (const auto &p : ctx.observables)
            num_qubits = std::max(num_qubits, (int)p.length());
        num_qubits = std::max(1, num_qubits);

        ScopedSimulator sim(num_qubits);
        if (sim.handle == 0)
            throw std::runtime_error("Failed to create simulator handle.");

        auto network = ConfigureNetwork(sim.handle, config);
        if (!network)
            throw std::runtime_error("Failed to configure network.");

        std::vector<double> group_expectations;
        auto start = std::chrono::high_resolution_clock::now();
        {
            nb::gil_scoped_release release;
            group_expectations = network->ExecuteOnHostExpectations(ctx.circuit, 0, ctx.observables);
        }
        execution_duration += std::chrono::high_resolution_clock::now() - start;

        // Scatter back to the caller's observable order.
        for (size_t k = 0; k < group.size() && k < group_expectations.size(); ++k)
            expectations[group[k]] = group_expectations[k] * ctx.expectation_factors[k];

        auto_reduced = auto_reduced || ctx.auto_reduced;
        qubits_before = std::max(qubits_before, ctx.qubits_before);
        largest_cluster_qubits = std::max(largest_cluster_qubits, ctx.qubits_after);
        total_cluster_qubits += ctx.qubits_after;
        max_bond_dim = std::max(max_bond_dim, network->GetCurrentMaxBondDimension());
        if (first)
        {
            simulator_type = (int)network->GetLastSimulatorType();
            method = (int)network->GetLastSimulationType();
            gpu_device = network->GetLastGpuDevice();
            first = false;
        }
    }

    // Convert to Python list
    nb::list exp_vals;
    for (double value : expectations)
        exp_vals.append(value);

    nb::dict py_result;
    py_result["expectation_values"] = exp_vals;
    py_result["auto_reduced"] = auto_reduced;
    py_result["qubits_before"] = qubits_before;
    py_result["qubits_after"] = largest_cluster_qubits;
    py_result["clusters_count"] = groups.size();
    py_result["largest_cluster_qubits"] = largest_cluster_qubits;
    py_result["mean_cluster_qubits"] = groups.empty() ? 0.0 : static_cast<double>(total_cluster_qubits) / static_cast<double>(groups.size());
    py_result["reduction_time"] = std::chrono::duration<double>(reduction_duration).count();
    py_result["time_taken"] = std::chrono::duration<double>(execution_duration + reduction_duration).count();
    py_result["simulator"] = simulator_type;
    py_result["method"] = method;
    if (gpu_device >= 0)
        py_result["gpu_device"] = gpu_device;
    if (max_bond_dim > 0)
        py_result["max_bond_dim_reached"] = max_bond_dim;

    return py_result;
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
