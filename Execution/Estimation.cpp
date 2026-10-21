#include "Estimation.h"

#include "SimulatorConfig.h"
#include "Transforms/BoundedPauli.h"
#include "Transforms/ConeAnalysis.h"
#include "Transforms/ObservableClustering.h"
#include "Transforms/TransformPipeline.h"
#include "Json/Json.h"

#include <chrono>
#include <numeric>

namespace MaestroExecution
{
namespace
{
using Clock = std::chrono::steady_clock;

size_t Width(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables)
{
    size_t width = circuit->GetMaxQubitIndex() + 1;
    for (const auto &p : observables)
        width = std::max(width, p.size());
    return std::max(size_t{1}, width);
}

std::optional<size_t> ClusterBudget(const SimulatorConfig &config)
{
    if (config.max_cluster_qubits)
        return config.max_cluster_qubits;
    // Bound dense batching by default (16 MiB of complex doubles). A single
    // larger cone remains intact: this is an exact grouping policy, not a cut.
    if (config.simulation_type == Simulators::SimulationType::kStatevector)
        return 20;
    if (config.simulation_type == Simulators::SimulationType::kDensityMatrix)
        return 10;
    return std::nullopt;
}

void PadFixedTensorNetwork(EstimationGroup &group, const SimulatorConfig &config)
{
    // The GPU tensor-network backend needs two qubits. Keep fixed selection by
    // supplying an idle identity wire instead of changing backend or failing a
    // request whose physical circuit was wide enough before cone reduction.
    if (config.fixed_backend && config.simulator_type == Simulators::SimulatorType::kGpuSim &&
        config.simulation_type == Simulators::SimulationType::kTensorNetwork && group.qubits == 1)
    {
        group.qubits = 2;
        for (auto &p : group.observables)
            p.resize(2, 'I');
    }
}

bool SameUnitaryCircuit(const EstimationGroup &a, const EstimationGroup &b)
{
    if (a.qubits != b.qubits)
        return false;
    const auto &left = a.circuit->GetOperations(), &right = b.circuit->GetOperations();
    if (left.size() != right.size())
        return false;
    for (size_t i = 0; i < left.size(); ++i)
    {
        const auto x = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(left[i]);
        const auto y = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(right[i]);
        // Do not combine stochastic trajectories or unknown custom matrices.
        if (!x || !y || x->GetGateType() == Circuits::QuantumGateType::kNone || x->GetGateType() != y->GetGateType() ||
            x->GetNumQubits() != y->GetNumQubits() || x->GetParams() != y->GetParams())
            return false;
        for (unsigned int q = 0; q < x->GetNumQubits(); ++q)
            if (x->GetQubit(q) != y->GetQubit(q))
                return false;
    }
    return true;
}

void FinishGroup(EstimationGroup group, const SimulatorConfig &config, EstimationPlan &plan)
{
    for (size_t i = group.indices.size(); i-- > 0;)
        if (!Simulators::IsDistributedGpuSimulator(config.simulator_type) &&
            (group.factors[i] == 0 || group.observables[i].find_first_not_of("Ii") == std::string::npos))
        {
            plan.resolved_values[group.indices[i]] = group.factors[i];
            group.indices.erase(group.indices.begin() + i);
            group.factors.erase(group.factors.begin() + i);
            group.observables.erase(group.observables.begin() + i);
        }
    if (group.indices.empty())
        return;
    group.qubits = Width(group.circuit, group.observables);
    PadFixedTensorNetwork(group, config);
    for (auto &existing : plan.groups)
        if (SameUnitaryCircuit(existing, group))
        {
            existing.indices.insert(existing.indices.end(), group.indices.begin(), group.indices.end());
            existing.observables.insert(existing.observables.end(), group.observables.begin(), group.observables.end());
            existing.factors.insert(existing.factors.end(), group.factors.begin(), group.factors.end());
            return;
        }
    plan.qubits_after = std::max(plan.qubits_after, group.qubits);
    plan.total_cluster_qubits += group.qubits;
    plan.groups.push_back(std::move(group));
}

bool PrepareCached(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables, const SimulatorConfig &config,
                   EstimationCache &cache, EstimationPlan &plan)
{
    if (config.auto_reduce || config.observable_error_tolerance || !config.enable_causal_cone_reduction ||
        Simulators::IsDistributedGpuSimulator(config.simulator_type) ||
        !CanTraceCone(circuit, observables, HasEnsembleState(config.simulation_type), config.allow_reset_marginalization))
        return false;
    std::vector<size_t> signature{Width(circuit, observables),
                                  static_cast<size_t>(config.simulator_type),
                                  static_cast<size_t>(config.simulation_type),
                                  config.enable_observable_clustering,
                                  config.fixed_backend,
                                  config.allow_reset_marginalization,
                                  ClusterBudget(config).value_or(0),
                                  config.min_cluster_qubits.value_or(0)};
    for (const auto &op : circuit->GetOperations())
    {
        const auto qs = op->AffectedQubits();
        if (qs.size() <= 1)
            continue;
        signature.push_back(qs.size());
        signature.insert(signature.end(), qs.begin(), qs.end());
    }
    if (signature != cache.signature || observables != cache.observables)
    {
        cache.signature = std::move(signature);
        cache.observables = observables;
        cache.groups.clear();
        cache.mappings.clear();
        if (config.enable_observable_clustering && BenefitsFromObservableClustering(config.simulator_type, config.simulation_type))
            cache.groups = ClusterObservables(circuit, observables, ClusterBudget(config), HasEnsembleState(config.simulation_type), config.min_cluster_qubits,
                                              false, config.allow_reset_marginalization);
        else
        {
            cache.groups.emplace_back(observables.size());
            std::iota(cache.groups.back().begin(), cache.groups.back().end(), size_t{0});
        }
        for (const auto &group : cache.groups)
        {
            std::vector<std::string> subset;
            for (auto i : group)
                subset.push_back(observables[i]);
            cache.mappings.push_back(TraceStructuralCone(circuit, subset).mapping);
        }
    }
    else
        ++cache.hits;
    // Noise is injected in full, before reuse. This preserves delay scheduling,
    // spectator crosstalk, and RNG consumption. Cached structural maps allow
    // extra local operations but never reuse a noiseless symmetry projection.
    for (size_t g = 0; g < cache.groups.size(); ++g)
    {
        const auto &mapping = cache.mappings[g];
        EstimationGroup group;
        group.circuit = std::make_shared<Circuits::Circuit<double>>();
        group.indices = cache.groups[g];
        group.factors.assign(group.indices.size(), 1);
        group.qubits = std::max(size_t{1}, mapping.size());
        for (auto i : group.indices)
        {
            std::string p(group.qubits, 'I');
            for (const auto &[original, compact] : mapping)
                if (original < observables[i].size())
                    p[compact] = observables[i][original];
            group.observables.push_back(std::move(p));
        }
        for (const auto &op : circuit->GetOperations())
        {
            const auto qs = op->AffectedQubits();
            if (!qs.empty() && std::all_of(qs.begin(), qs.end(), [&](size_t q) { return mapping.count(q); }))
                group.circuit->AddOperation(op->Remap(mapping, {}));
        }
        TransformContext local{group.circuit, group.observables, config.simulator_type, config};
        TransformPipeline::Tighten(local, group.qubits, false);
        FinishGroup(std::move(group), config, plan);
    }
    return true;
}

void Scatter(std::vector<double> &values, const EstimationGroup &group, const std::vector<double> &sample)
{
    if (sample.size() != group.indices.size())
        throw std::runtime_error("Backend returned an invalid expectation vector");
    for (size_t i = 0; i < sample.size(); ++i)
        values[group.indices[i]] = sample[i] * group.factors[i];
}
} // namespace

EstimationPlan PrepareEstimation(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                 const SimulatorConfig &config, EstimationCache *cache)
{
    if (!circuit)
        throw std::invalid_argument("Estimation requires a non-null circuit");
    config.Validate();
    EstimationPlan plan;
    plan.qubits_before = Width(circuit, observables);
    plan.resolved_values.assign(observables.size(), 0);
    plan.error_bounds.assign(observables.size(), 0);
    if (observables.empty())
        return plan;
    const auto start = Clock::now();
    auto prepared = circuit->RemoveEndMeasurements();
    if (cache && PrepareCached(prepared, observables, config, *cache, plan))
    {
        plan.reduction_time = std::chrono::duration<double>(Clock::now() - start).count();
        return plan;
    }
    std::vector<std::string> paulis;
    std::vector<size_t> source_indices;
    BoundedPauliWorkspace bounded_workspace;
    for (size_t i = 0; i < observables.size(); ++i)
    {
        if (!Simulators::IsDistributedGpuSimulator(config.simulator_type) && observables[i].find_first_not_of("Ii") == std::string::npos)
        {
            plan.resolved_values[i] = 1;
            if (config.observable_error_tolerance && HasEnsembleState(config.simulation_type))
                ++plan.bounded_observables;
            continue;
        }
        if (config.observable_error_tolerance && HasEnsembleState(config.simulation_type) && !Simulators::IsDistributedGpuSimulator(config.simulator_type))
            if (const auto value = EvaluateBoundedPauli(prepared, observables[i], *config.observable_error_tolerance, &bounded_workspace))
            {
                plan.resolved_values[i] = value->value;
                plan.error_bounds[i] = value->error;
                ++plan.bounded_observables;
                continue;
            }
        source_indices.push_back(i);
        paulis.push_back(observables[i]);
    }
    if (paulis.empty())
    {
        plan.reduction_time = std::chrono::duration<double>(Clock::now() - start).count();
        return plan;
    }
    TransformContext global{prepared, paulis, config.simulator_type, config};
    global.expectation_factors.assign(paulis.size(), 1.0);
    TransformPipeline::ApplyParity(global);
    plan.auto_reduced = global.auto_reduced;

    std::vector<std::vector<size_t>> groups;
    if (config.enable_causal_cone_reduction && config.enable_observable_clustering && paulis.size() > 1 &&
        !Simulators::IsDistributedGpuSimulator(config.simulator_type) && BenefitsFromObservableClustering(config.simulator_type, config.simulation_type))
        groups = ClusterObservables(prepared, paulis, ClusterBudget(config), HasEnsembleState(config.simulation_type), config.min_cluster_qubits, true,
                                    config.allow_reset_marginalization);
    else
    {
        groups.emplace_back(paulis.size());
        std::iota(groups.back().begin(), groups.back().end(), size_t{0});
    }

    for (auto &indices : groups)
    {
        EstimationGroup group;
        group.circuit = prepared;
        for (auto i : indices)
        {
            group.indices.push_back(source_indices[i]);
            group.observables.push_back(paulis[i]);
            group.factors.push_back(global.expectation_factors[i]);
        }
        TransformContext local{group.circuit, group.observables, config.simulator_type, config};
        local.expectation_factors.assign(group.indices.size(), 1.0);
        TransformPipeline::Tighten(local, Width(prepared, paulis), !global.auto_reduced);
        for (size_t i = 0; i < group.factors.size(); ++i)
            group.factors[i] *= local.expectation_factors[i];
        group.qubits = Width(group.circuit, group.observables);
        plan.auto_reduced = plan.auto_reduced || local.auto_reduced;
        FinishGroup(std::move(group), config, plan);
    }
    plan.reduction_time = std::chrono::duration<double>(Clock::now() - start).count();
    return plan;
}

EstimationResult EstimateCircuit(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                 const SimulatorConfig &config, EstimationCache *cache)
{
    EstimationResult result;
    result.values.assign(observables.size(), 0.0);
    result.simulator = config.simulator_type;
    result.method = config.simulation_type;
    result.plan = PrepareEstimation(circuit, observables, config, cache);
    result.values = result.plan.resolved_values;
    result.time_taken = result.plan.reduction_time;
    for (const auto &group : result.plan.groups)
    {
        GetMaestroObjectWithMute();
        const auto handle = CreateSimpleSimulator(static_cast<int>(group.qubits));
        if (!handle)
            throw std::runtime_error("Failed to create simulator handle");

        struct Cleanup
        {
            unsigned long handle;

            ~Cleanup()
            {
                DestroySimpleSimulator(handle);
            }
        } cleanup{handle};

        auto cluster_config = config;
        if (!config.fixed_backend && group.qubits == 1 && config.simulator_type == Simulators::SimulatorType::kGpuSim &&
            config.simulation_type == Simulators::SimulationType::kTensorNetwork)
            cluster_config.simulation_type = Simulators::SimulationType::kStatevector;
        auto network = ConfigureNetwork(handle, cluster_config);
        if (!network)
            throw std::runtime_error("Failed to configure network");
        if (config.fixed_backend && (network->GetSimulator()->GetType() != cluster_config.simulator_type ||
                                     network->GetSimulator()->GetSimulationType() != cluster_config.simulation_type))
            throw Error("backend_unavailable", "Requested fixed backend/method is unavailable");
        if (!Simulators::IsDistributedGpuSimulator(config.simulator_type) && config.native_options.count("gpu_device") &&
            network->GetSimulator()->GetGpuDevice() != std::stoi(config.native_options.at("gpu_device")))
            throw Error("backend_unavailable", "Requested GPU device was not selected");
        const auto start = Clock::now();
        Scatter(result.values, group, network->ExecuteOnHostExpectations(group.circuit, 0, group.observables));
        result.time_taken += std::chrono::duration<double>(Clock::now() - start).count();
        if (result.cluster_methods.empty())
        {
            result.simulator = network->GetLastSimulatorType();
            result.method = network->GetLastSimulationType();
            result.gpu_device = network->GetLastGpuDevice();
        }
        result.cluster_methods.push_back(network->GetLastSimulationType());
        result.gate_fusion = result.gate_fusion || network->WasGateFusionEnabled();
        result.gate_fusion_max_qubits = std::max(result.gate_fusion_max_qubits, static_cast<size_t>(network->GetLastGateFusionMaxQubits()));
        result.max_bond_dimension = std::max(result.max_bond_dimension, network->GetCurrentMaxBondDimension());
    }
    return result;
}

std::vector<double> EstimateOnNetwork(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                      const std::shared_ptr<Network::INetwork<double>> &network, std::optional<size_t> host, bool auto_reduce,
                                      EstimationCache *cache)
{
    if (observables.empty())
        return {};
    const auto simulator = network->GetSimulator();
    if (!simulator || (!host && network->GetNumHosts() != 1))
        return host ? network->ExecuteOnHostExpectations(circuit, *host, observables) : network->ExecuteExpectations(circuit, observables);
    SimulatorConfig config;
    config.simulator_type = simulator->GetType();
    config.simulation_type = simulator->GetSimulationType();
    // Preparation must respect the minimum width of the network's existing
    // backend, which this adapter does not reconfigure.
    config.fixed_backend = true;
    config.auto_reduce = auto_reduce;
    const auto plan = PrepareEstimation(circuit, observables, config, cache);
    auto values = plan.resolved_values;
    for (const auto &group : plan.groups)
        Scatter(values, group, network->ExecuteOnHostExpectations(group.circuit, host.value_or(0), group.observables));
    return values;
}
} // namespace MaestroExecution
