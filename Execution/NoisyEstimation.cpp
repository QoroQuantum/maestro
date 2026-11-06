#include "NoisyEstimation.h"

#include "../Utils/Threads/RealizationWorkers.h"
#include "CircuitHelpers.h"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace MaestroExecution
{
RecordCircuitPtr WithoutCircuitChannels(const RecordCircuitPtr &circuit)
{
    auto out = std::make_shared<Circuits::Circuit<double>>();
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() != Circuits::OperationType::kQuantumChannel)
            out->AddOperation(op);
    return out;
}

NoisyEstimate EstimateNoisy(const RecordCircuitPtr &circuit, const std::vector<std::string> &observables, const noise::NoiseModel &noise_model,
                            size_t noise_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed, NoiseLayer layer,
                            std::optional<bool> exact_channels)
{
    if (!circuit)
        throw std::invalid_argument("Circuit is null.");
    if (noise_realizations < 1)
        throw std::invalid_argument("noise_realizations must be >= 1.");
    auto rng = MakeNoiseRng(config, noise_seed);
    NoisyEstimate out;
    out.realizations = noise_realizations;
    if (observables.empty())
    {
        out.ideal = EstimateCircuit(WithoutCircuitChannels(circuit), observables, config);
        return out;
    }
    const size_t count = observables.size();
    const auto clean = circuit->RemoveEndMeasurements();
    const bool exact = exact_channels.value_or(UsesExactQuantumChannels(config));
    const noise::NoiseInjectionPlan plan(clean, noise_model);

    // What the aggregate needs of each realization, kept in realization order.
    struct Realization
    {
        std::vector<double> values, bounds;
        std::vector<Simulators::SimulationType> methods;
        bool reduced = false;
        size_t before = 0, after = 0, groups = 0, cluster_qubits = 0, bounded = 0;
        double reduction_time = 0;
    };

    std::vector<Realization> realizations(noise_realizations);
    const size_t qubits = std::max({size_t{1}, clean->GetMaxQubitIndex() + 1, plan.RegisterWidth()});
    const size_t workers = SamplingWorkers(config, noise_realizations, qubits);
    // A cache serves one thread (EstimationCache).
    std::vector<EstimationCache> caches(workers);
    const auto run = [&](EstimationCache &cache, size_t r, const RecordCircuitPtr &noisy) {
        auto result = EstimateCircuit(noisy, observables, NoiseExecutionConfig(config, noise_seed, r), &cache);
        auto &record = realizations[r];
        record.values = std::move(result.values);
        record.bounds = std::move(result.plan.error_bounds);
        record.methods = std::move(result.cluster_methods);
        record.reduced = result.plan.auto_reduced;
        record.before = result.plan.qubits_before;
        record.after = result.plan.qubits_after;
        record.groups = result.plan.groups.size();
        record.cluster_qubits = result.plan.total_cluster_qubits;
        record.bounded = result.plan.bounded_observables;
        record.reduction_time = result.plan.reduction_time;
    };
    const auto start = std::chrono::steady_clock::now();
    size_t first = 0, warm_hits = 0;
    if (workers > 1)
    {
        // Realization 0 warms one cache, which every worker then starts from,
        // so reuse matches a serial loop's whenever the noise keeps the
        // circuit's multi-qubit topology (the cache's signature).
        run(caches[0], 0, InjectPlanned(layer, plan, rng, exact));
        first = 1;
        warm_hits = caches[0].hits;
        caches[0].hits = 0;
        for (size_t worker = 1; worker < workers; ++worker)
            caches[worker] = caches[0];
    }
    Utils::RunRealizations(
        noise_realizations - first, workers, [&](size_t) { return InjectPlanned(layer, plan, rng, exact); },
        [&](size_t worker, size_t i, RecordCircuitPtr noisy) { run(caches[worker], i + first, noisy); });
    out.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // Summed in realization order, as a serial loop would.
    std::vector<double> sums(count, 0.0);
    out.error_bounds.assign(count, 0.0);
    size_t total_clusters = 0;
    double total_cluster_qubits = 0;
    for (const auto &record : realizations)
    {
        if (record.values.size() != count || record.bounds.size() != count)
            throw std::runtime_error("Estimation returned an invalid expectation vector");
        out.auto_reduced = out.auto_reduced || record.reduced;
        out.qubits_before = std::max(out.qubits_before, record.before);
        out.qubits_after = std::max(out.qubits_after, record.after);
        out.clusters_count = std::max(out.clusters_count, record.groups);
        total_clusters += record.groups;
        const double mean = record.groups ? static_cast<double>(record.cluster_qubits) / record.groups : 0.0;
        total_cluster_qubits += record.groups * mean;
        out.reduction_time += record.reduction_time;
        out.bounded_observables = std::max(out.bounded_observables, record.bounded);
        for (size_t i = 0; i < count; ++i)
        {
            out.error_bounds[i] += record.bounds[i] / noise_realizations;
            sums[i] += record.values[i];
        }
    }
    out.cluster_methods = realizations.back().methods;
    out.mean_cluster_qubits = total_clusters ? total_cluster_qubits / total_clusters : 0;
    out.cone_cache_hits = warm_hits;
    for (const auto &cache : caches)
        out.cone_cache_hits += cache.hits;
    out.values.resize(count);
    for (size_t i = 0; i < count; ++i)
        out.values[i] = sums[i] / noise_realizations;
    out.ideal = EstimateCircuit(WithoutCircuitChannels(clean), observables, config);
    out.ideal_values = out.ideal.values;
    return out;
}

double RealizationFidelity(const RecordCircuitPtr &ideal, const RecordCircuitPtr &noisy, const SimulatorConfig &config)
{
    auto combined = std::make_shared<Circuits::Circuit<double>>();
    // Fidelity is evaluated before terminal readout, as in inner_product.
    for (const auto &op : noisy->GetOperations())
        if (op->GetType() != Circuits::OperationType::kMeasurement)
            combined->AddOperation(op->Clone());
    const auto &ideal_ops = ideal->GetOperations();
    for (auto it = ideal_ops.rbegin(); it != ideal_ops.rend(); ++it)
        if (auto adjoint = adjoint_gate(*it, true))
            combined->AddOperation(adjoint);

    const int num_qubits = std::max(1, static_cast<int>(combined->GetMaxQubitIndex()) + 1);
    GetMaestroObjectWithMute();

    struct Handle
    {
        unsigned long value;

        ~Handle()
        {
            if (value)
                DestroySimpleSimulator(value);
        }
    } handle{CreateSimpleSimulator(num_qubits)};

    if (!handle.value)
        throw std::runtime_error("Failed to create simulator handle.");
    auto network = ConfigureNetwork(handle.value, config);
    if (!network)
        throw std::runtime_error("Failed to configure network.");
    network->CreateSimulator(config.simulator_type, config.simulation_type);
    auto simulator = network->GetSimulator();
    if (!simulator)
        throw std::runtime_error("noisy_fidelity: requested backend is unavailable.");
    if (config.seed)
        simulator->SetSeed(*config.seed);
    Circuits::OperationState state(num_qubits);
    combined->ExecuteBD(simulator, state);
    return simulator->Probability(0);
}

NoisyFidelityResult NoisyFidelity(const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, size_t noise_realizations,
                                  const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    if (!circuit)
        throw std::invalid_argument("Circuit is null.");
    if (noise_realizations < 1)
        throw std::invalid_argument("noise_realizations must be >= 1.");
    const auto ideal = WithoutCircuitChannels(circuit);
    auto rng = MakeNoiseRng(config, noise_seed);
    const bool exact = UsesExactQuantumChannels(config);
    const noise::NoiseInjectionPlan plan(circuit, noise_model);

    struct Realization
    {
        RecordCircuitPtr circuit;
        SimulatorConfig config;
    };

    const size_t qubits = std::max({size_t{1}, circuit->GetMaxQubitIndex() + 1, plan.RegisterWidth()});
    const size_t workers = SamplingWorkers(config, noise_realizations, qubits);
    std::vector<double> fidelities(noise_realizations);
    const auto start = std::chrono::steady_clock::now();
    Utils::RunRealizations(
        noise_realizations, workers,
        [&](size_t r) {
            auto noisy = InjectPlanned(NoiseLayer::kCombined, plan, rng, exact);
            // Reset collapse must use a fresh seed for each realization, drawn
            // after its injection from the same stream.
            auto realization_config = NoiseExecutionConfig(config, noise_seed, r);
            if (!realization_config.seed)
                realization_config.seed = rng();
            return Realization{std::move(noisy), std::move(realization_config)};
        },
        [&](size_t, size_t r, Realization realization) { fidelities[r] = RealizationFidelity(ideal, realization.circuit, realization.config); });

    NoisyFidelityResult out;
    out.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    out.realizations = noise_realizations;
    double sum = 0, squares = 0;
    for (const double fidelity : fidelities)
    {
        sum += fidelity;
        squares += fidelity * fidelity;
    }
    const double n = static_cast<double>(noise_realizations);
    out.fidelity = sum / n;
    const double variance = noise_realizations > 1 ? (squares - n * out.fidelity * out.fidelity) / (n - 1) : 0.0;
    out.std_error = std::sqrt(std::max(variance, 0.0) / n);
    return out;
}
} // namespace MaestroExecution
