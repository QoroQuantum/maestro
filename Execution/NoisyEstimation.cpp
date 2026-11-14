#include "NoisyEstimation.h"

#include "../Utils/Threads/RealizationWorkers.h"
#include "CircuitHelpers.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace MaestroExecution
{
RecordCircuitPtr WithoutCircuitChannels(const RecordCircuitPtr &circuit)
{
    auto out = std::make_shared<Circuits::Circuit<double>>();
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() == Circuits::OperationType::kComposite)
        {
            // Nested channels are noise too; rebuild only circuits that hold one.
            const auto nested = std::static_pointer_cast<Circuits::Circuit<double>>(op);
            out->AddOperation(nested->HasQuantumChannels() ? WithoutCircuitChannels(nested) : op);
        }
        else if (op->GetType() != Circuits::OperationType::kQuantumChannel)
            out->AddOperation(op);
    return out;
}

NoisyEstimate EstimateNoisy(const RecordCircuitPtr &circuit, const std::vector<std::string> &observables, const noise::NoiseModel &noise_model,
                            std::optional<size_t> requested_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed, NoiseLayer layer,
                            std::optional<bool> exact_channels)
{
    if (!circuit)
        throw std::invalid_argument("Circuit is null.");
    auto rng = MakeNoiseRng(config, noise_seed);
    NoisyEstimate out;
    out.policy = ResolveNoiseRealizations(requested_realizations, noise_model, config, layer, 100, circuit, true, exact_channels);
    const size_t noise_realizations = out.policy.realizations;
    out.realizations = noise_realizations;
    if (observables.empty())
    {
        out.ideal = EstimateCircuit(WithoutCircuitChannels(circuit), observables, config);
        out.ideal_values = out.ideal.values;
        out.qubits_before = out.ideal.plan.qubits_before;
        out.qubits_after = out.ideal.plan.qubits_after;
        out.time_taken = out.ideal.time_taken;
        return out;
    }
    const size_t count = observables.size();
    const auto clean = circuit->RemoveEndMeasurements();
    const bool exact = out.policy.exact_channels;
    if (exact && SamplesQuantumChannels(config))
        throw std::invalid_argument("Exact noise requires a density-matrix or MPO backend, or the QCSim Pauli propagator: this backend takes a "
                                    "channel as one sampled trajectory");
    const noise::NoiseInjectionPlan plan(clean, noise_model, InjectsRuntimeChannels(config, exact), OrdersForChain(config, exact));
    // A backend that samples channels draws all the noise as each realization
    // runs (ResolveRuntimeNoise): they then share one circuit and differ by seed.
    RecordCircuitPtr runtime_circuit, first_realization;
    ResolveRuntimeNoise(layer, plan, rng, runtime_circuit, first_realization);
    // Whether the executions draw channel branches, shared circuit or not: the noise seed then joins their seeds.
    RuntimeDraws draws(&plan, runtime_circuit, first_realization);
    // Called once per realization, in order: the one drawn to decide is the first.
    const auto realize = [&]() -> RecordCircuitPtr {
        RecordCircuitPtr noisy = runtime_circuit     ? runtime_circuit
                                 : first_realization ? std::exchange(first_realization, nullptr)
                                                     : InjectPlanned(layer, plan, rng, exact);
        draws.See(noisy);
        return noisy;
    };

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
    // Each realization's register also holds the idle qubits its observables
    // name: the host executor keeps every position of a Pauli string.
    size_t qubits = std::max({size_t{1}, clean->GetMaxQubitIndex() + 1, plan.RegisterWidth()});
    for (const auto &observable : observables)
        qubits = std::max(qubits, observable.size());
    const size_t workers = SamplingWorkers(config, noise_realizations, qubits);
    out.realization_workers = workers;
    // A cache serves one thread (EstimationCache).
    std::vector<EstimationCache> caches(workers);
    const auto run = [&](EstimationCache &cache, size_t r, const RecordCircuitPtr &noisy) {
        auto result = EstimateCircuit(noisy, observables,
                                      draws ? RuntimeNoiseExecutionConfig(config, noise_seed, r) : NoiseExecutionConfig(config, noise_seed, r), &cache);
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
        run(caches[0], 0, realize());
        first = 1;
        warm_hits = caches[0].hits;
        caches[0].hits = 0;
        for (size_t worker = 1; worker < workers; ++worker)
            caches[worker] = caches[0];
    }
    Utils::RunRealizations(
        noise_realizations - first, workers, [&](size_t) { return realize(); },
        [&](size_t worker, size_t i, RecordCircuitPtr noisy) { run(caches[worker], i + first, noisy); });
    out.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    // Summed in realization order, as a serial loop would.
    std::vector<double> sums(count, 0.0), squares(count, 0.0);
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
            squares[i] += record.values[i] * record.values[i];
        }
    }
    out.cluster_methods = realizations.back().methods;
    out.mean_cluster_qubits = total_clusters ? total_cluster_qubits / total_clusters : 0;
    out.cone_cache_hits = warm_hits;
    for (const auto &cache : caches)
        out.cone_cache_hits += cache.hits;
    out.values.resize(count);
    out.standard_errors.resize(count);
    for (size_t i = 0; i < count; ++i)
    {
        out.values[i] = sums[i] / noise_realizations;
        if (noise_realizations > 1)
            out.standard_errors[i] =
                std::sqrt(std::max(0.0, (squares[i] - noise_realizations * out.values[i] * out.values[i]) / (noise_realizations - 1) / noise_realizations));
    }
    const auto ideal = WithoutCircuitChannels(clean);
    // A dynamic baseline must average the same kind of measurement branches as
    // the noisy estimate. Returning one randomly collapsed ideal state makes
    // comparison with the noisy ensemble meaningless.
    const size_t ideal_realizations = HasStochasticEvolution(ideal, UsesExactQuantumChannels(config)) ? noise_realizations : 1;
    std::vector<double> ideal_sums(count, 0.0);
    EstimationCache ideal_cache;
    for (size_t r = 0; r < ideal_realizations; ++r)
    {
        const auto ideal_config = ideal_realizations > 1 ? NoiseExecutionConfig(config, noise_seed, r) : config;
        out.ideal = EstimateCircuit(ideal, observables, ideal_config, &ideal_cache);
        for (size_t i = 0; i < count; ++i)
            ideal_sums[i] += out.ideal.values[i];
    }
    for (size_t i = 0; i < count; ++i)
        out.ideal.values[i] = ideal_sums[i] / ideal_realizations;
    out.ideal_values = out.ideal.values;
    return out;
}

namespace
{
// A configured handle a realization takes over instead of allocating its
// register again: the backend probe of a candidate list. The first realization
// to run takes it, so it never outlives that realization: one of another
// width replaces its register (freeing the probe's) before allocating.
struct SpareRegister
{
    std::atomic<unsigned long> handle{0};

    unsigned long Take()
    {
        return handle.exchange(0);
    }

    ~SpareRegister()
    {
        if (const auto left = handle.exchange(0))
            DestroySimpleSimulator(left);
    }
};

double RealizationFidelityOn(const RecordCircuitPtr &ideal, const RecordCircuitPtr &noisy, const SimulatorConfig &config, SpareRegister *spare)
{
    auto combined = std::make_shared<Circuits::Circuit<double>>();
    // Fidelity is evaluated before terminal readout, as in inner_product.
    // This includes conditional terminal readout. NoisyFidelity validates that
    // no genuinely intermediate measurement remains in the input circuit.
    const auto before_readout = noisy->RemoveEndMeasurements();
    for (const auto &op : before_readout->GetOperations())
        combined->AddOperation(op->Clone());
    const auto &ideal_ops = ideal->GetOperations();
    for (auto it = ideal_ops.rbegin(); it != ideal_ops.rend(); ++it)
        if (auto adjoint = adjoint_gate(*it, true))
            combined->AddOperation(adjoint);

    const int num_qubits = std::max(1, static_cast<int>(combined->GetMaxQubitIndex()) + 1);
    GetMaestroObjectWithMute();
    // The spare was configured for this backend; the seed below replaces the
    // probe's, and a width mismatch replaces its register.
    const unsigned long adopted = spare ? spare->Take() : 0;

    struct Handle
    {
        unsigned long value;

        ~Handle()
        {
            if (value)
                DestroySimpleSimulator(value);
        }
    } handle{adopted ? adopted : CreateSimpleSimulator(num_qubits)};

    if (!handle.value)
        throw std::runtime_error("Failed to create simulator handle.");
    // Exactly the circuit's register: the default allocation also adds the
    // host's network-entangled qubit (2x statevector, 4x density-matrix work).
    const size_t register_qubits = static_cast<size_t>(num_qubits);
    auto network =
        adopted ? static_cast<Maestro *>(GetMaestroObject())->GetSimpleSimulator(handle.value) : ConfigureNetwork(handle.value, config, register_qubits);
    if (!network)
        throw std::runtime_error("Failed to configure network.");
    auto simulator = network->GetSimulator();
    // A fixed backend was already created at this width; replace only an
    // automatic-selection placeholder (or another width), releasing it first
    // so that the two registers never coexist.
    if (!simulator || simulator->GetType() != config.simulator_type || simulator->GetSimulationType() != config.simulation_type ||
        simulator->GetNumberOfQubits() != register_qubits)
    {
        simulator.reset();
        network->CreateSimulator(config.simulator_type, config.simulation_type, register_qubits);
        simulator = network->GetSimulator();
    }
    if (!simulator)
        throw std::runtime_error("noisy_fidelity: requested backend is unavailable.");
    if (config.seed)
        simulator->SetSeed(*config.seed);
    Circuits::OperationState state(num_qubits);
    combined->ExecuteBD(simulator, state);
    return simulator->Probability(0);
}
} // namespace

double RealizationFidelity(const RecordCircuitPtr &ideal, const RecordCircuitPtr &noisy, const SimulatorConfig &config)
{
    return RealizationFidelityOn(ideal, noisy, config, nullptr);
}

bool HasUnitaryIdeal(const RecordCircuitPtr &before_readout)
{
    std::unordered_set<Types::qubit_t> touched;
    return before_readout->HasDeterministicEvolutionWith(Circuits::Circuit<double>::ChannelDraws::kNone, false, true, touched);
}

// The unitary that the ideal circuit applies to a fresh register. A reset
// before any other operation on its wire (all NoisyFidelity admits) prepares
// its target from |0>: nothing for |0>, an X for |1>. The adjoint of that X
// then undoes the preparation in RealizationFidelity.
RecordCircuitPtr PreparedUnitary(const RecordCircuitPtr &circuit)
{
    auto out = std::make_shared<Circuits::Circuit<double>>();
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() == Circuits::OperationType::kReset)
        {
            const auto reset = std::static_pointer_cast<Circuits::Reset<double>>(op);
            const auto &qubits = reset->GetQubits();
            const auto &targets = reset->GetResetTargets();
            for (size_t i = 0; i < qubits.size() && i < targets.size(); ++i)
                if (targets[i])
                    out->AddOperation(std::make_shared<Circuits::XGate<double>>(qubits[i]));
        }
        else if (op->GetType() == Circuits::OperationType::kComposite)
            out->AddOperation(PreparedUnitary(std::static_pointer_cast<Circuits::Circuit<double>>(op)));
        else
            out->AddOperation(op);
    return out;
}

NoisyFidelityResult NoisyFidelity(const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, std::optional<size_t> requested_realizations,
                                  const SimulatorConfig &requested_config, std::optional<uint64_t> noise_seed)
{
    if (!circuit)
        throw std::invalid_argument("Circuit is null.");
    // Inspect measurements before stripping channels: a measurement before an
    // explicit channel is not a terminal readout even if the ideal omits it.
    // Gates, delays, barriers, channels (noise, absent from the ideal) and
    // nested circuits of them are admitted, as are resets that precede every
    // other operation on their wire: on the fresh register they prepare |0> or
    // |1>, which the ideal state includes (PreparedUnitary).
    const auto before_readout = circuit->RemoveEndMeasurements();
    if (!HasUnitaryIdeal(before_readout))
        throw std::invalid_argument("noisy_fidelity requires a unitary ideal circuit; mid-circuit measurements, resets after other operations "
                                    "on their qubit and classical control are unsupported.");
    const auto ideal = PreparedUnitary(WithoutCircuitChannels(circuit));
    noise::NoiseInjectionPlan plan(circuit, noise_model);
    // The register each realization evaluates: fidelity drops terminal
    // readout, so a measured qubit no gate touches is not part of it, while a
    // crosstalk spectator of the noise is (RealizationFidelity).
    size_t qubits = std::max<size_t>(1, before_readout->GetMaxQubitIndex() + 1);
    for (const auto spectator : plan.Spectators())
        qubits = std::max(qubits, spectator + 1);
    // Each realization creates its simulator explicitly: policy, allocation and
    // workers must all describe the one backend a candidate list resolves to,
    // tested on that register. Probing several candidates allocates the
    // winner's register; the first realization takes it over.
    SpareRegister spare;
    unsigned long probed = 0;
    const auto config = ResolveCandidateBackend(requested_config, qubits, probed);
    spare.handle = probed;
    const auto policy = ResolveNoiseRealizations(requested_realizations, noise_model, config, NoiseLayer::kCombined, 100, circuit, true);
    const size_t noise_realizations = policy.realizations;
    auto rng = MakeNoiseRng(config, noise_seed);
    const bool exact = policy.exact_channels;
    // A backend that samples channels from its state draws their branches as
    // each realization runs: its fidelity is that of one trajectory, and the
    // mean over realizations that of the channels, with nothing approximated.
    plan.KeepRuntimeChannels(policy.runtime_channels, policy.chain_order);
    // When it draws all the noise that way (ResolveRuntimeNoise), the realizations share one circuit and differ by
    // seed. Whenever a realization keeps channels that are drawn, shared circuit or not (RuntimeDraws), the noise
    // seed has no injection to drive for them and joins the seed of each execution (RuntimeNoiseExecutionConfig),
    // as in EstimateNoisy: either seed alone still changes the errors drawn.
    RecordCircuitPtr runtime_circuit, first_realization;
    ResolveRuntimeNoise(NoiseLayer::kCombined, plan, rng, runtime_circuit, first_realization);
    RuntimeDraws draws(&plan, runtime_circuit, first_realization);

    struct Realization
    {
        RecordCircuitPtr circuit;
        SimulatorConfig config;
    };

    const size_t workers = SamplingWorkers(config, noise_realizations, qubits);
    std::vector<double> fidelities(noise_realizations);
    const auto start = std::chrono::steady_clock::now();
    Utils::RunRealizations(
        noise_realizations, workers,
        [&](size_t r) {
            // Called once per realization, in order: the one drawn to decide is the first.
            auto noisy = runtime_circuit     ? runtime_circuit
                         : first_realization ? std::exchange(first_realization, nullptr)
                                             : InjectPlanned(NoiseLayer::kCombined, plan, rng, exact);
            draws.See(noisy);
            // Reset collapse must use a fresh seed for each realization, drawn
            // after its injection from the same stream.
            auto realization_config = draws ? RuntimeNoiseExecutionConfig(config, noise_seed, r) : NoiseExecutionConfig(config, noise_seed, r);
            if (!realization_config.seed)
                realization_config.seed = rng();
            return Realization{std::move(noisy), std::move(realization_config)};
        },
        [&](size_t, size_t r, Realization realization) { fidelities[r] = RealizationFidelityOn(ideal, realization.circuit, realization.config, &spare); });

    NoisyFidelityResult out;
    out.policy = policy;
    out.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    out.realizations = noise_realizations;
    out.realization_workers = workers;
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
