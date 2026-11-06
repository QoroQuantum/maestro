#include "SampleRecords.h"
#include "TerminalSampling.h"

#include "../Utils/Threads/RealizationWorkers.h"

#include <chrono>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace MaestroExecution
{
namespace
{
// A worker's simulator for direct rows: one register, allocated exactly once,
// on which every realization runs without per-realization network setup.
class RawSamplingSession
{
  public:
    // qubits is the register a circuit may touch, including noise spectators.
    RawSamplingSession(size_t qubits, const SimulatorConfig &config, bool multithreading)
        : registerQubits(std::max<size_t>(1, qubits)), optimizeCircuit(config.optimize_circuit && MergingChangesResults(config.simulation_type))
    {
        if (registerQubits > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Circuit width exceeds the simulator limit");
        GetMaestroObjectWithMute();
        handle = CreateSimpleSimulator(static_cast<int>(registerQubits));
        if (!handle)
            throw std::runtime_error("Failed to create sampling simulator");
        try
        {
            auto fixed = config;
            fixed.fixed_backend = true;
            // A default network simulator also allocates each host's entangled
            // qubit: one idle qubit doubles statevector and quadruples density-
            // matrix work. Allocate the circuit's register exactly, once.
            auto network = ConfigureNetwork(handle, fixed, registerQubits);
            if (!network || !(simulator = network->GetSimulator()))
                throw std::runtime_error("Requested sampling backend is unavailable");
            simulator->SetMultithreading(multithreading);
        }
        catch (...)
        {
            DestroySimpleSimulator(handle);
            throw;
        }
    }

    ~RawSamplingSession()
    {
        // The network owns the simulator; release ours before the handle.
        simulator.reset();
        if (handle)
            DestroySimpleSimulator(handle);
    }

    RawSamplingSession(const RawSamplingSession &) = delete;
    RawSamplingSession &operator=(const RawSamplingSession &) = delete;

    void Sample(const RecordCircuitPtr &circuit, size_t shots, std::optional<uint64_t> seed, uint8_t *output, size_t width)
    {
        if (seed)
            simulator->SetSeed(*seed);
        // Classical Random operations own mutable generators, including nested
        // and conditional operations. Never mutate generators in the input.
        auto execution_circuit = CloneSamplingCircuitForExecution(circuit, seed);
        // The network executor runs an optimized copy of every circuit. Merging
        // gates can change more than rounding (MergingChangesResults); there,
        // run the same circuit. Optimize replaces operations rather than
        // editing them and never enters nested circuits, so a new container
        // suffices to leave the caller's circuit unchanged.
        if (optimizeCircuit)
        {
            auto optimized = std::make_shared<Circuits::Circuit<double>>(execution_circuit->GetOperations());
            optimized->Optimize();
            execution_circuit = std::move(optimized);
        }
        // Nested circuits share the enclosing classical state, so inlining them
        // changes no result; their terminal measurements and gates then take
        // part in terminal sampling and prefix sharing (after optimization,
        // which never entered them).
        if (execution_circuit->HasCompositeOperations())
            execution_circuit = execution_circuit->InlinedComposites();
        if (!execution_circuit->GetOperations().empty() && execution_circuit->GetMaxQubitIndex() >= registerQubits)
            throw std::runtime_error("Sampling circuit touches qubit " + std::to_string(execution_circuit->GetMaxQubitIndex()) + " outside its " +
                                     std::to_string(registerQubits) + "-qubit register");
        Circuits::OperationState state(width);
        // A single shot keeps its trajectory, so one-shot noisy realizations
        // retain their seeded rows.
        if (shots > 1 && CanSampleTerminal(*execution_circuit, *simulator, width, true))
        {
            simulator->Reset();
            SamplePreparedTerminal(
                *execution_circuit, simulator, shots, state, seed,
                [&](size_t row, const Circuits::OperationState &sample) {
                    const auto &bits = sample.GetAllBits();
                    for (size_t bit = 0; bit < width; ++bit)
                        output[row * width + bit] = bit < bits.size() && bits[bit];
                },
                &maxBondDimension, true);
            ++terminalCalls;
            return;
        }
        // Save only deterministic quantum work. On pure-state backends resets,
        // including nested ones, can consume measurement randomness even when
        // their outcome is known; keep the legacy replay stream for those
        // circuits. Gate-only nested circuits are prefix work like gates.
        bool reusePrefix = shots > 1 && (simulator->SupportsQuantumChannels() ||
                                         std::none_of(execution_circuit->GetOperations().begin(), execution_circuit->GetOperations().end(), [](const auto &op) {
                                             return op->GetType() == Circuits::OperationType::kReset ||
                                                    (op->GetType() == Circuits::OperationType::kComposite &&
                                                     !std::static_pointer_cast<Circuits::Circuit<double>>(op)->HasOnlyDeterministicOperations(false));
                                         }));
        auto suffix = execution_circuit;
        if (reusePrefix)
        {
            simulator->Reset();
            simulator->SetGatesCounter(0);
            auto executed = execution_circuit->ExecuteNonMeasurements(simulator, state, &maxBondDimension);
            const size_t prefix_operations =
                execution_circuit->GetOperations().size() - executed.size() + static_cast<size_t>(std::count(executed.begin(), executed.end(), true));
            if (prefix_operations == 0)
                reusePrefix = false; // nothing to share: a saved |0> only adds a state copy per shot
            else
            {
                suffix = execution_circuit->RemoveExecutedOperations(executed);
                simulator->SaveState();
            }
        }
        for (size_t shot = 0; shot < shots; ++shot)
        {
            if (reusePrefix)
                simulator->RestoreState();
            else
                simulator->Reset();
            simulator->SetGatesCounter(0);
            state.Reset();
            suffix->ExecuteBD(simulator, state, &maxBondDimension);
            const auto &bits = state.GetAllBits();
            for (size_t bit = 0; bit < width; ++bit)
                output[shot * width + bit] = bit < bits.size() && bits[bit];
        }
        ++trajectoryCalls;
    }

    size_t MaxBondDimension() const
    {
        return maxBondDimension;
    }

    // Calls sampled from one prepared state, and calls replayed shot by shot.
    size_t TerminalCalls() const
    {
        return terminalCalls;
    }

    size_t TrajectoryCalls() const
    {
        return trajectoryCalls;
    }

    const std::shared_ptr<Simulators::ISimulator> &Simulator() const
    {
        return simulator;
    }

  private:
    // Circuit optimization merges and cancels gates. On the exact dense
    // methods that changes only floating-point rounding, so their direct
    // sessions skip its cost (about 40 us per realization of a 400-gate
    // circuit). Elsewhere it changes results or admissibility: path-integral
    // pruning and MPS/MPO truncation act per gate, extended-stabilizer cost
    // follows the T count, and T;T becomes a Clifford S the stabilizer accepts.
    static bool MergingChangesResults(Simulators::SimulationType method)
    {
        return method != Simulators::SimulationType::kStatevector && method != Simulators::SimulationType::kDensityMatrix;
    }

    unsigned long handle = 0;
    size_t registerQubits;
    bool optimizeCircuit;
    std::shared_ptr<Simulators::ISimulator> simulator;
    size_t maxBondDimension = 0;
    size_t terminalCalls = 0;
    size_t trajectoryCalls = 0;
};

// Expand a stable outcome order into independently permuted complete rows.
// This preserves the sampled multiset and correlations without allocating a
// second matrix. Each prefix is a uniform subsample of this realization.
void CopyCountSamples(const SamplingResult &result, uint8_t *output, size_t shots, size_t width, std::optional<uint64_t> seed)
{
    const auto order = SampleRowOrder::Permutation(shots, seed, SampleRowOrder::kHistogramStream);
    using Outcome = Network::INetwork<double>::ExecuteResults::value_type;
    std::vector<const Outcome *> outcomes;
    outcomes.reserve(result.counts.size());
    for (const auto &outcome : result.counts)
        outcomes.push_back(&outcome);
    std::sort(outcomes.begin(), outcomes.end(), [](const Outcome *a, const Outcome *b) { return a->first < b->first; });
    size_t row = 0;
    for (const auto *outcome : outcomes)
        for (size_t copy = 0; copy < outcome->second; ++copy)
        {
            if (row >= shots)
                throw std::runtime_error("Sampling returned too many shots");
            const auto &bits = outcome->first;
            for (size_t bit = 0; bit < width; ++bit)
                output[order[row] * width + bit] = bit < bits.size() && bits[bit];
            ++row;
        }
    if (row != shots)
        throw std::runtime_error("Sampling returned an incomplete shot matrix");
}

void RecordFusion(ShotRecords &records, const Simulators::ISimulator &simulator)
{
    records.gate_fusion = records.gate_fusion || simulator.IsGateFusionEnabled();
    records.gate_fusion_max_qubits = std::max<size_t>(records.gate_fusion_max_qubits, simulator.GetGateFusionMaxQubits());
}

void RecordFusion(ShotRecords &records, const SamplingResult &result)
{
    records.gate_fusion = records.gate_fusion || result.gate_fusion;
    records.gate_fusion_max_qubits = std::max(records.gate_fusion_max_qubits, result.gate_fusion_max_qubits);
    if (result.gpu_device >= 0)
        records.gpu_device = result.gpu_device;
}
} // namespace

ShotRecords SampleCircuitRows(const RecordCircuitPtr &circuit, const SimulatorConfig &config, size_t shots)
{
    if (!circuit)
        throw std::invalid_argument("Sampling requires a non-null circuit");
    config.Validate();
    ShotRecords records;
    records.shots = shots;
    records.width = ClassicalWidth(*circuit);
    SampleStorageSize(shots, records.width);
    bool reference = false, terminal = false;
    records.simulator = config.simulator_type;
    records.method = config.simulation_type;
    records.components = 1;
    records.largest_component = std::max(size_t{1}, circuit->GetMaxQubitIndex() + 1);
    const auto start = std::chrono::steady_clock::now();
    // Sampling reduction keeps the network executor, as for counts: it splits
    // disjoint measurement cones.
    if (AdmitsReferenceFrames(config) && !config.enable_sampling_reduction)
    {
        if (auto plan = CompileReferenceSampler(circuit, std::max(size_t{1}, circuit->GetMaxQubitIndex() + 1), records.width))
        {
            records.rows = plan->Sample(shots, config.seed ? *config.seed : Utils::RandomStream::FreshSeed());
            reference = true;
        }
    }
    if (!reference && shots)
    {
        records.rows.resize(SampleStorageSize(shots, records.width));
        if (SupportsDirectSamples(config))
        {
            // Simulate only the qubits the circuit touches, as the network executor does.
            const QubitCompaction compaction(*circuit);
            RawSamplingSession session(compaction.Width(), config, true);
            session.Sample(compaction.Apply(circuit), shots, config.seed, records.rows.data(), records.width);
            records.max_bond_dimension = session.MaxBondDimension();
            terminal = session.TerminalCalls() != 0;
            RecordFusion(records, *session.Simulator());
        }
        else
        {
            // One simulator, so seeded rows do not depend on max_simulators;
            // terminal measurements are sampled from one prepared state.
            auto single = config;
            single.max_simulators = 1;
            const auto result = SampleCircuit(circuit, single, shots);
            CopyCountSamples(result, records.rows.data(), shots, records.width, config.seed);
            records.max_bond_dimension = result.max_bond_dimension;
            terminal = shots > 1 && !circuit->HasOpsAfterMeasurements(UsesExactQuantumChannels(config));
            RecordFusion(records, result);
            // Automatic selection may have chosen another backend.
            records.simulator = result.simulator;
            records.method = result.method;
            records.components = result.components;
            records.largest_component = result.largest_component;
        }
    }
    if (reference)
        std::tie(records.simulator, records.method) = std::pair{Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer};
    records.sampling_strategy = reference ? "reference_frame_v1" : terminal ? "terminal_v1" : "trajectory_v1";
    records.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return records;
}

ShotRecords ExecuteNoisy(const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, size_t shots,
                         std::optional<size_t> noise_realizations, std::optional<uint64_t> noise_seed, const NoisyExecutionOptions &options)
{
    if (!circuit)
        throw std::invalid_argument("Circuit is null.");
    if (shots < 1)
        throw std::invalid_argument("shots must be >= 1.");
    config.Validate();
    const bool return_samples = options.rows;
    const bool exact = options.exact_channels.value_or(UsesExactQuantumChannels(config));
    const NoiseLayer layer = options.layer;
    auto rng = MakeNoiseRng(config, noise_seed);
    auto policy = ResolveNoiseRealizations(noise_realizations, noise_model, config, layer, 64, circuit, false, exact);
    const size_t batches = std::min(shots, policy.realizations);
    policy.realizations = batches;
    const size_t base_batch = shots / batches;
    const size_t leftover = shots % batches;

    const size_t qubits = std::max(size_t{1}, circuit->GetMaxQubitIndex() + 1);
    const size_t width = ClassicalWidth(*circuit);
    // Unseeded one-shot realizations may skip per-realization network setup;
    // seeded and shared-realization counts keep their existing executor. So do
    // circuits without measurements: the network keys their records by its own
    // rule, not by the padded classical width used here.
    const bool direct_counts = !return_samples && batches == shots && PreferDirectTrajectoryCounts(config, noise_seed) && HasMeasurements(circuit);
    const bool direct = return_samples || direct_counts;
    const bool direct_sessions = direct && SupportsDirectSamples(config);
    const uint64_t unseeded_measurements = direct && !config.seed && !noise_seed ? Utils::RandomStream::FreshSeed() : 0;
    // Direct sessions simulate only the qubits the circuit touches, compacted
    // in increasing order as the network executor does (QubitCompaction).
    // Crosstalk can rotate spectators the circuit never touches, at any index:
    // they stay in |0>, so their rotations are a global phase and the direct
    // sessions drop them (drop_spectator_phases_if) instead of allocating
    // them. Frames never apply here: they reject crosstalk.
    //
    // The network executor keeps the spectators. As in 0.3.13, its host spans
    // the noisy register (RegisterWidth) and its count keys are padded to it;
    // direct counts pad theirs the same way. It compacts each realization to
    // the circuit's qubits plus the untouched spectators, which is what its
    // workers' memory budget counts.
    const QubitCompaction compaction(*circuit);
    std::unique_ptr<noise::NoiseInjectionPlan> plan;
    bool trim_spectators = false;
    size_t host_qubits = qubits, untouched_spectators = 0;
    if (noise_model.has_crosstalk())
    {
        plan = std::make_unique<noise::NoiseInjectionPlan>(circuit, noise_model);
        trim_spectators = compaction.Sparse() || plan->RegisterWidth() > qubits;
        host_qubits = std::max(qubits, plan->RegisterWidth());
        for (const auto spectator : plan->Spectators())
            untouched_spectators += !compaction.Contains(spectator);
    }
    const size_t count_width = std::max(width, host_qubits);
    // The ordinary executor cannot create a host beyond the simulator's
    // register limit (0.3.13 failed there too). Padded count keys would be as
    // wide; raw samples keep the circuit's classical width and still run.
    if (!return_samples && host_qubits > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Circuit width exceeds the simulator limit: a crosstalk spectator lies beyond it; use return_samples=True");
    const size_t budget_qubits = compaction.Width() + (direct_sessions ? 0 : untouched_spectators);
    // Seeded realizations with several shots keep the executor's own shot
    // partition, which sets their sequence. Unseeded ones that it samples from
    // one prepared state (terminal measurements on a dense method) run
    // single-threaded there anyway, so realization workers only add parallelism.
    using Method = Simulators::SimulationType;
    const bool prepared_batches = !config.seed && !noise_seed && !circuit->HasOpsAfterMeasurements(UsesExactQuantumChannels(config)) &&
                                  (config.simulation_type == Method::kStatevector || config.simulation_type == Method::kDensityMatrix ||
                                   config.simulation_type == Method::kMatrixProductOperator || config.simulation_type == Method::kPathIntegral);
    const size_t workers = (return_samples || batches == shots || prepared_batches) ? SamplingWorkers(config, batches, budget_qubits) : 1;
    std::vector<SamplingResult> results(workers);
    std::vector<std::unique_ptr<SamplingSession>> sessions(workers);
    std::vector<std::unique_ptr<RawSamplingSession>> raw_sessions(workers);
    std::vector<std::vector<uint8_t>> worker_rows(workers);
    // Direct counts key records packed 64 bits per word (FrameDetail::PackedRowHash).
    using PackedCounts = std::unordered_map<std::vector<uint64_t>, size_t, FrameDetail::PackedRowHash>;
    std::vector<PackedCounts> packed_counts(workers);
    std::vector<size_t> network_terminal(workers), network_trajectory(workers);
    // The backend the network executor ran each worker's realizations on:
    // automatic selection may choose another than the requested one.
    using Backend = std::pair<Simulators::SimulatorType, Simulators::SimulationType>;

    struct ExecutedBackend
    {
        std::optional<Backend> backend;
        bool mixed = false;
    };

    std::vector<ExecutedBackend> executed(workers);
    ShotRecords records;
    records.policy = policy;
    records.shots = shots;
    records.width = width;
    FrameOutcomes frame_counts;
    bool reference = false;
    if (return_samples)
        SampleStorageSize(shots, width);
    const auto start = std::chrono::steady_clock::now();
    // This opt-in policy represents independent Pauli errors per shot. A
    // shared-noise realization (multiple shots) must keep its old meaning.
    // Sampling reduction keeps the network executor, as for counts.
    if ((return_samples || PreferFrameCounts(config, noise_seed)) && batches == shots && AdmitsReferenceFrames(config) && !config.enable_sampling_reduction &&
        noise_model.inject_noise_covers_model() && layer != NoiseLayer::kCoherent)
    {
        auto channels = noise::inject_exact_noise(circuit, noise_model);
        noise::attach_readout_error(channels, noise_model);
        if (auto frames = CompileReferenceSampler(channels, qubits, width))
        {
            const auto measurement_seed = config.seed ? *config.seed : noise_seed ? *noise_seed : Utils::RandomStream::FreshSeed();
            const auto frame_seed = Simulators::IState::DeriveSeed(measurement_seed, noise_seed.value_or(measurement_seed));
            if (return_samples)
                records.rows = frames->Sample(shots, frame_seed);
            else
                frame_counts = frames->SampleCounts(shots, frame_seed);
            reference = true;
        }
    }
    if (!reference)
    {
        if (return_samples)
            records.rows.resize(SampleStorageSize(shots, width));
        uint8_t *samples = records.rows.data();
        // Resolved once; each realization draws from the noise RNG in order.
        if (!plan)
            plan = std::make_unique<noise::NoiseInjectionPlan>(circuit, noise_model);
        Utils::RunRealizations(
            batches, workers, [&](size_t) { return InjectPlanned(layer, *plan, rng, exact); },
            [&](size_t worker, size_t batch, RecordCircuitPtr noisy) {
                const size_t count = base_batch + (batch < leftover ? 1 : 0);
                const size_t first = batch * base_batch + std::min(batch, leftover);
                auto execution_config = NoiseExecutionConfig(config, noise_seed, batch);
                // Only the complete request may select a new count policy.
                // Injected shared realizations retain their existing sequence.
                execution_config.counts_policy = "legacy";
                if (direct && !execution_config.seed)
                    execution_config.seed = Simulators::IState::DeriveSeed(unseeded_measurements, batch);
                // Raw rows on the network executor (MPS, MPO, other backends) run
                // each realization on one simulator: its own shot workers would
                // make seeded rows depend on max_simulators. Realization workers
                // supply the parallelism instead.
                if (return_samples && !direct_sessions)
                    execution_config.max_simulators = 1;
                if (direct_sessions)
                {
                    if (trim_spectators)
                        noisy = noise::drop_spectator_phases_if(
                            noisy, [&compaction](Types::qubit_t q) { return compaction.Contains(q); },
                            [](Types::qubit_t q) { return "Noise acts on qubit " + std::to_string(q) + ", which the circuit does not use"; });
                    noisy = compaction.Apply(noisy);
                    if (!raw_sessions[worker])
                        raw_sessions[worker] = std::make_unique<RawSamplingSession>(compaction.Width(), execution_config, workers == 1);
                    if (return_samples)
                        raw_sessions[worker]->Sample(noisy, count, execution_config.seed, samples + first * width, width);
                    else
                    {
                        auto &rows = worker_rows[worker];
                        rows.resize(count * width);
                        raw_sessions[worker]->Sample(noisy, count, execution_config.seed, rows.data(), width);
                        std::vector<uint64_t> key((width + 63) / 64);
                        for (size_t shot = 0; shot < count; ++shot)
                        {
                            std::fill(key.begin(), key.end(), 0);
                            for (size_t bit = 0; bit < width; ++bit)
                                key[bit / 64] |= static_cast<uint64_t>(rows[shot * width + bit] != 0) << (bit % 64);
                            ++packed_counts[worker][key];
                        }
                        auto &local = results[worker];
                        local.components = 1;
                        local.largest_component = qubits;
                    }
                }
                else
                {
                    SamplingResult result;
                    if (batches == shots && config.simulator_type == Simulators::SimulatorType::kQCSim && !config.enable_sampling_reduction)
                    {
                        if (!sessions[worker])
                            sessions[worker] = std::make_unique<SamplingSession>(host_qubits, execution_config);
                        result = sessions[worker]->Sample(noisy, count, execution_config.seed);
                    }
                    else
                        result = SampleCircuit(noisy, execution_config, count);
                    if (return_samples)
                    {
                        CopyCountSamples(result, samples + first * width, count, width, execution_config.seed);
                        // The network samples terminal measurements from one prepared state.
                        ++(count > 1 && !noisy->HasOpsAfterMeasurements(UsesExactQuantumChannels(execution_config)) ? network_terminal
                                                                                                                    : network_trajectory)[worker];
                    }
                    else
                        for (const auto &[bits, n] : result.counts)
                            results[worker].counts[bits] += n;
                    auto &seen = executed[worker];
                    const Backend ran{result.simulator, result.method};
                    seen.mixed = seen.mixed || (seen.backend && *seen.backend != ran);
                    seen.backend = ran;
                    auto &local = results[worker];
                    local.components = std::max(local.components, result.components);
                    local.largest_component = std::max(local.largest_component, result.largest_component);
                    local.max_bond_dimension = std::max(local.max_bond_dimension, result.max_bond_dimension);
                    local.gate_fusion = local.gate_fusion || result.gate_fusion;
                    local.gate_fusion_max_qubits = std::max(local.gate_fusion_max_qubits, result.gate_fusion_max_qubits);
                    if (result.gpu_device >= 0)
                        local.gpu_device = result.gpu_device;
                }
            });
    }
    records.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    for (const auto &session : raw_sessions)
        if (session)
            RecordFusion(records, *session->Simulator());
    for (const auto &result : results)
        RecordFusion(records, result);

    records.simulator = config.simulator_type;
    records.method = config.simulation_type;
    std::optional<Backend> ran;
    bool mixed = false;
    for (const auto &seen : executed)
        if (seen.backend)
        {
            mixed = mixed || seen.mixed || (ran && *ran != *seen.backend);
            ran = seen.backend;
        }
    // Report a selected backend only when every realization ran on it.
    if (ran && !mixed)
        std::tie(records.simulator, records.method) = *ran;
    if (reference)
        std::tie(records.simulator, records.method) = std::pair{Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStabilizer};
    records.realizations = batches;
    if (return_samples)
    {
        records.components = 1;
        records.largest_component = qubits;
        // Realizations on the network executor report its partition: sampling
        // reduction can split them into disjoint measurement cones.
        if (!reference && !direct_sessions)
        {
            records.components = 0;
            records.largest_component = 0;
            for (const auto &result : results)
            {
                records.components = std::max(records.components, result.components);
                records.largest_component = std::max(records.largest_component, result.largest_component);
            }
            records.components = std::max<size_t>(1, records.components);
            if (!records.largest_component)
                records.largest_component = qubits;
        }
        size_t terminal_calls = 0, trajectory_calls = 0;
        for (const auto &session : raw_sessions)
            if (session)
            {
                records.max_bond_dimension = std::max(records.max_bond_dimension, session->MaxBondDimension());
                terminal_calls += session->TerminalCalls();
                trajectory_calls += session->TrajectoryCalls();
            }
        // Methods without a direct session (MPS, MPO) ran on the network executor.
        for (const auto &result : results)
            records.max_bond_dimension = std::max(records.max_bond_dimension, result.max_bond_dimension);
        for (size_t worker = 0; worker < workers; ++worker)
        {
            terminal_calls += network_terminal[worker];
            trajectory_calls += network_trajectory[worker];
        }
        // terminal_v1 only when every realization was drawn from one prepared
        // state; mixed_v1 when some were (e.g. uneven splits leave one-shot
        // realizations, which keep their trajectory), as checkpoints report.
        records.sampling_strategy = reference                             ? "reference_frame_v1"
                                    : terminal_calls && !trajectory_calls ? "terminal_v1"
                                    : terminal_calls                      ? "mixed_v1"
                                                                          : "trajectory_v1";
        return records;
    }

    Network::INetwork<double>::ExecuteResults combined;
    for (const auto &result : results)
    {
        for (const auto &[bits, n] : result.counts)
            combined[bits] += n;
        records.components = std::max(records.components, result.components);
        records.largest_component = std::max(records.largest_component, result.largest_component);
        records.max_bond_dimension = std::max(records.max_bond_dimension, result.max_bond_dimension);
    }
    for (const auto &session : raw_sessions)
        if (session)
            records.max_bond_dimension = std::max(records.max_bond_dimension, session->MaxBondDimension());
    if (reference)
    {
        records.components = 1;
        records.largest_component = qubits;
        records.sampling_strategy = "reference_frame_v1";
    }
    else if (direct_counts)
        records.sampling_strategy = "trajectory_v1";
    // Exactly one source holds counts: frame outcomes (already distinct),
    // direct packed records (keys padded to count_width), or the network
    // executor's histograms.
    for (size_t worker = 1; worker < packed_counts.size(); ++worker)
        for (const auto &[row, count] : packed_counts[worker])
            packed_counts[0][row] += count;
    records.counts.reserve(frame_counts.size() + (packed_counts.empty() ? 0 : packed_counts[0].size()) + combined.size());
    for (auto &entry : frame_counts)
        records.counts.push_back(std::move(entry));
    if (!packed_counts.empty())
        for (const auto &[row, count] : packed_counts[0])
        {
            std::vector<bool> bits(count_width, false);
            for (size_t bit = 0; bit < width; ++bit)
                bits[bit] = (row[bit / 64] >> (bit % 64)) & 1U;
            records.counts.emplace_back(std::move(bits), count);
        }
    for (const auto &[bits, count] : combined)
        records.counts.emplace_back(bits, count);
    return records;
}
} // namespace MaestroExecution
