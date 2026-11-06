#include "PrefixCheckpoint.h"
#include "TerminalSampling.h"

#include "../Utils/Threads/RealizationWorkers.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace MaestroExecution
{
PrefixCheckpoint::PrefixCheckpoint(const RecordCircuitPtr &prefix, int num_qubits, const SimulatorConfig &config, const noise::NoiseModel *prefix_noise)
    : num_qubits_(std::max(1, num_qubits)), config_(config)
{
    if (prefix && num_qubits <= 0)
        num_qubits_ = std::max(1, static_cast<int>(prefix->GetMaxQubitIndex()) + 1);
    // The simulator is created explicitly below: resolve a candidate list to the
    // one backend it creates, so suffix noise policy and metadata describe it.
    config_ = ResolveCandidateBackend(config, static_cast<size_t>(num_qubits_));
    auto executed_prefix = prefix;
    if (prefix && prefix_noise && prefix_noise->has_any())
    {
        const auto policy = ResolveNoiseRealizations(std::nullopt, *prefix_noise, config_, NoiseLayer::kCombined, 64, prefix);
        if (!policy.exact_channels || policy.stochastic_noise)
            throw std::invalid_argument("Prefix noise must be deterministic: exact channels on a density-matrix or MPO backend, "
                                        "without coherent or correlated noise");
        std::optional<uint64_t> seed = 0; // no sampled layer reads it
        auto rng = MakeNoiseRng(config_, seed);
        const noise::NoiseInjectionPlan plan(prefix, *prefix_noise);
        executed_prefix = InjectPlanned(NoiseLayer::kCombined, plan, rng, true);
        // As for suffixes: crosstalk spectators beyond a prefix that fits the
        // register stay in |0>, where their rotations are a global phase.
        const size_t register_qubits = static_cast<size_t>(num_qubits_);
        if (plan.RegisterWidth() > register_qubits && (prefix->GetOperations().empty() || prefix->GetMaxQubitIndex() < register_qubits))
            executed_prefix = noise::drop_outside_spectator_phases(executed_prefix, register_qubits);
    }
    GetMaestroObjectWithMute();
    handle_ = CreateSimpleSimulator(num_qubits_);
    if (handle_ == 0)
        throw std::runtime_error("PrefixCheckpointedSimulator: failed to create simulator handle.");
    try
    {
        // Exactly the requested register, from the first allocation: the default
        // also adds the host's network-entangled qubit (2x statevector, 4x
        // density-matrix work), which may not even fit.
        const size_t register_qubits = static_cast<size_t>(num_qubits_);
        network_ = ConfigureNetwork(handle_, config_, register_qubits);
        if (!network_)
            throw std::runtime_error("PrefixCheckpointedSimulator: failed to configure network.");
        // A fixed backend already has this register; replace a placeholder.
        const auto created = network_->GetSimulator();
        if (!created || created->GetType() != config_.simulator_type || created->GetSimulationType() != config_.simulation_type ||
            created->GetNumberOfQubits() != register_qubits)
            network_->CreateSimulator(config_.simulator_type, config_.simulation_type, register_qubits);
        simulator_ = network_->GetSimulator();
        if (!simulator_)
            throw std::runtime_error("PrefixCheckpointedSimulator: requested simulator/simulation type is not available.");
        network_->GetController()->SetOptimizeCircuit(false);
        network_->SetInitialQubitsMapOptimization(false);
        network_->SetMPSOptimizeSwaps(false);
        if (config_.seed)
            simulator_->SetSeed(*config_.seed);
        Circuits::OperationState state(num_qubits_);
        if (executed_prefix && !executed_prefix->GetOperations().empty())
            executed_prefix->ExecuteBD(simulator_, state, &max_bond_dim_);
        simulator_->SaveState();
    }
    catch (...)
    {
        simulator_.reset();
        network_.reset();
        DestroySimpleSimulator(handle_);
        throw;
    }
}

PrefixCheckpoint::~PrefixCheckpoint()
{
    simulator_.reset();
    network_.reset();
    if (handle_)
        DestroySimpleSimulator(handle_);
}

bool PrefixCheckpoint::HasNoise(const RecordCircuitPtr &suffix, const noise::NoiseModel *noise_model)
{
    if (noise_model && noise_model->has_any())
        return true;
    // Circuit-level channels, including nested ones, need per-realization
    // injection like a model does.
    return suffix && suffix->HasQuantumChannels();
}

ShotRecords PrefixCheckpoint::ExecuteSuffix(const RecordCircuitPtr &suffix, size_t shots, const noise::NoiseModel *noise_model,
                                            std::optional<size_t> noise_realizations, std::optional<uint64_t> noise_seed, size_t num_measurements, bool rows)
{
    if (!suffix)
        throw std::invalid_argument("suffix_circuit is null.");
    if (shots < 1)
        throw std::invalid_argument("shots must be >= 1.");

    size_t total_cbits = std::max(static_cast<size_t>(num_qubits_), num_measurements);
    const auto cbits = suffix->GetBits();
    if (!cbits.empty())
        total_cbits = std::max(total_cbits, *cbits.rbegin() + 1);

    static const noise::NoiseModel no_noise;
    const bool has_noise = HasNoise(suffix, noise_model);
    if (!noise_model)
        noise_model = &no_noise;
    // The suffix continues the saved prefix state, not a fresh register.
    const auto policy = ResolveNoiseRealizations(noise_realizations, *noise_model, config_, NoiseLayer::kCombined, 64, suffix, false, std::nullopt, false);

    // The config is fixed when the checkpoint is built, so unlike
    // full_noise_execute its seed must not override the per-call seed: that
    // would replay the same samples on every call. A call's noise_seed seeds
    // both noise injection and measurement/readout. A seeded checkpoint called
    // without one derives a fresh seed per call, so the sequence of calls is
    // reproducible but no batch repeats.
    if (!noise_seed && config_.seed)
        noise_seed = Simulators::IState::DeriveSeed(*config_.seed, calls_) & UINT32_MAX;
    auto call_config = config_;
    call_config.seed.reset();
    auto rng = MakeNoiseRng(call_config, noise_seed);
    ++calls_;
    // Each batch gets its own measurement stream derived from the call seed.
    const auto seed_batch = [&](const std::shared_ptr<Simulators::ISimulator> &simulator, size_t batch) {
        if (const auto seed = NoiseExecutionConfig(call_config, noise_seed, batch).seed)
            simulator->SetSeed(*seed);
    };

    ShotRecords records;
    records.shots = shots;
    records.width = total_cbits;
    records.simulator = config_.simulator_type;
    records.method = config_.simulation_type;
    records.components = 1;
    records.largest_component = static_cast<size_t>(num_qubits_);
    records.policy = policy;
    Circuits::Circuit<>::ExecuteResults combined;
    if (rows)
        records.rows.assign(SampleStorageSize(shots, total_cbits), 0);
    const auto record = [&](const Circuits::OperationState &state, size_t shot, Circuits::Circuit<>::ExecuteResults &counts) {
        const auto &bits = state.GetAllBits();
        const size_t written = std::min(bits.size(), total_cbits);
        if (rows)
        {
            uint8_t *row = records.rows.data() + shot * total_cbits;
            for (size_t i = 0; i < written; ++i)
                row[i] = static_cast<uint8_t>(bits[i]);
        }
        else
        {
            std::vector<bool> key(total_cbits, false);
            for (size_t i = 0; i < written; ++i)
                key[i] = bits[i];
            ++counts[std::move(key)];
        }
    };
    // The sampler operates on the current state without replacing the saved
    // prefix. Dynamic suffixes keep their per-shot branches; their
    // deterministic head evolves once on a copy when two more states fit the
    // realization budget (HeadCopyFits).
    const bool reuse_heads = HeadCopyFits(config_, static_cast<size_t>(num_qubits_));
    const auto sample_suffix = [&](const RecordCircuitPtr &circuit, const std::shared_ptr<Simulators::ISimulator> &simulator, Circuits::OperationState &state,
                                   size_t batch_shots, size_t first_shot, std::optional<uint64_t> seed, Circuits::Circuit<>::ExecuteResults &counts,
                                   size_t &max_bond) {
        state.Reset();
        simulator->RestoreState();
        // A suffix begins in an arbitrary saved state: even its first reset
        // may branch on a pure-state backend, so CanSampleTerminal sends any
        // pure-state reset to the shot path. Backends whose sampler uses the
        // save slot also replay shots, keeping the prefix intact.
        if (batch_shots > 1 && PreservesSavedStateWhileSampling(*simulator) &&
            SamplePreparedTerminal(
                *circuit, simulator, batch_shots, state, seed,
                [&](size_t shot, const Circuits::OperationState &sample) { record(sample, first_shot + shot, counts); }, &max_bond))
            return true;
        if (reuse_heads && SampleSharedHead(
                               *circuit, simulator, batch_shots, state, seed,
                               [&](size_t shot, const Circuits::OperationState &sample) { record(sample, first_shot + shot, counts); }, &max_bond))
            return false;
        for (size_t shot = 0; shot < batch_shots; ++shot)
        {
            if (shot > 0)
                simulator->RestoreState();
            simulator->SetGatesCounter(0);
            simulator->SetUpcomingGates(circuit->GetOperations());
            circuit->ExecuteBD(simulator, state, &max_bond);
            record(state, first_shot + shot, counts);
        }
        return false;
    };
    const auto start = std::chrono::steady_clock::now();
    size_t terminal_calls = 0, trajectory_calls = 0;

    if (!has_noise)
    {
        seed_batch(simulator_, 0);
        const auto execution_circuit = rows ? CloneSamplingCircuitForExecution(suffix, noise_seed) : suffix;
        Circuits::OperationState state(total_cbits);
        if (sample_suffix(execution_circuit, simulator_, state, shots, 0, NoiseExecutionConfig(call_config, noise_seed, 0).seed, combined, max_bond_dim_))
            ++terminal_calls;
        else
            ++trajectory_calls;
    }
    else
    {
        const size_t batches = std::min(shots, policy.realizations);
        const size_t base_batch = shots / batches;
        const size_t leftover = shots % batches;
        const size_t workers = RealizationWorkers(batches, reuse_heads ? 4 : 2);

        struct Worker
        {
            std::shared_ptr<Simulators::ISimulator> simulator;
            Circuits::OperationState state;
            Circuits::Circuit<>::ExecuteResults counts;
            size_t max_bond_dim = 0;
            size_t terminal_calls = 0;
            size_t trajectory_calls = 0;
        };

        std::vector<Worker> worker_states;
        worker_states.reserve(workers);
        // Clone serially: adapters assign clone streams by mutating their
        // source. Saved prefix states and all execution state stay private.
        for (size_t worker = 0; worker < workers; ++worker)
        {
            std::shared_ptr<Simulators::ISimulator> simulator = workers == 1 ? simulator_ : std::shared_ptr<Simulators::ISimulator>(simulator_->Clone());
            if (workers > 1)
                simulator->SetMultithreading(false);
            worker_states.push_back({std::move(simulator), Circuits::OperationState(total_cbits), {}, max_bond_dim_});
        }
        // An unseeded call must not copy the same measurement RNG into
        // multiple workers. This stream is separate from noise injection.
        const uint64_t unseeded_measurements = noise_seed ? 0 : Utils::RandomStream::FreshSeed();
        const noise::NoiseInjectionPlan plan(suffix, *noise_model);
        const bool exact = policy.exact_channels;
        // The checkpoint register is fixed. Crosstalk spectators beyond a
        // suffix that fits it stay in |0>, where their rotations are a global
        // phase (drop_outside_spectator_phases).
        const size_t register_qubits = static_cast<size_t>(num_qubits_);
        const bool trim_spectators =
            plan.RegisterWidth() > register_qubits && (suffix->GetOperations().empty() || suffix->GetMaxQubitIndex() < register_qubits);
        Utils::RunRealizations(
            batches, workers, [&](size_t) { return InjectPlanned(NoiseLayer::kCombined, plan, rng, exact); },
            [&](size_t worker, size_t batch, RecordCircuitPtr noisy_suffix) {
                auto &local = worker_states[worker];
                if (trim_spectators)
                    noisy_suffix = noise::drop_outside_spectator_phases(noisy_suffix, register_qubits);
                if (noise_seed)
                    seed_batch(local.simulator, batch);
                else if (workers > 1)
                    local.simulator->SetSeed(Simulators::IState::DeriveSeed(unseeded_measurements, batch));
                if (rows)
                    noisy_suffix = CloneSamplingCircuitForExecution(noisy_suffix, NoiseExecutionConfig(call_config, noise_seed, batch).seed);
                const size_t batch_shots = base_batch + (batch < leftover ? 1 : 0);
                const size_t first_shot = batch * base_batch + std::min(batch, leftover);
                if (sample_suffix(noisy_suffix, local.simulator, local.state, batch_shots, first_shot,
                                  NoiseExecutionConfig(call_config, noise_seed, batch).seed, local.counts, local.max_bond_dim))
                    ++local.terminal_calls;
                else
                    ++local.trajectory_calls;
            });
        for (const auto &local : worker_states)
        {
            max_bond_dim_ = std::max(max_bond_dim_, local.max_bond_dim);
            terminal_calls += local.terminal_calls;
            trajectory_calls += local.trajectory_calls;
            for (const auto &[bits, count] : local.counts)
                combined[bits] += count;
        }
        records.realizations = batches;
        records.policy.realizations = batches;
    }
    records.time_taken = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    records.max_bond_dimension = max_bond_dim_;
    records.sampling_strategy = trajectory_calls == 0 ? "terminal_v1" : terminal_calls == 0 ? "trajectory_v1" : "mixed_v1";
    if (!rows)
    {
        records.counts.reserve(combined.size());
        for (auto &[bits, count] : combined)
            records.counts.emplace_back(bits, count);
    }
    return records;
}

size_t PrefixCheckpoint::RealizationWorkers(size_t batches, size_t copies_per_worker) const
{
    // Other backends can own a GPU context or distributed communicator.
    if (config_.simulator_type != Simulators::SimulatorType::kQCSim)
        return 1;
    size_t workers = Utils::RealizationWorkerCount(batches, network_->GetMaxSimulators());
    const auto state = RealizationStateOf(config_.simulation_type);
    if (!state)
        return 1; // Unbounded representations cannot promise a memory limit.
    size_t cap = 0;
    if (*state == Utils::RealizationState::kMatrixProductState || *state == Utils::RealizationState::kMatrixProductOperator)
    {
        const auto &options = simulator_->GetConfigMap();
        const bool mpo = config_.simulation_type == Simulators::SimulationType::kMatrixProductOperator;
        auto found = options.find(mpo ? "matrix_product_operator_max_bond_dimension" : "matrix_product_state_max_bond_dimension");
        if (found == options.end())
            found = options.find("matrix_product_state_max_bond_dimension");
        if (found == options.end() || std::stoull(found->second) == 0)
            return 1;
        cap = static_cast<size_t>(std::stoull(found->second));
    }
    const long double state_bytes = Utils::RealizationStateBytes(*state, static_cast<size_t>(num_qubits_), cap);
    if (!(state_bytes > 0) || !std::isfinite(state_bytes) || state_bytes >= static_cast<long double>(std::numeric_limits<size_t>::max()))
        return 1;
    // Source and cloned simulators each retain a live and a saved state; a
    // worker holding a dynamic suffix's head keeps two more.
    return Utils::LimitRealizationWorkersByMemory(workers, static_cast<size_t>(state_bytes), RealizationBudgetMB(config_), 2, copies_per_worker);
}
} // namespace MaestroExecution
