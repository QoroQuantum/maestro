// Raw shot records and noisy shot execution for C++ callers, the native request
// API and the Python bindings. This header has no Python dependency.
#pragma once

#include "../Noise/NoiseModel.h"
#include "NoisePolicy.h"
#include "ReferenceFrameSampler.h"
#include "Sampling.h"
#include "SimulatorConfig.h"
#include "Transforms/Export.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace MaestroExecution
{
using RecordCircuitPtr = std::shared_ptr<Circuits::Circuit<double>>;

// Columns of a circuit's shot records: every classical bit it writes and, as
// in 0.3.13's count keys, at least one column per qubit.
inline size_t ClassicalWidth(const Circuits::Circuit<double> &circuit)
{
    const auto bits = circuit.GetBits();
    return std::max(std::max(size_t{1}, circuit.GetMaxQubitIndex() + 1), bits.empty() ? size_t{0} : *bits.rbegin() + 1);
}

inline size_t SampleStorageSize(size_t shots, size_t width)
{
    if (width && shots > std::numeric_limits<size_t>::max() / width)
        throw std::length_error("Sample matrix dimensions overflow");
    return shots * width;
}

// Methods a direct session runs as the network executor would. MPS and MPO
// are excluded: the network places their initial qubit map and lookahead swap
// routing per execution (OptimizeMPSInitialQubitsMap), which changes
// truncation accuracy, so their raw samples and counts use the network
// executor. Tensor networks and Pauli propagation have no direct sampler. A
// direct session runs the requested method on the whole circuit, so a
// candidate list, from which the network selects, and sampling reduction,
// which the network applies by splitting disjoint measurement cones, keep the
// network executor too.
inline bool SupportsDirectSamples(const SimulatorConfig &config)
{
    using Method = Simulators::SimulationType;
    return config.simulator_type == Simulators::SimulatorType::kQCSim && !HasCandidateList(config) && !config.enable_sampling_reduction &&
           config.simulation_type != Method::kTensorNetwork && config.simulation_type != Method::kPauliPropagator &&
           config.simulation_type != Method::kMatrixProductState && config.simulation_type != Method::kMatrixProductOperator;
}

// Auto may run each independent one-shot realization directly on a worker's
// simulator and histogram its record, instead of mapping, optimizing and
// allocating a backend per realization. Its derived seed makes either path repeatable.
inline bool PreferDirectTrajectoryCounts(const SimulatorConfig &config)
{
    return SupportsDirectSamples(config) && EffectiveCountsPolicy(config) == "auto";
}

inline size_t SamplingWorkers(const SimulatorConfig &config, size_t realizations, size_t qubits, size_t copies_per_worker = 2)
{
    if (!RunsOnlyQCSim(config))
        return 1;
    const auto requested = EffectiveMaxSimulators(config);
    auto workers = Utils::RealizationWorkerCount(realizations, requested.value_or(0));
    // Unknown/unbounded state sizes cannot promise a memory limit. A candidate
    // list is sized for the largest register any of its entries needs.
    const long double state_bytes = MaxRealizationBytes(config, qubits);
    if (!(state_bytes > 0))
        return 1;
    workers = Utils::LimitRealizationWorkersByMemory(workers, static_cast<size_t>(state_bytes), RealizationBudgetMB(config), 0, copies_per_worker);
    // A default ceiling only: an explicit max_simulators is honored, within
    // the CPU budget RealizationWorkerCount already applies.
    return requested ? workers : std::min(workers, size_t{16});
}

// Dense backends allocate every index up to the highest qubit. The network
// executor compacts each circuit to the qubits it uses (RemapToContinuous);
// direct sessions apply the same increasing relabelling, computed once per
// call. Remap keeps classical bits, so keys and sample columns are unchanged.
class QubitCompaction
{
  public:
    explicit QubitCompaction(const Circuits::Circuit<double> &circuit)
    {
        const auto used = circuit.GetQubits();
        highest = used.empty() ? 0 : *used.rbegin();
        width = std::max<size_t>(1, used.size());
        if (used.size() < highest + 1)
            for (const auto q : used)
                map.emplace(q, map.size());
    }

    // Qubits a session allocates.
    size_t Width() const noexcept
    {
        return width;
    }

    bool Sparse() const noexcept
    {
        return !map.empty();
    }

    // Whether the circuit touches qubit q, i.e. q survives compaction.
    bool Contains(Types::qubit_t q) const
    {
        return Sparse() ? map.count(q) != 0 : q < width;
    }

    RecordCircuitPtr Apply(const RecordCircuitPtr &circuit) const
    {
        return Sparse() ? std::static_pointer_cast<Circuits::Circuit<double>>(circuit->Remap(map)) : circuit;
    }

  private:
    size_t highest = 0, width = 1;
    std::unordered_map<Types::qubit_t, Types::qubit_t> map;
};

inline bool HasMeasurements(const RecordCircuitPtr &circuit)
{
    for (const auto &operation : circuit->GetOperations())
    {
        if (operation->GetType() == Circuits::OperationType::kMeasurement)
            return true;
        if (operation->GetType() == Circuits::OperationType::kComposite && HasMeasurements(std::static_pointer_cast<Circuits::Circuit<double>>(operation)))
            return true;
    }
    return false;
}

inline bool HasClassicalRandomOperations(const RecordCircuitPtr &circuit)
{
    for (const auto &operation : circuit->GetOperations())
    {
        if (operation->GetType() == Circuits::OperationType::kRandomGen || operation->GetType() == Circuits::OperationType::kConditionalRandomGen)
            return true;
        if (operation->GetType() == Circuits::OperationType::kComposite &&
            HasClassicalRandomOperations(std::static_pointer_cast<Circuits::Circuit<double>>(operation)))
            return true;
    }
    return false;
}

// Classical Random operations own mutable generators, including nested and
// conditional ones. Executions run a clone and never mutate the input.
inline RecordCircuitPtr CloneSamplingCircuitForExecution(const RecordCircuitPtr &circuit, std::optional<uint64_t> seed)
{
    if (!HasClassicalRandomOperations(circuit))
        return circuit;
    const uint64_t entropy = !seed && circuit->HasUnseededRandomOperations() ? Utils::RandomStream::FreshSeed() : 0;
    return circuit->CloneForExecution(Simulators::IState::DeriveSeed(seed.value_or(0), 0x434c415353494341ULL), entropy);
}

namespace SampleRowOrder
{
inline constexpr uint64_t kTerminalStream = 0x5445524d524f5753ULL;  // "TERMROWS"
inline constexpr uint64_t kHistogramStream = 0x434f554e54524f57ULL; // "COUNTROW"

// std::uniform_int_distribution differs between standard libraries.
inline uint64_t UniformBelow(std::mt19937_64 &engine, uint64_t bound)
{
    const uint64_t threshold = (uint64_t(0) - bound) % bound; // 2^64 mod bound
    for (;;)
        if (const uint64_t value = engine(); value >= threshold)
            return value % bound;
}

inline std::vector<size_t> Permutation(size_t shots, std::optional<uint64_t> seed, uint64_t stream)
{
    std::vector<size_t> order(shots);
    std::iota(order.begin(), order.end(), size_t{0});
    std::mt19937_64 shuffle(seed ? Simulators::IState::DeriveSeed(*seed, stream) : Utils::RandomStream::FreshSeed());
    for (size_t remaining = shots; remaining > 1; --remaining)
        std::swap(order[remaining - 1], order[UniformBelow(shuffle, remaining)]);
    return order;
}
} // namespace SampleRowOrder

// The warning for a sampled thermal approximation on these qubits, or empty.
inline std::string ThermalApproximationMessage(const std::vector<int> &qubits)
{
    if (qubits.empty())
        return {};
    std::ostringstream message;
    message << "Sampled thermal approximation for circuit qubits [";
    for (size_t i = 0; i < qubits.size(); ++i)
    {
        if (i)
            message << ", ";
        message << qubits[i];
    }
    message << "]: effective T2 clamped to T1. Use density matrix or a supported "
               "MPO backend to preserve calibrated T2. Kraus trajectories are "
               "not supported by Maestro's current SV/MPS noise path.";
    return message.str();
}

// The warning for a sampled thermal approximation, or empty. Callers emit it
// once per execution, not per realization. The circuit form names only the
// qubits whose relaxation that circuit draws in the clamped window.
inline std::string ThermalApproximationWarning(const noise::NoiseModel &noise_model, bool exact_channels)
{
    if (exact_channels || noise_model.has_additional_quantum_channels())
        return {};
    return ThermalApproximationMessage(noise_model.thermal_approximation_qubits());
}

inline std::string ThermalApproximationWarning(const noise::NoiseModel &noise_model, bool exact_channels, const Circuits::Circuit<double> &circuit)
{
    if (exact_channels || noise_model.has_additional_quantum_channels())
        return {};
    return ThermalApproximationMessage(noise_model.thermal_approximation_qubits_on(circuit));
}

// The noise seed is 32 bits, as in the JSON API's noise.seed.
inline void RequireNoiseSeed(std::optional<uint64_t> noise_seed)
{
    if (noise_seed && *noise_seed > UINT32_MAX)
        throw std::invalid_argument("noise_seed must fit in 32 bits");
}

// An omitted noise seed falls back to the config seed's low 32 bits, as in the
// JSON API, so a config seed alone reproduces the injected noise. All MPI ranks
// must submit the same stochastic circuit, so MPI resolves an unseeded call
// once and shares it with measurement/readout execution.
inline std::mt19937 MakeNoiseRng(const SimulatorConfig &config, std::optional<uint64_t> &noise_seed)
{
    RequireNoiseSeed(noise_seed);
    const auto configured_seed = EffectiveSamplingSeed(config);
    if (!noise_seed && configured_seed)
        noise_seed = *configured_seed & UINT32_MAX;
    else if (!noise_seed && config.simulator_type == Simulators::SimulatorType::kDistMpiGpuSim)
        noise_seed = Simulators::GenerateRandomSeed(config.simulator_type, config.distributed_options) & UINT32_MAX;
    if (noise_seed)
        return std::mt19937(static_cast<uint32_t>(*noise_seed));
    return std::mt19937(static_cast<uint32_t>(Utils::RandomStream::FreshSeed()));
}

// Keep simulator randomness separate from circuit-noise injection. An explicit
// config seed takes precedence; otherwise the noise seed also seeds
// measurement/readout and reset collapse. Every batch needs its own stream,
// including one-shot executions and expectation-value realizations.
inline SimulatorConfig NoiseExecutionConfig(const SimulatorConfig &config, std::optional<uint64_t> noise_seed, uint64_t batch)
{
    auto execution_config = config;
    // Noisy workflows retain their requested backend, even for realizations
    // in which stochastic injection happens to insert no error.
    execution_config.auto_reduce = false;
    execution_config.seed = EffectiveSamplingSeed(config);
    if (!execution_config.seed && noise_seed)
        execution_config.seed = *noise_seed;
    if (execution_config.seed)
        execution_config.seed = Simulators::IState::DeriveSeed(*execution_config.seed, batch);
    return execution_config;
}

// One realization of circuit under the given layers, readout included.
inline RecordCircuitPtr InjectNoise(NoiseLayer layer, const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, std::mt19937 &rng, bool exact)
{
    RecordCircuitPtr noisy;
    if (layer == NoiseLayer::kPauli)
        noisy = exact ? noise::inject_exact_noise(circuit, noise_model) : noise::inject_noise(circuit, noise_model, rng);
    else if (layer == NoiseLayer::kCoherent)
        noisy = exact ? noise::inject_coherent_noise_exact(circuit, noise_model, rng) : noise::inject_coherent_noise(circuit, noise_model, rng);
    else
        noisy = exact ? noise::inject_combined_noise_exact(circuit, noise_model, rng) : noise::inject_combined_noise(circuit, noise_model, rng);
    noise::attach_readout_error(noisy, noise_model);
    return noisy;
}

// InjectNoise(layer, plan.circuit(), plan.model(), rng, exact) drawn from the
// plan: identical operations and RNG use, without per-realization lookups.
inline RecordCircuitPtr InjectPlanned(NoiseLayer layer, const noise::NoiseInjectionPlan &plan, std::mt19937 &rng, bool exact)
{
    RecordCircuitPtr noisy;
    if (layer == NoiseLayer::kPauli)
        noisy = exact ? plan.ExactPauli() : plan.Pauli(rng);
    else if (layer == NoiseLayer::kCoherent)
        noisy = plan.Coherent(rng, exact);
    else
        noisy = plan.Combined(rng, exact);
    plan.AttachReadout(noisy);
    return noisy;
}

// Shot records of one execution: a row matrix, or distinct records and counts.
struct ShotRecords
{
    NoiseRealizationPolicy policy;
    // Row-major shots x width bytes (0 or 1), classical bit 0 first, when rows
    // were requested. Rows are exchangeable: any leading slice of a seeded
    // matrix is a uniform subsample of it.
    std::vector<uint8_t> rows;
    // Otherwise each distinct record once, with its count.
    std::vector<std::pair<std::vector<bool>, size_t>> counts;
    size_t shots = 0;
    size_t width = 0;
    Simulators::SimulatorType simulator = Simulators::SimulatorType::kQCSim;
    Simulators::SimulationType method = Simulators::SimulationType::kStatevector;
    int gpu_device = -1;
    bool gate_fusion = false;
    size_t gate_fusion_max_qubits = 0;
    size_t components = 0;
    size_t largest_component = 0;
    size_t max_bond_dimension = 0;
    // Noise realizations executed; zero for a noiseless execution.
    size_t realizations = 0;
    // reference_frame_v1, extended_frame_v1, terminal_v1 or trajectory_v1 when a dedicated
    // sampler drew the records; empty when the ordinary executor counted them.
    std::string sampling_strategy;
    double time_taken = 0;
};

// A shots x ClassicalWidth(circuit) row matrix of a noiseless circuit, drawn
// by the cheapest exact sampler: reference frames for Clifford circuits the
// configuration admits, one prepared state for terminal measurements, or
// shot-by-shot execution. Seeded matrices are reproducible.
MAESTRO_TRANSFORM_API ShotRecords SampleCircuitRows(const RecordCircuitPtr &circuit, const SimulatorConfig &config, size_t shots);

struct NoisyExecutionOptions
{
    NoiseLayer layer = NoiseLayer::kPauli;
    // A row matrix instead of counts.
    bool rows = false;
    // Exact channels instead of sampled trajectories; by default
    // UsesExactQuantumChannels(config).
    std::optional<bool> exact_channels;
};

// Splits shots across min(shots, noise_realizations) realizations, each with
// its own injected noise. noise_seed (32 bits) seeds injection and, unless
// config.seed is set, measurement and readout. With independent noise per shot
// and a stabilizer-admitting configuration, rows (and counts under the
// configuration's counts_policy) come from compiled reference frames. Emits no
// warnings: see ThermalApproximationWarning.
MAESTRO_TRANSFORM_API ShotRecords ExecuteNoisy(const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config,
                                               size_t shots, std::optional<size_t> noise_realizations, std::optional<uint64_t> noise_seed,
                                               const NoisyExecutionOptions &options);

// Only gates (and delays or no-ops), including in nested circuits: a prefix
// that evolves deterministically, so frames may evolve it once for all shots.
inline bool IsGateOnlyCircuit(const Circuits::Circuit<double> &circuit)
{
    for (const auto &op : circuit.GetOperations())
    {
        if (!op)
            return false;
        const auto type = op->GetType();
        if (type == Circuits::OperationType::kComposite)
        {
            if (!IsGateOnlyCircuit(*std::static_pointer_cast<Circuits::Circuit<double>>(op)))
                return false;
        }
        else if (type != Circuits::OperationType::kGate && type != Circuits::OperationType::kNoOp && type != Circuits::OperationType::kDelay)
            return false;
    }
    return true;
}

// Batched extended-stabilizer frames (sampling_strategy extended_frame_v1):
// noise_model injected as exact channels after circuit's gates, readout
// attached, and shots split into realizations as ExecuteNoisy splits them
// (shots of one realization share its errors). prefix, when not null, is a
// noiseless gate-only circuit applied before circuit, evolved once per call
// as a checkpointed prefix is. noise_seed drives circuit errors and
// measurement_seed the measurements and readout. Fills samples (rows, shots x
// width bytes) or counts (width-bit keys). Returns false, having drawn
// nothing, when frames do not apply: an unsupported circuit; coherent,
// correlated or crosstalk noise; a non-Pauli channel; relaxation that sampled
// execution draws as resets, when realizations are shared; or thermal noise
// with T2 > T1, which sampled execution clamps. The memory budget is soft: a
// shot that alone exceeds it runs by itself beyond it, keeping its draws.
// Callers decide the policy (PreferExtendedFrames) and keep exact channels
// and the coherent layer on their own executors.
MAESTRO_TRANSFORM_API bool SampleExtendedFrames(const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config,
                                                size_t qubits, size_t width, size_t shots, size_t realizations, uint64_t noise_seed, uint64_t measurement_seed,
                                                bool rows, std::vector<uint8_t> &samples, FrameOutcomes &counts, const RecordCircuitPtr &prefix = nullptr);
} // namespace MaestroExecution
