// Shared realization policy for exact channels and sampled noise/branches.
#pragma once

#include "../Noise/NoiseModel.h"
#include "SimulatorConfig.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace MaestroExecution
{
using RecordCircuitPtr = std::shared_ptr<Circuits::Circuit<double>>;

enum class NoiseLayer
{
    kPauli,
    kCoherent,
    kCombined
};

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

// A backend that samples channels from its state can draw every branch of
// the noise as it runs. Where sampled realizations keep a channel for it to
// sample anyway, so that no two shots could share one, and the layer draws
// nothing of its own per realization, every realization is instead the same
// circuit, holding all the noise as channels: it is built once, no
// realization is prepared, and each execution is a trajectory of its own.
//
// The first realization of the plan decides. If it keeps a channel (which ones
// a realization keeps does not depend on what it draws), `shared` receives that
// circuit and `first` is dropped. Otherwise `first` holds the realization,
// drawn from `rng` as the first one always is, for the caller to run first:
// injection has lowered all the noise to gates and resets, and those
// realizations are the ones every pure-state method runs. Nothing is drawn
// when the plan keeps no channels, the layer has noise of its own per
// realization, or the circuit holds operations that executions must not share.
inline void ResolveRuntimeNoise(NoiseLayer layer, const noise::NoiseInjectionPlan &plan, std::mt19937 &rng, RecordCircuitPtr &shared, RecordCircuitPtr &first)
{
    if (!plan.RuntimeChannels())
        return;
    const auto &model = plan.model();
    if ((layer != NoiseLayer::kPauli && model.has_coherent()) || (layer == NoiseLayer::kCombined && model.has_correlated()))
        return;
    using Type = Circuits::OperationType;
    for (const auto &op : plan.circuit()->GetOperations())
        switch (op->GetType())
        {
        case Type::kGate:
        case Type::kDelay:
        case Type::kNoOp:
        case Type::kReset:
        case Type::kQuantumChannel:
        case Type::kMeasurement:
            break;
        default: // classical generators, conditionals and nested circuits carry state of their own
            return;
        }
    first = InjectPlanned(layer, plan, rng, false);
    // A kept channel with one branch only is a gate (IsDeterministicChannel): the realization draws nothing as it runs.
    if (!first->HasDrawnChannels())
        return;
    first.reset();
    std::mt19937 unused;
    shared = InjectPlanned(layer, plan, unused, true);
}

// Whether the executions of a plan's realizations draw channel branches as they run: the realizations keep channels
// for the backend (plan.RuntimeChannels()) and at least one of those is drawn. The noise seed has no injection to
// drive for such branches, so it joins the seed of each execution (RuntimeNoiseExecutionConfig), whether the
// realizations share one circuit or a layer drawn for each of them keeps them apart: either seed alone then changes
// the errors drawn.
//
// Decided once for a call, never per realization. From what ResolveRuntimeNoise found: a shared circuit draws, a
// first realization it left has nothing to draw, and a plan that keeps no channels draws nothing. Otherwise (a
// layer of its own per realization, operations that executions must not share) the first realization prepared
// tells: which channels a realization keeps does not depend on what it draws.
class RuntimeDraws
{
  public:
    RuntimeDraws(const noise::NoiseInjectionPlan *plan, const RecordCircuitPtr &shared, const RecordCircuitPtr &first)
    {
        if (shared)
            draws_ = true;
        else if (first || !plan || !plan->RuntimeChannels())
            draws_ = false;
    }

    // To be called with the realizations as they are prepared, in order; only the first one is looked at.
    void See(const RecordCircuitPtr &realization)
    {
        if (!draws_)
            draws_ = realization && realization->HasDrawnChannels();
    }

    explicit operator bool() const
    {
        return draws_.value_or(false);
    }

  private:
    std::optional<bool> draws_;
};

// An execution seed with the noise seed joined to it, for executions that draw channel branches as they run
// (RuntimeDraws) under two given seeds: every entry point joins them this way, each to the execution seed it
// derives for a realization or for a call.
inline uint64_t JoinNoiseSeed(uint64_t execution_seed, uint64_t noise_seed)
{
    return Simulators::IState::DeriveSeed(execution_seed, noise_seed ^ 0x52544E4F495345ULL); // 'RTNOISE'
}

// Measurement trajectories a dynamic estimate averages when no noise
// realization count applies (native estimate requests and Python estimate).
inline constexpr size_t kMeasurementTrajectories = 100;

// Whether one backend evolves the ensemble with exact Kraus channels: a
// density matrix or MPO, which hold the mixed state, and the QCSim Pauli
// propagator, which applies the adjoint channels to the observable instead.
// Composite simulators still contain pure-state components and cannot
// represent ensemble channels exactly as a single mixed state.
inline bool SupportsExactQuantumChannels(Simulators::SimulatorType type, Simulators::SimulationType method)
{
    return Simulators::EvolvesExactQuantumChannels(type, method);
}

// An explicit candidate list replaces the requested pair (ConfigureNetwork):
// exact channels may be injected only when every backend it may select
// applies them. A mixed list samples trajectories, which every backend runs.
inline bool UsesExactQuantumChannels(const SimulatorConfig &config)
{
    if (HasCandidateList(config))
        return std::all_of(config.optimization_candidates.begin(), config.optimization_candidates.end(),
                           [](const auto &candidate) { return SupportsExactQuantumChannels(candidate.first, candidate.second); });
    return SupportsExactQuantumChannels(config.simulator_type, config.simulation_type);
}

// Whether every backend a request may run takes a quantum channel as one
// sampled Kraus branch read from its state (ISimulator::SamplesQuantumChannels).
inline bool SamplesQuantumChannels(const SimulatorConfig &config)
{
    if (HasCandidateList(config))
        return std::all_of(config.optimization_candidates.begin(), config.optimization_candidates.end(),
                           [](const auto &candidate) { return Simulators::SamplesQuantumChannels(candidate.first, candidate.second); });
    return Simulators::SamplesQuantumChannels(config.simulator_type, config.simulation_type);
}

// Whether some backend the configuration may select holds channel
// instructions, applying them exactly or as branches sampled from its state.
// That is enough for a circuit that runs as it is (no noise injection): the
// network tries its candidates in turn, and one that cannot run the circuit
// yields to the next.
inline bool SomeBackendHoldsQuantumChannels(const SimulatorConfig &config)
{
    const auto holds = [](Simulators::SimulatorType type, Simulators::SimulationType method) {
        return SupportsExactQuantumChannels(type, method) || Simulators::SamplesQuantumChannels(type, method);
    };
    if (HasCandidateList(config))
        return std::any_of(config.optimization_candidates.begin(), config.optimization_candidates.end(),
                           [&](const auto &candidate) { return holds(candidate.first, candidate.second); });
    return holds(config.simulator_type, config.simulation_type);
}

// Whether sampled realizations keep, as channel operations for the backend to
// sample when it runs them, the noise that has no exact sampled lowering:
// relaxation with T2 > T1, amplitude damping and Kraus maps. Without it that
// noise is approximated (T2 clamped to T1, damping as a reset) or rejected.
inline bool InjectsRuntimeChannels(const SimulatorConfig &config, bool exact_channels)
{
    return !exact_channels && SamplesQuantumChannels(config);
}

// The order realizations are put in for a tensor chain (Circuits::ChainOrdered),
// whose work follows the order of the operations once it is off its canonical
// form: when every backend the request may run is the QCSim matrix product
// state, which samples channels, or the QCSim matrix product operator, which
// applies them.
inline Circuits::ChainOrder OrdersForChain(const SimulatorConfig &config, bool exact_channels)
{
    const auto all = [&](Simulators::SimulationType wanted) {
        const auto chain = [&](Simulators::SimulatorType type, Simulators::SimulationType method) {
            return type == Simulators::SimulatorType::kQCSim && method == wanted;
        };
        if (HasCandidateList(config))
            return std::all_of(config.optimization_candidates.begin(), config.optimization_candidates.end(),
                               [&](const auto &candidate) { return chain(candidate.first, candidate.second); });
        return chain(config.simulator_type, config.simulation_type);
    };
    if (InjectsRuntimeChannels(config, exact_channels) && all(Simulators::SimulationType::kMatrixProductState))
        return Circuits::ChainOrder::kSampledChannels;
    if (exact_channels && all(Simulators::SimulationType::kMatrixProductOperator))
        return Circuits::ChainOrder::kMixedState;
    return Circuits::ChainOrder::kNone;
}

// Whether a request may run the Pauli propagator.
inline bool MayRunPauliPropagator(const SimulatorConfig &config)
{
    const auto propagator = [](Simulators::SimulationType method) { return method == Simulators::SimulationType::kPauliPropagator; };
    if (HasCandidateList(config))
        return std::any_of(config.optimization_candidates.begin(), config.optimization_candidates.end(),
                           [&](const auto &candidate) { return propagator(candidate.second); });
    return propagator(config.simulation_type);
}

// Whether noise is injected as exact channels when the caller does not say.
// A mixed-state backend always does. The Pauli propagator applies channels to
// the observable, which costs about one extra pass per channel while the
// channels are diagonal in the Pauli basis. A relaxation channel (T1, thermal)
// also sends Z to the identity. Without a truncation threshold those branches
// fill the operator space of the observable's light cone, 4^width terms,
// which is far more than the sampled trajectories it would replace. Such a
// model therefore keeps its trajectories on the propagator unless exact
// evaluation is requested, or unless it holds channels that cannot be sampled
// at all (generalized amplitude damping, correlated phase flips, Kraus maps).
// This form judges the model alone; pass the circuits when they are known.
inline bool InjectsExactChannels(const SimulatorConfig &config, const noise::NoiseModel &model)
{
    if (!UsesExactQuantumChannels(config))
        return false;
    if (!MayRunPauliPropagator(config))
        return true;
    return model.exact_channels_are_pauli_diagonal() || model.has_additional_quantum_channels();
}

// Whether a circuit holds a channel instruction that is not a Pauli mixture.
// No trajectory can sample one: it runs only as an exact channel.
inline bool HasKrausInstruction(const Circuits::Circuit<double> &circuit)
{
    for (const auto &op : circuit.GetOperations())
    {
        if (op->GetType() == Circuits::OperationType::kComposite)
        {
            if (HasKrausInstruction(*std::static_pointer_cast<Circuits::Circuit<double>>(op)))
                return true;
        }
        else if (op->GetType() == Circuits::OperationType::kQuantumChannel &&
                 std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(op)->GetChannel().GetPauliProbabilities().empty())
            return true;
    }
    return false;
}

// The same decision once the circuits the noise is injected into are known
// (several for a checkpoint: its suffixes). On the Pauli propagator two more
// requests are exact:
//   - a model whose relaxation never acts on the circuits, since the channels
//     injected into them are then all diagonal;
//   - one with a Kraus instruction in a circuit. It cannot be sampled, so it
//     asks for exact evaluation as a Kraus channel of the model does.
// The circuits are read only for a model that holds relaxation.
using NoisyCircuits = std::vector<const Circuits::Circuit<double> *>;

inline bool InjectsExactChannels(const SimulatorConfig &config, const noise::NoiseModel &model, const NoisyCircuits &circuits)
{
    if (!UsesExactQuantumChannels(config))
        return false;
    if (!MayRunPauliPropagator(config) || model.has_additional_quantum_channels() || model.exact_channels_are_pauli_diagonal())
        return true;
    for (const auto *circuit : circuits)
        if (circuit && HasKrausInstruction(*circuit))
            return true;
    for (const auto *circuit : circuits)
        if (circuit && !model.exact_channels_are_pauli_diagonal_on(*circuit))
            return false;
    return true;
}

inline bool InjectsExactChannels(const SimulatorConfig &config, const noise::NoiseModel &model, const Circuits::Circuit<double> &circuit)
{
    return InjectsExactChannels(config, model, NoisyCircuits{&circuit});
}

// Terminal measurements are ignored by expectation evaluation. Any remaining
// measurement/control/classical operation may select a branch. Native mixed-
// state reset is a deterministic channel. A pure-state reset can collapse,
// except one that precedes every other operation on its wire of a fresh
// register; pass fresh_register=false for a circuit continuing a saved state.
//
// A circuit that is executed as it is draws a branch of every channel it holds
// when the backend samples channels (ChannelDraws::kAll). A circuit that noise
// injection will sample first keeps only the channels a Pauli draw cannot
// replace (kFromState).
inline bool HasStochasticEvolution(const std::shared_ptr<Circuits::Circuit<double>> &circuit, bool exact_channels, bool fresh_register = true,
                                   Circuits::Circuit<double>::ChannelDraws draws = Circuits::Circuit<double>::ChannelDraws::kAll)
{
    if (!circuit)
        return false;
    const auto unmeasured = circuit->RemoveEndMeasurements();
    std::unordered_set<Types::qubit_t> touched;
    return !unmeasured->HasDeterministicEvolutionWith(draws, exact_channels, fresh_register, touched);
}

struct NoiseRealizationPolicy
{
    std::optional<size_t> requested;
    size_t realizations = 1;
    bool exact_channels = false;
    bool stochastic_noise = false;
    bool stochastic_execution = false;
    // Sampled realizations keep, as channel operations for a backend that
    // samples them from its state, the noise that has no exact sampled
    // lowering (InjectsRuntimeChannels): nothing is clamped or refused.
    bool runtime_channels = false;
    // The realizations are in chain order, for a tensor chain (OrdersForChain).
    Circuits::ChainOrder chain_order = Circuits::ChainOrder::kNone;
};

// Explicit counts keep their meaning. Auto uses one prepared mixed state only
// when injection is deterministic; estimates must also average circuit branches.
// Counts already sample those branches in their ordinary per-shot execution.
// fresh_register=false describes a circuit that continues a saved state (a
// checkpoint suffix).
inline NoiseRealizationPolicy ResolveNoiseRealizations(std::optional<size_t> requested, const noise::NoiseModel &model, const SimulatorConfig &config,
                                                       NoiseLayer layer, size_t fallback, const std::shared_ptr<Circuits::Circuit<double>> &circuit = {},
                                                       bool estimation = false, std::optional<bool> exact_channels = std::nullopt, bool fresh_register = true)
{
    if ((requested && *requested == 0) || fallback == 0)
        throw std::invalid_argument("noise_realizations must be >= 1.");
    NoiseRealizationPolicy result;
    result.requested = requested;
    result.exact_channels = exact_channels ? *exact_channels : circuit ? InjectsExactChannels(config, model, *circuit) : InjectsExactChannels(config, model);
    result.runtime_channels = InjectsRuntimeChannels(config, result.exact_channels);
    result.chain_order = OrdersForChain(config, result.exact_channels);
    result.stochastic_noise = (layer != NoiseLayer::kPauli && model.has_coherent()) || (layer == NoiseLayer::kCombined && model.has_correlated());
    if (!result.exact_channels)
        result.stochastic_noise = result.stochastic_noise || model.has_non_readout_noise() || (circuit && circuit->HasDrawnChannels());
    result.stochastic_execution =
        HasStochasticEvolution(circuit, UsesExactQuantumChannels(config), fresh_register, Circuits::Circuit<double>::ChannelDraws::kFromState);
    result.realizations = requested.value_or(result.exact_channels && !result.stochastic_noise && (!estimation || !result.stochastic_execution) ? 1 : fallback);
    return result;
}
} // namespace MaestroExecution
