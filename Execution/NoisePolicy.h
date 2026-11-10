// Shared realization policy for exact channels and sampled noise/branches.
#pragma once

#include "../Noise/NoiseModel.h"
#include "SimulatorConfig.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_set>

namespace MaestroExecution
{
enum class NoiseLayer
{
    kPauli,
    kCoherent,
    kCombined
};

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
inline bool HasStochasticEvolution(const std::shared_ptr<Circuits::Circuit<double>> &circuit, bool exact_channels, bool fresh_register = true)
{
    if (!circuit)
        return false;
    const auto unmeasured = circuit->RemoveEndMeasurements();
    std::unordered_set<Types::qubit_t> touched;
    return !unmeasured->HasDeterministicEvolution(exact_channels, fresh_register, touched);
}

struct NoiseRealizationPolicy
{
    std::optional<size_t> requested;
    size_t realizations = 1;
    bool exact_channels = false;
    bool stochastic_noise = false;
    bool stochastic_execution = false;
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
    result.stochastic_noise = (layer != NoiseLayer::kPauli && model.has_coherent()) || (layer == NoiseLayer::kCombined && model.has_correlated());
    if (!result.exact_channels)
        result.stochastic_noise = result.stochastic_noise || model.has_non_readout_noise() || (circuit && circuit->HasQuantumChannels());
    result.stochastic_execution = HasStochasticEvolution(circuit, UsesExactQuantumChannels(config), fresh_register);
    result.realizations = requested.value_or(result.exact_channels && !result.stochastic_noise && (!estimation || !result.stochastic_execution) ? 1 : fallback);
    return result;
}
} // namespace MaestroExecution
