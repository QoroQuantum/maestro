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

// Whether one backend evolves the ensemble with exact Kraus channels.
// Composite simulators still contain pure-state components and cannot
// represent ensemble channels exactly as a single mixed state.
inline bool SupportsExactQuantumChannels(Simulators::SimulatorType type, Simulators::SimulationType method)
{
    using Backend = Simulators::SimulatorType;
    using Method = Simulators::SimulationType;
#ifndef NO_QISKIT_AER
    if (type == Backend::kQiskitAer)
        return method == Method::kDensityMatrix;
#endif
    return (type == Backend::kQCSim || type == Backend::kGpuSim) && (method == Method::kDensityMatrix || method == Method::kMatrixProductOperator);
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
    result.exact_channels = exact_channels.value_or(UsesExactQuantumChannels(config));
    result.stochastic_noise = (layer != NoiseLayer::kPauli && model.has_coherent()) || (layer == NoiseLayer::kCombined && model.has_correlated());
    if (!result.exact_channels)
        result.stochastic_noise = result.stochastic_noise || model.has_non_readout_noise() || (circuit && circuit->HasQuantumChannels());
    result.stochastic_execution = HasStochasticEvolution(circuit, UsesExactQuantumChannels(config), fresh_register);
    result.realizations = requested.value_or(result.exact_channels && !result.stochastic_noise && (!estimation || !result.stochastic_execution) ? 1 : fallback);
    return result;
}
} // namespace MaestroExecution
