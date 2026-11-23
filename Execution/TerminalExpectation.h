#pragma once

#include "Estimation.h"

namespace noise
{
class NoiseInjectionPlan;
}

namespace MaestroExecution
{
enum class NoiseLayer;

// Each requested observable is a linear combination of prefix_observables after
// applying the terminal channels' adjoints. The terms of each row must be summed
// on the same realization before computing a trajectory variance.
struct TerminalExpectation
{
    std::shared_ptr<Circuits::Circuit<double>> prefix;
    std::vector<std::string> prefix_observables;
    std::vector<std::vector<std::pair<size_t, double>>> rows;
    size_t original_width = 0;
    // Only the prefix decides whether the transformed estimate needs repeated trajectories.
    bool stochastic_prefix = false;
};

MAESTRO_TRANSFORM_API std::optional<TerminalExpectation> PrepareTerminalExpectation(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                                    const std::vector<std::string> &observables, const SimulatorConfig &config);
// For a noise-model call: reuse a circuit with runtime channels if available.
// Otherwise restore the terminal noise that would have been pre-drawn,
// while preserving any outer disorder and the sampled prefix's schedule.
MAESTRO_TRANSFORM_API std::optional<TerminalExpectation> PrepareTerminalNoiseExpectation(const noise::NoiseInjectionPlan &plan, NoiseLayer layer,
                                                                                         const std::shared_ptr<Circuits::Circuit<double>> &runtime_circuit,
                                                                                         const std::vector<std::string> &observables,
                                                                                         const SimulatorConfig &config);
MAESTRO_TRANSFORM_API EstimationResult EstimateTerminalExpectation(const TerminalExpectation &plan, const SimulatorConfig &config,
                                                                   EstimationCache *cache = nullptr);
} // namespace MaestroExecution
