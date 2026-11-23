#pragma once

#include "Estimation.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace noise
{
class NoiseInjectionPlan;
}

namespace MaestroExecution
{
enum class NoiseLayer;

// Each folded observable is a linear combination of prefix_observables. The terms of
// a row must be summed on the same realization before computing its trajectory variance.
struct TerminalExpectation
{
    std::shared_ptr<Circuits::Circuit<double>> prefix;
    std::shared_ptr<Circuits::Circuit<double>> full_circuit;
    std::vector<std::string> prefix_observables;
    std::vector<std::vector<std::pair<size_t, double>>> rows;
    std::vector<size_t> sampled_indices;
    std::vector<std::string> sampled_observables;
    // First channel after an unbroken sampled prefix, when one evolution can serve both subsets.
    std::optional<size_t> prefix_read_position;
    size_t original_width = 0;
    bool stochastic_prefix = false;

    bool RequiresTrajectories() const
    {
        return stochastic_prefix || !sampled_indices.empty();
    }

    bool IsExactRow(size_t i) const
    {
        return !stochastic_prefix && !std::binary_search(sampled_indices.begin(), sampled_indices.end(), i);
    }

    MAESTRO_TRANSFORM_API std::vector<double> Combine(const std::vector<double> &prefix_values, const std::vector<double> &sampled_values,
                                                      bool bounds = false) const;
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
// For a mixed request with a deterministic prefix, evaluate its folded subset once before starting the trajectory workers.
MAESTRO_TRANSFORM_API std::optional<EstimationResult> EstimateDeterministicTerminalPrefix(const TerminalExpectation &plan, const SimulatorConfig &config);
// Observe a group during execution while retaining the caller's original circuits in the estimation plan.
EstimationResult EstimateCircuitTrajectory(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                           const SimulatorConfig &config, EstimationCache *cache,
                                           const std::function<std::shared_ptr<Circuits::Circuit<double>>(const EstimationGroup &)> &execution_circuit);
MAESTRO_TRANSFORM_API EstimationResult EstimateTerminalExpectation(const TerminalExpectation &plan, const SimulatorConfig &config,
                                                                   EstimationCache *cache = nullptr, const EstimationResult *deterministic_prefix = nullptr,
                                                                   const std::shared_ptr<Circuits::Circuit<double>> &sampled_circuit = {});
} // namespace MaestroExecution
