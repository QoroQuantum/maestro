// Monte Carlo noisy expectation values and fidelities over independent noise
// realizations, run on realization workers. Shared by C++ callers and the
// Python bindings. This header has no Python dependency.
#pragma once

#include "Estimation.h"
#include "SampleRecords.h"

#include <optional>
#include <string>
#include <vector>

namespace MaestroExecution
{
// The circuit without its own channel operations: the noiseless baseline,
// which a pure-state backend could only sample per realization.
MAESTRO_TRANSFORM_API RecordCircuitPtr WithoutCircuitChannels(const RecordCircuitPtr &circuit);

struct NoisyEstimate
{
    // Means over the realizations, and the noiseless baseline's values.
    std::vector<double> values;
    std::vector<double> ideal_values;
    // Mean of the realizations' truncation bounds.
    std::vector<double> error_bounds;
    size_t realizations = 0;
    bool auto_reduced = false;
    size_t bounded_observables = 0;
    // Reduction-plan reuses. Workers start from a cache warmed by realization
    // 0, so this matches a serial loop unless the noise changes the circuit's
    // multi-qubit topology between realizations.
    size_t cone_cache_hits = 0;
    size_t qubits_before = 0;
    size_t qubits_after = 0;
    size_t clusters_count = 0;
    double mean_cluster_qubits = 0;
    double reduction_time = 0;
    // Methods of the last realization's clusters.
    std::vector<Simulators::SimulationType> cluster_methods;
    // The noiseless baseline's execution (simulator, method, GPU device).
    EstimationResult ideal;
    // The realizations only, excluding the baseline.
    double time_taken = 0;
};

// Averages expectation values over noise_realizations independent injections
// of the circuit's terminal-measurement-free part. noise_seed (32 bits) seeds
// injection and, unless config.seed is set, each realization's simulator.
// Realizations run on workers within max_simulators and the realization memory
// budget; each keeps its seed and the values are summed in realization order,
// so results do not depend on the worker count. Empty observables return the
// baseline's metadata only. Emits no warnings: see ThermalApproximationWarning.
MAESTRO_TRANSFORM_API NoisyEstimate EstimateNoisy(const RecordCircuitPtr &circuit, const std::vector<std::string> &observables,
                                                  const noise::NoiseModel &noise_model, size_t noise_realizations, const SimulatorConfig &config,
                                                  std::optional<uint64_t> noise_seed, NoiseLayer layer, std::optional<bool> exact_channels = std::nullopt);

// |<ideal|noisy>|^2 of one realization: the noisy circuit without its terminal
// measurements followed by the ideal circuit's adjoint, read as P(|0...0>).
MAESTRO_TRANSFORM_API double RealizationFidelity(const RecordCircuitPtr &ideal, const RecordCircuitPtr &noisy, const SimulatorConfig &config);

struct NoisyFidelityResult
{
    double fidelity = 0;
    double std_error = 0;
    size_t realizations = 0;
    double time_taken = 0;
};

// Mean fidelity of noise_realizations combined-noise realizations of the
// circuit to its channel-free ideal, with the standard error of that mean.
// Realizations run on workers; each keeps its seeds, so results do not depend
// on the worker count.
MAESTRO_TRANSFORM_API NoisyFidelityResult NoisyFidelity(const RecordCircuitPtr &circuit, const noise::NoiseModel &noise_model, size_t noise_realizations,
                                                        const SimulatorConfig &config, std::optional<uint64_t> noise_seed);
} // namespace MaestroExecution
