#include "Execution/NoisyEstimation.h"
#include "core.h"
#include "samples.h"

namespace maestro_bindings
{
using MaestroExecution::NoiseLayer;

// Warn at the execution boundary, while holding the GIL, rather than once
// per injected realization, for the relaxation the circuit draws. The shared
// model retains its calibrated channels.
void warn_thermal_approximation(const noise::NoiseModel &noise_model, const SimulatorConfig &config, const CircuitPtr &circuit)
{
    const auto message = MaestroExecution::ThermalApproximationWarning(noise_model, MaestroExecution::UsesExactQuantumChannels(config), *circuit);
    if (message.empty())
        return;
    if (PyErr_WarnEx(PyExc_RuntimeWarning, message.c_str(), 1) < 0)
        throw nb::python_error();
}

// One realization drawn from the plan, with this configuration's channel rule.
static CircuitPtr inject_planned(NoiseLayer layer, const noise::NoiseInjectionPlan &plan, std::mt19937 &rng, const SimulatorConfig &config)
{
    return MaestroExecution::InjectPlanned(layer, plan, rng, MaestroExecution::UsesExactQuantumChannels(config));
}

void require_realizations(std::optional<int> noise_realizations)
{
    if (noise_realizations && *noise_realizations < 1)
        throw nb::value_error("noise_realizations must be >= 1.");
}

static std::optional<size_t> requested_realizations(std::optional<int> requested)
{
    require_realizations(requested);
    return requested ? std::optional<size_t>(static_cast<size_t>(*requested)) : std::nullopt;
}

void AddNoisePolicyMetadata(nb::dict &out, const MaestroExecution::NoiseRealizationPolicy &policy)
{
    out["noise_realizations_requested"] = policy.requested ? nb::cast(*policy.requested) : nb::none();
    out["noise_realizations_policy"] = policy.requested ? "explicit" : "auto";
    out["noise_evaluation"] = policy.exact_channels ? "exact_channels" : "trajectories";
    out["stochastic_noise"] = policy.stochastic_noise;
    out["stochastic_execution"] = policy.stochastic_execution;
}

size_t ResolveSamplingRealizations(const CircuitPtr &circuit, const nb::object &noise_model, const SimulatorConfig &config, int fallback, bool estimation)
{
    require_circuit(circuit);
    if (fallback < 1)
        throw nb::value_error("fallback must be >= 1.");
    const noise::NoiseModel empty;
    const auto &model = noise_model.is_none() ? empty : nb::cast<const noise::NoiseModel &>(noise_model);
    return MaestroExecution::ResolveNoiseRealizations(std::nullopt, model, config, NoiseLayer::kCombined, static_cast<size_t>(fallback), circuit, estimation)
        .realizations;
}

// Shots are split across min(shots, noise_realizations) batches, each with its
// own injected noise; the result reports the number of batches used. The
// algorithm is MaestroExecution::ExecuteNoisy, shared with C++ callers.
static nb::dict execute_noise_batches(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                                      std::optional<int> noise_realizations, std::optional<uint64_t> noise_seed, NoiseLayer layer, const char *noise_type,
                                      bool return_samples)
{
    if (shots < 1)
        throw nb::value_error("shots must be >= 1.");
    require_realizations(noise_realizations);
    MaestroExecution::NoisyExecutionOptions options;
    options.layer = layer;
    options.rows = return_samples;
    MaestroExecution::ShotRecords records;
    {
        nb::gil_scoped_release release;
        records = MaestroExecution::ExecuteNoisy(circuit, noise_model, config, static_cast<size_t>(shots), requested_realizations(noise_realizations),
                                                 noise_seed, options);
    }

    nb::dict out;
    if (return_samples)
    {
        out["samples"] = MakeSamples(std::move(records.rows), records.shots, records.width);
        out["sampling_strategy"] = records.sampling_strategy;
    }
    else
    {
        if (!records.sampling_strategy.empty())
            out["sampling_strategy"] = records.sampling_strategy;
        out["counts"] = CountsDict(records.counts);
    }
    if (records.gpu_device >= 0)
        out["gpu_device"] = records.gpu_device;
    if (records.max_bond_dimension)
        out["max_bond_dim_reached"] = records.max_bond_dimension;
    out["time_taken"] = records.time_taken;
    out["simulator"] = records.simulator;
    out["method"] = records.method;
    out["noise_realizations"] = records.realizations;
    AddNoisePolicyMetadata(out, records.policy);
    out["sampling_components"] = records.components;
    out["largest_component_qubits"] = records.largest_component;
    if (noise_type)
        out["noise_type"] = noise_type;
    return out;
}

// The noiseless baseline of a circuit drops its own channel operations, which a
// pure-state backend can only sample per realization, never run directly.
CircuitPtr without_circuit_channels(const CircuitPtr &circuit)
{
    return MaestroExecution::WithoutCircuitChannels(circuit);
}

// Averages expectation values over noise_realizations independent injections;
// the loop is MaestroExecution::EstimateNoisy, which runs realization workers.
static nb::dict estimate_noise_realizations(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model,
                                            std::optional<int> noise_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed,
                                            NoiseLayer layer, const char *noise_type)
{
    require_realizations(noise_realizations);
    auto paulis = ParseObservables(observables);
    MaestroExecution::NoisyEstimate result;
    {
        nb::gil_scoped_release release;
        result = MaestroExecution::EstimateNoisy(circuit, paulis, noise_model, requested_realizations(noise_realizations), config, noise_seed, layer);
    }
    nb::dict out;
    out["expectation_values"] = nb::cast(result.values);
    out["ideal_expectation_values"] = nb::cast(result.ideal_values);
    out["time_taken"] = result.time_taken;
    out["simulator"] = result.ideal.simulator;
    out["method"] = result.ideal.method;
    // GPU estimates resolved without simulation retain -1 (no device used).
    if (result.ideal.simulator == Simulators::SimulatorType::kGpuSim || result.ideal.gpu_device >= 0)
        out["gpu_device"] = result.ideal.gpu_device;
    out["noise_realizations"] = result.realizations;
    AddNoisePolicyMetadata(out, result.policy);
    const bool stochastic = result.policy.stochastic_noise || result.policy.stochastic_execution;
    out["estimate_evaluation"] = stochastic ? "trajectory_average" : "deterministic";
    out["realization_standard_errors"] = stochastic && result.realizations < 2 ? nb::object(nb::none()) : nb::cast(result.standard_errors);
    out["auto_reduced"] = result.auto_reduced;
    out["expectation_error_bounds"] = nb::cast(result.error_bounds);
    out["bounded_observables"] = result.bounded_observables;
    out["cone_cache_hits"] = result.cone_cache_hits;
    out["qubits_before"] = result.qubits_before;
    out["qubits_after"] = result.qubits_after;
    out["clusters_count"] = result.clusters_count;
    out["largest_cluster_qubits"] = result.qubits_after;
    out["mean_cluster_qubits"] = result.mean_cluster_qubits;
    out["reduction_time"] = result.reduction_time;
    nb::list methods;
    for (auto method : result.cluster_methods)
        methods.append(method);
    out["cluster_methods"] = methods;
    if (noise_type)
        out["noise_type"] = noise_type;
    return out;
}

void require_circuit(const CircuitPtr &circuit)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");
}

// The coherent injector applies only coherent rotations (plus readout), so
// any other layer would be silently dropped; reject it as the JSON API does.
static void require_coherent(const noise::NoiseModel &noise_model)
{
    if (!noise_model.has_coherent())
        throw nb::value_error("NoiseModel has no coherent noise set. Use "
                              "set_coherent_depolarizing(), set_coherent_rotation(), etc.");
    if (!noise_model.inject_coherent_noise_covers_model())
        throw nb::value_error("coherent_execute/coherent_estimate apply only coherent and readout noise, "
                              "but this NoiseModel has other layers. Use full_noise_execute or "
                              "full_noise_estimate to apply all of them.");
}

// The Pauli/T1 injector does not apply coherent, correlated or crosstalk
// layers, so a model with them would silently simulate something else.
static void require_inject_noise_covers(const noise::NoiseModel &noise_model)
{
    if (!noise_model.inject_noise_covers_model())
        throw nb::value_error("noisy_execute/noisy_estimate_montecarlo do not apply coherent, correlated "
                              "or crosstalk noise, which this NoiseModel has. Use full_noise_execute or "
                              "full_noise_estimate to apply all layers.");
}

void require_any_noise(const noise::NoiseModel &noise_model)
{
    if (!noise_model.has_any())
        throw nb::value_error("NoiseModel has no noise configured.");
}

// Channel operations in the circuit itself (e.g. translated Stim noise) are
// noise in their own right, even under an empty NoiseModel.
bool has_circuit_channels(const CircuitPtr &circuit)
{
    return circuit->HasQuantumChannels();
}

nb::dict NoisyExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                      std::optional<int> noise_realizations, std::optional<uint64_t> noise_seed, bool return_samples)
{
    require_circuit(circuit);
    require_inject_noise_covers(noise_model);
    warn_thermal_approximation(noise_model, config, circuit);
    return execute_noise_batches(circuit, noise_model, config, shots, noise_realizations, noise_seed, NoiseLayer::kPauli, nullptr, return_samples);
}

nb::dict CoherentExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                         std::optional<int> noise_realizations, std::optional<uint64_t> noise_seed, bool return_samples)
{
    require_circuit(circuit);
    require_coherent(noise_model);
    return execute_noise_batches(circuit, noise_model, config, shots, noise_realizations, noise_seed, NoiseLayer::kCoherent, "coherent", return_samples);
}

nb::dict FullNoiseExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                          std::optional<int> noise_realizations, std::optional<uint64_t> noise_seed, bool return_samples)
{
    require_circuit(circuit);
    if (!has_circuit_channels(circuit))
        require_any_noise(noise_model);
    warn_thermal_approximation(noise_model, config, circuit);
    return execute_noise_batches(circuit, noise_model, config, shots, noise_realizations, noise_seed, NoiseLayer::kCombined, "combined", return_samples);
}

nb::dict NoisyEstimateMonteCarlo(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model,
                                 std::optional<int> noise_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_inject_noise_covers(noise_model);
    warn_thermal_approximation(noise_model, config, circuit);
    return estimate_noise_realizations(circuit, observables, noise_model, noise_realizations, config, noise_seed, NoiseLayer::kPauli, nullptr);
}

nb::dict CoherentEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, std::optional<int> noise_realizations,
                          const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_coherent(noise_model);
    return estimate_noise_realizations(circuit, observables, noise_model, noise_realizations, config, noise_seed, NoiseLayer::kCoherent, "coherent");
}

nb::dict FullNoiseEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model,
                           std::optional<int> noise_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    if (!has_circuit_channels(circuit))
        require_any_noise(noise_model);
    warn_thermal_approximation(noise_model, config, circuit);
    return estimate_noise_realizations(circuit, observables, noise_model, noise_realizations, config, noise_seed, NoiseLayer::kCombined, "combined");
}

// Ideal expectation values scaled by compute_damping, which only the all-gates
// Pauli layer contributes to; any other layer would be silently ignored.
nb::dict NoisyEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, const SimulatorConfig &config)
{
    require_circuit(circuit);
    if (!noise_model.compute_damping_covers_model())
        throw nb::value_error("noisy_estimate models only the all-gates Pauli layer (set_depolarizing, "
                              "set_dephasing, set_bit_flip, set_qubit_noise), but this NoiseModel has other "
                              "layers. Use noisy_estimate_montecarlo or full_noise_estimate instead.");
    // Damping scales one deterministic expectation; it cannot describe noise
    // acting between measurement branches (the native estimate rejects it too).
    if (MaestroExecution::HasStochasticEvolution(circuit, MaestroExecution::UsesExactQuantumChannels(config)))
        throw nb::value_error("noisy_estimate (analytical damping) requires deterministic evolution; mid-circuit measurements and classical "
                              "control need noisy_estimate_montecarlo or full_noise_estimate to average measurement branches.");
    auto paulis = ParseObservables(observables);
    nb::dict result = estimate_core(circuit, paulis, config);

    nb::list ideal = nb::cast<nb::list>(result["expectation_values"]);
    nb::list noisy_vals;
    for (size_t i = 0; i < paulis.size(); ++i)
        noisy_vals.append(noise_model.compute_damping(paulis[i]) * nb::cast<double>(ideal[i]));

    result["expectation_values"] = noisy_vals;
    result["ideal_expectation_values"] = ideal;
    return result;
}

nb::dict NoisyFidelity(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, std::optional<int> noise_realizations, const SimulatorConfig &config,
                       std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_realizations(noise_realizations);
    if (!has_circuit_channels(circuit))
        require_any_noise(noise_model);
    warn_thermal_approximation(noise_model, config, circuit);
    MaestroExecution::NoisyFidelityResult result;
    {
        nb::gil_scoped_release release;
        result = MaestroExecution::NoisyFidelity(circuit, noise_model, requested_realizations(noise_realizations), config, noise_seed);
    }
    nb::dict out;
    out["fidelity"] = result.fidelity;
    out["infidelity"] = 1.0 - result.fidelity;
    const bool stochastic = result.policy.stochastic_noise || result.policy.stochastic_execution;
    out["std_error"] = stochastic && result.realizations < 2 ? nb::object(nb::none()) : nb::cast(result.std_error);
    out["time_taken"] = result.time_taken;
    out["noise_realizations"] = result.realizations;
    AddNoisePolicyMetadata(out, result.policy);
    out["estimate_evaluation"] = stochastic ? "trajectory_average" : "deterministic";
    return out;
}

} // namespace maestro_bindings
