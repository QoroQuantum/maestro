#include "core.h"

namespace maestro_bindings
{

// Density-matrix and QCSim MPO configurations can retain the full ensemble in
// one state. Route their Markovian noise through circuit channel operations;
// pure-state/MPS configurations retain the legacy trajectory implementation.
static bool uses_exact_quantum_channels(const SimulatorConfig &config)
{
    // Composite simulators still contain pure-state components and cannot
    // represent ensemble channels exactly as a single mixed state.
    const bool qcsim = config.simulator_type == Simulators::SimulatorType::kQCSim;
    const bool gpu = config.simulator_type == Simulators::SimulatorType::kGpuSim;
#ifndef NO_QISKIT_AER
    const bool aer = config.simulator_type == Simulators::SimulatorType::kQiskitAer;
#else
    const bool aer = false;
#endif

    return (qcsim && (config.simulation_type == Simulators::SimulationType::kDensityMatrix ||
                      config.simulation_type == Simulators::SimulationType::kMatrixProductOperator)) ||
           (aer && config.simulation_type == Simulators::SimulationType::kDensityMatrix) ||
           (gpu && (config.simulation_type == Simulators::SimulationType::kDensityMatrix ||
                    config.simulation_type == Simulators::SimulationType::kMatrixProductOperator));
}

// Warn at the execution boundary, while holding the GIL, rather than once
// per injected realization. The shared model retains its calibrated channels.
void warn_thermal_approximation(const noise::NoiseModel &noise_model, const SimulatorConfig &config)
{
    if (uses_exact_quantum_channels(config) || noise_model.has_additional_quantum_channels())
        return;
    const auto qubits = noise_model.thermal_approximation_qubits();
    if (qubits.empty())
        return;
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
    if (PyErr_WarnEx(PyExc_RuntimeWarning, message.str().c_str(), 1) < 0)
        throw nb::python_error();
}

// The noise seed is 32 bits, as in the JSON API's noise.seed.
void RequireNoiseSeed(std::optional<uint64_t> noise_seed)
{
    if (noise_seed && *noise_seed > UINT32_MAX)
        throw nb::value_error("noise_seed must fit in 32 bits");
}

// An omitted noise seed falls back to the config seed's low 32 bits, as in the
// JSON API, so SimulatorConfig(seed=...) alone reproduces the injected noise.
// All MPI ranks must submit the same stochastic circuit, so MPI resolves an
// unseeded call once and shares it with measurement/readout execution.
std::mt19937 MakeNoiseRng(const SimulatorConfig &config, std::optional<uint64_t> &noise_seed)
{
    RequireNoiseSeed(noise_seed);
    if (!noise_seed && config.seed)
        noise_seed = *config.seed & UINT32_MAX;
    else if (!noise_seed && config.simulator_type == Simulators::SimulatorType::kDistMpiGpuSim)
        noise_seed = Simulators::GenerateRandomSeed(config.simulator_type, config.distributed_options) & UINT32_MAX;
    if (noise_seed)
        return std::mt19937(static_cast<uint32_t>(*noise_seed));
    return std::mt19937(std::random_device{}());
}

// Keep simulator randomness separate from circuit-noise injection. An explicit
// config seed takes precedence; otherwise the noise seed also seeds
// measurement/readout and reset collapse. Every batch needs its own stream,
// including one-shot executions and expectation-value realizations.
SimulatorConfig NoiseExecutionConfig(const SimulatorConfig &config, std::optional<uint64_t> noise_seed, uint64_t batch)
{
    auto execution_config = config;
    // Noisy workflows retain their requested backend, even for realizations
    // in which stochastic injection happens to insert no error.
    execution_config.auto_reduce = false;
    if (!execution_config.seed && noise_seed)
        execution_config.seed = *noise_seed;
    if (execution_config.seed)
        execution_config.seed = Simulators::IState::DeriveSeed(*execution_config.seed, batch);
    return execution_config;
}

std::shared_ptr<Circuits::Circuit<double>> inject_noise_for_config(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                   const noise::NoiseModel &noise_model, std::mt19937 &rng, const SimulatorConfig &config)
{
    auto noisy = uses_exact_quantum_channels(config) ? noise::inject_exact_noise(circuit, noise_model) : noise::inject_noise(circuit, noise_model, rng);
    noise::attach_readout_error(noisy, noise_model);
    return noisy;
}

std::shared_ptr<Circuits::Circuit<double>> inject_combined_noise_for_config(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                            const noise::NoiseModel &noise_model, std::mt19937 &rng,
                                                                            const SimulatorConfig &config)
{
    auto noisy = uses_exact_quantum_channels(config) ? noise::inject_combined_noise_exact(circuit, noise_model, rng)
                                                     : noise::inject_combined_noise(circuit, noise_model, rng);
    noise::attach_readout_error(noisy, noise_model);
    return noisy;
}

using NoiseInjector = CircuitPtr (*)(const CircuitPtr &, const noise::NoiseModel &, std::mt19937 &, const SimulatorConfig &);

static CircuitPtr inject_coherent_for_config(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, std::mt19937 &rng, const SimulatorConfig &)
{
    auto noisy = noise::inject_coherent_noise(circuit, noise_model, rng);
    noise::attach_readout_error(noisy, noise_model);
    return noisy;
}

void require_realizations(int noise_realizations)
{
    if (noise_realizations < 1)
        throw nb::value_error("noise_realizations must be >= 1.");
}

// Shots are split across min(shots, noise_realizations) batches, each with its
// own injected noise; the result reports the number of batches used.
static nb::dict execute_noise_batches(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                                      int noise_realizations, std::optional<uint64_t> noise_seed, NoiseInjector inject, const char *noise_type)
{
    if (shots < 1)
        throw nb::value_error("shots must be >= 1.");
    require_realizations(noise_realizations);
    auto rng = MakeNoiseRng(config, noise_seed);
    const int batches = std::min(shots, noise_realizations);
    const int base_batch = shots / batches;
    const int leftover = shots % batches;

    std::unordered_map<std::string, size_t> combined;
    size_t sampling_components = 0, largest_component = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < batches; ++b)
    {
        auto noisy = inject(circuit, noise_model, rng, config);
        nb::dict r = execute_core(noisy, NoiseExecutionConfig(config, noise_seed, b), base_batch + (b < leftover ? 1 : 0));
        sampling_components = std::max(sampling_components, nb::cast<size_t>(r["sampling_components"]));
        largest_component = std::max(largest_component, nb::cast<size_t>(r["largest_component_qubits"]));
        for (auto item : nb::cast<nb::dict>(r["counts"]))
            combined[nb::cast<std::string>(nb::str(item.first))] += nb::cast<size_t>(item.second);
    }
    auto end = std::chrono::high_resolution_clock::now();

    nb::dict py_counts;
    for (const auto &[k, v] : combined)
        py_counts[k.c_str()] = v;

    nb::dict out;
    out["counts"] = py_counts;
    out["time_taken"] = std::chrono::duration<double>(end - start).count();
    out["simulator"] = (int)config.simulator_type;
    out["method"] = (int)config.simulation_type;
    out["noise_realizations"] = batches;
    out["sampling_components"] = sampling_components;
    out["largest_component_qubits"] = largest_component;
    if (noise_type)
        out["noise_type"] = noise_type;
    return out;
}

// The noiseless baseline of a circuit drops its own channel operations, which a
// pure-state backend can only sample per realization, never run directly.
CircuitPtr without_circuit_channels(const CircuitPtr &circuit)
{
    auto out = std::make_shared<Circuits::Circuit<double>>();
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() != Circuits::OperationType::kQuantumChannel)
            out->AddOperation(op);
    return out;
}

// Averages expectation values over noise_realizations independent injections.
static nb::dict estimate_noise_realizations(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model,
                                            int noise_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed, NoiseInjector inject,
                                            const char *noise_type)
{
    require_realizations(noise_realizations);
    auto paulis = ParseObservables(observables);
    auto rng = MakeNoiseRng(config, noise_seed);
    const size_t n_obs = paulis.size();
    if (!n_obs)
    {
        auto out = estimate_core(without_circuit_channels(circuit), paulis, config);
        out["ideal_expectation_values"] = nb::list();
        out["noise_realizations"] = noise_realizations;
        out["cone_cache_hits"] = 0;
        if (noise_type)
            out["noise_type"] = noise_type;
        return out;
    }
    std::vector<double> sum_vals(n_obs, 0.0), error_bounds(n_obs, 0.0);
    size_t bounded_observables = 0;
    MaestroExecution::EstimationCache cache;

    auto clean_circuit = circuit->RemoveEndMeasurements();

    auto start = std::chrono::high_resolution_clock::now();
    bool any_reduced = false;
    int qubits_before = 0;
    int qubits_after = 0;
    size_t clusters_count = 0, total_clusters = 0;
    double total_cluster_qubits = 0, reduction_time = 0;
    nb::list cluster_methods;
    for (int r = 0; r < noise_realizations; ++r)
    {
        auto noisy = inject(clean_circuit, noise_model, rng, config);
        nb::dict result = estimate_core(noisy, paulis, NoiseExecutionConfig(config, noise_seed, r), &cache);
        if (result.contains("auto_reduced") && nb::cast<bool>(result["auto_reduced"]))
            any_reduced = true;
        if (result.contains("qubits_before"))
            qubits_before = std::max(qubits_before, nb::cast<int>(result["qubits_before"]));
        if (result.contains("qubits_after"))
            qubits_after = std::max(qubits_after, nb::cast<int>(result["qubits_after"]));
        const auto count = nb::cast<size_t>(result["clusters_count"]);
        clusters_count = std::max(clusters_count, count);
        total_clusters += count;
        total_cluster_qubits += count * nb::cast<double>(result["mean_cluster_qubits"]);
        reduction_time += nb::cast<double>(result["reduction_time"]);
        cluster_methods = nb::cast<nb::list>(result["cluster_methods"]);
        bounded_observables = std::max(bounded_observables, nb::cast<size_t>(result["bounded_observables"]));
        const auto bounds = nb::cast<std::vector<double>>(result["expectation_error_bounds"]);
        for (size_t i = 0; i < n_obs; ++i)
            error_bounds[i] += bounds[i] / noise_realizations;
        nb::list ev = nb::cast<nb::list>(result["expectation_values"]);
        for (size_t i = 0; i < n_obs; ++i)
            sum_vals[i] += nb::cast<double>(ev[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    nb::dict ideal_result = estimate_core(without_circuit_channels(clean_circuit), paulis, config);
    nb::list noisy_vals, ideal_vals;
    nb::list ideal_ev = nb::cast<nb::list>(ideal_result["expectation_values"]);
    for (size_t i = 0; i < n_obs; ++i)
    {
        noisy_vals.append(sum_vals[i] / noise_realizations);
        ideal_vals.append(nb::cast<double>(ideal_ev[i]));
    }

    nb::dict out;
    out["expectation_values"] = noisy_vals;
    out["ideal_expectation_values"] = ideal_vals;
    out["time_taken"] = std::chrono::duration<double>(end - start).count();
    out["simulator"] = ideal_result["simulator"];
    out["method"] = ideal_result["method"];
    if (ideal_result.contains("gpu_device"))
        out["gpu_device"] = ideal_result["gpu_device"];
    out["noise_realizations"] = noise_realizations;
    out["auto_reduced"] = any_reduced;
    out["expectation_error_bounds"] = nb::cast(error_bounds);
    out["bounded_observables"] = bounded_observables;
    out["cone_cache_hits"] = cache.hits;
    out["qubits_before"] = qubits_before;
    out["qubits_after"] = qubits_after;
    out["clusters_count"] = clusters_count;
    out["largest_cluster_qubits"] = qubits_after;
    out["mean_cluster_qubits"] = total_clusters ? total_cluster_qubits / total_clusters : 0;
    out["reduction_time"] = reduction_time;
    out["cluster_methods"] = cluster_methods;
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
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() == Circuits::OperationType::kQuantumChannel)
            return true;
    return false;
}

nb::dict NoisyExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots, int noise_realizations,
                      std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_inject_noise_covers(noise_model);
    warn_thermal_approximation(noise_model, config);
    return execute_noise_batches(circuit, noise_model, config, shots, noise_realizations, noise_seed, inject_noise_for_config, nullptr);
}

nb::dict CoherentExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots, int noise_realizations,
                         std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_coherent(noise_model);
    return execute_noise_batches(circuit, noise_model, config, shots, noise_realizations, noise_seed, inject_coherent_for_config, "coherent");
}

nb::dict FullNoiseExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots, int noise_realizations,
                          std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    if (!has_circuit_channels(circuit))
        require_any_noise(noise_model);
    warn_thermal_approximation(noise_model, config);
    return execute_noise_batches(circuit, noise_model, config, shots, noise_realizations, noise_seed, inject_combined_noise_for_config, "combined");
}

nb::dict NoisyEstimateMonteCarlo(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, int noise_realizations,
                                 const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_inject_noise_covers(noise_model);
    warn_thermal_approximation(noise_model, config);
    return estimate_noise_realizations(circuit, observables, noise_model, noise_realizations, config, noise_seed, inject_noise_for_config, nullptr);
}

nb::dict CoherentEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, int noise_realizations,
                          const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_coherent(noise_model);
    return estimate_noise_realizations(circuit, observables, noise_model, noise_realizations, config, noise_seed, inject_coherent_for_config, "coherent");
}

nb::dict FullNoiseEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, int noise_realizations,
                           const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    if (!has_circuit_channels(circuit))
        require_any_noise(noise_model);
    warn_thermal_approximation(noise_model, config);
    return estimate_noise_realizations(circuit, observables, noise_model, noise_realizations, config, noise_seed, inject_combined_noise_for_config, "combined");
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

nb::dict NoisyFidelity(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, int noise_realizations, const SimulatorConfig &config,
                       std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
    require_realizations(noise_realizations);
    if (!has_circuit_channels(circuit))
        require_any_noise(noise_model);
    const auto ideal = without_circuit_channels(circuit);

    warn_thermal_approximation(noise_model, config);
    auto rng = MakeNoiseRng(config, noise_seed);
    double sum_fid = 0.0;
    double sum_fid_sq = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < noise_realizations; ++r)
    {
        auto noisy = inject_combined_noise_for_config(circuit, noise_model, rng, config);
        // Reset collapse must use a fresh seed for each realization.
        auto realization_config = NoiseExecutionConfig(config, noise_seed, r);
        if (!realization_config.seed)
            realization_config.seed = rng();
        double fid = noisy_fidelity_core(ideal, noisy, realization_config);
        sum_fid += fid;
        sum_fid_sq += fid * fid;
    }
    auto end = std::chrono::high_resolution_clock::now();

    double mean_fid = sum_fid / noise_realizations;
    double var = (noise_realizations > 1) ? (sum_fid_sq - noise_realizations * mean_fid * mean_fid) / (noise_realizations - 1) : 0.0;
    double se = std::sqrt(std::max(var, 0.0) / noise_realizations);

    nb::dict out;
    out["fidelity"] = mean_fid;
    out["infidelity"] = 1.0 - mean_fid;
    out["std_error"] = se;
    out["time_taken"] = std::chrono::duration<double>(end - start).count();
    out["noise_realizations"] = noise_realizations;
    return out;
}

} // namespace maestro_bindings
