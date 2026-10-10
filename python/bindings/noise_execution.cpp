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
    return noise::inject_coherent_noise(circuit, noise_model, rng);
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
    auto start = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < batches; ++b)
    {
        auto noisy = inject(circuit, noise_model, rng, config);
        nb::dict r = execute_core(noisy, NoiseExecutionConfig(config, noise_seed, b), base_batch + (b < leftover ? 1 : 0));
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
    if (noise_type)
        out["noise_type"] = noise_type;
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
    std::vector<double> sum_vals(n_obs, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < noise_realizations; ++r)
    {
        auto noisy = inject(circuit, noise_model, rng, config);
        nb::dict result = estimate_core(noisy, paulis, NoiseExecutionConfig(config, noise_seed, r));
        nb::list ev = nb::cast<nb::list>(result["expectation_values"]);
        for (size_t i = 0; i < n_obs; ++i)
            sum_vals[i] += nb::cast<double>(ev[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();

    nb::dict ideal_result = estimate_core(circuit, paulis, config);
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
    if (noise_type)
        out["noise_type"] = noise_type;
    return out;
}

void require_circuit(const CircuitPtr &circuit)
{
    if (!circuit)
        throw nb::value_error("Circuit is null.");
}

static void require_coherent(const noise::NoiseModel &noise_model)
{
    if (!noise_model.has_coherent())
        throw nb::value_error("NoiseModel has no coherent noise set. Use "
                              "set_coherent_depolarizing(), set_coherent_rotation(), etc.");
}

void require_any_noise(const noise::NoiseModel &noise_model)
{
    if (!noise_model.has_any())
        throw nb::value_error("NoiseModel has no noise configured.");
}

nb::dict NoisyExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots, int noise_realizations,
                      std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
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
    require_any_noise(noise_model);
    warn_thermal_approximation(noise_model, config);
    return execute_noise_batches(circuit, noise_model, config, shots, noise_realizations, noise_seed, inject_combined_noise_for_config, "combined");
}

nb::dict NoisyEstimateMonteCarlo(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, int noise_realizations,
                                 const SimulatorConfig &config, std::optional<uint64_t> noise_seed)
{
    require_circuit(circuit);
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
    require_any_noise(noise_model);
    warn_thermal_approximation(noise_model, config);
    return estimate_noise_realizations(circuit, observables, noise_model, noise_realizations, config, noise_seed, inject_combined_noise_for_config, "combined");
}

} // namespace maestro_bindings
