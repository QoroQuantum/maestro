#pragma once
#include "Execution/Estimation.h"
#include "Execution/SampleRecords.h"
#include "common.h"

// Shared tail of the noisy execute docstrings (functions and methods).
#define MAESTRO_NOISY_EXECUTE_DOC                                                                                                                              \
    "\n\nnoise_realizations=None selects one realization for deterministic exact channels (density matrix, "                                                   \
    "MPO, and the QCSim Pauli propagator for the noise it applies exactly by default) and "                                                                    \
    "64 otherwise. An explicit positive count is honored, capped to shots. "                                                                                   \
    "noise_seed (also accepted as seed) is an unsigned 32-bit integer. It seeds noise injection and, "                                                         \
    "unless config.seed is set, measurement and readout. return_samples=True returns 'samples', a "                                                            \
    "(shots, clbits) uint8 array of shot records, instead of 'counts'. Results report 'sampling_strategy' "                                                    \
    "when a dedicated sampler ran: 'reference_frame_v1' (noisy-Clifford frames), 'extended_frame_v1' "                                                         \
    "(batched extended-stabilizer frames), 'trajectory_v1' (each realization executed directly) or "                                                           \
    "'terminal_v1'. SimulatorConfig.counts_policy ('auto', 'legacy', "                                                                                         \
    "'frame_v1', 'multinomial_v1') controls count sampling. 'auto' allows adaptive multinomial histograms with or without "                                    \
    "a seed. Repeating the same request with the same seeds and configuration reproduces its histogram on the same build."

namespace maestro_bindings
{
using CircuitPtr = std::shared_ptr<Circuits::Circuit<double>>;
using OperationPtr = std::shared_ptr<Circuits::IOperation<double>>;

// Noise seeding and injection shared with the native API (Execution/SampleRecords.h).
using MaestroExecution::CloneSamplingCircuitForExecution;
using MaestroExecution::MakeNoiseRng;
using MaestroExecution::NoiseExecutionConfig;
using MaestroExecution::RequireNoiseSeed;

void require_circuit(const CircuitPtr &circuit);
void require_realizations(std::optional<int> noise_realizations);
void AddNoisePolicyMetadata(nb::dict &out, const MaestroExecution::NoiseRealizationPolicy &policy);
size_t ResolveSamplingRealizations(const CircuitPtr &circuit, const nb::object &noise_model, const SimulatorConfig &config, int fallback,
                                   bool estimation = false);
void require_any_noise(const noise::NoiseModel &noise_model);
bool has_circuit_channels(const CircuitPtr &circuit);
CircuitPtr without_circuit_channels(const CircuitPtr &circuit);

std::vector<std::string> ParseObservables(const nb::object &observables);
// Emits ThermalApproximationWarning as a Python RuntimeWarning, once per call.
// The configuration form decides as an executor that takes the configuration
// as given does; a checkpoint passes the one it resolved. noisy_fidelity
// resolves a candidate list inside the call, and then passes the evaluation
// the call reports instead.
void warn_thermal_approximation(const noise::NoiseModel &noise_model, const SimulatorConfig &config, const CircuitPtr &circuit);
void warn_thermal_approximation(const noise::NoiseModel &noise_model, bool exact_channels, const CircuitPtr &circuit);
nb::dict execute_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots, bool return_samples = false);
nb::dict estimate_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const std::vector<std::string> &paulis, const SimulatorConfig &config,
                       MaestroExecution::EstimationCache *cache = nullptr);
std::vector<std::complex<double>> statevector_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config);
nb::object mirror_fidelity_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots, bool full_amplitude,
                                const std::optional<nb::object> &observables = std::nullopt);
std::complex<double> inner_product_core(const std::shared_ptr<Circuits::Circuit<double>> &circuit_1,
                                        const std::shared_ptr<Circuits::Circuit<double>> &circuit_2, const SimulatorConfig &config);
nb::dict incremental_evolve_core(std::shared_ptr<Circuits::Circuit<double>> init_circuit, std::shared_ptr<Circuits::Circuit<double>> trotter_step,
                                 const std::vector<int> &measure_at_steps, const std::vector<std::string> &paulis, const SimulatorConfig &config);
nb::dict NoisyExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                      std::optional<int> noise_realizations, std::optional<uint64_t> noise_seed, bool return_samples = false);
nb::dict CoherentExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                         std::optional<int> noise_realizations, std::optional<uint64_t> noise_seed, bool return_samples = false);
nb::dict FullNoiseExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots,
                          std::optional<int> noise_realizations, std::optional<uint64_t> noise_seed, bool return_samples = false);
nb::dict NoisyEstimateMonteCarlo(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model,
                                 std::optional<int> noise_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed);
nb::dict CoherentEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, std::optional<int> noise_realizations,
                          const SimulatorConfig &config, std::optional<uint64_t> noise_seed);
nb::dict FullNoiseEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model,
                           std::optional<int> noise_realizations, const SimulatorConfig &config, std::optional<uint64_t> noise_seed);
nb::dict NoisyEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, const SimulatorConfig &config);
nb::dict NoisyFidelity(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, std::optional<int> noise_realizations, const SimulatorConfig &config,
                       std::optional<uint64_t> noise_seed);
} // namespace maestro_bindings
