#pragma once
#include "common.h"

namespace maestro_bindings
{
using CircuitPtr = std::shared_ptr<Circuits::Circuit<double>>;
using OperationPtr = std::shared_ptr<Circuits::IOperation<double>>;

void RequireNoiseSeed(std::optional<uint64_t> noise_seed);
void require_circuit(const CircuitPtr &circuit);
void require_realizations(int noise_realizations);
void require_any_noise(const noise::NoiseModel &noise_model);

std::vector<std::string> ParseObservables(const nb::object &observables);
void warn_thermal_approximation(const noise::NoiseModel &noise_model, const SimulatorConfig &config);
std::mt19937 MakeNoiseRng(const SimulatorConfig &config, std::optional<uint64_t> &noise_seed);
SimulatorConfig NoiseExecutionConfig(const SimulatorConfig &config, std::optional<uint64_t> noise_seed, uint64_t batch);
std::shared_ptr<Circuits::Circuit<double>> inject_noise_for_config(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                   const noise::NoiseModel &noise_model, std::mt19937 &rng, const SimulatorConfig &config);
std::shared_ptr<Circuits::Circuit<double>> inject_combined_noise_for_config(const std::shared_ptr<Circuits::Circuit<double>> &circuit,
                                                                            const noise::NoiseModel &noise_model, std::mt19937 &rng,
                                                                            const SimulatorConfig &config);
nb::dict execute_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots);
nb::dict estimate_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const std::vector<std::string> &paulis, const SimulatorConfig &config);
std::vector<std::complex<double>> statevector_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config);
double mirror_fidelity_core(std::shared_ptr<Circuits::Circuit<double>> circuit, const SimulatorConfig &config, int shots, bool full_amplitude);
std::complex<double> inner_product_core(const std::shared_ptr<Circuits::Circuit<double>> &circuit_1,
                                        const std::shared_ptr<Circuits::Circuit<double>> &circuit_2, const SimulatorConfig &config);
double noisy_fidelity_core(const std::shared_ptr<Circuits::Circuit<double>> &ideal, const std::shared_ptr<Circuits::Circuit<double>> &noisy,
                           const SimulatorConfig &config);
nb::dict incremental_evolve_core(std::shared_ptr<Circuits::Circuit<double>> init_circuit, std::shared_ptr<Circuits::Circuit<double>> trotter_step,
                                 const std::vector<int> &measure_at_steps, const std::vector<std::string> &paulis, const SimulatorConfig &config);
nb::dict NoisyExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots, int noise_realizations,
                      std::optional<uint64_t> noise_seed);
nb::dict CoherentExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots, int noise_realizations,
                         std::optional<uint64_t> noise_seed);
nb::dict FullNoiseExecute(const CircuitPtr &circuit, const noise::NoiseModel &noise_model, const SimulatorConfig &config, int shots, int noise_realizations,
                          std::optional<uint64_t> noise_seed);
nb::dict NoisyEstimateMonteCarlo(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, int noise_realizations,
                                 const SimulatorConfig &config, std::optional<uint64_t> noise_seed);
nb::dict CoherentEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, int noise_realizations,
                          const SimulatorConfig &config, std::optional<uint64_t> noise_seed);
nb::dict FullNoiseEstimate(const CircuitPtr &circuit, const nb::object &observables, const noise::NoiseModel &noise_model, int noise_realizations,
                           const SimulatorConfig &config, std::optional<uint64_t> noise_seed);
} // namespace maestro_bindings
