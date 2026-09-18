/**
 * @file NoiseAdd.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Add noise to a circuit, using the NoiseModel defined in noise.h.
 */

#pragma once

#ifndef __NOISE_ADD_H_
#define __NOISE_ADD_H_

#include "NoiseModel.h"

#include "../Network/Network.h"
#include "../Simulators/RandomSeed.h"

#include <functional>
#include <iostream>
#include <optional>
#include <sstream>

#define MAESTRO_NOISE_ADD_VERSION 3

namespace noise {

class NoiseAdd {
 public:
  NoiseAdd() : rng(std::random_device{}()) {}

  /**
   * Select the exact-channel (CPTP) realization of the Markovian noise
   * layers instead of sampled trajectories.
   *
   * Enable this when the network executes on a backend that carries a mixed
   * state -- QCSim density_matrix or matrix_product_operator, Aer
   * density_matrix -- so the noise is applied as deterministic Kraus channels
   * in one pass rather than averaged over realizations. It also unlocks the
   * channels that have no stochastic realization (generalized amplitude
   * damping, correlated phase flips, custom Kraus maps).
   *
   * Off by default, which keeps the sampled behavior that pure-state and MPS
   * backends require. NoiseAdd cannot detect the backend on its own: the
   * network only reports which simulator it used after a circuit has run.
   * The Python bindings pick the path automatically from SimulatorConfig
   * (see uses_exact_quantum_channels in bindings.cpp).
   */
  // Bindings may supply a language-level warning (e.g. Python RuntimeWarning).
  void set_warning_handler(std::function<void(const std::string&)> handler) {
    warning_handler = std::move(handler);
  }

  void set_exact_channels(bool exact) { exact_channels = exact; }
  bool uses_exact_channels() const { return exact_channels; }

  std::shared_ptr<Circuits::Circuit<double>> inject(
      const std::shared_ptr<Circuits::Circuit<double>>& circ,
      const NoiseModel& nm) {
    WarnThermalApproximation(nm);
    return Inject(circ, nm, Layer::Pauli);
  }

  std::shared_ptr<Circuits::Circuit<double>> inject_coherent(
      const std::shared_ptr<Circuits::Circuit<double>>& circ,
      const NoiseModel& nm) {
    // Coherent noise is a sampled unitary trajectory on every backend: there
    // is no single channel that reproduces a systematic miscalibration.
    return inject_coherent_noise(circ, nm, rng);
  }

  std::shared_ptr<Circuits::Circuit<double>> inject_combined(
      const std::shared_ptr<Circuits::Circuit<double>>& circ,
      const NoiseModel& nm) {
    WarnThermalApproximation(nm);
    return Inject(circ, nm, Layer::Combined);
  }

  // Legacy utility for identity qubit-to-bit mappings only. Noisy execution
  // attaches rates to measurements instead; never apply this to those counts.
  void apply_readout_error_to_counts(
      Circuits::Circuit<>::ExecuteResults& counts,
      const noise::NoiseModel& nm) {
    if (!nm.has_readout_error()) return;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Circuits::Circuit<>::ExecuteResults new_counts;

    for (const auto& [bitstring, count] : counts) {
      for (size_t shot = 0; shot < count; ++shot) {
        auto noisy_bs = bitstring;
        for (size_t i = 0; i < noisy_bs.size(); ++i) {
          int qubit_idx = static_cast<int>(i);
          const auto* re = nm.get_readout_error(qubit_idx);
          if (!re) continue;
          double r = dist(rng);
          if (!noisy_bs[i] && r < re->p_meas1_prep0)
            noisy_bs[i] = true;
          else if (noisy_bs[i] && r < re->p_meas0_prep1)
            noisy_bs[i] = false;
        }
        new_counts[noisy_bs]++;
      }
    }
    counts = std::move(new_counts);
  }

  // Noise draws and simulator draws have independent streams. Re-seeding
  // restarts both; successive calls on this object advance the streams.
  void seed(unsigned int s) {
    rng.seed(s);
    public_seed = s;
    next_simulator_stream = 0;
  }

  Circuits::Circuit<>::ExecuteResults noisy_execute(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const NoiseModel& nm, int shots, int noise_realizations) {
    return Execute(circuit, network, hostId, nm, shots, noise_realizations,
                   Layer::Pauli);
  }

  Circuits::Circuit<>::ExecuteResults coherent_execute(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const NoiseModel& nm, int shots, int noise_realizations) {
    return Execute(circuit, network, hostId, nm, shots, noise_realizations,
                   Layer::Coherent);
  }

  Circuits::Circuit<>::ExecuteResults full_noise_execute(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const NoiseModel& nm, int shots, int noise_realizations) {
    return Execute(circuit, network, hostId, nm, shots, noise_realizations,
                   Layer::Combined);
  }

  Circuits::Circuit<>::ExecuteResults noisy_execute_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, const NoiseModel& nm,
      int shots, int noise_realizations) {
    return Execute(circuit, network, std::nullopt, nm, shots,
                   noise_realizations, Layer::Pauli);
  }

  Circuits::Circuit<>::ExecuteResults coherent_execute_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, const NoiseModel& nm,
      int shots, int noise_realizations) {
    return Execute(circuit, network, std::nullopt, nm, shots,
                   noise_realizations, Layer::Coherent);
  }

  Circuits::Circuit<>::ExecuteResults full_noise_execute_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, const NoiseModel& nm,
      int shots, int noise_realizations) {
    return Execute(circuit, network, std::nullopt, nm, shots,
                   noise_realizations, Layer::Combined);
  }

  std::vector<double> noisy_estimate(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const std::vector<std::string>& paulis, const NoiseModel& nm) {
    if (!network || !circuit) return {};

    auto ideal = network->ExecuteOnHostExpectations(circuit, hostId, paulis);
    std::vector<double> noisy;
    noisy.reserve(paulis.size());
    for (size_t i = 0; i < paulis.size(); ++i) {
      const double damping = nm.compute_damping(paulis[i]);
      noisy.push_back(damping * ideal[i]);
    }

    return noisy;
  }

  std::vector<double> noisy_estimate_montecarlo(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    return Estimate(circuit, network, hostId, paulis, nm, noise_realizations,
                    Layer::Pauli);
  }

  std::vector<double> coherent_estimate(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    return Estimate(circuit, network, hostId, paulis, nm, noise_realizations,
                    Layer::Coherent);
  }

  std::vector<double> full_noise_estimate(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    return Estimate(circuit, network, hostId, paulis, nm, noise_realizations,
                    Layer::Combined);
  }

  std::vector<double> noisy_estimate_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const std::vector<std::string>& paulis, const NoiseModel& nm) {
    if (!network || !circuit) return {};

    auto ideal = network->ExecuteExpectations(circuit, paulis);
    std::vector<double> noisy;
    noisy.reserve(paulis.size());
    for (size_t i = 0; i < paulis.size(); ++i) {
      const double damping = nm.compute_damping(paulis[i]);
      noisy.push_back(damping * ideal[i]);
    }

    return noisy;
  }

  std::vector<double> noisy_estimate_montecarlo_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    return Estimate(circuit, network, std::nullopt, paulis, nm,
                    noise_realizations, Layer::Pauli);
  }

  std::vector<double> coherent_estimate_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    return Estimate(circuit, network, std::nullopt, paulis, nm,
                    noise_realizations, Layer::Coherent);
  }

  std::vector<double> full_noise_estimate_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    return Estimate(circuit, network, std::nullopt, paulis, nm,
                    noise_realizations, Layer::Combined);
  }

 private:
  enum class Layer { Pauli, Coherent, Combined };
  using CircuitPtr = std::shared_ptr<Circuits::Circuit<double>>;
  using NetworkPtr = std::shared_ptr<Network::INetwork<>>;

  struct ExecutionContext {
    NetworkPtr network;
    std::optional<uint64_t> seed;
  };

  ExecutionContext PrepareExecution(const NetworkPtr& network) {
    auto base_seed = public_seed;
    const auto simulator = network->GetSimulator();
    if (simulator) {
      const auto configured = simulator->GetConfiguration("seed");
      if (!configured.empty()) base_seed = std::stoull(configured);
      if (simulator->GetType() == Simulators::SimulatorType::kDistMpiGpuSim) {
        if (!base_seed)
          base_seed = Simulators::GenerateRandomSeed(simulator->GetType(),
                                                     simulator->GetConfigMap());
        if (!public_seed && !mpi_noise_seeded) {
          rng.seed(static_cast<uint32_t>(*base_seed));
          mpi_noise_seeded = true;
        }
      }
    }
    // Work on a pristine network so derived batch seeds never replace the
    // caller's configuration, including on exception paths.
    auto execution_network = network->Clone();
    if (!execution_network)
      throw std::runtime_error("Cannot clone network for noisy execution");
    execution_network->SetMaxSimulators(network->GetMaxSimulators());
    // Timing-based backend selection changes the random sequence between
    // otherwise identical calls. Exact channels also require the chosen
    // mixed-state backend to survive optimization.
    if (simulator && (base_seed || exact_channels))
      execution_network->RemoveAllOptimizationSimulatorsAndAdd(
          simulator->GetType(), simulator->GetSimulationType());
    return {std::move(execution_network), base_seed};
  }

  void SeedExecution(const ExecutionContext& execution) {
    if (!execution.seed) return;
    const auto seed = Simulators::IState::DeriveSeed(*execution.seed,
                                                     next_simulator_stream++);
    execution.network->Configure("seed", std::to_string(seed).c_str());
    // Also cover networks that execute their existing simulator directly.
    if (const auto simulator = execution.network->GetSimulator())
      simulator->SetSeed(seed);
  }

  CircuitPtr Inject(const CircuitPtr& circuit, const NoiseModel& nm,
                    Layer layer) {
    if (!circuit) return nullptr;
    if (layer == Layer::Coherent)
      return inject_coherent_noise(circuit, nm, rng);
    auto noisy =
        layer == Layer::Pauli
            ? (exact_channels ? inject_exact_noise(circuit, nm)
                              : inject_noise(circuit, nm, rng))
            : (exact_channels ? inject_combined_noise_exact(circuit, nm, rng)
                              : inject_combined_noise(circuit, nm, rng));
    attach_readout_error(noisy, nm);
    return noisy;
  }

  void WarnThermalApproximation(const NoiseModel& nm) const {
    if (exact_channels || nm.has_additional_quantum_channels()) return;
    const auto qubits = nm.thermal_approximation_qubits();
    if (qubits.empty()) return;
    std::ostringstream message;
    message << "Sampled thermal approximation for circuit qubits [";
    for (size_t i = 0; i < qubits.size(); ++i) {
      if (i) message << ", ";
      message << qubits[i];
    }
    message << "]: effective T2 clamped to T1. Use density matrix or a "
               "supported MPO backend to preserve calibrated T2.";
    if (warning_handler)
      warning_handler(message.str());
    else
      std::cerr << "Warning: " << message.str() << '\n';
  }

  Circuits::Circuit<>::ExecuteResults Execute(const CircuitPtr& circuit,
                                              const NetworkPtr& network,
                                              std::optional<size_t> host,
                                              const NoiseModel& nm, int shots,
                                              int realizations, Layer layer) {
    if (shots < 0) throw std::invalid_argument("shots must be nonnegative");
    if (!network || !circuit || shots == 0) return {};
    if (layer != Layer::Coherent) WarnThermalApproximation(nm);
    auto execution = PrepareExecution(network);
    const int batches = std::min(shots, std::max(1, realizations));
    Circuits::Circuit<>::ExecuteResults combined;
    for (int b = 0; b < batches; ++b) {
      const int batch_shots = shots / batches + (b < shots % batches ? 1 : 0);
      auto noisy = Inject(circuit, nm, layer);
      SeedExecution(execution);
      const auto counts =
          host ? execution.network->RepeatedExecuteOnHost(noisy, *host,
                                                          batch_shots)
               : execution.network->RepeatedExecute(noisy, batch_shots);
      for (const auto& entry : counts) combined[entry.first] += entry.second;
    }
    return combined;
  }

  std::vector<double> Estimate(const CircuitPtr& circuit,
                               const NetworkPtr& network,
                               std::optional<size_t> host,
                               const std::vector<std::string>& paulis,
                               const NoiseModel& nm, int realizations,
                               Layer layer) {
    if (realizations <= 0)
      throw std::invalid_argument("noise_realizations must be positive");
    if (!network || !circuit) return {};
    if (layer != Layer::Coherent) WarnThermalApproximation(nm);
    auto execution = PrepareExecution(network);
    std::vector<double> values(paulis.size(), 0.0);
    for (int r = 0; r < realizations; ++r) {
      auto noisy = Inject(circuit, nm, layer);
      SeedExecution(execution);
      const auto sample =
          host ? execution.network->ExecuteOnHostExpectations(noisy, *host,
                                                              paulis)
               : execution.network->ExecuteExpectations(noisy, paulis);
      for (size_t i = 0; i < values.size(); ++i) values[i] += sample[i];
    }
    for (auto& value : values) value /= realizations;
    return values;
  }

  bool exact_channels = false;
  std::mt19937 rng;
  std::optional<uint64_t> public_seed;
  uint64_t next_simulator_stream = 0;
  bool mpi_noise_seeded = false;
  std::function<void(const std::string&)> warning_handler;
};

}  // namespace noise

#endif
