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

#include "../python/noise.h"

#include "Circuit/Circuit.h"

namespace noise {

class NoiseAdd {
 public:
  NoiseAdd() : rng(std::random_device{}()) {}

  std::shared_ptr<Circuits::Circuit<double>> inject(
      const std::shared_ptr<Circuits::Circuit<double>>& circ,
      const NoiseModel& nm) {
    return inject_noise(circ, nm, rng);
  }

  std::shared_ptr<Circuits::Circuit<double>> inject_coherent(
      const std::shared_ptr<Circuits::Circuit<double>>& circ,
      const NoiseModel& nm) {
    return inject_coherent_noise(circ, nm, rng);
  }

  std::shared_ptr<Circuits::Circuit<double>> inject_combined(
      const std::shared_ptr<Circuits::Circuit<double>>& circ,
      const NoiseModel& nm) {
    return inject_combined_noise(circ, nm, rng);
  }

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

  void seed(unsigned int s) { rng.seed(s); }

  Circuits::Circuit<>::ExecuteResults noisy_execute(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const NoiseModel& nm, int shots, int noise_realizations) {
    if (!network || !circuit) return {};

    const int batches = std::min(shots, std::max(1, noise_realizations));
    const int base_batch = shots / batches;
    int leftover = shots % batches;

    Circuits::Circuit<>::ExecuteResults combined;

    for (int b = 0; b < batches; ++b) {
      int batch_shots = base_batch + (b < leftover ? 1 : 0);
      if (batch_shots <= 0) continue;

      auto noisy = inject(circuit, nm);
      auto r = network->RepeatedExecuteOnHost(noisy, hostId, batch_shots);

      for (auto item : r) combined[item.first] += item.second;
    }

    // Apply readout error (classical post-measurement channel)
    apply_readout_error_to_counts(combined, nm);

    return combined;
  }

  Circuits::Circuit<>::ExecuteResults coherent_execute(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const NoiseModel& nm, int shots, int noise_realizations) {
    if (!network || !circuit) return {};

    const int batches = std::min(shots, std::max(1, noise_realizations));
    const int base_batch = shots / batches;
    int leftover = shots % batches;

    Circuits::Circuit<>::ExecuteResults combined;

    for (int b = 0; b < batches; ++b) {
      int batch_shots = base_batch + (b < leftover ? 1 : 0);
      if (batch_shots <= 0) continue;

      auto noisy = inject_coherent(circuit, nm);
      auto r = network->RepeatedExecuteOnHost(noisy, hostId, batch_shots);

      for (auto item : r) combined[item.first] += item.second;
    }

    return combined;
  }

  Circuits::Circuit<>::ExecuteResults full_noise_execute(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const NoiseModel& nm, int shots, int noise_realizations) {
    if (!network || !circuit) return {};

    const int batches = std::min(shots, std::max(1, noise_realizations));
    const int base_batch = shots / batches;
    int leftover = shots % batches;

    Circuits::Circuit<>::ExecuteResults combined;

    for (int b = 0; b < batches; ++b) {
      int batch_shots = base_batch + (b < leftover ? 1 : 0);
      if (batch_shots <= 0) continue;

      auto noisy = inject_combined(circuit, nm);
      auto r = network->RepeatedExecuteOnHost(noisy, hostId, batch_shots);

      for (auto item : r) combined[item.first] += item.second;
    }

    // Apply readout error (classical post-measurement channel)
    apply_readout_error_to_counts(combined, nm);

    return combined;
  }



  Circuits::Circuit<>::ExecuteResults noisy_execute_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const NoiseModel& nm, int shots, int noise_realizations) {
    if (!network || !circuit) return {};

    const int batches = std::min(shots, std::max(1, noise_realizations));
    const int base_batch = shots / batches;
    int leftover = shots % batches;

    Circuits::Circuit<>::ExecuteResults combined;

    for (int b = 0; b < batches; ++b) {
      int batch_shots = base_batch + (b < leftover ? 1 : 0);
      if (batch_shots <= 0) continue;

      auto noisy = inject(circuit, nm);
      auto r = network->RepeatedExecute(noisy, batch_shots);

      for (auto item : r) combined[item.first] += item.second;
    }

    // Apply readout error (classical post-measurement channel)
    apply_readout_error_to_counts(combined, nm);

    return combined;
  }

  Circuits::Circuit<>::ExecuteResults coherent_execute_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const NoiseModel& nm, int shots, int noise_realizations) {
    if (!network || !circuit) return {};

    const int batches = std::min(shots, std::max(1, noise_realizations));
    const int base_batch = shots / batches;
    int leftover = shots % batches;

    Circuits::Circuit<>::ExecuteResults combined;

    for (int b = 0; b < batches; ++b) {
      int batch_shots = base_batch + (b < leftover ? 1 : 0);
      if (batch_shots <= 0) continue;

      auto noisy = inject_coherent(circuit, nm);
      auto r = network->RepeatedExecute(noisy, batch_shots);

      for (auto item : r) combined[item.first] += item.second;
    }

    return combined;
  }

  Circuits::Circuit<>::ExecuteResults full_noise_execute_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const NoiseModel& nm, int shots, int noise_realizations) {
    if (!network || !circuit) return {};

    const int batches = std::min(shots, std::max(1, noise_realizations));
    const int base_batch = shots / batches;
    int leftover = shots % batches;

    Circuits::Circuit<>::ExecuteResults combined;

    for (int b = 0; b < batches; ++b) {
      int batch_shots = base_batch + (b < leftover ? 1 : 0);
      if (batch_shots <= 0) continue;

      auto noisy = inject_combined(circuit, nm);
      auto r = network->RepeatedExecuteOnHost(noisy, batch_shots);

      for (auto item : r) combined[item.first] += item.second;
    }

    // Apply readout error (classical post-measurement channel)
    apply_readout_error_to_counts(combined, nm);

    return combined;
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
    if (!network || !circuit) return {};

    const int n_obs = paulis.size();
    std::vector<double> vals(n_obs, 0.0);

    for (int r = 0; r < noise_realizations; ++r) {
      auto noisy = inject(circuit, nm);

      auto ideal = network->ExecuteOnHostExpectations(noisy, hostId, paulis);
      for (size_t i = 0; i < n_obs; ++i) vals[i] += ideal[i];
    }

    for (size_t i = 0; i < n_obs; ++i) vals[i] /= noise_realizations;

    return vals;
  }

  std::vector<double> coherent_estimate(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    if (!network || !circuit) return {};

    const int n_obs = paulis.size();
    std::vector<double> vals(n_obs, 0.0);

    for (int r = 0; r < noise_realizations; ++r) {
      auto noisy = inject_coherent(circuit, nm);

      auto ideal = network->ExecuteOnHostExpectations(noisy, hostId, paulis);
      for (size_t i = 0; i < n_obs; ++i) vals[i] += ideal[i];
    }

    for (size_t i = 0; i < n_obs; ++i) vals[i] /= noise_realizations;

    return vals;
  }

  std::vector<double> full_noise_estimate(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network, size_t hostId,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    if (!network || !circuit) return {};

    const int n_obs = paulis.size();
    std::vector<double> vals(n_obs, 0.0);

    for (int r = 0; r < noise_realizations; ++r) {
      auto noisy = inject_combined(circuit, nm);

      auto ideal = network->ExecuteOnHostExpectations(noisy, hostId, paulis);
      for (size_t i = 0; i < n_obs; ++i) vals[i] += ideal[i];
    }

    for (size_t i = 0; i < n_obs; ++i) vals[i] /= noise_realizations;

    return vals;
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
    if (!network || !circuit) return {};

    const int n_obs = paulis.size();
    std::vector<double> vals(n_obs, 0.0);

    for (int r = 0; r < noise_realizations; ++r) {
      auto noisy = inject(circuit, nm);

      auto ideal = network->ExecuteExpectations(noisy, paulis);
      for (size_t i = 0; i < n_obs; ++i) vals[i] += ideal[i];
    }

    for (size_t i = 0; i < n_obs; ++i) vals[i] /= noise_realizations;

    return vals;
  }

  std::vector<double> coherent_estimate_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    if (!network || !circuit) return {};

    const int n_obs = paulis.size();
    std::vector<double> vals(n_obs, 0.0);

    for (int r = 0; r < noise_realizations; ++r) {
      auto noisy = inject_coherent(circuit, nm);

      auto ideal = network->ExecuteExpectations(noisy, paulis);
      for (size_t i = 0; i < n_obs; ++i) vals[i] += ideal[i];
    }

    for (size_t i = 0; i < n_obs; ++i) vals[i] /= noise_realizations;

    return vals;
  }

  std::vector<double> full_noise_estimate_distributed(
      const std::shared_ptr<Circuits::Circuit<double>>& circuit,
      const std::shared_ptr<Network::INetwork<>>& network,
      const std::vector<std::string>& paulis, const NoiseModel& nm,
      int noise_realizations) {
    if (!network || !circuit) return {};

    const int n_obs = paulis.size();
    std::vector<double> vals(n_obs, 0.0);

    for (int r = 0; r < noise_realizations; ++r) {
      auto noisy = inject_combined(circuit, nm);

      auto ideal = network->ExecuteExpectations(noisy, paulis);
      for (size_t i = 0; i < n_obs; ++i) vals[i] += ideal[i];
    }

    for (size_t i = 0; i < n_obs; ++i) vals[i] /= noise_realizations;

    return vals;
  }

 private:
  std::mt19937 rng;
};

}  // namespace noise

#endif
