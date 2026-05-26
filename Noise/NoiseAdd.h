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
      const NoiseModel& nm)
  {
    return inject_combined_noise(circ, nm, rng);
  }

  std::shared_ptr<Circuits::Circuit<double>> inject_coherent(
    const std::shared_ptr<Circuits::Circuit<double>> &circ,
    const NoiseModel &nm)
  {
    return inject_coherent_noise(circ, nm, rng);
  }

  std::shared_ptr<Circuits::Circuit<double>> inject_combined(
      const std::shared_ptr<Circuits::Circuit<double>>& circ,
      const NoiseModel& nm)
  {
    return inject_combined_noise(circ, nm, rng);
  }

  void apply_readout_error_to_counts(
      Circuits::Circuit<>::ExecuteResults &counts,
      const noise::NoiseModel &nm) {
    if (!nm.has_readout_error()) return;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Circuits::Circuit<>::ExecuteResults new_counts;

    for (const auto &[bitstring, count] : counts) {
      for (size_t shot = 0; shot < count; ++shot) {
        auto noisy_bs = bitstring;
        for (size_t i = 0; i < noisy_bs.size(); ++i) {
          int qubit_idx = static_cast<int>(i);
          const auto *re = nm.get_readout_error(qubit_idx);
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

 private:
  std::mt19937 rng;
};

}

#endif

