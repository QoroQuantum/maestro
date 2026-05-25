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

 private:
  std::mt19937 rng;
};

}

#endif

