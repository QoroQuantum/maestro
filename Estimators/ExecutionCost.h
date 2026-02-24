/**
 * @file ExecutionCost.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Estimate the execution cost for a simulator, circuit and number of
 * shots.
 *
 * Rough estimtion of the execution cost (not time, we have estimators for that), based on O() complexity and some
 * guesses. Unlike the estimators, this does not take into account multithreading and implementation details of the simulators.
 */

#pragma once

#ifndef __EXECUTION_COST_H_
#define __EXECUTION_COST_H_

#include "../Simulators/QcsimPauliPropagator.h"

namespace Estimators {

class ExecutionCost {
 public:
  static double EstimateExecutionCost(
      Simulators::SimulationType method, size_t nrQubits,
      const std::shared_ptr<Circuits::Circuit<>>& circuit, size_t maxBondDim) {
    if (method == Simulators::SimulationType::kPauliPropagator)
      return Simulators::QcsimPauliPropagator::GetCost(circuit);
    else if (method == Simulators::SimulationType::kStatevector)
    {
      const double opOrder = exp2(nrQubits);
      return opOrder * circuit->size(); // this is very rough, it can be improved
    } else if (method == Simulators::SimulationType::kMatrixProductState)
    {
      const double twoQubitOpOrder = pow(maxBondDim, 3);
      const double oneQubitOpOrder = pow(maxBondDim, 2);


    } else if (method == Simulators::SimulationType::kStabilizer)
    {
      const double measOrder = pow(nrQubits, 2);
      const double opOrder = nrQubits;

    } 

    // for tensor network is hard to guess, it depends on contraction path

    return std::numeric_limits<double>::infinity();
  }

  static double EstimateSamplingCost(
      Simulators::SimulationType method, size_t nrQubits,
      size_t nrQubitsSampled, size_t samples,
      const std::shared_ptr<Circuits::Circuit<>>& circuit, size_t maxBondDim) {
    if (method == Simulators::SimulationType::kPauliPropagator)
      return Simulators::QcsimPauliPropagator::GetSamplingCost(
          circuit, nrQubitsSampled, samples);

    // sampling works very differenty for such circuits
    const bool hasMeasurementsInTheMiddle = circuit->HasOpsAfterMeasurements();
    
if (method == Simulators::SimulationType::kStatevector) {
      const double opOrder = exp2(nrQubits);

    } else if (method == Simulators::SimulationType::kMatrixProductState) {
      const double twoQubitOpOrder = pow(maxBondDim, 3);
      const double oneQubitOpOrder = pow(maxBondDim, 2);

    } else if (method == Simulators::SimulationType::kStabilizer) {
      const double measOrder = pow(nrQubits, 2);
      const double opOrder = nrQubits;
    } 

    // for tensor network is hard to guess, it depends on contraction path

    return std::numeric_limits<double>::infinity();
  }
};

}  // namespace Estimators

#endif
