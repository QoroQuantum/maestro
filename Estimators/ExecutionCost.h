/**
 * @file ExecutionCost.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Estimate the execution cost for a simulator, circuit and number of
 * shots.
 *
 * Rough estimtion of the execution cost (not time, we have estimators for
 * that), based on O() complexity and some guesses. Unlike the estimators, this
 * does not take into account multithreading and implementation details of the
 * simulators.
 */

#pragma once

#ifndef __EXECUTION_COST_H_
#define __EXECUTION_COST_H_

#include "../Simulators/QcsimPauliPropagator.h"

namespace Estimators {

// this is not for time estimation, it can be used for something very, very, VERY rough, to decide just if it would execute in a reasonable time or not
// it cannot be used to compare runtime for two simulators of the same circuit, but it sort of can be used to compare two circuits in the same simulator (especially if the same number of samples is used for both circuits)
// for something better we have the estimators which got the constants from benchmarking and
// rely on implementation details of the simulators, so they are more accurate,
// but also less general
class ExecutionCost {
 public:
  static double EstimateExecutionCost(
      Simulators::SimulationType method, size_t nrQubits,
      const std::shared_ptr<Circuits::Circuit<>>& circuit, size_t maxBondDim) {
    if (method == Simulators::SimulationType::kPauliPropagator)
      return Simulators::QcsimPauliPropagator::GetCost(circuit);
    else if (method == Simulators::SimulationType::kStatevector) {
      const double opOrder = exp2(nrQubits);

      double cost = 0;
      for (const auto& op : *circuit) {
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement || op->GetType() == Circuits::OperationType::kConditionalMeasurement)
          cost += opOrder * affectedQubits.size();
        if (op->GetType() == Circuits::OperationType::kReset)
          cost += 2. * opOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
          op->GetType() == Circuits::OperationType::kConditionalGate) {
            if (affectedQubits.size() == 1)
                cost += opOrder;
            else if (affectedQubits.size() == 2)
                cost += 4 * opOrder;
            else if (affectedQubits.size() == 3)
                cost += 16 * opOrder;
        }
      }
      return cost;
    } else if (method == Simulators::SimulationType::kMatrixProductState) {
      const double twoQubitOpOrder = pow(maxBondDim, 3);
      const double oneQubitOpOrder = pow(maxBondDim, 2);

      double cost = 0;
      for (const auto& op : *circuit) {
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          // it's not that simple at all
          // a qubit measurement works by applying a projector on the qubit tensor, which would cost as a one quibit gate but then we need to
          // propagate the effect of the measurement along the chain, left and right,
          // which is like applying a two qubit gate (SVD is the costlier operation there) on all the qubits that are
          // entangled with the measured one, and in the worst case this can be
          // all the other qubits if more than one qubits are measured at the same time, then
          // we can have some optimizations, but let's just say that it's like
          // measuring them one by one, so the cost is multiplied by the number
          // of measured qubits
          cost += twoQubitOpOrder * affectedQubits.size() * nrQubits / 2.;
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
          if (affectedQubits.size() == 1)
            cost += oneQubitOpOrder;
          else if (affectedQubits.size() == 2)
            // I wish it were simple, but applying a gate involves swapping the
            // qubits next to each other, then applying the gate
            cost += twoQubitOpOrder * nrQubits / 3.;
          else if (affectedQubits.size() == 3)
            // qiskit aer has three qubit ops, qcsim has not, they are
            // decomposed into one and two qubit gates
            cost += twoQubitOpOrder * nrQubits * 2; // very rough estimation
        }
      }
      return cost;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      const double measOrder = pow(nrQubits, 2);
      const double opOrder = nrQubits;

      double cost = 0;
      for (const auto& op : *circuit) {
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          cost += measOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate)
          cost += opOrder * affectedQubits.size();
      }
      return cost;
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
    Circuits::OperationState dummyState;
    const bool hasMeasurementsInTheMiddle = circuit->HasOpsAfterMeasurements();
    const std::vector<bool> executedOps =
        circuit->ExecuteNonMeasurements(nullptr, dummyState);

    if (method == Simulators::SimulationType::kStatevector) {
      const double opOrder = exp2(nrQubits);

      double cost = 0;
      for (size_t i = 0; i < executedOps.size(); ++i) {
        if (!executedOps[i])
          continue;  
        const auto& op = (*circuit)[i];
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement)
          cost += opOrder * affectedQubits.size();
        if (op->GetType() == Circuits::OperationType::kReset)
          cost += 2. * opOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
          if (affectedQubits.size() == 1)
            cost += opOrder;
          else if (affectedQubits.size() == 2)
            cost += 4 * opOrder;
          else if (affectedQubits.size() == 3)
            cost += 16 * opOrder;
        }
      }

      // the sampling cost depends on the simulator
      // qiskit aer constucts an index gowing over the statevector, qcsim does something similar to have the samples then O(nrQubits), but in maestro for qcsim (and quest) this is replaced by alias sampling with O(1) for each sample

      // some dummy cost, it's not going to fit all anyways
      return cost + 30 * opOrder + samples * nrQubits;
    } else if (method == Simulators::SimulationType::kMatrixProductState) {
      const double twoQubitOpOrder = pow(maxBondDim, 3);
      const double oneQubitOpOrder = pow(maxBondDim, 2);

      double cost = 0;
      for (size_t i = 0; i < executedOps.size(); ++i) {
        if (!executedOps[i]) continue;
        const auto& op = (*circuit)[i];
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          // it's not that simple at all
          // a qubit measurement works by applying a projector on the qubit
          // tensor, which would cost as a one quibit gate but then we need to
          // propagate the effect of the measurement along the chain, left and
          // right, which is like applying a two qubit gate (SVD is the costlier
          // operation there) on all the qubits that are entangled with the
          // measured one, and in the worst case this can be all the other
          // qubits if more than one qubits are measured at the same time, then
          // we can have some optimizations, but let's just say that it's like
          // measuring them one by one, so the cost is multiplied by the number
          // of measured qubits
          cost += twoQubitOpOrder * affectedQubits.size() * nrQubits / 2.;
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
          if (affectedQubits.size() == 1)
            cost += oneQubitOpOrder;
          else if (affectedQubits.size() == 2)
            // I wish it were simple, but applying a gate involves swapping the
            // qubits next to each other, then applying the gate
            cost += twoQubitOpOrder * nrQubits / 3.;
          else if (affectedQubits.size() == 3)
            // qiskit aer has three qubit ops, qcsim has not, they are
            // decomposed into one and two qubit gates
            cost += twoQubitOpOrder * nrQubits * 2;  // very rough estimation
        }
      }

      // sampling can be done here (and for other simulator types as well) either by saving the state, measuring then restoring the state
      // or simply going along the chain, computing probabilities, throwing biased coins and doing matrix multiplications (qcsim does this if all qubits are sampled, otherwise the measurement method is used)

      // some dummy cost, it's not going to fit all anyways
      return cost + samples * twoQubitOpOrder * nrQubits * nrQubits;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      const double measOrder = pow(nrQubits, 2);
      const double opOrder = nrQubits;

      double cost = 0;
      for (size_t i = 0; i < executedOps.size(); ++i) {
        if (!executedOps[i]) continue;
        const auto& op = (*circuit)[i];
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          cost += measOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate)
          cost += opOrder * affectedQubits.size();
      }

      // sampling is done with saving state, measuring and restoring state
      // the overhead for saving / restoring is not here (it's of the same order as measuring), but measurements are not all O(n^2) anyways
      return cost + samples * measOrder * nrQubitsSampled;
    }

    // for tensor network is hard to guess, it depends on contraction path

    return std::numeric_limits<double>::infinity();
  }
};

}  // namespace Estimators

#endif
