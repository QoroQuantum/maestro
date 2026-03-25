/**
 * @file QcsimPauliPropagator.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The qcsim pauli propagator class.
 *
 * The main role is to extend the qcsim pauli propagator with more gates.
 */

#pragma once

#ifndef _QCSIM_PAULI_PROPAGATOR_H
#define _QCSIM_PAULI_PROPAGATOR_H 1

#include "PauliPropagator.h"
#include "../Circuit/Circuit.h"

namespace Simulators {

class QcsimPauliPropagator : public QC::PauliPropagator {
 public:
  void ApplyP(int qubit, double lambda) { ApplyRZ(qubit, lambda); }

  void ApplyT(int qubit) { ApplyRZ(qubit, M_PI_4); }

  void ApplyTDG(int qubit) { ApplyRZ(qubit, -M_PI_4); }

  void ApplyU(int qubit, double theta, double phi, double lambda,
              double gamma = 0.0) {
    ApplyRZ(qubit, lambda);
    ApplyRY(qubit, theta);
    ApplyRZ(qubit, phi);
  }

  void ApplyCH(int controlQubit, int targetQubit) {
    ApplyH(targetQubit);
    ApplySDG(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyH(targetQubit);
    ApplyT(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyT(targetQubit);
    ApplyH(targetQubit);
    ApplyS(targetQubit);
    ApplyX(targetQubit);
    ApplyS(controlQubit);
  }

  void ApplyCU(int controlQubit, int targetQubit, double theta, double phi,
               double lambda, double gamma = 0.0) {
    if (gamma != 0.0) ApplyP(controlQubit, gamma);

    const double lambdaPlusPhiHalf = 0.5 * (lambda + phi);
    const double halfTheta = 0.5 * theta;
    ApplyP(targetQubit, 0.5 * (lambda - phi));
    ApplyP(controlQubit, lambdaPlusPhiHalf);
    ApplyCX(controlQubit, targetQubit);
    ApplyU(targetQubit, -halfTheta, 0, -lambdaPlusPhiHalf);
    ApplyCX(controlQubit, targetQubit);
    ApplyU(targetQubit, halfTheta, phi, 0);
  }

  void ApplyCRX(int controlQubit, int targetQubit, double angle) {
    const double halfAngle = angle * 0.5;

    ApplyH(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyRZ(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRZ(targetQubit, halfAngle);
    ApplyH(targetQubit);
  }

  void ApplyCRY(int controlQubit, int targetQubit, double angle) {
    const double halfAngle = angle * 0.5;
    ApplyRY(targetQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRY(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
  }

  void ApplyCRZ(int controlQubit, int targetQubit, double angle) {
    const double halfAngle = angle * 0.5;

    ApplyRZ(targetQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRZ(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
  }

  void ApplyCP(int controlQubit, int targetQubit, double lambda) {
    const double halfAngle = lambda * 0.5;
    ApplyP(controlQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyP(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyP(targetQubit, halfAngle);
  }

  void ApplyCS(int controlQubit, int targetQubit) {
    ApplyT(controlQubit);
    ApplyT(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyTDG(targetQubit);
    ApplyCX(controlQubit, targetQubit);
  }

  void ApplyCSDAG(int controlQubit, int targetQubit) {
    ApplyCX(controlQubit, targetQubit);
    ApplyT(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyTDG(controlQubit);
    ApplyTDG(targetQubit);
  }

  void ApplyCSX(int controlQubit, int targetQubit) {
    ApplyH(targetQubit);
    ApplyCS(controlQubit, targetQubit);
    ApplyH(targetQubit);
  }

  void ApplyCSXDAG(int controlQubit, int targetQubit) {
    ApplyH(targetQubit);
    ApplyCSDAG(controlQubit, targetQubit);
    ApplyH(targetQubit);
  }

  void ApplyCSwap(int controlQubit, int targetQubit1, int targetQubit2) {
    const int q1 = controlQubit;  // control
    const int q2 = targetQubit1;
    const int q3 = targetQubit2;

    ApplyCX(q3, q2);
    ApplyCSX(q2, q3);  // 3 rotaations
    ApplyCX(q1, q2);

    ApplyP(q3, M_PI);     // 1 rotation
    ApplyP(q2, -M_PI_2);  // 1 rotation

    ApplyCSX(q2, q3);  // 3 rotations
    ApplyCX(q1, q2);

    ApplyP(q3, M_PI);  // 1 rotation
    ApplyCSX(q1, q3);  // 3 rotations
    ApplyCX(q3, q2);
  }

  void ApplyCCX(int controlQubit1, int controlQubit2, int targetQubit) {
    const int q1 = controlQubit1;  // control 1
    const int q2 = controlQubit2;  // control 2
    const int q3 = targetQubit;    // target

    ApplyCSX(q2, q3);  // 3 rotations
    ApplyCX(q1, q2);
    ApplyCSXDAG(q2, q3);  // 3 rotations
    ApplyCX(q1, q2);
    ApplyCSX(q1, q3);  // 3 rotations
  }

  std::unique_ptr<QcsimPauliPropagator> Clone() const {
    auto clone = std::make_unique<QcsimPauliPropagator>();
    clone->SetNrQubits(GetNrQubits());
    clone->SetPauliWeightThreshold(GetPauliWeightThreshold());
    clone->SetBatchSize(GetBatchSize());
    clone->SetBatchSizeForSum(GetBatchSizeForSum());
    clone->SetCoefficientThreshold(GetCoefficientThreshold());
    clone->SetParallelThreshold(GetParallelThreshold());
    clone->SetParallelThresholdForSum(GetParallelThresholdForSum());
    clone->SetStepsBetweenDeduplication(StepsBetweenDeduplication());
    clone->SetStepsBetweenTrims(StepsBetweenTrims());
    clone->SetOperations(GetOperations());
    clone->SetSavePosition(GetSavePosition());
    if (IsParallelEnabled()) clone->EnableParallel();

    return clone;
  }

  static double GetSamplingCost(
      const std::shared_ptr<Circuits::Circuit<>>& circuit,
      size_t nrQubitsSampled, size_t samples) {
    return samples * GetCost(circuit) * exp2(nrQubitsSampled - 1);
  }

  static double GetCost(const std::shared_ptr<Circuits::Circuit<>>& circuit) {
    if (!circuit) return 0.0;

    double cost = 0.0;
    double doublingCost = 1;

    // go backwards, the same way as the propagator does, to get the cost of the
    // operations in the circuit

    for (int pos = static_cast<int>(circuit->size()) - 1; pos >= 0; --pos) {
      const std::shared_ptr<Circuits::IOperation<>>& op = (*circuit)[pos];
      cost += GetOpCost(circuit, op, pos) * doublingCost;
      doublingCost *= GetOpMultiplication(op);
    }

    // summing up all pauli strings expectations
    cost += doublingCost;

    return cost;
  }

 private:
  static double GetOpCost(const std::shared_ptr<Circuits::Circuit<>>& circuit,
                          const std::shared_ptr<Circuits::IOperation<>>& op,
                          int pos) {
    if (!circuit || !op || pos < 0 || pos >= static_cast<int>(circuit->size()))
      return 0.;

    if (op->GetType() != Circuits::OperationType::kGate) {
      if (op->GetType() == Circuits::OperationType::kMeasurement) {
        // measurement costs a pauli string propagated down the circuit from the
        // point of measurement... but then it adds a projector in the circuit
        // in its place, which means a doubling at that point for subsequent
        // propagations
        if (pos == 0) return 1.;

        const auto& prevOp = (*circuit)[pos - 1];
        return 1. + GetOpCost(circuit, prevOp, pos - 1);
      } else if (op->GetType() == Circuits::OperationType::kReset) {
        // a measurement followed by a conditional X gate, assume X half of the
        // times
        if (pos == 0) return 1.5;
        const auto& prevOp = (*circuit)[pos - 1];
        return 1.5 + GetOpCost(circuit, prevOp, pos - 1);
      } else if (op->GetType() == Circuits::OperationType::kConditionalGate) {
        const auto conditionalGate =
            std::static_pointer_cast<Circuits::ConditionalGate<>>(op);
        return GetOpCost(circuit, conditionalGate->GetOperation(), pos);
      } else if (op->GetType() ==
                 Circuits::OperationType::kConditionalMeasurement) {
        const auto conditionalMeasurement =
            std::static_pointer_cast<Circuits::ConditionalMeasurement<>>(op);
        return GetOpCost(circuit, conditionalMeasurement->GetOperation(), pos);
      }
      return 0.;
    }

    const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<>>(op);

    switch (gate->GetGateType()) {
      // cliffords
      case Circuits::QuantumGateType::kXGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kYGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kZGateType:
        return 0.5;  // just a sign flip
      case Circuits::QuantumGateType::kHadamardGateType:
        return 1.;
      case Circuits::QuantumGateType::kKGateType:
        return 3.5;
      case Circuits::QuantumGateType::kSGateType:
        return 1.;
      case Circuits::QuantumGateType::kSdgGateType:
        return 1.5;
      case Circuits::QuantumGateType::kSxGateType:
        return 4.;
      case Circuits::QuantumGateType::kSxDagGateType:
        return 3.;
      case Circuits::QuantumGateType::kCXGateType:
        return 1.;
      case Circuits::QuantumGateType::kCYGateType:
        return 3.5;
      case Circuits::QuantumGateType::kCZGateType:
        return 3.;
      case Circuits::QuantumGateType::kSwapGateType:
        return 3.;

      // non-cliffords
      // the ones below just rotations, they split in two operators
      case Circuits::QuantumGateType::kPhaseGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kTGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kTdgGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kRxGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kRyGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kRzGateType:
        return 2.;

      case Circuits::QuantumGateType::kUGateType:
        return 8.;  // implemented with three rotations

      case Circuits::QuantumGateType::kCPGateType:
        // p, cx, p, cx, p
        return 20;
      case Circuits::QuantumGateType::kCRxGateType:
        // h, cx, rz, cx, rz, h
        return 14;
      case Circuits::QuantumGateType::kCRyGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRzGateType:
        return 12;
      case Circuits::QuantumGateType::kCHGateType:
        return 26.5;
      case Circuits::QuantumGateType::kCSxGateType:
        return 35;
      case Circuits::QuantumGateType::kCSxDagGateType:
        return 26;

      case Circuits::QuantumGateType::kCUGateType:
        return 546;

      case Circuits::QuantumGateType::kCCXGateType:
        return 2555;
      case Circuits::QuantumGateType::kCSwapGateType:
        return 23996;

      default:
        return 1.;
    }

    return 1.;
  }

  static double GetOpMultiplication(
      const std::shared_ptr<Circuits::IOperation<>>& op) {
    if (!op) return 1.;

    if (op->GetType() != Circuits::OperationType::kGate) {
      if (op->GetType() == Circuits::OperationType::kMeasurement) {
        return 2.;
      } else if (op->GetType() == Circuits::OperationType::kReset) {
        return 2.;
      } else if (op->GetType() == Circuits::OperationType::kConditionalGate) {
        const auto conditionalGate =
            std::static_pointer_cast<Circuits::ConditionalGate<>>(op);
        return GetOpMultiplication(conditionalGate->GetOperation());
      } else if (op->GetType() ==
                 Circuits::OperationType::kConditionalMeasurement) {
        return 2.;  // actually it depends on the measurement result
      }
      return 1.;
    }

    const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<>>(op);
    switch (gate->GetGateType()) {
      // all cliffords retun 1, as they do not increase the number of pauli
      // strings to propagate
      case Circuits::QuantumGateType::kXGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kYGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kZGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kHadamardGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kKGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kSGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kSdgGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kSxGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kSxDagGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCXGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCYGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCZGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kSwapGateType:
        return 1.;

      case Circuits::QuantumGateType::kPhaseGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kTGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kTdgGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kRxGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kRyGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kRzGateType:
        return 2.;  // just duplication of the pauli string... I guess this
                    // would be proportional with the number of qubits, but
                    // let's just put a constant for now

      case Circuits::QuantumGateType::kUGateType:
        return 8.;  // implemented with three rotations

      case Circuits::QuantumGateType::kCPGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRxGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRyGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRzGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCHGateType:
        return 4;  // implemented with 2 rotations, the rest are cliffords

      case Circuits::QuantumGateType::kCSxGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCSxDagGateType:
        return 8;  // implemented with 3 rotations, the rest are cliffords

      case Circuits::QuantumGateType::kCUGateType:
        return 256;  // implemented with 8 rotations

      // three qubit gates, particularly costly!
      case Circuits::QuantumGateType::kCSwapGateType:
        return 4096;  // 12 rotations!!!!!!
      case Circuits::QuantumGateType::kCCXGateType:
        return 512;  // 9 rotations!!!!!!

      default:
        return 1.;
    }
    return 1.;
  }
};

}  // namespace Simulators

#endif  // _QCSIM_PAULI_PROPAGATOR_H
