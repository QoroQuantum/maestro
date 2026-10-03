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

  // U and controlled gates are inherited from QCSim's native packed API.
  // Each is one recorded operation, so approximation intervals count full gates.

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

    clone->ShareOperationsFrom(*this);

    clone->SetSavePosition(GetSavePosition());

    clone->SetSamplingCacheMaxNodes(GetSamplingCacheMaxNodes());
    if (IsParallelEnabled()) clone->EnableParallel(GetThreadCount());

    return clone;
  }

  static double GetSamplingCost(
      const std::shared_ptr<Circuits::Circuit<>>& circuit,
      size_t nrQubitsSampled, size_t samples) {
    if (nrQubitsSampled == 0) return 0.0;
    return samples * GetCost(circuit) * exp2(nrQubitsSampled - 1);
  }

  static double GetCost(const std::shared_ptr<Circuits::Circuit<>>& circuit) {
    if (!circuit) return 0.0;

    // Conservative branching estimates, not elapsed time: commuting inputs and
    // deduplication often reduce the actual expansion.
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
        return 3.;  // at most three Pauli outputs

      case Circuits::QuantumGateType::kCPGateType:
        return 4.;
      case Circuits::QuantumGateType::kCRxGateType:
        return 4.;
      case Circuits::QuantumGateType::kCRyGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRzGateType:
        return 4.;
      case Circuits::QuantumGateType::kCHGateType:
        return 4.;
      case Circuits::QuantumGateType::kCSxGateType:
        return 4.;
      case Circuits::QuantumGateType::kCSxDagGateType:
        return 4.;

      case Circuits::QuantumGateType::kCUGateType:
        return 8.;

      case Circuits::QuantumGateType::kCCXGateType:
        return 4.;
      case Circuits::QuantumGateType::kCSwapGateType:
        return 4.;

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
        return 3.;  // at most three Pauli outputs

      case Circuits::QuantumGateType::kCPGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRxGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRyGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCRzGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCHGateType:
        return 4.;  // maximum outputs of the native local action

      case Circuits::QuantumGateType::kCSxGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCSxDagGateType:
        return 4.;

      case Circuits::QuantumGateType::kCUGateType:
        return 8.;  // includes a nonzero controlled global phase

      // Native three-qubit actions have at most four outputs.
      case Circuits::QuantumGateType::kCSwapGateType:
        return 4.;
      case Circuits::QuantumGateType::kCCXGateType:
        return 4.;

      default:
        return 1.;
    }
    return 1.;
  }
};

}  // namespace Simulators

#endif  // _QCSIM_PAULI_PROPAGATOR_H
