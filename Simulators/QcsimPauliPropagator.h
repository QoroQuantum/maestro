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


namespace Simulators {

class QcsimPauliPropagator : public QC::PauliPropagator {
 public:
  void ApplyP(int qubit, double lambda) { ApplyRZ(qubit, lambda); }

  void ApplyT(int qubit) { ApplyRZ(qubit, M_PI_4); }

  void ApplyTDG(int qubit) { ApplyRZ(qubit, -M_PI_4); }

  void ApplyU(int qubit, double theta, double phi, double lambda, double gamma = 0.0) {
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
    ApplyCSX(q2, q3);
    ApplyCX(q1, q2);

    ApplyP(q3, M_PI);
    ApplyP(q2, -M_PI_2);
    
    ApplyCSX(q2, q3);
    ApplyCX(q1, q2);

    ApplyP(q3, M_PI);
    ApplyCSX(q1, q3);
    ApplyCX(q3, q2);
  }

  void ApplyCCX(int controlQubit1, int controlQubit2, int targetQubit) {
    const int q1 = controlQubit1;  // control 1
    const int q2 = controlQubit2;  // control 2
    const int q3 = targetQubit;  // target

    ApplyCSX(q2, q3);
    ApplyCX(q1, q2);
    ApplyCSXDAG(q2, q3);
    ApplyCX(q1, q2);
    ApplyCSX(q1, q3);
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
    clone->SetOperations(std::move(GetOperations()));
    clone->SetSavePosition(GetSavePosition());
    if (IsParallelEnabled()) clone->EnableParallel();
    
    return clone;
  }
};

}  // namespace Simulators

#endif  // _QCSIM_PAULI_PROPAGATOR_H
