/**
 * @file QCSimExtendedStabilizer.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The QCSim extended stabilizer class.
 *
 * The main role is to extend the QCSim extended stabilizer with the complete
 * gate interface used by Maestro.
 */

#pragma once

#ifndef _QCSIM_EXTENDED_STABILIZER_H
#define _QCSIM_EXTENDED_STABILIZER_H 1

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ExtendedStabilizer.h"

namespace Simulators {

class QCSimExtendedStabilizer {
 public:
  explicit QCSimExtendedStabilizer(size_t nrQubits)
      : simulator(std::make_unique<QC::ExtendedStabilizer>(nrQubits)) {}

  QCSimExtendedStabilizer(
      size_t nrQubits,
      const QC::ExtendedStabilizerApproximationPolicy& policy)
      : simulator(std::make_unique<QC::ExtendedStabilizer>(nrQubits, policy)) {}

  size_t GetNrQubits() const { return simulator->GetNrQubits(); }

  void Reset(size_t nrQubits) { simulator->Reset(nrQubits); }

  void SetRandomSeed(std::mt19937::result_type seed) {
    simulator->SetRandomSeed(seed);
  }

  void ApplyH(size_t qubit) { simulator->ApplyH(qubit); }

  void ApplyS(size_t qubit) { simulator->ApplyS(qubit); }

  void ApplyX(size_t qubit) { simulator->ApplyX(qubit); }

  void ApplyY(size_t qubit) { simulator->ApplyY(qubit); }

  void ApplyZ(size_t qubit) { simulator->ApplyZ(qubit); }

  void ApplyK(size_t qubit) { simulator->ApplyK(qubit); }

  bool Measure(size_t qubit) { return simulator->Measure(qubit); }

  double GetQubitProbability(size_t qubit) const {
    return simulator->GetQubitProbability(qubit);
  }

  double ExpectationValue(const std::string& pauliString) const {
    return simulator->ExpectationValue(pauliString);
  }

  void SaveState() { simulator->SaveState(); }

  void RestoreState() { simulator->RestoreState(); }

  std::unique_ptr<QCSimExtendedStabilizer> Clone() const {
    return std::unique_ptr<QCSimExtendedStabilizer>(
        new QCSimExtendedStabilizer(simulator->Clone()));
  }

  const std::vector<QC::ExtendedFrame>& GetFrames() const noexcept {
    return simulator->GetFrames();
  }

  const QC::ExtendedStabilizerApproximationPolicy& GetApproximationPolicy()
      const noexcept {
    return simulator->GetApproximationPolicy();
  }

  const QC::ExtendedStabilizerApproximationStatistics&
  GetApproximationStatistics() const noexcept {
    return simulator->GetApproximationStatistics();
  }

  double GetApproximationErrorBound() const noexcept {
    return simulator->GetApproximationErrorBound();
  }

  void SetApproximationPolicy(
      const QC::ExtendedStabilizerApproximationPolicy& policy) {
    simulator->SetApproximationPolicy(policy);
  }

  // Keep the gate names and control/target order consistent with the other
  // Maestro QCSim wrappers. QCSim's ExtendedStabilizer uses target/control.
  void ApplySDG(size_t qubit) { simulator->ApplySdg(qubit); }

  void ApplySX(size_t qubit) { simulator->ApplySx(qubit); }

  void ApplySXDG(size_t qubit) { simulator->ApplySxDag(qubit); }

  void ApplySxDAG(size_t qubit) { ApplySXDG(qubit); }

  void ApplyCX(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCX(targetQubit, controlQubit);
  }

  void ApplyCY(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCY(targetQubit, controlQubit);
  }

  void ApplyCZ(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCZ(targetQubit, controlQubit);
  }

  void ApplySWAP(size_t qubit1, size_t qubit2) {
    simulator->ApplySwap(qubit1, qubit2);
  }

  void ApplyISWAP(size_t qubit1, size_t qubit2) {
    simulator->ApplyISwap(qubit1, qubit2);
  }

  void ApplyISWAPDG(size_t qubit1, size_t qubit2) {
    simulator->ApplyISwapDag(qubit1, qubit2);
  }

  void ApplyRX(size_t qubit, double angle) {
    simulator->ApplyRx(qubit, angle);
  }

  void ApplyRY(size_t qubit, double angle) {
    simulator->ApplyRy(qubit, angle);
  }

  void ApplyRZ(size_t qubit, double angle) {
    simulator->ApplyRz(qubit, angle);
  }

  void ApplyP(size_t qubit, double lambda) { ApplyRZ(qubit, lambda); }

  void ApplyT(size_t qubit) { ApplyRZ(qubit, kPi / 4.0); }

  void ApplyTDG(size_t qubit) { ApplyRZ(qubit, -kPi / 4.0); }

  void ApplyU(size_t qubit, double theta, double phi, double lambda,
              double gamma = 0.0) {
    // A global phase has no observable effect for a non-controlled U gate.
    (void)gamma;
    ApplyRZ(qubit, lambda);
    ApplyRY(qubit, theta);
    ApplyRZ(qubit, phi);
  }

  void ApplyCH(size_t controlQubit, size_t targetQubit) {
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

  void ApplyCU(size_t controlQubit, size_t targetQubit, double theta,
               double phi, double lambda, double gamma = 0.0) {
    if (gamma != 0.0) ApplyP(controlQubit, gamma);

    const double lambdaPlusPhiHalf = 0.5 * (lambda + phi);
    const double halfTheta = 0.5 * theta;
    ApplyP(targetQubit, 0.5 * (lambda - phi));
    ApplyP(controlQubit, lambdaPlusPhiHalf);
    ApplyCX(controlQubit, targetQubit);
    ApplyU(targetQubit, -halfTheta, 0.0, -lambdaPlusPhiHalf);
    ApplyCX(controlQubit, targetQubit);
    ApplyU(targetQubit, halfTheta, phi, 0.0);
  }

  void ApplyCRX(size_t controlQubit, size_t targetQubit, double angle) {
    const double halfAngle = angle * 0.5;

    ApplyH(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyRZ(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRZ(targetQubit, halfAngle);
    ApplyH(targetQubit);
  }

  void ApplyCRx(size_t controlQubit, size_t targetQubit, double angle) {
    ApplyCRX(controlQubit, targetQubit, angle);
  }

  void ApplyCRY(size_t controlQubit, size_t targetQubit, double angle) {
    const double halfAngle = angle * 0.5;
    ApplyRY(targetQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRY(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
  }

  void ApplyCRy(size_t controlQubit, size_t targetQubit, double angle) {
    ApplyCRY(controlQubit, targetQubit, angle);
  }

  void ApplyCRZ(size_t controlQubit, size_t targetQubit, double angle) {
    const double halfAngle = angle * 0.5;

    ApplyRZ(targetQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyRZ(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
  }

  void ApplyCRz(size_t controlQubit, size_t targetQubit, double angle) {
    ApplyCRZ(controlQubit, targetQubit, angle);
  }

  void ApplyCP(size_t controlQubit, size_t targetQubit, double lambda) {
    const double halfAngle = lambda * 0.5;
    ApplyP(controlQubit, halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyP(targetQubit, -halfAngle);
    ApplyCX(controlQubit, targetQubit);
    ApplyP(targetQubit, halfAngle);
  }

  void ApplyCS(size_t controlQubit, size_t targetQubit) {
    ApplyT(controlQubit);
    ApplyT(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyTDG(targetQubit);
    ApplyCX(controlQubit, targetQubit);
  }

  void ApplyCSDAG(size_t controlQubit, size_t targetQubit) {
    ApplyCX(controlQubit, targetQubit);
    ApplyT(targetQubit);
    ApplyCX(controlQubit, targetQubit);
    ApplyTDG(controlQubit);
    ApplyTDG(targetQubit);
  }

  void ApplyCSX(size_t controlQubit, size_t targetQubit) {
    ApplyH(targetQubit);
    ApplyCS(controlQubit, targetQubit);
    ApplyH(targetQubit);
  }

  void ApplyCSx(size_t controlQubit, size_t targetQubit) {
    ApplyCSX(controlQubit, targetQubit);
  }

  void ApplyCSXDAG(size_t controlQubit, size_t targetQubit) {
    ApplyH(targetQubit);
    ApplyCSDAG(controlQubit, targetQubit);
    ApplyH(targetQubit);
  }

  void ApplyCSxDAG(size_t controlQubit, size_t targetQubit) {
    ApplyCSXDAG(controlQubit, targetQubit);
  }

  void ApplyCSwap(size_t controlQubit, size_t targetQubit1,
                  size_t targetQubit2) {
    const size_t q1 = controlQubit;
    const size_t q2 = targetQubit1;
    const size_t q3 = targetQubit2;

    ApplyCX(q3, q2);
    ApplyCSX(q2, q3);
    ApplyCX(q1, q2);

    ApplyP(q3, kPi);
    ApplyP(q2, -kPi / 2.0);

    ApplyCSX(q2, q3);
    ApplyCX(q1, q2);

    ApplyP(q3, kPi);
    ApplyCSX(q1, q3);
    ApplyCX(q3, q2);
  }

  void ApplyCCX(size_t controlQubit1, size_t controlQubit2,
                size_t targetQubit) {
    const size_t q1 = controlQubit1;
    const size_t q2 = controlQubit2;
    const size_t q3 = targetQubit;

    ApplyCSX(q2, q3);
    ApplyCX(q1, q2);
    ApplyCSXDAG(q2, q3);
    ApplyCX(q1, q2);
    ApplyCSX(q1, q3);
  }

 private:
  explicit QCSimExtendedStabilizer(
      std::unique_ptr<QC::ExtendedStabilizer> simulatorToOwn)
      : simulator(std::move(simulatorToOwn)) {}

  std::unique_ptr<QC::ExtendedStabilizer> simulator;
  static constexpr double kPi = 3.141592653589793238462643383279502884;
};

}  // namespace Simulators

#endif  // _QCSIM_EXTENDED_STABILIZER_H
