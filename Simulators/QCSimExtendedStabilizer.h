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
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
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

  // Uses all 64 bits, as SetSeed does for the other QCSim backends.
  void SetSeed(uint64_t seed) { simulator->SetSeed(seed); }

  void SetMultithreading(bool enable = true) {
    simulator->SetMultithreading(enable);
  }

  bool GetMultithreading() const { return simulator->GetMultithreading(); }

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

  // Counts preserve the live and saved state and draw the same random numbers
  // as restoring the state and measuring the qubits in order for each shot.
  std::unordered_map<size_t, size_t> SampleCounts(
      const std::vector<size_t>& qubits, size_t shots) {
    return simulator->SampleCounts(qubits, shots);
  }

  std::unordered_map<std::vector<bool>, size_t> SampleCountsMany(
      const std::vector<size_t>& qubits, size_t shots) {
    return simulator->SampleCountsMany(qubits, shots);
  }

  void SaveState() { simulator->SaveState(); }

  void RestoreState() { simulator->RestoreState(); }

  std::unique_ptr<QCSimExtendedStabilizer> Clone() const {
    return std::unique_ptr<QCSimExtendedStabilizer>(
        new QCSimExtendedStabilizer(simulator->Clone()));
  }

  // A backend Clone is a complete snapshot, including its RNG position.
  // Sampling and execution workers instead get a seed from their owner's stream.
  std::unique_ptr<QCSimExtendedStabilizer> CloneWithSeed(uint64_t seed) const {
    auto clone = simulator->Clone();
    clone->SetSeed(seed);
    return std::unique_ptr<QCSimExtendedStabilizer>(
        new QCSimExtendedStabilizer(std::move(clone)));
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

  // QCSim applies these gates in one pass each, without the rotation
  // decompositions; at Clifford angles they only update the stabilizer basis.
  void ApplyU(size_t qubit, double theta, double phi, double lambda,
              double gamma = 0.0) {
    // A global phase has no observable effect for a non-controlled U gate.
    (void)gamma;
    simulator->ApplyU(qubit, theta, phi, lambda);
  }

  void ApplyCH(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCH(targetQubit, controlQubit);
  }

  void ApplyCU(size_t controlQubit, size_t targetQubit, double theta,
               double phi, double lambda, double gamma = 0.0) {
    simulator->ApplyCU(targetQubit, controlQubit, theta, phi, lambda, gamma);
  }

  void ApplyCRX(size_t controlQubit, size_t targetQubit, double angle) {
    simulator->ApplyCRx(targetQubit, controlQubit, angle);
  }

  void ApplyCRx(size_t controlQubit, size_t targetQubit, double angle) {
    ApplyCRX(controlQubit, targetQubit, angle);
  }

  void ApplyCRY(size_t controlQubit, size_t targetQubit, double angle) {
    simulator->ApplyCRy(targetQubit, controlQubit, angle);
  }

  void ApplyCRy(size_t controlQubit, size_t targetQubit, double angle) {
    ApplyCRY(controlQubit, targetQubit, angle);
  }

  void ApplyCRZ(size_t controlQubit, size_t targetQubit, double angle) {
    simulator->ApplyCRz(targetQubit, controlQubit, angle);
  }

  void ApplyCRz(size_t controlQubit, size_t targetQubit, double angle) {
    ApplyCRZ(controlQubit, targetQubit, angle);
  }

  void ApplyCP(size_t controlQubit, size_t targetQubit, double lambda) {
    simulator->ApplyCP(targetQubit, controlQubit, lambda);
  }

  void ApplyCS(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCS(targetQubit, controlQubit);
  }

  void ApplyCSDAG(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCSdg(targetQubit, controlQubit);
  }

  void ApplyCSX(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCSx(targetQubit, controlQubit);
  }

  void ApplyCSx(size_t controlQubit, size_t targetQubit) {
    ApplyCSX(controlQubit, targetQubit);
  }

  void ApplyCSXDAG(size_t controlQubit, size_t targetQubit) {
    simulator->ApplyCSxDag(targetQubit, controlQubit);
  }

  void ApplyCSxDAG(size_t controlQubit, size_t targetQubit) {
    ApplyCSXDAG(controlQubit, targetQubit);
  }

  void ApplyCSwap(size_t controlQubit, size_t targetQubit1,
                  size_t targetQubit2) {
    simulator->ApplyCSwap(targetQubit1, targetQubit2, controlQubit);
  }

  void ApplyCCX(size_t controlQubit1, size_t controlQubit2,
                size_t targetQubit) {
    simulator->ApplyCCX(targetQubit, controlQubit1, controlQubit2);
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
