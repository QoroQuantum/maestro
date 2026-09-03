/** @file GpuMPO.h
 * Thin C++ wrapper around the optional GPU matrix-product-operator C API.
 */
#pragma once

#ifndef _GPU_MPO_H_
#define _GPU_MPO_H_

#ifdef __linux__

#include <memory>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "GpuLibrary.h"

namespace Simulators {

class GpuMPO {
 public:
  explicit GpuMPO(const std::shared_ptr<GpuLibrary>& lib)
      : lib(lib), obj(lib ? lib->CreateMPO() : nullptr) {}
  GpuMPO(const std::shared_ptr<GpuLibrary>& lib, void* obj)
      : lib(lib), obj(obj) {}
  GpuMPO() = delete;
  GpuMPO(const GpuMPO&) = delete;
  GpuMPO& operator=(const GpuMPO&) = delete;
  ~GpuMPO() { if (lib && obj) lib->DestroyMPO(obj); }

  bool Create(unsigned int n) { return lib->MPOCreate(obj, n); }
  bool CreateWithState(unsigned int n, const double* state) {
    return lib->MPOCreateWithState(obj, n, state);
  }
  bool CreateWithBasisState(unsigned int n, unsigned long long state) {
    return lib->MPOCreateWithBasisState(obj, n, state);
  }
  bool CreateWithBasisStateBits(unsigned int n,
                                const std::vector<unsigned char>& stateBits) {
    return lib->MPOCreateWithBasisStateBits(obj, n, stateBits.data());
  }
  bool CreateWithMixtureOfBasisStates(
      unsigned int n,
      const std::vector<std::pair<unsigned long long, double>>& mixture) {
    std::vector<unsigned long long> states;
    std::vector<double> weights;
    states.reserve(mixture.size());
    weights.reserve(mixture.size());
    for (const auto& [state, weight] : mixture) {
      states.push_back(state);
      weights.push_back(weight);
    }
    return lib->MPOCreateWithMixtureOfBasisStates(
        obj, n, states.data(), weights.data(),
        static_cast<int>(states.size()));
  }
  bool CreateWithMixtureOfBasisStatesBits(
      unsigned int n, const std::vector<unsigned char>& stateBitsFlat,
      const std::vector<double>& weights) {
    return lib->MPOCreateWithMixtureOfBasisStatesBits(
        obj, n, stateBitsFlat.data(), weights.data(),
        static_cast<int>(weights.size()));
  }
  void Reset() {
    if (!lib->MPOReset(obj))
      throw std::runtime_error("GPU matrix-product-operator reset failed");
  }
  bool SetInitialQubitsMap(const std::vector<long long int>& initialMap) {
    return lib->MPOSetInitialQubitsMap(obj, initialMap);
  }
  bool SetUseOptimalMeetingPosition(bool useOptimalMeetingPosition) {
    return lib->MPOSetUseOptimalMeetingPosition(obj, useOptimalMeetingPosition);
  }
  bool GetUseOptimalMeetingPosition() const {
    return lib->MPOGetUseOptimalMeetingPosition(obj);
  }
  bool SetCallbackContext(void* context) {
    return lib->MPOSetCallbackContext(obj, context);
  }
  bool SetMeetingPositionCallback(
      int64_t (*callback)(void*, const int64_t*)) {
    return lib->MPOSetMeetingPositionCallback(obj, callback);
  }
  bool SetBondDimensionsCallback(void (*callback)(void*, const int64_t*)) {
    return lib->MPOSetBondDimensionsCallback(obj, callback);
  }
  bool IsCreated() const { return lib->MPOIsCreated(obj); }
  void SetDataType(bool useDouble) {
    if (!lib->MPOSetDataType(obj, useDouble))
      throw std::runtime_error(
          "GPU matrix-product-operator precision configuration failed");
  }
  void SetCutoff(double singularValueThreshold) {
    lib->MPOSetCutoff(obj, singularValueThreshold);
  }
  double GetCutoff() const { return lib->MPOGetCutoff(obj); }
  // mode: 0 = RelativeToMax, 1 = DiscardedWeight (default). See
  // TruncationMode in the GPU library's lib/truncationmode.hpp.
  bool SetTruncationMode(int mode) {
    return lib->MPOSetTruncationMode(obj, mode);
  }
  int GetTruncationMode() const { return lib->MPOGetTruncationMode(obj); }
  bool SetGesvdJ(bool enable) { return lib->MPOSetGesvdJ(obj, enable); }
  bool GetGesvdJ() const { return lib->MPOGetGesvdJ(obj); }
  void SetMaxExtent(long int chi) { lib->MPOSetMaxExtent(obj, chi); }
  long int GetMaxExtent() const { return lib->MPOGetMaxExtent(obj); }
  std::vector<long long int> GetBondDimensions(size_t nrQubits) const {
    if (nrQubits < 2) return {};
    std::vector<long long int> bondDims(nrQubits - 1);
    if (!lib->MPOGetBondDimensions(obj, bondDims.data())) return {};
    return bondDims;
  }
  void ReCanonicalize(int centerSite = 0) {
    if (!lib->MPOReCanonicalize(obj, centerSite)) throw std::runtime_error("GPU MPO canonicalization failed");
  }
  void Trim(double cutoff = -1., long int maxExtent = -1, int centerSite = 0) {
    if (!lib->MPOTrim(obj, cutoff, maxExtent, centerSite)) throw std::runtime_error("GPU MPO trim failed");
  }
  bool Measure(unsigned int q) { return lib->MPOMeasureQubitCollapse(obj, q); }
  bool MeasureNoCollapse(unsigned int q) { return lib->MPOMeasureQubitNoCollapse(obj, q); }
  bool MeasureQubits(std::vector<int>& qubits, std::vector<int>& bits, bool collapse = true) {
    if (qubits.size() != bits.size()) throw std::invalid_argument("Measurement vectors must have equal size");
    return collapse ? lib->MPOMeasureQubitsCollapse(obj, qubits.data(), bits.data(), static_cast<int>(bits.size()))
                    : lib->MPOMeasureQubitsNoCollapse(obj, qubits.data(), bits.data(), static_cast<int>(bits.size()));
  }
  unsigned long long MeasureAll(bool collapse = true) { return collapse ? lib->MPOMeasureAllQubitsCollapse(obj) : lib->MPOMeasureAllQubitsNoCollapse(obj); }
  bool Sample(unsigned int nSamples, long int* samples, unsigned int nBits,
             int* bitOrdering) {
    return lib->MPOSample(obj, nSamples, samples, nBits, bitOrdering);
  }
  bool SampleAll(unsigned int shots, long int* samples) {
    return lib->MPOSampleAll(obj, shots, samples);
  }
  double Probability(long long outcome) const {
    return lib->MPOBasisStateProbability(obj, outcome);
  }
  std::complex<double> GetElement(long long row, long long col) const {
    double re = 0., im = 0.;
    if (!lib->MPOGetElement(obj, row, col, &re, &im)) throw std::runtime_error("GPU MPO element query failed");
    return {re, im};
  }
  void AllProbabilities(double* probabilities) {
    if (!lib->MPOAllProbabilities(obj, probabilities))
      throw std::runtime_error(
          "GPU matrix-product-operator probability enumeration failed");
  }
  double ExpectationValue(const std::string& pauli) const {
    return lib->MPOExpectationValue(obj, pauli.c_str(), pauli.size());
  }
  double QubitProbability0(unsigned int q) const { return lib->MPOQubitProbability0(obj, q); }
  double Trace() const { return lib->MPOTrace(obj); }
  double Purity() const { return lib->MPOPurity(obj); }
  double HermiticityResidual() const { return lib->MPOHermiticityResidual(obj); }
  bool IsHermitian(double eps = 1e-10) const { return lib->MPOIsHermitian(obj, eps); }
  double TraceOfSquare() const { return lib->MPOTraceOfSquare(obj); }
  void RestoreTrace() { if (!lib->MPORestoreTrace(obj)) throw std::runtime_error("GPU MPO trace restoration failed"); }
  void Hermitize() { if (!lib->MPOHermitize(obj)) throw std::runtime_error("GPU MPO hermitization failed"); }
  bool SetKrausCompletenessCheck(int mode) { return lib->MPOSetKrausCompletenessCheck(obj, mode); }
  int GetKrausCompletenessCheck() const { return lib->MPOGetKrausCompletenessCheck(obj); }
  std::vector<std::complex<double>> PartialTrace(const std::vector<int>& qubits) const {
    const size_t dim = size_t{1} << qubits.size();
    std::vector<double> raw(2 * dim * dim);
    if (!lib->MPOPartialTrace(obj, qubits.data(), static_cast<int>(qubits.size()), raw.data())) throw std::runtime_error("GPU MPO partial trace failed");
    std::vector<std::complex<double>> result(dim * dim);
    for (size_t i = 0; i < result.size(); ++i) result[i] = {raw[2*i], raw[2*i+1]};
    return result;
  }
  std::complex<double> HilbertSchmidtOverlap(const GpuMPO& other) const {
    double re = 0., im = 0.;
    if (!lib->MPOHilbertSchmidtOverlap(obj, other.obj, &re, &im)) throw std::runtime_error("GPU MPO overlap failed");
    return {re, im};
  }
  double FidelityWithStatevector(const double* state) const {
    double result = 0.;
    if (!lib->MPOFidelityWithStatevector(obj, state, &result)) throw std::runtime_error("GPU MPO fidelity failed");
    return result;
  }
  void SaveState() {
    if (!lib->MPOSaveState(obj))
      throw std::runtime_error("GPU matrix-product-operator state save failed");
  }
  void RestoreState() {
    if (!lib->MPORestoreState(obj))
      throw std::runtime_error(
          "GPU matrix-product-operator state restore failed");
  }
  void CleanSavedState() {
    if (!lib->MPOCleanSavedState(obj))
      throw std::runtime_error(
          "GPU matrix-product-operator saved-state cleanup failed");
  }
  std::unique_ptr<GpuMPO> Clone() const {
    void* cloned = lib->MPOClone(obj);
    if (!cloned) return nullptr;
    return std::make_unique<GpuMPO>(lib, cloned);
  }
  bool ApplyKraus(const std::vector<int>& qubits, int count,
                  const double* operators) {
    return lib->MPOApplyKraus(obj, qubits.size(), qubits.data(), count,
                              operators);
  }
  bool ApplyOneQubitMatrix(int qubit, const double* matrixInterleaved) {
    return lib->MPOApplyOneQubitMatrix(obj, qubit, matrixInterleaved);
  }
  bool ApplyTwoQubitMatrix(int qubit1, int qubit2,
                           const double* matrixInterleaved) {
    return lib->MPOApplyTwoQubitMatrix(obj, qubit1, qubit2,
                                       matrixInterleaved);
  }

#define GPU_MPO_CHECK(call, name)                                             \
  do {                                                                        \
    if (!(call))                                                              \
      throw std::runtime_error("GPU matrix-product-operator " name           \
                                " failed");                                   \
  } while (false)
#define GPU_MPO_GATE1(name)                                                   \
  void name(int q) { GPU_MPO_CHECK(lib->MPO##name(obj, q), #name); }
#define GPU_MPO_GATE2(name)                                                   \
  void name(int a, int b) { GPU_MPO_CHECK(lib->MPO##name(obj, a, b), #name); }
#define GPU_MPO_ROT1(name)                                                    \
  void name(int q, double x) {                                               \
    GPU_MPO_CHECK(lib->MPO##name(obj, q, x), #name);                         \
  }
#define GPU_MPO_ROT2(name)                                                    \
  void name(int a, int b, double x) {                                        \
    GPU_MPO_CHECK(lib->MPO##name(obj, a, b, x), #name);                      \
  }
  GPU_MPO_GATE1(ApplyReset)
  GPU_MPO_ROT1(ApplyBitFlipNoise) GPU_MPO_ROT1(ApplyPhaseFlipNoise)
  GPU_MPO_ROT1(ApplyDepolarizingNoise) GPU_MPO_ROT1(ApplyAmplitudeDamping)
  GPU_MPO_ROT1(ApplyPhaseDamping) GPU_MPO_GATE1(ApplyNonSelectiveMeasurement)
  GPU_MPO_GATE1(ApplyX) GPU_MPO_GATE1(ApplyY) GPU_MPO_GATE1(ApplyZ)
  GPU_MPO_GATE1(ApplyH) GPU_MPO_GATE1(ApplyS) GPU_MPO_GATE1(ApplySDG)
  GPU_MPO_GATE1(ApplyT) GPU_MPO_GATE1(ApplyTDG) GPU_MPO_GATE1(ApplySX)
  GPU_MPO_GATE1(ApplySXDG) GPU_MPO_GATE1(ApplyK)
  GPU_MPO_ROT1(ApplyP) GPU_MPO_ROT1(ApplyRx) GPU_MPO_ROT1(ApplyRy)
  GPU_MPO_ROT1(ApplyRz)
  void ApplyU(int q, double a, double b, double c, double d) {
    GPU_MPO_CHECK(lib->MPOApplyU(obj, q, a, b, c, d), "ApplyU");
  }
  GPU_MPO_GATE2(ApplyCX) GPU_MPO_GATE2(ApplyCY) GPU_MPO_GATE2(ApplyCZ)
  GPU_MPO_GATE2(ApplyCH) GPU_MPO_GATE2(ApplyCSX) GPU_MPO_GATE2(ApplyCSXDG)
  GPU_MPO_ROT2(ApplyCP) GPU_MPO_ROT2(ApplyCRx) GPU_MPO_ROT2(ApplyCRy)
  GPU_MPO_ROT2(ApplyCRz)
  GPU_MPO_GATE2(ApplySwap)
  void ApplyCU(int a, int b, double c, double d, double e, double f) {
    GPU_MPO_CHECK(lib->MPOApplyCU(obj, a, b, c, d, e, f), "ApplyCU");
  }
#undef GPU_MPO_ROT2
#undef GPU_MPO_ROT1
#undef GPU_MPO_GATE2
#undef GPU_MPO_GATE1
#undef GPU_MPO_CHECK

 private:
  std::shared_ptr<GpuLibrary> lib;
  void* obj = nullptr;
};

}  // namespace Simulators
#endif
#endif
