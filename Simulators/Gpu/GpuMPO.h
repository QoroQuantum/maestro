/** @file GpuMPO.h
 * Thin C++ wrapper around the optional GPU matrix-product-operator C API.
 */
#pragma once

#ifndef _GPU_MPO_H_
#define _GPU_MPO_H_

#ifdef __linux__

#include <memory>
#include <limits>
#include <unordered_map>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "GpuDeviceContext.h"
#include "../TensorNetworks/TensorQueries.h"

namespace Simulators {

class GpuMPO {
 public:
  explicit GpuMPO(const std::shared_ptr<GpuLibrary>& lib, int device = -1)
      : lib(lib), obj(nullptr) {
    if (lib) {
      auto lock = lib->LockInitialization();
      if (lib->SetGpuDevice(device == -1 ? lib->GetCreationDevice() : device))
        obj = lib->CreateMPO();
    }
  }

  int GetGpuDevice() const { return lib ? lib->MPOGetGpuId(obj) : -1; }

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
  bool SetSeed(uint64_t seed) { return lib->MPOSetSeed(obj, seed); }
  bool SetInitialQubitsMap(const std::vector<long long int>& initialMap) {
    return lib->MPOSetInitialQubitsMap(obj, initialMap);
  }
  bool SetUseOptimalMeetingPosition(bool useOptimalMeetingPosition) {
    return lib->MPOSetUseOptimalMeetingPosition(obj, useOptimalMeetingPosition);
  }
  bool GetUseOptimalMeetingPosition() const {
    return lib->MPOGetUseOptimalMeetingPosition(obj);
  }
  std::vector<long long> GetQubitsMap() const {
    std::vector<long long> map(lib->MPOGetNrQubits(obj));
    if (!lib->MPOGetQubitsMap(obj, map.data(), static_cast<int>(map.size()))) return {};
    return map;
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
  void InstallBondSummary(void (*summary)(void*, int64_t),
                          void (*full)(void*, const int64_t*)) {
    const bool ok = lib->HasMPOBondSummary()
                        ? lib->MPOSetBondDimensionSummaryCallback(obj, summary)
                        : SetBondDimensionsCallback(full);
    if (!ok)
      throw std::runtime_error("GPU MPO bond callback installation failed");
  }
  std::vector<double> ExpectationValues(
      const std::vector<std::string>& paulis) const {
    TensorQueries::PauliBatch batch(paulis);
    std::vector<double> values(paulis.size());
    if (paulis.empty()) return values;
    if (lib->HasMPOExpectationValues()) {
      if (!lib->MPOExpectationValues(obj, TensorQueries::Count(paulis.size()),
                                     batch.strings.data(), batch.lengths.data(),
                                     values.data()))
        throw std::runtime_error("GPU MPO batch expectation failed");
    } else {
      for (size_t i = 0; i < paulis.size(); ++i)
        values[i] = ExpectationValueComplex(paulis[i]).real();
    }
    return values;
  }
  std::vector<std::complex<double>> ExpectationValuesComplex(
      const std::vector<std::string>& paulis, bool normalized = true) const {
    TensorQueries::PauliBatch batch(paulis);
    std::vector<std::complex<double>> values(paulis.size());
    if (paulis.empty()) return values;
    if (lib->HasMPOExpectationValuesComplex()) {
      std::vector<double> re(paulis.size()), im(paulis.size());
      if (!lib->MPOExpectationValuesComplex(
              obj, TensorQueries::Count(paulis.size()), batch.strings.data(),
              batch.lengths.data(), normalized, re.data(), im.data()))
        throw std::runtime_error("GPU MPO complex batch expectation failed");
      for (size_t i = 0; i < paulis.size(); ++i) values[i] = {re[i], im[i]};
    } else {
      for (size_t i = 0; i < paulis.size(); ++i)
        values[i] = ExpectationValueComplex(paulis[i], normalized);
    }
    return values;
  }
  bool IsCreated() const { return lib->MPOIsCreated(obj); }
  bool IsDoublePrecision() const { return lib->MPOIsDoublePrecision(obj); }
  void SetDataType(bool useDouble) {
    if (!lib->MPOSetDataType(obj, useDouble))
      throw std::runtime_error(
          "GPU matrix-product-operator precision configuration failed");
  }
  void SetCutoff(double singularValueThreshold) {
    if (!lib->MPOSetCutoff(obj, singularValueThreshold))
      throw std::runtime_error("GPU MPO cutoff configuration failed");
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
  // Enabling any of J/P/R clears the other two selectors in the plugin.
  bool SetGesvdP(bool enable) {
    return obj && lib->MPOSetGesvdP(obj, enable);
  }
  bool GetGesvdP() const { return lib->MPOGetGesvdP(obj); }
  bool SetGesvdR(bool enable) {
    return obj && lib->MPOSetGesvdR(obj, enable);
  }
  bool GetGesvdR() const { return lib->MPOGetGesvdR(obj); }
  int GetLastSvdAlgo() const { return lib->MPOGetLastSvdAlgo(obj); }

  void SetMaxExtent(long int chi) {
    if (!lib->MPOSetMaxExtent(obj, chi))
      throw std::runtime_error("GPU MPO bond cap configuration failed");
  }
  long int GetMaxExtent() const { return lib->MPOGetMaxExtent(obj); }
  std::vector<long long int> GetBondDimensions(size_t nrQubits) const {
    if (nrQubits < 2) return {};
    std::vector<long long int> bondDims(nrQubits - 1);
    if (!lib->MPOGetBondDimensions(obj, bondDims.data())) return {};
    return bondDims;
  }
  void ReCanonicalize() {
    if (!lib->MPOReCanonicalize(obj))
      throw std::runtime_error("GPU MPO canonicalization failed");
  }
  void Trim(double cutoff = -1., long int maxExtent = -1) {
    if (!lib->MPOTrim(obj, cutoff, maxExtent))
      throw std::runtime_error("GPU MPO trim failed");
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
    if (qubits.size() > 13)
      throw std::invalid_argument(
          "GPU MPO partial trace supports at most 13 qubits");
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
  // Bit vectors are in logical-qubit order. Histogram entries instead follow
  // the caller's qubit order (indices must be distinct).
  double ProbabilityBits(const std::vector<unsigned char>& bits) const {
    ValidateBits(bits);
    const double probability =
        lib->MPOBasisStateProbabilityBits(obj, bits.data());
    if (!std::isfinite(probability))
      throw std::runtime_error(
          "GPU MPO probability requires a safely positive trace");
    return probability;
  }
  std::complex<double> GetElementBits(
      const std::vector<unsigned char>& row,
      const std::vector<unsigned char>& col) const {
    ValidateBits(row);
    ValidateBits(col);
    double re = 0., im = 0.;
    if (!lib->MPOGetElementBits(obj, row.data(), col.data(), &re, &im))
      throw std::runtime_error("GPU MPO element query failed");
    return {re, im};
  }
  std::complex<double> TraceComplex() const {
    double re = 0., im = 0.;
    if (!lib->MPOTraceComplex(obj, &re, &im))
      throw std::runtime_error("GPU MPO complex trace failed");
    return {re, im};
  }
  std::complex<double> ExpectationValueComplex(const std::string& pauli,
                                               bool normalized = true) const {
    double re = 0., im = 0.;
    const bool ok = normalized
                        ? lib->MPOExpectationValueComplex(
                              obj, pauli.c_str(), pauli.size(), &re, &im)
                        : lib->MPOUnnormalizedExpectationValue(
                              obj, pauli.c_str(), pauli.size(), &re, &im);
    if (!ok) throw std::runtime_error("GPU MPO complex expectation failed");
    return {re, im};
  }
  std::vector<std::complex<double>> GetDensityMatrix(
      bool normalized = true) const {
    const int n = lib->MPOGetNrQubits(obj);
    if (n < 0 || n > 13)
      throw std::invalid_argument(
          "GPU MPO dense output supports at most 13 qubits");
    const size_t count = size_t{1} << (2 * n);
    std::vector<double> raw(2 * count);
    const bool ok = normalized
                        ? lib->MPOGetDensityMatrix(obj, raw.data())
                        : lib->MPOGetUnnormalizedDensityMatrix(obj, raw.data());
    if (!ok) throw std::runtime_error("GPU MPO density matrix query failed");
    std::vector<std::complex<double>> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = {raw[2 * i], raw[2 * i + 1]};
    return result;
  }
  bool SetRestoreTraceAfterTruncation(bool enable) {
    return lib->MPOSetRestoreTraceAfterTruncation(obj, enable) == 1;
  }
  bool GetRestoreTraceAfterTruncation() const {
    return lib->MPOGetRestoreTraceAfterTruncation(obj) == 1;
  }
  bool SetHermitizeAfterTruncation(bool enable) {
    return lib->MPOSetHermitizeAfterTruncation(obj, enable) == 1;
  }
  bool GetHermitizeAfterTruncation() const {
    return lib->MPOGetHermitizeAfterTruncation(obj) == 1;
  }
  void MoveAtBeginningOfChain(const std::vector<int>& qubits) {
    if (!lib->MPOMoveAtBeginningOfChain(obj, qubits.data(), qubits.size()))
      throw std::runtime_error("GPU MPO chain routing failed");
  }
  bool ApplyOperator(const std::vector<int>& qubits, const double* matrix,
                     bool normalize = false) {
    return normalize ? lib->MPOApplyOperatorAndNormalize(
                           obj, qubits.size(), qubits.data(), matrix) == 1
                     : lib->MPOApplyOperator(obj, qubits.size(), qubits.data(),
                                             matrix) == 1;
  }
  std::vector<unsigned char> SampleBits(unsigned int shots,
                                        const std::vector<int>& qubits) {
    if (!shots || qubits.empty()) return {};
    if (qubits.size() > std::numeric_limits<unsigned int>::max() ||
        qubits.size() > std::numeric_limits<size_t>::max() / shots)
      throw std::length_error("GPU MPO sample buffer too large");
    std::vector<unsigned char> result(size_t(shots) * qubits.size());
    if (!lib->MPOSampleBits(obj, shots, qubits.size(), qubits.data(),
                            result.data()))
      throw std::runtime_error("GPU MPO bit sampling failed");
    return result;
  }
  std::unordered_map<std::vector<bool>, int64_t> SampleHistogram(
      size_t shots, std::vector<unsigned int> qubits) {
    if (!shots || qubits.empty()) return {};
    if (shots > std::numeric_limits<unsigned int>::max() ||
        shots > static_cast<size_t>(std::numeric_limits<long int>::max()) ||
        qubits.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("GPU MPO sampling size exceeds API limits");
    using Map = std::unordered_map<std::vector<bool>, int64_t>;
    auto deleter = [this](Map* map) { lib->MPOFreeMapForSample(map); };
    std::unique_ptr<Map, decltype(deleter)> map(
        static_cast<Map*>(lib->MPOGetMapForSample()), deleter);
    if (!map || !lib->MPOSampleHistogram(obj, shots, qubits.size(),
                                         qubits.data(), map.get()))
      throw std::runtime_error("GPU MPO histogram sampling failed");
    return *map;
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
  void ValidateBits(const std::vector<unsigned char>& bits) const {
    if (bits.size() != static_cast<size_t>(lib->MPOGetNrQubits(obj)))
      throw std::invalid_argument("GPU MPO bit vector width " +
                                  std::to_string(bits.size()) +
                                  " does not match register width " +
                                  std::to_string(lib->MPOGetNrQubits(obj)));
  }
  GpuDeviceContext lib;
  void* obj = nullptr;
};

}  // namespace Simulators
#endif
#endif
