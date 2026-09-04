/** @file GpuDensityMatrix.h
 * Thin C++ wrapper around the optional GPU density-matrix C API.
 */
#pragma once

#ifndef _GPU_DENSITY_MATRIX_H_
#define _GPU_DENSITY_MATRIX_H_

#ifdef __linux__

#include <memory>
#include <complex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "GpuLibrary.h"

namespace Simulators {

class GpuDensityMatrix {
 public:
  explicit GpuDensityMatrix(const std::shared_ptr<GpuLibrary>& lib)
      : lib(lib), obj(lib ? lib->CreateDensityMatrix() : nullptr) {}
  GpuDensityMatrix(const std::shared_ptr<GpuLibrary>& lib, void* obj)
      : lib(lib), obj(obj) {}
  GpuDensityMatrix() = delete;
  GpuDensityMatrix(const GpuDensityMatrix&) = delete;
  GpuDensityMatrix& operator=(const GpuDensityMatrix&) = delete;
  ~GpuDensityMatrix() { if (lib && obj) lib->DestroyDensityMatrix(obj); }

  bool Create(unsigned int n) { return lib->DMCreate(obj, n); }
  bool CreateWithState(unsigned int n, const double* state) {
    return lib->DMCreateWithState(obj, n, state);
  }
  bool CreateWithBasisState(unsigned int n, unsigned long long state) {
    return lib->DMCreateWithBasisState(obj, n, state);
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
    return lib->DMCreateWithMixtureOfBasisStates(
        obj, n, states.data(), weights.data(),
        static_cast<int>(states.size()));
  }
  void Reset() {
    if (!lib->DMReset(obj))
      throw std::runtime_error("GPU density-matrix reset failed");
  }
  bool SetSeed(uint64_t seed) { return lib->DMSetSeed(obj, seed); }
  bool IsCreated() const { return lib->DMIsCreated(obj); }
  void SetDataType(bool useDouble) {
    if (!lib->DMSetDataType(obj, useDouble))
      throw std::runtime_error(
          "GPU density-matrix precision configuration failed");
  }
  bool Measure(unsigned int q) { return lib->DMMeasureQubitCollapse(obj, q); }
  bool MeasureNoCollapse(unsigned int q) { return lib->DMMeasureQubitNoCollapse(obj, q); }
  bool MeasureQubits(std::vector<int>& qubits, std::vector<int>& bits, bool collapse = true) {
    if (qubits.size() != bits.size()) throw std::invalid_argument("Measurement vectors must have equal size");
    return collapse ? lib->DMMeasureQubitsCollapse(obj, qubits.data(), bits.data(), static_cast<int>(bits.size()))
                    : lib->DMMeasureQubitsNoCollapse(obj, qubits.data(), bits.data(), static_cast<int>(bits.size()));
  }
  unsigned long long MeasureAll(bool collapse = true) { return collapse ? lib->DMMeasureAllQubitsCollapse(obj) : lib->DMMeasureAllQubitsNoCollapse(obj); }
  bool Sample(unsigned int n, long int* samples, unsigned int nBits, int* order) { return lib->DMSample(obj, n, samples, nBits, order); }
  bool SampleAll(unsigned int shots, long int* samples) {
    return lib->DMSampleAll(obj, shots, samples);
  }
  double Probability(long long outcome) const {
    return lib->DMBasisStateProbability(obj, outcome);
  }
  std::complex<double> GetElement(long long row, long long col) const {
    double re = 0., im = 0.;
    if (!lib->DMGetElement(obj, row, col, &re, &im)) throw std::runtime_error("GPU density-matrix element query failed");
    return {re, im};
  }
  void AllProbabilities(double* probabilities) {
    if (!lib->DMAllProbabilities(obj, probabilities))
      throw std::runtime_error(
          "GPU density-matrix probability enumeration failed");
  }
  double ExpectationValue(const std::string& pauli) const {
    return lib->DMExpectationValue(obj, pauli.c_str(), pauli.size());
  }
  double QubitProbability0(unsigned int q) const { return lib->DMQubitProbability0(obj, q); }
  double Trace() const { return lib->DMTrace(obj); }
  double Purity() const { return lib->DMPurity(obj); }
  bool IsHermitian(double eps = 1e-10) const { return lib->DMIsHermitian(obj, eps); }
  std::vector<std::complex<double>> PartialTrace(const std::vector<int>& qubits) const {
    const size_t dim = size_t{1} << qubits.size();
    std::vector<double> raw(2 * dim * dim);
    if (!lib->DMPartialTrace(obj, qubits.data(), static_cast<int>(qubits.size()), raw.data()))
      throw std::runtime_error("GPU density-matrix partial trace failed");
    std::vector<std::complex<double>> result(dim * dim);
    for (size_t i = 0; i < result.size(); ++i) result[i] = {raw[2*i], raw[2*i+1]};
    return result;
  }
  std::complex<double> HilbertSchmidtOverlap(const GpuDensityMatrix& other) const {
    double re = 0., im = 0.;
    if (!lib->DMHilbertSchmidtOverlap(obj, other.obj, &re, &im))
      throw std::runtime_error("GPU density-matrix overlap failed");
    return {re, im};
  }
  double FidelityWithStatevector(const double* state) const {
    double result = 0.;
    if (!lib->DMFidelityWithStatevector(obj, state, &result))
      throw std::runtime_error("GPU density-matrix fidelity failed");
    return result;
  }
  void SaveState() {
    if (!lib->DMSaveState(obj))
      throw std::runtime_error("GPU density-matrix state save failed");
  }
  void RestoreState() {
    if (!lib->DMRestoreState(obj))
      throw std::runtime_error("GPU density-matrix state restore failed");
  }
  void CleanSavedState() {
    if (!lib->DMCleanSavedState(obj))
      throw std::runtime_error("GPU density-matrix saved-state cleanup failed");
  }
  std::unique_ptr<GpuDensityMatrix> Clone() const {
    void* cloned = lib->DMClone(obj);
    if (!cloned) return nullptr;
    return std::make_unique<GpuDensityMatrix>(lib, cloned);
  }
  bool ApplyKraus(const std::vector<int>& qubits, int count,
                  const double* operators) {
    return lib->DMApplyKraus(obj, qubits.size(), qubits.data(), count,
                             operators);
  }

#define GPU_DM_CHECK(call, name)                                              \
  do {                                                                        \
    if (!(call))                                                              \
      throw std::runtime_error("GPU density-matrix " name " failed");       \
  } while (false)
#define GPU_DM_GATE1(name)                                                    \
  void name(int q) { GPU_DM_CHECK(lib->DM##name(obj, q), #name); }
#define GPU_DM_GATE2(name)                                                    \
  void name(int a, int b) { GPU_DM_CHECK(lib->DM##name(obj, a, b), #name); }
#define GPU_DM_ROT1(name)                                                     \
  void name(int q, double x) { GPU_DM_CHECK(lib->DM##name(obj, q, x), #name); }
#define GPU_DM_ROT2(name)                                                     \
  void name(int a, int b, double x) {                                        \
    GPU_DM_CHECK(lib->DM##name(obj, a, b, x), #name);                         \
  }
  GPU_DM_GATE1(ApplyReset)
  GPU_DM_ROT1(ApplyBitFlipNoise) GPU_DM_ROT1(ApplyPhaseFlipNoise)
  GPU_DM_ROT1(ApplyDepolarizingNoise) GPU_DM_ROT1(ApplyAmplitudeDamping)
  GPU_DM_ROT1(ApplyPhaseDamping) GPU_DM_GATE1(ApplyNonSelectiveMeasurement)
  GPU_DM_GATE1(ApplyX) GPU_DM_GATE1(ApplyY) GPU_DM_GATE1(ApplyZ)
  GPU_DM_GATE1(ApplyH) GPU_DM_GATE1(ApplyS) GPU_DM_GATE1(ApplySDG)
  GPU_DM_GATE1(ApplyT) GPU_DM_GATE1(ApplyTDG) GPU_DM_GATE1(ApplySX)
  GPU_DM_GATE1(ApplySXDG) GPU_DM_GATE1(ApplyK)
  GPU_DM_ROT1(ApplyP) GPU_DM_ROT1(ApplyRx) GPU_DM_ROT1(ApplyRy)
  GPU_DM_ROT1(ApplyRz)
  void ApplyU(int q, double a, double b, double c, double d) {
    GPU_DM_CHECK(lib->DMApplyU(obj, q, a, b, c, d), "ApplyU");
  }
  GPU_DM_GATE2(ApplyCX) GPU_DM_GATE2(ApplyCY) GPU_DM_GATE2(ApplyCZ)
  GPU_DM_GATE2(ApplyCH) GPU_DM_GATE2(ApplyCSX) GPU_DM_GATE2(ApplyCSXDG)
  GPU_DM_ROT2(ApplyCP) GPU_DM_ROT2(ApplyCRx) GPU_DM_ROT2(ApplyCRy)
  GPU_DM_ROT2(ApplyCRz)
  void ApplyCCX(int a, int b, int c) {
    GPU_DM_CHECK(lib->DMApplyCCX(obj, a, b, c), "ApplyCCX");
  }
  GPU_DM_GATE2(ApplySwap)
  void ApplyCSwap(int a, int b, int c) {
    GPU_DM_CHECK(lib->DMApplyCSwap(obj, a, b, c), "ApplyCSwap");
  }
  void ApplyCU(int a, int b, double c, double d, double e, double f) {
    GPU_DM_CHECK(lib->DMApplyCU(obj, a, b, c, d, e, f), "ApplyCU");
  }
#undef GPU_DM_ROT2
#undef GPU_DM_ROT1
#undef GPU_DM_GATE2
#undef GPU_DM_GATE1
#undef GPU_DM_CHECK

 private:
  std::shared_ptr<GpuLibrary> lib;
  void* obj = nullptr;
};

}  // namespace Simulators
#endif
#endif
