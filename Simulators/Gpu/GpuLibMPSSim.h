/**
 * @file GpuLibMPSSim.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The Gpu MPS library class.
 *
 * Just a wrapped aroung c api functions, should not be used directly but with a
 * adapter/bridge pattern to expose the same interface as the other ones.
 */

#pragma once

#ifndef _GPU_LIB_MPS_SIM_H_
#define _GPU_LIB_MPS_SIM_H_

#ifdef __linux__

#include <memory>
#include <vector>

#include "GpuDeviceContext.h"
#include "../TensorNetworks/TensorQueries.h"

namespace Simulators {

class GpuLibMPSSim {
 public:
  explicit GpuLibMPSSim(const std::shared_ptr<GpuLibrary>& lib, int device = -1)
      : lib(lib), obj(nullptr) {
    if (lib) {
      auto lock = lib->LockInitialization();
      if (lib->SetGpuDevice(device == -1 ? lib->GetCreationDevice() : device))
        obj = lib->CreateMPS();
    }
  }

  int GetGpuDevice() const { return lib ? lib->MPSGetGpuId(obj) : -1; }

  GpuLibMPSSim(const std::shared_ptr<GpuLibrary> &lib, void *obj)
      : lib(lib), obj(obj) {}

  GpuLibMPSSim() = delete;
  GpuLibMPSSim(const GpuLibMPSSim &) = delete;
  GpuLibMPSSim &operator=(const GpuLibMPSSim &) = delete;
  GpuLibMPSSim(GpuLibMPSSim &&) = default;
  GpuLibMPSSim &operator=(GpuLibMPSSim &&) = default;

  ~GpuLibMPSSim() {
    if (lib && obj) lib->DestroyMPS(obj);
  }

  bool Create(unsigned int nrQubits) {
    if (obj) return lib->MPSCreate(obj, nrQubits);

    return false;
  }

  bool CreateWithBasisState(unsigned int nrQubits, unsigned long long state) {
    if (obj) return lib->MPSCreateWithBasisState(obj, nrQubits, state);

    return false;
  }

  bool CreateWithBasisStateBits(unsigned int nrQubits,
                                const std::vector<unsigned char> &stateBits) {
    if (obj)
      return lib->MPSCreateWithBasisStateBits(obj, nrQubits,
                                              stateBits.data());

    return false;
  }

  bool Reset() {
    if (obj) return lib->MPSReset(obj);

    return false;
  }

  bool SetSeed(uint64_t seed) { return obj && lib->MPSSetSeed(obj, seed); }

  // Explicit compression with the current cutoff, truncation mode and maximum
  // extent: Trim re-splits over-cap bonds, ReCanonicalize repairs the chain.
  bool Trim() { return obj && lib->MPSTrim(obj); }

  bool ReCanonicalize() { return obj && lib->MPSReCanonicalize(obj); }

  bool SetInitialQubitsMap(const std::vector<long long int> &initialMap) {
    if (obj) return lib->MPSSetInitialQubitsMap(obj, initialMap);

    return false;
  }

  bool SetUseOptimalMeetingPosition(bool useOptimalMeetingPosition) {
    if (obj) return lib->MPSSetUseOptimalMeetingPosition(obj, useOptimalMeetingPosition);
    return false;
  }

  bool GetUseOptimalMeetingPosition() const {
    if (obj) return lib->MPSGetUseOptimalMeetingPosition(obj);
    return false;
  }

  bool IsValid() const {
    if (obj) return lib->MPSIsValid(obj);

    return false;
  }

  bool IsCreated() const {
    if (obj) return lib->MPSIsCreated(obj);

    return false;
  }

  bool SetDataType(int useDoublePrecision) {
    if (obj) return lib->MPSSetDataType(obj, useDoublePrecision);

    return false;
  }

  bool IsDoublePrecision() const {
    if (obj) return lib->MPSIsDoublePrecision(obj);

    return false;
  }

  bool SetCutoff(double val) {
    if (obj) return lib->MPSSetCutoff(obj, val);

    return false;
  }

  double GetCutoff() const {
    if (obj) return lib->MPSGetCutoff(obj);

    return 0.;
  }

  // mode: 0 = RelativeToMax, 1 = DiscardedWeight (default). See
  // TruncationMode in the GPU library's lib/truncationmode.hpp.
  bool SetTruncationMode(int mode) {
    if (obj) return lib->MPSSetTruncationMode(obj, mode);

    return false;
  }

  int GetTruncationMode() const {
    if (obj) return lib->MPSGetTruncationMode(obj);

    return 0;
  }

  bool SetGesvdJ(int val) {
    if (obj) return lib->MPSSetGesvdJ(obj, val);

    return false;
  }

  bool GetGesvdJ() const {
    if (obj) return lib->MPSGetGesvdJ(obj);

    return false;
  }

  // Enabling any of J/P/R clears the other two selectors in the plugin.
  bool SetGesvdP(bool enable) {
    return obj && lib->MPSSetGesvdP(obj, enable);
  }
  bool GetGesvdP() const { return lib->MPSGetGesvdP(obj); }
  bool SetGesvdR(bool enable) {
    return obj && lib->MPSSetGesvdR(obj, enable);
  }
  bool GetGesvdR() const { return lib->MPSGetGesvdR(obj); }
  int GetLastSvdAlgo() const { return lib->MPSGetLastSvdAlgo(obj); }

  bool SetMaxExtent(long int val) {
    if (obj) return lib->MPSSetMaxExtent(obj, val);

    return false;
  }

  long int GetMaxExtent() const {
    if (obj) return lib->MPSGetMaxExtent(obj);

    return 0;
  }

  int GetNrQubits() const {
    if (obj) return lib->MPSGetNrQubits(obj);

    return 0;
  }

  std::vector<long long int> GetBondDimensions(size_t nrQubits) const {
    if (!obj || nrQubits < 2) return {};
    std::vector<long long int> bondDims(nrQubits - 1);
    if (!lib->MPSGetBondDimensions(obj, bondDims.data())) return {};
    return bondDims;
  }

  std::vector<long long> GetQubitsMap() const {
    std::vector<long long> map(GetNrQubits());
    if (!lib->MPSGetQubitsMap(obj, map.data(), static_cast<int>(map.size()))) return {};
    return map;
  }
  bool SetCallbackContext(void *context) {
    if (obj) return lib->MPSSetCallbackContext(obj, context);
    return false;
  }

  bool SetMeetingPositionCallback(int64_t (*callback)(void *, const int64_t *)) {
    if (obj) return lib->MPSSetMeetingPositionCallback(obj, callback);
    return false;
  }

  bool SetBondDimensionsCallback(void (*callback)(void*, const int64_t*)) {
    if (obj) return lib->MPSSetBondDimensionsCallback(obj, callback);
    return false;
  }

  void InstallBondSummary(void (*summary)(void *, int64_t),
                          void (*full)(void *, const int64_t *)) {
    const bool ok = lib->HasMPSBondSummary()
                        ? lib->MPSSetBondDimensionSummaryCallback(obj, summary)
                        : SetBondDimensionsCallback(full);
    if (!ok)
      throw std::runtime_error("GPU MPS bond callback installation failed");
  }

  std::vector<std::complex<double>> GetStateVector() {
    const auto n = GetNrQubits();
    if (n < 0 || n > 40)
      throw std::length_error(
          "GPU MPS dense output supports at most 40 qubits");
    const auto count = TensorQueries::DenseSize(n);
    std::vector<std::complex<double>> result(count);
    if (lib->HasMPSGetStateVector()) {
      // std::complex<double> arrays have the interleaved layout required by
      // the C API. The result is private until the native call succeeds.
      if (!lib->MPSGetStateVector(obj,
                                 reinterpret_cast<double *>(result.data())))
        throw std::runtime_error("GPU MPS statevector query failed");
    } else {
      std::vector<long int> bits(n);
      for (size_t i = 0; i < count; ++i) {
        for (int q = 0; q < n; ++q) bits[q] = (i >> q) & 1;
        double re = 0., im = 0.;
        if (!Amplitude(n, bits.data(), &re, &im) || !std::isfinite(re) ||
            !std::isfinite(im))
          throw std::runtime_error("GPU MPS amplitude query failed");
        result[i] = {re, im};
      }
    }
    return result;
  }

  std::vector<double> AllProbabilities() {
    const auto n = GetNrQubits();
    if (n < 0 || n > 40)
      throw std::length_error(
          "GPU MPS dense output supports at most 40 qubits");
    std::vector<double> result(TensorQueries::DenseSize(n));
    if (lib->HasMPSAllProbabilities()) {
      if (!lib->MPSAllProbabilities(obj, result.data()))
        throw std::runtime_error("GPU MPS probability enumeration failed");
    } else {
      const auto amplitudes = GetStateVector();
      for (size_t i = 0; i < result.size(); ++i)
        result[i] = std::norm(amplitudes[i]);
    }
    return result;
  }

  void MoveAtBeginningOfChain(const std::vector<int> &qubits) {
    if (!lib->HasMPSMoveAtBeginningOfChain())
      throw std::runtime_error(
          "GPU MPS chain movement requires an updated GPU plugin");
    if (!lib->MPSMoveAtBeginningOfChain(obj, qubits.data(),
                                        TensorQueries::Count(qubits.size())))
      throw std::runtime_error("GPU MPS chain movement failed");
  }

  std::complex<double> ExpectationValueOperators(
      const std::vector<int> &qubits, const std::vector<double> &matrices) {
    if (!lib->HasMPSExpectationValueOperators())
      throw std::runtime_error(
          "GPU MPS operator expectations require an updated GPU plugin");
    double re = 0., im = 0.;
    if (!lib->MPSExpectationValueOperators(
            obj, TensorQueries::Count(qubits.size()), qubits.data(),
            matrices.data(), &re, &im))
      throw std::runtime_error("GPU MPS operator expectation failed");
    return {re, im};
  }

  std::vector<double> ExpectationValues(
      const std::vector<std::string> &paulis) {
    TensorQueries::PauliBatch batch(paulis);
    std::vector<double> values(paulis.size());
    if (paulis.empty()) return values;
    if (lib->HasMPSExpectationValues()) {
      if (!lib->MPSExpectationValues(obj, TensorQueries::Count(paulis.size()),
                                     batch.strings.data(), batch.lengths.data(),
                                     values.data()))
        throw std::runtime_error("GPU MPS batch expectation failed");
    } else {
      for (size_t i = 0; i < paulis.size(); ++i) {
        values[i] = ExpectationValue(paulis[i]);
        if (!std::isfinite(values[i]))
          throw std::runtime_error("GPU MPS expectation failed");
      }
    }
    return values;
  }

  std::unordered_map<std::vector<bool>, int64_t> SampleHistogram(
      size_t shots, std::vector<unsigned int> qubits) {
    if (shots == 0 || qubits.empty()) return {};
    if (shots > static_cast<size_t>(std::numeric_limits<long int>::max()) ||
        qubits.size() >
            static_cast<size_t>(std::numeric_limits<long int>::max()))
      throw std::length_error("GPU MPS sampling size exceeds API limits");
    for (auto q : qubits)
      if (q >= static_cast<size_t>(GetNrQubits()))
        throw std::out_of_range("GPU MPS sampled qubit is out of range");
    using Map = std::unordered_map<std::vector<bool>, int64_t>;
    auto deleter = [this](Map *map) { FreeMapForSample(map); };
    std::unique_ptr<Map, decltype(deleter)> map(GetMapForSample(), deleter);
    if (!map || !Sample(shots, qubits.size(), qubits.data(), map.get()))
      throw std::runtime_error("GPU MPS sampling failed");
    size_t total = 0;
    for (const auto &entry : *map) {
      if (entry.second < 0 || static_cast<size_t>(entry.second) > shots - total)
        throw std::runtime_error("GPU MPS returned invalid sample counts");
      total += static_cast<size_t>(entry.second);
    }
    if (total != shots)
      throw std::runtime_error("GPU MPS returned incomplete sample counts");
    return *map;
  }

  bool Amplitude(long int numFixedValues, long int *fixedValues, double *real,
                 double *imaginary) const {
    if (obj)
      return lib->MPSAmplitude(obj, numFixedValues, fixedValues, real,
                               imaginary);

    return false;
  }

  double Probability0(unsigned int qubit) const {
    if (obj) return lib->MPSProbability0(obj, qubit);

    return 0.;
  }

  bool Measure(unsigned int qubit) {
    if (obj) return lib->MPSMeasure(obj, qubit);

    return 0;
  }

  bool MeasureQubits(long int numQubits, unsigned int *qubits, int *result) {
    if (obj) return lib->MPSMeasureQubits(obj, numQubits, qubits, result);

    return false;
  }

  std::unordered_map<std::vector<bool>, int64_t> *GetMapForSample() const {
    if (lib) return lib->MPSGetMapForSample();

    return nullptr;
  }

  bool FreeMapForSample(
      std::unordered_map<std::vector<bool>, int64_t> *map) const {
    if (lib) return lib->MPSFreeMapForSample(map);

    return false;
  }

  bool Sample(long int numShots, long int numQubits, unsigned int *qubits,
              void *resultMap) {
    if (obj) return lib->MPSSample(obj, numShots, numQubits, qubits, resultMap);

    return false;
  }

  bool SampleRaw(unsigned int nSamples, long int *samples, unsigned int nBits,
                const unsigned int *bitOrdering) {
    if (obj)
      return lib->MPSSampleRaw(obj, nSamples, samples, nBits, bitOrdering);

    return false;
  }

  bool SampleAll(unsigned int nSamples, long int *samples) {
    if (obj) return lib->MPSSampleAll(obj, nSamples, samples);

    return false;
  }

  bool SaveState() {
    if (obj) return lib->MPSSaveState(obj);

    return false;
  }

  bool RestoreState() {
    if (obj) return lib->MPSRestoreState(obj);

    return false;
  }

  bool CleanSavedState() {
    if (obj) return lib->MPSCleanSavedState(obj);

    return false;
  }

  std::unique_ptr<GpuLibMPSSim> Clone() const {
    if (obj) {
      auto *cloned = lib->MPSClone(obj);
      if (!cloned) throw std::runtime_error("GPU MPS clone failed");
      return std::make_unique<GpuLibMPSSim>(lib, cloned);
    }

    return nullptr;
  }

  double ExpectationValue(const std::string &pauliString) {
    if (obj)
      return lib->MPSExpectationValue(obj, pauliString.c_str(),
                                      pauliString.length());

    return 0.0;
  }

  std::complex<double> ProjectOnZero() {
    if (obj) return lib->MPSProjectOnZero(obj);
    
    return std::complex<double>(0., 0.);
  }

  bool ApplyX(unsigned int siteA) {
    if (obj) return lib->MPSApplyX(obj, siteA);

    return false;
  }

  bool ApplyY(unsigned int siteA) {
    if (obj) return lib->MPSApplyY(obj, siteA);

    return false;
  }

  bool ApplyZ(unsigned int siteA) {
    if (obj) return lib->MPSApplyZ(obj, siteA);

    return false;
  }

  bool ApplyH(unsigned int siteA) {
    if (obj) return lib->MPSApplyH(obj, siteA);

    return false;
  }

  bool ApplyS(unsigned int siteA) {
    if (obj) return lib->MPSApplyS(obj, siteA);

    return false;
  }

  bool ApplySDG(unsigned int siteA) {
    if (obj) return lib->MPSApplySDG(obj, siteA);

    return false;
  }

  bool ApplyT(unsigned int siteA) {
    if (obj) return lib->MPSApplyT(obj, siteA);

    return false;
  }

  bool ApplyTDG(unsigned int siteA) {
    if (obj) return lib->MPSApplyTDG(obj, siteA);

    return false;
  }

  bool ApplySX(unsigned int siteA) {
    if (obj) return lib->MPSApplySX(obj, siteA);

    return false;
  }

  bool ApplySXDG(unsigned int siteA) {
    if (obj) return lib->MPSApplySXDG(obj, siteA);

    return false;
  }

  bool ApplyK(unsigned int siteA) {
    if (obj) return lib->MPSApplyK(obj, siteA);

    return false;
  }

  bool ApplyP(unsigned int siteA, double theta) {
    if (obj) return lib->MPSApplyP(obj, siteA, theta);

    return false;
  }

  bool ApplyRx(unsigned int siteA, double theta) {
    if (obj) return lib->MPSApplyRx(obj, siteA, theta);

    return false;
  }

  bool ApplyRy(unsigned int siteA, double theta) {
    if (obj) return lib->MPSApplyRy(obj, siteA, theta);

    return false;
  }

  bool ApplyRz(unsigned int siteA, double theta) {
    if (obj) return lib->MPSApplyRz(obj, siteA, theta);

    return false;
  }

  bool ApplyU(unsigned int siteA, double theta, double phi, double lambda,
              double gamma) {
    if (obj) return lib->MPSApplyU(obj, siteA, theta, phi, lambda, gamma);

    return false;
  }

  bool ApplyOneQubitMatrix(unsigned int qubit,
                           const double *matrixInterleaved) {
    if (obj) return lib->MPSApplyOneQubitMatrix(obj, qubit, matrixInterleaved);

    return false;
  }

  bool ApplyTwoQubitMatrix(unsigned int qubit1, unsigned int qubit2,
                           const double *matrixInterleaved) {
    if (obj)
      return lib->MPSApplyTwoQubitMatrix(obj, qubit1, qubit2,
                                         matrixInterleaved);

    return false;
  }

  bool ApplySwap(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->MPSApplySwap(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCX(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->MPSApplyCX(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCY(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->MPSApplyCY(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCZ(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->MPSApplyCZ(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCH(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->MPSApplyCH(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCSX(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->MPSApplyCSX(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCSXDG(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->MPSApplyCSXDG(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCP(unsigned int controlQubit, unsigned int targetQubit,
               double theta) {
    if (obj) return lib->MPSApplyCP(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCRx(unsigned int controlQubit, unsigned int targetQubit,
                double theta) {
    if (obj) return lib->MPSApplyCRx(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCRy(unsigned int controlQubit, unsigned int targetQubit,
                double theta) {
    if (obj) return lib->MPSApplyCRy(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCRz(unsigned int controlQubit, unsigned int targetQubit,
                double theta) {
    if (obj) return lib->MPSApplyCRz(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCU(unsigned int controlQubit, unsigned int targetQubit,
               double theta, double phi, double lambda, double gamma) {
    if (obj)
      return lib->MPSApplyCU(obj, controlQubit, targetQubit, theta, phi, lambda,
                             gamma);

    return false;
  }

 private:
  GpuDeviceContext lib;
  void *obj;
};

}  // namespace Simulators

#endif  // __linux__

#endif  // _GPU_LIB_MPS_SIM_H_
