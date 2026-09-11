// Owns one distributed state and keeps its originating plugin alive.
#pragma once
#ifdef __linux__
#include "DistributedGpuLibrary.h"
#include <cmath>
#include <utility>
namespace Simulators {
class DistributedGpuLibStateVectorSim {
 public:
  DistributedGpuLibStateVectorSim(std::shared_ptr<DistributedGpuLibrary> lib,
                                  void *obj)
      : lib(std::move(lib)), obj(obj) {
    if (!obj) throw std::runtime_error("Null distributed GPU state");
  }
  virtual ~DistributedGpuLibStateVectorSim() { lib->DestroyNative(obj); }
  DistributedGpuLibStateVectorSim(const DistributedGpuLibStateVectorSim &) =
      delete;
  DistributedGpuLibStateVectorSim &operator=(
      const DistributedGpuLibStateVectorSim &) = delete;
  DistributedGpuLibStateVectorSim(
      DistributedGpuLibStateVectorSim &&other) noexcept
      : lib(other.lib), obj(std::exchange(other.obj, nullptr)) {}
  DistributedGpuLibStateVectorSim &operator=(
      DistributedGpuLibStateVectorSim &&) = delete;
  std::unique_ptr<DistributedGpuLibStateVectorSim> Clone() const {
    return std::make_unique<DistributedGpuLibStateVectorSim>(
        lib, lib->CloneNative(obj));
  }
  int CheckLicense() const {
    auto result = lib->CheckLicense(obj);
    lib->Check(result, "CheckLicense");
    return result;
  }
  int GetBackend() const {
    auto result = lib->GetBackend(obj);
    if (result < 0) lib->Fail("GetBackend");
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
    return result;
  }
  int SetExExecutionConfig(
      const DistributedGpuApi::MgdExExecutionConfig *config) const {
    auto result = lib->SetExExecutionConfig(obj, config);
    lib->Check(result, "SetExExecutionConfig");
    return result;
  }
  int SetDataType(int useDoublePrecision) const {
    auto result = lib->SetDataType(obj, useDoublePrecision);
    lib->Check(result, "SetDataType");
    return result;
  }
  int SetSeed(unsigned long long seed) const {
    auto result = lib->SetSeed(obj, seed);
    lib->Check(result, "SetSeed");
    return result;
  }
  int IsDoublePrecision() const {
    auto result = lib->IsDoublePrecision(obj);
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
    return result;
  }
  int GetNrQubits() const {
    auto result = lib->GetNrQubits(obj);
    if (result < 0) lib->Fail("GetNrQubits");
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
    return result;
  }
  int GetStateVectorGpuId() const {
    auto result = lib->GetStateVectorGpuId(obj);
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
    return result;
  }
  int Create(unsigned int nrQubits) const {
    auto result = lib->Create(obj, nrQubits);
    lib->Check(result, "Create");
    return result;
  }
  int CreateWithState(unsigned int nrQubits, const double *state) const {
    auto result = lib->CreateWithState(obj, nrQubits, state);
    lib->Check(result, "CreateWithState");
    return result;
  }
  int Reset() const {
    auto result = lib->Reset(obj);
    lib->Check(result, "Reset");
    return result;
  }
  int MeasureQubitCollapse(int qubitIndex) const {
    auto result = lib->MeasureQubitCollapse(obj, qubitIndex);
    if (result != 0 && result != 1) lib->Fail("MeasureQubitCollapse");
    return result;
  }
  int MeasureQubitNoCollapse(int qubitIndex) const {
    auto result = lib->MeasureQubitNoCollapse(obj, qubitIndex);
    if (result != 0 && result != 1) lib->Fail("MeasureQubitNoCollapse");
    return result;
  }
  int MeasureQubitsCollapse(int *qubits, int *bitstring,
                            int bitstringLen) const {
    auto result =
        lib->MeasureQubitsCollapse(obj, qubits, bitstring, bitstringLen);
    lib->Check(result, "MeasureQubitsCollapse");
    return result;
  }
  int MeasureQubitsNoCollapse(int *qubits, int *bitstring,
                              int bitstringLen) const {
    auto result =
        lib->MeasureQubitsNoCollapse(obj, qubits, bitstring, bitstringLen);
    lib->Check(result, "MeasureQubitsNoCollapse");
    return result;
  }
  unsigned long long MeasureAllQubitsCollapse() const {
    auto result = lib->MeasureAllQubitsCollapse(obj);
    if (result == UINT64_MAX) lib->Fail("MeasureAllQubitsCollapse");
    return result;
  }
  unsigned long long MeasureAllQubitsNoCollapse() const {
    auto result = lib->MeasureAllQubitsNoCollapse(obj);
    if (result == UINT64_MAX) lib->Fail("MeasureAllQubitsNoCollapse");
    return result;
  }
  int SaveState() const {
    auto result = lib->SaveState(obj);
    lib->Check(result, "SaveState");
    return result;
  }
  int SaveStateToHost() const {
    auto result = lib->SaveStateToHost(obj);
    lib->Check(result, "SaveStateToHost");
    return result;
  }
  int SaveStateDestructive() const {
    auto result = lib->SaveStateDestructive(obj);
    lib->Check(result, "SaveStateDestructive");
    return result;
  }
  int RestoreStateFreeSaved() const {
    auto result = lib->RestoreStateFreeSaved(obj);
    lib->Check(result, "RestoreStateFreeSaved");
    return result;
  }
  int RestoreStateNoFreeSaved() const {
    auto result = lib->RestoreStateNoFreeSaved(obj);
    lib->Check(result, "RestoreStateNoFreeSaved");
    return result;
  }
  void FreeSavedState() const {
    lib->FreeSavedState(obj);
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
  }
  int Sample(unsigned int nSamples, long int *samples, unsigned int nBits,
             int *bitOrdering) const {
    auto result = lib->Sample(obj, nSamples, samples, nBits, bitOrdering);
    lib->Check(result, "Sample");
    return result;
  }
  int SampleAll(unsigned int nSamples, long int *samples) const {
    auto result = lib->SampleAll(obj, nSamples, samples);
    lib->Check(result, "SampleAll");
    return result;
  }
  int Amplitude(long long int state, double *real, double *imaginary) const {
    auto result = lib->Amplitude(obj, state, real, imaginary);
    lib->Check(result, "Amplitude");
    return result;
  }
  double Probability(int *qubits, int *mask, int len) const {
    auto result = lib->Probability(obj, qubits, mask, len);
    if (!std::isfinite(result)) lib->Fail("Probability");
    return result;
  }
  double BasisStateProbability(long long int state) const {
    auto result = lib->BasisStateProbability(obj, state);
    if (!std::isfinite(result)) lib->Fail("BasisStateProbability");
    return result;
  }
  int AllProbabilities(double *probabilities) const {
    auto result = lib->AllProbabilities(obj, probabilities);
    lib->Check(result, "AllProbabilities");
    return result;
  }
  double ExpectationValue(const char *pauliString, int len) const {
    auto result = lib->ExpectationValue(obj, pauliString, len);
    if (!std::isfinite(result)) lib->Fail("ExpectationValue");
    return result;
  }
  int ApplyX(int qubit) const {
    auto result = lib->ApplyX(obj, qubit);
    lib->Check(result, "ApplyX");
    return result;
  }
  int ApplyY(int qubit) const {
    auto result = lib->ApplyY(obj, qubit);
    lib->Check(result, "ApplyY");
    return result;
  }
  int ApplyZ(int qubit) const {
    auto result = lib->ApplyZ(obj, qubit);
    lib->Check(result, "ApplyZ");
    return result;
  }
  int ApplyH(int qubit) const {
    auto result = lib->ApplyH(obj, qubit);
    lib->Check(result, "ApplyH");
    return result;
  }
  int ApplyS(int qubit) const {
    auto result = lib->ApplyS(obj, qubit);
    lib->Check(result, "ApplyS");
    return result;
  }
  int ApplySDG(int qubit) const {
    auto result = lib->ApplySDG(obj, qubit);
    lib->Check(result, "ApplySDG");
    return result;
  }
  int ApplyT(int qubit) const {
    auto result = lib->ApplyT(obj, qubit);
    lib->Check(result, "ApplyT");
    return result;
  }
  int ApplyTDG(int qubit) const {
    auto result = lib->ApplyTDG(obj, qubit);
    lib->Check(result, "ApplyTDG");
    return result;
  }
  int ApplySX(int qubit) const {
    auto result = lib->ApplySX(obj, qubit);
    lib->Check(result, "ApplySX");
    return result;
  }
  int ApplySXDG(int qubit) const {
    auto result = lib->ApplySXDG(obj, qubit);
    lib->Check(result, "ApplySXDG");
    return result;
  }
  int ApplyK(int qubit) const {
    auto result = lib->ApplyK(obj, qubit);
    lib->Check(result, "ApplyK");
    return result;
  }
  int ApplyP(int qubit, double theta) const {
    auto result = lib->ApplyP(obj, qubit, theta);
    lib->Check(result, "ApplyP");
    return result;
  }
  int ApplyRx(int qubit, double theta) const {
    auto result = lib->ApplyRx(obj, qubit, theta);
    lib->Check(result, "ApplyRx");
    return result;
  }
  int ApplyRy(int qubit, double theta) const {
    auto result = lib->ApplyRy(obj, qubit, theta);
    lib->Check(result, "ApplyRy");
    return result;
  }
  int ApplyRz(int qubit, double theta) const {
    auto result = lib->ApplyRz(obj, qubit, theta);
    lib->Check(result, "ApplyRz");
    return result;
  }
  int ApplyU(int qubit, double theta, double phi, double lambda,
             double gamma) const {
    auto result = lib->ApplyU(obj, qubit, theta, phi, lambda, gamma);
    lib->Check(result, "ApplyU");
    return result;
  }
  int ApplyCX(int controlQubit, int targetQubit) const {
    auto result = lib->ApplyCX(obj, controlQubit, targetQubit);
    lib->Check(result, "ApplyCX");
    return result;
  }
  int ApplyCY(int controlQubit, int targetQubit) const {
    auto result = lib->ApplyCY(obj, controlQubit, targetQubit);
    lib->Check(result, "ApplyCY");
    return result;
  }
  int ApplyCZ(int controlQubit, int targetQubit) const {
    auto result = lib->ApplyCZ(obj, controlQubit, targetQubit);
    lib->Check(result, "ApplyCZ");
    return result;
  }
  int ApplyCH(int controlQubit, int targetQubit) const {
    auto result = lib->ApplyCH(obj, controlQubit, targetQubit);
    lib->Check(result, "ApplyCH");
    return result;
  }
  int ApplyCSX(int controlQubit, int targetQubit) const {
    auto result = lib->ApplyCSX(obj, controlQubit, targetQubit);
    lib->Check(result, "ApplyCSX");
    return result;
  }
  int ApplyCSXDG(int controlQubit, int targetQubit) const {
    auto result = lib->ApplyCSXDG(obj, controlQubit, targetQubit);
    lib->Check(result, "ApplyCSXDG");
    return result;
  }
  int ApplyCP(int controlQubit, int targetQubit, double theta) const {
    auto result = lib->ApplyCP(obj, controlQubit, targetQubit, theta);
    lib->Check(result, "ApplyCP");
    return result;
  }
  int ApplyCRx(int controlQubit, int targetQubit, double theta) const {
    auto result = lib->ApplyCRx(obj, controlQubit, targetQubit, theta);
    lib->Check(result, "ApplyCRx");
    return result;
  }
  int ApplyCRy(int controlQubit, int targetQubit, double theta) const {
    auto result = lib->ApplyCRy(obj, controlQubit, targetQubit, theta);
    lib->Check(result, "ApplyCRy");
    return result;
  }
  int ApplyCRz(int controlQubit, int targetQubit, double theta) const {
    auto result = lib->ApplyCRz(obj, controlQubit, targetQubit, theta);
    lib->Check(result, "ApplyCRz");
    return result;
  }
  int ApplyCCX(int controlQubit1, int controlQubit2, int targetQubit) const {
    auto result = lib->ApplyCCX(obj, controlQubit1, controlQubit2, targetQubit);
    lib->Check(result, "ApplyCCX");
    return result;
  }
  int ApplySwap(int qubit1, int qubit2) const {
    auto result = lib->ApplySwap(obj, qubit1, qubit2);
    lib->Check(result, "ApplySwap");
    return result;
  }
  int ApplyCSwap(int controlQubit, int qubit1, int qubit2) const {
    auto result = lib->ApplyCSwap(obj, controlQubit, qubit1, qubit2);
    lib->Check(result, "ApplyCSwap");
    return result;
  }
  int ApplyCU(int controlQubit, int targetQubit, double theta, double phi,
              double lambda, double gamma) const {
    auto result =
        lib->ApplyCU(obj, controlQubit, targetQubit, theta, phi, lambda, gamma);
    lib->Check(result, "ApplyCU");
    return result;
  }
  int ConfigureDistribution(
      const DistributedGpuApi::MgdDistributionConfig *config) const {
    auto result = lib->ConfigureDistribution(obj, config);
    lib->Check(result, "ConfigureDistribution");
    return result;
  }
  int GetShardDevices(int32_t *devices, uint32_t capacity) const {
    auto result = lib->GetShardDevices(obj, devices, capacity);
    if (result < 0) lib->Fail("GetShardDevices");
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
    return result;
  }
  int GetGlobalQubits(int32_t *qubits, uint32_t capacity) const {
    auto result = lib->GetGlobalQubits(obj, qubits, capacity);
    if (result < 0) lib->Fail("GetGlobalQubits");
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
    return result;
  }
  int GetQubitLayout(int32_t *wires, uint32_t capacity) const {
    auto result = lib->GetQubitLayout(obj, wires, capacity);
    if (result < 0) lib->Fail("GetQubitLayout");
    const char *error = lib->GetLastError();
    if (error && *error) throw std::runtime_error(error);
    return result;
  }
  int Redistribute(const int32_t *global_qubits, uint32_t count) const {
    auto result = lib->Redistribute(obj, global_qubits, count);
    lib->Check(result, "Redistribute");
    return result;
  }
  int SwapGlobalLocalQubits(const int32_t *global_qubits,
                            const int32_t *local_qubits, uint32_t count) const {
    auto result =
        lib->SwapGlobalLocalQubits(obj, global_qubits, local_qubits, count);
    lib->Check(result, "SwapGlobalLocalQubits");
    return result;
  }
  int Synchronize() const {
    auto result = lib->Synchronize(obj);
    lib->Check(result, "Synchronize");
    return result;
  }
  int GetStateRange(uint64_t begin, uint64_t end, double *output) const {
    auto result = lib->GetStateRange(obj, begin, end, output);
    lib->Check(result, "GetStateRange");
    return result;
  }
  int SetStateRange(uint64_t begin, uint64_t end, const double *input) const {
    auto result = lib->SetStateRange(obj, begin, end, input);
    lib->Check(result, "SetStateRange");
    return result;
  }
  int GetLocalStateBounds(uint64_t *begin, uint64_t *end) const {
    auto result = lib->GetLocalStateBounds(obj, begin, end);
    lib->Check(result, "GetLocalStateBounds");
    return result;
  }
  int GetLocalStateRange(uint64_t begin, uint64_t end, double *output) const {
    auto result = lib->GetLocalStateRange(obj, begin, end, output);
    lib->Check(result, "GetLocalStateRange");
    return result;
  }
  int SetLocalStateRange(uint64_t begin, uint64_t end,
                         const double *input) const {
    auto result = lib->SetLocalStateRange(obj, begin, end, input);
    lib->Check(result, "SetLocalStateRange");
    return result;
  }
  int CreateWithBasisState(uint32_t nrQubits, uint64_t basis) const {
    auto result = lib->CreateWithBasisState(obj, nrQubits, basis);
    lib->Check(result, "CreateWithBasisState");
    return result;
  }
  int ApplyOneQubitMatrix(int qubit, const double *matrix) const {
    auto result = lib->ApplyOneQubitMatrix(obj, qubit, matrix);
    lib->Check(result, "ApplyOneQubitMatrix");
    return result;
  }
  int ApplyOneQubitMatrixWithLayout(int qubit, const double *matrix,
                                    int layout) const {
    auto result =
        lib->ApplyOneQubitMatrixWithLayout(obj, qubit, matrix, layout);
    lib->Check(result, "ApplyOneQubitMatrixWithLayout");
    return result;
  }
  int ApplyTwoQubitMatrix(int qubit0, int qubit1, const double *matrix) const {
    auto result = lib->ApplyTwoQubitMatrix(obj, qubit0, qubit1, matrix);
    lib->Check(result, "ApplyTwoQubitMatrix");
    return result;
  }
  int ApplyTwoQubitMatrixWithLayout(int qubit0, int qubit1,
                                    const double *matrix, int layout) const {
    auto result =
        lib->ApplyTwoQubitMatrixWithLayout(obj, qubit0, qubit1, matrix, layout);
    lib->Check(result, "ApplyTwoQubitMatrixWithLayout");
    return result;
  }

 protected:
  std::shared_ptr<DistributedGpuLibrary> lib;
  void *obj;
};
}  // namespace Simulators
#endif
