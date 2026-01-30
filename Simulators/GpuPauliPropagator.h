/**
 * @file GpuPauliPropagator.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The Gpu Pauli Propagator library class.
 *
 * Just a wrapper around c api functions.
 */

#pragma once

#ifndef _GPU_PAULI_PROPAGATOR_H
#define _GPU_PAULI_PROPAGATOR_H 1

#ifdef __linux__

#include "GpuLibrary.h"

namespace Simulators {

class GpuPauliPropagator {
 public:
  explicit GpuPauliPropagator(const std::shared_ptr<GpuLibrary> &lib)
      : lib(lib), obj(nullptr) {}

  GpuPauliPropagator() = delete;
  GpuPauliPropagator(const GpuPauliPropagator &) = delete;
  GpuPauliPropagator &operator=(const GpuPauliPropagator &) = delete;
  GpuPauliPropagator(GpuPauliPropagator &&) = default;
  GpuPauliPropagator &operator=(GpuPauliPropagator &&) = default;

  ~GpuPauliPropagator() {
    if (lib && obj) lib->DestroyPauliPropSimulator(obj);
  }

  bool CreateSimulator(int numQubits) {
    if (lib) {
      obj = lib->CreatePauliPropSimulator(numQubits);
      
      return obj != nullptr;
    }
    return false;
  }

  int GetNrQubits() {
    if (lib) {
      return lib->PauliPropGetNrQubits(obj);
    }
    return 0;
  }

  double GetCoefficientTruncationCutoff() {
    if (lib) {
      return lib->PauliPropGetCoefficientTruncationCutoff(obj);
    }
    return 0.0;
  }

  void SetCoefficientTruncationCutoff(double cutoff) {
    if (lib) {
      lib->PauliPropSetCoefficientTruncationCutoff(obj, cutoff);
    }
  }

  double GetWeightTruncationCutoff() {
    if (lib) {
      return lib->PauliPropGetWeightTruncationCutoff(obj);
    }
    return 0.0;
  }

  void SetWeightTruncationCutoff(double cutoff) {
    if (lib) {
      lib->PauliPropSetWeightTruncationCutoff(obj, cutoff);
    }
  }

  int GetNumGatesBetweenTruncations()
  {
    if (lib) {
      return lib->PauliPropGetNumGatesBetweenTruncations(obj);
    }
    return 0;
  }

  void SetNumGatesBetweenTruncations(int numGates)
  {
    if (lib) {
      lib->PauliPropSetNumGatesBetweenTruncations(obj, numGates);
    }
  }

  bool ClearOperators()
  {
    if (lib) {
      return lib->PauliPropClearOperators(obj);
    }
    return false;
  }

  bool AllocateMemory(double percentage)
  {
    if (lib) {
      return lib->PauliPropAllocateMemory(obj, percentage);
    }
    return false;
  }

  double GetExpectationValue()
  {
    if (lib) {
      return lib->PauliPropGetExpectationValue(obj);
    }
    return 0.0;
  }

  bool Execute()
  {
    if (lib) {
      return lib->PauliPropExecute(obj);
    }
    return false;
  }

  bool SetInPauliExpansionUnique(const std::string& pauliStr)
  {
    if (lib) {
      return lib->PauliPropSetInPauliExpansionUnique(obj, pauliStr.c_str());
    }
    return false;
  }

  bool SetInPauliExpansionMultiple(const std::vector<std::string> &pauliStrs, const std::vector<double> &coefficients)
  {
    if (lib && !pauliStrs.empty() && pauliStrs.size() == coefficients.size()) {
      std::vector<char *> cStrs(pauliStrs.size());
      for (size_t i = 0; i < pauliStrs.size(); ++i) {
        cStrs[i] = const_cast<char *>(pauliStrs[i].c_str());
      }
      return lib->PauliPropSetInPauliExpansionMultiple(
          obj, (const char **)cStrs.data(), coefficients.data(), static_cast<int>(pauliStrs.size()));
    }
    return false;
  }

  bool ApplyX(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplyX(obj, qubit);
    }
    return false;
  }

  bool ApplyY(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplyY(obj, qubit);
    }
    return false;
  }

  bool ApplyZ(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplyZ(obj, qubit);
    }
    return false;
  }

  bool ApplyH(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplyH(obj, qubit);
    }
    return false;
  }

  bool ApplyS(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplyS(obj, qubit);
    }
    return false;
  }

  bool ApplySQRTX(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplySQRTX(obj, qubit);
    }
    return false;
  }

  bool ApplySQRTY(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplySQRTY(obj, qubit);
    }
    return false;
  }

  bool ApplySQRTZ(int qubit)
  {
    if (lib) {
      return lib->PauliPropApplySQRTZ(obj, qubit);
    }
    return false;
  }

  bool ApplyCX(int controlQubit, int targetQubit)
  {
    if (lib) {
      return lib->PauliPropApplyCX(obj, controlQubit, targetQubit);
    }
    return false;
  }

  bool ApplyCY(int controlQubit, int targetQubit)
  {
    if (lib) {
      return lib->PauliPropApplyCY(obj, controlQubit, targetQubit);
    }
    return false;
  }

  bool ApplyCZ(int controlQubit, int targetQubit)
  {
    if (lib) {
      return lib->PauliPropApplyCZ(obj, controlQubit, targetQubit);
    }
    return false;
  }

  bool ApplySWAP(int qubit1, int qubit2)
  {
    if (lib) {
      return lib->PauliPropApplySWAP(obj, qubit1, qubit2);
    }
    return false;
  }

  bool ApplyISWAP(int qubit1, int qubit2) {
    if (lib) {
      return lib->PauliPropApplyISWAP(obj, qubit1, qubit2);
    }
    return false;
  }

  bool ApplyRX(int qubit, double angle)
  {
    if (lib) {
      return lib->PauliPropApplyRX(obj, qubit, angle);
    }
    return false;
  }

  bool ApplyRY(int qubit, double angle)
  {
    if (lib) {
      return lib->PauliPropApplyRY(obj, qubit, angle);
    }
    return false;
  }

  bool ApplyRZ(int qubit, double angle)
  {
    if (lib) {
      return lib->PauliPropApplyRZ(obj, qubit, angle);
    }
    return false;
  }

  bool AddNoiseX(int qubit, double probability)
  {
    if (lib) {
      return lib->PauliPropAddNoiseX(obj, qubit, probability);
    }
    return false;
  }

  bool AddNoiseY(int qubit, double probability)
  {
    if (lib) {
      return lib->PauliPropAddNoiseY(obj, qubit, probability);
    }
    return false;
  }

  bool AddNoiseZ(int qubit, double probability)
  {
    if (lib) {
      return lib->PauliPropAddNoiseZ(obj, qubit, probability);
    }
    return false;
  }

  bool AddNoiseXYZ(int qubit, double px, double py, double pz)
  {
    if (lib) {
      return lib->PauliPropAddNoiseXYZ(obj, qubit, px, py, pz);
    }
    return false;
  }

  bool AddAmplitudeDamping(int qubit, double dampingProb, double exciteProb)
  {
    if (lib) {
      return lib->PauliPropAddAmplitudeDamping(
          obj, qubit, dampingProb, exciteProb);
    }
    return false;
  }

  double QubitProbability0(int qubit)
  {
    if (lib) {
      return lib->PauliPropQubitProbability0(obj, qubit);
    }
    return 0.0;
  }

  bool MeasureQubit(int qubit)
  {
    if (lib) {
      return lib->PauliPropMeasureQubit(obj, qubit);
    }
    return false;
  }

  std::vector<bool> SampleQubits(const std::vector<int>& qubits)
  {
    std::vector<bool> results;
    if (lib && !qubits.empty()) {
      std::vector<int> cQubits = qubits;
      unsigned char *res = lib->PauliPropSampleQubits(
          obj, qubits.data(), static_cast<int>(cQubits.size()));
      if (!res)
        return results;
      
      results.reserve(cQubits.size());
      for (size_t i = 0; i < cQubits.size(); ++i) 
      {
          // results are encoded as bits
          const bool bit = ((res[i / 8] >> (i % 8)) & 1) == 1;
          results.push_back(bit);
      }

      lib->PauliPropFreeSampledQubits(res);
    }
    return results;
  }

  void SaveState()
  {
    if (lib) {
      lib->PauliPropSaveState(obj);
    } 
  }

  void RestoreState()
  {
    if (lib) {
      lib->PauliPropRestoreState(obj);
    } 
  }

 private:
  std::shared_ptr<GpuLibrary> lib;
  void *obj;
};

}  // namespace Simulators

#endif
#endif
