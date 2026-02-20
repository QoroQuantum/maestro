/**
 * @file QuestLibSim.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The Quest library class.
 *
 * Allows loading the Quest library dynamically and exposes the C API functions.
 */

#pragma once

#ifndef _QUEST_LIB_SIM_H_
#define _QUEST_LIB_SIM_H_

#include "../Utils/Library.h"
#include <complex>
#include <vector>

namespace Simulators {

// use it as a singleton
class QuestLibSim : public Utils::Library {
 public:
  QuestLibSim(const QuestLibSim &) = delete;
  QuestLibSim &operator=(const QuestLibSim &) = delete;

  QuestLibSim(QuestLibSim &&) = default;
  QuestLibSim &operator=(QuestLibSim &&) = default;

  QuestLibSim() noexcept {}

  virtual ~QuestLibSim() {
    if (initialized && fFinalize) {
      fFinalize();
      initialized = false;
    }
  }

  bool Init(const char *libName) noexcept override {
    if (Utils::Library::Init(libName)) {
      fInitialize = (void (*)())GetFunction("Initialize");
      CheckFunction((void *)fInitialize, __LINE__);
      fFinalize = (void (*)())GetFunction("Finalize");
      CheckFunction((void *)fFinalize, __LINE__);

      fCreateSimulator =
          (unsigned long int (*)(int))GetFunction("CreateSimulator");
      CheckFunction((void *)fCreateSimulator, __LINE__);
      fDestroySimulator =
          (void (*)(unsigned long int))GetFunction("DestroySimulator");
      CheckFunction((void *)fDestroySimulator, __LINE__);
      fCloneSimulator =
          (unsigned long int (*)(void *))GetFunction("CloneSimulator");
      CheckFunction((void *)fCloneSimulator, __LINE__);
      fGetSimulator =
          (void *(*)(unsigned long int))GetFunction("GetSimulator");
      CheckFunction((void *)fGetSimulator, __LINE__);

      fGetNumQubits = (int (*)(void *))GetFunction("GetNumQubits");
      CheckFunction((void *)fGetNumQubits, __LINE__);
      fGetQubitProbability0 =
          (double (*)(void *, int))GetFunction("GetQubitProbability0");
      CheckFunction((void *)fGetQubitProbability0, __LINE__);
      fGetQubitProbability1 =
          (double (*)(void *, int))GetFunction("GetQubitProbability1");
      CheckFunction((void *)fGetQubitProbability1, __LINE__);
      fGetOutcomeProbability =
          (double (*)(void *, long long int))GetFunction(
              "GetOutcomeProbability");
      CheckFunction((void *)fGetOutcomeProbability, __LINE__);
      fGetExpectationValue =
          (double (*)(void *, const char *))GetFunction("GetExpectationValue");
      CheckFunction((void *)fGetExpectationValue, __LINE__);

      fMeasure = (int (*)(void *, int))GetFunction("Measure");
      CheckFunction((void *)fMeasure, __LINE__);
      fMeasureQubits =
          (long long int (*)(void *, int *, int))GetFunction("MeasureQubits");
      CheckFunction((void *)fMeasureQubits, __LINE__);

      fApplyP = (void (*)(void *, int, double))GetFunction("ApplyP");
      CheckFunction((void *)fApplyP, __LINE__);
      fApplyX = (void (*)(void *, int))GetFunction("ApplyX");
      CheckFunction((void *)fApplyX, __LINE__);
      fApplyY = (void (*)(void *, int))GetFunction("ApplyY");
      CheckFunction((void *)fApplyY, __LINE__);
      fApplyZ = (void (*)(void *, int))GetFunction("ApplyZ");
      CheckFunction((void *)fApplyZ, __LINE__);
      fApplyH = (void (*)(void *, int))GetFunction("ApplyH");
      CheckFunction((void *)fApplyH, __LINE__);
      fApplyS = (void (*)(void *, int))GetFunction("ApplyS");
      CheckFunction((void *)fApplyS, __LINE__);
      fApplyT = (void (*)(void *, int))GetFunction("ApplyT");
      CheckFunction((void *)fApplyT, __LINE__);

      fApplyRx = (void (*)(void *, int, double))GetFunction("ApplyRx");
      CheckFunction((void *)fApplyRx, __LINE__);
      fApplyRy = (void (*)(void *, int, double))GetFunction("ApplyRy");
      CheckFunction((void *)fApplyRy, __LINE__);
      fApplyRz = (void (*)(void *, int, double))GetFunction("ApplyRz");
      CheckFunction((void *)fApplyRz, __LINE__);

      fApplyCS = (void (*)(void *, int, int))GetFunction("ApplyCS");
      CheckFunction((void *)fApplyCS, __LINE__);
      fApplyCT = (void (*)(void *, int, int))GetFunction("ApplyCT");
      CheckFunction((void *)fApplyCT, __LINE__);
      fApplyCH = (void (*)(void *, int, int))GetFunction("ApplyCH");
      CheckFunction((void *)fApplyCH, __LINE__);
      fApplySwap = (void (*)(void *, int, int))GetFunction("ApplySwap");
      CheckFunction((void *)fApplySwap, __LINE__);
      fApplyCX = (void (*)(void *, int, int))GetFunction("ApplyCX");
      CheckFunction((void *)fApplyCX, __LINE__);
      fApplyCY = (void (*)(void *, int, int))GetFunction("ApplyCY");
      CheckFunction((void *)fApplyCY, __LINE__);
      fApplyCZ = (void (*)(void *, int, int))GetFunction("ApplyCZ");
      CheckFunction((void *)fApplyCZ, __LINE__);

      fApplyCRx =
          (void (*)(void *, int, int, double))GetFunction("ApplyCRx");
      CheckFunction((void *)fApplyCRx, __LINE__);
      fApplyCRy =
          (void (*)(void *, int, int, double))GetFunction("ApplyCRy");
      CheckFunction((void *)fApplyCRy, __LINE__);
      fApplyCRz =
          (void (*)(void *, int, int, double))GetFunction("ApplyCRz");
      CheckFunction((void *)fApplyCRz, __LINE__);

      fApplyCSwap =
          (void (*)(void *, int, int, int))GetFunction("ApplyCSwap");
      CheckFunction((void *)fApplyCSwap, __LINE__);
      fApplyCCX =
          (void (*)(void *, int, int, int))GetFunction("ApplyCCX");
      CheckFunction((void *)fApplyCCX, __LINE__);

      fApplySdg = (void (*)(void *, int))GetFunction("ApplySdg");
      CheckFunction((void *)fApplySdg, __LINE__);
      fApplyTdg = (void (*)(void *, int))GetFunction("ApplyTdg");
      CheckFunction((void *)fApplyTdg, __LINE__);
      fApplySx = (void (*)(void *, int))GetFunction("ApplySx");
      CheckFunction((void *)fApplySx, __LINE__);
      fApplySxDg = (void (*)(void *, int))GetFunction("ApplySxDg");
      CheckFunction((void *)fApplySxDg, __LINE__);
      fApplyK = (void (*)(void *, int))GetFunction("ApplyK");
      CheckFunction((void *)fApplyK, __LINE__);

      fApplyU = (void (*)(void *, int, double, double, double,
                          double))GetFunction("ApplyU");
      CheckFunction((void *)fApplyU, __LINE__);
      fApplyCU = (void (*)(void *, int, int, double, double, double,
                           double))GetFunction("ApplyCU");
      CheckFunction((void *)fApplyCU, __LINE__);
      fApplyCP =
          (void (*)(void *, int, int, double))GetFunction("ApplyCP");
      CheckFunction((void *)fApplyCP, __LINE__);
      fApplyCSx = (void (*)(void *, int, int))GetFunction("ApplyCSx");
      CheckFunction((void *)fApplyCSx, __LINE__);
      fApplyCSxDg = (void (*)(void *, int, int))GetFunction("ApplyCSxDg");
      CheckFunction((void *)fApplyCSxDg, __LINE__);

      fGetAmplitudes = (int (*)(void *, void *, unsigned long long int))GetFunction("GetAmplitudes");
      CheckFunction((void *)fGetAmplitudes, __LINE__);
      fGetAmplitude = (int (*)(void *, long long int, void *, unsigned long long int))GetFunction("GetAmplitude");
      CheckFunction((void *)fGetAmplitude, __LINE__);
      fIsDoublePrecision = (int (*)())GetFunction("IsDoublePrecision");
      CheckFunction((void *)fIsDoublePrecision, __LINE__);

      if (fInitialize) {
        fInitialize();
        initialized = true;
        return true;
      }
    }

    return false;
  }

  static void CheckFunction(void *func, int line) noexcept {
    if (!func) {
      std::cerr << "QuestLibSim: Unable to load function, line #: " << line;

#ifdef __linux__
      const char *dlsym_error = dlerror();
      if (dlsym_error) std::cerr << ", error: " << dlsym_error;
#elif defined(_WIN32)
      const DWORD error = GetLastError();
      std::cerr << ", error code: " << error;
#endif

      std::cerr << std::endl;
    }
  }

  bool IsValid() const { return initialized; }

  // simulator management

  unsigned long int CreateSimulator(int nrQubits) {
    if (initialized)
      return fCreateSimulator(nrQubits);
    else
      throw std::runtime_error("QuestLibSim: Unable to create simulator");

    return 0;
  }

  void DestroySimulator(unsigned long int simHandle) {
    if (initialized)
      fDestroySimulator(simHandle);
    else
      throw std::runtime_error("QuestLibSim: Unable to destroy simulator");
  }

  unsigned long int CloneSimulator(void *sim) {
    if (initialized)
      return fCloneSimulator(sim);
    else
      throw std::runtime_error("QuestLibSim: Unable to clone simulator");

    return 0;
  }

  void *GetSimulator(unsigned long int simHandle) {
    if (initialized)
      return fGetSimulator(simHandle);
    else
      throw std::runtime_error("QuestLibSim: Unable to get simulator");

    return nullptr;
  }

  // state query functions

  int GetNumQubits(void *sim) const {
    if (initialized)
      return fGetNumQubits(sim);
    else
      throw std::runtime_error("QuestLibSim: Unable to get number of qubits");

    return 0;
  }

  double GetQubitProbability0(void *sim, int qubit) const {
    if (initialized)
      return fGetQubitProbability0(sim, qubit);
    else
      throw std::runtime_error(
          "QuestLibSim: Unable to get qubit probability 0");

    return 0;
  }

  double GetQubitProbability1(void *sim, int qubit) const {
    if (initialized)
      return fGetQubitProbability1(sim, qubit);
    else
      throw std::runtime_error(
          "QuestLibSim: Unable to get qubit probability 1");

    return 0;
  }

  double GetOutcomeProbability(void *sim, long long int outcome) const {
    if (initialized)
      return fGetOutcomeProbability(sim, outcome);
    else
      throw std::runtime_error(
          "QuestLibSim: Unable to get outcome probability");

    return 0;
  }

  double GetExpectationValue(void *sim, const char *pauliStr) const {
    if (initialized)
      return fGetExpectationValue(sim, pauliStr);
    else
      throw std::runtime_error(
          "QuestLibSim: Unable to get expectation value");

    return 0;
  }

  // measurement functions

  int Measure(void *sim, int qubit) {
    if (initialized)
      return fMeasure(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to measure qubit");

    return 0;
  }

  long long int MeasureQubits(void *sim, int *qubits, int numQubits) {
    if (initialized)
      return fMeasureQubits(sim, qubits, numQubits);
    else
      throw std::runtime_error("QuestLibSim: Unable to measure qubits");

    return 0;
  }

  // single-qubit gate functions

  void ApplyP(void *sim, int qubit, double angle) {
    if (initialized)
      fApplyP(sim, qubit, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply P gate");
  }

  void ApplyX(void *sim, int qubit) {
    if (initialized)
      fApplyX(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply X gate");
  }

  void ApplyY(void *sim, int qubit) {
    if (initialized)
      fApplyY(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Y gate");
  }

  void ApplyZ(void *sim, int qubit) {
    if (initialized)
      fApplyZ(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Z gate");
  }

  void ApplyH(void *sim, int qubit) {
    if (initialized)
      fApplyH(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply H gate");
  }

  void ApplyS(void *sim, int qubit) {
    if (initialized)
      fApplyS(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply S gate");
  }

  void ApplyT(void *sim, int qubit) {
    if (initialized)
      fApplyT(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply T gate");
  }

  void ApplySdg(void *sim, int qubit) {
    if (initialized)
      fApplySdg(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Sdg gate");
  }

  void ApplyTdg(void *sim, int qubit) {
    if (initialized)
      fApplyTdg(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Tdg gate");
  }

  void ApplySx(void *sim, int qubit) {
    if (initialized)
      fApplySx(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Sx gate");
  }

  void ApplySxDg(void *sim, int qubit) {
    if (initialized)
      fApplySxDg(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply SxDg gate");
  }

  void ApplyK(void *sim, int qubit) {
    if (initialized)
      fApplyK(sim, qubit);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply K gate");
  }

  // single-qubit rotation gate functions

  void ApplyRx(void *sim, int qubit, double angle) {
    if (initialized)
      fApplyRx(sim, qubit, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Rx gate");
  }

  void ApplyRy(void *sim, int qubit, double angle) {
    if (initialized)
      fApplyRy(sim, qubit, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Ry gate");
  }

  void ApplyRz(void *sim, int qubit, double angle) {
    if (initialized)
      fApplyRz(sim, qubit, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Rz gate");
  }

  void ApplyU(void *sim, int qubit, double theta, double phi, double lambda,
              double gamma) {
    if (initialized)
      fApplyU(sim, qubit, theta, phi, lambda, gamma);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply U gate");
  }

  // two-qubit gate functions

  void ApplyCS(void *sim, int control, int target) {
    if (initialized)
      fApplyCS(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CS gate");
  }

  void ApplyCT(void *sim, int control, int target) {
    if (initialized)
      fApplyCT(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CT gate");
  }

  void ApplyCH(void *sim, int control, int target) {
    if (initialized)
      fApplyCH(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CH gate");
  }

  void ApplySwap(void *sim, int qubit1, int qubit2) {
    if (initialized)
      fApplySwap(sim, qubit1, qubit2);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply Swap gate");
  }

  void ApplyCX(void *sim, int control, int target) {
    if (initialized)
      fApplyCX(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CX gate");
  }

  void ApplyCY(void *sim, int control, int target) {
    if (initialized)
      fApplyCY(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CY gate");
  }

  void ApplyCZ(void *sim, int control, int target) {
    if (initialized)
      fApplyCZ(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CZ gate");
  }

  void ApplyCRx(void *sim, int control, int target, double angle) {
    if (initialized)
      fApplyCRx(sim, control, target, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CRx gate");
  }

  void ApplyCRy(void *sim, int control, int target, double angle) {
    if (initialized)
      fApplyCRy(sim, control, target, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CRy gate");
  }

  void ApplyCRz(void *sim, int control, int target, double angle) {
    if (initialized)
      fApplyCRz(sim, control, target, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CRz gate");
  }

  void ApplyCP(void *sim, int control, int target, double angle) {
    if (initialized)
      fApplyCP(sim, control, target, angle);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CP gate");
  }

  void ApplyCU(void *sim, int control, int target, double theta, double phi,
               double lambda, double gamma) {
    if (initialized)
      fApplyCU(sim, control, target, theta, phi, lambda, gamma);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CU gate");
  }

  void ApplyCSx(void *sim, int control, int target) {
    if (initialized)
      fApplyCSx(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CSx gate");
  }

  void ApplyCSxDg(void *sim, int control, int target) {
    if (initialized)
      fApplyCSxDg(sim, control, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CSxDg gate");
  }

  // three-qubit gate functions

  void ApplyCSwap(void *sim, int control, int qubit1, int qubit2) {
    if (initialized)
      fApplyCSwap(sim, control, qubit1, qubit2);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CSwap gate");
  }

  void ApplyCCX(void *sim, int control1, int control2, int target) {
    if (initialized)
      fApplyCCX(sim, control1, control2, target);
    else
      throw std::runtime_error("QuestLibSim: Unable to apply CCX gate");
  }

  // amplitude functions

  bool GetAmplitudes(void *sim, std::vector<std::complex<double>>& amplitudes) const {
    if (initialized) {
      if (IsDoublePrecision())
        return fGetAmplitudes(sim, amplitudes.data(), amplitudes.size() * sizeof(std::complex<double>)) == 1;
      else
      {
        std::vector<std::complex<float>> amplitudesSingle(amplitudes.size());
        if (fGetAmplitudes(sim, amplitudesSingle.data(), amplitudesSingle.size() * sizeof(std::complex<float>)) == 1) {
          for (size_t i = 0; i < amplitudes.size(); ++i)
            amplitudes[i] = std::complex<double>(static_cast<double>(amplitudesSingle[i].real()), static_cast<double>(amplitudesSingle[i].imag()));
          
          return true;
        }
      }
    }
    else
      throw std::runtime_error("QuestLibSim: Unable to get amplitudes");

    return false;
  }

  bool GetAmplitude(void *sim, long long int index, std::complex<double>& amplitude) const {
    if (initialized) {
      if (IsDoublePrecision())
        return fGetAmplitude(sim, index, &amplitude, sizeof(std::complex<double>)) == 1;
      else
      {
        std::complex<float> ampSingle;
        if (fGetAmplitude(sim, index, &ampSingle, sizeof(std::complex<float>)) == 1) {
          amplitude = std::complex<double>(static_cast<double>(ampSingle.real()),
                                          static_cast<double>(ampSingle.imag()));
          return true;
        }
      }
    }
    else
      throw std::runtime_error("QuestLibSim: Unable to get amplitude");
    return false;
  }

  bool IsDoublePrecision() const {
    if (initialized)
      return fIsDoublePrecision() == 1;
    else
      throw std::runtime_error("QuestLibSim: Unable to check double precision");
    return false;
  }

 private:
  bool initialized = false;

  void (*fInitialize)() = nullptr;
  void (*fFinalize)() = nullptr;

  unsigned long int (*fCreateSimulator)(int) = nullptr;
  void (*fDestroySimulator)(unsigned long int) = nullptr;
  unsigned long int (*fCloneSimulator)(void *) = nullptr;
  void *(*fGetSimulator)(unsigned long int) = nullptr;

  int (*fGetNumQubits)(void *) = nullptr;
  double (*fGetQubitProbability0)(void *, int) = nullptr;
  double (*fGetQubitProbability1)(void *, int) = nullptr;
  double (*fGetOutcomeProbability)(void *, long long int) = nullptr;
  double (*fGetExpectationValue)(void *, const char *) = nullptr;

  int (*fMeasure)(void *, int) = nullptr;
  long long int (*fMeasureQubits)(void *, int *, int) = nullptr;

  void (*fApplyP)(void *, int, double) = nullptr;
  void (*fApplyX)(void *, int) = nullptr;
  void (*fApplyY)(void *, int) = nullptr;
  void (*fApplyZ)(void *, int) = nullptr;
  void (*fApplyH)(void *, int) = nullptr;
  void (*fApplyS)(void *, int) = nullptr;
  void (*fApplyT)(void *, int) = nullptr;
  void (*fApplyRx)(void *, int, double) = nullptr;
  void (*fApplyRy)(void *, int, double) = nullptr;
  void (*fApplyRz)(void *, int, double) = nullptr;

  void (*fApplyCS)(void *, int, int) = nullptr;
  void (*fApplyCT)(void *, int, int) = nullptr;
  void (*fApplyCH)(void *, int, int) = nullptr;
  void (*fApplySwap)(void *, int, int) = nullptr;
  void (*fApplyCX)(void *, int, int) = nullptr;
  void (*fApplyCY)(void *, int, int) = nullptr;
  void (*fApplyCZ)(void *, int, int) = nullptr;
  void (*fApplyCRx)(void *, int, int, double) = nullptr;
  void (*fApplyCRy)(void *, int, int, double) = nullptr;
  void (*fApplyCRz)(void *, int, int, double) = nullptr;

  void (*fApplyCSwap)(void *, int, int, int) = nullptr;
  void (*fApplyCCX)(void *, int, int, int) = nullptr;

  void (*fApplySdg)(void *, int) = nullptr;
  void (*fApplyTdg)(void *, int) = nullptr;
  void (*fApplySx)(void *, int) = nullptr;
  void (*fApplySxDg)(void *, int) = nullptr;
  void (*fApplyK)(void *, int) = nullptr;

  void (*fApplyU)(void *, int, double, double, double, double) = nullptr;
  void (*fApplyCU)(void *, int, int, double, double, double, double) = nullptr;
  void (*fApplyCP)(void *, int, int, double) = nullptr;
  void (*fApplyCSx)(void *, int, int) = nullptr;
  void (*fApplyCSxDg)(void *, int, int) = nullptr;
  int (*fGetAmplitudes)(void *, void *, unsigned long long int) = nullptr;
  int (*fGetAmplitude)(void *, long long int, void *, unsigned long long int) = nullptr;
  int (*fIsDoublePrecision)() = nullptr;
};

}

#endif // _QUEST_LIB_SIM_H_