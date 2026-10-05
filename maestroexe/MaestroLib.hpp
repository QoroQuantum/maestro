#pragma once

#include "../Utils/Library.h"
#include "../maestrolib/InterfaceTypes.h"
#include <complex>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <vector>

class MaestroLibrary : public Utils::Library {
 public:
  MaestroLibrary(const MaestroLibrary&) = delete;
  MaestroLibrary& operator=(const MaestroLibrary&) = delete;

  MaestroLibrary(MaestroLibrary&&) = default;
  MaestroLibrary& operator=(MaestroLibrary&&) = default;

  MaestroLibrary() noexcept {}

  virtual ~MaestroLibrary() {}

  bool Init(const char* libName) noexcept override {
    if (Utils::Library::Init(libName)) {
      fGetMaestroObject = (void* (*)())GetFunction("GetMaestroObjectWithMute");
      CheckFunction((void*)fGetMaestroObject, __LINE__);
      if (fGetMaestroObject) {
        maestro = fGetMaestroObject();
        if (maestro) {
          fCreateSimpleSimulator =
              (unsigned long int (*)(int))GetFunction("CreateSimpleSimulator");
          CheckFunction((void*)fCreateSimpleSimulator, __LINE__);
          fDestroySimpleSimulator = (void (*)(unsigned long int))GetFunction(
              "DestroySimpleSimulator");
          CheckFunction((void*)fDestroySimpleSimulator, __LINE__);

          fRemoveAllOptimizationSimulatorsAndAdd = (int (*)(
              unsigned long int simHandle, int simType, int simExecType))
              GetFunction("RemoveAllOptimizationSimulatorsAndAdd");
          CheckFunction((void*)fRemoveAllOptimizationSimulatorsAndAdd,
                        __LINE__);
          fAddOptimizationSimulator =
              (int (*)(unsigned long int simHandle, int simType,
                       int simExecType))GetFunction("AddOptimizationSimulator");
          CheckFunction((void*)fAddOptimizationSimulator, __LINE__);

          fSimpleExecute = (char* (*)(unsigned long int, const char*,
                                      const char*))GetFunction("SimpleExecute");
          CheckFunction((void*)fSimpleExecute, __LINE__);

          fSimpleEstimate =
              (char* (*)(unsigned long int, const char*, const char*,
                         const char*))GetFunction("SimpleEstimate");
          CheckFunction((void*)fSimpleEstimate, __LINE__);

          fFreeResult = (void (*)(char*))GetFunction("FreeResult");
          CheckFunction((void*)fFreeResult, __LINE__);

          fCreateSimulator =
              (unsigned long int (*)(int, int))GetFunction("CreateSimulator");
          CheckFunction((void*)fCreateSimulator, __LINE__);
          fGetSimulator =
              (void* (*)(unsigned long int))GetFunction("GetSimulator");
          CheckFunction((void*)fGetSimulator, __LINE__);
          // Optional exports keep loading older libraries compatible.
          fExpectationValues = reinterpret_cast<decltype(fExpectationValues)>(
              GetFunction("MaestroExpectationValues"));
          fExpectationValuesComplex =
              reinterpret_cast<decltype(fExpectationValuesComplex)>(
                  GetFunction("MaestroExpectationValuesComplex"));
          fGetStateVector = reinterpret_cast<decltype(fGetStateVector)>(
              GetFunction("MaestroGetStateVector"));
          fExpectationValueOperators =
              reinterpret_cast<decltype(fExpectationValueOperators)>(
                  GetFunction("MaestroExpectationValueOperators"));
          fMoveAtBeginningOfChain =
              reinterpret_cast<decltype(fMoveAtBeginningOfChain)>(
                  GetFunction("MaestroMoveAtBeginningOfChain"));
          fRunRequestJson = reinterpret_cast<decltype(fRunRequestJson)>(
              GetFunction("MaestroRunRequestJson"));
          fValidateRequestJson =
              reinterpret_cast<decltype(fValidateRequestJson)>(
                  GetFunction("MaestroValidateRequestJson"));
          fDestroySimulator =
              (void (*)(unsigned long int))GetFunction("DestroySimulator");
          CheckFunction((void*)fDestroySimulator, __LINE__);

          fInitializeSimulator =
              (int (*)(void*))GetFunction("InitializeSimulator");
          CheckFunction((void*)fInitializeSimulator, __LINE__);
          fResetSimulator = (int (*)(void*))GetFunction("ResetSimulator");
          CheckFunction((void*)fResetSimulator, __LINE__);
          fConfigureSimulator =
              (int (*)(void*, const char*, const char*))GetFunction(
                  "ConfigureSimulator");
          CheckFunction((void*)fConfigureSimulator, __LINE__);
          fGetConfiguration =
              (char* (*)(void*, const char*))GetFunction("GetConfiguration");
          CheckFunction((void*)fGetConfiguration, __LINE__);
          fAllocateQubits = (unsigned long int (*)(
              void*, unsigned long int))GetFunction("AllocateQubits");
          CheckFunction((void*)fAllocateQubits, __LINE__);
          fGetNumberOfQubits =
              (unsigned long int (*)(void*))GetFunction("GetNumberOfQubits");
          CheckFunction((void*)fGetNumberOfQubits, __LINE__);
          fClearSimulator = (int (*)(void*))GetFunction("ClearSimulator");
          CheckFunction((void*)fClearSimulator, __LINE__);
          fMeasure = (unsigned long long int (*)(
              void*, const unsigned long int*,
              unsigned long int))GetFunction("Measure");
          CheckFunction((void*)fMeasure, __LINE__);
          fApplyReset = (int (*)(void*, const unsigned long int*,
                                 unsigned long int))GetFunction("ApplyReset");
          CheckFunction((void*)fApplyReset, __LINE__);
          fProbability = (double (*)(void*, unsigned long long int))GetFunction(
              "Probability");
          CheckFunction((void*)fProbability, __LINE__);
          fFreeDoubleVector =
              (void (*)(double*))GetFunction("FreeDoubleVector");
          CheckFunction((void*)fFreeDoubleVector, __LINE__);
          fFreeULLIVector =
              (void (*)(unsigned long long int*))GetFunction("FreeULLIVector");
          CheckFunction((void*)fFreeULLIVector, __LINE__);
          fAmplitude = (double* (*)(void*, unsigned long long int))GetFunction(
              "Amplitude");
          CheckFunction((void*)fAmplitude, __LINE__);
          fAllProbabilities =
              (double* (*)(void*))GetFunction("AllProbabilities");
          CheckFunction((void*)fAllProbabilities, __LINE__);
          fProbabilities =
              (double* (*)(void*, const unsigned long long int*,
                           unsigned long int))GetFunction("Probabilities");
          CheckFunction((void*)fProbabilities, __LINE__);
          fSampleCounts =
              (unsigned long long int* (*)(void*, const unsigned long long int*,
                                           unsigned long int,
                                           unsigned long int))
                  GetFunction("SampleCounts");
          CheckFunction((void*)fSampleCounts, __LINE__);
          fGetSimulatorType = (int (*)(void*))GetFunction("GetSimulatorType");
          CheckFunction((void*)fGetSimulatorType, __LINE__);
          fGetSimulationType = (int (*)(void*))GetFunction("GetSimulationType");
          CheckFunction((void*)fGetSimulationType, __LINE__);
          // Optional exports preserve loading compatibility with older
          // libraries.
          fGetGateFusionMaxQubits =
              reinterpret_cast<decltype(fGetGateFusionMaxQubits)>(
                  GetFunction("GetGateFusionMaxQubits"));
          fIsGateFusionEnabled =
              reinterpret_cast<decltype(fIsGateFusionEnabled)>(
                  GetFunction("IsGateFusionEnabled"));
          fGetGateFusionStatistics =
              reinterpret_cast<decltype(fGetGateFusionStatistics)>(
                  GetFunction("GetGateFusionStatistics"));
          fApplyGenericOneQubitGate =
              reinterpret_cast<decltype(fApplyGenericOneQubitGate)>(
                  GetFunction("ApplyGenericOneQubitGate"));
          fApplyGenericTwoQubitGate =
              reinterpret_cast<decltype(fApplyGenericTwoQubitGate)>(
                  GetFunction("ApplyGenericTwoQubitGate"));
          fApplyGenericThreeQubitGate =
              reinterpret_cast<decltype(fApplyGenericThreeQubitGate)>(
                  GetFunction("ApplyGenericThreeQubitGate"));
          fFlushSimulator = (int (*)(void*))GetFunction("FlushSimulator");
          CheckFunction((void*)fFlushSimulator, __LINE__);
          fSaveStateToInternalDestructive =
              (int (*)(void*))GetFunction("SaveStateToInternalDestructive");
          CheckFunction((void*)fSaveStateToInternalDestructive, __LINE__);
          fRestoreInternalDestructiveSavedState = (int (*)(void*))GetFunction(
              "RestoreInternalDestructiveSavedState");
          CheckFunction((void*)fRestoreInternalDestructiveSavedState, __LINE__);
          fSaveState = (int (*)(void*))GetFunction("SaveState");
          CheckFunction((void*)fSaveState, __LINE__);
          fRestoreState = (int (*)(void*))GetFunction("RestoreState");
          CheckFunction((void*)fRestoreState, __LINE__);
          fSetMultithreading =
              (int (*)(void*, int))GetFunction("SetMultithreading");
          CheckFunction((void*)fSetMultithreading, __LINE__);
          fGetMultithreading = (int (*)(void*))GetFunction("GetMultithreading");
          CheckFunction((void*)fGetMultithreading, __LINE__);
          fIsQcsim = (int (*)(void*))GetFunction("IsQcsim");
          CheckFunction((void*)fIsQcsim, __LINE__);
          fMeasureNoCollapse = (unsigned long long int (*)(void*))GetFunction(
              "MeasureNoCollapse");
          CheckFunction((void*)fMeasureNoCollapse, __LINE__);

          fApplyX = (int (*)(void*, int))GetFunction("ApplyX");
          CheckFunction((void*)fApplyX, __LINE__);
          fApplyY = (int (*)(void*, int))GetFunction("ApplyY");
          CheckFunction((void*)fApplyY, __LINE__);
          fApplyZ = (int (*)(void*, int))GetFunction("ApplyZ");
          CheckFunction((void*)fApplyZ, __LINE__);
          fApplyH = (int (*)(void*, int))GetFunction("ApplyH");
          CheckFunction((void*)fApplyH, __LINE__);
          fApplyS = (int (*)(void*, int))GetFunction("ApplyS");
          CheckFunction((void*)fApplyS, __LINE__);
          fApplySDG = (int (*)(void*, int))GetFunction("ApplySDG");
          CheckFunction((void*)fApplySDG, __LINE__);
          fApplyT = (int (*)(void*, int))GetFunction("ApplyT");
          CheckFunction((void*)fApplyT, __LINE__);
          fApplyTDG = (int (*)(void*, int))GetFunction("ApplyTDG");
          CheckFunction((void*)fApplyTDG, __LINE__);
          fApplySX = (int (*)(void*, int))GetFunction("ApplySX");
          CheckFunction((void*)fApplySX, __LINE__);
          fApplySXDG = (int (*)(void*, int))GetFunction("ApplySXDG");
          CheckFunction((void*)fApplySXDG, __LINE__);
          fApplyK = (int (*)(void*, int))GetFunction("ApplyK");
          CheckFunction((void*)fApplyK, __LINE__);
          fApplyP = (int (*)(void*, int, double))GetFunction("ApplyP");
          CheckFunction((void*)fApplyP, __LINE__);
          fApplyRx = (int (*)(void*, int, double))GetFunction("ApplyRx");
          CheckFunction((void*)fApplyRx, __LINE__);
          fApplyRy = (int (*)(void*, int, double))GetFunction("ApplyRy");
          CheckFunction((void*)fApplyRy, __LINE__);
          fApplyRz = (int (*)(void*, int, double))GetFunction("ApplyRz");
          CheckFunction((void*)fApplyRz, __LINE__);
          fApplyU = (int (*)(void*, int, double, double, double,
                             double))GetFunction("ApplyU");
          CheckFunction((void*)fApplyU, __LINE__);
          fApplyCX = (int (*)(void*, int, int))GetFunction("ApplyCX");
          CheckFunction((void*)fApplyCX, __LINE__);
          fApplyCY = (int (*)(void*, int, int))GetFunction("ApplyCY");
          CheckFunction((void*)fApplyCY, __LINE__);
          fApplyCZ = (int (*)(void*, int, int))GetFunction("ApplyCZ");
          CheckFunction((void*)fApplyCZ, __LINE__);
          fApplyCH = (int (*)(void*, int, int))GetFunction("ApplyCH");
          CheckFunction((void*)fApplyCH, __LINE__);
          fApplyCSX = (int (*)(void*, int, int))GetFunction("ApplyCSX");
          CheckFunction((void*)fApplyCSX, __LINE__);
          fApplyCSXDG = (int (*)(void*, int, int))GetFunction("ApplyCSXDG");
          CheckFunction((void*)fApplyCSXDG, __LINE__);
          fApplyCP = (int (*)(void*, int, int, double))GetFunction("ApplyCP");
          CheckFunction((void*)fApplyCP, __LINE__);
          fApplyCRx = (int (*)(void*, int, int, double))GetFunction("ApplyCRx");
          CheckFunction((void*)fApplyCRx, __LINE__);
          fApplyCRy = (int (*)(void*, int, int, double))GetFunction("ApplyCRy");
          CheckFunction((void*)fApplyCRy, __LINE__);
          fApplyCRz = (int (*)(void*, int, int, double))GetFunction("ApplyCRz");
          CheckFunction((void*)fApplyCRz, __LINE__);
          fApplyCCX = (int (*)(void*, int, int, int))GetFunction("ApplyCCX");
          CheckFunction((void*)fApplyCCX, __LINE__);
          fApplySwap = (int (*)(void*, int, int))GetFunction("ApplySwap");
          CheckFunction((void*)fApplySwap, __LINE__);
          fApplyCSwap =
              (int (*)(void*, int, int, int))GetFunction("ApplyCSwap");
          CheckFunction((void*)fApplyCSwap, __LINE__);
          fApplyCU = (int (*)(void*, int, int, double, double, double,
                              double))GetFunction("ApplyCU");
          CheckFunction((void*)fApplyCU, __LINE__);

          return true;
        }
      } else
        std::cerr << "MaestroLibrary: Unable to get initialization function "
                     "for library"
                  << std::endl;
    } else
      std::cerr << "MaestroLibrary: Unable to load the library" << std::endl;

    return false;
  }

  static void CheckFunction(void* func, int line) noexcept {
    if (!func) {
      std::cerr << "MaestroLibrary: Unable to load function, line #: " << line;

#ifdef __linux__
      const char* dlsym_error = dlerror();
      if (dlsym_error) std::cerr << ", error: " << dlsym_error;
#elif defined(_WIN32)
      const DWORD error = GetLastError();
      std::cerr << ", error code: " << error;
#endif

      std::cerr << std::endl;
    }
  }

  bool IsValid() const { return maestro != nullptr; }

  virtual unsigned long int CreateSimpleSimulator(int nrQubits) {
    if (maestro && fCreateSimpleSimulator)
      return fCreateSimpleSimulator(nrQubits);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to create the simple simulator.");

    return 0;
  }

  void DestroySimpleSimulator(unsigned long int simHandle) {
    if (maestro && fDestroySimpleSimulator)
      fDestroySimpleSimulator(simHandle);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to destroy the simple simulator.");
  }

  int RemoveAllOptimizationSimulatorsAndAdd(unsigned long int simHandle,
                                            int simType, int simExecType) {
    if (maestro && fRemoveAllOptimizationSimulatorsAndAdd)
      return fRemoveAllOptimizationSimulatorsAndAdd(simHandle, simType,
                                                    simExecType);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to remove all optimization simulators and "
          "add a new one.");

    return 0;
  }

  int AddOptimizationSimulator(unsigned long int simHandle, int simType,
                               int simExecType) {
    if (maestro && fAddOptimizationSimulator)
      return fAddOptimizationSimulator(simHandle, simType, simExecType);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to add an optimization simulator.");

    return 0;
  }

  char* SimpleExecute(unsigned long int simpleSim, const char* jsonCircuit,
                      const char* jsonConfig) {
    if (maestro && fSimpleExecute)
      return fSimpleExecute(simpleSim, jsonCircuit, jsonConfig);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to execute the simple simulator.");

    return nullptr;
  }

  char* SimpleEstimate(unsigned long int simpleSim, const char* jsonCircuit,
                       const char* observableStr, const char* jsonConfig) {
    if (maestro && fSimpleEstimate)
      return fSimpleEstimate(simpleSim, jsonCircuit, observableStr, jsonConfig);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to execute the simple simulator for "
          "expectation values.");

    return nullptr;
  }

  virtual void FreeResult(char* result) {
    if (maestro && fFreeResult)
      fFreeResult(result);
    else
      throw std::runtime_error("MaestroLibrary: Unable to free the result.");
  }

  virtual unsigned long int CreateSimulator(int simType, int simExecType) {
    if (maestro && fCreateSimulator)
      return fCreateSimulator(simType, simExecType);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to create the simulator.");

    return 0;
  }

  void* GetSimulator(unsigned long int simHandle) {
    if (maestro && fGetSimulator)
      return fGetSimulator(simHandle);
    else
      throw std::runtime_error("MaestroLibrary: Unable to get the simulator.");
  }

  void DestroySimulator(unsigned long int simHandle) {
    if (maestro && fDestroySimulator)
      fDestroySimulator(simHandle);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to destroy the simulator.");
  }

  int InitializeSimulator(void* sim) {
    if (maestro && sim && fInitializeSimulator)
      return fInitializeSimulator(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to initialize the simulator.");

    return 0;
  }

  int ResetSimulator(void* sim) {
    if (maestro && sim && fResetSimulator)
      return fResetSimulator(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to reset the simulator.");

    return 0;
  }

  int ConfigureSimulator(void* sim, const char* key, const char* value) {
    if (maestro && sim && fConfigureSimulator)
      return fConfigureSimulator(sim, key, value);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to configure the simulator.");

    return 0;
  }

  char* GetConfiguration(void* sim, const char* key) {
    if (maestro && sim && fGetConfiguration)
      return fGetConfiguration(sim, key);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get the configuration of the simulator.");
    return nullptr;
  }

  unsigned long int AllocateQubits(void* sim, unsigned long int nrQubits) {
    if (maestro && sim && fAllocateQubits)
      return fAllocateQubits(sim, nrQubits);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to allocate qubits in the simulator.");
    return 0;
  }

  unsigned long int GetNumberOfQubits(void* sim) {
    if (maestro && sim && fGetNumberOfQubits)
      return fGetNumberOfQubits(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get the number of qubits in the "
          "simulator.");
    return 0;
  }

  int ClearSimulator(void* sim) {
    if (maestro && sim && fClearSimulator)
      return fClearSimulator(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to clear the simulator.");
    return 0;
  }

  unsigned long long int Measure(void* sim, const unsigned long int* qubits,
                                 unsigned long int nrQubits) {
    if (maestro && sim && fMeasure)
      return fMeasure(sim, qubits, nrQubits);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to measure the simulator.");
    return 0;
  }

  int ApplyReset(void* sim, const unsigned long int* qubits,
                 unsigned long int nrQubits) {
    if (maestro && sim && fApplyReset)
      return fApplyReset(sim, qubits, nrQubits);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to apply reset to the simulator.");
    return 0;
  }

  double Probability(void* sim, unsigned long long int outcome) {
    if (maestro && sim && fProbability)
      return fProbability(sim, outcome);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get the probability of an outcome.");
    return 0.0;
  }

  virtual void FreeDoubleVector(double* vec) {
    if (maestro && fFreeDoubleVector)
      fFreeDoubleVector(vec);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to free the double vector.");
  }

  virtual void FreeULLIVector(unsigned long long int* vec) {
    if (maestro && fFreeULLIVector)
      fFreeULLIVector(vec);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to free the unsigned long long int vector.");
  }

  double* Amplitude(void* sim, unsigned long long int outcome) {
    if (maestro && sim && fAmplitude)
      return fAmplitude(sim, outcome);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get the amplitude of an outcome.");
    return nullptr;
  }

  double* AllProbabilities(void* sim) {
    if (maestro && sim && fAllProbabilities)
      return fAllProbabilities(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get all probabilities.");
    return nullptr;
  }

  double* Probabilities(void* sim, const unsigned long long int* qubits,
                        unsigned long int nrQubits) {
    if (maestro && sim && fProbabilities)
      return fProbabilities(sim, qubits, nrQubits);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get probabilities for specified qubits.");
    return nullptr;
  }

  unsigned long long int* SampleCounts(void* sim,
                                       const unsigned long long int* qubits,
                                       unsigned long int nrQubits,
                                       unsigned long int shots) {
    if (maestro && sim && fSampleCounts)
      return fSampleCounts(sim, qubits, nrQubits, shots);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get sample counts for specified qubits.");
    return nullptr;
  }

  int GetSimulatorType(void* sim) {
    if (maestro && sim && fGetSimulatorType)
      return fGetSimulatorType(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get the simulator type.");
    return -1;
  }

  int GetSimulationType(void* sim) {
    if (maestro && sim && fGetSimulationType)
      return fGetSimulationType(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get the simulation type.");
    return -1;
  }

  unsigned GetGateFusionMaxQubits(void* sim) {
    return maestro && sim && fGetGateFusionMaxQubits
               ? fGetGateFusionMaxQubits(sim)
               : 0;
  }
  int IsGateFusionEnabled(void* sim) {
    return maestro && sim && fIsGateFusionEnabled ? fIsGateFusionEnabled(sim)
                                                  : 0;
  }
  int GetGateFusionStatistics(void* sim,
                              MaestroGateFusionStatistics* statistics) {
    return maestro && sim && fGetGateFusionStatistics
               ? fGetGateFusionStatistics(sim, statistics)
               : 0;
  }
  int ApplyGenericOneQubitGate(void* sim, unsigned long q0,
                               const double* matrix) {
    return maestro && sim && fApplyGenericOneQubitGate
               ? fApplyGenericOneQubitGate(sim, q0, matrix)
               : 0;
  }
  int ApplyGenericTwoQubitGate(void* sim, unsigned long q0, unsigned long q1,
                               const double* matrix) {
    return maestro && sim && fApplyGenericTwoQubitGate
               ? fApplyGenericTwoQubitGate(sim, q0, q1, matrix)
               : 0;
  }
  int ApplyGenericThreeQubitGate(void* sim, unsigned long q0, unsigned long q1,
                                 unsigned long q2, const double* matrix) {
    return maestro && sim && fApplyGenericThreeQubitGate
               ? fApplyGenericThreeQubitGate(sim, q0, q1, q2, matrix)
               : 0;
  }

  int FlushSimulator(void* sim) {
    if (maestro && sim && fFlushSimulator)
      return fFlushSimulator(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to flush the simulator.");
    return 0;
  }

  int SaveStateToInternalDestructive(void* sim) {
    if (maestro && sim && fSaveStateToInternalDestructive)
      return fSaveStateToInternalDestructive(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to save the state to internal destructive "
          "storage.");
    return 0;
  }

  int RestoreInternalDestructiveSavedState(void* sim) {
    if (maestro && sim && fRestoreInternalDestructiveSavedState)
      return fRestoreInternalDestructiveSavedState(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to restore the state from internal "
          "destructive storage.");
    return 0;
  }

  int SaveState(void* sim) {
    if (maestro && sim && fSaveState)
      return fSaveState(sim);
    else
      throw std::runtime_error("MaestroLibrary: Unable to save the state.");
    return 0;
  }

  int RestoreState(void* sim) {
    if (maestro && sim && fRestoreState)
      return fRestoreState(sim);
    else
      throw std::runtime_error("MaestroLibrary: Unable to restore the state.");
    return 0;
  }

  int SetMultithreading(void* sim, int multithreading) {
    if (maestro && sim && fSetMultithreading)
      return fSetMultithreading(sim, multithreading);
    else
      throw std::runtime_error("MaestroLibrary: Unable to set multithreading.");
    return 0;
  }

  int GetMultithreading(void* sim) {
    if (maestro && sim && fGetMultithreading)
      return fGetMultithreading(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to get multithreading status.");
    return 0;
  }

  int IsQcsim(void* sim) {
    if (maestro && sim && fIsQcsim)
      return fIsQcsim(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to check if the simulator is a QCSIM.");
    return 0;
  }

  unsigned long long int MeasureNoCollapse(void* sim) {
    if (maestro && sim && fMeasureNoCollapse)
      return fMeasureNoCollapse(sim);
    else
      throw std::runtime_error(
          "MaestroLibrary: Unable to measure without collapse.");
    return 0;
  }

  int ApplyX(void* sim, int qubit) {
    if (maestro && sim && fApplyX)
      return fApplyX(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply X gate.");
    return 0;
  }

  int ApplyY(void* sim, int qubit) {
    if (maestro && sim && fApplyY)
      return fApplyY(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply Y gate.");
    return 0;
  }

  int ApplyZ(void* sim, int qubit) {
    if (maestro && sim && fApplyZ)
      return fApplyZ(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply Z gate.");
    return 0;
  }

  int ApplyH(void* sim, int qubit) {
    if (maestro && sim && fApplyH)
      return fApplyH(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply H gate.");
    return 0;
  }

  int ApplyS(void* sim, int qubit) {
    if (maestro && sim && fApplyS)
      return fApplyS(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply S gate.");
    return 0;
  }

  int ApplySDG(void* sim, int qubit) {
    if (maestro && sim && fApplySDG)
      return fApplySDG(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply SDG gate.");
    return 0;
  }

  int ApplyT(void* sim, int qubit) {
    if (maestro && sim && fApplyT)
      return fApplyT(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply T gate.");
    return 0;
  }

  int ApplyTDG(void* sim, int qubit) {
    if (maestro && sim && fApplyTDG)
      return fApplyTDG(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply TDG gate.");
    return 0;
  }

  int ApplySX(void* sim, int qubit) {
    if (maestro && sim && fApplySX)
      return fApplySX(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply SX gate.");
    return 0;
  }

  int ApplySXDG(void* sim, int qubit) {
    if (maestro && sim && fApplySXDG)
      return fApplySXDG(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply SXDG gate.");
    return 0;
  }

  int ApplyK(void* sim, int qubit) {
    if (maestro && sim && fApplyK)
      return fApplyK(sim, qubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply K gate.");
    return 0;
  }

  int ApplyP(void* sim, int qubit, double theta) {
    if (maestro && sim && fApplyP)
      return fApplyP(sim, qubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply P gate.");
    return 0;
  }

  int ApplyRx(void* sim, int qubit, double theta) {
    if (maestro && sim && fApplyRx)
      return fApplyRx(sim, qubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply Rx gate.");
    return 0;
  }

  int ApplyRy(void* sim, int qubit, double theta) {
    if (maestro && sim && fApplyRy)
      return fApplyRy(sim, qubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply Ry gate.");
    return 0;
  }

  int ApplyRz(void* sim, int qubit, double theta) {
    if (maestro && sim && fApplyRz)
      return fApplyRz(sim, qubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply Rz gate.");
    return 0;
  }

  int ApplyU(void* sim, int qubit, double theta, double phi, double lambda,
             double gamma) {
    if (maestro && sim && fApplyU)
      return fApplyU(sim, qubit, theta, phi, lambda, gamma);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply U gate.");
    return 0;
  }

  int ApplyCX(void* sim, int controlQubit, int targetQubit) {
    if (maestro && sim && fApplyCX)
      return fApplyCX(sim, controlQubit, targetQubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CX gate.");
    return 0;
  }

  int ApplyCY(void* sim, int controlQubit, int targetQubit) {
    if (maestro && sim && fApplyCY)
      return fApplyCY(sim, controlQubit, targetQubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CY gate.");
    return 0;
  }

  int ApplyCZ(void* sim, int controlQubit, int targetQubit) {
    if (maestro && sim && fApplyCZ)
      return fApplyCZ(sim, controlQubit, targetQubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CZ gate.");
    return 0;
  }

  int ApplyCH(void* sim, int controlQubit, int targetQubit) {
    if (maestro && sim && fApplyCH)
      return fApplyCH(sim, controlQubit, targetQubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CH gate.");
    return 0;
  }

  int ApplyCSX(void* sim, int controlQubit, int targetQubit) {
    if (maestro && sim && fApplyCSX)
      return fApplyCSX(sim, controlQubit, targetQubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CSX gate.");
    return 0;
  }

  int ApplyCSXDG(void* sim, int controlQubit, int targetQubit) {
    if (maestro && sim && fApplyCSXDG)
      return fApplyCSXDG(sim, controlQubit, targetQubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CSXDG gate.");
    return 0;
  }

  int ApplyCP(void* sim, int controlQubit, int targetQubit, double theta) {
    if (maestro && sim && fApplyCP)
      return fApplyCP(sim, controlQubit, targetQubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CP gate.");
    return 0;
  }

  int ApplyCRx(void* sim, int controlQubit, int targetQubit, double theta) {
    if (maestro && sim && fApplyCRx)
      return fApplyCRx(sim, controlQubit, targetQubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CRx gate.");
    return 0;
  }

  int ApplyCRy(void* sim, int controlQubit, int targetQubit, double theta) {
    if (maestro && sim && fApplyCRy)
      return fApplyCRy(sim, controlQubit, targetQubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CRy gate.");
    return 0;
  }

  int ApplyCRz(void* sim, int controlQubit, int targetQubit, double theta) {
    if (maestro && sim && fApplyCRz)
      return fApplyCRz(sim, controlQubit, targetQubit, theta);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CRz gate.");
    return 0;
  }

  int ApplyCCX(void* sim, int controlQubit1, int controlQubit2,
               int targetQubit) {
    if (maestro && sim && fApplyCCX)
      return fApplyCCX(sim, controlQubit1, controlQubit2, targetQubit);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CCX gate.");
    return 0;
  }

  int ApplySwap(void* sim, int qubit1, int qubit2) {
    if (maestro && sim && fApplySwap)
      return fApplySwap(sim, qubit1, qubit2);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply Swap gate.");
    return 0;
  }

  int ApplyCSwap(void* sim, int controlQubit, int qubit1, int qubit2) {
    if (maestro && sim && fApplyCSwap)
      return fApplyCSwap(sim, controlQubit, qubit1, qubit2);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CSwap gate.");
    return 0;
  }

  int ApplyCU(void* sim, int controlQubit, int targetQubit, double theta,
              double phi, double lambda, double gamma) {
    if (maestro && sim && fApplyCU)
      return fApplyCU(sim, controlQubit, targetQubit, theta, phi, lambda,
                      gamma);
    else
      throw std::runtime_error("MaestroLibrary: Unable to apply CU gate.");
    return 0;
  }

  std::vector<double> ExpectationValues(
      void* sim, const std::vector<std::string>& paulis) {
    if (!maestro || !sim || !fExpectationValues)
      throw std::runtime_error("MaestroLibrary: Batch expectations unavailable.");
    std::vector<const char*> strings;
    strings.reserve(paulis.size());
    for (const auto& pauli : paulis) strings.push_back(pauli.c_str());
    std::vector<double> values(paulis.size());
    if (!fExpectationValues(sim, strings.data(), strings.size(), values.data(),
                            values.size()))
      throw std::runtime_error("MaestroLibrary: Batch expectations failed.");
    return values;
  }

  std::vector<std::complex<double>> ExpectationValuesComplex(
      void* sim, const std::vector<std::string>& paulis, bool normalized = true) {
    if (!maestro || !sim || !fExpectationValuesComplex)
      throw std::runtime_error("MaestroLibrary: Complex batch expectations unavailable.");
    std::vector<const char*> strings;
    strings.reserve(paulis.size());
    for (const auto& pauli : paulis) strings.push_back(pauli.c_str());
    std::vector<double> real(paulis.size()), imag(paulis.size());
    if (!fExpectationValuesComplex(sim, strings.data(), strings.size(),
                                   normalized ? 1 : 0, real.data(), imag.data(),
                                   real.size()))
      throw std::runtime_error("MaestroLibrary: Complex batch expectations failed.");
    std::vector<std::complex<double>> values;
    values.reserve(paulis.size());
    for (size_t i = 0; i < paulis.size(); ++i)
      values.emplace_back(real[i], imag[i]);
    return values;
  }

  // Versioned JSON requests return their complete response envelope.
  std::string RunRequestJson(const std::string& request) {
    return RequestJson(request, fRunRequestJson);
  }
  std::string ValidateRequestJson(const std::string& request) {
    return RequestJson(request, fValidateRequestJson);
  }

  std::vector<std::complex<double>> GetStateVector(void* sim) {
    if (!maestro || !sim || !fGetStateVector)
      throw std::runtime_error(
          "MaestroLibrary: Statevector query unavailable.");
    const auto n = GetNumberOfQubits(sim);
    if (n >= std::numeric_limits<size_t>::digits)
      throw std::length_error("Statevector exceeds the basis-index range");
    const size_t count = n ? size_t{1} << n : 0;
    if (count > std::vector<double>().max_size() / 2)
      throw std::length_error("Statevector output is too large");
    std::vector<std::complex<double>> result(count);
    if (!fGetStateVector(sim, reinterpret_cast<double*>(result.data()), count))
      throw std::runtime_error("MaestroLibrary: Statevector query failed.");
    return result;
  }

  // Four entries per operator, in row-major order; repeated targets are
  // allowed.
  std::complex<double> ExpectationValueOperators(
      void* sim, const std::vector<unsigned long>& qubits,
      const std::vector<std::array<std::complex<double>, 4>>& matrices) {
    if (!maestro || !sim || !fExpectationValueOperators)
      throw std::runtime_error(
          "MaestroLibrary: Operator expectation unavailable.");
    if (qubits.size() != matrices.size())
      throw std::invalid_argument("Each operator must have one target qubit");
    if (matrices.size() > std::vector<double>().max_size() / 8)
      throw std::length_error("Too many operators");
    std::vector<double> raw;
    raw.reserve(8 * matrices.size());
    for (const auto& matrix : matrices)
      for (const auto& entry : matrix) {
        raw.push_back(entry.real());
        raw.push_back(entry.imag());
      }
    double re = 0., im = 0.;
    if (!fExpectationValueOperators(sim, qubits.data(), qubits.size(),
                                    raw.data(), &re, &im))
      throw std::runtime_error("MaestroLibrary: Operator expectation failed.");
    return {re, im};
  }

  void MoveAtBeginningOfChain(void* sim,
                              const std::vector<unsigned long>& qubits) {
    if (!maestro || !sim || !fMoveAtBeginningOfChain)
      throw std::runtime_error("MaestroLibrary: Chain movement unavailable.");
    if (!fMoveAtBeginningOfChain(sim, qubits.data(), qubits.size()))
      throw std::runtime_error("MaestroLibrary: Chain movement failed.");
  }

 private:
  std::string RequestJson(const std::string& request,
                          char* (*call)(const char*)) {
    if (!maestro || !call || !fFreeResult)
      throw std::runtime_error("MaestroLibrary: JSON request API unavailable.");
    auto release = [this](char* p) { FreeResult(p); };
    std::unique_ptr<char, decltype(release)> result(call(request.c_str()),
                                                    release);
    if (!result)
      throw std::runtime_error(
          "MaestroLibrary: JSON request returned no response.");
    return result.get();
  }
  char* (*fRunRequestJson)(const char*) = nullptr;
  char* (*fValidateRequestJson)(const char*) = nullptr;
  int (*fGetStateVector)(void*, double*, size_t) = nullptr;
  int (*fExpectationValueOperators)(void*, const unsigned long*, size_t,
                                    const double*, double*, double*) = nullptr;
  int (*fMoveAtBeginningOfChain)(void*, const unsigned long*, size_t) = nullptr;
  int (*fExpectationValues)(void*, const char* const*, size_t, double*, size_t) = nullptr;
  int (*fExpectationValuesComplex)(void*, const char* const*, size_t, int,
                                   double*, double*, size_t) = nullptr;
  void* maestro = nullptr;

  void* (*fGetMaestroObject)();

  unsigned long int (*fCreateSimpleSimulator)(int);
  void (*fDestroySimpleSimulator)(unsigned long int);

  int (*fRemoveAllOptimizationSimulatorsAndAdd)(unsigned long int, int, int);
  int (*fAddOptimizationSimulator)(unsigned long int, int, int);

  char* (*fSimpleExecute)(unsigned long int, const char*, const char*);
  char* (*fSimpleEstimate)(unsigned long int, const char*, const char*,
                           const char*);
  void (*fFreeResult)(char*);

  unsigned long int (*fCreateSimulator)(int, int);
  void* (*fGetSimulator)(unsigned long int);
  void (*fDestroySimulator)(unsigned long int);

  int (*fInitializeSimulator)(void*);
  int (*fResetSimulator)(void*);
  int (*fConfigureSimulator)(void*, const char*, const char*);
  char* (*fGetConfiguration)(void*, const char*);
  unsigned long int (*fAllocateQubits)(void*, unsigned long int);
  unsigned long int (*fGetNumberOfQubits)(void*);
  int (*fClearSimulator)(void*);
  unsigned long long int (*fMeasure)(void*, const unsigned long int*,
                                     unsigned long int);
  int (*fApplyReset)(void*, const unsigned long int*, unsigned long int);
  double (*fProbability)(void*, unsigned long long int);
  void (*fFreeDoubleVector)(double*);
  void (*fFreeULLIVector)(unsigned long long int*);
  double* (*fAmplitude)(void*, unsigned long long int);
  double* (*fAllProbabilities)(void*);
  double* (*fProbabilities)(void*, const unsigned long long int*,
                            unsigned long int);
  unsigned long long int* (*fSampleCounts)(void*, const unsigned long long int*,
                                           unsigned long int,
                                           unsigned long int);
  int (*fGetSimulatorType)(void*);
  int (*fGetSimulationType)(void*);
  unsigned (*fGetGateFusionMaxQubits)(void*) = nullptr;
  int (*fIsGateFusionEnabled)(void*) = nullptr;
  int (*fGetGateFusionStatistics)(void*,
                                  MaestroGateFusionStatistics*) = nullptr;
  int (*fApplyGenericOneQubitGate)(void*, unsigned long,
                                   const double*) = nullptr;
  int (*fApplyGenericTwoQubitGate)(void*, unsigned long, unsigned long,
                                   const double*) = nullptr;
  int (*fApplyGenericThreeQubitGate)(void*, unsigned long, unsigned long,
                                     unsigned long, const double*) = nullptr;
  int (*fFlushSimulator)(void*);
  int (*fSaveStateToInternalDestructive)(void*);
  int (*fRestoreInternalDestructiveSavedState)(void*);
  int (*fSaveState)(void*);
  int (*fRestoreState)(void*);
  int (*fSetMultithreading)(void*, int);
  int (*fGetMultithreading)(void*);
  int (*fIsQcsim)(void*);
  unsigned long long int (*fMeasureNoCollapse)(void*);

  int (*fApplyX)(void*, int);
  int (*fApplyY)(void*, int);
  int (*fApplyZ)(void*, int);
  int (*fApplyH)(void*, int);
  int (*fApplyS)(void*, int);
  int (*fApplySDG)(void*, int);
  int (*fApplyT)(void*, int);
  int (*fApplyTDG)(void*, int);
  int (*fApplySX)(void*, int);
  int (*fApplySXDG)(void*, int);
  int (*fApplyK)(void*, int);
  int (*fApplyP)(void*, int, double);
  int (*fApplyRx)(void*, int, double);
  int (*fApplyRy)(void*, int, double);
  int (*fApplyRz)(void*, int, double);
  int (*fApplyU)(void*, int, double, double, double, double);
  int (*fApplyCX)(void*, int, int);
  int (*fApplyCY)(void*, int, int);
  int (*fApplyCZ)(void*, int, int);
  int (*fApplyCH)(void*, int, int);
  int (*fApplyCSX)(void*, int, int);
  int (*fApplyCSXDG)(void*, int, int);
  int (*fApplyCP)(void*, int, int, double);
  int (*fApplyCRx)(void*, int, int, double);
  int (*fApplyCRy)(void*, int, int, double);
  int (*fApplyCRz)(void*, int, int, double);
  int (*fApplyCCX)(void*, int, int, int);
  int (*fApplySwap)(void*, int, int);
  int (*fApplyCSwap)(void*, int, int, int);
  int (*fApplyCU)(void*, int, int, double, double, double, double);
};
