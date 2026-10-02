/**
 * @file interface.h
 * @ingroup core
 * @version 1.0
 *
 * @section DESCRIPTION
 * C interface API for the maestro library.
 */

#pragma once

#ifndef _MAESTRO_INTERFACE_H_
#define _MAESTRO_INTERFACE_H_

#include "InterfaceTypes.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
// Fusion queries do not flush. Width zero means Maestro fusion is unsupported.
// Statistics are cumulative for this simulator, including restored executions.
#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned GetGateFusionMaxQubits(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int IsGateFusionEnabled(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int GetGateFusionStatistics(void *sim,
                                MaestroGateFusionStatistics *statistics);

// Matrix buffers contain row-major, interleaved real/imaginary doubles:
// 8, 32 or 128 doubles for a 2x2, 4x4 or 8x8 matrix. The first target is
// the least-significant local matrix bit. Return 1 on success, 0 on error;
// no exception crosses these entry points. Buffers are copied during the call.
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyGenericOneQubitGate(void *sim, unsigned long qubit,
                                 const double *matrix);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyGenericTwoQubitGate(void *sim, unsigned long qubit0,
                                 unsigned long qubit1, const double *matrix);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyGenericThreeQubitGate(void *sim, unsigned long qubit0,
                                   unsigned long qubit1, unsigned long qubit2,
                                   const double *matrix);

// Native request API, schema version 2. All returned strings (including errors)
// are UTF-8 JSON owned by the caller and must be released with FreeResult.
// No C++ exception crosses these entry points. Null means allocation failure.
#ifdef _WIN32
__declspec(dllexport)
#endif
    char *MaestroRunRequestJson(const char *request);
#ifdef _WIN32
__declspec(dllexport)
#endif
    char *MaestroValidateRequestJson(const char *request);
#ifdef _WIN32
__declspec(dllexport)
#endif
    char *MaestroGetCapabilitiesJson(void);
#ifdef _WIN32
__declspec(dllexport)
#endif
    char *MaestroFinalizeDistributedMpiGpuJson(void);

#ifdef _WIN32
__declspec(dllexport)
#endif
    void *GetMaestroObject();

#ifdef _WIN32
__declspec(dllexport)
#endif
    void *GetMaestroObjectWithMute();

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int CreateSimpleSimulator(int nrQubits);
#ifdef _WIN32
__declspec(dllexport)
#endif
    void DestroySimpleSimulator(unsigned long int simHandle);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int RemoveAllOptimizationSimulatorsAndAdd(unsigned long int simHandle,
                                              int simType, int simExecType);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int AddOptimizationSimulator(unsigned long int simHandle, int simType,
                                 int simExecType);
#ifdef _WIN32
__declspec(dllexport)
#endif
    char *SimpleExecute(unsigned long int simpleSim, const char *circuitStr,
                        const char *jsonConfig);
#ifdef _WIN32
__declspec(dllexport)
#endif
    char *SimpleEstimate(unsigned long int simpleSim, const char *circuitStr,
                         const char *observableStr, const char *jsonConfig);
#ifdef _WIN32
__declspec(dllexport)
#endif
    void FreeResult(char *result);

#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int CreateSimulator(int simType, int simExecType);
#ifdef _WIN32
__declspec(dllexport)
#endif
    void *GetSimulator(unsigned long int simHandle);
#ifdef _WIN32
__declspec(dllexport)
#endif
    void DestroySimulator(unsigned long int simHandle);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int InitializeSimulator(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ResetSimulator(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ConfigureSimulator(void *sim, const char *key, const char *value);
#ifdef _WIN32
__declspec(dllexport)
#endif
    char *GetConfiguration(void *sim, const char *key);
#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int AllocateQubits(void *sim, unsigned long int nrQubits);
#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long int GetNumberOfQubits(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ClearSimulator(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long long int Measure(void *sim, const unsigned long int *qubits,
                                   unsigned long int nrQubits);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyReset(void *sim, const unsigned long int *qubits,
                   unsigned long int nrQubits);
#ifdef _WIN32
__declspec(dllexport)
#endif
    double Probability(void *sim, unsigned long long int outcome);
#ifdef _WIN32
__declspec(dllexport)
#endif
    void FreeDoubleVector(double *vec);
#ifdef _WIN32
__declspec(dllexport)
#endif
    void FreeULLIVector(unsigned long long int *vec);
#ifdef _WIN32
__declspec(dllexport)
#endif
    double *Amplitude(void *sim, unsigned long long int outcome);
#ifdef _WIN32
__declspec(dllexport)
#endif
    double *AllProbabilities(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    double *Probabilities(void *sim, const unsigned long long int *qubits,
                          unsigned long int nrQubits);
#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long long int *SampleCounts(void *sim,
                                         const unsigned long long int *qubits,
                                         unsigned long int nrQubits,
                                         unsigned long int shots);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int GetSimulatorType(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int GetSimulationType(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int FlushSimulator(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int SaveStateToInternalDestructive(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int RestoreInternalDestructiveSavedState(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int SaveState(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int RestoreState(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int SetMultithreading(void *sim, int multithreading);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int GetMultithreading(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int IsQcsim(void *sim);
#ifdef _WIN32
__declspec(dllexport)
#endif
    unsigned long long int MeasureNoCollapse(void *sim);

#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyX(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyY(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyZ(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyH(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyS(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySDG(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyT(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyTDG(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySX(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySXDG(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyK(void *sim, int qubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyP(void *sim, int qubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyRx(void *sim, int qubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyRy(void *sim, int qubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyRz(void *sim, int qubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyU(void *sim, int qubit, double theta, double phi, double lambda,
               double gamma);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCX(void *sim, int controlQubit, int targetQubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCY(void *sim, int controlQubit, int targetQubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCZ(void *sim, int controlQubit, int targetQubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCH(void *sim, int controlQubit, int targetQubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCSX(void *sim, int controlQubit, int targetQubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCSXDG(void *sim, int controlQubit, int targetQubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCP(void *sim, int controlQubit, int targetQubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCRx(void *sim, int controlQubit, int targetQubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCRy(void *sim, int controlQubit, int targetQubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCRz(void *sim, int controlQubit, int targetQubit, double theta);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCCX(void *sim, int controlQubit1, int controlQubit2,
                 int targetQubit);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplySwap(void *sim, int qubit1, int qubit2);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCSwap(void *sim, int controlQubit, int qubit1, int qubit2);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int ApplyCU(void *sim, int controlQubit, int targetQubit, double theta,
                double phi, double lambda, double gamma);
// MPO extensions. Return 1 on success, 0 on invalid arguments/backend failure.
// Bit vectors are q0-first; matrices are column-major interleaved doubles.
// GetDensityMatrix capacity counts doubles. SampleCountsBits capacity counts
// histogram entries and must be >= shots; outcomes needs capacity*width bytes,
// counts needs capacity entries. Each outcome follows the supplied qubit order.
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroProbabilityBits(void *sim, const unsigned char *bits,
                               size_t width, double *result);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroDensityMatrixElementBits(void *sim, const unsigned char *row,
                                        const unsigned char *col, size_t width,
                                        double *real, double *imag);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroDensityMatrixTrace(void *sim, double *real, double *imag);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroExpectationValueComplex(void *sim, const char *pauli,
                                       int normalized, double *real,
                                       double *imag);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroGetDensityMatrix(void *sim, int normalized, double *interleaved,
                                size_t capacity);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroApplyOperator(void *sim, const unsigned long int *qubits,
                             size_t count, const double *interleaved,
                             int normalize);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroMoveAtBeginningOfChain(void *sim,
                                      const unsigned long int *qubits,
                                      size_t count);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroSampleCountsBits(void *sim, const unsigned long int *qubits,
                                size_t width, size_t shots,
                                unsigned char *outcomes,
                                unsigned long long *counts, size_t capacity,
                                size_t *written);
#ifdef _WIN32
__declspec(dllexport)
#endif
    int MaestroMeasureBits(void *sim, const unsigned long int *qubits,
                           size_t width, unsigned char *outcomes);

#ifdef __cplusplus
}
#endif

#endif
