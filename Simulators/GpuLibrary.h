/**
 * @file GpuLibrary.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The Gpu library class.
 *
 * Allows loading the gpu library dynamically and exposes the c api functions.
 */

#pragma once

#ifndef _GPU_LIBRARY_H
#define _GPU_LIBRARY_H

#ifdef __linux__

#include "../Utils/Library.h"

#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace Simulators {

// use it as a singleton
class GpuLibrary : public Utils::Library {
 public:
  GpuLibrary(const GpuLibrary &) = delete;
  GpuLibrary &operator=(const GpuLibrary &) = delete;

  GpuLibrary(GpuLibrary &&) = default;
  GpuLibrary &operator=(GpuLibrary &&) = default;

  GpuLibrary() noexcept {}

  virtual ~GpuLibrary() {
    if (LibraryHandle) FreeLib();
  }

  bool Init(const char *libName) noexcept override {
    if (Utils::Library::Init(libName)) {
      InitLib = (void *(*)())GetFunction("InitLib");
      CheckFunction((void *)InitLib, __LINE__);
      if (InitLib) {
        LibraryHandle = InitLib();
        if (LibraryHandle) {
          FreeLib = (void (*)())GetFunction("FreeLib");
          CheckFunction((void *)FreeLib, __LINE__);

          // state vector api functions

          fCreateStateVector =
              (void *(*)(void *))GetFunction("CreateStateVector");
          CheckFunction((void *)fCreateStateVector, __LINE__);
          fDestroyStateVector =
              (void (*)(void *))GetFunction("DestroyStateVector");
          CheckFunction((void *)fDestroyStateVector, __LINE__);

          fCreate = (int (*)(void *, unsigned int))GetFunction("Create");
          CheckFunction((void *)fCreate, __LINE__);
          fCreateWithState =
              (int (*)(void *, unsigned int, const double *))GetFunction(
                  "CreateWithState");
          CheckFunction((void *)fCreateWithState, __LINE__);
          fReset = (int (*)(void *))GetFunction("Reset");
          CheckFunction((void *)fReset, __LINE__);

          fSetDataType = (int (*)(void *, int))GetFunction("SetDataType");
          CheckFunction((void *)fSetDataType, __LINE__);
          fIsDoublePrecision =
              (int (*)(void *))GetFunction("IsDoublePrecision");
          CheckFunction((void *)fIsDoublePrecision, __LINE__);
          fGetNrQubits = (int (*)(void *))GetFunction("GetNrQubits");
          CheckFunction((void *)fGetNrQubits, __LINE__);

          fMeasureQubitCollapse =
              (int (*)(void *, int))GetFunction("MeasureQubitCollapse");
          CheckFunction((void *)fMeasureQubitCollapse, __LINE__);
          fMeasureQubitNoCollapse =
              (int (*)(void *, int))GetFunction("MeasureQubitNoCollapse");
          CheckFunction((void *)fMeasureQubitNoCollapse, __LINE__);
          fMeasureQubitsCollapse = (int (*)(
              void *, int *, int *, int))GetFunction("MeasureQubitsCollapse");
          CheckFunction((void *)fMeasureQubitsCollapse, __LINE__);
          fMeasureQubitsNoCollapse = (int (*)(
              void *, int *, int *, int))GetFunction("MeasureQubitsNoCollapse");
          CheckFunction((void *)fMeasureQubitsNoCollapse, __LINE__);
          fMeasureAllQubitsCollapse = (unsigned long long (*)(
              void *))GetFunction("MeasureAllQubitsCollapse");
          CheckFunction((void *)fMeasureAllQubitsCollapse, __LINE__);
          fMeasureAllQubitsNoCollapse = (unsigned long long (*)(
              void *))GetFunction("MeasureAllQubitsNoCollapse");
          CheckFunction((void *)fMeasureAllQubitsNoCollapse, __LINE__);

          fSaveState = (int (*)(void *))GetFunction("SaveState");
          CheckFunction((void *)fSaveState, __LINE__);
          fSaveStateToHost = (int (*)(void *))GetFunction("SaveStateToHost");
          CheckFunction((void *)fSaveStateToHost, __LINE__);
          fSaveStateDestructive =
              (int (*)(void *))GetFunction("SaveStateDestructive");
          CheckFunction((void *)fSaveStateDestructive, __LINE__);
          fRestoreStateFreeSaved =
              (int (*)(void *))GetFunction("RestoreStateFreeSaved");
          CheckFunction((void *)fRestoreStateFreeSaved, __LINE__);
          fRestoreStateNoFreeSaved =
              (int (*)(void *))GetFunction("RestoreStateNoFreeSaved");
          CheckFunction((void *)fRestoreStateNoFreeSaved, __LINE__);
          fFreeSavedState = (void (*)(void *))GetFunction("FreeSavedState");
          CheckFunction((void *)fFreeSavedState, __LINE__);
          fClone = (void *(*)(void *))GetFunction("Clone");
          CheckFunction((void *)fClone, __LINE__);

          fSample = (int (*)(void *, unsigned int, long int *, unsigned int,
                             int *))GetFunction("Sample");
          CheckFunction((void *)fSample, __LINE__);
          fSampleAll = (int (*)(void *, unsigned int, long int *))GetFunction(
              "SampleAll");
          CheckFunction((void *)fSampleAll, __LINE__);
          fAmplitude = (int (*)(void *, long long int, double *,
                                double *))GetFunction("Amplitude");
          CheckFunction((void *)fAmplitude, __LINE__);
          fProbability =
              (double (*)(void *, int *, int *, int))GetFunction("Probability");
          CheckFunction((void *)fProbability, __LINE__);
          fBasisStateProbability = (double (*)(
              void *, long long int))GetFunction("BasisStateProbability");
          CheckFunction((void *)fBasisStateProbability, __LINE__);
          fAllProbabilities = (int (*)(
              void *obj, double *probabilities))GetFunction("AllProbabilities");
          CheckFunction((void *)fAllProbabilities, __LINE__);
          fExpectationValue = (double (*)(void *, const char *,
                                          int))GetFunction("ExpectationValue");
          CheckFunction((void *)fExpectationValue, __LINE__);

          fApplyX = (int (*)(void *, int))GetFunction("ApplyX");
          CheckFunction((void *)fApplyX, __LINE__);
          fApplyY = (int (*)(void *, int))GetFunction("ApplyY");
          CheckFunction((void *)fApplyY, __LINE__);
          fApplyZ = (int (*)(void *, int))GetFunction("ApplyZ");
          CheckFunction((void *)fApplyZ, __LINE__);
          fApplyH = (int (*)(void *, int))GetFunction("ApplyH");
          CheckFunction((void *)fApplyH, __LINE__);
          fApplyS = (int (*)(void *, int))GetFunction("ApplyS");
          CheckFunction((void *)fApplyS, __LINE__);
          fApplySDG = (int (*)(void *, int))GetFunction("ApplySDG");
          CheckFunction((void *)fApplySDG, __LINE__);
          fApplyT = (int (*)(void *, int))GetFunction("ApplyT");
          CheckFunction((void *)fApplyT, __LINE__);
          fApplyTDG = (int (*)(void *, int))GetFunction("ApplyTDG");
          CheckFunction((void *)fApplyTDG, __LINE__);
          fApplySX = (int (*)(void *, int))GetFunction("ApplySX");
          CheckFunction((void *)fApplySX, __LINE__);
          fApplySXDG = (int (*)(void *, int))GetFunction("ApplySXDG");
          CheckFunction((void *)fApplySXDG, __LINE__);
          fApplyK = (int (*)(void *, int))GetFunction("ApplyK");
          CheckFunction((void *)fApplyK, __LINE__);
          fApplyP = (int (*)(void *, int, double))GetFunction("ApplyP");
          CheckFunction((void *)fApplyP, __LINE__);
          fApplyRx = (int (*)(void *, int, double))GetFunction("ApplyRx");
          CheckFunction((void *)fApplyRx, __LINE__);
          fApplyRy = (int (*)(void *, int, double))GetFunction("ApplyRy");
          CheckFunction((void *)fApplyRy, __LINE__);
          fApplyRz = (int (*)(void *, int, double))GetFunction("ApplyRz");
          CheckFunction((void *)fApplyRz, __LINE__);
          fApplyU = (int (*)(void *, int, double, double, double,
                             double))GetFunction("ApplyU");
          CheckFunction((void *)fApplyU, __LINE__);
          fApplyCX = (int (*)(void *, int, int))GetFunction("ApplyCX");
          CheckFunction((void *)fApplyCX, __LINE__);
          fApplyCY = (int (*)(void *, int, int))GetFunction("ApplyCY");
          CheckFunction((void *)fApplyCY, __LINE__);
          fApplyCZ = (int (*)(void *, int, int))GetFunction("ApplyCZ");
          CheckFunction((void *)fApplyCZ, __LINE__);
          fApplyCH = (int (*)(void *, int, int))GetFunction("ApplyCH");
          CheckFunction((void *)fApplyCH, __LINE__);
          fApplyCSX = (int (*)(void *, int, int))GetFunction("ApplyCSX");
          CheckFunction((void *)fApplyCSX, __LINE__);
          fApplyCSXDG = (int (*)(void *, int, int))GetFunction("ApplyCSXDG");
          CheckFunction((void *)fApplyCSXDG, __LINE__);
          fApplyCP = (int (*)(void *, int, int, double))GetFunction("ApplyCP");
          CheckFunction((void *)fApplyCP, __LINE__);
          fApplyCRx =
              (int (*)(void *, int, int, double))GetFunction("ApplyCRx");
          CheckFunction((void *)fApplyCRx, __LINE__);
          fApplyCRy =
              (int (*)(void *, int, int, double))GetFunction("ApplyCRy");
          CheckFunction((void *)fApplyCRy, __LINE__);
          fApplyCRz =
              (int (*)(void *, int, int, double))GetFunction("ApplyCRz");
          CheckFunction((void *)fApplyCRz, __LINE__);
          fApplyCCX = (int (*)(void *, int, int, int))GetFunction("ApplyCCX");
          CheckFunction((void *)fApplyCCX, __LINE__);
          fApplySwap = (int (*)(void *, int, int))GetFunction("ApplySwap");
          CheckFunction((void *)fApplySwap, __LINE__);
          fApplyCSwap =
              (int (*)(void *, int, int, int))GetFunction("ApplyCSwap");
          CheckFunction((void *)fApplyCSwap, __LINE__);
          fApplyCU = (int (*)(void *, int, int, double, double, double,
                              double))GetFunction("ApplyCU");
          CheckFunction((void *)fApplyCU, __LINE__);

          // mps api functions

          fCreateMPS = (void *(*)(void *))GetFunction("CreateMPS");
          CheckFunction((void *)fCreateMPS, __LINE__);
          fDestroyMPS = (void (*)(void *))GetFunction("DestroyMPS");
          CheckFunction((void *)fDestroyMPS, __LINE__);

          fMPSCreate = (int (*)(void *, unsigned int))GetFunction("MPSCreate");
          CheckFunction((void *)fMPSCreate, __LINE__);
          fMPSReset = (int (*)(void *))GetFunction("MPSReset");
          CheckFunction((void *)fMPSReset, __LINE__);

          fMPSIsValid = (int (*)(void *))GetFunction("MPSIsValid");
          CheckFunction((void *)fMPSIsValid, __LINE__);
          fMPSIsCreated = (int (*)(void *))GetFunction("MPSIsCreated");
          CheckFunction((void *)fMPSIsCreated, __LINE__);

          fMPSSetDataType = (int (*)(void *, int))GetFunction("MPSSetDataType");
          CheckFunction((void *)fMPSSetDataType, __LINE__);
          fMPSIsDoublePrecision =
              (int (*)(void *))GetFunction("MPSIsDoublePrecision");
          CheckFunction((void *)fMPSIsDoublePrecision, __LINE__);
          fMPSSetCutoff = (int (*)(void *, double))GetFunction("MPSSetCutoff");
          CheckFunction((void *)fMPSSetCutoff, __LINE__);
          fMPSGetCutoff = (double (*)(void *))GetFunction("MPSGetCutoff");
          CheckFunction((void *)fMPSGetCutoff, __LINE__);
          fMPSSetGesvdJ = (int (*)(void *, int))GetFunction("MPSSetGesvdJ");
          CheckFunction((void *)fMPSSetGesvdJ, __LINE__);
          fMPSGetGesvdJ = (int (*)(void *))GetFunction("MPSGetGesvdJ");
          CheckFunction((void *)fMPSGetGesvdJ, __LINE__);
          fMPSSetMaxExtent =
              (int (*)(void *, long int))GetFunction("MPSSetMaxExtent");
          CheckFunction((void *)fMPSSetMaxExtent, __LINE__);
          fMPSGetMaxExtent =
              (long int (*)(void *))GetFunction("MPSGetMaxExtent");
          CheckFunction((void *)fMPSGetMaxExtent, __LINE__);
          fMPSGetNrQubits = (int (*)(void *))GetFunction("MPSGetNrQubits");
          CheckFunction((void *)fMPSGetNrQubits, __LINE__);
          fMPSAmplitude = (int (*)(void *, long int, long int *, double *,
                                   double *))GetFunction("MPSAmplitude");
          CheckFunction((void *)fMPSAmplitude, __LINE__);
          fMPSProbability0 =
              (double (*)(void *, unsigned int))GetFunction("MPSProbability0");
          CheckFunction((void *)fMPSProbability0, __LINE__);
          fMPSMeasure =
              (int (*)(void *, unsigned int))GetFunction("MPSMeasure");
          CheckFunction((void *)fMPSMeasure, __LINE__);
          fMPSMeasureQubits = (int (*)(void *, long int, unsigned int *,
                                       int *))GetFunction("MPSMeasureQubits");
          CheckFunction((void *)fMPSMeasureQubits, __LINE__);

          fMPSGetMapForSample = (void *(*)())GetFunction("MPSGetMapForSample");
          CheckFunction((void *)fMPSGetMapForSample, __LINE__);
          fMPSFreeMapForSample =
              (int (*)(void *))GetFunction("MPSFreeMapForSample");
          CheckFunction((void *)fMPSFreeMapForSample, __LINE__);
          fMPSSample = (int (*)(void *, long int, long int, unsigned int *,
                                void *))GetFunction("MPSSample");
          CheckFunction((void *)fMPSSample, __LINE__);

          fMPSSaveState = (int (*)(void *))GetFunction("MPSSaveState");
          CheckFunction((void *)fMPSSaveState, __LINE__);
          fMPSRestoreState = (int (*)(void *))GetFunction("MPSRestoreState");
          CheckFunction((void *)fMPSRestoreState, __LINE__);
          fMPSCleanSavedState =
              (int (*)(void *))GetFunction("MPSCleanSavedState");
          CheckFunction((void *)fMPSCleanSavedState, __LINE__);
          fMPSClone = (void *(*)(void *))GetFunction("MPSClone");
          CheckFunction((void *)fMPSClone, __LINE__);

          fMPSExpectationValue = (double (*)(
              void *, const char *, int))GetFunction("MPSExpectationValue");
          CheckFunction((void *)fMPSExpectationValue, __LINE__);

          fMPSApplyX = (int (*)(void *, unsigned int))GetFunction("MPSApplyX");
          CheckFunction((void *)fMPSApplyX, __LINE__);
          fMPSApplyY = (int (*)(void *, unsigned int))GetFunction("MPSApplyY");
          CheckFunction((void *)fMPSApplyY, __LINE__);
          fMPSApplyZ = (int (*)(void *, unsigned int))GetFunction("MPSApplyZ");
          CheckFunction((void *)fMPSApplyZ, __LINE__);
          fMPSApplyH = (int (*)(void *, unsigned int))GetFunction("MPSApplyH");
          CheckFunction((void *)fMPSApplyH, __LINE__);
          fMPSApplyS = (int (*)(void *, unsigned int))GetFunction("MPSApplyS");
          CheckFunction((void *)fMPSApplyS, __LINE__);
          fMPSApplySDG =
              (int (*)(void *, unsigned int))GetFunction("MPSApplySDG");
          CheckFunction((void *)fMPSApplySDG, __LINE__);
          fMPSApplyT = (int (*)(void *, unsigned int))GetFunction("MPSApplyT");
          CheckFunction((void *)fMPSApplyT, __LINE__);
          fMPSApplyTDG =
              (int (*)(void *, unsigned int))GetFunction("MPSApplyTDG");
          CheckFunction((void *)fMPSApplyTDG, __LINE__);
          fMPSApplySX =
              (int (*)(void *, unsigned int))GetFunction("MPSApplySX");
          CheckFunction((void *)fMPSApplySX, __LINE__);
          fMPSApplySXDG =
              (int (*)(void *, unsigned int))GetFunction("MPSApplySXDG");
          CheckFunction((void *)fMPSApplySXDG, __LINE__);
          fMPSApplyK = (int (*)(void *, unsigned int))GetFunction("MPSApplyK");
          CheckFunction((void *)fMPSApplyK, __LINE__);
          fMPSApplyP =
              (int (*)(void *, unsigned int, double))GetFunction("MPSApplyP");
          CheckFunction((void *)fMPSApplyP, __LINE__);
          fMPSApplyRx =
              (int (*)(void *, unsigned int, double))GetFunction("MPSApplyRx");
          CheckFunction((void *)fMPSApplyRx, __LINE__);
          fMPSApplyRy =
              (int (*)(void *, unsigned int, double))GetFunction("MPSApplyRy");
          CheckFunction((void *)fMPSApplyRy, __LINE__);
          fMPSApplyRz =
              (int (*)(void *, unsigned int, double))GetFunction("MPSApplyRz");
          CheckFunction((void *)fMPSApplyRz, __LINE__);
          fMPSApplyU = (int (*)(void *, unsigned int, double, double, double,
                                double))GetFunction("MPSApplyU");
          CheckFunction((void *)fMPSApplyU, __LINE__);
          fMPSApplySwap = (int (*)(void *, unsigned int,
                                   unsigned int))GetFunction("MPSApplySwap");
          CheckFunction((void *)fMPSApplySwap, __LINE__);
          fMPSApplyCX = (int (*)(void *, unsigned int,
                                 unsigned int))GetFunction("MPSApplyCX");
          CheckFunction((void *)fMPSApplyCX, __LINE__);
          fMPSApplyCY = (int (*)(void *, unsigned int,
                                 unsigned int))GetFunction("MPSApplyCY");
          CheckFunction((void *)fMPSApplyCY, __LINE__);
          fMPSApplyCZ = (int (*)(void *, unsigned int,
                                 unsigned int))GetFunction("MPSApplyCZ");
          CheckFunction((void *)fMPSApplyCZ, __LINE__);
          fMPSApplyCH = (int (*)(void *, unsigned int,
                                 unsigned int))GetFunction("MPSApplyCH");
          CheckFunction((void *)fMPSApplyCH, __LINE__);
          fMPSApplyCSX = (int (*)(void *, unsigned int,
                                  unsigned int))GetFunction("MPSApplyCSX");
          CheckFunction((void *)fMPSApplyCSX, __LINE__);
          fMPSApplyCSXDG = (int (*)(void *, unsigned int,
                                    unsigned int))GetFunction("MPSApplyCSXDG");
          CheckFunction((void *)fMPSApplyCSXDG, __LINE__);
          fMPSApplyCP = (int (*)(void *, unsigned int, unsigned int,
                                 double))GetFunction("MPSApplyCP");
          CheckFunction((void *)fMPSApplyCP, __LINE__);
          fMPSApplyCRx = (int (*)(void *, unsigned int, unsigned int,
                                  double))GetFunction("MPSApplyCRx");
          CheckFunction((void *)fMPSApplyCRx, __LINE__);
          fMPSApplyCRy = (int (*)(void *, unsigned int, unsigned int,
                                  double))GetFunction("MPSApplyCRy");
          CheckFunction((void *)fMPSApplyCRy, __LINE__);
          fMPSApplyCRz = (int (*)(void *, unsigned int, unsigned int,
                                  double))GetFunction("MPSApplyCRz");
          CheckFunction((void *)fMPSApplyCRz, __LINE__);
          fMPSApplyCU =
              (int (*)(void *, unsigned int, unsigned int, double, double,
                       double, double))GetFunction("MPSApplyCU");
          CheckFunction((void *)fMPSApplyCU, __LINE__);

          // tensor network api functions

          fCreateTensorNet = (void *(*)(void *))GetFunction("CreateTensorNet");
          CheckFunction((void *)fCreateTensorNet, __LINE__);
          fDestroyTensorNet = (void (*)(void *))GetFunction("DestroyTensorNet");
          CheckFunction((void *)fDestroyTensorNet, __LINE__);

          fTNCreate = (int (*)(void *, unsigned int))GetFunction("TNCreate");
          CheckFunction((void *)fTNCreate, __LINE__);
          fTNReset = (int (*)(void *))GetFunction("TNReset");
          CheckFunction((void *)fTNReset, __LINE__);

          fTNIsValid = (int (*)(void *))GetFunction("TNIsValid");
          CheckFunction((void *)fTNIsValid, __LINE__);
          fTNIsCreated = (int (*)(void *))GetFunction("TNIsCreated");
          CheckFunction((void *)fTNIsCreated, __LINE__);

          fTNSetDataType = (int (*)(void *, int))GetFunction("TNSetDataType");
          CheckFunction((void *)fTNSetDataType, __LINE__);
          fTNIsDoublePrecision =
              (int (*)(void *))GetFunction("TNIsDoublePrecision");
          CheckFunction((void *)fTNIsDoublePrecision, __LINE__);
          fTNSetCutoff = (int (*)(void *, double))GetFunction("TNSetCutoff");
          CheckFunction((void *)fTNSetCutoff, __LINE__);
          fTNGetCutoff = (double (*)(void *))GetFunction("TNGetCutoff");
          CheckFunction((void *)fTNGetCutoff, __LINE__);
          fTNSetGesvdJ = (int (*)(void *, int))GetFunction("TNSetGesvdJ");
          CheckFunction((void *)fTNSetGesvdJ, __LINE__);
          fTNGetGesvdJ = (int (*)(void *))GetFunction("TNGetGesvdJ");
          CheckFunction((void *)fTNGetGesvdJ, __LINE__);
          fTNSetMaxExtent =
              (int (*)(void *, long int))GetFunction("TNSetMaxExtent");
          CheckFunction((void *)fTNSetMaxExtent, __LINE__);
          fTNGetMaxExtent = (long int (*)(void *))GetFunction("TNGetMaxExtent");
          CheckFunction((void *)fTNGetMaxExtent, __LINE__);
          fTNGetNrQubits = (int (*)(void *))GetFunction("TNGetNrQubits");
          CheckFunction((void *)fTNGetNrQubits, __LINE__);
          fTNAmplitude = (int (*)(void *, long int, long int *, double *,
                                  double *))GetFunction("TNAmplitude");
          CheckFunction((void *)fTNAmplitude, __LINE__);
          fTNProbability0 =
              (double (*)(void *, unsigned int))GetFunction("TNProbability0");
          CheckFunction((void *)fTNProbability0, __LINE__);
          fTNMeasure = (int (*)(void *, unsigned int))GetFunction("TNMeasure");
          CheckFunction((void *)fTNMeasure, __LINE__);
          fTNMeasureQubits = (int (*)(void *, long int, unsigned int *,
                                      int *))GetFunction("TNMeasureQubits");
          CheckFunction((void *)fTNMeasureQubits, __LINE__);

          fTNGetMapForSample = (void *(*)())GetFunction("TNGetMapForSample");
          CheckFunction((void *)fTNGetMapForSample, __LINE__);
          fTNFreeMapForSample =
              (int (*)(void *))GetFunction("TNFreeMapForSample");
          CheckFunction((void *)fTNFreeMapForSample, __LINE__);
          fTNSample = (int (*)(void *, long int, long int, unsigned int *,
                               void *))GetFunction("TNSample");
          CheckFunction((void *)fTNSample, __LINE__);

          fTNSaveState = (int (*)(void *))GetFunction("TNSaveState");
          CheckFunction((void *)fTNSaveState, __LINE__);
          fTNRestoreState = (int (*)(void *))GetFunction("TNRestoreState");
          CheckFunction((void *)fTNRestoreState, __LINE__);
          fTNCleanSavedState =
              (int (*)(void *))GetFunction("TNCleanSavedState");
          CheckFunction((void *)fTNCleanSavedState, __LINE__);
          fTNClone = (void *(*)(void *))GetFunction("TNClone");
          CheckFunction((void *)fTNClone, __LINE__);

          fTNExpectationValue = (double (*)(
              void *, const char *, int))GetFunction("TNExpectationValue");
          CheckFunction((void *)fTNExpectationValue, __LINE__);

          fTNApplyX = (int (*)(void *, unsigned int))GetFunction("TNApplyX");
          CheckFunction((void *)fTNApplyX, __LINE__);
          fTNApplyY = (int (*)(void *, unsigned int))GetFunction("TNApplyY");
          CheckFunction((void *)fTNApplyY, __LINE__);
          fTNApplyZ = (int (*)(void *, unsigned int))GetFunction("TNApplyZ");
          CheckFunction((void *)fTNApplyZ, __LINE__);
          fTNApplyH = (int (*)(void *, unsigned int))GetFunction("TNApplyH");
          CheckFunction((void *)fTNApplyH, __LINE__);
          fTNApplyS = (int (*)(void *, unsigned int))GetFunction("TNApplyS");
          CheckFunction((void *)fTNApplyS, __LINE__);
          fTNApplySDG =
              (int (*)(void *, unsigned int))GetFunction("TNApplySDG");
          CheckFunction((void *)fTNApplySDG, __LINE__);
          fTNApplyT = (int (*)(void *, unsigned int))GetFunction("TNApplyT");
          CheckFunction((void *)fTNApplyT, __LINE__);
          fTNApplyTDG =
              (int (*)(void *, unsigned int))GetFunction("TNApplyTDG");
          CheckFunction((void *)fTNApplyTDG, __LINE__);
          fTNApplySX = (int (*)(void *, unsigned int))GetFunction("TNApplySX");
          CheckFunction((void *)fTNApplySX, __LINE__);
          fTNApplySXDG =
              (int (*)(void *, unsigned int))GetFunction("TNApplySXDG");
          CheckFunction((void *)fTNApplySXDG, __LINE__);
          fTNApplyK = (int (*)(void *, unsigned int))GetFunction("TNApplyK");
          CheckFunction((void *)fTNApplyK, __LINE__);
          fTNApplyP =
              (int (*)(void *, unsigned int, double))GetFunction("TNApplyP");
          CheckFunction((void *)fTNApplyP, __LINE__);
          fTNApplyRx =
              (int (*)(void *, unsigned int, double))GetFunction("TNApplyRx");
          CheckFunction((void *)fTNApplyRx, __LINE__);
          fTNApplyRy =
              (int (*)(void *, unsigned int, double))GetFunction("TNApplyRy");
          CheckFunction((void *)fTNApplyRy, __LINE__);
          fTNApplyRz =
              (int (*)(void *, unsigned int, double))GetFunction("TNApplyRz");
          CheckFunction((void *)fTNApplyRz, __LINE__);
          fTNApplyU = (int (*)(void *, unsigned int, double, double, double,
                               double))GetFunction("TNApplyU");
          CheckFunction((void *)fTNApplyU, __LINE__);
          fTNApplySwap = (int (*)(void *, unsigned int,
                                  unsigned int))GetFunction("TNApplySwap");
          CheckFunction((void *)fTNApplySwap, __LINE__);
          fTNApplyCX = (int (*)(void *, unsigned int, unsigned int))GetFunction(
              "TNApplyCX");
          CheckFunction((void *)fTNApplyCX, __LINE__);
          fTNApplyCY = (int (*)(void *, unsigned int, unsigned int))GetFunction(
              "TNApplyCY");
          CheckFunction((void *)fTNApplyCY, __LINE__);
          fTNApplyCZ = (int (*)(void *, unsigned int, unsigned int))GetFunction(
              "TNApplyCZ");
          CheckFunction((void *)fTNApplyCZ, __LINE__);
          fTNApplyCH = (int (*)(void *, unsigned int, unsigned int))GetFunction(
              "TNApplyCH");
          CheckFunction((void *)fTNApplyCH, __LINE__);
          fTNApplyCSX = (int (*)(void *, unsigned int,
                                 unsigned int))GetFunction("TNApplyCSX");
          CheckFunction((void *)fTNApplyCSX, __LINE__);
          fTNApplyCSXDG = (int (*)(void *, unsigned int,
                                   unsigned int))GetFunction("TNApplyCSXDG");
          CheckFunction((void *)fTNApplyCSXDG, __LINE__);
          fTNApplyCP = (int (*)(void *, unsigned int, unsigned int,
                                double))GetFunction("TNApplyCP");
          CheckFunction((void *)fTNApplyCP, __LINE__);
          fTNApplyCRx = (int (*)(void *, unsigned int, unsigned int,
                                 double))GetFunction("TNApplyCRx");
          CheckFunction((void *)fTNApplyCRx, __LINE__);
          fTNApplyCRy = (int (*)(void *, unsigned int, unsigned int,
                                 double))GetFunction("TNApplyCRy");
          CheckFunction((void *)fTNApplyCRy, __LINE__);
          fTNApplyCRz = (int (*)(void *, unsigned int, unsigned int,
                                 double))GetFunction("TNApplyCRz");
          CheckFunction((void *)fTNApplyCRz, __LINE__);
          fTNApplyCU =
              (int (*)(void *, unsigned int, unsigned int, double, double,
                       double, double))GetFunction("TNApplyCU");
          CheckFunction((void *)fTNApplyCU, __LINE__);

          fTNApplyCCX = (int (*)(void *, unsigned int, unsigned int,
                                 unsigned int))GetFunction("TNApplyCCX");
          CheckFunction((void *)fTNApplyCCX, __LINE__);
          fTNApplyCSwap = (int (*)(void *, unsigned int, unsigned int,
                                   unsigned int))GetFunction("TNApplyCSwap");
          CheckFunction((void *)fTNApplyCSwap, __LINE__);

          // stabilizer simulator functions
          fCreateStabilizerSimulator = (void *(*)(long long int, long long int,
                                                  long long int, long long int))
              GetFunction("CreateStabilizerSimulator");
          CheckFunction((void *)fCreateStabilizerSimulator, __LINE__);
          fDestroyStabilizerSimulator =
              (void (*)(void *))GetFunction("DestroyStabilizerSimulator");
          CheckFunction((void *)fDestroyStabilizerSimulator, __LINE__);
          fExecuteStabilizerCircuit = (int (*)(
              void *, const char *, int,
              unsigned long long int))GetFunction("ExecuteStabilizerCircuit");
          CheckFunction((void *)fExecuteStabilizerCircuit, __LINE__);
          fGetStabilizerXZTableSize =
              (long long (*)(void *))GetFunction("GetStabilizerXZTableSize");
          CheckFunction((void *)fGetStabilizerXZTableSize, __LINE__);
          fGetStabilizerMTableSize =
              (long long (*)(void *))GetFunction("GetStabilizerMTableSize");
          CheckFunction((void *)fGetStabilizerMTableSize, __LINE__);

          fGetStabilizerTableStrideMajor = (long long (*)(void *))GetFunction(
              "GetStabilizerTableStrideMajor");
          CheckFunction((void *)fGetStabilizerTableStrideMajor, __LINE__);

          fGetStabilizerNumQubits =
              (long long (*)(void *))GetFunction("GetStabilizerNumQubits");
          CheckFunction((void *)fGetStabilizerNumQubits, __LINE__);
          fGetStabilizerNumShots =
              (long long (*)(void *))GetFunction("GetStabilizerNumShots");
          CheckFunction((void *)fGetStabilizerNumShots, __LINE__);
          fGetStabilizerNumMeasurements = (long long (*)(void *))GetFunction(
              "GetStabilizerNumMeasurements");
          CheckFunction((void *)fGetStabilizerNumMeasurements, __LINE__);
          fGetStabilizerNumDetectors =
              (long long (*)(void *))GetFunction("GetStabilizerNumDetectors");
          CheckFunction((void *)fGetStabilizerNumDetectors, __LINE__);
          fCopyStabilizerXTable = (int (*)(void *, unsigned int *))GetFunction(
              "CopyStabilizerXTable");
          CheckFunction((void *)fCopyStabilizerXTable, __LINE__);
          fCopyStabilizerZTable = (int (*)(void *, unsigned int *))GetFunction(
              "CopyStabilizerZTable");
          CheckFunction((void *)fCopyStabilizerZTable, __LINE__);
          fCopyStabilizerMTable = (int (*)(void *, unsigned int *))GetFunction(
              "CopyStabilizerMTable");
          CheckFunction((void *)fCopyStabilizerMTable, __LINE__);

          fInitStabilizerXTable = (int (*)(
              void *, const unsigned int *))GetFunction("InitXTable");
          CheckFunction((void *)fInitStabilizerXTable, __LINE__);
          fInitStabilizerZTable = (int (*)(
              void *, const unsigned int *))GetFunction("InitZTable");
          CheckFunction((void *)fInitStabilizerZTable, __LINE__);

          // pauli propagation functions
          fCreatePauliPropSimulator = (void *(*)(int))GetFunction("CreatePauliPropSimulator");
          CheckFunction((void *)fCreatePauliPropSimulator, __LINE__);
          fDestroyPauliPropSimulator = (void (*)(void *))GetFunction("DestroyPauliPropSimulator");
          CheckFunction((void *)fDestroyPauliPropSimulator, __LINE__);

          fPauliPropGetNrQubits = (int (*)(void *))GetFunction("PauliPropGetNrQubits");
          CheckFunction((void *)fPauliPropGetNrQubits, __LINE__);
          fPauliPropSetWillUseSampling = (int (*)(void *, int))GetFunction("PauliPropSetWillUseSampling");
          CheckFunction((void *)fPauliPropSetWillUseSampling, __LINE__);
          fPauliPropGetWillUseSampling = (int (*)(void *))GetFunction("PauliPropGetWillUseSampling");
          CheckFunction((void *)fPauliPropGetWillUseSampling, __LINE__);

          fPauliPropGetCoefficientTruncationCutoff = (double (*)(void *))GetFunction("PauliPropGetCoefficientTruncationCutoff");
          CheckFunction((void *)fPauliPropGetCoefficientTruncationCutoff, __LINE__);
          fPauliPropSetCoefficientTruncationCutoff = (void (*)(void *, double))GetFunction("PauliPropSetCoefficientTruncationCutoff");
          CheckFunction((void *)fPauliPropSetCoefficientTruncationCutoff, __LINE__);
          fPauliPropGetWeightTruncationCutoff = (double (*)(void *))GetFunction("PauliPropGetWeightTruncationCutoff");
          CheckFunction((void *)fPauliPropGetWeightTruncationCutoff, __LINE__);
          fPauliPropSetWeightTruncationCutoff = (void (*)(void *, double))GetFunction("PauliPropSetWeightTruncationCutoff");
          CheckFunction((void *)fPauliPropSetWeightTruncationCutoff, __LINE__);
          fPauliPropGetNumGatesBetweenTruncations = (int (*)(void *))GetFunction("PauliPropGetNumGatesBetweenTruncations");
          CheckFunction((void *)fPauliPropGetNumGatesBetweenTruncations, __LINE__);
          fPauliPropSetNumGatesBetweenTruncations = (void (*)(void *, int))GetFunction("PauliPropSetNumGatesBetweenTruncations");
          CheckFunction((void *)fPauliPropSetNumGatesBetweenTruncations, __LINE__);
          fPauliPropGetNumGatesBetweenDeduplications = (int (*)(void *))GetFunction("PauliPropGetNumGatesBetweenDeduplications");
          CheckFunction((void *)fPauliPropGetNumGatesBetweenDeduplications, __LINE__);
          fPauliPropSetNumGatesBetweenDeduplications = (void (*)(void *, int))GetFunction("PauliPropSetNumGatesBetweenDeduplications");
          CheckFunction((void *)fPauliPropSetNumGatesBetweenDeduplications, __LINE__);


          fPauliPropClearOperators = (int (*)(void *))GetFunction("PauliPropClearOperators");
          CheckFunction((void *)fPauliPropClearOperators, __LINE__);
          fPauliPropAllocateMemory = (int (*)(void *, double))GetFunction("PauliPropAllocateMemory");
          CheckFunction((void *)fPauliPropAllocateMemory, __LINE__);

          fPauliPropGetExpectationValue = (double (*)(void *))GetFunction("PauliPropGetExpectationValue");
          CheckFunction((void *)fPauliPropGetExpectationValue, __LINE__);
          fPauliPropExecute = (int (*)(void *))GetFunction("PauliPropExecute");
          CheckFunction((void *)fPauliPropExecute, __LINE__);
          fPauliPropSetInPauliExpansionUnique = (int (*)(void *, const char *))GetFunction("PauliPropSetInPauliExpansionUnique");
          CheckFunction((void *)fPauliPropSetInPauliExpansionUnique, __LINE__);
          fPauliPropSetInPauliExpansionMultiple = (int (*)(void *, const char **,
                                                       const double *, int))GetFunction("PauliPropSetInPauliExpansionMultiple");
          CheckFunction((void *)fPauliPropSetInPauliExpansionMultiple, __LINE__);

          fPauliPropApplyX = (int (*)(void *, int))GetFunction("PauliPropApplyX");
          CheckFunction((void *)fPauliPropApplyX, __LINE__);
          fPauliPropApplyY = (int (*)(void *, int))GetFunction("PauliPropApplyY");
          CheckFunction((void *)fPauliPropApplyY, __LINE__);
          fPauliPropApplyZ = (int (*)(void *, int))GetFunction("PauliPropApplyZ");
          CheckFunction((void *)fPauliPropApplyZ, __LINE__);
          fPauliPropApplyH = (int (*)(void *, int))GetFunction("PauliPropApplyH");
          CheckFunction((void *)fPauliPropApplyH, __LINE__);
          fPauliPropApplyS = (int (*)(void *, int))GetFunction("PauliPropApplyS");
          CheckFunction((void *)fPauliPropApplyS, __LINE__);

          fPauliPropApplySQRTX = (int (*)(void *, int))GetFunction("PauliPropApplySQRTX");
          CheckFunction((void *)fPauliPropApplySQRTX, __LINE__);
          fPauliPropApplySQRTY = (int (*)(void *, int))GetFunction("PauliPropApplySQRTY");
          CheckFunction((void *)fPauliPropApplySQRTY, __LINE__);
          fPauliPropApplySQRTZ = (int (*)(void *, int))GetFunction("PauliPropApplySQRTZ");
          CheckFunction((void *)fPauliPropApplySQRTZ, __LINE__);
          fPauliPropApplyCX = (int (*)(void *, int, int))GetFunction("PauliPropApplyCX");
          CheckFunction((void *)fPauliPropApplyCX, __LINE__);
          fPauliPropApplyCY = (int (*)(void *, int, int))GetFunction("PauliPropApplyCY");
          CheckFunction((void *)fPauliPropApplyCY, __LINE__);
          fPauliPropApplyCZ = (int (*)(void *, int, int))GetFunction("PauliPropApplyCZ");
          CheckFunction((void *)fPauliPropApplyCZ, __LINE__);
          fPauliPropApplySWAP = (int (*)(void *, int, int))GetFunction("PauliPropApplySWAP");
          CheckFunction((void *)fPauliPropApplySWAP, __LINE__);
          fPauliPropApplyISWAP = (int (*)(void *, int, int))GetFunction("PauliPropApplyISWAP");
          CheckFunction((void *)fPauliPropApplyISWAP, __LINE__);
          fPauliPropApplyRX = (int (*)(void *, int, double))GetFunction("PauliPropApplyRX");
          CheckFunction((void *)fPauliPropApplyRX, __LINE__);
          fPauliPropApplyRY = (int (*)(void *, int, double))GetFunction("PauliPropApplyRY");
          CheckFunction((void *)fPauliPropApplyRY, __LINE__);
          fPauliPropApplyRZ = (int (*)(void *, int, double))GetFunction("PauliPropApplyRZ");
          CheckFunction((void *)fPauliPropApplyRZ, __LINE__);

          fPauliPropAddNoiseX = (int (*)(void *, int, double))GetFunction("PauliPropAddNoiseX");
          CheckFunction((void *)fPauliPropAddNoiseX, __LINE__);
          fPauliPropAddNoiseY = (int (*)(void *, int, double))GetFunction("PauliPropAddNoiseY");
          CheckFunction((void *)fPauliPropAddNoiseY, __LINE__);
          fPauliPropAddNoiseZ = (int (*)(void *, int, double))GetFunction("PauliPropAddNoiseZ");
          CheckFunction((void *)fPauliPropAddNoiseZ, __LINE__);
          fPauliPropAddNoiseXYZ = (int (*)(void *, int, double, double, double))GetFunction("PauliPropAddNoiseXYZ");
          CheckFunction((void *)fPauliPropAddNoiseXYZ, __LINE__);
          fPauliPropAddAmplitudeDamping = (int (*)(void *, int, double, double))GetFunction("PauliPropAddAmplitudeDamping");
          CheckFunction((void *)fPauliPropAddAmplitudeDamping, __LINE__);
          fPauliPropQubitProbability0 = (double (*)(void *, int))GetFunction("PauliPropQubitProbability0");
          CheckFunction((void *)fPauliPropQubitProbability0, __LINE__);
          fPauliPropProbability = (double (*)(void *, int, unsigned long long int))GetFunction("PauliPropProbability");
          CheckFunction((void *)fPauliPropProbability, __LINE__);

          fPauliPropMeasureQubit = (int (*)(void *, int))GetFunction("PauliPropMeasureQubit");
          CheckFunction((void *)fPauliPropMeasureQubit, __LINE__);

          fPauliPropSampleQubits = (unsigned char *(*)(void *, const int *, int))GetFunction("PauliPropSampleQubits");
          CheckFunction((void *)fPauliPropSampleQubits, __LINE__);
          fPauliPropFreeSampledQubits = (void (*)(unsigned char *))GetFunction("PauliPropFreeSampledQubits");
          CheckFunction((void *)fPauliPropFreeSampledQubits, __LINE__);
          fPauliPropSaveState = (void (*)(void *))GetFunction("PauliPropSaveState");
          CheckFunction((void *)fPauliPropSaveState, __LINE__);
          fPauliPropRestoreState = (void (*)(void *))GetFunction("PauliPropRestoreState");
          CheckFunction((void *)fPauliPropRestoreState, __LINE__);

          return true;
        } else
          std::cerr << "GpuLibrary: Unable to initialize gpu library"
                    << std::endl;
      } else
        std::cerr << "GpuLibrary: Unable to get initialization function for "
                     "gpu library"
                  << std::endl;
    } else if (!Utils::Library::IsMuted())
      std::cerr << "GpuLibrary: Unable to load gpu library" << std::endl;

    return false;
  }

  static void CheckFunction(void *func, int line) {
    if (!func) {
      std::cerr << "GpuLibrary: Unable to load function, line #: " << line;
      const char *dlsym_error = dlerror();
      if (dlsym_error) std::cerr << ", error: " << dlsym_error;

      std::cerr << std::endl;
    }
  }

  bool IsValid() const { return LibraryHandle != nullptr; }

  // statevector functions

  void *CreateStateVector() {
    if (LibraryHandle)
      return fCreateStateVector(LibraryHandle);
    else
      throw std::runtime_error("GpuLibrary: Unable to create state vector");
  }

  void DestroyStateVector(void *obj) {
    if (LibraryHandle)
      fDestroyStateVector(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to destroy state vector");
  }

  bool Create(void *obj, unsigned int nrQubits) {
    if (LibraryHandle)
      return fCreate(obj, nrQubits) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to create state vector state");

    return false;
  }

  bool CreateWithState(void *obj, unsigned int nrQubits, const double *state) {
    if (LibraryHandle)
      return fCreateWithState(obj, nrQubits, state) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to create state vector state with a state");

    return false;
  }

  bool Reset(void *obj) {
    if (LibraryHandle)
      return fReset(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to reset state vector");

    return false;
  }

  bool SetDataType(void *obj, int dataType) {
    if (LibraryHandle)
      return fSetDataType(obj, dataType) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to set data type");

    return false;
  }

  bool IsDoublePrecision(void *obj) const {
    if (LibraryHandle)
      return fIsDoublePrecision(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to check if double precision");

    return false;
  }

  int GetNrQubits(void *obj) const {
    if (LibraryHandle)
      return fGetNrQubits(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to get number of qubits");
    return 0;
  }

  bool MeasureQubitCollapse(void *obj, int qubitIndex) {
    if (LibraryHandle)
      return fMeasureQubitCollapse(obj, qubitIndex) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure qubit with collapse");

    return false;
  }

  bool MeasureQubitNoCollapse(void *obj, int qubitIndex) {
    if (LibraryHandle)
      return fMeasureQubitNoCollapse(obj, qubitIndex) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure qubit no collapse");

    return false;
  }

  bool MeasureQubitsCollapse(void *obj, int *qubits, int *bitstring,
                             int bitstringLen) {
    if (LibraryHandle)
      return fMeasureQubitsCollapse(obj, qubits, bitstring, bitstringLen) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure qubits with collapse");

    return false;
  }

  bool MeasureQubitsNoCollapse(void *obj, int *qubits, int *bitstring,
                               int bitstringLen) {
    if (LibraryHandle)
      return fMeasureQubitsNoCollapse(obj, qubits, bitstring, bitstringLen) ==
             1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure qubits with no collapse");

    return false;
  }

  unsigned long long MeasureAllQubitsCollapse(void *obj) {
    if (LibraryHandle)
      return fMeasureAllQubitsCollapse(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure all qubits with collapse");

    return 0;
  }

  unsigned long long MeasureAllQubitsNoCollapse(void *obj) {
    if (LibraryHandle)
      return fMeasureAllQubitsNoCollapse(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure all qubits with no collapse");

    return 0;
  }

  bool SaveState(void *obj) {
    if (LibraryHandle)
      return fSaveState(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to save state");

    return false;
  }

  bool SaveStateToHost(void *obj) {
    if (LibraryHandle)
      return fSaveStateToHost(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to save state to host");

    return false;
  }

  bool SaveStateDestructive(void *obj) {
    if (LibraryHandle)
      return fSaveStateDestructive(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to save state destructively");

    return false;
  }

  bool RestoreStateFreeSaved(void *obj) {
    if (LibraryHandle)
      return fRestoreStateFreeSaved(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to restore state free saved");

    return false;
  }

  bool RestoreStateNoFreeSaved(void *obj) {
    if (LibraryHandle)
      return fRestoreStateNoFreeSaved(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to restore state no free saved");

    return false;
  }

  void FreeSavedState(void *obj) {
    if (LibraryHandle)
      fFreeSavedState(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to free saved state");
  }

  void *Clone(void *obj) const {
    if (LibraryHandle)
      return fClone(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to clone state vector");

    return nullptr;
  }

  bool Sample(void *obj, unsigned int nSamples, long int *samples,
              unsigned int nBits, int *bits) {
    if (LibraryHandle)
      return fSample(obj, nSamples, samples, nBits, bits) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to sample state vector");

    return false;
  }

  bool SampleAll(void *obj, unsigned int nSamples, long int *samples) {
    if (LibraryHandle)
      return fSampleAll(obj, nSamples, samples) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to sample state vector");

    return false;
  }

  bool Amplitude(void *obj, long long int state, double *real,
                 double *imaginary) const {
    if (LibraryHandle)
      return fAmplitude(obj, state, real, imaginary) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to get amplitude");

    return false;
  }

  double Probability(void *obj, int *qubits, int *mask, int len) const {
    if (LibraryHandle)
      return fProbability(obj, qubits, mask, len);
    else
      throw std::runtime_error("GpuLibrary: Unable to get probability");

    return 0;
  }

  double BasisStateProbability(void *obj, long long int state) const {
    if (LibraryHandle)
      return fBasisStateProbability(obj, state);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get basis state probability");

    return 0;
  }

  bool AllProbabilities(void *obj, double *probabilities) const {
    if (LibraryHandle)
      return fAllProbabilities(obj, probabilities) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to get all probabilities");

    return false;
  }

  double ExpectationValue(void *obj, const char *pauliString, int len) const {
    if (LibraryHandle)
      return fExpectationValue(obj, pauliString, len);
    else
      throw std::runtime_error("GpuLibrary: Unable to get expectation value");

    return 0;
  }

  bool ApplyX(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyX(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply X gate");

    return false;
  }

  bool ApplyY(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyY(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Y gate");

    return false;
  }

  bool ApplyZ(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyZ(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Z gate");

    return false;
  }

  bool ApplyH(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyH(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply H gate");

    return false;
  }

  bool ApplyS(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyS(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply S gate");

    return false;
  }

  bool ApplySDG(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplySDG(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply SDG gate");

    return false;
  }

  bool ApplyT(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyT(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply T gate");

    return false;
  }

  bool ApplyTDG(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyTDG(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply TDG gate");

    return false;
  }

  bool ApplySX(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplySX(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply SX gate");

    return false;
  }

  bool ApplySXDG(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplySXDG(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply SXDG gate");

    return false;
  }

  bool ApplyK(void *obj, int qubit) {
    if (LibraryHandle)
      return fApplyK(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply K gate");

    return false;
  }

  bool ApplyP(void *obj, int qubit, double theta) {
    if (LibraryHandle)
      return fApplyP(obj, qubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply P gate");

    return false;
  }

  bool ApplyRx(void *obj, int qubit, double theta) {
    if (LibraryHandle)
      return fApplyRx(obj, qubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Rx gate");

    return false;
  }

  bool ApplyRy(void *obj, int qubit, double theta) {
    if (LibraryHandle)
      return fApplyRy(obj, qubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Ry gate");

    return false;
  }

  bool ApplyRz(void *obj, int qubit, double theta) {
    if (LibraryHandle)
      return fApplyRz(obj, qubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Rz gate");

    return false;
  }

  bool ApplyU(void *obj, int qubit, double theta, double phi, double lambda,
              double gamma) {
    if (LibraryHandle)
      return fApplyU(obj, qubit, theta, phi, lambda, gamma) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply U gate");

    return false;
  }

  bool ApplyCX(void *obj, int controlQubit, int targetQubit) {
    if (LibraryHandle)
      return fApplyCX(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CX gate");

    return false;
  }

  bool ApplyCY(void *obj, int controlQubit, int targetQubit) {
    if (LibraryHandle)
      return fApplyCY(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CY gate");

    return false;
  }

  bool ApplyCZ(void *obj, int controlQubit, int targetQubit) {
    if (LibraryHandle)
      return fApplyCZ(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CZ gate");

    return false;
  }

  bool ApplyCH(void *obj, int controlQubit, int targetQubit) {
    if (LibraryHandle)
      return fApplyCH(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CH gate");

    return false;
  }

  bool ApplyCSX(void *obj, int controlQubit, int targetQubit) {
    if (LibraryHandle)
      return fApplyCSX(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CSX gate");

    return false;
  }

  bool ApplyCSXDG(void *obj, int controlQubit, int targetQubit) {
    if (LibraryHandle)
      return fApplyCSXDG(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CSXDG gate");

    return false;
  }

  bool ApplyCP(void *obj, int controlQubit, int targetQubit, double theta) {
    if (LibraryHandle)
      return fApplyCP(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CP gate");

    return false;
  }

  bool ApplyCRx(void *obj, int controlQubit, int targetQubit, double theta) {
    if (LibraryHandle)
      return fApplyCRx(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CRx gate");

    return false;
  }

  bool ApplyCRy(void *obj, int controlQubit, int targetQubit, double theta) {
    if (LibraryHandle)
      return fApplyCRy(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CRy gate");

    return false;
  }

  bool ApplyCRz(void *obj, int controlQubit, int targetQubit, double theta) {
    if (LibraryHandle)
      return fApplyCRz(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CRz gate");

    return false;
  }

  bool ApplyCCX(void *obj, int controlQubit1, int controlQubit2,
                int targetQubit) {
    if (LibraryHandle)
      return fApplyCCX(obj, controlQubit1, controlQubit2, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CCX gate");

    return false;
  }

  bool ApplySwap(void *obj, int qubit1, int qubit2) {
    if (LibraryHandle)
      return fApplySwap(obj, qubit1, qubit2) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Swap gate");

    return false;
  }

  bool ApplyCSwap(void *obj, int controlQubit, int qubit1, int qubit2) {
    if (LibraryHandle)
      return fApplyCSwap(obj, controlQubit, qubit1, qubit2) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CSwap gate");

    return false;
  }

  bool ApplyCU(void *obj, int controlQubit, int targetQubit, double theta,
               double phi, double lambda, double gamma) {
    if (LibraryHandle)
      return fApplyCU(obj, controlQubit, targetQubit, theta, phi, lambda,
                      gamma) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CU gate");

    return false;
  }

  // mps functions

  void *CreateMPS() {
    if (LibraryHandle)
      return fCreateMPS(LibraryHandle);
    else
      throw std::runtime_error("GpuLibrary: Unable to create mps");
  }

  void DestroyMPS(void *obj) {
    if (LibraryHandle)
      fDestroyMPS(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to destroy mps");
  }

  bool MPSCreate(void *obj, unsigned int nrQubits) {
    if (LibraryHandle)
      return fMPSCreate(obj, nrQubits) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to create mps with the "
          "specified number of qubits");

    return false;
  }

  bool MPSReset(void *obj) {
    if (LibraryHandle)
      return fMPSReset(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to reset mps");

    return false;
  }

  bool MPSIsValid(void *obj) const {
    if (LibraryHandle)
      return fMPSIsValid(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to check if mps is valid");

    return false;
  }

  bool MPSIsCreated(void *obj) const {
    if (LibraryHandle)
      return fMPSIsCreated(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to check if mps is created");

    return false;
  }

  bool MPSSetDataType(void *obj, int useDoublePrecision) {
    if (LibraryHandle)
      return fMPSSetDataType(obj, useDoublePrecision) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to set precision for mps");

    return false;
  }

  bool MPSIsDoublePrecision(void *obj) const {
    if (LibraryHandle)
      return fMPSIsDoublePrecision(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to get precision for mps");

    return false;
  }

  bool MPSSetCutoff(void *obj, double val) {
    if (LibraryHandle)
      return fMPSSetCutoff(obj, val) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to set cutoff for mps");

    return false;
  }

  double MPSGetCutoff(void *obj) const {
    if (LibraryHandle)
      return fMPSGetCutoff(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to get cutoff for mps");
  }

  bool MPSSetGesvdJ(void *obj, int val) {
    if (LibraryHandle)
      return fMPSSetGesvdJ(obj, val) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to set GesvdJ for mps");

    return false;
  }

  bool MPSGetGesvdJ(void *obj) const {
    if (LibraryHandle)
      return fMPSGetGesvdJ(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to get GesvdJ for mps");

    return false;
  }

  bool MPSSetMaxExtent(void *obj, long int val) {
    if (LibraryHandle)
      return fMPSSetMaxExtent(obj, val) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to set max extent for mps");

    return false;
  }

  long int MPSGetMaxExtent(void *obj) {
    if (LibraryHandle)
      return fMPSGetMaxExtent(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to get max extent for mps");

    return 0;
  }

  int MPSGetNrQubits(void *obj) {
    if (LibraryHandle)
      return fMPSGetNrQubits(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to get nr qubits for mps");

    return 0;
  }

  bool MPSAmplitude(void *obj, long int numFixedValues, long int *fixedValues,
                    double *real, double *imaginary) {
    if (LibraryHandle)
      return fMPSAmplitude(obj, numFixedValues, fixedValues, real, imaginary) ==
             1;
    else
      throw std::runtime_error("GpuLibrary: Unable to get mps amplitude");

    return false;
  }

  double MPSProbability0(void *obj, unsigned int qubit) {
    if (LibraryHandle)
      return fMPSProbability0(obj, qubit);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get probability for 0 for mps");

    return 0.0;
  }

  bool MPSMeasure(void *obj, unsigned int qubit) {
    if (LibraryHandle)
      return fMPSMeasure(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to measure qubit on mps");

    return false;
  }

  bool MPSMeasureQubits(void *obj, long int numQubits, unsigned int *qubits,
                        int *result) {
    if (LibraryHandle)
      return fMPSMeasureQubits(obj, numQubits, qubits, result) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to measure qubits on mps");

    return false;
  }

  std::unordered_map<std::vector<bool>, int64_t> *MPSGetMapForSample() {
    if (LibraryHandle)
      return (std::unordered_map<std::vector<bool>, int64_t> *)
          fMPSGetMapForSample();
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get map for sample for mps");

    return nullptr;
  }

  bool MPSFreeMapForSample(
      std::unordered_map<std::vector<bool>, int64_t> *map) {
    if (LibraryHandle)
      return fMPSFreeMapForSample((void *)map) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to free map for sample for mps");

    return false;
  }

  bool MPSSample(void *obj, long int numShots, long int numQubits,
                 unsigned int *qubits, void *resultMap) {
    if (LibraryHandle)
      return fMPSSample(obj, numShots, numQubits, qubits, resultMap) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to sample mps");

    return false;
  }

  bool MPSSaveState(void *obj) {
    if (LibraryHandle)
      return fMPSSaveState(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to save mps state");

    return false;
  }

  bool MPSRestoreState(void *obj) {
    if (LibraryHandle)
      return fMPSRestoreState(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to restore mps state");

    return false;
  }

  bool MPSCleanSavedState(void *obj) {
    if (LibraryHandle)
      return fMPSCleanSavedState(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to clean mps saved state");

    return false;
  }

  void *MPSClone(void *obj) {
    if (LibraryHandle)
      return fMPSClone(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to clone mps");

    return nullptr;
  }

  double MPSExpectationValue(void *obj, const char *pauliString,
                             int len) const {
    if (LibraryHandle)
      return fMPSExpectationValue(obj, pauliString, len);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get mps expectation value");

    return 0;
  }

  bool MPSApplyX(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyX(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply X gate on mps");

    return false;
  }

  bool MPSApplyY(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyY(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Y gate on mps");

    return false;
  }

  bool MPSApplyZ(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyZ(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Z gate on mps");

    return false;
  }

  bool MPSApplyH(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyH(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply H gate on mps");

    return false;
  }

  bool MPSApplyS(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyS(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply S gate on mps");

    return false;
  }

  bool MPSApplySDG(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplySDG(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply sdg gate on mps");

    return false;
  }

  bool MPSApplyT(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyT(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply t gate on mps");

    return false;
  }

  bool MPSApplyTDG(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyTDG(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to appl tdg gate on mps");

    return false;
  }

  bool MPSApplySX(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplySX(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply sx gate on mps");

    return false;
  }

  bool MPSApplySXDG(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplySXDG(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply sxdg gate on mps");

    return false;
  }

  bool MPSApplyK(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fMPSApplyK(obj, siteA) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply k gate on mps");

    return false;
  }

  bool MPSApplyP(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fMPSApplyP(obj, siteA, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply p gate on mps");
    return false;
  }

  bool MPSApplyRx(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fMPSApplyRx(obj, siteA, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply rx gate on mps");

    return false;
  }

  bool MPSApplyRy(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fMPSApplyRy(obj, siteA, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply ry gate on mps");

    return false;
  }

  bool MPSApplyRz(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fMPSApplyRz(obj, siteA, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply rz gate on mps");

    return false;
  }

  bool MPSApplyU(void *obj, unsigned int siteA, double theta, double phi,
                 double lambda, double gamma) {
    if (LibraryHandle)
      return fMPSApplyU(obj, siteA, theta, phi, lambda, gamma) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply u gate on mps");

    return false;
  }

  bool MPSApplySwap(void *obj, unsigned int controlQubit,
                    unsigned int targetQubit) {
    if (LibraryHandle)
      return fMPSApplySwap(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply swap gate on mps");

    return false;
  }

  bool MPSApplyCX(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit) {
    if (LibraryHandle)
      return fMPSApplyCX(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply cx gate on mps");

    return false;
  }

  bool MPSApplyCY(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit) {
    if (LibraryHandle)
      return fMPSApplyCY(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply cy gate on mps");

    return false;
  }

  bool MPSApplyCZ(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit) {
    if (LibraryHandle)
      return fMPSApplyCZ(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply cz gate on mps");

    return false;
  }

  bool MPSApplyCH(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit) {
    if (LibraryHandle)
      return fMPSApplyCH(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply ch gate on mps");

    return false;
  }

  bool MPSApplyCSX(void *obj, unsigned int controlQubit,
                   unsigned int targetQubit) {
    if (LibraryHandle)
      return fMPSApplyCSX(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply csx gate on mps");
  }

  bool MPSApplyCSXDG(void *obj, unsigned int controlQubit,
                     unsigned int targetQubit) {
    if (LibraryHandle)
      return fMPSApplyCSXDG(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply csxdg gate on mps");

    return false;
  }

  bool MPSApplyCP(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit, double theta) {
    if (LibraryHandle)
      return fMPSApplyCP(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply cp gate on mps");

    return false;
  }

  bool MPSApplyCRx(void *obj, unsigned int controlQubit,
                   unsigned int targetQubit, double theta) {
    if (LibraryHandle)
      return fMPSApplyCRx(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply crx gate on mps");

    return false;
  }

  bool MPSApplyCRy(void *obj, unsigned int controlQubit,
                   unsigned int targetQubit, double theta) {
    if (LibraryHandle)
      return fMPSApplyCRy(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply cry gate on mps");

    return false;
  }

  bool MPSApplyCRz(void *obj, unsigned int controlQubit,
                   unsigned int targetQubit, double theta) {
    if (LibraryHandle)
      return fMPSApplyCRz(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply crz gate on mps");

    return false;
  }

  bool MPSApplyCU(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit, double theta, double phi,
                  double lambda, double gamma) {
    if (LibraryHandle)
      return fMPSApplyCU(obj, controlQubit, targetQubit, theta, phi, lambda,
                         gamma) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply cu gate on mps");

    return false;
  }

  // tensor network functions

  void *CreateTensorNet() {
    if (LibraryHandle)
      return fCreateTensorNet(LibraryHandle);
    else
      throw std::runtime_error("GpuLibrary: Unable to create tensor network");
  }

  void DestroyTensorNet(void *obj) {
    if (LibraryHandle)
      fDestroyTensorNet(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to destroy tensor network");
  }

  bool TNCreate(void *obj, unsigned int nrQubits) {
    if (LibraryHandle)
      return fTNCreate(obj, nrQubits) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to create tensor network with the "
          "specified number of qubits");

    return false;
  }

  bool TNReset(void *obj) {
    if (LibraryHandle)
      return fTNReset(obj) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to reset tensor network");

    return false;
  }

  bool TNIsValid(void *obj) const {
    if (LibraryHandle)
      return fTNIsValid(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to check if tensor network is valid");

    return false;
  }

  bool TNIsCreated(void *obj) const {
    if (LibraryHandle)
      return fTNIsCreated(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to check if tensor network is created");

    return false;
  }

  bool TNSetDataType(void *obj, int useDoublePrecision) {
    if (LibraryHandle)
      return fTNSetDataType(obj, useDoublePrecision) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set precision for tensor network");

    return false;
  }

  bool TNIsDoublePrecision(void *obj) const {
    if (LibraryHandle)
      return fTNIsDoublePrecision(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get precision for tensor network");

    return false;
  }

  bool TNSetCutoff(void *obj, double val) {
    if (LibraryHandle)
      return fTNSetCutoff(obj, val) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set cutoff for tensor network");

    return false;
  }

  double TNGetCutoff(void *obj) const {
    if (LibraryHandle)
      return fTNGetCutoff(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get cutoff for tensor network");
  }

  bool TNSetGesvdJ(void *obj, int val) {
    if (LibraryHandle)
      return fTNSetGesvdJ(obj, val) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set GesvdJ for tensor network");

    return false;
  }

  bool TNGetGesvdJ(void *obj) const {
    if (LibraryHandle)
      return fTNGetGesvdJ(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get GesvdJ for tensor network");

    return false;
  }

  bool TNSetMaxExtent(void *obj, long int val) {
    if (LibraryHandle)
      return fTNSetMaxExtent(obj, val) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set max extent for tensor network");

    return false;
  }

  long int TNGetMaxExtent(void *obj) {
    if (LibraryHandle)
      return fTNGetMaxExtent(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get max extent for tensor network");

    return 0;
  }

  int TNGetNrQubits(void *obj) {
    if (LibraryHandle)
      return fTNGetNrQubits(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get nr qubits for tensor network");

    return 0;
  }

  bool TNAmplitude(void *obj, long int numFixedValues, long int *fixedValues,
                   double *real, double *imaginary) {
    if (LibraryHandle)
      return fTNAmplitude(obj, numFixedValues, fixedValues, real, imaginary) ==
             1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get tensor network amplitude");

    return false;
  }

  double TNProbability0(void *obj, unsigned int qubit) {
    if (LibraryHandle)
      return fTNProbability0(obj, qubit);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get probability for 0 for tensor network");

    return 0.0;
  }

  bool TNMeasure(void *obj, unsigned int qubit) {
    if (LibraryHandle)
      return fTNMeasure(obj, qubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure qubit on tensor network");

    return false;
  }

  bool TNMeasureQubits(void *obj, long int numQubits, unsigned int *qubits,
                       int *result) {
    if (LibraryHandle)
      return fTNMeasureQubits(obj, numQubits, qubits, result) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure qubits on tensor network");

    return false;
  }

  std::unordered_map<std::vector<bool>, int64_t> *TNGetMapForSample() {
    if (LibraryHandle)
      return (
          std::unordered_map<std::vector<bool>, int64_t> *)fTNGetMapForSample();
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get map for sample for tensor network");

    return nullptr;
  }

  bool TNFreeMapForSample(std::unordered_map<std::vector<bool>, int64_t> *map) {
    if (LibraryHandle)
      return fTNFreeMapForSample((void *)map) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to free map for sample for tensor network");

    return false;
  }

  bool TNSample(void *obj, long int numShots, long int numQubits,
                unsigned int *qubits, void *resultMap) {
    if (LibraryHandle)
      return fTNSample(obj, numShots, numQubits, qubits, resultMap) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to sample tensor network");

    return false;
  }

  bool TNSaveState(void *obj) {
    if (LibraryHandle)
      return fTNSaveState(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to save tensor network state");

    return false;
  }

  bool TNRestoreState(void *obj) {
    if (LibraryHandle)
      return fTNRestoreState(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to restore tensor network state");

    return false;
  }

  bool TNCleanSavedState(void *obj) {
    if (LibraryHandle)
      return fTNCleanSavedState(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to clean tensor network saved state");

    return false;
  }

  void *TNClone(void *obj) {
    if (LibraryHandle)
      return fTNClone(obj);
    else
      throw std::runtime_error("GpuLibrary: Unable to clone tensor network");

    return nullptr;
  }

  double TNExpectationValue(void *obj, const char *pauliString, int len) const {
    if (LibraryHandle)
      return fTNExpectationValue(obj, pauliString, len);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get tensor network expectation value");

    return 0;
  }

  bool TNApplyX(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyX(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply X gate on tensor network");

    return false;
  }

  bool TNApplyY(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyY(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply Y gate on tensor network");

    return false;
  }

  bool TNApplyZ(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyZ(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply Z gate on tensor network");

    return false;
  }

  bool TNApplyH(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyH(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply H gate on tensor network");

    return false;
  }

  bool TNApplyS(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyS(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply S gate on tensor network");

    return false;
  }

  bool TNApplySDG(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplySDG(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply sdg gate on tensor network");

    return false;
  }

  bool TNApplyT(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyT(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply t gate on tensor network");

    return false;
  }

  bool TNApplyTDG(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyTDG(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply tdg gate on tensor network");

    return false;
  }

  bool TNApplySX(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplySX(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply sx gate on tensor network");

    return false;
  }

  bool TNApplySXDG(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplySXDG(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply sxdg gate on tensor network");

    return false;
  }

  bool TNApplyK(void *obj, unsigned int siteA) {
    if (LibraryHandle)
      return fTNApplyK(obj, siteA) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply k gate on tensor network");

    return false;
  }

  bool TNApplyP(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fTNApplyP(obj, siteA, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply p gate on tensor network");
    return false;
  }

  bool TNApplyRx(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fTNApplyRx(obj, siteA, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply rx gate on tensor network");

    return false;
  }

  bool TNApplyRy(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fTNApplyRy(obj, siteA, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply ry gate on tensor network");

    return false;
  }

  bool TNApplyRz(void *obj, unsigned int siteA, double theta) {
    if (LibraryHandle)
      return fTNApplyRz(obj, siteA, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply rz gate on tensor network");

    return false;
  }

  bool TNApplyU(void *obj, unsigned int siteA, double theta, double phi,
                double lambda, double gamma) {
    if (LibraryHandle)
      return fTNApplyU(obj, siteA, theta, phi, lambda, gamma) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply u gate on tensor network");

    return false;
  }

  bool TNApplySwap(void *obj, unsigned int controlQubit,
                   unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplySwap(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply swap gate on tensor network");

    return false;
  }

  bool TNApplyCX(void *obj, unsigned int controlQubit,
                 unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplyCX(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply cx gate on tensor network");

    return false;
  }

  bool TNApplyCY(void *obj, unsigned int controlQubit,
                 unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplyCY(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply cy gate on tensor network");

    return false;
  }

  bool TNApplyCZ(void *obj, unsigned int controlQubit,
                 unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplyCZ(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply cz gate on tensor network");

    return false;
  }

  bool TNApplyCH(void *obj, unsigned int controlQubit,
                 unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplyCH(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply ch gate on tensor network");

    return false;
  }

  bool TNApplyCSX(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplyCSX(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply csx gate on tensor network");
  }

  bool TNApplyCSXDG(void *obj, unsigned int controlQubit,
                    unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplyCSXDG(obj, controlQubit, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply csxdg gate on tensor network");

    return false;
  }

  bool TNApplyCP(void *obj, unsigned int controlQubit, unsigned int targetQubit,
                 double theta) {
    if (LibraryHandle)
      return fTNApplyCP(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply cp gate on tensor network");

    return false;
  }

  bool TNApplyCRx(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit, double theta) {
    if (LibraryHandle)
      return fTNApplyCRx(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply crx gate on tensor network");

    return false;
  }

  bool TNApplyCRy(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit, double theta) {
    if (LibraryHandle)
      return fTNApplyCRy(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply cry gate on tensor network");

    return false;
  }

  bool TNApplyCRz(void *obj, unsigned int controlQubit,
                  unsigned int targetQubit, double theta) {
    if (LibraryHandle)
      return fTNApplyCRz(obj, controlQubit, targetQubit, theta) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply crz gate on tensor network");

    return false;
  }

  bool TNApplyCU(void *obj, unsigned int controlQubit, unsigned int targetQubit,
                 double theta, double phi, double lambda, double gamma) {
    if (LibraryHandle)
      return fTNApplyCU(obj, controlQubit, targetQubit, theta, phi, lambda,
                        gamma) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply cu gate on tensor network");

    return false;
  }

  bool TNApplyCCX(void *obj, unsigned int controlQubit1,
                  unsigned int controlQubit2, unsigned int targetQubit) {
    if (LibraryHandle)
      return fTNApplyCCX(obj, controlQubit1, controlQubit2, targetQubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply ccx gate on tensor network");
    return false;
  }

  bool TNApplyCSwap(void *obj, unsigned int controlQubit,
                    unsigned int targetQubit1, unsigned int targetQubit2) {
    if (LibraryHandle)
      return fTNApplyCSwap(obj, controlQubit, targetQubit1, targetQubit2) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to apply cswap gate on tensor network");
    return false;
  }

  // stabilizer functions
  void *CreateStabilizerSimulator(long long int numQubits,
                                  long long int numShots,
                                  long long int numMeasurements,
                                  long long int numDetectors) {
    if (LibraryHandle)
      return fCreateStabilizerSimulator(numQubits, numShots, numMeasurements,
                                        numDetectors);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to create stabilizer simulator");

    return nullptr;
  }

  void DestroyStabilizerSimulator(void *obj) {
    if (!obj) return;
    if (LibraryHandle)
      fDestroyStabilizerSimulator(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to destroy stabilizer simulator");
  }

  bool ExecuteStabilizerCircuit(void *obj, const char *circuitStr,
                                int randomizeMeasurements,
                                unsigned long long int seed) {
    if (!obj) return false;
    if (LibraryHandle)
      return fExecuteStabilizerCircuit(obj, circuitStr, randomizeMeasurements,
                                       seed) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to execute stabilizer circuit");

    return false;
  }

  long long GetStabilizerXZTableSize(void *obj) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fGetStabilizerXZTableSize(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get stabilizer XZ table size");

    return 0;
  }

  long long GetStabilizerMTableSize(void *obj) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fGetStabilizerMTableSize(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get stabilizer M table size");

    return 0;
  }

  long long GetStabilizerTableStrideMajor(void *obj) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fGetStabilizerTableStrideMajor(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get stabilizer table stride major");
    return 0;
  }

  long long GetStabilizerNumQubits(void *obj) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fGetStabilizerNumQubits(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get stabilizer number of qubits");

    return 0;
  }

  long long GetStabilizerNumShots(void *obj) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fGetStabilizerNumShots(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get stabilizer number of shots");

    return 0;
  }

  long long GetStabilizerNumMeasurements(void *obj) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fGetStabilizerNumMeasurements(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get stabilizer number of measurements");

    return 0;
  }

  long long GetStabilizerNumDetectors(void *obj) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fGetStabilizerNumDetectors(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get stabilizer number of detectors");

    return 0;
  }

  int CopyStabilizerXTable(void *obj, unsigned int *xtable) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fCopyStabilizerXTable(obj, xtable);
    else
      throw std::runtime_error("GpuLibrary: Unable to copy stabilizer X table");
    return 0;
  }

  int CopyStabilizerZTable(void *obj, unsigned int *ztable) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fCopyStabilizerZTable(obj, ztable);
    else
      throw std::runtime_error("GpuLibrary: Unable to copy stabilizer Z table");
    return 0;
  }

  int CopyStabilizerMTable(void *obj, unsigned int *mtable) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fCopyStabilizerMTable(obj, mtable);
    else
      throw std::runtime_error("GpuLibrary: Unable to copy stabilizer M table");
    return 0;
  }

  int InitStabilizerXTable(void *obj, const unsigned int *xtable) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fInitStabilizerXTable(obj, xtable);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to initialize stabilizer X table");
    return 0;
  }

  int InitStabilizerZTable(void *obj, const unsigned int *ztable) {
    if (!obj) return 0;
    if (LibraryHandle)
      return fInitStabilizerZTable(obj, ztable);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to initialize stabilizer Z table");
    return 0;
  }

  // pauli propagation functions
  void* CreatePauliPropSimulator(int nrQubits)
  {
    if (LibraryHandle)
      return fCreatePauliPropSimulator(nrQubits);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to create pauli propagation simulator");
    return nullptr;
  }

  void DestroyPauliPropSimulator(void* obj)
  {
    if (!obj) return;
    if (LibraryHandle)
      fDestroyPauliPropSimulator(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to destroy pauli propagation simulator");
  }

  int PauliPropGetNrQubits(void *obj) 
  {
    if (!obj) return 0;
    if (LibraryHandle)
      return fPauliPropGetNrQubits(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get number of qubits in pauli propagation simulator");
    return 0;
  }

  int PauliPropSetWillUseSampling(void* obj, int willUseSampling)
  {
    if (!obj) return 0;
    if (LibraryHandle)
      return fPauliPropSetWillUseSampling(obj, willUseSampling) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set 'will use sampling' in pauli propagation simulator");
    return 0;
  }

  int PauliPropGetWillUseSampling(void* obj)
  {
    if (!obj) return 0;
    if (LibraryHandle)
      return fPauliPropGetWillUseSampling(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get 'will use sampling' in pauli propagation simulator");
    return 0;
  }
    
  double PauliPropGetCoefficientTruncationCutoff(void* obj)
  {
    if (!obj) return 0.0;
    if (LibraryHandle)
      return fPauliPropGetCoefficientTruncationCutoff(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get coefficient truncation cutoff in pauli propagation simulator");
    return 0.0;
  }

  void PauliPropSetCoefficientTruncationCutoff(void* obj, double cutoff)
  {
    if (!obj) return;
    if (LibraryHandle)
      fPauliPropSetCoefficientTruncationCutoff(obj, cutoff);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set coefficient truncation cutoff in pauli "
          "propagation simulator");
  }

  double PauliPropGetWeightTruncationCutoff(void* obj)
  {
    if (!obj) return 0.0;
    if (LibraryHandle)
      return fPauliPropGetWeightTruncationCutoff(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get weight truncation cutoff in pauli propagation simulator");
    return 0.0;
  }

  void PauliPropSetWeightTruncationCutoff(void* obj, double cutoff)
  {
    if (!obj) return;
    if (LibraryHandle)
      fPauliPropSetWeightTruncationCutoff(obj, cutoff);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set weight truncation cutoff in pauli "
          "propagation simulator");
  }

  int PauliPropGetNumGatesBetweenTruncations(void* obj)
  {
    if (!obj) return 0;
    if (LibraryHandle)
      return fPauliPropGetNumGatesBetweenTruncations(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get number of gates between truncations in pauli propagation simulator");
    return 0;
  }

  void PauliPropSetNumGatesBetweenTruncations(void* obj, int numGates)
  {
    if (!obj) return;
    if (LibraryHandle)
      fPauliPropSetNumGatesBetweenTruncations(obj, numGates);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set number of gates between truncations in pauli "
          "propagation simulator");
  }

  int PauliPropGetNumGatesBetweenDeduplications(void* obj)
  {
    if (!obj) return 0;
    if (LibraryHandle)
      return fPauliPropGetNumGatesBetweenDeduplications(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get number of gates between deduplications in pauli propagation simulator");
    return 0;
  }

  void PauliPropSetNumGatesBetweenDeduplications(void* obj, int numGates)
  {
    if (!obj) return;
    if (LibraryHandle)
      fPauliPropSetNumGatesBetweenDeduplications(obj, numGates);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set number of gates between deduplications in pauli "
          "propagation simulator");
  }

  bool PauliPropClearOperators(void* obj)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropClearOperators(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to clear operators in pauli propagation simulator");
    return false;
  }

  bool PauliPropAllocateMemory(void* obj, double percentage)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropAllocateMemory(obj, percentage) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to allocate memory in pauli propagation simulator");
    return false;
  }

  double PauliPropGetExpectationValue(void* obj)
  {
    if (!obj) return 0.0;
    if (LibraryHandle)
      return fPauliPropGetExpectationValue(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get expectation value in pauli propagation simulator");
    return 0.0;
  }

  bool PauliPropExecute(void* obj)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropExecute(obj) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to execute pauli propagation simulator");
    return false;
  }

  bool PauliPropSetInPauliExpansionUnique(void* obj, const char* pauliString)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropSetInPauliExpansionUnique(obj, pauliString) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set unique pauli in pauli propagation simulator");
    return false;
  }

  bool PauliPropSetInPauliExpansionMultiple(void* obj, const char** pauliStrings,
      const double* coefficients,
      int nrPaulis)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropSetInPauliExpansionMultiple(obj, pauliStrings, coefficients, nrPaulis) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to set multiple pauli in pauli propagation simulator");
    return false;
  }

  bool PauliPropApplyX(void* obj, int qubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyX(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply X gate on mps");
    return false;
  }

  bool PauliPropApplyY(void* obj, int qubit) {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyY(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Y gate on mps");
    return false;
  }

  bool PauliPropApplyZ(void *obj, int qubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyZ(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply Z gate on mps");
    return false;
  }

  bool PauliPropApplyH(void *obj, int qubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyH(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply H gate on mps");
    return false;
  }

  bool PauliPropApplyS(void *obj, int qubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyS(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply S gate on mps");
    return false;
  }

  bool PauliPropApplySQRTX(void *obj, int qubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplySQRTX(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply SQRTX gate on mps");
    return false;
  }

  bool PauliPropApplySQRTY(void *obj, int qubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplySQRTY(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply SQRTY gate on mps");
    return false;
  }

  bool PauliPropApplySQRTZ(void *obj, int qubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplySQRTZ(obj, qubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply SQRTZ gate on mps");
    return false;
  }

  bool PauliPropApplyCX(void *obj, int targetQubit, int controlQubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyCX(obj, targetQubit, controlQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CX gate on mps");
    return false;
  }

  bool PauliPropApplyCY(void *obj, int targetQubit, int controlQubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyCY(obj, targetQubit, controlQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CY gate on mps");
    return false;
  }

  bool PauliPropApplyCZ(void *obj, int targetQubit, int controlQubit)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyCZ(obj, targetQubit, controlQubit) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply CZ gate on mps");
    return false;
  }

  bool PauliPropApplySWAP(void *obj, int qubit1, int qubit2)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplySWAP(obj, qubit1, qubit2) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply SWAP gate on mps");
    return false;
  }

  bool PauliPropApplyISWAP(void *obj, int qubit1, int qubit2)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyISWAP(obj, qubit1, qubit2) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply ISWAP gate on mps");
    return false;
  }

  bool PauliPropApplyRX(void *obj, int qubit, double angle)
  {	
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyRX(obj, qubit, angle) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply RX gate on mps");
    return false;
  }

  bool PauliPropApplyRY(void *obj, int qubit, double angle)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyRY(obj, qubit, angle) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply RY gate on mps");
    return false;
  }

  bool PauliPropApplyRZ(void *obj, int qubit, double angle)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropApplyRZ(obj, qubit, angle) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to apply RZ gate on mps");
    return false;
  }

  bool PauliPropAddNoiseX(void *obj, int qubit, double probability)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropAddNoiseX(obj, qubit, probability) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to add X noise on mps");
    return false;
  }

  bool PauliPropAddNoiseY(void *obj, int qubit, double probability)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropAddNoiseY(obj, qubit, probability) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to add Y noise on mps");
    return false;
  }

  bool PauliPropAddNoiseZ(void* obj, int qubit, double probability)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropAddNoiseZ(obj, qubit, probability) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to add Z noise on mps");
    return false;
  }

  bool PauliPropAddNoiseXYZ(void* obj, int qubit, double probabilityX,
      double probabilityY, double probabilityZ)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropAddNoiseXYZ(obj, qubit, probabilityX,
          probabilityY, probabilityZ) == 1;
    else
      throw std::runtime_error("GpuLibrary: Unable to add XYZ noise on mps");
    return false;
  }

  bool PauliPropAddAmplitudeDamping(void* obj, int qubit, double dampingProb, double exciteProb)
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropAddAmplitudeDamping(obj, qubit, dampingProb, exciteProb) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to add amplitude damping on mps");
    return false;
  }

  double PauliPropQubitProbability0(void *obj, int qubit)
  {
    if (!obj) return 0.0;
    if (LibraryHandle)
      return fPauliPropQubitProbability0(obj, qubit);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get qubit probability 0 in pauli propagation simulator");
    return 0.0;
  }

  double PauliPropProbability(void* obj, unsigned long long int outcome)
  {
    if (!obj) return 0.0;
    if (LibraryHandle)
      return fPauliPropProbability(obj, outcome);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to get probability of outcome in pauli propagation simulator");
    return 0.0;
  }

  bool PauliPropMeasureQubit(void* obj, int qubit) 
  {
    if (!obj) return false;
    if (LibraryHandle)
      return fPauliPropMeasureQubit(obj, qubit) == 1;
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to measure qubit in pauli propagation simulator");
    return false;
  }

  unsigned char* PauliPropSampleQubits(void* obj, const int* qubits,
      int nrQubits)
  {
    if (!obj) return nullptr;
    if (LibraryHandle)
      return fPauliPropSampleQubits(obj, qubits, nrQubits);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to sample qubits in pauli propagation simulator");
    return nullptr;
  }

  void PauliPropFreeSampledQubits(unsigned char* samples)
  {
    if (!samples) return;
    if (LibraryHandle)
      fPauliPropFreeSampledQubits(samples);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to free sampled qubits in pauli propagation "
          "simulator");
  }

  void PauliPropSaveState(void *obj)
  {
    if (!obj) return;
    if (LibraryHandle)
      fPauliPropSaveState(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to save state in pauli propagation simulator");
  }

  void PauliPropRestoreState(void *obj)
  {
    if (!obj) return;
    if (LibraryHandle)
      fPauliPropRestoreState(obj);
    else
      throw std::runtime_error(
          "GpuLibrary: Unable to restore state in pauli propagation simulator");
  }

 private:
  void *LibraryHandle = nullptr;

  void *(*InitLib)();
  void (*FreeLib)();

  void *(*fCreateStateVector)(void *);
  void (*fDestroyStateVector)(void *);

  // statevector functions
  int (*fSetDataType)(void *, int);
  int (*fIsDoublePrecision)(void *);
  int (*fGetNrQubits)(void *);
  int (*fCreate)(void *, unsigned int);
  int (*fReset)(void *);
  int (*fCreateWithState)(void *, unsigned int, const double *);
  int (*fMeasureQubitCollapse)(void *, int);
  int (*fMeasureQubitNoCollapse)(void *, int);
  int (*fMeasureQubitsCollapse)(void *, int *, int *, int);
  int (*fMeasureQubitsNoCollapse)(void *, int *, int *, int);
  unsigned long long (*fMeasureAllQubitsCollapse)(void *);
  unsigned long long (*fMeasureAllQubitsNoCollapse)(void *);

  int (*fSaveState)(void *);
  int (*fSaveStateToHost)(void *);
  int (*fSaveStateDestructive)(void *);
  int (*fRestoreStateFreeSaved)(void *);
  int (*fRestoreStateNoFreeSaved)(void *);
  void (*fFreeSavedState)(void *obj);
  void *(*fClone)(void *);
  int (*fSample)(void *, unsigned int, long int *, unsigned int, int *);
  int (*fSampleAll)(void *, unsigned int, long int *);
  int (*fAmplitude)(void *, long long int, double *, double *);
  double (*fProbability)(void *, int *, int *, int);
  double (*fBasisStateProbability)(void *, long long int);
  int (*fAllProbabilities)(void *, double *);
  double (*fExpectationValue)(void *, const char *, int);

  int (*fApplyX)(void *, int);
  int (*fApplyY)(void *, int);
  int (*fApplyZ)(void *, int);
  int (*fApplyH)(void *, int);
  int (*fApplyS)(void *, int);
  int (*fApplySDG)(void *, int);
  int (*fApplyT)(void *, int);
  int (*fApplyTDG)(void *, int);
  int (*fApplySX)(void *, int);
  int (*fApplySXDG)(void *, int);
  int (*fApplyK)(void *, int);
  int (*fApplyP)(void *, int, double);
  int (*fApplyRx)(void *, int, double);
  int (*fApplyRy)(void *, int, double);
  int (*fApplyRz)(void *, int, double);
  int (*fApplyU)(void *, int, double, double, double, double);
  int (*fApplyCX)(void *, int, int);
  int (*fApplyCY)(void *, int, int);
  int (*fApplyCZ)(void *, int, int);
  int (*fApplyCH)(void *, int, int);
  int (*fApplyCSX)(void *, int, int);
  int (*fApplyCSXDG)(void *, int, int);
  int (*fApplyCP)(void *, int, int, double);
  int (*fApplyCRx)(void *, int, int, double);
  int (*fApplyCRy)(void *, int, int, double);
  int (*fApplyCRz)(void *, int, int, double);
  int (*fApplyCCX)(void *, int, int, int);
  int (*fApplySwap)(void *, int, int);
  int (*fApplyCSwap)(void *, int, int, int);
  int (*fApplyCU)(void *, int, int, double, double, double, double);

  // mps functions
  void *(*fCreateMPS)(void *);
  void (*fDestroyMPS)(void *);

  int (*fMPSCreate)(void *, unsigned int);
  int (*fMPSReset)(void *);

  int (*fMPSIsValid)(void *);
  int (*fMPSIsCreated)(void *);

  int (*fMPSSetDataType)(void *, int);
  int (*fMPSIsDoublePrecision)(void *);
  int (*fMPSSetCutoff)(void *, double);
  double (*fMPSGetCutoff)(void *);
  int (*fMPSSetGesvdJ)(void *, int);
  int (*fMPSGetGesvdJ)(void *);
  int (*fMPSSetMaxExtent)(void *, long int);
  long int (*fMPSGetMaxExtent)(void *);
  int (*fMPSGetNrQubits)(void *);
  int (*fMPSAmplitude)(void *, long int, long int *, double *, double *);
  double (*fMPSProbability0)(void *, unsigned int);
  int (*fMPSMeasure)(void *, unsigned int);
  int (*fMPSMeasureQubits)(void *, long int, unsigned int *, int *);

  void *(*fMPSGetMapForSample)();
  int (*fMPSFreeMapForSample)(void *);
  int (*fMPSSample)(void *, long int, long int, unsigned int *, void *);

  int (*fMPSSaveState)(void *);
  int (*fMPSRestoreState)(void *);
  int (*fMPSCleanSavedState)(void *);
  void *(*fMPSClone)(void *);

  double (*fMPSExpectationValue)(void *, const char *, int);

  int (*fMPSApplyX)(void *, unsigned int);
  int (*fMPSApplyY)(void *, unsigned int);
  int (*fMPSApplyZ)(void *, unsigned int);
  int (*fMPSApplyH)(void *, unsigned int);
  int (*fMPSApplyS)(void *, unsigned int);
  int (*fMPSApplySDG)(void *, unsigned int);
  int (*fMPSApplyT)(void *, unsigned int);
  int (*fMPSApplyTDG)(void *, unsigned int);
  int (*fMPSApplySX)(void *, unsigned int);
  int (*fMPSApplySXDG)(void *, unsigned int);
  int (*fMPSApplyK)(void *, unsigned int);
  int (*fMPSApplyP)(void *, unsigned int, double);
  int (*fMPSApplyRx)(void *, unsigned int, double);
  int (*fMPSApplyRy)(void *, unsigned int, double);
  int (*fMPSApplyRz)(void *, unsigned int, double);
  int (*fMPSApplyU)(void *, unsigned int, double, double, double, double);
  int (*fMPSApplySwap)(void *, unsigned int, unsigned int);
  int (*fMPSApplyCX)(void *, unsigned int, unsigned int);
  int (*fMPSApplyCY)(void *, unsigned int, unsigned int);
  int (*fMPSApplyCZ)(void *, unsigned int, unsigned int);
  int (*fMPSApplyCH)(void *, unsigned int, unsigned int);
  int (*fMPSApplyCSX)(void *, unsigned int, unsigned int);
  int (*fMPSApplyCSXDG)(void *, unsigned int, unsigned int);
  int (*fMPSApplyCP)(void *, unsigned int, unsigned int, double);
  int (*fMPSApplyCRx)(void *, unsigned int, unsigned int, double);
  int (*fMPSApplyCRy)(void *, unsigned int, unsigned int, double);
  int (*fMPSApplyCRz)(void *, unsigned int, unsigned int, double);
  int (*fMPSApplyCU)(void *, unsigned int, unsigned int, double, double, double,
                     double);

  // tensor network functions
  void *(*fCreateTensorNet)(void *);
  void (*fDestroyTensorNet)(void *);

  int (*fTNCreate)(void *, unsigned int);
  int (*fTNReset)(void *);

  int (*fTNIsValid)(void *);
  int (*fTNIsCreated)(void *);

  int (*fTNSetDataType)(void *, int);
  int (*fTNIsDoublePrecision)(void *);
  int (*fTNSetCutoff)(void *, double);
  double (*fTNGetCutoff)(void *);
  int (*fTNSetGesvdJ)(void *, int);
  int (*fTNGetGesvdJ)(void *);
  int (*fTNSetMaxExtent)(void *, long int);
  long int (*fTNGetMaxExtent)(void *);
  int (*fTNGetNrQubits)(void *);
  int (*fTNAmplitude)(void *, long int, long int *, double *, double *);
  double (*fTNProbability0)(void *, unsigned int);
  int (*fTNMeasure)(void *, unsigned int);
  int (*fTNMeasureQubits)(void *, long int, unsigned int *, int *);

  void *(*fTNGetMapForSample)();
  int (*fTNFreeMapForSample)(void *);
  int (*fTNSample)(void *, long int, long int, unsigned int *, void *);

  int (*fTNSaveState)(void *);
  int (*fTNRestoreState)(void *);
  int (*fTNCleanSavedState)(void *);
  void *(*fTNClone)(void *);

  double (*fTNExpectationValue)(void *, const char *, int);

  int (*fTNApplyX)(void *, unsigned int);
  int (*fTNApplyY)(void *, unsigned int);
  int (*fTNApplyZ)(void *, unsigned int);
  int (*fTNApplyH)(void *, unsigned int);
  int (*fTNApplyS)(void *, unsigned int);
  int (*fTNApplySDG)(void *, unsigned int);
  int (*fTNApplyT)(void *, unsigned int);
  int (*fTNApplyTDG)(void *, unsigned int);
  int (*fTNApplySX)(void *, unsigned int);
  int (*fTNApplySXDG)(void *, unsigned int);
  int (*fTNApplyK)(void *, unsigned int);
  int (*fTNApplyP)(void *, unsigned int, double);
  int (*fTNApplyRx)(void *, unsigned int, double);
  int (*fTNApplyRy)(void *, unsigned int, double);
  int (*fTNApplyRz)(void *, unsigned int, double);
  int (*fTNApplyU)(void *, unsigned int, double, double, double, double);
  int (*fTNApplySwap)(void *, unsigned int, unsigned int);
  int (*fTNApplyCX)(void *, unsigned int, unsigned int);
  int (*fTNApplyCY)(void *, unsigned int, unsigned int);
  int (*fTNApplyCZ)(void *, unsigned int, unsigned int);
  int (*fTNApplyCH)(void *, unsigned int, unsigned int);
  int (*fTNApplyCSX)(void *, unsigned int, unsigned int);
  int (*fTNApplyCSXDG)(void *, unsigned int, unsigned int);
  int (*fTNApplyCP)(void *, unsigned int, unsigned int, double);
  int (*fTNApplyCRx)(void *, unsigned int, unsigned int, double);
  int (*fTNApplyCRy)(void *, unsigned int, unsigned int, double);
  int (*fTNApplyCRz)(void *, unsigned int, unsigned int, double);
  int (*fTNApplyCU)(void *, unsigned int, unsigned int, double, double, double,
                    double);
  int (*fTNApplyCCX)(void *, unsigned int, unsigned int, unsigned int);
  int (*fTNApplyCSwap)(void *, unsigned int, unsigned int, unsigned int);

  // stabilizer functions
  void *(*fCreateStabilizerSimulator)(long long int, long long int,
                                      long long int, long long int);
  void (*fDestroyStabilizerSimulator)(void *);
  int (*fExecuteStabilizerCircuit)(void *, const char *, int,
                                   unsigned long long int);
  long long (*fGetStabilizerXZTableSize)(void *);
  long long (*fGetStabilizerMTableSize)(void *);
  long long (*fGetStabilizerTableStrideMajor)(void *);
  long long (*fGetStabilizerNumQubits)(void *);
  long long (*fGetStabilizerNumShots)(void *);
  long long (*fGetStabilizerNumMeasurements)(void *);
  long long (*fGetStabilizerNumDetectors)(void *);
  int (*fCopyStabilizerXTable)(void *, unsigned int *);
  int (*fCopyStabilizerZTable)(void *, unsigned int *);
  int (*fCopyStabilizerMTable)(void *, unsigned int *);
  int (*fInitStabilizerXTable)(void *, const unsigned int *);
  int (*fInitStabilizerZTable)(void *, const unsigned int *);

  // Pauli propagation functions
  void *(*fCreatePauliPropSimulator)(int);
  void (*fDestroyPauliPropSimulator)(void *);

  int (*fPauliPropGetNrQubits)(void *);
  int (*fPauliPropSetWillUseSampling)(void *, int);
  int (*fPauliPropGetWillUseSampling)(void *);
  double (*fPauliPropGetCoefficientTruncationCutoff)(void *);
  void (*fPauliPropSetCoefficientTruncationCutoff)(void *, double);
  double (*fPauliPropGetWeightTruncationCutoff)(void *);
  void (*fPauliPropSetWeightTruncationCutoff)(void *, double);
  int (*fPauliPropGetNumGatesBetweenTruncations)(void *);
  void (*fPauliPropSetNumGatesBetweenTruncations)(void *, int);
  int (*fPauliPropGetNumGatesBetweenDeduplications)(void *);
  void (*fPauliPropSetNumGatesBetweenDeduplications)(void *, int);
  int (*fPauliPropClearOperators)(void *);
  int (*fPauliPropAllocateMemory)(void *, double);
  double (*fPauliPropGetExpectationValue)(void *);
  int (*fPauliPropExecute)(void *);
  int (*fPauliPropSetInPauliExpansionUnique)(void *, const char *);
  int (*fPauliPropSetInPauliExpansionMultiple)(void *, const char **,
                                           const double *, int);

  int (*fPauliPropApplyX)(void *, int);
  int (*fPauliPropApplyY)(void *, int);
  int (*fPauliPropApplyZ)(void *, int);
  int (*fPauliPropApplyH)(void *, int);
  int (*fPauliPropApplyS)(void *, int);
  int (*fPauliPropApplySQRTX)(void *, int);
  int (*fPauliPropApplySQRTY)(void *, int);
  int (*fPauliPropApplySQRTZ)(void *, int);
  int (*fPauliPropApplyCX)(void *, int, int);
  int (*fPauliPropApplyCY)(void *, int, int);
  int (*fPauliPropApplyCZ)(void *, int, int);
  int (*fPauliPropApplySWAP)(void *, int, int);
  int (*fPauliPropApplyISWAP)(void *, int, int);
  int (*fPauliPropApplyRX)(void *, int, double);
  int (*fPauliPropApplyRY)(void *, int, double);
  int (*fPauliPropApplyRZ)(void *, int, double);

  int (*fPauliPropAddNoiseX)(void *, int, double);
  int (*fPauliPropAddNoiseY)(void *, int, double);
  int (*fPauliPropAddNoiseZ)(void *, int, double);
  int (*fPauliPropAddNoiseXYZ)(void *, int, double, double, double);
  int (*fPauliPropAddAmplitudeDamping)(void *, int, double, double);
  double (*fPauliPropQubitProbability0)(void *, int);
  double (*fPauliPropProbability)(void *, unsigned long long int);

  int (*fPauliPropMeasureQubit)(void *, int);
  unsigned char *(*fPauliPropSampleQubits)(void *, const int *, int);
  void (*fPauliPropFreeSampledQubits)(unsigned char *);
  void (*fPauliPropSaveState)(void *);
  void (*fPauliPropRestoreState)(void *);
};
}  // namespace Simulators

#endif
#endif
