/**
 * @file GpuLibTNSim.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The Gpu tensor network library class.
 *
 * Just a wrapped aroung c api functions, should not be used directly but with a
 * adapter/bridge pattern to expose the same interface as the other ones.
 */

#pragma once

#ifndef _GPU_LIB_TN_SIM_H_
#define _GPU_LIB_TN_SIM_H_

#ifdef __linux__

#include <memory>

#include "GpuLibrary.h"

namespace Simulators {

class GpuLibTNSim {
 public:
  explicit GpuLibTNSim(const std::shared_ptr<GpuLibrary> &lib) : lib(lib) {
    if (lib)
      obj = lib->CreateTensorNet();
    else
      obj = nullptr;
  }

  GpuLibTNSim(const std::shared_ptr<GpuLibrary> &lib, void *obj)
      : lib(lib), obj(obj) {}

  GpuLibTNSim() = delete;
  GpuLibTNSim(const GpuLibTNSim &) = delete;
  GpuLibTNSim &operator=(const GpuLibTNSim &) = delete;
  GpuLibTNSim(GpuLibTNSim &&) = default;
  GpuLibTNSim &operator=(GpuLibTNSim &&) = default;

  ~GpuLibTNSim() {
    if (lib && obj) lib->DestroyTensorNet(obj);
  }

  bool Create(unsigned int nrQubits) {
    if (obj) return lib->TNCreate(obj, nrQubits);

    return false;
  }

  bool Reset() {
    if (obj) return lib->TNReset(obj);

    return false;
  }

  bool IsValid() const {
    if (obj) return lib->TNIsValid(obj);

    return false;
  }

  bool IsCreated() const {
    if (obj) return lib->TNIsCreated(obj);

    return false;
  }

  bool SetDataType(int useDoublePrecision) {
    if (obj) return lib->TNSetDataType(obj, useDoublePrecision);

    return false;
  }

  bool IsDoublePrecision() const {
    if (obj) return lib->TNIsDoublePrecision(obj);

    return false;
  }

  bool SetCutoff(double val) {
    if (obj) return lib->TNSetCutoff(obj, val);

    return false;
  }

  double GetCutoff() const {
    if (obj) return lib->TNGetCutoff(obj);

    return 0.;
  }

  bool SetGesvdJ(int val) {
    if (obj) return lib->TNSetGesvdJ(obj, val);

    return false;
  }

  bool GetGesvdJ() const {
    if (obj) return lib->TNGetGesvdJ(obj);

    return false;
  }

  bool SetMaxExtent(long int val) {
    if (obj) return lib->TNSetMaxExtent(obj, val);

    return false;
  }

  long int GetMaxExtent() const {
    if (obj) return lib->TNGetMaxExtent(obj);

    return 0;
  }

  int GetNrQubits() const {
    if (obj) return lib->TNGetNrQubits(obj);

    return 0;
  }

  bool Amplitude(long int numFixedValues, long int *fixedValues, double *real,
                 double *imaginary) const {
    if (obj)
      return lib->TNAmplitude(obj, numFixedValues, fixedValues, real,
                              imaginary);

    return false;
  }

  double Probability0(unsigned int qubit) const {
    if (obj) return lib->TNProbability0(obj, qubit);

    return 0.;
  }

  bool Measure(unsigned int qubit) {
    if (obj) return lib->TNMeasure(obj, qubit);

    return 0;
  }

  bool MeasureQubits(long int numQubits, unsigned int *qubits, int *result) {
    if (obj) return lib->TNMeasureQubits(obj, numQubits, qubits, result);

    return false;
  }

  std::unordered_map<std::vector<bool>, int64_t> *GetMapForSample() const {
    if (lib) return lib->TNGetMapForSample();

    return nullptr;
  }

  bool FreeMapForSample(
      std::unordered_map<std::vector<bool>, int64_t> *map) const {
    if (lib) return lib->TNFreeMapForSample(map);

    return false;
  }

  bool Sample(long int numShots, long int numQubits, unsigned int *qubits,
              void *resultMap) {
    if (obj) return lib->TNSample(obj, numShots, numQubits, qubits, resultMap);

    return false;
  }

  bool SaveState() {
    if (obj) return lib->TNSaveState(obj);

    return false;
  }

  bool RestoreState() {
    if (obj) return lib->TNRestoreState(obj);

    return false;
  }

  bool CleanSavedState() {
    if (obj) return lib->TNCleanSavedState(obj);

    return false;
  }

  std::unique_ptr<GpuLibTNSim> Clone() const {
    // if (obj) return std::make_unique<GpuLibTNSim>(lib, lib->TNClone(obj));

    return nullptr;
  }

  double ExpectationValue(const std::string &pauliString) const {
    if (obj)
      return lib->TNExpectationValue(obj, pauliString.c_str(),
                                     pauliString.length());

    return 0.0;
  }

  bool ApplyX(unsigned int siteA) {
    if (obj) return lib->TNApplyX(obj, siteA);

    return false;
  }

  bool ApplyY(unsigned int siteA) {
    if (obj) return lib->TNApplyY(obj, siteA);

    return false;
  }

  bool ApplyZ(unsigned int siteA) {
    if (obj) return lib->TNApplyZ(obj, siteA);

    return false;
  }

  bool ApplyH(unsigned int siteA) {
    if (obj) return lib->TNApplyH(obj, siteA);

    return false;
  }

  bool ApplyS(unsigned int siteA) {
    if (obj) return lib->TNApplyS(obj, siteA);

    return false;
  }

  bool ApplySDG(unsigned int siteA) {
    if (obj) return lib->TNApplySDG(obj, siteA);

    return false;
  }

  bool ApplyT(unsigned int siteA) {
    if (obj) return lib->TNApplyT(obj, siteA);

    return false;
  }

  bool ApplyTDG(unsigned int siteA) {
    if (obj) return lib->TNApplyTDG(obj, siteA);

    return false;
  }

  bool ApplySX(unsigned int siteA) {
    if (obj) return lib->TNApplySX(obj, siteA);

    return false;
  }

  bool ApplySXDG(unsigned int siteA) {
    if (obj) return lib->TNApplySXDG(obj, siteA);

    return false;
  }

  bool ApplyK(unsigned int siteA) {
    if (obj) return lib->TNApplyK(obj, siteA);

    return false;
  }

  bool ApplyP(unsigned int siteA, double theta) {
    if (obj) return lib->TNApplyP(obj, siteA, theta);

    return false;
  }

  bool ApplyRx(unsigned int siteA, double theta) {
    if (obj) return lib->TNApplyRx(obj, siteA, theta);

    return false;
  }

  bool ApplyRy(unsigned int siteA, double theta) {
    if (obj) return lib->TNApplyRy(obj, siteA, theta);

    return false;
  }

  bool ApplyRz(unsigned int siteA, double theta) {
    if (obj) return lib->TNApplyRz(obj, siteA, theta);

    return false;
  }

  bool ApplyU(unsigned int siteA, double theta, double phi, double lambda,
              double gamma) {
    if (obj) return lib->TNApplyU(obj, siteA, theta, phi, lambda, gamma);

    return false;
  }

  bool ApplySwap(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->TNApplySwap(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCX(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->TNApplyCX(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCY(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->TNApplyCY(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCZ(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->TNApplyCZ(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCH(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->TNApplyCH(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCSX(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->TNApplyCSX(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCSXDG(unsigned int controlQubit, unsigned int targetQubit) {
    if (obj) return lib->TNApplyCSXDG(obj, controlQubit, targetQubit);

    return false;
  }

  bool ApplyCP(unsigned int controlQubit, unsigned int targetQubit,
               double theta) {
    if (obj) return lib->TNApplyCP(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCRx(unsigned int controlQubit, unsigned int targetQubit,
                double theta) {
    if (obj) return lib->TNApplyCRx(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCRy(unsigned int controlQubit, unsigned int targetQubit,
                double theta) {
    if (obj) return lib->TNApplyCRy(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCRz(unsigned int controlQubit, unsigned int targetQubit,
                double theta) {
    if (obj) return lib->TNApplyCRz(obj, controlQubit, targetQubit, theta);

    return false;
  }

  bool ApplyCU(unsigned int controlQubit, unsigned int targetQubit,
               double theta, double phi, double lambda, double gamma) {
    if (obj)
      return lib->TNApplyCU(obj, controlQubit, targetQubit, theta, phi, lambda,
                            gamma);

    return false;
  }

  bool ApplyCCX(unsigned int controlQubit1, unsigned int controlQubit2,
                unsigned int targetQubit) {
    if (obj)
      return lib->TNApplyCCX(obj, controlQubit1, controlQubit2, targetQubit);
    return false;
  }

  bool ApplyCSwap(unsigned int controlQubit, unsigned int qubit1,
                  unsigned int qubit2) {
    if (obj) return lib->TNApplyCSwap(obj, controlQubit, qubit1, qubit2);
    return false;
  }

 private:
  std::shared_ptr<GpuLibrary> lib;
  void *obj;
};

}  // namespace Simulators

#endif  // __linux__

#endif  // _GPU_LIB_TN_SIM_H_
#pragma once
