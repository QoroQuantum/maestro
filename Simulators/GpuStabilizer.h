/**
 * @file GpuStabilizer.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The Gpu Stabilizer library class.
 *
 * Just a wrapped aroung c api functions.
 */

#pragma once

#ifndef _GPU_STABILIZER_H
#define _GPU_STABILIZER_H 1

#ifdef __linux__

#include "GpuLibrary.h"

namespace Simulators {

class GpuStabilizer {
 public:
  explicit GpuStabilizer(const std::shared_ptr<GpuLibrary> &lib)
      : lib(lib), obj(nullptr) 
  {
  }

  GpuStabilizer() = delete;
  GpuStabilizer(const GpuStabilizer &) = delete;
  GpuStabilizer &operator=(const GpuStabilizer &) = delete;
  GpuStabilizer(GpuStabilizer &&) = default;
  GpuStabilizer &operator=(GpuStabilizer &&) = default;

  ~GpuStabilizer()
  {
    if (lib && obj) lib->DestroyStabilizerSimulator(obj);
  }

  bool CreateSimulator(long long int numQubits,
                                 long long int numShots,
                                 long long int numMeasurements,
                                 long long int numDetectors) {
    if (lib) {
      obj = lib->CreateStabilizerSimulator(numQubits, numShots, numMeasurements,
                                           numDetectors);
      return obj != nullptr;
    }

    return false;
  }

  bool ExecuteCircuit(const std::string& circuitStr,
                                int randomizeMeasurements,
                                unsigned long long int seed) {
    if (obj)
      return lib->ExecuteStabilizerCircuit(obj, circuitStr,
                                          randomizeMeasurements, seed);
    return false;
  }

  bool Clear()
  {
    if (obj)
    {
      lib->DestroyStabilizerSimulator(obj);
      obj = nullptr;
      return true;
    }

    return false;
  }

  long long GetNumQubits() 
  {
    if (obj)
      return lib->GetStabilizerNumQubits(obj);
    return 0;
  }

  long long GetNumShots() 
  {
    if (obj)
      return lib->GetStabilizerNumShots(obj);
    return 0;
  }

  long long GetNumMeasurements() 
  {
    if (obj)
      return lib->GetStabilizerNumMeasurements(obj);
    return 0;
  }

  long long GetNumDetectors() 
  {
    if (obj)
      return lib->GetStabilizerNumDetectors(obj);
    return 0;
  }

  bool IsCreated() const
  {
    return obj != nullptr;
  }

  std::vector<std::vector<bool>> GetXTable() {
    if (!obj) return std::vector<std::vector<bool>>();

    std::vector<unsigned int> xTableRaw(GetStabilizerXZTableSize());
    lib->CopyStabilizerXTable(obj, xTableRaw.data());

    return ConvertToBoolVectorVector(xTableRaw, GetStabilizerNumQubits(), GetStabilizerNumShots());
  }

  std::vector<std::vector<bool>> GetZTable()
  {
    if (!obj) return std::vector<std::vector<bool>>();

    std::vector<unsigned int> zTableRaw(GetStabilizerXZTableSize());

    std::vector<unsigned int> zTableRaw(GetStabilizerXZTableSize());
    lib->CopyStabilizerZTable(obj, zTableRaw.data());

    return ConvertToBoolVectorVector(zTableRaw, GetStabilizerNumQubits(), GetStabilizerNumShots());
  }

  std::vector<std::vector<bool>> GetMTable() 
  {
    if (!obj) return std::vector<std::vector<bool>>();

    std::vector<unsigned int> mTableRaw(GetStabilizerMTableSize());
    lib->CopyStabilizerMTable(obj, mTableRaw.data());

    return ConvertToBoolVectorVector(mTableRaw, GetStabilizerNumMeasurements(), GetStabilizerNumShots());
  }

 protected:
  std::vector<std::vector<bool>> ConvertToBoolVectorVector(
      const std::vector<unsigned int> &tableRaw, long long int num,
      long long int shots) {
    const long long words_per_qubitormeas = (shots + 31) / 32;

    std::vector<std::vector<bool>> result(num);

    for (long long qubitormeas = 0; qubitormeas < num; ++qubitormeas) {
      for (long long shot = 0; shot < shots; ++shot) {
        const unsigned int word = tableRaw[qubitormeas * words_per_qubitormeas + (shot / 32)];
        const bool bit = ((word >> (shot % 32)) & 1) == 1;
        result[qubitormeas].push_back(bit);
      }
    }

    return result;
  }

  long long GetStabilizerXZTableSize() {
    if (obj) return lib->GetStabilizerXZTableSize(obj);
    return 0;
  }

  long long GetStabilizerMTableSize() {
    if (obj) return lib->GetStabilizerMTableSize(obj);
    return 0;
  }

  long long GetStabilizerTableStrideMajor() {
    if (obj) return lib->GetStabilizerTableStrideMajor(obj);
    return 0;
  }

 private:
  std::shared_ptr<GpuLibrary> lib;
  void *obj;
};

}  // namespace Simulators

#endif
#endif

