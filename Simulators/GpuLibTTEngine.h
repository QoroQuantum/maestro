/**
 * @file GpuLibTTEngine.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The GPU TT Engine library class.
 *
 * Wrapper around TT Engine C API functions from gpusim.h.
 * Follows the same pattern as GpuLibMPSSim.h.
 */

#pragma once

#ifndef _GPU_LIB_TT_ENGINE_H_
#define _GPU_LIB_TT_ENGINE_H_

#ifdef __linux__

#include <memory>
#include <vector>

#include "GpuLibrary.h"

namespace Simulators {

class GpuLibTTEngine {
 public:
  explicit GpuLibTTEngine(const std::shared_ptr<GpuLibrary> &lib) : lib(lib) {
    if (lib)
      obj = lib->CreateTTEngine();
    else
      obj = nullptr;
  }

  GpuLibTTEngine() = delete;
  GpuLibTTEngine(const GpuLibTTEngine &) = delete;
  GpuLibTTEngine &operator=(const GpuLibTTEngine &) = delete;
  GpuLibTTEngine(GpuLibTTEngine &&) = default;
  GpuLibTTEngine &operator=(GpuLibTTEngine &&) = default;

  ~GpuLibTTEngine() {
    if (lib && obj) lib->DestroyTTEngine(obj);
  }

  bool IsValid() const {
    if (obj) return lib->TTIsValid(obj);
    return false;
  }

  bool IsCreated() const {
    if (obj) return lib->TTIsCreated(obj);
    return false;
  }

  bool Create(const std::vector<int> &physExtents) {
    if (obj)
      return lib->TTCreate(obj, static_cast<int>(physExtents.size()),
                           physExtents.data());
    return false;
  }

  bool Reset() {
    if (obj) return lib->TTReset(obj);
    return false;
  }

  bool SetMaxRank(int maxRank) {
    if (obj) return lib->TTSetMaxRank(obj, maxRank);
    return false;
  }

  int GetMaxRank() const {
    if (obj) return lib->TTGetMaxRank(obj);
    return 0;
  }

  bool SetCutoff(double cutoff) {
    if (obj) return lib->TTSetCutoff(obj, cutoff);
    return false;
  }

  double GetCutoff() const {
    if (obj) return lib->TTGetCutoff(obj);
    return 0.0;
  }

  int GetDimension() const {
    if (obj) return lib->TTGetDimension(obj);
    return 0;
  }

  bool SetCore(int site, const std::vector<double> &data,
               int leftRank, int physExtent, int rightRank) {
    if (obj)
      return lib->TTSetCore(obj, site, data.data(),
                            leftRank, physExtent, rightRank);
    return false;
  }

  std::vector<double> GetCore(int site) const {
    if (!obj) return {};
    int lR, pE, rR;
    if (!lib->TTGetCoreShape(obj, site, &lR, &pE, &rR))
      return {};
    std::vector<double> data(lR * pE * rR);
    if (!lib->TTGetCore(obj, site, data.data()))
      return {};
    return data;
  }

  std::vector<int> GetCoreShape(int site) const {
    if (!obj) return {};
    int lR, pE, rR;
    if (!lib->TTGetCoreShape(obj, site, &lR, &pE, &rR))
      return {};
    return {lR, pE, rR};
  }

  std::vector<int> GetRanks() const {
    if (!obj) return {};
    int d = lib->TTGetDimension(obj);
    std::vector<int> ranks(d + 1);
    if (!lib->TTGetRanks(obj, ranks.data()))
      return {};
    return ranks;
  }

  long long GetTotalElements() const {
    if (obj) return lib->TTGetTotalElements(obj);
    return 0;
  }

  bool Truncate(double cutoff, int maxRank) {
    if (obj) return lib->TTTruncate(obj, cutoff, maxRank);
    return false;
  }

  double Evaluate(const std::vector<int> &indices) const {
    if (obj) return lib->TTEvaluate(obj, indices.data());
    return 0.0;
  }

  std::vector<double> EvaluateBatch(const std::vector<int> &indices,
                                     int numPoints) const {
    if (!obj) return {};
    std::vector<double> results(numPoints);
    if (!lib->TTEvaluateBatch(obj, indices.data(), numPoints, results.data()))
      return {};
    return results;
  }

  bool Save(const std::string &filepath) const {
    if (obj) return lib->TTSave(obj, filepath.c_str());
    return false;
  }

  bool Load(const std::string &filepath) {
    if (obj) return lib->TTLoad(obj, filepath.c_str());
    return false;
  }

 private:
  std::shared_ptr<GpuLibrary> lib;
  void *obj;
};

}  // namespace Simulators

#endif  // __linux__

#endif  // _GPU_LIB_TT_ENGINE_H_
