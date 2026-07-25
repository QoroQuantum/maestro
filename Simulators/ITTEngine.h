/**
 * @file ITTEngine.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Abstract base class for Tensor Train Engine simulators.
 * Provides a polymorphic interface so both CPU and GPU backends
 * can be used interchangeably through the Python bindings.
 */

#pragma once

#ifndef _I_TT_ENGINE_H_
#define _I_TT_ENGINE_H_

#include <string>
#include <vector>

namespace Simulators {

class ITTEngine {
 public:
  virtual ~ITTEngine() = default;

  virtual bool IsValid() const = 0;
  virtual bool IsCreated() const = 0;
  
  virtual bool Create(const std::vector<int> &physExtents) = 0;
  virtual bool Reset() = 0;
  
  virtual bool SetMaxRank(int maxRank) = 0;
  virtual int GetMaxRank() const = 0;
  
  virtual bool SetCutoff(double cutoff) = 0;
  virtual double GetCutoff() const = 0;
  
  virtual int GetDimension() const = 0;
  
  virtual bool SetCore(int site, const std::vector<double> &data,
                       int leftRank, int physExtent, int rightRank) = 0;
  virtual std::vector<double> GetCore(int site) const = 0;
  virtual std::vector<int> GetCoreShape(int site) const = 0;
  
  virtual std::vector<int> GetRanks() const = 0;
  virtual long long GetTotalElements() const = 0;
  
  virtual bool Truncate(double cutoff, int maxRank) = 0;
  
  virtual double Evaluate(const std::vector<int> &indices) const = 0;
  virtual std::vector<double> EvaluateBatch(const std::vector<int> &indices,
                                            int numPoints) const = 0;
                                            
  virtual bool Save(const std::string &filepath) const = 0;
  virtual bool Load(const std::string &filepath) = 0;
};

}  // namespace Simulators

#endif  // _I_TT_ENGINE_H_
