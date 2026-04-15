/**
 * @file Alias.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Alias sampling for O(1) sampling with a O(N) preprocessing step.
 */

#pragma once

#ifndef _ALIAS_H_
#define _ALIAS_H_

#include <complex>
#include <vector>

#include <Eigen/Eigen>

#include "PathIntegral.h"

namespace Utils {

class Alias {
 public:
  Alias() = delete;

  template <class T = Eigen::VectorXcd>
  Alias(const T &statevector) {
    std::vector<double> probabilities(statevector.size());

    double accum = 0.;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(statevector.size());
         ++i) {
      probabilities[i] = std::norm(statevector[i]);
      accum += probabilities[i];
      if (accum > 1. - std::numeric_limits<double>::epsilon()) {
        probabilities.resize(i + 1);
        break;
      }
    }

    SetAliasTable(probabilities);
  }

  Alias(const std::unordered_map<QC::PathIntegral::FastVectorBool, std::complex<double>,
        QC::PathIntegral::FastVectorBoolHash> &amplitudesMap) {
    std::vector<AliasEntry> under;
    std::vector<AliasEntry> over;

    // this uses four times the memory of the sampling with the binary search
    under.reserve(amplitudesMap.size());
    over.reserve(amplitudesMap.size());

    size_t maxState = 0;
    for (const auto &valPair : amplitudesMap) {
      const double prob = std::norm(valPair.second) * amplitudesMap.size();
      const size_t state = valPair.first.getWords()[0];
      if (prob < 1.)
        under.emplace_back(prob, state);
      else
        over.emplace_back(prob, state);

      if (state > maxState) maxState = state;
    }

    aliasTable.resize(maxState + 1);
    std::fill(aliasTable.begin(), aliasTable.end(), AliasEntry(1., -1));

    SetAliasTable(under, over);
  }

  size_t Sample(double v) const {
    static const double oneMinusEps =
        1. - std::numeric_limits<double>::epsilon();

    const double vadj = v * aliasTable.size();
    const size_t offset = std::min<size_t>(static_cast<size_t>(vadj), aliasTable.size() - 1);
    const double up = std::min<double>(vadj - offset, oneMinusEps);

    return up < aliasTable[offset].probability ? offset
                                               : aliasTable[offset].alias;
  }

 private:
  class AliasEntry {
   public:
    AliasEntry() : probability(1.), alias(-1) {}
    AliasEntry(double prob, long long int ali)
        : probability(prob), alias(ali) {}

    double probability;
    long long int alias;
  };

  void SetAliasTable(std::vector<double>& probabilities)
  {
    aliasTable.resize(probabilities.size());

    std::vector<AliasEntry> under;
    std::vector<AliasEntry> over;

    // this uses four times the memory of the sampling with the binary search
    under.reserve(probabilities.size());
    over.reserve(probabilities.size());

    for (Eigen::Index i = 0;
         i < static_cast<Eigen::Index>(probabilities.size()); ++i) {
      const double prob = probabilities[i] * probabilities.size();
      if (prob < 1.)
        under.emplace_back(prob, i);
      else
        over.emplace_back(prob, i);
    }

    SetAliasTable(under, over);
  }

  void SetAliasTable(std::vector<AliasEntry>& under,
                     std::vector<AliasEntry>& over)
  {
    while (!under.empty() && !over.empty()) {
      const AliasEntry &u = under.back();
      const AliasEntry &o = over.back();

      const size_t index = o.alias;

      aliasTable[u.alias] = AliasEntry(u.probability, index);

      const double rem = o.probability + u.probability - 1.;

      under.pop_back();
      over.pop_back();

      if (rem < 1.)
        under.emplace_back(rem, index);
      else
        over.emplace_back(rem, index);
    }

    const AliasEntry one(1., -1);

    for (; !under.empty(); under.pop_back())
      aliasTable[under.back().alias] = one;

    for (; !over.empty(); over.pop_back()) 
      aliasTable[over.back().alias] = one;
  }

  std::vector<AliasEntry> aliasTable;
};

}  // namespace Utils

#endif  // _ALIAS_H_
