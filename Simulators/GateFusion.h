#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "../Types.h"

namespace Simulators {

// Backend-independent cache. Gate supplies Targets() and Matrix(); the same
// algorithm runs without matrices when preparing a routing sequence.
template <class Gate>
class GateFusion {
 public:
  // Constant-size provenance: submissions are checked against the prepared
  // source operations before entering the cache. The first/last IDs and count
  // then detect an early flush without copying a growing gate list on every
  // merge (which would make a long same-qubit sequence quadratic).
  struct Sources {
    uint64_t first, last, count;
    bool operator==(const Sources& other) const {
      return first == other.first && last == other.last && count == other.count;
    }
    bool operator!=(const Sources& other) const { return !(*this == other); }
  };
  struct Block {
    Types::qubits_vector targets;
    Eigen::MatrixXcd matrix;
    Gate original;
    Sources sources;
    bool preserveStructure = false;
    bool IsSingle() const { return sources.count == 1; }
  };

  explicit GateFusion(unsigned width = 2, bool matrices = true)
      : width_(width), matrices_(matrices) {}

  bool Empty() const { return blocks_.empty(); }
  size_t Size() const { return blocks_.size(); }
  void Clear() {
    blocks_.clear();
    owners_.clear();
  }
  void SetWidth(unsigned width) {
    if (!Empty()) throw std::logic_error("Flush before changing fusion width");
    width_ = width;
  }

  // First target is the least-significant local matrix bit. Also performs
  // target permutations when from and to contain the same qubits.
  static Eigen::MatrixXcd Embed(const Eigen::MatrixXcd& matrix,
                                const Types::qubits_vector& from,
                                const Types::qubits_vector& to) {
    if (from == to) return matrix;
    const size_t dim = size_t{1} << to.size();
    std::vector<size_t> bits;
    size_t mask = 0;
    for (auto q : from) {
      const auto pos = std::find(to.begin(), to.end(), q);
      if (pos == to.end()) throw std::logic_error("Invalid fusion embedding");
      const auto bit = size_t{1} << (pos - to.begin());
      bits.push_back(bit);
      mask |= bit;
    }
    auto local = [&](size_t index) {
      size_t result = 0;
      for (size_t i = 0; i < bits.size(); ++i)
        if (index & bits[i]) result |= size_t{1} << i;
      return result;
    };
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (size_t row = 0; row < dim; ++row)
      for (size_t col = 0; col < dim; ++col)
        if ((row & ~mask) == (col & ~mask))
          result(row, col) = matrix(local(row), local(col));
    return result;
  }

  template <class Emit>
  void Submit(const Gate& gate, uint64_t source, Emit&& emit,
              bool preserveStructure = false) {
    auto targets = gate.Targets();
    if (targets.empty() || targets.size() > width_)
      throw std::invalid_argument("Gate exceeds fusion width");
    std::sort(targets.begin(), targets.end());
    if (std::adjacent_find(targets.begin(), targets.end()) != targets.end())
      throw std::invalid_argument("Gate targets must be distinct");

    std::vector<uint64_t> overlapping;
    for (auto q : targets) {
      auto it = owners_.find(q);
      if (it != owners_.end()) overlapping.push_back(it->second);
    }
    std::sort(overlapping.begin(), overlapping.end());
    overlapping.erase(std::unique(overlapping.begin(), overlapping.end()),
                      overlapping.end());
    // Keep native structured operations when a merge would only replace
    // diagonal/permutation kernels with a dense one. Mixed groups can still
    // benefit from fusion. Emitting only overlaps preserves disjoint caching.
    if (preserveStructure && !overlapping.empty() &&
        std::all_of(overlapping.begin(), overlapping.end(), [&](auto id) {
          return blocks_.at(id).preserveStructure;
        })) {
      for (auto id : overlapping) EmitOne(id, emit);
      overlapping.clear();
    }
    auto support = targets;
    for (auto id : overlapping) {
      const auto& old = blocks_.at(id);
      support.insert(support.end(), old.targets.begin(), old.targets.end());
    }
    std::sort(support.begin(), support.end());
    support.erase(std::unique(support.begin(), support.end()), support.end());
    if (support.size() > width_) {
      // Keep any block that still fits. In particular, an unrelated partner
      // on one target must not force a cached single-qubit gate on the other
      // target to be emitted as well. Disjoint cached blocks commute, so the
      // rejected blocks can be emitted before the retained ones.
      support = targets;
      std::vector<uint64_t> retained;
      for (auto id : overlapping) {
        auto candidate = support;
        const auto& old = blocks_.at(id);
        candidate.insert(candidate.end(), old.targets.begin(),
                         old.targets.end());
        std::sort(candidate.begin(), candidate.end());
        candidate.erase(std::unique(candidate.begin(), candidate.end()),
                        candidate.end());
        if (candidate.size() <= width_) {
          retained.push_back(id);
          support = std::move(candidate);
        } else
          EmitOne(id, emit);
      }
      overlapping = std::move(retained);
    }

    Block next{support,
               {},
               gate,
               {source, source, 1},
               preserveStructure && overlapping.empty()};
    if (!overlapping.empty()) {
      if (matrices_) {
        const auto dim = size_t{1} << support.size();
        next.matrix = Eigen::MatrixXcd::Identity(dim, dim);
      }
      for (auto id : overlapping) {
        const auto& old = blocks_.at(id);
        next.sources.first = std::min(next.sources.first, old.sources.first);
        next.sources.count += old.sources.count;
        if (matrices_) {
          const auto matrix =
              old.IsSingle() ? old.original.Matrix() : old.matrix;
          const auto& order =
              old.IsSingle() ? old.original.Targets() : old.targets;
          next.matrix = (Embed(matrix, order, support) * next.matrix).eval();
        }
      }
      if (matrices_)
        next.matrix =
            (Embed(gate.Matrix(), gate.Targets(), support) * next.matrix)
                .eval();
    }
    // Compute the replacement before removing anything, so allocation/matrix
    // errors cannot lose pending operations.
    for (auto id : overlapping) Remove(id);
    const auto id = next.sources.first;
    blocks_.emplace(id, std::move(next));
    for (auto q : support) owners_[q] = id;
  }

  template <class Emit>
  void Flush(Emit&& emit) {
    while (!blocks_.empty()) EmitOne(blocks_.begin()->first, emit);
  }

 private:
  void Remove(uint64_t id) {
    for (auto q : blocks_.at(id).targets) owners_.erase(q);
    blocks_.erase(id);
  }
  template <class Emit>
  void EmitOne(uint64_t id, Emit& emit) {
    emit(blocks_.at(id));
    Remove(id);
  }
  unsigned width_;
  bool matrices_;
  std::map<uint64_t, Block> blocks_;
  std::unordered_map<Types::qubit_t, uint64_t> owners_;
};
}  // namespace Simulators
