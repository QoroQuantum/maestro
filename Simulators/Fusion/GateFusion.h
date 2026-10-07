#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "../../Types.h"

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
    const size_t dim = size_t{1} << to.size(), sub = size_t{1} << from.size();
    std::vector<size_t> offsets(sub);
    const size_t mask = Offsets(from, to, offsets.data());
    // Only entries whose bits outside `from` agree are non-zero.
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (size_t base = 0; base < dim; ++base) {
      if (base & mask) continue;
      for (size_t col = 0; col < sub; ++col)
        for (size_t row = 0; row < sub; ++row)
          result(base | offsets[row], base | offsets[col]) = matrix(row, col);
    }
    return result;
  }

  // matrix <- embed(gate on `order`) * matrix, without forming the embedding:
  // the gate acts on the row index bits of `order` within `to`, costing
  // dim^2 * 2^k instead of dim^3 multiply-adds.
  static void ApplyLeft(const Eigen::MatrixXcd& gate,
                        const Types::qubits_vector& order,
                        const Types::qubits_vector& to,
                        Eigen::MatrixXcd& matrix) {
    const size_t k = order.size(), sub = size_t{1} << k;
    if (k > 3) throw std::logic_error("Fused gates act on at most 3 qubits");
    size_t offsets[8] = {};
    const size_t mask = Offsets(order, to, offsets);
    // Explicit real arithmetic: std::complex multiplication goes through a
    // NaN/Inf-checking library call without -ffast-math.
    double gr[64], gi[64];
    for (size_t r = 0; r < sub; ++r)
      for (size_t l = 0; l < sub; ++l) {
        gr[r * sub + l] = gate(r, l).real();
        gi[r * sub + l] = gate(r, l).imag();
      }
    auto* data = matrix.data();
    switch (matrix.rows() * 4 + k) {  // fixed sizes let the loops unroll
      case 2 * 4 + 1: return ApplyLeftFixed<1, 2>(gr, gi, offsets, mask, data);
      case 4 * 4 + 1: return ApplyLeftFixed<1, 4>(gr, gi, offsets, mask, data);
      case 4 * 4 + 2: return ApplyLeftFixed<2, 4>(gr, gi, offsets, mask, data);
      case 8 * 4 + 1: return ApplyLeftFixed<1, 8>(gr, gi, offsets, mask, data);
      case 8 * 4 + 2: return ApplyLeftFixed<2, 8>(gr, gi, offsets, mask, data);
      case 8 * 4 + 3: return ApplyLeftFixed<3, 8>(gr, gi, offsets, mask, data);
      default:
        throw std::logic_error("Fused blocks act on at most 3 qubits");
    }
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

    if (overlapping.empty()) {
      for (auto q : support) owners_[q] = source;
      blocks_.emplace(source, Block{std::move(support), {}, gate,
                                    {source, source, 1}, preserveStructure});
      return;
    }
    // A merged block is emitted through its matrix, never as `original`.
    Block next{{}, {}, Gate{}, {source, source, 1}, false};
    {
      // Start from the first block's matrix and left-apply the other
      // (disjoint, hence commuting) blocks and the new gate as small gates,
      // instead of multiplying full dense embeddings into an identity.
      bool first = true;
      for (auto id : overlapping) {
        const auto& old = blocks_.at(id);
        next.sources.first = std::min(next.sources.first, old.sources.first);
        next.sources.count += old.sources.count;
        if (!matrices_) continue;
        if (old.IsSingle()) {
          const auto order = old.original.Targets();
          if (first)
            next.matrix = Embed(old.original.Matrix(), order, support);
          else
            ApplyLeft(old.original.Matrix(), order, support, next.matrix);
        } else if (first)
          next.matrix = Embed(old.matrix, old.targets, support);
        else
          ApplyLeft(old.matrix, old.targets, support, next.matrix);
        first = false;
      }
      if (matrices_)
        ApplyLeft(gate.Matrix(), gate.Targets(), support, next.matrix);
    }
    // Compute the replacement before removing anything, so allocation/matrix
    // errors cannot lose pending operations. The merged block keeps the key
    // of the oldest overlapping block (keys are each block's first source),
    // and every qubit of the merged blocks is in `support`, so owner entries
    // are overwritten instead of erased and reinserted.
    const auto id = next.sources.first;
    for (auto q : support) owners_[q] = id;
    next.targets = std::move(support);
    bool reuse = false;
    for (auto old : overlapping)
      if (old == id)
        reuse = true;
      else
        blocks_.erase(old);
    if (reuse)
      blocks_.at(id) = std::move(next);
    else  // a source older than every merged block
      blocks_.emplace(id, std::move(next));
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
  // Row offset of every local index of `from` within `to`; returns the mask.
  static size_t Offsets(const Types::qubits_vector& from,
                        const Types::qubits_vector& to, size_t* offsets) {
    const size_t sub = size_t{1} << from.size();
    size_t mask = 0;
    std::fill(offsets, offsets + sub, 0);
    for (size_t i = 0; i < from.size(); ++i) {
      const auto pos = std::find(to.begin(), to.end(), from[i]);
      if (pos == to.end()) throw std::logic_error("Invalid fusion embedding");
      const size_t bit = size_t{1} << (pos - to.begin());
      mask |= bit;
      for (size_t l = 0; l < sub; ++l)
        if (l & (size_t{1} << i)) offsets[l] |= bit;
    }
    return mask;
  }
  template <size_t K, size_t D>
  static void ApplyLeftFixed(const double* gr, const double* gi,
                             const size_t* offsets, size_t mask,
                             std::complex<double>* data) {
    constexpr size_t S = size_t{1} << K;
    for (size_t col = 0; col < D; ++col) {
      auto* column = data + col * D;
      for (size_t base = 0; base < D; ++base) {
        if (base & mask) continue;
        double inr[S], ini[S];
        for (size_t l = 0; l < S; ++l) {
          inr[l] = column[base | offsets[l]].real();
          ini[l] = column[base | offsets[l]].imag();
        }
        for (size_t r = 0; r < S; ++r) {
          double re = 0, im = 0;
          for (size_t l = 0; l < S; ++l) {
            re += gr[r * S + l] * inr[l] - gi[r * S + l] * ini[l];
            im += gr[r * S + l] * ini[l] + gi[r * S + l] * inr[l];
          }
          column[base | offsets[r]] = {re, im};
        }
      }
    }
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
