/** Portable dense contraction kernels. Axis zero is contiguous (Fortran order).
 */
#pragma once

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <valarray>
#include <vector>

#include <Eigen/Core>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Utils {
namespace detail {

inline size_t TensorSizeProduct(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::length_error("Tensor dimensions overflow size_t");
  return a * b;
}

// Most circuit tensors fit on the stack, including the bookkeeping for their
// free axes. Retain arbitrary runtime rank for circuit-cutting callers.
class TensorAxisMask {
 public:
  explicit TensorAxisMask(size_t rank)
      : extended(rank > local.size() ? rank : 0, 0) {}
  unsigned char &operator[](size_t i) {
    return extended.empty() ? local[i] : extended[i];
  }
  unsigned char operator[](size_t i) const {
    return extended.empty() ? local[i] : extended[i];
  }

 private:
  std::array<unsigned char, 32> local{};
  std::vector<unsigned char> extended;
};

struct TensorAxis {
  size_t extent;
  size_t stride;
};

struct TensorAxisPattern {
  std::vector<TensorAxis> axes;
  size_t count = 1;

  void Add(size_t extent, size_t stride) {
    count = TensorSizeProduct(count, extent);
    if (extent == 1) return;
    if (!axes.empty() &&
        stride == TensorSizeProduct(axes.back().extent, axes.back().stride))
      axes.back().extent = TensorSizeProduct(axes.back().extent, extent);
    else
      axes.push_back({extent, stride});
  }
  bool Linear() const { return axes.size() <= 1; }
  size_t Stride() const { return axes.empty() ? 0 : axes[0].stride; }
};

inline bool TensorMatrixContiguous(const TensorAxisPattern &rows,
                                    const TensorAxisPattern &columns) {
  return rows.Linear() && rows.Stride() == 1 && columns.Linear() &&
         columns.Stride() == rows.count;
}

// Irregular axes are decoded once, not in each multiply-add. Linear patterns
// (including ordinary matrix layouts) require no offset table.
class TensorAxisOffsets {
 public:
  TensorAxisOffsets() = default;
  explicit TensorAxisOffsets(const TensorAxisPattern &pattern)
      : stride(pattern.Stride()) {
    if (pattern.Linear()) return;
    offsets.resize(pattern.count);
    size_t n = 1;
    for (const auto &axis : pattern.axes) {
      for (size_t x = 1; x < axis.extent; ++x)
        for (size_t i = 0; i < n; ++i)
          offsets[i + x * n] = offsets[i] + x * axis.stride;
      n *= axis.extent;
    }
  }
  size_t operator[](size_t i) const {
    return offsets.empty() ? i * stride : offsets[i];
  }
  size_t Bytes() const { return offsets.capacity() * sizeof(size_t); }

 private:
  size_t stride = 0;
  std::vector<size_t> offsets;
};

template <class Pairs>
std::vector<size_t> TensorContractionDimensions(const std::vector<size_t> &da,
                                                const std::vector<size_t> &db,
                                                const Pairs &pairs,
                                                TensorAxisMask &usedA,
                                                TensorAxisMask &usedB) {
  for (const auto &pair : pairs) {
    if (pair.first >= da.size() || pair.second >= db.size())
      throw std::invalid_argument("Contraction axis is out of range");
    if (usedA[pair.first] || usedB[pair.second])
      throw std::invalid_argument("Contraction axes must be unique");
    if (da[pair.first] != db[pair.second])
      throw std::invalid_argument("Contracted dimensions must match");
    usedA[pair.first] = usedB[pair.second] = 1;
  }
  std::vector<size_t> dimensions;
  dimensions.reserve(da.size() + db.size() - 2 * pairs.size());
  for (size_t i = 0; i < da.size(); ++i)
    if (!usedA[i]) dimensions.push_back(da[i]);
  for (size_t i = 0; i < db.size(); ++i)
    if (!usedB[i]) dimensions.push_back(db[i]);
  if (dimensions.empty()) dimensions.push_back(1);
  return dimensions;
}

template <class Pairs>
void TensorContractionPatterns(const std::vector<size_t> &da,
                               const std::vector<size_t> &db,
                               const TensorAxisMask &usedA,
                               const TensorAxisMask &usedB, const Pairs &pairs,
                               TensorAxisPattern &fa, TensorAxisPattern &fb,
                               TensorAxisPattern &ka, TensorAxisPattern &kb) {
  std::vector<size_t> sa(da.size()), sb(db.size());
  size_t stride = 1;
  for (size_t i = 0; i < da.size(); ++i) {
    sa[i] = stride;
    stride = TensorSizeProduct(stride, da[i]);
    if (!usedA[i]) fa.Add(da[i], sa[i]);
  }
  stride = 1;
  for (size_t i = 0; i < db.size(); ++i) {
    sb[i] = stride;
    stride = TensorSizeProduct(stride, db[i]);
    if (!usedB[i]) fb.Add(db[i], sb[i]);
  }
  // Favor contiguous left-hand reduction axes, preserving the axis pairing.
  std::vector<std::pair<size_t, size_t>> ordered(pairs.begin(), pairs.end());
  std::sort(ordered.begin(), ordered.end());
  for (const auto &pair : ordered) {
    ka.Add(da[pair.first], sa[pair.first]);
    kb.Add(db[pair.second], sb[pair.second]);
  }
}

// Positive sizes denote a gate on the right, negative sizes one on the left.
// The existing stack kernel continues to handle two small binary tensors.
inline int TensorGateSize(const std::vector<size_t> &da,
                          const std::vector<size_t> &db, size_t pairs) {
  if (pairs != 1 && pairs != 2) return 0;
  auto isGate = [pairs](const auto &dims) {
    return dims.size() == 2 * pairs &&
           std::all_of(dims.begin(), dims.end(),
                       [](size_t d) { return d == 2; });
  };
  if (da.size() > 4 && isGate(db)) return 1 << pairs;
  if (db.size() > 4 && isGate(da)) return -(1 << pairs);
  return 0;
}

// Subtraction/division checks prevent overflow before optional allocations.
inline bool TensorAddBytes(size_t &used, size_t count, size_t elementSize,
                           size_t limit) {
  if (used > limit || count > (limit - used) / elementSize) return false;
  used += count * elementSize;
  return true;
}

// A full reduction can be as large as both inputs. An odometer avoids adding
// input-sized index arrays just to compute one scalar.
class TensorAxisCursor {
 public:
  TensorAxisCursor(const TensorAxisPattern &pattern, size_t linear)
      : pattern(pattern), indices(pattern.axes.size(), 0) {
    for (size_t i = 0; i < indices.size(); ++i) {
      indices[i] = linear % pattern.axes[i].extent;
      linear /= pattern.axes[i].extent;
      offset += indices[i] * pattern.axes[i].stride;
    }
  }
  void Next() {
    for (size_t i = 0; i < indices.size(); ++i) {
      const auto &axis = pattern.axes[i];
      if (++indices[i] < axis.extent) {
        offset += axis.stride;
        return;
      }
      indices[i] = 0;
      offset -= (axis.extent - 1) * axis.stride;
    }
  }
  size_t offset = 0;

 private:
  const TensorAxisPattern &pattern;
  std::vector<size_t> indices;
};

inline int TensorContractionThreads(size_t jobs, double work, bool allow) {
#ifdef _OPENMP
  if (allow && !omp_in_parallel()) {
    // Amortize a team over arithmetic work, rather than output size alone.
    // Respect OMP_NUM_THREADS / omp_set_num_threads and limit useful workers.
    const double useful =
        std::min(static_cast<double>(jobs), std::max(1.0, work / 65536.0));
    return std::max(
        1, static_cast<int>(std::min<double>(omp_get_max_threads(), useful)));
  }
#else
  (void)jobs;
  (void)work;
  (void)allow;
#endif
  return 1;
}

template <class Function>
void TensorParallelRanges(size_t jobs, int threads, const Function &function) {
#ifdef _OPENMP
  if (threads > 1) {
#pragma omp parallel num_threads(threads)
    {
      const size_t count = static_cast<size_t>(omp_get_num_threads());
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      const size_t block = jobs / count;
      const size_t extra = jobs % count;
      const size_t begin = tid * block + std::min(tid, extra);
      function(begin, begin + block + (tid < extra ? 1 : 0), tid);
    }
    return;
  }
#else
  (void)threads;
#endif
  function(0, jobs, 0);
}

template <class T, class Storage>
constexpr bool TensorEigenStorage =
    (std::is_same_v<T, float> || std::is_same_v<T, double> ||
     std::is_same_v<T, std::complex<float>> ||
     std::is_same_v<T, std::complex<double>>) &&
    (std::is_same_v<Storage, std::valarray<T>> ||
     std::is_same_v<Storage, std::vector<T>> ||
     std::is_same_v<Storage, const T *> || std::is_same_v<Storage, T *>);

template <class T, class Storage>
T TensorScalarContraction(const Storage &a, const Storage &b,
                          const TensorAxisPattern &ka,
                          const TensorAxisPattern &kb, bool allow) {
  const int threads =
      TensorContractionThreads(ka.count, double(ka.count), allow);
  auto reduce = [&](size_t begin, size_t end) -> T {
    if constexpr (TensorEigenStorage<T, Storage>) {
      if (ka.Linear() && kb.Linear() && ka.Stride() == 1 && kb.Stride() == 1) {
        using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
        const Eigen::Map<const Vector> av(&a[begin], end - begin);
        const Eigen::Map<const Vector> bv(&b[begin], end - begin);
        // dot()/adjoint() conjugate complex values; contraction does not.
        return av.cwiseProduct(bv).sum();
      }
    }
    TensorAxisCursor ai(ka, begin), bi(kb, begin);
    T sum{};
    for (size_t k = begin; k < end; ++k) {
      sum = sum + a[ai.offset] * b[bi.offset];
      ai.Next();
      bi.Next();
    }
    return sum;
  };
  if (threads == 1) return reduce(0, ka.count);
  std::vector<T> partials(threads, T{});
  TensorParallelRanges(ka.count, threads,
                       [&](size_t begin, size_t end, size_t tid) {
                         partials[tid] = reduce(begin, end);
                       });
  T sum{};
  for (const auto &partial : partials) sum = sum + partial;
  return sum;
}

template <class T, class Storage, class Pairs>
bool TensorSmallContraction(const Storage &a, const Storage &b, Storage &out,
                            const std::vector<size_t> &da,
                            const std::vector<size_t> &db,
                            const TensorAxisMask &usedA,
                            const TensorAxisMask &usedB, const Pairs &pairs) {
  if (da.size() > 4 || db.size() > 4) return false;
  for (auto d : da)
    if (d == 0 || d > 2) return false;
  for (auto d : db)
    if (d == 0 || d > 2) return false;
  std::array<size_t, 4> sa{}, sb{};
  size_t stride = 1;
  for (size_t i = 0; i < da.size(); ++i) {
    sa[i] = stride;
    stride *= da[i];
  }
  stride = 1;
  for (size_t i = 0; i < db.size(); ++i) {
    sb[i] = stride;
    stride *= db[i];
  }
  std::array<size_t, 16> fa{}, fb{}, ka{}, kb{};
  auto append = [](auto &offsets, size_t &count, size_t d, size_t s) {
    if (d == 2)
      for (size_t i = 0; i < count; ++i) offsets[i + count] = offsets[i] + s;
    count *= d;
  };
  size_t m = 1, n = 1, k = 1, kbCount = 1;
  for (size_t i = 0; i < da.size(); ++i)
    if (!usedA[i]) append(fa, m, da[i], sa[i]);
  for (size_t i = 0; i < db.size(); ++i)
    if (!usedB[i]) append(fb, n, db[i], sb[i]);
  for (const auto &pair : pairs) {
    append(ka, k, da[pair.first], sa[pair.first]);
    append(kb, kbCount, db[pair.second], sb[pair.second]);
  }
  for (size_t j = 0; j < n; ++j)
    for (size_t i = 0; i < m; ++i) {
      T sum{};
      for (size_t x = 0; x < k; ++x)
        sum = sum + a[fa[i] + ka[x]] * b[fb[j] + kb[x]];
      out[i + m * j] = sum;
    }
  return true;
}

template <size_t G, class T, size_t... X>
T TensorGateProduct(const std::array<T, G> &values, const T *coefficients,
                    std::index_sequence<X...>) {
  return ((values[X] * coefficients[X]) + ...);
}

template <size_t G, bool SmallLeft, class T, class Input, class Output>
void TensorGateContraction(const Input &a, const Input &b, Output &out,
                           size_t count, const TensorAxisOffsets &af,
                           const TensorAxisOffsets &bf,
                           const TensorAxisOffsets &ac,
                           const TensorAxisOffsets &bc, bool allow) {
  const auto &big = SmallLeft ? b : a;
  const auto &gate = SmallLeft ? a : b;
  const auto &free = SmallLeft ? bf : af;
  const auto &contract = SmallLeft ? bc : ac;
  const auto &gateFree = SmallLeft ? af : bf;
  const auto &gateContract = SmallLeft ? ac : bc;
  std::array<T, G * G> coefficients;
  std::array<size_t, G> offsets;
  for (size_t x = 0; x < G; ++x) {
    offsets[x] = contract[x];
    for (size_t j = 0; j < G; ++j)
      coefficients[j * G + x] = gate[gateFree[j] + gateContract[x]];
  }
  const int threads =
      TensorContractionThreads(count, double(count) * G * G, allow);
  TensorParallelRanges(count, threads, [&](size_t begin, size_t end, size_t) {
    for (size_t i = begin; i < end; ++i) {
      const size_t base = free[i];
      std::array<T, G> values;
      for (size_t x = 0; x < G; ++x) values[x] = big[base + offsets[x]];
      for (size_t j = 0; j < G; ++j)
        out[SmallLeft ? j + G * i : i + count * j] = TensorGateProduct<G>(
            values, coefficients.data() + j * G, std::make_index_sequence<G>{});
    }
  });
}

// Shared execution for one-off contractions and prepared network plans.
template <class T, class Input, class Output>
void TensorExecuteContraction(
    const Input &a, const Input &b, Output &out, const TensorAxisPattern &fa,
    const TensorAxisPattern &fb, const TensorAxisPattern &ka,
    const TensorAxisPattern &kb, const TensorAxisOffsets &af,
    const TensorAxisOffsets &bf, const TensorAxisOffsets &ac,
    const TensorAxisOffsets &bc, int gateSize, bool allow,
    T *packing = nullptr) {
  const size_t m = fa.count, n = fb.count, k = ka.count;
  if (m == 1 && n == 1) {
    out[0] = TensorScalarContraction<T>(a, b, ka, kb, allow);
    return;
  }
  const double work = double(m) * double(n) * double(k);
  if constexpr (TensorEigenStorage<T, Input> && TensorEigenStorage<T, Output>) {
    switch (gateSize) {
      case 2:
        TensorGateContraction<2, false, T>(a, b, out, m, af, bf, ac, bc, allow);
        return;
      case 4:
        TensorGateContraction<4, false, T>(a, b, out, m, af, bf, ac, bc, allow);
        return;
      case -2:
        TensorGateContraction<2, true, T>(a, b, out, n, af, bf, ac, bc, allow);
        return;
      case -4:
        TensorGateContraction<4, true, T>(a, b, out, n, af, bf, ac, bc, allow);
        return;
      default:
        break;
    }
    // Small/skinny products have too little reuse to amortize GEMM packing.
    if (m >= 8 && n >= 8 && k >= 8) {
      using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      const T *ap = &a[0], *bp = &b[0];
      Matrix packedA, packedB;
      if (!TensorMatrixContiguous(fa, ka)) {
        T *packed = packing;
        if (packed)
          packing += m * k;
        else {
          packedA.resize(m, k);
          packed = packedA.data();
        }
        for (size_t x = 0; x < k; ++x)
          for (size_t i = 0; i < m; ++i)
            packed[i + m * x] = a[af[i] + ac[x]];
        ap = packed;
      }
      if (!TensorMatrixContiguous(kb, fb)) {
        T *packed = packing;
        if (!packed) {
          packedB.resize(k, n);
          packed = packedB.data();
        }
        for (size_t j = 0; j < n; ++j)
          for (size_t x = 0; x < k; ++x)
            packed[x + k * j] = b[bf[j] + bc[x]];
        bp = packed;
      }
      // Bounded row tiles keep Eigen's GEMM serial. Our outer work partition
      // owns parallelism, without changing process-wide Eigen thread settings.
      constexpr size_t rowsPerTile = 32, colsPerTile = 128;
      using Tile = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::ColMajor, 32, Eigen::Dynamic>;
      const size_t rowTiles = (m - 1) / rowsPerTile + 1;
      const size_t colTiles = (n - 1) / colsPerTile + 1;
      const size_t jobs = TensorSizeProduct(rowTiles, colTiles);
      const int threads = TensorContractionThreads(jobs, work, allow);
      TensorParallelRanges(
          jobs, threads, [&](size_t begin, size_t end, size_t) {
            for (size_t job = begin; job < end; ++job) {
              const size_t row = (job % rowTiles) * rowsPerTile;
              const size_t col = (job / rowTiles) * colsPerTile;
              const size_t rows = std::min(rowsPerTile, m - row);
              const size_t cols = std::min(colsPerTile, n - col);
              const Eigen::Map<const Matrix, 0, Eigen::OuterStride<>> left(
                  ap + row, rows, k, Eigen::OuterStride<>(m));
              const Eigen::Map<const Matrix> right(bp + k * col, k, cols);
              Eigen::Map<Tile, 0, Eigen::OuterStride<>> dest(
                  &out[row + m * col], rows, cols, Eigen::OuterStride<>(m));
#if defined(EIGEN_USE_BLAS) || defined(EIGEN_GEMM_THREADPOOL)
              // External runtimes own their thread pools. Avoid starting one
              // behind the caller's serial switch or inside an OpenMP worker.
              dest.noalias() = left.lazyProduct(right);
#else
              dest.noalias() = left * right;
#endif
            }
          });
      return;
    }
  }
  const size_t jobs = TensorSizeProduct(m, n);
  const int threads = TensorContractionThreads(jobs, work, allow);
  TensorParallelRanges(jobs, threads, [&](size_t begin, size_t end, size_t) {
    size_t i = begin % m, j = begin / m;
    for (size_t offset = begin; offset < end; ++offset) {
      const size_t baseA = af[i], baseB = bf[j];
      T sum{};
      for (size_t x = 0; x < k; ++x)
        sum = sum + a[baseA + ac[x]] * b[baseB + bc[x]];
      out[offset] = sum;
      if (++i == m) {
        i = 0;
        ++j;
      }
    }
  });
}

template <class T, class Storage>
void TensorDenseContraction(const Storage &a, const Storage &b, Storage &out,
                            const TensorAxisPattern &fa,
                            const TensorAxisPattern &fb,
                            const TensorAxisPattern &ka,
                            const TensorAxisPattern &kb, bool allow,
                            int gateSize = 0) {
  if (fa.count == 1 && fb.count == 1) {
    out[0] = TensorScalarContraction<T>(a, b, ka, kb, allow);
    return;
  }
  const TensorAxisOffsets af(fa), bf(fb), ac(ka), bc(kb);
  TensorExecuteContraction<T>(a, b, out, fa, fb, ka, kb, af, bf, ac, bc,
                              gateSize, allow);
}

// Internal prepared metadata. Execute requires validated shapes and distinct
// live input/output storage; ForestContractor enforces these at plan reuse.
class TensorContractionPlan {
 public:
  template <class Pairs>
  bool Prepare(const std::vector<size_t> &da, const std::vector<size_t> &db,
               const Pairs &pairs, size_t byteLimit) {
    TensorContractionPlan candidate;
    if (!candidate.Initialize(da, db, pairs, byteLimit)) return false;
    *this = std::move(candidate);
    return true;
  }
  const std::vector<size_t> &GetDims() const { return dimensions; }
  size_t GetSize() const { return size; }
  // Optional GEMM scratch in elements. Overflow disables retained packing.
  size_t GetPackingSize() const {
    const size_t m = fa.count, n = fb.count, k = ka.count;
    if (gateSize || m < 8 || n < 8 || k < 8) return 0;
    size_t count = 0;
    const size_t limit = std::numeric_limits<size_t>::max();
    if ((!TensorMatrixContiguous(fa, ka) &&
         !TensorAddBytes(count, m, k, limit)) ||
        (!TensorMatrixContiguous(kb, fb) &&
         !TensorAddBytes(count, k, n, limit)))
      return limit;
    return count;
  }
  size_t ExtraBytes() const {
    return dimensions.capacity() * sizeof(size_t) +
           (fa.axes.capacity() + fb.axes.capacity() + ka.axes.capacity() +
            kb.axes.capacity()) *
               sizeof(TensorAxis) +
           af.Bytes() + bf.Bytes() + ac.Bytes() + bc.Bytes();
  }
  // Scratch, if supplied, holds GetPackingSize() elements and must not alias
  // live inputs or outputs. Values are packed anew on every execution.
  template <class T>
  void Execute(const T *a, const T *b, T *out, bool allow,
               T *packing = nullptr) const {
    TensorExecuteContraction<T>(a, b, out, fa, fb, ka, kb, af, bf, ac, bc,
                                gateSize, allow, packing);
  }

 private:
  template <class Pairs>
  bool Initialize(const std::vector<size_t> &da, const std::vector<size_t> &db,
                  const Pairs &pairs, size_t byteLimit) {
    TensorAxisMask usedA(da.size()), usedB(db.size());
    dimensions = TensorContractionDimensions(da, db, pairs, usedA, usedB);
    if (std::find(da.begin(), da.end(), 0) != da.end() ||
        std::find(db.begin(), db.end(), 0) != db.end())
      return false;
    TensorContractionPatterns(da, db, usedA, usedB, pairs, fa, fb, ka, kb);
    size = TensorSizeProduct(fa.count, fb.count);
    gateSize = TensorGateSize(da, db, pairs.size());
    size_t bytes = ExtraBytes();
    if (bytes > byteLimit) return false;
    // A scalar reduction uses cursors, never input-sized offset tables.
    if (fa.count == 1 && fb.count == 1) return true;
    for (const auto *pattern : {&fa, &fb, &ka, &kb})
      if (!pattern->Linear() &&
          !TensorAddBytes(bytes, pattern->count, sizeof(size_t), byteLimit))
        return false;
    af = TensorAxisOffsets(fa);
    bf = TensorAxisOffsets(fb);
    ac = TensorAxisOffsets(ka);
    bc = TensorAxisOffsets(kb);
    return ExtraBytes() <= byteLimit;
  }
  std::vector<size_t> dimensions;
  TensorAxisPattern fa, fb, ka, kb;
  TensorAxisOffsets af, bf, ac, bc;
  size_t size = 0;
  int gateSize = 0;
};

}  // namespace detail
}  // namespace Utils
