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

// Irregular axes are decoded once, not in each multiply-add. Linear patterns
// (including ordinary matrix layouts) require no offset table.
class TensorAxisOffsets {
 public:
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

 private:
  size_t stride;
  std::vector<size_t> offsets;
};

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
     std::is_same_v<Storage, std::vector<T>>);

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

template <class T, class Storage>
void TensorDenseContraction(const Storage &a, const Storage &b, Storage &out,
                            const TensorAxisPattern &fa,
                            const TensorAxisPattern &fb,
                            const TensorAxisPattern &ka,
                            const TensorAxisPattern &kb, bool allow) {
  const size_t m = fa.count, n = fb.count, k = ka.count;
  if (m == 1 && n == 1) {
    out[0] = TensorScalarContraction<T>(a, b, ka, kb, allow);
    return;
  }
  const TensorAxisOffsets af(fa), bf(fb), ac(ka), bc(kb);
  const double work = double(m) * double(n) * double(k);
  if constexpr (TensorEigenStorage<T, Storage>) {
    // Small/skinny products have too little reuse to amortize GEMM packing.
    if (m >= 8 && n >= 8 && k >= 8) {
      using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
      const T *ap = &a[0], *bp = &b[0];
      Matrix packedA, packedB;
      if (!(fa.Linear() && fa.Stride() == 1 && ka.Linear() &&
            ka.Stride() == m)) {
        packedA.resize(m, k);
        for (size_t x = 0; x < k; ++x)
          for (size_t i = 0; i < m; ++i) packedA(i, x) = a[af[i] + ac[x]];
        ap = packedA.data();
      }
      if (!(kb.Linear() && kb.Stride() == 1 && fb.Linear() &&
            fb.Stride() == k)) {
        packedB.resize(k, n);
        for (size_t j = 0; j < n; ++j)
          for (size_t x = 0; x < k; ++x) packedB(x, j) = b[bf[j] + bc[x]];
        bp = packedB.data();
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

}  // namespace detail
}  // namespace Utils
