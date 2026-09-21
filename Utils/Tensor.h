/**
 * @file Tensor.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The tensor class.
 *
 * The problem whith the Eigen library tensors is that the rank is set at
 * compile time, we need a tensor class that can have a rank specified at
 * runtime (it depends on the particular circuit cutting).
 */

#pragma once

#ifndef _TENSOR_H_
#define _TENSOR_H_ 1

#include <complex>
#include <initializer_list>
#include <numeric>
#include <unordered_set>
#include <valarray>
#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/valarray.hpp>
#include <boost/serialization/vector.hpp>

#include "QubitRegisterCalculator.h"
#include "TensorContraction.h"

namespace Utils {

// WARNING: Not all functions are tested, some of them are tested only
// superficially, use with caution! Some were tested before switching to fortran
// layout, so they might not work properly anymore! Accessing values and
// contractions are working since they are used in circuit cutting and in the
// tensor networks simulator.
template <class T = std::complex<double>, class Storage = std::valarray<T>>
class Tensor {
 protected:
  Storage values;
  std::vector<size_t> dims;
  mutable size_t sz = 0;

 public:
  friend class boost::serialization::access;

  Tensor() : sz(0) {}

  // a dummy tensor should be used only for incrementing indices, not for
  // storing values!
  Tensor(const std::vector<size_t> &dims, bool dummy = false)
      : dims(dims), sz(0) {
    if (!dummy) {
      if constexpr (std::is_convertible<int, T>::value)
        values.resize(GetSize(), static_cast<T>(0));
      else
        values.resize(GetSize());
    }
  }

  Tensor(const std::vector<int> &dims, bool dummy = false)
      : dims(dims.cbegin(), dims.cend()), sz(0) {
    if (!dummy) {
      if constexpr (std::is_convertible<int, T>::value)
        values.resize(GetSize(), static_cast<T>(0));
      else
        values.resize(GetSize());
    }
  }

  template <class indT = size_t>
  Tensor(std::initializer_list<indT> dims, bool dummy = false)
      : dims(dims.begin(), dims.end()), sz(0) {
    if (!dummy) {
      if constexpr (std::is_convertible<int, T>::value)
        values.resize(GetSize(), static_cast<T>(0));
      else
        values.resize(GetSize());
    }
  }

  Tensor(const Tensor<T, Storage> &other)
      : values(other.values), dims(other.dims), sz(other.sz) {}

  Tensor(Tensor<T, Storage> &&other) noexcept {
    dims.swap(other.dims);
    values.swap(other.values);
    std::swap(sz, other.sz);
  }

  virtual ~Tensor() = default;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int /*version*/) {
    if (typename Archive::is_loading()) values.clear();

    ar &dims &sz;
  }

  Tensor<T, Storage> &operator=(const Tensor<T, Storage> &other) {
    Tensor temp(other);
    *this = std::move(temp);

    return *this;
  }

  Tensor<T, Storage> &operator=(Tensor<T, Storage> &&other) noexcept {
    if (this != &other) {
      dims.swap(other.dims);
      values.swap(other.values);
      std::swap(sz, other.sz);
    }

    return *this;
  }

  void Swap(Tensor<T, Storage> &other) {
    dims.swap(other.dims);
    values.swap(other.values);
    std::swap(sz, other.sz);
  }

  size_t GetSize() const {
    if (!sz) {
      size_t count = 1;
      for (auto dimension : dims)
        count = detail::TensorSizeProduct(count, dimension);
      sz = count;
    }

    return sz;
  }

  size_t GetDim(size_t index) const {
    assert(index < dims.size());

    return dims[index];
  }

  const std::vector<size_t> &GetDims() const { return dims; }

  const size_t GetRank() const { return dims.size(); }

  inline bool IsDummy() const { return values.size() == 0; }

  inline bool IncrementIndex(std::vector<size_t> &indices) const {
    long long pos = dims.size() - 1;

    while (pos >= 0) {
      if (indices[pos] < dims[pos] - 1) {
        ++indices[pos];
        break;
      } else {
        indices[pos] = 0;
        --pos;
      }
    }

    return pos >= 0;
  }

  inline bool IncrementIndexSkip(std::vector<size_t> &indices, int skip) const {
    long long pos = dims.size() - 1;

    while (pos >= 0) {
      const bool notOnPos = pos != skip;
      if (notOnPos && indices[pos] < dims[pos] - 1) {
        ++indices[pos];
        break;
      } else {
        if (notOnPos) indices[pos] = 0;
        --pos;
      }
    }

    return pos >= 0;
  }

  inline bool IncrementIndexSkip(std::vector<size_t> &indices,
                                 const std::vector<int> &skipv) const {
    long long int pos = dims.size() - 1;

    int skip = -1;
    int skipPos = -1;
    if (!skipv.empty()) {
      skip = skipv.back();
      skipPos = static_cast<int>(skipv.size()) - 1;
    }

    while (pos >= 0) {
      const bool notOnPos = pos != skip;
      if (notOnPos && indices[pos] < dims[pos] - 1) {
        ++indices[pos];
        break;
      } else {
        if (notOnPos) indices[pos] = 0;
        --pos;

        if (pos < skip) {
          --skipPos;
          if (skipPos >= 0)
            skip = skipv[skipPos];
          else
            skip = -1;
        }
      }
    }

    return pos >= 0;
  }

  void Clear() {
    dims.clear();
    ClearValues();
  }

  void ClearValues() {
    sz = 0;
    values.resize(0);
  }

  bool Empty() const { return dims.empty(); }

  void Resize(const std::vector<size_t> &newdims) {
    Clear();

    dims = newdims;
    sz = 0;
    values.resize(GetSize());
  }

  template <class indT = size_t>
  void Resize(std::initializer_list<indT> newdims) {
    Clear();

    dims = newdims;
    sz = 0;
    values.resize(GetSize());
  }

  const T at(const std::vector<size_t> &indices) const {
    assert(indices.size() == dims.size());

    return values[GetOffset(indices)];
  }

  inline T &operator[](const std::vector<size_t> &indices) {
    assert(indices.size() == dims.size());

    return values[GetOffset(indices)];
  }

  inline const T &operator[](const std::vector<size_t> &indices) const {
    return values[GetOffset(indices)];
  }

  T atOffset(size_t offset) {
    assert(values.size() > offset);

    return values[offset];
  }

  inline T &operator[](size_t offset) {
    assert(values.size() > offset);

    return values[offset];
  }

  inline const T &operator[](size_t offset) const { return values[offset]; }

  template <class indT = size_t>
  T &operator[](std::initializer_list<indT> args) {
    assert(args.size() == dims.size());

    const std::vector<size_t> indices(args.begin(), args.end());

    return values[GetOffset(indices)];
  }

  template <class indT = size_t>
  const T &operator[](std::initializer_list<indT> args) const {
    assert(args.size() == dims.size());

    const std::vector<size_t> indices(args.begin(), args.end());

    return values[GetOffset(indices)];
  }

  inline T &operator()(const std::vector<size_t> &indices) {
    assert(indices.size() == dims.size());

    return values[GetOffset(indices)];
  }

  inline const T &operator()(const std::vector<size_t> &indices) const {
    return values[GetOffset(indices)];
  }

  template <class indT = size_t>
  T &operator()(std::initializer_list<indT> args) {
    assert(args.size() == dims.size());

    const std::vector<size_t> indices(args.begin(), args.end());

    return values[GetOffset(indices)];
  }

  template <class indT = size_t>
  const T &operator()(std::initializer_list<indT> args) const {
    assert(args.size() == dims.size());

    const std::vector<size_t> indices(args.begin(), args.end());

    return values[GetOffset(indices)];
  }

  inline const Storage &GetValues() const { return values; }

  inline Storage &GetValues() { return values; }

  Tensor<T, Storage> operator+(const Tensor<T, Storage> &other) const {
    assert(dims == other.dims);

    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] + other.values[i];

    return result;
  }

  Tensor<T, Storage> operator-(const Tensor<T, Storage> &other) const {
    assert(dims == other.dims);

    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] - other.values[i];

    return result;
  }

  Tensor<T, Storage> operator*(const Tensor<T, Storage> &other) const {
    assert(dims == other.dims);

    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] * other.values[i];

    return result;
  }

  Tensor<T, Storage> operator/(const Tensor<T, Storage> &other) const {
    assert(dims == other.dims);

    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] / other.values[i];

    return result;
  }

  Tensor<T, Storage> operator+(const T &scalar) const {
    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] + scalar;

    return result;
  }

  Tensor<T, Storage> operator-(const T &scalar) const {
    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] - scalar;

    return result;
  }

  Tensor<T, Storage> operator*(const T &scalar) const {
    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] * scalar;

    return result;
  }

  Tensor<T, Storage> operator/(const T &scalar) const {
    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i)
      result.values[i] = values[i] / scalar;

    return result;
  }

  Tensor<T, Storage> &operator+=(const Tensor<T, Storage> &other) {
    assert(dims == other.dims);

    for (size_t i = 0; i < GetSize(); ++i) values[i] += other.values[i];

    return *this;
  }

  Tensor<T, Storage> &operator-=(const Tensor<T, Storage> &other) {
    assert(dims == other.dims);

    for (size_t i = 0; i < GetSize(); ++i) values[i] -= other.values[i];

    return *this;
  }

  Tensor<T, Storage> &operator*=(const Tensor<T, Storage> &other) {
    assert(dims == other.dims);

    for (size_t i = 0; i < GetSize(); ++i) values[i] *= other.values[i];

    return *this;
  }

  Tensor<T, Storage> &operator/=(const Tensor<T, Storage> &other) {
    assert(dims == other.dims);

    for (size_t i = 0; i < GetSize(); ++i) values[i] /= other.values[i];

    return *this;
  }

  Tensor<T, Storage> &operator+=(const T &scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] += scalar;

    return *this;
  }

  Tensor<T, Storage> &operator-=(const T &scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] -= scalar;

    return *this;
  }

  Tensor<T, Storage> &operator*=(const T &scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] *= scalar;

    return *this;
  }

  Tensor<T, Storage> &operator/=(const T &scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] /= scalar;

    return *this;
  }

  // the following four are for the special case then each value is a map of
  // results (for cutting)
  Tensor<T, Storage> &operator+=(const double scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] = values[i] + scalar;

    return *this;
  }

  Tensor<T, Storage> &operator-=(const double scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] = values[i] - scalar;

    return *this;
  }

  Tensor<T, Storage> &operator*=(const double scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] = scalar * values[i];

    return *this;
  }

  Tensor<T, Storage> &operator/=(const double scalar) {
    for (size_t i = 0; i < GetSize(); ++i) values[i] = values[i] / scalar;

    return *this;
  }

  Tensor<T, Storage> operator-() const {
    Tensor<T, Storage> result(dims);

    for (size_t i = 0; i < GetSize(); ++i) result.values[i] = -values[i];

    return result;
  }

  void Conj() {
    for (size_t i = 0; i < GetSize(); ++i) values[i] = std::conj(values[i]);
  }

  Tensor<T, Storage> TensorProduct(const Tensor<T, Storage> &other) const {
    std::vector<size_t> newdims(dims.size() + other.dims.size());

    for (size_t i = 0; i < dims.size(); ++i) newdims[i] = dims[i];

    for (size_t i = 0; i < other.dims.size(); ++i)
      newdims[dims.size() + i] = other.dims[i];

    Tensor<T, Storage> result(newdims, values.size() == 0);

    if (!IsDummy())
      for (size_t i = 0; i < GetSize(); ++i)
        for (size_t j = 0; j < other.GetSize(); ++j)
          result.values[i * other.GetSize() + j] = values[i] * other.values[j];

    return result;
  }

  Tensor<T, Storage> Contract(const Tensor<T, Storage> &other, size_t ind1,
                              size_t ind2,
                              bool allowMultithreading = true) const {
    const std::array<std::pair<size_t, size_t>, 1> indices{{{ind1, ind2}}};
    return ContractImpl(other, indices, allowMultithreading);
  }

  Tensor<T, Storage> Contract(
      const Tensor<T, Storage> &other,
      const std::vector<std::pair<size_t, size_t>> &indices,
      bool allowMultithreading = true) const {
    return ContractImpl(other, indices, allowMultithreading);
  }

 private:
  template <class Pairs>
  Tensor<T, Storage> ContractImpl(const Tensor<T, Storage> &other,
                                  const Pairs &indices, bool allow) const {
    detail::TensorAxisMask usedA(dims.size()), usedB(other.dims.size());
    const auto newdims = detail::TensorContractionDimensions(
        dims, other.dims, indices, usedA, usedB);

    Tensor<T, Storage> result(newdims, IsDummy() || other.IsDummy());
    if (result.IsDummy()) return result;
    if (detail::TensorSmallContraction<T>(values, other.values, result.values,
                                          dims, other.dims, usedA, usedB,
                                          indices))
      return result;

    detail::TensorAxisPattern fa, fb, ka, kb;
    detail::TensorContractionPatterns(dims, other.dims, usedA, usedB, indices,
                                      fa, fb, ka, kb);
    detail::TensorDenseContraction<T>(values, other.values, result.values, fa,
                                      fb, ka, kb, allow,
                                      detail::TensorGateSize(dims, other.dims,
                                                            indices.size()));
    return result;
  }

 public:
  T Trace() const {
    assert(dims.size() > 0);

    T result = 0;

    if (values.size() == 0) return result;

    size_t dimMin = dims[0];
    for (size_t i = 1; i < dims.size(); ++i)
      if (dims[i] < dimMin) dimMin = dims[i];

    for (size_t i = 0; i < dimMin; ++i) {
      const std::vector<size_t> indices(dims.size(), i);

      result += values[GetOffset(indices)];
    }

    return result;
  }

  Tensor<T, Storage> Trace(size_t ind1, size_t ind2) const {
    assert(ind1 < dims.size() && ind2 < dims.size());

    std::vector<size_t> newdims;

    size_t newsize = dims.size() - 2;

    if (newsize == 0) {
      newsize = 1;
      newdims.push_back(1);
    } else {
      newdims.reserve(newsize);

      for (size_t i = 0; i < dims.size(); ++i)
        if (i != ind1 && i != ind2) newdims.push_back(dims[i]);
    }

    size_t dimMin = dims[ind1];
    if (dims[ind2] < dimMin) dimMin = dims[ind2];

    Tensor<T, Storage> result(newdims, values.size() == 0);

    if (values.size() != 0) {
      for (size_t offset = 0; offset < result.GetSize(); ++offset) {
        std::vector<size_t> indices1(dims.size());
        const std::vector<size_t> indices = result.IndexFromOffset(offset);

        size_t skip = 0;
        for (size_t i = 0; i < dims.size(); ++i)
          if (i != ind1 && i != ind2)
            indices1[i] = indices[i - skip];
          else
            ++skip;

        for (size_t i = 0; i < dimMin; ++i) {
          indices1[ind1] = indices1[ind2] = i;

          result[offset] += values[GetOffset(indices1)];
        }
      }
    }

    return result;
  }

  Tensor<T, Storage> Trace(const std::vector<size_t> &tindices) const {
    std::vector<size_t> newdims;

    size_t newsize = dims.size() - tindices.size();

    if (newsize == 0) {
      newsize = 1;
      newdims.push_back(1);
    } else {
      newdims.reserve(newsize);

      for (size_t i = 0; i < dims.size(); ++i) {
        bool skip = false;
        for (size_t j = 0; j < tindices.size(); ++j)
          if (i == tindices[j]) {
            skip = true;
            break;
          }

        if (!skip) newdims.push_back(dims[i]);
      }
    }

    size_t dimMin = dims[tindices[0]];
    for (size_t i = 1; i < tindices.size(); ++i)
      if (dims[tindices[i]] < dimMin) dimMin = dims[tindices[i]];

    Tensor<T, Storage> result(newdims, values.size() == 0);

    if (values.size() != 0) {
      for (size_t offset = 0; offset < result.GetSize(); ++offset) {
        std::vector<size_t> indices1(dims.size());
        const std::vector<size_t> indices = result.IndexFromOffset(offset);

        size_t skipv = 0;
        for (size_t i = 0; i < dims.size(); ++i) {
          bool skip = false;
          for (size_t j = 0; j < tindices.size(); ++j)
            if (i == tindices[j]) {
              skip = true;
              break;
            }

          if (!skip)
            indices1[i] = indices[i - skipv];
          else
            ++skipv;
        }

        for (size_t i = 0; i < dimMin; ++i) {
          for (size_t j = 0; j < tindices.size(); ++j)
            indices1[tindices[j]] = i;

          result[offset] += values[GetOffset(indices1)];
        }
      }
    }

    return result;
  }

  template <class indT = size_t>
  Tensor<T, Storage> Trace(std::initializer_list<indT> args) const {
    const std::vector<size_t> indices(args.begin(), args.end());

    return Trace(indices);
  }

  Tensor<T, Storage> Shuffle(const std::vector<size_t> &indices) const {
    assert(indices.size() == dims.size());

    std::vector<size_t> newdims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) newdims[i] = dims[indices[i]];

    Tensor<T, Storage> result(newdims, values.size() == 0);

    if (values.size() != 0) {
      std::vector<size_t> indices2(dims.size());
      for (size_t offset = 0; offset < GetSize(); ++offset) {
        const std::vector<size_t> indices1 = IndexFromOffset(offset);

        for (size_t i = 0; i < dims.size(); ++i)
          indices2[indices[i]] = indices1[i];

        result[indices2] = values[offset];
      }
    }

    return result;
  }

  template <class indT = size_t>
  Tensor<T, Storage> Shuffle(std::initializer_list<indT> args) const {
    const std::vector<size_t> indices(args.begin(), args.end());

    return Shuffle(indices);
  }

  Tensor<T, Storage> Reshape(const std::vector<size_t> &newdims) const {
    Tensor<T, Storage> result(newdims, values.size() == 0);

    assert(result.GetSize() == GetSize());

    if (values.size() != 0)
      for (size_t offset = 0; offset < GetSize(); ++offset)
        result[offset] = values[offset];

    return result;
  }

  template <class indT = size_t>
  Tensor<T, Storage> Reshape(std::initializer_list<indT> args) const {
    const std::vector<size_t> newdims(args.begin(), args.end());

    return Reshape(newdims);
  }

  // changing the layout format is as easy as changing what's called in the
  // following two functions

  inline size_t GetOffset(const std::vector<size_t> &indices) const {
    return GetFortranOffset(indices);
  }

  inline std::vector<size_t> IndexFromOffset(size_t offset) const {
    return IndexFromFortranOffset(offset);
  }

 private:
  // C/C++ layout (row-major order)

  inline size_t GetCPPOffset(const std::vector<size_t> &indices) const {
    assert(indices.size() == dims.size());

    size_t result = 0;

    for (size_t i = 0; i < dims.size(); ++i) {
      assert(indices[i] < dims[i]);

      result = result * dims[i] + indices[i];
    }

    return result;
  }

  inline std::vector<size_t> IndexFromCPPOffset(size_t offset) const {
    std::vector<size_t> indices(dims.size(), 0);

    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      indices[i] = offset % dims[i];
      offset /= dims[i];
      if (0 == offset) break;
    }

    return indices;
  }

  // the following two are for fortran layout (column-major order)
  // for passing to cuda we need probably to convert anyway (as in from
  // complex<double> to complex<float>) the tensors will be small so converting
  // them should not be computationally expensive

  inline size_t GetFortranOffset(const std::vector<size_t> &indices) const {
    assert(indices.size() == dims.size());

    size_t result = 0;

    for (long long int i = dims.size() - 1; i >= 0; --i) {
      assert(indices[i] < dims[i]);

      result = result * dims[i] + indices[i];
    }

    return result;
  }

  inline std::vector<size_t> IndexFromFortranOffset(size_t offset) const {
    std::vector<size_t> indices(dims.size(), 0);

    for (size_t i = 0; i < dims.size(); ++i) {
      indices[i] = offset % dims[i];
      offset /= dims[i];
      if (0 == offset) break;
    }

    return indices;
  }
};

}  // namespace Utils

#endif
