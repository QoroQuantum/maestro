#pragma once

#include <Eigen/Dense>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../../Types.h"

namespace Simulators::TensorQueries {
// Check shifts and host buffer arithmetic before attempting dense allocation.
inline size_t DenseSize(size_t qubits) {
  if (qubits >= std::numeric_limits<size_t>::digits)
    throw std::length_error("Full state output exceeds the basis-index range");
  const size_t count = size_t{1} << qubits;
  if (count > std::vector<std::complex<double>>().max_size() ||
      count > std::vector<double>().max_size() / 2)
    throw std::length_error("Full state output is too large");
  return count;
}

inline int Count(size_t count) {
  if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("Tensor query exceeds the backend count limit");
  return static_cast<int>(count);
}

inline void Operators(const Types::qubits_vector& qubits,
                      const std::vector<Eigen::MatrixXcd>& matrices,
                      size_t width) {
  if (qubits.size() != matrices.size())
    throw std::invalid_argument("Each operator must have one target qubit");
  Count(qubits.size());
  if (matrices.size() > std::vector<double>().max_size() / 8)
    throw std::length_error("Operator matrices exceed the buffer size limit");
  for (size_t i = 0; i < qubits.size(); ++i) {
    if (qubits[i] >= width)
      throw std::out_of_range("Operator target qubit is out of range");
    if (matrices[i].rows() != 2 || matrices[i].cols() != 2 ||
        !matrices[i].allFinite())
      throw std::invalid_argument(
          "Expectation operators must be finite 2x2 matrices");
  }
}

// Owns only pointers into the input strings, which must outlive the native
// call.
struct PauliBatch {
  std::vector<const char*> strings;
  std::vector<int> lengths;
  explicit PauliBatch(const std::vector<std::string>& paulis) {
    Count(paulis.size());
    strings.reserve(paulis.size());
    lengths.reserve(paulis.size());
    for (const auto& pauli : paulis) {
      strings.push_back(pauli.c_str());
      lengths.push_back(Count(pauli.size()));
    }
  }
};
}  // namespace Simulators::TensorQueries
