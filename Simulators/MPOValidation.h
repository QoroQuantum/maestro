#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "../Types.h"

namespace Simulators::MPOValidation {
inline void Bits(const std::vector<bool>& bits, size_t width) {
  if (bits.size() != width)
    throw std::invalid_argument("MPO bit vector must match the register width");
}
inline void Pauli(const std::string& pauli, size_t width) {
  if (pauli.size() != width ||
      pauli.find_first_not_of("IXYZ") != std::string::npos)
    throw std::invalid_argument("Pauli string must have one I/X/Y/Z per qubit");
}
inline std::vector<int> Qubits(const Types::qubits_vector& qubits,
                               size_t width) {
  std::vector<int> result;
  for (auto q : qubits) {
    if (q >= width || q > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::out_of_range("Target qubit is out of range");
    if (std::find(result.begin(), result.end(), static_cast<int>(q)) !=
        result.end())
      throw std::invalid_argument("Target qubits must be distinct");
    result.push_back(static_cast<int>(q));
  }
  return result;
}
inline std::vector<int> Operator(const Types::qubits_vector& qubits,
                                 const Eigen::MatrixXcd& matrix, size_t width) {
  if (qubits.empty() || qubits.size() > 2 ||
      matrix.rows() != (Eigen::Index{1} << qubits.size()) ||
      matrix.cols() != matrix.rows() || !matrix.allFinite())
    throw std::invalid_argument(
        "Operator must be a finite 2x2 or 4x4 matrix matching its targets");
  return Qubits(qubits, width);
}
}  // namespace Simulators::MPOValidation
