#pragma once
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include "Simulator.h"

namespace Simulators {
// Shared validation for public language and C bindings. Generic operators need
// not be unitary; the simulator decides whether it supports the operation.
template <class Matrix>
void ValidateGenericGate(const ISimulator& simulator,
                         std::initializer_list<Types::qubit_t> targets,
                         const Matrix& matrix) {
  for (auto it = targets.begin(); it != targets.end(); ++it) {
    if (*it >= simulator.GetNumberOfQubits())
      throw std::invalid_argument("Generic gate target is out of range");
    if (std::find(targets.begin(), it, *it) != it)
      throw std::invalid_argument("Generic gate targets must be distinct");
  }
  const auto dimension = Eigen::Index{1} << targets.size();
  if (matrix.rows() != dimension || matrix.cols() != dimension)
    throw std::invalid_argument("Generic gate matrix has the wrong dimensions");
  if (!matrix.allFinite())
    throw std::invalid_argument(
        "Generic gate matrix must contain finite values");
}
}  // namespace Simulators
