/**
 * @file CausalCone.h
 * @brief Exact backward lightcone extraction for unitary circuits.
 */
#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "Circuit.h"

namespace Circuits {

template <typename Time = Types::time_type>
struct ReducedObservableCone {
  static constexpr size_t inactive_qubit = std::numeric_limits<size_t>::max();
  std::shared_ptr<Circuit<Time>> reduced_circuit;
  std::string reduced_pauli_string;
  // Original -> compact index; inactive_qubit marks discarded wires.
  std::vector<size_t> active_qubits_map;
  size_t GetNumberOfQubits() const { return reduced_pauli_string.size(); }
};

// A quantum-only cone cannot account for classical feed-forward or sampled
// measurement/reset trajectories. Channels may also require a mixed-state
// backend. Preserve full execution for these operations and composite circuits.
template <typename Time = Types::time_type>
bool SupportsObservableCone(const std::shared_ptr<Circuit<Time>>& circuit) {
  if (!circuit) return false;
  for (const auto& op : circuit->GetOperations()) {
    if (!op) return false;
    switch (op->GetType()) {
      case OperationType::kGate:
      case OperationType::kNoOp:
      case OperationType::kDelay:
        break;
      default:
        return false;
    }
  }
  return true;
}

// O(M + N) for bounded-arity gates and an N-character observable/register.
// Unsupported circuits return the original circuit and an identity mapping.
// An identity observable has an empty cone and expectation one.
template <typename Time = Types::time_type>
ReducedObservableCone<Time> ExtractObservableCone(
    const std::shared_ptr<Circuit<Time>>& circuit,
    const std::string& pauli_string) {
  const size_t width = std::max(pauli_string.size(),
      circuit ? circuit->GetMaxQubitIndex() + 1 : size_t{0});
  ReducedObservableCone<Time> cone;
  cone.active_qubits_map.assign(width, ReducedObservableCone<Time>::inactive_qubit);
  if (!SupportsObservableCone(circuit)) {
    cone.reduced_circuit = circuit;
    cone.reduced_pauli_string = pauli_string;
    cone.reduced_pauli_string.resize(width, 'I');
    for (size_t q = 0; q < width; ++q) cone.active_qubits_map[q] = q;
    return cone;
  }

  std::vector<bool> active(width, false);
  for (size_t q = 0; q < pauli_string.size(); ++q)
    active[q] = pauli_string[q] != 'I' && pauli_string[q] != 'i';

  typename Circuit<Time>::OperationsVector retained;
  retained.reserve(circuit->GetOperations().size());
  const auto& ops = circuit->GetOperations();
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    const auto qubits = (*it)->AffectedQubits();
    if (std::any_of(qubits.begin(), qubits.end(),
                    [&](size_t q) { return active[q]; })) {
      retained.push_back(*it);
      for (auto q : qubits) active[q] = true;
    }
  }

  typename Circuit<Time>::BitMapping mapping;
  mapping.reserve(width);
  cone.reduced_pauli_string.reserve(width);
  for (size_t q = 0; q < width; ++q) {
    if (!active[q]) continue;
    const size_t compact = cone.reduced_pauli_string.size();
    cone.active_qubits_map[q] = compact;
    mapping[q] = compact;
    cone.reduced_pauli_string.push_back(q < pauli_string.size() ? pauli_string[q] : 'I');
  }

  typename Circuit<Time>::OperationsVector remapped;
  remapped.reserve(retained.size());
  for (auto it = retained.rbegin(); it != retained.rend(); ++it)
    remapped.push_back((*it)->Remap(mapping, {}));
  cone.reduced_circuit = std::make_shared<Circuit<Time>>(remapped);
  return cone;
}

}  // namespace Circuits
