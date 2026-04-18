/**
 * @file QiskitAerState.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The qiskit aer state class, derived from AER::AerState.
 *
 * Should not be used directly, this should allow exposing more from
 * AER::AerState than it's available through the public interface, if needed.
 */

#pragma once

#ifndef _QISKIT_AER_STATE_H_
#define _QISKIT_AER_STATE_H_

#ifndef NO_QISKIT_AER

#ifdef INCLUDED_BY_FACTORY

#include "controllers/state_controller.hpp"

namespace AER {

class AerStateFake {
 public:
  virtual ~AerStateFake() = default;

  bool initialized_;
  uint_t num_of_qubits_;
  RngEngine rng_;
  int seed_ = std::random_device()();
  std::shared_ptr<QuantumState::Base> state_;
  json_t configs_;
  ExperimentResult last_result_;
};

}  // namespace AER

namespace Simulators {
// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private {

/**
 * @class QiskitAerState
 * @brief Class for the qiskit aer simulator.
 *
 * Derived from AER::AerState, could add more functionality and expose more than
 * it's available through the base class public interface. Do not use this class
 * directly.
 */
class QiskitAerState : public AER::AerState {
 public:
  const std::shared_ptr<AER::QuantumState::Base> &get_state() const {
    const AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;
    return fakeState->state_;
  }

  double expval_pauli(const reg_t &qubits, const std::string &pauli) {
    if (qubits.empty() || pauli.empty()) return 1.;

    const auto &state = get_state();
    if (!state) return 0.;

    flush_ops();

    return state->expval_pauli(qubits, pauli);
  }

  std::vector<bool> apply_measure_many(const reg_t &qubits) {
    const auto &state = get_state();
    if (!state) return {};

    flush_ops();

    AER::Operations::Op op;
    op.type = AER::Operations::OpType::measure;
    op.name = "measure";
    op.qubits = qubits;
    op.memory = qubits;
    op.registers = qubits;

    AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;
    fakeState->last_result_ = AER::ExperimentResult();
    state->apply_op(op, fakeState->last_result_, fakeState->rng_);

    std::vector<bool> bitstring(qubits.size(), false);
    uint_t mem_size = state->creg().memory_size();
    for (size_t q = 0; q < qubits.size(); ++q) {
      const auto qubit = qubits[q];
      if (state->creg().creg_memory()[mem_size - qubit - 1] == '1')
        bitstring[q] = true;
    }
    return bitstring;
  }

  std::unordered_map<std::vector<bool>, uint_t> sample_counts_many(
      const reg_t &qubits, uint_t shots) {
    const auto &state = get_state();
    if (!state) return {};

    flush_ops();

    AER::AerStateFake *fakeState = (AER::AerStateFake *)(void *)this;

    // Aer's MPS backend returns samples sorted by ascending qubit index
    // regardless of the order of `qubits` (see sort_measured_values in
    // matrix_product_state_internal.cpp), while the statevector backend
    // returns them in the order of `qubits`. To get consistent behavior
    // across backends we sort the qubits ourselves and then remap the
    // bits back to the caller-requested order.
    reg_t sorted_qubits = qubits;
    std::vector<size_t> order(qubits.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&qubits](size_t a, size_t b) { return qubits[a] < qubits[b]; });
    for (size_t i = 0; i < qubits.size(); ++i)
      sorted_qubits[i] = qubits[order[i]];

    std::vector<AER::SampleVector> samples =
        state->sample_measure(sorted_qubits, shots, fakeState->rng_);
    std::unordered_map<std::vector<bool>, uint_t> ret;

    std::vector<bool> bitstring(qubits.size());
    for (const auto &sample : samples) {
      for (size_t i = 0; i < qubits.size(); ++i)
        bitstring[order[i]] = sample[i] == 1;

      ++ret[bitstring];
    }
    return ret;
  }
};

}  // namespace Private
}  // namespace Simulators

#endif
#endif
#endif
