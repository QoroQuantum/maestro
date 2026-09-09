/**
 * @file Delay.h
 * @ingroup circuits
 * @brief Idle / delay operation for quantum circuits.
 */

#pragma once

#ifndef _CIRCUIT_DELAY_H_
#define _CIRCUIT_DELAY_H_

#include <cmath>
#include <stdexcept>
#include "Operations.h"

namespace Circuits {

/**
 * @class Delay
 * @brief Delay (idle) operation class.
 *
 * Represents a physical delay or idling period on a target qubit with duration tau.
 * In ideal noiseless simulation, it acts as an identity operation.
 * In noisy simulation, it triggers wall-clock thermal relaxation, continuous
 * Ornstein-Uhlenbeck dephasing, and coherent detuning rotation.
 * @tparam Time The time type used for operation timing.
 * @sa IOperation
 */
template <typename Time = Types::time_type>
class Delay : public IOperation<Time> {
 public:
  /**
   * @brief Construct a new Delay object.
   * @param qubit The target qubit.
   * @param duration The physical delay duration in seconds.
   */
  Delay(Types::qubit_t qubit = 0, Time duration = 0)
      : IOperation<Time>(duration), qubit_(qubit) {
    if (std::isnan(static_cast<double>(duration)) ||
        std::isinf(static_cast<double>(duration)) ||
        duration < 0) {
      throw std::invalid_argument("Delay duration must be finite and nonnegative");
    }
  }

  /**
   * @brief Execute the delay operation on the simulator.
   *
   * In noiseless simulation, delay does not alter the quantum state.
   */
  void Execute(const std::shared_ptr<Simulators::ISimulator> &sim,
               OperationState &state) const override {
    (void)sim;
    (void)state;
  }

  /**
   * @brief Get the operation type.
   */
  OperationType GetType() const override { return OperationType::kDelay; }

  /**
   * @brief Return the affected qubits.
   */
  Types::qubits_vector AffectedQubits() const override { return {qubit_}; }

  /**
   * @brief Get the target qubit.
   */
  Types::qubit_t GetQubit() const { return qubit_; }

  /**
   * @brief Set the target qubit.
   */
  void SetQubit(Types::qubit_t q) { qubit_ = q; }

  /**
   * @brief Get the delay duration.
   */
  Time GetDuration() const { return IOperation<Time>::GetDelay(); }

  /**
   * @brief Set the delay duration.
   */
  void SetDuration(Time d) { IOperation<Time>::SetDelay(d); }

  bool CanAffectQuantumState() const override { return false; }
  bool IsClifford() const override { return true; }

  std::shared_ptr<IOperation<Time>> Clone() const override {
    return std::make_shared<Delay<Time>>(qubit_, GetDuration());
  }

  std::shared_ptr<IOperation<Time>> Remap(
      const std::unordered_map<Types::qubit_t, Types::qubit_t> &qubitsMap,
      const std::unordered_map<Types::qubit_t, Types::qubit_t> &bitsMap = {})
      const override {
    (void)bitsMap;
    Types::qubit_t mappedQubit = qubit_;
    const auto it = qubitsMap.find(qubit_);
    if (it != qubitsMap.end()) mappedQubit = it->second;
    return std::make_shared<Delay<Time>>(mappedQubit, GetDuration());
  }

 private:
  Types::qubit_t qubit_ = 0;
};

}  // namespace Circuits

#endif  // _CIRCUIT_DELAY_H_
