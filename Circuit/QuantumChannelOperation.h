/**
 * @file QuantumChannelOperation.h
 * @ingroup circuits
 * @brief Circuit operation for an exact local CPTP quantum channel.
 */

#pragma once

#ifndef _CIRCUIT_QUANTUM_CHANNEL_OPERATION_H_
#define _CIRCUIT_QUANTUM_CHANNEL_OPERATION_H_

#include <stdexcept>
#include <unordered_set>

#include "Operations.h"

namespace Circuits {

/** A one- or two-qubit channel that remains in Kraus form until execution. */
template <typename Time = Types::time_type>
class QuantumChannelOperation : public IOperation<Time> {
 public:
  QuantumChannelOperation(const Types::qubits_vector& targets,
                          const Simulators::QuantumChannel& channel,
                          Time delay = 0)
      : IOperation<Time>(delay), targets_(targets), channel_(channel) {
    if (targets_.size() != channel_.GetNumberOfQubits())
      throw std::invalid_argument(
          "The number of channel targets does not match its Kraus operators");
    if (targets_.empty() || targets_.size() > 2)
      throw std::invalid_argument(
          "Maestro circuit channels support only one or two qubits");

    std::unordered_set<Types::qubit_t> uniqueTargets;
    for (const Types::qubit_t target : targets_)
      if (!uniqueTargets.insert(target).second)
        throw std::invalid_argument(
            "Quantum-channel target qubits must be distinct");
  }

  void Execute(const std::shared_ptr<Simulators::ISimulator>& simulator,
               OperationState& state) const override {
    (void)state;
    simulator->ApplyQuantumChannel(targets_, channel_);
  }

  OperationType GetType() const override {
    return OperationType::kQuantumChannel;
  }

  Types::qubits_vector AffectedQubits() const override { return targets_; }

  bool CanAffectQuantumState() const override { return true; }

  bool NeedsEntanglementForDistribution() const override {
    return targets_.size() > 1;
  }

  const Simulators::QuantumChannel& GetChannel() const { return channel_; }

  std::shared_ptr<IOperation<Time>> Clone() const override {
    return std::make_shared<QuantumChannelOperation<Time>>(
        targets_, channel_, IOperation<Time>::GetDelay());
  }

  std::shared_ptr<IOperation<Time>> Remap(
      const std::unordered_map<Types::qubit_t, Types::qubit_t>& qubitsMap,
      const std::unordered_map<Types::qubit_t, Types::qubit_t>& bitsMap = {})
      const override {
    (void)bitsMap;
    Types::qubits_vector remappedTargets = targets_;
    for (Types::qubit_t& target : remappedTargets) {
      const auto mapped = qubitsMap.find(target);
      if (mapped != qubitsMap.end()) target = mapped->second;
    }
    return std::make_shared<QuantumChannelOperation<Time>>(
        remappedTargets, channel_, IOperation<Time>::GetDelay());
  }

 private:
  Types::qubits_vector targets_;
  Simulators::QuantumChannel channel_;
};

}  // namespace Circuits

#endif  // _CIRCUIT_QUANTUM_CHANNEL_OPERATION_H_
