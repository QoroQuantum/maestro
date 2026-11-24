#pragma once
#include "../../Circuit/Operations/Operations.h"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace Simulators::RoutingConfig
{
inline constexpr const char *kStrategy = "mps_routing_strategy";
inline constexpr const char *kVirtualBonds = "mps_routing_virtual_bonds";
inline constexpr const char *kLegacyCap = "mps_routing_legacy_cap";
inline constexpr const char *kTruncationWeight = "mps_routing_truncation_weight";

enum class Strategy
{
    Combined,
    Legacy
};

inline Strategy ParseStrategy(const std::string &value)
{
    if (value == "combined")
        return Strategy::Combined;
    if (value == "legacy")
        return Strategy::Legacy;
    throw std::invalid_argument("mps_routing_strategy must be combined or legacy");
}

template <class Config> Strategy GetStrategy(const Config &config)
{
    return config.IsSet(kStrategy) ? ParseStrategy(config.GetConfiguration(kStrategy)) : Strategy::Legacy;
}

inline bool ParseFlag(const std::string &value)
{
    if (value == "true" || value == "1")
        return true;
    if (value == "false" || value == "0")
        return false;
    throw std::invalid_argument("Routing flags must be true, false, 1 or 0");
}

inline double ParseWeight(const std::string &value)
{
    size_t consumed = 0;
    const double weight = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(weight) || weight < 0.)
        throw std::invalid_argument("mps_routing_truncation_weight must be finite and nonnegative");
    return weight;
}

template <class Config> bool VirtualBondsEnabled(const Config &config)
{
    // The old opt-out remains an override for existing C++ callers.
    if (config.IsSet(kLegacyCap) && ParseFlag(config.GetConfiguration(kLegacyCap)))
        return false;
    return config.IsSet(kVirtualBonds) && ParseFlag(config.GetConfiguration(kVirtualBonds));
}

// When mid-circuit disentangling operations (measurements or resets) are present,
// bond dimensions oscillate across syndrome rounds instead of saturating globally.
// Auto-enabling virtual-bond truncation overflow pricing protects accuracy unless
// explicitly disabled by mps_routing_legacy_cap or mps_routing_virtual_bonds=false.
template <class Config> bool VirtualBondsAutoAllowed(const Config &config)
{
    if (config.IsSet(kLegacyCap) && ParseFlag(config.GetConfiguration(kLegacyCap)))
        return false;
    if (config.IsSet(kVirtualBonds))
        return ParseFlag(config.GetConfiguration(kVirtualBonds));
    return true;
}

inline bool IsDisentanglingOperationType(Circuits::OperationType type) noexcept
{
    return type == Circuits::OperationType::kMeasurement || type == Circuits::OperationType::kConditionalMeasurement || type == Circuits::OperationType::kReset;
}

template <class OpPtr> bool ContainsDisentanglingOperations(const std::vector<OpPtr> &operations) noexcept
{
    for (const auto &op : operations)
        if (op && IsDisentanglingOperationType(op->GetType()))
            return true;
    return false;
}

// Whether the operations hold a measurement or reset that two-qubit gates still follow and that can change
// the entanglement of the chain: a measurement before the last two-qubit gate, or such a reset of a qubit that
// an operation on several qubits acted on since its last reset. Final measurements do not count (nothing is
// routed after them). Nor do resets of qubits that are still products: the resets a circuit starts with, however
// often they repeat, and a reset after one-qubit gates only. A measurement counts whatever came before it: the
// operations can be the rest of a circuit (a checkpoint suffix, a routing window), whose first measurements
// act on qubits entangled earlier.
template <class OpPtr> bool ContainsMidCircuitDisentangling(const std::vector<OpPtr> &operations)
{
    // Most circuits hold none before their end: find the first by type alone, without reading a qubit list.
    size_t first = operations.size();
    for (size_t i = 0; i < operations.size(); ++i)
        if (operations[i] && IsDisentanglingOperationType(operations[i]->GetType()))
        {
            first = i;
            break;
        }
    if (first == operations.size())
        return false;
    size_t lastTwoQubitGate = operations.size();
    for (size_t i = operations.size(); i-- > first;)
        if (operations[i] && operations[i]->GetType() == Circuits::OperationType::kGate && operations[i]->AffectedQubits().size() >= 2)
        {
            lastTwoQubitGate = i;
            break;
        }
    if (lastTwoQubitGate == operations.size())
        return false;
    // Only a reset asks whether its qubit can be entangled: whether an operation on several qubits acted on it.
    std::vector<bool> entangled;
    for (size_t i = 0; i < lastTwoQubitGate; ++i)
    {
        const auto &op = operations[i];
        if (!op)
            continue;
        const auto type = op->GetType();
        if (type == Circuits::OperationType::kMeasurement || type == Circuits::OperationType::kConditionalMeasurement)
            return true;
        if (type == Circuits::OperationType::kDelay || type == Circuits::OperationType::kNoOp)
            continue;
        const auto qubits = op->AffectedQubits();
        if (type == Circuits::OperationType::kReset)
        {
            // A reset leaves its qubit a product state, so it marks nothing.
            for (const auto q : qubits)
                if (q < entangled.size() && entangled[q])
                    return true;
            continue;
        }
        if (qubits.size() < 2)
            continue; // a one-qubit operation keeps a product state a product state
        for (const auto q : qubits)
        {
            if (q >= entangled.size())
                entangled.resize(static_cast<size_t>(q) + 1, false);
            entangled[q] = true;
        }
    }
    return false;
}

template <class Config> double TruncationWeight(const Config &config)
{
    return config.IsSet(kTruncationWeight) ? ParseWeight(config.GetConfiguration(kTruncationWeight)) : 1.;
}
} // namespace Simulators::RoutingConfig
