#pragma once
#include "../../Circuit/Operations/QuantumGates.h"
#include "RoutingGateInfo.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
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

// A projection matters to routing when a multi-site operation follows it. Resets
// of known product sites leave the cuts alone. A measurement can also be the
// first operation of a suffix whose prefix entangled the measured qubits.
template <class OpPtr>
bool ContainsMidCircuitDisentangling(const std::vector<OpPtr> &operations, const std::vector<uint8_t> &untouched = {},
                                     const std::vector<double> &siteRanks = {})
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
        if (operations[i] && operations[i]->NeedsEntanglementForDistribution())
        {
            lastTwoQubitGate = i;
            break;
        }
    if (lastTwoQubitGate == operations.size())
        return false;
    if (operations[first]->GetType() != Circuits::OperationType::kReset)
        return true;
    // A fresh register starts in a product state. A suffix can instead start
    // with entanglement from its saved prefix, including on its first reset.
    std::vector<uint8_t> entangled(untouched.size());
    for (size_t q = 0; q < untouched.size(); ++q)
        entangled[q] = !untouched[q] && (q >= siteRanks.size() || siteRanks[q] > 1. + 1e-9);
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
        const auto width = static_cast<size_t>(*std::max_element(qubits.begin(), qubits.end())) + 1;
        if (entangled.size() < width)
            entangled.resize(width);
        if (type == Circuits::OperationType::kGate && qubits.size() == 2)
        {
            const auto *info = dynamic_cast<const RoutingGateInfo *>(op.get());
            const auto *gate = dynamic_cast<const Circuits::IQuantumGate<> *>(op.get());
            if (info ? info->isSwap : (gate && gate->GetGateType() == Circuits::QuantumGateType::kSwapGateType))
            {
                std::swap(entangled[qubits[0]], entangled[qubits[1]]);
                continue;
            }
        }
        for (const auto q : qubits)
            entangled[q] = true;
    }
    return false;
}

template <class Config> double TruncationWeight(const Config &config)
{
    return config.IsSet(kTruncationWeight) ? ParseWeight(config.GetConfiguration(kTruncationWeight)) : 1.;
}
} // namespace Simulators::RoutingConfig
