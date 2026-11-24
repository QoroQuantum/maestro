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

template <class Config> double TruncationWeight(const Config &config)
{
    return config.IsSet(kTruncationWeight) ? ParseWeight(config.GetConfiguration(kTruncationWeight)) : 1.;
}
} // namespace Simulators::RoutingConfig
