#pragma once
#include <cmath>
#include <stdexcept>
#include <string>

namespace Simulators::RoutingConfig
{
inline constexpr const char *kVirtualBonds = "mps_routing_virtual_bonds";
inline constexpr const char *kLegacyCap = "mps_routing_legacy_cap";
inline constexpr const char *kTruncationWeight = "mps_routing_truncation_weight";

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

template <class Config> double TruncationWeight(const Config &config)
{
    return config.IsSet(kTruncationWeight) ? ParseWeight(config.GetConfiguration(kTruncationWeight)) : 1.;
}
} // namespace Simulators::RoutingConfig
