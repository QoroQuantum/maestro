/**
 * @file Configuration.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The simulator configuration class.
 */

#pragma once

#ifndef _SIMULATOR_CONFIGURATION_H_
#define _SIMULATOR_CONFIGURATION_H_

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "../Interfaces/State.h"
#include "../TensorNetworks/RoutingConfig.h"
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace Simulators
{

// The settings of the two extended stabilizers. The simulators that read them
// and the layers that accept them (SimulatorConfig, native requests, the legacy
// C calls) check a value with the same function, so one rule decides it.
namespace ExtendedStabilizerConfig
{
// QCSim prunes its components. Unset (both), it runs exactly.
inline constexpr const char *kAmplitudeTolerance = "extended_stabilizer_amplitude_tolerance";
inline constexpr const char *kMaxComponents = "extended_stabilizer_max_components";
// Qiskit Aer samples a stabilizer-rank decomposition whose size follows from
// the approximation error, and samples outcomes from it. Unset, a setting
// keeps Aer's default.
inline constexpr const char *kApproximationError = "extended_stabilizer_approximation_error";
inline constexpr const char *kSamplingMethod = "extended_stabilizer_sampling_method";
inline constexpr const char *kMetropolisMixingTime = "extended_stabilizer_metropolis_mixing_time";
inline constexpr const char *kNormEstimationSamples = "extended_stabilizer_norm_estimation_samples";
inline constexpr const char *kNormEstimationRepetitions = "extended_stabilizer_norm_estimation_repetitions";

inline constexpr std::array<const char *, 2> kPruningKeys{kAmplitudeTolerance, kMaxComponents};
inline constexpr std::array<const char *, 5> kAerKeys{kApproximationError, kSamplingMethod, kMetropolisMixingTime, kNormEstimationSamples,
                                                      kNormEstimationRepetitions};
inline constexpr std::string_view kPrefix = "extended_stabilizer_";

inline bool IsPruningKey(std::string_view key)
{
    return key == kAmplitudeTolerance || key == kMaxComponents;
}

inline bool IsAerCountKey(std::string_view key)
{
    return key == kMetropolisMixingTime || key == kNormEstimationSamples || key == kNormEstimationRepetitions;
}

inline bool IsAerKey(std::string_view key)
{
    return key == kApproximationError || key == kSamplingMethod || IsAerCountKey(key);
}

inline const std::vector<std::string> &SamplingMethods()
{
    static const std::vector<std::string> values{"resampled_metropolis", "metropolis", "norm_estimation"};
    return values;
}

// Aer keeps its integer settings in an int.
inline constexpr uint64_t kMaxAerCount = static_cast<uint64_t>(std::numeric_limits<int>::max());

// The range of each setting: an empty string for a valid value, otherwise the
// error message.
inline std::string AmplitudeToleranceError(double value)
{
    return std::isfinite(value) && value >= 0 && value < 1 ? "" : "extended_stabilizer_amplitude_tolerance must be finite and in [0, 1)";
}

inline std::string MaxComponentsError(uint64_t value)
{
    return value ? "" : "extended_stabilizer_max_components must be a positive integer";
}

inline std::string ApproximationErrorError(double value)
{
    return std::isfinite(value) && value > 0 && value <= 1 ? "" : "extended_stabilizer_approximation_error must be finite and in (0, 1]";
}

inline std::string SamplingMethodError(const std::string &value)
{
    for (const auto &method : SamplingMethods())
        if (value == method)
            return "";
    return "extended_stabilizer_sampling_method must be one of 'resampled_metropolis', 'metropolis', 'norm_estimation'; got '" + value + "'.";
}

inline std::string AerCountError(std::string_view key, uint64_t value)
{
    return value >= 1 && value <= kMaxAerCount ? "" : std::string(key) + " must be between 1 and " + std::to_string(kMaxAerCount);
}

// The whole text as a number, without surrounding spaces. strtod rather than
// std::stod: a subnormal value is a number, not a range error.
inline bool ParseNumber(const std::string &text, double &value)
{
    if (text.empty() || text.front() == ' ' || text.front() == '\t' || text.front() == '\n')
        return false;
    char *end = nullptr;
    value = std::strtod(text.c_str(), &end);
    return end == text.c_str() + text.size();
}

inline bool ParseCount(const std::string &text, uint64_t &value)
{
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc() && parsed.ptr == text.data() + text.size();
}

// Throws the message of a failed check; an empty one passes.
inline void Reject(const std::string &error)
{
    if (!error.empty())
        throw std::invalid_argument(error);
}

// The value of a setting from its text. Each throws std::invalid_argument for
// text that is not a valid value.
inline double ParseAmplitudeTolerance(const std::string &text)
{
    double value = 0;
    Reject(ParseNumber(text, value) ? AmplitudeToleranceError(value) : AmplitudeToleranceError(-1));
    return value;
}

inline uint64_t ParseMaxComponents(const std::string &text)
{
    uint64_t value = 0;
    Reject(ParseCount(text, value) ? MaxComponentsError(value) : MaxComponentsError(0));
    return value;
}

inline double ParseApproximationError(const std::string &text)
{
    double value = 0;
    Reject(ParseNumber(text, value) ? ApproximationErrorError(value) : ApproximationErrorError(-1));
    return value;
}

inline uint64_t ParseAerCount(std::string_view key, const std::string &text)
{
    uint64_t value = 0;
    Reject(ParseCount(text, value) ? AerCountError(key, value) : AerCountError(key, 0));
    return value;
}

// Checks the value of an extended stabilizer setting. Any other key passes.
inline void Validate(std::string_view key, const std::string &value)
{
    if (key.substr(0, kPrefix.size()) != kPrefix)
        return;
    if (key == kAmplitudeTolerance)
        ParseAmplitudeTolerance(value);
    else if (key == kMaxComponents)
        ParseMaxComponents(value);
    else if (key == kApproximationError)
        ParseApproximationError(value);
    else if (key == kSamplingMethod)
        Reject(SamplingMethodError(value));
    else if (IsAerCountKey(key))
        ParseAerCount(key, value);
}
} // namespace ExtendedStabilizerConfig

class Configuration
{
  public:
    /**
     * @brief The constructor.
     *
     * The constructor for the simulator configuration.
     */
    Configuration() = default;

    /**
     * @brief The destructor.
     *
     * The destructor for the simulator configuration.
     */
    virtual ~Configuration() = default;

    /**
     * @brief The copy constructor.
     *
     * The copy constructor for the simulator configuration.
     */
    Configuration(const Configuration &) = default;

    /**
     * @brief The move constructor.
     *
     * The move constructor for the simulator configuration.
     */
    Configuration(Configuration &&) = default;

    /**
     * @brief The copy assignment operator.
     *
     * The copy assignment operator for the simulator configuration.
     */
    Configuration &operator=(const Configuration &) = default;

    /**
     * @brief The move assignment operator.
     *
     * The move assignment operator for the simulator configuration.
     */
    Configuration &operator=(Configuration &&) = default;

    /**
     * @brief Set a configuration value.
     *
     * Set a configuration value for the simulator.
     *
     * @param key The key of the configuration value.
     * @param value The value of the configuration.
     */
    static int ParseGpuDevice(const std::string &value)
    {
        int device = -1;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), device);
        if (result.ec != std::errc() || result.ptr != value.data() + value.size() || device < 0)
            throw std::invalid_argument("gpu_device must be a nonnegative integer");
        return device;
    }

    // Keep mutually exclusive GPU SVD flags consistent before they are replayed
    // from an unordered map into a new simulator (including network clones/jobs).
    static std::string GpuSvdSettingGroup(const std::string &key)
    {
        for (const char *group : {"matrix_product_state_use_gesvd", "matrix_product_operator_use_gesvd", "tensor_network_use_gesvd"})
        {
            const size_t length = std::char_traits<char>::length(group);
            if (key == group)
                return group;
            if (key.size() == length + 1 && key.compare(0, length, group) == 0 && (key.back() == 'j' || key.back() == 'p' || key.back() == 'r'))
                return group;
        }
        return {};
    }

    static bool ParseGpuSvdFlag(const std::string &value)
    {
        if (value == "true" || value == "1")
            return true;
        if (value == "false" || value == "0")
            return false;
        throw std::invalid_argument("GPU SVD flags must be true, false, 1 or 0");
    }

    static bool ParsePrecision(const std::string &key, const std::string &value)
    {
        if (key == "precision")
        {
            if (value == "double")
                return true;
            if (value == "single")
                return false;
            throw std::invalid_argument("precision must be single or double");
        }
        if (value == "true" || value == "1")
            return true;
        if (value == "false" || value == "0")
            return false;
        throw std::invalid_argument("use_double_precision must be true, false, 1 or 0");
    }

    void SetConfiguration(const std::string &key, const std::string &value)
    {
        if (key == "matrix_product_operator_truncation_threshold")
        {
            SetConfiguration("matrix_product_state_truncation_threshold", value);
            return;
        }
        if (key == RoutingConfig::kStrategy)
            RoutingConfig::ParseStrategy(value);
        else if (key == kMeasurementOrderOption)
            IsWrittenMeasurementOrder(value);
        else if (key == RoutingConfig::kVirtualBonds || key == RoutingConfig::kLegacyCap)
            RoutingConfig::ParseFlag(value);
        else if (key == RoutingConfig::kTruncationWeight)
            RoutingConfig::ParseWeight(value);
        // Before the value is stored: a network keeps what it is given and
        // replays it to every simulator it creates.
        ExtendedStabilizerConfig::Validate(key, value);
        if (key == "precision" || key == "use_double_precision")
        {
            const bool useDouble = ParsePrecision(key, value);
            configMap["precision"] = useDouble ? "double" : "single";
            configMap["use_double_precision"] = useDouble ? "1" : "0";
            return;
        }
        if (key == "gpu_device")
            ParseGpuDevice(value);
        const auto svdGroup = GpuSvdSettingGroup(key);
        if (!svdGroup.empty())
        {
            const bool enabled = ParseGpuSvdFlag(value);
            configMap[key] = enabled ? "true" : "false";
            if (enabled)
            {
                for (const char *algorithm : {"", "j", "p", "r"})
                {
                    const auto otherKey = svdGroup + algorithm;
                    const auto other = configMap.find(otherKey);
                    // Do not introduce unused optional settings on older plugins, or
                    // invalidate iterators while an existing map is being replayed.
                    if (otherKey != key && other != configMap.end())
                        other->second = "false";
                }
            }
            return;
        }
        if ((key == "matrix_product_state_truncation_mode" || key == "matrix_product_operator_truncation_mode") && value != "relative_max" &&
            value != "discarded_weight")
            throw std::invalid_argument("Invalid truncation mode: expected relative_max or "
                                        "discarded_weight");

        // specially handled, ignore it
        if (IgnoredSetting(key))
            return;
        configMap[key] = value;
    }

    void SetConfiguration(const std::string &key, long long int value)
    {
        // specially handled, ignore it
        if (IgnoredSetting(key))
            return;
        SetConfiguration(key, std::to_string(value));
    }

    void SetConfiguration(const std::string &key, double value)
    {
        // specially handled, ignore it
        if (IgnoredSetting(key))
            return;
        // std::to_string rounds small cutoffs (including 1e-15) to zero.
        std::ostringstream text;
        text << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        SetConfiguration(key, text.str());
    }

    /**
     * @brief Remove a configuration value.
     *
     * Remove a configuration value, if set, so the simulator default applies.
     *
     * @param key The key of the configuration value.
     */
    void RemoveConfiguration(const std::string &key)
    {
        configMap.erase(CanonicalKey(key));
    }

    /**
     * @brief Check if a configuration value is set.
     *
     * Check if a configuration value is set for the simulator.
     *
     * @param key The key of the configuration value.
     * @return True if the configuration value is set, false otherwise.
     */
    bool IsSet(const std::string &key) const
    {
        return configMap.find(CanonicalKey(key)) != configMap.end();
    }

    /**
     * @brief Get a configuration value.
     *
     * Get a configuration value for the simulator.
     *
     * @param key The key of the configuration value.
     * @return The configuration value as a string.
     */
    std::string GetConfiguration(const std::string &key) const
    {
        auto it = configMap.find(CanonicalKey(key));
        if (it != configMap.end())
        {
            return it->second;
        }
        return "";
    }

    long long int GetConfigurationAsInt(const std::string &key) const
    {
        auto it = configMap.find(CanonicalKey(key));
        if (it != configMap.end())
        {
            return std::stoll(it->second);
        }
        return 0;
    }

    // For size_t-valued settings: the signed and double getters cannot represent
    // the whole range, and a value near SIZE_MAX misconverts through double.
    unsigned long long int GetConfigurationAsUnsigned(const std::string &key) const
    {
        auto it = configMap.find(CanonicalKey(key));
        if (it != configMap.end())
        {
            return std::stoull(it->second);
        }
        return 0;
    }

    double GetConfigurationAsDouble(const std::string &key) const
    {
        auto it = configMap.find(CanonicalKey(key));
        if (it != configMap.end())
        {
            return std::stod(it->second);
        }
        return 0.0;
    }

    static bool CanBeAppliedOnInitializedSimulator(const std::string &key)
    {
        if (key == "gpu_device" || key == "method" || key == "use_double_precision" || key == "precision" || key == "max_parallel_threads" ||
            key == "parallel_state_update" || key == "statevector_parallel_threshold")
            return false;

        return true;
    }

    static bool IgnoredSetting(const std::string &key)
    {
        if (key == "max_simulators" || key == "method")
            return true;

        return false;
    }

    void ApplyConfigurationToSimulator(const std::shared_ptr<Simulators::IState> &simulator) const
    {
        for (const auto &[key, value] : configMap)
        {
            if (key == "gate_fusion" && simulator->GetType() != SimulatorType::kQCSim && simulator->GetType() != SimulatorType::kCompositeQCSim &&
                !IsGpuSimulator(simulator->GetType()))
                continue;

            // Networks retain GPU placement when an optimizer selects a CPU backend.
            // Replay it only to GPU simulators; Aer rejects this plugin-specific key.
            if (key == "gpu_device" && !IsGpuSimulator(simulator->GetType()))
                continue;

            // Route the seed through SetSeed, whose contract is to seed EVERY random
            // stream the simulator owns -- including the auxiliary one used for
            // measurement-time readout error. Its default implementation still
            // records the value with Configure, so this subsumes the plain call.
            if (key == "seed")
            {
                try
                {
                    simulator->SetSeed(std::stoull(value));
                    continue;
                }
                catch (const std::exception &)
                {
                    // Not a number: fall through and let the simulator reject it.
                }
            }

            simulator->Configure(key.c_str(), value.c_str());
        }
    }

    void ApplyConfigurationFromSimulator(const std::shared_ptr<Simulators::IState> &simulator)
    {
        if (!simulator)
            return;
        ApplyConfigurationFromMap(simulator->GetExplicitConfigMap());
    }

    void ApplyConfigurationFromMap(const std::unordered_map<std::string, std::string> &config)
    {
        for (const auto &[key, value] : config)
            SetConfiguration(key, value);
    }

    const std::unordered_map<std::string, std::string> &GetConfigMap() const
    {
        return configMap;
    }

    bool WasApplied(const std::string &key, const std::string &value) const
    {
        auto it = configMap.find(CanonicalKey(key));
        if (it != configMap.end())
            return it->second == value;

        return false;
    }

    // Store one spelling so aliases work before capture and replay is ordered
    // by the caller's updates, never by unordered-map iteration.
    static std::string CanonicalKey(const std::string &key)
    {
        if (key == "matrix_product_operator_truncation_threshold")
            return "matrix_product_state_truncation_threshold";
        return key;
    }

  private:
    std::unordered_map<std::string, std::string> configMap; /**< The configuration map. */
};

} // namespace Simulators

#endif
