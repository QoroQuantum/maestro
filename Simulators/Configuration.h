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

#include <stdexcept>
#include <string>
#include <unordered_map>


#include "State.h"
#include <charconv>
#include <stdexcept>

namespace Simulators {


class Configuration {
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
  Configuration(const Configuration&) = default;

  /**
   * @brief The move constructor.
   *
   * The move constructor for the simulator configuration.
   */
  Configuration(Configuration&&) = default;

  /**
   * @brief The copy assignment operator.
   *
   * The copy assignment operator for the simulator configuration.
   */
  Configuration& operator=(const Configuration&) = default;

  /**
   * @brief The move assignment operator.
   *
   * The move assignment operator for the simulator configuration.
   */
  Configuration& operator=(Configuration&&) = default;

  /**
   * @brief Set a configuration value.
   *
   * Set a configuration value for the simulator.
   *
   * @param key The key of the configuration value.
   * @param value The value of the configuration.
   */
  static int ParseGpuDevice(const std::string& value) {
    int device = -1;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), device);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size() || device < 0)
      throw std::invalid_argument("gpu_device must be a nonnegative integer");
    return device;
  }

  // Keep mutually exclusive GPU SVD flags consistent before they are replayed
  // from an unordered map into a new simulator (including network clones/jobs).
  static std::string GpuSvdSettingGroup(const std::string& key) {
    for (const char* group : {"matrix_product_state_use_gesvd",
                              "matrix_product_operator_use_gesvd",
                              "tensor_network_use_gesvd"}) {
      const size_t length = std::char_traits<char>::length(group);
      if (key == group) return group;
      if (key.size() == length + 1 && key.compare(0, length, group) == 0 &&
          (key.back() == 'j' || key.back() == 'p' || key.back() == 'r'))
        return group;
    }
    return {};
  }

  static bool ParseGpuSvdFlag(const std::string& value) {
    if (value == "true" || value == "1") return true;
    if (value == "false" || value == "0") return false;
    throw std::invalid_argument("GPU SVD flags must be true, false, 1 or 0");
  }

  void SetConfiguration(const std::string& key, const std::string& value) {
    if (key == "gpu_device") ParseGpuDevice(value);
    const auto svdGroup = GpuSvdSettingGroup(key);
    if (!svdGroup.empty()) {
      const bool enabled = ParseGpuSvdFlag(value);
      configMap[key] = enabled ? "true" : "false";
      if (enabled) {
        for (char algorithm : {'j', 'p', 'r'}) {
          const auto otherKey = svdGroup + algorithm;
          const auto other = configMap.find(otherKey);
          // Do not introduce unused optional settings on older plugins, or
          // invalidate iterators while an existing map is being replayed.
          if (otherKey != key && other != configMap.end()) other->second = "false";
        }
      }
      return;
    }
    if ((key == "matrix_product_state_truncation_mode" ||
         key == "matrix_product_operator_truncation_mode") &&
        value != "relative_max" && value != "discarded_weight")
      throw std::invalid_argument(
          "Invalid truncation mode: expected relative_max or "
          "discarded_weight");

    // specially handled, ignore it
    if (IgnoredSetting(key)) return;
    configMap[key] = value;
  }

  void SetConfiguration(const std::string& key, long long int value) {
    // specially handled, ignore it
    if (IgnoredSetting(key)) return;
    SetConfiguration(key, std::to_string(value));
  }

  void SetConfiguration(const std::string& key, double value) {
    // specially handled, ignore it
    if (IgnoredSetting(key)) return;
    SetConfiguration(key, std::to_string(value));
  }

  /**
   * @brief Check if a configuration value is set.
   *
   * Check if a configuration value is set for the simulator.
   *
   * @param key The key of the configuration value.
   * @return True if the configuration value is set, false otherwise.
   */
  bool IsSet(const std::string& key) const {
    return configMap.find(key) != configMap.end();
  }

  /**
   * @brief Get a configuration value.
   *
   * Get a configuration value for the simulator.
   *
   * @param key The key of the configuration value.
   * @return The configuration value as a string.
   */
  std::string GetConfiguration(const std::string& key) const {
    auto it = configMap.find(key);
    if (it != configMap.end()) {
      return it->second;
    }
    return "";
  }

  long long int GetConfigurationAsInt(const std::string& key) const {
    auto it = configMap.find(key);
    if (it != configMap.end()) {
      return std::stoll(it->second);
    }
    return 0;
  }

  // For size_t-valued settings: the signed and double getters cannot represent
  // the whole range, and a value near SIZE_MAX misconverts through double.
  unsigned long long int GetConfigurationAsUnsigned(
      const std::string& key) const {
    auto it = configMap.find(key);
    if (it != configMap.end()) {
      return std::stoull(it->second);
    }
    return 0;
  }

  double GetConfigurationAsDouble(const std::string& key) const {
    auto it = configMap.find(key);
    if (it != configMap.end()) {
      return std::stod(it->second);
    }
    return 0.0;
  }

  static bool CanBeAppliedOnInitializedSimulator(const std::string& key) {
    if (key == "gpu_device" || key == "method" || key == "use_double_precision" ||
        key == "precision" ||
        key == "max_parallel_threads" || key == "parallel_state_update" ||
        key == "statevector_parallel_threshold")
      return false;
    
    return true;
  }

  static bool IgnoredSetting(const std::string& key) {
    if (key == "max_simulators" || key == "method")
      return true;

    return false;
  }

  void ApplyConfigurationToSimulator(const std::shared_ptr<Simulators::IState>& simulator) const {
    for (const auto& [key, value] : configMap)
      simulator->Configure(key.c_str(), value.c_str());
  }

  void ApplyConfigurationFromSimulator(const std::shared_ptr<Simulators::IState>& simulator) {
    if (!simulator) return;
    ApplyConfigurationFromMap(simulator->GetConfigMap());
  }

  void ApplyConfigurationFromMap(const std::unordered_map<std::string, std::string>& config) {
    for (const auto& [key, value] : config) SetConfiguration(key, value);
  }

  const std::unordered_map<std::string, std::string>& GetConfigMap() const {
    return configMap;
  }

  bool WasApplied(const std::string& key, const std::string& value) const {
    auto it = configMap.find(key);
    if (it != configMap.end())
      return it->second == value;

    return false;
  }

  private:
  std::unordered_map<std::string, std::string>
      configMap; /**< The configuration map. */
};

}

#endif

