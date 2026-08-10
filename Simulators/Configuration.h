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

#include <string>
#include <unordered_map>

#include "State.h"

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
  void SetConfiguration(const std::string& key, const std::string& value) {
    configMap[key] = value;
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

  void ApplyConfigurationToSimulator(std::shared_ptr<IState> simulator) const {
    for (const auto& [key, value] : configMap)
        simulator->Configure(key.c_str(), value.c_str());
  }

  private:
  std::unordered_map<std::string, std::string>
      configMap; /**< The configuration map. */
};

}

#endif


