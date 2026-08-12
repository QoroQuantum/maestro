/**
 * @file Configuration.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The network configuration class.
 */

#pragma once

#ifndef _NETWORK_CONFIGURATION_H_
#define _NETWORK_CONFIGURATION_H_

#include "Network.h"
#include "../Simulators/Configuration.h"

namespace Network {

template <typename Time = Types::time_type>
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
    simulatorConfig.SetConfiguration(key, value);
  }

  void SetConfiguration(const std::string& key, long long int value) {
    simulatorConfig.SetConfiguration(key, value);
  }

  void SetConfiguration(const std::string& key, double value) {
    simulatorConfig.SetConfiguration(key, value);
  }

  /**
   * @brief Check if a configuration value is set.
   *
   * Check if a configuration value is set for the simulator.
   *
   * @param key The key of the configuration value.
   * @return True if the configuration value is set, false otherwise.
   */
  bool IsSet(const std::string& key) const { return simulatorConfig.IsSet(key); }

  /**
   * @brief Get a configuration value.
   *
   * Get a configuration value for the simulator.
   *
   * @param key The key of the configuration value.
   * @return The configuration value as a string.
   */
  std::string GetConfiguration(const std::string& key) const {
    return simulatorConfig.GetConfiguration(key);
  }

  long long int GetConfigurationAsInt(const std::string& key) const {
    return simulatorConfig.GetConfigurationAsInt(key);
  }

  double GetConfigurationAsDouble(const std::string& key) const {
    return simulatorConfig.GetConfigurationAsDouble(key);
  }

  bool CanBeAppliedOnInitializedSimulator(const std::string& key) const {
    return simulatorConfig.CanBeAppliedOnInitializedSimulator(key);
  }

  void SetSimulatorConfiguration(const Simulators::Configuration& config) {
    simulatorConfig = config;
  }

  void ApplyConfigurationToSimulator(
      const std::shared_ptr<Simulators::IState>& simulator) const {
    simulatorConfig.ApplyConfigurationToSimulator(simulator);
  }

  void ApplyConfigurationFromSimulator(
      const std::shared_ptr<Simulators::IState>& simulator) {
    simulatorConfig.ApplyConfigurationFromSimulator(simulator);
  }

  void ApplyConfigurationFromNetwork(
      const std::shared_ptr<INetwork<Time>>& network) {
    if (!network) return;
    ApplyConfigurationFromMap(network->GetConfigMap());
  }

  void ApplyConfigurationToNetwork(
      const std::shared_ptr<INetwork<Time>>& network) const {
    for (const auto& [key, value] : simulatorConfig.GetConfigMap())
      network->Configure(key.c_str(), value.c_str());
  }

  const std::unordered_map<std::string, std::string>& GetConfigMap() const {
    return simulatorConfig.GetConfigMap();
  }

 private:
  Simulators::Configuration
      simulatorConfig; /**< The simulator configuration. */
};

}

#endif