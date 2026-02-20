/**
 * @file QuestState.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The quest state class.
 *
 * Should not be used directly, create an instance with the factory and use the
 * generic simulator interface.
 */

#pragma once

#ifndef _QUESTSTATE_H_
#define _QUESTSTATE_H_

#ifdef INCLUDED_BY_FACTORY

#include "QuestLibSim.h"

#include <random>
#include <unordered_map>
#include <vector>

namespace Simulators {
namespace Private {
/**
 * @class QuestState
 * @brief Class for the quest simulator state.
 *
 * Implements the quest state.
 * Do not use this class directly, use the factory to create an instance.
 * @sa ISimulator
 * @sa IState
 * @sa QuestLibSim
 */
class QuestState : public ISimulator {
 public:
  QuestState() : rng(std::random_device{}()), uniformZeroOne(0, 1) {}

  /**
   * @brief Initializes the state.
   *
   * This function is called when the simulator is initialized.
   * Call it after the qubits allocation.
   * @sa QuestState::AllocateQubits
   */
  void Initialize() override {
    if (!questLib) questLib = SimulatorsFactory::GetQuestLibrary();
    if (nrQubits && questLib && questLib->IsValid()) {
      simHandle = questLib->CreateSimulator(static_cast<int>(nrQubits));
      sim = questLib->GetSimulator(simHandle);
      if (!sim)
        throw std::runtime_error(
            "QuestState::Initialize: Failed to create "
            "and initialize the statevector state.");
    }
  }

  /**
   * @brief Initializes the state.
   *
   * This function is called when the simulator is initialized.
   * Call it only on a non-initialized state.
   * This works only for 'statevector' method.
   *
   * @param num_qubits The number of qubits to initialize the state with.
   * @param amplitudes A vector with the amplitudes to initialize the state
   * with.
   */
  void InitializeState(size_t num_qubits,
                       std::vector<std::complex<double>> &amplitudes) override {
    throw std::runtime_error(
        "QuestState::InitializeState: Not supported for Quest simulator.");
  }

  /**
   * @brief Initializes the state.
   *
   * This function is called when the simulator is initialized.
   * Call it only on a non-initialized state.
   * This works only for 'statevector' method.
   *
   * @param num_qubits The number of qubits to initialize the state with.
   * @param amplitudes A vector with the amplitudes to initialize the state
   * with.
   */
#ifndef NO_QISKIT_AER
  void InitializeState(size_t num_qubits,
                       AER::Vector<std::complex<double>> &amplitudes) override {
    throw std::runtime_error(
        "QuestState::InitializeState: Not supported for Quest simulator.");
  }
#endif

  /**
   * @brief Initializes the state.
   *
   * This function is called when the simulator is initialized.
   * Call it only on a non-initialized state.
   * This works only for 'statevector' method.
   *
   * @param num_qubits The number of qubits to initialize the state with.
   * @param amplitudes A vector with the amplitudes to initialize the state
   * with.
   */
  void InitializeState(size_t num_qubits,
                       Eigen::VectorXcd &amplitudes) override {
    throw std::runtime_error(
        "QuestState::InitializeState: Not supported for Quest simulator.");
  }

  /**
   * @brief Just resets the state to 0.
   *
   * Does not destroy the internal state, just resets it to zero (as a 'reset'
   * op on each qubit would do).
   */
  void Reset() override {
    if (sim && questLib) {
      questLib->DestroySimulator(simHandle);
      simHandle = questLib->CreateSimulator(static_cast<int>(nrQubits));
      sim = questLib->GetSimulator(simHandle);
    }
  }

  /**
   * @brief Configures the state.
   *
   * This function is called to configure the simulator.
   * Quest only supports statevector, so this is a no-op.
   *
   * @param key The key of the configuration option.
   * @param value The value of the configuration.
   */
  void Configure(const char *key, const char *value) override {
    // Quest only supports statevector, nothing to configure
  }

  /**
   * @brief Returns configuration value.
   *
   * This function is called get a configuration value.
   * @param key The key of the configuration value.
   * @return The configuration value as a string.
   */
  std::string GetConfiguration(const char *key) const override {
    if (std::string("method") == key) return "statevector";
    return "";
  }

  /**
   * @brief Allocates qubits.
   *
   * This function is called to allocate qubits.
   * @param num_qubits The number of qubits to allocate.
   * @return The index of the first qubit allocated.
   */
  size_t AllocateQubits(size_t num_qubits) override {
    if (sim) return 0;

    const size_t oldNrQubits = nrQubits;
    nrQubits += num_qubits;

    return oldNrQubits;
  }

  /**
   * @brief Returns the number of qubits.
   *
   * This function is called to obtain the number of the allocated qubits.
   * @return The number of qubits.
   */
  size_t GetNumberOfQubits() const override { return nrQubits; }

  /**
   * @brief Clears the state.
   *
   * Sets the number of allocated qubits to 0 and clears the state.
   * After this qubits allocation is required then calling
   * IState::AllocateQubits in order to use the simulator.
   */
  void Clear() override {
    if (sim && questLib) {
      questLib->DestroySimulator(simHandle);
      sim = nullptr;
      simHandle = 0;
    }
    if (savedSim && questLib) {
      questLib->DestroySimulator(savedSimHandle);
      savedSim = nullptr;
      savedSimHandle = 0;
    }
    nrQubits = 0;
  }

  /**
   * @brief Performs a measurement on the specified qubits.
   *
   * Don't use it if the number of qubits is larger than the number of bits in
   * the size_t type (usually 64), as the outcome will be undefined
   *
   * @param qubits A vector with the qubits to be measured.
   * @return The outcome of the measurements, the first qubit result is the
   * least significant bit.
   */
  size_t Measure(const Types::qubits_vector &qubits) override {
    if (qubits.size() > sizeof(size_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the size_t type, the outcome will be undefined"
          << std::endl;

    DontNotify();

    std::vector<int> qb(qubits.begin(), qubits.end());
    const size_t res = static_cast<size_t>(questLib->MeasureQubits(sim, qb.data(), static_cast<int>(qubits.size())));

    Notify();
    NotifyObservers(qubits);

    return res;
  }

  /**
   * @brief Performs a measurement on the specified qubits.
   *
   * @param qubits A vector with the qubits to be measured.
   * @return The outcome of the measurements
   */
  std::vector<bool> MeasureMany(const Types::qubits_vector &qubits) override {
    std::vector<bool> res(qubits.size(), false);

    DontNotify();

    std::vector<int> qb(qubits.begin(), qubits.end());
    const size_t resm = static_cast<size_t>(questLib->MeasureQubits(sim, qb.data(), static_cast<int>(qubits.size())));

    size_t mask = 1ULL;
    for (size_t i = 0; i < qubits.size(); ++i) {
      res[i] = (resm & mask) != 0;
      mask <<= 1;
    }

    Notify();
    NotifyObservers(qubits);

    return res;
  }

  /**
   * @brief Performs a reset of the specified qubits.
   *
   * Measures the qubits and for those that are 1, applies X on them
   * @param qubits A vector with the qubits to be reset.
   */
  void ApplyReset(const Types::qubits_vector &qubits) override {
    DontNotify();
    std::vector<int> qb(qubits.begin(), qubits.end());
    const size_t res = static_cast<size_t>(questLib->MeasureQubits(sim, qb.data(), static_cast<int>(qubits.size())));
    size_t mask = 1ULL;
    for (size_t i = 0; i < qubits.size(); ++i) {
      if (res & mask) questLib->ApplyX(sim, static_cast<int>(qubits[i]));
      mask <<= 1;
    }

    Notify();
    NotifyObservers(qubits);
  }

  /**
   * @brief Returns the probability of the specified outcome.
   *
   * Use it to obtain the probability to obtain the specified outcome, if all
   * qubits are measured.
   * @sa QuestState::Amplitude
   * @sa QuestState::Probabilities
   *
   * @param outcome The outcome to obtain the probability for.
   * @return The probability of the specified outcome.
   */
  double Probability(Types::qubit_t outcome) override {
    return questLib->GetOutcomeProbability(sim, static_cast<long long int>(outcome));
  }

  /**
   * @brief Returns the amplitude of the specified state.
   *
   * Use it to obtain the amplitude of the specified state.
   * @sa QuestState::Probability
   * @sa QuestState::Probabilities
   *
   * @param outcome The outcome to obtain the amplitude for.
   * @return The amplitude of the specified outcome.
   */
  std::complex<double> Amplitude(Types::qubit_t outcome) override {
    std::complex<double> amplitude;
    if (questLib->GetAmplitude(sim, static_cast<long long int>(outcome), amplitude)) {
      return amplitude;
    }
    return std::complex<double>(0.0, 0.0);
  }

  /**
   * @brief Returns the probabilities of all possible outcomes.
   *
   * Use it to obtain the probabilities of all possible outcomes.
   * @sa QuestState::Probability
   * @sa QuestState::Amplitude
   * @sa QuestState::AllProbabilities
   *
   * @return A vector with the probabilities of all possible outcomes.
   */
  std::vector<double> AllProbabilities() override {
    if (nrQubits == 0) return {};
    const size_t numStates = 1ULL << nrQubits;
    std::vector<std::complex<double>> amplitudes(numStates);
    questLib->GetAmplitudes(sim, amplitudes);

    std::vector<double> result(numStates);
    for (size_t i = 0; i < numStates; ++i)
      result[i] = std::norm(amplitudes[i]);
    return result;
  }

  /**
   * @brief Returns the probabilities of the specified outcomes.
   *
   * Use it to obtain the probabilities of the specified outcomes.
   * @sa QuestState::Probability
   * @sa QuestState::Amplitude
   *
   * @param qubits A vector with the qubits configuration outcomes.
   * @return A vector with the probabilities for the specified qubit
   * configurations.
   */
  std::vector<double> Probabilities(
      const Types::qubits_vector &qubits) override {
    std::vector<double> result(qubits.size());
    for (size_t i = 0; i < qubits.size(); ++i)
      result[i] = questLib->GetOutcomeProbability(sim, static_cast<long long int>(qubits[i]));
    return result;
  }

  /**
   * @brief Returns the counts of the outcomes of measurement of the specified
   * qubits, for repeated measurements.
   *
   * Use it to obtain the counts of the outcomes of the specified qubits
   * measurements. The state is not collapsed, so the measurement can be
   * repeated 'shots' times.
   *
   * Don't use it if the number of qubits is larger than the number of bits in
   * the Types::qubit_t type (usually 64), as the outcome will be undefined.
   *
   * @param qubits A vector with the qubits to be measured.
   * @param shots The number of shots to perform.
   * @return A map with the counts for the otcomes of measurements of the
   * specified qubits.
   */
  std::unordered_map<Types::qubit_t, Types::qubit_t> SampleCounts(
      const Types::qubits_vector &qubits, size_t shots = 1000) override {
    if (qubits.empty() || shots == 0) return {};

    if (qubits.size() > sizeof(Types::qubit_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the Types::qubit_t type, the outcome will be "
             "undefined"
          << std::endl;

    std::unordered_map<Types::qubit_t, Types::qubit_t> result;

    if (shots > 1) {
      const size_t numStates = 1ULL << nrQubits;
      std::vector<std::complex<double>> amplitudes(numStates);
      questLib->GetAmplitudes(sim, amplitudes);

      const Utils::Alias alias(amplitudes);

      for (size_t shot = 0; shot < shots; ++shot) {
        const double prob = 1. - uniformZeroOne(rng);
        const size_t measRaw = alias.Sample(prob);

        size_t meas = 0;
        size_t mask = 1ULL;
        for (auto q : qubits) {
          const size_t qubitMask = 1ULL << q;
          if ((measRaw & qubitMask) != 0) meas |= mask;
          mask <<= 1ULL;
        }

        ++result[meas];
      }
    } else {
      for (size_t shot = 0; shot < shots; ++shot) {
        const size_t measRaw = MeasureNoCollapse();
        size_t meas = 0;
        size_t mask = 1ULL;

        for (auto q : qubits) {
          const size_t qubitMask = 1ULL << q;
          if ((measRaw & qubitMask) != 0) meas |= mask;
          mask <<= 1ULL;
        }

        ++result[meas];
      }
    }

    Notify();
    NotifyObservers(qubits);

    return result;
  }

  /**
   * @brief Returns the counts of the outcomes of measurement of the specified
   * qubits, for repeated measurements.
   *
   * Use it to obtain the counts of the outcomes of the specified qubits
   * measurements. The state is not collapsed, so the measurement can be
   * repeated 'shots' times.
   *
   * @param qubits A vector with the qubits to be measured.
   * @param shots The number of shots to perform.
   * @return A map with the counts for the otcomes of measurements of the
   * specified qubits.
   */
  std::unordered_map<std::vector<bool>, Types::qubit_t> SampleCountsMany(
      const Types::qubits_vector &qubits, size_t shots = 1000) override {
    if (qubits.empty() || shots == 0) return {};

    std::unordered_map<std::vector<bool>, Types::qubit_t> result;

    if (shots > 1) {
      const size_t numStates = 1ULL << nrQubits;
      std::vector<std::complex<double>> amplitudes(numStates);
      questLib->GetAmplitudes(sim, amplitudes);

      const Utils::Alias alias(amplitudes);

      for (size_t shot = 0; shot < shots; ++shot) {
        const double prob = 1. - uniformZeroOne(rng);
        const size_t measRaw = alias.Sample(prob);

        std::vector<bool> meas(qubits.size(), false);

        for (size_t i = 0; i < qubits.size(); ++i)
          if (((measRaw >> qubits[i]) & 1) == 1) meas[i] = true;

        ++result[meas];
      }
    } else {
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto measRaw = MeasureNoCollapseMany();
        std::vector<bool> meas(qubits.size(), false);

        for (size_t i = 0; i < qubits.size(); ++i)
          if (measRaw[qubits[i]]) meas[i] = true;

        ++result[meas];
      }
    }

    Notify();
    NotifyObservers(qubits);

    return result;
  }

  /**
   * @brief Returns the expected value of a Pauli string.
   *
   * Use it to obtain the expected value of a Pauli string.
   * The Pauli string is a string of characters representing the Pauli
   * operators, e.g. "XIZY". The length of the string should be less or equal to
   * the number of qubits (if it's less, it's completed with I).
   *
   * @param pauliString The Pauli string to obtain the expected value for.
   * @return The expected value of the specified Pauli string.
   */
  double ExpectationValue(const std::string &pauliString) override {
    return questLib->GetExpectationValue(sim, pauliString.c_str());
  }

  /**
   * @brief Returns the type of simulator.
   *
   * Returns the type of simulator.
   * @return The type of simulator.
   * @sa SimulatorType
   */
  SimulatorType GetType() const override { return SimulatorType::kQuestSim; }

  /**
   * @brief Returns the type of simulation.
   *
   * Returns the type of simulation.
   *
   * @return The type of simulation.
   * @sa SimulationType
   */
  SimulationType GetSimulationType() const override {
    return SimulationType::kStatevector;
  }

  /**
   * @brief Flushes the applied operations
   *
   * This function is called to flush the applied operations.
   * Quest applies them right away, so this has no effect.
   */
  void Flush() override {}

  /**
   * @brief Saves the state to internal storage.
   *
   * Saves the state to internal storage by cloning the simulator.
   * Calling this should consider as the simulator is gone to uninitialized.
   * Either do not use it except for getting amplitudes, or reinitialize the
   * simulator after calling it.
   */
  void SaveStateToInternalDestructive() override {
    if (savedSim && questLib) {
      questLib->DestroySimulator(savedSimHandle);
      savedSim = nullptr;
      savedSimHandle = 0;
    }
    savedSimHandle = simHandle;
    simHandle = 0;
    savedSim = sim;
    sim = nullptr;
  }

  /**
   * @brief Restores the state from the internally saved state
   *
   * Restores the state from the internally saved state and frees the saved
   * state.
   */
  void RestoreInternalDestructiveSavedState() override {
    if (sim && questLib) questLib->DestroySimulator(simHandle);
    simHandle = savedSimHandle;
    sim = savedSim;
    savedSimHandle = 0;
    savedSim = nullptr;
  }

  /**
   * @brief Saves the state to internal storage.
   *
   * Saves the state to internal storage by cloning the simulator.
   * Calling this will not destroy the internal state, unlike the 'Destructive'
   * variant. To be used in order to recover the state after doing measurements,
   * for multiple shots executions.
   */
  void SaveState() override {
    if (savedSim && questLib) {
      questLib->DestroySimulator(savedSimHandle);
      savedSim = nullptr;
      savedSimHandle = 0;
    }
    savedSimHandle = questLib->CloneSimulator(sim);
    savedSim = questLib->GetSimulator(savedSimHandle);
  }

  /**
   * @brief Restores the state from the internally saved state
   *
   * Restores the state from the internally saved state, if needed.
   * To be used in order to recover the state after doing measurements, for
   * multiple shots executions.
   */
  void RestoreState() override {
    if (sim && questLib) questLib->DestroySimulator(simHandle);
    simHandle = questLib->CloneSimulator(savedSim);
    sim = questLib->GetSimulator(simHandle);
  }

  /**
   * @brief Gets the amplitude.
   *
   * Gets the amplitude, from the internal storage if needed.
   */
  std::complex<double> AmplitudeRaw(Types::qubit_t outcome) override {
    return Amplitude(outcome);
  }

  /**
   * @brief Enable/disable multithreading.
   *
   * Enable/disable multithreading. Default is enabled.
   *
   * @param multithreading A flag to indicate if multithreading should be
   * enabled.
   */
  void SetMultithreading(bool multithreading = true) override {
    // Quest manages its own threading
  }

  /**
   * @brief Get the multithreading flag.
   *
   * Returns the multithreading flag.
   *
   * @return The multithreading flag.
   */
  bool GetMultithreading() const override { return true; }

  /**
   * @brief Returns if the simulator is a qcsim simulator.
   *
   * Returns if the simulator is a qcsim simulator.
   *
   * @return True if the simulator is a qcsim simulator, false otherwise.
   */
  bool IsQcsim() const override { return false; }

  /**
   * @brief Measures all the qubits without collapsing the state.
   *
   * Measures all the qubits without collapsing the state, allowing to perform
   * multiple shots. Samples from the probability distribution computed from
   * the statevector amplitudes.
   *
   * Don't use this for more qubits than the size of Types::qubit_t, as the
   * result is packed in a limited number of bits (e.g. 64 bits for uint64_t)
   *
   * @return The result of the measurements, the first qubit result is the least
   * significant bit.
   */
  Types::qubit_t MeasureNoCollapse() override {
    if (nrQubits > sizeof(Types::qubit_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the Types::qubit_t type, the outcome will be "
             "undefined"
          << std::endl;

    const size_t numStates = 1ULL << nrQubits;
    std::vector<std::complex<double>> amplitudes(numStates);
    questLib->GetAmplitudes(sim, amplitudes);

    std::vector<double> probs(numStates);
    for (size_t i = 0; i < numStates; ++i)
      probs[i] = std::norm(amplitudes[i]);

    std::discrete_distribution<Types::qubit_t> dist(probs.begin(), probs.end());

    return dist(rng);
  }

  /**
   * @brief Measures all the qubits without collapsing the state.
   *
   * Measures all the qubits without collapsing the state, allowing to perform
   * multiple shots. Samples from the probability distribution computed from
   * the statevector amplitudes.
   *
   * Use this for more qubits than the size of Types::qubit_t
   *
   * @return The result of the measurements
   */
  std::vector<bool> MeasureNoCollapseMany() override {
    const auto meas = MeasureNoCollapse();
    std::vector<bool> result(nrQubits, false);
    for (size_t i = 0; i < nrQubits; ++i)
      result[i] = ((meas >> i) & 1) == 1;
    return result;
  }

 protected:
  std::shared_ptr<QuestLibSim> questLib; /**< The quest library. */
  unsigned long int simHandle = 0;       /**< The simulator handle. */
  void *sim = nullptr;                   /**< The simulator pointer. */
  size_t nrQubits = 0;  /**< The number of allocated qubits. */
  unsigned long int savedSimHandle = 0;  /**< The saved simulator handle. */
  void *savedSim = nullptr;              /**< The saved simulator pointer. */
  std::mt19937_64 rng;
  std::uniform_real_distribution<double> uniformZeroOne;
};
}  // namespace Private
}  // namespace Simulators

#endif

#endif  // _QUESTSTATE_H_
