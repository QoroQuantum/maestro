/**
 * @file QCSimState.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The qcsim state class.
 *
 * Should not be used directly, create an instance with the factory and use the
 * generic simulator interface.
 */

#pragma once

#ifndef _QCSIMSTATE_H_
#define _QCSIMSTATE_H_

#ifdef INCLUDED_BY_FACTORY

#include <algorithm>

#include "Simulator.h"

#include "Clifford.h"
#include "MPSSimulator.h"
#include "QubitRegister.h"
#include "QcsimPauliPropagator.h"

#include "../TensorNetworks/ForestContractor.h"
#include "../TensorNetworks/TensorNetwork.h"

#include "../Utils/Alias.h"

#include "MPSDummySimulator.h"

namespace Simulators {
// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private {

/**
 * @class QCSimState
 * @brief Class for the qcsim simulator state.
 *
 * Implements the qcsim state.
 * Do not use this class directly, use the factory to create an instance.
 * @sa ISimulator
 * @sa IState
 * @sa QCSimSimulator
 */
class QCSimState : public ISimulator {
 public:
  QCSimState() : rng(std::random_device{}()), uniformZeroOne(0, 1) {}

  /**
   * @brief Initializes the state.
   *
   * This function is called when the simulator is initialized.
   * Call it after the qubits allocation.
   * @sa QCSimState::AllocateQubits
   */
  void Initialize() override {
    if (nrQubits != 0) {
      if (simulationType == SimulationType::kMatrixProductState) {
        mpsSimulator =
            std::make_unique<QC::TensorNetworks::MPSSimulator>(nrQubits);
        if (limitEntanglement && singularValueThreshold > 0.)
          mpsSimulator->setLimitEntanglement(singularValueThreshold);
        if (limitSize && chi > 0) mpsSimulator->setLimitBondDimension(chi);
      } else if (simulationType == SimulationType::kStabilizer)
        cliffordSimulator =
            std::make_unique<QC::Clifford::StabilizerSimulator>(nrQubits);
      else if (simulationType == SimulationType::kTensorNetwork) {
        tensorNetwork =
            std::make_unique<TensorNetworks::TensorNetwork>(nrQubits);
        // for now the only used contractor is the forest one, but we'll use
        // more in the future
        const auto tensorContractor =
            std::make_shared<TensorNetworks::ForestContractor>();
        tensorNetwork->SetContractor(tensorContractor);
      } else if (simulationType == SimulationType::kPauliPropagator) {
        pp = std::make_unique<Simulators::QcsimPauliPropagator>();
        pp->SetNrQubits(static_cast<int>(nrQubits));
      } else
        state = std::make_unique<QC::QubitRegister<>>(nrQubits);

      SetMultithreading(enableMultithreading);
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
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();
    if (simulationType != SimulationType::kStatevector)
      throw std::runtime_error(
          "QCSimState::InitializeState: Invalid "
          "simulation type for initializing the state.");

    Eigen::VectorXcd amplitudesEigen(
        Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(),
                                                       amplitudes.size()));
    state->setRegisterStorageFastNoNormalize(amplitudesEigen);
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
  /*
  void InitializeState(size_t num_qubits, std::vector<std::complex<double>,
  avoid_init_allocator<std::complex<double>>>& amplitudes) override
  {
          Clear();
          nrQubits = num_qubits;
          Initialize();
          Eigen::VectorXcd amplitudesEigen(Eigen::Map<Eigen::VectorXcd,
  Eigen::Unaligned>(amplitudes.data(), amplitudes.size()));
          state->setRegisterStorageFastNoNormalize(amplitudesEigen);
  }
  */

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
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();
    if (simulationType != SimulationType::kStatevector)
      throw std::runtime_error(
          "QCSimState::InitializeState: Invalid "
          "simulation type for initializing the state.");

    Eigen::VectorXcd amplitudesEigen(
        Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(),
                                                       amplitudes.size()));
    state->setRegisterStorageFastNoNormalize(amplitudesEigen);
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
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();

    if (simulationType != SimulationType::kStatevector)
      throw std::runtime_error(
          "QCSimState::InitializeState: Invalid "
          "simulation type for initializing the state.");

    state = std::make_unique<QC::QubitRegister<>>(nrQubits, amplitudes);
    state->SetMultithreading(enableMultithreading);
  }

  /**
   * @brief Just resets the state to 0.
   *
   * Does not destroy the internal state, just resets it to zero (as a 'reset'
   * op on each qubit would do).
   */
  void Reset() override {
    if (mpsSimulator)
      mpsSimulator->Clear();
    else if (cliffordSimulator)
      cliffordSimulator->Reset();
    else if (tensorNetwork)
      tensorNetwork->Clear();
    else if (state)
      state->Reset();
    else if (pp)
      pp->ClearOperations();

    upcomingGateIndex = 0;
  }

  /**
   * @brief Returns if the simulator supports MPS swap optimization.
   *
   * Used to check if the simulator supports MPS swap optimization.
   * @return True if the simulator supports MPS swap optimization, false
   * otherwise.
   */
  bool SupportsMPSSwapOptimization() const override { return true; }

  /**
   * @brief Sets the initial qubits map, if possible.
   *
   * This will do nothing for most simulators, but for the MPS simulator it will
   * set the initial qubits if it supports it - that is, for qcsim and the gpu
   * simulator it can set the mapping of the qubits to the positions in the
   * chain, which can be used to optimize the swapping cost.
   */
  void SetInitialQubitsMap(
      const std::vector<long long int> &initialMap) override {
    if (mpsSimulator) {
      mpsSimulator->SetInitialQubitsMap(initialMap);
      if (!dummySim || dummySim->getNrQubits() != initialMap.size()) {
        dummySim = std::make_unique<Simulators::MPSDummySimulator>(initialMap.size());
        dummySim->SetMaxBondDimension(
            limitSize ? static_cast<long long int>(chi) : 0);
      }
      dummySim->SetInitialQubitsMap(initialMap);
    }
  }

  void SetUseOptimalMeetingPosition(bool enable) override {
    useOptimalMeetingPositionOnly = enable;
    if (mpsSimulator)
        mpsSimulator->SetUseOptimalMeetingPosition(enable);
  }

  void SetLookaheadDepth(int depth) override {
    lookaheadDepth = depth;
    if (mpsSimulator && depth > 0 && !useOptimalMeetingPositionOnly)
      mpsSimulator->SetUseOptimalMeetingPosition(true);
  }

  void SetLookaheadDepthWithHeuristic(int depth) override 
  {
    lookaheadDepthWithHeuristic = depth;
    if (lookaheadDepth < depth) SetLookaheadDepth(depth);
  }

  void SetUpcomingGates(
      const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &gates) override {
    upcomingGates = gates;
    upcomingGateIndex = 0;

    if (!mpsSimulator || lookaheadDepth <= 0) return;

    // Register an observer that advances the gate index
    ClearObservers(); // for now we only have this observer, so this should be fine
    gateCounterObserver = std::make_shared<GateCounterObserver>(upcomingGateIndex);
    RegisterObserver(gateCounterObserver);

    // Set up a meeting position callback that uses MPSDummySimulator
    // for lookahead evaluation with actual bond dimensions
    // the callback is called only for two qubits gates and only if executing them would require a swap
    mpsSimulator->SetMeetingPositionCallback(
        [this](/*const auto &qMap,*/ const auto &bondDims)
            -> QC::TensorNetworks::MPSSimulatorInterface::IndexType {
          const size_t nQ = bondDims.size() + 1;

          if (!dummySim || dummySim->getNrQubits() != nQ) {
            dummySim = std::make_unique<Simulators::MPSDummySimulator>(nQ);
            dummySim->SetMaxBondDimension(limitSize ? static_cast<long long int>(chi) : 0);
          }

          // Seed dummy with current real simulator state
          //std::vector<long long int> map64(qMap.begin(), qMap.end());
          //dummySim->SetInitialQubitsMap(map64);
          dummySim->setTotalSwappingCost(0);


          // check qubits map:
          /*
          auto qbitmMap = dummySim->getQubitsMap();
          for (size_t i = 0; i < nQ; ++i) {
            if (qbitmMap[i] != qMap[i]) {
              std::cerr << "Error: qubits map mismatch at index " << i
                        << ": dummySim has " << qbitmMap[i]
                        << " but real sim has " << qMap[i] << std::endl;
              exit(0);
            }
          }
          */

          // check them, they should be the same, otherwise something is wrong


          // Convert actual bond dims to doubles
          std::vector<double> bondDimsD(bondDims.begin(), bondDims.end());
          dummySim->SetCurrentBondDimensions(bondDimsD);

          // display bond dimensions for debugging
          /*
          std::cout << "Bond dimensions before swapping and applying the gate: ";
          for (size_t i = 0; i < bondDims.size(); ++i) {
            std::cout << bondDims[i] << " ";
          }
          std::cout << std::endl;
          */

          const auto &op = upcomingGates[upcomingGateIndex];
          const auto qbits = op->AffectedQubits();

          if (qbits.size() != 2) {
            std::cerr << "Error: Meeting position callback called for a gate that does not have exactly 2 qubits." << std::endl;

            return -1; // will fallback 
          }
          
          /*
          const auto &qmap = dummySim->getQubitsMap();

          std::cout << "Applying 2-qubit gate on physical qubits " << qmap[qbits[0]] << " and "
                    << qmap[qbits[1]]
                    << std::endl;
          */
          /*
          std::cout << "Finding best meeting position for upcoming gates starting at index "
                    << upcomingGateIndex << " with lookahead depth " << lookaheadDepth << " and heuristic depth "
                    << lookaheadDepthWithHeuristic << std::endl;



          std::cout << "Affected qubits: ";
          for (const auto &q : qbits) std::cout << q << " ";
          std::cout << std::endl;

          std::cout << "Current qubits map: ";
          for (size_t i = 0; i < qMap.size(); ++i) std::cout << qMap[i] << " ";
          std::cout << std::endl;

          std::cout << "Current inverse qubits map: ";
          for (size_t i = 0; i < qMapInv.size(); ++i) std::cout << qMapInv[i] << " ";
          std::cout << std::endl;
          */

          double bestCost = std::numeric_limits<double>::infinity();
          auto res = dummySim->FindBestMeetingPosition(upcomingGates, upcomingGateIndex, lookaheadDepth, lookaheadDepthWithHeuristic, 0, bestCost);

          //std::cout << "Swapping the two qubits on position: " << res << " and " << (res + 1) << std::endl;

          dummySim->SwapQubitsToPosition(qbits[0], qbits[1], res);
          dummySim->ApplyGate(op);

          // display the expected bond dimensions after applying the gate for
          // debugging
          
          /*
          const auto &expectedBondDims = dummySim->getCurrentBondDimensions();
          std::cout << "Expected bond dimensions after swapping and applying "
                       "the gate: ";
          for (size_t i = 0; i < expectedBondDims.size(); ++i) {
            std::cout << expectedBondDims[i] << " ";
          }
          std::cout << std::endl;
          */
          
          //std::cout << "Best meeting position: " << res
          //          << " with estimated cost: " << bestCost << std::endl;

          return res;
        });
  }

  /**
   * @brief Returns the gates counter.
   *
   * Usually does nothing, except for MPS simulators that support swap
   * optimization.
   *
   * @return The number of gates executed in the circuit.
   */
  long long int GetGatesCounter() const override { return upcomingGateIndex; }

  /**
   * @brief Sets the gates counter.
   *
   * Usually does nothing, except for MPS simulators that support swap
   * optimization.
   *
   * @param counter The position in the circuit from where the execution should
   * continue.
   */
  void SetGatesCounter(long long int counter) override { 
      upcomingGateIndex = counter;
  }

  /**
   * @brief Increments the gates counter.
   *
   * Usually does nothing, except for MPS simulators that support swap
   * optimization. Increments the position in the circuit from where the
   * execution should continue. Useful for classically controlled gates, for the
   * case when the controlled gate is not executed.
   */
  void IncrementGatesCounter() override {
      ++upcomingGateIndex;
  }

  /**
   * @brief Configures the state.
   *
   * This function is called to configure the simulator.
   *
   * @param key The key of the configuration option.
   * @param value The value of the configuration.
   */
  void Configure(const char *key, const char *value) override {
    if (std::string("method") == key) {
      if (std::string("statevector") == value)
        simulationType = SimulationType::kStatevector;
      else if (std::string("matrix_product_state") == value)
        simulationType = SimulationType::kMatrixProductState;
      else if (std::string("stabilizer") == value)
        simulationType = SimulationType::kStabilizer;
      else if (std::string("tensor_network") == value)
        simulationType = SimulationType::kTensorNetwork;
      else if (std::string("pauli_propagator") == value)
        simulationType = SimulationType::kPauliPropagator;
    } else if (std::string("matrix_product_state_truncation_threshold") ==
               key) {
      singularValueThreshold = std::stod(value);
      if (singularValueThreshold > 0.) {
        limitEntanglement = true;
        if (mpsSimulator)
          mpsSimulator->setLimitEntanglement(singularValueThreshold);
      } else
        limitEntanglement = false;
    } else if (std::string("matrix_product_state_max_bond_dimension") == key) {
      chi = std::stoi(value);
      if (chi > 0) {
        limitSize = true;
        if (mpsSimulator) mpsSimulator->setLimitBondDimension(chi);
        if (dummySim) dummySim->SetMaxBondDimension(static_cast<long long int>(chi));
      } else {
        limitSize = false;
        if (mpsSimulator) mpsSimulator->setLimitBondDimension(0);
        if (dummySim) dummySim->SetMaxBondDimension(0);
      }
    } else if (std::string("mps_sample_measure_algorithm") == key)
      useMPSMeasureNoCollapse = std::string("mps_probabilities") == value;
  }

  /**
   * @brief Returns configuration value.
   *
   * This function is called get a configuration value.
   * @param key The key of the configuration value.
   * @return The configuration value as a string.
   */
  std::string GetConfiguration(const char *key) const override {
    if (std::string("method") == key) {
      switch (simulationType) {
        case SimulationType::kStatevector:
          return "statevector";
        case SimulationType::kMatrixProductState:
          return "matrix_product_state";
        case SimulationType::kStabilizer:
          return "stabilizer";
        case SimulationType::kTensorNetwork:
          return "tensor_network";
        case SimulationType::kPauliPropagator:
          return "pauli_propagator";
        default:
          return "other";
      }
    } else if (std::string("matrix_product_state_truncation_threshold") ==
               key) {
      if (limitEntanglement && singularValueThreshold > 0.)
        return std::to_string(singularValueThreshold);
    } else if (std::string("matrix_product_state_max_bond_dimension") == key) {
      if (limitSize && chi > 0) return std::to_string(chi);
    } else if (std::string("mps_sample_measure_algorithm") == key) {
      return useMPSMeasureNoCollapse ? "mps_probabilities"
                                     : "mps_apply_measure";
    }

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
    if ((simulationType == SimulationType::kStatevector && state) ||
        (simulationType == SimulationType::kMatrixProductState &&
         mpsSimulator) ||
        (simulationType == SimulationType::kStabilizer && cliffordSimulator) ||
        (simulationType == SimulationType::kTensorNetwork && tensorNetwork))
      return 0;

    const size_t oldNrQubits = nrQubits;
    nrQubits += num_qubits;
    if (simulationType == SimulationType::kPauliPropagator)
      if (pp) pp->SetNrQubits(static_cast<int>(nrQubits));

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
    state = nullptr;
    mpsSimulator = nullptr;
    cliffordSimulator = nullptr;
    tensorNetwork = nullptr;
    pp = nullptr;
    dummySim = nullptr;
    nrQubits = 0;
    upcomingGateIndex = 0;
    upcomingGates.clear();
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
    // TODO: this is inefficient, maybe implement it better in qcsim
    // for now it has the possibility of measuring a qubits interval, but not a
    // list of qubits
    if (qubits.size() > sizeof(size_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the size_t type, the outcome will be undefined"
          << std::endl;

    size_t res = 0;
    size_t mask = 1ULL;

    DontNotify();
    if (simulationType == SimulationType::kStatevector) {
      for (size_t qubit : qubits) {
        if (state->MeasureQubit(static_cast<unsigned int>(qubit))) res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kStabilizer) {
      for (size_t qubit : qubits) {
        if (cliffordSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
          res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kTensorNetwork) {
      for (size_t qubit : qubits) {
        if (tensorNetwork->Measure(static_cast<unsigned int>(qubit)))
          res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qubitsInt(qubits.begin(), qubits.end());
      const auto res = pp->Measure(qubitsInt);
      Types::qubit_t result = 0;
      for (size_t i = 0; i < res.size(); ++i) {
        if (res[i]) result |= mask;
        mask <<= 1;
      }
      return result;
    } else {
      /*
      for (size_t qubit : qubits)
      {
              if (mpsSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
                      res |= mask;
              mask <<= 1;
      }
      */
      const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
      auto measured = mpsSimulator->MeasureQubits(qubitsSet);
      for (Types::qubit_t qubit : qubits) {
        if (measured[qubit]) res |= mask;
        mask <<= 1;
      }
    }
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

    if (simulationType == SimulationType::kStatevector) {
      for (size_t q = 0; q < qubits.size(); ++q)
        if (state->MeasureQubit(static_cast<unsigned int>(qubits[q])))
          res[q] = true;
    } else if (simulationType == SimulationType::kStabilizer) {
      for (size_t q = 0; q < qubits.size(); ++q)
        if (cliffordSimulator->MeasureQubit(
                static_cast<unsigned int>(qubits[q])))
          res[q] = true;
    } else if (simulationType == SimulationType::kTensorNetwork) {
      for (size_t q = 0; q < qubits.size(); ++q)
        if (tensorNetwork->Measure(static_cast<unsigned int>(qubits[q])))
          res[q] = true;
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qubitsInt(qubits.begin(), qubits.end());
      res = pp->Measure(qubitsInt);
    } else {
      const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
      auto measured = mpsSimulator->MeasureQubits(qubitsSet);
      for (size_t q = 0; q < qubits.size(); ++q)
        if (measured[qubits[q]]) res[q] = true;
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
    QC::Gates::PauliXGate xGate;

    DontNotify();
    if (simulationType == SimulationType::kStatevector) {
      for (size_t qubit : qubits)
        if (state->MeasureQubit(static_cast<unsigned int>(qubit)))
          state->ApplyGate(xGate, static_cast<unsigned int>(qubit));
    } else if (simulationType == SimulationType::kStabilizer) {
      for (size_t qubit : qubits)
        if (cliffordSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
          cliffordSimulator->ApplyX(static_cast<unsigned int>(qubit));
    } else if (simulationType == SimulationType::kTensorNetwork) {
      for (size_t qubit : qubits)
        if (tensorNetwork->Measure(static_cast<unsigned int>(qubit)))
          tensorNetwork->AddGate(xGate, static_cast<unsigned int>(qubit));
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qubitsInt(qubits.begin(), qubits.end());
      const auto res = pp->Measure(qubitsInt);
      for (size_t i = 0; i < res.size(); ++i) {
        if (res[i]) pp->ApplyX(qubitsInt[i]);
      }
    } else {
      for (size_t qubit : qubits)
        if (mpsSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
          mpsSimulator->ApplyGate(xGate, static_cast<unsigned int>(qubit));
    }
    Notify();

    NotifyObservers(qubits);
  }

  /**
   * @brief Returns the probability of the specified outcome.
   *
   * Use it to obtain the probability to obtain the specified outcome, if all
   * qubits are measured.
   * @sa QCSimState::Amplitude
   * @sa QCSimState::Probabilities
   *
   * @param outcome The outcome to obtain the probability for.
   * @return The probability of the specified outcome.
   */
  double Probability(Types::qubit_t outcome) override {
    if (simulationType == SimulationType::kMatrixProductState)
      return mpsSimulator->getBasisStateProbability(
          static_cast<unsigned int>(outcome));
    else if (simulationType == SimulationType::kStabilizer)
      return cliffordSimulator->getBasisStateProbability(
          static_cast<unsigned int>(outcome));
    else if (simulationType == SimulationType::kTensorNetwork)
      return tensorNetwork->getBasisStateProbability(outcome);
    else if (simulationType == SimulationType::kPauliPropagator)
      return pp->Probability(outcome);

    return state->getBasisStateProbability(static_cast<unsigned int>(outcome));
  }

  /**
   * @brief Returns the amplitude of the specified state.
   *
   * Use it to obtain the amplitude of the specified state.
   * @sa QCSimState::Probability
   * @sa QCSimState::Probabilities
   *
   * @param outcome The outcome to obtain the amplitude for.
   * @return The amplitude of the specified outcome.
   */
  std::complex<double> Amplitude(Types::qubit_t outcome) override {
    if (simulationType == SimulationType::kMatrixProductState)
      return mpsSimulator->getBasisStateAmplitude(
          static_cast<unsigned int>(outcome));
    else if (simulationType == SimulationType::kStabilizer)
      throw std::runtime_error(
          "QCSimState::Amplitude: Invalid simulation type for obtaining the "
          "amplitude of the specified outcome.");
    else if (simulationType == SimulationType::kTensorNetwork)
      throw std::runtime_error(
          "QCSimState::Amplitude: Not supported for the "
          "tensor network simulator.");
    else if (simulationType == SimulationType::kPauliPropagator)
      throw std::runtime_error(
          "QCSimState::Amplitude: Invalid simulation type for obtaining the "
          "amplitude of the specified outcome.");

    return state->getBasisStateAmplitude(static_cast<unsigned int>(outcome));
  }

  /**
   * @brief Projects the state onto the zero state.
   *
   * Use it to project the state onto the zero state.
   * For most simulator is the same as calling Amplitude(0), but for some
   * simulators it can be optimized to be faster than calling Amplitude(0).
   * This for now is done for qcsim mps and gpu mps.
   *
   * @sa IState::Amplitude
   * @sa IState::Probability
   *
   * @return The inner product result as a complex number.
   */
  std::complex<double> ProjectOnZero() override { 
      if (simulationType == SimulationType::kMatrixProductState)
        return mpsSimulator->ProjectOnZero();
      return Amplitude(0); 
  }

  /**
   * @brief Returns the probabilities of all possible outcomes.
   *
   * Use it to obtain the probabilities of all possible outcomes.
   * @sa QCSimState::Probability
   * @sa QCSimState::Amplitude
   * @sa QCSimState::AllProbabilities
   *
   * @return A vector with the probabilities of all possible outcomes.
   */
  std::vector<double> AllProbabilities() override {
    // TODO: In principle this could be done, but why? It should be costly.
    if (simulationType == SimulationType::kTensorNetwork)
      throw std::runtime_error(
          "QCSimState::AllProbabilities: Invalid "
          "simulation type for obtaining probabilities.");
    else if (simulationType == SimulationType::kStabilizer)
      return cliffordSimulator->AllProbabilities();
    else if (simulationType == SimulationType::kPauliPropagator) {
      const size_t nrBasisStates = 1ULL << GetNumberOfQubits();
      std::vector<double> result(nrBasisStates);
      for (size_t i = 0; i < nrBasisStates; ++i) result[i] = pp->Probability(i);
      return result;
    }

    const Eigen::VectorXcd probs =
        simulationType == SimulationType::kMatrixProductState
            ? mpsSimulator->getRegisterStorage().cwiseAbs2()
            : state->getRegisterStorage().cwiseAbs2();

    std::vector<double> result(probs.size());

    for (int i = 0; i < probs.size(); ++i) result[i] = probs[i].real();

    return result;
  }

  /**
   * @brief Returns the probabilities of the specified outcomes.
   *
   * Use it to obtain the probabilities of the specified outcomes.
   * @sa QCSimState::Probability
   * @sa QCSimState::Amplitude
   *
   * @param qubits A vector with the qubits configuration outcomes.
   * @return A vector with the probabilities for the specified qubit
   * configurations.
   */
  std::vector<double> Probabilities(
      const Types::qubits_vector &qubits) override {
    if (simulationType == SimulationType::kStabilizer)
      throw std::runtime_error(
          "QCSimState::Probabilities: Invalid simulation "
          "type for obtaining probabilities.");
    else if (simulationType == SimulationType::kTensorNetwork) {
      // TODO: Implement this!!!
      throw std::runtime_error(
          "QCSimState::Probabilities: Not implemented yet "
          "for the tensor network simulator.");
    }

    std::vector<double> result(qubits.size());

    if (simulationType == SimulationType::kMatrixProductState) {
      for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
        result[i] = mpsSimulator->getBasisStateProbability(qubits[i]);
    } else if (simulationType == SimulationType::kPauliPropagator) {
      for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
        result[i] = pp->Probability(qubits[i]);
    } else {
      const Eigen::VectorXcd &reg = state->getRegisterStorage();

      for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
        result[i] = std::norm(reg[qubits[i]]);
    }

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

    if (qubits.size() > sizeof(size_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the size_t type, the outcome will be undefined"
          << std::endl;

    // TODO: this is inefficient, maybe implement it better in qcsim
    // for now it has the possibility of measuring a qubits interval, but not a
    // list of qubits
    std::unordered_map<Types::qubit_t, Types::qubit_t> result;

    DontNotify();

    if (simulationType == SimulationType::kMatrixProductState) {
      bool normal = true;
      if (useMPSMeasureNoCollapse) {
        // check to see if it can be used
        const std::set<Eigen::Index> qset(qubits.begin(), qubits.end());
        if (qset.size() == GetNumberOfQubits()) {
          // it can!
          normal = false;
          for (size_t shot = 0; shot < shots; ++shot) {
            const size_t measRaw = MeasureNoCollapse();
            size_t meas = 0;
            size_t mask = 1ULL;

            // translate the measurement
            for (auto q : qubits) {
              const size_t qubitMask = 1ULL << q;
              if (measRaw & qubitMask) meas |= mask;
              mask <<= 1ULL;
            }

            ++result[meas];
          }
        } else if (qset.size() > 1) {
          mpsSimulator->MoveAtBeginningOfChain(qset);
          // now sample
          normal = false;
          for (size_t shot = 0; shot < shots; ++shot) {
            const auto measRaw = mpsSimulator->MeasureNoCollapse(qset);
            size_t meas = 0;
            size_t mask = 1ULL;

            // might not be in the requested order
            // translate the measurement            
            for (auto q : qubits) {
              if (measRaw.at(q)) meas |= mask;
              mask <<= 1ULL;
            }

            ++result[meas];
          }

        } else if (qset.size() == 1) {
          // if only one qubit is measured, we can use the probability
          normal = false;
          const auto prob0 = mpsSimulator->GetProbability(qubits[0]);
          for (size_t shot = 0; shot < shots; ++shot) {
            const size_t meas = uniformZeroOne(rng) < prob0 ? 0ULL : 1ULL;
            size_t m = meas;
            // why would somebody set more than one time?
            for (size_t i = 1; i < qubits.size(); ++i) {
              m <<= 1ULL;
              m |= meas;
            }
            ++result[m];
          }
        }
      }

      if (normal) {
        auto savedState = mpsSimulator->getState();
        for (size_t shot = 0; shot < shots; ++shot) {
          const size_t meas = Measure(qubits);
          ++result[meas];
          mpsSimulator->setState(savedState);
        }
      }
    } else if (simulationType == SimulationType::kStabilizer) {
      cliffordSimulator->SaveState();
      for (size_t shot = 0; shot < shots; ++shot) {
        const size_t meas = Measure(qubits);
        ++result[meas];
        cliffordSimulator->RestoreState();
      }
      cliffordSimulator->ClearSavedState();
    } else if (simulationType == SimulationType::kTensorNetwork) {
      tensorNetwork->SaveState();
      for (size_t shot = 0; shot < shots; ++shot) {
        const size_t meas = Measure(qubits);
        ++result[meas];
        tensorNetwork->RestoreState();
      }
      tensorNetwork->ClearSavedState();
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qubitsInt(qubits.begin(), qubits.end());
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto res = pp->Sample(qubitsInt);

        size_t meas = 0;
        for (size_t i = 0; i < qubits.size(); ++i) {
          if (res[i]) meas |= (1ULL << i);
        }

        ++result[meas];
      }
    } else {
      if (shots > 1) {
        const auto &statev = state->getRegisterStorage();

        const Utils::Alias alias(statev);

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

    DontNotify();

    if (simulationType == SimulationType::kMatrixProductState) {
      bool normal = true;
      if (useMPSMeasureNoCollapse) {
        // check to see if it can be used
        const std::set<Eigen::Index> qset(qubits.begin(), qubits.end());
        if (qset.size() == GetNumberOfQubits()) {
          // it can!
          normal = false;
          for (size_t shot = 0; shot < shots; ++shot) {
            const auto meas = MeasureNoCollapseMany();

            // might not be in the requested order
            // translate the measurement
            std::vector<bool> measVec(qubits.size());
            for (size_t i = 0; i < qubits.size(); ++i)
              measVec[i] = meas[qubits[i]];

            ++result[measVec];
          }
        } else if (qset.size() > 1) {
          mpsSimulator->MoveAtBeginningOfChain(qset);
          // now sample
          normal = false;
          for (size_t shot = 0; shot < shots; ++shot) {
            const auto meas = mpsSimulator->MeasureNoCollapse(qset);
            
            // might not be in the requested order
            // translate the measurement            
            std::vector<bool> measVec(qubits.size());
            for (size_t i = 0; i < qubits.size(); ++i)
              measVec[i] = meas.at(qubits[i]);

            ++result[measVec];
          }
        } else if (qset.size() == 1) {
          // if only one qubit is measured, we can use the probability
          normal = false;
          const auto prob0 = mpsSimulator->GetProbability(qubits[0]);
          for (size_t shot = 0; shot < shots; ++shot) {
            const size_t meas = uniformZeroOne(rng) < prob0 ? 0ULL : 1ULL;
            const std::vector<bool> m(qubits.size(), meas);
            ++result[m];
          }
        }
      }

      if (normal) {
        auto savedState = mpsSimulator->getState();
        for (size_t shot = 0; shot < shots; ++shot) {
          const auto meas = MeasureMany(qubits);

          ++result[meas];
          mpsSimulator->setState(savedState);
        }
      }
    } else if (simulationType == SimulationType::kStabilizer) {
      cliffordSimulator->SaveState();
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto meas = MeasureMany(qubits);
        ++result[meas];
        cliffordSimulator->RestoreState();
      }
      cliffordSimulator->ClearSavedState();
    } else if (simulationType == SimulationType::kTensorNetwork) {
      tensorNetwork->SaveState();
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto meas = MeasureMany(qubits);
        ++result[meas];
        tensorNetwork->RestoreState();
      }
      tensorNetwork->ClearSavedState();
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qubitsInt(qubits.begin(), qubits.end());
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto meas = pp->Sample(qubitsInt);
        ++result[meas];
      }
    } else {
      if (shots > 1) {
        const auto &statev = state->getRegisterStorage();

        const Utils::Alias alias(statev);

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
  double ExpectationValue(const std::string &pauliStringOrig) override {
    if (pauliStringOrig.empty()) return 1.0;

    std::string pauliString = pauliStringOrig;
    if (pauliString.size() > GetNumberOfQubits()) {
      for (size_t i = GetNumberOfQubits(); i < pauliString.size(); ++i) {
        const auto pauliOp = toupper(pauliString[i]);
        if (pauliOp != 'I' && pauliOp != 'Z') return 0.0;
      }

      pauliString.resize(GetNumberOfQubits());
    }

    if (simulationType == SimulationType::kStabilizer)
      return cliffordSimulator->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kTensorNetwork)
      return tensorNetwork->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kPauliPropagator)
      return pp->ExpectationValue(pauliString);

    // statevector or mps
    static const QC::Gates::PauliXGate<> xgate;
    static const QC::Gates::PauliYGate<> ygate;
    static const QC::Gates::PauliZGate<> zgate;

    std::vector<QC::Gates::AppliedGate<Eigen::MatrixXcd>> pauliStringVec;
    pauliStringVec.reserve(pauliString.size());

    for (size_t q = 0; q < pauliString.size(); ++q) {
      switch (toupper(pauliString[q])) {
        case 'X': {
          QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(
              xgate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
          pauliStringVec.emplace_back(std::move(ag));
        } break;
        case 'Y': {
          QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(
              ygate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
          pauliStringVec.emplace_back(std::move(ag));
        } break;
        case 'Z': {
          QC::Gates::AppliedGate<Eigen::MatrixXcd> ag(
              zgate.getRawOperatorMatrix(), static_cast<Types::qubit_t>(q));
          pauliStringVec.emplace_back(std::move(ag));
        } break;
        case 'I':
          [[fallthrough]];
        default:
          break;
      }
    }

    if (pauliStringVec.empty()) return 1.0;

    if (simulationType == SimulationType::kMatrixProductState)
      return mpsSimulator->ExpectationValue(pauliStringVec).real();

    return state->ExpectationValue(pauliStringVec).real();
  }

  /**
   * @brief Returns the type of simulator.
   *
   * Returns the type of simulator.
   * @return The type of simulator.
   * @sa SimulatorType
   */
  SimulatorType GetType() const override { return SimulatorType::kQCSim; }

  /**
   * @brief Returns the type of simulation.
   *
   * Returns the type of simulation.
   *
   * @return The type of simulation.
   * @sa SimulationType
   */
  SimulationType GetSimulationType() const override { return simulationType; }

  /**
   * @brief Flushes the applied operations
   *
   * This function is called to flush the applied operations.
   * It is used to flush the operations that were applied to the state.
   * qcsim applies them right away, so this has no effect on it, but qiskit aer
   * does not.
   */
  void Flush() override {}

  /**
   * @brief Saves the state to internal storage.
   *
   * Saves the state to internal storage, if needed.
   * Calling this should consider as the simulator is gone to uninitialized.
   * Either do not use it except for getting amplitudes, or reinitialize the
   * simulator after calling it. This is needed only for the composite
   * simulator, for an optimization for qiskit aer. For qcsim it does nothing.
   */
  void SaveStateToInternalDestructive() override {}

  /**
   * @brief Restores the state from the internally saved state
   *
   * Restores the state from the internally saved state, if needed.
   * This does something only for qiskit aer.
   */
  void RestoreInternalDestructiveSavedState() override {}

  /**
   * @brief Saves the state to internal storage.
   *
   * Saves the state to internal storage, if needed.
   * Calling this will not destroy the internal state, unlike the 'Destructive'
   * variant. To be used in order to recover the state after doing measurements,
   * for multiple shots executions. In the first phase, only qcsim will
   * implement this.
   */
  void SaveState() override {
    if (simulationType == SimulationType::kMatrixProductState)
      mpsSimulator->SaveState();
    else if (simulationType == SimulationType::kStabilizer)
      cliffordSimulator->SaveState();
    else if (simulationType == SimulationType::kTensorNetwork)
      tensorNetwork->SaveState();
    else if (simulationType == SimulationType::kPauliPropagator)
      pp->SaveState();
    else
      state->SaveState();
  }

  /**
   * @brief Restores the state from the internally saved state
   *
   * Restores the state from the internally saved state, if needed.
   * To be used in order to recover the state after doing measurements, for
   * multiple shots executions. In the first phase, only qcsim will implement
   * this.
   */
  void RestoreState() override {
    if (simulationType == SimulationType::kMatrixProductState)
      mpsSimulator->RestoreState();
    else if (simulationType == SimulationType::kStabilizer)
      cliffordSimulator->RestoreState();
    else if (simulationType == SimulationType::kTensorNetwork)
      tensorNetwork->RestoreState();
    else if (simulationType == SimulationType::kPauliPropagator)
      pp->RestoreState();
    else
      state->RestoreState();
  }

  /**
   * @brief Gets the amplitude.
   *
   * Gets the amplitude, from the internal storage if needed.
   * This is needed only for the composite simulator, for an optimization for
   * qiskit aer. For qcsim it does the same thing as Amplitude.
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
    enableMultithreading = multithreading;
    if (state) state->SetMultithreading(multithreading);
    if (cliffordSimulator) cliffordSimulator->SetMultithreading(multithreading);
    if (tensorNetwork) tensorNetwork->SetMultithreading(multithreading);
    if (pp) {
      if (multithreading)
        pp->EnableParallel();
      else
        pp->DisableParallel();
    }
  }

  /**
   * @brief Get the multithreading flag.
   *
   * Returns the multithreading flag.
   *
   * @return The multithreading flag.
   */
  bool GetMultithreading() const override { return enableMultithreading; }

  /**
   * @brief Returns if the simulator is a qcsim simulator.
   *
   * Returns if the simulator is a qcsim simulator.
   * This is just a helper function to ease things up: qcsim has different
   * functionality exposed sometimes so it's good to know if we deal with qcsim
   * or with qiskit aer.
   *
   * @return True if the simulator is a qcsim simulator, false otherwise.
   */
  bool IsQcsim() const override { return true; }

  /**
   * @brief Measures all the qubits without collapsing the state.
   *
   * Measures all the qubits without collapsing the state, allowing to perform
   * multiple shots. This is to be used only internally, only for the
   * statevector simulators (or those based on them, as the composite ones). For
   * the qiskit aer case, SaveStateToInternalDestructive is needed to be called
   * before this. If one wants to use the simulator after such measurement(s),
   * RestoreInternalDestructiveSavedState should be called at the end.
   *
   * Don't use this for more qubits than the size of Types::qubit_t, as the
   * result is packed in a limited number of bits (e.g. 64 bits for uint64_t)
   *
   * @return The result of the measurements, the first qubit result is the least
   * significant bit.
   */
  Types::qubit_t MeasureNoCollapse() override {
    if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the Types::qubit_t type, the outcome will be "
             "undefined"
          << std::endl;

    if (simulationType == SimulationType::kStatevector)
      return state->MeasureNoCollapse();
    else if (simulationType == SimulationType::kMatrixProductState) {
      const auto measured = mpsSimulator->MeasureNoCollapse();
      Types::qubit_t result = 0;
      Types::qubit_t mask = 1;
      for (Types::qubit_t q = 0; q < measured.size(); ++q) {
        if (measured.at(q)) result |= mask;
        mask <<= 1;
      }
      return result;
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qubitsInt(GetNumberOfQubits());
      std::iota(qubitsInt.begin(), qubitsInt.end(), 0);
      const auto res = pp->Sample(qubitsInt);
      Types::qubit_t result = 0;
      for (size_t i = 0; i < res.size(); ++i) {
        if (res[i]) result |= (1ULL << i);
      }
      return result;
    }

    throw std::runtime_error(
        "QCSimState::MeasureNoCollapse: Invalid simulation type for measuring "
        "all the qubits without collapsing the state.");

    return 0;
  }

  /**
   * @brief Measures all the qubits without collapsing the state.
   *
   * Measures all the qubits without collapsing the state, allowing to perform
   * multiple shots. This is to be used only internally, only for the
   * statevector simulators (or those based on them, as the composite ones). For
   * the qiskit aer case, SaveStateToInternalDestructive is needed to be called
   * before this. If one wants to use the simulator after such measurement(s),
   * RestoreInternalDestructiveSavedState should be called at the end.
   *
   * Use this for more qubits than the size of Types::qubit_t
   *
   * @return The result of the measurements
   */
  std::vector<bool> MeasureNoCollapseMany() override {
    if (simulationType == SimulationType::kStatevector) {
      auto state = MeasureNoCollapse();
      std::vector<bool> res(nrQubits);
      for (size_t i = 0; i < nrQubits; ++i) res[i] = ((state >> i) & 1) == 1;
      return res;
    } else if (simulationType == SimulationType::kMatrixProductState) {
      const auto measured = mpsSimulator->MeasureNoCollapse();
      std::vector<bool> res(nrQubits);
      for (size_t i = 0; i < nrQubits; ++i) res[i] = measured.at(i);
      return res;
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qubitsInt(GetNumberOfQubits());
      std::iota(qubitsInt.begin(), qubitsInt.end(), 0);
      return pp->Sample(qubitsInt);
    }

    throw std::runtime_error(
        "QCSimState::MeasureNoCollapseMany: Invalid simulation type for "
        "measuring all the qubits without collapsing the state.");

    return {};
  }

 protected:
  SimulationType simulationType =
      SimulationType::kStatevector; /**< The simulation type. */

  std::unique_ptr<QC::QubitRegister<>> state; /**< The qcsim state. */
  std::unique_ptr<QC::TensorNetworks::MPSSimulator>
      mpsSimulator; /**< The qcsim mps simulator. */
  std::unique_ptr<QC::Clifford::StabilizerSimulator>
      cliffordSimulator; /**< The qcsim clifford simulator. */
  std::unique_ptr<TensorNetworks::TensorNetwork>
      tensorNetwork;                        /**< The qcsim tensor network. */
  std::unique_ptr<QcsimPauliPropagator> pp; /**< The qcsim pauli propagator. */

  size_t nrQubits = 0; /**< The number of allocated qubits. */
  bool limitSize = false;
  bool limitEntanglement = false;
  Eigen::Index chi = 10;               // if limitSize is true
  double singularValueThreshold = 0.;  // if limitEntanglement is true
  bool enableMultithreading = true;    /**< The multithreading flag. */
  bool useMPSMeasureNoCollapse =
      true; /**< The flag to use the mps measure no collapse algorithm. */

  int lookaheadDepth = 0;
  int lookaheadDepthWithHeuristic = 0;
  bool useOptimalMeetingPositionOnly = false;
  std::vector<std::shared_ptr<Circuits::IOperation<>>> upcomingGates;
  long long int upcomingGateIndex = 0;
  std::unique_ptr<Simulators::MPSDummySimulator> dummySim;

  // Observer that counts applied gates to track position in upcomingGates
  class GateCounterObserver : public ISimulatorObserver {
   public:
    GateCounterObserver(long long int &indexRef) : index(indexRef) {}
    void Update(const Types::qubits_vector &) override { 
        ++index;
    }
   private:
    long long int &index;
  };
  std::shared_ptr<GateCounterObserver> gateCounterObserver;

  std::mt19937_64 rng;
  std::uniform_real_distribution<double> uniformZeroOne;
};

}  // namespace Private
}  // namespace Simulators

#endif

#endif  // !_QCSIMSTATE_H_
