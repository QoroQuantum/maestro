/**
 * @file ImmediateQCSimState.h
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

#ifndef _IMMEDIATE_QCSIMSTATE_H_
#define _IMMEDIATE_QCSIMSTATE_H_

#ifdef INCLUDED_BY_FACTORY

#include <algorithm>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>
#include <utility>

#include "Simulator.h"
#include "RoutingMap.h"

#include "Clifford.h"
#include "DensityMatrix.h"
#include "MPOSimulator.h"
#include "MPSSimulator.h"
#include "QubitRegister.h"
#include "QcsimPauliPropagator.h"
#include "QCSimExtendedStabilizer.h"
#include "PathIntegralSimulator.h"

#include "../TensorNetworks/ForestContractor.h"
#include "../TensorNetworks/TensorNetwork.h"

#include "../Utils/Alias.h"

#include "MPSDummySimulator.h"
#include "MPOValidation.h"
#include "Configuration.h"

namespace Simulators {
// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private {

template <typename T, typename = void>
struct HasSetSeed : std::false_type {};
template <typename T>
struct HasSetSeed<T, std::void_t<decltype(std::declval<T &>().SetSeed(
                         std::declval<uint64_t>()))>> : std::true_type {};

template <typename T>
void SeedBackend(T *backend, uint64_t seed) {
  if constexpr (HasSetSeed<T>::value) backend->SetSeed(seed);
}

// Keep compatibility with the pinned QCSim dependency while allowing local
// and newer QCSim builds to use the summary and batch APIs.
template <typename T, typename = void>
struct HasBondSummary : std::false_type {};
template <typename T>
struct HasBondSummary<T, std::void_t<decltype(std::declval<T &>().SetBondDimensionSummaryCallback(
                            std::declval<std::function<void(Eigen::Index)>>()))>> : std::true_type {};

template <typename T, typename Callback>
void InstallBondSummary(T &backend, Callback callback) {
  if constexpr (HasBondSummary<T>::value) {
    backend.SetBondDimensionSummaryCallback(std::move(callback));
  } else {
    backend.SetBondDimensionCallback([callback = std::move(callback)](const auto &dims) {
      Eigen::Index maximum = 1;
      for (auto dim : dims) maximum = std::max(maximum, dim);
      callback(maximum);
    });
  }
}

template <typename T, typename = void>
struct HasTensorExpectationBatch : std::false_type {};
template <typename T>
struct HasTensorExpectationBatch<T, std::void_t<decltype(std::declval<T &>().ExpectationValues(
                                      std::declval<const std::vector<std::string> &>()))>> : std::true_type {};

template <typename T, typename = void>
struct HasNormalizedTensorExpectationBatch : std::false_type {};
template <typename T>
struct HasNormalizedTensorExpectationBatch<T, std::void_t<decltype(
    std::declval<const T &>().ExpectationValues(
        std::declval<const std::vector<std::string> &>(), true))>>
    : std::true_type {};

template <typename T>
std::vector<std::complex<double>> ComplexTensorExpectationBatch(
    const T &backend, const std::vector<std::string> &paulis, bool normalized) {
  if constexpr (HasNormalizedTensorExpectationBatch<T>::value) {
    return backend.ExpectationValues(paulis, normalized);
  } else {
    std::vector<std::complex<double>> values;
    values.reserve(paulis.size());
    for (const auto &pauli : paulis)
      values.push_back(normalized ? backend.ExpectationValue(pauli)
                                  : backend.UnnormalizedExpectationValue(pauli));
    return values;
  }
}

template <typename T>
std::vector<std::complex<double>> TensorExpectationBatch(
    T &backend, const std::vector<std::string> &paulis) {
  if constexpr (HasTensorExpectationBatch<T>::value) {
    return backend.ExpectationValues(paulis);
  } else {
    std::vector<std::complex<double>> values;
    values.reserve(paulis.size());
    for (const auto &pauli : paulis) {
      if constexpr (std::is_same_v<T, QC::TensorNetworks::MPSSimulator>) {
        static const QC::Gates::PauliXGate<> x;
        static const QC::Gates::PauliYGate<> y;
        static const QC::Gates::PauliZGate<> z;
        std::vector<QC::Gates::AppliedGate<>> gates;
        for (size_t q = 0; q < pauli.size(); ++q) {
          if (pauli[q] == 'X') gates.emplace_back(x.getRawOperatorMatrix(), q);
          if (pauli[q] == 'Y') gates.emplace_back(y.getRawOperatorMatrix(), q);
          if (pauli[q] == 'Z') gates.emplace_back(z.getRawOperatorMatrix(), q);
        }
        values.push_back(backend.ExpectationValue(gates));
      } else {
        values.push_back(backend.ExpectationValue(pauli));
      }
    }
    return values;
  }
}

/**
 * @class ImmediateQCSimState
 * @brief Class for the qcsim simulator state.
 *
 * Implements the qcsim state.
 * Do not use this class directly, use the factory to create an instance.
 * @sa ISimulator
 * @sa IState
 * @sa ImmediateQCSimSimulator
 */

// #define LOG_CALLBACK_INFO 1

class ImmediateQCSimState : public ISimulator {
 public:
  ImmediateQCSimState() : rng(std::random_device{}()), uniformZeroOne(0, 1) {
    meetingPositionCallback = [this](/*const auto &qMap,*/ const auto &bondDims)
        -> QC::TensorNetworks::MPSSimulatorInterface::IndexType {
      try {
        if (lookaheadDepth <= 0 ||
            lookaheadDepth == std::numeric_limits<int>::max())
          return -1;  // will fallback to default behavior

        if (upcomingGates.empty() ||
            upcomingGateIndex >= static_cast<long long>(upcomingGates.size())) {
          return -1;  // will fallback to default behavior
        }

        const size_t nQ = bondDims.size() + 1;

        if (!dummySim || dummySim->getNrQubits() != nQ ||
            dummySim->IsOperatorChain() != (simulationType == SimulationType::kMatrixProductOperator)) {
          dummySim = std::make_unique<Simulators::MPSDummySimulator>(
              nQ, simulationType == SimulationType::kMatrixProductOperator);
          dummySim->SetMaxBondDimension(
              configuration.GetConfigurationAsInt(MaxBondDimensionConfigKey()));
          dummySim->setGrowthFactorGate(growthFactorGate);
          dummySim->setGrowthFactorSwap(growthFactorSwap);
        }

        const auto actualMap = CurrentRoutingMap();
        if (actualMap.size() != nrQubits) return -1;
        dummySim->SetInitialQubitsMap(actualMap);
        dummySim->setTotalSwappingCost(0);

        // Convert actual bond dims to doubles
        std::vector<double> bondDimsD(bondDims.begin(), bondDims.end());
        dummySim->SetCurrentBondDimensions(bondDimsD);

        // display bond dimensions for debugging
#ifdef LOG_CALLBACK_INFO
        std::cerr << "Bond dimensions before swapping and applying the gate:";
        for (size_t i = 0; i < bondDims.size(); ++i) {
          std::cerr << bondDims[i] << " ";
        }
        std::cerr << std::endl;
#endif

        const auto &op = upcomingGates[upcomingGateIndex];
        const auto qbits = op->AffectedQubits();

        if (qbits.size() != 2) return -1;  // will fallback

#ifdef LOG_CALLBACK_INFO
        const auto &qmap = dummySim->getQubitsMap();

        std::cerr << "Applying 2-qubit gate on physical qubits "
                  << qmap[qbits[0]] << " and " << qmap[qbits[1]] << std::endl;

        std::cerr << "Finding best meeting position for upcoming gates "
                     "starting at index "
                  << upcomingGateIndex << " with lookahead depth "
                  << lookaheadDepth << " and heuristic depth "
                  << lookaheadDepthWithHeuristic << std::endl;

        std::cerr << "Affected qubits: ";
        for (const auto &q : qbits) std::cerr << q << " ";
        std::cerr << std::endl;
#endif

        double bestCost = std::numeric_limits<double>::infinity();
        auto res = dummySim->FindBestMeetingPosition(
            upcomingGates, upcomingGateIndex, lookaheadDepth,
            lookaheadDepthWithHeuristic, 0, bestCost);

#ifdef LOG_CALLBACK_INFO
        std::cerr << "Swapping the two qubits on position: " << res << " and "
                  << (res + 1) << std::endl;
#endif

        dummySim->SwapQubitsToPosition(qbits[0], qbits[1], res);
        dummySim->ApplyGate(op);

        // display the expected bond dimensions after applying the gate for
        // debugging

#ifdef LOG_CALLBACK_INFO
        const auto &expectedBondDims = dummySim->getCurrentBondDimensions();
        std::cerr << "Expected bond dimensions after swapping and applying "
                     "the gate: ";
        for (size_t i = 0; i < expectedBondDims.size(); ++i) {
          std::cerr << expectedBondDims[i] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "Best meeting position: " << res
                  << " with estimated cost: " << bestCost << std::endl;
#endif

        return res;
      } catch (...) {
        // Optimization must never make an otherwise valid gate fail.
        return -1;
      }
    };

    bondDimensionCallback = [this](auto maximum) {
      curMaxBondDim = std::max(curMaxBondDim, static_cast<size_t>(maximum));
    };
  }

  /**
   * @brief Initializes the state.
   *
   * This function is called when the simulator is initialized.
   * Call it after the qubits allocation.
   * @sa ImmediateQCSimState::AllocateQubits
   */
  void Initialize() override {
    if (nrQubits != 0) {
      if (simulationType == SimulationType::kMatrixProductState) {
        mpsSimulator =
            std::make_unique<QC::TensorNetworks::MPSSimulator>(nrQubits);

        // default is true
        if (!useOptimalMeetingPosition)
          mpsSimulator->SetUseOptimalMeetingPosition(false);
        InstallBondSummary(*mpsSimulator, bondDimensionCallback);

        curMaxBondDim = 1;
      } else if (simulationType == SimulationType::kMatrixProductOperator) {
        mpoSimulator =
            std::make_unique<QC::TensorNetworks::MPOSimulator>(nrQubits);
        if (!useOptimalMeetingPosition)
          mpoSimulator->SetUseOptimalMeetingPosition(false);
        InstallBondSummary(*mpoSimulator, bondDimensionCallback);
        curMaxBondDim = 1;
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
      } else if (simulationType == SimulationType::kPathIntegral) {
        pathIntegralSimulator = std::make_unique<PathIntegralSimulator>();
        pathIntegralSimulator->SetStartZeroState(nrQubits);
      } else if (simulationType == SimulationType::kDensityMatrix) {
        densityMatrix = std::make_unique<QC::DensityMatrix<>>(nrQubits);
      } else if (simulationType == SimulationType::kExtendedStabilizer) {
        extendedStabilizer =
            std::make_unique<Simulators::QCSimExtendedStabilizer>(nrQubits);
      } else
        state = std::make_unique<QC::QubitRegister<>>(nrQubits);

      SetMultithreading(enableMultithreading);

      // ensure the config settings are applied, they need to be applied after
      // the simulator is created
      for (const auto &[key, value] : configuration.GetConfigMap())
        if (key != "method") Configure(key.c_str(), value.c_str());
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
    if (simulationType != SimulationType::kStatevector &&
        simulationType != SimulationType::kDensityMatrix)
      throw std::runtime_error(
          "QCSimState::InitializeState: Invalid "
          "simulation type for initializing the state.");

    Eigen::VectorXcd amplitudesEigen(
        Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(),
                                                       amplitudes.size()));
    if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->setFromStatevector(amplitudesEigen);
    else
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
    if (simulationType != SimulationType::kStatevector &&
        simulationType != SimulationType::kDensityMatrix)
      throw std::runtime_error(
          "QCSimState::InitializeState: Invalid "
          "simulation type for initializing the state.");

    Eigen::VectorXcd amplitudesEigen(
        Eigen::Map<Eigen::VectorXcd, Eigen::Unaligned>(amplitudes.data(),
                                                       amplitudes.size()));
    if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->setFromStatevector(amplitudesEigen);
    else
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

    if (simulationType != SimulationType::kStatevector &&
        simulationType != SimulationType::kDensityMatrix)
      throw std::runtime_error(
          "QCSimState::InitializeState: Invalid "
          "simulation type for initializing the state.");

    if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->setFromStatevector(amplitudes);
    else {
      state = std::make_unique<QC::QubitRegister<>>(nrQubits, amplitudes);
      state->SetMultithreading(enableMultithreading);
    }
  }

  /**
   * @brief Initializes the state to a computational basis state.
   *
   * The density matrix, matrix product operator, matrix product state and
   * statevector methods use their own direct primitive; every other method
   * falls back to the generic ISimulator implementation (reset to |0...0>,
   * then apply X on every set bit).
   *
   * @param num_qubits The number of qubits to initialize the state with.
   * @param basisState The computational basis state, bit i selects qubit i.
   */
  void InitializeToBasisState(size_t num_qubits,
                              Types::qubit_t basisState) override {
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();

    if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->setToBasisState(static_cast<size_t>(basisState));
    else if (simulationType == SimulationType::kMatrixProductOperator)
      mpoSimulator->setToBasisState(static_cast<size_t>(basisState));
    else if (simulationType == SimulationType::kMatrixProductState)
      mpsSimulator->setToBasisState(static_cast<size_t>(basisState));
    else if (simulationType == SimulationType::kStatevector)
      state->setToBasisState(static_cast<size_t>(basisState));
    else
      for (size_t q = 0; q < num_qubits; ++q)
        if ((basisState >> q) & 1ULL) ApplyX(static_cast<Types::qubit_t>(q));
  }

  /**
   * @brief Initializes the state to a computational basis state.
   *
   * Same as the Types::qubit_t overload, but not limited to 64 qubits. The
   * matrix product operator and matrix product state methods use their own
   * direct primitive; every other method falls back to the generic
   * ISimulator implementation (reset to |0...0>, then apply X on every set
   * bit) - which is not limited either, unlike the other methods' native
   * Types::qubit_t-based primitives.
   *
   * @param num_qubits The number of qubits to initialize the state with.
   * @param basisState The computational basis state, entry i selects qubit i.
   */
  void InitializeToBasisState(size_t num_qubits,
                              const std::vector<bool> &basisState) override {
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();

    if (simulationType == SimulationType::kMatrixProductOperator)
      mpoSimulator->setToBasisState(basisState);
    else if (simulationType == SimulationType::kMatrixProductState)
      mpsSimulator->setToBasisState(basisState);
    else
      for (size_t q = 0; q < num_qubits && q < basisState.size(); ++q)
        if (basisState[q]) ApplyX(static_cast<Types::qubit_t>(q));
  }

  /**
   * @brief Initializes the state to a classical mixture of computational
   * basis states.
   *
   * Works for the density matrix and matrix product operator methods. There
   * is no generic fallback for other methods, since a mixture cannot be
   * reached with unitary gates alone.
   *
   * @param num_qubits The number of qubits to initialize the state with.
   * @param mixture The mixture, as pairs of (basis state, weight).
   */
  void InitializeToMixtureOfBasisStates(
      size_t num_qubits,
      const std::vector<std::pair<Types::qubit_t, double>> &mixture) override {
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();

    if (simulationType != SimulationType::kDensityMatrix &&
        simulationType != SimulationType::kMatrixProductOperator)
      throw std::runtime_error(
          "QCSimState::InitializeToMixtureOfBasisStates: Invalid simulation "
          "type for initializing to a mixture of basis states.");

    std::vector<std::pair<size_t, double>> converted;
    converted.reserve(mixture.size());
    for (const auto &[basisState, weight] : mixture)
      converted.emplace_back(static_cast<size_t>(basisState), weight);

    if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->setToMixtureOfBasisStates(converted);
    else
      mpoSimulator->setToMixtureOfBasisStates(converted);
  }

  /**
   * @brief Initializes the state to a classical mixture of computational
   * basis states.
   *
   * Same as the Types::qubit_t-keyed overload, but not limited to 64 qubits.
   * Only the matrix product operator method supports this.
   *
   * @param num_qubits The number of qubits to initialize the state with.
   * @param mixture The mixture, as pairs of (basis state, weight).
   */
  void InitializeToMixtureOfBasisStates(
      size_t num_qubits,
      const std::vector<std::pair<std::vector<bool>, double>> &mixture)
      override {
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();

    if (simulationType != SimulationType::kMatrixProductOperator)
      throw std::runtime_error(
          "QCSimState::InitializeToMixtureOfBasisStates: Invalid simulation "
          "type for initializing to a mixture of basis states.");

    mpoSimulator->setToMixtureOfBasisStates(mixture);
  }

  /**
   * @brief Just resets the state to 0.
   *
   * Does not destroy the internal state, just resets it to zero (as a 'reset'
   * op on each qubit would do).
   */
  void Reset() override {
    if (mpsSimulator) {
      mpsSimulator->Clear();
      curMaxBondDim = 1;
    } else if (mpoSimulator) {
      mpoSimulator->Clear();
      curMaxBondDim = 1;
    } else if (cliffordSimulator)
      cliffordSimulator->Reset();
    else if (tensorNetwork)
      tensorNetwork->Clear();
    else if (state)
      state->Reset();
    else if (pp)
      pp->ClearOperations();
    else if (pathIntegralSimulator) {
      pathIntegralSimulator->Reset();
      pathIntegralSimulator->SetStartZeroState(nrQubits);
    } else if (densityMatrix)
      densityMatrix->Reset();
    else if (extendedStabilizer)
      extendedStabilizer->Reset(nrQubits);

    upcomingGateIndex = 0;
    ResetDummySimulator();
  }

  /**
   * @brief Returns if the simulator supports MPS swap optimization.
   *
   * Used to check if the simulator supports MPS swap optimization.
   * @return True if the simulator supports MPS swap optimization, false
   * otherwise.
   */
  bool SupportsMPSSwapOptimization() const override { return true; }
  bool IsRoutingLookaheadEnabled() const override {
    return useOptimalMeetingPosition && lookaheadDepth > 0 &&
           lookaheadDepth != std::numeric_limits<int>::max() && nrQubits != 0 &&
           CurrentRoutingMap().size() == nrQubits;
  }

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
    if (mpsSimulator || mpoSimulator) {
      if (mpsSimulator)
        mpsSimulator->SetInitialQubitsMap(initialMap);
      else
        mpoSimulator->SetInitialQubitsMap(initialMap);

      if (!dummySim || dummySim->getNrQubits() != initialMap.size() ||
          dummySim->IsOperatorChain() != (simulationType == SimulationType::kMatrixProductOperator)) {
        dummySim = std::make_unique<Simulators::MPSDummySimulator>(
            initialMap.size(), simulationType == SimulationType::kMatrixProductOperator);
        dummySim->SetMaxBondDimension(
            configuration.GetConfigurationAsInt(MaxBondDimensionConfigKey()));
      }
      dummySim->setGrowthFactorGate(growthFactorGate);
      dummySim->setGrowthFactorSwap(growthFactorSwap);
      dummySim->SetInitialQubitsMap(initialMap);
    }
  }

  void SetUseOptimalMeetingPosition(bool enable) override {
    useOptimalMeetingPosition = enable;
    if (mpsSimulator) mpsSimulator->SetUseOptimalMeetingPosition(enable);
    if (mpoSimulator) mpoSimulator->SetUseOptimalMeetingPosition(enable);
    RefreshRoutingCallback();
  }
  void SetLookaheadDepth(int depth) override {
    lookaheadDepth = depth;
    RefreshRoutingCallback();
  }
  void SetLookaheadDepthWithHeuristic(int depth) override {
    lookaheadDepthWithHeuristic = depth;
    if (lookaheadDepth < depth) SetLookaheadDepth(depth);
  }

  void SetUpcomingGates(
      const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &gates)
      override {
    upcomingGates = gates;
    upcomingGateIndex = 0;
    if (!gateCounterObserver)
      gateCounterObserver =
          std::make_shared<GateCounterObserver>(upcomingGateIndex);
    RegisterObserver(gateCounterObserver);
    RefreshRoutingCallback();
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
  void IncrementGatesCounter() override { ++upcomingGateIndex; }

  double getGrowthFactorSwap() const override { return growthFactorSwap; }
  double getGrowthFactorGate() const override { return growthFactorGate; }

  void setGrowthFactorSwap(double factor) override {
    growthFactorSwap = factor;
    if (dummySim) dummySim->setGrowthFactorSwap(factor);
  }

  void setGrowthFactorGate(double factor) override {
    growthFactorGate = factor;
    if (dummySim) dummySim->setGrowthFactorGate(factor);
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
      else if (std::string("matrix_product_operator") == value)
        simulationType = SimulationType::kMatrixProductOperator;
      else if (std::string("stabilizer") == value)
        simulationType = SimulationType::kStabilizer;
      else if (std::string("tensor_network") == value)
        simulationType = SimulationType::kTensorNetwork;
      else if (std::string("pauli_propagator") == value)
        simulationType = SimulationType::kPauliPropagator;
      else if (std::string("path_integral") == value)
        simulationType = SimulationType::kPathIntegral;
      else if (std::string("density_matrix") == value)
        simulationType = SimulationType::kDensityMatrix;
      else if (std::string("extended_stabilizer") == value)
        simulationType = SimulationType::kExtendedStabilizer;
    }

    if (simulationType == SimulationType::kMatrixProductOperator &&
        (std::string(key) == "matrix_product_state_max_bond_dimension" ||
         std::string(key) == "matrix_product_operator_max_bond_dimension")) {
      const auto limit = std::stoll(value);
      if (limit < 0) throw std::invalid_argument("Negative MPO bond dimension");
      if (mpoSimulator) {
        if (limit == 0)
          mpoSimulator->dontLimitBondDimension();
        else
          mpoSimulator->setLimitBondDimension(limit);
      }
      const std::string limitValue(value);
      configuration.SetConfiguration("matrix_product_state_max_bond_dimension",
                                     limitValue);
      configuration.SetConfiguration(
          "matrix_product_operator_max_bond_dimension", limitValue);
      if (dummySim) dummySim->SetMaxBondDimension(limit);
      return;
    }

    if (!configuration.WasApplied(key, value))
      configuration.SetConfiguration(key, value);

    if (std::string("seed") == key) {
      const uint64_t seed = std::stoull(value);
      SeedAuxiliaryRng(seed);
      nextSeedStream = 0;
      rng.seed(seed);
      if (state) SeedBackend(state.get(), seed);
      if (mpsSimulator) SeedBackend(mpsSimulator.get(), seed);
      if (mpoSimulator) SeedBackend(mpoSimulator.get(), seed);
      if (cliffordSimulator) SeedBackend(cliffordSimulator.get(), seed);
      if (tensorNetwork) tensorNetwork->SetSeed(seed);
      if (pp) SeedBackend(pp.get(), seed);
      if (pathIntegralSimulator) pathIntegralSimulator->SetSeed(seed);
      if (densityMatrix) SeedBackend(densityMatrix.get(), seed);
      if (extendedStabilizer) SeedBackend(extendedStabilizer.get(), seed);
      return;
    }

    if (mpsSimulator) {
      if (std::string(key) == "matrix_product_state_max_bond_dimension") {
        mpsSimulator->setLimitBondDimension(
            configuration.GetConfigurationAsInt(key));
      } else if (std::string(key) ==
                 "matrix_product_state_truncation_threshold") {
        const double threshold = configuration.GetConfigurationAsDouble(key);
        // Zero must also replace any previously configured cutoff.
        if (threshold >= 0.) mpsSimulator->setLimitEntanglement(threshold);
      } else if (std::string(key) == "matrix_product_state_truncation_mode") {
        // "relative_max" -> RelativeToMax, "discarded_weight" ->
        // DiscardedWeight (the default -- see
        // QC::TensorNetworks::MPSSimulatorInterface::TruncationMode).
        if (std::string(value) == "relative_max")
          mpsSimulator->setTruncationMode(
              QC::TensorNetworks::MPSSimulator::TruncationMode::RelativeToMax);
        else if (std::string(value) == "discarded_weight")
          mpsSimulator->setTruncationMode(QC::TensorNetworks::MPSSimulator::
                                              TruncationMode::DiscardedWeight);
      }
    }

    if (mpoSimulator) {
      if (std::string(key) == "matrix_product_state_max_bond_dimension" ||
          std::string(key) == "matrix_product_operator_max_bond_dimension") {
        mpoSimulator->setLimitBondDimension(
            configuration.GetConfigurationAsInt(key));
      } else if (std::string(key) ==
                     "matrix_product_state_truncation_threshold" ||
                 std::string(key) ==
                     "matrix_product_operator_truncation_threshold") {
        const double threshold = configuration.GetConfigurationAsDouble(key);
        // Zero must also replace any previously configured cutoff.
        if (threshold >= 0.) mpoSimulator->setLimitEntanglement(threshold);
      } else if (std::string(key) == "matrix_product_state_truncation_mode" ||
                 std::string(key) ==
                     "matrix_product_operator_truncation_mode") {
        // See the mpsSimulator branch above for the value convention.
        if (std::string(value) == "relative_max")
          mpoSimulator->setTruncationMode(
              QC::TensorNetworks::MPOSimulator::TruncationMode::RelativeToMax);
        else if (std::string(value) == "discarded_weight")
          mpoSimulator->setTruncationMode(QC::TensorNetworks::MPOSimulator::
                                              TruncationMode::DiscardedWeight);
      } else if (std::string(key) ==
                 "matrix_product_operator_kraus_completeness_check") {
        using Check = QC::TensorNetworks::MPOSimulator::KrausCompletenessCheck;
        if (std::string(value) == "ignore")
          mpoSimulator->setKrausCompletenessCheck(Check::Ignore);
        else if (std::string(value) == "warn")
          mpoSimulator->setKrausCompletenessCheck(Check::Warn);
        else if (std::string(value) == "strict")
          mpoSimulator->setKrausCompletenessCheck(Check::Strict);
      } else if (std::string(key) ==
                 "matrix_product_operator_restore_trace_after_truncation") {
        mpoSimulator->setRestoreTraceAfterTruncation(
            std::string(value) == "1" || std::string(value) == "true");
      } else if (std::string(key) ==
                 "matrix_product_operator_hermitize_after_truncation") {
        mpoSimulator->setHermitizeAfterTruncation(std::string(value) == "1" ||
                                                  std::string(value) == "true");
      }
    }

    if (std::string(key) == "pauli_propagator_workers") {
      const auto workers = configuration.GetConfigurationAsUnsigned(key);
      if (workers > 1024)
        throw std::invalid_argument("pauli_propagator_workers exceeds 1024");
      pauliWorkerCount = static_cast<size_t>(workers);
      if (pp && enableMultithreading) pp->EnableParallel(pauliWorkerCount);
    }
    if (pp) {
      if (std::string(key) == "pauli_propagator_sampling_cache_nodes") {
        pp->SetSamplingCacheMaxNodes(
            configuration.GetConfigurationAsUnsigned(key));
      } else if (std::string(key) == "pauli_propagator_coefficient_threshold") {
        pp->SetCoefficientThreshold(
            configuration.GetConfigurationAsDouble(key));
      } else if (std::string(key) ==
                 "pauli_propagator_pauli_weight_threshold") {
        pp->SetPauliWeightThreshold(
            configuration.GetConfigurationAsUnsigned(key));
      } else if (std::string(key) == "pauli_propagator_steps_between_trims") {
        pp->SetStepsBetweenTrims(configuration.GetConfigurationAsInt(key));
      } else if (std::string(key) ==
                 "pauli_propagator_num_gates_between_deduplications") {
        pp->SetStepsBetweenDeduplication(
            configuration.GetConfigurationAsInt(key));
      }
    }

    if (pathIntegralSimulator) {
      if (std::string(key) == "path_integral_threshold") {
        pathIntegralSimulator->SetTrimValue(
            configuration.GetConfigurationAsDouble(key));
      }
    }
  }

  /**
   * @brief Returns configuration value.
   *
   * This function is called get a configuration value.
   * @param key The key of the configuration value.
   * @return The configuration value as a string.
   */
  std::string GetConfiguration(const char *key) const override {
    if (!key) return {};
    if (std::string(key) == "precision") return "double";
    if (std::string(key) == "use_double_precision") return "1";
    if (mpoSimulator) {
      if (std::string(key) ==
          "matrix_product_operator_restore_trace_after_truncation")
        return mpoSimulator->getRestoreTraceAfterTruncation() ? "true"
                                                              : "false";
      if (std::string(key) ==
          "matrix_product_operator_hermitize_after_truncation")
        return mpoSimulator->getHermitizeAfterTruncation() ? "true" : "false";
    }
    if (std::string("method") == key) {
      switch (simulationType) {
        case SimulationType::kStatevector:
          return "statevector";
        case SimulationType::kMatrixProductState:
          return "matrix_product_state";
        case SimulationType::kMatrixProductOperator:
          return "matrix_product_operator";
        case SimulationType::kStabilizer:
          return "stabilizer";
        case SimulationType::kTensorNetwork:
          return "tensor_network";
        case SimulationType::kPauliPropagator:
          return "pauli_propagator";
        case SimulationType::kPathIntegral:
          return "path_integral";
        case SimulationType::kDensityMatrix:
          return "density_matrix";
        case SimulationType::kExtendedStabilizer:
          return "extended_stabilizer";
        default:
          return "other";
      }
    }

    return configuration.GetConfiguration(key);
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
        (simulationType == SimulationType::kMatrixProductOperator &&
         mpoSimulator) ||
        (simulationType == SimulationType::kStabilizer && cliffordSimulator) ||
        (simulationType == SimulationType::kTensorNetwork && tensorNetwork) ||
        (simulationType == SimulationType::kDensityMatrix && densityMatrix) ||
        (simulationType == SimulationType::kExtendedStabilizer &&
         extendedStabilizer))
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
    mpoSimulator = nullptr;
    cliffordSimulator = nullptr;
    tensorNetwork = nullptr;
    pp = nullptr;
    pathIntegralSimulator = nullptr;
    densityMatrix = nullptr;
    extendedStabilizer = nullptr;
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
    if (mpoSimulator && qubits.size() > sizeof(size_t) * 8)
      throw std::invalid_argument(
          "Use MeasureMany for more than 64 measured qubits");
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
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t qubit : qubits) {
        if (densityMatrix->MeasureQubit(qubit)) res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
      const auto measured = mpoSimulator->MeasureQubits(qubitsSet);
      for (Types::qubit_t qubit : qubits) {
        if (measured.at(static_cast<Eigen::Index>(qubit))) res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      for (size_t qubit : qubits) {
        if (extendedStabilizer->Measure(qubit)) res |= mask;
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
      std::vector<int> qubitsInt;
      qubitsInt.reserve(qubits.size());
      for (const auto q : qubits) qubitsInt.push_back(static_cast<int>(q));
      const auto res = pp->Measure(qubitsInt);
      Types::qubit_t result = 0;
      for (size_t i = 0; i < res.size(); ++i) {
        if (res[i]) result |= mask;
        mask <<= 1;
      }
      return result;
    } else if (simulationType == SimulationType::kPathIntegral) {
      for (size_t qubit : qubits) {
        if (pathIntegralSimulator->MeasureQubit(qubit)) res |= mask;
        mask <<= 1;
      }
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
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t q = 0; q < qubits.size(); ++q)
        if (densityMatrix->MeasureQubit(qubits[q])) res[q] = true;
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
      const auto measured = mpoSimulator->MeasureQubits(qubitsSet);
      for (size_t q = 0; q < qubits.size(); ++q)
        res[q] = measured.at(static_cast<Eigen::Index>(qubits[q]));
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      for (size_t q = 0; q < qubits.size(); ++q)
        if (extendedStabilizer->Measure(qubits[q])) res[q] = true;
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
    } else if (simulationType == SimulationType::kPathIntegral) {
      for (size_t q = 0; q < qubits.size(); ++q)
        if (pathIntegralSimulator->MeasureQubit(qubits[q])) res[q] = true;
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
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t qubit : qubits) densityMatrix->ApplyReset(qubit);
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      for (size_t qubit : qubits)
        mpoSimulator->ApplyReset(static_cast<Eigen::Index>(qubit));
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      for (size_t qubit : qubits)
        if (extendedStabilizer->Measure(qubit))
          extendedStabilizer->ApplyX(qubit);
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
    } else if (simulationType == SimulationType::kPathIntegral) {
      for (size_t qubit : qubits)
        if (pathIntegralSimulator->MeasureQubit(qubit)) {
          QC::Gates::AppliedGate<> gate(xGate.getRawOperatorMatrix(), qubit);
          pathIntegralSimulator->PropagateStep(
              gate, pathIntegralSimulator->Amplitudes());
        }
    } else {
      for (size_t qubit : qubits)
        if (mpsSimulator->MeasureQubit(static_cast<unsigned int>(qubit)))
          mpsSimulator->ApplyGate(xGate, static_cast<unsigned int>(qubit));
    }
    Notify();

    NotifyObservers(qubits);
  }

  bool SupportsQuantumChannels() const override {
    return simulationType == SimulationType::kDensityMatrix ||
           simulationType == SimulationType::kMatrixProductOperator;
  }

  /**
   * @brief Applies a deterministic one- or two-qubit CPTP channel.
   *
   * QCSim's dense backend calls DensityMatrix::ApplyChannel and its MPO
   * backend calls MPOSimulator::ApplyKrausOperators. QuantumChannel performs
   * the common completeness validation first, since the native MPO primitive
   * also permits non-trace-preserving maps. MPO evolution is subject to any
   * configured bond-dimension or singular-value truncation.
   */
  void ApplyQuantumChannel(const Types::qubits_vector &targets,
                           const QuantumChannel &channel) override {
    if (!SupportsQuantumChannels())
      throw std::runtime_error(
          "QCSim quantum channels require density_matrix or "
          "matrix_product_operator simulation");
    if (targets.size() != channel.GetNumberOfQubits())
      throw std::invalid_argument(
          "The number of channel targets does not match its Kraus operators");
    if (targets.empty() || targets.size() > 2)
      throw std::invalid_argument(
          "QCSim supports only one- and two-qubit local channels");

    std::unordered_set<Types::qubit_t> uniqueTargets;
    for (const Types::qubit_t target : targets) {
      if (target >= nrQubits)
        throw std::invalid_argument("Quantum-channel qubit is out of range");
      if (!uniqueTargets.insert(target).second)
        throw std::invalid_argument(
            "Quantum-channel target qubits must be distinct");
    }

    const auto &krausOperators = channel.GetKrausOperators();
    if (simulationType == SimulationType::kDensityMatrix) {
      if (!densityMatrix)
        throw std::runtime_error(
            "QCSim density-matrix state is not initialized");
      if (targets.size() == 1)
        densityMatrix->ApplyChannel(krausOperators, targets[0]);
      else
        densityMatrix->ApplyChannel(krausOperators, targets[0], targets[1]);
    } else {
      if (!mpoSimulator)
        throw std::runtime_error("QCSim MPO state is not initialized");
      if (targets.size() == 1)
        mpoSimulator->ApplyKrausOperators(
            krausOperators, static_cast<Eigen::Index>(targets[0]));
      else
        mpoSimulator->ApplyKrausOperators(
            krausOperators, static_cast<Eigen::Index>(targets[0]),
            static_cast<Eigen::Index>(targets[1]));
    }

    NotifyObservers(targets);
  }

  double ProbabilityBits(const std::vector<bool> &bits) override {
    if (!mpoSimulator) return IState::ProbabilityBits(bits);
    MPOValidation::Bits(bits, nrQubits);
    return mpoSimulator->getBasisStateProbability(bits);
  }
  std::complex<double> DensityMatrixElementBits(
      const std::vector<bool> &row,
      const std::vector<bool> &col) const override {
    if (!mpoSimulator) return IState::DensityMatrixElementBits(row, col);
    MPOValidation::Bits(row, nrQubits);
    MPOValidation::Bits(col, nrQubits);
    return mpoSimulator->getBasisStateMatrixElement(row, col);
  }
  Eigen::MatrixXcd GetDensityMatrix(bool normalized = true) const override {
    if (!mpoSimulator) return IState::GetDensityMatrix(normalized);
    if (nrQubits > 13)
      throw std::invalid_argument(
          "MPO dense output supports at most 13 qubits");
    return normalized ? mpoSimulator->getDensityMatrix()
                      : mpoSimulator->getUnnormalizedDensityMatrix();
  }
  std::complex<double> ExpectationValueComplex(
      const std::string &pauli, bool normalized = true) const override {
    if (!mpoSimulator)
      return IState::ExpectationValueComplex(pauli, normalized);
    MPOValidation::Pauli(pauli, nrQubits);
    return normalized ? mpoSimulator->ExpectationValue(pauli)
                      : mpoSimulator->UnnormalizedExpectationValue(pauli);
  }
  std::vector<std::complex<double>> ExpectationValuesComplex(
      const std::vector<std::string> &paulis,
      bool normalized = true) const override {
    if (!mpoSimulator)
      return IState::ExpectationValuesComplex(paulis, normalized);
    for (const auto &pauli : paulis) MPOValidation::Pauli(pauli, nrQubits);
    return ComplexTensorExpectationBatch(*mpoSimulator, paulis, normalized);
  }
  void ApplyOperator(const Types::qubits_vector &qubits,
                     const Eigen::MatrixXcd &matrix,
                     bool normalize = false) override {
    if (!mpoSimulator) return IState::ApplyOperator(qubits, matrix, normalize);
    const auto selected = MPOValidation::Operator(qubits, matrix, nrQubits);
    const QC::Gates::AppliedGate<> op(matrix, selected[0],
                                      selected.size() == 2 ? selected[1] : 0);
    if (normalize)
      mpoSimulator->ApplyOperatorAndNormalize(op);
    else
      mpoSimulator->ApplyOperator(op);
    NotifyObservers(qubits);
  }
  void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override {
    if (!mpoSimulator) return IState::MoveAtBeginningOfChain(qubits);
    const auto selected = MPOValidation::Qubits(qubits, nrQubits);
    mpoSimulator->MoveAtBeginningOfChain(
        std::set<Eigen::Index>(selected.begin(), selected.end()));
  }

  std::complex<double> DensityMatrixTrace() const override {
    if (densityMatrix) return densityMatrix->Trace();
    if (mpoSimulator) return mpoSimulator->Trace();
    throw std::runtime_error(
        "Mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  double DensityMatrixPurity() const override {
    if (densityMatrix) return densityMatrix->Purity();
    if (mpoSimulator) return mpoSimulator->Purity();
    throw std::runtime_error(
        "Mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  std::complex<double> DensityMatrixTraceOfSquare() const override {
    if (densityMatrix) {
      const auto &rho = densityMatrix->getDensityMatrix();
      return (rho * rho).trace();
    }
    if (mpoSimulator) return mpoSimulator->TraceOfSquare();
    throw std::runtime_error(
        "Mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  std::complex<double> DensityMatrixOverlap(
      const IState &other) const override {
    const auto *rhs = dynamic_cast<const ImmediateQCSimState *>(&other);
    if (!rhs)
      throw std::invalid_argument(
          "Density-matrix overlap requires matching QCSim backends");
    if (densityMatrix && rhs->densityMatrix)
      return densityMatrix->HilbertSchmidtOverlap(*rhs->densityMatrix);
    if (mpoSimulator && rhs->mpoSimulator)
      return mpoSimulator->HilbertSchmidtOverlap(*rhs->mpoSimulator);
    throw std::invalid_argument(
        "Density-matrix overlap requires two density matrices or two MPOs");
  }
  double DensityMatrixHermiticityResidual() const override {
    if (densityMatrix)
      return (densityMatrix->getDensityMatrix() -
              densityMatrix->getDensityMatrix().adjoint())
          .norm();
    if (mpoSimulator) return mpoSimulator->HermiticityResidual();
    throw std::runtime_error(
        "Mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  bool IsDensityMatrixHermitian(double eps = 1e-10) const override {
    if (densityMatrix) return densityMatrix->IsHermitian(eps);
    if (mpoSimulator) return mpoSimulator->IsHermitian(eps);
    throw std::runtime_error(
        "Mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  Eigen::MatrixXcd PartialTrace(
      const Types::qubits_vector &qubits) const override {
    if (densityMatrix)
      return densityMatrix->PartialTrace(
          std::vector<size_t>(qubits.begin(), qubits.end()));
    if (mpoSimulator)
      return mpoSimulator->PartialTrace(
          std::vector<Eigen::Index>(qubits.begin(), qubits.end()));
    throw std::runtime_error(
        "Partial trace requires density_matrix or matrix_product_operator");
  }
  double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override {
    if (densityMatrix) return densityMatrix->FidelityWithStatevector(psi);
    if (mpoSimulator) return mpoSimulator->FidelityWithStatevector(psi);
    throw std::runtime_error(
        "Mixed-state fidelity requires density_matrix or "
        "matrix_product_operator");
  }
  void RestoreDensityMatrixTrace() override {
    if (!mpoSimulator)
      throw std::runtime_error(
          "Trace restoration is only available for QCSim MPO");
    mpoSimulator->RestoreTrace();
  }
  void HermitizeDensityMatrix() override {
    if (!mpoSimulator)
      throw std::runtime_error("Hermitization is only available for QCSim MPO");
    mpoSimulator->Hermitize();
  }
  void Trim() override {
    if (mpsSimulator)
      mpsSimulator->Trim();
    else if (mpoSimulator)
      mpoSimulator->Trim();
    else
      throw std::runtime_error("Trim is only available for QCSim MPS and MPO");
  }
  void ReCanonicalize() override {
    if (mpsSimulator)
      mpsSimulator->ReCanonicalize();
    else if (mpoSimulator)
      mpoSimulator->ReCanonicalize();
    else
      throw std::runtime_error(
          "Canonicalization is only available for QCSim MPS and MPO");
  }

  /**
   * @brief Returns the probability of the specified outcome.
   *
   * Use it to obtain the probability to obtain the specified outcome, if all
   * qubits are measured.
   * @sa ImmediateQCSimState::Amplitude
   * @sa ImmediateQCSimState::Probabilities
   *
   * @param outcome The outcome to obtain the probability for.
   * @return The probability of the specified outcome.
   */
  double Probability(Types::qubit_t outcome) override {
    if (simulationType == SimulationType::kMatrixProductState)
      return mpsSimulator->getBasisStateProbability(
          static_cast<unsigned int>(outcome));
    else if (simulationType == SimulationType::kStabilizer) {
      if constexpr (std::numeric_limits<size_t>::digits >=
                    std::numeric_limits<Types::qubit_t>::digits) {
        return cliffordSimulator->getBasisStateProbability(
            static_cast<size_t>(outcome));
      } else {
        const size_t n = cliffordSimulator->getNrQubits();
        if (n < std::numeric_limits<Types::qubit_t>::digits &&
            (outcome >> n) != 0)
          return 0.0;
        std::vector<bool> bits(n);
        for (size_t q = 0;
             q < n && q < std::numeric_limits<Types::qubit_t>::digits; ++q)
          bits[q] = ((outcome >> q) & 1) != 0;
        return cliffordSimulator->getBasisStateProbability(bits);
      }
    } else if (simulationType == SimulationType::kTensorNetwork)
      return tensorNetwork->getBasisStateProbability(outcome);
    else if (simulationType == SimulationType::kPauliPropagator)
      return pp->Probability(outcome);
    else if (simulationType == SimulationType::kPathIntegral)
      return pathIntegralSimulator->Probability(outcome);
    else if (simulationType == SimulationType::kDensityMatrix)
      return densityMatrix->getBasisStateProbability(outcome);
    else if (simulationType == SimulationType::kMatrixProductOperator) {
      if (nrQubits < sizeof(Types::qubit_t) * 8 &&
          outcome >= (Types::qubit_t{1} << nrQubits))
        throw std::out_of_range("MPO basis state is out of range");
      std::vector<bool> bits(nrQubits);
      for (size_t q = 0; q < std::min(nrQubits, sizeof(Types::qubit_t) * 8);
           ++q)
        bits[q] = ((outcome >> q) & 1) != 0;
      return ProbabilityBits(bits);
    } else if (simulationType == SimulationType::kExtendedStabilizer)
      return ExtendedStabilizerBasisProbability(outcome);

    return state->getBasisStateProbability(static_cast<unsigned int>(outcome));
  }

  /**
   * @brief Returns the amplitude of the specified state.
   *
   * Use it to obtain the amplitude of the specified state.
   * @sa ImmediateQCSimState::Probability
   * @sa ImmediateQCSimState::Probabilities
   *
   * @param outcome The outcome to obtain the amplitude for.
   * @return The amplitude of the specified outcome.
   */
  std::complex<double> Amplitude(Types::qubit_t outcome) override {
    if (simulationType == SimulationType::kMatrixProductState)
      return mpsSimulator->getBasisStateAmplitude(
          static_cast<unsigned int>(outcome));
    else if (simulationType == SimulationType::kPathIntegral)
      return pathIntegralSimulator->AmplitudeForOutcome(outcome);
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
    else if (simulationType == SimulationType::kDensityMatrix)
      throw std::runtime_error(
          "QCSimState::Amplitude: Amplitudes are not defined for the density "
          "matrix simulator.");
    else if (simulationType == SimulationType::kMatrixProductOperator)
      throw std::runtime_error(
          "QCSimState::Amplitude: Amplitudes are not defined for the matrix "
          "product operator simulator.");
    else if (simulationType == SimulationType::kExtendedStabilizer)
      throw std::runtime_error(
          "QCSimState::Amplitude: Amplitudes are not exposed by the extended "
          "stabilizer simulator.");

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
   * @sa ImmediateQCSimState::Probability
   * @sa ImmediateQCSimState::Amplitude
   * @sa ImmediateQCSimState::AllProbabilities
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
    } else if (simulationType == SimulationType::kPathIntegral) {
      const size_t nrBasisStates = 1ULL << GetNumberOfQubits();
      std::vector<double> result(nrBasisStates);
      for (size_t i = 0; i < nrBasisStates; ++i)
        result[i] = pathIntegralSimulator->Probability(i);
      return result;
    } else if (simulationType == SimulationType::kDensityMatrix) {
      const size_t nrBasisStates = densityMatrix->getNrBasisStates();
      std::vector<double> result(nrBasisStates);
      for (size_t i = 0; i < nrBasisStates; ++i)
        result[i] = densityMatrix->getBasisStateProbability(i);
      return result;
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      const size_t nrBasisStates = CheckedBasisStateCountForQueries();
      std::vector<double> result(nrBasisStates);
      for (size_t i = 0; i < nrBasisStates; ++i)
        result[i] = mpoSimulator->getBasisStateProbability(i);
      return result;
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      const size_t nrBasisStates = CheckedBasisStateCountForQueries();
      std::vector<double> result(nrBasisStates);
      for (size_t i = 0; i < nrBasisStates; ++i)
        result[i] = ExtendedStabilizerBasisProbability(i);
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
   * @sa ImmediateQCSimState::Probability
   * @sa ImmediateQCSimState::Amplitude
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
    } else if (simulationType == SimulationType::kPathIntegral) {
      for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
        result[i] = pathIntegralSimulator->Probability(qubits[i]);
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
        result[i] = densityMatrix->getBasisStateProbability(qubits[i]);
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
        result[i] = Probability(qubits[i]);
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      for (int i = 0; i < static_cast<int>(qubits.size()); ++i)
        result[i] = ExtendedStabilizerBasisProbability(qubits[i]);
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
    if (mpoSimulator && qubits.size() > sizeof(Types::qubit_t) * 8)
      throw std::invalid_argument(
          "Use SampleCountsMany for more than 64 measured qubits");
    if (mpoSimulator)
      for (auto q : qubits)
        if (q >= nrQubits)
          throw std::out_of_range("MPO sampled qubit is out of range");
    if (qubits.empty() || shots == 0) return {};

    if (simulationType == SimulationType::kStabilizer) {
      const auto selected = CliffordSamplingQubits(qubits);
      const auto counts = cliffordSimulator->SampleCounts(selected, shots);
      std::unordered_map<Types::qubit_t, Types::qubit_t> result;
      for (const auto &item : counts) result.emplace(item.first, item.second);
      NotifyObservers(qubits);
      return result;
    }

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
      if (!configuration.IsSet("mps_sample_measure_algorithm") ||
          configuration.GetConfiguration("mps_sample_measure_algorithm") ==
              "mps_probabilities") {
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
      if (qubits.size() > 64)
        throw std::invalid_argument(
            "Use SampleCountsMany for more than 64 measured qubits");
      const auto counts = pp->SampleCounts(qubitsInt, shots);
      for (const auto &[bits, count] : counts) {
        size_t meas = 0;
        for (size_t i = 0; i < bits.size(); ++i)
          if (bits[i]) meas |= (1ULL << i);
        result[meas] += count;
      }
    } else if (simulationType == SimulationType::kPathIntegral) {
      if (nrQubits < 64) {
        if (shots > 1) {
          const auto &amplitudes = pathIntegralSimulator->Amplitudes();
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
      } else {
        throw std::runtime_error(
            "QCSimState::SampleCounts: The path integral simulator does not "
            "support sampling for more than 63 qubits into 64 bits integers.");
      }
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t shot = 0; shot < shots; ++shot) {
        const size_t measured = densityMatrix->MeasureNoCollapse();
        Types::qubit_t packed = 0;
        for (size_t i = 0; i < qubits.size(); ++i)
          if ((measured & (1ULL << qubits[i])) != 0) packed |= 1ULL << i;
        ++result[packed];
      }
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto measured = mpoSimulator->MeasureNoCollapse(qubitsSet);
        Types::qubit_t packed = 0;
        for (size_t i = 0; i < qubits.size(); ++i)
          if (measured.at(static_cast<Eigen::Index>(qubits[i])))
            packed |= 1ULL << i;
        ++result[packed];
      }
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      const std::vector<size_t> selected(qubits.begin(), qubits.end());
      auto sampler = extendedStabilizer->CloneWithSeed(rng());
      for (const auto &item : sampler->SampleCounts(selected, shots))
        result[item.first] += item.second;
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
    if (mpoSimulator)
      for (auto q : qubits)
        if (q >= nrQubits)
          throw std::out_of_range("MPO sampled qubit is out of range");
    if (qubits.empty() || shots == 0) return {};

    if (simulationType == SimulationType::kStabilizer) {
      const auto selected = CliffordSamplingQubits(qubits);
      const auto counts = cliffordSimulator->SampleCountsMany(selected, shots);
      std::unordered_map<std::vector<bool>, Types::qubit_t> result;
      for (const auto &item : counts) result.emplace(item.first, item.second);
      NotifyObservers(qubits);
      return result;
    }

    std::unordered_map<std::vector<bool>, Types::qubit_t> result;

    DontNotify();

    if (simulationType == SimulationType::kMatrixProductState) {
      bool normal = true;
      if (!configuration.IsSet("mps_sample_measure_algorithm") ||
          configuration.GetConfiguration("mps_sample_measure_algorithm") ==
              "mps_probabilities") {
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
      const auto counts = pp->SampleCounts(qubitsInt, shots);
      for (const auto &[bits, count] : counts) result[bits] += count;
    } else if (simulationType == SimulationType::kPathIntegral) {
      if (nrQubits < 64) {
        if (shots > 1) {
          const auto &amplitudes = pathIntegralSimulator->Amplitudes();
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
      } else {
        if (shots > 1) {
          const auto &amplitudes = pathIntegralSimulator->Amplitudes();
          const Utils::AliasBig alias(amplitudes);

          for (size_t shot = 0; shot < shots; ++shot) {
            const double prob = 1. - uniformZeroOne(rng);
            const auto measRaw = alias.Sample(prob);
            std::vector<bool> meas(qubits.size(), false);
            for (size_t i = 0; i < qubits.size(); ++i)
              if (measRaw.get(qubits[i])) meas[i] = true;
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
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t shot = 0; shot < shots; ++shot) {
        const size_t measured = densityMatrix->MeasureNoCollapse();
        std::vector<bool> packed(qubits.size(), false);
        for (size_t i = 0; i < qubits.size(); ++i)
          packed[i] = (measured & (1ULL << qubits[i])) != 0;
        ++result[packed];
      }
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      const std::set<Eigen::Index> qubitsSet(qubits.begin(), qubits.end());
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto measured = mpoSimulator->MeasureNoCollapse(qubitsSet);
        std::vector<bool> packed(qubits.size(), false);
        for (size_t i = 0; i < qubits.size(); ++i)
          packed[i] = measured.at(static_cast<Eigen::Index>(qubits[i]));
        ++result[packed];
      }
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      const std::vector<size_t> selected(qubits.begin(), qubits.end());
      auto sampler = extendedStabilizer->CloneWithSeed(rng());
      for (const auto &item : sampler->SampleCountsMany(selected, shots))
        result[item.first] += item.second;
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

  std::vector<double> ExpectationValues(
      const std::vector<std::string> &paulis) override {
    const bool mps = simulationType == SimulationType::kMatrixProductState;
    if (!mps && simulationType != SimulationType::kMatrixProductOperator)
      return ISimulator::ExpectationValues(paulis);
    std::vector<double> result(paulis.size(), 1.);
    std::vector<std::string> selected;
    std::vector<size_t> indices;
    const size_t n = GetNumberOfQubits();
    for (size_t i = 0; i < paulis.size(); ++i) {
      if (paulis[i].empty()) continue;
      bool zero = false;
      for (size_t q = n; q < paulis[i].size(); ++q) {
        const auto p = toupper(static_cast<unsigned char>(paulis[i][q]));
        if (p != 'I' && p != 'Z') { zero = true; break; }
      }
      if (zero) { result[i] = 0.; continue; }
      std::string pauli = paulis[i];
      pauli.resize(n, 'I');
      for (char &p : pauli) {
        p = static_cast<char>(toupper(static_cast<unsigned char>(p)));
        // The existing MPS single-query interface ignores unknown characters.
        if (mps && p != 'X' && p != 'Y' && p != 'Z') p = 'I';
      }
      indices.push_back(i);
      selected.push_back(std::move(pauli));
    }
    if (selected.empty()) return result;
    const auto values = mps ? TensorExpectationBatch(*mpsSimulator, selected)
                            : TensorExpectationBatch(*mpoSimulator, selected);
    for (size_t i = 0; i < indices.size(); ++i) result[indices[i]] = values[i].real();
    return result;
  }

  /**
   * @brief Returns the expected value of a Pauli string.
   *
   * Use it to obtain the expected value of a Pauli string.
   * The Pauli string is a string of characters representing the Pauli
   * operators, e.g. "XIZY". The length of the string should be less or equal
   * to the number of qubits (if it's less, it's completed with I).
   *
   * @param pauliStringOrig The Pauli string to obtain the expected value for.
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
    else if (simulationType == SimulationType::kPathIntegral)
      return pathIntegralSimulator->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kDensityMatrix) {
      pauliString.resize(GetNumberOfQubits(), 'I');
      return densityMatrix->ExpectationValue(pauliString).real();
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      pauliString.resize(GetNumberOfQubits(), 'I');
      return mpoSimulator->ExpectationValue(pauliString).real();
    } else if (simulationType == SimulationType::kExtendedStabilizer)
      return extendedStabilizer->ExpectationValue(pauliString);

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
   * qcsim applies them right away, so this has no effect on it, but qiskit
   * aer does not.
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
   * Calling this will not destroy the internal state, unlike the
   * 'Destructive' variant. To be used in order to recover the state after
   * doing measurements, for multiple shots executions. In the first phase,
   * only qcsim will implement this.
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
    else if (simulationType == SimulationType::kPathIntegral)
      pathIntegralSimulator->SaveState();
    else if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->SaveState();
    else if (simulationType == SimulationType::kMatrixProductOperator)
      mpoSimulator->SaveState();
    else if (simulationType == SimulationType::kExtendedStabilizer)
      extendedStabilizer->SaveState();
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
    else if (simulationType == SimulationType::kPathIntegral)
      pathIntegralSimulator->RestoreState();
    else if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->RestoreState();
    else if (simulationType == SimulationType::kMatrixProductOperator)
      mpoSimulator->RestoreState();
    else if (simulationType == SimulationType::kExtendedStabilizer)
      extendedStabilizer->RestoreState();
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
    if (extendedStabilizer)
      extendedStabilizer->SetMultithreading(multithreading);
    if (tensorNetwork) tensorNetwork->SetMultithreading(multithreading);
    if (densityMatrix) densityMatrix->SetMultithreading(multithreading);
    if (mpsSimulator) mpsSimulator->SetMultithreading(multithreading);
    if (mpoSimulator) mpoSimulator->SetMultithreading(multithreading);
    if (pp) {
      if (multithreading)
        pp->EnableParallel(pauliWorkerCount);
      else
        pp->DisableParallel();
    }
    if (pathIntegralSimulator) {
      enableMultithreading = false;  // not supported for now
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
   * functionality exposed sometimes so it's good to know if we deal with
   * qcsim or with qiskit aer.
   *
   * @return True if the simulator is a qcsim simulator, false otherwise.
   */
  bool IsQcsim() const override { return true; }

  /**
   * @brief Measures all the qubits without collapsing the state.
   *
   * Measures all the qubits without collapsing the state, allowing to perform
   * multiple shots. This is to be used only internally, only for the
   * statevector simulators (or those based on them, as the composite ones).
   * For the qiskit aer case, SaveStateToInternalDestructive is needed to be
   * called before this. If one wants to use the simulator after such
   * measurement(s), RestoreInternalDestructiveSavedState should be called at
   * the end.
   *
   * Don't use this for more qubits than the size of Types::qubit_t, as the
   * result is packed in a limited number of bits (e.g. 64 bits for uint64_t)
   *
   * @return The result of the measurements, the first qubit result is the
   * least significant bit.
   */
  Types::qubit_t MeasureNoCollapse() override {
    if (mpoSimulator && nrQubits > sizeof(Types::qubit_t) * 8)
      throw std::invalid_argument(
          "Use MeasureNoCollapseMany for more than 64 qubits");
    if (GetNumberOfQubits() > sizeof(Types::qubit_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the Types::qubit_t type, the outcome will be "
             "undefined"
          << std::endl;

    if (simulationType == SimulationType::kStatevector)
      return state->MeasureNoCollapse();
    else if (simulationType == SimulationType::kDensityMatrix)
      return densityMatrix->MeasureNoCollapse();
    else if (simulationType == SimulationType::kMatrixProductOperator) {
      const auto measured = mpoSimulator->MeasureNoCollapse();
      Types::qubit_t result = 0;
      for (size_t qubit = 0; qubit < nrQubits; ++qubit)
        if (measured.at(static_cast<Eigen::Index>(qubit)))
          result |= 1ULL << qubit;
      return result;
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      auto sampler = extendedStabilizer->CloneWithSeed(rng());
      Types::qubit_t result = 0;
      for (size_t qubit = 0; qubit < nrQubits; ++qubit)
        if (sampler->Measure(qubit)) result |= 1ULL << qubit;
      return result;
    } else if (simulationType == SimulationType::kMatrixProductState) {
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
    } else if (simulationType == SimulationType::kPathIntegral) {
      if (nrQubits < 64) {
        const auto measured = pathIntegralSimulator->MeasureNoCollapse();
        Types::qubit_t result = 0;
        Types::qubit_t mask = 1;
        for (Types::qubit_t q = 0; q < measured.size(); ++q) {
          if (measured.get(q)) result |= mask;
          mask <<= 1;
        }
        return result;
      } else {
        throw std::runtime_error(
            "QCSimState::MeasureNoCollapse: The path integral simulator does "
            "not "
            "support measuring more than 63 qubits into 64 bits integers.");
      }
    }

    throw std::runtime_error(
        "QCSimState::MeasureNoCollapse: Invalid simulation type for "
        "measuring "
        "all the qubits without collapsing the state.");

    return 0;
  }

  /**
   * @brief Measures all the qubits without collapsing the state.
   *
   * Measures all the qubits without collapsing the state, allowing to perform
   * multiple shots. This is to be used only internally, only for the
   * statevector simulators (or those based on them, as the composite ones).
   * For the qiskit aer case, SaveStateToInternalDestructive is needed to be
   * called before this. If one wants to use the simulator after such
   * measurement(s), RestoreInternalDestructiveSavedState should be called at
   * the end.
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
    } else if (simulationType == SimulationType::kDensityMatrix) {
      const auto measured = densityMatrix->MeasureNoCollapse();
      std::vector<bool> res(nrQubits);
      for (size_t i = 0; i < nrQubits; ++i) res[i] = ((measured >> i) & 1) == 1;
      return res;
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      const auto measured = mpoSimulator->MeasureNoCollapse();
      std::vector<bool> res(nrQubits);
      for (size_t i = 0; i < nrQubits; ++i)
        res[i] = measured.at(static_cast<Eigen::Index>(i));
      return res;
    } else if (simulationType == SimulationType::kExtendedStabilizer) {
      auto sampler = extendedStabilizer->CloneWithSeed(rng());
      std::vector<bool> res(nrQubits);
      for (size_t i = 0; i < nrQubits; ++i) res[i] = sampler->Measure(i);
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
    } else if (simulationType == SimulationType::kPathIntegral) {
      const auto measured = pathIntegralSimulator->MeasureNoCollapse();
      std::vector<bool> res(nrQubits);
      for (size_t i = 0; i < nrQubits; ++i) res[i] = measured.get(i);
      return res;
    }

    throw std::runtime_error(
        "QCSimState::MeasureNoCollapseMany: Invalid simulation type for "
        "measuring all the qubits without collapsing the state.");

    return {};
  }

  /**
   * @brief Returns the maximum bond dimension reached.
   *
   * Returns the maximum bond dimension reached during execution, if applicable
   * (mps simulator, either qcsim or gpu).
   */
  size_t GetCurrentMaxBondDimension() const override { return curMaxBondDim; }

  const Configuration &GetConfiguration() const { return configuration; }

  const std::unordered_map<std::string, std::string> &GetConfigMap()
      const override {
    return configuration.GetConfigMap();
  }

 protected:
  std::vector<size_t> CliffordSamplingQubits(
      const Types::qubits_vector &qubits) const {
    std::vector<size_t> selected;
    selected.reserve(qubits.size());
    for (const auto q : qubits) {
      if (q >= cliffordSimulator->getNrQubits())
        throw std::out_of_range("Qubit index out of range");
      selected.push_back(static_cast<size_t>(q));
    }
    return selected;
  }

  std::vector<long long> CurrentRoutingMap() const {
    if (mpsSimulator) return ReadRoutingMap(mpsSimulator.get());
    if (mpoSimulator) return ReadRoutingMap(mpoSimulator.get());
    return {};
  }
  void RefreshRoutingCallback() {
    const bool active = IsRoutingLookaheadEnabled() && !upcomingGates.empty();
    if (mpsSimulator)
      mpsSimulator->SetMeetingPositionCallback(active ? meetingPositionCallback
                                                      : nullptr);
    if (mpoSimulator)
      mpoSimulator->SetMeetingPositionCallback(active ? meetingPositionCallback
                                                      : nullptr);
  }

  const char *MaxBondDimensionConfigKey() const {
    return simulationType == SimulationType::kMatrixProductOperator &&
                   configuration.IsSet(
                       "matrix_product_operator_max_bond_dimension")
               ? "matrix_product_operator_max_bond_dimension"
               : "matrix_product_state_max_bond_dimension";
  }

  void ResetDummySimulator() {
    if (!dummySim) return;

    std::vector<long long int> identityMap(nrQubits);
    for (size_t qubit = 0; qubit < nrQubits; ++qubit)
      identityMap[qubit] = static_cast<long long int>(qubit);
    dummySim->SetInitialQubitsMap(identityMap);
    dummySim->setTotalSwappingCost(0.);
    if (nrQubits > 1)
      dummySim->SetCurrentBondDimensions(std::vector<double>(nrQubits - 1, 1.));
  }

  size_t CheckedBasisStateCountForQueries() const {
    if (nrQubits >= std::numeric_limits<size_t>::digits)
      throw std::runtime_error(
          "ImmediateQCSimState: Too many qubits for enumerating basis states.");
    return 1ULL << nrQubits;
  }

  double ExtendedStabilizerBasisProbability(Types::qubit_t outcome) const {
    const size_t nrBasisStates = CheckedBasisStateCountForQueries();
    if (outcome >= nrBasisStates) return 0.0;

    double probability = 0.0;
    std::string pauliString(nrQubits, 'I');
    for (size_t mask = 0; mask < nrBasisStates; ++mask) {
      double sign = 1.0;
      size_t parityBits = mask & static_cast<size_t>(outcome);
      while (parityBits != 0) {
        sign = -sign;
        parityBits &= parityBits - 1;
      }

      for (size_t qubit = 0; qubit < nrQubits; ++qubit)
        pauliString[qubit] = ((mask >> qubit) & 1ULL) == 0 ? 'I' : 'Z';
      probability += sign * extendedStabilizer->ExpectationValue(pauliString);
    }

    probability /= static_cast<double>(nrBasisStates);
    return std::max(0.0, std::min(1.0, probability));
  }

  SimulationType simulationType =
      SimulationType::kStatevector; /**< The simulation type. */

  std::unique_ptr<QC::QubitRegister<>> state; /**< The qcsim state. */
  std::unique_ptr<QC::TensorNetworks::MPSSimulator>
      mpsSimulator; /**< The qcsim mps simulator. */
  std::unique_ptr<QC::TensorNetworks::MPOSimulator>
      mpoSimulator; /**< The qcsim mpo simulator. */
  std::unique_ptr<QC::Clifford::StabilizerSimulator>
      cliffordSimulator; /**< The qcsim clifford simulator. */
  std::unique_ptr<TensorNetworks::TensorNetwork>
      tensorNetwork;                        /**< The qcsim tensor network. */
  std::unique_ptr<QcsimPauliPropagator> pp; /**< The qcsim pauli propagator. */
  std::unique_ptr<PathIntegralSimulator>
      pathIntegralSimulator; /**< The qcsim path integral simulator. */
  std::unique_ptr<QC::DensityMatrix<>>
      densityMatrix; /**< The qcsim density matrix simulator. */
  std::unique_ptr<Simulators::QCSimExtendedStabilizer>
      extendedStabilizer; /**< The qcsim extended stabilizer simulator. */

  size_t nrQubits = 0; /**< The number of allocated qubits. */

  size_t pauliWorkerCount =
      0;  // Pool workers; zero selects hardware concurrency.
  bool enableMultithreading = true; /**< The multithreading flag. */

  int lookaheadDepth = 0;
  int lookaheadDepthWithHeuristic = 0;
  bool useOptimalMeetingPosition = true;
  std::vector<std::shared_ptr<Circuits::IOperation<>>> upcomingGates;
  long long int upcomingGateIndex = 0;
  double growthFactorSwap = 1.;
  double growthFactorGate = 0.65;

  std::unique_ptr<Simulators::MPSDummySimulator> dummySim;

  size_t curMaxBondDim = 0;
  QC::TensorNetworks::MPSSimulator::MeetingPositionCallback
      meetingPositionCallback = nullptr;
  std::function<void(Eigen::Index)> bondDimensionCallback = nullptr;

  // Observer that counts applied gates to track position in upcomingGates
  class GateCounterObserver : public ISimulatorObserver {
   public:
    GateCounterObserver(long long int &indexRef) : index(indexRef) {}
    void Update(const Types::qubits_vector &) override { ++index; }

   private:
    long long int &index;
  };
  std::shared_ptr<GateCounterObserver> gateCounterObserver;

  std::mt19937_64 rng;
  uint64_t nextSeedStream = 0;
  std::uniform_real_distribution<double> uniformZeroOne;

  Configuration configuration; /**< The configuration of the simulator. */
};

}  // namespace Private
}  // namespace Simulators

#endif

#endif  // !_QCSIMSTATE_H_
