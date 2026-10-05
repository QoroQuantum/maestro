/**
 * @file ImmediateGpuState.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The gpu state class.
 *
 * Should not be used directly, create an instance with the factory and use the
 * generic simulator interface.
 */

#pragma once

#ifndef _IMMEDIATE_GPUSTATE_H_
#define _IMMEDIATE_GPUSTATE_H_

#ifdef __linux__

#ifdef INCLUDED_BY_FACTORY

#include "MPSDummySimulator.h"
#include "MPOValidation.h"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include "Configuration.h"

namespace Simulators {
// TODO: Maybe use the pimpl idiom
// https://en.cppreference.com/w/cpp/language/pimpl to hide the implementation
// for good but during development this should be good enough
namespace Private {

/**
 * @class ImmediateGpuState
 * @brief Class for the gpu simulator state.
 *
 * Implements the gpu state.
 * Do not use this class directly, use the factory to create an instance.
 * @sa ISimulator
 * @sa IState
 * @sa ImmediateGpuSimulator
 */
class ImmediateGpuState : public ISimulator {
 public:
  /**
   * @brief Initializes the state.
   *
   * This function is called when the simulator is initialized.
   * Call it after the qubits allocation.
   * @sa ImmediateGpuState::AllocateQubits
   */
  void Initialize() override {
    if (nrQubits) {
      auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
      const int gpuDevice =
          configuration.IsSet("gpu_device")
              ? Configuration::ParseGpuDevice(
                    configuration.GetConfiguration("gpu_device"))
              : SimulatorsFactory::ResolveGpuDevice();
      configuration.SetConfiguration("gpu_device", std::to_string(gpuDevice));
      if (!SimulatorsFactory::GetGpuLibrary(gpuDevice))
        throw std::runtime_error(
            "GpuState::Initialize: Unable to initialize GPU device " +
            std::to_string(gpuDevice));
      if (simulationType == SimulationType::kStatevector) {
        state = SimulatorsFactory::CreateGpuLibStateVectorSim(gpuDevice);
        if (state) {
          // ensure the config settings are applied, they need to be applied
          // after the simulator is created
          for (const auto &[key, value] : configuration.GetConfigMap())
            if (key != "method") Configure(key.c_str(), value.c_str());

          const bool res = state->Create(nrQubits);
          if (!res)
            throw std::runtime_error(
                "GpuState::Initialize: Failed to create "
                "and initialize the statevector state.");
        } else
          throw std::runtime_error(
              "GpuState::Initialize: Failed to create the statevector state.");
      } else if (simulationType == SimulationType::kDensityMatrix) {
        densityMatrix = SimulatorsFactory::CreateGpuDensityMatrix(gpuDevice);
        if (!densityMatrix)
          throw std::runtime_error(
              "GpuState::Initialize: Failed to create the density matrix "
              "state.");
        // Precision must be selected before native density-matrix storage is
        // allocated.
        for (const auto &[key, value] : configuration.GetConfigMap())
          if (key != "method") Configure(key.c_str(), value.c_str());
        if (!densityMatrix->Create(nrQubits))
          throw std::runtime_error(
              "GpuState::Initialize: Failed to initialize the density matrix "
              "state.");
      } else if (simulationType == SimulationType::kMatrixProductOperator) {
        mpo = SimulatorsFactory::CreateGpuMPO(gpuDevice);
        if (!mpo)
          throw std::runtime_error(
              "GpuState::Initialize: Failed to create the matrix product "
              "operator state.");
        mpo->SetCallbackContext((void *)this);
        curMaxBondDim = 1;
        mpo->InstallBondSummary(&ImmediateGpuState::BondSummaryCallback,
                                &ImmediateGpuState::BondDimCallback);

        // Precision and truncation controls must be selected before native
        // storage is allocated.
        for (const auto &[key, value] : configuration.GetConfigMap())
          if (key != "method") Configure(key.c_str(), value.c_str());
        if (!mpo->Create(nrQubits))
          throw std::runtime_error(
              "GpuState::Initialize: Failed to initialize the matrix product "
              "operator state.");
        // default is true
        if (!useOptimalMeetingPosition)
          mpo->SetUseOptimalMeetingPosition(false);
      } else if (simulationType == SimulationType::kMatrixProductState) {
        mps = SimulatorsFactory::CreateGpuLibMPSSim(gpuDevice);
        if (mps) {
          mps->SetCallbackContext((void *)this);
          curMaxBondDim = 1;
          mps->InstallBondSummary(&ImmediateGpuState::BondSummaryCallback,
                                  &ImmediateGpuState::BondDimCallback);

          // ensure the config settings are applied, they need to be applied
          // after the simulator is created but before the state is created
          for (const auto &[key, value] : configuration.GetConfigMap())
            if (key != "method") Configure(key.c_str(), value.c_str());

          const bool res = mps->Create(nrQubits);
          if (!res)
            throw std::runtime_error(
                "GpuState::Initialize: Failed to create "
                "and initialize the MPS state.");
        } else
          throw std::runtime_error(
              "GpuState::Initialize: Failed to create the MPS state.");
        // default is true
        if (!useOptimalMeetingPosition)
          mps->SetUseOptimalMeetingPosition(false);
      } else if (simulationType == SimulationType::kTensorNetwork) {
        tn = SimulatorsFactory::CreateGpuLibTensorNetSim(gpuDevice);
        if (tn) {
          // ensure the config settings are applied, they need to be applied
          // after the simulator is created but before the state is created
          for (const auto &[key, value] : configuration.GetConfigMap())
            if (key != "method") Configure(key.c_str(), value.c_str());

          const bool res = tn->Create(nrQubits);
          if (!res)
            throw std::runtime_error(
                "GpuState::Initialize: Failed to create "
                "and initialize the tensor network state.");
        } else
          throw std::runtime_error(
              "GpuState::Initialize: Failed to create the tensor network "
              "state.");
      } else if (simulationType == SimulationType::kPauliPropagator) {
        pp = SimulatorsFactory::CreateGpuPauliPropagatorSimulatorUnique(
            gpuDevice);
        if (pp) {
          const bool res = pp->CreateSimulator(nrQubits);
          if (!res)
            throw std::runtime_error(
                "GpuState::Initialize: Failed to create "
                "and initialize the Pauli propagator state.");

          // Pauli settings require the native handle created above.
          for (const auto &[key, value] : configuration.GetConfigMap())
            if (key != "method") Configure(key.c_str(), value.c_str());

          pp->SetWillUseSampling(true);  // TODO: check setting
          if (!pp->AllocateMemory(0.9))
            throw std::runtime_error(
                "GpuState::Initialize: Failed to allocate memory for the "
                "Pauli propagator state.");
        } else
          throw std::runtime_error(
              "GpuState::Initialize: Failed to create the Pauli propagator "
              "state.");
      } else
        throw std::runtime_error(
            "GpuState::Initialize: Invalid simulation "
            "type for initializing the state.");
      if (GetGpuDevice() != gpuDevice)
        throw std::runtime_error(
            "GpuState::Initialize: GPU plugin did not confirm the requested "
            "device; update the GPU library");
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
        simulationType != SimulationType::kDensityMatrix &&
        simulationType != SimulationType::kMatrixProductOperator)
      throw std::runtime_error(
          "GpuState::InitializeState: Invalid simulation "
          "type for initializing the state.");

    const bool created =
        simulationType == SimulationType::kDensityMatrix
            ? densityMatrix->CreateWithState(
                  nrQubits, reinterpret_cast<const double *>(amplitudes.data()))
        : simulationType == SimulationType::kMatrixProductOperator
            ? mpo->CreateWithState(
                  nrQubits, reinterpret_cast<const double *>(amplitudes.data()))
            : state->CreateWithState(nrQubits, reinterpret_cast<const double *>(
                                                   amplitudes.data()));
    if (!created)
      throw std::runtime_error(
          "GpuState::InitializeState: Failed to initialize the state.");
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
    if (num_qubits == 0) return;
    Clear();
    nrQubits = num_qubits;
    Initialize();

    if (simulationType != SimulationType::kStatevector &&
        simulationType != SimulationType::kDensityMatrix &&
        simulationType != SimulationType::kMatrixProductOperator)
      throw std::runtime_error(
          "GpuState::InitializeState: Invalid simulation "
          "type for initializing the state.");

    const bool created =
        simulationType == SimulationType::kDensityMatrix
            ? densityMatrix->CreateWithState(
                  nrQubits, reinterpret_cast<const double *>(amplitudes.data()))
        : simulationType == SimulationType::kMatrixProductOperator
            ? mpo->CreateWithState(
                  nrQubits, reinterpret_cast<const double *>(amplitudes.data()))
            : state->CreateWithState(nrQubits, reinterpret_cast<const double *>(
                                                   amplitudes.data()));
    if (!created)
      throw std::runtime_error(
          "GpuState::InitializeState: Failed to initialize the state.");
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
        simulationType != SimulationType::kDensityMatrix &&
        simulationType != SimulationType::kMatrixProductOperator)
      throw std::runtime_error(
          "GpuState::InitializeState: Invalid simulation "
          "type for initializing the state.");

    const bool created =
        simulationType == SimulationType::kDensityMatrix
            ? densityMatrix->CreateWithState(
                  nrQubits, reinterpret_cast<const double *>(amplitudes.data()))
        : simulationType == SimulationType::kMatrixProductOperator
            ? mpo->CreateWithState(
                  nrQubits, reinterpret_cast<const double *>(amplitudes.data()))
            : state->CreateWithState(nrQubits, reinterpret_cast<const double *>(
                                                   amplitudes.data()));
    if (!created)
      throw std::runtime_error(
          "GpuState::InitializeState: Failed to initialize the state.");
  }

  /**
   * @brief Initializes the state to a computational basis state.
   *
   * The density matrix, matrix product operator and matrix product state
   * methods use their own direct primitive; every other method (statevector,
   * tensor network, Pauli propagator) falls back to the generic ISimulator
   * implementation (reset to |0...0>, then apply X on every set bit).
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

    bool created = true;
    if (simulationType == SimulationType::kDensityMatrix)
      created = densityMatrix->CreateWithBasisState(
          nrQubits, static_cast<unsigned long long>(basisState));
    else if (simulationType == SimulationType::kMatrixProductOperator)
      created = mpo->CreateWithBasisState(
          nrQubits, static_cast<unsigned long long>(basisState));
    else if (simulationType == SimulationType::kMatrixProductState)
      created = mps->CreateWithBasisState(
          nrQubits, static_cast<unsigned long long>(basisState));
    else
      for (size_t q = 0; q < num_qubits; ++q)
        if ((basisState >> q) & 1ULL) ApplyX(static_cast<Types::qubit_t>(q));

    if (!created)
      throw std::runtime_error(
          "GpuState::InitializeToBasisState: Failed to initialize the "
          "state.");
  }

  /**
   * @brief Initializes the state to a computational basis state.
   *
   * Same as the Types::qubit_t overload, but not limited to 64 qubits. The
   * matrix product operator and matrix product state methods use their own
   * direct primitive; every other method falls back to the generic
   * ISimulator implementation (reset to |0...0>, then apply X on every set
   * bit) - which is not limited either, unlike the density matrix method's
   * native Types::qubit_t-based primitive.
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

    bool created = true;
    if (simulationType == SimulationType::kMatrixProductOperator ||
        simulationType == SimulationType::kMatrixProductState) {
      std::vector<unsigned char> stateBits(num_qubits, 0);
      for (size_t q = 0; q < num_qubits && q < basisState.size(); ++q)
        stateBits[q] = basisState[q] ? 1 : 0;
      created = simulationType == SimulationType::kMatrixProductOperator
                    ? mpo->CreateWithBasisStateBits(nrQubits, stateBits)
                    : mps->CreateWithBasisStateBits(nrQubits, stateBits);
    } else
      for (size_t q = 0; q < num_qubits && q < basisState.size(); ++q)
        if (basisState[q]) ApplyX(static_cast<Types::qubit_t>(q));

    if (!created)
      throw std::runtime_error(
          "GpuState::InitializeToBasisState: Failed to initialize the "
          "state.");
  }

  /**
   * @brief Initializes the state to a classical mixture of computational
   * basis states.
   *
   * Works for the density matrix and matrix product operator methods. There
   * is no generic fallback for other methods.
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
          "GpuState::InitializeToMixtureOfBasisStates: Invalid simulation "
          "type for initializing to a mixture of basis states.");

    std::vector<std::pair<unsigned long long, double>> converted;
    converted.reserve(mixture.size());
    for (const auto &[basisState, weight] : mixture)
      converted.emplace_back(static_cast<unsigned long long>(basisState),
                             weight);

    const bool created =
        simulationType == SimulationType::kDensityMatrix
            ? densityMatrix->CreateWithMixtureOfBasisStates(nrQubits, converted)
            : mpo->CreateWithMixtureOfBasisStates(nrQubits, converted);

    if (!created)
      throw std::runtime_error(
          "GpuState::InitializeToMixtureOfBasisStates: Failed to initialize "
          "the state.");
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
          "GpuState::InitializeToMixtureOfBasisStates: Invalid simulation "
          "type for initializing to a mixture of basis states.");

    std::vector<double> weights;
    weights.reserve(mixture.size());
    std::vector<unsigned char> stateBitsFlat;
    stateBitsFlat.reserve(mixture.size() * num_qubits);
    for (const auto &[basisState, weight] : mixture) {
      weights.push_back(weight);
      for (size_t q = 0; q < num_qubits; ++q)
        stateBitsFlat.push_back((q < basisState.size() && basisState[q]) ? 1
                                                                         : 0);
    }

    if (!mpo->CreateWithMixtureOfBasisStatesBits(nrQubits, stateBitsFlat,
                                                 weights))
      throw std::runtime_error(
          "GpuState::InitializeToMixtureOfBasisStates: Failed to initialize "
          "the state.");
  }

  /**
   * @brief Just resets the state to 0.
   *
   * Does not destroy the internal state, just resets it to zero (as a 'reset'
   * op on each qubit would do).
   */
  void Reset() override {
    if (state)
      state->Reset();
    else if (densityMatrix)
      densityMatrix->Reset();
    else if (mpo) {
      mpo->Reset();
      curMaxBondDim = 1;
    } else if (mps) {
      mps->Reset();
      curMaxBondDim = 1;
    } else if (tn)
      tn->Reset();
    else if (pp)
      pp->ClearOperators();

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
    if (mps || mpo) {
      if (mps)
        mps->SetInitialQubitsMap(initialMap);
      else
        mpo->SetInitialQubitsMap(initialMap);
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
    if (mps) mps->SetUseOptimalMeetingPosition(enable);
    if (mpo) mpo->SetUseOptimalMeetingPosition(enable);
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
    if (!key || !value) return;
    if (std::string("gpu_device") == key) {
      const int device = Configuration::ParseGpuDevice(value);
      if ((state || densityMatrix || mpo || mps || tn || pp) &&
          device != Configuration::ParseGpuDevice(
                        configuration.GetConfiguration(key)))
        throw std::invalid_argument(
            "gpu_device cannot change after initialization; clear the "
            "simulator first");
      configuration.SetConfiguration(key, std::to_string(device));
      return;
    }
    const auto svdGroup = Configuration::GpuSvdSettingGroup(key);
    if (!svdGroup.empty()) {
      const bool enabled = Configuration::ParseGpuSvdFlag(value);
      const bool gesvd = svdGroup == key;
      const char algorithm = std::string(key).back();
      const auto apply = [algorithm, enabled](auto &backend) {
        if (!backend) return true;  // Applied after native object creation.
        if (algorithm == 'j') return backend->SetGesvdJ(enabled);
        if (algorithm == 'p') return backend->SetGesvdP(enabled);
        return backend->SetGesvdR(enabled);
      };
      const auto applyGesvd = [enabled](auto &backend) {
        // GESVD has no native flag. A false entry is an inactive selector,
        // so replaying it must not clear a subsequently selected algorithm.
        if (!enabled || !backend) return true;
        return backend->SetGesvdJ(false) && backend->SetGesvdP(false) &&
               backend->SetGesvdR(false);
      };
      bool applied = true;
      if (svdGroup == "matrix_product_state_use_gesvd")
        applied = gesvd ? applyGesvd(mps) : apply(mps);
      else if (svdGroup == "matrix_product_operator_use_gesvd")
        applied = gesvd ? applyGesvd(mpo) : apply(mpo);
      else if (svdGroup == "tensor_network_use_gesvd")
        applied = gesvd ? applyGesvd(tn) : apply(tn);
      if (!applied)
        throw std::runtime_error(std::string("GPU library cannot apply ") +
                                 key +
                                 "; an updated GPU plugin may be required");
      configuration.SetConfiguration(key, value);
      return;
    }
    if (std::string("method") == key) {
      if (std::string("statevector") == value)
        simulationType = SimulationType::kStatevector;
      else if (std::string("matrix_product_state") == value)
        simulationType = SimulationType::kMatrixProductState;
      else if (std::string("density_matrix") == value)
        simulationType = SimulationType::kDensityMatrix;
      else if (std::string("matrix_product_operator") == value)
        simulationType = SimulationType::kMatrixProductOperator;
      else if (std::string("tensor_network") == value)
        simulationType = SimulationType::kTensorNetwork;
      else if (std::string("pauli_propagator") == value)
        simulationType = SimulationType::kPauliPropagator;

      // Match the GPU library's default cap.
      if ((simulationType == SimulationType::kMatrixProductState ||
           simulationType == SimulationType::kMatrixProductOperator) &&
          !configuration.IsSet("matrix_product_state_max_bond_dimension"))
        configuration.SetConfiguration(
            "matrix_product_state_max_bond_dimension", "128");
      if (simulationType == SimulationType::kMatrixProductOperator) {
        const auto bondDimension =
            configuration.GetConfiguration(MaxBondDimensionConfigKey());
        configuration.SetConfiguration(
            "matrix_product_state_max_bond_dimension", bondDimension);
        configuration.SetConfiguration(
            "matrix_product_operator_max_bond_dimension", bondDimension);
      }
    }

    if (std::string(key) == "precision" ||
        std::string(key) == "use_double_precision") {
      const bool useDouble = Configuration::ParsePrecision(key, value);
      // The data type is fixed once a backend is allocated. A precision
      // reapplied afterwards (e.g. by a configuration replay) is ignored; a
      // new one takes effect after Clear and the next Initialize.
      if ((state && state->GetNrQubits() > 0) || (mps && mps->IsCreated()) ||
          (tn && tn->IsCreated()) ||
          (densityMatrix && densityMatrix->IsCreated()) ||
          (mpo && mpo->IsCreated()))
        return;
      if ((state && !state->SetDataType(useDouble)) ||
          (mps && !mps->SetDataType(useDouble)) ||
          (tn && !tn->SetDataType(useDouble)))
        throw std::runtime_error("GPU precision configuration failed");
      // Density-matrix and MPO setters throw on native API failure.
      if (densityMatrix) densityMatrix->SetDataType(useDouble);
      if (mpo) mpo->SetDataType(useDouble);
      configuration.SetConfiguration(key, value);
      return;
    }

    if (!configuration.WasApplied(key, value))
      configuration.SetConfiguration(key, value);

    if (std::string("seed") == key) {
      const uint64_t seed = std::stoull(value);
      SeedAuxiliaryRng(seed);
      nextSeedStream = 0;
      if (state) state->SetSeed(seed);
      if (densityMatrix) densityMatrix->SetSeed(seed);
      if (mpo) mpo->SetSeed(seed);
      if (mps) mps->SetSeed(seed);
      if (tn) tn->SetSeed(seed);
      if (pp) pp->SetSeed(seed);
      return;
    }

    if (std::string("matrix_product_state_truncation_threshold") == key ||
        std::string("matrix_product_operator_truncation_threshold") == key) {
      // SetCutoff() sets the numeric threshold value. How that number is
      // interpreted -- relative to the largest singular value at each
      // bond/split, or as a cumulative discarded-weight budget -- is a
      // separate, independently configurable setting; see
      // matrix_product_state_truncation_mode /
      // matrix_product_operator_truncation_mode below. All three GPU backends
      // default to discarded-weight (matching Qiskit Aer's and ITensor's
      // convention) unless relative_max is explicitly requested -- see
      // TruncationMode in maestro-gpu-simulators' lib/truncationmode.hpp and
      // its use in mpsimpl.cu/mpo.cu/tensornet.cu.
      const double singularValueThreshold = std::stod(value);
      // Zero must also replace any previously configured cutoff.
      if (singularValueThreshold >= 0.) {
        if (mps) mps->SetCutoff(singularValueThreshold);
        if (tn) tn->SetCutoff(singularValueThreshold);
        if (mpo) mpo->SetCutoff(singularValueThreshold);
      }
    } else if (std::string("matrix_product_state_truncation_mode") == key ||
               std::string("matrix_product_operator_truncation_mode") == key) {
      // "relative_max" -> TruncationMode::RelativeToMax (0), "discarded_weight"
      // -> TruncationMode::DiscardedWeight (1, the default -- see
      // lib/truncationmode.hpp in maestro-gpu-simulators).
      int truncationMode = -1;
      if (std::string("relative_max") == value)
        truncationMode = 0;
      else if (std::string("discarded_weight") == value)
        truncationMode = 1;
      if (truncationMode >= 0) {
        if (mps) mps->SetTruncationMode(truncationMode);
        if (tn) tn->SetTruncationMode(truncationMode);
        if (mpo) mpo->SetTruncationMode(truncationMode);
      }
    } else if (std::string("matrix_product_state_max_bond_dimension") == key ||
               std::string("matrix_product_operator_max_bond_dimension") ==
                   key) {
      const long long int chi = std::stoi(value);
      if (simulationType == SimulationType::kMatrixProductOperator) {
        // Both names address the same MPO setting; the latest write wins.
        const std::string bondDimension(value);
        for (const char *alias : {"matrix_product_state_max_bond_dimension",
                                  "matrix_product_operator_max_bond_dimension"})
          if (!configuration.WasApplied(alias, bondDimension))
            configuration.SetConfiguration(alias, bondDimension);
      }
      if (mpo) mpo->SetMaxExtent(chi);
      if (chi > 0) {
        if (mps) mps->SetMaxExtent(chi);
        if (tn) tn->SetMaxExtent(chi);

        if (dummySim) dummySim->SetMaxBondDimension(chi);
      }

    } else if (std::string(
                   "matrix_product_operator_kraus_completeness_check") == key) {
      int mode = -1;
      if (std::string(value) == "ignore")
        mode = 0;
      else if (std::string(value) == "warn")
        mode = 1;
      else if (std::string(value) == "strict")
        mode = 2;
      if (mpo && mode >= 0 && !mpo->SetKrausCompletenessCheck(mode))
        throw std::runtime_error("Invalid GPU MPO Kraus completeness mode");
    } else if (std::string(
                   "matrix_product_operator_restore_trace_after_truncation") ==
               key) {
      if (mpo && !mpo->SetRestoreTraceAfterTruncation(
                     std::string(value) == "true" || std::string(value) == "1"))
        throw std::runtime_error("GPU MPO truncation configuration failed");
    } else if (std::string(
                   "matrix_product_operator_hermitize_after_truncation") ==
               key) {
      if (mpo && !mpo->SetHermitizeAfterTruncation(
                     std::string(value) == "true" || std::string(value) == "1"))
        throw std::runtime_error("GPU MPO truncation configuration failed");
    }

    if (pp) {
      if (std::string("pauli_propagator_coefficient_threshold") == key) {
        const double coefficientThreshold = std::stod(value);
        pp->SetCoefficientTruncationCutoff(coefficientThreshold);
      } else if (std::string("pauli_propagator_pauli_weight_threshold") ==
                 key) {
        const double pauliWeightThreshold = std::stod(value);
        pp->SetWeightTruncationCutoff(pauliWeightThreshold);
      } else if (std::string("pauli_propagator_steps_between_trims") == key) {
        const int stepsBetweenTrims = std::stoi(value);
        pp->SetNumGatesBetweenTruncations(stepsBetweenTrims);
      } else if (std::string(
                     "pauli_propagator_num_gates_between_deduplications") ==
                 key) {
        const int numGatesBetweenDeduplications = std::stoi(value);
        pp->SetNumGatesBetweenDeduplications(numGatesBetweenDeduplications);
        if (!configuration.IsSet("pauli_propagator_steps_between_trims"))
          pp->SetNumGatesBetweenTruncations(numGatesBetweenDeduplications);
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
    if (std::string(key) == "precision" ||
        std::string(key) == "use_double_precision") {
      const bool useDouble =
          state           ? state->IsDoublePrecision()
          : mps           ? mps->IsDoublePrecision()
          : tn            ? tn->IsDoublePrecision()
          : densityMatrix ? densityMatrix->IsDoublePrecision()
          : mpo           ? mpo->IsDoublePrecision()
                : configuration.GetConfiguration("precision") == "double";
      return std::string(key) == "precision" ? (useDouble ? "double" : "single")
                                             : (useDouble ? "1" : "0");
    }
    if (mpo) {
      if (std::string(key) ==
          "matrix_product_operator_restore_trace_after_truncation")
        return mpo->GetRestoreTraceAfterTruncation() ? "true" : "false";
      if (std::string(key) ==
          "matrix_product_operator_hermitize_after_truncation")
        return mpo->GetHermitizeAfterTruncation() ? "true" : "false";
    }

    if (!key) return {};
    const auto svdGroup = Configuration::GpuSvdSettingGroup(key);
    if (!svdGroup.empty()) {
      if (svdGroup == key) {
        const auto readGesvd = [](const auto &backend) {
          return !backend->GetGesvdJ() && !backend->GetGesvdP() &&
                 !backend->GetGesvdR();
        };
        if (svdGroup == "matrix_product_state_use_gesvd" && mps)
          return readGesvd(mps) ? "true" : "false";
        if (svdGroup == "matrix_product_operator_use_gesvd" && mpo)
          return readGesvd(mpo) ? "true" : "false";
        if (svdGroup == "tensor_network_use_gesvd" && tn)
          return readGesvd(tn) ? "true" : "false";
      }
      const char algorithm = std::string(key).back();
      const auto read = [algorithm](const auto &backend) {
        if (algorithm == 'j') return backend->GetGesvdJ();
        if (algorithm == 'p') return backend->GetGesvdP();
        return backend->GetGesvdR();
      };
      if (svdGroup == "matrix_product_state_use_gesvd" && mps)
        return read(mps) ? "true" : "false";
      if (svdGroup == "matrix_product_operator_use_gesvd" && mpo)
        return read(mpo) ? "true" : "false";
      if (svdGroup == "tensor_network_use_gesvd" && tn)
        return read(tn) ? "true" : "false";
    }
    if (std::string("method") == key) {
      switch (simulationType) {
        case SimulationType::kStatevector:
          return "statevector";
        case SimulationType::kMatrixProductState:
          return "matrix_product_state";
        case SimulationType::kDensityMatrix:
          return "density_matrix";
        case SimulationType::kMatrixProductOperator:
          return "matrix_product_operator";
        case SimulationType::kTensorNetwork:
          return "tensor_network";
        case SimulationType::kPauliPropagator:
          return "pauli_propagator";
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
        (simulationType == SimulationType::kDensityMatrix && densityMatrix) ||
        (simulationType == SimulationType::kMatrixProductOperator && mpo) ||
        (simulationType == SimulationType::kMatrixProductState && mps) ||
        (simulationType == SimulationType::kPauliPropagator && pp))
      return 0;

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
    state = nullptr;
    densityMatrix = nullptr;
    mpo = nullptr;
    mps = nullptr;
    tn = nullptr;
    pp = nullptr;
    nrQubits = 0;
    dummySim = nullptr;
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
    if (mpo && qubits.size() > sizeof(size_t) * 8)
      throw std::invalid_argument(
          "Use MeasureMany for more than 64 measured qubits");
    // TODO: this is inefficient, maybe implement it better in gpu sim
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
      // TODO: measure all qubits in one shot?
      for (size_t qubit : qubits) {
        if (state->MeasureQubitCollapse(static_cast<int>(qubit))) res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t qubit : qubits) {
        if (densityMatrix->Measure(static_cast<unsigned int>(qubit)))
          res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      for (size_t qubit : qubits) {
        if (mpo->Measure(static_cast<unsigned int>(qubit))) res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kMatrixProductState) {
      // TODO: measure all qubits in one shot?
      for (size_t qubit : qubits) {
        if (mps->Measure(static_cast<unsigned int>(qubit))) res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kTensorNetwork) {
      // TODO: measure all qubits in one shot?
      for (size_t qubit : qubits) {
        if (tn->Measure(static_cast<unsigned int>(qubit))) res |= mask;
        mask <<= 1;
      }
    } else if (simulationType == SimulationType::kPauliPropagator) {
      // TODO: measure all qubits in one shot?
      for (size_t qubit : qubits) {
        if (pp->MeasureQubit(static_cast<int>(qubit))) res |= mask;
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
      for (size_t i = 0; i < qubits.size(); ++i)
        res[i] = state->MeasureQubitCollapse(static_cast<int>(qubits[i]));
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t i = 0; i < qubits.size(); ++i)
        res[i] = densityMatrix->Measure(qubits[i]);
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      for (size_t i = 0; i < qubits.size(); ++i)
        res[i] = mpo->Measure(static_cast<unsigned int>(qubits[i]));
    } else if (simulationType == SimulationType::kMatrixProductState) {
      for (size_t i = 0; i < qubits.size(); ++i)
        res[i] = mps->Measure(static_cast<unsigned int>(qubits[i]));
    } else if (simulationType == SimulationType::kTensorNetwork) {
      for (size_t i = 0; i < qubits.size(); ++i)
        res[i] = tn->Measure(static_cast<unsigned int>(qubits[i]));
    } else if (simulationType == SimulationType::kPauliPropagator) {
      for (size_t i = 0; i < qubits.size(); ++i)
        res[i] = pp->MeasureQubit(static_cast<int>(qubits[i]));
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
    if (simulationType == SimulationType::kStatevector) {
      for (size_t qubit : qubits)
        if (state->MeasureQubitCollapse(static_cast<int>(qubit)))
          state->ApplyX(static_cast<int>(qubit));
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t qubit : qubits) densityMatrix->ApplyReset(qubit);
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      for (size_t qubit : qubits) mpo->ApplyReset(static_cast<int>(qubit));
    } else if (simulationType == SimulationType::kMatrixProductState) {
      for (size_t qubit : qubits)
        if (mps->Measure(static_cast<unsigned int>(qubit)))
          mps->ApplyX(static_cast<unsigned int>(qubit));
    } else if (simulationType == SimulationType::kTensorNetwork) {
      for (size_t qubit : qubits)
        if (tn->Measure(static_cast<unsigned int>(qubit)))
          tn->ApplyX(static_cast<unsigned int>(qubit));
    } else if (simulationType == SimulationType::kPauliPropagator) {
      for (size_t qubit : qubits)
        if (pp->MeasureQubit(static_cast<int>(qubit)))
          pp->ApplyX(static_cast<int>(qubit));
    }

    Notify();
    NotifyObservers(qubits);
  }

  bool SupportsQuantumChannels() const override {
    return simulationType == SimulationType::kDensityMatrix ||
           simulationType == SimulationType::kMatrixProductOperator;
  }

  void ApplyQuantumChannel(const Types::qubits_vector &targets,
                           const QuantumChannel &channel) override {
    if (!densityMatrix && !mpo)
      throw std::runtime_error(
          "GPU quantum channels require an initialized density matrix or "
          "matrix product operator");
    if (targets.size() != channel.GetNumberOfQubits() || targets.empty() ||
        targets.size() > 2)
      throw std::invalid_argument(
          "GPU density matrices and matrix product operators support one- "
          "and two-qubit local channels");
    std::vector<int> gpuTargets;
    gpuTargets.reserve(targets.size());
    for (auto target : targets) {
      if (target >= nrQubits || std::find(gpuTargets.begin(), gpuTargets.end(),
                                          target) != gpuTargets.end())
        throw std::invalid_argument("Invalid GPU quantum-channel target");
      gpuTargets.push_back(static_cast<int>(target));
    }
    const auto &kraus = channel.GetKrausOperators();
    std::vector<double> interleaved;
    interleaved.reserve(kraus.size() * kraus.front().size() * 2);
    for (const auto &op : kraus)
      for (Eigen::Index i = 0; i < op.size(); ++i) {
        interleaved.push_back(op.data()[i].real());
        interleaved.push_back(op.data()[i].imag());
      }
    const bool applied =
        densityMatrix
            ? densityMatrix->ApplyKraus(gpuTargets, kraus.size(),
                                        interleaved.data())
            : mpo->ApplyKraus(gpuTargets, kraus.size(), interleaved.data());
    if (!applied)
      throw std::runtime_error(
          "GPU density-matrix/matrix-product-operator channel application "
          "failed");
    NotifyObservers(targets);
  }

  double ProbabilityBits(const std::vector<bool> &bits) override {
    if (!mpo) return IState::ProbabilityBits(bits);
    return mpo->ProbabilityBits(
        std::vector<unsigned char>(bits.begin(), bits.end()));
  }
  std::complex<double> DensityMatrixElementBits(
      const std::vector<bool> &row,
      const std::vector<bool> &col) const override {
    if (!mpo) return IState::DensityMatrixElementBits(row, col);
    return mpo->GetElementBits(
        std::vector<unsigned char>(row.begin(), row.end()),
        std::vector<unsigned char>(col.begin(), col.end()));
  }
  Eigen::MatrixXcd GetDensityMatrix(bool normalized = true) const override {
    if (!mpo) return IState::GetDensityMatrix(normalized);
    const auto values = mpo->GetDensityMatrix(normalized);
    const Eigen::Index dim = Eigen::Index{1} << nrQubits;
    return Eigen::Map<const Eigen::MatrixXcd>(values.data(), dim, dim);
  }
  std::complex<double> ExpectationValueComplex(
      const std::string &pauli, bool normalized = true) const override {
    if (!mpo) return IState::ExpectationValueComplex(pauli, normalized);
    MPOValidation::Pauli(pauli, nrQubits);
    return mpo->ExpectationValueComplex(pauli, normalized);
  }
  std::vector<std::complex<double>> ExpectationValuesComplex(
      const std::vector<std::string> &paulis,
      bool normalized = true) const override {
    if (!mpo) return IState::ExpectationValuesComplex(paulis, normalized);
    for (const auto &pauli : paulis) MPOValidation::Pauli(pauli, nrQubits);
    return mpo->ExpectationValuesComplex(paulis, normalized);
  }
  std::complex<double> ExpectationValueOperators(
      const Types::qubits_vector &qubits,
      const std::vector<Eigen::MatrixXcd> &matrices) override {
    if (!mps) return IState::ExpectationValueOperators(qubits, matrices);
    TensorQueries::Operators(qubits, matrices, nrQubits);
    std::vector<int> targets;
    std::vector<double> raw;
    targets.reserve(qubits.size());
    raw.reserve(8 * matrices.size());
    for (size_t i = 0; i < qubits.size(); ++i) {
      targets.push_back(TensorQueries::Count(qubits[i]));
      for (Eigen::Index e = 0; e < 4; ++e) {
        raw.push_back(matrices[i].data()[e].real());
        raw.push_back(matrices[i].data()[e].imag());
      }
    }
    return mps->ExpectationValueOperators(targets, raw);
  }
  void ApplyOperator(const Types::qubits_vector &qubits,
                     const Eigen::MatrixXcd &matrix,
                     bool normalize = false) override {
    if (!mpo) return IState::ApplyOperator(qubits, matrix, normalize);
    const auto selected = MPOValidation::Operator(qubits, matrix, nrQubits);
    std::vector<double> raw(2 * matrix.size());
    for (Eigen::Index i = 0; i < matrix.size(); ++i) {
      raw[2 * i] = matrix.data()[i].real();
      raw[2 * i + 1] = matrix.data()[i].imag();
    }
    if (!mpo->ApplyOperator(selected, raw.data(), normalize))
      throw std::runtime_error("GPU MPO operator application failed");
    NotifyObservers(qubits);
  }
  void MoveAtBeginningOfChain(const Types::qubits_vector &qubits) override {
    if (!mpo && !mps) return IState::MoveAtBeginningOfChain(qubits);
    const auto selected = MPOValidation::Qubits(qubits, nrQubits);
    if (selected.empty()) return;
    if (mps)
      mps->MoveAtBeginningOfChain(selected);
    else
      mpo->MoveAtBeginningOfChain(selected);
  }

  std::vector<std::complex<double>> GetStateVector() override {
    if (!mps) return IState::GetStateVector();
    return mps->GetStateVector();
  }
  std::complex<double> DensityMatrixTrace() const override {
    if (densityMatrix) return densityMatrix->Trace();
    if (mpo) return mpo->TraceComplex();
    throw std::runtime_error(
        "GPU mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  double DensityMatrixPurity() const override {
    if (densityMatrix) return densityMatrix->Purity();
    if (mpo) return mpo->Purity();
    throw std::runtime_error(
        "GPU mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  std::complex<double> DensityMatrixTraceOfSquare() const override {
    if (mpo) return mpo->TraceOfSquare();
    if (densityMatrix) {
      const double tr = densityMatrix->Trace();
      return densityMatrix->Purity() * tr * tr;
    }
    throw std::runtime_error(
        "GPU mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  std::complex<double> DensityMatrixOverlap(
      const IState &other) const override {
    const auto *rhs = dynamic_cast<const ImmediateGpuState *>(&other);
    if (!rhs)
      throw std::invalid_argument(
          "Density-matrix overlap requires matching GPU backends");
    if (densityMatrix && rhs->densityMatrix)
      return densityMatrix->HilbertSchmidtOverlap(*rhs->densityMatrix);
    if (mpo && rhs->mpo) return mpo->HilbertSchmidtOverlap(*rhs->mpo);
    throw std::invalid_argument(
        "Density-matrix overlap requires two density matrices or two MPOs");
  }
  double DensityMatrixHermiticityResidual() const override {
    if (mpo) return mpo->HermiticityResidual();
    if (densityMatrix)
      return densityMatrix->IsHermitian()
                 ? 0.
                 : std::numeric_limits<double>::infinity();
    throw std::runtime_error(
        "GPU mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  bool IsDensityMatrixHermitian(double eps = 1e-10) const override {
    if (densityMatrix) return densityMatrix->IsHermitian(eps);
    if (mpo) return mpo->IsHermitian(eps);
    throw std::runtime_error(
        "GPU mixed-state diagnostics require density_matrix or "
        "matrix_product_operator");
  }
  Eigen::MatrixXcd PartialTrace(
      const Types::qubits_vector &qubits) const override {
    std::vector<int> keep(qubits.begin(), qubits.end());
    const auto values =
        densityMatrix ? densityMatrix->PartialTrace(keep)
        : mpo         ? mpo->PartialTrace(keep)
                      : throw std::runtime_error(
                    "GPU partial trace requires a mixed-state backend");
    const Eigen::Index dim =
        static_cast<Eigen::Index>(size_t{1} << keep.size());
    return Eigen::Map<const Eigen::MatrixXcd>(values.data(), dim, dim);
  }
  double FidelityWithStatevector(const Eigen::VectorXcd &psi) const override {
    std::vector<double> raw(2 * static_cast<size_t>(psi.size()));
    for (Eigen::Index i = 0; i < psi.size(); ++i) {
      raw[2 * i] = psi[i].real();
      raw[2 * i + 1] = psi[i].imag();
    }
    if (densityMatrix)
      return densityMatrix->FidelityWithStatevector(raw.data());
    if (mpo) return mpo->FidelityWithStatevector(raw.data());
    throw std::runtime_error(
        "GPU mixed-state fidelity requires density_matrix or "
        "matrix_product_operator");
  }
  void RestoreDensityMatrixTrace() override {
    if (!mpo)
      throw std::runtime_error(
          "Trace restoration is only available for GPU MPO");
    mpo->RestoreTrace();
  }
  void HermitizeDensityMatrix() override {
    if (!mpo)
      throw std::runtime_error("Hermitization is only available for GPU MPO");
    mpo->Hermitize();
  }
  void Trim() override {
    if (mps) {
      if (!mps->Trim()) throw std::runtime_error("GPU MPS trim failed");
    } else if (mpo)
      mpo->Trim();
    else
      throw std::runtime_error("Trim is only available for GPU MPS and MPO");
  }
  void ReCanonicalize() override {
    if (mps) {
      if (!mps->ReCanonicalize())
        throw std::runtime_error("GPU MPS canonicalization failed");
    } else if (mpo)
      mpo->ReCanonicalize();
    else
      throw std::runtime_error(
          "Canonicalization is only available for GPU MPS and MPO");
  }

  /**
   * @brief Returns the probability of the specified outcome.
   *
   * Use it to obtain the probability to obtain the specified outcome, if all
   * qubits are measured.
   * @sa ImmediateGpuState::Amplitude
   * @sa ImmediateGpuState::Probabilities
   *
   * @param outcome The outcome to obtain the probability for.
   * @return The probability of the specified outcome.
   */
  double Probability(Types::qubit_t outcome) override {
    if (simulationType == SimulationType::kStatevector)
      return state->BasisStateProbability(outcome);
    else if (simulationType == SimulationType::kDensityMatrix)
      return densityMatrix->Probability(outcome);
    else if (simulationType == SimulationType::kMatrixProductOperator) {
      if (nrQubits < sizeof(Types::qubit_t) * 8 &&
          outcome >= (Types::qubit_t{1} << nrQubits))
        throw std::out_of_range("GPU MPO basis state is out of range");
      std::vector<bool> bits(nrQubits);
      for (size_t q = 0; q < std::min(nrQubits, sizeof(Types::qubit_t) * 8);
           ++q)
        bits[q] = ((outcome >> q) & 1) != 0;
      return ProbabilityBits(bits);
    } else if (simulationType == SimulationType::kMatrixProductState ||
               simulationType == SimulationType::kTensorNetwork) {
      const auto ampl = Amplitude(outcome);
      return std::norm(ampl);
    } else if (simulationType == SimulationType::kPauliPropagator) {
      return pp->Probability(outcome);
    }

    return 0.0;
  }

  /**
   * @brief Returns the amplitude of the specified state.
   *
   * Use it to obtain the amplitude of the specified state.
   * @sa ImmediateGpuState::Probability
   * @sa ImmediateGpuState::Probabilities
   *
   * @param outcome The outcome to obtain the amplitude for.
   * @return The amplitude of the specified outcome.
   */
  std::complex<double> Amplitude(Types::qubit_t outcome) override {
    double real = 0.0;
    double imag = 0.0;

    if (simulationType == SimulationType::kStatevector)
      state->Amplitude(outcome, &real, &imag);
    else if (simulationType == SimulationType::kDensityMatrix)
      throw std::runtime_error(
          "GpuState::Amplitude: Amplitudes are not defined for density "
          "matrices.");
    else if (simulationType == SimulationType::kMatrixProductOperator)
      throw std::runtime_error(
          "GpuState::Amplitude: Amplitudes are not defined for matrix "
          "product operators.");
    else if (simulationType == SimulationType::kMatrixProductState ||
             simulationType == SimulationType::kTensorNetwork) {
      std::vector<long int> fixedValues(nrQubits);
      for (size_t i = 0; i < nrQubits; ++i)
        fixedValues[i] = i < std::numeric_limits<Types::qubit_t>::digits
                             ? (outcome >> i) & 1
                             : 0;
      if (simulationType == SimulationType::kMatrixProductState) {
        if (!mps->Amplitude(nrQubits, fixedValues.data(), &real, &imag) ||
            !std::isfinite(real) || !std::isfinite(imag))
          throw std::runtime_error("GPU MPS amplitude query failed");
      } else if (simulationType == SimulationType::kTensorNetwork)
        tn->Amplitude(nrQubits, fixedValues.data(), &real, &imag);
    } else if (simulationType == SimulationType::kPauliPropagator) {
      // Pauli propagator does not support amplitude calculation
      throw std::runtime_error(
          "GpuState::Amplitude: Invalid simulation type for amplitude "
          "calculation.");
    }

    return std::complex<double>(real, imag);
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
      return mps->ProjectOnZero();

    return Amplitude(0);
  }

  /**
   * @brief Returns the probabilities of all possible outcomes.
   *
   * Use it to obtain the probabilities of all possible outcomes.
   * @sa Gputate::Probability
   * @sa ImmediateGpuState::Amplitude
   * @sa ImmediateGpuState::AllProbabilities
   *
   * @return A vector with the probabilities of all possible outcomes.
   */
  std::vector<double> AllProbabilities() override {
    if (nrQubits == 0) return {};
    if (mps) return mps->AllProbabilities();
    if (nrQubits >= std::numeric_limits<size_t>::digits ||
        (mpo && nrQubits >= 63))
      throw std::length_error(
          "Full probability enumeration exceeds the basis-index API; query "
          "selected bit vectors");
    const size_t numStates = size_t{1} << nrQubits;
    std::vector<double> result(numStates);

    if (simulationType == SimulationType::kStatevector)
      state->AllProbabilities(result.data());
    else if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->AllProbabilities(result.data());
    else if (simulationType == SimulationType::kMatrixProductOperator)
      mpo->AllProbabilities(result.data());
    else if (simulationType == SimulationType::kMatrixProductState ||
             simulationType == SimulationType::kTensorNetwork) {
      // this is very slow, it should be used only for tests!
      for (Types::qubit_t i = 0; i < (Types::qubit_t)numStates; ++i) {
        const auto val = Amplitude(i);
        result[i] = std::norm(std::complex<double>(val.real(), val.imag()));
      }
    } else if (simulationType == SimulationType::kPauliPropagator) {
      // this is very slow, it should be used only for tests!
      for (Types::qubit_t i = 0; i < (Types::qubit_t)numStates; ++i) {
        result[i] = pp->Probability(i);
      }
    }

    return result;
  }

  /**
   * @brief Returns the probabilities of the specified outcomes.
   *
   * Use it to obtain the probabilities of the specified outcomes.
   * @sa ImmediateGpuState::Probability
   * @sa ImmediateGpuState::Amplitude
   *
   * @param qubits A vector with the qubits configuration outcomes.
   * @return A vector with the probabilities for the specified qubit
   * configurations.
   */
  std::vector<double> Probabilities(
      const Types::qubits_vector &qubits) override {
    std::vector<double> result(qubits.size());

    if (simulationType == SimulationType::kStatevector) {
      for (size_t i = 0; i < qubits.size(); ++i)
        result[i] = state->BasisStateProbability(qubits[i]);
    } else if (simulationType == SimulationType::kDensityMatrix) {
      for (size_t i = 0; i < qubits.size(); ++i)
        result[i] = densityMatrix->Probability(qubits[i]);
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      for (size_t i = 0; i < qubits.size(); ++i)
        result[i] = Probability(qubits[i]);
    } else if (simulationType == SimulationType::kMatrixProductState ||
               simulationType == SimulationType::kTensorNetwork) {
      for (size_t i = 0; i < qubits.size(); ++i) {
        const auto ampl = Amplitude(qubits[i]);
        result[i] = std::norm(ampl);
      }
    } else if (simulationType == SimulationType::kPauliPropagator) {
      for (size_t i = 0; i < qubits.size(); ++i)
        result[i] = pp->Probability(qubits[i]);
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
    if (mpo || mps) {
      if (qubits.size() > sizeof(Types::qubit_t) * 8)
        throw std::invalid_argument(
            "Use SampleCountsMany for more than 64 measured qubits");
      std::unordered_map<Types::qubit_t, Types::qubit_t> result;
      for (const auto &entry : SampleCountsMany(qubits, shots)) {
        Types::qubit_t value = 0;
        for (size_t i = 0; i < entry.first.size(); ++i)
          if (entry.first[i]) value |= Types::qubit_t{1} << i;
        result[value] += entry.second;
      }
      return result;
    }
    if (qubits.empty() || shots == 0) return {};

    if (qubits.size() > sizeof(Types::qubit_t) * 8)
      std::cerr
          << "Warning: The number of qubits to measure is larger than the "
             "number of bits in the Types::qubit_t type, the outcome will be "
             "undefined"
          << std::endl;

    std::unordered_map<Types::qubit_t, Types::qubit_t> result;

    DontNotify();

    if (simulationType == SimulationType::kStatevector) {
      std::vector<long int> samples(shots);
      state->SampleAll(shots, samples.data());

      for (auto outcome : samples) {
        // qubits might not be in order, translate the outcome to the correct
        // order
        Types::qubit_t translatedOutcome = 0;
        Types::qubit_t mask = 1ULL;
        for (size_t i = 0; i < qubits.size(); ++i) {
          if (outcome & (1ULL << qubits[i])) translatedOutcome |= mask;
          mask <<= 1;
        }
        ++result[translatedOutcome];
      }
    } else if (simulationType == SimulationType::kDensityMatrix) {
      std::vector<long int> samples(shots);
      if (!densityMatrix->SampleAll(shots, samples.data())) {
        Notify();
        throw std::runtime_error(
            "GpuState::SampleCounts: Density-matrix sampling failed.");
      }
      for (auto outcome : samples) {
        Types::qubit_t translatedOutcome = 0;
        for (size_t i = 0; i < qubits.size(); ++i)
          if (outcome & (1ULL << qubits[i])) translatedOutcome |= 1ULL << i;
        ++result[translatedOutcome];
      }
    } else if (simulationType == SimulationType::kTensorNetwork) {
      std::unordered_map<std::vector<bool>, int64_t> *map =
          tn->GetMapForSample();
      std::vector<unsigned int> qubitsIndices(qubits.begin(), qubits.end());
      tn->Sample(shots, qubitsIndices.size(), qubitsIndices.data(), map);
      const auto positions = SampleBitPositions(qubits);
      // put the results in the result map
      for (const auto &[meas, cnt] : *map) {
        Types::qubit_t outcome = 0;
        Types::qubit_t mask = 1ULL;
        for (Types::qubit_t q = 0; q < qubits.size(); ++q) {
          if (meas[positions[q]]) outcome |= mask;
          mask <<= 1;
        }
        result[outcome] += cnt;
      }
      tn->FreeMapForSample(map);
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qb(qubits.begin(), qubits.end());
      for (size_t shot = 0; shot < shots; ++shot) {
        size_t meas = 0;
        auto res = pp->SampleQubits(qb);
        for (size_t i = 0; i < qubits.size(); ++i) {
          if (res[i]) meas |= (1ULL << i);
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
    if (mpo || mps) {
      std::vector<unsigned int> selected;
      for (auto q : qubits) {
        if (q >= nrQubits ||
            q > static_cast<size_t>(std::numeric_limits<int>::max()))
          throw std::out_of_range("GPU tensor sampled qubit is out of range");
        selected.push_back(static_cast<unsigned int>(q));
      }
      if (qubits.empty() || shots == 0) return {};
      std::sort(selected.begin(), selected.end());
      selected.erase(std::unique(selected.begin(), selected.end()),
                     selected.end());
      const auto counts =
          mps ? mps->SampleHistogram(shots, std::move(selected))
              : mpo->SampleHistogram(shots, std::move(selected));
      const auto positions = SampleBitPositions(qubits);
      std::unordered_map<std::vector<bool>, Types::qubit_t> result;
      for (const auto &entry : counts) {
        std::vector<bool> bits(qubits.size());
        for (size_t i = 0; i < bits.size(); ++i)
          bits[i] = entry.first[positions[i]];
        result[std::move(bits)] += entry.second;
      }
      NotifyObservers(qubits);
      return result;
    }
    if (qubits.empty() || shots == 0) return {};

    std::unordered_map<std::vector<bool>, Types::qubit_t> result;

    DontNotify();

    if (simulationType == SimulationType::kStatevector) {
      std::vector<long int> samples(shots);
      state->SampleAll(shots, samples.data());

      std::vector<bool> outcomeVec(qubits.size());
      for (auto outcome : samples) {
        for (size_t i = 0; i < qubits.size(); ++i)
          outcomeVec[i] = ((outcome >> qubits[i]) & 1) == 1;
        ++result[outcomeVec];
      }
    } else if (simulationType == SimulationType::kDensityMatrix) {
      std::vector<long int> samples(shots);
      if (!densityMatrix->SampleAll(shots, samples.data())) {
        Notify();
        throw std::runtime_error(
            "GpuState::SampleCountsMany: Density-matrix sampling failed.");
      }
      std::vector<bool> outcomeVec(qubits.size());
      for (auto outcome : samples) {
        for (size_t i = 0; i < qubits.size(); ++i)
          outcomeVec[i] = ((outcome >> qubits[i]) & 1) != 0;
        ++result[outcomeVec];
      }
    } else if (simulationType == SimulationType::kTensorNetwork) {
      std::unordered_map<std::vector<bool>, int64_t> *map =
          tn->GetMapForSample();
      std::vector<unsigned int> qubitsIndices(qubits.begin(), qubits.end());
      tn->Sample(shots, qubitsIndices.size(), qubitsIndices.data(), map);
      const auto positions = SampleBitPositions(qubits);
      // put the results in the result map
      for (const auto &[meas, cnt] : *map) {
        std::vector<bool> ordered(qubits.size());
        for (size_t q = 0; q < qubits.size(); ++q)
          ordered[q] = meas[positions[q]];
        result[ordered] += cnt;
      }
      tn->FreeMapForSample(map);
    } else if (simulationType == SimulationType::kPauliPropagator) {
      std::vector<int> qb(qubits.begin(), qubits.end());
      for (size_t shot = 0; shot < shots; ++shot) {
        const auto res = pp->SampleQubits(qb);
        ++result[res];
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
    double result = 0.0;

    if (simulationType == SimulationType::kStatevector)
      result = state->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kDensityMatrix)
      result = densityMatrix->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kMatrixProductOperator)
      result = mpo->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kMatrixProductState)
      result = mps->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kTensorNetwork)
      result = tn->ExpectationValue(pauliString);
    else if (simulationType == SimulationType::kPauliPropagator)
      result = pp->ExpectationValue(pauliString);
    else
      throw std::runtime_error(
          "GpuState::ExpectationValue: Invalid simulation type for expectation "
          "value calculation.");

    return result;
  }

  std::vector<double> ExpectationValues(
      const std::vector<std::string> &paulis) override {
    if (!mps && !mpo) return IState::ExpectationValues(paulis);
    // Preserve GPU scalar semantics: empty strings are identity, invalid or
    // over-width strings return zero. Only valid strings enter the native
    // batch.
    std::vector<double> result(paulis.size(), 0.);
    std::vector<std::string> selected;
    std::vector<size_t> indices;
    for (size_t i = 0; i < paulis.size(); ++i) {
      if (paulis[i].empty()) {
        result[i] = 1.;
        continue;
      }
      if (paulis[i].size() > nrQubits) continue;
      auto pauli = paulis[i];
      for (char &p : pauli)
        p = static_cast<char>(std::toupper(static_cast<unsigned char>(p)));
      if (pauli.find_first_not_of("IXYZ") != std::string::npos) continue;
      indices.push_back(i);
      selected.push_back(std::move(pauli));
    }
    if (selected.empty()) return result;
    const auto values = mps ? mps->ExpectationValues(selected)
                            : mpo->ExpectationValues(selected);
    for (size_t i = 0; i < indices.size(); ++i) result[indices[i]] = values[i];
    return result;
  }

  /**
   * @brief Returns the type of simulator.
   *
   * Returns the type of simulator.
   * @return The type of simulator.
   * @sa SimulatorType
   */
  int GetGpuDevice() const override {
    if (state) return state->GetGpuDevice();
    if (densityMatrix) return densityMatrix->GetGpuDevice();
    if (mpo) return mpo->GetGpuDevice();
    if (mps) return mps->GetGpuDevice();
    if (tn) return tn->GetGpuDevice();
    if (pp) return pp->GetGpuDevice();
    return -1;
  }

  SimulatorType GetType() const override { return SimulatorType::kGpuSim; }

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
   * the gpu simulator applies them right away, so this has no effect on it, but
   * qiskit aer does not.
   */
  // Gate calls only queue work on the device; flushing waits for it, so a
  // flushed state is complete (e.g. for timing or cross-stream use).
  void Flush() override {
    const bool done = state              ? state->Synchronize()
                      : densityMatrix    ? densityMatrix->Synchronize()
                                         : true;
    if (!done)
      throw std::runtime_error("GpuState::Flush: device synchronization failed");
  }

  /**
   * @brief Saves the state to internal storage.
   *
   * Saves the state to internal storage, if needed.
   * Calling this should consider as the simulator is gone to uninitialized.
   * Either do not use it except for getting amplitudes, or reinitialize the
   * simulator after calling it. This is needed only for the composite
   * simulator, for an optimization for qiskit aer. For the others it does
   * nothing.
   */
  void SaveStateToInternalDestructive() override {
    if (simulationType == SimulationType::kStatevector)
      state->SaveStateDestructive();
    else if (simulationType == SimulationType::kPauliPropagator)
      return;
    else
      throw std::runtime_error(
          "GpuState::SaveStateToInternalDestructive: Invalid simulation type "
          "for saving the state destructively.");
  }

  /**
   * @brief Restores the state from the internally saved state
   *
   * Restores the state from the internally saved state, if needed.
   * This does something only for qiskit aer.
   */
  void RestoreInternalDestructiveSavedState() override {
    if (simulationType == SimulationType::kStatevector)
      state->RestoreStateFreeSaved();
    else if (simulationType == SimulationType::kPauliPropagator)
      return;
    else
      throw std::runtime_error(
          "GpuState::RestoreInternalDestructiveSavedState: Invalid simulation "
          "type for restoring the state destructively.");
  }

  /**
   * @brief Saves the state to internal storage.
   *
   * Saves the state to internal storage, if needed.
   * Calling this will not destroy the internal state, unlike the 'Destructive'
   * variant. To be used in order to recover the state after doing measurements,
   * for multiple shots executions.
   */
  void SaveState() override {
    if (simulationType == SimulationType::kStatevector)
      state->SaveState();
    else if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->SaveState();
    else if (simulationType == SimulationType::kMatrixProductOperator)
      mpo->SaveState();
    else if (simulationType == SimulationType::kMatrixProductState) {
      if (!mps->SaveState())
        throw std::runtime_error("GPU MPS state save failed");
    } else if (simulationType == SimulationType::kTensorNetwork)
      tn->SaveState();
    else if (simulationType == SimulationType::kPauliPropagator)
      pp->SaveState();
  }

  /**
   * @brief Restores the state from the internally saved state
   *
   * Restores the state from the internally saved state, if needed.
   * To be used in order to recover the state after doing measurements, for
   * multiple shots executions.
   */
  void RestoreState() override {
    if (simulationType == SimulationType::kStatevector)
      state->RestoreStateNoFreeSaved();
    else if (simulationType == SimulationType::kDensityMatrix)
      densityMatrix->RestoreState();
    else if (simulationType == SimulationType::kMatrixProductOperator)
      mpo->RestoreState();
    else if (simulationType == SimulationType::kMatrixProductState) {
      if (!mps->RestoreState())
        throw std::runtime_error("GPU MPS state restore failed");
    } else if (simulationType == SimulationType::kTensorNetwork)
      tn->RestoreState();
    else if (simulationType == SimulationType::kPauliPropagator)
      pp->RestoreState();
  }

  /**
   * @brief Gets the amplitude.
   *
   * Gets the amplitude, from the internal storage if needed.
   * This is needed only for the composite simulator, for an optimization for
   * qiskit aer. For qcsim and gpu sim it does the same thing as Amplitude.
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
    // don't do anything here, the multithreading is always enabled
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
   * This is just a helper function to ease things up: qcsim has different
   * functionality exposed sometimes so it's good to know if we deal with qcsim
   * or with qiskit aer.
   *
   * @return True if the simulator is a qcsim simulator, false otherwise.
   */
  bool IsQcsim() const override { return false; }

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
    if (mpo && nrQubits > sizeof(Types::qubit_t) * 8)
      throw std::invalid_argument(
          "Use MeasureNoCollapseMany for more than 64 qubits");
    if (simulationType == SimulationType::kStatevector)
      return state->MeasureAllQubitsNoCollapse();
    else if (simulationType == SimulationType::kDensityMatrix) {
      std::vector<long int> samples(1);
      if (!densityMatrix->SampleAll(1, samples.data()))
        throw std::runtime_error(
            "GpuState::MeasureNoCollapse: Density-matrix sampling failed.");
      return static_cast<Types::qubit_t>(samples.front());
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      std::vector<long int> samples(1);
      if (!mpo->SampleAll(1, samples.data()))
        throw std::runtime_error(
            "GpuState::MeasureNoCollapse: Matrix-product-operator sampling "
            "failed.");
      return static_cast<Types::qubit_t>(samples.front());
    } else if (simulationType == SimulationType::kMatrixProductState ||
               simulationType == SimulationType::kTensorNetwork ||
               simulationType == SimulationType::kPauliPropagator) {
      if (nrQubits > sizeof(Types::qubit_t) * 8)
        std::cerr
            << "Warning: The number of qubits to measure is larger than the "
               "number of bits in the Types::qubit_t type, the outcome will be "
               "undefined"
            << std::endl;

      Types::qubits_vector fixedValues(nrQubits);
      std::iota(fixedValues.begin(), fixedValues.end(), 0);
      const auto res = SampleCounts(fixedValues, 1);
      if (res.empty()) return 0;
      return res.begin()
          ->first;  // return the first outcome, as it is the only one
    }

    throw std::runtime_error(
        "GpuState::MeasureNoCollapse: Invalid simulation type for measuring "
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
      const auto meas = state->MeasureAllQubitsNoCollapse();
      std::vector<bool> result(nrQubits, false);
      for (size_t i = 0; i < nrQubits; ++i) result[i] = ((meas >> i) & 1) == 1;
      return result;
    } else if (simulationType == SimulationType::kDensityMatrix) {
      std::vector<long int> samples(1);
      if (!densityMatrix->SampleAll(1, samples.data()))
        throw std::runtime_error(
            "GpuState::MeasureNoCollapseMany: Density-matrix sampling failed.");
      std::vector<bool> result(nrQubits, false);
      for (size_t i = 0; i < nrQubits; ++i)
        result[i] = ((samples.front() >> i) & 1) != 0;
      return result;
    } else if (simulationType == SimulationType::kMatrixProductOperator) {
      std::vector<int> qubits(nrQubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      const auto bits = mpo->SampleBits(1, qubits);
      return std::vector<bool>(bits.begin(), bits.end());
    } else if (simulationType == SimulationType::kMatrixProductState ||
               simulationType == SimulationType::kTensorNetwork ||
               simulationType == SimulationType::kPauliPropagator) {
      Types::qubits_vector fixedValues(nrQubits);
      std::iota(fixedValues.begin(), fixedValues.end(), 0);
      const auto res = SampleCountsMany(fixedValues, 1);
      if (res.empty()) return std::vector<bool>(nrQubits, false);
      return res.begin()
          ->first;  // return the first outcome, as it is the only one
    }

    throw std::runtime_error(
        "GpuState::MeasureNoCollapseMany: Invalid simulation type for "
        "measuring "
        "all the qubits without collapsing the state.");

    return std::vector<bool>(nrQubits, false);
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
  // MPSSample/TNSample consume a set and return bits in ascending logical
  // qubit order. Maestro's sampling interface preserves the caller's order.
  static std::vector<size_t> SampleBitPositions(
      const Types::qubits_vector &qubits) {
    auto sorted = qubits;
    std::sort(sorted.begin(), sorted.end());
    sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
    std::vector<size_t> positions;
    positions.reserve(qubits.size());
    for (const auto q : qubits)
      positions.push_back(std::lower_bound(sorted.begin(), sorted.end(), q) -
                          sorted.begin());
    return positions;
  }

  // MPO accepts both names; Configure keeps their values synchronized.
  // The fallback also handles configuration before the method is selected.
  std::vector<long long> CurrentRoutingMap() const {
    if (mps) return mps->GetQubitsMap();
    if (mpo) return mpo->GetQubitsMap();
    return {};
  }
  void RefreshRoutingCallback() {
    const bool active = IsRoutingLookaheadEnabled() && !upcomingGates.empty();
    if (mps)
      mps->SetMeetingPositionCallback(
          active ? &ImmediateGpuState::FindBestMeetingPosition : nullptr);
    if (mpo)
      mpo->SetMeetingPositionCallback(
          active ? &ImmediateGpuState::FindBestMeetingPosition : nullptr);
  }

  const char *MaxBondDimensionConfigKey() const {
    return simulationType == SimulationType::kMatrixProductOperator &&
                   configuration.IsSet(
                       "matrix_product_operator_max_bond_dimension")
               ? "matrix_product_operator_max_bond_dimension"
               : "matrix_product_state_max_bond_dimension";
  }

  static int64_t FindBestMeetingPosition(void *thisPtr,
                                         const int64_t *bondDims) noexcept {
    ImmediateGpuState *self = static_cast<ImmediateGpuState *>(thisPtr);

    try {
      return self->FindBestMeetingPositionFunc(bondDims);
    } catch (...) {
      // No exception may escape the optimization callback into the plugin.
      return -1;
    }
  };

  int64_t FindBestMeetingPositionFunc(const int64_t *bondDims) {
    const size_t nQ = GetNumberOfQubits();

    if (lookaheadDepth <= 0 ||
        lookaheadDepth == std::numeric_limits<int>::max())
      return -1;

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
    std::vector<double> bondDimsD(bondDims, bondDims + nrQubits - 1);
    dummySim->SetCurrentBondDimensions(bondDimsD);

    // display bond dimensions for debugging
#ifdef LOG_CALLBACK_INFO
    std::cerr << "Bond dimensions before swapping and applying the gate:";
    for (size_t i = 0; i < nrQubits - 1; ++i) {
      std::cerr << bondDims[i] << " ";
    }
    std::cerr << std::endl;
#endif

    if (upcomingGates.size() <= static_cast<size_t>(upcomingGateIndex)) {
      return -1;  // will fallback
    }

    const auto &op = upcomingGates[upcomingGateIndex];
    const auto qbits = op->AffectedQubits();

    if (qbits.size() != 2) {
      std::cerr << "Error: Meeting position callback called for a gate "
                   "that does not have exactly 2 qubits."
                << std::endl;

      return -1;  // will fallback
    }

#ifdef LOG_CALLBACK_INFO
    const auto &qmap = dummySim->getQubitsMap();

    std::cerr << "Applying 2-qubit gate on physical qubits " << qmap[qbits[0]]
              << " and " << qmap[qbits[1]] << std::endl;

    std::cerr
        << "Finding best meeting position for upcoming gates starting at index "
        << upcomingGateIndex << " with lookahead depth " << lookaheadDepth
        << " and heuristic depth " << lookaheadDepthWithHeuristic << std::endl;

    std::cerr << "Affected qubits: ";
    for (const auto &q : qbits) std::cerr << q << " ";
    std::cerr << std::endl;
#endif

    double bestCost = std::numeric_limits<double>::infinity();
    int64_t res = dummySim->FindBestMeetingPosition(
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
  }

  static void BondDimCallback(void *thisPtr, const int64_t *bondDims) {
    ImmediateGpuState *self = static_cast<ImmediateGpuState *>(thisPtr);

    return self->BondDimCallbackFunc(bondDims);
  }

  static void BondSummaryCallback(void *thisPtr, int64_t maximum) {
    auto *self = static_cast<ImmediateGpuState *>(thisPtr);
    if (maximum > 0)
      self->curMaxBondDim =
          std::max(self->curMaxBondDim, static_cast<size_t>(maximum));
  }

  void BondDimCallbackFunc(const int64_t *bondDims) {
    if (bondDims) {
      const size_t nQ = GetNumberOfQubits();
      for (int i = 0; i < static_cast<int>(nQ) - 1; ++i)
        if (static_cast<size_t>(bondDims[i]) > curMaxBondDim)
          curMaxBondDim = static_cast<size_t>(bondDims[i]);
    }
  }

  SimulationType simulationType =
      SimulationType::kStatevector; /**< The simulation type. */
  uint64_t nextSeedStream = 0;

  std::unique_ptr<GpuLibStateVectorSim>
      state; /**< The gpu statevector simulator. */
  std::unique_ptr<GpuDensityMatrix>
      densityMatrix; /**< The gpu density matrix simulator. */
  std::unique_ptr<GpuMPO>
      mpo; /**< The gpu matrix product operator simulator. */
  std::unique_ptr<GpuLibMPSSim> mps; /**< The gpu MPS simulator. */
  std::unique_ptr<GpuLibTNSim> tn;   /**< The gpu tensor network simulator. */
  std::unique_ptr<GpuPauliPropagator>
      pp; /**< The gpu Pauli propagator simulator. */

  size_t nrQubits = 0; /**< The number of allocated qubits. */

  int lookaheadDepth = 0;
  int lookaheadDepthWithHeuristic = 0;
  bool useOptimalMeetingPosition = true;
  std::vector<std::shared_ptr<Circuits::IOperation<>>> upcomingGates;
  long long int upcomingGateIndex = 0;
  double growthFactorSwap = 1.;
  double growthFactorGate = 0.65;

  std::unique_ptr<Simulators::MPSDummySimulator> dummySim;

  // Observer that counts applied gates to track position in upcomingGates
  class GateCounterObserver : public ISimulatorObserver {
   public:
    GateCounterObserver(long long int &indexRef) : index(indexRef) {}
    void Update(const Types::qubits_vector &) override { ++index; }

   private:
    long long int &index;
  };

  std::shared_ptr<GateCounterObserver> gateCounterObserver;
  size_t curMaxBondDim = 0;

  Configuration configuration; /**< The configuration of the simulator. */
};

}  // namespace Private
}  // namespace Simulators

#endif
#endif
#endif
