/**
 * @file Factory.h
 * @ingroup simulators
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Factory for simulators.
 *
 * Currently only two simulators are supported: qiskit aer and qcsim.
 * Can be esily extended to support more simulators.
 * Call CreateSimulator with the desired simulator type to create a simulator
 * returned as a shared pointer.
 */

#pragma once

#ifndef _SIMULATORS_FACTORY_H_
#define _SIMULATORS_FACTORY_H_

#include "GpuLibStateVectorSim.h"
#include "DistributedGpuLibrary.h"
#include "DistributedMpiGpuLibrary.h"
#include "GpuDensityMatrix.h"
#include "GpuMPO.h"
#include "GpuLibMPSSim.h"
#include "GpuLibTNSim.h"
#include "GpuStabilizer.h"
#include "GpuPauliPropagator.h"
#include "QuestLibSim.h"
#include "PathIntegralSimulator.h"

#include "Simulator.h"

namespace Simulators {

/**
 * @class SimulatorsFactory
 * @brief Factory for simulators.
 *
 * Create either a qiskit aer or qcsim simulator.
 */
class SimulatorsFactory {
 public:
  /**
   * @brief Create a quantum computing simulator.
   *
   * @param t The type of simulator to create.
   * @return The simulator wrapped in a shared pointer.
   */
  static std::shared_ptr<ISimulator> CreateSimulator(
      SimulatorType t = SimulatorType::kQCSim,
      SimulationType method = SimulationType::kMatrixProductState);

  /**
   * @brief Create a quantum computing simulator.
   *
   * @param t The type of simulator to create.
   * @return The simulator wrapped in a unique pointer.
   */
  static std::unique_ptr<ISimulator> CreateSimulatorUnique(
      SimulatorType t = SimulatorType::kQCSim,
      SimulationType method = SimulationType::kMatrixProductState);

#ifdef __linux__
  // Defined in the core library so Python's hidden-visibility extension
  // shares the same plugin instances and native-state lifetime counters.
  static std::shared_ptr<DistributedGpuLibrary> GetDistributedGpuLibrary();
  static std::shared_ptr<DistributedMpiGpuLibrary> GetDistributedMpiGpuLibrary();
  static bool IsDistributedGpuAvailable() noexcept;
  static void FinalizeDistributedMpiGpuBackend();
  static bool InitGpuLibrary();
  static bool InitGpuLibraryWithMute();

  // Default for subsequently created simulators; explicit gpu_device wins.
  static void SelectGpuDevice(int deviceId);
  static int ResolveGpuDevice(int deviceId = -1);
  static int GetGpuDeviceCount();
  static bool IsGpuLibraryAvailable(int deviceId = -1);
  static std::shared_ptr<GpuLibrary> GetGpuLibrary(int deviceId = -1);

  // Legacy estimators construct simulators synchronously without accepting a
  // configuration map. Scope their default to the requesting network without
  // changing the process-wide default or another network's worker thread.
  class ScopedGpuDevice {
   public:
    explicit ScopedGpuDevice(int deviceId);
    ~ScopedGpuDevice();
    ScopedGpuDevice(const ScopedGpuDevice&) = delete;
    ScopedGpuDevice& operator=(const ScopedGpuDevice&) = delete;
   private:
    int previous;
  };

  static std::unique_ptr<GpuLibStateVectorSim> CreateGpuLibStateVectorSim(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->IsValid()) return nullptr;

    return std::make_unique<GpuLibStateVectorSim>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

  static std::unique_ptr<GpuDensityMatrix> CreateGpuDensityMatrix(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->HasDensityMatrixAPI()) return nullptr;
    return std::make_unique<GpuDensityMatrix>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

  static std::unique_ptr<GpuMPO> CreateGpuMPO(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->HasMPOAPI()) return nullptr;
    return std::make_unique<GpuMPO>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

  static std::unique_ptr<GpuLibMPSSim> CreateGpuLibMPSSim(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->IsValid()) return nullptr;

    return std::make_unique<GpuLibMPSSim>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

  static std::unique_ptr<GpuLibTNSim> CreateGpuLibTensorNetSim(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->IsValid()) return nullptr;

    return std::make_unique<GpuLibTNSim>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

  static std::shared_ptr<GpuStabilizer> CreateGpuStabilizerSimulator(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->IsValid()) return nullptr;
    return std::make_shared<GpuStabilizer>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

  static std::shared_ptr<GpuPauliPropagator>
  CreateGpuPauliPropagatorSimulator(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->IsValid()) return nullptr;
    return std::make_shared<GpuPauliPropagator>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

  static std::unique_ptr<GpuPauliPropagator>
  CreateGpuPauliPropagatorSimulatorUnique(int deviceId = -1) {
    auto initializationLock = GpuLibrary::GetInstance()->LockInitialization();
    auto gpuLibrary = GetGpuLibrary(deviceId);
    if (!gpuLibrary || !gpuLibrary->IsValid()) return nullptr;
    return std::make_unique<GpuPauliPropagator>(gpuLibrary, gpuLibrary->GetCreationDevice());
  }

 private:
  static std::atomic_int requestedGpuDeviceId;
  static thread_local int scopedGpuDeviceId;

 public:
#else
  static bool IsGpuLibraryAvailable(int = -1) { return false; }

  static bool InitGpuLibrary() { return false; }

  static void SelectGpuDevice(int) {}

  static int GetGpuDeviceCount() { return 0; }
#endif
  static bool InitQuestLibrary();
  static bool InitQuestLibraryWithMute();
  static bool IsQuestLibraryAvailable() {
    return questLibrary && questLibrary->IsValid();
  }

  static std::shared_ptr<QuestLibSim> GetQuestLibrary() {
    if (!questLibrary || !questLibrary->IsValid()) return nullptr;
    return questLibrary;
  }

  static std::shared_ptr<PathIntegralSimulator> CreatePathIntegralSimulator() {
    return std::make_shared<PathIntegralSimulator>();
  }

 private:
  static std::shared_ptr<QuestLibSim> questLibrary;
  static std::atomic_bool firstTimeQuest;
};

}  // namespace Simulators

#endif  // !_SIMULATORS_FACTORY_H_
