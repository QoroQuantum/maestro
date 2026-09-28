#pragma once
#if defined(__linux__) && defined(INCLUDED_BY_FACTORY)
#include "ImmediateDistributedGpuSimulator.h"
#include "FusionSimulator.h"
namespace Simulators::Private {
template <class Immediate>
class DistributedFusionSimulator : public FusionSimulator<FusionState> {
 public:
  DistributedFusionSimulator()
      : FusionSimulator(std::make_shared<Immediate>()) {}
  unsigned GetGateFusionMaxQubits() const override {
    // Ex local (the default) and Ex MPI already queue/fuse in the plugin.
    if (GetType() == SimulatorType::kDistMpiGpuSim ||
        GetConfiguration("distributed_backend") != "conventional")
      return 0;
    return DistributedGpuLibrary::GetInstance()->HasThreeQubitMatrixAPI() ? 3
                                                                          : 2;
  }
  std::string GetConfiguration(const char* key) const override {
    // Layout is an observation of executed gates, just like amplitudes.
    if (std::string(key) == "distributed_qubit_layout")
      const_cast<DistributedFusionSimulator*>(this)->Flush();
    return FusionSimulator<FusionState>::GetConfiguration(key);
  }
  std::unique_ptr<ISimulator> Clone() override {
    auto copy = std::make_unique<DistributedFusionSimulator>();
    CloneInto(*copy);
    return copy;
  }

 protected:
  bool PreserveStructuredGates() const override { return true; }
  bool UsesDestructiveStateStorage() const override { return true; }
};
using DistributedGpuSimulator =
    DistributedFusionSimulator<ImmediateDistributedGpuSimulator>;
using DistributedMpiGpuSimulator =
    DistributedFusionSimulator<ImmediateDistributedMpiGpuSimulator>;
}  // namespace Simulators::Private
#endif
