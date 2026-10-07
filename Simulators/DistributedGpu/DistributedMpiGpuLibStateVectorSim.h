#pragma once
#ifdef __linux__
#include "DistributedGpuLibStateVectorSim.h"
#include "DistributedMpiGpuLibrary.h"
namespace Simulators {
class DistributedMpiGpuLibStateVectorSim
    : public DistributedGpuLibStateVectorSim {
 public:
  DistributedMpiGpuLibStateVectorSim(
      std::shared_ptr<DistributedMpiGpuLibrary> lib,
      const DistributedMpiGpuLibrary::Communicator* comm, int device,
      unsigned p2pBits)
      : DistributedGpuLibStateVectorSim(
            lib, lib->CreateMpiNative(comm, device, p2pBits)) {}
};
}  // namespace Simulators
#endif
