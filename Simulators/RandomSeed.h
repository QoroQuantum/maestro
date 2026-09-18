// Fresh execution seeds, including one shared seed for an MPI communicator.
#pragma once
#include "State.h"
#ifdef __linux__
#include "DistributedMpiGpuLibrary.h"
#endif
#include <charconv>

namespace Simulators {
inline uint64_t GenerateRandomSeed(
    SimulatorType type,
    const std::unordered_map<std::string, std::string>& options = {}) {
  if (type == SimulatorType::kDistMpiGpuSim) {
#ifdef __linux__
    DistributedMpiGpuLibrary::Communicator descriptor{sizeof(descriptor), 0, 0};
    const auto found = options.find("mpi_communicator");
    const auto* communicator = &descriptor;
    if (found == options.end()) {
      communicator = nullptr;
    } else {
      const auto& value = found->second;
      const auto parsed = std::from_chars(
          value.data(), value.data() + value.size(), descriptor.fortran_handle);
      if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("Invalid MPI communicator handle");
    }
    return DistributedMpiGpuLibrary::GetInstance()->GenerateSeed(communicator);
#else
    throw std::runtime_error("MPI GPU execution requires Linux");
#endif
  }
  std::random_device entropy;
  return std::uniform_int_distribution<uint64_t>{}(entropy);
}
}  // namespace Simulators
