/** Compatibility facade for the process-wide GPU library singleton. */
#pragma once

#ifdef __linux__
#include "GpuLibrary.h"
#include <memory>
#include <string>
#include <utility>

namespace Simulators {
class GpuLibraryRegistry {
 public:
  explicit GpuLibraryRegistry(std::string path = "libmaestro_gpu_simulators.so")
      : path(std::move(path)) {}

  std::shared_ptr<GpuLibrary> Acquire(int device, bool mute = false) {
    auto library = GpuLibrary::GetInstance();
    return library->InitializeForDevice(path.c_str(), device, mute) ? library : nullptr;
  }

  int DeviceCount() {
    return GpuLibrary::GetInstance()->DiscoverDevices(path.c_str());
  }

 private:
  std::string path;
};
}  // namespace Simulators
#endif
