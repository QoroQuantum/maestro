/** Lazy, synchronized ownership of isolated GPU plugin instances. */
#pragma once

#ifdef __linux__
#include "GpuLibrary.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Simulators {
class GpuLibraryRegistry {
 public:
  explicit GpuLibraryRegistry(std::string path = "libmaestro_gpu_simulators.so")
      : path(std::move(path)) {}

  std::shared_ptr<GpuLibrary> Acquire(int device, bool mute = false) {
    if (device < 0)
      throw std::invalid_argument("gpu_device must be nonnegative");
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = libraries.find(device);
    if (found != libraries.end()) return found->second;

    // Reject out-of-range requests using a loaded namespace, before spending
    // another of glibc's limited namespaces just to discover the same count.
    if (!libraries.empty()) {
      const int count = libraries.begin()->second->GetGpuDeviceCount();
      if (device >= count) {
        if (!mute)
          std::cerr << "GpuLibrary: GPU device " << device << " is unavailable ("
                    << count << " visible devices)" << std::endl;
        return nullptr;
      }
    }

    auto library = probe ? std::move(probe) : std::make_shared<GpuLibrary>();
    library->SetMute(mute);
    library->SetRequestedGpuDevice(device);
    if (!library->Load(path.c_str())) return nullptr;
    const int count = library->GetGpuDeviceCount();
    if (device >= count) {
      // Retain the uninitialized namespace for discovery / a valid request.
      probe = std::move(library);
      if (!mute)
        std::cerr << "GpuLibrary: GPU device " << device << " is unavailable ("
                  << count << " visible devices)" << std::endl;
      return nullptr;
    }
    if (!library->Init(path.c_str())) return nullptr;
    libraries.emplace(device, library);
    return library;
  }

  int DeviceCount() {
    std::lock_guard<std::mutex> lock(mutex);
    if (!libraries.empty())
      return libraries.begin()->second->GetGpuDeviceCount();
    if (!probe) probe = std::make_shared<GpuLibrary>();
    probe->SetMute(true);
    if (!probe->Load(path.c_str())) {
      probe.reset();
      return 0;
    }
    return probe->GetGpuDeviceCount();
  }

 private:
  std::string path;
  std::mutex mutex;
  std::unordered_map<int, std::shared_ptr<GpuLibrary>> libraries;
  std::shared_ptr<GpuLibrary> probe;
};
}  // namespace Simulators
#endif
