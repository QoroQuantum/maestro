/** Per-device resource ownership and CUDA activation at the plugin boundary. */
#pragma once

#ifdef __linux__
#include "GpuLibrary.h"
#include <memory>

namespace Simulators {

// Wrappers retain this resource, including when cloned. A future plugin with
// explicit device contexts can replace namespace isolation behind this
// boundary.
class GpuDeviceContext {
 public:
  GpuDeviceContext(const std::shared_ptr<GpuLibrary>& library)
      : library(library) {}

  class Call {
   public:
    explicit Call(GpuLibrary& library) : library(library), device(library) {}
    GpuLibrary* operator->() const noexcept { return &library; }

   private:
    GpuLibrary& library;
    GpuLibrary::DeviceScope device;
  };

  // The temporary lives until the complete plugin call (including callbacks)
  // finishes. CUDA functions are resolved from this library's own namespace.
  Call operator->() const { return Call(*library); }
  explicit operator bool() const noexcept { return bool(library); }
  operator const std::shared_ptr<GpuLibrary>&() const noexcept {
    return library;
  }
  bool operator==(const GpuDeviceContext& other) const noexcept {
    return library == other.library;
  }
  bool operator!=(const GpuDeviceContext& other) const noexcept {
    return !(*this == other);
  }

 private:
  std::shared_ptr<GpuLibrary> library;
};
}  // namespace Simulators
#endif
