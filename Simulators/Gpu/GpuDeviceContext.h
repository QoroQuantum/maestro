/** Compatibility name: device ownership now lives in the GPU plugin. */
#pragma once

#ifdef __linux__
#include "GpuLibrary.h"
#include <memory>

namespace Simulators {
using GpuDeviceContext = std::shared_ptr<GpuLibrary>;
}  // namespace Simulators
#endif
