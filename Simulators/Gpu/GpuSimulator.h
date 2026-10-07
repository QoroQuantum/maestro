#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Fusion/FusionSimulator.h"
#include "GpuState.h"

namespace Simulators::Private
{
class GpuSimulator : public FusionSimulator<GpuState>
{
  public:
    std::unique_ptr<ISimulator> Clone() override
    {
        auto copy = std::make_unique<GpuSimulator>();
        CloneInto(*copy);
        return copy;
    }
};
} // namespace Simulators::Private
#endif
