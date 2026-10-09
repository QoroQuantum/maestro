#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../Fusion/FusionSimulator.h"
#include "GpuState.h"

namespace Simulators::Private
{
class GpuSimulator : public FusionSimulator<GpuState>
{
  public:
    using FusionSimulator::FusionSimulator;

    std::unique_ptr<ISimulator> Clone() override
    {
        auto copy = std::make_unique<GpuSimulator>(immediate_);
        CloneInto(*copy);
        return copy;
    }
};
} // namespace Simulators::Private
#endif
