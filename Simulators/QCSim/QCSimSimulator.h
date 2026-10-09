#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../Fusion/FusionSimulator.h"
#include "QCSimState.h"

namespace Simulators::Private
{
class QCSimSimulator : public FusionSimulator<QCSimState>
{
  public:
    using FusionSimulator::FusionSimulator;

    std::unique_ptr<ISimulator> Clone() override
    {
        auto copy = std::make_unique<QCSimSimulator>(immediate_);
        CloneInto(*copy);
        return copy;
    }

    std::unique_ptr<ISimulator> CloneForExecution(uint64_t seed) override
    {
        auto copy = std::make_unique<QCSimSimulator>(immediate_);
        CloneInto(*copy, &seed);
        return copy;
    }
};
} // namespace Simulators::Private
#endif
