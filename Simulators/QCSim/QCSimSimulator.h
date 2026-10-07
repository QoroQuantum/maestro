#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../Fusion/FusionSimulator.h"
#include "QCSimState.h"

namespace Simulators::Private
{
class QCSimSimulator : public FusionSimulator<QCSimState>
{
  public:
    std::unique_ptr<ISimulator> Clone() override
    {
        auto copy = std::make_unique<QCSimSimulator>();
        CloneInto(*copy);
        return copy;
    }
};
} // namespace Simulators::Private
#endif
