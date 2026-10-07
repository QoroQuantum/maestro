#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../Fusion/FusionState.h"
#include "ImmediateQCSimSimulator.h"

namespace Simulators::Private
{
class QCSimState : public FusionState
{
  public:
    QCSimState() : FusionState(std::make_shared<ImmediateQCSimSimulator>())
    {
    }

    unsigned GetGateFusionMaxQubits() const override
    {
        switch (GetSimulationType())
        {
        case SimulationType::kStatevector:
        case SimulationType::kDensityMatrix:
            return 3;
        case SimulationType::kMatrixProductState:
        case SimulationType::kMatrixProductOperator:
        case SimulationType::kTensorNetwork:
            return 2;
        default:
            return 0;
        }
    }

  protected:
    // Measured break-even of fused against native execution (geometric mean
    // over random, QAOA, QFT, quantum-volume and GHZ circuits). Below it the
    // state fits in cache, so fusing saves no memory passes and the dense
    // kernels cost more than the native gates they replace.
    size_t DefaultGateFusionMinQubits() const override
    {
        switch (GetSimulationType())
        {
        case SimulationType::kStatevector:
            return 11;
        case SimulationType::kDensityMatrix:
            return 5;
        default:
            return 0;
        }
    }
};
} // namespace Simulators::Private
#endif
