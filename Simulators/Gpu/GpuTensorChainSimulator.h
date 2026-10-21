#pragma once
#if defined(INCLUDED_BY_FACTORY) && defined(__linux__)
#include "../TensorNetworks/TensorChainRouting.h"
#include "ImmediateGpuSimulator.h"

namespace Simulators::Private
{
class GpuTensorChainSimulator : public TensorChainRouting<ImmediateGpuSimulator>
{
  protected:
    explicit GpuTensorChainSimulator(bool isOperatorChain) : TensorChainRouting(isOperatorChain)
    {
    }

    using MeetingPositionCallback = int64_t (*)(void *, const int64_t *);

    virtual MeetingPositionCallback GetMeetingPositionCallback() const
    {
        return &FindBestMeetingPosition;
    }

    static int64_t FindBestMeetingPosition(void *thisPtr, const int64_t *bonds) noexcept
    {
        auto *self = static_cast<GpuTensorChainSimulator *>(thisPtr);
        try
        {
            return self->PlanMeetingPosition(bonds, self->nrQubits ? self->nrQubits - 1 : 0);
        }
        catch (...)
        {
            self->InvalidateDummyHistory();
            return -1;
        }
    }

    static void BondDimCallback(void *thisPtr, const int64_t *bondDims)
    {
        GpuTensorChainSimulator *self = static_cast<GpuTensorChainSimulator *>(thisPtr);
        return self->BondDimCallbackFunc(bondDims);
    }

    static void BondSummaryCallback(void *thisPtr, int64_t maximum)
    {
        auto *self = static_cast<GpuTensorChainSimulator *>(thisPtr);
        if (maximum > 0)
            self->curMaxBondDim = std::max(self->curMaxBondDim, static_cast<size_t>(maximum));
    }

    void BondDimCallbackFunc(const int64_t *bondDims)
    {
        if (bondDims)
        {
            const size_t nQ = GetNumberOfQubits();
            for (int i = 0; i < static_cast<int>(nQ) - 1; ++i)
                if (static_cast<size_t>(bondDims[i]) > curMaxBondDim)
                    curMaxBondDim = static_cast<size_t>(bondDims[i]);
        }
    }
};
} // namespace Simulators::Private
#endif
