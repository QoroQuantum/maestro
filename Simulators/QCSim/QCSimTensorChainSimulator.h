#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../TensorNetworks/TensorChainRouting.h"
#include "QCSimGateSimulator.h"
#include "QCSimTensorUtils.h"
#include "RoutingMap.h"

namespace Simulators::Private
{
class QCSimTensorChainSimulator : public TensorChainRouting<QCSimGateSimulator>
{
  protected:
    explicit QCSimTensorChainSimulator(bool isOperatorChain) : TensorChainRouting(isOperatorChain, true)
    {
        meetingPositionCallback = [this](const auto &bonds) -> QC::TensorNetworks::MPSSimulatorInterface::IndexType {
            try
            {
                return PlanMeetingPosition(bonds.data(), bonds.size());
            }
            catch (...)
            {
                InvalidateDummyHistory();
                return -1;
            }
        };
        bondDimensionCallback = [this](auto maximum) { curMaxBondDim = std::max(curMaxBondDim, static_cast<size_t>(maximum)); };
    }

    QC::TensorNetworks::MPSSimulator::MeetingPositionCallback meetingPositionCallback = nullptr;
    std::function<void(Eigen::Index)> bondDimensionCallback = nullptr;
};
} // namespace Simulators::Private
#endif
