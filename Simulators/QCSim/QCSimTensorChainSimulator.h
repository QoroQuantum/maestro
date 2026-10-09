#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../TensorNetworks/MPSDummySimulator.h"
#include "QCSimGateSimulator.h"
#include "QCSimTensorUtils.h"
#include "RoutingMap.h"

namespace Simulators::Private
{
// Lookahead and bond-dimension tracking shared by the MPS and MPO backends.
class QCSimTensorChainSimulator : public QCSimGateSimulator
{
  protected:
    explicit QCSimTensorChainSimulator(bool isOperatorChain) : operatorChain(isOperatorChain)
    {
        meetingPositionCallback = [this](const auto &bondDims) -> QC::TensorNetworks::MPSSimulatorInterface::IndexType {
            try
            {
                if (lookaheadDepth <= 0 || lookaheadDepth == std::numeric_limits<int>::max())
                    return -1;
                if (upcomingGates.empty() || upcomingGateIndex >= static_cast<long long>(upcomingGates.size()))
                {
                    return -1;
                }
                const size_t nQ = bondDims.size() + 1;
                if (!dummySim || dummySim->getNrQubits() != nQ || dummySim->IsOperatorChain() != (operatorChain))
                {
                    dummySim = std::make_unique<Simulators::MPSDummySimulator>(nQ, operatorChain);
                    dummySim->SetMaxBondDimension(configuration.GetConfigurationAsInt(MaxBondDimensionConfigKey()));
                    dummySim->setGrowthFactorGate(growthFactorGate);
                    dummySim->setGrowthFactorSwap(growthFactorSwap);
                }
                const auto actualMap = CurrentRoutingMap();
                if (actualMap.size() != nrQubits)
                    return -1;
                dummySim->SetInitialQubitsMap(actualMap);
                dummySim->setTotalSwappingCost(0);
                std::vector<double> bondDimsD(bondDims.begin(), bondDims.end());
                dummySim->SetCurrentBondDimensions(bondDimsD);
#ifdef LOG_CALLBACK_INFO
                std::cerr << "Bond dimensions before swapping and applying the gate:";
                for (size_t i = 0; i < bondDims.size(); ++i)
                {
                    std::cerr << bondDims[i] << " ";
                }
                std::cerr << std::endl;
#endif
                const auto &op = upcomingGates[upcomingGateIndex];
                const auto qbits = op->AffectedQubits();
                if (qbits.size() != 2)
                    return -1;
#ifdef LOG_CALLBACK_INFO
                const auto &qmap = dummySim->getQubitsMap();
                std::cerr << "Applying 2-qubit gate on physical qubits " << qmap[qbits[0]] << " and " << qmap[qbits[1]] << std::endl;
                std::cerr << "Finding best meeting position for upcoming gates "
                             "starting at index "
                          << upcomingGateIndex << " with lookahead depth " << lookaheadDepth << " and heuristic depth " << lookaheadDepthWithHeuristic
                          << std::endl;
                std::cerr << "Affected qubits: ";
                for (const auto &q : qbits)
                    std::cerr << q << " ";
                std::cerr << std::endl;
#endif
                double bestCost = std::numeric_limits<double>::infinity();
                auto res = dummySim->FindBestMeetingPosition(upcomingGates, upcomingGateIndex, lookaheadDepth, lookaheadDepthWithHeuristic, 0, bestCost);
#ifdef LOG_CALLBACK_INFO
                std::cerr << "Swapping the two qubits on position: " << res << " and " << (res + 1) << std::endl;
#endif
                dummySim->SwapQubitsToPosition(qbits[0], qbits[1], res);
                dummySim->ApplyGate(op);
#ifdef LOG_CALLBACK_INFO
                const auto &expectedBondDims = dummySim->getCurrentBondDimensions();
                std::cerr << "Expected bond dimensions after swapping and applying "
                             "the gate: ";
                for (size_t i = 0; i < expectedBondDims.size(); ++i)
                {
                    std::cerr << expectedBondDims[i] << " ";
                }
                std::cerr << std::endl;
                std::cerr << "Best meeting position: " << res << " with estimated cost: " << bestCost << std::endl;
#endif
                return res;
            }
            catch (...)
            {
                return -1;
            }
        };
        bondDimensionCallback = [this](auto maximum) { curMaxBondDim = std::max(curMaxBondDim, static_cast<size_t>(maximum)); };
    }

    virtual std::vector<long long> CurrentRoutingMap() const = 0;
    const bool operatorChain;

  public:
    bool IsRoutingLookaheadEnabled() const override
    {
        return useOptimalMeetingPosition && lookaheadDepth > 0 && lookaheadDepth != std::numeric_limits<int>::max() && nrQubits != 0 &&
               CurrentRoutingMap().size() == nrQubits;
    }

    void SetLookaheadDepth(int depth) override
    {
        lookaheadDepth = depth;
        RefreshRoutingCallback();
    }

    void SetLookaheadDepthWithHeuristic(int depth) override
    {
        lookaheadDepthWithHeuristic = depth;
        if (lookaheadDepth < depth)
            SetLookaheadDepth(depth);
    }

    double getGrowthFactorSwap() const override
    {
        return growthFactorSwap;
    }

    double getGrowthFactorGate() const override
    {
        return growthFactorGate;
    }

    void setGrowthFactorSwap(double factor) override
    {
        growthFactorSwap = factor;
        if (dummySim)
            dummySim->setGrowthFactorSwap(factor);
    }

    void setGrowthFactorGate(double factor) override
    {
        growthFactorGate = factor;
        if (dummySim)
            dummySim->setGrowthFactorGate(factor);
    }

    size_t GetCurrentMaxBondDimension() const override
    {
        return curMaxBondDim;
    }

  protected:
    const char *MaxBondDimensionConfigKey() const
    {
        return operatorChain && configuration.IsSet("matrix_product_operator_max_bond_dimension") ? "matrix_product_operator_max_bond_dimension"
                                                                                                  : "matrix_product_state_max_bond_dimension";
    }

    void ResetDummySimulator()
    {
        if (!dummySim)
            return;
        std::vector<long long int> identityMap(nrQubits);
        for (size_t qubit = 0; qubit < nrQubits; ++qubit)
            identityMap[qubit] = static_cast<long long int>(qubit);
        dummySim->SetInitialQubitsMap(identityMap);
        dummySim->setTotalSwappingCost(0.);
        if (nrQubits > 1)
            dummySim->SetCurrentBondDimensions(std::vector<double>(nrQubits - 1, 1.));
    }

    int lookaheadDepth = 0;

    int lookaheadDepthWithHeuristic = 0;

    bool useOptimalMeetingPosition = true;

    double growthFactorSwap = 1.;

    double growthFactorGate = 0.65;

    std::unique_ptr<Simulators::MPSDummySimulator> dummySim;

    size_t curMaxBondDim = 0;

    QC::TensorNetworks::MPSSimulator::MeetingPositionCallback meetingPositionCallback = nullptr;

    std::function<void(Eigen::Index)> bondDimensionCallback = nullptr;
};
} // namespace Simulators::Private
#endif
