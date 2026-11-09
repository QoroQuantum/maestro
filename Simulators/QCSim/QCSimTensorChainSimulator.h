#pragma once
#if defined(INCLUDED_BY_FACTORY)
#include "../TensorNetworks/BoundedRoutingPlanner.h"
#include "../TensorNetworks/TensorChainRouting.h"
#include "QCSimGateSimulator.h"
#include "QCSimTensorUtils.h"
#include "RoutingMap.h"

#include <optional>

namespace Simulators::Private
{
class QCSimTensorChainSimulator : public TensorChainRouting<QCSimGateSimulator>
{
    using RoutingBase = TensorChainRouting<QCSimGateSimulator>;

  public:
    bool IsRoutingLookaheadEnabled() const override
    {
        return RoutingBase::IsRoutingLookaheadEnabled() || (UsesBoundedRouting() && CurrentRoutingMap().size() == nrQubits);
    }

    void SetLookaheadDepth(int depth) override
    {
        automaticRoutingDepth = false;
        InvalidateBoundedRouting();
        RoutingBase::SetLookaheadDepth(depth);
    }

    void SetAutomaticLookaheadDepth(int depth) override
    {
        automaticRoutingDepth = true;
        InvalidateBoundedRouting();
        RoutingBase::SetLookaheadDepth(depth);
    }

    void SetUpcomingGates(const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &gates) override
    {
        InvalidateBoundedRouting(true);
        RoutingBase::SetUpcomingGates(gates);
    }

    void setGrowthFactorSwap(double factor) override
    {
        RoutingBase::setGrowthFactorSwap(factor);
        RefreshRoutingCallback();
    }

    void setGrowthFactorGate(double factor) override
    {
        RoutingBase::setGrowthFactorGate(factor);
        RefreshRoutingCallback();
    }

    std::string GetConfiguration(const char *key) const override
    {
        if (key && std::string(key) == RoutingConfig::kStrategy && !configuration.IsSet(key))
            return "legacy";
        return RoutingBase::GetConfiguration(key);
    }

  protected:
    // One distinct measured qubit, possibly repeated. Its marginal comes from a
    // single chain contraction and one binomial draw splits all shots; native
    // terminal sampling contracts once per shot instead. probabilityZero(q)
    // returns P(q = 0) for a logical qubit, or nothing when the native sampler
    // must keep the request. No RNG is consumed before that decision.
    template <bool Many, class ProbabilityZero>
    bool SingleQubitCounts(const Types::qubits_vector &qubits, size_t shots, Utils::Sampling::Counts<Many> &result, const ProbabilityZero &probabilityZero)
    {
        if (qubits.empty() || !Utils::Sampling::PreferMultinomial(shots, 2) || !AllowMultinomialCounts() ||
            !std::all_of(qubits.begin(), qubits.end(), [&](auto q) { return q == qubits.front(); }))
            return false;
        const std::optional<double> marginal = probabilityZero(qubits.front());
        if (!marginal)
            return false;
        const double prob0 = *marginal;
        if (!std::isfinite(prob0))
            throw std::domain_error("Tensor-chain sampling requires a finite marginal probability");
        const size_t zeros = Utils::Sampling::BinomialCount(shots, std::clamp(prob0, 0., 1.), [&] { return Utils::RandomStream::Uniform(rng); });
        if constexpr (Many)
        {
            if (zeros)
                result.emplace(std::vector<bool>(qubits.size(), false), zeros);
            if (shots > zeros)
                result.emplace(std::vector<bool>(qubits.size(), true), shots - zeros);
        }
        else
        {
            if (zeros)
                result.emplace(0, zeros);
            if (shots > zeros)
                result.emplace(Types::BitAt(qubits.size()) - 1, shots - zeros);
        }
        NotifyObservers(qubits);
        return true;
    }

    explicit QCSimTensorChainSimulator(bool isOperatorChain) : TensorChainRouting(isOperatorChain, true)
    {
        meetingPositionCallback = [this](const auto &bonds) -> QC::TensorNetworks::MPSSimulatorInterface::IndexType {
            try
            {
                if (!decompositionDepth && upcomingGateIndex >= 0)
                {
                    if (!boundedRoutingPrepared && UsesBoundedRouting() && static_cast<size_t>(upcomingGateIndex) < upcomingGates.size())
                    {
                        const auto &operation = upcomingGates[static_cast<size_t>(upcomingGateIndex)];
                        // Native routing invokes this callback only for nonadjacent
                        // operands. Unsupported boundaries need no operation table.
                        if (operation && operation->GetType() == Circuits::OperationType::kGate && operation->AffectedQubits().size() == 2)
                            PrepareBoundedRouting();
                    }
                    if (boundedRouting)
                        return boundedRouting->Select(CurrentRoutingMap(), bonds.data(), bonds.size(), static_cast<size_t>(upcomingGateIndex));
                }
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

    bool UsesBoundedRouting() const
    {
        // Apply the measured policy only to automatic CPU routing. Explicit
        // depth and cost-model overrides continue to select the legacy policy.
        return automaticRoutingDepth && useOptimalMeetingPosition && lookaheadDepth == 0 && nrQubits > 1 &&
               RoutingConfig::GetStrategy(configuration) == RoutingConfig::Strategy::Combined && !virtualBondsEnabled && growthFactorGate == 0.7 &&
               growthFactorSwap == 1. && (operatorChain ? nrQubits <= 8 : nrQubits > 10);
    }

    void InvalidateBoundedRouting(bool operationsChanged = false)
    {
        boundedRouting.reset();
        boundedRoutingPrepared = false;
        if (operationsChanged)
        {
            boundedRoutingOperations.reset();
            boundedRoutingOperationsQubits = 0;
        }
    }

    // Setters and cloning update eligibility without parsing gates or allocating
    // forecast scratch. Actual nonadjacent execution performs preparation once.
    void RefreshBoundedRouting()
    {
        if (boundedRoutingOperations && boundedRoutingOperationsQubits != nrQubits)
            InvalidateBoundedRouting(true);
        else if (!UsesBoundedRouting() || upcomingGates.empty() ||
                 (boundedRoutingPrepared && (boundedRoutingQubits != nrQubits || boundedRoutingCap != physicalBondCap)))
            InvalidateBoundedRouting();
    }

    void PrepareBoundedRouting()
    {
        if (!UsesBoundedRouting() || upcomingGates.empty())
        {
            InvalidateBoundedRouting();
            return;
        }
        if (boundedRoutingPrepared && boundedRoutingQubits == nrQubits && boundedRoutingCap == physicalBondCap)
            return;
        boundedRouting.reset();
        boundedRoutingPrepared = true;
        boundedRoutingQubits = nrQubits;
        boundedRoutingCap = physicalBondCap;
        try
        {
            if (!boundedRoutingOperations || boundedRoutingOperationsQubits != nrQubits)
            {
                boundedRoutingOperations = BoundedRoutingPlanner::PrepareOperations(nrQubits, upcomingGates);
                boundedRoutingOperationsQubits = nrQubits;
            }
            boundedRouting =
                std::make_unique<BoundedRoutingPlanner>(nrQubits, operatorChain, physicalBondCap, boundedRoutingOperations,
                                                        operatorChain ? BoundedRoutingPlanner::Policy::Work4 : BoundedRoutingPlanner::Policy::Work24);
        }
        catch (...)
        {
            // Routing metadata is advisory. A failed preparation must leave
            // execution on the native routing fallback, without retrying per gate.
        }
    }

    bool automaticRoutingDepth = false;
    std::unique_ptr<BoundedRoutingPlanner> boundedRouting;
    BoundedRoutingPlanner::SharedOperations boundedRoutingOperations;
    size_t boundedRoutingOperationsQubits = 0;
    bool boundedRoutingPrepared = false;
    size_t boundedRoutingQubits = 0;
    MPSDummySimulator::IndexType boundedRoutingCap = 0;

    QC::TensorNetworks::MPSSimulator::MeetingPositionCallback meetingPositionCallback = nullptr;
    std::function<void(Eigen::Index)> bondDimensionCallback = nullptr;
};
} // namespace Simulators::Private
#endif
