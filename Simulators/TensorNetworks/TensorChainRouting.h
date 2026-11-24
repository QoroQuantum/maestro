#pragma once
#include "MPSDummySimulator.h"
#include "RoutingConfig.h"

namespace Simulators::Private
{
// CPU and GPU adapters supply native maps and bonds; all history and planning
// policy lives here so a boundary has identical semantics on both backends.
template <class Backend> class TensorChainRouting : public Backend
{
  protected:
    using Backend::configuration;
    using Backend::nrQubits;
    using Backend::RefreshRoutingCallback;
    using Backend::upcomingGateIndex;
    using Backend::upcomingGates;

    explicit TensorChainRouting(bool isOperatorChain, bool relabelSwaps = false) : operatorChain(isOperatorChain), swapGatesRelabelQubits(relabelSwaps)
    {
    }

    // Only an exact SWAP can use the logical relabelling path. Approximate SWAPs
    // still update tensors and must be recorded as ordinary matrix gates.
    static bool IsExactSwapMatrix(const Eigen::Matrix4cd &gate)
    {
        for (Eigen::Index row = 0; row < 4; ++row)
            for (Eigen::Index col = 0; col < 4; ++col)
            {
                const Eigen::Index swapped = ((col & 1) << 1) | (col >> 1);
                if (gate(row, col) != std::complex<double>(row == swapped ? 1. : 0., 0.))
                    return false;
            }
        return true;
    }

    virtual std::vector<long long> CurrentRoutingMap() const = 0;
    virtual std::vector<double> CurrentBondDimensions() const = 0;
    const bool operatorChain;
    bool swapGatesRelabelQubits;

  public:
    bool SwapGatesRelabelQubits() const override
    {
        return swapGatesRelabelQubits;
    }

    bool IsRoutingLookaheadEnabled() const override
    {
        return useOptimalMeetingPosition && lookaheadDepth > 0 && lookaheadDepth != std::numeric_limits<int>::max() && nrQubits > 1 &&
               CurrentRoutingMap().size() == nrQubits;
    }

    void SetLookaheadDepth(int depth) override
    {
        lookaheadDepth = depth;
        RefreshRoutingSettings();
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

    void SetUpcomingGates(const std::vector<std::shared_ptr<Circuits::IOperation<double>>> &gates) override
    {
        upcomingHasDisentangle = RoutingConfig::ContainsDisentanglingOperations(gates);
        RefreshRoutingSettings();
        Backend::SetUpcomingGates(gates);
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

    MPSDummySimulator::IndexType EffectiveLookaheadMaxBondDimension(size_t nQ, MPSDummySimulator::IndexType cap) const
    {
        return virtualBondsEnabled ? MPSDummySimulator::EffectiveLookaheadMaxBondDimension(nQ, operatorChain, cap) : cap;
    }

    MPSDummySimulator::IndexType EffectiveLookaheadMaxBondDimension(size_t nQ) const
    {
        return EffectiveLookaheadMaxBondDimension(nQ, physicalBondCap);
    }

    // Called when configuration, register size or lookahead changes, never on
    // the gate hot path. Invalidation retains memoized plans: their keys include
    // the full map, bond estimates and upcoming gate window.
    void RefreshRoutingSettings()
    {
        const auto cap = configuration.GetConfigurationAsInt(MaxBondDimensionConfigKey());
        const bool explicitEnabled = RoutingConfig::VirtualBondsEnabled(configuration);
        const bool enabled = explicitEnabled || (TracksDisentangling() && upcomingHasDisentangle && RoutingConfig::VirtualBondsAutoAllowed(configuration));
        const double weight = RoutingConfig::TruncationWeight(configuration);
        const bool tracking = enabled && useOptimalMeetingPosition && lookaheadDepth > 0 && lookaheadDepth != std::numeric_limits<int>::max() && nrQubits > 1 &&
                              cap > 0 && MPSDummySimulator::EffectiveLookaheadMaxBondDimension(nrQubits, operatorChain, cap) > cap;
        if (cap != physicalBondCap || explicitEnabled != explicitVirtualBondsEnabled || enabled != virtualBondsEnabled || tracking != trackVirtualBonds ||
            weight != truncationWeight)
            InvalidateDummyHistory();
        physicalBondCap = cap;
        explicitVirtualBondsEnabled = explicitEnabled;
        virtualBondsEnabled = enabled;
        truncationWeight = weight;
        trackVirtualBonds = tracking;
        if (dummySim)
            ConfigureDummy();
    }

    bool ShouldTrackVirtualBonds() const
    {
        return trackVirtualBonds;
    }

    void InvalidateDummyHistory() noexcept
    {
        dummyHistoryValid = false;
    }

    void ValidateDummyHistoryMap() noexcept
    {
        if (!trackVirtualBonds || !dummyHistoryValid || !dummySim)
            return;
        try
        {
            if (dummySim->getQubitsMap() != CurrentRoutingMap())
                InvalidateDummyHistory();
        }
        catch (...)
        {
            InvalidateDummyHistory();
        }
    }

    void SetSwapGatesRelabelQubits(bool enabled)
    {
        if (swapGatesRelabelQubits == enabled)
            return;
        swapGatesRelabelQubits = enabled;
        InvalidateDummyHistory();
        if (dummySim)
            ConfigureDummy();
    }

    void ConfigureDummy()
    {
        dummySim->SetSwapGatesRelabelQubits(swapGatesRelabelQubits);
        dummySim->SetModelDisentangling(TracksDisentangling());
        dummySim->SetMaxBondDimension(EffectiveLookaheadMaxBondDimension(dummySim->getNrQubits()));
        dummySim->SetRoutingCostModel(virtualBondsEnabled ? physicalBondCap : 0, truncationWeight);
        dummySim->setGrowthFactorGate(growthFactorGate);
        dummySim->setGrowthFactorSwap(growthFactorSwap);
    }

    void EnsureDummySimulator(size_t nQ)
    {
        if (!dummySim || dummySim->getNrQubits() != nQ)
        {
            dummySim = std::make_unique<MPSDummySimulator>(nQ, operatorChain);
            InvalidateDummyHistory();
            ConfigureDummy();
        }
    }

    virtual void DeflateLiveBondDimensions(const std::vector<long long> & /*map*/, double * /*dims*/, size_t /*count*/) const
    {
    }

    // Whether the backend reports the qubits its measurements and resets collapse (RecordDisentangledQubits), so
    // that the site ranks follow the state. Only then do measurements and resets enter the routing model and
    // switch its overflow pricing on; a backend that does not (the GPU chains) keeps the gate-only model.
    virtual bool TracksDisentangling() const
    {
        return false;
    }

    void EnsureLiveSiteRanks()
    {
        if (nrQubits > 0 && liveSiteRank.size() != nrQubits)
            liveSiteRank.assign(nrQubits, operatorChain ? 4. : 2.);
    }

    void ResetLiveSiteRanks()
    {
        if (nrQubits > 0)
            liveSiteRank.assign(nrQubits, operatorChain ? 4. : 2.);
        else
            liveSiteRank.clear();
    }

    void SeedDummyHistory(const std::vector<long long> &map)
    {
        InvalidateDummyHistory();
        const auto bonds = CurrentBondDimensions();
        if (map.size() != nrQubits || bonds.size() + 1 != nrQubits)
            return;
        dummySim->SetRoutingMap(map);
        dummySim->SetCurrentBondDimensions(bonds);
        if (liveSiteRank.size() == nrQubits)
            dummySim->SetCurrentSiteRanks(liveSiteRank);
        dummyHistoryValid = true;
    }

    template <class Bond> long long PlanMeetingPosition(const Bond *bonds, size_t count)
    {
        if (!useOptimalMeetingPosition || lookaheadDepth <= 0 || lookaheadDepth == std::numeric_limits<int>::max() || decompositionDepth || !bonds ||
            nrQubits <= 1 || count + 1 != nrQubits || upcomingGateIndex < 0 || static_cast<size_t>(upcomingGateIndex) >= upcomingGates.size())
            return -1;
        const auto &op = upcomingGates[upcomingGateIndex];
        if (!op || op->GetType() != Circuits::OperationType::kGate)
            return -1;
        const auto qubits = op->AffectedQubits();
        if (qubits.size() != 2 || qubits[0] >= nrQubits || qubits[1] >= nrQubits || qubits[0] == qubits[1])
            return -1;
        const auto map = CurrentRoutingMap();
        if (map.size() != nrQubits)
            return -1;
        EnsureDummySimulator(nrQubits);
        EnsureLiveSiteRanks();
        std::vector<double> estimates(bonds, bonds + count);
        DeflateLiveBondDimensions(map, estimates.data(), estimates.size());
        if (trackVirtualBonds && dummyHistoryValid && dummySim->getQubitsMap() == map)
        {
            const auto &previous = dummySim->getCurrentBondDimensions();
            for (size_t i = 0; i < count; ++i)
                if (estimates[i] >= static_cast<double>(physicalBondCap))
                    estimates[i] = std::max(estimates[i], previous[i]);
        }
        dummySim->SetRoutingMap(map);
        dummySim->SetCurrentBondDimensions(estimates);
        if (liveSiteRank.size() == nrQubits)
            dummySim->SetCurrentSiteRanks(liveSiteRank);
        dummySim->setTotalSwappingCost(0.);
        dummyHistoryValid = trackVirtualBonds;
        double bestCost = std::numeric_limits<double>::infinity();
        const auto position =
            dummySim->SelectMeetingPositionAdaptive(upcomingGates, upcomingGateIndex, lookaheadDepth, lookaheadDepthWithHeuristic, 0., bestCost);
        if (position < std::min(map[qubits[0]], map[qubits[1]]) || position >= std::max(map[qubits[0]], map[qubits[1]]))
            return -1;
        if (trackVirtualBonds)
            dummySim->SwapQubitsToPosition(qubits[0], qubits[1], position);
        return position;
    }

    // Effective Schmidt rank (EffectiveSchmidtRank) of a QCSim gate object.
    template <class Gate> static double NativeGateSchmidtRank(const Gate &gate)
    {
        return gate.isSwapGate() ? 4. : EffectiveSchmidtRank(gate.getRawOperatorMatrix());
    }

    // schmidtRank: the executed gate's effective Schmidt rank (2 for CX, 4 for
    // SWAP or a generic gate, near 1 for a small-angle controlled rotation).
    void RecordExecutedTwoQubitGate(Types::qubit_t q0, Types::qubit_t q1, bool isSwap, double schmidtRank) noexcept
    {
        try
        {
            if (q0 < nrQubits && q1 < nrQubits && q0 != q1 && !isSwap && schmidtRank > 1. + 1e-9)
            {
                EnsureLiveSiteRanks();
                const double d = operatorChain ? 4. : 2.;
                const double fullRank = operatorChain ? schmidtRank * schmidtRank : schmidtRank;
                const double factor = 1. + growthFactorGate * (std::min(d, fullRank) - 1.);
                liveSiteRank[q0] = std::clamp(liveSiteRank[q0] * factor, 1., d);
                liveSiteRank[q1] = std::clamp(liveSiteRank[q1] * factor, 1., d);
            }
        }
        catch (...)
        {
        }

        if (!trackVirtualBonds)
            return;
        // Recording is advisory. Never throw after a successful native gate:
        // observers must still advance the logical operation index.
        try
        {
            if (q0 >= nrQubits || q1 >= nrQubits || q0 == q1)
                return InvalidateDummyHistory();
            const auto map = CurrentRoutingMap();
            if (map.size() != nrQubits)
                return InvalidateDummyHistory();
            EnsureDummySimulator(nrQubits);
            if (isSwap && swapGatesRelabelQubits)
            {
                if (!dummyHistoryValid)
                    return SeedDummyHistory(map);
                dummySim->RelabelQubits(q0, q1);
                if (dummySim->getQubitsMap() != map)
                    SeedDummyHistory(map);
                return; // The tensors and their bond estimates have not changed.
            }
            if (std::abs(map[q0] - map[q1]) != 1)
                return InvalidateDummyHistory();
            if (!dummyHistoryValid)
                return SeedDummyHistory(map); // Native bonds already include this gate.
            const auto &predicted = dummySim->getQubitsMap();
            const auto position = std::min(map[q0], map[q1]);
            if (std::abs(predicted[q0] - predicted[q1]) > 1 && position >= std::min(predicted[q0], predicted[q1]) &&
                position < std::max(predicted[q0], predicted[q1]))
                dummySim->SwapQubitsToPosition(q0, q1, position);
            // Never attach estimates to different chain cuts by overwriting
            // the permutation. Unknown native reorders require a fresh seed.
            if (dummySim->getQubitsMap() != map)
                return SeedDummyHistory(map);
            // The rank describes the actual native call: a lone fused CX keeps
            // rank 2; a merged matrix is ranked from its own entries.
            dummySim->RecordAdjacentTwoQubitGate(position, isSwap, schmidtRank);
        }
        catch (...)
        {
            InvalidateDummyHistory();
        }
    }

    void RecordDisentangledQubits(const Types::qubits_vector &qubits, double probability = 1.) noexcept
    {
        try
        {
            EnsureLiveSiteRanks();
            for (const Types::qubit_t q : qubits)
                if (q < liveSiteRank.size())
                    liveSiteRank[q] = probability >= 1. ? 1. : std::max(1., liveSiteRank[q] + probability * (1. - liveSiteRank[q]));

            if (!trackVirtualBonds || upcomingGates.empty() || !dummyHistoryValid || !dummySim)
                return InvalidateDummyHistory();
            const auto map = CurrentRoutingMap();
            if (map.size() != nrQubits || dummySim->getQubitsMap() != map)
                return InvalidateDummyHistory();
            for (const Types::qubit_t q : qubits)
                if (q < nrQubits)
                    dummySim->ApplyDisentangleQubit(static_cast<MPSDummySimulator::IndexType>(q), probability);
            const auto &dims = dummySim->getCurrentBondDimensions();
            if (std::all_of(dims.begin(), dims.end(), [](double b) { return b <= 1. + 1e-9; }))
                InvalidateDummyHistory();
        }
        catch (...)
        {
            InvalidateDummyHistory();
        }
    }

    void NoteTwoQubitNonUnitaryTouch(Types::qubit_t q0, Types::qubit_t q1) noexcept
    {
        try
        {
            EnsureLiveSiteRanks();
            const double d = operatorChain ? 4. : 2.;
            if (q0 < liveSiteRank.size())
                liveSiteRank[q0] = d;
            if (q1 < liveSiteRank.size())
                liveSiteRank[q1] = d;
        }
        catch (...)
        {
        }
    }

    std::vector<double> CaptureDummyHistory() const
    {
        return trackVirtualBonds && dummyHistoryValid && dummySim ? dummySim->getCurrentBondDimensions() : std::vector<double>{};
    }

    void RestoreDummyHistory(const std::vector<double> &bonds)
    {
        InvalidateDummyHistory();
        if (!trackVirtualBonds || bonds.empty() || bonds.size() + 1 != nrQubits)
            return;
        EnsureDummySimulator(nrQubits);
        const auto map = CurrentRoutingMap();
        if (map.size() != nrQubits)
            return;
        dummySim->SetRoutingMap(map);
        dummySim->SetCurrentBondDimensions(bonds);
        if (liveSiteRank.size() == nrQubits)
            dummySim->SetCurrentSiteRanks(liveSiteRank);
        dummyHistoryValid = true;
    }

    void SaveDummyState()
    {
        savedDummyBondDims = CaptureDummyHistory();
        savedSiteRank = liveSiteRank;
        hasSavedSiteRank = true;
    }

    void RestoreDummyState()
    {
        if (hasSavedSiteRank)
            liveSiteRank = savedSiteRank;
        RestoreDummyHistory(savedDummyBondDims);
    }

    void ResetDummySimulator()
    {
        ResetLiveSiteRanks();
        InvalidateDummyHistory();
        if (!dummySim || !trackVirtualBonds)
            return;
        const auto map = CurrentRoutingMap();
        if (map.size() != nrQubits)
            return;
        dummySim->SetRoutingMap(map);
        dummySim->setTotalSwappingCost(0.);
        dummySim->ResetBondDimensions();
        dummyHistoryValid = true;
    }

    void ClearDummySimulator()
    {
        dummySim.reset();
        savedDummyBondDims.clear();
        liveSiteRank.clear();
        savedSiteRank.clear();
        hasSavedSiteRank = false;
        upcomingHasDisentangle = false;
        InvalidateDummyHistory();
    }

    // Decompositions must advance the public circuit counter once. Their
    // primitive gates use native routing because the lookahead entry describes
    // the enclosing three-qubit operation, not a particular primitive.
    class DecompositionScope
    {
      public:
        explicit DecompositionScope(TensorChainRouting &owner) : owner(owner), notificationsEnabled(owner.NotificationsEnabled())
        {
            ++owner.decompositionDepth;
            owner.DontNotify();
        }

        ~DecompositionScope()
        {
            if (notificationsEnabled)
                owner.Notify();
            else
                owner.DontNotify();
            --owner.decompositionDepth;
        }

      private:
        TensorChainRouting &owner;
        const bool notificationsEnabled;
    };

    int lookaheadDepth = 0, lookaheadDepthWithHeuristic = 0;
    bool useOptimalMeetingPosition = true;
    double growthFactorSwap = 1., growthFactorGate = 0.7;
    std::unique_ptr<MPSDummySimulator> dummySim;
    std::vector<double> savedDummyBondDims;
    std::vector<double> liveSiteRank;
    std::vector<double> savedSiteRank;
    bool hasSavedSiteRank = false;
    bool upcomingHasDisentangle = false;
    bool dummyHistoryValid = false;
    size_t curMaxBondDim = 0;
    MPSDummySimulator::IndexType physicalBondCap = 0;
    bool explicitVirtualBondsEnabled = false;
    bool virtualBondsEnabled = false, trackVirtualBonds = false;
    double truncationWeight = 1.;
    unsigned decompositionDepth = 0;
};
} // namespace Simulators::Private
