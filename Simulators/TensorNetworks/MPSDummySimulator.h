/**
 * @file MPSDummySimulator.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The MPS dummy simulator class.
 *
 * The purpose of this class is to be able to follow the internal mapping of the
 * qubits and the operations that are applied to them, without actually
 * simulating anything.
 * The swapping cost of the MPS simulator is not important, this will be used to
 * evaluate it.
 */

#pragma once

#ifndef _MPSDUMMYSIMULATOR_H_
#define _MPSDUMMYSIMULATOR_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <iostream>
#include <numeric>
#include <random>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "MPSSimulatorInterface.h"
#include "QuantumGate.h"
#include "RoutingConfig.h"
#include "RoutingGateInfo.h"

namespace Simulators
{

class MPSDummySimulator
{
  public:
    using IndexType = long long int;
    using MatrixClass = QC::TensorNetworks::MPSSimulatorInterface::MatrixClass;
    using GateClass = QC::TensorNetworks::MPSSimulatorInterface::GateClass;

    // operatorChain selects the MPO model: every site carries an operator
    // (ket and bra legs, physical dimension 4) instead of a qubit state.
    explicit MPSDummySimulator(size_t N, bool operatorChain = false) : nrQubits(N), maxVirtualExtent(0)
    {
        physExtent = operatorChain ? 4 : 2;
        // MPS: the cost model overrates scrambled layouts for circuits such as
        // QFT, so only a large predicted gain moves away from the identity. MPO
        // QFT measures the other way (the searched layouts are really better at
        // predicted gains of 2-4), so an MPO accepts any improvement.
        minLayoutGain = operatorChain ? 1. : 4.;
        InitQubitsMap();

        SetMaxBondDimension(0);
    }

    std::unique_ptr<MPSDummySimulator> Clone() const
    {
        auto clone = std::unique_ptr<MPSDummySimulator>(new MPSDummySimulator(nrQubits, LightweightInitTag{}));
        clone->qubitsMap = qubitsMap;
        clone->qubitsMapInv = qubitsMapInv;
        clone->physExtent = physExtent;
        clone->maxVirtualExtent = maxVirtualExtent;
        clone->runtimeBondCap = runtimeBondCap;
        clone->truncationWeight = truncationWeight;
        clone->swapGatesRelabelQubits = swapGatesRelabelQubits;
        clone->bondCost = bondCost;
        clone->maxBondDim = maxBondDim;
        clone->currentBondDim = currentBondDim;

        clone->totalSwappingCost = totalSwappingCost;

        clone->growthFactorSwap = growthFactorSwap;
        clone->growthFactorGate = growthFactorGate;
        clone->minLayoutGain = minLayoutGain;

        return clone;
    }

    size_t getNrQubits() const
    {
        return nrQubits;
    }

    bool IsOperatorChain() const
    {
        return physExtent == 4;
    }

    void SetSwapGatesRelabelQubits(bool enabled)
    {
        if (swapGatesRelabelQubits == enabled)
            return;
        swapGatesRelabelQubits = enabled;
        InvalidateMemoCache();
    }

    // Logical SWAP: exchange labels, keeping tensor sites, bonds and cost intact.
    // SwapQubits/SwapQubitsToPosition model physical routing and remain separate.
    void RelabelQubits(IndexType q1, IndexType q2)
    {
        if (q1 < 0 || q2 < 0 || q1 >= static_cast<IndexType>(nrQubits) || q2 >= static_cast<IndexType>(nrQubits))
            throw std::invalid_argument("Qubit index out of bounds");
        std::swap(qubitsMap[q1], qubitsMap[q2]);
        qubitsMapInv[qubitsMap[q1]] = q1;
        qubitsMapInv[qubitsMap[q2]] = q2;
    }

    double getGrowthFactorSwap() const
    {
        return growthFactorSwap;
    }

    double getGrowthFactorGate() const
    {
        return growthFactorGate;
    }

    static constexpr double kPostSwapCostWeight = 0.25;
    static constexpr double kPostGateCostWeight = 0.5;

    // Allow two extra rank doublings above the physical cap. This bounded
    // horizon distinguishes saturated cuts without estimates growing with
    // register size. Never exceed the exact central-cut rank (4 per MPO site).
    static IndexType EffectiveLookaheadMaxBondDimension(size_t nrQubits, bool operatorChain, IndexType cap)
    {
        if (cap <= 0)
            return cap;
        const auto limit = std::numeric_limits<IndexType>::max();
        const auto headroom = cap > limit / 4 ? limit : 4 * cap;
        const size_t half = nrQubits / 2;
        const size_t safeExponent = operatorChain ? 31 : 62;
        const auto physical = half > safeExponent ? limit : static_cast<IndexType>(1ULL << (half * (operatorChain ? 2 : 1)));
        return std::max(cap, std::min(headroom, physical));
    }

    static IndexType EffectiveMapMaxBondDimension(size_t nrQubits, bool operatorChain, IndexType cap)
    {
        return EffectiveLookaheadMaxBondDimension(nrQubits, operatorChain, cap);
    }

    static bool IsControlledTwoQubitGateType(Circuits::QuantumGateType type)
    {
        return IsControlledRoutingGate(type);
    }

    // A capped runtime proxy plus a separately weighted overflow penalty.
    // Weight 1 adds one cap-cubed unit for each cap-sized overflow; weight 0
    // charges runtime alone. Standalone planners retain the old cubic default.
    double BondCost(double dimension) const
    {
        const double runtimeDim = runtimeBondCap > 0 ? std::min(dimension, static_cast<double>(runtimeBondCap)) : dimension;
        const double runtime = runtimeDim * runtimeDim * runtimeDim;
        const double overflow = runtimeBondCap > 0 ? std::max(0., dimension / runtimeBondCap - 1.) : 0.;
        return runtime + truncationWeight * runtime * overflow;
    }

    void SetRoutingCostModel(IndexType cap, double weight)
    {
        if (cap < 0 || !std::isfinite(weight) || weight < 0.)
            throw std::invalid_argument("Invalid routing cost model");
        if (runtimeBondCap == cap && truncationWeight == weight)
            return;
        runtimeBondCap = cap;
        truncationWeight = weight;
        for (size_t i = 0; i < bondCost.size(); ++i)
            bondCost[i] = BondCost(currentBondDim[i]);
        InvalidateMemoCache();
    }

    void InvalidateMemoCache()
    {
        for (auto &entry : memoCache)
            entry.valid = false;
    }

    void setGrowthFactorSwap(double factor)
    {
        if (growthFactorSwap == factor)
            return;
        growthFactorSwap = factor;
        InvalidateMemoCache();
    }

    void setGrowthFactorGate(double factor)
    {
        if (growthFactorGate == factor)
            return;
        growthFactorGate = factor;
        InvalidateMemoCache();
    }

    // ComputeOptimalQubitsMap keeps the identity layout unless another layout
    // is estimated to cost at least this many times less (1: any improvement).
    // Defaults: 4 for an MPS, 1 for an MPO.
    double getMinLayoutGain() const
    {
        return minLayoutGain;
    }

    void setMinLayoutGain(double gain)
    {
        if (!(gain >= 1.))
            throw std::invalid_argument("Minimum layout gain must be at least 1");
        minLayoutGain = gain;
    }

    void Clear()
    {
        InitQubitsMap();
        InvalidateMemoCache();
    }

    void ResetBondDimensions()
    {
        if (nrQubits == 0)
        {
            currentBondDim.clear();
            bondCost.clear();
            return;
        }
        currentBondDim.assign(nrQubits - 1, 1.0);
        bondCost.assign(nrQubits - 1, 1.0);
    }

    void SetMaxBondDimension(IndexType val)
    {
        if (nrQubits == 0)
        {
            if (maxVirtualExtent != val)
            {
                maxVirtualExtent = val;
                InvalidateMemoCache();
            }
            return;
        }

        const double untruncatedMaxExtent = std::pow(physExtent, nrQubits / 2);
        IndexType maxVirtualExtentLimit = static_cast<IndexType>(untruncatedMaxExtent);

        if (untruncatedMaxExtent >= std::numeric_limits<IndexType>::max() || std::isnan(untruncatedMaxExtent) || std::isinf(untruncatedMaxExtent))
            maxVirtualExtentLimit = std::numeric_limits<IndexType>::max() - 1;
        else if (untruncatedMaxExtent < 2)
            maxVirtualExtentLimit = 2;

        const IndexType resolvedMaxVirtualExtent = val == 0 ? maxVirtualExtentLimit : std::min(val, maxVirtualExtentLimit);
        const bool alreadyInitialized = (maxBondDim.size() + 1 == nrQubits && currentBondDim.size() + 1 == nrQubits && bondCost.size() + 1 == nrQubits);
        if (alreadyInitialized && maxVirtualExtent == resolvedMaxVirtualExtent)
            return;

        maxVirtualExtent = resolvedMaxVirtualExtent;
        InvalidateMemoCache();

        maxBondDim.resize(nrQubits - 1);
        if (!alreadyInitialized)
        {
            currentBondDim.assign(nrQubits - 1, 1.0);
            bondCost.assign(nrQubits - 1, 1.0);
        }

        // the checks here are overkill, but better safe than sorry
        // we're dealing with large values here, overflows are to be expected
        for (IndexType i = 0; i < static_cast<IndexType>(nrQubits); ++i)
        {
            double maxExtent1 = std::pow((double)physExtent, (double)i + 1.);
            double maxExtent2 = std::pow((double)physExtent, (double)nrQubits - i - 1.);

            if (maxExtent1 >= (double)std::numeric_limits<size_t>::max() || std::isnan(maxExtent1) || std::isinf(maxExtent1))
                maxExtent1 = (double)std::numeric_limits<size_t>::max() - 1;
            else if (maxExtent1 < 1)
                maxExtent1 = 1;

            if (maxExtent2 > (double)std::numeric_limits<size_t>::max() || std::isnan(maxExtent2) || std::isinf(maxExtent2))
                maxExtent2 = (double)std::numeric_limits<size_t>::max() - 1;
            else if (maxExtent2 < 1)
                maxExtent2 = 1;

            size_t maxRightExtent = (size_t)std::min<double>({maxExtent1, maxExtent2, (double)maxVirtualExtent});
            if (maxRightExtent < 2)
                maxRightExtent = 2;

            if (i < static_cast<IndexType>(nrQubits) - 1)
            {
                maxBondDim[i] = static_cast<double>(maxRightExtent);
                if (alreadyInitialized)
                {
                    currentBondDim[i] = std::clamp(currentBondDim[i], 1.0, maxBondDim[i]);
                    bondCost[i] = BondCost(currentBondDim[i]);
                }
            }
        }
    }

    void print() const
    {
        std::cout << "Qubits map: ";
        for (int q = 0; q < static_cast<int>(qubitsMap.size()); ++q)
            std::cout << q << "->" << qubitsMap[q] << " ";
        std::cout << std::endl;
    }

    void ApplyGate(const QC::Gates::AppliedGate<MatrixClass> &gate)
    {
        ApplyGate(gate, gate.getQubit1(), gate.getQubit2());
    }

    void ApplyGate(const std::shared_ptr<Circuits::IOperation<>> &gate)
    {
        if (gate->GetType() != Circuits::OperationType::kGate)
            return;
        const auto qbits = gate->AffectedQubits();

        if (qbits.size() == 1)
        {
            // 1-qubit gates are no-ops in the dummy simulator (no swap logic)
            return;
        }
        else if (qbits.size() == 2)
        {
            static const QC::Gates::SwapGate<MatrixClass> dummySwapGate;

            const auto gateptr = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(gate);
            const auto info = std::dynamic_pointer_cast<RoutingGateInfo>(gate);
            if (info)
                ApplyTwoQubitFastGate(qbits[0], qbits[1], info->isSwap, info->schmidtRank);
            else if (gateptr && gateptr->GetGateType() == Circuits::QuantumGateType::kSwapGateType)
                ApplyGate(dummySwapGate, qbits[0], qbits[1]);
            else
                ApplyTwoQubitGate(qbits[0], qbits[1], gateptr ? GateSchmidtRank(*gateptr) : 4.);
        }
        else if (qbits.size() == 3 && gate->GetType() == Circuits::OperationType::kGate)
        {
            static const QC::Gates::CNOTGate<MatrixClass> dummyControlled2qGate;

            const auto gateptr = std::static_pointer_cast<Circuits::IQuantumGate<>>(gate);

            const size_t q1 = qbits[0]; // control 1
            const size_t q2 = qbits[1]; // control 2
            const size_t q3 = qbits[2]; // target
            if (gateptr->GetGateType() == Circuits::QuantumGateType::kCCXGateType)
            {
                ApplyGate(dummyControlled2qGate, q3, q2);
                ApplyGate(dummyControlled2qGate, q2, q1);
                ApplyGate(dummyControlled2qGate, q3, q2);
                ApplyGate(dummyControlled2qGate, q2, q1);
                ApplyGate(dummyControlled2qGate, q3, q1);
            }
            else
            { // cswap
                ApplyGate(dummyControlled2qGate, q2, q3);
                ApplyGate(dummyControlled2qGate, q3, q2);
                ApplyGate(dummyControlled2qGate, q2, q1);
                ApplyGate(dummyControlled2qGate, q3, q2);
                ApplyGate(dummyControlled2qGate, q2, q1);
                ApplyGate(dummyControlled2qGate, q3, q1);
                ApplyGate(dummyControlled2qGate, q2, q3);
            }
        }
    }

    void ApplyGate(const GateClass &gate, IndexType qubit, IndexType controllingQubit1 = 0)
    {
        if (qubit < 0 || qubit >= static_cast<IndexType>(getNrQubits()))
            throw std::invalid_argument("Qubit index out of bounds");
        else if (controllingQubit1 < 0 || controllingQubit1 >= static_cast<IndexType>(getNrQubits()))
            throw std::invalid_argument("Qubit index out of bounds");

        // for two qubit gates:
        // if the qubits are not adjacent, apply swap gates until they are
        // don't forget to update the qubitsMap
        if (gate.getQubitsNumber() > 1)
        {
            if (gate.getQubitsNumber() == 2 && gate.isSwapGate() && swapGatesRelabelQubits)
                return RelabelQubits(qubit, controllingQubit1);
            // QCSim gate objects carry no angle: controlled ones count as rank 2,
            // the others as 4.
            ApplyTwoQubitGate(qubit, controllingQubit1, gate.isControlled() ? 2. : 4., gate.isSwapGate());
        }
    }

    // Effective Schmidt rank of a named or generic two-qubit gate.
    static double GateSchmidtRank(const Circuits::IQuantumGate<> &gate)
    {
        if (gate.GetNumQubits() != 2)
            return 4.;
        const auto params = gate.GetParams();
        return RoutingSchmidtRank(gate.GetGateType(), params.empty() ? 0. : params[0], [&gate] { return gate.GetMatrix(); });
    }

    // Route two qubits together if needed, then grow the bond between them.
    void ApplyTwoQubitGate(IndexType qubit, IndexType controllingQubit1, double schmidtRank, bool isSwapGate = false)
    {
        if (qubit < 0 || qubit >= static_cast<IndexType>(getNrQubits()))
            throw std::invalid_argument("Qubit index out of bounds");
        else if (controllingQubit1 < 0 || controllingQubit1 >= static_cast<IndexType>(getNrQubits()))
            throw std::invalid_argument("Qubit index out of bounds");
        IndexType qubit1 = qubitsMap[qubit];
        IndexType qubit2 = qubitsMap[controllingQubit1];

        if (abs(qubit1 - qubit2) > 1)
        {
            SwapQubits(qubit, controllingQubit1);
            qubit1 = qubitsMap[qubit];
            qubit2 = qubitsMap[controllingQubit1];
            assert(abs(qubit1 - qubit2) == 1);
        }

        // any 2-qubit gate (whether swaps were needed or not) grows the
        // bond dimension at the bond between the two adjacent qubits.
        // Charge both the pre-gate contraction cost and a post-growth SVD/reconstruction
        // cost so immediate entanglement growth and gate contraction overhead at this
        // bond are reflected even near the end of the lookahead horizon.
        // The growth factor is the gate's effective Schmidt rank
        // (EffectiveSchmidtRank): 2 for CX/CZ, 4 for SWAP, near 1 for small angles.
        RecordAdjacentTwoQubitGate(std::min(qubit1, qubit2), isSwapGate, schmidtRank);
    }

    void RecordAdjacentTwoQubitGate(IndexType bond, bool isSwapGate, double schmidtRank)
    {
        if (bond < 0 || bond >= static_cast<IndexType>(currentBondDim.size()))
            return;
        totalSwappingCost += bondCost[bond];
        growBondDimension(bond, isSwapGate, schmidtRank);
        totalSwappingCost += kPostGateCostWeight * bondCost[bond];
    }

    void ApplyGates(const std::vector<QC::Gates::AppliedGate<MatrixClass>> &gates)
    {
        for (const auto &gate : gates)
            ApplyGate(gate);
    }

    void ApplyGates(const std::vector<std::shared_ptr<Circuits::IOperation<>>> &gates)
    {
        for (const auto &gate : gates)
            ApplyGate(gate);
    }

    void SetRoutingMap(const std::vector<long long int> &actualMap)
    {
        qubitsMap = actualMap;
        for (size_t i = 0; i < actualMap.size(); ++i)
            qubitsMapInv[actualMap[i]] = i;
    }

    void SetInitialQubitsMap(const std::vector<long long int> &initialMap)
    {
        SetRoutingMap(initialMap);
        totalSwappingCost = 0;
        ResetBondDimensions();
    }

    void SetCurrentBondDimensions(const std::vector<double> &dims)
    {
        assert(dims.size() == currentBondDim.size());
        for (size_t i = 0; i < dims.size(); ++i)
            currentBondDim[i] = std::min(dims[i], maxBondDim[i]);

        // update bondCost as well, since it depends on the current bond dimensions
        for (size_t i = 0; i < currentBondDim.size(); ++i)
            bondCost[i] = BondCost(currentBondDim[i]);
    }

    const std::vector<double> &getCurrentBondDimensions() const
    {
        return currentBondDim;
    }

    const std::vector<double> &getMaxBondDimensions() const
    {
        return maxBondDim;
    }

    const std::vector<IndexType> &getQubitsMap() const
    {
        return qubitsMap;
    }

    const std::vector<IndexType> &getQubitsMapInv() const
    {
        return qubitsMapInv;
    }

    void setTotalSwappingCost(double cost)
    {
        totalSwappingCost = cost;
    }

    double getTotalSwappingCost() const
    {
        return totalSwappingCost;
    }

    enum class HeuristicPolicy
    {
        kCombinedCostAndGravity = 0,
        kMinOneStepCost,
        kNextPartnerGravity,
        kAnchorHeavierQubit,
        kMinBondCost,
        kSwapLeft,
        kSwapRight,
        kCloserToEnd,
        kCloserToCenter
    };

    // Evaluate the total cost of moving the current gate's qubits to meetPosition,
    // applying the current 2-qubit gate, and then simulating the next
    // lookaheadDepth 2-qubit gates from upcomingGates.
    void EvaluateMeetingPositionCost(IndexType meetPosition, const std::vector<std::shared_ptr<Circuits::IOperation<>>> &upcomingGates,
                                     long long int currentGateIndex, int lookaheadDepth, int lookaheadDepthWithHeuristic, double currentCost, double &bestCost,
                                     bool useSameDummy = false)
    {
        const size_t maxNeeded = static_cast<size_t>(std::max(0, lookaheadDepth)) + 10;
        std::array<FastGateOp, 128> stackWindow;
        std::vector<FastGateOp> heapWindow;
        FastGateOp *windowPtr = stackWindow.data();
        if (maxNeeded > stackWindow.size())
        {
            heapWindow.resize(maxNeeded);
            windowPtr = heapWindow.data();
        }
        const size_t windowSize = ExtractFastGateWindow(upcomingGates, currentGateIndex, maxNeeded, windowPtr);
        if (windowSize == 0)
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return;
        }

        EvaluateMeetingPositionCostFast(meetPosition, windowPtr, windowSize, 0, lookaheadDepth, lookaheadDepthWithHeuristic, currentCost, bestCost,
                                        useSameDummy);
    }

    // Find the meeting position that minimizes the estimated cost of the
    // current swap + lookahead gates using exact branching for the first
    // (lookaheadDepth - lookaheadDepthWithHeuristic) steps and a multi-policy
    // rollout portfolio for the remaining heuristic horizon.
    // bestCost receives the total accumulated cost for the chosen position.
    IndexType FindBestMeetingPosition(const std::vector<std::shared_ptr<Circuits::IOperation<>>> &upcomingGates, long long int currentGateIndex,
                                      int lookaheadDepth, int lookaheadDepthWithHeuristic, double currentCost, double &bestCost)
    {
        if (currentGateIndex < 0 || currentGateIndex >= static_cast<long long int>(upcomingGates.size()))
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return -1;
        }

        const size_t maxNeeded = static_cast<size_t>(std::max(0, lookaheadDepth)) + 10;
        std::array<FastGateOp, 128> stackWindow;
        std::vector<FastGateOp> heapWindow;
        FastGateOp *windowPtr = stackWindow.data();
        if (maxNeeded > stackWindow.size())
        {
            heapWindow.resize(maxNeeded);
            windowPtr = heapWindow.data();
        }
        const size_t windowSize = ExtractFastGateWindow(upcomingGates, currentGateIndex, maxNeeded, windowPtr);
        if (windowSize == 0)
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return -1;
        }

        uint64_t stateHash = 14695981039346656037ULL;
        auto mixHash = [&](uint64_t v) {
            stateHash ^= v;
            stateHash *= 1099511628211ULL;
        };
        auto mixDouble = [&](double d) {
            uint64_t bits = 0;
            std::memcpy(&bits, &d, sizeof(double));
            mixHash(bits);
        };

        mixHash(static_cast<uint64_t>(windowSize));
        mixHash(static_cast<uint64_t>(lookaheadDepth));
        mixHash(static_cast<uint64_t>(lookaheadDepthWithHeuristic));
        mixHash(static_cast<uint64_t>(physExtent));
        mixHash(static_cast<uint64_t>(maxVirtualExtent));
        mixHash(static_cast<uint64_t>(swapGatesRelabelQubits));
        mixDouble(growthFactorSwap);
        mixDouble(growthFactorGate);
        for (IndexType q : qubitsMap)
            mixHash(static_cast<uint64_t>(q));
        for (double d : currentBondDim)
            mixDouble(d);
        for (double c : bondCost)
            mixDouble(c);
        for (double m : maxBondDim)
            mixDouble(m);
        for (size_t w = 0; w < windowSize; ++w)
        {
            const FastGateOp &fg = windowPtr[w];
            mixHash(static_cast<uint64_t>(fg.q1));
            mixHash(static_cast<uint64_t>(fg.q2));
            mixHash(static_cast<uint64_t>(fg.q3));
            mixHash(static_cast<uint64_t>(fg.numQubits));
            mixHash(static_cast<uint64_t>(fg.isSwap ? 1 : 0));
            mixDouble(fg.schmidtRank);
        }

        MemoEntry &slot = memoCache[(static_cast<size_t>(currentGateIndex) ^ static_cast<size_t>(stateHash)) & (kMemoCacheSize - 1)];
        if (slot.valid && slot.hash == stateHash)
        {
            const double candCost = currentCost + slot.costDelta;
            if (candCost < bestCost)
                bestCost = candCost;
            return slot.bestPos;
        }

        double localBestCost = std::numeric_limits<double>::infinity();
        const IndexType bestPos = FindBestMeetingPositionFast(windowPtr, windowSize, 0, lookaheadDepth, lookaheadDepthWithHeuristic, 0.0, localBestCost);

        slot.valid = true;
        slot.hash = stateHash;
        slot.bestPos = bestPos;
        slot.costDelta = localBestCost;

        if (currentCost + localBestCost < bestCost)
            bestCost = currentCost + localBestCost;

        return bestPos;
    }

    std::vector<long long int> ComputeOptimalQubitsMap(const std::vector<std::shared_ptr<Circuits::Circuit<>>> &layers, int nrShuffles = 0 /*25*/,
                                                       int nrSwaps = 0 /*10*/)
    {
        const IndexType nrQubits = getNrQubits();

        if (layers.empty() || nrQubits <= 2)
            return qubitsMap;

        auto evaluateCost = [&, this](const std::vector<IndexType> &candidateMap) -> double {
            auto saveQubitsMap = qubitsMap;
            auto saveQubitsMapInv = qubitsMapInv;
            auto saveCurrentBondDim = currentBondDim;
            auto saveTotalSwappingCost = totalSwappingCost;
            auto saveBondCost = bondCost;

            SetInitialQubitsMap(candidateMap);

            for (const auto &layer : layers)
                ApplyGates(layer->GetOperations());
            auto cost = getTotalSwappingCost();
            qubitsMap = std::move(saveQubitsMap);
            qubitsMapInv = std::move(saveQubitsMapInv);
            currentBondDim = std::move(saveCurrentBondDim);
            totalSwappingCost = saveTotalSwappingCost;
            bondCost = std::move(saveBondCost);

            return cost;
        };

        auto evaluateCostBounded = [&, this](const std::vector<IndexType> &candidateMap, double bound) -> double {
            auto saveQubitsMap = qubitsMap;
            auto saveQubitsMapInv = qubitsMapInv;
            auto saveCurrentBondDim = currentBondDim;
            auto saveTotalSwappingCost = totalSwappingCost;
            auto saveBondCost = bondCost;

            SetInitialQubitsMap(candidateMap);

            for (const auto &layer : layers)
            {
                ApplyGates(layer->GetOperations());
                auto cost = getTotalSwappingCost();
                if (cost >= bound)
                {
                    // restore state
                    qubitsMap = std::move(saveQubitsMap);
                    qubitsMapInv = std::move(saveQubitsMapInv);
                    currentBondDim = std::move(saveCurrentBondDim);
                    totalSwappingCost = saveTotalSwappingCost;
                    bondCost = std::move(saveBondCost);
                    return cost;
                }
            }
            auto cost = getTotalSwappingCost();
            // restore state
            qubitsMap = std::move(saveQubitsMap);
            qubitsMapInv = std::move(saveQubitsMapInv);
            currentBondDim = std::move(saveCurrentBondDim);
            totalSwappingCost = saveTotalSwappingCost;
            bondCost = std::move(saveBondCost);

            return cost;
        };

        // Collect 2-qubit pairs from each layer, preserving layer boundaries
        struct QubitPair
        {
            IndexType q1, q2;
        };

        std::vector<std::vector<QubitPair>> layerPairs;
        std::vector<IndexType> initialLabels(nrQubits);
        std::iota(initialLabels.begin(), initialLabels.end(), 0);

        for (size_t li = 0; li < layers.size(); ++li)
        {
            const auto &layer = layers[li];
            std::vector<QubitPair> lp;
            for (const auto &op : layer->GetOperations())
            {
                auto qbits = op->AffectedQubits();
                if (qbits.size() >= 2)
                {
                    if (swapGatesRelabelQubits)
                    {
                        const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(op);
                        const auto info = std::dynamic_pointer_cast<RoutingGateInfo>(op);
                        if (qbits.size() == 2 && ((info && info->isSwap) || (gate && gate->GetGateType() == Circuits::QuantumGateType::kSwapGateType)))
                        {
                            std::swap(initialLabels[qbits[0]], initialLabels[qbits[1]]);
                            continue;
                        }
                        // Express later partners in the labels of the initial
                        // state; a free SWAP itself creates no layout edge.
                        for (auto &q : qbits)
                            q = initialLabels[q];
                    }
                    if (qbits[0] > qbits[1])
                        std::swap(qbits[0], qbits[1]);

                    lp.push_back({static_cast<IndexType>(qbits[0]), static_cast<IndexType>(qbits[1])});
                }
            }
            if (!lp.empty())
                layerPairs.push_back(std::move(lp));
        }
        if (layerPairs.empty())
            return qubitsMap;

        // Build a linear qubit chain from ordered pairs using a group-merging
        // strategy.  Pairs are processed in order (earlier = more important).
        // - If neither qubit is placed, create a new group [q1, q2].
        // - If one qubit is already in a group, add the other at the group
        //   end (front or back) that minimizes distance to the existing one.
        // - If both are in different groups, merge as [gA][gB] or [gB][gA],
        //   whichever places q1 and q2 closer together.
        // - If both are already in the same group, do nothing.
        auto buildChain = [&](const std::vector<std::vector<QubitPair>> &orderedLP) -> std::vector<long long int> {
            std::vector<std::deque<IndexType>> groups;
            std::vector<int> qubitGroup(nrQubits, -1);
            std::unordered_set<IndexType> placedQubits;

            for (const auto &lp : orderedLP)
            {
                for (const auto &p : lp)
                {
                    const int g1 = qubitGroup[p.q1];
                    const int g2 = qubitGroup[p.q2];

                    if (g1 < 0 && g2 < 0)
                    {
                        // Neither placed: create a new group
                        const int gIdx = static_cast<int>(groups.size());
                        groups.push_back({p.q1, p.q2});
                        qubitGroup[p.q1] = gIdx;
                        qubitGroup[p.q2] = gIdx;
                        placedQubits.insert(p.q1);
                        placedQubits.insert(p.q2);
                    }
                    else if (g1 >= 0 && g2 < 0)
                    {
                        // q1 placed, q2 not: add q2 at the closer end of q1's group
                        auto &grp = groups[g1];
                        size_t idx = 0;
                        for (size_t i = 0; i < grp.size(); ++i)
                            if (grp[i] == p.q1)
                            {
                                idx = i;
                                break;
                            }
                        // front distance: idx + 1, back distance: grp.size() - idx
                        if (idx + 1 <= grp.size() - idx)
                            grp.push_front(p.q2);
                        else
                            grp.push_back(p.q2);
                        qubitGroup[p.q2] = g1;
                        placedQubits.insert(p.q2);
                    }
                    else if (g1 < 0 && g2 >= 0)
                    {
                        // q2 placed, q1 not: add q1 at the closer end of q2's group
                        auto &grp = groups[g2];
                        size_t idx = 0;
                        for (size_t i = 0; i < grp.size(); ++i)
                            if (grp[i] == p.q2)
                            {
                                idx = i;
                                break;
                            }
                        if (idx + 1 <= grp.size() - idx)
                            grp.push_front(p.q1);
                        else
                            grp.push_back(p.q1);
                        qubitGroup[p.q1] = g2;
                        placedQubits.insert(p.q1);
                    }
                    else if (g1 != g2)
                    {
                        // Both in different groups: merge to minimize distance
                        auto &grpA = groups[g1];
                        auto &grpB = groups[g2];
                        size_t idxA = 0, idxB = 0;
                        for (size_t i = 0; i < grpA.size(); ++i)
                            if (grpA[i] == p.q1)
                            {
                                idxA = i;
                                break;
                            }
                        for (size_t i = 0; i < grpB.size(); ++i)
                            if (grpB[i] == p.q2)
                            {
                                idxB = i;
                                break;
                            }
                        // [A][B]: q1 at idxA, q2 at |A|+idxB -> dist = |A|+idxB-idxA
                        // [B][A]: q2 at idxB, q1 at |B|+idxA -> dist = |B|+idxA-idxB
                        const size_t distAB = grpA.size() + idxB - idxA;
                        const size_t distBA = grpB.size() + idxA - idxB;

                        int mergedGroup;
                        if (distAB <= distBA)
                        {
                            for (const auto q : grpB)
                                grpA.push_back(q);
                            grpB.clear();
                            mergedGroup = g1;
                        }
                        else
                        {
                            for (const auto q : grpA)
                                grpB.push_back(q);
                            grpA.clear();
                            mergedGroup = g2;
                        }
                        for (const auto q : groups[mergedGroup])
                            qubitGroup[q] = mergedGroup;
                    }
                    // Both in the same group: do nothing

                    if (placedQubits.size() == static_cast<size_t>(nrQubits))
                        break; // all qubits placed, can stop processing pairs
                }

                if (placedQubits.size() == static_cast<size_t>(nrQubits))
                    break; // all qubits placed, can stop processing pairs
            }

            // Concatenate all non-empty groups
            std::vector<IndexType> chain;
            chain.reserve(nrQubits);
            for (const auto &grp : groups)
                for (const auto q : grp)
                    chain.push_back(q);

            // Append any remaining unplaced qubits
            for (IndexType q = 0; q < nrQubits; ++q)
                if (qubitGroup[q] < 0)
                    chain.push_back(q);

            assert(chain.size() == static_cast<size_t>(nrQubits));

            // Convert chain to qubitsMap: chain[physPos] = logicalQubit
            std::vector<long long int> result(nrQubits);
            for (size_t i = 0; i < chain.size(); ++i)
                result[chain[i]] = static_cast<long long int>(i);

            return result;
        };

        // Build a time-decayed full-circuit interaction graph across all layers
        // so deeper gates and multi-layer topology (ladders, 2D grids, islands)
        // are visible even after all qubits have appeared once. Uses sparse
        // adjacency lists O(|E|) instead of an O(N^2) dense matrix.
        struct WeightedEdge
        {
            IndexType u;
            IndexType v;
            double w;
        };

        const size_t nQ = static_cast<size_t>(nrQubits);
        std::vector<std::vector<std::pair<IndexType, double>>> adjList(nQ);
        auto addEdgeWeight = [&](IndexType u, IndexType v, double w) {
            auto &listU = adjList[static_cast<size_t>(u)];
            for (auto &nb : listU)
            {
                if (nb.first == v)
                {
                    nb.second += w;
                    return;
                }
            }
            listU.push_back({v, w});
        };

        const double decay = std::exp(-2.5 / static_cast<double>(std::max<size_t>(1, layerPairs.size())));
        double layerWeight = 1.0;
        for (const auto &lp : layerPairs)
        {
            for (const auto &p : lp)
            {
                if (p.q1 != p.q2 && p.q1 >= 0 && p.q1 < nrQubits && p.q2 >= 0 && p.q2 < nrQubits)
                {
                    addEdgeWeight(p.q1, p.q2, layerWeight);
                    addEdgeWeight(p.q2, p.q1, layerWeight);
                }
            }
            layerWeight *= decay;
        }

        std::vector<WeightedEdge> edges;
        for (IndexType u = 0; u < nrQubits; ++u)
        {
            auto &listU = adjList[static_cast<size_t>(u)];
            std::sort(listU.begin(), listU.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            for (const auto &nb : listU)
            {
                if (u < nb.first && nb.second > 0.0)
                    edges.push_back({u, nb.first, nb.second});
            }
        }
        std::sort(edges.begin(), edges.end(), [](const WeightedEdge &a, const WeightedEdge &b) {
            if (a.w != b.w)
                return a.w > b.w;
            return std::tie(a.u, a.v) < std::tie(b.u, b.v);
        });

        auto getEdgeWeight = [&](IndexType u, IndexType v) -> double {
            if (u == v)
                return 0.0;
            const auto &list = (adjList[u].size() <= adjList[v].size()) ? adjList[u] : adjList[v];
            const IndexType target = (adjList[u].size() <= adjList[v].size()) ? v : u;
            auto it = std::lower_bound(list.begin(), list.end(), target, [](const std::pair<IndexType, double> &elem, IndexType val) {
                return elem.first < val;
            });
            return (it != list.end() && it->first == target) ? it->second : 0.0;
        };

        // Superlinear (convex) swap distance penalty: penalizes long-range spans
        // more than a pair of medium spans so ladders and 2D grids interleave cleanly.
        auto pairSwapPenalty = [](IndexType posU, IndexType posV) -> double {
            const double swaps = static_cast<double>(std::max<IndexType>(0, std::abs(posU - posV) - 1));
            return swaps * (1.0 + 0.25 * std::sqrt(swaps));
        };

        auto qubitGraphCost = [&](IndexType u, IndexType posU, const std::vector<IndexType> &pos) -> double {
            double c = 0.0;
            for (const auto &nb : adjList[u])
                c += nb.second * pairSwapPenalty(posU, pos[nb.first]);
            return c;
        };

        auto totalGraphCost = [&](const std::vector<IndexType> &pos) -> double {
            double c = 0.0;
            for (const auto &e : edges)
                c += e.w * pairSwapPenalty(pos[e.u], pos[e.v]);
            return c;
        };

        // Bounded O(|E| * W) 2-opt swap + incremental adjacent-slide insertion
        // local search on the convex interaction graph cost.
        auto refineGraphLayout = [&](std::vector<IndexType> pos) -> std::vector<IndexType> {
            if (edges.empty() || nrQubits <= 2)
                return pos;

            // Early exit when all interacting pairs are already adjacent.
            if (totalGraphCost(pos) <= 1e-12)
                return pos;

            std::vector<IndexType> order(nQ);
            for (IndexType q = 0; q < nrQubits; ++q)
                order[pos[q]] = q;

            const IndexType maxWindow = std::clamp<IndexType>(static_cast<IndexType>(512 / std::max<size_t>(1, nQ / 16)), 8, 48);
            const int maxIter = (nQ > 64) ? 5 : 12;
            std::vector<IndexType> candPositions;
            candPositions.reserve(static_cast<size_t>(maxWindow) + 16);

            for (int iter = 0; iter < maxIter; ++iter)
            {
                bool improved = false;
                for (IndexType p1 = 0; p1 < nrQubits; ++p1)
                {
                    IndexType u = order[p1];
                    if (qubitGraphCost(u, p1, pos) <= 1e-12)
                        continue;

                    candPositions.clear();
                    const IndexType p2Max = std::min<IndexType>(nrQubits - 1, p1 + maxWindow);
                    for (IndexType p2 = p1 + 1; p2 <= p2Max; ++p2)
                        candPositions.push_back(p2);
                    for (const auto &nb : adjList[u])
                    {
                        const IndexType np = pos[nb.first];
                        for (IndexType delta = -1; delta <= 1; ++delta)
                        {
                            const IndexType p2 = np + delta;
                            if (p2 > p1 + maxWindow && p2 < nrQubits)
                                candPositions.push_back(p2);
                        }
                    }

                    for (IndexType p2 : candPositions)
                    {
                        u = order[p1];
                        const IndexType v = order[p2];
                        const double wuv = getEdgeWeight(u, v);
                        const double pairTerm = wuv * pairSwapPenalty(p1, p2);
                        const double before = qubitGraphCost(u, p1, pos) + qubitGraphCost(v, p2, pos) - pairTerm;

                        pos[u] = p2;
                        pos[v] = p1;
                        const double after = qubitGraphCost(u, p2, pos) + qubitGraphCost(v, p1, pos) - pairTerm;

                        if (after + 1e-9 < before)
                        {
                            order[p1] = v;
                            order[p2] = u;
                            improved = true;
                        }
                        else
                        {
                            pos[u] = p1;
                            pos[v] = p2;
                        }
                    }
                }

                // Incremental single-qubit insertion via adjacent-swap sliding:
                // each 1-slot shift only changes incident edge distances of the two
                // swapped qubits, avoiding full-graph rescans and vector copies.
                auto swapAdjacentPositions = [&](IndexType pA, IndexType pB) -> double {
                    const IndexType qA = order[pA];
                    const IndexType qB = order[pB];
                    // Note: the edge (qA, qB) has |pA - pB| == 1 both before and after,
                    // where pairSwapPenalty is 0.
                    const double before = qubitGraphCost(qA, pA, pos) + qubitGraphCost(qB, pB, pos);
                    pos[qA] = pB;
                    pos[qB] = pA;
                    order[pA] = qB;
                    order[pB] = qA;
                    const double after = qubitGraphCost(qA, pB, pos) + qubitGraphCost(qB, pA, pos);
                    return after - before;
                };

                for (IndexType fromP = 0; fromP < nrQubits; ++fromP)
                {
                    const IndexType q = order[fromP];
                    if (qubitGraphCost(q, fromP, pos) <= 1e-12)
                        continue;

                    const IndexType maxRight = std::min<IndexType>(nrQubits - 1, fromP + maxWindow);
                    const IndexType minLeft = std::max<IndexType>(0, fromP - maxWindow);

                    IndexType bestP = fromP;
                    double bestDelta = 0.0;
                    double cumDelta = 0.0;

                    for (IndexType curP = fromP; curP < maxRight; ++curP)
                    {
                        cumDelta += swapAdjacentPositions(curP, curP + 1);
                        if (cumDelta + 1e-9 < bestDelta)
                        {
                            bestDelta = cumDelta;
                            bestP = curP + 1;
                        }
                    }
                    for (IndexType curP = maxRight; curP > fromP; --curP)
                        swapAdjacentPositions(curP, curP - 1);

                    cumDelta = 0.0;
                    for (IndexType curP = fromP; curP > minLeft; --curP)
                    {
                        cumDelta += swapAdjacentPositions(curP, curP - 1);
                        if (cumDelta + 1e-9 < bestDelta)
                        {
                            bestDelta = cumDelta;
                            bestP = curP - 1;
                        }
                    }
                    for (IndexType curP = minLeft; curP < bestP; ++curP)
                        swapAdjacentPositions(curP, curP + 1);

                    if (bestP != fromP)
                        improved = true;
                }

                if (!improved || totalGraphCost(pos) <= 1e-12)
                    break;
            }
            return pos;
        };

        // Multi-layer weighted group merging with 4-way orientation selection,
        // using sparse adjacency lookups O(|E|).
        auto buildWeightedChain = [&]() -> std::vector<IndexType> {
            std::vector<std::deque<IndexType>> groups;
            std::vector<int> qubitGroup(nrQubits, -1);
            std::vector<IndexType> posInGroup(nrQubits, 0);

            auto refreshGroupPositions = [&](int gIdx) {
                const auto &grp = groups[gIdx];
                for (size_t i = 0; i < grp.size(); ++i)
                    posInGroup[grp[i]] = static_cast<IndexType>(i);
            };

            for (const auto &e : edges)
            {
                const int g1 = qubitGroup[e.u];
                const int g2 = qubitGroup[e.v];
                if (g1 < 0 && g2 < 0)
                {
                    const int gIdx = static_cast<int>(groups.size());
                    groups.push_back({e.u, e.v});
                    qubitGroup[e.u] = gIdx;
                    qubitGroup[e.v] = gIdx;
                    posInGroup[e.u] = 0;
                    posInGroup[e.v] = 1;
                }
                else if (g1 >= 0 && g2 < 0)
                {
                    auto &grp = groups[g1];
                    const IndexType gSize = static_cast<IndexType>(grp.size());
                    double costFront = 0.0;
                    double costBack = 0.0;
                    for (const auto &nb : adjList[e.v])
                    {
                        if (qubitGroup[nb.first] == g1)
                        {
                            const IndexType idx = posInGroup[nb.first];
                            costFront += nb.second * pairSwapPenalty(0, idx + 1);
                            costBack += nb.second * pairSwapPenalty(gSize, idx);
                        }
                    }
                    if (costFront <= costBack)
                    {
                        grp.push_front(e.v);
                        qubitGroup[e.v] = g1;
                        refreshGroupPositions(g1);
                    }
                    else
                    {
                        posInGroup[e.v] = gSize;
                        grp.push_back(e.v);
                        qubitGroup[e.v] = g1;
                    }
                }
                else if (g1 < 0 && g2 >= 0)
                {
                    auto &grp = groups[g2];
                    const IndexType gSize = static_cast<IndexType>(grp.size());
                    double costFront = 0.0;
                    double costBack = 0.0;
                    for (const auto &nb : adjList[e.u])
                    {
                        if (qubitGroup[nb.first] == g2)
                        {
                            const IndexType idx = posInGroup[nb.first];
                            costFront += nb.second * pairSwapPenalty(0, idx + 1);
                            costBack += nb.second * pairSwapPenalty(gSize, idx);
                        }
                    }
                    if (costFront <= costBack)
                    {
                        grp.push_front(e.u);
                        qubitGroup[e.u] = g2;
                        refreshGroupPositions(g2);
                    }
                    else
                    {
                        posInGroup[e.u] = gSize;
                        grp.push_back(e.u);
                        qubitGroup[e.u] = g2;
                    }
                }
                else if (g1 != g2)
                {
                    // Merge smaller group into larger group to keep position updates O(N log N)
                    const int gA = (groups[g1].size() >= groups[g2].size()) ? g1 : g2;
                    const int gB = (gA == g1) ? g2 : g1;
                    auto &grpA = groups[gA];
                    auto &grpB = groups[gB];
                    const IndexType sizeA = static_cast<IndexType>(grpA.size());
                    const IndexType sizeB = static_cast<IndexType>(grpB.size());

                    std::array<double, 4> orientCosts{0.0, 0.0, 0.0, 0.0};
                    for (IndexType ib = 0; ib < sizeB; ++ib)
                    {
                        const IndexType qb = grpB[ib];
                        for (const auto &nb : adjList[qb])
                        {
                            if (qubitGroup[nb.first] == gA)
                            {
                                const IndexType ia = posInGroup[nb.first];
                                for (int flipA = 0; flipA < 2; ++flipA)
                                {
                                    const IndexType effA = flipA ? (sizeA - 1 - ia) : ia;
                                    for (int flipB = 0; flipB < 2; ++flipB)
                                    {
                                        const IndexType effB = sizeA + (flipB ? (sizeB - 1 - ib) : ib);
                                        orientCosts[flipA * 2 + flipB] += nb.second * pairSwapPenalty(effA, effB);
                                    }
                                }
                            }
                        }
                    }

                    int bestOrient = 0;
                    for (int k = 1; k < 4; ++k)
                        if (orientCosts[k] < orientCosts[bestOrient])
                            bestOrient = k;

                    if (bestOrient & 2)
                        std::reverse(grpA.begin(), grpA.end());
                    if (bestOrient & 1)
                        std::reverse(grpB.begin(), grpB.end());
                    for (const auto q : grpB)
                    {
                        grpA.push_back(q);
                        qubitGroup[q] = gA;
                    }
                    grpB.clear();
                    refreshGroupPositions(gA);
                }
            }

            std::vector<IndexType> chain;
            chain.reserve(nQ);
            for (const auto &grp : groups)
                for (const auto q : grp)
                    chain.push_back(q);
            for (IndexType q = 0; q < nrQubits; ++q)
                if (qubitGroup[q] < 0)
                    chain.push_back(q);

            std::vector<IndexType> result(nQ);
            for (size_t i = 0; i < chain.size(); ++i)
                result[chain[i]] = static_cast<IndexType>(i);
            return result;
        };

        // Spectral Fiedler vector ordering of the time-decayed interaction graph
        auto buildSpectralChain = [&]() -> std::vector<IndexType> {
            std::vector<double> deg(nQ, 0.0);
            for (size_t u = 0; u < nQ; ++u)
                for (const auto &nb : adjList[u])
                    deg[u] += nb.second;

            std::vector<double> v0(nQ, 0.0);
            double v0Norm2 = 0.0;
            for (size_t u = 0; u < nQ; ++u)
            {
                v0[u] = std::sqrt(std::max(deg[u], 1e-12));
                v0Norm2 += v0[u] * v0[u];
            }
            const double invV0Norm = 1.0 / std::sqrt(std::max(v0Norm2, 1e-24));
            for (size_t u = 0; u < nQ; ++u)
                v0[u] *= invV0Norm;

            std::vector<double> x(nQ);
            for (size_t u = 0; u < nQ; ++u)
                x[u] = static_cast<double>(u) - 0.5 * static_cast<double>(nQ - 1);

            for (int iter = 0; iter < 40; ++iter)
            {
                double dot0 = 0.0;
                for (size_t u = 0; u < nQ; ++u)
                    dot0 += x[u] * v0[u];
                for (size_t u = 0; u < nQ; ++u)
                    x[u] -= dot0 * v0[u];

                std::vector<double> y(nQ, 0.0);
                for (size_t u = 0; u < nQ; ++u)
                {
                    y[u] = 0.5 * x[u];
                    if (deg[u] > 1e-12)
                    {
                        const double invSqrtDu = 1.0 / std::sqrt(deg[u]);
                        for (const auto &nb : adjList[u])
                            if (deg[nb.first] > 1e-12)
                                y[u] += 0.5 * nb.second * invSqrtDu * (x[nb.first] / std::sqrt(deg[nb.first]));
                    }
                }

                dot0 = 0.0;
                for (size_t u = 0; u < nQ; ++u)
                    dot0 += y[u] * v0[u];
                double norm2 = 0.0;
                for (size_t u = 0; u < nQ; ++u)
                {
                    y[u] -= dot0 * v0[u];
                    norm2 += y[u] * y[u];
                }
                if (norm2 < 1e-24)
                    break;
                const double invNorm = 1.0 / std::sqrt(norm2);
                for (size_t u = 0; u < nQ; ++u)
                    x[u] = y[u] * invNorm;
            }

            // Precompute quantized integer keys so comparator is a strict weak ordering (transitive)
            std::vector<int64_t> quantKey(nQ, 0);
            for (size_t u = 0; u < nQ; ++u)
            {
                const double coord = x[u] / std::sqrt(std::max(deg[u], 1e-12));
                quantKey[u] = std::isfinite(coord) ? static_cast<int64_t>(std::llround(coord * 1e9)) : 0;
            }

            std::vector<IndexType> order(nQ);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](IndexType a, IndexType b) {
                return std::tie(quantKey[a], a) < std::tie(quantKey[b], b);
            });

            std::vector<IndexType> result(nQ);
            for (size_t i = 0; i < nQ; ++i)
                result[order[i]] = static_cast<IndexType>(i);
            return result;
        };

        // The cost model is a rough proxy: for some circuits (QFT, for one) it
        // rates scrambled layouts a few times cheaper than the identity, while
        // they can run far slower. Leave the identity only for a large predicted
        // gain; a circuit whose labels really are scrambled relative to its
        // interactions shows gains of orders of magnitude.
        std::vector<long long int> identityMap(nrQubits);
        std::iota(identityMap.begin(), identityMap.end(), 0);
        auto optMap = buildChain(layerPairs);
        auto optCost = evaluateCost(optMap);
        const double identityCost = optMap == identityMap ? optCost : evaluateCost(identityMap);
        if (identityCost < optCost)
        {
            optCost = identityCost;
            optMap = identityMap;
        }

        // The graph candidates come from a local search whose cost grows with the
        // qubits' degree. On a dense interaction graph (average degree above 8:
        // QFT, quantum volume) no line ordering keeps the interacting pairs close,
        // the search dominates the layout time (about 40 ms for a 64-qubit QFT)
        // and its candidates did not win, so it is skipped there.
        const bool denseGraph = edges.size() > 4 * nQ;
        if (!denseGraph && totalGraphCost(optMap) > 1e-12)
        {
            std::vector<std::vector<IndexType>> graphCandidates;
            graphCandidates.push_back(refineGraphLayout(identityMap));
            graphCandidates.push_back(refineGraphLayout(optMap));
            if (layers.size() > 2)
            {
                graphCandidates.push_back(refineGraphLayout(buildWeightedChain()));
                graphCandidates.push_back(refineGraphLayout(buildSpectralChain()));
            }

            for (const auto &cand : graphCandidates)
            {
                if (cand == identityMap || cand == optMap)
                    continue;
                const double cost = evaluateCostBounded(cand, optCost);
                if (cost < optCost)
                {
                    optCost = cost;
                    optMap = cand;
                }
            }
        }

        if (layers.size() <= 2)
            return optMap != identityMap && identityCost < minLayoutGain * optCost ? identityMap : optMap;

        // nrShuffles full random shuffles, then a perturbation search (1-3 random
        // swaps per try) that always runs: MPO layouts depend on it, and gating it
        // on nrShuffles (0 by default) left MPO QFT with worse layouts.
        {
            std::vector<long long int> trialMap = identityMap;
            std::mt19937 rng(42);
            for (int i = 0; i < nrShuffles; ++i)
            {
                std::shuffle(trialMap.begin(), trialMap.end(), rng);
                const double tryCost = evaluateCostBounded(trialMap, optCost);
                if (tryCost < optCost)
                {
                    optCost = tryCost;
                    optMap = trialMap;
                }
            }

            std::uniform_int_distribution<IndexType> qubitDist(0, nrQubits - 1);
            std::uniform_int_distribution<int> nrSwapsDist(1, std::min<int>(3, static_cast<int>(nrQubits) / 2));

            const int maxNoImprove = std::max(nrShuffles, static_cast<int>(nrQubits));
            const int maxTotalShuffles = maxNoImprove * 3;
            int noImproveCount = 0;

            for (int s = 0; s < maxTotalShuffles && noImproveCount < maxNoImprove; ++s)
            {
                auto tryMap = optMap;
                const int swapsCount = nrSwapsDist(rng);
                for (int sw = 0; sw < swapsCount; ++sw)
                {
                    const IndexType a = qubitDist(rng);
                    IndexType b = qubitDist(rng);
                    while (b == a)
                        b = qubitDist(rng);
                    std::swap(tryMap[a], tryMap[b]);
                }

                const auto cost = evaluateCostBounded(tryMap, optCost);
                if (cost < optCost)
                {
                    optMap = tryMap;
                    optCost = cost;
                    noImproveCount = 0;
                }
                else
                {
                    ++noImproveCount;
                }
            }
        }

        // Optional explicit 2-opt local search in MPSDummySimulator when nrSwaps > 0
        if (nrSwaps > 0)
        {
            auto candidate = optMap;
            bool improved = true;
            for (int improvementCount = 0; improved && improvementCount < nrSwaps; ++improvementCount)
            {
                improved = false;
                for (IndexType i = 0; i < nrQubits; ++i)
                {
                    for (IndexType j = i + 1; j < nrQubits; ++j)
                    {
                        std::swap(candidate[i], candidate[j]);
                        const auto cost = evaluateCostBounded(candidate, optCost);
                        if (cost < optCost)
                        {
                            optMap = candidate;
                            optCost = cost;
                            improved = true;
                        }
                        else
                        {
                            std::swap(candidate[i], candidate[j]);
                        }
                    }
                }
            }
        }

        if (optMap != identityMap && identityCost < minLayoutGain * optCost)
            return identityMap;
        return optMap;
    }

  private:
    void InitQubitsMap()
    {
        qubitsMap.resize(getNrQubits());
        qubitsMapInv.resize(getNrQubits());

        for (IndexType i = 0; i < static_cast<IndexType>(getNrQubits()); ++i)
            qubitsMapInv[i] = qubitsMap[i] = i;

        totalSwappingCost = 0;
        if (!currentBondDim.empty())
            std::fill(currentBondDim.begin(), currentBondDim.end(), 1.0);
        if (!bondCost.empty())
            std::fill(bondCost.begin(), bondCost.end(), 1.0);
    }

    struct LightweightInitTag
    {
    };

    // Lightweight constructor: sets up qubit maps but skips the expensive
    // SetMaxBondDimension computation.  Caller must populate bond arrays.
    MPSDummySimulator(size_t N, LightweightInitTag) : nrQubits(N)
    {
    }

    void SwapQubits(IndexType qubit1, IndexType qubit2)
    {
        IndexType realq1 = qubitsMap[qubit1];
        IndexType realq2 = qubitsMap[qubit2];
        if (realq1 > realq2)
        {
            std::swap(realq1, realq2);
            std::swap(qubit1, qubit2);
        }

        if (realq2 - realq1 <= 1)
            return;

        const IndexType meetPos = ComputeHeuristicMeetPosition(realq1, realq2);
        SwapQubitsToPosition(qubit1, qubit2, meetPos);
    }

    // Pick the meeting position with the lowest bond cost (the lowest current
    // bond dimension) between the two qubits. QCSim's own local heuristic now
    // uses a cut-preserving score, but this path routes every gate itself, and
    // here the plain rule works better both as a rollout policy (it adds a route
    // the other cost-based policies don't) and in the layout search's cost model.
    IndexType ComputeHeuristicMeetPosition(IndexType realq1, IndexType realq2) const
    {
        assert(realq1 < realq2);

        IndexType bestPos = realq1;
        double bestCost = bondCost[realq1];

        for (IndexType m = realq1 + 1; m < realq2; ++m)
        {
            if (bondCost[m] < bestCost)
            {
                bestCost = bondCost[m];
                bestPos = m;
            }
        }

        return bestPos;
    }

    // An MPO site has physical dimension 4, and a gate acts on it as
    // rho -> U rho U^dagger, whose operator Schmidt rank is the square of the
    // gate's (the swap special case still holds, it only relabels the sites).
    // Below rank 2 (small-angle gates) the result is interpolated linearly
    // between rank 1, which leaves the bond unchanged, and rank 2, so the
    // calibrated growth factors still apply unchanged to integer ranks.
    double EstimateGrownBondDimension(IndexType bond, double leftDim, double betweenDim, double rightNeighborDim, bool swap, double schmidtRank) const
    {
        const double rank = IsOperatorChain() ? schmidtRank * schmidtRank : schmidtRank;
        auto grownAt = [&](double r) {
            double newMaxDim = (swap && leftDim == rightNeighborDim) ? betweenDim : static_cast<double>(physExtent) * std::min(leftDim, rightNeighborDim);
            newMaxDim = std::min(newMaxDim, betweenDim * r);
            const double growthFactor = swap ? growthFactorSwap : growthFactorGate;
            return std::max(std::min(newMaxDim * growthFactor, maxBondDim[bond]), 1.);
        };
        if (rank >= 2.)
            return grownAt(rank);
        return betweenDim + (std::max(rank, 1.) - 1.) * (grownAt(2.) - betweenDim);
    }

    // Anchor the qubit surrounded by higher bond cost so the less-entangled
    // qubit travels across the cheaper region instead of dragging a heavy
    // entangled cluster across the chain.
    IndexType ComputeAnchorHeavierMeetPosition(IndexType realq1, IndexType realq2) const
    {
        assert(realq1 < realq2);

        const double leftOuter = (realq1 > 0) ? bondCost[realq1 - 1] : 0.0;
        const double rightOuter = (realq2 < static_cast<IndexType>(bondCost.size())) ? bondCost[realq2] : 0.0;
        const double c1 = std::max(leftOuter, bondCost[realq1]);
        const double c2 = std::max(rightOuter, bondCost[realq2 - 1]);

        if (c1 > c2)
            return realq1;
        if (c2 > c1)
            return realq2 - 1;

        return (realq1 <= static_cast<IndexType>(nrQubits) - 1 - realq2) ? realq1 : (realq2 - 1);
    }

    struct FastGateOp
    {
        IndexType q1 = 0;
        IndexType q2 = 0;
        IndexType q3 = 0;
        uint8_t numQubits = 2;
        bool isSwap = false;
        double schmidtRank = 4;
    };

    static constexpr size_t kMaxStackQubits = 64;

    struct FastStateSnapshot
    {
        std::array<IndexType, kMaxStackQubits> qMap;
        std::array<IndexType, kMaxStackQubits> qMapInv;
        std::array<double, kMaxStackQubits> curBondDim;
        std::array<double, kMaxStackQubits> bCost;
        std::vector<IndexType> heapMap;
        std::vector<double> heapBonds;
        double totalCost = 0.0;

        void SaveFrom(const MPSDummySimulator &sim)
        {
            const size_t n = sim.nrQubits;
            const size_t nb = (n > 0) ? (n - 1) : 0;
            totalCost = sim.totalSwappingCost;
            if (n <= kMaxStackQubits)
            {
                std::memcpy(qMap.data(), sim.qubitsMap.data(), n * sizeof(IndexType));
                std::memcpy(qMapInv.data(), sim.qubitsMapInv.data(), n * sizeof(IndexType));
                if (nb > 0)
                {
                    std::memcpy(curBondDim.data(), sim.currentBondDim.data(), nb * sizeof(double));
                    std::memcpy(bCost.data(), sim.bondCost.data(), nb * sizeof(double));
                }
            }
            else
            {
                heapMap.resize(2 * n);
                std::memcpy(heapMap.data(), sim.qubitsMap.data(), n * sizeof(IndexType));
                std::memcpy(heapMap.data() + n, sim.qubitsMapInv.data(), n * sizeof(IndexType));
                heapBonds.resize(2 * nb);
                if (nb > 0)
                {
                    std::memcpy(heapBonds.data(), sim.currentBondDim.data(), nb * sizeof(double));
                    std::memcpy(heapBonds.data() + nb, sim.bondCost.data(), nb * sizeof(double));
                }
            }
        }

        void RestoreTo(MPSDummySimulator &sim) const
        {
            const size_t n = sim.nrQubits;
            const size_t nb = (n > 0) ? (n - 1) : 0;
            sim.totalSwappingCost = totalCost;
            if (n <= kMaxStackQubits)
            {
                std::memcpy(sim.qubitsMap.data(), qMap.data(), n * sizeof(IndexType));
                std::memcpy(sim.qubitsMapInv.data(), qMapInv.data(), n * sizeof(IndexType));
                if (nb > 0)
                {
                    std::memcpy(sim.currentBondDim.data(), curBondDim.data(), nb * sizeof(double));
                    std::memcpy(sim.bondCost.data(), bCost.data(), nb * sizeof(double));
                }
            }
            else
            {
                std::memcpy(sim.qubitsMap.data(), heapMap.data(), n * sizeof(IndexType));
                std::memcpy(sim.qubitsMapInv.data(), heapMap.data() + n, n * sizeof(IndexType));
                if (nb > 0)
                {
                    std::memcpy(sim.currentBondDim.data(), heapBonds.data(), nb * sizeof(double));
                    std::memcpy(sim.bondCost.data(), heapBonds.data() + nb, nb * sizeof(double));
                }
            }
        }
    };

    size_t ExtractFastGateWindow(const std::vector<std::shared_ptr<Circuits::IOperation<>>> &upcomingGates, long long int startGateIndex, size_t maxCount,
                                 FastGateOp *outOps) const
    {
        size_t count = 0;
        if (startGateIndex < 0)
            return 0;

        for (long long int idx = startGateIndex; idx < static_cast<long long int>(upcomingGates.size()) && count < maxCount; ++idx)
        {
            const auto &op = upcomingGates[idx];
            if (!op || op->GetType() != Circuits::OperationType::kGate)
                break;
            const auto qbits = op->AffectedQubits();
            if (qbits.size() < 2)
                continue;

            FastGateOp fg;
            fg.q1 = static_cast<IndexType>(qbits[0]);
            fg.q2 = static_cast<IndexType>(qbits[1]);
            fg.numQubits = static_cast<uint8_t>(qbits.size());
            fg.isSwap = false;
            fg.schmidtRank = 4;

            if (qbits.size() == 2)
            {
                const auto gateptr = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(op);
                const auto info = std::dynamic_pointer_cast<RoutingGateInfo>(op);
                if (info)
                {
                    fg.isSwap = info->isSwap;
                    fg.schmidtRank = info->schmidtRank;
                }
                else if (gateptr && gateptr->GetGateType() == Circuits::QuantumGateType::kSwapGateType)
                {
                    fg.isSwap = true;
                    fg.schmidtRank = 4;
                }
                else if (gateptr)
                {
                    fg.schmidtRank = GateSchmidtRank(*gateptr);
                }
            }
            else if (qbits.size() == 3)
            {
                fg.q3 = static_cast<IndexType>(qbits[2]);
                const auto gateptr = std::static_pointer_cast<Circuits::IQuantumGate<>>(op);
                fg.isSwap = (gateptr->GetGateType() != Circuits::QuantumGateType::kCCXGateType);
                fg.schmidtRank = 2;
            }

            outOps[count++] = fg;
        }

        return count;
    }

    void ApplyTwoQubitFastGate(IndexType qubit, IndexType controllingQubit1, bool isSwap, double schmidtRank)
    {
        if (isSwap && swapGatesRelabelQubits)
            return RelabelQubits(qubit, controllingQubit1);
        IndexType qubit1 = qubitsMap[qubit];
        IndexType qubit2 = qubitsMap[controllingQubit1];

        if (abs(qubit1 - qubit2) > 1)
        {
            SwapQubits(qubit, controllingQubit1);
            qubit1 = qubitsMap[qubit];
            qubit2 = qubitsMap[controllingQubit1];
        }

        const IndexType bond = std::min(qubit1, qubit2);
        totalSwappingCost += bondCost[bond];
        growBondDimension(bond, isSwap, schmidtRank);
        totalSwappingCost += kPostGateCostWeight * bondCost[bond];
    }

    void ApplyFastGate(const FastGateOp &g)
    {
        if (g.numQubits == 2)
        {
            ApplyTwoQubitFastGate(g.q1, g.q2, g.isSwap, g.schmidtRank);
        }
        else if (g.numQubits == 3)
        {
            if (!g.isSwap) // CCX
            {
                ApplyTwoQubitFastGate(g.q3, g.q2, false, 2);
                ApplyTwoQubitFastGate(g.q2, g.q1, false, 2);
                ApplyTwoQubitFastGate(g.q3, g.q2, false, 2);
                ApplyTwoQubitFastGate(g.q2, g.q1, false, 2);
                ApplyTwoQubitFastGate(g.q3, g.q1, false, 2);
            }
            else // CSwap
            {
                ApplyTwoQubitFastGate(g.q2, g.q3, false, 2);
                ApplyTwoQubitFastGate(g.q3, g.q2, false, 2);
                ApplyTwoQubitFastGate(g.q2, g.q1, false, 2);
                ApplyTwoQubitFastGate(g.q3, g.q2, false, 2);
                ApplyTwoQubitFastGate(g.q2, g.q1, false, 2);
                ApplyTwoQubitFastGate(g.q3, g.q1, false, 2);
                ApplyTwoQubitFastGate(g.q2, g.q3, false, 2);
            }
        }
    }

    bool IsRelabelingSwap(const FastGateOp &gate) const
    {
        return swapGatesRelabelQubits && gate.numQubits == 2 && gate.isSwap;
    }

    struct PartnerPull
    {
        IndexType pos = 0;
        double weight = 0.0;
    };

    // Scan upcoming pre-parsed gates to collect weighted physical positions of
    // the next partners of qubit1 and qubit2.
    size_t CollectNextPartnerPullsFast(IndexType qubit1, IndexType qubit2, const FastGateOp *window, size_t windowSize, size_t stepIdx,
                                       std::array<PartnerPull, 16> &pulls, double &totalWeight) const
    {
        size_t numPulls = 0;
        totalWeight = 0.0;
        bool foundQ1 = false;
        bool foundQ2 = false;
        double weight = 1.0;
        const size_t maxStep = std::min(windowSize, stepIdx + 9);
        std::array<std::pair<IndexType, IndexType>, 8> relabels;
        size_t numRelabels = 0;
        auto originalLabel = [&](IndexType q) {
            for (size_t i = numRelabels; i > 0; --i)
            {
                const auto [a, b] = relabels[i - 1];
                if (q == a)
                    q = b;
                else if (q == b)
                    q = a;
            }
            return q;
        };

        for (size_t s = stepIdx + 1; s < maxStep; ++s)
        {
            const FastGateOp &fg = window[s];
            if (IsRelabelingSwap(fg))
            {
                relabels[numRelabels++] = {fg.q1, fg.q2};
                continue;
            }
            const IndexType qA = originalLabel(fg.q1);
            const IndexType qB = originalLabel(fg.q2);

            auto recordPartner = [&](IndexType activeQ, IndexType partnerQ) {
                if (partnerQ == qubit1 || partnerQ == qubit2)
                    return;
                const bool isFirst = (activeQ == qubit1) ? !foundQ1 : !foundQ2;
                const double w = isFirst ? weight : (weight * 0.35);
                if (numPulls < pulls.size())
                {
                    pulls[numPulls++] = {qubitsMap[partnerQ], w};
                    totalWeight += w;
                }
                if (activeQ == qubit1)
                    foundQ1 = true;
                else
                    foundQ2 = true;
            };

            if (qA == qubit1 || qA == qubit2)
                recordPartner(qA, qB);
            if (qB == qubit1 || qB == qubit2)
                recordPartner(qB, qA);

            if (foundQ1 && foundQ2)
                break;
            weight *= 0.6;
        }

        return numPulls;
    }

    IndexType ComputeNextPartnerGravityMeetPositionFast(IndexType realq1, IndexType realq2, IndexType qubit1, IndexType qubit2, const FastGateOp *window,
                                                        size_t windowSize, size_t stepIdx) const
    {
        assert(realq1 < realq2);

        std::array<PartnerPull, 16> pulls;
        double totalWeight = 0.0;
        const size_t numPulls = CollectNextPartnerPullsFast(qubit1, qubit2, window, windowSize, stepIdx, pulls, totalWeight);
        if (numPulls == 0 || totalWeight <= 0.0)
            return ComputeAnchorHeavierMeetPosition(realq1, realq2);

        IndexType bestPos = realq1;
        double bestDist = std::numeric_limits<double>::infinity();
        double bestLocalBondCost = std::numeric_limits<double>::infinity();

        for (IndexType m = realq1; m < realq2; ++m)
        {
            double dist = 0.0;
            for (size_t k = 0; k < numPulls; ++k)
                dist += pulls[k].weight * static_cast<double>(std::abs(m - pulls[k].pos));

            const double localCost = bondCost[m];
            if (dist < bestDist - 1e-9 || (std::abs(dist - bestDist) <= 1e-9 && localCost < bestLocalBondCost))
            {
                bestDist = dist;
                bestLocalBondCost = localCost;
                bestPos = m;
            }
        }

        return bestPos;
    }

    // Compute the meeting position that minimizes the exact post-step bond cost
    // in O(realq2 - realq1) time, optionally tilted by L1 partner distance.
    IndexType ComputeMinOneStepCostMeetPositionFast(IndexType realq1, IndexType realq2, IndexType qubit1, IndexType qubit2, const FastGateOp *window,
                                                    size_t windowSize, size_t stepIdx, bool includeGravity) const
    {
        assert(realq1 < realq2);
        const IndexType span = realq2 - realq1;
        if (span <= 1)
            return realq1;

        const FastGateOp &fg = window[stepIdx];
        const bool isSwap = (fg.numQubits == 2) ? fg.isSwap : false;
        const double schmidtRank = fg.schmidtRank;

        constexpr IndexType kMaxStackSpan = 128;
        std::array<double, kMaxStackSpan> rightSweepDimStack{};
        std::array<double, kMaxStackSpan> prefixPostCostStack{};
        std::array<double, kMaxStackSpan> leftSweepDimStack{};
        std::array<double, kMaxStackSpan> suffixPostCostStack{};

        std::vector<double> heapBuf;
        double *rightSweepDim = rightSweepDimStack.data();
        double *prefixPostCost = prefixPostCostStack.data();
        double *leftSweepDim = leftSweepDimStack.data();
        double *suffixPostCost = suffixPostCostStack.data();

        if (span > kMaxStackSpan)
        {
            heapBuf.resize(static_cast<size_t>(span) * 4, 0.0);
            rightSweepDim = heapBuf.data();
            prefixPostCost = rightSweepDim + span;
            leftSweepDim = prefixPostCost + span;
            suffixPostCost = leftSweepDim + span;
        }

        prefixPostCost[0] = 0.0;
        for (IndexType i = 0; i < span - 1; ++i)
        {
            const IndexType bond = realq1 + i;
            const double leftD = (i == 0) ? ((realq1 > 0) ? currentBondDim[realq1 - 1] : 1.0) : rightSweepDim[i - 1];
            const double betweenD = currentBondDim[bond];
            const double rightD = currentBondDim[bond + 1];
            const double newD = EstimateGrownBondDimension(bond, leftD, betweenD, rightD, true, 4);
            rightSweepDim[i] = newD;
            prefixPostCost[i + 1] = prefixPostCost[i] + BondCost(newD);
        }

        suffixPostCost[span - 1] = 0.0;
        for (IndexType i = span - 1; i > 0; --i)
        {
            const IndexType bond = realq1 + i;
            const double leftD = currentBondDim[bond - 1];
            const double betweenD = currentBondDim[bond];
            const double rightD =
                (i == span - 1) ? ((realq2 < static_cast<IndexType>(currentBondDim.size())) ? currentBondDim[realq2] : 1.0) : leftSweepDim[i + 1];
            const double newD = EstimateGrownBondDimension(bond, leftD, betweenD, rightD, true, 4);
            leftSweepDim[i] = newD;
            suffixPostCost[i - 1] = suffixPostCost[i] + BondCost(newD);
        }

        std::array<PartnerPull, 16> pulls;
        double totalPullWeight = 0.0;
        const size_t numPulls = CollectNextPartnerPullsFast(qubit1, qubit2, window, windowSize, stepIdx, pulls, totalPullWeight);
        const double gravityWeight = includeGravity ? 0.15 : 1e-6;
        const double invNorm = (numPulls > 0 && totalPullWeight > 0.0) ? (gravityWeight / (static_cast<double>(span) * totalPullWeight)) : 0.0;

        IndexType bestPos = realq1;
        double bestScore = std::numeric_limits<double>::infinity();

        for (IndexType i = 0; i < span; ++i)
        {
            const IndexType m = realq1 + i;
            const double leftD = (i == 0) ? ((realq1 > 0) ? currentBondDim[realq1 - 1] : 1.0) : rightSweepDim[i - 1];
            const double rightD =
                (i == span - 1) ? ((realq2 < static_cast<IndexType>(currentBondDim.size())) ? currentBondDim[realq2] : 1.0) : leftSweepDim[i + 1];
            const double gateD = EstimateGrownBondDimension(m, leftD, currentBondDim[m], rightD, isSwap, schmidtRank);
            const double gateCost = BondCost(gateD);
            // Share the exact post-growth cost coefficients (kPostSwapCostWeight * S +
            // kPostGateCostWeight * G) with SwapQubitsToPosition and ApplyGate so the
            // one-step objective matches the rollout's cost model exactly.
            const double postBondCost = kPostSwapCostWeight * (prefixPostCost[i] + suffixPostCost[i]) + kPostGateCostWeight * gateCost;

            double score = postBondCost;
            if (invNorm > 0.0)
            {
                double pullDist = 0.0;
                for (size_t k = 0; k < numPulls; ++k)
                    pullDist += pulls[k].weight * static_cast<double>(std::abs(m - pulls[k].pos));
                score *= (1.0 + invNorm * pullDist);
            }

            if (score < bestScore)
            {
                bestScore = score;
                bestPos = m;
            }
        }

        return bestPos;
    }

    IndexType SelectMeetPositionForPolicyFast(HeuristicPolicy policy, IndexType realq1, IndexType realq2, IndexType qubit1, IndexType qubit2,
                                              const FastGateOp *window, size_t windowSize, size_t stepIdx) const
    {
        if (realq2 - realq1 <= 1)
            return realq1;

        switch (policy)
        {
        case HeuristicPolicy::kCombinedCostAndGravity:
            return ComputeMinOneStepCostMeetPositionFast(realq1, realq2, qubit1, qubit2, window, windowSize, stepIdx, true);
        case HeuristicPolicy::kMinOneStepCost:
            return ComputeMinOneStepCostMeetPositionFast(realq1, realq2, qubit1, qubit2, window, windowSize, stepIdx, false);
        case HeuristicPolicy::kNextPartnerGravity:
            return ComputeNextPartnerGravityMeetPositionFast(realq1, realq2, qubit1, qubit2, window, windowSize, stepIdx);
        case HeuristicPolicy::kAnchorHeavierQubit:
            return ComputeAnchorHeavierMeetPosition(realq1, realq2);
        case HeuristicPolicy::kMinBondCost:
            return ComputeHeuristicMeetPosition(realq1, realq2);
        case HeuristicPolicy::kSwapLeft:
            return realq1;
        case HeuristicPolicy::kSwapRight:
            return realq2 - 1;
        case HeuristicPolicy::kCloserToEnd:
            return (realq1 <= static_cast<IndexType>(nrQubits) - 1 - realq2) ? realq1 : (realq2 - 1);
        case HeuristicPolicy::kCloserToCenter: {
            const IndexType center = std::max<IndexType>(0, (static_cast<IndexType>(nrQubits) - 2) / 2);
            return std::clamp(center, realq1, realq2 - 1);
        }
        }

        return ComputeHeuristicMeetPosition(realq1, realq2);
    }

    // Evaluate a portfolio of heuristic rollout policies over the remaining
    // lookaheadDepth + 1 two-qubit gates using pre-parsed FastGateOp entries and
    // zero-allocation stack snapshots.
    IndexType EvaluateHeuristicRolloutsFast(const FastGateOp *window, size_t windowSize, size_t stepIdx, int lookaheadDepth, double currentCost,
                                            double &bestCost)
    {
        if (stepIdx >= windowSize)
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return 0;
        }

        const FastGateOp &firstOp = window[stepIdx];
        const IndexType firstQ1 = firstOp.q1;
        const IndexType firstQ2 = firstOp.q2;
        IndexType firstReal1 = qubitsMap[firstQ1];
        IndexType firstReal2 = qubitsMap[firstQ2];
        if (firstReal1 > firstReal2)
            std::swap(firstReal1, firstReal2);

        const HeuristicPolicy defaultFirstPolicy = (lookaheadDepth <= 0) ? HeuristicPolicy::kMinOneStepCost : HeuristicPolicy::kCombinedCostAndGravity;
        IndexType bestFirstPosition = IsRelabelingSwap(firstOp)       ? -1
                                      : (firstReal2 - firstReal1 > 1) ? SelectMeetPositionForPolicyFast(defaultFirstPolicy, firstReal1, firstReal2, firstQ1,
                                                                                                        firstQ2, window, windowSize, stepIdx)
                                                                      : firstReal1;

        static constexpr std::array<HeuristicPolicy, 9> kFullPortfolio = {
            HeuristicPolicy::kCombinedCostAndGravity,
            HeuristicPolicy::kMinOneStepCost,
            HeuristicPolicy::kMinBondCost,
            HeuristicPolicy::kAnchorHeavierQubit,
            HeuristicPolicy::kCloserToEnd,
            HeuristicPolicy::kNextPartnerGravity,
            HeuristicPolicy::kSwapLeft,
            HeuristicPolicy::kSwapRight,
            HeuristicPolicy::kCloserToCenter,
        };

        const size_t numPolicies = (lookaheadDepth <= 0) ? 1 : kFullPortfolio.size();
        const size_t endStep = std::min(windowSize, stepIdx + static_cast<size_t>(std::max(0, lookaheadDepth)) + 1);

        FastStateSnapshot baseState;
        baseState.SaveFrom(*this);

        for (size_t pi = 0; pi < numPolicies; ++pi)
        {
            const HeuristicPolicy policy = (lookaheadDepth <= 0) ? HeuristicPolicy::kMinOneStepCost : kFullPortfolio[pi];
            if (pi > 0)
                baseState.RestoreTo(*this);
            setTotalSwappingCost(0);

            double rolloutCost = currentCost;
            IndexType chosenFirstPos = bestFirstPosition;
            bool pruned = false;

            for (size_t s = stepIdx; s < endStep; ++s)
            {
                const FastGateOp &fg = window[s];
                const IndexType q1 = fg.q1;
                const IndexType q2 = fg.q2;
                IndexType rq1 = qubitsMap[q1];
                IndexType rq2 = qubitsMap[q2];
                if (rq1 > rq2)
                    std::swap(rq1, rq2);

                IndexType meetPos = IsRelabelingSwap(fg) ? -1 : rq1;
                if (!IsRelabelingSwap(fg) && rq2 - rq1 > 1)
                {
                    meetPos = SelectMeetPositionForPolicyFast(policy, rq1, rq2, q1, q2, window, windowSize, s);
                    SwapQubitsToPosition(q1, q2, meetPos);
                }

                if (s == stepIdx)
                    chosenFirstPos = meetPos;

                ApplyFastGate(fg);
                rolloutCost = currentCost + getTotalSwappingCost();

                if (rolloutCost >= bestCost)
                {
                    pruned = true;
                    break;
                }
            }

            if (!pruned && rolloutCost < bestCost)
            {
                bestCost = rolloutCost;
                bestFirstPosition = chosenFirstPos;
            }
        }

        baseState.RestoreTo(*this);
        return bestFirstPosition;
    }

    void EvaluateMeetingPositionCostFast(IndexType meetPosition, const FastGateOp *window, size_t windowSize, size_t stepIdx, int lookaheadDepth,
                                         int lookaheadDepthWithHeuristic, double currentCost, double &bestCost, bool mutateSelf)
    {
        if (stepIdx >= windowSize)
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return;
        }

        FastStateSnapshot saved;
        if (!mutateSelf)
            saved.SaveFrom(*this);

        const double ccost = getTotalSwappingCost();
        const FastGateOp &fg = window[stepIdx];
        IndexType realq1 = qubitsMap[fg.q1];
        IndexType realq2 = qubitsMap[fg.q2];
        if (realq1 > realq2)
            std::swap(realq1, realq2);

        if (!IsRelabelingSwap(fg) && realq2 - realq1 > 1)
            SwapQubitsToPosition(fg.q1, fg.q2, meetPosition);

        ApplyFastGate(fg);
        currentCost += (getTotalSwappingCost() - ccost);

        if (currentCost < bestCost)
        {
            if (lookaheadDepth <= 0 || stepIdx + 1 >= windowSize)
            {
                bestCost = currentCost;
            }
            else
            {
                FindBestMeetingPositionFast(window, windowSize, stepIdx + 1, lookaheadDepth - 1, lookaheadDepthWithHeuristic, currentCost, bestCost);
            }
        }

        if (!mutateSelf)
            saved.RestoreTo(*this);
    }

    IndexType FindBestMeetingPositionFast(const FastGateOp *window, size_t windowSize, size_t stepIdx, int lookaheadDepth, int lookaheadDepthWithHeuristic,
                                          double currentCost, double &bestCost)
    {
        if (stepIdx >= windowSize)
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return 0;
        }

        const FastGateOp &fg = window[stepIdx];
        const IndexType qubit1 = fg.q1;
        const IndexType qubit2 = fg.q2;

        if (IsRelabelingSwap(fg))
        {
            EvaluateMeetingPositionCostFast(-1, window, windowSize, stepIdx, lookaheadDepth, lookaheadDepthWithHeuristic, currentCost, bestCost, false);
            return -1; // A logical SWAP needs no meeting position or branching.
        }

        IndexType realq1 = qubitsMap[qubit1];
        IndexType realq2 = qubitsMap[qubit2];
        if (realq1 > realq2)
            std::swap(realq1, realq2);

        if (lookaheadDepth <= 0 || lookaheadDepth <= lookaheadDepthWithHeuristic)
        {
            return EvaluateHeuristicRolloutsFast(window, windowSize, stepIdx, std::max(0, lookaheadDepth), currentCost, bestCost);
        }

        if (realq2 - realq1 <= 1)
        {
            EvaluateMeetingPositionCostFast(realq1, window, windowSize, stepIdx, lookaheadDepth, lookaheadDepthWithHeuristic, currentCost, bestCost, false);
            return realq1;
        }

        // At the root (stepIdx == 0), first evaluate the full 9-policy rollout
        // portfolio to establish a tight upper bound in bestCost for pruning.
        IndexType bestPosition = realq1;
        if (stepIdx == 0)
        {
            bestPosition = EvaluateHeuristicRolloutsFast(window, windowSize, stepIdx, lookaheadDepth, currentCost, bestCost);
        }

        const size_t span = static_cast<size_t>(realq2 - realq1);
        std::array<IndexType, 128> stackCandidates;
        std::vector<IndexType> heapCandidates;
        IndexType *candidates = stackCandidates.data();
        size_t maxCandidates = stackCandidates.size();
        if (span > stackCandidates.size())
        {
            heapCandidates.resize(span);
            candidates = heapCandidates.data();
            maxCandidates = span;
        }

        size_t numCandidates = 0;
        auto addCandidate = [&](IndexType pos) {
            if (pos < realq1 || pos >= realq2)
                return;
            for (size_t i = 0; i < numCandidates; ++i)
                if (candidates[i] == pos)
                    return;
            if (numCandidates < maxCandidates)
                candidates[numCandidates++] = pos;
        };

        if (stepIdx == 0)
            addCandidate(bestPosition);
        addCandidate(ComputeMinOneStepCostMeetPositionFast(realq1, realq2, qubit1, qubit2, window, windowSize, stepIdx, true));
        addCandidate(ComputeMinOneStepCostMeetPositionFast(realq1, realq2, qubit1, qubit2, window, windowSize, stepIdx, false));
        addCandidate(ComputeHeuristicMeetPosition(realq1, realq2));
        addCandidate(ComputeAnchorHeavierMeetPosition(realq1, realq2));
        addCandidate(ComputeNextPartnerGravityMeetPositionFast(realq1, realq2, qubit1, qubit2, window, windowSize, stepIdx));
        addCandidate(realq1);
        addCandidate(realq2 - 1);
        const IndexType centerPos = std::clamp(std::max<IndexType>(0, (static_cast<IndexType>(nrQubits) - 2) / 2), realq1, realq2 - 1);
        addCandidate(centerPos);

        // Enumerate all remaining positions in [realq1, realq2 - 1] after the
        // priority heuristic candidates so branch-and-bound prunes early.
        for (IndexType m = realq1; m < realq2; ++m)
            addCandidate(m);

        if (stepIdx > 0)
            bestPosition = candidates[0];

        for (size_t i = 0; i < numCandidates; ++i)
        {
            const IndexType m = candidates[i];
            const double oldCost = bestCost;
            EvaluateMeetingPositionCostFast(m, window, windowSize, stepIdx, lookaheadDepth, lookaheadDepthWithHeuristic, currentCost, bestCost, false);

            if (bestCost < oldCost)
                bestPosition = m;
        }

        return bestPosition;
    }

    // P0: Cut-preserving meeting position selector with O(span) prefix/suffix
    // future displacement tables for both active endpoints and all shifted
    // intermediate qubits in (realq1, realq2).
    IndexType SelectMeetingPositionP0Fast(IndexType realq1, IndexType realq2, IndexType qubit1, IndexType qubit2, const FastGateOp *window, size_t windowSize,
                                          size_t stepIdx, bool includeGravity) const
    {
        assert(realq1 < realq2);
        const IndexType span = realq2 - realq1;
        if (span <= 1)
            return realq1;

        const IndexType nBonds = static_cast<IndexType>(currentBondDim.size());
        constexpr IndexType kMaxStackSpan = 128;
        std::array<double, kMaxStackSpan + 2> bondsStack{};
        std::array<double, kMaxStackSpan> prefixCutStack{};
        std::array<double, kMaxStackSpan> suffixCutStack{};
        std::array<double, kMaxStackSpan> interShiftStack{};
        std::vector<double> heapBuf;

        double *bonds = bondsStack.data();
        double *prefixCut = prefixCutStack.data();
        double *suffixCut = suffixCutStack.data();
        double *interShift = interShiftStack.data();
        if (span > kMaxStackSpan)
        {
            heapBuf.resize(static_cast<size_t>(span) * 4 + 2, 0.0);
            bonds = heapBuf.data();
            prefixCut = bonds + span + 2;
            suffixCut = prefixCut + span;
            interShift = suffixCut + span;
        }

        bonds[0] = (realq1 > 0) ? currentBondDim[realq1 - 1] : 1.0;
        for (IndexType i = 0; i < span; ++i)
            bonds[i + 1] = currentBondDim[realq1 + i];
        bonds[span + 1] = (realq2 < nBonds) ? currentBondDim[realq2] : 1.0;

        const double E1 = std::max(bonds[0], bonds[1]);
        const double E2 = std::max(bonds[span], bonds[span + 1]);
        const double d = static_cast<double>(physExtent);

        prefixCut[0] = 0.0;
        for (IndexType i = 0; i < span - 1; ++i)
        {
            const double bDim = bonds[i + 1];
            double dmg = 0.0;
            if (E1 > bDim)
            {
                const double inflated = std::min(E1, d * bDim);
                dmg = std::max(0.0, BondCost(inflated) - bondCost[realq1 + i]);
            }
            prefixCut[i + 1] = prefixCut[i] + dmg;
        }

        suffixCut[span - 1] = 0.0;
        for (IndexType i = span - 1; i > 0; --i)
        {
            const double bDim = bonds[i + 1];
            double dmg = 0.0;
            if (E2 > bDim)
            {
                const double inflated = std::min(E2, d * bDim);
                dmg = std::max(0.0, BondCost(inflated) - bondCost[realq1 + i]);
            }
            suffixCut[i - 1] = suffixCut[i] + dmg;
        }

        std::array<PartnerPull, 16> pulls;
        double totalPullWeight = 0.0;
        size_t numPulls = 0;

        if (includeGravity && window && stepIdx + 1 < windowSize)
        {
            numPulls = CollectNextPartnerPullsFast(qubit1, qubit2, window, windowSize, stepIdx, pulls, totalPullWeight);

            // Compute O(span) prefix/suffix future distance change for all
            // intermediate qubits shifted by meeting at m = realq1 + i:
            // - sites k in (realq1, m] shift left by -1
            // - sites k in [m + 1, realq2) shift right by +1
            std::array<IndexType, kMaxStackSpan> nextPartnerPosStack;
            std::array<double, kMaxStackSpan> nextPartnerWStack{};
            std::vector<IndexType> nextPartnerPosHeap;
            std::vector<double> nextPartnerWHeap;
            IndexType *nextPartnerPos = nextPartnerPosStack.data();
            double *nextPartnerW = nextPartnerWStack.data();
            if (span > kMaxStackSpan)
            {
                nextPartnerPosHeap.assign(static_cast<size_t>(span), -1);
                nextPartnerWHeap.assign(static_cast<size_t>(span), 0.0);
                nextPartnerPos = nextPartnerPosHeap.data();
                nextPartnerW = nextPartnerWHeap.data();
            }
            else
            {
                for (IndexType i = 0; i < span; ++i)
                {
                    nextPartnerPos[i] = -1;
                    nextPartnerW[i] = 0.0;
                }
            }

            const size_t maxLook = std::min(windowSize, stepIdx + 13);
            double wDecay = 1.0;
            int foundCount = 0;
            std::array<std::pair<IndexType, IndexType>, 12> relabels;
            size_t numRelabels = 0;
            auto originalLabel = [&](IndexType q) {
                for (size_t i = numRelabels; i > 0; --i)
                {
                    const auto [a, b] = relabels[i - 1];
                    if (q == a)
                        q = b;
                    else if (q == b)
                        q = a;
                }
                return q;
            };

            for (size_t s = stepIdx + 1; s < maxLook && foundCount < span - 1; ++s)
            {
                const FastGateOp &fg = window[s];
                if (IsRelabelingSwap(fg))
                {
                    if (numRelabels < relabels.size())
                        relabels[numRelabels++] = {fg.q1, fg.q2};
                    continue;
                }
                const IndexType qA = originalLabel(fg.q1);
                const IndexType qB = originalLabel(fg.q2);
                const IndexType rA = qubitsMap[qA];
                const IndexType rB = qubitsMap[qB];
                if (rA > realq1 && rA < realq2 && nextPartnerPos[rA - realq1] < 0)
                {
                    nextPartnerPos[rA - realq1] = rB;
                    nextPartnerW[rA - realq1] = wDecay;
                    ++foundCount;
                }
                if (rB > realq1 && rB < realq2 && nextPartnerPos[rB - realq1] < 0)
                {
                    nextPartnerPos[rB - realq1] = rA;
                    nextPartnerW[rB - realq1] = wDecay;
                    ++foundCount;
                }
                wDecay *= 0.82;
            }

            double runningLeftShift = 0.0;
            interShift[0] = 0.0;
            for (IndexType i = 1; i < span; ++i)
            {
                const IndexType p = nextPartnerPos[i];
                if (p >= 0)
                {
                    const IndexType kPos = realq1 + i;
                    const double delta = (p < kPos) ? -1.0 : ((p > kPos) ? +1.0 : 0.0);
                    runningLeftShift += nextPartnerW[i] * delta;
                }
                interShift[i] = runningLeftShift;
            }

            double runningRightShift = 0.0;
            for (IndexType i = span - 2; i >= 0; --i)
            {
                const IndexType p = nextPartnerPos[i + 1];
                if (p >= 0)
                {
                    const IndexType kPos = realq1 + i + 1;
                    const double delta = (p > kPos) ? -1.0 : ((p < kPos) ? +1.0 : 0.0);
                    runningRightShift += nextPartnerW[i + 1] * delta;
                }
                interShift[i] += runningRightShift;
            }
        }

        const double c1 = bonds[0] * bonds[1];
        const double c2 = bonds[span] * bonds[span + 1];
        const double effCap = (runtimeBondCap > 0) ? static_cast<double>(runtimeBondCap) : std::numeric_limits<double>::infinity();

        IndexType bestPos = realq1;
        double bestScore = std::numeric_limits<double>::infinity();

        for (IndexType i = 0; i < span; ++i)
        {
            const IndexType m = realq1 + i;
            const double bLeft = std::min(bonds[i], effCap);
            const double bMid = std::min(bonds[i + 1], effCap);
            const double bRight = std::min(bonds[i + 2], effCap);
            const double localBondCost = bondCost[m];
            const double cutDamage = 0.5 * (prefixCut[i] + suffixCut[i]);
            const double outerContext = 0.05 * (bLeft * bMid * bRight);
            const double tieBreak = (c2 > c1) ? (1e-6 * static_cast<double>(span - 1 - i)) : (1e-6 * static_cast<double>(i));
            const double baseCost = localBondCost + cutDamage + outerContext + tieBreak;

            double pullCost = 0.0;
            if (includeGravity)
            {
                for (size_t k = 0; k < numPulls; ++k)
                    pullCost += pulls[k].weight * static_cast<double>(std::abs(m - pulls[k].pos));
                pullCost += 1.15 * interShift[i];
            }

            const double score = baseCost + 0.35 * localBondCost * (pullCost / static_cast<double>(span));
            if (score < bestScore)
            {
                bestScore = score;
                bestPos = m;
            }
        }

        return bestPos;
    }

  public:
    // 7b: Adaptive meeting-position planner with dynamic multi-step pruned rollout
    // activation, common gate horizon, and configurable branching depth.
    IndexType SelectMeetingPositionAdaptive(const std::vector<std::shared_ptr<Circuits::IOperation<>>> &upcomingGates, long long int currentGateIndex,
                                            int lookaheadDepth, int lookaheadDepthWithHeuristic, double currentCost, double &bestCost)
    {
        if (currentGateIndex < 0 || currentGateIndex >= static_cast<long long int>(upcomingGates.size()))
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return -1;
        }

        const int exactSteps = std::max(0, lookaheadDepth - std::max(0, lookaheadDepthWithHeuristic));

        // Delegate zero-heuristic exhaustive search (lookaheadDepthWithHeuristic <= 0)
        // or depth-0 queries to FindBestMeetingPosition.
        if (lookaheadDepth <= 0 || lookaheadDepthWithHeuristic <= 0)
            return FindBestMeetingPosition(upcomingGates, currentGateIndex, lookaheadDepth, lookaheadDepthWithHeuristic, currentCost, bestCost);

        const size_t maxNeeded = static_cast<size_t>(std::max(16, 2 * std::max(0, lookaheadDepth) + 10));
        std::array<FastGateOp, 128> stackWindow;
        std::vector<FastGateOp> heapWindow;
        FastGateOp *windowPtr = stackWindow.data();
        if (maxNeeded > stackWindow.size())
        {
            heapWindow.resize(maxNeeded);
            windowPtr = heapWindow.data();
        }
        const size_t windowSize = ExtractFastGateWindow(upcomingGates, currentGateIndex, maxNeeded, windowPtr);
        if (windowSize == 0)
        {
            if (currentCost < bestCost)
                bestCost = currentCost;
            return -1;
        }

        // Common gate horizon shared by all candidates and rollout policies so
        // routes that make future gates adjacent are never penalized by running more gates.
        const size_t endStep = std::min(windowSize, static_cast<size_t>(std::max(0, lookaheadDepth)) + 1);

        const FastGateOp &fg0 = windowPtr[0];
        if (IsRelabelingSwap(fg0))
        {
            EvaluateMeetingPositionCostFast(-1, windowPtr, windowSize, 0, lookaheadDepth, lookaheadDepthWithHeuristic, currentCost, bestCost, false);
            return -1;
        }

        auto searchAdaptive = [&](auto &self, size_t stepIdx, double stepStartCost, double &boundCost) -> IndexType {
            if (stepIdx >= endStep)
            {
                if (stepStartCost < boundCost)
                    boundCost = stepStartCost;
                return -1;
            }

            const FastGateOp &fg = windowPtr[stepIdx];
            IndexType rq1 = -1;
            IndexType rq2 = -1;
            if (!IsRelabelingSwap(fg))
            {
                rq1 = qubitsMap[fg.q1];
                rq2 = qubitsMap[fg.q2];
                if (rq1 > rq2)
                    std::swap(rq1, rq2);
            }

            FastStateSnapshot stepState;
            stepState.SaveFrom(*this);

            // Always evaluate the full heuristic rollout portfolio from stepIdx over
            // [stepIdx, endStep) first. This serves as the leaf evaluation when
            // stepIdx >= exactSteps and preserves the heuristic portfolio's best result
            // as a tight upper-bound baseline before any branching at stepIdx < exactSteps.
            // The full portfolio matters: positional policies (swap left/right, closer
            // to an end or the center) win different decisions than the cost-based
            // ones, and a three-policy subset lost exactness on large QFTs.
            IndexType bestPos = IsRelabelingSwap(fg) ? -1
                                : (rq2 - rq1 <= 1)   ? rq1
                                                     : SelectMeetingPositionP0Fast(rq1, rq2, fg.q1, fg.q2, windowPtr, windowSize, stepIdx, true);

            const double boundBefore = boundCost;
            const IndexType rolloutPos =
                EvaluateHeuristicRolloutsFast(windowPtr, windowSize, stepIdx, static_cast<int>(endStep - stepIdx) - 1, stepStartCost, boundCost);
            if (boundCost < boundBefore)
                bestPos = rolloutPos;

            stepState.RestoreTo(*this);
            if (stepIdx >= static_cast<size_t>(exactSteps))
                return bestPos;

            if (IsRelabelingSwap(fg))
            {
                ApplyFastGate(fg);
                if (stepStartCost < boundCost)
                {
                    if (stepIdx + 1 >= endStep)
                        boundCost = stepStartCost;
                    else
                        (void)self(self, stepIdx + 1, stepStartCost, boundCost);
                }
                stepState.RestoreTo(*this);
                return -1;
            }

            if (rq2 - rq1 <= 1)
            {
                setTotalSwappingCost(0.0);
                ApplyFastGate(fg);
                const double nextCost = stepStartCost + getTotalSwappingCost();
                if (nextCost < boundCost)
                {
                    if (stepIdx + 1 >= endStep)
                        boundCost = nextCost;
                    else
                        (void)self(self, stepIdx + 1, nextCost, boundCost);
                }
                stepState.RestoreTo(*this);
                return rq1;
            }

            const IndexType mP0Grav = SelectMeetingPositionP0Fast(rq1, rq2, fg.q1, fg.q2, windowPtr, windowSize, stepIdx, true);
            const IndexType mP0Pure = SelectMeetingPositionP0Fast(rq1, rq2, fg.q1, fg.q2, windowPtr, windowSize, stepIdx, false);
            const IndexType mAnchor = ComputeAnchorHeavierMeetPosition(rq1, rq2);

            double minBondInSpan = currentBondDim[rq1];
            for (IndexType b = rq1 + 1; b < rq2; ++b)
                minBondInSpan = std::min(minBondInSpan, currentBondDim[b]);

            const double E1 = std::max((rq1 > 0) ? currentBondDim[rq1 - 1] : 1.0, currentBondDim[rq1]);
            const double E2 = std::max(currentBondDim[rq2 - 1], (rq2 < static_cast<IndexType>(currentBondDim.size())) ? currentBondDim[rq2] : 1.0);
            const bool hasLowEntanglementCut =
                (minBondInSpan < 16.0) && (runtimeBondCap <= 0 || minBondInSpan < static_cast<double>(runtimeBondCap)) && (minBondInSpan <= 0.5 * std::min(E1, E2));

            const IndexType span = rq2 - rq1;
            std::array<IndexType, 128> stackCandidates{};
            std::vector<IndexType> heapCandidates;
            IndexType *candidates = stackCandidates.data();
            size_t maxCandidates = stackCandidates.size();
            if (static_cast<size_t>(span) > stackCandidates.size())
            {
                heapCandidates.resize(static_cast<size_t>(span));
                candidates = heapCandidates.data();
                maxCandidates = heapCandidates.size();
            }

            size_t numCandidates = 0;
            auto addCand = [&](IndexType pos) {
                if (pos < rq1 || pos >= rq2)
                    return;
                for (size_t i = 0; i < numCandidates; ++i)
                    if (candidates[i] == pos)
                        return;
                if (numCandidates < maxCandidates)
                    candidates[numCandidates++] = pos;
            };

            addCand(bestPos);
            addCand(mP0Grav);
            if (!hasLowEntanglementCut)
            {
                addCand(mP0Pure);
                addCand(mAnchor);
                addCand(rq1);
                addCand(rq2 - 1);
                addCand(rq1 + span / 2);
                if (span <= 12 || stepIdx > 0 || exactSteps >= 2)
                {
                    for (IndexType m = rq1; m < rq2; ++m)
                        addCand(m);
                }
            }

            for (size_t ci = 0; ci < numCandidates; ++ci)
            {
                const IndexType candPos = candidates[ci];
                stepState.RestoreTo(*this);
                setTotalSwappingCost(0.0);

                SwapQubitsToPosition(fg.q1, fg.q2, candPos);
                ApplyFastGate(fg);
                const double nextCost = stepStartCost + getTotalSwappingCost();
                if (nextCost >= boundCost)
                    continue;

                if (stepIdx + 1 >= endStep)
                {
                    boundCost = nextCost;
                    bestPos = candPos;
                }
                else
                {
                    const double prevBound = boundCost;
                    (void)self(self, stepIdx + 1, nextCost, boundCost);
                    if (boundCost < prevBound)
                        bestPos = candPos;
                }
            }

            stepState.RestoreTo(*this);
            return bestPos;
        };

        double bestRolloutCost = bestCost;
        const IndexType chosenPos = searchAdaptive(searchAdaptive, 0, currentCost, bestRolloutCost);
        if (bestRolloutCost < bestCost)
            bestCost = bestRolloutCost;
        return chosenPos;
    }
    // Swap two logical qubits so they meet at a specified bond position.
    // meetPosition is the bond index (in real/chain coordinates) where
    // the two qubits will end up adjacent: one at meetPosition, the other
    // at meetPosition+1.
    // meetPosition must be in [min(realq1,realq2), max(realq1,realq2)-1].
    void SwapQubitsToPosition(IndexType qubit1, IndexType qubit2, IndexType meetPosition)
    {
        IndexType realq1 = qubitsMap[qubit1];
        IndexType realq2 = qubitsMap[qubit2];
        if (realq1 > realq2)
        {
            std::swap(realq1, realq2);
            std::swap(qubit1, qubit2);
        }

        if (realq2 - realq1 <= 1)
            return;

        assert(meetPosition >= realq1 && meetPosition < realq2);

        // Move lower qubit (qubit1) rightward from realq1 to meetPosition
        {
            IndexType movingReal = realq1;
            while (movingReal < meetPosition)
            {
                const IndexType toReal = movingReal + 1;
                const IndexType toInv = qubitsMapInv[toReal];

                qubitsMap[toInv] = movingReal;
                qubitsMapInv[movingReal] = toInv;

                qubitsMap[qubit1] = toReal;
                qubitsMapInv[toReal] = qubit1;

                totalSwappingCost += bondCost[movingReal];
                growBondDimension(movingReal, true);
                totalSwappingCost += kPostSwapCostWeight * bondCost[movingReal];
                movingReal = toReal;
            }
        }

        // Move upper qubit (qubit2) leftward from realq2 to meetPosition+1
        {
            IndexType movingReal = realq2;
            while (movingReal > meetPosition + 1)
            {
                const IndexType toReal = movingReal - 1;
                const IndexType toInv = qubitsMapInv[toReal];

                qubitsMap[toInv] = movingReal;
                qubitsMapInv[movingReal] = toInv;

                qubitsMap[qubit2] = toReal;
                qubitsMapInv[toReal] = qubit2;

                totalSwappingCost += bondCost[toReal];
                growBondDimension(toReal, true);
                totalSwappingCost += kPostSwapCostWeight * bondCost[toReal];
                movingReal = toReal;
            }
        }

        assert(abs(qubitsMap[qubit1] - qubitsMap[qubit2]) == 1);
    }

  private:
    struct MemoEntry
    {
        uint64_t hash = 0;
        IndexType bestPos = 0;
        double costDelta = 0.0;
        bool valid = false;
    };

    static constexpr size_t kMemoCacheSize = 64;
    std::array<MemoEntry, kMemoCacheSize> memoCache{};

    size_t nrQubits;

    std::vector<IndexType> qubitsMap;
    std::vector<IndexType> qubitsMapInv;

    size_t physExtent = 2; // 2 for an MPS, 4 for an MPO (operator sites)
    IndexType maxVirtualExtent = 0;
    std::vector<double> bondCost;
    IndexType runtimeBondCap = 0;
    double truncationWeight = 0.;
    bool swapGatesRelabelQubits = false;
    std::vector<double> maxBondDim;
    std::vector<double> currentBondDim;

    double totalSwappingCost = 0;

    double growthFactorSwap = 1.;
    double growthFactorGate = 0.7;
    double minLayoutGain = 4.; // set per chain type by the constructor

    void growBondDimension(IndexType bond, bool swap = true, double schmidtRank = 4)
    {
        // the left and right bond dimensions are relevant because:
        // the initial configuration before applying the swap or other gate is:

        // - O - O -
        //   |   |

        // The two physical legs have dimension 2
        // this is contracted into:

        //    ---
        //  -|   |-
        //    ---
        //    | |

        // the left and right dimensions stay the same and also the physical legs have dimension d (2 for an MPS, 4 for an MPO)

        // then the swap or the other gate is applied, getting a result that looks graphically as above, but of course with different values inside the tensor
        // swap is special, just swaps the values for (0, 1) and (1, 0) in
        // the physical legs, while other gates can change all values in the tensor

        // then the tensor is reshaped into a matrix, having dimensions d * leftDim x d * rightNeighborDim on this matrix SVD is applied, to separate out the
        // qubits tensors again, and the bond dimension is the number of singular
        // values kept after truncation (if done), or the number of non-zero
        // singular values if no truncation is done. The bond dimension can be at
        // most min(d * min(leftDim, rightNeighborDim), maxBondDim[bond]) and the minimum is obviously 1

        const IndexType leftBond = bond - 1;
        const IndexType rightNeigborBond = bond + 1;
        const double betweenDim = currentBondDim[bond];

        const double leftDim = leftBond >= 0 ? currentBondDim[leftBond] : 1;
        const double rightNeighborDim = rightNeigborBond < static_cast<IndexType>(currentBondDim.size()) ? currentBondDim[rightNeigborBond] : 1;

        currentBondDim[bond] = EstimateGrownBondDimension(bond, leftDim, betweenDim, rightNeighborDim, swap, schmidtRank);

        bondCost[bond] = BondCost(currentBondDim[bond]);
    }
};

} // namespace Simulators

#endif
