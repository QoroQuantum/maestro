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
#include <random>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "MPSSimulatorInterface.h"
#include "QuantumGate.h"

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
        clone->bondCost = bondCost;
        clone->maxBondDim = maxBondDim;
        clone->currentBondDim = currentBondDim;

        clone->totalSwappingCost = totalSwappingCost;

        clone->growthFactorSwap = growthFactorSwap;
        clone->growthFactorGate = growthFactorGate;

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

    static bool IsControlledTwoQubitGateType(Circuits::QuantumGateType gt)
    {
        switch (gt)
        {
        case Circuits::QuantumGateType::kCXGateType:
        case Circuits::QuantumGateType::kCYGateType:
        case Circuits::QuantumGateType::kCZGateType:
        case Circuits::QuantumGateType::kCPGateType:
        case Circuits::QuantumGateType::kCRxGateType:
        case Circuits::QuantumGateType::kCRyGateType:
        case Circuits::QuantumGateType::kCRzGateType:
        case Circuits::QuantumGateType::kCHGateType:
        case Circuits::QuantumGateType::kCSxGateType:
        case Circuits::QuantumGateType::kCSxDagGateType:
        case Circuits::QuantumGateType::kCUGateType:
            return true;
        default:
            return false;
        }
    }

    void InvalidateMemoCache()
    {
        for (auto &entry : memoCache)
            entry.valid = false;
    }

    void setGrowthFactorSwap(double factor)
    {
        growthFactorSwap = factor;
        InvalidateMemoCache();
    }

    void setGrowthFactorGate(double factor)
    {
        growthFactorGate = factor;
        InvalidateMemoCache();
    }

    void Clear()
    {
        InitQubitsMap();
        InvalidateMemoCache();
    }

    void SetMaxBondDimension(IndexType val)
    {
        maxVirtualExtent = val;
        InvalidateMemoCache();

        if (nrQubits == 0)
            return;

        const double untruncatedMaxExtent = std::pow(physExtent, nrQubits / 2);
        IndexType maxVirtualExtentLimit = static_cast<IndexType>(untruncatedMaxExtent);

        if (untruncatedMaxExtent >= std::numeric_limits<IndexType>::max() || std::isnan(untruncatedMaxExtent) || std::isinf(untruncatedMaxExtent))
            maxVirtualExtentLimit = std::numeric_limits<IndexType>::max() - 1;
        else if (untruncatedMaxExtent < 2)
            maxVirtualExtentLimit = 2;

        maxVirtualExtent = maxVirtualExtent == 0 ? maxVirtualExtentLimit : std::min(maxVirtualExtent, maxVirtualExtentLimit);

        bondCost.assign(nrQubits - 1, 1.0);
        maxBondDim.resize(nrQubits - 1);
        currentBondDim.assign(nrQubits - 1, 1.0);

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
            static const QC::Gates::CNOTGate<MatrixClass> dummyControlled2qGate;
            static const QC::Gates::TwoQubitsGate<MatrixClass> dummyGeneric2qGate;

            const auto gateptr = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(gate);
            if (gateptr && gateptr->GetGateType() == Circuits::QuantumGateType::kSwapGateType)
                ApplyGate(dummySwapGate, qbits[0], qbits[1]);
            else if (gateptr && IsControlledTwoQubitGateType(gateptr->GetGateType()))
                ApplyGate(dummyControlled2qGate, qbits[0], qbits[1]);
            else
                ApplyGate(dummyGeneric2qGate, qbits[0], qbits[1]);
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
            const IndexType bond = std::min(qubit1, qubit2);
            totalSwappingCost += bondCost[bond];
            // TODO: This is basic, the two qubit gates can have rank 2 and 4 (1 if
            // can be decomposed into two 1-qubit gates)
            // the controlled ones have Schmidt rank 2
            // here we assume that the others have Schmidt rank 4,
            // but that might not be the case for all gates
            // (3 is not possible for two qubit gates)
            growBondDimension(bond, gate.isSwapGate(), gate.isControlled() ? 2 : 4);
            totalSwappingCost += kPostGateCostWeight * bondCost[bond];
        }
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

    void SetInitialQubitsMap(const std::vector<long long int> &initialMap)
    {
        qubitsMap = initialMap;
        for (size_t i = 0; i < initialMap.size(); ++i)
            qubitsMapInv[initialMap[i]] = i;

        totalSwappingCost = 0;
        std::fill(currentBondDim.begin(), currentBondDim.end(), 1.0);
        for (size_t i = 0; i < bondCost.size(); ++i)
            bondCost[i] = 1;
    }

    void SetCurrentBondDimensions(const std::vector<double> &dims)
    {
        assert(dims.size() == currentBondDim.size());
        for (size_t i = 0; i < dims.size(); ++i)
            currentBondDim[i] = std::min(dims[i], maxBondDim[i]);

        // update bondCost as well, since it depends on the current bond dimensions
        for (size_t i = 0; i < currentBondDim.size(); ++i)
            bondCost[i] = currentBondDim[i] * currentBondDim[i] * currentBondDim[i];
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
            return 0;
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
            return 0;
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
            mixHash(static_cast<uint64_t>(fg.schmidtRank));
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

        for (size_t li = 0; li < layers.size(); ++li)
        {
            const auto &layer = layers[li];
            std::vector<QubitPair> lp;
            for (const auto &op : layer->GetOperations())
            {
                auto qbits = op->AffectedQubits();
                if (qbits.size() >= 2)
                {
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

        auto optMap = buildChain(layerPairs);
        if (layers.size() <= 2)
            return optMap;
        auto optCost = evaluateCost(optMap);

        std::vector<long long int> qubitsMap(nrQubits);
        std::iota(qubitsMap.begin(), qubitsMap.end(), 0);
        double tryCost = evaluateCostBounded(qubitsMap, optCost);
        if (tryCost < optCost)
        {
            optCost = tryCost;
            optMap = qubitsMap;
        }

        // try some random shuffles as well, in case the heuristic ordering is not
        // optimal
        std::mt19937 rng(42);
        for (int i = 0; i < nrShuffles; ++i)
        {
            std::shuffle(qubitsMap.begin(), qubitsMap.end(), rng);
            tryCost = evaluateCostBounded(qubitsMap, optCost);
            if (tryCost < optCost)
            {
                optCost = tryCost;
                optMap = qubitsMap;
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
            const int nrSwaps = nrSwapsDist(rng);
            for (int sw = 0; sw < nrSwaps; ++sw)
            {
                const IndexType a = qubitDist(rng);
                IndexType b = qubitDist(rng);
                while (b == a)
                    b = qubitDist(rng);
                std::swap(tryMap[a], tryMap[b]);
            }

            auto cost = evaluateCostBounded(tryMap, optCost);
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

        // 2-opt local search: iteratively swap pairs of positions in the best map
        // and keep improvements, until no single swap can reduce the cost
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
                        // swap the mapped positions of qubits i and j
                        std::swap(candidate[i], candidate[j]);
                        auto cost = evaluateCostBounded(candidate, optCost);
                        if (cost < optCost)
                        {
                            optMap = candidate;
                            optCost = cost;
                            improved = true;
                        }
                        else
                        {
                            // revert
                            std::swap(candidate[i], candidate[j]);
                        }
                    }
                }
            }
        }

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

    // Pick the meeting position with the lowest bond cost (i.e., lowest
    // current bond dimension), matching the real MPS simulator's
    // FindBestMeetingPositionLocal behavior.
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

    double EstimateGrownBondDimension(IndexType bond, double leftDim, double betweenDim, double rightNeighborDim, bool swap, int schmidtRank) const
    {
        const double rank = IsOperatorChain() ? static_cast<double>(schmidtRank) * schmidtRank : static_cast<double>(schmidtRank);
        double newMaxDim = (swap && leftDim == rightNeighborDim) ? betweenDim : static_cast<double>(physExtent) * std::min(leftDim, rightNeighborDim);
        newMaxDim = std::min(newMaxDim, betweenDim * rank);
        const double growthFactor = swap ? growthFactorSwap : growthFactorGate;
        const double grown = std::min(newMaxDim * growthFactor, maxBondDim[bond]);
        return std::max(grown, 1.);
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
        int schmidtRank = 4;
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
                if (gateptr && gateptr->GetGateType() == Circuits::QuantumGateType::kSwapGateType)
                {
                    fg.isSwap = true;
                    fg.schmidtRank = 4;
                }
                else if (gateptr && IsControlledTwoQubitGateType(gateptr->GetGateType()))
                {
                    fg.schmidtRank = 2;
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

    void ApplyTwoQubitFastGate(IndexType qubit, IndexType controllingQubit1, bool isSwap, int schmidtRank)
    {
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

        for (size_t s = stepIdx + 1; s < maxStep; ++s)
        {
            const FastGateOp &fg = window[s];
            const IndexType qA = fg.q1;
            const IndexType qB = fg.q2;

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
        const int schmidtRank = fg.schmidtRank;

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
            prefixPostCost[i + 1] = prefixPostCost[i] + newD * newD * newD;
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
            suffixPostCost[i - 1] = suffixPostCost[i] + newD * newD * newD;
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
            const double gateCost = gateD * gateD * gateD;
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
        IndexType bestFirstPosition = (firstReal2 - firstReal1 > 1)
                                          ? SelectMeetPositionForPolicyFast(defaultFirstPolicy, firstReal1, firstReal2, firstQ1, firstQ2, window, windowSize,
                                                                            stepIdx)
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

                IndexType meetPos = rq1;
                if (rq2 - rq1 > 1)
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

        if (realq2 - realq1 > 1)
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

  public:
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
    std::vector<double> maxBondDim;
    std::vector<double> currentBondDim;

    double totalSwappingCost = 0;

    double growthFactorSwap = 1.;
    double growthFactorGate = 0.7;

    void growBondDimension(IndexType bond, bool swap = true, int schmidtRank = 4)
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

        // an MPO site has physical dimension 4, and a gate acts on it as
        // rho -> U rho U^dagger, whose operator Schmidt rank is the square of the
        // gate's (the swap special case still holds, it only relabels the sites)
        const double rank = IsOperatorChain() ? static_cast<double>(schmidtRank) * schmidtRank : static_cast<double>(schmidtRank);
        double newMaxDim = (swap && leftDim == rightNeighborDim) ? betweenDim : static_cast<double>(physExtent) * std::min(leftDim, rightNeighborDim);
        newMaxDim = std::min(newMaxDim, betweenDim * rank);

        const double growthFactor = swap ? growthFactorSwap : growthFactorGate;

        currentBondDim[bond] = std::min(newMaxDim * growthFactor, maxBondDim[bond]);

        currentBondDim[bond] = std::max(currentBondDim[bond], 1.);

        bondCost[bond] = currentBondDim[bond] * currentBondDim[bond] * currentBondDim[bond];
    }
};

} // namespace Simulators

#endif
