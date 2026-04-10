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
#include <deque>
#include <random>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "QuantumGate.h"
#include "MPSSimulatorInterface.h"

namespace Simulators {

class MPSDummySimulator {
 public:
  using IndexType = long long int;
  using MatrixClass = QC::TensorNetworks::MPSSimulatorInterface::MatrixClass;
  using GateClass = QC::TensorNetworks::MPSSimulatorInterface::GateClass;

  MPSDummySimulator(size_t N) : nrQubits(N), maxVirtualExtent(0) {
    InitQubitsMap();

    SetMaxBondDimension(0);
  }

  std::unique_ptr<MPSDummySimulator> Clone() const {
    auto clone = std::unique_ptr<MPSDummySimulator>(
        new MPSDummySimulator(nrQubits, LightweightInitTag{}));
    clone->qubitsMap = qubitsMap;
    clone->qubitsMapInv = qubitsMapInv;
    clone->maxVirtualExtent = maxVirtualExtent;
    clone->bondCost = bondCost;
    clone->maxBondDim = maxBondDim;
    clone->currentBondDim = currentBondDim;

    clone->totalSwappingCost = totalSwappingCost;

    return clone;
  }

  size_t getNrQubits() const { return nrQubits; }

  void Clear() { InitQubitsMap(); }

  void SetMaxBondDimension(IndexType val) {
    maxVirtualExtent = val;

    if (nrQubits == 0) return;

    const double untruncatedMaxExtent = std::pow(physExtent, nrQubits / 2);
    IndexType maxVirtualExtentLimit =
        static_cast<IndexType>(untruncatedMaxExtent);

    if (untruncatedMaxExtent >= std::numeric_limits<IndexType>::max() ||
        std::isnan(untruncatedMaxExtent) || std::isinf(untruncatedMaxExtent))
      maxVirtualExtentLimit = std::numeric_limits<IndexType>::max() - 1;
    else if (untruncatedMaxExtent < 2)
      maxVirtualExtentLimit = 2;

    maxVirtualExtent = maxVirtualExtent == 0
                           ? maxVirtualExtentLimit
                           : std::min(maxVirtualExtent, maxVirtualExtentLimit);

    bondCost.resize(nrQubits - 1);
    maxBondDim.resize(nrQubits - 1);
    currentBondDim.assign(nrQubits - 1, 1.0);

    // the checks here are overkill, but better safe than sorry
    // we're dealing with large values here, overflows are to be expected
    for (IndexType i = 0; i < static_cast<IndexType>(nrQubits); ++i) {
      double maxExtent1 = std::pow((double)physExtent, (double)i + 1.);
      double maxExtent2 =
          std::pow((double)physExtent, (double)nrQubits - i - 1.);

      if (maxExtent1 >= (double)std::numeric_limits<size_t>::max() ||
          std::isnan(maxExtent1) || std::isinf(maxExtent1))
        maxExtent1 = (double)std::numeric_limits<size_t>::max() - 1;
      else if (maxExtent1 < 1)
        maxExtent1 = 1;

      if (maxExtent2 > (double)std::numeric_limits<size_t>::max() ||
          std::isnan(maxExtent2) || std::isinf(maxExtent2))
        maxExtent2 = (double)std::numeric_limits<size_t>::max() - 1;
      else if (maxExtent2 < 1)
        maxExtent2 = 1;

      size_t maxRightExtent = (size_t)std::min<double>(
          {maxExtent1, maxExtent2, (double)maxVirtualExtent});
      if (maxRightExtent < 2) maxRightExtent = 2;

      if (i < static_cast<IndexType>(nrQubits) - 1) {
        maxBondDim[i] = static_cast<double>(maxRightExtent);
        const double maxRightExtentD = static_cast<double>(maxRightExtent);
        bondCost[i] = maxRightExtentD * maxRightExtentD * maxRightExtentD;
      }
    }
  }

  void print() const {
    std::cout << "Qubits map: ";
    for (int q = 0; q < static_cast<int>(qubitsMap.size()); ++q)
      std::cout << q << "->" << qubitsMap[q] << " ";
    std::cout << std::endl;
  }

  void ApplyGate(const QC::Gates::AppliedGate<MatrixClass>& gate) {
    ApplyGate(gate, gate.getQubit1(), gate.getQubit2());
  }

  void ApplyGate(const std::shared_ptr<Circuits::IOperation<>>& gate) {
    const auto qbits = gate->AffectedQubits();

    if (qbits.size() == 1) {
      // 1-qubit gates are no-ops in the dummy simulator (no swap logic)
      return;
    } else if (qbits.size() == 2) {
      // Reuse a single static dummy 2-qubit gate to avoid heap allocation.
      // Only getQubitsNumber() is checked; the matrix is never read.
      static const QC::Gates::AppliedGate<MatrixClass> dummy2qGate(
          MatrixClass::Identity(4, 4), 0, 1);
      ApplyGate(dummy2qGate, qbits[0], qbits[1]);
    } else {
      throw std::invalid_argument("Unsupported number of qubits for the gate");
    }
  }

  void ApplyGate(const GateClass& gate, IndexType qubit,
                 IndexType controllingQubit1 = 0) {
    if (qubit < 0 || qubit >= static_cast<IndexType>(getNrQubits()))
      throw std::invalid_argument("Qubit index out of bounds");
    else if (controllingQubit1 < 0 ||
             controllingQubit1 >= static_cast<IndexType>(getNrQubits()))
      throw std::invalid_argument("Qubit index out of bounds");

    // for two qubit gates:
    // if the qubits are not adjacent, apply swap gates until they are
    // don't forget to update the qubitsMap
    if (gate.getQubitsNumber() > 1) {
      IndexType qubit1 = qubitsMap[qubit];
      IndexType qubit2 = qubitsMap[controllingQubit1];

      if (abs(qubit1 - qubit2) > 1) {
        SwapQubits(qubit, controllingQubit1);
        qubit1 = qubitsMap[qubit];
        qubit2 = qubitsMap[controllingQubit1];
        assert(abs(qubit1 - qubit2) == 1);
      }

      // any 2-qubit gate (whether swaps were needed or not) grows the
      // bond dimension at the bond between the two adjacent qubits
      const IndexType bond = std::min(qubit1, qubit2);
      growBondDimension(bond, false);
      totalSwappingCost += bondCost[bond];
    }
  }

  void ApplyGates(
      const std::vector<QC::Gates::AppliedGate<MatrixClass>>& gates) {
    for (const auto& gate : gates) ApplyGate(gate);
  }

  void ApplyGates(
      const std::vector<std::shared_ptr<Circuits::IOperation<>>>& gates) {
    for (const auto& gate : gates) ApplyGate(gate);
  }

  void SetInitialQubitsMap(const std::vector<long long int>& initialMap) {
    qubitsMap = initialMap;
    for (size_t i = 0; i < initialMap.size(); ++i)
      qubitsMapInv[initialMap[i]] = i;

    totalSwappingCost = 0;
    std::fill(currentBondDim.begin(), currentBondDim.end(), 1.0);
    for (size_t i = 0; i < bondCost.size(); ++i) bondCost[i] = 1;
  }

  void SetCurrentBondDimensions(const std::vector<double>& dims) {
    assert(dims.size() == currentBondDim.size());
    for (size_t i = 0; i < dims.size(); ++i)
      currentBondDim[i] = std::min(dims[i], maxBondDim[i]);

    // update bondCost as well, since it depends on the current bond dimensions
    for (size_t i = 0; i < currentBondDim.size(); ++i)
      bondCost[i] = currentBondDim[i] * currentBondDim[i] * currentBondDim[i];
  }

  const std::vector<double>& getCurrentBondDimensions() const {
    return currentBondDim;
  }

  const std::vector<double>& getMaxBondDimensions() const { return maxBondDim; }

  const std::vector<IndexType>& getQubitsMap() const { return qubitsMap; }

  const std::vector<IndexType>& getQubitsMapInv() const { return qubitsMapInv; }

  void setTotalSwappingCost(double cost) { totalSwappingCost = cost; }
  double getTotalSwappingCost() const { return totalSwappingCost; }

  // Evaluate the total cost
  // position, applying the current 2-qubit gate, and then simulating the
  // next lookaheadDepth 2-qubit gates from upcomingGates.
  void EvaluateMeetingPositionCost(
      IndexType meetPosition,
      const std::vector<std::shared_ptr<Circuits::IOperation<>>>& upcomingGates,
      long long int currentGateIndex, int lookaheadDepth,
      int lookaheadDepthWithHeuristic, double currentCost, double& bestCost,
      bool useSameDummy = false) {
    if (currentGateIndex >= static_cast<long long int>(upcomingGates.size())) {
      if (currentCost < bestCost) bestCost = currentCost;
      return;
    }

    // skip the 1 qubit gates, advance to the next 2-qubit gate
    while (currentGateIndex <
               static_cast<long long int>(upcomingGates.size()) &&
           upcomingGates[currentGateIndex]->AffectedQubits().size() < 2)
      ++currentGateIndex;

    if (currentGateIndex >= static_cast<long long int>(upcomingGates.size())) {
      if (currentCost < bestCost) bestCost = currentCost;
      return;
    }

    const auto& op = upcomingGates[currentGateIndex];
    const auto qbits = op->AffectedQubits();

    assert(qbits.size() >= 2);

    const IndexType qubit1 = static_cast<IndexType>(qbits[0]);
    const IndexType qubit2 = static_cast<IndexType>(qbits[1]);

    // Perform the swap to the meeting position;
    // totalSwappingCost starts at 0 and SwapQubitsToPosition accumulates
    // per-bond costs, so the cost correctly depends on meetPosition.

    IndexType realq1 = qubitsMap[qubit1];
    IndexType realq2 = qubitsMap[qubit2];
    if (realq1 > realq2) std::swap(realq1, realq2);

    if (useSameDummy) {
      const double ccost = getTotalSwappingCost();

      if (realq2 - realq1 > 1)
        SwapQubitsToPosition(qubit1, qubit2, meetPosition);

      ApplyGate(op);

      // const double importanceFactor =
      //     pow(dimFactor, (lookaheadDepthWithHeuristic - lookaheadDepth - 1));
      currentCost += (getTotalSwappingCost() - ccost) /** importanceFactor*/;

      if (currentCost >= bestCost) return;
      // No more lookahead depth: return without further recursion
      if (lookaheadDepth <= 0) {
        if (currentCost < bestCost) bestCost = currentCost;
        return;
      }

      // skip the 1 qubit gates, advance over the current gate on the next
      // 2-qubit gate
      ++currentGateIndex;
      while (currentGateIndex <
                 static_cast<long long int>(upcomingGates.size()) &&
             upcomingGates[currentGateIndex]->AffectedQubits().size() < 2)
        ++currentGateIndex;

      if (currentGateIndex >=
          static_cast<long long int>(upcomingGates.size())) {
        if (currentCost < bestCost) bestCost = currentCost;
        return;
      }

      FindBestMeetingPosition(upcomingGates, currentGateIndex,
                              lookaheadDepth - 1, lookaheadDepthWithHeuristic,
                              currentCost, bestCost);
    } else {
      MPSDummySimulator dummySim(nrQubits, LightweightInitTag{});
      dummySim.maxVirtualExtent = maxVirtualExtent;
      dummySim.maxBondDim = maxBondDim;
      dummySim.currentBondDim = currentBondDim;
      dummySim.bondCost = bondCost;
      dummySim.qubitsMap = qubitsMap;
      dummySim.qubitsMapInv.resize(qubitsMap.size());
      for (size_t i = 0; i < qubitsMap.size(); ++i)
        dummySim.qubitsMapInv[qubitsMap[i]] = static_cast<IndexType>(i);

      dummySim.setTotalSwappingCost(0);

      if (realq2 - realq1 > 1)
        dummySim.SwapQubitsToPosition(qubit1, qubit2, meetPosition);

      dummySim.ApplyGate(op);  // this also updates the bond dimensions in
                               // dummySim according to the applied gate

      currentCost += dummySim.getTotalSwappingCost();

      // Pruning: if current accumulated cost already exceeds best known, prune
      if (currentCost >= bestCost) return;

      // No more lookahead depth: return without further recursion
      if (lookaheadDepth <= 0) {
        if (currentCost < bestCost) bestCost = currentCost;
        return;
      }

      // skip the 1 qubit gates, advance over the current gate on the next
      // 2-qubit gate
      ++currentGateIndex;
      while (currentGateIndex <
                 static_cast<long long int>(upcomingGates.size()) &&
             upcomingGates[currentGateIndex]->AffectedQubits().size() < 2)
        ++currentGateIndex;

      if (currentGateIndex >=
          static_cast<long long int>(upcomingGates.size())) {
        if (currentCost < bestCost) bestCost = currentCost;
        return;
      }

      dummySim.FindBestMeetingPosition(
          upcomingGates, currentGateIndex, lookaheadDepth - 1,
          lookaheadDepthWithHeuristic, currentCost, bestCost);
    }
  }

  // Find the meeting position that minimizes the combined cost of the
  // current swap + lookahead gates.  Returns the optimal bond index.
  // outBestCost receives the total accumulated cost for the best position.
  IndexType FindBestMeetingPosition(
      const std::vector<std::shared_ptr<Circuits::IOperation<>>>& upcomingGates,
      long long int currentGateIndex, int lookaheadDepth,
      int lookaheadDepthWithHeuristic, double currentCost, double& bestCost) {
    const auto& op = upcomingGates[currentGateIndex];
    const auto qbits = op->AffectedQubits();

    assert(qbits.size() >= 2);

    const IndexType qubit1 = static_cast<IndexType>(qbits[0]);
    const IndexType qubit2 = static_cast<IndexType>(qbits[1]);

    IndexType realq1 = qubitsMap[qubit1];
    IndexType realq2 = qubitsMap[qubit2];

    if (realq1 > realq2) std::swap(realq1, realq2);

    if (lookaheadDepth <= 0) {
      IndexType pos = (realq2 - realq1 > 1)
                          ? ComputeHeuristicMeetPosition(realq1, realq2)
                          : realq1;
      EvaluateMeetingPositionCost(pos, upcomingGates, currentGateIndex, 0,
                                  lookaheadDepthWithHeuristic, currentCost,
                                  bestCost, false);
      return pos;
    }

    if (realq2 - realq1 <= 1) {
      EvaluateMeetingPositionCost(
          realq1, upcomingGates, currentGateIndex, lookaheadDepth,
          lookaheadDepthWithHeuristic, currentCost, bestCost,
          lookaheadDepth <= lookaheadDepthWithHeuristic);

      return realq1;
    }

    IndexType bestPosition = realq1;

    if (lookaheadDepth <= lookaheadDepthWithHeuristic) {
      bestPosition = ComputeHeuristicMeetPosition(realq1, realq2);
      EvaluateMeetingPositionCost(bestPosition, upcomingGates, currentGateIndex,
                                  lookaheadDepth, lookaheadDepthWithHeuristic,
                                  currentCost, bestCost, true);
    } else {
      for (IndexType m = realq1; m < realq2; ++m) {
        const double oldCost = bestCost;
        EvaluateMeetingPositionCost(m, upcomingGates, currentGateIndex,
                                    lookaheadDepth, lookaheadDepthWithHeuristic,
                                    currentCost, bestCost);

        if (bestCost < oldCost) bestPosition = m;
      }
    }

    return bestPosition;
  }

  std::vector<long long int> ComputeOptimalQubitsMap(
      const std::vector<std::shared_ptr<Circuits::Circuit<>>>& layers,
      int nrShuffles = 10) {
    const IndexType nrQubits = getNrQubits();

    if (layers.empty() || nrQubits <= 2) return qubitsMap;

    
    auto evaluateCost =
        [&, this](const std::vector<IndexType>& candidateMap) -> double {
      auto saveQubitsMap = qubitsMap;
      auto saveQubitsMapInv = qubitsMapInv;
      auto saveCurrentBondDim = currentBondDim;
      auto saveTotalSwappingCost = totalSwappingCost;
      auto saveBondCost = bondCost;

      SetInitialQubitsMap(candidateMap);

      for (const auto& layer : layers) 
          ApplyGates(layer->GetOperations());
      auto cost = getTotalSwappingCost();
      qubitsMap = std::move(saveQubitsMap);
      qubitsMapInv = std::move(saveQubitsMapInv);
      currentBondDim = std::move(saveCurrentBondDim);
      totalSwappingCost = saveTotalSwappingCost;
      bondCost = std::move(saveBondCost);

      return cost;
    };

    auto evaluateCostBounded = [&, this](
                                   const std::vector<IndexType>& candidateMap,
                                   double bound) -> double {
      auto saveQubitsMap = qubitsMap;
      auto saveQubitsMapInv = qubitsMapInv;
      auto saveCurrentBondDim = currentBondDim;
      auto saveTotalSwappingCost = totalSwappingCost;
      auto saveBondCost = bondCost;

      SetInitialQubitsMap(candidateMap);

      for (const auto& layer : layers) {
        ApplyGates(layer->GetOperations());
        auto cost = getTotalSwappingCost();
        if (cost >= bound) {
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
    struct QubitPair {
      IndexType q1, q2;
    };
    std::vector<std::vector<QubitPair>> layerPairs;

    for (size_t li = 0; li < layers.size(); ++li) {
      const auto& layer = layers[li];
      std::vector<QubitPair> lp;
      for (const auto& op : layer->GetOperations()) {
        auto qbits = op->AffectedQubits();
        if (qbits.size() >= 2) {
          if (qbits[0] > qbits[1]) std::swap(qbits[0], qbits[1]);

          lp.push_back({static_cast<IndexType>(qbits[0]),
                        static_cast<IndexType>(qbits[1])});
        }
      }
      if (!lp.empty()) layerPairs.push_back(std::move(lp));
    }
    if (layerPairs.empty()) return qubitsMap;

    // Build a linear qubit chain from ordered pairs using a group-merging
    // strategy.  Pairs are processed in order (earlier = more important).
    // - If neither qubit is placed, create a new group [q1, q2].
    // - If one qubit is already in a group, add the other at the group
    //   end (front or back) that minimizes distance to the existing one.
    // - If both are in different groups, merge as [gA][gB] or [gB][gA],
    //   whichever places q1 and q2 closer together.
    // - If both are already in the same group, do nothing.
    auto buildChain = [&](const std::vector<std::vector<QubitPair>>& orderedLP)
        -> std::vector<long long int> {
      std::vector<std::deque<IndexType>> groups;
      std::vector<int> qubitGroup(nrQubits, -1);
      std::unordered_set<IndexType> placedQubits;

      for (const auto& lp : orderedLP) {
        for (const auto& p : lp) {
          const int g1 = qubitGroup[p.q1];
          const int g2 = qubitGroup[p.q2];

          if (g1 < 0 && g2 < 0) {
            // Neither placed: create a new group
            const int gIdx = static_cast<int>(groups.size());
            groups.push_back({p.q1, p.q2});
            qubitGroup[p.q1] = gIdx;
            qubitGroup[p.q2] = gIdx;
            placedQubits.insert(p.q1);
            placedQubits.insert(p.q2);
          } else if (g1 >= 0 && g2 < 0) {
            // q1 placed, q2 not: add q2 at the closer end of q1's group
            auto& grp = groups[g1];
            size_t idx = 0;
            for (size_t i = 0; i < grp.size(); ++i)
              if (grp[i] == p.q1) {
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
          } else if (g1 < 0 && g2 >= 0) {
            // q2 placed, q1 not: add q1 at the closer end of q2's group
            auto& grp = groups[g2];
            size_t idx = 0;
            for (size_t i = 0; i < grp.size(); ++i)
              if (grp[i] == p.q2) {
                idx = i;
                break;
              }
            if (idx + 1 <= grp.size() - idx)
              grp.push_front(p.q1);
            else
              grp.push_back(p.q1);
            qubitGroup[p.q1] = g2;
            placedQubits.insert(p.q1);
          } else if (g1 != g2) {
            // Both in different groups: merge to minimize distance
            auto& grpA = groups[g1];
            auto& grpB = groups[g2];
            size_t idxA = 0, idxB = 0;
            for (size_t i = 0; i < grpA.size(); ++i)
              if (grpA[i] == p.q1) {
                idxA = i;
                break;
              }
            for (size_t i = 0; i < grpB.size(); ++i)
              if (grpB[i] == p.q2) {
                idxB = i;
                break;
              }
            // [A][B]: q1 at idxA, q2 at |A|+idxB -> dist = |A|+idxB-idxA
            // [B][A]: q2 at idxB, q1 at |B|+idxA -> dist = |B|+idxA-idxB
            const size_t distAB = grpA.size() + idxB - idxA;
            const size_t distBA = grpB.size() + idxA - idxB;

            int mergedGroup;
            if (distAB <= distBA) {
              for (const auto q : grpB) grpA.push_back(q);
              grpB.clear();
              mergedGroup = g1;
            } else {
              for (const auto q : grpA) grpB.push_back(q);
              grpA.clear();
              mergedGroup = g2;
            }
            for (const auto q : groups[mergedGroup])
              qubitGroup[q] = mergedGroup;
          }
          // Both in the same group: do nothing

          if (placedQubits.size() == static_cast<size_t>(nrQubits))
            break;  // all qubits placed, can stop processing pairs
        }

        if (placedQubits.size() == static_cast<size_t>(nrQubits))
          break;  // all qubits placed, can stop processing pairs
      }

      // Concatenate all non-empty groups
      std::vector<IndexType> chain;
      chain.reserve(nrQubits);
      for (const auto& grp : groups)
        for (const auto q : grp) chain.push_back(q);

      // Append any remaining unplaced qubits
      for (IndexType q = 0; q < nrQubits; ++q)
        if (qubitGroup[q] < 0) chain.push_back(q);

      assert(chain.size() == static_cast<size_t>(nrQubits));

      // Convert chain to qubitsMap: chain[physPos] = logicalQubit
      std::vector<long long int> result(nrQubits);
      for (size_t i = 0; i < chain.size(); ++i)
        result[chain[i]] = static_cast<long long int>(i);

      return result;
    };

    auto optMap = buildChain(layerPairs);
    if (layers.size() <= 2) return optMap;
    auto optCost = evaluateCost(optMap);

    std::vector<long long int> qubitsMap(nrQubits);
    std::iota(qubitsMap.begin(), qubitsMap.end(), 0);
    double tryCost = evaluateCostBounded(qubitsMap, optCost);
    if (tryCost < optCost) {
      optCost = tryCost;
      optMap = qubitsMap;
    }

    // try some random shuffles as well, in case the heuristic ordering is not
    // optimal
    std::mt19937 rng(42);
    for (int i = 0; i < nrShuffles; ++i) {
      std::shuffle(qubitsMap.begin(), qubitsMap.end(), rng);
      tryCost = evaluateCostBounded(qubitsMap, optCost);
      if (tryCost < optCost) {
        optCost = tryCost;
        optMap = qubitsMap;
      }
    }


    std::uniform_int_distribution<IndexType> qubitDist(0, nrQubits - 1);
    std::uniform_int_distribution<int> nrSwapsDist(
        1, std::min<int>(3, static_cast<int>(nrQubits) / 2));

    const int maxNoImprove = std::max(nrShuffles, static_cast<int>(nrQubits));
    const int maxTotalShuffles = maxNoImprove * 3;
    int noImproveCount = 0;

    for (int s = 0; s < maxTotalShuffles && noImproveCount < maxNoImprove; ++s) {
      auto tryMap = optMap;
      const int nrSwaps = nrSwapsDist(rng);
      for (int sw = 0; sw < nrSwaps; ++sw) {
        const IndexType a = qubitDist(rng);
        IndexType b = qubitDist(rng);
        while (b == a) b = qubitDist(rng);
        std::swap(tryMap[a], tryMap[b]);
      }

      auto cost = evaluateCostBounded(tryMap, optCost);
      if (cost < optCost) {
        optMap = tryMap;
        optCost = cost;
        noImproveCount = 0;
      } else {
        ++noImproveCount;
      }
    }

    // 2-opt local search: iteratively swap pairs of positions in the best map
    // and keep improvements, until no single swap can reduce the cost
    {
      auto candidate = optMap;
      bool improved = true;
      while (improved) {
        improved = false;
        for (IndexType i = 0; i < nrQubits; ++i) {
          for (IndexType j = i + 1; j < nrQubits; ++j) {
            // swap the mapped positions of qubits i and j
            std::swap(candidate[i], candidate[j]);
            auto cost = evaluateCostBounded(candidate, optCost);
            if (cost < optCost) {
              optMap = candidate;
              optCost = cost;
              improved = true;
            } else {
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
  void InitQubitsMap() {
    qubitsMap.resize(getNrQubits());
    qubitsMapInv.resize(getNrQubits());

    for (IndexType i = 0; i < static_cast<IndexType>(getNrQubits()); ++i)
      qubitsMapInv[i] = qubitsMap[i] = i;

    totalSwappingCost = 0;
    if (!currentBondDim.empty())
      std::fill(currentBondDim.begin(), currentBondDim.end(), 1.0);
  }

  struct LightweightInitTag {};

  // Lightweight constructor: sets up qubit maps but skips the expensive
  // SetMaxBondDimension computation.  Caller must populate bond arrays.
  MPSDummySimulator(size_t N, LightweightInitTag)
      : nrQubits(N) {
  }

  void SwapQubits(IndexType qubit1, IndexType qubit2) {
    IndexType realq1 = qubitsMap[qubit1];
    IndexType realq2 = qubitsMap[qubit2];
    if (realq1 > realq2) {
      std::swap(realq1, realq2);
      std::swap(qubit1, qubit2);
    }

    if (realq2 - realq1 <= 1) return;

    const IndexType meetPos = ComputeHeuristicMeetPosition(realq1, realq2);
    SwapQubitsToPosition(qubit1, qubit2, meetPos);
  }

  // Pick the meeting position with the lowest bond cost (i.e., lowest
  // current bond dimension), matching the real MPS simulator's
  // FindBestMeetingPositionLocal behavior.
  IndexType ComputeHeuristicMeetPosition(IndexType realq1,
                                         IndexType realq2) const {
    assert(realq1 < realq2);

    IndexType bestPos = realq1;
    double bestCost = bondCost[realq1];

    for (IndexType m = realq1 + 1; m < realq2; ++m) {
      if (bondCost[m] < bestCost) {
        bestCost = bondCost[m];
        bestPos = m;
      }
    }

    return bestPos;
  }

 public:
  // Swap two logical qubits so they meet at a specified bond position.
  // meetPosition is the bond index (in real/chain coordinates) where
  // the two qubits will end up adjacent: one at meetPosition, the other
  // at meetPosition+1.
  // meetPosition must be in [min(realq1,realq2), max(realq1,realq2)-1].
  void SwapQubitsToPosition(IndexType qubit1, IndexType qubit2,
                            IndexType meetPosition) {
    IndexType realq1 = qubitsMap[qubit1];
    IndexType realq2 = qubitsMap[qubit2];
    if (realq1 > realq2) {
      std::swap(realq1, realq2);
      std::swap(qubit1, qubit2);
    }

    if (realq2 - realq1 <= 1) return;

    assert(meetPosition >= realq1 && meetPosition < realq2);

    // Move lower qubit (qubit1) rightward from realq1 to meetPosition
    {
      IndexType movingReal = realq1;
      while (movingReal < meetPosition) {
        const IndexType toReal = movingReal + 1;
        const IndexType toInv = qubitsMapInv[toReal];

        qubitsMap[toInv] = movingReal;
        qubitsMapInv[movingReal] = toInv;

        qubitsMap[qubit1] = toReal;
        qubitsMapInv[toReal] = qubit1;

        totalSwappingCost += bondCost[movingReal];
        growBondDimension(movingReal, true);
        movingReal = toReal;
      }
    }

    // Move upper qubit (qubit2) leftward from realq2 to meetPosition+1
    {
      IndexType movingReal = realq2;
      while (movingReal > meetPosition + 1) {
        const IndexType toReal = movingReal - 1;
        const IndexType toInv = qubitsMapInv[toReal];

        qubitsMap[toInv] = movingReal;
        qubitsMapInv[movingReal] = toInv;

        qubitsMap[qubit2] = toReal;
        qubitsMapInv[toReal] = qubit2;

        totalSwappingCost += bondCost[toReal];
        growBondDimension(toReal, true);
        movingReal = toReal;
      }
    }

    assert(abs(qubitsMap[qubit1] - qubitsMap[qubit2]) == 1);
  }

 private:
  size_t nrQubits;

  std::vector<IndexType> qubitsMap;
  std::vector<IndexType> qubitsMapInv;

  static constexpr size_t physExtent = 2;
  IndexType maxVirtualExtent = 0;
  std::vector<double> bondCost;
  std::vector<double> maxBondDim;
  std::vector<double> currentBondDim;

  double totalSwappingCost = 0;
  // double dimFactor = 0.95;

  void growBondDimension(IndexType bond, bool swap = true) {
    constexpr double growthFactorSwap = 1.;
    constexpr double growthFactorGate = 0.7; 
    
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

    // the left and right dimensions stay the same and also the physical legs have dimension 2

    // then the swap or the other gate is applied, getting a result that looks graphically as above, but of course with different values inside the tensor
    // swap is special, just swaps the values for (0, 1) and (1, 0) in
    // the physical legs, while other gates can change all values in the tensor
    
    // then the tensor is reshaped into a matrix, having dimensions 2 * leftDim x 2 * rightNeighborDim on this matrix SVD is applied, to separate out the
    // qubits tensors again, and the bond dimension is the number of singular
    // values kept after truncation (if done), or the number of non-zero
    // singular values if no truncation is done. The bond dimension can be at
    // most min(2 * min(leftDim, rightNeighborDim), maxBondDim[bond]) and the minimum is obviously 1


    const IndexType leftBond = bond - 1;
    const IndexType rightNeigborBond = bond + 1;
    const double betweenDim = currentBondDim[bond];

    const double leftDim = leftBond >= 0 ? currentBondDim[leftBond] : 1;
    const double rightNeighborDim = rightNeigborBond < static_cast<IndexType>(currentBondDim.size()) ? currentBondDim[rightNeigborBond] : 1;
    
    const double newMaxDim = (swap && leftDim == rightNeighborDim) ? betweenDim : 2. * std::min(leftDim, rightNeighborDim);

    const double growthFactor = swap ? growthFactorSwap : growthFactorGate;

    currentBondDim[bond] = std::min(newMaxDim * growthFactor, maxBondDim[bond]);

    currentBondDim[bond] = std::max(currentBondDim[bond], 1.);

    bondCost[bond] =
        currentBondDim[bond] * currentBondDim[bond] * currentBondDim[bond];
  }
};

}  // namespace Simulators

#endif
