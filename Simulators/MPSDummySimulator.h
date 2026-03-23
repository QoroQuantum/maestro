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
    IndexType maxVirtualExtentLimit = static_cast<IndexType>(untruncatedMaxExtent);

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
      double maxExtent2 = std::pow((double)physExtent, (double)nrQubits - i - 1.);

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
      growBondDimension(bond);
      totalSwappingCost += bondCost[bond];
    }
  }

  void ApplyGates(
      const std::vector<QC::Gates::AppliedGate<MatrixClass>>& gates) {
    for (const auto& gate : gates) ApplyGate(gate);
  }

  void ApplyGates(const std::vector<std::shared_ptr<Circuits::IOperation<>>>& gates) {
    for (const auto& gate : gates) ApplyGate(gate);
  }

  void SetInitialQubitsMap(const std::vector<long long int>& initialMap) {
    qubitsMap = initialMap;
    for (size_t i = 0; i < initialMap.size(); ++i)
      qubitsMapInv[initialMap[i]] = i;

    totalSwappingCost = 0;
    std::fill(currentBondDim.begin(), currentBondDim.end(), 1.0);
    for (size_t i = 0; i < bondCost.size(); ++i)
      bondCost[i] = 1;
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

  const std::vector<double>& getMaxBondDimensions() const {
    return maxBondDim;
  }

  const std::vector<IndexType>& getQubitsMap() const {
    return qubitsMap;
  }

  const std::vector<IndexType>& getQubitsMapInv() const {
    return qubitsMapInv;
  }

  double getTotalSwappingCost() const
  {
    return totalSwappingCost; 
  }

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



  // Evaluate the total cost of swapping qubit1 and qubit2 to a given meeting
  // position, applying the current 2-qubit gate, and then simulating the
  // next lookaheadDepth 2-qubit gates from upcomingGates.
  // The state is saved and restored, so this is non-destructive.
  void EvaluateMeetingPositionCost(IndexType meetPosition,
      const std::vector<std::shared_ptr<Circuits::IOperation<>>>& upcomingGates,
      size_t currentGateIndex, int lookaheadDepth,
      int lookaheadDepthWithHeuristic, double currentCost, double& bestCost, bool useSameDummy = false) 
  {
    if (currentGateIndex >= upcomingGates.size()) 
    {
      if (currentCost < bestCost) bestCost = currentCost;
      return;
    }

    // skip the 1 qubit gates, advance to the next 2-qubit gate
    while (currentGateIndex < upcomingGates.size() && upcomingGates[currentGateIndex]->AffectedQubits().size() < 2)
      ++currentGateIndex;

    if (currentGateIndex >= upcomingGates.size()) 
    {
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


    if (useSameDummy)
    {
      const double ccost = getTotalSwappingCost();
      
      if (realq2 - realq1 > 1)
        SwapQubitsToPosition(qubit1, qubit2, meetPosition);

      ApplyGate(op);

      currentCost += getTotalSwappingCost() - ccost;
      if (currentCost >= bestCost) return;
      // No more lookahead depth: return without further recursion
      if (lookaheadDepth <= 0) {
        if (currentCost < bestCost) bestCost = currentCost;
        return;
      }

      // skip the 1 qubit gates, advance over the current gate on the next
      // 2-qubit gate
      ++currentGateIndex;
      while (currentGateIndex < upcomingGates.size() &&
             upcomingGates[currentGateIndex]->AffectedQubits().size() < 2)
        ++currentGateIndex;

      if (currentGateIndex >= upcomingGates.size()) {
        if (currentCost < bestCost) bestCost = currentCost;
        return;
      }

      FindBestMeetingPosition(
          upcomingGates, currentGateIndex, lookaheadDepth - 1,
          lookaheadDepthWithHeuristic, currentCost, bestCost);
    } else {
      MPSDummySimulator dummySim(nrQubits, LightweightInitTag{});
      dummySim.maxVirtualExtent = maxVirtualExtent;
      dummySim.maxBondDim = maxBondDim;
      dummySim.currentBondDim = currentBondDim;
      dummySim.bondCost = bondCost;
      dummySim.qubitsMap = qubitsMap;
      for (size_t i = 0; i < qubitsMap.size(); ++i)
        dummySim.qubitsMapInv[qubitsMap[i]] = static_cast<IndexType>(i);

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
      while (currentGateIndex < upcomingGates.size() &&
             upcomingGates[currentGateIndex]->AffectedQubits().size() < 2)
        ++currentGateIndex;

      if (currentGateIndex >= upcomingGates.size()) {
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
      const std::vector<std::shared_ptr<Circuits::IOperation<>>>& upcomingGates, size_t currentGateIndex,
      int lookaheadDepth, int lookaheadDepthWithHeuristic, double currentCost, double& bestCost) 
    {
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
                                  bestCost);
      return pos;
    }

    if (realq2 - realq1 <= 1) {
      EvaluateMeetingPositionCost(realq1, upcomingGates, currentGateIndex,
                                  lookaheadDepth, lookaheadDepthWithHeuristic,
                                  currentCost, bestCost);

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

        if (bestCost < oldCost)
          bestPosition = m;
      }
    }

    return bestPosition;
  }


  std::vector<long long int> ComputeOptimalQubitsMap(
      const std::vector<std::shared_ptr<Circuits::Circuit<>>>& layersPassed,
      int nrShuffles = 25, int nrLayersToConsider = 30) {

    std::vector<std::shared_ptr<Circuits::Circuit<>>> layers = layersPassed;
    layers = std::vector<std::shared_ptr<Circuits::Circuit<>>>(
        layers.begin(),
        layers.size() > static_cast<size_t>(nrLayersToConsider)
            ? layers.begin() + nrLayersToConsider : layers.end());

    const IndexType nrQubits = getNrQubits();
    std::vector<long long int> qubitsMap(nrQubits);
    std::iota(qubitsMap.begin(), qubitsMap.end(), 0);

    if (layers.empty() || nrQubits <= 2) return qubitsMap;

    auto evaluateCost =
        [&, this](const std::vector<IndexType>& candidateMap) -> double {
      auto saveQubitsMap = qubitsMap;
      auto saveQubitsMapInv = qubitsMapInv;
      auto saveCurrentBondDim = currentBondDim;
      auto saveTotalSwappingCost = totalSwappingCost;
      auto saveBondCost = bondCost;

      SetInitialQubitsMap(candidateMap);
      for (const auto& layer : layersPassed) ApplyGates(layer->GetOperations());
      auto res = getTotalSwappingCost();
      // restore state
      qubitsMap = std::move(saveQubitsMap);
      qubitsMapInv = std::move(saveQubitsMapInv);
      currentBondDim = std::move(saveCurrentBondDim);
      totalSwappingCost = saveTotalSwappingCost;
      bondCost = std::move(saveBondCost);

      return res;
    };

    // Bounded variant: exits early when accumulated cost already exceeds
    // the known best, avoiding full re-simulation for losing candidates.
    // Most impactful in the O(n^2) 2-opt inner loop.
    auto evaluateCostBounded = [&, this](
                                   const std::vector<IndexType>& candidateMap,
                                   double bound) -> double {
      auto saveQubitsMap = qubitsMap;
      auto saveQubitsMapInv = qubitsMapInv;
      auto saveCurrentBondDim = currentBondDim;
      auto saveTotalSwappingCost = totalSwappingCost;
      auto saveBondCost = bondCost;

      SetInitialQubitsMap(candidateMap);
      for (const auto& layer : layersPassed) {
        ApplyGates(layer->GetOperations());
        if (getTotalSwappingCost() >= bound) {
          auto res = getTotalSwappingCost();
          // restore state
          qubitsMap = std::move(saveQubitsMap);
          qubitsMapInv = std::move(saveQubitsMapInv);
          currentBondDim = std::move(saveCurrentBondDim);
          totalSwappingCost = saveTotalSwappingCost;
          bondCost = std::move(saveBondCost);
          return res;
        }
      }
      auto res = getTotalSwappingCost();
      // restore state
      qubitsMap = std::move(saveQubitsMap);
      qubitsMapInv = std::move(saveQubitsMapInv);
      currentBondDim = std::move(saveCurrentBondDim);
      totalSwappingCost = saveTotalSwappingCost;
      bondCost = std::move(saveBondCost);

      return res;
    };

    // Collect 2-qubit pairs from each layer, preserving layer boundaries
    struct QubitPair { IndexType q1, q2; };
    std::vector<std::vector<QubitPair>> layerPairs;

    size_t gateCount = 0;
    size_t secondLayerLimit = std::numeric_limits<size_t>::max();

    for (size_t li = 0; li < layers.size(); ++li) {
      const auto& layer = layers[li];
      std::vector<QubitPair> lp;
      for (const auto& op : layer->GetOperations()) {
        auto qbits = op->AffectedQubits();
        if (qbits.size() >= 2) {
          if (qbits[0] > qbits[1]) std::swap(qbits[0], qbits[1]);

          lp.push_back({static_cast<IndexType>(qbits[0]),
                        static_cast<IndexType>(qbits[1])});

          ++gateCount;
          if (li == 1) secondLayerLimit = gateCount;
        }
      }
      if (!lp.empty())
        layerPairs.push_back(std::move(lp));
    }
    if (layerPairs.empty()) return qubitsMap;

    // Build a linear qubit chain from ordered pairs.
    // First-layer pairs are placed adjacent; later layers insert
    // unvisited qubits next to their already-placed partner.
    auto buildChain = [&](const std::vector<std::vector<QubitPair>>& orderedLP)
        -> std::vector<long long int> {
      std::vector<IndexType> chain;
      std::unordered_set<IndexType> placed;

      std::unordered_set<IndexType> above;
      std::unordered_set<IndexType> below;

      size_t gateCount = 0;
      for (const auto& lp : orderedLP) {
        if (placed.size() == static_cast<size_t>(nrQubits)) break;
        for (const auto& p : lp) {
          if (placed.size() == static_cast<size_t>(nrQubits)) break;
          const bool q1In = placed.count(p.q1) > 0;
          const bool q2In = placed.count(p.q2) > 0;

          if (!q1In && !q2In) {
            // Neither placed: add both adjacent at the end
            chain.push_back(p.q1);
            chain.push_back(p.q2);
            placed.insert(p.q1);
            placed.insert(p.q2);

            below.insert(p.q1);
            above.insert(p.q2);
          } 
          else if (q1In && !q2In && gateCount < secondLayerLimit) {
            // q1 placed, q2 not: insert q2 next to q1
            auto it = std::find(chain.begin(), chain.end(), p.q1);
            if (it == chain.begin())
              chain.insert(it, p.q2);
            else if (it + 1 == chain.end())
              chain.push_back(p.q2);
            else {
                if (below.count(p.q1) > 0) {
                    chain.insert(it, p.q2);
                    below.insert(p.q2);
                } else {
                    chain.insert(it + 1, p.q2);
                    above.insert(p.q2);
                }
            }
            placed.insert(p.q2);
          } else if (!q1In && q2In && gateCount < secondLayerLimit) {
            // q2 placed, q1 not: insert q1 next to q2
            auto it = std::find(chain.begin(), chain.end(), p.q2);
            if (it == chain.begin())
              chain.insert(it, p.q1);
            else if (it + 1 == chain.end())
              chain.push_back(p.q1);
            else
            {
                if (below.count(p.q2) > 0) {
                    chain.insert(it, p.q1);
                    below.insert(p.q1);
                } else {
                    chain.insert(it + 1, p.q1);
                    above.insert(p.q1);
                }
            }
            placed.insert(p.q1);
          }
          // Both placed: nothing to do
          ++gateCount;
        }
      }

      // Append any remaining unplaced qubits
      for (IndexType q = 0; q < nrQubits; ++q)
        if (placed.count(q) == 0) chain.push_back(q);

      assert(chain.size() == nrQubits);

      // Convert chain to qubitsMap: chain[physPos] = logicalQubit
      std::vector<long long int> result(nrQubits);
      for (size_t i = 0; i < chain.size(); ++i)
        result[chain[i]] = static_cast<long long int>(i);
      return result;
    };

    auto bestMap = buildChain(layerPairs);
    auto bestCost = evaluateCost(bestMap);

    // Build a bond-saturation-aware weight matrix from ALL layers.
    // Each two-qubit gate can at most double the bond dimension of the
    // bond it acts on.  With gates spread across ~(nrQubits-1) bonds,
    // the average bond dimension grows as
    //   chi ~ 2^(cumulative_2q_gates / (nrQubits-1))
    // clamped at maxVirtualExtent.  Swap cost scales as chi^3, so we
    // weight each layer's interactions by the estimated cube of the
    // current effective bond dimension.  This gives late (saturated)
    // layers orders-of-magnitude higher weight than the initial ones,
    // correctly focusing the optimizer on the expensive phase.
    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(nrQubits, nrQubits);
    {
      const double bondsCount =
          std::max(1., static_cast<double>(nrQubits) - 1.);
      const double maxBondDimD = static_cast<double>(maxVirtualExtent);
      int cumulativeTwoQubitGates = 0;

      for (const auto& layer : layers) {
        const auto& ops = layer->GetOperations();
        int twoQubitGateCount = 0;

        const double effectiveDepthPerBond =
            static_cast<double>(cumulativeTwoQubitGates) / bondsCount;
        const double rawBondDim =
            std::pow(static_cast<double>(physExtent), effectiveDepthPerBond);
        const double effectiveBondDim = std::min(rawBondDim, maxBondDimD);
        const double factor = std::max(1., std::pow(effectiveBondDim, 3.));

        for (const auto& op : ops) {
          const auto qubits = op->AffectedQubits();
          if (qubits.size() < 2) continue;
          const IndexType q1 = static_cast<IndexType>(qubits[0]);
          const IndexType q2 = static_cast<IndexType>(qubits[1]);
          if (q1 < 0 || q1 >= nrQubits || q2 < 0 || q2 >= nrQubits) continue;
          weights(q1, q2) += factor;
          weights(q2, q1) += factor;
          ++twoQubitGateCount;
        }

        cumulativeTwoQubitGates += twoQubitGateCount;
      }
    }

    // Build an adjacency list (max degree 2) from the weight matrix by
    // greedily selecting the highest-weight edges.  This preserves
    // the path/cycle structure required by the traversal below while
    // incorporating interaction data from ALL layers.
    std::vector<std::vector<IndexType>> adj(nrQubits);
    {
      struct WeightedEdge {
        double w;
        IndexType q1, q2;
        bool operator<(const WeightedEdge& o) const { return w > o.w; }
      };
      std::vector<WeightedEdge> edges;
      edges.reserve(nrQubits * (nrQubits - 1) / 2);
      for (IndexType i = 0; i < nrQubits; ++i)
        for (IndexType j = i + 1; j < nrQubits; ++j)
          if (weights(i, j) > 0) edges.push_back({weights(i, j), i, j});
      std::sort(edges.begin(), edges.end());

      for (const auto& e : edges) {
        if (adj[e.q1].size() >= 2 || adj[e.q2].size() >= 2) continue;
        adj[e.q1].push_back(e.q2);
        adj[e.q2].push_back(e.q1);
      }
    }

    // Traverse connected components (paths / cycles with max degree 2).
    // Start each traversal from a degree-1 endpoint when one exists;
    // for cycles every node has degree 2 so any start works.
    std::vector<bool> visited(nrQubits, false);
    IndexType pos = 0;

    for (IndexType start = 0; start < nrQubits; ++start) {
      if (visited[start] || adj[start].empty()) continue;

      // Walk to find an endpoint (degree-1 node) of this component
      IndexType endpoint = start;
      if (adj[start].size() == 2) {
        IndexType cur = adj[start][0];
        IndexType prev = start;
        while (adj[cur].size() == 2) {
          const IndexType next =
              (adj[cur][0] == prev) ? adj[cur][1] : adj[cur][0];
          if (next == start) break;  // cycle
          prev = cur;
          cur = next;
        }
        if (adj[cur].size() == 1) endpoint = cur;
      }

      // Lay out the path/cycle starting from the endpoint
      IndexType cur = endpoint;
      IndexType prev = -1;
      while (!visited[cur]) {
        visited[cur] = true;
        qubitsMap[cur] = pos++;

        IndexType next = -1;
        for (auto neighbor : adj[cur])
          if (neighbor != prev && !visited[neighbor]) {
            next = neighbor;
            break;
          }

        if (next < 0) break;
        prev = cur;
        cur = next;
      }
    }

    // Place remaining truly isolated vertices
    for (IndexType q = 0; q < nrQubits; ++q)
      if (!visited[q]) qubitsMap[q] = pos++;

    // this, despite minimizing swap cost for the weighted adjacency, may not
    // be the best solution for the entire circuit, so we can try some shuffles
    // and pick the best one, also check the origial map to at least have it at
    // the same cost as the original one, that is, ensure we don't get a worse
    // solution

    // now check execution against the original map (0, 1, ...) and also shuffle
    // the qubits several times and pick the order that minimizes the swapping
    // cost

    auto newCost = evaluateCostBounded(qubitsMap, bestCost);
    if (newCost < bestCost) {
      bestCost = newCost;
      bestMap = qubitsMap;
    }

    // Try the identity map
    std::vector<long long int> tryMap(nrQubits);
    std::iota(tryMap.begin(), tryMap.end(), 0);

    auto identityCost = evaluateCostBounded(tryMap, bestCost);
    if (identityCost < bestCost) {
      bestMap = tryMap;
      bestCost = identityCost;

      // std::cout << "Identity was better than the initial optimization, cost:
      // " << identityCost << std::endl;
    }

    // Try a greedy layout: prioritize adjacency for qubit pairs with the most
    // two-qubit gates, weighted by estimated swap cost at each circuit depth.
    {
      IndexType seedQ1 = 0, seedQ2 = 1;
      double maxW = -1;
      for (IndexType i = 0; i < nrQubits; ++i)
        for (IndexType j = i + 1; j < nrQubits; ++j)
          if (weights(i, j) > maxW) {
            maxW = weights(i, j);
            seedQ1 = i;
            seedQ2 = j;
          }

      std::deque<IndexType> chain;
      chain.push_back(seedQ1);
      chain.push_back(seedQ2);
      std::vector<bool> placed(nrQubits, false);
      placed[seedQ1] = true;
      placed[seedQ2] = true;

      while (static_cast<IndexType>(chain.size()) < nrQubits) {
        IndexType bestCandidate = -1;
        double bestConn = -1;

        for (IndexType q = 0; q < nrQubits; ++q) {
          if (placed[q]) continue;
          double maxConn = 0;
          for (const auto p : chain)
            if (weights(q, p) > maxConn) maxConn = weights(q, p);
          if (maxConn > bestConn) {
            bestConn = maxConn;
            bestCandidate = q;
          }
        }

        if (bestCandidate < 0) break;
        placed[bestCandidate] = true;

        // Interior insertion: evaluate every possible position in the
        // chain (including front and back) and pick the one that
        // minimizes total weighted bond-cost distance to all interaction
        // partners.  O(chainLen^2) per insertion, O(n^3) total, which
        // is fine for typical qubit counts.
        const size_t chainLen = chain.size();

        // Precompute prefix sums of bond costs so distance between any
        // two chain positions is a constant-time lookup.
        // prefixBondCost[i] = sum of bondCost[offset..offset+i-1],
        // with [0] = 0.  After inserting at position p the final chain
        // has chainLen+1 elements, so we need up to chainLen bonds.
        //
        // Center the partial chain within the full N-qubit bond cost
        // array so that costs reflect realistic final positions rather
        // than always using the cheapest end-of-chain costs.
        const size_t nrBondsNeeded = chainLen;
        const size_t bcOffset = (bondCost.size() > nrBondsNeeded)
                                    ? (bondCost.size() - nrBondsNeeded) / 2
                                    : 0;

        std::vector<double> prefixBondCost(chainLen + 2, 0.);
        for (size_t b = 0; b < chainLen + 1; ++b) {
          const size_t bondIdx = bcOffset + b;
          const double bc =
              (bondIdx < bondCost.size())
                  ? bondCost[bondIdx]
                  : (bondCost.empty() ? 1. : bondCost[bondCost.size() - 1]);
          prefixBondCost[b + 1] = prefixBondCost[b] + bc;
        }

        double bestInsertCost = std::numeric_limits<double>::infinity();
        size_t bestInsertPos = 0;

        for (size_t insertPos = 0; insertPos <= chainLen; ++insertPos) {
          double cost = 0;
          for (size_t idx = 0; idx < chainLen; ++idx) {
            const double w = weights(bestCandidate, chain[idx]);
            if (w <= 0) continue;
            // After insertion the existing element at index idx shifts:
            //   new position = idx  if idx < insertPos
            //   new position = idx+1  if idx >= insertPos
            // Candidate goes to insertPos.
            const size_t partnerNewPos = idx < insertPos ? idx : idx + 1;
            const size_t lo = std::min(insertPos, partnerNewPos);
            const size_t hi = std::max(insertPos, partnerNewPos);
            const double dist = prefixBondCost[hi] - prefixBondCost[lo];
            cost += w * dist;
          }
          if (cost < bestInsertCost) {
            bestInsertCost = cost;
            bestInsertPos = insertPos;
          }
        }

        if (bestInsertPos == 0)
          chain.push_front(bestCandidate);
        else if (bestInsertPos == chainLen)
          chain.push_back(bestCandidate);
        else
          chain.insert(
              chain.begin() + static_cast<std::ptrdiff_t>(bestInsertPos),
              bestCandidate);
      }

      std::vector<long long int> greedyMap(nrQubits);
      for (IndexType i = 0; i < static_cast<IndexType>(chain.size()); ++i)
        greedyMap[chain[i]] = i;

      auto greedyCost = evaluateCostBounded(greedyMap, bestCost);
      if (greedyCost < bestCost) {
        bestMap = greedyMap;
        bestCost = greedyCost;
      }

      // Also try the reversed chain: may place heavy pairs at cheaper bond
      // positions since bond costs are non-uniform (cheapest at ends)
      {
        std::vector<long long int> reversedMap(nrQubits);
        const auto chainSize = static_cast<long long int>(chain.size());
        for (IndexType i = 0; i < chainSize; ++i)
          reversedMap[chain[i]] = chainSize - 1 - i;

        auto reversedCost = evaluateCostBounded(reversedMap, bestCost);
        if (reversedCost < bestCost) {
          bestMap = reversedMap;
          bestCost = reversedCost;
        }
      }

      // Also try building from one end: seed pair at positions 0-1
      // (cheapest bond), extending only to the right. This directly
      // places the heaviest-interaction pair at the lowest-cost bond.
      {
        std::vector<IndexType> endChain;
        endChain.reserve(nrQubits);
        endChain.push_back(seedQ1);
        endChain.push_back(seedQ2);
        std::fill(placed.begin(), placed.end(), false);
        placed[seedQ1] = true;
        placed[seedQ2] = true;

        while (static_cast<IndexType>(endChain.size()) < nrQubits) {
          IndexType bestCand = -1;
          double bestC = -1;
          for (IndexType q = 0; q < nrQubits; ++q) {
            if (placed[q]) continue;
            double maxC = 0;
            for (const auto p : endChain)
              if (weights(q, p) > maxC) maxC = weights(q, p);
            if (maxC > bestC) {
              bestC = maxC;
              bestCand = q;
            }
          }
          if (bestCand < 0) break;
          placed[bestCand] = true;
          endChain.push_back(bestCand);
        }

        for (IndexType q = 0; q < nrQubits; ++q)
          if (!placed[q]) {
            placed[q] = true;
            endChain.push_back(q);
          }

        std::vector<long long int> endMap(nrQubits);
        for (IndexType i = 0; i < static_cast<IndexType>(endChain.size()); ++i)
          endMap[endChain[i]] = i;

        auto endCost = evaluateCostBounded(endMap, bestCost);
        if (endCost < bestCost) {
          bestMap = endMap;
          bestCost = endCost;
        }
      }

      // Spectral ordering via Fiedler vector: use the graph Laplacian's
      // second-smallest eigenvector to find a globally-aware linear arrangement
      if (nrQubits >= 3) {
        Eigen::MatrixXd laplacian = -weights;
        for (IndexType i = 0; i < nrQubits; ++i)
          laplacian(i, i) = weights.row(i).sum();

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(laplacian);
        if (solver.info() == Eigen::Success) {
          // Fiedler vector: eigenvector for the second-smallest eigenvalue
          Eigen::VectorXd fiedler = solver.eigenvectors().col(1);

          std::vector<IndexType> order(nrQubits);
          std::iota(order.begin(), order.end(), 0);
          std::sort(order.begin(), order.end(),
                    [&fiedler](IndexType a, IndexType b) {
                      return fiedler(a) < fiedler(b);
                    });

          std::vector<long long int> spectralMap(nrQubits);
          for (IndexType i = 0; i < nrQubits; ++i) spectralMap[order[i]] = i;

          auto spectralCost = evaluateCostBounded(spectralMap, bestCost);
          if (spectralCost < bestCost) {
            bestMap = spectralMap;
            bestCost = spectralCost;
          }

          // Also try the reversed spectral ordering since bond costs
          // are non-uniform (cheapest at the ends of the chain)
          {
            std::vector<long long int> revSpectralMap(nrQubits);
            for (IndexType i = 0; i < nrQubits; ++i)
              revSpectralMap[order[i]] = nrQubits - 1 - i;

            auto revSpectralCost =
                evaluateCostBounded(revSpectralMap, bestCost);
            if (revSpectralCost < bestCost) {
              bestMap = revSpectralMap;
              bestCost = revSpectralCost;
            }
          }
        }
      }
    }

    // Neighborhood perturbation: perturb the current best map by swapping
    // a small number of random positions rather than generating full random
    // permutations, keeping the search in a promising neighborhood.
    // Adaptive: scale budget with problem size and reset the no-improvement
    // counter whenever an improvement is found.
    std::mt19937 rng(42);
    std::uniform_int_distribution<IndexType> qubitDist(0, nrQubits - 1);
    std::uniform_int_distribution<int> nrSwapsDist(
        1, std::min<int>(3, static_cast<int>(nrQubits) / 2));

    const int maxNoImprove = std::max(nrShuffles, static_cast<int>(nrQubits));
    const int maxTotalShuffles = maxNoImprove * 3;
    int noImproveCount = 0;

    for (int s = 0; s < maxTotalShuffles && noImproveCount < maxNoImprove;
         ++s) {
      tryMap = bestMap;
      const int nrSwaps = nrSwapsDist(rng);
      for (int sw = 0; sw < nrSwaps; ++sw) {
        const IndexType a = qubitDist(rng);
        IndexType b = qubitDist(rng);
        while (b == a) b = qubitDist(rng);
        std::swap(tryMap[a], tryMap[b]);
      }

      auto cost = evaluateCostBounded(tryMap, bestCost);
      if (cost < bestCost) {
        bestMap = tryMap;
        bestCost = cost;
        noImproveCount = 0;

        // std::cout << "Shuffle " << s << " was better than the previous best,
        // cost: " << cost << std::endl;
      } else {
        ++noImproveCount;
      }
    }

    // 2-opt local search: iteratively swap pairs of positions in the best map
    // and keep improvements, until no single swap can reduce the cost
    {
      auto candidate = bestMap;
      bool improved = true;
      while (improved) {
        improved = false;
        for (IndexType i = 0; i < nrQubits; ++i) {
          for (IndexType j = i + 1; j < nrQubits; ++j) {
            // swap the mapped positions of qubits i and j
            std::swap(candidate[i], candidate[j]);
            auto cost = evaluateCostBounded(candidate, bestCost);
            if (cost < bestCost) {
              bestMap = candidate;
              bestCost = cost;
              improved = true;
            } else {
              // revert
              std::swap(candidate[i], candidate[j]);
            }
          }
        }
      }
    }

    return bestMap;
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
      : nrQubits(N), maxVirtualExtent(0) {
    InitQubitsMap();
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

  size_t nrQubits;

  std::vector<IndexType> qubitsMap;
  std::vector<IndexType> qubitsMapInv;

  static constexpr size_t physExtent = 2;
  IndexType maxVirtualExtent;
  std::vector<double> bondCost;
  std::vector<double> maxBondDim;
  std::vector<double> currentBondDim;

  double totalSwappingCost = 0;

  void growBondDimension(IndexType bond, bool swap = true) {
    constexpr double growthFactorSwap = 1.5; 
    // some average growth factor per 2-qubit gate, can be tuned for
    // better correlation with actual MPS behavior
    constexpr double growthFactorGate = 1.8;

    if (swap) {
      const double neighborDim =
          (bond + 1 < static_cast<IndexType>(currentBondDim.size()))
              ? currentBondDim[bond + 1]
              : currentBondDim[bond];
      currentBondDim[bond] =
          std::min(std::max(currentBondDim[bond], neighborDim) * growthFactorSwap,
                   maxBondDim[bond]);
    } else
        currentBondDim[bond] = std::min(
            currentBondDim[bond] * growthFactorGate,
            maxBondDim[bond]);

    bondCost[bond] = currentBondDim[bond] * currentBondDim[bond] * currentBondDim[bond];
  }
};

}  // namespace Simulators

#endif
