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

  MPSDummySimulator() = delete;

  MPSDummySimulator(size_t N) : nrQubits(N), maxVirtualExtent(0) {
    InitQubitsMap();

    SetMaxBondDimension(0); 
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

      if (i < static_cast<IndexType>(nrQubits) - 1) bondCost[i] = std::pow(maxRightExtent, 3.);
    }
  }

  void InitOnesState() {
    InitQubitsMap();
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

    std::shared_ptr<QC::Gates::AppliedGate<MatrixClass>> appliedGate;

    if (qbits.size() == 1) {
      appliedGate = std::make_shared<QC::Gates::AppliedGate<MatrixClass>>(
          MatrixClass::Identity(
              2,
              2),  // this is a dummy gate, the actual matrix is not important
          qbits[0]);
    } else if (qbits.size() == 2) {
      appliedGate = std::make_shared<QC::Gates::AppliedGate<MatrixClass>>(
          MatrixClass::Identity(
              4,
              4),  // this is a dummy gate, the actual matrix is not important
          qbits[0], qbits[1]);
    } else {
      throw std::invalid_argument("Unsupported number of qubits for the gate");
    }

    ApplyGate(*appliedGate);
  }

  void ApplyGate(const GateClass& gate, IndexType qubit,
                 IndexType controllingQubit1 = 0) {
    if (qubit < 0 || qubit >= static_cast<IndexType>(getNrQubits()))
      throw std::invalid_argument("Qubit index out of bounds");
    else if (controllingQubit1 < 0 ||
             controllingQubit1 >= static_cast<IndexType>(getNrQubits()))
      throw std::invalid_argument("Qubit index out of bounds");

    IndexType qubit1 = qubitsMap[qubit];
    IndexType qubit2 = qubitsMap[controllingQubit1];

    // for two qubit gates:
    // if the qubits are not adjacent, apply swap gates until they are
    // don't forget to update the qubitsMap
    if (gate.getQubitsNumber() > 1 && abs(qubit1 - qubit2) > 1) {

      totalSwappingCost += getSwappingCost(qubit, controllingQubit1);

      if (swapAmplifyingFactor < maxVirtualExtent)
        swapAmplifyingFactor *= swapAmplifyingFactor;

      SwapQubits(qubit, controllingQubit1);
      qubit1 = qubitsMap[qubit];
      qubit2 = qubitsMap[controllingQubit1];
      assert(abs(qubit1 - qubit2) == 1);
    }
  }

  void ApplyGates(
      const std::vector<QC::Gates::AppliedGate<MatrixClass>>& gates) {
    for (const auto& gate : gates) ApplyGate(gate);
  }

  void ApplyGates(std::vector<std::shared_ptr<Circuits::IOperation<>>> gates) {
    for (const auto& gate : gates) ApplyGate(gate);
  }

  void MoveAtBeginningOfChain(const std::set<IndexType>& qubits) {
    IndexType qubitPos = 0;

    for (const auto qubit : qubits) {
      IndexType realQubit = qubitsMap[qubit];
      while (realQubit != qubitPos) {
        const IndexType toQubitReal = realQubit - 1;
        const IndexType toQubitInv = qubitsMapInv[toQubitReal];

        qubitsMap[toQubitInv] = realQubit;
        qubitsMapInv[realQubit] = toQubitInv;

        qubitsMap[qubit] = toQubitReal;
        qubitsMapInv[toQubitReal] = qubit;

        realQubit = toQubitReal;
      }

      ++qubitPos;
    }
  }

  double getSwappingCost(IndexType q1, IndexType q2) const
  {
    const IndexType realq1 = qubitsMap[q1];
    const IndexType realq2 = qubitsMap[q2];

    //return abs(realq1 - realq2) - 1;
    double swappingCost = 0;
    for (IndexType i = std::min(realq1, realq2); i < std::max(realq1, realq2); ++i)
      swappingCost += bondCost[i];

    return swappingCost * swapAmplifyingFactor;
  }

  // returns true even for passing the same qubit twice
  bool AreQubitsAdjacent(IndexType q1, IndexType q2) const
  {
    const IndexType realq1 = qubitsMap[q1];
    const IndexType realq2 = qubitsMap[q2];

    return abs(realq1 - realq2) < 2;
  }

  void SetInitialQubitsMap(const std::vector<long long int>& initialMap) {
    qubitsMap = initialMap;
    for (size_t i = 0; i < initialMap.size(); ++i)
      qubitsMapInv[initialMap[i]] = i;

    totalSwappingCost = 0;
    swapAmplifyingFactor = initialSwapAmplifyingFactor;
  }

  double getTotalSwappingCost() const
  {
    return totalSwappingCost; 
  }

  Eigen::MatrixXd getDistancesMatrix() const
  {
    // floyd-warshall algorithm to compute the distances matrix
    Eigen::MatrixXd distances(qubitsMap.size(), qubitsMap.size());

    for (size_t i = 0; i < qubitsMap.size(); ++i)
      for (size_t j = 0; j < qubitsMap.size(); ++j) {
        const double dist = getSwappingCost(i, j);
        distances(i, j) = dist <= 0 ? 0 : dist;
      }

    for (size_t k = 0; k < qubitsMap.size(); ++k)
      for (size_t i = 0; i < qubitsMap.size(); ++i)
        for (size_t j = 0; j < qubitsMap.size(); ++j) {
          const double newDist = distances(i, k) + distances(k, j);
          if (distances(i, j) > newDist)
                distances(i, j) = newDist;
        }

    return distances;
  }

  Eigen::MatrixXi getCouplingsMatrix() const {
    Eigen::MatrixXi couplings =
        Eigen::MatrixXi::Identity(qubitsMap.size(), qubitsMap.size());

    for (size_t i = 0; i < qubitsMap.size() - 1; ++i) {
      couplings(qubitsMapInv[i], qubitsMapInv[i + 1]) = 1;
      couplings(qubitsMapInv[i + 1], qubitsMapInv[i]) = 1;
    }

    return couplings;
  }

  std::vector<long long int> ComputeOptimalQubitsMap(
      const std::vector<std::shared_ptr<Circuits::Circuit<>>>& layers,
      int nrShuffles = 25) {
    const IndexType nrQubits = getNrQubits();
    std::vector<long long int> qubitsMap(nrQubits);
    std::iota(qubitsMap.begin(), qubitsMap.end(), 0);

    if (layers.empty() || nrQubits <= 2) return qubitsMap;

    std::vector<std::vector<IndexType>> adj(nrQubits);
    const size_t layersToConsider = layers.size() < 2 ? layers.size() : 2;

    for (size_t l = 0; l < layersToConsider; ++l) {
      const auto& ops = layers[l]->GetOperations();
      for (const auto& op : ops) {
        const auto qubits = op->AffectedQubits();
        if (qubits.size() < 2) continue;

        const IndexType q1 = static_cast<IndexType>(qubits[0]);
        const IndexType q2 = static_cast<IndexType>(qubits[1]);
        if (q1 < 0 || q1 >= nrQubits || q2 < 0 || q2 >= nrQubits) continue;

        bool exists = false;
        for (auto n : adj[q1])
          if (n == q2) {
            exists = true;
            break;
          }

        if (!exists) {
          adj[q1].push_back(q2);
          adj[q2].push_back(q1);
        }
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

    // Place isolated vertices, using information from subsequent layers
    // to keep pairs that appear together in two-qubit gates adjacent
    std::vector<IndexType> isolated;
    for (IndexType q = 0; q < nrQubits; ++q)
      if (!visited[q]) isolated.push_back(q);

    if (!isolated.empty() && layers.size() > layersToConsider) {
      std::unordered_set<IndexType> isolatedSet(isolated.begin(),
                                                isolated.end());
      std::vector<std::vector<IndexType>> adjIso(nrQubits);

      for (size_t l = layersToConsider; l < layers.size(); ++l) {
        const auto& ops = layers[l]->GetOperations();
        for (const auto& op : ops) {
          const auto qubits = op->AffectedQubits();
          if (qubits.size() < 2) continue;

          const IndexType q1 = static_cast<IndexType>(qubits[0]);
          const IndexType q2 = static_cast<IndexType>(qubits[1]);
          if (q1 < 0 || q1 >= nrQubits || q2 < 0 || q2 >= nrQubits) continue;
          if (isolatedSet.find(q1) == isolatedSet.end() ||
              isolatedSet.find(q2) == isolatedSet.end())
            continue;

          adjIso[q1].push_back(q2);
          adjIso[q2].push_back(q1);

          isolatedSet.erase(q1);
          isolatedSet.erase(q2);
        }
      }

      // Traverse connected components among isolated vertices the same way
      for (IndexType i = 0; i < static_cast<IndexType>(isolated.size()); ++i) {
        const IndexType start = isolated[i];
        if (visited[start] || adjIso[start].empty()) continue;

        IndexType endpoint = start;
        if (adjIso[start].size() == 2) {
          IndexType cur = adjIso[start][0];
          IndexType prev = start;
          while (adjIso[cur].size() == 2) {
            const IndexType next =
                (adjIso[cur][0] == prev) ? adjIso[cur][1] : adjIso[cur][0];
            if (next == start) break;
            prev = cur;
            cur = next;
          }
          if (adjIso[cur].size() == 1) endpoint = cur;
        }

        IndexType cur = endpoint;
        IndexType prev = -1;
        while (!visited[cur]) {
          visited[cur] = true;
          qubitsMap[cur] = pos++;

          IndexType next = -1;
          for (auto neighbor : adjIso[cur])
            if (neighbor != prev && !visited[neighbor]) {
              next = neighbor;
              break;
            }

          if (next < 0) break;
          prev = cur;
          cur = next;
        }
      }
    }

    // Place remaining truly isolated vertices
    for (IndexType q = 0; q < nrQubits; ++q)
      if (!visited[q]) qubitsMap[q] = pos++;

    // this, despite making the swap cost zero for the first two layers, may not
    // be the best solution for the entire circuit, so we can try some shuffles
    // and pick the best one, also check the origial map to at least have it at
    // the same cost as the original one, that is, ensure we don't get a worse
    // solution

    // now check execution against the original map (0, 1, ...) and also shuffle the qubits several times and pick the order that minimizes the swapping cost
    auto evaluateCost = [&, this](const std::vector<IndexType>& candidateMap) -> double {
      SetInitialQubitsMap(candidateMap);
      for (const auto& layer : layers) ApplyGates(layer->GetOperations());
      return getTotalSwappingCost();
    };

    auto bestMap = qubitsMap;
    auto bestCost = evaluateCost(bestMap);

    // Try the identity map
    std::vector<long long int> tryMap(nrQubits);
    std::iota(tryMap.begin(), tryMap.end(), 0);

    auto identityCost = evaluateCost(tryMap);
    if (identityCost < bestCost) {
      bestMap = tryMap;
      bestCost = identityCost;

      // std::cout << "Identity was better than the initial optimization, cost:
      // " << identityCost << std::endl;
    }

    // Try a greedy layout: prioritize adjacency for qubit pairs with the most
    // two-qubit gates
    {
      Eigen::MatrixXi weights = Eigen::MatrixXi::Zero(nrQubits, nrQubits);
      for (const auto& layer : layers) {
        const auto& ops = layer->GetOperations();
        for (const auto& op : ops) {
          const auto qubits = op->AffectedQubits();
          if (qubits.size() < 2) continue;
          const IndexType q1 = static_cast<IndexType>(qubits[0]);
          const IndexType q2 = static_cast<IndexType>(qubits[1]);
          if (q1 < 0 || q1 >= nrQubits || q2 < 0 || q2 >= nrQubits) continue;
          ++weights(q1, q2);
          ++weights(q2, q1);
        }
      }

      IndexType seedQ1 = 0, seedQ2 = 1;
      IndexType maxW = -1;
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
        IndexType bestConn = -1;

        for (IndexType q = 0; q < nrQubits; ++q) {
          if (placed[q]) continue;
          IndexType maxConn = 0;
          for (const auto p : chain)
            if (weights(q, p) > maxConn) maxConn = weights(q, p);
          if (maxConn > bestConn) {
            bestConn = maxConn;
            bestCandidate = q;
          }
        }

        if (bestCandidate < 0) break;
        placed[bestCandidate] = true;

        if (weights(bestCandidate, chain.front()) >=
            weights(bestCandidate, chain.back()))
          chain.push_front(bestCandidate);
        else
          chain.push_back(bestCandidate);
      }

      std::vector<long long int> greedyMap(nrQubits);
      for (IndexType i = 0; i < static_cast<IndexType>(chain.size()); ++i)
        greedyMap[chain[i]] = i;

      auto greedyCost = evaluateCost(greedyMap);
      if (greedyCost < bestCost) {
        bestMap = greedyMap;
        bestCost = greedyCost;

        // std::cout << "Greedy layout was better than the previous
        // optimization, cost: " << greedyCost << std::endl;
      }
    }

    // Try several random shuffles
    std::mt19937 rng(42);
    for (int s = 0; s < nrShuffles; ++s) {
      std::shuffle(tryMap.begin(), tryMap.end(), rng);
      auto cost = evaluateCost(tryMap);
      if (cost < bestCost) {
        bestMap = tryMap;
        bestCost = cost;

        // std::cout << "Shuffle " << s << " was better than the previous best,
        // cost: " << cost << std::endl;
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
    swapAmplifyingFactor = initialSwapAmplifyingFactor;
  }

  void SwapQubits(IndexType qubit1, IndexType qubit2, bool towardsMiddle = true,
                  size_t stepsUp = 0) {
    // if the qubits are not adjacent, apply swap gates until they are
    // don't forget to update the qubitsMap
    IndexType realq1 = qubitsMap[qubit1];
    IndexType realq2 = qubitsMap[qubit2];
    if (realq1 > realq2) {
      std::swap(realq1, realq2);
      std::swap(qubit1, qubit2);
    }

    if (realq2 - realq1 <= 1) return;

    const IndexType mid =
        towardsMiddle ? (qubitsMap.size() - 1) >> 1 : realq1 + stepsUp;
    if (realq1 < mid && realq2 > mid)  // is the middle between the two qubits?
    {
      const IndexType mappedMid = qubitsMapInv[mid];
      SwapQubits(qubit1, mappedMid);  // this brings qubit1 near the middle
      realq1 = qubitsMap[qubit1];
      // the other qubit is above the middle, so it won't be affected by the
      // swap the code that follows will bring qubit2 in the middle
    }  // otherwise the qubit that's near an end of the chain will be moved
       // towards the other qubit

    // this is just a heuristic, better solutions that minimize the number of
    // swaps would be possible
    const bool swapDown =
        static_cast<IndexType>(qubitsMap.size()) - realq2 <= realq1;

    const IndexType targetQubitReal = swapDown ? realq1 + 1 : realq2 - 1;
    IndexType movingQubitReal = swapDown ? realq2 : realq1;
    const IndexType movingQubitInv = swapDown ? qubit2 : qubit1;

    do {
      const IndexType toQubitReal = movingQubitReal + (swapDown ? -1 : 1);
      const IndexType toQubitInv = qubitsMapInv[toQubitReal];

      qubitsMap[toQubitInv] = movingQubitReal;
      qubitsMapInv[movingQubitReal] = toQubitInv;

      qubitsMap[movingQubitInv] = toQubitReal;
      qubitsMapInv[toQubitReal] = movingQubitInv;

      movingQubitReal = toQubitReal;
    } while (movingQubitReal != targetQubitReal);
  }

  size_t nrQubits;

  std::vector<IndexType> qubitsMap;
  std::vector<IndexType> qubitsMapInv;

  static constexpr size_t physExtent = 2;
  IndexType maxVirtualExtent;
  std::vector<double> bondCost;

  double totalSwappingCost = 0;
  static constexpr double initialSwapAmplifyingFactor = 1.05;
  double swapAmplifyingFactor = initialSwapAmplifyingFactor;
};

}  // namespace Simulators

#endif
