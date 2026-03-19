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
        swapAmplifyingFactor *= initialSwapAmplifyingFactor;

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

    // Bounded variant: exits early when accumulated cost already exceeds
    // the known best, avoiding full re-simulation for losing candidates.
    // Most impactful in the O(n^2) 2-opt inner loop.
    auto evaluateCostBounded = [&, this](const std::vector<IndexType>& candidateMap, double bound) -> double {
      SetInitialQubitsMap(candidateMap);
      for (const auto& layer : layers) {
        ApplyGates(layer->GetOperations());
        if (getTotalSwappingCost() >= bound)
          return getTotalSwappingCost();
      }
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

    // Interaction weights matrix: weight each qubit pair by how often they
    // appear together in two-qubit gates, scaled by estimated swap cost at
    // that circuit depth.  Used by both the greedy layout and 2-opt delta
    // evaluation.
    Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(nrQubits, nrQubits);
    {
      // Model bond saturation
      const double bondsCount = std::max(1., static_cast<double>(nrQubits) - 1.);
      const double maxBondDimD = static_cast<double>(maxVirtualExtent);
      int cumulativeTwoQubitGates = 0;

      for (const auto& layer : layers) {
        const auto& ops = layer->GetOperations();
        int twoQubitGateCount = 0;

        const double effectiveDepthPerBond =
            static_cast<double>(cumulativeTwoQubitGates) / bondsCount;
        const double rawBondDim =
            std::pow(static_cast<double>(physExtent), effectiveDepthPerBond);
        const double effectiveBondDim =
            std::min(rawBondDim, maxBondDimD);
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
          double totalConn = 0;
          for (const auto p : chain)
            totalConn += weights(q, p);
          if (totalConn > bestConn) {
            bestConn = totalConn;
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
        // prefixBondCost[i] = sum of bondCost[0..i-1], with [0] = 0.
        // After inserting at position p the final chain has chainLen+1
        // elements, so we need up to chainLen bonds.
        std::vector<double> prefixBondCost(chainLen + 2, 0.);
        for (size_t b = 0; b < chainLen + 1 && b < bondCost.size(); ++b)
          prefixBondCost[b + 1] = prefixBondCost[b] + bondCost[b];
        // Clamp: if chain grows beyond bondCost.size(), extend with
        // the last known bond cost (the maximum, at the chain center)
        for (size_t b = bondCost.size() + 1; b < chainLen + 2; ++b)
          prefixBondCost[b] = prefixBondCost[b - 1] +
                              (bondCost.empty() ? 1. : bondCost.back());

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
            const size_t partnerNewPos =
                idx < insertPos ? idx : idx + 1;
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
          chain.insert(chain.begin() +
                           static_cast<std::ptrdiff_t>(bestInsertPos),
                       bestCandidate);
      }

      std::vector<long long int> greedyMap(nrQubits);
      for (IndexType i = 0; i < static_cast<IndexType>(chain.size()); ++i)
        greedyMap[chain[i]] = i;

      auto greedyCost = evaluateCost(greedyMap);
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

        auto reversedCost = evaluateCost(reversedMap);
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
            double totalC = 0;
            for (const auto p : endChain)
              totalC += weights(q, p);
            if (totalC > bestC) {
              bestC = totalC;
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

        auto endCost = evaluateCost(endMap);
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
          for (IndexType i = 0; i < nrQubits; ++i)
            spectralMap[order[i]] = i;

          auto spectralCost = evaluateCost(spectralMap);
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

            auto revSpectralCost = evaluateCost(revSpectralMap);
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
    std::uniform_int_distribution<int> nrSwapsDist(1, std::min<int>(3, static_cast<int>(nrQubits) / 2));

    const int maxNoImprove = std::max(nrShuffles, static_cast<int>(nrQubits));
    const int maxTotalShuffles = maxNoImprove * 3;
    int noImproveCount = 0;

    for (int s = 0; s < maxTotalShuffles && noImproveCount < maxNoImprove; ++s) {
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

    // 2-opt local search with delta evaluation: use the interaction weights
    // matrix as a proxy for the full simulation cost.  For a given map,
    //   proxy_cost = sum_{(a,b)} weights(a,b) * bondDist(map[a], map[b]).
    // When swapping map positions of qubits i and j, only interactions
    // involving i or j change, making each evaluation O(n) instead of O(G).
    // After convergence, verify with full simulation.
    {
      // Precompute prefix sums of bond costs for O(1) distance lookup
      std::vector<double> prefixBond(nrQubits, 0.);
      for (IndexType k = 1; k < nrQubits; ++k)
        prefixBond[k] = prefixBond[k - 1] + bondCost[k - 1];

      auto bondDist = [&prefixBond](IndexType a, IndexType b) -> double {
        return a <= b ? prefixBond[b] - prefixBond[a]
                      : prefixBond[a] - prefixBond[b];
      };

      // Compute the delta in proxy cost when swapping map[i] <-> map[j].
      // Only interactions involving qubit i or j are affected.
      // The (i,j) pair itself has symmetric distance so it cancels out.
      auto computeDelta = [&](const std::vector<IndexType>& m,
                              IndexType i, IndexType j) -> double {
        double delta = 0;
        const IndexType posI = m[i];
        const IndexType posJ = m[j];
        for (IndexType k = 0; k < nrQubits; ++k) {
          if (k == i || k == j) continue;
          const IndexType posK = m[k];
          const double wik = weights(i, k);
          if (wik > 0)
            delta += wik * (bondDist(posJ, posK) - bondDist(posI, posK));
          const double wjk = weights(j, k);
          if (wjk > 0)
            delta += wjk * (bondDist(posI, posK) - bondDist(posJ, posK));
        }
        return delta;
      };

      auto candidate = bestMap;
      bool improved = true;
      while (improved) {
        improved = false;
        for (IndexType i = 0; i < nrQubits; ++i) {
          for (IndexType j = i + 1; j < nrQubits; ++j) {
            const double delta = computeDelta(candidate, i, j);
            if (delta < -1e-10) {
              std::swap(candidate[i], candidate[j]);
              improved = true;
            }
          }
        }
      }

      // Verify the proxy-optimized map with full simulation
      auto fullCost = evaluateCost(candidate);
      if (fullCost < bestCost) {
        bestMap = candidate;
        bestCost = fullCost;
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

  void SwapQubits(IndexType qubit1, IndexType qubit2) {
    // if the qubits are not adjacent, apply swap gates until they are
    // don't forget to update the qubitsMap
    IndexType realq1 = qubitsMap[qubit1];
    IndexType realq2 = qubitsMap[qubit2];
    if (realq1 > realq2) {
      std::swap(realq1, realq2);
      std::swap(qubit1, qubit2);
    }

    if (realq2 - realq1 <= 1) return;

    // Move the pair towards the closest end of the chain:
    // end bonds have smaller dimensions and therefore cheaper swap costs
    const bool swapDown =
        realq1 + realq2 <= static_cast<IndexType>(qubitsMap.size()) - 1;

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
  static constexpr double initialSwapAmplifyingFactor = 1.005;
  double swapAmplifyingFactor = initialSwapAmplifyingFactor;
};

}  // namespace Simulators

#endif
