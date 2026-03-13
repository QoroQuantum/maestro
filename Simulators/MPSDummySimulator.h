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
 * The swapping cost of the MPS simulator is not important, this will be used to evaluate it.
 */

#pragma once

#ifdef _MPSDUMMYSIMULATOR_H_
#define _MPSDUMMYSIMULATOR_H_

#include "MPSSimulator.h"

#include <Eigen/Eigen>

namespace Simulators {

class MPSDummySimulator {
 public:
  using IndexType = MPSSimulatorInterface::IndexType;

  MPSDummySimulator() = delete;

  MPSDummySimulator(size_t N) {
    InitQubitsMap();
  }

  size_t getNrQubits() const final { return qubitsMap.size(); }

  void Clear() {
    InitQubitsMap();
  }

  void InitOnesState() {
    InitQubitsMap();
  }

  void print() const override {
    impl.print();

    std::cout << "Qubits map: ";
    for (int q = 0; q < static_cast<int>(qubitsMap.size()); ++q)
      std::cout << q << "->" << qubitsMap[q] << " ";
    std::cout << std::endl;
  }

  void ApplyGate(const Gates::AppliedGate<MatrixClass>& gate) {
    ApplyGate(gate, gate.getQubit1(), gate.getQubit2());
  }

  void ApplyGate(const GateClass& gate, IndexType qubit,
                 IndexType controllingQubit1 = 0) override {
    if (qubit < 0 || qubit >= static_cast<IndexType>(impl.getNrQubits()))
      throw std::invalid_argument("Qubit index out of bounds");
    else if (controllingQubit1 < 0 ||
             controllingQubit1 >= static_cast<IndexType>(impl.getNrQubits()))
      throw std::invalid_argument("Qubit index out of bounds");

    IndexType qubit1 = qubitsMap[qubit];
    IndexType qubit2 = qubitsMap[controllingQubit1];

    // for two qubit gates:
    // if the qubits are not adjacent, apply swap gates until they are
    // don't forget to update the qubitsMap
    if (gate.getQubitsNumber() > 1 && abs(qubit1 - qubit2) > 1) {

      totalSwappingCost += abs(qubit1 - qubit2) - 1;

      SwapQubits(qubit, controllingQubit1);
      qubit1 = qubitsMap[qubit];
      qubit2 = qubitsMap[controllingQubit1];
      assert(abs(qubit1 - qubit2) == 1);
    }
  }

  void ApplyGates(
      const std::vector<Gates::AppliedGate<MatrixClass>>& gates) {
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

  IndexType getSwappingCost(IndexType q1, IndexType q2) const
  {
    const IndexType realq1 = qubitsMap[q1];
    const IndexType realq2 = qubitsMap[q2];

    return abs(realq1 - realq2) - 1;
  }

  // returns true even for passing the same qubit twice
  bool AreQubitsAdjacent(IndexType q1, IndexType q2) const
  {
    return getSwappingCost(q1, q2) <= 0;
  }

  void SetInitialQubitsMap(const std::vector<IndexType>& initialMap) {
    qubitsMap = initialMap;
    for (size_t i = 0; i < initialMap.size(); ++i)
      qubitsMapInv[initialMap[i]] = i;

    totalSwappingCost = 0;
  }

  IndexType getTotalSwappingCost() const
  {
    return totalSwappingCost; 
  }

  Eigen::MatrixXi getDistancesMatrix() const
  {
    // floyd-warshall algorithm to compute the distances matrix
    Eigen::MatrixXi distances(qubitsMap.size(), qubitsMap.size());

    for (IndexType i = 0; i < qubitsMap.size(); ++i)
      for (IndexType j = 0; j < qubitsMap.size(); ++j) {
        const IndexType dist = getSwappingCost(i, j);
        distances(i, j) = dist <= 0 ? 0 : dist;
      }

    for (IndexType k = 0; k < qubitsMap.size(); ++k)
      for (IndexType i = 0; i < qubitsMap.size(); ++i)
        for (IndexType j = 0; j < qubitsMap.size(); ++j) {
          const IndexType newDist = distances(i, k) + distances(k, j);
          if (distances(i, j) > newDist)
                distances(i, j) = newDist;
        }
              
    return distances;
  }

  Eigen::MatrixXi getCouplingsMatrix() const
  {
    Eigen::MatrixXi couplings =
        Eigen::MatrixXi::Eye(qubitsMap.size(), qubitsMap.size());

    for (IndexType i = 0; i < qubitsMap.size() - 1; ++i) {
      couplings(qubitsMapInv[i], qubitsMapInv[i + 1]) = 1;
      couplings(qubitsMapInv[i + 1], qubitsMapInv[i]) = 1;
    }

    return couplings;
  }

 private:
  void InitQubitsMap() {
    qubitsMap.resize(getNrQubits());
    qubitsMapInv.resize(getNrQubits());

    for (IndexType i = 0; i < static_cast<IndexType>(getNrQubits()); ++i)
      qubitsMapInv[i] = qubitsMap[i] = i;

    totalSwappingCost = 0;
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

    const IndexType mid = (qubitsMap.size() - 1) >> 1;
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
    const bool swapDown = qubitsMap.size() - realq2 <= realq1;

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

  std::vector<IndexType> qubitsMap;
  std::vector<IndexType> qubitsMapInv;

  IndexType totalSwappingCost = 0;
};

}

#endif

