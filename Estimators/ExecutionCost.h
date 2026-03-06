/**
 * @file ExecutionCost.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Estimate the execution cost for a simulator, circuit and number of
 * shots.
 *
 * Rough estimtion of the execution cost (not time, we have estimators for
 * that), based on O() complexity and some guesses. Unlike the estimators, this
 * does not take into account multithreading and implementation details of the
 * simulators.
 */

#pragma once

#ifndef __EXECUTION_COST_H_
#define __EXECUTION_COST_H_

#include "../Simulators/QcsimPauliPropagator.h"
#include "../Simulators/Factory.h"
#include "../Utils/LogFile.h"

#include <cstddef>

namespace Estimators {

// this is not for time estimation, it can be used for something very, very,
// VERY rough, to decide just if it would execute in a reasonable time or not it
// cannot be used to compare runtime for two simulators of the same circuit, but
// it sort of can be used to compare two circuits in the same simulator
// (especially if the same number of samples is used for both circuits) for
// something better we have the estimators which got the constants from
// benchmarking and rely on implementation details of the simulators, so they
// are more accurate, but also less general

// Except for pauli propagation, where the operation cost depends on the
// position in the circuit (if the circuit is non-clifford), using a neural
// network (or even a simpler regressor) to learn the relation between the
// estimated cost and the actual execution time (single threaded, multithreading
// rises some other issues) would be probably possible
class ExecutionCost {
 public:
  struct CircuitInfo {
    size_t nrQubits = 0;
    size_t nrOneQubitOps = 0;
    size_t nrTwoQubitOps = 0;
    size_t nrThreeQubitOps = 0;
    size_t nrMiddleMeasurementOps = 0;
    size_t nrEndMeasurementOps = 0;
    size_t nrOneQubitOpsExecutedOnce = 0;
    size_t nrTwoQubitOpsExecutedOnce = 0;
    size_t nrThreeQubitOpsExecutedOnce = 0;
  };

  struct ExecutionInfo : public CircuitInfo {
    size_t nrSamples = 0;
    size_t nrQubitsSampled = 0;
    size_t maxBondDim = 0;
    size_t nrPauliOps = 0;
    double executionCost = 0;
    double samplingCost = 0;
  };

  static double EstimateExecutionCost(
      Simulators::SimulationType method, size_t nrQubits,
      const std::shared_ptr<Circuits::Circuit<>>& circuit, size_t maxBondDim) {
    if (method == Simulators::SimulationType::kPauliPropagator)
      return Simulators::QcsimPauliPropagator::GetCost(circuit);
    else if (method == Simulators::SimulationType::kStatevector) {
      const double opOrder = exp2(nrQubits);

      double cost = 0;
      for (const auto& op : *circuit) {
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement)
          cost += opOrder * affectedQubits.size();
        if (op->GetType() == Circuits::OperationType::kReset)
          cost += 2. * opOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
          if (affectedQubits.size() == 1)
            cost += opOrder;
          else if (affectedQubits.size() == 2)
            cost += 4 * opOrder;
          else if (affectedQubits.size() == 3)
            cost += 16 * opOrder;
        }
      }
      return cost;
    } else if (method == Simulators::SimulationType::kMatrixProductState) {
      const double twoQubitOpOrder = pow(maxBondDim, 3);
      const double oneQubitOpOrder = pow(maxBondDim, 2);

      double cost = 0;
      for (const auto& op : *circuit) {
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          // it's not that simple at all
          // a qubit measurement works by applying a projector on the qubit
          // tensor, which would cost as a one quibit gate but then we need to
          // propagate the effect of the measurement along the chain, left and
          // right, which is like applying a two qubit gate (SVD is the costlier
          // operation there) on all the qubits that are entangled with the
          // measured one, and in the worst case this can be all the other
          // qubits if more than one qubits are measured at the same time, then
          // we can have some optimizations, but let's just say that it's like
          // measuring them one by one, so the cost is multiplied by the number
          // of measured qubits
          cost += twoQubitOpOrder * affectedQubits.size() * nrQubits / 2.;
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
          if (affectedQubits.size() == 1)
            cost += oneQubitOpOrder;
          else if (affectedQubits.size() == 2)
            // I wish it were simple, but applying a gate involves swapping the
            // qubits next to each other, then applying the gate
            cost += twoQubitOpOrder * nrQubits / 3.;
          else if (affectedQubits.size() == 3)
            // qiskit aer has three qubit ops, qcsim has not, they are
            // decomposed into one and two qubit gates
            cost += twoQubitOpOrder * nrQubits * 2;  // very rough estimation
        }
      }
      return cost;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      const double measOrder = pow(nrQubits, 2);
      const double opOrder = nrQubits;

      double cost = 0;
      for (const auto& op : *circuit) {
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          cost += measOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate)
          cost += opOrder * affectedQubits.size();
      }
      return cost;
    }

    // for tensor network is hard to guess, it depends on contraction path

    return std::numeric_limits<double>::infinity();
  }

  static double EstimateSamplingCost(
      Simulators::SimulationType method, size_t nrQubits,
      size_t nrQubitsSampled, size_t samples,
      const std::shared_ptr<Circuits::Circuit<>>& circuit, size_t maxBondDim) {
    // sampling works very differenty for such circuits
    Circuits::OperationState dummyState;
    const bool hasMeasurementsInTheMiddle = circuit->HasOpsAfterMeasurements();
    const std::vector<bool> executedOps =
        circuit->ExecuteNonMeasurements(nullptr, dummyState);

    const size_t dif = circuit->size() - executedOps.size();

    if (method == Simulators::SimulationType::kPauliPropagator)
      return Simulators::QcsimPauliPropagator::GetSamplingCost(
          circuit, nrQubitsSampled, samples);

    if (method == Simulators::SimulationType::kStatevector) {
      const double opOrder = exp2(nrQubits);

      double cost = 0;
      for (size_t i = 0; i < circuit->size(); ++i) {
        if (i >= dif && !executedOps[i - dif]) continue;

        const auto& op = (*circuit)[i];
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement)
          cost += opOrder * affectedQubits.size();
        if (op->GetType() == Circuits::OperationType::kReset)
          cost += 2. * opOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
          if (affectedQubits.size() == 1)
            cost += opOrder;
          else if (affectedQubits.size() == 2)
            cost += 4 * opOrder;
          else if (affectedQubits.size() == 3)
            cost += 16 * opOrder;
        }
      }

      // the sampling cost depends on the simulator
      // qiskit aer constucts an index gowing over the statevector, qcsim does
      // something similar to have the samples then O(nrQubits), but in maestro
      // for qcsim (and quest) this is replaced by alias sampling with O(1) for
      // each sample

      if (hasMeasurementsInTheMiddle) {
        double samplingCost = 0;
        for (size_t i = dif; i < circuit->size(); ++i) {
          if (executedOps[i - dif]) continue;

          const auto& op = (*circuit)[i];
          const auto affectedQubits = op->AffectedQubits();
          if (op->GetType() == Circuits::OperationType::kMeasurement ||
              op->GetType() == Circuits::OperationType::kConditionalMeasurement)
            samplingCost += opOrder * affectedQubits.size();
          if (op->GetType() == Circuits::OperationType::kReset)
            samplingCost += 2. * opOrder * affectedQubits.size();
          else if (op->GetType() == Circuits::OperationType::kGate ||
                   op->GetType() == Circuits::OperationType::kConditionalGate) {
            if (affectedQubits.size() == 1)
              samplingCost += opOrder;
            else if (affectedQubits.size() == 2)
              samplingCost += 4 * opOrder;
            else if (affectedQubits.size() == 3)
              samplingCost += 16 * opOrder;
          }
        }

        samplingCost += opOrder * nrQubitsSampled;

        return cost + samplingCost * samples;
      }

      // some dummy cost, it's not going to fit all anyways
      return cost + 30 * opOrder + samples * nrQubits;
    } else if (method == Simulators::SimulationType::kMatrixProductState) {
      const double twoQubitOpOrder = pow(maxBondDim, 3);
      const double oneQubitOpOrder = pow(maxBondDim, 2);

      double cost = 0;
      for (size_t i = 0; i < circuit->size(); ++i) {
        if (i >= dif && !executedOps[i - dif]) continue;

        const auto& op = (*circuit)[i];
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          // it's not that simple at all
          // a qubit measurement works by applying a projector on the qubit
          // tensor, which would cost as a one quibit gate but then we need to
          // propagate the effect of the measurement along the chain, left and
          // right, which is like applying a two qubit gate (SVD is the costlier
          // operation there) on all the qubits that are entangled with the
          // measured one, and in the worst case this can be all the other
          // qubits if more than one qubits are measured at the same time, then
          // we can have some optimizations, but let's just say that it's like
          // measuring them one by one, so the cost is multiplied by the number
          // of measured qubits
          cost += twoQubitOpOrder * affectedQubits.size() * nrQubits / 2.;
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
          if (affectedQubits.size() == 1)
            cost += oneQubitOpOrder;
          else if (affectedQubits.size() == 2)
            // I wish it were simple, but applying a gate involves swapping the
            // qubits next to each other, then applying the gate
            cost += twoQubitOpOrder * nrQubits / 3.;
          else if (affectedQubits.size() == 3)
            // qiskit aer has three qubit ops, qcsim has not, they are
            // decomposed into one and two qubit gates
            cost += twoQubitOpOrder * nrQubits * 2;  // very rough estimation
        }
      }

      if (hasMeasurementsInTheMiddle) {
        double samplingCost = 0;
        for (size_t i = dif; i < circuit->size(); ++i) {
          if (executedOps[i - dif]) continue;

          const auto& op = (*circuit)[i];
          const auto affectedQubits = op->AffectedQubits();
          if (op->GetType() == Circuits::OperationType::kMeasurement ||
              op->GetType() ==
                  Circuits::OperationType::kConditionalMeasurement ||
              op->GetType() == Circuits::OperationType::kReset)
            samplingCost +=
                twoQubitOpOrder * affectedQubits.size() * nrQubits / 2.;
          else if (op->GetType() == Circuits::OperationType::kGate ||
                   op->GetType() == Circuits::OperationType::kConditionalGate) {
            if (affectedQubits.size() == 1)
              samplingCost += oneQubitOpOrder;
            else if (affectedQubits.size() == 2)
              // I wish it were simple, but applying a gate involves swapping
              // the qubits next to each other, then applying the gate
              samplingCost += twoQubitOpOrder * nrQubits / 3.;
            else if (affectedQubits.size() == 3)
              // qiskit aer has three qubit ops, qcsim has not, they are
              // decomposed into one and two qubit gates
              samplingCost +=
                  twoQubitOpOrder * nrQubits * 2;  // very rough estimation
          }
        }

        samplingCost += twoQubitOpOrder * nrQubits * nrQubits;

        return cost + samplingCost * samples;
      }

      // sampling can be done here (and for other simulator types as well)
      // either by saving the state, measuring then restoring the state or
      // simply going along the chain, computing probabilities, throwing biased
      // coins and doing matrix multiplications (qcsim does this if all qubits
      // are sampled, otherwise the measurement method is used)

      // some dummy cost, it's not going to fit all anyways
      return cost + samples * twoQubitOpOrder * nrQubits * nrQubits;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      const double measOrder = pow(nrQubits, 2);
      const double opOrder = nrQubits;

      double cost = 0;
      for (size_t i = 0; i < circuit->size(); ++i) {
        if (i >= dif && !executedOps[i - dif]) continue;

        const auto& op = (*circuit)[i];
        const auto affectedQubits = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kMeasurement ||
            op->GetType() == Circuits::OperationType::kConditionalMeasurement ||
            op->GetType() == Circuits::OperationType::kReset)
          cost += measOrder * affectedQubits.size();
        else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate)
          cost += opOrder * affectedQubits.size();
      }

      if (hasMeasurementsInTheMiddle) {
        double samplingCost = 0;
        for (size_t i = dif; i < circuit->size(); ++i) {
          if (executedOps[i - dif]) continue;

          const auto& op = (*circuit)[i];
          const auto affectedQubits = op->AffectedQubits();
          if (op->GetType() == Circuits::OperationType::kMeasurement ||
              op->GetType() ==
                  Circuits::OperationType::kConditionalMeasurement ||
              op->GetType() == Circuits::OperationType::kReset)
            samplingCost += measOrder * affectedQubits.size();
          else if (op->GetType() == Circuits::OperationType::kGate ||
                   op->GetType() == Circuits::OperationType::kConditionalGate)
            samplingCost += opOrder * affectedQubits.size();
        }

        samplingCost += measOrder * nrQubitsSampled;

        return cost + samplingCost * samples;
      }

      // sampling is done with saving state, measuring and restoring state
      // the overhead for saving / restoring is not here (it's of the same order
      // as measuring), but measurements are not all O(n^2) anyways
      return cost + samples * measOrder * nrQubitsSampled;
    }

    // for tensor network is hard to guess, it depends on contraction path

    return std::numeric_limits<double>::infinity();
  }

  static double EstimatePauliExpectationCost(
      const std::string& pauliString, Simulators::SimulationType method,
      size_t nrQubits, const std::shared_ptr<Circuits::Circuit<>>& circuit,
      size_t maxBondDim) {
    // a pauli string propagated
    if (method == Simulators::SimulationType::kPauliPropagator)
      return Simulators::QcsimPauliPropagator::GetCost(circuit);

    size_t pauliCnt = 0;
    for (char c : pauliString) {
      if (c == 'X' || c == 'x' || c == 'Y' || c == 'y' || c == 'Z' || c == 'z')
        ++pauliCnt;
    }
    double cost = EstimateExecutionCost(method, nrQubits, circuit, maxBondDim);

    if (method == Simulators::SimulationType::kStatevector) {
      const double opOrder = exp2(nrQubits);

      // each pauli matrix is applied to the statevector, then the product with
      // the original is computed
      cost += opOrder * (pauliCnt + 1);

      return cost;
    } else if (method == Simulators::SimulationType::kMatrixProductState) {
      const double twoQubitOpOrder = pow(maxBondDim, 3);
      const double oneQubitOpOrder = pow(maxBondDim, 2);

      // each pauli is applied to the chain then the resulted chain is
      // contracted with the original one
      cost += oneQubitOpOrder * pauliCnt + nrQubits * twoQubitOpOrder;

      return cost;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      cost += nrQubits * nrQubits;
      return cost;
    }

    return std::numeric_limits<double>::infinity();
  }

  static std::shared_ptr<Circuits::Circuit<>> GenerateRandomCircuit(
      size_t nrQubits, size_t depth, double measureInsideProbability = 0.,
      size_t nrMeasAtEnd = 0, bool isClifford = false,
      size_t nrNonCliffordGatesLimit = 0) {
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    std::random_device rdev;
    std::mt19937 rng(rdev());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> paramDist(-2 * M_PI, 2 * M_PI);
    std::uniform_int_distribution<Types::qubit_t> qubitDist(0, nrQubits - 1);
    std::uniform_int_distribution<int> gateDist(
        0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));

    std::uniform_int_distribution<int> gateDistOneQubit(
        0, static_cast<int>(Circuits::QuantumGateType::kUGateType));

    std::vector<Types::qubit_t> qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);

    size_t nrNonCliffordGates = 0;
    if (nrNonCliffordGatesLimit == 0 && !isClifford)
      nrNonCliffordGatesLimit = depth;

    for (size_t i = 0; i < depth; ++i) {
      if (dist(rng) < measureInsideProbability) {
        Types::qubit_t q = qubitDist(rng);
        std::vector<std::pair<Types::qubit_t, size_t>> qs = {{q, q}};
        circuit->AddOperation(
            std::make_shared<Circuits::MeasurementOperation<>>(qs));
        continue;
      }

      std::shuffle(qubits.begin(), qubits.end(), rng);
      const auto q1 = qubits[0];
      const auto q2 = qubits[1];
      const auto q3 = qubits[2];

      auto gateType = static_cast<Circuits::QuantumGateType>(
          nrNonCliffordGatesLimit < depth
              ? gateDistOneQubit(rng)  // avoid three qubit gates, they are
                                       // non-clifford and they cost a lot
              : gateDist(rng));
      auto param1 = paramDist(rng);
      auto param2 = paramDist(rng);
      auto param3 = paramDist(rng);
      auto param4 = paramDist(rng);

      auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3, param4);
      if (isClifford) {
        while (!theGate->IsClifford()) {
          gateType = static_cast<Circuits::QuantumGateType>(gateDist(rng));
          theGate = Circuits::CircuitFactory<>::CreateGate(
              gateType, q1, q2, q3, param1, param2, param3, param4);
        }
      }

      if (!theGate->IsClifford()) ++nrNonCliffordGates;

      if (nrNonCliffordGates > nrNonCliffordGatesLimit) {
        // replace the non clifford gate with a clifford one
        gateType =
            static_cast<Circuits::QuantumGateType>(gateDistOneQubit(rng));
        theGate = Circuits::CircuitFactory<>::CreateGate(
            gateType, q1, q2, q3, param1, param2, param3, param4);
        while (!theGate->IsClifford()) {
          gateType =
              static_cast<Circuits::QuantumGateType>(gateDistOneQubit(rng));
          theGate = Circuits::CircuitFactory<>::CreateGate(
              gateType, q1, q2, q3, param1, param2, param3, param4);
        }
        --nrNonCliffordGates;
      }

      circuit->AddOperation(theGate);
    }

    if (nrMeasAtEnd > 0) {
      if (nrMeasAtEnd > nrQubits) nrMeasAtEnd = nrQubits;
      std::shuffle(qubits.begin(), qubits.end(), rng);

      for (size_t i = 0; i < nrMeasAtEnd; ++i) {
        std::vector<std::pair<Types::qubit_t, size_t>> qs = {{qubits[i], i}};
        circuit->AddOperation(
            std::make_shared<Circuits::MeasurementOperation<>>(qs));
      }
    }

    return circuit;
  }

  static double MeasureExecutionTime(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      size_t nrQubits, const std::shared_ptr<Circuits::Circuit<>>& circuit,
      size_t nrReps, size_t maxBondDim) {
    auto sim = GetSimulator(simType, method, nrQubits, maxBondDim);

    Circuits::OperationState dummyState(nrQubits);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < nrReps; ++i) circuit->Execute(sim, dummyState);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
  }

  static double MeasureSamplingTime(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      size_t nrQubits, const std::shared_ptr<Circuits::Circuit<>>& circuit,
      size_t nrQubitsSampled, size_t nrSamples, size_t nrReps,
      size_t maxBondDim) {
    auto sim = GetSimulator(simType, method, nrQubits, maxBondDim);
    Circuits::OperationState dummyState(nrQubits);

    if (nrQubitsSampled > nrQubits) nrQubitsSampled = nrQubits;
    Types::qubits_vector qubitsSampled(nrQubitsSampled);
    std::iota(qubitsSampled.begin(), qubitsSampled.end(), 0);

    const bool hasMeasurementsInTheMiddle = circuit->HasOpsAfterMeasurements();

    auto start = std::chrono::high_resolution_clock::now();

    if (hasMeasurementsInTheMiddle) {
      for (size_t i = 0; i < nrReps; ++i) {
        const auto executedOps = circuit->ExecuteNonMeasurements(
            sim, dummyState);  // execute the circuit up to measurements

        // now sample
        for (size_t sample = 0; sample < nrSamples; ++sample) {
          circuit->ExecuteMeasurements(
              sim, dummyState, executedOps);  // execute the measurements
        }
      }
    } else {
      for (size_t i = 0; i < nrReps; ++i) {
        circuit->Execute(sim, dummyState);  // execute the circuit first
        sim->SampleCountsMany(qubitsSampled, nrSamples);
      }
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end - start).count();
  }

  static double MeasurePauliExpectationTime(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      size_t nrQubits, const std::shared_ptr<Circuits::Circuit<>>& circuit,
      const std::string& pauliString, size_t nrReps, size_t maxBondDim) {
    auto sim = GetSimulator(simType, method, nrQubits, maxBondDim);
    Circuits::OperationState dummyState(nrQubits);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < nrReps; ++i) {
      circuit->Execute(sim, dummyState);  // execute the circuit first
      sim->ExpectationValue(pauliString);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
  }

  static std::shared_ptr<Simulators::ISimulator> GetSimulator(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      size_t nrQubits, size_t maxBondDim) {
    auto sim = Simulators::SimulatorsFactory::CreateSimulator(simType, method);
    if (method == Simulators::SimulationType::kMatrixProductState) {
      const auto strVal = std::to_string(maxBondDim);
      sim->Configure("matrix_product_state_max_bond_dimension", strVal.c_str());
    }
    sim->AllocateQubits(nrQubits);
    sim->SetMultithreading(false);
    sim->Initialize();
    return sim;
  }

  static CircuitInfo GetCircuitInfo(
      const std::shared_ptr<Circuits::Circuit<>>& circuit) {
    CircuitInfo info;

    info.nrQubits = circuit->GetMaxQubitIndex() + 1;
    Circuits::OperationState dummyState(info.nrQubits);
    const bool hasMeasurementsInTheMiddle = circuit->HasOpsAfterMeasurements();
    const std::vector<bool> executedOps =
        circuit->ExecuteNonMeasurements(nullptr, dummyState);

    const size_t dif = circuit->size() - executedOps.size();

    size_t i = 0;
    for (const auto& op : *circuit) {
      const auto affectedQubits = op->AffectedQubits();
      if (op->GetType() == Circuits::OperationType::kMeasurement ||
          op->GetType() == Circuits::OperationType::kConditionalMeasurement) {
        if (hasMeasurementsInTheMiddle)
          ++info.nrMiddleMeasurementOps;
        else
          ++info.nrEndMeasurementOps;
      } else if (op->GetType() == Circuits::OperationType::kGate ||
                 op->GetType() == Circuits::OperationType::kConditionalGate) {
        if (affectedQubits.size() == 1) {
          ++info.nrOneQubitOps;
          if (i < dif || executedOps[i - dif]) ++info.nrOneQubitOpsExecutedOnce;
        } else if (affectedQubits.size() == 2) {
          ++info.nrTwoQubitOps;
          if (i < dif || executedOps[i - dif]) ++info.nrTwoQubitOpsExecutedOnce;
        } else if (affectedQubits.size() == 3) {
          ++info.nrThreeQubitOps;
          if (i < dif || executedOps[i - dif])
            ++info.nrThreeQubitOpsExecutedOnce;
        }
      }
      ++i;
    }

    return info;
  }

  static std::vector<ExecutionInfo> ReadLog(const std::string& logFilePath) {
    std::vector<ExecutionInfo> executionInfos;

    std::ifstream logFile(logFilePath);
    if (!logFile.is_open()) {
      std::cerr << "Failed to open log file: " << logFilePath << std::endl;
      return executionInfos;
    }

    std::string line;

    while (std::getline(logFile, line)) {
      std::stringstream ss(line);
      std::string value;
      ExecutionInfo info;
      std::getline(ss, value, ',');
      info.nrQubits = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrOneQubitOps = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrTwoQubitOps = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrThreeQubitOps = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrMiddleMeasurementOps = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrEndMeasurementOps = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrOneQubitOpsExecutedOnce = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrTwoQubitOpsExecutedOnce = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrThreeQubitOpsExecutedOnce = std::stoul(value);

      std::getline(ss, value, ',');
      info.nrSamples = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrQubitsSampled = std::stoul(value);
      std::getline(ss, value, ',');
      info.maxBondDim = std::stoul(value);
      std::getline(ss, value, ',');
      info.nrPauliOps = std::stoul(value);

      std::getline(ss, value, ',');
      info.executionCost = std::stod(value);
      std::getline(ss, value, ',');
      info.samplingCost = std::stod(value);

      if (info.nrSamples < 1) info.nrSamples = 1;

      executionInfos.push_back(std::move(info));
    }

    return executionInfos;
  }

  static void BenchmarkAndLogExecution(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      size_t nrReps, size_t nrMinQubits, size_t nrMaxQubits, size_t stepQubits,
      size_t depthMin, size_t depthMax, size_t stepDepth,
      double measureInsideProbability, size_t nrMeasAtEndMin,
      size_t nrMeasAtEndMax, size_t stepMeasAtEnd,
      size_t nrRandomCircuitsPerConfig, size_t maxBondDim,
      const std::string& logFilePath) {
    bool isClifford = (method == Simulators::SimulationType::kStabilizer);
    int nrNonCliffordGates = depthMax;  // a limit
    // but if it's pauli...
    if (method == Simulators::SimulationType::kPauliPropagator) {
      nrNonCliffordGates = 1;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      nrNonCliffordGates = 0;
    }

    std::cout << "Benchmarking execution for simType: "
              << static_cast<int>(simType)
              << ", method: " << static_cast<int>(method) << std::endl;

    Utils::LogFile log(logFilePath);

    for (size_t nrQubits = nrMinQubits; nrQubits <= nrMaxQubits;
         nrQubits += stepQubits) {
      std::cout << "  Qubits: " << nrQubits << std::endl;
      for (size_t depth = depthMin; depth <= depthMax; depth += stepDepth) {
        std::cout << "    Depth: " << depth << std::endl;
        for (size_t nrMeasAtEnd = nrMeasAtEndMin;
             nrMeasAtEnd <= std::min(nrMeasAtEndMax, nrQubits);
             nrMeasAtEnd += stepMeasAtEnd) {
          std::cout << "      Measurements at end: " << nrMeasAtEnd
                    << std::endl;
          for (size_t i = 0; i < nrRandomCircuitsPerConfig; ++i) {
            std::cout << "        Random circuit: " << i + 1 << "/"
                      << nrRandomCircuitsPerConfig << std::endl;
            if (method == Simulators::SimulationType::kPauliPropagator)
              isClifford = !isClifford;

            const auto circuit = GenerateRandomCircuit(
                nrQubits, depth, measureInsideProbability, nrMeasAtEnd,
                isClifford, nrNonCliffordGates);
            BenchmarkAndLogExecution(simType, method, circuit, nrReps,
                                     maxBondDim, log);
          }
        }
      }
    }
  }

  static std::string GeneratePauliString(size_t nrQubits) {
    std::string pauli;
    pauli.resize(nrQubits);
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dist(0, 3);

    for (size_t i = 0; i < nrQubits; ++i) {
      const int v = dist(g);
      switch (v) {
        case 0:
          pauli[i] = 'I';
          break;
        case 1:
          pauli[i] = 'X';
          break;
        case 2:
          pauli[i] = 'Y';
          break;
        case 3:
          pauli[i] = 'Z';
          break;
      }
    }

    return pauli;
  }

  static void BenchmarkAndLogPauliExpectation(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      size_t nrReps, size_t nrMinQubits, size_t nrMaxQubits, size_t stepQubits,
      size_t depthMin, size_t depthMax, size_t stepDepth,
      size_t nrRandomCircuitsPerConfig, size_t maxBondDim,
      const std::string& logFilePath) {
    bool isClifford = (method == Simulators::SimulationType::kStabilizer);
    int nrNonCliffordGates = depthMax;  // a limit
    // but if it's pauli...
    if (method == Simulators::SimulationType::kPauliPropagator) {
      nrNonCliffordGates = 1;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      nrNonCliffordGates = 0;
    }
    Utils::LogFile log(logFilePath);

    std::cout << "Benchmarking Pauli expectation for simType: "
              << static_cast<int>(simType)
              << ", method: " << static_cast<int>(method) << std::endl;

    for (size_t nrQubits = nrMinQubits; nrQubits <= nrMaxQubits;
         nrQubits += stepQubits) {
      std::cout << "  Qubits: " << nrQubits << std::endl;
      for (size_t depth = depthMin; depth <= depthMax; depth += stepDepth) {
        std::cout << "    Depth: " << depth << std::endl;
        for (size_t i = 0; i < nrRandomCircuitsPerConfig; ++i) {
          if (method == Simulators::SimulationType::kPauliPropagator)
            isClifford = !isClifford;

          const auto circuit = GenerateRandomCircuit(
              nrQubits, depth, 0., 0, isClifford, nrNonCliffordGates);
          const std::string pauliString = GeneratePauliString(nrQubits);

          std::cout << "      Random circuit: " << i + 1 << "/"
                    << nrRandomCircuitsPerConfig
                    << " Pauli string: " << pauliString << std::endl;
          BenchmarkAndLogPauliExpectation(simType, method, circuit, pauliString,
                                          nrReps, maxBondDim, log);
        }
      }
    }
  }

  static void BenchmarkAndLogSampling(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      size_t nrReps, size_t nrMinQubits, size_t nrMaxQubits, size_t stepQubits,
      size_t depthMin, size_t depthMax, size_t stepDepth, size_t nrMeasAtEndMin,
      size_t nrMeasAtEndMax, size_t stepMeasAtEnd, size_t nrSamplesMin,
      size_t nrSamplesMax, size_t multiplierSamples,
      size_t nrRandomCircuitsPerConfig, size_t maxBondDim,
      const std::string& logFilePath) {
    bool isClifford = (method == Simulators::SimulationType::kStabilizer);
    int nrNonCliffordGates = depthMax;  // a limit
    // but if it's pauli...
    if (method == Simulators::SimulationType::kPauliPropagator) {
      nrNonCliffordGates = 1;
    } else if (method == Simulators::SimulationType::kStabilizer) {
      nrNonCliffordGates = 0;
    }
    Utils::LogFile log(logFilePath);

    std::cout << "Benchmarking sampling for simType: "
              << static_cast<int>(simType)
              << ", method: " << static_cast<int>(method) << std::endl;

    for (size_t nrQubits = nrMinQubits; nrQubits <= nrMaxQubits;
         nrQubits += stepQubits) {
      std::cout << "  Qubits: " << nrQubits << std::endl;
      for (size_t depth = depthMin; depth <= depthMax; depth += stepDepth) {
        std::cout << "    Depth: " << depth << std::endl;
        for (size_t nrSamples = nrSamplesMin; nrSamples <= nrSamplesMax;
             nrSamples *= multiplierSamples) {
          std::cout << "      Samples: " << nrSamples << std::endl;
          for (size_t i = 0; i < nrRandomCircuitsPerConfig; ++i) {
            if (method == Simulators::SimulationType::kPauliPropagator)
              isClifford = !isClifford;
            const auto circuit = GenerateRandomCircuit(
                nrQubits, depth, 0., 0, isClifford, nrNonCliffordGates);
            for (size_t nrQubitsSampled = nrQubits; nrQubitsSampled >= 1;
                 nrQubitsSampled /= 2) {
              std::cout << "        Random circuit: " << i + 1 << "/"
                        << nrRandomCircuitsPerConfig
                        << " Qubits sampled: " << nrQubitsSampled << std::endl;
              BenchmarkAndLogSampling(simType, method, circuit, nrQubitsSampled,
                                      nrSamples, nrReps, maxBondDim, log);
            }
          }
        }
      }
    }
  }

  static void BenchmarkAndLogExecution(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      const std::shared_ptr<Circuits::Circuit<>>& circuit, size_t nrReps,
      size_t maxBondDim, Utils::LogFile& log) {
    const auto info = GetCircuitInfo(circuit);
    const double estimatedCost =
        EstimateExecutionCost(method, info.nrQubits, circuit, maxBondDim);
    const double executionTime =
        MeasureExecutionTime(simType, method, info.nrQubits, circuit, nrReps,
                             maxBondDim) /
        nrReps;

    std::stringstream ss;

    ss << info.nrQubits << "," << info.nrOneQubitOps << ","
       << info.nrTwoQubitOps << "," << info.nrThreeQubitOps << ","
       << info.nrMiddleMeasurementOps << "," << info.nrEndMeasurementOps << ","
       << info.nrOneQubitOpsExecutedOnce << ","
       << info.nrTwoQubitOpsExecutedOnce << ","
       << info.nrThreeQubitOpsExecutedOnce << ","
       << "0,0," << maxBondDim << ","
       << "0," << estimatedCost << "," << executionTime;

    log.Log(ss.str());
  }

  static void BenchmarkAndLogSampling(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      const std::shared_ptr<Circuits::Circuit<>>& circuit,
      size_t nrQubitsSampled, size_t nrSamples, size_t nrReps,
      size_t maxBondDim, Utils::LogFile& log) {
    const auto info = GetCircuitInfo(circuit);
    const double estimatedCost = EstimateSamplingCost(
        method, info.nrQubits, nrQubitsSampled, nrSamples, circuit, maxBondDim);
    const double samplingTime =
        MeasureSamplingTime(simType, method, info.nrQubits, circuit,
                            nrQubitsSampled, nrSamples, nrReps, maxBondDim) /
        nrReps;
    std::stringstream ss;
    ss << info.nrQubits << "," << info.nrOneQubitOps << ","
       << info.nrTwoQubitOps << "," << info.nrThreeQubitOps << ","
       << info.nrMiddleMeasurementOps << "," << info.nrEndMeasurementOps << ","
       << info.nrOneQubitOpsExecutedOnce << ","
       << info.nrTwoQubitOpsExecutedOnce << ","
       << info.nrThreeQubitOpsExecutedOnce << "," << nrSamples << ","
       << nrQubitsSampled << "," << maxBondDim << ","
       << "0," << estimatedCost << "," << samplingTime;
    log.Log(ss.str());
  }

  static void BenchmarkAndLogPauliExpectation(
      Simulators::SimulatorType simType, Simulators::SimulationType method,
      const std::shared_ptr<Circuits::Circuit<>>& circuit,
      const std::string& pauliString, size_t nrReps, size_t maxBondDim,
      Utils::LogFile& log) {
    auto info = GetCircuitInfo(circuit);
    info.nrQubits = std::max(info.nrQubits, pauliString.size());
    const double estimatedCost = EstimatePauliExpectationCost(
        pauliString, method, info.nrQubits, circuit, maxBondDim);
    const double expectationTime =
        MeasurePauliExpectationTime(simType, method, info.nrQubits, circuit,
                                    pauliString, nrReps, maxBondDim) /
        nrReps;

    size_t cntPauli = 0;
    for (char c : pauliString) {
      if (c == 'X' || c == 'x' || c == 'Y' || c == 'y' || c == 'Z' || c == 'z')
        ++cntPauli;
    }

    std::stringstream ss;
    ss << info.nrQubits << "," << info.nrOneQubitOps << ","
       << info.nrTwoQubitOps << "," << info.nrThreeQubitOps << ","
       << info.nrMiddleMeasurementOps << "," << info.nrEndMeasurementOps << ","
       << info.nrOneQubitOpsExecutedOnce << ","
       << info.nrTwoQubitOpsExecutedOnce << ","
       << info.nrThreeQubitOpsExecutedOnce << ","
       << "0,0," << maxBondDim << "," << cntPauli << "," << estimatedCost << ","
       << expectationTime;
    log.Log(ss.str());
  }
};

}  // namespace Estimators

#endif
