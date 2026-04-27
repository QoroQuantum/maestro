/**
 * @file NetworkJob.h
 * @ingroup network
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * A network job class.
 */

#pragma once

#ifndef _NETWORK_JOB_H
#define _NETWORK_JOB_H

#include "../Types.h"
#include "../Utils/ThreadsPool.h"

#include "../Simulators/MPSDummySimulator.h"

#include "Network.h"

namespace Network {

template <typename Time = Types::time_type>
class ExecuteJob {
 public:
  using ExecuteResults = typename Circuits::Circuit<Time>::ExecuteResults;

  ExecuteJob() = delete;

  explicit ExecuteJob(const std::shared_ptr<Circuits::Circuit<Time>> &c,
                      ExecuteResults &r, size_t cnt, size_t nq, size_t nc,
                      size_t ncr, Simulators::SimulatorType t,
                      Simulators::SimulationType m, std::mutex &mut)
      : dcirc(c),
        res(r),
        curCnt(cnt),
        nrQubits(nq),
        nrCbits(nc),
        nrResultCbits(ncr),
        simType(t),
        method(m),
        resultsMutex(mut) {}

  void DoWork() {
    if (curCnt == 0) return;

    Circuits::OperationState state;
    state.AllocateBits(nrCbits);

    const bool hasMeasurementsOnlyAtEnd = !dcirc->HasOpsAfterMeasurements();
    const bool optimiseMultipleShots = optimiseMultipleShotsExecution;
    const bool specialOptimizationForStatevector =
        optimiseMultipleShots &&
        method == Simulators::SimulationType::kStatevector &&
        hasMeasurementsOnlyAtEnd;
    const bool specialOptimizationForMPS =
        optimiseMultipleShots &&
        method == Simulators::SimulationType::kMatrixProductState &&
        hasMeasurementsOnlyAtEnd;

    dcirc = dcirc->RemoveExecutedOperations(executedGates);

    if (!optSim) {
      optSim = Simulators::SimulatorsFactory::CreateSimulator(simType, method);
      if (!optSim) return;

      if (!maxBondDim.empty())
        optSim->Configure("matrix_product_state_max_bond_dimension",
                          maxBondDim.c_str());
      if (!singularValueThreshold.empty())
        optSim->Configure("matrix_product_state_truncation_threshold",
                          singularValueThreshold.c_str());
      if (!mpsSample.empty())
        optSim->Configure("mps_sample_measure_algorithm", mpsSample.c_str());

      optSim->AllocateQubits(nrQubits);
      optSim->Initialize();

      OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);

      if (optimiseMultipleShots) {
        executedGates = dcirc->ExecuteNonMeasurements(optSim, state);

        if (!specialOptimizationForStatevector && !specialOptimizationForMPS &&
            curCnt > 1)
          optSim->SaveState();

        dcirc = dcirc->RemoveExecutedOperations(executedGates);
        if (network->GetMPSOptimizeSwaps())
          optSim->SetUpcomingGates(dcirc->GetOperations());
      }
    }

    std::shared_ptr<Circuits::MeasurementOperation<Time>> measurementsOp;

    const std::vector<bool> executed = std::move(executedGates);

    if (optimiseMultipleShots && hasMeasurementsOnlyAtEnd) {
      bool isQiskitAer = false;
#ifndef NO_QISKIT_AER
      if (optSim->GetType() == Simulators::SimulatorType::kQiskitAer) {
        isQiskitAer = true;
      }
#endif
      measurementsOp = dcirc->GetLastMeasurements(executed, isQiskitAer);
      const auto &qbits = measurementsOp->GetQubits();
      if (qbits.empty()) {
        auto bits = state.GetAllBits();
        bits.resize(nrResultCbits, false);

        const std::lock_guard lock(resultsMutex);
        res[bits] += curCnt;

        return;
      }
    }

    ExecuteResults localRes;

    if (optimiseMultipleShots &&
        (specialOptimizationForStatevector || hasMeasurementsOnlyAtEnd)) {
      const auto &qbits = measurementsOp->GetQubits();

      const auto sampleres = optSim->SampleCountsMany(qbits, curCnt);

      for (const auto &[mstate, cnt] : sampleres) {
        measurementsOp->SetStateFromSample(mstate, state);

        auto bits = state.GetAllBits();
        bits.resize(nrResultCbits, false);

        localRes[bits] += cnt;

        state.Reset();
      }

      const std::lock_guard lock(resultsMutex);
      for (const auto &r : localRes) res[r.first] += r.second;

      return;
    }

    const auto curCnt1 = curCnt > 0 ? curCnt - 1 : 0;
    for (size_t i = 0; i < curCnt; ++i) {
      if (optimiseMultipleShots) {
        if (i > 0) {
          optSim->RestoreState();
          optSim->SetGatesCounter(0);
        }
        dcirc->ExecuteMeasurements(optSim, state, executed);
      } else {
        dcirc->Execute(optSim, state);
        if (i < curCnt1) {
          optSim->Reset();
          optSim->SetGatesCounter(0);
        }
      }

      auto bits = state.GetAllBits();
      bits.resize(nrResultCbits, false);

      ++localRes[bits];

      state.Reset();
    }

    const std::lock_guard lock(resultsMutex);
    for (const auto &r : localRes) res[r.first] += r.second;
  }

  void DoWorkNoLock() {
    if (curCnt == 0) return;

    Circuits::OperationState state;
    state.AllocateBits(nrCbits);

    const bool hasMeasurementsOnlyAtEnd = !dcirc->HasOpsAfterMeasurements();
    const bool optimiseMultipleShots = optimiseMultipleShotsExecution;
    const bool specialOptimizationForStatevector =
        optimiseMultipleShots &&
        method == Simulators::SimulationType::kStatevector &&
        hasMeasurementsOnlyAtEnd;
    const bool specialOptimizationForMPS =
        optimiseMultipleShots &&
        method == Simulators::SimulationType::kMatrixProductState &&
        hasMeasurementsOnlyAtEnd;

    if (optSim) {
      optSim->SetMultithreading(true);

      if (optSim->GetNumberOfQubits() != nrQubits) {
        optSim->Clear();

        if (!maxBondDim.empty())
          optSim->Configure("matrix_product_state_max_bond_dimension",
                            maxBondDim.c_str());
        if (!singularValueThreshold.empty())
          optSim->Configure("matrix_product_state_truncation_threshold",
                            singularValueThreshold.c_str());
        if (!mpsSample.empty())
          optSim->Configure("mps_sample_measure_algorithm", mpsSample.c_str());

        optSim->AllocateQubits(nrQubits);
        optSim->Initialize();

        OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);

        if (optimiseMultipleShots) {
          executedGates = dcirc->ExecuteNonMeasurements(optSim, state);

          if (!specialOptimizationForStatevector &&
              !specialOptimizationForMPS && curCnt > 1)
            optSim->SaveState();
          dcirc = dcirc->RemoveExecutedOperations(executedGates);
          if (network->GetMPSOptimizeSwaps())
            optSim->SetUpcomingGates(dcirc->GetOperations());
        }
      } else if (executedGates.size() == dcirc->size()) {
        // special case for when the simulator is passed from the network
        // and no gates were executed yet
        bool needToExecuteGates = true;
        for (const bool val : executedGates) {
          if (val) {
            needToExecuteGates = false;
            break;
          }
        }
        if (needToExecuteGates && optimiseMultipleShots) {
          executedGates = dcirc->ExecuteNonMeasurements(optSim, state);
          if (!specialOptimizationForStatevector &&
              !specialOptimizationForMPS && curCnt > 1)
            optSim->SaveState();
          dcirc = dcirc->RemoveExecutedOperations(executedGates);
          optSim->SetUpcomingGates(dcirc->GetOperations());
        } else {
          dcirc = dcirc->RemoveExecutedOperations(executedGates);
          optSim->SetUpcomingGates(dcirc->GetOperations());
        }
      } else {
        dcirc = dcirc->RemoveExecutedOperations(executedGates);
        optSim->SetUpcomingGates(dcirc->GetOperations());
      }
    } else {
      optSim = Simulators::SimulatorsFactory::CreateSimulator(simType, method);
      if (!optSim) return;

      optSim->SetMultithreading(true);

      if (!maxBondDim.empty())
        optSim->Configure("matrix_product_state_max_bond_dimension",
                          maxBondDim.c_str());
      if (!singularValueThreshold.empty())
        optSim->Configure("matrix_product_state_truncation_threshold",
                          singularValueThreshold.c_str());
      if (!mpsSample.empty())
        optSim->Configure("mps_sample_measure_algorithm", mpsSample.c_str());

      optSim->AllocateQubits(nrQubits);
      optSim->Initialize();

      OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);

      if (optimiseMultipleShots) {
        executedGates = dcirc->ExecuteNonMeasurements(optSim, state);

        if (!specialOptimizationForStatevector && !specialOptimizationForMPS &&
            curCnt > 1)
          optSim->SaveState();

        dcirc = dcirc->RemoveExecutedOperations(executedGates);
        if (network->GetMPSOptimizeSwaps())
          optSim->SetUpcomingGates(dcirc->GetOperations());
      }
    }

    std::shared_ptr<Circuits::MeasurementOperation<Time>> measurementsOp;

    const std::vector<bool> executed = std::move(executedGates);

    if (optimiseMultipleShots && hasMeasurementsOnlyAtEnd) {
      bool isQiskitAer = false;
#ifndef NO_QISKIT_AER
      if (optSim->GetType() == Simulators::SimulatorType::kQiskitAer) {
        isQiskitAer = true;
      }
#endif
      measurementsOp = dcirc->GetLastMeasurements(executed, isQiskitAer);
      const auto &qbits = measurementsOp->GetQubits();
      if (qbits.empty()) {
        auto bits = state.GetAllBits();
        bits.resize(nrResultCbits, false);

        res[bits] += curCnt;

        return;
      }
    }

    if (optimiseMultipleShots &&
        (specialOptimizationForStatevector || hasMeasurementsOnlyAtEnd)) {
      const auto &qbits = measurementsOp->GetQubits();

      const auto sampleres = optSim->SampleCountsMany(qbits, curCnt);

      for (const auto &[mstate, cnt] : sampleres) {
        measurementsOp->SetStateFromSample(mstate, state);

        auto bits = state.GetAllBits();
        bits.resize(nrResultCbits, false);

        res[bits] += cnt;

        state.Reset();
      }

      return;
    }

    const auto curCnt1 = curCnt > 0 ? curCnt - 1 : 0;
    for (size_t i = 0; i < curCnt; ++i) {
      if (optimiseMultipleShots) {
        if (i > 0) {
          optSim->RestoreState();
          optSim->SetGatesCounter(0);
        }
        dcirc->ExecuteMeasurements(optSim, state, executed);
      } else {
        dcirc->Execute(optSim, state);
        if (i < curCnt1) {
          optSim->Reset();  // leave the simulator state for the last iteration
          optSim->SetGatesCounter(0);
        }
      }

      auto bits = state.GetAllBits();
      bits.resize(nrResultCbits, false);

      ++res[bits];

      state.Reset();
    }
  }

  static bool IsOptimisableForMultipleShots(Simulators::SimulatorType t,
                                            size_t curCnt) {
    return curCnt > 1;
  }

  size_t GetJobCount() const { return curCnt; }

 private:
  void OptimizeMPSInitialQubitsMap(
      std::shared_ptr<Simulators::ISimulator> &sim,
      std::shared_ptr<Circuits::Circuit<Time>> &dcirc, size_t nrQubits) const {
    if (sim->GetSimulationType() ==
            Simulators::SimulationType::kMatrixProductState &&
        (network->GetInitialQubitsMapOptimization() ||
         network->GetMPSOptimizeSwaps()) &&
        sim->SupportsMPSSwapOptimization()) {
      if (network->GetMPSOptimizationQubitsNumberThreshold() <= nrQubits) {
        const auto bondDimThreshold =
            network->GetMPSOptimizationBondDimensionThreshold();
        const auto maxBondDimValue =
            maxBondDim.empty() ? 0 : std::stoi(maxBondDim);

        if (maxBondDim.empty() ||
            static_cast<int>(bondDimThreshold) <= maxBondDimValue) {
          // need to be sure the circuit is correctly converted
          dcirc->ConvertForCutting();  // convert the three qubit gates
          auto layers = dcirc->ToMultipleQubitsLayersNoClone();

          Simulators::MPSDummySimulator dummySim(nrQubits);
          dummySim.setGrowthFactorGate(network->getGrowthFactorGate());
          dummySim.setGrowthFactorSwap(network->getGrowthFactorSwap());
          if (!maxBondDim.empty())
            dummySim.SetMaxBondDimension(maxBondDimValue);

          if (network->GetInitialQubitsMapOptimization()) {
            const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
            sim->SetInitialQubitsMap(optimalMap);
          }

          dcirc = Circuits::Circuit<Time>::LayersToCircuit(layers);

          if (network->GetMPSOptimizeSwaps()) {
            // TODO: come up with something better!
            int lookaheadDepthLocal = network->GetLookaheadDepth();

            if (lookaheadDepthLocal == std::numeric_limits<int>::max()) {
              double avgTwoQubitGatesPerLayer = 0.0;
              for (const auto &layer : layers) {
                int twoQubitGates = 0;
                for (const auto &op : layer->GetOperations()) {
                  if (op->AffectedQubits().size() >= 2) {
                    ++twoQubitGates;
                  }
                }
                avgTwoQubitGatesPerLayer += twoQubitGates;
              }
              avgTwoQubitGatesPerLayer /= layers.size();

              int lookaheadVal =
                  static_cast<int>(4. * avgTwoQubitGatesPerLayer);
              if (lookaheadVal > 15) lookaheadVal = 15;

              lookaheadDepthLocal = layers.size() < 10 || nrQubits <= 10 ? 0
                                    : layers.size() < 20
                                        ? static_cast<int>(lookaheadVal)
                                    : layers.size() < 35 ? 1.5 * lookaheadVal
                                                         : 2 * lookaheadVal;
            }

            int lookaheadHeuristicDepthLocal =
                network->GetLookaheadDepthWithHeuristic();

            if (lookaheadHeuristicDepthLocal == std::numeric_limits<int>::max())
              lookaheadHeuristicDepthLocal =
                  layers.size() < 10 || nrQubits <= 10 ? 0
                  : layers.size() < 20                 ? lookaheadDepthLocal - 1
                                       : lookaheadDepthLocal - 2;

            sim->SetUseOptimalMeetingPosition(true);
            sim->SetLookaheadDepth(lookaheadDepthLocal);
            sim->SetLookaheadDepthWithHeuristic(lookaheadHeuristicDepthLocal);
            sim->setGrowthFactorGate(network->getGrowthFactorGate());
            sim->setGrowthFactorSwap(network->getGrowthFactorSwap());
            sim->SetUpcomingGates(dcirc->GetOperations());
          }
        }
      }
    }
  }

 public:
  std::shared_ptr<Circuits::Circuit<Time>> dcirc;
  ExecuteResults &res;
  const size_t curCnt;
  const size_t nrQubits;
  const size_t nrCbits;
  const size_t nrResultCbits;

  const Simulators::SimulatorType simType;
  const Simulators::SimulationType method;
  std::mutex &resultsMutex;

  bool optimiseMultipleShotsExecution = true;
  std::shared_ptr<Simulators::ISimulator> optSim;
  std::vector<bool> executedGates;

  // only fill them if passing null simulator
  std::string maxBondDim;
  std::string singularValueThreshold;
  std::string mpsSample;

  std::shared_ptr<Network::INetwork<Time>> network;
};

}  // namespace Network

#endif  // ! _NETWORK_JOB_H
