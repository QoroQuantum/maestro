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
#include "../Utils/Threads/ThreadsPool.h"

#include "../Simulators/Core/RandomSeed.h"
#include "../Simulators/TensorNetworks/MPSDummySimulator.h"

#include "Network.h"

#include "Configuration.h"
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Network
{

template <typename Time = Types::time_type> class ExecuteJob
{
  public:
    using ExecuteResults = typename Circuits::Circuit<Time>::ExecuteResults;
    std::function<void(const Simulators::ISimulator &)> onSimulatorReady;

    ExecuteJob() = delete;

    explicit ExecuteJob(const std::shared_ptr<Circuits::Circuit<Time>> &c, ExecuteResults &r, size_t cnt, size_t nq, size_t nc, size_t ncr,
                        Simulators::SimulatorType t, Simulators::SimulationType m, std::mutex &mut)
        : dcirc(c), res(r), curCnt(cnt), nrQubits(nq), nrCbits(nc), nrResultCbits(ncr), simType(t), method(m), resultsMutex(mut)
    {
    }

    void DoWork()
    {
        if (curCnt == 0)
            return;
        ScopedInternalThreads threadBudget(internalThreads);
        if (cloneSource)
        {
            // Clone only when a worker starts, so queued logical blocks do not
            // each retain another statevector/density matrix. Fusion flushes
            // during cloning, hence the short lock around the source.
            const std::lock_guard lock(*cloneMutex);
            optSim = cloneSource->CloneForExecution(std::stoull(config.GetConfiguration("seed")));
        }
        // The dispatcher assigns this job a share of the internal thread budget.
        ConfigureTrajectoryArithmetic();
        if (optSim && optSim->GetType() == Simulators::SimulatorType::kQCSim &&
            (method == Simulators::SimulationType::kPauliPropagator || optSim->GetConfiguration("sampling_policy") == "reproducible_v1"))
            optSim->SetMultithreading(allowInternalMultithreading);

        // Per-shot streams must clone the original instruction seeds, not a
        // job-seeded copy (which would make classical RNGs depend on batching).
        if (!independentShotStreams)
            PrepareCircuitForExecution();

        Circuits::OperationState state;
        state.AllocateBits(nrCbits);

        const bool hasMeasurementsOnlyAtEnd = !dcirc->HasOpsAfterMeasurements();
        const bool optimiseMultipleShots = optimiseMultipleShotsExecution;
        const bool specialOptimizationForStatevector = optimiseMultipleShots && method == Simulators::SimulationType::kStatevector && hasMeasurementsOnlyAtEnd;

        dcirc = dcirc->RemoveExecutedOperations(executedGates);

        size_t curMaxBondDimLocal = 0;

        if (!optSim)
        {
            optSim = Simulators::SimulatorsFactory::CreateSimulator(simType, method);
            if (!optSim)
                return;
            config.ApplyConfigurationToSimulator(optSim);
            if (optSim->GetType() == Simulators::SimulatorType::kQCSim &&
                (method == Simulators::SimulationType::kPauliPropagator || optSim->GetConfiguration("sampling_policy") == "reproducible_v1"))
                optSim->SetMultithreading(allowInternalMultithreading);
            optSim->AllocateQubits(nrQubits);
            optSim->Initialize();

            OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);

            if (optimiseMultipleShots)
            {
                executedGates = dcirc->ExecuteNonMeasurements(optSim, state, &curMaxBondDimLocal);

                dcirc = dcirc->RemoveExecutedOperations(executedGates);
                if ((method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator) &&
                    network->GetMPSOptimizeSwaps())
                {
                    // auto circ =
                    // std::static_pointer_cast<Circuits::Circuit<Time>>(dcirc->Clone());
                    // circ->ConvertForCutting();
                    optSim->SetUpcomingGates(dcirc->GetOperations());
                }
                // The snapshot must describe the suffix executed by every shot.
                if (!hasMeasurementsOnlyAtEnd && curCnt > 1)
                    optSim->SaveState();
            }
        }
        else if ((method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator) &&
                 network->GetMPSOptimizeSwaps())
        {
            auto circ = std::static_pointer_cast<Circuits::Circuit<Time>>(dcirc->Clone());
            circ->ConvertForCutting();
            optSim->SetUpcomingGates(circ->GetOperations());
        }

        if (onSimulatorReady)
            onSimulatorReady(*optSim);
        std::shared_ptr<Circuits::MeasurementOperation<Time>> measurementsOp;

        const std::vector<bool> executed = std::move(executedGates);

        if (optimiseMultipleShots && hasMeasurementsOnlyAtEnd)
        {
            bool isQiskitAer = false;
#ifndef NO_QISKIT_AER
            if (optSim->GetType() == Simulators::SimulatorType::kQiskitAer)
            {
                isQiskitAer = true;
            }
#endif
            measurementsOp = dcirc->GetLastMeasurements(executed, isQiskitAer);
            const auto &qbits = measurementsOp->GetQubits();
            if (qbits.empty())
            {
                auto bits = state.GetAllBits();
                bits.resize(nrResultCbits, false);

                const std::lock_guard lock(resultsMutex);
                res[bits] += curCnt;

                if (curMaxBondDim && curMaxBondDimLocal > *curMaxBondDim)
                    *curMaxBondDim = curMaxBondDimLocal;

                return;
            }
        }

        ExecuteResults localRes;

        if (optimiseMultipleShots && (specialOptimizationForStatevector || hasMeasurementsOnlyAtEnd))
        {
            const auto &qbits = measurementsOp->GetQubits();

            const auto sampleres = optSim->SampleCountsMany(qbits, curCnt);

            AccumulateSamples(*measurementsOp, sampleres, state, localRes);

            const std::lock_guard lock(resultsMutex);
            for (const auto &r : localRes)
                res[r.first] += r.second;

            if (curMaxBondDim && curMaxBondDimLocal > *curMaxBondDim)
                *curMaxBondDim = curMaxBondDimLocal;

            return;
        }

        const auto curCnt1 = curCnt > 0 ? curCnt - 1 : 0;
        const auto shotCircuit = dcirc;
        for (size_t i = 0; i < curCnt; ++i)
        {
            if (independentShotStreams)
            {
                const uint64_t seed = Simulators::IState::DeriveSeed(trajectorySeed, firstShot + i);
                optSim->SetSeed(seed); // Backend and readout have separate streams.
                dcirc = shotCircuit->CloneForExecution(Simulators::IState::DeriveSeed(seed, 0x434c415353494341ULL));
            }
            if (optimiseMultipleShots)
            {
                if (i > 0)
                {
                    optSim->RestoreState();
                    optSim->SetGatesCounter(0);
                }
                dcirc->ExecuteMeasurements(optSim, state, executed, &curMaxBondDimLocal);
            }
            else
            {
                dcirc->ExecuteBD(optSim, state, &curMaxBondDimLocal);
                if (i < curCnt1)
                {
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
        for (const auto &r : localRes)
            res[r.first] += r.second;

        if (curMaxBondDim && curMaxBondDimLocal > *curMaxBondDim)
            *curMaxBondDim = curMaxBondDimLocal;
    }

    void DoWorkNoLock()
    {
        if (curCnt == 0)
            return;

        ConfigureTrajectoryArithmetic();
        PrepareCircuitForExecution();

        Circuits::OperationState state;
        state.AllocateBits(nrCbits);

        const bool hasMeasurementsOnlyAtEnd = !dcirc->HasOpsAfterMeasurements();
        const bool optimiseMultipleShots = optimiseMultipleShotsExecution;
        const bool specialOptimizationForStatevector = optimiseMultipleShots && method == Simulators::SimulationType::kStatevector && hasMeasurementsOnlyAtEnd;

        if (optSim)
        {
            optSim->SetMultithreading(true);

            if (optSim->GetNumberOfQubits() != nrQubits)
            {
                optSim->Clear();
                config.ApplyConfigurationToSimulator(optSim);

                optSim->AllocateQubits(nrQubits);
                optSim->Initialize();

                OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);

                if (optimiseMultipleShots)
                {
                    executedGates = dcirc->ExecuteNonMeasurements(optSim, state, curMaxBondDim);

                    dcirc = dcirc->RemoveExecutedOperations(executedGates);
                    if ((method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator) &&
                        network->GetMPSOptimizeSwaps())
                    {
                        // auto circ =
                        // std::static_pointer_cast<Circuits::Circuit<Time>>(dcirc->Clone());
                        // circ->ConvertForCutting();
                        optSim->SetUpcomingGates(dcirc->GetOperations());
                    }
                    if (!hasMeasurementsOnlyAtEnd && curCnt > 1)
                        optSim->SaveState();
                }
            }
            else if (executedGates.size() == dcirc->size())
            {
                // special case for when the simulator is passed from the network
                // and no gates were executed yet
                bool needToExecuteGates = true;
                for (const bool val : executedGates)
                {
                    if (val)
                    {
                        needToExecuteGates = false;
                        break;
                    }
                }
                if (needToExecuteGates && optimiseMultipleShots)
                {
                    executedGates = dcirc->ExecuteNonMeasurements(optSim, state, curMaxBondDim);
                    dcirc = dcirc->RemoveExecutedOperations(executedGates);
                    if ((method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator) &&
                        network->GetMPSOptimizeSwaps())
                    {
                        // auto circ =
                        // std::static_pointer_cast<Circuits::Circuit<Time>>(dcirc->Clone());
                        // circ->ConvertForCutting();
                        optSim->SetUpcomingGates(dcirc->GetOperations());
                    }
                    if (!hasMeasurementsOnlyAtEnd && curCnt > 1)
                        optSim->SaveState();
                }
                else
                {
                    dcirc = dcirc->RemoveExecutedOperations(executedGates);
                    if ((method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator) &&
                        network->GetMPSOptimizeSwaps())
                    {
                        // auto circ =
                        // std::static_pointer_cast<Circuits::Circuit<Time>>(dcirc->Clone());
                        // circ->ConvertForCutting();
                        optSim->SetUpcomingGates(dcirc->GetOperations());
                    }
                }
            }
            else
            {
                dcirc = dcirc->RemoveExecutedOperations(executedGates);
                if ((method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator) &&
                    network->GetMPSOptimizeSwaps())
                {
                    auto circ = std::static_pointer_cast<Circuits::Circuit<Time>>(dcirc->Clone());
                    circ->ConvertForCutting();
                    optSim->SetUpcomingGates(circ->GetOperations());
                }
            }
        }
        else
        {
            optSim = Simulators::SimulatorsFactory::CreateSimulator(simType, method);
            if (!optSim)
                return;

            optSim->SetMultithreading(true);
            config.ApplyConfigurationToSimulator(optSim);

            optSim->AllocateQubits(nrQubits);
            optSim->Initialize();

            OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);

            if (optimiseMultipleShots)
            {
                executedGates = dcirc->ExecuteNonMeasurements(optSim, state, curMaxBondDim);
                dcirc = dcirc->RemoveExecutedOperations(executedGates);
                if ((method == Simulators::SimulationType::kMatrixProductState || method == Simulators::SimulationType::kMatrixProductOperator) &&
                    network->GetMPSOptimizeSwaps())
                {
                    // auto circ =
                    // std::static_pointer_cast<Circuits::Circuit<Time>>(dcirc->Clone());
                    // circ->ConvertForCutting();
                    optSim->SetUpcomingGates(dcirc->GetOperations());
                }
                if (!hasMeasurementsOnlyAtEnd && curCnt > 1)
                    optSim->SaveState();
            }
        }

        if (onSimulatorReady)
            onSimulatorReady(*optSim);
        std::shared_ptr<Circuits::MeasurementOperation<Time>> measurementsOp;

        const std::vector<bool> executed = std::move(executedGates);

        if (optimiseMultipleShots && hasMeasurementsOnlyAtEnd)
        {
            bool isQiskitAer = false;
#ifndef NO_QISKIT_AER
            if (optSim->GetType() == Simulators::SimulatorType::kQiskitAer)
            {
                isQiskitAer = true;
            }
#endif
            measurementsOp = dcirc->GetLastMeasurements(executed, isQiskitAer);
            const auto &qbits = measurementsOp->GetQubits();
            if (qbits.empty())
            {
                auto bits = state.GetAllBits();
                bits.resize(nrResultCbits, false);

                res[bits] += curCnt;

                return;
            }
        }

        if (optimiseMultipleShots && (specialOptimizationForStatevector || hasMeasurementsOnlyAtEnd))
        {
            const auto &qbits = measurementsOp->GetQubits();

            const auto sampleres = optSim->SampleCountsMany(qbits, curCnt);

            AccumulateSamples(*measurementsOp, sampleres, state, res);

            return;
        }

        const auto curCnt1 = curCnt > 0 ? curCnt - 1 : 0;
        for (size_t i = 0; i < curCnt; ++i)
        {
            if (optimiseMultipleShots)
            {
                if (i > 0)
                {
                    optSim->RestoreState();
                    optSim->SetGatesCounter(0);
                }
                dcirc->ExecuteMeasurements(optSim, state, executed, curMaxBondDim);
            }
            else
            {
                dcirc->ExecuteBD(optSim, state, curMaxBondDim);
                if (i < curCnt1)
                {
                    optSim->Reset(); // leave the simulator state for the last iteration
                    optSim->SetGatesCounter(0);
                }
            }

            auto bits = state.GetAllBits();
            bits.resize(nrResultCbits, false);

            ++res[bits];

            state.Reset();
        }
    }

    static bool IsOptimisableForMultipleShots(Simulators::SimulatorType t, size_t curCnt)
    {
        return curCnt > 1;
    }

    size_t GetJobCount() const
    {
        return curCnt;
    }

  private:
    void ConfigureTrajectoryArithmetic()
    {
        // Existing and prepared simulators need the current execution mode too:
        // a terminal batch following trajectories must not retain this mode.
        if (optSim && optSim->GetType() == Simulators::SimulatorType::kQCSim && config.IsSet("reproducible_trajectory"))
            optSim->Configure("reproducible_trajectory", config.GetConfiguration("reproducible_trajectory").c_str());
    }

    class ScopedInternalThreads
    {
      public:
        explicit ScopedInternalThreads(int threads)
        {
#ifdef _OPENMP
            if (threads > 0)
            {
                previous = omp_get_max_threads();
                omp_set_num_threads(threads);
            }
#else
            (void)threads;
#endif
        }

        ~ScopedInternalThreads()
        {
#ifdef _OPENMP
            if (previous > 0)
                omp_set_num_threads(previous);
#endif
        }

      private:
#ifdef _OPENMP
        int previous = 0;
#endif
    };

    template <class Samples>
    void AccumulateSamples(Circuits::MeasurementOperation<Time> &measurement, const Samples &samples, Circuits::OperationState &state, ExecuteResults &target)
    {
        auto add = [&](const auto &sample) {
            const auto &[bits, count] = sample;
            const size_t draws = measurement.HasReadout() ? count : 1;
            for (size_t i = 0; i < draws; ++i)
            {
                measurement.SetStateFromSample(bits, state, measurement.HasReadout() ? optSim.get() : nullptr);
                auto output = state.GetAllBits();
                output.resize(nrResultCbits, false);
                target[output] += measurement.HasReadout() ? 1 : count;
                state.Reset();
            }
        };
        if (measurement.HasReadout() && optSim->GetConfiguration("sampling_policy") == "reproducible_v1")
        {
            // Histogram insertion order can change with the execution kernel.
            // Apply auxiliary readout draws in lexicographic outcome order.
            std::vector<const typename Samples::value_type *> ordered;
            ordered.reserve(samples.size());
            for (const auto &sample : samples)
                ordered.push_back(&sample);
            std::sort(ordered.begin(), ordered.end(), [](auto a, auto b) { return a->first < b->first; });
            for (const auto *sample : ordered)
                add(*sample);
        }
        else
            for (const auto &sample : samples)
                add(sample);
    }

    void PrepareCircuitForExecution()
    {
        const uint64_t stream = config.IsSet("seed") ? std::stoull(config.GetConfiguration("seed")) : randomStream;
        // Only unseeded instructions need fresh entropy. For MPI the helper shares
        // it across ranks, so classical controls follow the same execution path.
        const uint64_t defaultSeed =
            !config.IsSet("seed") && dcirc->HasUnseededRandomOperations() ? Simulators::GenerateRandomSeed(simType, config.GetConfigMap()) : 0;
        dcirc = dcirc->CloneForExecution(stream, defaultSeed);
    }

    void OptimizeMPSInitialQubitsMap(std::shared_ptr<Simulators::ISimulator> &sim, std::shared_ptr<Circuits::Circuit<Time>> &dcirc, size_t nrQubits) const
    {
        if ((sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductState ||
             sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator) &&
            (network->GetInitialQubitsMapOptimization() || network->GetMPSOptimizeSwaps()) && sim->SupportsMPSSwapOptimization() &&
            !dcirc->HasCompositeOperations())
        {
            // an MPO site carries a two-qubit (operator) physical index, so its
            // bonds grow like those of an MPS twice as long
            const size_t routedQubits = sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator ? 2 * nrQubits : nrQubits;
            if (network->GetMPSOptimizationQubitsNumberThreshold() <= routedQubits)
            {
                const auto bondDimThreshold = network->GetMPSOptimizationBondDimensionThreshold();
                const auto maxBondDimValue = config.GetConfigurationAsInt(sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator &&
                                                                                  config.IsSet("matrix_product_operator_max_bond_dimension")
                                                                              ? "matrix_product_operator_max_bond_dimension"
                                                                              : "matrix_product_state_max_bond_dimension");

                if (maxBondDimValue == 0 || static_cast<int>(bondDimThreshold) <= maxBondDimValue)
                {
                    // need to be sure the circuit is correctly converted
                    dcirc->ConvertForCutting(); // convert the three qubit gates
                    auto layers = dcirc->ToMultipleQubitsLayersNoClone();
                    // Finalize source ordering before preparing fusion. Routing layers
                    // are only a cost model and must never replace executable operations.
                    auto ordered = Circuits::Circuit<Time>::LayersToCircuit(layers);
                    dcirc->SetOperations(ordered->GetOperations());
                    sim->SetUpcomingGates(dcirc->GetOperations());
                    if (sim->IsGateFusionEnabled() && !sim->GetUpcomingRoutingOperations().empty())
                    {
                        Circuits::Circuit<Time> routingCircuit;
                        for (const auto &op : sim->GetUpcomingRoutingOperations())
                            if (op->GetType() == Circuits::OperationType::kGate)
                                routingCircuit.AddOperation(op);
                        layers = routingCircuit.ToMultipleQubitsLayersNoClone();
                    }

                    Simulators::MPSDummySimulator dummySim(nrQubits, sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator);
                    dummySim.setGrowthFactorGate(network->getGrowthFactorGate());
                    dummySim.setGrowthFactorSwap(network->getGrowthFactorSwap());
                    if (maxBondDimValue != 0)
                        dummySim.SetMaxBondDimension(maxBondDimValue);

                    if (network->GetInitialQubitsMapOptimization())
                    {
                        const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
                        sim->SetInitialQubitsMap(optimalMap);
                    }

                    if (network->GetMPSOptimizeSwaps())
                    {
                        // TODO: come up with something better!
                        int lookaheadDepthLocal = network->GetLookaheadDepth();

                        if (lookaheadDepthLocal == std::numeric_limits<int>::max())
                        {
                            double avgTwoQubitGatesPerLayer = 0.0;
                            for (const auto &layer : layers)
                            {
                                int twoQubitGates = 0;
                                for (const auto &op : layer->GetOperations())
                                {
                                    if (op->AffectedQubits().size() >= 2)
                                    {
                                        ++twoQubitGates;
                                    }
                                }
                                avgTwoQubitGatesPerLayer += twoQubitGates;
                            }
                            avgTwoQubitGatesPerLayer /= layers.size();

                            int lookaheadVal = static_cast<int>(4. * avgTwoQubitGatesPerLayer);
                            if (lookaheadVal > 15)
                                lookaheadVal = 15;

                            lookaheadDepthLocal = layers.size() < 8 || routedQubits <= 10 ? 0
                                                  : layers.size() < 15                    ? static_cast<int>(lookaheadVal)
                                                  : layers.size() < 25                    ? static_cast<int>(1.5 * lookaheadVal)
                                                                                          : 2 * lookaheadVal;
                        }

                        int lookaheadHeuristicDepthLocal = network->GetLookaheadDepthWithHeuristic();

                        if (lookaheadHeuristicDepthLocal == std::numeric_limits<int>::max())
                            lookaheadHeuristicDepthLocal = layers.size() < 10 || routedQubits <= 10 ? 0
                                                           : layers.size() < 20                     ? lookaheadDepthLocal - 1
                                                                                                    : lookaheadDepthLocal - 2;

                        if (lookaheadHeuristicDepthLocal < 0)
                            lookaheadHeuristicDepthLocal = 0;

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
    bool allowInternalMultithreading = false;
    // Zero leaves the caller's OpenMP settings untouched (ordinary jobs).
    int internalThreads = 0;
    // Distinguishes jobs even when the caller did not configure a simulator seed.
    uint64_t randomStream = 0;
    bool independentShotStreams = false;
    uint64_t trajectorySeed = 0;
    size_t firstShot = 0;
    std::shared_ptr<Simulators::ISimulator> optSim;
    std::shared_ptr<Simulators::ISimulator> cloneSource;
    std::shared_ptr<std::mutex> cloneMutex;
    std::vector<bool> executedGates;

    // relevant only if the simulator is not passed or the simulator doesn't have
    // the proper number of qubits, otherwise the simulator is already configured
    Configuration<Time> config;

    std::shared_ptr<Network::INetwork<Time>> network;
    size_t *curMaxBondDim = nullptr;
};

} // namespace Network

#endif // ! _NETWORK_JOB_H
