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
#include "../Utils/Threads/RealizationWorkers.h"
#include "../Utils/Threads/ThreadsPool.h"

#include "../Circuit/ChainOrder.h"
#include "../Simulators/Core/RandomSeed.h"
#include "../Simulators/TensorNetworks/MPSDummySimulator.h"

#include "Network.h"

#include "Configuration.h"
#include <algorithm>
#include <condition_variable>
#include <exception>
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

        // If this job fails, the jobs of the execution that wait for a clone
        // turn are released: the pool runs no job it takes after a failure, so
        // the turns of those jobs never come.
        struct CancelCloneTurns
        {
            std::shared_ptr<CloneOrder> order;
            int exceptions = std::uncaught_exceptions();

            ~CancelCloneTurns()
            {
                if (!order || std::uncaught_exceptions() <= exceptions)
                    return;
                {
                    const std::lock_guard lock(order->mutex);
                    order->cancelled = true;
                }
                order->turn.notify_all();
            }
        } cancelCloneTurns{cloneOrder};

        ScopedInternalThreads threadBudget(internalThreads);
        if (cloneSource)
        {
            // Clone only when a worker starts, so queued jobs do not each
            // retain another statevector/density matrix. Fusion flushes
            // during cloning, hence the short lock around the source.
            if (cloneOrder)
            {
                // The shot workers outside the logical shot blocks: a full
                // clone, seeded only when the job has a seed. Clones are made
                // in job order, as when each job was given its clone before it
                // was queued: Clone() advances the source's own clone seeds,
                // so the order decides what an unseeded job draws from a
                // seeded source. The pool hands jobs out in order, so the
                // earlier jobs are already running and cloning.
                std::unique_lock lock(cloneOrder->mutex);
                cloneOrder->turn.wait(lock, [&] { return cloneOrder->cancelled || cloneOrder->next == cloneIndex; });
                // Another job of the execution failed: its exception is what the
                // caller gets, and this job's shots are not run.
                if (cloneOrder->cancelled)
                    return;

                // The next job's turn comes also if the clone throws.
                struct Advance
                {
                    CloneOrder &order;
                    std::unique_lock<std::mutex> &lock;

                    ~Advance()
                    {
                        ++order.next;
                        lock.unlock();
                        order.turn.notify_all();
                    }
                } advance{*cloneOrder, lock};

                optSim = cloneSource->Clone();
                if (config.IsSet("seed"))
                    optSim->SetSeed(std::stoull(config.GetConfiguration("seed")));
            }
            else
            {
                const std::lock_guard lock(*cloneMutex);
                optSim = cloneSource->CloneForExecution(std::stoull(config.GetConfiguration("seed")));
            }
        }
        // The dispatcher assigns this job a share of the internal thread budget.
        ConfigureTrajectoryArithmetic();
        if (optSim && optSim->GetType() == Simulators::SimulatorType::kQCSim && UsesAssignedThreads())
            optSim->SetMultithreading(allowInternalMultithreading);
        // Multithreading off has to hold for what the simulator calls on this thread too, not only for
        // its own loops: a BLAS built on OpenMP would otherwise start a thread team for each shot worker.
        threadBudget.Limit(SimulatorThreadLimit());

        // Per-shot streams must clone the original instruction seeds, not a
        // job-seeded copy (which would make classical RNGs depend on batching).
        if (!independentShotStreams)
            PrepareCircuitForExecution();

        Circuits::OperationState state;
        state.AllocateBits(nrCbits);

        const bool hasMeasurementsOnlyAtEnd = HasTerminalMeasurements();
        const bool optimiseMultipleShots = optimiseMultipleShotsExecution;
        const bool specialOptimizationForStatevector = optimiseMultipleShots && method == Simulators::SimulationType::kStatevector && hasMeasurementsOnlyAtEnd;

        size_t curMaxBondDimLocal = 0;

        // A retained simulator may contain the caller's input state rather than
        // an executed prefix. Prepare the same suffix as a newly created worker
        // so per-shot classical RNG positions do not depend on simulator reuse.
        if (optSim && optimiseMultipleShots && executedGates.size() == dcirc->size() &&
            std::none_of(executedGates.begin(), executedGates.end(), [](bool executed) { return executed; }))
        {
            executedGates = ExecuteSharedPrefix(state, &curMaxBondDimLocal);
            if (!hasMeasurementsOnlyAtEnd && curCnt > 1)
                optSim->SaveState();
        }

        dcirc = dcirc->RemoveExecutedOperations(executedGates);

        if (!optSim)
        {
            optSim = Simulators::SimulatorsFactory::CreateSimulator(simType, method);
            if (!optSim)
                return;
            ApplyConfiguration(optSim);
            if (optSim->GetType() != Simulators::SimulatorType::kQCSim || UsesAssignedThreads())
                optSim->SetMultithreading(allowInternalMultithreading);
            threadBudget.Limit(SimulatorThreadLimit());
            optSim->AllocateQubits(nrQubits);
            optSim->Initialize();
            ApplyTransientSeed(optSim);

            OptimizeMPSInitialQubitsMap(optSim, dcirc, nrQubits);

            if (optimiseMultipleShots)
            {
                executedGates = ExecuteSharedPrefix(state, &curMaxBondDimLocal);

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
        // Only classical generators own a stream; without them every shot can
        // run the same circuit instead of a copy of its operations.
        const bool classicalStreams = independentShotStreams && OwnsClassicalStreams(*shotCircuit);
        for (size_t i = 0; i < curCnt; ++i)
        {
            if (independentShotStreams)
            {
                const uint64_t seed = Simulators::IState::DeriveSeed(trajectorySeed, firstShot + i);
                // Backend and readout have separate streams.
                if (transientSeed)
                    optSim->SeedForExecution(seed);
                else
                    optSim->SetSeed(seed);
                if (classicalStreams)
                {
                    // An explicit instruction seed must remain reproducible even
                    // when the request has no seed. Fresh request entropy belongs
                    // only to instructions without their own seed.
                    const uint64_t classicalStream = Simulators::IState::DeriveSeed(transientSeed ? firstShot + i : seed, 0x434c415353494341ULL);
                    dcirc = shotCircuit->CloneForExecution(classicalStream, transientSeed ? seed : 0);
                }
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
                state.Reset();
                dcirc->ExecutePhysicalOperations(optSim, state, &curMaxBondDimLocal);
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

        const bool hasMeasurementsOnlyAtEnd = HasTerminalMeasurements();
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
                    executedGates = ExecuteSharedPrefix(state, curMaxBondDim);

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
                    executedGates = ExecuteSharedPrefix(state, curMaxBondDim);
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
                executedGates = ExecuteSharedPrefix(state, curMaxBondDim);
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
                state.Reset();
                dcirc->ExecutePhysicalOperations(optSim, state, curMaxBondDim);
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

    // A transient seed is left out of the simulator configuration. It seeds the
    // job's streams through ApplyTransientSeed once the simulator is initialized.
    void ApplyConfiguration(const std::shared_ptr<Simulators::ISimulator> &sim) const
    {
        if (!transientSeed)
        {
            config.ApplyConfigurationToSimulator(sim);
            return;
        }
        auto unseeded = config;
        unseeded.RemoveConfiguration("seed");
        unseeded.ApplyConfigurationToSimulator(sim);
    }

    void ApplyTransientSeed(const std::shared_ptr<Simulators::ISimulator> &sim) const
    {
        if (transientSeed)
            sim->SeedForExecution(std::stoull(config.GetConfiguration("seed")));
    }

    // The QCSim methods whose seeded samples do not depend on how the shots
    // are split into jobs: the network schedules them as logical shot blocks.
    static bool ReproducibleMethod(Simulators::SimulatorType type, Simulators::SimulationType method)
    {
        using Method = Simulators::SimulationType;
        return type == Simulators::SimulatorType::kQCSim &&
               (method == Method::kStatevector || method == Method::kDensityMatrix || method == Method::kPathIntegral || method == Method::kStabilizer ||
                method == Method::kExtendedStabilizer || method == Method::kMatrixProductState || method == Method::kMatrixProductOperator);
    }

  private:
    std::vector<bool> ExecuteSharedPrefix(Circuits::OperationState &state, size_t *maxBond) const
    {
        const Simulators::ShotPrefixPolicy byWire{false, false};
        return dcirc->ExecuteNonMeasurements(optSim, state, maxBond, false, false, shareResetWires ? &byWire : nullptr);
    }

    bool HasTerminalMeasurements()
    {
        // A serial job can receive a smaller template register which it will resize and configure.
        // Whether a bond cap can truncate depends on the width that actually executes the circuit.
        const auto policy = optSim && optSim->GetNumberOfQubits() == nrQubits
                                ? optSim->GetShotPrefixPolicy()
                                : Simulators::ShotPrefixPolicy::FromConfiguration(simType, method, config.GetConfigMap(), {}, nrQubits);
        if (!dcirc->HasOpsAfterMeasurementsOn(policy))
            return true;
        // The network plans before dispatch. A standalone job can also do so
        // when its initial register is known to be fresh and no prefix ran.
        if (policy.contiguous && method == Simulators::SimulationType::kMatrixProductState && curCnt > 1 && optimiseMultipleShotsExecution &&
            !network->GetOptimizeSimulator() && (!optSim || startsFromZero) && !independentShotStreams &&
            std::none_of(executedGates.begin(), executedGates.end(), [](bool ran) { return ran; }))
        {
            auto terminal = Circuits::IsolatedMpsTerminalOrder(dcirc, nrQubits, simType);
            if (terminal != dcirc)
            {
                dcirc = std::move(terminal);
                return true;
            }
        }
        return false;
    }

    // Whether the job's simulator samples reproducibly. Only the dense methods
    // report that policy when it is unset, so the others count unless they were
    // configured for the ordinary sampler: setting the default explicitly then
    // changes nothing.
    bool SamplesReproducibly() const
    {
        return optSim && ReproducibleMethod(optSim->GetType(), method) && optSim->GetConfiguration("sampling_policy") != "legacy";
    }

    // Whether the simulator may use the internal threads assigned to this job:
    // the Pauli propagator, and the dense methods under the reproducible policy,
    // whose measurement keeps a serial scan and reduction order.
    bool UsesAssignedThreads() const
    {
        using Method = Simulators::SimulationType;
        if (method == Method::kPauliPropagator)
            return true;
        return (method == Method::kStatevector || method == Method::kDensityMatrix || method == Method::kPathIntegral) &&
               optSim->GetConfiguration("sampling_policy") == "reproducible_v1";
    }

    // Whether the circuit has a classical random generator, nested ones
    // included: the operations Circuit::CloneForExecution gives their own stream.
    static bool OwnsClassicalStreams(const Circuits::Circuit<Time> &circuit)
    {
        for (const auto &op : circuit.GetOperations())
        {
            const auto type = op->GetType();
            if (type == Circuits::OperationType::kRandomGen || type == Circuits::OperationType::kConditionalRandomGen)
                return true;
            if (type == Circuits::OperationType::kComposite && OwnsClassicalStreams(*std::static_pointer_cast<Circuits::Circuit<Time>>(op)))
                return true;
        }
        return false;
    }

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
            Limit(threads);
        }

        void Limit(int threads)
        {
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
            processBudget.Limit(threads);
#elif defined(_OPENMP)
            if (threads > 0)
            {
                if (previous == 0)
                    previous = omp_get_max_threads();
                omp_set_num_threads(threads);
            }
#else
            (void)threads;
#endif
        }

        ~ScopedInternalThreads()
        {
#if defined(_OPENMP) && (!defined(_MSC_VER) || defined(__clang__) || defined(_OPENMP_LLVM_RUNTIME))
            if (previous > 0)
                omp_set_num_threads(previous);
#endif
        }

      private:
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
        Utils::RealizationDetail::ScopedProcessOpenMPBudget processBudget{0};
#elif defined(_OPENMP)
        int previous = 0;
#endif
    };

    int SimulatorThreadLimit() const
    {
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__) && !defined(_OPENMP_LLVM_RUNTIME)
        // Legacy MSVC OpenMP keeps this setting process-wide, per-worker save/restore would race.
        return 0;
#else
        return optSim && !optSim->GetMultithreading() ? 1 : 0;
#endif
    }

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
        if (measurement.HasReadout() && SamplesReproducibly())
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
        const bool explicitSeed = config.IsSet("seed") && !transientSeed;
        const uint64_t stream = explicitSeed ? std::stoull(config.GetConfiguration("seed")) : randomStream;
        // Only unseeded instructions need fresh entropy. For MPI the helper shares
        // it across ranks, so classical controls follow the same execution path.
        const uint64_t defaultSeed =
            !explicitSeed && dcirc->HasUnseededRandomOperations()
                ? transientSeed ? std::stoull(config.GetConfiguration("seed")) : Simulators::GenerateRandomSeed(simType, config.GetConfigMap())
                : 0;
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
            bool planned = false;
            if (network->GetMPSOptimizationQubitsNumberThreshold() <= routedQubits)
            {
                const auto bondDimThreshold = network->GetMPSOptimizationBondDimensionThreshold();
                const auto maxBondDimValue = config.GetConfigurationAsInt(sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator &&
                                                                                  config.IsSet("matrix_product_operator_max_bond_dimension")
                                                                              ? "matrix_product_operator_max_bond_dimension"
                                                                              : "matrix_product_state_max_bond_dimension");

                if (maxBondDimValue == 0 || static_cast<int>(bondDimThreshold) <= maxBondDimValue)
                {
                    planned = true;
                    // need to be sure the circuit is correctly converted
                    dcirc->ConvertForCutting(); // convert the three qubit gates
                    auto layers = dcirc->ToMultipleQubitsLayersNoClone();
                    // Finalize source ordering before preparing fusion. Routing layers
                    // are only a cost model and must never replace executable operations.
                    auto ordered = sim->GetShotPrefixPolicy().contiguous ? dcirc : Circuits::Circuit<Time>::LayersToCircuit(layers);
                    // As in the network's copy of this function: qubits that form lines out of index
                    // order are placed along them, which needs no swap (Circuits::ChainLine).
                    const std::vector<long long int> line = network->GetInitialQubitsMapOptimization()
                                                                ? Circuits::ChainLine(ordered, nrQubits, sim->SwapGatesRelabelQubits())
                                                                : std::vector<long long int>();
                    if (ordered != dcirc)
                        dcirc->SetOperations(ordered->GetOperations());
                    sim->SetUpcomingGates(dcirc->GetOperations());
                    // Measurements and resets enter the model of a backend that reports them (the QCSim chains).
                    const bool tracksDisentangling = sim->GetType() == Simulators::SimulatorType::kQCSim;
                    const bool circuitHasDisentangle =
                        tracksDisentangling && Simulators::RoutingConfig::ContainsDisentanglingOperations(dcirc->GetOperations());
                    if (sim->IsGateFusionEnabled() && !sim->GetUpcomingRoutingOperations().empty())
                    {
                        Circuits::Circuit<Time> routingCircuit;
                        for (const auto &op : sim->GetUpcomingRoutingOperations())
                            if (op && (op->GetType() == Circuits::OperationType::kGate ||
                                       (tracksDisentangling && Simulators::RoutingConfig::IsDisentanglingOperationType(op->GetType()))))
                                routingCircuit.AddOperation(op);
                        layers = routingCircuit.ToMultipleQubitsLayersNoClone();
                    }

                    Simulators::MPSDummySimulator dummySim(nrQubits, sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator);
                    dummySim.SetSwapGatesRelabelQubits(sim->SwapGatesRelabelQubits());
                    dummySim.SetModelDisentangling(tracksDisentangling);
                    dummySim.setGrowthFactorGate(network->getGrowthFactorGate());
                    dummySim.setGrowthFactorSwap(network->getGrowthFactorSwap());
                    if (maxBondDimValue != 0)
                    {
                        const bool virtualBonds = Simulators::RoutingConfig::VirtualBondsEnabled(config) ||
                                                  (circuitHasDisentangle && Simulators::RoutingConfig::VirtualBondsAutoAllowed(config));
                        dummySim.SetMaxBondDimension(
                            virtualBonds ? Simulators::MPSDummySimulator::EffectiveMapMaxBondDimension(
                                               nrQubits, sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductOperator, maxBondDimValue)
                                         : maxBondDimValue);
                        dummySim.SetRoutingCostModel(virtualBonds ? maxBondDimValue : 0, Simulators::RoutingConfig::TruncationWeight(config));
                    }

                    if (network->GetInitialQubitsMapOptimization())
                    {
                        const auto optimalMap = line.empty() ? dummySim.ComputeOptimalQubitsMap(layers) : line;
                        sim->SetInitialQubitsMap(optimalMap);
                    }

                    if (network->GetMPSOptimizeSwaps())
                    {
                        int lookaheadDepthLocal = network->GetLookaheadDepth();
                        size_t effectiveLayerCount = layers.size();

                        if (lookaheadDepthLocal == std::numeric_limits<int>::max())
                        {
                            double avgTwoQubitGatesPerLayer = 0.0;
                            size_t realTwoQubitLayers = 0;
                            for (const auto &layer : layers)
                            {
                                int twoQubitGates = 0;
                                for (const auto &op : layer->GetOperations())
                                {
                                    if (op && op->GetType() == Circuits::OperationType::kGate && op->AffectedQubits().size() >= 2)
                                    {
                                        ++twoQubitGates;
                                    }
                                }
                                avgTwoQubitGatesPerLayer += twoQubitGates;
                                if (twoQubitGates > 0 || !circuitHasDisentangle)
                                    ++realTwoQubitLayers;
                            }
                            effectiveLayerCount = circuitHasDisentangle ? realTwoQubitLayers : layers.size();
                            if (effectiveLayerCount > 0)
                                avgTwoQubitGatesPerLayer /= static_cast<double>(effectiveLayerCount);

                            int lookaheadVal = static_cast<int>(4. * avgTwoQubitGatesPerLayer);
                            if (lookaheadVal > 15)
                                lookaheadVal = 15;

                            lookaheadDepthLocal = effectiveLayerCount < 8 || routedQubits <= 10 ? 0
                                                  : effectiveLayerCount < 15                    ? static_cast<int>(lookaheadVal)
                                                  : effectiveLayerCount < 25                    ? static_cast<int>(1.5 * lookaheadVal)
                                                                                                : 2 * lookaheadVal;
                        }

                        int lookaheadHeuristicDepthLocal = network->GetLookaheadDepthWithHeuristic();

                        if (lookaheadHeuristicDepthLocal == std::numeric_limits<int>::max())
                            lookaheadHeuristicDepthLocal = effectiveLayerCount < 8 || routedQubits <= 10 ? 0 : lookaheadDepthLocal - 1;

                        if (lookaheadHeuristicDepthLocal < 0)
                            lookaheadHeuristicDepthLocal = 0;

                        sim->SetUseOptimalMeetingPosition(true);
                        if (network->GetLookaheadDepth() == std::numeric_limits<int>::max())
                            sim->SetAutomaticLookaheadDepth(lookaheadDepthLocal);
                        else
                            sim->SetLookaheadDepth(lookaheadDepthLocal);
                        sim->SetLookaheadDepthWithHeuristic(lookaheadHeuristicDepthLocal);
                        sim->setGrowthFactorGate(network->getGrowthFactorGate());
                        sim->setGrowthFactorSwap(network->getGrowthFactorSwap());
                        // CPU routing already owns this unchanged, ordered window
                        // from before layout planning. Its setters refresh callbacks
                        // and fusion plans; reinstalling it repeats that preparation.
                        if (sim->GetType() != Simulators::SimulatorType::kQCSim)
                            sim->SetUpcomingGates(dcirc->GetOperations());
                    }
                }
            }
            // As in the network's copy: a register of that size whose bond limit is below the one a
            // layout is planned for still gets the qubits of lines out of index order placed along them.
            if (!planned && network->GetInitialQubitsMapOptimization() && network->GetMPSOptimizationQubitsNumberThreshold() <= routedQubits)
            {
                const auto line = Circuits::ChainLine(dcirc, nrQubits, sim->SwapGatesRelabelQubits());
                if (!line.empty())
                    sim->SetInitialQubitsMap(line);
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
    bool startsFromZero = false;  // The network explicitly reset the supplied simulator before this job.
    bool shareResetWires = false; // Fresh QCSim MPS with a layout searched before scheduling unobserved resets.
    bool allowInternalMultithreading = false;
    // Zero leaves the caller's OpenMP settings untouched (ordinary jobs).
    int internalThreads = 0;
    // Distinguishes jobs even when the caller did not configure a simulator seed.
    uint64_t randomStream = 0;
    // The network is unseeded: the seed in config derives this job's streams,
    // but the network may keep the job's simulator, which must stay unseeded.
    bool transientSeed = false;
    bool independentShotStreams = false;
    uint64_t trajectorySeed = 0;
    size_t firstShot = 0;
    std::shared_ptr<Simulators::ISimulator> optSim;
    std::shared_ptr<Simulators::ISimulator> cloneSource;
    std::shared_ptr<std::mutex> cloneMutex;

    // For jobs that take a full clone of cloneSource (Clone() and an optional
    // seed) in the order they were queued, instead of CloneForExecution(seed)
    // under cloneMutex: the turn shared by the jobs of one execution, and this
    // job's place in it.
    struct CloneOrder
    {
        std::mutex mutex;
        std::condition_variable turn;
        uint64_t next = 0;
        bool cancelled = false; // a job failed: the turns of the jobs the pool then skips never come
    };

    std::shared_ptr<CloneOrder> cloneOrder;
    uint64_t cloneIndex = 0;
    std::vector<bool> executedGates;

    // relevant only if the simulator is not passed or the simulator doesn't have
    // the proper number of qubits, otherwise the simulator is already configured
    Configuration<Time> config;

    std::shared_ptr<Network::INetwork<Time>> network;
    size_t *curMaxBondDim = nullptr;
};

} // namespace Network

#endif // ! _NETWORK_JOB_H
