/**
 * @file SimulatorsEstimatorInterface.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Interface class for the simulators estimators, exposes a common interface.
 */

#pragma once

#ifndef __SIMULATORS_ESTIMATOR_INTERFACE_H_
#define __SIMULATORS_ESTIMATOR_INTERFACE_H_

#include "../Circuit/Circuit.h"
#include "../Simulators/Interfaces/Simulator.h"

namespace Estimators
{

/**
 * @class SimulatorsEstimatorInterface
 * @brief An interface for runtime estimators.
 *
 * A class derived from this is able to estimate the time to simulate a circuit,
 * using various means (for example using the O() complexity of the algorithms).
 * It uses the estimation to choose the best simulator for a given circuit and
 * number of shots.
 *
 * @tparam Time The time type used for operation timing.
 */
template <typename Time = Types::time_type> class SimulatorsEstimatorInterface
{
  public:
    virtual ~SimulatorsEstimatorInterface() = default;

    virtual bool IsInitialized() const = 0;

    virtual std::shared_ptr<Simulators::ISimulator> ChooseBestSimulator(
        const std::vector<std::pair<Simulators::SimulatorType, Simulators::SimulationType>> &simulatorTypes, std::shared_ptr<Circuits::Circuit<Time>> &dcirc,
        size_t &counts, size_t nrQubits, size_t nrCbits, size_t nrResultCbits, Simulators::SimulatorType &simType, Simulators::SimulationType &method,
        std::vector<bool> &executed, const long long int maxBondDim, const double singularValueThreshold, const std::string &truncationMode,
        const std::string &mpsSample, size_t maxSimulators, const std::vector<std::string> *paulis, bool multithreading = false) const = 0;

    static void ExecuteUpToMeasurements(const std::shared_ptr<Circuits::Circuit<Time>> &dcirc, size_t nrQubits, size_t nrCbits, size_t nrResultCbits,
                                        const std::shared_ptr<Simulators::ISimulator> &sim, std::vector<bool> &executed, size_t *curMaxBondDim = nullptr)
    {
        Circuits::OperationState state;
        state.AllocateBits(nrCbits);

        const bool hasMeasurementsOnlyAtEnd = !dcirc->HasOpsAfterMeasurements();
        const bool specialOptimizationForStatevector = sim->GetSimulationType() == Simulators::SimulationType::kStatevector && hasMeasurementsOnlyAtEnd;
        const bool specialOptimizationForMPS = sim->GetSimulationType() == Simulators::SimulationType::kMatrixProductState && hasMeasurementsOnlyAtEnd;

        executed = dcirc->ExecuteNonMeasurements(sim, state, curMaxBondDim);

        if (!specialOptimizationForStatevector && !specialOptimizationForMPS)
        {
            // Workers continue from this suffix, including when they receive a
            // clone of the estimator's simulator and its saved routing context.
            // RemoveExecutedOperations rewrites its mask. Preserve the caller's
            // mask, whose indices still describe dcirc, for the execution workers.
            auto suffixMask = executed;
            if (suffixMask.empty())
                sim->SetUpcomingGates({}); // The whole prefix was executed.
            else
                sim->SetUpcomingGates(dcirc->RemoveExecutedOperations(suffixMask)->GetOperations());
            sim->SaveState();
        }
    }
};

} // namespace Estimators

#endif // !__SIMULATORS_ESTIMATOR_INTERFACE_H_
