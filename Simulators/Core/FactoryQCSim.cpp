/**
 * @file FactoryQCSim.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simulator factory part for the qcsim backends.
 *
 * The matrix product state and matrix product operator simulators are the most
 * expensive to compile, so they are created in FactoryQCSimTensorChain.cpp.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "FactoryBackends.h"

#define INCLUDED_BY_FACTORY
#include "../QCSim/QCSimDensityMatrixSimulator.h"
#include "../QCSim/QCSimExtendedStabilizerSimulator.h"
#include "../QCSim/QCSimPathIntegralSimulator.h"
#include "../QCSim/QCSimPauliPropagatorSimulator.h"
#include "../QCSim/QCSimSimulator.h"
#include "../QCSim/QCSimStabilizerSimulator.h"
#include "../QCSim/QCSimStatevectorSimulator.h"
#include "../QCSim/QCSimTensorNetworkSimulator.h"

namespace Simulators::FactoryBackends
{

std::unique_ptr<ISimulator> CreateQCSimImmediate(SimulationType method)
{
    switch (method)
    {
    case SimulationType::kStatevector:
        return std::make_unique<Private::QCSimStatevectorSimulator>();
    case SimulationType::kMatrixProductState:
        return CreateQCSimMPSImmediate();
    case SimulationType::kMatrixProductOperator:
        return CreateQCSimMPOImmediate();
    case SimulationType::kStabilizer:
        return std::make_unique<Private::QCSimStabilizerSimulator>();
    case SimulationType::kTensorNetwork:
        return std::make_unique<Private::QCSimTensorNetworkSimulator>();
    case SimulationType::kPauliPropagator:
        return std::make_unique<Private::QCSimPauliPropagatorSimulator>();
    case SimulationType::kPathIntegral:
        return std::make_unique<Private::QCSimPathIntegralSimulator>();
    case SimulationType::kDensityMatrix:
        return std::make_unique<Private::QCSimDensityMatrixSimulator>();
    case SimulationType::kExtendedStabilizer:
        return std::make_unique<Private::QCSimExtendedStabilizerSimulator>();
    default:
        throw std::invalid_argument("Simulation Type not supported for QCSim");
    }
}

std::unique_ptr<ISimulator> CreateQCSimSimulator(std::unique_ptr<ISimulator> immediate)
{
    return std::make_unique<Private::QCSimSimulator>(std::move(immediate));
}

} // namespace Simulators::FactoryBackends
