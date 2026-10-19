/**
 * @file Factory.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simulator factory implementation
 *
 * Call CreateSimulator with the desired simulator type to create a simulator
 * returned as a shared pointer. Currently only two simulators are supported:
 * qiskit aer and qcsim. Can be esily extended to support more simulators. Just
 * implement the interface for another simulator, add its type to the enum and
 * add another case to the switch statement.
 *
 * This file only dispatches on the simulator type. The backends are created in
 * the other Factory*.cpp files (see FactoryBackends.h), which must be compiled
 * together with this one.
 */

#include "FactoryBackends.h"

namespace Simulators
{

std::unique_ptr<ISimulator> SimulatorsFactory::CreateImmediateSimulatorUnique(SimulatorType type, SimulationType method)
{
    switch (type)
    {
    case SimulatorType::kQCSim:
        return FactoryBackends::CreateQCSimImmediate(method);
#ifndef NO_QISKIT_AER
    case SimulatorType::kQiskitAer:
        // Aer owns its method selection and does not use the fusion adapter.
        return CreateSimulatorUnique(type, method);
#endif
#ifdef __linux__
    case SimulatorType::kGpuSim:
        return FactoryBackends::CreateGpuImmediate(method);
#endif
    default:
        throw std::invalid_argument("Unsupported immediate backend");
    }
}

std::shared_ptr<ISimulator> SimulatorsFactory::CreateSimulator(SimulatorType t, SimulationType m)
{
    return CreateSimulatorUnique(t, m);
}

std::unique_ptr<ISimulator> SimulatorsFactory::CreateSimulatorUnique(SimulatorType t, SimulationType m)
{
    switch (t)
    {
    case SimulatorType::kQCSim:
        return FactoryBackends::CreateQCSimSimulator(CreateImmediateSimulatorUnique(t, m));
#ifndef NO_QISKIT_AER
    case SimulatorType::kQiskitAer:
        return FactoryBackends::CreateAerSimulator(m);
    case SimulatorType::kCompositeQiskitAer:
        return FactoryBackends::CreateImmediateCompositeSimulator(SimulatorType::kQiskitAer);
#endif
    case SimulatorType::kCompositeQCSim:
        return FactoryBackends::CreateCompositeSimulator(SimulatorType::kQCSim);
#ifdef __linux__
    case SimulatorType::kDistGpuSim:
        if (m != SimulationType::kStatevector)
            throw std::invalid_argument("Distributed GPU supports only statevector");
        if (!IsDistributedGpuAvailable())
            return nullptr;
        return FactoryBackends::CreateDistributedGpuSimulator();
    case SimulatorType::kDistMpiGpuSim:
        if (m != SimulationType::kStatevector)
            throw std::invalid_argument("Distributed MPI GPU supports only statevector");
        return FactoryBackends::CreateDistributedMpiGpuSimulator();

    case SimulatorType::kGpuSim:
        // Library initialization is checked before advertising the backend;
        // device resources remain lazy and configuration follows creation.
        if (GetGpuDeviceCount() <= 0)
            return nullptr;
        if (auto immediate = CreateImmediateSimulatorUnique(t, m))
            return FactoryBackends::CreateGpuSimulator(std::move(immediate));
        return nullptr;
#endif
    case SimulatorType::kQuestSim:
        if (m != SimulationType::kStatevector)
            throw std::invalid_argument("Simulation Type not supported for Quest Simulator");
        else if (IsQuestLibraryAvailable())
        {
            return FactoryBackends::CreateQuestSimulator();
        }
        return nullptr;
    default:
        break;
    }

    throw std::invalid_argument("Simulator Type not supported");

    return nullptr; // keep compillers happy
}
} // namespace Simulators
