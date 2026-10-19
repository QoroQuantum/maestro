/**
 * @file FactoryBackends.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Internal backend creators used by the simulator factory.
 *
 * The simulator backends are header-only and expensive to compile, so they are
 * instantiated in several translation units (Factory*.cpp) instead of all of
 * them in Factory.cpp. This lets the parts compile in parallel and limits
 * rebuilds to the part whose headers changed. Factory.cpp only dispatches to
 * the functions declared here.
 *
 * Some third party headers define non-inline symbols, which limits the split:
 * everything that includes qiskit aer (also the composite simulators) must
 * stay in FactoryAerComposite.cpp, and the qcsim MPS and MPO simulators must
 * stay together in FactoryQCSimTensorChain.cpp.
 *
 * All Factory*.cpp files in this folder must be compiled together.
 */

#pragma once

#ifndef _SIMULATORS_FACTORY_BACKENDS_H_
#define _SIMULATORS_FACTORY_BACKENDS_H_

#include "Factory.h"

namespace Simulators::FactoryBackends
{

// FactoryQCSim.cpp
std::unique_ptr<ISimulator> CreateQCSimImmediate(SimulationType method);
std::unique_ptr<ISimulator> CreateQCSimSimulator(std::unique_ptr<ISimulator> immediate);

// FactoryQCSimTensorChain.cpp
std::unique_ptr<ISimulator> CreateQCSimMPSImmediate();
std::unique_ptr<ISimulator> CreateQCSimMPOImmediate();

// FactoryAerComposite.cpp
#ifndef NO_QISKIT_AER
std::unique_ptr<ISimulator> CreateAerSimulator(SimulationType method);
#endif
std::unique_ptr<ISimulator> CreateCompositeSimulator(SimulatorType type);
std::unique_ptr<ISimulator> CreateImmediateCompositeSimulator(SimulatorType type);

#ifdef __linux__
// FactoryGpu.cpp
std::unique_ptr<ISimulator> CreateGpuImmediate(SimulationType method);
std::unique_ptr<ISimulator> CreateGpuSimulator(std::unique_ptr<ISimulator> immediate);
std::unique_ptr<ISimulator> CreateDistributedGpuSimulator();
std::unique_ptr<ISimulator> CreateDistributedMpiGpuSimulator();
#endif

// FactoryQuest.cpp
std::unique_ptr<ISimulator> CreateQuestSimulator();

} // namespace Simulators::FactoryBackends

#endif // !_SIMULATORS_FACTORY_BACKENDS_H_
