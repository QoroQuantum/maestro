/**
 * @file FactoryAerComposite.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simulator factory part for the qiskit aer backend and the composite
 * simulators.
 *
 * Qiskit aer's headers define non-inline functions and static data, so they
 * can be compiled in only one translation unit. The composite simulators use
 * aer, so they are created here as well. This file also compiles the aer AVX2
 * statevector kernels.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#ifndef NO_QISKIT_AER
#ifndef __APPLE__
#ifndef _QV_AVX2_IMPL
#define _QV_AVX2_IMPL
#pragma warning(push)
#pragma warning(disable : 4789)
#include "simulators/statevector/qv_avx2.cpp"
#pragma warning(pop)
#endif
#endif
#endif

#include "FactoryBackends.h"

#define INCLUDED_BY_FACTORY
#ifndef NO_QISKIT_AER
#include "../Aer/AerSimulator.h"
#endif
#include "../Composite/Composite.h"

namespace Simulators::FactoryBackends
{

#ifndef NO_QISKIT_AER
std::unique_ptr<ISimulator> CreateAerSimulator(SimulationType method)
{
    auto sim = std::make_unique<Private::AerSimulator>();
    if (method == SimulationType::kMatrixProductState)
        sim->Configure("method", "matrix_product_state");
    else if (method == SimulationType::kStabilizer)
        sim->Configure("method", "stabilizer");
    else if (method == SimulationType::kTensorNetwork)
        sim->Configure("method", "tensor_network");
    else if (method == SimulationType::kExtendedStabilizer)
        sim->Configure("method", "extended_stabilizer");
    else if (method == SimulationType::kDensityMatrix)
        sim->Configure("method", "density_matrix");
    else if (method == SimulationType::kStatevector)
        sim->Configure("method", "statevector");
    else
        throw std::invalid_argument("Simulation Type not supported for Qiskit Aer");

    return sim;
}
#endif

std::unique_ptr<ISimulator> CreateCompositeSimulator(SimulatorType type)
{
    return std::make_unique<Private::CompositeSimulator>(type);
}

std::unique_ptr<ISimulator> CreateImmediateCompositeSimulator(SimulatorType type)
{
    return std::make_unique<Private::ImmediateCompositeSimulator>(type);
}

} // namespace Simulators::FactoryBackends
