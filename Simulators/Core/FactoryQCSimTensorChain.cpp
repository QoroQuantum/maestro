/**
 * @file FactoryQCSimTensorChain.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Simulator factory part for the qcsim matrix product state and matrix product
 * operator simulators.
 *
 * They are the most expensive qcsim simulators to compile, so they are kept
 * apart from the other qcsim backends. They share this translation unit
 * because the qcsim MPS headers define a non-inline function, which may be
 * compiled only once.
 */

#define _CRT_SECURE_NO_WARNINGS 1

#include "FactoryBackends.h"

#define INCLUDED_BY_FACTORY
#include "../QCSim/QCSimMPOSimulator.h"
#include "../QCSim/QCSimMPSSimulator.h"

namespace Simulators::FactoryBackends
{

std::unique_ptr<ISimulator> CreateQCSimMPSImmediate()
{
    return std::make_unique<Private::QCSimMPSSimulator>();
}

std::unique_ptr<ISimulator> CreateQCSimMPOImmediate()
{
    return std::make_unique<Private::QCSimMPOSimulator>();
}

} // namespace Simulators::FactoryBackends
