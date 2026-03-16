/**
 * @file mpsswapsopttests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the MPS swap optimization.
 */

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;

#include <framework/avx2_detect.hpp>

#undef min
#undef max

#include <numeric>
#include <algorithm>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Simulators/MPSDummySimulator.h"


BOOST_AUTO_TEST_SUITE(MPSSwapOptTests)

BOOST_AUTO_TEST_CASE(OptimalQubitsMapZeroSwapCost)
{
    constexpr int nrQubits = 8;
    constexpr int nrGates = 30;

    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::random_device rd;
    std::mt19937 g(rd());

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));
    auto gateGenIter = gateGen.begin();

    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        const double param1 = *dblGenIter; ++dblGenIter;
        const double param2 = *dblGenIter; ++dblGenIter;
        const double param3 = *dblGenIter; ++dblGenIter;
        const double param4 = *dblGenIter; ++dblGenIter;

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, param1, param2, param3, param4);
        randomCirc->AddOperation(theGate);
    }

    const auto layers = randomCirc->ToMultipleQubitsLayers();
    const auto optimalMap = Simulators::MPSDummySimulator::ComputeOptimalQubitsMap(nrQubits, layers);

    Simulators::MPSDummySimulator dummySim(nrQubits);
    dummySim.SetInitialQubitsMap(optimalMap);

    const size_t layersToExecute = layers.size() < 2 ? layers.size() : 2;
    for (size_t l = 0; l < layersToExecute; ++l)
      dummySim.ApplyGates(layers[l]->GetOperations());
    
    BOOST_CHECK_EQUAL(dummySim.getTotalSwappingCost(), 0);
}

BOOST_AUTO_TEST_SUITE_END()

