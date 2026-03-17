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
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Simulators/MPSDummySimulator.h"
#include "../Simulators/Factory.h"

extern bool checkClose(std::complex<double> a, std::complex<double> b,
                       double dif);

BOOST_AUTO_TEST_SUITE(MPSSwapOptTests)


constexpr std::array numGates{30, 50, 70, 100, 150, 270, 340, 500};

BOOST_DATA_TEST_CASE(OptimalQubitsMapZeroSwapCost, numGates, nrGates) {
    constexpr int nrQubits = 10;

    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::random_device rd;
    std::mt19937 g(rd());

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto gateGen = bdata::random(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));
    auto gateGenIter = gateGen.begin();

    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(*gateGenIter);

        // parameters are not important for this test, we just need to have some
        // gates that affect the qubits
        // neither is the type of the gate, beyond the number of qubits it affects
        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, 0.0, 0.0, 0.0, 0.0);
        randomCirc->AddOperation(theGate);
    }

    const auto layers = randomCirc->ToMultipleQubitsLayers();

    Simulators::MPSDummySimulator dummySim(nrQubits);
    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);


    dummySim.SetInitialQubitsMap(optimalMap);

    const auto optCirc = randomCirc->LayersToCircuit(layers);

    std::vector<long long int> origqubits(nrQubits);
    std::iota(origqubits.begin(), origqubits.end(), 0);
    dummySim.SetInitialQubitsMap(origqubits);

    dummySim.ApplyGates(optCirc->GetOperations());
    const auto origSwappingCost = dummySim.getTotalSwappingCost();

    dummySim.SetInitialQubitsMap(optimalMap);
    dummySim.ApplyGates(optCirc->GetOperations());
    const auto optSwappingCost = dummySim.getTotalSwappingCost();

    BOOST_CHECK_LE(optSwappingCost, origSwappingCost);

    // print the results for debugging
    BOOST_TEST_MESSAGE("Original swapping cost: " << origSwappingCost);
    BOOST_TEST_MESSAGE("Optimized swapping cost: " << optSwappingCost);
}

BOOST_DATA_TEST_CASE(OptimalQubitsMapSimulationMatch, numGates, nrGates) {
    constexpr int nrQubits = 14;
    const size_t nrStates = 1ULL << nrQubits;

    // build a random circuit with 1- and 2-qubit gates
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

    // compute the optimal qubits map using the dummy simulator
    const auto layers = randomCirc->ToMultipleQubitsLayers();

    Simulators::MPSDummySimulator dummySim(nrQubits);
    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);

    // execute the original circuit with the identity (original) qubits map
    auto qcsimOrig = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kMatrixProductState);
    qcsimOrig->AllocateQubits(nrQubits);
    qcsimOrig->Configure("matrix_product_state_max_bond_dimension", "64");
    qcsimOrig->Initialize();

    Circuits::OperationState stateOrig;
    stateOrig.AllocateBits(nrQubits);

    auto startOrig = std::chrono::system_clock::now();
    randomCirc->Execute(qcsimOrig, stateOrig);
    auto endOrig = std::chrono::system_clock::now();
    double origTime = std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;

    // execute the same circuit with the optimized qubits map
    auto qcsimOpt = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kMatrixProductState);
    qcsimOpt->AllocateQubits(nrQubits);
    qcsimOpt->Configure("matrix_product_state_max_bond_dimension", "64");
    qcsimOpt->Initialize();
    qcsimOpt->SetInitialQubitsMap(
        std::vector<Eigen::Index>(optimalMap.begin(), optimalMap.end()));

    Circuits::OperationState stateOpt;
    stateOpt.AllocateBits(nrQubits);

    auto startOpt = std::chrono::system_clock::now();
    randomCirc->Execute(qcsimOpt, stateOpt);
    auto endOpt = std::chrono::system_clock::now();
    double optTime = std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

    BOOST_TEST_MESSAGE("Original mapping execution time: " << origTime << " ms");
    BOOST_TEST_MESSAGE("Optimized mapping execution time: " << optTime << " ms");
    BOOST_TEST_MESSAGE("Optimized is " << origTime / optTime << "x vs original");

    // compute swapping costs with the dummy simulator
    const auto optCirc = randomCirc->LayersToCircuit(layers);

    std::vector<long long int> origqubits(nrQubits);
    std::iota(origqubits.begin(), origqubits.end(), 0);
    dummySim.SetInitialQubitsMap(origqubits);
    dummySim.ApplyGates(optCirc->GetOperations());
    const auto origSwappingCost = dummySim.getTotalSwappingCost();

    dummySim.SetInitialQubitsMap(optimalMap);
    dummySim.ApplyGates(optCirc->GetOperations());
    const auto optSwappingCost = dummySim.getTotalSwappingCost();

    BOOST_TEST_MESSAGE("Original swapping cost: " << origSwappingCost);
    BOOST_TEST_MESSAGE("Optimized swapping cost: " << optSwappingCost);

    // compare the amplitudes of the two simulations
    for (size_t s = 0; s < nrStates; ++s) {
        const auto aOrig = qcsimOrig->Amplitude(s);
        const auto aOpt = qcsimOpt->Amplitude(s);
        BOOST_CHECK_PREDICATE(checkClose, (aOrig)(aOpt)(0.05)); // since max bond dimension is set, they won't match so well
    }
}

BOOST_AUTO_TEST_SUITE_END()

