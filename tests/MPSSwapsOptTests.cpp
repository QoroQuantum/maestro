/**
 * @file MPSSwapsOptTests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the MPS swap optimization.
 */

#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;

#ifndef NO_QISKIT_AER
#include <framework/avx2_detect.hpp>
#endif

#undef min
#undef max

#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Network/SimpleDisconnectedNetwork.h"
#include "../Simulators/Core/Factory.h"
#include "../Simulators/Fusion/FusionGate.h"
#include "../Simulators/TensorNetworks/MPSDummySimulator.h"

extern bool checkClose(std::complex<double> a, std::complex<double> b, double dif);

BOOST_AUTO_TEST_SUITE(MPSSwapOptTests)

constexpr std::array numGates{10, 20, 30, 50, 80, 100, 150, 200, 270, 340, 500, 800, 1000};

namespace
{

std::shared_ptr<Circuits::Circuit<>> CreateRandomMPSCircuit(int nrQubits, int nrGates, unsigned int seed)
{
    auto circuit = std::make_shared<Circuits::Circuit<>>();

    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> parameterDistribution(-2. * M_PI, 2. * M_PI);
    std::uniform_int_distribution<int> gateDistribution(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), generator);

        const auto gateType = static_cast<Circuits::QuantumGateType>(gateDistribution(generator));
        circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(gateType, qubits[0], qubits[1], 0, parameterDistribution(generator),
                                                                     parameterDistribution(generator), parameterDistribution(generator),
                                                                     parameterDistribution(generator)));
    }

    return circuit;
}

std::shared_ptr<Simulators::ISimulator> CreateGpuMPSSimulator(int nrQubits, int maxBondDimension)
{
#ifdef __linux__
    Simulators::SimulatorsFactory::InitGpuLibrary();
    auto simulator = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kGpuSim, Simulators::SimulationType::kMatrixProductState);
    if (simulator)
    {
        simulator->AllocateQubits(nrQubits);
        simulator->Configure("matrix_product_state_max_bond_dimension", std::to_string(maxBondDimension).c_str());
        simulator->Initialize();
    }
    return simulator;
#else
    return nullptr;
#endif
}

void CheckSimulatorAmplitudes(const std::shared_ptr<Simulators::ISimulator> &expected, const std::shared_ptr<Simulators::ISimulator> &actual, size_t nrStates,
                              double tolerance)
{
    for (size_t state = 0; state < nrStates; ++state)
    {
        const auto expectedAmplitude = expected->Amplitude(state);
        const auto actualAmplitude = actual->Amplitude(state);
        BOOST_CHECK_PREDICATE(checkClose, (expectedAmplitude)(actualAmplitude)(tolerance));
    }
}

} // namespace

BOOST_DATA_TEST_CASE(ConvertForCuttingLayersRoundtripMatchesStatevector, numGates, nrGates)
{
    constexpr int nrQubits = 13;
    const size_t nrStates = 1ULL << nrQubits;

    // build a random circuit with 1- and 2-qubit gates
    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::mt19937 g(std::random_device{}());
    std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        const double param1 = dblDist(g);
        const double param2 = dblDist(g);
        const double param3 = dblDist(g);
        const double param4 = dblDist(g);

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, param1, param2, param3, param4);
        randomCirc->AddOperation(theGate);
    }

    // --- Reference: execute the original circuit on a statevector simulator ---
    auto svSim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kStatevector);
    svSim->AllocateQubits(nrQubits);
    svSim->Initialize();

    Circuits::OperationState stateSV;
    stateSV.AllocateBits(nrQubits);
    randomCirc->Execute(svSim, stateSV);

    // --- Convert the circuit the same way OptimizeMPSInitialQubitsMap does ---
    auto convertedCirc = std::make_shared<Circuits::Circuit<>>(*randomCirc);
    convertedCirc->ConvertForCutting();
    auto layers = convertedCirc->ToMultipleQubitsLayersNoClone();
    auto finalCirc = Circuits::Circuit<>::LayersToCircuit(layers);

    // BOOST_TEST_MESSAGE("Original gate count: " << randomCirc->size());
    // BOOST_TEST_MESSAGE("Converted gate count: " << finalCirc->size());
    // BOOST_TEST_MESSAGE("Number of layers: " << layers.size());

    // --- Execute the converted circuit on an MPS simulator (no bond dim cap) ---
    auto mpsSim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    mpsSim->AllocateQubits(nrQubits);
    mpsSim->Initialize();

    Circuits::OperationState stateMPS;
    stateMPS.AllocateBits(nrQubits);
    finalCirc->Execute(mpsSim, stateMPS);

    // --- Compare amplitudes ---
    for (size_t s = 0; s < nrStates; ++s)
    {
        const auto aSV = svSim->Amplitude(s);
        const auto aMPS = mpsSim->Amplitude(s);
        BOOST_CHECK_PREDICATE(checkClose,
                              (aSV)(aMPS)(1e-5)); // no bond dim limit, expect close match
    }
}

BOOST_DATA_TEST_CASE(OptimalQubitsMapZeroSwapCost, numGates, nrGates)
{
    constexpr int nrQubits = 13;

    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::mt19937 g(
        /*static_cast<unsigned>(nrGates) * 12345u*/ std::random_device{}());
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        // parameters are not important for this test, we just need to have some
        // gates that affect the qubits
        // neither is the type of the gate, beyond the number of qubits it affects
        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, 0.0, 0.0, 0.0, 0.0);
        randomCirc->AddOperation(theGate);
    }

    const auto layers = randomCirc->ToMultipleQubitsLayers();

    Simulators::MPSDummySimulator dummySim(nrQubits);
    dummySim.SetMaxBondDimension(64);
    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
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

    const double improvementPct = origSwappingCost > 0 ? (1.0 - optSwappingCost / origSwappingCost) * 100.0 : 0.0;
    BOOST_TEST_MESSAGE("Swap cost reduction: " << improvementPct << "%");
    BOOST_TEST_MESSAGE("Speedup ratio: " << (optSwappingCost > 0 ? origSwappingCost / optSwappingCost : 0.0) << "x");
}

BOOST_DATA_TEST_CASE(OptimalQubitsMapSimulationMatch, numGates, nrGates)
{
    constexpr int nrQubits = 13;
    const size_t nrStates = 1ULL << nrQubits;

    // build a random circuit with 1- and 2-qubit gates
    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::mt19937 g(/*static_cast<unsigned>(nrGates) * 12345u*/ std::random_device{}());
    std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        const double param1 = dblDist(g);
        const double param2 = dblDist(g);
        const double param3 = dblDist(g);
        const double param4 = dblDist(g);

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, param1, param2, param3, param4);
        randomCirc->AddOperation(theGate);
    }

    auto startOpt = std::chrono::system_clock::now();

    // compute the optimal qubits map using the dummy simulator
    const auto layers = randomCirc->ToMultipleQubitsLayers();

    // display some stats
    BOOST_TEST_MESSAGE("Number of 2-qubit gate layers: " << layers.size());
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
    BOOST_TEST_MESSAGE("Average number of 2-qubit gates per layer: " << avgTwoQubitGatesPerLayer);

    Simulators::MPSDummySimulator dummySim(nrQubits);
    dummySim.SetMaxBondDimension(64);

    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
    auto endOpt = std::chrono::system_clock::now();
    double optMapTime = std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;
    BOOST_TEST_MESSAGE("Optimization time: " << optMapTime << " ms");

    // execute the original circuit with the identity (original) qubits map
    auto qcsimOrig = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
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
    auto qcsimOpt = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    qcsimOpt->AllocateQubits(nrQubits);
    qcsimOpt->Configure("matrix_product_state_max_bond_dimension", "64");
    qcsimOpt->Initialize();

    // qcsimOpt->SetUseOptimalMeetingPosition(true);

    qcsimOpt->SetInitialQubitsMap(std::vector<long long int>(optimalMap.begin(), optimalMap.end()));

    Circuits::OperationState stateOpt;
    stateOpt.AllocateBits(nrQubits);

    const auto optCirc = randomCirc->LayersToCircuit(layers);

    auto startExecOpt = std::chrono::system_clock::now();
    optCirc->Execute(qcsimOpt, stateOpt);
    auto endExecOpt = std::chrono::system_clock::now();
    double optTime = std::chrono::duration<double>(endExecOpt - startExecOpt).count() * 1000.;

    BOOST_TEST_MESSAGE("Original mapping execution time: " << origTime << " ms");
    BOOST_TEST_MESSAGE("Optimized mapping execution time: " << optTime << " ms");
    BOOST_TEST_MESSAGE("Optimized is " << origTime / optTime << "x vs original");

    // compute swapping costs with the dummy simulator

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

    const double improvementPct = origSwappingCost > 0 ? (1.0 - optSwappingCost / origSwappingCost) * 100.0 : 0.0;
    BOOST_TEST_MESSAGE("Swap cost reduction: " << improvementPct << "%");
    BOOST_TEST_MESSAGE("Speedup ratio: " << (optSwappingCost > 0 ? origSwappingCost / optSwappingCost : 0.0) << "x");

    // compare the amplitudes of the two simulations
    for (size_t s = 0; s < nrStates; ++s)
    {
        const auto aOrig = qcsimOrig->Amplitude(s);
        const auto aOpt = qcsimOpt->Amplitude(s);
        BOOST_CHECK_PREDICATE(checkClose, (aOrig)(aOpt)(0.05)); // since max bond dimension is set,
                                                                // they won't match so well
    }
}

BOOST_DATA_TEST_CASE(LookaheadSwapOptimization, numGates, nrGates)
{
    constexpr int nrQubits = 13;

    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::mt19937 g(/*static_cast<unsigned>(nrGates) * 99999u*/ std::random_device{}());
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, 0.0, 0.0, 0.0, 0.0);
        randomCirc->AddOperation(theGate);
    }

    const auto layers = randomCirc->ToMultipleQubitsLayers();

    Simulators::MPSDummySimulator dummySim(nrQubits);
    dummySim.SetMaxBondDimension(64);
    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
    const auto optCirc = randomCirc->LayersToCircuit(layers);

    // Baseline: heuristic swap cost
    dummySim.SetInitialQubitsMap(optimalMap);
    dummySim.ApplyGates(optCirc->GetOperations());
    const auto heuristicCost = dummySim.getTotalSwappingCost();

    const auto &ops = optCirc->GetOperations();
    const auto opsVec = std::vector<std::shared_ptr<Circuits::IOperation<>>>(ops.begin(), ops.end());

    dummySim.SetInitialQubitsMap(optimalMap);

    int lookaheadDepth = 30;
    int lookaheadDepthWithHeuristic = 27;

    for (size_t i = 0; i < opsVec.size(); ++i)
    {
        const auto qbits = opsVec[i]->AffectedQubits();
        if (qbits.size() < 2)
        {
            continue;
        }

        auto q1 = static_cast<long long int>(qbits[0]);
        auto q2 = static_cast<long long int>(qbits[1]);

        // Build lookahead window from remaining ops, counting only 2-qubit
        // gates towards the depth budget so the effective lookahead is accurate
        std::vector<std::shared_ptr<Circuits::IOperation<>>> window;
        window.reserve(lookaheadDepth);
        int twoQubitCount = 0;
        for (size_t j = i; j < opsVec.size() && twoQubitCount < lookaheadDepth; ++j)
        {
            window.push_back(opsVec[j]);
            if (opsVec[j]->AffectedQubits().size() >= 2)
                ++twoQubitCount;
        }

        double meetBestCost = std::numeric_limits<double>::infinity();
        const auto meetPos = dummySim.FindBestMeetingPosition(window, 0, lookaheadDepth, lookaheadDepthWithHeuristic, 0.0, meetBestCost);

        dummySim.SwapQubitsToPosition(q1, q2, meetPos);
        dummySim.ApplyGate(opsVec[i]);
    }

    const auto lookaheadCost = dummySim.getTotalSwappingCost();

    BOOST_TEST_MESSAGE("Heuristic swap cost: " << heuristicCost);
    BOOST_TEST_MESSAGE("Lookahead (depth " << lookaheadDepth << ") swap cost: " << lookaheadCost);

    const double improvementPct = heuristicCost > 0 ? (1.0 - lookaheadCost / heuristicCost) * 100.0 : 0.0;
    BOOST_TEST_MESSAGE("Lookahead improvement: " << improvementPct << "%");

    // Lookahead should generally be no worse than heuristic, but due to the
    // limited search window and greedy per-gate decisions it can occasionally
    // produce a marginally higher cost.  Allow a small tolerance (10%).
    const double tolerance = 1.10;
    BOOST_CHECK_LE(lookaheadCost, heuristicCost * tolerance);
}

BOOST_DATA_TEST_CASE(OptimalMeetingPositionSimulationMatch, numGates, nrGates)
{
    constexpr int nrQubits = 13;
    const size_t nrStates = 1ULL << nrQubits;

    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::mt19937 g(/*static_cast<unsigned>(nrGates) * 77777u*/ std::random_device{}());
    std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        const double param1 = dblDist(g);
        const double param2 = dblDist(g);
        const double param3 = dblDist(g);
        const double param4 = dblDist(g);

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, param1, param2, param3, param4);
        randomCirc->AddOperation(theGate);
    }

    // Execute with default heuristic
    auto qcsimOrig = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    qcsimOrig->AllocateQubits(nrQubits);
    qcsimOrig->Configure("matrix_product_state_max_bond_dimension", "64");
    qcsimOrig->Initialize();
    qcsimOrig->SetUseOptimalMeetingPosition(false);

    Circuits::OperationState stateOrig;
    stateOrig.AllocateBits(nrQubits);

    auto startOrig = std::chrono::system_clock::now();
    randomCirc->Execute(qcsimOrig, stateOrig);
    auto endOrig = std::chrono::system_clock::now();
    double origTime = std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;

    // Execute with optimal meeting position enabled
    auto qcsimOpt = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    qcsimOpt->AllocateQubits(nrQubits);
    qcsimOpt->Configure("matrix_product_state_max_bond_dimension", "64");
    qcsimOpt->Initialize();
    qcsimOpt->SetUseOptimalMeetingPosition(true);

    Circuits::OperationState stateOpt;
    stateOpt.AllocateBits(nrQubits);

    auto startOpt = std::chrono::system_clock::now();
    randomCirc->Execute(qcsimOpt, stateOpt);
    auto endOpt = std::chrono::system_clock::now();
    double optTime = std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

    BOOST_TEST_MESSAGE("Default heuristic execution time: " << origTime << " ms");
    BOOST_TEST_MESSAGE("Optimal meeting position execution time: " << optTime << " ms");
    BOOST_TEST_MESSAGE("Optimized is " << origTime / optTime << "x vs default");

    // Both executions should produce the same amplitudes
    // (swap route may differ but both are valid)
    for (size_t s = 0; s < nrStates; ++s)
    {
        const auto aOrig = qcsimOrig->Amplitude(s);
        const auto aOpt = qcsimOpt->Amplitude(s);
        BOOST_CHECK_PREDICATE(checkClose, (aOrig)(aOpt)(0.05));
    }
}

BOOST_DATA_TEST_CASE(WindowOptimizedVsOriginalSimulation, numGates, nrGates)
{
#define MAX_BOND_DIMENSION 64
    constexpr int nrQubits = 13;
    const size_t nrStates = 1ULL << nrQubits;

    // build a random circuit with 1- and 2-qubit gates
    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    std::mt19937 g(
        /*static_cast<unsigned>(nrGates) * 31415u*/ std::random_device{}());
    std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];

        const double param1 = dblDist(g);
        const double param2 = dblDist(g);
        const double param3 = dblDist(g);
        const double param4 = dblDist(g);

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0, param1, param2, param3, param4);
        randomCirc->AddOperation(theGate);
    }

    // --- Original execution: no optimizations, identity qubit map ---
    auto qcsimOrig = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    qcsimOrig->AllocateQubits(nrQubits);
    qcsimOrig->Configure("matrix_product_state_max_bond_dimension", std::to_string(MAX_BOND_DIMENSION).c_str());
    qcsimOrig->Initialize();

    Circuits::OperationState stateOrig;
    stateOrig.AllocateBits(nrQubits);

    auto startOrig = std::chrono::system_clock::now();
    randomCirc->Execute(qcsimOrig, stateOrig);
    auto endOrig = std::chrono::system_clock::now();
    double origTime = std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;

    // --- Window-optimized execution ---
    // Compute optimal initial qubit mapping via the dummy simulator
    auto startOptMap = std::chrono::system_clock::now();
    const auto layers = randomCirc->ToMultipleQubitsLayers();

    // display some stats
    BOOST_TEST_MESSAGE("Number of 2-qubit gate layers: " << layers.size());

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
    BOOST_TEST_MESSAGE("Average number of 2-qubit gates per layer: " << avgTwoQubitGatesPerLayer);

    Simulators::MPSDummySimulator dummySim(nrQubits);
    dummySim.SetMaxBondDimension(MAX_BOND_DIMENSION);
    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);

    const auto optCirc = randomCirc->LayersToCircuit(layers);
    auto endOptMap = std::chrono::system_clock::now();
    double optMapTime = std::chrono::duration<double>(endOptMap - startOptMap).count() * 1000.;
    BOOST_TEST_MESSAGE("Optimization (map + layers) time: " << optMapTime << " ms");

    /*
    std::cout << "Circuit, on qubits:" << std::endl;
    for (size_t i = 0; i < optCirc->size(); ++i) {
      std::cout << "Gate " << i
                << " on qubits ";
      const auto qubits = optCirc->GetOperations()[i]->AffectedQubits();
      for (size_t j = 0; j < qubits.size(); ++j) {
        std::cout << qubits[j];
        if (j < qubits.size() - 1) std::cout << ", ";
      }
      std::cout << std::endl;
    }
    */

    // Create the optimized simulator with optimal map + meeting position + lookahead
    int lookaheadVal = nrQubits;
    if (nrQubits > 15)
        lookaheadVal = 15;

    int lookaheadDepth = layers.size() < 10   ? 0
                         : layers.size() < 20 ? static_cast<int>(lookaheadVal)
                         : layers.size() < 35 ? static_cast<int>(1.5 * lookaheadVal)
                                              : 2 * lookaheadVal;

    int lookaheadHeuristicDepth = layers.size() < 10 ? 0 : layers.size() < 20 ? lookaheadDepth - 1 : lookaheadDepth - 2;

    if (nrQubits < 10 || MAX_BOND_DIMENSION < 32)
    {
        lookaheadDepth = 0;
        lookaheadHeuristicDepth = 0;
    }

    auto qcsimOpt = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    qcsimOpt->AllocateQubits(nrQubits);
    qcsimOpt->Configure("matrix_product_state_max_bond_dimension", std::to_string(MAX_BOND_DIMENSION).c_str());
    qcsimOpt->Initialize();

    qcsimOpt->SetInitialQubitsMap(std::vector<long long int>(optimalMap.begin(), optimalMap.end()));

    qcsimOpt->SetUseOptimalMeetingPosition(true);
    qcsimOpt->SetLookaheadDepth(lookaheadDepth);
    qcsimOpt->SetLookaheadDepthWithHeuristic(lookaheadHeuristicDepth);
    qcsimOpt->SetUpcomingGates(optCirc->GetOperations());

    Circuits::OperationState stateOpt;
    stateOpt.AllocateBits(nrQubits);

    auto startExecOpt = std::chrono::system_clock::now();
    optCirc->Execute(qcsimOpt, stateOpt);
    auto endExecOpt = std::chrono::system_clock::now();
    double optTime = std::chrono::duration<double>(endExecOpt - startExecOpt).count() * 1000.;

    // --- Runtime comparison ---
    BOOST_TEST_MESSAGE("Original execution time: " << origTime << " ms");
    BOOST_TEST_MESSAGE("Window-optimized execution time: " << optTime << " ms");
    BOOST_TEST_MESSAGE("Window-optimized is " << origTime / optTime << "x vs original");

    // --- Amplitude comparison ---
    for (size_t s = 0; s < nrStates; ++s)
    {
        const auto aOrig = qcsimOrig->Amplitude(s);
        const auto aOpt = qcsimOpt->Amplitude(s);
        BOOST_CHECK_PREDICATE(checkClose, (aOrig)(aOpt)(0.05)); // max bond dimension is set,
                                                                // tolerance needed
    }
}

#ifdef _SLOW_GPU_TESTS

constexpr std::array numGatesForGpuExecution{10, 50, 100};

BOOST_DATA_TEST_CASE(GpuOptimalQubitsMapSimulationMatch, numGatesForGpuExecution, nrGates)
{
    constexpr int nrQubits = 13;
    constexpr int maxBondDimension = 64;
    constexpr double tolerance = 1e-4;
    const size_t nrStates = 1ULL << nrQubits;

    auto gpuOrig = CreateGpuMPSSimulator(nrQubits, maxBondDimension);
    auto gpuOpt = CreateGpuMPSSimulator(nrQubits, maxBondDimension);
    if (!gpuOrig || !gpuOpt)
    {
        BOOST_TEST_MESSAGE("GPU MPS simulator is not available; skipping swap optimization "
                           "test.");
        return;
    }

    const auto randomCirc = CreateRandomMPSCircuit(nrQubits, nrGates, static_cast<unsigned int>(nrGates) * 12345u);
    const auto layers = randomCirc->ToMultipleQubitsLayers();

    Simulators::MPSDummySimulator dummySim(nrQubits);
    dummySim.SetMaxBondDimension(maxBondDimension);
    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
    const auto optCirc = randomCirc->LayersToCircuit(layers);

    Circuits::OperationState stateOrig;
    Circuits::OperationState stateOpt;
    stateOrig.AllocateBits(nrQubits);
    stateOpt.AllocateBits(nrQubits);

    const auto startOrig = std::chrono::system_clock::now();
    randomCirc->Execute(gpuOrig, stateOrig);
    const auto endOrig = std::chrono::system_clock::now();

    gpuOpt->SetInitialQubitsMap(std::vector<long long int>(optimalMap.begin(), optimalMap.end()));

    const auto startOpt = std::chrono::system_clock::now();
    optCirc->Execute(gpuOpt, stateOpt);
    const auto endOpt = std::chrono::system_clock::now();

    const double origTime = std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;
    const double optTime = std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

    BOOST_TEST_MESSAGE("GPU original mapping execution time: " << origTime << " ms");
    BOOST_TEST_MESSAGE("GPU optimized mapping execution time: " << optTime << " ms");
    if (optTime > 0.)
    {
        BOOST_TEST_MESSAGE("GPU optimized mapping is " << origTime / optTime << "x vs original");
    }

    CheckSimulatorAmplitudes(gpuOrig, gpuOpt, nrStates, tolerance);
}

BOOST_DATA_TEST_CASE(GpuOptimalMeetingPositionSimulationMatch, numGatesForGpuExecution, nrGates)
{
    constexpr int nrQubits = 13;
    constexpr int maxBondDimension = 64;
    constexpr double tolerance = 1e-4;
    const size_t nrStates = 1ULL << nrQubits;

    auto gpuOrig = CreateGpuMPSSimulator(nrQubits, maxBondDimension);
    auto gpuOpt = CreateGpuMPSSimulator(nrQubits, maxBondDimension);
    if (!gpuOrig || !gpuOpt)
    {
        BOOST_TEST_MESSAGE("GPU MPS simulator is not available; skipping swap optimization "
                           "test.");
        return;
    }

    const auto randomCirc = CreateRandomMPSCircuit(nrQubits, nrGates, static_cast<unsigned int>(nrGates) * 77777u);

    gpuOrig->SetUseOptimalMeetingPosition(false);
    gpuOpt->SetUseOptimalMeetingPosition(true);

    Circuits::OperationState stateOrig;
    Circuits::OperationState stateOpt;
    stateOrig.AllocateBits(nrQubits);
    stateOpt.AllocateBits(nrQubits);

    const auto startOrig = std::chrono::system_clock::now();
    randomCirc->Execute(gpuOrig, stateOrig);
    const auto endOrig = std::chrono::system_clock::now();

    const auto startOpt = std::chrono::system_clock::now();
    randomCirc->Execute(gpuOpt, stateOpt);
    const auto endOpt = std::chrono::system_clock::now();

    const double origTime = std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;
    const double optTime = std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

    BOOST_TEST_MESSAGE("GPU default heuristic execution time: " << origTime << " ms");
    BOOST_TEST_MESSAGE("GPU optimal meeting position execution time: " << optTime << " ms");
    if (optTime > 0.)
    {
        BOOST_TEST_MESSAGE("GPU optimal meeting position is " << origTime / optTime << "x vs default");
    }

    CheckSimulatorAmplitudes(gpuOrig, gpuOpt, nrStates, tolerance);
}

BOOST_DATA_TEST_CASE(GpuWindowOptimizedVsOriginalSimulation, numGatesForGpuExecution, nrGates)
{
    constexpr int nrQubits = 13;
    constexpr int maxBondDimension = 64;
    constexpr double tolerance = 1e-4;
    const size_t nrStates = 1ULL << nrQubits;

    auto gpuOrig = CreateGpuMPSSimulator(nrQubits, maxBondDimension);
    auto gpuOpt = CreateGpuMPSSimulator(nrQubits, maxBondDimension);
    if (!gpuOrig || !gpuOpt)
    {
        BOOST_TEST_MESSAGE("GPU MPS simulator is not available; skipping swap optimization "
                           "test.");
        return;
    }

    const auto randomCirc = CreateRandomMPSCircuit(nrQubits, nrGates, static_cast<unsigned int>(nrGates) * 31415u);
    const auto layers = randomCirc->ToMultipleQubitsLayers();

    Simulators::MPSDummySimulator dummySim(nrQubits);
    dummySim.SetMaxBondDimension(maxBondDimension);
    const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
    const auto optCirc = randomCirc->LayersToCircuit(layers);

    const int lookaheadDepth = std::min(30, static_cast<int>(layers.size()));
    const int lookaheadHeuristicDepth = std::max(0, lookaheadDepth - 2);

    gpuOrig->SetUseOptimalMeetingPosition(false);
    gpuOpt->SetInitialQubitsMap(std::vector<long long int>(optimalMap.begin(), optimalMap.end()));
    gpuOpt->SetUseOptimalMeetingPosition(true);
    gpuOpt->SetLookaheadDepth(lookaheadDepth);
    gpuOpt->SetLookaheadDepthWithHeuristic(lookaheadHeuristicDepth);
    gpuOpt->SetUpcomingGates(optCirc->GetOperations());

    Circuits::OperationState stateOrig;
    Circuits::OperationState stateOpt;
    stateOrig.AllocateBits(nrQubits);
    stateOpt.AllocateBits(nrQubits);

    const auto startOrig = std::chrono::system_clock::now();
    randomCirc->Execute(gpuOrig, stateOrig);
    const auto endOrig = std::chrono::system_clock::now();

    const auto startOpt = std::chrono::system_clock::now();
    optCirc->Execute(gpuOpt, stateOpt);
    const auto endOpt = std::chrono::system_clock::now();

    const double origTime = std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;
    const double optTime = std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

    BOOST_TEST_MESSAGE("GPU original execution time: " << origTime << " ms");
    BOOST_TEST_MESSAGE("GPU window-optimized execution time: " << optTime << " ms");
    if (optTime > 0.)
    {
        BOOST_TEST_MESSAGE("GPU window-optimized is " << origTime / optTime << "x vs original");
    }

    CheckSimulatorAmplitudes(gpuOrig, gpuOpt, nrStates, tolerance);
}

#endif

constexpr std::array numGatesForNetworkExecution{10, 20, 30, 50, 80, 100};

BOOST_DATA_TEST_CASE(NetworkOptimizedSwapsVsUnoptimizedOnHost, numGatesForNetworkExecution, nrGates)
{
    constexpr int nrQubits = 13;
    constexpr size_t shots = 4000;

    // build a random circuit with 1- and 2-qubit gates and mid-circuit
    // measurements followed by conditional gates
    auto randomCirc = std::make_shared<Circuits::Circuit<>>();

    // Fixed seed for reproducibility. Stay at two-qubit gates: Toffolis force
    // decompositions that blow past bond dim 64 and make opt vs unopt sampling
    // diverge even when both are "correct" truncated MPS runs.
    std::mt19937 g(static_cast<unsigned>(nrGates) * 424242u);
    std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
    std::uniform_int_distribution<int> gateDist(0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));
    std::uniform_int_distribution<int> qubitDist(0, nrQubits - 1);
    std::bernoulli_distribution measureCoin(0.05);

    for (int gateNr = 0; gateNr < nrGates; ++gateNr)
    {
        // occasionally insert a mid-circuit measurement + conditional X
        if (gateNr > 0 && measureCoin(g))
        {
            const int mq = qubitDist(g);
            const size_t cbit = static_cast<size_t>(mq);
            randomCirc->AddOperation(
                Circuits::CircuitFactory<>::CreateMeasurement(std::vector<std::pair<Types::qubit_t, size_t>>{{static_cast<Types::qubit_t>(mq), cbit}}));

            // conditional X on a different qubit driven by the measurement result
            int tgt = (mq + 1) % nrQubits;
            randomCirc->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(
                Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kXGateType, tgt),
                Circuits::CircuitFactory<>::CreateEqualCondition(std::vector<size_t>{cbit}, std::vector<bool>{true})));

            continue;
        }

        // regular gate
        Types::qubits_vector qubits(nrQubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        std::shuffle(qubits.begin(), qubits.end(), g);
        auto q1 = qubits[0];
        auto q2 = qubits[1];
        auto q3 = qubits[2];

        const double param1 = dblDist(g);
        const double param2 = dblDist(g);
        const double param3 = dblDist(g);
        const double param4 = dblDist(g);

        Circuits::QuantumGateType gateType = static_cast<Circuits::QuantumGateType>(gateDist(g));

        auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, q3, param1, param2, param3, param4);
        randomCirc->AddOperation(theGate);
    }

    // final measurements on all qubits
    std::vector<std::pair<Types::qubit_t, size_t>> finalMeas;
    for (int q = 0; q < nrQubits; ++q)
        finalMeas.emplace_back(static_cast<Types::qubit_t>(q), static_cast<size_t>(q));
    randomCirc->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(finalMeas));

    // Unoptimized network (swaps optimization disabled)
    auto netUnopt = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
                                                                           std::vector<size_t>{static_cast<size_t>(nrQubits)});
    netUnopt->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    netUnopt->CreateSimulator();
    netUnopt->Configure("matrix_product_state_max_bond_dimension", "128");
    netUnopt->SetInitialQubitsMapOptimization(false);
    netUnopt->SetMPSOptimizeSwaps(false);

    auto startUnopt = std::chrono::system_clock::now();
    const auto resUnopt = netUnopt->RepeatedExecuteOnHost(randomCirc, 0, shots);
    auto endUnopt = std::chrono::system_clock::now();
    double unoptTime = std::chrono::duration<double>(endUnopt - startUnopt).count() * 1000.;

    BOOST_TEST_MESSAGE("Unoptimized execution time: " << unoptTime << " ms");

    // Optimized network (swaps optimization enabled, the default)
    auto netOpt = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
                                                                         std::vector<size_t>{static_cast<size_t>(nrQubits)});
    netOpt->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
    netOpt->CreateSimulator();
    netOpt->Configure("matrix_product_state_max_bond_dimension", "128");
    netOpt->SetInitialQubitsMapOptimization(true);
    netOpt->SetMPSOptimizeSwaps(true);

    auto startOpt = std::chrono::system_clock::now();
    const auto resOpt = netOpt->RepeatedExecuteOnHost(randomCirc, 0, shots);
    auto endOpt = std::chrono::system_clock::now();
    double optTime = std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

    BOOST_TEST_MESSAGE("Optimized execution time: " << optTime << " ms");
    if (optTime > 0.)
        BOOST_TEST_MESSAGE("Speedup: " << unoptTime / optTime << "x");

    // Both networks must produce non-empty results
    BOOST_CHECK(!resUnopt.empty());
    BOOST_CHECK(!resOpt.empty());

    // Compare measurement distributions.
    // Truncated MPS plus swap reordering can disagree on rare outcomes; check
    // overall TV and absolute gaps on well-sampled bins only (relative checks
    // on ~1% bins are dominated by shot noise).
    std::set<std::vector<bool>> allKeys;
    for (const auto &kv : resUnopt)
        allKeys.insert(kv.first);
    for (const auto &kv : resOpt)
        allKeys.insert(kv.first);

    double totalVariation = 0.0;
    for (const auto &key : allKeys)
    {
        const double pUnopt = (resUnopt.count(key) ? resUnopt.at(key) : 0) / static_cast<double>(shots);
        const double pOpt = (resOpt.count(key) ? resOpt.at(key) : 0) / static_cast<double>(shots);
        const double gap = std::abs(pUnopt - pOpt);
        totalVariation += gap;

        if (std::min(pUnopt, pOpt) >= 0.05)
            BOOST_CHECK_LE(gap, 0.08);
    }
    totalVariation *= 0.5;

    BOOST_TEST_MESSAGE("Total variation distance: " << totalVariation);

    // With a bond dimension cap, some divergence is expected
    BOOST_CHECK_LT(totalVariation, 0.5);
}

// The operator-chain (MPO) model has physical dimension 4 per site and gates
// act with the squared Schmidt rank, so bonds are allowed to grow faster.
BOOST_AUTO_TEST_CASE(DummyOperatorChainModel)
{
    constexpr size_t nrQubits = 6;
    Simulators::MPSDummySimulator mps(nrQubits);
    Simulators::MPSDummySimulator mpo(nrQubits, true);
    BOOST_CHECK(!mps.IsOperatorChain());
    BOOST_CHECK(mpo.IsOperatorChain());
    BOOST_CHECK(mpo.Clone()->IsOperatorChain());

    const std::vector<double> mpsCaps{2, 4, 8, 4, 2};
    const std::vector<double> mpoCaps{4, 16, 64, 16, 4};
    BOOST_CHECK(mps.getMaxBondDimensions() == mpsCaps);
    BOOST_CHECK(mpo.getMaxBondDimensions() == mpoCaps);

    const auto cx = Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kCXGateType, 2, 3);
    mps.ApplyGate(cx);
    mpo.ApplyGate(cx);
    BOOST_CHECK_GT(mpo.getCurrentBondDimensions()[2], mps.getCurrentBondDimensions()[2]);
}

// Layering moves operations on idle qubits forward. Measurements writing a
// classical bit must still follow earlier writes to it and the conditional
// operations reading its previous value.
BOOST_AUTO_TEST_CASE(LayersKeepClassicalBitOrder)
{
    using CF = Circuits::CircuitFactory<>;
    using Kind = Circuits::QuantumGateType;
    const auto x = [](Types::qubit_t q) { return std::static_pointer_cast<Circuits::IGateOperation<>>(CF::CreateGate(Kind::kXGateType, q)); };
    auto circuit = CF::CreateCircuit({CF::CreateGate(Kind::kXGateType, 2), CF::CreateGate(Kind::kCXGateType, 2, 3), CF::CreateGate(Kind::kCXGateType, 3, 2),
                                      CF::CreateMeasurement({{3, 0}}), CF::CreateSimpleConditionalGate(x(1), 0), CF::CreateMeasurement({{0, 0}}),
                                      CF::CreateMeasurement({{1, 1}})});
    const auto ops = circuit->GetOperations();
    const auto position = [&](const auto &op) { return std::find(ops.begin(), ops.end(), op) - ops.begin(); };
    const auto check = [&](const auto &layers, bool cloned, const char *name) {
        std::vector<std::shared_ptr<Circuits::IOperation<>>> flat;
        for (const auto &layer : layers)
            for (const auto &op : layer->GetOperations())
                flat.push_back(op);
        BOOST_REQUIRE_EQUAL(flat.size(), ops.size());
        // the classical bit 0 is written, read, written again
        std::vector<Circuits::OperationType> bitZero;
        for (size_t i = 0; i < flat.size(); ++i)
        {
            const auto bits = flat[i]->AffectedBits();
            if (std::find(bits.begin(), bits.end(), 0) != bits.end())
                bitZero.push_back(flat[i]->GetType());
        }
        const std::vector<Circuits::OperationType> expected{Circuits::OperationType::kMeasurement, Circuits::OperationType::kConditionalGate,
                                                            Circuits::OperationType::kMeasurement};
        BOOST_CHECK_MESSAGE(bitZero == expected, name << " reordered the uses of a classical bit");
        if (!cloned)
        {
            // the last write to bit 0 measures qubit 0
            for (auto it = flat.rbegin(); it != flat.rend(); ++it)
            {
                const auto bits = (*it)->AffectedBits();
                if ((*it)->GetType() == Circuits::OperationType::kMeasurement && std::find(bits.begin(), bits.end(), 0) != bits.end())
                {
                    BOOST_CHECK_MESSAGE(position(*it) == 5, name << " changed the last write to a bit");
                    break;
                }
            }
        }
    };
    check(circuit->ToLayers(), true, "ToLayers");
    check(circuit->ToLayersNoClone(), false, "ToLayersNoClone");
    check(circuit->ToMultipleQubitsLayers(), true, "ToMultipleQubitsLayers");
    check(circuit->ToMultipleQubitsLayersNoClone(), false, "ToMultipleQubitsLayersNoClone");

    // Through the network the optimized MPS executes the layered order.
    for (auto method : {Simulators::SimulationType::kMatrixProductState, Simulators::SimulationType::kMatrixProductOperator})
    {
        auto network = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{4}, std::vector<size_t>{2});
        network->SetMPSOptimizeSwaps(true);
        network->SetInitialQubitsMapOptimization(true);
        network->SetMPSOptimizationQubitsNumberThreshold(0);
        network->SetMPSOptimizationBondDimensionThreshold(0);
        network->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, method);
        network->CreateSimulator(Simulators::SimulatorType::kQCSim, method);
        // qubit 3 is 1, so the conditional X sets qubit 1; qubit 0 stays 0
        const auto counts = network->RepeatedExecuteOnHost(circuit, 0, 100);
        BOOST_REQUIRE_EQUAL(counts.size(), 1);
        BOOST_CHECK(counts.begin()->first == std::vector<bool>({false, true}));
    }
}

namespace
{

class GenericRank4TwoQubitGate final : public Circuits::TwoQubitsGate<>
{
  public:
    GenericRank4TwoQubitGate(Types::qubit_t q1, Types::qubit_t q2) : Circuits::TwoQubitsGate<>(q1, q2)
    {
    }

    Circuits::QuantumGateType GetGateType() const override
    {
        return Circuits::QuantumGateType::kNone;
    }

    Eigen::MatrixXcd GetMatrix() const override
    {
        return Eigen::Matrix4cd::Identity();
    }

    void Execute(const std::shared_ptr<Simulators::ISimulator> &, Circuits::OperationState &) const override
    {
    }

    std::shared_ptr<Circuits::IOperation<>> Clone() const override
    {
        return std::make_shared<GenericRank4TwoQubitGate>(GetQubit(0), GetQubit(1));
    }
};

} // namespace

BOOST_AUTO_TEST_CASE(MemoCacheKeyAndInvalidation)
{
    using CF = Circuits::CircuitFactory<>;
    using Kind = Circuits::QuantumGateType;

    constexpr size_t nrQubits = 8;
    Simulators::MPSDummySimulator sim(nrQubits);
    sim.SetCurrentBondDimensions({2, 3, 4, 3, 5, 3, 2});

    // Build a gate list where 12 single-qubit gates separate the first 2-qubit
    // gate from the second 2-qubit gate (pushing the second 2-qubit gate past
    // the raw operation index lookaheadDepth + 10 when lookaheadDepth == 1).
    std::vector<std::shared_ptr<Circuits::IOperation<>>> opsA;
    opsA.push_back(CF::CreateGate(Kind::kCXGateType, 0, 6));
    for (int i = 0; i < 12; ++i)
        opsA.push_back(CF::CreateGate(Kind::kXGateType, i % nrQubits));
    auto trailingGate = CF::CreateGate(Kind::kCXGateType, 0, 1);
    opsA.push_back(trailingGate);

    double bestCostA1 = std::numeric_limits<double>::infinity();
    const auto posA1 = sim.FindBestMeetingPosition(opsA, 0, 2, 2, 0.0, bestCostA1);

    // Repeat identical query (should hit cache and return identical result)
    double bestCostA2 = std::numeric_limits<double>::infinity();
    const auto posA2 = sim.FindBestMeetingPosition(opsA, 0, 2, 2, 0.0, bestCostA2);
    BOOST_CHECK_EQUAL(posA1, posA2);
    BOOST_CHECK_CLOSE(bestCostA1, bestCostA2, 1e-12);

    // Mutate the trailing 2-qubit gate in-place so its operands pull toward the
    // right end (5, 6) instead of (0, 1), keeping the same pointer address.
    trailingGate->SetQubit(6, 0);
    trailingGate->SetQubit(5, 1);

    double bestCostMutated = std::numeric_limits<double>::infinity();
    const auto posMutated = sim.FindBestMeetingPosition(opsA, 0, 2, 2, 0.0, bestCostMutated);
    BOOST_CHECK(posMutated != posA1 || std::abs(bestCostMutated - bestCostA1) > 1e-6);

    // Verify that changing growthFactorGate, growthFactorSwap, or MaxBondDimension
    // invalidates the cache and updates the returned cost.
    double bestCostBeforeFactor = std::numeric_limits<double>::infinity();
    (void)sim.FindBestMeetingPosition(opsA, 0, 2, 2, 0.0, bestCostBeforeFactor);

    sim.setGrowthFactorGate(0.95);
    double bestCostAfterGateFactor = std::numeric_limits<double>::infinity();
    (void)sim.FindBestMeetingPosition(opsA, 0, 2, 2, 0.0, bestCostAfterGateFactor);
    BOOST_CHECK_GT(bestCostAfterGateFactor, bestCostBeforeFactor);

    sim.setGrowthFactorSwap(0.5);
    double bestCostAfterSwapFactor = std::numeric_limits<double>::infinity();
    (void)sim.FindBestMeetingPosition(opsA, 0, 2, 2, 0.0, bestCostAfterSwapFactor);
    BOOST_CHECK_LT(bestCostAfterSwapFactor, bestCostAfterGateFactor);

    sim.SetMaxBondDimension(2);
    double bestCostAfterCap = std::numeric_limits<double>::infinity();
    (void)sim.FindBestMeetingPosition(opsA, 0, 2, 2, 0.0, bestCostAfterCap);
    BOOST_CHECK_LT(bestCostAfterCap, bestCostAfterSwapFactor);

    // Verify the 5-qubit MPS/MPO bondCost initialization and reset transitions
    // from swaps_opt_issues2.md Finding #1:
    // 1. Construct a 5-qubit dummy simulator with default settings.
    // 2. Query a single CX(0, 4) using lookahead depth 1 and heuristic depth 1.
    // 3. Call SetCurrentBondDimensions({1, 1, 1, 1}), Clear(), and SetInitialQubitsMap().
    // 4. Repeat the query and compare with a fresh evaluation on a Clone() (empty cache).
    for (bool isMpo : {false, true})
    {
        const double expectedFreshCost = isMpo ? 15.726 : 6.122;
        Simulators::MPSDummySimulator fiveQSim(5, isMpo);
        std::vector<std::shared_ptr<Circuits::IOperation<>>> cx04{CF::CreateGate(Kind::kCXGateType, 0, 4)};

        double initialCost = std::numeric_limits<double>::infinity();
        const auto initialPos = fiveQSim.FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, initialCost);
        BOOST_CHECK_EQUAL(initialPos, 0);
        BOOST_CHECK_CLOSE(initialCost, expectedFreshCost, 1e-9);

        fiveQSim.SetCurrentBondDimensions({1.0, 1.0, 1.0, 1.0});
        double cachedCostAfterSetDims = std::numeric_limits<double>::infinity();
        const auto posAfterSetDims = fiveQSim.FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, cachedCostAfterSetDims);

        auto freshClone = fiveQSim.Clone();
        double freshCost = std::numeric_limits<double>::infinity();
        const auto freshPos = freshClone->FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, freshCost);

        BOOST_CHECK_EQUAL(posAfterSetDims, freshPos);
        BOOST_CHECK_CLOSE(cachedCostAfterSetDims, freshCost, 1e-9);
        BOOST_CHECK_CLOSE(cachedCostAfterSetDims, expectedFreshCost, 1e-9);

        // Grow interior bonds via gate applications and explicit bond dimension
        // updates, populate the cache in the grown state, then verify Clear()
        // resets both currentBondDim and bondCost back to 1.
        fiveQSim.SetCurrentBondDimensions({2.0, 3.0, 3.0, 2.0});
        fiveQSim.ApplyGate(CF::CreateGate(Kind::kCXGateType, 1, 2));
        fiveQSim.ApplyGate(CF::CreateGate(Kind::kCXGateType, 2, 3));
        double grownCost = std::numeric_limits<double>::infinity();
        (void)fiveQSim.FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, grownCost);
        BOOST_CHECK_GT(grownCost, expectedFreshCost);

        fiveQSim.Clear();
        double costAfterClear = std::numeric_limits<double>::infinity();
        const auto posAfterClear = fiveQSim.FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, costAfterClear);
        auto cloneAfterClear = fiveQSim.Clone();
        double cloneCostAfterClear = std::numeric_limits<double>::infinity();
        const auto clonePosAfterClear = cloneAfterClear->FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, cloneCostAfterClear);
        BOOST_CHECK_EQUAL(posAfterClear, clonePosAfterClear);
        BOOST_CHECK_CLOSE(costAfterClear, cloneCostAfterClear, 1e-9);
        BOOST_CHECK_CLOSE(costAfterClear, expectedFreshCost, 1e-9);

        // Grow bonds and permute qubits, then verify SetInitialQubitsMap() also
        // resets both currentBondDim and bondCost and matches a fresh Clone().
        fiveQSim.SetCurrentBondDimensions({2.0, 3.0, 3.0, 2.0});
        fiveQSim.ApplyGate(CF::CreateGate(Kind::kCXGateType, 1, 3));
        double grownCost2 = std::numeric_limits<double>::infinity();
        (void)fiveQSim.FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, grownCost2);
        BOOST_CHECK_GT(grownCost2, expectedFreshCost);

        fiveQSim.SetInitialQubitsMap({0, 1, 2, 3, 4});
        double costAfterSetMap = std::numeric_limits<double>::infinity();
        const auto posAfterSetMap = fiveQSim.FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, costAfterSetMap);
        auto cloneAfterSetMap = fiveQSim.Clone();
        double cloneCostAfterSetMap = std::numeric_limits<double>::infinity();
        const auto clonePosAfterSetMap = cloneAfterSetMap->FindBestMeetingPosition(cx04, 0, 1, 1, 0.0, cloneCostAfterSetMap);
        BOOST_CHECK_EQUAL(posAfterSetMap, clonePosAfterSetMap);
        BOOST_CHECK_CLOSE(costAfterSetMap, cloneCostAfterSetMap, 1e-9);
        BOOST_CHECK_CLOSE(costAfterSetMap, expectedFreshCost, 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(MinOneStepCostMatchesRolloutFormulaAndControlledRank)
{
    using CF = Circuits::CircuitFactory<>;
    using Kind = Circuits::QuantumGateType;

    // 1. Exact arithmetic check from review Finding #2:
    // bondDims = [2, 3, 4, 3, 5, 3, 2], identity map, generic rank-4 gate on (0, 4),
    // growthFactorGate = 0.65, growthFactorSwap = 1.0, no additional bond cap.
    constexpr size_t nrQubits = 8;
    const std::vector<double> initBonds{2, 3, 4, 3, 5, 3, 2};
    const std::array<double, 4> expectedRolloutCosts{325.0985, 318.788, 342.304, 335.3125};

    auto genericGate = std::make_shared<GenericRank4TwoQubitGate>(0, 4);
    std::vector<std::shared_ptr<Circuits::IOperation<>>> singleGateWindow{genericGate};

    for (long long int m = 0; m < 4; ++m)
    {
        Simulators::MPSDummySimulator rolloutSim(nrQubits);
        rolloutSim.setGrowthFactorGate(0.65);
        rolloutSim.setGrowthFactorSwap(1.0);
        rolloutSim.SetCurrentBondDimensions(initBonds);
        rolloutSim.SwapQubitsToPosition(0, 4, m);
        rolloutSim.ApplyGate(genericGate);
        BOOST_CHECK_CLOSE(rolloutSim.getTotalSwappingCost(), expectedRolloutCosts[static_cast<size_t>(m)], 1e-9);
    }

    Simulators::MPSDummySimulator searchSim(nrQubits);
    searchSim.setGrowthFactorGate(0.65);
    searchSim.setGrowthFactorSwap(1.0);
    searchSim.SetCurrentBondDimensions(initBonds);

    double bestCost = std::numeric_limits<double>::infinity();
    const auto chosenPos = searchSim.FindBestMeetingPosition(singleGateWindow, 0, 0, 0, 0.0, bestCost);
    BOOST_CHECK_EQUAL(chosenPos, 1);
    BOOST_CHECK_CLOSE(bestCost, 318.788, 1e-9);

    // 2. Verify controlled two-qubit gates use Schmidt rank 2 (and rank^2 = 4 in MPO)
    // vs generic two-qubit gates using Schmidt rank 4 (and rank^2 = 16 in MPO).
    for (bool isMpo : {false, true})
    {
        Simulators::MPSDummySimulator controlledSim(nrQubits, isMpo);
        Simulators::MPSDummySimulator genericSim(nrQubits, isMpo);
        controlledSim.setGrowthFactorGate(1.0);
        genericSim.setGrowthFactorGate(1.0);
        const std::vector<double> uniformBonds(nrQubits - 1, 4.0);
        controlledSim.SetCurrentBondDimensions(uniformBonds);
        genericSim.SetCurrentBondDimensions(uniformBonds);

        const auto cxGate = CF::CreateGate(Kind::kCXGateType, 3, 4);
        const auto genAdjGate = std::make_shared<GenericRank4TwoQubitGate>(3, 4);

        controlledSim.ApplyGate(cxGate);
        genericSim.ApplyGate(genAdjGate);

        Simulators::MPSDummySimulator controlledMpoCheck(nrQubits, isMpo);
        Simulators::MPSDummySimulator genericMpoCheck(nrQubits, isMpo);
        controlledMpoCheck.setGrowthFactorGate(1.0);
        genericMpoCheck.setGrowthFactorGate(1.0);
        controlledMpoCheck.SetCurrentBondDimensions({8, 8, 8, 2, 8, 8, 8});
        genericMpoCheck.SetCurrentBondDimensions({8, 8, 8, 2, 8, 8, 8});
        controlledMpoCheck.ApplyGate(cxGate);
        genericMpoCheck.ApplyGate(genAdjGate);

        BOOST_CHECK_LT(controlledMpoCheck.getCurrentBondDimensions()[3], genericMpoCheck.getCurrentBondDimensions()[3]);

        // Also verify FindBestMeetingPosition at lookaheadDepth = 0 picks the exact
        // minimum-cost position for controlled gates as well.
        Simulators::MPSDummySimulator cxSearchSim(nrQubits, isMpo);
        cxSearchSim.setGrowthFactorGate(0.65);
        cxSearchSim.setGrowthFactorSwap(1.0);
        cxSearchSim.SetCurrentBondDimensions(initBonds);

        const auto cxSpanGate = CF::CreateGate(Kind::kCXGateType, 0, 4);
        std::vector<std::shared_ptr<Circuits::IOperation<>>> cxWindow{cxSpanGate};

        double minExplicitCost = std::numeric_limits<double>::infinity();
        long long int minExplicitPos = -1;
        for (long long int m = 0; m < 4; ++m)
        {
            Simulators::MPSDummySimulator explicitSim(nrQubits, isMpo);
            explicitSim.setGrowthFactorGate(0.65);
            explicitSim.setGrowthFactorSwap(1.0);
            explicitSim.SetCurrentBondDimensions(initBonds);
            explicitSim.SwapQubitsToPosition(0, 4, m);
            explicitSim.ApplyGate(cxSpanGate);
            if (explicitSim.getTotalSwappingCost() < minExplicitCost)
            {
                minExplicitCost = explicitSim.getTotalSwappingCost();
                minExplicitPos = m;
            }
        }

        double cxBestCost = std::numeric_limits<double>::infinity();
        const auto cxChosenPos = cxSearchSim.FindBestMeetingPosition(cxWindow, 0, 0, 0, 0.0, cxBestCost);
        BOOST_CHECK_EQUAL(cxChosenPos, minExplicitPos);
        BOOST_CHECK_CLOSE(cxBestCost, minExplicitCost, 1e-9);
    }
}

BOOST_AUTO_TEST_CASE(ConfigurableBranchingDepthAndLargeChainFallbacks)
{
    using CF = Circuits::CircuitFactory<>;
    using Kind = Circuits::QuantumGateType;

    // 1. Verify deeper branching (lookaheadDepth - lookaheadDepthWithHeuristic)
    // is never worse than shallower branching on the same lookahead horizon, and
    // strictly improves routes when the optimal meeting position (m = 3 for gate
    // (1, 9) on 11 qubits) preserves adjacency of passive qubit pairs (0,2), (2,3),
    // (4,5), (5,6), (6,7), (7,8), (8,10) that do not touch qubits 1 or 9 (so
    // partner-gravity has zero pulls and single-step policies only test m in {1, 4, 5, 8}).
    constexpr size_t nrQubits = 11;
    std::vector<std::shared_ptr<Circuits::IOperation<>>> ops{
        CF::CreateGate(Kind::kCXGateType, 0, 1), // step 0 (adjacent)
        CF::CreateGate(Kind::kCXGateType, 1, 9), // step 1: only m1 = 3 keeps all of steps 2..8 adjacent
        CF::CreateGate(Kind::kCXGateType, 0, 2), // step 2: eliminates m1 = 1
        CF::CreateGate(Kind::kCXGateType, 2, 3), // step 3: eliminates m1 = 2
        CF::CreateGate(Kind::kCXGateType, 4, 5), // step 4: eliminates m1 = 4
        CF::CreateGate(Kind::kCXGateType, 5, 6), // step 5: eliminates m1 = 5
        CF::CreateGate(Kind::kCXGateType, 6, 7), // step 6: eliminates m1 = 6
        CF::CreateGate(Kind::kCXGateType, 7, 8), // step 7: eliminates m1 = 7
        CF::CreateGate(Kind::kCXGateType, 8, 10) // step 8: eliminates m1 = 8
    };

    const std::vector<double> uniformBonds(nrQubits - 1, 2.0);
    auto makeUniformSim = [&]() {
        Simulators::MPSDummySimulator s(nrQubits);
        s.SetMaxBondDimension(2);
        s.setGrowthFactorGate(1.0);
        s.setGrowthFactorSwap(1.0);
        s.SetCurrentBondDimensions(uniformBonds);
        return s;
    };

    // Starting at index 1 (where (1, 9) is at stepIdx == 0):
    // 1-step branching (depth 7, heuristic 6) enumerates m = 3 and achieves 0 extra swaps on steps 2..8,
    // strictly beating 0-step branching (depth 7, heuristic 7) which only tests {1, 4, 5, 8}.
    auto simRoot0 = makeUniformSim();
    auto simRoot1 = makeUniformSim();
    double costRoot0 = std::numeric_limits<double>::infinity();
    double costRoot1 = std::numeric_limits<double>::infinity();
    const auto posRoot0 = simRoot0.FindBestMeetingPosition(ops, 1, 7, 7, 0.0, costRoot0);
    const auto posRoot1 = simRoot1.FindBestMeetingPosition(ops, 1, 7, 6, 0.0, costRoot1);
    BOOST_CHECK_EQUAL(posRoot1, 3);
    BOOST_CHECK_NE(posRoot0, 3);
    BOOST_CHECK_LT(costRoot1, costRoot0);

    // Starting at index 0 (where (0, 1) is at stepIdx == 0 and (1, 9) is at stepIdx == 1):
    // 2-step branching (depth 8, heuristic 6) branches at stepIdx == 1 and finds m1 = 3,
    // strictly beating 1-step branching (depth 8, heuristic 7) and 0-step branching (depth 8, heuristic 8).
    auto sim0 = makeUniformSim();
    auto sim1 = makeUniformSim();
    auto sim2 = makeUniformSim();

    double costBranch0 = std::numeric_limits<double>::infinity();
    double costBranch1 = std::numeric_limits<double>::infinity();
    double costBranch2 = std::numeric_limits<double>::infinity();

    (void)sim0.FindBestMeetingPosition(ops, 0, 8, 8, 0.0, costBranch0);
    (void)sim1.FindBestMeetingPosition(ops, 0, 8, 7, 0.0, costBranch1);
    (void)sim2.FindBestMeetingPosition(ops, 0, 8, 6, 0.0, costBranch2);

    BOOST_CHECK_LE(costBranch1, costBranch0 + 1e-9);
    BOOST_CHECK_LT(costBranch2, costBranch1);

    // 2. Exercise > 64 qubits snapshot heap fallback and > 128 bond span candidate heap fallback.
    constexpr size_t largeNrQubits = 140;
    Simulators::MPSDummySimulator largeSim(largeNrQubits);
    largeSim.SetMaxBondDimension(32);

    std::vector<double> largeBonds(largeNrQubits - 1, 2.0);
    for (size_t i = 0; i < largeBonds.size(); ++i)
        largeBonds[i] = static_cast<double>(2 + (i % 5));
    largeSim.SetCurrentBondDimensions(largeBonds);

    const auto mapBefore = largeSim.getQubitsMap();
    const auto bondsBefore = largeSim.getCurrentBondDimensions();
    const double costBefore = largeSim.getTotalSwappingCost();

    std::vector<std::shared_ptr<Circuits::IOperation<>>> largeOps{CF::CreateGate(Kind::kCXGateType, 1, 136), // span = 135 > 128 stack candidate buffer
                                                                  CF::CreateGate(Kind::kCXGateType, 1, 70), CF::CreateGate(Kind::kCXGateType, 70, 136)};

    double largeBestCost = std::numeric_limits<double>::infinity();
    const auto largeMeetPos = largeSim.FindBestMeetingPosition(largeOps, 0, 2, 1, 0.0, largeBestCost);

    BOOST_CHECK_GE(largeMeetPos, 1);
    BOOST_CHECK_LT(largeMeetPos, 136);
    BOOST_CHECK(std::isfinite(largeBestCost));
    BOOST_CHECK(largeSim.getQubitsMap() == mapBefore);
    BOOST_CHECK(largeSim.getCurrentBondDimensions() == bondsBefore);
    BOOST_CHECK_CLOSE(largeSim.getTotalSwappingCost(), costBefore, 1e-12);
}

BOOST_AUTO_TEST_CASE(RoutingOverheadVsSimulationTimeBenchmarkMPSAndMPO)
{
    constexpr std::array<unsigned int, 4> circuitSeeds{13579u, 24680u, 97531u, 86420u};
    constexpr int numRepeats = 2;

    for (auto method : {Simulators::SimulationType::kMatrixProductState, Simulators::SimulationType::kMatrixProductOperator})
    {
        const bool isMpo = (method == Simulators::SimulationType::kMatrixProductOperator);
        const char *label = isMpo ? "MPO" : "MPS";
        const int nrQubits = isMpo ? 11 : 13;
        const int nrGates = isMpo ? 70 : 180;
        const int maxBondDim = isMpo ? 32 : 64;

        double totalMapMs = 0.0;
        double totalColdPlanMs = 0.0;
        double totalWarmPlanMs = 0.0;
        double totalSimFixedEndMs = 0.0;
        double totalSimLocalOptMs = 0.0;
        double totalSimLookaheadMs = 0.0;
        double totalHeuristicSwapCost = 0.0;
        double totalLookaheadSwapCost = 0.0;

        for (unsigned int seed : circuitSeeds)
        {
            const auto randomCirc = CreateRandomMPSCircuit(nrQubits, nrGates, seed);
            const auto layers = randomCirc->ToMultipleQubitsLayers();
            const auto optCirc = randomCirc->LayersToCircuit(layers);
            const auto &ops = optCirc->GetOperations();
            const std::vector<std::shared_ptr<Circuits::IOperation<>>> opsVec(ops.begin(), ops.end());

            // 1. Measure standalone routing planning time (cold cache vs warm cache)
            Simulators::MPSDummySimulator planner(nrQubits, isMpo);
            planner.SetMaxBondDimension(maxBondDim);

            const auto tMapStart = std::chrono::steady_clock::now();
            const auto optimalMap = planner.ComputeOptimalQubitsMap(layers);
            const auto tMapEnd = std::chrono::steady_clock::now();
            totalMapMs += std::chrono::duration<double, std::milli>(tMapEnd - tMapStart).count();

            // Baseline dummy swap cost with local min-bond-cost heuristic only
            planner.SetInitialQubitsMap(optimalMap);
            planner.ApplyGates(ops);
            totalHeuristicSwapCost += planner.getTotalSwappingCost();

            auto runOnlinePlanningPass = [&](double &planningMsOut) {
                planner.SetInitialQubitsMap(optimalMap);
                planner.setTotalSwappingCost(0);
                planner.SetCurrentBondDimensions(std::vector<double>(nrQubits - 1, 1.0));
                const auto tPlanStart = std::chrono::steady_clock::now();
                for (size_t i = 0; i < opsVec.size(); ++i)
                {
                    const auto qbits = opsVec[i]->AffectedQubits();
                    if (qbits.size() == 2)
                    {
                        double bestCost = std::numeric_limits<double>::infinity();
                        const auto pos = planner.FindBestMeetingPosition(opsVec, static_cast<long long int>(i), 12, 11, 0.0, bestCost);
                        planner.SwapQubitsToPosition(static_cast<long long int>(qbits[0]), static_cast<long long int>(qbits[1]), pos);
                    }
                    planner.ApplyGate(opsVec[i]);
                }
                const auto tPlanEnd = std::chrono::steady_clock::now();
                planningMsOut = std::chrono::duration<double, std::milli>(tPlanEnd - tPlanStart).count();
                return planner.getTotalSwappingCost();
            };

            double coldPlanMs = 0.0;
            double warmPlanMs = 0.0;
            const double coldCost = runOnlinePlanningPass(coldPlanMs);
            const double warmCost = runOnlinePlanningPass(warmPlanMs);
            BOOST_CHECK_CLOSE(coldCost, warmCost, 1e-9);
            totalColdPlanMs += coldPlanMs;
            totalWarmPlanMs += warmPlanMs;
            totalLookaheadSwapCost += coldCost;

            // 2. Measure actual simulator execution time across 3 configurations:
            //    (a) Fixed-end routing: SetUseOptimalMeetingPosition(false)
            //    (b) Default local bond-aware routing: SetUseOptimalMeetingPosition(true), SetLookaheadDepth(0)
            //    (c) Lookahead routing: SetUseOptimalMeetingPosition(true), SetLookaheadDepth(12), SetLookaheadDepthWithHeuristic(11)
            for (int rep = 0; rep < numRepeats; ++rep)
            {
                auto simFixed = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, method);
                simFixed->AllocateQubits(nrQubits);
                simFixed->Configure("matrix_product_state_max_bond_dimension", std::to_string(maxBondDim).c_str());
                simFixed->Configure("matrix_product_operator_max_bond_dimension", std::to_string(maxBondDim).c_str());
                simFixed->Initialize();
                simFixed->SetInitialQubitsMap(std::vector<long long int>(optimalMap.begin(), optimalMap.end()));
                simFixed->SetUseOptimalMeetingPosition(false);

                Circuits::OperationState stateFixed;
                stateFixed.AllocateBits(nrQubits);
                const auto tSimFixedStart = std::chrono::steady_clock::now();
                optCirc->Execute(simFixed, stateFixed);
                const auto tSimFixedEnd = std::chrono::steady_clock::now();
                totalSimFixedEndMs += std::chrono::duration<double, std::milli>(tSimFixedEnd - tSimFixedStart).count();

                auto simLocalOpt = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, method);
                simLocalOpt->AllocateQubits(nrQubits);
                simLocalOpt->Configure("matrix_product_state_max_bond_dimension", std::to_string(maxBondDim).c_str());
                simLocalOpt->Configure("matrix_product_operator_max_bond_dimension", std::to_string(maxBondDim).c_str());
                simLocalOpt->Initialize();
                simLocalOpt->SetInitialQubitsMap(std::vector<long long int>(optimalMap.begin(), optimalMap.end()));
                simLocalOpt->SetUseOptimalMeetingPosition(true);
                simLocalOpt->SetLookaheadDepth(0);

                Circuits::OperationState stateLocalOpt;
                stateLocalOpt.AllocateBits(nrQubits);
                const auto tSimLocalStart = std::chrono::steady_clock::now();
                optCirc->Execute(simLocalOpt, stateLocalOpt);
                const auto tSimLocalEnd = std::chrono::steady_clock::now();
                totalSimLocalOptMs += std::chrono::duration<double, std::milli>(tSimLocalEnd - tSimLocalStart).count();

                auto simLookahead = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, method);
                simLookahead->AllocateQubits(nrQubits);
                simLookahead->Configure("matrix_product_state_max_bond_dimension", std::to_string(maxBondDim).c_str());
                simLookahead->Configure("matrix_product_operator_max_bond_dimension", std::to_string(maxBondDim).c_str());
                simLookahead->Initialize();
                simLookahead->SetInitialQubitsMap(std::vector<long long int>(optimalMap.begin(), optimalMap.end()));
                simLookahead->SetUseOptimalMeetingPosition(true);
                simLookahead->SetLookaheadDepth(12);
                simLookahead->SetLookaheadDepthWithHeuristic(11);
                simLookahead->SetUpcomingGates(optCirc->GetOperations());

                Circuits::OperationState stateLookahead;
                stateLookahead.AllocateBits(nrQubits);
                const auto tSimOptStart = std::chrono::steady_clock::now();
                optCirc->Execute(simLookahead, stateLookahead);
                const auto tSimOptEnd = std::chrono::steady_clock::now();
                totalSimLookaheadMs += std::chrono::duration<double, std::milli>(tSimOptEnd - tSimOptStart).count();
            }
        }

        const double nCircs = static_cast<double>(circuitSeeds.size());
        const double nRuns = nCircs * numRepeats;
        BOOST_TEST_MESSAGE("[" << label << " avg over " << circuitSeeds.size() << " circuits] "
                               << "Initial map: " << (totalMapMs / nCircs) << " ms | "
                               << "Online planning cold: " << (totalColdPlanMs / nCircs) << " ms, warm: " << (totalWarmPlanMs / nCircs) << " ms | "
                               << "Modeled cost (local heuristic -> lookahead): " << (totalHeuristicSwapCost / nCircs) << " -> "
                               << (totalLookaheadSwapCost / nCircs) << " | "
                               << "Sim time fixed-end: " << (totalSimFixedEndMs / nRuns) << " ms, "
                               << "default local-opt (depth 0): " << (totalSimLocalOptMs / nRuns) << " ms, "
                               << "lookahead (total incl. planning): " << (totalSimLookaheadMs / nRuns) << " ms");
    }
}

BOOST_AUTO_TEST_CASE(VirtualBondHistoryLifecycleAndSaturatedRouting)
{
    using CF = Circuits::CircuitFactory<>;
    using Kind = Circuits::QuantumGateType;

    // 1. Direct MPSDummySimulator check: SetMaxBondDimension must not reset currentBondDim
    //    when re-applying an unchanged limit or clamping to an updated limit, and
    //    RecordAdjacentTwoQubitGate must grow only the specified bond.
    {
        Simulators::MPSDummySimulator dummy(8, false);
        dummy.SetMaxBondDimension(4);
        const std::vector<double> customBonds{2.0, 3.5, 4.0, 4.0, 3.0, 2.0, 2.0};
        dummy.SetCurrentBondDimensions(customBonds);
        BOOST_CHECK(dummy.getCurrentBondDimensions() == customBonds);

        // Re-applying the same limit is a no-op and preserves currentBondDim
        dummy.SetMaxBondDimension(4);
        BOOST_CHECK(dummy.getCurrentBondDimensions() == customBonds);

        // Lowering the limit clamps existing bond dimensions without resetting to 1.0
        dummy.SetMaxBondDimension(3);
        const std::vector<double> clampedBonds{2.0, 3.0, 3.0, 3.0, 3.0, 2.0, 2.0};
        BOOST_CHECK(dummy.getCurrentBondDimensions() == clampedBonds);

        // SetRoutingMap updates the map without resetting currentBondDim
        dummy.SetRoutingMap({7, 6, 5, 4, 3, 2, 1, 0});
        BOOST_CHECK(dummy.getCurrentBondDimensions() == clampedBonds);

        // ResetBondDimensions explicitly resets currentBondDim to 1.0
        dummy.ResetBondDimensions();
        const std::vector<double> unitBonds(7, 1.0);
        BOOST_CHECK(dummy.getCurrentBondDimensions() == unitBonds);

        dummy.setGrowthFactorGate(1.0);
        dummy.RecordAdjacentTwoQubitGate(2, false, true);
        BOOST_CHECK_CLOSE(dummy.getCurrentBondDimensions()[2], 2.0, 1e-12);
    }

    // 2. Saturated-bond lifecycle tests through ISimulator. Compare uninterrupted
    //    execution with cloning/restoration and skipped operations, independently
    //    of which route the configured accuracy/runtime heuristic prefers.
    constexpr size_t n = 9;
    auto buildPrefixOps = []() {
        return std::vector<std::shared_ptr<Circuits::IOperation<>>>{
            CF::CreateGate(Kind::kHadamardGateType, 0),
            CF::CreateGate(Kind::kHadamardGateType, 1),
            CF::CreateGate(Kind::kCPGateType, 0, 1, 0, 1e-3), // bond 0 -> phys 2/4, virt 2/4
            CF::CreateGate(Kind::kHadamardGateType, 2),
            CF::CreateGate(Kind::kCPGateType, 1, 2, 0, 1e-3), // overlaps {0,1} -> emits (0,1); bond 1 -> phys 2/4, virt 2/4
            CF::CreateGate(Kind::kCXGateType, 2, 3),          // overlaps {1,2} -> emits first (1,2); Bell pair #1 on (2,3) (bond 2 -> phys 2/4, virt 2/4)
            CF::CreateGate(Kind::kCPGateType, 1, 2, 0, 1e-3), // overlaps {2,3} -> emits (2,3); second gate on (1,2) (bond 1 -> virt 4/16)
            CF::CreateGate(Kind::kHadamardGateType, 4),
            CF::CreateGate(Kind::kCPGateType, 3, 4, 0, 1e-3), // bond 3 -> phys 2/4, virt 2/4
            CF::CreateGate(Kind::kCXGateType, 4, 5),          // overlaps {3,4} -> emits first (3,4); Bell pair #2 on (4,5) (bond 4 -> phys 2/4, virt 2/4)
            CF::CreateGate(Kind::kCPGateType, 3, 4, 0, 1e-3)  // overlaps {4,5} -> emits (4,5); trailing adjacent gate on (3,4) (bond 3 -> virt 4/16)
        };
    };
    auto buildPrefixEndingWithAdjacentOps = [&]() { return buildPrefixOps(); };
    auto buildSuffixOps = []() {
        return std::vector<std::shared_ptr<Circuits::IOperation<>>>{
            CF::CreateGate(Kind::kCPGateType, 2, 4, 0, 0.0), // non-adjacent callback #2 on (2,4): m=3 swaps (2,3) (exact), m=2 swaps (3,4) (truncates!)
            CF::CreateGate(Kind::kCPGateType, 3, 4, 0, 0.0)  // lookahead gate on (3,4): m=3 wins when bonds {1,2,3,4} = {4,2,4,2}; m=2 wins when bond 3 is 2/4
        };
    };
    auto bellCoherenceError = [](const std::shared_ptr<Simulators::ISimulator> &sim) {
        const double xx23 = sim->ExpectationValue("IIXXIIIII");
        const double xx45 = sim->ExpectationValue("IIIIXXIII");
        const double zz23 = sim->ExpectationValue("IIZZIIIII");
        const double zz45 = sim->ExpectationValue("IIIIZZIII");
        BOOST_REQUIRE(std::isfinite(xx23));
        BOOST_REQUIRE(std::isfinite(xx45));
        BOOST_REQUIRE(std::isfinite(zz23));
        BOOST_REQUIRE(std::isfinite(zz45));
        const double xxxx = sim->ExpectationValue("IIXXXXIII");
        const double zzzz = sim->ExpectationValue("IIZZZZIII");
        const double yyyy = sim->ExpectationValue("IIYYYYIII");
        BOOST_REQUIRE(std::isfinite(xxxx));
        BOOST_REQUIRE(std::isfinite(zzzz));
        BOOST_REQUIRE(std::isfinite(yyyy));
        const double err = std::abs(xx23 - 1.0) + std::abs(xx45 - 1.0) + std::abs(zz23 - 1.0) + std::abs(zz45 - 1.0) + std::abs(xxxx - 1.0) +
                           std::abs(zzzz - 1.0) + std::abs(yyyy - 1.0);
        return err;
    };

    for (const auto simType : {Simulators::SimulationType::kMatrixProductState, Simulators::SimulationType::kMatrixProductOperator})
    {
        const bool isMpo = (simType == Simulators::SimulationType::kMatrixProductOperator);
        const char *physCapStr = isMpo ? "4" : "2";

        for (const char *fusionSetting : {"true", "false"})
        {
            auto configureSim = [&](const std::shared_ptr<Simulators::ISimulator> &sim) {
                sim->AllocateQubits(n);
                sim->Configure("gate_fusion", fusionSetting);
                sim->Configure("mps_routing_virtual_bonds", "true");
                sim->Configure("matrix_product_state_max_bond_dimension", physCapStr);
                sim->Configure("matrix_product_operator_max_bond_dimension", physCapStr);
                sim->Configure("matrix_product_state_truncation_threshold", "1e-16");
                sim->Configure("matrix_product_operator_truncation_threshold", "1e-16");
                sim->Initialize();
                sim->setGrowthFactorGate(1.0);
                sim->setGrowthFactorSwap(1.0);
                sim->SetUseOptimalMeetingPosition(true);
                sim->SetLookaheadDepth(2);
                sim->SetLookaheadDepthWithHeuristic(0);
            };

            auto executePrefixAndRouteFirstSuffixGate =
                [](const std::shared_ptr<Simulators::ISimulator> &sim, const std::vector<std::shared_ptr<Circuits::IOperation<>>> &prefOps,
                   const std::vector<std::shared_ptr<Circuits::IOperation<>>> &sfxOps, Circuits::OperationState &state) {
                    for (size_t i = 0; i < prefOps.size(); ++i)
                        prefOps[i]->Execute(sim, state);
                    sfxOps[0]->Execute(sim, state);
                };

            double uninterruptedError = 0.;
            // The compatibility override and explicit virtual opt-out agree.
            {
                auto simLegacy = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                simLegacy->Configure("mps_routing_legacy_cap", "true");
                configureSim(simLegacy);
                const auto prefOps = buildPrefixOps();
                const auto sfxOps = buildSuffixOps();
                auto allOps = prefOps;
                allOps.insert(allOps.end(), sfxOps.begin(), sfxOps.end());
                simLegacy->SetUpcomingGates(allOps);
                Circuits::OperationState state;
                state.AllocateBits(n);
                executePrefixAndRouteFirstSuffixGate(simLegacy, prefOps, sfxOps, state);
                auto plain = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(plain);
                plain->Configure("mps_routing_virtual_bonds", "false");
                plain->SetUpcomingGates(allOps);
                executePrefixAndRouteFirstSuffixGate(plain, prefOps, sfxOps, state);
                BOOST_CHECK_SMALL(bellCoherenceError(simLegacy) - bellCoherenceError(plain), 1e-9);
            }

            // (a) Reference run, without lifecycle boundaries.
            {
                auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(sim);
                const auto prefOps = buildPrefixOps();
                const auto sfxOps = buildSuffixOps();
                auto allOps = prefOps;
                allOps.insert(allOps.end(), sfxOps.begin(), sfxOps.end());
                sim->SetUpcomingGates(allOps);
                Circuits::OperationState state;
                state.AllocateBits(n);
                executePrefixAndRouteFirstSuffixGate(sim, prefOps, sfxOps, state);
                uninterruptedError = bellCoherenceError(sim);
            }

            // (b) [P2 #1 regression] Prefix ends with adjacent two-qubit gates on (2,3) and
            //     SetUpcomingGates(suffix) is called directly without an intervening SaveState or Clone.
            {
                auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(sim);
                auto prefCirc = CF::CreateCircuit(buildPrefixEndingWithAdjacentOps());
                sim->SetUpcomingGates(prefCirc->GetOperations());
                Circuits::OperationState state;
                state.AllocateBits(n);
                prefCirc->Execute(sim, state);

                const auto sfxOps = buildSuffixOps();
                sim->SetUpcomingGates(sfxOps);
                sfxOps[0]->Execute(sim, state);
                BOOST_CHECK_SMALL(bellCoherenceError(sim) - uninterruptedError, 1e-9);
            }

            // (c) Clone() after prefix + SetUpcomingGates(suffix) preserves virtual bond history
            {
                auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(sim);
                auto prefCirc = CF::CreateCircuit(buildPrefixEndingWithAdjacentOps());
                sim->SetUpcomingGates(prefCirc->GetOperations());
                Circuits::OperationState state;
                state.AllocateBits(n);
                prefCirc->Execute(sim, state);

                std::shared_ptr<Simulators::ISimulator> cloned(sim->Clone());
                const auto sfxOps = buildSuffixOps();
                cloned->SetUpcomingGates(sfxOps);
                sfxOps[0]->Execute(cloned, state);
                BOOST_CHECK_SMALL(bellCoherenceError(cloned) - uninterruptedError, 1e-9);
            }

            // (d) [P2 #2 regression] Clone() and SaveState()->Reset()->RestoreState() midway through
            //     the SAME installed circuit (without reinstalling a separate suffix) must NOT replay
            //     already-accounted-for prefix gates on bond 3 when execution resumes.
            {
                auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(sim);
                const auto prefOps = buildPrefixOps();
                const auto sfxOps = buildSuffixOps();
                auto allOps = prefOps;
                allOps.insert(allOps.end(), sfxOps.begin(), sfxOps.end());
                sim->SetUpcomingGates(allOps);

                Circuits::OperationState state;
                state.AllocateBits(n);
                for (size_t i = 0; i < prefOps.size(); ++i)
                    prefOps[i]->Execute(sim, state);

                // Resume midway on a Clone() of the same installed circuit
                std::shared_ptr<Simulators::ISimulator> cloned(sim->Clone());
                sfxOps[0]->Execute(cloned, state);
                BOOST_CHECK_SMALL(bellCoherenceError(cloned) - uninterruptedError, 1e-9);

                // SaveState -> Reset -> RestoreState midway on the same installed circuit
                sim->SaveState();
                sim->Reset();
                sim->RestoreState();
                sfxOps[0]->Execute(sim, state);
                BOOST_CHECK_SMALL(bellCoherenceError(sim) - uninterruptedError, 1e-9);
            }

            // (e) [P2 #3 regression] ExecuteNonMeasurements() stepping over deferred adjacent two-qubit
            //     gates (dependent on a mid-circuit measurement) must NOT count those unexecuted gates
            //     as having grown the dummy simulator's bond dimensions.
            {
                auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(sim);
                auto nonMeasOps = buildPrefixOps();
                nonMeasOps.back() = CF::CreateGate(Kind::kCXGateType, 5, 6);   // flushes CX(4,5) under gate_fusion while leaving bond 3 at 2/4 (single CP(3,4))
                nonMeasOps.push_back(CF::CreateMeasurement({{0, 0}}));         // mid-circuit measurement on qubit 0
                nonMeasOps.push_back(CF::CreateGate(Kind::kCXGateType, 0, 4)); // deferred: marks qubit 4 as measurement-dependent
                nonMeasOps.push_back(CF::CreateGate(Kind::kCPGateType, 3, 4, 0, 0.0)); // deferred adjacent gate on bond 3 (MUST NOT grow bond 3 to 4/16!)
                nonMeasOps.push_back(CF::CreateGate(Kind::kCPGateType, 3, 4, 0, 0.0)); // deferred adjacent gate on bond 3 (MUST NOT grow bond 3 to 4/16!)
                nonMeasOps.push_back(
                    CF::CreateGate(Kind::kCPGateType, 6, 8, 0, 0.0)); // independent non-adjacent gate on (6,8): executes in ExecuteNonMeasurements!
                auto nonMeasCirc = CF::CreateCircuit(nonMeasOps);
                sim->SetUpcomingGates(nonMeasCirc->GetOperations());
                Circuits::OperationState state;
                state.AllocateBits(n);
                size_t maxBondDim = 0;
                const auto executed = nonMeasCirc->ExecuteNonMeasurements(sim, state, &maxBondDim);
                BOOST_REQUIRE_EQUAL(executed.size(), 5U); // from Measure(0) onward: F, F, F, F, T
                BOOST_CHECK(!executed[0] && !executed[1] && !executed[2] && !executed[3] && executed[4]);

                // Compare to a fresh run containing only operations that executed.
                // Deferred gates must influence neither native state nor history.
                const auto sfxOps = buildSuffixOps();
                sim->SetUpcomingGates(sfxOps);
                sfxOps[0]->Execute(sim, state);
                auto clean = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(clean);
                auto executedOps = buildPrefixOps();
                executedOps.back() = CF::CreateGate(Kind::kCXGateType, 5, 6);
                executedOps.push_back(nonMeasOps.back());
                clean->SetUpcomingGates(executedOps);
                for (const auto &op : executedOps)
                    op->Execute(clean, state);
                clean->SetUpcomingGates(sfxOps);
                sfxOps[0]->Execute(clean, state);
                BOOST_CHECK_SMALL(bellCoherenceError(sim) - bellCoherenceError(clean), 1e-9);
            }

            // (f) SaveState() before dummySim creation -> pollute bond 3 to 4/16 -> RestoreState()
            //     wipes the post-SaveState virtual bond history back to 1.0s.
            {
                auto sim = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(sim);
                sim->SaveState();

                auto polluteCirc = CF::CreateCircuit(buildPrefixOps());
                sim->SetUpcomingGates(polluteCirc->GetOperations());
                Circuits::OperationState state;
                state.AllocateBits(n);
                polluteCirc->Execute(sim, state);

                sim->RestoreState();

                // After RestoreState(), running a prefix with only a single CP(3,4) gate must NOT
                // inherit the pre-RestoreState saturation (4/16) of bond 3.
                auto unsatOps = buildPrefixOps();
                unsatOps.back() = CF::CreateGate(Kind::kCXGateType, 5, 6); // flushes CX(4,5) under gate_fusion while leaving bond 3 at 2/4 (single CP(3,4))
                const auto sfxOps = buildSuffixOps();
                auto allUnsatOps = unsatOps;
                allUnsatOps.insert(allUnsatOps.end(), sfxOps.begin(), sfxOps.end());
                sim->SetUpcomingGates(allUnsatOps);
                executePrefixAndRouteFirstSuffixGate(sim, unsatOps, sfxOps, state);
                auto clean = Simulators::SimulatorsFactory::CreateSimulator(Simulators::SimulatorType::kQCSim, simType);
                configureSim(clean);
                clean->SetUpcomingGates(allUnsatOps);
                executePrefixAndRouteFirstSuffixGate(clean, unsatOps, sfxOps, state);
                BOOST_CHECK_SMALL(bellCoherenceError(sim) - bellCoherenceError(clean), 1e-9);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(BoundedVirtualRoutingCostModel)
{
    using Dummy = Simulators::MPSDummySimulator;
    for (bool mpo : {false, true})
        for (size_t n : {1U, 4U, 9U, 20U, 100U})
            for (long long cap : {1LL, 4LL, 64LL, 512LL})
            {
                const auto effective = Dummy::EffectiveLookaheadMaxBondDimension(n, mpo, cap);
                BOOST_CHECK_GE(effective, cap);
                BOOST_CHECK_LE(effective, 4 * cap);
                BOOST_CHECK_EQUAL(Dummy::EffectiveMapMaxBondDimension(n, mpo, cap), effective);
                BOOST_CHECK_EQUAL(Dummy::EffectiveLookaheadMaxBondDimension(n, mpo, 0), 0);
            }
    Dummy dummy(20);
    dummy.SetRoutingCostModel(4, 0.);
    BOOST_CHECK_EQUAL(dummy.BondCost(2), 8.);
    BOOST_CHECK_EQUAL(dummy.BondCost(16), 64.); // Capped runtime, no risk penalty.
    dummy.SetRoutingCostModel(4, 1.);
    BOOST_CHECK_EQUAL(dummy.BondCost(16), 256.); // Three cap-sized overflow units.
    BOOST_CHECK_EQUAL(dummy.Clone()->BondCost(16), dummy.BondCost(16));
    dummy.SetRoutingCostModel(0, 1.);
    BOOST_CHECK_EQUAL(dummy.BondCost(16), 4096.); // Unlimited runtime.
    BOOST_CHECK_THROW(dummy.SetRoutingCostModel(4, -1.), std::invalid_argument);
    BOOST_CHECK_THROW(dummy.SetRoutingCostModel(4, std::numeric_limits<double>::infinity()), std::invalid_argument);
    for (bool mpo : {false, true})
        for (double weight : {0., 0.5, 2.})
        {
            Dummy planner(8, mpo);
            planner.SetMaxBondDimension(16);
            planner.SetRoutingCostModel(4, weight);
            planner.SetCurrentBondDimensions({2, 4, 8, 8, 4, 2, 2});
            const auto gate = Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kCXGateType, 1, 6);
            double minimum = std::numeric_limits<double>::infinity();
            for (long long position = 1; position < 6; ++position)
            {
                auto candidate = planner.Clone();
                candidate->SwapQubitsToPosition(1, 6, position);
                candidate->ApplyGate(gate);
                minimum = std::min(minimum, candidate->getTotalSwappingCost());
            }
            double cost = std::numeric_limits<double>::infinity();
            const auto chosen = planner.FindBestMeetingPosition({gate}, 0, 1, 0, 0., cost);
            BOOST_CHECK_GE(chosen, 1);
            BOOST_CHECK_LT(chosen, 6);
            BOOST_CHECK_CLOSE(cost, minimum, 1e-9);
        }
}

BOOST_AUTO_TEST_CASE(LogicalSwapRoutingModel)
{
    using Dummy = Simulators::MPSDummySimulator;
    using CF = Circuits::CircuitFactory<>;
    using Kind = Circuits::QuantumGateType;
    using Op = std::shared_ptr<Circuits::IOperation<>>;
    const auto swap = CF::CreateGate(Kind::kSwapGateType, 0, 5);
    for (bool mpo : {false, true})
    {
        Dummy planner(6, mpo);
        planner.SetMaxBondDimension(16);
        planner.SetCurrentBondDimensions({2, 4, 4, 4, 2});
        const auto bonds = planner.getCurrentBondDimensions();
        const auto map = planner.getQubitsMap();

        // Changing backend semantics must invalidate a previously cached cost.
        double physicalCost = std::numeric_limits<double>::infinity();
        planner.FindBestMeetingPosition({swap}, 0, 0, 0, 0., physicalCost);
        BOOST_CHECK_GT(physicalCost, 0.);
        planner.SetSwapGatesRelabelQubits(true);
        for (int heuristic : {0, 3})
        {
            double cost = std::numeric_limits<double>::infinity();
            BOOST_CHECK_EQUAL(planner.FindBestMeetingPosition({swap}, 0, 3, heuristic, 0., cost), -1);
            BOOST_CHECK_EQUAL(cost, 0.);
        }
        BOOST_CHECK(planner.getQubitsMap() == map);
        BOOST_CHECK(planner.getCurrentBondDimensions() == bonds);
        auto relabeled = planner.Clone();
        relabeled->setTotalSwappingCost(17.);
        relabeled->ApplyGate(swap);
        QC::Gates::SwapGate<> nativeSwap;
        relabeled->ApplyGate(nativeSwap, 1, 3);
        relabeled->ApplyGate(std::make_shared<Simulators::FusionRoutingOperation>(Types::qubits_vector{2, 4}, 4, true));
        BOOST_CHECK(relabeled->getQubitsMap() == std::vector<long long>({5, 3, 4, 1, 2, 0}));
        BOOST_CHECK(relabeled->getCurrentBondDimensions() == bonds);
        BOOST_CHECK_EQUAL(relabeled->getTotalSwappingCost(), 17.);
        for (size_t q = 0; q < map.size(); ++q)
            BOOST_CHECK_EQUAL(relabeled->getQubitsMapInv()[relabeled->getQubitsMap()[q]], q);
        relabeled->SwapQubitsToPosition(0, 5, 2);
        BOOST_CHECK_GT(relabeled->getTotalSwappingCost(), 17.); // Physical routing still costs work.

        const std::vector<Op> operations{CF::CreateGate(Kind::kCXGateType, 0, 5), CF::CreateGate(Kind::kSwapGateType, 1, 5),
                                         CF::CreateGate(Kind::kSwapGateType, 2, 4), CF::CreateGate(Kind::kCXGateType, 0, 2),
                                         CF::CreateGate(Kind::kCYGateType, 5, 4)};
        // Exhaustively enumerate actual tensor routes, independently exchanging
        // the logical labels for SWAPs. Check both exact and heuristic windows.
        std::function<double(const Dummy &, size_t)> enumerate = [&](const Dummy &state, size_t index) -> double {
            if (index == operations.size())
                return state.getTotalSwappingCost();
            const auto qs = operations[index]->AffectedQubits();
            if (index == 1 || index == 2)
            {
                auto next = state.Clone();
                auto nextMap = next->getQubitsMap();
                std::swap(nextMap[qs[0]], nextMap[qs[1]]);
                next->SetRoutingMap(nextMap);
                return enumerate(*next, index + 1);
            }
            const auto &positions = state.getQubitsMap();
            double best = std::numeric_limits<double>::infinity();
            for (long long p = std::min(positions[qs[0]], positions[qs[1]]); p < std::max(positions[qs[0]], positions[qs[1]]); ++p)
            {
                auto next = state.Clone();
                next->SwapQubitsToPosition(qs[0], qs[1], p);
                next->ApplyGate(operations[index]);
                best = std::min(best, enumerate(*next, index + 1));
            }
            return best;
        };
        const double minimum = enumerate(planner, 0);
        for (int heuristic : {0, 4})
        {
            double cost = std::numeric_limits<double>::infinity();
            const auto position = planner.FindBestMeetingPosition(operations, 0, 4, heuristic, 0., cost);
            BOOST_CHECK_GE(position, 0);
            BOOST_CHECK_LT(position, 5);
            if (heuristic == 0)
                BOOST_CHECK_CLOSE(cost, minimum, 1e-9);
            else
            {
                BOOST_CHECK_GE(cost + 1e-9, minimum);
                const std::vector<Op> rewritten{CF::CreateGate(Kind::kCXGateType, 0, 5), CF::CreateGate(Kind::kCXGateType, 0, 4),
                                                CF::CreateGate(Kind::kCYGateType, 1, 2)};
                double reference = std::numeric_limits<double>::infinity();
                planner.FindBestMeetingPosition(rewritten, 0, 2, 2, 0., reference);
                BOOST_CHECK_CLOSE(cost, reference, 1e-9);
            }
            BOOST_CHECK(planner.getQubitsMap() == map);
            BOOST_CHECK(planner.getCurrentBondDimensions() == bonds);
        }

        const std::vector<Op> relabelThenGate{swap, CF::CreateGate(Kind::kCXGateType, 0, 1)};
        const std::vector<Op> equivalent{CF::CreateGate(Kind::kCXGateType, 5, 1)};
        // A relabel followed by an interaction has the same cost and initial
        // layout as the interaction on the corresponding original labels.
        for (int heuristic : {0, 1})
        {
            double cost = std::numeric_limits<double>::infinity(), reference = std::numeric_limits<double>::infinity();
            planner.FindBestMeetingPosition(relabelThenGate, 0, 1, heuristic, 0., cost);
            planner.FindBestMeetingPosition(equivalent, 0, 0, 0, 0., reference);
            BOOST_CHECK_CLOSE(cost, reference, 1e-9);
        }
        BOOST_CHECK(planner.ComputeOptimalQubitsMap({CF::CreateCircuit(relabelThenGate)}) == planner.ComputeOptimalQubitsMap({CF::CreateCircuit(equivalent)}));
        planner.SetSwapGatesRelabelQubits(false);
        double restored = std::numeric_limits<double>::infinity();
        planner.FindBestMeetingPosition({swap}, 0, 0, 0, 0., restored);
        BOOST_CHECK_CLOSE(restored, physicalCost, 1e-9);
    }
}

// Explicit opt-in: this test performs large live simulations and reports
// runtime and error tradeoffs; it is not part of the default unit-test suite.
BOOST_AUTO_TEST_CASE(VirtualRoutingLiveBenchmark, *boost::unit_test::disabled() * boost::unit_test::label("benchmark"))
{
    // Fixed 64-bit seed constants (replacing std::hash<std::string>) and portable double conversion
    // so circuit parameters and observable permutations are bit-for-bit identical across platforms.
    constexpr uint64_t kQftFamilySeed = 0x22c070c2f5ccbd65ULL;
    constexpr uint64_t kQftObsSeed = 0x31f87d007a853846ULL;
    constexpr uint64_t kQaoaFamilySeed = 0xb36d2071152d0104ULL;
    auto nextAngle = [](std::mt19937_64 &rng) -> double {
        return (static_cast<double>(rng()) / (static_cast<double>(std::mt19937_64::max()) + 1.0)) * (2.0 * M_PI);
    };

    auto makeQftCircuit = [&](size_t n) {
        auto circ = std::make_shared<Circuits::Circuit<>>();
        std::mt19937_64 rng(kQftFamilySeed ^ (static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ULL));
        for (size_t q = 0; q < n; ++q)
        {
            const double theta = nextAngle(rng);
            const double phi = nextAngle(rng);
            const double lambda = nextAngle(rng);
            circ->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kUGateType, q, 0, 0, theta, phi, lambda, 0.0));
        }
        for (long long i = static_cast<long long>(n) - 1; i >= 0; --i)
        {
            circ->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kHadamardGateType, static_cast<Types::qubit_t>(i)));
            for (long long j = i - 1; j >= 0; --j)
            {
                const double angle = M_PI / static_cast<double>(1ULL << std::min<long long>(i - j, 62));
                circ->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kCPGateType, static_cast<Types::qubit_t>(j),
                                                                          static_cast<Types::qubit_t>(i), 0, angle));
            }
        }
        for (size_t i = 0; i < n / 2; ++i)
            circ->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kSwapGateType, i, n - 1 - i));
        return circ;
    };

    auto makeBenchmarkObservables = [&](size_t n) {
        std::vector<std::string> obs;
        obs.reserve(3 * n - 1);
        for (size_t q = 0; q < n; ++q)
        {
            std::string z(n, 'I');
            z[q] = 'Z';
            obs.push_back(std::move(z));
        }
        for (size_t q = 0; q + 1 < n; ++q)
        {
            std::string zz(n, 'I');
            zz[q] = 'Z';
            zz[q + 1] = 'Z';
            obs.push_back(std::move(zz));
        }
        std::mt19937_64 rng(kQftObsSeed ^ (static_cast<uint64_t>(n) * 0x9E3779B97F4A7C15ULL));
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        for (size_t k = 0; k < n; ++k)
        {
            std::string s(n, 'I');
            for (size_t i = n - 1; i > 0; --i)
                std::swap(perm[i], perm[rng() % (i + 1)]);
            for (size_t w = 0; w < std::min<size_t>(4, n); ++w)
                s[perm[w]] = "XYZ"[rng() % 3];
            obs.push_back(std::move(s));
        }
        return obs;
    };

    auto makeQaoaCircuit = [&](size_t numQ) {
        auto qaoaCirc = std::make_shared<Circuits::Circuit<>>();
        std::mt19937_64 rng(kQaoaFamilySeed ^ (static_cast<uint64_t>(numQ) * 0x9E3779B97F4A7C15ULL));
        for (size_t q = 0; q < numQ; ++q)
            qaoaCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kHadamardGateType, q));
        for (int layer = 0; layer < 3; ++layer)
        {
            const double gamma = nextAngle(rng);
            const double beta = nextAngle(rng);
            for (size_t i = 0; i < numQ; ++i)
            {
                const size_t j = (i + 1) % numQ;
                qaoaCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kCXGateType, i, j));
                qaoaCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kRzGateType, j, 0, 0, gamma));
                qaoaCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kCXGateType, i, j));
            }
            for (size_t q = 0; q < numQ; ++q)
                qaoaCirc->AddOperation(Circuits::CircuitFactory<>::CreateGate(Circuits::QuantumGateType::kRxGateType, q, 0, 0, beta));
        }
        return qaoaCirc;
    };

    struct SameChiBenchmarkCase
    {
        size_t numQubits;
        Simulators::SimulationType simType;
        bool isQaoa;
        const char *label;
    };

    double baselineRms = 0., virtualRms = 0.;
    for (const auto &tc : std::vector<SameChiBenchmarkCase>{{20, Simulators::SimulationType::kMatrixProductState, false, "MPS QFT-20"},
                                                            {24, Simulators::SimulationType::kMatrixProductState, false, "MPS QFT-24"},
                                                            {10, Simulators::SimulationType::kMatrixProductOperator, false, "MPO QFT-10"},
                                                            {12, Simulators::SimulationType::kMatrixProductOperator, true, "MPO QAOA-12"}})
    {
        const bool isMpo = (tc.simType == Simulators::SimulationType::kMatrixProductOperator);
        const auto rawCirc = tc.isQaoa ? makeQaoaCircuit(tc.numQubits) : makeQftCircuit(tc.numQubits);
        const auto testObs = makeBenchmarkObservables(tc.numQubits);

        // High-accuracy reference at chi = 256
        auto netRef = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(tc.numQubits)},
                                                                             std::vector<size_t>{tc.numQubits});
        netRef->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, Simulators::SimulationType::kMatrixProductState);
        netRef->CreateSimulator();
        netRef->Configure("matrix_product_state_max_bond_dimension", "256");
        netRef->Configure("matrix_product_state_truncation_threshold", "1e-16");
        const auto valsRef = netRef->ExecuteOnHostExpectations(std::make_shared<Circuits::Circuit<>>(*rawCirc), 0, testObs);
        BOOST_REQUIRE_EQUAL(valsRef.size(), testObs.size());
        for (double v : valsRef)
            BOOST_REQUIRE(std::isfinite(v));

        auto runLiveChi64 = [&](bool virtualRouting, double &execMsOut, double &maxErrOut, double &rmsErrOut) {
            auto netChi64 = std::make_shared<Network::SimpleDisconnectedNetwork<>>(std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(tc.numQubits)},
                                                                                   std::vector<size_t>{tc.numQubits});
            netChi64->RemoveAllOptimizationSimulatorsAndAdd(Simulators::SimulatorType::kQCSim, tc.simType);
            netChi64->CreateSimulator();
            netChi64->Configure("matrix_product_state_max_bond_dimension", "64");
            netChi64->Configure("matrix_product_operator_max_bond_dimension", "64");
            netChi64->Configure("matrix_product_state_truncation_threshold", "1e-16");
            netChi64->Configure("matrix_product_operator_truncation_threshold", "1e-16");
            netChi64->Configure("mps_routing_virtual_bonds", virtualRouting ? "true" : "false");
            const auto tStart = std::chrono::steady_clock::now();
            const auto valsChi64 = netChi64->ExecuteOnHostExpectations(std::make_shared<Circuits::Circuit<>>(*rawCirc), 0, testObs);
            execMsOut = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tStart).count();
            BOOST_REQUIRE_EQUAL(valsChi64.size(), testObs.size());

            maxErrOut = 0.0;
            double sqSum = 0.0;
            for (size_t i = 0; i < testObs.size(); ++i)
            {
                BOOST_REQUIRE(std::isfinite(valsChi64[i]));
                const double diff = std::abs(valsChi64[i] - valsRef[i]);
                maxErrOut = std::max(maxErrOut, diff);
                sqSum += diff * diff;
            }
            rmsErrOut = std::sqrt(sqSum / static_cast<double>(testObs.size()));
        };

        double execMsPhysical = 0.0, maxErrPhysical = 0.0, rmsErrPhysical = 0.0;
        double execMsChanged = 0.0, maxErrChanged = 0.0, rmsErrChanged = 0.0;
        runLiveChi64(false, execMsPhysical, maxErrPhysical, rmsErrPhysical);
        runLiveChi64(true, execMsChanged, maxErrChanged, rmsErrChanged);

        BOOST_TEST_MESSAGE("[" << tc.label << " at physical chi=64] "
                               << "totalExec(physical->virtual)=" << execMsPhysical << "->" << execMsChanged << " ms | "
                               << "maxErr(physical->virtual)=" << maxErrPhysical << "->" << maxErrChanged << " | "
                               << "rmsErr(physical->virtual)=" << rmsErrPhysical << "->" << rmsErrChanged);
        baselineRms += rmsErrPhysical;
        virtualRms += rmsErrChanged;
    }
    // A 5% aggregate budget tolerates different routing choices on individual
    // circuits. Per-case maximum error and wall-clock time remain visible above.
    BOOST_CHECK_LE(virtualRms, 1.05 * baselineRms + 1e-8);
}

BOOST_AUTO_TEST_SUITE_END()
