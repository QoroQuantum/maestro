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
#include "../Network/SimpleDisconnectedNetwork.h"

extern bool checkClose(std::complex<double> a, std::complex<double> b,
                       double dif);

BOOST_AUTO_TEST_SUITE(MPSSwapOptTests)

constexpr std::array numGates{10,  20,  30,  50,  80,  100, 150,
                              200, 270, 340, 500, 800, 1000};

BOOST_DATA_TEST_CASE(ConvertForCuttingLayersRoundtripMatchesStatevector,
                     numGates, nrGates) {
  constexpr int nrQubits = 10;
  const size_t nrStates = 1ULL << nrQubits;

  // build a random circuit with 1- and 2-qubit gates
  auto randomCirc = std::make_shared<Circuits::Circuit<>>();

  std::mt19937 g(std::random_device{}());
  std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

  for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);
    auto q1 = qubits[0];
    auto q2 = qubits[1];

    const double param1 = dblDist(g);
    const double param2 = dblDist(g);
    const double param3 = dblDist(g);
    const double param4 = dblDist(g);

    Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(gateDist(g));

    auto theGate = Circuits::CircuitFactory<>::CreateGate(
        gateType, q1, q2, 0, param1, param2, param3, param4);
    randomCirc->AddOperation(theGate);
  }

  // --- Reference: execute the original circuit on a statevector simulator ---
  auto svSim = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
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
  auto mpsSim = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  mpsSim->AllocateQubits(nrQubits);
  mpsSim->Initialize();

  Circuits::OperationState stateMPS;
  stateMPS.AllocateBits(nrQubits);
  finalCirc->Execute(mpsSim, stateMPS);

  // --- Compare amplitudes ---
  for (size_t s = 0; s < nrStates; ++s) {
    const auto aSV = svSim->Amplitude(s);
    const auto aMPS = mpsSim->Amplitude(s);
    BOOST_CHECK_PREDICATE(
        checkClose,
        (aSV)(aMPS)(1e-5));  // no bond dim limit, expect close match
  }
}

BOOST_DATA_TEST_CASE(OptimalQubitsMapZeroSwapCost, numGates, nrGates) {
  constexpr int nrQubits = 10;

  auto randomCirc = std::make_shared<Circuits::Circuit<>>();

  std::mt19937 g(
      /*static_cast<unsigned>(nrGates) * 12345u*/ std::random_device{}());
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

  for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);
    auto q1 = qubits[0];
    auto q2 = qubits[1];

    Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(gateDist(g));

    // parameters are not important for this test, we just need to have some
    // gates that affect the qubits
    // neither is the type of the gate, beyond the number of qubits it affects
    auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0,
                                                          0.0, 0.0, 0.0, 0.0);
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

  const double improvementPct =
      origSwappingCost > 0 ? (1.0 - optSwappingCost / origSwappingCost) * 100.0
                           : 0.0;
  BOOST_TEST_MESSAGE("Swap cost reduction: " << improvementPct << "%");
  BOOST_TEST_MESSAGE("Speedup ratio: "
                     << (optSwappingCost > 0
                             ? origSwappingCost / optSwappingCost
                             : 0.0)
                     << "x");
}

BOOST_DATA_TEST_CASE(OptimalQubitsMapSimulationMatch, numGates, nrGates) {
  constexpr int nrQubits = 12;
  const size_t nrStates = 1ULL << nrQubits;

  // build a random circuit with 1- and 2-qubit gates
  auto randomCirc = std::make_shared<Circuits::Circuit<>>();

  std::mt19937 g(
      /*static_cast<unsigned>(nrGates) * 12345u*/ std::random_device{}());
  std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

  for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);
    auto q1 = qubits[0];
    auto q2 = qubits[1];

    const double param1 = dblDist(g);
    const double param2 = dblDist(g);
    const double param3 = dblDist(g);
    const double param4 = dblDist(g);

    Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(gateDist(g));

    auto theGate = Circuits::CircuitFactory<>::CreateGate(
        gateType, q1, q2, 0, param1, param2, param3, param4);
    randomCirc->AddOperation(theGate);
  }

  auto startOpt = std::chrono::system_clock::now();

  // compute the optimal qubits map using the dummy simulator
  const auto layers = randomCirc->ToMultipleQubitsLayers();

  // display some stats
  BOOST_TEST_MESSAGE("Number of 2-qubit gate layers: " << layers.size());
  double avgTwoQubitGatesPerLayer = 0.0;
  for (const auto& layer : layers) {
    int twoQubitGates = 0;
    for (const auto& op : layer->GetOperations()) {
      if (op->AffectedQubits().size() >= 2) {
        ++twoQubitGates;
      }
    }
    avgTwoQubitGatesPerLayer += twoQubitGates;
  }
  avgTwoQubitGatesPerLayer /= layers.size();
  BOOST_TEST_MESSAGE("Average number of 2-qubit gates per layer: "
                     << avgTwoQubitGatesPerLayer);

  Simulators::MPSDummySimulator dummySim(nrQubits);
  dummySim.SetMaxBondDimension(64);

  const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);
  auto endOpt = std::chrono::system_clock::now();
  double optMapTime =
      std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;
  BOOST_TEST_MESSAGE("Optimization time: " << optMapTime << " ms");

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
  double origTime =
      std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;

  // execute the same circuit with the optimized qubits map
  auto qcsimOpt = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  qcsimOpt->AllocateQubits(nrQubits);
  qcsimOpt->Configure("matrix_product_state_max_bond_dimension", "64");
  qcsimOpt->Initialize();

  // qcsimOpt->SetUseOptimalMeetingPosition(true);

  qcsimOpt->SetInitialQubitsMap(
      std::vector<long long int>(optimalMap.begin(), optimalMap.end()));

  Circuits::OperationState stateOpt;
  stateOpt.AllocateBits(nrQubits);

  const auto optCirc = randomCirc->LayersToCircuit(layers);

  auto startExecOpt = std::chrono::system_clock::now();
  optCirc->Execute(qcsimOpt, stateOpt);
  auto endExecOpt = std::chrono::system_clock::now();
  double optTime =
      std::chrono::duration<double>(endExecOpt - startExecOpt).count() * 1000.;

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

  const double improvementPct =
      origSwappingCost > 0 ? (1.0 - optSwappingCost / origSwappingCost) * 100.0
                           : 0.0;
  BOOST_TEST_MESSAGE("Swap cost reduction: " << improvementPct << "%");
  BOOST_TEST_MESSAGE("Speedup ratio: "
                     << (optSwappingCost > 0
                             ? origSwappingCost / optSwappingCost
                             : 0.0)
                     << "x");

  // compare the amplitudes of the two simulations
  for (size_t s = 0; s < nrStates; ++s) {
    const auto aOrig = qcsimOrig->Amplitude(s);
    const auto aOpt = qcsimOpt->Amplitude(s);
    BOOST_CHECK_PREDICATE(
        checkClose, (aOrig)(aOpt)(0.05));  // since max bond dimension is set,
                                           // they won't match so well
  }
}

BOOST_DATA_TEST_CASE(LookaheadSwapOptimization, numGates, nrGates) {
  constexpr int nrQubits = 10;

  auto randomCirc = std::make_shared<Circuits::Circuit<>>();

  std::mt19937 g(
      /*static_cast<unsigned>(nrGates) * 99999u*/ std::random_device{}());
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

  for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);
    auto q1 = qubits[0];
    auto q2 = qubits[1];

    Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(gateDist(g));

    auto theGate = Circuits::CircuitFactory<>::CreateGate(gateType, q1, q2, 0,
                                                          0.0, 0.0, 0.0, 0.0);
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

  const auto& ops = optCirc->GetOperations();
  const auto opsVec = std::vector<std::shared_ptr<Circuits::IOperation<>>>(
      ops.begin(), ops.end());

  dummySim.SetInitialQubitsMap(optimalMap);

  int lookaheadDepth = 30;
  int lookaheadDepthWithHeuristic = 27;

  for (size_t i = 0; i < opsVec.size(); ++i) {
    const auto qbits = opsVec[i]->AffectedQubits();
    if (qbits.size() < 2) {
      continue;
    }

    auto q1 = static_cast<long long int>(qbits[0]);
    auto q2 = static_cast<long long int>(qbits[1]);

    // Build lookahead window from remaining ops, counting only 2-qubit
    // gates towards the depth budget so the effective lookahead is accurate
    std::vector<std::shared_ptr<Circuits::IOperation<>>> window;
    window.reserve(lookaheadDepth);
    int twoQubitCount = 0;
    for (size_t j = i; j < opsVec.size() && twoQubitCount < lookaheadDepth;
         ++j) {
      window.push_back(opsVec[j]);
      if (opsVec[j]->AffectedQubits().size() >= 2) ++twoQubitCount;
    }

    double meetBestCost = std::numeric_limits<double>::infinity();
    const auto meetPos = dummySim.FindBestMeetingPosition(
        window, 0, lookaheadDepth, lookaheadDepthWithHeuristic, 0.0,
        meetBestCost);

    dummySim.SwapQubitsToPosition(q1, q2, meetPos);
    dummySim.ApplyGate(opsVec[i]);
  }

  const auto lookaheadCost = dummySim.getTotalSwappingCost();

  BOOST_TEST_MESSAGE("Heuristic swap cost: " << heuristicCost);
  BOOST_TEST_MESSAGE("Lookahead (depth " << lookaheadDepth
                                         << ") swap cost: " << lookaheadCost);

  const double improvementPct =
      heuristicCost > 0 ? (1.0 - lookaheadCost / heuristicCost) * 100.0 : 0.0;
  BOOST_TEST_MESSAGE("Lookahead improvement: " << improvementPct << "%");

  // Lookahead should generally be no worse than heuristic, but due to the
  // limited search window and greedy per-gate decisions it can occasionally
  // produce a marginally higher cost.  Allow a small tolerance (10%).
  const double tolerance = 1.10;
  BOOST_CHECK_LE(lookaheadCost, heuristicCost * tolerance);
}

BOOST_DATA_TEST_CASE(OptimalMeetingPositionSimulationMatch, numGates, nrGates) {
  constexpr int nrQubits = 12;
  const size_t nrStates = 1ULL << nrQubits;

  auto randomCirc = std::make_shared<Circuits::Circuit<>>();

  std::mt19937 g(
      /*static_cast<unsigned>(nrGates) * 77777u*/ std::random_device{}());
  std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

  for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);
    auto q1 = qubits[0];
    auto q2 = qubits[1];

    const double param1 = dblDist(g);
    const double param2 = dblDist(g);
    const double param3 = dblDist(g);
    const double param4 = dblDist(g);

    Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(gateDist(g));

    auto theGate = Circuits::CircuitFactory<>::CreateGate(
        gateType, q1, q2, 0, param1, param2, param3, param4);
    randomCirc->AddOperation(theGate);
  }

  // Execute with default heuristic
  auto qcsimOrig = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  qcsimOrig->AllocateQubits(nrQubits);
  qcsimOrig->Configure("matrix_product_state_max_bond_dimension", "64");
  qcsimOrig->Initialize();
  qcsimOrig->SetUseOptimalMeetingPosition(false);

  Circuits::OperationState stateOrig;
  stateOrig.AllocateBits(nrQubits);

  auto startOrig = std::chrono::system_clock::now();
  randomCirc->Execute(qcsimOrig, stateOrig);
  auto endOrig = std::chrono::system_clock::now();
  double origTime =
      std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;

  // Execute with optimal meeting position enabled
  auto qcsimOpt = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  qcsimOpt->AllocateQubits(nrQubits);
  qcsimOpt->Configure("matrix_product_state_max_bond_dimension", "64");
  qcsimOpt->Initialize();
  qcsimOpt->SetUseOptimalMeetingPosition(true);

  Circuits::OperationState stateOpt;
  stateOpt.AllocateBits(nrQubits);

  auto startOpt = std::chrono::system_clock::now();
  randomCirc->Execute(qcsimOpt, stateOpt);
  auto endOpt = std::chrono::system_clock::now();
  double optTime =
      std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

  BOOST_TEST_MESSAGE("Default heuristic execution time: " << origTime << " ms");
  BOOST_TEST_MESSAGE("Optimal meeting position execution time: " << optTime
                                                                 << " ms");
  BOOST_TEST_MESSAGE("Optimized is " << origTime / optTime << "x vs default");

  // Both executions should produce the same amplitudes
  // (swap route may differ but both are valid)
  for (size_t s = 0; s < nrStates; ++s) {
    const auto aOrig = qcsimOrig->Amplitude(s);
    const auto aOpt = qcsimOpt->Amplitude(s);
    BOOST_CHECK_PREDICATE(checkClose, (aOrig)(aOpt)(0.05));
  }
}

BOOST_DATA_TEST_CASE(WindowOptimizedVsOriginalSimulation, numGates, nrGates) {
#define MAX_BOND_DIMENSION 64
  constexpr int nrQubits = 12;
  const size_t nrStates = 1ULL << nrQubits;

  // build a random circuit with 1- and 2-qubit gates
  auto randomCirc = std::make_shared<Circuits::Circuit<>>();

  std::mt19937 g(
      /*static_cast<unsigned>(nrGates) * 31415u*/ std::random_device{}());
  std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));

  for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);
    auto q1 = qubits[0];
    auto q2 = qubits[1];

    const double param1 = dblDist(g);
    const double param2 = dblDist(g);
    const double param3 = dblDist(g);
    const double param4 = dblDist(g);

    Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(gateDist(g));

    auto theGate = Circuits::CircuitFactory<>::CreateGate(
        gateType, q1, q2, 0, param1, param2, param3, param4);
    randomCirc->AddOperation(theGate);
  }

  // --- Original execution: no optimizations, identity qubit map ---
  auto qcsimOrig = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  qcsimOrig->AllocateQubits(nrQubits);
  qcsimOrig->Configure("matrix_product_state_max_bond_dimension",
                       std::to_string(MAX_BOND_DIMENSION).c_str());
  qcsimOrig->Initialize();

  Circuits::OperationState stateOrig;
  stateOrig.AllocateBits(nrQubits);

  auto startOrig = std::chrono::system_clock::now();
  randomCirc->Execute(qcsimOrig, stateOrig);
  auto endOrig = std::chrono::system_clock::now();
  double origTime =
      std::chrono::duration<double>(endOrig - startOrig).count() * 1000.;

  // --- Window-optimized execution ---
  // Compute optimal initial qubit mapping via the dummy simulator
  auto startOptMap = std::chrono::system_clock::now();
  const auto layers = randomCirc->ToMultipleQubitsLayers();

  // display some stats
  BOOST_TEST_MESSAGE("Number of 2-qubit gate layers: " << layers.size());

  double avgTwoQubitGatesPerLayer = 0.0;
  for (const auto& layer : layers) {
    int twoQubitGates = 0;
    for (const auto& op : layer->GetOperations()) {
      if (op->AffectedQubits().size() >= 2) {
        ++twoQubitGates;
      }
    }
    avgTwoQubitGatesPerLayer += twoQubitGates;
  }
  avgTwoQubitGatesPerLayer /= layers.size();
  BOOST_TEST_MESSAGE("Average number of 2-qubit gates per layer: "
                     << avgTwoQubitGatesPerLayer);

  Simulators::MPSDummySimulator dummySim(nrQubits);
  dummySim.SetMaxBondDimension(MAX_BOND_DIMENSION);
  const auto optimalMap = dummySim.ComputeOptimalQubitsMap(layers);

  const auto optCirc = randomCirc->LayersToCircuit(layers);
  auto endOptMap = std::chrono::system_clock::now();
  double optMapTime =
      std::chrono::duration<double>(endOptMap - startOptMap).count() * 1000.;
  BOOST_TEST_MESSAGE("Optimization (map + layers) time: " << optMapTime
                                                          << " ms");

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

  // Create the optimized simulator with optimal map + meeting position +
  // lookahead
  int lookaheadVal = nrQubits;
  if (nrQubits > 15) lookaheadVal = 15;

  int lookaheadDepth = layers.size() < 10   ? 0
                       : layers.size() < 20 ? static_cast<int>(lookaheadVal)
                       : layers.size() < 35 ? 1.5 * lookaheadVal
                                            : 2 * lookaheadVal;

  int lookaheadHeuristicDepth = layers.size() < 10   ? 0
                                : layers.size() < 20 ? lookaheadDepth - 1
                                                     : lookaheadDepth - 2;

  if (nrQubits < 10 || MAX_BOND_DIMENSION < 32) {
    lookaheadDepth = 0;
    lookaheadHeuristicDepth = 0;
  }

  auto qcsimOpt = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  qcsimOpt->AllocateQubits(nrQubits);
  qcsimOpt->Configure("matrix_product_state_max_bond_dimension",
                      std::to_string(MAX_BOND_DIMENSION).c_str());
  qcsimOpt->Initialize();

  qcsimOpt->SetInitialQubitsMap(
      std::vector<long long int>(optimalMap.begin(), optimalMap.end()));

  qcsimOpt->SetUseOptimalMeetingPosition(true);
  qcsimOpt->SetLookaheadDepth(lookaheadDepth);
  qcsimOpt->SetLookaheadDepthWithHeuristic(lookaheadHeuristicDepth);
  qcsimOpt->SetUpcomingGates(optCirc->GetOperations());

  Circuits::OperationState stateOpt;
  stateOpt.AllocateBits(nrQubits);

  auto startExecOpt = std::chrono::system_clock::now();
  optCirc->Execute(qcsimOpt, stateOpt);
  auto endExecOpt = std::chrono::system_clock::now();
  double optTime =
      std::chrono::duration<double>(endExecOpt - startExecOpt).count() * 1000.;

  // --- Runtime comparison ---
  BOOST_TEST_MESSAGE("Original execution time: " << origTime << " ms");
  BOOST_TEST_MESSAGE("Window-optimized execution time: " << optTime << " ms");
  BOOST_TEST_MESSAGE("Window-optimized is " << origTime / optTime
                                            << "x vs original");

  // --- Amplitude comparison ---
  for (size_t s = 0; s < nrStates; ++s) {
    const auto aOrig = qcsimOrig->Amplitude(s);
    const auto aOpt = qcsimOpt->Amplitude(s);
    BOOST_CHECK_PREDICATE(checkClose,
                          (aOrig)(aOpt)(0.05));  // max bond dimension is set,
                                                 // tolerance needed
  }
}

constexpr std::array numGatesForNetworkExecution{10, 20, 30, 50, 80, 100};

BOOST_DATA_TEST_CASE(NetworkOptimizedSwapsVsUnoptimizedOnHost,
                     numGatesForNetworkExecution, nrGates) {
  constexpr int nrQubits = 8;
  constexpr size_t shots = 3000;

  // build a random circuit with 1- and 2-qubit gates and mid-circuit
  // measurements followed by conditional gates
  auto randomCirc = std::make_shared<Circuits::Circuit<>>();

  std::mt19937 g(std::random_device{}());
  std::uniform_real_distribution<double> dblDist(-2. * M_PI, 2. * M_PI);
  std::uniform_int_distribution<int> gateDist(
      0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
  std::uniform_int_distribution<int> qubitDist(0, nrQubits - 1);
  std::bernoulli_distribution measureCoin(0.05);

  for (int gateNr = 0; gateNr < nrGates; ++gateNr) {
    // occasionally insert a mid-circuit measurement + conditional X
    if (gateNr > 0 && measureCoin(g)) {
      const int mq = qubitDist(g);
      const size_t cbit = static_cast<size_t>(mq);
      randomCirc->AddOperation(Circuits::CircuitFactory<>::CreateMeasurement(
          std::vector<std::pair<Types::qubit_t, size_t>>{
              {static_cast<Types::qubit_t>(mq), cbit}}));

      // conditional X on a different qubit driven by the measurement result
      int tgt = (mq + 1) % nrQubits;
      randomCirc->AddOperation(
          Circuits::CircuitFactory<>::CreateConditionalGate(
              Circuits::CircuitFactory<>::CreateGate(
                  Circuits::QuantumGateType::kXGateType, tgt),
              Circuits::CircuitFactory<>::CreateEqualCondition(
                  std::vector<size_t>{cbit}, std::vector<bool>{true})));

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

    Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(gateDist(g));

    auto theGate = Circuits::CircuitFactory<>::CreateGate(
        gateType, q1, q2, q3, param1, param2, param3, param4);
    randomCirc->AddOperation(theGate);
  }

  // final measurements on all qubits
  std::vector<std::pair<Types::qubit_t, size_t>> finalMeas;
  for (int q = 0; q < nrQubits; ++q)
    finalMeas.emplace_back(static_cast<Types::qubit_t>(q),
                           static_cast<size_t>(q));
  randomCirc->AddOperation(
      Circuits::CircuitFactory<>::CreateMeasurement(finalMeas));

  // Unoptimized network (swaps optimization disabled)
  auto netUnopt = std::make_shared<Network::SimpleDisconnectedNetwork<>>(
      std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
      std::vector<size_t>{static_cast<size_t>(nrQubits)});
  netUnopt->RemoveAllOptimizationSimulatorsAndAdd(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  netUnopt->CreateSimulator();
  netUnopt->Configure("matrix_product_state_max_bond_dimension", "64");
  netUnopt->SetInitialQubitsMapOptimization(false);
  netUnopt->SetMPSOptimizeSwaps(false);

  auto startUnopt = std::chrono::system_clock::now();
  const auto resUnopt = netUnopt->RepeatedExecuteOnHost(randomCirc, 0, shots);
  auto endUnopt = std::chrono::system_clock::now();
  double unoptTime =
      std::chrono::duration<double>(endUnopt - startUnopt).count() * 1000.;

  // Optimized network (swaps optimization enabled, the default)
  auto netOpt = std::make_shared<Network::SimpleDisconnectedNetwork<>>(
      std::vector<Types::qubit_t>{static_cast<Types::qubit_t>(nrQubits)},
      std::vector<size_t>{static_cast<size_t>(nrQubits)});
  netOpt->RemoveAllOptimizationSimulatorsAndAdd(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kMatrixProductState);
  netOpt->CreateSimulator();
  netOpt->Configure("matrix_product_state_max_bond_dimension", "64");
  netOpt->SetInitialQubitsMapOptimization(true);
  netOpt->SetMPSOptimizeSwaps(true);

  auto startOpt = std::chrono::system_clock::now();
  const auto resOpt = netOpt->RepeatedExecuteOnHost(randomCirc, 0, shots);
  auto endOpt = std::chrono::system_clock::now();
  double optTime =
      std::chrono::duration<double>(endOpt - startOpt).count() * 1000.;

  BOOST_TEST_MESSAGE("Unoptimized execution time: " << unoptTime << " ms");
  BOOST_TEST_MESSAGE("Optimized execution time: " << optTime << " ms");
  if (optTime > 0.)
    BOOST_TEST_MESSAGE("Speedup: " << unoptTime / optTime << "x");

  // Both networks must produce non-empty results
  BOOST_CHECK(!resUnopt.empty());
  BOOST_CHECK(!resOpt.empty());

  // Compare measurement distributions.
  std::set<std::vector<bool>> allKeys;
  for (const auto& kv : resUnopt) allKeys.insert(kv.first);
  for (const auto& kv : resOpt) allKeys.insert(kv.first);

  double totalVariation = 0.0;
  for (const auto& key : allKeys) {
    const double pUnopt = (resUnopt.count(key) ? resUnopt.at(key) : 0) /
                          static_cast<double>(shots);
    const double pOpt =
        (resOpt.count(key) ? resOpt.at(key) : 0) / static_cast<double>(shots);
    totalVariation += std::abs(pUnopt - pOpt);

    if (pUnopt > 1E-2 && pOpt > 1E-2) {
      BOOST_CHECK_CLOSE(pUnopt, pOpt,
                        (pUnopt < 0.05 || pOpt < 0.05) ? 100.
                        : (pUnopt < 0.1 || pOpt < 0.1) ? 50.
                                                       : 30.);
    }
  }
  totalVariation *= 0.5;

  BOOST_TEST_MESSAGE("Total variation distance: " << totalVariation);

  // With a bond dimension cap, some divergence is expected
  BOOST_CHECK_LT(totalVariation, 0.5);
}

BOOST_AUTO_TEST_SUITE_END()
