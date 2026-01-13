/**
 * @file gputests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Basic tests for the gpu simulator.
 *
 * Just tests for simulator creation, qubits allocation, initialization, a
 * single X gate and measurements of all qubits.
 */

#ifdef __linux__

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;

#undef min
#undef max

#include <numeric>	
#include <algorithm>
#include <array>
#include <random>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Simulators/Factory.h" 

#include "../Circuit/Circuit.h"
#include "../Circuit/Conditional.h"
#include "../Circuit/Measurements.h"
#include "../Circuit/QuantumGates.h"
#include "../Circuit/RandomOp.h"
#include "../Circuit/Reset.h"
#include "../Circuit/Factory.h"

extern bool checkClose(std::complex<double> a, std::complex<double> b,
                       double dif);

std::shared_ptr<Circuits::Circuit<>> GenerateCircuit(int nrGates,
                                                     int nrQubits) {
  std::shared_ptr<Circuits::Circuit<>> randomCirc = std::make_shared<Circuits::Circuit<>>();
  std::random_device rd;
  std::mt19937 g(rd());

  auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
  auto dblGenIter = dblGen.begin();

  auto gateGen = bdata::random(
      0, static_cast<int>(Circuits::QuantumGateType::kCCXGateType));
  auto gateGenIter = gateGen.begin();

  // TODO: Maybe insert from time to time a random number generating 'gate' and
  // a conditioned random one, those should not affect results?
  for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
    // create a random gate and add it to the circuit

    // first, pick randomly three qubits (depending on the randomly chosen gate
    // type, not all of them will be used)
    Types::qubits_vector qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);
    std::shuffle(qubits.begin(), qubits.end(), g);
    auto q1 = qubits[0];
    auto q2 = qubits[1];
    auto q3 = qubits[2];

    // now some random parameters, again, they might be ignored
    const double param1 = *dblGenIter;
    ++dblGenIter;
    const double param2 = *dblGenIter;
    ++dblGenIter;
    const double param3 = *dblGenIter;
    ++dblGenIter;
    const double param4 = *dblGenIter;
    ++dblGenIter;

    const Circuits::QuantumGateType gateType =
        static_cast<Circuits::QuantumGateType>(*gateGenIter);

    auto theGate = Circuits::CircuitFactory<>::CreateGate(
        gateType, q1, q2, q3, param1, param2, param3, param4);
    randomCirc->AddOperation(theGate);
  }
  return randomCirc;
}

BOOST_AUTO_TEST_SUITE(gpu_simulator_tests)

BOOST_AUTO_TEST_CASE(simple_test) {
  Simulators::SimulatorsFactory::InitGpuLibrary();

  std::shared_ptr<Simulators::ISimulator> gpusim;

  try {
    gpusim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kStatevector);
    if (gpusim) {
      gpusim->AllocateQubits(3);
      gpusim->Initialize();

      gpusim->ApplyX(0);

      auto res = gpusim->Measure({0, 1, 2});

      BOOST_TEST(res == 1ULL);
      BOOST_CHECK_CLOSE(gpusim->Probability(res), 1.0, 0.000001);
    }
    gpusim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kMatrixProductState);
    if (gpusim) {
      gpusim->AllocateQubits(3);
      gpusim->Initialize();

      gpusim->ApplyX(0);

      auto res = gpusim->Measure({0, 1, 2});

      BOOST_TEST(res == 1ULL);
      BOOST_CHECK_CLOSE(gpusim->Probability(res), 1.0, 0.000001);
    }
    gpusim = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kTensorNetwork);
    if (gpusim) {
      gpusim->AllocateQubits(3);
      gpusim->Initialize();

      gpusim->ApplyX(0);

      auto res = gpusim->Measure({0, 1, 2});

      BOOST_TEST(res == 1ULL);
      BOOST_CHECK_CLOSE(gpusim->Probability(res), 1.0, 0.000001);
    }
  } catch (const std::exception& e) {
    BOOST_TEST_MESSAGE(
        "Exception: "
        << e.what() << ". Please ensure the proper gpu library is available.");
  } catch (...) {
    BOOST_TEST_MESSAGE("Unknown exception");
  }
}


constexpr std::array numGates{5, 10, 20, 40};

BOOST_DATA_TEST_CASE(random_circuits_test, numGates, nGates) {
  Simulators::SimulatorsFactory::InitGpuLibrary();

  const int nrQubits = 5;
  const int nrShots = 10000;
  const double precision = 0.03;
  const double precisionSamples = 0.05;

  Circuits::OperationState state;
  state.AllocateBits(nrQubits);

  Types::qubits_vector qubits(nrQubits);
  std::iota(qubits.begin(), qubits.end(), 0);

  std::shared_ptr<Simulators::ISimulator> gpusimStatevector;
  std::shared_ptr<Simulators::ISimulator> gpusimMPS;
  std::shared_ptr<Simulators::ISimulator> gpusimTN;

  auto circuit = GenerateCircuit(nGates, nrQubits);

  try {
    gpusimStatevector = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kStatevector);
    if (gpusimStatevector) {
      gpusimStatevector->AllocateQubits(nrQubits);
      gpusimStatevector->Initialize();
      circuit->Execute(gpusimStatevector, state);
    }

    gpusimMPS = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kMatrixProductState);
    if (gpusimMPS) {
      gpusimMPS->AllocateQubits(nrQubits);
      gpusimMPS->Initialize();
      circuit->Execute(gpusimMPS, state);
    }

    gpusimTN = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kGpuSim,
        Simulators::SimulationType::kTensorNetwork);
    if (gpusimTN) {
      gpusimTN->AllocateQubits(nrQubits);
      gpusimTN->Initialize();
      circuit->Execute(gpusimTN, state);
    }

    if (gpusimStatevector == nullptr && gpusimMPS == nullptr && gpusimTN == nullptr) {
      BOOST_TEST_MESSAGE(
          "Could not create any gpu simulator. Please ensure the proper gpu "
          "library is available.");
      return;
    }

    for (int q = 0; q < nrQubits; ++q) {
      auto probStatevector = gpusimStatevector != nullptr ? gpusimStatevector->Probability(q) : 0.0;

      auto probMPS = gpusimMPS != nullptr ? gpusimMPS->Probability(q) : 0.0;
      auto probTN = gpusimTN != nullptr ? gpusimTN->Probability(q) : 0.0;

      BOOST_CHECK_PREDICATE(checkClose, (probStatevector)(probMPS)(precision));
      BOOST_CHECK_PREDICATE(checkClose, (probStatevector)(probTN)(precision));
    }

    auto resultsStatevector =
        gpusimStatevector != nullptr
            ? gpusimStatevector->SampleCounts(qubits, nrShots)
            : std::unordered_map<Types::qubit_t, Types::qubit_t>();
    auto resultsMPS =
        gpusimMPS != nullptr
            ? gpusimMPS->SampleCounts(qubits, nrShots)
            : std::unordered_map<Types::qubit_t, Types::qubit_t>();
    auto resultsTN = gpusimTN != nullptr
            ? gpusimTN->SampleCounts(qubits, nrShots)
            : std::unordered_map<Types::qubit_t, Types::qubit_t>();

    for (const auto& [outcome, count] : resultsStatevector) {
      auto itMPS = resultsMPS.find(outcome);
      
      if (itMPS != resultsMPS.end()) {
        BOOST_CHECK_PREDICATE(checkClose,
                              (static_cast<double>(count) /
                               nrShots)(static_cast<double>(itMPS->second) /
                                        nrShots)(precisionSamples));
      } else
        BOOST_TEST(false);

      auto itTN = resultsTN.find(outcome);
 
      if (itTN != resultsTN.end()) {
        BOOST_CHECK_PREDICATE(
            checkClose,
            (static_cast<double>(count) /
                               nrShots)(static_cast<double>(itMPS->second) /
                                        nrShots)(precisionSamples));
      } else
        BOOST_TEST(false);
    }
  } catch (const std::exception& e) {
    BOOST_TEST_MESSAGE(
        "Exception: "
        << e.what() << ". Please ensure the proper gpu library is available.");
  } catch (...) {
    BOOST_TEST_MESSAGE("Unknown exception");
  }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
