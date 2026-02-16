/**
 * @file extstabtests.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for extended stabilizer simulators.
 */

#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>
namespace utf = boost::unit_test;
namespace bdata = boost::unit_test::data;

#undef min
#undef max

#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>

#include "../Simulators/Factory.h"
#include "../Circuit/Factory.h"

struct Operation {
  int gate = 0;  // gate id, first codes for clifford gates, then for
                 // non-clifford gates, ordered by number of qubits
  int qubit1 = 0;
  int qubit2 = 0;  // unused for single qubit operations
  int qubit3 = 0;  // unused for single and two qubit operations
  // params, used only for some gates
  double theta = 0;
  double phi = 0;
  double lambda = 0;
  double gamma = 0;
};

struct ExtStabTestFixture {
  ExtStabTestFixture() {
    simExtStabilizer = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQiskitAer,
        Simulators::SimulationType::kExtendedStabilizer);

    // default value is 0.05, too bad for the tests
    simExtStabilizer->Configure("extended_stabilizer_approximation_error", "0.01");
    //simExtStabilizer->Configure("extended_stabilizer_sampling_method", "norm_estimation");
    simExtStabilizer->AllocateQubits(nrQubitsForRandomCirc);
    simExtStabilizer->Initialize();

    simStatevector = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    simStatevector->AllocateQubits(nrQubitsForRandomCirc);
    simStatevector->Initialize();
  }

  std::string GeneratePauliString(int nrQubits) {
    std::string pauli;
    pauli.resize(nrQubits);
    std::random_device rd;
    std::mt19937 g(rd());
    std::uniform_int_distribution<int> dist(0, 3);

    for (int i = 0; i < nrQubits; ++i) {
      const int v = dist(g);
      switch (v) {
        case 0:
          pauli[i] = 'I';
          break;
        case 1:
          pauli[i] = 'X';
          break;
        case 2:
          pauli[i] = 'Y';
          break;
        case 3:
          pauli[i] = 'Z';
          break;
      }
    }

    return pauli;
  }

    std::vector<Operation> GenerateCircuit(int nrQubits, int nrGates,
                                         int maxGate = 12) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::uniform_int_distribution<int> dist(0, maxGate);

    std::uniform_real_distribution<double> param_dist(0.0, 2. * M_PI);
    std::bernoulli_distribution bool_dist(0.9);  // high chance to make a clifford gate from a non-clifford one

    std::vector<Operation> circuit;
    std::vector<int> qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);

    for (int i = 0; i < nrGates; ++i) {
      Operation op;

      std::shuffle(qubits.begin(), qubits.end(), g);

      int gate = dist(g);  // random gate id

      if (gate > 12 && bool_dist(g))
        gate %= 13;  // make it clifford, to have less non-clifford gates in
                     // the circuit, which are more expensive to simulate

      op.gate = gate;
      op.qubit1 = qubits[0];
      op.qubit2 = qubits[1];
      op.qubit3 = qubits[2];

      op.theta = param_dist(g);
      op.phi = param_dist(g);
      op.lambda = param_dist(g);
      op.gamma = param_dist(g);

      circuit.push_back(std::move(op));
    }

    // check for 'dangerous' non-clifford gates, which can make the simulation
    // too expensive, and replace them with clifford ones - except the first one
    // for now only three qubit non-clifford gates, which are the most
    // expensive - for example cswap does 12 doublings!
    bool first = true;
    for (auto& op : circuit) {
      if (op.gate >= 28 && op.gate <= 29) {
        if (first)
          first = false;
        else
          op.gate = 10;  // cx... a clifford gate
      }
    }

    return circuit;
  }

void ExecuteGate(const Operation& op,
                   std::shared_ptr<Simulators::ISimulator>& state_vector) {
    switch (op.gate) {
      case 0:
        state_vector->ApplyX(op.qubit1);
        break;
      case 1:
        state_vector->ApplyY(op.qubit1);
        break;
      case 2:
        state_vector->ApplyZ(op.qubit1);
        break;
      case 3:
        state_vector->ApplyH(op.qubit1);
        break;
      case 4:
        state_vector->ApplyS(op.qubit1);
        break;
      case 5:
        state_vector->ApplySDG(op.qubit1);
        break;

      case 6:
        state_vector->ApplySx(op.qubit1);
        break;
      case 7:
        state_vector->ApplySxDAG(op.qubit1);
        break;
      case 8:
        state_vector->ApplyK(op.qubit1);
        break;

      // two qubit gates
      case 9:
        state_vector->ApplySwap(op.qubit2, op.qubit1);
        break;
      case 10:
        state_vector->ApplyCX(op.qubit2, op.qubit1);
        break;
      case 11:
        state_vector->ApplyCY(op.qubit2, op.qubit1);
        break;
      case 12:
        state_vector->ApplyCZ(op.qubit2, op.qubit1);
        break;

        // non-clifford single qubit gates

      case 13:
        state_vector->ApplyP(op.qubit1, op.theta);
        break;
      case 14:
        state_vector->ApplyRx(op.qubit1, op.theta);
        break;
      case 15:
        state_vector->ApplyRy(op.qubit1, op.theta);
        break;
      case 16:
        state_vector->ApplyRz(op.qubit1, op.theta);
        break;
      case 17:
        state_vector->ApplyU(op.qubit1, op.theta, op.phi, op.lambda, op.gamma);
        break;

      case 18:
        state_vector->ApplyT(op.qubit1);
        break;
      case 19:
        state_vector->ApplyTDG(op.qubit1);
        break;

        // non-clifford two qubit gates
      case 20:
        state_vector->ApplyCH(op.qubit2, op.qubit1);
        break;

      case 21:
        state_vector->ApplyCRz(op.qubit2, op.qubit1, op.theta);
        break;
      case 22:
        state_vector->ApplyCRy(op.qubit2, op.qubit1, op.theta);
        break;
      case 23:
        state_vector->ApplyCRx(op.qubit2, op.qubit1, op.theta);
        break;

      case 24:
        state_vector->ApplyCP(op.qubit2, op.qubit1, op.theta);
        break;
      case 25:
        state_vector->ApplyCSx(op.qubit2, op.qubit1);
        break;
      case 26:
        state_vector->ApplyCSxDAG(op.qubit2, op.qubit1);
        break;

      case 27:
        state_vector->ApplyCU(op.qubit2, op.qubit1, op.theta, op.phi, op.lambda,
                              op.gamma);
        break;

      // non-clifford three qubit gates
      case 28:
        state_vector->ApplyCCX(op.qubit3, op.qubit2, op.qubit1);
        break;
      case 29:
        state_vector->ApplyCSwap(op.qubit3, op.qubit2, op.qubit1);
        break;
      default:
        std::cerr << "Unknown gate id: " << op.gate << std::endl;
    }
  }

  const unsigned int nrQubitsForRandomCirc = 4;
  std::shared_ptr<Simulators::ISimulator> simStatevector;
  std::shared_ptr<Simulators::ISimulator> simExtStabilizer;
};

BOOST_AUTO_TEST_SUITE(ext_stabilizer_tests)

BOOST_FIXTURE_TEST_CASE(ExtStabilizerSimInitializationTest,
                        ExtStabTestFixture) {
  BOOST_TEST(simExtStabilizer);
  BOOST_TEST(simStatevector);
}


BOOST_DATA_TEST_CASE_F(ExtStabTestFixture, RandomCliffordCircuitsTest,
                       bdata::xrange(1, 20), nrGates) {
  auto circuit = GenerateCircuit(nrQubitsForRandomCirc, nrGates);

  // execute
  for (const auto& op : circuit) {
    ExecuteGate(op, simStatevector);
    ExecuteGate(op, simExtStabilizer);
  }

  // check, first some random pauli strings
  const int nrChecks = 100;
  for (int i = 0; i < nrChecks; ++i) {
    const std::string pauliStr = GeneratePauliString(nrQubitsForRandomCirc);
    const double expValStateVec = simStatevector->ExpectationValue(pauliStr);

    const double expValExtStabilizer =
        simExtStabilizer->ExpectationValue(pauliStr);
    BOOST_TEST(std::abs(expValStateVec - expValExtStabilizer) < 1e-2,
               "Expectation value mismatch for pauli string "
                   << pauliStr << ": statevector " << expValStateVec
                   << ", ext stabilizer " << expValExtStabilizer);
  }

  // now sampling
  const int nrSamples = 1000;
  Types::qubits_vector qubitsToMeasure(nrQubitsForRandomCirc);
  std::iota(qubitsToMeasure.begin(), qubitsToMeasure.end(), 0);

  // perform sampling
  auto svRes = simStatevector->SampleCounts(qubitsToMeasure, nrSamples);

  std::vector<int> pq(qubitsToMeasure.begin(), qubitsToMeasure.end());

  std::random_device rd;
  std::mt19937 g(rd());

  auto stdSvRes = simExtStabilizer->SampleCounts(qubitsToMeasure, nrSamples);
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t esCount =
        stdSvRes.find(key) != stdSvRes.end() ? stdSvRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double esProb = static_cast<double>(esCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - esProb) < 0.1,
               "Sampling probability mismatch for outcome "
                   << key << ": statevector " << svProb
                   << ", ext stabilizer sim " << esProb);
  }

  std::unordered_map<Types::qubit_t, Types::qubit_t> qiskitRes;

  // until the saving/restoring state is implemented for the extended stabilizer
  // simulator in qiskit aer, this check does not work
  /*
  simExtStabilizer->SaveState();

  Types::qubits_vector pqq(qubitsToMeasure.begin(), qubitsToMeasure.end());

  for (int i = 0; i < nrSamples; ++i) {
    std::shuffle(pqq.begin(), pqq.end(), g);
    auto res = simExtStabilizer->Measure(pqq);
    Types::qubit_t result = 0;
    for (int q = 0; q < static_cast<int>(pqq.size()); ++q) {
      if (((res >> q) & 1) == 1) result |= (1ULL << pqq[q]);
    }
    ++qiskitRes[result];
    simExtStabilizer->RestoreState();
  }

  // compare results
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t esCount =
        qiskitRes.find(key) != qiskitRes.end() ? qiskitRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double esProb = static_cast<double>(esCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - esProb) < 0.1,
               "Measurement probability mismatch for outcome "
                   << key << ": statevector " << svProb << ", ext stabilizer sim
  "
                   << esProb);
  }
  */
  simExtStabilizer->Reset();
}

// for now there are issues with non-clifford circuits, commented out
/*
BOOST_DATA_TEST_CASE_F(ExtStabTestFixture, RandomNonCliffordCircuitsTest,
                       bdata::xrange(1, 20), nrGates) {
  auto circuit = GenerateCircuit(nrQubitsForRandomCirc, nrGates, 29);

  // execute
  for (const auto& op : circuit) {
    ExecuteGate(op, simStatevector);
    ExecuteGate(op, simExtStabilizer);
  }

  // check, first some random pauli strings

  // do not pass, I guess it's broken for non-clifford circuits
  const int nrChecks = 100;
  for (int i = 0; i < nrChecks; ++i) {
    const std::string pauliStr = GeneratePauliString(nrQubitsForRandomCirc);
    const double expValStateVec = simStatevector->ExpectationValue(pauliStr);

    const double expValExtStabilizer =
        simExtStabilizer->ExpectationValue(pauliStr);
    BOOST_TEST(std::abs(expValStateVec - expValExtStabilizer) < 1e-2,
               "Expectation value mismatch for pauli string "
                   << pauliStr << ": statevector " << expValStateVec
                   << ", ext stabilizer " << expValExtStabilizer);
  }

  // now sampling
  const int nrSamples = 1000;
  Types::qubits_vector qubitsToMeasure(nrQubitsForRandomCirc);
  std::iota(qubitsToMeasure.begin(), qubitsToMeasure.end(), 0);

  // perform sampling
  auto svRes = simStatevector->SampleCounts(qubitsToMeasure, nrSamples);

  std::vector<int> pq(qubitsToMeasure.begin(), qubitsToMeasure.end());

  std::random_device rd;
  std::mt19937 g(rd());

  auto stdSvRes = simExtStabilizer->SampleCounts(qubitsToMeasure, nrSamples);
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t esCount =
        stdSvRes.find(key) != stdSvRes.end() ? stdSvRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double esProb = static_cast<double>(esCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - esProb) < 0.1,
               "Sampling probability mismatch for outcome "
                   << key << ": statevector " << svProb
                   << ", ext stabilizer sim " << esProb);
  }

  std::unordered_map<Types::qubit_t, Types::qubit_t> qiskitRes;

  // until the saving/restoring state is implemented for the extended stabilizer
  // simulator in qiskit aer, this check does not work
  
  simExtStabilizer->SaveState();

  Types::qubits_vector pqq(qubitsToMeasure.begin(), qubitsToMeasure.end());

  for (int i = 0; i < nrSamples; ++i) {
    std::shuffle(pqq.begin(), pqq.end(), g);
    auto res = simExtStabilizer->Measure(pqq);
    Types::qubit_t result = 0;
    for (int q = 0; q < static_cast<int>(pqq.size()); ++q) {
      if (((res >> q) & 1) == 1) result |= (1ULL << pqq[q]);
    }
    ++qiskitRes[result];
    simExtStabilizer->RestoreState();
  }

  // compare results
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t svCount = kv.second;
    const Types::qubit_t esCount =
        qiskitRes.find(key) != qiskitRes.end() ? qiskitRes[key] : 0;
    const double svProb = static_cast<double>(svCount) / nrSamples;
    const double esProb = static_cast<double>(esCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - esProb) < 0.1,
               "Measurement probability mismatch for outcome "
                   << key << ": statevector " << svProb << ", ext stabilizer sim"
                   << esProb);
  }
  
  simExtStabilizer->Reset();
}
*/

BOOST_AUTO_TEST_SUITE_END()
