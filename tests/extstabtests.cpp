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
    simExtStabilizer->Configure("extended_stabilizer_approximation_error",
                                "0.02");
    simExtStabilizer->Configure("extended_stabilizer_norm_estimation_samples",
                                "3000");
    simExtStabilizer->Configure(
        "extended_stabilizer_norm_estimation_repetitions", "7");
    simExtStabilizer->Configure("seed_simulator", "1515870810");
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
    std::uniform_int_distribution<int> dist(0, 3);

    for (int i = 0; i < nrQubits; ++i) {
      const int v = dist(generator);
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
    std::uniform_int_distribution<int> dist(0, maxGate);

    std::uniform_real_distribution<double> param_dist(0.0, 2. * M_PI);
    std::bernoulli_distribution bool_dist(
        0.9);  // high chance to make a clifford gate from a non-clifford one

    std::vector<Operation> circuit;
    std::vector<int> qubits(nrQubits);
    std::iota(qubits.begin(), qubits.end(), 0);

    for (int i = 0; i < nrGates; ++i) {
      Operation op;

      std::shuffle(qubits.begin(), qubits.end(), generator);

      int gate = dist(generator);  // random gate id

      if (gate > 12 && bool_dist(generator))
        gate %= 13;  // make it clifford, to have less non-clifford gates in
                     // the circuit, which are more expensive to simulate

      op.gate = gate;
      op.qubit1 = qubits[0];
      op.qubit2 = qubits[1];
      op.qubit3 = qubits[2];

      op.theta = param_dist(generator);
      op.phi = param_dist(generator);
      op.lambda = param_dist(generator);
      op.gamma = param_dist(generator);

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
  std::mt19937 generator{0x5A17B1E3U};
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

BOOST_FIXTURE_TEST_CASE(IncrementalNonCliffordEvolutionTest,
                        ExtStabTestFixture) {
  const auto checkProbabilities = [&]() {
    const size_t dimension = 1ULL << nrQubitsForRandomCirc;
    for (size_t outcome = 0; outcome < dimension; ++outcome) {
      const double expected = simStatevector->Probability(outcome);
      const double actual = simExtStabilizer->Probability(outcome);
      BOOST_TEST(std::abs(expected - actual) < 0.08,
                 "Probability mismatch for outcome "
                     << outcome << ": statevector " << expected
                     << ", extended stabilizer " << actual);
    }
  };

  simStatevector->ApplyH(0);
  simExtStabilizer->ApplyH(0);
  simStatevector->ApplyP(0, 0.37);
  simExtStabilizer->ApplyP(0, 0.37);
  simStatevector->ApplyH(0);
  simExtStabilizer->ApplyH(0);
  checkProbabilities();

  // Exercise a Clifford-only controller flush after the decomposition exists.
  simStatevector->ApplyH(1);
  simExtStabilizer->ApplyH(1);
  checkProbabilities();

  // Exercise a second non-Clifford controller flush without reinitializing the
  // already-expanded runner.
  simStatevector->ApplyRz(1, -0.29);
  simExtStabilizer->ApplyRz(1, -0.29);
  simStatevector->ApplyH(1);
  simExtStabilizer->ApplyH(1);
  checkProbabilities();
}

BOOST_DATA_TEST_CASE_F(ExtStabTestFixture, RandomNonCliffordCircuitsTest,
                       bdata::xrange(1, 4), nrGates) {
  auto circuit = GenerateCircuit(nrQubitsForRandomCirc, nrGates, 27);
  if (std::none_of(circuit.begin(), circuit.end(),
                   [](const Operation& op) { return op.gate > 12; })) {
    circuit.front().gate = 18;  // guarantee at least one T gate
  }

  for (const auto& op : circuit) {
    ExecuteGate(op, simStatevector);
    ExecuteGate(op, simExtStabilizer);
  }

  const int nrChecks = 1;
  for (int i = 0; i < nrChecks; ++i) {
    const std::string pauliStr = GeneratePauliString(nrQubitsForRandomCirc);
    const double expValStateVec = simStatevector->ExpectationValue(pauliStr);
    const double expValExtStabilizer =
        simExtStabilizer->ExpectationValue(pauliStr);
    BOOST_TEST(std::abs(expValStateVec - expValExtStabilizer) < 0.25,
               "Expectation value mismatch for pauli string "
                   << pauliStr << ": statevector " << expValStateVec
                   << ", ext stabilizer " << expValExtStabilizer);
  }

  const int nrSamples = 300;
  Types::qubits_vector qubitsToMeasure(nrQubitsForRandomCirc);
  std::iota(qubitsToMeasure.begin(), qubitsToMeasure.end(), 0);
  auto svRes = simStatevector->SampleCounts(qubitsToMeasure, nrSamples);
  auto extRes = simExtStabilizer->SampleCounts(qubitsToMeasure, nrSamples);
  for (const auto& kv : svRes) {
    const Types::qubit_t key = kv.first;
    const Types::qubit_t esCount =
        extRes.find(key) != extRes.end() ? extRes[key] : 0;
    const double svProb = static_cast<double>(kv.second) / nrSamples;
    const double esProb = static_cast<double>(esCount) / nrSamples;
    BOOST_TEST(std::abs(svProb - esProb) < 0.2,
               "Sampling probability mismatch for outcome "
                   << key << ": statevector " << svProb
                   << ", ext stabilizer sim " << esProb);
  }
}

BOOST_DATA_TEST_CASE_F(ExtStabTestFixture, NonCliffordGateCoverageTest,
                       bdata::xrange(13, 29), gate) {
  for (Types::qubit_t qubit = 0; qubit < 3; ++qubit) {
    simStatevector->ApplyH(qubit);
    simExtStabilizer->ApplyH(qubit);
  }

  const Operation operation{gate, 0, 1, 2, 0.37, -0.29, 0.41, 0.17};
  ExecuteGate(operation, simStatevector);
  ExecuteGate(operation, simExtStabilizer);
  for (Types::qubit_t qubit = 0; qubit < 3; ++qubit) {
    simStatevector->ApplyH(qubit);
    simExtStabilizer->ApplyH(qubit);
  }

  const size_t dimension = 1ULL << nrQubitsForRandomCirc;
  for (size_t outcome = 0; outcome < dimension; ++outcome) {
    const double expected = simStatevector->Probability(outcome);
    const double actual = simExtStabilizer->Probability(outcome);
    BOOST_TEST(std::abs(expected - actual) < 0.15,
               "Gate " << gate << " probability mismatch for outcome "
                       << outcome << ": statevector " << expected
                       << ", extended stabilizer " << actual);
  }
}

BOOST_FIXTURE_TEST_CASE(ExtendedStabilizerStateQueriesTest,
                        ExtStabTestFixture) {
  simStatevector->ApplyH(0);
  simExtStabilizer->ApplyH(0);
  simStatevector->ApplyRz(0, M_PI / 3.0);
  simExtStabilizer->ApplyRz(0, M_PI / 3.0);

  for (Types::qubit_t outcome = 0; outcome < 2; ++outcome)
    BOOST_TEST(std::abs(simStatevector->Amplitude(outcome) -
                        simExtStabilizer->Amplitude(outcome)) < 0.08);

  const auto expected = simStatevector->AllProbabilities();
  const auto actual = simExtStabilizer->AllProbabilities();
  BOOST_REQUIRE_EQUAL(actual.size(), expected.size());
  for (size_t outcome = 0; outcome < expected.size(); ++outcome)
    BOOST_TEST(std::abs(expected[outcome] - actual[outcome]) < 0.08);

  const Types::qubits_vector selectedOutcomes = {0, 3, 5, 7};
  const auto selected = simExtStabilizer->Probabilities(selectedOutcomes);
  BOOST_REQUIRE_EQUAL(selected.size(), selectedOutcomes.size());
  for (size_t index = 0; index < selected.size(); ++index)
    BOOST_TEST(std::abs(selected[index] - actual[selectedOutcomes[index]]) <
               1e-10);
}

BOOST_FIXTURE_TEST_CASE(SaveRestoreMeasurementTest, ExtStabTestFixture) {
  simExtStabilizer->ApplyH(0);
  simExtStabilizer->ApplyT(0);
  simExtStabilizer->ApplyCX(0, 1);

  constexpr size_t nrSamples = 300;
  const Types::qubits_vector qubitsToMeasure = {0, 1, 2, 3};
  const auto expected =
      simExtStabilizer->SampleCounts(qubitsToMeasure, nrSamples);
  std::unordered_map<Types::qubit_t, Types::qubit_t> actual;

  simExtStabilizer->SaveState();

  auto clone = simExtStabilizer->Clone();
  BOOST_REQUIRE(clone);
  const auto originalProbabilities = simExtStabilizer->AllProbabilities();
  const auto clonedProbabilities = clone->AllProbabilities();
  BOOST_REQUIRE_EQUAL(clonedProbabilities.size(), originalProbabilities.size());
  for (size_t outcome = 0; outcome < originalProbabilities.size(); ++outcome)
    BOOST_TEST(std::abs(clonedProbabilities[outcome] -
                        originalProbabilities[outcome]) < 1e-10);

  for (size_t sample = 0; sample < nrSamples; ++sample) {
    ++actual[simExtStabilizer->Measure(qubitsToMeasure)];
    simExtStabilizer->RestoreState();
  }

  for (const auto& kv : expected) {
    const Types::qubit_t count =
        actual.find(kv.first) == actual.end() ? 0 : actual[kv.first];
    const double expectedProbability =
        static_cast<double>(kv.second) / nrSamples;
    const double actualProbability = static_cast<double>(count) / nrSamples;
    BOOST_TEST(std::abs(expectedProbability - actualProbability) < 0.15,
               "Restored measurement mismatch for outcome "
                   << kv.first << ": sampled " << expectedProbability
                   << ", measured " << actualProbability);
  }
}

BOOST_AUTO_TEST_SUITE_END()
