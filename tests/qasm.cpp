/**
 * @file qasm.cpp
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Tests for the qasm parser and generator.
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

// project being tested
#include "../Simulators/Factory.h"

#include "../Circuit/Circuit.h"
#include "../Circuit/Conditional.h"
#include "../Circuit/Measurements.h"
#include "../Circuit/QuantumGates.h"
#include "../Circuit/RandomOp.h"
#include "../Circuit/Reset.h"
#include "../Circuit/Factory.h"

#include "../qasm/QasmCirc.h"
#include "../qasm/CircQasm.h"

struct QasmTestFixture {
  QasmTestFixture() {
    state.AllocateBits(nrQubitsForRandomCirc);
    resetCirc = std::make_shared<Circuits::Circuit<>>();
    Types::qubits_vector qubits(nrQubitsForRandomCirc);
    std::iota(qubits.begin(), qubits.end(), 0);
    resetCirc->AddOperation(std::make_shared<Circuits::Reset<>>(qubits));

    randomCirc = std::make_shared<Circuits::Circuit<>>();

    qc = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qc->AllocateQubits(nrQubitsForRandomCirc);
    qc->Initialize();

    qc2 = Simulators::SimulatorsFactory::CreateSimulator(
        Simulators::SimulatorType::kQCSim,
        Simulators::SimulationType::kStatevector);
    qc2->AllocateQubits(nrQubitsForRandomCirc);
    qc2->Initialize();
  }

  // fills randomly the circuit with gates
  void GenerateCircuit(int nrGates, int nrQubits, double probReset = 0.,
                       double probMeasurement = 0.) {
    std::random_device rd;
    std::mt19937 g(rd());

    auto dblGen = bdata::random(-2. * M_PI, 2. * M_PI);
    auto dblGenIter = dblGen.begin();

    auto gateGen = bdata::random(
        0, static_cast<int>(Circuits::QuantumGateType::kCUGateType));
    auto gateGenIter = gateGen.begin();

    auto typeGen = bdata::random();
    auto typeGenIter = typeGen.begin();

    const double cumProb = probReset + probMeasurement;

    // TODO: Maybe insert from time to time a random number generating 'gate'
    // and a conditioned random one, those should not affect results?
    for (int gateNr = 0; gateNr < nrGates; ++gateNr, ++gateGenIter) {
      const double typeOpProb = *typeGenIter;
      ++typeGenIter;

      Types::qubits_vector qubits(nrQubits);
      std::iota(qubits.begin(), qubits.end(), 0);
      std::shuffle(qubits.begin(), qubits.end(), g);

      if (typeOpProb < probReset) {
        // reset operation
        auto resetOp = Circuits::CircuitFactory<>::CreateReset({qubits[0]});
        randomCirc->AddOperation(resetOp);
        continue;
      } else if (typeOpProb < cumProb) {
        // measurement operation
        auto measOp = Circuits::CircuitFactory<>::CreateMeasurement(
            {{qubits[0], qubits[0]}});
        randomCirc->AddOperation(measOp);
        continue;
      }

      // create a random gate and add it to the circuit

      // first, pick randomly three qubits (depending on the randomly chosen
      // gate type, not all of them will be used)
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

      const Circuits::QuantumGateType gateType =
          static_cast<Circuits::QuantumGateType>(*gateGenIter);

      auto theGate = Circuits::CircuitFactory<>::CreateGate(
          gateType, q1, q2, q3, param1, param2, param3);
      randomCirc->AddOperation(theGate);
    }
  }

  void PrintCircuit(const std::shared_ptr<Circuits::Circuit<>> &circuit) {
    std::cout << "Circuit with " << circuit->size()
              << " operations:" << std::endl;
    for (const auto &op : circuit->GetOperations()) {
      std::cout << "Operation type: ";
      switch (op->GetType()) {
        case Circuits::OperationType::kGate:
          std::cout << "Gate ";
          {
            auto gatePtr = static_cast<Circuits::IQuantumGate<> *>(op.get());
            PrintGateName(gatePtr->GetGateType());

            auto params = gatePtr->GetParams();
            if (!params.empty()) {
              std::cout << " (";
              for (size_t i = 0; i < params.size(); ++i) {
                std::cout << params[i];
                if (i < params.size() - 1) std::cout << ", ";
              }
              std::cout << ")";
            }

            auto qubits = op->AffectedQubits();
            std::cout << " ";
            for (size_t i = 0; i < qubits.size(); ++i) {
              std::cout << qubits[i];
              if (i < qubits.size() - 1) std::cout << ", ";
            }
          }
          break;
        case Circuits::OperationType::kMeasurement:
          std::cout << "Measurement ";
          {
            auto qubits = op->AffectedQubits();
            std::cout << " (";
            for (size_t i = 0; i < qubits.size(); ++i) {
              std::cout << qubits[i];
              if (i < qubits.size() - 1) std::cout << ", ";
            }
            std::cout << ")";

            std::cout << " -> ";

            auto bits = op->AffectedBits();
            std::cout << " (";
            for (size_t i = 0; i < bits.size(); ++i) {
              std::cout << bits[i];
              if (i < bits.size() - 1) std::cout << ", ";
            }
            std::cout << ")";
          }
          break;
        case Circuits::OperationType::kReset:
          std::cout << "Reset";
          {
            auto qubits = op->AffectedQubits();
            std::cout << " (";
            for (size_t i = 0; i < qubits.size(); ++i) {
              std::cout << qubits[i];
              if (i < qubits.size() - 1) std::cout << ", ";
            }
            std::cout << ")";
          }
          break;
        case Circuits::OperationType::kConditionalGate:
          std::cout << "Conditional";
          break;
        default:
          std::cout << "Other";
          break;
      }
      std::cout << std::endl;
    }
  }

  void PrintGateName(Circuits::QuantumGateType gateType) {
    std::string name;

    switch (gateType) {
      case Circuits::QuantumGateType::kPhaseGateType:
        name = "p";
        break;
      case Circuits::QuantumGateType::kXGateType:
        name = "x";
        break;
      case Circuits::QuantumGateType::kYGateType:
        name = "y";
        break;
      case Circuits::QuantumGateType::kZGateType:
        name = "z";
        break;
      case Circuits::QuantumGateType::kHadamardGateType:
        name = "h";
        break;
      case Circuits::QuantumGateType::kSGateType:
        name = "s";
        break;
      case Circuits::QuantumGateType::kSdgGateType:
        name = "sdg";
        break;
      case Circuits::QuantumGateType::kTGateType:
        name = "t";
        break;
      case Circuits::QuantumGateType::kTdgGateType:
        name = "tdg";
        break;

      case Circuits::QuantumGateType::kSxGateType:
        name = "sx";
        break;
      case Circuits::QuantumGateType::kSxDagGateType:
        name = "sxdg";
        break;
      case Circuits::QuantumGateType::kKGateType:
        name = "k";
        break;
      case Circuits::QuantumGateType::kRxGateType:
        name = "rx";
        break;
      case Circuits::QuantumGateType::kRyGateType:
        name = "ry";
        break;
      case Circuits::QuantumGateType::kRzGateType:
        name = "rz";
        break;

      case Circuits::QuantumGateType::kUGateType:
        name = "u";
        break;

      case Circuits::QuantumGateType::kCXGateType:
        name = "cx";
        // standard gate
        break;
      case Circuits::QuantumGateType::kCYGateType:
        name = "cy";
        break;
      case Circuits::QuantumGateType::kCZGateType:
        name = "cz";
        break;
      case Circuits::QuantumGateType::kCPGateType:
        name = "cp";
        break;

      case Circuits::QuantumGateType::kCRxGateType:
        name = "crx";
        break;
      case Circuits::QuantumGateType::kCRyGateType:
        name = "cry";
        break;

      case Circuits::QuantumGateType::kCRzGateType:
        name = "crz";
        break;
      case Circuits::QuantumGateType::kCHGateType:
        name = "ch";
        break;

      case Circuits::QuantumGateType::kCSxGateType:
        name = "csx";
        break;
      case Circuits::QuantumGateType::kCSxDagGateType:
        name = "csxdg";
        break;

      case Circuits::QuantumGateType::kCUGateType:
        name = "cu";
        break;
      case Circuits::QuantumGateType::kSwapGateType:
        name = "swap";
        break;
      case Circuits::QuantumGateType::kCSwapGateType:
        name = "cswap";
        break;
      case Circuits::QuantumGateType::kCCXGateType:
        name = "ccx";
        break;
      default:
        name = "unknown";
        break;
    }

    std::cout << " " << name << " ";
  }

  Circuits::OperationState state;
  std::shared_ptr<Simulators::ISimulator> qc;
  std::shared_ptr<Simulators::ISimulator> qc2;

  // for testing randomly generated circuits
  const unsigned int nrQubitsForRandomCirc = 5;

  std::shared_ptr<Circuits::Circuit<>> randomCirc;
  std::shared_ptr<Circuits::Circuit<>> resetCirc;
};

extern bool checkClose(std::complex<double> a, std::complex<double> b,
                       double dif);

BOOST_AUTO_TEST_SUITE(qasm_tests)

BOOST_FIXTURE_TEST_CASE(InitializationTests, QasmTestFixture) {
  BOOST_TEST(qc);
  BOOST_TEST(qc2);
  BOOST_TEST(randomCirc);
  BOOST_TEST(resetCirc);
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsTest,
                       bdata::xrange(1, 30), nrGates) {
  size_t nrStates = 1ULL << nrQubitsForRandomCirc;

  for (int i = 0; i < 5; ++i) {
    GenerateCircuit(nrGates, nrQubitsForRandomCirc);

    randomCirc->Execute(qc, state);

    // export to qasm
    std::string qasmStr = qasm::CircToQasm<>::Generate(randomCirc);

    // import from qasm
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

    if (!parser.Failed()) {
      // execute imported circuit
      circuit->Execute(qc2, state);

      // compare results
      for (size_t i = 0; i < nrStates; ++i) {
        double prob1 = qc->Probability(i);
        double prob2 = qc2->Probability(i);

        BOOST_TEST(checkClose(std::complex<double>(prob1, 0.),
                              std::complex<double>(prob2, 0.), 0.0001),
                   "Probability mismatch for state |" << i << ">: " << prob1
                                                      << " vs " << prob2
                                                      << ", qasm: " << qasmStr);

        if (!checkClose(std::complex<double>(prob1, 0.),
                        std::complex<double>(prob2, 0.), 0.0001)) {
          std::cout << "Original circuit:" << std::endl;
          PrintCircuit(randomCirc);

          std::cout << "\nConverted circuit:" << std::endl;
          PrintCircuit(circuit);
        }
      }
    }

    randomCirc->Clear();

    resetCirc->Execute(qc, state);
    resetCirc->Execute(qc2, state);
    state.Reset();
  }
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsWithMeasAndResetTest,
                       bdata::xrange(20, 40), nrGates) {
  static const int nrShots = 5000;

  for (int i = 0; i < 5; ++i) {
    GenerateCircuit(nrGates, nrQubitsForRandomCirc, 0.025, 0.15);

    // export to qasm
    std::string qasmStr = qasm::CircToQasm<>::Generate(randomCirc);

    // import from qasm
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

    /*
    std::cout << "Original circuit:" << std::endl;
    PrintCircuit(randomCirc);
    std::cout << "\nConverted circuit:" << std::endl;
    PrintCircuit(circuit);

    std::cout << "QASM:" << std::endl;
    std::cout << qasmStr << std::endl;

    exit(0);
    */

    if (!parser.Failed()) {
      std::unordered_map<std::vector<bool>, size_t> results1;
      std::unordered_map<std::vector<bool>, size_t> results2;

      for (int trial = 0; trial < nrShots; ++trial) {
        randomCirc->Execute(qc, state);

        results1[state.GetAllBits()]++;

        state.Reset();

        // execute imported circuit
        circuit->Execute(qc2, state);

        results2[state.GetAllBits()]++;

        resetCirc->Execute(qc, state);
        resetCirc->Execute(qc2, state);
        state.Reset();
      }

      for (const auto &[key, cnt] : results1) {
        double val = static_cast<double>(cnt) / static_cast<double>(nrShots);

        if (val < 0.03) continue;

        double val2 = 0;
        if (results2.find(key) != results2.end())
          val2 =
              static_cast<double>(results2[key]) / static_cast<double>(nrShots);

        BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
      }

      for (const auto &[key, cnt] : results2) {
        double val = static_cast<double>(cnt) / static_cast<double>(nrShots);
        if (val < 0.03) continue;

        double val2 = 0;
        if (results1.find(key) != results1.end())
          val2 =
              static_cast<double>(results1[key]) / static_cast<double>(nrShots);
        BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
      }
    }

    randomCirc->Clear();
  }
}

// ****************************************************************************
// Task 7: QASM3 export mode - same randomized round-trip coverage as the two
// cases above, but exported through QasmVersion::V3 so the QASM3 import path
// (qubit[]/bit[] declarations, `= measure` assignment form) gets exercised
// too. Near-copies of RandomCircuitsTest/RandomCircuitsWithMeasAndResetTest,
// differing only in the version argument passed to Generate.

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsV3Test,
                       bdata::xrange(1, 30), nrGates) {
  size_t nrStates = 1ULL << nrQubitsForRandomCirc;

  for (int i = 0; i < 5; ++i) {
    GenerateCircuit(nrGates, nrQubitsForRandomCirc);

    randomCirc->Execute(qc, state);

    // export to qasm3
    std::string qasmStr = qasm::CircToQasm<>::Generate(
        randomCirc, qasm::CircToQasm<>::QasmVersion::V3);

    // import from qasm
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

    if (!parser.Failed()) {
      // execute imported circuit
      circuit->Execute(qc2, state);

      // compare results
      for (size_t i = 0; i < nrStates; ++i) {
        double prob1 = qc->Probability(i);
        double prob2 = qc2->Probability(i);

        BOOST_TEST(checkClose(std::complex<double>(prob1, 0.),
                              std::complex<double>(prob2, 0.), 0.0001),
                   "Probability mismatch for state |" << i << ">: " << prob1
                                                      << " vs " << prob2
                                                      << ", qasm: " << qasmStr);

        if (!checkClose(std::complex<double>(prob1, 0.),
                        std::complex<double>(prob2, 0.), 0.0001)) {
          std::cout << "Original circuit:" << std::endl;
          PrintCircuit(randomCirc);

          std::cout << "\nConverted circuit:" << std::endl;
          PrintCircuit(circuit);
        }
      }
    }

    randomCirc->Clear();

    resetCirc->Execute(qc, state);
    resetCirc->Execute(qc2, state);
    state.Reset();
  }
}

BOOST_DATA_TEST_CASE_F(QasmTestFixture, RandomCircuitsWithMeasAndResetV3Test,
                       bdata::xrange(20, 40), nrGates) {
  static const int nrShots = 5000;

  for (int i = 0; i < 5; ++i) {
    GenerateCircuit(nrGates, nrQubitsForRandomCirc, 0.025, 0.15);

    // export to qasm3
    std::string qasmStr = qasm::CircToQasm<>::Generate(
        randomCirc, qasm::CircToQasm<>::QasmVersion::V3);

    // import from qasm
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

    if (!parser.Failed()) {
      std::unordered_map<std::vector<bool>, size_t> results1;
      std::unordered_map<std::vector<bool>, size_t> results2;

      for (int trial = 0; trial < nrShots; ++trial) {
        randomCirc->Execute(qc, state);

        results1[state.GetAllBits()]++;

        state.Reset();

        // execute imported circuit
        circuit->Execute(qc2, state);

        results2[state.GetAllBits()]++;

        resetCirc->Execute(qc, state);
        resetCirc->Execute(qc2, state);
        state.Reset();
      }

      for (const auto &[key, cnt] : results1) {
        double val = static_cast<double>(cnt) / static_cast<double>(nrShots);

        if (val < 0.03) continue;

        double val2 = 0;
        if (results2.find(key) != results2.end())
          val2 =
              static_cast<double>(results2[key]) / static_cast<double>(nrShots);

        BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
      }

      for (const auto &[key, cnt] : results2) {
        double val = static_cast<double>(cnt) / static_cast<double>(nrShots);
        if (val < 0.03) continue;

        double val2 = 0;
        if (results1.find(key) != results1.end())
          val2 =
              static_cast<double>(results1[key]) / static_cast<double>(nrShots);
        BOOST_CHECK_CLOSE(val, val2, val2 < 0.1 ? 66 : 33);
      }
    }

    randomCirc->Clear();
  }
}

// ****************************************************************************
// Task 1: QASM3 foundation - identifier widening regression and stdgates.inc
// aliases.

namespace {

// Fetches the gate type of the nr-th operation in the circuit, requiring
// that operation to actually be a quantum gate.
Circuits::QuantumGateType NthGateType(
    const std::shared_ptr<Circuits::Circuit<>> &circuit, size_t nr) {
  BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() > nr);

  const auto &op = circuit->GetOperations()[nr];
  BOOST_TEST_REQUIRE((op->GetType() == Circuits::OperationType::kGate));

  return std::static_pointer_cast<Circuits::IQuantumGate<>>(op)->GetGateType();
}

}  // namespace

BOOST_AUTO_TEST_CASE(QASM2CXStillDispatchesAsCXGate) {
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "CX q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kCXGateType));
}

BOOST_AUTO_TEST_CASE(QASM2LowercaseCxStillDispatchesAsCXGate) {
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "cx q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kCXGateType));
}

BOOST_AUTO_TEST_CASE(QASM2UStillDispatchesAsUGate) {
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "U(0.1, 0.2, 0.3) q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kUGateType));
}

BOOST_AUTO_TEST_CASE(QASM2LowercaseUStillDispatchesAsUGate) {
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "u(0.1, 0.2, 0.3) q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kUGateType));
}

BOOST_AUTO_TEST_CASE(QASM3IdentifierAllowsUppercaseInitialGateName) {
  // OpenQASM 3's `Identifier: [A-Za-z_][A-Za-z0-9_]*` admits an uppercase
  // initial letter, so a gate may be declared and called by such a name. This
  // used to be asserted under an `OPENQASM 2.0;` header, where QASM2's
  // `id := [a-z][A-Za-z0-9_]*` does not allow it - see
  // QASM2IdentifierRejectsUppercaseAndUnderscoreInitials for that half.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "gate MyGate a { x a; }\n"
      "MyGate q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kXGateType));
}

BOOST_AUTO_TEST_CASE(QASM3IdentifierAllowsUnderscoreInitialIdentifier) {
  // The other half of QASM3's identifier rule: an underscore-initial name.
  // Also under a 3.0 header for the same reason as above.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg _q[1];\n"
      "x _q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kXGateType));
}

BOOST_AUTO_TEST_CASE(PhaseIsAliasForP) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "phase(0.5) q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kPhaseGateType));
}

BOOST_AUTO_TEST_CASE(CphaseIsAliasForCp) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "cphase(0.5) q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kCPGateType));
}

BOOST_FIXTURE_TEST_CASE(PhaseAndCphaseMatchPAndCpProbabilities,
                        QasmTestFixture) {
  // 'phase'/'cphase' are stdgates.inc names, so both programs carry a 3.0
  // header; 'p'/'cp' are accepted in both dialects.
  const std::string pStyleQasm =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "p(0.5) q[0];\n"
      "cp(0.3) q[0], q[1];\n";
  const std::string phaseStyleQasm =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "phase(0.5) q[0];\n"
      "cphase(0.3) q[0], q[1];\n";

  qasm::QasmToCirc<> pParser;
  auto pCircuit = pParser.ParseAndTranslate(pStyleQasm);
  BOOST_TEST(!pParser.Failed(), pParser.GetErrorMessage());

  qasm::QasmToCirc<> phaseParser;
  auto phaseCircuit = phaseParser.ParseAndTranslate(phaseStyleQasm);
  BOOST_TEST(!phaseParser.Failed(), phaseParser.GetErrorMessage());

  if (!pParser.Failed() && !phaseParser.Failed()) {
    pCircuit->Execute(qc, state);
    state.Reset();
    phaseCircuit->Execute(qc2, state);
    state.Reset();

    const size_t nrStates = 1ULL << nrQubitsForRandomCirc;
    for (size_t i = 0; i < nrStates; ++i) {
      const double prob1 = qc->Probability(i);
      const double prob2 = qc2->Probability(i);

      BOOST_TEST(checkClose(std::complex<double>(prob1, 0.),
                            std::complex<double>(prob2, 0.), 0.0001),
                 "Probability mismatch for state |" << i << ">: " << prob1
                                                    << " vs " << prob2);
    }
  }
}

BOOST_AUTO_TEST_CASE(GphaseParsesAsNoOp) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "gphase(0.5);\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

  if (!parser.Failed())
    BOOST_TEST(circuit->GetNumberOfOperations() == 0u,
               "gphase should not add any operation to the circuit, got "
                   << circuit->GetNumberOfOperations());
}

BOOST_AUTO_TEST_CASE(GphaseRejectsMultipleParameters) {
  // 'gphase' takes a single angle argument; it belongs only in
  // allowedOneParamGates, so a two-parameter call must be rejected.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "gphase(0.5, 0.6);\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(parser.Failed(),
             "gphase with two parameters should have been rejected");
}

BOOST_AUTO_TEST_CASE(UnsupportedParameterizedGateWithoutQubitsIsRejected) {
  // The ExpGatecallType branch of AddGateExpr must reject unrecognized gate
  // names symmetrically with the SimpleGatecallType branch (which already
  // throws "Unsupported gate without parameters"), rather than silently
  // discarding the call. This is the zero-qubit form, which expGatecall now
  // parses (to allow gphase's qubit-less call).
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "foobar(0.5);\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(parser.Failed(),
             "Unsupported gate 'foobar' should have been rejected, not "
             "silently ignored");
}

BOOST_AUTO_TEST_CASE(UnsupportedParameterizedGateWithQubitIsRejected) {
  // Same as above, but with a qubit argument supplied - the form that
  // motivates the symmetry with the SimpleGatecallType branch.
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "foobar(0.5) q[0];\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(parser.Failed(),
             "Unsupported gate 'foobar' should have been rejected, not "
             "silently ignored");
}

// ****************************************************************************
// Task 2: QASM3 declarations - `qubit[n]` / `bit[n]` and the bare (size-1)
// spellings `qubit` / `bit`.

BOOST_AUTO_TEST_CASE(QASM3SizedDeclarationsProduceWorkingCircuit) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "qubit[2] q;\n"
      "bit[2] c;\n"
      "h q[0];\n"
      "cx q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(circuit != nullptr);

  BOOST_TEST(circuit->GetQubits().size() == 2u,
             "Expected 2 qubits touched by the circuit, got "
                 << circuit->GetQubits().size());
}

BOOST_FIXTURE_TEST_CASE(QASM3DeclarationsMatchQASM2Probabilities,
                        QasmTestFixture) {
  // The QASM3 spelling and its QASM2 twin must produce circuits with
  // identical per-state probabilities - this is what actually proves the
  // MakeIndexedId/AddQreg/AddCreg reuse is correct, not just that parsing
  // succeeds.
  const std::string qasm3Str =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "qubit[2] q;\n"
      "bit[2] c;\n"
      "h q[0];\n"
      "cx q[0], q[1];\n";
  const std::string qasm2Str =
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "creg c[2];\n"
      "h q[0];\n"
      "cx q[0], q[1];\n";

  qasm::QasmToCirc<> qasm3Parser;
  auto qasm3Circuit = qasm3Parser.ParseAndTranslate(qasm3Str);
  BOOST_TEST(!qasm3Parser.Failed(), qasm3Parser.GetErrorMessage());

  qasm::QasmToCirc<> qasm2Parser;
  auto qasm2Circuit = qasm2Parser.ParseAndTranslate(qasm2Str);
  BOOST_TEST(!qasm2Parser.Failed(), qasm2Parser.GetErrorMessage());

  if (!qasm3Parser.Failed() && !qasm2Parser.Failed()) {
    qasm3Circuit->Execute(qc, state);
    state.Reset();
    qasm2Circuit->Execute(qc2, state);
    state.Reset();

    const size_t nrStates = 1ULL << nrQubitsForRandomCirc;
    for (size_t i = 0; i < nrStates; ++i) {
      const double prob1 = qc->Probability(i);
      const double prob2 = qc2->Probability(i);

      BOOST_TEST(checkClose(std::complex<double>(prob1, 0.),
                            std::complex<double>(prob2, 0.), 0.0001),
                 "Probability mismatch for state |" << i << ">: " << prob1
                                                    << " vs " << prob2);
    }
  }
}

BOOST_AUTO_TEST_CASE(QASM3BareDeclarationYieldsSingleQubitRegister) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "bit c;\n"
      "x q;\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(circuit != nullptr);

  BOOST_TEST(circuit->GetQubits().size() == 1u,
             "Expected 1 qubit touched by the circuit, got "
                 << circuit->GetQubits().size());
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kXGateType));
}

BOOST_AUTO_TEST_CASE(QASM3MultipleBareDeclarationsAllocateDistinctQubits) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit a;\n"
      "qubit b;\n"
      "x a;\n"
      "x b;\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(circuit != nullptr);
  BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() >= 2u);

  const auto qubitA = std::static_pointer_cast<Circuits::IQuantumGate<>>(
                          circuit->GetOperations()[0])
                          ->GetQubit(0);
  const auto qubitB = std::static_pointer_cast<Circuits::IQuantumGate<>>(
                          circuit->GetOperations()[1])
                          ->GetQubit(0);

  BOOST_TEST(qubitA == 0u,
             "Expected bare qubit 'a' at offset 0, got " << qubitA);
  BOOST_TEST(qubitB == 1u,
             "Expected bare qubit 'b' at offset 1, got " << qubitB);
}

BOOST_AUTO_TEST_CASE(QASM3MultipleRegistersAllocateCorrectBaseOffsets) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[2] a;\n"
      "qubit[3] b;\n"
      "x b[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(circuit != nullptr);
  BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() >= 1u);

  const auto gateQubit = std::static_pointer_cast<Circuits::IQuantumGate<>>(
                             circuit->GetOperations()[0])
                             ->GetQubit(0);

  BOOST_TEST(gateQubit == 2u,
             "Expected b[0] to be allocated at qubit index 2 (after a's 2 "
             "qubits), got "
                 << gateQubit);
}

BOOST_AUTO_TEST_CASE(QASM3QubitDeclRejectedUnderQasm2Version) {
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qubit[2] q;\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(parser.Failed(),
             "qubit[n] declaration should not parse under OPENQASM 2.0");
}

BOOST_AUTO_TEST_CASE(QASM3QubitDeclRejectedWithNoVersionDeclared) {
  const std::string qasmStr = "qubit[2] q;\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(parser.Failed(),
             "qubit[n] declaration should not parse with no OPENQASM "
             "version line present (isQasm3 defaults to false)");
}

BOOST_AUTO_TEST_CASE(QASM2StyleDeclarationsWorkUnderBothVersions) {
  for (const std::string &versionLine :
       {"OPENQASM 2.0;\n", "OPENQASM 3.0;\n"}) {
    const std::string qasmStr = versionLine +
                                "qreg q[2];\n"
                                "creg c[2];\n"
                                "h q[0];\n"
                                "cx q[0], q[1];\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage()
                                     << " (version line: " << versionLine
                                     << ")");
    BOOST_TEST_REQUIRE(circuit != nullptr);

    BOOST_TEST(circuit->GetQubits().size() == 2u,
               "Expected 2 qubits touched by the circuit under "
                   << versionLine << ", got " << circuit->GetQubits().size());
  }
}

// ****************************************************************************
// Task 3: measurement assignment (`c[0] = measure q[0];`) and braced
// conditionals (`if (c == 1) { ... }`).

namespace {

// A freshly-initialized simulator with its own statevector, so tests don't
// need to reason about whether Circuit::Execute resets the simulator (it
// doesn't - only the classical OperationState is reset automatically).
std::shared_ptr<Simulators::ISimulator> MakeInitializedSimulator(
    unsigned int nrQubits) {
  auto sim = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
  sim->AllocateQubits(nrQubits);
  sim->Initialize();
  return sim;
}

}  // namespace

BOOST_AUTO_TEST_CASE(QASM3MeasureAssignMatchesArrowMeasure) {
  // Prepare a known |1> state so the measurement outcome is deterministic,
  // then drive a conditional off the classical bit it writes - an
  // unprepared |0> qubit would pass even if the measurement assignment
  // silently dropped the measurement altogether.
  const std::string qasm3Str =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "c[0] = measure q[0];\n"
      "if (c==1) x q[0];\n";
  const std::string qasm2Str =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "if (c==1) x q[0];\n";

  qasm::QasmToCirc<> qasm3Parser;
  auto qasm3Circuit = qasm3Parser.ParseAndTranslate(qasm3Str);
  BOOST_TEST(!qasm3Parser.Failed(), qasm3Parser.GetErrorMessage());

  qasm::QasmToCirc<> qasm2Parser;
  auto qasm2Circuit = qasm2Parser.ParseAndTranslate(qasm2Str);
  BOOST_TEST(!qasm2Parser.Failed(), qasm2Parser.GetErrorMessage());

  BOOST_TEST_REQUIRE(!qasm3Parser.Failed());
  BOOST_TEST_REQUIRE(!qasm2Parser.Failed());

  auto qc3 = MakeInitializedSimulator(1);
  Circuits::OperationState state3(1);
  qasm3Circuit->Execute(qc3, state3);

  auto qc2 = MakeInitializedSimulator(1);
  Circuits::OperationState state2(1);
  qasm2Circuit->Execute(qc2, state2);

  for (size_t i = 0; i < 2; ++i) {
    BOOST_TEST(
        checkClose(std::complex<double>(qc3->Probability(i), 0.),
                   std::complex<double>(qc2->Probability(i), 0.), 0.0001),
        "Probability mismatch for state |" << i << ">: " << qc3->Probability(i)
                                           << " vs " << qc2->Probability(i));
  }

  // Both must have collapsed back to |0>: this only happens if the
  // measurement assignment actually populated c[0], letting the conditional
  // fire.
  BOOST_TEST(checkClose(std::complex<double>(qc3->Probability(0), 0.),
                        std::complex<double>(1., 0.), 0.0001),
             "Expected c[0] = measure q[0] to have observed 1 and the "
             "conditional to flip the qubit back to |0>, got P(|0>) = "
                 << qc3->Probability(0));
}

BOOST_AUTO_TEST_CASE(QASM3WholeRegisterMeasureMapsBitsInOrder) {
  // Only q[0] is prepared to |1>; a reversed or dropped whole-register
  // mapping would make c != 1, the conditional would not fire, and the
  // qubit would stay |1> instead of collapsing back to |0>.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "creg c[2];\n"
      "x q[0];\n"
      "c = measure q;\n"
      "if (c==1) x q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  auto qc = MakeInitializedSimulator(2);
  Circuits::OperationState state(2);
  circuit->Execute(qc, state);

  BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                        std::complex<double>(1., 0.), 0.0001),
             "Expected c = measure q to map q[0]->c[0], q[1]->c[1], firing "
             "the conditional and leaving |00>; got P(|00>) = "
                 << qc->Probability(0));
}

BOOST_AUTO_TEST_CASE(QASM3BracedSingleStatementMatchesUnbraced) {
  // Exercise both the taken (measurement observes 1) and untaken
  // (measurement observes 0) branches.
  for (const bool prepareQubit : {true, false}) {
    const std::string prep = prepareQubit ? "x q[0];\n" : "";

    const std::string qasm3Str =
        "OPENQASM 3.0;\n"
        "qreg q[1];\n"
        "creg c[1];\n" +
        prep +
        "measure q[0] -> c[0];\n"
        "if (c==1) { x q[0]; }\n";
    const std::string qasm2Str =
        "OPENQASM 2.0;\n"
        "qreg q[1];\n"
        "creg c[1];\n" +
        prep +
        "measure q[0] -> c[0];\n"
        "if (c==1) x q[0];\n";

    qasm::QasmToCirc<> qasm3Parser;
    auto qasm3Circuit = qasm3Parser.ParseAndTranslate(qasm3Str);
    BOOST_TEST(!qasm3Parser.Failed(), qasm3Parser.GetErrorMessage());

    qasm::QasmToCirc<> qasm2Parser;
    auto qasm2Circuit = qasm2Parser.ParseAndTranslate(qasm2Str);
    BOOST_TEST(!qasm2Parser.Failed(), qasm2Parser.GetErrorMessage());

    BOOST_TEST_REQUIRE(!qasm3Parser.Failed());
    BOOST_TEST_REQUIRE(!qasm2Parser.Failed());

    auto qc3 = MakeInitializedSimulator(1);
    Circuits::OperationState state3(1);
    qasm3Circuit->Execute(qc3, state3);

    auto qc2 = MakeInitializedSimulator(1);
    Circuits::OperationState state2(1);
    qasm2Circuit->Execute(qc2, state2);

    for (size_t i = 0; i < 2; ++i) {
      BOOST_TEST(
          checkClose(std::complex<double>(qc3->Probability(i), 0.),
                     std::complex<double>(qc2->Probability(i), 0.), 0.0001),
          "Braced vs unbraced conditional mismatch at |"
              << i << "> with prepareQubit=" << prepareQubit << ": "
              << qc3->Probability(i) << " vs " << qc2->Probability(i));
    }
  }
}

BOOST_AUTO_TEST_CASE(QASM3BracedMultipleStatementsAppliesAllBodyStatements) {
  // A braced body with two gates must become two conditioned statements, not
  // one. Compared against a twin that repeats the unbraced form once per
  // body statement, which is already-trusted behaviour.
  const std::string qasm3Str =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "if (c==1) { x q[0]; x q[1]; }\n";
  const std::string qasm2TwinStr =
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "if (c==1) x q[0];\n"
      "if (c==1) x q[1];\n";

  qasm::QasmToCirc<> qasm3Parser;
  auto qasm3Circuit = qasm3Parser.ParseAndTranslate(qasm3Str);
  BOOST_TEST(!qasm3Parser.Failed(), qasm3Parser.GetErrorMessage());

  qasm::QasmToCirc<> qasm2Parser;
  auto qasm2Circuit = qasm2Parser.ParseAndTranslate(qasm2TwinStr);
  BOOST_TEST(!qasm2Parser.Failed(), qasm2Parser.GetErrorMessage());

  BOOST_TEST_REQUIRE(!qasm3Parser.Failed());
  BOOST_TEST_REQUIRE(!qasm2Parser.Failed());

  auto qc3 = MakeInitializedSimulator(2);
  Circuits::OperationState state3(1);
  qasm3Circuit->Execute(qc3, state3);

  auto qc2 = MakeInitializedSimulator(2);
  Circuits::OperationState state2(1);
  qasm2Circuit->Execute(qc2, state2);

  const size_t nrStates = 1ULL << 2;
  for (size_t i = 0; i < nrStates; ++i) {
    BOOST_TEST(
        checkClose(std::complex<double>(qc3->Probability(i), 0.),
                   std::complex<double>(qc2->Probability(i), 0.), 0.0001),
        "Multi-statement braced conditional mismatch at |"
            << i << ">: " << qc3->Probability(i) << " vs "
            << qc2->Probability(i));
  }
}

BOOST_AUTO_TEST_CASE(QASM3MeasurementAndBracedConditionalRejectedUnderQasm2) {
  const std::string measureAssignUnderQasm2 =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "creg c[1];\n"
      "c[0] = measure q[0];\n";
  qasm::QasmToCirc<> measureParser;
  measureParser.ParseAndTranslate(measureAssignUnderQasm2);
  BOOST_TEST(measureParser.Failed(),
             "c[0] = measure q[0] should not parse under OPENQASM 2.0");

  const std::string bracedUnderQasm2 =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "creg c[1];\n"
      "if (c==1) { x q[0]; }\n";
  qasm::QasmToCirc<> bracedParser;
  bracedParser.ParseAndTranslate(bracedUnderQasm2);
  BOOST_TEST(bracedParser.Failed(),
             "braced if (c==1) { ... } should not parse under OPENQASM 2.0");
}

BOOST_AUTO_TEST_CASE(
    QASM3MeasurementAndBracedConditionalRejectedWithNoVersion) {
  const std::string measureAssignNoVersion =
      "qreg q[1];\n"
      "creg c[1];\n"
      "c[0] = measure q[0];\n";
  qasm::QasmToCirc<> measureParser;
  measureParser.ParseAndTranslate(measureAssignNoVersion);
  BOOST_TEST(measureParser.Failed(),
             "c[0] = measure q[0] should not parse with no OPENQASM version "
             "line present (isQasm3 defaults to false)");

  const std::string bracedNoVersion =
      "qreg q[1];\n"
      "creg c[1];\n"
      "if (c==1) { x q[0]; }\n";
  qasm::QasmToCirc<> bracedParser;
  bracedParser.ParseAndTranslate(bracedNoVersion);
  BOOST_TEST(bracedParser.Failed(),
             "braced if (c==1) { ... } should not parse with no OPENQASM "
             "version line present (isQasm3 defaults to false)");
}

BOOST_AUTO_TEST_CASE(QASM2MeasurementAndCondWorkUnderBothVersions) {
  for (const std::string &versionLine :
       {"OPENQASM 2.0;\n", "OPENQASM 3.0;\n"}) {
    const std::string qasmStr = versionLine +
                                "qreg q[1];\n"
                                "creg c[1];\n"
                                "x q[0];\n"
                                "measure q[0] -> c[0];\n"
                                "if (c==1) x q[0];\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage()
                                     << " (version line: " << versionLine
                                     << ")");
    BOOST_TEST_REQUIRE(circuit != nullptr);

    auto qc = MakeInitializedSimulator(1);
    Circuits::OperationState state(1);
    circuit->Execute(qc, state);

    BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                          std::complex<double>(1., 0.), 0.0001),
               "Expected arrow measurement + unbraced if to still collapse "
               "to |0> under "
                   << versionLine << ", got P(|0>) = " << qc->Probability(0));
  }
}

// ****************************************************************************
// Task 4: gate modifiers (`ctrl @`, `negctrl @`, `inv @`, `pow(k) @`).

namespace {

// Parses two programs and compares the resulting statevector probabilities.
// The modifier lowerings inject parameters (ctrl @ t becomes cp(pi/4)), so a
// gate-type assertion alone would not notice a wrong angle - only a numeric
// comparison against a directly written reference circuit does.
void CheckSameProbabilities(const std::string &qasmStr,
                            const std::string &referenceQasmStr,
                            unsigned int nrQubits) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  qasm::QasmToCirc<> referenceParser;
  auto referenceCircuit = referenceParser.ParseAndTranslate(referenceQasmStr);
  BOOST_TEST(!referenceParser.Failed(), referenceParser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!referenceParser.Failed());

  auto qc = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState state(nrQubits);
  circuit->Execute(qc, state);

  auto referenceQc = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState referenceState(nrQubits);
  referenceCircuit->Execute(referenceQc, referenceState);

  const size_t nrStates = 1ULL << nrQubits;
  for (size_t i = 0; i < nrStates; ++i) {
    const double prob = qc->Probability(i);
    const double referenceProb = referenceQc->Probability(i);

    BOOST_TEST(checkClose(std::complex<double>(prob, 0.),
                          std::complex<double>(referenceProb, 0.), 0.0001),
               "Probability mismatch for state |" << i << ">: " << prob
                                                  << " vs " << referenceProb
                                                  << "\nmodified:\n"
                                                  << qasmStr << "reference:\n"
                                                  << referenceQasmStr);
  }
}

// The complement of CheckSameProbabilities: asserts the two programs are
// distinguishable at all. Used to show that a parameter an equivalence test
// relies on is actually observable, so that equivalence is not vacuous.
void CheckDifferentProbabilities(const std::string &qasmStr,
                                 const std::string &otherQasmStr,
                                 unsigned int nrQubits) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  qasm::QasmToCirc<> otherParser;
  auto otherCircuit = otherParser.ParseAndTranslate(otherQasmStr);
  BOOST_TEST(!otherParser.Failed(), otherParser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!otherParser.Failed());

  auto qc = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState state(nrQubits);
  circuit->Execute(qc, state);

  auto otherQc = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState otherState(nrQubits);
  otherCircuit->Execute(otherQc, otherState);

  bool differs = false;
  const size_t nrStates = 1ULL << nrQubits;
  for (size_t i = 0; i < nrStates && !differs; ++i)
    differs =
        !checkClose(std::complex<double>(qc->Probability(i), 0.),
                    std::complex<double>(otherQc->Probability(i), 0.), 0.0001);

  BOOST_TEST(differs,
             "Expected these to be distinguishable by probability, "
             "so that an equivalence test on them can fail:\n"
                 << qasmStr << "versus:\n"
                 << otherQasmStr);
}

// Parses a program expected to be rejected and checks the error message names
// the offending construct, rather than failing with an opaque parse error.
void CheckRejectedWithMessage(const std::string &qasmStr,
                              const std::string &expectedFragment) {
  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(parser.Failed(), "Expected rejection of:\n" << qasmStr);
  if (!parser.Failed()) return;

  BOOST_TEST(
      parser.GetErrorMessage().find(expectedFragment) != std::string::npos,
      "Expected the error to mention '"
          << expectedFragment << "', got: " << parser.GetErrorMessage());
}

}  // namespace

BOOST_AUTO_TEST_CASE(UnmodifiedCallsAreUnchangedByTheModifierRestructure) {
  // `qop` now goes through `modifiedUop` (modifiers >> uop) instead of `uop`
  // directly; with an empty modifier list that must lower exactly as before.
  // The randomized round-trip tests above are the broad guard, this pins the
  // three uop shapes - cxgateCall, ugateCall and gatecall - explicitly.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[0];\n"
      "cx q[0], q[1];\n"
      "u(0.1, 0.2, 0.3) q[2];\n"
      "ccx q[0], q[1], q[2];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetNumberOfOperations() == 4u);
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kXGateType));
  BOOST_TEST(
      (NthGateType(circuit, 1) == Circuits::QuantumGateType::kCXGateType));
  BOOST_TEST(
      (NthGateType(circuit, 2) == Circuits::QuantumGateType::kUGateType));
  BOOST_TEST(
      (NthGateType(circuit, 3) == Circuits::QuantumGateType::kCCXGateType));
}

BOOST_AUTO_TEST_CASE(ModifiersRejectedUnderQasm2AndWithNoVersion) {
  // The modifier rules are gated on isQasm3, so under 2.0 (or with no version
  // line at all) `ctrl @ x ...` is not a gate call and the input is left
  // unparsed.
  for (const std::string &versionLine : {std::string("OPENQASM 2.0;\n"),
                                         std::string()}) {
    for (const std::string &modifier :
         {"ctrl @ x q[0], q[1];\n", "negctrl @ x q[0], q[1];\n",
          "inv @ s q[0];\n", "pow(2) @ x q[0];\n"}) {
      const std::string qasmStr = versionLine + "qreg q[2];\n" + modifier;

      qasm::QasmToCirc<> parser;
      parser.ParseAndTranslate(qasmStr);

      BOOST_TEST(parser.Failed(), "Expected '"
                                      << modifier
                                      << "' to be rejected without a QASM3 "
                                         "version line, program:\n"
                                      << qasmStr);
    }
  }
}

BOOST_AUTO_TEST_CASE(InvOnSelfInverseAndDaggerGates) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "inv @ s q[0];\n"
      "inv @ sdg q[0];\n"
      "inv @ t q[1];\n"
      "inv @ sx q[1];\n"
      "inv @ inv @ h q[2];\n"
      "inv @ cswap q[0], q[1], q[2];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetNumberOfOperations() == 6u);
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kSdgGateType));
  BOOST_TEST(
      (NthGateType(circuit, 1) == Circuits::QuantumGateType::kSGateType));
  BOOST_TEST(
      (NthGateType(circuit, 2) == Circuits::QuantumGateType::kTdgGateType));
  BOOST_TEST(
      (NthGateType(circuit, 3) == Circuits::QuantumGateType::kSxDagGateType));
  // inv @ inv @ g must cancel back to g, not to its dagger.
  BOOST_TEST((NthGateType(circuit, 4) ==
              Circuits::QuantumGateType::kHadamardGateType));
  BOOST_TEST(
      (NthGateType(circuit, 5) == Circuits::QuantumGateType::kCSwapGateType));
}

BOOST_AUTO_TEST_CASE(InvNegatesRotationAngle) {
  // The probe matters: after h then s the state is (|0> + i|1>)/sqrt(2), on
  // which rx(theta) and rx(-theta) give different probabilities. Starting
  // from |0> they would not, and the test could not fail.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "s q[0];\n"
      "inv @ rx(0.3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "s q[0];\n"
      "rx(-0.3) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(InvOnUSwapsAndNegatesPhiAndLambda) {
  // The inverse of u(theta, phi, lambda) is u(-theta, -lambda, -phi): phi and
  // lambda swap as well as being negated. The fourth parameter the gate
  // accepts is a global phase, which simply negates.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "inv @ u(0.7, 1.1, 0.3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "u(-0.7, -0.3, -1.1) q[0];\n",
      1);

  // ... and composing the gate with its inverse must be the identity, which
  // a merely-negated (unswapped) inverse would not be.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "u(0.7, 1.1, 0.3) q[0];\n"
      "inv @ u(0.7, 1.1, 0.3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(InvOnCuComposesToIdentity) {
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "h q[1];\n"
      "cu(0.7, 1.1, 0.3) q[0], q[1];\n"
      "inv @ cu(0.7, 1.1, 0.3) q[0], q[1];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "h q[1];\n",
      2);
}

BOOST_AUTO_TEST_CASE(InvRejectsGatesWithoutAnInverseInTheTable) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "inv @ k q[0];\n",
      "inv @ is not supported for the gate: k");
}

BOOST_AUTO_TEST_CASE(PowScalesTheAngleOfRotationGates) {
  // rx(theta)^s == rx(s * theta) exactly, so a rotation gate is scaled rather
  // than repeated - one gate out, whatever the exponent.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(2) @ rz(0.3) q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());
  BOOST_TEST(circuit->GetNumberOfOperations() == 1u);

  // rz only changes the relative phase, so it has to be probed between two
  // Hadamards for the doubled angle to be observable at all.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "pow(2) @ rz(0.3) q[0];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "rz(0.6) q[0];\n"
      "h q[0];\n",
      1);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(0.5) @ rx(0.4) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(0.2) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(PowRepeatsGatesWhoseAngleCannotBeScaled) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(3) @ x q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetNumberOfOperations() == 3u);
  for (size_t i = 0; i < 3; ++i)
    BOOST_TEST(
        (NthGateType(circuit, i) == Circuits::QuantumGateType::kXGateType));

  // Three X gates leave |1>, one leaves |1> too - so also check the count
  // above, not just the state.
  CheckSameProbabilities(qasmStr,
                         "OPENQASM 3.0;\n"
                         "qubit[1] q;\n"
                         "x q[0];\n",
                         1);
}

BOOST_AUTO_TEST_CASE(PowZeroEmitsNothing) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(0) @ x q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetNumberOfOperations() == 0u,
             "pow(0) @ x should emit nothing, got "
                 << circuit->GetNumberOfOperations() << " operation(s)");
}

BOOST_AUTO_TEST_CASE(PowWithNegativeExponentInvertsFirst) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "pow(-1) @ s q[0];\n"
      "pow(-2) @ t q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetNumberOfOperations() == 3u);
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kSdgGateType));
  BOOST_TEST(
      (NthGateType(circuit, 1) == Circuits::QuantumGateType::kTdgGateType));
  BOOST_TEST(
      (NthGateType(circuit, 2) == Circuits::QuantumGateType::kTdgGateType));

  // sdg followed by s is the identity, whereas the wrong lowering (s) would
  // compose to z and flip |+> to |->.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "pow(-1) @ s q[0];\n"
      "s q[0];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n",
      1);
}

BOOST_AUTO_TEST_CASE(PowRejectsFractionalExponentOnNonRotationGate) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(0.5) @ u(0.1, 0.2, 0.3) q[0];\n",
      "pow(k) @ with a fractional exponent is not supported for the gate: u");
}

BOOST_AUTO_TEST_CASE(CtrlLowersToTheNativeControlledGate) {
  // Every controlled form must be one native gate, not a multi-gate
  // expansion of the uncontrolled one.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "ctrl @ x q[0], q[1];\n"
      "ctrl @ y q[0], q[1];\n"
      "ctrl @ z q[0], q[1];\n"
      "ctrl @ h q[0], q[1];\n"
      "ctrl @ sx q[0], q[1];\n"
      "ctrl @ sxdg q[0], q[1];\n"
      "ctrl @ swap q[0], q[1], q[2];\n"
      "ctrl @ p(0.3) q[0], q[1];\n"
      "ctrl @ rx(0.3) q[0], q[1];\n"
      "ctrl @ ry(0.3) q[0], q[1];\n"
      "ctrl @ rz(0.3) q[0], q[1];\n"
      "ctrl @ u(0.1, 0.2, 0.3) q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  const std::vector<Circuits::QuantumGateType> expected = {
      Circuits::QuantumGateType::kCXGateType,
      Circuits::QuantumGateType::kCYGateType,
      Circuits::QuantumGateType::kCZGateType,
      Circuits::QuantumGateType::kCHGateType,
      Circuits::QuantumGateType::kCSxGateType,
      Circuits::QuantumGateType::kCSxDagGateType,
      Circuits::QuantumGateType::kCSwapGateType,
      Circuits::QuantumGateType::kCPGateType,
      Circuits::QuantumGateType::kCRxGateType,
      Circuits::QuantumGateType::kCRyGateType,
      Circuits::QuantumGateType::kCRzGateType,
      Circuits::QuantumGateType::kCUGateType};

  BOOST_TEST(circuit->GetNumberOfOperations() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    BOOST_TEST((NthGateType(circuit, i) == expected[i]),
               "Wrong gate type at position " << i);
}

BOOST_AUTO_TEST_CASE(CtrlOnPhaseGatesEqualsCpAtTheMatchingAngle) {
  // t == p(pi/4) and s == p(pi/2), so their controlled forms are cp at those
  // angles. The control has to be superposed and mapped back by a second
  // Hadamard, otherwise the injected phase is not observable at all and a
  // wrong angle would pass.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "ctrl @ t q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "cp(pi/4) q[0], q[1];\n"
      "h q[0];\n",
      2);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "ctrl @ s q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "cp(pi/2) q[0], q[1];\n"
      "h q[0];\n",
      2);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "ctrl @ tdg q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "cp(-pi/4) q[0], q[1];\n"
      "h q[0];\n",
      2);
}

BOOST_AUTO_TEST_CASE(TwoControlsReachCcxBothWaysAround) {
  for (const std::string &call :
       {"ctrl @ ctrl @ x q[0], q[1], q[2];\n", "ctrl @ cx q[0], q[1], q[2];\n"}) {
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qubit[3] q;\n" +
        call;

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    BOOST_TEST(circuit->GetNumberOfOperations() == 1u,
               "Expected a single native gate for " << call);
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kCCXGateType),
        "Expected ccx for " << call);

    CheckSameProbabilities(
        "OPENQASM 3.0;\n"
        "qubit[3] q;\n"
        "x q[0];\n"
        "x q[1];\n" +
            call,
        "OPENQASM 3.0;\n"
        "qubit[3] q;\n"
        "x q[0];\n"
        "x q[1];\n"
        "ccx q[0], q[1], q[2];\n",
        3);
  }
}

BOOST_AUTO_TEST_CASE(CtrlOnIdentityEmitsNothing) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "ctrl @ id q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetNumberOfOperations() == 0u);
}

BOOST_AUTO_TEST_CASE(CtrlRejectionsNameTheConstruct) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[4] q;\n"
      "ctrl @ ctrl @ ctrl @ x q[0], q[1], q[2], q[3];\n",
      "ctrl @ with more than two controls is not supported");

  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "gate mygate a { x a; }\n"
      "ctrl @ mygate q[0], q[1];\n",
      "Gate modifiers cannot be applied to the user-defined gate: mygate");

  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[4] q;\n"
      "ctrl @ ccx q[0], q[1], q[2], q[3];\n",
      "ctrl @ is not supported for the gate: ccx");
}

BOOST_AUTO_TEST_CASE(ModifierInsideGateBodyIsRejectedByName) {
  // Gate bodies keep using the plain `uop`, so a modifier there is a real
  // limitation - it has to be reported as one rather than left to surface as
  // an opaque leftover-input parse error.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "gate mygate a, b { ctrl @ x a, b; }\n"
      "mygate q[0], q[1];\n",
      "not supported inside a gate declaration body");

  // An unmodified gate body must keep working.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "gate mygate a, b { cx a, b; }\n"
      "mygate q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetNumberOfOperations() == 1u);
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kCXGateType));
}

// These three pin the exact-arity check in AddGateExpr (SyntaxTree.h:114-117
// for the no-parameter path, SyntaxTree.h:205-208 for the parameterised
// path). That check is what turns every malformed-arity case into a clean
// error instead of a silently wrong circuit, and this branch newly routes
// two shapes into it: `ugateCall`/`cxgateCall` fall through to `gatecall` on
// an over-long argument list (qasm.h:181,184), and `expGatecall`'s trailing
// qubit list is now optional, so a parameterised call with zero qubits
// reaches the same check. Both went error->error, so nothing broke, but if
// the arity check is ever relaxed to a modulus (broadcasting elsewhere in
// this file does use `%`, so that would look like a reasonable change),
// these silently start producing wrong circuits instead of failing to
// parse. These tests exist purely to fail loudly if that ever happens.
BOOST_AUTO_TEST_CASE(ArityCheckRejectsMismatchedGateCalls) {
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg q[3];\n"
      "cx q[0], q[1], q[2];\n",
      "requires exactly 2 qubits");

  // QASM3 header: `U` is an identifier only under QASM3's lexical rule, and
  // it is `gatecall` picking the over-long call up that produces the arity
  // message. Under QASM2 `U` is a keyword, so the same call is a syntax error
  // there - see BothCasesOfTheQasm2UAndCXBuiltinsStillWork.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "U(0.1, 0.2, 0.3) q[0], q[1];\n",
      "requires exactly 1 qubits");

  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rz(0.5);\n",
      "requires exactly 1 qubits");
}

BOOST_AUTO_TEST_CASE(NegCtrlEqualsTheXConjugatedTriple) {
  // The control is left in |0> so the negative control actually fires; with
  // it in |1> nothing may happen, which is the half that catches a negctrl
  // silently lowered as an ordinary ctrl.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "negctrl @ x q[0], q[1];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "x q[0];\n"
      "cx q[0], q[1];\n"
      "x q[0];\n",
      2);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "x q[0];\n"
      "negctrl @ x q[0], q[1];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "x q[0];\n",
      2);
}

BOOST_AUTO_TEST_CASE(NegCtrlNegatesOnlyItsOwnControlPosition) {
  // `ctrl @ negctrl @ x` fires on q[0] == 1 and q[1] == 0; getting the
  // negated position wrong would swap those conditions.
  const std::string modifiedCall = "ctrl @ negctrl @ x q[0], q[1], q[2];\n";
  const std::string referenceCall =
      "x q[1];\n"
      "ccx q[0], q[1], q[2];\n"
      "x q[1];\n";

  for (const std::string &preparation :
       {"", "x q[0];\n", "x q[1];\n", "x q[0];\nx q[1];\n"}) {
    CheckSameProbabilities(
        "OPENQASM 3.0;\n"
        "qubit[3] q;\n" +
            preparation + modifiedCall,
        "OPENQASM 3.0;\n"
        "qubit[3] q;\n" +
            preparation + referenceCall,
        3);
  }
}

BOOST_AUTO_TEST_CASE(NegCtrlOnPhaseGateEqualsCpAtTheMatchingAngle) {
  // The parameter-injecting lowering must survive the x conjugation too.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "negctrl @ t q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "x q[0];\n"
      "cp(pi/4) q[0], q[1];\n"
      "x q[0];\n"
      "h q[0];\n",
      2);
}

BOOST_AUTO_TEST_CASE(ModifiersUnderAConditionalDoNotDivideByZero) {
  // A modifier that lowers to nothing produces a statement with no gate type
  // and no gate declaration. The Uop path skips that shape; the CondUop path
  // used to fall through to the defined-gate branch and evaluate 0 % 0, which
  // raises SIGFPE and takes the process down instead of reporting an error.
  for (const std::string &call : {"pow(0) @ x q[0];\n", "ctrl @ id q[0], q[1];\n",
                                  "negctrl @ x q[0], q[1];\n",
                                  "ctrl @ x q[0], q[1];\n"}) {
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "creg c[1];\n"
        "if (c==1) " +
        call;

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage() << " for " << call);
    BOOST_TEST_REQUIRE(!parser.Failed());
    BOOST_TEST_REQUIRE(circuit != nullptr);
  }

  // The conditioned modifier must also still do the right thing when it
  // fires: c is measured as 1, so the negative control on |0> applies x.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "creg c[1];\n"
      "x q[2];\n"
      "measure q[2] -> c[0];\n"
      "if (c==1) negctrl @ x q[0], q[1];\n",
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[2];\n"
      "x q[1];\n",
      3);
}

BOOST_AUTO_TEST_CASE(PowRejectsAnExponentAboveTheRepetitionLimit) {
  // An unbounded exponent would overflow the cast to int - undefined
  // behaviour, so a silently wrong repetition count - and allocate before
  // emitting anything.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(3000000000) @ x q[0];\n",
      "pow(k) @ with an exponent above 1024 is not supported");

  // Also when the limit is only reached by accumulation.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(100) @ pow(100) @ x q[0];\n",
      "pow(k) @ with an exponent above 1024 is not supported");
}

BOOST_AUTO_TEST_CASE(CtrlAndInvCarryTheGlobalPhaseParameterOfU) {
  // The fourth parameter of u is a global phase, which is unobservable on its
  // own - but controlling the gate makes it a relative phase on the |1> block
  // of cu, where it is observable. Pin that it is carried through ctrl @ and
  // negated by inv @, and check first that it is observable at all, so the
  // equivalences below are not vacuous.
  CheckDifferentProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "cu(0.7, 1.1, 0.3, 0.9) q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "cu(0.7, 1.1, 0.3) q[0], q[1];\n"
      "h q[0];\n",
      2);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ u(0.7, 1.1, 0.3, 0.9) q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "cu(0.7, 1.1, 0.3, 0.9) q[0], q[1];\n"
      "h q[0];\n",
      2);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "cu(0.7, 1.1, 0.3, 0.9) q[0], q[1];\n"
      "inv @ cu(0.7, 1.1, 0.3, 0.9) q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n",
      2);
}

BOOST_AUTO_TEST_CASE(ModifiersAcceptTheSameGateAliasesAsPlainCalls) {
  // 'phase'/'cphase' are stdgates aliases the parser already accepts, and
  // u3/cu3 are the QASM2 spellings of u/cu; a modified call must not be
  // pickier about the spelling than an unmodified one.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "ctrl @ phase(0.3) q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "x q[1];\n"
      "ctrl @ p(0.3) q[0], q[1];\n"
      "h q[0];\n",
      2);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "inv @ u3(0.7, 1.1, 0.3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "inv @ u(0.7, 1.1, 0.3) q[0];\n",
      1);
}

// ****************************************************************************
// Task 5: QASM3 `input` declarations - free parameters bound to concrete
// values at parse time via ParseAndTranslate's params map. No symbolic
// parameter survives into the Circuit: an unbound one is a parse error.

namespace {

// Parses qasmStr with the given input bindings and compares the resulting
// probabilities against a reference program that spells the same value out as
// a literal instead of using `input`. Mirrors CheckSameProbabilities above,
// but only the first parse takes a params map.
void CheckBoundInputMatchesLiteral(
    const std::string &qasmStr,
    const std::unordered_map<std::string, double> &params,
    const std::string &referenceQasmStr, unsigned int nrQubits) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr, params);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  qasm::QasmToCirc<> referenceParser;
  auto referenceCircuit = referenceParser.ParseAndTranslate(referenceQasmStr);
  BOOST_TEST(!referenceParser.Failed(), referenceParser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!referenceParser.Failed());

  auto qc = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState state(nrQubits);
  circuit->Execute(qc, state);

  auto referenceQc = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState referenceState(nrQubits);
  referenceCircuit->Execute(referenceQc, referenceState);

  const size_t nrStates = 1ULL << nrQubits;
  for (size_t i = 0; i < nrStates; ++i) {
    const double prob = qc->Probability(i);
    const double referenceProb = referenceQc->Probability(i);

    BOOST_TEST(checkClose(std::complex<double>(prob, 0.),
                          std::complex<double>(referenceProb, 0.), 0.0001),
               "Probability mismatch for state |" << i << ">: " << prob
                                                  << " vs " << referenceProb
                                                  << "\nbound:\n"
                                                  << qasmStr << "reference:\n"
                                                  << referenceQasmStr);
  }
}

}  // namespace

BOOST_AUTO_TEST_CASE(BoundInputValueIsSubstitutedIntoGateParameter) {
  // rz is diagonal, so it changes no probabilities on its own - a dropped
  // substitution (angle silently 0) would still pass a bare rz(theta) probe.
  // Sandwiching it between two h gates folds the resulting relative phase
  // back into a probability difference: starting from (|0>+|1>)/sqrt(2),
  // rz(theta) then h leaves P(0) = cos^2(theta/2), which does depend on theta.
  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "rz(theta) q[0];\n"
      "h q[0];\n",
      {{"theta", 0.7}},
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "rz(0.7) q[0];\n"
      "h q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(BoundInputValueDiffersFromDefaultToCatchSilentZero) {
  // Same program, two different bindings for theta. If binding were ignored
  // (or silently defaulted to 0, the hole Task 4 found and fixed for a
  // different Variable path), both parses would produce the same circuit and
  // this would not be able to fail.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "rz(theta) q[0];\n"
      "h q[0];\n";

  qasm::QasmToCirc<> parserA;
  auto circuitA = parserA.ParseAndTranslate(qasmStr, {{"theta", 0.7}});
  BOOST_TEST(!parserA.Failed(), parserA.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parserA.Failed());

  qasm::QasmToCirc<> parserB;
  auto circuitB = parserB.ParseAndTranslate(qasmStr, {{"theta", 1.9}});
  BOOST_TEST(!parserB.Failed(), parserB.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parserB.Failed());

  auto qcA = MakeInitializedSimulator(1);
  Circuits::OperationState stateA(1);
  circuitA->Execute(qcA, stateA);

  auto qcB = MakeInitializedSimulator(1);
  Circuits::OperationState stateB(1);
  circuitB->Execute(qcB, stateB);

  bool differs = false;
  for (size_t i = 0; i < 2 && !differs; ++i)
    differs =
        !checkClose(std::complex<double>(qcA->Probability(i), 0.),
                    std::complex<double>(qcB->Probability(i), 0.), 0.0001);

  BOOST_TEST(differs,
             "Expected theta=0.7 and theta=1.9 to produce distinguishable "
             "probabilities, otherwise a silent-zero or ignored-map bug "
             "would pass undetected");
}

BOOST_AUTO_TEST_CASE(BoundInputWorksInsideAnExpression) {
  const double theta = 0.7;
  const double expectedAngle = 2 * theta + M_PI / 4;

  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "rz(2*theta + pi/4) q[0];\n"
      "h q[0];\n",
      {{"theta", theta}},
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "rz(" +
          std::to_string(expectedAngle) +
          ") q[0];\n"
          "h q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(UnboundInputFailsNamingTheVariable) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "rz(theta) q[0];\n");

  BOOST_TEST(parser.Failed());
  BOOST_TEST(circuit == nullptr);
  BOOST_TEST(parser.GetErrorMessage().find("theta") != std::string::npos,
             "Expected the error to name the unbound variable, got: "
                 << parser.GetErrorMessage());
}

BOOST_AUTO_TEST_CASE(GetInputsReturnsDeclaredNamesInOrder) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "input int[32] shots;\n"
      "qubit[1] q;\n",
      {{"theta", 0.5}, {"shots", 10.}});

  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  const std::vector<std::string> &inputs = parser.GetInputs();
  BOOST_TEST_REQUIRE(inputs.size() == 2u);
  BOOST_TEST(inputs[0] == "theta");
  BOOST_TEST(inputs[1] == "shots");
}

BOOST_AUTO_TEST_CASE(TopLevelInputDoesNotShadowGateFormalParameterOfSameName) {
  // theta is bound as a top-level input to 1.9 and is also the formal
  // parameter name of gate g, called with 0.2. If macro expansion ever
  // inherited the top-level map instead of building its own fresh one from
  // paramsDecl, the call site's 0.2 would be shadowed by the top-level 1.9,
  // and this would fail against the rz(0.2) reference.
  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "gate g(theta) a { rz(theta) a; }\n"
      "h q[0];\n"
      "g(0.2) q[0];\n"
      "h q[0];\n",
      {{"theta", 1.9}},
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "rz(0.2) q[0];\n"
      "h q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(SingleArgumentParseAndTranslateStillCompilesAndWorks) {
  // Backward compatibility: the pre-Task-5 one-argument call must still
  // compile and behave identically for a plain QASM2 program.
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "h q[0];\n");

  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());
  BOOST_TEST(parser.GetInputs().empty());

  auto qc = MakeInitializedSimulator(1);
  Circuits::OperationState state(1);
  circuit->Execute(qc, state);

  BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                        std::complex<double>(0.5, 0.), 0.0001));
  BOOST_TEST(checkClose(std::complex<double>(qc->Probability(1), 0.),
                        std::complex<double>(0.5, 0.), 0.0001));
}

BOOST_AUTO_TEST_CASE(InputDeclarationRejectedUnderQasm2) {
  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "input float[64] theta;\n");

  BOOST_TEST(parser.Failed(),
             "input declaration should not parse under OPENQASM 2.0");
}

BOOST_AUTO_TEST_CASE(MultipleInputTypesParse) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "input float[64] a;\n"
      "input int[32] b;\n"
      "input angle c;\n"
      "input bool d;\n"
      "qubit[1] q;\n",
      {{"a", 1.0}, {"b", 2.0}, {"c", 3.0}, {"d", 1.0}});

  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  const std::vector<std::string> &inputs = parser.GetInputs();
  BOOST_TEST_REQUIRE(inputs.size() == 4u);
  BOOST_TEST(inputs[0] == "a");
  BOOST_TEST(inputs[1] == "b");
  BOOST_TEST(inputs[2] == "c");
  BOOST_TEST(inputs[3] == "d");
}

// ****************************************************************************
// Follow-up to the review of the above: gate parameter expressions under a
// QASM3 call-site modifier (`ctrl @`, `negctrl @`, `inv @`, `pow(k) @`) must
// resolve a top-level `input` exactly as an unmodified call does.
// AddModifiedGateExpr's own rewriting path (Canonicalize) evaluated
// parameter expressions against a hardcoded empty map even after the plain
// delegation path (modifiers.empty()) was threaded with `variables` -
// `ctrl @ rz(theta) ...` is not an edge case, it is the common shape of a
// parameterised, controlled-rotation circuit.

BOOST_AUTO_TEST_CASE(CtrlModifierResolvesTopLevelInputInGateParameter) {
  // Phase-kickback probe, the same pattern CtrlAndInvCarryTheGlobalPhaseParameterOfU
  // above uses for u's global-phase parameter: a controlled-rz on a |0>
  // target changes no basis populations by itself, but with the control
  // qubit sandwiched in h ... h, the phase becomes an observable
  // interference pattern.
  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ rz(theta) q[0], q[1];\n"
      "h q[0];\n",
      {{"theta", 0.7}},
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ rz(0.7) q[0], q[1];\n"
      "h q[0];\n",
      2);
}

BOOST_AUTO_TEST_CASE(CtrlModifierInputDiffersFromDefaultToCatchIgnoredMap) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ rz(theta) q[0], q[1];\n"
      "h q[0];\n";

  qasm::QasmToCirc<> parserA;
  auto circuitA = parserA.ParseAndTranslate(qasmStr, {{"theta", 0.7}});
  BOOST_TEST(!parserA.Failed(), parserA.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parserA.Failed());

  qasm::QasmToCirc<> parserB;
  auto circuitB = parserB.ParseAndTranslate(qasmStr, {{"theta", 1.9}});
  BOOST_TEST(!parserB.Failed(), parserB.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parserB.Failed());

  auto qcA = MakeInitializedSimulator(2);
  Circuits::OperationState stateA(2);
  circuitA->Execute(qcA, stateA);

  auto qcB = MakeInitializedSimulator(2);
  Circuits::OperationState stateB(2);
  circuitB->Execute(qcB, stateB);

  bool differs = false;
  for (size_t i = 0; i < 4 && !differs; ++i)
    differs =
        !checkClose(std::complex<double>(qcA->Probability(i), 0.),
                    std::complex<double>(qcB->Probability(i), 0.), 0.0001);

  BOOST_TEST(differs,
             "Expected theta=0.7 and theta=1.9 under ctrl @ rz to produce "
             "distinguishable probabilities, otherwise an ignored map would "
             "pass undetected");
}

BOOST_AUTO_TEST_CASE(InvModifierResolvesTopLevelInputInGateParameter) {
  // The probe matters: after h then s the state is (|0> + i|1>)/sqrt(2), on
  // which rz(theta) and rz(-theta) give different probabilities - see
  // InvNegatesRotationAngle above.
  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "s q[0];\n"
      "inv @ rz(theta) q[0];\n",
      {{"theta", 0.7}},
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "s q[0];\n"
      "rz(-0.7) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(CtrlOnPGateResolvesTopLevelInputInGateParameter) {
  // p is diagonal like rz, so the target needs the control fixed at |1>
  // (via x) and the control qubit itself needs the h ... h sandwich to turn
  // the controlled phase into an observable population difference.
  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[2] q;\n"
      "x q[1];\n"
      "h q[0];\n"
      "ctrl @ p(theta) q[0], q[1];\n"
      "h q[0];\n",
      {{"theta", 0.7}},
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "x q[1];\n"
      "h q[0];\n"
      "cp(0.7) q[0], q[1];\n"
      "h q[0];\n",
      2);
}

BOOST_AUTO_TEST_CASE(UnboundInputUnderModifierStillFailsNamingTheVariable) {
  // The fix threads `inputValues` into Canonicalize; it must not turn an
  // unbound input into a silent zero for a modified call either.
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[2] q;\n"
      "ctrl @ rz(theta) q[0], q[1];\n");

  BOOST_TEST(parser.Failed());
  BOOST_TEST(circuit == nullptr);
  BOOST_TEST(parser.GetErrorMessage().find("theta") != std::string::npos,
             "Expected the error to name the unbound variable, got: "
                 << parser.GetErrorMessage());
}

// pow(k) @'s exponent is evaluated at grammar-parse time by
// MakePowModifierExpression, a different code path than Canonicalize's
// gate-parameter evaluation above. Threading std::ref(inputValues) into it
// (via the `powMod` rule) turned out to be a clean, same-shape change - see
// MakeIndexedIdExpression for the precedent of a two-argument phoenix functor
// with one templated and one grammar-member argument - so it was fixed too
// rather than merely pinned.

BOOST_AUTO_TEST_CASE(PowModifierResolvesTopLevelInputInExponent) {
  // rx is a rotation gate, so pow(k) scales its angle rather than repeating
  // the gate (see ApplyPow): pow(theta) @ rx(0.3) with theta=2.0 is rx(0.6).
  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "s q[0];\n"
      "pow(theta) @ rx(0.3) q[0];\n",
      {{"theta", 2.0}},
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "s q[0];\n"
      "rx(0.6) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(PowModifierInputDiffersFromDefaultToCatchIgnoredMap) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "s q[0];\n"
      "pow(theta) @ rx(0.3) q[0];\n";

  qasm::QasmToCirc<> parserA;
  auto circuitA = parserA.ParseAndTranslate(qasmStr, {{"theta", 2.0}});
  BOOST_TEST(!parserA.Failed(), parserA.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parserA.Failed());

  qasm::QasmToCirc<> parserB;
  auto circuitB = parserB.ParseAndTranslate(qasmStr, {{"theta", 3.0}});
  BOOST_TEST(!parserB.Failed(), parserB.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parserB.Failed());

  auto qcA = MakeInitializedSimulator(1);
  Circuits::OperationState stateA(1);
  circuitA->Execute(qcA, stateA);

  auto qcB = MakeInitializedSimulator(1);
  Circuits::OperationState stateB(1);
  circuitB->Execute(qcB, stateB);

  bool differs = false;
  for (size_t i = 0; i < 2 && !differs; ++i)
    differs =
        !checkClose(std::complex<double>(qcA->Probability(i), 0.),
                    std::complex<double>(qcB->Probability(i), 0.), 0.0001);

  BOOST_TEST(differs,
             "Expected theta=2.0 and theta=3.0 under pow @ to produce "
             "distinguishable probabilities, otherwise an ignored map would "
             "pass undetected");
}

// Task 6: unsupported-construct diagnostics. Each of the eight reserved
// keywords must be rejected with a message naming the actual construct,
// rather than Spirit's generic "unparsed input" error or a misleading
// "Unsupported gate" error from gatecall swallowing the keyword as a gate
// name.
BOOST_AUTO_TEST_CASE(UnsupportedConstructsAreRejectedByName) {
  const std::pair<std::string, std::string> cases[] = {
      {"for i in [0:3] { x q[0]; }\n", "'for' loops"},
      {"while (true) { x q[0]; }\n", "'while' loops"},
      {"def foo(qubit q) { x q; }\n", "subroutine definitions"},
      {"let alias = q;\n", "register aliases"},
      {"duration d = 300ns;\n", "duration declarations"},
      {"delay[300ns] q[0];\n", "delay instructions"},
      {"box { x q[0]; }\n", "box blocks"},
      {"array[int[32], 4] arr;\n", "array declarations"},
  };

  for (const auto &[construct, fragment] : cases) {
    const std::string qasmStr = "OPENQASM 3.0;\nqubit[1] q;\n" + construct;
    CheckRejectedWithMessage(qasmStr, fragment);
  }
}

// Placement (brief requirement 1): if `unsupportedConstruct` were tried after
// `qop` instead of before it, gatecall's identifier would swallow "for" as a
// gate name and this would fail with the generic
// "Unsupported gate without parameters: for" instead of naming the loop.
BOOST_AUTO_TEST_CASE(ForLoopIsNamedRatherThanReportedAsUnsupportedGate) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "for i in [0:3] { x q[0]; }\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(parser.Failed(), "Expected rejection of:\n" << qasmStr);
  if (!parser.Failed()) return;

  BOOST_TEST(
      parser.GetErrorMessage().find("'for' loops") != std::string::npos,
      "Expected the specific 'for' message, got: " << parser.GetErrorMessage());
  BOOST_TEST(
      parser.GetErrorMessage().find("Unsupported gate") == std::string::npos,
      "Placement regression: got the generic unsupported-gate message: "
          << parser.GetErrorMessage());
}

// Guard B (keyword boundary): a gate named "format", "delayed", "boxcar" or
// "arrayed" must still parse as an ordinary gate call under OPENQASM 3.0.
// Each of these is a defined gate whose name merely starts with one of the
// eight reserved keywords; without the `!qi::char_("a-zA-Z0-9_")` boundary
// guard, `unsupportedConstruct` would match the keyword prefix (e.g. "for"
// inside "format") and misreport these as the reserved construct.
BOOST_AUTO_TEST_CASE(IdentifierStartingWithReservedKeywordStillParses) {
  for (const std::string &gateName :
       {"format", "delayed", "boxcar", "arrayed"}) {
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n"
        "gate " +
        gateName + " a { x a; }\n" + gateName + " q[0];\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), "Expected '"
                                     << gateName
                                     << "' to parse as an ordinary gate "
                                        "name, got: "
                                     << parser.GetErrorMessage());
    if (!parser.Failed()) BOOST_TEST(circuit->GetNumberOfOperations() == 1u);
  }
}

// Guard A (version gating): "for" used as an ordinary gate name must still
// parse under OPENQASM 2.0, and with no version line at all - these words
// are only reserved once a program declares itself as QASM3. Without the
// `qi::eps(phx::ref(isQasm3))` guard, `unsupportedConstruct` would fire here
// too and break QASM2 programs that legitimately use these words.
BOOST_AUTO_TEST_CASE(ForAsOrdinaryIdentifierParsesUnderQasm2AndNoVersion) {
  for (const std::string &versionLine :
       {std::string("OPENQASM 2.0;\n"), std::string()}) {
    const std::string qasmStr = versionLine +
                                "qreg q[1];\n"
                                "gate for a { x a; }\n"
                                "for q[0];\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(),
               "Expected 'for' as a gate name to parse without a QASM3 "
               "version line, got: "
                   << parser.GetErrorMessage());
    if (!parser.Failed()) BOOST_TEST(circuit->GetNumberOfOperations() == 1u);
  }
}

// `input` is a supported QASM3 feature (Task 5); `unsupportedConstruct` must
// never reject it.
BOOST_AUTO_TEST_CASE(InputDeclarationIsNotTreatedAsUnsupportedConstruct) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "input float[64] theta;\n"
      "qubit[1] q;\n"
      "rx(theta) q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr, {{"theta", 0.5}});
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());
  BOOST_TEST(circuit->GetNumberOfOperations() == 1u);
}

// ****************************************************************************
// Task 7: QASM3 export mode - targeted emission assertions. The randomized
// round-trip cases above prove behavioural equivalence, but not that the
// emitted text is textually QASM3-conformant; a circuit could round-trip
// while still emitting subtly wrong spellings. These pin the exact tokens
// that must (and must not) appear.

namespace {

// A small fixed (non-random) circuit exercising a single-qubit gate, a
// two-qubit gate and a measurement, so both V2 and V3 emission paths touch
// register declarations, gate calls and the measurement statement.
std::shared_ptr<Circuits::Circuit<>> MakeFixedExportCircuit() {
  auto circuit = std::make_shared<Circuits::Circuit<>>();

  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kHadamardGateType, 0));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kCXGateType, 0, 1));
  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateMeasurement({{0, 0}, {1, 1}}));

  return circuit;
}

}  // namespace

BOOST_AUTO_TEST_CASE(V3ExportEmitsVersionLineAndStdgatesInclude) {
  const auto circuit = MakeFixedExportCircuit();

  const std::string qasmStr = qasm::CircToQasm<>::Generate(
      circuit, qasm::CircToQasm<>::QasmVersion::V3);

  BOOST_TEST(qasmStr.find("OPENQASM 3.0;\n") != std::string::npos);
  BOOST_TEST(qasmStr.find("include \"stdgates.inc\";\n") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(V3ExportEmitsQubitAndBitDeclarationsNotQregCreg) {
  const auto circuit = MakeFixedExportCircuit();

  const std::string qasmStr = qasm::CircToQasm<>::Generate(
      circuit, qasm::CircToQasm<>::QasmVersion::V3);

  BOOST_TEST(qasmStr.find("qubit[") != std::string::npos);
  BOOST_TEST(qasmStr.find("bit[") != std::string::npos);
  BOOST_TEST(qasmStr.find("qreg") == std::string::npos);
  BOOST_TEST(qasmStr.find("creg") == std::string::npos);
}

BOOST_AUTO_TEST_CASE(V3ExportEmitsMeasureAssignmentNotArrowForm) {
  const auto circuit = MakeFixedExportCircuit();

  const std::string qasmStr = qasm::CircToQasm<>::Generate(
      circuit, qasm::CircToQasm<>::QasmVersion::V3);

  BOOST_TEST(qasmStr.find("= measure") != std::string::npos);
  BOOST_TEST(qasmStr.find("->") == std::string::npos);
}

// Regression guard for the defaulted QasmVersion parameter added in Task 7:
// V2 output for a fixed circuit must be byte-identical to what it was before
// this change.
BOOST_AUTO_TEST_CASE(V2ExportIsByteIdenticalForFixedCircuit) {
  const auto circuit = MakeFixedExportCircuit();

  const std::string qasmStr = qasm::CircToQasm<>::Generate(circuit);

  const std::string expected =
      "OPENQASM 2.0;\n"
      "include \"qelib1.inc\";\n"
      "qreg q[2];\n"
      "creg c0[1];\n"
      "creg c1[1];\n"
      "h q[0];\n"
      "CX q[0],q[1];\n"
      "measure q[0]->c0[0];\n"
      "measure q[1]->c1[0];\n";

  BOOST_TEST(qasmStr == expected);
}

// ****************************************************************************
// Task 8: routed coverage gaps deferred from earlier reviews (Requirement 2).

BOOST_AUTO_TEST_CASE(
    QASM3StatementAfterBracedConditionalAppliesAfterNotBefore) {
  // Requirement 2a. Task 3 rewrote `statements` to splice a braced
  // conditional's body into the statement stream via
  // phx::insert(..., phx::end(qi::_val), ...); no existing test places a
  // statement *after* a braced conditional, so the splice position itself
  // was unverified.
  //
  // h q[1] followed by (fired) s;s == z followed by h q[1] again takes
  // |0> -> |+> -> |-> -> |1>, deterministically. That final h has to land
  // strictly after the body for this to hold:
  //  - if it were dropped entirely, the result would be |-> (P(|1>) = 0.5).
  //  - if it were spliced *before* the conditional instead of after, the two
  //    h's would cancel first (|0> -> |+> -> |0>), and z on |0> is a
  //    phase-only no-op, leaving P(|1>) = 0 instead of 1.
  const std::string bracedQasm =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "h q[1];\n"
      "if (c==1) { s q[1]; s q[1]; }\n"
      "h q[1];\n";
  const std::string unbracedTwin =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "h q[1];\n"
      "if (c==1) s q[1];\n"
      "if (c==1) s q[1];\n"
      "h q[1];\n";
  CheckSameProbabilities(bracedQasm, unbracedTwin, 2);

  // Sensitivity check, proving the equivalence above is not vacuous: moving
  // the trailing 'h q[1]' to *before* the conditional changes the outcome
  // (see comment above).
  const std::string wrongOrderTwin =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "h q[1];\n"
      "h q[1];\n"
      "if (c==1) s q[1];\n"
      "if (c==1) s q[1];\n";
  CheckDifferentProbabilities(bracedQasm, wrongOrderTwin, 2);
}

BOOST_AUTO_TEST_CASE(QASM3EmptyBracedConditionalBodyIsANoOp) {
  // Requirement 2b. `if (c == 1) { }` is defensible as a no-op but was
  // undocumented and untested - nothing distinguished "accepted as a no-op"
  // from "body silently dropped, and possibly the rest of the program with
  // it". Pin the current behaviour: the program still parses and the
  // surrounding statements still execute normally.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "if (c==1) { }\n"
      "x q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  auto qc = MakeInitializedSimulator(1);
  Circuits::OperationState state(1);
  circuit->Execute(qc, state);

  // x; measure; (empty if, no-op); x -> back to |0>.
  BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                        std::complex<double>(1., 0.), 0.0001),
             "Expected the empty braced conditional to be a no-op and the "
             "trailing x q[0] to still apply, got P(|0>) = "
                 << qc->Probability(0));
}

BOOST_AUTO_TEST_CASE(CuParsesUnderQasm2EvenThoughNotAQelib1Gate) {
  // Requirement 2c. Task 4 added "cu" to allowedMultipleParamsGates, a
  // version-agnostic set, so `cu(theta,phi,lambda)` now parses under
  // `OPENQASM 2.0;` even though 'cu' is not a qelib1.inc gate ('cu3' is).
  // Pin the current behaviour so a future tightening back to qelib1-only
  // gate names under QASM2 is a deliberate choice, not an accidental
  // regression.
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "cu(0.7, 1.1, 0.3) q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());
  BOOST_TEST(circuit->GetNumberOfOperations() == 1u);
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kCUGateType));
}

// ****************************************************************************
// Task 8: Qiskit-dialect fixtures (Requirement 3). Qiskit's qasm3.dumps()
// emits `OPENQASM 3.0;` + `include "stdgates.inc";`, qubit[n]/bit[n]
// declarations, `c[i] = measure q[i];` assignment measurement, lowercase
// canonical gate names, `gate` declarations, and `barrier`.

BOOST_FIXTURE_TEST_CASE(QiskitDialectPlainCircuitMatchesQasm2Equivalent,
                        QasmTestFixture) {
  // Deliberately deterministic (x, not h, seeds the register): measurement
  // is a random collapse, and comparing two *independently* executed
  // simulators' post-measurement probabilities is only meaningful if the
  // pre-measurement state is a computational basis state - otherwise the two
  // Execute() calls draw from the RNG at different points and can land on
  // different (equally valid) outcomes, an apparent mismatch that is a test
  // artifact, not a parser bug. rz/sdg are phase-only in the Z basis, so they
  // do not disturb the deterministic outcome and are kept for gate-name
  // coverage.
  const std::string qiskitStr =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "qubit[3] q;\n"
      "bit[3] c;\n"
      "x q[0];\n"
      "cx q[0], q[1];\n"
      "cx q[1], q[2];\n"
      "rz(0.4) q[2];\n"
      "sdg q[0];\n"
      "barrier q;\n"
      "c[0] = measure q[0];\n"
      "c[1] = measure q[1];\n"
      "c[2] = measure q[2];\n";
  const std::string qasm2Str =
      "OPENQASM 2.0;\n"
      "include \"qelib1.inc\";\n"
      "qreg q[3];\n"
      "creg c[3];\n"
      "x q[0];\n"
      "cx q[0], q[1];\n"
      "cx q[1], q[2];\n"
      "rz(0.4) q[2];\n"
      "sdg q[0];\n"
      "measure q[0] -> c[0];\n"
      "measure q[1] -> c[1];\n"
      "measure q[2] -> c[2];\n";

  qasm::QasmToCirc<> qiskitParser;
  auto qiskitCircuit = qiskitParser.ParseAndTranslate(qiskitStr);
  BOOST_TEST(!qiskitParser.Failed(), qiskitParser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!qiskitParser.Failed());

  qasm::QasmToCirc<> qasm2Parser;
  auto qasm2Circuit = qasm2Parser.ParseAndTranslate(qasm2Str);
  BOOST_TEST(!qasm2Parser.Failed(), qasm2Parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!qasm2Parser.Failed());

  auto qc1 = MakeInitializedSimulator(3);
  Circuits::OperationState state1(3);
  qiskitCircuit->Execute(qc1, state1);

  auto qc2 = MakeInitializedSimulator(3);
  Circuits::OperationState state2(3);
  qasm2Circuit->Execute(qc2, state2);

  const size_t nrStates = 1ULL << 3;
  for (size_t i = 0; i < nrStates; ++i) {
    BOOST_TEST(
        checkClose(std::complex<double>(qc1->Probability(i), 0.),
                   std::complex<double>(qc2->Probability(i), 0.), 0.0001),
        "Probability mismatch for state |" << i << ">: " << qc1->Probability(i)
                                           << " vs " << qc2->Probability(i));
  }
}

BOOST_AUTO_TEST_CASE(
    QiskitDialectCustomGateDeclarationMatchesInlinedEquivalent) {
  // No measurement here: the custom 'bell' gate produces a genuine
  // superposition, and (as noted above) comparing collapsed measurement
  // outcomes across two independently-executed simulators is not a valid
  // equivalence check. CheckSameProbabilities compares the pre-measurement
  // amplitudes directly instead, which is deterministic.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "gate bell a, b {\n"
      "  h a;\n"
      "  cx a, b;\n"
      "}\n"
      "qubit[2] q;\n"
      "bell q[0], q[1];\n",
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "cx q[0], q[1];\n",
      2);
}

BOOST_AUTO_TEST_CASE(
    QiskitDialectInputWithCtrlModifierMatchesLiteralEquivalent) {
  // The most realistic parameterised-circuit shape: an `input` used inside a
  // call-site modifier, the combination Task 5's fix specifically enabled.
  CheckBoundInputMatchesLiteral(
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "input float[64] theta;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ rz(theta) q[0], q[1];\n"
      "h q[0];\n",
      {{"theta", 0.7}},
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ rz(0.7) q[0], q[1];\n"
      "h q[0];\n",
      2);
}

// ****************************************************************************
// Task 8: deliberately awkward hand-written QASM3 (Requirement 4). Real
// hand-written input is messier than generated output: comments interleaved
// between statements (including immediately before/after a braced
// conditional) and irregular whitespace/newlines mid-statement.

BOOST_AUTO_TEST_CASE(
    CommentsAndIrregularWhitespaceAroundBracedConditionalParse) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[2];\n"
      "creg c[1];\n"
      "// prepare q[0] before measuring\n"
      "x    q[0]\n"
      "  ;\n"
      "measure q[0]\n"
      "   -> c[0];\n"
      "\n"
      "// comment right before the braced conditional\n"
      "if (c==1)\n"
      "{\n"
      "    x\n"
      "      q[1]\n"
      "      ;\n"
      "}\n"
      "// comment right after the braced conditional\n"
      "x q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  // x q[0]; measure; conditional x q[1]; trailing x q[1] - four operations,
  // none dropped by the surrounding comments/whitespace.
  BOOST_TEST(circuit->GetNumberOfOperations() == 4u,
             "Expected 4 operations, got " << circuit->GetNumberOfOperations());
}

BOOST_AUTO_TEST_CASE(
    UnderscoreAndUppercaseIdentifiersWithQregCregAndBothMeasurementForms) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg _Q[2];\n"
      "creg _C[2];\n"
      "gate MyGate a { x a; }\n"
      "MyGate _Q[0];\n"
      "measure _Q[0] -> _C[0];\n"
      "_C[1] = measure _Q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST(circuit->GetQubits().size() == 2u);
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kXGateType));
}

// ****************************************************************************
// Gap fix: real Qiskit qasm3.dumps() output uses four conditional spellings
// -  `if (c == 2) { ... }` (already supported), `if (c[0]) { ... }` (bare
// bit), `if (!c[0]) { ... }` (negated bit), and an `else` clause - but only
// the register-comparison form parsed before this fix. Verified against
// Qiskit 2.5.2.

// The three fixtures below are pasted verbatim from Qiskit 2.5.2's
// `qasm3.dumps()` output (confirmed against a real Qiskit installation, not
// paraphrased), including that Qiskit emits `bit[2] c;` before `qubit[2] q;`
// - classical registers first, the opposite order every other fixture in
// this file uses. Each is only a parse check here (the body is `h q[0];`
// before the measurement, so the branch taken is not deterministic and
// proves nothing about behaviour); the deterministic behavioural checks
// below use separately-crafted, bit-prepared QASM.

BOOST_AUTO_TEST_CASE(QASM3BareBitConditionFixtureFromQiskitParses) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "bit[2] c;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "c[0] = measure q[0];\n"
      "if (c[0]) {\n"
      "  x q[1];\n"
      "}\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
}

BOOST_AUTO_TEST_CASE(QASM3NegatedBitConditionFixtureFromQiskitParses) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "bit[2] c;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "c[0] = measure q[0];\n"
      "if (!c[0]) {\n"
      "  x q[1];\n"
      "}\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
}

BOOST_AUTO_TEST_CASE(QASM3BitConditionWithElseFixtureFromQiskitParses) {
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "bit[2] c;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "c[0] = measure q[0];\n"
      "if (c[0]) {\n"
      "  x q[1];\n"
      "} else {\n"
      "  z q[1];\n"
      "}\n";

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
}

BOOST_AUTO_TEST_CASE(QASM3BareBitConditionFiresExactlyWhenBitIsOne) {
  // Prepare c[0] deterministically (x then measure) so the branch taken is
  // known, and exercise both the taken and untaken case - an unprepared |0>
  // qubit would pass even if `if (c[0])` were silently a no-op.
  //
  // Circuits::ISimulator::Probability(outcome) takes a *joint* basis-state
  // index over all allocated qubits (bit i of outcome is qubit i, q[0]
  // being the least significant bit - see
  // QASM3WholeRegisterMeasureMapsBitsInOrder above), not a per-qubit
  // marginal, so the expected outcome is computed as q[0] + 2*q[1] from the
  // two qubits' known final values rather than read off a single qubit.
  for (const bool prepareBit : {true, false}) {
    const std::string prep = prepareBit ? "x q[0];\n" : "";
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qreg q[2];\n"
        "creg c[1];\n" +
        prep +
        "measure q[0] -> c[0];\n"
        "if (c[0]) { x q[1]; }\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    auto qc = MakeInitializedSimulator(2);
    Circuits::OperationState state(2);
    circuit->Execute(qc, state);

    // q[0] keeps whatever it was prepared to (measurement doesn't disturb a
    // basis state); q[1] flips to 1 iff c[0] was 1, i.e. iff prepareBit.
    const int q0 = prepareBit ? 1 : 0;
    const int q1 = prepareBit ? 1 : 0;
    const int expectedOutcome = q0 + 2 * q1;
    BOOST_TEST(
        checkClose(std::complex<double>(qc->Probability(expectedOutcome), 0.),
                   std::complex<double>(1., 0.), 0.0001),
        "if (c[0]) with prepareBit="
            << prepareBit << " gave P(|" << expectedOutcome
            << ">) = " << qc->Probability(expectedOutcome) << ", expected 1");
  }
}

BOOST_AUTO_TEST_CASE(QASM3NegatedBitConditionFiresExactlyWhenBitIsZero) {
  // The exact inverse of the bare-bit test above: `if (!c[0])` must fire
  // when the bit is 0 and stay quiet when it's 1 - the opposite polarity of
  // the bare form - so a swapped polarity in the implementation makes this
  // fail while the bare-bit test above still passes. Same joint-outcome
  // indexing convention as above (q[0] + 2*q[1]).
  for (const bool prepareBit : {true, false}) {
    const std::string prep = prepareBit ? "x q[0];\n" : "";
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qreg q[2];\n"
        "creg c[1];\n" +
        prep +
        "measure q[0] -> c[0];\n"
        "if (!c[0]) { x q[1]; }\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    auto qc = MakeInitializedSimulator(2);
    Circuits::OperationState state(2);
    circuit->Execute(qc, state);

    // q[1] flips to 1 iff c[0] was 0, i.e. iff prepareBit was false - the
    // inverse mapping of the bare-bit test above.
    const int q0 = prepareBit ? 1 : 0;
    const int q1 = prepareBit ? 0 : 1;
    const int expectedOutcome = q0 + 2 * q1;
    BOOST_TEST(
        checkClose(std::complex<double>(qc->Probability(expectedOutcome), 0.),
                   std::complex<double>(1., 0.), 0.0001),
        "if (!c[0]) with prepareBit="
            << prepareBit << " gave P(|" << expectedOutcome
            << ">) = " << qc->Probability(expectedOutcome) << ", expected 1");
  }
}

BOOST_AUTO_TEST_CASE(QASM3BitConditionElseClauseFiresExactlyOneBranch) {
  // Each branch targets a different qubit so both are independently
  // observable in the final statevector regardless of which one fires -
  // conditioning both branches' effect on the same qubit (e.g. x vs z on a
  // |0>) would make an untaken z-branch indistinguishable from a dropped
  // else clause, since z|0> == |0>. Joint-outcome index is
  // q[0] + 2*q[1] + 4*q[2].
  for (const bool prepareBit : {true, false}) {
    const std::string prep = prepareBit ? "x q[0];\n" : "";
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qreg q[3];\n"
        "creg c[1];\n" +
        prep +
        "measure q[0] -> c[0];\n"
        "if (c[0]) { x q[1]; } else { x q[2]; }\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    auto qc = MakeInitializedSimulator(3);
    Circuits::OperationState state(3);
    circuit->Execute(qc, state);

    // if-branch (q[1]) fires iff the bit is 1; else-branch (q[2]) fires iff
    // the bit is 0 - exactly one of the two, matching prepareBit.
    const int q0 = prepareBit ? 1 : 0;
    const int q1 = prepareBit ? 1 : 0;
    const int q2 = prepareBit ? 0 : 1;
    const int expectedOutcome = q0 + 2 * q1 + 4 * q2;
    BOOST_TEST(
        checkClose(std::complex<double>(qc->Probability(expectedOutcome), 0.),
                   std::complex<double>(1., 0.), 0.0001),
        "if/else with prepareBit="
            << prepareBit << " gave P(|" << expectedOutcome
            << ">) = " << qc->Probability(expectedOutcome) << ", expected 1");
  }
}

BOOST_AUTO_TEST_CASE(QASM3ElseOnRegisterComparisonConditionIsRejected) {
  // `if (c == 2) { ... } else { ... }` would need a not-equal condition to
  // express the else-branch, and Circuit/Factory.h's CircuitFactory only
  // offers CreateEqualCondition - no not-equal primitive exists - so this
  // must be rejected with an error naming the construct, not silently
  // mishandled or left to a generic parse failure.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "creg c[2];\n"
      "if (c == 2) { x q[0]; } else { z q[0]; }\n";

  CheckRejectedWithMessage(qasmStr, "register-comparison condition");
}

BOOST_AUTO_TEST_CASE(
    QASM3RegisterComparisonConditionalStillWorksAfterBitFormAdded) {
  // Regression: adding the bit-form condHead alternative and the optional
  // else-clause to condOpBraced must not disturb the pre-existing
  // register-comparison form when else is absent.
  for (const bool prepareQubit : {true, false}) {
    const std::string prep = prepareQubit ? "x q[0];\n" : "";
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qreg q[1];\n"
        "creg c[1];\n" +
        prep +
        "measure q[0] -> c[0];\n"
        "if (c==1) { x q[0]; }\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    auto qc = MakeInitializedSimulator(1);
    Circuits::OperationState state(1);
    circuit->Execute(qc, state);

    // x q[0] then measure then "if (c==1) x q[0]" toggles q[0] back to |0>
    // whenever it was prepared to |1>, and leaves it at |0> otherwise -
    // either way q[0] ends at |0>.
    BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                          std::complex<double>(1., 0.), 0.0001),
               "Register-comparison braced conditional regressed with "
               "prepareQubit="
                   << prepareQubit << ", got P(|0>) = " << qc->Probability(0));
  }
}

BOOST_AUTO_TEST_CASE(QASM2UnbracedConditionalStillWorksAfterBitFormAdded) {
  // Regression: the QASM2 unbraced condOp path is untouched by this fix (it
  // still only supports the register-comparison form), so it must keep
  // working exactly as before.
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "creg c[1];\n"
      "x q[0];\n"
      "measure q[0] -> c[0];\n"
      "if (c==1) x q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  auto qc = MakeInitializedSimulator(1);
  Circuits::OperationState state(1);
  circuit->Execute(qc, state);

  BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                        std::complex<double>(1., 0.), 0.0001),
             "QASM2 unbraced conditional regressed, got P(|0>) = "
                 << qc->Probability(0));
}

// ****************************************************************************
// Conformance gaps against the official OpenQASM 3 ANTLR grammar
// (openqasm/openqasm, source/grammar/qasm3Parser.g4).
//
// A1: '^' is bitwise XOR in QASM3 and exponentiation in QASM2; '**' is
//     exponentiation in QASM3. Keeping the QASM2 meaning unconditionally
//     turned `rx(2 ^ 3)` into rx(8) where the spec says rx(1) - a wrong
//     rotation angle with no error at all, which is why these probe the angle
//     numerically rather than asserting on the parse.

BOOST_AUTO_TEST_CASE(QASM3CaretIsBitwiseXorNotPower) {
  // 2 ^ 3 is 1 under QASM3 (bitwise XOR) and 8 under QASM2 (power). rx's
  // angle is directly observable from |0>: P(|0>) = cos^2(theta/2), which is
  // 0.770 at theta=1 and 0.427 at theta=8 - far apart, so a wrong angle
  // cannot hide inside the comparison tolerance.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(1) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(QASM3CaretIsNotThePowerItUsedToBe) {
  // The complement of the above, and the test that actually pins the bug: if
  // '^' were still exponentiation under QASM3, rx(2 ^ 3) would be rx(8) and
  // this would fail. It also shows rx(1) and rx(8) are distinguishable at
  // all, so the equivalence asserted above is not vacuous.
  CheckDifferentProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(8) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(QASM3DoubleAsteriskIsPower) {
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(2 ** 3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(8) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(QASM3DoubleAsteriskIsRightAssociative) {
  // <assoc=right> on DOUBLE_ASTERISK: 2 ** 3 ** 2 is 2 ** 9 = 512, not
  // (2 ** 3) ** 2 = 64. Both are large enough that rx wraps around several
  // times, so they are compared against the literal each would produce
  // rather than against each other by magnitude.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(2 ** 3 ** 2) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(512) q[0];\n",
      1);

  CheckDifferentProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(2 ** 3 ** 2) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(64) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(QASM2CaretIsStillPower) {
  // The regression guard for A1: QASM2 has no '**' and spells exponentiation
  // '^', so nothing about its expressions may change. 2 ^ 3 stays 8.
  CheckSameProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(8) q[0];\n",
      1);

  CheckDifferentProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(1) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(QASM3XorBindsLooserThanAddition) {
  // In the official expression grammar bitwiseXorExpression sits below the
  // additive operators, so `1 + 1 ^ 3` is (1 + 1) ^ 3 = 2 XOR 3 = 1, not
  // 1 + (1 ^ 3) = 3. That is the opposite of where QASM2's '^' sat (tighter
  // than '*'), so the precedence move is part of the fix, not a detail.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(1 + 1 ^ 3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(1) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(QASM3XorRejectsNonIntegerOperands) {
  // XOR is only defined on integers. Truncating 0.5 to 0 would produce a
  // wrong angle silently, which is the very failure mode this group is
  // closing, so a non-integral operand is an error naming the operator.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(0.5 ^ 1) q[0];\n",
      "requires integer operands");
}

BOOST_AUTO_TEST_CASE(QASM2PowerStillAcceptsFractionalOperands) {
  // The integer restriction belongs to XOR only - QASM2's '^' is still a
  // real-valued power, so 4 ^ 0.5 is 2 there.
  CheckSameProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(4 ^ 0.5) q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(2) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(QASM3PowModifierIsUnaffectedByTheCaretChange) {
  // `pow(k) @` is a gate modifier, a separate mechanism from the '^'/'**'
  // expression operators, and must keep working exactly as it did.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "pow(2) @ rx(0.3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(0.6) q[0];\n",
      1);
}

// ****************************************************************************
// A2: `simpleGatecall` duplicated its `identifier` prefix across two
// alternatives. Qi does not clear a std::string attribute when it backtracks
// out of a failed alternative, so the identifier accumulated: `x() q[0];` was
// reported as a call to gate "xx". Empty parentheses are also spec-legal
// (`gateCallStatement: ... (LPAREN expressionList? RPAREN)? ...`) and were
// rejected outright.

BOOST_AUTO_TEST_CASE(EmptyParameterListParsesAsTheSameCall) {
  // h to make the x observable as a probability change rather than only as a
  // basis relabelling, and a second qubit left alone as a control.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "x() q[0];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "x q[0];\n"
      "h q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(EmptyParameterListParsesUnderQasm2Too) {
  // Empty parens are legal QASM2 as well, and the doubling bug predates the
  // QASM3 work entirely - so the fix has to hold on both sides.
  CheckSameProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "cx() q[0], q[1];\n",
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "cx q[0], q[1];\n",
      2);
}

BOOST_AUTO_TEST_CASE(ParameterisedCallIsUnaffectedByTheEmptyParenSupport) {
  // The optional empty-paren group consumes '(' before discovering the ')'
  // is missing; if it did not restore the position, every parameterised call
  // would break. rx(0.3) from |0> leaves P(|0>) = cos^2(0.15).
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(0.3) q[0];\n");
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  auto qc = MakeInitializedSimulator(1);
  Circuits::OperationState state(1);
  circuit->Execute(qc, state);

  const double expected = std::cos(0.15) * std::cos(0.15);
  BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                        std::complex<double>(expected, 0.), 0.0001),
             "rx(0.3) gave P(|0>) = " << qc->Probability(0) << ", expected "
                                      << expected);
}

BOOST_AUTO_TEST_CASE(UnsupportedEmptyParenCallNamesTheGateOnce) {
  // The doubling was only ever visible through the error message, so that is
  // where it is pinned: the name must appear exactly as written.
  for (const std::string gateName : {std::string("nosuchgate"),
                                     std::string("rx")}) {
    qasm::QasmToCirc<> parser;
    parser.ParseAndTranslate(
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n" +
        gateName + "() q[0];\n");

    BOOST_TEST(parser.Failed(),
               "Expected " << gateName << "() q[0]; to be rejected");
    if (!parser.Failed()) continue;

    const std::string &message = parser.GetErrorMessage();
    BOOST_TEST(message.find(gateName + gateName) == std::string::npos,
               "Gate name doubled in the error message: " << message);
    BOOST_TEST(message.find(gateName) != std::string::npos,
               "Expected the error to name the gate, got: " << message);
  }
}

// ****************************************************************************
// B1: `measureArrowAssignmentStatement: measureExpression (ARROW
// indexedIdentifier)? SEMICOLON` - the arrow target is optional. The IR has no
// discard-measurement (CreateMeasurement takes qubit/classical-bit pairs), so
// this is a clean rejection naming the construct rather than an invented
// classical bit. What must not survive is the old misleading message, which
// blamed an "unsupported gate" called measure.

BOOST_AUTO_TEST_CASE(MeasurementWithoutTargetIsRejectedNamingTheConstruct) {
  for (const std::string qasmStr :
       {std::string("OPENQASM 3.0;\n"
                    "qubit[1] q;\n"
                    "measure q[0];\n"),
        std::string("OPENQASM 3.0;\n"
                    "qubit[2] q;\n"
                    "measure q;\n"),
        std::string("OPENQASM 2.0;\n"
                    "qreg q[1];\n"
                    "measure q[0];\n")}) {
    CheckRejectedWithMessage(qasmStr, "without a classical target");

    qasm::QasmToCirc<> parser;
    parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(
        parser.GetErrorMessage().find("Unsupported gate") == std::string::npos,
        "Expected the measurement to be named, not reported as an "
        "unsupported gate: "
            << parser.GetErrorMessage());
  }
}

BOOST_AUTO_TEST_CASE(MeasurementWithATargetStillWorks) {
  // Regression guard for the rule inserted ahead of the gate-call
  // fall-through: both spellings that do name a classical bit must be
  // untouched by it.
  for (const std::string measurement :
       {std::string("measure q[0] -> c[0];\n"),
        std::string("c[0] = measure q[0];\n")}) {
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n"
        "bit[1] c;\n"
        "x q[0];\n" +
        measurement + "if (c[0]) { x q[0]; }\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    auto qc = MakeInitializedSimulator(1);
    Circuits::OperationState state(1);
    circuit->Execute(qc, state);

    BOOST_TEST(checkClose(std::complex<double>(qc->Probability(0), 0.),
                          std::complex<double>(1., 0.), 0.0001),
               "Measurement regressed for:\n"
                   << qasmStr);
  }
}

// ****************************************************************************
// B2: `barrierStatement: BARRIER gateOperandList? SEMICOLON` - the operand
// list is optional and a bare `barrier;` covers every qubit.

namespace {

// Barriers are dropped when a Program is lowered to a Circuit (see the
// Barrier case in Program::AddToCircuit), so the qubits a barrier covers are
// only observable on the parsed Program - hence driving the grammar directly
// here, the same grammar QasmToCirc drives.
std::vector<int> ParseBarrierQubits(const std::string &qasmStr) {
  std::string input = qasmStr;
  qasm::QasmGrammar<> grammar;
  qasm::Program program;

  auto it = input.begin();
  const bool parsed = boost::spirit::qi::phrase_parse(
      it, input.end(), grammar, qasm::ascii::space, program);

  BOOST_TEST(parsed, "Failed to parse:\n" << qasmStr);
  BOOST_TEST((it == input.end()), "Unparsed input remaining: '"
                                      << std::string(it, input.end()) << "'");

  std::vector<int> qubits;
  for (const auto &stmt : program.statements)
    if (stmt.opType == qasm::QoperationStatement::OperationType::Barrier)
      qubits = stmt.qubits;

  return qubits;
}

}  // namespace

BOOST_AUTO_TEST_CASE(BareBarrierCoversAllQubits) {
  const std::vector<int> qubits = ParseBarrierQubits(
      "OPENQASM 3.0;\n"
      "qubit[2] a;\n"
      "qubit[3] b;\n"
      "barrier;\n");

  const std::vector<int> expected = {0, 1, 2, 3, 4};
  BOOST_TEST(qubits == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(BareBarrierParsesInAFullProgram) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "barrier;\n"
      "cx q[0], q[1];\n");

  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());
  BOOST_TEST(circuit != nullptr);
}

BOOST_AUTO_TEST_CASE(BarrierWithOperandsStillCoversOnlyThose) {
  // Regression guard for the added bare-barrier alternative: the operand
  // form must keep resolving exactly its own operands.
  const std::vector<int> qubits = ParseBarrierQubits(
      "OPENQASM 3.0;\n"
      "qubit[2] a;\n"
      "qubit[3] b;\n"
      "barrier b;\n");

  const std::vector<int> expected = {2, 3, 4};
  BOOST_TEST(qubits == expected, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(GateNameStartingWithBarrierIsNotMistakenForOne) {
  // The bare-barrier alternative matches the keyword with no following
  // space, so it needs the identifier-boundary lookahead or it would claim
  // the prefix of a gate named "barriers".
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "gate barriers a { x a; }\n"
      "barriers q[0];\n");

  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  auto qc = MakeInitializedSimulator(1);
  Circuits::OperationState state(1);
  circuit->Execute(qc, state);

  BOOST_TEST(
      checkClose(std::complex<double>(qc->Probability(1), 0.),
                 std::complex<double>(1., 0.), 0.0001),
      "Gate 'barriers' did not run, got P(|1>) = " << qc->Probability(1));
}

// ****************************************************************************
// B3: `gateModifier: ... (CTRL | NEGCTRL) (LPAREN expression RPAREN)? AT` -
// the control modifiers take an optional count. The count routes into the
// existing lowering, which supports one control generally and two only where
// a two-control gate exists (ccx), so anything beyond that must reach the
// existing multi-control error rather than the misleading "Unsupported gate
// with parameters: ctrl".

BOOST_AUTO_TEST_CASE(CtrlWithCountOneEqualsCtrlEqualsCx) {
  const std::string reference =
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "cx q[0], q[1];\n";

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ x q[0], q[1];\n",
      reference, 2);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl(1) @ x q[0], q[1];\n",
      reference, 2);
}

BOOST_AUTO_TEST_CASE(CtrlWithCountTwoEqualsCcx) {
  // Both controls prepared to |1> so the target actually flips - with either
  // control left at |0> the ccx is the identity and the comparison would
  // hold vacuously.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[0];\n"
      "x q[1];\n"
      "ctrl(2) @ x q[0], q[1], q[2];\n",
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[0];\n"
      "x q[1];\n"
      "ccx q[0], q[1], q[2];\n",
      3);

  // ... and the target is genuinely flipped, so the equivalence above is not
  // an identity-versus-identity comparison.
  CheckDifferentProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[0];\n"
      "x q[1];\n"
      "ctrl(2) @ x q[0], q[1], q[2];\n",
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[0];\n"
      "x q[1];\n",
      3);
}

BOOST_AUTO_TEST_CASE(CtrlCountTwoMatchesTwoNestedCtrls) {
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[0];\n"
      "x q[1];\n"
      "ctrl(2) @ x q[0], q[1], q[2];\n",
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "x q[0];\n"
      "x q[1];\n"
      "ctrl @ ctrl @ x q[0], q[1], q[2];\n",
      3);
}

BOOST_AUTO_TEST_CASE(CtrlWithCountThreeReachesTheMultiControlError) {
  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "qubit[4] q;\n"
      "ctrl(3) @ x q[0], q[1], q[2], q[3];\n");

  BOOST_TEST(parser.Failed());
  const std::string &message = parser.GetErrorMessage();

  BOOST_TEST(message.find("more than two controls") != std::string::npos,
             "Expected the existing multi-control error, got: " << message);
  BOOST_TEST(
      message.find("Unsupported gate with parameters") == std::string::npos,
      "ctrl(n) must not be misreported as an unknown parameterised "
      "gate: "
          << message);
}

BOOST_AUTO_TEST_CASE(NegCtrlWithCountOneEqualsNegCtrl) {
  // negctrl controls on |0>, so with the control left at |0> the target
  // flips; the two spellings must agree.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "negctrl(1) @ x q[0], q[1];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "negctrl @ x q[0], q[1];\n",
      2);
}

BOOST_AUTO_TEST_CASE(NegCtrlWithCountTwoNegatesBothItsControls) {
  // negctrl(2) owns two consecutive qubit arguments and negates both, so it
  // must match two nested negctrls rather than only negating the first.
  for (const std::string preparation :
       {std::string(""), std::string("x q[0];\n"), std::string("x q[1];\n"),
        std::string("x q[0];\nx q[1];\n")}) {
    CheckSameProbabilities(
        "OPENQASM 3.0;\n"
        "qubit[3] q;\n" +
            preparation + "negctrl(2) @ x q[0], q[1], q[2];\n",
        "OPENQASM 3.0;\n"
        "qubit[3] q;\n" +
            preparation + "negctrl @ negctrl @ x q[0], q[1], q[2];\n",
        3);
  }
}

BOOST_AUTO_TEST_CASE(CtrlWithCountRejectsANonPositiveOrFractionalCount) {
  for (const std::string count :
       {std::string("0"), std::string("-1"), std::string("1.5")}) {
    CheckRejectedWithMessage(
        "OPENQASM 3.0;\n"
        "qubit[2] q;\n"
        "ctrl(" +
            count + ") @ x q[0], q[1];\n",
        "must be a positive whole number");
  }
}

BOOST_AUTO_TEST_CASE(CtrlWithCountIsStillRejectedUnderQasm2) {
  // Modifiers are QASM3-only; adding the count form must not open a door in
  // QASM2.
  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "ctrl(1) @ x q[0], q[1];\n");

  BOOST_TEST(parser.Failed(), "ctrl(1) @ should not parse under OPENQASM 2.0");
}

// ****************************************************************************
// Gate-name mapping: every accepted name must resolve to the gate it names.
//
// Two bugs shared one shape - a name listed in an allowed-gate set with no
// correct arm in AddGateExpr::GetGateType. "id" fell through to kNone, whose
// arity is 0, so its qubit argument was rejected and a Qiskit circuit with an
// identity gate could not be imported at all; "sxdag" was matched by the
// "sdg" arm before reaching its own, so it silently applied S-dagger instead
// of SX-dagger, and its intended partner "sdag" mapped to nothing. The
// table-driven test below walks the allowed-gate sets themselves, so a name
// added to one of them without a mapping fails loudly rather than waiting to
// be noticed.

namespace {

// One accepted gate name, the shape of a minimal call to it, and the gate the
// call must produce. kNone means the name is a legitimate no-op that must
// parse and emit nothing - true of exactly "id" and "gphase", which are
// spelled out here rather than being excluded from the table, so that a name
// silently resolving to nothing cannot hide among them.
struct AllowedGateExpectation {
  const char *name;
  int nrParams;
  int nrQubits;
  Circuits::QuantumGateType gateType;
  // True for the stdgates.inc-only names, which resolve as recorded here
  // under a 3.0 header and are rejected under a 2.0 one. The allowed-gate
  // sets themselves are dialect-agnostic - the version gate lives in the
  // grammar (`qasm2RejectedGate`) - so the dialect has to be recorded here
  // rather than derived from those sets.
  bool qasm3Only = false;
};

const std::vector<AllowedGateExpectation> &AllowedGateExpectations() {
  using GateType = Circuits::QuantumGateType;
  static const std::vector<AllowedGateExpectation> expectations = {
      // Parameterless, one qubit.
      {"x", 0, 1, GateType::kXGateType},
      {"y", 0, 1, GateType::kYGateType},
      {"z", 0, 1, GateType::kZGateType},
      {"h", 0, 1, GateType::kHadamardGateType},
      {"s", 0, 1, GateType::kSGateType},
      {"sdg", 0, 1, GateType::kSdgGateType},
      {"sdag", 0, 1, GateType::kSdgGateType},
      {"t", 0, 1, GateType::kTGateType},
      {"tdg", 0, 1, GateType::kTdgGateType},
      {"tdag", 0, 1, GateType::kTdgGateType},
      {"sx", 0, 1, GateType::kSxGateType},
      {"sxdg", 0, 1, GateType::kSxDagGateType},
      {"sxdag", 0, 1, GateType::kSxDagGateType},
      {"k", 0, 1, GateType::kKGateType},
      // Parameterless, two and three qubits.
      {"swap", 0, 2, GateType::kSwapGateType},
      {"cx", 0, 2, GateType::kCXGateType},
      // The uppercase QASM2 builtin spellings, the only two names matched in
      // any case other than lowercase.
      {"CX", 0, 2, GateType::kCXGateType},
      {"cy", 0, 2, GateType::kCYGateType},
      {"cz", 0, 2, GateType::kCZGateType},
      {"ch", 0, 2, GateType::kCHGateType},
      {"csx", 0, 2, GateType::kCSxGateType},
      {"csxdg", 0, 2, GateType::kCSxDagGateType},
      {"csxdag", 0, 2, GateType::kCSxDagGateType},
      {"cswap", 0, 3, GateType::kCSwapGateType},
      {"ccx", 0, 3, GateType::kCCXGateType},
      // The identity: a standard gate in qelib1.inc and stdgates.inc that
      // Qiskit emits, and that must parse while contributing no operation.
      {"id", 0, 1, GateType::kNone},
      // One parameter.
      {"p", 1, 1, GateType::kPhaseGateType},
      {"phase", 1, 1, GateType::kPhaseGateType, true},
      {"rx", 1, 1, GateType::kRxGateType},
      {"ry", 1, 1, GateType::kRyGateType},
      {"rz", 1, 1, GateType::kRzGateType},
      {"u1", 1, 1, GateType::kUGateType},
      {"cp", 1, 2, GateType::kCPGateType},
      {"cphase", 1, 2, GateType::kCPGateType, true},
      {"crx", 1, 2, GateType::kCRxGateType},
      {"cry", 1, 2, GateType::kCRyGateType},
      {"crz", 1, 2, GateType::kCRzGateType},
      {"cu1", 1, 2, GateType::kCUGateType},
      // A global phase, which is unobservable and so emits nothing. It is the
      // reason kNone cannot simply be made to accept any arity: its arity is
      // genuinely zero.
      {"gphase", 1, 0, GateType::kNone, true},
      // Several parameters.
      {"u", 3, 1, GateType::kUGateType},
      {"U", 3, 1, GateType::kUGateType},
      {"u3", 3, 1, GateType::kUGateType},
      {"cu", 3, 2, GateType::kCUGateType},
      {"cu3", 3, 2, GateType::kCUGateType}};

  return expectations;
}

// Builds a one-line program calling `gate` with the arity the table records.
std::string SingleCallProgram(const std::string &versionLine,
                              const std::string &registerDecl,
                              const AllowedGateExpectation &gate) {
  std::string call = gate.name;

  if (gate.nrParams > 0) {
    call += "(";
    for (int p = 0; p < gate.nrParams; ++p) {
      if (p > 0) call += ", ";
      call += "0." + std::to_string(p + 1);
    }
    call += ")";
  }

  for (int q = 0; q < gate.nrQubits; ++q)
    call += (q == 0 ? " q[" : ", q[") + std::to_string(q) + "]";

  return versionLine + registerDecl + call + ";\n";
}

// The two names that are legitimately no-ops: the identity, and a global
// phase, which is unobservable. Nothing else may expect kNone.
bool IsLegitimateNoOpGateName(const std::string &name) {
  return name == "id" || name == "gphase";
}

// The rule closing the hole this guard used to have. A table entry whose
// expected type was kNone used to be short-circuited into "must add zero
// operations", so a name could be added to an allowed-gate set together with
// a kNone row and pass both guard tests while being exactly the silent no-op
// the guard exists to catch. kNone is now not an escape hatch but a claim
// that has to match the name: only "id" and "gphase" may make it, and they
// must. Factored out of the guard so it can be checked on a hypothetical
// entry without mutating the real table - see
// GuardRejectsANoOpExpectationForAnyOtherName.
bool ExpectationIsSelfConsistent(const AllowedGateExpectation &gate) {
  return IsLegitimateNoOpGateName(gate.name) ==
         (gate.gateType == Circuits::QuantumGateType::kNone);
}

// The union of the three allowed-gate sets, which is what the parser accepts.
std::unordered_set<std::string> AllAllowedGateNames() {
  std::unordered_set<std::string> names;

  for (const auto *set : {&qasm::AddGateExpr::AllowedNoParamGates(),
                          &qasm::AddGateExpr::AllowedOneParamGates(),
                          &qasm::AddGateExpr::AllowedMultipleParamsGates()})
    names.insert(set->begin(), set->end());

  return names;
}

}  // namespace

BOOST_AUTO_TEST_CASE(ExpectationTableCoversExactlyTheAllowedGateNames) {
  // The drift guard. Without it, adding a name to an allowed-gate set and
  // forgetting its GetGateType arm - which is how both known gate-name bugs
  // arose - would leave the table below silently not testing it.
  const std::unordered_set<std::string> allowedNames = AllAllowedGateNames();

  std::unordered_set<std::string> tableNames;
  for (const auto &gate : AllowedGateExpectations())
    tableNames.insert(gate.name);

  for (const auto &name : allowedNames)
    BOOST_TEST(tableNames.count(name) == 1u,
               "The parser accepts the gate name '"
                   << name
                   << "' but the expectation table in this file has no entry "
                      "for it; add one naming the gate it must resolve to");

  for (const auto &name : tableNames)
    BOOST_TEST(allowedNames.count(name) == 1u,
               "The expectation table lists the gate name '"
                   << name << "' which the parser no longer accepts");
}

BOOST_AUTO_TEST_CASE(EveryAllowedGateNameResolvesToTheGateItNames) {
  for (const auto &versions : {std::make_pair(std::string("OPENQASM 2.0;\n"),
                                              std::string("qreg q[3];\n")),
                               std::make_pair(std::string("OPENQASM 3.0;\n"),
                                              std::string("qubit[3] q;\n"))}) {
    const bool isQasm3 = versions.first == "OPENQASM 3.0;\n";

    for (const auto &gate : AllowedGateExpectations()) {
      const std::string qasmStr =
          SingleCallProgram(versions.first, versions.second, gate);

      // The stdgates.inc-only names are not gates under a 2.0 header at all,
      // so under that dialect the assertion is the rejection rather than the
      // resolution. Pinned here, on the same table walk, so a name that
      // stopped being gated would be caught by the guard that walks every
      // accepted name rather than only by the dedicated tests.
      if (gate.qasm3Only && !isQasm3) {
        CheckRejectedWithMessage(qasmStr, gate.name);
        continue;
      }

      qasm::QasmToCirc<> parser;
      auto circuit = parser.ParseAndTranslate(qasmStr);

      BOOST_TEST(!parser.Failed(),
                 "Gate '" << gate.name << "' should parse, got: "
                          << parser.GetErrorMessage() << "\nprogram:\n"
                          << qasmStr);
      if (parser.Failed()) continue;

      // kNone is a claim only "id" and "gphase" may make, not a way for any
      // name to opt out of the gate-type assertion below.
      BOOST_TEST(
          ExpectationIsSelfConsistent(gate),
          "Gate '" << gate.name
                   << "': only 'id' and 'gphase' may expect kNone (they are "
                      "the only legitimate no-ops), and both must; every "
                      "other name must name the gate it resolves to");

      if (IsLegitimateNoOpGateName(gate.name)) {
        BOOST_TEST(circuit->GetNumberOfOperations() == 0u,
                   "Gate '" << gate.name
                            << "' is a no-op and must add no operation, got "
                            << circuit->GetNumberOfOperations());
        continue;
      }

      BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() == 1u);
      BOOST_TEST((NthGateType(circuit, 0) == gate.gateType),
                 "Gate '" << gate.name
                          << "' resolved to the wrong gate type, program:\n"
                          << qasmStr);
    }
  }
}

BOOST_AUTO_TEST_CASE(GuardRejectsANoOpExpectationForAnyOtherName) {
  // The guard above used to treat an expected type of kNone as "this name is
  // a no-op, only check it adds nothing", so a new name could be added to an
  // allowed-gate set together with a kNone row and pass both guard tests
  // while silently adding no operation - the exact failure the guard exists
  // to catch. ExpectationIsSelfConsistent is that hole closed, and this
  // checks the rule itself on hypothetical entries, since the real table
  // cannot contain a failing one without breaking the guard.
  using GateType = Circuits::QuantumGateType;

  // The two legitimate no-ops, which must expect kNone.
  BOOST_TEST(ExpectationIsSelfConsistent({"id", 0, 1, GateType::kNone}));
  BOOST_TEST(ExpectationIsSelfConsistent({"gphase", 1, 0, GateType::kNone}));

  // Any other name expecting kNone is the silent-no-op failure, whether it is
  // a real gate name or one that does not exist yet.
  BOOST_TEST(!ExpectationIsSelfConsistent({"x", 0, 1, GateType::kNone}));
  BOOST_TEST(!ExpectationIsSelfConsistent({"ccx", 0, 3, GateType::kNone}));
  BOOST_TEST(!ExpectationIsSelfConsistent({"newgate", 0, 1, GateType::kNone}));

  // And the converse: "id"/"gphase" may not claim a gate type either, which
  // would let a genuine no-op masquerade as an emitted gate.
  BOOST_TEST(!ExpectationIsSelfConsistent({"id", 0, 1, GateType::kXGateType}));
  BOOST_TEST(
      !ExpectationIsSelfConsistent({"gphase", 1, 0, GateType::kPhaseGateType}));
}

BOOST_AUTO_TEST_CASE(DaggerSpellingsResolveToTheirOwnGate) {
  // "sdag"/"sxdag" are the QASM2-era spellings of "sdg"/"sxdg"; each pair must
  // land on the same gate. The bug had "sxdag" caught by the "sdg" arm, so it
  // applied S-dagger, and left "sdag" unmapped.
  for (const auto &versions : {std::make_pair(std::string("OPENQASM 2.0;\n"),
                                              std::string("qreg q[4];\n")),
                               std::make_pair(std::string("OPENQASM 3.0;\n"),
                                              std::string("qubit[4] q;\n"))}) {
    const std::string qasmStr = versions.first + versions.second +
                                "sdg q[0];\n"
                                "sdag q[1];\n"
                                "sxdg q[2];\n"
                                "sxdag q[3];\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    const std::vector<Circuits::QuantumGateType> expected = {
        Circuits::QuantumGateType::kSdgGateType,
        Circuits::QuantumGateType::kSdgGateType,
        Circuits::QuantumGateType::kSxDagGateType,
        Circuits::QuantumGateType::kSxDagGateType};

    BOOST_TEST(circuit->GetNumberOfOperations() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
      BOOST_TEST((NthGateType(circuit, i) == expected[i]),
                 "Wrong gate type at position " << i << " in:\n"
                                                << qasmStr);
  }
}

BOOST_AUTO_TEST_CASE(SdgAndSxdgAreDistinguishableByTheProbeUsedBelow) {
  // Sdg is diagonal, so it is invisible from |0>; the h-sandwich is what makes
  // the equivalence test that follows able to fail. This shows the probe
  // separates Sdg from SXdg, which is exactly the substitution the bug made.
  CheckDifferentProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "h q[0];\n"
      "sdg q[0];\n"
      "h q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "h q[0];\n"
      "sxdg q[0];\n"
      "h q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(SdagIsNumericallyEquivalentToSdg) {
  CheckSameProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "h q[0];\n"
      "sdag q[0];\n"
      "h q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "h q[0];\n"
      "sdg q[0];\n"
      "h q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(SxdagIsNumericallyEquivalentToSxdg) {
  // SXdg is not diagonal, so |0> alone already separates it from Sdg, which is
  // what the buggy mapping produced here.
  CheckSameProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "sxdag q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "sxdg q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(DaggerSpellingsAcceptTheSameModifiersAsTheirTwin) {
  // NormalizeGateName's invariant is that a modified call accepts exactly what
  // the same unmodified call accepts; the "dag" spellings were excluded from
  // it only because GetGateType mapped them inconsistently.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "inv @ sdag q[0];\n"
      "h q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "h q[0];\n"
      "inv @ sdg q[0];\n"
      "h q[0];\n",
      1);

  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ sxdag q[0], q[1];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "h q[0];\n"
      "ctrl @ sxdg q[0], q[1];\n",
      2);
}

BOOST_AUTO_TEST_CASE(IdentityGateParsesAndAddsNoOperation) {
  // Qiskit emits "id" in both dialects, indexed and (via broadcasting) over a
  // whole register. Each form must parse and contribute nothing.
  for (const auto &versions :
       {std::make_pair(std::string("OPENQASM 2.0;\n"),
                       std::string("qreg q[3];\n")),
        std::make_pair(std::string("OPENQASM 3.0;\n"),
                       std::string("qubit[3] q;\n"))}) {
    for (const std::string idCall :
         {std::string("id q[0];\n"), std::string("id q;\n")}) {
      const std::string qasmStr = versions.first + versions.second + idCall;

      qasm::QasmToCirc<> parser;
      auto circuit = parser.ParseAndTranslate(qasmStr);
      BOOST_TEST(!parser.Failed(), parser.GetErrorMessage() << "\nprogram:\n"
                                                            << qasmStr);
      if (parser.Failed()) continue;

      BOOST_TEST(circuit->GetNumberOfOperations() == 0u,
                 "'id' must add no operation, got "
                     << circuit->GetNumberOfOperations() << " for:\n"
                     << qasmStr);
    }
  }
}

BOOST_AUTO_TEST_CASE(IdentityGateDoesNotDisturbSurroundingOperations) {
  const std::string qasmStr =
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "x q[0];\n"
      "id q[0];\n"
      "id q;\n"
      "cx q[0], q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() == 2u);
  BOOST_TEST(
      (NthGateType(circuit, 0) == Circuits::QuantumGateType::kXGateType));
  BOOST_TEST(
      (NthGateType(circuit, 1) == Circuits::QuantumGateType::kCXGateType));
}

BOOST_AUTO_TEST_CASE(GphaseStillParsesAsANoOpAfterTheIdentityFix) {
  // "id" is exempted from the exact-arity check individually; kNone itself
  // still has arity zero, which is what gphase relies on. Only under a 3.0
  // header - "gphase" is a stdgates.inc name and no dialect of qelib1.inc has
  // it, so the loop over both dialects this used to run has become a QASM3
  // check plus the rejection tests further down.
  {
    const std::string qasmStr =
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n"
        "gphase(0.5);\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());

    if (!parser.Failed())
      BOOST_TEST(circuit->GetNumberOfOperations() == 0u,
                 "gphase should add no operation, got "
                     << circuit->GetNumberOfOperations());
  }

  // And it still takes no qubit argument, which is the arity kNone encodes.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "gphase(0.5) q[0];\n",
      "requires exactly 0 qubits");
}

BOOST_AUTO_TEST_CASE(UnknownGateNameIsStillRejected) {
  // The regression guard on the "id" fix: exempting "id" alone must not turn
  // an unrecognised name into a silent no-op. A typo has to be loud.
  for (const auto &versions :
       {std::make_pair(std::string("OPENQASM 2.0;\n"),
                       std::string("qreg q[2];\n")),
        std::make_pair(std::string("OPENQASM 3.0;\n"),
                       std::string("qubit[2] q;\n"))}) {
    for (const std::string call :
         {std::string("notagate q[0];\n"), std::string("notagate q;\n"),
          std::string("idd q[0];\n"), std::string("i q[0];\n"),
          std::string("notagate q[0], q[1];\n")})
      CheckRejectedWithMessage(versions.first + versions.second + call,
                               "Unsupported gate without parameters");
  }
}

BOOST_AUTO_TEST_CASE(Cu1RejectsExtraParametersJustAsU1Does) {
  // "cu1" takes exactly one angle. It used to be listed in both the one- and
  // the multiple-parameter sets, so cu1(a, b, c) was accepted and silently
  // lowered as cu3 - asymmetric with u1, which rejected the extra angles.
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "cu1(0.1, 0.2, 0.3) q[0], q[1];\n",
      "does not allow multiple parameters");
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "u1(0.1, 0.2, 0.3) q[0];\n",
      "does not allow multiple parameters");

  // The single-parameter form is untouched: cu1(lambda) is cu with theta and
  // phi at zero.
  CheckSameProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "x q[1];\n"
      "cu1(0.7) q[0], q[1];\n"
      "h q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[2];\n"
      "h q[0];\n"
      "x q[1];\n"
      "cu(0.0, 0.0, 0.7) q[0], q[1];\n"
      "h q[0];\n",
      2);
}

// ****************************************************************************
// Expression arithmetic: the associativity of '-' and '/', and where a
// leading sign sits relative to '**'.
//
// Both were silently wrong. `additive` and `product` recursed into themselves
// on the right, making every operator right-associative: `rx(1 - 2 - 3)`
// applied 2 instead of -4 and `rx(8 / 4 / 2)` applied 4 instead of 1. And a
// leading sign bound tighter than '**' (partly through `unary` sitting inside
// `factor`, partly through qi::double_ swallowing the sign itself), so
// `2 ** -3 ** 2` was 2 ** ((-3) ** 2) = 512 where the spec's precedence -
// parenthesis > index > '**' > unary > '* /' > '+ -' - makes it 2 ** -(3 **
// 2), and `-2 ** 2` was 4 rather than -4.

namespace {

// Builds a program whose measured probabilities are *odd* in the angle
// written into `angleExpr`: h puts the qubit on +x, rz(theta) turns it to
// (cos theta, sin theta, 0), and rx(pi/2) tips that onto the z axis, leaving
// P(|0>) = (1 + sin theta) / 2.
//
// This matters more than it looks. The obvious probe, rx(theta) on |0>, has
// P(|0>) = cos^2(theta/2), which is *even* in theta - it cannot tell -4 from
// 4, and a sign flip is exactly what an associativity or precedence bug
// produces. Every expression assertion below therefore goes through this
// probe, and SignProbeDistinguishesAnAngleFromItsNegation checks the probe
// itself actually has that property before anything relies on it.
std::string SignedAngleProbe(const std::string &versionLine,
                             const std::string &registerDecl,
                             const std::string &angleExpr) {
  return versionLine + registerDecl + "h q[0];\nrz(" + angleExpr +
         ") q[0];\nrx(pi / 2) q[0];\n";
}

// The two dialects, each with the register declaration it spells. Expression
// parsing is shared by both, so every arithmetic assertion is made twice.
const std::vector<std::pair<std::string, std::string>> &Dialects() {
  static const std::vector<std::pair<std::string, std::string>> dialects = {
      {"OPENQASM 2.0;\n", "qreg q[1];\n"},
      {"OPENQASM 3.0;\n", "qubit[1] q;\n"}};

  return dialects;
}

// Asserts, in both dialects, that `angleExpr` evaluates to `expected` and -
// so the equivalence cannot be vacuous - that it does not evaluate to
// `wrongValue`, which is what the bug being pinned used to produce.
void CheckAngleExpressionInDialect(
    const std::pair<std::string, std::string> &dialect,
    const std::string &angleExpr, const std::string &expected,
    const std::string &wrongValue) {
  CheckSameProbabilities(
      SignedAngleProbe(dialect.first, dialect.second, angleExpr),
      SignedAngleProbe(dialect.first, dialect.second, expected), 1);
  CheckDifferentProbabilities(
      SignedAngleProbe(dialect.first, dialect.second, angleExpr),
      SignedAngleProbe(dialect.first, dialect.second, wrongValue), 1);
}

void CheckAngleExpression(const std::string &angleExpr,
                          const std::string &expected,
                          const std::string &wrongValue) {
  for (const auto &dialect : Dialects())
    CheckAngleExpressionInDialect(dialect, angleExpr, expected, wrongValue);
}

// For expressions spelled with '**', which is a QASM3 powerExpression and has
// no counterpart in QASM2's `exp` production - the precedence and
// associativity being pinned are properties of the operator, so they can only
// be asserted in the dialect that has it. `expected` and `wrongValue` are
// evaluated in the same dialect, so they may use '**' too.
void CheckQasm3AngleExpression(const std::string &angleExpr,
                               const std::string &expected,
                               const std::string &wrongValue) {
  CheckAngleExpressionInDialect({"OPENQASM 3.0;\n", "qubit[1] q;\n"}, angleExpr,
                                expected, wrongValue);
}

}  // namespace

BOOST_AUTO_TEST_CASE(SignProbeDistinguishesAnAngleFromItsNegation) {
  // The premise every assertion below rests on: this probe is sensitive to
  // the sign of its angle. If it ever stops being (a convention change in rz
  // or rx would do it), the tests below would keep passing while no longer
  // testing what they claim, so the property is asserted directly.
  for (const auto &dialect : Dialects()) {
    CheckDifferentProbabilities(
        SignedAngleProbe(dialect.first, dialect.second, "1"),
        SignedAngleProbe(dialect.first, dialect.second, "0 - 1"), 1);
    CheckDifferentProbabilities(
        SignedAngleProbe(dialect.first, dialect.second, "4"),
        SignedAngleProbe(dialect.first, dialect.second, "0 - 4"), 1);
  }
}

BOOST_AUTO_TEST_CASE(SubtractionIsLeftAssociative) {
  // The headline case: `1 - 2 - 3` is (1 - 2) - 3 = -4. Right association
  // gives 1 - (2 - 3) = 2, which is what the parser used to apply.
  CheckAngleExpression("1 - 2 - 3", "-4", "2");
  CheckAngleExpression("10 - 1 - 2 - 3", "4", "8");
  CheckAngleExpression("2 - 3 + 4", "3", "-5");
}

BOOST_AUTO_TEST_CASE(DivisionIsLeftAssociative) {
  // `8 / 4 / 2` is (8 / 4) / 2 = 1; right association gives 8 / (4 / 2) = 4.
  CheckAngleExpression("8 / 4 / 2", "1", "4");
  CheckAngleExpression("12 / 2 / 3", "2", "18");
}

BOOST_AUTO_TEST_CASE(AssociativeOperatorsAreUnchangedByTheFold) {
  // '+' and '*' are associative, so the left fold cannot change their value -
  // but it does change the tree shape, so they are checked rather than
  // assumed.
  CheckAngleExpression("1 + 2 + 3", "6", "5");
  CheckAngleExpression("1 * 2 * 3", "6", "5");
}

BOOST_AUTO_TEST_CASE(MultiplicationStillBindsTighterThanAddition) {
  // The fold is per precedence level, so the levels themselves must still
  // nest the way they did.
  CheckAngleExpression("1 - 2 * 3", "-5", "-3");
  CheckAngleExpression("1 + 8 / 2", "5", "4.5");
  CheckAngleExpression("(1 - 2) * 3", "-3", "-5");
}

BOOST_AUTO_TEST_CASE(UnaryMinusBindsLooserThanPower) {
  // The spec's precedence puts unary below '**', so the sign applies to the
  // result of the power, not to its base: `-2 ** 2` is -(2 ** 2) = -4. It
  // used to be (-2) ** 2 = 4, both because `unary` sat inside `factor` (the
  // operand of '**') and because qi::double_ consumed the sign itself.
  //
  // '**' exists only in QASM3, so the two assertions that spell it are made
  // there; the sign's behaviour against an operator both dialects have is
  // still checked in both.
  CheckQasm3AngleExpression("-2 ** 2", "-4", "4");
  CheckQasm3AngleExpression("-(2 ** 2)", "-4", "4");
  CheckAngleExpression("-2 * 3", "-6", "6");

  // QASM2 spells the same precedence with '^', and the sign sits below it
  // there too - so the property this pins is not lost under QASM2, only
  // spelled differently. See QASM2CaretHasTheSameUnaryPrecedenceAsDoubleAsterisk.
}

BOOST_AUTO_TEST_CASE(PowerIsRightAssociativeThroughAUnaryOperand) {
  // 2 ** -3 ** 2 is 2 ** -(3 ** 2) = 2^-9 = 0.001953125. Reading the sign as
  // part of the base gives 2 ** ((-3) ** 2) = 2 ** 9 = 512, which is what the
  // parser used to produce. QASM3 only - '**' is its operator.
  CheckQasm3AngleExpression("2 ** -3 ** 2", "0.001953125", "512");
  CheckQasm3AngleExpression("2 ** -3", "0.125", "8");
}

BOOST_AUTO_TEST_CASE(PowerRemainsRightAssociative) {
  // Unchanged by the precedence fix, and easy to break with it: `2 ** 3 ** 2`
  // is 2 ** 9 = 512, not (2 ** 3) ** 2 = 64. QASM3 only - '**' is its
  // operator.
  CheckQasm3AngleExpression("2 ** 3 ** 2", "512", "64");
}

BOOST_AUTO_TEST_CASE(ALeadingSignOnItsOwnStillWorks) {
  // The sign is now the `unary` rule's alone - the number parser no longer
  // accepts one - so the plain negative literal is worth pinning: `rx(-2)`
  // must still be -2 and must not have lost its sign.
  CheckAngleExpression("-2", "0 - 2", "2");
  CheckAngleExpression("+2", "2", "-2");
  CheckAngleExpression("- -2", "2", "-2");
  CheckAngleExpression("1 - -2", "3", "-1");
  CheckAngleExpression("-0.5", "0 - 0.5", "0.5");
  CheckAngleExpression("-pi / 2", "0 - 1.5707963267948966",
                       "1.5707963267948966");
}

BOOST_AUTO_TEST_CASE(QASM2CaretHasTheSameUnaryPrecedenceAsDoubleAsterisk) {
  // QASM2 spells exponentiation '^' and it sits at the same precedence, so
  // `-2 ^ 2` is -(2 ^ 2) = -4 there too.
  CheckSameProbabilities(
      SignedAngleProbe("OPENQASM 2.0;\n", "qreg q[1];\n", "-2 ^ 2"),
      SignedAngleProbe("OPENQASM 2.0;\n", "qreg q[1];\n", "-4"), 1);
  CheckDifferentProbabilities(
      SignedAngleProbe("OPENQASM 2.0;\n", "qreg q[1];\n", "-2 ^ 2"),
      SignedAngleProbe("OPENQASM 2.0;\n", "qreg q[1];\n", "4"), 1);
}

// ****************************************************************************
// Gate-name lookup is case-sensitive, as OpenQASM itself is.
//
// `definedGates` was keyed and looked up with the name as written while every
// builtin lookup lowercased first, so a user-defined gate could never shadow
// a builtin that differed only in case: `gate X a { h a; }` followed by
// `x q[0];` silently applied the builtin X. The same mismatch accepted `SDG`,
// `RX(0.5)` and `CCX`, none of which is a gate in either dialect.

BOOST_AUTO_TEST_CASE(UserGateAndBuiltinDifferingOnlyInCaseCoexist) {
  // `x` is the builtin X; `X` is the user's Hadamard. Two different gate
  // types from two calls that differ only in case, which is the whole point.
  //
  // QASM3 only: declaring a gate named `X` needs an uppercase-initial
  // identifier, which QASM2's `id := [a-z][A-Za-z0-9_]*` does not have. The
  // QASM2 half of this loop is now
  // QASM2IdentifierRejectsUppercaseAndUnderscoreInitials.
  for (const auto &dialect : {std::make_pair(std::string("OPENQASM 3.0;\n"),
                                             std::string("qubit[1] q;\n"))}) {
    const std::string qasmStr = dialect.first + dialect.second +
                                "gate X a { h a; }\n"
                                "x q[0];\n"
                                "X q[0];\n";

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage() << "\nprogram:\n"
                                                          << qasmStr);
    if (parser.Failed()) continue;

    BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() == 2u);
    BOOST_TEST(
        (NthGateType(circuit, 0) == Circuits::QuantumGateType::kXGateType),
        "'x' must be the builtin X, not the user's gate");
    BOOST_TEST((NthGateType(circuit, 1) ==
                Circuits::QuantumGateType::kHadamardGateType),
               "'X' must be the user-declared gate, not the builtin X");
  }
}

BOOST_AUTO_TEST_CASE(UppercaseSpellingsOfBuiltinsAreNotGates) {
  // Previously accepted, and silently lowered onto the lowercase builtin.
  // OpenQASM is case-sensitive: none of these names exists.
  //
  // The named diagnostics are asserted under a 3.0 header, because they come
  // from the gate-name lookup and the name has to reach it: under QASM2 an
  // uppercase-initial name is not an identifier at all
  // (`id := [a-z][A-Za-z0-9_]*`), so the same calls are rejected one layer
  // earlier, by the grammar. Both dialects reject them; only the dialect
  // where they parse as names can say which name.
  const std::vector<std::string> noParamCalls = {"SDG q[0];\n", "H q[0];\n",
                                                 "CCX q[0], q[1], q[2];\n",
                                                 "Swap q[0], q[1];\n"};
  const std::vector<std::string> paramCalls = {
      "RX(0.5) q[0];\n", "Rz(0.5) q[0];\n", "CP(0.5) q[0], q[1];\n"};

  for (const std::string &call : noParamCalls)
    CheckRejectedWithMessage("OPENQASM 3.0;\nqubit[3] q;\n" + call,
                             "Unsupported gate without parameters");

  for (const std::string &call : paramCalls)
    CheckRejectedWithMessage("OPENQASM 3.0;\nqubit[3] q;\n" + call,
                             "Unsupported gate with parameters");

  // Under QASM2 the requirement is only that they are rejected, so this half
  // asserts exactly that and nothing about the wording.
  for (const auto &call : {noParamCalls, paramCalls})
    for (const std::string &oneCall : call) {
      const std::string qasmStr = "OPENQASM 2.0;\nqreg q[3];\n" + oneCall;

      qasm::QasmToCirc<> parser;
      parser.ParseAndTranslate(qasmStr);

      BOOST_TEST(parser.Failed(), "Expected rejection of:\n" << qasmStr);
    }
}

BOOST_AUTO_TEST_CASE(BothCasesOfTheQasm2UAndCXBuiltinsStillWork) {
  // The one deliberate exception: 'U' and 'CX' are the QASM2 builtins the
  // language itself spells in uppercase, and both cases of each are accepted.
  for (const auto &gateCall :
       {std::make_pair(std::string("U(0.1, 0.2, 0.3) q[0];\n"),
                       Circuits::QuantumGateType::kUGateType),
        std::make_pair(std::string("u(0.1, 0.2, 0.3) q[0];\n"),
                       Circuits::QuantumGateType::kUGateType),
        std::make_pair(std::string("CX q[0], q[1];\n"),
                       Circuits::QuantumGateType::kCXGateType),
        std::make_pair(std::string("cx q[0], q[1];\n"),
                       Circuits::QuantumGateType::kCXGateType)}) {
    const std::string qasmStr = "OPENQASM 2.0;\nqreg q[2];\n" + gateCall.first;

    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage() << "\nprogram:\n"
                                                          << qasmStr);
    if (parser.Failed()) continue;

    BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() == 1u);
    BOOST_TEST((NthGateType(circuit, 0) == gateCall.second), "wrong gate for:\n"
                                                                 << qasmStr);
  }

  // And an over-long argument list on either still names the arity, rather
  // than falling through to "unknown gate" - that fall-through is why both
  // spellings are listed in the allowed-gate sets as well as in their own
  // parse rules. Under a 3.0 header, where an uppercase-initial name is an
  // identifier and so `gatecall` can pick the over-long call up.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "CX q[0], q[1], q[2];\n",
      "requires exactly 2 qubits");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "U(0.1, 0.2, 0.3) q[0], q[1];\n",
      "requires exactly 1 qubits");

  // Under QASM2 those same over-long calls are plain syntax errors: `U` and
  // `CX` are keywords of that grammar, matched only by `ugateCall`/
  // `cxgateCall`, and QASM2's identifier rule cannot spell either of them, so
  // there is no `gatecall` fall-through to reach the arity check. Still
  // rejected - which is what matters - just without the arity wording.
  for (const std::string &program :
       {std::string("OPENQASM 2.0;\nqreg q[3];\nCX q[0], q[1], q[2];\n"),
        std::string(
            "OPENQASM 2.0;\nqreg q[2];\nU(0.1, 0.2, 0.3) q[0], q[1];\n")}) {
    qasm::QasmToCirc<> parser;
    parser.ParseAndTranslate(program);

    BOOST_TEST(parser.Failed(), "Expected rejection of:\n" << program);
  }
}

BOOST_AUTO_TEST_CASE(UppercaseUserGateStillResolvesToItsOwnDefinition) {
  // The other half of the case-sensitivity fix: an uppercase name that is not
  // declared is an error, but one that is declared resolves to its
  // definition. `SDG` here is the user's Hadamard, not the builtin sdg.
  // Under a 3.0 header, since the name is uppercase-initial and QASM2 has no
  // such identifier.
  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "gate SDG a { h a; }\n"
      "SDG q[0];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() == 1u);
  BOOST_TEST((NthGateType(circuit, 0) ==
              Circuits::QuantumGateType::kHadamardGateType));
}

BOOST_AUTO_TEST_CASE(ModifiersDoNotReintroduceCaseFolding) {
  // The modified-gate path lowercased the name of its own accord before
  // consulting the lowering tables, so it needs the same treatment: an
  // uppercase name is not a builtin there either.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "ctrl @ SDG q[0], q[1];\n",
      "SDG");

  // The lowercase spelling of the same call is unaffected.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "x q[0];\n"
      "ctrl @ sdg q[0], q[1];\n",
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "x q[0];\n"
      "cp(-1.5707963267948966) q[0], q[1];\n",
      2);
}

// ****************************************************************************
// 'id' has an arity again.
//
// Commit a384bf4 wrapped the exact-arity check in `if (gateNameLower != "id")`
// to let `id q[0];` through, which removed the only bound on it: `id q[0],
// q[1];`, `id q[0], q[1], q[2];` and even the duplicate-qubit `id q[0],
// q[0];` all parsed silently. It is a one-qubit gate and takes exactly one
// argument, like any other.

BOOST_AUTO_TEST_CASE(IdentityGateTakesExactlyOneQubitArgument) {
  for (const auto &versions : {std::make_pair(std::string("OPENQASM 2.0;\n"),
                                              std::string("qreg q[3];\n")),
                               std::make_pair(std::string("OPENQASM 3.0;\n"),
                                              std::string("qubit[3] q;\n"))}) {
    // Still accepted, and still contributing nothing: the indexed form and
    // the whole-register broadcast, which is one argument too.
    for (const std::string &idCall :
         {std::string("id q[0];\n"), std::string("id q;\n")}) {
      const std::string qasmStr = versions.first + versions.second + idCall;

      qasm::QasmToCirc<> parser;
      auto circuit = parser.ParseAndTranslate(qasmStr);
      BOOST_TEST(!parser.Failed(), parser.GetErrorMessage() << "\nprogram:\n"
                                                            << qasmStr);
      if (!parser.Failed())
        BOOST_TEST(circuit->GetNumberOfOperations() == 0u,
                   "'id' must add no operation, got "
                       << circuit->GetNumberOfOperations() << " for:\n"
                       << qasmStr);
    }

    // Rejected, including the duplicate-qubit spelling, which no other
    // one-qubit gate would accept either.
    for (const std::string &idCall :
         {std::string("id q[0], q[1];\n"),
          std::string("id q[0], q[1], q[2];\n"),
          std::string("id q[0], q[0];\n"), std::string("id q, q;\n")})
      CheckRejectedWithMessage(versions.first + versions.second + idCall,
                               "requires exactly 1 qubits");
  }

  // The two neighbours of the fix, unchanged: the qubit-less no-op still
  // parses and an unknown name is still an error, not a silent no-op.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "gphase(0.5);\n"
      "x q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "x q[0];\n",
      1);
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "notagate q[0];\n",
      "Unsupported gate without parameters: notagate");
}

// ****************************************************************************
// Cross-version leniencies: QASM3-only syntax and gate names must not parse
// under an `OPENQASM 2.0;` header.
//
// Six constructs were introduced without a version gate and so were accepted
// in both dialects: the widened identifier first-character class, bare
// `barrier;`, the '**' operator, and the three stdgates.inc gate names
// 'phase', 'cphase' and 'gphase'. Each is now guarded on the grammar's
// isQasm3 flag, the same mechanism the QASM3 modifiers, `input` declarations
// and braced conditionals already use.
//
// Every group below asserts all three of: accepted under `OPENQASM 3.0;`,
// rejected under `OPENQASM 2.0;`, and rejected with no version line at all -
// the third because the version is optional in this grammar and its absence
// defaults to QASM2, so a gate keyed off "saw a 3.0 line" and a gate keyed
// off "did not see a 2.0 line" would differ exactly there.

namespace {

// The three headers each construct is exercised against. The QASM2 spellings
// of the register declarations are used throughout, since `qreg`/`creg` are
// accepted under a 3.0 header too (the QASM3 `statement` production includes
// oldStyleDeclarationStatement) - that keeps the three programs differing in
// the version line and nothing else, which is the whole point.
const std::string kQasm3Header = "OPENQASM 3.0;\n";
const std::string kQasm2Header = "OPENQASM 2.0;\n";
const std::string kNoHeader = "";

// Parses `header + body` and asserts it succeeded.
void CheckAccepted(const std::string &header, const std::string &body) {
  const std::string qasmStr = header + body;

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);

  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage() << "\nprogram:\n"
                                                        << qasmStr);
}

// Asserts `body` is accepted under a 3.0 header and rejected both under a 2.0
// one and with no version line, with the rejection naming `expectedFragment`.
// For the purely syntactic gates that fragment is simply the offending text,
// which the "unparsed input remaining" message quotes back.
void CheckQasm3Only(const std::string &body,
                    const std::string &expectedFragment) {
  CheckAccepted(kQasm3Header, body);
  CheckRejectedWithMessage(kQasm2Header + body, expectedFragment);
  CheckRejectedWithMessage(kNoHeader + body, expectedFragment);
}

}  // namespace

BOOST_AUTO_TEST_CASE(UppercaseInitialIdentifierIsQasm3Only) {
  // QASM2's lexical rule is `id := [a-z][A-Za-z0-9_]*`; QASM3's is
  // `[A-Za-z_][A-Za-z0-9_]*`.
  CheckQasm3Only(
      "qreg q[1];\n"
      "gate MyGate a { x a; }\n"
      "MyGate q[0];\n",
      "MyGate");

  // Not only in a gate name - the class is the identifier's, so a register
  // declared with one is gated the same way.
  CheckQasm3Only(
      "qreg Q[1];\n"
      "x Q[0];\n",
      "Q[1]");
}

BOOST_AUTO_TEST_CASE(UnderscoreInitialIdentifierIsQasm3Only) {
  CheckQasm3Only(
      "qreg _q[1];\n"
      "x _q[0];\n",
      "_q[1]");
}

BOOST_AUTO_TEST_CASE(QASM2IdentifierStillAcceptsEverythingItsOwnRuleAllows) {
  // The falsifiability half of the two gates above: the narrowed class is
  // narrowed at the *first* character only. QASM2's trailing class is the
  // same `[A-Za-z0-9_]*` QASM3's is, so an uppercase letter, a digit and an
  // underscore are all still fine from the second character on - if this
  // failed, the gate would be rejecting far more than the dialect says and
  // the two tests above would pass for the wrong reason.
  for (const std::string &header : {kQasm2Header, kNoHeader, kQasm3Header})
    CheckAccepted(header,
                  "qreg qA_1[1];\n"
                  "gate myGate2_X a { x a; }\n"
                  "myGate2_X qA_1[0];\n");
}

BOOST_AUTO_TEST_CASE(QASM2IdentifierGateIsDrivenByTheVersionLineAlone) {
  // The complement of the falsifiability check: one program text, three
  // headers, opposite outcomes. Nothing but the version line differs, so the
  // gate cannot be an accident of the rest of the program.
  const std::string body =
      "qreg q[1];\n"
      "gate Gate1 a { x a; }\n"
      "Gate1 q[0];\n";

  CheckAccepted(kQasm3Header, body);

  for (const std::string &header : {kQasm2Header, kNoHeader}) {
    qasm::QasmToCirc<> parser;
    parser.ParseAndTranslate(header + body);
    BOOST_TEST(parser.Failed(), "Expected rejection under:\n"
                                    << header << body);
  }

  // And the lowercased spelling of the very same program is accepted in all
  // three, so what is being rejected is the initial character and not the
  // shape of the program.
  const std::string lowercased =
      "qreg q[1];\n"
      "gate gate1 a { x a; }\n"
      "gate1 q[0];\n";

  for (const std::string &header : {kQasm2Header, kNoHeader, kQasm3Header})
    CheckAccepted(header, lowercased);
}

BOOST_AUTO_TEST_CASE(BareBarrierIsQasm3Only) {
  // `barrierStatement: BARRIER gateOperandList? SEMICOLON` in QASM3, but
  // `statement: "barrier" anylist ";"` in QASM2 - the operands are required
  // there.
  CheckQasm3Only(
      "qreg q[2];\n"
      "h q[0];\n"
      "barrier;\n"
      "cx q[0], q[1];\n",
      "barrier;");
}

BOOST_AUTO_TEST_CASE(BarrierWithOperandsIsAcceptedInBothDialects) {
  // The falsifiability half: only the *bare* form is gated. The operand form
  // is QASM2's own spelling and must stay accepted everywhere, or the gate
  // above would be passing because barriers stopped parsing at all.
  for (const std::string &header : {kQasm2Header, kNoHeader, kQasm3Header})
    CheckAccepted(header,
                  "qreg q[2];\n"
                  "h q[0];\n"
                  "barrier q;\n"
                  "barrier q[0], q[1];\n"
                  "cx q[0], q[1];\n");
}

BOOST_AUTO_TEST_CASE(DoubleAsteriskIsQasm3Only) {
  // '**' is a QASM3 powerExpression; QASM2's `exp` production has no such
  // operator and spells exponentiation '^'.
  CheckQasm3Only(
      "qreg q[1];\n"
      "rx(2 ** 3) q[0];\n",
      "**");
}

BOOST_AUTO_TEST_CASE(CaretRemainsThePowerOperatorUnderQasm2) {
  // The falsifiability half of the '**' gate, and the guard on the one
  // dialect-dependent operator that was already correct: '^' is
  // exponentiation under QASM2 (and with no version line) and bitwise XOR
  // under QASM3. Gating '**' must not have disturbed either.
  CheckSameProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(8) q[0];\n",
      1);
  CheckSameProbabilities(
      "qreg q[1];\n"
      "rx(2 ^ 3) q[0];\n",
      "qreg q[1];\n"
      "rx(8) q[0];\n",
      1);
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "rx(1) q[0];\n",
      1);
}

BOOST_AUTO_TEST_CASE(StdgatesOnlyGateNamesAreQasm3Only) {
  // 'phase', 'cphase' and 'gphase' are stdgates.inc names. None of them
  // appears in any qelib1.inc, including the extended one Qiskit ships, so
  // none is a gate under a 2.0 header. Unlike the three syntactic gates
  // above, these are rejected by name, since the name is exactly what is
  // wrong with the program.
  CheckQasm3Only(
      "qreg q[1];\n"
      "phase(0.5) q[0];\n",
      "'phase' is an OpenQASM 3 stdgates.inc gate");
  CheckQasm3Only(
      "qreg q[2];\n"
      "cphase(0.5) q[0], q[1];\n",
      "'cphase' is an OpenQASM 3 stdgates.inc gate");
  CheckQasm3Only(
      "qreg q[1];\n"
      "gphase(0.5);\n",
      "'gphase' is an OpenQASM 3 stdgates.inc gate");

  // The diagnostic points at the qelib1.inc spelling where there is one,
  // which is the whole reason this rejection is by name rather than a syntax
  // error.
  CheckRejectedWithMessage(kQasm2Header + "qreg q[1];\nphase(0.5) q[0];\n",
                           "'p'");
  CheckRejectedWithMessage(
      kQasm2Header + "qreg q[2];\ncphase(0.5) q[0], q[1];\n", "'cp'");
}

BOOST_AUTO_TEST_CASE(StdgatesOnlyGateNamesAreGatedOnEveryCallPath) {
  // The gate sits at the head of `uop`, not at statement level, so that every
  // route to a gate call passes through it. These are the routes that do not
  // go straight through `statement`.
  CheckRejectedWithMessage(kQasm2Header +
                               "qreg q[1];\n"
                               "creg c[1];\n"
                               "if (c == 1) phase(0.5) q[0];\n",
                           "'phase' is an OpenQASM 3 stdgates.inc gate");
  CheckRejectedWithMessage(kQasm2Header +
                               "qreg q[1];\n"
                               "gate g a { phase(0.5) a; }\n"
                               "g q[0];\n",
                           "'phase' is an OpenQASM 3 stdgates.inc gate");
}

BOOST_AUTO_TEST_CASE(NamesMerelyStartingWithAStdgatesNameAreUnaffected) {
  // The identifier-boundary lookahead: "phased" is not "phase", so a QASM2
  // program may still declare and call it.
  CheckAccepted(kQasm2Header,
                "qreg q[1];\n"
                "gate phased a { x a; }\n"
                "phased q[0];\n");

  // And a program that declares a gate by one of the gated names itself is
  // calling its own gate, not the stdgates one - which is why the rule fails
  // rather than throwing when the name is user-defined.
  CheckAccepted(kQasm2Header,
                "qreg q[1];\n"
                "gate phase(x) a { rz(x) a; }\n"
                "phase(0.5) q[0];\n");
}

BOOST_AUTO_TEST_CASE(QelibGateNamesRemainAcceptedUnderQasm2) {
  // The deliberate non-scope list, pinned so that tightening it further is a
  // decision rather than a side effect. These eight are absent from the 2017
  // paper's qelib1.inc but present in the extended qelib1.inc Qiskit ships -
  // verified against Qiskit 2.5.2's QASM2 exporter, which emits all of them
  // under `OPENQASM 2.0;`. Rejecting them would break real-world QASM2.
  for (const std::string &header : {kQasm2Header, kNoHeader, kQasm3Header})
    CheckAccepted(header,
                  "qreg q[3];\n"
                  "sx q[0];\n"
                  "swap q[0], q[1];\n"
                  "cswap q[0], q[1], q[2];\n"
                  "p(0.5) q[0];\n"
                  "cp(0.5) q[0], q[1];\n"
                  "crx(0.5) q[0], q[1];\n"
                  "cry(0.5) q[0], q[1];\n"
                  "cu(0.1, 0.2, 0.3) q[0], q[1];\n");
}

BOOST_AUTO_TEST_CASE(MaestroExtensionGateNamesRemainAcceptedUnderQasm2) {
  // Likewise for maestro's own extensions, which predate the QASM3 work and
  // are deliberate simulator support accepted equally in both dialects.
  for (const std::string &header : {kQasm2Header, kNoHeader, kQasm3Header})
    CheckAccepted(header,
                  "qreg q[2];\n"
                  "sdag q[0];\n"
                  "tdag q[0];\n"
                  "sxdg q[0];\n"
                  "sxdag q[0];\n"
                  "k q[0];\n"
                  "csx q[0], q[1];\n"
                  "csxdg q[0], q[1];\n"
                  "csxdag q[0], q[1];\n"
                  "u(0.1, 0.2, 0.3) q[0];\n");
}

BOOST_AUTO_TEST_CASE(OldStyleRegisterDeclarationsStayLegalUnderQasm3) {
  // Not a leniency: the QASM3 `statement` production includes
  // oldStyleDeclarationStatement, so `qreg`/`creg` under a 3.0 header is
  // conformant and must keep parsing. Pinned because every group above uses
  // the QASM2 spelling under all three headers and would silently stop
  // testing anything if this regressed.
  CheckAccepted(kQasm3Header,
                "qreg q[2];\n"
                "creg c[2];\n"
                "h q[0];\n"
                "measure q[0] -> c[0];\n");
}

BOOST_AUTO_TEST_SUITE_END()
