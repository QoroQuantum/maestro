#ifndef MAESTRO_TESTS_QASM_TEST_UTILS_H_
#define MAESTRO_TESTS_QASM_TEST_UTILS_H_

#include <boost/test/unit_test.hpp>

#include <complex>
#include <memory>
#include <string>
#include <unordered_map>

#include "../Circuit/Circuit.h"
#include "../Circuit/Factory.h"
#include "../Circuit/QuantumGates.h"
#include "../Simulators/Factory.h"
#include "../qasm/QasmCirc.h"

extern bool checkClose(std::complex<double> a, std::complex<double> b,
                       double dif);

namespace qasm_test {

inline Circuits::QuantumGateType NthGateType(
    const std::shared_ptr<Circuits::Circuit<>> &circuit, size_t nr) {
  BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() > nr);
  const auto &operation = circuit->GetOperations()[nr];
  BOOST_TEST_REQUIRE((operation->GetType() == Circuits::OperationType::kGate));
  return std::static_pointer_cast<Circuits::IQuantumGate<>>(operation)
      ->GetGateType();
}

inline std::shared_ptr<Simulators::ISimulator> MakeInitializedSimulator(
    unsigned int nrQubits) {
  auto simulator = Simulators::SimulatorsFactory::CreateSimulator(
      Simulators::SimulatorType::kQCSim,
      Simulators::SimulationType::kStatevector);
  simulator->AllocateQubits(nrQubits);
  simulator->Initialize();
  return simulator;
}

inline void CheckRejectedWithMessage(const std::string &qasmStr,
                                     const std::string &expectedMessage) {
  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(parser.Failed(), "Expected parser failure for:\n" << qasmStr);
  if (parser.Failed()) {
    BOOST_TEST(
        parser.GetErrorMessage().find(expectedMessage) != std::string::npos,
        "Expected error containing '"
            << expectedMessage << "', got: " << parser.GetErrorMessage());
  }
}

inline void CheckSameProbabilities(const std::string &qasmStr,
                                   const std::string &referenceQasmStr,
                                   unsigned int nrQubits) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  qasm::QasmToCirc<> referenceParser;
  auto reference = referenceParser.ParseAndTranslate(referenceQasmStr);
  BOOST_TEST(!referenceParser.Failed(), referenceParser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!referenceParser.Failed());

  auto simulator = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState state(nrQubits);
  circuit->Execute(simulator, state);

  auto referenceSimulator = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState referenceState(nrQubits);
  reference->Execute(referenceSimulator, referenceState);

  for (size_t basisState = 0; basisState < (1ULL << nrQubits); ++basisState) {
    BOOST_TEST(checkClose(
        std::complex<double>(simulator->Probability(basisState), 0.),
        std::complex<double>(referenceSimulator->Probability(basisState), 0.),
        0.0001));
  }
}

inline void CheckDifferentProbabilities(const std::string &qasmStr,
                                        const std::string &referenceQasmStr,
                                        unsigned int nrQubits) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  qasm::QasmToCirc<> referenceParser;
  auto reference = referenceParser.ParseAndTranslate(referenceQasmStr);
  BOOST_TEST(!referenceParser.Failed(), referenceParser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!referenceParser.Failed());

  auto simulator = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState state(nrQubits);
  circuit->Execute(simulator, state);

  auto referenceSimulator = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState referenceState(nrQubits);
  reference->Execute(referenceSimulator, referenceState);

  bool differs = false;
  for (size_t basisState = 0; basisState < (1ULL << nrQubits); ++basisState) {
    differs |= !checkClose(
        std::complex<double>(simulator->Probability(basisState), 0.),
        std::complex<double>(referenceSimulator->Probability(basisState), 0.),
        0.0001);
  }
  BOOST_TEST(differs);
}

inline void CheckBoundInputMatchesLiteral(
    const std::string &qasmStr,
    const std::unordered_map<std::string, double> &params,
    const std::string &referenceQasmStr, unsigned int nrQubits) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslateWithParams(qasmStr, params);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  qasm::QasmToCirc<> referenceParser;
  auto reference = referenceParser.ParseAndTranslate(referenceQasmStr);
  BOOST_TEST(!referenceParser.Failed(), referenceParser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!referenceParser.Failed());

  auto simulator = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState state(nrQubits);
  circuit->Execute(simulator, state);

  auto referenceSimulator = MakeInitializedSimulator(nrQubits);
  Circuits::OperationState referenceState(nrQubits);
  reference->Execute(referenceSimulator, referenceState);

  for (size_t basisState = 0; basisState < (1ULL << nrQubits); ++basisState) {
    BOOST_TEST(checkClose(
        std::complex<double>(simulator->Probability(basisState), 0.),
        std::complex<double>(referenceSimulator->Probability(basisState), 0.),
        0.0001));
  }
}

}  // namespace qasm_test

#endif  // MAESTRO_TESTS_QASM_TEST_UTILS_H_
