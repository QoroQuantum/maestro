#include "../qasm/CircQasm.h"
#include "qasm_test_fixture.h"

BOOST_AUTO_TEST_SUITE(qasm_integration_tests)

using qasm_test::CheckDifferentProbabilities;
using qasm_test::CheckRejectedWithMessage;
using qasm_test::CheckSameProbabilities;
using qasm_test::MakeInitializedSimulator;
using qasm_test::NthGateType;

// Targeted export assertions complement randomized round-trip coverage. The
// round-trip cases prove behavioural equivalence, but not that the
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

std::shared_ptr<Circuits::Circuit<>> MakeConditionalExportCircuit() {
  auto circuit = std::make_shared<Circuits::Circuit<>>();
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(
      Circuits::CircuitFactory<>::CreateGate(
          Circuits::QuantumGateType::kXGateType, 1),
      Circuits::CircuitFactory<>::CreateEqualCondition({0}, {true})));
  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateConditionalMeasurement(
          std::make_shared<Circuits::MeasurementOperation<>>(
              std::vector<std::pair<Types::qubit_t, size_t>>{{1, 1}}),
          Circuits::CircuitFactory<>::CreateEqualCondition({0}, {false})));
  return circuit;
}

std::shared_ptr<Circuits::Circuit<>> MakeMultiBitConditionalExportCircuit(
    bool firstBit, bool secondBit) {
  auto circuit = std::make_shared<Circuits::Circuit<>>();
  if (firstBit)
    circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
        Circuits::QuantumGateType::kXGateType, 0));
  if (secondBit)
    circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
        Circuits::QuantumGateType::kXGateType, 1));
  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateMeasurement({{0, 0}, {1, 1}}));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(
      Circuits::CircuitFactory<>::CreateGate(
          Circuits::QuantumGateType::kXGateType, 2),
      Circuits::CircuitFactory<>::CreateEqualCondition({0, 1}, {true, false})));
  return circuit;
}

std::shared_ptr<Circuits::Circuit<>> MakeConditionalMeasurementRoundTripCircuit(
    bool predicateValue = true) {
  auto circuit = std::make_shared<Circuits::Circuit<>>();
  if (predicateValue)
    circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
        Circuits::QuantumGateType::kXGateType, 0));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kXGateType, 1));
  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateMeasurement({{0, 0}}));
  circuit->AddOperation(
      Circuits::CircuitFactory<>::CreateConditionalMeasurement(
          std::make_shared<Circuits::MeasurementOperation<>>(
              std::vector<std::pair<Types::qubit_t, size_t>>{{1, 1}}),
          Circuits::CircuitFactory<>::CreateEqualCondition({0}, {true})));
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateConditionalGate(
      Circuits::CircuitFactory<>::CreateGate(
          Circuits::QuantumGateType::kXGateType, 2),
      Circuits::CircuitFactory<>::CreateEqualCondition({1}, {true})));
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

  BOOST_TEST(qasmStr.find("\nqubit[") != std::string::npos);
  BOOST_TEST(qasmStr.find("\nbit[") != std::string::npos);
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

BOOST_AUTO_TEST_CASE(V3ExportUsesBracedBitConditions) {
  const std::string qasmStr = qasm::CircToQasm<>::Generate(
      MakeConditionalExportCircuit(), qasm::CircToQasm<>::QasmVersion::V3);

  BOOST_TEST(qasmStr.find("if (c0[0]) {\n  x q[1];\n}\n") != std::string::npos);
  BOOST_TEST(qasmStr.find("if (!c0[0]) {\n  c1[0] = measure q[1];\n}\n") !=
             std::string::npos);

  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
}

BOOST_AUTO_TEST_CASE(V3ExportCombinesMultiBitConditions) {
  for (const bool firstBit : {false, true}) {
    for (const bool secondBit : {false, true}) {
      const std::string qasmStr = qasm::CircToQasm<>::Generate(
          MakeMultiBitConditionalExportCircuit(firstBit, secondBit),
          qasm::CircToQasm<>::QasmVersion::V3);

      BOOST_TEST(qasmStr.find("if (c0[0] && !c1[0]) {\n  x q[2];\n}\n") !=
                 std::string::npos);

      qasm::QasmToCirc<> parser;
      auto circuit = parser.ParseAndTranslate(qasmStr);
      BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
      BOOST_TEST_REQUIRE(!parser.Failed());

      auto qc = MakeInitializedSimulator(3);
      Circuits::OperationState state(2);
      circuit->Execute(qc, state);

      const bool conditionMet = firstBit && !secondBit;
      const int expectedOutcome = static_cast<int>(firstBit) +
                                  2 * static_cast<int>(secondBit) +
                                  4 * static_cast<int>(conditionMet);
      BOOST_TEST(
          checkClose(std::complex<double>(qc->Probability(expectedOutcome), 0.),
                     std::complex<double>(1., 0.), 0.0001),
          "c0=" << firstBit << ", c1=" << secondBit
                << " produced the wrong conjunction result");
    }
  }
}

BOOST_AUTO_TEST_CASE(QASM3LongBitConjunctionPreservesHighOrderValues) {
  std::string condition;
  for (int bit = 0; bit < 32; ++bit) {
    if (!condition.empty()) condition += " && ";
    condition += "!c[" + std::to_string(bit) + "]";
  }
  condition += " && c[32]";

  const std::string qasmStr =
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "bit[34] c;\n"
      "x q[0];\n"
      "c[32] = measure q[0];\n"
      "if (" +
      condition +
      ") { x q[1]; }\n"
      "c[33] = measure q[1];\n";

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(!parser.Failed());

  auto qc = MakeInitializedSimulator(2);
  Circuits::OperationState state(34);
  circuit->Execute(qc, state);
  BOOST_TEST(state.GetBit(33));
}

BOOST_AUTO_TEST_CASE(V2ConditionalExportRetainsLegacySpelling) {
  const std::string qasmStr = qasm::CircToQasm<>::Generate(
      MakeConditionalExportCircuit(), qasm::CircToQasm<>::QasmVersion::V2);

  BOOST_TEST(qasmStr.find("if(c0==1) x q[1];\n") != std::string::npos);
}

// The default export remains byte-identical to an explicit V2 export:
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

BOOST_AUTO_TEST_CASE(SxExportUsesALocalDefinitionOnlyForV2) {
  auto circuit = std::make_shared<Circuits::Circuit<>>();
  circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
      Circuits::QuantumGateType::kSxGateType, 0));

  const std::string qasm2 = qasm::CircToQasm<>::Generate(
      circuit, qasm::CircToQasm<>::QasmVersion::V2);
  const std::string qasm3 = qasm::CircToQasm<>::Generate(
      circuit, qasm::CircToQasm<>::QasmVersion::V3);

  BOOST_TEST(qasm2.find("include \"qelib1.inc\";") != std::string::npos);
  BOOST_TEST(qasm2.find("gate sx a") != std::string::npos);
  BOOST_TEST(qasm3.find("include \"stdgates.inc\";") != std::string::npos);
  BOOST_TEST(qasm3.find("gate sx a") == std::string::npos);
}

BOOST_AUTO_TEST_CASE(ControlledGateExportUsesDialectSpecificNames) {
  const auto generate = [](Circuits::QuantumGateType gateType, double p1,
                           double p2 = 0., double p3 = 0., double p4 = 0.,
                           qasm::CircToQasm<>::QasmVersion version =
                               qasm::CircToQasm<>::QasmVersion::V3) {
    auto circuit = std::make_shared<Circuits::Circuit<>>();
    circuit->AddOperation(Circuits::CircuitFactory<>::CreateGate(
        gateType, 0, 1, 0, p1, p2, p3, p4));
    return qasm::CircToQasm<>::Generate(circuit, version);
  };

  const auto reparseExport = [](const std::string &qasmStr,
                                Circuits::QuantumGateType expectedType,
                                const std::string &expectedInclude) {
    BOOST_TEST(qasmStr.find(expectedInclude) != std::string::npos);
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(circuit != nullptr);
    BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() == 1u);
    const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<>>(
        circuit->GetOperation(0));
    BOOST_TEST((gate->GetGateType() == expectedType));
    return gate;
  };

  const std::string cpV3 =
      generate(Circuits::QuantumGateType::kCPGateType, 0.2);
  BOOST_TEST(cpV3.find("cp(0.200000) q[0],q[1];") != std::string::npos);
  BOOST_TEST(cpV3.find("cu1(") == std::string::npos);
  reparseExport(cpV3, Circuits::QuantumGateType::kCPGateType,
                "include \"stdgates.inc\";");

  for (const double gamma : {0., 0.5}) {
    const std::string cuV3 =
        generate(Circuits::QuantumGateType::kCUGateType, 0.2, 0.3, 0.4, gamma);
    BOOST_TEST(cuV3.find("cu(0.200000,0.300000,0.400000," +
                         std::to_string(gamma) + ") q[0], q[1];") !=
               std::string::npos);
    BOOST_TEST(cuV3.find("cu3(") == std::string::npos);
    const auto gate =
        reparseExport(cuV3, Circuits::QuantumGateType::kCUGateType,
                      "include \"stdgates.inc\";");
    BOOST_TEST_REQUIRE(gate->GetParams().size() == 4u);
    BOOST_TEST(gate->GetParams()[3] == gamma);
  }

  for (const auto gateType : {Circuits::QuantumGateType::kCRxGateType,
                              Circuits::QuantumGateType::kCRyGateType}) {
    const std::string gateName =
        gateType == Circuits::QuantumGateType::kCRxGateType ? "crx" : "cry";
    const std::string qasm3 = generate(gateType, 0.2);
    BOOST_TEST(qasm3.find("gate " + gateName) == std::string::npos);
    BOOST_TEST(qasm3.find("cu3(") == std::string::npos);
    const auto gate =
        reparseExport(qasm3, gateType, "include \"stdgates.inc\";");
    BOOST_TEST_REQUIRE(!gate->GetParams().empty());
    BOOST_TEST(gate->GetParams()[0] == 0.2);

    const std::string qasm2 = generate(gateType, 0.2, 0., 0., 0.,
                                       qasm::CircToQasm<>::QasmVersion::V2);
    BOOST_TEST(qasm2.find("gate " + gateName + "(theta)") != std::string::npos);
    BOOST_TEST(qasm2.find(gateName + "(0.200000) q[0],q[1];") !=
               std::string::npos);
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasm2);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(circuit != nullptr);
    BOOST_TEST_REQUIRE(circuit->GetNumberOfOperations() == 1u);
    const auto expandedGate =
        std::static_pointer_cast<Circuits::IQuantumGate<>>(
            circuit->GetOperation(0));
    BOOST_TEST_REQUIRE(!expandedGate->GetParams().empty());
    BOOST_TEST(expandedGate->GetParams()[0] == 0.2);
  }

  const auto v2 = qasm::CircToQasm<>::QasmVersion::V2;
  BOOST_TEST(
      generate(Circuits::QuantumGateType::kCPGateType, 0.2, 0., 0., 0., v2)
          .find("cu1(0.200000)") != std::string::npos);
  BOOST_TEST(
      generate(Circuits::QuantumGateType::kCUGateType, 0.2, 0.3, 0.4, 0., v2)
          .find("cu3(0.200000,0.300000,0.400000)") != std::string::npos);
  BOOST_CHECK_THROW(
      generate(Circuits::QuantumGateType::kCUGateType, 0.2, 0.3, 0.4, 0.5, v2),
      std::runtime_error);
}

BOOST_AUTO_TEST_CASE(ConditionalMeasurementExportRoundTripsSemantics) {
  const auto executeExport = [](const std::string &qasmStr, bool predicateValue,
                                size_t predicateBit = 0,
                                size_t measuredBit = 1) {
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(circuit != nullptr);

    auto qc = MakeInitializedSimulator(3);
    Circuits::OperationState state(3);
    circuit->Execute(qc, state);
    const size_t expectedState = predicateValue ? 7u : 2u;
    BOOST_TEST(
        checkClose(std::complex<double>(qc->Probability(expectedState), 0.),
                   std::complex<double>(1., 0.), 0.0001));
    BOOST_TEST(state.GetBit(predicateBit) == predicateValue);
    BOOST_TEST(state.GetBit(measuredBit) == predicateValue);
  };

  for (const auto version : {qasm::CircToQasm<>::QasmVersion::V2,
                             qasm::CircToQasm<>::QasmVersion::V3}) {
    for (const bool predicateValue : {false, true})
      executeExport(
          qasm::CircToQasm<>::Generate(
              MakeConditionalMeasurementRoundTripCircuit(predicateValue),
              version),
          predicateValue);
  }

  const std::unordered_map<Types::qubit_t, Types::qubit_t> reversed = {
      {0, 2}, {1, 1}, {2, 0}};
  const std::string mappedQasm = qasm::CircToQasm<>::GenerateWithMapping(
      MakeConditionalMeasurementRoundTripCircuit(), reversed,
      qasm::CircToQasm<>::QasmVersion::V3);
  BOOST_TEST(mappedQasm.find("x q[2];\n") != std::string::npos);
  BOOST_TEST(mappedQasm.find("c2[0] = measure q[2];\n") != std::string::npos);
  BOOST_TEST(mappedQasm.find("if (c2[0]) {\n  c1[0] = measure q[1];\n}\n") !=
             std::string::npos);
  BOOST_TEST(mappedQasm.find("if (c1[0]) {\n  x q[0];\n}\n") !=
             std::string::npos);
  executeExport(mappedQasm, true, 2, 1);
}

// ****************************************************************************
// Export and statement-order boundary coverage.

BOOST_AUTO_TEST_CASE(
    QASM3StatementAfterBracedConditionalAppliesAfterNotBefore) {
  // A statement after a braced
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
  // `if (c == 1) { }` is defensible as a no-op but was
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

// ****************************************************************************
// Qiskit-dialect fixtures. Qiskit's qasm3.dumps()
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

// ****************************************************************************
// Deliberately awkward hand-written QASM3. Real
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
  for (const bool negateCondition : {false, true}) {
    for (const bool prepareBit : {true, false}) {
      const std::string prep = prepareBit ? "x q[0];\n" : "";
      const std::string head = negateCondition ? "!c[0]" : "c[0]";
      const std::string qasmStr =
          "OPENQASM 3.0;\n"
          "qreg q[3];\n"
          "creg c[1];\n" +
          prep +
          "measure q[0] -> c[0];\n"
          "if (" +
          head + ") { x q[1]; } else { x q[2]; }\n";

      qasm::QasmToCirc<> parser;
      auto circuit = parser.ParseAndTranslate(qasmStr);
      BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
      BOOST_TEST_REQUIRE(!parser.Failed());

      auto qc = MakeInitializedSimulator(3);
      Circuits::OperationState state(3);
      circuit->Execute(qc, state);

      const bool conditionMet = negateCondition ? !prepareBit : prepareBit;
      const int expectedOutcome = static_cast<int>(prepareBit) +
                                  2 * static_cast<int>(conditionMet) +
                                  4 * static_cast<int>(!conditionMet);
      BOOST_TEST(
          checkClose(std::complex<double>(qc->Probability(expectedOutcome), 0.),
                     std::complex<double>(1., 0.), 0.0001),
          "if/else with negateCondition="
              << negateCondition << " and prepareBit=" << prepareBit
              << " gave P(|" << expectedOutcome
              << ">) = " << qc->Probability(expectedOutcome) << ", expected 1");
    }
  }
}

BOOST_AUTO_TEST_CASE(QASM3BracedConditionRejectsPredicateMutation) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "bit[3] c;\n"
      "x q[0];\n"
      "c[0] = measure q[0];\n"
      "if (c[0]) { c[0] = measure q[1]; x q[2]; }\n",
      "changes a predicate bit");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "bit[1] c;\n"
      "x q[0];\n"
      "c[0] = measure q[0];\n"
      "if (c[0]) { c[0] = measure q[1]; } else { x q[2]; }\n",
      "changes a predicate bit");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "bit[1] c;\n"
      "if (c[0]) { x q[2]; } else { c[0] = measure q[1]; x q[2]; }\n",
      "changes a predicate bit");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "bit[1] c;\n"
      "if (c == 0) { c[0] = measure q[1]; x q[2]; }\n",
      "changes a predicate bit");

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "bit[1] c;\n"
      "x q[1];\n"
      "if (!c[0]) { c[0] = measure q[1]; }\n");
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(circuit != nullptr);
  auto qc = MakeInitializedSimulator(2);
  Circuits::OperationState state(1);
  circuit->Execute(qc, state);
  BOOST_TEST(state.GetBit(0));
}

BOOST_AUTO_TEST_CASE(ConditionalMeasurementExecutesInBothDialects) {
  const std::vector<std::string> programs = {
      "OPENQASM 2.0;\n"
      "qreg q[3];\n"
      "creg c[2];\n"
      "x q[0];\n"
      "x q[1];\n"
      "measure q[0] -> c[0];\n"
      "if (c==1) measure q[1] -> c[1];\n"
      "if (c==3) x q[2];\n",
      "OPENQASM 3.0;\n"
      "qubit[3] q;\n"
      "bit[2] c;\n"
      "x q[0];\n"
      "x q[1];\n"
      "c[0] = measure q[0];\n"
      "if (c[0]) { c[1] = measure q[1]; }\n"
      "if (c[1]) { x q[2]; }\n"};

  for (const auto &qasmStr : programs) {
    qasm::QasmToCirc<> parser;
    auto circuit = parser.ParseAndTranslate(qasmStr);
    BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
    BOOST_TEST_REQUIRE(!parser.Failed());

    auto qc = MakeInitializedSimulator(3);
    Circuits::OperationState state(2);
    circuit->Execute(qc, state);

    BOOST_TEST(checkClose(std::complex<double>(qc->Probability(7), 0.),
                          std::complex<double>(1., 0.), 0.0001),
               "Conditional measurement did not produce |111> for:\n"
                   << qasmStr);
  }
}

BOOST_AUTO_TEST_CASE(ConditionalResetIsRejectedInBothDialects) {
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "creg c[1];\n"
      "if (c==0) reset q[0];\n",
      "Conditional 'reset'");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "bit c;\n"
      "if (!c[0]) { reset q[0]; }\n",
      "Conditional 'reset'");
}

BOOST_AUTO_TEST_CASE(QASM3ElseOnMultiBitConditionIsRejected) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "bit[2] c;\n"
      "if (c[0] && !c[1]) { x q[0]; } else { z q[0]; }\n",
      "'else' on a multi-bit condition");
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

// ****************************************************************************
// Conformance gaps against the official OpenQASM 3 ANTLR grammar
// (openqasm/openqasm, source/grammar/qasm3Parser.g4).
//
// A1: '^' is bitwise XOR in QASM3 and exponentiation in QASM2; '**' is
//     exponentiation in QASM3. Keeping the QASM2 meaning unconditionally
//     turned `rx(2 ^ 3)` into rx(8) where the spec says rx(1) - a wrong
//     rotation angle with no error at all, which is why these probe the angle
//     numerically rather than asserting on the parse.

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

  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "rx(ln(-1) ^ 1) q[0];\n",
      "requires finite integer operands");
}

BOOST_AUTO_TEST_CASE(GateParametersRejectNonFiniteExpressions) {
  for (const std::string expression : {"ln(-1)", "1 / 0"})
    CheckRejectedWithMessage(
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n"
        "rx(" +
            expression + ") q[0];\n",
        "Gate parameters must be finite");
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

// ****************************************************************************
// A2: `simpleGatecall` duplicated its `identifier` prefix across two
// alternatives. Qi does not clear a std::string attribute when it backtracks
// out of a failed alternative, so the identifier accumulated: `x() q[0];` was
// reported as a call to gate "xx". Empty parentheses are also spec-legal
// (`gateCallStatement: ... (LPAREN expressionList? RPAREN)? ...`) and were
// rejected outright.

BOOST_AUTO_TEST_CASE(EmptyParameterListParsesAsTheSameCall) {
  // Applied directly to |0>, x() must move all probability to |1>; dropping
  // the call would therefore differ from the unparenthesized reference.
  CheckSameProbabilities(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "x() q[0];\n",
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "x q[0];\n",
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

BOOST_AUTO_TEST_CASE(UnsupportedEmptyParenCallNamesTheGateOnce) {
  // The doubling was only ever visible through the error message, so that is
  // where it is pinned: the name must appear exactly as written.
  for (const std::string &gateName :
       {std::string("nosuchgate"), std::string("rx")}) {
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

BOOST_AUTO_TEST_SUITE_END()
