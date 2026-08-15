#include "qasm_test_fixture.h"

BOOST_AUTO_TEST_SUITE(qasm_core_tests)

using qasm_test::CheckDifferentProbabilities;
using qasm_test::MakeInitializedSimulator;
using qasm_test::NthGateType;

BOOST_FIXTURE_TEST_CASE(PhaseAndCphaseMatchPAndCpProbabilities,
                        QasmTestFixture) {
  // 'phase'/'cphase' are stdgates.inc names, so both programs carry a 3.0
  // header; 'p'/'cp' are accepted in both dialects.
  const std::string pStyleQasm =
      "OPENQASM 3.0;\n"
      "qreg q[3];\n"
      "h q[0];\n"
      "p(0.5) q[0];\n"
      "h q[0];\n"
      "x q[1];\n"
      "h q[2];\n"
      "cp(0.3) q[1], q[2];\n"
      "h q[2];\n";
  const std::string phaseStyleQasm =
      "OPENQASM 3.0;\n"
      "qreg q[3];\n"
      "h q[0];\n"
      "phase(0.5) q[0];\n"
      "h q[0];\n"
      "x q[1];\n"
      "h q[2];\n"
      "cphase(0.3) q[1], q[2];\n"
      "h q[2];\n";
  const std::string zeroPhaseQasm =
      "OPENQASM 3.0;\n"
      "qreg q[3];\n"
      "h q[0];\n"
      "phase(0) q[0];\n"
      "h q[0];\n"
      "x q[1];\n"
      "h q[2];\n"
      "cphase(0) q[1], q[2];\n"
      "h q[2];\n";

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

  CheckDifferentProbabilities(phaseStyleQasm, zeroPhaseQasm, 3);
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
// QASM3 declarations - `qubit[n]` / `bit[n]` and the bare (size-1)
// spellings `qubit` / `bit`.

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
  BOOST_TEST_REQUIRE(qasm3Circuit != nullptr);
  BOOST_TEST(qasm3Circuit->GetQubits().size() == 2u);

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

// ****************************************************************************
// Measurement assignment (`c[0] = measure q[0];`) and braced
// conditionals (`if (c == 1) { ... }`).

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
  for (const bool prepareCondition : {false, true}) {
    const std::string prep = prepareCondition ? "x q[0];\n" : "";
    const std::string qasm3Str =
        "OPENQASM 3.0;\n"
        "qreg q[2];\n"
        "creg c[1];\n" +
        prep +
        "measure q[0] -> c[0];\n"
        "if (c==1) { x q[0]; x q[1]; }\n";
    const std::string qasm2TwinStr =
        "OPENQASM 2.0;\n"
        "qreg q[2];\n"
        "creg c[1];\n" +
        prep +
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
              << i << "> with prepareCondition=" << prepareCondition << ": "
              << qc3->Probability(i) << " vs " << qc2->Probability(i));
    }
  }
}

BOOST_AUTO_TEST_CASE(QASM3NestedBracedConditionalIsRejected) {
  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "bit c;\n"
      "if (c[0]) { if (!c[0]) { x q[1]; } }\n");
  BOOST_TEST(parser.Failed());
  BOOST_TEST(parser.GetErrorMessage().find("Unparsed input") !=
             std::string::npos);
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
  for (const std::string versionLine : {"OPENQASM 2.0;\n", "OPENQASM 3.0;\n"}) {
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

BOOST_AUTO_TEST_SUITE_END()
