#include "qasm_test_utils.h"

BOOST_AUTO_TEST_SUITE(qasm_syntax_tests)

using qasm_test::CheckDifferentProbabilities;
using qasm_test::CheckRejectedWithMessage;
using qasm_test::CheckSameProbabilities;
using qasm_test::MakeInitializedSimulator;

// `measureArrowAssignmentStatement: measureExpression (ARROW
// indexedIdentifier)? SEMICOLON` - the arrow target is optional. The IR has no
// discard-measurement (CreateMeasurement takes qubit/classical-bit pairs), so
// this is a clean rejection naming the construct rather than an invented
// classical bit. What must not survive is the old misleading message, which
// blamed an "unsupported gate" called measure.

BOOST_AUTO_TEST_CASE(MeasurementWithoutTargetIsRejectedNamingTheConstruct) {
  for (const std::string &qasmStr : {std::string("OPENQASM 3.0;\n"
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
  for (const std::string &measurement :
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
// `barrierStatement: BARRIER gateOperandList? SEMICOLON` - the operand
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
  qasm::QasmSkipper<std::string::iterator> skipper;
  const bool parsed = boost::spirit::qi::phrase_parse(it, input.end(), grammar,
                                                      skipper, program);

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
// `gateModifier: ... (CTRL | NEGCTRL) (LPAREN expression RPAREN)? AT` -
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
  for (const std::string &preparation :
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
  for (const std::string &count :
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

BOOST_AUTO_TEST_SUITE_END()
