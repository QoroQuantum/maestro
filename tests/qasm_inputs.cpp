#include "qasm_test_utils.h"

#include <limits>

BOOST_AUTO_TEST_SUITE(qasm_input_tests)

using qasm_test::CheckBoundInputMatchesLiteral;
using qasm_test::CheckRejectedWithMessage;
using qasm_test::MakeInitializedSimulator;

namespace {

void CheckBindingRejected(
    const std::string &qasmStr,
    const std::unordered_map<std::string, double> &bindings,
    const std::string &messageFragment) {
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslateWithParams(qasmStr, bindings);
  BOOST_TEST(parser.Failed(), "Expected parser failure for:\n" << qasmStr);
  BOOST_TEST(circuit == nullptr);
  BOOST_TEST(
      parser.GetErrorMessage().find(messageFragment) != std::string::npos,
      "Expected error containing '" << messageFragment
                                    << "', got: " << parser.GetErrorMessage());
}

}  // namespace

// QASM3 `input` declarations are bound to concrete
// values at parse time via ParseAndTranslate's params map. No symbolic
// parameter survives into the Circuit: an unbound one is a parse error.

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
  auto circuit = parser.ParseAndTranslateWithParams(
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

BOOST_AUTO_TEST_CASE(UnboundInputStillReportsTheRequiredNames) {
  qasm::QasmToCirc<> parser;
  parser.ParseAndTranslate(
      "OPENQASM 3.0;\n"
      "input float theta;\n"
      "input int shots;\n"
      "qubit q;\n"
      "rx(theta) q;\n");

  BOOST_TEST(parser.Failed());
  BOOST_TEST(parser.GetInputs() ==
             std::vector<std::string>({"theta", "shots"}));
}

BOOST_AUTO_TEST_CASE(ParseAndTranslateClearsInputsBetweenPrograms) {
  qasm::QasmToCirc<> parser;
  auto parameterized = parser.ParseAndTranslateWithParams(
      "OPENQASM 3.0;\n"
      "input float theta;\n"
      "qubit q;\n"
      "rx(theta) q;\n",
      {{"theta", 0.5}});
  BOOST_TEST_REQUIRE(!parser.Failed());
  BOOST_TEST_REQUIRE(parameterized != nullptr);
  BOOST_TEST(parser.GetInputs() == std::vector<std::string>({"theta"}));

  auto qasm2 = parser.ParseAndTranslate(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "x q[0];\n");
  BOOST_TEST_REQUIRE(!parser.Failed());
  BOOST_TEST_REQUIRE(qasm2 != nullptr);
  BOOST_TEST(parser.GetInputs().empty());
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
  auto circuit = parser.ParseAndTranslateWithParams(
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

BOOST_AUTO_TEST_CASE(InputBindingsRequirePriorQasm3Declarations) {
  CheckBindingRejected("OPENQASM 3.0;\nqubit q;\nrx(theta) q;\n",
                       {{"theta", 0.7}}, "Variable not found: theta");
  CheckBindingRejected(
      "OPENQASM 3.0;\nqubit q;\nrx(theta) q;\ninput float theta;\n",
      {{"theta", 0.7}}, "Variable not found: theta");
  CheckBindingRejected("OPENQASM 2.0;\nqreg q[1];\nrx(theta) q[0];\n",
                       {{"theta", 0.7}}, "Variable not found: theta");
  CheckBindingRejected("OPENQASM 3.0;\nqubit q;\n", {{"ghost", 1.}},
                       "No input declaration for binding 'ghost'");
}

BOOST_AUTO_TEST_CASE(InputBindingsRespectDeclaredScalarTypes) {
  const auto program = [](const std::string &declaration,
                          const std::string &name) {
    return "OPENQASM 3.0;\ninput " + declaration + " " + name +
           ";\nqubit q;\nrx(" + name + ") q;\n";
  };

  CheckBindingRejected(program("bool", "b"), {{"b", 0.5}}, "boolean");
  CheckBindingRejected(program("int[32]", "n"), {{"n", 1.5}}, "integer");
  CheckBindingRejected(program("uint[32]", "n"), {{"n", -1.}}, "non-negative");
  CheckBindingRejected(program("int[2]", "n"), {{"n", 2.}}, "does not fit");
  CheckBindingRejected(program("uint[2]", "n"), {{"n", 4.}}, "does not fit");
  CheckBindingRejected(program("float", "x"),
                       {{"x", std::numeric_limits<double>::infinity()}},
                       "finite");
  CheckBindingRejected(program("float[32]", "x"), {{"x", 1e300}}, "float[32]");
  CheckBindingRejected(program("angle[8]", "theta"), {{"theta", 0.7}},
                       "sized angle inputs");
}

BOOST_AUTO_TEST_CASE(
    InputDeclarationsRejectUnsupportedTypeShapesWithoutBindings) {
  const std::pair<std::string, std::string> cases[] = {
      {"bool[8] b", "cannot have a width"},
      {"float[16] x", "unsupported float width"},
      {"angle[8] theta", "sized angle inputs"},
      {"int[0] n", "positive type width"},
      {"uint[65] n", "width above 64"},
  };

  for (const auto &[declaration, message] : cases)
    CheckBindingRejected(
        "OPENQASM 3.0;\ninput " + declaration + ";\nqubit q;\nx q;\n", {},
        message);
}

BOOST_AUTO_TEST_CASE(Float32InputBindingUsesFloat32Precision) {
  constexpr double supplied = 0.123456789012345;
  const double expected = static_cast<double>(static_cast<float>(supplied));

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslateWithParams(
      "OPENQASM 3.0;\ninput float[32] theta;\nqubit q;\nrx(theta) q;\n",
      {{"theta", supplied}});
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(circuit != nullptr);

  const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<>>(
      circuit->GetOperation(0));
  BOOST_TEST(gate->GetParams()[0] == expected);
}

BOOST_AUTO_TEST_CASE(UnsizedAngleInputBindingIsNormalized) {
  constexpr double supplied = -0.25;
  const double expected = 2. * M_PI + supplied;

  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslateWithParams(
      "OPENQASM 3.0;\ninput angle theta;\nqubit q;\nrx(theta) q;\n",
      {{"theta", supplied}});
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
  BOOST_TEST_REQUIRE(circuit != nullptr);

  const auto gate = std::static_pointer_cast<Circuits::IQuantumGate<>>(
      circuit->GetOperation(0));
  BOOST_TEST(gate->GetParams()[0] == expected,
             boost::test_tools::tolerance(1e-12));
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
  // Phase-kickback probe, the same pattern
  // CtrlAndInvCarryTheGlobalPhaseParameterOfU above uses for u's global-phase
  // parameter: a controlled-rz on a |0> target changes no basis populations by
  // itself, but with the control qubit sandwiched in h ... h, the phase becomes
  // an observable interference pattern.
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

// Unsupported-construct diagnostics. Each named reserved keyword must be
// rejected with a message naming the actual construct,
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
      {"output float result;\n", "output declarations"},
      {"const int[32] n = 5;\n", "constant declarations"},
      {"extern foo(qubit q);\n", "external subroutines"},
      {"stretch d;\n", "stretch declarations"},
      {"pragma maestro test\n", "pragma statements"},
      {"defcal x $0 { }\n", "calibration definitions"},
  };

  for (const auto &[construct, fragment] : cases) {
    const std::string qasmStr = "OPENQASM 3.0;\nqubit[1] q;\n" + construct;
    CheckRejectedWithMessage(qasmStr, fragment);
  }
}

// Keyword boundary: a gate named "format", "delayed", "boxcar" or
// "arrayed" must still parse as an ordinary gate call under OPENQASM 3.0.
// Each of these is a defined gate whose name merely starts with one of the
// eight reserved keywords; without the `!qi::char_("a-zA-Z0-9_")` boundary
// guard, `unsupportedConstruct` would match the keyword prefix (e.g. "for"
// inside "format") and misreport these as the reserved construct.
BOOST_AUTO_TEST_CASE(IdentifierStartingWithReservedKeywordStillParses) {
  for (const std::string gateName :
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

// Version gating: "for" used as an ordinary gate name must still
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

// ****************************************************************************

BOOST_AUTO_TEST_SUITE_END()
