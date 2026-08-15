#include "qasm_test_utils.h"

BOOST_AUTO_TEST_SUITE(qasm_modifier_tests)

using qasm_test::CheckDifferentProbabilities;
using qasm_test::CheckRejectedWithMessage;
using qasm_test::CheckSameProbabilities;
using qasm_test::NthGateType;

// Gate modifiers (`ctrl @`, `negctrl @`, `inv @`, `pow(k) @`).

BOOST_AUTO_TEST_CASE(ModifiersRejectedUnderQasm2AndWithNoVersion) {
  // The modifier rules are gated on isQasm3, so under 2.0 (or with no version
  // line at all) `ctrl @ x ...` is not a gate call and the input is left
  // unparsed.
  for (const std::string &versionLine :
       {std::string("OPENQASM 2.0;\n"), std::string()}) {
    for (const std::string modifier :
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
  for (const std::string call : {"ctrl @ ctrl @ x q[0], q[1], q[2];\n",
                                 "ctrl @ cx q[0], q[1], q[2];\n"}) {
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

BOOST_AUTO_TEST_CASE(CtrlOnGlobalPhaseNamesTheUnsupportedConstruct) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "ctrl @ gphase(0.5) q[0];\n",
      "Controlled global phase ('ctrl @ gphase') is not supported");
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
// two shapes into it: `cxgateCall` falls through to `gatecall` on an
// over-long argument list, and `expGatecall`'s trailing
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

  for (const std::string preparation :
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
  for (const std::string call :
       {"pow(0) @ x q[0];\n", "ctrl @ id q[0], q[1];\n",
        "negctrl @ x q[0], q[1];\n", "ctrl @ x q[0], q[1];\n"}) {
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

BOOST_AUTO_TEST_CASE(ModifierArgumentsRejectNonFiniteExpressions) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2] q;\n"
      "ctrl(ln(-1)) @ x q[0], q[1];\n",
      "control count of ctrl(n) @ / negctrl(n) @ must be finite");

  for (const std::string gate : {"rx(0.2)", "x"})
    CheckRejectedWithMessage(
        "OPENQASM 3.0;\n"
        "qubit[1] q;\n"
        "pow(ln(-1)) @ " +
            gate + " q[0];\n",
        "pow(k) @ requires a finite exponent");
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

BOOST_AUTO_TEST_SUITE_END()
