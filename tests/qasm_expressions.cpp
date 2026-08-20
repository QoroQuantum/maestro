#include "qasm_test_utils.h"

BOOST_AUTO_TEST_SUITE(qasm_expression_tests)

using qasm_test::CheckDifferentProbabilities;
using qasm_test::CheckRejectedWithMessage;
using qasm_test::CheckSameProbabilities;
using qasm_test::NthGateType;

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
  // spelled differently. See
  // QASM2CaretHasTheSameUnaryPrecedenceAsDoubleAsterisk.
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

BOOST_AUTO_TEST_CASE(BuiltinUAndCXOverlongCallsAreRejected) {
  // An over-long argument list on either still names the arity, rather
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
    // The whole-register broadcast remains accepted and contributes nothing.
    // The indexed form is covered by the exhaustive gate-contract table.
    for (const std::string &idCall : {std::string("id q;\n")}) {
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
}

// ****************************************************************************

BOOST_AUTO_TEST_SUITE_END()
