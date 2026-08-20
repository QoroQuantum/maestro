#include "qasm_test_utils.h"

BOOST_AUTO_TEST_SUITE(qasm_dialect_tests)

using qasm_test::CheckDifferentProbabilities;
using qasm_test::CheckRejectedWithMessage;
using qasm_test::CheckSameProbabilities;

// QASM3-only syntax and gate names must not parse
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
  CheckDifferentProbabilities(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "rx(1) q[0];\n",
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
  CheckDifferentProbabilities(
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "rx(2 ^ 3) q[0];\n",
      "OPENQASM 3.0;\n"
      "qreg q[1];\n"
      "rx(8) q[0];\n",
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

// ****************************************************************************

BOOST_AUTO_TEST_SUITE_END()
