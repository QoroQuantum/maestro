#include "qasm_test_utils.h"

BOOST_AUTO_TEST_SUITE(qasm_validation_tests)

namespace {

using qasm_test::CheckRejectedWithMessage;

const std::string kQasm2Header = "OPENQASM 2.0;\n";
const std::string kQasm3Header = "OPENQASM 3.0;\n";

void CheckAccepted(const std::string &header, const std::string &body) {
  const std::string qasmStr = header + body;
  qasm::QasmToCirc<> parser;
  auto circuit = parser.ParseAndTranslate(qasmStr);
  BOOST_TEST(!parser.Failed(), parser.GetErrorMessage() << "\n" << qasmStr);
  BOOST_TEST_REQUIRE(circuit != nullptr);
}

}  // namespace

// QASM2 and QASM3 spell registers differently,
// but both feed the same Circuit IR and therefore must enforce the same symbol
// and allocation invariants before any operation is lowered.

BOOST_AUTO_TEST_CASE(RegisterDeclarationsRequirePositiveSizes) {
  CheckRejectedWithMessage("OPENQASM 2.0;\nqreg q[0];\n",
                           "must have a positive size");
  CheckRejectedWithMessage("OPENQASM 2.0;\ncreg c[0];\n",
                           "must have a positive size");
  CheckRejectedWithMessage("OPENQASM 2.0;\nqreg q[-1];\n",
                           "must have a positive size");
  CheckRejectedWithMessage("OPENQASM 2.0;\ncreg c[-1];\n",
                           "must have a positive size");
  CheckRejectedWithMessage("OPENQASM 3.0;\nqubit[0] q;\n",
                           "must have a positive size");
  CheckRejectedWithMessage("OPENQASM 3.0;\nbit[0] c;\n",
                           "must have a positive size");
  CheckRejectedWithMessage("OPENQASM 3.0;\nqubit[-1] q;\n",
                           "must have a positive size");
  CheckRejectedWithMessage("OPENQASM 3.0;\nbit[-1] c;\n",
                           "must have a positive size");
}

BOOST_AUTO_TEST_CASE(CumulativeRegisterAllocationOverflowIsRejectedByParser) {
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg full[2147483647];\n"
      "qreg overflow[1];\n",
      "allocation exceeds supported maximum");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2147483647] full;\n"
      "qubit overflow;\n",
      "allocation exceeds supported maximum");
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "creg full[2147483647];\n"
      "creg overflow[1];\n",
      "allocation exceeds supported maximum");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "bit[2147483647] full;\n"
      "bit overflow;\n",
      "allocation exceeds supported maximum");
}

BOOST_AUTO_TEST_CASE(DuplicateDeclarationsAreRejectedInBothDialects) {
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "qreg q[1];\n",
      "Duplicate declaration of 'q'");
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "creg c[1];\n"
      "creg c[1];\n",
      "Duplicate declaration of 'c'");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "qubit q;\n",
      "Duplicate declaration of 'q'");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "bit c;\n"
      "bit c;\n",
      "Duplicate declaration of 'c'");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "input float theta;\n"
      "input float theta;\n",
      "Duplicate declaration of 'theta'");
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "gate duplicate a { x a; }\n"
      "gate duplicate a { z a; }\n",
      "Duplicate declaration of 'duplicate'");
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "opaque duplicate a;\n"
      "opaque duplicate a;\n",
      "Duplicate declaration of 'duplicate'");
}

BOOST_AUTO_TEST_CASE(GlobalDeclarationNameCollisionsAreRejected) {
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg shared[1];\n"
      "creg shared[1];\n",
      "Declaration of 'shared' conflicts");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "input float shared;\n"
      "qubit shared;\n",
      "Declaration of 'shared' conflicts");
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg shared[1];\n"
      "gate shared a { x a; }\n",
      "Declaration of 'shared' conflicts");
  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "opaque shared a;\n"
      "qreg shared[1];\n",
      "Declaration of 'shared' conflicts");
}

namespace {

void CheckRejectedOperandInBothDialects(const std::string &body,
                                        const std::string &message) {
  CheckRejectedWithMessage("OPENQASM 2.0;\n" + body, message);
  CheckRejectedWithMessage("OPENQASM 3.0;\n" + body, message);
}

}  // namespace

BOOST_AUTO_TEST_CASE(UndeclaredOperandsAreRejectedAtEveryUseSite) {
  const std::string declarations = "qreg q[1];\ncreg c[1];\n";

  CheckRejectedOperandInBothDialects(declarations + "x missing[0];\n",
                                     "Undeclared quantum register 'missing'");
  CheckRejectedOperandInBothDialects(
      declarations + "measure missing[0] -> c[0];\n",
      "Undeclared quantum register 'missing'");
  CheckRejectedOperandInBothDialects(
      declarations + "measure q[0] -> missing[0];\n",
      "Undeclared classical register 'missing'");
  CheckRejectedOperandInBothDialects(
      declarations + "measure missingq[0] -> missingc[0];\n",
      "Undeclared quantum register 'missingq'");
  CheckRejectedOperandInBothDialects(declarations + "reset missing[0];\n",
                                     "Undeclared quantum register 'missing'");
  CheckRejectedOperandInBothDialects(declarations + "barrier missing[0];\n",
                                     "Undeclared quantum register 'missing'");

  CheckRejectedWithMessage(
      "OPENQASM 2.0;\n"
      "qreg q[1];\n"
      "if (missing == 1) x q[0];\n",
      "Undeclared classical register 'missing'");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "if (missing[0]) { x q[0]; }\n",
      "Undeclared classical register 'missing'");
}

BOOST_AUTO_TEST_CASE(IndexedOperandsMustBeWithinTheirDeclaredRegisters) {
  const std::string declarations = "qreg q[1];\ncreg c[1];\n";

  CheckRejectedOperandInBothDialects(
      declarations + "x q[-1];\n",
      "Quantum register 'q' index -1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "x q[1];\n",
      "Quantum register 'q' index 1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "measure q[1] -> c[0];\n",
      "Quantum register 'q' index 1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "measure q[0] -> c[1];\n",
      "Classical register 'c' index 1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "measure q[-1] -> c[0];\n",
      "Quantum register 'q' index -1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "measure q[0] -> c[-1];\n",
      "Classical register 'c' index -1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "reset q[1];\n",
      "Quantum register 'q' index 1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "reset q[-1];\n",
      "Quantum register 'q' index -1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "barrier q[1];\n",
      "Quantum register 'q' index 1 is out of range");
  CheckRejectedOperandInBothDialects(
      declarations + "barrier q[-1];\n",
      "Quantum register 'q' index -1 is out of range");

  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "bit c;\n"
      "if (c[1]) { x q[0]; }\n",
      "Classical register 'c' index 1 is out of range");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "bit c;\n"
      "if (c[-1]) { x q[0]; }\n",
      "Classical register 'c' index -1 is out of range");
}

BOOST_AUTO_TEST_CASE(RegisterConditionValuesMustFitTheirRegisters) {
  for (const std::string &header : {kQasm2Header, kQasm3Header}) {
    const std::string declarations = header + "qreg q[1];\ncreg c[2];\n";
    const std::string suffix =
        header == kQasm3Header ? " { x q[0]; }\n" : " x q[0];\n";

    CheckRejectedWithMessage(declarations + "if (c == -1)" + suffix,
                             "non-negative");
    CheckRejectedWithMessage(declarations + "if (c == 4)" + suffix,
                             "does not fit");
    CheckAccepted(header, "qreg q[1];\ncreg c[2];\nif (c == 3)" + suffix);
  }
}

BOOST_AUTO_TEST_CASE(ModifiedIdentityStillValidatesTheCompleteCall) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "inv @ id missing[0];\n",
      "Undeclared quantum register 'missing'");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "ctrl @ id q[99], q[0];\n",
      "Quantum register 'q' index 99 is out of range");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[4] q;\n"
      "ctrl(3) @ id q[0], q[1], q[2], q[3];\n",
      "more than two controls");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "inv @ id q[0], q[0];\n",
      "requires exactly 1 qubits");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "ctrl @ id q[0];\n",
      "requires exactly 2 qubits");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "inv @ id(0.5) q[0];\n",
      "does not allow one parameter");
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[2] controls;\n"
      "qubit[3] targets;\n"
      "ctrl @ id controls, targets;\n",
      "Mismatched qubits sizes");
}

BOOST_AUTO_TEST_CASE(WholeRegisterOperandWidthsMustMatch) {
  CheckRejectedOperandInBothDialects(
      "qreg q[2];\n"
      "creg c[1];\n"
      "measure q -> c;\n",
      "Measurement operands must have the same width");

  CheckRejectedOperandInBothDialects(
      "qreg controls[2];\n"
      "qreg targets[3];\n"
      "cx controls, targets;\n",
      "Mismatched qubits sizes for CX gate arguments");

  CheckRejectedOperandInBothDialects(
      "qreg left[2];\n"
      "qreg right[3];\n"
      "swap left, right;\n",
      "Mismatched qubits sizes for gate arguments");
}

BOOST_AUTO_TEST_CASE(ValidIndexedAndBroadcastOperandsRemainAccepted) {
  for (const std::string &header : {kQasm2Header, kQasm3Header})
    CheckAccepted(header,
                  "qreg q[2];\n"
                  "qreg r[2];\n"
                  "creg c[2];\n"
                  "h q;\n"
                  "cx q, r;\n"
                  "barrier q[0], r;\n"
                  "reset r[1];\n"
                  "measure q -> c;\n");
}

BOOST_AUTO_TEST_CASE(OnlySupportedExplicitMajorVersionsAreAccepted) {
  CheckAccepted("OPENQASM 2.7;\n", "qreg q[1];\nx q[0];\n");
  CheckAccepted("OPENQASM 3.1;\n", "qubit q;\nx q[0];\n");
  CheckAccepted("", "qreg q[1];\nx q[0];\n");

  CheckRejectedWithMessage("OPENQASM 1.0;\nqreg q[1];\n",
                           "Unsupported OpenQASM major version 1");
  CheckRejectedWithMessage("OPENQASM 4.0;\nqubit q;\n",
                           "Unsupported OpenQASM major version 4");
}

BOOST_AUTO_TEST_CASE(MinorVersionIsOptionalOnlyInQasm3) {
  CheckAccepted("OPENQASM 3;\n", "qubit[2] q;\nctrl @ x q[0], q[1];\n");
  CheckAccepted("OPENQASM 3;\n", "qubit q;\nbit c;\nc = measure q;\n");

  CheckRejectedWithMessage("OPENQASM 2;\nqreg q[1];\n",
                           "OpenQASM 2 requires an explicit minor version");
  CheckRejectedWithMessage("OPENQASM 1;\nqreg q[1];\n",
                           "Unsupported OpenQASM major version 1");
  CheckRejectedWithMessage("OPENQASM 3 . 0;\nqubit q;\n", "Unparsed input");
}

BOOST_AUTO_TEST_CASE(BlockCommentsAreWhitespaceInBothDialects) {
  CheckAccepted("/* before the header */\nOPENQASM 2.0;\n",
                "qreg /* between tokens */ q[1] /* before semicolon */ ;\n"
                "x /* before operand */ q[0];\n"
                "/* after statement */\n");
  CheckAccepted("/* before the header */\nOPENQASM 3.0;\n",
                "qubit /* before size */ [1] q;\n"
                "rx(/* inside arguments */ 0.5) q[0];\n"
                "/* after statement */\n");
  CheckAccepted("OPENQASM 3.0;\n", "/**/\nqubit q;\nx q[0];\n");
}

BOOST_AUTO_TEST_CASE(LineCommentAtEndOfFileIsAcceptedInBothDialects) {
  CheckAccepted("OPENQASM 2.0;\n", "qreg q[1];\nx q[0];\n// eof");
  CheckAccepted("OPENQASM 3.0;\n", "qubit q;\nx q[0];\n// eof");
}

BOOST_AUTO_TEST_CASE(UnterminatedBlockCommentIsRejected) {
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit q;\n"
      "/* never closed\n"
      "x q[0];\n",
      "/* never closed");
}

BOOST_AUTO_TEST_CASE(RecognizedPreludeMarkersWorkUnderEitherDialect) {
  const std::vector<std::string> headers = {kQasm2Header, kQasm3Header};
  const std::vector<std::string> preludes = {"qelib1.inc", "stdgates.inc"};

  for (const auto &header : headers) {
    for (const auto &prelude : preludes) {
      const std::string qasmStr =
          header + "include \"" + prelude + "\";\nqreg q[1];\nx q[0];\n";
      qasm::QasmToCirc<> parser;
      parser.ParseAndTranslate(qasmStr);
      BOOST_TEST(!parser.Failed(), parser.GetErrorMessage());
      BOOST_TEST(parser.GetIncludes().size() == 1U);
      if (!parser.GetIncludes().empty())
        BOOST_TEST(parser.GetIncludes().front() == prelude);
    }
  }
}

BOOST_AUTO_TEST_CASE(CustomPreludeIncludesAreRejectedExplicitly) {
  for (const auto &header : {kQasm2Header, kQasm3Header})
    CheckRejectedWithMessage(header +
                                 "include \"custom-gates.inc\";\n"
                                 "qreg q[1];\n",
                             "Unsupported OpenQASM include 'custom-gates.inc'");
}

BOOST_AUTO_TEST_CASE(MidProgramIncludesRemainUnsupported) {
  for (const auto &header : {kQasm2Header, kQasm3Header})
    CheckRejectedWithMessage(header +
                                 "qreg q[1];\n"
                                 "include \"qelib1.inc\";\n",
                             "include");
}

BOOST_AUTO_TEST_SUITE_END()
