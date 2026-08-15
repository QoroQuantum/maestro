#include "qasm_test_utils.h"

BOOST_AUTO_TEST_SUITE(qasm_gate_contract_tests)

using qasm_test::CheckDifferentProbabilities;
using qasm_test::CheckRejectedWithMessage;
using qasm_test::CheckSameProbabilities;
using qasm_test::NthGateType;

// Every accepted gate name must resolve to the gate it names.
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
// must.
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

BOOST_AUTO_TEST_CASE(GphaseRejectsAQubitArgument) {
  // kNone still has arity zero, which is what gphase relies on.
  CheckRejectedWithMessage(
      "OPENQASM 3.0;\n"
      "qubit[1] q;\n"
      "gphase(0.5) q[0];\n",
      "requires exactly 0 qubits");
}

BOOST_AUTO_TEST_CASE(UnknownGateNameIsStillRejected) {
  // The regression guard on the "id" fix: exempting "id" alone must not turn
  // an unrecognised name into a silent no-op. A typo has to be loud.
  for (const auto &versions : {std::make_pair(std::string("OPENQASM 2.0;\n"),
                                              std::string("qreg q[2];\n")),
                               std::make_pair(std::string("OPENQASM 3.0;\n"),
                                              std::string("qubit[2] q;\n"))}) {
    for (const std::string &call :
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

BOOST_AUTO_TEST_SUITE_END()
