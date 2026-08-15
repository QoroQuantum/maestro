/**
 * @file SyntaxTree.h
 * @ingroup qasm
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Classes for the qasm parser and interpreter.
 *
 * It's supposed to support only open qasm 2.0.
 */

#pragma once

#ifndef _SYNTAXTREE_H_
#define _SYNTAXTREE_H_

#include "../Circuit/Factory.h"
#include "SimpleOps.h"

namespace qasm {
struct AddGateExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const UopType &uop,
      const std::unordered_map<std::string, IndexedId> &qreg_map,
      const std::unordered_map<std::string, StatementType> &opaqueGates,
      const std::unordered_map<std::string, StatementType> &definedGates,
      const std::unordered_map<std::string, double> &variables = {}) const {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::Uop;

    if (std::holds_alternative<UGateCallType>(uop)) {
      const auto &gate = std::get<UGateCallType>(uop);
      const std::vector<Expression> &params = boost::fusion::at_c<0>(gate);
      const ArgumentType &arg = boost::fusion::at_c<1>(gate);

      stmt.gateType = Circuits::QuantumGateType::kUGateType;

      for (const auto &p : params) stmt.parameters.push_back(p.Eval(variables));

      stmt.qubits = ParseQubits(arg, qreg_map);
    } else if (std::holds_alternative<CXGateCallType>(uop)) {
      const auto &gate = std::get<CXGateCallType>(uop);

      stmt.gateType = Circuits::QuantumGateType::kCXGateType;

      // two arguments
      const ArgumentType &arg1 = boost::fusion::at_c<0>(gate);
      const ArgumentType &arg2 = boost::fusion::at_c<1>(gate);

      const std::vector<int> qubits1 = ParseQubits(arg1, qreg_map);
      const std::vector<int> qubits2 = ParseQubits(arg2, qreg_map);

      // four cases
      if (qubits1.size() == 1 && qubits2.size() == 1) {
        // cx q1[i], q2[j]
        stmt.qubits.push_back(qubits1[0]);
        stmt.qubits.push_back(qubits2[0]);
      } else if (qubits1.size() == 1 && qubits2.size() > 1) {
        // cx q1[i], q2
        for (const auto &q2 : qubits2) {
          if (q2 == qubits1[0])
            throw std::invalid_argument(
                "Control and target qubits cannot be the same for CX gate");
          stmt.qubits.push_back(qubits1[0]);
          stmt.qubits.push_back(q2);
        }
      } else if (qubits1.size() > 1 && qubits2.size() == 1) {
        // cx q[], q[j]
        for (const auto &q1 : qubits1) {
          if (q1 == qubits2[0])
            throw std::invalid_argument(
                "Control and target qubits cannot be the same for CX gate");
          stmt.qubits.push_back(q1);
          stmt.qubits.push_back(qubits2[0]);
        }
      } else if (qubits1.size() == qubits2.size()) {
        // cx q1, q2
        for (size_t i = 0; i < qubits1.size(); ++i) {
          if (qubits1[i] == qubits2[i])
            throw std::invalid_argument(
                "Control and target qubits cannot be the same for CX gate");
          stmt.qubits.push_back(qubits1[i]);
          stmt.qubits.push_back(qubits2[i]);
        }
      } else
        throw std::invalid_argument(
            "Mismatched qubits sizes for CX gate arguments");
    } else if (std::holds_alternative<GatecallType>(uop)) {
      const auto &gateCall = std::get<GatecallType>(uop);
      // can be either simple or with expressions
      if (std::holds_alternative<SimpleGatecallType>(gateCall)) {
        // gate without parameters
        const auto &gate = std::get<SimpleGatecallType>(gateCall);
        const std::string &gateName = boost::fusion::at_c<0>(gate);

        const MixedListType &args = boost::fusion::at_c<1>(gate);

        // Both lookups below - the user-defined one and the builtin one - are
        // case-sensitive, because OpenQASM is a case-sensitive language: `sdg`
        // is a gate and `SDG` is not. They used to disagree: `definedGates`
        // was keyed and searched by the name as written while the builtin
        // tables were consulted with it lowercased, so every uppercase
        // spelling of a builtin was silently accepted as that builtin -
        // `SDG q[0];`, `RX(0.5) q[0];`, `CCX ...`, none of which is a gate in
        // either dialect - and no user-declared gate could ever own such a
        // name. Agreeing on the exact spelling removes the collision class
        // entirely: `x` is the builtin, `X` is whatever the program declared,
        // and the two coexist. (QASM2's genuinely uppercase `U` and `CX` are
        // matched by their own rules, `ugateCall`/`cxgateCall`, which accept
        // both cases; they reach here only via an over-long argument list,
        // and the allowed-gate sets below list both spellings for that.)
        const auto it = definedGates.find(gateName);
        if (it == definedGates.end()) {
          if (!IsSuppportedGate(gateName))
            throw std::invalid_argument(
                "Unsupported gate without parameters: " + gateName);

          // A supported gate that is not one of the parameterless ones has
          // arrived here with no parameter list at all - `rx q[0];` or, now
          // that the spec-legal empty parameter list parses, `rx() q[0];`.
          // Falling through would build the gate with its angle defaulted to
          // zero, i.e. silently emit a different rotation than the program
          // asked for; the missing parameters are an error instead.
          if (!IsSuppportedNoParamGate(gateName))
            throw std::invalid_argument("Gate " + gateName +
                                        " requires parameters");

          stmt.gateType = GetGateType(gateName);

          // "id" is the identity gate: standard in both qelib1.inc and
          // stdgates.inc, emitted by Qiskit, and contributing no operation.
          // It has no gate type of its own, so it lands on kNone - whose
          // GateNrQubits is 0, the arity the qubit-less "gphase" needs - and
          // the generic arity below would reject its qubit argument. Its own
          // arity is therefore spelled out by name: "id" is a one-qubit gate
          // and takes exactly one argument (a whole-register broadcast, `id
          // q;`, is one argument too), like any other one-qubit gate. Only
          // "id" gets a name-specific arity, and the check itself is never
          // skipped: skipping it let `id q[0], q[1];` and even `id q[0],
          // q[0];` through silently, and letting kNone accept any arity would
          // make every unmapped name a silent no-op - the very failure this
          // guards against. The arguments are still resolved and
          // broadcast-checked below; the resulting statement has no gate type
          // and no gate declaration, which is what Program::AddToCircuit
          // skips.
          const int expectedNrQubits =
              gateName == "id" ? 1 : GateNrQubits(stmt.gateType);
          if (static_cast<int>(args.size()) != expectedNrQubits)
            throw std::invalid_argument(
                "Gate " + gateName + " requires exactly " +
                std::to_string(expectedNrQubits) + " qubits");
        } else {
          // defined gate, check if the number of qubits match
          const StatementType &definedGateStmt = it->second;
          const int expectedNrQubits =
              static_cast<int>(definedGateStmt.qubitsDecl.size());
          if (static_cast<int>(args.size()) != expectedNrQubits)
            throw std::invalid_argument(
                "Defined gate " + gateName + " requires exactly " +
                std::to_string(expectedNrQubits) + " qubits");
          if (definedGateStmt.parameters.size() != 0)
            throw std::invalid_argument(
                "Defined gate " + gateName +
                " requires parameters, but none were provided");

          stmt.comment = gateName;

          // copy the gate definition here, as it might be redefined later
          // (actually not yet supported)
          stmt.qubitsDecl = definedGateStmt.qubitsDecl;
          stmt.declOps = definedGateStmt.declOps;
        }

        // first, check the qubits
        int qubitsCounter = 1;

        std::vector<std::vector<int>> allQubits(args.size());
        for (int i = 0; i < static_cast<int>(args.size()); ++i) {
          const ArgumentType &arg = args[i];
          std::vector<int> qubits = ParseQubits(arg, qreg_map);

          if (qubits.size() != 1) {
            if (qubitsCounter == 1)
              qubitsCounter = static_cast<int>(qubits.size());
            else if (qubitsCounter != static_cast<int>(qubits.size()))
              throw std::invalid_argument(
                  "Mismatched qubits sizes for gate arguments");
          }

          allQubits[i] = std::move(qubits);
        }

        // now set them
        for (int i = 0; i < qubitsCounter; ++i) {
          for (const auto &qlist : allQubits) {
            if (qlist.size() == 1)
              stmt.qubits.push_back(qlist[0]);
            else
              stmt.qubits.push_back(qlist[i]);
          }
        }
      } else if (std::holds_alternative<ExpGatecallType>(gateCall)) {
        // gate with parameters
        const auto &gate = std::get<ExpGatecallType>(gateCall);
        const std::string &gateName = boost::fusion::at_c<0>(gate);

        const std::vector<Expression> &params = boost::fusion::at_c<1>(gate);
        const MixedListType &args = boost::fusion::at_c<2>(gate);

        for (const auto &p : params)
          stmt.parameters.push_back(p.Eval(variables));

        // Case-sensitive, exactly as in the parameterless branch above: `rx`
        // is the builtin rotation and `RX` is not a gate at all unless the
        // program declared one by that name.
        const auto it = definedGates.find(gateName);
        if (it == definedGates.end()) {
          if (IsSuppportedGate(gateName)) {
            if (params.empty())
              throw std::invalid_argument("Gate " + gateName +
                                          " requires parameters");
            else if (params.size() == 1) {
              if (!IsSuppportedOneParamGate(gateName))
                throw std::invalid_argument(
                    "This gate does not allow one parameter: " + gateName);
            } else if (params.size() > 1) {
              if (!IsSuppportedMultipleParamsGate(gateName))
                throw std::invalid_argument(
                    "This gate does not allow multiple parameters: " +
                    gateName);

              if (params.size() > 4)
                throw std::invalid_argument("Too many parameters for gate: " +
                                            gateName);
            }

            stmt.gateType = GetGateType(gateName);
            const int expectedNrQubits = GateNrQubits(stmt.gateType);
            if (static_cast<int>(args.size()) != expectedNrQubits)
              throw std::invalid_argument(
                  "Gate " + gateName + " requires exactly " +
                  std::to_string(expectedNrQubits) + " qubits");

            if (gateName == "u1") {
              const double lambda = stmt.parameters[0];
              stmt.parameters[0] = 0.0;  // theta
              stmt.parameters.push_back(0.);
              stmt.parameters.push_back(lambda);
            } else if (gateName == "u2") {
              const double phi = stmt.parameters[0];
              const double lambda = stmt.parameters[1];

              stmt.parameters[0] = M_PI / 2.0;  // theta
              stmt.parameters[1] = phi;
              stmt.parameters.push_back(lambda);
            } else if (gateName == "cu1") {
              const double lambda = stmt.parameters[0];
              stmt.parameters[0] = 0.0;  // theta
              stmt.parameters.push_back(0.);
              stmt.parameters.push_back(lambda);
            } else if (gateName == "cu2") {
              const double phi = stmt.parameters[0];
              const double lambda = stmt.parameters[1];

              stmt.parameters[0] = M_PI / 2.0;  // theta
              stmt.parameters[1] = phi;
              stmt.parameters.push_back(lambda);
            }
          } else if (gateName == "u2") {
            // Only names absent from the allowed-gate sets reach this chain:
            // "u2" and "cu2", qelib1.inc gates with no gate type of their own.
            // "u1", "u3", "cu1" and "cu3" are all listed in those sets, so the
            // branch above handles them and arms for them here were dead.
            stmt.gateType = Circuits::QuantumGateType::kUGateType;

            if (stmt.parameters.size() != 2)
              throw std::invalid_argument("Gate u2 requires two parameters");

            const double phi = stmt.parameters[0];
            const double lambda = stmt.parameters[1];

            stmt.parameters[0] = M_PI / 2.0;  // theta
            stmt.parameters[1] = phi;
            stmt.parameters.push_back(lambda);

            if (args.size() != 1)
              throw std::invalid_argument("Gate " + gateName +
                                          " requires exactly 1 qubit");
          } else if (gateName == "cu2") {
            stmt.gateType = Circuits::QuantumGateType::kCUGateType;

            if (stmt.parameters.size() != 2)
              throw std::invalid_argument("Gate cu2 requires two parameters");

            const double phi = stmt.parameters[0];
            const double lambda = stmt.parameters[1];

            stmt.parameters[0] = M_PI / 2.0;  // theta
            stmt.parameters[1] = phi;
            stmt.parameters.push_back(lambda);

            if (args.size() != 2)
              throw std::invalid_argument("Gate " + gateName +
                                          " requires exactly 2 qubits");
          } else {
            throw std::invalid_argument("Unsupported gate with parameters: " +
                                        gateName);
          }
        } else {
          // defined gate, check if the number of parameters and qubits match
          const StatementType &definedGateStmt = it->second;
          const int expectedNrQubits =
              static_cast<int>(definedGateStmt.qubitsDecl.size());
          if (static_cast<int>(args.size()) != expectedNrQubits)
            throw std::invalid_argument(
                "Defined gate " + gateName + " requires exactly " +
                std::to_string(expectedNrQubits) + " qubits");

          if (definedGateStmt.paramsDecl.size() != stmt.parameters.size())
            throw std::invalid_argument(
                "Defined gate " + gateName + " requires a different number (" +
                std::to_string(definedGateStmt.paramsDecl.size()) +
                ") of parameters than " +
                std::to_string(stmt.parameters.size()));

          stmt.comment = gateName;

          // copy the gate definition here, as it might be redefined later
          stmt.paramsDecl = definedGateStmt.paramsDecl;
          stmt.qubitsDecl = definedGateStmt.qubitsDecl;
          stmt.declOps = definedGateStmt.declOps;
        }

        // first, check the qubits
        int qubitsCounter = 1;

        std::vector<std::vector<int>> allQubits(args.size());
        for (int i = 0; i < static_cast<int>(args.size()); ++i) {
          const ArgumentType &arg = args[i];
          std::vector<int> qubits = ParseQubits(arg, qreg_map);

          if (qubits.size() != 1) {
            if (qubitsCounter == 1)
              qubitsCounter = static_cast<int>(qubits.size());
            else if (qubitsCounter != static_cast<int>(qubits.size()))
              throw std::invalid_argument(
                  "Mismatched qubits sizes for gate arguments");
          }

          allQubits[i] = std::move(qubits);
        }

        // now set them
        for (int i = 0; i < qubitsCounter; ++i) {
          for (const auto &qlist : allQubits) {
            if (qlist.size() == 1)
              stmt.qubits.push_back(qlist[0]);
            else
              stmt.qubits.push_back(qlist[i]);
          }
        }
      }
    }

    return stmt;
  }

  static std::vector<int> ParseQubits(
      const ArgumentType &arg,
      const std::unordered_map<std::string, IndexedId> &qreg_map) {
    std::vector<int> qubits;
    // there are two possibilities here, either it's an indexed id or a simple
    // id
    if (std::holds_alternative<IndexedId>(arg)) {
      const IndexedId &indexedId = std::get<IndexedId>(arg);
      auto it = qreg_map.find(indexedId.id);
      if (it != qreg_map.end()) {
        int base = it->second.base;
        qubits.push_back(base + indexedId.index);
      }
    } else if (std::holds_alternative<std::string>(arg)) {
      const std::string &id = std::get<std::string>(arg);
      auto it = qreg_map.find(id);
      if (it != qreg_map.end()) {
        int base = it->second.base;
        int size = static_cast<int>(std::round(it->second.Eval()));
        for (int i = 0; i < size; ++i) qubits.push_back(base + i);
      }
    }

    return qubits;
  }

  static bool IsSuppportedNoParamGate(const std::string &gateName) {
    return allowedNoParamGates.find(gateName) != allowedNoParamGates.end();
  }

  static bool IsSuppportedOneParamGate(const std::string &gateName) {
    return allowedOneParamGates.find(gateName) != allowedOneParamGates.end();
  }

  static bool IsSuppportedMultipleParamsGate(const std::string &gateName) {
    return allowedMultipleParamsGates.find(gateName) !=
           allowedMultipleParamsGates.end();
  }

  static bool IsSuppportedGate(const std::string &gateName) {
    return allowedNoParamGates.find(gateName) != allowedNoParamGates.end() ||
           allowedOneParamGates.find(gateName) != allowedOneParamGates.end() ||
           allowedMultipleParamsGates.find(gateName) !=
               allowedMultipleParamsGates.end();
  }

  // Case-sensitive, like the allowed-gate sets it is paired with: it maps the
  // canonical lowercase spelling of a builtin and nothing else. It used to
  // lowercase its argument, which was one half of the mismatch that let an
  // uppercase call reach a builtin while user-defined gates were looked up as
  // written - see the note at the SimpleGatecallType branch above.
  Circuits::QuantumGateType GetGateType(const std::string &gateName) const {
    // "U" and "CX" are the two QASM2 builtins that are legitimately spelled
    // in uppercase, and both spellings of each are accepted - see the note on
    // the allowed-gate sets below. They are the only uppercase names here.
    if (gateName == "x")
      return Circuits::QuantumGateType::kXGateType;
    else if (gateName == "y")
      return Circuits::QuantumGateType::kYGateType;
    else if (gateName == "z")
      return Circuits::QuantumGateType::kZGateType;
    else if (gateName == "h")
      return Circuits::QuantumGateType::kHadamardGateType;
    else if (gateName == "s")
      return Circuits::QuantumGateType::kSGateType;
    else if (gateName == "sdg" || gateName == "sdag")
      return Circuits::QuantumGateType::kSdgGateType;
    else if (gateName == "t")
      return Circuits::QuantumGateType::kTGateType;
    else if (gateName == "tdg" || gateName == "tdag")
      return Circuits::QuantumGateType::kTdgGateType;
    else if (gateName == "sx")
      return Circuits::QuantumGateType::kSxGateType;
    else if (gateName == "sxdg" || gateName == "sxdag")
      return Circuits::QuantumGateType::kSxDagGateType;
    else if (gateName == "k")
      return Circuits::QuantumGateType::kKGateType;
    else if (gateName == "swap")
      return Circuits::QuantumGateType::kSwapGateType;
    else if (gateName == "cx" || gateName == "CX")
      return Circuits::QuantumGateType::kCXGateType;
    else if (gateName == "cy")
      return Circuits::QuantumGateType::kCYGateType;
    else if (gateName == "cz")
      return Circuits::QuantumGateType::kCZGateType;
    else if (gateName == "ch")
      return Circuits::QuantumGateType::kCHGateType;
    else if (gateName == "csx")
      return Circuits::QuantumGateType::kCSxGateType;
    else if (gateName == "csxdg" || gateName == "csxdag")
      return Circuits::QuantumGateType::kCSxDagGateType;
    else if (gateName == "cswap")
      return Circuits::QuantumGateType::kCSwapGateType;
    else if (gateName == "ccx")
      return Circuits::QuantumGateType::kCCXGateType;
    else if (gateName == "p" || gateName == "phase")
      return Circuits::QuantumGateType::kPhaseGateType;
    else if (gateName == "rx")
      return Circuits::QuantumGateType::kRxGateType;
    else if (gateName == "ry")
      return Circuits::QuantumGateType::kRyGateType;
    else if (gateName == "rz")
      return Circuits::QuantumGateType::kRzGateType;
    else if (gateName == "cp" || gateName == "cphase")
      return Circuits::QuantumGateType::kCPGateType;
    else if (gateName == "crx")
      return Circuits::QuantumGateType::kCRxGateType;
    else if (gateName == "cry")
      return Circuits::QuantumGateType::kCRyGateType;
    else if (gateName == "crz")
      return Circuits::QuantumGateType::kCRzGateType;
    else if (gateName == "U" || gateName == "u" || gateName == "u3" ||
             gateName == "u1")
      return Circuits::QuantumGateType::kUGateType;
    else if (gateName == "cu" || gateName == "cu3" || gateName == "cu1")
      return Circuits::QuantumGateType::kCUGateType;

    // Returned for an unmapped name, and for the two accepted names that
    // legitimately have no gate of their own: 'id' and 'gphase', both of
    // which are dropped rather than emitted. The callers pair this with an
    // arity check, so an unmapped name is still rejected rather than being
    // silently treated as a no-op.
    return Circuits::QuantumGateType::kNone;
  }

  static int GateNrQubits(Circuits::QuantumGateType gateType) {
    const int gateT = static_cast<int>(gateType);

    if (gateT < static_cast<int>(Circuits::QuantumGateType::kSwapGateType))
      return 1;
    else if (gateT <
             static_cast<int>(Circuits::QuantumGateType::kCSwapGateType))
      return 2;
    else if (gateT <= static_cast<int>(Circuits::QuantumGateType::kCCXGateType))
      return 3;

    return 0;
  }

  // The accepted gate names, exposed so a regression test can walk every one
  // of them and check it actually resolves. Both of the gate-name bugs this
  // guards against were a name listed here with no matching arm in
  // GetGateType, which no test could reach without the lists themselves.
  static const std::unordered_set<std::string> &AllowedNoParamGates() {
    return allowedNoParamGates;
  }

  static const std::unordered_set<std::string> &AllowedOneParamGates() {
    return allowedOneParamGates;
  }

  static const std::unordered_set<std::string> &AllowedMultipleParamsGates() {
    return allowedMultipleParamsGates;
  }

 private:
  // Gate names are matched exactly - OpenQASM is case-sensitive, so "sdg" is
  // a gate and "SDG" is not - with exactly two exceptions: "CX" and "U", the
  // QASM2 builtins the language itself spells in uppercase. Their own call
  // rules (`cxgateCall`/`ugateCall` in qasm.h) already accept either case, so
  // both spellings are listed here too. That matters only for the calls that
  // fall out of those rules - an over-long argument list, which `gatecall`
  // picks up - and it is what keeps `U(...) q[0], q[1];` reported as the
  // qubit-count error it is instead of as an unknown gate.
  static inline std::unordered_set<std::string> allowedNoParamGates = {
      "x",    "y",  "z",    "h",     "s",      "sdg",   "sdag", "t",  "tdg",
      "tdag", "sx", "sxdg", "sxdag", "k",      "swap",  "cx",   "CX", "cy",
      "cz",   "ch", "csx",  "csxdg", "csxdag", "cswap", "ccx",  "id"};
  static inline std::unordered_set<std::string> allowedOneParamGates = {
      "p",   "rx", "ry",  "rz",    "cp",     "crx",   "cry",
      "crz", "u1", "cu1", "phase", "cphase", "gphase"};
  // "cu1" is deliberately absent: like "u1" it takes exactly one angle, and
  // listing it here made cu1(a, b, c) accepted and silently lowered as cu3.
  static inline std::unordered_set<std::string> allowedMultipleParamsGates = {
      "u", "U", "u3", "cu", "cu3"};  // max 4
};

inline phx::function<AddGateExpr> AddGate;

// Lowers a QASM3 modified gate call by rewriting it into the equivalent
// unmodified call and delegating to AddGateExpr. It wraps AddGateExpr rather
// than modifying it, because AddGateExpr is also the macro inliner used by
// Program::AddToCircuit for user-defined gates; with an empty modifier list
// this delegates straight through, so an unmodified call behaves exactly as
// it did before.
struct AddModifiedGateExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const ModifiedUopType &modifiedUop,
      const std::unordered_map<std::string, IndexedId> &qreg_map,
      const std::unordered_map<std::string, StatementType> &opaqueGates,
      const std::unordered_map<std::string, StatementType> &definedGates,
      const std::unordered_map<std::string, double> &variables = {}) const {
    const ModifierListType &modifiers = boost::fusion::at_c<0>(modifiedUop);
    const UopType &uop = boost::fusion::at_c<1>(modifiedUop);

    AddGateExpr addGate;
    if (modifiers.empty())
      return addGate(uop, qreg_map, opaqueGates, definedGates, variables);

    std::string gateName;
    std::vector<double> params;
    MixedListType args;
    Canonicalize(uop, gateName, params, args, variables);

    if (definedGates.find(gateName) != definedGates.end())
      throw std::invalid_argument(
          "Gate modifiers cannot be applied to the user-defined gate: " +
          gateName);

    // Only aliases are folded here, not case: gate names are matched exactly,
    // so `inv @ SDG` is no more a call to sdg than the unmodified `SDG q[0];`
    // is. Lowercasing here would have reintroduced, on the modified path
    // alone, the very silent uppercase-reaches-a-builtin behaviour the plain
    // call path no longer has. Runs after the user-defined lookup above,
    // which must see the name exactly as written.
    std::string canonicalName = gateName;
    NormalizeGateName(canonicalName);

    // The identity gate stays the identity whatever is applied to it, so it
    // is dropped before the tables below, which do not list it.
    if (canonicalName == "id") return NoOpStatement();

    // `ctrl(n) @` contributes n controls, not one, so the count is summed
    // rather than the modifiers counted - this is what makes `ctrl(3) @ x`
    // report the multi-control limit below instead of being lowered as a
    // single control.
    size_t nrControls = 0;
    for (const auto &modifier : modifiers)
      if (IsControl(modifier))
        nrControls += static_cast<size_t>(modifier.count);

    if (nrControls > 2)
      throw std::invalid_argument(
          "ctrl @ with more than two controls is not supported, " +
          std::to_string(nrControls) +
          " were requested for the gate: " + canonicalName);

    // The number of times the rewritten gate is emitted; pow(k) on a gate
    // whose angle cannot be scaled turns into repetition instead.
    int repetitions = 1;

    // Modifiers apply innermost first, i.e. right to left in source order.
    for (auto it = modifiers.rbegin(); it != modifiers.rend(); ++it) {
      switch (it->kind) {
        case ModifierKind::Inv:
          ApplyInv(canonicalName, params);
          break;
        case ModifierKind::Pow:
          ApplyPow(it->exponent, canonicalName, params, repetitions);
          break;
        case ModifierKind::Ctrl:
        case ModifierKind::NegCtrl:
          // negctrl is lowered as a positive control here and conjugated with
          // x gates on the negated control below. `ctrl(n) @` is n nested
          // single controls, e.g. ctrl(2) @ x is ctrl @ ctrl @ x, i.e. ccx.
          for (int c = 0; c < it->count; ++c) ApplyCtrl(canonicalName, params);
          break;
      }
    }

    // The rewritten call is validated even when it is ultimately dropped or
    // repeated, so pow(0) @ h q[0], q[1] is still a qubit-count error.
    StatementType stmt = addGate(MakeCall(canonicalName, params, args),
                                 qreg_map, opaqueGates, definedGates);

    if (repetitions == 0) return NoOpStatement();

    // Program::AddToCircuit emits one gate per group of qubits, so repeating
    // the resolved qubit list repeats the gate - including under
    // broadcasting, where the whole broadcast round is repeated.
    const std::vector<int> singleRound = stmt.qubits;
    for (int r = 1; r < repetitions; ++r)
      stmt.qubits.insert(stmt.qubits.end(), singleRound.begin(),
                         singleRound.end());

    const std::vector<size_t> negatedControls = NegatedControls(modifiers);
    if (negatedControls.empty()) return stmt;

    return ConjugateNegatedControls(stmt, canonicalName, params,
                                    negatedControls);
  }

  // The qubit positions a negctrl modifier controls. Control modifiers
  // consume qubit arguments left to right in source order - one each for the
  // countless spelling, `count` of them for `negctrl(n) @`, all of which are
  // negated - so a control modifier owns as many consecutive qubit arguments
  // as its count. inv and pow, which consume none, are skipped.
  static std::vector<size_t> NegatedControls(
      const ModifierListType &modifiers) {
    std::vector<size_t> negatedControls;
    size_t controlPos = 0;

    for (const auto &modifier : modifiers) {
      if (!IsControl(modifier)) continue;
      for (int c = 0; c < modifier.count; ++c, ++controlPos)
        if (modifier.kind == ModifierKind::NegCtrl)
          negatedControls.push_back(controlPos);
    }

    return negatedControls;
  }

  // `negctrl @ g c, t` means "control on |0>", which is `x c; ctrl @ g c, t;
  // x c;`. A qop synthesizes a single statement, so the triple is emitted
  // through the defined-gate inlining path already in Program::AddToCircuit:
  // a Uop with no gate type, a qubitsDecl naming the gate's qubits and a
  // declOps list is expanded op by op, with the resolved qubits substituted
  // for the declared names. Parameters are baked into declOps as constants,
  // so the synthetic declaration needs none.
  static QoperationStatement ConjugateNegatedControls(
      const QoperationStatement &gate, const std::string &gateName,
      const std::vector<double> &params,
      const std::vector<size_t> &negatedControls) {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::Uop;
    stmt.gateType = Circuits::QuantumGateType::kNone;
    stmt.qubits = gate.qubits;

    const int nrQubits = AddGateExpr::GateNrQubits(gate.gateType);
    for (int q = 0; q < nrQubits; ++q)
      stmt.qubitsDecl.push_back("__negctrl_q" + std::to_string(q));

    MixedListType gateArgs;
    for (const auto &qubitName : stmt.qubitsDecl)
      gateArgs.push_back(ArgumentType(qubitName));

    MixedListType flipArgs;
    for (const size_t control : negatedControls)
      flipArgs.push_back(ArgumentType(stmt.qubitsDecl[control]));

    for (const auto &flipArg : flipArgs)
      stmt.declOps.push_back(MakeCall("x", {}, MixedListType{flipArg}));
    stmt.declOps.push_back(MakeCall(gateName, params, gateArgs));
    for (const auto &flipArg : flipArgs)
      stmt.declOps.push_back(MakeCall("x", {}, MixedListType{flipArg}));

    return stmt;
  }

  // Flattens any of the three uop shapes into the (name, evaluated
  // parameters, qubit arguments) triple the rewriting below works on. The
  // dedicated U and CX call rules carry no gate name of their own, so they
  // get their canonical one here.
  //
  // `variables` is the same map AddModifiedGateExpr::operator() received -
  // top-level `input` bindings, since modifiers cannot appear inside a gate
  // declaration body, so no macro formal parameter is ever in scope here.
  // Passing it through (rather than an empty map) is what lets a modified
  // call's own parameter expression, e.g. `ctrl @ rz(theta) ...`, resolve an
  // `input` the same way the unmodified path (AddGateExpr, reached when the
  // modifier list is empty) already does. Any identifier not found in it is
  // still an error, not a silent zero - see Variable::Eval.
  static void Canonicalize(
      const UopType &uop, std::string &gateName, std::vector<double> &params,
      MixedListType &args,
      const std::unordered_map<std::string, double> &variables) {
    if (std::holds_alternative<UGateCallType>(uop)) {
      const auto &gate = std::get<UGateCallType>(uop);
      gateName = "u";
      for (const auto &p : boost::fusion::at_c<0>(gate))
        params.push_back(p.Eval(variables));
      args.push_back(boost::fusion::at_c<1>(gate));
    } else if (std::holds_alternative<CXGateCallType>(uop)) {
      const auto &gate = std::get<CXGateCallType>(uop);
      gateName = "cx";
      args.push_back(boost::fusion::at_c<0>(gate));
      args.push_back(boost::fusion::at_c<1>(gate));
    } else {
      const auto &gateCall = std::get<GatecallType>(uop);
      if (std::holds_alternative<SimpleGatecallType>(gateCall)) {
        const auto &gate = std::get<SimpleGatecallType>(gateCall);
        gateName = boost::fusion::at_c<0>(gate);
        args = boost::fusion::at_c<1>(gate);
      } else {
        const auto &gate = std::get<ExpGatecallType>(gateCall);
        gateName = boost::fusion::at_c<0>(gate);
        for (const auto &p : boost::fusion::at_c<1>(gate))
          params.push_back(p.Eval(variables));
        args = boost::fusion::at_c<2>(gate);
      }
    }
  }

  // Maps the accepted spellings of a gate onto the one name the modifier
  // tables are keyed by, so that `ctrl @ phase(0.3)` behaves like
  // `ctrl @ p(0.3)`. Only aliases GetGateType maps identically are listed,
  // which keeps a modified call accepting exactly what the same unmodified
  // call accepts - including the "dag" spellings, each of which GetGateType
  // now resolves to the same gate as its "dg" twin.
  static void NormalizeGateName(std::string &gateName) {
    static const std::unordered_map<std::string, std::string> aliases = {
        {"phase", "p"},  {"cphase", "cp"},   {"u3", "u"},
        {"cu3", "cu"},   {"sdag", "sdg"},    {"sxdag", "sxdg"},
        {"tdag", "tdg"}, {"csxdag", "csxdg"}};

    const auto it = aliases.find(gateName);
    if (it != aliases.end()) gateName = it->second;
  }

  // Builds the rewritten, unmodified gate call. Parameters are re-emitted as
  // Constant expressions, so all of AddGateExpr's qubit resolution,
  // broadcasting and validation applies to it unchanged.
  static UopType MakeCall(const std::string &gateName,
                          const std::vector<double> &params,
                          const MixedListType &args) {
    if (params.empty()) return GatecallType(SimpleGatecallType(gateName, args));

    std::vector<Expression> paramExprs;
    for (const double p : params) paramExprs.push_back(Expression(Constant(p)));

    return GatecallType(ExpGatecallType(gateName, paramExprs, args));
  }

  // A statement that adds nothing to the circuit: Program::AddToCircuit skips
  // a Uop with no gate type and no gate declaration.
  static QoperationStatement NoOpStatement() {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::Uop;
    stmt.gateType = Circuits::QuantumGateType::kNone;

    return stmt;
  }

  static void ApplyInv(std::string &gateName, std::vector<double> &params) {
    static const std::unordered_set<std::string> selfInverseGates = {
        "x", "y", "z", "h", "cx", "cy", "cz", "ch", "swap", "cswap", "ccx"};
    static const std::unordered_map<std::string, std::string> daggerGates = {
        {"s", "sdg"},   {"sdg", "s"},   {"t", "tdg"},     {"tdg", "t"},
        {"sx", "sxdg"}, {"sxdg", "sx"}, {"csx", "csxdg"}, {"csxdg", "csx"}};

    if (selfInverseGates.find(gateName) != selfInverseGates.end()) return;

    const auto daggerIt = daggerGates.find(gateName);
    if (daggerIt != daggerGates.end()) {
      gateName = daggerIt->second;
      return;
    }

    if (rotationGates.find(gateName) != rotationGates.end()) {
      for (auto &p : params) p = -p;
      return;
    }

    if (gateName == "u" || gateName == "cu") {
      // The gate is u(theta, phi, lambda, gamma) = exp(i*gamma) *
      // u(theta, phi, lambda) - the fourth parameter is a global phase - so
      // the inverse is u(-theta, -lambda, -phi, -gamma): phi and lambda swap
      // as well as being negated.
      params.resize(std::max<size_t>(params.size(), 3), 0.);
      std::swap(params[1], params[2]);
      for (auto &p : params) p = -p;
      return;
    }

    throw std::invalid_argument("inv @ is not supported for the gate: " +
                                gateName);
  }

  static void ApplyPow(double exponent, std::string &gateName,
                       std::vector<double> &params, int &repetitions) {
    if (exponent < 0.) {
      // g^(-k) is (g^-1)^k.
      ApplyInv(gateName, params);
      exponent = -exponent;
    }

    if (exponent == 0.) {
      repetitions = 0;
      return;
    }

    // Scaling the angle is exact for any real exponent and produces a single
    // gate, so it is preferred over repetition even for integer exponents.
    if (rotationGates.find(gateName) != rotationGates.end()) {
      for (auto &p : params) p *= exponent;
      return;
    }

    const double rounded = std::round(exponent);
    if (std::abs(exponent - rounded) > 1e-12)
      throw std::invalid_argument(
          "pow(k) @ with a fractional exponent is not supported for the "
          "gate: " +
          gateName);

    // The gate is materialised once per repetition, so an unbounded exponent
    // would overflow the cast to int - undefined behaviour, and a silently
    // wrong repetition count - and allocate before emitting anything.
    if (rounded > static_cast<double>(kMaxRepetitions) ||
        static_cast<double>(repetitions) * rounded >
            static_cast<double>(kMaxRepetitions))
      throw std::invalid_argument(
          "pow(k) @ with an exponent above " + std::to_string(kMaxRepetitions) +
          " is not supported for the gate: " + gateName);

    repetitions *= static_cast<int>(rounded);
  }

  static void ApplyCtrl(std::string &gateName, std::vector<double> &params) {
    static const std::unordered_map<std::string, std::string> controlledGates =
        {{"x", "cx"},   {"y", "cy"},       {"z", "cz"},       {"h", "ch"},
         {"sx", "csx"}, {"sxdg", "csxdg"}, {"p", "cp"},       {"rx", "crx"},
         {"ry", "cry"}, {"rz", "crz"},     {"swap", "cswap"}, {"cx", "ccx"}};
    // A controlled diagonal-phase gate is exactly cp at the matching angle,
    // since t == p(pi/4) and s == p(pi/2). This is what keeps every lowering
    // on a gate type that already exists.
    static const std::unordered_map<std::string, double> phaseAngles = {
        {"s", M_PI / 2.},
        {"sdg", -M_PI / 2.},
        {"t", M_PI / 4.},
        {"tdg", -M_PI / 4.}};

    if (gateName == "u") {
      // cu takes the same (theta, phi, lambda[, gamma]) parameters as u; a
      // shorter u call leaves the remaining angles at zero.
      params.resize(std::max<size_t>(params.size(), 3), 0.);
      gateName = "cu";
      return;
    }

    const auto controlledIt = controlledGates.find(gateName);
    if (controlledIt != controlledGates.end()) {
      gateName = controlledIt->second;
      return;
    }

    const auto phaseIt = phaseAngles.find(gateName);
    if (phaseIt != phaseAngles.end()) {
      gateName = "cp";
      params.assign(1, phaseIt->second);
      return;
    }

    throw std::invalid_argument("ctrl @ is not supported for the gate: " +
                                gateName);
  }

  static bool IsControl(const ModifierType &modifier) {
    return modifier.kind == ModifierKind::Ctrl ||
           modifier.kind == ModifierKind::NegCtrl;
  }

 private:
  // Gates whose single parameter is an angle they are linear in, so that
  // g(theta)^s == g(s * theta) for any real s.
  static inline std::unordered_set<std::string> rotationGates = {
      "rx", "ry", "rz", "p", "cp", "crx", "cry", "crz"};

  // The largest number of copies pow(k) may expand a gate into.
  static constexpr int kMaxRepetitions = 1024;
};

inline phx::function<AddModifiedGateExpr> AddModifiedGate;

// `gatedeclop` keeps using the plain `uop`, since using `modifiedUop` there
// would change the element type of StatementType::declOps and ripple into the
// inliner. A modifier in a gate body would otherwise leave the declaration
// unparsable and be reported as leftover input, so it is caught here and
// named. Nothing else in a gate body can start with a modifier, so this
// action only ever runs on a committed parse.
struct RejectGateBodyModifierExpr : public AbstractSyntaxTree {
  struct result {
    typedef UopType type;
  };

  UopType operator()(const ModifierListType & /*modifiers*/) const {
    throw std::invalid_argument(
        "Gate modifiers (ctrl, negctrl, inv, pow) are not supported inside a "
        "gate declaration body");
  }
};

inline phx::function<RejectGateBodyModifierExpr> RejectGateBodyModifier;

// QASM3 constructs outside our supported subset (for/while loops, subroutine
// definitions, register aliases, duration/delay, box blocks, array
// declarations) are recognised by keyword at statement position - see
// `unsupportedConstruct` in qasm.h - and rejected by name instead of falling
// through to Spirit's generic "unparsed input" error or being misparsed as a
// gate call. The recognising rule already carries the per-construct message
// as its attribute (looked up from the unsupportedKeywords symbol table), so
// this functor only has to throw it.
struct RejectUnsupportedConstructExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(const std::string &message) const {
    throw std::invalid_argument(message);
  }
};

inline phx::function<RejectUnsupportedConstructExpr> RejectUnsupportedConstruct;

// 'phase', 'cphase' and 'gphase' are OpenQASM 3 stdgates.inc names: none of
// them appears in any version of qelib1.inc, including the extended one
// Qiskit ships, so none of them is a gate under an `OPENQASM 2.0;` header.
//
// The dialect is not visible where the allowed-gate sets and GetGateType are
// consulted, and threading it there would have made GetGateType something
// other than the pure name -> type function it is. So the filter sits in the
// grammar instead, at `qasm2RejectedGate` in qasm.h - the one place that does
// know the dialect - and the gate tables stay dialect-agnostic.
//
// Two outcomes rather than one, hence the bool result. Under QASM2 these are
// ordinary identifiers, so a program may legitimately declare a gate of its
// own by one of these names; when it has, the rule *fails* (returns false)
// and the call falls through to `gatecall`, which resolves it against the
// declaration. Otherwise it is a call to a gate that does not exist in this
// dialect, and that is thrown by name - the point of the rule being that the
// diagnostic says so rather than leaving a bare "unparsed input".
struct RejectQasm3OnlyGateExpr : public AbstractSyntaxTree {
  template <typename, typename>
  struct result {
    typedef bool type;
  };

  bool operator()(const std::string &gateName,
                  const std::unordered_map<std::string, StatementType>
                      &definedGates) const {
    if (definedGates.find(gateName) != definedGates.end()) return false;

    const std::string qasm2Spelling = gateName == "phase"    ? "p"
                                      : gateName == "cphase" ? "cp"
                                                             : "";

    throw std::invalid_argument(
        "'" + gateName +
        "' is an OpenQASM 3 stdgates.inc gate and is not available under "
        "'OPENQASM 2.0;'" +
        (qasm2Spelling.empty()
             ? " (it has no qelib1.inc equivalent)"
             : "; the qelib1.inc spelling is '" + qasm2Spelling + "'"));
  }
};

inline phx::function<RejectQasm3OnlyGateExpr> RejectQasm3OnlyGate;

struct AddCondQopExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      CondOpType &condOp,
      const std::unordered_map<std::string, IndexedId> &qreg_map,
      const std::unordered_map<std::string, IndexedId> &creg_map,
      const std::unordered_map<std::string, StatementType> &opaqueGates,
      const std::unordered_map<std::string, StatementType> &definedGates)
      const {
    StatementType stmt;

    const std::string &condId = boost::fusion::at_c<0>(condOp);
    int condVal = boost::fusion::at_c<1>(condOp);
    const QoperationStatement &op = boost::fusion::at_c<2>(condOp);

    stmt = op;

    stmt.opType = QoperationStatement::OperationType::CondUop;
    stmt.condValue = condVal;

    if (creg_map.find(condId) == creg_map.end())
      throw std::invalid_argument("Condition register not found: " + condId);
    else {
      const IndexedId &condCreg = creg_map.at(condId);

      for (int c = 0; c < condCreg.Eval(); ++c)
        stmt.cbits.push_back(condCreg.base + c);
    }

    return stmt;
  }
};

inline phx::function<AddCondQopExpr> AddCondQop;

// Builds the CondUop statements for one braced conditional body, given an
// already-resolved single classical bit and the boolean value it must equal
// for these statements to fire. Shared by the if-branch and (when present)
// the else-branch of a bit-form condition, since the two only differ in
// which expected value they condition on (see AddCondQopBracedExpr).
struct AddBitCondBranchExpr : public AbstractSyntaxTree {
  struct result {
    typedef std::vector<QoperationStatement> type;
  };

  std::vector<QoperationStatement> operator()(const std::vector<QopType> &body,
                                              int absoluteBit,
                                              bool expectedValue) const {
    std::vector<QoperationStatement> stmts;
    stmts.reserve(body.size());

    for (const auto &bodyStmt : body) {
      StatementType stmt = bodyStmt;
      stmt.opType = QoperationStatement::OperationType::CondUop;
      stmt.cbits = {absoluteBit};
      stmt.condValue = expectedValue ? 1 : 0;
      stmts.push_back(stmt);
    }

    return stmts;
  }
};

// A QASM3 braced conditional, `if ( <condHead> ) { <body> }`, with an
// optional `else { <elseBody> }`. `condHead` (see CondHeadType) already
// tells us which of the two condition shapes we parsed:
//
//  - Register form (`c == 2`): delegates per-statement cbit population to
//    the existing AddCondQopExpr, one call per body statement, exactly as
//    before this else-handling was added. `else` is rejected here rather
//    than silently mishandled: expressing "the register does NOT equal 2"
//    would need a not-equal condition primitive, and Circuit/Factory.h only
//    offers CreateEqualCondition (see CircuitFactory::CreateEqualCondition
//    in Circuit/Factory.h) - no not-equal primitive exists to build the
//    else-branch's condition, and adding one is out of scope here.
//
//  - Bit form (`c[0]` / `!c[0]`): both branches are expressible with
//    CreateEqualCondition on the very same bit - the else-branch is just the
//    complementary expected value - so it is fully supported.
struct AddCondQopBracedExpr : public AbstractSyntaxTree {
  struct result {
    typedef std::vector<QoperationStatement> type;
  };

  std::vector<QoperationStatement> operator()(
      const CondHeadType &condHead, const std::vector<QopType> &body,
      const boost::optional<std::vector<QopType>> &elseBody,
      const std::unordered_map<std::string, IndexedId> &qreg_map,
      const std::unordered_map<std::string, IndexedId> &creg_map,
      const std::unordered_map<std::string, StatementType> &opaqueGates,
      const std::unordered_map<std::string, StatementType> &definedGates)
      const {
    if (condHead.isBitForm) {
      if (creg_map.find(condHead.bit.id) == creg_map.end())
        throw std::invalid_argument("Condition register not found: " +
                                    condHead.bit.id);

      const IndexedId &condCreg = creg_map.at(condHead.bit.id);
      const int absoluteBit = condCreg.base + condHead.bit.index;

      AddBitCondBranchExpr addBranch;
      std::vector<QoperationStatement> stmts =
          addBranch(body, absoluteBit, condHead.bitExpected);

      if (elseBody) {
        std::vector<QoperationStatement> elseStmts =
            addBranch(*elseBody, absoluteBit, !condHead.bitExpected);
        stmts.insert(stmts.end(), elseStmts.begin(), elseStmts.end());
      }

      return stmts;
    }

    // Register-comparison form.
    if (elseBody)
      throw std::invalid_argument(
          "'else' on a register-comparison condition (if (" + condHead.regId +
          " == " + std::to_string(condHead.regValue) +
          ") { ... } else { ... }) is not supported: it would require a "
          "not-equal condition, and no such primitive exists. Only "
          "single-bit conditions (if (" +
          condHead.regId + "[i]) { ... } else { ... }) support 'else'.");

    std::vector<QoperationStatement> stmts;
    stmts.reserve(body.size());

    // Delegate the per-statement cbit population to the existing
    // AddCondQopExpr rather than duplicating its logic here.
    AddCondQopExpr addCondQop;
    for (const auto &bodyStmt : body) {
      CondOpType singleCondOp(condHead.regId, condHead.regValue, bodyStmt);
      stmts.push_back(addCondQop(singleCondOp, qreg_map, creg_map, opaqueGates,
                                 definedGates));
    }

    return stmts;
  }
};

inline phx::function<AddCondQopBracedExpr> AddCondQopBraced;

struct Program {
  double version = 2.0;
  std::vector<StatementType> statements;
  std::vector<std::string> comments;
  std::vector<std::string> includes;

  Program(const ProgramType &program = {}) {
    comments = boost::fusion::at_c<0>(program);
    version = boost::fusion::at_c<1>(program);
    includes = boost::fusion::at_c<2>(program);
    statements = boost::fusion::at_c<3>(program);
  }

  void clear() {
    comments.clear();
    includes.clear();
    statements.clear();
    version = 2.0;
  }

  template <typename Time = Types::time_type>
  std::shared_ptr<Circuits::Circuit<Time>> ToCircuit(
      std::unordered_map<std::string, StatementType> &opaqueGates,
      std::unordered_map<std::string, StatementType> &definedGates) const {
    auto circuit = std::make_shared<Circuits::Circuit<Time>>();

    for (const auto &stmt : statements)
      AddToCircuit(circuit, stmt, opaqueGates, definedGates);

    return circuit;
  }

  template <typename Time = Types::time_type>
  static void AddToCircuit(
      const std::shared_ptr<Circuits::Circuit<Time>> &circuit,
      const StatementType &stmt,
      std::unordered_map<std::string, StatementType> &opaqueGates,
      std::unordered_map<std::string, StatementType> &definedGates) {
    switch (stmt.opType) {
      case QoperationStatement::OperationType::Measurement: {
        if (stmt.qubits.size() != stmt.cbits.size())
          throw std::invalid_argument(
              "Measurement operation: number of qubits "
              "and classical bits do not match.");

        std::vector<std::pair<Types::qubit_t, size_t>> qs;
        for (size_t i = 0; i < stmt.qubits.size(); ++i)
          qs.push_back({static_cast<Types::qubit_t>(stmt.qubits[i]),
                        static_cast<size_t>(stmt.cbits[i])});

        auto measureOp = Circuits::CircuitFactory<Time>::CreateMeasurement(qs);
        circuit->AddOperation(measureOp);
      } break;
      case QoperationStatement::OperationType::Reset: {
        Types::qubits_vector qubits(stmt.qubits.begin(), stmt.qubits.end());
        auto resetOp = Circuits::CircuitFactory<Time>::CreateReset(qubits);
        circuit->AddOperation(resetOp);
      } break;

      case QoperationStatement::OperationType::Uop: {
        if (stmt.gateType == Circuits::QuantumGateType::kNone &&
            stmt.qubitsDecl.empty()) {
          // Identity gate ("id") or unrecognised no-op — skip silently.
          break;
        }
        if (stmt.gateType != Circuits::QuantumGateType::kNone) {
          // can add more than one gate here depending on what's in qubits
          double param1 = stmt.parameters.size() > 0 ? stmt.parameters[0] : 0;
          double param2 = stmt.parameters.size() > 1 ? stmt.parameters[1] : 0;
          double param3 = stmt.parameters.size() > 2 ? stmt.parameters[2] : 0;
          double param4 = stmt.parameters.size() > 3 ? stmt.parameters[3] : 0;

          int nrQubits = AddGateExpr::GateNrQubits(stmt.gateType);
          if (stmt.qubits.size() % nrQubits != 0)
            throw std::invalid_argument(
                "Uop operation: number of qubits does "
                "not match the gate requirements.");

          for (int pos = 0; pos < static_cast<int>(stmt.qubits.size());
               pos += nrQubits) {
            Types::qubits_vector gateQubits;
            for (int q = 0; q < nrQubits; ++q)
              gateQubits.push_back(
                  static_cast<Types::qubit_t>(stmt.qubits[pos + q]));

            auto gateOp = Circuits::CircuitFactory<Time>::CreateGate(
                stmt.gateType, gateQubits[0], nrQubits > 1 ? gateQubits[1] : 0,
                nrQubits > 2 ? gateQubits[2] : 0, param1, param2, param3,
                param4);

            circuit->AddOperation(gateOp);
          }
        } else {
          // it's a defined gate, check further and implement
          // will add several gates to the circuit
          if (stmt.paramsDecl.size() != stmt.parameters.size())
            throw std::invalid_argument(
                "Uop operation: number of parameters do "
                "not match the declaration.");

          std::unordered_map<std::string, double> variables;

          for (size_t i = 0; i < stmt.paramsDecl.size(); ++i)
            variables[stmt.paramsDecl[i]] = stmt.parameters[i];

          int nrQubits = static_cast<int>(stmt.qubitsDecl.size());
          if (stmt.qubits.size() % nrQubits != 0)
            throw std::invalid_argument(
                "Defined Uop operation: number of qubits "
                "does not match the gate requirements.");

          for (int pos = 0; pos < static_cast<int>(stmt.qubits.size());
               pos += nrQubits) {
            std::unordered_map<std::string, IndexedId> qubitMap;
            for (int q = 0; q < nrQubits; ++q) {
              IndexedId id;
              id.id = stmt.qubitsDecl[q];
              id.base = stmt.qubits[pos + q];
              id.index = 1;
              qubitMap[id.id] = id;
            }

            // now walk over all gates in declOps and add them to the circuit
            AddGateExpr addGate;
            for (const auto &op : stmt.declOps) {
              StatementType gateStmt =
                  addGate(op, qubitMap, opaqueGates, definedGates, variables);

              AddToCircuit(circuit, gateStmt, opaqueGates, definedGates);
            }
          }
        }
      } break;
      case QoperationStatement::OperationType::CondUop: {
        // Same guard as the Uop case above: without it a no-op statement -
        // "id", or a modifier that lowers to nothing such as `pow(0) @` or
        // `ctrl @ id` - reaches the defined-gate branch with an empty
        // qubitsDecl and evaluates 0 % 0, which raises SIGFPE and cannot be
        // caught and reported as a parse error.
        if (stmt.gateType == Circuits::QuantumGateType::kNone &&
            stmt.qubitsDecl.empty())
          break;

        unsigned long long int condValue =
            static_cast<unsigned long long int>(stmt.condValue);

        if (stmt.gateType != Circuits::QuantumGateType::kNone) {
          // can add more than one gate here depending on what's in qubits
          double param1 = stmt.parameters.size() > 0 ? stmt.parameters[0] : 0;
          double param2 = stmt.parameters.size() > 1 ? stmt.parameters[1] : 0;
          double param3 = stmt.parameters.size() > 2 ? stmt.parameters[2] : 0;
          double param4 = stmt.parameters.size() > 3 ? stmt.parameters[3] : 0;

          int nrQubits = AddGateExpr::GateNrQubits(stmt.gateType);
          if (stmt.qubits.size() % nrQubits != 0)
            throw std::invalid_argument(
                "Uop operation: number of qubits does "
                "not match the gate requirements.");

          std::vector<size_t> ind;
          std::vector<bool> condBits;

          for (size_t i = 0; i < stmt.cbits.size(); ++i) {
            ind.push_back(static_cast<size_t>(stmt.cbits[i]));
            condBits.push_back((condValue & 1) == 1);
            condValue >>= 1;
          }

          const auto condition =
              Circuits::CircuitFactory<Time>::CreateEqualCondition(ind,
                                                                   condBits);

          for (int pos = 0; pos < static_cast<int>(stmt.qubits.size());
               pos += nrQubits) {
            Types::qubits_vector gateQubits;
            for (int q = 0; q < nrQubits; ++q)
              gateQubits.push_back(
                  static_cast<Types::qubit_t>(stmt.qubits[pos + q]));

            auto gateOp = Circuits::CircuitFactory<Time>::CreateGate(
                stmt.gateType, gateQubits[0], nrQubits > 1 ? gateQubits[1] : 0,
                nrQubits > 2 ? gateQubits[2] : 0, param1, param2, param3,
                param4);

            auto condOp = Circuits::CircuitFactory<Time>::CreateConditionalGate(
                gateOp, condition);
            circuit->AddOperation(condOp);
          }
        } else {
          // it's a defined gate, check further and implement
          // will add several gates to the circuit
          if (stmt.paramsDecl.size() != stmt.parameters.size())
            throw std::invalid_argument(
                "Uop operation: number of parameters do "
                "not match the declaration.");

          std::unordered_map<std::string, double> variables;

          for (size_t i = 0; i < stmt.paramsDecl.size(); ++i)
            variables[stmt.paramsDecl[i]] = stmt.parameters[i];

          int nrQubits = static_cast<int>(stmt.qubitsDecl.size());
          if (stmt.qubits.size() % nrQubits != 0)
            throw std::invalid_argument(
                "Defined Uop operation: number of qubits "
                "does not match the gate requirements.");

          for (int pos = 0; pos < static_cast<int>(stmt.qubits.size());
               pos += nrQubits) {
            std::unordered_map<std::string, IndexedId> qubitMap;
            for (int q = 0; q < nrQubits; ++q) {
              IndexedId id;
              id.id = stmt.qubitsDecl[q];
              id.base = stmt.qubits[pos + q];
              id.index = 1;
              qubitMap[id.id] = id;
            }

            // now walk over all gates in declOps and add them to the circuit
            AddGateExpr addGate;
            for (const auto &op : stmt.declOps) {
              StatementType gateStmt =
                  addGate(op, qubitMap, opaqueGates, definedGates, variables);

              // make each of them conditioned on the original condition
              gateStmt.opType = QoperationStatement::OperationType::CondUop;
              gateStmt.condValue = static_cast<int>(condValue);
              gateStmt.cbits = stmt.cbits;

              AddToCircuit(circuit, gateStmt, opaqueGates, definedGates);
            }
          }
        }
      } break;
      case QoperationStatement::OperationType::Comment:
      case QoperationStatement::OperationType::Declaration:
      case QoperationStatement::OperationType::Barrier:
      case QoperationStatement::OperationType::OpaqueDecl:
      case QoperationStatement::OperationType::
          GateDecl:  // do not generate anything here, it's already handled when
                     // the gate is called
      default:
        // those are ignored
        break;
    }
  }
};

}  // namespace qasm

#endif  // !_SYNTAXTREE_H_
