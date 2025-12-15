/**
 * @file SyntaxTree.h
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
        std::string gateNameLower = gateName;
        std::transform(gateNameLower.begin(), gateNameLower.end(),
                       gateNameLower.begin(), ::tolower);

        const MixedListType &args = boost::fusion::at_c<1>(gate);

        const auto it = definedGates.find(gateName);
        if (it == definedGates.end()) {
          if (!IsSuppportedGate(gateNameLower))
            throw std::invalid_argument(
                "Unsupported gate without parameters: " + gateName);

          stmt.gateType = GetGateType(gateNameLower);
          const int expectedNrQubits = GateNrQubits(stmt.gateType);
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
        std::string gateNameLower = gateName;
        std::transform(gateNameLower.begin(), gateNameLower.end(),
                       gateNameLower.begin(), ::tolower);

        const std::vector<Expression> &params = boost::fusion::at_c<1>(gate);
        const MixedListType &args = boost::fusion::at_c<2>(gate);

        for (const auto &p : params)
          stmt.parameters.push_back(p.Eval(variables));

        const auto it = definedGates.find(gateName);
        if (it == definedGates.end()) {
          if (IsSuppportedGate(gateNameLower)) {
            if (params.empty())
              throw std::invalid_argument("Gate " + gateName +
                                          " requires parameters");
            else if (params.size() == 1) {
              if (!IsSuppportedOneParamGate(gateNameLower))
                throw std::invalid_argument(
                    "This gate does not allow one parameter: " + gateName);
            } else if (params.size() > 1) {
              if (!IsSuppportedMultipleParamsGate(gateNameLower))
                throw std::invalid_argument(
                    "This gate does not allow multiple parameters: " +
                    gateName);

              if (params.size() > 4)
                throw std::invalid_argument("Too many parameters for gate: " +
                                            gateName);
            }

            stmt.gateType = GetGateType(gateNameLower);
            const int expectedNrQubits = GateNrQubits(stmt.gateType);
            if (static_cast<int>(args.size()) != expectedNrQubits)
              throw std::invalid_argument(
                  "Gate " + gateName + " requires exactly " +
                  std::to_string(expectedNrQubits) + " qubits");

            if (gateNameLower == "u1") {
              const double lambda = stmt.parameters[0];
              stmt.parameters[0] = 0.0;  // theta
              stmt.parameters.push_back(0.);
              stmt.parameters.push_back(lambda);
            } else if (gateNameLower == "u2") {
              const double phi = stmt.parameters[0];
              const double lambda = stmt.parameters[1];

              stmt.parameters[0] = M_PI / 2.0;  // theta
              stmt.parameters[1] = phi;
              stmt.parameters.push_back(lambda);
            } else if (gateNameLower == "cu1") {
              const double lambda = stmt.parameters[0];
              stmt.parameters[0] = 0.0;  // theta
              stmt.parameters.push_back(0.);
              stmt.parameters.push_back(lambda);
            } else if (gateNameLower == "cu2") {
              const double phi = stmt.parameters[0];
              const double lambda = stmt.parameters[1];

              stmt.parameters[0] = M_PI / 2.0;  // theta
              stmt.parameters[1] = phi;
              stmt.parameters.push_back(lambda);
            }
          } else if (gateNameLower == "u1") {
            stmt.gateType = Circuits::QuantumGateType::kUGateType;

            if (stmt.parameters.size() != 1)
              throw std::invalid_argument(
                  "Gate u1 requires exactly one parameter");

            const double lambda = stmt.parameters[0];
            stmt.parameters[0] = 0.0;  // theta
            stmt.parameters.push_back(0.);
            stmt.parameters.push_back(lambda);

            if (args.size() != 1)
              throw std::invalid_argument("Gate " + gateName +
                                          " requires exactly 1 qubit");
          } else if (gateNameLower == "u2") {
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
          } else if (gateNameLower == "u3") {
            stmt.gateType = Circuits::QuantumGateType::kUGateType;

            if (stmt.parameters.size() != 3)
              throw std::invalid_argument("Gate u3 requires three parameters");
            // nothing to do, u3 already has the correct parameters

            if (args.size() != 1)
              throw std::invalid_argument("Gate " + gateName +
                                          " requires exactly 1 qubit");
          } else if (gateNameLower == "cu1") {
            stmt.gateType = Circuits::QuantumGateType::kCUGateType;

            if (stmt.parameters.size() != 1)
              throw std::invalid_argument(
                  "Gate cu1 requires exactly one parameter");

            const double lambda = stmt.parameters[0];
            stmt.parameters[0] = 0.0;  // theta
            stmt.parameters.push_back(0.);
            stmt.parameters.push_back(lambda);

            if (args.size() != 2)
              throw std::invalid_argument("Gate " + gateName +
                                          " requires exactly 2 qubits");
          } else if (gateNameLower == "cu2") {
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
          } else if (gateNameLower == "cu3") {
            stmt.gateType = Circuits::QuantumGateType::kCUGateType;

            if (stmt.parameters.size() != 3)
              throw std::invalid_argument("Gate cu3 requires three parameters");
            // nothing to do, cu3 already has the correct parameters

            if (args.size() != 2)
              throw std::invalid_argument("Gate " + gateName +
                                          " requires exactly 2 qubits");
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

  Circuits::QuantumGateType GetGateType(const std::string &gateN) const {
    std::string gateName = gateN;
    std::transform(gateName.begin(), gateName.end(), gateName.begin(),
                   ::tolower);

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
    else if (gateName == "sdg" || gateName == "sxdag")
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
    else if (gateName == "cx")
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
    else if (gateName == "p")
      return Circuits::QuantumGateType::kPhaseGateType;
    else if (gateName == "rx")
      return Circuits::QuantumGateType::kRxGateType;
    else if (gateName == "ry")
      return Circuits::QuantumGateType::kRyGateType;
    else if (gateName == "rz")
      return Circuits::QuantumGateType::kRzGateType;
    else if (gateName == "cp")
      return Circuits::QuantumGateType::kCPGateType;
    else if (gateName == "crx")
      return Circuits::QuantumGateType::kCRxGateType;
    else if (gateName == "cry")
      return Circuits::QuantumGateType::kCRyGateType;
    else if (gateName == "crz")
      return Circuits::QuantumGateType::kCRzGateType;
    else if (gateName == "u" || gateName == "u3" || gateName == "u1")
      return Circuits::QuantumGateType::kUGateType;
    else if (gateName == "cu" || gateName == "cu3" || gateName == "cu1")
      return Circuits::QuantumGateType::kCUGateType;

    return Circuits::QuantumGateType::kNone;  // this will be returnd also for
                                              // 'id'
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

 private:
  static inline std::unordered_set<std::string> allowedNoParamGates = {
      "x",    "y",   "z",     "h",      "s",     "sdg",  "sdag", "t",  "tdg",
      "tdag", "sx",  "sxdg",  "sxdag",  "k",     "swap", "cx",   "cy", "cz",
      "ch",   "csx", "csxdg", "csxdag", "cswap", "ccx",  "id"};
  static inline std::unordered_set<std::string> allowedOneParamGates = {
      "p", "rx", "ry", "rz", "cp", "crx", "cry", "crz", "u1", "cu1"};
  static inline std::unordered_set<std::string> allowedMultipleParamsGates = {
      "u", "u3", "cu1", "cu3"};  // max 4
};

phx::function<AddGateExpr> AddGate;

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

phx::function<AddCondQopExpr> AddCondQop;

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
