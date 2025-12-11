/**
 * @file SimpleOps.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Classes for the qasm parser and interpreter, dealing with the simple
 * declarations and operations.
 *
 * It's supposed to support only open qasm 2.0.
 */
#pragma once

#ifndef _SIMPLEOPS_H_
#define _SIMPLEOPS_H_

#include "Expr.h"

namespace qasm {
// something like this id[value] used for example by qreg and creg declarations
// also when a qubit or cbit is referenced
class IndexedId : public AbstractSyntaxTree {
 public:
  IndexedId() : index(0) {}
  IndexedId(const std::string &id, int index) : id(id), index(index) {}

  ~IndexedId() {}

  double Eval() const { return index; }

  operator std::string() const {
    return declType + " " + id + "[" + std::to_string(index) + "]";
  }

  std::string id;
  int index;
  int base = 0;  // to be used when allocating the qubits/cbits in the circuit
  std::string declType;  // "qreg" or "creg" or "id"
};

struct MakeIndexedIdExpression {
  template <typename, typename>
  struct result {
    typedef IndexedId type;
  };

  template <typename ID, typename IND>
  IndexedId operator()(const ID &id, IND index) const {
    return IndexedId(id, index);
  }
};

phx::function<MakeIndexedIdExpression> MakeIndexedId;

using SimpleExpType = std::variant<double, int, std::string>;

using ArgumentType = std::variant<std::string, IndexedId>;
using MixedListType = std::vector<ArgumentType>;

using SimpleGatecallType = boost::fusion::vector<std::string, MixedListType>;
using ExpGatecallType =
    boost::fusion::vector<std::string, std::vector<Expression>, MixedListType>;
using GatecallType = std::variant<SimpleGatecallType, ExpGatecallType>;

using UGateCallType =
    boost::fusion::vector<std::vector<Expression>, ArgumentType>;
using CXGateCallType = boost::fusion::vector<ArgumentType, ArgumentType>;

using UopType = std::variant<UGateCallType, CXGateCallType, GatecallType>;

struct QoperationStatement : public AbstractSyntaxTree {
  enum class OperationType {
    Comment,
    Declaration,  // creg, qreg
    Barrier,
    Measurement,
    Reset,
    OpaqueDecl,
    GateDecl,
    Uop,
    CondUop,
  };

  OperationType opType = OperationType::Comment;
  Circuits::QuantumGateType gateType = Circuits::QuantumGateType::kNone;

  std::string comment;

  IndexedId declaration;

  std::vector<int> qubits;
  std::vector<int> cbits;

  std::vector<double> parameters;

  std::vector<std::string> paramsDecl;
  std::vector<std::string> qubitsDecl;

  int condValue = 0;
  std::vector<UopType> declOps;
};

using StatementType = QoperationStatement;

using ProgramType =
    boost::fusion::vector<std::vector<std::string>, double,
                          std::vector<std::string>, std::vector<StatementType>>;

using ResetType = ArgumentType;
using MeasureType = boost::fusion::vector<ArgumentType, ArgumentType>;
using BarrierType = MixedListType;
// using QopType = std::variant<UopType, ResetType, MeasureType, BarrierType>;
using QopType = StatementType;
using CondOpType = boost::fusion::vector<std::string, int, QopType>;

using GateDeclType =
    boost::fusion::vector<std::string, std::vector<std::string>,
                          std::vector<std::string>>;
using SimpleBarrierType = std::vector<std::string>;
using GateDeclOpType = std::variant<UopType, SimpleBarrierType>;
using OpaqueDeclType =
    boost::fusion::vector<std::string, std::vector<std::string>,
                          std::vector<std::string>>;

struct AddCregExpr : public AbstractSyntaxTree {
  struct result {
    typedef IndexedId type;
  };

  IndexedId operator()(int &counter,
                       std::unordered_map<std::string, IndexedId> &creg_map,
                       const IndexedId &id) const {
    IndexedId id_copy = id;
    id_copy.base = counter;

    counter += static_cast<int>(std::round(id_copy.Eval()));

    creg_map[id_copy.id] = id_copy;

    id_copy.declType = "creg";

    return id_copy;
  }
};

phx::function<AddCregExpr> AddCreg;

struct AddQregExpr : public AbstractSyntaxTree {
  struct result {
    typedef IndexedId type;
  };

  IndexedId operator()(int &counter,
                       std::unordered_map<std::string, IndexedId> &qreg_map,
                       const IndexedId &id) const {
    IndexedId id_copy = id;
    id_copy.base = counter;

    counter += static_cast<int>(std::round(id_copy.Eval()));
    qreg_map[id_copy.id] = id_copy;

    id_copy.declType = "qreg";

    return id_copy;
  }
};

phx::function<AddQregExpr> AddQreg;

struct AddCommentExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(const std::string &comment) const {
    QoperationStatement stmt;

    stmt.opType = QoperationStatement::OperationType::Comment;
    stmt.comment = comment;

    return stmt;
  }
};

phx::function<AddCommentExpr> AddComment;

struct AddDeclarationExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(const IndexedId &id) const {
    QoperationStatement stmt;
    stmt.opType = QoperationStatement::OperationType::Declaration;
    stmt.declaration = id;

    return stmt;
  }
};

phx::function<AddDeclarationExpr> AddDeclaration;

struct AddMeasureExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const MeasureType &measure,
      const std::unordered_map<std::string, IndexedId> &creg_map,
      const std::unordered_map<std::string, IndexedId> &qreg_map) const {
    QoperationStatement stmt;
    stmt.opType = QoperationStatement::OperationType::Measurement;

    ArgumentType arg1 = boost::fusion::at_c<0>(measure);  // qubits info

    // there are two possibilities here, either it's an indexed id or a simple
    // id
    if (std::holds_alternative<IndexedId>(arg1)) {
      IndexedId indexedId = std::get<IndexedId>(arg1);
      auto it = qreg_map.find(indexedId.id);
      if (it != qreg_map.end()) {
        int base = it->second.base;
        stmt.qubits.push_back(base + indexedId.index);
      }
    } else if (std::holds_alternative<std::string>(arg1)) {
      std::string id = std::get<std::string>(arg1);
      auto it = qreg_map.find(id);
      if (it != qreg_map.end()) {
        int base = it->second.base;
        int size = static_cast<int>(std::round(it->second.Eval()));
        for (int i = 0; i < size; ++i) stmt.qubits.push_back(base + i);
      }
    }

    ArgumentType arg2 = boost::fusion::at_c<1>(measure);  // cbits info
    // there are two possibilities here, either it's an indexed id or a simple
    // id

    if (std::holds_alternative<IndexedId>(arg2)) {
      IndexedId indexedId = std::get<IndexedId>(arg2);
      auto it = creg_map.find(indexedId.id);
      if (it != creg_map.end()) {
        int base = it->second.base;
        stmt.cbits.push_back(base + indexedId.index);
      }
    } else if (std::holds_alternative<std::string>(arg2)) {
      std::string id = std::get<std::string>(arg2);
      auto it = creg_map.find(id);
      if (it != creg_map.end()) {
        int base = it->second.base;
        int size = static_cast<int>(std::round(it->second.Eval()));
        for (int i = 0; i < size; ++i) stmt.cbits.push_back(base + i);
      }
    }

    return stmt;
  }
};

phx::function<AddMeasureExpr> AddMeasure;

struct AddResetExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const ResetType &reset,
      const std::unordered_map<std::string, IndexedId> &qreg_map) const {
    QoperationStatement stmt;
    stmt.opType = QoperationStatement::OperationType::Reset;

    // there are two possibilities here, either it's an indexed id or a simple
    // id
    if (std::holds_alternative<IndexedId>(reset)) {
      IndexedId indexedId = std::get<IndexedId>(reset);
      auto it = qreg_map.find(indexedId.id);
      if (it != qreg_map.end()) {
        int base = it->second.base;
        stmt.qubits.push_back(base + indexedId.index);
      }
    } else if (std::holds_alternative<std::string>(reset)) {
      std::string id = std::get<std::string>(reset);
      auto it = qreg_map.find(id);
      if (it != qreg_map.end()) {
        int base = it->second.base;
        int size = static_cast<int>(std::round(it->second.Eval()));
        for (int i = 0; i < size; ++i) stmt.qubits.push_back(base + i);
      }
    }

    return stmt;
  }
};

phx::function<AddResetExpr> AddReset;

struct AddBarrierExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const BarrierType &barrier,
      const std::unordered_map<std::string, IndexedId> &qreg_map) const {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::Barrier;
    std::set<int> qubit_set;

    for (const auto &b : barrier) {
      // there are two possibilities here, either it's an indexed id or a simple
      // id
      if (std::holds_alternative<IndexedId>(b)) {
        IndexedId indexedId = std::get<IndexedId>(b);
        auto it = qreg_map.find(indexedId.id);
        if (it != qreg_map.end()) {
          int base = it->second.base;
          qubit_set.insert(base + indexedId.index);
        }
      } else if (std::holds_alternative<std::string>(b)) {
        std::string id = std::get<std::string>(b);
        auto it = qreg_map.find(id);
        if (it != qreg_map.end()) {
          int base = it->second.base;
          int size = static_cast<int>(std::round(it->second.Eval()));
          for (int i = 0; i < size; ++i) qubit_set.insert(base + i);
        }
      }
    }

    stmt.qubits.assign(qubit_set.begin(), qubit_set.end());

    return stmt;
  }
};

phx::function<AddBarrierExpr> AddBarrier;

struct AddOpaqueDeclExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const OpaqueDeclType &opaqueDecl,
      std::unordered_map<std::string, StatementType> &opaqueGates,
      const std::unordered_map<std::string, IndexedId> &qreg_map) const {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::OpaqueDecl;

    std::string gateName = boost::fusion::at_c<0>(opaqueDecl);

    stmt.comment = gateName;

    // maybe take some other infor from opaqueDecl if needed
    const std::vector<std::string> &params = boost::fusion::at_c<1>(opaqueDecl);
    const std::vector<std::string> &args = boost::fusion::at_c<2>(opaqueDecl);

    stmt.paramsDecl = params;
    stmt.qubitsDecl = args;

    // save into the map as well
    opaqueGates[gateName] = stmt;

    return stmt;
  }
};

phx::function<AddOpaqueDeclExpr> AddOpaqueDecl;

struct AddGateDeclExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const boost::fusion::vector<GateDeclType, std::vector<GateDeclOpType>>
          &gateDecl,
      std::unordered_map<std::string, StatementType> &definedGates) const {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::GateDecl;

    const GateDeclType &declInfo = boost::fusion::at_c<0>(gateDecl);

    const std::string &gateName = boost::fusion::at_c<0>(declInfo);
    const std::vector<std::string> &params = boost::fusion::at_c<1>(declInfo);
    const std::vector<std::string> &args = boost::fusion::at_c<2>(declInfo);

    if (args.empty())
      throw std::invalid_argument(
          "Gate declaration must have at least one qubit argument: " +
          gateName);
    else if (definedGates.find(gateName) !=
             definedGates
                 .end())  // for now do not allow redefinition, the
                          // biggest problem is that defined gates can be
                          // used inside other defined gates, otherwise
                          // redefinition would be simple to handle
      throw std::invalid_argument("Gate already defined: " + gateName);

    stmt.comment = gateName;
    stmt.paramsDecl = params;
    stmt.qubitsDecl = args;

    const std::vector<GateDeclOpType> &declOps =
        boost::fusion::at_c<1>(gateDecl);

    for (const auto &op : declOps) {
      if (std::holds_alternative<UopType>(op)) {
        const UopType &uop = std::get<UopType>(op);

        stmt.declOps.push_back(uop);
      }
      // ignore barriers
      // else if (std::holds_alternative<SimpleBarrierType>(op))
      //{
      //}
    }

    definedGates[gateName] = stmt;

    return stmt;
  }
};

phx::function<AddGateDeclExpr> AddGateDecl;
}  // namespace qasm

#endif
