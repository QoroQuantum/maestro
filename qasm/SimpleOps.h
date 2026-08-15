/**
 * @file SimpleOps.h
 * @ingroup qasm
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

inline phx::function<MakeIndexedIdExpression> MakeIndexedId;

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

// QASM3 call-site gate modifiers: `ctrl @`, `negctrl @`, `inv @` and
// `pow(k) @`.
enum class ModifierKind { Ctrl, NegCtrl, Inv, Pow };

struct ModifierType {
  ModifierType() = default;
  ModifierType(ModifierKind kind, double exponent = 0., int count = 1)
      : kind(kind), exponent(exponent), count(count) {}

  ModifierKind kind = ModifierKind::Ctrl;
  double exponent = 0.;  // only meaningful for ModifierKind::Pow
  // The optional control count of `ctrl(n) @` / `negctrl(n) @`; 1 for the
  // countless spelling and for every non-control modifier. Kept on the
  // modifier instead of expanded into n separate ModifierType entries so that
  // one parsed modifier remains one element of ModifierListType - the list's
  // source order is what tells AddModifiedGateExpr which qubit arguments a
  // control owns.
  int count = 1;
};

// Modifiers are kept in source order, i.e. outermost first: in
// `ctrl @ inv @ s q[0], q[1]` the control is applied last, to the inverted
// gate, and it consumes the first qubit argument.
using ModifierListType = std::vector<ModifierType>;
using ModifiedUopType = boost::fusion::vector<ModifierListType, UopType>;

struct MakePowModifierExpression {
  template <typename, typename>
  struct result {
    typedef ModifierType type;
  };

  // `variables` carries the top-level `input` bindings (threaded from the
  // `powMod` rule via std::ref(inputValues), the same grammar member
  // AddModifiedGate's call site uses), so `pow(theta) @ x q[0];` resolves
  // theta the same way an ordinary gate parameter does. A modified call
  // cannot appear inside a gate declaration body, so no macro formal
  // parameter is ever in scope here - any identifier absent from `variables`
  // is still an error (Variable::Eval throws), not a silent zero.
  template <typename E, typename V>
  ModifierType operator()(const E &exponent, const V &variables) const {
    return ModifierType(ModifierKind::Pow, exponent.Eval(variables));
  }
};

inline phx::function<MakePowModifierExpression> MakePowModifier;

// Builds a `ctrl @` / `negctrl @` modifier, with the optional control count
// of `ctrl(n) @`. `count` is absent for the countless spelling, in which case
// it is 1 - so `ctrl(1) @ g` and `ctrl @ g` produce identical modifiers.
//
// `variables` is threaded exactly as MakePowModifierExpression's is, so
// `ctrl(n) @` may name a top-level `input`; an identifier absent from it is
// an error via Variable::Eval rather than a silent zero. A count that is not
// a positive whole number is rejected here, where the spelling is still
// known. How many controls the lowering can actually realise is not decided
// here - that stays in AddModifiedGateExpr, which already reports it.
struct MakeCtrlModifierExpression {
  template <typename, typename, typename>
  struct result {
    typedef ModifierType type;
  };

  template <typename K, typename E, typename V>
  ModifierType operator()(K kind, const E &count, const V &variables) const {
    if (!count) return ModifierType(kind);

    const double value = count->Eval(variables);
    const double rounded = std::round(value);

    if (std::abs(value - rounded) > 1e-9 || rounded < 1. || rounded > 64.)
      throw std::invalid_argument(
          "The control count of ctrl(n) @ / negctrl(n) @ must be a positive "
          "whole number, got: " +
          std::to_string(value));

    return ModifierType(kind, 0., static_cast<int>(rounded));
  }
};

inline phx::function<MakeCtrlModifierExpression> MakeCtrlModifier;

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

// The parsed condition head of a QASM3 braced conditional, i.e. everything
// inside `if ( ... )`. Two shapes are folded into one type rather than a
// std::variant, matching the style of ModifierType above: each alternative
// of the `condHead` rule (register-comparison, bare bit, negated bit) has
// its own semantic action constructing one of these, so no
// BOOST_FUSION_ADAPT_STRUCT/attribute propagation is needed - only direct
// construction via qi::_val = Make...CondHead(...).
//
// Register form (`c == 2`): isBitForm is false, regId/regValue hold the
// register name and comparison value - the same information CondOpType
// carried before this type existed.
//
// Bit form (`c[0]` or `!c[0]`): isBitForm is true, bit holds the indexed
// classical bit and bitExpected holds the value that bit must equal for the
// condition to be true (true for the bare form, false for the negated
// form). This is deliberately expressed as a single bit + expected value,
// not a register + mask, since CreateEqualCondition already takes bit
// indices and expected booleans directly (see AddCondQopBracedExpr).
struct CondHeadType {
  bool isBitForm = false;

  std::string regId;
  int regValue = 0;

  IndexedId bit;
  bool bitExpected = true;
};

struct MakeRegCondHeadExpression {
  struct result {
    typedef CondHeadType type;
  };

  CondHeadType operator()(const std::string &regId, int regValue) const {
    CondHeadType head;
    head.isBitForm = false;
    head.regId = regId;
    head.regValue = regValue;
    return head;
  }
};

inline phx::function<MakeRegCondHeadExpression> MakeRegCondHead;

struct MakeBitCondHeadExpression {
  struct result {
    typedef CondHeadType type;
  };

  CondHeadType operator()(const IndexedId &bit, bool bitExpected) const {
    CondHeadType head;
    head.isBitForm = true;
    head.bit = bit;
    head.bitExpected = bitExpected;
    return head;
  }
};

inline phx::function<MakeBitCondHeadExpression> MakeBitCondHead;

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

inline phx::function<AddCregExpr> AddCreg;

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

inline phx::function<AddQregExpr> AddQreg;

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

inline phx::function<AddCommentExpr> AddComment;

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

inline phx::function<AddDeclarationExpr> AddDeclaration;

// Records a QASM3 `input <type> <name>;` declaration. The type annotation is
// accepted by the grammar but discarded before it reaches here - every input
// value arrives as a plain double via QasmToCirc::ParseAndTranslate's params
// map, so there is no type to track. The name is appended to inputNames in
// declaration order so callers can discover what a circuit requires (see
// QasmToCirc::GetInputs); the value itself is looked up later, at the point a
// gate parameter expression references it, via Variable::Eval against
// QasmGrammar::inputValues.
struct AddInputDeclExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(const std::string &name,
                                 std::vector<std::string> &inputNames) const {
    inputNames.push_back(name);

    QoperationStatement stmt;
    stmt.opType = QoperationStatement::OperationType::Declaration;
    // Recorded in `declaration`, the same IndexedId field AddDeclarationExpr
    // (the qreg/creg/qubit/bit declaration functor) populates, so a
    // Declaration-typed statement always carries its declared name in one
    // place regardless of which kind of declaration produced it. `index` is
    // left at its default 0: an `input` has no size to record.
    stmt.declaration = IndexedId(name, 0);
    stmt.declaration.declType = "input";

    return stmt;
  }
};

inline phx::function<AddInputDeclExpr> AddInputDecl;

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

inline phx::function<AddMeasureExpr> AddMeasure;

// The OpenQASM 3 grammar makes a measurement's `-> c` target optional, so
// `measure q;` ("measure and throw the result away") is spec-legal. It is not
// representable here: Circuits::CircuitFactory::CreateMeasurement takes
// (qubit, classical bit) pairs and MeasurementOperation always writes a bit,
// so there is no discard form to lower onto - and fabricating a classical bit
// to absorb the result would silently grow the program's classical register
// and change what a subsequent `if (c == ...)` sees. So the construct is
// recognised (see `measureNoTarget` in qasm.h) and rejected by name. Before
// this rule existed it fell through to the gate-call path and was reported as
// "Unsupported gate without parameters: measure", which named the wrong
// thing entirely.
struct RejectMeasureWithoutTargetExpr : public AbstractSyntaxTree {
  template <typename>
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(const ArgumentType & /*qubits*/) const {
    throw std::invalid_argument(
        "A measurement without a classical target ('measure q;') is not "
        "supported: every measurement must name the classical bit that "
        "receives its result, as in 'measure q -> c;' or 'c = measure q;'");
  }
};

inline phx::function<RejectMeasureWithoutTargetExpr> RejectMeasureWithoutTarget;

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

inline phx::function<AddResetExpr> AddReset;

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

    // A bare `barrier;` (no operand list) applies to every qubit. That is the
    // only way an empty list reaches here: the operand-list alternative of
    // `barrierOp` goes through `mixedList`, which requires at least one
    // operand.
    if (barrier.empty()) {
      for (const auto &[name, reg] : qreg_map) {
        const int size = static_cast<int>(std::round(reg.Eval()));
        for (int i = 0; i < size; ++i) qubit_set.insert(reg.base + i);
      }

      stmt.qubits.assign(qubit_set.begin(), qubit_set.end());

      return stmt;
    }

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

inline phx::function<AddBarrierExpr> AddBarrier;

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

inline phx::function<AddOpaqueDeclExpr> AddOpaqueDecl;

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

inline phx::function<AddGateDeclExpr> AddGateDecl;
}  // namespace qasm

#endif
