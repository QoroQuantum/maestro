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

#include <cctype>
#include <limits>
#include <string_view>

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

enum class DeclarationKind { Qubit, Bit, Input, Gate, Opaque };

using DeclarationRegistry = std::unordered_map<std::string, DeclarationKind>;

inline std::string_view DeclarationKindName(DeclarationKind kind) {
  switch (kind) {
    case DeclarationKind::Qubit:
      return "quantum register";
    case DeclarationKind::Bit:
      return "classical register";
    case DeclarationKind::Input:
      return "input";
    case DeclarationKind::Gate:
      return "gate";
    case DeclarationKind::Opaque:
      return "opaque gate";
  }

  return "declaration";
}

inline void RegisterDeclaration(DeclarationRegistry &declarations,
                                const std::string &name, DeclarationKind kind) {
  const auto [it, inserted] = declarations.emplace(name, kind);
  if (inserted) return;

  if (it->second == kind)
    throw std::invalid_argument("Duplicate declaration of '" + name + "'.");

  throw std::invalid_argument(
      "Declaration of '" + name + "' conflicts with an existing " +
      std::string(DeclarationKindName(it->second)) + ".");
}

inline int ValidateRegisterAllocation(const IndexedId &id, int currentSize,
                                      std::string_view kind) {
  const int size = id.index;
  if (size <= 0)
    throw std::invalid_argument(std::string(kind) + " register '" + id.id +
                                "' must have a positive size.");

  if (currentSize < 0 || currentSize > std::numeric_limits<int>::max() - size)
    throw std::overflow_error(std::string(kind) +
                              " register allocation exceeds supported "
                              "maximum.");

  return size;
}

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
using RegisterMap = std::unordered_map<std::string, IndexedId>;
using MixedListType = std::vector<ArgumentType>;
using InputDeclType =
    boost::fusion::vector<std::string, boost::optional<int>, std::string>;

inline std::vector<int> ResolveRegisterOperand(const ArgumentType &argument,
                                               const RegisterMap &registers,
                                               std::string_view kind) {
  const std::string &name = std::holds_alternative<IndexedId>(argument)
                                ? std::get<IndexedId>(argument).id
                                : std::get<std::string>(argument);
  const auto it = registers.find(name);
  if (it == registers.end())
    throw std::invalid_argument("Undeclared " + std::string(kind) +
                                " register '" + name + "'.");

  const IndexedId &declaration = it->second;
  if (std::holds_alternative<IndexedId>(argument)) {
    const int index = std::get<IndexedId>(argument).index;
    if (index < 0 || index >= declaration.index) {
      std::string titledKind(kind);
      titledKind[0] = static_cast<char>(
          std::toupper(static_cast<unsigned char>(titledKind[0])));
      throw std::out_of_range(titledKind + " register '" + name + "' index " +
                              std::to_string(index) + " is out of range [0, " +
                              std::to_string(declaration.index) + ").");
    }

    return {declaration.base + index};
  }

  std::vector<int> resolved;
  resolved.reserve(static_cast<size_t>(declaration.index));
  for (int index = 0; index < declaration.index; ++index)
    resolved.push_back(declaration.base + index);
  return resolved;
}

inline void RequireMatchingRegisterWidths(const std::vector<int> &left,
                                          const std::vector<int> &right,
                                          std::string_view operation) {
  if (left.size() != right.size())
    throw std::invalid_argument(std::string(operation) +
                                " operands must have the same width.");
}

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
    if (!std::isfinite(value))
      throw std::invalid_argument(
          "The control count of ctrl(n) @ / negctrl(n) @ must be finite, "
          "got: " +
          std::to_string(value));

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

  // Expected values are stored explicitly instead of packed into an integer,
  // so QASM3 conjunctions are not limited by the host integer width.
  std::vector<bool> condExpected;
  // Condition bits for a conditional Measurement/Reset; CondUop uses cbits.
  std::vector<int> condBits;
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
// Bit form (`c[0]`, `!c[0]`, or a `&&`-joined chain of either): isBitForm is
// true and bits holds one entry per tested bit, each pairing the indexed
// classical bit with the value it must equal for the condition to be true
// (true for the bare form, false for the negated form). This is deliberately
// expressed as bits + expected values, not a register + mask, since
// CreateEqualCondition already takes bit indices and expected booleans
// directly (see AddCondQopBracedExpr).
struct CondBitTest {
  IndexedId bit;
  bool expected = true;
};

struct CondHeadType {
  bool isBitForm = false;

  std::string regId;
  int regValue = 0;

  std::vector<CondBitTest> bits;
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

struct MakeCondBitTestExpression {
  struct result {
    typedef CondBitTest type;
  };

  CondBitTest operator()(const IndexedId &bit, bool expected) const {
    CondBitTest test;
    test.bit = bit;
    test.expected = expected;
    return test;
  }
};

inline phx::function<MakeCondBitTestExpression> MakeCondBitTest;

struct MakeBitCondHeadExpression {
  struct result {
    typedef CondHeadType type;
  };

  CondHeadType operator()(const std::vector<CondBitTest> &bits) const {
    CondHeadType head;
    head.isBitForm = true;
    head.bits = bits;
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
                       DeclarationRegistry &declarations,
                       const IndexedId &id) const {
    IndexedId id_copy = id;
    const int size = ValidateRegisterAllocation(id_copy, counter, "Classical");
    RegisterDeclaration(declarations, id_copy.id, DeclarationKind::Bit);

    id_copy.base = counter;
    id_copy.declType = "creg";

    counter += size;
    creg_map[id_copy.id] = id_copy;

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
                       DeclarationRegistry &declarations,
                       const IndexedId &id) const {
    IndexedId id_copy = id;
    const int size = ValidateRegisterAllocation(id_copy, counter, "Quantum");
    RegisterDeclaration(declarations, id_copy.id, DeclarationKind::Qubit);

    id_copy.base = counter;
    id_copy.declType = "qreg";

    counter += size;
    qreg_map[id_copy.id] = id_copy;

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

inline void ValidateInputDeclaration(const std::string &name,
                                     const std::string &type,
                                     const boost::optional<int> &width) {
  if (width && *width <= 0)
    throw std::invalid_argument("Input '" + name +
                                "' must have a positive type width.");

  if (type == "bool") {
    if (width)
      throw std::invalid_argument("Boolean input '" + name +
                                  "' cannot have a width designator.");
    return;
  }

  if (type == "float") {
    if (width && *width != 32 && *width != 64)
      throw std::invalid_argument(
          "Input '" + name +
          "' uses an unsupported float width; only float, float[32], and "
          "float[64] are supported.");
    return;
  }

  if (type == "angle") {
    if (width)
      throw std::invalid_argument(
          "Precisely quantized sized angle inputs are not supported.");
    return;
  }

  if (width && *width > 64)
    throw std::invalid_argument("Input '" + name +
                                "' uses an integer width above 64, which "
                                "the numeric binding API cannot represent.");
}

inline double ValidateInputBinding(const std::string &name,
                                   const std::string &type,
                                   const boost::optional<int> &width,
                                   double value) {
  ValidateInputDeclaration(name, type, width);

  if (!std::isfinite(value))
    throw std::invalid_argument("Input binding '" + name + "' must be finite.");

  if (type == "bool") {
    if (value != 0. && value != 1.)
      throw std::invalid_argument("Input binding '" + name +
                                  "' must be a boolean value (0 or 1).");
    return value;
  }

  if (type == "float") {
    if (!width || *width == 64) return value;
    const float narrowed = static_cast<float>(value);
    if (!std::isfinite(narrowed))
      throw std::invalid_argument("Input binding '" + name +
                                  "' does not fit float[32].");
    return static_cast<double>(narrowed);
  }

  if (type == "angle") {
    double normalized = std::fmod(value, 2. * M_PI);
    if (normalized < 0.) normalized += 2. * M_PI;
    return normalized;
  }

  const bool isUnsigned = type == "uint";
  if (std::trunc(value) != value)
    throw std::invalid_argument("Input binding '" + name +
                                "' must be an integer value.");
  if (isUnsigned && value < 0.)
    throw std::invalid_argument("Unsigned input binding '" + name +
                                "' must be non-negative.");

  if (width) {
    const double upper = std::ldexp(1., isUnsigned ? *width : *width - 1);
    const double lower = isUnsigned ? 0. : -upper;
    if (value < lower || value >= upper)
      throw std::invalid_argument("Input binding '" + name +
                                  "' does not fit its declared " + type + "[" +
                                  std::to_string(*width) + "] type.");
  }

  return value;
}

// Records a QASM3 input declaration and publishes a validated caller binding
// to the expression environment only when that declaration is reached.
struct AddInputDeclExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(
      const InputDeclType &inputDecl, std::vector<std::string> &inputNames,
      DeclarationRegistry &declarations,
      const std::unordered_map<std::string, double> &inputBindings,
      std::unordered_map<std::string, double> &visibleInputValues) const {
    const std::string &type = boost::fusion::at_c<0>(inputDecl);
    const boost::optional<int> &width = boost::fusion::at_c<1>(inputDecl);
    const std::string &name = boost::fusion::at_c<2>(inputDecl);
    ValidateInputDeclaration(name, type, width);
    RegisterDeclaration(declarations, name, DeclarationKind::Input);
    inputNames.push_back(name);

    const auto binding = inputBindings.find(name);
    if (binding != inputBindings.end())
      visibleInputValues[name] =
          ValidateInputBinding(name, type, width, binding->second);

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

  QoperationStatement operator()(const MeasureType &measure,
                                 const RegisterMap &creg_map,
                                 const RegisterMap &qreg_map) const {
    QoperationStatement stmt;
    stmt.opType = QoperationStatement::OperationType::Measurement;

    stmt.qubits = ResolveRegisterOperand(boost::fusion::at_c<0>(measure),
                                         qreg_map, "quantum");
    stmt.cbits = ResolveRegisterOperand(boost::fusion::at_c<1>(measure),
                                        creg_map, "classical");
    RequireMatchingRegisterWidths(stmt.qubits, stmt.cbits, "Measurement");

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

  QoperationStatement operator()(const ResetType &reset,
                                 const RegisterMap &qreg_map) const {
    QoperationStatement stmt;
    stmt.opType = QoperationStatement::OperationType::Reset;
    stmt.qubits = ResolveRegisterOperand(reset, qreg_map, "quantum");
    return stmt;
  }
};

inline phx::function<AddResetExpr> AddReset;

struct AddBarrierExpr : public AbstractSyntaxTree {
  struct result {
    typedef QoperationStatement type;
  };

  QoperationStatement operator()(const BarrierType &barrier,
                                 const RegisterMap &qreg_map) const {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::Barrier;
    std::set<int> qubit_set;

    // A bare QASM3 barrier applies to every declared qubit. The Circuit IR
    // has no barrier operation, so Program intentionally erases this statement.
    if (barrier.empty()) {
      for (const auto &[name, reg] : qreg_map)
        for (int index = 0; index < reg.index; ++index)
          qubit_set.insert(reg.base + index);
    } else {
      for (const auto &operand : barrier) {
        const std::vector<int> resolved =
            ResolveRegisterOperand(operand, qreg_map, "quantum");
        qubit_set.insert(resolved.begin(), resolved.end());
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
      const std::unordered_map<std::string, IndexedId> &qreg_map,
      DeclarationRegistry &declarations) const {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::OpaqueDecl;

    std::string gateName = boost::fusion::at_c<0>(opaqueDecl);
    RegisterDeclaration(declarations, gateName, DeclarationKind::Opaque);

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
      std::unordered_map<std::string, StatementType> &definedGates,
      DeclarationRegistry &declarations) const {
    StatementType stmt;
    stmt.opType = QoperationStatement::OperationType::GateDecl;

    const GateDeclType &declInfo = boost::fusion::at_c<0>(gateDecl);

    const std::string &gateName = boost::fusion::at_c<0>(declInfo);
    const std::vector<std::string> &params = boost::fusion::at_c<1>(declInfo);
    const std::vector<std::string> &args = boost::fusion::at_c<2>(declInfo);

    RegisterDeclaration(declarations, gateName, DeclarationKind::Gate);

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
