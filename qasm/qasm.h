/**
 * @file qasm.h
 * @ingroup qasm
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Classes for the qasm parser and interpreter.
 *
 * Supports the static-circuit subset of OpenQASM 2 and OpenQASM 3.
 */

#pragma once

#ifndef _QASM_H_
#define _QASM_H_

#include "SyntaxTree.h"

namespace qasm {

struct error_handler_ {
  template <typename, typename, typename>
  struct result {
    typedef void type;
  };

  template <typename Iterator>
  void operator()(qi::info const &what, Iterator err_pos, Iterator last) const {
    std::cout << "Error! Expecting " << what << " here: \""
              << std::string(err_pos, last) << "\"\n";
  }
};

inline phx::function<error_handler_> const error_handler = error_handler_();
}  // namespace qasm

BOOST_FUSION_ADAPT_STRUCT(qasm::Program,
                          (std::vector<std::string>, comments)(double, version)(
                              std::vector<std::string>,
                              includes)(std::vector<qasm::StatementType>,
                                        statements))

namespace qasm {

inline void printd(const double &v) { std::cout << "version: " << v << "\n"; }

inline void prints(const std::string &s) {
  std::cout << "statement: " << s << "\n";
}

// TODO:
// 1. 'opaque' will be parsed but ignored in the first phase.
// 2. 'barrier' will be parsed, but ignored in the first phase. In this case we
// might want to add a 'barrier' operation in our circuit. For now it's not
// existent. Adding it would have implications in circuit execution with the
// discrete event simulator and also in the transpiler functionality.

template <typename Iterator>
struct QasmSkipper : qi::grammar<Iterator> {
  QasmSkipper() : QasmSkipper::base_type(skip) {
    // Block comments are accepted as whitespace in both dialects, including
    // before a header, as a harmless compatibility extension.
    blockComment = qi::lexeme[qi::lit("/*") >> *(qi::char_ - qi::lit("*/")) >>
                              qi::lit("*/")];
    skip = ascii::space | blockComment;
  }

  qi::rule<Iterator> skip;
  qi::rule<Iterator> blockComment;
};

struct MakeSupportedVersionExpr {
  template <typename, typename, typename>
  struct result {
    typedef double type;
  };

  // `VersionSpecifier: [0-9]+ ('.' [0-9]+)?` - the minor version is optional
  // in QASM3, but QASM2 spells the field as `real`, which is never bare.
  double operator()(unsigned int major,
                    const boost::optional<unsigned int> &minor,
                    bool &isQasm3) const {
    if (major != 2U && major != 3U)
      throw std::invalid_argument("Unsupported OpenQASM major version " +
                                  std::to_string(major) +
                                  ". Only versions 2.x and 3.x are supported.");

    if (major == 2U && !minor)
      throw std::invalid_argument(
          "OpenQASM 2 requires an explicit minor version, as in "
          "'OPENQASM 2.0;'.");

    isQasm3 = major == 3U;
    if (!minor) return static_cast<double>(major);

    return std::stod(std::to_string(major) + "." + std::to_string(*minor));
  }
};

inline phx::function<MakeSupportedVersionExpr> MakeSupportedVersion;

struct ValidateIncludeExpr {
  template <typename>
  struct result {
    typedef std::string type;
  };

  std::string operator()(const std::string &includeName) const {
    if (includeName != "qelib1.inc" && includeName != "stdgates.inc")
      throw std::invalid_argument(
          "Unsupported OpenQASM include '" + includeName +
          "'. Only qelib1.inc and "
          "stdgates.inc are recognized prelude markers.");
    return includeName;
  }
};

inline phx::function<ValidateIncludeExpr> ValidateInclude;

template <typename Iterator = std::string::iterator,
          typename Skipper = QasmSkipper<Iterator>>
struct QasmGrammar : qi::grammar<Iterator, Program(), Skipper> {
  QasmGrammar() : QasmGrammar::base_type{program} {
    version = (qi::omit[qi::lexeme[qi::lit("OPENQASM") >> qi::space]] >>
               qi::lexeme[qi::uint_ >> -('.' >> qi::uint_)] >>
               ';')[qi::_val = MakeSupportedVersion(qi::_1, qi::_2,
                                                    phx::ref(isQasm3))];

    // Keyword -> diagnostic message table for QASM3 constructs outside our
    // supported subset. Table-driven rather than one rule per keyword, since
    // every entry follows the same "recognise keyword, throw its message"
    // shape (see `unsupportedConstruct` below).
    unsupportedKeywords.add("for", "OpenQASM 3 'for' loops are not supported.")(
        "while", "OpenQASM 3 'while' loops are not supported.")(
        "def", "OpenQASM 3 subroutine definitions ('def') are not supported.")(
        "let", "OpenQASM 3 register aliases ('let') are not supported.")(
        "duration",
        "OpenQASM 3 duration declarations ('duration') are not supported.")(
        "delay", "OpenQASM 3 delay instructions ('delay') are not supported.")(
        "box", "OpenQASM 3 box blocks ('box') are not supported.")(
        "array", "OpenQASM 3 array declarations ('array') are not supported.")(
        "output",
        "OpenQASM 3 output declarations ('output') are not supported.")(
        "const",
        "OpenQASM 3 constant declarations ('const') are not supported.")(
        "extern",
        "OpenQASM 3 external subroutines ('extern') are not supported.")(
        "stretch",
        "OpenQASM 3 stretch declarations ('stretch') are not supported.")(
        "pragma", "OpenQASM 3 pragma statements are not supported.")(
        "defcal",
        "OpenQASM 3 calibration definitions ('defcal') are not supported.");

    // The gate names that exist only in OpenQASM 3's stdgates.inc, kept as a
    // symbol table for the same reason unsupportedKeywords is one: the
    // recognition is a name lookup. Each name maps to itself, so the rule's
    // attribute is the offending spelling and the diagnostic can quote it.
    qasm3OnlyGates.add("phase", "phase")("cphase", "cphase")("gphase",
                                                             "gphase");

    comments %= *comment;
    includes %= *include;

    // An absent header intentionally retains the parser's historical QASM2
    // compatibility mode; an explicit header selects and validates a dialect.
    program = comments >> (-version) >> includes >> statements;

    // condOpBraced is tried before the plain `statement` alternative (which
    // contains the unbraced condOp) so that a braced conditional isn't first
    // matched up to `if ( ... )` by condOp only to fail on '{'. condOpBraced
    // synthesizes a std::vector<StatementType> (one conditioned statement per
    // body qop) and is spliced in, while a plain statement is pushed as a
    // single element - this is the flattening the braced multi-statement form
    // requires.
    statements =
        *(condOpBraced[phx::insert(qi::_val, phx::end(qi::_val),
                                   phx::begin(qi::_1), phx::end(qi::_1))] |
          statement[phx::push_back(qi::_val, qi::_1)]);

    statement =
        comment[qi::_val = AddComment(qi::_1)] |
        decl[qi::_val = AddDeclaration(qi::_1)] |
        opaque[qi::_val =
                   AddOpaqueDecl(qi::_1, std::ref(opaqueGates),
                                 std::ref(qreg_map), std::ref(declarations))] |
        condOp[qi::_val =
                   AddCondQop(qi::_1, std::ref(qreg_map), std::ref(creg_map),
                              std::ref(opaqueGates), std::ref(definedGates))] |
        gatedeclfull[qi::_val = AddGateDecl(qi::_1, std::ref(definedGates),
                                            std::ref(declarations))] |
        inputdecl[qi::_val = AddInputDecl(
                      qi::_1, std::ref(inputNames), std::ref(declarations),
                      std::ref(inputBindings), std::ref(inputValues))] |
        unsupportedConstruct[qi::_val = qi::_1] | qop[qi::_val = qi::_1];

    // this is the opaque gate declaration, it will be simply ignored (in 3.0 is
    // supposed to be ignored)

    opaque %= qi::omit[qi::lexeme[qi::lit("opaque") >> qi::space]] >>
              identifier >>
              (('(' >> idList >> ')') | ('(' >> qi::eps >> ')') | qi::eps) >>
              idList >> ';';

    // **************************************************************************************************************************************************************

    // some of the more complex things

    gatedecl %= qi::omit[qi::lexeme[qi::lit("gate") >> qi::space]] >>
                identifier >>
                (('(' >> idList >> ')') | ('(' >> qi::eps >> ')') | qi::eps) >>
                idList >> '{';

    simplebarrier %=
        qi::omit[qi::lexeme[qi::lit("barrier") >> qi::space]] >> idList >> ';';
    gateBodyModifier = (+modifier)[qi::_val = RejectGateBodyModifier(qi::_1)];
    gatedeclop %= simplebarrier | (uop >> ';') | gateBodyModifier;

    gatedeclfull %= gatedecl >> *gatedeclop >> '}';

    // **************************************************************************************************************************************************************

    // `==` and the `!` below are supported only in conditional heads; they are
    // intentionally not part of the general expression grammar.
    condOp %= qi::lit("if") >> '(' >> identifier >> qi::lit("==") >> qi::int_ >>
              ')' >> qop;

    // The condition head inside `if ( ... )` for the QASM3 braced form. Two
    // shapes, each with its own semantic action (per-alternative, not one
    // action on the whole alternation - see the file-level Spirit lessons
    // note) so CondHeadType can be constructed directly without needing a
    // BOOST_FUSION_ADAPT_STRUCT for an asymmetric union of the two shapes:
    //   - register comparison: `c == 2`
    //   - negated bit: `!c[0]`   -> bit must equal 0
    //   - bare bit:    `c[0]`    -> bit must equal 1
    //   - `&&` chain:  `c0[0] && !c1[0]` -> every listed bit must match
    // Register form is tried first only to mirror condOp/condOpBraced's
    // pre-existing ordering; the two are unambiguous regardless of order
    // since '==' vs '[' immediately disambiguate them after `identifier`.
    condBitTest =
        (qi::lit('!') >> indexedId)[qi::_val = MakeCondBitTest(qi::_1, false)] |
        indexedId[qi::_val = MakeCondBitTest(qi::_1, true)];

    // A `&&`-joined chain is what CircToQasm emits for a condition spanning
    // more than one classical bit, so accepting it here is what makes
    // circuit -> QASM3 -> circuit round-trip for those conditions.
    condHead =
        (identifier >> qi::lit("==") >>
         qi::int_)[qi::_val = MakeRegCondHead(qi::_1, qi::_2)] |
        (condBitTest % qi::lit("&&"))[qi::_val = MakeBitCondHead(qi::_1)];

    // QASM3 braced conditional: `if (c == 1) { x q[0]; y q[1]; }`, plus the
    // three additional Qiskit spellings this grammar accepts via condHead:
    // `if (c[0]) { ... }`, `if (!c[0]) { ... }`, and an optional
    // `else { ... }` clause. The else-clause is parsed for both condHead
    // shapes - AddCondQopBraced is where a register-form else is rejected
    // with a clear error, since accepting it here and only failing deep in
    // circuit construction would be a worse diagnostic. The cbit-population
    // logic is not duplicated here; it is delegated to AddCondQopExpr
    // (register form) or done directly against the resolved single bit
    // (bit form) inside AddCondQopBraced.
    //
    // `*qop` accepts zero repetitions, so `if (c == 1) { }` parses to an
    // empty body vector, which splices in nothing at the call site in
    // `statements` - i.e. it is accepted as a documented no-op, not rejected
    // and not a way to silently drop the rest of the program. See
    // QASM3EmptyBracedConditionalBodyIsANoOp in tests/qasm.cpp.
    condOpBraced =
        qi::eps(phx::ref(isQasm3)) >>
        (qi::lit("if") >> '(' >> condHead >> ')' >> '{' >> *qop >> '}' >>
         -(qi::lit("else") >> '{' >> *qop >> '}'))
            [qi::_val = AddCondQopBraced(
                 qi::_1, qi::_2, qi::_3, std::ref(qreg_map), std::ref(creg_map),
                 std::ref(opaqueGates), std::ref(definedGates))];

    // `gateCallStatement: ... (LPAREN expressionList? RPAREN)? ...` - an empty
    // parameter list is legal and means the same as no parentheses at all.
    // The shared `identifier` prefix is parsed exactly once, with the empty
    // parens as an optional in the middle, rather than duplicated across two
    // alternatives: a std::string attribute is not cleared when Qi backtracks
    // out of a failed alternative, so the two-alternative form appended the
    // identifier a second time and reported `x() q[0];` as a call to gate
    // "xx". The optional is qi::omit-ed so the rule's attribute stays exactly
    // SimpleGatecallType's (std::string, MixedListType) pair.
    simpleGatecall %=
        identifier >> qi::omit[-(qi::lit('(') >> qi::lit(')'))] >> mixedList;
    // 'gphase' is the sole zero-qubit call (global phase, e.g. "gphase(pi);"),
    // so the trailing qubit list is optional here, defaulting to empty.
    expGatecall %= identifier >> '(' >> expList >> ')' >>
                   (mixedList | qi::attr(MixedListType()));

    gatecall %= simpleGatecall | expGatecall;

    // The trailing `!qi::lit(',')` on the two fixed-arity calls below stops
    // them from swallowing the first arguments of a longer list and leaving
    // the rest unparsable: `ctrl @ cx a, b, c` and `ctrl @ u(...) a, b` must
    // fall through to `gatecall`, since Qi does not re-enter an alternative
    // once a later element of the enclosing sequence fails. For unmodified
    // QASM2 those over-long calls were parse errors before and are now
    // reported by AddGateExpr as a qubit-count error instead.
    // Lowercase `u` and `cx` are retained QASM2 compatibility aliases for the
    // specification's uppercase `U` and `CX` builtins.
    ugateCall %= (qi::lit("U") | qi::lit("u")) >> '(' >> expList >> ')' >>
                 argument >> !qi::lit(',');
    cxgateCall %=
        qi::omit[qi::lexeme[(qi::lit("CX") | qi::lit("cx")) >> qi::space]] >>
        argument >> ',' >> argument >> !qi::lit(',');

    // The stdgates.inc-only gate names, rejected under a 2.0 header. The rule
    // sits at the head of `uop` rather than at statement level so that every
    // path to a gate call goes through it - a plain call, a modified call, a
    // conditioned one, and a call inside a gate declaration body all reduce
    // to `uop`. It is safe for a rule that may still backtrack to throw here
    // because both of its guards have already fired by then: the dialect is
    // QASM2, the name is one of the three (with an identifier-boundary
    // lookahead, so "phased" is not "phase"), and the name is not one the
    // program declared itself - that last case makes the rule fail instead,
    // and the call falls through to `gatecall` below. See
    // RejectQasm3OnlyGateExpr for why the filter lives here and not in the
    // allowed-gate sets.
    qasm2RejectedGate = qi::eps(!phx::ref(isQasm3)) >>
                        qi::lexeme[qasm3OnlyGates >> !qi::char_("a-zA-Z0-9_")]
                                  [qi::_pass = RejectQasm3OnlyGate(
                                       qi::_1, std::ref(definedGates))];

    uop %= qasm2RejectedGate | cxgateCall | ugateCall | gatecall;

    // QASM3 gate modifiers. These rules stay pure - no semantic action that
    // can throw - because they sit where the parse may still backtrack; the
    // only throwing action is AddModifiedGate in `qop`, which runs once the
    // choice is resolved. `modifiers` is zero-or-more, so `modifiedUop` with
    // an empty list is exactly the plain `uop` it replaced in `qop`.
    // `gateModifier: ... (CTRL | NEGCTRL) (LPAREN expression RPAREN)? AT` -
    // the control count is optional and defaults to 1, so `ctrl(2) @ x a, b,
    // c` is two controls. The count is folded into ModifierType::count rather
    // than expanded into repeated modifiers here, so that one parsed modifier
    // still maps to one ModifierType; AddModifiedGateExpr applies the
    // lowering `count` times, which is what routes `ctrl(2) @ x` into ccx and
    // `ctrl(3) @ x` into the existing multi-control error.
    ctrlMod = (qi::lit("ctrl") >> -('(' >> expression >> ')') >>
               '@')[qi::_val = MakeCtrlModifier(ModifierKind::Ctrl, qi::_1,
                                                std::ref(inputValues))];
    negctrlMod =
        (qi::lit("negctrl") >> -('(' >> expression >> ')') >>
         '@')[qi::_val = MakeCtrlModifier(ModifierKind::NegCtrl, qi::_1,
                                          std::ref(inputValues))];
    invMod %=
        qi::lit("inv") >> '@' >> qi::attr(ModifierType(ModifierKind::Inv));
    powMod = (qi::lit("pow") >> '(' >> expression >> ')' >>
              '@')[qi::_val = MakePowModifier(qi::_1, std::ref(inputValues))];

    modifier %=
        qi::eps(phx::ref(isQasm3)) >> (ctrlMod | negctrlMod | invMod | powMod);
    modifiers %= *modifier;
    modifiedUop %= modifiers >> uop;

    qop = (measureOp[qi::_val = AddMeasure(qi::_1, std::ref(creg_map),
                                           std::ref(qreg_map))] |
           measureAssignOp[qi::_val = AddMeasure(qi::_1, std::ref(creg_map),
                                                 std::ref(qreg_map))] |
           measureNoTarget[qi::_val = qi::_1] |
           resetOp[qi::_val = AddReset(qi::_1, std::ref(qreg_map))] |
           barrierOp[qi::_val = AddBarrier(qi::_1, std::ref(qreg_map))] |
           modifiedUop[qi::_val = AddModifiedGate(
                           qi::_1, std::ref(qreg_map), std::ref(opaqueGates),
                           std::ref(definedGates), std::ref(inputValues))]) >>
          ';';

    // **************************************************************************************************************************************************************

    qregdecl %= (qi::omit[qi::lexeme[qi::lit("qreg") >> qi::space]] >>
                 indexedId)[qi::_val = AddQreg(std::ref(qreg_counter),
                                               std::ref(qreg_map),
                                               std::ref(declarations), qi::_1)];
    cregdecl %= (qi::omit[qi::lexeme[qi::lit("creg") >> qi::space]] >>
                 indexedId)[qi::_val = AddCreg(std::ref(creg_counter),
                                               std::ref(creg_map),
                                               std::ref(declarations), qi::_1)];

    // QASM3 register declarations: the size comes before the name, so we
    // reuse MakeIndexedId with swapped placeholders instead of a new functor.
    // Bare (size-1) forms are supported via qi::attr(1) standing in for the
    // missing size. The sized form is tried first so it wins over the bare
    // form on the shared "qubit"/"bit" prefix. Each alternative carries its
    // own action (rather than one action on the alternative as a whole) so
    // that qi::_1/qi::_2 are split from that alternative's own sequence
    // attribute, matching the `expression` rule's convention below.
    qubitdecl =
        qi::eps(phx::ref(isQasm3)) >>
        ((qi::lit("qubit") >> '[' >> qi::int_ >> ']' >>
          identifier)[qi::_val =
                          AddQreg(std::ref(qreg_counter), std::ref(qreg_map),
                                  std::ref(declarations),
                                  MakeIndexedId(qi::_2, qi::_1))] |
         (qi::omit[qi::lexeme[qi::lit("qubit") >> qi::space]] >> qi::attr(1) >>
          identifier)[qi::_val =
                          AddQreg(std::ref(qreg_counter), std::ref(qreg_map),
                                  std::ref(declarations),
                                  MakeIndexedId(qi::_2, qi::_1))]);
    bitdecl =
        qi::eps(phx::ref(isQasm3)) >>
        ((qi::lit("bit") >> '[' >> qi::int_ >> ']' >>
          identifier)[qi::_val =
                          AddCreg(std::ref(creg_counter), std::ref(creg_map),
                                  std::ref(declarations),
                                  MakeIndexedId(qi::_2, qi::_1))] |
         (qi::omit[qi::lexeme[qi::lit("bit") >> qi::space]] >> qi::attr(1) >>
          identifier)[qi::_val =
                          AddCreg(std::ref(creg_counter), std::ref(creg_map),
                                  std::ref(declarations),
                                  MakeIndexedId(qi::_2, qi::_1))]);

    decl %= (qregdecl | cregdecl | qubitdecl | bitdecl) >> ';';

    // QASM3 free-parameter declaration: `input <type> <name>;`. Must be tried
    // before `qop` in `statement`, or gatecall's identifier would match
    // "input" as a gate name. The type keyword and its optional bit-width
    // qualifier ("float[64]", "int[32]", "uint", "bool", "angle", ...) are
    // retained so AddInputDecl can validate and normalize the API's double
    // carrier before publishing the value to expression evaluation.
    // The type keyword's `!qi::char_(...)` lookahead is wrapped in its own
    // qi::lexeme, like "input"'s own keyword guard above, so a name that
    // merely starts with a type keyword (e.g. "intx") cannot be mistaken for
    // the keyword itself - the lookahead must run before the skipper can eat
    // any whitespace, or it would never see the very next character.
    inputType %= qi::lexeme[(qi::string("float") | qi::string("int") |
                             qi::string("uint") | qi::string("bool") |
                             qi::string("angle")) >>
                            !qi::char_("a-zA-Z0-9_")];
    inputdecl %= qi::eps(phx::ref(isQasm3)) >>
                 qi::omit[qi::lexeme[qi::lit("input") >> qi::space]] >>
                 inputType >> -('[' >> qi::int_ >> ']') >> identifier >> ';';

    // QASM3 constructs outside our supported subset. Tried before `qop` in
    // `statement`, or gatecall's identifier would swallow the keyword as a
    // gate name and report the generic "Unsupported gate" error instead of
    // naming the actual construct. Guard A: gated on isQasm3, since these
    // words are ordinary identifiers in QASM2 - "for" is a fine register or
    // gate name there. Guard B: the keyword-boundary lookahead sits inside
    // the same qi::lexeme as the symbol lookup, following the `inputdecl`
    // idiom above, so a name that merely starts with a reserved word (e.g.
    // "format", "delayed") is not mistaken for the keyword itself. Throwing
    // here is only safe because of this placement plus both guards - see
    // RejectUnsupportedConstructExpr.
    unsupportedConstruct =
        qi::eps(phx::ref(isQasm3)) >>
        qi::lexeme[unsupportedKeywords >> !qi::char_("a-zA-Z0-9_")]
                  [qi::_val = RejectUnsupportedConstruct(qi::_1)];

    measureOp %= qi::omit[qi::lexeme[qi::lit("measure") >> qi::space]] >>
                 argument >> qi::lit("->") >> argument;
    // QASM3 assignment-style measurement: `c[0] = measure q[0];`. The
    // classical target is parsed first, so the (cbits, qubits) pair is
    // swapped in-place via phx::construct into the same MeasureType consumed
    // by AddMeasure above, avoiding a duplicate of AddMeasureExpr's
    // indexed/whole-register handling.
    measureAssignOp =
        qi::eps(phx::ref(isQasm3)) >>
        (argument >> '=' >>
         qi::omit[qi::lexeme[qi::lit("measure") >> qi::space]] >>
         argument)[qi::_val = phx::construct<MeasureType>(qi::_2, qi::_1)];
    // `measureArrowAssignmentStatement: measureExpression (ARROW
    // indexedIdentifier)? SEMICOLON` - the arrow target is optional, i.e.
    // `measure q;` means "measure and discard". This IR has no such
    // operation: Circuits::CircuitFactory::CreateMeasurement takes
    // (qubit, classical bit) pairs and every measurement writes a bit. Rather
    // than invent a classical bit to hold a result the program never asked
    // for, the form is recognised and rejected by name - the point of the
    // rule is that the diagnostic says "measurement without a classical
    // target" instead of the misleading "Unsupported gate without parameters:
    // measure" the fall-through to `gatecall` used to produce. Tried after
    // both real measurement rules, so it only ever sees a genuinely
    // target-less measurement.
    measureNoTarget = (qi::omit[qi::lexeme[qi::lit("measure") >> qi::space]] >>
                       argument)[qi::_val = RejectMeasureWithoutTarget(qi::_1)];
    resetOp %= qi::omit[qi::lexeme[qi::lit("reset") >> qi::space]] >> argument;
    // `barrierStatement: BARRIER gateOperandList? SEMICOLON` - the operand
    // list is optional and a bare `barrier;` applies to every qubit. The
    // empty MixedListType stands for exactly that, and AddBarrierExpr expands
    // it over the whole qreg map; `mixedList` matches one operand at minimum,
    // so an empty list cannot arrive from the first alternative. The
    // identifier-boundary lookahead keeps a gate named e.g. "barriers" from
    // matching the bare form's keyword prefix.
    //
    // QASM3 only: QASM2's grammar is `statement: "barrier" anylist ";"`, with
    // the operand list required, so the bare alternative is gated on isQasm3
    // and `barrier;` under a 2.0 header is the syntax error it is there. The
    // operand form is shared by both dialects and stays ungated.
    barrierOp %=
        (qi::omit[qi::lexeme[qi::lit("barrier") >> qi::space]] >> mixedList) |
        (qi::eps(phx::ref(isQasm3)) >>
         qi::omit[qi::lexeme[qi::lit("barrier") >> !qi::char_("a-zA-Z0-9_")]] >>
         qi::attr(MixedListType()));

    // **************************************************************************************************************************************************************

    idList %= identifier % ',';

    indexedId = (identifier >> '[' >> qi::int_ >>
                 ']')[qi::_val = MakeIndexedId(qi::_1, qi::_2)];

    argument %= indexedId | identifier;
    mixedList %= argument % ',';

    // **************************************************************************************************************************************************************
    // expressions

    expList %= expression % ',';

    // '^' is exponentiation in QASM2 but bitwise XOR in QASM3, which also
    // introduces '**' for exponentiation:
    //   expression: <assoc=right> expression DOUBLE_ASTERISK expression
    //             | expression CARET expression
    // The two spellings do not merely differ in name, they sit at opposite
    // ends of the precedence table - '**' binds tighter than '*', while XOR
    // binds looser than '+' - so they cannot share a rule. Hence:
    //   - `expression` (this level, loosest) carries the QASM3-only XOR;
    //   - `additive` is the former top of the chain, unchanged;
    //   - `factor2` (tightest binary level) carries the QASM3-only '**' and
    //     '^'-as-power only in QASM2.
    // Every one of the three rules is guarded on isQasm3, so QASM2 semantics
    // are exactly what QASM2 says they are: `2 ^ 3` is 8 there and 1 (2 XOR 3)
    // here, and '**' - which QASM2's `exp` production does not have at all -
    // is a syntax error under a 2.0 header.
    expression = (qi::eps(phx::ref(isQasm3)) >> additive >> qi::lit('^') >>
                  expression)[qi::_val = MakeBinary('X', qi::_1, qi::_2)] |
                 additive[qi::_val = qi::_1];

    // Left-associative, as '-' and '/' are in every dialect of QASM and in
    // ordinary arithmetic: `1 - 2 - 3` is (1 - 2) - 3 = -4 and `8 / 4 / 2` is
    // (8 / 4) / 2 = 1. Both rules used to recurse into themselves on the
    // right, which made them right-associative and silently produced 2 and 4
    // instead - a wrong gate angle with no error at all. The fix is the
    // standard Spirit left fold: parse one operand into qi::_val, then
    // accumulate each following (operator, operand) pair onto it, so the left
    // operand of each new node is the whole accumulated left-hand side rather
    // than the first operand alone. '+' and '*' are associative and so
    // numerically unaffected, but all four operators go through the same fold
    // to keep one shape per precedence level.
    additive = product[qi::_val = qi::_1] >>
               *(qi::char_("+-") >>
                 product)[qi::_val = MakeBinary(qi::_1, qi::_val, qi::_2)];
    product = unary[qi::_val = qi::_1] >>
              *(qi::char_("*/") >>
                unary)[qi::_val = MakeBinary(qi::_1, qi::_val, qi::_2)];

    // The official grammar orders parenthesis > index > '**' (right-assoc) >
    // unary > '* / %' > '+ -', i.e. a leading sign binds *looser* than '**':
    // `-2 ** 2` is -(2 ** 2) = -4 and `2 ** -3 ** 2` is 2 ** -(3 ** 2). So
    // `unary` - not `factor2` - is the operand of '*' and '/' above, and it
    // applies the sign to the result of a whole power expression. It recurses
    // into itself so repeated signs (`--2`) still work, and so the operand of
    // a sign is itself allowed to be a power.
    unary = (qi::char_("+-") >> unary)[qi::_val = MakeUnary(qi::_1, qi::_2)] |
            factor2[qi::_val = qi::_1];

    // Right-associative, per DOUBLE_ASTERISK's <assoc=right> above: `2 ** 3 **
    // 2` is 2 ** 9 = 512. The right operand is `unary` rather than `factor2`
    // so that both the right-associativity and the spec's `2 ** -3` are
    // expressible; the left operand is `factor`, which carries no sign of its
    // own - a sign there would have to have been consumed by `unary` first,
    // one precedence level out. '**' is a QASM3 powerExpression and has no
    // counterpart in QASM2's `exp` production, so it is gated on isQasm3; the
    // '^'-as-power alternative below is its exact complement, which is why
    // the two can be ordered either way without `2 ** 3` ever being read as
    // `2 ^ (* 3)`.
    factor2 = (qi::eps(phx::ref(isQasm3)) >> factor >> qi::lit("**") >>
               unary)[qi::_val = MakeBinary('^', qi::_1, qi::_2)] |
              (qi::eps(!phx::ref(isQasm3)) >> factor >> qi::lit('^') >>
               unary)[qi::_val = MakeBinary('^', qi::_1, qi::_2)] |
              factor[qi::_val = qi::_1];
    factor = group[qi::_val = qi::_1] | constant[qi::_val = qi::_1] |
             (funcName >> group)[qi::_val = MakeFunction(qi::_1, qi::_2)] |
             identifier[qi::_val = MakeVariable(qi::_1)];
    // Unsigned on purpose. qi::double_ (and qi::int_) consume a leading sign
    // themselves, so a signed number parsed here would attach the sign to the
    // *base* of a power - `-2 ** 2` would be (-2) ** 2 = 4 - defeating the
    // precedence the `unary` rule above establishes. Every sign is the
    // `unary` rule's business; this parser only ever sees the digits. The
    // former qi::int_ alternative is gone with it: it was unreachable (the
    // real parser already matches an integer literal) and would have been
    // another way for a sign to slip in below `unary`.
    constant = qi::real_parser<double, qi::ureal_policies<double>>()
                   [qi::_val = MakeConstant(qi::_1)] |
               pi[qi::_val = MakeConstant(qi::_1)];
    group %= '(' >> expression >> ')';

    funcName %= qi::string("sin") | qi::string("cos") | qi::string("tan") |
                qi::string("exp") | qi::string("ln") | qi::string("sqrt");
    pi %= qi::lit("pi")[qi::_val = M_PI];

    // **************************************************************************************************************************************************************

    // very basic stuff
    comment %= qi::lexeme[qi::lit("//") >> *(qi::char_ - qi::eol) >> -qi::eol];
    quoted_string %= qi::lexeme['"' >> +(qi::char_ - '"') >> '"'];
    // Includes are validated prelude markers only; this parser never loads an
    // external file, and the program rule permits markers only at the start.
    include = (qi::omit[qi::lexeme[qi::lit("include") >> qi::space]] >>
               quoted_string >> ';')[qi::_val = ValidateInclude(qi::_1)];
    // The two dialects differ on the *first* character of an identifier and
    // nowhere else. QASM2's lexical rule is `id := [a-z][A-Za-z0-9_]*`, i.e.
    // the initial character must be a lowercase letter; QASM3's is
    // `Identifier: [A-Za-z_][A-Za-z0-9_]*`, which also admits an uppercase
    // letter or an underscore. So the first character is an alternation gated
    // on isQasm3 while the trailing class stays shared. The whole thing
    // remains a single qi::lexeme - the dialect choice is spelled inline
    // rather than delegated to a sub-rule precisely so that no skipper can
    // run between the first character and the rest, which would let
    // `_ q` parse as the identifier "_q" under QASM3.
    //
    // QASM2's genuinely uppercase builtins are unaffected: `U` and `CX` are
    // keywords there, matched by qi::lit in `ugateCall`/`cxgateCall`, not by
    // this rule.
    identifier %=
        qi::lexeme[((qi::eps(phx::ref(isQasm3)) >> qi::char_("a-zA-Z_")) |
                    qi::char_("a-z")) >>
                   *qi::char_("a-zA-Z0-9_")];

    // Debugging and error handling and reporting support.
    BOOST_SPIRIT_DEBUG_NODE(version);
    BOOST_SPIRIT_DEBUG_NODE(program);
    BOOST_SPIRIT_DEBUG_NODE(statement);
    BOOST_SPIRIT_DEBUG_NODE(statements);

    BOOST_SPIRIT_DEBUG_NODE(opaque);

    BOOST_SPIRIT_DEBUG_NODE(gatedecl);
    BOOST_SPIRIT_DEBUG_NODE(simplebarrier);
    BOOST_SPIRIT_DEBUG_NODE(gateBodyModifier);
    BOOST_SPIRIT_DEBUG_NODE(gatedeclop);
    BOOST_SPIRIT_DEBUG_NODE(gatedeclfull);

    BOOST_SPIRIT_DEBUG_NODE(condOp);
    BOOST_SPIRIT_DEBUG_NODE(condBitTest);
    BOOST_SPIRIT_DEBUG_NODE(condHead);
    BOOST_SPIRIT_DEBUG_NODE(condOpBraced);
    BOOST_SPIRIT_DEBUG_NODE(simpleGatecall);
    BOOST_SPIRIT_DEBUG_NODE(expGatecall);
    BOOST_SPIRIT_DEBUG_NODE(gatecall);
    BOOST_SPIRIT_DEBUG_NODE(qasm2RejectedGate);
    BOOST_SPIRIT_DEBUG_NODE(ugateCall);
    BOOST_SPIRIT_DEBUG_NODE(cxgateCall);
    BOOST_SPIRIT_DEBUG_NODE(uop);
    BOOST_SPIRIT_DEBUG_NODE(ctrlMod);
    BOOST_SPIRIT_DEBUG_NODE(negctrlMod);
    BOOST_SPIRIT_DEBUG_NODE(invMod);
    BOOST_SPIRIT_DEBUG_NODE(powMod);
    BOOST_SPIRIT_DEBUG_NODE(modifier);
    BOOST_SPIRIT_DEBUG_NODE(modifiers);
    BOOST_SPIRIT_DEBUG_NODE(modifiedUop);
    BOOST_SPIRIT_DEBUG_NODE(qop);

    BOOST_SPIRIT_DEBUG_NODE(qregdecl);
    BOOST_SPIRIT_DEBUG_NODE(cregdecl);
    BOOST_SPIRIT_DEBUG_NODE(qubitdecl);
    BOOST_SPIRIT_DEBUG_NODE(bitdecl);
    BOOST_SPIRIT_DEBUG_NODE(decl);
    BOOST_SPIRIT_DEBUG_NODE(inputdecl);
    BOOST_SPIRIT_DEBUG_NODE(unsupportedConstruct);
    BOOST_SPIRIT_DEBUG_NODE(resetOp);
    BOOST_SPIRIT_DEBUG_NODE(measureOp);
    BOOST_SPIRIT_DEBUG_NODE(measureAssignOp);
    BOOST_SPIRIT_DEBUG_NODE(measureNoTarget);
    BOOST_SPIRIT_DEBUG_NODE(barrierOp);

    BOOST_SPIRIT_DEBUG_NODE(idList);
    BOOST_SPIRIT_DEBUG_NODE(indexedId);
    BOOST_SPIRIT_DEBUG_NODE(argument);
    BOOST_SPIRIT_DEBUG_NODE(mixedList);

    BOOST_SPIRIT_DEBUG_NODE(expList);

    BOOST_SPIRIT_DEBUG_NODE(expression);
    BOOST_SPIRIT_DEBUG_NODE(additive);
    BOOST_SPIRIT_DEBUG_NODE(product);
    BOOST_SPIRIT_DEBUG_NODE(factor2);
    BOOST_SPIRIT_DEBUG_NODE(unary);
    BOOST_SPIRIT_DEBUG_NODE(factor);
    BOOST_SPIRIT_DEBUG_NODE(constant);
    BOOST_SPIRIT_DEBUG_NODE(group);
    BOOST_SPIRIT_DEBUG_NODE(funcName);
    BOOST_SPIRIT_DEBUG_NODE(pi);

    BOOST_SPIRIT_DEBUG_NODE(comment);
    BOOST_SPIRIT_DEBUG_NODE(quoted_string);
    BOOST_SPIRIT_DEBUG_NODE(include);
    BOOST_SPIRIT_DEBUG_NODE(identifier);

    // Error handling
    qi::on_error<qi::fail>(expression, error_handler(qi::_4, qi::_3, qi::_2));
    // TODO: add more error handlers if needed
    qi::on_error<qi::fail>(program, error_handler(qi::_4, qi::_3, qi::_2));
  }

  void clear() {
    creg_counter = 0;
    qreg_counter = 0;
    creg_map.clear();
    qreg_map.clear();
    declarations.clear();
    opaqueGates.clear();
    definedGates.clear();
    inputNames.clear();
    inputBindings.clear();
    inputValues.clear();
    isQasm3 = false;
  }

  qi::rule<Iterator, Program(), Skipper> program;

  qi::rule<Iterator, double(), Skipper> version;

  qi::rule<Iterator, StatementType, Skipper> statement;
  qi::rule<Iterator, std::vector<StatementType>(), Skipper> statements;

  qi::rule<Iterator, OpaqueDeclType(), Skipper> opaque;

  qi::rule<Iterator, GateDeclType(), Skipper> gatedecl;
  qi::rule<Iterator, SimpleBarrierType(), Skipper> simplebarrier;
  qi::rule<Iterator, UopType(), Skipper> gateBodyModifier;
  qi::rule<Iterator, GateDeclOpType(), Skipper> gatedeclop;
  qi::rule<Iterator,
           boost::fusion::vector<GateDeclType, std::vector<GateDeclOpType>>(),
           Skipper>
      gatedeclfull;

  qi::rule<Iterator, CondOpType(), Skipper> condOp;
  qi::rule<Iterator, CondBitTest(), Skipper> condBitTest;
  qi::rule<Iterator, CondHeadType(), Skipper> condHead;
  qi::rule<Iterator, std::vector<StatementType>(), Skipper> condOpBraced;

  qi::rule<Iterator, UGateCallType, Skipper> ugateCall;
  qi::rule<Iterator, CXGateCallType, Skipper> cxgateCall;

  qi::rule<Iterator, SimpleGatecallType(), Skipper> simpleGatecall;
  qi::rule<Iterator, ExpGatecallType(), Skipper> expGatecall;
  qi::rule<Iterator, GatecallType(), Skipper> gatecall;
  qi::rule<Iterator, UopType(), Skipper> qasm2RejectedGate;
  qi::rule<Iterator, UopType(), Skipper> uop;

  qi::rule<Iterator, ModifierType(), Skipper> ctrlMod;
  qi::rule<Iterator, ModifierType(), Skipper> negctrlMod;
  qi::rule<Iterator, ModifierType(), Skipper> invMod;
  qi::rule<Iterator, ModifierType(), Skipper> powMod;
  qi::rule<Iterator, ModifierType(), Skipper> modifier;
  qi::rule<Iterator, ModifierListType(), Skipper> modifiers;
  qi::rule<Iterator, ModifiedUopType(), Skipper> modifiedUop;

  qi::rule<Iterator, QopType(), Skipper> qop;

  qi::rule<Iterator, IndexedId(), Skipper> qregdecl;
  qi::rule<Iterator, IndexedId(), Skipper> cregdecl;
  qi::rule<Iterator, IndexedId(), Skipper> qubitdecl;
  qi::rule<Iterator, IndexedId(), Skipper> bitdecl;
  qi::rule<Iterator, IndexedId(), Skipper> decl;

  qi::rule<Iterator, std::string(), Skipper> inputType;
  qi::rule<Iterator, InputDeclType(), Skipper> inputdecl;

  qi::rule<Iterator, StatementType, Skipper> unsupportedConstruct;

  qi::rule<Iterator, ResetType(), Skipper> resetOp;
  qi::rule<Iterator, MeasureType(), Skipper> measureOp;
  qi::rule<Iterator, MeasureType(), Skipper> measureAssignOp;
  qi::rule<Iterator, QopType(), Skipper> measureNoTarget;
  qi::rule<Iterator, BarrierType(), Skipper> barrierOp;

  qi::rule<Iterator, std::vector<std::string>(), Skipper> idList;

  qi::rule<Iterator, IndexedId(), Skipper> indexedId;

  qi::rule<Iterator, ArgumentType(), Skipper> argument;
  qi::rule<Iterator, MixedListType(), Skipper> mixedList;

  qi::rule<Iterator, std::vector<Expression>(), Skipper> expList;

  // `unary` is an Expression like every other level: it is the whole
  // "optionally signed power expression" level, not just the signed form, so
  // it must be able to carry through the unsigned case as well.
  qi::rule<Iterator, Expression(), Skipper> expression, additive, group,
      product, factor, factor2, unary;
  qi::rule<Iterator, Constant(), Skipper> constant;

  qi::rule<Iterator, std::string(), Skipper> funcName;
  qi::rule<Iterator, std::string(), Skipper> comment;
  qi::rule<Iterator, std::vector<std::string>(), Skipper> comments;
  qi::rule<Iterator, std::string(), Skipper> include;
  qi::rule<Iterator, std::vector<std::string>(), Skipper> includes;
  qi::rule<Iterator, std::string(), Skipper> quoted_string;
  qi::rule<Iterator, std::string(), Skipper> identifier;
  qi::rule<Iterator, double(), Skipper> pi;

  // Keyword -> diagnostic message table backing `unsupportedConstruct`.
  qi::symbols<char, std::string> unsupportedKeywords;

  // The stdgates.inc-only gate names backing `qasm2RejectedGate`; each maps
  // to its own spelling.
  qi::symbols<char, std::string> qasm3OnlyGates;

  int creg_counter = 0;
  int qreg_counter = 0;

  bool isQasm3 = false;

  std::unordered_map<std::string, IndexedId> creg_map;
  std::unordered_map<std::string, IndexedId> qreg_map;
  DeclarationRegistry declarations;

  std::unordered_map<std::string, StatementType> opaqueGates;
  std::unordered_map<std::string, StatementType> definedGates;

  // Raw caller bindings are separate from values made visible by declarations,
  // so QASM2, undeclared names, and forward references cannot consume them.
  std::vector<std::string> inputNames;
  std::unordered_map<std::string, double> inputBindings;
  std::unordered_map<std::string, double> inputValues;

  void ValidateInputBindings() const {
    for (const auto &[name, value] : inputBindings) {
      (void)value;
      if (std::find(inputNames.begin(), inputNames.end(), name) ==
          inputNames.end())
        throw std::invalid_argument("No input declaration for binding '" +
                                    name + "'.");
    }
  }
};

}  // namespace qasm

#endif  // !_QASM_H_
