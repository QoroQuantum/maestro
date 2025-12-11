/**
 * @file qasm.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Classes for the qasm parser and interpreter.
 *
 * Not fully implemented yet, it's supposed to support only open qasm 2.0.
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

phx::function<error_handler_> const error_handler = error_handler_();
}  // namespace qasm

BOOST_FUSION_ADAPT_STRUCT(qasm::Program,
                          (std::vector<std::string>, comments)(double, version)(
                              std::vector<std::string>,
                              includes)(std::vector<qasm::StatementType>,
                                        statements))

namespace qasm {

void printd(const double &v) { std::cout << "version: " << v << "\n"; }

void prints(const std::string &s) { std::cout << "statement: " << s << "\n"; }

// TODO:
// 1. 'opaque' will be parsed but ignored in the first phase.
// 2. 'barrier' will be parsed, but ignored in the first phase. In this case we
// might want to add a 'barrier' operation in our circuit. For now it's not
// existent. Adding it would have implications in circuit execution with the
// discrete event simulator and also in the transpiler functionality.

template <typename Iterator = std::string::iterator,
          typename Skipper = ascii::space_type>
struct QasmGrammar : qi::grammar<Iterator, Program(), Skipper> {
  QasmGrammar() : QasmGrammar::base_type{program} {
    version = (qi::omit[qi::lexeme[qi::lit("OPENQASM") >> qi::space]] >>
               qi::double_ >> ';')[qi::_val = qi::_1];

    comments %= *comment;
    includes %= *include;

    program = comments >> (-version) >> includes >> statements;

    statements %= *statement;

    statement =
        comment[qi::_val = AddComment(qi::_1)] |
        decl[qi::_val = AddDeclaration(qi::_1)] |
        opaque[qi::_val = AddOpaqueDecl(qi::_1, std::ref(opaqueGates),
                                        std::ref(qreg_map))] |
        condOp[qi::_val =
                   AddCondQop(qi::_1, std::ref(qreg_map), std::ref(creg_map),
                              std::ref(opaqueGates), std::ref(definedGates))] |
        gatedeclfull[qi::_val = AddGateDecl(qi::_1, std::ref(definedGates))] |
        qop[qi::_val = qi::_1];

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
    gatedeclop %= simplebarrier | (uop >> ';');

    gatedeclfull %= gatedecl >> *gatedeclop >> '}';

    // **************************************************************************************************************************************************************

    condOp %= qi::lit("if") >> '(' >> identifier >> qi::lit("==") >> qi::int_ >>
              ')' >> qop;

    simpleGatecall %=
        (identifier >> mixedList) | (identifier >> '(' >> ')' >> mixedList);
    expGatecall %= identifier >> '(' >> expList >> ')' >> mixedList;

    gatecall %= simpleGatecall | expGatecall;

    ugateCall %=
        (qi::lit("U") | qi::lit("u")) >> '(' >> expList >> ')' >> argument;
    cxgateCall %=
        qi::omit[qi::lexeme[(qi::lit("CX") | qi::lit("cx")) >> qi::space]] >>
        argument >> ',' >> argument;

    uop %= cxgateCall | ugateCall | gatecall;

    qop = (measureOp[qi::_val = AddMeasure(qi::_1, std::ref(creg_map),
                                           std::ref(qreg_map))] |
           resetOp[qi::_val = AddReset(qi::_1, std::ref(qreg_map))] |
           barrierOp[qi::_val = AddBarrier(qi::_1, std::ref(qreg_map))] |
           uop[qi::_val =
                   AddGate(qi::_1, std::ref(qreg_map), std::ref(opaqueGates),
                           std::ref(definedGates))]) >>
          ';';

    // **************************************************************************************************************************************************************

    qregdecl %= (qi::omit[qi::lexeme[qi::lit("qreg") >> qi::space]] >>
                 indexedId)[qi::_val = AddQreg(std::ref(qreg_counter),
                                               std::ref(qreg_map), qi::_1)];
    cregdecl %= (qi::omit[qi::lexeme[qi::lit("creg") >> qi::space]] >>
                 indexedId)[qi::_val = AddCreg(std::ref(creg_counter),
                                               std::ref(creg_map), qi::_1)];

    decl %= (qregdecl | cregdecl) >> ';';

    measureOp %= qi::omit[qi::lexeme[qi::lit("measure") >> qi::space]] >>
                 argument >> qi::lit("->") >> argument;
    resetOp %= qi::omit[qi::lexeme[qi::lit("reset") >> qi::space]] >> argument;
    barrierOp %=
        qi::omit[qi::lexeme[qi::lit("barrier") >> qi::space]] >> mixedList;

    // **************************************************************************************************************************************************************

    idList %= identifier % ',';

    indexedId = (identifier >> '[' >> qi::int_ >>
                 ']')[qi::_val = MakeIndexedId(qi::_1, qi::_2)];

    argument %= indexedId | identifier;
    mixedList %= argument % ',';

    // **************************************************************************************************************************************************************
    // expressions

    expList %= expression % ',';

    expression = (product >> qi::char_("+-") >>
                  expression)[qi::_val = MakeBinary(qi::_2, qi::_1, qi::_3)] |
                 product[qi::_val = qi::_1];
    product = (factor2 >> qi::char_("*/") >>
               product)[qi::_val = MakeBinary(qi::_2, qi::_1, qi::_3)] |
              factor2[qi::_val = qi::_1];

    factor2 =
        (factor >> '^' >> factor2)[qi::_val = MakeBinary('^', qi::_1, qi::_2)] |
        factor[qi::_val = qi::_1];
    unary = (qi::char_("+-") >> factor)[qi::_val = MakeUnary(qi::_1, qi::_2)];
    factor = group[qi::_val = qi::_1] | constant[qi::_val = qi::_1] |
             unary[qi::_val = qi::_1] |
             (funcName >> group)[qi::_val = MakeFunction(qi::_1, qi::_2)] |
             identifier[qi::_val = MakeVariable(qi::_1)];
    constant = qi::double_[qi::_val = MakeConstant(qi::_1)] |
               qi::int_[qi::_val = MakeConstant(qi::_1)] |
               pi[qi::_val = MakeConstant(qi::_1)];
    group %= '(' >> expression >> ')';

    funcName %= qi::string("sin") | qi::string("cos") | qi::string("tan") |
                qi::string("exp") | qi::string("ln") | qi::string("sqrt");
    pi %= qi::lit("pi")[qi::_val = M_PI];

    // **************************************************************************************************************************************************************

    // very basic stuff
    comment %= qi::lexeme[qi::lit("//") >> *(qi::char_ - qi::eol) >> qi::eol];
    quoted_string %= qi::lexeme['"' >> +(qi::char_ - '"') >> '"'];
    include %= qi::omit[qi::lexeme[qi::lit("include") >> qi::space]] >>
               quoted_string >> ';';
    identifier %= (qi::lexeme[qi::char_("a-z") >> *qi::char_("a-zA-Z0-9_")]);

    // Debugging and error handling and reporting support.
    BOOST_SPIRIT_DEBUG_NODE(version);
    BOOST_SPIRIT_DEBUG_NODE(program);
    BOOST_SPIRIT_DEBUG_NODE(statement);
    BOOST_SPIRIT_DEBUG_NODE(statements);

    BOOST_SPIRIT_DEBUG_NODE(opaque);

    BOOST_SPIRIT_DEBUG_NODE(gatedecl);
    BOOST_SPIRIT_DEBUG_NODE(simplebarrier);
    BOOST_SPIRIT_DEBUG_NODE(gatedeclop);
    BOOST_SPIRIT_DEBUG_NODE(gatedeclfull);

    BOOST_SPIRIT_DEBUG_NODE(condOp);
    BOOST_SPIRIT_DEBUG_NODE(simpleGatecall);
    BOOST_SPIRIT_DEBUG_NODE(expGatecall);
    BOOST_SPIRIT_DEBUG_NODE(gatecall);
    BOOST_SPIRIT_DEBUG_NODE(ugateCall);
    BOOST_SPIRIT_DEBUG_NODE(cxgateCall);
    BOOST_SPIRIT_DEBUG_NODE(uop);
    BOOST_SPIRIT_DEBUG_NODE(qop);

    BOOST_SPIRIT_DEBUG_NODE(qregdecl);
    BOOST_SPIRIT_DEBUG_NODE(cregdecl);
    BOOST_SPIRIT_DEBUG_NODE(decl);
    BOOST_SPIRIT_DEBUG_NODE(resetOp);
    BOOST_SPIRIT_DEBUG_NODE(measureOp);
    BOOST_SPIRIT_DEBUG_NODE(barrierOp);

    BOOST_SPIRIT_DEBUG_NODE(idList);
    BOOST_SPIRIT_DEBUG_NODE(indexedId);
    BOOST_SPIRIT_DEBUG_NODE(argument);
    BOOST_SPIRIT_DEBUG_NODE(mixedList);

    BOOST_SPIRIT_DEBUG_NODE(expList);

    BOOST_SPIRIT_DEBUG_NODE(expression);
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
    opaqueGates.clear();
    definedGates.clear();
  }

  qi::rule<Iterator, Program(), Skipper> program;

  qi::rule<Iterator, double(), Skipper> version;

  qi::rule<Iterator, StatementType, Skipper> statement;
  qi::rule<Iterator, std::vector<StatementType>(), Skipper> statements;

  qi::rule<Iterator, OpaqueDeclType(), Skipper> opaque;

  qi::rule<Iterator, GateDeclType(), Skipper> gatedecl;
  qi::rule<Iterator, SimpleBarrierType(), Skipper> simplebarrier;
  qi::rule<Iterator, GateDeclOpType(), Skipper> gatedeclop;
  qi::rule<Iterator,
           boost::fusion::vector<GateDeclType, std::vector<GateDeclOpType>>(),
           Skipper>
      gatedeclfull;

  qi::rule<Iterator, CondOpType(), Skipper> condOp;

  qi::rule<Iterator, UGateCallType, Skipper> ugateCall;
  qi::rule<Iterator, CXGateCallType, Skipper> cxgateCall;

  qi::rule<Iterator, SimpleGatecallType(), Skipper> simpleGatecall;
  qi::rule<Iterator, ExpGatecallType(), Skipper> expGatecall;
  qi::rule<Iterator, GatecallType(), Skipper> gatecall;
  qi::rule<Iterator, UopType(), Skipper> uop;
  qi::rule<Iterator, QopType(), Skipper> qop;

  qi::rule<Iterator, IndexedId(), Skipper> qregdecl;
  qi::rule<Iterator, IndexedId(), Skipper> cregdecl;
  qi::rule<Iterator, IndexedId(), Skipper> decl;

  qi::rule<Iterator, ResetType(), Skipper> resetOp;
  qi::rule<Iterator, MeasureType(), Skipper> measureOp;
  qi::rule<Iterator, BarrierType(), Skipper> barrierOp;

  qi::rule<Iterator, std::vector<std::string>(), Skipper> idList;

  qi::rule<Iterator, IndexedId(), Skipper> indexedId;

  qi::rule<Iterator, ArgumentType(), Skipper> argument;
  qi::rule<Iterator, MixedListType(), Skipper> mixedList;

  qi::rule<Iterator, std::vector<Expression>(), Skipper> expList;

  qi::rule<Iterator, Expression(), Skipper> expression, group, product, factor,
      factor2;
  qi::rule<Iterator, UnaryOperator(), Skipper> unary;
  qi::rule<Iterator, Constant(), Skipper> constant;

  qi::rule<Iterator, std::string(), Skipper> funcName;
  qi::rule<Iterator, std::string(), Skipper> comment;
  qi::rule<Iterator, std::vector<std::string>(), Skipper> comments;
  qi::rule<Iterator, std::string(), Skipper> include;
  qi::rule<Iterator, std::vector<std::string>(), Skipper> includes;
  qi::rule<Iterator, std::string(), Skipper> quoted_string;
  qi::rule<Iterator, std::string(), Skipper> identifier;
  qi::rule<Iterator, double(), Skipper> pi;

  int creg_counter = 0;
  int qreg_counter = 0;

  std::unordered_map<std::string, IndexedId> creg_map;
  std::unordered_map<std::string, IndexedId> qreg_map;

  std::unordered_map<std::string, StatementType> opaqueGates;
  std::unordered_map<std::string, StatementType> definedGates;
};

}  // namespace qasm

#endif  // !_QASM_H_
