/**
 * @file QasmCirc.h
 * @ingroup qasm
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Class for parsing qasm and translating it to a circuit.
 *
 * Supports the static-circuit subset of OpenQASM 2 and OpenQASM 3.
 */
#pragma once

#ifndef _QASMCIRC_H_
#define _QASMCIRC_H_

#include "qasm.h"

namespace qasm {
template <typename Time = Types::time_type>
class QasmToCirc {
 public:
  void clear() {
    grammar.clear();
    program.clear();
    errorMessage.clear();
    error = false;
  }

  // QASM3 `input` values are parse-time substitutions represented as doubles;
  // no runtime classical state or symbolic value enters the Circuit IR.
  std::shared_ptr<Circuits::Circuit<Time>> ParseAndTranslateWithParams(
      const std::string& qasmInputStr,
      const std::unordered_map<std::string, double>& params = {}) {
    clear();

    grammar.inputBindings = params;

    std::string qasmInput = qasmInputStr;

    return ParseAndTranslateImpl(qasmInput);
  }

  std::shared_ptr<Circuits::Circuit<Time>> ParseAndTranslate(
      const std::string &qasmInputStr) {
    clear();

    std::string qasmInput = qasmInputStr;

    return ParseAndTranslateImpl(qasmInput);
  }

  bool Failed() const { return error; }

  const std::string &GetErrorMessage() const { return errorMessage; }

  double GetVersion() const { return program.version; }

  const std::vector<std::string> &GetComments() const {
    return program.comments;
  }

  const std::vector<std::string> &GetIncludes() const {
    return program.includes;
  }

  // The names declared by QASM3 `input` statements, in declaration order, so
  // callers can discover what a circuit requires before supplying values via
  // ParseAndTranslate's params map.
  const std::vector<std::string> &GetInputs() const {
    return grammar.inputNames;
  }

 protected:
  std::shared_ptr<Circuits::Circuit<Time>> ParseAndTranslateImpl(
      std::string& qasmInput) {
    try {
      auto it = qasmInput.begin();
      QasmSkipper<std::string::iterator> skipper;
      if (boost::spirit::qi::phrase_parse(it, qasmInput.end(), grammar, skipper,
                                          program)) {
        if (it == qasmInput.end()) {
          grammar.ValidateInputBindings();
          return program.ToCircuit<Time>(grammar.opaqueGates,
                                         grammar.definedGates);
        } else {
          error = true;
          errorMessage = "Error: Unparsed input remaining: '" +
                         std::string(it, qasmInput.end()) + "'";
        }
      } else {
        error = true;
        errorMessage = "Error: Parsing failed, unparsed input remaining: '" +
                       std::string(it, qasmInput.end()) + "'";
      }
    } catch (const std::exception& ex) {
      error = true;
      errorMessage = ex.what();
    }

    return nullptr;
  }

  qasm::QasmGrammar<> grammar;
  qasm::Program program;
  std::string errorMessage;
  bool error = false;
};
}  // namespace qasm

#endif  //_QASMCIRC_H_
