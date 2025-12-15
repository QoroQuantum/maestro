/**
 * @file QasmCirc.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Class for parsing qasm and translating it to a circuit.
 *
 * It's supposed to support only open qasm 2.0.
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

  std::shared_ptr<Circuits::Circuit<Time>> ParseAndTranslate(
      const std::string &qasmInputStr) {
    clear();

    std::string qasmInput = qasmInputStr;

    try {
      auto it = qasmInput.begin();
      if (boost::spirit::qi::phrase_parse(it, qasmInput.end(), grammar,
                                          qasm::ascii::space, program)) {
        if (it == qasmInput.end()) {
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
    } catch (const std::exception &ex) {
      error = true;
      errorMessage = ex.what();
    }

    return nullptr;
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

 protected:
  qasm::QasmGrammar<> grammar;
  qasm::Program program;
  std::string errorMessage;
  bool error = false;
};
}  // namespace qasm

#endif  //_QASMCIRC_H_
