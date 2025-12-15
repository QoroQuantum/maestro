/**
 * @file CircQasm.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Classes for translating a circuit to qasm.
 *
 * It's supposed to support only open qasm 2.0.
 */

#pragma once

#ifndef _CIRCQASM_H_
#define _CIRCQASM_H_

#include "Circuit/Circuit.h"

namespace qasm {

template <typename Time = Types::time_type>
class CircToQasm {
 public:
  enum class QasmGateType : size_t {
    X,     // XGate
    Y,     // YGate
    Z,     // ZGate
    H,     // HadamardGate
    S,     // SGate
    SDG,   // SdgGate
    Sx,    // SxGate
    SxDG,  // SxDagGate
    K,     // KGate
    T,     // TGate
    TDG,   // TdgGate
    Rx,    // RxGate
    Ry,    // RyGate
    Rz,    // RzGate
    U,     // UGate
    CZ,    // CZGate
    CY,    // CYGate
    CH,    // CHGate
    CRZ,   // CRzGate
    CU1,   // CPGate
    CU3,   // CUGate
    CRX,   // CRxGate
    CRY,   // CRyGate
    CS,
    CSDAG,
    CSX,     // CSxGate
    CSXDAG,  // CSxDagGate
    SWAP,
    IncludedGate,
    NoGate
  };

  // still need to be defined additionally:

  /*
  kCUGateType
  */

  static std::string GenerateWithMapping(
      const std::shared_ptr<Circuits::Circuit<Time>> &circuit,
      const std::unordered_map<Types::qubit_t, Types::qubit_t> &bitsMap) {
    const auto mappedCircuit =
        std::static_pointer_cast<Circuits::Circuit<Time>>(
            circuit->Remap(bitsMap, bitsMap));

    return GenerateFromCircuit(mappedCircuit, false);
  }

  // look over the circuit and convert it to qasm
  // identify which gates are used, generate gate definitions only for those
  // (that do not exist in the qasm standard) the standard ones are U3 and CX
  // attention: the definitions depend on other ones, so the order is important,
  // also some gates that are not used in the circuit still need to be defined
  // if they are used in the definitions of the gates that are used

  static std::string Generate(
      const std::shared_ptr<Circuits::Circuit<Time>> &circuit) {
    return GenerateFromCircuit(circuit, true);
  }

 private:
  static std::string GenerateFromCircuit(
      const std::shared_ptr<Circuits::Circuit<Time>> &circuit, bool clone) {
    if (circuit->empty()) return "";

    const auto circ =
        (clone ? std::static_pointer_cast<Circuits::Circuit<Time>>(
                     circuit->Clone())
               : circuit);

    circ->ConvertForDistribution();  // get rid of swap and 3 qubit gates

    std::string qasm = QasmHeader();

    qasm += QasmGatesAndRegsDefinitions(circ);

    // iterate over the circuit and generate the qasm
    for (const auto &gate : circ->GetOperations())
      qasm += OperationToQasm(gate);

    return qasm;
  }

  static std::string OperationToQasm(
      const std::shared_ptr<Circuits::IOperation<Time>> &operation) {
    std::string qasm;

    switch (operation->GetType()) {
      case Circuits::OperationType::kGate:
        qasm += GateToQasm(operation);
        break;
      case Circuits::OperationType::kMeasurement: {
        auto qbits = operation->AffectedQubits();
        auto bits = operation->AffectedBits();

        assert(qbits.size() == bits.size());

        for (size_t i = 0; i < qbits.size(); ++i)
          qasm += "measure q[" + std::to_string(qbits[i]) + "]->c" +
                  std::to_string(bits[i]) + "[0];\n";
      } break;
      case Circuits::OperationType::kReset: {
        auto qbits = operation->AffectedQubits();
        for (const auto &qbit : qbits)
          qasm += "reset q[" + std::to_string(qbit) + "];\n";
      } break;
      case Circuits::OperationType::kConditionalGate:
        [[fallthrough]];
      case Circuits::OperationType::kConditionalMeasurement:
        // conditionals are similar, generate an if and then call again for the
        // conditioned operations
        {
          auto condop =
              std::static_pointer_cast<Circuits::IConditionalOperation<Time>>(
                  operation);
          auto bits = condop->AffectedBits();
          auto vals = std::static_pointer_cast<Circuits::EqualCondition>(
                          condop->GetCondition())
                          ->GetAllBits();
          assert(bits.size() == vals.size());

          for (size_t i = 0; i < bits.size(); ++i)
            qasm += "if(c" + std::to_string(bits[i]) +
                    "==" + std::to_string(vals[i] ? 1 : 0) + ") ";

          const auto theop = condop->GetOperation();
          qasm += OperationToQasm(theop);
        }
        break;
      case Circuits::OperationType::kNoOp:
        qasm += "barrier q;\n";
        break;
      case Circuits::OperationType::kRandomGen:
        [[fallthrough]];
      case Circuits::OperationType::kConditionalRandomGen:
        [[fallthrough]];
      case Circuits::OperationType::kComposite:
        throw std::runtime_error("Not supported!");
        break;
    }

    return qasm;
  }

  static std::string GateToQasm(
      const std::shared_ptr<Circuits::IOperation<Time>> &operation) {
    if (!operation || operation->GetType() != Circuits::OperationType::kGate)
      return "";

    const auto gate =
        std::static_pointer_cast<Circuits::IQuantumGate<Time>>(operation);

    std::string qasm;

    switch (gate->GetGateType()) {
      case Circuits::QuantumGateType::kPhaseGateType:
        qasm += "U(0,0," + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kXGateType:
        qasm += "x q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kYGateType:
        qasm += "y q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kZGateType:
        qasm += "z q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kHadamardGateType:
        qasm += "h q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kSGateType:
        qasm += "s q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kSdgGateType:
        qasm += "sdg q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kTGateType:
        qasm += "t q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kTdgGateType:
        qasm += "tdg q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;

      //*************************************************************************************************
      // defined here, not in the 'standard' header
      case Circuits::QuantumGateType::kSxGateType:
        qasm += "sx q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kSxDagGateType:
        qasm += "sxdg q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kKGateType:
        qasm += "k q[" + std::to_string(gate->GetQubit(0)) + "];\n";
        break;
        //*************************************************************************************************

      case Circuits::QuantumGateType::kRxGateType:
        qasm += "rx(" + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kRyGateType:
        qasm += "ry(" + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kRzGateType:
        qasm += "rz(" + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "];\n";
        break;
      case Circuits::QuantumGateType::kUGateType:
        if (gate->GetParams()[3] == 0)
          qasm += "U(" + std::to_string(gate->GetParams()[0]) + "," +
                  std::to_string(gate->GetParams()[1]) + "," +
                  std::to_string(gate->GetParams()[2]) + ") q[" +
                  std::to_string(gate->GetQubit(0)) + "];\n";
        else
          throw std::runtime_error("U with gamma non zero not supported yet!");
        break;

      case Circuits::QuantumGateType::kCXGateType:
        // first is the control qubit!
        qasm += "CX q[" + std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
      case Circuits::QuantumGateType::kCYGateType:
        qasm += "cy q[" + std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
      case Circuits::QuantumGateType::kCZGateType:
        qasm += "cz q[" + std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
      case Circuits::QuantumGateType::kCPGateType:
        qasm += "cu1(" + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;

      //*************************************************************************************************
      // defined here, not in the 'standard' header
      case Circuits::QuantumGateType::kCRxGateType:
        qasm += "crx(" + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
      case Circuits::QuantumGateType::kCRyGateType:
        qasm += "cry(" + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
        //*************************************************************************************************

      case Circuits::QuantumGateType::kCRzGateType:
        qasm += "crz(" + std::to_string(gate->GetParams()[0]) + ") q[" +
                std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
      case Circuits::QuantumGateType::kCHGateType:
        qasm += "ch q[" + std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;

      //*************************************************************************************************
      // defined here, not in the 'standard' header
      case Circuits::QuantumGateType::kCSxGateType:
        qasm += "csx q[" + std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
      case Circuits::QuantumGateType::kCSxDagGateType:
        qasm += "csxdag q[" + std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;

      // we have a problem with this, our CU is with 4 parameters, so not fully
      // converted if the 4th parameter is not zero!
      case Circuits::QuantumGateType::kCUGateType:
        if (gate->GetParams()[3] == 0)
          qasm += "cu3(" + std::to_string(gate->GetParams()[0]) + "," +
                  std::to_string(gate->GetParams()[1]) + "," +
                  std::to_string(gate->GetParams()[2]) + ") q[" +
                  std::to_string(gate->GetQubit(0)) + "], q[" +
                  std::to_string(gate->GetQubit(1)) + "];\n";
        else
          throw std::runtime_error("CU with gamma non zero not supported yet!");
        break;
      //*************************************************************************************************

      // swap is converted to three CX gates
      case Circuits::QuantumGateType::kSwapGateType:
        qasm += "swap q[" + std::to_string(gate->GetQubit(0)) + "],q[" +
                std::to_string(gate->GetQubit(1)) + "];\n";
        break;
      // three qubit gates, do not need to be converted as they are converted to
      // two qubit gates already
      case Circuits::QuantumGateType::kCSwapGateType:
        [[fallthrough]];
      case Circuits::QuantumGateType::kCCXGateType:
        throw std::runtime_error("Not supported!");
        break;
    }

    return qasm;
  }

  static std::string QasmHeader() {
    std::string qasm = "OPENQASM 2.0;\n";

    return qasm;
  }

  // assume qubits and cbits starting from 0, map the circuit to that if not
  // already like that
  static std::string QasmRegisters(
      const std::shared_ptr<Circuits::Circuit<Time>> &circuit) {
    const auto nrq = circuit->GetMaxQubitIndex() + 1;

    const std::string nrq_str = std::to_string(nrq);

    std::string qasm = "qreg q[" + nrq_str + "];\n";

    std::set<size_t> measQubits;

    for (const auto &op : *circuit) {
      const auto bits = op->AffectedBits();

      for (auto bit : bits) measQubits.insert(bit);
    }

    int creg_count = 0;
    for (auto bit : measQubits) {
      // this completion is needed to have a proper total cregister definition
      // we need this to be able to address the bits properly, otherwise
      // conversion circuit -> qasm -> circuit would not work properly
      if (creg_count < static_cast<int>(bit))
        qasm += "creg c" + std::to_string(creg_count) + "[" +
                std::to_string(bit - creg_count) + "];\n";

      qasm += "creg c" + std::to_string(bit) + "[1];\n";
      creg_count = bit + 1;
    }

    return qasm;
  }

  static std::string QasmGatesAndRegsDefinitions(
      const std::shared_ptr<Circuits::Circuit<Time>> &circuit) {
    std::vector<bool> neededGates(static_cast<size_t>(QasmGateType::NoGate),
                                  false);

    // #define DONT_USE_HEADER_DEFINITIONS 1

    for (const auto &op : circuit->GetOperations()) {
      const auto opType = op->GetType();
      if (opType == Circuits::OperationType::kGate ||
          opType == Circuits::OperationType::kConditionalGate) {
        const auto &gate =
            std::static_pointer_cast<Circuits::IQuantumGate<Time>>(
                opType == Circuits::OperationType::kConditionalGate
                    ? std::static_pointer_cast<
                          Circuits::IConditionalOperation<Time>>(op)
                          ->GetOperation()
                    : op);

        switch (gate->GetGateType()) {
          case Circuits::QuantumGateType::kPhaseGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::U)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kXGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::X)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kYGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::Y)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kZGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::Z)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kHadamardGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::H)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kSGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::S)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kSdgGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::SDG)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kTGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::T)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kTdgGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::TDG)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;

          //*************************************************************************************************
          // defined here, not in the 'standard' header
          case Circuits::QuantumGateType::kSxGateType:
            neededGates[static_cast<size_t>(QasmGateType::Sx)] = true;
#ifndef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kSxDagGateType:
            neededGates[static_cast<size_t>(QasmGateType::SxDG)] = true;
#ifndef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kKGateType:
            neededGates[static_cast<size_t>(QasmGateType::K)] = true;
#ifndef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
            //*************************************************************************************************

          case Circuits::QuantumGateType::kRxGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::Rx)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kRyGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::Ry)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kRzGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::Rz)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;

          case Circuits::QuantumGateType::kUGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::U)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;

          case Circuits::QuantumGateType::kCXGateType:
            // standard gate
#ifndef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kCYGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::CY)] = true;
            neededGates[static_cast<size_t>(QasmGateType::SDG)] = true;
            neededGates[static_cast<size_t>(QasmGateType::S)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kCZGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::CZ)] = true;
            neededGates[static_cast<size_t>(QasmGateType::H)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kCPGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::CU1)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;

          //*************************************************************************************************
          // defined here, not in the 'standard' header
          case Circuits::QuantumGateType::kCRxGateType:
            neededGates[static_cast<size_t>(QasmGateType::CRX)] = true;
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::CU3)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kCRyGateType:
            neededGates[static_cast<size_t>(QasmGateType::CRY)] = true;
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::CU3)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
            //*************************************************************************************************

          case Circuits::QuantumGateType::kCRzGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::CRZ)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;
          case Circuits::QuantumGateType::kCHGateType:
#ifdef DONT_USE_HEADER_DEFINITIONS
            neededGates[static_cast<size_t>(QasmGateType::CH)] = true;
            neededGates[static_cast<size_t>(QasmGateType::H)] = true;
            neededGates[static_cast<size_t>(QasmGateType::SDG)] = true;
            neededGates[static_cast<size_t>(QasmGateType::T)] = true;
            neededGates[static_cast<size_t>(QasmGateType::S)] = true;
            neededGates[static_cast<size_t>(QasmGateType::X)] = true;
#else
            neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] = true;
#endif
            break;

            //*************************************************************************************************
            // defined here, not in the 'standard' header
          case Circuits::QuantumGateType::kCSxGateType:
            neededGates[static_cast<size_t>(QasmGateType::CSX)] = true;
            neededGates[static_cast<size_t>(QasmGateType::CS)] = true;
            break;
          case Circuits::QuantumGateType::kCSxDagGateType:
            neededGates[static_cast<size_t>(QasmGateType::CSXDAG)] = true;
            neededGates[static_cast<size_t>(QasmGateType::CSDAG)] = true;
            break;

            // we have a problem with this, our CU is with 4 parameters, so not
            // fully converted if the 4th parameter is not zero!
          case Circuits::QuantumGateType::kCUGateType:
            if (gate->GetParams()[3] == 0) {
#ifdef DONT_USE_HEADER_DEFINITIONS
              neededGates[static_cast<size_t>(QasmGateType::CU3)] = true;
#else
              neededGates[static_cast<size_t>(QasmGateType::IncludedGate)] =
                  true;
#endif
            } else
              throw std::runtime_error(
                  "CU with gamma non zero not supported yet!");
            break;
            //*************************************************************************************************

            // swap is converted to three CX gates
          case Circuits::QuantumGateType::kSwapGateType:
            neededGates[static_cast<size_t>(QasmGateType::SWAP)] = true;
            break;
            // three qubit gates, do not need to be converted as they are
            // converted to two qubit gates already
          case Circuits::QuantumGateType::kCSwapGateType:
            [[fallthrough]];
          case Circuits::QuantumGateType::kCCXGateType:
            throw std::runtime_error("Not supported!");
            break;
        }
      }
    }

    std::string qasm;

    if (neededGates[static_cast<size_t>(QasmGateType::IncludedGate)])
      qasm += "include \"qelib1.inc\";\n";

    qasm += QasmRegisters(circuit);

    // WARNING: order matters, so be sure you won't define gates based on gates
    // that are defined later here

    if (neededGates[static_cast<size_t>(QasmGateType::X)])
      qasm += XGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::Y)])
      qasm += YGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::Z)])
      qasm += ZGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::H)])
      qasm += HGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::S)])
      qasm += SGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::SDG)])
      qasm += SDGGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::Sx)])
      qasm += SxGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::SxDG)])
      qasm += SxDGGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::K)])
      qasm += KGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::T)])
      qasm += TGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::TDG)])
      qasm += TDGGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::Rx)])
      qasm += RxGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::Ry)])
      qasm += RyGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::Rz)])
      qasm += RzGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::CZ)])
      qasm += CZGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CY)])
      qasm += CYGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CH)])
      qasm += CHGateDefinition();
    // qasm += CCXGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::CRZ)])
      qasm += CRZGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CU1)])
      qasm += CU1GateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CU3)])
      qasm += CU3GateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CRX)])
      qasm += CRXGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CRY)])
      qasm += CRYGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::SWAP)])
      qasm += SwapGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::CS)])
      qasm += CSGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CSDAG)])
      qasm += CSDAGGateDefinition();

    if (neededGates[static_cast<size_t>(QasmGateType::CSX)])
      qasm += CSXGateDefinition();
    if (neededGates[static_cast<size_t>(QasmGateType::CSXDAG)])
      qasm += CSXDAGGateDefinition();

    return qasm;
  }

  // gates definitions
  static std::string U3GateDefinition() {
    return "gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }\n";
  }

  static std::string U2GateDefinition() {
    return "gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }\n";
  }

  static std::string U1GateDefinition() {
    return "gate u1(lambda) q { U(0,0,lambda) q; }\n";
  }

  static std::string XGateDefinition() {
    return "gate x a { U(pi,0,pi) a; }\n";
  }

  static std::string YGateDefinition() {
    return "gate y a { U(pi,pi/2,pi/2) a; }\n";
  }

  static std::string ZGateDefinition() { return "gate z a { U(0,0,pi) a; }\n"; }

  static std::string HGateDefinition() {
    return "gate h a { U(pi/2,0,pi) a; }\n";
  }

  static std::string SGateDefinition() {
    return "gate s a { U(0,0,pi/2) a; }\n";
  }

  static std::string SDGGateDefinition() {
    return "gate sdg a { U(0,0,-pi/2) a; }\n";
  }

  // the following two introduce a global phase compared with the operators for
  // sx and sxdg, but that should be ok
  static std::string SxGateDefinition() {
    return "gate sx a { U(pi/2,-pi/2,pi/2) a; }\n";  // this is a rotation,
                                                     // equivalent up to a
                                                     // global phase
  }

  static std::string SxDGGateDefinition() {
    return "gate sxdg a { U(-pi/2,-pi/2,pi/2) a; }\n";
  }

  static std::string KGateDefinition() {
    return "gate k a { U(pi/2,pi/2,pi/2) a; }\n";
  }

  static std::string TGateDefinition() {
    return "gate t a { U(0,0,pi/4) a; }\n";
  }

  static std::string TDGGateDefinition() {
    return "gate tdg a { U(0,0,-pi/4) a; }\n";
  }

  static std::string RxGateDefinition() {
    return "gate rx(theta) a { U(theta,-pi/2,pi/2) a; }\n";
  }

  static std::string RyGateDefinition() {
    return "gate ry(theta) a { U(theta,0,0) a; }\n";
  }

  static std::string RzGateDefinition() {
    return "gate rz(phi) a { U(0,0,phi) a; }\n";
  }

  static std::string SwapGateDefinition() {
    return "gate swap a,b { CX a,b; CX b,a; CX a,b; }\n";
  }

  // with hadamard it's going to the x basis... then after cx, back to the z
  // basis, applying hadamard again
  static std::string CZGateDefinition() {
    return "gate cz a,b { h b; CX a,b; h b; }\n";
  }

  static std::string CYGateDefinition() {
    return "gate cy a,b { sdg b; CX a,b; s b; }\n";
  }

  static std::string CHGateDefinition() {
    return "gate ch a,b { h b; sdg b; CX a,b; h b; t b; CX a,b; t b; h b; s b; "
           "x b; s a; }\n";
  }

  static std::string CCXGateDefinition() {
    return "gate ccx a,b,c\
			{\
				h c;\
				cx b, c; tdg c;\
				cx a, c; t c;\
				cx b, c; tdg c;\
				cx a, c; t b; t c; h c;\
				cx a, b; t a; tdg b;\
				cx a, b;\
			}\n";
  }

  static std::string CU1GateDefinition() {
    return "gate cu1(lambda) a,b { U(0,0,lambda/2) a; CX a,b; U(0,0,-lambda/2) "
           "b; CX a,b; U(0,0,lambda/2) b; }\n";
  }

  static std::string CU3GateDefinition() {
    return "gate cu3(theta,phi,lambda) c,t { U(0,0,(lambda+phi)/2) c; "
           "U(0,0,(lambda-phi)/2) t; CX c,t; U(-theta/2,0,-(phi+lambda)/2) t; "
           "CX c,t; U(theta/2,phi,0) t; }\n";
  }

  //*************************************************************************************************
  // defined here, not in the 'standard' header

  static std::string CRXGateDefinition() {
    return "gate crx(theta) a,b { cu3(theta,-pi/2,pi/2) a,b; }\n";
  }

  static std::string CRYGateDefinition() {
    return "gate cry(theta) a,b { cu3(theta,0,0) a,b; }\n";
  }

  static std::string CRZGateDefinition() {
    return "gate crz(lambda) a,b { U(0,0,lambda/2) b; CX a,b; U(0,0,-lambda/2) "
           "b; CX a,b; }\n";
  }

  static std::string CSGateDefinition() {
    return "gate cs c,t { U(0,0,pi/4) c; U(0,0,pi/4) t; CX c,t; U(0,0,-pi/4) "
           "t; CX c,t; }\n";
  }

  static std::string CSDAGGateDefinition() {
    return "gate csdag c,t { CX c,t; U(0,0,pi/4) t; CX c,t; U(0,0,-pi/4) c; "
           "U(0,0,-pi/4) t; }\n";
  }

  static std::string CSXGateDefinition() {
    return "gate csx c,t { U(pi/2,0,pi) t; cs c,t; U(pi/2,0,pi) t; }\n";
  }

  static std::string CSXDAGGateDefinition() {
    return "gate csxdag c,t { U(pi/2,0,pi) t; csdag c,t; U(pi/2,0,pi) t; }\n";
  }
};

}  // namespace qasm

#endif
