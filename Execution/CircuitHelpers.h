#pragma once
#include "Json.h"
#include "Circuit/Circuit.h"
namespace MaestroExecution {
using OperationPtr = std::shared_ptr<Circuits::IOperation<double>>;

inline OperationPtr adjoint_gate(const OperationPtr& op) {
  if (op->GetType() != Circuits::OperationType::kGate) return nullptr;

  auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op);
  if (!gate) return nullptr;

  const auto gt = gate->GetGateType();
  const auto params = gate->GetParams();

  switch (gt) {
    // ---- Self-inverse (Hermitian) gates ----
    case Circuits::QuantumGateType::kXGateType:
    case Circuits::QuantumGateType::kYGateType:
    case Circuits::QuantumGateType::kZGateType:
    case Circuits::QuantumGateType::kHadamardGateType:
    case Circuits::QuantumGateType::kKGateType:
    case Circuits::QuantumGateType::kCXGateType:
    case Circuits::QuantumGateType::kCYGateType:
    case Circuits::QuantumGateType::kCZGateType:
    case Circuits::QuantumGateType::kCHGateType:
    case Circuits::QuantumGateType::kSwapGateType:
    case Circuits::QuantumGateType::kCCXGateType:
    case Circuits::QuantumGateType::kCSwapGateType:
      return op->Clone();

    // ---- Paired gates ----
    case Circuits::QuantumGateType::kSGateType:
      return std::make_shared<Circuits::SdgGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSdgGateType:
      return std::make_shared<Circuits::SGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kTGateType:
      return std::make_shared<Circuits::TdgGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kTdgGateType:
      return std::make_shared<Circuits::TGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSxGateType:
      return std::make_shared<Circuits::SxDagGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kSxDagGateType:
      return std::make_shared<Circuits::SxGate<>>(gate->GetQubit());
    case Circuits::QuantumGateType::kCSxGateType:
      return std::make_shared<Circuits::CSxDagGate<>>(gate->GetQubit(0),
                                                      gate->GetQubit(1));
    case Circuits::QuantumGateType::kCSxDagGateType:
      return std::make_shared<Circuits::CSxGate<>>(gate->GetQubit(0),
                                                   gate->GetQubit(1));

    // ---- Parametric single-qubit: negate angle ----
    case Circuits::QuantumGateType::kPhaseGateType:
      return std::make_shared<Circuits::PhaseGate<>>(gate->GetQubit(),
                                                     -params[0]);
    case Circuits::QuantumGateType::kRxGateType:
      return std::make_shared<Circuits::RxGate<>>(gate->GetQubit(), -params[0]);
    case Circuits::QuantumGateType::kRyGateType:
      return std::make_shared<Circuits::RyGate<>>(gate->GetQubit(), -params[0]);
    case Circuits::QuantumGateType::kRzGateType:
      return std::make_shared<Circuits::RzGate<>>(gate->GetQubit(), -params[0]);

    // ---- U gate: U†(θ,φ,λ,γ) = U(-θ, -λ, -φ, -γ) ----
    case Circuits::QuantumGateType::kUGateType:
      return std::make_shared<Circuits::UGate<>>(
          gate->GetQubit(), -params[0], -params[2], -params[1], -params[3]);

    // ---- Controlled parametric: negate angle ----
    case Circuits::QuantumGateType::kCPGateType:
      return std::make_shared<Circuits::CPGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRxGateType:
      return std::make_shared<Circuits::CRxGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRyGateType:
      return std::make_shared<Circuits::CRyGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);
    case Circuits::QuantumGateType::kCRzGateType:
      return std::make_shared<Circuits::CRzGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0]);

    // ---- CU gate: CU†(θ,φ,λ,γ) = CU(-θ, -λ, -φ, -γ) ----
    case Circuits::QuantumGateType::kCUGateType:
      return std::make_shared<Circuits::CUGate<>>(
          gate->GetQubit(0), gate->GetQubit(1), -params[0], -params[2],
          -params[1], -params[3]);

    default:
      throw Error("unsupported_capability", "Adjoint is not implemented for this gate");
  }
}


}
