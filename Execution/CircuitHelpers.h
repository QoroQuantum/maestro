#pragma once
#include "Circuit/Factory.h"
#include "Circuit/UnitaryPreparation.h"
#include "Json/Json.h"

namespace MaestroExecution
{
using OperationPtr = std::shared_ptr<Circuits::IOperation<double>>;

// An inverse with the same gate type may still require transformed parameters.
inline Circuits::QuantumGateType AdjointGateType(Circuits::QuantumGateType type)
{
    using Gate = Circuits::QuantumGateType;
    switch (type)
    {
    case Gate::kXGateType:
    case Gate::kYGateType:
    case Gate::kZGateType:
    case Gate::kHadamardGateType:
    case Gate::kKGateType:
    case Gate::kCXGateType:
    case Gate::kCYGateType:
    case Gate::kCZGateType:
    case Gate::kCHGateType:
    case Gate::kSwapGateType:
    case Gate::kCCXGateType:
    case Gate::kCSwapGateType:
    case Gate::kPhaseGateType:
    case Gate::kRxGateType:
    case Gate::kRyGateType:
    case Gate::kRzGateType:
    case Gate::kUGateType:
    case Gate::kCPGateType:
    case Gate::kCRxGateType:
    case Gate::kCRyGateType:
    case Gate::kCRzGateType:
    case Gate::kCUGateType:
        return type;
    case Gate::kSGateType:
        return Gate::kSdgGateType;
    case Gate::kSdgGateType:
        return Gate::kSGateType;
    case Gate::kTGateType:
        return Gate::kTdgGateType;
    case Gate::kTdgGateType:
        return Gate::kTGateType;
    case Gate::kSxGateType:
        return Gate::kSxDagGateType;
    case Gate::kSxDagGateType:
        return Gate::kSxGateType;
    case Gate::kCSxGateType:
        return Gate::kCSxDagGateType;
    case Gate::kCSxDagGateType:
        return Gate::kCSxGateType;
    default:
        return Gate::kNone;
    }
}

// Both overlap operands must be reversible, including the one only executed forward.
inline std::shared_ptr<Circuits::Circuit<>> UnitaryPreparation(const std::shared_ptr<Circuits::Circuit<>> &circuit, const std::string &query)
{
    auto result = std::make_shared<Circuits::Circuit<>>();
    Circuits::ForEachUnitaryGate(circuit, query, Circuits::UnitaryPreparationQuery::Overlap, [&](const auto &gate) {
        if (AdjointGateType(gate->GetGateType()) == Circuits::QuantumGateType::kNone)
            throw std::invalid_argument(query + " does not support gate type " + std::to_string(static_cast<int>(gate->GetGateType())) + " (unknown adjoint).");
        result->AddOperation(gate);
    });
    result->SetGlobalPhase(circuit->GetGlobalPhase());
    return result;
}

// Only delays and no-ops have an empty inverse in a preparation.
inline OperationPtr adjoint_gate(const OperationPtr &op)
{
    if (!op)
        throw std::invalid_argument("Cannot take the adjoint of a null operation.");
    if (op->GetType() == Circuits::OperationType::kComposite)
    {
        const auto &nested = std::static_pointer_cast<Circuits::Circuit<double>>(op)->GetOperations();
        auto adjoint = std::make_shared<Circuits::Circuit<double>>();
        adjoint->SetGlobalPhase(-std::static_pointer_cast<Circuits::Circuit<double>>(op)->GetGlobalPhase());
        for (auto it = nested.rbegin(); it != nested.rend(); ++it)
            if (auto inverse = adjoint_gate(*it))
                adjoint->AddOperation(inverse);
        return adjoint;
    }
    if (op->GetType() == Circuits::OperationType::kDelay || op->GetType() == Circuits::OperationType::kNoOp)
        return nullptr;

    auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op);
    if (!gate)
        throw std::invalid_argument("Only unitary gates have a gate adjoint.");

    const auto type = gate->GetGateType();
    const auto inverse = AdjointGateType(type);
    if (inverse == Circuits::QuantumGateType::kNone)
        throw Error("unsupported_capability", "Adjoint is not implemented for this gate");

    auto params = gate->GetParams();
    if (inverse == type && params.empty())
        return op->Clone();

    // Rotations and phase gates negate their angle. U and CU also exchange phi and lambda.
    for (double &param : params)
        param = -param;
    if (type == Circuits::QuantumGateType::kUGateType || type == Circuits::QuantumGateType::kCUGateType)
        std::swap(params[1], params[2]);
    return Circuits::CircuitFactory<>::CreateGateWithVectors(inverse, gate->AffectedQubits(), params);
}

} // namespace MaestroExecution
