#pragma once

#include "Circuit.h"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace Circuits
{
// Visit the gates preparing the coherent state before readout. A measurement may
// precede gates on other qubits, including qubits entangled with the measured one.
template <typename VisitGate> void ForEachUnitaryGate(const std::shared_ptr<Circuit<>> &circuit, const std::string &query, VisitGate visit)
{
    if (!circuit)
        throw std::invalid_argument(query + " received a null circuit.");

    const auto dynamicOperation = [&](const std::string &operation) {
        return std::invalid_argument(query + " does not support " + operation + ". Use execute() or estimate() with SimulationType.PathIntegral.");
    };
    std::unordered_set<Types::qubit_t> measured;
    for (const auto &op : circuit->GetOperations())
    {
        if (!op)
            throw std::invalid_argument(query + " encountered a null operation.");

        switch (op->GetType())
        {
        case OperationType::kGate: {
            const auto gate = std::dynamic_pointer_cast<IQuantumGate<>>(op);
            if (!gate)
                throw std::invalid_argument(query + " encountered an unsupported gate.");
            if (!measured.empty())
                for (unsigned int i = 0; i < gate->GetNumQubits(); ++i)
                    if (measured.count(gate->GetQubit(i)))
                        throw dynamicOperation("mid-circuit measurement on qubit " + std::to_string(gate->GetQubit(i)));
            visit(gate);
            break;
        }
        case OperationType::kMeasurement: {
            const auto &qubits = std::static_pointer_cast<MeasurementOperation<>>(op)->GetQubits();
            measured.insert(qubits.begin(), qubits.end());
            break;
        }
        case OperationType::kDelay:
        case OperationType::kNoOp:
            break;
        case OperationType::kReset:
            throw dynamicOperation("reset");
        case OperationType::kConditionalGate:
            throw dynamicOperation("conditional gate");
        case OperationType::kConditionalMeasurement:
            throw dynamicOperation("conditional measurement");
        case OperationType::kConditionalRandomGen:
            throw dynamicOperation("conditional random bit generation");
        case OperationType::kRandomGen:
            throw dynamicOperation("random bit generation");
        case OperationType::kComposite:
            throw std::invalid_argument(query + " does not support composite operations. Pass the constituent operations directly.");
        case OperationType::kQuantumChannel:
            throw std::invalid_argument(query + " does not support quantum channels. Use execute() or estimate() with SimulationType.DensityMatrix.");
        default:
            throw std::invalid_argument(query + " encountered an unknown operation type.");
        }
    }
}
} // namespace Circuits
