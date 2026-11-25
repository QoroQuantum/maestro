#pragma once

#include "Circuit.h"

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace Circuits
{
enum class UnitaryPreparationQuery
{
    Amplitude,
    Overlap
};

// Visit the gates preparing the coherent state before readout. A measurement may
// precede gates on other qubits, including qubits entangled with the measured one.
template <typename VisitGate>
void ForEachUnitaryGate(const std::shared_ptr<Circuit<>> &circuit, const std::string &query, UnitaryPreparationQuery kind, VisitGate visit)
{
    if (!circuit)
        throw std::invalid_argument(query + " received a null circuit.");

    const auto nonunitaryOperation = [&](const std::string &operation, const char *backend = "SimulationType.PathIntegral") {
        const auto message = query + " does not support " + operation + ".";
        if (kind == UnitaryPreparationQuery::Overlap)
            return std::invalid_argument(message + " This comparison requires a unitary preparation.");
        return std::invalid_argument(message + " Use execute() for measurement counts or estimate() for observable expectations with " + backend + ".");
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
                        throw nonunitaryOperation("mid-circuit measurement on qubit " + std::to_string(gate->GetQubit(i)));
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
            throw nonunitaryOperation("reset");
        case OperationType::kConditionalGate:
            throw nonunitaryOperation("conditional gate");
        case OperationType::kConditionalMeasurement:
            throw nonunitaryOperation("conditional measurement");
        case OperationType::kConditionalRandomGen:
            throw nonunitaryOperation("conditional random bit generation");
        case OperationType::kRandomGen:
            throw nonunitaryOperation("random bit generation");
        case OperationType::kComposite:
            throw std::invalid_argument(query + " does not support composite operations. Pass the constituent operations directly.");
        case OperationType::kQuantumChannel:
            throw nonunitaryOperation("quantum channels", "SimulationType.DensityMatrix");
        default:
            throw std::invalid_argument(query + " encountered an unknown operation type.");
        }
    }
}
} // namespace Circuits
