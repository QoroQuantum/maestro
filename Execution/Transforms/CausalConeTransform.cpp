#include "CausalConeTransform.h"

#include "Execution/SimulatorConfig.h"

#include <algorithm>

namespace MaestroExecution
{

std::string_view CausalConeTransform::Name() const
{
    return "CausalCone";
}

bool CausalConeTransform::IsApplicable(const TransformContext &ctx) const
{
    // Distributed layouts and shard counts refer to the original register.
    return ctx.config.enable_causal_cone_reduction && !Simulators::IsDistributedGpuSimulator(ctx.simulator_type);
}

void CausalConeTransform::Apply(TransformContext &ctx)
{
    auto &circuit = ctx.circuit;
    auto &paulis = ctx.observables;
    const auto &operations = circuit->GetOperations();
    // Classical dependencies and nonunitary trajectories need full execution.
    // In particular, never discard a measurement controlling a retained gate.
    for (const auto &operation : operations)
    {
        if (!operation)
            return;
        switch (operation->GetType())
        {
        case Circuits::OperationType::kGate:
        case Circuits::OperationType::kNoOp:
        case Circuits::OperationType::kDelay:
            break;
        default:
            return;
        }
    }

    size_t width = circuit->GetMaxQubitIndex() + 1;
    for (const auto &pauli : paulis)
    {
        // Leave malformed observables to the existing estimation path.
        if (pauli.find_first_not_of("IXYZixyz") != std::string::npos)
            return;
        width = std::max(width, pauli.size());
    }
    std::vector<bool> active(width, false);
    for (const auto &pauli : paulis)
        for (size_t q = 0; q < pauli.size(); ++q)
            active[q] = active[q] || (pauli[q] != 'I' && pauli[q] != 'i');

    Circuits::Circuit<double>::OperationsVector retained;
    for (auto it = operations.rbegin(); it != operations.rend(); ++it)
    {
        const auto qubits = (*it)->AffectedQubits();
        if (std::any_of(qubits.begin(), qubits.end(), [&](size_t q) { return active[q]; }))
        {
            retained.push_back(*it);
            for (auto q : qubits)
                active[q] = true;
        }
    }

    Circuits::Circuit<double>::BitMapping mapping;
    for (size_t q = 0; q < width; ++q)
        if (active[q])
            mapping.emplace(q, mapping.size());

    auto reduced = std::make_shared<Circuits::Circuit<double>>();
    for (auto it = retained.rbegin(); it != retained.rend(); ++it)
        reduced->AddOperation((*it)->Remap(mapping, {}));

    for (auto &pauli : paulis)
    {
        // Backends require at least one qubit, even for identity-only requests.
        std::string compact(std::max(size_t{1}, mapping.size()), 'I');
        for (const auto &[original, mapped] : mapping)
            if (original < pauli.size())
                compact[mapped] = pauli[original];
        pauli = std::move(compact);
    }
    circuit = std::move(reduced);
}

} // namespace MaestroExecution
