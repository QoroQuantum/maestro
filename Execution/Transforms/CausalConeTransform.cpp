#include "CausalConeTransform.h"
#include "ConeAnalysis.h"
#include "PauliCone.h"

#include "Execution/SimulatorConfig.h"

#include <algorithm>
#include <cctype>

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
    if (!CanTraceCone(circuit, paulis, HasEnsembleState(ctx.config.simulation_type), ctx.config.allow_reset_marginalization))
        return;

    size_t width = circuit->GetMaxQubitIndex() + 1;
    for (const auto &pauli : paulis)
    {
        // Leave malformed observables to the existing estimation path.
        if (pauli.find_first_not_of("IXYZixyz") != std::string::npos)
            return;
        width = std::max(width, pauli.size());
    }
    std::vector<bool> active(width, false);
    std::vector<uint64_t> x(width, 0), z(width, 0);
    for (const auto &pauli : paulis)
        for (size_t q = 0; q < pauli.size(); ++q)
        {
            const auto p = std::toupper(static_cast<unsigned char>(pauli[q]));
            x[q] |= p == 'X' || p == 'Y';
            z[q] |= p == 'Z' || p == 'Y';
            active[q] = (x[q] | z[q]) != 0;
        }

    const auto initial_identities = InitialIdentities(operations, width);
    Circuits::Circuit<double>::OperationsVector retained;
    for (size_t i = operations.size(); i-- > 0;)
    {
        const auto &op = operations[i];
        if (initial_identities[i])
            continue;
        const auto qubits = op->AffectedQubits();
        if (!std::any_of(qubits.begin(), qubits.end(), [&](size_t q) { return (x[q] | z[q]) != 0; }))
            continue;
        if (!TracePauliOperation(op, x, z, HasEnsembleState(ctx.config.simulation_type) || ctx.config.allow_reset_marginalization))
            continue;
        retained.push_back(op);
        for (auto q : qubits)
            active[q] = true;
    }

    Circuits::Circuit<double>::BitMapping mapping;
    for (size_t q = 0; q < width; ++q)
        if (active[q])
            mapping.emplace(q, mapping.size());

    const bool remapped = std::any_of(mapping.begin(), mapping.end(), [](const auto &entry) { return entry.first != entry.second; });
    if (retained.size() != operations.size() || remapped)
    {
        auto reduced = std::make_shared<Circuits::Circuit<double>>();
        for (auto it = retained.rbegin(); it != retained.rend(); ++it)
            reduced->AddOperation((*it)->Remap(mapping, {}));
        circuit = std::move(reduced);
    }

    for (auto &pauli : paulis)
    {
        // Backends require at least one qubit, even for identity-only requests.
        std::string compact(std::max(size_t{1}, mapping.size()), 'I');
        for (const auto &[original, mapped] : mapping)
            if (original < pauli.size())
                compact[mapped] = pauli[original];
        pauli = std::move(compact);
    }
}

} // namespace MaestroExecution
