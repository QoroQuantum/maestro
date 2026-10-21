#pragma once

#include "Circuit/Circuit.h"
#include "Simulators/Interfaces/State.h"

#include <algorithm>
#include <numeric>

namespace MaestroExecution
{
inline bool HasEnsembleState(Simulators::SimulationType method)
{
    return method == Simulators::SimulationType::kDensityMatrix || method == Simulators::SimulationType::kMatrixProductOperator;
}

// CPTP maps cancel outside the observable support in ensemble evolution. A
// sampled reset can instead collapse an entangled spectator and change an
// individual trajectory. Only discard such resets in disconnected components;
// retain the existing trajectory path whenever a reset shares a component with
// any requested observable. Classical dependencies always require full execution.
inline bool CanTraceConeSupport(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<size_t> &support, bool ensemble = false,
                                bool marginalize_resets = false)
{
    if (!circuit)
        return false;
    size_t width = circuit->GetMaxQubitIndex() + 1;
    for (auto q : support)
        width = std::max(width, q + 1);
    bool resets = false;
    for (const auto &op : circuit->GetOperations())
    {
        if (!op)
            return false;
        switch (op->GetType())
        {
        case Circuits::OperationType::kGate:
        case Circuits::OperationType::kNoOp:
        case Circuits::OperationType::kDelay:
            break;
        case Circuits::OperationType::kQuantumChannel:
            if (!ensemble)
                return false;
            break;
        case Circuits::OperationType::kReset:
            resets = true;
            break;
        default:
            return false;
        }
    }
    if (ensemble || marginalize_resets || !resets)
        return true;
    std::vector<size_t> parent(width);
    std::iota(parent.begin(), parent.end(), size_t{0});
    const auto find = [&](size_t q) {
        while (parent[q] != q)
        {
            parent[q] = parent[parent[q]];
            q = parent[q];
        }
        return q;
    };
    for (const auto &op : circuit->GetOperations())
    {
        // A multi-target reset/delay is a product operation, not an interaction.
        if (op->GetType() != Circuits::OperationType::kGate)
            continue;
        const auto qubits = op->AffectedQubits();
        for (size_t i = 1; i < qubits.size(); ++i)
            parent[find(qubits[i])] = find(qubits[0]);
    }
    std::vector<bool> observed(width, false);
    for (auto q : support)
        observed[find(q)] = true;
    for (const auto &op : circuit->GetOperations())
        if (op->GetType() == Circuits::OperationType::kReset)
            for (auto q : op->AffectedQubits())
                if (observed[find(q)])
                    return false;
    return true;
}

inline bool CanTraceCone(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables, bool ensemble = false,
                         bool marginalize_resets = false)
{
    std::vector<size_t> support;
    for (const auto &pauli : observables)
    {
        if (pauli.find_first_not_of("IXYZixyz") != std::string::npos)
            return false;
        for (size_t q = 0; q < pauli.size(); ++q)
            if (pauli[q] != 'I' && pauli[q] != 'i')
                support.push_back(q);
    }
    return CanTraceConeSupport(circuit, support, ensemble, marginalize_resets);
}

struct StructuralCone
{
    Circuits::Circuit<double>::BitMapping mapping;
    std::vector<size_t> operations;
};

// One active label per wire, merged on backwards interactions. This computes
// disjoint measurement cones without a dense observable-by-register matrix.
inline std::vector<StructuralCone> TraceMeasurementCones(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<size_t> &measured)
{
    size_t width = circuit->GetMaxQubitIndex() + 1;
    for (auto q : measured)
        width = std::max(width, q + 1);
    std::vector<size_t> parent(width), group(width, width);
    std::iota(parent.begin(), parent.end(), size_t{0});
    const auto find = [&](size_t q) {
        while (parent[q] != q)
        {
            parent[q] = parent[parent[q]];
            q = parent[q];
        }
        return q;
    };
    std::vector<bool> active(width);
    for (auto q : measured)
        active[q] = true;
    const auto &ops = circuit->GetOperations();
    std::vector<bool> retained(ops.size());
    for (size_t i = ops.size(); i-- > 0;)
    {
        const auto qs = ops[i]->AffectedQubits();
        if (std::none_of(qs.begin(), qs.end(), [&](size_t q) { return active[q]; }))
            continue;
        retained[i] = true;
        for (auto q : qs)
        {
            active[q] = true;
            parent[find(q)] = find(qs.front());
        }
    }
    std::vector<StructuralCone> cones;
    for (size_t q = 0; q < width; ++q)
        if (active[q])
        {
            const auto root = find(q);
            if (group[root] == width)
            {
                group[root] = cones.size();
                cones.emplace_back();
            }
            auto &map = cones[group[root]].mapping;
            map.emplace(q, map.size());
        }
    for (size_t i = 0; i < ops.size(); ++i)
        if (retained[i])
            cones[group[find(ops[i]->AffectedQubits().front())]].operations.push_back(i);
    return cones;
}

inline StructuralCone TraceStructuralCone(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables)
{
    size_t width = circuit->GetMaxQubitIndex() + 1;
    for (const auto &p : observables)
        width = std::max(width, p.size());
    std::vector<bool> active(width);
    for (const auto &p : observables)
        for (size_t q = 0; q < p.size(); ++q)
            active[q] = active[q] || (p[q] != 'I' && p[q] != 'i');
    StructuralCone result;
    const auto &ops = circuit->GetOperations();
    for (size_t i = ops.size(); i-- > 0;)
    {
        const auto qs = ops[i]->AffectedQubits();
        if (std::any_of(qs.begin(), qs.end(), [&](size_t q) { return active[q]; }))
        {
            result.operations.push_back(i);
            for (auto q : qs)
                active[q] = true;
        }
    }
    std::reverse(result.operations.begin(), result.operations.end());
    for (size_t q = 0; q < width; ++q)
        if (active[q])
            result.mapping.emplace(q, result.mapping.size());
    return result;
}
} // namespace MaestroExecution
