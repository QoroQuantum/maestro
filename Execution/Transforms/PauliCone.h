#pragma once
#include "Circuit/Circuit.h"
#include <algorithm>

namespace MaestroExecution
{
using Gate = Circuits::QuantumGateType;

inline bool Diagonal(Gate type)
{
    switch (type)
    {
    case Gate::kZGateType:
    case Gate::kSGateType:
    case Gate::kSdgGateType:
    case Gate::kTGateType:
    case Gate::kTdgGateType:
    case Gate::kRzGateType:
    case Gate::kPhaseGateType:
    case Gate::kCZGateType:
    case Gate::kCPGateType:
    case Gate::kCRzGateType:
        return true;
    default:
        return false;
    }
}

inline size_t Controls(Gate type)
{
    switch (type)
    {
    case Gate::kCCXGateType:
        return 2;
    case Gate::kCXGateType:
    case Gate::kCYGateType:
    case Gate::kCZGateType:
    case Gate::kCPGateType:
    case Gate::kCRxGateType:
    case Gate::kCRyGateType:
    case Gate::kCRzGateType:
    case Gate::kCHGateType:
    case Gate::kCSxGateType:
    case Gate::kCSxDagGateType:
    case Gate::kCUGateType:
    case Gate::kCSwapGateType:
        return 1;
    default:
        return 0;
    }
}

// Sound product-state facts only. Noise and unknown gates invalidate facts;
// noise operations themselves remain in place and retain their original timing.
inline std::vector<bool> InitialIdentities(const Circuits::Circuit<double>::OperationsVector &ops, size_t width)
{
    std::vector<bool> zero(width, true), skip(ops.size(), false);
    for (size_t i = 0; i < ops.size(); ++i)
    {
        const auto &op = ops[i];
        const auto qs = op->AffectedQubits();
        if (op->GetType() == Circuits::OperationType::kNoOp || op->GetType() == Circuits::OperationType::kDelay)
            continue;
        if (op->GetType() == Circuits::OperationType::kReset)
        {
            const auto reset = std::static_pointer_cast<Circuits::Reset<double>>(op);
            const auto &targets = reset->GetResetTargets();
            const auto &qubits = reset->GetQubits();
            for (size_t j = 0; j < qubits.size(); ++j)
                zero[qubits[j]] = j >= targets.size() || !targets[j];
            continue;
        }
        const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op);
        if (gate)
        {
            const auto type = gate->GetGateType();
            const auto controls = Controls(type);
            for (unsigned int c = 0; c < controls; ++c)
                skip[i] = skip[i] || zero[gate->GetQubit(c)];
            if (Diagonal(type))
                skip[i] = skip[i] || std::all_of(qs.begin(), qs.end(), [&](size_t q) { return zero[q]; });
            if (skip[i] || Diagonal(type))
                continue;
            // A control is never modified by its controlled unitary.
            for (unsigned int q = static_cast<unsigned int>(controls); q < gate->GetNumQubits(); ++q)
                zero[gate->GetQubit(q)] = false;
        }
        else
            for (auto q : qs)
                zero[q] = false;
    }
    return skip;
}

// A bit identifies one observable. Axis unions bound every term, including
// non-Clifford branches. Remove only gates commuting with every possible term.
template <class Mask> Mask TracePauliGate(const Circuits::IQuantumGate<double> &gate, std::vector<Mask> &x, std::vector<Mask> &z)
{
    const auto qs = gate.AffectedQubits();
    const auto type = gate.GetGateType();
    Mask active = 0, xs = 0;
    for (auto q : qs)
    {
        active |= x[q] | z[q];
        xs |= x[q];
    }
    if (!active)
        return 0;
    const auto q = gate.GetQubit(0);
    if (type == Gate::kRxGateType || type == Gate::kXGateType || type == Gate::kSxGateType || type == Gate::kSxDagGateType)
    {
        const auto keep = z[q];
        if (type != Gate::kXGateType)
            x[q] |= z[q];
        return keep;
    }
    if (type == Gate::kCXGateType)
    {
        const auto t = gate.GetQubit(1);
        const auto keep = x[q] | z[t];
        x[t] |= x[q];
        z[q] |= z[t];
        return keep;
    }
    if (type == Gate::kHadamardGateType)
    {
        std::swap(x[q], z[q]);
        return active;
    }
    if (type == Gate::kSwapGateType)
    {
        const auto t = gate.GetQubit(1);
        std::swap(x[q], x[t]);
        std::swap(z[q], z[t]);
        return active;
    }
    if (qs.size() == 1 && Diagonal(type))
    {
        if (type != Gate::kZGateType)
            z[q] |= x[q];
        return xs;
    }
    auto keep = Diagonal(type) ? xs : active;
    if (type == Gate::kCCXGateType)
        keep = x[q] | x[gate.GetQubit(1)] | z[gate.GetQubit(2)];
    else if (Controls(type) && !Diagonal(type))
    {
        keep = x[q]; // Z on the control alone commutes with controlled U.
        for (unsigned int j = 1; j < gate.GetNumQubits(); ++j)
            keep |= x[gate.GetQubit(j)] | z[gate.GetQubit(j)];
    }
    if (type != Gate::kYGateType)
        for (auto t : qs)
        {
            if (!Diagonal(type))
                x[t] |= keep;
            z[t] |= keep;
        }
    return keep;
}

// Shared by single-cone and bit-batched analysis. Used wires are tracked by
// callers even when a reset terminates their backwards axis propagation.
template <class Mask>
Mask TracePauliOperation(const std::shared_ptr<Circuits::IOperation<double>> &op, std::vector<Mask> &x, std::vector<Mask> &z, bool marginalize_resets)
{
    if (op->GetType() == Circuits::OperationType::kNoOp || op->GetType() == Circuits::OperationType::kDelay)
        return 0;
    if (const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<double>>(op))
        return TracePauliGate(*gate, x, z);
    const auto qs = op->AffectedQubits();
    Mask keep = 0;
    for (auto q : qs)
        keep |= x[q] | z[q];
    for (auto q : qs)
    {
        if (marginalize_resets && op->GetType() == Circuits::OperationType::kReset)
            x[q] = z[q] = 0;
        else
        {
            x[q] |= keep;
            z[q] |= keep;
        }
    }
    return keep;
}
} // namespace MaestroExecution
