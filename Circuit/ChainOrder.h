/**
 * @file ChainOrder.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * The order of a circuit's operations for a tensor chain (a matrix product
 * state that samples quantum channels, or a matrix product operator).
 */

#pragma once

#ifndef _CIRCUIT_CHAIN_ORDER_H_
#define _CIRCUIT_CHAIN_ORDER_H_

#include "Circuit.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <vector>

namespace Circuits
{

/**
 * @brief What takes a tensor chain off its canonical form, and so what its work depends on.
 */
enum class ChainOrder
{
    kNone,            /**< leave the circuit as it is */
    kSampledChannels, /**< a matrix product state that samples channels: a channel that reads the state */
    kMixedState       /**< a matrix product operator: any channel */
};

/**
 * @brief The circuit with its operations in an order that keeps the work of a tensor chain local.
 *
 * A chain is cheap to update while it is in its canonical form. An operator
 * that is not unitary takes it off that form around its site, and the chain
 * then pays for the distance between the operations that need the form
 * restored:
 *
 * - a matrix product state that samples quantum channels keeps the
 *   environments of its bonds explicitly, and a sampled channel or a two-qubit
 *   gate costs one product of the bond dimension cubed for every site between
 *   itself and the last such operation
 *   (QC::TensorNetworks::MPSSimulatorImpl::ApplyKrausOperator);
 * - a matrix product operator keeps an orthogonality centre, and a two-qubit
 *   update costs one QR decomposition for every site the centre has to move
 *   (QC::TensorNetworks::MPOSimulatorImpl::PrepareBond).
 *
 * A circuit lists its operations in an order that jumps along the chain, and
 * that travel then costs as much as the operations. Operations on different
 * qubits commute, so a circuit may run in any order that keeps, on every
 * qubit, the order its operations had. This returns such an order: among the
 * operations whose predecessors on their qubits have been placed, the next one
 * is the one that needs the least travel, the earliest in the circuit among
 * equals. Operations that need none (one-qubit gates, delays, channels a
 * matrix product state draws without its state) keep their place among
 * themselves.
 *
 * The position of a qubit on the chain is its index, or `positions[qubit]`
 * when the backend was given another layout (ChainLine). Either holds while
 * nothing has to be routed. The circuit is therefore returned as it is when a
 * gate or a channel acts on two qubits that are not neighbours on the chain,
 * or on more than two. It is also returned as it is when nothing in it takes
 * the chain off its canonical form. Measurements, resets, classical control
 * and nested circuits keep their place: everything before one of them stays
 * before it.
 *
 * The result shares the operations of the circuit and describes the same map.
 * On a matrix product state the channels draw their branches in the new order,
 * so a seeded trajectory differs from the one of the original order; the
 * distribution of the trajectories is the same.
 *
 * @param circuit The circuit.
 * @param chain The kind of chain that runs it.
 * @param positions The position of every qubit on the chain, or null for the index.
 * @return The reordered circuit, or `circuit` itself.
 */
template <typename Time = Types::time_type>
std::shared_ptr<Circuit<Time>> ChainOrdered(const std::shared_ptr<Circuit<Time>> &circuit, ChainOrder chain = ChainOrder::kSampledChannels,
                                            const std::vector<long long int> *positions = nullptr)
{
    using Channel = Simulators::QuantumChannel;
    if (!circuit || chain == ChainOrder::kNone)
        return circuit;
    const auto &operations = circuit->GetOperations();
    const size_t count = operations.size();
    static constexpr size_t unplaced = std::numeric_limits<size_t>::max();
    const auto place = [positions](Types::qubit_t qubit) {
        if (!positions)
            return static_cast<size_t>(qubit);
        const size_t index = static_cast<size_t>(qubit);
        return index < positions->size() && (*positions)[index] >= 0 ? static_cast<size_t>((*positions)[index]) : unplaced;
    };

    struct Entry
    {
        size_t low = 0, high = 0;
        bool pair = false;
        bool barrier = false;
        bool reads = false;   // needs the canonical form around its sites: pays for the travel
        bool changes = false; // leaves the chain off that form away from its sites
        bool starts = false;  // takes a canonical chain off its form
    };

    std::vector<Entry> entries(count);
    size_t width = 0;
    bool anyStart = false;
    for (size_t i = 0; i < count; ++i)
    {
        const auto &op = operations[i];
        const auto type = op->GetType();
        Entry &entry = entries[i];
        if (type != OperationType::kGate && type != OperationType::kQuantumChannel && type != OperationType::kDelay)
        {
            entry.barrier = true;
            continue;
        }
        const auto qubits = op->AffectedQubits();
        if (qubits.empty())
        {
            entry.barrier = true;
            continue;
        }
        if (qubits.size() > 2)
            return circuit;
        const size_t first = place(qubits.front()), last = place(qubits.back());
        if (first == unplaced || last == unplaced)
            return circuit;
        entry.low = std::min(first, last);
        entry.high = std::max(first, last);
        entry.pair = qubits.size() == 2;
        if (entry.pair && entry.high - entry.low != 1)
            return circuit;
        width = std::max(width, entry.high + 1);
        if (type == OperationType::kDelay)
            continue;
        if (type == OperationType::kGate)
        {
            // A two-qubit gate is decomposed, and may be truncated there.
            entry.reads = entry.changes = entry.pair;
            continue;
        }
        if (chain == ChainOrder::kMixedState)
        {
            // Every channel is a map on the operator: one site is updated in place, two are decomposed.
            entry.starts = entry.changes = true;
            entry.reads = entry.pair;
        }
        else
        {
            // Only what is drawn from the state touches the chain as more than a gate: a Pauli part is
            // one-qubit gates, a mixture of unitaries on two qubits one gate.
            const auto &channel = static_cast<const QuantumChannelOperation<Time> &>(*op).GetChannel();
            Channel::ThermalMixture mixture;
            for (const auto &factor : channel.GetFactors())
                if (factor.kind == Channel::FactorKind::kKraus ||
                    (factor.kind == Channel::FactorKind::kThermal && !Channel::MixtureOf(factor.thermal, mixture)))
                    entry.starts = entry.reads = entry.changes = true;
                else if (factor.kind == Channel::FactorKind::kUnitaryMixture && factor.qubits.size() == 2)
                    entry.reads = entry.changes = true;
        }
        anyStart = anyStart || entry.starts;
    }
    if (!anyStart)
        return circuit;

    constexpr size_t none = std::numeric_limits<size_t>::max();
    typename Circuit<Time>::OperationsVector ordered;
    ordered.reserve(count);
    std::vector<std::vector<size_t>> queues(width);
    std::vector<size_t> heads(width);
    size_t begin = 0;
    while (begin < count)
    {
        size_t end = begin;
        while (end < count && !entries[end].barrier)
            ++end;
        for (auto &queue : queues)
            queue.clear();
        std::fill(heads.begin(), heads.end(), size_t{0});
        for (size_t i = begin; i < end; ++i)
        {
            queues[entries[i].low].push_back(i);
            if (entries[i].pair)
                queues[entries[i].high].push_back(i);
        }
        // Where the chain is in its canonical form: to the left of bond `left`, and from bond `right` on to
        // the right. Whatever precedes the segment (the start, a measurement, a reset) leaves all of it so,
        // and it stays so until an operation takes it off.
        size_t left = width, right = 0;
        bool offForm = false;
        for (size_t placed = begin; placed < end; ++placed)
        {
            size_t best = none, bestCost = 0;
            for (size_t q = 0; q < width; ++q)
            {
                if (heads[q] >= queues[q].size())
                    continue;
                const size_t index = queues[q][heads[q]];
                const Entry &entry = entries[index];
                if (entry.pair && (queues[entry.low][heads[entry.low]] != index || queues[entry.high][heads[entry.high]] != index))
                    continue; // another operation comes first on one of its qubits
                size_t cost = 0;
                if (offForm && entry.reads)
                    cost = (entry.low > left ? entry.low - left : 0) + (right > entry.high ? right - entry.high : 0);
                if (best == none || cost < bestCost || (cost == bestCost && index < best))
                {
                    best = index;
                    bestCost = cost;
                }
            }
            const Entry &entry = entries[best];
            ordered.push_back(operations[best]);
            ++heads[entry.low];
            if (entry.pair)
                ++heads[entry.high];
            offForm = offForm || entry.starts;
            if (!offForm)
                continue;
            if (entry.reads)
            {
                // The form was restored up to its sites.
                left = std::max(left, entry.low);
                right = std::min(right, entry.high);
            }
            if (entry.changes)
            {
                if (entry.pair)
                {
                    // Decomposed at its bond: canonical on both sides of it.
                    left = entry.low + 1;
                    right = entry.low;
                }
                else
                {
                    left = std::min(left, entry.low);
                    right = std::max(right, entry.low);
                }
            }
        }
        if (end < count)
            ordered.push_back(operations[end]);
        begin = end + 1;
    }
    return std::make_shared<Circuit<Time>>(ordered);
}

/**
 * @brief The layout along a chain of a circuit whose two-qubit operations form lines.
 *
 * A tensor chain holds its qubits in a row, and an operation on two qubits
 * that are not neighbours there has to be routed with swaps. Which qubit sits
 * where is free before the first gate: the register is a product state, so a
 * layout is a relabelling and costs nothing.
 *
 * A circuit laid out on a device acts on the qubits of the device, and a line
 * through a device does not run in the order of their indices. When every
 * qubit has at most two partners in the two-qubit operations of the circuit
 * and the partners close no ring, the qubits form lines, and in the order
 * along them every such operation acts on neighbours: no swap is needed, and
 * no layout can cost less. This returns that order. The lines follow each
 * other by their smallest qubit and start from their end with the smaller
 * index; a qubit without a partner is a line of its own. A chain that already
 * runs in index order therefore keeps it.
 *
 * Nothing is returned when the circuit needs no other layout than the index
 * order or has none without swaps: a qubit with three partners, a ring, an
 * operation on more than two qubits, a nested circuit. Nor when a SWAP gate
 * is present and the backend relabels its qubits there instead of applying
 * it, since the layout then changes along the circuit.
 *
 * @param circuit The circuit.
 * @param qubits The size of the register.
 * @param relabelingSwaps Whether the backend relabels its qubits on a SWAP gate.
 * @return The position of every qubit of the register, or nothing.
 */
template <typename Time = Types::time_type>
std::vector<long long int> ChainLine(const std::shared_ptr<Circuit<Time>> &circuit, size_t qubits, bool relabelingSwaps = false)
{
    constexpr size_t none = std::numeric_limits<size_t>::max();
    if (!circuit || qubits < 3)
        return {};
    std::vector<std::array<size_t, 2>> partners(qubits, std::array<size_t, 2>{none, none});
    bool offIndexOrder = false;
    for (const auto &op : circuit->GetOperations())
    {
        const auto type = op->GetType();
        // These act on their qubits one at a time.
        if (type == OperationType::kMeasurement || type == OperationType::kConditionalMeasurement || type == OperationType::kReset ||
            type == OperationType::kDelay)
            continue;
        if (type == OperationType::kComposite)
            return {};
        const auto targets = op->AffectedQubits();
        if (targets.size() < 2)
            continue;
        if (targets.size() > 2)
            return {};
        const size_t a = static_cast<size_t>(targets[0]), b = static_cast<size_t>(targets[1]);
        if (a == b || a >= qubits || b >= qubits)
            return {};
        if (relabelingSwaps)
        {
            const auto gate = std::dynamic_pointer_cast<IQuantumGate<Time>>(op);
            if (gate && gate->GetGateType() == QuantumGateType::kSwapGateType)
                return {};
        }
        if (partners[a][0] == b || partners[a][1] == b)
            continue;
        if (partners[a][1] != none || partners[b][1] != none)
            return {}; // a third partner
        partners[a][partners[a][0] == none ? 0 : 1] = b;
        partners[b][partners[b][0] == none ? 0 : 1] = a;
        offIndexOrder = offIndexOrder || (a > b ? a - b : b - a) != 1;
    }
    if (!offIndexOrder)
        return {};

    // The other end of the line of `start`, walking away from it through `next`; `none` if the walk comes back.
    const auto walk = [&](size_t start, size_t next) {
        size_t previous = start, current = next;
        while (current != none && current != start)
        {
            const size_t onward = partners[current][0] == previous ? partners[current][1] : partners[current][0];
            if (onward == none)
                return current;
            previous = current;
            current = onward;
        }
        return current == start ? none : start;
    };

    std::vector<long long int> positions(qubits, -1);
    long long int position = 0;
    for (size_t q = 0; q < qubits; ++q)
    {
        if (positions[q] >= 0)
            continue;
        // The two ends of the line of q. A qubit with one partner, or none, is an end itself.
        size_t ends[2] = {q, q};
        for (int side = 0; side < 2; ++side)
            if (partners[q][side] != none)
            {
                ends[side] = walk(q, partners[q][side]);
                if (ends[side] == none)
                    return {}; // a ring
            }
        size_t previous = none, current = std::min(ends[0], ends[1]);
        while (current != none)
        {
            positions[current] = position++;
            const size_t onward = partners[current][0] != previous ? partners[current][0] : partners[current][1];
            previous = current;
            current = onward != none && positions[onward] < 0 ? onward : none;
        }
    }
    return positions;
}

} // namespace Circuits

#endif
