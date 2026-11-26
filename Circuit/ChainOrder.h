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

// Starting from zero, a qubit touched only by local gates up to its last
// use can be measured together with the terminal readouts. Plan this before
// layout so the chain is prepared just as if those readouts were written last.
// Keep their relative order, including repeated qubits and classical writes.
template <typename Time>
std::shared_ptr<Circuit<Time>> IsolatedMpsTerminalOrder(const std::shared_ptr<Circuit<Time>> &circuit, size_t width, Simulators::SimulatorType type)
{
    if (type != Simulators::SimulatorType::kQCSim
#ifndef NO_QISKIT_AER
        && type != Simulators::SimulatorType::kQiskitAer
#endif
    )
        return circuit;
    const auto &ops = circuit->GetOperations();
    const auto isMeasurement = [](const auto &op) { return op->GetType() == OperationType::kMeasurement; };
    const auto lastGate = std::find_if_not(ops.rbegin(), ops.rend(), isMeasurement).base();
    if (std::none_of(ops.begin(), lastGate, isMeasurement))
        return circuit;

    std::vector<bool> coupled(width), measured(width);
    for (auto it = ops.begin(); it != ops.end(); ++it)
    {
        const auto &op = *it;
        if (isMeasurement(op))
        {
            if (!dynamic_cast<const MeasurementOperation<Time> *>(op.get()))
                return circuit;
            if (it < lastGate)
                for (const auto q : op->AffectedQubits())
                {
                    if (q >= width || coupled[q])
                        return circuit;
                    measured[q] = true;
                }
            continue;
        }
        if (op->GetType() != OperationType::kGate)
            return circuit;
        // Opaque operations can change normalization or use classical state.
        const auto *gate = dynamic_cast<const IQuantumGate<Time> *>(op.get());
        if (!gate || gate->GetGateType() == QuantumGateType::kNone)
            return circuit;
        const auto targets = op->AffectedQubits();
        for (const auto q : targets)
        {
            if (q >= width || measured[q])
                return circuit;
            if (targets.size() > 1)
                coupled[q] = true;
        }
    }
    auto ordered = ops;
    std::stable_partition(ordered.begin(), ordered.end(), [&](const auto &op) { return !isMeasurement(op); });
    return std::make_shared<Circuit<Time>>(ordered, circuit->GetGlobalPhase());
}

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
 * The result shares the operations of the circuit and describes the same map
 * in exact arithmetic. A seeded MPS trajectory draws branches in the chosen
 * order. If a bond truncates, a different schedule can discard a different
 * subspace. Callers that require written-order truncation select kNone.
 *
 * @param circuit The circuit.
 * @param chain The kind of chain that runs it.
 * @param positions The position of every qubit on the chain, or null for the index.
 * @param preserveExplicitDraws Keep the relative order of channel draws if the circuit contains explicit channels.
 * @return The reordered circuit, or `circuit` itself.
 */
template <typename Time = Types::time_type>
std::shared_ptr<Circuit<Time>> ChainOrdered(const std::shared_ptr<Circuit<Time>> &circuit, ChainOrder chain = ChainOrder::kSampledChannels,
                                            const std::vector<long long int> *positions = nullptr, bool preserveExplicitDraws = false)
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
        bool channel = false;
        bool reads = false;   // needs the canonical form around its sites: pays for the travel
        bool changes = false; // leaves the chain off that form away from its sites
        bool starts = false;  // takes a canonical chain off its form
    };

    std::vector<Entry> entries(count);
    size_t width = 0;
    bool anyStart = false;
    bool preserveChannelOrder = false;
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
        entry.channel = true;
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
            const auto &instruction = static_cast<const QuantumChannelOperation<Time> &>(*op);
            preserveChannelOrder = preserveChannelOrder || (preserveExplicitDraws && !instruction.IsFromNoiseModel());
            const auto &channel = instruction.GetChannel();
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
        size_t nextChannel = begin;
        const auto advanceChannel = [&] {
            while (nextChannel < end && !entries[nextChannel].channel)
                ++nextChannel;
        };
        if (preserveChannelOrder)
            advanceChannel();
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
                if (preserveChannelOrder && entry.channel && index != nextChannel)
                    continue;
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
            if (preserveChannelOrder && entry.channel)
            {
                ++nextChannel;
                advanceChannel();
            }
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
    return std::make_shared<Circuit<Time>>(ordered, circuit->GetGlobalPhase());
}

// From a product state, a controlled gate can at most double the Schmidt rank
// on its bond (its operator is a sum of two tensor products). A one-site
// unitary, projective measurement or individual Kraus branch cannot increase
// any rank. Counting all crossings bounds every prefix, including any order
// preserving each wire's dependencies. Work on the emitted, noise-injected
// instructions and require physical neighbours so no routing swaps are omitted.
template <typename Time> bool FitsMpsBondCap(const Circuit<Time> &circuit, size_t qubits, size_t cap, bool fusionMayRelabel)
{
    if (!qubits || !cap)
        return false;
    std::vector<size_t> ranks(qubits - 1, 1), lastBond;
    if (fusionMayRelabel)
        lastBond.assign(qubits, qubits);
    for (const auto &op : circuit.GetOperations())
    {
        const auto targets = op->AffectedQubits();
        if (std::any_of(targets.begin(), targets.end(), [qubits](auto q) { return q >= qubits; }))
            return false;
        if (op->GetType() == OperationType::kMeasurement || op->GetType() == OperationType::kDelay || op->GetType() == OperationType::kNoOp)
            continue;
        if (op->GetType() == OperationType::kQuantumChannel)
        {
            if (targets.size() != 1 || !dynamic_cast<const QuantumChannelOperation<Time> *>(op.get()))
                return false;
            continue;
        }
        if (op->GetType() != OperationType::kGate)
            return false;
        const auto *gate = dynamic_cast<const IQuantumGate<Time> *>(op.get());
        if (!gate || gate->GetGateType() == QuantumGateType::kNone)
            return false;
        if (targets.size() == 1)
            continue;
        if (targets.size() != 2)
            return false;
        const size_t low = std::min(targets[0], targets[1]), high = std::max(targets[0], targets[1]);
        if (high - low != 1)
            return false;
        if (!IsControlledTwoQubitGate(gate->GetGateType()))
            return false; // Includes SWAP, which may relabel the sites.
        if (fusionMayRelabel)
        {
            // A two-site fused block with one controlled gate and local
            // factors still has rank <= 2 and cannot be SWAP (rank 4).
            // With repeated crossings in one block its result could relabel
            // the chain, invalidating all subsequent physical-cut counts.
            if (lastBond[low] == low && lastBond[high] == low)
                return false;
            lastBond[low] = lastBond[high] = low;
        }
        const size_t bits = std::min(high, qubits - high);
        if (bits < std::numeric_limits<size_t>::digits && (size_t{1} << bits) <= cap)
            continue; // the smaller half of this physical cut fits regardless of gates
        if (ranks[low] > cap / 2)
            return false;
        ranks[low] *= 2;
    }
    return true;
}

// Call only for fresh, fixed-backend shot executions. Ordering the circuit
// itself keeps the simulator's conservative policy for all other circuits,
// restored checkpoints and later measurements on an existing state.
template <typename Time>
std::shared_ptr<Circuit<Time>> ScheduleMpsShots(const std::shared_ptr<Circuit<Time>> &circuit, size_t qubits, const Simulators::ISimulator &sim,
                                                const std::unordered_map<std::string, std::string> &options, bool identityLayout,
                                                bool *shareResetWires = nullptr)
{
    using Policy = Simulators::ShotPrefixPolicy;
    if (shareResetWires)
        *shareResetWires = false;
    const auto method = sim.GetSimulationType();
    const auto type = sim.GetType();
    const auto cap = options.find("matrix_product_state_max_bond_dimension");
    if (!circuit || method != Simulators::SimulationType::kMatrixProductState ||
        (type != Simulators::SimulatorType::kQCSim
#ifndef NO_QISKIT_AER
         && type != Simulators::SimulatorType::kQiskitAer
#endif
         ) ||
        !Policy::TruncatesChain(type, method, options, qubits))
        return circuit;
    const auto &original = circuit->GetOperations();
    const auto firstDeferred = std::find_if(original.begin(), original.end(), [](const auto &op) {
        return op->GetType() == OperationType::kMeasurement || op->GetType() == OperationType::kQuantumChannel || op->GetType() == OperationType::kReset;
    });
    if (std::none_of(firstDeferred, original.end(), [](const auto &op) { return op->GetType() == OperationType::kGate; }))
        return circuit;

    // A reset discards its outcome. With only gates and resets before the terminal
    // reads, independent wires can use the normal QCSim schedule. As with gate
    // fusion, the chosen order can affect the approximation at a finite bond cap.
    auto scheduled = circuit;
    if (!identityLayout && type == Simulators::SimulatorType::kQCSim && (*firstDeferred)->GetType() == OperationType::kReset)
    {
        const auto readout = std::find_if(firstDeferred, original.end(), [](const auto &op) { return op->GetType() == OperationType::kMeasurement; });
        const bool wireLocalResets =
            std::all_of(original.begin(), readout,
                        [](const auto &op) {
                            if (const auto *reset = dynamic_cast<const Reset<Time> *>(op.get()))
                                return !reset->GetQubits().empty();
                            const auto *gate = dynamic_cast<const IQuantumGate<Time> *>(op.get());
                            return op->GetType() == OperationType::kGate && gate && gate->GetGateType() != QuantumGateType::kNone;
                        }) &&
            std::all_of(readout, original.end(), [](const auto &op) { return dynamic_cast<const MeasurementOperation<Time> *>(op.get()); });
        if (wireLocalResets)
        {
            // The layout optimizer normally orders work in layers. Keep terminal
            // classical writes in sequence, including repeated destinations.
            Circuit<Time> body(typename Circuit<Time>::OperationsVector(original.begin(), readout), circuit->GetGlobalPhase());
            body.ConvertForCutting(); // Use the same gate decomposition as the layout and lookahead planners.
            scheduled = Circuit<Time>::LayersToCircuit(body.ToMultipleQubitsLayersNoClone());
            for (auto it = readout; it != original.end(); ++it)
                scheduled->AddOperation(*it);
            // Choose the layout from these layers. Only then should a worker
            // lift the shared gates into its prefix and refresh its routing window.
            if (shareResetWires)
                *shareResetWires = true;
            return scheduled;
        }
    }
    const bool rankBounded = identityLayout && cap != options.end() && !Policy::TruncatesChain(type, method, options, qubits, size_t{0}) &&
                             FitsMpsBondCap(*circuit, qubits, std::stoull(cap->second), sim.SwapGatesRelabelQubits());
    if (!rankBounded)
    {
        // A noise model specifies errors on each gate. Schedule those local updates
        // along the chain, keeping every wire dependency and measurement barrier.
        // Truncation can depend on the schedule. Explicit draws retain their
        // circuit order unless the stronger rank bound above succeeds.
        if (type == Simulators::SimulatorType::kQCSim && circuit->HasQuantumChannels() && std::all_of(original.begin(), original.end(), [](const auto &op) {
                if (op->GetType() != OperationType::kQuantumChannel)
                    return true;
                const auto *channel = dynamic_cast<const QuantumChannelOperation<Time> *>(op.get());
                return channel && channel->IsFromNoiseModel();
            }))
            return ChainOrdered(circuit, ChainOrder::kSampledChannels);
        return circuit;
    }

    // The QCSim gate and trajectory caches merge overlapping supports up to
    // two sites on MPS. A product of the certified local factors on those
    // sites obeys the same cut bounds as the individual instructions.
    if (type == Simulators::SimulatorType::kQCSim && circuit->HasQuantumChannels())
        scheduled = ChainOrdered(circuit, ChainOrder::kSampledChannels, nullptr, true);
    OperationState dry;
    const auto mask = scheduled->PlanNonMeasurements(dry, {false, false});
    if (std::find(mask.begin(), mask.end(), true) == mask.end())
        return scheduled;
    const auto &ops = scheduled->GetOperations();
    const size_t offset = ops.size() - mask.size();
    // Even a one-branch channel can consume an RNG word in an adapter. Keep
    // it behind earlier draws rather than lifting it into the shared head.
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i] && ops[offset + i]->GetType() == OperationType::kQuantumChannel)
            return scheduled;
    if (!scheduled->HasDrawnChannels())
    {
        // A terminal sampler may move a proper subset of measured sites
        // together using swaps. Those cuts were not part of the gate bound.
        std::vector<bool> measured(qubits);
        for (const auto &op : ops)
            if (op->GetType() == OperationType::kMeasurement)
                for (const auto q : op->AffectedQubits())
                    measured[q] = true;
        if (std::find(measured.begin(), measured.end(), false) != measured.end())
            return circuit;
    }
    typename Circuit<Time>::OperationsVector ordered(ops.begin(), ops.begin() + offset);
    ordered.reserve(ops.size());
    for (bool shared : {true, false})
        for (size_t i = 0; i < mask.size(); ++i)
            if (mask[i] == shared)
                ordered.push_back(ops[offset + i]);
    return std::make_shared<Circuit<Time>>(ordered, circuit->GetGlobalPhase());
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
