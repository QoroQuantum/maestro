#pragma once

#include "SimulatorConfig.h"
#include <optional>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace MaestroExecution
{
// A completed low-rank round is a sample of the actual work for a repeated
// measurement circuit. Keep its gate sequence independent of the caller's circuit.
class LowBondCheckpointRound
{
    using CircuitPtr = std::shared_ptr<Circuits::Circuit<double>>;
    using Gate = std::tuple<Circuits::QuantumGateType, Types::qubit_t, Types::qubit_t>;
    static constexpr size_t maxPrefixOperations = 256;
    // A low first-round peak is evidence for short repeats only. Projecting an
    // ancilla can still leave growing entanglement between the data qubits.
    static constexpr size_t maxRepeatedRounds = 4;
    std::vector<Gate> gates_;

    static std::optional<Gate> FixedClifford(const Circuits::IOperation<double> &op)
    {
        const auto *gate = dynamic_cast<const Circuits::IQuantumGate<double> *>(&op);
        if (op.GetType() != Circuits::OperationType::kGate || !gate || !gate->IsClifford() || gate->GetGateType() == Circuits::QuantumGateType::kNone ||
            !gate->GetParams().empty())
            return std::nullopt;
        const auto width = gate->GetNumQubits();
        if (width != 1 && width != 2)
            return std::nullopt;
        const auto a = gate->GetQubit(0);
        return Gate{gate->GetGateType(), a, width == 1 ? a : gate->GetQubit(1)};
    }

  public:
    static LowBondCheckpointRound Capture(const SimulatorConfig &config, const CircuitPtr &prefix, size_t peakBond, bool prefixLookahead)
    {
        using namespace Simulators;
        // At peak four, replaying a matched syndrome round cost less than
        // forecasting its swaps on both QCSim chains. A cap hit can hide growth.
        // Bound the extra work and storage even for long, entirely separable prefixes.
        if (!prefix || prefix->size() > maxPrefixOperations || !peakBond || peakBond > 4 || prefixLookahead || config.simulator_type != SimulatorType::kQCSim ||
            (config.simulation_type != SimulationType::kMatrixProductState && config.simulation_type != SimulationType::kMatrixProductOperator) ||
            config.disable_optimized_swapping || config.lookahead_depth >= 0 || config.mps_routing_strategy || config.mps_routing_virtual_bonds ||
            config.mps_routing_truncation_weight)
            return {};
        for (const char *key : {RoutingConfig::kStrategy, RoutingConfig::kVirtualBonds, RoutingConfig::kLegacyCap, RoutingConfig::kTruncationWeight})
            if (config.native_options.count(key))
                return {};
        if (const size_t cap = ConfiguredBondCap(config); cap && peakBond >= cap)
            return {};

        LowBondCheckpointRound round;
        bool hasPair = false;
        for (const auto &op : prefix->GetOperations())
        {
            if (!op)
                return {};
            if (op->GetType() == Circuits::OperationType::kNoOp || op->GetType() == Circuits::OperationType::kDelay ||
                (round.gates_.empty() && op->GetType() == Circuits::OperationType::kReset))
                continue;
            const auto gate = FixedClifford(*op);
            if (!gate)
                return {};
            hasPair = hasPair || std::get<1>(*gate) != std::get<2>(*gate);
            round.gates_.push_back(*gate);
        }
        if (!hasPair)
            return {};
        return round;
    }

    bool Repeats(const CircuitPtr &suffix) const
    {
        if (gates_.empty() || !suffix || suffix->size() <= gates_.size() || suffix->size() > (maxRepeatedRounds + 1) * maxPrefixOperations)
            return false;
        size_t next = 0;
        size_t completed = 0;
        std::unordered_set<Types::qubit_t> projected;
        for (const auto &op : suffix->GetOperations())
        {
            if (!op)
                return false;
            const auto type = op->GetType();
            if (type == Circuits::OperationType::kNoOp || type == Circuits::OperationType::kDelay)
                continue;
            if (type == Circuits::OperationType::kMeasurement || type == Circuits::OperationType::kReset)
            {
                if (next)
                {
                    if (next != gates_.size())
                        return false;
                    next = 0;
                    projected.clear();
                }
                const auto qubits = op->AffectedQubits();
                projected.insert(qubits.begin(), qubits.end());
                continue;
            }
            if (next == gates_.size() || completed == maxRepeatedRounds)
                return false;
            const auto gate = FixedClifford(*op);
            if (!gate || *gate != gates_[next])
                return false;
            const auto &[kind, a, b] = *gate;
            // Each repeated interaction must involve a site projected since
            // the last round. An unrelated readout does not bound its cost.
            if (a != b && !projected.count(a) && !projected.count(b))
                return false;
            ++next;
            if (next == gates_.size())
                ++completed;
        }
        return completed > 0 && (next == 0 || next == gates_.size());
    }
};
} // namespace MaestroExecution
