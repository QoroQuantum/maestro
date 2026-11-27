#pragma once

#include "NoisePolicy.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

namespace MaestroExecution
{
inline bool UsesChannelSamplingStatevector(const SimulatorConfig &config)
{
    const auto supports = [](Simulators::SimulatorType type, Simulators::SimulationType method) {
        return method == Simulators::SimulationType::kStatevector && Simulators::SamplesQuantumChannels(type, method);
    };
    if (HasCandidateList(config))
        return std::all_of(config.optimization_candidates.begin(), config.optimization_candidates.end(),
                           [&](const auto &candidate) { return supports(candidate.first, candidate.second); });
    return supports(config.simulator_type, config.simulation_type);
}

// A direct terminal sampler uses one known register. Other methods, selection candidates and measurement-cone
// reduction keep their existing dispatch. A named composite retains its factorized register.
inline bool AllowsTerminalPopulationSampling(const SimulatorConfig &config)
{
    const auto &policy = EffectiveCountsPolicy(config);
    const auto sampling = config.native_options.find("sampling_policy");
    return config.terminal_channel_policy == "auto" && !UsesWrittenMeasurementOrder(config) && UsesChannelSamplingStatevector(config) &&
           !HasCandidateList(config) && !config.enable_sampling_reduction && (policy == "auto" || policy == "multinomial_v1") &&
           (sampling == config.native_options.end() || sampling->second != "legacy");
}

// A pre-drawn model can be recovered as channels when all its randomness is
// terminal. Avoid building a second channel-bearing circuit for common models
// that already add noise before another gate on the same wire. Other models get the full check
// on the injected circuit.
inline RecordCircuitPtr RecoverTerminalNoise(const noise::NoiseInjectionPlan &plan, NoiseLayer layer)
{
    const auto &model = plan.model();
    if (layer != NoiseLayer::kCoherent)
    {
        std::unordered_set<Types::qubit_t> noisy_wires;
        for (const auto &op : plan.circuit()->GetOperations())
        {
            if (op->GetType() != Circuits::OperationType::kGate)
                continue;
            const auto targets = op->AffectedQubits();
            if (std::any_of(targets.begin(), targets.end(), [&](auto q) { return noisy_wires.count(q); }))
                return {};
            if (op->IsNoiseless())
                continue;
            const bool multi = targets.size() > 1;
            for (const auto q : targets)
            {
                const auto *all = model.get(static_cast<int>(q));
                const auto *gate = multi ? model.get_2q_gate_noise(static_cast<int>(q)) : model.get_1q_gate_noise(static_cast<int>(q));
                if ((all && all->total() > 0.) || (gate && gate->total() > 0.) || model.get_phase_damping_flip_probability(static_cast<int>(q)) > 0. ||
                    model.get_thermal_relaxation_params(static_cast<int>(q), multi))
                    noisy_wires.insert(q);
            }
            if (targets.size() == 2 && model.get_2q_depolarizing(static_cast<int>(targets[0]), static_cast<int>(targets[1])) > 0.)
                noisy_wires.insert(targets.begin(), targets.end());
        }
    }
    std::mt19937 unused;
    return InjectPlanned(layer, plan, unused, true);
}

struct TerminalChannels
{
    std::shared_ptr<Circuits::Circuit<double>> prefix;
    std::vector<std::shared_ptr<Circuits::QuantumChannelOperation<>>> channels;
};

// Accepted custom Kraus input can depart from trace preservation by more than roundoff.
// Normalizing its trajectories can then change the marginal of an entangled spectator.
inline bool PreservesTerminalTrace(const Simulators::QuantumChannel &channel)
{
    const auto &super = channel.Superoperator();
    const size_t dim = size_t{1} << channel.GetNumberOfQubits();
    for (size_t input = 0; input < dim * dim; ++input)
    {
        std::complex<double> trace = 0.0;
        for (size_t output = 0; output < dim; ++output)
            trace += super(output * (dim + 1), input);
        if (std::abs(trace - double(input % dim == input / dim)) > 64 * std::numeric_limits<double>::epsilon())
            return false;
    }
    return true;
}

// Collect channels that commute past the remaining operations onto the final measurements or observable.
// A retained gate, reset, delay or channel blocks its complete support, including controls. Both the
// prefix and extracted channels stay in their original relative order. Keep classical destinations,
// repeated writes and readout errors in the trailing measurements. Dynamic and opaque circuits keep
// their ordinary executor.
inline TerminalChannels SplitTerminalChannels(const std::shared_ptr<Circuits::Circuit<double>> &circuit)
{
    using Type = Circuits::OperationType;
    if (!circuit)
        return {};
    const auto &ops = circuit->GetOperations();
    auto end = ops.end();
    while (end != ops.begin() && ((*(end - 1))->GetType() == Type::kMeasurement || (*(end - 1))->GetType() == Type::kNoOp))
        --end;
    const auto first = std::find_if(ops.begin(), end, [](const auto &op) { return op->GetType() == Type::kQuantumChannel; });
    if (first == end)
        return {};

    std::unordered_set<Types::qubit_t> blocked;
    std::vector<bool> extracted(end - first);
    TerminalChannels result;
    for (auto it = end; it != ops.begin();)
    {
        const auto &op = *--it;
        switch (op->GetType())
        {
        case Type::kGate: {
            const auto *gate = dynamic_cast<const Circuits::IQuantumGate<double> *>(op.get());
            if (!gate || gate->GetGateType() == Circuits::QuantumGateType::kNone)
                return {};
            break;
        }
        case Type::kNoOp:
            continue;
        case Type::kQuantumChannel:
        case Type::kReset:
        case Type::kDelay:
            break;
        default:
            return {};
        }
        if (it < first)
            continue;
        const auto targets = op->AffectedQubits();
        if (op->GetType() == Type::kQuantumChannel && std::none_of(targets.begin(), targets.end(), [&](auto q) { return blocked.count(q); }))
        {
            extracted[it - first] = true;
            result.channels.push_back(std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(op));
        }
        else
            blocked.insert(targets.begin(), targets.end());
    }
    if (result.channels.empty())
        return {};
    std::reverse(result.channels.begin(), result.channels.end());
    Circuits::Circuit<double>::OperationsVector prefix;
    prefix.reserve(ops.size() - result.channels.size());
    prefix.insert(prefix.end(), ops.begin(), first);
    bool crossed = false;
    for (auto it = first; it != end; ++it)
    {
        if (extracted[it - first])
            crossed = true;
        else
        {
            if (crossed && (*it)->GetType() == Type::kQuantumChannel &&
                !PreservesTerminalTrace(std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(*it)->GetChannel()))
                return {};
            prefix.push_back(*it);
        }
    }
    prefix.insert(prefix.end(), end, ops.end());
    result.prefix = std::make_shared<Circuits::Circuit<double>>(prefix);
    return result;
}
} // namespace MaestroExecution
