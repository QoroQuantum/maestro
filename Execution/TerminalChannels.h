#pragma once

#include "NoisePolicy.h"

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
    return UsesChannelSamplingStatevector(config) && !HasCandidateList(config) && !config.enable_sampling_reduction &&
           (policy == "auto" || policy == "multinomial_v1") && (sampling == config.native_options.end() || sampling->second != "legacy");
}

// A pre-drawn model can be recovered as channels when all its randomness is
// terminal. Avoid building a second channel-bearing circuit for common models
// that already add noise before another gate. Other models get the full check
// on the injected circuit.
inline RecordCircuitPtr RecoverTerminalNoise(const noise::NoiseInjectionPlan &plan, NoiseLayer layer)
{
    const auto &model = plan.model();
    if (layer != NoiseLayer::kCoherent)
    {
        bool earlier_noise = false;
        for (const auto &op : plan.circuit()->GetOperations())
        {
            if (op->GetType() != Circuits::OperationType::kGate)
                continue;
            if (earlier_noise)
                return {};
            if (op->IsNoiseless())
                continue;
            const auto targets = op->AffectedQubits();
            const bool multi = targets.size() > 1;
            for (const auto q : targets)
            {
                const auto *all = model.get(static_cast<int>(q));
                const auto *gate = multi ? model.get_2q_gate_noise(static_cast<int>(q)) : model.get_1q_gate_noise(static_cast<int>(q));
                if ((all && all->total() > 0.) || (gate && gate->total() > 0.) || model.get_phase_damping_flip_probability(static_cast<int>(q)) > 0. ||
                    model.get_thermal_relaxation_params(static_cast<int>(q), multi))
                {
                    earlier_noise = true;
                    break;
                }
            }
            if (targets.size() == 2 && model.get_2q_depolarizing(static_cast<int>(targets[0]), static_cast<int>(targets[1])) > 0.)
                earlier_noise = true;
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

// Remove a contiguous channel layer immediately before the terminal measurements (or the
// observable). Keep the original measurement operations, including their classical destinations
// and readout errors. A measurement in the prefix can condition its state and must stay on the
// ordinary dynamic-circuit path; removing its successor could otherwise make it look terminal.
inline TerminalChannels SplitTerminalChannels(const std::shared_ptr<Circuits::Circuit<double>> &circuit)
{
    using Type = Circuits::OperationType;
    if (!circuit)
        return {};
    const auto &ops = circuit->GetOperations();
    auto end = ops.end();
    while (end != ops.begin() && ((*(end - 1))->GetType() == Type::kMeasurement || (*(end - 1))->GetType() == Type::kNoOp))
        --end;
    auto first = end;
    while (first != ops.begin() && ((*(first - 1))->GetType() == Type::kQuantumChannel || (*(first - 1))->GetType() == Type::kNoOp))
        --first;
    if (first == end)
        return {};

    for (auto it = ops.begin(); it != first; ++it)
        switch ((*it)->GetType())
        {
        case Type::kGate: {
            const auto *gate = dynamic_cast<const Circuits::IQuantumGate<double> *>(it->get());
            if (!gate || gate->GetGateType() == Circuits::QuantumGateType::kNone)
                return {};
            break;
        }
        case Type::kNoOp:
        case Type::kQuantumChannel:
        case Type::kReset:
        case Type::kDelay:
            break;
        default:
            return {};
        }

    TerminalChannels result;
    for (auto it = first; it != end; ++it)
        if ((*it)->GetType() == Type::kQuantumChannel)
            result.channels.push_back(std::static_pointer_cast<Circuits::QuantumChannelOperation<>>(*it));
    if (result.channels.empty())
        return {};
    Circuits::Circuit<double>::OperationsVector prefix(ops.begin(), first);
    prefix.insert(prefix.end(), end, ops.end());
    result.prefix = std::make_shared<Circuits::Circuit<double>>(prefix);
    return result;
}
} // namespace MaestroExecution
