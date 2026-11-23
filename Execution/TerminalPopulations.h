#pragma once

#include "../Simulators/Core/FactoryBackends.h"
#include "../Utils/RandomStream.h"
#include "../Utils/Sampling/Multinomial.h"
#include "TerminalChannels.h"

#include <array>
#include <map>
#include <unordered_set>

namespace MaestroExecution
{
// Local transitions of computational-basis populations. If the channel maps a coherence into a population,
// the incoming bit string is insufficient and the caller keeps its quantum trajectories.
class TerminalPopulations
{
  public:
    bool Compile(const TerminalChannels &layer, const Types::qubits_vector &measured)
    {
        // Walk backwards from the recorded measurements. A later two-qubit transition can make an otherwise
        // unmeasured input matter. An unrelated trace-preserving channel leaves the measured marginal unchanged.
        std::unordered_set<Types::qubit_t> needed(measured.begin(), measured.end());
        for (auto it = layer.channels.rbegin(); it != layer.channels.rend(); ++it)
        {
            const auto targets = (*it)->AffectedQubits();
            if (std::none_of(targets.begin(), targets.end(), [&](auto q) { return needed.count(q); }))
            {
                // Validated custom input can depart from trace preservation by more than roundoff. A normalized
                // trajectory on an entangled spectator can then change this marginal.
                if (!PreservesTrace((*it)->GetChannel()))
                    return false;
                continue;
            }
            const size_t previous = transitions_.size();
            if (!Append((*it)->GetChannel(), targets))
                return false;
            if (transitions_.size() != previous)
                needed.insert(targets.begin(), targets.end());
        }
        std::reverse(transitions_.begin(), transitions_.end());
        return true;
    }

    bool IsIdentity() const
    {
        return transitions_.empty();
    }

    // Split counts of the SAME incoming joint bit string, including all channel targets. The binomial splitting
    // has the law of independent per-shot draws. Projecting at the end keeps spectator and repeated-wire correlations.
    std::unordered_map<std::vector<bool>, Types::qubit_t> Sample(Simulators::ISimulator &simulator, const Types::qubits_vector &measured, size_t shots,
                                                                 uint64_t channel_seed) const
    {
        auto qubits = qubits_;
        std::vector<size_t> projection;
        for (const auto q : measured)
            projection.push_back(Position(qubits, q));
        std::map<std::vector<bool>, size_t> counts;
        for (const auto &[bits, count] : Simulators::FactoryBackends::SamplePreparedStatevector(simulator, qubits, shots))
            counts.emplace(bits, count);
        std::mt19937_64 engine(channel_seed);
        const auto uniform = [&] { return Utils::RandomStream::Uniform(engine); };
        for (const auto &transition : transitions_)
        {
            std::map<std::vector<bool>, size_t> next;
            for (const auto &[bits, count] : counts)
            {
                size_t input = 0;
                for (size_t i = 0; i < transition.positions.size(); ++i)
                    input |= size_t(bits[transition.positions[i]]) << i;
                size_t remaining = count;
                for (const auto &[output, conditional] : transition.columns[input])
                {
                    const size_t drawn = Utils::Sampling::BinomialCount(remaining, conditional, uniform);
                    if (drawn)
                    {
                        auto updated = bits;
                        for (size_t i = 0; i < transition.positions.size(); ++i)
                            updated[transition.positions[i]] = (output >> i) & 1;
                        next[std::move(updated)] += drawn;
                        remaining -= drawn;
                    }
                    if (!remaining)
                        break;
                }
            }
            counts.swap(next);
        }
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        std::vector<bool> output(projection.size());
        for (const auto &[bits, count] : counts)
        {
            for (size_t i = 0; i < projection.size(); ++i)
                output[i] = bits[projection[i]];
            result[output] += count;
        }
        return result;
    }

  private:
    static constexpr double probabilityTolerance = 64 * std::numeric_limits<double>::epsilon();

    static bool PreservesTrace(const Simulators::QuantumChannel &channel)
    {
        const auto &super = channel.Superoperator();
        const size_t dim = size_t{1} << channel.GetNumberOfQubits();
        for (size_t input = 0; input < dim * dim; ++input)
        {
            std::complex<double> trace = 0.0;
            for (size_t output = 0; output < dim; ++output)
                trace += super(output * (dim + 1), input);
            if (std::abs(trace - double(input % dim == input / dim)) > probabilityTolerance)
                return false;
        }
        return true;
    }

    bool Append(const Simulators::QuantumChannel &channel, const Types::qubits_vector &targets)
    {
        if (targets.empty() || targets.size() > 2)
            return false;
        const size_t dim = size_t{1} << targets.size();
        const auto &super = channel.Superoperator();
        Transition transition;
        bool changes = false;
        for (size_t input = 0; input < dim; ++input)
        {
            auto &column = transition.columns[input];
            for (size_t output = 0; output < dim; ++output)
            {
                const auto row = output * (dim + 1);
                for (size_t other = 0; other < dim; ++other)
                    if (other != input && super(row, input + dim * other) != std::complex<double>{})
                        return false;
                const double probability = super(row, input * (dim + 1)).real();
                if (!(probability >= 0.0))
                    return false;
                if (probability > 0.0)
                {
                    column.emplace_back(output, probability);
                    changes = changes || input != output;
                }
            }
            // Draw rare outcomes first, so the probability of a rare transition survives even when 1-p rounds to one.
            std::sort(column.begin(), column.end(), [](const auto &a, const auto &b) { return a.second < b.second; });
            double remaining = 0.0;
            for (auto it = column.rbegin(); it != column.rend(); ++it)
            {
                remaining += it->second;
                it->second /= remaining;
            }
            if (std::abs(remaining - 1.0) > probabilityTolerance)
                return false;
        }
        if (changes)
        {
            for (const auto q : targets)
                transition.positions.push_back(Position(qubits_, q));
            transitions_.push_back(std::move(transition));
        }
        return true;
    }

    static size_t Position(Types::qubits_vector &qubits, Types::qubit_t q)
    {
        const auto it = std::find(qubits.begin(), qubits.end(), q);
        if (it != qubits.end())
            return static_cast<size_t>(it - qubits.begin());
        qubits.push_back(q);
        return qubits.size() - 1;
    }

    struct Transition
    {
        std::array<std::vector<std::pair<size_t, double>>, 4> columns;
        std::vector<size_t> positions;
    };

    Types::qubits_vector qubits_;
    std::vector<Transition> transitions_;
};
} // namespace MaestroExecution
