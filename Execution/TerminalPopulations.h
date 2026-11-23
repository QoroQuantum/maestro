#pragma once

#include "../Simulators/Core/FactoryBackends.h"
#include "../Utils/RandomStream.h"
#include "../Utils/Sampling/Multinomial.h"
#include "TerminalChannels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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
                if (!PreservesTerminalTrace((*it)->GetChannel()))
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
        {
            auto sampled = Simulators::FactoryBackends::SamplePreparedStatevector(simulator, qubits, shots);
            // Canonical input order for seeded draws. Move the strings as we go to avoid a second copy of them.
            while (!sampled.empty())
            {
                auto entry = sampled.extract(sampled.begin());
                counts.emplace(std::move(entry.key()), entry.mapped());
            }
        }
        std::mt19937_64 engine(channel_seed);
        const auto uniform = [&] { return Utils::RandomStream::Uniform(engine); };
        std::unordered_map<std::vector<bool>, Types::qubit_t> result;
        std::vector<bool> output(projection.size()), evolved;
        const auto record = [&](const auto &bits, size_t count) {
            for (size_t i = 0; i < projection.size(); ++i)
                output[i] = bits[projection[i]];
            result[output] += count;
        };
        size_t activeShots = shots;
        for (auto it = transitions_.begin(); it != transitions_.end(); ++it)
        {
            // When many strings occur only once or twice, take their remaining transitions in place.
            // Common strings still share binomial draws and coalesce in the next histogram.
            const bool sparse = counts.size() >= Utils::Sampling::smallTableRows && !Utils::Sampling::PreferMultinomial(activeShots, counts.size());
            const bool last = it + 1 == transitions_.end();
            std::map<std::vector<bool>, size_t> next;
            for (const auto &[bits, count] : counts)
            {
                if (sparse && count <= Utils::Sampling::multinomialShotsPerDraw)
                {
                    activeShots -= count;
                    for (size_t shot = 0; shot < count; ++shot)
                    {
                        evolved = bits;
                        for (auto step = it; step != transitions_.end(); ++step)
                            step->Draw(evolved, engine);
                        record(evolved, 1);
                    }
                    continue;
                }
                const size_t input = it->Input(bits);
                size_t remaining = count;
                for (const auto &[value, conditional] : it->columns[input])
                {
                    const size_t drawn = Utils::Sampling::BinomialCount(remaining, conditional, uniform);
                    if (drawn)
                    {
                        auto updated = bits;
                        if (value != input)
                            it->Write(updated, value);
                        if (last)
                            record(updated, drawn);
                        else
                            next[std::move(updated)] += drawn;
                        remaining -= drawn;
                    }
                    if (!remaining)
                        break;
                }
            }
            counts.swap(next);
            if (counts.empty())
                break;
        }
        for (const auto &[bits, count] : counts)
            record(bits, count);
        return result;
    }

  private:
    static constexpr double probabilityTolerance = 64 * std::numeric_limits<double>::epsilon();

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

        size_t Input(const std::vector<bool> &bits) const
        {
            size_t input = 0;
            for (size_t i = 0; i < positions.size(); ++i)
                input |= size_t(bits[positions[i]]) << i;
            return input;
        }

        void Write(std::vector<bool> &bits, size_t value) const
        {
            for (size_t i = 0; i < positions.size(); ++i)
                bits[positions[i]] = (value >> i) & 1;
        }

        void Draw(std::vector<bool> &bits, std::mt19937_64 &engine) const
        {
            const auto input = Input(bits);
            for (const auto &[value, probability] : columns[input])
                if (probability == 1. || Bernoulli(probability, engine))
                {
                    if (input != value)
                        Write(bits, value);
                    break;
                }
        }

        // Compare an independent random binary fraction with p. Small nonzero probabilities can need more than
        // one word of precision. Draw the next word only when the first leaves the comparison undecided.
        static bool Bernoulli(double p, std::mt19937_64 &engine)
        {
            while (p > 0.)
            {
                const double scaled = p * 0x1p64;
                const auto whole = static_cast<uint64_t>(scaled);
                const auto word = engine();
                if (word != whole)
                    return word < whole;
                p = scaled - static_cast<double>(whole);
            }
            return false;
        }
    };

    Types::qubits_vector qubits_;
    std::vector<Transition> transitions_;
};
} // namespace MaestroExecution
