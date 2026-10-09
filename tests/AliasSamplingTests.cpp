#include "../Utils/Sampling/Alias.h"

#include <iostream>
#include <numeric>
#include <random>
#include <string>

namespace
{
void Require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Sampler, class Outcome> void CheckDistribution(const Sampler &sampler, const std::vector<Outcome> &outcomes, const std::vector<double> &weights)
{
    const double total = std::accumulate(weights.begin(), weights.end(), 0.);
    std::vector<size_t> counts(outcomes.size());
    const auto indexOf = [&](double uniform) {
        const auto outcome = sampler.Sample(uniform);
        const auto found = std::find(outcomes.begin(), outcomes.end(), outcome);
        Require(found != outcomes.end(), "sampled an unknown or truncated state");
        const size_t index = found - outcomes.begin();
        Require(weights[index] > 0., "sampled a zero-probability state");
        return index;
    };

    // Midpoints of equal intervals give a deterministic distribution check.
    // Each alias column contributes at most one count of rounding error.
    const size_t shots = 16384 * outcomes.size();
    for (size_t i = 0; i < shots; ++i)
        ++counts[indexOf((i + 0.5) / shots)];
    for (size_t i = 0; i < outcomes.size(); ++i)
        Require(std::abs(counts[i] - weights[i] / total * shots) <= outcomes.size(), "incorrect sampling distribution");

    for (const double uniform : {0., std::nextafter(1., 0.), 1.})
        indexOf(uniform);
}

void Statevectors()
{
    const auto check = [](const std::vector<double> &weights) {
        const double total = std::accumulate(weights.begin(), weights.end(), 0.);
        Eigen::VectorXcd state(weights.size());
        std::vector<size_t> outcomes(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            state[i] = std::sqrt(weights[i] / total);
            outcomes[i] = i;
        }
        CheckDistribution(Utils::Alias(state), outcomes, weights);
    };
    for (const std::vector<double> &weights :
         {std::vector<double>{1.}, {1., 0., 0., 0.}, {0., 0., 0., 1.}, {1., 1., 1., 1.}, {1., 3., 2., 2.}, {1., 3., 0., 0.}, {0., 1., 0., 4., 0., 3.}})
        check(weights);

    std::mt19937 rng(891);
    for (size_t count : {3, 7, 17, 32, 257})
    {
        std::vector<double> weights(count);
        for (auto &weight : weights)
            weight = rng() % 31;
        check(weights);
    }

    // Preserve the statevector constructor's epsilon filtering, including a
    // skipped leading entry that requires a nonidentity outcome mapping.
    const double tiny = std::numeric_limits<double>::epsilon() / 2.;
    std::vector<std::complex<double>> state{std::sqrt(tiny), 0.5, std::sqrt(0.75 - tiny)};
    CheckDistribution(Utils::Alias(state), std::vector<size_t>{0, 1, 2}, {0., 0.25, 0.75});
}

using State = QC::PathIntegral::FastVectorBool;

std::vector<State> PathIntegralStates(size_t width)
{
    std::vector<State> states(4, State(width));
    states[0].set(0, true);
    states[1].set(0, true);
    states[1].set(width - 1, true); // Same low bits, different high word for width > 64.
    states[2].set(31, true);
    if (width > 64)
        states[2].set(63, true);
    if (width > 128)
        states[2].set(127, true);
    return states;
}

void PathIntegralLabels()
{
    const std::vector<double> weights{1., 3., 4., 0.};
    for (size_t width : {33, 40, 63, 64, 65, 130, 1024})
    {
        const auto states = PathIntegralStates(width);
        Utils::PathIntegralAmplitudeMap amplitudes;
        for (size_t i = 0; i < states.size(); ++i)
            amplitudes[states[i]] = std::sqrt(weights[i] / 32.); // Retained mass is only 1/4.

        const Utils::AliasBig wide(amplitudes);
        if (width <= std::numeric_limits<size_t>::digits)
        {
            const Utils::Alias packed(amplitudes);
            std::vector<size_t> outcomes;
            for (const auto &state : states)
                outcomes.push_back(state.getWords()[0]);
            CheckDistribution(packed, outcomes, weights);
            // Both samplers preserve row order and the mapping of a uniform
            // draw to an outcome, including the caller's closed upper endpoint.
            for (size_t i = 0; i <= 8192; ++i)
                Require(packed.Sample(i / 8192.) == wide.Sample(i / 8192.).getWords()[0], "packed/wide sample mismatch");
        }
        amplitudes.clear();
        CheckDistribution(wide, states, weights); // Labels must be owned, not views into the input.
    }

    Utils::PathIntegralAmplitudeMap zeroQubits;
    zeroQubits[State(size_t(0))] = 0.5;
    CheckDistribution(Utils::AliasBig(zeroQubits), std::vector<State>{State(size_t(0))}, {1.});
}

// Exercise the 64-bit construction and sampling path with a small table;
// allocating >2^32 entries is deliberately unnecessary for this regression.
class WideTableProbe : public Utils::AliasBase
{
  public:
    using AliasBase::UseCompactIndices;

    explicit WideTableProbe(const std::vector<double> &weights)
    {
        const double total = std::accumulate(weights.begin(), weights.end(), 0.);
        probabilities.resize(weights.size());
        wideAliases.resize(weights.size());
        BuildAliasTable(wideAliases, [&](const auto &add) {
            for (size_t i = 0; i < weights.size(); ++i)
                add(i, weights[i] / total * weights.size());
        });
    }

    size_t Sample(double uniform) const
    {
        return SampleIndex(uniform);
    }
};

void IndexWidths()
{
    const size_t maxCompactIndex = std::numeric_limits<uint32_t>::max();
    Require(WideTableProbe::UseCompactIndices(maxCompactIndex), "32-bit index range");
    if constexpr (std::numeric_limits<size_t>::digits > 32)
    {
        Require(WideTableProbe::UseCompactIndices(maxCompactIndex + 1), "exactly 2^32 entries still fit");
        Require(!WideTableProbe::UseCompactIndices(maxCompactIndex + 2), "larger tables need 64-bit indices");
    }
    for (const std::vector<double> &weights : {std::vector<double>{1.}, {1., 1., 1., 1.}, {0., 1., 3., 4., 0.}})
    {
        std::vector<size_t> outcomes(weights.size());
        std::iota(outcomes.begin(), outcomes.end(), 0);
        CheckDistribution(WideTableProbe(weights), outcomes, weights);
    }
}

template <class Function> void RequireDomainError(const Function &function)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::domain_error &)
    {
        rejected = true;
    }
    Require(rejected, "invalid probability support must be rejected");
}

void InvalidSupport()
{
    RequireDomainError([] { Utils::Alias sampler(std::vector<std::complex<double>>{}); });
    RequireDomainError([] { Utils::Alias sampler(std::vector<std::complex<double>>(4)); });
    Utils::PathIntegralAmplitudeMap amplitudes;
    RequireDomainError([&] { Utils::Alias sampler(amplitudes); });
    RequireDomainError([&] { Utils::AliasBig sampler(amplitudes); });
    for (double amplitude : {0., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        amplitudes[State(65)] = amplitude;
        RequireDomainError([&] { Utils::Alias sampler(amplitudes); });
        RequireDomainError([&] { Utils::AliasBig sampler(amplitudes); });
    }
}
} // namespace

int main()
{
    try
    {
        Statevectors();
        PathIntegralLabels();
        IndexWidths();
        InvalidSupport();
        std::cout << "Alias distributions, wide labels, index widths and invalid support passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
