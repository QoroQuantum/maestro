#pragma once

#include <algorithm>
#include <boost/random/binomial_distribution.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <type_traits>

namespace Utils::Sampling
{
// Direct simulators and network shot partitioning interpret an unset policy
// as auto. An explicit low-level sampler retains its own shot sequence.
inline bool AllowsMultinomialCounts(std::string_view policy, bool samplerPinned)
{
    return !samplerPinned && (policy.empty() || policy == "auto" || policy == "multinomial_v1");
}

// Count maps only: each binomial draw must amortize its higher constant cost
// and the probability pass. Measured single- and multi-threaded crossovers lie
// between 8 and 16 shots per draw (the existing per-shot draw is RNG-bound).
inline constexpr size_t multinomialShotsPerDraw = 16;

// Divide to avoid overflow.
inline bool PreferMultinomial(size_t shots, size_t draws)
{
    return shots >= 256 && draws <= shots / multinomialShotsPerDraw;
}

// A per-shot table of at most this many rows stays in cache, which makes its
// draws cheaper: there the measured break-even is at 16 shots per row, so
// count-level sampling requires 24.
inline constexpr size_t smallTableRows = 256;
inline constexpr size_t smallTableShotsPerDraw = 24;

// The same choice against per-shot draws over `tableRows` rows, for `draws`
// binomial draws (one per row for a plain plan).
inline bool PreferMultinomialDraws(size_t shots, size_t draws, size_t tableRows)
{
    return PreferMultinomial(shots, draws) && (tableRows > smallTableRows || draws <= shots / smallTableShotsPerDraw);
}

inline bool PreferMultinomialRows(size_t shots, size_t rows)
{
    return PreferMultinomialDraws(shots, rows, rows);
}

// Rows expected to receive at most this many shots share one binomial draw per
// block; their hits are then placed by categorical draws (rare-outcome bunching,
// arXiv:2610.35119 Sec. 3.3.1). The split is exact for any threshold.
inline constexpr double rareExpectedShots = 2.;

// Grouped hits cost a categorical draw each. Above this share of the mass the
// existing per-shot samplers are as fast.
inline constexpr double maxRareMassFraction = .25;

inline constexpr size_t maxExtendedMarginalBits = 8;

inline bool PreferExtendedMarginalCounts(size_t shots, size_t measuredBits)
{
    return shots >= 256 && measuredBits > 0 && measuredBits <= maxExtendedMarginalBits && (size_t{1} << measuredBits) <= shots / 8;
}

// Boost.Random accepts a floating-point generator over [0,1). Keep using the
// backend's uniform stream, including its existing engine and bit conversion.
template <class Uniform> struct BinomialUniform
{
    using result_type = double;
    Uniform &uniform;

    static constexpr double min()
    {
        return 0.;
    }

    static constexpr double max()
    {
        return 1.;
    }

    double operator()()
    {
        return uniform();
    }
};

template <class Uniform> size_t BinomialCount(size_t shots, double p, Uniform &&uniform)
{
    if (!(p >= 0. && p <= 1.))
        throw std::domain_error("Binomial sampling requires a probability in [0,1]");
    if (shots == 0 || p == 0.)
        return 0;
    if (p == 1.)
        return shots;
    const double rareProbability = std::min(p, 1. - p);
    size_t successes = 0;
    constexpr uint64_t maxChunk = uint64_t{1} << 48;
    if (rareProbability < 1.e-8 && rareProbability * static_cast<double>(std::min<uint64_t>(shots, maxChunk)) < 11.)
    {
        // Computing (1-p)^shots for inversion can round to one even when the
        // expected number of events is substantial. Skip failures geometrically
        // using log1p; the expected event count per precision-bounded chunk is small.
        const double logFailure = std::log1p(-rareProbability);
        size_t remaining = shots;
        while (remaining)
        {
            const double u = uniform();
            if (u == 0.)
                break;
            const double gap = std::floor(std::log(u) / logFailure);
            if (gap >= static_cast<double>(remaining))
                break;
            remaining -= static_cast<size_t>(gap) + 1;
            ++successes;
        }
    }
    else
    {
        // Boost uses inversion for small means and Hormann's BTRD rejection
        // sampler otherwise. Split exceptionally large counts to keep integer
        // candidates representable in the generator's floating-point arithmetic.
        // The sum of independent Bin(n_i,p) is Bin(sum n_i,p).
        BinomialUniform<std::remove_reference_t<Uniform>> engine{uniform};
        size_t remaining = shots;
        do
        {
            const auto chunk = static_cast<int64_t>(std::min<uint64_t>(remaining, maxChunk));
            successes += static_cast<size_t>(boost::random::binomial_distribution<int64_t>(chunk, rareProbability)(engine));
            remaining -= static_cast<size_t>(chunk);
        } while (remaining);
    }
    return p > .5 ? shots - successes : successes;
}
} // namespace Utils::Sampling
