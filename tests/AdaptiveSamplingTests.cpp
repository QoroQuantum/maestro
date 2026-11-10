// Sampling policy v1: numerical boundaries, budgets and execution invariance.
#include "../Utils/Sampling/Sampling.h"
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <random>

namespace
{
static_assert(std::is_same_v<Utils::Sampling::Counts<false>, std::unordered_map<Types::qubit_t, Types::qubit_t>>);
static_assert(std::is_same_v<Utils::Sampling::Counts<true>, std::unordered_map<std::vector<bool>, Types::qubit_t>>);

void Require(bool value, const char *message)
{
    if (!value)
        throw std::runtime_error(message);
}

template <class Error, class Function> void Reject(Function f)
{
    bool rejected = false;
    try
    {
        f();
    }
    catch (const Error &)
    {
        rejected = true;
    }
    Require(rejected, "invalid sampling input was accepted");
}

void Boundaries()
{
    for (const auto family : {Utils::Sampling::Family::Cdf, Utils::Sampling::Family::Alias})
    {
        const std::vector<double> p{0., 1., 0., 3., 0.};
        Utils::Sampling::Options options;
        options.family = family;
        const Utils::Sampling::Prepared plan(
            p.size(), 1024, [&](size_t row) { return p[row]; }, options);
        std::vector<size_t> counts(p.size(), 0);
        for (size_t i = 0; i < 20000; ++i)
            ++counts[plan.Sample((i + 0.5) / 20000.)];
        Require(counts[1] == 5000 && counts[3] == 15000, "sampler changed normalized weights or sampled a zero");
        Require(p[plan.Sample(0.)] > 0. && p[plan.Sample(std::nextafter(1., 0.))] > 0., "endpoint sampled a zero");
    }
    const std::vector<double> invalid{0., -0.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()};
    for (double p : invalid)
        Reject<std::domain_error>([&] { Utils::Sampling::Prepared plan(1, 10, [&](size_t) { return p; }); });
    const Utils::Sampling::Prepared single(1, 100, [](size_t) { return 1E-250; });
    Require(single.Sample(0.) == 0 && single.Sample(std::nextafter(1., 0.)) == 0, "tiny positive support");
}

void ExecutionInvariance()
{
    std::mt19937_64 rng(0);
    std::vector<double> uniforms(65539);
    for (auto &u : uniforms)
        u = Utils::RandomStream::Uniform(rng);
    uniforms[0] = 0.;
    uniforms[1] = std::nextafter(1., 0.);
    for (const size_t size : {size_t{1}, size_t{37}, size_t{65536}, size_t{65537}, size_t{131101}})
    {
        std::vector<double> p(size);
        for (size_t i = 0; i < size; ++i)
            p[i] = i % 23 == 0 ? 0. : (1 + i % 157) * 0.00013;
        p[size / 2] = 1.;
        for (auto family : {Utils::Sampling::Family::Cdf, Utils::Sampling::Family::Alias})
        {
            Utils::Sampling::Options options;
            options.family = family;
            const auto probability = [&](size_t row) { return p[row]; };
            const Utils::Sampling::Prepared reference(size, uniforms.size(), probability, options);
            std::vector<size_t> expected;
            reference.Draw(uniforms, expected, 1);
            for (size_t i = 0; i < uniforms.size(); ++i)
                Require(expected[i] == reference.Sample(uniforms[i]), "sorted sweep changed shot order or boundaries");
            for (int threads : {1, 3, 8})
            {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#endif
                options.multithreading = true;
                const Utils::Sampling::Prepared parallel(size, uniforms.size(), probability, options);
                std::vector<size_t> actual;
                parallel.Draw(uniforms, actual, threads);
                Require(actual == expected, "physical threads changed samples");
            }
            auto count = [&](size_t batch, bool multithreading) {
                options.batchSize = batch;
                options.multithreading = multithreading;
                const Utils::Sampling::Prepared plan(size, uniforms.size(), probability, options);
                size_t shot = 0;
                return Utils::Sampling::Count<false>(
                    plan, uniforms.size(), 18, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return uniforms[shot++]; });
            };
            Require(count(257, true) == count(100000, false), "batch size or histogram changed counts");
        }
    }
}

void MemoryBudgets()
{
    Utils::Sampling::Options options;
    options.memoryBytes = 4096;
    const Utils::Sampling::Prepared plan(
        8192, 100000, [](size_t) { return 1.; }, options);
    Require(plan.Method() == Utils::Sampling::Family::Cdf, "over-budget alias did not fall back to CDF");
    std::mt19937_64 rng(42);
    const auto counts = Utils::Sampling::Count<false>(
        plan, 20000, 13, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
    size_t total = 0;
    for (const auto &[row, count] : counts)
        total += count;
    Require(total == 20000, "bounded sampling lost shots");
    options.family = Utils::Sampling::Family::Alias;
    Reject<std::length_error>([&] {
        Utils::Sampling::Prepared forced(
            8192, 100000, [](size_t) { return 1.; }, options);
    });
    Reject<std::length_error>([&] {
        Utils::Sampling::Prepared huge(
            1000000, 100000, [](size_t) { return 1.; }, options);
    });
}

void GrowingBlocks()
{
    constexpr size_t size = size_t{1} << 20;
    const auto probability = [](size_t row) { return row % 4096 ? 1. : 0.; };
    using Plan = Utils::Sampling::Prepared<decltype(probability)>;
    // Exercise 30/32-qubit and extreme row-count sizing without allocating a
    // huge quantum state. Reservations must fit the caller's share of memory.
    for (size_t rows : {size_t{1} << 30, size_t{1} << 32, std::numeric_limits<size_t>::max()})
        for (size_t shots : {size_t{2}, size_t{1} << 26})
            for (size_t budget : {size_t{4096}, size_t{1} << 20, size_t{256} << 20})
                Require(Plan::MinimumMemoryBytes(rows, shots, budget) <= budget, "large index reservation exceeded budget");
    Require(Plan::MinimumMemoryBytes(65536, 1000) == 256 * 17 + 8 + 1024, "ordinary block layout changed");
    for (bool parallel : {false, true})
    {
        Utils::Sampling::Options options;
        options.memoryBytes = 4096;
        options.multithreading = parallel;
        const Plan plan(size, 100000, probability, options);
        Require(plan.Method() == Utils::Sampling::Family::Cdf && plan.ScratchAvailable() >= 1024, "grown blocks did not leave a bounded CDF draw batch");
        size_t shot = 0;
        const auto counts = Utils::Sampling::Count<false>(
            plan, 10000, 20, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return (shot++ + 0.5) / 10000.; });
        Utils::Sampling::Counts<false> expected;
        for (size_t i = 0; i < 10000; ++i)
        {
            const size_t positive = static_cast<size_t>(((i + 0.5) / 10000.) * (size - size / 4096));
            ++expected[positive / 4095 * 4096 + 1 + positive % 4095];
        }
        Require(counts == expected, "grown blocks changed weights, boundaries or labels");
    }
}

void StreamingCdf()
{
    struct Row
    {
        double probability;
        Types::qubit_t label;
    };

    const Types::qubit_t high = Types::qubit_t{1} << (std::numeric_limits<Types::qubit_t>::digits - 1);
    const std::vector<Row> rows{{0., 0}, {1., high}, {0., 0}, {3., high | 1}, {0., 0}};
    const auto p = [](const Row &row) { return row.probability; };
    const auto bit = [](const Row &row, size_t b) { return ((row.label >> b) & 1) != 0; };
    for (size_t batch : {size_t{7}, size_t{100000}})
    {
        Utils::Sampling::Options options;
        options.memoryBytes = 4096;
        options.batchSize = batch;
        size_t shot = 0;
        const auto counts = Utils::Sampling::CountCdfRange<false>(
            rows, 20000, std::numeric_limits<Types::qubit_t>::digits, p, bit, [&] { return ++shot == 20000 ? std::nextafter(1., 0.) : (shot - 1) / 20000.; },
            options);
        Require(counts.size() == 2 && counts.at(high) == 5000 && counts.at(high | 1) == 15000 && shot == 20000,
                "bounded streaming CDF changed normalization, endpoints or draws");
        shot = 0;
        const auto many = Utils::Sampling::CountCdfRange<true>(
            rows, 20000, 3, p, [&](const Row &row, size_t b) { return bit(row, b == 1 ? 0 : std::numeric_limits<Types::qubit_t>::digits - 1); },
            [&] { return shot++ / 20000.; }, options);
        Require(many.at({true, false, true}) == 5000 && many.at({true, true, true}) == 15000, "streaming CDF changed wide/repeated labels");
    }
    for (double invalid : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto bad = rows;
        bad.back().probability = invalid;
        Reject<std::domain_error>([&] {
            Utils::Sampling::CountCdfRange<false>(bad, 10, 1, p, bit,
                                                  []() -> double { throw std::runtime_error("invalid range consumed RNG before validation"); });
        });
    }
    Require(Utils::Sampling::Single<false>(
                std::numeric_limits<Types::qubit_t>::digits,
                [](size_t bit) {
                    return bit == std::numeric_limits<Types::qubit_t>::digits - 1;
                }).at(high) == 1,
            "single output truncated its high bit");
}

void SparseSupport()
{
    constexpr size_t size = 131101, shots = 98304;
    const auto probability = [=](size_t row) { return row == 0 ? 1. : row == size / 2 ? 2. : row == size - 1 ? 3. : 0.; };
    for (bool multithreading : {false, true})
    {
        Utils::Sampling::Options options;
        options.multithreading = multithreading;
        const Utils::Sampling::Prepared plan(size, shots, probability, options);
        Require(plan.Categories() == 3, "sparse distribution retained a dense alias table");
        size_t shot = 0;
        const auto counts = Utils::Sampling::Count<false>(
            plan, shots, 18, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return (++shot - 0.5) / shots; });
        Require(counts.size() == 3 && counts.at(0) == shots / 6 && counts.at(size / 2) == shots / 3 && counts.at(size - 1) == shots / 2,
                "sparse histogram lost original row labels or weights");
        Require(probability(plan.Sample(0.)) > 0. && probability(plan.Sample(std::nextafter(1., 0.))) > 0., "sparse alias sampled a zero");
        const auto spread = [](size_t row) { return row < 131072 && row % 1024 == 0 ? 1. : 0.; };
        const Utils::Sampling::Prepared spreadPlan(size, shots, spread, options);
        shot = 0;
        const auto spreadCounts = Utils::Sampling::Count<false>(
            spreadPlan, shots, 18, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return (++shot - 0.5) / shots; });
        Require(spreadCounts.size() == 128, "parallel sparse histogram changed support");
        for (size_t row = 0; row < 128; ++row)
            Require(spreadCounts.at(row * 1024) == shots / 128, "parallel sparse histogram changed original labels or counts");
    }
}

void MultinomialDraws()
{
    using Utils::Sampling::BinomialCount;
    using Utils::Sampling::PreferMultinomial;
    Require(PreferMultinomial(1000000, 2) && !PreferMultinomial(2, 1000000), "unexpected count sampler crossover");
    const auto noDraw = []() -> double { throw std::runtime_error("deterministic binomial consumed randomness"); };
    const size_t largest = std::numeric_limits<size_t>::max();
    Require(BinomialCount(0, .37, noDraw) == 0 && BinomialCount(largest, 0., noDraw) == 0 && BinomialCount(largest, 1., noDraw) == largest,
            "deterministic binomial endpoint lost shots");

    std::mt19937_64 rng(89);
    const auto uniform = [&] { return Utils::RandomStream::Uniform(rng); };
    const auto checkMoments = [&](size_t shots, double p) {
        constexpr size_t repeats = 4096;
        const double mean = shots * p, variance = mean * (1. - p);
        double deviation = 0., sumSquares = 0.;
        for (size_t i = 0; i < repeats; ++i)
        {
            const size_t hits = BinomialCount(shots, p, uniform);
            Require(hits <= shots, "invalid binomial count");
            const double residual = static_cast<double>(hits) - mean;
            deviation += residual;
            sumSquares += residual * residual;
        }
        Require(std::abs(deviation) < 7. * std::sqrt(variance * repeats), "binomial sample mean drifted");
        Require(std::abs(sumSquares / repeats - variance) < .15 * variance, "binomial sample variance drifted");
    };
    checkMoments(8192, .25);
    checkMoments(4096, .99);
    checkMoments(1000000, .00001);
    if constexpr (std::numeric_limits<size_t>::digits >= 56)
        checkMoments(static_cast<size_t>(uint64_t{1} << 55), std::ldexp(1., -53));
}

void MultinomialPlans()
{
    using namespace Utils::Sampling;
    Options options;
    options.multinomialCounts = true;
    constexpr size_t rows = 262144;
    const auto probability = [=](size_t row) { return row == 1 || row == rows - 1 ? 3. : 0.; };
    const Prepared indexed(rows, 1000000, probability, options);
    Require(indexed.Method() == Family::Multinomial, "sparse stored support did not qualify for count-level sampling");
    options.memoryBytes = 4096;
    const Prepared bounded(rows, 1000000, probability, options);
    Require(bounded.Method() != Family::Multinomial, "count-level sampling exceeded the block scratch budget");
    options.family = Family::Multinomial;
    Reject<std::length_error>([&] { Prepared forced(rows, 1000000, probability, options); });

    if constexpr (std::numeric_limits<size_t>::digits >= 56)
    {
        // A small probability survives even when adding it to the other weight rounds to one.
        // Exercise both sides of a split, in one block and across block boundaries.
        constexpr auto shots = static_cast<size_t>(uint64_t{1} << 55);
        const double rare = std::ldexp(1., -55);
        options.memoryBytes = Options{}.memoryBytes;
        for (size_t last : {size_t{2}, size_t{256}})
            for (size_t rareRow : {size_t{0}, last})
            {
                const Prepared plan(
                    last + 1, shots, [&](size_t row) { return row == rareRow ? rare : (row == last - rareRow ? 1. : 0.); }, options);
                std::mt19937_64 rng(1234);
                size_t rareHits = 0;
                for (unsigned trial = 0; trial < 128; ++trial)
                {
                    const auto counts = Count<false>(
                        plan, shots, 9, [](size_t row, size_t bit) { return (row >> bit) & 1; }, [&] { return Utils::RandomStream::Uniform(rng); });
                    const auto it = counts.find(rareRow);
                    const auto hits = it == counts.end() ? 0 : it->second;
                    Require(counts.at(last - rareRow) == shots - hits, "count lost while splitting a rare outcome");
                    rareHits += hits;
                }
                Require(rareHits > 60 && rareHits < 200, "count sampler discarded rare mass");
            }
    }
}

// Pearson goodness of fit, merging cells in the given order until each expects
// at least 20. The bound is a 4.5-sigma Wilson-Hilferty quantile (p ~ 3e-6);
// fixed seeds make every check deterministic.
bool ChiSquareFits(const std::vector<double> &expected, const std::vector<double> &observed)
{
    double chi = 0., e = 0., o = 0.;
    int dof = -1;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        e += expected[i];
        o += observed[i];
        if (e >= 20.)
        {
            chi += (o - e) * (o - e) / e;
            ++dof;
            e = o = 0.;
        }
    }
    if (e > 0.)
    {
        chi += (o - e) * (o - e) / e;
        ++dof;
    }
    if (dof < 1)
        return true;
    const double k = 2. / (9. * dof);
    return chi < dof * std::pow(1. - k + 4.5 * std::sqrt(k), 3.);
}

// Every BinomialCount branch against the exact probability mass function:
// inversion, BTRD, the complement, geometric skips and the cutoff between them.
void BinomialGoodnessOfFit()
{
    const auto check = [](size_t n, double p, uint64_t seed) {
        constexpr size_t draws = 100000;
        std::mt19937_64 rng(seed);
        std::unordered_map<size_t, double> hits;
        for (size_t i = 0; i < draws; ++i)
            hits[Utils::Sampling::BinomialCount(n, p, [&] { return Utils::RandomStream::Uniform(rng); })] += 1.;
        const double mean = n * p, sd = std::sqrt(mean * (1. - p));
        const size_t low = static_cast<size_t>(std::max(0., std::floor(mean - 10. * sd - 20.)));
        const size_t high = static_cast<size_t>(std::min(double(n), std::ceil(mean + 10. * sd + 20.)));
        std::vector<double> expected, observed;
        double inside = 0.;
        for (size_t x = low; x <= high; ++x)
        {
            const double log = std::lgamma(n + 1.) - std::lgamma(x + 1.) - std::lgamma(n - x + 1.) + x * std::log(p) + (n - x) * std::log1p(-p);
            expected.push_back(std::exp(log) * draws);
            observed.push_back(hits.count(x) ? hits[x] : 0.);
            inside += observed.back();
        }
        Require(inside == draws && ChiSquareFits(expected, observed), "binomial counts do not follow the binomial distribution");
    };
    check(20, .3, 1);
    check(1000, .25, 2);
    check(5000, .97, 3);
    check(1000000000, 5e-9, 4);
    check(1000000000, 1e-8 + 1e-12, 5);
    check(100000000000, 2e-10, 6);
}

// Exact joint distributions: every composition of a few shots over the
// populated rows, across blocks and inside one. Few shots make most rows
// rare, so these also cover grouped hits beside individually drawn rows.
void MultinomialGoodnessOfFit()
{
    using namespace Utils::Sampling;
    const auto check = [](size_t count, const std::vector<std::pair<size_t, double>> &weights, size_t shots, Options options, uint64_t seed) {
        constexpr size_t trials = 200000;
        std::vector<double> p(count, 0.);
        double total = 0.;
        for (const auto &[row, weight] : weights)
            total += p[row] = weight;
        options.family = Family::Multinomial;
        const Prepared plan(count, shots, [&](size_t row) { return p[row]; }, options);
        std::mt19937_64 rng(seed);
        std::map<std::vector<size_t>, double> joint;
        for (size_t trial = 0; trial < trials; ++trial)
        {
            const auto counts = Count<false>(
                plan, shots, 16, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
            std::vector<size_t> key(weights.size(), 0);
            size_t sum = 0;
            for (const auto &[row, hits] : counts)
            {
                const auto it = std::find_if(weights.begin(), weights.end(), [&](const auto &entry) { return entry.first == row; });
                Require(it != weights.end(), "multinomial sampling drew a zero-probability row");
                key[static_cast<size_t>(it - weights.begin())] = hits;
                sum += hits;
            }
            Require(sum == shots, "multinomial sampling lost shots");
            joint[key] += 1.;
        }
        std::vector<std::pair<double, double>> cells;
        std::vector<size_t> key(weights.size(), 0);
        const auto enumerate = [&](const auto &self, size_t part, size_t left) -> void {
            if (part + 1 == weights.size())
            {
                key[part] = left;
                double log = std::lgamma(shots + 1.);
                for (size_t i = 0; i < key.size(); ++i)
                    log += key[i] * std::log(weights[i].second / total) - std::lgamma(key[i] + 1.);
                cells.emplace_back(std::exp(log) * trials, joint.count(key) ? joint[key] : 0.);
                return;
            }
            for (size_t c = 0; c <= left; ++c)
            {
                key[part] = c;
                self(self, part + 1, left - c);
            }
        };
        enumerate(enumerate, 0, shots);
        std::sort(cells.begin(), cells.end());
        std::vector<double> expected, observed;
        for (const auto &[e, o] : cells)
        {
            expected.push_back(e);
            observed.push_back(o);
        }
        Require(ChiSquareFits(expected, observed), "multinomial counts do not follow the joint distribution");
    };
    const Options defaults;
    check(600, {{5, .2}, {300, .5}, {590, .3}}, 10, defaults, 11);
    check(600, {{5, .1}, {7, .4}, {300, .35}, {590, .15}}, 8, defaults, 12);
    check(600, {{0, 1e-3}, {255, 1.}, {256, .5}, {599, 1e-2}}, 12, defaults, 13);
    check(600, {{1, .05}, {2, .05}, {3, .05}, {4, .05}, {9, .8}}, 6, defaults, 14);
    // 127 blocks of 256 rows leave room for the local prefix and a single
    // grouped-hit target, so every grouped hit takes its own batch.
    Options tight;
    tight.memoryBytes = 4096;
    check(32512, {{1, .05}, {2, .05}, {3, .05}, {4, .05}, {9, .8}}, 6, tight, 15);
}

// A long tail of tiny rows no longer forces the per-shot samplers when the
// tail can be grouped; a distribution spread over many rows still does.
void RareGroupingPlans()
{
    using namespace Utils::Sampling;
    Options options;
    options.multinomialCounts = true;
    constexpr size_t rows = 65536, shots = 65536;
    const auto tail = [](size_t row) { return row == 3 || row == 40000 ? .5 : 1e-9; };
    const Prepared grouped(rows, shots, tail, options);
    Require(grouped.Method() == Family::Multinomial, "a concentrated distribution with a tiny tail did not group its tail");
    const Prepared spread(
        rows, shots, [](size_t) { return 1.; }, options);
    Require(spread.Method() != Family::Multinomial, "grouping accepted a distribution whose tail holds most of the mass");

    std::mt19937_64 rng(77);
    size_t tailHits = 0;
    constexpr size_t trials = 200;
    for (size_t trial = 0; trial < trials; ++trial)
    {
        const auto counts = Count<false>(
            grouped, shots, 16, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
        size_t sum = 0;
        for (const auto &[row, hits] : counts)
        {
            sum += hits;
            if (row != 3 && row != 40000)
                tailHits += hits;
        }
        Require(sum == shots, "grouped tail lost shots");
    }
    const double tailMass = (rows - 2) * 1e-9 / (1. + (rows - 2) * 1e-9), expected = tailMass * shots * trials;
    Require(std::abs(tailHits - expected) < 6. * std::sqrt(expected), "grouped tail received the wrong number of shots");
}

// Scaling every weight by a power of two must change nothing, down to the
// smallest double and up to 2^1000: neither the rare threshold nor a random
// target may round to the subnormal grid, and no subtotal may overflow. The
// fixtures reach the count path with grouped and individually drawn rows, the
// alias table, the CDF and the bounded fallback.
void ScaleInvariance()
{
    using namespace Utils::Sampling;
    const double scales[] = {std::numeric_limits<double>::denorm_min(), 0x1p1000};
    struct Fixture
    {
        size_t rows, shots;
        double (*weight)(size_t);
    };
    const Fixture fixtures[] = {
        {64, 384, [](size_t row) { return row == 32 ? 160. : row % 2 ? 1. : 0.; }},
        {17, 400, [](size_t row) { return row < 17 ? 1. : 0.; }},
        {2, 4, [](size_t) { return 1.; }},
    };
    const auto histogram = [](const auto &plan, size_t shots, uint64_t seed) {
        std::mt19937_64 rng(seed);
        return Count<false>(
            plan, shots, 7, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
    };
    for (const auto &fixture : fixtures)
    {
        for (const double scale : scales)
            for (int setting = 0; setting < 5; ++setting)
            {
                Options options;
                options.multinomialCounts = setting == 0;
                options.family = setting == 2 ? Family::Cdf : setting == 3 ? Family::Alias : setting == 4 ? Family::Multinomial : Family::Auto;
                const Prepared ordinary(fixture.rows, fixture.shots, fixture.weight, options);
                const Prepared scaled(
                    fixture.rows, fixture.shots, [&](size_t row) { return fixture.weight(row) * scale; }, options);
                Require(ordinary.Method() == scaled.Method(), "scaling the weights changed the sampler");
                for (uint64_t seed = 1; seed <= 64; ++seed)
                    Require(histogram(ordinary, fixture.shots, seed) == histogram(scaled, fixture.shots, seed), "scaling the weights changed the counts");
            }
        std::vector<size_t> rows(fixture.rows);
        std::iota(rows.begin(), rows.end(), size_t{0});
        const auto bounded = [&](double scale, uint64_t seed) {
            std::mt19937_64 rng(seed);
            return CountCdfRange<false>(
                rows, fixture.shots, 7, [&](size_t row) { return fixture.weight(row) * scale; },
                [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
        };
        for (const double scale : scales)
            for (uint64_t seed = 1; seed <= 64; ++seed)
                Require(bounded(1., seed) == bounded(scale, seed), "scaling the weights changed the bounded CDF counts");
    }
}

// Four common rows of DBL_MAX / 4 sum to exactly DBL_MAX, and four rare rows
// between them hold a 2^-54 share. The row-order total is finite, but the
// common and rare subtotals, added separately, overflowed: the rare rows got
// no shots. The shifted plan must match the same weights at ordinary scale.
void HugeSubtotalsStayFinite()
{
    using namespace Utils::Sampling;
    const double common = std::numeric_limits<double>::max() / 4., rare = 0x1p968;
    const auto huge = [&](size_t row) { return row >= 2 && row < 6 ? rare : common; };
    const auto ordinary = [&](size_t row) { return huge(row) * 0x1p-1000; };
    Options options;
    options.multinomialCounts = true;
    constexpr size_t shots = size_t{1} << 56;
    const Prepared hugePlan(8, shots, huge, options);
    const Prepared ordinaryPlan(8, shots, ordinary, options);
    Require(hugePlan.Method() == Family::Multinomial, "the huge-weight plan left the count path");
    size_t rareHits = 0;
    constexpr uint64_t seeds = 128;
    for (uint64_t seed = 1; seed <= seeds; ++seed)
    {
        const auto draw = [&](const auto &plan) {
            std::mt19937_64 rng(seed);
            return Count<false>(
                plan, shots, 3, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
        };
        const auto counts = draw(hugePlan);
        Require(counts == draw(ordinaryPlan), "huge weights changed the counts");
        for (const auto &[row, hits] : counts)
            if (row >= 2 && row < 6)
                rareHits += hits;
    }
    const double expected = double(seeds) * double(shots) * 0x1p-54;
    Require(std::abs(double(rareHits) - expected) < 6. * std::sqrt(expected), "huge weights lost their rare rows");
}

// Sixteen common rows and 256 negligible rows of 2^-1070, scaled by 2^1000.
// Shifting the huge total into [0.5, 1) zeroed the negligible rows, which
// changed the sampler; shifting it only below 2^900 keeps them positive. Each
// family and the bounded fallback must match the ordinary scale.
void HugeScaleKeepsNegligibleRows()
{
    using namespace Utils::Sampling;
    constexpr size_t rows = 272, shots = 368;
    const auto ordinary = [](size_t row) { return row < 16 ? 1. : 0x1p-1070; };
    const auto huge = [&](size_t row) { return ordinary(row) * 0x1p1000; };
    const auto bit = [](size_t row, size_t b) { return ((row >> b) & 1) != 0; };
    for (int setting = 0; setting < 4; ++setting)
    {
        Options options;
        options.multinomialCounts = setting == 0;
        options.family = setting == 2 ? Family::Cdf : setting == 3 ? Family::Alias : Family::Auto;
        const Prepared ordinaryPlan(rows, shots, ordinary, options);
        const Prepared hugePlan(rows, shots, huge, options);
        Require(ordinaryPlan.Method() == hugePlan.Method(), "a huge scale changed the sampler");
        for (uint64_t seed = 0; seed < 64; ++seed)
        {
            const auto draw = [&](const auto &plan) {
                std::mt19937_64 rng(seed);
                return Count<false>(
                    plan, shots, 9, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
            };
            Require(draw(ordinaryPlan) == draw(hugePlan), "a huge scale changed the counts");
        }
    }
    std::vector<size_t> all(rows);
    std::iota(all.begin(), all.end(), size_t{0});
    for (uint64_t seed = 0; seed < 64; ++seed)
    {
        const auto bounded = [&](const auto &weight) {
            std::mt19937_64 rng(seed);
            return CountCdfRange<false>(all, shots, 9, weight, bit, [&] { return Utils::RandomStream::Uniform(rng); });
        };
        Require(bounded(ordinary) == bounded(huge), "a huge scale changed the bounded CDF counts");
    }
}

// A total below Options::minimumTotal is rejected with TotalUnderflow before
// any draw, by every family and the bounded fallback, including a zero total.
// The default never rejects, and an ordinary total passes the minimum.
void MinimumTotalRejectsBeforeDrawing()
{
    using namespace Utils::Sampling;
    const auto tiny = [](size_t row) { return row < 4 ? 0x1p-1000 : 0.; };
    const auto zero = [](size_t) { return 0.; };
    const auto ordinary = [](size_t row) { return row < 4 ? 1. : 0.; };
    const auto bit = [](size_t row, size_t b) { return ((row >> b) & 1) != 0; };
    const auto rejects = [](const auto &action) {
        try
        {
            action();
        }
        catch (const TotalUnderflow &)
        {
            return true;
        }
        return false;
    };
    for (const Family family : {Family::Auto, Family::Cdf, Family::Alias, Family::Multinomial})
    {
        Options options;
        options.family = family;
        options.minimumTotal = 0x1p-970;
        Require(rejects([&] { const Prepared plan(16, 1024, tiny, options); }), "a total below the minimum was prepared");
        Require(rejects([&] { const Prepared plan(16, 1024, zero, options); }), "a zero total escaped the minimum");
        const Prepared accepted(16, 1024, ordinary, options);
        options.minimumTotal = 0.;
        const Prepared unchecked(16, 1024, tiny, options);
    }
    std::vector<size_t> rows(16);
    std::iota(rows.begin(), rows.end(), size_t{0});
    Options options;
    options.minimumTotal = 0x1p-970;
    size_t draws = 0;
    const auto uniform = [&] {
        ++draws;
        return .5;
    };
    Require(rejects([&] { CountCdfRange<false>(rows, 1024, 4, tiny, bit, uniform, options); }) && !draws,
            "the bounded sampler drew before rejecting a small total");
    size_t total = 0;
    for (const auto &[key, hits] : CountCdfRange<false>(rows, 1024, 4, ordinary, bit, uniform, options))
        total += hits;
    Require(total == 1024, "the bounded sampler rejected an ordinary total");
}

// M - 3U followed by four weights of 9U/16, with M = DBL_MAX and U = 2^971,
// sums exactly to M - 3U/4, but adding them in order overflows. The sampler
// must accept these finite weights and match the same weights scaled down.
void OverflowingSumIsSampled()
{
    using namespace Utils::Sampling;
    const double M = std::numeric_limits<double>::max(), U = 0x1p971;
    const auto huge = [=](size_t row) { return row == 0 ? M - 3. * U : 9. * U / 16.; };
    const auto scaled = [&](size_t row) { return huge(row) * 0x1p-128; };
    for (int setting = 0; setting < 4; ++setting)
    {
        Options options;
        options.multinomialCounts = setting == 0;
        options.family = setting == 2 ? Family::Cdf : setting == 3 ? Family::Alias : Family::Auto;
        const Prepared hugePlan(5, 1024, huge, options);
        const Prepared scaledPlan(5, 1024, scaled, options);
        for (uint64_t seed = 0; seed < 32; ++seed)
        {
            const auto draw = [&](const auto &plan) {
                std::mt19937_64 rng(seed);
                return Count<false>(
                    plan, 1024, 3, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
            };
            Require(draw(hugePlan) == draw(scaledPlan), "an overflowing sum changed the counts");
        }
    }
    const std::vector<size_t> rows{0, 1, 2, 3, 4};
    for (uint64_t seed = 0; seed < 32; ++seed)
    {
        const auto bounded = [&](const auto &weight) {
            std::mt19937_64 rng(seed);
            return CountCdfRange<false>(
                rows, 1024, 3, weight, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
        };
        Require(bounded(huge) == bounded(scaled), "an overflowing sum changed the bounded CDF counts");
    }
}

// multinomial_v1 is a replayable policy. A change to these decisions or to
// this seeded stream needs a new policy version (multinomial_v2), not new
// expected values. The stream is the same with MSVC and GCC.
void PinnedMultinomialStream()
{
    using namespace Utils::Sampling;
    Options options;
    options.multinomialCounts = true;
    const auto uniform = [](size_t) { return 1.; };
    const auto multinomial = [&](size_t rows, size_t shots) { return Prepared(rows, shots, uniform, options).Method() == Family::Multinomial; };
    Require(!multinomial(1024, 16383) && multinomial(1024, 16384), "multinomial_v1 changed its 16-shots-per-row crossover");
    Require(!multinomial(256, 6143) && multinomial(256, 6144), "multinomial_v1 changed its small-table crossover");
    Require(!multinomial(8, 255) && multinomial(8, 256), "multinomial_v1 changed its minimum shot count");

    constexpr size_t rows = 3000, shots = 50000;
    const auto weight = [](size_t row) { return row % 11 == 0 ? 0. : row < 40 ? 1. + double(row % 5) : row % 3 == 0 ? 2e-4 : 1e-6; };
    const Prepared plan(rows, shots, weight, options);
    Require(plan.Method() == Family::Multinomial, "multinomial_v1 rejected its pinned distribution");
    std::mt19937_64 rng(2026);
    const auto counts = Count<false>(
        plan, shots, 12, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(rng); });
    uint64_t digest = 0;
    size_t occupied = 0, tail = 0;
    for (size_t row = 0; row < rows; ++row)
        if (const auto it = counts.find(row); it != counts.end())
        {
            digest = digest * 1000003u + row * 65537u + it->second;
            ++occupied;
            if (row >= 40)
                tail += it->second;
        }
    Require(occupied == 105 && tail == 69 && digest == 3744179436325232569ull, "multinomial_v1 changed its seeded stream");

    // The tail-grouping gate, which the fixture above bypasses: 4,096 positive
    // rows exceed 8,192 / 16, so only grouping accepts these distributions.
    // Its rare-mass cutoff (a quarter) and its draw limit (16 populated blocks
    // plus common rows, against 512) are pinned on both sides.
    const auto tailMethod = [&](auto weight) { return Prepared(4096, 8192, weight, options).Method(); };
    Require(tailMethod([](size_t row) { return row < 2 ? 1. : .6 / 4094.; }) == Family::Multinomial &&
                tailMethod([](size_t row) { return row < 2 ? 1. : .7 / 4094.; }) != Family::Multinomial,
            "multinomial_v1 changed its rare-mass cutoff");
    Require(tailMethod([](size_t row) { return row < 496 ? 1. : 1e-9; }) == Family::Multinomial &&
                tailMethod([](size_t row) { return row < 497 ? 1. : 1e-9; }) != Family::Multinomial,
            "multinomial_v1 changed its grouped draw limit");
    const auto tailWeight = [](size_t row) { return row == 7 || row == 2000 || row == 4000 ? 1. : row % 13 == 0 ? 0. : 2e-6; };
    const Prepared tailPlan(4096, 8192, tailWeight, options);
    Require(tailPlan.Method() == Family::Multinomial, "multinomial_v1 rejected its pinned tail");
    std::mt19937_64 tailRng(2027);
    const auto tailCounts = Count<false>(
        tailPlan, 8192, 13, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(tailRng); });
    digest = 0;
    occupied = 0;
    for (size_t row = 0; row < 4096; ++row)
        if (const auto it = tailCounts.find(row); it != tailCounts.end())
        {
            digest = digest * 1000003u + row * 65537u + it->second;
            ++occupied;
        }
    Require(occupied == 19 && digest == 4078924110512412999ull, "multinomial_v1 changed its grouped-tail stream");

    // Geometric waiting times for a probability below 1e-8, which no
    // histogram above reaches.
    std::mt19937_64 geometricRng(99);
    std::vector<size_t> geometric;
    for (int draw = 0; draw < 8; ++draw)
        geometric.push_back(BinomialCount(size_t{1000000000000}, 3e-12, [&] { return Utils::RandomStream::Uniform(geometricRng); }));
    Require(geometric == std::vector<size_t>{3, 4, 3, 5, 3, 5, 0, 2}, "multinomial_v1 changed its geometric binomial branch");

    // The policy's constants, and behavior on both sides of each.
    static_assert(multinomialShotsPerDraw == 16 && smallTableRows == 256 && smallTableShotsPerDraw == 24, "multinomial_v1 crossover constants changed");
    static_assert(rareExpectedShots == 2. && maxRareMassFraction == .25, "multinomial_v1 grouping constants changed");
    Require(!multinomial(17, 407) && multinomial(17, 408), "multinomial_v1 changed its small-table rule for odd row counts");
    // 63 rows expected to receive 2.5 shots each are drawn one by one; with a
    // rare row defined by three expected hits they would be grouped instead.
    const Prepared middle(
        64, 4096, [](size_t row) { return row == 0 ? 1575.4 : 1.; }, options);
    std::mt19937_64 middleRng(2028);
    const auto middleCounts = Count<false>(
        middle, 4096, 6, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return Utils::RandomStream::Uniform(middleRng); });
    digest = 0;
    occupied = 0;
    for (size_t row = 0; row < 64; ++row)
        if (const auto it = middleCounts.find(row); it != middleCounts.end())
        {
            digest = digest * 1000003u + row * 65537u + it->second;
            ++occupied;
        }
    Require(middle.Method() == Family::Multinomial && occupied == 57 && digest == 13282814273920193656ull,
            "multinomial_v1 changed which rows it draws individually");
    // Both sides of the geometric cutoff at p = 1e-8, near a mean of ten.
    const auto draws = [](double p) {
        std::mt19937_64 cutoffRng(100);
        std::vector<size_t> values;
        for (int draw = 0; draw < 8; ++draw)
            values.push_back(BinomialCount(size_t{1000000000}, p, [&] { return Utils::RandomStream::Uniform(cutoffRng); }));
        return values;
    };
    Require(draws(0.99e-8) == std::vector<size_t>{16, 13, 9, 5, 9, 6, 10, 12} && draws(1.01e-8) == std::vector<size_t>{16, 9, 17, 7, 14, 10, 16, 12},
            "multinomial_v1 changed its geometric cutoff");
}

// A sole weight near the largest double must not overflow the rare threshold:
// its certain outcome takes no random draws.
void HugeWeightNeedsNoDraws()
{
    using namespace Utils::Sampling;
    Options options;
    options.multinomialCounts = true;
    constexpr size_t shots = 1000000;
    const Prepared plan(
        1, shots, [](size_t) { return 1e308; }, options);
    size_t draws = 0;
    std::mt19937_64 rng(1);
    const auto counts = Count<false>(
        plan, shots, 1, [](size_t, size_t) { return false; },
        [&] {
            ++draws;
            return Utils::RandomStream::Uniform(rng);
        });
    Require(plan.Method() == Family::Multinomial && counts.at(0) == shots && draws == 0, "a huge sole weight was sampled shot by shot");
}
} // namespace

int main()
{
    try
    {
        Boundaries();
        ExecutionInvariance();
        MemoryBudgets();
        GrowingBlocks();
        StreamingCdf();
        SparseSupport();
        MultinomialDraws();
        MultinomialPlans();
        BinomialGoodnessOfFit();
        MultinomialGoodnessOfFit();
        RareGroupingPlans();
        ScaleInvariance();
        HugeSubtotalsStayFinite();
        HugeScaleKeepsNegligibleRows();
        OverflowingSumIsSampled();
        MinimumTotalRejectsBeforeDrawing();
        PinnedMultinomialStream();
        HugeWeightNeedsNoDraws();
        std::cout << "PASS adaptive sampling: boundaries, serial/parallel mapping, batches and memory budgets\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
