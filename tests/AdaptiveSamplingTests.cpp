// Sampling policy v1: numerical boundaries, budgets and execution invariance.
#include "../Utils/Sampling/Sampling.h"
#include <iostream>
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
    try { f(); }
    catch (const Error &) { rejected = true; }
    Require(rejected, "invalid sampling input was accepted");
}

void Boundaries()
{
    for (const auto family : {Utils::Sampling::Family::Cdf, Utils::Sampling::Family::Alias})
    {
        const std::vector<double> p{0., 1., 0., 3., 0.};
        Utils::Sampling::Options options;
        options.family = family;
        const Utils::Sampling::Prepared plan(p.size(), 1024, [&](size_t row) { return p[row]; }, options);
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
                return Utils::Sampling::Count<false>(plan, uniforms.size(), 18,
                    [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return uniforms[shot++]; });
            };
            Require(count(257, true) == count(100000, false), "batch size or histogram changed counts");
        }
    }
}

void MemoryBudgets()
{
    Utils::Sampling::Options options;
    options.memoryBytes = 4096;
    const Utils::Sampling::Prepared plan(8192, 100000, [](size_t) { return 1.; }, options);
    Require(plan.Method() == Utils::Sampling::Family::Cdf, "over-budget alias did not fall back to CDF");
    std::mt19937_64 rng(42);
    const auto counts = Utils::Sampling::Count<false>(plan, 20000, 13, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; },
                                                   [&] { return Utils::RandomStream::Uniform(rng); });
    size_t total = 0;
    for (const auto &[row, count] : counts)
        total += count;
    Require(total == 20000, "bounded sampling lost shots");
    options.family = Utils::Sampling::Family::Alias;
    Reject<std::length_error>([&] { Utils::Sampling::Prepared forced(8192, 100000, [](size_t) { return 1.; }, options); });
    Reject<std::length_error>([&] { Utils::Sampling::Prepared huge(1000000, 100000, [](size_t) { return 1.; }, options); });
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
        Require(plan.Method() == Utils::Sampling::Family::Cdf && plan.ScratchAvailable() >= 1024,
                "grown blocks did not leave a bounded CDF draw batch");
        size_t shot = 0;
        const auto counts = Utils::Sampling::Count<false>(plan, 10000, 20,
            [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return (shot++ + 0.5) / 10000.; });
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
    struct Row { double probability; Types::qubit_t label; };
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
        const auto counts = Utils::Sampling::CountCdfRange<false>(rows, 20000, std::numeric_limits<Types::qubit_t>::digits,
            p, bit, [&] { return ++shot == 20000 ? std::nextafter(1., 0.) : (shot - 1) / 20000.; }, options);
        Require(counts.size() == 2 && counts.at(high) == 5000 && counts.at(high | 1) == 15000 && shot == 20000,
                "bounded streaming CDF changed normalization, endpoints or draws");
        shot = 0;
        const auto many = Utils::Sampling::CountCdfRange<true>(rows, 20000, 3, p,
            [&](const Row &row, size_t b) { return bit(row, b == 1 ? 0 : std::numeric_limits<Types::qubit_t>::digits - 1); },
            [&] { return shot++ / 20000.; }, options);
        Require(many.at({true, false, true}) == 5000 && many.at({true, true, true}) == 15000,
                "streaming CDF changed wide/repeated labels");
    }
    for (double invalid : {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        auto bad = rows;
        bad.back().probability = invalid;
        Reject<std::domain_error>([&] {
            Utils::Sampling::CountCdfRange<false>(bad, 10, 1, p, bit, []() -> double {
                throw std::runtime_error("invalid range consumed RNG before validation");
            });
        });
    }
    Require(Utils::Sampling::Single<false>(std::numeric_limits<Types::qubit_t>::digits,
        [](size_t bit) { return bit == std::numeric_limits<Types::qubit_t>::digits - 1; }).at(high) == 1, "single output truncated its high bit");
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
        const auto counts = Utils::Sampling::Count<false>(plan, shots, 18, [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; },
                                                       [&] { return (++shot - 0.5) / shots; });
        Require(counts.size() == 3 && counts.at(0) == shots / 6 && counts.at(size / 2) == shots / 3 && counts.at(size - 1) == shots / 2,
                "sparse histogram lost original row labels or weights");
        Require(probability(plan.Sample(0.)) > 0. && probability(plan.Sample(std::nextafter(1., 0.))) > 0., "sparse alias sampled a zero");
        const auto spread = [](size_t row) { return row < 131072 && row % 1024 == 0 ? 1. : 0.; };
        const Utils::Sampling::Prepared spreadPlan(size, shots, spread, options);
        shot = 0;
        const auto spreadCounts = Utils::Sampling::Count<false>(spreadPlan, shots, 18,
            [](size_t row, size_t bit) { return ((row >> bit) & 1) != 0; }, [&] { return (++shot - 0.5) / shots; });
        Require(spreadCounts.size() == 128, "parallel sparse histogram changed support");
        for (size_t row = 0; row < 128; ++row)
            Require(spreadCounts.at(row * 1024) == shots / 128, "parallel sparse histogram changed original labels or counts");
    }
}
}

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
        std::cout << "PASS adaptive sampling: boundaries, serial/parallel mapping, batches and memory budgets\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
