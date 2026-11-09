// End-to-end count API benchmark. Initialization and correctness checks are
// outside timing; preparation, RNG, draw, projection and histogram are inside.
#define INCLUDED_BY_FACTORY
#include "../Circuit/Circuit.h"
#include "../Simulators/QCSim/QCSimDensityMatrixSimulator.h"
#include "../Simulators/QCSim/QCSimExtendedStabilizerSimulator.h"
#include "../Simulators/QCSim/QCSimMPOSimulator.h"
#include "../Simulators/QCSim/QCSimMPSSimulator.h"
#include "../Simulators/QCSim/QCSimPathIntegralSimulator.h"
#include "../Simulators/QCSim/QCSimStatevectorSimulator.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

using Clock = std::chrono::steady_clock;
using namespace Simulators::Private;

struct DensityInput : QCSimDensityMatrixSimulator
{
    void Populate(const std::vector<double> &p)
    {
        auto &matrix = const_cast<Eigen::MatrixXcd &>(densityMatrix->getDensityMatrix());
        matrix.setZero();
        for (size_t row = 0; row < p.size(); ++row)
            matrix(row, row) = p[row];
    }
};

struct PathInput : QCSimPathIntegralSimulator
{
    void Populate(const std::vector<double> &p, size_t width, size_t active)
    {
        auto &amplitudes = pathIntegralSimulator->Amplitudes();
        amplitudes.clear();
        amplitudes.reserve(p.size());
        for (size_t row = 0; row < p.size(); ++row)
        {
            QC::PathIntegral::FastVectorBool key(width);
            for (size_t bit = 0; bit < active; ++bit)
                key.set(bit, ((row >> bit) & 1) != 0);
            if (width > active)
                key.set(width - 1, true);
            amplitudes[key] = std::sqrt(p[row]);
        }
    }
};

// Force the candidate on both sides of the crossover. Backend measurements
// below also include the actual policy decision, backend preparation and mapping.
template <bool Many>
double TimeKernel(const std::vector<double> &probabilities, const Types::qubits_vector &selected, size_t shots, int threads, uint64_t seed, bool multinomial,
                  size_t scratchBytes)
{
    std::mt19937_64 rng(seed);
    Utils::Sampling::Options options;
    options.multithreading = threads > 0;
    if (scratchBytes)
        options.memoryBytes = scratchBytes;
    if (multinomial)
        options.family = Utils::Sampling::Family::Multinomial;
    const auto start = Clock::now();
    const Utils::Sampling::Prepared plan(probabilities.size(), shots, [&](size_t row) { return probabilities[row]; }, options);
    const auto counts = Utils::Sampling::Count<Many>(
        plan, shots, selected.size(), [&](size_t row, size_t bit) { return (row >> selected[bit]) & 1; }, [&] { return Utils::RandomStream::Uniform(rng); });
    const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    size_t total = 0;
    for (const auto &[key, count] : counts)
        total += count;
    if (total != shots)
        throw std::runtime_error("kernel benchmark lost shots");
    return elapsed;
}

int main(int argc, char **argv)
try
{
    if (argc < 10 || argc > 13)
        throw std::invalid_argument(
            "usage: sampling_benchmarks sv|dm|path|mps|mpo|extended|kernel qubits active_bits shots threads random|uniform|basis|last|ghz|rotated|tail measured_bits "
            "packed|many repetitions [outer_workers [multinomial [scratch_mb]]]");
    const std::string method = argv[1], kind = argv[6];
    const size_t width = std::stoull(argv[2]), active = std::stoull(argv[3]), shots = std::stoull(argv[4]);
    const int threads = std::stoi(argv[5]), reps = std::stoi(argv[9]);
    const size_t measured = std::stoull(argv[7]);
    const bool many = std::string(argv[8]) == "many";
    const size_t outer = argc >= 11 ? std::stoull(argv[10]) : 1;
    const bool multinomial = argc >= 12 && std::string_view(argv[11]) == "multinomial";
    if (argc >= 12 && !multinomial)
        throw std::invalid_argument("expected multinomial comparison");
    // An optional sampling scratch budget in whole MiB, for example 1 to time
    // the bounded path-integral fallback. The kernel and every backend use it.
    size_t scratchMiB = 0;
    if (argc == 13)
    {
        constexpr size_t limit = std::numeric_limits<size_t>::max() >> 20;
        for (const char c : std::string_view(argv[12]))
        {
            if (c < '0' || c > '9' || scratchMiB > (limit - size_t(c - '0')) / 10)
                throw std::invalid_argument("scratch_mb must be a positive whole number of MiB");
            scratchMiB = scratchMiB * 10 + size_t(c - '0');
        }
        if (!scratchMiB)
            throw std::invalid_argument("scratch_mb must be a positive whole number of MiB");
    }
    if (method == "kernel" && (!multinomial || outer != 1 || active != width))
        throw std::invalid_argument("kernel comparison requires multinomial, one worker, and active_bits equal to qubits");
    if (active >= std::numeric_limits<size_t>::digits || active > width || !measured || measured > width)
        throw std::invalid_argument("invalid register or measurement width");
    if (!outer || reps < 1 || (outer > 1 && threads != 0))
        throw std::invalid_argument("outer workers require internal threading disabled (threads=0)");
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, threads));
#endif
    const size_t count = size_t{1} << active;
    std::vector<double> probabilities(method == "mps" || method == "mpo" || method == "extended" ? 0 : count);
    std::mt19937_64 random(17);
    std::exponential_distribution<double> exponential;
    double mass = 0.;
    for (size_t row = 0; row < probabilities.size(); ++row)
    {
        // tail: two outcomes hold 99% of the mass, every other row shares 1%.
        // last: only the final row has mass, after zero rows a path-integral
        // state can retain when pruning is disabled.
        probabilities[row] = kind == "basis"     ? double(row == 0)
                             : kind == "last"    ? double(row == count - 1)
                             : kind == "ghz"     ? double(row == 0 || row == count - 1)
                             : kind == "uniform" ? 1.
                             : kind == "tail"    ? (row == 0 || row == count - 1 ? .495 : .01 / double(count - 2))
                                                 : exponential(random);
        mass += probabilities[row];
    }
    for (auto &p : probabilities)
        p /= mass;
    std::unique_ptr<Simulators::ISimulator> sim;
    if (method == "sv")
    {
        auto state = std::make_unique<QCSimStatevectorSimulator>();
        if (kind == "basis")
        {
            state->AllocateQubits(width);
            state->Initialize();
        }
        else
        {
            Eigen::VectorXcd amplitudes = Eigen::VectorXcd::Zero(size_t{1} << width);
            for (size_t row = 0; row < count; ++row)
                amplitudes[row] = std::sqrt(probabilities[row]);
            state->InitializeState(width, amplitudes);
        }
        sim = std::move(state);
    }
    else if (method == "dm")
    {
        auto state = std::make_unique<DensityInput>();
        state->AllocateQubits(width);
        state->Initialize();
        state->Populate(probabilities);
        sim = std::move(state);
    }
    else if (method == "path")
    {
        auto state = std::make_unique<PathInput>();
        state->AllocateQubits(width);
        state->Initialize();
        state->Populate(probabilities, width, active);
        sim = std::move(state);
    }
    else if (method == "mps")
    {
        if (kind != "uniform" || active != width)
            throw std::invalid_argument("MPS fixture requires a uniform state on all qubits");
        auto state = std::make_unique<QCSimMPSSimulator>();
        state->AllocateQubits(width);
        state->Initialize();
        for (size_t bit = 0; bit < width; ++bit)
            state->ApplyH(bit);
        sim = std::move(state);
    }
    else if (method == "mpo")
    {
        // random: active_bits layers of random Ry on every qubit and a CX chain;
        // each layer can multiply the operator bond dimension by four. With
        // active_bits < qubits the measured bit is the last site of the chain.
        if (kind != "uniform" && kind != "random")
            throw std::invalid_argument("MPO fixture requires a uniform state or random layers");
        auto state = std::make_unique<QCSimMPOSimulator>();
        state->AllocateQubits(width);
        state->Initialize();
        std::uniform_real_distribution<double> angle(0., std::acos(-1.));
        for (size_t bit = 0; bit < width; ++bit)
            state->ApplyH(bit);
        for (size_t layer = 0; kind == "random" && layer < active; ++layer)
        {
            for (size_t bit = 0; bit < width; ++bit)
                state->ApplyRy(bit, angle(random));
            for (size_t bit = 0; bit + 1 < width; ++bit)
                state->ApplyCX(bit, bit + 1);
        }
        std::cerr << "MPO max bond dimension " << state->GetCurrentMaxBondDimension() << std::endl;
        sim = std::move(state);
    }
    else if (method == "extended")
    {
        if (!active || (kind != "ghz" && kind != "rotated" && kind != "uniform"))
            throw std::invalid_argument("extended fixture requires a nonempty ghz, rotated or uniform state");
        auto state = std::make_unique<QCSimExtendedStabilizerSimulator>();
        state->AllocateQubits(width);
        state->Initialize();
        // Match the output selection for sparse registers: one live bit is at
        // the far end, so the measured parity strings can span the whole register.
        const auto wire = [&](size_t bit) { return bit + 1 == active ? width - 1 : bit; };
        state->ApplyH(wire(0));
        for (size_t bit = 1; bit < active; ++bit)
        {
            if (kind == "uniform")
                state->ApplyH(wire(bit));
            else
                state->ApplyCX(wire(bit - 1), wire(bit));
        }
        if (kind == "rotated")
            for (size_t bit = 0; bit < std::min<size_t>(active, 6); bit += 2)
                state->ApplyRy(wire(bit), std::acos(-1.) / 7.);
        sim = std::move(state);
    }
    else if (method != "kernel")
        throw std::invalid_argument("unknown backend");
    if (sim)
        sim->SetMultithreading(threads > 0);
    if (sim && scratchMiB)
        sim->Configure("sampling_max_memory_mb", std::to_string(scratchMiB).c_str());
    Types::qubits_vector selected;
    for (size_t bit = 0; bit < measured; ++bit)
        selected.push_back(bit == measured - 1 && width > active ? width - 1 : bit);
    std::vector<std::unique_ptr<Simulators::ISimulator>> clones;
    if (outer > 1)
        for (size_t worker = 0; worker < outer; ++worker)
            clones.push_back(sim->CloneForExecution(Simulators::IState::DeriveSeed(0, worker)));
    auto run = [&](bool candidate, uint64_t seed) {
        if (method == "kernel")
            return many ? TimeKernel<true>(probabilities, selected, shots, threads, seed, candidate, scratchMiB << 20)
                        : TimeKernel<false>(probabilities, selected, shots, threads, seed, candidate, scratchMiB << 20);
        const char *key = multinomial ? "counts_policy" : "sampling_policy";
        const char *policy = candidate ? (multinomial ? "multinomial_v1" : "reproducible_v1") : "legacy";
        if (outer > 1)
        {
            std::vector<Utils::Sampling::Counts<false>> packed(outer);
            std::vector<Utils::Sampling::Counts<true>> wide(outer);
            std::vector<std::exception_ptr> errors(outer);
            for (size_t worker = 0; worker < outer; ++worker)
            {
                clones[worker]->Configure(key, policy);
                clones[worker]->SetSeed(Simulators::IState::DeriveSeed(seed, worker));
            }
            std::vector<std::thread> workers;
            const auto start = Clock::now();
            for (size_t worker = 0; worker < outer; ++worker)
                workers.emplace_back([&, worker] {
                    try
                    {
                        if (many)
                            wide[worker] = clones[worker]->SampleCountsMany(selected, shots);
                        else
                            packed[worker] = clones[worker]->SampleCounts(selected, shots);
                    }
                    catch (...)
                    {
                        errors[worker] = std::current_exception();
                    }
                });
            for (auto &worker : workers)
                worker.join();
            const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            for (size_t worker = 0; worker < outer; ++worker)
            {
                if (errors[worker])
                    std::rethrow_exception(errors[worker]);
                size_t total = 0;
                if (many)
                    for (const auto &[key, count] : wide[worker])
                        total += count;
                else
                    for (const auto &[key, count] : packed[worker])
                        total += count;
                if (total != shots)
                    throw std::runtime_error("concurrent benchmark lost shots");
            }
            return elapsed;
        }
        sim->Configure(key, policy);
        sim->SetSeed(seed);
        const auto start = Clock::now();
        size_t total = 0;
        double elapsed = 0.;
        if (many)
        {
            const auto counts = sim->SampleCountsMany(selected, shots);
            elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            for (const auto &[bits, count] : counts)
                total += count;
        }
        else
        {
            const auto counts = sim->SampleCounts(selected, shots);
            elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            for (const auto &[bits, count] : counts)
                total += count;
        }
        if (total != shots)
            throw std::runtime_error("benchmark lost shots");
        return elapsed;
    };
    run(false, 0);
    run(true, 0);
    std::vector<double> baseline, candidate;
    for (int rep = 0; rep < reps; ++rep)
    {
        // Alternate order to limit warm-cache and clock bias.
        const uint64_t seed = multinomial ? static_cast<uint64_t>(rep + 1) : 0;
        for (bool second : {rep % 2 != 0, rep % 2 == 0})
            (second ? candidate : baseline).push_back(run(second, seed));
    }
    std::sort(baseline.begin(), baseline.end());
    std::sort(candidate.begin(), candidate.end());
    const double before = baseline[reps / 2], after = candidate[reps / 2];
    std::cout << std::fixed << std::setprecision(multinomial ? 5 : 3) << method << ',' << width << ',' << count << ',' << shots << ',' << threads << ',' << kind
              << ',' << measured << ',' << (many ? "many" : "packed") << ',' << before << ',' << after << ',' << before / after << ',' << outer << std::endl;
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
