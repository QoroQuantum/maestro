// End-to-end count API benchmark. Initialization and correctness checks are
// outside timing; preparation, RNG, draw, projection and histogram are inside.
#define INCLUDED_BY_FACTORY
#include <iostream>
#include <thread>
#include "../Circuit/Circuit.h"
#include "../Simulators/QCSim/QCSimStatevectorSimulator.h"
#include "../Simulators/QCSim/QCSimDensityMatrixSimulator.h"
#include "../Simulators/QCSim/QCSimPathIntegralSimulator.h"
#include <chrono>
#include <iomanip>
#include <iostream>

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

int main(int argc, char **argv)
try
{
    if (argc != 10 && argc != 11)
        throw std::invalid_argument("usage: sampling_benchmarks sv|dm|path qubits active_bits shots threads random|uniform|basis measured_bits packed|many repetitions [outer_workers]");
    const std::string method = argv[1], kind = argv[6];
    const size_t width = std::stoull(argv[2]), active = std::stoull(argv[3]), shots = std::stoull(argv[4]);
    const int threads = std::stoi(argv[5]), reps = std::stoi(argv[9]);
    const size_t measured = std::stoull(argv[7]);
    const bool many = std::string(argv[8]) == "many";
    const size_t outer = argc == 11 ? std::stoull(argv[10]) : 1;
    if (!outer || reps < 1 || (outer > 1 && threads != 0))
        throw std::invalid_argument("outer workers require internal threading disabled (threads=0)");
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, threads));
#endif
    const size_t count = size_t{1} << active;
    std::vector<double> probabilities(count);
    std::mt19937_64 random(17);
    std::exponential_distribution<double> exponential;
    double mass = 0.;
    for (size_t row = 0; row < count; ++row)
    {
        probabilities[row] = kind == "basis" ? double(row == 0) : kind == "ghz" ? double(row == 0 || row == count - 1) :
                             kind == "uniform" ? 1. : exponential(random);
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
    else
        throw std::invalid_argument("unknown backend");
    sim->SetMultithreading(threads > 0);
    Types::qubits_vector selected;
    for (size_t bit = 0; bit < measured; ++bit)
        selected.push_back(bit == measured - 1 && width > active ? width - 1 : bit);
    std::vector<std::unique_ptr<Simulators::ISimulator>> clones;
    if (outer > 1)
        for (size_t worker = 0; worker < outer; ++worker)
            clones.push_back(sim->CloneForExecution(Simulators::IState::DeriveSeed(0, worker)));
    auto run = [&](const char *policy) {
        if (outer > 1)
        {
            std::vector<Utils::Sampling::Counts<false>> packed(outer);
            std::vector<Utils::Sampling::Counts<true>> wide(outer);
            std::vector<std::exception_ptr> errors(outer);
            for (size_t worker = 0; worker < outer; ++worker)
            {
                clones[worker]->Configure("sampling_policy", policy);
                clones[worker]->SetSeed(Simulators::IState::DeriveSeed(0, worker));
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
                    catch (...) { errors[worker] = std::current_exception(); }
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
        sim->Configure("sampling_policy", policy);
        sim->SetSeed(0);
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
    run("legacy");
    run("reproducible_v1");
    std::vector<double> legacy, adaptive;
    for (int rep = 0; rep < reps; ++rep)
    {
        // Alternate order to limit warm-cache and clock bias.
        if (rep % 2 == 0)
        {
            legacy.push_back(run("legacy"));
            adaptive.push_back(run("reproducible_v1"));
        }
        else
        {
            adaptive.push_back(run("reproducible_v1"));
            legacy.push_back(run("legacy"));
        }
    }
    std::sort(legacy.begin(), legacy.end());
    std::sort(adaptive.begin(), adaptive.end());
    const double before = legacy[reps / 2], after = adaptive[reps / 2];
    std::cout << std::fixed << std::setprecision(3) << method << ',' << width << ',' << count << ',' << shots << ',' << threads << ',' << kind << ','
              << measured << ',' << (many ? "many" : "packed") << ',' << before << ',' << after << ',' << before / after << ',' << outer << std::endl;
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
