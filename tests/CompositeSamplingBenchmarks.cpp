// End-to-end composite sampling benchmark; link against the library under test.
// Initialization, cloning and correctness checks are outside the timed region.
#include "../Simulators/Core/Factory.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv)
try
{
    if (argc < 8 || argc > 10)
        throw std::invalid_argument("usage: composite_sampling_benchmarks components bits_per_component shots threads measured_per_component packed|many repetitions [workers] [spread|ghz]");
    const size_t components = std::stoull(argv[1]), bits = std::stoull(argv[2]), shots = std::stoull(argv[3]);
    const int threads = std::stoi(argv[4]), repetitions = std::stoi(argv[7]);
    const size_t measured = std::stoull(argv[5]), workers = argc >= 9 ? std::stoull(argv[8]) : 1;
    const bool many = std::string(argv[6]) == "many", ghz = argc == 10 && std::string(argv[9]) == "ghz";
    if (!components || !bits || !shots || !measured || measured > bits || !workers || repetitions < 1 || (workers > 1 && threads))
        throw std::invalid_argument("invalid benchmark dimensions or nested threading");
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(std::max(1, threads));
#endif
    using namespace Simulators;
    auto source = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kCompositeQCSim, SimulationType::kStatevector);
    source->Configure("gate_fusion", "false");
    source->SetMultithreading(threads > 0);
    source->AllocateQubits(components * bits);
    source->Initialize();
    Types::qubits_vector selected;
    for (size_t component = 0; component < components; ++component)
    {
        const size_t first = component * bits;
        if (ghz)
            source->ApplyH(first);
        else
            for (size_t bit = 0; bit < bits; ++bit)
                source->ApplyRy(first + bit, 0.4 + 0.071 * ((bit * 17 + component * 3) % 23));
        for (size_t bit = 1; bit < bits; ++bit)
            source->ApplyCX(first + bit - 1, first + bit);
        for (size_t bit = 0; bit < measured; ++bit)
            selected.push_back(first + bits - bit - 1);
    }
    std::vector<std::unique_ptr<ISimulator>> sims;
    for (size_t worker = 0; worker < workers; ++worker)
        sims.push_back(source->CloneForExecution(IState::DeriveSeed(519, worker)));
    using Entries = std::vector<std::pair<std::vector<bool>, size_t>>;
    std::vector<Entries> results(workers);
    auto run = [&] {
        for (size_t worker = 0; worker < workers; ++worker)
        {
            sims[worker]->SetSeed(IState::DeriveSeed(519, worker));
            results[worker].clear();
        }
        std::vector<std::unordered_map<Types::qubit_t, Types::qubit_t>> packed(workers);
        std::vector<std::unordered_map<std::vector<bool>, Types::qubit_t>> wide(workers);
        std::vector<std::exception_ptr> errors(workers);
        auto sample = [&](size_t worker) {
            try
            {
                if (many) wide[worker] = sims[worker]->SampleCountsMany(selected, shots);
                else packed[worker] = sims[worker]->SampleCounts(selected, shots);
            }
            catch (...) { errors[worker] = std::current_exception(); }
        };
        const auto start = std::chrono::steady_clock::now();
        if (workers == 1) sample(0);
        else
        {
            std::vector<std::thread> tasks;
            for (size_t worker = 0; worker < workers; ++worker) tasks.emplace_back(sample, worker);
            for (auto &task : tasks) task.join();
        }
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        for (size_t worker = 0; worker < workers; ++worker)
        {
            if (errors[worker]) std::rethrow_exception(errors[worker]);
            if (many) results[worker].assign(wide[worker].begin(), wide[worker].end());
            else
                for (const auto &[key, count] : packed[worker])
                {
                    std::vector<bool> output(selected.size());
                    for (size_t bit = 0; bit < output.size(); ++bit) output[bit] = ((key >> bit) & 1) != 0;
                    results[worker].emplace_back(std::move(output), count);
                }
            size_t total = 0;
            for (const auto &entry : results[worker]) total += entry.second;
            if (total != shots) throw std::runtime_error("benchmark lost shots");
            std::sort(results[worker].begin(), results[worker].end());
        }
        return elapsed;
    };
    run();
    const auto expected = results;
    std::vector<double> times;
    for (int repetition = 0; repetition < repetitions; ++repetition)
    {
        times.push_back(run());
        if (results != expected) throw std::runtime_error("seeded replay changed");
    }
    std::sort(times.begin(), times.end());
    uint64_t checksum = 14695981039346656037ULL;
    for (const auto &entries : results)
        for (const auto &[key, count] : entries)
        {
            for (bool bit : key) checksum = (checksum ^ uint64_t(bit)) * 1099511628211ULL;
            checksum = (checksum ^ count) * 1099511628211ULL;
        }
    std::cout << components << ',' << bits << ',' << shots << ',' << threads << ',' << measured << ',' << (many ? "many" : "packed")
              << ',' << workers << ',' << (ghz ? "ghz" : "spread") << ',' << std::fixed << std::setprecision(3) << times[times.size() / 2]
              << ',' << checksum << '\n';
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
