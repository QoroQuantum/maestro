// Manual adapter benchmark. Compare the previous per-shot vector histogram
// with production batching on the same prepared MPO and random stream.
#define INCLUDED_BY_FACTORY
#include "../Circuit/Circuit.h"
#include "../Simulators/QCSim/QCSimMPOSimulator.h"
#include <chrono>
#include <iomanip>
#include <iostream>

using Counts = std::unordered_map<std::vector<bool>, Types::qubit_t>;
using Clock = std::chrono::steady_clock;

struct MPOProbe : Simulators::Private::QCSimMPOSimulator
{
    Counts PreviousCounts(const Types::qubits_vector &qubits, size_t shots)
    {
        Counts result;
        DontNotify();
        const std::set<Eigen::Index> selected(qubits.begin(), qubits.end());
        for (size_t shot = 0; shot < shots; ++shot)
        {
            const auto measured = mpoSimulator->MeasureNoCollapse(selected);
            std::vector<bool> bits(qubits.size());
            for (size_t bit = 0; bit < qubits.size(); ++bit)
                bits[bit] = measured.at(static_cast<Eigen::Index>(qubits[bit]));
            ++result[bits];
        }
        Notify();
        NotifyObservers(qubits);
        return result;
    }
};

int main(int argc, char **argv)
try
{
    const size_t repetitions = argc > 1 ? std::stoull(argv[1]) : 7;
    if (!repetitions)
        throw std::invalid_argument("repetitions must be positive");
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    struct Case
    {
        const char *name;
        size_t qubits, shots;
        bool entangled;
        Types::qubits_vector selected;
    };

    Types::qubits_vector wide(70);
    std::iota(wide.begin(), wide.end(), 0);
    for (const auto &test : {Case{"one_qubit", 1, 100000, false, {0}}, Case{"mixed_permuted_duplicates", 8, 20000, true, {7, 0, 3, 7, 2, 1}},
                             Case{"wide_product", 70, 2000, false, wide}})
    {
        MPOProbe sim;
        sim.Configure("counts_policy", "legacy");
        sim.SetMultithreading(false);
        sim.AllocateQubits(test.qubits);
        sim.Initialize();
        sim.ApplyH(0);
        if (test.entangled)
            for (size_t q = 1; q < test.qubits; ++q)
                sim.ApplyCX(0, q);
        sim.ApplyAmplitudeDamping(0, 0.23);

        sim.SetSeed(19);
        const auto expected = sim.PreviousCounts(test.selected, test.shots);
        const auto following = sim.SampleCounts({0}, 37);
        sim.SetSeed(19);
        const auto actual = sim.SampleCountsMany(test.selected, test.shots);
        if (actual != expected || sim.SampleCounts({0}, 37) != following)
            throw std::runtime_error("MPO batching changed seeded outcomes or RNG continuation");
        size_t total = 0;
        for (const auto &[bits, count] : actual)
        {
            if (bits.size() != test.selected.size())
                throw std::runtime_error("MPO batching changed output width");
            total += count;
        }
        if (total != test.shots)
            throw std::runtime_error("MPO batching lost shots");

        std::vector<double> before, after;
        for (size_t rep = 0; rep < repetitions; ++rep)
            for (bool optimized : {rep % 2 == 0, rep % 2 != 0})
            {
                sim.SetSeed(rep + 21);
                const auto start = Clock::now();
                const auto counts = optimized ? sim.SampleCountsMany(test.selected, test.shots) : sim.PreviousCounts(test.selected, test.shots);
                const double millis = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                (optimized ? after : before).push_back(millis);
                if (counts.empty())
                    throw std::runtime_error("MPO sampling returned no outcomes");
            }
        std::sort(before.begin(), before.end());
        std::sort(after.begin(), after.end());
        const double oldMs = before[before.size() / 2], newMs = after[after.size() / 2];
        std::cout << std::fixed << std::setprecision(3) << "{\"case\":\"" << test.name << "\",\"shots\":" << test.shots << ",\"previous_ms\":" << oldMs
                  << ",\"batched_ms\":" << newMs << ",\"speedup\":" << oldMs / newMs << "}\n";
    }
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
