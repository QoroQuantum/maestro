// Regression tests for caller-ordered sampling, including mapped internal states.
#include "../Simulators/Core/Factory.h"
#define INCLUDED_BY_FACTORY
#ifndef NO_QISKIT_AER
#include "../Simulators/Aer/AerSimulator.h"
#endif
#include "../Simulators/Composite/Individual.h"
#include "../Simulators/Composite/ImmediateComposite.h"
#include "../Simulators/QCSim/QCSimExtendedStabilizerSimulator.h"
#include "../Simulators/QCSim/QCSimStabilizerSimulator.h"
#include "../Simulators/QCSim/QCSimDensityMatrixSimulator.h"
#include <array>
#include <iostream>
#include <stdexcept>

using namespace Simulators;

namespace
{
static_assert(std::is_same_v<Utils::Sampling::Counts<false>, decltype(std::declval<ISimulator &>().SampleCounts({}))>);
static_assert(std::is_same_v<Utils::Sampling::Counts<true>, decltype(std::declval<ISimulator &>().SampleCountsMany({}))>);

void Require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

// Test every computational basis state so symmetric bit patterns cannot hide
// permutations. Include full permutations, subsets and single-qubit queries.
void CheckSamples(ISimulator &sim, const std::vector<Types::qubit_t> &ids)
{
    sim.AllocateQubits(3);
    sim.Initialize();
    unsigned previous = 0;
    for (unsigned basis = 0; basis < 8; ++basis)
    {
        for (unsigned bit = 0; bit < 3; ++bit)
            if (((previous ^ basis) >> bit) & 1)
                sim.ApplyX(ids[bit]);
        previous = basis;
        const std::vector<std::vector<unsigned>> orders{{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {2, 0}, {1, 2}, {2}, {0}};
        for (const auto &order : orders)
        {
            Types::qubits_vector qubits;
            std::vector<bool> expected;
            Types::qubit_t packed = 0;
            for (size_t i = 0; i < order.size(); ++i)
            {
                qubits.push_back(ids[order[i]]);
                expected.push_back((basis >> order[i]) & 1);
                if (expected.back())
                    packed |= Types::qubit_t(1) << i;
            }
            for (size_t shots : {size_t(1), size_t(16)})
            {
                const auto smallv = sim.SampleCounts(qubits, shots);
                const auto wide = sim.SampleCountsMany(qubits, shots);
                const std::string context = "basis=" + std::to_string(basis) + " first_qubit=" + std::to_string(qubits.front()) +
                                            " width=" + std::to_string(qubits.size()) + " shots=" + std::to_string(shots);
                Require(smallv.size() == 1 && smallv.count(packed) && smallv.at(packed) == shots, "packed sampling: " + context);
                Require(wide.size() == 1 && wide.count(expected) && wide.at(expected) == shots, "vector sampling: " + context);
            }
        }
    }
    Require(sim.SampleCounts({ids[0]}, 0).empty() && sim.SampleCountsMany({ids[0]}, 0).empty(), "zero shots");
}

void Backend(SimulatorType type, SimulationType method, const char *algorithm = nullptr)
{
    auto sim = SimulatorsFactory::CreateSimulator(type, method);
    Require(bool(sim), "simulator unavailable");
    if (algorithm)
        sim->Configure("mps_sample_measure_algorithm", algorithm);
    CheckSamples(*sim, {0, 1, 2});
}

void SeededBatchSampling(SimulationType method, bool legacy = false)
{
    auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, method);
    if (legacy)
        sim->Configure("sampling_policy", "legacy");
    sim->AllocateQubits(4);
    sim->Initialize();
    sim->ApplyH(0);
    sim->ApplyCX(0, 2);
    sim->ApplyRy(1, 0.37);
    sim->ApplyRy(3, 0.81);
    const Types::qubits_vector selected{3, 0, 2, 3};
    constexpr size_t shots = 257;

    for (unsigned iteration = 0; iteration < 2; ++iteration)
    {
        const auto before = sim->AllProbabilities();
        sim->SetSeed(891);
        const auto packed = sim->SampleCounts(selected, shots);
        const auto next = sim->MeasureNoCollapse();
        sim->SetSeed(891);
        const auto many = sim->SampleCountsMany(selected, shots);
        Require(sim->MeasureNoCollapse() == next, "count formats preserve RNG consumption");

        std::unordered_map<std::vector<bool>, Types::qubit_t> expected;
        for (const auto &[outcome, count] : packed)
        {
            std::vector<bool> bits(selected.size());
            for (size_t i = 0; i < bits.size(); ++i)
                bits[i] = ((outcome >> i) & 1) != 0;
            expected[bits] = count;
        }
        Require(many == expected, "seeded count formats agree for reordered and duplicate qubits");

        if (method == SimulationType::kDensityMatrix && legacy)
        {
            sim->SetSeed(891);
            std::unordered_map<Types::qubit_t, Types::qubit_t> reference;
            for (size_t shot = 0; shot < shots; ++shot)
            {
                const auto raw = sim->MeasureNoCollapse();
                Types::qubit_t outcome = 0;
                for (size_t i = 0; i < selected.size(); ++i)
                    if ((raw >> selected[i]) & 1)
                        outcome |= Types::qubit_t(1) << i;
                ++reference[outcome];
            }
            Require(packed == reference, "density batch matches repeated seeded draws");
            Require(sim->MeasureNoCollapse() == next, "density batch preserves RNG consumption");
        }

        Require(sim->AllProbabilities() == before, "batch sampling must not collapse the state");
        sim->ApplyX(3);
    }
}

void PathIntegralWideSampling()
{
    for (const size_t width : {40, 64, 70, 130, 1024})
    {
        auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, SimulationType::kPathIntegral);
        sim->AllocateQubits(width);
        sim->Initialize();
        sim->ApplyX(width - 1);
        sim->ApplyX(32);
        sim->ApplyH(0);
        sim->ApplyCX(0, width - 2);
        const Types::qubits_vector selected{width - 1, 32, 0, width - 2, 32};
        const std::vector<bool> low{true, true, false, false, true}, high{true, true, true, true, true};
        sim->SetSeed(891);
        const auto counts = sim->SampleCountsMany(selected, 128);
        Require(counts.size() == 2 && counts.count(low) && counts.count(high) && counts.at(low) + counts.at(high) == 128,
                "path integral sampling preserves high bits and qubit order");
        sim->SetSeed(891);
        Require(sim->SampleCountsMany(selected, 128) == counts, "path integral sampling preserves state and seeded draws");
        if (width <= 1024)
        {
            sim->SetSeed(891);
            const auto packed = sim->SampleCounts(selected, 128);
            Require(packed.size() == 2 && packed.at(19) == counts.at(low) && packed.at(31) == counts.at(high),
                    "path integral packed counts preserve bits above 32");
        }
        sim->SetSeed(12);
        const auto single = sim->SampleCounts(selected, 1).begin()->first;
        sim->SetSeed(12);
        const auto singleMany = sim->SampleCountsMany(selected, 1).begin()->first;
        for (size_t bit = 0; bit < selected.size(); ++bit)
            Require(singleMany[bit] == bool((single >> bit) & 1), "wide single-shot count formats disagree");
    }
}

void AdaptiveStreams(SimulationType method, SimulatorType type = SimulatorType::kQCSim)
{
    const size_t width = method == SimulationType::kPathIntegral ? 70 : method == SimulationType::kDensityMatrix ? 8 : 17;
    auto sim = SimulatorsFactory::CreateSimulatorUnique(type, method);
    sim->SetMultithreading(false);
    sim->AllocateQubits(width);
    sim->Initialize();
    for (size_t q = 0; q < std::min<size_t>(width, 17); ++q)
        sim->ApplyH(q);
    if (width > 17)
        sim->ApplyX(width - 1);
    const Types::qubits_vector selected{width - 1, 0, 3, 0, 1, 7};
    for (size_t shots : {size_t{1}, size_t{10000}, size_t{100000}})
    {
        sim->SetMultithreading(false);
        sim->SetSeed(0);
        const auto expected = sim->SampleCountsMany(selected, shots);
        const auto next = sim->SampleCountsMany(selected, 1000);
        for (int threads : {1, 3, 8})
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif
            sim->SetMultithreading(true);
            sim->SetSeed(0);
            Require(sim->SampleCountsMany(selected, shots) == expected, "sampler changed with physical thread count");
            Require(sim->SampleCountsMany(selected, 1000) == next, "sampler changed RNG consumption with threads");
        }
    }
    sim->SetMultithreading(false);
    for (uint64_t seed : {uint64_t{0}, uint64_t{1} << 40, UINT64_MAX})
    {
        sim->SetSeed(seed);
        const auto parent = sim->SampleCountsMany(selected, 8192);
        auto first = sim->Clone();
        auto second = sim->Clone();
        auto grandchild = first->Clone();
        const auto a = first->SampleCountsMany(selected, 8192);
        const auto b = second->SampleCountsMany(selected, 8192);
        const auto c = grandchild->SampleCountsMany(selected, 8192);
        Require(parent != a && a != b && a != c && b != c, "clones duplicated a parent, sibling or grandchild stream");
        sim->SetSeed(seed);
        // An explicitly addressed execution clone cannot consume ordinary fork ids.
        auto execution = sim->CloneForExecution(101);
        Require(execution->GetConfiguration("seed") == "101", "execution clone did not use its explicit stream");
        auto replay = sim->Clone();
        Require(replay->SampleCountsMany(selected, 8192) == a, "seeded clone tree did not replay");
        Require(replay->Clone()->SampleCountsMany(selected, 8192) == c, "seeded grandchild did not replay");
        Require(sim->SampleCountsMany(selected, 8192) == parent, "cloning advanced the parent's sampling RNG");
    }
    auto unseeded = SimulatorsFactory::CreateSimulatorUnique(type, method);
    unseeded->AllocateQubits(3);
    unseeded->Initialize();
    unseeded->ApplyH(0);
    unseeded->ApplyH(1);
    unseeded->ApplyH(2);
    const auto root = unseeded->GetConfiguration("sampling_seed");
    std::vector<double> auxiliary;
    for (size_t draw = 0; draw < 20; ++draw)
        auxiliary.push_back(unseeded->RandomUniform());
    auto child = unseeded->Clone();
    const auto draws = child->SampleCountsMany({0, 1, 2}, 8192);
    Require(child->GetConfiguration("sampling_seed") != root, "unseeded clone copied its parent's effective seed");
    Require(draws != unseeded->Clone()->SampleCountsMany({0, 1, 2}, 8192), "unseeded siblings duplicated their draws");
    if (type == SimulatorType::kQCSim)
    {
        auto fresh = SimulatorsFactory::CreateSimulatorUnique(type, method);
        fresh->AllocateQubits(2);
        fresh->Initialize();
        const auto previous = fresh->GetConfiguration("sampling_seed");
        fresh->Clear();
        fresh->AllocateQubits(2);
        fresh->Initialize();
        Require(fresh->GetConfiguration("sampling_seed") != previous, "unseeded reinitialization restarted its old stream");
        const auto current = fresh->GetConfiguration("sampling_seed");
        const auto readout = fresh->RandomUniform();
        fresh->SetSeed(std::stoull(current));
        Require(fresh->RandomUniform() == readout, "reinitialized wrapper lost its effective readout seed");
    }
    unseeded->SetSeed(std::stoull(root));
    for (const auto expected : auxiliary)
        Require(unseeded->RandomUniform() == expected, "recorded random root cannot replay the readout stream");
    Require(unseeded->Clone()->SampleCountsMany({0, 1, 2}, 8192) == draws, "recorded random root cannot replay clone streams");
}

// Structural changes are part of a continuing execution. Compare the draws
// around them, not just the final state or a replay after explicitly reseeding.
void CompositeStructureStreams()
{
    using Private::IndividualSimulator;
    auto make = [](size_t width, uint64_t seed) {
        auto sim = std::make_unique<IndividualSimulator>(SimulatorType::kQCSim);
        sim->AllocateQubits(width);
        sim->Initialize();
        sim->SetSeed(seed);
        for (size_t q = 0; q < width; ++q)
            sim->GetQubitsMap()[q] = q;
        return sim;
    };
    for (bool threaded : {false, true})
        for (const char *policy : {"reproducible_v1", "legacy"})
        {
            const size_t otherWidth = threaded ? 13 : 1; // exercise the OpenMP join as well
            auto parent = make(1, 173), reference = make(1, 173), other = make(otherWidth, 901);
            parent->Configure("sampling_policy", policy);
            reference->Configure("sampling_policy", policy);
            parent->SetMultithreading(threaded);
            reference->SetMultithreading(threaded);
            other->GetQubitsMap().clear();
            for (size_t q = 0; q < otherWidth; ++q)
                other->GetQubitsMap()[q + 1] = q;
            parent->ApplyH(0);
            reference->ApplyH(0);
            for (size_t draw = 0; draw < 7; ++draw)
                Require(parent->MeasureNoCollapse() == reference->MeasureNoCollapse(), "initial native stream");
            Require(parent->SampleCounts({0}, 517) == reference->SampleCounts({0}, 517), "initial batch stream");
            Require(parent->Clone()->GetConfiguration("seed") == reference->Clone()->GetConfiguration("seed"), "initial clone stream");
            // Join a deterministic high qubit; the low marginal is unchanged.
            std::vector<size_t> mapping(otherWidth + 1, 1);
            mapping[0] = 0;
            parent->Join(0, other, mapping, threaded);
            for (size_t draw = 0; draw < 32; ++draw)
                Require(parent->MeasureNoCollapse() == reference->MeasureNoCollapse(), "join restarted the native RNG");
            Require(parent->SampleCounts({0}, 517) == reference->SampleCounts({0}, 517), "join restarted the batch RNG");
            // The deterministic measurement still consumes one native draw.
            parent->Measure({1});
            reference->MeasureNoCollapse();
            auto detached = parent->Split(1, false, threaded, 902);
            for (size_t draw = 0; draw < 32; ++draw)
                Require(parent->MeasureNoCollapse() == reference->MeasureNoCollapse(), "split restarted the native RNG");
            Require(parent->SampleCounts({0}, 517) == reference->SampleCounts({0}, 517), "split restarted the batch RNG");
            Require(parent->Clone()->GetConfiguration("seed") == reference->Clone()->GetConfiguration("seed"),
                    "rebuilding a component restarted its clone ordinal");
            parent->SaveState();
            parent->ApplyRy(0, 0.37);
            parent->SaveState(); // replace, rather than append to, the snapshot
            parent->ApplyH(0);
            parent->RestoreState();
            reference->ApplyRy(0, 0.37);
            for (size_t draw = 0; draw < 32; ++draw)
                Require(parent->MeasureNoCollapse() == reference->MeasureNoCollapse(), "restore restarted the native RNG");
            Require(parent->SampleCounts({0}, 517) == reference->SampleCounts({0}, 517), "restore restarted the batch RNG");
            Require(detached->GetConfiguration("seed") == "902", "detached component copied its parent's seed");
        }
}

void CompositeCollapseDistribution()
{
    for (const char *policy : {"reproducible_v1", "legacy"})
        for (bool fusion : {false, true})
            for (bool threaded : {false, true})
            {
                auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kCompositeQCSim, SimulationType::kStatevector);
                sim->Configure("sampling_policy", policy);
                sim->Configure("gate_fusion", fusion ? "true" : "false");
                sim->SetMultithreading(threaded);
                sim->AllocateQubits(2);
                sim->Initialize();
                std::array<size_t, 4> counts{}, detachedCounts{};
                for (uint64_t seed = 0; seed < 1024; ++seed)
                {
                    sim->SetSeed(seed);
                    sim->Reset();
                    sim->ApplyH(0);
                    sim->ApplyH(1);
                    sim->ApplyCX(0, 1); // |++>, with a joined component
                    const auto bits = sim->MeasureMany({1, 0});
                    ++counts[size_t(bits[0]) | (size_t(bits[1]) << 1)];
                    // This qubit now lives in a new component. Reusing the
                    // source seed would correlate its next draw with bits[0].
                    sim->ApplyH(1);
                    ++detachedCounts[size_t(bits[0]) | (sim->Measure({1}) << 1)];
                }
                for (const auto count : counts)
                    Require(count > 190 && count < 325, "composite collapse correlated independent qubits");
                for (const auto count : detachedCounts)
                    Require(count > 190 && count < 325, "detached component reused the source RNG stream");
            }
}

void CompositeBatchReference()
{
    Private::ImmediateCompositeSimulator sim(SimulatorType::kQCSim);
    sim.SetMultithreading(false);
    sim.AllocateQubits(35);
    sim.Initialize();
    std::array<Private::QCSimStatevectorSimulator, 3> reference;
    for (size_t group = 0; group < 3; ++group)
    {
        const size_t width = group == 2 ? 1 : 17, first = group * 17;
        reference[group].SetMultithreading(false);
        reference[group].AllocateQubits(width);
        reference[group].Initialize();
        for (size_t bit = 0; bit < width; ++bit)
        {
            const double angle = 0.3 + 0.079 * ((bit * 17 + group * 3) % 23);
            sim.ApplyRy(first + bit, angle);
            reference[group].ApplyRy(bit, angle);
            if (bit)
            {
                sim.ApplyCX(first + bit - 1, first + bit);
                reference[group].ApplyCX(bit - 1, bit);
            }
        }
    }
    const Types::qubits_vector selected{31, 0, 17, 3, 31, 16};
    for (size_t shots : {size_t{10000}, size_t{100000}})
    {
        // Scalar component-by-component oracle with the original RNG streams.
        std::array<std::mt19937_64, 3> engines;
        for (size_t group = 0; group < 3; ++group) engines[group].seed(IState::DeriveSeed(919, group * 17));
        auto scalar = [&](const Types::qubits_vector &requested, size_t count) {
            std::vector<Private::QCSimStatevectorSimulator::SamplingPlan> plans;
            for (auto &component : reference) plans.push_back(component.PrepareSampling(count));
            std::unordered_map<std::vector<bool>, Types::qubit_t> expected;
            for (size_t shot = 0; shot < count; ++shot)
            {
                std::array<size_t, 3> rows;
                for (size_t group = 0; group < 3; ++group) rows[group] = plans[group].Sample(Utils::RandomStream::Uniform(engines[group]));
                std::vector<bool> bits(requested.size());
                for (size_t bit = 0; bit < requested.size(); ++bit)
                    bits[bit] = ((rows[requested[bit] / 17] >> (requested[bit] % 17)) & 1) != 0;
                ++expected[bits];
            }
            return expected;
        };
        const auto expected = scalar(selected, shots);
        const Types::qubits_vector followup{34, 17, 0};
        const auto next = scalar(followup, 517);
        for (int threads : {0, 1, 3, 8})
        {
#ifdef _OPENMP
            omp_set_num_threads(std::max(1, threads));
#endif
            sim.SetMultithreading(threads != 0);
            sim.SetSeed(919);
            Require(sim.SampleCountsMany(selected, shots) == expected, "composite batch differs from scalar draws");
            Require(sim.SampleCountsMany(followup, 517) == next, "composite batch changed an unselected component's RNG");
        }
    }
    // Force several bounded batches and CDF fallback within a shared budget.
    sim.Configure("sampling_max_memory_mb", "1");
    sim.SetMultithreading(false);
    sim.SetSeed(77);
    const auto bounded = sim.SampleCountsMany(selected, 100000);
    const auto next = sim.SampleCountsMany({34, 1}, 1000);
    sim.SetMultithreading(true);
    sim.SetSeed(77);
    Require(sim.SampleCountsMany(selected, 100000) == bounded, "bounded composite batches changed with threads");
    Require(sim.SampleCountsMany({34, 1}, 1000) == next, "bounded batches changed RNG continuation");
}

void CompositeSingleComponent()
{
    Private::ImmediateCompositeSimulator sim(SimulatorType::kQCSim);
    Private::QCSimStatevectorSimulator reference;
    sim.AllocateQubits(4);
    sim.Initialize();
    reference.AllocateQubits(4);
    reference.Initialize();
    // Joining in this order maps global [3,1,0,2] to local [0,1,2,3].
    sim.ApplyH(3); reference.ApplyH(0);
    sim.ApplyCX(3, 1); reference.ApplyCX(0, 1);
    sim.ApplyRy(0, 0.37); reference.ApplyRy(2, 0.37);
    sim.ApplyCX(3, 0); reference.ApplyCX(0, 2);
    sim.ApplyCX(0, 2); reference.ApplyCX(2, 3);
    for (const size_t shots : {size_t{1}, size_t{31}, size_t{100000}})
        for (bool threads : {false, true})
        {
            sim.SetMultithreading(threads);
            reference.SetMultithreading(threads);
            sim.SetSeed(881);
            reference.SetSeed(IState::DeriveSeed(881, 3));
            Require(sim.SampleCountsMany({2, 3, 1, 2, 0}, shots) == reference.SampleCountsMany({3, 0, 1, 3, 2}, shots),
                    "single component forwarding changed mapping or RNG");
        }
    sim.SaveState();
    const auto before = sim.AllProbabilities();
    sim.SampleCounts({3, 0}, 10000);
    Require(sim.AllProbabilities() == before, "forwarding collapsed the state");
    sim.ApplyX(0);
    sim.RestoreState();
    Require(sim.AllProbabilities() == before, "forwarding destroyed the saved state");
}

void CompositeWideBatches()
{
    Private::ImmediateCompositeSimulator sim(SimulatorType::kQCSim);
    sim.AllocateQubits(130);
    sim.Initialize();
    sim.ApplyH(0);
    sim.ApplyCX(0, 129);
    sim.ApplyX(32);
    sim.ApplyX(64);
    const Types::qubits_vector selected{129, 64, 0, 32, 129, 65};
    for (size_t shots : {size_t{1}, size_t{2}, size_t{10000}})
    {
        sim.SetSeed(712);
        const auto packed = sim.SampleCounts(selected, shots);
        sim.SetSeed(712);
        const auto many = sim.SampleCountsMany(selected, shots);
        for (auto [outcome, count] : packed)
        {
            Require(outcome == 10 || outcome == 31, "wide composite lost high global IDs");
            std::vector<bool> bits(selected.size());
            for (size_t bit = 0; bit < bits.size(); ++bit) bits[bit] = ((outcome >> bit) & 1) != 0;
            Require(many.at(bits) == count, "wide composite count formats disagree");
        }
    }
    Types::qubits_vector all(130);
    std::iota(all.begin(), all.end(), 0);
    sim.SetSeed(781);
    const auto many = sim.SampleCountsMany(all, 10000);
    Require(many.size() == 2, "wide joint sampling lost the Bell pair correlation");
    for (const auto &[bits, count] : many)
    {
        Require(bits[0] == bits[129] && bits[32] && bits[64], "multiword composite projection is incorrect");
        for (size_t bit = 1; bit < 129; ++bit)
            if (bit != 32 && bit != 64) Require(!bits[bit], "multiword projection set an unrelated bit");
    }
    sim.SetSeed(781);
    const auto duplicate = sim.SampleCountsMany(Types::qubits_vector(130, 129), 10000);
    for (const auto &[bits, count] : duplicate)
        Require(std::all_of(bits.begin(), bits.end(), [&](bool bit) { return bit == bits[0]; }), "wide duplicate outputs disagree");
    bool rejected = false;
    try { sim.SampleCounts(all, 1); }
    catch (const std::invalid_argument &) { rejected = true; }
    Require(rejected, "wide composite packed output was accepted");
    struct Observer : ISimulatorObserver
    {
        size_t calls = 0;
        void Update(const Types::qubits_vector &) override { ++calls; }
    };
    const auto observer = std::make_shared<Observer>();
    sim.RegisterObserver(observer);
    rejected = false;
    try { sim.SampleCountsMany({130}, 1000); }
    catch (const std::out_of_range &) { rejected = true; }
    Require(rejected, "invalid composite qubit was accepted");
    sim.SampleCountsMany(selected, 1000);
    Require(observer->calls == 1, "sampling error disabled observer notifications");
}

void CompositeTightBudget()
{
    Private::ImmediateCompositeSimulator sim(SimulatorType::kQCSim);
    sim.Configure("sampling_max_memory_mb", "1");
    sim.SetMultithreading(false);
    sim.AllocateQubits(42);
    sim.Initialize();
    // Two 21-qubit components used to reserve more index space than the shared
    // budget, even though each has only two populated rows.
    for (size_t first : {size_t{0}, size_t{21}})
    {
        sim.ApplyH(first);
        for (size_t q = 1; q < 21; ++q) sim.ApplyCX(first + q - 1, first + q);
    }
    const Types::qubits_vector selected{41, 0, 21, 20, 41};
    sim.SetSeed(819);
    const auto expected = sim.SampleCounts(selected, 131072);
    Require(expected.size() == 4, "small composite budget lost support");
    for (auto [bits, count] : expected)
        Require((bits == 0 || bits == 10 || bits == 21 || bits == 31) && count > 30000 && count < 35500,
                "small composite budget changed correlations or probabilities");
    sim.SetSeed(819);
    sim.SetMultithreading(true);
    Require(sim.SampleCounts(selected, 131072) == expected, "grown composite indexes depend on threading");
}

void SamplingConfigurationValidation()
{
    const std::vector<std::pair<std::string, std::string>> invalid{
        {"sampling_policy", "unknown"}, {"sampling_policy", ""},
        {"sampling_max_memory_mb", "0"}, {"sampling_max_memory_mb", "-1"},
        {"sampling_max_memory_mb", "1x"}, {"sampling_max_memory_mb", " 1"},
        {"sampling_max_memory_mb", "18446744073709551616"},
        {"sampling_max_memory_mb", std::to_string((std::numeric_limits<size_t>::max() >> 20) + 1)}};
    for (auto backend : {SimulatorType::kCompositeQCSim, SimulatorType::kQCSim})
    {
        auto sim = SimulatorsFactory::CreateSimulatorUnique(backend, SimulationType::kStatevector);
        sim->Configure("sampling_policy", "reproducible_v1");
        sim->Configure("sampling_max_memory_mb", "1");
        for (bool initialized : {false, true})
        {
            if (initialized)
            {
                sim->AllocateQubits(3);
                sim->Initialize();
                sim->ApplyH(0);
                sim->ApplyCX(0, 2);
            }
            const auto before = sim->GetConfigMap();
            for (const auto &[key, value] : invalid)
            {
                bool rejected = false;
                try { sim->Configure(key.c_str(), value.c_str()); }
                catch (const std::invalid_argument &) { rejected = true; }
                Require(rejected && sim->GetConfigMap() == before, "bad sampling setting was accepted or mutated configuration");
            }
        }
        auto clone = sim->Clone();
        Require(clone->SampleCounts({2, 0}, 1000).size() == 2, "rejected sampling configuration poisoned cloning");
        Require(sim->SampleCountsMany({2, 0}, 1000).size() == 2, "rejected sampling configuration poisoned sampling");
    }
}

void CompositeSingleShotStreams()
{
    // Compare a partial query (skipping an entire component) with sampling all
    // components and discarding the extra output. Subsequent native draws must
    // match, including after a saved-state restore.
    Private::ImmediateCompositeSimulator partial(SimulatorType::kQCSim), full(SimulatorType::kQCSim);
    for (auto *sim : {&partial, &full})
    {
        sim->AllocateQubits(19);
        sim->Initialize();
        for (size_t q = 0; q < 19; ++q)
        {
            sim->ApplyRy(q, 0.21 + 0.087 * q);
            if (q > 1) sim->ApplyCX(q - 1, q);
        }
        sim->SetSeed(471);
        sim->SaveState();
    }
    for (size_t shot = 0; shot < 64; ++shot)
    {
        const auto first = partial.SampleCounts({0}, 1).begin()->first;
        const auto both = full.SampleCounts({0, 18}, 1).begin()->first;
        Require(first == (both & 1), "skipped component changed selected single-shot sample");
        Require(partial.SampleCountsMany({18, 0, 17}, 1) == full.SampleCountsMany({18, 0, 17}, 1),
                "unmeasured component consumed the wrong native draw");
    }
    partial.RestoreState();
    full.RestoreState();
    Require(partial.SampleCounts({18, 0, 17}, 4096) == full.SampleCounts({18, 0, 17}, 4096),
            "single-shot skip changed state or batch stream");

    // The discard helper must use the native distribution, not assume an
    // implementation-specific number of underlying engine words.
    Private::SamplingQubitRegister actual(3), reference(3);
    actual.SetSeed(123);
    reference.SetSeed(123);
    QC::Gates::HadamardGate<> h;
    for (unsigned q = 0; q < 3; ++q) { actual.ApplyGate(h, q); reference.ApplyGate(h, q); }
    for (size_t draw = 0; draw < 100; ++draw)
    {
        actual.DiscardMeasurementDraw();
        reference.MeasureNoCollapse();
        Require(actual.MeasureNoCollapse() == reference.MeasureNoCollapse(), "native distribution discard changed RNG position");
    }
    Require(actual.getRegisterStorage() == reference.getRegisterStorage(), "discard changed amplitudes");
}

void PathIntegralTightBudget()
{
    auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, SimulationType::kPathIntegral);
    sim->AllocateQubits(130);
    sim->Initialize();
    for (size_t q = 0; q < 17; ++q) sim->ApplyH(q);
    sim->ApplyX(129);
    const Types::qubits_vector selected{129, 16, 0, 64, 16};
    // 131072 rows need a 2 MiB snapshot. The 1 MiB request must sweep the
    // original 130-bit labels instead of allocating that snapshot or failing.
    sim->Configure("sampling_max_memory_mb", "4");
    sim->SetSeed(748);
    const auto expected = sim->SampleCounts(selected, 4096);
    const auto next = sim->SampleCountsMany(selected, 4096);
    sim->Configure("sampling_max_memory_mb", "1");
    sim->SetSeed(748);
    const auto actual = sim->SampleCounts(selected, 4096);
    Require(actual == expected, "streaming path integral changed exact uniform probabilities or labels");
    Require(sim->SampleCountsMany(selected, 4096) == next, "streaming path integral changed RNG continuation");
    Types::qubits_vector all(130);
    std::iota(all.begin(), all.end(), 0);
    const auto wide = sim->SampleCountsMany(all, 517);
    size_t total = 0;
    for (const auto &[bits, count] : wide)
    {
        Require(bits.size() == 130 && bits[129], "streaming path integral truncated wide labels");
        for (size_t q = 17; q < 129; ++q) Require(!bits[q], "streaming path integral corrupted labels");
        total += count;
    }
    Require(total == 517, "streaming path integral lost shots");
    sim->SetSeed(748);
    Require(sim->SampleCounts(selected, 4096) == actual, "streaming sampling changed the state");
}

void SamplingRestoreStreams()
{
    for (const auto method : {SimulationType::kStatevector, SimulationType::kDensityMatrix, SimulationType::kPathIntegral})
        for (const char *policy : {"reproducible_v1", "legacy"})
        {
            auto make = [&] {
                auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, method);
                sim->Configure("sampling_policy", policy);
                sim->AllocateQubits(3);
                sim->Initialize();
                sim->ApplyH(0);
                sim->ApplyH(1);
                sim->ApplyH(2);
                sim->SetSeed(219);
                return sim;
            };
            auto sim = make(), reference = make();
            const Types::qubits_vector selected{2, 0, 1};
            sim->SaveState();
            Require(sim->SampleCounts(selected, 517) == reference->SampleCounts(selected, 517), "initial restore batch");
            for (size_t draw = 0; draw < 32; ++draw)
            {
                Require(sim->Measure(selected) == reference->Measure(selected), "restore changed measurement stream");
                sim->RestoreState();
                reference->Reset();
                reference->ApplyH(0);
                reference->ApplyH(1);
                reference->ApplyH(2);
                Require(sim->RandomUniform() == reference->RandomUniform(), "restore changed readout stream");
                Require(sim->SampleCounts(selected, 517) == reference->SampleCounts(selected, 517), "restore restarted batch stream");
            }
        }
}

void DensityProbabilities()
{
    auto sim = SimulatorsFactory::CreateSimulatorUnique(SimulatorType::kQCSim, SimulationType::kDensityMatrix);
    sim->AllocateQubits(3);
    sim->Initialize();
    sim->ApplyRy(0, 2. * std::asin(0.5));
    sim->ApplyCX(0, 2);
    sim->SetSeed(123);
    const auto counts = sim->SampleCounts({2, 0, 2}, 100000);
    Require(counts.size() == 2 && std::abs(double(counts.at(7)) / 100000. - 0.25) < 0.01,
            "density marginal lost correlations or squared probabilities");
}

void StatevectorSupport()
{
    Private::QCSimStatevectorSimulator sim;
    Eigen::VectorXcd state = Eigen::VectorXcd::Zero(size_t{1} << 20);
    state[0] = std::sqrt(0.75);
    state[15] = 0.5;
    sim.InitializeState(20, state);
    Require(sim.PrepareSampling(10000).Size() == 16, "imported sparse prefix was not tracked exactly");
    sim.ApplyX(19);
    Require(sim.SampleCounts({19}, 10000).at(1) == 10000, "a gate escaped the tracked support");
    sim.SaveState();
    sim.Reset();
    sim.RestoreState();
    Require(sim.SampleCounts({19}, 10000).at(1) == 10000, "restore truncated the saved support");
    sim.Reset();
    Eigen::Matrix2cd x;
    x << 0., 1., 1., 0.;
    sim.ApplyGenericOneQubitGate(19, x);
    Require(sim.SampleCounts({19}, 10000).at(1) == 10000, "generic gate escaped the tracked support");
}

struct DensitySamplingProbe : Private::QCSimDensityMatrixSimulator
{
    void Population(size_t row, std::complex<double> value)
    {
        const_cast<Eigen::MatrixXcd &>(densityMatrix->getDensityMatrix())(row, row) = value;
    }
};

void DensityValidation()
{
    DensitySamplingProbe sim;
    sim.AllocateQubits(2);
    sim.Initialize();
    sim.Population(0, 0.1875);
    sim.Population(3, 0.0625);
    sim.Population(1, -1E-13);
    sim.SetSeed(123);
    const auto counts = sim.SampleCounts({0, 1}, 100000);
    Require(counts.size() == 2 && std::abs(double(counts.at(3)) / 100000. - 0.25) < 0.01, "density retained mass was not normalized");
    for (const auto invalid : {std::complex<double>{-1E-4, 0.}, {0., 1E-4}, {std::numeric_limits<double>::quiet_NaN(), 0.},
                               {std::numeric_limits<double>::infinity(), 0.}})
    {
        sim.Population(1, invalid);
        bool rejected = false;
        try { sim.SampleCounts({0}, 10000); }
        catch (const std::domain_error &) { rejected = true; }
        Require(rejected, "invalid density population was accepted");
    }
    sim.Population(0, 0.);
    sim.Population(1, 0.);
    sim.Population(3, 0.);
    bool rejected = false;
    try { sim.SampleCounts({0}, 10000); }
    catch (const std::domain_error &) { rejected = true; }
    Require(rejected, "zero-trace density matrix was accepted");
}

void CliffordWideSampling()
{
    auto sim = SimulatorsFactory::CreateSimulator(SimulatorType::kQCSim, SimulationType::kStabilizer);
    sim->AllocateQubits(65);
    sim->Initialize();
    for (unsigned bit : {32, 63})
    {
        const Types::qubit_t outcome = Types::qubit_t(1) << bit;
        Require(sim->Probability(outcome) == 0.0, "high outcome bits must not alias zero");
        sim->ApplyX(bit);
        Require(sim->Probability(outcome) == 1.0 && sim->Probability(0) == 0.0, "high outcome bits must reach the Clifford backend");
        sim->ApplyX(bit);
    }
    sim->SaveState();
    sim->ApplyX(64);
    sim->ApplyH(0);
    sim->ApplyCX(0, 63);
    const Types::qubits_vector selected{64, 63, 0, 63};
    sim->SetSeed(891);
    const auto packed = sim->SampleCounts(selected, 128);
    Require(packed.size() == 2 && packed.count(1) && packed.count(15) && packed.at(1) + packed.at(15) == 128, "wide ordered marginal counts");
    sim->SetSeed(891);
    Require(sim->SampleCounts(selected, 128) == packed, "warm seeded Clifford counts");
    sim->SetSeed(891);
    const auto many = sim->SampleCountsMany(selected, 128);
    Require(many.size() == 2 && many.at(std::vector<bool>{true, false, false, false}) == packed.at(1) &&
                many.at(std::vector<bool>{true, true, true, true}) == packed.at(15),
            "packed/vector counts agree");
    bool rejected = false;
    try
    {
        sim->SampleCounts(Types::qubits_vector(65, 0), 1);
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Require(rejected, "reject packed outcomes wider than size_t");
    rejected = false;
    try
    {
        sim->SampleCountsMany({65}, 1);
    }
    catch (const std::out_of_range &)
    {
        rejected = true;
    }
    Require(rejected, "reject invalid measured qubits");
    sim->RestoreState();
    Require(sim->Probability(0) == 1.0, "sampling must preserve the caller's saved state");
}

// Stabilizer backends follow maestro's multithreading flag whether it is set
// before or after the backend is created.
template <class Backend> struct StabilizerThreadsProbe : Backend
{
    bool BackendMultithreading() const
    {
        if constexpr (std::is_same_v<Backend, Private::QCSimStabilizerSimulator>)
            return this->cliffordSimulator->GetMultithreading();
        else
            return this->extendedStabilizer->GetMultithreading();
    }
};

template <class Backend> void CheckStabilizerMultithreading()
{
    for (const bool enable : {false, true})
    {
        StabilizerThreadsProbe<Backend> sim;
        const std::string method = sim.MethodName();
        sim.SetMultithreading(enable);
        sim.AllocateQubits(4);
        sim.Initialize();
        Require(sim.BackendMultithreading() == enable, method + " ignored multithreading set before creation");
        sim.SetMultithreading(!enable);
        Require(sim.BackendMultithreading() == !enable, method + " ignored multithreading set after creation");
    }
}

void StabilizerMultithreading()
{
    CheckStabilizerMultithreading<Private::QCSimStabilizerSimulator>();
    CheckStabilizerMultithreading<Private::QCSimExtendedStabilizerSimulator>();
}

void Individual(SimulatorType type)
{
    for (const std::vector<Types::qubit_t> &ids : {std::vector<Types::qubit_t>{0, 1, 2}, {5, 7, 9}, {9, 5, 7}})
    {
        Private::IndividualSimulator sim(type);
        for (size_t i = 0; i < ids.size(); ++i)
            sim.GetQubitsMap()[ids[i]] = i;
        CheckSamples(sim, ids);
    }
}
} // namespace

int main(int argc, char **argv)
{
    int failed = 0, passed = 0;
    auto run = [&](const std::string &name, auto test) {
        try
        {
            test();
            ++passed;
            std::cout << "PASS " << name << std::endl;
        }
        catch (const std::exception &error)
        {
            ++failed;
            std::cerr << "FAIL " << name << ": " << error.what() << std::endl;
        }
    };
    const bool gpuOnly = argc == 2 && std::string(argv[1]) == "--gpu";
    if (gpuOnly)
    {
#ifdef __linux__
        const int count = SimulatorsFactory::GetGpuDeviceCount();
        if (count < 0)
        {
            std::cerr << "CUDA discovery failed\n";
            return 1;
        }
        if (count == 0)
        {
            std::cout << "SKIP: GPU plugin/device unavailable\n";
            return 77;
        }
        for (const auto method : {SimulationType::kStatevector, SimulationType::kMatrixProductState, SimulationType::kTensorNetwork,
                                  SimulationType::kDensityMatrix, SimulationType::kMatrixProductOperator, SimulationType::kPauliPropagator})
            run("GPU method " + std::to_string(int(method)), [&] { Backend(SimulatorType::kGpuSim, method); });
#else
        return 77;
#endif
    }
    else
    {
        for (const auto method : {SimulationType::kStatevector, SimulationType::kMatrixProductState, SimulationType::kTensorNetwork,
                                  SimulationType::kDensityMatrix, SimulationType::kMatrixProductOperator, SimulationType::kStabilizer,
                                  SimulationType::kExtendedStabilizer, SimulationType::kPauliPropagator, SimulationType::kPathIntegral})
            run("QCSim method " + std::to_string(int(method)), [&] { Backend(SimulatorType::kQCSim, method); });
        for (const auto method : {SimulationType::kStatevector, SimulationType::kDensityMatrix})
            run("QCSim seeded batch sampling " + std::to_string(int(method)), [&] { SeededBatchSampling(method); });
        run("QCSim legacy density sampling sequence", [&] { SeededBatchSampling(SimulationType::kDensityMatrix, true); });
        for (const auto method : {SimulationType::kStatevector, SimulationType::kDensityMatrix, SimulationType::kPathIntegral})
            run("QCSim adaptive streams " + std::to_string(int(method)), [&] { AdaptiveStreams(method); });
        run("Composite adaptive streams", [&] { AdaptiveStreams(SimulationType::kStatevector, SimulatorType::kCompositeQCSim); });
        run("Composite structural RNG continuity", CompositeStructureStreams);
        run("Composite collapse independence", CompositeCollapseDistribution);
        run("Composite batch scalar reference and memory budget", CompositeBatchReference);
        run("Composite single-component forwarding", CompositeSingleComponent);
        run("Composite wide batched outputs and observers", CompositeWideBatches);
        run("Composite growing indexes under shared budget", CompositeTightBudget);
        run("Sampling configuration validation", SamplingConfigurationValidation);
        run("Composite single-shot native stream continuation", CompositeSingleShotStreams);
        run("Path integral bounded streaming and wide labels", PathIntegralTightBudget);
        run("Sampling restore RNG continuity", SamplingRestoreStreams);
        run("Density joint marginal probabilities", DensityProbabilities);
        run("Density population validation", DensityValidation);
        run("Statevector support tracking", StatevectorSupport);
        run("QCSim path integral wide sampling", PathIntegralWideSampling);
        run("QCSim Clifford wide probabilities and marginal sampling", CliffordWideSampling);
        run("QCSim stabilizer backends follow multithreading", StabilizerMultithreading);
        run("QCSim MPS collapse sampler", [&] { Backend(SimulatorType::kQCSim, SimulationType::kMatrixProductState, "mps_apply_measure"); });
        run("Composite QCSim", [&] { Backend(SimulatorType::kCompositeQCSim, SimulationType::kStatevector); });
        run("Individual QCSim mappings", [&] { Individual(SimulatorType::kQCSim); });
#ifndef NO_QISKIT_AER
        for (const auto method :
             {SimulationType::kStatevector, SimulationType::kDensityMatrix, SimulationType::kStabilizer, SimulationType::kExtendedStabilizer})
            run("Aer method " + std::to_string(int(method)), [&] { Backend(SimulatorType::kQiskitAer, method); });
        for (const char *algorithm : {"mps_probabilities", "mps_apply_measure"})
            run(std::string("Aer MPS ") + algorithm, [&] { Backend(SimulatorType::kQiskitAer, SimulationType::kMatrixProductState, algorithm); });
        run("Composite Aer", [&] { Backend(SimulatorType::kCompositeQiskitAer, SimulationType::kStatevector); });
        run("Individual Aer mappings", [&] { Individual(SimulatorType::kQiskitAer); });
#else
        std::cout << "Aer coverage disabled in this build\n";
#endif
    }
    std::cout << passed << " passed, " << failed << " failed\n";
    return failed ? 1 : 0;
}
