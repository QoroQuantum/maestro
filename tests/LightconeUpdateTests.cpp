#include "Circuit/Factory.h"
#include "Execution/Estimation.h"
#include "Execution/Sampling.h"
#include "Execution/SimulatorConfig.h"
#include "Execution/Transforms/BoundedPauli.h"
#include "Execution/Transforms/ConeAnalysis.h"
#include "Execution/Transforms/PauliCone.h"
#include "Execution/Transforms/TransformPipeline.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace MaestroExecution;
using Circuit = Circuits::Circuit<double>;
using Factory = Circuits::CircuitFactory<double>;
using Counts = Network::INetwork<double>::ExecuteResults;

void Check(bool ok, const char *message)
{
    if (!ok)
        throw std::runtime_error(message);
}

void ResetTargets()
{
    for (auto method : {Simulators::SimulationType::kDensityMatrix, Simulators::SimulationType::kMatrixProductOperator})
        for (const auto &targets : {std::vector<bool>{true}, std::vector<bool>{true, false}, std::vector<bool>{false, true}, std::vector<bool>{}})
        {
            auto c = std::make_shared<Circuit>();
            c->AddOperation(std::make_shared<Circuits::Reset<double>>(Types::qubits_vector{0, 2}, 0, targets));
            c->AddOperation(Factory::CreateGate(Gate::kCXGateType, 0, 1));
            c->AddOperation(Factory::CreateGate(Gate::kCXGateType, 2, 3));
            SimulatorConfig config;
            config.auto_reduce = false;
            config.simulation_type = method;
            const auto reduced = EstimateCircuit(c, {"IZII", "IIIZ"}, config);
            config.enable_causal_cone_reduction = false;
            const auto full = EstimateCircuit(c, {"IZII", "IIIZ"}, config);
            for (size_t i = 0; i < 2; ++i)
            {
                const auto expected = i < targets.size() && targets[i] ? -1.0 : 1.0;
                Check(std::abs(reduced.values[i] - expected) < 1e-12 && std::abs(full.values[i] - expected) < 1e-12,
                      "Reset target was replaced with zero during cone pruning");
            }
        }
}

void CacheSafety()
{
    SimulatorConfig config;
    config.auto_reduce = false;
    auto clean = std::make_shared<Circuit>();
    clean->AddOperation(Factory::CreateGate(Gate::kZGateType, 0));
    clean->AddOperation(Factory::CreateGate(Gate::kCXGateType, 0, 1));
    clean->AddOperation(Factory::CreateGate(Gate::kCXGateType, 1, 0));
    EstimationCache cache;
    auto result = EstimateCircuit(clean, {"ZI"}, config, &cache);
    Check(result.plan.qubits_after == 1 && result.values[0] == 1, "Cached clean cone was not tightened");
    auto noisy = std::make_shared<Circuit>();
    noisy->AddOperation(clean->GetOperations()[0]);
    noisy->AddOperation(Factory::CreateGate(Gate::kXGateType, 0));
    noisy->AddOperation(clean->GetOperations()[1]);
    noisy->AddOperation(clean->GetOperations()[2]);
    result = EstimateCircuit(noisy, {"ZI"}, config, &cache);
    Check(cache.hits == 1 && result.plan.qubits_after == 2 && result.values[0] == 1, "Cache reused a clean-state fact invalidated by an injected Pauli");
    config.allow_reset_marginalization = true;
    EstimateCircuit(noisy, {"ZI"}, config, &cache);
    Check(cache.hits == 1, "Cache did not invalidate after reset policy change");
}

void SharedPropagation()
{
    std::vector<uint64_t> x{1, 0}, z{0, 0};
    auto cz = Factory::CreateGate(Gate::kCZGateType, 0, 1);
    TracePauliOperation(cz, x, z, true);
    Check(x == std::vector<uint64_t>({1, 0}) && z[1] == 1, "Diagonal gate introduced an X axis");
    auto swap = Factory::CreateGate(Gate::kSwapGateType, 0, 1);
    x = {1, 2};
    z = {4, 8};
    TracePauliOperation(swap, x, z, true);
    Check(x == std::vector<uint64_t>({2, 1}) && z == std::vector<uint64_t>({8, 4}), "Swap widened axes");
    auto delay = std::make_shared<Circuits::Delay<double>>(0, 2.0);
    Check(TracePauliOperation(delay, x, z, true) == 0 && x[0] == 2 && z[0] == 8, "Delay widened axes");
    auto reset = Factory::CreateReset({0});
    TracePauliOperation(reset, x, z, true);
    Check(x[0] == 0 && z[0] == 0 && x[1] == 1, "Ensemble reset did not stop backwards propagation");
    for (auto gate : {Gate::kCYGateType, Gate::kCHGateType, Gate::kCRxGateType, Gate::kCRyGateType, Gate::kCUGateType})
    {
        x = {0, 0};
        z = {1, 0};
        auto controlled = Factory::CreateGate(gate, 0, 1, 0, 0.3, 0.2, 0.1);
        Check(TracePauliOperation(controlled, x, z, true) == 0, "Controlled U failed to commute with control Z");
    }
}

void HistogramPairing()
{
    // Marginals and correlations within a component must survive combination.
    const size_t billion = 1000000000;
    Counts left{{{false, false, false}, billion / 2}, {{true, false, false}, billion / 2}};
    Counts right{{{false, false}, billion / 2}, {{true, true}, billion / 2}};
    const auto joined = CombineIndependentCounts(left, right, {1, 2}, 73);
    Check(joined == CombineIndependentCounts(left, right, {1, 2}, 73), "Histogram pairing is not reproducible");
    size_t total = 0, a = 0, b = 0;
    for (const auto &[bits, count] : joined)
    {
        Check(bits[1] == bits[2], "Pairing destroyed an entangled component's correlation");
        total += count;
        if (bits[0])
            a += count;
        if (bits[1])
            b += count;
    }
    Check(joined.size() == 4 && total == billion && a == billion / 2 && b == billion / 2, "Pairing changed a histogram marginal");
    // For a 2x2 table with margins (2,2), P(n00=0,1,2) = (1,4,1)/6.
    size_t frequencies[3]{};
    for (size_t seed = 0; seed < 12000; ++seed)
    {
        auto r = CombineIndependentCounts({{{false, false}, 2}, {{true, false}, 2}}, {{{false}, 2}, {{true}, 2}}, {1}, seed);
        ++frequencies[r[{false, false}]];
    }
    Check(std::abs(double(frequencies[0]) / 12000 - 1.0 / 6) < .015 && std::abs(double(frequencies[1]) / 12000 - 2.0 / 3) < .015 &&
              std::abs(double(frequencies[2]) / 12000 - 1.0 / 6) < .015,
          "Pairing does not follow the hypergeometric law");
    double sum = 0, squares = 0;
    for (size_t seed = 0; seed < 12000; ++seed)
    {
        auto r = CombineIndependentCounts({{{false, false}, 11}, {{true, false}, 26}}, {{{false}, 30}, {{true}, 7}}, {1}, seed);
        const double k = static_cast<double>(r[{false, true}]);
        sum += k;
        squares += k * k;
    }
    const double mean = sum / 12000, variance = squares / 12000 - mean * mean;
    Check(std::abs(mean - 11.0 * 7 / 37) < .04 && std::abs(variance - 11.0 * 7 / 37 * 30 / 37 * 26 / 36) < .07,
          "Asymmetric hypergeometric pairing has incorrect moments");
    bool rejected = false;
    try
    {
        CombineIndependentCounts(left, {{{false}, 1}}, {1}, 0);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    Check(rejected, "Invalid component shot count was accepted");
}

void MixedHistogramPairing()
{
    // Non-power-of-two support, an empty bucket, unequal weights, and alternating
    // small/large rows exercise both allocation paths and rebuilding the urn.
    const std::vector<size_t> bits{8, 5, 3, 7, 4};
    const std::vector<size_t> row_counts{2, 20, 2, 14, 2};
    Counts left, right;
    std::vector<size_t> capacities(17);
    for (size_t row = 0; row < row_counts.size(); ++row)
    {
        std::vector<bool> prefix(10);
        for (size_t b = 0; b < 3; ++b)
            prefix[b] = (row >> (2 - b)) & 1;
        prefix[6] = true;
        left[prefix] = row_counts[row];
    }
    for (size_t column = 0; column < capacities.size(); ++column)
    {
        std::vector<bool> outcome(bits.size());
        for (size_t b = 0; b < bits.size(); ++b)
            outcome[b] = (column >> b) & 1;
        capacities[column] = column ? 1 + column % 4 : 0;
        right[outcome] = capacities[column];
    }
    size_t frequencies[3]{}, repeats = 0;
    constexpr size_t trials = 12000;
    for (size_t seed = 0; seed < trials; ++seed)
    {
        const auto joined = CombineIndependentCounts(left, right, bits, seed);
        std::vector<size_t> rows(row_counts.size()), columns(capacities.size());
        size_t selected = 0;
        for (const auto &[outcome, count] : joined)
        {
            Check(count && outcome.size() == 10 && outcome[6] && !outcome[9], "Pairing changed an untouched bit or emitted an empty bucket");
            size_t row = 0, column = 0;
            for (size_t b = 0; b < 3; ++b)
                row = 2 * row + outcome[b];
            for (size_t b = 0; b < bits.size(); ++b)
                column |= size_t(outcome[bits[b]]) << b;
            Check(row < rows.size() && column < columns.size(), "Pairing created an unknown outcome");
            rows[row] += count;
            columns[column] += count;
            if (!row)
            {
                selected += column < 8 ? count : 0;
                repeats += count == 2;
            }
        }
        Check(rows == row_counts && columns == capacities, "Mixed sparse/dense allocation changed a marginal");
        ++frequencies[selected];
    }
    // The first row draws two of 40 shots. The first eight buckets contain 19
    // shots; repeated categories have probability sum(c*(c-1))/(40*39).
    Check(std::abs(double(frequencies[0]) / trials - 21.0 * 20 / (40 * 39)) < .015 &&
              std::abs(double(frequencies[1]) / trials - 2.0 * 19 * 21 / (40 * 39)) < .015 &&
              std::abs(double(frequencies[2]) / trials - 19.0 * 18 / (40 * 39)) < .015 && std::abs(double(repeats) / trials - 80.0 / (40 * 39)) < .008,
          "Sparse pairing is not weighted sampling without replacement");
    const auto expected = CombineIndependentCounts(left, right, bits, 73);
    Counts reordered_left, reordered_right;
    std::map<std::vector<bool>, size_t> sorted_left(left.begin(), left.end()), sorted_right(right.begin(), right.end());
    for (auto it = sorted_left.rbegin(); it != sorted_left.rend(); ++it)
        reordered_left.insert(*it);
    for (auto it = sorted_right.rbegin(); it != sorted_right.rend(); ++it)
        reordered_right.insert(*it);
    reordered_left.rehash(101);
    reordered_right.rehash(211);
    Check(expected == CombineIndependentCounts(reordered_left, reordered_right, bits, 73), "Pairing depends on input histogram iteration order");
    Check(CombineIndependentCounts({}, {}, {}, 1).empty(), "Empty histograms produced counts");
    Check(CombineIndependentCounts({{{false}, 0}}, {{{false}, 0}}, {0}, 1).empty(), "Zero-shot histograms produced counts");
    Check(CombineIndependentCounts({{{true, false}, 1}}, {{{true}, 1}}, {1}, 1) == Counts{{{true, true}, 1}}, "Single-shot pairing lost its outcome");
}

void SparseHistogramPairing()
{
    // This case previously scanned 2^33 row/column pairs. Its separate CTest
    // timeout guards against that scaling, with ample headroom for a sparse join.
    constexpr size_t row_size = size_t{1} << 17, column_size = size_t{1} << 16;
    Counts left, right;
    std::vector<size_t> bits;
    for (size_t b = 0; b < 16; ++b)
        bits.push_back(31 - 2 * b);
    for (size_t row = 0; row < row_size; ++row)
    {
        std::vector<bool> prefix(36);
        for (size_t b = 0; b < 17; ++b)
            prefix[2 * b] = (row >> b) & 1;
        prefix[34] = true;
        left[std::move(prefix)] = 1;
    }
    for (size_t column = 0; column < column_size; ++column)
    {
        std::vector<bool> outcome(16);
        for (size_t b = 0; b < 16; ++b)
            outcome[b] = (column >> b) & 1;
        right[std::move(outcome)] = 2;
    }
    const auto joined = CombineIndependentCounts(left, right, bits, 73);
    Check(joined.size() == row_size, "Sparse pairing lost distinct joint outcomes");
    std::vector<size_t> rows(row_size), columns(column_size);
    for (const auto &[outcome, count] : joined)
    {
        Check(count == 1 && outcome.size() == 36 && !outcome[33] && outcome[34] && !outcome[35], "Sparse pairing changed an untouched bit or row count");
        size_t row = 0, column = 0;
        for (size_t b = 0; b < 17; ++b)
            row |= size_t(outcome[2 * b]) << b;
        for (size_t b = 0; b < 16; ++b)
            column |= size_t(outcome[bits[b]]) << b;
        rows[row] += count;
        columns[column] += count;
    }
    Check(std::all_of(rows.begin(), rows.end(), [](auto n) { return n == 1; }) && std::all_of(columns.begin(), columns.end(), [](auto n) { return n == 2; }),
          "Sparse pairing changed a histogram marginal");
    Check(joined == CombineIndependentCounts(left, right, bits, 73), "Sparse pairing is not reproducible");
}

void SparseMeasurementCones()
{
    auto c = std::make_shared<Circuit>();
    std::vector<size_t> measured;
    for (size_t q = 0; q < 20000; ++q)
    {
        c->AddOperation(Factory::CreateGate(Gate::kHadamardGateType, q));
        measured.push_back(q);
    }
    c->AddOperation(Factory::CreateGate(Gate::kCXGateType, 0, 19999));
    const auto cones = TraceMeasurementCones(c, measured);
    Check(cones.size() == 19999 && cones.front().mapping.size() == 2 && cones.front().operations.size() == 3,
          "Sparse cone analysis merged unrelated measurements or lost operations");
}

void BoundedWork()
{
    auto c = std::make_shared<Circuit>();
    c->AddOperation(Factory::CreateGate(Gate::kRyGateType, 70, 0, 0, .4));
    Eigen::Matrix2cd id = Eigen::Matrix2cd::Identity();
    auto channel = std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{70}, Simulators::QuantumChannel({id}));
    c->AddOperation(channel);
    // A disjoint unsupported large gate must not prevent bounding this cone.
    c->AddOperation(Factory::CreateGate(Gate::kCCXGateType, 1, 2, 3));
    const auto observable = std::string(70, 'I') + "Z";
    BoundedPauliWorkspace workspace;
    auto value = EvaluateBoundedPauli(c, observable, 1e-6, &workspace);
    Check(value && std::abs(value->value - std::cos(.4)) <= value->error + 1e-12, "Cone-local compact Pauli evaluation changed a value");
    Check(workspace.transfers.size() == 2 && workspace.work < 100, "Bounded evaluator processed spectator gates");
    const auto entries = workspace.transfers.size();
    value = EvaluateBoundedPauli(c, observable, 1e-6, &workspace);
    Check(value && workspace.transfers.size() == entries, "Request-local transfer cache was not reused");
    workspace.work_limit = workspace.work;
    Check(!EvaluateBoundedPauli(c, observable, 1e-6, &workspace), "Bounded evaluator exceeded its work budget");
    auto wide = std::make_shared<Circuit>();
    wide->AddOperation(Factory::CreateGate(Gate::kRyGateType, 33, 0, 0, .4));
    for (size_t q = 0; q < 40; ++q)
        wide->AddOperation(std::make_shared<Circuits::QuantumChannelOperation<double>>(Types::qubits_vector{q}, Simulators::QuantumChannel({id})));
    auto multiword = EvaluateBoundedPauli(wide, std::string(40, 'Z'), 1e-6);
    Check(multiword && std::abs(multiword->value - std::cos(.4)) <= multiword->error + 1e-12, "Compact Pauli keys lost a word boundary");
    SimulatorConfig config;
    config.simulation_type = Simulators::SimulationType::kDensityMatrix;
    config.observable_error_tolerance = 1e-6;
    auto identity = PrepareEstimation(c, {std::string(71, 'I')}, config);
    Check(identity.groups.empty() && identity.resolved_values == std::vector<double>{1}, "Identity allocated a simulator group");
}

int main(int argc, char **argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--histogram-sparse")
        {
            SparseHistogramPairing();
            std::cout << "Sparse histogram pairing regressions passed\n";
            return 0;
        }
        ResetTargets();
        CacheSafety();
        SharedPropagation();
        HistogramPairing();
        MixedHistogramPairing();
        SparseMeasurementCones();
        BoundedWork();
        std::cout << "Lightcone update regressions passed\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
