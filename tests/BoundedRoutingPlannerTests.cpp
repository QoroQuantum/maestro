#include "Circuit/Factory.h"
#include "Circuit/Operations/QuantumChannelOperation.h"
#include "Simulators/TensorNetworks/BoundedRoutingPlanner.h"
#include "fixtures/BoundedRoutingChoices.h"
#include <iostream>
#include <numeric>
#include <type_traits>

namespace Simulators::Private
{
// No production counters or runtime debug mode: the friend exists solely to
// inspect immutable parsing, common-prefix limits and complete/pruned scores.
struct BoundedRoutingPlannerTestAccess
{
    using Planner = BoundedRoutingPlanner;

    struct Scores
    {
        std::vector<long long> positions;
        std::vector<double> full, pruned;
        size_t horizon;
    };

    static Scores Inspect(Planner &planner, const Planner::Map &map, const std::vector<double> &ranks, size_t op, size_t fixedHorizon = 0)
    {
        if (!planner.Seed(map, ranks.data(), ranks.size()))
            throw std::runtime_error("Fixture input rejected");
        const auto entry = planner.operationToGate_.at(op);
        const auto candidates = planner.CandidatesFor(entry);
        Scores result;
        result.horizon = planner.policy_ == Planner::Policy::Work4 ? 4 : planner.CommonHorizon(entry, candidates.count);
        if (fixedHorizon)
            result.horizon = fixedHorizon;
        double incumbent = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < candidates.count; ++i)
        {
            const auto position = candidates.positions[i];
            result.positions.push_back(position);
            result.full.push_back(planner.Evaluate(entry, position, result.horizon, std::numeric_limits<double>::infinity()));
            result.pruned.push_back(planner.Evaluate(entry, position, result.horizon, incumbent));
            incumbent = std::min(incumbent, result.pruned.back());
        }
        return result;
    }

    static size_t Horizon(Planner &planner, const Planner::Map &map, size_t count)
    {
        const std::vector<double> ranks(map.size() - 1, 1.);
        if (!planner.Seed(map, ranks.data(), ranks.size()))
            throw std::runtime_error("Horizon fixture rejected");
        return planner.CommonHorizon(0, count);
    }

    static const void *Scratch(const Planner &planner)
    {
        return planner.scratch_.map.data();
    }

    static const void *Initial(const Planner &planner)
    {
        return planner.initial_.map.data();
    }

    static bool Swap(const Planner::SharedOperations &operations, size_t gate)
    {
        return operations->gates.at(gate).swap;
    }

    static double Rank(const Planner::SharedOperations &operations, size_t gate)
    {
        return operations->gates.at(gate).rank;
    }

    static const auto &Singles(const Planner::SharedOperations &operations, size_t gate)
    {
        return operations->gates.at(gate).singlesBefore;
    }

    static size_t GateCount(const Planner::SharedOperations &operations)
    {
        return operations->gates.size();
    }
};
} // namespace Simulators::Private

namespace
{
using Planner = Simulators::Private::BoundedRoutingPlanner;
using Access = Simulators::Private::BoundedRoutingPlannerTestAccess;
using Op = Planner::Operation;
using Map = Planner::Map;
using Kind = Circuits::QuantumGateType;
using Factory = Circuits::CircuitFactory<>;

void Check(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function> void Invalid(Function function)
{
    bool rejected = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    Check(rejected, "Invalid planner construction was accepted");
}

Map Identity(size_t n)
{
    Map result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

Op Gate(size_t a, size_t b)
{
    return Factory::CreateGate(Kind::kCXGateType, a, b);
}

struct Metadata : Circuits::NoOperation<>, Simulators::RoutingGateInfo
{
    Types::qubits_vector targets;

    Metadata(size_t a, size_t b, double rank = 2., bool swap = false) : targets{a, b}
    {
        schmidtRank = rank;
        isSwap = swap;
    }

    Circuits::OperationType GetType() const override
    {
        return Circuits::OperationType::kGate;
    }

    Types::qubits_vector AffectedQubits() const override
    {
        return targets;
    }
};

struct MatrixGate : Circuits::CXGate<>
{
    Eigen::MatrixXcd matrix;

    MatrixGate(size_t a, size_t b, Eigen::MatrixXcd value) : Circuits::CXGate<>(a, b), matrix(std::move(value))
    {
    }

    Kind GetGateType() const override
    {
        return Kind::kNone;
    }

    Eigen::MatrixXcd GetMatrix() const override
    {
        return matrix;
    }
};

void ExpectedSelectionsAndPruning()
{
    size_t pruned = 0, horizonDifferences = 0;
    for (const auto &fixture : RoutingChoices::cases)
    {
        std::vector<Op> operations;
        for (const auto &gate : fixture.gates)
            operations.push_back(std::make_shared<Metadata>(gate.a, gate.b, gate.rank, gate.swap));
        const auto shared = Planner::PrepareOperations(fixture.n, operations);
        for (size_t mode = 0; mode < 4; ++mode)
        {
            const bool mpo = mode >= 2;
            const auto policy = mode % 2 ? Planner::Policy::Work24 : Planner::Policy::Work4;
            Planner planner(fixture.n, mpo, fixture.cap, shared, policy);
            const auto map = fixture.map;
            const auto ranks = fixture.ranks;
            Check(planner.Select(map, ranks, 0) == fixture.expected[mode], "Independent golden meeting-position choice changed");
            std::vector<long long> integers(ranks.begin(), ranks.end());
            Check(planner.Select(map, integers.data(), integers.size(), 0) == fixture.expected[mode], "Native integer/pointer selection differs");
            const auto scores = Access::Inspect(planner, map, ranks, 0);
            const size_t best = std::min_element(scores.full.begin(), scores.full.end()) - scores.full.begin();
            Check(scores.positions[best] == fixture.expected[mode], "Complete forecast differs from independent expected selection");
            double incumbent = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < scores.full.size(); ++i)
            {
                Check(scores.pruned[i] <= scores.full[i], "Abandoned score exceeds complete nonnegative work");
                if (scores.pruned[i] < scores.full[i])
                {
                    ++pruned;
                    Check(scores.pruned[i] >= incumbent && scores.full[i] >= incumbent, "Pruning abandoned a possible strict winner");
                }
                incumbent = std::min(incumbent, scores.pruned[i]);
            }
            Check(map == fixture.map && ranks == fixture.ranks, "Select mutated caller map or ranks");
            auto permuted = map;
            std::reverse(permuted.begin(), permuted.end());
            planner.Select(permuted, std::vector<double>(fixture.n - 1, 1.), 0);
            Check(planner.Select(map, ranks, 0) == fixture.expected[mode], "Forecast state leaked across live-rank queries");
        }
        horizonDifferences += fixture.expected[0] != fixture.expected[1];
        horizonDifferences += fixture.expected[2] != fixture.expected[3];
    }
    Check(pruned > 0 && horizonDifferences > 0, "Golden cases did not exercise pruning and distinct horizons");
}

void SingleQubitWorkGoldens()
{
    size_t differentChoices = 0, differentWork = 0;
    for (const auto &fixture : RoutingChoices::singleCases)
    {
        std::vector<Op> singleGates;
        for (size_t q = 0; q < fixture.n; ++q)
            singleGates.push_back(Factory::CreateGate(Kind::kHadamardGateType, q));
        std::vector<Op> operations, withoutSingles, withoutPrior;
        size_t firstOperation = 0;
        for (size_t step = 0; step < fixture.gates.size(); ++step)
        {
            for (const auto &[qubit, repeats] : fixture.singles.at(step))
                for (size_t count = 0; count < repeats; ++count)
                {
                    operations.push_back(singleGates.at(qubit));
                    if (step)
                        withoutPrior.push_back(singleGates.at(qubit));
                }
            if (!step)
                firstOperation = operations.size();
            const auto &gate = fixture.gates[step];
            const auto operation = std::make_shared<Metadata>(gate.a, gate.b, gate.rank, gate.swap);
            operations.push_back(operation);
            withoutSingles.push_back(operation);
            withoutPrior.push_back(operation);
        }
        const auto prepared = Planner::PrepareOperations(fixture.n, operations);
        const auto ignored = Planner::PrepareOperations(fixture.n, withoutSingles);
        const auto noPrior = Planner::PrepareOperations(fixture.n, withoutPrior);
        Check(Access::GateCount(prepared) == fixture.gates.size(), "Singles created a spurious two-qubit forecast entry");
        for (size_t step = 0; step < fixture.gates.size(); ++step)
        {
            // Independent dense counter establishes the required ascending
            // emission order even though source singles alternate and repeat.
            std::vector<size_t> counts(fixture.n);
            for (const auto &[q, repeat] : fixture.singles[step])
                counts[q] += repeat;
            std::vector<std::pair<long long, size_t>> expected;
            for (size_t q = 0; q < counts.size(); ++q)
                if (counts[q])
                    expected.emplace_back(static_cast<long long>(q), counts[q]);
            Check(Access::Singles(prepared, step) == expected, "Sparse singles lost count, ascending order or boundary reset");
        }
        for (size_t mode = 0; mode < 4; ++mode)
        {
            const auto policy = mode % 2 ? Planner::Policy::Work24 : Planner::Policy::Work4;
            Planner planner(fixture.n, mode >= 2, fixture.cap, prepared, policy);
            Planner omitted(fixture.n, mode >= 2, fixture.cap, ignored, policy);
            Planner priorRemoved(fixture.n, mode >= 2, fixture.cap, noPrior, policy);
            const auto &expected = fixture.expected[mode];
            Check(planner.Select(fixture.map, fixture.ranks, firstOperation) == expected.selected,
                  "Single-qubit work changed the independent expected meeting choice");
            const auto scores = Access::Inspect(planner, fixture.map, fixture.ranks, firstOperation);
            Check(scores.horizon == expected.horizon && scores.positions == expected.candidates,
                  "Single-qubit golden candidate order or common horizon changed");
            // Fix exactly the same two-qubit prefix for the counterfactual, so
            // a changed winner demonstrates work scoring, not budget changes.
            const auto omittedScores = Access::Inspect(omitted, fixture.map, fixture.ranks, 0, expected.horizon);
            const auto priorScores = Access::Inspect(priorRemoved, fixture.map, fixture.ranks, 0);
            Check(omittedScores.positions == expected.candidates && priorScores.positions == expected.candidates && priorScores.horizon == scores.horizon,
                  "Removing singles changed candidates or current-gate prefix semantics");
            for (size_t i = 0; i < scores.full.size(); ++i)
            {
                const auto close = [](double actual, double reference) {
                    return std::isfinite(actual) && std::abs(actual - reference) <= 1e-10 + 2e-12 * std::abs(reference);
                };
                Check(close(scores.full[i], expected.costs.at(i)) && close(omittedScores.full[i], expected.withoutCosts.at(i)),
                      "Single-qubit forecast work differs from independent scalar scores");
                Check(scores.full[i] == priorScores.full[i], "Singles before the current gate were charged twice");
                if (scores.full[i] > omittedScores.full[i] + 1e-6)
                    ++differentWork;
            }
            const size_t best = std::min_element(scores.full.begin(), scores.full.end()) - scores.full.begin();
            const size_t ignoredBest = std::min_element(omittedScores.full.begin(), omittedScores.full.end()) - omittedScores.full.begin();
            Check(scores.positions[best] == expected.selected && omittedScores.positions[ignoredBest] == expected.withoutSingles,
                  "Full single-qubit work argmin differs from immutable golden");
            differentChoices += expected.selected != expected.withoutSingles;
        }
    }
    Check(differentChoices >= 2 && differentWork > 0, "Single-qubit fixtures do not detect omitted work scoring");

    // A non-gate barrier must flush pending sparse singles and reset counts;
    // subsequent high-index/repeated sites belong only to the following gate.
    std::vector<Op> barrierOperations;
    for (size_t q : {size_t(99), size_t(2), size_t(99), size_t(70), size_t(2)})
        barrierOperations.push_back(Factory::CreateGate(Kind::kHadamardGateType, q));
    barrierOperations.push_back(Factory::CreateNoOp());
    for (size_t q : {size_t(70), size_t(1), size_t(70)})
        barrierOperations.push_back(Factory::CreateGate(Kind::kHadamardGateType, q));
    barrierOperations.push_back(Gate(0, 99));
    const auto barrier = Planner::PrepareOperations(100, barrierOperations);
    Check(Access::GateCount(barrier) == 2 && Access::Singles(barrier, 0) == std::vector<std::pair<long long, size_t>>{{2, 2}, {70, 1}, {99, 2}} &&
              Access::Singles(barrier, 1) == std::vector<std::pair<long long, size_t>>{{1, 1}, {70, 2}},
          "Barrier did not flush/reset sparse singles in ascending order");
}

void SparseDenseSingleAggregation()
{
    // Even widths exercise the exact half boundary; odd widths also check the
    // floor used by the sparse/dense switch. No scoring fixture is needed: the
    // independently accumulated records must agree at every gate and barrier.
    for (size_t n : {size_t(5), size_t(12), size_t(13), size_t(50), size_t(51), size_t(100), size_t(101)})
    {
        std::vector<Op> singleGates, operations;
        for (size_t q = 0; q < n; ++q)
            singleGates.push_back(Factory::CreateGate(Kind::kHadamardGateType, q));
        std::vector<std::vector<std::pair<long long, size_t>>> expected;
        const auto append = [&](size_t touched, bool barrier, size_t offset) {
            std::vector<size_t> counts(n);
            // Two reverse-order passes make repeated touches nonconsecutive.
            // Rotation changes which sites survive across successive chunks.
            for (size_t pass = 0; pass < 2; ++pass)
                for (size_t j = 0; j < touched; ++j)
                {
                    const size_t q = (n - 1 - j + offset) % n;
                    const size_t repeats = pass ? 1 + (q + expected.size()) % 3 : 1;
                    operations.insert(operations.end(), repeats, singleGates[q]);
                    counts[q] += repeats;
                }
            std::vector<std::pair<long long, size_t>> records;
            for (size_t q = 0; q < n; ++q)
                if (counts[q])
                    records.emplace_back(static_cast<long long>(q), counts[q]);
            expected.push_back(std::move(records));
            operations.push_back(barrier ? Factory::CreateNoOp() : Gate(0, n - 1));
        };
        for (size_t touched : {size_t(0), size_t(1), n / 2, n / 2 + 1, n})
        {
            append(touched, false, 0);
            append(0, true, 0); // A gate must reset both counts and touched sites.
            append(touched, true, n / 3);
            append(0, false, 0); // A barrier must perform the same reset.
        }
        const auto prepared = Planner::PrepareOperations(n, operations);
        Check(Access::GateCount(prepared) == expected.size(), "Sparse/dense aggregation changed boundary entries");
        for (size_t gate = 0; gate < expected.size(); ++gate)
            Check(Access::Singles(prepared, gate) == expected[gate], "Sparse/dense singles differ at half, majority, full or reset boundaries");
    }
}

void AnalyticTiesAndHorizons()
{
    for (bool mpo : {false, true})
        for (const auto policy : {Planner::Policy::Work4, Planner::Policy::Work24})
        {
            Planner planner(6, mpo, 8, std::vector<Op>{Gate(0, 5)}, policy);
            const auto map = Identity(6);
            const std::vector<double> ranks(5, 1.);
            const auto scores = Access::Inspect(planner, map, ranks, 0);
            // Four cheap product-site SWAPs, then one two-site gate:
            // MPS: 4*8 + 4*(1+2+1.4); MPO: 4*16 +16*(1+4+2.8).
            const double expected = mpo ? 188.8 : 49.6;
            for (double score : scores.full)
                Check(std::abs(score - expected) < 1e-10, "Analytic product-state work changed");
            Check(planner.Select(map, ranks, 0) == 0, "Exact native-first score tie changed");
        }
    for (size_t n : {size_t(50), size_t(100), size_t(130), size_t(512)})
        for (bool singles : {false, true})
        {
            std::vector<Op> operations;
            for (size_t step = 0; step < 30; ++step)
            {
                if (singles && step)
                    for (size_t q = 0; q < n; ++q)
                        operations.push_back(Factory::CreateGate(Kind::kHadamardGateType, q));
                operations.push_back(Gate(0, n - 1));
            }
            Planner planner(n, false, 16, operations, Planner::Policy::Work24);
            for (size_t candidates = 1; candidates <= 5; ++candidates)
            {
                const size_t budget = 4096 / candidates;
                const size_t first = n - 1, later = n - 1 + (singles ? n : 0);
                const size_t expected = first > budget ? 0 : std::min<size_t>(24, 1 + (budget - first) / later);
                Check(Access::Horizon(planner, Identity(n), candidates) == expected, "Common horizon differs from independent update-bound formula");
                if (expected)
                    Check(candidates * (first + (expected - 1) * later) <= 4096, "Aggregate forecast budget exceeded");
            }
            Check(planner.Select(Identity(n), std::vector<double>(n - 1, 1.), 0) >= 0, "Large-span heap meeting path rejected valid input");
        }
    Planner huge(5000, false, 8, std::vector<Op>{Gate(0, 4999)}, Planner::Policy::Work24);
    Check(Access::Horizon(huge, Identity(5000), 1) == 0 && huge.Select(Identity(5000), std::vector<double>(4999, 1.), 0) == 0,
          "Zero-horizon query did not retain the native proposal");
}

void InputsSwapsAndBarriers()
{
    const auto map = Identity(6);
    const std::vector<double> ranks{2, 4, 8, 4, 2};
    auto exact = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(Factory::CreateGate(Kind::kSwapGateType, 1, 4))->GetMatrix();
    auto nearby = exact;
    nearby(0, 0) = std::polar(1., .125);
    const auto channel = std::make_shared<Circuits::QuantumChannelOperation<>>(Types::qubits_vector{0}, Simulators::QuantumChannel::AmplitudeDamping(.1));
    std::vector<Op> operations{Gate(0, 5),
                               Factory::CreateGate(Kind::kHadamardGateType, 2),
                               Factory::CreateGate(Kind::kSwapGateType, 1, 4),
                               std::make_shared<MatrixGate>(1, 4, exact),
                               std::make_shared<MatrixGate>(1, 4, nearby),
                               channel,
                               Gate(1, 5),
                               Factory::CreateGate(Kind::kCCXGateType, 0, 1, 2),
                               Factory::CreateNoOp(),
                               nullptr,
                               std::make_shared<Metadata>(0, 0),
                               std::make_shared<Metadata>(0, 6)};
    const auto shared = Planner::PrepareOperations(6, operations);
    Check(Access::Swap(shared, 1) && Access::Swap(shared, 2) && !Access::Swap(shared, 3), "Named/exact/near SWAP parsing changed");
    for (const auto policy : {Planner::Policy::Work4, Planner::Policy::Work24})
    {
        Planner planner(6, false, 16, shared, policy);
        for (size_t op : {size_t(1), size_t(2), size_t(3), size_t(5), size_t(7), size_t(8), size_t(9), size_t(10), size_t(11), size_t(12)})
            Check(planner.Select(map, ranks, op) == -1, "Unsupported current operation lost native fallback");
        Check(planner.Select(map, ranks, 4) >= 0, "Near-SWAP unitary was treated as logical relabel");
        auto invalid = map;
        invalid[0] = invalid[1];
        Check(planner.Select(invalid, ranks, 0) == -1, "Duplicate physical position accepted");
        invalid = map;
        invalid[0] = -1;
        Check(planner.Select(invalid, ranks, 0) == -1, "Negative physical position accepted");
        invalid[0] = 6;
        Check(planner.Select(invalid, ranks, 0) == -1, "Out-of-range position accepted");
        Check(planner.Select(Map(5, 0), ranks, 0) == -1 && planner.Select(map, ranks.data(), 4, 0) == -1 && planner.Select<double>(map, nullptr, 5, 0) == -1,
              "Invalid input dimensions accepted");
        for (double bad : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            auto broken = ranks;
            broken[2] = bad;
            Check(planner.Select(map, broken, 0) == -1, "Invalid actual bond rank accepted");
        }
        std::vector<Op> before{Gate(0, 5), channel}, after = before;
        after.push_back(Gate(1, 4));
        after.push_back(Gate(2, 5));
        Planner prefix(6, false, 16, before, policy), suffix(6, false, 16, after, policy);
        Check(prefix.Select(map, ranks, 0) == suffix.Select(map, ranks, 0), "Forecast crossed channel boundary");
        Planner adjacent(6, false, 16, std::vector<Op>{Gate(2, 3)}, policy);
        Check(adjacent.Select(map, ranks, 0) == 2, "Adjacent pair did not use its existing bond");
    }
    Invalid([&] { Planner p(1, false, 16, operations, Planner::Policy::Work4); });
    Invalid([&] { Planner p(6, false, -1, operations, Planner::Policy::Work4); });
    Invalid([&] { Planner p(7, false, 16, shared, Planner::Policy::Work4); });
    Invalid([&] { Planner p(6, false, 16, Planner::SharedOperations{}, Planner::Policy::Work4); });
    Invalid([&] { Planner p(0, false, 16, std::make_shared<Planner::PreparedOperations>(), Planner::Policy::Work4); });
    Invalid([&] { Planner p(6, false, -1, shared, Planner::Policy::Work4); });
}

void ImmutableTablesAndIndependentScratch()
{
    auto metadata = std::make_shared<Metadata>(0, 5);
    std::vector<Op> operations{metadata, Factory::CreateGate(Kind::kHadamardGateType, 4), Factory::CreateGate(Kind::kHadamardGateType, 1),
                               Factory::CreateGate(Kind::kHadamardGateType, 4), Gate(1, 5)};
    const auto shared = Planner::PrepareOperations(6, operations);
    static_assert(std::is_const_v<Planner::SharedOperations::element_type>, "Shared operation table must remain immutable");
    const std::vector<std::pair<long long, size_t>> singles{{1, 1}, {4, 2}};
    Check(Access::Singles(shared, 1) == singles, "Single-qubit aggregation/order changed");
    Planner source(6, false, 16, shared, Planner::Policy::Work24);
    Planner clone(6, false, 16, source.GetPreparedOperations(), Planner::Policy::Work24);
    Check(source.GetPreparedOperations() == clone.GetPreparedOperations(), "Planner clone reparsed immutable operations");
    Check(Access::Scratch(source) != Access::Scratch(clone) && Access::Initial(source) != Access::Initial(clone), "Clones share mutable routing scratch");
    const auto map = Identity(6);
    const std::vector<double> ranks{2, 4, 8, 4, 2};
    const auto expected = source.Select(map, ranks, 0);
    auto other = map;
    std::reverse(other.begin(), other.end());
    clone.Select(other, std::vector<double>(5, 1.), 0);
    Check(source.Select(map, ranks, 0) == expected && clone.Select(map, ranks, 0) == expected, "Clone selection leaked mutable scratch");
    metadata->schmidtRank = 4.;
    const auto changed = Planner::PrepareOperations(6, operations);
    Check(Access::Rank(shared, 0) == 2. && Access::Rank(changed, 0) == 4., "Prepared data retained mutable source aliases");
    operations.clear();
    metadata.reset();
    Check(source.Select(map, ranks, 0) == expected, "Prepared operations depended on source-vector lifetime");
}
} // namespace

int main()
{
    try
    {
        ExpectedSelectionsAndPruning();
        SingleQubitWorkGoldens();
        SparseDenseSingleAggregation();
        AnalyticTiesAndHorizons();
        InputsSwapsAndBarriers();
        ImmutableTablesAndIndependentScratch();
        std::cout << "Bounded routing expected choices, pruning, horizons, inputs, SWAPs, barriers and immutable sharing passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
