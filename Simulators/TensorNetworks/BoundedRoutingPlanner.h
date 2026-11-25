/**
 * Bounded online routing based on current physical bond dimensions.
 *
 * Work4 uses a four-interaction forecast. Work24 uses a common-prefix
 * forecast budget across its candidate meeting positions. The caller selects
 * a policy and supplies the current operation window.
 * This planner neither executes gates nor changes the backend's state.
 */
#pragma once

#include "../../Circuit/Operations/QuantumGates.h"
#include "RoutingGateInfo.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Simulators
{
namespace Private
{

/**
 * A bounded forecast based on the current physical ranks and logical map.
 *
 * Construct again when the operation sequence, qubit count, bond cap or policy
 * changes. Compatible instances may share an immutable prepared operation table.
 * Every Select call starts from the supplied live state. No forecast ranks
 * persist across calls. Instances have scratch storage and are not safe for
 * concurrent calls.
 *
 * Unsupported operations (including channels and decompositions)
 * terminate a forecast. Select returns -1 for such an operation; the caller
 * must retain its native routing fallback and decomposition guard. Logical
 * SWAPs are assumed to relabel qubits, as in the CPU QCSim wrappers.
 */
class BoundedRoutingPlanner
{
  public:
    using Index = long long;
    using Map = std::vector<Index>;
    using Operation = std::shared_ptr<Circuits::IOperation<>>;
    enum class Policy
    {
        Work4,
        Work24
    };

  private:
    struct Disentangle
    {
        Index qubit = 0;
        double probability = 1.;
        size_t singlesEnd = 0; // Number of accumulated single-qubit entries before this projection.
    };

    struct Gate
    {
        Index a = 0, b = 0;
        bool valid = false, swap = false;
        double rank = 4.;
        std::vector<std::pair<Index, size_t>> singlesBefore;
        std::vector<Disentangle> disentanglesBefore;
    };

  public:
    // Only operation-derived values are shared. Each planner owns its physical
    // cut limits and mutable forecast workspace, so clones can route independently.
    struct PreparedOperations
    {
      private:
        friend class BoundedRoutingPlanner;
        friend struct BoundedRoutingPlannerTestAccess;
        size_t qubits = 0;
        std::vector<Gate> gates;
        std::vector<Index> operationToGate;
    };

    using SharedOperations = std::shared_ptr<const PreparedOperations>;

    // Applies to Work24 only. Work4 retains its original four-entry horizon.
    static constexpr size_t kForecastOperationBudget = 4096;

    BoundedRoutingPlanner(size_t n, bool operatorChain, Index cap, const std::vector<Operation> &operations, Policy policy)
        : BoundedRoutingPlanner(n, operatorChain, cap, PrepareOperations(n, operations), policy)
    {
    }

    BoundedRoutingPlanner(size_t n, bool operatorChain, Index cap, SharedOperations operations, Policy policy)
        : n_(n), operatorChain_(operatorChain), policy_(policy), operations_(ValidateOperations(n, std::move(operations))), gates_(operations_->gates),
          operationToGate_(operations_->operationToGate)
    {
        if (cap < 0)
            throw std::invalid_argument("Invalid bounded routing configuration");
        Allocate(initial_);
        Allocate(scratch_);
        maximum_.resize(n - 1);
        const double d = operatorChain ? 4. : 2.;
        const double limit = static_cast<double>(std::numeric_limits<Index>::max());
        const double physical = cap > 0 ? static_cast<double>(cap) : limit;
        for (size_t b = 0; b + 1 < n; ++b)
            maximum_[b] = CutMaximum(std::min(b + 1, n - b - 1), d, physical);
    }

    const SharedOperations &GetPreparedOperations() const noexcept
    {
        return operations_;
    }

    static SharedOperations PrepareOperations(size_t n, const std::vector<Operation> &operations)
    {
        if (n < 2 || n > static_cast<size_t>(std::numeric_limits<Index>::max()))
            throw std::invalid_argument("Invalid bounded routing qubit count");
        auto prepared = std::make_shared<PreparedOperations>();
        prepared->qubits = n;
        prepared->operationToGate.assign(operations.size(), -1);
        std::vector<size_t> singles(n, 0);
        std::vector<Index> touchedSingles;
        std::vector<std::pair<Index, size_t>> pendingSingles;
        auto flushSingles = [&] {
            const auto emitSingle = [&](Index q) {
                pendingSingles.emplace_back(q, singles[q]);
                singles[q] = 0;
            };
            // Both paths preserve ascending accumulation order. For a dense
            // prefix, scanning visits fewer than two counters per emitted entry;
            // sparse prefixes avoid a full-qubit scan.
            if (touchedSingles.size() > n / 2)
            {
                for (size_t q = 0; q < n; ++q)
                    if (singles[q])
                        emitSingle(static_cast<Index>(q));
            }
            else
            {
                std::sort(touchedSingles.begin(), touchedSingles.end());
                for (const auto q : touchedSingles)
                    emitSingle(q);
            }
            touchedSingles.clear();
        };
        std::vector<Disentangle> pendingDisentangles;
        for (size_t i = 0; i < operations.size(); ++i)
        {
            const auto &operation = operations[i];
            Gate item;
            if (operation)
            {
                const auto opType = operation->GetType();
                if (opType == Circuits::OperationType::kMeasurement || opType == Circuits::OperationType::kConditionalMeasurement ||
                    opType == Circuits::OperationType::kReset)
                {
                    const auto qubits = operation->AffectedQubits();
                    const bool validQubits = std::all_of(qubits.begin(), qubits.end(), [n](auto q) { return q < n; });
                    if (validQubits)
                    {
                        flushSingles();
                        const double prob = (opType == Circuits::OperationType::kConditionalMeasurement) ? 0.5 : 1.;
                        for (const auto q : qubits)
                            pendingDisentangles.push_back({static_cast<Index>(q), prob, pendingSingles.size()});
                        continue;
                    }
                }
                else if (opType == Circuits::OperationType::kGate)
                {
                    const auto qubits = operation->AffectedQubits();
                    const bool validQubits = std::all_of(qubits.begin(), qubits.end(), [n](auto q) { return q < n; });
                    if (validQubits && qubits.size() == 1)
                    {
                        // A majority-length list is a sentinel: the dense path
                        // scans and resets every counter, so later indices are unused.
                        if (!singles[qubits[0]] && touchedSingles.size() <= n / 2)
                            touchedSingles.push_back(static_cast<Index>(qubits[0]));
                        ++singles[qubits[0]];
                        continue;
                    }
                    if (validQubits && qubits.size() == 2 && qubits[0] != qubits[1])
                    {
                        item.a = qubits[0];
                        item.b = qubits[1];
                        item.valid = true;
                        const auto info = std::dynamic_pointer_cast<RoutingGateInfo>(operation);
                        const auto gate = std::dynamic_pointer_cast<Circuits::IQuantumGate<>>(operation);
                        item.swap = (info && info->isSwap) ||
                                    (gate && (gate->GetGateType() == Circuits::QuantumGateType::kSwapGateType || ExactSwapMatrix(gate->GetMatrix())));
                        item.rank = info ? info->schmidtRank : gate ? GateSchmidtRank(*gate) : 4.;
                        prepared->operationToGate[i] = static_cast<Index>(prepared->gates.size());
                    }
                }
            }
            flushSingles();
            item.disentanglesBefore.swap(pendingDisentangles);
            item.singlesBefore.swap(pendingSingles);
            prepared->gates.push_back(std::move(item));
        }
        return prepared;
    }

    /** Return a physical meeting bond, or -1 to use the native fallback. */
    template <class Bond>
    Index Select(const Map &map, const Bond *bonds, size_t bondCount, size_t operationIndex, const double *siteRanks = nullptr, size_t siteRankCount = 0)
    {
        const Index entry = operationIndex < operationToGate_.size() ? operationToGate_[operationIndex] : -1;
        if (entry < 0 || !gates_[entry].valid || gates_[entry].swap || !Seed(map, bonds, bondCount, siteRanks, siteRankCount))
            return -1;
        const auto &gate = gates_[entry];
        const Index low = std::min(map[gate.a], map[gate.b]);
        const Index high = std::max(map[gate.a], map[gate.b]);
        if (high - low <= 1)
            return low;

        const auto candidates = CandidatesFor(entry);
        const size_t horizon = policy_ == Policy::Work4 ? 4 : CommonHorizon(entry, candidates.count);
        Index chosen = candidates.positions[0];
        double best = std::numeric_limits<double>::infinity();
        if (horizon)
            for (size_t i = 0; i < candidates.count; ++i)
            {
                const double score = Evaluate(entry, candidates.positions[i], horizon, best);
                if (score < best)
                {
                    best = score;
                    chosen = candidates.positions[i];
                }
            }
        return chosen;
    }

    template <class Bond> Index Select(const Map &map, const std::vector<Bond> &bonds, size_t operationIndex)
    {
        return Select(map, bonds.data(), bonds.size(), operationIndex, nullptr, 0);
    }

    template <class Bond> Index Select(const Map &map, const std::vector<Bond> &bonds, size_t operationIndex, const std::vector<double> &siteRanks)
    {
        return Select(map, bonds.data(), bonds.size(), operationIndex, siteRanks.data(), siteRanks.size());
    }

  private:
    struct State
    {
        Map map, inverse;
        std::vector<double> bonds;
        std::vector<double> siteRanks;
        bool hasSiteRanks = false;
    };

    struct CandidateSet
    {
        std::array<Index, 5> positions{};
        size_t count = 0;
    };

    size_t n_;
    bool operatorChain_;
    Policy policy_;
    SharedOperations operations_;
    const std::vector<Gate> &gates_;
    const std::vector<Index> &operationToGate_;
    std::vector<double> maximum_;
    State initial_, scratch_;

    friend struct BoundedRoutingPlannerTestAccess;

    static SharedOperations ValidateOperations(size_t n, SharedOperations operations)
    {
        if (n < 2 || n > static_cast<size_t>(std::numeric_limits<Index>::max()) || !operations || operations->qubits != n)
            throw std::invalid_argument("Incompatible bounded routing operation table");
        return operations;
    }

    static double Cube(double x)
    {
        return x * x * x;
    }

    static double CutMaximum(size_t sites, double d, double cap)
    {
        double result = 1.;
        for (size_t k = 0; k < sites && result < cap; ++k)
            result = std::min(cap, result * d);
        return result;
    }

    static double GateSchmidtRank(const Circuits::IQuantumGate<> &gate)
    {
        if (gate.GetNumQubits() != 2)
            return 4.;
        const auto params = gate.GetParams();
        return RoutingSchmidtRank(gate.GetGateType(), params.empty() ? 0. : params[0], [&gate] { return gate.GetMatrix(); });
    }

    static bool ExactSwapMatrix(const Eigen::MatrixXcd &matrix)
    {
        if (matrix.rows() != 4 || matrix.cols() != 4)
            return false;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                if (matrix(row, col) != std::complex<double>(row == (((col & 1) << 1) | (col >> 1)) ? 1. : 0., 0.))
                    return false;
        return true;
    }

    void Allocate(State &state)
    {
        state.map.resize(n_);
        state.inverse.resize(n_);
        state.bonds.resize(n_ - 1);
        state.siteRanks.resize(n_, operatorChain_ ? 4. : 2.);
    }

    template <class Bond> bool Seed(const Map &map, const Bond *bonds, size_t bondCount, const double *siteRanks = nullptr, size_t siteRankCount = 0)
    {
        if (map.size() != n_ || bondCount != n_ - 1 || bonds == nullptr)
            return false;
        std::fill(initial_.inverse.begin(), initial_.inverse.end(), -1);
        for (size_t q = 0; q < n_; ++q)
        {
            if (map[q] < 0 || static_cast<size_t>(map[q]) >= n_ || initial_.inverse[map[q]] != -1)
                return false;
            initial_.map[q] = map[q];
            initial_.inverse[map[q]] = static_cast<Index>(q);
        }
        for (size_t b = 0; b < bondCount; ++b)
        {
            const double rank = static_cast<double>(bonds[b]);
            if (!std::isfinite(rank) || rank < 1.)
                return false;
            initial_.bonds[b] = rank;
        }
        const double d = operatorChain_ ? 4. : 2.;
        if (siteRanks != nullptr && siteRankCount == n_)
        {
            initial_.hasSiteRanks = true;
            for (size_t q = 0; q < n_; ++q)
            {
                const double given = std::isfinite(siteRanks[q]) ? siteRanks[q] : d;
                initial_.siteRanks[q] = std::clamp(given, 1., d);
            }
        }
        else
        {
            initial_.hasSiteRanks = false;
            std::fill(initial_.siteRanks.begin(), initial_.siteRanks.end(), d);
        }
        return true;
    }

    // The installed native local routing policy. This also chooses every
    // future meeting position in a forecast; only the first meeting varies.
    static Index LocalMeeting(const State &state, Index low, Index high, bool operatorChain)
    {
        const auto &dims = state.bonds;
        const Index span = high - low;
        if (span <= 1)
            return low;
        std::array<double, 130> bonds{};
        std::array<double, 128> prefixes{}, suffixes{};
        std::vector<double> heap;
        double *b = bonds.data(), *p = prefixes.data(), *s = suffixes.data();
        if (span > 128)
        {
            heap.assign(static_cast<size_t>(3 * span + 2), 0.);
            b = heap.data();
            p = b + span + 2;
            s = p + span;
        }
        b[0] = low ? dims[low - 1] : 1.;
        for (Index i = 0; i < span; ++i)
            b[i + 1] = dims[low + i];
        b[span + 1] = high < static_cast<Index>(dims.size()) ? dims[high] : 1.;
        const double d = operatorChain ? 4. : 2.;
        const double sr1 = state.hasSiteRanks ? state.siteRanks[state.inverse[low]] : d;
        const double sr2 = state.hasSiteRanks ? state.siteRanks[state.inverse[high]] : d;
        const double e1 = (sr1 <= 1. + 1e-9) ? 1. : std::max(b[0], b[1]);
        const double e2 = (sr2 <= 1. + 1e-9) ? 1. : std::max(b[span], b[span + 1]);
        for (Index i = 0; i < span - 1; ++i)
            p[i + 1] = p[i] + ((sr1 <= 1. + 1e-9) ? 0. : std::max(0., Cube(std::min(e1, sr1 * b[i + 1])) - Cube(b[i + 1])));
        for (Index i = span - 1; i > 0; --i)
            s[i - 1] = s[i] + ((sr2 <= 1. + 1e-9) ? 0. : std::max(0., Cube(std::min(e2, sr2 * b[i + 1])) - Cube(b[i + 1])));
        const double w1 = (sr1 <= 1. + 1e-9 ? 0. : 1.) * b[0] * b[1];
        const double w2 = (sr2 <= 1. + 1e-9 ? 0. : 1.) * b[span] * b[span + 1];
        Index chosen = low;
        double best = std::numeric_limits<double>::infinity();
        for (Index i = 0; i < span; ++i)
        {
            const double tie = 1e-6 * (w2 > w1 ? span - 1 - i : i);
            const double score = Cube(b[i + 1]) + .5 * (p[i] + s[i]) + .05 * b[i] * b[i + 1] * b[i + 2] + tie;
            if (score < best)
            {
                best = score;
                chosen = low + i;
            }
        }
        return chosen;
    }

    CandidateSet CandidatesFor(Index entry) const
    {
        CandidateSet out;
        const auto &gate = gates_[entry];
        const Index low = std::min(initial_.map[gate.a], initial_.map[gate.b]);
        const Index high = std::max(initial_.map[gate.a], initial_.map[gate.b]);
        const auto add = [&out](Index position) {
            for (size_t i = 0; i < out.count; ++i)
                if (out.positions[i] == position)
                    return;
            out.positions[out.count++] = position;
        };
        // Ordering is part of the policy: exact score ties keep the first.
        add(LocalMeeting(initial_, low, high, operatorChain_));
        add(low);
        add(high - 1);
        add(low + (high - low) / 2);
        Index minimum = low;
        for (Index b = low + 1; b < high; ++b)
            if (initial_.bonds[b] < initial_.bonds[minimum])
                minimum = b;
        add(minimum);
        return out;
    }

    size_t CommonHorizon(Index entry, size_t candidates) const
    {
        size_t steps = 0, upperBound = 0;
        const size_t perCandidateBudget = kForecastOperationBudget / candidates;
        for (size_t step = 0; step < 24 && static_cast<size_t>(entry) + step < gates_.size(); ++step)
        {
            const auto &gate = gates_[static_cast<size_t>(entry) + step];
            if (!gate.valid)
                break;
            // Every candidate sees the same prefix. Each projection may tighten
            // all n - 1 bonds during its outward sweeps.
            size_t updates = gate.swap ? 1 : n_ - 1;
            if (step == 0 && !gate.swap)
                updates = static_cast<size_t>(std::abs(initial_.map[gate.a] - initial_.map[gate.b]));
            const size_t singles = step ? gate.singlesBefore.size() : 0;
            if (updates > perCandidateBudget - upperBound || singles > perCandidateBudget - upperBound - updates)
                break;
            upperBound += updates + singles;
            const size_t collapses = step ? gate.disentanglesBefore.size() : 0;
            if (collapses)
            {
                if (collapses > (perCandidateBudget - upperBound) / (n_ - 1))
                    break;
                upperBound += collapses * (n_ - 1);
            }
            ++steps;
        }
        return steps;
    }

    double Grow(double left, double middle, double right, double maximum, bool swap, double gateRank, double eLeft, double eRight) const
    {
        const double d = operatorChain_ ? 4. : 2., rank = operatorChain_ ? gateRank * gateRank : gateRank;
        if (swap)
        {
            if (left == 1. && middle == 1. && right == 1.)
                return 1.;
            const bool leftProd = eLeft <= 1. + 1e-9;
            const bool rightProd = eRight <= 1. + 1e-9;
            if (leftProd && rightProd)
                return std::clamp(middle, 1., maximum);
            if (leftProd)
                return std::clamp(right, 1., maximum);
            if (rightProd)
                return std::clamp(left, 1., maximum);
            return std::max(1., std::min({eRight * left, eLeft * right, eLeft * eRight * middle, maximum}));
        }
        const auto at = [&](double r) { return std::max(1., std::min(.7 * std::min(d * std::min(left, right), middle * r), maximum)); };
        return rank >= 2. ? at(rank) : middle + (std::max(rank, 1.) - 1.) * (at(2.) - middle);
    }

    double StepWork(double left, double middle, double right, double post, bool swap) const
    {
        const double d = operatorChain_ ? 4. : 2.;
        if (swap && left == 1. && middle == 1. && right == 1.)
            return 4. * d;
        const double m = d * left, n = d * right;
        return m * n * (middle + std::min(m, n) + post);
    }

    void Relabel(Index a, Index b)
    {
        std::swap(scratch_.map[a], scratch_.map[b]);
        scratch_.inverse[scratch_.map[a]] = a;
        scratch_.inverse[scratch_.map[b]] = b;
    }

    void ApplyScratchDisentangle(Index q, double probability)
    {
        if (q < 0 || static_cast<size_t>(q) >= n_ || n_ < 2 || probability <= 0.)
            return;
        scratch_.hasSiteRanks = true;
        const Index p = scratch_.map[q];
        const double d = operatorChain_ ? 4. : 2.;
        const Index nBonds = static_cast<Index>(n_ - 1);
        const auto blend = [probability](double collapsed, double current) {
            return probability >= 1. ? collapsed : (current + probability * (collapsed - current));
        };

        Index sweepLeft = p - 1, sweepRight = p;
        if (p == 0)
        {
            scratch_.bonds[0] = std::clamp(blend(1., scratch_.bonds[0]), 1., maximum_[0]);
            sweepLeft = -1;
            sweepRight = 1;
        }
        else if (p == static_cast<Index>(n_ - 1))
        {
            scratch_.bonds[nBonds - 1] = std::clamp(blend(1., scratch_.bonds[nBonds - 1]), 1., maximum_[nBonds - 1]);
            sweepLeft = nBonds - 2;
            sweepRight = nBonds;
        }
        else
        {
            const double chiL = scratch_.bonds[p - 1], chiR = scratch_.bonds[p];
            const double chiMin = std::min(chiL, chiR), chiMax = std::max(chiL, chiR);
            double collapsed = chiMin;
            if (scratch_.siteRanks[q] > 1. + 1e-9)
            {
                const double eSym = std::max(1., scratch_.siteRanks[q] / (chiMax / std::max(1., chiMin)));
                collapsed = std::max(1., chiMin / std::pow(eSym, 0.75));
            }
            scratch_.bonds[p - 1] = std::clamp(blend(collapsed, chiL), 1., maximum_[p - 1]);
            scratch_.bonds[p] = std::clamp(blend(collapsed, chiR), 1., maximum_[p]);
            sweepLeft = p - 2;
            sweepRight = p + 1;
        }
        scratch_.siteRanks[q] = std::max(1., blend(1., scratch_.siteRanks[q]));
        for (Index b = std::max<Index>(1, sweepRight); b < nBonds; ++b)
        {
            const double limit = d * scratch_.bonds[b - 1];
            if (scratch_.bonds[b] > limit)
                scratch_.bonds[b] = std::clamp(limit, 1., maximum_[b]);
            else
                break;
        }
        for (Index b = std::min<Index>(nBonds - 2, sweepLeft); b >= 0; --b)
        {
            const double limit = d * scratch_.bonds[b + 1];
            if (scratch_.bonds[b] > limit)
                scratch_.bonds[b] = std::clamp(limit, 1., maximum_[b]);
            else
                break;
        }
    }

    void Adjacent(Index bond, bool swap, double gateRank, double &work)
    {
        const double d = operatorChain_ ? 4. : 2.;
        const double left = bond ? scratch_.bonds[bond - 1] : 1., middle = scratch_.bonds[bond];
        const double right = bond + 1 < static_cast<Index>(n_ - 1) ? scratch_.bonds[bond + 1] : 1.;
        const Index qL = scratch_.inverse[bond], qR = scratch_.inverse[bond + 1];
        const double eL = scratch_.hasSiteRanks ? scratch_.siteRanks[qL] : d;
        const double eR = scratch_.hasSiteRanks ? scratch_.siteRanks[qR] : d;
        const double post = Grow(left, middle, right, maximum_[bond], swap, gateRank, eL, eR);
        work += StepWork(left, middle, right, post, swap);
        scratch_.bonds[bond] = post;
        if (swap)
            Relabel(qL, qR);
        else if (scratch_.hasSiteRanks && gateRank > 1. + 1e-9)
        {
            const double fullRank = operatorChain_ ? gateRank * gateRank : gateRank;
            const double factor = 1. + .7 * (std::min(d, fullRank) - 1.);
            scratch_.siteRanks[qL] = std::clamp(scratch_.siteRanks[qL] * factor, 1., d);
            scratch_.siteRanks[qR] = std::clamp(scratch_.siteRanks[qR] * factor, 1., d);
        }
    }

    double Evaluate(Index entry, Index meeting, size_t horizon, double incumbent)
    {
        scratch_ = initial_;
        double work = 0.;
        const double d = operatorChain_ ? 4. : 2.;
        for (size_t step = 0; step < horizon && static_cast<size_t>(entry) + step < gates_.size(); ++step)
        {
            const auto &gate = gates_[static_cast<size_t>(entry) + step];
            if (!gate.valid)
                break;
            if (step)
            {
                size_t nextSingle = 0;
                const auto chargeSingles = [&](size_t end) {
                    for (; nextSingle < end; ++nextSingle)
                    {
                        const auto &[qubit, count] = gate.singlesBefore[nextSingle];
                        const Index position = scratch_.map[qubit];
                        const double left = position ? scratch_.bonds[position - 1] : 1.;
                        const double right = position < static_cast<Index>(n_ - 1) ? scratch_.bonds[position] : 1.;
                        work += static_cast<double>(count) * d * d * left * right;
                    }
                };
                for (const auto &dis : gate.disentanglesBefore)
                {
                    chargeSingles(dis.singlesEnd);
                    ApplyScratchDisentangle(dis.qubit, dis.probability);
                }
                chargeSingles(gate.singlesBefore.size());
            }
            if (gate.swap)
            {
                // A logical SWAP: each label now names the other state, and the site ranks describe states.
                Relabel(gate.a, gate.b);
                std::swap(scratch_.siteRanks[gate.a], scratch_.siteRanks[gate.b]);
                continue;
            }
            const Index low = std::min(scratch_.map[gate.a], scratch_.map[gate.b]);
            const Index high = std::max(scratch_.map[gate.a], scratch_.map[gate.b]);
            const Index meet = step == 0 ? meeting : LocalMeeting(scratch_, low, high, operatorChain_);
            // Native routing moves the lower endpoint right before moving the
            // upper endpoint left. Each step sees the updated adjacent ranks.
            for (Index bond = low; bond < meet; ++bond)
                Adjacent(bond, true, 4., work);
            for (Index bond = high - 1; bond > meet; --bond)
                Adjacent(bond, true, 4., work);
            Adjacent(meet, false, gate.rank, work);
            // Remaining work terms are nonnegative. A candidate that cannot
            // strictly improve the incumbent also cannot win its stable tie.
            if (work >= incumbent)
                return work;
        }
        return work;
    }
};

} // namespace Private
} // namespace Simulators
