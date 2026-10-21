#include "ObservableClustering.h"
#include "ConeAnalysis.h"
#include "PauliCone.h"
#include <cctype>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>

namespace MaestroExecution
{
namespace
{
constexpr size_t kNone = static_cast<size_t>(-1);

// Qubit set as a multiword bitset, so registers beyond 64 qubits are covered.
using Cone = std::vector<uint64_t>;

size_t PopCount(uint64_t x)
{
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<size_t>((x * 0x0101010101010101ULL) >> 56);
}

size_t OverlapCount(const Cone &a, const Cone &b)
{
    size_t count = 0;
    for (size_t w = 0; w < a.size(); ++w)
        count += PopCount(a[w] & b[w]);
    return count;
}

size_t Find(std::vector<size_t> &parent, size_t x)
{
    while (parent[x] != x)
    {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

void Union(std::vector<size_t> &parent, size_t a, size_t b)
{
    a = Find(parent, a);
    b = Find(parent, b);
    if (a == b)
        return;
    // Keep the smaller index as root so groups order by first member.
    if (a > b)
        std::swap(a, b);
    parent[b] = a;
}

// Connected components of the cone-overlap graph. Empty cones are skipped.
std::vector<std::vector<size_t>> ConnectedComponents(const std::vector<Cone> &cones, const std::vector<size_t> &sizes, size_t width)
{
    const size_t count = cones.size();
    std::vector<size_t> parent(count);
    std::iota(parent.begin(), parent.end(), size_t{0});
    std::vector<size_t> owner(width, kNone);
    for (size_t i = 0; i < count; ++i)
        for (size_t q = 0; q < width; ++q)
            if ((cones[i][q / 64] >> (q % 64)) & 1)
            {
                if (owner[q] == kNone)
                    owner[q] = i;
                else
                    Union(parent, i, owner[q]);
            }

    std::vector<std::vector<size_t>> groups;
    std::vector<size_t> group_of_root(count, kNone);
    for (size_t i = 0; i < count; ++i)
    {
        if (!sizes[i])
            continue;
        const size_t root = Find(parent, i);
        if (group_of_root[root] == kNone)
        {
            group_of_root[root] = groups.size();
            groups.emplace_back();
        }
        groups[group_of_root[root]].push_back(i);
    }
    return groups;
}

// Greedy bounded packing: seed each group with the largest remaining cone, then
// add the overlapping cone that grows the union least (ties: larger overlap,
// then lower index) while the union fits max(budget, seed size).
std::vector<std::vector<size_t>> PackCones(const std::vector<Cone> &cones, const std::vector<size_t> &sizes, size_t budget, size_t floor)
{
    std::vector<size_t> remaining;
    for (size_t i = 0; i < cones.size(); ++i)
        if (sizes[i])
            remaining.push_back(i);

    std::vector<std::vector<size_t>> groups;
    while (!remaining.empty())
    {
        // max_element keeps the first maximum, so ties seed the lower index.
        const auto seed = std::max_element(remaining.begin(), remaining.end(), [&](size_t a, size_t b) { return sizes[a] < sizes[b]; });
        std::vector<size_t> group{*seed};
        Cone mask = cones[*seed];
        size_t mask_size = sizes[*seed];
        const size_t limit = std::max(budget, mask_size);
        remaining.erase(seed);

        while (true)
        {
            // Absorb every contained cone together; none changes the mask.
            remaining.erase(std::remove_if(remaining.begin(), remaining.end(),
                                           [&](size_t i) {
                                               if (OverlapCount(mask, cones[i]) != sizes[i])
                                                   return false;
                                               group.push_back(i);
                                               return true;
                                           }),
                            remaining.end());
            auto best = remaining.end();
            size_t best_union = 0;
            size_t best_overlap = 0;
            for (auto it = remaining.begin(); it != remaining.end(); ++it)
            {
                const size_t overlap = OverlapCount(mask, cones[*it]);
                const size_t merged = mask_size + sizes[*it] - overlap;
                if (merged > limit || (!overlap && (!floor || merged > floor)))
                    continue;
                if (best == remaining.end() || merged < best_union || (merged == best_union && overlap > best_overlap))
                {
                    best = it;
                    best_union = merged;
                    best_overlap = overlap;
                }
            }
            if (best == remaining.end())
                break;
            for (size_t w = 0; w < mask.size(); ++w)
                mask[w] |= cones[*best][w];
            mask_size = best_union;
            group.push_back(*best);
            remaining.erase(best);
        }
        std::sort(group.begin(), group.end());
        groups.push_back(std::move(group));
    }
    return groups;
}
} // namespace

bool BenefitsFromObservableClustering(Simulators::SimulatorType simulator_type, Simulators::SimulationType simulation_type)
{
    return !(simulator_type == Simulators::SimulatorType::kGpuSim && simulation_type == Simulators::SimulationType::kPauliPropagator);
}

std::vector<std::vector<size_t>> ClusterObservables(const std::shared_ptr<Circuits::Circuit<double>> &circuit, const std::vector<std::string> &observables,
                                                    std::optional<size_t> max_cluster_qubits, bool ensemble, std::optional<size_t> min_cluster_qubits,
                                                    bool tighten_paulis, bool marginalize_resets)
{
    const size_t count = observables.size();
    if (!count)
        return {};
    std::vector<std::vector<size_t>> single(1);
    single[0].resize(count);
    std::iota(single[0].begin(), single[0].end(), size_t{0});
    if (!circuit || count <= 1 || !CanTraceCone(circuit, observables, ensemble, marginalize_resets))
        return single;

    const auto &operations = circuit->GetOperations();
    std::vector<std::vector<size_t>> op_qubits;
    op_qubits.reserve(operations.size());
    for (const auto &operation : operations)
    {
        const auto affected = operation->AffectedQubits();
        if (affected.size() > 1)
            op_qubits.emplace_back(affected.begin(), affected.end());
    }

    size_t width = circuit->GetMaxQubitIndex() + 1;
    for (const auto &pauli : observables)
    {
        if (pauli.find_first_not_of("IXYZixyz") != std::string::npos)
            return single;
        width = std::max(width, pauli.size());
    }

    // Backward lightcone of each observable; identity-only observables have an
    // empty cone and share one group.
    std::vector<Cone> cones(count, Cone((width + 63) / 64));
    std::vector<size_t> sizes(count, 0);
    std::vector<size_t> empty_group;
    // Each bit is an observable, propagated through all multi-qubit gates in
    // one backward pass. A final transpose produces arbitrary-width cone sets.
    std::vector<uint64_t> active(width), x(width), z(width);
    const auto initial_identities = tighten_paulis ? InitialIdentities(operations, width) : std::vector<bool>{};
    for (size_t base = 0; base < count; base += 64)
    {
        std::fill(active.begin(), active.end(), uint64_t{0});
        std::fill(x.begin(), x.end(), uint64_t{0});
        std::fill(z.begin(), z.end(), uint64_t{0});
        const auto batch = std::min(size_t{64}, count - base);
        for (size_t bit = 0; bit < batch; ++bit)
            for (size_t q = 0; q < observables[base + bit].size(); ++q)
                if (observables[base + bit][q] != 'I' && observables[base + bit][q] != 'i')
                {
                    const auto p = std::toupper(static_cast<unsigned char>(observables[base + bit][q]));
                    const auto mask = uint64_t{1} << bit;
                    active[q] |= mask;
                    if (p == 'X' || p == 'Y')
                        x[q] |= mask;
                    if (p == 'Z' || p == 'Y')
                        z[q] |= mask;
                }
        if (tighten_paulis)
        {
            for (size_t i = operations.size(); i-- > 0;)
            {
                if (initial_identities[i])
                    continue;
                const auto &op = operations[i];
                const auto qs = op->AffectedQubits();
                const auto keep = TracePauliOperation(op, x, z, ensemble || marginalize_resets);
                for (auto q : qs)
                    active[q] |= keep;
            }
        }
        else
        {
            for (auto it = op_qubits.rbegin(); it != op_qubits.rend(); ++it)
            {
                uint64_t affected = 0;
                for (auto q : *it)
                    affected |= active[q];
                for (auto q : *it)
                    active[q] = affected;
            }
        }
        for (size_t q = 0; q < width; ++q)
            for (size_t bit = 0; bit < batch; ++bit)
                if (active[q] & (uint64_t{1} << bit))
                {
                    cones[base + bit][q / 64] |= uint64_t{1} << (q % 64);
                    ++sizes[base + bit];
                }
    }
    for (size_t i = 0; i < count; ++i)
        if (!sizes[i])
            empty_group.push_back(i);

    auto groups = max_cluster_qubits || min_cluster_qubits
                      ? PackCones(cones, sizes, max_cluster_qubits.value_or(std::numeric_limits<size_t>::max()), min_cluster_qubits.value_or(0))
                      : ConnectedComponents(cones, sizes, width);
    if (!empty_group.empty())
        groups.push_back(std::move(empty_group));
    std::sort(groups.begin(), groups.end(), [](const auto &a, const auto &b) { return a.front() < b.front(); });
    return groups;
}
} // namespace MaestroExecution
