"""Reproduce the immutable C++ routing-choice fixtures without Maestro.

This scalar reference enumerates each declared candidate, runs its complete
forecast without pruning, and uses direct sums for the native meeting score.
It is fixture-generation documentation, not a build/runtime dependency.
"""
from pathlib import Path
import random


def local(bonds, low, high, d):
    outside_left = bonds[low - 1] if low else 1.0
    outside_right = bonds[high] if high < len(bonds) else 1.0
    e1, e2 = max(outside_left, bonds[low]), max(outside_right, bonds[high - 1])
    scores = []
    for p in range(low, high):
        damage = sum(max(0.0, min(e1, d * bonds[b]) ** 3 - bonds[b] ** 3) for b in range(low, p))
        damage += sum(max(0.0, min(e2, d * bonds[b]) ** 3 - bonds[b] ** 3) for b in range(p + 1, high))
        left = bonds[p - 1] if p else 1.0
        right = bonds[p + 1] if p + 1 < len(bonds) else 1.0
        tie = high - 1 - p if outside_right * bonds[high - 1] > outside_left * bonds[low] else p - low
        scores.append(bonds[p] ** 3 + .5 * damage + .05 * left * bonds[p] * right + 1e-6 * tie)
    return low + min(range(len(scores)), key=scores.__getitem__)


def evaluate(n, cap, mapping, ranks, gates, mpo, horizon, singles=None,
             include_single_work=True, fixed_horizon=None):
    d = 4 if mpo else 2
    a, b, _, _ = gates[0]
    low, high = sorted((mapping[a], mapping[b]))
    candidates = list(dict.fromkeys([local(ranks, low, high, d), low, high - 1,
                                    low + (high - low) // 2,
                                    min(range(low, high), key=ranks.__getitem__)]))
    singles = singles if singles is not None else [[] for _ in gates]
    if fixed_horizon is not None:
        horizon = fixed_horizon
    elif horizon == 24:
        # The same complete prefix is used for every candidate. Repeated
        # singles count once per touched logical site in the update budget.
        budget, used, retained = 4096 // len(candidates), 0, 0
        for step, gate in enumerate(gates[:24]):
            updates = 1 if gate[3] else high - low if step == 0 else n - 1
            updates += len({q for q, _ in singles[step]}) if step else 0
            if used + updates > budget:
                break
            used += updates
            retained += 1
        horizon = retained
    costs = []
    for candidate in candidates:
        positions, bonds = list(mapping), list(ranks)
        cost = 0.0
        for step, (a, b, gate_rank, swap) in enumerate(gates[:horizon]):
            if step and include_single_work:
                counts = {}
                for q, count in singles[step]:
                    counts[q] = counts.get(q, 0) + count
                # Native metadata emits touched logical sites in ascending
                # order; retain that order even when source singles alternate.
                for q, count in sorted(counts.items()):
                    p = positions[q]
                    left = bonds[p - 1] if p else 1.0
                    right = bonds[p] if p < n - 1 else 1.0
                    cost += count * d * d * left * right
            if swap:
                positions[a], positions[b] = positions[b], positions[a]
                continue
            low, high = sorted((positions[a], positions[b]))
            meeting = candidate if step == 0 else local(bonds, low, high, d)
            updates = [(q, True) for q in range(low, meeting)]
            updates += [(q, True) for q in range(high - 1, meeting, -1)]
            updates.append((meeting, False))
            for q, routing_swap in updates:
                left = bonds[q - 1] if q else 1.0
                middle = bonds[q]
                right = bonds[q + 1] if q + 1 < len(bonds) else 1.0
                maximum = min(cap, d ** min(q + 1, n - q - 1))
                rank = (4 if routing_swap else gate_rank) ** (2 if mpo else 1)
                if routing_swap:
                    post = 1.0 if left == middle == right == 1 else max(1.0, min(d * left, d * right, rank * middle, maximum))
                else:
                    def grow(r):
                        return max(1.0, min(.7 * min(d * min(left, right), middle * r), maximum))
                    post = grow(rank) if rank >= 2 else middle + (max(rank, 1) - 1) * (grow(2) - middle)
                cost += 4 * d if routing_swap and left == middle == right == 1 else d * left * d * right * (middle + min(d * left, d * right) + post)
                bonds[q] = post
                if routing_swap:
                    x, y = positions.index(q), positions.index(q + 1)
                    positions[x], positions[y] = positions[y], positions[x]
        costs.append(cost)
    choice = candidates[min(range(len(costs)), key=costs.__getitem__)]
    return choice, horizon, candidates, costs


def choose(n, cap, mapping, ranks, gates, mpo, horizon):
    return evaluate(n, cap, mapping, ranks, gates, mpo, horizon)[0]


def fixture(seed):
    rng = random.Random(seed)
    n, cap = (6, 8) if seed % 2 else (8, 16)
    mapping = list(range(n)); rng.shuffle(mapping)
    ranks = [rng.randint(1, min(cap, 2 ** min(q + 1, n - q - 1))) for q in range(n - 1)]
    gates = [(mapping.index(0), mapping.index(n - 1), 2.0, False)]
    for k in range(1, 30):
        a, b = rng.sample(range(n), 2)
        gates.append((a, b, 1.25 if k % 6 == 0 else 2.0, k % 7 == 0))
    expected = [choose(n, cap, mapping, ranks, gates, mpo, h) for mpo in (False, True) for h in (4, 24)]
    return n, cap, mapping, ranks, gates, expected


def single_fixture(n, seed):
    rng = random.Random(seed)
    cap = 16
    mapping = list(range(n)); rng.shuffle(mapping)
    ranks = [rng.randint(1, min(cap, 2 ** min(q + 1, n - q - 1))) for q in range(n - 1)]
    gates = [(mapping.index(0), mapping.index(n - 1), 2.0, False)]
    # Deliberately large singles before the current gate must NOT be charged:
    # the live state already includes them when the current routing query runs.
    singles = [[(n - 1, 23), (0, 5), (n - 1, 19)]]
    for step in range(1, 30):
        a, b = rng.sample(range(n), 2)
        gates.append((a, b, 1.25 if step % 6 == 0 else 2.0, step % 7 == 0))
        x, y, z = rng.sample(range(n), 3)
        # Interleave and repeat sites; parsing must aggregate counts and retain
        # ascending logical-site order without losing their numeric work.
        singles.append([(z, 3), (x, 701), (y, 5), (z, 11), (x, 431)])
    expected = []
    for mpo in (False, True):
        for horizon in (4, 24):
            selected, kept, candidates, costs = evaluate(n, cap, mapping, ranks, gates, mpo, horizon, singles)
            ignored, same, same_candidates, ignored_costs = evaluate(
                n, cap, mapping, ranks, gates, mpo, horizon, singles, False, kept)
            assert kept == same and candidates == same_candidates
            expected.append((selected, ignored, kept, candidates, costs, ignored_costs))
    return n, cap, mapping, ranks, gates, singles, expected


def main():
    # These seeds were fixed when introducing the independent scalar fixture.
    # Their scientific circuit performance is not being selected or measured.
    rows = [fixture(seed) for seed in (1, 2, 3, 4, 5, 6)]
    assert [row[-1] for row in rows] == [[4, 2, 4, 0], [2, 0, 2, 0], [4, 4, 2, 4],
                                       [4, 4, 4, 4], [2, 2, 2, 0], [2, 2, 2, 0]]
    text = ['// Generated by generate_bounded_routing_choices.py; independent scalar reference.',
            '// Expected order: MPS Work4, MPS Work24, MPO Work4, MPO Work24.',
            '#pragma once', '#include <array>', '#include <cstddef>', '#include <utility>', '#include <vector>',
            'namespace RoutingChoices {',
            'struct Gate { size_t a, b; double rank; bool swap; };',
            'struct Case { size_t n; long long cap; std::vector<long long> map; std::vector<double> ranks; std::vector<Gate> gates; std::array<long long,4> expected; };',
            'inline const std::vector<Case> cases{']
    for n, cap, mapping, ranks, gates, expected in rows:
        fields = lambda xs: '{' + ','.join(map(str, xs)) + '}'
        text.append(' {' + str(n) + ',' + str(cap) + ',' + fields(mapping) + ',' + fields(ranks) + ',{')
        text.extend('  {' + f'{a},{b},{rank},{str(swap).lower()}' + '},' for a, b, rank, swap in gates)
        text.append(' },' + fields(expected) + '},')
    text += ['};',
             'struct SingleExpected { long long selected, withoutSingles; size_t horizon; std::vector<long long> candidates; std::vector<double> costs, withoutCosts; };',
             'struct SingleCase { size_t n; long long cap; std::vector<long long> map; std::vector<double> ranks; std::vector<Gate> gates; std::vector<std::vector<std::pair<size_t,size_t>>> singles; std::array<SingleExpected,4> expected; };',
             'inline const std::vector<SingleCase> singleCases{']
    sensitive_choices = 0
    for n, seed in ((12, 4101), (50, 4102), (100, 4103)):
        n, cap, mapping, ranks, gates, singles, expected = single_fixture(n, seed)
        text.append(' {' + str(n) + ',' + str(cap) + ',' + fields(mapping) + ',' + fields(ranks) + ',{')
        text.extend('  {' + f'{a},{b},{rank},{str(swap).lower()}' + '},' for a, b, rank, swap in gates)
        text.append(' },{')
        text.extend('  {' + ','.join('{' + f'{q},{count}' + '}' for q, count in group) + '},' for group in singles)
        text.append(' },{{')
        for selected, ignored, kept, candidates, costs, ignored_costs in expected:
            sensitive_choices += selected != ignored
            text.append('  {' + f'{selected},{ignored},{kept},' + fields(candidates) + ',' + fields(costs) + ',' + fields(ignored_costs) + '},')
        text.append(' }}},')
        print('singles', n, [(e[0], e[1], e[2]) for e in expected])
    assert sensitive_choices >= 2, 'Fixtures must detect omitted single-qubit work, not merely budget changes'
    text += ['};', '} // namespace RoutingChoices', '']
    Path(__file__).with_name('BoundedRoutingChoices.h').write_text('\n'.join(text), encoding='utf-8')
    print([row[-1] for row in rows])


if __name__ == '__main__':
    main()
