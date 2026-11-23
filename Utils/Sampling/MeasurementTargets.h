#pragma once

#include "../../Types.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Utils
{
// A joint measurement collapses each wire once. The caller still receives one
// result for every requested position, in the order in which it listed them.
class MeasurementTargets
{
  public:
    explicit MeasurementTargets(const Types::qubits_vector &wires)
    {
        if (wires.size() < 2)
            return;
        // Test wires below 64 without allocating, even when they are out of order.
        uint64_t seen = 0;
        for (const auto wire : wires)
        {
            const auto bit = Types::BitAt(wire);
            if (!bit)
            {
                // An ascending list of wider wire IDs is also already unique.
                if (std::adjacent_find(wires.begin(), wires.end(), std::greater_equal<>{}) != wires.end())
                    FindDuplicates(wires);
                return;
            }
            if (seen & bit)
            {
                FindDuplicates(wires);
                return;
            }
            seen |= bit;
        }
    }

    template <class Measure> static auto Read(const Types::qubits_vector &wires, Measure &&measure)
    {
        if (wires.size() < 2)
            return measure(wires);
        const MeasurementTargets targets(wires);
        return targets.Expand(measure(targets.Physical(wires)));
    }

    const Types::qubits_vector &Physical(const Types::qubits_vector &wires) const
    {
        return positions.empty() ? wires : physical;
    }

  private:
    std::vector<bool> Expand(std::vector<bool> measured) const
    {
        if (positions.empty() || measured.empty())
            return measured;
        std::vector<bool> result(positions.size());
        for (size_t i = 0; i < positions.size(); ++i)
            result[i] = measured.at(positions[i]);
        return result;
    }

    uint64_t Expand(uint64_t measured) const
    {
        if (positions.empty())
            return measured;
        uint64_t result = 0;
        for (size_t i = 0; i < std::min<size_t>(positions.size(), 64); ++i)
            if (measured & Types::BitAt(positions[i]))
                result |= Types::BitAt(i);
        return result;
    }

    void FindDuplicates(const Types::qubits_vector &wires)
    {
        std::unordered_map<Types::qubit_t, size_t> index;
        for (const auto wire : wires)
            index.emplace(wire, index.size());
        if (index.size() == wires.size())
            return;
        physical.resize(index.size());
        positions.reserve(wires.size());
        for (const auto wire : wires)
        {
            const auto i = index.at(wire);
            physical[i] = wire;
            positions.push_back(i);
        }
    }

    Types::qubits_vector physical;
    std::vector<size_t> positions;
};
} // namespace Utils
