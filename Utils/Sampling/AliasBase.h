#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Utils
{
class AliasBase
{
  protected:
    // These are table positions, not state labels. Even a >32-qubit sparse
    // path-integral state can use compact indices when its support fits.
    static constexpr bool UseCompactIndices(size_t count)
    {
        return count == 0 || count - 1 <= std::numeric_limits<uint32_t>::max();
    }

    template <class Populate> void SetAliasTable(size_t count, const Populate &populate)
    {
        if (count == 0)
            throw std::domain_error("Alias sampling requires nonempty probability support");

        probabilities.resize(count);
        if (UseCompactIndices(count))
        {
            compactAliases.resize(count);
            BuildAliasTable(compactAliases, populate);
        }
        else
        {
            wideAliases.resize(count);
            BuildAliasTable(wideAliases, populate);
        }
    }

    template <class Index, class Populate> void BuildAliasTable(std::vector<Index> &aliases, const Populate &populate)
    {
        // Until a row is finalized, its alias slot links to the next row in
        // its under/over stack. This needs no separate construction buffers.
        // Counts avoid reserving a sentinel index, so all 2^32 compact rows fit.
        Index under = 0, over = 0;
        size_t numUnder = 0, numOver = 0;
        populate([&](size_t row, double probability) {
            probabilities[row] = probability;
            if (probability < 1.)
            {
                aliases[row] = under;
                under = static_cast<Index>(row);
                ++numUnder;
            }
            else
            {
                aliases[row] = over;
                over = static_cast<Index>(row);
                ++numOver;
            }
        });

        while (numUnder && numOver)
        {
            const Index u = under, o = over;
            under = aliases[u];
            over = aliases[o];
            --numUnder;
            --numOver;

            aliases[u] = o;
            probabilities[o] = probabilities[o] + probabilities[u] - 1.;
            if (probabilities[o] < 1.)
            {
                aliases[o] = under;
                under = o;
                ++numUnder;
            }
            else
            {
                aliases[o] = over;
                over = o;
                ++numOver;
            }
        }

        while (numUnder)
        {
            const Index row = under;
            under = aliases[row];
            --numUnder;
            probabilities[row] = 1.;
            aliases[row] = row;
        }
        while (numOver)
        {
            const Index row = over;
            over = aliases[row];
            --numOver;
            probabilities[row] = 1.;
            aliases[row] = row;
        }
    }

    size_t SampleIndex(double v) const
    {
        const double vadj = v * probabilities.size();
        const size_t offset = std::min<size_t>(static_cast<size_t>(vadj), probabilities.size() - 1);
        const double up = std::min<double>(vadj - offset, oneMinusEps);
        if (up < probabilities[offset])
            return offset;
        return compactAliases.empty() ? wideAliases[offset] : compactAliases[offset];
    }

    std::vector<double> probabilities;
    std::vector<uint32_t> compactAliases;
    std::vector<uint64_t> wideAliases;
    static constexpr double oneMinusEps = 1. - std::numeric_limits<double>::epsilon();
};

} // namespace Utils
