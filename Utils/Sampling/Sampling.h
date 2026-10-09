#pragma once

#include "AliasBase.h"
#include "../../Types.h"
#include "../RandomStream.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace Utils::Sampling
{
// Reproducible policy v1. Family/layout choices never depend on physical teams.
// The memory limit is explicit configuration, not fluctuating free system RAM.
enum class Family
{
    Auto,
    Cdf,
    Alias
};

struct Options
{
    bool multithreading = false;
    size_t memoryBytes = size_t{256} << 20;
    size_t batchSize = size_t{1} << 20;
    Family family = Family::Auto;
};

// Validate before any parent or component configuration is mutated.
// Returns false for settings owned by another part of the simulator.
inline bool ValidateSetting(std::string_view name, std::string_view setting)
{
    if (name == "sampling_policy")
    {
        if (setting != "reproducible_v1" && setting != "legacy")
            throw std::invalid_argument("sampling_policy must be reproducible_v1 or legacy");
    }
    else if (name == "sampling_max_memory_mb")
    {
        size_t parsed = 0;
        const auto result = std::from_chars(setting.data(), setting.data() + setting.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != setting.data() + setting.size() || !parsed ||
            parsed > std::numeric_limits<size_t>::max() / (size_t{1} << 20))
            throw std::invalid_argument("sampling_max_memory_mb must be a positive size");
    }
    else
        return false;
    return true;
}

inline int Threads(bool permitted, size_t work)
{
#ifdef _OPENMP
    if (permitted && !omp_in_parallel())
    {
        const size_t useful = std::max<size_t>(1, work / 16384);
        return static_cast<int>(std::min<size_t>(std::max(1, omp_get_max_threads()), useful));
    }
#else
    (void)permitted;
    (void)work;
#endif
    return 1;
}

inline void CheckMass(double total)
{
    if (!(total > 0.) || !std::isfinite(total))
        throw std::domain_error("Sampling requires finite positive probability mass");
}

class ProbabilityAlias : public AliasBase
{
  public:
    template <class Probability>
    ProbabilityAlias(size_t count, size_t sourceCount, double mass, const Probability &probability, size_t blockSize,
                     const std::vector<double> &masses)
    {
        if (count != sourceCount)
            sourceRows.resize(count);
        SetAliasTable(count, [&](const auto &add) {
            if (sourceRows.empty())
            {
                for (size_t i = 0; i < count; ++i)
                    add(i, (probability(i) / mass) * count);
            }
            else
            {
                size_t row = 0;
                // Zero-mass blocks are known exactly from validation. Sparse
                // states need no second full scan and no dense alias table.
                for (size_t b = 0; b < masses.size(); ++b)
                    if (masses[b] > 0.)
                        for (size_t i = b * blockSize, end = std::min(sourceCount, (b + 1) * blockSize); i < end; ++i)
                        {
                            const double p = probability(i);
                            if (p > 0.)
                            {
                                sourceRows[row] = i;
                                add(row++, (p / mass) * count);
                            }
                        }
            }
        });
    }

    size_t Sample(double value) const
    {
        return SampleIndex(value);
    }
    size_t SourceRow(size_t row) const { return sourceRows.empty() ? row : sourceRows[row]; }

  private:
    std::vector<size_t> sourceRows;
};

// Probability is a nonthrowing read of immutable storage. Preparation validates
// it before sampling or alias construction; no exceptions cross OpenMP regions.
template <class Probability> class Prepared
{
  public:
    // Reserve enough for the validation index and at least one draw batch.
    // Composite callers use this to share one memory budget across components.
    static size_t MinimumMemoryBytes(size_t count, size_t shots, size_t budget = Options{}.memoryBytes)
    {
        const size_t block = BlockSize(count, shots, budget);
        const size_t blocks = count / block + (count % block != 0);
        constexpr size_t bytesPerBlock = 2 * sizeof(double) + 1;
        return std::max<size_t>(4096, blocks * bytesPerBlock + sizeof(double) + 1024);
    }

    Prepared(size_t count, size_t shots, Probability probability, Options options = {})
        : count(count), probability(std::move(probability)), options(options)
    {
        if (!count)
            throw std::domain_error("Sampling requires nonempty probability support");
        if (options.memoryBytes < 4096 || !options.batchSize)
            throw std::invalid_argument("Sampling scratch budget is too small");
        // Each block has a fixed serial summation order, independent of teams.
        blockSize = BlockSize(count, shots, options.memoryBytes);
        const size_t blocks = count / blockSize + (count % blockSize != 0);
        if (blocks > (options.memoryBytes - sizeof(double)) / (2 * sizeof(double) + 1) ||
            blocks > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
            throw std::length_error("Sampling probability index exceeds the scratch budget");
        prefix.resize(blocks + 1, 0.);
        masses.resize(blocks, 0.);
        std::vector<unsigned char> invalid(blocks, 0);
        const int threads = Threads(options.multithreading, count);
        size_t nonzero = 0;
#ifdef _OPENMP
#pragma omp parallel for if (threads > 1) num_threads(threads) schedule(static) reduction(+:nonzero)
#endif
        for (int64_t b = 0; b < static_cast<int64_t>(blocks); ++b)
        {
            double sum = 0.;
            size_t positive = 0;
            const size_t begin = size_t(b) * blockSize, end = std::min(count, begin + blockSize);
            for (size_t i = begin; i < end; ++i)
            {
                const double p = this->probability(i);
                if (!std::isfinite(p) || p < 0.)
                    invalid[b] = 1;
                sum += p;
                positive += p > 0.;
            }
            masses[b] = sum;
            nonzero += positive;
        }
        if (std::find(invalid.begin(), invalid.end(), 1) != invalid.end())
            throw std::domain_error("Sampling probabilities must be finite and nonnegative");
        for (size_t b = 0; b < blocks; ++b)
            prefix[b + 1] = prefix[b] + masses[b];
        mass = prefix.back();
        CheckMass(mass);
        std::vector<unsigned char>().swap(invalid);
        aliasCount = nonzero <= count / 4 ? nonzero : count;

        // Leave space for at least one batch, even if the table almost fills the budget.
        const size_t available = options.memoryBytes - IndexBytes();
        const bool fitsAlias = available >= 1024 && aliasCount <= (available - 1024) / AliasRowBytes();
        const bool useAlias = options.family == Family::Alias ||
                              (options.family == Family::Auto && shots >= std::max<size_t>(512, aliasCount / 4) && fitsAlias);
        if (useAlias)
        {
            if (!fitsAlias)
                throw std::length_error("Alias table exceeds the sampling scratch budget");
            alias = std::make_unique<ProbabilityAlias>(aliasCount, count, mass, this->probability, blockSize, masses);
        }
        else if (count <= 65536 && available >= 1024 && count <= (available - 1024) / sizeof(double))
        {
            localCdf.resize(count);
            for (size_t b = 0; b < blocks; ++b)
            {
                double sum = 0.;
                const size_t begin = b * blockSize, end = std::min(count, begin + blockSize);
                for (size_t i = begin; i < end; ++i)
                    localCdf[i] = sum += this->probability(i);
            }
        }
    }

    size_t Size() const { return count; }
    size_t Categories() const { return alias ? aliasCount : count; }
    size_t SourceRow(size_t category) const { return alias ? alias->SourceRow(category) : category; }
    size_t SampleCategory(double uniform) const { return alias ? alias->Sample(uniform) : Sample(uniform); }
    Family Method() const { return alias ? Family::Alias : Family::Cdf; }
    const Options &Settings() const { return options; }
    size_t ScratchAvailable() const
    {
        const size_t tables = IndexBytes() + localCdf.size() * sizeof(double) + (alias ? aliasCount * AliasRowBytes() : 0);
        return options.memoryBytes - tables;
    }

    size_t Sample(double uniform) const
    {
        if (alias)
            return alias->SourceRow(alias->Sample(uniform));
        const double target = Target(uniform);
        const size_t b = Block(target);
        const double local = LocalTarget(target, b);
        const size_t begin = b * blockSize, end = std::min(count, begin + blockSize);
        if (!localCdf.empty())
        {
            const auto it = std::upper_bound(localCdf.begin() + begin, localCdf.begin() + end, local);
            if (it != localCdf.begin() + end)
                return static_cast<size_t>(it - localCdf.begin());
        }
        double sum = 0.;
        size_t lastPositive = begin;
        for (size_t i = begin; i < end; ++i)
        {
            const double p = probability(i);
            if (p > 0.)
                lastPositive = i;
            sum += p;
            if (local < sum)
                return i;
        }
        return lastPositive;
    }

    // Always scatter to original shot positions, including composite callers.
    void Draw(const std::vector<double> &uniforms, std::vector<size_t> &out, int threads) const
    {
        out.resize(uniforms.size());
        if (threads == 1 && !alias && count > 65536 && uniforms.size() >= 128)
        {
            std::vector<std::pair<double, size_t>> ordered;
            ordered.reserve(uniforms.size());
            for (size_t i = 0; i < uniforms.size(); ++i)
                ordered.emplace_back(Target(uniforms[i]), i);
            std::sort(ordered.begin(), ordered.end());
            size_t previousBlock = std::numeric_limits<size_t>::max(), row = 0, end = 0, lastPositive = 0;
            double sum = 0.;
            for (const auto &[target, shot] : ordered)
            {
                const size_t b = Block(target);
                if (b != previousBlock)
                {
                    previousBlock = b;
                    row = b * blockSize;
                    end = std::min(count, row + blockSize);
                    sum = probability(row);
                    lastPositive = row;
                }
                const double local = LocalTarget(target, b);
                while (sum <= local && row + 1 < end)
                {
                    const double p = probability(++row);
                    if (p > 0.)
                        lastPositive = row;
                    sum += p;
                }
                out[shot] = sum > local ? row : lastPositive;
            }
            return;
        }
#ifdef _OPENMP
#pragma omp parallel for if (threads > 1) num_threads(threads) schedule(static)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(uniforms.size()); ++i)
            out[i] = Sample(uniforms[i]);
    }

  private:
    static size_t BlockSize(size_t count, size_t shots, size_t budget)
    {
        if (budget < 4096)
            throw std::invalid_argument("Sampling scratch budget is too small");
        // Grow fixed blocks until validation and a draw batch fit. This depends
        // only on logical inputs, never on the physical thread count. The
        // minimum budget leaves at least 180 blocks, so doubling cannot overflow.
        const size_t maxBlocks = (budget - 1024 - sizeof(double)) / (2 * sizeof(double) + 1);
        size_t block = shots >= count / 16 && shots >= 65536 ? 64 : 256;
        while (count / block + (count % block != 0) > maxBlocks)
            block *= 2;
        return block;
    }
    size_t AliasRowBytes() const
    {
        return sizeof(double) + (aliasCount - 1 <= std::numeric_limits<uint32_t>::max() ? sizeof(uint32_t) : sizeof(uint64_t)) +
               (aliasCount != count ? sizeof(size_t) : 0);
    }
    size_t IndexBytes() const { return (prefix.size() + masses.size()) * sizeof(double); }
    double Target(double uniform) const { return std::min(uniform * mass, std::nextafter(mass, 0.)); }
    size_t Block(double target) const
    {
        return static_cast<size_t>(std::upper_bound(prefix.begin() + 1, prefix.end(), target) - prefix.begin() - 1);
    }
    double LocalTarget(double target, size_t block) const
    {
        return std::min(target - prefix[block], std::nextafter(masses[block], 0.));
    }

    size_t count, blockSize = 256, aliasCount = 0;
    Probability probability;
    Options options;
    double mass = 0.;
    std::vector<double> prefix, masses, localCdf;
    std::unique_ptr<ProbabilityAlias> alias;
};

template <bool Many> using Counts = std::unordered_map<std::conditional_t<Many, std::vector<bool>, Types::qubit_t>, Types::qubit_t>;

template <bool Many, class Bit> Counts<Many> Single(size_t bits, const Bit &bit)
{
    if constexpr (Many)
    {
        std::vector<bool> outcome(bits);
        for (size_t i = 0; i < bits; ++i)
            outcome[i] = bit(i);
        return {{std::move(outcome), 1}};
    }
    else
    {
        Types::qubit_t outcome = 0;
        for (size_t i = 0; i < bits; ++i)
            if (bit(i))
                outcome |= Types::qubit_t{1} << i;
        return {{outcome, 1}};
    }
}

template <class Qubits> void ValidateQubits(const Qubits &qubits, size_t width, bool many)
{
    if (!many && qubits.size() > std::numeric_limits<Types::qubit_t>::digits)
        throw std::invalid_argument("Packed sampling exceeds the result word width");
    for (const auto q : qubits)
        if (q >= width)
            throw std::out_of_range("Measured qubit is outside the register");
}

// Bounded fallback for immutable, sequential probability storage whose row
// snapshot cannot fit. Sorted targets allow one traversal per batch, with no
// per-row table or copied labels. Counts do not need the original shot order.
template <bool Many, class Range, class Probability, class Bit, class Uniform>
Counts<Many> CountCdfRange(const Range &rows, size_t shots, size_t bits, const Probability &probability,
                          const Bit &bit, Uniform &&uniform, Options options = {})
{
    Counts<Many> result;
    if (!shots || !bits)
        return result;
    if (options.memoryBytes < 4096 || !options.batchSize)
        throw std::invalid_argument("Sampling scratch budget is too small");
    double mass = 0.;
    for (const auto &row : rows)
    {
        const double p = probability(row);
        if (!std::isfinite(p) || p < 0.)
            throw std::domain_error("Sampling probabilities must be finite and nonnegative");
        mass += p;
    }
    CheckMass(mass);
    const size_t batch = std::min({shots, options.batchSize, options.memoryBytes / sizeof(double)});
    std::vector<double> targets;
    targets.reserve(batch);
    std::vector<bool> key;
    if constexpr (Many) key.resize(bits);
    for (size_t begin = 0; begin < shots;)
    {
        const size_t count = std::min(batch, shots - begin);
        targets.resize(count);
        for (double &target : targets)
            target = std::min(uniform() * mass, std::nextafter(mass, 0.));
        std::sort(targets.begin(), targets.end());
        size_t sampled = 0;
        double cumulative = 0.;
        for (const auto &row : rows)
        {
            cumulative += probability(row);
            const size_t first = sampled;
            while (sampled < count && targets[sampled] < cumulative)
                ++sampled;
            if (sampled != first)
            {
                if constexpr (Many)
                {
                    for (size_t i = 0; i < bits; ++i) key[i] = bit(row, i);
                    result[key] += sampled - first;
                }
                else
                {
                    Types::qubit_t packed = 0;
                    for (size_t i = 0; i < bits; ++i)
                        if (bit(row, i)) packed |= Types::qubit_t{1} << i;
                    result[packed] += sampled - first;
                }
            }
            if (sampled == count) break;
        }
        begin += count;
    }
    return result;
}

// Bit(row, outputBit) preserves ordered/duplicate selections and wide labels.
// Uniform belongs to the caller: this utility never seeds or clones an engine.
template <bool Many, class Plan, class Bit, class Uniform>
Counts<Many> Count(const Plan &plan, size_t shots, size_t bits, const Bit &bit, Uniform &&uniform)
{
    Counts<Many> result;
    if (!shots || !bits)
        return result;
    const auto &options = plan.Settings();
    const size_t count = plan.Categories();
    const size_t available = plan.ScratchAvailable();
    // Leave at least 128 unused bytes between workers' counters, including
    // when vector storage is not aligned to a cache line.
    constexpr size_t padding = 128 / sizeof(size_t);
    const size_t stride = count <= (size_t{1} << 20) ? ((count + padding - 1) / padding + 1) * padding : 0;
    const bool histogram = stride && shots >= std::max<size_t>(1024, count / 4) && stride <= available / (2 * sizeof(size_t));
    int threads = Threads(options.multithreading, shots);
    // Tiny alias tables are RNG-bound; staging uniforms and starting workers
    // costs more than their few arithmetic operations in the serial draw loop.
    if (count <= 64 && plan.Method() == Family::Alias)
        threads = 1;
    if (histogram)
        threads = static_cast<int>(std::min({size_t(threads), available / (2 * sizeof(size_t) * stride), std::max<size_t>(1, shots / count)}));
    std::vector<bool> key;
    if constexpr (Many)
        key.resize(bits);
    auto insert = [&](size_t row, size_t amount) {
        if constexpr (Many)
        {
            for (size_t i = 0; i < bits; ++i)
                key[i] = bit(row, i);
            result[key] += amount;
        }
        else
        {
            Types::qubit_t packed = 0;
            for (size_t i = 0; i < bits; ++i)
                if (bit(row, i))
                    packed |= Types::qubit_t{1} << i;
            result[packed] += amount;
        }
    };

    std::vector<size_t> bins(histogram ? stride * threads : 0, 0);
    if (threads == 1 && (histogram || plan.Method() == Family::Alias || shots < 128 || count <= 65536))
    {
        for (size_t shot = 0; shot < shots; ++shot)
        {
            if (histogram)
                ++bins[plan.SampleCategory(uniform())];
            else
                insert(plan.Sample(uniform()), 1);
        }
    }
    else
    {
        const size_t scratch = available - bins.size() * sizeof(size_t);
        const size_t batch = std::max<size_t>(1, std::min({shots, options.batchSize, scratch / 32}));
        std::vector<double> uniforms;
        std::vector<size_t> sampled;
        for (size_t begin = 0; begin < shots;)
        {
            const size_t size = std::min(batch, shots - begin);
            uniforms.resize(size);
            for (double &value : uniforms)
                value = uniform();
            if (histogram)
            {
#ifdef _OPENMP
#pragma omp parallel if (threads > 1) num_threads(threads)
#endif
                {
                    size_t worker = 0;
#ifdef _OPENMP
                    worker = static_cast<size_t>(omp_get_thread_num());
#endif
                    size_t *local = bins.data() + worker * stride;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
                    for (int64_t i = 0; i < static_cast<int64_t>(size); ++i)
                        ++local[plan.SampleCategory(uniforms[i])];
                }
            }
            else
            {
                plan.Draw(uniforms, sampled, threads);
                for (size_t row : sampled)
                    insert(row, 1);
            }
            begin += size;
        }
    }
    if (histogram)
        for (size_t row = 0; row < count; ++row)
        {
            size_t amount = bins[row];
            for (int worker = 1; worker < threads; ++worker)
                amount += bins[size_t(worker) * stride + row];
            if (amount)
                insert(plan.SourceRow(row), amount);
        }
    return result;
}
} // namespace Utils::Sampling
