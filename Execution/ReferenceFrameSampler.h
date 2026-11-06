#pragma once

#include "Transforms/Export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Circuits
{
template <typename Time> class Circuit;
}

namespace MaestroExecution
{
class ReferenceFrameSampler;

namespace FrameDetail
{
// SplitMix64 finalizer: a bijection in which every input bit affects every
// output bit, including the low bits that MSVC's unordered_map masks to pick a
// bucket. A plain FNV word combine leaves bit k of a word out of hash bits < k.
inline uint64_t MixWord(uint64_t value) noexcept
{
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

// Hash of one packed classical record, used to accumulate frame counts.
inline uint64_t HashPackedRow(const uint64_t *words, size_t count) noexcept
{
    uint64_t hash = 0x9e3779b97f4a7c15ULL ^ count;
    for (size_t i = 0; i < count; ++i)
        hash = MixWord(hash ^ words[i]);
    return hash;
}

// Counting key for records packed 64 bits per word. std::hash<vector<bool>>
// on MSVC hashes bytes with FNV-1a, which never carries bit 7 of a byte into
// the low hash bits, so records differing only there share 1/128 of buckets.
struct PackedRowHash
{
    size_t operator()(const std::vector<uint64_t> &row) const noexcept
    {
        return static_cast<size_t>(HashPackedRow(row.data(), row.size()));
    }
};
} // namespace FrameDetail

// Distinct classical records and their counts, each record listed once in an
// unspecified order. A list, unlike a vector<bool>-keyed map, lets converters
// to Python or JSON keys skip a second round of hashing.
using FrameOutcomes = std::vector<std::pair<std::vector<bool>, size_t>>;

// Unsupported gates, channels, classical control and measurement-free circuits
// return nullopt so the caller can use ordinary trajectory execution. Invalid
// dimensions and execution errors propagate; they are never silently hidden.
MAESTRO_TRANSFORM_API std::optional<ReferenceFrameSampler> CompileReferenceSampler(const std::shared_ptr<Circuits::Circuit<double>> &circuit, size_t numQubits,
                                                                                   size_t numClassicalBits);

// An immutable noisy-Clifford sampling plan. Each runtime word holds Pauli
// frames for 64 independent trajectories, rather than a coherent superposition.
// Shots run in batches of kBatchShots. Every random instruction draws from its
// own stream per batch, and sparse errors are found by geometric skipping, so
// noise costs scale with the error rate rather than with shots x channels.
class ReferenceFrameSampler
{
  public:
    static constexpr size_t kBatchWords = 16;
    static constexpr size_t kBatchShots = 64 * kBatchWords;

    size_t Width() const noexcept
    {
        return width_;
    }

    // The requested quantum register. Frames hold rows only for the qubits the
    // circuit touches (FrameQubits), compacted in increasing order.
    size_t Qubits() const noexcept
    {
        return qubits_;
    }

    size_t FrameQubits() const noexcept
    {
        return frameQubits_;
    }

    // Row-major bytes in classical-register order, including unused zero bits.
    // The rows are deterministic for a seed and independent of worker
    // scheduling, and the first n rows do not depend on the requested count.
    MAESTRO_TRANSFORM_API std::vector<uint8_t> Sample(size_t shots, uint64_t seed) const;

    // The histogram of precisely the same seeded rows as Sample(). Scratch
    // storage is bounded by one batch, independently of the requested shots;
    // the returned histogram still grows with the number of distinct outcomes.
    MAESTRO_TRANSFORM_API FrameOutcomes SampleCounts(size_t shots, uint64_t seed) const;

  private:
    enum class Kind : uint8_t
    {
        H,
        S,
        Sx,
        CX,
        CY,
        CZ,
        Swap,
        Measurement,
        Reset,
        Channel
    };

    // An error lane is a Bernoulli(rate) event. Sparse rates are sampled by
    // geometric gaps, inverseLogSurvival = 1 / log(1 - rate). Zero selects
    // direct division by log(1 - rate) when its reciprocal would overflow.
    struct ErrorRate
    {
        double rate = 0;
        double inverseLogSurvival = 0;
    };

    // One non-identity term of a Pauli channel, conditional on an error.
    struct PauliTerm
    {
        double cumulative = 1;
        bool firstX = false, firstZ = false, secondX = false, secondZ = false;
    };

    struct Instruction
    {
        Kind kind;
        size_t first = 0;
        size_t second = 0;
        bool referenceOne = false;
        bool random = false;
        std::vector<size_t> gaugeX;
        std::vector<size_t> gaugeZ;
        // Channel: the total error rate and the conditional choice of Pauli.
        ErrorRate error;
        std::vector<PauliTerm> terms;
        // Measurement: candidate flips at the larger readout rate, each kept
        // with probability (rate for the recorded value) / candidate rate.
        ErrorRate readout;
        double readout01 = 0;
        double readout10 = 0;
    };

    size_t width_ = 0;
    size_t qubits_ = 0;
    size_t frameQubits_ = 0;
    std::vector<Instruction> instructions_;

    void ForEachBatch(size_t shots, uint64_t seed, const std::function<void(size_t, size_t, const std::vector<uint64_t> &)> &consume) const;

    friend MAESTRO_TRANSFORM_API std::optional<ReferenceFrameSampler> CompileReferenceSampler(const std::shared_ptr<Circuits::Circuit<double>> &, size_t,
                                                                                              size_t);
};

} // namespace MaestroExecution
