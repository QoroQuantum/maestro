#pragma once

// Kernels shared by the Pauli-frame samplers (ReferenceFrameSampler and
// ExtendedFrameSampler): per-instruction random streams, Bernoulli events by
// geometric skipping, Pauli-mixture recognition of Kraus channels and the
// transposition of bit-packed classical columns into rows. Moved unchanged
// from ReferenceFrameSampler.cpp, so its seeded rows are preserved.

#include "../Simulators/Core/QuantumChannel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#if defined(_WIN32) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define MAESTRO_FRAME_LITTLE_ENDIAN 1
#endif

namespace MaestroExecution
{
namespace FrameKernels
{
// Above this rate a uniform draw per lane is cheaper than a logarithm per error.
constexpr double kDenseErrorRate = 1.0 / 16;

// rate and inverseLogSurvival = 1 / log(1 - rate); zero selects direct
// division by log(1 - rate) when its reciprocal would overflow.
template <class Rate> Rate MakeRate(double rate)
{
    Rate result;
    result.rate = std::clamp(rate, 0.0, 1.0);
    if (result.rate > 0 && result.rate < 1)
    {
        const double logSurvival = std::log1p(-result.rate);
        // Retain the existing arithmetic and seeded rows at ordinary rates.
        // Subnormal rates can have a finite logarithm whose reciprocal is not
        // representable; their gaps use division instead of multiplying infinity.
        if (-logSurvival > 1.0 / std::numeric_limits<double>::max())
            result.inverseLogSurvival = 1.0 / logSurvival;
    }
    return result;
}

// xoshiro256++, seeded through SplitMix64. Seeding is a few multiplications,
// so every random instruction can own an independent stream in every batch.
class Stream
{
  public:
    explicit Stream(uint64_t seed) noexcept
    {
        for (auto &word : state)
        {
            seed += 0x9e3779b97f4a7c15ULL;
            uint64_t value = seed;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            word = value ^ (value >> 31);
        }
    }

    uint64_t Next() noexcept
    {
        const uint64_t result = Rotate(state[0] + state[3], 23) + state[0];
        const uint64_t shifted = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        state[2] ^= shifted;
        state[3] = Rotate(state[3], 45);
        return result;
    }

    // [0, 1) and (0, 1], each from 53 bits of one engine word.
    double Uniform() noexcept
    {
        return static_cast<double>(Next() >> 11) * 0x1.0p-53;
    }

    double UniformPositive() noexcept
    {
        return static_cast<double>((Next() >> 11) + 1) * 0x1.0p-53;
    }

  private:
    static uint64_t Rotate(uint64_t value, int bits) noexcept
    {
        return (value << bits) | (value >> (64 - bits));
    }

    uint64_t state[4];
};

// Calls hit(lane) for every lane in [0, lanes) independently with probability
// rate, in increasing lane order. Sparse rates draw geometric gaps between
// errors: P(gap >= k) = (1 - rate)^k. Draws, including any hit() makes from the
// same stream, follow lane order, so fewer lanes observe a prefix of the events.
template <class Hit> void ForEachError(Stream &stream, double rate, double inverseLogSurvival, size_t lanes, Hit &&hit)
{
    if (rate <= 0)
        return;
    if (rate >= 1)
    {
        for (size_t lane = 0; lane < lanes; ++lane)
            hit(lane);
        return;
    }
    if (rate >= kDenseErrorRate)
    {
        for (size_t lane = 0; lane < lanes; ++lane)
            if (stream.Uniform() < rate)
                hit(lane);
        return;
    }
    for (size_t lane = 0; lane < lanes; ++lane)
    {
        const double logDraw = std::log(stream.UniformPositive());
        double gap;
        if (inverseLogSurvival != 0)
            gap = std::floor(logDraw * inverseLogSurvival);
        else
        {
            const double logSurvival = std::log1p(-rate);
            // Reject gaps beyond this batch before dividing: the quotient can
            // itself overflow for a microscopic rate, but its exact size is
            // irrelevant once it exceeds the remaining lanes.
            if (logDraw <= static_cast<double>(lanes - lane) * logSurvival)
                return;
            gap = std::floor(logDraw / logSurvival);
        }
        if (!(gap < static_cast<double>(lanes - lane)))
            return;
        lane += static_cast<size_t>(gap);
        hit(lane);
    }
}

// In place: bit c of block[r] becomes bit r of block[c].
inline void Transpose64(std::array<uint64_t, 64> &block) noexcept
{
    uint64_t mask = 0x00000000FFFFFFFFULL;
    for (unsigned j = 32; j != 0; j >>= 1, mask ^= mask << j)
        for (unsigned k = 0; k < 64; k = ((k | j) + 1) & ~j)
        {
            const uint64_t swapped = ((block[k] >> j) ^ block[k | j]) & mask;
            block[k] ^= swapped << j;
            block[k | j] ^= swapped;
        }
}

// Bit i of bits becomes byte out[i] (0 or 1), for i < count.
inline void WriteBits(uint64_t bits, uint8_t *out, size_t count) noexcept
{
    size_t i = 0;
    for (; i + 8 <= count; i += 8, bits >>= 8)
    {
        // Byte k keeps bit k of the replicated byte, then is normalized to 0 or 1.
        const uint64_t selected = ((bits & 0xFF) * 0x0101010101010101ULL) & 0x8040201008040201ULL;
        const uint64_t bytes = ((selected + 0x7F7F7F7F7F7F7F7FULL) >> 7) & 0x0101010101010101ULL;
#ifdef MAESTRO_FRAME_LITTLE_ENDIAN
        std::memcpy(out + i, &bytes, sizeof(bytes));
#else
        for (size_t k = 0; k < 8; ++k)
            out[i + k] = static_cast<uint8_t>(bytes >> (8 * k));
#endif
    }
    for (; i < count; ++i, bits >>= 1)
        out[i] = static_cast<uint8_t>(bits & 1U);
}

// A classical column word becomes one packed row word after transposition.
// Both output formats consume exactly the same immutable batch columns, stored
// column-major as classical[bit * batchWords + word].
template <class Consumer> void TransposeBatch(const std::vector<uint64_t> &classical, size_t width, size_t lanes, size_t batchWords, Consumer &&consume)
{
    std::array<uint64_t, 64> block;
    const size_t words = (lanes + 63) / 64;
    for (size_t w = 0; w < words && width; ++w)
    {
        const size_t laneCount = std::min<size_t>(64, lanes - 64 * w);
        for (size_t base = 0; base < width; base += 64)
        {
            const size_t count = std::min<size_t>(64, width - base);
            for (size_t r = 0; r < 64; ++r)
                block[r] = r < count ? classical[(base + r) * batchWords + w] : 0;
            Transpose64(block);
            consume(64 * w, base, count, laneCount, block);
        }
    }
}

// A channel given only by Kraus operators (phase damping, for example) is
// still a Pauli channel when its process matrix in the Pauli basis is
// diagonal. With K_i = sum_a c_ia P_a, chi_ab = sum_i c_ia conj(c_ib), and the
// channel is sum_a chi_aa P_a rho P_a exactly when every off-diagonal chi_ab
// vanishes. P_a follows QuantumChannel::Pauli: base-4 digit k of a (I, X, Y,
// Z) acts on target k. Returns the probabilities, or nothing for any other
// channel (amplitude damping, incomplete Kraus sets, three or more targets).
inline std::vector<double> PauliMixture(const Simulators::QuantumChannel &channel, size_t targets)
{
    return Simulators::QuantumChannel::PauliMixtureOf(channel.GetKrausOperators(), targets);
}
} // namespace FrameKernels
} // namespace MaestroExecution
