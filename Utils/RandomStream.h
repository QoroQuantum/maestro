#pragma once

#include <atomic>
#include <cstdint>
#include <random>

namespace Utils::RandomStream
{
// A versioned stream identity is separate from the engine's current position.
// For a fixed seed this mixer maps distinct 64-bit stream ids to distinct seeds.
inline uint64_t Derive(uint64_t seed, uint64_t stream)
{
    uint64_t value = seed + 0x9e3779b97f4a7c15ULL * (stream + 1);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline uint64_t FreshSeed()
{
    std::random_device entropy;
    static std::atomic<uint64_t> next{0};
    const uint64_t random = (uint64_t(entropy()) << 32) ^ uint64_t(entropy());
    return Derive(random, next.fetch_add(1, std::memory_order_relaxed));
}

// Exactly one engine word, with a specified [0,1) conversion on every standard
// library. Do not use thread ids or worker-local engines to generate a batch.
inline double Uniform(std::mt19937_64 &engine)
{
    return static_cast<double>(engine() >> 11) * (1.0 / 9007199254740992.0);
}
} // namespace Utils::RandomStream
