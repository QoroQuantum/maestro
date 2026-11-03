#pragma once

#include <atomic>
#include <cstdint>
#include <random>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

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

inline uint64_t ProcessId()
{
#ifdef _WIN32
    return uint64_t(_getpid());
#else
    return uint64_t(getpid());
#endif
}

// Hardware entropy is drawn once per process: a forked child sees a new process
// id and draws its own base. The statics are constant-initialised, so no
// initialisation lock exists for a fork to inherit.
inline uint64_t FreshSeed()
{
    static std::atomic<uint64_t> base{0};
    static std::atomic<uint64_t> owner{0};
    static std::atomic<uint64_t> next{0};
    const uint64_t pid = ProcessId();
    if (owner.load(std::memory_order_acquire) != pid)
    {
        std::random_device entropy;
        base.store((uint64_t(entropy()) << 32) ^ uint64_t(entropy()), std::memory_order_relaxed);
        owner.store(pid, std::memory_order_release);
    }
    return Derive(base.load(std::memory_order_relaxed), next.fetch_add(1, std::memory_order_relaxed));
}

// QCSim draws hardware entropy when its seed argument is zero.
inline unsigned int FreshNativeSeed()
{
    return static_cast<unsigned int>(FreshSeed()) | 1u;
}

// Exactly one engine word, with a specified [0,1) conversion on every standard
// library. Do not use thread ids or worker-local engines to generate a batch.
inline double Uniform(std::mt19937_64 &engine)
{
    return static_cast<double>(engine() >> 11) * (1.0 / 9007199254740992.0);
}
} // namespace Utils::RandomStream
