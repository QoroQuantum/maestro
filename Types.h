/**
 * @file Types.h
 * @ingroup core
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Some types to make the code portable.
 *
 */

#pragma once

#ifndef _TYPES_H_
#define _TYPES_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace Types
{
using qubit_t = uint_fast64_t;              /**< The type of a qubit. */
using qubits_vector = std::vector<qubit_t>; /**< The type of a vector of qubits. */
using time_type = double;                   /**< The type of time. */

/**
 * @brief The mask of bit `index` in a 64-bit outcome, or 0 beyond it.
 *
 * Bitmask measurement APIs cannot represent qubits from 64 on, and shifting by
 * that many bits is undefined; those bits are dropped instead (the APIs warn).
 */
constexpr uint64_t BitAt(size_t index) noexcept
{
    return index < 64 ? uint64_t{1} << index : 0;
}

/**
 * @brief Hash of a classical record, for count maps.
 *
 * std::hash<std::vector<bool>> need not spread its input: MSVC hashes bytes
 * with FNV-1a, which never carries bit 7 of a byte into the low hash bits an
 * unordered_map masks to pick a bucket, so records that differ only there
 * share 1/128 of the buckets. A SplitMix64 finalizer, a bijection in which
 * every input bit reaches every output bit, completes the standard hash.
 */
struct BitsHash
{
    size_t operator()(const std::vector<bool> &bits) const noexcept
    {
        uint64_t value = static_cast<uint64_t>(std::hash<std::vector<bool>>{}(bits));
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<size_t>(value ^ (value >> 31));
    }
};
} // namespace Types

#endif // !_TYPES_H_
