#pragma once

#include "core.h"
#include <cstdint>
#include <memory>
#include <nanobind/ndarray.h>

namespace maestro_bindings
{
using MaestroExecution::SampleStorageSize;

// NumPy owns the vector through a capsule. uint8 avoids vector<bool>'s packed,
// non-contiguous storage and permits disjoint workers to write separate rows.
inline nb::object MakeSamples(std::vector<uint8_t> &&values, size_t shots, size_t width)
{
    if (values.size() != SampleStorageSize(shots, width))
        throw std::invalid_argument("Sample matrix dimensions do not match its storage");
    auto storage = std::make_unique<std::vector<uint8_t>>(std::move(values));
    if (storage->empty())
        storage->reserve(1);
    auto *data = storage->data();
    nb::capsule owner(storage.get(), [](void *p) noexcept { delete static_cast<std::vector<uint8_t> *>(p); });
    storage.release();
    return nb::cast(nb::ndarray<nb::numpy, uint8_t, nb::ndim<2>, nb::c_contig>(data, {shots, width}, owner));
}

inline size_t ClassicalWidth(const CircuitPtr &circuit)
{
    return MaestroExecution::ClassicalWidth(*circuit);
}

// Bitstring keys, classical bit 0 first.
inline nb::dict CountsDict(const std::vector<std::pair<std::vector<bool>, size_t>> &counts)
{
    nb::dict out;
    std::string key;
    for (const auto &[bits, count] : counts)
    {
        key.assign(bits.size(), '0');
        for (size_t bit = 0; bit < bits.size(); ++bit)
            if (bits[bit])
                key[bit] = '1';
        out[key.c_str()] = count;
    }
    return out;
}
} // namespace maestro_bindings
