#pragma once
#include "common.h"

namespace maestro_bindings
{
// RAII Wrapper to ensure the simulator handle is destroyed strictly
struct ScopedSimulator
{
    unsigned long int handle;

    explicit ScopedSimulator(int num_qubits)
    {
        GetMaestroObjectWithMute();
        handle = CreateSimpleSimulator(num_qubits);
    }

    // Takes ownership of `adopted`, a handle configured elsewhere (a backend
    // probe), or creates one of num_qubits qubits when it is 0.
    ScopedSimulator(int num_qubits, unsigned long int adopted)
    {
        GetMaestroObjectWithMute();
        handle = adopted ? adopted : CreateSimpleSimulator(num_qubits);
    }

    ~ScopedSimulator()
    {
        if (handle != 0)
            DestroySimpleSimulator(handle);
    }

    // Disable copying to prevent double-free
    ScopedSimulator(const ScopedSimulator &) = delete;
    ScopedSimulator &operator=(const ScopedSimulator &) = delete;
};

// The register a simulator handle is created with for a circuit: its width,
// the highest qubit index plus one, which holds every qubit a network maps.
// The handle takes an int. Where the width does not fit one (an index of
// 2^31 - 1 or more), the number of qubits the circuit uses stands in: the
// network maps those onto its host, which only has to hold that many.
inline int HandleQubits(const Circuits::Circuit<double> &circuit)
{
    const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max());
    const size_t highest = circuit.GetMaxQubitIndex();
    if (highest < limit)
        return static_cast<int>(highest + 1);
    return static_cast<int>(std::clamp<size_t>(circuit.GetQubits().size(), 1, limit));
}

} // namespace maestro_bindings
