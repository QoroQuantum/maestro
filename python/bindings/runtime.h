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

} // namespace maestro_bindings
