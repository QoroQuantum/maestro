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

// The width of a circuit: its highest qubit index plus one.
inline size_t CircuitWidth(const Circuits::Circuit<double> &circuit)
{
    const size_t highest = circuit.GetMaxQubitIndex();
    return highest == std::numeric_limits<size_t>::max() ? highest : highest + 1;
}

// A simulator handle counts its qubits in an int. Two rules, for the two ways
// a call uses the handle.
//
// HandleQubits: the call hands the circuit to a network, which maps the qubits
// the circuit uses onto its host. The handle gets the circuit's width, which
// holds them all. Where the width does not fit an int (an index of 2^31 - 1 or
// more), the number of qubits used stands in: the host only has to hold those.
inline int HandleQubits(const Circuits::Circuit<double> &circuit)
{
    const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max());
    const size_t width = CircuitWidth(circuit);
    if (width <= limit)
        return static_cast<int>(width);
    return static_cast<int>(std::clamp<size_t>(circuit.GetQubits().size(), 1, limit));
}

// RegisterQubits: the call needs every index up to `width`, because it drives
// the simulator itself or measures each of those qubits. Nothing can stand in
// for a width that does not fit an int: the call is refused.
inline int RegisterQubits(size_t width)
{
    if (width > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw nb::value_error("The qubit indices do not fit a simulator register: the highest index must be below 2^31 - 1.");
    return static_cast<int>(std::max<size_t>(width, 1));
}

} // namespace maestro_bindings
