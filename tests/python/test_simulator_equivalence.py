"""Compare single-host circuit execution with the direct simulator API."""

from collections import Counter
import math
import random

import maestro
import pytest


# Circuit method, direct method, number of angle parameters.
SINGLE_GATES = [
    ("h", "ApplyH", 0), ("x", "ApplyX", 0), ("y", "ApplyY", 0),
    ("z", "ApplyZ", 0), ("s", "ApplyS", 0), ("sdg", "ApplySDG", 0),
    ("t", "ApplyT", 0), ("tdg", "ApplyTDG", 0),
    ("sx", "ApplySX", 0), ("sxdg", "ApplySXDG", 0), ("k", "ApplyK", 0),
    ("rx", "ApplyRx", 1), ("ry", "ApplyRy", 1), ("rz", "ApplyRz", 1),
    ("p", "ApplyP", 1), ("u", "ApplyU", 3),
]
TWO_QUBIT_GATES = [
    ("cx", "ApplyCX", 0), ("cy", "ApplyCY", 0), ("cz", "ApplyCZ", 0),
    ("ch", "ApplyCH", 0), ("csx", "ApplyCSX", 0), ("csxdg", "ApplyCSXDG", 0),
    ("swap", "ApplySwap", 0), ("cp", "ApplyCP", 1),
    ("crx", "ApplyCRx", 1), ("cry", "ApplyCRy", 1), ("crz", "ApplyCRz", 1),
    ("cu", "ApplyCU", 4),
]
BATCHES = 4
SAMPLES_PER_BATCH = 512
MEASUREMENTS_PER_BATCH = 128


def random_operations(num_qubits, depth, seed):
    rng = random.Random(seed)
    operations = []

    def append_gate(gate, qubits):
        circuit_name, direct_name, num_angles = gate
        angles = tuple(rng.uniform(-math.pi, math.pi) for _ in range(num_angles))
        operations.append((circuit_name, direct_name, tuple(qubits) + angles))

    # Touch every wire and start with asymmetric complex superpositions.
    for qubit in range(num_qubits):
        append_gate(("ry", "ApplyRy", 1), [qubit])
        append_gate(("rz", "ApplyRz", 1), [qubit])
    for _ in range(depth):
        for qubit in rng.sample(range(num_qubits), num_qubits):
            append_gate(rng.choice(SINGLE_GATES), [qubit])
        for _ in range(2):
            append_gate(rng.choice(TWO_QUBIT_GATES), rng.sample(range(num_qubits), 2))
    if num_qubits >= 3:
        append_gate(("ccx", "ApplyCCX", 0), rng.sample(range(num_qubits), 3))
        append_gate(("cswap", "ApplyCSwap", 0), rng.sample(range(num_qubits), 3))
    return operations


def circuit_from_operations(operations, measurement_order=None):
    circuit = maestro.circuits.QuantumCircuit()
    for circuit_name, _, args in operations:
        getattr(circuit, circuit_name)(*args)
    if measurement_order is not None:
        circuit.measure([(qubit, bit) for bit, qubit in enumerate(measurement_order)])
    return circuit


def prepare_direct(simulator, operations):
    simulator.ResetSimulator()
    for _, direct_name, args in operations:
        getattr(simulator, direct_name)(*args)
    simulator.FlushSimulator()


def pauli_expectation(amplitudes, pauli):
    """Compute <psi|P|psi> independently from the direct state's amplitudes.

    Pauli strings are q0 first, while q0 is the least-significant basis bit.
    This also checks Y phases without requiring a new expectation binding.
    """
    result = 0j
    for source, amplitude in enumerate(amplitudes):
        target, phase = source, 1
        for qubit, operator in enumerate(pauli):
            bit = (source >> qubit) & 1
            if operator in "XY":
                target ^= 1 << qubit
            if operator == "Y":
                phase *= -1j if bit else 1j
            elif operator == "Z" and bit:
                phase *= -1
        result += amplitudes[target].conjugate() * phase * amplitude
    assert abs(result.imag) < 1e-9
    return result.real


def measurement_rotations(basis):
    operations = []
    for qubit, operator in enumerate(basis):
        if operator == "Y":
            operations.append(("sdg", "ApplySDG", (qubit,)))
        if operator in "XY":
            operations.append(("h", "ApplyH", (qubit,)))
    return operations


def assert_count_statistics(counts, probabilities, expected_shots, label):
    assert sum(counts.values()) == expected_shots, label
    assert set(counts) <= set(range(len(probabilities))), label
    for outcome, probability in enumerate(probabilities):
        variance = max(0.0, probability * (1 - probability))
        # Six binomial standard deviations plus a small-count allowance.
        tolerance = 6 * math.sqrt(expected_shots * variance) + 6
        assert abs(counts[outcome] - expected_shots * probability) <= tolerance, (
            f"{label}: outcome {outcome}, count={counts[outcome]}, "
            f"expected={expected_shots * probability}, tolerance={tolerance}")


def assert_distributions_agree(left, right, label):
    n_left, n_right = sum(left.values()), sum(right.values())
    scale = 1 / n_left + 1 / n_right
    for outcome in left.keys() | right.keys():
        pooled = (left[outcome] + right[outcome]) / (n_left + n_right)
        tolerance = 6 * math.sqrt(pooled * (1 - pooled) * scale) + 6 * scale
        assert abs(left[outcome] / n_left - right[outcome] / n_right) <= tolerance, (
            f"{label}: outcome {outcome}, "
            f"frequencies={left[outcome] / n_left}, {right[outcome] / n_right}, "
            f"tolerance={tolerance}")


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
], ids=["statevector", "mps"])
@pytest.mark.parametrize("num_qubits,depth,seed", [
    (2, 4, 11), (3, 6, 29), (4, 8, 47), (5, 5, 83),
], ids=["2q-seed11", "3q-seed29", "4q-seed47", "5q-seed83"])
def test_random_circuits_simple_vs_direct(method, num_qubits, depth, seed):
    operations = random_operations(num_qubits, depth, seed)
    circuit = circuit_from_operations(operations)
    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=method,
        precision="double",
        max_bond_dimension=1 << num_qubits,
        singular_value_threshold=0.0,
        seed=seed,
    )
    owner = maestro.Maestro()
    handle = owner.create_simulator(maestro.SimulatorType.QCSim, method)
    simulator = owner.get_simulator(handle)
    try:
        simulator.ConfigureSimulator("matrix_product_state_max_bond_dimension",
                                     str(1 << num_qubits))
        simulator.ConfigureSimulator("matrix_product_state_truncation_threshold", "0")
        simulator.AllocateQubits(num_qubits)
        simulator.InitializeSimulator()
        prepare_direct(simulator, operations)
        amplitudes = [simulator.Amplitude(i) for i in range(1 << num_qubits)]
        assert sum(abs(a) ** 2 for a in amplitudes) == pytest.approx(1, abs=1e-9)
        assert maestro.get_probabilities(circuit, config=config) == pytest.approx(
            simulator.AllProbabilities(), rel=0, abs=1e-8)

        bases = ["Z" * num_qubits, "X" * num_qubits, "Y" * num_qubits,
                 "".join("XYZ"[q % 3] for q in range(num_qubits))]
        paulis = ["I" * num_qubits] + bases + [
            "I" * qubit + axis + "I" * (num_qubits - qubit - 1)
            for qubit in range(num_qubits) for axis in "XYZ"
        ]
        estimated = maestro.simple_estimate(circuit, paulis, config=config)
        exact = dict(zip(paulis, estimated["expectation_values"]))
        for pauli in paulis:
            assert exact[pauli] == pytest.approx(
                pauli_expectation(amplitudes, pauli), rel=0, abs=1e-8), pauli

        for basis_index, basis in enumerate(bases):
            # Explicitly permute q->c mapping so symmetric states cannot hide
            # a mismatch between q0-first strings and packed integer outcomes.
            order = list(range(num_qubits))
            random.Random(seed + basis_index).shuffle(order)
            if order == list(range(num_qubits)):
                order.reverse()
            measured_operations = operations + measurement_rotations(basis)
            measured_circuit = circuit_from_operations(measured_operations, order)
            prepare_direct(simulator, measured_operations)
            probabilities = simulator.AllProbabilities()
            ordered_probabilities = [0.0] * (1 << num_qubits)
            for state, probability in enumerate(probabilities):
                outcome = sum(((state >> qubit) & 1) << bit
                              for bit, qubit in enumerate(order))
                ordered_probabilities[outcome] = probability

            simple_counts, sampled_counts, measured_counts = Counter(), Counter(), Counter()
            # Execute the direct circuit once, then sample the existing state
            # repeatedly without appending measurements or replaying gates.
            simulator.set_seed(seed + 100_000 + basis_index)
            for batch in range(BATCHES):
                config.seed = seed + 1000 * basis_index + batch
                result = maestro.simple_execute(
                    measured_circuit, config=config, shots=SAMPLES_PER_BATCH)
                for bits, count in result["counts"].items():
                    assert len(bits) == num_qubits and set(bits) <= {"0", "1"}
                    simple_counts[int(bits[::-1], 2)] += count
                sampled_counts.update(simulator.SampleCounts(order, SAMPLES_PER_BATCH))
            assert simulator.AllProbabilities() == pytest.approx(
                probabilities, rel=0, abs=1e-9), "SampleCounts changed the state"

            # Also exercise actual collapse and fresh execution between shots.
            # Seed once; resetting the seed per shot would repeat one outcome.
            simulator.set_seed(seed + 200_000 + basis_index)
            for _ in range(BATCHES * MEASUREMENTS_PER_BATCH):
                prepare_direct(simulator, measured_operations)
                measured_counts[simulator.Measure(order)] += 1

            histograms = [
                ("simple", simple_counts, BATCHES * SAMPLES_PER_BATCH),
                ("direct sampling", sampled_counts, BATCHES * SAMPLES_PER_BATCH),
                ("direct measurements", measured_counts, BATCHES * MEASUREMENTS_PER_BATCH),
            ]
            for label, counts, shots in histograms:
                context = f"{basis}, order={order}, {label}"
                assert_count_statistics(counts, ordered_probabilities, shots, context)
                # Check sampled single-qubit and full-string Pauli expectations
                # against the simple simulator's exact estimates.
                for pauli in [basis] + [
                    "I" * q + basis[q] + "I" * (num_qubits - q - 1)
                    for q in range(num_qubits)
                ]:
                    mask = sum(1 << bit for bit, qubit in enumerate(order)
                               if pauli[qubit] != "I")
                    mean = sum((-1 if (outcome & mask).bit_count() % 2 else 1) * count
                               for outcome, count in counts.items()) / shots
                    variance = max(0.0, 1 - exact[pauli] ** 2)
                    tolerance = 6 * math.sqrt(variance / shots) + 12 / shots
                    assert abs(mean - exact[pauli]) <= tolerance, (
                        f"{context}: {pauli}, sampled={mean}, exact={exact[pauli]}")
            assert_distributions_agree(simple_counts, sampled_counts,
                                       f"{basis}: simple vs direct sampling")
            assert_distributions_agree(simple_counts, measured_counts,
                                       f"{basis}: simple vs direct measurements")
    finally:
        owner.destroy_simulator(handle)
