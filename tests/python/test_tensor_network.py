"""TensorNetwork sampling, collapse, and expectations on bounded circuits.

Use short circuits for general connectivity and shallow trees for wider states;
contraction cost depends on the circuit structure as well as its qubit count.
"""

from contextlib import contextmanager
import math
import random

import maestro
import pytest


def config(seed=123):
    return maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.TensorNetwork,
        seed=seed,
    )


def execute(gates, qubits, shots, seed=123):
    qasm = (
        'OPENQASM 2.0;include "qelib1.inc";'
        f'qreg q[{qubits}];creg c[{qubits}];{gates}measure q->c;'
    )
    return maestro.simple_execute(qasm, config=config(seed), shots=shots)["counts"]


@pytest.mark.parametrize("seed", [None, 123, 456])
@pytest.mark.parametrize(
    "gates,qubits,outcomes",
    [
        ("h q[0];", 1, {"0", "1"}),
        ("h q[0];cx q[0],q[1];", 2, {"00", "11"}),
    ],
)
def test_superposition_sampling(gates, qubits, outcomes, seed):
    counts = execute(gates, qubits, 2000, seed)
    assert set(counts) == outcomes
    assert sum(counts.values()) == 2000
    for count in counts.values():
        assert 800 < count < 1200


@pytest.mark.parametrize("seed", [None, 123])
def test_repeated_cnot_sampling(seed):
    # The original shared graph could segfault during parallel sampling.
    for _ in range(20):
        assert execute("cx q[0],q[1];", 2, 10, seed) == {"00": 10}


def test_seeded_sampling_is_repeatable():
    assert execute("h q[0];", 1, 1000) == execute("h q[0];", 1, 1000)


@contextmanager
def direct_simulator(method, qubits):
    owner = maestro.Maestro()
    handle = owner.create_simulator(maestro.SimulatorType.QCSim, method)
    sim = owner.get_simulator(handle)
    try:
        sim.AllocateQubits(qubits)
        sim.InitializeSimulator()
        sim.set_seed(713)
        yield sim
    finally:
        owner.destroy_simulator(handle)


def assert_counts(counts, probabilities, shots):
    assert sum(counts.values()) == shots
    assert set(counts) <= set(probabilities)
    for outcome, probability in probabilities.items():
        if probability < 1e-12:
            assert counts.get(outcome, 0) == 0
        elif probability > 1 - 1e-12:
            assert counts.get(outcome, 0) == shots
        else:
            tolerance = 6 * math.sqrt(shots * probability * (1 - probability)) + 2
            assert abs(counts.get(outcome, 0) - shots * probability) <= tolerance


@pytest.mark.parametrize("basis", range(8))
def test_basis_sampling_and_measurement_order(basis):
    with direct_simulator(maestro.SimulationType.TensorNetwork, 3) as sim:
        for qubit in range(3):
            if basis & (1 << qubit):
                sim.ApplyX(qubit)
        for order in ([2, 0, 1], [1, 2], [0]):
            expected = sum(((basis >> q) & 1) << bit for bit, q in enumerate(order))
            assert sim.SampleCounts(order, 16) == {expected: 16}
            assert sim.Measure(order) == expected
        assert sim.Probability(basis) == pytest.approx(1, abs=1e-10)


@pytest.mark.parametrize("qubits,seed", [
    (1, 17), (2, 29), (3, 41), (4, 47), (5, 53), (8, 67),
])
def test_short_complex_circuit_against_statevector(qubits, seed):
    rng = random.Random(seed)
    operations = []
    for qubit in range(qubits):
        operations.extend([
            ("ry", "ApplyRy", (qubit, rng.uniform(-2, 2))),
            ("rz", "ApplyRz", (qubit, rng.uniform(-2, 2))),
        ])
    for qubit in range(1, qubits):
        # A balanced tree keeps the wider cases shallow (at most 24 gates).
        operations.append(("cx", "ApplyCX", ((qubit - 1) // 2, qubit)))
    operations.append(("rx", "ApplyRx", (0, .37)))
    if qubits == 4:
        # Exercise non-tree connectivity without a long or wide dense circuit.
        operations.extend([("cz", "ApplyCZ", (3, 0)),
                           ("cry", "ApplyCRy", (2, 3, -.61))])
    circuit = maestro.circuits.QuantumCircuit()
    for name, _, args in operations:
        getattr(circuit, name)(*args)

    with direct_simulator(maestro.SimulationType.Statevector, qubits) as reference:
        for _, name, args in operations:
            getattr(reference, name)(*args)
        probabilities = reference.AllProbabilities()

    # Exact expectations include complex Y phases, identities, and mixed axes.
    paulis = ["I" * qubits, "X" * qubits, "Y" * qubits, "Z" * qubits]
    paulis += ["I" * q + axis + "I" * (qubits - q - 1)
               for q in range(qubits) for axis in "XYZ"]
    paulis += [("XYZ" * qubits)[:qubits], ("ZYX" * qubits)[:qubits]]
    expected = maestro.simple_estimate(
        circuit, paulis, config=maestro.SimulatorConfig(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
        ))["expectation_values"]
    actual = maestro.simple_estimate(circuit, paulis, config=config(seed))
    assert actual["expectation_values"] == pytest.approx(expected, abs=1e-9)

    with direct_simulator(maestro.SimulationType.TensorNetwork, qubits) as sim:
        for _, name, args in operations:
            getattr(sim, name)(*args)
        # Two batches from the same state catch sampling that leaves it collapsed.
        for order in (list(reversed(range(qubits))), [qubits - 1]):
            marginal = dict.fromkeys(range(1 << len(order)), 0.0)
            for basis, probability in enumerate(probabilities):
                packed = sum(((basis >> q) & 1) << bit for bit, q in enumerate(order))
                marginal[packed] += probability
            for _ in range(2):
                assert_counts(sim.SampleCounts(order, 384), marginal, 384)
        assert [sim.Probability(i) for i in range(1 << qubits)] == pytest.approx(
            probabilities, abs=1e-9)

    # The public circuit path uses vector-valued sampling and worker clones.
    order = list(reversed(range(qubits)))
    circuit.measure([(q, bit) for bit, q in enumerate(order)])
    expected_counts = {
        "".join(str((basis >> q) & 1) for q in order): probability
        for basis, probability in enumerate(probabilities)
    }
    counts = maestro.simple_execute(circuit, config=config(seed), shots=768)["counts"]
    assert_counts(counts, expected_counts, 768)


def test_direct_bell_measurement_collapse_and_restore():
    with direct_simulator(maestro.SimulationType.TensorNetwork, 2) as sim:
        sim.ApplyH(0)
        sim.ApplyCX(0, 1)
        sim.SaveState()
        outcomes = set()
        for _ in range(32):
            sim.RestoreState()
            first = sim.Measure([0])
            outcomes.add(first)
            assert sim.Measure([0]) == first
            assert sim.Measure([1]) == first
            sim.ApplyX(1)
            assert sim.Measure([1]) == 1 - first
        assert outcomes == {0, 1}
        sim.RestoreState()
        assert sim.Probability(0) == pytest.approx(.5, abs=1e-10)


@pytest.mark.parametrize("qubits,cbits,body,probabilities", [
    (1, 2, "h q[0];measure q[0]->c[0];x q[0];measure q[0]->c[1];",
     {"01": .5, "10": .5}),
    (1, 2, "h q[0];measure q[0]->c[0];h q[0];measure q[0]->c[1];",
     {"00": .25, "01": .25, "10": .25, "11": .25}),
    (2, 3, "h q[0];cx q[0],q[1];measure q[0]->c[0];"
     "x q[1];measure q[1]->c[1];measure q[0]->c[2];",
     {"010": .5, "101": .5}),
    (2, 2, "h q[0];measure q[0]->c[0];if(c==1) x q[1];measure q[1]->c[1];",
     {"00": .5, "11": .5}),
    (2, 2, "h q[0];cx q[0],q[1];measure q[0]->c[0];"
     "reset q[1];measure q[1]->c[1];",
     {"00": .5, "10": .5}),
], ids=["flip-after-collapse", "interference-after-collapse", "bell-collapse",
        "classical-control", "reset-entangled-qubit"])
def test_mid_circuit_measurements(qubits, cbits, body, probabilities):
    qasm = ('OPENQASM 2.0;include "qelib1.inc";'
            f'qreg q[{qubits}];creg c[{cbits}];{body}')
    counts = maestro.simple_execute(qasm, config=config(), shots=512)["counts"]
    assert_counts(counts, probabilities, 512)


@pytest.mark.parametrize("gates,qubits,paulis,expected", [
    ("h q[0];s q[0];", 1, "I;X;Y;Z;Y", [1, 0, 1, 0, 1]),
    ("h q[0];cx q[0],q[1];", 2, "II;XX;YY;ZZ;XI;IZ", [1, 1, -1, 1, 0, 0]),
    ("h q[0];s q[0];x q[1];h q[2];", 3, "YZX;III;IZI;YII;IIX",
     [-1, 1, -1, 1, 1]),
    ("h q[0];cx q[0],q[1];h q[6];s q[6];x q[7];", 8,
     "XXIIIIYZ;YYIIIIII;IIIIIIYI;IIIIIIIZ;IIIIIIII", [-1, -1, 1, -1, 1]),
], ids=["complex-phase", "bell", "disconnected-components", "wide-disconnected"])
def test_known_expectation_values(gates, qubits, paulis, expected):
    qasm = f'OPENQASM 2.0;include "qelib1.inc";qreg q[{qubits}];{gates}'
    values = maestro.simple_estimate(qasm, paulis, config=config())["expectation_values"]
    assert values == pytest.approx(expected, abs=1e-10)
