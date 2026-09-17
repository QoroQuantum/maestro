"""Low-level simulator operations corresponding to maestrolib/Interface.h."""

import cmath
import math

import maestro
import pytest


@pytest.fixture
def simulator():
    owner = maestro.Maestro()
    handle = owner.create_simulator(
        maestro.SimulatorType.QCSim, maestro.SimulationType.Statevector)
    sim = owner.get_simulator(handle)
    try:
        sim.AllocateQubits(3)
        sim.InitializeSimulator()
        sim.set_seed(123)
        yield sim
    finally:
        owner.destroy_simulator(handle)


S = math.sqrt(0.5)
THETA, PHI, LAMBDA, GAMMA = 0.73, -0.31, 0.49, 0.22
C, T = math.cos(THETA / 2), math.sin(THETA / 2)
U = tuple(tuple(cmath.exp(1j * GAMMA) * v for v in row) for row in (
    (C, -cmath.exp(1j * LAMBDA) * T),
    (cmath.exp(1j * PHI) * T, cmath.exp(1j * (PHI + LAMBDA)) * C),
))
GATES = [
    ("X", (), ((0, 1), (1, 0))),
    ("Y", (), ((0, -1j), (1j, 0))),
    ("Z", (), ((1, 0), (0, -1))),
    ("H", (), ((S, S), (S, -S))),
    ("S", (), ((1, 0), (0, 1j))),
    ("SDG", (), ((1, 0), (0, -1j))),
    ("T", (), ((1, 0), (0, cmath.exp(1j * math.pi / 4)))),
    ("TDG", (), ((1, 0), (0, cmath.exp(-1j * math.pi / 4)))),
    ("SX", (), (((1 + 1j) / 2, (1 - 1j) / 2),
                ((1 - 1j) / 2, (1 + 1j) / 2))),
    ("SXDG", (), (((1 - 1j) / 2, (1 + 1j) / 2),
                  ((1 + 1j) / 2, (1 - 1j) / 2))),
    ("K", (), ((S, -1j * S), (1j * S, -S))),
    ("P", (THETA,), ((1, 0), (0, cmath.exp(1j * THETA)))),
    ("Rx", (THETA,), ((C, -1j * T), (-1j * T, C))),
    ("Ry", (THETA,), ((C, -T), (T, C))),
    ("Rz", (THETA,), ((cmath.exp(-1j * THETA / 2), 0),
                       (0, cmath.exp(1j * THETA / 2)))),
    ("U", (THETA, PHI, LAMBDA, GAMMA), U),
]
CONTROLLED_GATES = [case for case in GATES if case[0] in {
    "X", "Y", "Z", "H", "SX", "SXDG", "P", "Rx", "Ry", "Rz", "U",
}]


def apply_matrix(matrix, state):
    return [sum(a * b for a, b in zip(row, state)) for row in matrix]


@pytest.mark.parametrize("name,args,matrix", GATES, ids=[g[0] for g in GATES])
def test_single_qubit_gate_amplitudes(simulator, name, args, matrix):
    # A complex superposition distinguishes phase gates and inverse gates.
    simulator.ApplyH(1)
    simulator.ApplyT(1)
    initial = [S, S * cmath.exp(1j * math.pi / 4)]

    getattr(simulator, "Apply" + name)(1, *args)

    expected = [0j] * 8
    expected[0], expected[2] = apply_matrix(matrix, initial)
    assert [simulator.Amplitude(i) for i in range(8)] == pytest.approx(expected)


@pytest.mark.parametrize("name,args,matrix", CONTROLLED_GATES,
                         ids=[g[0] for g in CONTROLLED_GATES])
def test_controlled_gate_amplitudes(simulator, name, args, matrix):
    # Keep both control sectors populated and use nonadjacent, reversed indices.
    simulator.ApplyH(2)
    simulator.ApplyH(0)
    simulator.ApplyT(0)
    initial = [0.5, 0.5 * cmath.exp(1j * math.pi / 4)]

    getattr(simulator, "ApplyC" + name)(2, 0, *args)

    expected = [0j] * 8
    expected[0], expected[1] = initial
    expected[4], expected[5] = apply_matrix(matrix, initial)
    assert [simulator.Amplitude(i) for i in range(8)] == pytest.approx(expected)


@pytest.mark.parametrize("gate,qubits,ones,outcome", [
    ("ApplySwap", (0, 2), [0], 4),
    ("ApplyCSwap", (1, 0, 2), [0, 1], 6),
    ("ApplyCSwap", (1, 0, 2), [0], 1),
    ("ApplyCCX", (2, 0, 1), [0, 2], 7),
    ("ApplyCCX", (2, 0, 1), [2], 4),
])
def test_swap_and_three_qubit_gates(simulator, gate, qubits, ones, outcome):
    for qubit in ones:
        simulator.ApplyX(qubit)
    getattr(simulator, gate)(*qubits)
    assert simulator.Probability(outcome) == pytest.approx(1)


def test_configuration_and_backend_queries(simulator):
    assert isinstance(simulator, maestro.Simulator)
    assert simulator.GetSimulatorType() == maestro.SimulatorType.QCSim
    assert simulator.GetSimulationType() == maestro.SimulationType.Statevector
    assert simulator.IsQcsim() is True
    assert simulator.GetConfiguration("method") == "statevector"
    simulator.ConfigureSimulator(key="max_bond_dimension", value="32")
    assert simulator.GetConfiguration("max_bond_dimension") == "32"
    simulator.Configure("max_bond_dimension", "64")
    assert simulator.GetConfiguration("max_bond_dimension") == "64"
    assert simulator.GetConfiguration("unknown_option") == ""
    simulator.SetMultithreading(False)
    assert simulator.GetMultithreading() is False
    simulator.SetMultithreading()
    assert simulator.GetMultithreading() is True


def test_probabilities_amplitudes_and_sampling(simulator):
    simulator.ApplyH(0)
    simulator.ApplyCX(0, 1)
    simulator.ApplyS(0)
    simulator.FlushSimulator()

    probabilities = simulator.AllProbabilities()
    assert probabilities == pytest.approx([0.5, 0, 0, 0.5, 0, 0, 0, 0])
    assert simulator.Probabilities([3, 0, 1]) == pytest.approx([0.5, 0.5, 0])
    assert simulator.Amplitude(0) == pytest.approx(S)
    assert simulator.Amplitude(3) == pytest.approx(1j * S)
    assert isinstance(simulator.Amplitude(3), complex)
    counts = simulator.SampleCounts(qubits=[0, 1], shots=512)
    assert set(counts) == {0, 3}
    assert sum(counts.values()) == 512
    assert sum(simulator.SampleCounts([0, 1]).values()) == 1000
    assert simulator.AllProbabilities() == pytest.approx(probabilities)

    # A returned list remains a snapshot after the native state changes.
    simulator.ResetSimulator()
    assert probabilities[0] == pytest.approx(0.5)
    assert simulator.Probability(0) == pytest.approx(1)


def test_measurement_order_and_reset(simulator):
    simulator.ApplyX(0)
    simulator.ApplyX(2)
    assert simulator.MeasureNoCollapse() == 5
    assert simulator.SampleCounts([2, 0, 1], 32) == {3: 32}
    assert simulator.Measure([2, 0, 1]) == 3
    simulator.ApplyReset([2])
    assert simulator.Probability(1) == pytest.approx(1)
    simulator.Reset()
    assert simulator.Probability(0) == pytest.approx(1)


def test_clear_and_reallocate(simulator):
    assert simulator.GetNumberOfQubits() == 3
    simulator.ClearSimulator()
    assert simulator.GetNumberOfQubits() == 0
    assert simulator.AllocateQubits(num_qubits=2) == 0
    simulator.Initialize()
    assert simulator.GetNumberOfQubits() == 2
    assert simulator.Probability(0) == pytest.approx(1)
    simulator.Clear()
    assert simulator.GetNumberOfQubits() == 0


def test_save_restore_and_flush(simulator):
    simulator.ApplyH(0)
    simulator.ApplyS(0)
    simulator.SaveState()
    simulator.Reset()
    simulator.RestoreState()
    simulator.Flush()
    assert simulator.Amplitude(0) == pytest.approx(S)
    assert simulator.Amplitude(1) == pytest.approx(1j * S)
    simulator.ApplyX(2)
    simulator.RestoreState()
    assert simulator.Probability(4) == pytest.approx(0)


@pytest.mark.parametrize("backend_name", ["QCSim", "QiskitAer"])
def test_internal_state_save_restore(backend_name):
    if not hasattr(maestro.SimulatorType, backend_name):
        pytest.skip(f"{backend_name} is not compiled in")
    owner = maestro.Maestro()
    backend = getattr(maestro.SimulatorType, backend_name)
    handle = owner.create_simulator(backend, maestro.SimulationType.Statevector)
    sim = owner.get_simulator(handle)
    try:
        sim.AllocateQubits(2)
        sim.InitializeSimulator()
        sim.ApplyX(1)
        sim.FlushSimulator()
        sim.SaveStateToInternalDestructive()
        assert sim.MeasureNoCollapse() == 2
        sim.RestoreInternalDestructiveSavedState()
        sim.ApplyX(0)
        sim.FlushSimulator()
        assert sim.Probability(3) == pytest.approx(1)
        assert sim.GetSimulatorType() == backend
    finally:
        owner.destroy_simulator(handle)


def test_u_keyword_arguments_and_default_phase(simulator):
    simulator.ApplyU(qubit=0, theta=math.pi, phi=0, lambda_=0)
    simulator.ApplyCU(control_qubit=0, target_qubit=2,
                      theta=math.pi, phi=0, lambda_=0)
    assert simulator.Probability(5) == pytest.approx(1)


def test_negative_unsigned_arguments_rejected(simulator):
    with pytest.raises(TypeError):
        simulator.ApplyX(-1)
    with pytest.raises(TypeError):
        simulator.Measure([-1])
    with pytest.raises(TypeError):
        simulator.SampleCounts([0], shots=-1)
