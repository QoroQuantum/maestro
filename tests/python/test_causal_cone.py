"""Binding-level causal cones preserve expectations and existing execution APIs."""

import math

import maestro
import pytest


METHODS = [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.TensorNetwork,
    maestro.SimulationType.Stabilizer,
    maestro.SimulationType.ExtendedStabilizer,
    maestro.SimulationType.PathIntegral,
    maestro.SimulationType.PauliPropagator,
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
]


@pytest.mark.parametrize("method", METHODS)
def test_reduction_preserves_batched_observables_and_backend(method):
    circuit = maestro.circuits.QuantumCircuit()
    circuit.h(1)
    circuit.cx(1, 4)
    circuit.s(4)
    circuit.x(2)
    circuit.h(5)
    config = maestro.SimulatorConfig(
        simulation_type=method, enable_causal_cone_reduction=False
    )
    observables = ["IZIIII", "IXIIYI", "IIZIII", "IIIIII", "IXIIYI"]
    full = maestro.simple_estimate(circuit, observables, config)
    assert full["expectation_values"] == pytest.approx([0, 1, -1, 1, 1], abs=1e-10)

    config_reduced = maestro.SimulatorConfig(
        simulation_type=method, enable_causal_cone_reduction=True
    )
    reduced = maestro.simple_estimate(circuit, observables, config_reduced)
    assert reduced["expectation_values"] == pytest.approx(
        full["expectation_values"], abs=1e-10
    )
    assert reduced["method"] == full["method"]
    assert reduced["simulator"] == full["simulator"]
    assert reduced["time_taken"] >= 0
    # Reduction must not mutate the source circuit.
    assert circuit.num_qubits == 6
    assert circuit.estimate(observables, config)["expectation_values"] == pytest.approx(
        full["expectation_values"], abs=1e-10
    )


def test_large_register_reduces_before_statevector_allocation():
    circuit = maestro.circuits.QuantumCircuit()
    circuit.ry(71, 0.7)
    circuit.cx(71, 109)
    circuit.h(117)
    observable = "I" * 71 + "Z" + "I" * 46
    cfg = maestro.SimulatorConfig(enable_causal_cone_reduction=True)
    result = circuit.estimate(observable, config=cfg)
    assert result["expectation_values"] == pytest.approx([math.cos(0.7)], abs=1e-12)


def test_backward_cone_keeps_earlier_entanglement():
    circuit = maestro.circuits.QuantumCircuit()
    circuit.h(3)
    circuit.cx(3, 1)
    circuit.cx(1, 0)
    circuit.x(3)  # Later operations outside the observable's cone can go.
    circuit.h(5)
    cfg = maestro.SimulatorConfig(enable_causal_cone_reduction=True)
    for observable in ["ZIIIII", "ZZIIII", "XIIXII"]:
        expected = circuit.estimate(observable)["expectation_values"]
        result = circuit.estimate(observable, config=cfg)
        assert result["expectation_values"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "observables,expected",
    [(["IIIIII"], [1.0]), (["IIIIIZ", "IIIIIX", "IIIIIY"], [1.0, 0.0, 0.0]), ([], [])],
)
def test_identity_idle_qubits_and_empty_observables(method, observables, expected):
    circuit = maestro.circuits.QuantumCircuit()
    circuit.x(1)
    config = maestro.SimulatorConfig(
        simulation_type=method, enable_causal_cone_reduction=True
    )
    result = circuit.estimate(observables, config=config)
    assert result["expectation_values"] == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("as_string", [False, True])
def test_qasm_and_circuit_overloads(as_string):
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[5] q; ry(0.4) q[3]; h q[4];'
    circuit = source if as_string else maestro.QasmToCirc().parse_and_translate(source)
    cfg = maestro.SimulatorConfig(enable_causal_cone_reduction=True)
    result = maestro.simple_estimate(circuit, "IIIZI;IIIII", config=cfg)
    assert result["expectation_values"] == pytest.approx([math.cos(0.4), 1.0])


@pytest.mark.parametrize(
    "body,expected",
    [
        ("x q[1]; c[2] = measure q[1]; if (c[2]) { x q[0]; }", -1.0),
        ("x q[0]; reset q[0];", 1.0),
        ("x q[0]; c[0] = measure q[0];", -1.0),
    ],
)
def test_nonunitary_and_classical_operations_fall_back(body, expected):
    source = 'OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; bit[3] c; ' + body
    for enabled in [False, True]:
        cfg = maestro.SimulatorConfig(enable_causal_cone_reduction=enabled)
        result = maestro.simple_estimate(source, "ZII", config=cfg)
        assert result["expectation_values"] == pytest.approx([expected], abs=1e-12)


def test_full_cone_and_explicitly_disabled_reduction():
    circuit = maestro.circuits.QuantumCircuit()
    circuit.ry(0, 0.7)
    circuit.cx(0, 1)
    circuit.x(2)
    for enabled in [False, True]:
        cfg = maestro.SimulatorConfig(enable_causal_cone_reduction=enabled)
        result = maestro.simple_estimate(circuit, "ZZZ", config=cfg)
        assert result["expectation_values"] == pytest.approx([-1.0], abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_non_clifford_cones_against_full_statevector(seed):
    import random

    rng = random.Random(seed)
    circuit = maestro.circuits.QuantumCircuit()
    for qubit in range(7):
        circuit.ry(qubit, rng.uniform(-math.pi, math.pi))
        circuit.rz(qubit, rng.uniform(-math.pi, math.pi))
    for _ in range(4):
        a, b = rng.sample([1, 3, 5], 2)
        circuit.cx(a, b)
        circuit.ry(b, rng.uniform(-math.pi, math.pi))
    circuit.swap(1, 3)
    circuit.ccx(1, 3, 5)
    circuit.cx(0, 6)  # Independent of all requested observables.
    observables = ["IXIIIII", "IIIZIYI", "IYIXIZI"]
    expected = circuit.estimate(
        observables,
        config=maestro.SimulatorConfig(enable_causal_cone_reduction=False),
    )["expectation_values"]
    for method in [
        maestro.SimulationType.Statevector,
        maestro.SimulationType.MatrixProductState,
        maestro.SimulationType.TensorNetwork,
        maestro.SimulationType.PauliPropagator,
    ]:
        result = circuit.estimate(
            observables,
            maestro.SimulatorConfig(
                simulation_type=method, enable_causal_cone_reduction=True
            ),
        )
        assert result["expectation_values"] == pytest.approx(expected, abs=1e-9)


def test_seeded_noisy_estimation_with_spectator_qubit_noise():
    """Verify that causal cone reduction preserves noisy estimation results.

    Tests that full_noise_estimate produces identical expectation values and
    ideal values with and without causal cone reduction when noise is configured
    on both active qubits and spectator qubits (including crosstalk to a spectator).
    """
    circuit = maestro.circuits.QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(1, 0.4)
    # Spectator qubit 2 does not interact with the observables' causal cone.
    circuit.x(2)
    circuit.rz(2, 0.5)

    nm = maestro.NoiseModel()
    nm.set_depolarizing(0, 0.02)
    nm.set_depolarizing(1, 0.02)
    # Noise explicitly associated with spectator qubit 2.
    nm.set_depolarizing(2, 0.05)
    nm.set_crosstalk(0, 2, 0.02)

    observables = ["ZZI", "XXI"]
    cfg_full = maestro.SimulatorConfig(enable_causal_cone_reduction=False)
    cfg_reduced = maestro.SimulatorConfig(enable_causal_cone_reduction=True)

    seed = 42
    realizations = 25
    res_full = circuit.full_noise_estimate(
        observables,
        nm,
        noise_realizations=realizations,
        config=cfg_full,
        noise_seed=seed,
    )
    res_reduced = circuit.full_noise_estimate(
        observables,
        nm,
        noise_realizations=realizations,
        config=cfg_reduced,
        noise_seed=seed,
    )

    assert res_reduced["expectation_values"] == pytest.approx(
        res_full["expectation_values"], abs=1e-10
    )
    assert res_reduced["ideal_expectation_values"] == pytest.approx(
        res_full["ideal_expectation_values"], abs=1e-10
    )
    assert res_reduced["time_taken"] >= 0


@pytest.mark.parametrize("seed", range(16))
def test_backward_cone_against_independent_statevector(seed):
    """Use an independent dense oracle, including phases and ordered controls."""
    import cmath
    import random

    rng = random.Random(seed)
    width = 6
    circuit = maestro.circuits.QuantumCircuit()
    state = [0j] * (1 << width)
    state[0] = 1
    active = rng.sample(range(width), 3)
    spectators = [q for q in range(width) if q not in active]

    def single(matrix, target, controls=()):
        for low in range(len(state)):
            if low & (1 << target) or not all(low & (1 << c) for c in controls):
                continue
            high = low | (1 << target)
            a, b = state[low], state[high]
            state[low] = matrix[0][0] * a + matrix[0][1] * b
            state[high] = matrix[1][0] * a + matrix[1][1] * b

    def swap(a, b, controls=()):
        for low in range(len(state)):
            if low & (1 << a) or not low & (1 << b):
                continue
            if all(low & (1 << c) for c in controls):
                high = low ^ (1 << a) ^ (1 << b)
                state[low], state[high] = state[high], state[low]

    x = ((0, 1), (1, 0))
    y = ((0, -1j), (1j, 0))
    z = ((1, 0), (0, -1))
    for q in range(width):
        theta = rng.uniform(-math.pi, math.pi)
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        circuit.ry(q, theta)
        single(((c, -s), (s, c)), q)
        theta = rng.uniform(-math.pi, math.pi)
        circuit.rz(q, theta)
        single(((cmath.exp(-0.5j * theta), 0), (0, cmath.exp(0.5j * theta))), q)

    # Every case exercises all gate arities with non-monotonic qubit indices.
    for gate in ["cx", "cy", "cz", "swap", "ccx", "cswap"] * 3:
        a, b, c = rng.sample(active, 3)
        if gate in ("cx", "cy", "cz"):
            getattr(circuit, gate)(a, b)
            single({"cx": x, "cy": y, "cz": z}[gate], b, (a,))
        elif gate == "swap":
            circuit.swap(a, b)
            swap(a, b)
        elif gate == "ccx":
            circuit.ccx(a, b, c)
            single(x, c, (a, b))
        else:
            circuit.cswap(a, b, c)
            swap(b, c, (a,))
        theta = rng.uniform(-math.pi, math.pi)
        circuit.rx(a, theta)
        cosine, sine = math.cos(theta / 2), -1j * math.sin(theta / 2)
        single(((cosine, sine), (sine, cosine)), a)

    # Earlier entanglement with a spectator must stay. A later local unitary
    # on that spectator must cancel, even though it is still entangled.
    circuit.cx(active[0], spectators[0])
    single(x, spectators[0], (active[0],))
    circuit.ry(spectators[0], 0.7)
    single(((math.cos(0.35), -math.sin(0.35)),
            (math.sin(0.35), math.cos(0.35))), spectators[0])
    circuit.swap(spectators[1], spectators[2])
    swap(spectators[1], spectators[2])

    observables = []
    for pauli in ("X", "Y", "Z"):
        for q in active:
            word = ["I"] * width
            word[q] = pauli
            observables.append("".join(word))
    for _ in range(8):
        word = ["I"] * width
        for q in active:
            word[q] = rng.choice("IXYZ")
        observables.append("".join(word))
    # Empty/short strings imply trailing identities; lowercase is accepted.
    observables.extend(["I" * width, "", "zi", observables[1].lower()])

    def expectation(word):
        result = 0j
        for basis, amplitude in enumerate(state):
            destination, phase = basis, 1
            for q, pauli in enumerate(word.upper()):
                bit = (basis >> q) & 1
                if pauli in "XY":
                    destination ^= 1 << q
                if pauli == "Y":
                    phase *= -1j if bit else 1j
                elif pauli == "Z":
                    phase *= -1 if bit else 1
            result += state[destination].conjugate() * phase * amplitude
        assert abs(result.imag) < 1e-12
        return result.real

    expected = [expectation(word) for word in observables]
    for fusion in (False, True):
        for reduced in (False, True):
            config = maestro.SimulatorConfig(
                gate_fusion=fusion, enable_causal_cone_reduction=reduced
            )
            actual = circuit.estimate(observables, config)["expectation_values"]
            assert actual == pytest.approx(expected, abs=2e-12)
        # Single-observable requests have smaller cones than the shared batch.
        for word, value in zip(observables, expected):
            actual = circuit.estimate([word], config)["expectation_values"]
            assert actual == pytest.approx([value], abs=2e-12)


def test_default_config_has_causal_cone_reduction_enabled():
    cfg = maestro.SimulatorConfig()
    assert cfg.enable_causal_cone_reduction is True

    # Default estimation on 70-qubit circuit with localized observable succeeds
    # with Statevector because causal cone reduction is active by default.
    circuit = maestro.circuits.QuantumCircuit()
    circuit.ry(0, 0.4)
    circuit.cx(0, 1)
    circuit.x(69)
    res = circuit.estimate(["ZI" + "I" * 68])
    assert res["expectation_values"] == pytest.approx([math.cos(0.4)], abs=1e-12)
