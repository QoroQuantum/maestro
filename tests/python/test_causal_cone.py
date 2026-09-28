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
    config = maestro.SimulatorConfig(simulation_type=method)
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
    expected = circuit.estimate(observables)["expectation_values"]
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
