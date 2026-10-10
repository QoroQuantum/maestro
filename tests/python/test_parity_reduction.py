"""Exact check-subspace reduction versus independent full-register execution."""
import copy
import itertools
import random

import pytest

import maestro


def circuit_for_checks(n, checks, *, dual=False, seed=7, layers=3):
    rng = random.Random(seed)
    circuit = maestro.circuits.QuantumCircuit()
    if not dual:
        for q in range(n):
            circuit.h(q)
    for _ in range(layers):
        for check in checks:
            # The zero-state dual reverses CNOTs and interchanges Rx/Rz.
            target = check[-1]
            for control in check[:-1]:
                circuit.cx(target, control) if dual else circuit.cx(control, target)
            rotation = circuit.rx if dual else circuit.rz
            rotation(target, rng.uniform(-2, 2))
            for control in reversed(check[:-1]):
                circuit.cx(target, control) if dual else circuit.cx(control, target)
        for q in range(n):
            rotation = circuit.rz if dual else circuit.rx
            rotation(q, rng.uniform(-2, 2))
    return circuit


def cfg(enabled, **kwargs):
    return maestro.SimulatorConfig(
        auto_reduce=enabled, enable_causal_cone_reduction=False, **kwargs
    )


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("seed", [1, 17, 32])
def test_all_paulis_with_dependent_checks(dual, seed):
    # Third check is the XOR of the first two. The fourth repeats a generator.
    circuit = circuit_for_checks(4, [(0, 1), (1, 2, 3), (0, 2, 3), (0, 1)], dual=dual, seed=seed)
    observables = ["".join(p) for p in itertools.product("IXYZ", repeat=4)]
    full = circuit.estimate(observables, cfg(False))
    reduced = circuit.estimate(observables, cfg(True))
    assert reduced["auto_reduced"]
    assert (reduced["qubits_before"], reduced["qubits_after"]) == (4, 2)
    assert reduced["expectation_values"] == pytest.approx(full["expectation_values"], abs=2e-12)
    assert circuit.num_qubits == 4


def test_preserves_requested_backend_and_config():
    circuit = circuit_for_checks(5, [(0, 1, 3), (1, 2, 4)])
    observables = ["ZZIZI", "IZZIZ", "YYYYI", "IIIII", "ZIIII"]
    config = cfg(True, simulation_type=maestro.SimulationType.MatrixProductState, max_bond_dimension=32)
    expected = circuit.estimate(observables, cfg(False, simulation_type=maestro.SimulationType.MatrixProductState, max_bond_dimension=32))["expectation_values"]
    before = repr(config)
    reduced = circuit.estimate(observables, config)
    assert reduced["auto_reduced"]
    assert reduced["method"] == maestro.SimulationType.MatrixProductState.value
    assert reduced["expectation_values"] == pytest.approx(expected, abs=2e-12)
    assert repr(config) == before
    assert copy.copy(config).auto_reduce
    assert "auto_reduce=True" in repr(config)


@pytest.mark.parametrize("breaker", ["h", "ry", "reset", "cx"])
def test_unsupported_operations_fall_back(breaker):
    circuit = circuit_for_checks(4, [(0, 1), (1, 2, 3)])
    if breaker == "ry":
        circuit.ry(0, 0.37)
    elif breaker == "cx":
        circuit.cx(0, 1)  # An uncomputed parity frame cannot be discarded.
    else:
        getattr(circuit, breaker)(0)
    full = circuit.estimate("ZZII", cfg(False, seed=1234))
    reduced = circuit.estimate("ZZII", cfg(True, seed=1234))
    assert not reduced["auto_reduced"]
    assert reduced["expectation_values"] == pytest.approx(full["expectation_values"], abs=1e-12)


def test_full_rank_and_partial_preparation_fall_back():
    circuit = circuit_for_checks(3, [(0,), (1,), (2,)])
    assert not circuit.estimate("ZZZ", cfg(True))["auto_reduced"]
    partial = maestro.circuits.QuantumCircuit()
    partial.h(0)
    partial.cx(0, 1)
    partial.rz(1, 0.3)
    partial.cx(0, 1)
    assert not partial.estimate("ZZ", cfg(True))["auto_reduced"]


def test_qasm_dependent_checks_and_causal_cone_composition():
    source = '''OPENQASM 3.0; include "stdgates.inc";
    qubit[4] q;
    h q;
    cx q[0],q[1]; rz(0.6) q[1]; cx q[0],q[1];
    cx q[1],q[2]; rz(-0.2) q[2]; cx q[1],q[2];
    cx q[0],q[2]; rz(0.8) q[2]; cx q[0],q[2];
    rx(-0.4) q;
    '''
    expected = maestro.simple_estimate(source, "ZZII;IZZI;ZIZI;IIII;YIII", cfg(False))
    actual = maestro.simple_estimate(source, "ZZII;IZZI;ZIZI;IIII;YIII", maestro.SimulatorConfig(auto_reduce=True))
    assert actual["auto_reduced"]
    assert actual["expectation_values"] == pytest.approx(expected["expectation_values"], abs=1e-12)


def test_large_register_reduces_before_allocation():
    # 37 physical qubits with 18 independent overlapping parity checks.
    checks = [(2 * j, 2 * j + 1, 36) for j in range(18)]
    circuit = circuit_for_checks(37, checks, layers=1)
    obs = ["".join("Z" if q in check else "I" for q in range(37)) for check in checks]
    result = circuit.estimate(obs, cfg(True, simulation_type=maestro.SimulationType.MatrixProductState))
    assert result["auto_reduced"]
    assert (result["qubits_before"], result["qubits_after"]) == (37, 18)
    assert result["method"] == maestro.SimulationType.MatrixProductState.value
    assert all(-1.00000001 <= value <= 1.00000001 for value in result["expectation_values"])


# Distance-7 triangular colour-code faces from qecbench.codes.color_code(7).
COLOUR_D7_CHECKS = [
    (0, 3, 2, 1), (1, 2, 5, 4), (6, 13, 12, 9), (7, 10, 15, 14),
    (18, 29, 28, 23), (19, 24, 31, 30), (24, 25, 32, 31), (26, 27, 34, 33),
    (28, 29, 36, 35), (2, 3, 6, 9, 8, 5), (4, 5, 8, 11, 10, 7),
    (8, 9, 12, 17, 16, 11), (10, 11, 16, 21, 20, 15),
    (12, 13, 18, 23, 22, 17), (14, 15, 20, 25, 24, 19),
    (16, 17, 22, 27, 26, 21), (20, 21, 26, 33, 32, 25), (22, 23, 28, 35, 34, 27),
]


@pytest.mark.parametrize("depth,energy", [
    (1, 1.0574878371864171),
    (3, -7.019864225140372),
    (10, -12.752135514922156),
])
def test_colour_code_against_reference_adapter(depth, energy):
    """References computed by the adapter's NumPy check_subspace_energy."""
    circuit = maestro.circuits.QuantumCircuit()
    for q in range(37):
        circuit.h(q)
    for layer in range(1, depth + 1):
        for check in COLOUR_D7_CHECKS:
            # Chain ladders exercise a different decomposition than the star
            # ladders used in the exhaustive small-register tests.
            pairs = list(itertools.pairwise(check))
            for control, target in pairs:
                circuit.cx(control, target)
            circuit.rz(check[-1], layer / depth)
            for control, target in reversed(pairs):
                circuit.cx(control, target)
        for q in range(37):
            circuit.rx(q, -(depth - layer + 1) / depth)
    observables = ["".join("Z" if q in check else "I" for q in range(37)) for check in COLOUR_D7_CHECKS]
    result = circuit.estimate(observables, cfg(True))
    assert result["auto_reduced"] and result["qubits_after"] == 18
    assert sum(result["expectation_values"]) == pytest.approx(energy, abs=1e-10)


def test_case_insensitive_and_short_observables():
    circuit = circuit_for_checks(4, [(0, 1), (1, 2, 3)])
    ref = circuit.estimate(["ZZII", "IZZI"], cfg(False))["expectation_values"]
    reduced = circuit.estimate(["zzii", "izzi"], cfg(True))
    assert reduced["auto_reduced"]
    assert reduced["expectation_values"] == pytest.approx(ref, abs=2e-12)

    # Shorter observable implicitly padded with identity
    short_reduced = circuit.estimate("zz", cfg(True))
    short_full = circuit.estimate("zz", cfg(False))
    assert short_reduced["auto_reduced"]
    assert short_reduced["expectation_values"] == pytest.approx(short_full["expectation_values"], abs=2e-12)


def test_out_of_span_observables_evaluate_to_zero():
    # Only checks (0, 1) and (1, 2, 3) are invariant; single-qubit Z observables
    # take the state into an orthogonal coset, so their expectation values must be 0.
    circuit = circuit_for_checks(4, [(0, 1), (1, 2, 3)])
    out_of_span = ["ZIII", "IZII", "IIZI", "IIIZ", "ZIZI"]
    res = circuit.estimate(out_of_span, cfg(True))
    assert res["auto_reduced"]
    assert res["expectation_values"] == [0.0] * len(out_of_span)


def test_repeated_estimations_preserve_circuit_integrity():
    circuit = circuit_for_checks(4, [(0, 1), (1, 2, 3)])

    res1 = circuit.estimate(["ZZII"], cfg(True))
    assert res1["auto_reduced"]

    res2 = circuit.estimate(["ZZII"], cfg(False))
    assert not res2["auto_reduced"]

    res3 = circuit.estimate(["ZZII"], cfg(True))
    assert res3["auto_reduced"]

    assert res1["expectation_values"] == pytest.approx(res2["expectation_values"], abs=2e-12)
    assert res1["expectation_values"] == pytest.approx(res3["expectation_values"], abs=2e-12)
    assert circuit.num_qubits == 4


def test_noisy_execution_bypasses_auto_reduction():
    circuit = circuit_for_checks(4, [(0, 1), (1, 2, 3)])
    nm = maestro.NoiseModel()
    nm.set_all_depolarizing(1, 0.01)

    res = circuit.noisy_estimate(["ZZII"], nm, cfg(True))
    assert "expectation_values" in res
    assert "ideal_expectation_values" in res

