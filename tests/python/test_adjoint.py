"""Adjoint expectation values and derivatives against the simulator path."""

import math
import random

import maestro
import numpy as np
import pytest
from maestro.circuits import QuantumCircuit

# (builder name, number of qubits, number of differentiable parameters)
PARAMETRIC = [("p", 1, 1), ("rx", 1, 1), ("ry", 1, 1), ("rz", 1, 1), ("u", 1, 3),
              ("cp", 2, 1), ("crx", 2, 1), ("cry", 2, 1), ("crz", 2, 1), ("cu", 2, 4)]
FIXED = [("x", 1), ("y", 1), ("z", 1), ("h", 1), ("s", 1), ("sdg", 1), ("t", 1), ("tdg", 1),
         ("sx", 1), ("sxdg", 1), ("k", 1), ("cx", 2), ("cy", 2), ("cz", 2), ("ch", 2),
         ("csx", 2), ("csxdg", 2), ("swap", 2), ("ccx", 3), ("cswap", 3)]


def random_layout(n, depth, seed):
    rng = random.Random(seed)
    layout = []
    for _ in range(depth):
        for name, nq, npar in PARAMETRIC:
            layout.append((name, rng.sample(range(n), nq), npar))
        for name, nq in FIXED:
            layout.append((name, rng.sample(range(n), nq), 0))
    rng.shuffle(layout)
    return layout


def build(n, layout, params):
    qc = QuantumCircuit()
    for q in range(n):
        qc.z(q)
        qc.z(q)
    i = 0
    for name, qubits, npar in layout:
        getattr(qc, name)(*qubits, *params[i:i + npar])
        i += npar
    return qc


def reference(n, layout, params, observables):
    """Central finite differences of simple_estimate."""

    def values(p):
        qc = build(n, layout, p)
        out = []
        for terms in observables:
            paulis = [s for _, s in terms]
            ev = maestro.simple_estimate(qc, paulis)["expectation_values"]
            out.append(sum(c * v for (c, _), v in zip(terms, ev)))
        return np.array(out)

    h = 1e-6
    jac = np.zeros((len(observables), len(params)))
    for k in range(len(params)):
        up, down = params.copy(), params.copy()
        up[k] += h
        down[k] -= h
        jac[:, k] = (values(up) - values(down)) / (2 * h)
    return values(params), jac


OBSERVABLES = [
    [(1.0, "ZIII")],
    [(0.7, "XIZI"), (-1.3, "IYII"), (0.25, "IIII")],
    [(2.0, "YXZX")],
]


@pytest.mark.parametrize("cone", [True, False])
def test_jacobian_matches_finite_differences(cone):
    n = 4
    layout = random_layout(n, 2, seed=3)
    num = sum(npar for *_, npar in layout)
    params = np.random.default_rng(0).uniform(-math.pi, math.pi, num)
    program = maestro.AdjointProgram(build(n, layout, params))
    assert program.num_params == num
    np.testing.assert_allclose(program.parameters, params)

    config = maestro.SimulatorConfig()
    config.enable_causal_cone_reduction = cone
    out = program.jacobian(OBSERVABLES, config=config)
    ref_vals, ref_jac = reference(n, layout, params, OBSERVABLES)
    np.testing.assert_allclose(out["expectation_values"][0], ref_vals, atol=1e-10)
    np.testing.assert_allclose(out["jacobian"][0], ref_jac, atol=1e-7)


def test_rows_vjp_and_expectations_agree():
    n = 4
    layout = random_layout(n, 1, seed=5)
    num = sum(npar for *_, npar in layout)
    rows = np.random.default_rng(1).uniform(-math.pi, math.pi, (5, num))
    program = maestro.AdjointProgram(build(n, layout, rows[0]))

    out = program.jacobian(OBSERVABLES, rows)
    assert out["jacobian"].shape == (5, len(OBSERVABLES), num)
    for b in (0, 3):
        ref_vals, ref_jac = reference(n, layout, rows[b], OBSERVABLES)
        np.testing.assert_allclose(out["expectation_values"][b], ref_vals, atol=1e-10)
        np.testing.assert_allclose(out["jacobian"][b], ref_jac, atol=1e-7)

    fwd = program.expectation_values(OBSERVABLES, rows)["expectation_values"]
    np.testing.assert_allclose(fwd, out["expectation_values"], atol=1e-12)

    dy = np.random.default_rng(2).normal(size=(5, len(OBSERVABLES)))
    vjp = program.vjp(OBSERVABLES, dy, rows)
    np.testing.assert_allclose(vjp["vjp"], np.einsum("bo,bok->bk", dy, out["jacobian"]), atol=1e-12)


def test_gates_outside_the_cone_have_zero_gradient():
    qc = QuantumCircuit()
    qc.rx(0, 0.3)
    qc.ry(3, 0.4)  # never reaches qubit 0
    qc.cx(1, 0)
    out = maestro.adjoint_jacobian(qc, ["ZIII"])
    np.testing.assert_allclose(out["jacobian"][0, 0], [-math.sin(0.3), 0.0], atol=1e-12)


def test_large_register_uses_parallel_kernels():
    n = 15
    layout = [("ry", [q], 1) for q in range(n)] + [("cx", [q, q + 1], 0) for q in range(n - 1)]
    params = np.linspace(0.1, 1.5, n)
    program = maestro.AdjointProgram(build(n, layout, params))
    obs = [[(1.0, "I" * (n - 1) + "Z")]]
    out = program.jacobian(obs)
    _, ref_jac = reference(n, layout, params, obs)
    np.testing.assert_allclose(out["jacobian"][0], ref_jac, atol=1e-7)


def test_rejects_non_unitary_circuits_and_bad_shapes():
    qc = QuantumCircuit()
    qc.rx(0, 0.1)
    qc.measure_all()
    with pytest.raises(ValueError):
        maestro.AdjointProgram(qc)

    qc = QuantumCircuit()
    qc.rx(0, 0.1)
    program = maestro.AdjointProgram(qc)
    with pytest.raises(ValueError):
        program.jacobian(["Z"], np.zeros((2, 3)))
    with pytest.raises(ValueError):
        program.jacobian(["Q"])
    config = maestro.SimulatorConfig()
    config.simulation_type = maestro.SimulationType.MatrixProductState
    with pytest.raises(ValueError):
        program.jacobian(["Z"], config=config)


def marginal(qc, qubits):
    """Probabilities of `qubits` from the full distribution (first qubit lowest bit)."""
    full = np.array(maestro.get_probabilities(qc))
    out = np.zeros(2 ** len(qubits))
    for i, p in enumerate(full):
        out[sum(((i >> q) & 1) << j for j, q in enumerate(qubits))] += p
    return out


@pytest.mark.parametrize("cone", [True, False])
def test_probabilities_values_and_jacobian(cone):
    n = 4
    layout = random_layout(n, 1, seed=11)
    num = sum(npar for *_, npar in layout)
    params = np.random.default_rng(4).uniform(-math.pi, math.pi, num)
    program = maestro.AdjointProgram(build(n, layout, params))
    readouts = [[(0.5, "ZIXI")], maestro.Probabilities([2, 0]), maestro.Probabilities([3])]

    config = maestro.SimulatorConfig()
    config.enable_causal_cone_reduction = cone
    out = program.jacobian(readouts, config=config)
    assert out["expectation_values"].shape == (1, 1 + 4 + 2)
    assert out["jacobian"].shape == (1, 7, num)

    def values(p):
        qc = build(n, layout, p)
        ev = maestro.simple_estimate(qc, ["ZIXI"])["expectation_values"][0]
        return np.concatenate([[0.5 * ev], marginal(qc, [2, 0]), marginal(qc, [3])])

    np.testing.assert_allclose(out["expectation_values"][0], values(params), atol=1e-10)
    h = 1e-6
    ref = np.zeros((7, num))
    for k in range(num):
        up, down = params.copy(), params.copy()
        up[k] += h
        down[k] -= h
        ref[:, k] = (values(up) - values(down)) / (2 * h)
    np.testing.assert_allclose(out["jacobian"][0], ref, atol=1e-7)

    fwd = program.expectation_values(readouts, config=config)["expectation_values"]
    np.testing.assert_allclose(fwd, out["expectation_values"], atol=1e-12)


def test_probabilities_vjp_matches_jacobian():
    n = 4
    layout = random_layout(n, 1, seed=12)
    num = sum(npar for *_, npar in layout)
    rows = np.random.default_rng(5).uniform(-math.pi, math.pi, (3, num))
    program = maestro.AdjointProgram(build(n, layout, rows[0]))
    readouts = [maestro.Probabilities([0, 1, 3]), [(1.0, "IYII")]]
    jac = program.jacobian(readouts, rows)["jacobian"]
    dy = np.random.default_rng(6).normal(size=(3, 9))
    out = program.vjp(readouts, dy, rows)
    np.testing.assert_allclose(out["vjp"], np.einsum("bc,bck->bk", dy, jac), atol=1e-12)
    np.testing.assert_allclose(out["expectation_values"][:, :8].sum(axis=1), 1.0, atol=1e-12)


def test_probabilities_rejects_bad_qubits():
    qc = QuantumCircuit()
    qc.rx(0, 0.1)
    program = maestro.AdjointProgram(qc)
    with pytest.raises(ValueError):
        program.jacobian([maestro.Probabilities([0, 0])])
    with pytest.raises(ValueError):
        program.jacobian([maestro.Probabilities([])])
    assert repr(maestro.Probabilities([1, 0])) == "Probabilities([1, 0])"
