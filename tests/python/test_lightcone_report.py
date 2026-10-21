"""Regression coverage for the lightcone review and its exact enhancements."""

import copy
import itertools
import math
import random

import maestro
import pytest


def config(**kw):
    return maestro.SimulatorConfig(**kw)


def full_config(**kw):
    return config(auto_reduce=False, enable_causal_cone_reduction=False, **kw)


def pair(c, a=0, b=1):
    c.h(a)
    c.h(b)
    c.cx(a, b)
    c.rz(b, 0.7)
    c.cx(a, b)
    c.rx(a, 0.2)
    c.rx(b, -0.3)


@pytest.mark.parametrize("padding", [0, 1, 70])
@pytest.mark.parametrize("zero_check", [False, True])
def test_padding_and_zero_checks_do_not_inflate_parity_rank(padding, zero_check):
    c = maestro.circuits.QuantumCircuit()
    pair(c)
    if zero_check:
        c.rz(0, 0.0)
    r = c.estimate("ZZ" + "I" * padding)
    assert r["auto_reduced"] and r["qubits_after"] == 1
    assert r["expectation_values"] == pytest.approx(
        [math.sin(0.7) * math.sin(-0.1)], abs=1e-12
    )


@pytest.mark.parametrize("width", [48, 70])
def test_parity_retry_below_global_rank_and_width_caps(width):
    c = maestro.circuits.QuantumCircuit()
    obs = []
    for a in range(0, width, 2):
        pair(c, a, a + 1)
        obs.append("I" * a + "ZZ" + "I" * (width - a - 2))
    r = c.estimate(obs)
    assert r["auto_reduced"] and r["qubits_after"] == 1
    assert r["clusters_count"] == 1  # identical local parity circuits share one run
    assert r["expectation_values"] == pytest.approx(
        [math.sin(0.7) * math.sin(-0.1)] * (width // 2), abs=1e-12
    )


def test_spectator_gate_removed_before_local_parity_retry():
    c = maestro.circuits.QuantumCircuit()
    pair(c)
    c.ry(2, 0.4)
    r = c.estimate("ZZI")
    assert r["auto_reduced"] and r["qubits_after"] == 1


@pytest.mark.parametrize(
    "method",
    [
        maestro.SimulationType.DensityMatrix,
        maestro.SimulationType.MatrixProductOperator,
    ],
)
def test_exact_noise_spectators_and_metadata(method):
    c = maestro.circuits.QuantumCircuit()
    c.x(0)
    c.x(5)
    nm = maestro.NoiseModel()
    nm.set_depolarizing(5, 0.1)
    r = c.noisy_estimate_montecarlo(
        ["ZIIIII"], nm, 2, config(simulation_type=method), noise_seed=17
    )
    assert r["qubits_after"] == 1 and r["clusters_count"] == 1
    assert r["cluster_methods"] == [method.value]
    assert r["expectation_values"] == pytest.approx([-1.0], abs=1e-12)
    assert r["reduction_time"] >= 0 and r["mean_cluster_qubits"] == 1


def test_disconnected_reset_can_be_pruned_but_entangled_reset_cannot():
    c = maestro.circuits.QuantumCircuit()
    c.x(0)
    c.h(30)
    c.reset(30)
    r = c.estimate("Z" + "I" * 30, config(seed=7))
    assert r["qubits_after"] == 1 and r["expectation_values"] == [-1.0]
    bell = maestro.circuits.QuantumCircuit()
    bell.h(0)
    bell.cx(0, 1)
    bell.reset(1)
    for seed in range(5):
        r = bell.estimate("ZI", config(seed=seed))
        full = bell.estimate("ZI", full_config(seed=seed))
        assert r["qubits_after"] == 2
        assert r["expectation_values"] == full["expectation_values"]


@pytest.mark.parametrize("reduce", [False, True])
def test_empty_observables_do_not_allocate_a_large_register(reduce):
    c = maestro.circuits.QuantumCircuit()
    c.x(100)
    c.reset(100)
    r = c.estimate([], config(enable_causal_cone_reduction=reduce))
    assert r["expectation_values"] == r["cluster_methods"] == []
    assert r["clusters_count"] == r["qubits_after"] == r["mean_cluster_qubits"] == 0


def test_dense_default_budget_and_tiny_disjoint_batching():
    c = maestro.circuits.QuantumCircuit()
    n = 80
    for q in range(n):
        c.ry(q, 0.1 + q / 100)
        c.rz(q, 0.2 + q / 100)
    for start in (0, 1):
        for q in range(start, n - 1, 2):
            c.cx(q, q + 1)
    obs = ["I" * q + "Y" + "I" * (n - q - 1) for q in range(1, n - 2, 2)]
    r = c.estimate(obs, config(auto_reduce=False))
    assert 1 < r["clusters_count"] and r["largest_cluster_qubits"] <= 20
    expected = c.estimate(obs, config(auto_reduce=False, max_cluster_qubits=5))
    assert r["expectation_values"] == pytest.approx(
        expected["expectation_values"], abs=1e-12
    )
    disjoint = maestro.circuits.QuantumCircuit()
    for q in range(12):
        disjoint.ry(q, 0.2 + q / 10)
    obs = ["I" * q + "Z" + "I" * (11 - q) for q in range(12)]
    batched = disjoint.estimate(obs, config(min_cluster_qubits=4, max_cluster_qubits=4))
    assert batched["clusters_count"] == 3 and batched["largest_cluster_qubits"] == 4
    assert batched["expectation_values"] == pytest.approx(
        [math.cos(0.2 + q / 10) for q in range(12)], abs=1e-12
    )
    cfg = config(min_cluster_qubits=4)
    assert copy.copy(cfg).min_cluster_qubits == 4
    with pytest.raises(ValueError):
        config(min_cluster_qubits=5, max_cluster_qubits=4)


@pytest.mark.parametrize(
    "gate", ["z", "s", "sdg", "t", "tdg", "p", "x", "sx", "sxdg", "cz"]
)
@pytest.mark.parametrize("dual", [False, True])
def test_extended_parity_gates_all_paulis(gate, dual):
    c = maestro.circuits.QuantumCircuit()
    if not dual:
        for q in range(4):
            c.h(q)
    # Two independent checks; the last two qubits provide unused symmetry.
    rotation = c.rx if dual else c.rz
    for q in (0, 1):
        rotation(q, 0.37 + q / 10)
    if gate == "cz":
        c.cz(0, 1)
    elif gate == "p":
        c.p(0, -0.23)
    else:
        getattr(c, gate)(0)
    for q in range(4):
        (c.rz if dual else c.rx)(q, -0.4)
    obs = ["".join(p) for p in itertools.product("IXYZ", repeat=4)]
    reduced = c.estimate(obs, config(enable_causal_cone_reduction=False))
    full = c.estimate(obs, full_config())
    assert reduced["auto_reduced"] and reduced["qubits_after"] == 2
    assert reduced["expectation_values"] == pytest.approx(
        full["expectation_values"], abs=2e-12
    )


@pytest.mark.parametrize("dual", [False, True])
def test_mixers_inside_nonidentity_cnot_frames(dual):
    c = maestro.circuits.QuantumCircuit()
    if not dual:
        for q in range(4):
            c.h(q)
    cx = lambda a, b: c.cx(b, a) if dual else c.cx(a, b)
    cx(0, 1)
    cx(1, 2)
    (c.rx if dual else c.rz)(2, 0.7)
    for q in (0, 1, 2):
        (c.rz if dual else c.rx)(q, -0.2 - q / 10)
    cx(1, 2)
    cx(0, 1)
    obs = ["".join(p) for p in itertools.product("IXYZ", repeat=4)]
    r = c.estimate(obs, config(enable_causal_cone_reduction=False))
    assert r["auto_reduced"] and r["qubits_after"] == 1
    assert r["expectation_values"] == pytest.approx(
        c.estimate(obs, full_config())["expectation_values"], abs=2e-12
    )


@pytest.mark.parametrize("seed", range(5))
def test_commutation_and_initial_state_pruning_against_all_paulis(seed):
    rng = random.Random(seed)
    c = maestro.circuits.QuantumCircuit()
    # Initially inactive controls, followed by non-Clifford gates and both
    # control/target support: checking all Paulis catches unsafe local shortcuts.
    c.cx(0, 1)
    c.cz(1, 2)
    c.ccx(0, 1, 2)
    for _ in range(4):
        for q in range(3):
            c.ry(q, rng.uniform(-2, 2))
        c.cx(0, 1)
        c.cz(1, 2)
        c.ccx(0, 1, 2)
        c.rx(2, rng.uniform(-2, 2))
        c.rz(0, rng.uniform(-2, 2))
    obs = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]
    expected = c.estimate(obs, full_config())["expectation_values"]
    # Evaluate separately to exercise single-observable axis information.
    actual = [
        c.estimate(p, config(auto_reduce=False))["expectation_values"][0] for p in obs
    ]
    assert actual == pytest.approx(expected, abs=2e-12)


def test_final_diagonal_entanglers_do_not_expand_z_cone():
    c = maestro.circuits.QuantumCircuit()
    c.ry(0, 0.3)
    for q in range(1, 80):
        c.ry(q, 0.7)
        c.cz(0, q)
    r = c.estimate("Z" + "I" * 79, config(auto_reduce=False))
    assert r["qubits_after"] == 1
    assert r["expectation_values"] == pytest.approx([math.cos(0.3)], abs=1e-12)
    obs = ["I" * q + "Z" + "I" * (79 - q) for q in range(80)]
    batched = c.estimate(obs, config(auto_reduce=False))
    assert batched["clusters_count"] == 2 and batched["qubits_after"] == 1


@pytest.mark.parametrize(
    "gate", ["cy", "cp", "crx", "cry", "crz", "ch", "csx", "csxdg", "ccx", "cswap"]
)
def test_controlled_gates_use_whole_operator_commutation(gate):
    def apply(c):
        args = (
            (0, 1, 2)
            if gate in ("ccx", "cswap")
            else (0, 1, 0.37) if gate in ("cp", "crx", "cry", "crz") else (0, 1)
        )
        getattr(c, gate)(*args)

    c = maestro.circuits.QuantumCircuit()
    c.h(1)
    c.ry(2, 0.41)
    apply(c)  # known-zero control, but targets are not computational zero
    c.ry(0, 0.77)
    c.rx(1, -0.2)
    apply(c)  # the control is now active
    obs = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]
    expected = c.estimate(obs, full_config())["expectation_values"]
    actual = [
        c.estimate(p, config(auto_reduce=False))["expectation_values"][0] for p in obs
    ]
    assert actual == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("gate", ["cy", "crz", "ccx", "cswap"])
def test_bounded_channel_propagation_handles_control_order_and_three_qubit_gates(gate):
    c = maestro.circuits.QuantumCircuit()
    c.ry(0, 0.4)
    c.h(1)
    c.rx(2, -0.3)
    args = (
        (2, 0, 1)
        if gate in ("ccx", "cswap")
        else (2, 0, 0.37) if gate == "crz" else (2, 0)
    )
    getattr(c, gate)(*args)
    nm = maestro.NoiseModel()
    nm.set_all_depolarizing(3, 0.07)
    obs = ["".join(p) for p in itertools.product("IXYZ", repeat=3)]
    cfg = config(simulation_type=maestro.SimulationType.DensityMatrix)
    expected = c.noisy_estimate_montecarlo(obs, nm, 1, cfg, noise_seed=13)
    cfg.observable_error_tolerance = 1e-5
    actual = c.noisy_estimate_montecarlo(obs, nm, 1, cfg, noise_seed=13)
    assert actual["bounded_observables"] == len(obs)
    for a, b, error in zip(
        actual["expectation_values"],
        expected["expectation_values"],
        actual["expectation_error_bounds"],
    ):
        assert abs(a - b) <= error + 1e-12


@pytest.mark.parametrize(
    "method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix]
)
def test_noise_cache_preserves_seeded_values_and_full_injection(method):
    c = maestro.circuits.QuantumCircuit()
    for q in range(5):
        c.ry(q, 0.2 + q / 5)
    c.cx(0, 1)
    c.rx(1, 0.3)
    c.cx(2, 3)
    nm = maestro.NoiseModel()
    nm.set_all_depolarizing(5, 0.15)
    obs = ["IZIII", "IIIZI", "IIIII"]
    r = c.noisy_estimate_montecarlo(
        obs, nm, 6, config(simulation_type=method), noise_seed=987
    )
    full = c.noisy_estimate_montecarlo(
        obs, nm, 6, full_config(simulation_type=method), noise_seed=987
    )
    assert r["cone_cache_hits"] == 5 and r["clusters_count"] == 2
    assert r["expectation_values"] == pytest.approx(
        full["expectation_values"], abs=2e-12
    )


@pytest.mark.parametrize("noise", ["depolarizing", "dephasing", "amplitude"])
def test_bounded_noisy_estimates_report_valid_errors(noise):
    c = maestro.circuits.QuantumCircuit()
    for _ in range(16):
        c.ry(0, 0.41)
        c.rx(1, 0.17)
        c.cx(1, 0)  # asymmetric control order exercises gate matrix conventions
        c.rz(1, -0.23)
    nm = maestro.NoiseModel()
    if noise == "amplitude":
        nm.set_generalized_amplitude_damping(0, 0.25, 0.15)
    else:
        getattr(nm, "set_all_" + noise)(2, 0.35)
    obs = ["".join(p) for p in itertools.product("IXYZ", repeat=2)]
    cfg = config(simulation_type=maestro.SimulationType.DensityMatrix)
    exact = c.noisy_estimate_montecarlo(obs, nm, 1, cfg, noise_seed=2)
    cfg.observable_error_tolerance = 1e-3
    bounded = c.noisy_estimate_montecarlo(obs, nm, 1, cfg, noise_seed=2)
    assert bounded["bounded_observables"] == len(obs)
    assert bounded["clusters_count"] == 0
    for a, b, error in zip(
        bounded["expectation_values"],
        exact["expectation_values"],
        bounded["expectation_error_bounds"],
    ):
        assert abs(a - b) <= error + 1e-12
        assert 0 <= error <= cfg.observable_error_tolerance


def test_dephasing_does_not_damp_z_and_pure_trajectories_are_not_truncated():
    c = maestro.circuits.QuantumCircuit()
    for _ in range(40):
        c.rz(0, 0.4)
    nm = maestro.NoiseModel()
    nm.set_dephasing(0, 0.5)
    for method in [
        maestro.SimulationType.DensityMatrix,
        maestro.SimulationType.MatrixProductOperator,
        maestro.SimulationType.Statevector,
    ]:
        r = c.noisy_estimate_montecarlo(
            ["Z"],
            nm,
            2,
            config(simulation_type=method, observable_error_tolerance=0.1),
            noise_seed=9,
        )
        assert r["expectation_values"] == pytest.approx([1], abs=1e-12)
        if method == maestro.SimulationType.Statevector:
            assert r["bounded_observables"] == 0
            assert r["expectation_error_bounds"] == [0]
    for invalid in [0, -0.1, 1.01, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            config(observable_error_tolerance=invalid)


def test_analytical_noise_overloads_preserve_reduction_diagnostics():
    c = maestro.circuits.QuantumCircuit()
    c.x(0)
    c.h(5)
    nm = maestro.NoiseModel()
    nm.set_depolarizing(0, 0.1)
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[6]; x q[0]; h q[5];'
    for r in [
        c.noisy_estimate(["ZIIIII"], nm),
        maestro.noisy_estimate(c, ["ZIIIII"], nm),
        maestro.noisy_estimate(qasm, ["ZIIIII"], nm),
    ]:
        assert r["clusters_count"] == r["qubits_after"] == r["mean_cluster_qubits"] == 1
        assert r["expectation_values"] == pytest.approx([-(1 - 4 * 0.1 / 3)], abs=1e-12)
        assert r["ideal_expectation_values"] == [-1.0]


def test_disjoint_sampling_preserves_independence_and_seed_reproducibility():
    c = maestro.circuits.QuantumCircuit()
    c.h(0)
    c.h(70)
    c.measure([(0, 2)])
    c.measure([(70, 0)])
    cfg = config(seed=124, enable_sampling_reduction=True)
    r = c.execute(cfg, 8000)
    assert r["sampling_components"] == 2 and r["largest_component_qubits"] == 1
    assert r["counts"] == c.execute(cfg, 8000)["counts"]
    assert set(r["counts"]) == {s + "0" * 68 for s in ["000", "001", "100", "101"]}
    for count in r["counts"].values():
        assert abs(count / 8000 - 0.25) < 0.04


def test_sampling_keeps_entangled_components_and_ignores_estimation_budget():
    c = maestro.circuits.QuantumCircuit()
    c.h(0)
    c.cx(0, 1)
    c.cx(1, 2)
    c.h(5)
    for q in [0, 1, 2, 5]:
        c.measure([(q, q)])
    r = c.execute(
        config(seed=77, enable_sampling_reduction=True, max_cluster_qubits=1), 4000
    )
    assert r["sampling_components"] == 2 and r["largest_component_qubits"] == 3
    assert sum(r["counts"].values()) == 4000
    assert len(r["counts"]) == 4
    assert all(
        bits[0] == bits[1] == bits[2] and bits[3:5] == "00" for bits in r["counts"]
    )


def test_sampling_dynamic_circuit_falls_back_and_readout_mapping_survives():
    source = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; measure q[0]->c[0]; if(c==1) x q[1]; measure q[1]->c[1];"
    cfg = config(seed=88, enable_sampling_reduction=True)
    r = maestro.simple_execute(source, cfg, 300)
    assert r["sampling_components"] == 1 and set(r["counts"]) <= {"00", "11"}
    c = maestro.circuits.QuantumCircuit()
    c.x(4)
    c.rz(0, 0.2)
    c.measure([(4, 0)])
    c.measure([(0, 2)])
    nm = maestro.NoiseModel()
    nm.set_readout_error(4, 0.0, 1.0)
    nm.set_readout_error(0, 1.0, 0.0)
    noisy = c.noisy_execute(nm, cfg, 100, 1, noise_seed=7)
    assert noisy["counts"] == {"00100": 100}
    assert noisy["sampling_components"] == 2 and noisy["largest_component_qubits"] == 1


@pytest.mark.parametrize(
    "method",
    [
        maestro.SimulationType.DensityMatrix,
        maestro.SimulationType.MatrixProductOperator,
    ],
)
def test_ensemble_reset_keeps_reduced_density_expectation(method):
    c = maestro.circuits.QuantumCircuit()
    c.h(0)
    c.cx(0, 1)
    c.reset(1)
    r = c.estimate("ZI", config(simulation_type=method))
    expected = c.estimate("ZI", full_config(simulation_type=method))
    assert r["qubits_after"] == 1
    assert r["expectation_values"] == pytest.approx(
        expected["expectation_values"], abs=1e-12
    )
    assert r["expectation_values"] == pytest.approx([0], abs=1e-12)


def test_empty_noisy_estimate_returns_without_a_simulator():
    c = maestro.circuits.QuantumCircuit()
    c.x(100)
    nm = maestro.NoiseModel()
    nm.set_depolarizing(100, 0.1)
    r = c.noisy_estimate_montecarlo([], nm, 100, full_config())
    assert r["clusters_count"] == r["qubits_after"] == 0
    assert (
        r["expectation_values"]
        == r["ideal_expectation_values"]
        == r["expectation_error_bounds"]
        == []
    )
