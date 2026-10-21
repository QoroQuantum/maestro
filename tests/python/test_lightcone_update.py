"""Public-path regressions for lightcone_issues.md and the follow-up review."""

import itertools
import math

import maestro
import pytest


def cfg(**kw):
    return maestro.SimulatorConfig(**kw)


def full(**kw):
    return cfg(auto_reduce=False, enable_causal_cone_reduction=False, **kw)


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_open_cnot_frame_preserves_every_pauli_and_sign(dual, seed):
    c = maestro.circuits.QuantumCircuit()
    if not dual:
        for q in range(4):
            c.h(q)
    check, mix = (c.rx, c.rz) if dual else (c.rz, c.rx)
    check(0, 0.43 + seed / 10)
    for q in range(4):
        mix(q, 0.17 * (q + 1))
    c.cx(0, 1)
    c.cx(1, 2)
    c.cx(3, 1)
    mix(1, -0.31)
    check(0, -0.22)
    c.cx(2, 3)
    observables = ["".join(p) for p in itertools.product("IXYZ", repeat=4)]
    expected = c.estimate(observables, full())
    result = c.estimate(observables, cfg(enable_causal_cone_reduction=False))
    assert result["auto_reduced"] and result["qubits_after"] < 4
    assert result["expectation_values"] == pytest.approx(
        expected["expectation_values"], abs=3e-12
    )


def test_known_identity_and_orthogonal_sector_skip_simulator():
    c = maestro.circuits.QuantumCircuit()
    for q in range(3):
        c.h(q)
    c.rz(0, 0.31)
    r = c.estimate(["III", "IZI", "IIY"])
    assert r["expectation_values"] == [1, 0, 0]
    assert r["qubits_after"] == r["clusters_count"] == 0
    assert r["cluster_methods"] == []


@pytest.mark.parametrize(
    "method",
    [
        maestro.SimulationType.Statevector,
        maestro.SimulationType.DensityMatrix,
        maestro.SimulationType.MatrixProductOperator,
    ],
)
def test_reset_reuse_terminates_cone_with_explicit_trajectory_policy(method):
    c = maestro.circuits.QuantumCircuit()
    c.ry(0, 0.8)
    c.h(2)
    c.cx(2, 0)
    c.reset(0)
    c.ry(0, 0.3)
    r = c.estimate(
        ["ZII", "XII", "YII"],
        cfg(
            auto_reduce=False, simulation_type=method, allow_reset_marginalization=True
        ),
    )
    expected = c.estimate(["ZII", "XII", "YII"], full(simulation_type=method))
    assert r["qubits_after"] == 1
    assert r["expectation_values"] == pytest.approx(
        expected["expectation_values"], abs=1e-12
    )


def test_reset_marginalization_preserves_ensemble_mean_not_trajectory():
    c = maestro.circuits.QuantumCircuit()
    c.h(0)
    c.cx(0, 1)
    c.reset(1)
    r = c.estimate("ZI", cfg(allow_reset_marginalization=True))
    assert r["qubits_after"] == 1 and abs(r["expectation_values"][0]) < 1e-12
    strict = [c.estimate("ZI", cfg(seed=s)) for s in range(96)]
    assert all(
        r["qubits_after"] == 2 and abs(r["expectation_values"][0]) > 0.99
        for r in strict
    )
    assert abs(sum(r["expectation_values"][0] for r in strict) / len(strict)) < 0.25
    c.x(2)  # Allow structural sampling to remove a spectator wire.
    c.measure([(0, 0)])
    strict_sample = c.execute(cfg(seed=72, enable_sampling_reduction=True), 80)
    assert strict_sample["largest_component_qubits"] == 3
    sampled = c.execute(
        cfg(seed=72, enable_sampling_reduction=True, allow_reset_marginalization=True),
        8000,
    )
    assert (
        sampled["largest_component_qubits"] == 2
    )  # structural cone retains the entangler
    assert set(sampled["counts"]) == {"000", "100"}
    assert abs(sampled["counts"]["000"] / 8000 - 0.5) < 0.04


def test_t1_monte_carlo_uses_tight_cones_when_marginalization_is_enabled():
    c = maestro.circuits.QuantumCircuit()
    for q in range(12):
        c.h(q)
    for q in range(11):
        c.cx(q, q + 1)
    nm = maestro.NoiseModel()
    nm.set_all_t1(12, 0.05)
    r = c.noisy_estimate_montecarlo(
        ["Z" + "I" * 11],
        nm,
        512,
        cfg(seed=9, allow_reset_marginalization=True),
        noise_seed=17,
    )
    assert r["qubits_after"] == 1
    assert r["cone_cache_hits"] == 511
    exact = c.noisy_estimate_montecarlo(
        ["Z" + "I" * 11],
        nm,
        1,
        cfg(simulation_type=maestro.SimulationType.DensityMatrix),
        noise_seed=17,
    )
    assert abs(r["expectation_values"][0] - exact["expectation_values"][0]) < 0.06


@pytest.mark.parametrize("gate", ["cy", "ch", "crx", "cry", "cp", "crz"])
def test_delay_and_control_z_commutation_reduce_width(gate):
    c = maestro.circuits.QuantumCircuit()
    c.ry(0, 0.3)
    c.ry(1, 0.7)
    (
        getattr(c, gate)(0, 1, 0.4)
        if gate.startswith("cr") or gate == "cp"
        else getattr(c, gate)(0, 1)
    )
    c.delay(0, 2.0)
    r = c.estimate("ZI", cfg(auto_reduce=False))
    assert r["qubits_after"] == 1
    assert r["expectation_values"] == pytest.approx([math.cos(0.3)], abs=1e-12)


def test_sampling_key_width_matches_full_path_with_sparse_measurements():
    c = maestro.circuits.QuantumCircuit()
    c.x(4)
    c.rz(0, 0.2)
    c.measure([(4, 0), (0, 2)])
    expected = c.execute(cfg(seed=13), 50)
    reduced = c.execute(cfg(seed=13, enable_sampling_reduction=True), 50)
    assert expected["counts"] == reduced["counts"] == {"10000": 50}
    c = maestro.circuits.QuantumCircuit()
    c.x(1)
    c.measure([(1, 7), (0, 2)])
    assert c.execute(cfg(enable_sampling_reduction=True), 50)["counts"] == {
        "00000001": 50
    }


def test_high_entropy_sampling_preserves_component_correlations_and_independence():
    c = maestro.circuits.QuantumCircuit()
    width, blocks, shots = 8, 3, 20000
    for block in range(blocks):
        start = block * width
        for q in range(start, start + width - 1):
            c.h(q)
        for q in range(start, start + width - 1):
            c.cx(q, q + 1)
    c.measure([(q, q) for q in range(width * blocks)])
    config = cfg(seed=73, enable_sampling_reduction=True)
    result = c.execute(config, shots)
    counts = result["counts"]
    assert result["sampling_components"] == blocks
    assert result["largest_component_qubits"] == width
    assert sum(counts.values()) == shots and len(counts) > 0.9 * shots
    assert counts == c.execute(config, shots)["counts"]
    patterns = [0] * (1 << blocks)
    for outcome, count in counts.items():
        assert len(outcome) == width * blocks and count > 0
        for block in range(blocks):
            # Each block has seven uniform bits, with the last bit copied from
            # the previous one. That correlation must survive the sparse join.
            assert outcome[block * width + 6] == outcome[block * width + 7]
        pattern = sum(int(outcome[block * width]) << block for block in range(blocks))
        patterns[pattern] += count
    assert [count / shots for count in patterns] == pytest.approx(
        [1 / 8] * 8, abs=0.015
    )


def test_duplicate_measurement_destinations_still_use_full_execution():
    c = maestro.circuits.QuantumCircuit()
    c.x(1)
    c.measure([(1, 0)])
    c.measure([(0, 0)])
    result = c.execute(cfg(seed=8, enable_sampling_reduction=True), 50)
    assert result["sampling_components"] == 1
    assert result["counts"] == c.execute(cfg(seed=8), 50)["counts"]


def test_marginalization_does_not_enable_classical_feedback_reduction():
    qasm = "OPENQASM 2.0; qreg q[2]; creg c[2]; h q[0]; measure q[0]->c[0]; if(c==1) x q[1]; reset q[0]; measure q[1]->c[1];"
    r = maestro.simple_execute(
        qasm,
        cfg(seed=51, enable_sampling_reduction=True, allow_reset_marginalization=True),
        1000,
    )
    assert r["sampling_components"] == 1 and set(r["counts"]) <= {"00", "11"}


def test_deduplication_keeps_distinct_parameters_separate():
    c = maestro.circuits.QuantumCircuit()
    c.ry(0, 0.3)
    c.ry(1, 0.3 + 1e-10)
    r = c.estimate(["ZI", "IZ"], cfg(auto_reduce=False))
    assert r["clusters_count"] == 2
    assert r["expectation_values"] == pytest.approx(
        [math.cos(0.3), math.cos(0.3 + 1e-10)], abs=1e-13
    )
