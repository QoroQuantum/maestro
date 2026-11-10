"""Count-policy dispatch, same-version seed replay and ordered shot records."""

import copy
import math
import pickle
from collections import Counter

import numpy as np
import pytest

import maestro
from maestro.circuits import QuantumCircuit


def _config(policy="auto", seed=None, method=maestro.SimulationType.Stabilizer):
    return maestro.SimulatorConfig(
        simulation_type=method, seed=seed, max_simulators=1,
        max_bond_dimension=16, counts_policy=policy,
    )


def _bell():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    return circuit


def _noisy_bell():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.pauli_channel([1], [0.8, 0.2, 0.0, 0.0])
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.1)
    return circuit, model


def _assert_histogram(result, shots, width):
    assert "counts" in result and "samples" not in result
    counts = result["counts"]
    assert sum(counts.values()) == shots
    assert all(len(bits) == width and set(bits) <= {"0", "1"} for bits in counts)
    assert all(isinstance(count, int) and count >= 0 for count in counts.values())
    return counts


def test_counts_policy_defaults_validation_rollback_and_pickle():
    config = maestro.SimulatorConfig()
    assert config.counts_policy == "auto"
    config.counts_policy = "legacy"
    for invalid in ["", "invalid", "FRAME_V1"]:
        with pytest.raises(ValueError, match="counts_policy"):
            config.counts_policy = invalid
        assert config.counts_policy == "legacy"
        with pytest.raises(ValueError, match="counts_policy"):
            maestro.SimulatorConfig(counts_policy=invalid)
    config.counts_policy = "frame_v1"
    config.seed = 0
    for restored in [copy.copy(config), pickle.loads(pickle.dumps(config))]:
        assert restored.counts_policy == "frame_v1"
        assert restored.seed == 0


def test_unseeded_plain_counts_use_frames_automatically():
    result = _bell().execute(config=_config(), shots=1024)
    counts = _assert_histogram(result, 1024, 2)
    assert result["sampling_strategy"] == "reference_frame_v1"
    assert set(counts) == {"00", "11"}
    assert 400 < counts["11"] < 624


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.PathIntegral, maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.MatrixProductOperator, maestro.SimulationType.ExtendedStabilizer,
])
def test_seeded_auto_uses_multinomial_counts_and_preserves_ordered_records(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(0, 2), (0, 69)])
    shots = 4096
    config = maestro.SimulatorConfig(simulation_type=method, fixed_backend=True, max_simulators=1, seed=0)

    def run(policy, rows=False):
        config.counts_policy = policy
        return circuit.execute(config=config, shots=shots, return_samples=rows)

    counts = _assert_histogram(run("auto"), shots, 70)
    assert set(counts) == {"0" * 70, "001" + "0" * 66 + "1"}
    assert counts == run("auto")["counts"] == run("multinomial_v1")["counts"]
    assert counts != run("legacy")["counts"]

    previous = run("legacy", rows=True)["samples"]
    assert previous.shape == (shots, 70)
    np.testing.assert_array_equal(previous[:, 2], previous[:, 69])
    assert 0 < previous[:, 2].sum() < shots
    for policy in ["auto", "multinomial_v1"]:
        np.testing.assert_array_equal(run(policy, rows=True)["samples"], previous)
    config.max_simulators = 4
    np.testing.assert_array_equal(run("auto", rows=True)["samples"], previous)


@pytest.mark.parametrize("config_seed,noise_seed", [(0, None), (None, 0), (17, 0), (0, 17)])
def test_seeded_auto_samples_each_shared_noise_realization_with_multinomial_counts(config_seed, noise_seed):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.2)
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector, fixed_backend=True,
                                     seed=config_seed, max_simulators=1)

    def run(policy):
        config.counts_policy = policy
        return circuit.full_noise_execute(model, config=config, shots=2048, noise_realizations=2, noise_seed=noise_seed)

    automatic = run("auto")
    assert automatic["noise_realizations"] == 2
    counts = _assert_histogram(automatic, 2048, 2)
    assert counts == run("auto")["counts"] == run("multinomial_v1")["counts"]
    assert counts != run("legacy")["counts"]


def test_seeded_auto_reaches_the_component_of_a_noisy_composite():
    """A fixed composite keeps auto for a shared realization, so its component samples the batch adaptively."""
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.2)
    config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.CompositeQCSim, simulation_type=maestro.SimulationType.Statevector,
                                     fixed_backend=True, seed=17, max_simulators=1)

    def run(policy, realizations=2):
        config.counts_policy = policy
        return circuit.full_noise_execute(model, config=config, shots=2048, noise_realizations=realizations, noise_seed=5)

    automatic = run("auto")
    assert automatic["noise_realizations"] == 2
    counts = _assert_histogram(automatic, 2048, 2)
    assert counts == run("auto")["counts"] == run("multinomial_v1")["counts"]
    assert counts != run("legacy")["counts"]
    # One shot per realization has no batch to sample: every policy agrees.
    single = run("auto", 2048)["counts"]
    assert single == run("legacy", 2048)["counts"] == run("multinomial_v1", 2048)["counts"]
    # A candidate list that names only the composite is the same request.
    config = maestro.SimulatorConfig(optimization_candidates=[(maestro.SimulatorType.CompositeQCSim, maestro.SimulationType.Statevector)],
                                     seed=17, max_simulators=1)
    listed = run("auto")
    assert listed["simulator"] == maestro.SimulatorType.CompositeQCSim
    assert listed["counts"] == counts == run("multinomial_v1")["counts"]
    assert listed["counts"] != run("legacy")["counts"]
    # A fallback beside the composite changes nothing while the composite runs.
    composite = (maestro.SimulatorType.CompositeQCSim, maestro.SimulationType.Statevector)
    config = maestro.SimulatorConfig(optimization_candidates=[composite, (maestro.SimulatorType.QCSim, maestro.SimulationType.PauliPropagator)],
                                     seed=17, max_simulators=1)
    assert run("auto")["counts"] == counts
    # So does a listed QCSim stabilizer: frames are never chosen inside a
    # realization, and this seeded, non-Clifford circuit could not take them.
    stabilizer = [composite, (maestro.SimulatorType.QCSim, maestro.SimulationType.Stabilizer)]
    config = maestro.SimulatorConfig(optimization_candidates=stabilizer, seed=17, max_simulators=1)
    assert run("auto")["counts"] == counts
    # A Clifford circuit moves to that stabilizer, which does not read the policy.
    clifford = QuantumCircuit()
    clifford.h(0)
    clifford.cx(0, 1)
    clifford.measure_all()
    moved = {}
    for policy in ("auto", "legacy", "multinomial_v1"):
        config.counts_policy = policy
        moved[policy] = clifford.full_noise_execute(model, config=config, shots=2048, noise_realizations=2, noise_seed=5)
        assert moved[policy]["method"] == maestro.SimulationType.Stabilizer
    assert moved["auto"]["counts"] == moved["legacy"]["counts"] == moved["multinomial_v1"]["counts"]
    # Unseeded, such a realization could take frames itself: it keeps the
    # per-shot sampler and still runs on the stabilizer.
    unseeded = maestro.SimulatorConfig(optimization_candidates=stabilizer, max_simulators=1)
    result = clifford.full_noise_execute(model, config=unseeded, shots=2048, noise_realizations=2)
    assert result["method"] == maestro.SimulationType.Stabilizer and result.get("sampling_strategy") != "reference_frame_v1"
    assert sum(result["counts"].values()) == 2048 and set(result["counts"]) <= {"00", "11", "01", "10"}


def test_unseeded_noisy_counts_compile_model_and_circuit_channels():
    circuit, model = _noisy_bell()
    result = circuit.full_noise_execute(model, config=_config(), shots=1024, noise_realizations=1024)
    counts = _assert_histogram(result, 1024, 2)
    assert result["sampling_strategy"] == "reference_frame_v1"
    assert set(counts) == {"00", "01", "10", "11"}
    assert 100 < counts["01"] + counts["10"] < 500


def test_unseeded_noisy_execute_selects_frames_automatically():
    circuit, model = _noisy_bell()
    result = circuit.noisy_execute(model, config=_config(), shots=1024, noise_realizations=1024)
    counts = _assert_histogram(result, 1024, 2)
    assert result["sampling_strategy"] == "reference_frame_v1"
    assert set(counts) == {"00", "01", "10", "11"}
    assert 100 < counts["01"] + counts["10"] < 500


@pytest.mark.parametrize("policy", ["auto", "legacy", "frame_v1"])
def test_plain_counts_policy_preserves_zero_shots(policy):
    result = _bell().execute(config=_config(policy), shots=0)
    assert _assert_histogram(result, 0, 2) == {}


NON_STABILIZER_METHODS = [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
]


@pytest.mark.parametrize("method", NON_STABILIZER_METHODS)
@pytest.mark.parametrize("noisy", [False, True])
def test_auto_counts_preserves_requested_nonstabilizer_method(method, noisy):
    # Python restricts automatic selection to the requested method, so the
    # ordinary executor honors it even for Clifford circuits; auto must too.
    circuit = _bell()
    config = _config(method=method)
    if noisy:
        model = maestro.NoiseModel()
        model.set_bit_flip(1, 0.1)
        result = circuit.full_noise_execute(model, config=config, shots=31, noise_realizations=31)
    else:
        result = circuit.execute(config=config, shots=31)
    assert result["method"] == method
    assert result.get("sampling_strategy") != "reference_frame_v1"
    _assert_histogram(result, 31, 2)


@pytest.mark.parametrize("method", NON_STABILIZER_METHODS)
def test_explicit_frame_policy_may_replace_an_automatically_selected_method(method):
    result = _bell().execute(config=_config("frame_v1", method=method), shots=257)
    assert result["sampling_strategy"] == "reference_frame_v1"
    assert result["method"] == maestro.SimulationType.Stabilizer
    assert set(_assert_histogram(result, 257, 2)) <= {"00", "11"}


@pytest.mark.parametrize("seed", [0, 29])
def test_seeded_auto_stabilizer_counts_keep_the_ordinary_executor(seed):
    circuit = _bell()
    automatic = circuit.execute(config=_config("auto", seed), shots=257)
    legacy = circuit.execute(config=_config("legacy", seed), shots=257)
    assert automatic["counts"] == legacy["counts"]
    assert automatic.get("sampling_strategy") != "reference_frame_v1"


@pytest.mark.parametrize("config_seed,noise_seed", [(None, 0), (0, None), (0, 0), (29, 0), (0, 17)])
def test_seeded_auto_stabilizer_trajectories_honor_both_seed_sources(config_seed, noise_seed):
    circuit, model = _noisy_bell()
    kwargs = dict(shots=257, noise_realizations=257, noise_seed=noise_seed)
    automatic = circuit.full_noise_execute(model, config=_config("auto", config_seed), **kwargs)
    legacy = circuit.full_noise_execute(model, config=_config("legacy", config_seed), **kwargs)
    assert automatic["counts"] == legacy["counts"]
    assert automatic.get("sampling_strategy") != "reference_frame_v1"


def test_legacy_policy_never_selects_frames_for_unseeded_counts():
    circuit = _bell()
    clean = circuit.execute(config=_config("legacy"), shots=31)
    assert clean.get("sampling_strategy") != "reference_frame_v1"
    _assert_histogram(clean, 31, 2)
    circuit, model = _noisy_bell()
    noisy = circuit.full_noise_execute(model, config=_config("legacy"), shots=31, noise_realizations=31)
    assert noisy.get("sampling_strategy") != "reference_frame_v1"
    _assert_histogram(noisy, 31, 2)


@pytest.mark.parametrize("seed", [0, 29])
def test_explicit_frame_policy_replays_plain_seeded_counts(seed):
    circuit = _bell()
    config = _config("frame_v1", seed)
    first = circuit.execute(config=config, shots=513)
    assert first["sampling_strategy"] == "reference_frame_v1"
    counts = _assert_histogram(first, 513, 2)
    assert set(counts) == {"00", "11"}
    assert circuit.execute(config=config, shots=513)["counts"] == counts


def test_explicit_frame_policy_replays_noisy_counts_and_seed_alias():
    circuit, model = _noisy_bell()
    config = _config("frame_v1")
    kwargs = dict(config=config, shots=513, noise_realizations=513)
    first = circuit.full_noise_execute(model, noise_seed=0, **kwargs)
    assert first["sampling_strategy"] == "reference_frame_v1"
    counts = _assert_histogram(first, 513, 2)
    assert circuit.full_noise_execute(model, noise_seed=0, **kwargs)["counts"] == counts
    assert circuit.full_noise_execute(model, seed=0, **kwargs)["counts"] == counts


def test_zero_seed_alias_keeps_the_stabilizer_trajectory_counts():
    circuit, model = _noisy_bell()
    kwargs = dict(shots=127, noise_realizations=127)
    automatic = circuit.full_noise_execute(model, config=_config(), seed=0, **kwargs)
    legacy = circuit.full_noise_execute(model, config=_config("legacy"), noise_seed=0, **kwargs)
    assert automatic["counts"] == legacy["counts"]
    assert automatic.get("sampling_strategy") != "reference_frame_v1"


@pytest.mark.parametrize("policy", ["auto", "frame_v1"])
def test_frame_counts_policy_preserves_shared_realization_semantics(policy):
    circuit = QuantumCircuit()
    circuit.pauli_channel([0], [0.5, 0.5, 0.0, 0.0])
    circuit.measure_all()
    kwargs = dict(shots=137, noise_realizations=1, noise_seed=0)
    result = circuit.full_noise_execute(maestro.NoiseModel(), config=_config(policy), **kwargs)
    counts = _assert_histogram(result, 137, 1)
    assert len(counts) == 1
    assert result.get("sampling_strategy") != "reference_frame_v1"
    assert counts == circuit.full_noise_execute(maestro.NoiseModel(), config=_config("legacy"), **kwargs)["counts"]


def test_frame_counts_policy_falls_back_for_coherent_trajectories():
    circuit = _bell()
    model = maestro.NoiseModel()
    model.set_coherent_rotation(0, 0.11, 0.03, -0.09)
    kwargs = dict(shots=31, noise_realizations=31, noise_seed=5)
    method = maestro.SimulationType.MatrixProductState
    result = circuit.full_noise_execute(model, config=_config("frame_v1", 23, method), **kwargs)
    legacy = circuit.full_noise_execute(model, config=_config("legacy", 23, method), **kwargs)
    assert result["counts"] == legacy["counts"]
    assert result.get("sampling_strategy") != "reference_frame_v1"


@pytest.mark.parametrize("kind", ["non_clifford", "classical_control"])
def test_plain_frame_counts_policy_falls_back_for_unsupported_circuits(kind):
    if kind == "non_clifford":
        circuit = QuantumCircuit()
        circuit.h(0)
        circuit.t(0)
        circuit.h(0)
        circuit.measure_all()
    else:
        circuit = maestro.QasmToCirc().parse_and_translate(
            "OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0]; "
            "if (c[0]) { x q[1]; } c[1] = measure q[1];")
    method = maestro.SimulationType.Statevector
    result = circuit.execute(config=_config("frame_v1", 0, method), shots=257)
    legacy = circuit.execute(config=_config("legacy", 0, method), shots=257)
    assert result["counts"] == legacy["counts"]
    assert result.get("sampling_strategy") != "reference_frame_v1"
    if kind == "classical_control":
        assert set(result["counts"]) == {"00", "11"}


def test_frame_counts_policy_falls_back_for_composite_backend():
    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.CompositeQCSim,
        simulation_type=maestro.SimulationType.Statevector,
        counts_policy="frame_v1", seed=23, max_simulators=1,
    )
    result = _bell().execute(config=config, shots=127)
    assert result.get("sampling_strategy") != "reference_frame_v1"
    assert set(_assert_histogram(result, 127, 2)) == {"00", "11"}


def test_counts_policy_does_not_change_seeded_raw_frame_rows():
    circuit = _bell()
    rows = [circuit.execute(config=_config(policy, 0), shots=257, return_samples=True)["samples"]
            for policy in ["auto", "legacy", "frame_v1"]]
    for replay in rows[1:]:
        np.testing.assert_array_equal(rows[0], replay)


@pytest.mark.parametrize("qubit,clbit", [(4, 0), (3, 1), (0, 17)])
@pytest.mark.parametrize("noisy", [False, True])
def test_frame_counts_use_classical_width_and_preserve_sparse_indices(qubit, clbit, noisy):
    circuit = QuantumCircuit()
    circuit.x(qubit)
    circuit.measure([(qubit, clbit)])
    kwargs = dict(shots=37)
    if noisy:
        model = maestro.NoiseModel()
        model.set_bit_flip(qubit, 1.0)
        kwargs.update(noise_realizations=37, noise_seed=0)
        frame = circuit.full_noise_execute(model, config=_config("frame_v1", 0), **kwargs)
        legacy = circuit.full_noise_execute(model, config=_config("legacy", 0), **kwargs)
    else:
        frame = circuit.execute(config=_config("frame_v1", 0), **kwargs)
        legacy = circuit.execute(config=_config("legacy", 0), **kwargs)
    width = max(qubit + 1, clbit + 1)
    expected = "0" * clbit + ("0" if noisy else "1") + "0" * (width - clbit - 1)
    assert frame["counts"] == legacy["counts"] == {expected: 37}
    assert frame["sampling_strategy"] == "reference_frame_v1"
    assert all(len(bits) == width for bits in frame["counts"])


def _dynamic_frame_circuit():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(0, 0)])
    circuit.pauli_channel([1], [0.7, 0.1, 0.1, 0.1])
    circuit.h(1)
    circuit.measure([(0, 1), (1, 2)])
    circuit.reset(0)
    circuit.h(0)
    circuit.measure([(0, 3)])
    return circuit


@pytest.mark.parametrize("shots", [1, 65, 1025])
def test_plain_frame_counts_match_raw_sample_histogram_for_the_same_seed(shots):
    circuit = _dynamic_frame_circuit()
    config = _config("frame_v1", 0)
    counts = circuit.execute(config=config, shots=shots)
    raw = circuit.execute(config=config, shots=shots, return_samples=True)
    assert counts["sampling_strategy"] == raw["sampling_strategy"] == "reference_frame_v1"
    assert raw["samples"].shape == (shots, 4)
    histogram = dict(Counter("".join(str(int(bit)) for bit in row) for row in raw["samples"]))
    assert counts["counts"] == histogram


@pytest.mark.parametrize("config_seed,noise_seed", [(0, 0), (29, 7)])
@pytest.mark.parametrize("shots", [1, 65, 1025])
def test_noisy_frame_counts_match_raw_sample_histogram_and_noise_seed_tag(config_seed, noise_seed, shots):
    circuit = _dynamic_frame_circuit()
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.07)
    model.set_readout_error(1, 0.13, 0.23)
    kwargs = dict(config=_config("frame_v1", config_seed), shots=shots,
                  noise_realizations=shots, noise_seed=noise_seed)
    counts = circuit.full_noise_execute(model, **kwargs)
    raw = circuit.full_noise_execute(model, return_samples=True, **kwargs)
    assert counts["sampling_strategy"] == raw["sampling_strategy"] == "reference_frame_v1"
    assert raw["samples"].shape == (shots, 4)
    histogram = dict(Counter("".join(str(int(bit)) for bit in row) for row in raw["samples"]))
    assert counts["counts"] == histogram


def _dynamic_non_clifford_circuit():
    # A T gate keeps the frame compiler out; the mid-circuit record, readout and
    # sparse classical index exercise the direct trajectory executor.
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.measure([(0, 5)])
    circuit.h(1)
    circuit.cx(1, 2)
    circuit.measure([(1, 0), (2, 1)])
    return circuit


def _noise_with_readout():
    model = maestro.NoiseModel()
    model.set_all_depolarizing(3, 0.05)
    model.set_readout_error(2, 0.1, 0.2)
    return model


def _marginals(counts, width):
    shots = sum(counts.values())
    return [sum(count for bits, count in counts.items() if bits[bit] == "1") / shots for bit in range(width)]


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix])
@pytest.mark.parametrize("api", ["noisy_execute", "full_noise_execute"])
def test_unseeded_independent_counts_run_direct_trajectories(method, api):
    circuit, model, shots = _dynamic_non_clifford_circuit(), _noise_with_readout(), 4000
    direct = getattr(circuit, api)(model, config=_config(method=method), shots=shots, noise_realizations=shots)
    assert direct["sampling_strategy"] == "trajectory_v1"
    assert direct["method"] == method
    assert direct["noise_realizations"] == shots
    counts = _assert_histogram(direct, shots, 6)
    assert all(bits[3] == bits[4] == "0" for bits in counts)
    # Same distribution as the ordinary executor, sampled with a seed.
    legacy = getattr(circuit, api)(model, config=_config("legacy", 11, method), shots=shots, noise_realizations=shots)
    assert set(len(bits) for bits in legacy["counts"]) == {6}
    for ours, theirs in zip(_marginals(counts, 6), _marginals(legacy["counts"], 6)):
        assert ours == pytest.approx(theirs, abs=5 * math.sqrt(0.25 * 2 / shots))


@pytest.mark.parametrize("seed_source", ["config", "noise", "policy", "shared", "mps"])
def test_direct_trajectory_counts_leave_other_requests_on_their_executor(seed_source):
    circuit, model = _dynamic_non_clifford_circuit(), _noise_with_readout()
    method = maestro.SimulationType.MatrixProductState if seed_source == "mps" else maestro.SimulationType.Statevector
    config = _config("legacy" if seed_source == "policy" else "auto", 3 if seed_source == "config" else None, method)
    kwargs = dict(shots=257, noise_realizations=16 if seed_source == "shared" else 257)
    if seed_source == "noise":
        kwargs["noise_seed"] = 0
    result = circuit.full_noise_execute(model, config=config, **kwargs)
    assert result.get("sampling_strategy") != "trajectory_v1"
    _assert_histogram(result, 257, 6)
    if seed_source in ("config", "noise"):
        legacy = _config("legacy", 3 if seed_source == "config" else None, method)
        assert result["counts"] == circuit.full_noise_execute(model, config=legacy, **kwargs)["counts"]


def test_noisy_execution_never_attaches_readout_to_the_callers_circuit():
    # Injection shares the caller's immutable gates; measurements must still be
    # per-realization copies because readout rates are attached in place.
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.t(0)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_readout_error(0, 0.0, 1.0)
    for policy, seed in [("auto", None), ("legacy", 5)]:
        noisy = circuit.noisy_execute(model, config=_config(policy, seed, maestro.SimulationType.Statevector), shots=64, noise_realizations=64)
        assert noisy["counts"] == {"0": 64}
    clean = circuit.execute(config=_config("legacy", 5, maestro.SimulationType.Statevector), shots=64)
    assert clean["counts"] == {"1": 64}


def _rotation_ladder():
    # Ten Ry(0.3) leave each step a 0.149 amplitude to flip, below a 0.2
    # path-integral threshold; merged into one Ry(3) the |0> amplitude is 0.07.
    circuit = QuantumCircuit()
    for _ in range(10):
        circuit.ry(0, 0.3)
    circuit.measure_all()
    return circuit


def test_direct_trajectories_optimize_circuits_like_the_network_executor():
    # Python always optimizes circuits; unmerged, every shot would prune to 0.
    circuit, model, shots = _rotation_ladder(), maestro.NoiseModel(), 101
    model.set_bit_flip(0, 1e-15)

    def config(policy):
        return maestro.SimulatorConfig(simulation_type=maestro.SimulationType.PathIntegral, path_integral_threshold=0.2, counts_policy=policy,
                                       max_simulators=1)

    legacy = circuit.noisy_execute(model, config=config("legacy"), shots=shots, noise_realizations=shots)
    assert legacy["counts"] == {"1": shots}
    direct = circuit.noisy_execute(model, config=config("auto"), shots=shots, noise_realizations=shots)
    assert direct["sampling_strategy"] == "trajectory_v1"
    assert direct["counts"] == {"1": shots}
    raw = circuit.noisy_execute(model, config=config("auto"), shots=shots, noise_realizations=shots, return_samples=True)
    assert int(raw["samples"].sum()) == shots
    clean = circuit.execute(config=config("auto"), shots=shots, return_samples=True)
    assert int(clean["samples"].sum()) == shots


def test_direct_trajectories_accept_gates_that_optimization_makes_clifford():
    # T;T merges into S, so the network executor runs this on the stabilizer.
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.t(0)
    circuit.h(0)
    circuit.measure_all()
    # Readout noise inserts no gate between the two T gates.
    model, shots = maestro.NoiseModel(), 400
    model.set_readout_error(0, 0.01, 0.01)
    stabilizer = maestro.SimulationType.Stabilizer
    for policy, seed in [("legacy", None), ("auto", None), ("auto", 4)]:
        counts = _assert_histogram(circuit.noisy_execute(model, config=_config(policy, seed, stabilizer), shots=shots, noise_realizations=shots), shots, 1)
        assert 0.4 < counts.get("1", 0) / shots < 0.6
    raw = circuit.noisy_execute(model, config=_config(method=stabilizer), shots=shots, noise_realizations=shots, return_samples=True)
    assert raw["samples"].shape == (shots, 1)


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix])
@pytest.mark.parametrize("spectator", [1, 3, 30])
def test_direct_trajectories_drop_crosstalk_phases_beyond_the_register(method, spectator):
    # A spectator beyond the circuit stays in |0>, so its crosstalk rotation is
    # a global phase: direct sessions simulate the circuit's one qubit. Count
    # keys are padded to the spectator, as the ordinary executor and 0.3.13
    # pad them; raw sample columns keep the circuit's classical width.
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.measure([(0, 0)])
    model, shots = maestro.NoiseModel(), 1000
    model.set_crosstalk(0, spectator, 0.2)
    tolerance = 5 * math.sqrt(0.25 / shots)
    direct = circuit.full_noise_execute(model, config=_config(method=method), shots=shots, noise_realizations=shots)
    assert direct["sampling_strategy"] == "trajectory_v1"
    counts = _assert_histogram(direct, shots, spectator + 1)
    assert all(bits[1:] == "0" * spectator for bits in counts)
    assert _marginals(counts, spectator + 1)[0] == pytest.approx(0.5, abs=tolerance)
    raw = circuit.full_noise_execute(model, config=_config(method=method), shots=shots, noise_realizations=shots, return_samples=True)
    assert raw["samples"].shape == (shots, 1)
    assert raw["samples"].mean() == pytest.approx(0.5, abs=tolerance)
    for realizations in (shots, 50):
        legacy = circuit.full_noise_execute(model, config=_config("legacy", 7, method), shots=shots, noise_realizations=realizations)
        assert set(_assert_histogram(legacy, shots, spectator + 1)) <= {"0" * (spectator + 1), "1" + "0" * spectator}


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix])
def test_raw_samples_ignore_a_distant_crosstalk_spectator(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.measure([(0, 0)])
    model, shots = maestro.NoiseModel(), 1000
    model.set_crosstalk(0, 2**31 - 1, 0.2)
    raw = circuit.full_noise_execute(model, config=_config(method=method), shots=shots, noise_realizations=shots, return_samples=True)
    assert raw["sampling_strategy"] == "trajectory_v1"
    assert raw["samples"].shape == (shots, 1)
    assert raw["samples"].mean() == pytest.approx(0.5, abs=5 * math.sqrt(0.25 / shots))


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix])
def test_direct_trajectories_keep_crosstalk_inside_the_register(method):
    # Qubit 1 belongs to the circuit: the Rz(1) that T on qubit 0 puts on it
    # while it is in |+> is physical, so H Rz(1) H gives P(1) = sin^2(1/2).
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.h(1)
    circuit.t(0)
    circuit.h(1)
    circuit.measure([(1, 0)])
    model, shots = maestro.NoiseModel(), 2000
    model.set_crosstalk(0, 1, 1.0)
    expected = math.sin(0.5) ** 2
    tolerance = 5 * math.sqrt(expected * (1 - expected) / shots)
    for policy, seed in [("auto", None), ("legacy", 5)]:
        result = circuit.full_noise_execute(model, config=_config(policy, seed, method), shots=shots, noise_realizations=shots)
        counts = _assert_histogram(result, shots, 2)
        assert sum(n for bits, n in counts.items() if bits[0] == "1") / shots == pytest.approx(expected, abs=tolerance)


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.Stabilizer])
def test_measurement_free_noisy_counts_keep_the_legacy_key(method):
    circuit = QuantumCircuit()
    circuit.x(4)
    model = maestro.NoiseModel()
    model.set_bit_flip(4, 0.1)
    for policy, seed in [("legacy", None), ("legacy", 5), ("auto", None), ("auto", 5), ("frame_v1", None)]:
        result = circuit.noisy_execute(model, config=_config(policy, seed, method), shots=17, noise_realizations=17)
        assert result["counts"] == {"0": 17}
        assert result.get("sampling_strategy") != "trajectory_v1"


def test_direct_trajectory_counts_pack_wide_records():
    # Records span two packed words; bit 64 is certain and bit 70 is random.
    circuit = QuantumCircuit()
    circuit.x(1)
    circuit.measure([(1, 64)])
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.measure([(0, 70)])
    model, shots = maestro.NoiseModel(), 2000
    model.set_readout_error(0, 0.05, 0.05)
    direct = circuit.noisy_execute(model, config=_config(method=maestro.SimulationType.Statevector), shots=shots, noise_realizations=shots)
    assert direct["sampling_strategy"] == "trajectory_v1"
    counts = _assert_histogram(direct, shots, 71)
    assert all(bits[64] == "1" and bits.count("1") == 1 + (bits[70] == "1") for bits in counts)
    # P(1) for H T H, then a symmetric 5% readout flip.
    ideal = (1 - math.cos(math.pi / 4)) / 2
    p_one = 0.95 * ideal + 0.05 * (1 - ideal)
    assert _marginals(counts, 71)[70] == pytest.approx(p_one, abs=5 * math.sqrt(p_one * (1 - p_one) / shots))


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix])
def test_direct_sessions_compact_sparse_qubit_indices(method):
    # Only qubit 40 is used: a dense register up to it would need 2^41
    # amplitudes. Keys and sample columns keep their 41-wide padding.
    circuit = QuantumCircuit()
    circuit.h(40)
    circuit.t(40)
    circuit.h(40)
    circuit.measure([(40, 0)])
    model, shots = maestro.NoiseModel(), 2000
    model.set_readout_error(40, 0.05, 0.05)
    ideal = (1 - math.cos(math.pi / 4)) / 2
    p_one = 0.95 * ideal + 0.05 * (1 - ideal)
    tolerance = 5 * math.sqrt(p_one * (1 - p_one) / shots)
    direct = circuit.noisy_execute(model, config=_config(method=method), shots=shots, noise_realizations=shots)
    assert direct["sampling_strategy"] == "trajectory_v1"
    counts = _assert_histogram(direct, shots, 41)
    assert _marginals(counts, 41)[0] == pytest.approx(p_one, abs=tolerance)
    raw = circuit.noisy_execute(model, config=_config(method=method), shots=shots, noise_realizations=shots, return_samples=True)
    assert raw["samples"].shape == (shots, 41)
    assert raw["samples"][:, 0].mean() == pytest.approx(p_one, abs=tolerance)
    assert not raw["samples"][:, 1:].any()
    clean = circuit.execute(config=_config(method=method), shots=shots, return_samples=True)
    assert clean["samples"].shape == (shots, 41)
    assert clean["samples"][:, 0].mean() == pytest.approx(ideal, abs=5 * math.sqrt(ideal * (1 - ideal) / shots))
    legacy = circuit.noisy_execute(model, config=_config("legacy", 3, method), shots=shots, noise_realizations=shots)
    assert _marginals(_assert_histogram(legacy, shots, 41), 41)[0] == pytest.approx(p_one, abs=tolerance)


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix])
def test_direct_sessions_drop_crosstalk_on_untouched_qubits_inside_the_register(method):
    # Qubits 0 and 19 are used; spectator 5 lies inside the register but no
    # operation touches it, so it stays in |0> and its rotation is a phase.
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.x(19)
    circuit.measure([(0, 0), (19, 1)])
    model, shots = maestro.NoiseModel(), 1000
    model.set_crosstalk(0, 5, 0.4)
    model.set_crosstalk(19, 7, 0.4)
    for return_samples in (False, True):
        result = circuit.full_noise_execute(model, config=_config(method=method), shots=shots, noise_realizations=shots, return_samples=return_samples)
        assert result["sampling_strategy"] == "trajectory_v1"
        if return_samples:
            rows = result["samples"]
            assert rows.shape == (shots, 20) and rows[:, 1].all()
            assert rows[:, 0].mean() == pytest.approx(0.5, abs=5 * math.sqrt(0.25 / shots))
        else:
            counts = _assert_histogram(result, shots, 20)
            assert all(bits[1] == "1" for bits in counts)
            assert _marginals(counts, 20)[0] == pytest.approx(0.5, abs=5 * math.sqrt(0.25 / shots))


def test_phase_damping_compiles_to_frames():
    # Phase damping is a Z flip with p = (1 - sqrt(1 - gamma)) / 2 = 0.1 here.
    # After the first H the qubit is |+>, where that flip makes |-> and the
    # second H then gives 1; after the second H a Z flip changes nothing.
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.h(0)
    circuit.measure([(0, 0)])
    model, shots = maestro.NoiseModel(), 20000
    model.set_phase_damping(0, 0.36)
    result = circuit.noisy_execute(model, config=_config(), shots=shots, noise_realizations=shots)
    assert result["sampling_strategy"] == "reference_frame_v1"
    counts = _assert_histogram(result, shots, 1)
    flips = counts.get("1", 0) / shots
    assert flips == pytest.approx(0.1, abs=5 * math.sqrt(0.09 / shots))


@pytest.mark.parametrize("method", [maestro.SimulationType.Stabilizer, maestro.SimulationType.MatrixProductState])
def test_explicit_trajectory_policy_makes_non_dense_counts_worker_independent(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.measure([(0, 0)])
    circuit.reset(0)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(0, 1), (1, 2)])

    def counts(workers, policy):
        config = maestro.SimulatorConfig(simulation_type=method, seed=11, max_simulators=workers, max_bond_dimension=8,
                                         trajectory_policy=policy, counts_policy="legacy")
        return circuit.execute(config=config, shots=600)["counts"]

    assert counts(1, "shot_v1") == counts(4, "shot_v1")
    assert sum(counts(4, None).values()) == 600


def test_composite_measures_qubits_beyond_64():
    circuit = QuantumCircuit()
    circuit.x(70)
    circuit.measure([(70, 0)])
    circuit.x(70)
    circuit.h(3)
    circuit.measure([(70, 1), (3, 2)])
    config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.CompositeQCSim, simulation_type=maestro.SimulationType.Statevector, seed=4)
    counts = circuit.execute(config=config, shots=200)["counts"]
    assert sum(counts.values()) == 200
    assert all(bits[0] == "1" and bits[1] == "0" for bits in counts)
    assert {bits[2] for bits in counts} == {"0", "1"}


def test_package_exports_and_seed_alias_on_checkpoints():
    import maestro as package
    assert not {"ast", "copyreg", "functools", "math", "numbers", "re", "sys"} & set(vars(package))
    assert package.noisy_execute.__wrapped__ is package._raw_maestro.noisy_execute
    prefix = QuantumCircuit()
    prefix.h(0)
    suffix = QuantumCircuit()
    suffix.measure([(0, 0)])
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.1)
    config = maestro.SimulatorConfig(seed=2, max_simulators=1)
    run = lambda **kwargs: maestro.PrefixCheckpointedSimulator(prefix, 1, config).execute_suffix(
        suffix, shots=64, noise_model=model, noise_realizations=64, **kwargs)["counts"]
    assert run(seed=9) == run(noise_seed=9)
    with pytest.raises(TypeError, match="either seed or noise_seed"):
        run(seed=9, noise_seed=9)


def test_noisy_counts_and_rows_report_the_bond_dimension():
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.1)
    config = _config("auto", 3, maestro.SimulationType.MatrixProductState)
    for return_samples in (False, True):
        result = _bell().full_noise_execute(model, config=config, shots=16, noise_realizations=4,
                                            noise_seed=1, return_samples=return_samples)
        assert result["max_bond_dim_reached"] == 2
        assert "gpu_device" not in result
