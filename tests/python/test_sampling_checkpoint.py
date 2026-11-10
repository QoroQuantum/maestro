"""Raw checkpoint records preserve noise streams and classical bit positions."""

import warnings
from collections import Counter

import numpy as np
import pytest

import maestro
from maestro.circuits import QuantumCircuit


def _histogram(samples):
    return dict(Counter("".join(str(int(bit)) for bit in row) for row in samples))


def _checkpoint(method, workers=1, seed=29, policy="auto"):
    prefix = QuantumCircuit()
    prefix.h(0)
    prefix.cx(0, 1)
    config = maestro.SimulatorConfig(
        simulation_type=method, seed=seed, max_simulators=workers,
        max_bond_dimension=16, counts_policy=policy,
    )
    return maestro.PrefixCheckpointedSimulator(prefix, 2, config)


def _suffix():
    circuit = QuantumCircuit()
    circuit.cx(0, 1)
    circuit.measure([(0, 0), (1, 1)])
    circuit.h(0)
    circuit.measure([(0, 2), (1, 3)])
    return circuit


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Stabilizer,
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
])
@pytest.mark.parametrize("realizations", [1, 19, 137])
def test_raw_checkpoint_matches_histogram_and_worker_count(method, realizations):
    model = maestro.NoiseModel()
    model.set_all_depolarizing(2, 0.15)
    model.set_readout_error_symmetric(0, 0.2)
    kwargs = dict(shots=137, noise_realizations=realizations, noise_model=model, noise_seed=11,
                  num_measurements=5)
    counts = _checkpoint(method).execute_suffix(_suffix(), **kwargs)
    sequential = _checkpoint(method).execute_suffix(_suffix(), return_samples=True, **kwargs)
    parallel = _checkpoint(method, workers=4).execute_suffix(_suffix(), return_samples=True, **kwargs)
    assert "counts" in counts and "samples" not in counts
    assert "samples" in sequential and "counts" not in sequential
    rows = sequential["samples"]
    assert rows.shape == (137, 5)
    assert rows.dtype == np.uint8 and rows.flags.c_contiguous
    assert not rows[:, 4].any()
    assert _histogram(rows) == counts["counts"]
    np.testing.assert_array_equal(parallel["samples"], rows)


@pytest.mark.parametrize("noisy", [False, True])
def test_checkpoint_raw_call_stream_advances_and_explicit_seed_replays(noisy):
    kwargs = dict(shots=251, num_measurements=4)
    if noisy:
        model = maestro.NoiseModel()
        model.set_all_depolarizing(2, 0.1)
        kwargs.update(noise_model=model, noise_realizations=251)

    def batches(checkpoint):
        return [checkpoint.execute_suffix(_suffix(), return_samples=True, **kwargs)["samples"]
                for _ in range(3)]

    method = maestro.SimulationType.Stabilizer
    first = batches(_checkpoint(method, workers=4))
    replay = batches(_checkpoint(method, workers=4))
    assert len({rows.tobytes() for rows in first}) == 3
    for actual, expected in zip(first, replay):
        np.testing.assert_array_equal(actual, expected)
    checkpoint = _checkpoint(method, workers=4)
    a = checkpoint.execute_suffix(_suffix(), noise_seed=7, return_samples=True, **kwargs)["samples"]
    b = checkpoint.execute_suffix(_suffix(), noise_seed=7, return_samples=True, **kwargs)["samples"]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.ExtendedStabilizer])
@pytest.mark.parametrize("checkpoint_seed,call_seed", [(0, None), (None, 0)])
def test_seeded_checkpoint_switches_between_terminal_counts_and_ordered_rows(method, checkpoint_seed, call_seed):
    suffix = QuantumCircuit()
    suffix.cx(0, 1)
    suffix.ry(0, 0.37)
    suffix.cx(0, 1)
    suffix.measure([(0, 0), (1, 2)])
    shots = 4096

    def sequence(policy):
        checkpoint = _checkpoint(method, seed=checkpoint_seed, policy=policy)
        options = dict(shots=shots, noise_seed=call_seed)
        first = checkpoint.execute_suffix(suffix, **options)["counts"]
        rows = checkpoint.execute_suffix(suffix, return_samples=True, **options)["samples"]
        last = checkpoint.execute_suffix(suffix, **options)["counts"]
        return (first, last), rows

    counts, rows = sequence("auto")
    repeat, repeat_rows = sequence("auto")
    adaptive, adaptive_rows = sequence("multinomial_v1")
    historical, historical_rows = sequence("legacy")
    assert counts == repeat == adaptive
    assert counts != historical
    for histogram in counts:
        assert set(histogram) == {"000", "101"}
        assert sum(histogram.values()) == shots
    assert (counts[0] == counts[1]) == (call_seed is not None)
    for expected in [repeat_rows, adaptive_rows, historical_rows]:
        np.testing.assert_array_equal(rows, expected)


@pytest.mark.parametrize("listed", [False, True])
def test_rejected_noisy_suffix_leaves_a_seeded_checkpoint_unchanged(listed):
    """A thermal warning raised as an error stops the call before it runs, so it takes no per-call seed."""
    clean = QuantumCircuit()
    clean.h(0)
    clean.measure_all()
    clamped = maestro.NoiseModel()
    clamped.set_thermal_relaxation(0, 1.0, 1.0, 2.0)  # T2 > T1: sampled relaxation clamps it and warns
    B, S = maestro.SimulatorType, maestro.SimulationType
    backend = (dict(optimization_candidates=[(B.QCSim, S.Statevector), (B.QCSim, S.PauliPropagator)]) if listed
               else dict(simulation_type=S.Statevector, fixed_backend=True))

    def checkpoint():
        return maestro.PrefixCheckpointedSimulator(QuantumCircuit(), 1, maestro.SimulatorConfig(seed=7, **backend))

    fresh = checkpoint()
    first = fresh.execute_suffix(clean, shots=200)["counts"]
    second = fresh.execute_suffix(clean, shots=200)["counts"]
    rejected = checkpoint()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeWarning, match="T2 clamped to T1"):
            rejected.execute_suffix(clean, shots=200, noise_model=clamped)
    assert rejected.execute_suffix(clean, shots=200)["counts"] == first
    # A call that only warns runs, and its successor then has the second seed.
    warned = checkpoint()
    with pytest.warns(RuntimeWarning, match="T2 clamped to T1"):
        noisy = warned.execute_suffix(clean, shots=200, noise_model=clamped)
    assert noisy["method"] == S.Statevector and noisy["noise_evaluation"] == "trajectories"
    assert warned.execute_suffix(clean, shots=200)["counts"] == second


def test_pauli_propagator_prefix_noise_is_exact_or_refused():
    """The propagator evolves a noisy prefix once for the noise it applies exactly, and says why it refuses relaxation."""
    prefix = QuantumCircuit()
    prefix.h(0)
    prefix.cx(0, 1)
    suffix = QuantumCircuit()
    suffix.measure([(0, 0), (1, 1)])
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.PauliPropagator, fixed_backend=True, seed=3)
    pauli = maestro.NoiseModel()
    pauli.set_all_depolarizing(2, 0.05)
    noisy = maestro.PrefixCheckpointedSimulator(prefix, 2, config, pauli).execute_suffix(suffix, shots=600)
    ideal = maestro.PrefixCheckpointedSimulator(prefix, 2, config).execute_suffix(suffix, shots=600)
    assert set(ideal["counts"]) == {"00", "11"} and noisy["sampling_strategy"] == "terminal_v1"
    assert set(noisy["counts"]) - {"00", "11"}, "the prefix noise left no trace in the saved state"
    # Exact suffix noise needs one realization there, as on a density matrix.
    result = maestro.PrefixCheckpointedSimulator(prefix, 2, config).execute_suffix(suffix, shots=600, noise_model=pauli)
    assert result["noise_evaluation"] == "exact_channels" and result["noise_realizations"] == 1
    relaxing = maestro.NoiseModel()
    relaxing.set_t1(0, 0.1)
    with pytest.raises(ValueError, match="Pauli propagator samples relaxation"):
        maestro.PrefixCheckpointedSimulator(prefix, 2, config, relaxing)
    with pytest.raises(ValueError, match="density-matrix or MPO backend or on the QCSim Pauli propagator"):
        statevector = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector, fixed_backend=True, seed=3)
        maestro.PrefixCheckpointedSimulator(prefix, 2, statevector, pauli)


def test_pauli_propagator_checkpoint_counts_follow_the_policy():
    """auto samples the saved prefix once; legacy keeps the per-shot replay and its seeded readout."""
    prefix = QuantumCircuit()
    prefix.h(0)
    prefix.cx(0, 1)
    suffix = QuantumCircuit()
    suffix.ry(1, 0.4)
    suffix.measure_all()
    readout = maestro.NoiseModel()
    readout.set_readout_error(0, 0.1, 0.05)

    def run(policy, model=None, rows=False):
        config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.PauliPropagator, fixed_backend=True, max_simulators=1,
                                         seed=29, counts_policy=policy)
        checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 2, config)
        return checkpoint.execute_suffix(suffix, shots=600, noise_model=model, return_samples=rows)

    for model in (None, readout):
        strategies = {}
        for policy in ("auto", "legacy"):
            result = run(policy, model)
            strategies[policy] = result["sampling_strategy"]
            assert sum(result["counts"].values()) == 600
            assert result["counts"] == run(policy, model)["counts"]
        assert strategies == {"auto": "terminal_v1", "legacy": "trajectory_v1"}
    # Ordered rows keep their shot order under legacy as well.
    assert run("legacy", rows=True)["sampling_strategy"] == "trajectory_v1"
    assert run("auto", rows=True)["sampling_strategy"] == "terminal_v1"


def test_raw_checkpoint_readout_precedes_classical_control():
    prefix = QuantumCircuit()
    prefix.x(1)
    suffix = maestro.QasmToCirc().parse_and_translate(
        "OPENQASM 3.0; qubit[2] q; bit[4] c; "
        "c[0] = measure q[0]; if (c[0]) { x q[1]; } c[3] = measure q[1];"
    )
    model = maestro.NoiseModel()
    model.set_readout_error_symmetric(0, 1.0)
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Stabilizer, max_simulators=4)
    checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 2, config)
    rows = checkpoint.execute_suffix(suffix, shots=73, noise_realizations=73, noise_seed=4,
                                     noise_model=model, return_samples=True)["samples"]
    np.testing.assert_array_equal(rows, np.tile([1, 0, 0, 0], (73, 1)))


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("return_samples", [False, True])
def test_checkpoint_clears_bits_not_written_by_a_conditional_measurement(noisy, return_samples):
    prefix = QuantumCircuit()
    prefix.x(1)
    suffix = maestro.QasmToCirc().parse_and_translate(
        "OPENQASM 3.0; qubit[2] q; bit[4] c; h q[0]; "
        "c[0] = measure q[0]; if (c[0]) { c[3] = measure q[1]; }"
    )
    model = maestro.NoiseModel()
    if noisy:
        model.set_readout_error_symmetric(0, 0.2)
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Stabilizer, max_simulators=4)
    checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 2, config)
    result = checkpoint.execute_suffix(suffix, shots=257, noise_realizations=19, noise_seed=13,
                                       noise_model=model, return_samples=return_samples)
    if return_samples:
        rows = result["samples"]
        np.testing.assert_array_equal(rows[:, 0], rows[:, 3])
        assert 0 < rows[:, 0].sum() < 257
    else:
        assert set(result["counts"]) == {"0000", "1001"}


def test_raw_samples_survive_checkpoint_destruction():
    rows = _checkpoint(maestro.SimulationType.Stabilizer).execute_suffix(
        _suffix(), shots=23, noise_seed=5, return_samples=True)["samples"]
    assert rows.shape == (23, 4)
    assert np.isin(rows, [0, 1]).all()


def _mixed_checkpoint(method, workers=1):
    prefix = QuantumCircuit()
    prefix.h(0)
    prefix.cx(0, 1)
    config = maestro.SimulatorConfig(
        simulation_type=method, fixed_backend=True, seed=29,
        max_simulators=workers, max_bond_dimension=16,
    )
    return maestro.PrefixCheckpointedSimulator(prefix, 2, config)


@pytest.mark.parametrize("method", [maestro.SimulationType.DensityMatrix,
                                    maestro.SimulationType.MatrixProductOperator])
def test_mixed_checkpoint_prepares_terminal_suffix_once_and_preserves_prefix(method):
    suffix = QuantumCircuit()
    suffix.pauli_channel([0], [0.8, 0.2, 0.0, 0.0])
    suffix.measure([(1, 0), (0, 2)])
    suffix.measure([(1, 3)])
    model = maestro.NoiseModel()
    model.set_readout_error_symmetric(0, 0.1)
    kwargs = dict(shots=2048, noise_model=model, return_samples=True)
    checkpoint = _mixed_checkpoint(method)
    first = checkpoint.execute_suffix(suffix, **kwargs)
    second = checkpoint.execute_suffix(suffix, **kwargs)
    for result in (first, second):
        assert result["method"] == method
        assert result["noise_realizations"] == 1
        assert result["sampling_strategy"] == "terminal_v1"
        rows = result["samples"]
        assert rows.shape == (2048, 4)
        assert not rows[:, 1].any()
        np.testing.assert_array_equal(rows[:, 0], rows[:, 3])
        assert abs(np.mean(rows[:, 0] != rows[:, 2]) - 0.26) < 0.04
    assert not np.array_equal(first["samples"], second["samples"])
    replay = _mixed_checkpoint(method)
    np.testing.assert_array_equal(first["samples"], replay.execute_suffix(suffix, **kwargs)["samples"])
    np.testing.assert_array_equal(second["samples"], replay.execute_suffix(suffix, **kwargs)["samples"])
    seeded = checkpoint.execute_suffix(suffix, noise_seed=7, **kwargs)
    np.testing.assert_array_equal(seeded["samples"], checkpoint.execute_suffix(suffix, noise_seed=7, **kwargs)["samples"])

    # Sampling a noisy suffix must never replace the saved Bell-pair prefix.
    undo = QuantumCircuit()
    undo.cx(0, 1)
    undo.h(0)
    undo.measure([(0, 0), (1, 1)])
    clean = checkpoint.execute_suffix(undo, shots=31)
    assert clean["counts"] == {"00": 31}
    assert clean["sampling_strategy"] == "terminal_v1"


@pytest.mark.parametrize("method", [maestro.SimulationType.DensityMatrix,
                                    maestro.SimulationType.MatrixProductOperator])
def test_mixed_checkpoint_honors_explicit_realizations_and_row_histogram(method):
    suffix = QuantumCircuit()
    suffix.pauli_channel([0], [0.7, 0.3, 0.0, 0.0])
    suffix.measure([(0, 2), (1, 0)])
    kwargs = dict(shots=301, noise_realizations=7, noise_seed=11)
    counts = _mixed_checkpoint(method).execute_suffix(suffix, **kwargs)
    rows = _mixed_checkpoint(method, workers=4).execute_suffix(suffix, return_samples=True, **kwargs)
    assert counts["noise_realizations"] == rows["noise_realizations"] == 7
    assert counts["sampling_strategy"] == rows["sampling_strategy"] == "terminal_v1"
    assert counts["counts"] == _histogram(rows["samples"])


@pytest.mark.parametrize("method", [maestro.SimulationType.DensityMatrix,
                                    maestro.SimulationType.MatrixProductOperator])
def test_mixed_checkpoint_keeps_dynamic_records_and_batches_unobserved_reset(method):
    model = maestro.NoiseModel()
    model.set_readout_error_symmetric(0, 0.1)
    dynamic = _mixed_checkpoint(method).execute_suffix(
        _suffix(), shots=257, noise_model=model, return_samples=True)
    assert dynamic["noise_realizations"] == 1
    assert dynamic["sampling_strategy"] == "trajectory_v1"
    assert dynamic["samples"].shape == (257, 4)
    assert not dynamic["samples"][:, [1, 3]].any()
    reset = maestro.QasmToCirc().parse_and_translate(
        "OPENQASM 3.0; qubit[2] q; bit[3] c; h q[0]; reset q[0]; x q[0]; "
        "c[2] = measure q[0]; c[0] = measure q[1];"
    )
    result = _mixed_checkpoint(method).execute_suffix(reset, shots=257, return_samples=True)
    assert result["sampling_strategy"] == "terminal_v1"
    assert result["samples"][:, 2].all()
    assert not result["samples"][:, 1].any()
    assert 80 < result["samples"][:, 0].sum() < 180


def test_sinter_raw_matrix_validates_shape_values_and_padding():
    pytest.importorskip("sinter")
    pytest.importorskip("stim")
    from maestro.sinter import samples_to_measurements

    rows = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    parsed = samples_to_measurements(rows, 2, 2, storage_width=3)
    np.testing.assert_array_equal(
        parsed, [[True, False], [False, True]])
    assert np.shares_memory(parsed, rows)
    for invalid in [np.zeros((1, 2), dtype=bool), np.array([[2, 0], [0, 0]], dtype=np.uint8),
                    np.zeros((2, 2), dtype=np.int32), np.zeros((2, 1), dtype=bool)]:
        with pytest.raises(ValueError):
            samples_to_measurements(invalid, 2, 2)
    rows[0, 2] = 1
    with pytest.raises(ValueError, match="outside"):
        samples_to_measurements(rows, 2, 2, storage_width=3)


def test_sinter_requests_raw_samples_and_preserves_record_order(monkeypatch):
    pytest.importorskip("sinter")
    stim = pytest.importorskip("stim")
    import maestro.sinter as integration

    def reject_histogram(*args, **kwargs):
        raise AssertionError("Sinter should use the direct sample matrix")

    monkeypatch.setattr(integration, "counts_to_measurements", reject_histogram)
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Stabilizer, max_simulators=4)
    for checkpoint, text in [
        (False, "R 0\nH 0\nM 0\nM 0"),
        (False, "R 0\nX_ERROR(0.3) 0\nM 0\nM 0"),
        (True, "R 0\nH 0\nX_ERROR(0.3) 0\nM 0\nM 0"),
    ]:
        sampler = integration.MaestroCompiledSampler(stim.Circuit(text), config=config,
                                                     enable_checkpoint=checkpoint, seed=17)
        first = sampler.sample_measurements(257, seed=9)
        np.testing.assert_array_equal(first, sampler.sample_measurements(257, seed=9))
        assert first.shape == (257, 2) and first.dtype == bool
        np.testing.assert_array_equal(first[:, 0], first[:, 1])
        assert 0 < first[:, 0].sum() < 257


def test_sinter_old_extension_fallback_does_not_mask_execution_errors():
    pytest.importorskip("sinter")
    pytest.importorskip("stim")
    from maestro.sinter import _sample_result

    def legacy(shots):
        return {"counts": {"0": shots}}

    assert _sample_result(legacy, shots=7) == {"counts": {"0": 7}}

    def broken(*args, **kwargs):
        raise TypeError("invalid circuit")

    with pytest.raises(TypeError, match="invalid circuit"):
        _sample_result(broken, shots=7)


@pytest.mark.parametrize("noisy", [False, True])
def test_default_sinter_mps_configuration_uses_clifford_frame_sampler(monkeypatch, noisy):
    """Default automatic backend choice must avoid MPS replay for Clifford QEC."""
    pytest.importorskip("sinter")
    stim = pytest.importorskip("stim")
    import maestro.sinter as integration

    results = []
    compile_reference = maestro._compile_reference_sampler

    class TracedPlan:
        def __init__(self, plan):
            self.plan = plan

        def sample(self, *args, **kwargs):
            result = self.plan.sample(*args, **kwargs)
            results.append(result)
            return result

    def trace_compile(*args):
        plan = compile_reference(*args)
        return TracedPlan(plan) if plan is not None else None

    monkeypatch.setattr(maestro, "_compile_reference_sampler", trace_compile)
    noise = "DEPOLARIZE1(0.1) 0\n" if noisy else ""
    circuit = stim.Circuit("R 0 1\nH 0\nCX 0 1\n" + noise + "M 0 1")
    sampler = integration.MaestroCompiledSampler(circuit, seed=31)
    assert sampler.config.simulation_type == maestro.SimulationType.MatrixProductState
    assert sampler._reference_sampler is not None
    assert sampler.checkpoint_sim is None
    rows = sampler.sample_measurements(1024, seed=7)
    assert results[-1]["sampling_strategy"] == "reference_frame_v1"
    assert results[-1]["method"] == maestro.SimulationType.Stabilizer
    assert rows.shape == (1024, 2) and rows.dtype == bool
    if noisy:
        assert 0.01 < np.mean(rows[:, 0] != rows[:, 1]) < 0.15
    else:
        np.testing.assert_array_equal(rows[:, 0], rows[:, 1])
    np.testing.assert_array_equal(rows, sampler.sample_measurements(1024, seed=7))


def test_sinter_older_extension_uses_checkpoint_when_compiler_is_absent(monkeypatch):
    pytest.importorskip("sinter")
    stim = pytest.importorskip("stim")
    from maestro.sinter import MaestroCompiledSampler

    monkeypatch.delattr(maestro, "_compile_reference_sampler", raising=False)
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Stabilizer)
    sampler = MaestroCompiledSampler(stim.Circuit("H 0\nCX 0 1\nM 0 1"), config=config, seed=5)
    assert sampler._reference_sampler is None
    assert sampler.checkpoint_sim is not None
    rows = sampler.sample_measurements(127, seed=9)
    np.testing.assert_array_equal(rows[:, 0], rows[:, 1])
    np.testing.assert_array_equal(rows, sampler.sample_measurements(127, seed=9))


@pytest.mark.parametrize("return_samples", [False, True])
@pytest.mark.parametrize("noise_seed", [None, 13])
def test_checkpoint_drops_crosstalk_phases_outside_its_register(return_samples, noise_seed):
    # Spectators outside the checkpoint register stay in |0>, where crosstalk is
    # a global phase: results match the same noise without crosstalk.
    prefix = QuantumCircuit()
    prefix.h(0)
    suffix = QuantumCircuit()
    suffix.t(0)
    suffix.h(0)
    suffix.measure([(0, 0)])
    base = maestro.NoiseModel()
    base.set_bit_flip(0, 0.05)
    crosstalk = maestro.NoiseModel()
    crosstalk.set_bit_flip(0, 0.05)
    crosstalk.set_crosstalk(0, 1, 0.2)
    crosstalk.set_crosstalk(0, 4, 0.3)
    config = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector, seed=3, max_simulators=1)
    kwargs = dict(shots=300, noise_realizations=300, noise_seed=noise_seed, return_samples=return_samples)
    with_crosstalk = maestro.PrefixCheckpointedSimulator(prefix, 1, config).execute_suffix(suffix, noise_model=crosstalk, **kwargs)
    without = maestro.PrefixCheckpointedSimulator(prefix, 1, config).execute_suffix(suffix, noise_model=base, **kwargs)
    key = "samples" if return_samples else "counts"
    if return_samples:
        assert with_crosstalk[key].shape == (300, 1)
        assert np.array_equal(with_crosstalk[key], without[key])
    else:
        assert sum(with_crosstalk[key].values()) == 300
        assert with_crosstalk[key] == without[key]


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.MatrixProductState])
def test_monte_carlo_estimates_and_fidelity_do_not_depend_on_worker_count(method):
    # libmaestro runs these realizations on workers; each keeps its seed and
    # values are summed in realization order.
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rx(1, 0.4)
    model = maestro.NoiseModel()
    model.set_all_depolarizing(2, 0.1)
    estimates, fidelities = [], []
    for workers in (1, 4):
        config = maestro.SimulatorConfig(simulation_type=method, seed=5, max_simulators=workers, max_bond_dimension=8)
        estimates.append(circuit.noisy_estimate_montecarlo(["ZZ", "XX", "ZI"], model, 32, config, noise_seed=3))
        fidelities.append(circuit.noisy_fidelity(model, noise_realizations=32, config=config, noise_seed=3))
    serial, parallel = estimates
    assert parallel["expectation_values"] == pytest.approx(serial["expectation_values"], abs=1e-12)
    assert parallel["ideal_expectation_values"] == pytest.approx(serial["ideal_expectation_values"], abs=1e-12)
    # Workers start from a cache warmed by the first realization.
    assert parallel["cone_cache_hits"] == serial["cone_cache_hits"] == 31
    assert parallel["noise_realizations"] == 32
    assert abs(serial["expectation_values"][0]) < serial["ideal_expectation_values"][0] - 0.01
    assert fidelities[1]["fidelity"] == pytest.approx(fidelities[0]["fidelity"], abs=1e-12)
    assert fidelities[1]["std_error"] == pytest.approx(fidelities[0]["std_error"], abs=1e-12)
    assert 0.3 < fidelities[0]["fidelity"] < 0.999


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector,
                                    maestro.SimulationType.Stabilizer,
                                    maestro.SimulationType.MatrixProductState])
def test_pure_checkpoint_samples_terminal_suffix_once_and_keeps_reset_branches(method):
    checkpoint = _checkpoint(method)
    suffix = QuantumCircuit()
    suffix.measure([(0, 0), (1, 1)])
    first = checkpoint.execute_suffix(suffix, shots=2000, return_samples=True)
    assert first["sampling_strategy"] == "terminal_v1"
    rows = first["samples"]
    np.testing.assert_array_equal(rows[:, 0], rows[:, 1])
    assert 800 < rows[:, 0].sum() < 1200
    # The saved Bell pair survives terminal sampling.
    undo = QuantumCircuit()
    undo.cx(0, 1)
    undo.h(0)
    undo.measure([(0, 0), (1, 1)])
    assert checkpoint.execute_suffix(undo, shots=64)["counts"] == {"00": 64}
    # A reset on the entangled saved state collapses a branch per shot.
    reset = QuantumCircuit()
    reset.reset(0)
    reset.measure([(1, 1)])
    result = checkpoint.execute_suffix(reset, shots=400, return_samples=True)
    assert result["sampling_strategy"] == "trajectory_v1"
    assert 120 < result["samples"][:, 1].sum() < 280
