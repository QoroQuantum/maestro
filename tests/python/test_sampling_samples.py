"""Direct shot records preserve circuit semantics and independent noise draws."""

import gc
import math
from collections import Counter

import numpy as np
import pytest

import maestro
from maestro.circuits import QuantumCircuit


METHODS = [
    maestro.SimulationType.Stabilizer,
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
]


def _config(method, workers=1, seed=23):
    return maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=method, seed=seed, max_simulators=workers,
        max_bond_dimension=16,
    )


def _records(result, shots, width):
    assert "samples" in result and "counts" not in result
    rows = result["samples"]
    assert rows.shape == (shots, width)
    assert rows.dtype == np.uint8 and rows.flags.c_contiguous
    assert np.isin(rows, [0, 1]).all()
    return rows


@pytest.mark.parametrize("method", METHODS)
def test_execute_keeps_counts_default_and_returns_owned_raw_matrix(method):
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.measure_all()
    config = _config(method)
    default = circuit.execute(config=config, shots=29)
    assert default["counts"] == {"1": 29}
    assert "samples" not in default
    result = maestro.simple_execute(circuit, config=config, shots=29, return_samples=True)
    rows = _records(result, 29, 1)
    del result, circuit
    gc.collect()
    np.testing.assert_array_equal(rows, np.ones((29, 1), dtype=np.uint8))


@pytest.mark.parametrize("method", METHODS)
def test_raw_execute_handles_zero_shots_and_unmeasured_storage(method):
    circuit = QuantumCircuit()
    circuit.x(4)
    config = _config(method)
    _records(circuit.execute(config=config, shots=0, return_samples=True), 0, 5)
    rows = _records(circuit.execute(config=config, shots=31, return_samples=True), 31, 5)
    # Records contain classical writes; an unmeasured quantum state contributes
    # no writes, just as in the existing counts path.
    np.testing.assert_array_equal(rows, np.zeros((31, 5), dtype=np.uint8))
    counts = circuit.execute(config=config, shots=31)["counts"]
    assert sum(counts.values()) == 31
    assert all(set(bits) <= {"0"} for bits in counts)


@pytest.mark.parametrize("method", METHODS)
def test_raw_execute_keeps_original_sparse_classical_indices(method):
    circuit = QuantumCircuit()
    circuit.x(4)
    circuit.measure([(4, 7), (0, 2)])
    rows = _records(circuit.execute(config=_config(method), shots=37, return_samples=True), 37, 8)
    expected = np.zeros((37, 8), dtype=np.uint8)
    expected[:, 7] = 1
    np.testing.assert_array_equal(rows, expected)


@pytest.mark.parametrize("method", METHODS)
def test_raw_measurement_records_preserve_collapse_and_repeated_qubits(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(0, 2)])
    circuit.h(1)
    circuit.measure([(0, 0), (1, 3)])
    rows = _records(circuit.execute(config=_config(method), shots=509, return_samples=True), 509, 4)
    np.testing.assert_array_equal(rows[:, 0], rows[:, 2])
    assert not rows[:, 1].any()
    assert rows[:, 0].mean() == pytest.approx(0.5, abs=0.1)
    assert rows[:, 3].mean() == pytest.approx(0.5, abs=0.1)


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.ExtendedStabilizer,
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
])
def test_raw_non_clifford_fallback_preserves_interference(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.measure_all()
    rows = _records(circuit.execute(config=_config(method), shots=2048, return_samples=True), 2048, 1)
    assert rows[:, 0].mean() == pytest.approx((1 - math.cos(math.pi / 4)) / 2, abs=0.045)
    np.testing.assert_array_equal(
        rows, circuit.execute(config=_config(method), shots=2048, return_samples=True)["samples"])


@pytest.mark.parametrize("seed", [0, 23])
def test_tensor_histogram_fallback_shuffles_complete_records_and_keeps_ownership(seed):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(1, 2), (0, 0), (1, 3)])
    config = _config(maestro.SimulationType.TensorNetwork, seed=seed)
    counts = circuit.execute(config=config, shots=2048)["counts"]
    result = circuit.execute(config=config, shots=2048, return_samples=True)
    rows = _records(result, 2048, 4)
    np.testing.assert_array_equal(rows[:, 0], rows[:, 2])
    np.testing.assert_array_equal(rows[:, 0], rows[:, 3])
    assert not rows[:, 1].any()
    assert np.count_nonzero(np.any(rows[1:] != rows[:-1], axis=1)) > 100
    assert rows[:256, 0].mean() == pytest.approx((1 - math.cos(math.pi / 4)) / 2, abs=0.08)
    assert dict(Counter("".join(str(int(bit)) for bit in row) for row in rows)) == counts
    np.testing.assert_array_equal(
        rows, circuit.execute(config=config, shots=2048, return_samples=True)["samples"])
    del result, circuit
    gc.collect()
    assert rows.shape == (2048, 4) and rows.flags.c_contiguous
    np.testing.assert_array_equal(rows[:, 0], rows[:, 3])


def test_noisy_tensor_histogram_fallback_replays_each_realization():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(0, 0), (1, 1)])
    model = maestro.NoiseModel()
    model.set_dephasing(0, 0.2)
    config = _config(maestro.SimulationType.TensorNetwork, seed=0)
    kwargs = dict(config=config, shots=513, noise_realizations=3, noise_seed=0)
    result = circuit.full_noise_execute(model, return_samples=True, **kwargs)
    rows = _records(result, 513, 2)
    np.testing.assert_array_equal(rows[:, 0], rows[:, 1])
    assert np.count_nonzero(rows[1:, 0] != rows[:-1, 0]) > 32
    np.testing.assert_array_equal(rows, circuit.full_noise_execute(model, return_samples=True, **kwargs)["samples"])
    counts = circuit.full_noise_execute(model, **kwargs)["counts"]
    assert dict(Counter("".join(str(int(bit)) for bit in row) for row in rows)) == counts


TERMINAL_METHODS = [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
    maestro.SimulationType.ExtendedStabilizer,
    maestro.SimulationType.PathIntegral,
]
# H T H leaves |1> with this probability; T keeps the circuit off the frame path.
T_FLIP = (1 - math.cos(math.pi / 4)) / 2


@pytest.mark.parametrize("method", TERMINAL_METHODS)
def test_raw_terminal_measurements_draw_rows_from_one_prepared_state(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.measure([(0, 2)])
    circuit.h(1)
    circuit.cx(1, 3)
    # Qubit 0 is measured twice with nothing in between: both records agree.
    circuit.measure([(0, 0), (1, 1), (3, 3)])
    result = circuit.execute(config=_config(method), shots=4000, return_samples=True)
    assert result["sampling_strategy"] == "terminal_v1"
    rows = _records(result, 4000, 4)
    np.testing.assert_array_equal(rows[:, 0], rows[:, 2])
    np.testing.assert_array_equal(rows[:, 1], rows[:, 3])
    assert rows[:, 0].mean() == pytest.approx(T_FLIP, abs=0.03)
    assert rows[:, 1].mean() == pytest.approx(0.5, abs=0.04)
    # A histogram expansion would group outcomes; every leading slice of the
    # seeded row order is a uniform subsample instead.
    assert rows[:500, 1].mean() == pytest.approx(0.5, abs=0.1)
    assert rows[:500, 0].any() and not rows[:500, 0].all()
    np.testing.assert_array_equal(
        rows, circuit.execute(config=_config(method), shots=4000, return_samples=True)["samples"])
    assert not np.array_equal(
        rows, circuit.execute(config=_config(method, seed=24), shots=4000, return_samples=True)["samples"])


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.MatrixProductState])
def test_raw_terminal_readout_is_drawn_per_row(method):
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.t(0)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_readout_error(0, 0.0, 0.3)
    for realizations in [1, 4]:
        result = circuit.noisy_execute(model, config=_config(method, workers=4), shots=3000,
                                       noise_realizations=realizations, noise_seed=8, return_samples=True)
        assert result["sampling_strategy"] == "terminal_v1"
        rows = _records(result, 3000, 1)
        assert rows[:, 0].mean() == pytest.approx(0.7, abs=0.04)


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix])
def test_raw_mid_circuit_measurement_and_single_shot_keep_trajectories(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.measure([(0, 0)])
    circuit.h(0)
    circuit.measure([(0, 1)])
    result = circuit.execute(config=_config(method), shots=64, return_samples=True)
    assert result["sampling_strategy"] == "trajectory_v1"
    _records(result, 64, 2)
    terminal = QuantumCircuit()
    terminal.h(0)
    terminal.t(0)
    terminal.measure_all()
    one = terminal.execute(config=_config(method), shots=1, return_samples=True)
    assert one["sampling_strategy"] == "trajectory_v1"
    _records(one, 1, 1)


@pytest.mark.parametrize("method", METHODS)
def test_raw_noise_samples_each_circuit_pauli_channel_independently(method):
    circuit = QuantumCircuit()
    circuit.pauli_channel([0], [0.65, 0.35, 0.0, 0.0])
    circuit.measure_all()
    kwargs = dict(config=_config(method, workers=4), shots=2048, noise_realizations=2048,
                  noise_seed=31, return_samples=True)
    result = circuit.full_noise_execute(maestro.NoiseModel(), **kwargs)
    rows = _records(result, 2048, 1)
    assert rows[:, 0].mean() == pytest.approx(0.35, abs=0.05)
    np.testing.assert_array_equal(
        rows, maestro.full_noise_execute(circuit, maestro.NoiseModel(), **kwargs)["samples"])


def test_frame_policy_does_not_change_shared_realization_semantics():
    circuit = QuantumCircuit()
    circuit.pauli_channel([0], [0.5, 0.5, 0.0, 0.0])
    circuit.measure_all()
    kwargs = dict(config=_config(maestro.SimulationType.Stabilizer, workers=4),
                  shots=137, noise_seed=9, return_samples=True)
    shared = circuit.full_noise_execute(maestro.NoiseModel(), noise_realizations=1, **kwargs)
    assert np.unique(_records(shared, 137, 1)).size == 1
    independent = circuit.full_noise_execute(maestro.NoiseModel(), noise_realizations=137, **kwargs)
    assert np.unique(_records(independent, 137, 1)).size == 2


@pytest.mark.parametrize("method", [maestro.SimulationType.Stabilizer, maestro.SimulationType.Statevector])
@pytest.mark.parametrize("config_seed,noise_seed", [
    (None, 0), (None, 17), (0, None), (17, None), ((1 << 64) - 1, None), (17, 0),
])
def test_raw_circuit_channel_with_empty_model_replays_seed_sources(method, config_seed, noise_seed):
    circuit = QuantumCircuit()
    circuit.pauli_channel([0], [0.5, 0.5, 0.0, 0.0])
    circuit.measure_all()
    config = _config(method, workers=4, seed=config_seed)
    kwargs = dict(config=config, shots=257, noise_realizations=257,
                  noise_seed=noise_seed, return_samples=True)
    first = _records(circuit.full_noise_execute(maestro.NoiseModel(), **kwargs), 257, 1)
    replay = circuit.full_noise_execute(maestro.NoiseModel(), **kwargs)["samples"]
    np.testing.assert_array_equal(first, replay)
    assert 70 < first[:, 0].sum() < 185
    assert config.seed == config_seed


@pytest.mark.parametrize("method", [maestro.SimulationType.Stabilizer, maestro.SimulationType.Statevector])
def test_raw_empty_model_circuit_noise_keeps_alias_and_unseeded_independence(method):
    circuit = QuantumCircuit()
    circuit.pauli_channel([0], [0.5, 0.5, 0.0, 0.0])
    circuit.measure_all()
    config = _config(method, workers=4, seed=None)
    kwargs = dict(config=config, shots=257, noise_realizations=257, return_samples=True)
    expected = circuit.full_noise_execute(maestro.NoiseModel(), noise_seed=0, **kwargs)["samples"]
    aliased = circuit.full_noise_execute(maestro.NoiseModel(), seed=0, **kwargs)["samples"]
    np.testing.assert_array_equal(expected, aliased)
    first = circuit.full_noise_execute(maestro.NoiseModel(), **kwargs)["samples"]
    second = circuit.full_noise_execute(maestro.NoiseModel(), **kwargs)["samples"]
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("method", [maestro.SimulationType.DensityMatrix, maestro.SimulationType.MatrixProductOperator])
def test_plain_raw_exact_circuit_channel_repeats_mixed_state_per_shot(method):
    circuit = QuantumCircuit()
    circuit.pauli_channel([0], [0.6, 0.4, 0.0, 0.0])
    circuit.measure_all()
    rows = _records(circuit.execute(config=_config(method), shots=2048, return_samples=True), 2048, 1)
    assert rows[:, 0].mean() == pytest.approx(0.4, abs=0.05)


@pytest.mark.parametrize("method", [
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
])
def test_raw_exact_channels_keep_the_ensemble_with_one_realization(method):
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_t1(0, 0.4)
    rows = _records(circuit.noisy_execute(
        model, config=_config(method), shots=2048, noise_realizations=1,
        noise_seed=17, return_samples=True), 2048, 1)
    assert rows[:, 0].mean() == pytest.approx(0.6, abs=0.05)


@pytest.mark.parametrize("return_samples", [False, True])
def test_general_kraus_noise_still_requires_an_exact_backend(return_samples):
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_kraus_channel([0], [
        [[1.0, 0.0], [0.0, math.sqrt(0.6)]],
        [[0.0, math.sqrt(0.4)], [0.0, 0.0]],
    ])
    for method in [maestro.SimulationType.Stabilizer, maestro.SimulationType.Statevector]:
        with pytest.raises(ValueError, match="density-matrix|MPO"):
            circuit.full_noise_execute(model, config=_config(method), shots=17,
                                       noise_realizations=17, return_samples=return_samples)


@pytest.mark.parametrize("method", METHODS)
def test_noisy_counts_still_reproduce_across_realization_worker_counts(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(0, 0)])
    circuit.h(1)
    circuit.measure([(0, 1), (1, 2)])
    model = maestro.NoiseModel()
    model.set_all_depolarizing(2, 0.17)
    model.set_readout_error_symmetric(0, 0.2)
    kwargs = dict(shots=139, noise_realizations=139, noise_seed=7)
    serial = circuit.full_noise_execute(model, config=_config(method), **kwargs)
    parallel = circuit.full_noise_execute(model, config=_config(method, workers=4), **kwargs)
    assert serial["counts"] == parallel["counts"]
    assert sum(serial["counts"].values()) == 139
    assert "samples" not in parallel


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
])
@pytest.mark.parametrize("realizations", [1, 17, 137])
def test_raw_trajectory_rows_are_reproducible_across_worker_counts(method, realizations):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([(0, 0)])
    circuit.h(1)
    circuit.measure([(0, 1), (1, 2)])
    model = maestro.NoiseModel()
    model.set_all_depolarizing(2, 0.13)
    model.set_coherent_rotation(1, 0.11, 0.07, -0.09)
    model.set_readout_error_symmetric(0, 0.2)
    kwargs = dict(shots=137, noise_realizations=realizations, noise_seed=3, return_samples=True)
    serial = _records(circuit.full_noise_execute(model, config=_config(method), **kwargs), 137, 3)
    parallel = _records(circuit.full_noise_execute(model, config=_config(method, workers=4), **kwargs), 137, 3)
    np.testing.assert_array_equal(parallel, serial)


@pytest.mark.parametrize("method", [maestro.SimulationType.Stabilizer, maestro.SimulationType.Statevector])
def test_raw_readout_flip_precedes_classical_control(method):
    circuit = maestro.QasmToCirc().parse_and_translate(
        "OPENQASM 3.0; qubit[2] q; bit[4] c; "
        "x q[1]; c[0] = measure q[0]; if (c[0]) { x q[1]; } c[3] = measure q[1];")
    model = maestro.NoiseModel()
    model.set_readout_error_symmetric(0, 1.0)
    rows = _records(circuit.noisy_execute(
        model, config=_config(method, workers=4), shots=71, noise_realizations=71,
        noise_seed=3, return_samples=True), 71, 4)
    np.testing.assert_array_equal(rows, np.tile([1, 0, 0, 0], (71, 1)))
    clean = circuit.execute(config=_config(method), shots=7, return_samples=True)["samples"]
    np.testing.assert_array_equal(clean, np.tile([0, 0, 0, 1], (7, 1)))


@pytest.mark.parametrize("return_samples", [False, True])
def test_raw_option_keeps_noise_layer_and_argument_validation(return_samples):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.measure_all()
    coherent = maestro.NoiseModel()
    coherent.set_coherent_rotation(0, 0.1, 0.0, 0.0)
    pauli = maestro.NoiseModel()
    pauli.set_bit_flip(0, 0.1)
    with pytest.raises(ValueError, match="coherent"):
        circuit.noisy_execute(coherent, shots=5, return_samples=return_samples)
    with pytest.raises(ValueError, match="coherent"):
        circuit.coherent_execute(pauli, shots=5, return_samples=return_samples)
    for invalid in [dict(shots=0), dict(noise_realizations=0), dict(noise_seed=1 << 32)]:
        with pytest.raises(ValueError):
            circuit.noisy_execute(pauli, return_samples=return_samples, **invalid)


@pytest.mark.parametrize("method", [maestro.SimulationType.MatrixProductState, maestro.SimulationType.MatrixProductOperator])
def test_raw_tensor_chain_samples_use_the_routed_executor(method):
    # MPS/MPO raw samples come from the network executor, which places the
    # initial qubit map and swap routing, and report its bond dimension.
    circuit = QuantumCircuit()
    for q in range(6):
        circuit.h(q)
    for q in range(0, 6, 2):
        circuit.cx(q, 5 - q if 5 - q != q else (q + 1) % 6)
    circuit.t(2)
    circuit.measure_all()
    config = maestro.SimulatorConfig(simulation_type=method, seed=5, max_bond_dimension=4)
    raw = circuit.execute(config=config, shots=256, return_samples=True)
    assert raw["samples"].shape == (256, 6)
    # Terminal measurements: the executor samples one prepared state.
    assert raw["sampling_strategy"] == "terminal_v1"
    assert raw.get("max_bond_dim_reached", 0) >= 1
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.05)
    noisy = circuit.noisy_execute(model, config=config, shots=64, noise_realizations=64, return_samples=True)
    assert noisy["samples"].shape == (64, 6) and noisy.get("max_bond_dim_reached", 0) >= 1


def test_public_reference_frame_sampler_reuses_one_compilation():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_bit_flip(1, 0.2)
    sampler = maestro.compile_reference_sampler(circuit, model)
    assert isinstance(sampler, maestro.ReferenceFrameSampler)
    assert maestro._compile_reference_sampler is not None
    assert (sampler.num_qubits, sampler.num_clbits) == (2, 2)
    first = sampler.sample(2000, seed=11)
    assert first["sampling_strategy"] == "reference_frame_v1"
    rows = _records(first, 2000, 2)
    np.testing.assert_array_equal(rows, sampler.sample(2000, seed=11)["samples"])
    assert not np.array_equal(rows, sampler.sample(2000, seed=12)["samples"])
    # Only the flip after the CX on qubit 1 separates the pair.
    assert 0.15 < np.mean(rows[:, 0] != rows[:, 1]) < 0.25
    counts = sampler.sample_counts(2000, seed=11)["counts"]
    assert counts == dict(Counter("".join(map(str, row)) for row in rows))
    assert sampler.sample(5)["samples"].shape == (5, 2)
    assert sampler.sample(0)["samples"].shape == (0, 2)
    with pytest.raises(ValueError):
        sampler.sample(-1)
    # A noiseless Clifford circuit compiles without a model.
    assert maestro.compile_reference_sampler(circuit) is not None


def test_public_reference_frame_sampler_rejects_what_frames_cannot_represent():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.measure_all()
    assert maestro.compile_reference_sampler(circuit) is None
    clifford = QuantumCircuit()
    clifford.h(0)
    clifford.measure_all()
    coherent = maestro.NoiseModel()
    coherent.set_coherent_rotation(0, 0.1, 0.0, 0.0)
    assert maestro.compile_reference_sampler(clifford, coherent) is None
    gpu = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.Gpu)
    assert maestro.compile_reference_sampler(clifford, None, gpu) is None


def test_noisy_reference_frames_keep_the_scalar_seed_and_thermal_rules():
    """noise_seed drives circuit errors and config.seed the outcomes; clamped thermal noise keeps trajectories."""
    flipped = QuantumCircuit()
    flipped.x(0)
    flipped.measure([(0, 0)])
    flip = maestro.NoiseModel()
    flip.set_bit_flip(0, 0.3)

    def rows(circuit, model, seed, noise_seed):
        result = circuit.noisy_execute(model, config=_config(maestro.SimulationType.Stabilizer, seed=seed), shots=200, noise_realizations=200,
                                       noise_seed=noise_seed, return_samples=True)
        assert result["sampling_strategy"] == "reference_frame_v1"
        return result["samples"]

    # A deterministic measurement: only the noise seed decides a row.
    assert np.array_equal(rows(flipped, flip, 1, 7), rows(flipped, flip, 2, 7))
    assert not np.array_equal(rows(flipped, flip, 1, 7), rows(flipped, flip, 1, 8))
    # A random measurement with noise elsewhere: only the measurement seed.
    random = QuantumCircuit()
    random.h(0)
    random.x(1)
    random.measure([(0, 0)])
    aside = maestro.NoiseModel()
    aside.set_bit_flip(1, 0.3)
    assert np.array_equal(rows(random, aside, 1, 7), rows(random, aside, 1, 8))
    assert not np.array_equal(rows(random, aside, 1, 7), rows(random, aside, 2, 7))
    # Trajectories clamp T2 > T1 to T1; exact channels on frames would not.
    clifford = QuantumCircuit()
    clifford.h(0)
    clifford.s(0)
    clifford.h(0)
    clifford.measure([(0, 0)])
    exact = maestro.NoiseModel()
    exact.set_thermal_relaxation(0, 1.0, 1.0, 1.0, 0.5)
    rows(clifford, exact, 3, 3)
    window = maestro.NoiseModel()
    window.set_thermal_relaxation(0, 1.0, 1.0, 2.0, 0.5)
    result = clifford.noisy_execute(window, config=_config(maestro.SimulationType.Stabilizer, seed=3), shots=200, noise_realizations=200,
                                    noise_seed=3, return_samples=True)
    assert result.get("sampling_strategy") != "reference_frame_v1"


@pytest.mark.parametrize("method", [maestro.SimulationType.Stabilizer, maestro.SimulationType.Statevector])
def test_rows_follow_sampling_reduction_like_counts(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.h(2)
    circuit.cx(2, 3)
    circuit.measure_all()
    config = _config(method, seed=3)
    config.enable_sampling_reduction = True
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.0)
    for result in (circuit.execute(config=config, shots=64, return_samples=True),
                   circuit.noisy_execute(model, config=config, shots=64, noise_realizations=64, return_samples=True)):
        rows = _records(result, 64, 4)
        assert result["sampling_components"] == 2
        assert result["sampling_strategy"] != "reference_frame_v1"
        assert (rows[:, 0] == rows[:, 1]).all() and (rows[:, 2] == rows[:, 3]).all()
    assert circuit.execute(config=config, shots=64)["sampling_components"] == 2
