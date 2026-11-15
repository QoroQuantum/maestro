"""Batched extended-stabilizer Pauli frames (sampling_strategy extended_frame_v1)."""

import warnings
from collections import Counter

import numpy as np
import pytest

import maestro
from maestro.circuits import QuantumCircuit


def _config(seed=None, policy="auto"):
    extra = {} if seed is None else {"seed": seed}
    return maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QCSim,
                                   simulation_type=maestro.SimulationType.ExtendedStabilizer,
                                   counts_policy=policy, **extra)


def _circuit():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.cx(0, 1)
    circuit.h(1)
    circuit.t(1)
    circuit.h(1)
    circuit.measure([(0, 0), (1, 1)])
    return circuit


def _model():
    model = maestro.NoiseModel()
    for qubit in range(2):
        model.set_depolarizing(qubit, 0.02)
        model.set_readout_error(qubit, 0.02, 0.03)
    return model


def _assert_same_distribution(a, b, shots):
    """Every outcome frequency agrees within six standard errors."""
    for key in set(a) | set(b):
        p, q = a.get(key, 0) / shots, b.get(key, 0) / shots
        sigma = np.sqrt(max(p * (1 - p) + q * (1 - q), 1e-9) / shots)
        assert abs(p - q) < 6 * sigma + 1e-3, (key, p, q)


@pytest.mark.parametrize("return_samples", [False, True])
@pytest.mark.parametrize("realizations", [None, 500])
def test_unseeded_noisy_execute_runs_on_frames(return_samples, realizations):
    kwargs = {} if realizations is None else {"noise_realizations": realizations}
    result = _circuit().noisy_execute(_model(), config=_config(), shots=500, return_samples=return_samples, **kwargs)
    assert result["sampling_strategy"] == "extended_frame_v1"
    assert result["method"] == maestro.SimulationType.ExtendedStabilizer
    if return_samples:
        assert result["samples"].shape == (500, 2)
    else:
        assert sum(result["counts"].values()) == 500


@pytest.mark.parametrize("policy", ["auto", "frame_v1"])
@pytest.mark.parametrize("seed", [0, 4])
def test_seeded_runs_use_reproducible_frames(policy, seed):
    first = _circuit().noisy_execute(_model(), config=_config(seed=seed, policy=policy), shots=200)
    second = _circuit().noisy_execute(_model(), config=_config(seed=seed, policy=policy), shots=200)
    assert first["sampling_strategy"] == "extended_frame_v1"
    assert first["counts"] == second["counts"]
    legacy = _circuit().noisy_execute(_model(), config=_config(policy="legacy"), shots=50)
    assert legacy.get("sampling_strategy") != "extended_frame_v1"


def test_full_noise_execute_runs_on_frames():
    result = _circuit().full_noise_execute(_model(), config=_config(), shots=300)
    assert result["sampling_strategy"] == "extended_frame_v1"


def test_frames_match_scalar_trajectories():
    shots = 20000
    frames = _circuit().noisy_execute(_model(), config=_config(), shots=shots, noise_realizations=shots)
    scalar = _circuit().noisy_execute(_model(), config=_config(seed=11, policy="legacy"), shots=shots, noise_realizations=shots)
    assert frames["sampling_strategy"] == "extended_frame_v1"
    _assert_same_distribution(frames["counts"], scalar["counts"], shots)


def test_compiled_extended_frame_sampler():
    sampler = maestro.compile_reference_sampler(_circuit(), _model(), _config())
    assert type(sampler).__name__ == "ExtendedFrameSampler"
    assert (sampler.num_qubits, sampler.num_clbits) == (2, 2)
    rows = sampler.sample(1000, 4)
    assert rows["sampling_strategy"] == "extended_frame_v1"
    assert rows["samples"].shape == (1000, 2)
    assert np.array_equal(rows["samples"], sampler.sample(1000, 4)["samples"])
    counts = sampler.sample_counts(1000, 4)["counts"]
    assert counts == dict(Counter("".join(str(bit) for bit in row) for row in rows["samples"]))
    shared = sampler.sample(1000, 4, noise_realizations=10)
    assert shared["noise_realizations"] == 10 and shared["samples"].shape == (1000, 2)
    with pytest.raises(ValueError):
        sampler.sample(10, 4, noise_realizations=0)
    # Sinter's contract: _compile_reference_sampler(...).sample(shots, seed)["samples"].
    assert maestro._compile_reference_sampler(_circuit(), maestro.NoiseModel(), _config()).sample(10, 3)["samples"].shape == (10, 2)
    # Other methods keep their earlier answer for a non-Clifford circuit.
    assert maestro.compile_reference_sampler(_circuit(), _model()) is None


def test_shared_thermal_realizations_keep_scalar_trajectories():
    """Sampled relaxation is a reset shared by a realization's shots; frames would draw Paulis."""
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.measure([(0, 0)])
    model = maestro.NoiseModel()
    model.set_thermal_relaxation(0, 1.0, 0.05, 0.05, 0.5)  # a reset in all but e^-20 of realizations
    shared = circuit.noisy_execute(model, config=_config(), shots=1000, noise_realizations=1)
    assert shared.get("sampling_strategy") != "extended_frame_v1"
    assert len(shared["counts"]) == 1  # the realization's reset fixes every shot
    per_shot = circuit.noisy_execute(model, config=_config(), shots=1000, noise_realizations=1000)
    assert per_shot["sampling_strategy"] == "extended_frame_v1"
    clamped = maestro.NoiseModel()
    clamped.set_thermal_relaxation(0, 0.1, 1.0, 1.9, 0.5)
    assert circuit.noisy_execute(clamped, config=_config(), shots=100, noise_realizations=100).get("sampling_strategy") != "extended_frame_v1"


def test_noise_seed_alone_drives_circuit_errors():
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.t(0)
    circuit.measure([(0, 0)])
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.3)

    def rows(seed, noise_seed):
        result = circuit.noisy_execute(model, config=_config(seed=seed, policy="frame_v1"), shots=200, noise_realizations=200,
                                       noise_seed=noise_seed, return_samples=True)
        assert result["sampling_strategy"] == "extended_frame_v1"
        return result["samples"]

    assert np.array_equal(rows(1, 7), rows(2, 7))
    assert not np.array_equal(rows(1, 7), rows(1, 8))


def _budget_config(megabytes, workers=None):
    extra = {} if workers is None else {"max_simulators": workers}
    return maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QCSim, simulation_type=maestro.SimulationType.ExtendedStabilizer,
                                   trajectory_max_memory_mb=megabytes, **extra)


def _rotations(qubits):
    circuit = QuantumCircuit()
    for qubit in range(qubits):
        circuit.ry(qubit, 1.2)
    circuit.measure([(qubit, qubit) for qubit in range(qubits)])
    return circuit


def test_budgets_are_honored_and_workers_do_not_change_rows():
    circuit = _rotations(12)
    model = maestro.NoiseModel()
    model.set_depolarizing(0, 0.01)
    # 1 MiB fits split batches; their rows are those of whole batches.
    tiny = circuit.noisy_execute(model, config=_budget_config(1), shots=512, noise_realizations=512)
    assert tiny["sampling_strategy"] == "extended_frame_v1"
    samples = [maestro.compile_reference_sampler(circuit, model, _budget_config(budget, workers)).sample(512, 5)["samples"]
               for budget, workers in ((128, 1), (128, 2), (128, 8), (1, 1), (1, 4))]
    assert all(np.array_equal(samples[0], other) for other in samples[1:])
    # A single shot above the budget: the budget is soft, as for trajectories,
    # so the shot stays on frames with the rows of an ample budget. Compiled
    # samplers (which maestro.sinter uses) and execution agree.
    wide = _rotations(17)
    compiled = [maestro._compile_reference_sampler(wide, model, _budget_config(megabytes)).sample(2, 5)["samples"] for megabytes in (1, 128)]
    assert np.array_equal(compiled[0], compiled[1])

    def wide_rows(megabytes):
        config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QCSim, simulation_type=maestro.SimulationType.ExtendedStabilizer,
                                         trajectory_max_memory_mb=megabytes, counts_policy="frame_v1", seed=3)
        result = wide.noisy_execute(model, config=config, shots=2, noise_realizations=2, noise_seed=3, return_samples=True)
        assert result["sampling_strategy"] == "extended_frame_v1"
        return result["samples"]

    assert np.array_equal(wide_rows(1), wide_rows(128))


def _controlled_ry(circuit, control, target, angle):
    circuit.ry(target, angle / 2)
    circuit.cx(control, target)
    circuit.ry(target, -angle / 2)
    circuit.cx(control, target)


def test_memory_retries_keep_every_draw():
    """An outcome that costs memory must not be redrawn when a batch splits."""
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.measure([(0, 0)])
    for target in range(1, 13):  # 4,096 amplitudes per shot only after a one
        _controlled_ry(circuit, 0, target, 0.8)
    circuit.measure([(target, target) for target in range(1, 13)])
    small = maestro.compile_reference_sampler(circuit, None, _budget_config(1))
    ample = maestro.compile_reference_sampler(circuit, None, _budget_config(128))
    ones = 0
    for seed in range(120):
        row = small.sample(1, seed)["samples"]
        assert np.array_equal(row, ample.sample(1, seed)["samples"])
        ones += int(row[0, 0])
    assert 35 < ones < 85  # one half, not the quarter that redraws gave
    for workers in (1, 4):
        assert np.array_equal(maestro.compile_reference_sampler(circuit, None, _budget_config(1, workers)).sample(600, 3)["samples"],
                              ample.sample(600, 3)["samples"])


def test_oversize_shots_keep_their_draws():
    """A shot whose outcome exceeds the budget must not be redrawn on another method."""
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.cx(0, 1)  # qubit 1 copies a 50% flip of qubit 0
    for target in range(2, 18):  # 2^16 amplitudes only after a one
        _controlled_ry(circuit, 1, target, 0.8)
    circuit.measure([(1, 0)])
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.5)

    def row(megabytes, seed):
        config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.QCSim, simulation_type=maestro.SimulationType.ExtendedStabilizer,
                                         trajectory_max_memory_mb=megabytes, counts_policy="frame_v1", seed=seed)
        result = circuit.noisy_execute(model, config=config, shots=1, noise_realizations=1, noise_seed=seed, return_samples=True)
        assert result["sampling_strategy"] == "extended_frame_v1"
        return int(result["samples"][0, 0])

    ones = 0
    for seed in range(40):
        bit = row(1, seed)
        assert bit == row(128, seed)
        ones += bit
    assert 8 < ones < 32  # one half, not the quarter a redraw gave


def test_compiled_sampler_refuses_shared_realizations_with_resets():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.measure([(0, 0)])
    relaxing = maestro.NoiseModel()
    relaxing.set_thermal_relaxation(0, 1.0, 0.2, 0.2, 0.5)  # a Pauli mixture that noisy_execute draws as a reset
    sampler = maestro.compile_reference_sampler(circuit, relaxing, _config())
    assert type(sampler).__name__ == "ExtendedFrameSampler"
    for call in (sampler.sample, sampler.sample_counts):
        with pytest.raises(ValueError):
            call(100, 1, noise_realizations=1)
        call(100, 1)
        call(100, 1, noise_realizations=100)
    # Infinite T1 never resets: shared realizations stay on frames everywhere.
    dephasing = maestro.NoiseModel()
    dephasing.set_thermal_relaxation(0, 1.0, float("inf"), 2.0)
    assert maestro.compile_reference_sampler(circuit, dephasing, _config()).sample(100, 1, noise_realizations=4)["noise_realizations"] == 4
    shared = circuit.noisy_execute(dephasing, config=_config(), shots=1000, noise_realizations=8)
    assert shared["sampling_strategy"] == "extended_frame_v1"
    # Relaxation that never resets this circuit does not count either.
    never = []
    for setup in (lambda m: m.set_t1(0, 0.0), lambda m: m.set_t1_from_time(0, 1.0, float("inf")), lambda m: m.set_idle_noise(0, 1.0, 1.0),
                  lambda m: m.set_thermal_relaxation(5, 1.0, 0.2, 0.2, 0.5)):
        model = maestro.NoiseModel()
        setup(model)
        model.set_bit_flip(0, 0.01)
        never.append(model)
    for model in never:
        assert maestro.compile_reference_sampler(circuit, model, _config()).sample_counts(100, 1, noise_realizations=1)["noise_realizations"] == 1
        assert circuit.noisy_execute(model, config=_config(), shots=1000, noise_realizations=8)["sampling_strategy"] == "extended_frame_v1"


def test_clamp_warnings_follow_the_circuit():
    """Only relaxation the circuit draws in the T1 < T2 <= 2 T1 window is clamped, and warned about."""
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.h(0)
    circuit.measure([(0, 0)])
    idle = maestro.NoiseModel()
    idle.set_idle_noise(0, 1.0, 2.0)
    idle.set_bit_flip(0, 0.01)
    instant = maestro.NoiseModel()
    instant.set_thermal_relaxation(0, 0.0, 1.0, 2.0, 0.5)  # zero duration: the identity
    instant.set_bit_flip(0, 0.01)
    for model in (idle, instant):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert circuit.noisy_execute(model, config=_config(), shots=200)["sampling_strategy"] == "extended_frame_v1"
    delayed = QuantumCircuit()
    delayed.h(0)
    delayed.delay(0, 0.5)
    delayed.h(0)
    delayed.measure([(0, 0)])
    with pytest.warns(RuntimeWarning, match="clamped to T1"):
        delayed.noisy_execute(idle, config=_config(), shots=200)


def test_repeated_operands_raise():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.t(0)
    circuit.cx(0, 0)
    circuit.measure([(0, 0)])
    with pytest.raises(ValueError, match="distinct"):
        maestro.compile_reference_sampler(circuit, None, _config())
    with pytest.raises(ValueError, match="distinct"):
        circuit.noisy_execute(_model(), config=_config(), shots=10)


def test_checkpoint_snapshots_its_prefix():
    prefix = QuantumCircuit()
    prefix.x(0)
    checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 1, _config())
    prefix.x(0)  # edited after the state was saved
    suffix = QuantumCircuit()
    suffix.measure([(0, 0)])
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 1e-9)
    noisy = checkpoint.execute_suffix(suffix, shots=100, noise_model=model, noise_realizations=100)
    assert noisy["sampling_strategy"] == "extended_frame_v1"
    assert noisy["counts"] == {"1": 100}
    assert checkpoint.execute_suffix(suffix, shots=100)["counts"] == {"1": 100}


def test_checkpoint_suffixes_run_on_frames():
    prefix = QuantumCircuit()
    prefix.h(0)
    prefix.t(0)
    prefix.cx(0, 1)
    suffix = QuantumCircuit()
    suffix.h(1)
    suffix.t(1)
    suffix.h(1)
    suffix.measure([(0, 0), (1, 1)])
    shots = 20000
    frames = maestro.PrefixCheckpointedSimulator(prefix, 2, _config()).execute_suffix(
        suffix, shots=shots, noise_model=_model(), noise_realizations=shots)
    scalar = maestro.PrefixCheckpointedSimulator(prefix, 2, _config(policy="legacy")).execute_suffix(
        suffix, shots=shots, noise_model=_model(), noise_realizations=shots, noise_seed=8)
    assert frames["sampling_strategy"] == "extended_frame_v1"
    assert scalar["sampling_strategy"] != "extended_frame_v1"
    _assert_same_distribution(frames["counts"], scalar["counts"], shots)
    rows = maestro.PrefixCheckpointedSimulator(prefix, 2, _config()).execute_suffix(
        suffix, shots=100, noise_model=_model(), return_samples=True)
    assert rows["sampling_strategy"] == "extended_frame_v1" and rows["samples"].shape == (100, 2)
    seeded = maestro.PrefixCheckpointedSimulator(prefix, 2, _config(seed=3)).execute_suffix(suffix, shots=100, noise_model=_model())
    assert seeded["sampling_strategy"] == "extended_frame_v1"
    replay = maestro.PrefixCheckpointedSimulator(prefix, 2, _config(seed=3)).execute_suffix(suffix, shots=100, noise_model=_model())
    assert seeded["counts"] == replay["counts"]
    # The end measurements of a prefix are removed, since only suffixes are
    # sampled: such a prefix is still gates only and runs on frames.
    measured = QuantumCircuit()
    measured.h(0)
    measured.measure([(0, 0)])
    ended = maestro.PrefixCheckpointedSimulator(measured, 2, _config()).execute_suffix(suffix, shots=100, noise_model=_model())
    assert ended["sampling_strategy"] == "extended_frame_v1"
    # A measurement that a later gate of the prefix follows runs, and
    # collapses once for every suffix shot: no frames.
    measured.h(0)
    collapsed = maestro.PrefixCheckpointedSimulator(measured, 2, _config()).execute_suffix(suffix, shots=100, noise_model=_model())
    assert collapsed["sampling_strategy"] != "extended_frame_v1"
