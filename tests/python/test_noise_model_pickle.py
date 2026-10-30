"""NoiseModel call logs preserve successful mutations for pickle and replay."""

import copy
import pickle

import pytest

import maestro


# Every public operation must be classified here so a new mutator with a name
# such as add_* cannot silently escape the serialization regression tests.
MUTATING_CALLS = {
    "set_qubit_noise": (0, 0.01, 0.02, 0.03),
    "set_depolarizing": (0, 0.1),
    "set_dephasing": (0, 0.1),
    "set_bit_flip": (0, 0.1),
    "set_all_depolarizing": (2, 0.1),
    "set_all_dephasing": (2, 0.1),
    "set_coherent_rotation": (0, 0.01, 0.02, 0.03),
    "set_coherent_depolarizing": (0, 0.1),
    "set_coherent_dephasing": (0, 0.1),
    "set_coherent_bit_flip": (0, 0.1),
    "set_all_coherent_depolarizing": (2, 0.1),
    "set_all_coherent_dephasing": (2, 0.1),
    "set_coherent_strength": (2, 0.1),
    "set_correlated_ar1": (0, 0.5, 0.01),
    "set_correlated_ou": (0, 15.0, 0.5, 1e-7),
    "set_correlated_ou_band": (0, 15.0, 0.5, 1e-7),
    "set_multi_correlated_ou": (0, [(15.0, 0.5), (20.0, 1.0)], 1e-7),
    "set_all_multi_correlated_ou": (2, [(15.0, 0.5)], 1e-7),
    "set_1_over_f_noise": (0, 0.001, 1.0, 100.0, 3, 1e-7),
    "set_all_correlated_ou": (2, 15.0, 0.5, 1e-7),
    "set_all_correlated_from_power": (2, 0.001, 0.5, 1e-7),
    "set_idle_noise": (0, 1e-4, 1e-4),
    "set_t1": (0, 0.1),
    "set_all_t1": (2, 0.1),
    "set_t1_from_time": (0, 1e-7, 1e-4),
    "set_phase_damping": (0, 0.1),
    "set_phase_damping_from_time": (0, 1e-7, 1e-4),
    "set_generalized_amplitude_damping": (0, 0.1, 0.2),
    "set_thermal_relaxation": (0, 1e-7, 1e-4, 1e-4),
    "set_thermal_relaxation_2q": (0, 1e-7, 1e-4, 1e-4),
    "set_correlated_phase_flip": (0, 1, 0.1, 0.5),
    "set_kraus_channel": ([0], [[[0, 1], [1, 0]]]),
    "set_t1_2q": (0, 0.1),
    "set_t1_2q_from_time": (0, 1e-7, 1e-4),
    "set_crosstalk": (0, 1, 0.01),
    "set_readout_error": (0, 0.1, 0.2),
    "set_readout_error_symmetric": (0, 0.1),
    "set_all_readout_error": (2, 0.1),
    "set_2q_depolarizing": (0, 1, 0.1),
    "set_1q_gate_depolarizing": (0, 0.1),
    "set_2q_gate_depolarizing": (0, 0.1),
    "set_all_1q_gate_depolarizing": (2, 0.1),
    "set_all_2q_gate_depolarizing": (2, 0.1),
}

QUERY_CALLS = {
    "compute_damping": ("XX",),
    "has_coherent": (),
    "has_correlated": (),
    "has_idle_noise": (),
    "has_t1": (),
    "has_thermal_relaxation": (),
    "has_additional_quantum_channels": (),
    "has_thermal_in_sampled_overdephasing_regime": (),
    "requires_exact_quantum_channels": (),
    "compute_damping_covers_model": (),
    "get_t1_2q": (0,),
    "get_t1_for_gate": (0, False),
    "has_crosstalk": (),
    "has_readout_error": (),
    "has_any_2q_depolarizing": (),
    "has_1q_gate_noise": (),
    "has_2q_gate_noise": (),
    "has_any": (),
}


def replay_call_log(call_log):
    """Reconstruct native state directly, independently of pickle's helper."""
    model = maestro._raw_maestro.NoiseModel()
    for name, args, kwargs in call_log:
        getattr(model, name)(*args, **kwargs)
    return model


def query_state(model):
    return {name: getattr(model, name)(*args) for name, args in QUERY_CALLS.items()}


def expectation(model, circuit, observable):
    config = maestro.SimulatorConfig(
        simulation_type=maestro.SimulationType.DensityMatrix)
    return maestro.full_noise_estimate(
        circuit, observable, model, noise_realizations=4,
        config=config, noise_seed=7)["expectation_values"][0]


def test_public_noise_model_operations_have_serialization_coverage():
    public_methods = {
        name for name in dir(maestro._raw_maestro.NoiseModel)
        if not name.startswith("_")
        and callable(getattr(maestro._raw_maestro.NoiseModel, name))
    }
    assert public_methods == MUTATING_CALLS.keys() | QUERY_CALLS.keys()


@pytest.mark.parametrize("name,args", MUTATING_CALLS.items(), ids=MUTATING_CALLS)
def test_every_mutator_is_recorded_once_and_can_be_replayed(name, args):
    model = maestro.NoiseModel()
    assert getattr(model, name)(*args) is None
    expected_log = [(name, args, {})]
    assert model._call_log == expected_log
    expected_state = query_state(model)
    assert model._call_log == expected_log  # Queries must not be recorded.

    restored = pickle.loads(pickle.dumps(model))
    assert restored._call_log == expected_log
    assert query_state(restored) == expected_state
    assert query_state(replay_call_log(model._call_log)) == expected_state


@pytest.mark.parametrize("value,error", [("invalid", TypeError), (-0.1, ValueError)])
def test_rejected_call_keeps_prior_log_and_model_picklable(value, error):
    model = maestro.NoiseModel()
    model.set_depolarizing(0, 0.1)
    original_log = copy.deepcopy(model._call_log)
    original_damping = model.compute_damping("X")

    with pytest.raises(error):
        model.set_depolarizing(0, value)

    assert model._call_log == original_log
    for candidate in (model, pickle.loads(pickle.dumps(model)),
                      replay_call_log(model._call_log)):
        assert candidate.compute_damping("X") == original_damping


@pytest.mark.parametrize("use_kwargs", [False, True], ids=["positional", "keywords"])
def test_ou_band_arguments_are_snapshotted(use_kwargs):
    model = maestro.NoiseModel()
    bands = [[500.0, 0.5], [800.0, 2.0]]
    if use_kwargs:
        model.set_multi_correlated_ou(qubit=0, bands=bands, gate_time=1e-6)
    else:
        model.set_multi_correlated_ou(0, bands, 1e-6)
    original_log = copy.deepcopy(model._call_log)
    circuit = maestro.circuits.QuantumCircuit()
    circuit.h(0)
    circuit.x(0)
    expected = expectation(model, circuit, "X")

    bands[0][0] = -1.0
    bands.append([10.0, 1.0])

    assert model._call_log == original_log
    for candidate in (model, pickle.loads(pickle.dumps(model)),
                      replay_call_log(model._call_log)):
        assert expectation(candidate, circuit, "X") == pytest.approx(expected)


@pytest.mark.parametrize("use_kwargs", [False, True], ids=["positional", "keywords"])
def test_nested_kraus_arguments_are_snapshotted(use_kwargs):
    model = maestro.NoiseModel()
    targets = [0]
    operators = [[[0, 1], [1, 0]]]
    if use_kwargs:
        model.set_kraus_channel(targets=targets, kraus_operators=operators)
    else:
        model.set_kraus_channel(targets, operators)
    original_log = copy.deepcopy(model._call_log)
    circuit = maestro.circuits.QuantumCircuit()
    circuit.x(0)
    assert expectation(model, circuit, "Z") == pytest.approx(1.0)

    targets[0] = 1
    operators[0][0][0] = 2

    assert model._call_log == original_log
    for candidate in (model, pickle.loads(pickle.dumps(model)),
                      replay_call_log(model._call_log)):
        assert expectation(candidate, circuit, "Z") == pytest.approx(1.0)


def test_snapshot_failure_does_not_mutate_native_model():
    class UncopyableBands(list):
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot snapshot bands")

    model = maestro.NoiseModel()
    model.set_depolarizing(0, 0.1)
    original_log = copy.deepcopy(model._call_log)
    original_damping = model.compute_damping("X")
    with pytest.raises(RuntimeError, match="cannot snapshot bands"):
        model.set_multi_correlated_ou(0, UncopyableBands([(15.0, 0.5)]), 1e-7)
    assert not model.has_correlated()
    assert model._call_log == original_log
    assert model.compute_damping("X") == original_damping
    assert pickle.loads(pickle.dumps(model)).compute_damping("X") == original_damping


def test_repeated_ou_append_and_replace_calls_preserve_order():
    model = maestro.NoiseModel()
    calls = [
        ("set_correlated_ou", (0, 900.0, 0.5, 1e-6), {}),
        ("set_correlated_ou_band", (0, 700.0, 2.0, 1e-6), {}),
        ("set_correlated_ou", (0, 500.0, 0.5, 1e-6), {}),
        ("set_correlated_ou_band", (0, 300.0, 2.0, 1e-6), {}),
        ("set_correlated_ou_band", (0, 200.0, 3.0, 1e-6), {}),
    ]
    for name, args, kwargs in calls:
        getattr(model, name)(*args, **kwargs)
    assert model._call_log == calls

    circuit = maestro.circuits.QuantumCircuit()
    circuit.h(0)
    circuit.x(0)
    expected_model = replay_call_log(calls[2:])
    expected = expectation(expected_model, circuit, "X")
    restored = pickle.loads(pickle.dumps(model))
    assert restored._call_log == calls
    for candidate in (model, restored, replay_call_log(model._call_log)):
        assert expectation(candidate, circuit, "X") == pytest.approx(expected)
    assert expectation(replay_call_log(calls[:2]), circuit, "X") != pytest.approx(expected)


@pytest.mark.parametrize("batch", [False, True], ids=["single", "batch"])
def test_rejected_t1_conflict_leaves_native_state_and_replay_unchanged(batch):
    model = maestro.NoiseModel()
    # For the batch call, the conflict is on the last qubit: earlier qubits
    # must not be partially updated before the conflict is discovered.
    model.set_thermal_relaxation(1, 1e-7, 1e-4, 1e-4)
    original_log = copy.deepcopy(model._call_log)
    with pytest.raises(ValueError, match="double-count T1"):
        if batch:
            model.set_all_t1(2, 0.1)
        else:
            model.set_t1(1, 0.1)

    assert model._call_log == original_log
    for candidate in (model, pickle.loads(pickle.dumps(model)),
                      replay_call_log(model._call_log)):
        assert candidate.has_thermal_relaxation()
        assert not candidate.has_t1()
        assert candidate.get_t1_for_gate(0, False) == 0.0
        assert candidate.get_t1_for_gate(1, False) == 0.0
