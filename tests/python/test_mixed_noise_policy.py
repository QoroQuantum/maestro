"""Automatic mixed-state execution averages only noise that remains stochastic."""

import math

import pytest

import maestro
from maestro.circuits import QuantumCircuit


MIXED = [
    pytest.param(("QCSim", maestro.SimulationType.DensityMatrix), id="qcsim-dm"),
    pytest.param(("QCSim", maestro.SimulationType.MatrixProductOperator), id="qcsim-mpo"),
    pytest.param(("QiskitAer", maestro.SimulationType.DensityMatrix), id="aer-dm"),
]


def config(method):
    backend = "QCSim"
    if isinstance(method, tuple):
        backend, method = method
    if not hasattr(maestro.SimulatorType, backend):
        pytest.skip(f"{backend} is not compiled in")
    return maestro.SimulatorConfig(
        simulator_type=getattr(maestro.SimulatorType, backend),
        simulation_type=method, fixed_backend=True, seed=31,
        max_simulators=1, max_bond_dimension=16,
    )


def bit_flip_circuit():
    circuit = QuantumCircuit()
    circuit.x(0)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.25)
    return circuit, model


@pytest.mark.parametrize("delta", [-8e-13, 8e-13])
def test_accepted_pauli_probability_rounding_keeps_exact_estimate(delta):
    circuit = QuantumCircuit()
    circuit.channel([0], maestro.QuantumChannel.pauli([0.65, 0.35 + delta, 0.0, 0.0]))
    expected = (0.30 - delta) / (1.0 + delta)
    for method in (maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix):
        result = circuit.estimate(["Z"], config=config(method))
        assert result["measurement_trajectories"] == 0
        assert result["expectation_values"] == pytest.approx([expected], rel=0, abs=5e-16)


def test_rounded_pauli_sampling_form_preserves_composed_coherence():
    np = pytest.importorskip("numpy")
    channel = maestro.QuantumChannel
    angle = 1e-15
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    ry = np.array([[c, -s], [s, c]])
    raw = channel.from_kraus([ry @ op for op in channel.pauli([0.7, 0.1, 0.1, 0.1]).kraus_operators])
    # Depolarization shrinks Z to 0.6 Z, Ry creates X, then dephasing halves X.
    placed = channel.on_qubit(raw.then(channel.pauli([0.75, 0.0, 0.0, 0.25])), 1)
    circuit = QuantumCircuit()
    circuit.channel([0, 1], placed)
    expected = 0.3 * math.sin(angle)
    for method in (maestro.SimulationType.Statevector, maestro.SimulationType.DensityMatrix):
        result = circuit.estimate(["IX"], config=config(method))
        assert result["measurement_trajectories"] == 0
        assert result["expectation_values"] == pytest.approx([expected], rel=1e-12, abs=0.0)


@pytest.mark.parametrize("method", MIXED)
@pytest.mark.parametrize("entry", ["noisy_execute", "full_noise_execute"])
def test_auto_mixed_counts_prepare_one_ensemble(method, entry):
    circuit, model = bit_flip_circuit()
    result = getattr(circuit, entry)(model, config=config(method), shots=4096)
    assert result["noise_realizations"] == 1
    assert result["noise_realizations_requested"] is None
    assert result["noise_realizations_policy"] == "auto"
    assert result["noise_evaluation"] == "exact_channels"
    assert not result["stochastic_noise"]
    assert sum(result["counts"].values()) == 4096
    assert result["counts"].get("0", 0) / 4096 == pytest.approx(0.25, abs=0.04)


@pytest.mark.parametrize("method", MIXED)
@pytest.mark.parametrize("entry", ["noisy_estimate_montecarlo", "full_noise_estimate"])
def test_auto_mixed_estimates_are_deterministic_and_explicit_counts_survive(method, entry):
    circuit, model = bit_flip_circuit()
    result = getattr(circuit, entry)(["Z"], model, config=config(method))
    assert result["noise_realizations"] == 1
    assert result["estimate_evaluation"] == "deterministic"
    assert result["expectation_values"] == pytest.approx([-0.5], abs=1e-10)
    assert result["ideal_expectation_values"] == pytest.approx([-1.0], abs=1e-10)
    explicit = getattr(circuit, entry)(["Z"], model, config=config(method), noise_realizations=3)
    assert explicit["noise_realizations"] == 3
    assert explicit["noise_realizations_requested"] == 3
    assert explicit["noise_realizations_policy"] == "explicit"
    assert explicit["expectation_values"] == pytest.approx(result["expectation_values"], abs=1e-10)


@pytest.mark.parametrize("method", MIXED)
def test_auto_mixed_fidelity_and_explicit_count_policy(method):
    circuit, model = bit_flip_circuit()
    result = circuit.noisy_fidelity(model, config=config(method))
    assert result["noise_realizations"] == 1
    assert result["estimate_evaluation"] == "deterministic"
    assert result["fidelity"] == pytest.approx(0.75, abs=1e-10)
    explicit = circuit.noisy_execute(model, config=config(method), shots=17, noise_realizations=50)
    assert explicit["noise_realizations"] == 17
    assert explicit["noise_realizations_requested"] == 50


@pytest.mark.parametrize("method", MIXED)
def test_auto_mixed_coherent_noise_retains_disorder_average(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.measure_all()
    model = maestro.NoiseModel()
    model.set_coherent_rotation(0, 0.0, math.pi / 2, 0.0)
    result = circuit.coherent_execute(model, config=config(method), shots=1024)
    assert result["noise_realizations"] == 64
    assert result["stochastic_noise"]
    assert set(result["counts"]) == {"0", "1"}
    estimate = circuit.coherent_estimate(["Z"], model, config=config(method))
    assert estimate["noise_realizations"] == 100
    assert estimate["estimate_evaluation"] == "trajectory_average"
    assert abs(estimate["expectation_values"][0]) < 0.4
    one = circuit.coherent_estimate(["Z"], model, config=config(method), noise_realizations=1)
    assert abs(one["expectation_values"][0]) == pytest.approx(1.0)
    assert one["realization_standard_errors"] is None


@pytest.mark.parametrize("method", MIXED)
def test_auto_policy_distinguishes_temporal_noise_and_deterministic_crosstalk(method):
    circuit, _ = bit_flip_circuit()
    temporal = maestro.NoiseModel()
    temporal.set_correlated_ou(0, sigma=0.2, alpha=1.0, gate_time=0.1)
    assert maestro._resolve_noise_realizations(circuit, temporal, config(method), 157) == 157
    deterministic = maestro.NoiseModel()
    deterministic.set_crosstalk(0, 1, 0.2)
    deterministic.set_idle_noise(0, 2.0, 1.0, detuning_hz=0.5)
    assert maestro._resolve_noise_realizations(circuit, deterministic, config(method), 157) == 1


def dynamic_circuit():
    return maestro.QasmToCirc().parse_and_translate(
        "OPENQASM 3.0; qubit[2] q; bit[2] c; "
        "h q[0]; c[0] = measure q[0]; if (c[0]) { x q[1]; }")


@pytest.mark.parametrize("method", MIXED)
def test_dynamic_estimates_average_branches_and_ideal_baseline(method):
    circuit = dynamic_circuit()
    result = circuit.noisy_estimate_montecarlo(["IZ"], maestro.NoiseModel(), config=config(method))
    assert result["noise_realizations"] == 100
    assert result["stochastic_execution"]
    assert result["estimate_evaluation"] == "trajectory_average"
    assert abs(result["expectation_values"][0]) < 0.4
    assert abs(result["ideal_expectation_values"][0]) < 0.4
    assert result["realization_standard_errors"][0] > 0.05
    one = circuit.noisy_estimate_montecarlo(["IZ"], maestro.NoiseModel(), config=config(method), noise_realizations=1)
    assert one["realization_standard_errors"] is None
    assert abs(one["expectation_values"][0]) == pytest.approx(1.0)
    plain = circuit.estimate(["IZ"], config=config(method))
    assert plain["stochastic_execution"]
    assert plain["estimate_evaluation"] == "trajectory_average"
    assert plain["measurement_trajectories"] == 100
    assert abs(plain["expectation_values"][0]) < 0.4
    assert plain["realization_standard_errors"][0] > 0.05
    assert plain["expectation_values"] == circuit.estimate(["IZ"], config=config(method))["expectation_values"]
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.1)
    with pytest.raises(ValueError, match="unitary ideal"):
        circuit.noisy_fidelity(model, config=config(method))


@pytest.mark.parametrize("method", MIXED)
def test_mixed_reset_remains_a_deterministic_channel(method):
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    result = circuit.noisy_estimate_montecarlo(["IZ"], maestro.NoiseModel(), config=config(method))
    assert result["noise_realizations"] == 1
    assert not result["stochastic_execution"]
    assert result["expectation_values"] == pytest.approx([0.0], abs=1e-10)
    assert circuit.estimate(["IZ"], config=config(method))["expectation_values"] == pytest.approx([0.0], abs=1e-10)


@pytest.mark.parametrize("entry", ["noisy_execute", "full_noise_execute", "noisy_fidelity"])
@pytest.mark.parametrize("invalid", [0, -1])
def test_realization_validation_still_rejects_nonpositive_counts(entry, invalid):
    circuit, model = bit_flip_circuit()
    with pytest.raises(ValueError, match="noise_realizations"):
        getattr(circuit, entry)(model, config=config(maestro.SimulationType.DensityMatrix), noise_realizations=invalid)


def test_pure_state_default_keeps_trajectory_budget():
    circuit, model = bit_flip_circuit()
    result = circuit.noisy_execute(model, config=config(maestro.SimulationType.Statevector), shots=128)
    assert result["noise_realizations"] == 64
    assert result["noise_evaluation"] == "trajectories"


@pytest.mark.parametrize("backend", ["QCSim", "CompositeQCSim", "QiskitAer", "CompositeQiskitAer"])
def test_statevector_terminal_channels_average_on_the_observable(backend):
    cfg = config((backend, maestro.SimulationType.Statevector))
    circuit = QuantumCircuit()
    circuit.x(0)
    model = maestro.NoiseModel()
    model.set_t1(0, 0.3)
    noisy = circuit.full_noise_estimate(["Z"], model, config=cfg)
    assert noisy["noise_realizations"] == 1
    assert noisy["noise_evaluation"] == "terminal_channels"
    assert noisy["expectation_values"] == pytest.approx([-0.4], abs=1e-12)
    assert noisy["ideal_expectation_values"] == [-1.0]
    assert noisy["realization_standard_errors"] == [0.0]

    circuit.channel([0], maestro.QuantumChannel.amplitude_damping(0.3))
    plain = circuit.estimate(["Z"], config=cfg)
    assert plain["terminal_channels_averaged"]
    assert plain["measurement_trajectories"] == 0
    assert plain["expectation_values"] == pytest.approx(noisy["expectation_values"], abs=1e-12)

    cfg.terminal_channel_policy = "trajectories"
    for result in (circuit.estimate(["Z"], config=cfg), maestro.simple_estimate(circuit, ["Z"], config=cfg)):
        assert result["measurement_trajectories"] == 100
        assert result["realization_standard_errors"][0] > 0.0
        assert not result.get("terminal_channels_averaged", False)


@pytest.mark.parametrize("entry", ["noisy_estimate_montecarlo", "full_noise_estimate"])
def test_module_and_circuit_noise_policy(entry):
    circuit = QuantumCircuit()
    circuit.x(0)
    model = maestro.NoiseModel()
    model.set_t1(0, 0.3)
    cfg = config(maestro.SimulationType.Statevector)
    for policy in ("auto", "trajectories"):
        cfg.terminal_channel_policy = policy
        kwargs = dict(config=cfg, noise_seed=17)
        bound = getattr(circuit, entry)(["Z"], model, **kwargs)
        free = getattr(maestro, entry)(circuit, ["Z"], model, **kwargs)
        assert free["expectation_values"] == bound["expectation_values"]
        assert free["realization_standard_errors"] == bound["realization_standard_errors"]
        if policy == "auto":
            assert free["noise_realizations"] == 1
            assert free["expectation_values"] == pytest.approx([-0.4], abs=1e-12)
            assert free["realization_standard_errors"] == [0.0]
        else:
            assert free["noise_realizations"] == 100
            assert free["realization_standard_errors"][0] > 0.0


@pytest.mark.parametrize("with_model", [False, True])
def test_execution_terminal_policy_is_shared_by_counts_and_rows(with_model):
    circuit = QuantumCircuit()
    circuit.x(0)
    model = maestro.NoiseModel()
    if with_model:
        model.set_t1(0, 0.3)
    else:
        circuit.channel([0], maestro.QuantumChannel.amplitude_damping(0.3))
    circuit.measure_all()
    cfg = config(maestro.SimulationType.Statevector)
    for policy in ("auto", "trajectories"):
        cfg.terminal_channel_policy = policy
        for rows in (False, True):
            kwargs = dict(config=cfg, shots=2000, return_samples=rows)
            if with_model:
                result = maestro.full_noise_execute(circuit, model, noise_seed=17, **kwargs)
            else:
                result = maestro.simple_execute(circuit, **kwargs)
            assert (result.get("sampling_strategy") == "terminal_v1") == (policy == "auto")
            if rows:
                assert result["samples"].shape == (2000, 1)
                assert result["samples"].mean() == pytest.approx(0.7, abs=0.05)
            else:
                assert sum(result["counts"].values()) == 2000
                assert result["counts"]["0"] / 2000 == pytest.approx(0.3, abs=0.05)


def test_fidelity_does_not_drop_measurement_before_inline_channel():
    circuit = QuantumCircuit()
    circuit.h(0)
    circuit.measure_all()
    circuit.pauli_channel([0], [0.5, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="unitary ideal"):
        circuit.noisy_fidelity(maestro.NoiseModel(), config=config(maestro.SimulationType.DensityMatrix))


@pytest.mark.parametrize("method", MIXED)
def test_fidelity_ignores_conditional_terminal_readout(method):
    circuit = maestro.QasmToCirc().parse_and_translate(
        "OPENQASM 2.0; qreg q[1]; creg c[1]; "
        "h q[0]; if(c==0) measure q[0] -> c[0];")
    model = maestro.NoiseModel()
    model.set_bit_flip(0, 0.0)
    result = circuit.noisy_fidelity(model, config=config(method))
    assert result["noise_realizations"] == 1
    assert result["fidelity"] == pytest.approx(1.0, abs=1e-10)


PURE = [
    pytest.param(maestro.SimulationType.Statevector, id="statevector"),
    pytest.param(maestro.SimulationType.MatrixProductState, id="mps"),
]


@pytest.mark.parametrize("method", PURE)
def test_pure_state_dynamic_estimate_averages_like_native_requests(method):
    circuit = dynamic_circuit()
    result = circuit.estimate(["IZ"], config=config(method))
    assert result["estimate_evaluation"] == "trajectory_average"
    assert result["measurement_trajectories"] == 100
    assert abs(result["expectation_values"][0]) < 0.4
    assert result["realization_standard_errors"][0] > 0.05
    model = maestro.NoiseModel()
    model.set_depolarizing(0, 0.01)
    with pytest.raises(ValueError, match="deterministic evolution"):
        circuit.noisy_estimate(["IZ"], model, config=config(method))


def leading_reset_circuit(after_gate=False):
    circuit = QuantumCircuit()
    if not after_gate:
        circuit.reset(0)
        circuit.reset(1)
    circuit.h(0)
    circuit.cx(0, 1)
    if after_gate:
        circuit.reset(1)
    return circuit


@pytest.mark.parametrize("method", PURE)
def test_pure_state_leading_resets_stay_deterministic(method):
    circuit = leading_reset_circuit()
    result = circuit.estimate(["XX"], config=config(method))
    assert not result["stochastic_execution"]
    assert result["estimate_evaluation"] == "deterministic"
    assert "realization_standard_errors" not in result
    assert result["expectation_values"] == pytest.approx([1.0], abs=1e-10)
    model = maestro.NoiseModel()
    model.set_depolarizing(0, 0.01)
    noisy = circuit.noisy_estimate_montecarlo(["XX"], model, config=config(method), noise_realizations=4, noise_seed=3)
    assert not noisy["stochastic_execution"]
    assert noisy["ideal_expectation_values"] == pytest.approx([1.0], abs=1e-10)
    # A reset on |0> is unitary for fidelity purposes too.
    fidelity = circuit.noisy_fidelity(model, config=config(method), noise_realizations=4, noise_seed=3)
    assert 0.0 < fidelity["fidelity"] <= 1.0
    # A reset after a gate is a genuine branch.
    later = leading_reset_circuit(after_gate=True)
    assert later.estimate(["ZI"], config=config(method))["estimate_evaluation"] == "trajectory_average"
    with pytest.raises(ValueError, match="unitary ideal"):
        later.noisy_fidelity(model, config=config(method))


def test_automatic_backend_selection_keeps_mixed_state_policy():
    # Python's default fixed_backend=False configures an MPO candidate and a
    # placeholder simulator: the policy and the prepared MPO must follow the
    # candidate (one realization), sampled by several workers.
    circuit, model = bit_flip_circuit()
    for method in (maestro.SimulationType.DensityMatrix, maestro.SimulationType.MatrixProductOperator):
        auto = maestro.SimulatorConfig(simulation_type=method, max_simulators=4, max_bond_dimension=16)
        result = circuit.noisy_execute(model, config=auto, shots=4000)
        assert result["noise_realizations"] == 1
        assert result["noise_evaluation"] == "exact_channels"
        assert result["counts"].get("0", 0) / 4000 == pytest.approx(0.25, abs=0.04)


def test_python_candidate_lists_select_inside_the_list():
    candidates = [(maestro.SimulatorType.QCSim, maestro.SimulationType.DensityMatrix),
                  (maestro.SimulatorType.QCSim, maestro.SimulationType.MatrixProductOperator)]
    auto = maestro.SimulatorConfig(simulation_type=maestro.SimulationType.Statevector, optimization_candidates=candidates,
                                   max_bond_dimension=16, seed=5)
    assert auto.optimization_candidates == candidates
    circuit, model = bit_flip_circuit()
    # Every candidate applies exact channels, so the requested statevector does
    # not decide the policy.
    result = circuit.noisy_execute(model, config=auto, shots=4000)
    assert result["noise_evaluation"] == "exact_channels"
    assert result["noise_realizations"] == 1
    assert result["counts"].get("0", 0) / 4000 == pytest.approx(0.25, abs=0.04)
    # Explicit simulators: a checkpoint resolves the list to one backend.
    prefix = QuantumCircuit()
    prefix.x(0)
    checkpoint = maestro.PrefixCheckpointedSimulator(prefix, 1, auto, prefix_noise_model=model)
    suffix = QuantumCircuit()
    suffix.measure([(0, 0)])
    rows = checkpoint.execute_suffix(suffix, shots=4000, return_samples=True)
    assert rows["method"] == maestro.SimulationType.DensityMatrix
    assert rows["samples"][:, 0].mean() == pytest.approx(0.75, abs=0.03)
    # A mixed list is not exact: trajectories run on whichever is chosen.
    mixed = maestro.SimulatorConfig(optimization_candidates=[(maestro.SimulatorType.QCSim, maestro.SimulationType.Statevector),
                                                            (maestro.SimulatorType.QCSim, maestro.SimulationType.DensityMatrix)])
    assert circuit.noisy_execute(model, config=mixed, shots=256)["noise_evaluation"] == "trajectories"
    import pickle
    assert pickle.loads(pickle.dumps(auto)).optimization_candidates == candidates
    with pytest.raises(ValueError, match="Distribution"):
        auto.optimization_candidates = [(maestro.SimulatorType.DistributedGpu, maestro.SimulationType.Statevector)]
    assert auto.optimization_candidates == candidates
