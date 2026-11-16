"""Truncation configuration must update the live MPS/MPO simulator."""

import math

import maestro
import pytest


@pytest.mark.parametrize("method", [
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.MatrixProductOperator,
])
def test_qcsim_default_and_live_threshold_aliases(method):
    owner = maestro.Maestro()
    handle = owner.create_simulator(maestro.SimulatorType.QCSim, method)
    sim = owner.get_simulator(handle)
    keys = ["matrix_product_state_truncation_threshold",
            "matrix_product_operator_truncation_threshold"]
    try:
        for key in keys:
            assert float(sim.GetConfiguration(key)) == 1e-15
        sim.AllocateQubits(2)
        sim.InitializeSimulator()

        def check_state(threshold):
            for key in keys:
                assert float(sim.GetConfiguration(key)) == threshold
            sim.ResetSimulator()
            sim.ApplyRy(0, 2e-8)
            sim.ApplyCX(0, 1)
            expected = math.sin(2e-8) if threshold < 1e-15 else 0.0
            assert sim.expectation_values(["XX"]) == pytest.approx([expected], abs=1e-12)

        check_state(1e-15)
        for key in keys:
            for threshold in (0.1, 0.0, 1e-18, 1e-15):
                sim.ConfigureSimulator(key, str(threshold))
                check_state(threshold)
    finally:
        owner.destroy_simulator(handle)


@pytest.mark.parametrize("method", [
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.MatrixProductOperator,
])
@pytest.mark.parametrize("fixed_backend", [False, True])
@pytest.mark.parametrize("threshold", [None, 1e-15, 0.1, 1e-18, 0.0])
def test_typed_threshold_controls_estimates(method, fixed_backend, threshold):
    from maestro.circuits import QuantumCircuit

    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=method,
        fixed_backend=fixed_backend,
        singular_value_threshold=threshold,
        auto_reduce=False,
        enable_causal_cone_reduction=False,
        enable_observable_clustering=False,
    )
    assert config.singular_value_threshold == threshold
    circuit = QuantumCircuit()
    circuit.ry(0, 2e-8)
    circuit.cx(0, 1)
    result = circuit.estimate(observables=["XX"], config=config)
    expected = math.sin(2e-8) if threshold is not None and threshold < 1e-15 else 0.0
    assert result["expectation_values"] == pytest.approx([expected], abs=1e-12)


@pytest.mark.parametrize("backend", [
    pytest.param(maestro.SimulatorType.QCSim, id="qcsim"),
    pytest.param(maestro.SimulatorType.Gpu, id="gpu"),
])
@pytest.mark.parametrize("method,prefix", [
    pytest.param(maestro.SimulationType.MatrixProductState,
                 "matrix_product_state", id="mps"),
    pytest.param(maestro.SimulationType.MatrixProductOperator,
                 "matrix_product_state", id="mpo-state-alias"),
    pytest.param(maestro.SimulationType.MatrixProductOperator,
                 "matrix_product_operator", id="mpo-operator-alias"),
])
@pytest.mark.parametrize("mode", ["discarded_weight", "relative_max"])
def test_zero_threshold_replaces_positive_threshold(backend, method, prefix, mode):
    if backend == maestro.SimulatorType.Gpu and not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")

    owner = maestro.Maestro()
    handle = owner.create_simulator(backend, method)
    sim = owner.get_simulator(handle)
    threshold_key = prefix + "_truncation_threshold"
    theta = 0.1
    expected = [math.cos(theta / 2)**2, 0, 0, math.sin(theta / 2)**2]
    try:
        assert sim.GetSimulatorType() == backend
        sim.ConfigureSimulator("use_double_precision", "true")
        sim.ConfigureSimulator(prefix + "_max_bond_dimension", "16")
        sim.ConfigureSimulator(prefix + "_truncation_mode", mode)
        sim.ConfigureSimulator(threshold_key, "0.1")
        sim.AllocateQubits(2)
        sim.InitializeSimulator()

        # The small |11> component is discarded at 0.1 in either mode.
        sim.ApplyRy(0, theta)
        sim.ApplyCX(0, 1)
        assert sim.Probability(3) == pytest.approx(0, abs=1e-10)

        # Reset only the quantum state, retaining the live backend and cutoff.
        # Exercise zero, re-enable truncation, then clear it again.
        for threshold in ("0", "0.1", "0"):
            sim.ResetSimulator()
            sim.ConfigureSimulator(threshold_key, threshold)
            sim.ApplyRy(0, theta)
            sim.ApplyCX(0, 1)
            if threshold == "0":
                assert sim.AllProbabilities() == pytest.approx(expected, abs=1e-10)
            else:
                assert sim.Probability(3) == pytest.approx(0, abs=1e-10)

        # Recreating the backend must also retain the final zero setting.
        sim.ClearSimulator()
        sim.AllocateQubits(2)
        sim.InitializeSimulator()
        sim.ApplyRy(0, theta)
        sim.ApplyCX(0, 1)
        assert sim.AllProbabilities() == pytest.approx(expected, abs=1e-10)
    finally:
        owner.destroy_simulator(handle)
