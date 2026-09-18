"""Truncation configuration must update the live MPS/MPO simulator."""

import math

import maestro
import pytest


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
