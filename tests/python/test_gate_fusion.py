"""Fusion configuration must reach both direct and circuit execution paths."""
import pickle

import maestro
import pytest


def test_config_default_constructor_and_pickle():
    assert maestro.SimulatorConfig().gate_fusion is True
    config = maestro.SimulatorConfig(gate_fusion=False)
    assert "gate_fusion=False" in repr(config)
    assert pickle.loads(pickle.dumps(config)).gate_fusion is False
    config.gate_fusion = True
    assert config.gate_fusion is True


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.MatrixProductOperator,
    maestro.SimulationType.DensityMatrix,
])
def test_direct_toggle_flushes_pending_gates(method):
    owner = maestro.Maestro()
    handle = owner.create_simulator(maestro.SimulatorType.QCSim, method)
    sim = owner.get_simulator(handle)
    try:
        assert sim.GetConfiguration("gate_fusion") == "true"
        sim.AllocateQubits(2)
        sim.InitializeSimulator()
        sim.ApplyH(0)
        sim.ApplyCX(0, 1)
        sim.Configure("gate_fusion", "false")
        assert sim.AllProbabilities() == pytest.approx([.5, 0, 0, .5])
        sim.Configure("gate_fusion", "true")
        sim.ApplyCX(0, 1)
        sim.ApplyH(0)
        assert sim.Probability(0) == pytest.approx(1)
    finally:
        owner.destroy_simulator(handle)


@pytest.mark.parametrize("fusion", [True, False])
def test_circuit_execution_with_measurement_and_condition(fusion):
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2]; creg c[2];
    h q[0]; h q[0]; x q[0];
    measure q[0] -> c[0];
    if(c==1) x q[1];
    measure q[1] -> c[1];
    """
    result = maestro.simple_execute(
        source, shots=32,
        config=maestro.SimulatorConfig(gate_fusion=fusion, seed=123))
    assert result["counts"] == {"11": 32}


def test_effective_capability_and_statistics():
    owner = maestro.Maestro()
    for method, width in [(maestro.SimulationType.Statevector, 3),
                          (maestro.SimulationType.MatrixProductState, 2),
                          (maestro.SimulationType.Stabilizer, 0)]:
        handle = owner.create_simulator(maestro.SimulatorType.QCSim, method)
        try:
            sim = owner.get_simulator(handle)
            sim.AllocateQubits(3)
            sim.Initialize()
            assert sim.GetGateFusionMaxQubits() == width
            assert sim.GetConfiguration("gate_fusion") == "true"
            assert sim.IsGateFusionEnabled() == bool(width)
            if width:
                sim.ApplyH(0)
                sim.ApplyH(0)
                before = sim.GetGateFusionStatistics()
                assert before.submittedGates == 2
                assert before.backendGates == 0  # Query must not flush.
                sim.Flush()
                after = sim.GetGateFusionStatistics()
                assert after.backendGates == 1
                assert after.fusedBlocks == 1
                sim.Configure("gate_fusion", "false")
                assert not sim.IsGateFusionEnabled()
                assert sim.GetGateFusionMaxQubits() == width
        finally:
            owner.destroy_simulator(handle)


@pytest.mark.parametrize("width,name", [(1, "One"), (2, "Two"), (3, "Three")])
@pytest.mark.parametrize("fusion", [True, False])
def test_generic_matrix_target_order_copy_and_validation(width, name, fusion):
    owner = maestro.Maestro()
    handle = owner.create_simulator(maestro.SimulatorType.QCSim,
                                    maestro.SimulationType.Statevector)
    try:
        sim = owner.get_simulator(handle)
        sim.AllocateQubits(3)
        sim.Initialize()
        sim.Configure("gate_fusion", str(fusion).lower())
        dim = 1 << width
        matrix = [[1j if row == (col + 1) % dim else 0j
                   for col in range(dim)] for row in range(dim)]
        targets = [2, 0, 1][:width]
        apply = getattr(sim, f"ApplyGeneric{name}QubitGate")
        sim.ApplyH(2)
        sim.ApplyH(2)
        apply(*targets, matrix)
        matrix[1][0] = 0  # Pending fusion must own its input.
        assert sim.Amplitude(4) == pytest.approx(1j)
        assert sim.Probability(4) == pytest.approx(1)
        with pytest.raises(ValueError):
            apply(*targets, [[1]])
        matrix[1][0] = complex(float('nan'), 0)
        with pytest.raises(ValueError):
            apply(*targets, matrix)
        matrix[1][0] = 1j
        with pytest.raises(ValueError):
            apply(*([3] + targets[1:]), matrix)
        if width > 1:
            with pytest.raises(ValueError):
                apply(*([2] * width), matrix)
        assert sim.Probability(4) == pytest.approx(1)
    finally:
        owner.destroy_simulator(handle)


def test_generic_numpy_matrix():
    np = pytest.importorskip('numpy')
    owner = maestro.Maestro()
    handle = owner.create_simulator(maestro.SimulatorType.QCSim,
                                    maestro.SimulationType.Statevector)
    try:
        sim = owner.get_simulator(handle)
        sim.AllocateQubits(1)
        sim.Initialize()
        sim.ApplyGenericOneQubitGate(0, np.array([[0, -1j], [1j, 0]]))
        assert sim.Amplitude(1) == pytest.approx(1j)
    finally:
        owner.destroy_simulator(handle)
