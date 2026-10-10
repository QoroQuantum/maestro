"""Public Python trajectory controls must reach execution and survive pickling."""
import ctypes
import pickle

import maestro
import pytest


def test_trajectory_config_roundtrip():
    values = dict(trajectory_policy="shot_v1", trajectory_max_memory_mb=64,
                  max_simulators=4)
    config = maestro.SimulatorConfig(enable_causal_cone_reduction=False, **values)
    restored = pickle.loads(pickle.dumps(config))
    assert maestro.SimulatorConfig().enable_causal_cone_reduction is True
    assert restored.enable_causal_cone_reduction is False
    for name, value in values.items():
        assert getattr(maestro.SimulatorConfig(), name) is None
        assert getattr(restored, name) == value
        assert f"{name}={value!r}" in repr(config)
        setattr(config, name, None)
        assert getattr(config, name) is None
        setattr(config, name, value)
        assert getattr(config, name) == value


@pytest.mark.parametrize("name,value", [
    ("trajectory_policy", "unknown"),
    ("trajectory_policy", 1),
    ("max_simulators", 0),
    ("max_simulators", -1),
    ("max_simulators", 1025),
    ("max_simulators", "4"),
    ("max_simulators", True),
    ("trajectory_max_memory_mb", 0),
    ("trajectory_max_memory_mb", -1),
    ("trajectory_max_memory_mb", "64"),
    ("trajectory_max_memory_mb", True),
    ("trajectory_max_memory_mb", 1 << (ctypes.sizeof(ctypes.c_size_t) * 8 - 20)),
])
def test_invalid_values_leave_config_unchanged(name, value):
    with pytest.raises((ValueError, TypeError)):
        maestro.SimulatorConfig(**{name: value})
    config = maestro.SimulatorConfig()
    with pytest.raises((ValueError, TypeError)):
        setattr(config, name, value)
    assert getattr(config, name) is None


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector,
                                    maestro.SimulationType.DensityMatrix,
                                    maestro.SimulationType.PathIntegral])
def test_seeded_policy_execution(method):
    source = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2]; creg c[3];
    ry(0.71) q[0]; measure q[0] -> c[0];
    if(c==1) x q[1];
    h q[0]; measure q[0] -> c[1]; measure q[1] -> c[2];
    """

    def run(policy=None, workers=None, memory=None):
        config = maestro.SimulatorConfig(
            simulation_type=method, seed=123, trajectory_policy=policy,
            max_simulators=workers, trajectory_max_memory_mb=memory)
        result = maestro.simple_execute(source, config=config, shots=257)
        assert result["simulator"] == maestro.SimulatorType.QCSim.value
        assert result["method"] == method.value
        counts = result["counts"]
        assert sum(counts.values()) == 257
        assert set(counts) <= {"000", "010", "101", "111"}
        return counts

    baseline = run()
    old_policy = run("block_v1", 1)
    assert run("block_v1", 4) == old_policy
    per_shot = run("shot_v1", 1, 1)
    # Catches a binding that accepts the option but silently ignores it.
    assert per_shot != old_policy
    assert per_shot == baseline
    assert run("shot_v1", 4, 64) == per_shot
    assert run("shot_v1", 3, 1) == per_shot
    assert run() == baseline
