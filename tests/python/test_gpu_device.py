"""Device configuration and optional real-GPU end-to-end checks."""
import pytest
import maestro


def test_gpu_device_config():
    config = maestro.SimulatorConfig()
    assert config.gpu_device is None
    config.gpu_device = 0
    assert config.gpu_device == 0
    assert "gpu_device=0" in repr(config)
    config.gpu_device = None
    assert config.gpu_device is None
    assert maestro.SimulatorConfig(gpu_device=1).gpu_device == 1
    with pytest.raises(ValueError, match="gpu_device"):
        maestro.SimulatorConfig(gpu_device=-1)
    with pytest.raises(ValueError, match="gpu_device"):
        config.gpu_device = -1


def test_gpu_svd_config():
    """All GPU SVD selections are available through SimulatorConfig."""
    names = [
        "mps_use_gesvd", "mps_use_gesvdj", "mps_use_gesvdp", "mps_use_gesvdr",
        "mpo_use_gesvd", "mpo_use_gesvdj", "mpo_use_gesvdp", "mpo_use_gesvdr",
        "tensor_network_use_gesvd", "tensor_network_use_gesvdj",
        "tensor_network_use_gesvdp", "tensor_network_use_gesvdr",
    ]
    config = maestro.SimulatorConfig()
    for name in names:
        assert getattr(config, name) is False
        setattr(config, name, True)
        assert getattr(config, name) is True


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
    maestro.SimulationType.TensorNetwork,
    maestro.SimulationType.PauliPropagator,
])
def test_gpu_device_execution(method):
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    qasm = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; x q[0]; x q[1]; measure q -> c;'
    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.Gpu,
        simulation_type=method,
        gpu_device=0,
    )
    result = maestro.simple_execute(qasm, shots=16, config=config)
    assert result["counts"] == {"11": 16}
    assert result["simulator"] == maestro.SimulatorType.Gpu.value
    # An explicit valid device must work even after changing the global default.
    try:
        maestro.select_gpu_device(maestro.get_gpu_device_count())
        assert maestro.simple_execute(qasm, shots=16, config=config)["counts"] == {"11": 16}
        config.gpu_device = maestro.get_gpu_device_count()
        with pytest.raises((RuntimeError, ValueError), match="GPU device"):
            maestro.simple_execute(qasm, shots=16, config=config)
    finally:
        maestro.select_gpu_device(0)
