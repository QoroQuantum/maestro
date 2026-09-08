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
    assert result["gpu_device"] == 0
    asymmetric = qasm.replace('x q[0]; ', '')
    assert maestro.simple_execute(asymmetric, shots=16, config=config)["counts"] == {"01": 16}
    estimate = maestro.simple_estimate(
        qasm.replace('measure q -> c;', ''), ['ZI', 'IZ'], config=config)
    assert estimate["gpu_device"] == 0
    assert estimate["expectation_values"] == pytest.approx([-1, -1], abs=1e-5)
    # An explicit valid device must work even after changing the global default.
    try:
        maestro.select_gpu_device(maestro.get_gpu_device_count())
        assert maestro.simple_execute(qasm, shots=16, config=config)["counts"] == {"11": 16}
        config.gpu_device = maestro.get_gpu_device_count()
        with pytest.raises((RuntimeError, ValueError), match="GPU device"):
            maestro.simple_execute(qasm, shots=16, config=config)
    finally:
        maestro.select_gpu_device(0)


@pytest.mark.parametrize("method", [
    maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.DensityMatrix,
    maestro.SimulationType.MatrixProductOperator,
    maestro.SimulationType.TensorNetwork,
    maestro.SimulationType.PauliPropagator,
])
def test_two_gpu_python_placement(method):
    count = maestro.get_gpu_device_count()
    assert count >= 0, "CUDA device discovery failed"
    if count < 2:
        pytest.skip("requires two visible GPUs")
    prefix = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; '
    configs = [maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.Gpu,
        simulation_type=method, gpu_device=device,
    ) for device in (0, 1)]
    try:
        for device in (1, 0, 1, 0):
            maestro.select_gpu_device(1 - device)
            qasm = prefix + f'x q[{device}]; measure q -> c;'
            result = maestro.simple_execute(qasm, shots=16, config=configs[device])
            assert result["gpu_device"] == device
            assert result["counts"] == {("10" if device == 0 else "01"): 16}
            circuit = prefix + f'h q[0]; ry({0.31 * (device + 1)}) q[1]; cx q[0],q[1]; rz(0.47) q[0];'
            observables = ['ZI', 'IZ', 'XX', 'XY', 'YY', 'ZZ']
            cpu = maestro.simple_estimate(circuit, observables, config=maestro.SimulatorConfig(
                simulator_type=maestro.SimulatorType.QCSim,
                simulation_type=maestro.SimulationType.Statevector))
            actual = maestro.simple_estimate(circuit, observables, config=configs[device])
            assert actual["gpu_device"] == device
            assert actual["expectation_values"] == pytest.approx(cpu["expectation_values"], abs=1e-5)
    finally:
        maestro.select_gpu_device(0)


def test_json_gpu_placement():
    """The C API must report execution placement after network recreation."""
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    import ctypes
    import json
    from pathlib import Path

    lib = ctypes.CDLL(str(Path(maestro.__file__).parent / "libmaestro.so"))
    lib.CreateSimpleSimulator.argtypes = [ctypes.c_int]
    lib.CreateSimpleSimulator.restype = ctypes.c_ulong
    lib.RemoveAllOptimizationSimulatorsAndAdd.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
    lib.SimpleExecute.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_char_p]
    lib.SimpleExecute.restype = ctypes.c_void_p
    lib.FreeResult.argtypes = [ctypes.c_void_p]
    lib.DestroySimpleSimulator.argtypes = [ctypes.c_ulong]
    handles = [lib.CreateSimpleSimulator(2) for _ in range(2)]
    try:
        for handle in handles:
            assert handle
            assert lib.RemoveAllOptimizationSimulatorsAndAdd(
                handle, maestro.SimulatorType.Gpu.value, maestro.SimulationType.Statevector.value)
        second = 1 if maestro.get_gpu_device_count() > 1 else 0
        for side in (1, 0, 1, 0):
            device = second if side else 0
            config = json.dumps({"gpu_device": device, "shots": 8}).encode()
            qasm = b'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; x q[0]; measure q -> c;'
            ptr = lib.SimpleExecute(handles[side], qasm, config)
            assert ptr
            try:
                result = json.loads(ctypes.string_at(ptr))
                assert result["gpu_device"] == device
                assert result["counts"] == {"10": 8}
            finally:
                lib.FreeResult(ptr)
    finally:
        for handle in handles:
            lib.DestroySimpleSimulator(handle)
