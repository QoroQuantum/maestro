"""Device configuration and optional real-GPU end-to-end checks."""
import math

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


SVD_SOLVER_FIELDS = ["mps_svd_solver", "mpo_svd_solver", "tensor_network_svd_solver"]


@pytest.mark.parametrize("field", SVD_SOLVER_FIELDS)
def test_gpu_svd_config(field):
    """All GPU SVD solvers are selectable through SimulatorConfig."""
    config = maestro.SimulatorConfig()
    assert getattr(config, field) is None
    for solver in ("gesvd", "gesvdj", "gesvdp", "gesvdr"):
        setattr(config, field, solver)
        assert getattr(config, field) == solver
        assert getattr(maestro.SimulatorConfig(**{field: solver}), field) == solver
    setattr(config, field, None)
    assert getattr(config, field) is None


@pytest.mark.parametrize("field", SVD_SOLVER_FIELDS)
def test_gpu_svd_config_rejects_unknown_solver(field):
    with pytest.raises(ValueError, match=field):
        maestro.SimulatorConfig(**{field: "gesvdx"})
    config = maestro.SimulatorConfig(**{field: "gesvdj"})
    with pytest.raises(ValueError, match=field):
        setattr(config, field, "gesvdx")
    assert getattr(config, field) == "gesvdj"


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


@pytest.mark.parametrize("backend", [maestro.SimulatorType.Gpu, maestro.SimulatorType.QCSim])
@pytest.mark.parametrize("method", [
    maestro.SimulationType.TensorNetwork,
    maestro.SimulationType.MatrixProductState,
    maestro.SimulationType.MatrixProductOperator,
])
@pytest.mark.parametrize("case", ["native", "reduced", "clustered", "identity"])
def test_one_qubit_tensor_estimates(backend, method, case):
    # Identity-only estimates do not allocate a simulator or require a GPU.
    if backend == maestro.SimulatorType.Gpu and case != "identity" and not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    config = maestro.SimulatorConfig(
        simulator_type=backend, simulation_type=method,
        gpu_device=0 if backend == maestro.SimulatorType.Gpu else None,
        auto_reduce=False, enable_causal_cone_reduction=case != "native",
        max_cluster_qubits=1,
    )
    circuit = maestro.circuits.QuantumCircuit()
    circuit.ry(0, 0.7)
    circuit.rz(0, 0.4)
    if case != "native":
        circuit.x(3)
    if case == "native":
        observables = ["X", "Y", "Z"]
        expected = [math.sin(0.7) * math.cos(0.4), math.sin(0.7) * math.sin(0.4), math.cos(0.7)]
    elif case == "reduced":
        observables, expected = ["IIIZ"], [-1.0]
    elif case == "clustered":
        observables = ["IIIZ", "XIII", "YIII", "IIII"]
        expected = [-1.0, math.sin(0.7) * math.cos(0.4), math.sin(0.7) * math.sin(0.4), 1.0]
    else:
        observables, expected = ["IIII"], [1.0]
    before = repr(config)
    result = circuit.estimate(observables, config)
    # Every method keeps one-qubit groups; the GPU tensor network holds them in
    # a two-qubit network itself.
    expected_method = method
    assert result["expectation_values"] == pytest.approx(expected, abs=1e-5)
    assert result["qubits_after"] == (0 if case == "identity" else 1)
    assert result["clusters_count"] == (
        2 if case == "clustered" else 0 if case == "identity" else 1
    )
    assert result["method"] == expected_method.value
    assert result["cluster_methods"] == [expected_method.value] * result["clusters_count"]
    assert result["simulator"] == backend.value
    if backend == maestro.SimulatorType.Gpu:
        assert result["gpu_device"] == (-1 if case == "identity" else 0)
    else:
        assert "gpu_device" not in result
    assert repr(config) == before


@pytest.mark.parametrize("reverse", [False, True])
def test_gpu_tensor_network_one_and_two_qubit_clusters(reverse):
    """Clusters of one and of two qubits both run on the GPU tensor network."""
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    circuit = maestro.circuits.QuantumCircuit()
    circuit.x(0)
    circuit.h(1)
    circuit.cx(1, 2)
    observables = ["ZII", "IZZ"]
    expected = [-1.0, 1.0]
    methods = [maestro.SimulationType.TensorNetwork.value] * 2
    if reverse:
        observables.reverse()
        expected.reverse()
        methods.reverse()
    config = maestro.SimulatorConfig(simulator_type=maestro.SimulatorType.Gpu,
                                     simulation_type=maestro.SimulationType.TensorNetwork, gpu_device=0)
    result = circuit.estimate(observables, config)
    assert result["expectation_values"] == pytest.approx(expected, abs=1e-5)
    assert result["clusters_count"] == 2
    assert result["largest_cluster_qubits"] == 2
    assert result["cluster_methods"] == methods
    assert result["method"] == methods[0]
    assert result["gpu_device"] == 0


def test_gpu_tensor_network_parity_reduced_to_one_qubit():
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    circuit = maestro.circuits.QuantumCircuit()
    circuit.h(0)
    circuit.h(1)
    circuit.cx(0, 1)
    circuit.rz(1, 0.7)
    circuit.cx(0, 1)
    circuit.rx(0, 0.3)
    circuit.rx(1, 0.4)
    observables = ["ZZ", "XI", "II"]
    expected = circuit.estimate(observables, maestro.SimulatorConfig(
        auto_reduce=False, enable_causal_cone_reduction=False))
    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.Gpu,
        simulation_type=maestro.SimulationType.TensorNetwork,
        gpu_device=0, enable_causal_cone_reduction=False,
    )
    result = circuit.estimate(observables, config)
    assert result["auto_reduced"]
    assert result["qubits_after"] == 1
    assert result["cluster_methods"] == [maestro.SimulationType.TensorNetwork.value]
    assert result["expectation_values"] == pytest.approx(expected["expectation_values"], abs=1e-5)


@pytest.mark.parametrize("fixed", [False, True])
def test_gpu_tensor_network_one_qubit_registers(fixed):
    """One-qubit registers run on the GPU tensor network itself.

    cuTensorNet needs two qubits; the simulator holds a one-qubit register in a
    two-qubit network. Before that, these returned no counts or wrong values
    without an error.
    """
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    tn = maestro.SimulationType.TensorNetwork
    config = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.Gpu, simulation_type=tn, gpu_device=0, fixed_backend=fixed,
    )
    cpu = maestro.SimulatorConfig()
    # A one-qubit circuit, and a wider one whose execution maps a single qubit.
    # rz keeps the outcomes deterministic without making the circuit Clifford.
    one = maestro.circuits.QuantumCircuit()
    one.x(0)
    one.rz(0, 0.3)
    one.measure([(0, 0)])
    sparse = maestro.circuits.QuantumCircuit()
    sparse.x(2)
    sparse.rz(2, 0.3)
    sparse.measure([(2, 0)])
    for circuit in (one, sparse):
        result = circuit.execute(config, 32)
        assert result["counts"] == circuit.execute(cpu, 32)["counts"]
        assert result["method"] == tn.value
    # Disjoint one-qubit components are each sampled on the tensor network.
    split = maestro.circuits.QuantumCircuit()
    split.x(0)
    split.rz(0, 0.3)
    split.x(3)
    split.rz(3, 0.5)
    split.measure([(0, 0), (3, 1)])
    reduced = maestro.SimulatorConfig(
        simulator_type=maestro.SimulatorType.Gpu, simulation_type=tn, gpu_device=0, fixed_backend=fixed,
        enable_sampling_reduction=True,
    )
    result = split.execute(reduced, 32)
    assert result["sampling_components"] == 2 and result["largest_component_qubits"] == 1
    assert result["counts"] == split.execute(cpu, 32)["counts"]
    assert result["method"] == tn.value
    # Estimates of a one-qubit register, which are no longer padded or moved.
    rotated = maestro.circuits.QuantumCircuit()
    rotated.ry(0, 0.7)
    estimate = rotated.estimate(["Z", "X"], config)
    assert estimate["expectation_values"] == pytest.approx([math.cos(0.7), math.sin(0.7)], abs=1e-5)
    assert estimate["qubits_after"] == 1
    assert set(estimate["cluster_methods"]) == {tn.value}


def test_native_automatic_gpu_device():
    """Automatic GPU selection checks and reports the device that ran, not the
    network's CPU backend placeholder."""
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    import ctypes
    import json
    from pathlib import Path

    lib = ctypes.CDLL(str(Path(maestro.__file__).parent / "libmaestro.so"))
    lib.MaestroRunRequestJson.argtypes = [ctypes.c_char_p]
    lib.MaestroRunRequestJson.restype = ctypes.c_void_p
    lib.FreeResult.argtypes = [ctypes.c_void_p]
    for options in ({"gpu_device": 0}, {}):
        request = {
            "schema_version": 2,
            "operation": "execute",
            "circuit": {
                "format": "openqasm",
                "num_qubits": 2,
                "num_clbits": 2,
                "source": "OPENQASM 2.0;\nqreg q[2]; creg c[2];\nx q[0]; measure q -> c;",
            },
            "simulator": {"backend": "gpu", "method": "statevector", "selection": "automatic", "options": options},
            "execution": {"seed": 7, "shots": 8},
        }
        ptr = lib.MaestroRunRequestJson(json.dumps(request).encode())
        assert ptr
        try:
            out = json.loads(ctypes.string_at(ptr))
        finally:
            lib.FreeResult(ptr)
        assert out["ok"], out
        metadata = out["execution_metadata"]
        assert metadata["backend"] == "gpu" and metadata["gpu_device"] == 0, metadata
        assert out["counts"] == {"10": 8}


def test_native_distributed_metadata_describes_the_register_that_ran():
    """Distributed placement metadata describes the executed register (the
    three declared qubits), not the network's re-created simulator, which
    also counts the host's entanglement qubit."""
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    import ctypes
    import json
    from pathlib import Path

    lib = ctypes.CDLL(str(Path(maestro.__file__).parent / "libmaestro.so"))
    lib.MaestroRunRequestJson.argtypes = [ctypes.c_char_p]
    lib.MaestroRunRequestJson.restype = ctypes.c_void_p
    lib.FreeResult.argtypes = [ctypes.c_void_p]
    request = {
        "schema_version": 2,
        "operation": "execute",
        "circuit": {
            "format": "openqasm",
            "num_qubits": 3,
            "num_clbits": 3,
            "source": "OPENQASM 2.0;\nqreg q[3]; creg c[3];\nx q[0]; measure q -> c;",
        },
        "simulator": {"backend": "distributed_gpu", "method": "statevector", "selection": "fixed"},
        "execution": {"seed": 7, "shots": 8},
    }
    ptr = lib.MaestroRunRequestJson(json.dumps(request).encode())
    assert ptr
    try:
        out = json.loads(ctypes.string_at(ptr))
    finally:
        lib.FreeResult(ptr)
    if not out["ok"] and out["error"]["code"] == "backend_unavailable":
        pytest.skip(out["error"]["message"])
    assert out["ok"], out
    assert out["counts"] == {"100": 8}
    layout = out["execution_metadata"]["distribution"]["distributed_qubit_layout"]
    assert len(layout.split(",")) == 3, layout


def test_json_gpu_placement():
    """The C API must report execution placement after network recreation."""
    if not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    import ctypes
    import json
    from pathlib import Path

    lib = ctypes.CDLL(str(Path(maestro.__file__).parent / "libmaestro.so"))
    lib.GetMaestroObjectWithMute.argtypes = []
    lib.GetMaestroObjectWithMute.restype = ctypes.c_void_p
    lib.CreateSimpleSimulator.argtypes = [ctypes.c_int]
    lib.CreateSimpleSimulator.restype = ctypes.c_ulong
    lib.RemoveAllOptimizationSimulatorsAndAdd.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_int]
    lib.SimpleExecute.argtypes = [ctypes.c_ulong, ctypes.c_char_p, ctypes.c_char_p]
    lib.SimpleExecute.restype = ctypes.c_void_p
    lib.FreeResult.argtypes = [ctypes.c_void_p]
    lib.DestroySimpleSimulator.argtypes = [ctypes.c_ulong]
    assert lib.GetMaestroObjectWithMute()
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
