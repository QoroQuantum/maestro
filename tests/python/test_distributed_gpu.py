"""Distributed backend configuration and optional GPU/MPI integration.

Run MPI cases with: mpirun -n 2 python -m pytest -q this_file --...
Set MAESTRO_TEST_MPI_GPU=1 to enable collective MPI tests.
"""
import os
import pickle
import pytest

# Initialize MPI before importing the extension, with the application's chosen
# thread level (the server uses MPI4PY_RC_THREAD_LEVEL=serialized).
if os.environ.get("MAESTRO_TEST_MPI_GPU") == "1":
    from mpi4py import MPI

import maestro


def config_for(mpi=False):
    config = maestro.SimulatorConfig(
        simulator_type=(maestro.SimulatorType.DistributedMpiGpu if mpi else
                        maestro.SimulatorType.DistributedGpu),
        simulation_type=maestro.SimulationType.Statevector,
        seed=19,
    )
    return config


def test_config_roundtrip():
    config = config_for()
    assert config.distributed_options == {}
    constructed = maestro.SimulatorConfig(distributed_options={"distributed_flags": "8"})
    assert constructed.distributed_options == {"distributed_flags": "8"}
    config.distributed_options = {"distributed_devices": "0,1", "distributed_flags": "8"}
    if hasattr(maestro, "_raw_maestro"):
        restored = pickle.loads(pickle.dumps(config))
        assert restored.distributed_options == config.distributed_options
        assert restored.simulator_type == config.simulator_type


def test_reject_unsupported_method():
    config = config_for()
    config.simulation_type = maestro.SimulationType.DensityMatrix
    with pytest.raises(ValueError, match="Statevector"):
        maestro.simple_execute('OPENQASM 2.0; include "qelib1.inc"; qreg q[2];', config=config)


@pytest.mark.parametrize("mpi", [False, True])
def test_execution(mpi):
    if mpi:
        if os.environ.get("MAESTRO_TEST_MPI_GPU") != "1":
            pytest.skip("MPI test requires an explicitly enabled collective launch")
        from mpi4py import MPI
        config = config_for(True)
        config.distributed_options = {"mpi_communicator": str(MPI.COMM_WORLD.py2f())}
    else:
        if not maestro.is_distributed_gpu_available():
            pytest.skip("Distributed plugin/GPU unavailable")
        config = config_for()
    prefix = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[4]; creg c[4]; '
    try:
        actual = maestro.simple_execute(prefix + 'x q[0]; x q[3]; measure q -> c;', shots=32, config=config)
        assert actual["simulator"] == config.simulator_type.value
        assert actual["counts"] == {"1001": 32}
        bell = maestro.simple_execute(prefix + 'h q[0]; cx q[0],q[3]; measure q -> c;', shots=512, config=config)
        assert set(bell["counts"]) == {"0000", "1001"}
        # Operations after measurement force repeated collapse/restore execution.
        mid = maestro.simple_execute(prefix + 'h q[0]; measure q[0] -> c[0]; '
                                    'x q[3]; measure q[3] -> c[3];', shots=64, config=config)
        assert sum(mid["counts"].values()) == 64
        assert set(mid["counts"]) <= {"0001", "1001"}
        circuit = prefix + 'h q[0]; ry(0.31) q[1]; cx q[0],q[3]; rz(0.27) q[0];'
        paulis = ['XIIX', 'YIIY', 'ZIIZ', 'IZII', 'IIXI', 'IIZI']
        reference = maestro.simple_estimate(circuit, paulis, config=maestro.SimulatorConfig())
        result = maestro.simple_estimate(circuit, paulis, config=config)
        assert result["expectation_values"] == pytest.approx(reference["expectation_values"], abs=1e-5)
        # Qubit 2 is idle; retain all four wires and logical output ordering.
        idle_circuit = maestro.circuits.QuantumCircuit()
        idle_circuit.x(3)
        amps = idle_circuit.get_statevector(config=config)
        if isinstance(amps, dict):
            amps = amps["statevector"]
        assert len(amps) == 16
        assert abs(amps[8] - 1) < 1e-5
        if mpi:
            assert len(set(MPI.COMM_WORLD.allgather(str(sorted(bell["counts"].items()))))) == 1
    finally:
        if mpi:
            maestro.finalize_distributed_mpi_gpu()
    if mpi:
        # Terminal shutdown must affect the instance used by the core library,
        # not a second singleton hidden inside the Python extension.
        with pytest.raises(RuntimeError, match="finalized"):
            maestro.simple_execute(prefix + 'h q[0]; cx q[0],q[3];', shots=1, config=config)


@pytest.mark.parametrize("library", ["/maestro-test-missing/plugin.so", "libc.so.6"])
def test_availability_probe_does_not_throw(library):
    """A fresh process isolates the singleton's cached library handle."""
    import subprocess
    import sys

    if sys.platform != "linux":
        pytest.skip("Distributed plugins are Linux-only")
    env = dict(os.environ, MAESTRO_DIST_GPU_LIBRARY=library)
    subprocess.run(
        [sys.executable, "-c", "import maestro; assert maestro.is_distributed_gpu_available() is False"],
        env=env, check=True, capture_output=True, text=True,
    )
