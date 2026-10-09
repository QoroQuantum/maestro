"""Public batch queries preserve order, state, and MPO normalization."""
from contextlib import contextmanager

import maestro
import pytest


@pytest.fixture(params=[("cpu", "double"), ("gpu", "double"), ("gpu", "single")])
def backend(request):
    name, precision = request.param
    if name == "gpu" and not maestro.is_gpu_available():
        pytest.skip("GPU plugin/device unavailable")
    return (maestro.SimulatorType.Gpu if name == "gpu" else maestro.SimulatorType.QCSim, precision)


@contextmanager
def simulator(method, backend):
    owner = maestro.Maestro()
    handle = owner.create_simulator(backend[0], method)
    sim = owner.get_simulator(handle)
    sim.Configure("precision", backend[1])
    sim.Configure("gate_fusion", "true")
    sim.AllocateQubits(2)
    sim.Initialize()
    try:
        yield sim
    finally:
        owner.destroy_simulator(handle)


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState, maestro.SimulationType.MatrixProductOperator])
def test_real_batches(method, backend):
    with simulator(method, backend) as sim:
        sim.ApplyH(0)
        sim.ApplyCX(0, 1)
        paulis = ["XX", "YY", "ZZ", "II", "XX"]
        assert sim.expectation_values(paulis) == pytest.approx([1, -1, 1, 1, 1])
        assert sim.ExpectationValues(paulis) == pytest.approx([1, -1, 1, 1, 1])
        assert sim.expectation_values([]) == []
        sim.SaveState()
        sim.ApplyX(0)
        assert sim.expectation_values(paulis) == pytest.approx([1, 1, -1, 1, 1])
        sim.RestoreState()
        assert sim.expectation_values(paulis) == pytest.approx([1, -1, 1, 1, 1])


def test_complex_batches(backend):
    with simulator(maestro.SimulationType.MatrixProductOperator, backend) as sim:
        sim.ApplyH(0)
        sim.ApplyCX(0, 1)
        # ApplyOperator takes an Eigen matrix via nanobind's numpy conversion.
        import numpy as np
        sim.apply_operator([0], 2 * np.eye(2, dtype=complex), normalize=False)
        sim.move_at_beginning_of_chain([1])
        paulis = ["XX", "YY", "II", "XX"]
        assert sim.expectation_values_complex(paulis) == pytest.approx([1, -1, 1, 1])
        assert sim.ExpectationValuesComplex(paulis, normalized=False) == pytest.approx([4, -4, 4, 4])
        assert sim.expectation_values_complex([]) == []
        with pytest.raises((ValueError, RuntimeError)):
            sim.expectation_values_complex(["XX", "invalid"])
        assert sim.expectation_values_complex(paulis) == pytest.approx([1, -1, 1, 1])


def test_complex_unsupported_backend(backend):
    with simulator(maestro.SimulationType.Statevector, backend) as sim:
        with pytest.raises(RuntimeError, match="ExpectationValueComplex"):
            sim.expectation_values_complex(["II"])


def test_dense_mps_and_ordered_operators(backend):
    import numpy as np
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    with simulator(maestro.SimulationType.MatrixProductState, backend) as sim:
        sim.ApplyX(0)  # Pending fusion must be flushed by the query.
        assert sim.expectation_value_operators([0, 0], [x, y]) == pytest.approx(1j, abs=2e-5)
        assert sim.ExpectationValueOperators([], []) == pytest.approx(1)
        assert sim.GetStateVector() == pytest.approx([0, 1, 0, 0], abs=2e-5)
        sim.move_at_beginning_of_chain([1])
        assert sim.get_statevector() == pytest.approx([0, 1, 0, 0], abs=2e-5)
        assert sim.AllProbabilities() == pytest.approx([0, 1, 0, 0], abs=2e-5)
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            sim.expectation_value_operators([2], [x])
        with pytest.raises((ValueError, RuntimeError)):
            sim.expectation_value_operators([0], [np.eye(4, dtype=complex)])
        with pytest.raises((ValueError, RuntimeError)):
            sim.expectation_value_operators([0], [np.full((2, 2), np.nan, dtype=complex)])
        assert sim.get_statevector() == pytest.approx([0, 1, 0, 0], abs=2e-5)
