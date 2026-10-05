"""Public batch queries preserve order, state, and MPO normalization."""
from contextlib import contextmanager

import maestro
import pytest


@contextmanager
def simulator(method):
    owner = maestro.Maestro()
    handle = owner.create_simulator(maestro.SimulatorType.QCSim, method)
    sim = owner.get_simulator(handle)
    sim.AllocateQubits(2)
    sim.Initialize()
    try:
        yield sim
    finally:
        owner.destroy_simulator(handle)


@pytest.mark.parametrize("method", [maestro.SimulationType.Statevector,
    maestro.SimulationType.MatrixProductState, maestro.SimulationType.MatrixProductOperator])
def test_real_batches(method):
    with simulator(method) as sim:
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


def test_complex_batches():
    with simulator(maestro.SimulationType.MatrixProductOperator) as sim:
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


def test_complex_unsupported_backend():
    with simulator(maestro.SimulationType.Statevector) as sim:
        with pytest.raises(RuntimeError, match="ExpectationValueComplex"):
            sim.expectation_values_complex(["II"])
