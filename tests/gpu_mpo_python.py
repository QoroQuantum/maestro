"""Run against either built extension: python gpu_mpo_python.py [--composer]."""
import sys
import numpy as np

composer = "--composer" in sys.argv
cpu = "--cpu" in sys.argv
if composer:
    import ComposerPython as api
else:
    import maestro as api


def make(width):
    if composer:
        sim = api.SimulatorsFactory.CreateSimulator(
            api.SimulatorType.QCSim if cpu else api.SimulatorType.Gpu, api.SimulationType.MatrixProductOperator)
        owner = None
    else:
        owner = api.Maestro()
        handle = owner.create_simulator(
            api.SimulatorType.QCSim if cpu else api.SimulatorType.Gpu, api.SimulationType.MatrixProductOperator)
        sim = owner.get_simulator(handle)
    assert sim is not None
    sim.Configure("use_double_precision", "true")
    sim.Configure("gate_fusion", "true")
    sim.AllocateQubits(width)
    sim.Initialize()
    return owner, sim


def method(sim, maestro_name, composer_name):
    return getattr(sim, composer_name if composer else maestro_name)


owner, wide = make(70)
wide.ApplyX(69)
wide.ApplyX(63)
expected = tuple(i in (63, 69) for i in range(70))
sample = method(wide, "SampleCountsMany", "SampleCountsManyWithList")
assert sample(list(range(70)), 5) == {expected: 5}
assert sample([69, 0, 63, 69], 5) == {(True, False, True, True): 5}
assert method(wide, "probability_bits", "ProbabilityBits")(list(expected)) == 1.0
assert abs(method(wide, "density_matrix_element_bits", "DensityMatrixElementBits")(
    list(expected), list(expected)) - 1.0) < 1e-8
# Preserve owner until its simulator is no longer used.
del wide, owner

owner, sim = make(2)
sim.ApplyH(0)
sim.ApplyCX(0, 1)
projector = np.array([[0, 0], [0, 1]], dtype=np.complex128)
apply = method(sim, "apply_operator", "ApplyOperator")
apply([0], projector.tolist() if composer else projector)
trace = method(sim, "density_matrix_trace", "DensityMatrixTrace")
dense = method(sim, "get_density_matrix", "GetDensityMatrix")
expectation = method(sim, "expectation_value_complex", "ExpectationValueComplex")
assert abs(trace() - 0.5) < 1e-8
assert abs(np.asarray(dense(False))[3, 3] - 0.5) < 1e-8
assert abs(np.asarray(dense())[3, 3] - 1.0) < 1e-8
assert abs(expectation("ZI") + 1.0) < 1e-8
assert abs(expectation("ZI", False) + 0.5) < 1e-8
method(sim, "move_at_beginning_of_chain", "MoveAtBeginningOfChain")([1])
method(sim, "recanonicalize", "ReCanonicalize")()
method(sim, "trim", "Trim")()
apply([0], projector.tolist() if composer else projector, True)
assert abs(trace() - 1.0) < 1e-8
for key in ("restore_trace", "hermitize"):
    key = "matrix_product_operator_" + key + "_after_truncation"
    sim.Configure(key, "true")
    assert sim.GetConfiguration(key) == "true"
print("PASS", "Composer" if composer else "Maestro", "Python MPO API and 70-qubit sampling")

# Check precision forwarding through both Python extension implementations.
selected_backend = api.SimulatorType.QiskitAer if cpu else api.SimulatorType.Gpu
methods = [api.SimulationType.StateVector if composer else api.SimulationType.Statevector,
           api.SimulationType.DensityMatrix]
if not cpu:
    methods += [api.SimulationType.MatrixProductState,
                api.SimulationType.MatrixProductOperator,
                api.SimulationType.TensorNetwork]
for selected_method in methods:
    for precision in ("single", "double"):
        if composer:
            sim = api.SimulatorsFactory.CreateSimulator(selected_backend, selected_method)
            owner = None
        else:
            owner = api.Maestro()
            handle = owner.create_simulator(selected_backend, selected_method)
            sim = owner.get_simulator(handle)
        sim.Configure("precision", "double" if precision == "single" else "single")
        sim.Configure("use_double_precision", "1" if precision == "double" else "0")
        sim.AllocateQubits(2)
        sim.Initialize()
        assert sim.GetConfiguration("precision") == precision
        sim.Configure("precision", precision)
        sim.ApplyRy(0, 0.713)
        sim.ApplyCX(0, 1)
        tolerance = 1e-10 if precision == "double" else 2e-5
        expected = np.sin(0.713 / 2) ** 2
        assert abs(sim.Probability(3) - expected) < tolerance
        sim.SaveState()
        sim.ApplyX(1)
        sim.RestoreState()
        assert abs(sim.Probability(3) - expected) < tolerance
        # Ignored after allocation: the data type stays as allocated.
        sim.Configure("precision", "double" if precision == "single" else "single")
        assert sim.GetConfiguration("precision") == precision
        del sim, owner
print("PASS", "Composer" if composer else "Maestro", "Python precision forwarding")
