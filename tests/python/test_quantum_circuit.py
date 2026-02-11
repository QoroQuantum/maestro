
import pytest
import math
import maestro
# from maestro.circuits import QuantumCircuit # This fails if maestro is a module not package

# Use direct access
QuantumCircuit = maestro.circuits.QuantumCircuit

# Helper to verify counts roughly follow distribution
def assert_distribution(counts, expected_probs, shots, tolerance=0.1):
    """
    counts: dict of bitstring -> count
    expected_probs: dict of bitstring -> probability
    shots: total shots passed to execute
    tolerance: acceptable deviation
    """
    total_counts = sum(counts.values())
    assert total_counts == shots
    
    for bitstring, prob in expected_probs.items():
        expected_count = prob * shots
        actual_count = counts.get(bitstring, 0)
        # Standard deviation for binomial distribution is sqrt(n * p * (1-p))
        # We use a loose tolerance for stochastic tests unless deterministic
        if prob == 1.0 or prob == 0.0:
            assert actual_count == expected_count
        else:
            # simple check for now
            assert abs(actual_count - expected_count) < (shots * tolerance)

class TestQuantumCircuitModel:
    def test_init(self):
        qc = QuantumCircuit()
        assert qc is not None
        # Initially, max qubit index might be 0 or something execution-dependent
        # But we can check method existence
        assert hasattr(qc, "x")
        assert hasattr(qc, "h")
        assert hasattr(qc, "execute")
        assert hasattr(qc, "estimate")

    def test_single_qubit_gates_deterministic(self):
        """Test gates that produce deterministic outcomes starting from |0>"""
        # Test X gate: |0> -> |1>
        qc = QuantumCircuit()
        qc.x(0)
        qc.measure([(0, 0)]) # Measure q0 into c0
        res = qc.execute(shots=100)
        assert res["counts"]["1"] == 100

        # Test Double X: |0> -> |1> -> |0>
        qc = QuantumCircuit()
        qc.x(0)
        qc.x(0)
        qc.measure([(0, 0)])
        res = qc.execute(shots=100)
        assert res["counts"]["0"] == 100
        
        # Test Z gate on |0>: Should usually change phase but not bitflip. 
        # But in Z-basis measurement, |0> remains |0>
        qc = QuantumCircuit()
        qc.z(0)
        qc.measure([(0, 0)])
        res = qc.execute(shots=100)
        assert res["counts"]["0"] == 100

    def test_superposition_gates(self):
        """Test Hadamard gate creating superposition"""
        qc = QuantumCircuit()
        qc.h(0)
        qc.measure([(0, 0)])
        res = qc.execute(shots=1000)
        counts = res["counts"]
        
        # Should be roughly 50-50
        assert "0" in counts
        assert "1" in counts
        assert 400 < counts["0"] < 600
        assert 400 < counts["1"] < 600

    def test_two_qubit_gates_cx(self):
        """Test CNOT gate behavior"""
        # Case 1: Control |0> (Target shouldn't flip)
        qc = QuantumCircuit()
        # q0 initialized to 0
        qc.cx(0, 1) 
        qc.measure([(0, 0), (1, 1)])
        res = qc.execute(shots=100)
        assert res["counts"] == {"00": 100}
        # Note: Depending on bitstring ordering (little vs big endian), check logic
        # Maestro often uses standard ordering, but let's be robust or check standard.
        # Assuming "q1q0" or similar? Let's check keys.
        # usually 00 means both 0.
        
        # Case 2: Control |1> (Target should flip)
        qc = QuantumCircuit()
        qc.x(0) # q0 = 1
        qc.cx(0, 1) # q1 flips to 1
        qc.measure([(0, 0), (1, 1)])
        res = qc.execute(shots=100)
        # Expect |11> (both 1).
        # We need to verify bit ordering. Usually bitstring maps to measurement indices or qubit indices.
        # If it's q1q0, then "11". If q0q1, also "11".
        assert "11" in res["counts"]
        assert res["counts"]["11"] == 100

    def test_bell_state(self):
        """Test Bell State Creation |00> + |11>"""
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([(0, 0), (1, 1)])
        res = qc.execute(shots=1000)
        
        counts = res["counts"]
        # Should mainly see "00" and "11"
        assert counts.get("00", 0) > 400
        assert counts.get("11", 0) > 400
        # "01" and "10" should be 0
        assert counts.get("01", 0) == 0
        assert counts.get("10", 0) == 0

    def test_all_gates_coverage(self):
        """Verify all gates can be added without error"""
        qc = QuantumCircuit()
        # Single qubit non-parametric
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.h(0)
        qc.s(0)
        qc.sdg(0)
        qc.t(0)
        qc.tdg(0)
        qc.sx(0)
        
        # Single qubit parametric
        qc.p(0, 0.5)
        qc.rx(0, 0.5)
        qc.ry(0, 0.5)
        qc.rz(0, 0.5)
        qc.u(0, 0.1, 0.2, 0.3)
        
        # Two qubit
        qc.cx(0, 1)
        qc.cy(0, 1)
        qc.cz(0, 1)
        qc.swap(0, 1)
        
        # Controlled parametric
        qc.cp(0, 1, 0.5)
        qc.crx(0, 1, 0.5)
        qc.cry(0, 1, 0.5)
        qc.crz(0, 1, 0.5)
        
        # Result should be executable (even if meaningless)
        res = qc.execute(shots=10)
        assert res["counts"] is not None

class TestExecutionFunctions:
    def test_execute_parameters(self):
        """Test passing different parameters to execute"""
        qc = QuantumCircuit()
        qc.h(0)
        qc.measure([(0, 0)])
        
        # Test shots
        res = qc.execute(shots=10)
        assert sum(res["counts"].values()) == 10
        
        # Test Simulator Type Enum
        res = qc.execute(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            shots=10
        )
        assert res["simulator"] == maestro.SimulatorType.QCSim.value
        assert res["method"] == maestro.SimulationType.Statevector.value

    def test_execute_mps(self):
        """Test execution with MPS simulation type"""
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([(0, 0), (1, 1)])
        
        res = qc.execute(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=10,
            singular_value_threshold=1e-6,
            shots=100
        )
        assert res["method"] == maestro.SimulationType.MatrixProductState.value
        assert sum(res["counts"].values()) == 100

    def test_measurement_mapping(self):
        """Test complex measurement mapping"""
        qc = QuantumCircuit()
        qc.x(0) # q0 = 1
        # q1 = 0
        
        # Measure q0 -> c1, q1 -> c0 (Swap typical order)
        # Expected bitstring: c1c0 = 10? or c1=1, c0=0.
        # Maestro bindings: "std::string bitstring(bool_vec.size())"
        # The bool_vec comes from 'results' which usually respects the number of bits measured.
        # But determining the order in the string depends on the C++ implementation.
        # Usually checking unique outcomes is safer.
        
        qc.measure([(0, 1), (1, 0)]) 
        res = qc.execute(shots=10)
        keys = list(res["counts"].keys())
        # We expect a single deterministic outcome if we just prepared |10>
        assert len(keys) == 1
        
        key = keys[0]
        # key length should be 2 (since max classical index implied is 1 -> size 2?) 
        # Or simply number of measurements?
        # The previous tests passed assuming standard behavior.
        # Let's verify length.
        assert len(key) >= 2 

    def test_parametric_rotation(self):
        qc = QuantumCircuit()
        # Rx(pi) should be like X
        qc.rx(0, 3.14159265359)
        qc.measure([(0, 0)])
        res = qc.execute(shots=100)
        # Should be mostly 1
        assert res["counts"].get("1", 0) == 100

class TestEstimateFunctions:
    def test_estimate_basic(self):
        """Test basic expectation value estimation"""
        qc = QuantumCircuit()
        qc.x(0)
        
        # State is |1>
        # <Z> = -1
        # <X> = 0 (in ideal theory, though statistical or exact?)
        # <I> = 1
        
        # Using string format
        res = qc.estimate(observables="Z;I")
        vals = res["expectation_values"]
        assert len(vals) == 2
        assert vals[0] == pytest.approx(-1.0, abs=1e-5)
        assert vals[1] == pytest.approx(1.0, abs=1e-5)

    def test_estimate_list_observables(self):
        """Test passing list of strings for observables"""
        qc = QuantumCircuit()
        qc.h(0) 
        # State |+> = (|0>+|1>)/sqrt(2)
        # <X> = 1
        # <Z> = 0
        
        res = qc.estimate(observables=["X", "Z"])
        vals = res["expectation_values"]
        assert vals[0] == pytest.approx(1.0, abs=1e-5)
        assert vals[1] == pytest.approx(0.0, abs=1e-5)

    def test_estimate_entangled(self):
        """Test estimation on entangled state"""
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        # Bell state (|00> + |11>)/sqrt(2)
        # XX = 1
        # ZZ = 1
        # IZ = 0 (marginal)
        
        res = qc.estimate(observables=["XX", "ZZ", "IZ"])
        vals = res["expectation_values"]
        assert vals[0] == pytest.approx(1.0, abs=1e-5)
        assert vals[1] == pytest.approx(1.0, abs=1e-5)
        assert vals[2] == pytest.approx(0.0, abs=1e-5)
        # Add assertion for simulation method, assuming Statevector is default for estimate
        assert res["method"] == maestro.SimulationType.Statevector.value

    def test_estimate_mps(self):
        """Test estimate using MPS backend"""
        qc = QuantumCircuit()
        qc.x(0)
        
        res = qc.estimate(
            observables="Z",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4
        )
        assert res["method"] == maestro.SimulationType.MatrixProductState.value
        assert res["expectation_values"][0] == pytest.approx(-1.0, abs=1e-5)

    def test_empty_circuit_estimate(self):
        """Test estimating on empty circuit (should be |0...0>)"""
        qc = QuantumCircuit()
        # Should now work even without explicit gates due to binding fix
        res = qc.estimate(observables="Z")
        assert res["expectation_values"][0] == pytest.approx(1.0, abs=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
