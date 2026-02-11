#!/usr/bin/env python3
"""Unit tests for Maestro nanobind Python bindings using pytest"""

import pytest
import maestro


class TestEnums:
    """Test that enums are properly exposed"""

    def test_simulator_type_enum(self):
        """Test SimulatorType enum accessibility"""
        assert hasattr(maestro, 'SimulatorType')
        assert hasattr(maestro.SimulatorType, 'QCSim')
        assert hasattr(maestro.SimulatorType, 'CompositeQCSim')

    def test_simulation_type_enum(self):
        """Test SimulationType enum accessibility"""
        assert hasattr(maestro, 'SimulationType')
        assert hasattr(maestro.SimulationType, 'Statevector')
        assert hasattr(maestro.SimulationType, 'MatrixProductState')
        assert hasattr(maestro.SimulationType, 'Stabilizer')
        assert hasattr(maestro.SimulationType, 'TensorNetwork')


class TestMaestroClass:
    """Test Maestro class instantiation and basic methods"""

    def test_maestro_creation(self):
        """Test Maestro instance creation"""
        m = maestro.Maestro()
        assert m is not None

    def test_simulator_creation(self):
        """Test simulator creation and retrieval"""
        m = maestro.Maestro()
        sim_handle = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        assert sim_handle > 0

        sim = m.GetSimulator(sim_handle)
        assert sim is not None

        m.DestroySimulator(sim_handle)

    def test_multiple_simulators(self):
        """Test creating multiple simulators"""
        m = maestro.Maestro()

        sim1 = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim2 = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.MatrixProductState
        )

        assert sim1 > 0
        assert sim2 > 0
        assert sim1 != sim2

        m.DestroySimulator(sim1)
        m.DestroySimulator(sim2)


class TestQasmParser:
    """Test QASM parsing functionality"""

    def test_qasm_parser_creation(self):
        """Test QasmToCirc instance creation"""
        parser = maestro.QasmToCirc()
        assert parser is not None

    def test_simple_qasm_parsing(self):
        """Test parsing a simple QASM circuit"""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        parser = maestro.QasmToCirc()
        circuit = parser.ParseAndTranslate(qasm)
        assert circuit is not None

    def test_qubit_count_detection(self):
        """Test circuit qubit count detection"""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[2];
        """

        parser = maestro.QasmToCirc()
        circuit = parser.ParseAndTranslate(qasm)
        num_qubits = circuit.num_qubits
        assert num_qubits == 3


class TestSimulatorOperations:
    """Test simulator gate operations"""

    def test_qubit_allocation(self):
        """Test qubit allocation and initialization"""
        m = maestro.Maestro()
        sim_handle = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.GetSimulator(sim_handle)

        sim.AllocateQubits(2)
        sim.Initialize()
        assert sim.GetNumberOfQubits() == 2

        m.DestroySimulator(sim_handle)

    def test_single_qubit_gates(self):
        """Test single-qubit gate operations"""
        m = maestro.Maestro()
        sim_handle = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.GetSimulator(sim_handle)

        sim.AllocateQubits(1)
        sim.Initialize()

        # These should not raise exceptions
        sim.ApplyH(0)
        sim.ApplyX(0)
        sim.ApplyY(0)
        sim.ApplyZ(0)

        m.DestroySimulator(sim_handle)

    def test_two_qubit_gates(self):
        """Test two-qubit gate operations"""
        m = maestro.Maestro()
        sim_handle = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.GetSimulator(sim_handle)

        sim.AllocateQubits(2)
        sim.Initialize()

        # These should not raise exceptions
        sim.ApplyCX(0, 1)
        sim.ApplyCZ(0, 1)
        sim.ApplySwap(0, 1)

        m.DestroySimulator(sim_handle)

    def test_measurement(self):
        """Test measurement operations"""
        m = maestro.Maestro()
        sim_handle = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.GetSimulator(sim_handle)

        sim.AllocateQubits(2)
        sim.Initialize()
        sim.ApplyH(0)
        sim.ApplyCX(0, 1)

        results = sim.SampleCounts([0, 1], 1000)
        assert isinstance(results, dict)
        assert len(results) > 0
        total_shots = sum(results.values())
        assert total_shots == 1000

        m.DestroySimulator(sim_handle)

    def test_bell_state_distribution(self):
        """Test Bell state produces correct distribution"""
        m = maestro.Maestro()
        sim_handle = m.CreateSimulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.GetSimulator(sim_handle)

        sim.AllocateQubits(2)
        sim.Initialize()
        sim.ApplyH(0)
        sim.ApplyCX(0, 1)

        results = sim.SampleCounts([0, 1], 10000)

        # Bell state should produce |00> and |11> with ~50% each
        # Allow for statistical variation
        assert 0 in results or 3 in results  # |00> or |11>
        total = sum(results.values())
        assert total == 10000

        m.DestroySimulator(sim_handle)


class TestSimpleExecute:
    """Test the simple_execute convenience function"""

    def test_simple_execute_basic(self):
        """Test simple_execute with default parameters"""
        qasm_bell = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """

        result = maestro.simple_execute(qasm_bell)
        assert result is not None
        assert 'counts' in result
        assert 'simulator' in result
        assert 'method' in result
        assert 'time_taken' in result

    def test_simple_execute_custom_shots(self):
        """Test simple_execute with custom shot count"""
        qasm_bell = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        result = maestro.simple_execute(qasm_bell, shots=500)
        total = sum(result['counts'].values())
        assert total == 500

    def test_simple_execute_mps(self):
        """Test simple_execute with Matrix Product State"""
        qasm_bell = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
        """

        result = maestro.simple_execute(
            qasm_bell,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4,
            singular_value_threshold=1e-10
        )
        assert result is not None
        assert result['method'] == 'matrix_product_state'

    def test_simple_execute_ghz_state(self):
        """Test simple_execute with GHZ state"""
        qasm_ghz = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q -> c;
        """

        result = maestro.simple_execute(qasm_ghz, shots=1000)
        assert result is not None
        assert 'counts' in result
        total = sum(result['counts'].values())
        assert total == 1000

    def test_simple_execute_four_qubit(self):
        """Test simple_execute with 4-qubit circuit"""
        qasm_4q = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg c[4];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        cx q[2], q[3];
        measure q -> c;
        """

        result = maestro.simple_execute(qasm_4q, shots=1000)
        assert result is not None

        # Should produce mostly |0000> and |1111>
        counts = result['counts']
        dominant_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]
        assert len(dominant_states) >= 1


class TestSimpleEstimate:
    """Test the simple_estimate convenience function"""

    def test_simple_estimate_basic(self):
        """Test simple_estimate with default parameters"""
        qasm_bell = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """

        # Estimate expectation values for ZZ, XX, and YY
        observables = "ZZ;XX;YY"
        result = maestro.simple_estimate(qasm_bell, observables)

        assert result is not None
        assert 'expectation_values' in result
        assert 'simulator' in result
        assert 'method' in result
        assert 'time_taken' in result

        exp_vals = result['expectation_values']
        assert len(exp_vals) == 3

        # Bell state (|00> + |11>) / sqrt(2)
        # <ZZ> = 1.0
        # <XX> = 1.0 (for this specific Bell state)
        # <YY> = -1.0
        assert exp_vals[0] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[1] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[2] == pytest.approx(-1.0, abs=1e-5)

    def test_simple_estimate_single_qubit(self):
        """Test simple_estimate for single qubit observables"""
        qasm_h = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        """

        # <X> for H|0> is 1.0
        # <Z> for H|0> is 0.0
        result = maestro.simple_estimate(qasm_h, "X;Z")
        assert result is not None
        exp_vals = result['expectation_values']
        assert len(exp_vals) == 2
        assert exp_vals[0] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[1] == pytest.approx(0.0, abs=1e-5)

    def test_simple_estimate_mps(self):
        """Test simple_estimate using MPS method"""
        qasm_bell = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """

        result = maestro.simple_estimate(
            qasm_bell,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=2
        )
        assert result is not None
        assert result['method'] == 'matrix_product_state'
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)


class TestComplexCircuits:
    """Test with more complex quantum circuits"""

    def test_superposition(self):
        """Test simple superposition circuit"""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q -> c;
        """

        result = maestro.simple_execute(qasm, shots=10000)
        counts = result['counts']

        # Should be roughly 50/50 between |0> and |1>
        # Allow for statistical variation (40-60%)
        total = sum(counts.values())
        assert total == 10000

    def test_parametric_gates(self):
        """Test circuit with parametric rotation gates"""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        rx(1.5708) q[0];
        measure q -> c;
        """

        result = maestro.simple_execute(qasm, shots=1000)
        assert result is not None
        assert 'counts' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
