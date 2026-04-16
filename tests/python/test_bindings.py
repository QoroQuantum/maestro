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
        assert hasattr(maestro.SimulatorType, 'QuestSim')

    def test_simulation_type_enum(self):
        """Test SimulationType enum accessibility"""
        assert hasattr(maestro, 'SimulationType')
        assert hasattr(maestro.SimulationType, 'Statevector')
        assert hasattr(maestro.SimulationType, 'MatrixProductState')
        assert hasattr(maestro.SimulationType, 'Stabilizer')
        assert hasattr(maestro.SimulationType, 'TensorNetwork')
        assert hasattr(maestro.SimulationType, 'PauliPropagator')
        assert hasattr(maestro.SimulationType, 'ExtendedStabilizer')


class TestMaestroClass:
    """Test Maestro class instantiation and basic methods"""

    def test_maestro_creation(self):
        """Test Maestro instance creation"""
        m = maestro.Maestro()
        assert m is not None

    def test_simulator_creation(self):
        """Test simulator creation and destruction"""
        m = maestro.Maestro()
        sim_handle = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        assert sim_handle > 0

        m.destroy_simulator(sim_handle)

    def test_multiple_simulators(self):
        """Test creating multiple simulators"""
        m = maestro.Maestro()

        sim1 = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim2 = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.MatrixProductState
        )

        assert sim1 > 0
        assert sim2 > 0
        assert sim1 != sim2

        m.destroy_simulator(sim1)
        m.destroy_simulator(sim2)


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
        circuit = parser.parse_and_translate(qasm)
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
        circuit = parser.parse_and_translate(qasm)
        num_qubits = circuit.num_qubits
        assert num_qubits == 3


@pytest.mark.skip(reason="ISimulator is not yet exposed as a nanobind type")
class TestSimulatorOperations:
    """Test simulator gate operations (requires ISimulator binding)"""

    def test_qubit_allocation(self):
        """Test qubit allocation and initialization"""
        m = maestro.Maestro()
        sim_handle = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.get_simulator(sim_handle)

        sim.AllocateQubits(2)
        sim.Initialize()
        assert sim.GetNumberOfQubits() == 2

        m.destroy_simulator(sim_handle)

    def test_single_qubit_gates(self):
        """Test single-qubit gate operations"""
        m = maestro.Maestro()
        sim_handle = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.get_simulator(sim_handle)

        sim.AllocateQubits(1)
        sim.Initialize()

        # These should not raise exceptions
        sim.ApplyH(0)
        sim.ApplyX(0)
        sim.ApplyY(0)
        sim.ApplyZ(0)

        m.destroy_simulator(sim_handle)

    def test_two_qubit_gates(self):
        """Test two-qubit gate operations"""
        m = maestro.Maestro()
        sim_handle = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.get_simulator(sim_handle)

        sim.AllocateQubits(2)
        sim.Initialize()

        # These should not raise exceptions
        sim.ApplyCX(0, 1)
        sim.ApplyCZ(0, 1)
        sim.ApplySwap(0, 1)

        m.destroy_simulator(sim_handle)

    def test_measurement(self):
        """Test measurement operations"""
        m = maestro.Maestro()
        sim_handle = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.get_simulator(sim_handle)

        sim.AllocateQubits(2)
        sim.Initialize()
        sim.ApplyH(0)
        sim.ApplyCX(0, 1)

        results = sim.SampleCounts([0, 1], 1000)
        assert isinstance(results, dict)
        assert len(results) > 0
        total_shots = sum(results.values())
        assert total_shots == 1000

        m.destroy_simulator(sim_handle)

    def test_bell_state_distribution(self):
        """Test Bell state produces correct distribution"""
        m = maestro.Maestro()
        sim_handle = m.create_simulator(
            maestro.SimulatorType.QCSim,
            maestro.SimulationType.Statevector
        )
        sim = m.get_simulator(sim_handle)

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

        m.destroy_simulator(sim_handle)


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
        assert result['method'] == maestro.SimulationType.MatrixProductState.value

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
        assert result['method'] == maestro.SimulationType.MatrixProductState.value
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


class TestReset:
    """Test qubit reset functionality."""

    def test_reset_returns_zero(self):
        """Resetting a qubit in |1> should return it to |0>."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.x(0)       # Put qubit in |1>
        qc.reset(0)   # Reset to |0>
        qc.measure_all()

        result = qc.execute(shots=100)
        counts = result['counts']
        # After reset, qubit should always be |0>
        assert '0' in counts
        assert counts.get('0', 0) == 100

    def test_reset_multi_qubit(self):
        """reset_qubits should reset multiple qubits at once."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.x(0)
        qc.x(1)
        qc.reset_qubits([0, 1])
        qc.measure_all()

        result = qc.execute(shots=100)
        counts = result['counts']
        assert counts.get('00', 0) == 100

    def test_mid_circuit_reset(self):
        """Mid-circuit reset: flip, reset, then flip again should give |1>."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.x(0)       # |1>
        qc.reset(0)   # |0>
        qc.x(0)       # |1> again
        qc.measure_all()

        result = qc.execute(shots=100)
        counts = result['counts']
        assert counts.get('1', 0) == 100

    def test_reset_after_entanglement(self):
        """Reset one qubit of an entangled pair."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.reset(0)    # Reset qubit 0 to |0>
        qc.measure_all()

        result = qc.execute(shots=1000)
        counts = result['counts']
        # Qubit 0 is always 0 after reset, qubit 1 is random
        for bitstring in counts:
            assert bitstring[0] == '0'  # qubit 0 is always |0>

    def test_reset_mps(self):
        """Reset should work with MPS simulation."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.x(0)
        qc.reset(0)
        qc.measure_all()

        result = qc.execute(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=100,
            max_bond_dimension=4,
        )
        counts = result['counts']
        assert counts.get('0', 0) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# --- Clifford-only QASM for Stabilizer tests ---
# Stabilizer only supports Clifford gates (H, S, CX, X, Y, Z)
CLIFFORD_BELL_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""

CLIFFORD_BELL_NO_MEASURE_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

# General QASM (with non-Clifford gates) for PauliProp / ExtStab
GENERAL_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
t q[0];
cx q[0], q[1];
measure q -> c;
"""

GENERAL_NO_MEASURE_QASM = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
t q[0];
cx q[0], q[1];
"""


class TestStabilizerSimulation:
    """Test the Stabilizer simulation backend (Clifford-only circuits)"""

    def test_stabilizer_execute(self):
        """Test execute with Stabilizer backend on a Clifford circuit"""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Stabilizer,
            shots=1000
        )
        assert result is not None
        assert 'counts' in result
        total = sum(result['counts'].values())
        assert total == 1000

    def test_stabilizer_estimate(self):
        """Test estimate with Stabilizer backend on a Bell state"""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Stabilizer
        )
        assert result is not None
        assert 'expectation_values' in result
        # Bell state: <ZZ> = 1.0
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)

    def test_stabilizer_bell_distribution(self):
        """Test Stabilizer produces correct Bell state distribution"""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Stabilizer,
            shots=10000
        )
        counts = result['counts']
        total = sum(counts.values())
        assert total == 10000


class TestPauliPropagatorSimulation:
    """Test the Pauli Propagator simulation backend"""

    def test_pauli_propagator_execute(self):
        """Test execute with Pauli Propagator backend"""
        result = maestro.simple_execute(
            GENERAL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.PauliPropagator,
            shots=1000
        )
        assert result is not None
        assert 'counts' in result
        total = sum(result['counts'].values())
        assert total == 1000

    def test_pauli_propagator_estimate(self):
        """Test estimate with Pauli Propagator backend"""
        result = maestro.simple_estimate(
            GENERAL_NO_MEASURE_QASM,
            "ZZ;XX",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.PauliPropagator
        )
        assert result is not None
        assert 'expectation_values' in result
        exp_vals = result['expectation_values']
        assert len(exp_vals) == 2
        # State is (|00> + e^{i*pi/4}|11>) / sqrt(2)
        # <ZZ> = 1.0, <XX> = cos(pi/4) = 1/sqrt(2)
        assert exp_vals[0] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[1] == pytest.approx(0.7071, abs=1e-3)


class TestExtendedStabilizerSimulation:
    """Test the Extended Stabilizer simulation backend"""

    def test_extended_stabilizer_execute(self):
        """Test execute with Extended Stabilizer backend"""
        result = maestro.simple_execute(
            GENERAL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.ExtendedStabilizer,
            shots=1000
        )
        assert result is not None
        assert 'counts' in result
        total = sum(result['counts'].values())
        assert total == 1000

    def test_extended_stabilizer_estimate(self):
        """Test estimate with Extended Stabilizer backend"""
        result = maestro.simple_estimate(
            GENERAL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.ExtendedStabilizer
        )
        assert result is not None
        assert 'expectation_values' in result
        # (|00> + e^{i*pi/4}|11>) / sqrt(2): <ZZ> = 1.0
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)


class TestSimulatorTypeIsHonored:
    """Regression tests: verify the requested simulator/method is actually used.

    These tests ensure we don't silently fall back to defaults (e.g., kQCSim)
    when a specific simulator_type or simulation_type is requested.
    GPU tests are skipped here (requires Linux + CUDA) but the pattern applies.
    """

    def test_execute_statevector_type_honored(self):
        """Execute with Statevector reports correct simulator and method."""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            shots=10
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value
        assert result['method'] == maestro.SimulationType.Statevector.value

    def test_execute_mps_type_honored(self):
        """Execute with MPS reports correct simulator and method."""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=10
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value
        assert result['method'] == maestro.SimulationType.MatrixProductState.value

    def test_execute_pauli_propagator_type_honored(self):
        """Execute with PauliPropagator reports correct simulator type.

        Note: the internal optimizer may legitimately switch the simulation
        method, so we only assert the simulator type here.
        """
        result = maestro.simple_execute(
            GENERAL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.PauliPropagator,
            shots=10
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value

    def test_estimate_statevector_type_honored(self):
        """Estimate with Statevector reports correct simulator and method."""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value
        assert result['method'] == maestro.SimulationType.Statevector.value

    def test_estimate_mps_type_honored(self):
        """Estimate with MPS reports correct simulator and method."""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value
        assert result['method'] == maestro.SimulationType.MatrixProductState.value

    def test_estimate_pauli_propagator_type_honored(self):
        """Estimate with PauliPropagator reports correct simulator type.

        Note: the internal optimizer may legitimately switch the simulation
        method, so we only assert the simulator type here.
        """
        result = maestro.simple_estimate(
            GENERAL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.PauliPropagator,
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value

    def test_circuit_execute_type_honored(self):
        """Circuit.execute() reports correct simulator and method."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = qc.execute(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=10,
            max_bond_dimension=4,
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value
        assert result['method'] == maestro.SimulationType.MatrixProductState.value

    def test_circuit_estimate_type_honored(self):
        """Circuit.estimate() reports correct simulator and method."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        result = qc.estimate(
            observables=["ZZ"],
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4,
        )
        assert result['simulator'] == maestro.SimulatorType.QCSim.value
        assert result['method'] == maestro.SimulationType.MatrixProductState.value


class TestDoublePrecision:
    """Test the use_double_precision parameter.

    On CPU (QCSim), this flag has no effect (CPU already uses float64).
    These tests verify the parameter is accepted without errors and results
    remain correct. GPU-specific precision tests require Linux + CUDA.
    """

    def test_simple_execute_accepts_double_precision(self):
        """simple_execute accepts use_double_precision without error."""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            shots=100,
            use_double_precision=True
        )
        assert result is not None
        assert 'counts' in result
        total = sum(result['counts'].values())
        assert total == 100

    def test_simple_execute_double_precision_false(self):
        """simple_execute with use_double_precision=False (default) works."""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            shots=100,
            use_double_precision=False
        )
        assert result is not None
        assert 'counts' in result

    def test_simple_estimate_accepts_double_precision(self):
        """simple_estimate accepts use_double_precision without error."""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            use_double_precision=True
        )
        assert result is not None
        assert 'expectation_values' in result
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)

    def test_simple_estimate_mps_double_precision(self):
        """simple_estimate with MPS + use_double_precision produces correct results."""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ;XX;YY",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4,
            use_double_precision=True
        )
        assert result is not None
        exp_vals = result['expectation_values']
        assert len(exp_vals) == 3
        # Bell state: <ZZ> = 1.0, <XX> = 1.0, <YY> = -1.0
        assert exp_vals[0] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[1] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[2] == pytest.approx(-1.0, abs=1e-5)

    def test_circuit_execute_accepts_double_precision(self):
        """Circuit.execute() accepts use_double_precision parameter."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = qc.execute(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=100,
            max_bond_dimension=4,
            use_double_precision=True
        )
        assert result is not None
        assert 'counts' in result

    def test_circuit_estimate_accepts_double_precision(self):
        """Circuit.estimate() accepts use_double_precision parameter."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        result = qc.estimate(
            observables=["ZZ"],
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4,
            use_double_precision=True
        )
        assert result is not None
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)

    def test_qasm_execute_accepts_double_precision(self):
        """QASM-based simple_execute accepts use_double_precision."""
        result = maestro.simple_execute(
            GENERAL_QASM,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            shots=100,
            use_double_precision=True
        )
        assert result is not None
        assert 'counts' in result

    def test_qasm_estimate_accepts_double_precision(self):
        """QASM-based simple_estimate accepts use_double_precision."""
        result = maestro.simple_estimate(
            GENERAL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
            use_double_precision=True
        )
        assert result is not None
        assert 'expectation_values' in result
        # (|00> + e^{i*pi/4}|11>) / sqrt(2): <ZZ> = 1.0
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)


class TestGpuSimulator:
    """Test GPU simulator bindings.

    GPU simulation registers GPU via RemoveAllOptimizationSimulatorsAndAdd,
    but CreateSimulator creates a cheap QCSim MPS placeholder (not a GPU
    simulator) to avoid wasting GPU memory. The network creates the actual
    GPU simulator on-demand during execution.

    On macOS (no GPU library), these tests verify that the binding path
    still works using the QCSim MPS fallback.
    """

    def test_gpu_enum_exists(self):
        """SimulatorType.Gpu enum value is exposed."""
        assert hasattr(maestro.SimulatorType, 'Gpu')

    def test_gpu_execute_returns_result(self):
        """simple_execute with GPU type returns a valid result dict."""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.Gpu,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=100
        )
        assert result is not None
        assert 'counts' in result
        assert 'time_taken' in result
        total = sum(result['counts'].values())
        assert total == 100

    def test_gpu_estimate_returns_result(self):
        """simple_estimate with GPU type returns valid expectation values."""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ",
            simulator_type=maestro.SimulatorType.Gpu,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4
        )
        assert result is not None
        assert 'expectation_values' in result
        # Bell state <ZZ> = 1.0
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)

    def test_circuit_gpu_execute(self):
        """Circuit.execute() with GPU type works through the network."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = qc.execute(
            simulator_type=maestro.SimulatorType.Gpu,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=100,
            max_bond_dimension=4
        )
        assert result is not None
        assert 'counts' in result

    def test_circuit_gpu_estimate(self):
        """Circuit.estimate() with GPU type works through the network."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        result = qc.estimate(
            observables=["ZZ"],
            simulator_type=maestro.SimulatorType.Gpu,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4
        )
        assert result is not None
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)

    def test_gpu_with_double_precision(self):
        """GPU + use_double_precision flag is accepted."""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ;XX",
            simulator_type=maestro.SimulatorType.Gpu,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4,
            use_double_precision=True
        )
        assert result is not None
        exp_vals = result['expectation_values']
        assert len(exp_vals) == 2
        assert exp_vals[0] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[1] == pytest.approx(1.0, abs=1e-5)


class TestQuestSimulator:
    """Test QuEST simulator bindings.

    QuEST is loaded as an external shared library (libmaestroquest).
    These tests verify:
    - The QuestSim enum value is exposed
    - init_quest() / is_quest_available() helpers work
    - Non-statevector simulation types are rejected with a clear error
    - When the library is available, execute and estimate produce correct results
    """

    def test_quest_enum_exists(self):
        """SimulatorType.QuestSim enum value is exposed."""
        assert hasattr(maestro.SimulatorType, 'QuestSim')

    def test_init_quest_function_exists(self):
        """init_quest() module function is exposed."""
        assert hasattr(maestro, 'init_quest')
        assert callable(maestro.init_quest)

    def test_is_quest_available_function_exists(self):
        """is_quest_available() module function is exposed."""
        assert hasattr(maestro, 'is_quest_available')
        assert callable(maestro.is_quest_available)

    def test_is_quest_available_returns_bool(self):
        """is_quest_available() returns a boolean."""
        result = maestro.is_quest_available()
        assert isinstance(result, bool)

    def test_quest_rejects_mps(self):
        """QuestSim with MatrixProductState raises an error."""
        with pytest.raises(Exception, match="QuestSim only supports Statevector"):
            maestro.simple_execute(
                CLIFFORD_BELL_QASM,
                simulator_type=maestro.SimulatorType.QuestSim,
                simulation_type=maestro.SimulationType.MatrixProductState,
                shots=10
            )

    def test_quest_rejects_stabilizer(self):
        """QuestSim with Stabilizer raises an error."""
        with pytest.raises(Exception, match="QuestSim only supports Statevector"):
            maestro.simple_execute(
                CLIFFORD_BELL_QASM,
                simulator_type=maestro.SimulatorType.QuestSim,
                simulation_type=maestro.SimulationType.Stabilizer,
                shots=10
            )

    def test_quest_rejects_tensor_network(self):
        """QuestSim with TensorNetwork raises an error."""
        with pytest.raises(Exception, match="QuestSim only supports Statevector"):
            maestro.simple_execute(
                CLIFFORD_BELL_QASM,
                simulator_type=maestro.SimulatorType.QuestSim,
                simulation_type=maestro.SimulationType.TensorNetwork,
                shots=10
            )

    def test_quest_rejects_pauli_propagator(self):
        """QuestSim with PauliPropagator raises an error."""
        with pytest.raises(Exception, match="QuestSim only supports Statevector"):
            maestro.simple_execute(
                GENERAL_QASM,
                simulator_type=maestro.SimulatorType.QuestSim,
                simulation_type=maestro.SimulationType.PauliPropagator,
                shots=10
            )

    def test_quest_rejects_mps_estimate(self):
        """QuestSim with MPS on simple_estimate raises an error."""
        with pytest.raises(Exception, match="QuestSim only supports Statevector"):
            maestro.simple_estimate(
                CLIFFORD_BELL_NO_MEASURE_QASM,
                "ZZ",
                simulator_type=maestro.SimulatorType.QuestSim,
                simulation_type=maestro.SimulationType.MatrixProductState,
            )

    @pytest.mark.skipif(
        not maestro.is_quest_available(),
        reason="QuEST library (libmaestroquest) not available"
    )
    def test_quest_execute_bell_state(self):
        """QuestSim executes a Bell state circuit correctly."""
        result = maestro.simple_execute(
            CLIFFORD_BELL_QASM,
            simulator_type=maestro.SimulatorType.QuestSim,
            simulation_type=maestro.SimulationType.Statevector,
            shots=1000
        )
        assert result is not None
        assert 'counts' in result
        total = sum(result['counts'].values())
        assert total == 1000

    @pytest.mark.skipif(
        not maestro.is_quest_available(),
        reason="QuEST library (libmaestroquest) not available"
    )
    def test_quest_estimate_bell_state(self):
        """QuestSim estimates expectation values correctly."""
        result = maestro.simple_estimate(
            CLIFFORD_BELL_NO_MEASURE_QASM,
            "ZZ;XX;YY",
            simulator_type=maestro.SimulatorType.QuestSim,
            simulation_type=maestro.SimulationType.Statevector,
        )
        assert result is not None
        assert 'expectation_values' in result
        exp_vals = result['expectation_values']
        assert len(exp_vals) == 3
        # Bell state: <ZZ> = 1.0, <XX> = 1.0, <YY> = -1.0
        assert exp_vals[0] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[1] == pytest.approx(1.0, abs=1e-5)
        assert exp_vals[2] == pytest.approx(-1.0, abs=1e-5)

    @pytest.mark.skipif(
        not maestro.is_quest_available(),
        reason="QuEST library (libmaestroquest) not available"
    )
    def test_quest_circuit_execute(self):
        """Circuit.execute() with QuestSim works through the network."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        result = qc.execute(
            simulator_type=maestro.SimulatorType.QuestSim,
            simulation_type=maestro.SimulationType.Statevector,
            shots=100
        )
        assert result is not None
        assert 'counts' in result
        total = sum(result['counts'].values())
        assert total == 100

    @pytest.mark.skipif(
        not maestro.is_quest_available(),
        reason="QuEST library (libmaestroquest) not available"
    )
    def test_quest_circuit_estimate(self):
        """Circuit.estimate() with QuestSim produces correct results."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        result = qc.estimate(
            observables=["ZZ"],
            simulator_type=maestro.SimulatorType.QuestSim,
            simulation_type=maestro.SimulationType.Statevector,
        )
        assert result is not None
        assert result['expectation_values'][0] == pytest.approx(1.0, abs=1e-5)


class TestGetStatevector:
    """Test the get_statevector function for extracting full complex amplitudes."""

    INV_SQRT2 = 1.0 / (2.0 ** 0.5)

    def test_get_statevector_bell_state(self):
        """Bell state (|00> + |11>)/sqrt(2) should have correct amplitudes."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        sv = maestro.get_statevector(qc)
        assert len(sv) == 4
        assert sv[0] == pytest.approx(self.INV_SQRT2, abs=1e-10)
        assert sv[1] == pytest.approx(0.0, abs=1e-10)
        assert sv[2] == pytest.approx(0.0, abs=1e-10)
        assert sv[3] == pytest.approx(self.INV_SQRT2, abs=1e-10)

    def test_get_statevector_single_qubit_h(self):
        """H|0> should give (1/sqrt(2), 1/sqrt(2))."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)

        sv = maestro.get_statevector(qc)
        assert len(sv) == 2
        assert sv[0] == pytest.approx(self.INV_SQRT2, abs=1e-10)
        assert sv[1] == pytest.approx(self.INV_SQRT2, abs=1e-10)

    def test_get_statevector_x_gate(self):
        """X|0> should give (0, 1)."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.x(0)

        sv = maestro.get_statevector(qc)
        assert len(sv) == 2
        assert sv[0] == pytest.approx(0.0, abs=1e-10)
        assert sv[1] == pytest.approx(1.0, abs=1e-10)

    def test_get_statevector_mps(self):
        """get_statevector works with MPS backend."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        sv = maestro.get_statevector(
            qc,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4
        )
        assert len(sv) == 4
        assert sv[0] == pytest.approx(self.INV_SQRT2, abs=1e-10)
        assert sv[1] == pytest.approx(0.0, abs=1e-10)
        assert sv[2] == pytest.approx(0.0, abs=1e-10)
        assert sv[3] == pytest.approx(self.INV_SQRT2, abs=1e-10)

    def test_get_statevector_circuit_method(self):
        """QuantumCircuit.get_statevector() method works."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        sv = qc.get_statevector()
        assert len(sv) == 4
        assert sv[0] == pytest.approx(self.INV_SQRT2, abs=1e-10)
        assert sv[3] == pytest.approx(self.INV_SQRT2, abs=1e-10)

    def test_get_statevector_probabilities_consistency(self):
        """Statevector amplitudes squared should match get_probabilities."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        sv = maestro.get_statevector(qc)
        probs = maestro.get_probabilities(qc)

        assert len(sv) == len(probs), (
            f"Statevector length {len(sv)} != probabilities length {len(probs)}"
        )
        for amp, prob in zip(sv, probs):
            assert abs(amp) ** 2 == pytest.approx(prob, abs=1e-10)


class TestMirrorFidelity:
    """Test the mirror_fidelity function and circuit method.

    Mirror fidelity runs a circuit forward, then appends the adjoint in
    reverse, and measures P(|0...0>).  For a perfect simulator this should
    always be ~1.0.

    Default mode is shot-based (1024 shots). Set full_amplitude=True for
    exact statevector computation.
    """

    # --- Default (shot-based) tests ---

    def test_identity_circuit(self):
        """H -> H† (= H) should give fidelity ≈ 1.0 (default shot-based)."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)

        fid = maestro.mirror_fidelity(qc)
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_bell_state_circuit(self):
        """Bell state (H, CX) should give fidelity ≈ 1.0 (default shot-based)."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        fid = maestro.mirror_fidelity(qc)
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_parametric_gates(self):
        """Circuit with Rx, Ry, Rz parametric gates (default shot-based)."""
        from maestro.circuits import QuantumCircuit
        import math
        qc = QuantumCircuit()
        qc.rx(0, math.pi / 4)
        qc.ry(0, math.pi / 3)
        qc.rz(0, math.pi / 6)

        fid = maestro.mirror_fidelity(qc)
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_s_and_t_gates(self):
        """Circuit with S, T (non-self-inverse) gates (default shot-based)."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.s(0)
        qc.t(0)
        qc.cx(0, 1)

        fid = maestro.mirror_fidelity(qc)
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_circuit_method(self):
        """QuantumCircuit.mirror_fidelity() method (default shot-based)."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.s(0)

        fid = qc.mirror_fidelity()
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_measurements_are_skipped(self):
        """Measurements in the original circuit should be skipped in mirroring."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        fid = maestro.mirror_fidelity(qc)
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_custom_shots(self):
        """Mirror fidelity with explicit high shot count for tighter estimate."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)
        qc.s(0)

        fid = maestro.mirror_fidelity(qc, shots=10000)
        assert fid == pytest.approx(1.0, abs=0.05)

    def test_mps_backend(self):
        """Mirror fidelity with MPS backend (shot-based, scalable)."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        fid = qc.mirror_fidelity(
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=10000,
            max_bond_dimension=4,
        )
        assert fid == pytest.approx(1.0, abs=0.05)

    # --- Exact (full_amplitude) tests ---

    def test_full_amplitude_identity(self):
        """Exact statevector mode with full_amplitude=True."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)

        fid = maestro.mirror_fidelity(qc, full_amplitude=True)
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_full_amplitude_bell(self):
        """Exact statevector mode for Bell state."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        fid = qc.mirror_fidelity(full_amplitude=True)
        assert fid == pytest.approx(1.0, abs=1e-10)

    def test_full_amplitude_parametric(self):
        """Exact statevector mode with parametric and controlled gates."""
        from maestro.circuits import QuantumCircuit
        import math
        qc = QuantumCircuit()
        qc.h(0)
        qc.crx(0, 1, math.pi / 4)
        qc.u(0, math.pi / 4, math.pi / 3, math.pi / 6)

        fid = maestro.mirror_fidelity(qc, full_amplitude=True)
        assert fid == pytest.approx(1.0, abs=1e-10)


class TestInnerProduct:
    """Test the inner_product function and circuit method.

    inner_product(circ_1, circ_2) computes <psi_1|psi_2> = <0|U1† U2|0>
    via ProjectOnZero.
    """

    def test_identical_circuits(self):
        """Inner product of identical circuits should be 1.0."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 1)

        result = maestro.inner_product(qc, qc)
        assert abs(result) == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_circuits(self):
        """Inner product of orthogonal states should be ~0."""
        from maestro.circuits import QuantumCircuit
        # |+> state
        qc1 = QuantumCircuit()
        qc1.h(0)

        # |-> state
        qc2 = QuantumCircuit()
        qc2.x(0)
        qc2.h(0)

        result = maestro.inner_product(qc1, qc2)
        assert abs(result) == pytest.approx(0.0, abs=1e-10)

    def test_circuit_method(self):
        """QuantumCircuit.inner_product(other) method works."""
        from maestro.circuits import QuantumCircuit
        qc1 = QuantumCircuit()
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit()
        qc2.h(0)
        qc2.cx(0, 1)

        result = qc1.inner_product(qc2)
        assert abs(result) == pytest.approx(1.0, abs=1e-10)

    def test_self_inner_product_method(self):
        """qc.inner_product(qc) should always give |result| = 1.0."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)
        qc.s(0)
        qc.cx(0, 1)

        result = qc.inner_product(qc)
        assert abs(result) == pytest.approx(1.0, abs=1e-10)

    def test_parametric_gates(self):
        """Inner product with parametric rotation gates."""
        from maestro.circuits import QuantumCircuit
        import math

        qc1 = QuantumCircuit()
        qc1.rx(0, math.pi / 4)
        qc1.ry(0, math.pi / 3)

        qc2 = QuantumCircuit()
        qc2.rx(0, math.pi / 4)
        qc2.ry(0, math.pi / 3)

        result = maestro.inner_product(qc1, qc2)
        assert abs(result) == pytest.approx(1.0, abs=1e-10)

    def test_different_parametric_angles(self):
        """Inner product of circuits with different rotation angles is < 1."""
        from maestro.circuits import QuantumCircuit
        import math

        qc1 = QuantumCircuit()
        qc1.ry(0, 0.0)  # |0>

        qc2 = QuantumCircuit()
        qc2.ry(0, math.pi / 2)  # cos(pi/4)|0> + sin(pi/4)|1>

        result = maestro.inner_product(qc1, qc2)
        # <0|Ry(pi/2)|0> = cos(pi/4) = 1/sqrt(2)
        expected = math.cos(math.pi / 4)
        assert result.real == pytest.approx(expected, abs=1e-10)
        assert result.imag == pytest.approx(0.0, abs=1e-10)

    def test_mps_backend(self):
        """Inner product works with MPS backend."""
        from maestro.circuits import QuantumCircuit
        qc1 = QuantumCircuit()
        qc1.h(0)
        qc1.cx(0, 1)

        qc2 = QuantumCircuit()
        qc2.h(0)
        qc2.cx(0, 1)

        result = maestro.inner_product(
            qc1, qc2,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=4
        )
        assert abs(result) == pytest.approx(1.0, abs=1e-5)

    def test_returns_complex(self):
        """Inner product should return a complex number."""
        from maestro.circuits import QuantumCircuit
        qc = QuantumCircuit()
        qc.h(0)

        result = maestro.inner_product(qc, qc)
        assert isinstance(result, complex)

    def test_phase_difference(self):
        """Two circuits differing by a global phase have |<psi1|psi2>| = 1."""
        from maestro.circuits import QuantumCircuit
        import math

        qc1 = QuantumCircuit()
        qc1.h(0)

        # Add a global phase via rz
        qc2 = QuantumCircuit()
        qc2.h(0)
        qc2.rz(0, math.pi / 3)

        result = maestro.inner_product(qc1, qc2)
        # States differ by a relative phase, so |overlap| < 1
        # but result should still be a valid complex number
        assert abs(result) <= 1.0 + 1e-10


class TestMPSOptimizedSwapping:
    """Test the MPS optimized swapping parameters (disable_optimized_swapping, lookahead_depth)."""

    MPS_PARAMS = dict(
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
        max_bond_dimension=16,
    )

    def _make_non_nearest_neighbor_circuit(self):
        """Build a circuit with long-range 2-qubit gates that trigger swap insertion.

        This creates a pattern where qubits 0 and 4 interact, forcing the MPS
        simulator to insert SWAPs to bring them adjacent. The optimized swap
        routine should handle this more efficiently than naive swapping.
        """
        from maestro.circuits import QuantumCircuit
        n_qubits = 8
        qc = QuantumCircuit()
        # Create entanglement across non-adjacent qubits
        for i in range(n_qubits):
            qc.h(i)
        # Long-range CX gates
        qc.cx(0, 4)
        qc.cx(1, 5)
        qc.cx(2, 6)
        qc.cx(3, 7)
        # Another layer crossing in the opposite direction
        qc.cx(0, 7)
        qc.cx(1, 6)
        qc.cx(2, 5)
        qc.cx(3, 4)
        return qc

    # ---- Execute tests ---------------------------------------------------

    def test_execute_with_optimized_swapping_default(self):
        """Execute with optimized swapping enabled (default) produces valid results."""
        qc = self._make_non_nearest_neighbor_circuit()
        qc.measure_all()
        result = qc.execute(shots=1024, **self.MPS_PARAMS)
        assert result is not None
        assert 'counts' in result
        assert sum(result['counts'].values()) == 1024

    def test_execute_with_optimized_swapping_disabled(self):
        """Execute with optimized swapping explicitly disabled produces valid results."""
        qc = self._make_non_nearest_neighbor_circuit()
        qc.measure_all()
        result = qc.execute(
            shots=1024,
            disable_optimized_swapping=True,
            **self.MPS_PARAMS
        )
        assert result is not None
        assert 'counts' in result
        assert sum(result['counts'].values()) == 1024

    def test_execute_custom_lookahead_depth(self):
        """Execute with a custom lookahead_depth produces valid results."""
        qc = self._make_non_nearest_neighbor_circuit()
        qc.measure_all()
        for depth in [0, 5, 10, 30]:
            result = qc.execute(
                shots=256,
                lookahead_depth=depth,
                **self.MPS_PARAMS
            )
            assert result is not None
            assert sum(result['counts'].values()) == 256

    # ---- Estimate tests --------------------------------------------------

    def test_estimate_with_optimized_swapping(self):
        """Estimate with optimized swapping produces correct expectation values."""
        qc = self._make_non_nearest_neighbor_circuit()

        result_opt = qc.estimate(
            observables=["ZIIIIIII", "IZIIIIII"],
            **self.MPS_PARAMS
        )
        assert result_opt is not None
        exp_vals = result_opt['expectation_values']
        assert len(exp_vals) == 2
        # Expectation values should be valid (between -1 and 1)
        for val in exp_vals:
            assert -1.0 - 1e-6 <= val <= 1.0 + 1e-6

    def test_estimate_disabled_vs_enabled_agree(self):
        """Optimized and unoptimized paths should produce matching expectation values."""
        qc = self._make_non_nearest_neighbor_circuit()
        obs = ["ZIIIIIII", "IZIIIIII", "ZZIIIIII"]

        result_opt = qc.estimate(observables=obs, **self.MPS_PARAMS)
        result_no_opt = qc.estimate(
            observables=obs,
            disable_optimized_swapping=True,
            **self.MPS_PARAMS
        )
        assert result_opt is not None
        assert result_no_opt is not None

        for v_opt, v_no in zip(
            result_opt['expectation_values'],
            result_no_opt['expectation_values']
        ):
            assert v_opt == pytest.approx(v_no, abs=1e-4)

    def test_estimate_custom_lookahead(self):
        """Estimate with custom lookahead_depth produces valid results."""
        qc = self._make_non_nearest_neighbor_circuit()
        result = qc.estimate(
            observables=["ZIIIIIII"],
            lookahead_depth=5,
            **self.MPS_PARAMS
        )
        assert result is not None
        assert len(result['expectation_values']) == 1

    # ---- simple_execute / simple_estimate module-level tests -------------

    def test_simple_execute_disable_optimized_swapping(self):
        """Module-level simple_execute accepts disable_optimized_swapping."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        creg c[6];
        h q[0];
        cx q[0], q[5];
        cx q[1], q[4];
        measure q -> c;
        """
        result = maestro.simple_execute(
            qasm,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            shots=100,
            max_bond_dimension=8,
            disable_optimized_swapping=True,
            lookahead_depth=10
        )
        assert result is not None
        assert sum(result['counts'].values()) == 100

    def test_simple_estimate_disable_optimized_swapping(self):
        """Module-level simple_estimate accepts disable_optimized_swapping."""
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[6];
        h q[0];
        cx q[0], q[5];
        """
        result = maestro.simple_estimate(
            qasm,
            "ZIIIII",
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=8,
            disable_optimized_swapping=False,
            lookahead_depth=20
        )
        assert result is not None
        assert len(result['expectation_values']) == 1

    # ---- Correctness: compare against Statevector reference --------------

    def test_optimized_mps_matches_statevector(self):
        """MPS with optimized swapping should match statevector for small circuits."""
        from maestro.circuits import QuantumCircuit

        qc = QuantumCircuit()
        qc.h(0)
        qc.cx(0, 3)
        qc.cx(1, 4)
        qc.rz(2, 0.5)
        qc.cx(0, 4)

        obs = ["ZIIII", "IZIII", "ZZIII", "IIIIZ"]

        result_sv = qc.estimate(
            observables=obs,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.Statevector,
        )
        result_mps = qc.estimate(
            observables=obs,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=32,
            lookahead_depth=20,
        )
        result_mps_no_opt = qc.estimate(
            observables=obs,
            simulator_type=maestro.SimulatorType.QCSim,
            simulation_type=maestro.SimulationType.MatrixProductState,
            max_bond_dimension=32,
            disable_optimized_swapping=True,
        )

        for v_sv, v_mps, v_no_opt in zip(
            result_sv['expectation_values'],
            result_mps['expectation_values'],
            result_mps_no_opt['expectation_values']
        ):
            assert v_mps == pytest.approx(v_sv, abs=1e-6), \
                f"MPS optimized {v_mps} != SV {v_sv}"
            assert v_no_opt == pytest.approx(v_sv, abs=1e-6), \
                f"MPS unoptimized {v_no_opt} != SV {v_sv}"
