#!/usr/bin/env python3
"""Test script for the new simple_execute function"""

import sys
sys.path.insert(0, '/Users/stephendiadamo/QoroSoftware/maestro/build')

import maestro_py

# Test 1: Simple Bell state with defaults (QCSim, Statevector)
print("Test 1: Bell state with defaults (QCSim, Statevector)")
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

result = maestro_py.simple_execute(qasm_bell)
if result:
    print(f"  Counts: {result['counts']}")
    print(f"  Simulator: {result['simulator']}")
    print(f"  Method: {result['method']}")
    print(f"  Time: {result['time_taken']:.6f}s")
    print()
else:
    print("  ERROR: Execution failed")
    print()

# Test 2: With custom shots
print("Test 2: Bell state with 2000 shots")
result = maestro_py.simple_execute(qasm_bell, shots=2000)
if result:
    print( f"  Counts: {result['counts']}")
    print(f"  Total shots: {sum(result['counts'].values())}")
    print()
else:
    print("  ERROR: Execution failed")
    print()

# Test 3: Using MatrixProductState simulation type
print("Test 3: Bell state with Matrix Product State")
result = maestro_py.simple_execute(
    qasm_bell,
    simulator_type=maestro_py.SimulatorType.QCSim,
    simulation_type=maestro_py.SimulationType.MatrixProductState
)
if result:
    print(f"  Counts: {result['counts']}")
    print(f"  Method: {result['method']}")
    print()
else:
    print("  ERROR: Execution failed")
    print()

# Test 4: GHZ state
print("Test 4: 3-qubit GHZ state")
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

result = maestro_py.simple_execute(qasm_ghz, shots=500)
if result:
    print(f"  Counts: {result['counts']}")
    print(f"  Simulator: {result['simulator']}")
    print()
else:
    print("  ERROR: Execution failed")
    print()

print("All tests completed!")
