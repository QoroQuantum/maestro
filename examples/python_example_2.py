#!/usr/bin/env python3
"""Test script for the new simple_execute function"""

import maestro

# --- QASM Circuits ---

QASM_BELL = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""

QASM_GHZ = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
"""

# --- Helper Functions ---


def print_result(test_name: str, result: dict) -> None:
    """Pretty prints the execution result."""
    print(f"{test_name}")
    if result:
        print(f"  Counts: {result.get('counts')}")
        if "simulator" in result:
            print(f"  Simulator: {result['simulator']}")
        if "method" in result:
            print(f"  Method: {result['method']}")
        if "time_taken" in result:
            print(f"  Time: {result['time_taken']:.6f}s")

        # Check total shots if we can infer it
        counts = result.get("counts", {})
        if counts:
            print(f"  Total shots: {sum(counts.values())}")
    else:
        print("  ERROR: Execution failed")
    print()


def main():
    print("Running Maestro...\n")

    # Test 1: Simple Bell state with defaults (QCSim, Statevector)
    result1 = maestro.simple_execute(QASM_BELL)
    print_result("Test 1: Bell state with defaults (QCSim, Statevector)", result1)

    # Test 2: With custom shots
    result2 = maestro.simple_execute(QASM_BELL, shots=2000)
    print_result("Test 2: Bell state with 2000 shots", result2)

    # Test 3: Using MatrixProductState simulation type
    result3 = maestro.simple_execute(
        QASM_BELL,
        simulator_type=maestro.SimulatorType.QCSim,
        simulation_type=maestro.SimulationType.MatrixProductState,
    )
    print_result("Test 3: Bell state with Matrix Product State", result3)

    # Test 4: GHZ state
    result4 = maestro.simple_execute(QASM_GHZ, shots=500)
    print_result("Test 4: 3-qubit GHZ state", result4)

    print("All tests completed!")


if __name__ == "__main__":
    main()
