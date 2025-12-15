import maestro


def main():
    print("Maestro Python Bindings Example")

    # Create the Maestro instance
    m = maestro.Maestro()

    # Create a simulator
    sim_handler = m.CreateSimulator(
        maestro.SimulatorType.QCSim, maestro.SimulationType.Statevector
    )
    simulator = m.GetSimulator(sim_handler)

    circ = (
        "OPENQASM 2.0;\n"
        + 'include "qelib1.inc";\n'
        + "qreg q[2];\n"
        + "creg c[2];\n"
        + "h q[0];\n"
        + "cx q[0], q[1];\n"
        + "measure q -> c;\n"
    )

    circuit_parser = maestro.QasmToCirc()
    circuit = circuit_parser.ParseAndTranslate(circ)
    if simulator:
        print("Simulator object obtained successfully")

        # Allocate qubits
        num_qubits = circuit.GetMaxQubitIndex() + 1
        simulator.AllocateQubits(num_qubits)
        simulator.Initialize()
        print(f"Allocated {num_qubits} qubits")

        # Apply gates (Bell State)
        simulator.ApplyH(0)
        simulator.ApplyCX(0, 1)
        print("Applied H(0) and CX(0, 1)")

        # Measure
        results = simulator.SampleCounts([0, 1], 1000)
        print(f"Measurement results: {results}")

    else:
        print("Failed to get simulator object")

    # Clean up
    m.DestroySimulator(sim_handler)
    print("Simulator destroyed")


if __name__ == "__main__":
    main()
