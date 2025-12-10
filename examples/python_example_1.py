import maestro_py

def main():
    print("Maestro Python Bindings Example")

    # Create the Maestro instance
    m = maestro_py.Maestro()

    # Create a simulator
    sim_handler = m.CreateSimulator(maestro_py.SimulatorType.QCSim, maestro_py.SimulationType.Statevector)
    simulator = m.GetSimulator(sim_handler)

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
